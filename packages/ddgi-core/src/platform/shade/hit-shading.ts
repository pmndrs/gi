/**
 * The platform's hit shading (contract/shade.ts): a WGSL module that
 * techniques include in their tracing kernels, plus the sky it reads and the
 * light grid it draws many lights from.
 *
 * Per frame (`encode`), it writes its live settings when they changed,
 * re-bakes the sky when the environment changed, and rebuilds the light grid
 * when the scene has more point and spot lights than `exactLightLimit`. The
 * module's resources are allocated once, except the light grid's picks
 * texture, which is made when a scene first needs the grid and grows with
 * its cells, and the texture store's array, which grows with the scene's
 * textures (textures/texture-store.ts); the module's `version` follows both,
 * so kernels rebuild their bind group only then. The sky-change module
 * (`sky-change.ts`) never changes.
 */

import type { Aabb } from '../contract/math.ts';
import type { EnvironmentDescriptor } from '../contract/scene.ts';
import type { HitShading, HitShadingSettings } from '../contract/shade.ts';
import type { GpuTimerScope } from '../contract/timing.ts';
import type { BindingSlot, KernelBuilder, WgslModule } from '../contract/wgsl.ts';
import type { TextureStoreBinding } from '../textures/texture-store.ts';
import { PlatformSky, type PlatformSkyOptions } from '../sky/platform-sky.ts';
import { PlatformDeferredDirect } from './direct.ts';
import { LIGHT_FALLOFF_MODULE } from './light-falloff.ts';
import { INACTIVE_LIGHT_GRID, LIGHT_GRID_MAX_CELLS, PlatformLightGrid, type LightGridLookup, type LightGridReduction } from './light-grid.ts';
import { SKY_CHANGE_MODULE_ENTRIES, SKY_RADIANCE_SH_BYTES, skyChangeModuleWgsl } from './sky-change.ts';
import { SHADE_MODULE_ENTRIES, SHADE_UNIFORM_LAYOUT, shadeModuleWgsl } from './wgsl.ts';

/** Defaults of the live settings. */
export const DEFAULT_HIT_SHADING_SETTINGS: Readonly<HitShadingSettings> = {
    maxShadowedLightsPerHit: 4,
    bounceAlbedoCap: 1,
    exactLightLimit: 8,
    lightGridMaxCells: 2048,
    lightGridShadowedLightsPerHit: 1,
    sampleTextures: true,
};

/** What hit shading needs to know of the scene's lights each frame, for the light grid. */
export interface HitShadingLights {
    /** The platform's frame number (the grid's picks are new every frame). */
    readonly frame: number;
    /** Point and spot lights in the scene. */
    readonly localLights: number;
    /** The scene's bounds: where the grid's cells go. */
    readonly bounds: Aabb;
}

/** `GPUBufferUsage` bits; the globals don't exist in Node, where tests import this file. */
const BUFFER_USAGE = { copySrc: 0x4, copyDst: 0x8, uniform: 0x40, storage: 0x80 } as const;

/** Bytes of the light grid's lookup in the uniforms (`SHADE_UNIFORM_LAYOUT.lightGrid`). */
const LIGHT_GRID_LOOKUP_BYTES = 40;

export interface PlatformHitShadingOptions {
    readonly label?: string;
    readonly warn?: (message: string) => void;
    /** The sky's SH reduction (see {@link PlatformSkyOptions.reduction}); for tests. */
    readonly skyReduction?: PlatformSkyOptions['reduction'];
    /** The light grid's scan: `subgroups` when the device has them (the default), or the declared fallback. */
    readonly lightGridReduction?: LightGridReduction;
}

export class PlatformHitShading implements HitShading {
    readonly module: WgslModule;
    readonly lightFalloff: WgslModule = LIGHT_FALLOFF_MODULE;
    readonly skyChange: WgslModule;
    /** Live: change the fields at any time; the next `encode` uploads them. */
    readonly settings: HitShadingSettings = { ...DEFAULT_HIT_SHADING_SETTINGS };
    readonly sky: PlatformSky;
    /** Deferred direct light (`gi/direct`, direct.ts); its kernels compile on a technique's first `prepare`. */
    readonly direct: PlatformDeferredDirect;
    /** Many-light sampling (`gi/light-grid`, light-grid.ts), built in frames with more than `exactLightLimit` point and spot lights. */
    readonly lightGrid: PlatformLightGrid;

    private readonly device: GPUDevice;
    /** `GiShadeUniforms`: the sky's harmonics (written by the bake), then the settings and the light grid's lookup. */
    private readonly uniforms: GPUBuffer;
    /** The sky's radiance SH when the techniques last saw an environment change (sky-change.ts). */
    private readonly skyRadianceBefore: GPUBuffer;
    private readonly radianceMapView: GPUTextureView;
    private readonly skySampler: GPUSampler;
    private readonly textureSampler: GPUSampler;
    private uploadedSettings = '';
    private uploadedLightGrid = '';
    private readonly lightGridWords = new ArrayBuffer(LIGHT_GRID_LOOKUP_BYTES);
    private disposed = false;

    /**
     * `textures` is the texture store hit shading samples at hits (the
     * platform's, which the scene copy fills): its array is bound, and
     * rebinding follows its `version`.
     */
    constructor(
        device: GPUDevice,
        kernels: KernelBuilder,
        sceneModule: WgslModule,
        raysModule: WgslModule,
        textures: TextureStoreBinding,
        options: PlatformHitShadingOptions = {},
    ) {
        const label = options.label ?? 'gi';
        this.device = device;
        this.sky = new PlatformSky(device, { label, warn: options.warn, reduction: options.skyReduction });
        // STORAGE because the sky bake writes the harmonics' range of it; tracing kernels bind it as a
        // uniform. COPY_SRC for the sky-change module's copy of the radiance SH.
        this.uniforms = device.createBuffer({
            label: `${label} hit shading uniforms`,
            size: SHADE_UNIFORM_LAYOUT.byteLength,
            usage: BUFFER_USAGE.uniform | BUFFER_USAGE.storage | BUFFER_USAGE.copyDst | BUFFER_USAGE.copySrc,
        });
        this.skyRadianceBefore = device.createBuffer({
            label: `${label} sky radiance SH last seen`,
            size: SKY_RADIANCE_SH_BYTES,
            usage: BUFFER_USAGE.uniform | BUFFER_USAGE.copyDst,
        });
        this.radianceMapView = this.sky.radianceMap.createView({ label: `${label} sky radiance map` });
        this.skySampler = device.createSampler({
            label: `${label} sky radiance map`,
            magFilter: 'linear',
            minFilter: 'linear',
            addressModeU: 'repeat',
            addressModeV: 'clamp-to-edge',
        });
        // Trilinear and repeating: the mip comes from each hit's ray cone, and clamp and mirror
        // are worked out on the uv in the shader (giShadeWrap), so one sampler serves every mapping.
        this.textureSampler = device.createSampler({
            label: `${label} texture store`,
            magFilter: 'linear',
            minFilter: 'linear',
            mipmapFilter: 'linear',
            addressModeU: 'repeat',
            addressModeV: 'repeat',
        });

        this.lightGrid = new PlatformLightGrid(device, kernels, sceneModule, LIGHT_FALLOFF_MODULE, { label, reduction: options.lightGridReduction });
        const lightGrid = this.lightGrid;
        const fixedResources: readonly GPUBindingResource[] = [{ buffer: this.uniforms }, this.radianceMapView, this.skySampler];
        const textureSampler = this.textureSampler;
        this.module = {
            name: 'gi-shade',
            dependencies: [sceneModule, raysModule, LIGHT_FALLOFF_MODULE],
            code: (slot: BindingSlot) => shadeModuleWgsl(slot),
            entries: SHADE_MODULE_ENTRIES,
            resources: () => [...fixedResources, lightGrid.view, textures.view, textureSampler],
            // Only the light grid's texture (when a scene first needs it, or its cells grow) and the
            // texture store's array (when textures need more layers) are ever reallocated. Both
            // versions only grow, so their sum changes whenever either does.
            get version() {
                return lightGrid.version + textures.version;
            },
            storageBufferCount: 0,
        };
        this.direct = new PlatformDeferredDirect(device, kernels, this.module, label);

        const skyChangeResources: readonly GPUBindingResource[] = [
            { buffer: this.uniforms, offset: SHADE_UNIFORM_LAYOUT.skyRadiance, size: SKY_RADIANCE_SH_BYTES },
            { buffer: this.skyRadianceBefore },
        ];
        this.skyChange = {
            name: 'gi-sky-change',
            dependencies: [],
            code: (slot: BindingSlot) => skyChangeModuleWgsl(slot),
            entries: SKY_CHANGE_MODULE_ENTRIES,
            resources: () => skyChangeResources,
            version: 0,
            storageBufferCount: 0,
        };
    }

    /** GPU bytes held: the uniform buffers, the sky, gi/direct's dispatch arguments and the light grid (techniques hold their ray scratch). */
    get memoryBytes(): number {
        return SHADE_UNIFORM_LAYOUT.byteLength + SKY_RADIANCE_SH_BYTES + this.sky.memoryBytes + this.direct.memoryBytes + this.lightGrid.memoryBytes;
    }

    /** Compiles the sky bake and the light grid's build. Called by the platform's `prepare`. */
    async prepare(): Promise<void> {
        await Promise.all([this.sky.prepare(), this.lightGrid.prepare()]);
    }

    /**
     * Uploads changed settings, re-bakes the environment when
     * `environmentVersion` changed (every `setEnvironment` call bumps it) and
     * the change needs a bake (`PlatformSky`), and, given the scene's
     * `lights`, rebuilds the light grid when there are more point and spot
     * lights than `exactLightLimit`, all into the host's encoder. Called by
     * the platform's `encode` after the scene copy's uploads and before any
     * technique. Without `lights` the grid is off.
     */
    encode(encoder: GPUCommandEncoder, environment: EnvironmentDescriptor, environmentVersion: number, timer?: GpuTimerScope, lights?: HitShadingLights): void {
        if (this.disposed) throw new Error('@pmndrs/ddgi: hit shading was disposed');
        this.uploadSettings();
        this.sky.encode(encoder, environment, environmentVersion, { buffer: this.uniforms, offset: SHADE_UNIFORM_LAYOUT.skyHarmonics }, timer);
        const lookup = lights
            ? this.lightGrid.encode(
                  encoder,
                  {
                      frame: lights.frame,
                      bounds: lights.bounds,
                      localLights: lights.localLights,
                      exactLightLimit: settingAtLeast(this.settings.exactLightLimit, DEFAULT_HIT_SHADING_SETTINGS.exactLightLimit, 0),
                      maxCells: Math.min(LIGHT_GRID_MAX_CELLS, settingAtLeast(this.settings.lightGridMaxCells, DEFAULT_HIT_SHADING_SETTINGS.lightGridMaxCells, 1)),
                  },
                  timer,
              )
            : INACTIVE_LIGHT_GRID;
        this.uploadLightGrid(lookup);
    }

    /**
     * Records that the techniques have now seen the sky as it is: copies the
     * radiance SH into the sky-change module's "before". Called by the
     * platform after the techniques of a frame whose changes included the
     * environment, so the copy lands after every kernel of that frame read it.
     */
    recordSkySeen(encoder: GPUCommandEncoder): void {
        encoder.copyBufferToBuffer(this.uniforms, SHADE_UNIFORM_LAYOUT.skyRadiance, this.skyRadianceBefore, 0, SKY_RADIANCE_SH_BYTES);
    }

    dispose(): void {
        if (this.disposed) return;
        this.disposed = true;
        this.sky.dispose();
        this.direct.dispose();
        this.lightGrid.dispose();
        this.uniforms.destroy();
        this.skyRadianceBefore.destroy();
    }

    private uploadSettings(): void {
        // At least one shadowed local light per hit, or the stochastic choice would have nothing to pick.
        const maxShadowedLights = Math.max(1, Math.floor(finiteOr(this.settings.maxShadowedLightsPerHit, DEFAULT_HIT_SHADING_SETTINGS.maxShadowedLightsPerHit)));
        const albedoCap = Math.max(0, finiteOr(this.settings.bounceAlbedoCap, DEFAULT_HIT_SHADING_SETTINGS.bounceAlbedoCap));
        const sampleTextures = this.settings.sampleTextures !== false;
        const key = `${maxShadowedLights}/${albedoCap}/${sampleTextures}`;
        if (key === this.uploadedSettings) return;
        const data = new ArrayBuffer(8);
        new Uint32Array(data, 0, 1)[0] = maxShadowedLights;
        new Float32Array(data, 4, 1)[0] = albedoCap;
        this.device.queue.writeBuffer(this.uniforms, SHADE_UNIFORM_LAYOUT.maxShadowedLightsPerHit, data);
        this.device.queue.writeBuffer(this.uniforms, SHADE_UNIFORM_LAYOUT.sampleTextures, new Uint32Array([sampleTextures ? 1 : 0]));
        this.uploadedSettings = key;
    }

    /**
     * Writes the light grid's lookup into the uniforms when it changed. A
     * queue write lands before the host submits this frame's encoder, so
     * every kernel of the frame reads the grid `gi/light-grid` just built.
     */
    private uploadLightGrid(lookup: LightGridLookup): void {
        const layout = lookup.layout;
        const shadowedLights = Math.max(1, Math.floor(finiteOr(this.settings.lightGridShadowedLightsPerHit, DEFAULT_HIT_SHADING_SETTINGS.lightGridShadowedLightsPerHit)));
        const key = lookup.active ? `${layout.origin.join(',')}/${layout.cellSize}/${layout.cells.join(',')}/${shadowedLights}` : 'off';
        if (key === this.uploadedLightGrid) return;
        const words = new Uint32Array(this.lightGridWords);
        const floats = new Float32Array(this.lightGridWords);
        // GiShadeUniforms from lightGridActive on: active, shadowed lights per hit, origin, 1 / cell size, cells, the outside cell.
        words[0] = lookup.active ? 1 : 0;
        words[1] = shadowedLights;
        floats.set(layout.origin, 2);
        floats[5] = 1 / layout.cellSize;
        words.set(layout.cells, 6);
        // The outside cell's index is the cell count.
        words[9] = layout.cellCount;
        this.device.queue.writeBuffer(this.uniforms, SHADE_UNIFORM_LAYOUT.lightGrid, this.lightGridWords);
        this.uploadedLightGrid = key;
    }
}

function finiteOr(value: number, fallback: number): number {
    return Number.isFinite(value) ? value : fallback;
}

/** A whole-number setting, at least `minimum`; Infinity stays Infinity (no limit). */
function settingAtLeast(value: number, fallback: number, minimum: number): number {
    if (value === Infinity) return Infinity;
    return Math.max(minimum, Math.floor(finiteOr(value, fallback)));
}
