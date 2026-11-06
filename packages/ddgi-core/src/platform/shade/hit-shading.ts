/**
 * The platform's hit shading (contract/shade.ts): a WGSL module that
 * techniques include in their tracing kernels, plus the sky it reads.
 *
 * Per frame (`encode`), it writes its live settings when they changed and
 * re-bakes the sky when the environment changed. The module's resources
 * never change after construction (the sky's textures and the uniform buffer
 * are allocated once), so its `version` stays 0 and kernels never rebuild
 * their bind group for it. The same holds for the sky-change module
 * (`sky-change.ts`).
 */

import type { EnvironmentDescriptor } from '../contract/scene.ts';
import type { HitShading, HitShadingSettings } from '../contract/shade.ts';
import type { GpuTimerScope } from '../contract/timing.ts';
import type { BindingSlot, KernelBuilder, WgslModule } from '../contract/wgsl.ts';
import { PlatformSky, type PlatformSkyOptions } from '../sky/platform-sky.ts';
import { PlatformDeferredDirect } from './direct.ts';
import { LIGHT_FALLOFF_MODULE } from './light-falloff.ts';
import { SKY_CHANGE_MODULE_ENTRIES, SKY_RADIANCE_SH_BYTES, skyChangeModuleWgsl } from './sky-change.ts';
import { SHADE_MODULE_ENTRIES, SHADE_UNIFORM_LAYOUT, shadeModuleWgsl } from './wgsl.ts';

/** Defaults of the live settings. */
export const DEFAULT_HIT_SHADING_SETTINGS: Readonly<HitShadingSettings> = {
    maxShadowedLightsPerHit: 4,
    bounceAlbedoCap: 1,
};

/** `GPUBufferUsage` bits; the globals don't exist in Node, where tests import this file. */
const BUFFER_USAGE = { copySrc: 0x4, copyDst: 0x8, uniform: 0x40, storage: 0x80 } as const;

export interface PlatformHitShadingOptions {
    readonly label?: string;
    readonly warn?: (message: string) => void;
    /** The sky's SH reduction (see {@link PlatformSkyOptions.reduction}); for tests. */
    readonly skyReduction?: PlatformSkyOptions['reduction'];
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

    private readonly device: GPUDevice;
    /** `GiShadeUniforms`: the sky's harmonics (written by the bake), then the settings. */
    private readonly uniforms: GPUBuffer;
    /** The sky's radiance SH when the techniques last saw an environment change (sky-change.ts). */
    private readonly skyRadianceBefore: GPUBuffer;
    private readonly radianceMapView: GPUTextureView;
    private readonly skySampler: GPUSampler;
    private uploadedSettings = '';
    private disposed = false;

    constructor(device: GPUDevice, kernels: KernelBuilder, sceneModule: WgslModule, raysModule: WgslModule, options: PlatformHitShadingOptions = {}) {
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

        const resources: readonly GPUBindingResource[] = [{ buffer: this.uniforms }, this.radianceMapView, this.skySampler];
        this.module = {
            name: 'gi-shade',
            dependencies: [sceneModule, raysModule, LIGHT_FALLOFF_MODULE],
            code: (slot: BindingSlot) => shadeModuleWgsl(slot),
            entries: SHADE_MODULE_ENTRIES,
            resources: () => resources,
            version: 0,
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

    /** GPU bytes held: the uniform buffers, the sky and gi/direct's dispatch arguments (techniques hold their ray scratch). */
    get memoryBytes(): number {
        return SHADE_UNIFORM_LAYOUT.byteLength + SKY_RADIANCE_SH_BYTES + this.sky.memoryBytes + this.direct.memoryBytes;
    }

    /** Compiles the sky bake. Called by the platform's `prepare`. */
    prepare(): Promise<void> {
        return this.sky.prepare();
    }

    /**
     * Uploads changed settings and re-bakes the environment when
     * `environmentVersion` changed (every `setEnvironment` call bumps it) and
     * the change needs a bake (`PlatformSky`), into the host's encoder.
     * Called by the platform's `encode` after the scene copy's uploads and
     * before any technique.
     */
    encode(encoder: GPUCommandEncoder, environment: EnvironmentDescriptor, environmentVersion: number, timer?: GpuTimerScope): void {
        if (this.disposed) throw new Error('@pmndrs/ddgi: hit shading was disposed');
        this.uploadSettings();
        this.sky.encode(encoder, environment, environmentVersion, { buffer: this.uniforms, offset: SHADE_UNIFORM_LAYOUT.skyHarmonics }, timer);
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
        this.uniforms.destroy();
        this.skyRadianceBefore.destroy();
    }

    private uploadSettings(): void {
        // At least one shadowed local light per hit, or the stochastic choice would have nothing to pick.
        const maxShadowedLights = Math.max(1, Math.floor(finiteOr(this.settings.maxShadowedLightsPerHit, DEFAULT_HIT_SHADING_SETTINGS.maxShadowedLightsPerHit)));
        const albedoCap = Math.max(0, finiteOr(this.settings.bounceAlbedoCap, DEFAULT_HIT_SHADING_SETTINGS.bounceAlbedoCap));
        const key = `${maxShadowedLights}/${albedoCap}`;
        if (key === this.uploadedSettings) return;
        const data = new ArrayBuffer(8);
        new Uint32Array(data, 0, 1)[0] = maxShadowedLights;
        new Float32Array(data, 4, 1)[0] = albedoCap;
        this.device.queue.writeBuffer(this.uniforms, SHADE_UNIFORM_LAYOUT.maxShadowedLightsPerHit, data);
        this.uploadedSettings = key;
    }
}

function finiteOr(value: number, fallback: number): number {
    return Number.isFinite(value) ? value : fallback;
}
