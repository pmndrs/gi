/**
 * The platform's hit shading (contract/shade.ts): a WGSL module that
 * techniques include in their tracing kernels, plus the sky it reads.
 *
 * Per frame (`encode`), it writes its live settings when they changed and
 * re-projects the sky when the environment changed. The module's resources
 * never change after construction (the sky's textures and the uniform buffer
 * are allocated once), so its `version` stays 0 and kernels never rebuild
 * their bind group for it.
 */

import type { EnvironmentDescriptor } from '../contract/scene.ts';
import type { HitShading, HitShadingSettings } from '../contract/shade.ts';
import type { BindingSlot, WgslModule } from '../contract/wgsl.ts';
import { PlatformSky, type PlatformSkyOptions } from '../sky/platform-sky.ts';
import { SHADE_MODULE_ENTRIES, SHADE_UNIFORM_LAYOUT, shadeModuleWgsl } from './wgsl.ts';

/** Defaults of the live settings. */
export const DEFAULT_HIT_SHADING_SETTINGS: Readonly<HitShadingSettings> = {
    maxShadowedLightsPerHit: 4,
    bounceAlbedoCap: 1,
};

/** `GPUBufferUsage` bits; the globals don't exist in Node, where tests import this file. */
const BUFFER_USAGE = { copyDst: 0x8, uniform: 0x40, storage: 0x80 } as const;

export interface PlatformHitShadingOptions {
    readonly label?: string;
    readonly warn?: (message: string) => void;
    /** The sky's SH reduction (see {@link PlatformSkyOptions.reduction}); for tests. */
    readonly skyReduction?: PlatformSkyOptions['reduction'];
}

export class PlatformHitShading implements HitShading {
    readonly module: WgslModule;
    /** Live: change the fields at any time; the next `encode` uploads them. */
    readonly settings: HitShadingSettings = { ...DEFAULT_HIT_SHADING_SETTINGS };
    readonly sky: PlatformSky;

    private readonly device: GPUDevice;
    /** `GiShadeUniforms`: the sky's irradiance SH (written by the projection), then the settings. */
    private readonly uniforms: GPUBuffer;
    private readonly radianceMapView: GPUTextureView;
    private readonly skySampler: GPUSampler;
    private uploadedSettings = '';
    private disposed = false;

    constructor(device: GPUDevice, sceneModule: WgslModule, raysModule: WgslModule, options: PlatformHitShadingOptions = {}) {
        const label = options.label ?? 'gi';
        this.device = device;
        this.sky = new PlatformSky(device, { label, warn: options.warn, reduction: options.skyReduction });
        // STORAGE because the sky projection writes the SH range of it; tracing kernels bind it as a uniform.
        this.uniforms = device.createBuffer({
            label: `${label} hit shading uniforms`,
            size: SHADE_UNIFORM_LAYOUT.byteLength,
            usage: BUFFER_USAGE.uniform | BUFFER_USAGE.storage | BUFFER_USAGE.copyDst,
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
            dependencies: [sceneModule, raysModule],
            code: (slot: BindingSlot) => shadeModuleWgsl(slot),
            entries: SHADE_MODULE_ENTRIES,
            resources: () => resources,
            version: 0,
            storageBufferCount: 0,
        };
    }

    /** GPU bytes held: the uniform buffer and the sky. */
    get memoryBytes(): number {
        return SHADE_UNIFORM_LAYOUT.byteLength + this.sky.memoryBytes;
    }

    /** Compiles the sky projection. Called by the platform's `prepare`. */
    prepare(): Promise<void> {
        return this.sky.prepare();
    }

    /**
     * Uploads changed settings and re-projects the environment when
     * `environmentVersion` changed (every `setEnvironment` call bumps it), into
     * the host's encoder. Called by the platform's `encode` after the scene
     * copy's uploads and before any technique.
     */
    encode(encoder: GPUCommandEncoder, environment: EnvironmentDescriptor, environmentVersion: number): void {
        if (this.disposed) throw new Error('@pmndrs/ddgi: hit shading was disposed');
        this.uploadSettings();
        this.sky.encode(encoder, environment, environmentVersion, { buffer: this.uniforms, offset: SHADE_UNIFORM_LAYOUT.skyIrradiance });
    }

    dispose(): void {
        if (this.disposed) return;
        this.disposed = true;
        this.sky.dispose();
        this.uniforms.destroy();
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
