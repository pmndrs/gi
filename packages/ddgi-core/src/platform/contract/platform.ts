/**
 * The host-facing platform: what an adapter (or a plain WebGPU app) calls.
 *
 * An adapter does four things and nothing else is required:
 *
 * 1. **Share the device.** The host creates the `GPUDevice` (asking for
 *    {@link requiredLimits}) and passes it in. The platform never creates,
 *    configures or destroys a device.
 * 2. **Describe the scene by handles,** incrementally, through {@link GIPlatform.scene}.
 * 3. **Drive a frame** with {@link GIPlatform.encode}, recording into the
 *    host's command encoder before the host renders.
 * 4. **Read GI while shading** through {@link GIPlatform.shading}.
 *
 * Typical use:
 *
 * ```ts
 * const device = await adapter.requestDevice({ requiredLimits: requiredLimits(adapter) });
 * const gi = createGI(device, { gpuTiming: true });
 * const ddgi = gi.addTechnique(new DDGITechnique({ volumes: [...] }));
 * gi.setOutput(ddgi);
 * // ...describe the scene with gi.scene.createMesh(...) etc.
 * await gi.prepare();
 *
 * // every frame:
 * const encoder = device.createCommandEncoder();
 * gi.encode(encoder, { deltaTime, budget: { gpuMs: 2 }, camera });
 * renderer.render(encoder); // shading reads gi.shading
 * device.queue.submit([encoder.finish()]);
 * ```
 */

import type { FrameInput, FrameScreen } from './frame';
import type { SceneApi } from './scene';
import type { HitShadingSettings } from './shade';
import type { ShadingModule } from './shading-module';
import type { GIStats } from './stats';
import type { GITechnique } from './technique';

/** The screen channels the attached techniques read (`GIPlatform.screenNeeds`). */
export interface ScreenNeeds {
    /** Some technique has a screen stage: call `encodeScreen` with depth and normals every frame. */
    readonly screen: boolean;
    readonly velocity: boolean;
    readonly albedo: boolean;
    readonly instance: boolean;
    readonly lighting: boolean;
}

export interface GIPlatformOptions {
    /**
     * Fill `stats.passes` with per-pass GPU times when the device has
     * `timestamp-query`; silently does nothing otherwise.
     */
    readonly gpuTiming?: boolean;
    /** Prefix for GPU object labels, so captures show whose objects are whose. */
    readonly label?: string;
    /**
     * Where mesh BVHs are built. `worker` keeps the main thread free; `main`
     * is for environments without module workers (tests, some bundlers).
     * Defaults to `worker`.
     */
    readonly buildIn?: 'worker' | 'main';
    /**
     * Texels on a side of each layer of the texture store, the platform's
     * own copy of every material texture that rays read: 256 (the default),
     * 128 or 64. A smaller store takes a quarter (or a sixteenth) of the
     * memory, and only rays hitting a texture up close see fewer texels: GI
     * rays mostly read coarse mips. The quality presets set it
     * (`ddgiPreset(name).textureLayerSize`). Fixed for the platform's life.
     */
    readonly textureLayerSize?: 64 | 128 | 256;
}

export interface GIPlatform {
    readonly device: GPUDevice;
    /** True after `prepare()` resolves; false again after device loss. */
    readonly isReady: boolean;

    readonly scene: SceneApi;

    /** Attach a technique. Several may be attached; one feeds `shading`. */
    addTechnique<T extends GITechnique>(technique: T): T;
    removeTechnique(technique: GITechnique): void;
    /** The technique whose shading module renderers read. */
    setOutput(technique: GITechnique): void;
    /** The technique `setOutput` chose, or undefined before it (adapters reach its optional controls through it, light group weights say). */
    readonly outputTechnique: GITechnique | undefined;
    /**
     * Stack techniques: `technique` uses `fallback`'s shading module as its
     * bounce source for distant or missed rays. `fallback` is encoded first
     * each frame.
     */
    setFallback(technique: GITechnique, fallback: GITechnique | undefined): void;

    /**
     * Compile kernels and allocate resources, asynchronously. Call after
     * attaching techniques, and again after changing settings that need new
     * resources. Encoding before it resolves throws `GINotReadyError`.
     */
    prepare(): Promise<void>;

    /**
     * Record this frame's world-space GI work into the host's encoder: scene
     * uploads, the top-level BVH rebuild, deformed-mesh refits, then each
     * technique's update within the budget. Call once per frame, before the
     * host's passes that shade with GI.
     */
    encode(encoder: GPUCommandEncoder, frame: FrameInput): void;
    /**
     * Screen-space work for techniques that need it, after the host has
     * written depth and normals and before it shades. A no-op when no attached
     * technique has a screen stage, so hosts may always call it. Call it after
     * the frame's `encode`: screen stages use that frame's camera, whose
     * projection must be the one `screen.depth` was drawn with.
     */
    encodeScreen(encoder: GPUCommandEncoder, screen: FrameScreen): void;
    /**
     * What the attached techniques read from `encodeScreen`, so a host
     * renders only those channels. Changes when techniques are added or
     * removed. `encodeScreen` throws if a channel asked for is missing.
     */
    readonly screenNeeds: ScreenNeeds;
    /**
     * The device limits the attached techniques need above WebGPU's defaults
     * (`TechniqueRequirements.limits`), the largest of each: for example the
     * radiance cache's `maxStorageBuffersPerShaderStage` of 10 (Apple GPUs in
     * Chrome allow 10, a device three.js makes itself 8). `requiredLimits`
     * asks for at least these where the adapter has them; `prepare` throws a
     * `GILimitError` naming the first one the device is below.
     */
    readonly limitNeeds: RequiredLimits;

    /** The output technique's shading module. Re-read `version` each frame. */
    readonly shading: ShadingModule;
    readonly stats: GIStats;
    /**
     * Hit shading's host-facing settings, live: set a field and the next
     * frame uses it, with no recompile. `sampleTextures` (default true): ray
     * hits read their materials' textures; false reads each texture's mean
     * colour (its last mip) instead, the flat colours of before M4, for
     * comparisons. Changing it logs no scene change, so the probes blend
     * towards the new look over their usual updates; call `resetHistory()`
     * to see it at once.
     */
    readonly hitShadingSettings: HostHitShadingSettings;

    /** Forget all accumulated history (a camera cut or a scene swap). */
    resetHistory(): void;
    /**
     * Measure again how much work fits in `budget.gpuMs` (budget v2), at the
     * next frame with a millisecond budget: a short burst of frames, best run
     * back to back behind a loading screen. The platform calibrates by itself
     * after `prepare`; a geometry change only marks the calibration stale
     * (`stats.counters['<technique> budget'].stale`), because a burst during
     * play makes frames heavier for a moment. A new `gpuMs` needs no new
     * calibration.
     */
    recalibrate(): void;
    /** Free everything the platform allocated. The device and host resources are untouched. */
    dispose(): void;
}

/**
 * The hit-shading settings a host may change, live (read every frame). Only
 * `sampleTextures` for now (Dennis, DECISIONS 2026-11-18): the others in
 * {@link HitShadingSettings} stay for techniques and the repository's tools
 * until a host needs them.
 */
export type HostHitShadingSettings = Pick<HitShadingSettings, 'sampleTextures'>;

/** Limits the platform would like for large scenes. Pass them to `requestDevice`. */
export type RequiredLimits = Partial<Record<Exclude<keyof GPUSupportedLimits, '__brand'>, number>>;

/**
 * The factory and static helpers the implementation exports. Declared here so
 * the contract is complete; implemented in M1.
 */
export interface GIPlatformModule {
    createGI(device: GPUDevice, options?: GIPlatformOptions): GIPlatform;
    /**
     * Limits to request from this adapter: the largest storage binding and
     * buffer, and every storage buffer per stage, capped at what the adapter
     * supports. Small scenes (Cornell, two-room, the furnished room) work at
     * default limits; large ones (Sponza, Bistro) need these raised limits,
     * because nothing is split across bindings (DECISIONS 2026-11-05). Without
     * them the platform throws a `GILimitError` naming the limit.
     */
    requiredLimits(adapter: GPUAdapter): RequiredLimits;
    /** Optional features worth requesting (`timestamp-query`, `float32-filterable`). */
    optionalFeatures(adapter: GPUAdapter): GPUFeatureName[];
    /** Whether this device can run the platform at all. */
    isSupported(device: GPUDevice): boolean;
}
