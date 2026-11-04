/**
 * The technique contract: what a GI method implements to plug into the
 * platform. DDGI is the first; voxel, surfel, SDF and cascade techniques are
 * meant to fit the same shape (see docs/api/platform.md for the walk-throughs).
 *
 * Techniques never talk to the host. Everything they know about the scene
 * comes through {@link TechniqueContext}.
 */

import type { FrameCamera, FrameScreen } from './frame';
import type { QueryKind, SceneQueries } from './queries';
import type { SceneChanges, SceneCopy } from './scene-copy';
import type { BounceSource, HitShading } from './shade';
import type { ShadingModule } from './shading-module';
import type { DebugView } from './stats';
import type { GpuTimerScope } from './timing';
import type { KernelBuilder } from './wgsl';

/** What a technique needs from the platform, checked before `prepare`. */
export interface TechniqueRequirements {
    readonly queries: readonly QueryKind[];
    /** Needs {@link FrameScreen} every frame (screen-space techniques). */
    readonly screen?: boolean;
    /** Needs another technique as a bounce source or fallback. */
    readonly fallback?: boolean;
}

/** Services the platform gives a technique when it is attached. */
export interface TechniqueContext {
    readonly device: GPUDevice;
    readonly scene: SceneCopy;
    readonly queries: SceneQueries;
    readonly shade: HitShading;
    readonly kernels: KernelBuilder;
    /** Another technique's output, when the host stacked one under this one. */
    readonly fallback: BounceSource | undefined;
    /** Timestamps for this technique's passes; does nothing without `timestamp-query`. */
    readonly timer: GpuTimerScope;
    /** Warnings meant for the developer (shown once per message). */
    warn(message: string): void;
}

/** Per-frame inputs a technique sees. */
export interface TechniqueFrame {
    readonly frame: number;
    readonly deltaTime: number;
    readonly camera: FrameCamera;
    readonly changes: SceneChanges;
    /**
     * Work units this technique may spend this frame (DDGI: probes), already
     * converted from the host's millisecond budget and shared between techniques.
     */
    readonly units: number;
}

/** Per-frame inputs for the optional screen stage. */
export interface TechniqueScreenFrame extends TechniqueFrame {
    readonly screen: FrameScreen;
}

/**
 * A GI technique. `Settings` is a live object: the host changes fields and the
 * technique reads them every frame, with no apply step. Settings that need new
 * GPU resources or pipelines (probe counts, texture sizes) take effect at the
 * next `prepare`.
 */
export interface GITechnique<Settings extends object = object> {
    /** Short stable id: `ddgi`, `voxel`, `surfels`. */
    readonly id: string;
    readonly requirements: TechniqueRequirements;
    readonly settings: Settings;

    /** Called once when added to a platform. Allocate nothing on the GPU yet. */
    attach(context: TechniqueContext): void;
    /**
     * Allocate resources and compile kernels asynchronously. Called by
     * `GIPlatform.prepare`, and again after settings that need new resources.
     */
    prepare(): Promise<void>;
    /**
     * World-space work: record this frame's updates into the host's encoder.
     * No queue submits and no CPU readbacks.
     *
     * May return the units it actually used (at most `frame.units`; DDGI: the
     * probes it updated, which can be fewer when the volume has fewer). The
     * platform measures cost per unit against that, so a budget bigger than
     * the work can't make units look cheaper than they are. Returning nothing
     * means all of `frame.units` were used.
     */
    encode(encoder: GPUCommandEncoder, frame: TechniqueFrame): number | void;
    /**
     * Optional screen-space work, run after the host has rendered its depth and
     * normals and before it shades (see `GIPlatform.encodeScreen`).
     */
    encodeScreen?(encoder: GPUCommandEncoder, frame: TechniqueScreenFrame): void;

    /** The technique's output for renderers and for other techniques' bounces. */
    readonly shading: ShadingModule;
    readonly debugViews: readonly DebugView[];
    /** Counters for the latest frame, plain-language names. */
    readonly counters: Readonly<Record<string, number>>;
    /** GPU bytes this technique allocated. */
    readonly memoryBytes: number;
    /**
     * Estimated GPU ms per work unit, from timings when available. The
     * platform uses it to turn the host's ms budget into units.
     */
    readonly msPerUnit: number | undefined;

    /** Forget accumulated history (a camera cut or a scene swap). */
    resetHistory(): void;
    /** Free what the technique allocated. Never touches host resources. */
    dispose(): void;
}
