/**
 * What every GI technique in the viewer provides, so the view, the wipe and
 * the benchmark treat three's `LightProbeGrid`, three's VXGI and (later) our
 * own DDGI the same way. SPEC: "one output contract".
 */

import type { PerspectiveCamera } from 'three';
import type { Node, WebGPURenderer } from 'three/webgpu';
import type { GiMode } from '../scene/layers';
import type { LoadedScene } from '../scene/load-scene';

export type TechniqueId = 'none' | 'lpg' | 'vxgi';

export const TECHNIQUE_LABELS: Record<TechniqueId, string> = {
    none: 'No GI (direct light only)',
    lpg: 'Light probe grid (three.js)',
    vxgi: 'VXGI (three.js)',
};

export interface TechniqueContext {
    renderer: WebGPURenderer;
    device: GPUDevice;
    loaded: LoadedScene;
}

/** Quality presets of the optional temporal upscaler. */
export type UpscaleQuality = 'native-aa' | 'quality' | 'balanced' | 'performance' | 'ultra-performance';

export interface UpscaleSettings {
    quality: UpscaleQuality;
}

export interface GiTechnique {
    readonly id: TechniqueId;
    readonly label: string;
    /** Linear HDR colour for the final composite. Changes when `setUpscale` or `setGiMode('off')` rebuilds the graph. */
    readonly outputNode: Node;
    /** The camera this technique renders with. Its layers decide what is visible. */
    readonly camera: PerspectiveCamera;

    /** See {@link GiMode}. */
    setGiMode(mode: GiMode): void;
    /** Whether `setUpscale` has any effect (VXGI's graph can't be upscaled by `upscaleScene`). */
    readonly supportsUpscale: boolean;
    /** Turns the temporal upscaler on or off. Rebuilds `outputNode`. */
    setUpscale(settings: UpscaleSettings | null): void;
    /** Copies the viewer camera onto this technique's camera. Call once per frame, before rendering. */
    syncCamera(viewerCamera: PerspectiveCamera): void;
    /** Draws anything the technique wants drawn outside the render pipeline. Called once per frame, after `syncCamera`. */
    renderBeforePipeline?(): void;

    /**
     * Set-up (a probe bake, a re-voxelization) is done in steps that only
     * record and submit GPU work; none of them waits for the GPU. Whoever
     * drives the steps decides when to wait: the viewer once per batch of
     * steps, the benchmark once at the very end.
     */
    readonly ready: boolean;
    /** Set-up progress from 0 to 1. */
    readonly progress: number;
    /**
     * Counts restarts. Anything that waits for the GPU after a step compares
     * it before writing results, so a step from before a restart can't
     * report on the new set-up.
     */
    readonly prepareGeneration: number;
    /** One plain-language line for the status area. */
    describe(): string;
    /** Records and submits the next slice of set-up. Returns true when no set-up is left. Never waits. */
    encodePrepareStep(): boolean;
    /** Throws the finished set-up away; the next step starts again from the beginning. */
    restartPrepare(): void;
    /**
     * Called once the GPU has finished the set-up of generation `generation`, with the wall
     * time it took from its first step. Ignored when the set-up has been restarted since.
     */
    notePrepareFinished(generation: number, wallMs: number): void;
    /** Sizes and settings worth recording with a measurement, such as probe count or voxel resolution. */
    describeSettings(): Record<string, number | string>;
    /** Runs a throwaway slice of set-up so shader compilation is not part of a timed one. Waits for the GPU. */
    warmUp(): Promise<void>;

    dispose(): void;
}
