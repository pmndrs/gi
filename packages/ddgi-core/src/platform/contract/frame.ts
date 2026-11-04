/**
 * Per-frame inputs: what the host tells the platform each frame, and how much
 * work it may do.
 */

import type { Mat4, Vec3 } from './math';

/**
 * How much GPU work the platform may record this frame.
 *
 * A millisecond budget can't be enforced directly: GPU timings arrive a few
 * frames late, and not at all without `timestamp-query`. So the platform
 * converts milliseconds into work units (probes for DDGI) using each
 * technique's measured cost per unit, and falls back to `units` when it has
 * no measurements yet. Probes near a change are spent first.
 */
export interface FrameBudget {
    /** Target GPU time for all GI updates this frame. */
    readonly gpuMs?: number;
    /**
     * Work units per frame when no timings are available, or a hard cap when
     * `gpuMs` is also set. The technique defines its unit (DDGI: probes).
     */
    readonly units?: number;
    /** Skip all update work this frame; shading keeps using the last result. */
    readonly paused?: boolean;
}

/** The camera, in plain data, for camera-following volumes and update priority. */
export interface FrameCamera {
    readonly position: Vec3;
    /** World-to-view matrix, column-major. */
    readonly view: Mat4;
    /** Unjittered projection matrix, column-major. */
    readonly projection: Mat4;
    readonly near: number;
    readonly far: number;
}

/**
 * The host's screen buffers, for screen-space techniques (surfel spawning,
 * screen-space GI). World-space techniques like DDGI ignore them. Same
 * fields as `@pmndrs/upscaler`'s temporal guides where they overlap, so one
 * set of buffers feeds GI, the upscaler and the denoiser.
 */
export interface FrameScreen {
    readonly width: number;
    readonly height: number;
    /** Depth, any float format. */
    readonly depth: GPUTexture;
    /** World-space normals in rgb, 0..1 encoded or signed float. */
    readonly normal: GPUTexture;
    readonly normalEncoding: 'unorm' | 'snorm-float';
    /** Screen-space motion (NDC delta, current − previous). */
    readonly velocity?: GPUTexture;
    /**
     * Which platform instance covers each pixel: `r32uint`, instance index + 1,
     * 0 for none. Lets surfel-style techniques attach what they spawn to the
     * object under the pixel so it moves with it. Rendering this is extra
     * adapter work, so only techniques that need it ask for it.
     */
    readonly instance?: GPUTexture;
    /** The depth is reversed-Z (1 = near). */
    readonly reversedDepth?: boolean;
}

export interface FrameInput {
    /** Seconds since the previous frame. */
    readonly deltaTime: number;
    readonly budget: FrameBudget;
    readonly camera: FrameCamera;
    /** Required only by screen-space techniques; see {@link GIPlatform.encodeScreen}. */
    readonly screen?: FrameScreen;
}
