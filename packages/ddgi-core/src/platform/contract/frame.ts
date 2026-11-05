/**
 * Per-frame inputs: what the host tells the platform each frame, and how much
 * work it may do.
 */

import type { Mat4, Vec3 } from './math';

/**
 * How much GPU work the platform may record this frame.
 *
 * Work is the control (budget v2, docs/design/ddgi.md): with `gpuMs`, the
 * platform calibrates once, with the GPU loaded, how many work units (probes
 * for DDGI) fit in `gpuMs`, and hands out that many every frame. It doesn't
 * steer on GPU milliseconds while frames arrive on time, because an idle GPU
 * lowers its clocks and makes the same work look slower. Only when frames are
 * late while the GPU is busy (the work's timings near their full-clock cost)
 * does it hand out fewer, down to the technique's floor. It calibrates after
 * `prepare` and on `GIPlatform.recalibrate()`; a new `gpuMs` is solved from
 * what it measured. Without GPU timings it uses `units`. Probes near a change
 * are spent first.
 */
export interface FrameBudget {
    /**
     * GPU time for all GI updates per frame, measured with the GPU's clocks
     * up. Needs GPU timings: the platform's `gpuTiming` option on a device
     * with `timestamp-query`. Without them `gpuMs` can't be measured, and the
     * hand-out is `units`, or 512 per technique.
     */
    readonly gpuMs?: number;
    /**
     * Work units per frame when no timings are available, or a hard cap when
     * `gpuMs` is also set (over the technique's floor too). The technique
     * defines its unit (DDGI: probes).
     */
    readonly units?: number;
    /**
     * The frame rate the host aims for, frames per second. Optional: without
     * it the platform takes the rate it observes from `deltaTime` when frames
     * are on time, if that is a display refresh rate (vsync, up to 165 Hz);
     * otherwise it treats the host as uncapped, with no frame-rate ceiling
     * (the batch never steps down). Used to tell when frames are late (the
     * only time the batch steps down) and for floors given in seconds (DDGI: a
     * full sweep within 4 s).
     */
    readonly targetFps?: number;
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
