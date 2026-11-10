/**
 * The denoiser's inputs, shaped as `@pmndrs/denoiser`'s temporal API names
 * them (`packages/denoiser-core/src/temporal.ts` in the pmndrs/denoiser repo:
 * `FrameCamera`, `FrameGuides`, `DiffuseInputs`, `TemporalConfig`), so a
 * caller of that API can hand the same objects over. They are declared here
 * rather than imported, so this package depends on nothing; TypeScript's
 * structural types make the two interchangeable where the fields agree.
 *
 * What GI-1 needs beyond that API is optional and marked as such: the
 * normals' encoding (`FrameGuides.normalEncoding`) and the input's alpha,
 * which carries Capsaicin's "denoiser hint".
 */

/** Column-major 4 × 4 matrices, as three.js `Matrix4.elements` and WGSL's `mat4x4f`. */
export type Mat4 = Float32Array | ArrayLike<number>;

/** Camera state for this frame and the previous one (for the reprojection). */
export interface FrameCamera {
    /** World-to-view, this frame. */
    readonly view: Mat4;
    /** Projection without jitter, this frame. */
    readonly projection: Mat4;
    /** World-to-view and projection (without jitter) of the previous frame. On the first frame, pass this frame's. */
    readonly prevView: Mat4;
    readonly prevProjection: Mat4;
    /**
     * This frame's sub-pixel jitter in pixels, x right and y down
     * (`@pmndrs/upscaler`'s `Upscaler.jitter`). Default `[0, 0]`. The
     * denoiser remembers it for the next frame, whose previous positions are
     * rebuilt with it.
     */
    readonly jitter?: readonly [number, number];
    readonly near: number;
    readonly far: number;
    /** World-space camera position. Default: the translation of `view`'s inverse. */
    readonly position?: readonly [number, number, number];
    /** The depth buffer is reversed-Z (1 = near). Default false. */
    readonly reversedDepth?: boolean;
}

/** G-buffer guides, the same size as the signal. */
export interface FrameGuides {
    /**
     * Hardware depth (`depth32float`, `depth24plus`) or a one-channel float
     * texture of device depth. A pixel at the far plane (1, or 0 when
     * reversed) is sky.
     */
    readonly depth: GPUTexture;
    /** World-space normals in rgb: unit vectors in [-1, 1] (any float or snorm format), or 0..1 encoded with `normalEncoding: 'unorm'`. */
    readonly normal: GPUTexture;
    /** Screen-space motion in rg: the NDC delta, current − previous (three's `velocity` node). */
    readonly motion: GPUTexture;
    /** How `normal` stores its vectors. Default `'signed'`. Not in `@pmndrs/denoiser`'s guides (which are signed). */
    readonly normalEncoding?: 'signed' | 'unorm';
}

/**
 * The signal: GI-1's interpolated irradiance (`InterpolateScreenProbes`,
 * gi1.comp:1638), `DiffuseInputs` in `@pmndrs/denoiser`.
 */
export interface DiffuseInputs {
    /**
     * Noisy irradiance in rgb (linear HDR, a float format). Alpha is
     * Capsaicin's denoiser hint: above 0 a sample to keep, 0 one the denoiser
     * should not accumulate when it has history (GI-1 writes 0 where the
     * interpolation fell back to unweighted probes, gi1.comp:1636). An input
     * without a hint should write alpha 1.
     */
    readonly radiance: GPUTexture;
}

export interface TemporalConfig {
    readonly width: number;
    readonly height: number;
}

/** Per-pass timestamp writes, from whatever timer the host uses (optional). */
export interface PassTimer {
    computePass(name: string): GPUComputePassTimestampWrites | undefined;
}
