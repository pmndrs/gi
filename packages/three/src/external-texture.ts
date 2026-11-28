/**
 * A same-device `GPUTexture` wrapped for three as an `ExternalTexture`, which
 * three binds without allocating or destroying it. What three must be told,
 * found in the r187 source and the T0.8 spike (NOTES, "Wrapping our
 * textures"):
 *
 * - `image` with the size, or three reads 1×1;
 * - `isArrayTexture` for a 2D array, `is3DTexture` for a 3D texture (three
 *   picks the WGSL type from these);
 * - `type` for the bind-group sample type: `FloatType` for an unfilterable
 *   float texture (`rgba32float`), `UnsignedIntType` for integer data;
 *   three's default is filterable float;
 * - filters and wraps, which also set the sampler a `wgslFn` sampler
 *   parameter gets from the texture's node.
 *
 * On reallocation wrap the new texture and assign it to the node's `value`;
 * `needsUpdate` on an `ExternalTexture` throws.
 */

import { ClampToEdgeWrapping, ExternalTexture, FloatType, LinearFilter, NearestFilter, RepeatWrapping, UnsignedIntType } from 'three/webgpu';

export interface GpuTextureWrapOptions {
    readonly dimension: '2d' | '2d-array' | '3d';
    /** `float` is filterable; `unfilterable-float` for rgba32float; `uint` for integer data. */
    readonly sampleType: 'float' | 'unfilterable-float' | 'uint';
    /** Default `nearest`. */
    readonly filtering?: 'linear' | 'nearest';
    /** Default `clamp-to-edge`. */
    readonly addressMode?: 'clamp-to-edge' | 'repeat';
}

export function wrapGpuTexture(gpuTexture: GPUTexture, options: GpuTextureWrapOptions): ExternalTexture {
    const wrapped = new ExternalTexture(gpuTexture as never);
    (wrapped as unknown as { image: unknown }).image = { width: gpuTexture.width, height: gpuTexture.height, depth: gpuTexture.depthOrArrayLayers };
    if (options.dimension === '2d-array') (wrapped as unknown as { isArrayTexture: boolean }).isArrayTexture = true;
    if (options.dimension === '3d') (wrapped as unknown as { is3DTexture: boolean }).is3DTexture = true;
    if (options.sampleType === 'unfilterable-float') wrapped.type = FloatType;
    if (options.sampleType === 'uint') wrapped.type = UnsignedIntType;
    const filter = options.filtering === 'linear' ? LinearFilter : NearestFilter;
    wrapped.magFilter = filter;
    wrapped.minFilter = filter;
    const wrap = options.addressMode === 'repeat' ? RepeatWrapping : ClampToEdgeWrapping;
    wrapped.wrapS = wrap;
    wrapped.wrapT = wrap;
    wrapped.generateMipmaps = false;
    return wrapped;
}
