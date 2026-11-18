/**
 * What the platform asks of a device: `requiredLimits`, `optionalFeatures`
 * and `isSupported` (contract: platform.ts, `GIPlatformModule`).
 *
 * The host creates the device, so these only advise. A host that passes
 * `requiredLimits(adapter)` to `requestDevice` gets the largest storage
 * bindings its adapter has, which large scenes need: the tracer's triangle
 * records for Bistro exterior don't fit the default storage binding.
 */

import { GILimitError } from './contract/errors.ts';
import type { RequiredLimits } from './contract/platform.ts';

/**
 * The platform's binding budget assumes the default 8 storage buffers per
 * shader stage: a DDGI update kernel uses 7 (docs/design/ddgi.md, "Bindings").
 */
export const MIN_STORAGE_BUFFERS_PER_STAGE = 8;

/** Platform modules share group 0; techniques use groups 1-3. */
export const MIN_BIND_GROUPS = 4;

/**
 * Compute limits the platform's kernels need where compatibility-mode
 * devices default lower (128 threads per workgroup; core devices default to
 * 256, so these change nothing there). The largest workgroups, 256 threads
 * along x: DDGI's blend kernel (techniques/ddgi/wgsl/blend.ts,
 * `BLEND_WORKGROUP_SIZE`) and the BVH refit's repair pass
 * (packages/bvh/src/gpu-refit.ts). Other dimensions stay at 8 or less.
 * Workgroup memory isn't listed: every kernel fits 16,384 bytes, the default
 * in both modes (all GPU pages run on default limits).
 */
export const KERNEL_COMPUTE_LIMITS = {
    maxComputeInvocationsPerWorkgroup: 256,
    maxComputeWorkgroupSizeX: 256,
} as const satisfies Partial<Record<Exclude<keyof GPUSupportedLimits, '__brand'>, number>>;

type ComputeLimitName = keyof typeof KERNEL_COMPUTE_LIMITS;
const COMPUTE_LIMIT_NAMES = Object.keys(KERNEL_COMPUTE_LIMITS) as ComputeLimitName[];

/**
 * Throws a `GILimitError` naming the first compute limit the device is below
 * (KERNEL_COMPUTE_LIMITS). Called by `prepare`, before any kernel compiles,
 * so a short device fails with the limit's name instead of a pipeline
 * validation error.
 */
export function checkComputeLimits(limits: Pick<GPUSupportedLimits, ComputeLimitName>): void {
    for (const name of COMPUTE_LIMIT_NAMES) {
        if (limits[name] < KERNEL_COMPUTE_LIMITS[name]) throw new GILimitError(name, KERNEL_COMPUTE_LIMITS[name], limits[name]);
    }
}

/**
 * Limits to request from this adapter: the largest storage binding and
 * buffer, every storage buffer per stage it offers, the most texture array
 * layers (one per texture in the texture store; the default is 256, adapters
 * give 2,048), the compute limits the kernels need (KERNEL_COMPUTE_LIMITS),
 * and the adapter's `maxColorAttachmentBytesPerSample`: a host's scene pass
 * that writes the screen inputs (`FrameScreen`) beside its own outputs needs
 * about 48-56 bytes per pixel, past WebGPU's default of 32 (RC1.1). Never more
 * than the adapter has, so `requestDevice` can't fail because of them; an
 * adapter below a kernel's need fails in `prepare` with a `GILimitError`
 * instead.
 */
export function requiredLimits(adapter: GPUAdapter): RequiredLimits {
    const limits: RequiredLimits = {
        maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
        maxBufferSize: adapter.limits.maxBufferSize,
        maxStorageBuffersPerShaderStage: adapter.limits.maxStorageBuffersPerShaderStage,
        maxTextureArrayLayers: adapter.limits.maxTextureArrayLayers,
        maxColorAttachmentBytesPerSample: adapter.limits.maxColorAttachmentBytesPerSample,
    };
    for (const name of COMPUTE_LIMIT_NAMES) limits[name] = Math.min(KERNEL_COMPUTE_LIMITS[name], adapter.limits[name]);
    return limits;
}

/**
 * Features worth requesting when the adapter has them:
 * - `subgroups`: compute kernels are written for them first (DECISIONS 2026-11-04);
 * - `timestamp-query`: per-pass GPU times in `stats.passes`;
 * - `float32-filterable`: lets shading modules filter `rgba32float` textures;
 * - `texture-formats-tier1`: makes `rg16float` a storage format, so DDGI's
 *   distance atlas takes half the memory (without it: `rgba16float`);
 * - `texture-compression-bc`, `-etc2`, `-astc`: the texture store reads
 *   compressed sources (`CompressedPixelData`) of the families the device has.
 */
export function optionalFeatures(adapter: GPUAdapter): GPUFeatureName[] {
    const wanted: GPUFeatureName[] = [
        'subgroups',
        'timestamp-query',
        'float32-filterable',
        'texture-formats-tier1',
        'texture-compression-bc',
        'texture-compression-etc2',
        'texture-compression-astc',
    ];
    return wanted.filter((feature) => adapter.features.has(feature));
}

/** Whether this device can run the platform at all. */
export function isSupported(device: GPUDevice): boolean {
    return (
        device.limits.maxStorageBuffersPerShaderStage >= MIN_STORAGE_BUFFERS_PER_STAGE &&
        device.limits.maxBindGroups >= MIN_BIND_GROUPS
    );
}
