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
import type { TechniqueRequirements } from './contract/technique.ts';

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
 * along x: DDGI's blend kernel (packages/ddgi/src/wgsl/blend.ts,
 * `BLEND_WORKGROUP_SIZE`) and the BVH refit's repair pass
 * (@pmndrs/bvh's GpuBvhRefit). Other dimensions stay at 8 or less.
 * Workgroup memory isn't listed here: every platform kernel fits 16,384
 * bytes, the default in both modes; `requiredLimits` asks for the adapter's
 * maximum anyway (below), for technique kernels sized to it.
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
 * about 48-56 bytes per pixel, past WebGPU's default of 32 (RC1.1), and the
 * adapter's `maxComputeWorkgroupStorageSize` (DECISIONS 2026-11-20: no WebGPU
 * default caps; 32 KiB on our GPUs against the default 16 KiB), which kernels
 * that tile by workgroup memory read and fit themselves to (the radiance
 * cache's neural tail, P6.4; a device at the default still works). Never more
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
        maxComputeWorkgroupStorageSize: adapter.limits.maxComputeWorkgroupStorageSize,
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
 *   compressed sources (`CompressedPixelData`) of the families the device has;
 * - `shader-f16`: half-precision math where a kernel offers it (DECISIONS
 *   2026-11-20, feature-forward: the radiance cache's neural tail runs its
 *   network in f16), with f32 the one declared fallback where it is missing.
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
        'shader-f16',
    ];
    return wanted.filter((feature) => adapter.features.has(feature));
}

/** The largest of each limit the techniques declare (`TechniqueRequirements.limits`). */
export function mergeLimitNeeds(requirements: readonly Pick<TechniqueRequirements, 'limits'>[]): RequiredLimits {
    const needs: RequiredLimits = {};
    for (const { limits } of requirements) {
        if (!limits) continue;
        for (const [name, value] of Object.entries(limits) as [keyof RequiredLimits, number | undefined][]) {
            if (value !== undefined) needs[name] = Math.max(needs[name] ?? 0, value);
        }
    }
    return needs;
}

/**
 * Throws a `GILimitError` naming the first declared limit the device is below.
 * Called by `prepare` before anything compiles, so a device short of a
 * technique's need (a 10-storage-buffer kernel on three's own device, with 8)
 * fails with the limit's name instead of a pipeline validation error.
 */
export function checkLimitNeeds(limits: GPUSupportedLimits, needs: RequiredLimits): void {
    for (const [name, needed] of Object.entries(needs) as [keyof RequiredLimits, number | undefined][]) {
        if (needed === undefined) continue;
        const available = limits[name];
        if (available < needed) throw new GILimitError(name, needed, available);
    }
}

/** Whether this device can run the platform at all. */
export function isSupported(device: GPUDevice): boolean {
    return (
        device.limits.maxStorageBuffersPerShaderStage >= MIN_STORAGE_BUFFERS_PER_STAGE &&
        device.limits.maxBindGroups >= MIN_BIND_GROUPS
    );
}
