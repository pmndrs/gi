/**
 * What the platform asks of a device: `requiredLimits`, `optionalFeatures`
 * and `isSupported` (contract: platform.ts, `GIPlatformModule`).
 *
 * The host creates the device, so these only advise. A host that passes
 * `requiredLimits(adapter)` to `requestDevice` gets the largest storage
 * bindings its adapter has, which large scenes need: the tracer's triangle
 * records for Bistro exterior don't fit the default storage binding.
 */

import type { RequiredLimits } from './contract/platform.ts';

/**
 * The platform's binding budget assumes the default 8 storage buffers per
 * shader stage: a DDGI update kernel uses 7 (docs/design/ddgi.md, "Bindings").
 */
export const MIN_STORAGE_BUFFERS_PER_STAGE = 8;

/** Platform modules share group 0; techniques use groups 1-3. */
export const MIN_BIND_GROUPS = 4;

/**
 * Limits to request from this adapter: the largest storage binding and
 * buffer, and every storage buffer per stage it offers. Never more than the
 * adapter has, so `requestDevice` can't fail because of them.
 */
export function requiredLimits(adapter: GPUAdapter): RequiredLimits {
    return {
        maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
        maxBufferSize: adapter.limits.maxBufferSize,
        maxStorageBuffersPerShaderStage: adapter.limits.maxStorageBuffersPerShaderStage,
    };
}

/**
 * Features worth requesting when the adapter has them:
 * - `subgroups`: compute kernels are written for them first (DECISIONS 2026-11-04);
 * - `timestamp-query`: per-pass GPU times in `stats.passes`;
 * - `float32-filterable`: lets shading modules filter `rgba32float` textures;
 * - `texture-formats-tier1`: makes `rg16float` a storage format, so DDGI's
 *   distance atlas takes half the memory (without it: `rgba16float`).
 */
export function optionalFeatures(adapter: GPUAdapter): GPUFeatureName[] {
    const wanted: GPUFeatureName[] = ['subgroups', 'timestamp-query', 'float32-filterable', 'texture-formats-tier1'];
    return wanted.filter((feature) => adapter.features.has(feature));
}

/** Whether this device can run the platform at all. */
export function isSupported(device: GPUDevice): boolean {
    return (
        device.limits.maxStorageBuffersPerShaderStage >= MIN_STORAGE_BUFFERS_PER_STAGE &&
        device.limits.maxBindGroups >= MIN_BIND_GROUPS
    );
}
