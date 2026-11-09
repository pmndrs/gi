// The radiance cache's world cache (phase 3): the slot the screen cache hands its probe rays
// to, and Capsaicin's hash grid behind it (docs/design/radiance-cache.md, "World cache as a slot").
export {
    checkHashGridCacheLimits,
    DEFAULT_HASH_GRID_CACHE_OPTIONS,
    HASH_GRID_COUNTER,
    hashGridCacheBufferLayout,
    hashGridCacheCapacity,
    type HashGridCacheBufferLayout,
    type HashGridCacheCapacity,
    type HashGridCacheOptions,
} from './capacity.ts';
export { HashGridCache, type HashGridCacheBuffers } from './hash-grid-cache.ts';
export { HASH_GRID_CACHE_GROUP_ENTRIES } from './hash-grid-wgsl.ts';
export {
    WORLD_CACHE_PROBE_RAY_BYPASS,
    WORLD_CACHE_PROBE_RAY_BYTES,
    WORLD_CACHE_PROBE_RAYS_HEADER_BYTES,
    WORLD_CACHE_PROBE_RAYS_WGSL,
    WORLD_CACHE_RADIANCE_SLOT_BYTES,
    type WorldCache,
    type WorldCacheContext,
    type WorldCacheFrame,
    type WorldCacheLookup,
    type WorldCacheProbeRays,
} from './world-cache.ts';
