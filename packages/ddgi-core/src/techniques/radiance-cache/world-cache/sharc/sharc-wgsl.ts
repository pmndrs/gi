/**
 * The SHaRC-style spatial hash cache's WGSL (phase 6 experiment P6.1,
 * docs/design/radiance-cache.md register E1-E7): an alternative backend for
 * the radiance cache's world-cache slot, beside Capsaicin's hash grid.
 *
 * Ported from the pmndrs/pathtracer project's hash-grid radiance cache
 * (`bench/nrc/hash-grid.ts` at dd771c8, MIT, Copyright (c) 2026 Dennis Smolek;
 * design in its `docs/design/nrc-hash-cache.md`, divergences in
 * `docs/design/nrc-diverges.md` §D). That design comes from the published
 * descriptions only: Binder et al., "Massively parallel path space filtering"
 * (arXiv 1902.05942), and SHaRC's integration notes. No NVIDIA SDK code was
 * read or copied, here or there (THIRD_PARTY.md).
 *
 * Taken as the pathtracer has it:
 *
 * - **The cell:** (level, the integer cell of size `minCell · 2^level`, a
 *   normal bin). The level grows with the distance to the camera so a cell
 *   covers about `cellPixels` pixels wherever it is (SHaRC's LOD rule:
 *   `size = max(d · pixelAngle · cellPixels, minCell)`,
 *   `level = floor(log2(size / minCell))`). Normal bin: the dominant axis and
 *   its sign (6 bins), or one bin.
 * - **The table:** `2^log2Entries` slots in buckets of 8. A slot's key word is
 *   a 32-bit checksum of the cell (0 = empty), a second hash independent of
 *   the bucket's (PCG for the bucket, murmur3's finaliser for the checksum),
 *   claimed by compare-exchange. Lookups scan the whole bucket, so evicting a
 *   cell never hides another.
 * - **The update:** each sample adds its radiance into its cell as u32 fixed
 *   point (×256, stochastic rounding, clamped at 1000) plus a count (WGSL has
 *   no float atomics). With the pyramid, it also adds into its
 *   `fallbackLevels` coarser cells.
 * - **The resolve:** every slot, every frame: an exponentially weighted sample
 *   mean, `w = w_old · (1 − 1/historyFrames) + n`, the history ageing only in
 *   frames with new samples; a slot without samples for `staleFrames` frames
 *   is evicted.
 * - **The lookup:** the finest of the cell and its coarser levels with at
 *   least `minSamples` of sample weight.
 *
 * What the slot changes (register E1-E7): the samples come from the probe
 * rays' hits, not from the pathtracer's training records. Each probe ray that
 * hit runs a short diffuse update path from its hit: one light sample there
 * (the platform's light choice and a shadow ray, as the hash grid lights its
 * hits), then up to `bounceRays` cosine-weighted bounces, the last ending in a
 * cache lookup ("the cache as the path's tail") or, where the cache has too
 * few samples or the segment is shorter than its cell, in the tail hit's own
 * light sample. The radiance found is carried back along the path, and each
 * vertex's estimate goes into its cell. Then the resolve, then each probe ray
 * reads its cell (or its own sample when the ray is shorter than its cell:
 * SHaRC's "segment longer than the voxel" rule, and GI-1.2's bypass).
 *
 * A cell holds the light reflected off its surface toward the side the rays
 * came from (the platform's materials are Lambert, so it is view-independent);
 * emission is added straight to the probe ray, never cached.
 */

import { CAPSAICIN_WGSL } from '../../wgsl/capsaicin.ts';
import { WORLD_CACHE_PROBE_RAYS_WGSL } from '../world-cache.ts';

/** `GPUShaderStage.COMPUTE` (the global doesn't exist in Node, where the tests import this file). */
const SHADER_STAGE_COMPUTE = 0x4;

/** Slots per bucket: a compile-time constant so the probe loops have a constant trip count (Tint loop counters on D3D12). */
export const SHARC_BUCKET = 8;
/** Fixed-point scale of the accumulators. */
export const SHARC_SCALE = 256;
/** Per-sample clamp, so a u32 sum holds at least 16k clamped samples a frame. */
export const SHARC_CLAMP = 1000;
/** The highest level the LOD rule picks. */
export const SHARC_MAX_LEVEL = 24;
/** The most bounce rays an update path traces (`SharcSettings.bounceRays`). */
export const SHARC_MAX_BOUNCE_RAYS = 2;
/** Words per slot in the accumulator buffer: r, g, b, count (atomic), then the frame it last had samples. */
export const SHARC_SLOT_WORDS = 5;
/** Words before the first slot in the accumulator buffer: the counters. */
export const SHARC_COUNTER_WORDS = 16;
/** Bytes of `SharcParams`. */
export const SHARC_PARAMS_BYTES = 64;
/** Bytes a probe ray keeps between the update and the lookup: two vec4f. */
export const SHARC_RAY_SAMPLE_BYTES = 32;
/** Threads per workgroup of the per-ray passes and of the resolve. */
export const SHARC_WORKGROUP_SIZE = 64;
export const SHARC_RESOLVE_WORKGROUP_SIZE = 256;

/** The counter words at the start of the accumulator buffer (cleared each frame, but `rayCount`, which is copied in). */
export const SHARC_COUNTER = {
    /** This frame's probe-ray count, copied from the screen cache's buffer for the indirect arguments. */
    rayCount: 0,
    /** Inserts refused because the bucket was full. */
    droppedInserts: 1,
    /** Probe rays that read their own sample (shorter than their cell). */
    ownReads: 2,
    /** Probe rays that read a cell coarser than their own. */
    coarseReads: 3,
    /** Probe rays whose cells, all levels, have fewer than `minSamples` (read the best-filled one). */
    thinReads: 4,
    /** Probe rays that read a cell. */
    cacheReads: 5,
    /** Update paths whose tail was read from the cache. */
    tailFromCache: 6,
    /** Update paths whose tail hit was lit by its own light sample instead (too few samples, or a short segment). */
    tailFromLight: 7,
    /** Update paths whose last bounce missed (the sky, with `skyOnMiss`). */
    tailMisses: 8,
} as const;

/** `w` of a probe ray's kept sample (`SharcRaySample.sample`): what the lookup does with the ray. */
export const SHARC_RAY = {
    /** A miss, a ray the screen cache lit, or out of range: nothing to add. */
    none: -1,
    /** The ray reads its cell. */
    lookup: 0,
    /** The ray is shorter than its cell: it reads its own sample. */
    own: 1,
} as const;

/** Entry points. */
export const SHARC_ENTRY_POINT = {
    update: 'SharcUpdate',
    resolve: 'SharcResolve',
    lookup: 'SharcLookup',
} as const;

/** The cache's group in its kernels (the platform's modules are group 0). */
export const SHARC_CACHE_GROUP = 1;
/** The frame's probe rays and their kept samples. */
export const SHARC_RAY_GROUP = 2;

export const SHARC_CACHE_BINDING = {
    params: 0,
    keys: 1,
    accum: 2,
    resolved: 3,
} as const;

export const SHARC_RAY_BINDING = {
    probeRays: 0,
    radiance: 1,
    raySamples: 2,
} as const;

export const SHARC_CACHE_GROUP_ENTRIES: readonly GPUBindGroupLayoutEntry[] = [
    { binding: SHARC_CACHE_BINDING.params, visibility: SHADER_STAGE_COMPUTE, buffer: { type: 'uniform' } },
    { binding: SHARC_CACHE_BINDING.keys, visibility: SHADER_STAGE_COMPUTE, buffer: { type: 'storage' } },
    { binding: SHARC_CACHE_BINDING.accum, visibility: SHADER_STAGE_COMPUTE, buffer: { type: 'storage' } },
    { binding: SHARC_CACHE_BINDING.resolved, visibility: SHADER_STAGE_COMPUTE, buffer: { type: 'storage' } },
];

export const SHARC_RAY_GROUP_ENTRIES: readonly GPUBindGroupLayoutEntry[] = [
    { binding: SHARC_RAY_BINDING.probeRays, visibility: SHADER_STAGE_COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: SHARC_RAY_BINDING.radiance, visibility: SHADER_STAGE_COMPUTE, buffer: { type: 'storage' } },
    { binding: SHARC_RAY_BINDING.raySamples, visibility: SHADER_STAGE_COMPUTE, buffer: { type: 'storage' } },
];

/** The read-only group `WorldCacheLookup` readers bind: the params, keys and resolved values. */
export const SHARC_LOOKUP_GROUP_ENTRIES: readonly GPUBindGroupLayoutEntry[] = [
    { binding: SHARC_CACHE_BINDING.params, visibility: SHADER_STAGE_COMPUTE, buffer: { type: 'uniform' } },
    { binding: SHARC_CACHE_BINDING.keys, visibility: SHADER_STAGE_COMPUTE, buffer: { type: 'read-only-storage' } },
    { binding: SHARC_CACHE_BINDING.resolved, visibility: SHADER_STAGE_COMPUTE, buffer: { type: 'read-only-storage' } },
];

/** What the compiled code bakes in: a change needs `prepare` again. */
export interface SharcCompileOptions {
    /** Coarser levels a lookup walks (and, with `pyramid`, an insert also adds into). */
    readonly fallbackLevels: number;
    /** Each sample also adds into its `fallbackLevels` coarser cells. */
    readonly pyramid: boolean;
}

/** `SharcParams`, word by word (64 bytes). */
export const SHARC_PARAMS_WORD = {
    /** vec3 eye, then the cell size per metre of distance (pixel angle × cellPixels). */
    camera: 0,
    cellSizePerMetre: 3,
    minCell: 4,
    decay: 5,
    minSamples: 6,
    frame: 7,
    mask: 8,
    staleFrames: 9,
    normalBins: 10,
    bounceRays: 11,
    cacheTail: 12,
    useDirectLighting: 13,
    skyOnMiss: 14,
    updateFraction: 15,
} as const;

/**
 * The cell, hash and lookup code every SHaRC kernel shares, with the cache's
 * bindings in `group`. `access` is `read_write` for the cache's own kernels
 * (keys atomic) and `read` for outside readers (`worldCacheLookup`).
 */
export function sharcCommonWgsl(group: number, options: SharcCompileOptions, access: 'read_write' | 'read'): string {
    const fallback = Math.max(0, Math.floor(options.fallbackLevels));
    const pyramid = options.pyramid ? fallback : 0;
    const own = access === 'read_write';
    const keyType = own ? 'array<atomic<u32>>' : 'array<u32>';
    const keyLoad = own ? 'atomicLoad(&g_SharcKeys[slot])' : 'g_SharcKeys[slot]';
    return /* wgsl */ `
// ---- SHaRC-style hash cache (world-cache/sharc; ported from pmndrs/pathtracer's bench/nrc/hash-grid.ts) ----

struct SharcParams {
    camera: vec4f,            // xyz: the eye; w: a cell's size per metre of distance (pixel angle x cellPixels)
    minCell: f32,             // the smallest cell, in scene units
    decay: f32,               // 1 - 1 / historyFrames
    minSamples: f32,          // a level with less sample weight is skipped by the lookup
    frame: u32,               // the frame number (wraps), for the age stamps and the random numbers
    mask: u32,                // slots - 1
    staleFrames: u32,         // a slot without samples for longer is evicted
    normalBins: u32,          // 1: 6 bins by the normal's dominant axis and sign; 0: one bin
    bounceRays: u32,          // bounce rays per update path (0-${SHARC_MAX_BOUNCE_RAYS})
    cacheTail: u32,           // 1: the path ends in a cache lookup; 0: in the tail hit's own light sample
    useDirectLighting: u32,   // RadianceCacheSettings.useDirectLighting: emissive hits add their emission
    skyOnMiss: u32,           // 1: a bounce ray that misses brings the sky's radiance
    updateFraction: f32,      // the share of probes (workgroups of 64 rays) that run the update path this frame
}

const SHARC_BUCKET: u32 = ${SHARC_BUCKET}u;
const SHARC_MAX_LEVEL: u32 = ${SHARC_MAX_LEVEL}u;
const SHARC_FALLBACK_LEVELS: u32 = ${fallback}u;
const SHARC_PYRAMID_LEVELS: u32 = ${pyramid}u;
const SHARC_SLOT_WORDS: u32 = ${SHARC_SLOT_WORDS}u;
const SHARC_COUNTER_WORDS: u32 = ${SHARC_COUNTER_WORDS}u;
const SHARC_SCALE: f32 = ${SHARC_SCALE}.0;

@group(${group}) @binding(${SHARC_CACHE_BINDING.params}) var<uniform> g_Sharc: SharcParams;
// Per slot: the cell's checksum, 0 when the slot is empty.
@group(${group}) @binding(${SHARC_CACHE_BINDING.keys}) var<storage, ${access}> g_SharcKeys: ${keyType};
// Per slot: the resolved radiance (rgb) and its sample weight (w).
@group(${group}) @binding(${SHARC_CACHE_BINDING.resolved}) var<storage, ${access}> g_SharcResolved: array<vec4f>;

// PCG hash (the pathtracer's hgMix): the bucket chain.
fn sharcMix(x: u32) -> u32 {
    let s = x * 747796405u + 2891336453u;
    let w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
    return (w >> 22u) ^ w;
}

// murmur3's finaliser (the pathtracer's hgMix2): the checksum chain, independent of the bucket's.
fn sharcMix2(x: u32) -> u32 {
    var h = x * 0x85ebca6bu;
    h ^= h >> 13u;
    h *= 0xc2b2ae35u;
    h ^= h >> 16u;
    return h;
}

// SHaRC's LOD rule: the level at which a cell covers about cellPixels pixels at this distance.
fn sharcLevel(position: vec3f) -> u32 {
    let size = max(distance(position, g_Sharc.camera.xyz) * g_Sharc.camera.w, g_Sharc.minCell);
    return min(u32(max(floor(log2(size / g_Sharc.minCell)), 0.0)), SHARC_MAX_LEVEL);
}

fn sharcCellSize(level: u32) -> f32 {
    return g_Sharc.minCell * exp2(f32(level));
}

// 1-6: the normal's dominant axis and its sign; 0 with one bin for all normals.
fn sharcNormalBin(normal: vec3f) -> u32 {
    if (g_Sharc.normalBins == 0u) {
        return 0u;
    }
    let a = abs(normal);
    if (a.x >= a.y && a.x >= a.z) {
        return select(1u, 2u, normal.x < 0.0);
    }
    if (a.y >= a.z) {
        return select(3u, 4u, normal.y < 0.0);
    }
    return select(5u, 6u, normal.z < 0.0);
}

// (first slot of the cell's bucket, the cell's checksum, never 0). WGSL has no 64-bit atomics, so
// SHaRC's 64-bit key is a bucket index plus a 32-bit checksum: a false match needs both to collide.
fn sharcKey(position: vec3f, bin: u32, level: u32) -> vec2u {
    let cell = bitcast<vec3u>(vec3i(floor(position / sharcCellSize(level))));
    let levelAndBin = level | (bin << 8u);
    let h = sharcMix(sharcMix(sharcMix(sharcMix(cell.x + 0x9e3779b9u) ^ cell.y) ^ cell.z) ^ levelAndBin);
    let g = sharcMix2(sharcMix2(sharcMix2(sharcMix2(cell.z ^ 0x27d4eb2fu) + cell.x) + cell.y) + levelAndBin * 0x165667b1u);
    return vec2u(h & g_Sharc.mask & ~(SHARC_BUCKET - 1u), max(g, 1u));
}

// The slot holding the cell, or -1. The whole bucket is scanned (no early stop at an empty slot).
fn sharcFind(key: vec2u) -> i32 {
    for (var i = 0u; i < SHARC_BUCKET; i++) {
        let slot = key.x + i;
        if (${keyLoad} == key.y) {
            return i32(slot);
        }
    }
    return -1;
}

struct SharcCellRead {
    radiance: vec3f,
    weight: f32,      // the read level's sample weight; 0 when no level has samples
    levelOffset: u32, // 0: the cell's own level
}

// The radiance leaving 'position' on the side 'normal' faces: the finest of the cell and its
// SHARC_FALLBACK_LEVELS coarser cells with at least minSamples of weight; if none has that many,
// the one with the most weight (so a young cache reads its best guess, not black).
fn sharcRead(position: vec3f, normal: vec3f) -> SharcCellRead {
    let level = sharcLevel(position);
    let bin = sharcNormalBin(normal);
    var best = SharcCellRead(vec3f(0.0), 0.0, 0u);
    for (var j = 0u; j <= SHARC_FALLBACK_LEVELS; j++) {
        let slot = sharcFind(sharcKey(position, bin, min(level + j, 31u)));
        if (slot < 0) {
            continue;
        }
        let value = g_SharcResolved[u32(slot)];
        if (value.w >= g_Sharc.minSamples) {
            return SharcCellRead(value.xyz, value.w, j);
        }
        if (value.w > best.weight) {
            best = SharcCellRead(value.xyz, value.w, j);
        }
    }
    return best;
}
`;
}

/** The probe rays, their radiance slots and what the update keeps per ray for the lookup. */
const RAY_GROUP_WGSL = /* wgsl */ `
${WORLD_CACHE_PROBE_RAYS_WGSL}

// What the update keeps per probe ray for the lookup: two vec4f.
//   [2i]:     the ray's own estimate (rgb) and what the lookup does (w: SHARC_RAY_*), as f32 bits
//   [2i + 1]: where the estimate was inserted (xyz, f32 bits) and the normal it was binned by (pack4x8snorm)
// Kept as words, not floats: the packed normal's bits could read as a NaN, which a float store may not keep.
@group(${SHARC_RAY_GROUP}) @binding(${SHARC_RAY_BINDING.probeRays}) var<storage, read> g_SharcProbeRays: WorldCacheProbeRays;
@group(${SHARC_RAY_GROUP}) @binding(${SHARC_RAY_BINDING.radiance}) var<storage, read_write> g_SharcProbeRadiance: array<vec2u>;
@group(${SHARC_RAY_GROUP}) @binding(${SHARC_RAY_BINDING.raySamples}) var<storage, read_write> g_SharcRaySamples: array<vec4u>;

const SHARC_RAY_NONE: f32 = ${SHARC_RAY.none}.0;
const SHARC_RAY_LOOKUP: f32 = ${SHARC_RAY.lookup}.0;
const SHARC_RAY_OWN: f32 = ${SHARC_RAY.own}.0;

// Adds to a probe ray's radiance, keeping its hit distance (w), rounded to the nearest half as the
// hash grid's ScreenProbes_AccumulateRadiance does (register D31: pack2x16float truncates on our GPUs).
fn sharcAddToProbeRay(ray: u32, radiance: vec3f) {
    let packed = g_SharcProbeRadiance[ray];
    let value = vec4f(unpack2x16float(packed.x), unpack2x16float(packed.y));
    let sum = rcNearestHalf4(vec4f(value.xyz + radiance, value.w));
    g_SharcProbeRadiance[ray] = vec2u(pack2x16float(sum.xy), pack2x16float(sum.zw));
}

// The workgroup a 2D-split dispatch is on, counted in rows, as the platform's indirect arguments lay them out.
fn sharcWorkgroupIndex(workgroup: vec3u, workgroups: vec3u) -> u32 {
    return workgroup.y * workgroups.x + workgroup.x;
}
`;

/** The accumulators, declared for the cache's own kernels only. */
const ACCUM_WGSL = /* wgsl */ `
// The counters (SHARC_COUNTER_WORDS words), then per slot: r, g, b and the sample count as fixed
// point (atomic adds), and the frame the slot last had samples.
@group(${SHARC_CACHE_GROUP}) @binding(${SHARC_CACHE_BINDING.accum}) var<storage, read_write> g_SharcAccum: array<atomic<u32>>;

fn sharcSlotWord(slot: u32) -> u32 {
    return SHARC_COUNTER_WORDS + SHARC_SLOT_WORDS * slot;
}

fn sharcCount(counter: u32, amount: u32) {
    if (amount > 0u) {
        atomicAdd(&g_SharcAccum[counter], amount);
    }
}
`;

/**
 * `SharcUpdate`: one thread per probe ray. A ray that hit (and that the
 * screen cache didn't light itself) adds its hit's emission to its radiance
 * (front faces, `useDirectLighting`), runs the update path from its hit (see
 * the file's comment), inserts each vertex's estimate into its cell, and
 * keeps its own estimate and the cell it went into for `SharcLookup`.
 *
 * Built with the platform's hit-shading module (group 0), which brings the
 * tracer (`giTraceClosest`, `giTraceAny`), the surface rebuild and the light
 * choice.
 */
export function sharcUpdateWgsl(options: SharcCompileOptions): string {
    return /* wgsl */ `
${CAPSAICIN_WGSL}
${sharcCommonWgsl(SHARC_CACHE_GROUP, options, 'read_write')}
${ACCUM_WGSL}
${RAY_GROUP_WGSL}

const SHARC_LIGHT_GROUPS: u32 = 0xfu;
const SHARC_INV_PI: f32 = 0.318309886183791;
const SHARC_TWO_PI: f32 = 6.28318530717959;
const SHARC_MAX_BOUNCE_RAYS: u32 = ${SHARC_MAX_BOUNCE_RAYS}u;
const SHARC_CLAMP: f32 = ${SHARC_CLAMP}.0;
// Keeps the update's random numbers apart from the screen cache's, which seeds by ray index too.
const SHARC_SEED_SALT: u32 = 0xa511e9b3u;

fn sharcFinite(value: vec3f) -> bool {
    return all((bitcast<vec3u>(value) & vec3u(0x7fffffffu)) < vec3u(0x7f800000u));
}

// One light sample at a Lambert surface, as the hash grid's GenerateReservoirs and PopulateCells
// light a hit: the platform's one-light choice, its weight, and a shadow ray. The light reflected
// toward the side the surface was reached from.
fn sharcDirect(surface: GiSurface, random: ptr<function, RcRandom>) -> vec3f {
    let choice = giDirectChooseLight(surface, SHARC_LIGHT_GROUPS, rcRand(random));
    let sampleParams = rcRand2(random);
    if (choice.light == GI_DIRECT_NO_LIGHT) {
        return vec3f(0.0);
    }
    let lightRay = giDirectLightRay(surface, choice.light, sampleParams);
    if (lightRay.castsShadows && giTraceAny(lightRay.shadowRay)) {
        return vec3f(0.0);
    }
    return surface.albedo * SHARC_INV_PI * lightRay.irradiance * choice.weight;
}

// A cosine-weighted direction about 'normal' (Duff et al. 2017's orthonormal basis). For a Lambert
// surface the path's weight for it is the albedo: (albedo / pi) cos / (cos / pi).
fn sharcCosineDirection(normal: vec3f, u: vec2f) -> vec3f {
    let s = select(-1.0, 1.0, normal.z >= 0.0);
    let a = -1.0 / (s + normal.z);
    let b = normal.x * normal.y * a;
    let tangent = vec3f(1.0 + s * normal.x * normal.x * a, s * b, -s * normal.x);
    let bitangent = vec3f(b, s + normal.y * normal.y * a, -normal.y);
    let r = sqrt(u.x);
    let phi = SHARC_TWO_PI * u.y;
    return normalize(tangent * (r * cos(phi)) + bitangent * (r * sin(phi)) + normal * sqrt(max(1.0 - u.x, 0.0)));
}

// The slot of the cell, claimed if absent; -1 when its bucket is full. atomicCompareExchangeWeak may
// fail spuriously, so a failure that leaves the slot empty is tried once more (register D22's rule).
fn sharcInsert(key: vec2u) -> i32 {
    for (var i = 0u; i < SHARC_BUCKET; i++) {
        if (atomicLoad(&g_SharcKeys[key.x + i]) == key.y) {
            return i32(key.x + i);
        }
    }
    for (var i = 0u; i < SHARC_BUCKET; i++) {
        let slot = key.x + i;
        if (atomicLoad(&g_SharcKeys[slot]) == 0u) {
            var result = atomicCompareExchangeWeak(&g_SharcKeys[slot], 0u, key.y);
            if (!result.exchanged && result.old_value == 0u) {
                result = atomicCompareExchangeWeak(&g_SharcKeys[slot], 0u, key.y);
            }
            if (result.exchanged || result.old_value == key.y) {
                return i32(slot);
            }
        }
    }
    return -1;
}

// Adds one sample of radiance into the cell at 'position' (and, with the pyramid, its coarser cells):
// fixed point with stochastic rounding, so the sum stays unbiased.
fn sharcInsertSample(position: vec3f, normal: vec3f, radiance: vec3f, random: ptr<function, RcRandom>) {
    let u = vec3f(rcRand(random), rcRand(random), rcRand(random));
    if (!sharcFinite(radiance)) {
        return;
    }
    let quantised = vec3u(floor(clamp(radiance, vec3f(0.0), vec3f(SHARC_CLAMP)) * SHARC_SCALE + u));
    let level = sharcLevel(position);
    let bin = sharcNormalBin(normal);
    for (var j = 0u; j <= SHARC_PYRAMID_LEVELS; j++) {
        let slot = sharcInsert(sharcKey(position, bin, min(level + j, 31u)));
        if (slot < 0) {
            sharcCount(${SHARC_COUNTER.droppedInserts}u, 1u);
            continue;
        }
        let word = sharcSlotWord(u32(slot));
        if (quantised.x > 0u) { atomicAdd(&g_SharcAccum[word + 0u], quantised.x); }
        if (quantised.y > 0u) { atomicAdd(&g_SharcAccum[word + 1u], quantised.y); }
        if (quantised.z > 0u) { atomicAdd(&g_SharcAccum[word + 2u], quantised.z); }
        atomicAdd(&g_SharcAccum[word + 3u], 1u);
    }
}

fn sharcProbeSurface(probeRay: WorldCacheProbeRay) -> GiSurface {
    let ray = GiRay(probeRay.origin, 0.0, worldCacheProbeRayDirection(probeRay), max(probeRay.hitT, 0.0));
    let hit = GiHit(probeRay.hitT, probeRay.hitInstance, probeRay.hitTriangle, probeRay.hitBarycentrics, probeRay.hitFrontFace != 0u);
    return giSurfaceAt(ray, hit);
}

@compute @workgroup_size(${SHARC_WORKGROUP_SIZE})
fn ${SHARC_ENTRY_POINT.update}(
    @builtin(workgroup_id) workgroup: vec3u,
    @builtin(num_workgroups) workgroups: vec3u,
    @builtin(local_invocation_index) localIndex: u32,
) {
    let did = sharcWorkgroupIndex(workgroup, workgroups) * ${SHARC_WORKGROUP_SIZE}u + localIndex;
    if (did >= g_SharcProbeRays.count) {
        return;
    }
    g_SharcRaySamples[2u * did] = bitcast<vec4u>(vec4f(0.0, 0.0, 0.0, SHARC_RAY_NONE));
    let probeRay = g_SharcProbeRays.rays[did];
    // A miss is the screen cache's (the sky), and so is a ray it lit itself.
    if (probeRay.hitT < 0.0 || (probeRay.flags & WORLD_CACHE_PROBE_RAY_BYPASS) != 0u) {
        return;
    }
    let surface = sharcProbeSurface(probeRay);
    // Emission goes straight to the ray, never into a cell (a cell would spread it over its
    // whole size, enlarging the light: the reason Capsaicin's GenerateReservoirs bypasses
    // emissive hits). Unlike Capsaicin's, the light the surface reflects is added too.
    if (g_Sharc.useDirectLighting != 0u && surface.frontFace && dot(surface.emission, surface.emission) > 0.0) {
        sharcAddToProbeRay(did, surface.emission);
    }

    // Whole probes (a workgroup holds one probe's 64 rays) update or don't, so the skipped ones trace
    // nothing and the others keep their rays together.
    let probe = did / ${SHARC_WORKGROUP_SIZE}u;
    let updates = f32(rcPcgHash(probe ^ rcPcgHash(g_Sharc.frame ^ SHARC_SEED_SALT)) >> 8u) * (1.0 / 16777216.0) < g_Sharc.updateFraction;
    if (!updates) {
        // Read the cache only (a ray shorter than its cell too; a cell with no samples reads black).
        g_SharcRaySamples[2u * did] = bitcast<vec4u>(vec4f(0.0, 0.0, 0.0, SHARC_RAY_LOOKUP));
        g_SharcRaySamples[2u * did + 1u] = vec4u(bitcast<vec3u>(surface.position), pack4x8snorm(vec4f(surface.normal, 0.0)));
        return;
    }

    var random = rcMakeRandom(did ^ SHARC_SEED_SALT, g_Sharc.frame);
    let bounces = min(g_Sharc.bounceRays, SHARC_MAX_BOUNCE_RAYS);

    // The update path. 'vertices' are the hits whose estimate goes into the cache (the probe ray's
    // hit first); the last bounce's hit is the tail, whose light comes from the cache or its own
    // light sample.
    var vertices: array<GiSurface, ${SHARC_MAX_BOUNCE_RAYS}>;
    var direct: array<vec3f, ${SHARC_MAX_BOUNCE_RAYS}>;
    // The emission of vertex k + 1 seen from vertex k (front faces only).
    var emissionAhead: array<vec3f, ${SHARC_MAX_BOUNCE_RAYS}>;
    vertices[0] = surface;
    direct[0] = sharcDirect(surface, &random);
    var vertexCount = 1u;
    // The radiance arriving at the last vertex along its bounce ray (0 without one).
    var tail = vec3f(0.0);
    for (var k = 0u; k < SHARC_MAX_BOUNCE_RAYS; k++) {
        if (k >= bounces) {
            break;
        }
        let start = vertices[k];
        let direction = sharcCosineDirection(start.normal, rcRand2(&random));
        let ray = GiRay(giShadeOffsetRayOrigin(start.position, giShadeRaySide(start)), 0.0, direction, RC_MAX_HIT_DISTANCE);
        let hit = giTraceClosest(ray);
        if (hit.t < 0.0) {
            if (g_Sharc.skyOnMiss != 0u) {
                tail = giSkyRadiance(direction);
            }
            sharcCount(${SHARC_COUNTER.tailMisses}u, 1u);
            break;
        }
        let next = giSurfaceAt(ray, hit);
        let emission = select(vec3f(0.0), next.emission, next.frontFace);
        if (k + 1u < bounces) {
            // Another vertex: its own light, and the path goes on from it.
            vertices[k + 1u] = next;
            direct[k + 1u] = sharcDirect(next, &random);
            emissionAhead[k] = emission;
            vertexCount = k + 2u;
            continue;
        }
        // The tail: the cache's value where it has enough samples and the segment is longer than the
        // cell (SHaRC's rule against reading a cell from too close), else the hit's own light sample.
        var reflected = vec3f(0.0);
        var fromCache = false;
        if (g_Sharc.cacheTail != 0u && hit.t >= sharcCellSize(sharcLevel(next.position))) {
            let read = sharcRead(next.position, next.normal);
            if (read.weight >= g_Sharc.minSamples) {
                reflected = read.radiance;
                fromCache = true;
            }
        }
        if (fromCache) {
            sharcCount(${SHARC_COUNTER.tailFromCache}u, 1u);
        } else {
            reflected = sharcDirect(next, &random);
            sharcCount(${SHARC_COUNTER.tailFromLight}u, 1u);
        }
        tail = emission + reflected;
    }

    // Carry the light back along the path: S_k = direct_k + albedo_k x (what arrives at k).
    var estimates: array<vec3f, ${SHARC_MAX_BOUNCE_RAYS}>;
    var arriving = tail;
    for (var j = 0u; j < SHARC_MAX_BOUNCE_RAYS; j++) {
        if (j >= vertexCount) {
            break;
        }
        let k = vertexCount - 1u - j;
        let estimate = direct[k] + vertices[k].albedo * arriving;
        estimates[k] = estimate;
        if (k > 0u) {
            arriving = emissionAhead[k - 1u] + estimate;
        }
    }

    // Each vertex's estimate into its cell. Past the first, only when the path ends in the cache:
    // without it a later vertex's estimate has fewer bounces than its cell stands for.
    for (var k = 0u; k < SHARC_MAX_BOUNCE_RAYS; k++) {
        if (k >= vertexCount) {
            break;
        }
        if (k == 0u || g_Sharc.cacheTail != 0u) {
            sharcInsertSample(vertices[k].position, vertices[k].normal, estimates[k], &random);
        }
    }

    // What the lookup needs: the ray's own estimate, whether it reads it (a ray shorter than its cell
    // would read light from beyond what it can see: SHaRC's segment rule, GI-1.2's bypass), and the
    // cell it went into.
    let own = probeRay.hitT < sharcCellSize(sharcLevel(surface.position));
    let estimate = select(vec3f(0.0), estimates[0], sharcFinite(estimates[0]));
    g_SharcRaySamples[2u * did] = bitcast<vec4u>(vec4f(estimate, select(SHARC_RAY_LOOKUP, SHARC_RAY_OWN, own)));
    g_SharcRaySamples[2u * did + 1u] = vec4u(bitcast<vec3u>(surface.position), pack4x8snorm(vec4f(surface.normal, 0.0)));
}
`;
}

/**
 * `SharcResolve`: one thread per slot. A slot with new samples moves its
 * running mean (weight `w_old · decay + n`), stamps its age and clears its
 * accumulators; one with none for more than `staleFrames` frames is evicted.
 */
export function sharcResolveWgsl(options: SharcCompileOptions): string {
    return /* wgsl */ `
${sharcCommonWgsl(SHARC_CACHE_GROUP, options, 'read_write')}
${ACCUM_WGSL}

@compute @workgroup_size(${SHARC_RESOLVE_WORKGROUP_SIZE})
fn ${SHARC_ENTRY_POINT.resolve}(
    @builtin(workgroup_id) workgroup: vec3u,
    @builtin(num_workgroups) workgroups: vec3u,
    @builtin(local_invocation_index) localIndex: u32,
) {
    let slot = (workgroup.y * workgroups.x + workgroup.x) * ${SHARC_RESOLVE_WORKGROUP_SIZE}u + localIndex;
    if (slot > g_Sharc.mask) {
        return;
    }
    if (atomicLoad(&g_SharcKeys[slot]) == 0u) {
        return;
    }
    let word = sharcSlotWord(slot);
    let count = atomicLoad(&g_SharcAccum[word + 3u]);
    if (count > 0u) {
        let sum = vec3f(f32(atomicLoad(&g_SharcAccum[word + 0u])), f32(atomicLoad(&g_SharcAccum[word + 1u])), f32(atomicLoad(&g_SharcAccum[word + 2u]))) / SHARC_SCALE;
        let previous = g_SharcResolved[slot];
        let history = previous.w * g_Sharc.decay;
        let weight = history + f32(count);
        g_SharcResolved[slot] = vec4f((previous.xyz * history + sum) / weight, weight);
        atomicStore(&g_SharcAccum[word + 0u], 0u);
        atomicStore(&g_SharcAccum[word + 1u], 0u);
        atomicStore(&g_SharcAccum[word + 2u], 0u);
        atomicStore(&g_SharcAccum[word + 3u], 0u);
        atomicStore(&g_SharcAccum[word + 4u], g_Sharc.frame);
    } else if (g_Sharc.frame - atomicLoad(&g_SharcAccum[word + 4u]) > g_Sharc.staleFrames) {
        atomicStore(&g_SharcKeys[slot], 0u);
        g_SharcResolved[slot] = vec4f(0.0);
    }
}
`;
}

/**
 * `SharcLookup`: one thread per probe ray, after the resolve. A ray the
 * update marked adds its own estimate or its cell's resolved radiance into
 * its radiance slot (`sharcRead`: the finest level with `minSamples`).
 */
export function sharcLookupWgsl(options: SharcCompileOptions): string {
    return /* wgsl */ `
${CAPSAICIN_WGSL}
${sharcCommonWgsl(SHARC_CACHE_GROUP, options, 'read_write')}
${ACCUM_WGSL}
${RAY_GROUP_WGSL}

var<workgroup> wg_SharcOwn: atomic<u32>;
var<workgroup> wg_SharcCoarse: atomic<u32>;
var<workgroup> wg_SharcThin: atomic<u32>;
var<workgroup> wg_SharcCache: atomic<u32>;

@compute @workgroup_size(${SHARC_WORKGROUP_SIZE})
fn ${SHARC_ENTRY_POINT.lookup}(
    @builtin(workgroup_id) workgroup: vec3u,
    @builtin(num_workgroups) workgroups: vec3u,
    @builtin(local_invocation_index) localIndex: u32,
) {
    if (localIndex == 0u) {
        atomicStore(&wg_SharcOwn, 0u);
        atomicStore(&wg_SharcCoarse, 0u);
        atomicStore(&wg_SharcThin, 0u);
        atomicStore(&wg_SharcCache, 0u);
    }
    workgroupBarrier();
    let did = sharcWorkgroupIndex(workgroup, workgroups) * ${SHARC_WORKGROUP_SIZE}u + localIndex;
    if (did < g_SharcProbeRays.count) {
        let kept = bitcast<vec4f>(g_SharcRaySamples[2u * did]);
        if (kept.w == SHARC_RAY_OWN) {
            sharcAddToProbeRay(did, kept.xyz);
            atomicAdd(&wg_SharcOwn, 1u);
        } else if (kept.w == SHARC_RAY_LOOKUP) {
            let inserted = g_SharcRaySamples[2u * did + 1u];
            let normal = unpack4x8snorm(inserted.w).xyz;
            let read = sharcRead(bitcast<vec3f>(inserted.xyz), normal);
            // The ray's own estimate went into its cell this frame, so a cell is always there unless
            // its bucket was full; then the ray reads its own estimate.
            sharcAddToProbeRay(did, select(kept.xyz, read.radiance, read.weight > 0.0));
            atomicAdd(&wg_SharcCache, 1u);
            if (read.levelOffset > 0u) {
                atomicAdd(&wg_SharcCoarse, 1u);
            }
            if (read.weight < g_Sharc.minSamples) {
                atomicAdd(&wg_SharcThin, 1u);
            }
        }
    }
    workgroupBarrier();
    if (localIndex == 0u) {
        sharcCount(${SHARC_COUNTER.ownReads}u, atomicLoad(&wg_SharcOwn));
        sharcCount(${SHARC_COUNTER.coarseReads}u, atomicLoad(&wg_SharcCoarse));
        sharcCount(${SHARC_COUNTER.thinReads}u, atomicLoad(&wg_SharcThin));
        sharcCount(${SHARC_COUNTER.cacheReads}u, atomicLoad(&wg_SharcCache));
    }
}
`;
}

/**
 * The slot's `worldCacheLookup` (world-cache.ts, `WorldCacheLookup`) over the
 * hash cache, its bindings in `group`: the radiance a cell holds for
 * `position` on the side `normal` faces, and a weight of 1 where some level
 * has samples (0 where the cache can't answer). `direction` is unused: a cell
 * is view-independent (Lambert).
 */
export function sharcLookupFunctionWgsl(group: number, options: SharcCompileOptions): string {
    return /* wgsl */ `
${sharcCommonWgsl(group, options, 'read')}

fn worldCacheLookup(position: vec3f, normal: vec3f, direction: vec3f) -> vec4f {
    let read = sharcRead(position, normal);
    return vec4f(read.radiance, select(0.0, 1.0, read.weight > 0.0));
}
`;
}
