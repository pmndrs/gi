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
 * What the slot changes (register E1-E7; the passes are listed at
 * `PATH_WGSL`): the samples come from the probe
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
/** Words per slot in the accumulator buffer: r, g, b, count (atomic), the frame it last had samples, and the cell (x, y, z, level and bin) for the change reset. */
export const SHARC_SLOT_WORDS = 9;
/** Words before the first slot in the accumulator buffer: the counters. */
export const SHARC_COUNTER_WORDS = 16;
/** Bytes of `SharcParams`. */
export const SHARC_PARAMS_BYTES = 608;
/** Changed boxes the params carry (the radiance cache's `MAX_HISTORY_CHANGE_BOXES`; more count as everything). */
export const SHARC_MAX_CHANGE_BOXES = 16;
/** Words a probe ray keeps for the lookup: its own estimate and flag, where its cell is (two vec4 of words). */
export const SHARC_RAY_SAMPLE_WORDS = 8;
/** Words of a path record (`PATH_WGSL`). */
export const SHARC_PATH_WORDS = 40;
/** Light entries a path can append: its hit, the first bounce's hit as a vertex, the tail. */
export const SHARC_LIGHTS_PER_PATH = 3;
/** Words per probe ray of the per-ray buffer: its sample, a path record, a second-bounce entry and its light entries. */
export const SHARC_WORDS_PER_RAY = SHARC_RAY_SAMPLE_WORDS + SHARC_PATH_WORDS + 1 + SHARC_LIGHTS_PER_PATH;
/**
 * RC3.4's `GI_DIRECT_SKY_LIGHT_BIT` (platform/shade/wgsl.ts on `rc3-4`): with it in the light mask,
 * `giDirectChooseLight` draws the sky too. TODO(rc3-4 merge): import `DIRECT_SKY_LIGHT_BIT` instead
 * and keep the unit test that checks they agree.
 */
export const SHARC_SKY_LIGHT_BIT = 0x10;
/** The light groups a cache hit draws its light from: all four. */
export const SHARC_LIGHT_GROUPS = 0xf;
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
    /** Update paths whose last bounce missed (the sky, unless the sky is a light). */
    tailMisses: 8,
    /** Path records this frame (updating probe rays): `SharcBounce1` and `SharcCombine` run on it. */
    paths: 9,
    /** Paths whose first bounce hit is a vertex: `SharcBounce2` runs on it. */
    secondBounces: 10,
    /** Light entries: `SharcLight` runs on it. */
    lightEntries: 11,
    /** Paths refused because the path list was full (those rays read their cell, or black). */
    refusedPaths: 12,
    /** Paths made for a ray shorter than its cell in a probe not updating this frame (the bypass needs its own estimate). */
    shortRayPaths: 13,
    /** Paths made for a ray whose cell has no samples at any level it may read, in a probe not updating this frame. */
    uncachedPaths: 14,
    /** Probe rays that read nothing (their path was refused and their cell is empty). */
    emptyReads: 15,
} as const;

/** `w` of a probe ray's kept sample (`SharcRaySample.sample`): what the lookup does with the ray. */
export const SHARC_RAY = {
    /** A miss, a ray the screen cache lit, or out of range: nothing to add. */
    none: -1,
    /** The ray reads its cell. */
    lookup: 0,
    /** The ray is shorter than its cell: it reads its own sample. */
    own: 1,
    /** The ray reads its cell, and has its own estimate to read where its cell has nothing (a refused insert). */
    lookupWithEstimate: 2,
} as const;

/** Entry points. */
export const SHARC_ENTRY_POINT = {
    begin: 'SharcBegin',
    bounce1: 'SharcBounce1',
    bounce2: 'SharcBounce2',
    light: 'SharcLight',
    combine: 'SharcCombine',
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
    /** The per-ray words: samples, path records and the lists (`SHARC_WORDS_PER_RAY` a ray). */
    rayWords: 2,
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
    { binding: SHARC_RAY_BINDING.rayWords, visibility: SHADER_STAGE_COMPUTE, buffer: { type: 'storage' } },
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
    lightMask: 14,
    updateFraction: 15,
    pathBase: 16,
    bounceListBase: 17,
    lightListBase: 18,
    pathCapacity: 19,
    changeBoxCount: 20,
    changeEverything: 21,
    changeKeepSamples: 22,
    changeReset: 23,
    /** 2 vec4f per box (min, max), from word 24. */
    changeBoxes: 24,
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
    lightMask: u32,           // the light groups a hit draws from, with the sky's bit when the sky is a light (then a miss brings nothing)
    updateFraction: f32,      // the share of probes (workgroups of 64 rays) that run the update path this frame
    pathBase: u32,            // where the path records start in g_SharcRayWords
    bounceListBase: u32,      // where the second-bounce list starts
    lightListBase: u32,       // where the light-entry list starts
    pathCapacity: u32,        // path records (and second-bounce entries) the lists hold; 3 light entries each
    changeBoxCount: u32,      // this frame's changed boxes (SceneChanges' regions)
    changeEverything: u32,    // 1: a change with no bounds (the sun, the environment, too many boxes)
    changeKeepSamples: f32,   // a changed cell keeps at most this much sample weight
    changeReset: u32,         // 1: changed cells drop their history to changeKeepSamples
    changeBoxes: array<vec4f, ${2 * SHARC_MAX_CHANGE_BOXES}>,  // min, max per box
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
// the one with the most weight (so a young cache reads its best guess, not black). Levels whose
// cell is larger than 'maxCellSize' are skipped: a ray read from a cell wider than its segment
// would take light from beyond what it can see (SHaRC's segment rule, applied at every level).
fn sharcRead(position: vec3f, normal: vec3f, maxCellSize: f32) -> SharcCellRead {
    let level = sharcLevel(position);
    let bin = sharcNormalBin(normal);
    var best = SharcCellRead(vec3f(0.0), 0.0, 0u);
    for (var j = 0u; j <= SHARC_FALLBACK_LEVELS; j++) {
        if (sharcCellSize(min(level + j, 31u)) > maxCellSize) {
            break;
        }
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

// The per-ray words. First, what each probe ray keeps for the lookup (SHARC_RAY_SAMPLE_WORDS a ray):
//   words 0-3: the ray's own estimate (rgb) and what the lookup does (SHARC_RAY_*), as f32 bits
//   words 4-7: where its cell is (xyz, f32 bits) and the normal it is binned by (pack4x8snorm)
// Kept as words, not floats: the packed normal's bits could read as a NaN, which a float store may not keep.
// Then, from g_Sharc.pathBase, bounceListBase and lightListBase, the update path's records and lists.
@group(${SHARC_RAY_GROUP}) @binding(${SHARC_RAY_BINDING.probeRays}) var<storage, read> g_SharcProbeRays: WorldCacheProbeRays;
@group(${SHARC_RAY_GROUP}) @binding(${SHARC_RAY_BINDING.radiance}) var<storage, read_write> g_SharcProbeRadiance: array<vec2u>;
@group(${SHARC_RAY_GROUP}) @binding(${SHARC_RAY_BINDING.rayWords}) var<storage, read_write> g_SharcRayWords: array<u32>;

fn sharcSetRaySample(ray: u32, estimate: vec3f, flag: f32) {
    let base = ray * ${SHARC_RAY_SAMPLE_WORDS}u;
    let bits = bitcast<vec4u>(vec4f(estimate, flag));
    g_SharcRayWords[base + 0u] = bits.x;
    g_SharcRayWords[base + 1u] = bits.y;
    g_SharcRayWords[base + 2u] = bits.z;
    g_SharcRayWords[base + 3u] = bits.w;
}

fn sharcRaySample(ray: u32) -> vec4f {
    let base = ray * ${SHARC_RAY_SAMPLE_WORDS}u;
    return bitcast<vec4f>(vec4u(g_SharcRayWords[base], g_SharcRayWords[base + 1u], g_SharcRayWords[base + 2u], g_SharcRayWords[base + 3u]));
}

fn sharcSetRayWhere(ray: u32, position: vec3f, normal: vec3f) {
    let base = ray * ${SHARC_RAY_SAMPLE_WORDS}u + 4u;
    let bits = bitcast<vec3u>(position);
    g_SharcRayWords[base + 0u] = bits.x;
    g_SharcRayWords[base + 1u] = bits.y;
    g_SharcRayWords[base + 2u] = bits.z;
    g_SharcRayWords[base + 3u] = pack4x8snorm(vec4f(normal, 0.0));
}

const SHARC_RAY_NONE: f32 = ${SHARC_RAY.none}.0;
const SHARC_RAY_LOOKUP: f32 = ${SHARC_RAY.lookup}.0;
const SHARC_RAY_OWN: f32 = ${SHARC_RAY.own}.0;
const SHARC_RAY_LOOKUP_WITH_ESTIMATE: f32 = ${SHARC_RAY.lookupWithEstimate}.0;

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
 * The update path's shared code: the path records, the surfaces they stand
 * for, the light sample. The path is split into one pass per ray type
 * (P6.1's split, as GI-1.2's world cache splits its population), so no kernel
 * carries the whole path:
 *
 * 1. `SharcBegin`, per probe ray: the hit's emission into the ray; whether its
 *    probe updates this frame; an updating hit gets a path record and a light
 *    entry for itself.
 * 2. `SharcBounce1`, per path: the first cosine-weighted bounce ray (closest
 *    hit). Its hit is a vertex (a light entry, and a second-bounce entry) or
 *    the tail (the cache's value, or a light entry).
 * 3. `SharcBounce2`, per second-bounce entry: the second bounce ray; its hit
 *    is the tail.
 * 4. `SharcLight`, per light entry: one light sample and its shadow ray (any
 *    hit).
 * 5. `SharcCombine`, per path: the light carried back along the path, each
 *    vertex's estimate into its cell, the ray's own estimate kept.
 *
 * Each list is appended through a counter (`SHARC_COUNTER`), and each pass is
 * sized on the GPU from its count through the platform's indirect arguments.
 */
const PATH_WGSL = /* wgsl */ `
const SHARC_LIGHT_GROUPS: u32 = 0xfu;
const SHARC_INV_PI: f32 = 0.318309886183791;
const SHARC_TWO_PI: f32 = 6.28318530717959;
const SHARC_CLAMP: f32 = ${SHARC_CLAMP}.0;
// Keeps the update's random numbers apart from the screen cache's (which seeds by ray index too) and each pass's apart.
const SHARC_SEED_SALT: u32 = 0xa511e9b3u;
const SHARC_SALT_BOUNCE1: u32 = 0x3c6ef372u;
const SHARC_SALT_BOUNCE2: u32 = 0xbb67ae85u;
const SHARC_SALT_LIGHT: u32 = 0x510e527fu;
const SHARC_SALT_INSERT: u32 = 0x9b05688cu;
// RC3.4's GI_DIRECT_SKY_LIGHT_BIT: the sky as a light in giDirectChooseLight.
const SHARC_SKY_LIGHT_BIT: u32 = ${SHARC_SKY_LIGHT_BIT}u;

// A path record: SHARC_PATH_WORDS words at g_Sharc.pathBase + path * SHARC_PATH_WORDS.
const SHARC_PATH_WORDS: u32 = ${SHARC_PATH_WORDS}u;
const SHARC_PATH_RAY: u32 = 0u;             // the probe ray
const SHARC_PATH_VERTICES: u32 = 1u;        // full vertices (1 or 2): the probe ray's hit, and the first bounce's hit with two bounce rays
const SHARC_PATH_HIT1: u32 = 2u;            // the first bounce's hit record (12 words)
const SHARC_PATH_HIT2: u32 = 14u;           // the second bounce's hit record (12 words)
const SHARC_PATH_DIRECT0: u32 = 26u;        // the probe ray hit's light sample (rgb, f32 bits)
const SHARC_PATH_DIRECT1: u32 = 29u;        // the first bounce hit's light sample, as a vertex
const SHARC_PATH_TAIL: u32 = 32u;           // the light arriving at the last vertex along its bounce
const SHARC_PATH_EMISSION1: u32 = 35u;      // the first bounce hit's emission (front faces)
// A light entry: path * 4 + slot.
const SHARC_SLOT_VERTEX0: u32 = 0u;          // the probe ray's hit, into DIRECT0
const SHARC_SLOT_VERTEX1: u32 = 1u;          // the first bounce's hit as a vertex, into DIRECT1
const SHARC_SLOT_TAIL1: u32 = 2u;            // the first bounce's hit as the tail, added to TAIL
const SHARC_SLOT_TAIL2: u32 = 3u;            // the second bounce's hit as the tail, added to TAIL

fn sharcPathWord(path: u32, word: u32) -> u32 {
    return g_SharcRayWords[g_Sharc.pathBase + path * SHARC_PATH_WORDS + word];
}

fn sharcSetPathWord(path: u32, word: u32, value: u32) {
    g_SharcRayWords[g_Sharc.pathBase + path * SHARC_PATH_WORDS + word] = value;
}

fn sharcPathVec3(path: u32, word: u32) -> vec3f {
    return bitcast<vec3f>(vec3u(sharcPathWord(path, word), sharcPathWord(path, word + 1u), sharcPathWord(path, word + 2u)));
}

fn sharcSetPathVec3(path: u32, word: u32, value: vec3f) {
    let bits = bitcast<vec3u>(value);
    sharcSetPathWord(path, word, bits.x);
    sharcSetPathWord(path, word + 1u, bits.y);
    sharcSetPathWord(path, word + 2u, bits.z);
}

// A bounce ray and its hit, kept in a path record: origin, direction, t, instance, triangle, barycentrics, front face.
fn sharcStoreHit(path: u32, word: u32, ray: GiRay, hit: GiHit) {
    sharcSetPathVec3(path, word, ray.origin);
    sharcSetPathVec3(path, word + 3u, ray.direction);
    sharcSetPathWord(path, word + 6u, bitcast<u32>(hit.t));
    sharcSetPathWord(path, word + 7u, hit.instance);
    sharcSetPathWord(path, word + 8u, hit.triangle);
    sharcSetPathWord(path, word + 9u, bitcast<u32>(hit.barycentrics.x));
    sharcSetPathWord(path, word + 10u, bitcast<u32>(hit.barycentrics.y));
    sharcSetPathWord(path, word + 11u, select(0u, 1u, hit.frontFace));
}

fn sharcStoredSurface(path: u32, word: u32) -> GiSurface {
    let t = bitcast<f32>(sharcPathWord(path, word + 6u));
    let ray = GiRay(sharcPathVec3(path, word), 0.0, sharcPathVec3(path, word + 3u), t);
    let barycentrics = vec2f(bitcast<f32>(sharcPathWord(path, word + 9u)), bitcast<f32>(sharcPathWord(path, word + 10u)));
    let hit = GiHit(t, sharcPathWord(path, word + 7u), sharcPathWord(path, word + 8u), barycentrics, sharcPathWord(path, word + 11u) != 0u);
    return giSurfaceAt(ray, hit);
}

fn sharcProbeSurface(probeRay: WorldCacheProbeRay) -> GiSurface {
    let ray = GiRay(probeRay.origin, 0.0, worldCacheProbeRayDirection(probeRay), max(probeRay.hitT, 0.0));
    let hit = GiHit(probeRay.hitT, probeRay.hitInstance, probeRay.hitTriangle, probeRay.hitBarycentrics, probeRay.hitFrontFace != 0u);
    return giSurfaceAt(ray, hit);
}

fn sharcFinite(value: vec3f) -> bool {
    return all((bitcast<vec3u>(value) & vec3u(0x7fffffffu)) < vec3u(0x7f800000u));
}

// With the sky in the light choice (RC3.4's sky light, register E4), a bounce ray that misses
// brings nothing: the sky reaches the hits through their light samples, and counting it on the
// miss too would count it twice.
fn sharcSkyOnMiss() -> bool {
    return (g_Sharc.lightMask & SHARC_SKY_LIGHT_BIT) == 0u;
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

fn sharcAppendLight(path: u32, slot: u32) {
    let entry = atomicAdd(&g_SharcAccum[${SHARC_COUNTER.lightEntries}u], 1u);
    g_SharcRayWords[g_Sharc.lightListBase + entry] = path * 4u + slot;
}

// A bounce ray from 'start' in a cosine-weighted direction, its closest hit, and what becomes of
// the hit: with 'asVertex', a vertex of the path (a light entry, and the next bounce); otherwise
// the tail, read from the cache where its cell has minSamples and the segment is longer than the
// cell (SHaRC's rule against reading a cell from too close), else lit by its own light entry.
// A miss is the sky (sharcSkyOnMiss) or nothing.
fn sharcBounce(path: u32, start: GiSurface, hitWord: u32, asVertex: bool, vertexSlot: u32, tailSlot: u32, u: vec2f) {
    let direction = sharcCosineDirection(start.normal, u);
    let ray = GiRay(giShadeOffsetRayOrigin(start.position, giShadeRaySide(start)), 0.0, direction, RC_MAX_HIT_DISTANCE);
    let hit = giTraceClosest(ray);
    if (hit.t < 0.0) {
        sharcSetPathVec3(path, SHARC_PATH_TAIL, select(vec3f(0.0), giSkyRadiance(direction), sharcSkyOnMiss()));
        sharcCount(${SHARC_COUNTER.tailMisses}u, 1u);
        return;
    }
    sharcStoreHit(path, hitWord, ray, hit);
    let next = giSurfaceAt(ray, hit);
    let emission = select(vec3f(0.0), next.emission, next.frontFace);
    if (asVertex) {
        sharcSetPathWord(path, SHARC_PATH_VERTICES, 2u);
        sharcSetPathVec3(path, SHARC_PATH_EMISSION1, emission);
        sharcAppendLight(path, vertexSlot);
        let entry = atomicAdd(&g_SharcAccum[${SHARC_COUNTER.secondBounces}u], 1u);
        g_SharcRayWords[g_Sharc.bounceListBase + entry] = path;
        return;
    }
    if (g_Sharc.cacheTail != 0u) {
        let read = sharcRead(next.position, next.normal, hit.t);
        if (read.weight >= g_Sharc.minSamples) {
            sharcSetPathVec3(path, SHARC_PATH_TAIL, emission + read.radiance);
            sharcCount(${SHARC_COUNTER.tailFromCache}u, 1u);
            return;
        }
    }
    // The light pass adds the tail hit's own light sample to its emission.
    sharcSetPathVec3(path, SHARC_PATH_TAIL, emission);
    sharcAppendLight(path, tailSlot);
    sharcCount(${SHARC_COUNTER.tailFromLight}u, 1u);
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
        let cellLevel = min(level + j, 31u);
        let cell = vec4u(bitcast<vec3u>(vec3i(floor(position / sharcCellSize(cellLevel)))), cellLevel);
        let slot = sharcInsert(sharcKey(position, bin, cellLevel), cell);
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

// The slot of the cell, claimed if absent; -1 when its bucket is full. atomicCompareExchangeWeak may
// fail spuriously, so a failure that leaves the slot empty is tried once more (register D22's rule).
fn sharcInsert(key: vec2u, cell: vec4u) -> i32 {
    for (var i = 0u; i < SHARC_BUCKET; i++) {
        if (atomicLoad(&g_SharcKeys[key.x + i]) == key.y) {
            return i32(key.x + i);
        }
    }
    for (var i = 0u; i < SHARC_BUCKET; i++) {
        let slot = key.x + i;
        let current = atomicLoad(&g_SharcKeys[slot]);
        // Another thread may have claimed a slot for this cell since the first scan: use it, or the
        // cell would get a second slot (the pathtracer's loop skips any taken slot; register E1).
        if (current == key.y) {
            return i32(slot);
        }
        if (current == 0u) {
            // The weak exchange may fail spuriously (leaving the slot 0): retry, bounded, so two
            // spurious failures in a row can't send this cell on to a second slot.
            var result = atomicCompareExchangeWeak(&g_SharcKeys[slot], 0u, key.y);
            for (var attempt = 0u; attempt < 8u; attempt++) {
                if (result.exchanged || result.old_value != 0u) {
                    break;
                }
                result = atomicCompareExchangeWeak(&g_SharcKeys[slot], 0u, key.y);
            }
            if (result.exchanged) {
                // The cell's place, for the change reset in SharcResolve.
                let word = sharcSlotWord(slot);
                atomicStore(&g_SharcAccum[word + 5u], cell.x);
                atomicStore(&g_SharcAccum[word + 6u], cell.y);
                atomicStore(&g_SharcAccum[word + 7u], cell.z);
                atomicStore(&g_SharcAccum[word + 8u], cell.w);
                return i32(slot);
            }
            if (result.old_value == key.y) {
                return i32(slot);
            }
        }
    }
    return -1;
}
`;

/** The head of every update pass: Capsaicin's helpers, the cache, its accumulators, the rays and the path code. */
function updateHeadWgsl(options: SharcCompileOptions): string {
    return `${CAPSAICIN_WGSL}\n${sharcCommonWgsl(SHARC_CACHE_GROUP, options, 'read_write')}\n${ACCUM_WGSL}\n${RAY_GROUP_WGSL}\n${PATH_WGSL}`;
}

/** The entry point's header: one thread per item of a 2D-split indirect dispatch. */
function perItemEntry(entryPoint: string): string {
    return /* wgsl */ `
@compute @workgroup_size(${SHARC_WORKGROUP_SIZE})
fn ${entryPoint}(
    @builtin(workgroup_id) workgroup: vec3u,
    @builtin(num_workgroups) workgroups: vec3u,
    @builtin(local_invocation_index) localIndex: u32,
) {
    let did = sharcWorkgroupIndex(workgroup, workgroups) * ${SHARC_WORKGROUP_SIZE}u + localIndex;`;
}

/**
 * `SharcBegin`: one thread per probe ray. A ray that hit (and that the screen
 * cache didn't light itself) adds its hit's emission to its radiance (front
 * faces, `useDirectLighting`) and keeps where its cell is for the lookup.
 * Whole probes (a workgroup is one probe's 64 rays) update this frame with
 * probability `updateFraction`; an updating ray gets a path record and a
 * light entry for its hit. Emission goes straight to the ray, never into a
 * cell (a cell would spread it over its size: Capsaicin's reason to bypass
 * emissive hits); unlike Capsaicin's, the light the surface reflects is added
 * too (register E5).
 */
export function sharcBeginWgsl(options: SharcCompileOptions): string {
    return /* wgsl */ `
${updateHeadWgsl(options)}
${perItemEntry(SHARC_ENTRY_POINT.begin)}
    if (did >= g_SharcProbeRays.count) {
        return;
    }
    sharcSetRaySample(did, vec3f(0.0), SHARC_RAY_NONE);
    let probeRay = g_SharcProbeRays.rays[did];
    // A miss is the screen cache's (the sky), and so is a ray it lit itself.
    if (probeRay.hitT < 0.0 || (probeRay.flags & WORLD_CACHE_PROBE_RAY_BYPASS) != 0u) {
        return;
    }
    let surface = sharcProbeSurface(probeRay);
    if (g_Sharc.useDirectLighting != 0u && surface.frontFace && dot(surface.emission, surface.emission) > 0.0) {
        sharcAddToProbeRay(did, surface.emission);
    }
    // Until SharcCombine says otherwise (an updating ray shorter than its cell reads its own estimate),
    // the ray reads its cell; a cell with no samples yet reads black.
    sharcSetRaySample(did, vec3f(0.0), SHARC_RAY_LOOKUP);
    sharcSetRayWhere(did, surface.position, surface.normal);
    let probe = did / ${SHARC_WORKGROUP_SIZE}u;
    let selected = f32(rcPcgHash(probe ^ rcPcgHash(g_Sharc.frame ^ SHARC_SEED_SALT)) >> 8u) * (1.0 / 16777216.0) < g_Sharc.updateFraction;
    // A ray in a probe not updating this frame still needs its own estimate when it is shorter than
    // its cell (the bypass), or when no cell it may read has samples (it would read black).
    var updates = selected;
    if (!selected) {
        if (probeRay.hitT < sharcCellSize(sharcLevel(surface.position))) {
            updates = true;
            sharcCount(${SHARC_COUNTER.shortRayPaths}u, 1u);
        } else if (sharcRead(surface.position, surface.normal, probeRay.hitT).weight <= 0.0) {
            updates = true;
            sharcCount(${SHARC_COUNTER.uncachedPaths}u, 1u);
        }
    }
    if (!updates) {
        return;
    }
    let path = atomicAdd(&g_SharcAccum[${SHARC_COUNTER.paths}u], 1u);
    if (path >= g_Sharc.pathCapacity) {
        // The lists are full (sized for updateFraction plus a margin): the ray reads its cell.
        sharcCount(${SHARC_COUNTER.refusedPaths}u, 1u);
        return;
    }
    sharcSetPathWord(path, SHARC_PATH_RAY, did);
    sharcSetPathWord(path, SHARC_PATH_VERTICES, 1u);
    sharcSetPathVec3(path, SHARC_PATH_TAIL, vec3f(0.0));
    sharcAppendLight(path, SHARC_SLOT_VERTEX0);
}
`;
}

/** `SharcBounce1`: one thread per path, the first bounce ray (`bounceRays` ≥ 1). */
export function sharcBounce1Wgsl(options: SharcCompileOptions): string {
    return /* wgsl */ `
${updateHeadWgsl(options)}
${perItemEntry(SHARC_ENTRY_POINT.bounce1)}
    if (did >= min(atomicLoad(&g_SharcAccum[${SHARC_COUNTER.paths}u]), g_Sharc.pathCapacity) || g_Sharc.bounceRays == 0u) {
        return;
    }
    let ray = sharcPathWord(did, SHARC_PATH_RAY);
    var random = rcMakeRandom(ray ^ SHARC_SALT_BOUNCE1, g_Sharc.frame);
    let surface = sharcProbeSurface(g_SharcProbeRays.rays[ray]);
    sharcBounce(did, surface, SHARC_PATH_HIT1, g_Sharc.bounceRays >= 2u, SHARC_SLOT_VERTEX1, SHARC_SLOT_TAIL1, rcRand2(&random));
}
`;
}

/** `SharcBounce2`: one thread per second-bounce entry (paths whose first bounce hit is a vertex), the second bounce ray. */
export function sharcBounce2Wgsl(options: SharcCompileOptions): string {
    return /* wgsl */ `
${updateHeadWgsl(options)}
${perItemEntry(SHARC_ENTRY_POINT.bounce2)}
    if (did >= atomicLoad(&g_SharcAccum[${SHARC_COUNTER.secondBounces}u])) {
        return;
    }
    let path = g_SharcRayWords[g_Sharc.bounceListBase + did];
    let ray = sharcPathWord(path, SHARC_PATH_RAY);
    var random = rcMakeRandom(ray ^ SHARC_SALT_BOUNCE2, g_Sharc.frame);
    sharcBounce(path, sharcStoredSurface(path, SHARC_PATH_HIT1), SHARC_PATH_HIT2, false, 0u, SHARC_SLOT_TAIL2, rcRand2(&random));
}
`;
}

/**
 * `SharcLight`: one thread per light entry, one light sample at a Lambert
 * surface as the hash grid's `GenerateReservoirs` and `PopulateCells` light a
 * hit: the platform's one-light choice over `lightMask`, its weight, and a
 * shadow ray. The light reflected toward the side the surface was reached
 * from, into the path's slot.
 */
export function sharcLightWgsl(options: SharcCompileOptions): string {
    return /* wgsl */ `
${updateHeadWgsl(options)}
${perItemEntry(SHARC_ENTRY_POINT.light)}
    if (did >= atomicLoad(&g_SharcAccum[${SHARC_COUNTER.lightEntries}u])) {
        return;
    }
    let entry = g_SharcRayWords[g_Sharc.lightListBase + did];
    let path = entry >> 2u;
    let slot = entry & 3u;
    let ray = sharcPathWord(path, SHARC_PATH_RAY);
    var surface: GiSurface;
    if (slot == SHARC_SLOT_VERTEX0) {
        surface = sharcProbeSurface(g_SharcProbeRays.rays[ray]);
    } else if (slot == SHARC_SLOT_TAIL2) {
        surface = sharcStoredSurface(path, SHARC_PATH_HIT2);
    } else {
        surface = sharcStoredSurface(path, SHARC_PATH_HIT1);
    }
    var random = rcMakeRandom((ray * 4u + slot) ^ SHARC_SALT_LIGHT, g_Sharc.frame);
    let choice = giDirectChooseLight(surface, g_Sharc.lightMask, rcRand(&random));
    let sampleParams = rcRand2(&random);
    var light = vec3f(0.0);
    if (choice.light != GI_DIRECT_NO_LIGHT) {
        let lightRay = giDirectLightRay(surface, choice.light, sampleParams);
        if (!(lightRay.castsShadows && giTraceAny(lightRay.shadowRay))) {
            light = surface.albedo * SHARC_INV_PI * lightRay.irradiance * choice.weight;
        }
    }
    if (slot == SHARC_SLOT_VERTEX0) {
        sharcSetPathVec3(path, SHARC_PATH_DIRECT0, light);
    } else if (slot == SHARC_SLOT_VERTEX1) {
        sharcSetPathVec3(path, SHARC_PATH_DIRECT1, light);
    } else {
        // One tail per path, so no other thread adds to it.
        sharcSetPathVec3(path, SHARC_PATH_TAIL, sharcPathVec3(path, SHARC_PATH_TAIL) + light);
    }
}
`;
}

/**
 * `SharcCombine`: one thread per path. The light carried back along the path,
 * S_k = direct_k + albedo_k × (what arrives at k); each vertex's estimate into
 * its cell (past the first only with the cache tail: without it a later
 * vertex's estimate has fewer bounces than its cell stands for); and the
 * probe ray's own estimate, which it reads instead of its cell when it is
 * shorter than the cell (SHaRC's segment rule, GI-1.2's bypass; register E6).
 */
export function sharcCombineWgsl(options: SharcCompileOptions): string {
    return /* wgsl */ `
${updateHeadWgsl(options)}
${perItemEntry(SHARC_ENTRY_POINT.combine)}
    if (did >= min(atomicLoad(&g_SharcAccum[${SHARC_COUNTER.paths}u]), g_Sharc.pathCapacity)) {
        return;
    }
    let ray = sharcPathWord(did, SHARC_PATH_RAY);
    let probeRay = g_SharcProbeRays.rays[ray];
    let first = sharcProbeSurface(probeRay);
    var random = rcMakeRandom(ray ^ SHARC_SALT_INSERT, g_Sharc.frame);
    var arriving = sharcPathVec3(did, SHARC_PATH_TAIL);
    if (sharcPathWord(did, SHARC_PATH_VERTICES) == 2u) {
        let second = sharcStoredSurface(did, SHARC_PATH_HIT1);
        let estimate = sharcPathVec3(did, SHARC_PATH_DIRECT1) + second.albedo * arriving;
        if (g_Sharc.cacheTail != 0u) {
            sharcInsertSample(second.position, second.normal, estimate, &random);
        }
        arriving = sharcPathVec3(did, SHARC_PATH_EMISSION1) + estimate;
    }
    let estimate = sharcPathVec3(did, SHARC_PATH_DIRECT0) + first.albedo * arriving;
    sharcInsertSample(first.position, first.normal, estimate, &random);
    let own = probeRay.hitT < sharcCellSize(sharcLevel(first.position));
    sharcSetRaySample(ray, select(vec3f(0.0), estimate, sharcFinite(estimate)), select(SHARC_RAY_LOOKUP_WITH_ESTIMATE, SHARC_RAY_OWN, own));
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

// Whether this frame's changes touch the cell in slot words 'word' (its integer cell and level, kept
// at its claim): everything, or a changed box overlapping the cell's box.
fn sharcSlotChanged(word: u32) -> bool {
    if (g_Sharc.changeEverything != 0u) {
        return true;
    }
    if (g_Sharc.changeBoxCount == 0u) {
        return false;
    }
    let cell = bitcast<vec3i>(vec3u(atomicLoad(&g_SharcAccum[word + 5u]), atomicLoad(&g_SharcAccum[word + 6u]), atomicLoad(&g_SharcAccum[word + 7u])));
    let size = sharcCellSize(atomicLoad(&g_SharcAccum[word + 8u]));
    let low = vec3f(cell) * size;
    let high = low + size;
    for (var box = 0u; box < ${SHARC_MAX_CHANGE_BOXES}u; box++) {
        if (box >= g_Sharc.changeBoxCount) {
            break;
        }
        let boxLow = g_Sharc.changeBoxes[2u * box].xyz;
        let boxHigh = g_Sharc.changeBoxes[2u * box + 1u].xyz;
        if (all(low <= boxHigh) && all(high >= boxLow)) {
            return true;
        }
    }
    return false;
}

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
    // A cell the scene changed in (a moved object's or a light's box, or everything for the sun or the
    // sky) keeps at most changeKeepSamples of history, so this frame's samples, taken in the changed
    // scene, outweigh what it learned before (the cache's version of D46's reset; register E8).
    var previous = g_SharcResolved[slot];
    if (g_Sharc.changeReset != 0u && previous.w > g_Sharc.changeKeepSamples && sharcSlotChanged(word)) {
        previous.w = g_Sharc.changeKeepSamples;
        g_SharcResolved[slot] = previous;
    }
    if (count > 0u) {
        let sum = vec3f(f32(atomicLoad(&g_SharcAccum[word + 0u])), f32(atomicLoad(&g_SharcAccum[word + 1u])), f32(atomicLoad(&g_SharcAccum[word + 2u]))) / SHARC_SCALE;
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
var<workgroup> wg_SharcEmpty: atomic<u32>;

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
        atomicStore(&wg_SharcEmpty, 0u);
    }
    workgroupBarrier();
    let did = sharcWorkgroupIndex(workgroup, workgroups) * ${SHARC_WORKGROUP_SIZE}u + localIndex;
    if (did < g_SharcProbeRays.count) {
        let kept = sharcRaySample(did);
        if (kept.w == SHARC_RAY_OWN) {
            sharcAddToProbeRay(did, kept.xyz);
            atomicAdd(&wg_SharcOwn, 1u);
        } else if (kept.w == SHARC_RAY_LOOKUP || kept.w == SHARC_RAY_LOOKUP_WITH_ESTIMATE) {
            let base = did * ${SHARC_RAY_SAMPLE_WORDS}u + 4u;
            let position = bitcast<vec3f>(vec3u(g_SharcRayWords[base], g_SharcRayWords[base + 1u], g_SharcRayWords[base + 2u]));
            let normal = unpack4x8snorm(g_SharcRayWords[base + 3u]).xyz;
            let read = sharcRead(position, normal, g_SharcProbeRays.rays[did].hitT);
            // A ray with its own estimate put it into its cell this frame, so a cell is there unless its
            // bucket was full; then it reads its own estimate. A ray without one (its path was refused)
            // and no cell reads nothing, counted.
            if (read.weight > 0.0) {
                sharcAddToProbeRay(did, read.radiance);
            } else if (kept.w == SHARC_RAY_LOOKUP_WITH_ESTIMATE) {
                sharcAddToProbeRay(did, kept.xyz);
            } else {
                atomicAdd(&wg_SharcEmpty, 1u);
            }
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
        sharcCount(${SHARC_COUNTER.emptyReads}u, atomicLoad(&wg_SharcEmpty));
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
    let read = sharcRead(position, normal, 3.0e38);
    return vec4f(read.radiance, select(0.0, 1.0, read.weight > 0.0));
}
`;
}
