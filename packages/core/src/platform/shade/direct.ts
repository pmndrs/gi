/**
 * Deferred direct light, `gi/direct` (contract/shade.ts `DeferredDirectLight`;
 * docs/design/m2-lights.md §2). T2.2, rebuilt from T2.0's spike.
 *
 * Why: in M1 a tracing kernel traced each hit's shadow rays itself, in a
 * kernel that also held the closest-hit walk, the bounce lookup and the
 * shading, and only the threads of a subgroup whose ray hit a lit front face
 * took part (T1.9a: shadow rays 53% of Sponza's trace, at 1/7 of the trace
 * bench's any-hit rate). Deferred, the tracing kernel writes a surface
 * record for each hit a light can reach and appends it to a
 * compacted work list; `gi/direct` then runs one thread per listed hit, so
 * every thread has a shadow ray to trace. T2.0 measured 2.2× the probes at
 * 2 ms on Sponza.
 *
 * **The ray scratch** is one storage buffer, so a tracing kernel binds the
 * ray results and the work list as one buffer (it is at 7 of the default 8):
 *
 * | Bytes                       | What                                                       |
 * | --------------------------- | ---------------------------------------------------------- |
 * | 0 .. 256                    | header (`GiRayScratch`'s fields before `cells`)            |
 * | 256 .. 256 + 8 × rays × L   | ray results, one `vec2u` each (L result layers per ray)    |
 * | then 32 × work items        | work items, four `vec2u` cells each                        |
 *
 * The header is 256 bytes so the ray results start at an offset every device
 * accepts for a storage binding (`minStorageBufferOffsetAlignment` is at most
 * 256): a technique's blend binds them alone, as `array<vec2u>`.
 *
 * **Result layers** (T2.5, light groups): a scratch may hold several results
 * per ray, `layers` of them side by side (ray `i`'s layer `l` at result
 * `i × layers + l`). The technique says which layer each light group's direct
 * light goes into (`DirectLightFrame.groupLayers`): DDGI keeps one layer per
 * light group, so a group's weight can change without new rays. The light
 * choice and the shadow rays are shared by every layer
 * (`giDirectGroupIrradiance`). With one layer (the default) every group goes
 * into it, as before.
 *
 * **The sun's layer** (T2.6b): a technique may also name a layer for the sun
 * (`DirectLightFrame.sunLayer`). The sun, the scene's first directional
 * light, then goes into that layer at a white reference instead of into its
 * group's (`giDirectSplitIrradiance` with `sunApart`), and the technique
 * weighs the layer by the sun's colour x intensity. The light choice and the
 * shadow rays are still shared.
 *
 * **A work item** (32 bytes): the surface record and the index of the ray
 * whose result gets the light.
 *
 * | Cell | x                                  | y                                     |
 * | ---- | ---------------------------------- | ------------------------------------- |
 * | 0    | position.x (f32 bits)              | position.y                            |
 * | 1    | position.z                         | ray index                             |
 * | 2    | shading normal, octahedral 16 + 16 | ray-side geometric normal, octahedral |
 * | 3    | albedo r, g (unorm16 each)         | albedo b (unorm16), scale exponent    |
 *
 * **What the record rounds, and why it is kept small.** Rounding a value to
 * nearest is not zero-mean for a material of one albedo or a wall of one
 * normal, so what the record rounds is a small *fixed* bias, not noise:
 *
 * - the position is kept exactly (the shadow ray starts there);
 * - the normals are octahedral snorm16, about 5e-5 radians of error, which
 *   moves a cosine by well under 1e-4;
 * - albedo is unorm16 per channel over a power-of-two scale (`albedo / 2^e`,
 *   e = 0 up to 15, so an albedo above 1 still fits): at most
 *   1/131,070 absolute, 0.012% of a dark 0.065 channel. `pack2x16unorm`
 *   rounds to nearest by WGSL's definition. T2.0's 10-bit albedo rounded by
 *   up to 1/2,046, 0.75% of that channel. Half floats would be worse here:
 *   `pack2x16float`'s rounding is the implementation's choice, and on the RTX
 *   3060 it truncates (T2.2, `hit-shading-deferred`: every case read about
 *   0.035% low, the mean of a truncation over half-float mantissas).
 *
 * The design's 24-byte record with 10:10:10 albedo became 28 bytes, 32 with
 * the ray index, which also keeps items on 8-byte cells (DECISIONS
 * 2026-11-06).
 *
 * **The ray's result** goes through half floats twice on this path (the
 * tracing kernel stores emission + bounce, `gi/direct` adds the direct light
 * and stores again), where an inline kernel stores once. So `gi/direct`
 * rounds its sum to a neighbouring half at random in proportion to the
 * distance (stochastic rounding) before packing: the second store adds noise
 * of under one half-float step and no bias. Packing it directly would add a
 * second truncation, −0.035% on every lit ray on the RTX.
 *
 * **Per tracing dispatch:** the tracing kernel appends; then one compute
 * pass, `<owner>/gi-direct`, runs two dispatches: `gi/direct-args` (one
 * thread) turns the count into indirect dispatch arguments in the platform's
 * own small buffer and empties the list for the next dispatch, then
 * `gi/direct` runs indirectly over the list. The arguments can't live in the
 * scratch: `gi/direct` binds the scratch as writable storage, and a buffer
 * can't be both that and the indirect buffer of the same dispatch.
 */

import { GILimitError } from '../contract/errors.ts';
import type { DeferredDirectLight, DirectLightFrame, DirectWorkList, RayScratch } from '../contract/shade.ts';
import type { ComputeKernel, KernelBuilder, WgslModule } from '../contract/wgsl.ts';

/** `GPUBufferUsage` bits; the globals don't exist in Node, where tests import this file. */
const BUFFER_USAGE = { copySrc: 0x4, copyDst: 0x8, indirect: 0x100, storage: 0x80 } as const;
const SHADER_STAGE_COMPUTE = 0x4;

/** Bytes of the scratch's header; the ray results start here (see the file comment). */
export const RAY_SCRATCH_HEADER_BYTES = 256;
/** Bytes of one ray result: radiance rg as halves, then b and the technique's half. */
export const RAY_RESULT_BYTES = 8;
/** `vec2u` cells per work item, and its bytes. */
export const WORK_ITEM_CELLS = 4;
export const WORK_ITEM_BYTES = WORK_ITEM_CELLS * 8;
/** Threads per `gi/direct` workgroup. */
export const DIRECT_WORKGROUP_SIZE = 64;

/**
 * Word offsets of the header's fields (`GiRayScratch` below; a unit test ties
 * the two together).
 */
export const RAY_SCRATCH_WORD = {
    /** Appended to by tracing kernels (atomic); emptied by `gi/direct-args`. */
    workCount: 0,
    /** The list `gi/direct` runs over this dispatch: `workCount` clamped to the capacity. */
    listed: 1,
    /** Where the work items start, in cells after the header (= the ray capacity × layers). Written at creation. */
    itemsStart: 2,
    /** Work items that fit. Written at creation. */
    itemCapacity: 3,
    /** The frame number (gi/direct's random numbers), written by `encode`. */
    frame: 4,
    /** The light groups that count, written by `encode`. */
    groupMask: 5,
    /** The frame the two counts below belong to. */
    countedFrame: 6,
    /** Hits `gi/direct` lit in `countedFrame`, over all its dispatches. */
    itemsThisFrame: 7,
    /** Hits dropped because the list was full, in `countedFrame`. */
    droppedThisFrame: 8,
    /** Results per ray (result layers). Written at creation. */
    layers: 9,
    /** For light groups 0-3, the result layer that gets the group's direct light, 8 bits each (group 0 lowest). Written by `encode`. */
    groupLayers: 10,
    /** The result layer that gets the sun apart, at a white reference; {@link NO_SUN_LAYER} when the sun is in its group's. Written by `encode`. */
    sunLayer: 11,
} as const;

/** `RAY_SCRATCH_WORD.sunLayer` when no layer holds the sun apart: the sun is in its group's layer. */
export const NO_SUN_LAYER = 0xffffffff;

/** The most result layers a scratch holds: each group's layer number is 8 bits of a header word. */
export const MAX_RAY_RESULT_LAYERS = 8;
/** Light groups of the contract (a light's `group`, 0-3). */
const LIGHT_GROUPS = 4;

/** The scratch's header and cells, as every kernel that binds a whole scratch declares it. No other module needed. */
const RAY_SCRATCH_STRUCT_WGSL = /* wgsl */ `
// ---- Ray scratch (platform/shade/direct.ts) ---------------------------------------------

// A technique's ray scratch: this header, then the ray results (cells
// 0 .. itemsStart), then the work list (cells itemsStart ..). 256 bytes of
// header, so the ray results can be bound on their own.
struct GiRayScratch {
    workCount: atomic<u32>,
    listed: u32,
    itemsStart: u32,
    itemCapacity: u32,
    frame: u32,
    groupMask: u32,
    countedFrame: u32,
    itemsThisFrame: u32,
    droppedThisFrame: u32,
    layers: u32,
    groupLayers: u32,
    sunLayer: u32,
    reserved: array<vec4u, 13>,
    cells: array<vec2u>,
}

const GI_DIRECT_ITEM_CELLS: u32 = ${WORK_ITEM_CELLS}u;
// GiRayScratch.sunLayer when the sun is a light like any other, in its group's layer.
const GI_DIRECT_NO_SUN_LAYER: u32 = ${NO_SUN_LAYER}u;
`;

/** The surface record and the ray result's packing; needs the hit-shading module (`GiSurface`). */
const SURFACE_RECORD_WGSL = /* wgsl */ `
// What gi/direct needs of a hit (see the file comment for the layout and the precision).
struct GiSurfaceRecord {
    position: vec3f,
    shadingNormal: u32,
    raySide: u32,
    albedo: vec2u,
}

// A direction as one word: octahedral, two snorm16, the encoder of the scene
// copy's normals (scene/octahedral.ts), so giSceneDecodeNormal reads it back.
// The lower hemisphere folds over the diagonals (Cigolle et al., JCGT 2014).
fn giDirectEncodeDirection(direction: vec3f) -> u32 {
    let sum = abs(direction.x) + abs(direction.y) + abs(direction.z);
    var encoded = direction.xy / max(sum, 1.0e-30);
    if (direction.z < 0.0) {
        let signs = select(vec2f(-1.0), vec2f(1.0), encoded >= vec2f(0.0));
        encoded = (vec2f(1.0) - abs(encoded.yx)) * signs;
    }
    return pack2x16snorm(encoded);
}

fn giSurfaceRecord(surface: GiSurface) -> GiSurfaceRecord {
    return GiSurfaceRecord(
        surface.position,
        giDirectEncodeDirection(surface.normal),
        giDirectEncodeDirection(giShadeRaySide(surface)),
        giDirectPackAlbedo(surface.albedo),
    );
}

// Albedo as unorm16 per channel over a scale 2^e, e the smallest power of two
// at or above the largest channel (0 up to 1): x = r, g; y = b, then e above.
fn giDirectPackAlbedo(albedo: vec3f) -> vec2u {
    let largest = max(max(albedo.r, albedo.g), max(albedo.b, 1.0));
    let exponent = u32(clamp(ceil(log2(largest)), 0.0, 15.0));
    let scaled = albedo / exp2(f32(exponent));
    return vec2u(pack2x16unorm(scaled.rg), (pack2x16unorm(vec2f(scaled.b, 0.0)) & 0xffffu) | (exponent << 16u));
}

fn giDirectUnpackAlbedo(packed: vec2u) -> vec3f {
    let scale = exp2(f32(packed.y >> 16u));
    return vec3f(unpack2x16unorm(packed.x), unpack2x16unorm(packed.y & 0xffffu).x) * scale;
}

// The surface again, with what giDirectIrradiance reads: the position, the
// shading normal and the ray side, stored as the geometric normal of a front
// face so giShadeRaySide gives it back. uv, emission, material and instance
// aren't stored: direct light doesn't use them.
fn giSurfaceFromRecord(record: GiSurfaceRecord) -> GiSurface {
    return GiSurface(
        record.position,
        giSceneDecodeNormal(record.shadingNormal),
        giSceneDecodeNormal(record.raySide),
        vec2f(0.0),
        giDirectUnpackAlbedo(record.albedo),
        vec3f(0.0),
        0u,
        0u,
        true,
    );
}

// A ray's result: radiance rg as halves, then b and the technique's own half.
fn giPackRayResult(radiance: vec3f, techniqueHalf: f32) -> vec2u {
    return vec2u(pack2x16float(radiance.rg), pack2x16float(vec2f(radiance.b, techniqueHalf)));
}

// Smallest normal half float, 2^-14. Below it half floats are evenly spaced.
const GI_DIRECT_SMALLEST_NORMAL_HALF: f32 = 6.103515625e-5;

// A value rounded to one of its two neighbouring half floats at random, in
// proportion to how close it is to each, so packing it is exact and its
// expected value is the value (see the file comment). 'random' is uniform in
// [0, 1). The same rounding as DDGI's ddgiStochasticHalf (packages/ddgi/src/wgsl/half.ts).
fn giDirectStochasticHalf(value: f32, random: f32) -> f32 {
    // Half floats have 10 fraction bits, so neighbours are 2^(e - 10) apart for
    // a value in [2^e, 2^(e+1)); frexp gives e exactly.
    let magnitude = max(abs(value), GI_DIRECT_SMALLEST_NORMAL_HALF);
    let step = exp2(f32(frexp(magnitude).exp - 1) - 10.0);
    let steps = value / step;
    let below = floor(steps);
    return (below + select(0.0, 1.0, random < steps - below)) * step;
}
`;

/** The cells of a work item, from its record and ray, and back (shared by the writers and `gi/direct`). */
const WORK_ITEM_WGSL = /* wgsl */ `
fn giDirectItemCells(record: GiSurfaceRecord, ray: u32) -> array<vec2u, ${WORK_ITEM_CELLS}> {
    return array<vec2u, ${WORK_ITEM_CELLS}>(
        bitcast<vec2u>(record.position.xy),
        vec2u(bitcast<u32>(record.position.z), ray),
        vec2u(record.shadingNormal, record.raySide),
        record.albedo,
    );
}
`;

/**
 * The tracing kernel's side, for a scratch bound as `variable` (see
 * `DeferredDirectLight` in contract/shade.ts for how a kernel uses it).
 */
export function rayScratchTraceWgsl(variable: string, workList: DirectWorkList): string {
    const append =
        workList === 'subgroups'
            ? /* wgsl */ `
// Appends one hit per thread that asks, with one atomic per subgroup: each
// thread's place from subgroupExclusiveAdd, the subgroup's base from one
// atomicAdd by its first active thread. A 32-wide subgroup with 12 hits makes
// 1 atomic instead of 12. Every thread of the workgroup must call it, in
// uniform control flow.
fn giDeferDirect(deferred: bool, record: GiSurfaceRecord, ray: u32) {
    let wanted = select(0u, 1u, deferred);
    let place = subgroupExclusiveAdd(wanted);
    let subgroupTotal = subgroupAdd(wanted);
    var base = 0u;
    // subgroupElect picks the lowest active thread, the one subgroupBroadcastFirst
    // reads, so every thread gets the base its subgroup reserved.
    if (subgroupElect() && subgroupTotal > 0u) {
        base = atomicAdd(&${variable}.workCount, subgroupTotal);
    }
    base = subgroupBroadcastFirst(base);
    if (deferred) {
        giDirectStoreItem(base + place, record, ray);
    }
}
`
            : /* wgsl */ `
// Appends one hit per thread that asks, one atomic per hit: the declared
// fallback for devices without subgroups. Every thread of the workgroup
// calls it, as with the subgroup version.
fn giDeferDirect(deferred: bool, record: GiSurfaceRecord, ray: u32) {
    if (deferred) {
        giDirectStoreItem(atomicAdd(&${variable}.workCount, 1u), record, ray);
    }
}
`;
    return /* wgsl */ `${workList === 'subgroups' ? 'enable subgroups;' : ''}
${RAY_SCRATCH_STRUCT_WGSL}
${SURFACE_RECORD_WGSL}
${WORK_ITEM_WGSL}

// Ray 'ray''s result in layer 'layer' (the scratch's layers sit side by side per ray).
fn giStoreRayLayerResult(ray: u32, layer: u32, radiance: vec3f, techniqueHalf: f32) {
    ${variable}.cells[ray * ${variable}.layers + layer] = giPackRayResult(radiance, techniqueHalf);
}

// Layer 0, the only one of a one-layer scratch.
fn giStoreRayResult(ray: u32, radiance: vec3f, techniqueHalf: f32) {
    giStoreRayLayerResult(ray, 0u, radiance, techniqueHalf);
}

// Does this hit go on the work list? Only if a light of this frame's groups
// can reach it; the sun weighed in white when it has a layer of its own, so
// its layer keeps updating while the sun's intensity is 0.
fn giNeedsDirect(surface: GiSurface) -> bool {
    return giDirectReachesSplit(surface, ${variable}.groupMask, ${variable}.sunLayer != GI_DIRECT_NO_SUN_LAYER);
}

// A hit past the capacity is counted (gi/direct-args reports it dropped) but not stored.
fn giDirectStoreItem(index: u32, record: GiSurfaceRecord, ray: u32) {
    if (index >= ${variable}.itemCapacity) {
        return;
    }
    let cells = giDirectItemCells(record, ray);
    let base = ${variable}.itemsStart + index * GI_DIRECT_ITEM_CELLS;
    for (var cell = 0u; cell < GI_DIRECT_ITEM_CELLS; cell++) {
        ${variable}.cells[base + cell] = cells[cell];
    }
}
${append}`;
}

/** `gi/direct-args`: one thread; group 1 = the scratch, the dispatch arguments. */
const DIRECT_ARGS_WGSL = /* wgsl */ `
${RAY_SCRATCH_STRUCT_WGSL}

@group(1) @binding(0) var<storage, read_write> giDirectScratch: GiRayScratch;
@group(1) @binding(1) var<storage, read_write> giDirectDispatch: array<u32, 3>;

override giDirectMaxGroupsPerRow: u32 = 65535u;

@compute @workgroup_size(1)
fn giDirectArgs() {
    // Empty the list for the next tracing dispatch while reading it.
    let appended = atomicExchange(&giDirectScratch.workCount, 0u);
    let listed = min(appended, giDirectScratch.itemCapacity);
    giDirectScratch.listed = listed;
    if (giDirectScratch.countedFrame != giDirectScratch.frame) {
        giDirectScratch.countedFrame = giDirectScratch.frame;
        giDirectScratch.itemsThisFrame = 0u;
        giDirectScratch.droppedThisFrame = 0u;
    }
    giDirectScratch.itemsThisFrame += listed;
    giDirectScratch.droppedThisFrame += appended - listed;

    // Rows of up to giDirectMaxGroupsPerRow workgroups along x, then y.
    let groups = (listed + ${DIRECT_WORKGROUP_SIZE - 1}u) / ${DIRECT_WORKGROUP_SIZE}u;
    let perRow = min(groups, giDirectMaxGroupsPerRow);
    giDirectDispatch[0] = perRow;
    giDirectDispatch[1] = select(0u, (groups + perRow - 1u) / max(perRow, 1u), groups > 0u);
    giDirectDispatch[2] = 1u;
}
`;

/**
 * `gi/direct`: one thread per listed hit. Group 0 is the platform's (scene
 * copy, rays, hit shading); group 1 the scratch. Storage buffers: tracer 3 +
 * scene copy 2 + scratch 1 = 6 (T2.3's light grid makes 7).
 */
const DIRECT_WGSL = /* wgsl */ `
${RAY_SCRATCH_STRUCT_WGSL}
${SURFACE_RECORD_WGSL}

@group(1) @binding(0) var<storage, read_write> giDirectScratch: GiRayScratch;

@compute @workgroup_size(${DIRECT_WORKGROUP_SIZE})
fn giDirect(
    @builtin(workgroup_id) group: vec3u,
    @builtin(num_workgroups) groups: vec3u,
    @builtin(local_invocation_index) thread: u32,
) {
    let index = (group.y * groups.x + group.x) * ${DIRECT_WORKGROUP_SIZE}u + thread;
    if (index >= giDirectScratch.listed) {
        return;
    }
    let base = giDirectScratch.itemsStart + index * GI_DIRECT_ITEM_CELLS;
    let position = giDirectScratch.cells[base];
    let positionZAndRay = giDirectScratch.cells[base + 1u];
    let normals = giDirectScratch.cells[base + 2u];
    let record = GiSurfaceRecord(
        vec3f(bitcast<vec2f>(position), bitcast<f32>(positionZAndRay.x)),
        normals.x,
        normals.y,
        giDirectScratch.cells[base + 3u],
    );
    let ray = positionZAndRay.y;
    let surface = giSurfaceFromRecord(record);

    // The same light selection and shadow rays as a kernel that calls
    // giDirectIrradiance inline (its four groups summed): one code path.
    // With a sun layer, the sun apart at its white reference.
    let random = giDirectRandom(ray, giDirectScratch.frame);
    let sunLayer = giDirectScratch.sunLayer;
    let byGroup = giDirectSplitIrradiance(surface, giDirectScratch.groupMask, sunLayer != GI_DIRECT_NO_SUN_LAYER, random);
    // Three more numbers per layer for the rounding, from the light's random numbers.
    let roundingSeed = giShadeHash(bitcast<u32>(random.x) ^ giShadeHash(bitcast<u32>(random.y) ^ 0x85ebca6bu));
    let layers = giDirectScratch.layers;
    let groupLayers = giDirectScratch.groupLayers;
    for (var layer = 0u; layer < layers; layer++) {
        // The groups whose direct light goes into this layer (all of them with
        // one layer), and the sun apart if this is its layer.
        let direct =
            select(vec3f(0.0), byGroup.group0, giDirectGroupLayer(groupLayers, 0u) == layer) +
            select(vec3f(0.0), byGroup.group1, giDirectGroupLayer(groupLayers, 1u) == layer) +
            select(vec3f(0.0), byGroup.group2, giDirectGroupLayer(groupLayers, 2u) == layer) +
            select(vec3f(0.0), byGroup.group3, giDirectGroupLayer(groupLayers, 3u) == layer) +
            select(vec3f(0.0), byGroup.sun, sunLayer == layer);
        if (all(direct == vec3f(0.0))) {
            // Every light of the layer blocked (or none in it): the result already holds the rest of the ray's light.
            continue;
        }
        // Add albedo / pi × direct to what the tracing kernel stored, rounded to
        // halves at random so this second store adds no bias; the technique's half stays.
        let cell = ray * layers + layer;
        let stored = giDirectScratch.cells[cell];
        let blueAndTechnique = unpack2x16float(stored.y);
        let radiance = vec3f(unpack2x16float(stored.x), blueAndTechnique.x) + surface.albedo * GI_SHADE_INV_PI * direct;
        // Layer 0 keeps the seed it had before layers existed; the others get their own.
        let layerSeed = roundingSeed ^ (layer * 0x9e3779b9u);
        let rounding = vec3f(
            f32(layerSeed >> 8u),
            f32(giShadeHash(layerSeed + 1u) >> 8u),
            f32(giShadeHash(layerSeed + 2u) >> 8u),
        ) * (1.0 / 16777216.0);
        let rounded = vec3f(
            giDirectStochasticHalf(radiance.r, rounding.x),
            giDirectStochasticHalf(radiance.g, rounding.y),
            giDirectStochasticHalf(radiance.b, rounding.z),
        );
        giDirectScratch.cells[cell] = giPackRayResult(rounded, blueAndTechnique.y);
    }
}

// The result layer light group 'group' goes into (RAY_SCRATCH_WORD.groupLayers: 8 bits a group).
fn giDirectGroupLayer(groupLayers: u32, group: u32) -> u32 {
    return (groupLayers >> (8u * group)) & 0xffu;
}
`;

/** `gi/direct-args` group 1. */
const ARGS_GROUP: GPUBindGroupLayoutDescriptor = {
    label: 'gi/direct-args group 1',
    entries: [
        { binding: 0, visibility: SHADER_STAGE_COMPUTE, buffer: { type: 'storage' } },
        { binding: 1, visibility: SHADER_STAGE_COMPUTE, buffer: { type: 'storage' } },
    ],
};

/** `gi/direct` group 1. */
const DIRECT_GROUP: GPUBindGroupLayoutDescriptor = {
    label: 'gi/direct group 1',
    entries: [{ binding: 0, visibility: SHADER_STAGE_COMPUTE, buffer: { type: 'storage' } }],
};

/** Bytes of a scratch for `rays` rays of `layers` results each and `workItems` items. */
export function rayScratchBytes(rays: number, workItems: number, layers = 1): number {
    return RAY_SCRATCH_HEADER_BYTES + rays * layers * RAY_RESULT_BYTES + workItems * WORK_ITEM_BYTES;
}

/** The header word of `sunLayer` (RAY_SCRATCH_WORD.sunLayer): the layer, or NO_SUN_LAYER when the sun stays in its group's. */
export function packSunLayer(sunLayer: number | undefined, layers: number): number {
    if (sunLayer === undefined) return NO_SUN_LAYER;
    if (!Number.isInteger(sunLayer) || sunLayer < 0 || sunLayer >= layers) {
        throw new RangeError(`@pmndrs/gi: the sun goes into result layer ${sunLayer}, but the ray scratch has ${layers} layer(s)`);
    }
    return sunLayer;
}

/** The header word of `groupLayers` (RAY_SCRATCH_WORD.groupLayers): 8 bits a group, every group in layer 0 by default. */
export function packGroupLayers(groupLayers: readonly number[] | undefined, layers: number): number {
    let packed = 0;
    for (let group = 0; group < LIGHT_GROUPS; group++) {
        const layer = groupLayers?.[group] ?? 0;
        if (!Number.isInteger(layer) || layer < 0 || layer >= layers) {
            throw new RangeError(`@pmndrs/gi: light group ${group} goes into result layer ${layer}, but the ray scratch has ${layers} layer(s)`);
        }
        packed |= layer << (8 * group);
    }
    return packed >>> 0;
}

interface DirectKernels {
    readonly args: ComputeKernel;
    readonly direct: ComputeKernel;
}

class PlatformRayScratch implements RayScratch {
    readonly buffer: GPUBuffer;
    readonly rayCapacity: number;
    readonly itemCapacity: number;
    readonly layers: number;
    readonly rayResults: GPUBufferBinding;
    readonly countsOffset = RAY_SCRATCH_WORD.itemsThisFrame * 4;
    readonly bytes: number;
    readonly argsGroup: GPUBindGroup;
    readonly directGroup: GPUBindGroup;
    /** The frame, group mask, group layers and sun layer last written into the header ("" before any). */
    writtenFrame = '';

    constructor(device: GPUDevice, kernels: DirectKernels, dispatch: GPUBuffer, label: string, rays: number, workItems: number, layers: number) {
        this.rayCapacity = rays;
        this.itemCapacity = workItems;
        this.layers = layers;
        this.bytes = rayScratchBytes(rays, workItems, layers);
        // COPY_SRC so a technique can copy the header's counts into its counters,
        // COPY_DST for encode's header writes.
        this.buffer = device.createBuffer({
            label,
            size: this.bytes,
            usage: BUFFER_USAGE.storage | BUFFER_USAGE.copySrc | BUFFER_USAGE.copyDst,
            mappedAtCreation: true,
        });
        // The list starts empty (new buffers are zero); where the items start and how many fit.
        const header = new Uint32Array(this.buffer.getMappedRange(0, RAY_SCRATCH_HEADER_BYTES));
        header[RAY_SCRATCH_WORD.itemsStart] = rays * layers;
        header[RAY_SCRATCH_WORD.itemCapacity] = workItems;
        header[RAY_SCRATCH_WORD.layers] = layers;
        header[RAY_SCRATCH_WORD.sunLayer] = NO_SUN_LAYER;
        // No frame is counted yet: the first frame's counts start from zero, whatever its number.
        header[RAY_SCRATCH_WORD.countedFrame] = 0xffffffff;
        this.buffer.unmap();
        this.rayResults = { buffer: this.buffer, offset: RAY_SCRATCH_HEADER_BYTES, size: Math.max(RAY_RESULT_BYTES, rays * layers * RAY_RESULT_BYTES) };
        this.argsGroup = device.createBindGroup({
            label: `${label} gi/direct-args`,
            layout: kernels.args.ownLayouts[0],
            entries: [
                { binding: 0, resource: { buffer: this.buffer } },
                { binding: 1, resource: { buffer: dispatch } },
            ],
        });
        this.directGroup = device.createBindGroup({ label: `${label} gi/direct`, layout: kernels.direct.ownLayouts[0], entries: [{ binding: 0, resource: { buffer: this.buffer } }] });
    }

    destroy(): void {
        this.buffer.destroy();
    }
}

export class PlatformDeferredDirect implements DeferredDirectLight {
    readonly defaultWorkList: DirectWorkList;

    private readonly device: GPUDevice;
    private readonly kernelBuilder: KernelBuilder;
    private readonly shadeModule: WgslModule;
    /** `gi/direct`'s indirect dispatch arguments, shared by every scratch: each encode writes them and uses them at once. */
    private readonly dispatch: GPUBuffer;
    private kernels: DirectKernels | null = null;
    private preparing: Promise<void> | null = null;
    /** The header's frame and group mask words, then (after the counts) its group layers and sun layer. */
    private readonly headerWords = new Uint32Array(2);
    private readonly layerWords = new Uint32Array(2);

    constructor(device: GPUDevice, kernels: KernelBuilder, shadeModule: WgslModule, label: string) {
        this.device = device;
        this.kernelBuilder = kernels;
        this.shadeModule = shadeModule;
        this.defaultWorkList = device.features.has('subgroups') ? 'subgroups' : 'atomics';
        this.dispatch = device.createBuffer({ label: `${label} gi/direct dispatch`, size: 16, usage: BUFFER_USAGE.storage | BUFFER_USAGE.indirect });
    }

    /** GPU bytes the platform holds for this (the dispatch arguments). */
    get memoryBytes(): number {
        return this.dispatch.size;
    }

    prepare(): Promise<void> {
        this.preparing ??= (async () => {
            const [args, direct] = await Promise.all([
                this.kernelBuilder.computeKernel({
                    label: 'gi/direct-args',
                    modules: [],
                    code: DIRECT_ARGS_WGSL,
                    entryPoint: 'giDirectArgs',
                    ownGroups: [ARGS_GROUP],
                    constants: { giDirectMaxGroupsPerRow: this.device.limits.maxComputeWorkgroupsPerDimension },
                }),
                this.kernelBuilder.computeKernel({ label: 'gi/direct', modules: [this.shadeModule], code: DIRECT_WGSL, entryPoint: 'giDirect', ownGroups: [DIRECT_GROUP] }),
            ]);
            this.kernels = { args, direct };
        })();
        return this.preparing;
    }

    traceWgsl(scratchVariable: string, workList: DirectWorkList): string {
        if (workList === 'subgroups' && !this.device.features.has('subgroups')) {
            throw new Error("@pmndrs/gi: the 'subgroups' work list needs a device with the 'subgroups' feature");
        }
        return rayScratchTraceWgsl(scratchVariable, workList);
    }

    createScratch(options: { readonly label: string; readonly rays: number; readonly workItems: number; readonly layers?: number }): RayScratch {
        const kernels = this.kernels;
        if (!kernels) throw new Error('@pmndrs/gi: await shade.direct.prepare() before creating a ray scratch');
        const rays = Math.max(1, Math.ceil(options.rays));
        const workItems = Math.max(0, Math.ceil(options.workItems));
        const layers = options.layers ?? 1;
        if (!Number.isInteger(layers) || layers < 1 || layers > MAX_RAY_RESULT_LAYERS) {
            throw new RangeError(`@pmndrs/gi: a ray scratch holds 1 to ${MAX_RAY_RESULT_LAYERS} result layers, got ${layers}`);
        }
        const bytes = rayScratchBytes(rays, workItems, layers);
        const allowed = this.device.limits.maxStorageBufferBindingSize;
        if (bytes > allowed) throw new GILimitError('maxStorageBufferBindingSize', bytes, allowed);
        return new PlatformRayScratch(this.device, kernels, this.dispatch, options.label, rays, workItems, layers);
    }

    encode(encoder: GPUCommandEncoder, scratch: RayScratch, frame: DirectLightFrame): void {
        const kernels = this.kernels;
        if (!kernels) throw new Error('@pmndrs/gi: gi/direct is not prepared');
        if (!(scratch instanceof PlatformRayScratch)) throw new Error('@pmndrs/gi: a ray scratch must come from shade.direct.createScratch');
        // The frame and groups for this frame's dispatches. A queue write lands before the
        // host submits this encoder, so the tracing kernel recorded before this call reads the
        // same group mask as gi/direct.
        const groupLayers = packGroupLayers(frame.groupLayers, scratch.layers);
        const sunLayer = packSunLayer(frame.sunLayer, scratch.layers);
        const frameKey = `${frame.frame >>> 0}/${frame.groupMask >>> 0}/${groupLayers}/${sunLayer}`;
        if (frameKey !== scratch.writtenFrame) {
            this.headerWords[0] = frame.frame >>> 0;
            this.headerWords[1] = frame.groupMask >>> 0;
            this.device.queue.writeBuffer(scratch.buffer, RAY_SCRATCH_WORD.frame * 4, this.headerWords);
            // groupLayers and sunLayer are neighbours (words 10 and 11): one write.
            this.layerWords[0] = groupLayers;
            this.layerWords[1] = sunLayer;
            this.device.queue.writeBuffer(scratch.buffer, RAY_SCRATCH_WORD.groupLayers * 4, this.layerWords);
            scratch.writtenFrame = frameKey;
        }
        // One pass, two dispatches: each dispatch is its own usage scope, so the arguments
        // written by the first are the indirect buffer of the second.
        const name = `${frame.owner}/gi-direct`;
        const pass = encoder.beginComputePass({ label: name, timestampWrites: frame.timer.computePass(name) });
        kernels.args.bind(pass);
        pass.setBindGroup(1, scratch.argsGroup);
        pass.dispatchWorkgroups(1);
        kernels.direct.bind(pass);
        pass.setBindGroup(1, scratch.directGroup);
        pass.dispatchWorkgroupsIndirect(this.dispatch, 0);
        pass.end();
    }

    dispose(): void {
        this.dispatch.destroy();
    }
}

/** For tests: the WGSL of the platform's two kernels. */
export const DIRECT_KERNELS_WGSL = { args: DIRECT_ARGS_WGSL, direct: DIRECT_WGSL } as const;
