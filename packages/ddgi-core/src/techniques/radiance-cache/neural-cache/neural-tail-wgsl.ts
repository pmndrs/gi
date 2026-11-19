/**
 * The neural tail's WGSL (P6.4, register E11): the kernels that gather its
 * records, and the hooks `@pathtracer/mlp`'s kernels call to read them.
 *
 * Records (vec4s, f32):
 *
 * - **Training set** (`NEURAL_TAIL_RECORD_VEC4` = 4 a record): position
 *   (w 1); the outgoing direction's and the shading normal's angles
 *   (`ntAngles`); albedo and roughness (1: diffuse, register D8); the target,
 *   the radiance the surface reflects from indirect light (w 1). Two kinds go
 *   in, appended through one counter:
 *   - *cell records* (`NeuralTail_GatherCellRecords`, inside the hash grid's
 *     frame after `UpdateTiles`): a random share of the first-bounce entries,
 *     each its hit's surface and its cell's filtered indirect value
 *     (`HashGridCache_FilteredRadianceIndirect`, as `ResolveCells` reads it),
 *     where the cell has indirect samples;
 *   - *screen records* (`NeuralTail_GatherScreenRecords`, after the screen's
 *     irradiance): one pixel in each `screen_stride`² block, its surface found
 *     by a primary ray through the pixel, the target albedo / π × E with E the
 *     screen probes' irradiance there. These teach the network the surfaces
 *     the probes stand on, which probe rays may never hit (the furnace's
 *     floor: every probe is on it and every probe ray goes up).
 * - **Queries** (`NEURAL_TAIL_QUERY_VEC4` = 3 a query,
 *   `NeuralTail_GatherQueries`): each second-bounce entry's hit, the first
 *   three vec4s of a record, w of the first 0 for a miss.
 *
 * Inference stores each query's answer as four halves (rgb, 1 where valid)
 * in the hash grid's per-ray lists (`neural_tail_buffer`, `NEURAL_TAIL_WORDS`
 * a second-bounce entry), where `UpdateMultibounceCells` adds it to the
 * second hit's direct light when `use_neural_tail` is set.
 *
 * The encoding is the pathtracer's NRC layout (`nrcEncodingWgsl`, VkNRC's):
 * position normalised to the scene box (triangle waves), direction and
 * normal as one-blob angles, roughness, diffuse and specular reflectance (0
 * here), two constants. The hooks' contract is `@pathtracer/mlp`'s
 * (`wgsl.ts`): `nnTrainLoad`, `nnInferLoad`, `nnInferStore` in group 1.
 */

import { nrcEncodingWgsl } from '@pathtracer/mlp';
import { CAPSAICIN_WGSL } from '../wgsl/capsaicin.ts';
import type { HashGridCacheBufferLayout } from '../world-cache/capacity.ts';
import { HASH_GRID_PROBE_RAY_SURFACE_WGSL, hashGridPopulationHeadWgsl } from '../world-cache/hash-grid-population-wgsl.ts';

/** `GPUShaderStage.COMPUTE` (the global doesn't exist in Node, where the tests import this file). */
const SHADER_STAGE_COMPUTE = 0x4;

/** vec4s a training record takes. */
export const NEURAL_TAIL_RECORD_VEC4 = 4;
/** vec4s a query takes. */
export const NEURAL_TAIL_QUERY_VEC4 = 3;
/** Bytes of `NeuralTailParams`. */
export const NEURAL_TAIL_PARAMS_BYTES = 144;
/** Words of the counts buffer: the query count (for inference), the training set's count (for training), two spare. */
export const NEURAL_TAIL_COUNT_WORD = { queries: 0, records: 1 } as const;
export const NEURAL_TAIL_COUNTS_BYTES = 16;
/** Workgroup size of the gather kernels. */
export const NEURAL_TAIL_WORKGROUP = 64;
/** Pixels across a screen-records workgroup (8 × 8 blocks). */
export const NEURAL_TAIL_SCREEN_GROUP = 8;

/** The gather kernels' own group (group 3 beside the hash grid's two; group 1 for the screen kernel). */
export const NEURAL_TAIL_BINDING = {
    params: 0,
    /** Training set (cell and screen kernels) or queries (query kernel), `array<vec4f>`. */
    records: 1,
    counts: 2,
    /** The screen kernel only: the screen's irradiance E. */
    irradiance: 3,
} as const;

export const NEURAL_TAIL_GROUP_ENTRIES: readonly GPUBindGroupLayoutEntry[] = [
    { binding: NEURAL_TAIL_BINDING.params, visibility: SHADER_STAGE_COMPUTE, buffer: { type: 'uniform' } },
    { binding: NEURAL_TAIL_BINDING.records, visibility: SHADER_STAGE_COMPUTE, buffer: { type: 'storage' } },
    { binding: NEURAL_TAIL_BINDING.counts, visibility: SHADER_STAGE_COMPUTE, buffer: { type: 'storage' } },
];

export const NEURAL_TAIL_SCREEN_GROUP_ENTRIES: readonly GPUBindGroupLayoutEntry[] = [
    ...NEURAL_TAIL_GROUP_ENTRIES,
    { binding: NEURAL_TAIL_BINDING.irradiance, visibility: SHADER_STAGE_COMPUTE, texture: { sampleType: 'unfilterable-float' } },
];

export const NEURAL_TAIL_ENTRY_POINT = {
    gatherCellRecords: 'NeuralTail_GatherCellRecords',
    gatherQueries: 'NeuralTail_GatherQueries',
    gatherScreenRecords: 'NeuralTail_GatherScreenRecords',
} as const;

/** `NeuralTailParams`, written once a frame (see `NeuralTail.writeParams`). Word offsets. */
export const NEURAL_TAIL_PARAMS_WORD = {
    boxLo: 0,
    boxScale: 4,
    inverseViewProjection: 8,
    eye: 24,
    frameIndex: 28,
    setCapacity: 29,
    queryCapacity: 30,
    cellRecords: 31,
    tailList: 32,
    screenStride: 33,
    screenWidth: 34,
    screenHeight: 35,
} as const;

const PARAMS_STRUCT_WGSL = /* wgsl */ `
struct NeuralTailParams {
    box_lo: vec4f,                       // the scene box's min corner
    box_scale: vec4f,                    // 2 / the box's extent, per axis
    inverse_view_projection: mat4x4f,    // camera-relative (camera.ts): world = eye + (M × ndc).xyz / w
    eye: vec4f,
    frame_index: u32,
    set_capacity: u32,                   // records the training set holds
    query_capacity: u32,                 // queries the query buffer holds
    cell_records: u32,                   // about this many first-bounce cells a frame
    tail_list: u32,                      // word where neural_tail_buffer starts in the hash grid's ray lists
    screen_stride: u32,                  // one screen record per stride × stride pixels (0: none)
    screen_width: u32,
    screen_height: u32,
}
`;

/** The angles and position the network reads; shared by the gather kernels and the hooks. CPU twins in `neural-tail-cpu.ts`. */
const INPUT_HELPERS_WGSL = /* wgsl */ `
const NT_PI: f32 = 3.14159265358979;

// A unit direction as (θ / π from +y, atan2(z, x) / 2π + 1/2), each in [0, 1] (neuralTailAngles).
fn ntAngles(direction: vec3f) -> vec2f {
    let d = normalize(direction);
    return vec2f(acos(clamp(d.y, -1.0, 1.0)) / NT_PI, atan2(d.z, d.x) / (2.0 * NT_PI) + 0.5);
}
`;

/** Gather-side: appending a record to the training set. Needs `g_NeuralTail_Records` and `g_NeuralTail_Counts`. */
const APPEND_RECORD_WGSL = /* wgsl */ `
fn NeuralTail_IsFinite(value: vec3f) -> bool {
    return all((bitcast<vec3u>(value) & vec3u(0x7f800000u)) != vec3u(0x7f800000u));
}

fn NeuralTail_AppendRecord(position: vec3f, outgoing: vec3f, normal: vec3f, albedo: vec3f, target_value: vec3f) {
    if (!NeuralTail_IsFinite(target_value) || !NeuralTail_IsFinite(position)) {
        return;
    }
    let slot = atomicAdd(&g_NeuralTail_Counts[${NEURAL_TAIL_COUNT_WORD.records}u], 1u);
    if (slot >= g_NeuralTail_Params.set_capacity) {
        return;
    }
    let base = ${NEURAL_TAIL_RECORD_VEC4}u * slot;
    g_NeuralTail_Records[base + 0u] = vec4f(position, 1.0);
    g_NeuralTail_Records[base + 1u] = vec4f(ntAngles(outgoing), ntAngles(normal));
    g_NeuralTail_Records[base + 2u] = vec4f(albedo, 1.0);
    g_NeuralTail_Records[base + 3u] = vec4f(max(target_value, vec3f(0.0)), 1.0);
}
`;

function groupDeclarationsWgsl(group: number, withIrradiance: boolean): string {
    return /* wgsl */ `
${PARAMS_STRUCT_WGSL}
@group(${group}) @binding(${NEURAL_TAIL_BINDING.params}) var<uniform> g_NeuralTail_Params: NeuralTailParams;
@group(${group}) @binding(${NEURAL_TAIL_BINDING.records}) var<storage, read_write> g_NeuralTail_Records: array<vec4f>;
@group(${group}) @binding(${NEURAL_TAIL_BINDING.counts}) var<storage, read_write> g_NeuralTail_Counts: array<atomic<u32>>;
${withIrradiance ? `@group(${group}) @binding(${NEURAL_TAIL_BINDING.irradiance}) var g_NeuralTail_Irradiance: texture_2d<f32>;` : ''}
`;
}

/** The neural tail's group beside the hash grid's (groups 1 and 2). */
export const NEURAL_TAIL_HASH_GRID_GROUP = 3;

/**
 * `NeuralTail_GatherCellRecords`: over the first-bounce entries
 * (`VisibilityCountBuffer0`, dispatched indirectly), a random share of about
 * `cell_records` of them append their hit and their cell's filtered
 * indirect value, where the cell has indirect samples. Built with the hash
 * grid's population head (groups 1 and 2) and the platform's hit shading.
 */
export function gatherCellRecordsWgsl(layout: HashGridCacheBufferLayout): string {
    return /* wgsl */ `
${CAPSAICIN_WGSL}
${hashGridPopulationHeadWgsl(layout)}
${HASH_GRID_PROBE_RAY_SURFACE_WGSL}
${groupDeclarationsWgsl(NEURAL_TAIL_HASH_GRID_GROUP, false)}
${INPUT_HELPERS_WGSL}
${APPEND_RECORD_WGSL}

@compute @workgroup_size(${NEURAL_TAIL_WORKGROUP})
fn ${NEURAL_TAIL_ENTRY_POINT.gatherCellRecords}(
    @builtin(workgroup_id) workgroup: vec3u,
    @builtin(num_workgroups) workgroups: vec3u,
    @builtin(local_invocation_index) local_index: u32,
) {
    let did = HashGridCache_WorkgroupIndex(workgroup, workgroups) * ${NEURAL_TAIL_WORKGROUP}u + local_index;
    let first_bounce_count = atomicLoad(&g_HashGridCache_TileWords[HASHGRIDCACHE_VISIBILITYCOUNTBUFFER0]);
    if (did >= first_bounce_count) {
        return;
    }
    // A random share, so the set gets about cell_records of them whatever the screen.
    var random = rcMakeRandom(g_NeuralTail_Params.frame_index ^ 0x6e657572u, did);
    let keep = f32(g_NeuralTail_Params.cell_records) / f32(first_bounce_count);
    if (rcRand(&random) >= keep) {
        return;
    }
    let cell_index = HashGridCache_RayListWord(g_HashGridCache_RayLists.visibility_cell_buffer, did);
    if (cell_index == kGI1_InvalidId) {
        return;
    }
    let indirect = HashGridCache_FilteredRadianceIndirect(cell_index);
    if (indirect.w <= 0.0) {
        return; // no indirect sample yet: nothing to learn
    }
    let probe_ray = g_ProbeRays.rays[HashGridCache_RayListWord(g_HashGridCache_RayLists.visibility_query_buffer, did)];
    let surface = HashGridCache_ProbeSurface(probe_ray);
    NeuralTail_AppendRecord(surface.position, -worldCacheProbeRayDirection(probe_ray), surface.normal, surface.albedo, indirect.rgb / max(indirect.w, 1.0));
}
`;
}

/**
 * `NeuralTail_GatherQueries`: over the second-bounce entries
 * (`VisibilityCountBuffer1`, dispatched indirectly), each writes its hit as
 * a query at its own index (a miss with w 0), so inference's record r is
 * second-bounce entry r.
 */
export function gatherQueriesWgsl(layout: HashGridCacheBufferLayout): string {
    return /* wgsl */ `
${CAPSAICIN_WGSL}
${hashGridPopulationHeadWgsl(layout)}
${HASH_GRID_PROBE_RAY_SURFACE_WGSL}
${groupDeclarationsWgsl(NEURAL_TAIL_HASH_GRID_GROUP, false)}
${INPUT_HELPERS_WGSL}

@compute @workgroup_size(${NEURAL_TAIL_WORKGROUP})
fn ${NEURAL_TAIL_ENTRY_POINT.gatherQueries}(
    @builtin(workgroup_id) workgroup: vec3u,
    @builtin(num_workgroups) workgroups: vec3u,
    @builtin(local_invocation_index) local_index: u32,
) {
    let did = HashGridCache_WorkgroupIndex(workgroup, workgroups) * ${NEURAL_TAIL_WORKGROUP}u + local_index;
    if (did >= atomicLoad(&g_HashGridCache_TileWords[HASHGRIDCACHE_VISIBILITYCOUNTBUFFER1]) || did >= g_NeuralTail_Params.query_capacity) {
        return;
    }
    let base = ${NEURAL_TAIL_QUERY_VEC4}u * did;
    let ray = HashGridCache_LoadMultibounceRay(did);
    if (ray.hitT < 0.0) {
        g_NeuralTail_Records[base + 0u] = vec4f(0.0);
        g_NeuralTail_Records[base + 1u] = vec4f(0.0);
        g_NeuralTail_Records[base + 2u] = vec4f(0.0);
        return;
    }
    let surface = HashGridCache_ProbeSurface(ray);
    g_NeuralTail_Records[base + 0u] = vec4f(surface.position, 1.0);
    g_NeuralTail_Records[base + 1u] = vec4f(ntAngles(-worldCacheProbeRayDirection(ray)), ntAngles(surface.normal));
    g_NeuralTail_Records[base + 2u] = vec4f(surface.albedo, 1.0);
}
`;
}

/**
 * `NeuralTail_GatherScreenRecords`: a thread per `screen_stride`² block of
 * the screen. A random pixel in the block, a primary ray through its centre
 * (the platform's tracer, from the eye), and where it hits, a record of the
 * hit with the target albedo / π × E, E the screen's irradiance at the pixel
 * (the screen probes' indirect irradiance, `InterpolateScreenProbes`).
 * Perspective cameras only (the ray starts at the eye). Built with the
 * platform's hit shading in group 0 and this group in group 1.
 */
export function gatherScreenRecordsWgsl(): string {
    return /* wgsl */ `
${CAPSAICIN_WGSL}
${groupDeclarationsWgsl(1, true)}
${INPUT_HELPERS_WGSL}
${APPEND_RECORD_WGSL}

@compute @workgroup_size(${NEURAL_TAIL_SCREEN_GROUP}, ${NEURAL_TAIL_SCREEN_GROUP})
fn ${NEURAL_TAIL_ENTRY_POINT.gatherScreenRecords}(@builtin(global_invocation_id) id: vec3u) {
    let params = g_NeuralTail_Params;
    let stride = params.screen_stride;
    let size = vec2u(params.screen_width, params.screen_height);
    if (stride == 0u || id.x * stride >= size.x || id.y * stride >= size.y) {
        return;
    }
    var random = rcMakeRandom(params.frame_index ^ 0x73637265u, id.y * 65536u + id.x);
    let offset = min(vec2u(rcRand2(&random) * f32(stride)), vec2u(stride - 1u));
    let pixel = min(id.xy * stride + offset, size - 1u);
    let uv = (vec2f(pixel) + 0.5) / vec2f(size);
    let ndc = vec4f(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, 0.5, 1.0);
    let relative = params.inverse_view_projection * ndc;
    let direction = normalize(relative.xyz / relative.w);
    let ray = GiRay(params.eye.xyz, 0.0, direction, 3.0e38);
    let hit = giTraceClosest(ray);
    if (hit.t < 0.0) {
        return;
    }
    let surface = giSurfaceAt(ray, hit);
    let irradiance = textureLoad(g_NeuralTail_Irradiance, vec2i(pixel), 0).rgb;
    NeuralTail_AppendRecord(surface.position, -direction, surface.normal, surface.albedo, surface.albedo / NT_PI * irradiance);
}
`;
}

/** The network's inputs from a record's first three vec4s (both hooks). */
function inputsWgsl(positionBands: number): string {
    return /* wgsl */ `
${nrcEncodingWgsl(positionBands)}
${INPUT_HELPERS_WGSL}
fn ntPosition(world: vec3f) -> vec3f {
    return clamp((world - ntParams.box_lo.xyz) * ntParams.box_scale.xyz - 1.0, vec3f(-1.0), vec3f(1.0));
}
fn ntInputs(r0: vec4f, r1: vec4f, r2: vec4f) -> array<f32, NN_IN> {
    return nnEncodeNrc(ntPosition(r0.xyz), r1.xy, r1.zw, r2.w, r2.xyz, vec3f(0.0));
}
`;
}

/**
 * `@pathtracer/mlp`'s training hooks (group 1): a step's record r is a
 * random record of the frame's training set (the pathtracer's
 * `nrcTrainRecordIndex`: a hash of r and the Adam step), so every step of
 * the frame draws its own batch.
 */
export function neuralTailTrainHooks(positionBands: number): string {
    return /* wgsl */ `
${PARAMS_STRUCT_WGSL}
@group(1) @binding(0) var<storage, read> ntSet: array<vec4f>;
@group(1) @binding(1) var<storage, read> ntCounts: array<u32>;
@group(1) @binding(2) var<uniform> ntParams: NeuralTailParams;
${inputsWgsl(positionBands)}
fn ntHash(x: u32) -> u32 {
    let s = x * 747796405u + 2891336453u;
    let w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
    return (w >> 22u) ^ w;
}
fn nnTrainLoad(r: u32, x: ptr<function, array<f32, NN_IN>>) -> vec3f {
    let n = max(min(ntCounts[${NEURAL_TAIL_COUNT_WORD.records}u], ntParams.set_capacity), 1u);
    let index = ntHash(r ^ ntHash(nnState.step * 0x9e3779b9u + 17u)) % n;
    let b = ${NEURAL_TAIL_RECORD_VEC4}u * index;
    *x = ntInputs(ntSet[b], ntSet[b + 1u], ntSet[b + 2u]);
    return ntSet[b + 3u].xyz;
}
`;
}

/**
 * `@pathtracer/mlp`'s inference hooks (group 1): query r in, its answer out
 * into the hash grid's per-ray lists at `tail_list + 2r` as four halves (rgb
 * at least 0 and finite, w 1), or zeros for a miss.
 */
export function neuralTailInferHooks(positionBands: number): string {
    return /* wgsl */ `
${PARAMS_STRUCT_WGSL}
@group(1) @binding(0) var<storage, read> ntQueries: array<vec4f>;
@group(1) @binding(1) var<storage, read_write> ntListWords: array<u32>;
@group(1) @binding(2) var<uniform> ntParams: NeuralTailParams;
${inputsWgsl(positionBands)}
fn nnInferLoad(r: u32) -> array<f32, NN_IN> {
    let b = ${NEURAL_TAIL_QUERY_VEC4}u * min(r, ntParams.query_capacity - 1u);
    return ntInputs(ntQueries[b], ntQueries[b + 1u], ntQueries[b + 2u]);
}
fn nnInferStore(r: u32, y: vec3f) {
    if (r >= ntParams.query_capacity) {
        return;
    }
    let valid = ntQueries[${NEURAL_TAIL_QUERY_VEC4}u * r].w > 0.0;
    let finite = all((bitcast<vec3u>(y) & vec3u(0x7f800000u)) != vec3u(0x7f800000u));
    // Halves top out at 65,504; an answer past it would store infinity.
    let value = select(vec3f(0.0), min(max(y, vec3f(0.0)), vec3f(60000.0)), valid && finite);
    let base = ntParams.tail_list + 2u * r;
    ntListWords[base + 0u] = pack2x16float(value.xy);
    ntListWords[base + 1u] = pack2x16float(vec2f(value.z, select(0.0, 1.0, valid)));
}
`;
}
