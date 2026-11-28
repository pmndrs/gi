/**
 * The neural tail's WGSL (P6.4, register E11): the kernels that gather its
 * records, and the hooks `@pathtracer/mlp`'s kernels call to read them.
 *
 * Records (vec4s, f32):
 *
 * - **Training set** (`NEURAL_TAIL_RECORD_VEC4` = 4 a record): position and
 *   its kind in w (`NEURAL_TAIL_RECORD_KIND`); the outgoing direction's and
 *   the shading normal's angles (`ntAngles`); albedo and roughness (1:
 *   diffuse, register D8); the target, the radiance the surface reflects
 *   from light that reached it from other surfaces (w 1). Two kinds go in,
 *   appended through one counter, both inside the hash grid's frame after
 *   `UpdateTiles`:
 *   - *cell records* (`NeuralTail_GatherCellRecords`): a random share of the
 *     first-bounce entries (about `cell_records`), each its hit's surface and
 *     its cell's filtered indirect value (`HashGridCache_FilteredRadianceIndirect`,
 *     as `ResolveCells` reads it), where the cell has indirect samples;
 *   - *screen records* (`NeuralTail_GatherScreenRecords`): a random share of
 *     the screen's `screen_stride`² blocks (about `screen_records`), one
 *     random pixel each, its surface found by a primary ray through it; one
 *     cosine-weighted ray from there, and the target albedo × the light
 *     leaving its hit as the cache holds it (`worldCacheLookup`: the cell's
 *     filtered direct plus indirect value), 0 where it misses. That is a
 *     one-sample estimate of the same quantity a cell's indirect value holds
 *     (GI-1.2's second bounce, where a miss is a sample of no light): the hits
 *     only, no sky (a cell's sky is its direct light, register D33 and RC3.4)
 *     and no emission (emitters' front faces aren't cached, register E5), so
 *     nothing is counted twice where the tail's answer is added to a second
 *     hit's direct light. These teach the network the surfaces the probes
 *     stand on, which probe rays may never hit (the furnace's floor: every
 *     probe is on it, every probe ray goes up). A ray whose hit has no cell
 *     adds no record.
 * - **Queries** (`NEURAL_TAIL_QUERY_VEC4` = 3 a query,
 *   `NeuralTail_GatherQueries`): each second-bounce entry's hit, the first
 *   three vec4s of a record, w of the first 0 for a miss. The query buffer
 *   holds `query_capacity`; an entry past it gets no answer (its list words
 *   are cleared here).
 *
 * Inference stores each query's answer as four halves (rgb, 1 where valid)
 * in the hash grid's per-ray lists (`neural_tail_buffer`, `NEURAL_TAIL_WORDS`
 * a second-bounce entry), where `UpdateMultibounceCells` adds it to the
 * second hit's direct light when `use_neural_tail` is set.
 *
 * The encoding is the pathtracer's NRC layout (`nrcEncodingWgsl`, VkNRC's,
 * after Müller et al. 2021): position normalised to the box (triangle waves),
 * direction and normal as one-blob angles, roughness, diffuse and specular
 * reflectance (0 here), two constants. The hooks' contract is
 * `@pathtracer/mlp`'s (`wgsl.ts`): `nnTrainLoad`, `nnInferLoad`,
 * `nnInferStore` in group 1. The encoding's WGSL comes from the package, which
 * is loaded at run time (`mlp-loader.ts`), so the hook builders take it as text.
 */

import { CAPSAICIN_WGSL } from '../wgsl/capsaicin.ts';
import { HASH_GRID_COUNTER, type HashGridCacheBufferLayout } from '../world-cache/capacity.ts';
import { NEURAL_TAIL_WORDS, PROBE_RAY_SURFACE_WGSL, populationHeadWgsl } from '../world-cache/hash-grid-population-wgsl.ts';
import { hashGridCacheWgsl } from '../world-cache/hash-grid-wgsl.ts';

/** `GPUShaderStage.COMPUTE` (the global doesn't exist in Node, where the tests import this file). */
const SHADER_STAGE_COMPUTE = 0x4;

/** vec4s a training record takes. */
export const NEURAL_TAIL_RECORD_VEC4 = 4;
/** vec4s a query takes. */
export const NEURAL_TAIL_QUERY_VEC4 = 3;
/** A training record's kind, in the w of its first vec4. */
export const NEURAL_TAIL_RECORD_KIND = { cell: 1, screen: 2 } as const;
/** Bytes of `NeuralTailParams`. */
export const NEURAL_TAIL_PARAMS_BYTES = 160;
/** Workgroup size of the gather kernels. */
export const NEURAL_TAIL_WORKGROUP = 64;
/** Blocks across a screen-records workgroup (8 × 8). */
export const NEURAL_TAIL_SCREEN_GROUP = 8;

/**
 * The tail's own group: group 3 beside the hash grid's two (cells, queries),
 * group 2 beside the grid's lookup (screen). The training set's record count
 * is a counter of the grid's words (`HASH_GRID_COUNTER.neuralTailRecords`), not
 * a buffer here, so the gather kernels bind 10 storage buffers or fewer
 * (register D6, pmndrs/gi#49).
 */
export const NEURAL_TAIL_BINDING = {
    params: 0,
    /** Training set (cell and screen kernels) or queries (query kernel), `array<vec4f>`. */
    records: 1,
} as const;

export const NEURAL_TAIL_GROUP_ENTRIES: readonly GPUBindGroupLayoutEntry[] = [
    { binding: NEURAL_TAIL_BINDING.params, visibility: SHADER_STAGE_COMPUTE, buffer: { type: 'uniform' } },
    { binding: NEURAL_TAIL_BINDING.records, visibility: SHADER_STAGE_COMPUTE, buffer: { type: 'storage' } },
];

export const NEURAL_TAIL_ENTRY_POINT = {
    gatherCellRecords: 'NeuralTail_GatherCellRecords',
    gatherQueries: 'NeuralTail_GatherQueries',
    gatherScreenRecords: 'NeuralTail_GatherScreenRecords',
} as const;

/** The screen kernel's groups: the hash grid (its lookup) in 1, the tail's in 2. */
export const NEURAL_TAIL_SCREEN_GRID_GROUP = 1;
export const NEURAL_TAIL_SCREEN_OWN_GROUP = 2;

/** `NeuralTailParams`, written once a frame (`NeuralTail.writeParams`). Word offsets. */
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
    screenRecords: 36,
} as const;

const PARAMS_STRUCT_WGSL = /* wgsl */ `
struct NeuralTailParams {
    box_lo: vec4f,                       // the box's min corner
    box_scale: vec4f,                    // 2 / the box's extent, per axis
    inverse_view_projection: mat4x4f,    // camera-relative (camera.ts): world = eye + (M × ndc).xyz / w
    eye: vec4f,
    frame_index: u32,
    set_capacity: u32,                   // records the training set holds
    query_capacity: u32,                 // queries the query buffer holds
    cell_records: u32,                   // about this many first-bounce cells a frame
    tail_list: u32,                      // word where neural_tail_buffer starts in the hash grid's ray lists
    screen_stride: u32,                  // screen records come from stride × stride blocks (0: none)
    screen_width: u32,
    screen_height: u32,
    screen_records: u32,                 // about this many screen records a frame
    padding0: u32,
    padding1: u32,
    padding2: u32,
}
`;

/** The angles the network reads; shared by the gather kernels and the hooks. CPU twin `neuralTailAngles`. */
const INPUT_HELPERS_WGSL = /* wgsl */ `
const NT_PI: f32 = 3.14159265358979;

// A unit direction as (θ / π from +y, atan2(z, x) / 2π + 1/2), each in [0, 1] (neuralTailAngles).
fn ntAngles(direction: vec3f) -> vec2f {
    let d = normalize(direction);
    return vec2f(acos(clamp(d.y, -1.0, 1.0)) / NT_PI, atan2(d.z, d.x) / (2.0 * NT_PI) + 0.5);
}
`;

/** Gather-side: appending a record to the training set. Needs `g_NeuralTail_Records` and the hash grid's words (its record counter). */
const APPEND_RECORD_WGSL = /* wgsl */ `
fn NeuralTail_IsFinite(value: vec3f) -> bool {
    return all((bitcast<vec3u>(value) & vec3u(0x7f800000u)) != vec3u(0x7f800000u));
}

fn NeuralTail_AppendRecord(kind: f32, position: vec3f, outgoing: vec3f, normal: vec3f, albedo: vec3f, target_value: vec3f) {
    if (!NeuralTail_IsFinite(target_value) || !NeuralTail_IsFinite(position)) {
        return;
    }
    let slot = atomicAdd(&g_HashGridCache_TileWords[HASHGRIDCACHE_NEURALTAILRECORDS], 1u);
    if (slot >= g_NeuralTail_Params.set_capacity) {
        return;
    }
    let base = ${NEURAL_TAIL_RECORD_VEC4}u * slot;
    g_NeuralTail_Records[base + 0u] = vec4f(position, kind);
    g_NeuralTail_Records[base + 1u] = vec4f(ntAngles(outgoing), ntAngles(normal));
    g_NeuralTail_Records[base + 2u] = vec4f(albedo, 1.0);
    g_NeuralTail_Records[base + 3u] = vec4f(max(target_value, vec3f(0.0)), 1.0);
}
`;

function groupDeclarationsWgsl(group: number): string {
    return /* wgsl */ `
${PARAMS_STRUCT_WGSL}
@group(${group}) @binding(${NEURAL_TAIL_BINDING.params}) var<uniform> g_NeuralTail_Params: NeuralTailParams;
@group(${group}) @binding(${NEURAL_TAIL_BINDING.records}) var<storage, read_write> g_NeuralTail_Records: array<vec4f>;
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
${populationHeadWgsl(layout)}
${PROBE_RAY_SURFACE_WGSL}
${groupDeclarationsWgsl(NEURAL_TAIL_HASH_GRID_GROUP)}
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
    NeuralTail_AppendRecord(${NEURAL_TAIL_RECORD_KIND.cell}.0, surface.position, -worldCacheProbeRayDirection(probe_ray), surface.normal, surface.albedo, indirect.rgb / max(indirect.w, 1.0));
}
`;
}

/**
 * `NeuralTail_GatherQueries`: over the second-bounce entries
 * (`VisibilityCountBuffer1`, dispatched indirectly), each writes its hit as
 * a query at its own index (a miss with w 0), so inference's record r is
 * second-bounce entry r; an entry past the query buffer clears its answer.
 */
export function gatherQueriesWgsl(layout: HashGridCacheBufferLayout): string {
    return /* wgsl */ `
${CAPSAICIN_WGSL}
${populationHeadWgsl(layout)}
${PROBE_RAY_SURFACE_WGSL}
${groupDeclarationsWgsl(NEURAL_TAIL_HASH_GRID_GROUP)}
${INPUT_HELPERS_WGSL}

@compute @workgroup_size(${NEURAL_TAIL_WORKGROUP})
fn ${NEURAL_TAIL_ENTRY_POINT.gatherQueries}(
    @builtin(workgroup_id) workgroup: vec3u,
    @builtin(num_workgroups) workgroups: vec3u,
    @builtin(local_invocation_index) local_index: u32,
) {
    let did = HashGridCache_WorkgroupIndex(workgroup, workgroups) * ${NEURAL_TAIL_WORKGROUP}u + local_index;
    if (did >= atomicLoad(&g_HashGridCache_TileWords[HASHGRIDCACHE_VISIBILITYCOUNTBUFFER1])) {
        return;
    }
    if (did >= g_NeuralTail_Params.query_capacity) {
        // No room for its query: no answer this frame (not last frame's).
        HashGridCache_SetRayListWord(g_NeuralTail_Params.tail_list, ${NEURAL_TAIL_WORDS}u * did, 0u);
        HashGridCache_SetRayListWord(g_NeuralTail_Params.tail_list, ${NEURAL_TAIL_WORDS}u * did + 1u, 0u);
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
 * the screen, a random share of about `screen_records` of them kept (so no
 * part of the screen is favoured at any size). A random pixel in the block, a
 * primary ray through it from the eye (perspective cameras), and where it
 * hits, one cosine-weighted ray from that surface: the record's target is
 * the surface's albedo × the light leaving that ray's hit as the cache holds
 * it (`worldCacheLookup`), or 0 for a miss; a hit with no cell adds nothing.
 * Built with the platform's hit shading (group 0), the hash grid (group 1,
 * its lookup) and the tail's group (2).
 */
export function gatherScreenRecordsWgsl(layout: HashGridCacheBufferLayout): string {
    return /* wgsl */ `
${CAPSAICIN_WGSL}
${hashGridCacheWgsl(NEURAL_TAIL_SCREEN_GRID_GROUP, layout)}
${groupDeclarationsWgsl(NEURAL_TAIL_SCREEN_OWN_GROUP)}
${INPUT_HELPERS_WGSL}
${APPEND_RECORD_WGSL}

// A cosine-weighted direction around n (Malley's method on a frame built from n).
fn NeuralTail_CosineDirection(n: vec3f, u: vec2f) -> vec3f {
    let r = sqrt(u.x);
    let phi = 2.0 * NT_PI * u.y;
    let helper = select(vec3f(1.0, 0.0, 0.0), vec3f(0.0, 1.0, 0.0), abs(n.x) > 0.9);
    let t = normalize(cross(helper, n));
    let b = cross(n, t);
    return normalize(t * (r * cos(phi)) + b * (r * sin(phi)) + n * sqrt(max(1.0 - u.x, 0.0)));
}

@compute @workgroup_size(${NEURAL_TAIL_SCREEN_GROUP}, ${NEURAL_TAIL_SCREEN_GROUP})
fn ${NEURAL_TAIL_ENTRY_POINT.gatherScreenRecords}(@builtin(global_invocation_id) id: vec3u) {
    let params = g_NeuralTail_Params;
    let stride = params.screen_stride;
    let size = vec2u(params.screen_width, params.screen_height);
    if (stride == 0u || id.x * stride >= size.x || id.y * stride >= size.y) {
        return;
    }
    var random = rcMakeRandom(params.frame_index ^ 0x73637265u, id.y * 65536u + id.x);
    let blocks = ((size.x + stride - 1u) / stride) * ((size.y + stride - 1u) / stride);
    if (rcRand(&random) >= f32(params.screen_records) / f32(blocks)) {
        return;
    }
    let offset = min(vec2u(rcRand2(&random) * f32(stride)), vec2u(stride - 1u));
    let pixel = min(id.xy * stride + offset, size - 1u);
    let uv = (vec2f(pixel) + 0.5) / vec2f(size);
    let ndc = vec4f(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, 0.5, 1.0);
    let relative = params.inverse_view_projection * ndc;
    let view_direction = normalize(relative.xyz / relative.w);
    let view_ray = GiRay(params.eye.xyz, 0.0, view_direction, 3.0e38);
    let view_hit = giTraceClosest(view_ray);
    if (view_hit.t < 0.0) {
        return;
    }
    let surface = giSurfaceAt(view_ray, view_hit);
    let side = giShadeRaySide(surface);
    let direction = NeuralTail_CosineDirection(side, rcRand2(&random));
    let ray = GiRay(giShadeOffsetRayOrigin(surface.position, side), 0.0, direction, RC_MAX_HIT_DISTANCE);
    let hit = giTraceClosest(ray);
    var target_value = vec3f(0.0);
    if (hit.t >= 0.0) {
        let lookup = worldCacheLookup(ray.origin + hit.t * ray.direction, vec3f(0.0), ray.direction);
        if (lookup.w <= 0.0) {
            return; // the hit has no cell: nothing known there
        }
        // Cosine-weighted: BRDF × cos / pdf = albedo for a Lambert surface.
        target_value = surface.albedo * lookup.rgb;
    }
    NeuralTail_AppendRecord(${NEURAL_TAIL_RECORD_KIND.screen}.0, surface.position, -view_direction, surface.normal, surface.albedo, target_value);
}
`;
}

/** The network's inputs from a record's first three vec4s (both hooks). `encoding` is `nrcEncodingWgsl(positionBands)` from the package. */
function inputsWgsl(encoding: string): string {
    return /* wgsl */ `
${encoding}
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
export function neuralTailTrainHooks(encoding: string): string {
    return /* wgsl */ `
${PARAMS_STRUCT_WGSL}
@group(1) @binding(0) var<storage, read> ntSet: array<vec4f>;
// The hash grid's counter block (HASH_GRID_COUNTER): the training set's count is one of its words.
@group(1) @binding(1) var<storage, read> ntCounts: array<u32>;
@group(1) @binding(2) var<uniform> ntParams: NeuralTailParams;
${inputsWgsl(encoding)}
fn ntHash(x: u32) -> u32 {
    let s = x * 747796405u + 2891336453u;
    let w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
    return (w >> 22u) ^ w;
}
fn nnTrainLoad(r: u32, x: ptr<function, array<f32, NN_IN>>) -> vec3f {
    let n = max(min(ntCounts[${HASH_GRID_COUNTER.neuralTailRecords}u], ntParams.set_capacity), 1u);
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
export function neuralTailInferHooks(encoding: string): string {
    return /* wgsl */ `
${PARAMS_STRUCT_WGSL}
@group(1) @binding(0) var<storage, read> ntQueries: array<vec4f>;
@group(1) @binding(1) var<storage, read_write> ntListWords: array<u32>;
@group(1) @binding(2) var<uniform> ntParams: NeuralTailParams;
${inputsWgsl(encoding)}
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
