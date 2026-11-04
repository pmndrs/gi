/**
 * WGSL for the sky projection: the compute kernels that turn the host's
 * environment (an equirect or cube texture, a colour, or the platform's own
 * procedural sky texture) into the platform's engine-neutral sky:
 *
 * - `giSkyResample`: the radiance map rays see when they miss, an
 *   equirectangular `rgba16float` texture (texture order: first row the nadir,
 *   as in the contract), one thread per texel, each averaging a few samples of
 *   the source over its texel;
 * - `giSkyProject`: the irradiance SH (bands 0, 1, 2, 4;
 *   `spherical-harmonics.ts`), one workgroup that integrates the source over
 *   the sphere and writes 18 coefficients, already convolved with the cosine
 *   lobe, so hit shading only evaluates the basis.
 *
 * Both run only when the environment changes (`platform-sky.ts`).
 *
 * The SH reduction is subgroups first (DECISIONS 2026-11-04): each subgroup
 * adds its threads' sums with `subgroupAdd`, and the subgroup totals meet in
 * workgroup memory. The declared fallback, for devices without the
 * `subgroups` feature, is a tree reduction in workgroup memory. Same
 * structure, same result up to float rounding order; which one is chosen when
 * the kernel is built, and a GPU test runs both on the same sky.
 */

import { SH_BASIS_WGSL, SH_COEFFICIENT_BANDS, SH_COEFFICIENT_COUNT, cosineLobeBandWeight } from './spherical-harmonics.ts';

/** Threads in the projection's single workgroup. 128 keeps the subgroup partials under the 16 KB default of workgroup memory even for 4-wide subgroups. */
export const PROJECTION_WORKGROUP_SIZE = 128;

/** Smallest subgroup any WebGPU device reports (`subgroupMinSize` is at least 4), so the most subgroup totals a workgroup can have. */
const MAX_SUBGROUPS_PER_WORKGROUP = PROJECTION_WORKGROUP_SIZE / 4;

/** Workgroup size of the resample kernel (8 x 8 texels). */
export const RESAMPLE_WORKGROUP_SIZE = 8;

/** Where the source comes from, in `GiSkyProjectionParams.sourceKind`. */
export const SKY_SOURCE_KIND = {
    /** A constant radiance (`color` and `none`). */
    constant: 0,
    /** An equirect 2D texture in the contract's texture order. */
    equirect: 1,
    /** A cube texture, sampled with WebGPU's own cube addressing. */
    cube: 2,
} as const;

/** How the projection walks the sphere, in `GiSkyProjectionParams.gridKind`. */
export const SKY_GRID_KIND = {
    /** gridWidth x gridHeight cells of longitude and latitude. */
    equirect: 0,
    /** Six cube faces of gridWidth x gridWidth cells each. */
    cube: 1,
} as const;

/** Byte size of the projection's parameter block (see the WGSL struct). */
export const PROJECTION_PARAMS_BYTES = 64;

/** Byte size of the SH output: 18 vec4f (rgb, unused w). */
export const SH_BYTES = SH_COEFFICIENT_COUNT * 16;

export type SkyReduction = 'subgroups' | 'workgroup';

/** Binding numbers of the projection kernels' one bind group. */
export const PROJECTION_BINDING = {
    params: 0,
    sourceEquirect: 1,
    sourceCube: 2,
    sourceSampler: 3,
    radianceMap: 4,
    irradianceSh: 5,
} as const;

const SHARED_WGSL = /* wgsl */ `
struct GiSkyProjectionParams {
    constantRadiance: vec3f,  // SKY_SOURCE_KIND.constant: the radiance everywhere
    intensity: f32,           // multiplies texture sources
    sourceKind: u32,
    gridKind: u32,
    gridWidth: u32,
    gridHeight: u32,          // cube grids: equal to gridWidth
    rotationCos: f32,         // cos and sin of rotationY, the environment's turn about +Y
    rotationSin: f32,
    supersample: u32,         // resample: samples per texel along each axis
    iterations: u32,          // project: grid cells per thread, rounded up (uniform, so the loop is too)
}

@group(0) @binding(${PROJECTION_BINDING.params}) var<uniform> params: GiSkyProjectionParams;
@group(0) @binding(${PROJECTION_BINDING.sourceEquirect}) var sourceEquirect: texture_2d<f32>;
@group(0) @binding(${PROJECTION_BINDING.sourceCube}) var sourceCube: texture_cube<f32>;
@group(0) @binding(${PROJECTION_BINDING.sourceSampler}) var sourceSampler: sampler;
@group(0) @binding(${PROJECTION_BINDING.radianceMap}) var radianceMap: texture_storage_2d<rgba16float, write>;
@group(0) @binding(${PROJECTION_BINDING.irradianceSh}) var<storage, read_write> irradianceSh: array<vec4f, ${SH_COEFFICIENT_COUNT}>;

const PI: f32 = 3.14159265358979;
const TWO_PI: f32 = 6.28318530717959;
const INV_TWO_PI: f32 = 0.159154943091895;
const INV_PI: f32 = 0.318309886183791;

// The contract's equirect layout in texture coordinates (three.js's
// equirectUV): u = atan2(z, x) / 2pi + 0.5, v = asin(y) / pi + 0.5, so v = 0 is
// the nadir row (the first in memory) and v = 1 the zenith.
fn equirectUv(direction: vec3f) -> vec2f {
    return vec2f(atan2(direction.z, direction.x) * INV_TWO_PI + 0.5, asin(clamp(direction.y, -1.0, 1.0)) * INV_PI + 0.5);
}

fn equirectDirection(uv: vec2f) -> vec3f {
    let azimuth = (uv.x - 0.5) * TWO_PI;
    let latitude = (uv.y - 0.5) * PI;
    return vec3f(cos(latitude) * cos(azimuth), sin(latitude), cos(latitude) * sin(azimuth));
}

// Radiance of the host's environment along a world direction. rotationY turns
// the environment about +Y (counter-clockwise seen from above): what the
// unrotated source shows along d appears along R(rotationY) d, so the source
// is read at R(-rotationY) d.
fn sourceRadiance(worldDirection: vec3f) -> vec3f {
    if (params.sourceKind == ${SKY_SOURCE_KIND.constant}u) {
        return params.constantRadiance;
    }
    let c = params.rotationCos;
    let s = params.rotationSin;
    let d = vec3f(c * worldDirection.x - s * worldDirection.z, worldDirection.y, s * worldDirection.x + c * worldDirection.z);
    if (params.sourceKind == ${SKY_SOURCE_KIND.equirect}u) {
        return textureSampleLevel(sourceEquirect, sourceSampler, equirectUv(d), 0.0).rgb * params.intensity;
    }
    return textureSampleLevel(sourceCube, sourceSampler, d, 0.0).rgb * params.intensity;
}

// The centre of cell (column, row) of cube face 'face', as a direction. The
// faces and their (s, t) axes are WebGPU's cube addressing (the same as
// Vulkan's and D3D's): +X, -X, +Y, -Y, +Z, -Z, with t = 0 the first row.
fn cubeCellDirection(face: u32, column: u32, row: u32, size: u32) -> vec3f {
    let s = (f32(column) + 0.5) / f32(size) * 2.0 - 1.0;
    let t = (f32(row) + 0.5) / f32(size) * 2.0 - 1.0;
    switch (face) {
        case 0u: { return normalize(vec3f(1.0, -t, -s)); }
        case 1u: { return normalize(vec3f(-1.0, -t, s)); }
        case 2u: { return normalize(vec3f(s, 1.0, t)); }
        case 3u: { return normalize(vec3f(s, -1.0, -t)); }
        case 4u: { return normalize(vec3f(s, -t, 1.0)); }
        default: { return normalize(vec3f(-s, -t, -1.0)); }
    }
}

// Solid angle of the rectangle [x0, x1] x [y0, y1] on a cube face at distance
// 1: the integral of (1 + x^2 + y^2)^(-3/2), whose antiderivative is
// atan(x y / sqrt(1 + x^2 + y^2)).
fn cubeAreaElement(x: f32, y: f32) -> f32 {
    return atan2(x * y, sqrt(x * x + y * y + 1.0));
}

fn cubeCellSolidAngle(column: u32, row: u32, size: u32) -> f32 {
    let x0 = f32(column) / f32(size) * 2.0 - 1.0;
    let x1 = f32(column + 1u) / f32(size) * 2.0 - 1.0;
    let y0 = f32(row) / f32(size) * 2.0 - 1.0;
    let y1 = f32(row + 1u) / f32(size) * 2.0 - 1.0;
    return cubeAreaElement(x0, y0) - cubeAreaElement(x0, y1) - cubeAreaElement(x1, y0) + cubeAreaElement(x1, y1);
}
`;

/** The resample kernel: one thread per radiance-map texel. */
const RESAMPLE_WGSL = /* wgsl */ `
@compute @workgroup_size(${RESAMPLE_WORKGROUP_SIZE}, ${RESAMPLE_WORKGROUP_SIZE})
fn giSkyResample(@builtin(global_invocation_id) id: vec3u) {
    let size = textureDimensions(radianceMap);
    if (id.x >= size.x || id.y >= size.y) {
        return;
    }
    // A box filter over the texel: samples spaced no wider than the source's
    // texels, weighted by cos(latitude), the solid angle each covers.
    let samples = max(params.supersample, 1u);
    var sum = vec3f(0.0);
    var weightSum = 0.0;
    for (var j = 0u; j < samples; j++) {
        for (var i = 0u; i < samples; i++) {
            let uv = (vec2f(id.xy) + (vec2f(f32(i), f32(j)) + 0.5) / f32(samples)) / vec2f(size);
            let direction = equirectDirection(uv);
            let weight = max(cos((uv.y - 0.5) * PI), 1.0e-6);
            sum += sourceRadiance(direction) * weight;
            weightSum += weight;
        }
    }
    textureStore(radianceMap, id.xy, vec4f(sum / weightSum, 1.0));
}
`;

/** WGSL that returns the workgroup-wide sum of a per-thread vec3f, to every thread. Must be called in uniform control flow. */
function reductionWgsl(reduction: SkyReduction): string {
    if (reduction === 'subgroups') {
        return /* wgsl */ `
// Subgroups first: each subgroup adds its threads' values in hardware, then
// one value per subgroup meets in workgroup memory. Subgroup sizes differ
// between devices (4 to 128) and nothing promises how subgroups map onto
// local_invocation_index, so each subgroup takes its slot with an atomic.
var<workgroup> subgroupTotals: array<vec4f, ${MAX_SUBGROUPS_PER_WORKGROUP}>;
var<workgroup> subgroupCount: atomic<u32>;

fn claimSubgroupSlot(lane: u32) -> u32 {
    var slot = 0u;
    if (lane == 0u) {
        slot = atomicAdd(&subgroupCount, 1u);
    }
    workgroupBarrier();
    return slot;
}

fn workgroupSum(value: vec3f, lane: u32, slot: u32) -> vec3f {
    let subgroupTotal = subgroupAdd(value);
    if (lane == 0u) {
        subgroupTotals[slot] = vec4f(subgroupTotal, 0.0);
    }
    workgroupBarrier();
    let count = atomicLoad(&subgroupCount);
    var total = vec3f(0.0);
    for (var index = 0u; index < count; index++) {
        total += subgroupTotals[index].xyz;
    }
    // Everyone has read the totals before the next call overwrites them.
    workgroupBarrier();
    return total;
}
`;
    }
    return /* wgsl */ `
// The declared fallback without subgroups: a tree reduction in workgroup
// memory, halving the active threads each step.
var<workgroup> partialSums: array<vec4f, ${PROJECTION_WORKGROUP_SIZE}>;

fn claimSubgroupSlot(lane: u32) -> u32 {
    return 0u;
}

fn workgroupSum(value: vec3f, local: u32, slot: u32) -> vec3f {
    partialSums[local] = vec4f(value, 0.0);
    workgroupBarrier();
    for (var stride = ${PROJECTION_WORKGROUP_SIZE / 2}u; stride > 0u; stride >>= 1u) {
        if (local < stride) {
            partialSums[local] += partialSums[local + stride];
        }
        workgroupBarrier();
    }
    let total = partialSums[0].xyz;
    workgroupBarrier();
    return total;
}
`;
}

/** The projection kernel: one workgroup sums radiance times basis times solid angle over the whole sphere. */
function projectWgsl(reduction: SkyReduction): string {
    const coefficients = Array.from({ length: SH_COEFFICIENT_COUNT }, (_, index) => index);
    const accumulators = coefficients.map((index) => `    var sum${index} = vec3f(0.0);`).join('\n');
    const accumulate = coefficients.map((index) => `            sum${index} += weighted * basis[${index}];`).join('\n');
    // Reduce, convolve with the clamped cosine (band l times A_l) and store.
    const reduceAndStore = coefficients
        .map((index) => {
            const weight = cosineLobeBandWeight(SH_COEFFICIENT_BANDS[index]);
            return `    let total${index} = workgroupSum(sum${index}, reductionIndex, slot);\n    if (local == 0u) { irradianceSh[${index}] = vec4f(total${index} * ${weight.toPrecision(17)}, 0.0); }`;
        })
        .join('\n');
    const laneBuiltin = reduction === 'subgroups' ? ', @builtin(subgroup_invocation_id) lane: u32' : '';
    // The fallback's reduction is indexed by the thread's place in the workgroup, the subgroup one by its lane.
    const reductionIndex = reduction === 'subgroups' ? 'lane' : 'local';

    return /* wgsl */ `
${reductionWgsl(reduction)}

@compute @workgroup_size(${PROJECTION_WORKGROUP_SIZE})
fn giSkyProject(@builtin(local_invocation_index) local: u32${laneBuiltin}) {
    let reductionIndex = ${reductionIndex};
    let slot = claimSubgroupSlot(reductionIndex);

    // Every thread runs the same number of iterations (a uniform value), so
    // the reductions after the loop are in uniform control flow.
    let isCube = params.gridKind == ${SKY_GRID_KIND.cube}u;
    let cellsPerFace = params.gridWidth * params.gridHeight;
    let cellCount = select(cellsPerFace, 6u * cellsPerFace, isCube);
${accumulators}
    for (var iteration = 0u; iteration < params.iterations; iteration++) {
        let cell = iteration * ${PROJECTION_WORKGROUP_SIZE}u + local;
        if (cell < cellCount) {
            var direction: vec3f;
            var solidAngle: f32;
            if (isCube) {
                let face = cell / cellsPerFace;
                let inFace = cell % cellsPerFace;
                let column = inFace % params.gridWidth;
                let row = inFace / params.gridWidth;
                direction = cubeCellDirection(face, column, row, params.gridWidth);
                solidAngle = cubeCellSolidAngle(column, row, params.gridWidth);
            } else {
                let column = cell % params.gridWidth;
                let row = cell / params.gridWidth;
                let uv = (vec2f(f32(column), f32(row)) + 0.5) / vec2f(f32(params.gridWidth), f32(params.gridHeight));
                direction = equirectDirection(uv);
                // A row of cells between latitudes a0 and a1 covers 2pi (sin a1 - sin a0); each cell its share.
                let latitude0 = (f32(row) / f32(params.gridHeight) - 0.5) * PI;
                let latitude1 = (f32(row + 1u) / f32(params.gridHeight) - 0.5) * PI;
                solidAngle = TWO_PI / f32(params.gridWidth) * (sin(latitude1) - sin(latitude0));
            }
            let weighted = sourceRadiance(direction) * solidAngle;
            let basis = giSkyShBasis(direction);
${accumulate}
        }
    }

${reduceAndStore}
}
`;
}

/** The whole projection shader (both entry points) for one reduction. */
export function projectionWgsl(reduction: SkyReduction): string {
    const enable = reduction === 'subgroups' ? 'enable subgroups;\n' : '';
    return `${enable}${SHARED_WGSL}\n${SH_BASIS_WGSL}\n${RESAMPLE_WGSL}\n${projectWgsl(reduction)}`;
}
