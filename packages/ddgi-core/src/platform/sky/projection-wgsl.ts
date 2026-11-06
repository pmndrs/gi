/**
 * WGSL for the sky bake: the compute kernels that turn the host's environment
 * (an equirect or cube texture, a colour, or a procedural Preetham sky the
 * GPU evaluates itself) into the platform's engine-neutral sky. Three
 * dispatches in one pass, in this order (`platform-sky.ts`):
 *
 * 1. `giSkyPartialSums`, many workgroups: each integrates its share of a grid
 *    over the sphere, radiance times each SH basis function times the cell's
 *    solid angle (bands 0, 1, 2 and 4; `spherical-harmonics.ts`), plus the
 *    irradiance the sky delivers to open, flat ground, and writes one partial
 *    sum per workgroup;
 * 2. `giSkySum`, one workgroup: adds the partial sums, works out the
 *    procedural sky's calibration scale from the ground's irradiance, and
 *    writes the sky's harmonics: its **radiance** SH (bands 0-2, which the
 *    instant sky needs, m2-lights.md §6.3) and its **irradiance** SH (the
 *    same sums times the clamped-cosine weights A_l, so hit shading only
 *    evaluates the basis);
 * 3. `giSkyResample`: the radiance map rays see when they miss, an
 *    equirectangular `rgba16float` texture (first row the nadir, as in the
 *    contract), one thread per texel, averaging a few samples of the source
 *    over its texel, times the calibration scale.
 *
 * A procedural sky has no source texture: the kernels evaluate Preetham at
 * each direction (`giSkyPreetham`) from per-sky numbers the CPU worked out
 * (`procedural-sky.ts`). Its calibration needs the unscaled sky's ground
 * irradiance, a sum over the sky, which is why the sum comes before the
 * resample. Everything is linear in the radiance, so the partial sums are
 * made on the unscaled sky and scaled once at the end.
 *
 * **Why two levels:** M1's projection ran in one workgroup. A procedural sky
 * re-bakes whenever the sun moves (time of day), so the bake must cost little
 * GPU time; one workgroup walking 131,072 cells leaves most of the GPU idle.
 *
 * **Why the reductions loop over the coefficients:** M1's kernel wrote out
 * 18 accumulators and 18 calls to the workgroup reduction, each inlined with
 * its barriers and its loop over the subgroup totals. That is what took about
 * 1 s to compile on the RTX 3060 (T1.9a). T2.6a measured that kernel at
 * 640 ms against about 60 ms for the same work written as loops over a
 * private array. The loops have constant bounds, and on the subgroup path the
 * one barrier is outside them.
 *
 * The reductions are subgroups first (DECISIONS 2026-11-04): each subgroup
 * adds its threads' sums with `subgroupAdd`, and the subgroup totals meet in
 * workgroup memory, all 19 values at once. The declared fallback, for
 * devices without the `subgroups` feature, is a tree reduction in workgroup
 * memory, one value at a time. Same result up to float rounding order;
 * which one is chosen when the kernels are built, and a GPU test runs both on
 * the same skies.
 */

import { SH_BASIS_WGSL, SH_COEFFICIENT_BANDS, SH_COEFFICIENT_COUNT, cosineLobeBandWeight } from './spherical-harmonics.ts';

/** Threads per workgroup of the partial sums and of the final sum. */
export const PROJECTION_WORKGROUP_SIZE = 128;

/**
 * The most partial sums the final workgroup adds (each of its threads adds
 * up to two). The CPU picks the cells per thread so that a grid never needs
 * more workgroups than this.
 */
export const MAX_PARTIAL_SUMS = 2 * PROJECTION_WORKGROUP_SIZE;

/** Smallest subgroup any WebGPU device reports (`subgroupMinSize` is at least 4), so the most subgroup totals a workgroup can have. */
const MAX_SUBGROUPS_PER_WORKGROUP = PROJECTION_WORKGROUP_SIZE / 4;

/** Workgroup size of the resample kernel (8 x 8 texels). */
export const RESAMPLE_WORKGROUP_SIZE = 8;

/** Radiance SH coefficients the bake writes: bands 0, 1 and 2 (the instant sky's 9 patterns, m2-lights.md §6.3). */
export const RADIANCE_SH_COUNT = 9;

/**
 * The values each partial sum holds: the 18 SH coefficients' radiance
 * integrals (rgb), then the irradiance on open, flat ground (rgb), which
 * calibrates a procedural sky.
 */
const SUMMED_VALUES = SH_COEFFICIENT_COUNT + 1;
const GROUND_IRRADIANCE_INDEX = SH_COEFFICIENT_COUNT;

/** Where the source comes from, in `GiSkyProjectionParams.sourceKind`. */
export const SKY_SOURCE_KIND = {
    /** A constant radiance (`color` and `none`). */
    constant: 0,
    /** An equirect 2D texture in the contract's texture order. */
    equirect: 1,
    /** A cube texture, sampled with WebGPU's own cube addressing. */
    cube: 2,
    /** The Preetham sky, evaluated on the GPU and calibrated against the ground (`procedural-sky.ts`). */
    procedural: 3,
} as const;

/** How the integration walks the sphere, in `GiSkyProjectionParams.gridKind`. */
export const SKY_GRID_KIND = {
    /** gridWidth x gridHeight cells of longitude and latitude. */
    equirect: 0,
    /** Six cube faces of gridWidth x gridWidth cells each. */
    cube: 1,
} as const;

/** Word offsets of the parameter block's fields (the WGSL struct `GiSkyProjectionParams`), for the CPU writer. */
export const PROJECTION_PARAMS_WORD = {
    constantRadiance: 0,
    intensity: 3,
    sourceKind: 4,
    gridKind: 5,
    gridWidth: 6,
    gridHeight: 7,
    rotationCos: 8,
    rotationSin: 9,
    supersample: 10,
    cellsPerThread: 11,
    sunDirection: 12,
    partialCount: 15,
    /** Three Perez distributions (`GiSkyPerez`, {@link PEREZ_WORDS} words each). */
    perezLuminance: 16,
    perezX: 24,
    perezY: 32,
    targetHorizontalLuminance: 40,
} as const;

/** Words of one `GiSkyPerez`: A, B, C, D, E, the zenith scale, two spare. */
export const PEREZ_WORDS = 8;

/** The parameter block's size: 41 words, rounded up to the struct's 16-byte alignment. */
export const PROJECTION_PARAMS_BYTES = 176;

/**
 * The sky's harmonics, as the bake writes them (`GiSkyHarmonics`): radiance
 * SH bands 0-2 first, then irradiance SH bands 0, 1, 2 and 4, each a vec4f
 * (rgb, w unused). Radiance comes first so a uniform binding of it alone
 * starts at offset 0 (uniform offsets must be multiples of 256).
 */
export const SKY_HARMONICS_LAYOUT = {
    radiance: 0,
    radianceBytes: RADIANCE_SH_COUNT * 16,
    irradiance: RADIANCE_SH_COUNT * 16,
    byteLength: (RADIANCE_SH_COUNT + SH_COEFFICIENT_COUNT) * 16,
} as const;

/** The bake's own storage (`GiSkyReduction`): the calibration scale (a vec4f), then the partial sums. */
export const REDUCTION_BYTES = 16 + MAX_PARTIAL_SUMS * SUMMED_VALUES * 16;

export type SkyReduction = 'subgroups' | 'workgroup';

/** Binding numbers of the bake kernels' one bind group. */
export const PROJECTION_BINDING = {
    params: 0,
    sourceEquirect: 1,
    sourceCube: 2,
    sourceSampler: 3,
    radianceMap: 4,
    reduction: 5,
    harmonics: 6,
} as const;

/** The bake's entry points, in dispatch order. */
export const PROJECTION_ENTRY_POINTS = { partialSums: 'giSkyPartialSums', sum: 'giSkySum', resample: 'giSkyResample' } as const;

const SHARED_WGSL = /* wgsl */ `
// One Perez distribution of the Preetham sky (preetham.ts): its five
// coefficients, and the quantity's zenith value divided by F(0, thetaSun),
// so that value = zenithScale * F(theta, gamma).
struct GiSkyPerez {
    a: f32,
    b: f32,
    c: f32,
    d: f32,
    e: f32,
    zenithScale: f32,
    spare0: f32,
    spare1: f32,
}

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
    cellsPerThread: u32,      // partial sums: grid cells per thread (uniform, so the loop is too)
    sunDirection: vec3f,      // procedural: unit, toward the sun
    partialCount: u32,        // workgroups of the partial sums, at most ${MAX_PARTIAL_SUMS}
    perezLuminance: GiSkyPerez,
    perezX: GiSkyPerez,
    perezY: GiSkyPerez,
    targetHorizontalLuminance: f32,  // procedural: what the ground's luminance-weighted irradiance is scaled to
}

// One workgroup's partial sums: the ${SH_COEFFICIENT_COUNT} SH coefficients' radiance integrals, then the ground's irradiance.
struct GiSkyPartialSum {
    values: array<vec4f, ${SUMMED_VALUES}>,
}

struct GiSkyReduction {
    scale: vec4f,             // x: the calibration scale (1 for every source but procedural)
    partials: array<GiSkyPartialSum, ${MAX_PARTIAL_SUMS}>,
}

// What the bake writes for hit shading: radiance SH bands 0-2, then
// irradiance SH bands 0, 1, 2 and 4 (rgb, w unused).
struct GiSkyHarmonics {
    radiance: array<vec4f, ${RADIANCE_SH_COUNT}>,
    irradiance: array<vec4f, ${SH_COEFFICIENT_COUNT}>,
}

@group(0) @binding(${PROJECTION_BINDING.params}) var<uniform> params: GiSkyProjectionParams;
@group(0) @binding(${PROJECTION_BINDING.sourceEquirect}) var sourceEquirect: texture_2d<f32>;
@group(0) @binding(${PROJECTION_BINDING.sourceCube}) var sourceCube: texture_cube<f32>;
@group(0) @binding(${PROJECTION_BINDING.sourceSampler}) var sourceSampler: sampler;
@group(0) @binding(${PROJECTION_BINDING.radianceMap}) var radianceMap: texture_storage_2d<rgba16float, write>;
@group(0) @binding(${PROJECTION_BINDING.reduction}) var<storage, read_write> reduction: GiSkyReduction;
@group(0) @binding(${PROJECTION_BINDING.harmonics}) var<storage, read_write> harmonics: GiSkyHarmonics;

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

// Perez et al.'s sky distribution (preetham.ts, perez), times the zenith scale:
//   F(theta, gamma) = (1 + A exp(B / cos theta)) (1 + C exp(D gamma) + E cos^2 gamma).
// At the horizon B / cos theta goes to minus infinity and the exponential to
// 0; the floor on cos theta gives that limit without dividing by 0.
fn giSkyPerezDistribution(perez: GiSkyPerez, cosTheta: f32, gamma: f32, cosGamma: f32) -> f32 {
    let horizonTerm = 1.0 + perez.a * exp(perez.b / max(cosTheta, 1.0e-6));
    let sunTerm = 1.0 + perez.c * exp(perez.d * gamma) + perez.e * cosGamma * cosGamma;
    return perez.zenithScale * horizonTerm * sunTerm;
}

// The unscaled Preetham sky along a unit direction, in linear Rec.709 RGB:
// preetham.ts's preethamSkyXyY then xyYToLinearRgb, black below the horizon
// (the ground is scene geometry), and no sun disc (the sun is a light).
fn giSkyPreetham(direction: vec3f) -> vec3f {
    if (direction.y <= 0.0) {
        return vec3f(0.0);
    }
    let sun = params.sunDirection;
    let cosGamma = dot(direction, sun);
    // The angle to the sun from atan2, not acos: in f32, acos loses most of
    // its precision near the sun, where the glow is brightest.
    let gamma = atan2(length(cross(direction, sun)), cosGamma);
    let luminance = giSkyPerezDistribution(params.perezLuminance, direction.y, gamma, cosGamma);
    let x = giSkyPerezDistribution(params.perezX, direction.y, gamma, cosGamma);
    let y = giSkyPerezDistribution(params.perezY, direction.y, gamma, cosGamma);
    if (y <= 0.0) {
        return vec3f(0.0);
    }
    // CIE xyY to XYZ, then the standard XYZ to linear sRGB matrix
    // (IEC 61966-2-1); channels outside the gamut clamp to 0, as on the CPU.
    let bigX = x / y * luminance;
    let bigZ = (1.0 - x - y) / y * luminance;
    let rgb = vec3f(
        3.2404542 * bigX - 1.5371385 * luminance - 0.4985314 * bigZ,
        -0.969266 * bigX + 1.8760108 * luminance + 0.041556 * bigZ,
        0.0556434 * bigX - 0.2040259 * luminance + 1.0572252 * bigZ,
    );
    return max(rgb, vec3f(0.0));
}

// Radiance of the environment along a world direction, before the
// calibration scale (which only a procedural sky has). rotationY turns the
// environment about +Y (counter-clockwise seen from above): what the
// unrotated source shows along d appears along R(rotationY) d, so the source
// is read at R(-rotationY) d.
fn sourceRadiance(worldDirection: vec3f) -> vec3f {
    if (params.sourceKind == ${SKY_SOURCE_KIND.constant}u) {
        return params.constantRadiance;
    }
    if (params.sourceKind == ${SKY_SOURCE_KIND.procedural}u) {
        return giSkyPreetham(worldDirection);
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

/** The resample kernel: one thread per radiance-map texel, after the sum (it reads the calibration scale). */
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
    textureStore(radianceMap, id.xy, vec4f(sum / weightSum * reduction.scale.x, 1.0));
}
`;

/**
 * WGSL for `giSkyReduce`: the workgroup's total of each of the
 * SUMMED_VALUES values in `giSkySums`, returned to thread `local` for value
 * number `local` (threads past the last value get 0). Called by every
 * thread, in uniform control flow: it has barriers.
 */
function reductionWgsl(reduction: SkyReduction): string {
    if (reduction === 'subgroups') {
        return /* wgsl */ `
// Subgroups first: each subgroup adds its threads' values in hardware, then
// one row of totals per subgroup meets in workgroup memory, every value at
// once, behind one barrier. Subgroup sizes differ between devices (4 to
// 128) and nothing promises how subgroups map onto local_invocation_index,
// so each subgroup takes its row with an atomic.
var<workgroup> giSkySubgroupTotals: array<array<vec4f, ${SUMMED_VALUES}>, ${MAX_SUBGROUPS_PER_WORKGROUP}>;
var<workgroup> giSkySubgroupCount: atomic<u32>;

fn giSkyReduce(local: u32, lane: u32) -> vec3f {
    var row = 0u;
    if (lane == 0u) {
        row = atomicAdd(&giSkySubgroupCount, 1u);
    }
    for (var value = 0u; value < ${SUMMED_VALUES}u; value++) {
        let subgroupTotal = subgroupAdd(giSkySums[value]);
        if (lane == 0u) {
            giSkySubgroupTotals[row][value] = vec4f(subgroupTotal, 0.0);
        }
    }
    workgroupBarrier();
    var total = vec3f(0.0);
    if (local < ${SUMMED_VALUES}u) {
        let rows = atomicLoad(&giSkySubgroupCount);
        for (var index = 0u; index < rows; index++) {
            total += giSkySubgroupTotals[index][local].xyz;
        }
    }
    return total;
}
`;
    }
    return /* wgsl */ `
// The declared fallback without subgroups: a tree reduction in workgroup
// memory, halving the active threads each step, one value at a time.
var<workgroup> giSkyTree: array<vec4f, ${PROJECTION_WORKGROUP_SIZE}>;
var<workgroup> giSkyTotals: array<vec4f, ${SUMMED_VALUES}>;

fn giSkyReduce(local: u32, lane: u32) -> vec3f {
    for (var value = 0u; value < ${SUMMED_VALUES}u; value++) {
        giSkyTree[local] = vec4f(giSkySums[value], 0.0);
        workgroupBarrier();
        for (var stride = ${PROJECTION_WORKGROUP_SIZE / 2}u; stride > 0u; stride >>= 1u) {
            if (local < stride) {
                giSkyTree[local] += giSkyTree[local + stride];
            }
            workgroupBarrier();
        }
        if (local == 0u) {
            giSkyTotals[value] = giSkyTree[0];
        }
        // Thread 0 has read the root before the next value overwrites the tree.
        workgroupBarrier();
    }
    return select(vec3f(0.0), giSkyTotals[min(local, ${SUMMED_VALUES - 1}u)].xyz, local < ${SUMMED_VALUES}u);
}
`;
}

/** The two reduction kernels: partial sums over the grid, then their sum. */
function sumsWgsl(reduction: SkyReduction): string {
    const laneBuiltin = reduction === 'subgroups' ? ', @builtin(subgroup_invocation_id) lane: u32' : '';
    // The fallback has no lanes: its reduction is indexed by the thread's place in the workgroup.
    const laneValue = reduction === 'subgroups' ? 'lane' : '0u';
    const coneWeights = SH_COEFFICIENT_BANDS.map((band) => cosineLobeBandWeight(band).toPrecision(17)).join(', ');

    return /* wgsl */ `
// Each thread's running sums: the SH coefficients' radiance integrals, then the ground's irradiance.
var<private> giSkySums: array<vec3f, ${SUMMED_VALUES}>;

${reductionWgsl(reduction)}

// The clamped cosine's weight A_l for each coefficient's band (Ramamoorthi
// and Hanrahan 2001; spherical-harmonics.ts, cosineLobeBandWeight).
const GI_SKY_CONE_WEIGHTS = array<f32, ${SH_COEFFICIENT_COUNT}>(${coneWeights});

// Integrates radiance times basis times solid angle, and the ground's
// irradiance, over this workgroup's cells of the grid.
@compute @workgroup_size(${PROJECTION_WORKGROUP_SIZE})
fn giSkyPartialSums(@builtin(workgroup_id) group: vec3u, @builtin(local_invocation_index) local: u32${laneBuiltin}) {
    let isCube = params.gridKind == ${SKY_GRID_KIND.cube}u;
    let cellsPerFace = params.gridWidth * params.gridHeight;
    let cellCount = select(cellsPerFace, 6u * cellsPerFace, isCube);
    // Every thread runs the same number of iterations (a uniform value), so
    // the reduction after the loop is in uniform control flow. Neighbouring
    // threads take neighbouring cells, so texture reads stay together.
    for (var iteration = 0u; iteration < params.cellsPerThread; iteration++) {
        let cell = (group.x * params.cellsPerThread + iteration) * ${PROJECTION_WORKGROUP_SIZE}u + local;
        if (cell < cellCount) {
            var direction: vec3f;
            var solidAngle: f32;
            var groundWeight: f32;
            if (isCube) {
                let face = cell / cellsPerFace;
                let inFace = cell % cellsPerFace;
                let column = inFace % params.gridWidth;
                let row = inFace / params.gridWidth;
                direction = cubeCellDirection(face, column, row, params.gridWidth);
                solidAngle = cubeCellSolidAngle(column, row, params.gridWidth);
                groundWeight = solidAngle * max(direction.y, 0.0);
            } else {
                let column = cell % params.gridWidth;
                let row = cell / params.gridWidth;
                let uv = (vec2f(f32(column), f32(row)) + 0.5) / vec2f(f32(params.gridWidth), f32(params.gridHeight));
                direction = equirectDirection(uv);
                // A row of cells between latitudes a0 and a1 covers 2pi (sin a1 - sin a0); each cell its share.
                let sin0 = sin((f32(row) / f32(params.gridHeight) - 0.5) * PI);
                let sin1 = sin((f32(row + 1u) / f32(params.gridHeight) - 0.5) * PI);
                solidAngle = TWO_PI / f32(params.gridWidth) * (sin1 - sin0);
                // On open, flat ground a cell above the horizon weighs the integral of
                // cos(theta) over it, (2pi / width) (sin^2 a1 - sin^2 a0) / 2, as the CPU
                // sky's calibration does (equirect.ts, rowHorizontalIrradianceWeight).
                groundWeight = select(0.0, TWO_PI / f32(params.gridWidth) * 0.5 * (sin1 * sin1 - sin0 * sin0), sin0 >= 0.0);
            }
            let radiance = sourceRadiance(direction);
            let basis = giSkyShBasis(direction);
            for (var coefficient = 0u; coefficient < ${SH_COEFFICIENT_COUNT}u; coefficient++) {
                giSkySums[coefficient] += radiance * (basis[coefficient] * solidAngle);
            }
            giSkySums[${GROUND_IRRADIANCE_INDEX}u] += radiance * groundWeight;
        }
    }

    let total = giSkyReduce(local, ${laneValue});
    if (local < ${SUMMED_VALUES}u) {
        reduction.partials[group.x].values[local] = vec4f(total, 0.0);
    }
}

// The calibration scale, shared by the threads that write the harmonics.
var<workgroup> giSkyScale: f32;

// Adds the partial sums, works out the calibration scale, and writes the
// sky's harmonics and the scale the resample applies.
@compute @workgroup_size(${PROJECTION_WORKGROUP_SIZE})
fn giSkySum(@builtin(local_invocation_index) local: u32${laneBuiltin}) {
    for (var partial = local; partial < params.partialCount; partial += ${PROJECTION_WORKGROUP_SIZE}u) {
        for (var value = 0u; value < ${SUMMED_VALUES}u; value++) {
            giSkySums[value] += reduction.partials[partial].values[value].xyz;
        }
    }
    let total = giSkyReduce(local, ${laneValue});

    // A procedural sky is scaled so that open ground gets its target
    // irradiance, luminance-weighted (Rec.709 Y, as calibrated-sky.ts
    // measures it); every other source is used as it is.
    if (local == ${GROUND_IRRADIANCE_INDEX}u) {
        var scale = 1.0;
        if (params.sourceKind == ${SKY_SOURCE_KIND.procedural}u) {
            let groundLuminance = dot(total, vec3f(0.2126, 0.7152, 0.0722));
            scale = select(0.0, params.targetHorizontalLuminance / groundLuminance, groundLuminance > 0.0);
        }
        giSkyScale = scale;
        reduction.scale = vec4f(scale, 0.0, 0.0, 0.0);
    }
    workgroupBarrier();

    if (local < ${SH_COEFFICIENT_COUNT}u) {
        let radiance = total * giSkyScale;
        harmonics.irradiance[local] = vec4f(radiance * GI_SKY_CONE_WEIGHTS[local], 0.0);
        if (local < ${RADIANCE_SH_COUNT}u) {
            harmonics.radiance[local] = vec4f(radiance, 0.0);
        }
    }
}
`;
}

/** The whole bake shader (all three entry points) for one reduction. */
export function projectionWgsl(reduction: SkyReduction): string {
    const enable = reduction === 'subgroups' ? 'enable subgroups;\n' : '';
    return `${enable}${SHARED_WGSL}\n${SH_BASIS_WGSL}\n${RESAMPLE_WGSL}\n${sumsWgsl(reduction)}`;
}
