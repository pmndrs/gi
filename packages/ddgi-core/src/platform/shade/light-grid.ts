/**
 * The light grid (T2.3; docs/design/m2-lights.md §3): many-light sampling at
 * a fixed cost per hit, whatever the number of point and spot lights.
 *
 * **Why.** Below the grid, `giDirectIrradiance` loops over every local light
 * twice per hit: once to weigh them, once to pick `maxShadowedLightsPerHit`
 * (K) of them. With `sponza-candles`' 1,024 candles that cost about 7 ms at
 * 16 probes (T2.1), and `ddgi/trace`'s "can a light reach this hit" test
 * looped over all of them too. With the grid a hit reads R = 16 light picks
 * from its cell, weighs them, and traces shadow rays to K of them.
 *
 * **What a cell holds.** The grid is a box of cubic cells over the scene's
 * bounds (`lightGridLayout`), plus one cell for points outside it. Each frame
 * `gi/light-grid` rebuilds every cell from scratch:
 *
 * 1. every local light gets a weight for the cell: an upper bound of its
 *    unshadowed irradiance anywhere in the cell's box (the nearest point of
 *    the box for the distance falloff and range window, the cone's nearest
 *    angle to the box's bounding sphere for a spot; no cosine, since the
 *    surface isn't known). An upper bound, not the value at the centre,
 *    because a hit at a cell's edge can be lit by a light the centre can't
 *    see (out of range, outside the cone); every light that can light a
 *    point of the cell must have a weight above zero, or the estimate is
 *    biased;
 * 2. R picks are drawn from the cell's weights by systematic sampling: one
 *    random number u per cell and frame, picks at (r + u) / R of the
 *    weights' running total, r = 0 .. R - 1. A pick drawn with probability
 *    p = weight / total estimates the cell's light as f / p, so the R picks
 *    estimate it as the sum of f / (R p). A light picked m times stores its
 *    share m / (R p) on its first pick and 0 on the others, so a hit weighs
 *    it once.
 *
 * At a hit (`giDirectIrradiance` in wgsl.ts) this is ReGIR's second stage:
 * all R picks are weighed with the real distance, cone and cosine (no rays),
 * and the K shadow rays go to them in proportion to that weight, with the
 * same systematic choice the exact path uses over all lights. Each pick is a
 * fair draw on its own, whatever the others are, so the sum is unbiased; the
 * picks are stratified, so it is less noisy than independent draws.
 *
 * **Why R = 16, not the design's 8:** after the second stage the noise left
 * is the cell's own picks (K = 1 and K = 4 differ little). On the
 * `hit-shading-many-lights` test scene (1,024 shadowed point lights, 1,000
 * points, 1,024 frames, K = 4): R = 8 left 98.4% of points within 5% (the
 * done-when's bar is 99%), median noise 38% a frame; R = 16, 99.8% and 27%.
 *
 * **Why every light, not ReGIR's C = 16 candidates from a power CDF** (the
 * design's sketch): with 1,024 candles of the same power, 16 random
 * candidates per reservoir find the candle next to a hit 1.6% of the time,
 * so a point lit mostly by one nearby candle is very noisy. Weighing every
 * light per cell costs cells x lights evaluations (4,096 x 1,024 = 4.2 M a
 * frame on Sponza), which the GPU does in a few hundredths of a ms, and the
 * dominant candle is in the cell's picks almost every frame. The cost grows
 * with the light count; above some thousands of lights a light tree or
 * candidate subsets are the next step (DECISIONS 2026-11-06).
 *
 * **Why the grid is rebuilt every frame:** its picks are random. A grid kept
 * fixed would make the field converge to that one draw instead of the true
 * light. A new u per frame (from the platform's frame number) makes the
 * average over frames the real answer.
 *
 * **The build kernel:** a first dispatch packs each local light into a
 * dense record (`GiLightGridLight`); then one workgroup weighs 16 cells, 8
 * threads (runs) a cell, against tiles of 128 lights loaded into workgroup
 * memory, so a light is read once per workgroup, not once per cell. Reading
 * every light once per cell, the build was bound by those reads: 0.28 ms for
 * 4,096 cells and 1,024 lights on the RTX 3060. Each run sums its weights;
 * a scan of a cell's 8 run sums gives each run's place in the cell's total
 * (subgroups first, a segmented scan; the declared fallback adds them in
 * order, also taken when a device's subgroups don't hold whole cells of
 * consecutive threads, which WebGPU doesn't promise). Each pick finds the
 * run holding it by a binary search over the scanned sums, then each run's
 * thread weighs its lights again and hands out the picks that fall in them.
 * Every pick has exactly one writer, so the result does not depend on thread
 * timing.
 *
 * **Storage:** the picks are a `rgba32uint` texture (light index, share as
 * f32 bits; z and w unused), so the hit-shading module still binds no
 * storage buffer: `ddgi/trace` stays at 7 storage buffers and `gi/direct` at
 * 6. Not the 8-byte `rg32uint`: compatibility-mode devices refuse it as a
 * storage texture (`ddgi-shading-compat`), and 16 bytes a pick is about
 * 0.5 MB at 2,048 cells.
 */

import type { Aabb } from '../contract/math.ts';
import type { GpuTimerScope } from '../contract/timing.ts';
import type { ComputeKernel, KernelBuilder, WgslModule } from '../contract/wgsl.ts';

/** `GPUBufferUsage` / `GPUTextureUsage` / `GPUShaderStage` bits; the globals don't exist in Node, where tests import this file. */
const BUFFER_USAGE = { copyDst: 0x8, uniform: 0x40, storage: 0x80 } as const;
const TEXTURE_USAGE = { copySrc: 0x1, textureBinding: 0x4, storageBinding: 0x8 } as const;
const SHADER_STAGE_COMPUTE = 0x4;

/** Picks per cell (R). */
export const LIGHT_GRID_PICKS_PER_CELL = 16;
/** Threads a cell in `gi/light-grid` (its runs of the lights), and cells a workgroup. */
export const LIGHT_GRID_RUNS_PER_CELL = 8;
export const LIGHT_GRID_CELLS_PER_GROUP = 16;
/** Threads per `gi/light-grid` workgroup, and lights per tile it loads. */
export const LIGHT_GRID_WORKGROUP_SIZE = LIGHT_GRID_RUNS_PER_CELL * LIGHT_GRID_CELLS_PER_GROUP;
/** Texels per row of the picks texture; a cell's R picks sit in one row (R divides it). */
export const LIGHT_GRID_TEXTURE_WIDTH = 2048;
/** The most cells a grid may have (the design's cap), so a pick's texel row stays well inside the texture limits. */
export const LIGHT_GRID_MAX_CELLS = 32768;
/** Bytes of one pick in the texture (`rgba32uint`: light, share, two unused words). */
export const LIGHT_GRID_PICK_BYTES = 16;
/** A pick slot with no light (a cell no local light can reach). */
export const LIGHT_GRID_NO_LIGHT = 0xffffffff;
/** The timed pass. */
export const LIGHT_GRID_PASS = 'gi/light-grid';

/**
 * How cells are padded beyond the scene's bounds, as a share of the bounds'
 * largest side (plus 1 cm): hit points carry rounding error, and a point
 * just outside the grid would fall to the outside cell, whose weights are
 * coarse.
 */
const BOUNDS_PADDING_SHARE = 1e-3;
const BOUNDS_PADDING_METRES = 0.01;

export type LightGridReduction = 'subgroups' | 'workgroup';

/** Where the cells are: a box of `cells` cubic cells of side `cellSize` from `origin`. */
export interface LightGridLayout {
    readonly origin: readonly [number, number, number];
    readonly cellSize: number;
    readonly cells: readonly [number, number, number];
    /** cells x * y * z; the outside cell has this index. */
    readonly cellCount: number;
}

/**
 * Cubic cells over `bounds` (padded a little), as small as fits in
 * `maxCells`: the smallest side s with ceil(x / s) ceil(y / s) ceil(z / s) <=
 * maxCells, found by bisection (the product only shrinks as s grows). Flat
 * scenes get cells sized by their area, not a volume of zero.
 */
export function lightGridLayout(bounds: Aabb, maxCells: number): LightGridLayout {
    const cellBudget = Math.max(1, Math.min(LIGHT_GRID_MAX_CELLS, Math.floor(maxCells)));
    const largest = Math.max(bounds.max[0] - bounds.min[0], bounds.max[1] - bounds.min[1], bounds.max[2] - bounds.min[2], 0);
    const padding = largest * BOUNDS_PADDING_SHARE + BOUNDS_PADDING_METRES;
    const min = [bounds.min[0] - padding, bounds.min[1] - padding, bounds.min[2] - padding] as const;
    const size = [0, 1, 2].map((axis) => bounds.max[axis] - bounds.min[axis] + 2 * padding);
    const cellsFor = (side: number) => size.map((extent) => Math.max(1, Math.ceil(extent / side)));
    const count = (side: number) => cellsFor(side).reduce((product, cells) => product * cells, 1);

    // One cell of the largest side always fits; shrink until the next step would not.
    let fits = Math.max(...size);
    let tooSmall = fits / cellBudget;
    if (count(tooSmall) <= cellBudget) fits = tooSmall;
    else {
        for (let step = 0; step < 48; step++) {
            const middle = 0.5 * (fits + tooSmall);
            if (count(middle) <= cellBudget) fits = middle;
            else tooSmall = middle;
        }
    }
    const cells = cellsFor(fits) as [number, number, number];
    return { origin: [min[0], min[1], min[2]], cellSize: fits, cells, cellCount: cells[0] * cells[1] * cells[2] };
}

/** Rows of the picks texture for `cellCount` cells plus the outside cell. */
export function lightGridTextureRows(cellCount: number): number {
    return Math.max(1, Math.ceil(((cellCount + 1) * LIGHT_GRID_PICKS_PER_CELL) / LIGHT_GRID_TEXTURE_WIDTH));
}

/**
 * The lookup's uniforms, as the hit-shading module reads them
 * (`GiShadeUniforms`'s light-grid fields, wgsl.ts): 48 bytes.
 */
export interface LightGridLookup {
    /** 1 when this frame's direct light reads the grid (more local lights than `exactLightLimit`). */
    readonly active: boolean;
    readonly layout: LightGridLayout;
}

export const INACTIVE_LIGHT_GRID: LightGridLookup = {
    active: false,
    layout: { origin: [0, 0, 0], cellSize: 1, cells: [1, 1, 1], cellCount: 1 },
};

/** The WGSL the hit-shading module includes to read a cell's picks (its texture binding is the module's). */
export function lightGridLookupWgsl(picksVariable: string): string {
    return /* wgsl */ `
// ---- Light grid lookup (platform/shade/light-grid.ts) ----------------------------------

const GI_LIGHT_GRID_PICKS: u32 = ${LIGHT_GRID_PICKS_PER_CELL}u;
const GI_LIGHT_GRID_TEXTURE_WIDTH: u32 = ${LIGHT_GRID_TEXTURE_WIDTH}u;
const GI_LIGHT_GRID_NO_LIGHT: u32 = ${LIGHT_GRID_NO_LIGHT}u;

// One of a cell's picks: a light, and what its unshadowed contribution
// counts for at a hit in the cell: m / (R p) on the first of the m picks
// of a light drawn with probability p, 0 on the others.
struct GiLightGridPick {
    light: u32,         // GI_LIGHT_GRID_NO_LIGHT when no local light can reach the cell
    share: f32,
}

// The cell a point is in, or the outside cell (whose weights are the lights'
// brightest anywhere) for a point beyond the grid.
fn giLightGridCellAt(position: vec3f) -> u32 {
    let cells = giShade.lightGridCells;
    let local = (position - giShade.lightGridOrigin) * giShade.lightGridInverseCellSize;
    if (any(local < vec3f(0.0)) || any(local >= vec3f(cells))) {
        return giShade.lightGridOutsideCell;
    }
    let cell = min(vec3u(local), cells - vec3u(1u));
    return cell.x + cells.x * (cell.y + cells.y * cell.z);
}

fn giLightGridPick(cell: u32, slot: u32) -> GiLightGridPick {
    let texel = cell * GI_LIGHT_GRID_PICKS + slot;
    let stored = textureLoad(${picksVariable}, vec2u(texel % GI_LIGHT_GRID_TEXTURE_WIDTH, texel / GI_LIGHT_GRID_TEXTURE_WIDTH), 0).xy;
    return GiLightGridPick(stored.x, bitcast<f32>(stored.y));
}
`;
}

/** `gi/light-grid`'s own uniforms (group 1, binding 0, `GiLightGridBuild`): 48 bytes. */
export const LIGHT_GRID_BUILD_UNIFORM_BYTES = 48;

/**
 * The per-cell scan of its RUNS_PER_CELL runs' sums. Both leave the cell's
 * inclusive sums in `giGridRunEnds[cellSlot * RUNS ..]` (run r covers
 * [runEnds[r - 1], runEnds[r]) of the cell's total) and return after a
 * barrier. Every thread of the workgroup calls it, in uniform control flow.
 */
function scanWgsl(reduction: LightGridReduction): string {
    const serialScan = /* wgsl */ `
// The declared fallback: the first thread of each cell adds its cell's
// RUNS_PER_CELL sums in order, in workgroup memory.
fn giLightGridSerialScan(sum: f32, thread: u32) {
    giGridRunEnds[thread] = sum;
    workgroupBarrier();
    if (thread % GI_LIGHT_GRID_RUNS_PER_CELL == 0u) {
        var running = 0.0;
        for (var run = 0u; run < GI_LIGHT_GRID_RUNS_PER_CELL; run++) {
            running += giGridRunEnds[thread + run];
            giGridRunEnds[thread + run] = running;
        }
    }
    workgroupBarrier();
}
`;
    if (reduction === 'workgroup') {
        return /* wgsl */ `
${serialScan}
fn giLightGridScan(sum: f32, thread: u32, lane: u32, laneCount: u32, segmented: bool) {
    giLightGridSerialScan(sum, thread);
}
`;
    }
    return /* wgsl */ `
${serialScan}
// A segmented scan from one subgroup scan: each cell's runs are
// RUNS_PER_CELL consecutive lanes of one subgroup (checked on entry), so a
// run's sum within its cell is the subgroup's inclusive sum less the
// subgroup's sum before the cell's first lane.
fn giLightGridSubgroupScan(sum: f32, thread: u32, lane: u32) {
    let inclusive = subgroupInclusiveAdd(sum);
    let firstLaneOfCell = lane - lane % GI_LIGHT_GRID_RUNS_PER_CELL;
    let beforeCell = subgroupShuffle(inclusive - sum, firstLaneOfCell);
    giGridRunEnds[thread] = inclusive - beforeCell;
    workgroupBarrier();
}

fn giLightGridScan(sum: f32, thread: u32, lane: u32, laneCount: u32, segmented: bool) {
    if (segmented) {
        giLightGridSubgroupScan(sum, thread, lane);
    } else {
        giLightGridSerialScan(sum, thread);
    }
}
`;
}

/**
 * `gi/light-grid`: CELLS_PER_GROUP cells a workgroup (the outside cell
 * last), RUNS_PER_CELL threads a cell, and the lights loaded into workgroup
 * memory a tile at a time, so each light is read once per workgroup and not
 * once per cell. Group 0: the scene copy and the light falloff.
 */
export function lightGridBuildWgsl(reduction: LightGridReduction): string {
    const subgroups = reduction === 'subgroups';
    const builtins = subgroups ? '@builtin(subgroup_invocation_id) lane: u32, @builtin(subgroup_size) laneCount: u32,' : '';
    const laneArguments = subgroups ? 'lane, laneCount' : '0u, 1u';
    const segmentCheck = subgroups
        ? `// The segmented scan needs subgroups of whole cells holding consecutive
    // threads in order; WebGPU promises neither, so this workgroup takes the
    // serial scan when any subgroup breaks it.
    if (subgroupBroadcastFirst(thread) + lane != thread || laneCount % GI_LIGHT_GRID_RUNS_PER_CELL != 0u) {
        atomicStore(&giGridUnsegmented, 1u);
    }`
        : '';
    return /* wgsl */ `${subgroups ? 'enable subgroups;' : ''}
// ---- Light grid build (platform/shade/light-grid.ts) -----------------------------------

const GI_LIGHT_GRID_PICKS: u32 = ${LIGHT_GRID_PICKS_PER_CELL}u;
const GI_LIGHT_GRID_WORKGROUP: u32 = ${LIGHT_GRID_WORKGROUP_SIZE}u;
const GI_LIGHT_GRID_RUNS_PER_CELL: u32 = ${LIGHT_GRID_RUNS_PER_CELL}u;
const GI_LIGHT_GRID_CELLS_PER_GROUP: u32 = ${LIGHT_GRID_CELLS_PER_GROUP}u;
const GI_LIGHT_GRID_GROUP_PICKS: u32 = ${LIGHT_GRID_CELLS_PER_GROUP * LIGHT_GRID_PICKS_PER_CELL}u;
const GI_LIGHT_GRID_TEXTURE_WIDTH: u32 = ${LIGHT_GRID_TEXTURE_WIDTH}u;
const GI_LIGHT_GRID_NO_LIGHT: u32 = ${LIGHT_GRID_NO_LIGHT}u;
// Cells are weighed over a box this share of a cell larger on every side:
// a hit point's cell comes from rounded arithmetic, so it may lie a hair
// outside the box the build used. A larger box only raises the bound.
const GI_LIGHT_GRID_CELL_PADDING: f32 = 1.0 / 1024.0;

struct GiLightGridBuild {
    origin: vec3f,
    cellSize: f32,
    cells: vec3u,
    outsideCell: u32,
    seed: u32,          // the platform's frame number: new picks every frame
    reserved0: u32,
    reserved1: u32,
    reserved2: u32,
}

// What the weight needs of a point or spot light, packed once a frame by
// giLightGridPackLights, so the build reads a dense array and not the scene
// copy's records through its header.
struct GiLightGridLight {
    position: vec3f,
    inverseRangeSquared: f32,   // 1 / range^2, 0 for a light with no range
    brightness: f32,            // luminance of colour x intensity
    isSpot: u32,
    cosOuterCone: f32,
    cosInnerCone: f32,
    direction: vec3f,           // spot lights: the axis; read only for them
    reserved: f32,
}

@group(1) @binding(0) var<uniform> giGridBuild: GiLightGridBuild;
@group(1) @binding(1) var giGridPicksOut: texture_storage_2d<rgba32uint, write>;
// The local lights, packed: entry i is light firstLocal + i.
@group(1) @binding(2) var<storage, read_write> giGridLights: array<GiLightGridLight>;

// One tile of lights, shared by the workgroup's cells.
var<workgroup> giGridTile: array<GiLightGridLight, GI_LIGHT_GRID_WORKGROUP>;
// Per thread (cell slot x RUNS_PER_CELL + run): after the scan, where the run
// ends in its cell's running total.
var<workgroup> giGridRunEnds: array<f32, GI_LIGHT_GRID_WORKGROUP>;
// Per pick of the workgroup (cell slot x R + r): the run that hands it out,
// and how far into that run's weight it falls.
var<workgroup> giGridPickRun: array<u32, GI_LIGHT_GRID_GROUP_PICKS>;
var<workgroup> giGridPickOffset: array<f32, GI_LIGHT_GRID_GROUP_PICKS>;
var<workgroup> giGridUnsegmented: atomic<u32>;

// Rec. 709 luminance, as hit shading weighs lights (giShadeLuminance).
fn giLightGridLuminance(rgb: vec3f) -> f32 {
    return dot(rgb, vec3f(0.2126, 0.7152, 0.0722));
}

// An upper bound of a point or spot light's unshadowed irradiance (as
// luminance) anywhere in the box, or anywhere at all for the outside cell.
fn giLightGridWeight(light: GiLightGridLight, boxMin: vec3f, boxMax: vec3f, outside: bool) -> f32 {
    // The falloff and its range window only fall with distance, so the box's
    // nearest point bounds them; nowhere is nearer than distance 0 for the
    // outside cell. giShadeDistanceAttenuation's formula, from the squared
    // distance and with no branch: 1 / max(d^2, 0.01), times
    // (1 - (d / range)^4)^2 = (1 - (d^2 / range^2)^2)^2, which is 1 with no
    // range (1 / range^2 packed as 0). Written this way because the build is
    // bound by this arithmetic: it runs twice per light and cell. A rounding
    // step's difference from the hit's own falloff is covered by the cells'
    // padding (GI_LIGHT_GRID_CELL_PADDING).
    let toNearest = select(clamp(light.position, boxMin, boxMax) - light.position, vec3f(0.0), outside);
    let squaredDistance = dot(toNearest, toNearest);
    let ratioSquared = squaredDistance * light.inverseRangeSquared;
    let window = clamp(1.0 - ratioSquared * ratioSquared, 0.0, 1.0);
    var bound = light.brightness * window * window / max(squaredDistance, 0.01);
    if (outside) {
        return bound;
    }
    if (light.isSpot != 0u && bound > 0.0) {
        // The cone only falls with the angle to its axis, so the smallest angle
        // to any point of the box's bounding sphere bounds it.
        let centre = 0.5 * (boxMin + boxMax);
        let sphereRadius = 0.5 * length(boxMax - boxMin);
        let toCentre = centre - light.position;
        let centreDistance = length(toCentre);
        if (centreDistance > sphereRadius) {
            let axisAngle = acos(clamp(dot(toCentre / centreDistance, light.direction), -1.0, 1.0));
            let sphereAngle = asin(sphereRadius / centreDistance);
            let nearestAngle = max(axisAngle - sphereAngle, 0.0);
            bound *= giShadeSpotAttenuation(cos(nearestAngle), light.cosOuterCone, light.cosInnerCone);
        }
    }
    return bound;
}

// One thread per local light: packs what the weights need (GiLightGridLight).
@compute @workgroup_size(GI_LIGHT_GRID_WORKGROUP)
fn giLightGridPackLights(@builtin(global_invocation_id) id: vec3u) {
    let lightCount = giLightCount();
    let firstLocal = min(giDirectionalLightCount(), lightCount);
    let local = id.x;
    if (local >= lightCount - firstLocal || local >= arrayLength(&giGridLights)) {
        return;
    }
    let light = giLight(firstLocal + local);
    giGridLights[local] = GiLightGridLight(
        light.position,
        select(0.0, 1.0 / (light.range * light.range), light.range > 0.0),
        giLightGridLuminance(light.intensity),
        select(0u, 1u, light.kind == GI_LIGHT_SPOT),
        light.cosOuterCone,
        light.cosInnerCone,
        light.direction,
        0.0,
    );
}

// PCG hash (as hit shading's giShadeHash): the cell's random offset u for this frame.
fn giLightGridHash(value: u32) -> u32 {
    let state = value * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

// What a light picked 'picks' times out of R counts for at a hit: each pick,
// drawn with probability p = weight / total, estimates the cell's light as
// f / p, so R picks estimate it as the sum of f / (R p).
fn giLightGridShare(picks: u32, weight: f32, total: f32) -> f32 {
    return f32(picks) * total / (f32(GI_LIGHT_GRID_PICKS) * weight);
}

fn giLightGridStore(cell: u32, slot: u32, light: u32, share: f32) {
    let texel = cell * GI_LIGHT_GRID_PICKS + slot;
    textureStore(giGridPicksOut, vec2u(texel % GI_LIGHT_GRID_TEXTURE_WIDTH, texel / GI_LIGHT_GRID_TEXTURE_WIDTH), vec4u(light, bitcast<u32>(share), 0u, 0u));
}

// The first pick at or after 'start' that run 'run' hands out (R when none is left).
fn giLightGridNextPick(firstPickOfCell: u32, start: u32, run: u32) -> u32 {
    var pick = start;
    while (pick < GI_LIGHT_GRID_PICKS && giGridPickRun[firstPickOfCell + pick] != run) {
        pick++;
    }
    return pick;
}

// A pick's place within its run's weight; past every weight (3e38, near the largest f32) when no pick is left.
fn giLightGridPickPlace(firstPickOfCell: u32, pick: u32) -> f32 {
    if (pick >= GI_LIGHT_GRID_PICKS) {
        return 3.0e38;
    }
    return giGridPickOffset[firstPickOfCell + pick];
}

// Loads tile 'tileStart' of the packed lights into workgroup memory. Every
// thread calls it (it holds barriers): one light each.
fn giLightGridLoadTile(tileStart: u32, localCount: u32, thread: u32) {
    workgroupBarrier();
    if (tileStart + thread < localCount) {
        giGridTile[thread] = giGridLights[tileStart + thread];
    }
    workgroupBarrier();
}

${scanWgsl(reduction)}

@compute @workgroup_size(GI_LIGHT_GRID_WORKGROUP)
fn giLightGridBuildCell(
    @builtin(workgroup_id) group: vec3u,
    @builtin(local_invocation_index) thread: u32,
    ${builtins}
) {
    // This thread's cell, and its run of that cell's lights: every
    // RUNS_PER_CELL-th light of each tile, from its own. The runs tile the
    // cell's total weight in run order; any fixed order serves systematic
    // sampling.
    let cellSlot = thread / GI_LIGHT_GRID_RUNS_PER_CELL;
    let run = thread % GI_LIGHT_GRID_RUNS_PER_CELL;
    let cell = group.x * GI_LIGHT_GRID_CELLS_PER_GROUP + cellSlot;
    // The last workgroup may hold slots past the outside cell: they take part
    // in the barriers and write nothing.
    let inGrid = cell <= giGridBuild.outsideCell;
    let outside = cell == giGridBuild.outsideCell;
    let cells = giGridBuild.cells;
    let coordinates = vec3u(cell % cells.x, (cell / cells.x) % cells.y, cell / (cells.x * cells.y));
    let padding = giGridBuild.cellSize * GI_LIGHT_GRID_CELL_PADDING;
    let boxMin = giGridBuild.origin + vec3f(coordinates) * giGridBuild.cellSize - vec3f(padding);
    let boxMax = giGridBuild.origin + vec3f(coordinates + vec3u(1u)) * giGridBuild.cellSize + vec3f(padding);

    // giGridUnsegmented starts at 0: WGSL zero-initializes workgroup memory.
    ${segmentCheck}
    let segmented = workgroupUniformLoad(&giGridUnsegmented) == 0u;

    // Directional lights come first and are always evaluated at the hit, so
    // they are not in the grid. Both counts come from read-only storage, so
    // the tile loops are uniform and may hold barriers.
    let lightCount = giLightCount();
    let firstLocal = min(giDirectionalLightCount(), lightCount);
    let localCount = min(lightCount - firstLocal, arrayLength(&giGridLights));

    // Pass 1: the run's total weight.
    var runSum = 0.0;
    for (var tileStart = 0u; tileStart < localCount; tileStart += GI_LIGHT_GRID_WORKGROUP) {
        giLightGridLoadTile(tileStart, localCount, thread);
        for (var inTile = run; inTile < GI_LIGHT_GRID_WORKGROUP && tileStart + inTile < localCount; inTile += GI_LIGHT_GRID_RUNS_PER_CELL) {
            runSum += giLightGridWeight(giGridTile[inTile], boxMin, boxMax, outside);
        }
    }
    giLightGridScan(runSum, thread, ${laneArguments}, segmented);

    // Each pick finds its run: the first run whose end is above the pick's
    // place (r + u) / R x total. The search keeps runEnds[t - 1] <= place <
    // runEnds[t] at every step, so it ends on one run even if the scan's
    // rounding made the sums dip by an ulp somewhere. A cell no light can
    // reach (total 0) hands out no picks.
    for (var pick = thread; pick < GI_LIGHT_GRID_GROUP_PICKS; pick += GI_LIGHT_GRID_WORKGROUP) {
        let pickCell = pick / GI_LIGHT_GRID_PICKS;
        let slot = pick % GI_LIGHT_GRID_PICKS;
        let firstRun = pickCell * GI_LIGHT_GRID_RUNS_PER_CELL;
        let total = giGridRunEnds[firstRun + GI_LIGHT_GRID_RUNS_PER_CELL - 1u];
        let u = f32(giLightGridHash((group.x * GI_LIGHT_GRID_CELLS_PER_GROUP + pickCell) ^ giLightGridHash(giGridBuild.seed)) >> 8u) * (1.0 / 16777216.0);
        let place = (f32(slot) + u) / f32(GI_LIGHT_GRID_PICKS) * total;
        var low = 0u;
        var high = GI_LIGHT_GRID_RUNS_PER_CELL - 1u;
        while (low < high) {
            let middle = (low + high) / 2u;
            if (giGridRunEnds[firstRun + middle] > place) {
                high = middle;
            } else {
                low = middle + 1u;
            }
        }
        let runBegins = select(0.0, giGridRunEnds[firstRun + max(low, 1u) - 1u], low > 0u);
        // No run hands out the picks of an empty cell, or of a slot past the outside cell.
        let pickInGrid = group.x * GI_LIGHT_GRID_CELLS_PER_GROUP + pickCell <= giGridBuild.outsideCell;
        giGridPickRun[pick] = select(GI_LIGHT_GRID_RUNS_PER_CELL, low, total > 0.0 && pickInGrid);
        giGridPickOffset[pick] = max(place - runBegins, 0.0);
    }
    workgroupBarrier();

    let cellTotal = giGridRunEnds[cellSlot * GI_LIGHT_GRID_RUNS_PER_CELL + GI_LIGHT_GRID_RUNS_PER_CELL - 1u];
    if (inGrid && !(cellTotal > 0.0) && run == 0u) {
        // No local light can reach any point of this cell.
        for (var slot = 0u; slot < GI_LIGHT_GRID_PICKS; slot++) {
            giLightGridStore(cell, slot, GI_LIGHT_GRID_NO_LIGHT, 0.0);
        }
    }

    // Pass 2: each run hands out its picks, in order of place, to the light
    // whose share of the run holds the place. A light that gets m picks
    // stores its share m / (R p) on the first of them and 0 on the rest, so
    // a hit weighs it once (giShadeLocalCandidate).
    let firstPickOfCell = cellSlot * GI_LIGHT_GRID_PICKS;
    var nextPick = giLightGridNextPick(firstPickOfCell, 0u, run);
    // The next pick's place, kept in a register: pass 2 compares it with every light.
    var nextPlace = giLightGridPickPlace(firstPickOfCell, nextPick);
    var running = 0.0;
    var lastLit = GI_LIGHT_GRID_NO_LIGHT;
    var lastWeight = 0.0;
    for (var tileStart = 0u; tileStart < localCount; tileStart += GI_LIGHT_GRID_WORKGROUP) {
        giLightGridLoadTile(tileStart, localCount, thread);
        for (var inTile = run; inTile < GI_LIGHT_GRID_WORKGROUP && tileStart + inTile < localCount && nextPick < GI_LIGHT_GRID_PICKS; inTile += GI_LIGHT_GRID_RUNS_PER_CELL) {
            let weight = giLightGridWeight(giGridTile[inTile], boxMin, boxMax, outside);
            if (!(weight > 0.0)) {
                continue;
            }
            let index = firstLocal + tileStart + inTile;
            running += weight;
            lastLit = index;
            lastWeight = weight;
            var firstPick = 0u;
            var picked = 0u;
            while (nextPlace < running) {
                if (picked == 0u) {
                    firstPick = nextPick;
                } else {
                    giLightGridStore(cell, nextPick, index, 0.0);
                }
                picked++;
                nextPick = giLightGridNextPick(firstPickOfCell, nextPick + 1u, run);
                nextPlace = giLightGridPickPlace(firstPickOfCell, nextPick);
            }
            if (picked > 0u) {
                giLightGridStore(cell, firstPick, index, giLightGridShare(picked, weight, cellTotal));
            }
        }
    }
    // Picks past the run's last light by rounding (the scan's sum and this
    // thread's running sum add in a different order) go to that light, each
    // counting as one pick of its own.
    while (inGrid && nextPick < GI_LIGHT_GRID_PICKS) {
        if (giGridPickRun[firstPickOfCell + nextPick] == run) {
            giLightGridStore(cell, nextPick, lastLit, select(0.0, giLightGridShare(1u, lastWeight, cellTotal), lastLit != GI_LIGHT_GRID_NO_LIGHT));
        }
        nextPick++;
    }
}
`;
}

/** `gi/light-grid` group 1: its uniforms and the picks texture as a storage texture. */
const BUILD_GROUP: GPUBindGroupLayoutDescriptor = {
    label: 'gi/light-grid group 1',
    entries: [
        { binding: 0, visibility: SHADER_STAGE_COMPUTE, buffer: { type: 'uniform' } },
        { binding: 1, visibility: SHADER_STAGE_COMPUTE, storageTexture: { access: 'write-only', format: 'rgba32uint', viewDimension: '2d' } },
        { binding: 2, visibility: SHADER_STAGE_COMPUTE, buffer: { type: 'storage' } },
    ],
};

/** Bytes of one packed light (`GiLightGridLight`). */
export const LIGHT_GRID_PACKED_LIGHT_BYTES = 48;

/** What the grid needs each frame. */
export interface LightGridFrame {
    /** The platform's frame number: the picks' random offsets. */
    readonly frame: number;
    /** The scene's bounds (every instance rays see). */
    readonly bounds: Aabb;
    /** Point and spot lights in the scene. */
    readonly localLights: number;
    /** From this many local lights plus one, direct light reads the grid. */
    readonly exactLightLimit: number;
    /** The most cells the grid may use. */
    readonly maxCells: number;
}

export interface PlatformLightGridOptions {
    readonly label?: string;
    /** Which scan to build: `subgroups` when the device has them (the default), or the declared fallback. */
    readonly reduction?: LightGridReduction;
}

/**
 * The grid's GPU side: the picks texture (bound by the hit-shading module
 * as a sampled texture) and `gi/light-grid`, which rewrites it each frame
 * the grid is in use.
 */
export class PlatformLightGrid {
    readonly reduction: LightGridReduction;

    private readonly device: GPUDevice;
    private readonly kernelBuilder: KernelBuilder;
    private readonly sceneModule: WgslModule;
    private readonly falloffModule: WgslModule;
    private readonly label: string;
    private readonly buildUniforms: GPUBuffer;
    private readonly buildWords = new ArrayBuffer(LIGHT_GRID_BUILD_UNIFORM_BYTES);
    private kernels: { readonly pack: ComputeKernel; readonly build: ComputeKernel } | null = null;
    /** The packed local lights (`GiLightGridLight`); grows by doubling with the scene's point and spot lights. */
    private packedLights: GPUBuffer | null = null;
    private preparing: Promise<void> | null = null;
    private picks: GPUTexture;
    private picksView: GPUTextureView;
    private buildGroup: GPUBindGroup | null = null;
    private textureVersion = 0;
    private lookup: LightGridLookup = INACTIVE_LIGHT_GRID;
    private layoutKey = '';
    private disposed = false;

    constructor(device: GPUDevice, kernels: KernelBuilder, sceneModule: WgslModule, falloffModule: WgslModule, options: PlatformLightGridOptions = {}) {
        this.device = device;
        this.kernelBuilder = kernels;
        this.sceneModule = sceneModule;
        this.falloffModule = falloffModule;
        this.label = options.label ?? 'gi';
        this.reduction = options.reduction ?? (device.features.has('subgroups') ? 'subgroups' : 'workgroup');
        if (this.reduction === 'subgroups' && !device.features.has('subgroups')) {
            throw new Error("@pmndrs/ddgi: the light grid's subgroup scan needs a device with the 'subgroups' feature");
        }
        this.buildUniforms = device.createBuffer({ label: `${this.label} light grid build`, size: LIGHT_GRID_BUILD_UNIFORM_BYTES, usage: BUFFER_USAGE.uniform | BUFFER_USAGE.copyDst });
        // One texel until a scene needs the grid: scenes with a few lights pay nothing for it.
        this.picks = this.createPicks(1, 1);
        this.picksView = this.picks.createView({ label: `${this.label} light grid picks` });
    }

    /** The picks texture, as the hit-shading module binds it. Changes when the grid grows (see {@link version}). */
    get view(): GPUTextureView {
        return this.picksView;
    }

    /** Bumped when the picks texture is reallocated. */
    get version(): number {
        return this.textureVersion;
    }

    /** What the hit-shading module's uniforms should say for the frame last encoded. */
    get current(): LightGridLookup {
        return this.lookup;
    }

    get memoryBytes(): number {
        return this.picks.width * this.picks.height * LIGHT_GRID_PICK_BYTES + LIGHT_GRID_BUILD_UNIFORM_BYTES + (this.packedLights?.size ?? 0);
    }

    /** The picks texture, for tests that read it back (it has COPY_SRC). */
    get texture(): GPUTexture {
        return this.picks;
    }

    prepare(): Promise<void> {
        this.preparing ??= (async () => {
            const options = { modules: [this.sceneModule, this.falloffModule], code: lightGridBuildWgsl(this.reduction), ownGroups: [BUILD_GROUP] };
            const [pack, build] = await Promise.all([
                this.kernelBuilder.computeKernel({ ...options, label: 'gi/light-grid pack', entryPoint: 'giLightGridPackLights' }),
                this.kernelBuilder.computeKernel({ ...options, label: `gi/light-grid (${this.reduction})`, entryPoint: 'giLightGridBuildCell' }),
            ]);
            this.kernels = { pack, build };
        })();
        return this.preparing;
    }

    /**
     * Decides whether this frame's direct light reads the grid and, if so,
     * records `gi/light-grid` into the encoder. Returns the lookup the
     * hit-shading module's uniforms must carry this frame.
     */
    encode(encoder: GPUCommandEncoder, frame: LightGridFrame, timer?: GpuTimerScope): LightGridLookup {
        if (this.disposed) throw new Error('@pmndrs/ddgi: the light grid was disposed');
        const kernels = this.kernels;
        // Until the kernels have compiled, direct light keeps the exact path (correct, only slower).
        if (!kernels || !(frame.localLights > frame.exactLightLimit)) {
            this.lookup = INACTIVE_LIGHT_GRID;
            return this.lookup;
        }
        const layout = this.layoutFor(frame.bounds, frame.maxCells);
        const rows = lightGridTextureRows(layout.cellCount);
        // The texture only grows, so bounds that change from frame to frame (moving objects) don't reallocate it.
        if (rows > this.picks.height || this.picks.width !== LIGHT_GRID_TEXTURE_WIDTH) this.growPicks(rows);
        const packedBytes = frame.localLights * LIGHT_GRID_PACKED_LIGHT_BYTES;
        if (!this.packedLights || this.packedLights.size < packedBytes) {
            this.packedLights?.destroy();
            const capacity = Math.max(64, 2 ** Math.ceil(Math.log2(frame.localLights)));
            this.packedLights = this.device.createBuffer({ label: `${this.label} light grid lights`, size: capacity * LIGHT_GRID_PACKED_LIGHT_BYTES, usage: BUFFER_USAGE.storage });
            this.buildGroup = null;
        }
        // The two kernels share one group layout (the same code, two entry points).
        this.buildGroup ??= this.device.createBindGroup({
            label: `${this.label} light grid build`,
            layout: kernels.build.ownLayouts[0],
            entries: [
                { binding: 0, resource: { buffer: this.buildUniforms } },
                { binding: 1, resource: this.picksView },
                { binding: 2, resource: { buffer: this.packedLights } },
            ],
        });
        const packGroup = this.device.createBindGroup({
            label: `${this.label} light grid pack`,
            layout: kernels.pack.ownLayouts[0],
            entries: [
                { binding: 0, resource: { buffer: this.buildUniforms } },
                { binding: 1, resource: this.picksView },
                { binding: 2, resource: { buffer: this.packedLights } },
            ],
        });

        const floats = new Float32Array(this.buildWords);
        const words = new Uint32Array(this.buildWords);
        floats.set(layout.origin, 0);
        floats[3] = layout.cellSize;
        words.set(layout.cells, 4);
        // The outside cell's index is the cell count.
        words[7] = layout.cellCount;
        words[8] = frame.frame >>> 0;
        this.device.queue.writeBuffer(this.buildUniforms, 0, this.buildWords);

        const pass = encoder.beginComputePass({ label: `${this.label} ${LIGHT_GRID_PASS}`, timestampWrites: timer?.computePass(LIGHT_GRID_PASS) });
        // Pack the lights, then weigh them per cell: each dispatch sees the one before it.
        kernels.pack.bind(pass);
        pass.setBindGroup(1, packGroup);
        pass.dispatchWorkgroups(Math.ceil(frame.localLights / LIGHT_GRID_WORKGROUP_SIZE));
        kernels.build.bind(pass);
        pass.setBindGroup(1, this.buildGroup);
        // CELLS_PER_GROUP cells a workgroup, the outside cell last (at most 2,049 workgroups).
        pass.dispatchWorkgroups(Math.ceil((layout.cellCount + 1) / LIGHT_GRID_CELLS_PER_GROUP));
        pass.end();

        this.lookup = { active: true, layout };
        return this.lookup;
    }

    dispose(): void {
        if (this.disposed) return;
        this.disposed = true;
        this.picks.destroy();
        this.buildUniforms.destroy();
        this.packedLights?.destroy();
    }

    /** The layout for these bounds, reusing the last one while the bounds and the cap are the same. */
    private layoutFor(bounds: Aabb, maxCells: number): LightGridLayout {
        const key = `${bounds.min.join(',')}/${bounds.max.join(',')}/${maxCells}`;
        if (key !== this.layoutKey || !this.lookup.active) {
            this.layoutKey = key;
            return lightGridLayout(bounds, maxCells);
        }
        return this.lookup.layout;
    }

    private createPicks(width: number, rows: number): GPUTexture {
        return this.device.createTexture({
            label: `${this.label} light grid picks`,
            size: { width, height: rows },
            format: 'rgba32uint',
            // COPY_SRC so a test can read the picks back.
            usage: TEXTURE_USAGE.textureBinding | TEXTURE_USAGE.storageBinding | TEXTURE_USAGE.copySrc,
        });
    }

    private growPicks(rows: number): void {
        this.picks.destroy();
        this.picks = this.createPicks(LIGHT_GRID_TEXTURE_WIDTH, rows);
        this.picksView = this.picks.createView({ label: `${this.label} light grid picks` });
        this.buildGroup = null;
        this.textureVersion++;
    }
}
