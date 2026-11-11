/**
 * A CPU twin of `FilterScreenProbes` (gi1.comp:1356-1443; the WGSL is in
 * wgsl/screen-irradiance.ts), in f64, for the GPU checks (RC4.2): both
 * passes as Capsaicin runs them (gi1.cpp:2588-2614), on an atlas, a probe
 * list, the probe mask's mips and the screen the GPU read.
 *
 * Each decision the kernel makes (the plane test, the hemisphere test, the
 * neighbour's cell, the reprojected angle) is a comparison of f32 values on
 * the GPU; where the twin's f64 value sits within a small margin of the
 * threshold, the cell is marked undecided rather than compared, and a
 * second-pass cell that read an undecided first-pass cell is undecided too.
 */

import { getOrthoVectors, halfBitsToFloat, floatToHalfBits, INVALID_ID, mapToHemiOctahedron, mapToHemiOctahedronInverse, nearestHalf, toProbeFrame, unpackSeed, type Direction } from './capsaicin-math.ts';
import { PROBE_SIZE } from './settings.ts';
import { BLUR_DIRECTIONS, FILTER_ANGLE_THRESHOLD, FILTER_RADIUS } from './wgsl/screen-irradiance.ts';

type Vector = [number, number, number];

/** How close to a threshold a value may be and still be compared: past these the cell is undecided. */
export const FILTER_DECISION_MARGINS = {
    /** The plane test, relative to the cell size. */
    plane: 1e-3,
    /** The hemisphere test's cosine. */
    hemisphere: 1e-4,
    /** The neighbour's cell, in cells (the map's square times 8). */
    cell: 1e-3,
    /** The reprojected angle's cosine. */
    angle: 1e-4,
} as const;

/** The screen and probes the filter reads, as the GPU had them. */
export interface ProbeFilterInputs {
    readonly width: number;
    readonly height: number;
    /** The host's depth, one per pixel (`r32float`). */
    readonly depth: Float32Array;
    /** The host's normals, signed, four per pixel (rgb used), not yet normalised. */
    readonly normal: Float32Array;
    /** Each probe-mask level, `r32uint`, with its size. */
    readonly maskLevels: readonly { readonly width: number; readonly height: number; readonly texels: Uint32Array }[];
    /** The listed probes' packed seeds (`RcProbeSpawn.seeds`, the first `count`). */
    readonly seeds: readonly number[];
    /** The camera-relative inverse view-projection (`cameraRelativeInverseViewProjection`), the eye, the view axis, Capsaicin's cell size per unit distance. */
    readonly inverseViewProjection: readonly number[];
    readonly eye: Direction;
    readonly viewForward: Direction;
    readonly cellSize: number;
}

/** An atlas: four values a texel (radiance, hit distance), `width` texels a row. */
export interface ProbeAtlas {
    readonly width: number;
    readonly height: number;
    readonly texels: Float32Array;
}

export interface ProbeFilterResult {
    /** The atlas after both passes (unlisted tiles as the input had them). */
    readonly atlas: ProbeAtlas;
    /** One flag per atlas texel: 1 where a decision sat within its margin, so the GPU may have taken the other side. */
    readonly undecided: Uint8Array;
    /** Neighbours accepted in each pass, over every listed cell (to show the filter did something). */
    readonly accepted: readonly [number, number];
}

const dot = (a: Direction, b: Direction): number => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const subtract = (a: Direction, b: Direction): Vector => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const length = (a: Direction): number => Math.hypot(a[0], a[1], a[2]);
const normalize = (a: Direction): Vector => {
    const size = length(a);
    return [a[0] / size, a[1] / size, a[2] / size];
};

/** A value stored to `rgba16float`, rounded to nearest (the GPU may round another way within a step). */
export function toHalf(value: number): number {
    return halfBitsToFloat(floatToHalfBits(value));
}

/** `GIDenoiser_RemoveNaNs` (gi_denoiser.hlsl:75-82) on one channel. */
function removeNaNs(value: number): number {
    let c = value / (1 + value);
    c = Math.min(Math.max(c, 0), 1);
    return c / Math.max(1 - c, 1e-4);
}

/** A probe-frame direction in world space (`rcFromProbeFrame`). */
function fromProbeFrame(direction: Direction, normal: Direction): Vector {
    const { b1, b2 } = getOrthoVectors(normal);
    return [0, 1, 2].map((axis) => direction[0] * b1[axis] + direction[1] * b2[axis] + direction[2] * normal[axis]) as Vector;
}

class FilterScreen {
    constructor(private readonly inputs: ProbeFilterInputs) {}

    normalAt(x: number, y: number): Vector {
        const index = 4 * (x + y * this.inputs.width);
        const n = this.inputs.normal;
        return normalize([n[index], n[index + 1], n[index + 2]]);
    }

    /** `rcWorldPositionAt`: the point the depth gives at a pixel's centre. */
    worldAt(x: number, y: number): Vector {
        const { width, height, inverseViewProjection: m, eye } = this.inputs;
        const u = (x + 0.5) / width;
        const v = (y + 0.5) / height;
        const ndc = [2 * u - 1, 2 * (1 - v) - 1, this.inputs.depth[x + y * width], 1];
        const h = [0, 1, 2, 3].map((row) => m[row] * ndc[0] + m[4 + row] * ndc[1] + m[8 + row] * ndc[2] + m[12 + row] * ndc[3]);
        return [eye[0] + h[0] / h[3], eye[1] + h[1] / h[3], eye[2] + h[2] / h[3]];
    }

    linearDepth(world: Direction): number {
        return dot(subtract(world, this.inputs.eye), this.inputs.viewForward);
    }

    /** `ScreenProbes_FindClosestProbe(pos, offset)` (screen_probes.hlsl:137-166). */
    findClosestProbeWithOffset(x: number, y: number, offset: readonly [number, number]): number {
        const levels = this.inputs.maskLevels;
        let dims: [number, number] = [levels[0].width, levels[0].height];
        let pos: [number, number] = [Math.min(Math.floor(x / PROBE_SIZE), dims[0] - 1), Math.min(Math.floor(y / PROBE_SIZE), dims[1] - 1)];
        for (let level = 0; level < levels.length; level++) {
            const loc = [pos[0] + offset[0], pos[1] + offset[1]];
            if (loc[0] < 0 || loc[1] < 0 || loc[0] >= dims[0] || loc[1] >= dims[1]) break;
            const probe = levels[level].texels[loc[0] + loc[1] * levels[level].width];
            if (probe !== INVALID_ID) return probe;
            dims = [Math.max(dims[0] >> 1, 1), Math.max(dims[1] >> 1, 1)];
            pos = [Math.min(pos[0] >> 1, dims[0] - 1), Math.min(pos[1] >> 1, dims[1] - 1)];
        }
        return INVALID_ID;
    }
}

/**
 * One pass of `FilterScreenProbes` along `direction` for every listed probe
 * cell: `input` is the atlas the pass reads, `output` (a copy of what the
 * pass's target held) gets the listed cells. Returns the neighbours accepted.
 */
function filterPass(
    screen: FilterScreen,
    inputs: ProbeFilterInputs,
    direction: readonly [number, number],
    input: ProbeAtlas,
    inputUndecided: Uint8Array,
    output: Float32Array,
    outputUndecided: Uint8Array,
    roundOutput: boolean,
): number {
    let accepted = 0;
    const read = (x: number, y: number): [number, number, number, number] => {
        const index = 4 * (x + y * input.width);
        return [input.texels[index], input.texels[index + 1], input.texels[index + 2], input.texels[index + 3]];
    };
    for (const packedSeed of inputs.seeds) {
        const [seedX, seedY] = unpackSeed(packedSeed);
        const normal = screen.normalAt(seedX, seedY);
        const world = screen.worldAt(seedX, seedY);
        const cellSize = length(subtract(world, inputs.eye)) * inputs.cellSize;
        const linearDepth = screen.linearDepth(world);
        const tileX = Math.floor(seedX / PROBE_SIZE) * PROBE_SIZE;
        const tileY = Math.floor(seedY / PROBE_SIZE) * PROBE_SIZE;
        for (let cell = 0; cell < PROBE_SIZE * PROBE_SIZE; cell++) {
            const cellX = cell % PROBE_SIZE;
            const cellY = Math.floor(cell / PROBE_SIZE);
            const posX = tileX + cellX;
            const posY = tileY + cellY;
            let undecided = inputUndecided[posX + posY * input.width] === 1;
            const radiance = read(posX, posY);
            const cellDirection = fromProbeFrame(mapToHemiOctahedron((cellX + 0.5) / PROBE_SIZE, (cellY + 0.5) / PROBE_SIZE), normal);
            let hitDistance = radiance[3];
            let totalWeight = 1;
            for (let i = 0; i < FILTER_RADIUS << 1; i++) {
                const step = ((i & 1) * 2 - 1) * ((i >> 1) + 1);
                const probeMask = screen.findClosestProbeWithOffset(seedX, seedY, [step * direction[0], step * direction[1]]);
                if (probeMask === INVALID_ID) continue;
                const [probeX, probeY] = unpackSeed(probeMask);
                const probeNormal = screen.normalAt(probeX, probeY);
                const probeWorld = screen.worldAt(probeX, probeY);
                const plane = Math.abs(dot(subtract(probeWorld, world), normal));
                const hemisphere = dot(cellDirection, probeNormal);
                if (Math.abs(plane - cellSize) < FILTER_DECISION_MARGINS.plane * cellSize || Math.abs(hemisphere) < FILTER_DECISION_MARGINS.hemisphere) undecided = true;
                if (plane > cellSize || hemisphere < 0) continue;

                // The neighbour's cell, kept inside its probe (register D51).
                const [u, v] = mapToHemiOctahedronInverse(toProbeFrame(cellDirection, probeNormal));
                const nearEdge = (value: number) => Math.abs(value - Math.round(value)) < FILTER_DECISION_MARGINS.cell && value > 0 && value < PROBE_SIZE;
                if (nearEdge(u * PROBE_SIZE) || nearEdge(v * PROBE_SIZE)) undecided = true;
                const probeCellX = Math.min(Math.trunc(u * PROBE_SIZE), PROBE_SIZE - 1);
                const probeCellY = Math.min(Math.trunc(v * PROBE_SIZE), PROBE_SIZE - 1);
                const probePosX = Math.floor(probeX / PROBE_SIZE) * PROBE_SIZE + probeCellX;
                const probePosY = Math.floor(probeY / PROBE_SIZE) * PROBE_SIZE + probeCellY;
                const probeDirection = normalize(fromProbeFrame(mapToHemiOctahedron((probeCellX + 0.5) / PROBE_SIZE, (probeCellY + 0.5) / PROBE_SIZE), probeNormal));

                const neighbour = read(probePosX, probePosY);
                if (inputUndecided[probePosX + probePosY * input.width] === 1) undecided = true;
                const probeHitDistance = Math.min(neighbour[3], hitDistance);
                const hitPoint: Vector = [0, 1, 2].map((axis) => probeWorld[axis] + probeDirection[axis] * probeHitDistance) as Vector;
                const angle = dot(cellDirection, normalize(subtract(hitPoint, world)));
                if (Math.abs(angle - FILTER_ANGLE_THRESHOLD) < FILTER_DECISION_MARGINS.angle) undecided = true;
                if (angle < FILTER_ANGLE_THRESHOLD) continue;

                const weight = Math.min(Math.max(1 - Math.abs(screen.linearDepth(probeWorld) - linearDepth) / linearDepth, 0), 1) ** 8;
                radiance[0] += weight * removeNaNs(neighbour[0]);
                radiance[1] += weight * removeNaNs(neighbour[1]);
                radiance[2] += weight * removeNaNs(neighbour[2]);
                radiance[3] += weight * probeHitDistance;
                totalWeight += weight;
                hitDistance = radiance[3] / totalWeight;
                accepted++;
            }
            const index = 4 * (posX + posY * input.width);
            for (let channel = 0; channel < 4; channel++) {
                const value = radiance[channel] / totalWeight;
                // The kernel rounds to the nearest half before its store (rcNearestHalf, register D59).
                output[index + channel] = roundOutput ? toHalf(nearestHalf(value)) : value;
            }
            outputUndecided[posX + posY * input.width] = undecided ? 1 : 0;
        }
    }
    return accepted;
}

/**
 * Both passes, as the frame runs them: the atlas copied into the previous
 * atlas (gi1.cpp:2540-2541), across the screen from the atlas into it, then
 * down it from there back into the atlas (gi1.cpp:2597-2611). The first
 * pass's results are stored as halves before the second reads them; the
 * second's are returned unrounded. `atlasUndecided` (one flag per texel)
 * marks input texels the caller isn't sure of: any result that reads one is
 * undecided too.
 */
export function filterScreenProbes(inputs: ProbeFilterInputs, atlas: ProbeAtlas, atlasUndecided?: Uint8Array): ProbeFilterResult {
    const screen = new FilterScreen(inputs);
    const texelCount = atlas.width * atlas.height;
    const noneUndecided = atlasUndecided ?? new Uint8Array(texelCount);
    const firstOutput = Float32Array.from(atlas.texels);
    const firstUndecided = new Uint8Array(texelCount);
    const across = filterPass(screen, inputs, BLUR_DIRECTIONS[0], atlas, noneUndecided, firstOutput, firstUndecided, true);
    const first: ProbeAtlas = { width: atlas.width, height: atlas.height, texels: firstOutput };
    const secondOutput = Float32Array.from(atlas.texels);
    const undecided = new Uint8Array(texelCount);
    const down = filterPass(screen, inputs, BLUR_DIRECTIONS[1], first, firstUndecided, secondOutput, undecided, false);
    return { atlas: { width: atlas.width, height: atlas.height, texels: secondOutput }, undecided, accepted: [across, down] };
}
