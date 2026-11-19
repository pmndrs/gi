/**
 * T1.5b: probe classification and relocation, checked against the geometry.
 *
 * On the point-light Cornell box (two solid blocks) and the cabinet row
 * (hollow 18 mm boards), DDGI runs on the manifest's volume, every probe
 * every frame, for FRAMES frames. Then every probe's offset and state are
 * read back and judged on the CPU from the scene's own triangles:
 *
 * 1. a probe whose final position lies inside a closed solid reads state 1
 *    ("off: inside geometry"); an active probe inside a solid would shade
 *    surfaces with black. Except within minFrontFaceDistance of the surface:
 *    where solids touch with coincident faces (a board standing on the
 *    floor, a wall running down through it), each ray reports one of the
 *    tied faces, and a probe a few millimetres inside can't tell. Those are
 *    counted and reported, not failed;
 * 2. a probe whose grid point is inside a solid and can't get out within
 *    ±0.45 spacing (every point of that box is inside a solid) reads 1;
 * 3. a probe that ends in the room's air reads 0 (active), or 2 ("off: no
 *    surface nearby") only when no surface crosses the box one spacing
 *    around it;
 * 4. a probe that ends in air outside the room reads 1 or 2;
 * 5. the `points` debug view holds every probe's world position and state;
 * 6. phase B's subgroup reduction and its declared fallback give identical
 *    states and offsets on the same probes (when the device has subgroups);
 * 7. relocation and classification settle: SETTLE_FRAMES more all-probe
 *    frames change no probe's state or offset.
 * 8. `resetProbes` (DV-12), on the settled field of the first reduction's
 *    run: `{}`, `{ offsets }`, `{ states }` and both, in turn. Each is
 *    followed by one frame that updates no probe, after which every probe's
 *    offset and state are what the rule says (`resetProbeDataTexel`: offsets
 *    to 0, off probes to state 3, the rest kept), the irradiance and
 *    distance atlases are byte for byte what they were, and the points view
 *    agrees with the probe data. Then FRAMES all-probe frames let relocation
 *    and classification find the probes again; how many end where they were
 *    before the reset is reported, not judged (the path back can differ).
 */

import { assert, runGpuTest } from '../../client.ts';
import { createDdgiScene, readProbeData, readTextureBytes, runFrames, type DdgiScene, type ProbeRecord, releaseDeviceOnPageHide } from '../ddgi-shared/harness.ts';
import type { ProceduralScene } from '../../../../scenes/src/scene-data.ts';
import { BYTES_PER_TEXEL, PROBE_STATE } from '../../../../packages/ddgi-core/src/techniques/ddgi/layout.ts';
import { AUTO_MIN_FRONT_FACE_SHARE, resolveBias } from '../../../../packages/ddgi-core/src/techniques/ddgi/settings.ts';
import { probeResetFlags, resetProbeDataTexel, type ProbeResetRequest } from '../../../../packages/ddgi-core/src/techniques/ddgi/probe-reset.ts';
import type { PhaseBReduction } from '../../../../packages/ddgi-core/src/techniques/ddgi/wgsl/blend.ts';
import { CORNELL_DEPTH_M, CORNELL_HEIGHT_M, CORNELL_WIDTH_M } from '../../../../scenes/src/procedural/cornell-box.ts';
import { CABINET_ROOM_SIZE_M } from '../../../../scenes/src/procedural/cabinet-row.ts';

/**
 * All-probe frames before the checks. Probes that are too close to a face and
 * whose most open direction runs along it slide along it until their offset
 * limit stops them, up to about 75 updates (blend.ts), so 120 leaves room.
 */
const FRAMES = 120;
/** All-probe frames after FRAMES in which no probe may change any more. */
const SETTLE_FRAMES = 20;
/** A probe may move at most this share of the spacing on each axis (design: relocation). */
const MAX_OFFSET = 0.45;
/** Points per axis when checking whether a probe's ±0.45 box has any way out. */
const ESCAPE_SAMPLES = 9;

type Vec3 = [number, number, number];

interface Case {
    readonly scene: string;
    /** The room's inside, for "in the room's air" against "outside the room". */
    readonly room: { readonly min: Vec3; readonly max: Vec3 };
}

const CASES: readonly Case[] = [
    { scene: 'cornell-box-point', room: { min: [0, 0, 0], max: [CORNELL_WIDTH_M, CORNELL_HEIGHT_M, CORNELL_DEPTH_M] } },
    { scene: 'cabinet-row', room: { min: [0, 0, 0], max: [...CABINET_ROOM_SIZE_M] as Vec3 } },
];

// ---- Geometry on the CPU ----

type Triangle = [Vec3, Vec3, Vec3];

/** World-space triangles, one list per instance (each instance is a closed box here). */
function instanceTriangles(data: ProceduralScene): Triangle[][] {
    return data.instances.map((instance) => {
        const mesh = data.meshes[instance.mesh];
        const m = instance.transform;
        const world = (vertex: number): Vec3 => {
            const [x, y, z] = [mesh.positions[vertex * 3], mesh.positions[vertex * 3 + 1], mesh.positions[vertex * 3 + 2]];
            return [0, 1, 2].map((row) => m[row] * x + m[4 + row] * y + m[8 + row] * z + m[12 + row]) as Vec3;
        };
        const triangles: Triangle[] = [];
        for (let index = 0; index < mesh.indices.length; index += 3) triangles.push([world(mesh.indices[index]), world(mesh.indices[index + 1]), world(mesh.indices[index + 2])]);
        return triangles;
    });
}

const sub = (a: Vec3, b: Vec3): Vec3 => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const dot = (a: Vec3, b: Vec3): number => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a: Vec3, b: Vec3): Vec3 => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];

/** Ray-parity: inside a closed mesh if a ray crosses it an odd number of times. The direction is skewed so it never runs along an edge. */
const PARITY_DIRECTION: Vec3 = [0.9271, 0.2913, 0.2357];
function insideMesh(point: Vec3, triangles: readonly Triangle[]): boolean {
    let crossings = 0;
    for (const [a, b, c] of triangles) {
        const edge1 = sub(b, a);
        const edge2 = sub(c, a);
        const p = cross(PARITY_DIRECTION, edge2);
        const determinant = dot(edge1, p);
        if (Math.abs(determinant) < 1e-12) continue;
        const toPoint = sub(point, a);
        const u = dot(toPoint, p) / determinant;
        if (u < 0 || u > 1) continue;
        const q = cross(toPoint, edge1);
        const v = dot(PARITY_DIRECTION, q) / determinant;
        if (v < 0 || u + v > 1) continue;
        if (dot(edge2, q) / determinant > 0) crossings++;
    }
    return crossings % 2 === 1;
}

/** Distance from a point to a triangle (Ericson, Real-Time Collision Detection, 5.1.5). */
function distanceToTriangle(point: Vec3, [a, b, c]: Triangle): number {
    const ab = sub(b, a);
    const ac = sub(c, a);
    const ap = sub(point, a);
    const d1 = dot(ab, ap);
    const d2 = dot(ac, ap);
    const along = (t: Vec3, origin: Vec3, scale: number): Vec3 => [origin[0] + t[0] * scale, origin[1] + t[1] * scale, origin[2] + t[2] * scale];
    const length = (v: Vec3): number => Math.hypot(v[0], v[1], v[2]);
    if (d1 <= 0 && d2 <= 0) return length(ap);
    const bp = sub(point, b);
    const d3 = dot(ab, bp);
    const d4 = dot(ac, bp);
    if (d3 >= 0 && d4 <= d3) return length(bp);
    const vc = d1 * d4 - d3 * d2;
    if (vc <= 0 && d1 >= 0 && d3 <= 0) return length(sub(point, along(ab, a, d1 / (d1 - d3))));
    const cp = sub(point, c);
    const d5 = dot(ab, cp);
    const d6 = dot(ac, cp);
    if (d6 >= 0 && d5 <= d6) return length(cp);
    const vb = d5 * d2 - d1 * d6;
    if (vb <= 0 && d2 >= 0 && d6 <= 0) return length(sub(point, along(ac, a, d2 / (d2 - d6))));
    const va = d3 * d6 - d5 * d4;
    if (va <= 0 && d4 - d3 >= 0 && d5 - d6 >= 0) return length(sub(point, along(sub(c, b), b, (d4 - d3) / (d4 - d3 + (d5 - d6)))));
    const denominator = 1 / (va + vb + vc);
    const v = vb * denominator;
    const w = vc * denominator;
    return length(sub(point, [a[0] + ab[0] * v + ac[0] * w, a[1] + ab[1] * v + ac[1] * w, a[2] + ab[2] * v + ac[2] * w]));
}

function distanceToSurface(point: Vec3, meshes: readonly Triangle[][]): number {
    let nearest = Infinity;
    for (const triangles of meshes) for (const triangle of triangles) nearest = Math.min(nearest, distanceToTriangle(point, triangle));
    return nearest;
}

function insideAnySolid(point: Vec3, meshes: readonly Triangle[][]): boolean {
    return meshes.some((triangles) => insideMesh(point, triangles));
}

/** Triangle against axis-aligned box, by the separating-axis theorem (Akenine-Möller 2001). */
function triangleOverlapsBox(triangle: Triangle, centre: Vec3, halfSize: Vec3): boolean {
    const v = triangle.map((corner) => sub(corner, centre)) as Triangle;
    const edges = [sub(v[1], v[0]), sub(v[2], v[1]), sub(v[0], v[2])];
    const axes: Vec3[] = [[1, 0, 0], [0, 1, 0], [0, 0, 1], cross(edges[0], edges[1])];
    for (const edge of edges) for (const axis of [[1, 0, 0], [0, 1, 0], [0, 0, 1]] as Vec3[]) axes.push(cross(edge, axis));
    for (const axis of axes) {
        if (dot(axis, axis) < 1e-20) continue;
        const projections = v.map((corner) => dot(corner, axis));
        const radius = halfSize[0] * Math.abs(axis[0]) + halfSize[1] * Math.abs(axis[1]) + halfSize[2] * Math.abs(axis[2]);
        if (Math.min(...projections) > radius || Math.max(...projections) < -radius) return false;
    }
    return true;
}

function surfaceInReach(position: Vec3, spacing: Vec3, meshes: readonly Triangle[][]): boolean {
    return meshes.some((triangles) => triangles.some((triangle) => triangleOverlapsBox(triangle, position, spacing)));
}

/** True if every sampled point of the ±0.45-spacing box around the grid point is inside a solid. */
function cannotEscape(gridPoint: Vec3, spacing: Vec3, meshes: readonly Triangle[][]): boolean {
    for (let i = 0; i < ESCAPE_SAMPLES; i++) {
        for (let j = 0; j < ESCAPE_SAMPLES; j++) {
            for (let k = 0; k < ESCAPE_SAMPLES; k++) {
                const step = [i, j, k].map((n) => (n / (ESCAPE_SAMPLES - 1)) * 2 - 1);
                const point = [0, 1, 2].map((axis) => gridPoint[axis] + step[axis] * MAX_OFFSET * spacing[axis]) as Vec3;
                if (!insideAnySolid(point, meshes)) return false;
            }
        }
    }
    return true;
}

/** The resets checked in turn (8.), by name. */
const RESETS: readonly { readonly name: string; readonly what: ProbeResetRequest }[] = [
    { name: 'nothing ({})', what: {} },
    { name: 'offsets', what: { offsets: true } },
    { name: 'states', what: { states: true } },
    { name: 'offsets and states', what: { offsets: true, states: true } },
];

interface ResetResult {
    /** Probes with an offset, and probes off, before the reset. */
    readonly movedBefore: number;
    readonly offBefore: number;
    /** Probes whose texel isn't what the rule says after the reset frame (0 to pass). */
    readonly wrongTexels: number;
    /** Bytes of the irradiance and distance atlases that changed in the reset frame (0 to pass). */
    readonly irradianceBytesChanged: number;
    readonly distanceBytesChanged: number;
    /** Points that disagree with the probe data after the reset frame (0 to pass). */
    readonly pointsWrong: number;
    /** Probes moved, off, and not updated yet right after the reset. */
    readonly movedAfter: number;
    readonly offAfter: number;
    readonly notUpdatedYetAfter: number;
    /** After FRAMES more all-probe frames: probes back at their offset and state from before the reset, and probes moved and off (reported). */
    readonly resettledSame: number;
    readonly resettledMoved: number;
    readonly resettledOff: number;
}

function changedBytes(before: Uint8Array, after: Uint8Array): number {
    let changed = 0;
    for (let index = 0; index < before.length; index++) if (before[index] !== after[index]) changed++;
    return changed;
}

const isOff = (record: ProbeRecord): boolean => record.state === PROBE_STATE.offInsideGeometry || record.state === PROBE_STATE.offNoSurfaceNearby;
const isMoved = (record: ProbeRecord): boolean => record.offset.some((value) => value !== 0);

/** 8. Every reset in turn on the settled field (the file comment). */
async function checkResets(device: GPUDevice, scene: DdgiScene, fail: (message: string) => void): Promise<Record<string, ResetResult>> {
    const field = scene.ddgi.field;
    assert(field, 'no field after prepare');
    const { origin, spacing, counts, probeCount } = field.grid;
    const pointsView = scene.ddgi.debugViews.find((view) => view.kind === 'points');
    assert(pointsView && pointsView.kind === 'points', 'no points debug view');
    const distanceBytes = BYTES_PER_TEXEL[field.layout.distanceFormat];
    const results: Record<string, ResetResult> = {};
    for (const reset of RESETS) {
        const before = await readProbeData(device, field);
        const irradianceBefore = await readTextureBytes(device, field.layeredIrradiance, 8);
        const distanceBefore = await readTextureBytes(device, field.distance, distanceBytes);
        scene.ddgi.resetProbes(reset.what);
        // One frame that updates no probe: only the reset changes anything.
        await runFrames(device, scene.platform, 1, 0);
        const after = await readProbeData(device, field);
        const irradianceAfter = await readTextureBytes(device, field.layeredIrradiance, 8);
        const distanceAfter = await readTextureBytes(device, field.distance, distanceBytes);
        const points = new Float32Array((await readTextureBytes(device, pointsView.texture(), 16)).buffer);

        const flags = probeResetFlags(reset.what);
        let wrongTexels = 0;
        let pointsWrong = 0;
        after.forEach((record, probeIndex) => {
            const expected = flags ? resetProbeDataTexel(before[probeIndex], flags) : before[probeIndex];
            const wrong = record.state !== expected.state || record.offset.some((value, axis) => value !== expected.offset[axis]);
            if (wrong) {
                wrongTexels++;
                if (wrongTexels <= 4) fail(`reset ${reset.name}: probe ${probeIndex} reads ${record.state}@${record.offset.join(',')}, expected ${expected.state}@${expected.offset.join(',')}`);
            }
            // Both cases are fixed volumes, so a slot is its window probe (a camera-following one would need its window here).
            const grid = [probeIndex % counts[0], Math.floor(probeIndex / (counts[0] * counts[2])), Math.floor(probeIndex / counts[0]) % counts[2]];
            const position = [0, 1, 2].map((axis) => origin[axis] + (grid[axis] + record.offset[axis]) * spacing[axis]);
            const point = points.subarray(probeIndex * 4, probeIndex * 4 + 4);
            if ([0, 1, 2].some((axis) => Math.abs(point[axis] - position[axis]) > 1e-3) || point[3] !== record.state) pointsWrong++;
        });
        const irradianceBytesChanged = changedBytes(irradianceBefore, irradianceAfter);
        const distanceBytesChanged = changedBytes(distanceBefore, distanceAfter);
        if (wrongTexels > 0) fail(`reset ${reset.name}: ${wrongTexels} of ${probeCount} probes not as the rule says`);
        if (irradianceBytesChanged > 0) fail(`reset ${reset.name}: ${irradianceBytesChanged} irradiance bytes changed in the reset frame`);
        if (distanceBytesChanged > 0) fail(`reset ${reset.name}: ${distanceBytesChanged} distance bytes changed in the reset frame`);
        if (pointsWrong > 0) fail(`reset ${reset.name}: ${pointsWrong} points disagree with the probe data`);

        // Relocation and classification find the probes again over their usual updates.
        await runFrames(device, scene.platform, FRAMES, probeCount);
        const resettled = await readProbeData(device, field);
        const same = resettled.filter((record, index) => record.state === before[index].state && record.offset.every((value, axis) => value === before[index].offset[axis])).length;
        results[reset.name] = {
            movedBefore: before.filter(isMoved).length,
            offBefore: before.filter(isOff).length,
            wrongTexels,
            irradianceBytesChanged,
            distanceBytesChanged,
            pointsWrong,
            movedAfter: after.filter(isMoved).length,
            offAfter: after.filter(isOff).length,
            notUpdatedYetAfter: after.filter((record) => record.state === PROBE_STATE.notUpdatedYet).length,
            resettledSame: same,
            resettledMoved: resettled.filter(isMoved).length,
            resettledOff: resettled.filter(isOff).length,
        };
    }
    return results;
}

interface CaseResult {
    readonly records: ProbeRecord[];
    readonly points: Float32Array;
    readonly grid: { origin: readonly number[]; spacing: readonly number[]; counts: readonly number[] };
    readonly data: ProceduralScene;
    readonly counters: Readonly<Record<string, number>>;
    readonly minFrontFaceDistance: number;
    /** Probes whose state or offset still changed in the SETTLE_FRAMES after FRAMES, with what they did. */
    readonly unsettled: string[];
    /** 8., on the run that checks the resets (null on the others), and what it found wrong. */
    readonly resets: Record<string, ResetResult> | null;
    readonly resetFailures: string[];
}

async function runCase(device: GPUDevice, testCase: Case, reduction: PhaseBReduction, checkReset: boolean): Promise<CaseResult> {
    const scene = await createDdgiScene(device, testCase.scene, {}, { technique: { reduction } });
    try {
        const field = scene.ddgi.field;
        assert(field, 'no field after prepare');
        assert(scene.ddgi.reduction === reduction, `asked for the ${reduction} reduction, got ${scene.ddgi.reduction}`);
        await runFrames(device, scene.platform, FRAMES, field.grid.probeCount);
        const settledRecords = await readProbeData(device, field);
        const history = settledRecords.map((record) => [record]);
        for (let frame = 0; frame < SETTLE_FRAMES; frame++) {
            await runFrames(device, scene.platform, 1, field.grid.probeCount);
            (await readProbeData(device, field)).forEach((record, index) => history[index].push(record));
        }
        const describe = (record: ProbeRecord): string => `${record.state}@${record.offset.map((value) => value.toFixed(3)).join(',')}`;
        const unsettled: string[] = [];
        history.forEach((records, index) => {
            const first = describe(records[0]);
            if (records.some((record) => describe(record) !== first)) unsettled.push(`probe ${index}: ${records.map(describe).join(' -> ')}`);
        });
        const records = history.map((states) => states[states.length - 1]);
        const pointsView = scene.ddgi.debugViews.find((view) => view.kind === 'points');
        assert(pointsView && pointsView.kind === 'points', 'no points debug view');
        const points = new Float32Array((await readTextureBytes(device, pointsView.texture(), 16)).buffer);
        // minFrontFaceDistance 'auto' is a share of the smallest spacing.
        const minFrontFaceDistance = resolveBias(scene.ddgi.settings.minFrontFaceDistance, field.grid, AUTO_MIN_FRONT_FACE_SHARE);
        const counters = scene.ddgi.counters;
        // 8. After everything above has been read: the resets change the field.
        const resetFailures: string[] = [];
        const resets = checkReset ? await checkResets(device, scene, (message) => resetFailures.push(message)) : null;
        return { records, points, grid: field.grid, data: scene.data, counters, minFrontFaceDistance, unsettled, resets, resetFailures };
    } finally {
        scene.platform.dispose();
    }
}

await runGpuTest(
    async ({ device, log }) => {
        releaseDeviceOnPageHide(device);
        const failures: string[] = [];
        const results: Record<string, unknown> = {};
        const reductions: PhaseBReduction[] = device.features.has('subgroups') ? ['subgroups', 'workgroup'] : ['workgroup'];

        for (const testCase of CASES) {
            const runs = new Map<PhaseBReduction, CaseResult>();
            for (const reduction of reductions) runs.set(reduction, await runCase(device, testCase, reduction, reduction === reductions[0]));
            const result = runs.get(reductions[0])!;
            const meshes = instanceTriangles(result.data);
            const { origin, spacing, counts } = result.grid;
            const spacing3 = spacing as Vec3;
            const fail = (message: string): void => {
                failures.push(`${testCase.scene}: ${message}`);
            };

            const tally = { inSolidOff: 0, stuckOff: 0, airActive: 0, airNoSurface: 0, outside: 0, offsetsMoved: 0, notUpdatedYet: 0, shallowInSolidOn: 0, deepestShallowMm: 0 };
            const minFrontFace = result.minFrontFaceDistance;
            result.records.forEach((record, probeIndex) => {
                const x = probeIndex % counts[0];
                const z = Math.floor(probeIndex / counts[0]) % counts[2];
                const y = Math.floor(probeIndex / (counts[0] * counts[2]));
                const grid: Vec3 = [x, y, z];
                const gridPoint = [0, 1, 2].map((axis) => origin[axis] + grid[axis] * spacing[axis]) as Vec3;
                const position = [0, 1, 2].map((axis) => origin[axis] + (grid[axis] + record.offset[axis]) * spacing[axis]) as Vec3;
                const where = `probe ${probeIndex} (${grid.join(', ')}) at ${position.map((value) => value.toFixed(3)).join(', ')}, offset ${record.offset.map((value) => value.toFixed(3)).join(', ')}, state ${record.state}`;
                if (record.offset.some((value) => value !== 0)) tally.offsetsMoved++;
                if (record.state === PROBE_STATE.notUpdatedYet) tally.notUpdatedYet++;

                // 5. The points view.
                const point = result.points.subarray(probeIndex * 4, probeIndex * 4 + 4);
                if ([0, 1, 2].some((axis) => Math.abs(point[axis] - position[axis]) > 1e-3) || point[3] !== record.state) fail(`points view disagrees for ${where}: (${Array.from(point).join(', ')})`);

                if (insideAnySolid(position, meshes)) {
                    // 1.
                    if (record.state === PROBE_STATE.offInsideGeometry) {
                        tally.inSolidOff++;
                    } else {
                        const depth = distanceToSurface(position, meshes);
                        if (depth > minFrontFace) fail(`${(depth * 1000).toFixed(1)} mm inside a solid but not off: ${where}`);
                        tally.shallowInSolidOn++;
                        tally.deepestShallowMm = Math.max(tally.deepestShallowMm, depth * 1000);
                    }
                    return;
                }
                // 2.
                if (insideAnySolid(gridPoint, meshes) && cannotEscape(gridPoint, spacing3, meshes)) {
                    if (record.state !== PROBE_STATE.offInsideGeometry) fail(`can't leave its solid but not off: ${where}`);
                    else tally.stuckOff++;
                    return;
                }
                const inRoom = [0, 1, 2].every((axis) => position[axis] > testCase.room.min[axis] && position[axis] < testCase.room.max[axis]);
                if (!inRoom) {
                    // 4.
                    tally.outside++;
                    if (record.state !== PROBE_STATE.offInsideGeometry && record.state !== PROBE_STATE.offNoSurfaceNearby) fail(`outside the room but not off: ${where}`);
                    return;
                }
                // 3.
                if (record.state === PROBE_STATE.active) tally.airActive++;
                else if (record.state === PROBE_STATE.offNoSurfaceNearby && !surfaceInReach(position, spacing3, meshes)) tally.airNoSurface++;
                else fail(`in the room's air but ${record.state === PROBE_STATE.offNoSurfaceNearby ? 'off with a surface in reach' : 'not active'}: ${where}`);
            });

            // 7. Settled.
            for (const line of result.unsettled.slice(0, 6)) fail(`never settles: ${line}`);
            if (result.unsettled.length > 6) fail(`${result.unsettled.length} probes never settle in all`);

            // 6. Subgroups against the fallback.
            let reductionMismatches = 0;
            if (runs.size === 2) {
                const other = runs.get('workgroup')!;
                result.records.forEach((record, index) => {
                    const second = other.records[index];
                    if (record.state !== second.state || record.offset.some((value, axis) => value !== second.offset[axis])) reductionMismatches++;
                });
                if (reductionMismatches > 0) fail(`${reductionMismatches} probes differ between the subgroup reduction and the fallback`);
            }

            // 8. resetProbes.
            for (const message of result.resetFailures) fail(message);

            results[testCase.scene] = {
                probes: result.records.length,
                frames: FRAMES,
                settleFrames: SETTLE_FRAMES,
                unsettled: result.unsettled.length,
                reductions,
                reductionMismatches,
                ...tally,
                resets: result.resets,
                counters: result.counters,
            };
            log(`${testCase.scene}: ${JSON.stringify(tally)}; subgroups vs fallback mismatches ${reductionMismatches}`);
            for (const [name, reset] of Object.entries(result.resets ?? {})) log(`${testCase.scene}: reset ${name}: ${JSON.stringify(reset)}`);
        }
        assert(failures.length === 0, `${failures.length} problems:\n${failures.slice(0, 12).join('\n')}`);
        return results;
    },
    { optionalFeatures: ['texture-formats-tier1', 'subgroups'] },
);
