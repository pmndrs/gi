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
 *    states and offsets on the same probes (when the device has subgroups).
 */

import { assert, runGpuTest } from '../../client.ts';
import { createDdgiScene, readProbeData, readTextureBytes, runFrames, type ProbeRecord } from '../ddgi-shared/harness.ts';
import type { ProceduralScene } from '../../../../scenes/src/scene-data.ts';
import { PROBE_STATE } from '../../../../packages/ddgi-core/src/techniques/ddgi/layout.ts';
import { AUTO_MIN_FRONT_FACE_SHARE, resolveBias } from '../../../../packages/ddgi-core/src/techniques/ddgi/settings.ts';
import type { PhaseBReduction } from '../../../../packages/ddgi-core/src/techniques/ddgi/wgsl/blend.ts';
import { CORNELL_DEPTH_M, CORNELL_HEIGHT_M, CORNELL_WIDTH_M } from '../../../../scenes/src/procedural/cornell-box.ts';
import { CABINET_ROOM_SIZE_M } from '../../../../scenes/src/procedural/cabinet-row.ts';

const FRAMES = 60;
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

interface CaseResult {
    readonly records: ProbeRecord[];
    readonly points: Float32Array;
    readonly grid: { origin: readonly number[]; spacing: readonly number[]; counts: readonly number[] };
    readonly data: ProceduralScene;
    readonly counters: Readonly<Record<string, number>>;
    readonly minFrontFaceDistance: number;
}

async function runCase(device: GPUDevice, testCase: Case, reduction: PhaseBReduction): Promise<CaseResult> {
    const scene = await createDdgiScene(device, testCase.scene, {}, { technique: { reduction } });
    try {
        const field = scene.ddgi.field;
        assert(field, 'no field after prepare');
        assert(scene.ddgi.reduction === reduction, `asked for the ${reduction} reduction, got ${scene.ddgi.reduction}`);
        await runFrames(device, scene.platform, FRAMES, field.grid.probeCount);
        // Two more frames with no updates, so the census of the last update is read back.
        await runFrames(device, scene.platform, 3, 0);
        const records = await readProbeData(device, field);
        const pointsView = scene.ddgi.debugViews.find((view) => view.kind === 'points');
        assert(pointsView && pointsView.kind === 'points', 'no points debug view');
        const points = new Float32Array((await readTextureBytes(device, pointsView.texture(), 16)).buffer);
        // minFrontFaceDistance 'auto' is a share of the smallest spacing.
        const minFrontFaceDistance = resolveBias(scene.ddgi.settings.minFrontFaceDistance, field.grid, AUTO_MIN_FRONT_FACE_SHARE);
        return { records, points, grid: field.grid, data: scene.data, counters: scene.ddgi.counters, minFrontFaceDistance };
    } finally {
        scene.platform.dispose();
    }
}

await runGpuTest(
    async ({ device, log }) => {
        const failures: string[] = [];
        const results: Record<string, unknown> = {};
        const reductions: PhaseBReduction[] = device.features.has('subgroups') ? ['subgroups', 'workgroup'] : ['workgroup'];

        for (const testCase of CASES) {
            const runs = new Map<PhaseBReduction, CaseResult>();
            for (const reduction of reductions) runs.set(reduction, await runCase(device, testCase, reduction));
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

            results[testCase.scene] = { probes: result.records.length, frames: FRAMES, reductions, reductionMismatches, ...tally, counters: result.counters };
            log(`${testCase.scene}: ${JSON.stringify(tally)}; subgroups vs fallback mismatches ${reductionMismatches}`);
        }
        assert(failures.length === 0, `${failures.length} problems:\n${failures.slice(0, 12).join('\n')}`);
        return results;
    },
    { optionalFeatures: ['texture-formats-tier1', 'subgroups'] },
);
