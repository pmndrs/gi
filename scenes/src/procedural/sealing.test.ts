/**
 * Sealing checks for the leak-test scenes: rays from inside a room, a closed
 * door's near side, a thin wall's lit side or a hollow cabinet must never get
 * out, including rays aimed exactly along the seams where two solids meet.
 *
 * Triangles are tested with a plain, strict ray-triangle test and no
 * tolerance, so a crack between two touching faces would show as an escaping
 * ray. The scenes avoid that by overlapping solids instead of touching them.
 */

import { describe, expect, it } from 'vitest';
import type { ProceduralScene, Vector3Tuple } from '../scene-data';
import { buildCabinetRow, PANEL_THICKNESS_M } from './cabinet-row';
import { buildFurnaceRoom, FURNACE_SIZE_M } from './furnace-room';
import { buildThinWall, THIN_WALL_THICKNESS_M } from './thin-wall';
import { buildClosedDoor, buildTwoRoom } from './two-rooms';

type Vec = [number, number, number];

/** All triangles of the scene in world space, as flat arrays of 9 numbers. */
function worldTriangles(scene: ProceduralScene): Float64Array {
    const out: number[] = [];
    for (const instance of scene.instances) {
        const mesh = scene.meshes[instance.mesh];
        const m = instance.transform;
        const world = (index: number): Vec => {
            const x = mesh.positions[index * 3];
            const y = mesh.positions[index * 3 + 1];
            const z = mesh.positions[index * 3 + 2];
            return [
                m[0] * x + m[4] * y + m[8] * z + m[12],
                m[1] * x + m[5] * y + m[9] * z + m[13],
                m[2] * x + m[6] * y + m[10] * z + m[14],
            ];
        };
        for (let i = 0; i < mesh.indices.length; i += 3) {
            out.push(...world(mesh.indices[i]), ...world(mesh.indices[i + 1]), ...world(mesh.indices[i + 2]));
        }
    }
    return Float64Array.from(out);
}

/** Nearest hit distance along the ray (double-sided, strict edges), or null when it escapes. */
function nearestHit(triangles: Float64Array, origin: Vec, direction: Vec): number | null {
    let nearest = Infinity;
    for (let t = 0; t < triangles.length; t += 9) {
        const e1x = triangles[t + 3] - triangles[t];
        const e1y = triangles[t + 4] - triangles[t + 1];
        const e1z = triangles[t + 5] - triangles[t + 2];
        const e2x = triangles[t + 6] - triangles[t];
        const e2y = triangles[t + 7] - triangles[t + 1];
        const e2z = triangles[t + 8] - triangles[t + 2];
        const px = direction[1] * e2z - direction[2] * e2y;
        const py = direction[2] * e2x - direction[0] * e2z;
        const pz = direction[0] * e2y - direction[1] * e2x;
        const det = e1x * px + e1y * py + e1z * pz;
        if (Math.abs(det) < 1e-14) continue;
        const inverse = 1 / det;
        const tx = origin[0] - triangles[t];
        const ty = origin[1] - triangles[t + 1];
        const tz = origin[2] - triangles[t + 2];
        const u = (tx * px + ty * py + tz * pz) * inverse;
        if (u < 0 || u > 1) continue;
        const qx = ty * e1z - tz * e1y;
        const qy = tz * e1x - tx * e1z;
        const qz = tx * e1y - ty * e1x;
        const v = (direction[0] * qx + direction[1] * qy + direction[2] * qz) * inverse;
        if (v < 0 || u + v > 1) continue;
        const distance = (e2x * qx + e2y * qy + e2z * qz) * inverse;
        if (distance > 1e-9 && distance < nearest) nearest = distance;
    }
    return nearest === Infinity ? null : nearest;
}

/** Evenly spread directions over the sphere. */
function sphereDirections(count: number): Vec[] {
    const golden = Math.PI * (3 - Math.sqrt(5));
    return Array.from({ length: count }, (_, i) => {
        const y = 1 - (2 * (i + 0.5)) / count;
        const radius = Math.sqrt(1 - y * y);
        return [Math.cos(golden * i) * radius, y, Math.sin(golden * i) * radius] as Vec;
    });
}

/** Points along the 12 edges of a box, where two walls meet. */
function boxEdgePoints(min: Vector3Tuple, max: Vector3Tuple, perEdge = 60): Vec[] {
    const points: Vec[] = [];
    for (const axis of [0, 1, 2]) {
        const others = [0, 1, 2].filter((a) => a !== axis);
        for (const cornerA of [min[others[0]], max[others[0]]]) {
            for (const cornerB of [min[others[1]], max[others[1]]]) {
                for (let i = 0; i <= perEdge; i++) {
                    const point: Vec = [0, 0, 0];
                    point[axis] = min[axis] + ((max[axis] - min[axis]) * i) / perEdge;
                    point[others[0]] = cornerA;
                    point[others[1]] = cornerB;
                    points.push(point);
                }
            }
        }
    }
    return points;
}

function towards(origin: Vec, target: Vec): Vec {
    const d: Vec = [target[0] - origin[0], target[1] - origin[1], target[2] - origin[2]];
    const length = Math.hypot(...d);
    return [d[0] / length, d[1] / length, d[2] / length];
}

/** Every ray (spread over the sphere, and aimed at each seam point) must hit something within `limit` on `axis`. */
function expectContained(
    scene: ProceduralScene,
    origin: Vec,
    seamPoints: Vec[],
    isInside: (point: Vec) => boolean,
    label: string,
): void {
    const triangles = worldTriangles(scene);
    const directions = [...sphereDirections(4000), ...seamPoints.map((point) => towards(origin, point))];
    let escaped = 0;
    let outside = 0;
    for (const direction of directions) {
        const distance = nearestHit(triangles, origin, direction);
        if (distance === null) {
            escaped++;
            continue;
        }
        const hit: Vec = [origin[0] + direction[0] * distance, origin[1] + direction[1] * distance, origin[2] + direction[2] * distance];
        if (!isInside(hit)) outside++;
    }
    expect({ label, escaped, outside }).toEqual({ label, escaped: 0, outside: 0 });
}

describe('sealing', () => {
    it('furnace room: no ray leaves, including rays along every seam', () => {
        const [x, y, z] = FURNACE_SIZE_M;
        const seams = boxEdgePoints([0, 0, 0], [x, y, z]);
        const scene = buildFurnaceRoom();
        for (const origin of [[2.5, 1.4, 2], [0.2, 0.3, 3.8], [4.5, 2.5, 0.3]] as Vec[]) {
            expectContained(scene, origin, seams, () => true, 'furnace');
        }
    });

    it('two rooms: no ray leaves the building, from either room', () => {
        const seams = boxEdgePoints([0, 0, 0], [8.2, 2.8, 4]);
        for (const scene of [buildTwoRoom(), buildClosedDoor()]) {
            for (const origin of [[2, 1.5, 2], [6.5, 1.5, 2]] as Vec[]) {
                expectContained(scene, origin, seams, () => true, 'two rooms');
            }
        }
    });

    it('closed door: room A never sees past the door, room B never sees past it the other way', () => {
        const scene = buildClosedDoor();
        const doorFaceX = 4.1 - 0.02;
        // Points around the door's frame, where the panel meets the partition.
        const frame: Vec[] = [];
        for (let i = 0; i <= 100; i++) {
            const along = i / 100;
            frame.push([doorFaceX, along * 2.1, 1.55], [doorFaceX, along * 2.1, 2.45], [doorFaceX, 2.1, 1.55 + along * 0.9]);
        }
        expectContained(scene, [2, 1.5, 2], frame, (hit) => hit[0] <= doorFaceX + 1e-6, 'room A of closed-door');
        const farFaceX = 4.1 + 0.02;
        const farFrame = frame.map((point): Vec => [farFaceX, point[1], point[2]]);
        expectContained(scene, [6.5, 1.5, 2], farFrame, (hit) => hit[0] >= farFaceX - 1e-6, 'room B of closed-door');
    });

    it('thin wall: neither side sees the other, including rays along the wall edges', () => {
        const scene = buildThinWall();
        const half = THIN_WALL_THICKNESS_M / 2;
        const seamsOn = (x: number): Vec[] => {
            const points: Vec[] = [];
            for (let i = 0; i <= 200; i++) {
                const along = i / 200;
                points.push([x, 0, along * 4], [x, 2.8, along * 4], [x, along * 2.8, 0], [x, along * 2.8, 4]);
            }
            return points;
        };
        expectContained(scene, [2, 1.5, 2], seamsOn(4 - half), (hit) => hit[0] <= 4 - half + 1e-6, 'lit side');
        expectContained(scene, [6, 1.5, 2], seamsOn(4 + half), (hit) => hit[0] >= 4 + half - 1e-6, 'dark side');
    });

    it('cabinets: no ray escapes from inside a closed cabinet, including along the inside seams', () => {
        const scene = buildCabinetRow();
        const p = PANEL_THICKNESS_M;
        const cabinets: Array<{ name: string; min: Vec; max: Vec }> = [
            { name: 'base unit', min: [0.6, 0.1, 0], max: [1.2, 0.82, 0.58] },
            { name: 'wall unit', min: [1.2, 1.45, 0], max: [1.8, 2.15, 0.35] },
            { name: 'tall unit', min: [0, 0, 0], max: [0.6, 2.2, 0.58] },
        ];
        for (const { name, min, max } of cabinets) {
            const inner: [Vec, Vec] = [
                [min[0] + p, min[1] + p, min[2] + p],
                [max[0] - p, max[1] - p, max[2] - p],
            ];
            const origin: Vec = [(inner[0][0] + inner[1][0]) / 2, (inner[0][1] + inner[1][1]) / 2, (inner[0][2] + inner[1][2]) / 2];
            expectContained(
                scene,
                origin,
                boxEdgePoints(inner[0], inner[1]),
                (hit) => [0, 1, 2].every((axis) => hit[axis] >= inner[0][axis] - 1e-6 && hit[axis] <= inner[1][axis] + 1e-6),
                name,
            );
        }
    });
});

/**
 * Strict ray tests in double precision cannot show the cracks a float32 GPU
 * ray tracer can find between touching faces, so also check the cause directly:
 * no two boards or wall slabs may meet face to face with nothing overlapping.
 */
describe('no flush seams', () => {
    interface Box {
        name: string;
        min: Vec;
        max: Vec;
    }

    function axisAlignedBoxes(scene: ProceduralScene, include: RegExp): Box[] {
        return scene.instances
            .filter((instance) => scene.meshes[instance.mesh].name === 'unit-box' && include.test(instance.name))
            .map((instance) => {
                const m = instance.transform;
                expect(m[1] === 0 && m[2] === 0 && m[4] === 0 && m[6] === 0 && m[8] === 0 && m[9] === 0, instance.name).toBe(true);
                const half = [Math.abs(m[0]) / 2, Math.abs(m[5]) / 2, Math.abs(m[10]) / 2];
                return {
                    name: instance.name,
                    min: [m[12] - half[0], m[13] - half[1], m[14] - half[2]] as Vec,
                    max: [m[12] + half[0], m[13] + half[1], m[14] + half[2]] as Vec,
                };
            });
    }

    /** Pairs of boxes that touch on a face: zero overlap on one axis, positive overlap on the other two. */
    function flushPairs(boxes: Box[], mayTouch: (a: string, b: string) => boolean = () => false): string[] {
        const found: string[] = [];
        for (let i = 0; i < boxes.length; i++) {
            for (let j = i + 1; j < boxes.length; j++) {
                const overlaps = [0, 1, 2].map((axis) => Math.min(boxes[i].max[axis], boxes[j].max[axis]) - Math.max(boxes[i].min[axis], boxes[j].min[axis]));
                const separated = overlaps.some((overlap) => overlap < -1e-9);
                const touching = overlaps.filter((overlap) => Math.abs(overlap) <= 1e-9).length === 1;
                const sharing = overlaps.filter((overlap) => overlap > 1e-9).length === 2;
                if (!separated && touching && sharing && !mayTouch(boxes[i].name, boxes[j].name)) {
                    found.push(`${boxes[i].name} / ${boxes[j].name}`);
                }
            }
        }
        return found;
    }

    it('thin wall and its room', () => {
        const boxes = axisAlignedBoxes(buildThinWall(), /^(room-|thin-wall)/);
        expect(flushPairs(boxes)).toEqual([]);
    });

    it('two rooms and the closed door', () => {
        const pattern = /^(floor|ceiling|wall-|partition|door-panel)/;
        expect(flushPairs(axisAlignedBoxes(buildTwoRoom(), pattern))).toEqual([]);
        expect(flushPairs(axisAlignedBoxes(buildClosedDoor(), pattern))).toEqual([]);
    });

    it('furnace room shell', () => {
        expect(flushPairs(axisAlignedBoxes(buildFurnaceRoom(), /^furnace-/))).toEqual([]);
    });

    it('boards within each cabinet', () => {
        const boxes = axisAlignedBoxes(buildCabinetRow(), /^(tall|base|wall)-unit/);
        const unitOf = (name: string) => name.replace(/-(bottom|top|side-left|side-right|back|door)$/, '');
        // Neighbouring cabinets stand side by side, as real ones do; only boards of one cabinet must not just touch.
        expect(flushPairs(boxes, (a, b) => unitOf(a) !== unitOf(b))).toEqual([]);
    });
});
