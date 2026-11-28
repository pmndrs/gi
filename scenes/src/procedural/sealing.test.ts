/**
 * Sealing checks for the leak-test scenes: rays from inside a room, a closed
 * door's near side, a thin wall's lit side or a hollow cabinet must never get
 * out, including rays aimed exactly along the seams where two solids meet.
 *
 * What the two kinds of check can and cannot see:
 * - The ray checks cast rays in double precision with a strict ray-triangle
 *   test and no tolerance. They catch missing faces and real gaps between
 *   solids, down to a few micrometres when a ray is aimed into the gap (the
 *   "checks themselves" tests below show a 2 µm slot being found). They do not
 *   catch a crack between two faces that touch exactly: in double precision a
 *   ray aimed at such a seam still hits one of the two faces, while a float32
 *   GPU tracer can slip through it.
 * - So the "no flush seams" checks look for that cause directly: no two boards
 *   or wall slabs may meet face to face without overlapping. The scenes always
 *   overlap their solids instead of letting them touch.
 */

import { describe, expect, it } from 'vitest';
import type { ProceduralScene, Vector3Tuple } from '../scene-data';
import { buildCabinetRow, PANEL_THICKNESS_M } from './cabinet-row';
import { buildDarkRoom, DARK_ROOM_SIZE_M } from './dark-room';
import { buildEmissiveRoom, EMISSIVE_ROOM_SIZE_M } from './emissive-room';
import { buildFurnaceRoom, FURNACE_SIZE_M } from './furnace-room';
import { SceneBuilder } from './geometry';
import { buildThinWall, THIN_WALL_THICKNESS_M } from './thin-wall';
import {
    buildClosedDoor,
    buildTwoRoom,
    DOOR_CENTRE_X_M,
    DOOR_THICKNESS_M,
    DOORWAY_HEIGHT_M,
    DOORWAY_Z0_M,
    DOORWAY_Z1_M,
    ROOM_B_X1_M,
    ROOM_HEIGHT_M,
    ROOM_SIZE_M,
} from './two-rooms';

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

/**
 * Casts rays spread over the sphere, plus one aimed at each seam point, and
 * counts those that hit nothing (`escaped`) or first hit something outside
 * the region `isInside` describes (`outside`).
 */
function countLeaks(scene: ProceduralScene, origin: Vec, seamPoints: Vec[], isInside: (point: Vec) => boolean): { escaped: number; outside: number } {
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
    return { escaped, outside };
}

function expectContained(scene: ProceduralScene, origin: Vec, seamPoints: Vec[], isInside: (point: Vec) => boolean, label: string): void {
    expect({ label, ...countLeaks(scene, origin, seamPoints, isInside) }).toEqual({ label, escaped: 0, outside: 0 });
}

interface Box {
    name: string;
    min: Vec;
    max: Vec;
}

/** The scene's unrotated unit-box instances whose names match `include`, as world-space boxes. Fails when none match. */
function axisAlignedBoxes(scene: ProceduralScene, include: RegExp): Box[] {
    const boxes = scene.instances
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
    // A pattern that matches nothing would make the flush check pass without checking anything.
    expect(boxes.length, `the pattern ${include} matched no boxes`).toBeGreaterThan(0);
    return boxes;
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

describe('sealing', () => {
    it('furnace room: no ray leaves, including rays along every seam', () => {
        const [x, y, z] = FURNACE_SIZE_M;
        const seams = boxEdgePoints([0, 0, 0], [x, y, z]);
        const scene = buildFurnaceRoom();
        for (const origin of [[2.5, 1.4, 2], [0.2, 0.3, 3.8], [4.5, 2.5, 0.3]] as Vec[]) {
            expectContained(scene, origin, seams, () => true, 'furnace');
        }
    });

    it('dark room: no ray leaves, including rays along every seam', () => {
        const seams = boxEdgePoints([0, 0, 0], DARK_ROOM_SIZE_M);
        for (const origin of [[1, 1.5, 3.5], [4.5, 2.2, 0.5]] as Vec[]) {
            expectContained(buildDarkRoom(), origin, seams, () => true, 'dark room');
        }
    });

    it('emissive room: no ray leaves, including rays along every seam', () => {
        const [x, y, z] = EMISSIVE_ROOM_SIZE_M;
        const seams = boxEdgePoints([0, 0, 0], [x, y, z]);
        const scene = buildEmissiveRoom();
        for (const origin of [[3, 1.4, 2.4], [0.3, 2.5, 0.3], [5.7, 0.2, 4.4]] as Vec[]) {
            expectContained(scene, origin, seams, () => true, 'emissive room');
        }
    });

    it('two rooms: no ray leaves the building, from either room', () => {
        const seams = boxEdgePoints([0, 0, 0], [ROOM_B_X1_M, ROOM_HEIGHT_M, ROOM_SIZE_M]);
        for (const scene of [buildTwoRoom(), buildClosedDoor()]) {
            for (const origin of [[2, 1.5, 2], [6.5, 1.5, 2]] as Vec[]) {
                expectContained(scene, origin, seams, () => true, 'two rooms');
            }
        }
    });

    it('closed door: room A never sees past the door, room B never sees past it the other way', () => {
        const scene = buildClosedDoor();
        const doorFaceX = DOOR_CENTRE_X_M - DOOR_THICKNESS_M / 2;
        const doorwayWidth = DOORWAY_Z1_M - DOORWAY_Z0_M;
        // Points around the door's frame, where the panel meets the partition: both sides and the top of the opening.
        const frame: Vec[] = [];
        for (let i = 0; i <= 100; i++) {
            const along = i / 100;
            frame.push(
                [doorFaceX, along * DOORWAY_HEIGHT_M, DOORWAY_Z0_M],
                [doorFaceX, along * DOORWAY_HEIGHT_M, DOORWAY_Z1_M],
                [doorFaceX, DOORWAY_HEIGHT_M, DOORWAY_Z0_M + along * doorwayWidth],
            );
        }
        expectContained(scene, [2, 1.5, 2], frame, (hit) => hit[0] <= doorFaceX + 1e-6, 'room A of closed-door');
        const farFaceX = DOOR_CENTRE_X_M + DOOR_THICKNESS_M / 2;
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

describe('no flush seams', () => {
    it('thin wall and its room', () => {
        const boxes = axisAlignedBoxes(buildThinWall(), /^(room-|thin-wall)/);
        expect(flushPairs(boxes)).toEqual([]);
    });

    it('two rooms and the closed door', () => {
        const pattern = /^(floor|ceiling|wall-|partition|door-panel)/;
        expect(flushPairs(axisAlignedBoxes(buildTwoRoom(), pattern))).toEqual([]);
        expect(flushPairs(axisAlignedBoxes(buildClosedDoor(), pattern))).toEqual([]);
    });

    it('dark room shell', () => {
        expect(flushPairs(axisAlignedBoxes(buildDarkRoom(), /^room-/))).toEqual([]);
    });

    it('emissive room: its shell and everything fixed to it (furniture may stand on the floor)', () => {
        const boxes = axisAlignedBoxes(buildEmissiveRoom(), /^(room-|strip-|screen-|cabinet|shelf)/);
        expect(flushPairs(boxes, (a, b) => a === 'room-floor' || b === 'room-floor')).toEqual([]);
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

/**
 * The checks above only mean something if they can fail. A unit room (inside
 * 0..1 m on each axis, walls 0.25 m thick) whose south wall is built from two
 * halves meeting at x = 0.5 shows what each check finds. The sizes are powers
 * of two, so every corner is exact in floating point.
 */
describe('the checks themselves', () => {
    const wall = 0.25;
    const seamX = 0.5;
    const origin: Vec = [0.5, 0.5, 0.5];
    const insideRoom = (point: Vec) => point.every((value) => value >= -1e-9 && value <= 1 + 1e-9);
    // Points along the seam between the two halves, on the room side of the south wall.
    const seamPoints: Vec[] = Array.from({ length: 41 }, (_, i): Vec => [seamX, 0.05 + (0.9 * i) / 40, 0]);

    /** The room, with the south wall's halves ending at `leftEnd` and starting at `rightStart`. */
    function testRoom(leftEnd: number, rightStart: number): ProceduralScene {
        const builder = new SceneBuilder();
        const grey = builder.addDiffuse('grey', [0.5, 0.5, 0.5]);
        builder.addBox('floor', grey, [-wall, -wall, -wall], [1 + wall, 0, 1 + wall]);
        builder.addBox('ceiling', grey, [-wall, 1, -wall], [1 + wall, 1 + wall, 1 + wall]);
        builder.addBox('west', grey, [-wall, -wall, -wall], [0, 1 + wall, 1 + wall]);
        builder.addBox('east', grey, [1, -wall, -wall], [1 + wall, 1 + wall, 1 + wall]);
        builder.addBox('north', grey, [-wall, -wall, 1], [1 + wall, 1 + wall, 1 + wall]);
        builder.addBox('south-left', grey, [-wall, -wall, -wall], [leftEnd, 1 + wall, 0]);
        builder.addBox('south-right', grey, [rightStart, -wall, -wall], [1 + wall, 1 + wall, 0]);
        return builder.build();
    }

    it('a room whose halves overlap passes both checks', () => {
        const scene = testRoom(seamX + 0.0625, seamX - 0.0625);
        expect(countLeaks(scene, origin, seamPoints, insideRoom)).toEqual({ escaped: 0, outside: 0 });
        expect(flushPairs(axisAlignedBoxes(scene, /^south-/))).toEqual([]);
    });

    it('the ray check finds a 2 µm slot between the halves', () => {
        const scene = testRoom(seamX - 1e-6, seamX + 1e-6);
        const leaks = countLeaks(scene, origin, seamPoints, insideRoom);
        expect(leaks.escaped + leaks.outside).toBeGreaterThan(0);
    });

    it('halves that touch exactly pass the ray check, which is why the flush check exists, and the flush check catches them', () => {
        const scene = testRoom(seamX, seamX);
        expect(countLeaks(scene, origin, seamPoints, insideRoom)).toEqual({ escaped: 0, outside: 0 });
        expect(flushPairs(axisAlignedBoxes(scene, /^south-/))).toEqual(['south-left / south-right']);
    });

    it('a name pattern that matches no boxes fails instead of passing with nothing checked', () => {
        expect(() => axisAlignedBoxes(testRoom(seamX, seamX), /^no-such-box/)).toThrow(/matched no boxes/);
    });
});
