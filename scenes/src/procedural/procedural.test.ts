import { describe, expect, it } from 'vitest';
import type { ProceduralScene, SceneInstanceData, SceneMeshData } from '../scene-data';
import { buildCabinetRow, PANEL_THICKNESS_M } from './cabinet-row';
import { buildCornellBox, CORNELL_DEPTH_M, CORNELL_HEIGHT_M, CORNELL_WIDTH_M } from './cornell-box';
import { buildDarkRoom, DARK_ROOM_SIZE_M, DRAG_BOX_NAME, DRAG_BOX_SIZE_M, DRAG_BOX_START } from './dark-room';
import { buildEmissiveRoom, EMISSIVE_ROOM_MATERIALS } from './emissive-room';
import {
    buildFurnaceRoom,
    FURNACE_ALBEDO,
    FURNACE_EMISSION,
    FURNACE_EXPECTED_RADIANCE,
} from './furnace-room';
import { createUnitBox, createUnitQuad, sceneBounds, triangleCount } from './geometry';
import { buildProcedural, proceduralBuilders } from './index';
import { buildThinWall, THIN_WALL_EMBED_M, THIN_WALL_THICKNESS_M } from './thin-wall';
import { buildClosedDoor, buildTwoRoom } from './two-rooms';

/** Size of an instance of the unit box along its local axes, read from the matrix columns. */
function boxSize(instance: SceneInstanceData): [number, number, number] {
    const m = instance.transform;
    return [Math.hypot(m[0], m[1], m[2]), Math.hypot(m[4], m[5], m[6]), Math.hypot(m[8], m[9], m[10])];
}

function findInstance(scene: ProceduralScene, name: string): SceneInstanceData {
    const instance = scene.instances.find((candidate) => candidate.name === name);
    if (!instance) throw new Error(`no instance named ${name}`);
    return instance;
}

describe('unit meshes', () => {
    it('has outward counter-clockwise faces whose winding agrees with the stored normals', () => {
        const box = createUnitBox();
        expect(box.indices.length / 3).toBe(12);
        for (let t = 0; t < box.indices.length; t += 3) {
            const [a, b, c] = [box.indices[t], box.indices[t + 1], box.indices[t + 2]];
            const p = (i: number) => [box.positions[i * 3], box.positions[i * 3 + 1], box.positions[i * 3 + 2]];
            const [pa, pb, pc] = [p(a), p(b), p(c)];
            const e1 = [pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2]];
            const e2 = [pc[0] - pa[0], pc[1] - pa[1], pc[2] - pa[2]];
            const geometricNormal = [
                e1[1] * e2[2] - e1[2] * e2[1],
                e1[2] * e2[0] - e1[0] * e2[2],
                e1[0] * e2[1] - e1[1] * e2[0],
            ];
            const stored = [box.normals[a * 3], box.normals[a * 3 + 1], box.normals[a * 3 + 2]];
            const dot = geometricNormal[0] * stored[0] + geometricNormal[1] * stored[1] + geometricNormal[2] * stored[2];
            expect(dot).toBeGreaterThan(0);
            // The face centre lies on the same side as the normal (the cube is centred on the origin).
            const centreDot = pa[0] * stored[0] + pa[1] * stored[1] + pa[2] * stored[2];
            expect(centreDot).toBeGreaterThan(0);
        }
    });

    it('has a unit quad facing +Z', () => {
        const quad = createUnitQuad();
        expect(quad.indices.length / 3).toBe(2);
        expect(Array.from(quad.normals.slice(0, 3))).toEqual([0, 0, 1]);
    });
});

describe.each(Object.keys(proceduralBuilders))('builder %s', (name) => {
    const scene = buildProcedural(name);

    it('has valid references and finite numbers', () => {
        expect(scene.instances.length).toBeGreaterThan(0);
        for (const instance of scene.instances) {
            expect(instance.mesh).toBeLessThan(scene.meshes.length);
            expect(instance.material).toBeLessThan(scene.materials.length);
            expect(instance.transform).toHaveLength(16);
            expect(instance.transform.every(Number.isFinite)).toBe(true);
        }
        for (const mesh of scene.meshes) {
            expect(mesh.normals.length).toBe(mesh.positions.length);
            expect(mesh.uvs.length / 2).toBe(mesh.positions.length / 3);
            expect(Math.max(...mesh.indices)).toBeLessThan(mesh.positions.length / 3);
        }
    });

    it('uses albedo in range and unique instance names', () => {
        for (const material of scene.materials) {
            expect(Math.max(...material.albedo)).toBeLessThanOrEqual(1);
            expect(Math.min(...material.albedo)).toBeGreaterThanOrEqual(0);
        }
        const names = scene.instances.map((instance) => instance.name);
        expect(new Set(names).size).toBe(names.length);
    });

    it('is built in realistic metres', () => {
        const { min, max } = sceneBounds(scene);
        // The open ground is a 200 m slab with a 1 m cube on it: open land under the
        // sky, deliberately wide and flat. Every other scene is room-sized.
        const limits = name === 'open-ground' ? { smallest: 1, largest: 250 } : { smallest: 2, largest: 20 };
        for (let axis = 0; axis < 3; axis++) {
            expect(max[axis] - min[axis]).toBeGreaterThan(limits.smallest);
            expect(max[axis] - min[axis]).toBeLessThan(limits.largest);
        }
    });
});

describe('buildProcedural', () => {
    it('names the known builders when asked for an unknown one', () => {
        expect(() => buildProcedural('nonexistent')).toThrow(/Known builders: cornell-box/);
    });
});

describe('cornell-box', () => {
    it('has the standard 5.55 m by 5.49 m by 5.59 m interior and a ceiling light of 1.3 m by 1.05 m', () => {
        const scene = buildCornellBox();
        const light = findInstance(scene, 'ceiling-light');
        const [sizeX, sizeY] = [Math.hypot(...light.transform.slice(0, 3)), Math.hypot(...light.transform.slice(4, 7))];
        expect(sizeX).toBeCloseTo(1.3);
        expect(sizeY).toBeCloseTo(1.05);
        expect(light.transform[13]).toBeCloseTo(CORNELL_HEIGHT_M - 0.002);
        // Facing down: the quad's normal (local +Z, third column) maps to world -Y.
        expect(light.transform[9]).toBeLessThan(0);
        expect(scene.materials[light.material].emissive).toEqual([17, 12, 4]);
        expect(scene.lights).toHaveLength(0);
        expect(CORNELL_WIDTH_M).toBeCloseTo(5.55);
        expect(CORNELL_DEPTH_M).toBeCloseTo(5.592);
    });

    it('swaps the emitter for one point light with the point option', () => {
        const scene = buildCornellBox({ light: 'point' });
        expect(scene.instances.some((instance) => instance.name === 'ceiling-light')).toBe(false);
        expect(scene.materials.every((material) => material.emissive.every((channel) => channel === 0))).toBe(true);
        expect(scene.lights).toHaveLength(1);
        expect(scene.lights[0].type).toBe('point');
    });

    it('rejects an unknown light option', () => {
        expect(() => buildCornellBox({ light: 'laser' })).toThrow(/"light"/);
    });
});

describe('furnace-room', () => {
    it('uses one material, whose closed-form answer is emission / (1 - albedo)', () => {
        const scene = buildFurnaceRoom();
        expect(scene.materials).toHaveLength(1);
        expect(scene.materials[0].albedo).toEqual([FURNACE_ALBEDO, FURNACE_ALBEDO, FURNACE_ALBEDO]);
        expect(scene.materials[0].emissive).toEqual([FURNACE_EMISSION, FURNACE_EMISSION, FURNACE_EMISSION]);
        expect(FURNACE_EXPECTED_RADIANCE).toBeCloseTo(0.5);
        // The same number from summing the bounce series directly.
        let seriesSum = 0;
        for (let bounce = 0; bounce < 60; bounce++) seriesSum += FURNACE_EMISSION * FURNACE_ALBEDO ** bounce;
        expect(seriesSum).toBeCloseTo(FURNACE_EXPECTED_RADIANCE, 10);
    });

    it('has no lights, since the emission is the light', () => {
        expect(buildFurnaceRoom().lights).toEqual([]);
    });
});

describe('emissive-room', () => {
    const scene = buildEmissiveRoom();

    it('has no lights: the strips and the screen are its only light', () => {
        expect(scene.lights).toEqual([]);
        const emitters = scene.instances.filter((instance) => Math.max(...scene.materials[instance.material].emissive) > 0).map((instance) => instance.name);
        expect(emitters.sort()).toEqual(['screen-panel', 'strip-east', 'strip-west']);
    });

    it('names the emitter materials as pages look them up', () => {
        expect(scene.materials[findInstance(scene, 'strip-west').material].name).toBe(EMISSIVE_ROOM_MATERIALS.strip);
        expect(scene.materials[findInstance(scene, 'strip-east').material].name).toBe(EMISSIVE_ROOM_MATERIALS.strip);
        expect(scene.materials[findInstance(scene, 'screen-panel').material].name).toBe(EMISSIVE_ROOM_MATERIALS.screen);
    });
});

describe('thin-wall', () => {
    it('has a partition 3 cm thick spanning the whole room, with a light on one side only', () => {
        const scene = buildThinWall();
        const wall = findInstance(scene, 'thin-wall');
        const [thickness, height, depth] = boxSize(wall);
        expect(thickness).toBeCloseTo(THIN_WALL_THICKNESS_M);
        // The room is 2.8 m by 4 m; the wall runs 1 cm into the shell at each end.
        expect(height).toBeCloseTo(2.8 + 2 * THIN_WALL_EMBED_M);
        expect(depth).toBeCloseTo(4 + 2 * THIN_WALL_EMBED_M);
        expect(scene.lights).toHaveLength(1);
        const partitionX = wall.transform[12];
        expect(scene.lights[0].type === 'point' && scene.lights[0].position[0]).toBeLessThan(partitionX);
    });
});

describe('two-room and closed-door', () => {
    it('shares one shell, and the closed door adds exactly one panel in the doorway', () => {
        const open = buildTwoRoom();
        const closed = buildClosedDoor();
        expect(closed.instances.length).toBe(open.instances.length + 1);
        const door = findInstance(closed, 'door-panel');
        const [thickness, height, width] = boxSize(door);
        expect(thickness).toBeCloseTo(0.04);
        // A little bigger than the 2.1 m by 0.9 m opening so it overlaps the frame.
        expect(height).toBeGreaterThan(2.1);
        expect(width).toBeGreaterThan(0.9);
        expect(height).toBeLessThan(2.12);
        expect(width).toBeLessThan(0.92);
    });

    it('lights only the left room', () => {
        for (const scene of [buildTwoRoom(), buildClosedDoor()]) {
            expect(scene.lights).toHaveLength(1);
            const light = scene.lights[0];
            expect(light.type === 'point' && light.position[0]).toBeLessThan(4);
        }
    });
});

describe('cabinet-row', () => {
    it('is made of 18 mm boards', () => {
        const scene = buildCabinetRow();
        const boardName = /^(tall|base|wall)-unit(-\d)?-(bottom|top|side-left|side-right|back|door)$/;
        const panels = scene.instances.filter((instance) => boardName.test(instance.name));
        expect(panels.length).toBeGreaterThan(40);
        for (const panel of panels) {
            const thinnest = Math.min(...boxSize(panel));
            expect(thinnest, panel.name).toBeCloseTo(PANEL_THICKNESS_M, 5);
        }
    });

    it('leaves out the doors with closedFronts false', () => {
        const closed = buildCabinetRow();
        const open = buildCabinetRow({ closedFronts: false });
        const doors = (scene: ProceduralScene) => scene.instances.filter((instance) => instance.name.endsWith('-door')).length;
        expect(doors(closed)).toBe(9);
        expect(doors(open)).toBe(0);
    });

    it('adds the shelf on legs and the small objects only with objects true, after everything else', () => {
        const plain = buildCabinetRow();
        const withObjects = buildCabinetRow({ objects: true });
        expect(plain.instances.some((instance) => instance.name.startsWith('shelf-'))).toBe(false);
        expect(withObjects.instances.filter((instance) => instance.name.startsWith('shelf-leg-'))).toHaveLength(4);
        // The plain scene's instances and materials come first, unchanged, so its references still hold.
        expect(withObjects.instances.slice(0, plain.instances.length)).toEqual(plain.instances);
        expect(withObjects.materials.slice(0, plain.materials.length)).toEqual(plain.materials);
        expect(() => buildCabinetRow({ objects: 'yes' })).toThrow(/objects/);
    });

    it('is a hollow volume: no board occupies the middle of a base unit', () => {
        const scene = buildCabinetRow();
        // Centre of the first base unit: x 0.6..1.2, y 0.1..0.82, z 0..0.58.
        const centre = [0.9, 0.46, 0.29];
        for (const instance of scene.instances.filter((candidate) => candidate.name.startsWith('base-unit-0'))) {
            const m = instance.transform;
            const half = boxSize(instance).map((extent) => extent / 2);
            const inside = [0, 1, 2].every((axis) => Math.abs(centre[axis] - m[12 + axis]) < half[axis]);
            expect(inside, instance.name).toBe(false);
        }
    });
});

describe('dark-room', () => {
    it('has the box the C4 page drags, standing on the floor inside the room', () => {
        const scene = buildDarkRoom();
        const box = findInstance(scene, DRAG_BOX_NAME);
        boxSize(box).forEach((extent) => expect(extent).toBeCloseTo(DRAG_BOX_SIZE_M));
        expect([box.transform[12], box.transform[13], box.transform[14]]).toEqual([...DRAG_BOX_START]);
        expect(DRAG_BOX_START[1] - DRAG_BOX_SIZE_M / 2).toBe(0);
        for (const axis of [0, 2]) {
            expect(DRAG_BOX_START[axis] - DRAG_BOX_SIZE_M / 2).toBeGreaterThan(0);
            expect(DRAG_BOX_START[axis] + DRAG_BOX_SIZE_M / 2).toBeLessThan(DARK_ROOM_SIZE_M[axis]);
        }
    });

    it('has a lamp and a flashlight and nothing else that gives light, so with both off the room is black', () => {
        const scene = buildDarkRoom();
        expect(scene.lights.map((light) => `${light.type} ${light.label}`)).toEqual(['point ceiling-lamp', 'spot flashlight']);
        for (const material of scene.materials) expect(material.emissive).toEqual([0, 0, 0]);
    });
});

describe('triangle counts (recorded in PROGRESS)', () => {
    it('matches 12 triangles per box and 2 per quad', () => {
        // Meshes of their own (the glass wall's boxes, the quad with a second uv set) count their own triangles.
        const trianglesOf = (mesh: SceneMeshData) => (mesh.name === 'unit-quad' ? 2 : mesh.name === 'unit-box' ? 12 : mesh.indices.length / 3);
        for (const name of Object.keys(proceduralBuilders)) {
            const scene = buildProcedural(name);
            const expected = scene.instances.reduce((total, instance) => total + trianglesOf(scene.meshes[instance.mesh]), 0);
            expect(triangleCount(scene)).toBe(expected);
        }
    });
});
