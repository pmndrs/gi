import { BoxGeometry, BufferAttribute, Group, Mesh, MeshStandardMaterial, Scene, Vector3 } from 'three';
import { describe, expect, it } from 'vitest';
import type { ManifestMotion, SceneManifest } from '../../../../scenes/src/manifest';
import type { LoadedScene } from '../scene/load-scene';
import { AddedCabinets, MAX_ADDED_CABINETS } from '../scene/added-cabinets';
import { MAX_WIND_REACH_M, ObjectDriver } from './motion-drivers';
import { MotionPlayer } from './motion-player';

/** A door panel like closed-door's: a unit box scaled to 0.04 × 2 × 0.9, its lit-side z0 edge at (4.08, 0, 1.55). */
function doorPanel(): Mesh {
    const door = new Mesh(new BoxGeometry(1, 1, 1), new MeshStandardMaterial());
    door.name = 'door-panel';
    door.scale.set(0.04, 2, 0.9);
    door.position.set(4.1, 1, 2);
    return door;
}

/** The fewest parts of a loaded scene the motion player reads. */
function loadedWith(content: Group, manifest: Partial<SceneManifest>): LoadedScene {
    const scene = new Scene();
    scene.add(content);
    scene.updateMatrixWorld(true);
    return { manifest: { lights: [], motions: [], ...manifest } as SceneManifest, scene, content, manifestLights: [] } as unknown as LoadedScene;
}

const swing: ManifestMotion = {
    id: 'door-swing',
    label: 'door',
    target: 'object:door-panel',
    loop: true,
    pivot: [4.08, 0, 1.55],
    keyframes: [
        { t: 0, rotationDeg: [0, 0, 0] },
        { t: 1, rotationDeg: [0, -90, 0] },
    ],
};

describe('object motions (T3.8)', () => {
    it('turns an object about its pivot from where it stood, and puts it back', () => {
        const content = new Group();
        const door = doorPanel();
        content.add(door);
        content.position.set(0, 0, 0);
        new Scene().add(content);
        content.updateMatrixWorld(true);
        const driver = new ObjectDriver(door, null, () => 1);
        const saved = driver.save();
        driver.apply({ rotationDeg: [0, -90, 0] }, swing);
        // A quarter turn about +y swings the panel from the hinge into the lit room (-x): its centre was 0.45 m along +z from the hinge.
        const centre = new Vector3().setFromMatrixPosition(door.matrixWorld);
        expect(centre.x).toBeCloseTo(4.08 - 0.45, 6);
        expect(centre.z).toBeCloseTo(1.55 + 0.02, 6);
        // Twice in a row gives the same place: every state starts from the rest, so nothing drifts.
        driver.apply({ rotationDeg: [0, -90, 0] }, swing);
        expect(new Vector3().setFromMatrixPosition(door.matrixWorld).x).toBeCloseTo(4.08 - 0.45, 6);
        driver.restore(saved);
        expect(door.position.toArray()).toEqual([4.1, 1, 2]);
        expect(door.scale.toArray()).toEqual([0.04, 2, 0.9]);
    });

    it('slides an object by an offset, inside a parent that is itself moved', () => {
        const content = new Group();
        content.position.set(10, 0, 0);
        const box = doorPanel();
        content.add(box);
        new Scene().add(content);
        content.updateMatrixWorld(true);
        const driver = new ObjectDriver(box, null, () => 1);
        driver.apply({ position: [0, 0, 1] }, { ...swing, pivot: undefined });
        expect(new Vector3().setFromMatrixPosition(box.matrixWorld).toArray()).toEqual([14.1, 1, 3]);
        expect(box.position.toArray()).toEqual([4.1, 1, 3]);
    });

    it('adds a prop while its motion plays and takes it away when it stops', () => {
        const content = new Group();
        const prop = { name: 'cabinet', shape: 'box' as const, size: [0.45, 0.85, 0.9] as [number, number, number], position: [-1.7, 0.43, -0.4] as [number, number, number], color: [0.45, 0.16, 0.06] as [number, number, number] };
        const slide: ManifestMotion = { id: 'cabinet-slide', label: 'slide', target: 'object:cabinet', loop: true, keyframes: [{ t: 0, position: [0, 0, 0] }, { t: 1, position: [0, 0, 1] }] };
        const player = new MotionPlayer(loadedWith(content, { motions: [slide], props: [prop] }));
        expect(player.motions.map((motion) => motion.id)).toEqual(['cabinet-slide']);
        expect(content.getObjectByName('cabinet')).toBeUndefined();
        player.play('cabinet-slide', 0);
        player.update(500);
        const cabinet = content.getObjectByName('cabinet')!;
        expect(cabinet).toBeDefined();
        expect(cabinet.position.z).toBeCloseTo(-0.4 + 0.5, 6);
        expect(player.changedBy(slide)).toBe(cabinet);
        player.stop();
        expect(content.getObjectByName('cabinet')).toBeUndefined();
        player.dispose();
    });

    it('sways meshes with a wind weight, marks them deforming from the start, and follows the wind strength', () => {
        const content = new Group();
        const tree = new Group();
        tree.name = 'wind-tree';
        // Two vertices: the foot (weight 0) and the crown (weight 1).
        const geometry = new BoxGeometry(1, 1, 1);
        const weights = new Float32Array(geometry.getAttribute('position').count * 4);
        for (let vertex = 0; vertex < weights.length / 4; vertex++) weights[vertex * 4 + 3] = geometry.getAttribute('position').getY(vertex) > 0 ? 1 : 0;
        geometry.setAttribute('color', new BufferAttribute(weights, 4));
        const crown = new Mesh(geometry, new MeshStandardMaterial());
        tree.add(crown);
        content.add(tree);
        const sway: ManifestMotion = { id: 'wind-lean', label: 'lean', target: 'object:wind-tree', interpolation: 'step', loop: true, keyframes: [{ t: 0, sway: 0 }, { t: 1, sway: 1 }, { t: 2, sway: 0 }] };
        const player = new MotionPlayer(loadedWith(content, { motions: [sway] }));
        // Marked before GI copies the scene, with room for the strongest wind.
        expect(crown.userData.giDeforming).toBe(true);
        expect(crown.userData.giDeformBounds.max[0]).toBeCloseTo(0.5 + MAX_WIND_REACH_M, 6);
        expect(player.hasWind).toBe(true);
        const position = geometry.getAttribute('position');
        const top = Array.from({ length: position.count }, (_, vertex) => vertex).find((vertex) => position.getY(vertex) > 0)!;
        const foot = Array.from({ length: position.count }, (_, vertex) => vertex).find((vertex) => position.getY(vertex) < 0)!;
        const restTop = position.getX(top);
        const restFoot = position.getX(foot);
        player.windStrength = 0.5;
        player.play('wind-lean', 0);
        player.update(1500);
        // At full lean and half strength the crown moves half the strongest reach along the wind (+x); the foot stays.
        expect(position.getX(top)).toBeCloseTo(restTop + 0.5 * MAX_WIND_REACH_M, 6);
        expect(position.getX(foot)).toBeCloseTo(restFoot, 6);
        // The slider acts on a tree held leaning too.
        player.windStrength = 1;
        expect(position.getX(top)).toBeCloseTo(restTop + MAX_WIND_REACH_M, 6);
        player.stop();
        expect(position.getX(top)).toBeCloseTo(restTop, 6);
    });
});

describe('added cabinets (T3.8)', () => {
    it('"add" makes a new shape, "copy" reuses the last one, up to 100, and "remove" takes them all away', () => {
        const parent = new Group();
        const cabinets = new AddedCabinets(parent);
        cabinets.add();
        cabinets.copy();
        cabinets.copy();
        cabinets.add();
        const meshes = parent.children as Mesh[];
        expect(meshes).toHaveLength(4);
        expect(meshes[1].geometry).toBe(meshes[0].geometry);
        expect(meshes[2].geometry).toBe(meshes[0].geometry);
        expect(meshes[3].geometry).not.toBe(meshes[0].geometry);
        expect(cabinets.shapes).toBe(2);
        for (let index = 0; index < 200; index++) cabinets.copy();
        expect(cabinets.count).toBe(MAX_ADDED_CABINETS);
        // None share a place.
        const places = new Set(meshes.map((mesh) => mesh.position.toArray().map((value) => value.toFixed(3)).join(',')));
        expect(places.size).toBe(MAX_ADDED_CABINETS);
        cabinets.clear();
        expect(parent.children).toHaveLength(0);
        cabinets.dispose();
    });
});
