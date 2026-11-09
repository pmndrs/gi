import { describe, expect, it } from 'vitest';
import { Bone, BoxGeometry, Group, Mesh, MeshBasicMaterial, Object3D, PerspectiveCamera, Scene, Skeleton, SkinnedMesh, Vector3 } from 'three';
import { MatrixWatch } from './matrix-watch.ts';

function sceneWithBox(): { scene: Scene; parent: Group; box: Mesh } {
    const scene = new Scene();
    const parent = new Group();
    const box = new Mesh(new BoxGeometry(), new MeshBasicMaterial());
    parent.add(box);
    scene.add(parent);
    scene.updateMatrixWorld();
    return { scene, parent, box };
}

describe('MatrixWatch (T3.0 spike)', () => {
    it('reports nothing for a still scene, frame after frame', () => {
        const { scene } = sceneWithBox();
        const watch = new MatrixWatch();
        watch.installTree(scene);
        for (let frame = 0; frame < 5; frame++) {
            scene.updateMatrixWorld();
            expect(watch.takeMoved()).toEqual([]);
        }
    });

    it('reports a moved object once, in the update that computed its matrix', () => {
        const { scene, box } = sceneWithBox();
        const watch = new MatrixWatch();
        watch.installTree(scene);
        box.position.x = 1;
        scene.updateMatrixWorld();
        scene.updateMatrixWorld();
        expect(watch.takeMoved()).toEqual([box]);
        scene.updateMatrixWorld();
        expect(watch.takeMoved()).toEqual([]);
    });

    it("reports children when only their parent moved", () => {
        const { scene, parent, box } = sceneWithBox();
        const watch = new MatrixWatch();
        watch.installTree(scene);
        parent.rotation.y = 0.5;
        scene.updateMatrixWorld();
        expect(new Set(watch.takeMoved())).toEqual(new Set([parent, box]));
    });

    it('reports a matrix three already updated elsewhere (updateWorldMatrix, getWorldPosition)', () => {
        const { scene, box } = sceneWithBox();
        const watch = new MatrixWatch();
        watch.installTree(scene);
        box.position.z = 2;
        box.getWorldPosition(new Vector3()); // updates matrixWorld through updateWorldMatrix
        scene.updateMatrixWorld();
        expect(watch.takeMoved()).toEqual([box]);
    });

    it('reports a manual matrix on an object with matrixAutoUpdate off', () => {
        const { scene, box } = sceneWithBox();
        const watch = new MatrixWatch();
        watch.installTree(scene);
        box.matrixAutoUpdate = false;
        box.matrix.makeTranslation(0, 3, 0);
        box.matrixWorldNeedsUpdate = true;
        scene.updateMatrixWorld();
        expect(watch.takeMoved()).toEqual([box]);
    });

    it("keeps subclasses' own updateMatrixWorld (camera inverse, skinned bind inverse)", () => {
        const scene = new Scene();
        const camera = new PerspectiveCamera();
        scene.add(camera);
        const bone = new Bone();
        const skinned = new SkinnedMesh(new BoxGeometry(), new MeshBasicMaterial());
        skinned.add(bone);
        skinned.bind(new Skeleton([bone]));
        scene.add(skinned);
        const watch = new MatrixWatch();
        watch.installTree(scene);
        camera.position.set(1, 2, 3);
        skinned.position.set(4, 0, 0);
        scene.updateMatrixWorld();
        expect(camera.matrixWorldInverse.clone().multiply(camera.matrixWorld).equals(new Object3D().matrix)).toBe(true);
        // Bind mode 'attached' (the default): SkinnedMesh.updateMatrixWorld sets bindMatrixInverse from the new matrixWorld.
        const expected = skinned.matrixWorld.clone().invert().elements;
        skinned.bindMatrixInverse.elements.forEach((value, index) => expect(value).toBeCloseTo(expected[index], 12));
        expect(skinned.bindMatrixInverse.elements[12]).toBeCloseTo(-4, 12);
        expect(new Set(watch.takeMoved())).toEqual(new Set([camera, skinned, bone]));
    });

    it('uninstall gives the object its method back and forgets it', () => {
        const { scene, box } = sceneWithBox();
        const watch = new MatrixWatch();
        watch.installTree(scene);
        expect(Object.prototype.hasOwnProperty.call(box, 'updateMatrixWorld')).toBe(true);
        box.position.x = 5;
        scene.updateMatrixWorld();
        watch.uninstall(box);
        // Assigned back, not deleted (matrix-watch.ts says why).
        expect(box.updateMatrixWorld).toBe(Mesh.prototype.updateMatrixWorld);
        expect(watch.takeMoved()).toEqual([]);
        box.position.x = 6;
        scene.updateMatrixWorld();
        expect(watch.takeMoved()).toEqual([]);
        expect(box.matrixWorld.elements[12]).toBe(6);
    });

    it('clones of a watched object are not watched (copy does not carry own methods)', () => {
        const { scene, box } = sceneWithBox();
        const watch = new MatrixWatch();
        watch.installTree(scene);
        const clone = box.clone();
        expect(Object.prototype.hasOwnProperty.call(clone, 'updateMatrixWorld')).toBe(false);
    });
});
