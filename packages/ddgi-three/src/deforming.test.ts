import { Bone, Box3, BoxGeometry, Float32BufferAttribute, InstancedMesh, Mesh, MeshStandardMaterial, Skeleton, SkinnedMesh, Uint16BufferAttribute, Vector3 } from 'three';
import { describe, expect, it } from 'vitest';
import { CpuPose, deformBoundsFor, deformKind } from './deforming.ts';

/** A box with one absolute morph target (every vertex pushed out along x by 1, so the box is 1 wider on each side). */
function absoluteMorphBox(): Mesh {
    const geometry = new BoxGeometry(1, 1, 1, 2, 2, 2);
    const position = geometry.getAttribute('position');
    const target = new Float32Array(position.count * 3);
    for (let vertex = 0; vertex < position.count; vertex++) {
        const x = position.getX(vertex);
        target[vertex * 3] = x + Math.sign(x);
        target[vertex * 3 + 1] = position.getY(vertex);
        target[vertex * 3 + 2] = position.getZ(vertex);
    }
    geometry.morphAttributes.position = [new Float32BufferAttribute(target, 3)];
    geometry.morphTargetsRelative = false;
    const mesh = new Mesh(geometry, new MeshStandardMaterial());
    mesh.updateMorphTargets();
    return mesh;
}

/** The same box, skinned to three bones in a chain, with its morph target too. */
function skinnedMorphBox(): { mesh: SkinnedMesh; bones: Bone[] } {
    const source = absoluteMorphBox();
    const geometry = source.geometry;
    const position = geometry.getAttribute('position');
    const skinIndex: number[] = [];
    const skinWeight: number[] = [];
    for (let vertex = 0; vertex < position.count; vertex++) {
        const t = position.getY(vertex) + 0.5;
        skinIndex.push(0, 1, 2, 0);
        skinWeight.push((1 - t) * 0.5, 0.5, t * 0.5, 0);
    }
    geometry.setAttribute('skinIndex', new Uint16BufferAttribute(skinIndex, 4));
    geometry.setAttribute('skinWeight', new Float32BufferAttribute(skinWeight, 4));
    const bones = [new Bone(), new Bone(), new Bone()];
    bones[1].position.y = 0.3;
    bones[2].position.y = 0.3;
    bones[0].add(bones[1]);
    bones[1].add(bones[2]);
    const mesh = new SkinnedMesh(geometry, source.material);
    mesh.position.set(2, 0, -1);
    mesh.add(bones[0]);
    mesh.bind(new Skeleton(bones));
    mesh.updateMorphTargets();
    return { mesh, bones };
}

function threePose(mesh: Mesh): Float32Array {
    const position = mesh.geometry.getAttribute('position');
    const out = new Float32Array(position.count * 3);
    const vertex = new Vector3();
    for (let index = 0; index < position.count; index++) mesh.getVertexPosition(index, vertex).toArray(out, index * 3);
    return out;
}

function largestDifference(a: ArrayLike<number>, b: ArrayLike<number>): number {
    let largest = 0;
    for (let index = 0; index < a.length; index++) largest = Math.max(largest, Math.abs(a[index] - b[index]));
    return largest;
}

describe('deforming meshes', () => {
    it('tells the kinds apart, and leaves instanced and opted-out meshes alone', () => {
        expect(deformKind(absoluteMorphBox())).toBe('morphed');
        expect(deformKind(skinnedMorphBox().mesh)).toBe('skinned');
        const plain = new Mesh(new BoxGeometry(), new MeshStandardMaterial());
        expect(deformKind(plain)).toBeNull();
        plain.userData.giDeforming = true;
        expect(deformKind(plain)).toBe('animated');
        const optedOut = absoluteMorphBox();
        optedOut.userData.giDeforming = false;
        expect(deformKind(optedOut)).toBeNull();
        expect(deformKind(new InstancedMesh(new BoxGeometry(), new MeshStandardMaterial(), 2))).toBeNull();
    });

    it('bounds an absolute morph target exactly, and takes a Box3 the host gives', () => {
        const mesh = absoluteMorphBox();
        expect(deformBoundsFor(mesh, 'morphed')).toEqual({ min: [-1.5, -0.5, -0.5], max: [1.5, 0.5, 0.5] });
        mesh.userData.giDeformBounds = new Box3(new Vector3(-3, -2, -1), new Vector3(3, 2, 1));
        expect(deformBoundsFor(mesh, 'morphed')).toEqual({ min: [-3, -2, -1], max: [3, 2, 1] });
        // Not a box: ignored, and the estimate is used.
        mesh.userData.giDeformBounds = { min: [0, 0], max: [1, 1, 1] };
        expect(deformBoundsFor(mesh, 'morphed').min[0]).toBe(-1.5);
    });

    it('works poses out as three does: absolute morph targets, and skinning on top of them', () => {
        const morphed = absoluteMorphBox();
        const morphedPose = new CpuPose(morphed, 'morphed', deformBoundsFor(morphed, 'morphed'));
        expect(morphedPose.nextPose()).toBeNull();
        morphed.morphTargetInfluences![0] = 0.25;
        expect(largestDifference(morphedPose.nextPose()!, threePose(morphed))).toBeLessThan(1e-6);

        const { mesh, bones } = skinnedMorphBox();
        const skinnedPose = new CpuPose(mesh, 'skinned', deformBoundsFor(mesh, 'skinned'));
        mesh.updateMatrixWorld(true);
        expect(skinnedPose.nextPose()).toBeNull();
        bones[1].rotation.z = 0.6;
        bones[2].rotation.x = -0.4;
        mesh.morphTargetInfluences![0] = 0.5;
        mesh.updateMatrixWorld(true);
        const pose = skinnedPose.nextPose()!;
        expect(largestDifference(pose, threePose(mesh))).toBeLessThan(1e-5);
        expect(largestDifference(pose, mesh.geometry.getAttribute('position').array)).toBeGreaterThan(0.1);
        // Unchanged inputs: nothing to send.
        expect(skinnedPose.nextPose()).toBeNull();
    });
});
