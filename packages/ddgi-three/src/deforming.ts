/**
 * Deforming meshes in the three adapter (T3.7; docs/design/m3-moving.md §6
 * and §7): which meshes deform, the box their poses stay in, and where each
 * frame's pose comes from.
 *
 * **Which meshes.** A mesh that contributes to GI deforms when it is:
 * - a `SkinnedMesh` with a skeleton (`skinned`);
 * - a mesh whose geometry has position morph targets (`morphed`);
 * - a plain mesh with `userData.giDeforming = true`, whose position attribute
 *   the host animates on the CPU (`animated`; the viewer's wind tree).
 *
 * `userData.giDeforming = false` keeps a skinned or morphed mesh in its bind
 * pose. Instanced meshes never deform here. Skinned meshes contribute to GI
 * only when they opt in (`giContributes`, user-data.ts), so a character
 * costs nothing by default.
 *
 * **Where a pose comes from.** {@link PoseSource} is the seam, and
 * {@link poseSourceFor} chooses:
 * - **GPU** (`GpuSkinningPose`, gpu-skinning.ts; T3.3): a skinned mesh,
 *   without position morphs, when the host gave the renderer
 *   (`ThreeGIOptions.renderer`). three's own `computeSkinning` writes the
 *   positions into a storage `ExternalTexture` we own, and the platform gets
 *   that texture: no position reaches the CPU.
 * - **CPU** ({@link CpuPose}) for the rest: when the pose's inputs change
 *   (bone matrices, morph influences, the position attribute's version), it
 *   works the positions out on the CPU, the way three's own
 *   `getVertexPosition` does, and hands them to `setDeformedPositions` as a
 *   `Float32Array`.
 *
 * Either way the platform refits the mesh's tree from the pose in the next
 * frame (T3.3), so a pose costs no tree build.
 */

import { Matrix4, type BufferAttribute, type BufferGeometry, type InterleavedBufferAttribute, type Mesh, type SkinnedMesh } from 'three';
import type { Aabb, DeformedPositions } from '@pmndrs/ddgi-core';
import { readFloatAttribute } from './convert.ts';
import { GpuSkinningPose, type GpuPoseContext } from './gpu-skinning.ts';
import { deformBoundsFlag, deformingFlag } from './user-data.ts';

type Attribute = BufferAttribute | InterleavedBufferAttribute;

export type DeformKind = 'skinned' | 'morphed' | 'animated';

/**
 * How a mesh deforms, or null when GI keeps its shape fixed. The caller also
 * requires the mesh to contribute to GI (scene-sync.ts): a receiver's shape
 * isn't traced, so its poses would cost CPU time for nothing.
 */
export function deformKind(mesh: Mesh): DeformKind | null {
    const flags = mesh as Mesh & { isInstancedMesh?: boolean; isSkinnedMesh?: boolean };
    if (flags.isInstancedMesh) return null;
    const flag = deformingFlag(mesh);
    if (flag === false) return null;
    if (flags.isSkinnedMesh && (mesh as SkinnedMesh).skeleton) return 'skinned';
    if (hasPositionMorphs(mesh.geometry) && mesh.morphTargetInfluences) return 'morphed';
    return flag === true ? 'animated' : null;
}

function hasPositionMorphs(geometry: BufferGeometry): boolean {
    const morphs = geometry.morphAttributes.position;
    return morphs !== undefined && morphs.length > 0;
}

/**
 * Of a skinned mesh's bind-pose box, how far the estimated pose box reaches
 * out on every side, as a share of the box's largest side. A limb swings
 * about its own length; half the body's largest side holds a raised arm or
 * a folded half. A guess: hosts that know better set `userData.giDeformBounds`.
 */
const SKINNED_REACH = 0.5;
/** The same for a CPU-animated mesh (a swaying tree): its poses stay near the rest pose. */
const ANIMATED_REACH = 0.25;

/**
 * The box, in the mesh's own space, that every pose stays inside
 * (`MeshDescriptor.deformBounds`). Too small is safe: the platform grows it
 * to hold a pose that leaves it, with one console warning. Too big makes the
 * changed regions larger than they need be.
 *
 * - `userData.giDeformBounds`, when the host gives one;
 * - morphed: exact for influences between 0 and 1: the rest box plus, on
 *   each axis, every target's most negative and most positive offset
 *   (an offset can only add when its influence is at most 1);
 * - skinned and animated: the rest box (with the morph reach, if any) grown
 *   by {@link SKINNED_REACH} or {@link ANIMATED_REACH}.
 */
export function deformBoundsFor(mesh: Mesh, kind: DeformKind): Aabb {
    const given = deformBoundsFlag(mesh);
    if (given) return given;
    const geometry = mesh.geometry;
    const position = geometry.getAttribute('position') as Attribute;
    const box = attributeBox(position);
    if (hasPositionMorphs(geometry)) addMorphReach(geometry, position, box);
    const reach = kind === 'skinned' ? SKINNED_REACH : kind === 'animated' ? ANIMATED_REACH : 0;
    if (reach > 0) {
        const pad = reach * Math.max(box[3] - box[0], box[4] - box[1], box[5] - box[2]);
        for (let axis = 0; axis < 3; axis++) {
            box[axis] -= pad;
            box[axis + 3] += pad;
        }
    }
    return { min: [box[0], box[1], box[2]], max: [box[3], box[4], box[5]] };
}

/** An attribute's box as `[minX, minY, minZ, maxX, maxY, maxZ]`. */
function attributeBox(attribute: Attribute): number[] {
    const box = [Infinity, Infinity, Infinity, -Infinity, -Infinity, -Infinity];
    for (let vertex = 0; vertex < attribute.count; vertex++) {
        const x = attribute.getX(vertex);
        const y = attribute.getY(vertex);
        const z = attribute.getZ(vertex);
        if (x < box[0]) box[0] = x;
        if (y < box[1]) box[1] = y;
        if (z < box[2]) box[2] = z;
        if (x > box[3]) box[3] = x;
        if (y > box[4]) box[4] = y;
        if (z > box[5]) box[5] = z;
    }
    return box;
}

/** Grows `box` by each morph target's reach: its most negative offset to min, its most positive to max, per axis. */
function addMorphReach(geometry: BufferGeometry, position: Attribute, box: number[]): void {
    const relative = geometry.morphTargetsRelative;
    for (const target of geometry.morphAttributes.position as Attribute[]) {
        const low = [0, 0, 0];
        const high = [0, 0, 0];
        for (let vertex = 0; vertex < position.count; vertex++) {
            // An absolute target holds positions; its offset is the target minus the rest pose (three's getVertexPosition).
            const dx = target.getX(vertex) - (relative ? 0 : position.getX(vertex));
            const dy = target.getY(vertex) - (relative ? 0 : position.getY(vertex));
            const dz = target.getZ(vertex) - (relative ? 0 : position.getZ(vertex));
            low[0] = Math.min(low[0], dx);
            low[1] = Math.min(low[1], dy);
            low[2] = Math.min(low[2], dz);
            high[0] = Math.max(high[0], dx);
            high[1] = Math.max(high[1], dy);
            high[2] = Math.max(high[2], dz);
        }
        for (let axis = 0; axis < 3; axis++) {
            box[axis] += low[axis];
            box[axis + 3] += high[axis];
        }
    }
}

/**
 * Where a deforming mesh's poses come from (the file comment). The scene sync
 * asks once per update while the mesh is shown, and passes whatever it gets
 * straight to `setDeformedPositions`.
 */
export interface PoseSource {
    /** The pose to send now, or null when it hasn't changed since the last one returned. */
    nextPose(): Float32Array | DeformedPositions | null;
    /** The mesh's shape was replaced (`updateMesh`): the platform's pose is the new rest pose again. */
    reset(): void;
    /** The mesh left GI: free what the source holds on the GPU. */
    dispose?(): void;
}

/**
 * The pose source for a deforming mesh (the file comment): three's GPU
 * skinning for a skinned mesh without position morphs when `gpu` (the
 * renderer and its device) is given; the CPU otherwise.
 */
export function poseSourceFor(mesh: Mesh, kind: DeformKind, bounds: Aabb, gpu: GpuPoseContext | null = null): PoseSource {
    if (kind === 'skinned' && gpu && !hasPositionMorphs(mesh.geometry)) return new GpuSkinningPose(mesh as SkinnedMesh, gpu);
    return new CpuPose(mesh, kind, bounds);
}

/**
 * Poses that differ from the last one sent by no more than this share of the
 * pose box's diagonal, on every coordinate, aren't sent. Why: a skinned rig
 * moved as a whole changes every bone matrix, but its mesh-space pose only
 * by float rounding, and each pose sent is a copy and a refit for the
 * platform. Compared against the last pose *sent*, so a slow drift still
 * goes out once it adds up.
 */
const POSE_TOLERANCE = 1e-6;

/**
 * A pose worked out on the CPU, in the mesh's own space, with three's
 * formulas (`Mesh.getVertexPosition`, `SkinnedMesh.applyBoneTransform`):
 *
 *   morphed = rest + Σ influence × offset       (offset = target, or target − rest when absolute)
 *   skinned = bindMatrixInverse × Σ weight × (bone.matrixWorld × boneInverse) × bindMatrix × morphed
 *
 * The bone products are made once per pose, not once per vertex as
 * `applyBoneTransform` does. The pose's inputs (the "key") are compared first,
 * so a still mesh costs a few compares per update.
 */
export class CpuPose implements PoseSource {
    private readonly mesh: Mesh;
    private readonly kind: DeformKind;
    /** The pose's inputs when last read: position version, morph influences, then (skinned) bind matrices and bone world matrices. */
    private key = new Float64Array(0);
    /** readKey's place in `key`, and whether a value it read differed (fields, so no closure is made each update). */
    private keyAt = 0;
    private keyChanged = false;
    /** The last pose returned (the rest pose at first: what the platform was created with). */
    private sent: Float32Array;
    /** Where the next pose is worked out; swapped with `sent` when it goes out. */
    private scratch: Float32Array;
    private readonly tolerance: number;
    /** Skinned: each bone's matrixWorld × boneInverse for this pose, 16 numbers per bone. */
    private boneMatrices = new Float64Array(0);
    private readonly boneScratch = new Matrix4();

    constructor(mesh: Mesh, kind: DeformKind, bounds: Aabb) {
        this.mesh = mesh;
        this.kind = kind;
        this.sent = Float32Array.from(readFloatAttribute(mesh.geometry.getAttribute('position') as Attribute, 3));
        this.scratch = new Float32Array(this.sent.length);
        const diagonal = Math.hypot(bounds.max[0] - bounds.min[0], bounds.max[1] - bounds.min[1], bounds.max[2] - bounds.min[2]);
        this.tolerance = Math.max(diagonal * POSE_TOLERANCE, 1e-9);
        // The key starts empty, so the first nextPose works the pose out: a skinned mesh is rarely in its bind pose.
    }

    reset(): void {
        this.sent = Float32Array.from(readFloatAttribute(this.mesh.geometry.getAttribute('position') as Attribute, 3));
        this.scratch = new Float32Array(this.sent.length);
        this.key = new Float64Array(0);
    }

    nextPose(): Float32Array | null {
        if (!this.readKey()) return null;
        this.computePose(this.scratch);
        if (withinTolerance(this.scratch, this.sent, this.tolerance)) return null;
        const pose = this.scratch;
        this.scratch = this.sent;
        this.sent = pose;
        return pose;
    }

    /** Reads the pose's inputs into the key; returns whether any changed. */
    private readKey(): boolean {
        const mesh = this.mesh;
        const position = mesh.geometry.getAttribute('position') as Attribute;
        const influences = this.kind === 'animated' ? null : mesh.morphTargetInfluences;
        const skeleton = this.kind === 'skinned' ? (mesh as SkinnedMesh).skeleton : null;
        const length = 1 + (influences?.length ?? 0) + (skeleton ? 32 + 16 * skeleton.bones.length : 0);
        this.keyChanged = false;
        this.keyAt = 0;
        if (this.key.length !== length) {
            this.key = new Float64Array(length);
            this.keyChanged = true;
        }
        this.writeKey(attributeVersion(position));
        if (influences) for (const influence of influences) this.writeKey(influence);
        if (skeleton) {
            const skinned = mesh as SkinnedMesh;
            for (const value of skinned.bindMatrix.elements) this.writeKey(value);
            for (const value of skinned.bindMatrixInverse.elements) this.writeKey(value);
            for (const bone of skeleton.bones) for (const value of bone.matrixWorld.elements) this.writeKey(value);
        }
        return this.keyChanged;
    }

    /** The key's next value: stored, and noted when it differs from the one read last time. */
    private writeKey(value: number): void {
        if (this.key[this.keyAt] !== value) {
            this.key[this.keyAt] = value;
            this.keyChanged = true;
        }
        this.keyAt++;
    }

    private computePose(out: Float32Array): void {
        const mesh = this.mesh;
        const geometry = mesh.geometry;
        const position = geometry.getAttribute('position') as Attribute;
        const count = Math.min(position.count, out.length / 3);
        const morphs = this.kind === 'animated' ? undefined : (geometry.morphAttributes.position as Attribute[] | undefined);
        const influences = mesh.morphTargetInfluences;
        const relative = geometry.morphTargetsRelative;
        const skinned = this.kind === 'skinned' ? (mesh as SkinnedMesh) : null;
        if (skinned) this.makeBoneMatrices(skinned);
        const skinIndex = skinned ? (geometry.getAttribute('skinIndex') as Attribute | undefined) : undefined;
        const skinWeight = skinned ? (geometry.getAttribute('skinWeight') as Attribute | undefined) : undefined;
        const bind = skinned?.bindMatrix.elements;
        const unbind = skinned?.bindMatrixInverse.elements;
        const bones = this.boneMatrices;

        for (let vertex = 0; vertex < count; vertex++) {
            const restX = position.getX(vertex);
            const restY = position.getY(vertex);
            const restZ = position.getZ(vertex);
            let x = restX;
            let y = restY;
            let z = restZ;
            if (morphs && influences) {
                for (let target = 0; target < morphs.length; target++) {
                    const influence = influences[target] ?? 0;
                    if (influence === 0) continue;
                    const morph = morphs[target];
                    x += influence * (morph.getX(vertex) - (relative ? 0 : restX));
                    y += influence * (morph.getY(vertex) - (relative ? 0 : restY));
                    z += influence * (morph.getZ(vertex) - (relative ? 0 : restZ));
                }
            }
            if (bind && unbind && skinIndex && skinWeight) {
                // Into bind space (a point: w = 1), blend the bones, and back into the mesh's space.
                const bx = bind[0] * x + bind[4] * y + bind[8] * z + bind[12];
                const by = bind[1] * x + bind[5] * y + bind[9] * z + bind[13];
                const bz = bind[2] * x + bind[6] * y + bind[10] * z + bind[14];
                let sx = 0;
                let sy = 0;
                let sz = 0;
                for (let influence = 0; influence < 4; influence++) {
                    const weight = skinWeight.getComponent(vertex, influence);
                    if (weight === 0) continue;
                    const m = skinIndex.getComponent(vertex, influence) * 16;
                    sx += weight * (bones[m] * bx + bones[m + 4] * by + bones[m + 8] * bz + bones[m + 12]);
                    sy += weight * (bones[m + 1] * bx + bones[m + 5] * by + bones[m + 9] * bz + bones[m + 13]);
                    sz += weight * (bones[m + 2] * bx + bones[m + 6] * by + bones[m + 10] * bz + bones[m + 14]);
                }
                x = unbind[0] * sx + unbind[4] * sy + unbind[8] * sz + unbind[12];
                y = unbind[1] * sx + unbind[5] * sy + unbind[9] * sz + unbind[13];
                z = unbind[2] * sx + unbind[6] * sy + unbind[10] * sz + unbind[14];
            }
            out[vertex * 3] = x;
            out[vertex * 3 + 1] = y;
            out[vertex * 3 + 2] = z;
        }
    }

    /** Each bone's matrixWorld × boneInverse, as three's skinning uses them (Skeleton.update makes the same products). */
    private makeBoneMatrices(mesh: SkinnedMesh): void {
        const skeleton = mesh.skeleton;
        const count = skeleton.bones.length;
        if (this.boneMatrices.length !== count * 16) this.boneMatrices = new Float64Array(count * 16);
        for (let bone = 0; bone < count; bone++) {
            const inverse = skeleton.boneInverses[bone];
            this.boneScratch.multiplyMatrices(skeleton.bones[bone].matrixWorld, inverse ?? IDENTITY);
            this.boneMatrices.set(this.boneScratch.elements, bone * 16);
        }
    }
}

const IDENTITY = new Matrix4();

function attributeVersion(attribute: Attribute): number {
    return (attribute as InterleavedBufferAttribute).isInterleavedBufferAttribute ? (attribute as InterleavedBufferAttribute).data.version : (attribute as BufferAttribute).version;
}

/** Whether every coordinate of `a` is within `tolerance` of `b`'s. */
function withinTolerance(a: Float32Array, b: Float32Array, tolerance: number): boolean {
    for (let index = 0; index < a.length; index++) if (Math.abs(a[index] - b[index]) > tolerance) return false;
    return true;
}
