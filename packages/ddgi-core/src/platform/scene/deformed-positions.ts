/**
 * What the scene keeps for a mesh created `deforming: true`, and the CPU
 * arithmetic for new positions from `setDeformedPositions`.
 *
 * **How a pose reaches rays (T3.3).** The scene writes the new positions
 * into the mesh's own range of the vertex buffer (the scene copy's positions
 * are plain xyz floats, 3 words a vertex, so no layout changes): with
 * `queue.writeBuffer` for the CPU form, with one GPU copy for the GPU forms
 * (gpu-pose-copy.ts). It reports the mesh in `SceneUpload.meshesPosed`, and
 * the tracer refits the mesh's tree from the scene copy's positions on the
 * GPU in the same frame's commands (trace/deforming-trees.ts). The shape of
 * the tree stays; only a drift held for about a second rebuilds it
 * (DECISIONS 2026-11-09).
 */

import type { Aabb } from '../contract/math.ts';
import type { GpuPose } from './gpu-pose-copy.ts';
import { BOX_NUMBERS, MATRIX_NUMBERS } from './instance-transforms.ts';

export class DeformState {
    /** The host's `deformBounds`, min then max in mesh space, or null when it gave none. */
    readonly declaredBounds: Float32Array | null;
    /**
     * The positions the scene copy holds now, when the CPU knows them: the
     * last CPU pose applied (the rest pose at first). After a GPU pose it is
     * out of date (`currentKnown` false) until the next CPU pose.
     */
    current: Float32Array<ArrayBuffer>;
    currentKnown = true;
    /** The newest CPU positions given and not applied yet. Its array is reused from pose to pose (swapped with `current`). */
    waiting: Float32Array<ArrayBuffer>;
    /** The newest GPU pose given and not applied yet; it replaces a waiting CPU pose and the other way round. */
    waitingOnGpu: GpuPose | null = null;
    hasWaiting = false;
    /** Positions outside `declaredBounds` were already reported once. */
    warnedOutside = false;

    constructor(restPositions: Float32Array, declaredBounds: Float32Array | null) {
        this.current = Float32Array.from(restPositions);
        this.waiting = new Float32Array(restPositions.length);
        this.declaredBounds = declaredBounds;
    }

    get vertexCount(): number {
        return this.current.length / 3;
    }

    /** The waiting CPU pose becomes the current one; the old current array is reused for the next pose. */
    swap(): void {
        const previous = this.current;
        this.current = this.waiting;
        this.waiting = previous;
        this.currentKnown = true;
        this.hasWaiting = false;
    }

    /** The waiting GPU pose is applied: the CPU no longer knows the positions. */
    takeGpuPose(): GpuPose | null {
        const pose = this.waitingOnGpu;
        this.waitingOnGpu = null;
        this.currentKnown = false;
        this.hasWaiting = false;
        return pose;
    }
}

/** The diagonal of a box (min then max): the farthest any point inside it can move and stay inside. */
export function boxDiagonal(box: ArrayLike<number>): number {
    return Math.hypot(box[3] - box[0], box[4] - box[1], box[5] - box[2]);
}

/** `deformBounds` as `[minX, minY, minZ, maxX, maxY, maxZ]`, checked: finite, and min ≤ max on every axis. */
export function checkedDeformBounds(bounds: Aabb, what: string): Float32Array {
    const flat = new Float32Array([...bounds.min, ...bounds.max]);
    if (flat.length !== BOX_NUMBERS || !flat.every((value) => Number.isFinite(value))) {
        throw new RangeError(`@pmndrs/ddgi: ${what}: deformBounds must be a box of finite numbers`);
    }
    for (let axis = 0; axis < 3; axis++) {
        if (flat[axis] > flat[axis + 3]) throw new RangeError(`@pmndrs/ddgi: ${what}: deformBounds has min above max on axis ${axis}`);
    }
    return flat;
}

/** Copies a pose into `target`, throwing (before anything is kept) if a number isn't finite. */
export function copyCheckedPositions(source: Float32Array, target: Float32Array, what: string): void {
    // Any non-finite number makes the sum of number × 0 NaN: one pass with no branch per number.
    let check = 0;
    for (let index = 0; index < source.length; index++) check += source[index] * 0;
    if (check !== 0) {
        const bad = source.findIndex((value) => !Number.isFinite(value));
        throw new RangeError(`@pmndrs/ddgi: ${what}: vertex ${Math.floor(bad / 3)} has a non-finite position`);
    }
    target.set(source);
}

/**
 * One pass over a new pose against the pose before: writes the new pose's
 * box (min then max) into `boundsOut`, and returns the largest distance any
 * vertex travelled, in mesh space. That distance is what `DirtyRegion.moved`
 * reports for a deforming mesh, so a technique can size the change the way
 * it sizes a rigid move (docs/design/m3-moving.md §5.1).
 */
export function scanPose(next: Float32Array, previous: Float32Array, boundsOut: Float32Array): number {
    let minX = Infinity;
    let minY = Infinity;
    let minZ = Infinity;
    let maxX = -Infinity;
    let maxY = -Infinity;
    let maxZ = -Infinity;
    let largestSquared = 0;
    for (let at = 0; at < next.length; at += 3) {
        const x = next[at];
        const y = next[at + 1];
        const z = next[at + 2];
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
        if (z < minZ) minZ = z;
        if (z > maxZ) maxZ = z;
        const dx = x - previous[at];
        const dy = y - previous[at + 1];
        const dz = z - previous[at + 2];
        const squared = dx * dx + dy * dy + dz * dz;
        if (squared > largestSquared) largestSquared = squared;
    }
    boundsOut[0] = minX;
    boundsOut[1] = minY;
    boundsOut[2] = minZ;
    boundsOut[3] = maxX;
    boundsOut[4] = maxY;
    boundsOut[5] = maxZ;
    return Math.sqrt(largestSquared);
}

/**
 * Grows `box` (min then max) to hold `other`. Returns whether it grew.
 * The box only ever grows: a smaller pose keeps the larger box, so poses
 * that come and go (a sway) never change the instances' boxes.
 */
export function growBox(box: Float32Array, other: ArrayLike<number>): boolean {
    let grew = false;
    for (let axis = 0; axis < 3; axis++) {
        if (other[axis] < box[axis]) {
            box[axis] = other[axis];
            grew = true;
        }
        if (other[axis + 3] > box[axis + 3]) {
            box[axis + 3] = other[axis + 3];
            grew = true;
        }
    }
    return grew;
}

/** Whether `inner` (min then max) lies inside `outer`. */
export function boxInside(inner: ArrayLike<number>, outer: ArrayLike<number>): boolean {
    for (let axis = 0; axis < 3; axis++) {
        if (inner[axis] < outer[axis] || inner[axis + 3] > outer[axis + 3]) return false;
    }
    return true;
}

/**
 * How much a slot's transform stretches lengths, at most: its longest column
 * (the x, y or z axis's world length). A vertex's travel in mesh space times
 * this bounds its travel in world space for rotations and scales along the
 * mesh's axes (exact for uniform scales); a sheared transform can stretch a
 * little more, which only makes the reported move slightly small.
 */
export function largestScale(matrices: Float64Array, slot: number): number {
    const m = slot * MATRIX_NUMBERS;
    let largest = 0;
    for (let column = 0; column < 3; column++) {
        const x = matrices[m + column * 4];
        const y = matrices[m + column * 4 + 1];
        const z = matrices[m + column * 4 + 2];
        largest = Math.max(largest, x * x + y * y + z * z);
    }
    return Math.sqrt(largest);
}
