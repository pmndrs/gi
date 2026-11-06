/**
 * Every instance's transform state, by slot, in flat typed arrays: its
 * object-to-world matrix, its mesh box's world frame (for how far a move
 * takes its corners), its padded world box, its mesh's box, and whether its
 * transform mirrors or is singular.
 *
 * *Why flat arrays, not arrays on each instance:* `setTransforms` moves
 * thousands of instances a frame (T2.M1's bar: 10,000 in 2 ms of CPU), and
 * reading small arrays hung off thousands of objects cost about twice the
 * arithmetic. Indexing a few large arrays by slot keeps the loop in plain
 * numbers. They are also plain `ArrayBuffer`s that can be transferred, for
 * moving the scene copy off the main thread later (T2.13).
 *
 * Slots are the scene copy's instance slots (SlotAllocator), so they are
 * dense and reused; the arrays double when a slot past their end is used.
 */

import { WORLD_BOX_PADDING } from '../trace/top-level.ts';

/** Numbers per slot in each array. */
export const MATRIX_NUMBERS = 16;
/**
 * A mesh box's world frame: its lowest corner, then its three edges from it
 * (xyz each), then how far the world box reaches from the corner, down and
 * up (xyz each): the sums of the edges pointing down and up (FRAME_WORD).
 */
export const FRAME_NUMBERS = 18;
/** Where each part of a frame starts. */
export const FRAME_WORD = { corner: 0, edgeX: 3, edgeY: 6, edgeZ: 9, reachDown: 12, reachUp: 15 } as const;
export const BOX_NUMBERS = 6;


const FIRST_CAPACITY = 64;

export class InstanceTransforms {
    /** Object-to-world, column-major, MATRIX_NUMBERS per slot. */
    matrices = new Float64Array(FIRST_CAPACITY * MATRIX_NUMBERS);
    /** The mesh box's world frame, FRAME_NUMBERS per slot (transformFrame). */
    frames = new Float64Array(FIRST_CAPACITY * FRAME_NUMBERS);
    /** The padded world box, min then max, BOX_NUMBERS per slot. */
    bounds = new Float32Array(FIRST_CAPACITY * BOX_NUMBERS);
    /** The instance's mesh's box, min then max, BOX_NUMBERS per slot (kept here so a move reads no mesh object). */
    meshBounds = new Float32Array(FIRST_CAPACITY * BOX_NUMBERS);
    /** The record flags the transform decides (mirrored, singular), one byte per slot. */
    flags = new Uint8Array(FIRST_CAPACITY);

    get capacity(): number {
        return this.flags.length;
    }

    /** Makes room for `slot`, doubling the arrays as needed. */
    ensure(slot: number): void {
        if (slot < this.capacity) return;
        let capacity = this.capacity;
        while (capacity <= slot) capacity *= 2;
        const grown = <T extends Float64Array | Float32Array | Uint8Array>(old: T, next: T): T => {
            next.set(old as never);
            return next;
        };
        this.matrices = grown(this.matrices, new Float64Array(capacity * MATRIX_NUMBERS));
        this.frames = grown(this.frames, new Float64Array(capacity * FRAME_NUMBERS));
        this.bounds = grown(this.bounds, new Float32Array(capacity * BOX_NUMBERS));
        this.meshBounds = grown(this.meshBounds, new Float32Array(capacity * BOX_NUMBERS));
        this.flags = grown(this.flags, new Uint8Array(capacity));
    }

    /** A slot's matrix as a view (valid until the arrays next grow; copy it to keep it). */
    matrix(slot: number): Float64Array {
        return this.matrices.subarray(slot * MATRIX_NUMBERS, slot * MATRIX_NUMBERS + MATRIX_NUMBERS);
    }

    /** A slot's world box as a view (valid until the arrays next grow). */
    box(slot: number): Float32Array {
        return this.bounds.subarray(slot * BOX_NUMBERS, slot * BOX_NUMBERS + BOX_NUMBERS);
    }

    setMeshBounds(slot: number, meshBounds: ArrayLike<number>): void {
        for (let index = 0; index < BOX_NUMBERS; index++) this.meshBounds[slot * BOX_NUMBERS + index] = meshBounds[index];
    }

    /**
     * Copies a matrix (16 numbers from `offset` in `source`) into the slot.
     * Returns whether its linear part (rotation, scale: the first 12 numbers)
     * changed, so a move that only translates can keep its normal matrix.
     */
    setMatrix(slot: number, source: ArrayLike<number>, offset: number): boolean {
        const matrices = this.matrices;
        const base = slot * MATRIX_NUMBERS;
        let linearChanged = false;
        for (let index = 0; index < 12; index++) {
            const value = source[offset + index];
            if (value !== matrices[base + index]) linearChanged = true;
            matrices[base + index] = value;
        }
        for (let index = 12; index < MATRIX_NUMBERS; index++) matrices[base + index] = source[offset + index];
        return linearChanged;
    }

    /**
     * Works the slot's world frame and world box out from its matrix and mesh
     * box, and returns how far its mesh box's corners moved from the frame it
     * held: the largest distance any of the eight travelled
     * (`DirtyRegion.moved`; meaningless the first time). Corners rather than
     * the world box, so a turn in place that leaves the box as it was still
     * counts (transformFrame says how).
     */
    place(slot: number): number {
        return transformFrame(this.meshBounds, slot * BOX_NUMBERS, this.matrices, slot * MATRIX_NUMBERS, this.frames, slot * FRAME_NUMBERS, this.bounds, slot * BOX_NUMBERS);
    }

    /**
     * `place` for a move that only translated (`setMatrix` said the linear part
     * stayed): the frame's edges are as they were, so every corner moved by
     * the same amount as the lowest one, and the box moves with it. The same
     * numbers as `place` up to rounding, with a fraction of the work; most
     * drags and many animations are slides.
     */
    slide(slot: number): number {
        const m = slot * MATRIX_NUMBERS;
        const f = slot * FRAME_NUMBERS;
        const b = slot * BOX_NUMBERS;
        const e = this.matrices;
        const mesh = this.meshBounds;
        const frames = this.frames;
        const x0 = mesh[b], y0 = mesh[b + 1], z0 = mesh[b + 2];
        const cornerX = e[m] * x0 + e[m + 4] * y0 + e[m + 8] * z0 + e[m + 12];
        const cornerY = e[m + 1] * x0 + e[m + 5] * y0 + e[m + 9] * z0 + e[m + 13];
        const cornerZ = e[m + 2] * x0 + e[m + 6] * y0 + e[m + 10] * z0 + e[m + 14];
        const moveX = cornerX - frames[f];
        const moveY = cornerY - frames[f + 1];
        const moveZ = cornerZ - frames[f + 2];
        frames[f] = cornerX;
        frames[f + 1] = cornerY;
        frames[f + 2] = cornerZ;
        writeSlidBox(cornerX, cornerY, cornerZ, frames, f, this.bounds, b);
        return Math.sqrt(moveX * moveX + moveY * moveY + moveZ * moveZ);
    }
}

/**
 * Writes an instance's world frame (FRAME_NUMBERS from `frameAt`) and its
 * padded world box (BOX_NUMBERS from `boundsAt`), from its matrix and its
 * mesh's box, and returns how far the mesh box's corners moved from the
 * frame that was there: the largest distance any of the eight travelled.
 *
 * *Why a frame and not the eight corners:* the box around the corners is
 * the frame's corner plus, per axis, the edges that point down (min) or up
 * (max), and a corner's move is the corner's move plus the edges' moves, so
 * nothing is worked out eight times; the eight-corner version took about
 * twice as long. The box equals `trace/top-level.ts`' `transformBounds` up
 * to rounding, far inside the padding both add.
 */
export function transformFrame(
    meshBounds: ArrayLike<number>,
    meshAt: number,
    matrices: ArrayLike<number>,
    matrixAt: number,
    frames: Float64Array,
    frameAt: number,
    bounds: Float32Array,
    boundsAt: number,
): number {
    const e = matrices;
    const m = matrixAt;
    const x0 = meshBounds[meshAt];
    const y0 = meshBounds[meshAt + 1];
    const z0 = meshBounds[meshAt + 2];
    const sizeX = meshBounds[meshAt + 3] - x0;
    const sizeY = meshBounds[meshAt + 4] - y0;
    const sizeZ = meshBounds[meshAt + 5] - z0;
    // The lowest corner, and the three edges: each column of the matrix times the box's size along it.
    const cornerX = e[m] * x0 + e[m + 4] * y0 + e[m + 8] * z0 + e[m + 12];
    const cornerY = e[m + 1] * x0 + e[m + 5] * y0 + e[m + 9] * z0 + e[m + 13];
    const cornerZ = e[m + 2] * x0 + e[m + 6] * y0 + e[m + 10] * z0 + e[m + 14];
    const edgeXx = e[m] * sizeX, edgeXy = e[m + 1] * sizeX, edgeXz = e[m + 2] * sizeX;
    const edgeYx = e[m + 4] * sizeY, edgeYy = e[m + 5] * sizeY, edgeYz = e[m + 6] * sizeY;
    const edgeZx = e[m + 8] * sizeZ, edgeZy = e[m + 9] * sizeZ, edgeZz = e[m + 10] * sizeZ;

    // How far each part moved since the frame held.
    const f = frameAt;
    const moveCornerX = cornerX - frames[f], moveCornerY = cornerY - frames[f + 1], moveCornerZ = cornerZ - frames[f + 2];
    const moveXx = edgeXx - frames[f + 3], moveXy = edgeXy - frames[f + 4], moveXz = edgeXz - frames[f + 5];
    const moveYx = edgeYx - frames[f + 6], moveYy = edgeYy - frames[f + 7], moveYz = edgeYz - frames[f + 8];
    const moveZx = edgeZx - frames[f + 9], moveZy = edgeZy - frames[f + 10], moveZz = edgeZz - frames[f + 11];
    // Each corner's move is the lowest corner's move plus the moves of the edges that lead to it,
    // built up corner from corner: the x edge for bit 0 of its index, y for bit 1, z for bit 2.
    const move1x = moveCornerX + moveXx, move1y = moveCornerY + moveXy, move1z = moveCornerZ + moveXz;
    const move2x = moveCornerX + moveYx, move2y = moveCornerY + moveYy, move2z = moveCornerZ + moveYz;
    const move3x = move1x + moveYx, move3y = move1y + moveYy, move3z = move1z + moveYz;
    const largestSquared = Math.max(
        squaredLength(moveCornerX, moveCornerY, moveCornerZ),
        squaredLength(move1x, move1y, move1z),
        squaredLength(move2x, move2y, move2z),
        squaredLength(move3x, move3y, move3z),
        squaredLength(moveCornerX + moveZx, moveCornerY + moveZy, moveCornerZ + moveZz),
        squaredLength(move1x + moveZx, move1y + moveZy, move1z + moveZz),
        squaredLength(move2x + moveZx, move2y + moveZy, move2z + moveZz),
        squaredLength(move3x + moveZx, move3y + moveZy, move3z + moveZz),
    );

    frames[f] = cornerX;
    frames[f + 1] = cornerY;
    frames[f + 2] = cornerZ;
    frames[f + 3] = edgeXx;
    frames[f + 4] = edgeXy;
    frames[f + 5] = edgeXz;
    frames[f + 6] = edgeYx;
    frames[f + 7] = edgeYy;
    frames[f + 8] = edgeYz;
    frames[f + 9] = edgeZx;
    frames[f + 10] = edgeZy;
    frames[f + 11] = edgeZz;

    writeBox(cornerX, cornerY, cornerZ, frames, f, bounds, boundsAt);
    return Math.sqrt(largestSquared);
}

/**
 * The padded world box of a frame whose lowest corner is (cornerX, cornerY,
 * cornerZ) and whose edges are at `frames[frameAt + 3 ...]`: per axis, the
 * edges pointing down make the box's min, those pointing up its max.
 */
function writeBox(cornerX: number, cornerY: number, cornerZ: number, frames: Float64Array, frameAt: number, bounds: Float32Array, boundsAt: number): void {
    const f = frameAt;
    // Per axis, the edges pointing down make the box's min, those pointing up its max.
    const downX = Math.min(frames[f + 3], 0) + Math.min(frames[f + 6], 0) + Math.min(frames[f + 9], 0);
    const downY = Math.min(frames[f + 4], 0) + Math.min(frames[f + 7], 0) + Math.min(frames[f + 10], 0);
    const downZ = Math.min(frames[f + 5], 0) + Math.min(frames[f + 8], 0) + Math.min(frames[f + 11], 0);
    const upX = Math.max(frames[f + 3], 0) + Math.max(frames[f + 6], 0) + Math.max(frames[f + 9], 0);
    const upY = Math.max(frames[f + 4], 0) + Math.max(frames[f + 7], 0) + Math.max(frames[f + 10], 0);
    const upZ = Math.max(frames[f + 5], 0) + Math.max(frames[f + 8], 0) + Math.max(frames[f + 11], 0);
    frames[f + FRAME_WORD.reachDown] = downX;
    frames[f + FRAME_WORD.reachDown + 1] = downY;
    frames[f + FRAME_WORD.reachDown + 2] = downZ;
    frames[f + FRAME_WORD.reachUp] = upX;
    frames[f + FRAME_WORD.reachUp + 1] = upY;
    frames[f + FRAME_WORD.reachUp + 2] = upZ;
    writePaddedBox(cornerX + downX, cornerY + downY, cornerZ + downZ, cornerX + upX, cornerY + upY, cornerZ + upZ, bounds, boundsAt);
}

/**
 * The box of a frame that only moved (a slide): its reach from the corner
 * is as `writeBox` last stored it.
 */
function writeSlidBox(cornerX: number, cornerY: number, cornerZ: number, frames: Float64Array, frameAt: number, bounds: Float32Array, boundsAt: number): void {
    const down = frameAt + FRAME_WORD.reachDown;
    const up = frameAt + FRAME_WORD.reachUp;
    writePaddedBox(cornerX + frames[down], cornerY + frames[down + 1], cornerZ + frames[down + 2], cornerX + frames[up], cornerY + frames[up + 1], cornerZ + frames[up + 2], bounds, boundsAt);
}

/** A box grown by the tracer's padding (trace/top-level.ts, WORLD_BOX_PADDING), so rounding can't make a ray miss it. */
function writePaddedBox(lowX: number, lowY: number, lowZ: number, highX: number, highY: number, highZ: number, bounds: Float32Array, boundsAt: number): void {
    const padX = (Math.max(Math.abs(lowX), Math.abs(highX)) + (highX - lowX)) * WORLD_BOX_PADDING;
    const padY = (Math.max(Math.abs(lowY), Math.abs(highY)) + (highY - lowY)) * WORLD_BOX_PADDING;
    const padZ = (Math.max(Math.abs(lowZ), Math.abs(highZ)) + (highZ - lowZ)) * WORLD_BOX_PADDING;
    const b = boundsAt;
    bounds[b] = lowX - padX;
    bounds[b + 1] = lowY - padY;
    bounds[b + 2] = lowZ - padZ;
    bounds[b + 3] = highX + padX;
    bounds[b + 4] = highY + padY;
    bounds[b + 5] = highZ + padZ;
}

function squaredLength(x: number, y: number, z: number): number {
    return x * x + y * y + z * z;
}
