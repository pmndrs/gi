/**
 * Node-by-node check of the tracer's top level as the GPU holds it, shared by
 * the B1a pages (tlas-oracle, tlas-bench):
 *
 * - every inner child box is exactly the union of that child's own two boxes
 *   (unions of f32 values are exact, so a correct refit matches bit for bit);
 * - every instance box contains the instance's exact world box, the box
 *   around its object box's eight transformed corners (what an instance box
 *   is, on both the GPU and the CPU path), and isn't much looser;
 * - instances that are hidden, gone or singular have the empty box no ray reaches;
 * - every record's world-to-object rows match the double-precision inverse.
 */

import { BVH_FLOAT_MAX } from '@pmndrs/bvh';
import {
    BVH_NODE_WORDS,
    EMPTY_CHILD_INDEX,
    INSTANCE_RECORD_WORDS,
    INSTANCE_WORD,
    invertAffine,
    NODE_WORD,
    TOP_HEADER_WORD,
    TOP_LEVEL_HEADER_WORDS,
} from '../../../../packages/core/src/platform/trace/index.ts';

/** An instance box may be at most this much looser than the exact box, relative to the coordinates' size. */
const BOX_SLACK = 1e-4;
/** Largest accepted error of a record's world-to-object entries against the double-precision inverse, relative. */
const INVERSE_TOLERANCE = 1e-5;
/** Failures collected before the check stops listing more. */
const MAX_FAILURES = 10;

/** One instance as the check needs it, found by its scene slot. */
export interface CheckedInstance {
    readonly id: number;
    /** Object-space box `[min, max]` of the instance's mesh. */
    readonly objectBox: readonly number[];
    readonly objectToWorld: readonly number[];
    readonly visible: boolean;
}

export async function readBuffer(device: GPUDevice, source: GPUBuffer, size: number): Promise<ArrayBuffer> {
    const readback = device.createBuffer({ label: 'top-level check readback', size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const encoder = device.createCommandEncoder({ label: 'top-level check readback' });
    encoder.copyBufferToBuffer(source, 0, readback, 0, size);
    device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    const copy = readback.getMappedRange().slice(0);
    readback.unmap();
    readback.destroy();
    return copy;
}

/** The object-space box `[min, max]` around packed positions. */
export function objectBoxOf(positions: ArrayLike<number>): number[] {
    const box = [Infinity, Infinity, Infinity, -Infinity, -Infinity, -Infinity];
    for (let index = 0; index < positions.length; index++) {
        box[index % 3] = Math.min(box[index % 3], positions[index]);
        box[(index % 3) + 3] = Math.max(box[(index % 3) + 3], positions[index]);
    }
    return box;
}

/** The box around an object box's eight corners after `matrix` (column-major), in doubles. */
export function exactWorldBox(objectBox: readonly number[], matrix: readonly number[]): number[] {
    const box = [Infinity, Infinity, Infinity, -Infinity, -Infinity, -Infinity];
    for (let corner = 0; corner < 8; corner++) {
        const [x, y, z] = [objectBox[corner & 1 ? 3 : 0], objectBox[corner & 2 ? 4 : 1], objectBox[corner & 4 ? 5 : 2]];
        for (let axis = 0; axis < 3; axis++) {
            const value = matrix[axis] * x + matrix[4 + axis] * y + matrix[8 + axis] * z + matrix[12 + axis];
            box[axis] = Math.min(box[axis], value);
            box[axis + 3] = Math.max(box[axis + 3], value);
        }
    }
    return box;
}

/** Reads `topLevel` back and checks it. Returns failures (at most a few) and the number of child slots checked. */
export async function checkTopLevel(
    device: GPUDevice,
    topLevel: GPUBuffer,
    bySlot: ReadonlyMap<number, CheckedInstance>,
): Promise<{ failures: string[]; checked: number }> {
    const bytes = await readBuffer(device, topLevel, topLevel.size);
    const words = new Uint32Array(bytes);
    const floats = new Float32Array(bytes);
    const recordCount = words[TOP_HEADER_WORD.instanceCount];
    const nodeCount = words[TOP_HEADER_WORD.nodeCount];
    const nodeBase = TOP_LEVEL_HEADER_WORDS + recordCount * INSTANCE_RECORD_WORDS;
    const failures: string[] = [];
    const floatMax = Math.fround(BVH_FLOAT_MAX);
    const slotBox = (node: number, side: number): number[] => {
        const base = nodeBase + node * BVH_NODE_WORDS + side * 8;
        return [floats[base], floats[base + 1], floats[base + 2], floats[base + 4], floats[base + 5], floats[base + 6]];
    };
    const isNeverHit = (box: number[]): boolean => box.every((value) => value === floatMax);
    let checked = 0;
    for (let node = 0; node < nodeCount && failures.length < MAX_FAILURES; node++) {
        for (let side = 0; side < 2; side++) {
            const base = nodeBase + node * BVH_NODE_WORDS;
            const index = words[base + (side === 0 ? NODE_WORD.leftIndex : NODE_WORD.rightIndex)];
            const count = words[base + (side === 0 ? NODE_WORD.leftCount : NODE_WORD.rightCount)];
            if (index === EMPTY_CHILD_INDEX) continue;
            checked++;
            const box = slotBox(node, side);
            if (count === 0) {
                // An inner child: exactly the union of its own two boxes, empty ones left out.
                const parts = [slotBox(index, 0)];
                if (words[nodeBase + index * BVH_NODE_WORDS + NODE_WORD.rightIndex] !== EMPTY_CHILD_INDEX) parts.push(slotBox(index, 1));
                const live = parts.filter((part) => !isNeverHit(part));
                const expected =
                    live.length === 0
                        ? null
                        : [0, 1, 2].map((axis) => Math.min(...live.map((part) => part[axis]))).concat([0, 1, 2].map((axis) => Math.max(...live.map((part) => part[axis + 3]))));
                const same = expected === null ? isNeverHit(box) : expected.every((value, axis) => value === box[axis]);
                if (!same) failures.push(`node ${node} side ${side}: box ${box.join(',')} is not the union of node ${index}'s boxes (${expected?.join(',') ?? 'empty'})`);
                continue;
            }
            const record = index;
            const recordBase = TOP_LEVEL_HEADER_WORDS + record * INSTANCE_RECORD_WORDS;
            const slot = words[recordBase + INSTANCE_WORD.instanceIndex];
            const instance = bySlot.get(slot);
            const inverse = instance ? invertAffine(instance.objectToWorld) : null;
            if (!instance || !instance.visible || inverse === null) {
                // Gone, hidden or singular: no ray may reach it.
                if (!isNeverHit(box)) failures.push(`record ${record} (slot ${slot}) can't be hit now but has box ${box.join(',')}`);
                continue;
            }
            const exact = exactWorldBox(instance.objectBox, instance.objectToWorld);
            for (let axis = 0; axis < 3; axis++) {
                const size = 1 + Math.max(Math.abs(exact[axis]), Math.abs(exact[axis + 3]));
                const lowGap = exact[axis] - box[axis];
                const highGap = box[axis + 3] - exact[axis + 3];
                if (lowGap < 0 || highGap < 0) failures.push(`instance ${instance.id}: box ${box.join(',')} doesn't contain the exact box ${exact.join(',')}`);
                else if (lowGap > BOX_SLACK * size || highGap > BOX_SLACK * size) failures.push(`instance ${instance.id}: box ${box.join(',')} is loose around ${exact.join(',')}`);
            }
            const rowWords = [INSTANCE_WORD.worldToObjectRow0, INSTANCE_WORD.worldToObjectRow1, INSTANCE_WORD.worldToObjectRow2];
            rowWords.forEach((rowWord, row) => {
                for (let column = 0; column < 4; column++) {
                    const expected = inverse.rows[row * 4 + column];
                    const actual = floats[recordBase + rowWord + column];
                    if (Math.abs(actual - expected) > INVERSE_TOLERANCE * (1 + Math.abs(expected))) {
                        failures.push(`instance ${instance.id}: world-to-object [${row}][${column}] ${actual}, expected ${expected}`);
                    }
                }
            });
        }
    }
    return { failures, checked };
}
