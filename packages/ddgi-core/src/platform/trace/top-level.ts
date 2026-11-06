/**
 * The top level of the two-level BVH: one record per instance, and a small
 * tree over the instances' world-space boxes.
 *
 * This file builds it whole on the CPU. On a device with subgroups that is
 * only the tests' and benchmarks' path: the tracer keeps the top level on the
 * GPU instead (top-level-gpu.ts, B1a). Without subgroups it is the declared
 * fallback, rebuilt whenever a transform changes. Moving an object never
 * touches its mesh tree either way: the GPU moves the ray into the object's
 * space instead (see the WGSL).
 *
 * Buffer layout: `[64-byte header][instance records][top-level nodes]`, all
 * in tree order (records are sorted so a leaf's `index` is a record index).
 */

import type { Mat4 } from '../contract/math.ts';
import {
    BVH_NODE_BYTES,
    INSTANCE_FLAG,
    INSTANCE_RECORD_BYTES,
    INSTANCE_RECORD_WORDS,
    INSTANCE_WORD,
    TOP_HEADER_WORD,
    TOP_LEVEL_HEADER_BYTES,
} from './layout.ts';
import { buildSahTree, DEFAULT_SAH_OPTIONS, type SahBuildOptions } from './sah-builder.ts';

/** One placed copy of a mesh, as the top-level builder sees it. */
export interface TraceInstance {
    /** The platform instance index. Comes back as `GiHit.instance`. */
    readonly instanceIndex: number;
    /** Where this mesh's nodes start in the shared node buffer, counted in nodes. */
    readonly nodeOffset: number;
    /** Where this mesh's triangles start in the shared triangle buffer, counted in triangles. */
    readonly triangleOffset: number;
    /** The mesh's object-space box: `[minX, minY, minZ, maxX, maxY, maxZ]`. */
    readonly meshBounds: ArrayLike<number>;
    /** Object-to-world, column-major 4x4 (affine; mirrors and non-uniform scales allowed). */
    readonly objectToWorld: Mat4;
    /** Defaults to true. An instance can be drawn but hidden from GI, or the reverse. */
    readonly contributesToGi?: boolean;
    /** Defaults to true. */
    readonly visible?: boolean;
}

export interface TopLevelBvh {
    /** Header, records and nodes in one buffer, ready for `writeBuffer`. */
    readonly buffer: ArrayBuffer;
    /** Instances that made it into the tree (hidden and non-invertible ones do not). */
    readonly instanceCount: number;
    readonly nodeCount: number;
    /** Levels of nodes on the longest top-level path. */
    readonly maxDepth: number;
    /** Instances left out because their transform cannot be inverted. */
    readonly skippedSingular: number;
    /** `traceableIndex` order: `recordOrder[recordIndex]` is the position in the input array. */
    readonly recordOrder: Uint32Array;
}

/**
 * Relative amount each world box is grown by, so rounding in the ray's move
 * into object space cannot make a ray miss the box of an instance it touches.
 * About four f32 steps.
 */
const WORLD_BOX_PADDING = 2 ** -21;

/** Smallest determinant magnitude we still treat as an invertible transform. */
const MIN_DETERMINANT = 1e-30;

/**
 * Inverts an affine column-major 4x4 and returns the top three rows of the
 * result (12 numbers, row-major) plus the sign of the determinant.
 * Returns null if the matrix is singular or has non-finite entries.
 */
export function invertAffine(objectToWorld: Mat4): { rows: Float64Array; determinant: number } | null {
    const rows = new Float64Array(12);
    const determinant = invertAffineInto(objectToWorld, rows);
    return determinant === 0 ? null : { rows, determinant };
}

/**
 * {@link invertAffine} into `out` (12 numbers), without allocating: for the
 * scene copy's per-instance writes, which `setTransforms` makes thousands
 * of a frame. Returns the determinant, or 0 when the matrix is singular or
 * the result isn't finite (then `out` holds garbage).
 */
export function invertAffineInto(objectToWorld: Mat4, out: Float64Array): number {
    const e = objectToWorld;
    if (e.length < 16) throw new RangeError('objectToWorld must have 16 entries');
    // The linear part L, with L[row][column] = e[column * 4 + row].
    const m00 = e[0], m01 = e[4], m02 = e[8];
    const m10 = e[1], m11 = e[5], m12 = e[9];
    const m20 = e[2], m21 = e[6], m22 = e[10];
    const tx = e[12], ty = e[13], tz = e[14];

    // Cofactors of the first column and the determinant.
    const c00 = m11 * m22 - m12 * m21;
    const c10 = m12 * m20 - m10 * m22;
    const c20 = m10 * m21 - m11 * m20;
    const determinant = m00 * c00 + m01 * c10 + m02 * c20;
    if (!Number.isFinite(determinant) || !(Math.abs(determinant) > MIN_DETERMINANT)) return 0;
    const inverseDeterminant = 1 / determinant;

    // inverse(L) = adjugate(L) / det, written out by rows.
    const i00 = c00 * inverseDeterminant;
    const i01 = (m02 * m21 - m01 * m22) * inverseDeterminant;
    const i02 = (m01 * m12 - m02 * m11) * inverseDeterminant;
    const i10 = c10 * inverseDeterminant;
    const i11 = (m00 * m22 - m02 * m20) * inverseDeterminant;
    const i12 = (m02 * m10 - m00 * m12) * inverseDeterminant;
    const i20 = c20 * inverseDeterminant;
    const i21 = (m01 * m20 - m00 * m21) * inverseDeterminant;
    const i22 = (m00 * m11 - m01 * m10) * inverseDeterminant;

    // p_object = inverse(L) * (p_world - t) = inverse(L) * p_world - inverse(L) * t
    out[0] = i00;
    out[1] = i01;
    out[2] = i02;
    out[3] = -(i00 * tx + i01 * ty + i02 * tz);
    out[4] = i10;
    out[5] = i11;
    out[6] = i12;
    out[7] = -(i10 * tx + i11 * ty + i12 * tz);
    out[8] = i20;
    out[9] = i21;
    out[10] = i22;
    out[11] = -(i20 * tx + i21 * ty + i22 * tz);
    for (let index = 0; index < 12; index++) if (!Number.isFinite(out[index])) return 0;
    return determinant;
}

/** transformBounds' running box: one scratch, so a call allocates nothing (JS is single-threaded, nothing re-enters it). */
const boundsLow = new Float64Array(3);
const boundsHigh = new Float64Array(3);

/** Axis-aligned box around an object-space box after `objectToWorld`, grown by {@link WORLD_BOX_PADDING}. */
export function transformBounds(meshBounds: ArrayLike<number>, objectToWorld: Mat4, out: Float32Array, offset: number): void {
    const e = objectToWorld;
    const low = boundsLow.fill(Infinity);
    const high = boundsHigh.fill(-Infinity);
    for (let corner = 0; corner < 8; corner++) {
        const x = corner & 1 ? meshBounds[3] : meshBounds[0];
        const y = corner & 2 ? meshBounds[4] : meshBounds[1];
        const z = corner & 4 ? meshBounds[5] : meshBounds[2];
        for (let axis = 0; axis < 3; axis++) {
            const value = e[axis] * x + e[4 + axis] * y + e[8 + axis] * z + e[12 + axis];
            if (value < low[axis]) low[axis] = value;
            if (value > high[axis]) high[axis] = value;
        }
    }
    for (let axis = 0; axis < 3; axis++) {
        const pad = (Math.max(Math.abs(low[axis]), Math.abs(high[axis])) + (high[axis] - low[axis])) * WORLD_BOX_PADDING;
        out[offset + axis] = low[axis] - pad;
        out[offset + 3 + axis] = high[axis] + pad;
    }
}

export function buildTopLevel(instances: readonly TraceInstance[], options: Partial<SahBuildOptions> = {}): TopLevelBvh {
    // Keep only instances that can be hit.
    const kept: { input: number; inverse: Float64Array; mirrored: boolean }[] = [];
    let skippedSingular = 0;
    instances.forEach((instance, input) => {
        if (instance.visible === false || instance.contributesToGi === false) return;
        const inverse = invertAffine(instance.objectToWorld);
        if (inverse === null) {
            skippedSingular++;
            return;
        }
        kept.push({ input, inverse: inverse.rows, mirrored: inverse.determinant < 0 });
    });

    const boxes = new Float32Array(kept.length * 6);
    kept.forEach((entry, slot) => {
        const instance = instances[entry.input];
        transformBounds(instance.meshBounds, instance.objectToWorld, boxes, slot * 6);
    });

    // One instance per leaf: the instance's own tree is where the real work is.
    const tree = buildSahTree(boxes, { ...DEFAULT_SAH_OPTIONS, ...options, maxLeafPrimitives: 1 });

    const recordBytes = kept.length * INSTANCE_RECORD_BYTES;
    const nodeBytes = tree.nodeCount * BVH_NODE_BYTES;
    const buffer = new ArrayBuffer(TOP_LEVEL_HEADER_BYTES + recordBytes + nodeBytes);
    const words = new Uint32Array(buffer);
    const floats = new Float32Array(buffer);

    words[TOP_HEADER_WORD.instanceCount] = kept.length;
    words[TOP_HEADER_WORD.nodeCount] = tree.nodeCount;
    words[TOP_HEADER_WORD.maxDepth] = tree.maxDepth;

    const recordOrder = new Uint32Array(kept.length);
    for (let recordIndex = 0; recordIndex < kept.length; recordIndex++) {
        const entry = kept[tree.order[recordIndex]];
        const instance = instances[entry.input];
        recordOrder[recordIndex] = entry.input;

        const base = TOP_LEVEL_HEADER_BYTES / 4 + recordIndex * INSTANCE_RECORD_WORDS;
        floats.set(entry.inverse.subarray(0, 4), base + INSTANCE_WORD.worldToObjectRow0);
        floats.set(entry.inverse.subarray(4, 8), base + INSTANCE_WORD.worldToObjectRow1);
        floats.set(entry.inverse.subarray(8, 12), base + INSTANCE_WORD.worldToObjectRow2);
        words[base + INSTANCE_WORD.rootNodeOffset] = instance.nodeOffset;
        words[base + INSTANCE_WORD.triangleOffset] = instance.triangleOffset;
        words[base + INSTANCE_WORD.instanceIndex] = instance.instanceIndex;
        words[base + INSTANCE_WORD.flags] =
            INSTANCE_FLAG.contributesToGi | INSTANCE_FLAG.visible | (entry.mirrored ? INSTANCE_FLAG.mirrored : 0);
    }

    new Uint8Array(buffer, TOP_LEVEL_HEADER_BYTES + recordBytes, nodeBytes).set(new Uint8Array(tree.nodes));

    return {
        buffer,
        instanceCount: kept.length,
        nodeCount: tree.nodeCount,
        maxDepth: tree.maxDepth,
        skippedSingular,
        recordOrder,
    };
}
