import { describe, expect, it } from 'vitest';
import { EMPTY_CHILD_INDEX, NODE_WORD, nodeView } from './layout.ts';
import { createRandom } from './oracle-fixtures.ts';
import { buildSahTree, DEFAULT_SAH_OPTIONS, type SahBuildOptions } from './sah-builder.ts';

/**
 * A slow, plain binned-SAH builder written straight from the description in
 * sah-builder.ts: recursive, every bin swept, boxes recomputed from scratch.
 * The real builder skips work (empty bins, recomputed boxes, per-node
 * allocation) for speed; this checks that none of that changes the tree.
 *
 * It returns the tree as nested text, one line per child: the box and either
 * the leaf's primitives (as input indices, in stored order) or the subtree.
 */
function referenceTree(bounds: Float32Array, options: SahBuildOptions): string {
    const count = bounds.length / 6;
    const centroid = (primitive: number, axis: number): number => Math.fround(0.5 * (bounds[primitive * 6 + axis] + bounds[primitive * 6 + 3 + axis]));
    const boxOf = (primitives: number[]): number[] => {
        const box = [Infinity, Infinity, Infinity, -Infinity, -Infinity, -Infinity];
        for (const primitive of primitives) {
            for (let axis = 0; axis < 3; axis++) {
                box[axis] = Math.min(box[axis], bounds[primitive * 6 + axis]);
                box[axis + 3] = Math.max(box[axis + 3], bounds[primitive * 6 + 3 + axis]);
            }
        }
        return box;
    };
    const area = (box: number[]): number => {
        const [dx, dy, dz] = [box[3] - box[0], box[4] - box[1], box[5] - box[2]];
        return dx < 0 || dy < 0 || dz < 0 ? 0 : 2 * (dx * dy + dy * dz + dz * dx);
    };

    // The builder's in-place partition (swap from both ends), so leaves list primitives in the same order.
    const partition = (primitives: number[], goesLeft: (primitive: number) => boolean): [number[], number[]] => {
        const list = [...primitives];
        let left = 0;
        let right = list.length - 1;
        while (left <= right) {
            if (goesLeft(list[left])) {
                left++;
            } else {
                [list[left], list[right]] = [list[right], list[left]];
                right--;
            }
        }
        return [list.slice(0, left), list.slice(left)];
    };

    type Split = { cost: number; left: number[]; right: number[] } | null;
    const bestSplit = (primitives: number[]): Split => {
        let best: Split = null;
        const parentArea = area(boxOf(primitives));
        for (let axis = 0; axis < 3; axis++) {
            const values = primitives.map((primitive) => centroid(primitive, axis));
            const low = Math.min(...values);
            const extent = Math.max(...values) - low;
            if (!(extent > 0)) continue;
            const scale = options.binCount / extent;
            const binOf = (primitive: number) => Math.min(options.binCount - 1, Math.max(0, Math.floor((centroid(primitive, axis) - low) * scale)));
            for (let boundary = 1; boundary < options.binCount; boundary++) {
                const [left, right] = partition(primitives, (primitive) => binOf(primitive) < boundary);
                if (left.length === 0 || right.length === 0) continue;
                const tests =
                    parentArea > 0
                        ? (area(boxOf(left)) * left.length + area(boxOf(right)) * right.length) / parentArea
                        : (left.length ** 2 + right.length ** 2) / primitives.length;
                const cost = options.traversalCost + options.intersectionCost * tests;
                if (best === null || cost < best.cost) best = { cost, left, right };
            }
        }
        return best;
    };

    const describeChild = (primitives: number[], isRoot: boolean, indent: string): string => {
        const box = boxOf(primitives).join(' ');
        const split = primitives.length === 1 ? null : bestSplit(primitives);
        const isLeaf =
            !isRoot &&
            (primitives.length === 1 ||
                (primitives.length <= options.maxLeafPrimitives && (split === null || options.intersectionCost * primitives.length <= split.cost)));
        if (isLeaf) return `${indent}leaf [${box}] ${primitives.join(',')}\n`;
        // No separating plane: the builder halves the list in its current order.
        const middle = primitives.length >>> 1;
        const left = split ? split.left : primitives.slice(0, middle);
        const right = split ? split.right : primitives.slice(middle);
        const header = isRoot ? '' : `${indent}node [${box}]\n`;
        return header + describeChild(left, false, `${indent}  `) + describeChild(right, false, `${indent}  `);
    };
    return describeChild(
        Array.from({ length: count }, (_, primitive) => primitive),
        true,
        '',
    );
}

/** The real builder's tree in the same text form. */
function builtTree(bounds: Float32Array, options: SahBuildOptions): string {
    const tree = buildSahTree(bounds, options);
    const { floats, words } = nodeView(tree.nodes);
    const describeNode = (node: number, indent: string): string => {
        let text = '';
        for (const side of ['left', 'right'] as const) {
            const base = node * 16;
            const index = words[base + (side === 'left' ? NODE_WORD.leftIndex : NODE_WORD.rightIndex)];
            if (index === EMPTY_CHILD_INDEX) continue;
            const count = words[base + (side === 'left' ? NODE_WORD.leftCount : NODE_WORD.rightCount)];
            const minWord = base + (side === 'left' ? NODE_WORD.leftMin : NODE_WORD.rightMin);
            const maxWord = base + (side === 'left' ? NODE_WORD.leftMax : NODE_WORD.rightMax);
            const box = [...floats.subarray(minWord, minWord + 3), ...floats.subarray(maxWord, maxWord + 3)].join(' ');
            if (count > 0) {
                text += `${indent}leaf [${box}] ${Array.from(tree.order.subarray(index, index + count)).join(',')}\n`;
            } else {
                text += `${indent}node [${box}]\n${describeNode(index, `${indent}  `)}`;
            }
        }
        return text;
    };
    return describeNode(0, '  ');
}

/**
 * Random boxes, quantized to a coarse grid so that many centroids coincide and
 * many bins stay empty: the cases where skipping empty bins could go wrong.
 */
function randomBoxes(seed: number, count: number, grid: number): Float32Array {
    const random = createRandom(seed);
    const bounds = new Float32Array(count * 6);
    for (let primitive = 0; primitive < count; primitive++) {
        for (let axis = 0; axis < 3; axis++) {
            const low = Math.floor(random() * grid) / 4;
            const size = Math.floor(random() * 3) / 4;
            bounds[primitive * 6 + axis] = low;
            bounds[primitive * 6 + 3 + axis] = low + size;
        }
    }
    return bounds;
}

describe('buildSahTree', () => {
    it('builds the same tree as a plain binned-SAH builder that sweeps every bin', () => {
        for (const [seed, count, grid] of [
            [1, 2, 4],
            [2, 7, 2],
            [3, 40, 6],
            [4, 300, 40],
            [5, 600, 8],
            [6, 64, 1],
        ]) {
            for (const maxLeafPrimitives of [1, 4]) {
                const options = { ...DEFAULT_SAH_OPTIONS, maxLeafPrimitives };
                const bounds = randomBoxes(seed, count, grid);
                expect(builtTree(bounds, options), `seed ${seed}, ${count} boxes, leaves of ${maxLeafPrimitives}`).toBe(
                    referenceTree(bounds, options),
                );
            }
        }
    });
});
