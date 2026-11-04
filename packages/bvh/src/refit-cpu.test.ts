import { describe, expect, it } from 'vitest';
import { GPU_LAYOUT_EMPTY_CHILD, GPU_LAYOUT_NO_PARENT, GPU_LAYOUT_NODE_WORD, GPU_LAYOUT_NODE_WORDS } from './gpu-layout.ts';
import { bvhRefitWgsl } from './gpu-refit.ts';
import { BVH_FLOAT_MAX, halfArea, leafSlotsOf, parentSlotsOf, PRIMITIVE_BOX_FLOATS, refitOnCpu, relativeSahCost } from './refit-cpu.ts';

/**
 * A three-primitive tree in the GPU layout:
 *
 *     node 0: left = node 1 (inner), right = leaf of primitive 2
 *     node 1: left = leaf of primitive 0, right = leaf of primitive 1
 */
function threePrimitiveTree(): { floats: Float32Array; words: Uint32Array; parents: Uint32Array } {
    const words = new Uint32Array(2 * GPU_LAYOUT_NODE_WORDS);
    const node = (index: number, left: [number, number], right: [number, number]): void => {
        const base = index * GPU_LAYOUT_NODE_WORDS;
        words[base + GPU_LAYOUT_NODE_WORD.leftIndex] = left[0];
        words[base + GPU_LAYOUT_NODE_WORD.leftCount] = left[1];
        words[base + GPU_LAYOUT_NODE_WORD.rightIndex] = right[0];
        words[base + GPU_LAYOUT_NODE_WORD.rightCount] = right[1];
    };
    node(0, [1, 0], [2, 1]);
    node(1, [0, 1], [1, 1]);
    return { floats: new Float32Array(words.buffer), words, parents: new Uint32Array([GPU_LAYOUT_NO_PARENT, 0]) };
}

function boxes(...list: number[][]): Float32Array {
    const out = new Float32Array(list.length * PRIMITIVE_BOX_FLOATS);
    list.forEach((box, index) => {
        out.set(box.slice(0, 3), index * PRIMITIVE_BOX_FLOATS);
        out.set(box.slice(3, 6), index * PRIMITIVE_BOX_FLOATS + 4);
    });
    return out;
}

/** The six box floats of child slot `slot` (node * 2 + side). */
function slotBox(floats: Float32Array, slot: number): number[] {
    const base = (slot >> 1) * GPU_LAYOUT_NODE_WORDS + (slot & 1) * 8;
    return [floats[base], floats[base + 1], floats[base + 2], floats[base + 4], floats[base + 5], floats[base + 6]];
}

describe('refit topology', () => {
    it('lists leaf slots in primitive order and parent slots by side', () => {
        const tree = threePrimitiveTree();
        expect([...leafSlotsOf(tree.words, 0, 2)]).toEqual([2, 3, 1]);
        expect([...parentSlotsOf(tree.words, 0, 2, tree.parents)]).toEqual([GPU_LAYOUT_NO_PARENT, 0]);
    });

    it('skips the missing right child of a one-primitive tree', () => {
        const words = new Uint32Array(GPU_LAYOUT_NODE_WORDS);
        words[GPU_LAYOUT_NODE_WORD.leftCount] = 1;
        words[GPU_LAYOUT_NODE_WORD.rightIndex] = GPU_LAYOUT_EMPTY_CHILD;
        expect([...leafSlotsOf(words, 0, 1)]).toEqual([0]);
    });
});

describe('refitOnCpu', () => {
    it('gives every slot the union below it, and sums the inner nodes’ half-areas', () => {
        const tree = threePrimitiveTree();
        const result = refitOnCpu(tree.floats, tree.words, 0, 2, tree.parents, boxes([0, 0, 0, 1, 1, 1], [2, 0, 0, 3, 1, 1], [0, 5, 0, 1, 6, 1]));
        expect(slotBox(tree.floats, 2)).toEqual([0, 0, 0, 1, 1, 1]);
        expect(slotBox(tree.floats, 3)).toEqual([2, 0, 0, 3, 1, 1]);
        expect(slotBox(tree.floats, 0)).toEqual([0, 0, 0, 3, 1, 1]);
        expect(slotBox(tree.floats, 1)).toEqual([0, 5, 0, 1, 6, 1]);
        expect([...result.rootBox]).toEqual([0, 0, 0, 3, 6, 1]);
        const expectedSum = halfArea([0, 0, 0, 3, 1, 1]) + halfArea([0, 0, 0, 3, 6, 1]);
        expect(result.summedHalfArea).toBeCloseTo(expectedSum, 10);
        expect(relativeSahCost(tree.floats, tree.words, 0, 2)).toBeCloseTo(expectedSum / halfArea([0, 0, 0, 3, 6, 1]), 10);
    });

    it('stores an empty primitive as a point at the largest float, and leaves it out of unions', () => {
        const tree = threePrimitiveTree();
        const empty = [BVH_FLOAT_MAX, BVH_FLOAT_MAX, BVH_FLOAT_MAX, -BVH_FLOAT_MAX, -BVH_FLOAT_MAX, -BVH_FLOAT_MAX];
        refitOnCpu(tree.floats, tree.words, 0, 2, tree.parents, boxes([0, 0, 0, 1, 1, 1], empty, [0, 5, 0, 1, 6, 1]));
        expect(slotBox(tree.floats, 3)).toEqual(new Array(6).fill(Math.fround(BVH_FLOAT_MAX)));
        expect(slotBox(tree.floats, 0)).toEqual([0, 0, 0, 1, 1, 1]);
    });
});

describe('refit WGSL', () => {
    const wgsl = bvhRefitWgsl();

    it('needs subgroups and declares both kernels', () => {
        expect(wgsl.startsWith('enable subgroups;')).toBe(true);
        expect(wgsl).toContain('fn bvhRefitClear(');
        expect(wgsl).toContain('fn bvhRefitClimb(');
        expect(wgsl).toContain('subgroupShuffleDown(');
    });

    it('uses no WGSL reserved word as a name', () => {
        // A few of WGSL's reserved words that read like ordinary names; `active` once broke the shader.
        const reserved = ['active', 'attribute', 'cast', 'common', 'demote', 'filter', 'handle', 'layout', 'local', 'module', 'precise', 'sample', 'self', 'shared', 'signed', 'target', 'unsigned', 'volatile'];
        const declared = [...wgsl.matchAll(/\b(?:let|var|fn|const)\s+([A-Za-z_]\w*)/g)].map((match) => match[1]);
        const parameters = [...wgsl.matchAll(/\b([A-Za-z_]\w*)\s*:\s*(?:u32|f32|vec[234][uf]|BvhBox)\b/g)].map((match) => match[1]);
        expect([...declared, ...parameters].filter((name) => reserved.includes(name))).toEqual([]);
    });
});
