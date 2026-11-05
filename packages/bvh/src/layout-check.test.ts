import { describe, expect, it } from 'vitest';
import { GPU_LAYOUT_NODE_WORD, GPU_LAYOUT_NODE_WORDS, GPU_LAYOUT_TRIANGLE_BYTES, GPU_LAYOUT_TRIANGLE_WORD, type GpuLayoutMeshBvh } from './gpu-layout.ts';
import { planGpuLbvhBatches } from './gpu-lbvh.ts';
import { checkGpuLayout } from './layout-check.ts';
import { createTinyBvhBuilder } from './tinybvh.ts';

function randomSoup(count: number, seed: number): Float32Array {
    let state = seed >>> 0;
    const random = () => ((state = (state * 1664525 + 1013904223) >>> 0) / 2 ** 32);
    return Float32Array.from({ length: count * 9 }, () => random() * 10);
}

/** A copy of the layout whose buffers can be changed without touching the original. */
function copy(layout: GpuLayoutMeshBvh): GpuLayoutMeshBvh {
    return { ...layout, nodes: layout.nodes.slice(0), triangles: layout.triangles.slice(0), parents: layout.parents.slice(), bounds: layout.bounds.slice() };
}

/** The first child slot of the root that is a leaf, as (word of its count, word of its min). */
function firstLeafSlot(layout: GpuLayoutMeshBvh): { countWord: number; minWord: number } {
    const words = new Uint32Array(layout.nodes);
    for (let node = 0; node < layout.nodeCount; node++) {
        for (const side of [0, 1]) {
            const countWord = node * GPU_LAYOUT_NODE_WORDS + (side === 0 ? GPU_LAYOUT_NODE_WORD.leftCount : GPU_LAYOUT_NODE_WORD.rightCount);
            if (words[countWord] > 1) return { countWord, minWord: node * GPU_LAYOUT_NODE_WORDS + (side === 0 ? GPU_LAYOUT_NODE_WORD.leftMin : GPU_LAYOUT_NODE_WORD.rightMin) };
        }
    }
    throw new Error('no leaf of more than one triangle');
}

describe('checkGpuLayout catches broken trees', async () => {
    const geometry = { positions: randomSoup(2000, 3) };
    const layout = (await createTinyBvhBuilder()).buildLayout(geometry).layout;
    const triangleWords = GPU_LAYOUT_TRIANGLE_BYTES / 4;

    it('passes the tree as built', () => {
        expect(checkGpuLayout(layout, geometry)).toEqual([]);
    });

    const breakages: [string, (broken: GpuLayoutMeshBvh) => void][] = [
        ['a box one float step too big', (broken) => {
            const floats = new Float32Array(broken.nodes);
            const { minWord } = firstLeafSlot(broken);
            floats[minWord] = floats[minWord] - Math.abs(floats[minWord]) * 2 ** -23 - 1e-30;
        }],
        ['a triangle placed twice', (broken) => {
            const words = new Uint32Array(broken.triangles);
            words[GPU_LAYOUT_TRIANGLE_WORD.originalIndex] = words[triangleWords + GPU_LAYOUT_TRIANGLE_WORD.originalIndex];
        }],
        ['a leaf one triangle short', (broken) => {
            new Uint32Array(broken.nodes)[firstLeafSlot(broken).countWord] -= 1;
        }],
        ['a moved corner', (broken) => {
            new Float32Array(broken.triangles)[GPU_LAYOUT_TRIANGLE_WORD.vertex1 + 1] += 0.5;
        }],
        ['a wrong parent', (broken) => {
            broken.parents[broken.nodeCount - 1] = broken.nodeCount - 1;
        }],
        ['a wrong mesh box', (broken) => {
            broken.bounds[3] += 1;
        }],
        ['a wrong depth', (broken) => {
            Object.assign(broken, { maxDepth: broken.maxDepth + 1 });
        }],
    ];
    for (const [name, breakIt] of breakages) {
        it(`reports ${name}`, () => {
            const broken = copy(layout);
            breakIt(broken);
            expect(checkGpuLayout(broken, geometry).length).toBeGreaterThan(0);
        });
    }
});

describe('planGpuLbvhBatches', () => {
    // Only the limits are read.
    const device = (limit: number) => ({ limits: { maxStorageBufferBindingSize: limit, maxBufferSize: limit } }) as unknown as GPUDevice;
    const mesh = (triangles: number) => ({ positions: new Float32Array(9), indices: new Uint32Array(triangles * 3) });

    it('keeps meshes in order and splits where the next would not fit', () => {
        // 76 bytes per triangle at most, so a 76 kB limit holds 1000 triangles.
        const plan = planGpuLbvhBatches(device(76_000), [mesh(600), mesh(300), mesh(200), mesh(900), mesh(50)]);
        expect(plan.batches).toEqual([[0, 1], [2], [3, 4]]);
        expect(plan.tooBig).toEqual([]);
    });

    it('sends a mesh too big for any batch to tooBig', () => {
        const plan = planGpuLbvhBatches(device(76_000), [mesh(10), mesh(2000), mesh(10)]);
        expect(plan.batches).toEqual([[0, 2]]);
        expect(plan.tooBig).toEqual([1]);
    });
});
