/**
 * Register D67: the world cache's multibounce arrays are sized from
 * `useMultibounce` at `prepare`, as Capsaicin's `USE_MULTI_BOUNCE` define
 * does. Prepared without it, the indirect values and their update scratch
 * aren't made (535 MB at the default capacity), no kernel carries code that
 * reads them, and the three second-bounce kernels aren't built.
 */

import { describe, expect, it } from 'vitest';
import type { GpuIndirectArgs } from '@pmndrs/gi-core/platform/contract/compute';
import type { ComputeKernel, ComputeKernelOptions } from '@pmndrs/gi-core/platform/contract/wgsl';
import { claimCellsWgsl, prioritizeCellsWgsl, scatterCellsWgsl, shadeCellsWgsl } from './cell-shading-wgsl.ts';
import { DEFAULT_HASH_GRID_CACHE_OPTIONS, hashGridCacheBufferLayout, hashGridCacheCapacity, type HashGridCacheBufferLayout } from './capacity.ts';
import { HashGridCache } from './hash-grid-cache.ts';
import { generateReservoirsWgsl, populateCellsWgsl, populateScreenProbesWgsl, resolveCellsWgsl, updateTilesWgsl } from './hash-grid-population-wgsl.ts';
import { clearCountersWgsl, hashGridCacheWgsl, purgeTilesWgsl } from './hash-grid-wgsl.ts';
import type { WorldCacheContext } from './world-cache.ts';

const capacity = hashGridCacheCapacity(DEFAULT_HASH_GRID_CACHE_OPTIONS);
/** The indirect values (a `uint2` per cell) and their update scratch (four words per cell): gi1.cpp:497-510, 535-552. */
const INDIRECT_BYTES = capacity.numCells * 8 + capacity.numCells * 16;

/** Every kernel built with or without multibounce, by its builder's name. */
function kernelsBuiltEitherWay(layout: HashGridCacheBufferLayout): Record<string, string> {
    return {
        hashGridCacheWgsl: hashGridCacheWgsl(1, layout),
        clearCountersWgsl: clearCountersWgsl(layout),
        purgeTilesWgsl: purgeTilesWgsl(layout),
        populateScreenProbesWgsl: populateScreenProbesWgsl(layout),
        generateReservoirsWgsl: generateReservoirsWgsl(layout),
        populateCellsWgsl: populateCellsWgsl(layout),
        updateTilesWgsl: updateTilesWgsl(layout),
        resolveCellsWgsl: resolveCellsWgsl(layout),
        claimCellsWgsl: claimCellsWgsl(layout),
        prioritizeCellsWgsl: prioritizeCellsWgsl(layout),
        scatterCellsWgsl: scatterCellsWgsl(layout),
        shadeCellsWgsl: shadeCellsWgsl(layout),
    };
}

/** What a fake `prepare` made: buffers by label, kernels by label. */
interface Made {
    readonly buffers: Map<string, number>;
    readonly kernels: string[];
}

/**
 * A context whose device and builders only record what `prepare` asks for,
 * so the test sees the buffers' sizes and the kernels made without a GPU.
 */
function recordingContext(): { context: WorldCacheContext; made: Made; warnings: string[] } {
    const made: Made = { buffers: new Map(), kernels: [] };
    const warnings: string[] = [];
    const device = {
        limits: { maxStorageBufferBindingSize: 2 ** 31 - 4, maxBufferSize: 2 ** 31 },
        createBuffer: (descriptor: GPUBufferDescriptor) => {
            made.buffers.set(descriptor.label ?? '', descriptor.size);
            return { label: descriptor.label, size: descriptor.size, destroy: () => {} };
        },
        createBindGroup: () => ({}),
        createTexture: () => ({ createView: () => ({}), destroy: () => {} }),
        createSampler: () => ({}),
        pushErrorScope: () => {},
        popErrorScope: () => Promise.resolve(null),
        queue: { writeBuffer: () => {} },
    } as unknown as GPUDevice;
    const kernels = {
        computeKernel: (options: ComputeKernelOptions): Promise<ComputeKernel> => {
            made.kernels.push(options.label);
            return Promise.resolve({ pipeline: { label: options.label } as GPUComputePipeline, ownLayouts: [{}, {}] as GPUBindGroupLayout[], bind: () => {} });
        },
    };
    const compute = {
        indirectModule: {},
        createIndirectArgs: (): Promise<GpuIndirectArgs> => Promise.resolve({ buffer: {} as GPUBuffer, memoryBytes: 12, encode: () => {}, dispose: () => {} } as unknown as GpuIndirectArgs),
    };
    const context = { device, kernels, compute, shade: { module: {} }, timer: { computePass: () => undefined }, warn: (message: string) => warnings.push(message) } as unknown as WorldCacheContext;
    return { context, made, warnings };
}

const SECOND_BOUNCE_KERNELS = ['radiance-cache/PopulateMultibounceCells', 'radiance-cache/GenerateMultibounceReservoirs', 'radiance-cache/UpdateMultibounceCells'];

describe('the multibounce arrays sized at prepare (register D67)', () => {
    it("leaves out the indirect values and their scratch, 534,773,760 bytes at Capsaicin's default capacity", () => {
        const on = hashGridCacheBufferLayout(capacity, true);
        const off = hashGridCacheBufferLayout(capacity, false);
        expect(INDIRECT_BYTES).toBe(534_773_760);
        expect(on.totalBytes - off.totalBytes).toBe(INDIRECT_BYTES);
        expect(off.cellValues.bytes).toBe(capacity.numCells * 8);
        expect(off.updateScratch.bytes).toBe(capacity.numCells * 16);
        expect(off.tileWords).toEqual(on.tileWords);
        // Main's packing (register D6) unchanged: the indirect arrays are the last section of both
        // buffers, so without them each buffer ends where they would start and nothing else moves.
        expect(off.updateScratch.updateCellValueBuffer).toBe(on.updateScratch.updateCellValueBuffer);
        expect(off.words.bytes).toBe((on.updateScratch.updateCellValueIndirectBuffer ?? Number.NaN) * 4);
        expect(off.cellValues.bytes).toBe((on.cellValues.valueIndirectBuffer ?? Number.NaN) * 8);
        expect(off.cellValues.valueIndirectBuffer).toBeNull();
        expect(off.updateScratch.updateCellValueIndirectBuffer).toBeNull();
        expect(on.multibounce).toBe(true);
        expect(off.multibounce).toBe(false);
    });

    it('builds every kernel without the code that reads the indirect arrays, and keeps it with them', () => {
        const indirectCode = /INDIRECTBUFFER|ValueIndirectBuffer|FilteredRadianceIndirect|SumIndirect/;
        for (const [builder, code] of Object.entries(kernelsBuiltEitherWay(hashGridCacheBufferLayout(capacity, false)))) {
            expect(code, builder).not.toMatch(indirectCode);
        }
        const withArrays = kernelsBuiltEitherWay(hashGridCacheBufferLayout(capacity, true));
        for (const builder of ['hashGridCacheWgsl', 'updateTilesWgsl', 'resolveCellsWgsl', 'claimCellsWgsl']) expect(withArrays[builder], builder).toMatch(indirectCode);
    });

    it('prepare without useMultibounce makes the smaller buffers and no second-bounce kernel; with it, both', async () => {
        const off = recordingContext();
        const cacheOff = new HashGridCache({ ...DEFAULT_HASH_GRID_CACHE_OPTIONS, useMultibounce: false });
        await cacheOff.prepare(off.context);
        const scratchStartBytes = hashGridCacheBufferLayout(capacity, false).updateScratch.updateCellValueBuffer * 4;
        expect(off.made.buffers.get('radiance-cache/hash-grid cell values')).toBe(capacity.numCells * 8);
        // The words: the tile words, then the direct update scratch only.
        expect(off.made.buffers.get('radiance-cache/hash-grid words')).toBe(scratchStartBytes + capacity.numCells * 16);
        for (const kernel of SECOND_BOUNCE_KERNELS) expect(off.made.kernels).not.toContain(kernel);
        expect(cacheOff.preparedWithMultibounce).toBe(false);
        expect(cacheOff.stats['multibounce arrays']).toBe(0);

        const on = recordingContext();
        const cacheOn = new HashGridCache({ ...DEFAULT_HASH_GRID_CACHE_OPTIONS });
        await cacheOn.prepare(on.context);
        expect(on.made.buffers.get('radiance-cache/hash-grid cell values')).toBe(capacity.numCells * 16);
        expect(on.made.buffers.get('radiance-cache/hash-grid words')).toBe(scratchStartBytes + capacity.numCells * 32);
        for (const kernel of SECOND_BOUNCE_KERNELS) expect(on.made.kernels).toContain(kernel);
        expect(cacheOn.preparedWithMultibounce).toBe(true);
        expect(cacheOn.stats['multibounce arrays']).toBe(1);

        // The arrays and the three passes' indirect arguments (12 bytes each here).
        expect(cacheOn.memoryBytes - cacheOff.memoryBytes).toBe(INDIRECT_BYTES + 3 * 12);
    });

    it('a prepare after the option changes makes the other size', async () => {
        const { context } = recordingContext();
        const options = { ...DEFAULT_HASH_GRID_CACHE_OPTIONS, useMultibounce: false };
        const cache = new HashGridCache(options);
        await cache.prepare(context);
        expect(cache.layout.cellValues.bytes).toBe(capacity.numCells * 8);
        options.useMultibounce = true;
        await cache.prepare(context);
        expect(cache.preparedWithMultibounce).toBe(true);
        expect(cache.layout.cellValues.bytes).toBe(capacity.numCells * 16);
    });
});
