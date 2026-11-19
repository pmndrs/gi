import { describe, expect, it } from 'vitest';
import { GILimitError } from '../../../platform/contract/errors.ts';
import type { Vec3 } from '../../../platform/contract/math.ts';
import { checkHashGridCacheLimits, DEFAULT_HASH_GRID_CACHE_OPTIONS, hashGridCacheBufferLayout, hashGridCacheCapacity } from './capacity.ts';
import { HashGridCache } from './hash-grid-cache.ts';
import {
    hashGridCacheCellIndex,
    hashGridCacheCellOffsetMip0,
    hashGridCacheCellSizeConstant,
    hashGridCacheFilteredRadiance,
    hashGridCacheGetCellSize,
    hashGridCacheGetDesc,
    hashGridCacheRoundHalfBits,
    hashGridCacheTileAge,
    pcgHash,
    updateTilesAccumulate,
    verticalFieldOfView,
    xxHash,
    type HashGridCacheDescConstants,
} from './hash-grid-cpu.ts';
import { checkHashGridRayListLimits, hashGridCacheRayListLayout, RESERVOIR_WORDS, updateTilesPerWorkgroup } from './hash-grid-population-wgsl.ts';
import { floatToHalfBits, halfBitsToFloat } from '../capsaicin-math.ts';
import { HASH_GRID_CONSTANTS_BYTES, HASH_GRID_CONSTANTS_WORD, hashGridCacheWgsl } from './hash-grid-wgsl.ts';
import type { WorldCacheContext } from './world-cache.ts';

// ---- Independent references: the HLSL hashes in BigInt arithmetic, mod 2^32 at every step ----

const MASK = 0xffffffffn;

function pcgHashBigInt(value: number): number {
    const state = (BigInt(value) * 747796405n + 2891336453n) & MASK;
    const shift = (state >> 28n) + 4n;
    const word = (((state >> shift) ^ state) * 277803737n) & MASK;
    return Number((word >> 22n) ^ word);
}

function xxHashBigInt(value: number): number {
    const rotateLeft17 = (x: bigint): bigint => ((x << 17n) | (x >> 15n)) & MASK;
    let ret = (BigInt(value) + 374761393n) & MASK;
    ret = (668265263n * rotateLeft17(ret)) & MASK;
    ret = (2246822519n * (ret ^ (ret >> 15n))) & MASK;
    ret = (3266489917n * (ret ^ (ret >> 13n))) & MASK;
    return Number(ret ^ (ret >> 16n));
}

const add = (a: number, b: number): number => (a + b) >>> 0;

/** `hash_grid_cache.hlsl:153-156` as written there, with the BigInt pcg. */
function bucketChainWrittenOut(l: number, c: readonly number[], d: readonly number[], t: number, numBuckets: number): number {
    const p = pcgHashBigInt;
    const dz = p(add(d[2], p(t)));
    const dy = p(add(d[1], dz));
    const dx = p(add(d[0], dy));
    const cz = p(add(c[2], dx));
    const cy = p(add(c[1], cz));
    const cx = p(add(c[0], cy));
    return p(add(l, cx)) % numBuckets;
}

/** `hash_grid_cache.hlsl:158-162` as written there, with the BigInt xxHash. */
function checksumChainWrittenOut(l: number, c: readonly number[], d: readonly number[], t: number): number {
    const x = xxHashBigInt;
    const dz = x(add(d[2], x(t)));
    const dy = x(add(d[1], dz));
    const dx = x(add(d[0], dy));
    const cz = x(add(c[2], dx));
    const cy = x(add(c[1], cz));
    const cx = x(add(c[0], cy));
    return Math.max(1, x(add(l, cx)));
}

/** A seeded LCG in [0, 1). */
function seeded(seed: number): () => number {
    let state = seed >>> 0;
    return () => {
        state = (Math.imul(state, 1664525) + 1013904223) >>> 0;
        return state / 2 ** 32;
    };
}

/** 60° vertical field of view at 1280 × 720 with Capsaicin's default 32-pixel cells. */
const CELL_SIZE_60_720P = hashGridCacheCellSizeConstant(Math.PI / 3, 1280, 720, 32);

const CONSTANTS: HashGridCacheDescConstants = { cellSize: CELL_SIZE_60_720P, minCellSize: 0.1, tileCellRatio: 8, numBuckets: 1 << 14 };
const EYE: Vec3 = [0, 0, 0];

describe('pcgHash and xxHash (math/hash.hlsl)', () => {
    const random = seeded(11);
    const inputs = [0, 1, 2, 3, 12345, 0x7fffffff, 0x80000000, 0xfffffffe, 0xffffffff, ...Array.from({ length: 1000 }, () => Math.floor(random() * 2 ** 32))];

    it('pcgHash equals the BigInt reference', () => {
        for (const value of inputs) expect(pcgHash(value)).toBe(pcgHashBigInt(value));
    });

    it('xxHash equals the BigInt reference', () => {
        for (const value of inputs) expect(xxHash(value)).toBe(xxHashBigInt(value));
    });
});

describe('HashGridCache_GetDesc (hash_grid_cache.hlsl:133-192)', () => {
    it('sizes cells by distance: 10 m at 60° and 720p gives the 0.256 m cell and 2.048 m tile', () => {
        // source-map.md §4.2's example: about 0.31 m rounds down to a power of two in millimetres.
        expect(CELL_SIZE_60_720P).toBeCloseTo(0.04658, 4);
        const desc = hashGridCacheGetDesc({ eyePosition: EYE, hitPosition: [0, 0, -10], direction: [0, 0, -1], hitDistance: 10 }, CONSTANTS);
        expect(desc.level).toBe(8);
        expect(desc.cellSize).toBe(Math.fround(Math.fround(1e-3) * 256));
        expect(desc.tileSize).toBe(Math.fround(desc.cellSize * 8));
        expect(hashGridCacheGetCellSize([0, 0, -10], EYE, CONSTANTS)).toBe(desc.cellSize);
    });

    it('never goes below the minimum cell: 0.1 m gives level 6 (64 mm cells)', () => {
        const desc = hashGridCacheGetDesc({ eyePosition: EYE, hitPosition: [0.1, 0.2, 0.3], direction: [1, 0, 0], hitDistance: 1 }, CONSTANTS);
        expect(desc.level).toBe(6);
        expect(desc.cellSize).toBe(Math.fround(Math.fround(1e-3) * 64));
    });

    it('bins directions into five per axis', () => {
        const binOf = (direction: Vec3): Vec3 => hashGridCacheGetDesc({ eyePosition: EYE, hitPosition: [1, 1, 1], direction, hitDistance: 5 }, CONSTANTS).directionBin;
        expect(binOf([1, 0, 0])).toEqual([4, 2, 2]);
        expect(binOf([-1, 0, 0])).toEqual([0, 2, 2]);
        expect(binOf([0.6, 0.8, 0])).toEqual([3, 4, 2]);
        expect(binOf([0, -0.6, -0.8])).toEqual([2, 1, 0]);
    });

    it('takes negative tile coordinates as their u32 bits, and flags rays shorter than a tile', () => {
        const hit: Vec3 = [-0.5, 0.3, -3.1];
        const long = hashGridCacheGetDesc({ eyePosition: EYE, hitPosition: hit, direction: [0, 0, -1], hitDistance: 100 }, CONSTANTS);
        const short = hashGridCacheGetDesc({ eyePosition: EYE, hitPosition: hit, direction: [0, 0, -1], hitDistance: long.tileSize * 0.5 }, CONSTANTS);
        expect(long.tileCoordinate[0]).toBe(-1);
        expect(long.shortRay).toBe(0);
        expect(short.shortRay).toBe(1);
        expect(short.tileHash).not.toBe(long.tileHash);
    });

    it('builds the bucket and the checksum exactly as the HLSL chains are written', () => {
        const random = seeded(5);
        for (let index = 0; index < 500; index++) {
            const hit: Vec3 = [(random() - 0.5) * 80, (random() - 0.5) * 80, (random() - 0.5) * 80];
            const direction: Vec3 = [random() - 0.5, random() - 0.5, random() - 0.5];
            const length = Math.hypot(...direction);
            const data = { eyePosition: [1, 2, 3] as Vec3, hitPosition: hit, direction: direction.map((value) => value / length) as unknown as Vec3, hitDistance: random() * 4 };
            const desc = hashGridCacheGetDesc(data, CONSTANTS);
            const c = desc.tileCoordinate.map((value) => value >>> 0);
            const d = desc.directionBin.map((value) => value >>> 0);
            expect(desc.bucketIndex).toBe(bucketChainWrittenOut(desc.level, c, d, desc.shortRay, CONSTANTS.numBuckets));
            expect(desc.tileHash).toBe(checksumChainWrittenOut(desc.level, c, d, desc.shortRay));
            expect(desc.tileHash).toBeGreaterThanOrEqual(1);
            expect(desc.bucketIndex).toBeLessThan(CONSTANTS.numBuckets);
        }
    });

    it('projects the cell inside its tile along the ray\'s dominant axis', () => {
        // Level 6 (64 mm cells, 512 mm tiles) near the eye; the hit is in cell (1, 2, 3) of tile (0, 0, 0).
        const cell = Math.fround(Math.fround(1e-3) * 64);
        const hit: Vec3 = [1.5 * cell, 2.5 * cell, 3.5 * cell];
        const offsetFor = (direction: Vec3): readonly number[] => hashGridCacheGetDesc({ eyePosition: EYE, hitPosition: hit, direction, hitDistance: 1 }, CONSTANTS).cellOffset;
        expect(offsetFor([1, 0, 0])).toEqual([2, 3]);
        expect(offsetFor([0, -1, 0])).toEqual([1, 3]);
        expect(offsetFor([0, 0, 1])).toEqual([1, 2]);
        // A tie goes to x first, then y (the HLSL's if-else order).
        expect(offsetFor([Math.SQRT1_2, 0, Math.SQRT1_2])).toEqual([2, 3]);
    });
});

describe('the tile layout (gi1.cpp:442-461) and cell indices (hash_grid_cache.hlsl:195-246)', () => {
    const capacity = hashGridCacheCapacity(DEFAULT_HASH_GRID_CACHE_OPTIONS);

    it('has Capsaicin\'s default capacity: 2^14 buckets × 16 tiles × 85 cells', () => {
        expect(capacity.numBuckets).toBe(16384);
        expect(capacity.numTilesPerBucket).toBe(16);
        expect(capacity.sizeTileMip).toEqual([8, 4, 2, 1]);
        expect(capacity.numCellsPerTileMip).toEqual([64, 16, 4, 1]);
        expect(capacity.numCellsPerTile).toBe(85);
        expect(capacity.firstCellOffsetTileMip).toEqual([0, 64, 80, 84]);
        expect(capacity.numTiles).toBe(262144);
        expect(capacity.numCells).toBe(22282240);
    });

    it('maps a finest-level cell to its coarser levels and back', () => {
        expect(hashGridCacheCellIndex([5, 6], 3, 0, capacity)).toBe(3 * 85 + 5 + 6 * 8);
        expect(hashGridCacheCellIndex([5, 6], 3, 1, capacity)).toBe(3 * 85 + 64 + 2 + 3 * 4);
        expect(hashGridCacheCellIndex([5, 6], 3, 2, capacity)).toBe(3 * 85 + 80 + 1 + 1 * 2);
        expect(hashGridCacheCellIndex([5, 6], 3, 3, capacity)).toBe(3 * 85 + 84);
        for (let y = 0; y < 8; y++) {
            for (let x = 0; x < 8; x++) {
                const back = hashGridCacheCellOffsetMip0(hashGridCacheCellIndex([x, y], 1234, 0, capacity), capacity);
                expect(back.tileIndex).toBe(1234);
                expect(back.cellOffset).toEqual([x, y]);
            }
        }
    });

    it('refuses a tile cell ratio that would need a fifth level', () => {
        expect(() => hashGridCacheCapacity({ ...DEFAULT_HASH_GRID_CACHE_OPTIONS, hashGridCacheTileCellRatio: 16 })).toThrow(RangeError);
    });
});

describe('the packed buffers (register D6) and the device limits (D14)', () => {
    const capacity = hashGridCacheCapacity(DEFAULT_HASH_GRID_CACHE_OPTIONS);
    const layout = hashGridCacheBufferLayout(capacity);

    it('packs the value, indirect value and both update scratch arrays at Capsaicin\'s sizes', () => {
        // gi1.cpp:487-547: uint2 per cell twice, num_cells << 2 uints twice.
        expect(layout.cellValues.bytes).toBe(2 * 22282240 * 8);
        expect(layout.updateScratch.bytes).toBe(2 * 22282240 * 4 * 4);
        // Counters (64 words), then hash, decay, the two packed lists and the update-listed word (D32), one word per tile each.
        expect(layout.tileWords.bytes).toBe((64 + 5 * 262144) * 4);
        expect(layout.totalBytes).toBe(356515840 + 713031680 + 5243136);
        // Offsets on the 256-byte storage-binding alignment.
        for (const offsetWords of [layout.tileWords.hashBuffer, layout.tileWords.decayTileBuffer, layout.tileWords.packedTileIndexBuffer0, layout.tileWords.packedTileIndexBuffer1, layout.tileWords.updateListedTileBuffer]) {
            expect((offsetWords * 4) % 256).toBe(0);
        }
        expect((layout.cellValues.valueIndirectBuffer * 8) % 256).toBe(0);
    });

    it('throws GILimitError naming maxStorageBufferBindingSize at WebGPU\'s default limits', () => {
        const defaults = { maxStorageBufferBindingSize: 134217728, maxBufferSize: 268435456 };
        expect(() => checkHashGridCacheLimits(layout, defaults)).toThrow(GILimitError);
        try {
            checkHashGridCacheLimits(layout, defaults);
        } catch (error) {
            expect((error as GILimitError).limit).toBe('maxStorageBufferBindingSize');
            expect((error as GILimitError).needed).toBe(layout.updateScratch.bytes);
        }
    });

    it('names maxBufferSize when only the buffer limit is short, and passes at the adapters\' 2 GiB', () => {
        expect(() => checkHashGridCacheLimits(layout, { maxStorageBufferBindingSize: 2 ** 31, maxBufferSize: 2 ** 28 })).toThrow(/maxBufferSize/);
        expect(() => checkHashGridCacheLimits(layout, { maxStorageBufferBindingSize: 2 ** 31 - 4, maxBufferSize: 2 ** 31 })).not.toThrow();
    });

    it('prepare refuses a default-limits device before making anything', async () => {
        const device = {
            limits: { maxStorageBufferBindingSize: 134217728, maxBufferSize: 268435456 },
            createBuffer: () => {
                throw new Error('nothing should be made');
            },
        } as unknown as GPUDevice;
        const context = { device, kernels: {}, compute: {} } as unknown as WorldCacheContext;
        await expect(new HashGridCache().prepare(context)).rejects.toThrow(GILimitError);
    });
});

describe('PurgeTiles\' decay age (gi1.comp:1726-1738)', () => {
    it('counts frames since the stamp, across the frame counter\'s wrap', () => {
        expect(hashGridCacheTileAge(100, 50)).toBe(50);
        expect(hashGridCacheTileAge(50, 50)).toBe(0);
        expect(hashGridCacheTileAge(10, 0xfffffff0)).toBe(26);
        expect(hashGridCacheTileAge(0, 0xffffffff)).toBe(1);
        expect(hashGridCacheTileAge(2 ** 32 + 33, 2 ** 32 - 16)).toBe(49);
    });
});

describe('the constants and the shared WGSL', () => {
    it('lays HashGridCacheConstants out in 32 words, the eye first', () => {
        expect(HASH_GRID_CONSTANTS_BYTES).toBe(128);
        expect(HASH_GRID_CONSTANTS_WORD.frameIndex).toBe(3);
        expect(HASH_GRID_CONSTANTS_WORD.maxMultibounceSampleCount).toBeLessThan(32);
    });

    it('reads the vertical field of view back from a projection', () => {
        const fovY = Math.PI / 3;
        const projection = new Array<number>(16).fill(0);
        projection[5] = 1 / Math.tan(fovY / 2);
        expect(verticalFieldOfView(projection)).toBeCloseTo(fovY, 12);
    });

    it('declares every ported function once, with the packed offsets baked in', () => {
        const layout = hashGridCacheBufferLayout(hashGridCacheCapacity(DEFAULT_HASH_GRID_CACHE_OPTIONS));
        const code = hashGridCacheWgsl(1, layout);
        for (const name of [
            'pcgHash',
            'xxHash',
            'packHalf4',
            'unpackHalf4',
            'HashGridCache_GetCellSize',
            'HashGridCache_GetTileSize',
            'HashGridCache_GetDesc',
            'HashGridCache_CellOffsetMip0',
            'HashGridCache_CellIndex',
            'HashGridCache_CellIndexLinear',
            'HashGridCache_InsertCell',
            'HashGridCache_FindCell',
            'HashGridCache_TouchTile',
            'HashGridCache_AddNewTile',
            'HashGridCache_QuantizeRadiance',
            'HashGridCache_RecoverRadiance',
            'HashGridCache_PackRadiance',
            'HashGridCache_UnpackRadiance',
            'HashGridCache_FilteredRadianceDirect',
            'HashGridCache_FilteredRadianceIndirect',
            'worldCacheLookup',
        ]) {
            expect(code.match(new RegExp(`\\bfn ${name}\\(`, 'g'))?.length, name).toBe(1);
        }
        expect(code).toContain(`const HASHGRIDCACHE_DECAYTILEBUFFER: u32 = ${layout.tileWords.decayTileBuffer}u;`);
        expect(code).toContain('@group(1) @binding(1) var<storage, read_write> g_HashGridCache_TileWords: array<atomic<u32>>;');
    });
});

describe('UpdateTiles accumulation (gi1.comp:2162-2177)', () => {
    it('takes the first samples as they are, stored times their count up to the cap', () => {
        expect(updateTilesAccumulate([0, 0, 0, 0], [6, 3, 0, 3], 16)).toEqual([6, 3, 0, 3]);
        // 20 samples of radiance 0.5: the mean, stored times the capped count.
        expect(updateTilesAccumulate([0, 0, 0, 0], [10, 10, 10, 20], 16)).toEqual([8, 8, 8, 16]);
    });

    it('caps the sample count at max_sample_count and blends by 1 / count', () => {
        // History: 16 samples of 1. New: 4 samples of 2. Blend 1 + (2 - 1) / 16, times 16.
        const result = updateTilesAccumulate([16, 16, 16, 16], [8, 8, 8, 4], 16);
        expect(result[3]).toBe(16);
        expect(result[0]).toBeCloseTo(17, 5);
        // Below the cap the count grows by the new samples.
        const growing = updateTilesAccumulate([5, 5, 5, 5], [3, 3, 3, 3], 16);
        expect(growing[3]).toBe(8);
        expect(growing[0] / growing[3]).toBeCloseTo(1, 6);
    });

    it("keeps the mean but loses a sample when a cell of an updated tile gets none (Capsaicin's behaviour)", () => {
        const result = updateTilesAccumulate([32, 32, 32, 16], [0, 0, 0, 0], 16);
        expect(result[3]).toBeCloseTo(15, 5);
        expect(result[0] / result[3]).toBeCloseTo(2, 5);
    });
});

describe('the filtered readback (hash_grid_cache.hlsl:434-494)', () => {
    const mip0: [number, number, number, number] = [3, 3, 3, 3];
    const mip1: [number, number, number, number] = [20, 20, 20, 10];
    const mip2: [number, number, number, number] = [60, 60, 60, 30];
    const mip3: [number, number, number, number] = [100, 100, 100, 50];

    it('reads the finest cell once it has max_sample_count samples', () => {
        expect(hashGridCacheFilteredRadiance([[16, 16, 16, 16], mip1, mip2, mip3], 16).level).toBe(0);
    });

    it('climbs to a coarser level while the value in hand has fewer samples', () => {
        // 3 < 16 takes mip 1; its 10 < 16 takes mip 2, whose 30 stops the climb.
        expect(hashGridCacheFilteredRadiance([mip0, mip1, mip2, mip3], 16)).toEqual({ level: 2, value: mip2 });
        expect(hashGridCacheFilteredRadiance([mip0, mip1, [1, 1, 1, 1], mip3], 16)).toEqual({ level: 3, value: mip3 });
    });

    it('stops where a level has no cells', () => {
        expect(hashGridCacheFilteredRadiance([mip0, mip1, null, null], 16)).toEqual({ level: 1, value: mip1 });
    });
});

describe('the per-ray lists (gi1.cpp:637-695)', () => {
    it("sizes them as Capsaicin does with multibounce on, one after another", () => {
        // 1280 x 720: 80 x 45 spawn tiles of 64 rays.
        const maxRayCount = 80 * 45 * 64;
        const layout = hashGridCacheRayListLayout(maxRayCount, 22282240);
        expect(layout.maxCombinedRayCount).toBe(2 * maxRayCount);
        expect(layout.updateTileBuffer).toBe(0);
        expect(layout.visibilityCellBuffer).toBe(2 * maxRayCount);
        expect(layout.visibilityQueryBuffer).toBe(4 * maxRayCount);
        expect(layout.visibilityRayBuffer).toBe(6 * maxRayCount);
        expect(layout.resolveBuffer).toBe(8 * maxRayCount);
        expect(layout.reservoirBuffer).toBe(9 * maxRayCount);
        // Then GI-1.2's second-bounce lists (RC3.3): the BRDF and pdf, and the ray (register D36).
        expect(layout.multibounceInfoBuffer).toBe(9 * maxRayCount + RESERVOIR_WORDS * 2 * maxRayCount);
        expect(layout.bytes).toBe((layout.multibounceInfoBuffer + (2 + 12 + 1 + 2) * maxRayCount) * 4);
    });

    it('holds at most one update entry per cell', () => {
        expect(hashGridCacheRayListLayout(1000, 100).visibilityCellBuffer).toBe(100);
    });

    it('throws GILimitError naming the limit the lists pass (register D14)', () => {
        // 3840 x 2160: 240 x 135 spawn tiles of 64 rays, 265 MB of lists: past WebGPU's default 128 MiB binding.
        const layout = hashGridCacheRayListLayout(240 * 135 * 64, 22282240);
        expect(layout.bytes).toBe(32 * 240 * 135 * 64 * 4);
        const defaults = { maxStorageBufferBindingSize: 134217728, maxBufferSize: 268435456 };
        expect(() => checkHashGridRayListLimits(layout, defaults)).toThrow(/maxStorageBufferBindingSize/);
        expect(() => checkHashGridRayListLimits(layout, { maxStorageBufferBindingSize: 2 ** 31 - 4, maxBufferSize: 2 ** 31 })).not.toThrow();
        const small = { maxStorageBufferBindingSize: 64 * 2 ** 20, maxBufferSize: 268435456 };
        expect(() => checkHashGridRayListLimits(layout, small)).toThrow(GILimitError);
        try {
            checkHashGridRayListLimits(layout, small);
        } catch (error) {
            expect((error as GILimitError).limit).toBe('maxStorageBufferBindingSize');
            expect((error as GILimitError).needed).toBe(layout.bytes);
        }
        expect(() => checkHashGridRayListLimits(layout, { maxStorageBufferBindingSize: 2 ** 31, maxBufferSize: 64 * 2 ** 20 })).toThrow(/maxBufferSize/);
    });
});

describe('UpdateTiles workgroups (gi1.comp:2116-2140)', () => {
    it('covers whole tiles in its 8 x 8 group', () => {
        expect(updateTilesPerWorkgroup(8)).toBe(1);
        expect(updateTilesPerWorkgroup(4)).toBe(4);
        expect(updateTilesPerWorkgroup(1)).toBe(64);
        expect(() => updateTilesPerWorkgroup(3)).toThrow(RangeError);
    });
});

describe('HashGridCache_RoundHalfBits (register D31)', () => {
    /** The half below or at `value` (what a truncating pack2x16float gives) and the one above. */
    function neighbours(value: number): { truncated: number; roundedUp: number } {
        const nearest = floatToHalfBits(value);
        const truncated = Math.abs(halfBitsToFloat(nearest)) > Math.abs(Math.fround(value)) ? nearest - 1 : nearest;
        const roundedUp = Math.abs(halfBitsToFloat(truncated)) < Math.abs(Math.fround(value)) ? truncated + 1 : truncated;
        return { truncated, roundedUp };
    }

    it('rounds an exact tie to the even half, as f32tof16 does, from either neighbour', () => {
        // 1 + 2^-11 lies halfway between 0x3c00 (even) and 0x3c01; 1 + 3 x 2^-11 between 0x3c01 and 0x3c02 (even).
        for (const [value, expected] of [
            [1 + 2 ** -11, 0x3c00],
            [1 + 3 * 2 ** -11, 0x3c02],
            [-(1 + 2 ** -11), 0xbc00],
            [-(1 + 3 * 2 ** -11), 0xbc02],
            [1000 + 0.25, 0x63d0], // 1000 is 0x63d0, the next half 1000.5: a tie to the even 1000
            [3 * 2 ** -25, 0x0002], // between the subnormals 1 and 2 (x 2^-24): to 2
        ] as const) {
            const { truncated, roundedUp } = neighbours(value);
            expect(truncated).not.toBe(roundedUp);
            expect(hashGridCacheRoundHalfBits(value, truncated)).toBe(expected);
            expect(hashGridCacheRoundHalfBits(value, roundedUp)).toBe(expected);
            expect(floatToHalfBits(value)).toBe(expected);
        }
    });

    it('gives the nearest half away from a tie, and keeps exact halves and the largest one', () => {
        const random = (() => {
            let state = 12345;
            return () => ((state = (Math.imul(state, 1664525) + 1013904223) >>> 0) / 2 ** 32);
        })();
        for (let index = 0; index < 5000; index++) {
            const value = Math.fround((random() - 0.5) * 2 ** (random() * 40 - 24));
            if (Math.abs(value) > 65504) continue;
            const { truncated, roundedUp } = neighbours(value);
            expect(hashGridCacheRoundHalfBits(value, truncated)).toBe(floatToHalfBits(value));
            expect(hashGridCacheRoundHalfBits(value, roundedUp)).toBe(floatToHalfBits(value));
        }
        expect(hashGridCacheRoundHalfBits(1, 0x3c00)).toBe(0x3c00);
        expect(hashGridCacheRoundHalfBits(65504, 0x7bff)).toBe(0x7bff);
        expect(hashGridCacheRoundHalfBits(65503, 0x7bfe)).toBe(0x7bff);
    });
});
