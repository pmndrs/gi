/**
 * The world-cache page's analytic gate on GI-1.2's second bounce, for the other pages that judge
 * a furnace (RC3.4, the lead's ruling of 2026-11-23): `radiance-cache-cell-shading` and
 * `radiance-cache-ddgi-bounce`. The world-cache page has its own copy over its own readback
 * (`secondBounceSamples` and `nearFloorLandings` in main.ts); this one reads what it needs itself.
 *
 * Why the gate: a hash-grid cell is a column through its tile (NOTES), so a far floor cell can
 * hold ceiling hits too. Since register D63 those count as samples of no light, so such a cell
 * reads the floor's light times its floor share, and a second bounce landing there brings that,
 * not L_f. A near ceiling ray's indirect value is its cell's mean over landings of different floor
 * shares, so single rays, or the near rays' mean, aren't ρ L_f any more. The second bounces whose
 * first hit is in a near ceiling cell and whose second hit is on the floor in a tile under 2 m
 * (which can't reach the ceiling 3 m up, so its cells hold the floor alone) are each ρ L_f by the
 * geometry, read back through the cache: a bias in the cells' direct values, or in the read,
 * moves their mean off ρ L_f; the column share can't.
 *
 * Each landing's sample is computed as `UpdateMultibounceCells` makes it this frame, from the
 * lists and the landing cells' direct values as read back after the frame: the light at the
 * second hit (its cell's filtered direct mean, or its own light where its segment bypasses) ×
 * BRDF / pdf, brightest channel (the furnace is grey). With cells lit on a budget (E8) a landing
 * cell no level of which has been lit adds no sample, as on the GPU.
 */

import type { Vec3 } from '../../../../packages/core/src/platform/contract/math.ts';
import { halfBitsToFloat, unpackHalf3 } from '../../../../packages/cache/src/capsaicin-math.ts';
import type { RadianceCacheTechnique } from '../../../../packages/cache/src/radiance-cache-technique.ts';
import { HASH_GRID_COUNTER } from '../../../../packages/cache/src/world-cache/capacity.ts';
import type { HashGridCache } from '../../../../packages/cache/src/world-cache/hash-grid-cache.ts';
import {
    hashGridCacheCellIndex,
    hashGridCacheCellOffsetMip0,
    hashGridCacheCellSizeConstant,
    hashGridCacheFilteredRadiance,
    hashGridCacheGetCellSize,
    verticalFieldOfView,
    type CellValue,
} from '../../../../packages/cache/src/world-cache/hash-grid-cpu.ts';
import { MULTIBOUNCE_DIRECT_WORDS, MULTIBOUNCE_INFO_WORDS, MULTIBOUNCE_MISS_CELL, MULTIBOUNCE_RAY_WORDS } from '../../../../packages/cache/src/world-cache/hash-grid-population-wgsl.ts';
import { multibounceSample } from '../../../../packages/cache/src/world-cache/multibounce-cpu.ts';
import { WORLD_CACHE_PROBE_RAY_BYPASS, WORLD_CACHE_PROBE_RAY_BYTES, WORLD_CACHE_PROBE_RAYS_HEADER_BYTES } from '../../../../packages/cache/src/world-cache/world-cache.ts';
import { frameCamera, type PageCamera } from '../radiance-cache-probes/screen.ts';

/** A tile under this size (its finest cells × the hit's cell size) can't reach from the ceiling to the floor 3 m below. */
export const NEAR_TILE_SIZE = 2;

export interface LandingOptions {
    readonly camera: PageCamera;
    readonly width: number;
    readonly height: number;
    /** The ceiling's height: first hits on it in tiles under NEAR_TILE_SIZE are the ones judged. */
    readonly ceilingY: number;
}

export interface LandingReading {
    /** The second bounces from near ceiling cells landing on the floor in tiles under NEAR_TILE_SIZE: ρ L_f each. */
    readonly nearFloorLandings: { readonly mean: number; readonly entries: number };
    /** Every second bounce from near ceiling cells that adds a sample, misses (0) included: what the cache gives them this frame. */
    readonly throughCache: { readonly mean: number; readonly entries: number };
    /** First-bounce cells on the ceiling in tiles under NEAR_TILE_SIZE. */
    readonly nearCeilingCells: number;
}

async function readRange(device: GPUDevice, source: GPUBuffer, offset: number, byteLength: number): Promise<ArrayBuffer> {
    const size = Math.max(4, Math.ceil(byteLength / 4) * 4);
    const readback = device.createBuffer({ label: 'second-bounce landings readback', size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const encoder = device.createCommandEncoder({ label: 'second-bounce landings readback' });
    encoder.copyBufferToBuffer(source, offset, readback, 0, size);
    device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    const copy = readback.getMappedRange().slice(0, byteLength);
    readback.unmap();
    readback.destroy();
    return copy;
}

function unpackHalves(words: Uint32Array): Float32Array {
    const out = new Float32Array(words.length * 2);
    for (let index = 0; index < words.length; index++) {
        out[index * 2] = halfBitsToFloat(words[index] & 0xffff);
        out[index * 2 + 1] = halfBitsToFloat(words[index] >>> 16);
    }
    return out;
}

/** The direct values of whole tiles (all four levels), unpacked: `numCellsPerTile` × 4 floats per tile, in `tiles` order. */
async function readDirectTiles(device: GPUDevice, cache: HashGridCache, tiles: readonly number[]): Promise<Float32Array> {
    const capacity = cache.capacity;
    const tileBytes = capacity.numCellsPerTile * 8;
    const staging = device.createBuffer({ label: 'second-bounce landing tiles', size: Math.max(4, tiles.length * tileBytes), usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
    const encoder = device.createCommandEncoder({ label: 'second-bounce landing tiles' });
    const valueBase = cache.layout.cellValues.valueBuffer * 8;
    tiles.forEach((tile, index) => encoder.copyBufferToBuffer(cache.buffers.cellValues, valueBase + tile * tileBytes, staging, index * tileBytes, tileBytes));
    device.queue.submit([encoder.finish()]);
    const words = new Uint32Array(await readRange(device, staging, 0, tiles.length * tileBytes));
    staging.destroy();
    return unpackHalves(words);
}

/** What Capsaicin's filtered readback picks for `cell` from tile number `slot` of `values` (the world-cache page's `pickLevel`). */
function filteredDirect(values: Float32Array, cache: HashGridCache, slot: number, cell: number, maxCount: number): CellValue {
    const capacity = cache.capacity;
    const { tileIndex, cellOffset } = hashGridCacheCellOffsetMip0(cell, capacity);
    const levels = [0, 1, 2, 3].map((level): CellValue | null => {
        if (capacity.sizeTileMip[level] <= 0) return null;
        const base = (slot * capacity.numCellsPerTile + hashGridCacheCellIndex(cellOffset, tileIndex, level, capacity) - tileIndex * capacity.numCellsPerTile) * 4;
        return [values[base], values[base + 1], values[base + 2], values[base + 3]];
    });
    return hashGridCacheFilteredRadiance(levels, maxCount).value;
}

/** Reads this frame's lists and the landing cells after the frame (call after `onSubmittedWorkDone`), and judges nothing. */
export async function readSecondBounceLandings(device: GPUDevice, technique: RadianceCacheTechnique, cache: HashGridCache, options: LandingOptions): Promise<LandingReading> {
    const probes = technique.probes;
    const lists = cache.rayLists;
    if (!probes || !lists) throw new Error('no probes or ray lists: run a frame first');
    const settings = technique.settings;
    const capacity = cache.capacity;
    const tileSide = capacity.sizeTileMip[0];
    const constants = {
        cellSize: hashGridCacheCellSizeConstant(verticalFieldOfView(frameCamera(options.camera, options.width, options.height).projection), options.width, options.height, settings.hashGridCacheCellSize),
        minCellSize: settings.hashGridCacheMinCellSize,
    };
    const eye = options.camera.position as Vec3;
    const inNearTile = (hit: Vec3): boolean => hashGridCacheGetCellSize(hit, eye, constants) * tileSide < NEAR_TILE_SIZE;

    // The probe rays' first hits.
    const rayCount = new Uint32Array(await readRange(device, probes.probeRays, 0, 4))[0];
    const rayWords = new Uint32Array(await readRange(device, probes.probeRays, 0, WORLD_CACHE_PROBE_RAYS_HEADER_BYTES + rayCount * WORLD_CACHE_PROBE_RAY_BYTES));
    const rayFloats = new Float32Array(rayWords.buffer);
    const probeHit = (ray: number): Vec3 | null => {
        const base = (WORLD_CACHE_PROBE_RAYS_HEADER_BYTES + ray * WORLD_CACHE_PROBE_RAY_BYTES) / 4;
        const hitT = rayFloats[base + 6];
        if (!(hitT >= 0)) return null;
        const direction = unpackHalf3([rayWords[base + 4], rayWords[base + 5]]);
        return [rayFloats[base] + hitT * direction[0], rayFloats[base + 1] + hitT * direction[1], rayFloats[base + 2] + hitT * direction[2]];
    };

    // The lists.
    const counters = new Uint32Array(await readRange(device, cache.buffers.tileWords, 0, 64 * 4));
    const words = new Uint32Array(await readRange(device, lists.buffer, 0, lists.buffer.size));
    const layout = lists.layout;
    const count0 = counters[HASH_GRID_COUNTER.visibilityCount0];
    const count1 = counters[HASH_GRID_COUNTER.visibilityCount1];
    const visibilityCell = words.subarray(layout.visibilityCellBuffer, layout.visibilityCellBuffer + count0);
    const visibilityQuery = words.subarray(layout.visibilityQueryBuffer, layout.visibilityQueryBuffer + count0);
    const multibounceCell = words.subarray(layout.visibilityCellBuffer + count0, layout.visibilityCellBuffer + count0 + count1);
    const multibounceQuery = words.subarray(layout.visibilityQueryBuffer + count0, layout.visibilityQueryBuffer + count0 + count1);
    const multibounceInfo = unpackHalves(words.slice(layout.multibounceInfoBuffer, layout.multibounceInfoBuffer + MULTIBOUNCE_INFO_WORDS * count1));
    const multibounceRays = words.slice(layout.multibounceRayBuffer, layout.multibounceRayBuffer + MULTIBOUNCE_RAY_WORDS * count1);
    const multibounceRayFloats = new Float32Array(multibounceRays.buffer);
    const multibounceDirect = unpackHalves(words.slice(layout.multibounceDirectBuffer, layout.multibounceDirectBuffer + MULTIBOUNCE_DIRECT_WORDS * count1));

    // The first-bounce cells on the ceiling in tiles under NEAR_TILE_SIZE (they can't reach the floor).
    const nearCeiling = new Set<number>();
    visibilityCell.forEach((cell, visibility) => {
        const hit = probeHit(visibilityQuery[visibility]);
        if (hit && Math.abs(hit[1] - options.ceilingY) < 1e-3 && inNearTile(hit)) nearCeiling.add(cell);
    });

    // The entries judged, and the tiles their second hits land in.
    const entries: number[] = [];
    multibounceQuery.forEach((query, entry) => {
        if (nearCeiling.has(visibilityCell[query])) entries.push(entry);
    });
    const tileOf = (cell: number): number => Math.floor(cell / capacity.numCellsPerTile);
    const landingTiles = [...new Set(entries.filter((entry) => multibounceCell[entry] !== MULTIBOUNCE_MISS_CELL).map((entry) => tileOf(multibounceCell[entry])))];
    const slotOfTile = new Map(landingTiles.map((tile, slot) => [tile, slot]));
    const direct = await readDirectTiles(device, cache, landingTiles);
    const budgeted = settings.cellShading === 'budgeted';

    let landingSum = 0;
    let landingEntries = 0;
    let cacheSum = 0;
    let cacheEntries = 0;
    for (const entry of entries) {
        const cell = multibounceCell[entry];
        if (cell === MULTIBOUNCE_MISS_CELL) {
            cacheEntries++;
            continue;
        }
        const base = entry * MULTIBOUNCE_RAY_WORDS;
        let second: Vec3;
        if ((multibounceRays[base + 3] & WORLD_CACHE_PROBE_RAY_BYPASS) !== 0) {
            second = [multibounceDirect[entry * 4], multibounceDirect[entry * 4 + 1], multibounceDirect[entry * 4 + 2]];
        } else {
            const value = filteredDirect(direct, cache, slotOfTile.get(tileOf(cell)) ?? 0, cell, settings.hashGridCacheMaxSampleCount);
            // E8: a landing cell no level of which has been lit holds no estimate and adds no sample.
            if (budgeted && value[3] <= 0) continue;
            second = [value[0] / Math.max(value[3], 1), value[1] / Math.max(value[3], 1), value[2] / Math.max(value[3], 1)];
        }
        const info = multibounceInfo.subarray(entry * 4, entry * 4 + 4);
        const sample = Math.max(...multibounceSample(second, [info[0], info[1], info[2]], info[3]));
        cacheSum += sample;
        cacheEntries++;
        const direction = unpackHalf3([multibounceRays[base + 4], multibounceRays[base + 5]]);
        const t = multibounceRayFloats[base + 6];
        const hit: Vec3 = [multibounceRayFloats[base] + t * direction[0], multibounceRayFloats[base + 1] + t * direction[1], multibounceRayFloats[base + 2] + t * direction[2]];
        if (Math.abs(hit[1]) > 0.01 || !inNearTile(hit)) continue;
        landingSum += sample;
        landingEntries++;
    }
    return {
        nearFloorLandings: { mean: landingSum / Math.max(landingEntries, 1), entries: landingEntries },
        throughCache: { mean: cacheSum / Math.max(cacheEntries, 1), entries: cacheEntries },
        nearCeilingCells: nearCeiling.size,
    };
}
