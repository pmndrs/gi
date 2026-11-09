/**
 * Capsaicin's hash-grid radiance cache behind the {@link WorldCache} slot
 * (docs/design/radiance-cache.md, "World cache as a slot").
 *
 * RC3.1 builds the grid itself: the buffers at Capsaicin's default capacity
 * (`capacity.ts`), the shared WGSL every hash-grid pass includes
 * (`hash-grid-wgsl.ts`: descriptor, insert, find, decay stamps, cell
 * indices, value packing), and the per-frame maintenance Capsaicin runs
 * before the screen probes (`GI1::render`, gi1.cpp:2238-2253): the purge's
 * dispatch arguments from last frame's tile count, `ClearCounters` and
 * `PurgeTiles`. Population, update and resolve (the rest of `encodeFrame`)
 * are RC3.2.
 */

import type { GpuIndirectArgs, GpuWord } from '../../../platform/contract/compute.ts';
import type { ComputeKernel } from '../../../platform/contract/wgsl.ts';
import {
    checkHashGridCacheLimits,
    DEFAULT_HASH_GRID_CACHE_OPTIONS,
    HASH_GRID_COUNTER,
    hashGridCacheBufferLayout,
    hashGridCacheCapacity,
    type HashGridCacheBufferLayout,
    type HashGridCacheCapacity,
    type HashGridCacheOptions,
} from './capacity.ts';
import { hashGridCacheCellSizeConstant, verticalFieldOfView } from './hash-grid-cpu.ts';
import {
    clearCountersWgsl,
    HASH_GRID_CACHE_BINDING,
    HASH_GRID_CACHE_GROUP_ENTRIES,
    HASH_GRID_CONSTANTS_BYTES,
    HASH_GRID_CONSTANTS_WORD,
    HASH_GRID_ENTRY_POINT,
    hashGridCacheWgsl,
    PURGE_TILES_WORKGROUP_SIZE,
    purgeTilesWgsl,
} from './hash-grid-wgsl.ts';
import type { WorldCache, WorldCacheContext, WorldCacheFrame, WorldCacheLookup, WorldCacheProbeRays } from './world-cache.ts';

/** `GPUBufferUsage` bits (the global doesn't exist in Node, where the tests import this file). */
const BUFFER_USAGE = { copySrc: 0x4, copyDst: 0x8, uniform: 0x40, storage: 0x80 } as const;

const WORD_BYTES = 4;

/** The three packed buffers and the constant block. */
export interface HashGridCacheBuffers {
    /** Counters, then per tile: hash slot, decay stamp, packed tile lists 0 and 1 (`capacity.ts`). */
    readonly tileWords: GPUBuffer;
    /** Direct, then indirect values: `vec2u` per cell. */
    readonly cellValues: GPUBuffer;
    /** Direct, then indirect update scratch: four words per cell. */
    readonly updateScratch: GPUBuffer;
    /** `HashGridCacheConstants`, written every frame. */
    readonly constants: GPUBuffer;
}

/** What `prepare` makes. */
interface Prepared {
    readonly device: GPUDevice;
    readonly capacity: HashGridCacheCapacity;
    readonly layout: HashGridCacheBufferLayout;
    readonly buffers: HashGridCacheBuffers;
    readonly clearCounters: ComputeKernel;
    readonly purgeTiles: ComputeKernel;
    readonly clearCountersGroup: GPUBindGroup;
    readonly purgeTilesGroup: GPUBindGroup;
    /** PurgeTiles' dispatch arguments for each value of `buffer_ping_pong`: each reads the other parity's count, last frame's. */
    readonly purgeArgs: readonly [GpuIndirectArgs, GpuIndirectArgs];
    readonly memoryBytes: number;
}

export class HashGridCache implements WorldCache {
    readonly id = 'hash-grid';
    /** Live: the capacity fields take effect at the next `prepare`, the rest every frame. */
    readonly options: HashGridCacheOptions;

    private prepared: Prepared | null = null;
    /** Capsaicin's `radiance_cache_hash_buffer_ping_pong_` (gi1.h:272): which packed list the next frame fills. */
    private bufferPingPong = 0;
    /** The value the last `encodeFrame` used, or null before the first. */
    private lastFramePingPong: number | null = null;

    constructor(options: HashGridCacheOptions = DEFAULT_HASH_GRID_CACHE_OPTIONS) {
        this.options = options;
    }

    /** The capacity and the buffer layout, once prepared. */
    get capacity(): HashGridCacheCapacity {
        return this.require().capacity;
    }

    get layout(): HashGridCacheBufferLayout {
        return this.require().layout;
    }

    get buffers(): HashGridCacheBuffers {
        return this.require().buffers;
    }

    /** Inserts refused this frame because their bucket was full (register D21): a word of the tile words, cleared by `ClearCounters`. */
    get bucketOverflowCount(): GpuWord {
        return { buffer: this.require().buffers.tileWords, offset: HASH_GRID_COUNTER.bucketOverflowCount * WORD_BYTES };
    }

    /**
     * The packed tile list of the frame `encodeFrame` last recorded (the
     * tiles alive after its purge, plus those its passes add): its count, and
     * the word of the tile words where its tile indices start.
     */
    get packedTiles(): { readonly count: GpuWord; readonly firstWord: number } {
        const { buffers, layout } = this.require();
        if (this.lastFramePingPong === null) throw new Error('@pmndrs/ddgi: the world cache has no packed tile list before its first encodeFrame');
        const first = this.lastFramePingPong === 0;
        const countWord = first ? HASH_GRID_COUNTER.packedTileCount0 : HASH_GRID_COUNTER.packedTileCount1;
        return {
            count: { buffer: buffers.tileWords, offset: countWord * WORD_BYTES },
            firstWord: first ? layout.tileWords.packedTileIndexBuffer0 : layout.tileWords.packedTileIndexBuffer1,
        };
    }

    get memoryBytes(): number {
        return this.prepared?.memoryBytes ?? 0;
    }

    get stats(): Readonly<Record<string, number>> {
        const prepared = this.prepared;
        if (!prepared) return {};
        const { capacity } = prepared;
        return {
            buckets: capacity.numBuckets,
            'tile slots per bucket': capacity.numTilesPerBucket,
            tiles: capacity.numTiles,
            'cells per tile': capacity.numCellsPerTile,
            cells: capacity.numCells,
            'GPU memory (bytes)': prepared.memoryBytes,
        };
    }

    /**
     * The shared WGSL with the hash grid's bindings in `group`: every
     * hash-grid pass (and RC3.2's) includes it. Bind {@link createBindGroup}'s
     * group there, with a layout made from {@link HASH_GRID_CACHE_GROUP_ENTRIES}.
     */
    wgsl(group: number): string {
        return hashGridCacheWgsl(group, this.require().layout);
    }

    /** A bind group of the hash grid's buffers for a layout made from {@link HASH_GRID_CACHE_GROUP_ENTRIES}. */
    createBindGroup(layout: GPUBindGroupLayout, label = 'radiance-cache/hash-grid'): GPUBindGroup {
        const { device, buffers } = this.require();
        return device.createBindGroup({ label, layout, entries: groupEntries(buffers) });
    }

    readonly lookup: WorldCacheLookup = {
        code: (group: number) => this.wgsl(group),
        entries: HASH_GRID_CACHE_GROUP_ENTRIES,
        createBindGroup: (layout: GPUBindGroupLayout) => this.createBindGroup(layout, 'radiance-cache/hash-grid lookup'),
    };

    async prepare(context: WorldCacheContext): Promise<void> {
        const { device, kernels, compute } = context;
        const capacity = hashGridCacheCapacity(this.options);
        const layout = hashGridCacheBufferLayout(capacity);
        // Before anything is made: a device that can't hold the cache gets the limit's name (register D14).
        checkHashGridCacheLimits(layout, device.limits);
        this.dispose();

        // Made zeroed (WebGPU clears new buffers), which is the empty cache: every hash slot 0
        // (Capsaicin clears the hash buffer and the update scratch at creation, gi1.cpp:478, 530).
        const storageUsage = BUFFER_USAGE.storage | BUFFER_USAGE.copySrc | BUFFER_USAGE.copyDst;
        device.pushErrorScope('out-of-memory');
        const buffers: HashGridCacheBuffers = {
            tileWords: device.createBuffer({ label: 'radiance-cache/hash-grid tile words', size: layout.tileWords.bytes, usage: storageUsage }),
            cellValues: device.createBuffer({ label: 'radiance-cache/hash-grid cell values', size: layout.cellValues.bytes, usage: storageUsage }),
            updateScratch: device.createBuffer({ label: 'radiance-cache/hash-grid update scratch', size: layout.updateScratch.bytes, usage: storageUsage }),
            constants: device.createBuffer({ label: 'radiance-cache/hash-grid constants', size: HASH_GRID_CONSTANTS_BYTES, usage: BUFFER_USAGE.uniform | BUFFER_USAGE.copyDst }),
        };
        const outOfMemory = await device.popErrorScope();
        if (outOfMemory) {
            for (const buffer of Object.values(buffers)) buffer.destroy();
            throw new Error(`@pmndrs/ddgi: the GPU ran out of memory making the world cache's ${layout.totalBytes} bytes: ${outOfMemory.message}`);
        }

        const ownGroups = [{ label: 'radiance-cache/hash-grid group', entries: HASH_GRID_CACHE_GROUP_ENTRIES }];
        const [clearCounters, purgeTiles, purgeArgs0, purgeArgs1] = await Promise.all([
            kernels.computeKernel({
                label: 'radiance-cache/ClearCounters (hash grid)',
                modules: [],
                code: clearCountersWgsl(layout),
                entryPoint: HASH_GRID_ENTRY_POINT.clearCounters,
                ownGroups,
            }),
            kernels.computeKernel({
                label: 'radiance-cache/PurgeTiles',
                modules: [compute.indirectModule],
                code: purgeTilesWgsl(layout),
                entryPoint: HASH_GRID_ENTRY_POINT.purgeTiles,
                ownGroups,
            }),
            // With buffer_ping_pong 0, last frame's count is PackedTileCountBuffer1, and the other
            // way round (hash_grid_cache.hlsl:75-76; gi1.cpp:2241-2244).
            compute.createIndirectArgs({
                label: 'radiance-cache/PurgeTiles (ping-pong 0)',
                count: { buffer: buffers.tileWords, offset: HASH_GRID_COUNTER.packedTileCount1 * WORD_BYTES },
                itemsPerWorkgroup: PURGE_TILES_WORKGROUP_SIZE,
            }),
            compute.createIndirectArgs({
                label: 'radiance-cache/PurgeTiles (ping-pong 1)',
                count: { buffer: buffers.tileWords, offset: HASH_GRID_COUNTER.packedTileCount0 * WORD_BYTES },
                itemsPerWorkgroup: PURGE_TILES_WORKGROUP_SIZE,
            }),
        ]);

        const purgeArgs = [purgeArgs0, purgeArgs1] as const;
        const memoryBytes = layout.totalBytes + HASH_GRID_CONSTANTS_BYTES + purgeArgs0.memoryBytes + purgeArgs1.memoryBytes;
        this.prepared = {
            device,
            capacity,
            layout,
            buffers,
            clearCounters,
            purgeTiles,
            clearCountersGroup: device.createBindGroup({ label: 'radiance-cache/ClearCounters (hash grid)', layout: clearCounters.ownLayouts[0], entries: groupEntries(buffers) }),
            purgeTilesGroup: device.createBindGroup({ label: 'radiance-cache/PurgeTiles', layout: purgeTiles.ownLayouts[0], entries: groupEntries(buffers) }),
            purgeArgs,
            memoryBytes,
        };
        this.bufferPingPong = 0;
        this.lastFramePingPong = null;
    }

    /**
     * Writes the frame's constants and records the cache's per-frame work.
     * RC3.1: the maintenance Capsaicin runs before the screen probes
     * (gi1.cpp:2238-2253), in its order: the purge's dispatch arguments from
     * last frame's packed tile count, `ClearCounters`, then `PurgeTiles`.
     * RC3.2 adds population, update and resolve into `rays`.
     */
    encodeFrame(encoder: GPUCommandEncoder, frame: WorldCacheFrame, rays: WorldCacheProbeRays): void {
        void rays; // RC3.2: population and resolve read and write them.
        const prepared = this.require();
        this.writeConstants(prepared, frame);

        const pass = encoder.beginComputePass({ label: 'radiance-cache/PurgeRadianceCache' });
        prepared.purgeArgs[this.bufferPingPong].encode(pass);
        prepared.clearCounters.bind(pass);
        pass.setBindGroup(1, prepared.clearCountersGroup);
        pass.dispatchWorkgroups(1);
        prepared.purgeTiles.bind(pass);
        pass.setBindGroup(1, prepared.purgeTilesGroup);
        pass.dispatchWorkgroupsIndirect(prepared.purgeArgs[this.bufferPingPong].buffer, 0);
        pass.end();

        this.lastFramePingPong = this.bufferPingPong;
        // Flip the buffers (gi1.cpp:3097-3099). The constants already on the queue keep this
        // frame's value for every pass recorded after this call in the same submit.
        this.bufferPingPong = 1 - this.bufferPingPong;
    }

    dispose(): void {
        const prepared = this.prepared;
        if (!prepared) return;
        for (const buffer of Object.values(prepared.buffers)) buffer.destroy();
        for (const args of prepared.purgeArgs) args.dispose();
        this.prepared = null;
    }

    /** `HashGridCacheConstants` for this frame (gi1.cpp:1852-1893). */
    private writeConstants(prepared: Prepared, frame: WorldCacheFrame): void {
        const { capacity } = prepared;
        const options = this.options;
        const bytes = new ArrayBuffer(HASH_GRID_CONSTANTS_BYTES);
        const floats = new Float32Array(bytes);
        const words = new Uint32Array(bytes);
        const word = HASH_GRID_CONSTANTS_WORD;
        const cellSize = hashGridCacheCellSizeConstant(verticalFieldOfView(frame.camera.projection), frame.width, frame.height, options.hashGridCacheCellSize);

        floats.set(frame.camera.position, word.eyePosition);
        // g_FrameIndex is a uint32 that wraps; PurgeTiles handles the wrap (gi1.comp:1729-1736).
        words[word.frameIndex] = frame.frame >>> 0;
        floats[word.cellSize] = cellSize;
        floats[word.minCellSize] = options.hashGridCacheMinCellSize;
        floats[word.tileSize] = cellSize * options.hashGridCacheTileCellRatio;
        floats[word.tileCellRatio] = options.hashGridCacheTileCellRatio;
        words[word.numBuckets] = capacity.numBuckets;
        words[word.numTiles] = capacity.numTiles;
        words[word.numCells] = capacity.numCells;
        words[word.numTilesPerBucket] = capacity.numTilesPerBucket;
        words.set(capacity.sizeTileMip, word.sizeTileMip0);
        words.set(capacity.numCellsPerTileMip, word.numCellsPerTileMip0);
        words[word.numCellsPerTile] = capacity.numCellsPerTile;
        words.set(capacity.firstCellOffsetTileMip, word.firstCellOffsetTileMip0);
        words[word.bufferPingPong] = this.bufferPingPong;
        floats[word.maxSampleCount] = options.hashGridCacheMaxSampleCount;
        floats[word.discardMultibounceRayProbability] = options.hashGridCacheDiscardMultibounceRayProbability;
        floats[word.maxMultibounceSampleCount] = options.hashGridCacheMaxMultibounceSampleCount;
        prepared.device.queue.writeBuffer(prepared.buffers.constants, 0, bytes);
    }

    private require(): Prepared {
        if (!this.prepared) throw new Error('@pmndrs/ddgi: the world cache is used before prepare()');
        return this.prepared;
    }
}

function groupEntries(buffers: HashGridCacheBuffers): GPUBindGroupEntry[] {
    return [
        { binding: HASH_GRID_CACHE_BINDING.constants, resource: { buffer: buffers.constants } },
        { binding: HASH_GRID_CACHE_BINDING.tileWords, resource: { buffer: buffers.tileWords } },
        { binding: HASH_GRID_CACHE_BINDING.cellValues, resource: { buffer: buffers.cellValues } },
        { binding: HASH_GRID_CACHE_BINDING.updateScratch, resource: { buffer: buffers.updateScratch } },
    ];
}
