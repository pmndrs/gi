/**
 * Capsaicin's hash-grid radiance cache behind the {@link WorldCache} slot
 * (docs/design/radiance-cache.md, "World cache as a slot").
 *
 * The grid itself (RC3.1): the buffers at Capsaicin's default capacity
 * (`capacity.ts`), the shared WGSL every hash-grid pass includes
 * (`hash-grid-wgsl.ts`: descriptor, insert, find, decay stamps, cell
 * indices, value packing).
 *
 * Its per-frame work (`encodeFrame`), in Capsaicin's order (`GI1::render`,
 * gi1.cpp:2238-2253 before the screen probes, 2335-2528 after them):
 *
 * 1. `PurgeRadianceCache`: the purge's dispatch arguments from last frame's
 *    tile count, `ClearCounters` and `PurgeTiles` (RC3.1);
 * 2. `PopulateScreenProbesCells`: the cache half of `PopulateScreenProbes`,
 *    over the probe rays the screen cache traced (RC3.2);
 * 3. `GenerateReservoirs`, `PopulateRadianceCache` (`PopulateCells`),
 *    `UpdateRadianceCache` (`GenerateUpdateTilesDispatch`, `UpdateTiles`) and
 *    `ResolveCells` (RC3.2; `hash-grid-population-wgsl.ts`).
 *
 * Every pass after the purge is sized on the GPU, through the platform's
 * indirect-dispatch arguments, where Capsaicin dispatches indirectly
 * (`generateDispatch`).
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
    generateReservoirsWgsl,
    hashGridCacheRayListLayout,
    POPULATION_ENTRY_POINT,
    POPULATION_WORKGROUP_SIZE,
    populateCellsWgsl,
    populateScreenProbesWgsl,
    RAY_DATA_BINDING,
    RAY_DATA_GROUP,
    RAY_DATA_GROUP_ENTRIES,
    RAY_LISTS_UNIFORM_BYTES,
    resolveCellsWgsl,
    updateTilesPerWorkgroup,
    updateTilesWgsl,
    type HashGridCacheRayListLayout,
} from './hash-grid-population-wgsl.ts';
import {
    clearCountersWgsl,
    HASH_GRID_CACHE_BINDING,
    HASH_GRID_CACHE_GROUP_ENTRIES,
    HASH_GRID_CONSTANTS_BYTES,
    HASH_GRID_CONSTANTS_WORD,
    HASH_GRID_ENTRY_POINT,
    HASH_GRID_KERNEL_GROUP,
    hashGridCacheWgsl,
    PURGE_TILES_WORKGROUP_SIZE,
    purgeTilesWgsl,
} from './hash-grid-wgsl.ts';
import type { WorldCache, WorldCacheContext, WorldCacheFrame, WorldCacheLookup, WorldCacheProbeRays } from './world-cache.ts';

/** `GPUBufferUsage` bits (the global doesn't exist in Node, where the tests import this file). */
const BUFFER_USAGE = { copySrc: 0x4, copyDst: 0x8, uniform: 0x40, storage: 0x80 } as const;

const WORD_BYTES = 4;

/** Timed as `radiance-cache/<section>`, Capsaicin's `TimedSection` names. */
const TIMER_PREFIX = 'radiance-cache';

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

/** The per-ray lists of the latest frame, for tests and debugging: the packed words and where each list starts. */
export interface HashGridCacheRayLists {
    readonly buffer: GPUBuffer;
    readonly layout: HashGridCacheRayListLayout;
}

/** The population and resolve passes: a kernel and its indirect arguments each. */
interface PopulationPass {
    readonly kernel: ComputeKernel;
    readonly args: GpuIndirectArgs;
    readonly cacheGroup: GPUBindGroup;
}

interface PopulationPasses {
    readonly populateScreenProbes: PopulationPass;
    readonly generateReservoirs: PopulationPass;
    readonly populateCells: PopulationPass;
    readonly updateTiles: PopulationPass;
    readonly resolveCells: PopulationPass;
}

type PopulationPassName = keyof PopulationPasses;

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
    readonly population: PopulationPasses;
    /** `HashGridCacheRayLists`: where each per-ray list starts. */
    readonly rayListsUniform: GPUBuffer;
    readonly timer: WorldCacheContext['timer'];
    readonly memoryBytes: number;
}

/** The per-ray lists for one screen's probe rays, and the ray-data bind groups over them. */
interface RayData {
    readonly rays: WorldCacheProbeRays;
    readonly layout: HashGridCacheRayListLayout;
    readonly listWords: GPUBuffer;
    readonly groups: Readonly<Record<PopulationPassName, GPUBindGroup>>;
}

export class HashGridCache implements WorldCache {
    readonly id = 'hash-grid';
    /** Live: the capacity fields take effect at the next `prepare`, the rest every frame. */
    readonly options: HashGridCacheOptions;

    private prepared: Prepared | null = null;
    private rayData: RayData | null = null;
    /** Capsaicin's `radiance_cache_hash_buffer_ping_pong_` (gi1.h:272): which packed list the next frame fills. */
    private bufferPingPong = 0;
    /** The value the last `encodeFrame` used, or null before the first. */
    private lastFramePingPong: number | null = null;

    /** `options` is kept and read live (the technique passes its settings); without it, a copy of Capsaicin's defaults. */
    constructor(options: HashGridCacheOptions = { ...DEFAULT_HASH_GRID_CACHE_OPTIONS }) {
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

    /** The per-ray lists (visibility, resolve, update tiles, reservoirs), or null before the first `encodeFrame`. */
    get rayLists(): HashGridCacheRayLists | null {
        const rayData = this.rayData;
        return rayData ? { buffer: rayData.listWords, layout: rayData.layout } : null;
    }

    /** Inserts refused this frame because their bucket was full (register D23): a word of the tile words, cleared by `ClearCounters`. */
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

    /** GPU bytes: the cache's buffers at its capacity, plus the per-ray lists of the latest screen. */
    get memoryBytes(): number {
        return (this.prepared?.memoryBytes ?? 0) + (this.rayData?.listWords.size ?? 0);
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
            'per-ray lists (bytes)': this.rayData?.listWords.size ?? 0,
            'GPU memory (bytes)': this.memoryBytes,
        };
    }

    /**
     * The shared WGSL with the hash grid's bindings in `group`: every
     * hash-grid pass includes it. Bind {@link createBindGroup}'s group there,
     * with a layout made from {@link HASH_GRID_CACHE_GROUP_ENTRIES}.
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
        const { device, kernels, compute, shade } = context;
        const capacity = hashGridCacheCapacity(this.options);
        const layout = hashGridCacheBufferLayout(capacity);
        // Before anything is made: a device that can't hold the cache gets the limit's name (register D14).
        checkHashGridCacheLimits(layout, device.limits);
        const tilesPerUpdateGroup = updateTilesPerWorkgroup(this.options.hashGridCacheTileCellRatio);
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

        const cacheGroup = { label: 'radiance-cache/hash-grid group', entries: HASH_GRID_CACHE_GROUP_ENTRIES };
        const rayDataGroup = { label: 'radiance-cache/hash-grid ray data', entries: RAY_DATA_GROUP_ENTRIES };
        const word = (counter: number): GpuWord => ({ buffer: buffers.tileWords, offset: counter * WORD_BYTES });
        /** A population kernel: the hash grid in group 1, the ray data in group 2, the platform's hit shading in group 0 when it shades. */
        const populationKernel = (label: string, code: string, entryPoint: string, shades: boolean): Promise<ComputeKernel> =>
            kernels.computeKernel({ label: `radiance-cache/${label}`, modules: shades ? [shade.module] : [], code, entryPoint, ownGroups: [cacheGroup, rayDataGroup] });
        const indirectArgs = (label: string, counter: number, itemsPerWorkgroup: number): Promise<GpuIndirectArgs> =>
            compute.createIndirectArgs({ label: `radiance-cache/${label}`, count: word(counter), itemsPerWorkgroup });

        const [clearCounters, purgeTiles, purgeArgs0, purgeArgs1, ...population] = await Promise.all([
            kernels.computeKernel({
                label: 'radiance-cache/ClearCounters (hash grid)',
                modules: [],
                code: clearCountersWgsl(layout),
                entryPoint: HASH_GRID_ENTRY_POINT.clearCounters,
                ownGroups: [cacheGroup],
            }),
            kernels.computeKernel({
                label: 'radiance-cache/PurgeTiles',
                modules: [compute.indirectModule],
                code: purgeTilesWgsl(layout),
                entryPoint: HASH_GRID_ENTRY_POINT.purgeTiles,
                ownGroups: [cacheGroup],
            }),
            // With buffer_ping_pong 0, last frame's count is PackedTileCountBuffer1, and the other
            // way round (hash_grid_cache.hlsl:75-76; gi1.cpp:2241-2244).
            indirectArgs('PurgeTiles (ping-pong 0)', HASH_GRID_COUNTER.packedTileCount1, PURGE_TILES_WORKGROUP_SIZE),
            indirectArgs('PurgeTiles (ping-pong 1)', HASH_GRID_COUNTER.packedTileCount0, PURGE_TILES_WORKGROUP_SIZE),
            populationKernel('PopulateScreenProbes (cache half)', populateScreenProbesWgsl(layout), POPULATION_ENTRY_POINT.populateScreenProbes, false),
            populationKernel('GenerateReservoirs', generateReservoirsWgsl(layout), POPULATION_ENTRY_POINT.generateReservoirs, true),
            populationKernel('PopulateCells', populateCellsWgsl(layout), POPULATION_ENTRY_POINT.populateCells, true),
            populationKernel('UpdateTiles', updateTilesWgsl(layout), POPULATION_ENTRY_POINT.updateTiles, false),
            populationKernel('ResolveCells', resolveCellsWgsl(layout), POPULATION_ENTRY_POINT.resolveCells, false),
            // Capsaicin dispatches PopulateScreenProbes over every possible ray; here it is its own
            // pass after the screen cache's trace, so it runs on the probe-ray count.
            indirectArgs('PopulateScreenProbes (cache half) args', HASH_GRID_COUNTER.probeRayCount, POPULATION_WORKGROUP_SIZE.populateScreenProbes),
            // generateDispatch(radiance_cache_visibility_count_buffer0_, ...) (gi1.cpp:2423-2426)
            indirectArgs('GenerateReservoirs args', HASH_GRID_COUNTER.visibilityCount0, POPULATION_WORKGROUP_SIZE.generateReservoirs),
            // generateDispatch(radiance_cache_visibility_ray_count_buffer_, ...) (gi1.cpp:2494-2499)
            indirectArgs('PopulateCells args', HASH_GRID_COUNTER.visibilityRayCount, POPULATION_WORKGROUP_SIZE.populateCells),
            // GenerateUpdateTilesDispatch (gi1.comp:2129-2140): ceil(update tiles / num_tiles_by_group) groups.
            indirectArgs('GenerateUpdateTilesDispatch', HASH_GRID_COUNTER.updateTileCount, tilesPerUpdateGroup),
            // generateDispatch(radiance_cache_resolve_count_buffer_, ...) (gi1.cpp:2523-2527)
            indirectArgs('ResolveCells args', HASH_GRID_COUNTER.resolveCount, POPULATION_WORKGROUP_SIZE.resolveCells),
        ]);
        const [populateScreenProbes, generateReservoirs, populateCells, updateTiles, resolveCells] = population.slice(0, 5) as ComputeKernel[];
        const [populateArgs, reservoirArgs, cellArgs, updateArgs, resolveArgs] = population.slice(5) as GpuIndirectArgs[];
        const pass = (kernel: ComputeKernel, args: GpuIndirectArgs): PopulationPass => ({
            kernel,
            args,
            cacheGroup: device.createBindGroup({ label: `${kernel.pipeline.label} hash grid`, layout: kernel.ownLayouts[0], entries: groupEntries(buffers) }),
        });
        const populationPasses: PopulationPasses = {
            populateScreenProbes: pass(populateScreenProbes, populateArgs),
            generateReservoirs: pass(generateReservoirs, reservoirArgs),
            populateCells: pass(populateCells, cellArgs),
            updateTiles: pass(updateTiles, updateArgs),
            resolveCells: pass(resolveCells, resolveArgs),
        };
        const rayListsUniform = device.createBuffer({ label: 'radiance-cache/hash-grid ray lists', size: RAY_LISTS_UNIFORM_BYTES, usage: BUFFER_USAGE.uniform | BUFFER_USAGE.copyDst });

        const purgeArgs = [purgeArgs0, purgeArgs1] as const;
        const argsBytes = [...purgeArgs, populateArgs, reservoirArgs, cellArgs, updateArgs, resolveArgs].reduce((sum, args) => sum + args.memoryBytes, 0);
        const memoryBytes = layout.totalBytes + HASH_GRID_CONSTANTS_BYTES + RAY_LISTS_UNIFORM_BYTES + argsBytes;
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
            population: populationPasses,
            rayListsUniform,
            timer: context.timer,
            memoryBytes,
        };
        this.bufferPingPong = 0;
        this.lastFramePingPong = null;
    }

    /**
     * Empties the cache: the tile words (every hash slot, decay stamp, both
     * packed lists and the counters) back to zero, as `prepare` leaves them
     * (Capsaicin clears the hash buffer when it makes it, gi1.cpp:478). The
     * cell values and scratch stay: a tile's finest cells are cleared when it
     * is inserted again (`HashGridCache_AddNewTile`), its coarser levels are
     * rewritten in the same frame (`UpdateTiles`), and the scratch is zero
     * between frames (`UpdateTiles` clears what it reads, gi1.comp:2213-2224).
     */
    reset(encoder: GPUCommandEncoder): void {
        encoder.clearBuffer(this.require().buffers.tileWords);
    }

    /**
     * Writes the frame's constants and records the cache's work for the
     * frame, in Capsaicin's order (see the file's comment): purge, insert the
     * probe rays' hits, light one sample per hit, update the tiles, resolve
     * into `rays.radiance`.
     */
    encodeFrame(encoder: GPUCommandEncoder, frame: WorldCacheFrame, rays: WorldCacheProbeRays): void {
        const prepared = this.require();
        const rayData = this.rayDataFor(prepared, rays);
        this.writeConstants(prepared, frame);

        // The probe-ray count where the insert pass's indirect arguments read it.
        encoder.copyBufferToBuffer(rays.count.buffer, rays.count.offset, prepared.buffers.tileWords, HASH_GRID_COUNTER.probeRayCount * WORD_BYTES, WORD_BYTES);

        const purge = this.beginPass(encoder, prepared, 'PurgeRadianceCache');
        prepared.purgeArgs[this.bufferPingPong].encode(purge);
        prepared.clearCounters.bind(purge);
        purge.setBindGroup(HASH_GRID_KERNEL_GROUP, prepared.clearCountersGroup);
        purge.dispatchWorkgroups(1);
        prepared.purgeTiles.bind(purge);
        purge.setBindGroup(HASH_GRID_KERNEL_GROUP, prepared.purgeTilesGroup);
        purge.dispatchWorkgroupsIndirect(prepared.purgeArgs[this.bufferPingPong].buffer, 0);
        purge.end();

        this.encodePopulationPass(encoder, prepared, rayData, 'populateScreenProbes', 'PopulateScreenProbesCells');
        this.encodePopulationPass(encoder, prepared, rayData, 'generateReservoirs', 'GenerateReservoirs');
        this.encodePopulationPass(encoder, prepared, rayData, 'populateCells', 'PopulateRadianceCache');
        this.encodePopulationPass(encoder, prepared, rayData, 'updateTiles', 'UpdateRadianceCache');
        this.encodePopulationPass(encoder, prepared, rayData, 'resolveCells', 'ResolveCells');

        this.lastFramePingPong = this.bufferPingPong;
        // Flip the buffers (gi1.cpp:3097-3099). The constants already on the queue keep this
        // frame's value for every pass recorded after this call in the same submit.
        this.bufferPingPong = 1 - this.bufferPingPong;
    }

    dispose(): void {
        this.releaseRayData();
        const prepared = this.prepared;
        if (!prepared) return;
        for (const buffer of Object.values(prepared.buffers)) buffer.destroy();
        prepared.rayListsUniform.destroy();
        for (const args of prepared.purgeArgs) args.dispose();
        for (const populationPass of Object.values(prepared.population)) populationPass.args.dispose();
        this.prepared = null;
    }

    /** One of Capsaicin's timed sections as a compute pass: its indirect arguments from the GPU count, then the kernel. */
    private encodePopulationPass(encoder: GPUCommandEncoder, prepared: Prepared, rayData: RayData, name: PopulationPassName, section: string): void {
        const populationPass = prepared.population[name];
        const pass = this.beginPass(encoder, prepared, section);
        populationPass.args.encode(pass);
        populationPass.kernel.bind(pass);
        pass.setBindGroup(HASH_GRID_KERNEL_GROUP, populationPass.cacheGroup);
        pass.setBindGroup(RAY_DATA_GROUP, rayData.groups[name]);
        pass.dispatchWorkgroupsIndirect(populationPass.args.buffer, 0);
        pass.end();
    }

    private beginPass(encoder: GPUCommandEncoder, prepared: Prepared, section: string): GPUComputePassEncoder {
        const name = `${TIMER_PREFIX}/${section}`;
        return encoder.beginComputePass({ label: name, timestampWrites: prepared.timer.computePass(name) });
    }

    /**
     * The per-ray lists for these probe rays, made when the screen's ray
     * capacity or the buffers handed over change (Capsaicin's
     * `ensureMemoryIsAllocated`, gi1.cpp:637-695): buffers and bind groups,
     * never pipelines.
     */
    private rayDataFor(prepared: Prepared, rays: WorldCacheProbeRays): RayData {
        const current = this.rayData;
        if (current && sameRays(current.rays, rays)) return current;
        this.releaseRayData();
        const { device } = prepared;
        const layout = hashGridCacheRayListLayout(rays.capacity, prepared.capacity.numCells);
        const listWords = device.createBuffer({ label: 'radiance-cache/hash-grid per-ray lists', size: layout.bytes, usage: BUFFER_USAGE.storage | BUFFER_USAGE.copySrc });
        device.queue.writeBuffer(
            prepared.rayListsUniform,
            0,
            new Uint32Array([
                layout.maxRayCount,
                layout.maxCombinedRayCount,
                layout.updateTileBuffer,
                layout.visibilityCellBuffer,
                layout.visibilityQueryBuffer,
                layout.visibilityRayBuffer,
                layout.resolveBuffer,
                layout.reservoirBuffer,
            ]),
        );
        const entries: GPUBindGroupEntry[] = [
            { binding: RAY_DATA_BINDING.rayLists, resource: { buffer: prepared.rayListsUniform } },
            { binding: RAY_DATA_BINDING.probeRays, resource: { buffer: rays.buffer } },
            { binding: RAY_DATA_BINDING.radiance, resource: rays.radiance },
            { binding: RAY_DATA_BINDING.listWords, resource: { buffer: listWords } },
        ];
        const group = (name: PopulationPassName): GPUBindGroup =>
            device.createBindGroup({ label: `${prepared.population[name].kernel.pipeline.label} ray data`, layout: prepared.population[name].kernel.ownLayouts[1], entries });
        const rayData: RayData = {
            rays,
            layout,
            listWords,
            groups: {
                populateScreenProbes: group('populateScreenProbes'),
                generateReservoirs: group('generateReservoirs'),
                populateCells: group('populateCells'),
                updateTiles: group('updateTiles'),
                resolveCells: group('resolveCells'),
            },
        };
        this.rayData = rayData;
        return rayData;
    }

    private releaseRayData(): void {
        this.rayData?.listWords.destroy();
        this.rayData = null;
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
        words[word.useDirectLighting] = options.useDirectLighting ? 1 : 0;
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

/** The same probe-ray buffers as before (a new screen size or new buffers need new lists and groups). */
function sameRays(a: WorldCacheProbeRays, b: WorldCacheProbeRays): boolean {
    return (
        a.buffer === b.buffer &&
        a.capacity === b.capacity &&
        a.radiance.buffer === b.radiance.buffer &&
        (a.radiance.offset ?? 0) === (b.radiance.offset ?? 0) &&
        a.radiance.size === b.radiance.size
    );
}
