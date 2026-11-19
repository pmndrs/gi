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
 * 3. GI-1.2's `PopulateMultibounceCells` (RC3.3), with multibounce on;
 * 4. `GenerateReservoirs`, then `GenerateMultibounceReservoirs` with
 *    multibounce, `PopulateRadianceCache` (`PopulateCells`),
 *    `UpdateRadianceCache` (`GenerateUpdateTilesDispatch`, `UpdateTiles`),
 *    `UpdateMultibounceCells` with multibounce, and `ResolveCells` (RC3.2,
 *    RC3.3; `hash-grid-population-wgsl.ts`).
 *
 * Every pass after the purge is sized on the GPU, through the platform's
 * indirect-dispatch arguments, where Capsaicin dispatches indirectly
 * (`generateDispatch`).
 *
 * Lever E8 (`cellShading: 'budgeted'`, TASKS P6.2): the hits only find their
 * cells, and `ClaimCells`, `PrioritizeCells` and `ShadeCells`
 * (`cell-shading-wgsl.ts`) light a budget of cells between `PopulateCells`
 * and `UpdateTiles`.
 */

import type { GpuIndirectArgs, GpuWord } from '../../../platform/contract/compute.ts';
import type { ComputeKernel } from '../../../platform/contract/wgsl.ts';
import {
    CELL_SHADING_BINS,
    checkHashGridCacheLimits,
    DEFAULT_HASH_GRID_CACHE_OPTIONS,
    HASH_GRID_COUNTER,
    hashGridCacheBufferLayout,
    hashGridCacheCapacity,
    type HashGridCacheBufferLayout,
    type HashGridCacheCapacity,
    type HashGridCacheOptions,
} from './capacity.ts';
import {
    CELL_SHADING_CHANGE_FRAMES,
    CELL_SHADING_ENTRY_POINT,
    CELL_SHADING_GROUP_ENTRIES,
    CELL_SHADING_MAX_CHANGE_BOXES,
    CELL_SHADING_MAX_ENTRIES,
    CELL_SHADING_UNIFORM_BYTES,
    CELL_SHADING_WORKGROUP_SIZE,
    claimCellsWgsl,
    prioritizeCellsWgsl,
    scatterCellsWgsl,
    shadeCellsWgsl,
} from './cell-shading-wgsl.ts';
import { hashGridCacheCellSizeConstant, verticalFieldOfView } from './hash-grid-cpu.ts';
import {
    generateMultibounceReservoirsWgsl,
    generateReservoirsWgsl,
    checkHashGridRayListLimits,
    hashGridCacheRayListLayout,
    POPULATION_ENTRY_POINT,
    POPULATION_WORKGROUP_SIZE,
    populateCellsWgsl,
    populateMultibounceCellsWgsl,
    populateScreenProbesWgsl,
    RAY_DATA_BINDING,
    RAY_DATA_GROUP,
    RAY_DATA_GROUP_ENTRIES,
    CELL_SHADING_FLAG,
    RAY_LISTS_CELL_SHADING_WORD,
    RAY_LISTS_UNIFORM_BYTES,
    resolveCellsWgsl,
    updateMultibounceCellsWgsl,
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
import type { WorldCache, WorldCacheChanges, WorldCacheContext, WorldCacheFrame, WorldCacheLookup, WorldCacheProbeRays } from './world-cache.ts';

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
    readonly populateMultibounceCells: PopulationPass;
    readonly generateReservoirs: PopulationPass;
    readonly generateMultibounceReservoirs: PopulationPass;
    readonly populateCells: PopulationPass;
    readonly updateTiles: PopulationPass;
    readonly updateMultibounceCells: PopulationPass;
    readonly resolveCells: PopulationPass;
}

/** Lever E8's passes (cell-shading-wgsl.ts): kernels with their indirect arguments (none for the one-thread `PrioritizeCells`). */
interface CellShadingPasses {
    readonly claimFirstBounceCells: PopulationPass;
    readonly claimSecondBounceCells: PopulationPass;
    readonly prioritizeCells: Omit<PopulationPass, 'args'>;
    readonly scatterCells: PopulationPass;
    readonly shadeCells: PopulationPass;
    /** `HashGridCacheCellShading`: the budget and the frame's changes. */
    readonly uniform: GPUBuffer;
    /** Group 3 of each kernel: the uniform. */
    readonly groups: Readonly<Record<CellShadingPassName, GPUBindGroup>>;
    /** Group 3 of `PopulateCells`, which reads the changes for E9. */
    readonly populateCellsGroup: GPUBindGroup;
}

type PopulationPassName = keyof PopulationPasses;
type CellShadingPassName = 'claimFirstBounceCells' | 'claimSecondBounceCells' | 'prioritizeCells' | 'scatterCells' | 'shadeCells';
type RayDataPassName = PopulationPassName | CellShadingPassName;

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
    readonly cellShading: CellShadingPasses;
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
    readonly groups: Readonly<Record<RayDataPassName, GPUBindGroup>>;
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
    /** The options whose change empties the cache, as the last `encodeFrame` read them; null before the first. */
    private lastClearOptions: HashGridCacheClearOptions | null = null;
    /** Lever E8: the scene changes of the last `CELL_SHADING_CHANGE_FRAMES` frames, kept so cells hit only now and then are caught too. */
    private recentChanges: RecentCellChanges = [];

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
        if (this.lastFramePingPong === null) throw new Error('@pmndrs/gi: the world cache has no packed tile list before its first encodeFrame');
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
            throw new Error(`@pmndrs/gi: the GPU ran out of memory making the world cache's ${layout.totalBytes} bytes: ${outOfMemory.message}`);
        }

        const cacheGroup = { label: 'radiance-cache/hash-grid group', entries: HASH_GRID_CACHE_GROUP_ENTRIES };
        const rayDataGroup = { label: 'radiance-cache/hash-grid ray data', entries: RAY_DATA_GROUP_ENTRIES };
        const word = (counter: number): GpuWord => ({ buffer: buffers.tileWords, offset: counter * WORD_BYTES });
        /** A population kernel: the hash grid in group 1, the ray data in group 2, the platform's hit shading in group 0 when it shades. */
        const populationKernel = (label: string, code: string, entryPoint: string, shades: boolean): Promise<ComputeKernel> =>
            kernels.computeKernel({ label: `radiance-cache/${label}`, modules: shades ? [shade.module] : [], code, entryPoint, ownGroups: [cacheGroup, rayDataGroup] });
        // Lever E8's kernels add its uniform in group 3.
        const cellShadingGroup = { label: 'radiance-cache/cell shading', entries: CELL_SHADING_GROUP_ENTRIES };
        const cellShadingKernel = (label: string, code: string, entryPoint: string, shades: boolean): Promise<ComputeKernel> =>
            kernels.computeKernel({ label: `radiance-cache/${label}`, modules: shades ? [shade.module] : [], code, entryPoint, ownGroups: [cacheGroup, rayDataGroup, cellShadingGroup] });
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
            // With the changes' uniform in group 3 (experiment E9).
            cellShadingKernel('PopulateCells', populateCellsWgsl(layout), POPULATION_ENTRY_POINT.populateCells, true),
            populationKernel('UpdateTiles', updateTilesWgsl(layout), POPULATION_ENTRY_POINT.updateTiles, false),
            populationKernel('ResolveCells', resolveCellsWgsl(layout), POPULATION_ENTRY_POINT.resolveCells, false),
            // GI-1.2's multibounce (compiled always; run only with useMultibounce).
            populationKernel('PopulateMultibounceCells', populateMultibounceCellsWgsl(layout), POPULATION_ENTRY_POINT.populateMultibounceCells, true),
            populationKernel('GenerateMultibounceReservoirs', generateMultibounceReservoirsWgsl(layout), POPULATION_ENTRY_POINT.generateMultibounceReservoirs, true),
            populationKernel('UpdateMultibounceCells', updateMultibounceCellsWgsl(layout), POPULATION_ENTRY_POINT.updateMultibounceCells, false),
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
            // generateDispatch(radiance_cache_visibility_count_buffer0_, ...) (gi1.cpp:2385-2390)
            indirectArgs('PopulateMultibounceCells args', HASH_GRID_COUNTER.visibilityCount0, POPULATION_WORKGROUP_SIZE.populateMultibounceCells),
            // generateDispatch(radiance_cache_visibility_count_buffer1_, ...) (gi1.cpp:2431-2432, 2512-2513)
            indirectArgs('GenerateMultibounceReservoirs args', HASH_GRID_COUNTER.visibilityCount1, POPULATION_WORKGROUP_SIZE.generateMultibounceReservoirs),
            indirectArgs('UpdateMultibounceCells args', HASH_GRID_COUNTER.visibilityCount1, POPULATION_WORKGROUP_SIZE.updateMultibounceCells),
        ]);
        // Lever E8 (compiled always, run only with cellShading 'budgeted').
        const claimCode = claimCellsWgsl(layout);
        const [claimFirstKernel, claimSecondKernel, prioritizeKernel, scatterKernel, shadeKernel, claimFirstArgs, claimSecondArgs, scatterArgs, shadeArgs] = await Promise.all([
            cellShadingKernel('ClaimCells (first bounce)', claimCode, CELL_SHADING_ENTRY_POINT.claimFirstBounceCells, false),
            cellShadingKernel('ClaimCells (second bounce)', claimCode, CELL_SHADING_ENTRY_POINT.claimSecondBounceCells, false),
            cellShadingKernel('PrioritizeCells', prioritizeCellsWgsl(layout), CELL_SHADING_ENTRY_POINT.prioritizeCells, false),
            cellShadingKernel('ScatterCells', scatterCellsWgsl(layout), CELL_SHADING_ENTRY_POINT.scatterCells, false),
            cellShadingKernel('ShadeCells', shadeCellsWgsl(layout), CELL_SHADING_ENTRY_POINT.shadeCells, true),
            indirectArgs('ClaimCells (first bounce) args', HASH_GRID_COUNTER.visibilityCount0, CELL_SHADING_WORKGROUP_SIZE.claimCells),
            indirectArgs('ClaimCells (second bounce) args', HASH_GRID_COUNTER.visibilityCount1, CELL_SHADING_WORKGROUP_SIZE.claimCells),
            indirectArgs('ScatterCells args', HASH_GRID_COUNTER.cellCandidateCount, CELL_SHADING_WORKGROUP_SIZE.scatterCells),
            indirectArgs('ShadeCells args', HASH_GRID_COUNTER.cellShadedCount, CELL_SHADING_WORKGROUP_SIZE.shadeCells),
        ]);
        const [populateScreenProbes, generateReservoirs, populateCells, updateTiles, resolveCells, populateMultibounceCells, generateMultibounceReservoirs, updateMultibounceCells] =
            population.slice(0, 8) as ComputeKernel[];
        const [populateArgs, reservoirArgs, cellArgs, updateArgs, resolveArgs, multibounceCellArgs, multibounceReservoirArgs, multibounceUpdateArgs] = population.slice(8) as GpuIndirectArgs[];
        const pass = (kernel: ComputeKernel, args: GpuIndirectArgs): PopulationPass => ({
            kernel,
            args,
            cacheGroup: device.createBindGroup({ label: `${kernel.pipeline.label} hash grid`, layout: kernel.ownLayouts[0], entries: groupEntries(buffers) }),
        });
        const populationPasses: PopulationPasses = {
            populateScreenProbes: pass(populateScreenProbes, populateArgs),
            populateMultibounceCells: pass(populateMultibounceCells, multibounceCellArgs),
            generateReservoirs: pass(generateReservoirs, reservoirArgs),
            generateMultibounceReservoirs: pass(generateMultibounceReservoirs, multibounceReservoirArgs),
            populateCells: pass(populateCells, cellArgs),
            updateTiles: pass(updateTiles, updateArgs),
            updateMultibounceCells: pass(updateMultibounceCells, multibounceUpdateArgs),
            resolveCells: pass(resolveCells, resolveArgs),
        };
        const rayListsUniform = device.createBuffer({ label: 'radiance-cache/hash-grid ray lists', size: RAY_LISTS_UNIFORM_BYTES, usage: BUFFER_USAGE.uniform | BUFFER_USAGE.copyDst });
        const cellShadingUniform = device.createBuffer({ label: 'radiance-cache/cell shading', size: CELL_SHADING_UNIFORM_BYTES, usage: BUFFER_USAGE.uniform | BUFFER_USAGE.copyDst });
        const cellShadingGroupFor = (kernel: ComputeKernel): GPUBindGroup =>
            device.createBindGroup({ label: `${kernel.pipeline.label} cell shading`, layout: kernel.ownLayouts[2], entries: [{ binding: 0, resource: { buffer: cellShadingUniform } }] });
        const cellShading: CellShadingPasses = {
            claimFirstBounceCells: pass(claimFirstKernel, claimFirstArgs),
            claimSecondBounceCells: pass(claimSecondKernel, claimSecondArgs),
            prioritizeCells: {
                kernel: prioritizeKernel,
                cacheGroup: device.createBindGroup({ label: `${prioritizeKernel.pipeline.label} hash grid`, layout: prioritizeKernel.ownLayouts[0], entries: groupEntries(buffers) }),
            },
            scatterCells: pass(scatterKernel, scatterArgs),
            shadeCells: pass(shadeKernel, shadeArgs),
            uniform: cellShadingUniform,
            groups: {
                claimFirstBounceCells: cellShadingGroupFor(claimFirstKernel),
                claimSecondBounceCells: cellShadingGroupFor(claimSecondKernel),
                prioritizeCells: cellShadingGroupFor(prioritizeKernel),
                scatterCells: cellShadingGroupFor(scatterKernel),
                shadeCells: cellShadingGroupFor(shadeKernel),
            },
            populateCellsGroup: cellShadingGroupFor(populateCells),
        };

        const purgeArgs = [purgeArgs0, purgeArgs1] as const;
        const allArgs = [...purgeArgs, ...Object.values(populationPasses).map((populationPass) => populationPass.args), claimFirstArgs, claimSecondArgs, scatterArgs, shadeArgs];
        const argsBytes = allArgs.reduce((sum, args) => sum + args.memoryBytes, 0);
        const memoryBytes = layout.totalBytes + HASH_GRID_CONSTANTS_BYTES + RAY_LISTS_UNIFORM_BYTES + CELL_SHADING_UNIFORM_BYTES + argsBytes;
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
            cellShading,
            rayListsUniform,
            timer: context.timer,
            memoryBytes,
        };
        this.bufferPingPong = 0;
        this.lastFramePingPong = null;
        this.lastClearOptions = null;
    }

    /**
     * Empties the cache: the tile words (every hash slot, decay stamp, both
     * packed lists and the counters) back to zero, as `prepare` leaves them
     * (Capsaicin clears the hash buffer when it makes it, gi1.cpp:478, and in
     * `clearHashGridCache`, gi1.cpp:3286-3291). The cell values and scratch
     * stay: a tile's finest cells are cleared when it is inserted again
     * (`HashGridCache_AddNewTile`), its coarser levels are rewritten in the
     * same frame (`UpdateTiles`), and the direct scratch is zero between
     * frames (`UpdateTiles` clears what it reads, gi1.comp:2213-2224). The
     * indirect scratch is not: `UpdateMultibounceCells` fills it after
     * `UpdateTiles` for the next frame, so what the last frame left there is
     * blended into whichever new tile takes those cells, once. Capsaicin's
     * clear leaves it too.
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

        // Capsaicin empties the cache when the cell size, the smallest cell size or multibounce
        // changes (needs_hash_grid_clear, gi1.cpp:1740-1745, cleared at 1786-1789): cells keyed at
        // the old sizes would otherwise live on beside the new ones until they decay. Its other two
        // reasons are the debug view's mip level and propagation (we have no such options), and its
        // first frame (our buffers start empty from prepare). Turning multibounce on or off also
        // remakes its indirect buffers, the scratch cleared (gi1.cpp:497-552).
        // Before the count below: emptying the cache zeroes the tile words it is copied into.
        const clearOptions = hashGridCacheClearOptions(this.options);
        const last = this.lastClearOptions;
        if (last && !sameClearOptions(last, clearOptions)) {
            this.reset(encoder);
            if (last.useMultibounce !== clearOptions.useMultibounce) {
                const scratch = prepared.layout.updateScratch;
                const indirectStart = scratch.updateCellValueIndirectBuffer * WORD_BYTES;
                encoder.clearBuffer(prepared.buffers.updateScratch, indirectStart, scratch.bytes - indirectStart);
            }
        }
        this.lastClearOptions = clearOptions;
        const useMultibounce = clearOptions.useMultibounce;
        const budgeted = this.writeCellShading(encoder, prepared, rayData, frame);

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

        // Capsaicin's order (gi1.cpp:2335-2528): the multibounce passes in their slots.
        this.encodePopulationPass(encoder, prepared, rayData, 'populateScreenProbes', 'PopulateScreenProbesCells');
        if (useMultibounce) this.encodePopulationPass(encoder, prepared, rayData, 'populateMultibounceCells', 'PopulateMultibounceCells');
        this.encodePopulationPass(encoder, prepared, rayData, 'generateReservoirs', 'GenerateReservoirs');
        if (useMultibounce) this.encodePopulationPass(encoder, prepared, rayData, 'generateMultibounceReservoirs', 'GenerateMultibounceReservoirs');
        this.encodePopulationPass(encoder, prepared, rayData, 'populateCells', 'PopulateRadianceCache');
        if (budgeted) this.encodeCellShading(encoder, prepared, rayData, useMultibounce);
        this.encodePopulationPass(encoder, prepared, rayData, 'updateTiles', 'UpdateRadianceCache');
        if (useMultibounce) this.encodePopulationPass(encoder, prepared, rayData, 'updateMultibounceCells', 'UpdateMultibounceCells');
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
        const cellShading = prepared.cellShading;
        cellShading.uniform.destroy();
        for (const cellPass of [cellShading.claimFirstBounceCells, cellShading.claimSecondBounceCells, cellShading.scatterCells, cellShading.shadeCells]) cellPass.args.dispose();
        this.prepared = null;
    }

    /**
     * Lever E8: writes this frame's way of lighting cells into the ray lists'
     * uniform (every population pass reads it) and, when budgeted, the budget
     * and the changes into the lever's uniform, and records clearing its
     * per-frame counters. Returns whether the cells are lit on the budget.
     */
    private writeCellShading(encoder: GPUCommandEncoder, prepared: Prepared, rayData: RayData, frame: WorldCacheFrame): boolean {
        // Kept whichever way cells are lit, so switching to budgeted catches a change just made.
        this.recentChanges = keepRecentChanges(this.recentChanges, frame.frame, frame.changes);
        const { device, buffers } = prepared;
        const options = this.options;
        const budgeted = options.cellShading === 'budgeted';
        if (!budgeted && options.cellShading !== 'per-hit') throw new RangeError(`@pmndrs/gi: cellShading is 'per-hit' or 'budgeted', got ${String(options.cellShading)}`);
        const changeReset = !budgeted && options.cellChangeReset === true;
        const history = options.cellChangeResetHistory ?? 0;
        if (!Number.isInteger(history) || history < 0 || history > 15) throw new RangeError(`@pmndrs/gi: cellChangeResetHistory is a whole number of samples from 0 to 15, got ${history}`);
        const flags = (budgeted ? CELL_SHADING_FLAG.budgeted : 0) | (changeReset ? CELL_SHADING_FLAG.changeReset | (history << CELL_SHADING_FLAG.historyShift) : 0);
        device.queue.writeBuffer(prepared.rayListsUniform, RAY_LISTS_CELL_SHADING_WORD * WORD_BYTES, new Uint32Array([flags]));
        if (changeReset) device.queue.writeBuffer(prepared.cellShading.uniform, 0, cellShadingWords(0, recentChangesUnion(this.recentChanges)));
        if (!budgeted) return false;

        const budget = options.cellShadingBudget;
        if (!Number.isInteger(budget) || budget < 0 || budget > 0xffffffff) throw new RangeError(`@pmndrs/gi: cellShadingBudget is a whole number of cells, 0 or more, got ${budget}`);
        // A candidate's word holds its visibility entry in 23 bits (cell-shading-wgsl.ts).
        if (rayData.layout.maxCombinedRayCount > CELL_SHADING_MAX_ENTRIES) {
            throw new RangeError(`@pmndrs/gi: cells lit on a budget take at most ${CELL_SHADING_MAX_ENTRIES} hits a frame; this screen can have ${rayData.layout.maxCombinedRayCount}`);
        }
        device.queue.writeBuffer(prepared.cellShading.uniform, 0, cellShadingWords(budget, recentChangesUnion(this.recentChanges)));
        // The candidate and lit counts and the priority bins start at zero (the turn period is kept).
        encoder.clearBuffer(buffers.tileWords, HASH_GRID_COUNTER.cellCandidateCount * WORD_BYTES, 2 * WORD_BYTES);
        encoder.clearBuffer(rayData.listWords, rayData.layout.cellPriorityBins * WORD_BYTES, CELL_SHADING_BINS * WORD_BYTES);
        return true;
    }

    /** Lever E8's passes (cell-shading-wgsl.ts): claim, prioritise and scatter, light. */
    private encodeCellShading(encoder: GPUCommandEncoder, prepared: Prepared, rayData: RayData, useMultibounce: boolean): void {
        const cellShading = prepared.cellShading;
        const indirect = (pass: GPUComputePassEncoder, name: Exclude<CellShadingPassName, 'prioritizeCells'>): void => {
            const cellPass = cellShading[name];
            cellPass.args.encode(pass);
            cellPass.kernel.bind(pass);
            pass.setBindGroup(HASH_GRID_KERNEL_GROUP, cellPass.cacheGroup);
            pass.setBindGroup(RAY_DATA_GROUP, rayData.groups[name]);
            pass.setBindGroup(CELL_SHADING_GROUP_INDEX, cellShading.groups[name]);
            pass.dispatchWorkgroupsIndirect(cellPass.args.buffer, 0);
        };

        const claim = this.beginPass(encoder, prepared, 'ClaimCells');
        indirect(claim, 'claimFirstBounceCells');
        if (useMultibounce) indirect(claim, 'claimSecondBounceCells');
        claim.end();

        const prioritize = this.beginPass(encoder, prepared, 'PrioritizeCells');
        cellShading.prioritizeCells.kernel.bind(prioritize);
        prioritize.setBindGroup(HASH_GRID_KERNEL_GROUP, cellShading.prioritizeCells.cacheGroup);
        prioritize.setBindGroup(RAY_DATA_GROUP, rayData.groups.prioritizeCells);
        prioritize.setBindGroup(CELL_SHADING_GROUP_INDEX, cellShading.groups.prioritizeCells);
        prioritize.dispatchWorkgroups(1);
        indirect(prioritize, 'scatterCells');
        prioritize.end();

        const shade = this.beginPass(encoder, prepared, 'ShadeCells');
        indirect(shade, 'shadeCells');
        shade.end();
    }

    /** One of Capsaicin's timed sections as a compute pass: its indirect arguments from the GPU count, then the kernel. */
    private encodePopulationPass(encoder: GPUCommandEncoder, prepared: Prepared, rayData: RayData, name: PopulationPassName, section: string): void {
        const populationPass = prepared.population[name];
        const pass = this.beginPass(encoder, prepared, section);
        populationPass.args.encode(pass);
        populationPass.kernel.bind(pass);
        pass.setBindGroup(HASH_GRID_KERNEL_GROUP, populationPass.cacheGroup);
        pass.setBindGroup(RAY_DATA_GROUP, rayData.groups[name]);
        if (name === 'populateCells') pass.setBindGroup(CELL_SHADING_GROUP_INDEX, prepared.cellShading.populateCellsGroup);
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
        // The lists grow with the screen: a device that can't bind them gets GILimitError before anything is made (D14).
        checkHashGridRayListLimits(layout, device.limits);
        const listWords = device.createBuffer({ label: 'radiance-cache/hash-grid per-ray lists', size: layout.bytes, usage: BUFFER_USAGE.storage | BUFFER_USAGE.copySrc | BUFFER_USAGE.copyDst });
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
                layout.multibounceInfoBuffer,
                layout.multibounceRayBuffer,
                layout.firstBounceBypassBuffer,
                layout.multibounceDirectBuffer,
                layout.cellCandidateBuffer,
                layout.cellShadeBuffer,
                0, // cell_shading: written every frame (writeCellShading)
                layout.cellPriorityBins,
            ]),
        );
        const entries: GPUBindGroupEntry[] = [
            { binding: RAY_DATA_BINDING.rayLists, resource: { buffer: prepared.rayListsUniform } },
            { binding: RAY_DATA_BINDING.probeRays, resource: { buffer: rays.buffer } },
            { binding: RAY_DATA_BINDING.radiance, resource: rays.radiance },
            { binding: RAY_DATA_BINDING.listWords, resource: { buffer: listWords } },
        ];
        const groupFor = (kernel: ComputeKernel): GPUBindGroup => device.createBindGroup({ label: `${kernel.pipeline.label} ray data`, layout: kernel.ownLayouts[1], entries });
        const group = (name: PopulationPassName): GPUBindGroup => groupFor(prepared.population[name].kernel);
        const cellShading = prepared.cellShading;
        const rayData: RayData = {
            rays,
            layout,
            listWords,
            groups: {
                populateScreenProbes: group('populateScreenProbes'),
                populateMultibounceCells: group('populateMultibounceCells'),
                generateReservoirs: group('generateReservoirs'),
                generateMultibounceReservoirs: group('generateMultibounceReservoirs'),
                populateCells: group('populateCells'),
                updateTiles: group('updateTiles'),
                updateMultibounceCells: group('updateMultibounceCells'),
                resolveCells: group('resolveCells'),
                claimFirstBounceCells: groupFor(cellShading.claimFirstBounceCells.kernel),
                claimSecondBounceCells: groupFor(cellShading.claimSecondBounceCells.kernel),
                prioritizeCells: groupFor(cellShading.prioritizeCells.kernel),
                scatterCells: groupFor(cellShading.scatterCells.kernel),
                shadeCells: groupFor(cellShading.shadeCells.kernel),
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
        words[word.useMultibounce] = options.useMultibounce ? 1 : 0;
        prepared.device.queue.writeBuffer(prepared.buffers.constants, 0, bytes);
    }

    private require(): Prepared {
        if (!this.prepared) throw new Error('@pmndrs/gi: the world cache is used before prepare()');
        return this.prepared;
    }
}

/** The options whose change empties the cache (gi1.cpp:1740-1745), read once per frame. */
export interface HashGridCacheClearOptions {
    readonly cellSize: number;
    readonly minCellSize: number;
    readonly useMultibounce: boolean;
}

export function hashGridCacheClearOptions(options: HashGridCacheOptions): HashGridCacheClearOptions {
    return { cellSize: options.hashGridCacheCellSize, minCellSize: options.hashGridCacheMinCellSize, useMultibounce: options.useMultibounce };
}

export function sameClearOptions(a: HashGridCacheClearOptions, b: HashGridCacheClearOptions): boolean {
    return a.cellSize === b.cellSize && a.minCellSize === b.minCellSize && a.useMultibounce === b.useMultibounce;
}

/** Lever E8: a frame's changes and its number, newest last. */
export type RecentCellChanges = readonly { readonly frame: number; readonly changes: WorldCacheChanges }[];

/**
 * Lever E8: the changes of the frames in the window ending at `frame`
 * (`CELL_SHADING_CHANGE_FRAMES`), with this frame's added. A cell is marked
 * changed only on a frame it is hit, and on a still camera the jittered
 * probes reach some cells only now and then; a change remembered for a few
 * frames catches them (on the GPU, cells hit only on the change's frame left
 * the sun's lag at 31 frames with 20k cells lit, against 37 per hit).
 */
export function keepRecentChanges(recent: RecentCellChanges, frame: number, changes: WorldCacheChanges | undefined): RecentCellChanges {
    const kept = recent.filter((entry) => frame - entry.frame < CELL_SHADING_CHANGE_FRAMES && entry.frame <= frame);
    if (changes && (changes.everything || changes.boxes.length > 0)) kept.push({ frame, changes });
    return kept;
}

/** The window's changes as one: everything if any frame's was, else every box (`cellShadingWords` takes too many boxes as everything). */
export function recentChangesUnion(recent: RecentCellChanges): WorldCacheChanges {
    if (recent.some((entry) => entry.changes.everything)) return { everything: true, boxes: [] };
    return { everything: false, boxes: recent.flatMap((entry) => entry.changes.boxes) };
}

/** The group lever E8's kernels put their uniform in (cell-shading-wgsl.ts's `CELL_SHADING_GROUP`). */
const CELL_SHADING_GROUP_INDEX = 3;

/**
 * `HashGridCacheCellShading` (cell-shading-wgsl.ts): the budget and the
 * frame's changes, as boxes, or everywhere for a change with no bounds or
 * more boxes than it holds.
 */
export function cellShadingWords(budget: number, changes: WorldCacheChanges | undefined): ArrayBuffer {
    const bytes = new ArrayBuffer(CELL_SHADING_UNIFORM_BYTES);
    const words = new Uint32Array(bytes);
    const floats = new Float32Array(bytes);
    const boxes = changes?.boxes ?? [];
    const everything = (changes?.everything ?? false) || boxes.length > CELL_SHADING_MAX_CHANGE_BOXES;
    words[0] = budget;
    words[1] = everything ? 1 : 0;
    words[2] = everything ? 0 : boxes.length;
    if (!everything) {
        boxes.forEach((box, index) => {
            floats.set([box.min[0], box.min[1], box.min[2], 0, box.max[0], box.max[1], box.max[2], 0], 4 + index * 8);
        });
    }
    return bytes;
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
