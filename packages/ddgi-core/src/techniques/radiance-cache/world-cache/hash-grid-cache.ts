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
 * 3. `PopulateMultibounceCells` (RC3.3, with `useMultibounce`);
 * 4. `GenerateReservoirs`, `GenerateMultibounceReservoirs` (RC3.3),
 *    `PopulateRadianceCache` (`PopulateCells`), `UpdateRadianceCache`
 *    (`GenerateUpdateTilesDispatch`, `UpdateTiles`), `UpdateMultibounceCells`
 *    (RC3.3) and `ResolveCells` (RC3.2; `hash-grid-population-wgsl.ts`).
 *
 * Every pass after the purge is sized on the GPU, through the platform's
 * indirect-dispatch arguments, where Capsaicin dispatches indirectly
 * (`generateDispatch`).
 *
 * With lever 1's `cellShading: 'budget'` (RC-L1, read at `prepare`),
 * `GenerateReservoirs` gives way to four kernels (`hash-grid-cell-shading-wgsl.ts`):
 * the hits offer themselves to their cells, the cells in use are ranked by
 * age, the first `cellShadingBudget` are chosen and lit; the second bounce
 * leaves from the lit cells' points; `PopulateCells` traces only the bypass
 * cases' and the second bounce's shadow rays; `UpdateTiles` leaves a cell
 * with no new sample as it is. Everything else is unchanged.
 */

import type { GpuIndirectArgs, GpuPrefixScan, GpuWord } from '../../../platform/contract/compute.ts';
import type { ComputeKernel } from '../../../platform/contract/wgsl.ts';
import {
    checkCellShadingOptions,
    checkHashGridCacheLimits,
    DEFAULT_HASH_GRID_CACHE_OPTIONS,
    HASH_GRID_COUNTER,
    hashGridCacheBufferLayout,
    hashGridCacheCapacity,
    type HashGridCacheBufferLayout,
    type HashGridCacheCapacity,
    type HashGridCacheOptions,
    UPDATE_SCRATCH_WORDS_PER_CELL,
} from './capacity.ts';
import {
    CELL_SHADING_BINDINGS,
    CELL_SHADING_ENTRY_POINT,
    CELL_SHADING_WORKGROUP_SIZE,
    cellShadingChooseWgsl,
    cellShadingLightWgsl,
    cellShadingPickWgsl,
    cellShadingRankWgsl,
    type CellShadingPassName,
} from './hash-grid-cell-shading-wgsl.ts';
import { hashGridCacheCellSizeConstant, verticalFieldOfView } from './hash-grid-cpu.ts';
import {
    CELL_SHADING_CLASS_WORDS,
    CELL_SHADING_CLASSES,
    generateMultibounceReservoirsWgsl,
    generateReservoirsWgsl,
    hashGridCacheRayListLayout,
    POPULATION_BINDINGS,
    POPULATION_ENTRY_POINT,
    POPULATION_WORKGROUP_SIZE,
    populateCellsWgsl,
    populateMultibounceCellsWgsl,
    populateScreenProbesWgsl,
    RAY_DATA_BINDING,
    RAY_DATA_GROUP,
    rayDataGroupEntries,
    RAY_LISTS_UNIFORM_BYTES,
    resolveCellsWgsl,
    updateMultibounceCellsWgsl,
    updateTilesPerWorkgroup,
    updateTilesWgsl,
    type HashGridCacheRayListLayout,
    type RayDataBindingSet,
} from './hash-grid-population-wgsl.ts';
import {
    clearCountersWgsl,
    HASH_GRID_CACHE_ALL_BUFFERS,
    HASH_GRID_CACHE_BINDING,
    HASH_GRID_CACHE_GROUP_ENTRIES,
    HASH_GRID_CACHE_WORDS_ONLY,
    HASH_GRID_CONSTANTS_BYTES,
    HASH_GRID_CONSTANTS_WORD,
    HASH_GRID_ENTRY_POINT,
    HASH_GRID_KERNEL_GROUP,
    hashGridCacheGroupEntries,
    hashGridCacheWgsl,
    PURGE_TILES_WORKGROUP_SIZE,
    purgeTilesWgsl,
    type HashGridCacheBindingSet,
} from './hash-grid-wgsl.ts';
import type { WorldCache, WorldCacheContext, WorldCacheFrame, WorldCacheLookup, WorldCacheProbeRays } from './world-cache.ts';

/** `GPUBufferUsage` bits (the global doesn't exist in Node, where the tests import this file). */
const BUFFER_USAGE = { copySrc: 0x4, copyDst: 0x8, uniform: 0x40, storage: 0x80 } as const;

const WORD_BYTES = 4;

/** Timed as `radiance-cache/<section>`, Capsaicin's `TimedSection` names. */
const TIMER_PREFIX = 'radiance-cache';

/** The two packed buffers and the constant block. */
export interface HashGridCacheBuffers {
    /** Counters; per tile the hash slot, decay stamp, packed tile lists 0 and 1, update-listed frame; per cell the direct and indirect update scratch (`capacity.ts`). */
    readonly words: GPUBuffer;
    /** Direct, then indirect values: `vec2u` per cell. */
    readonly cellValues: GPUBuffer;
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

/**
 * Every pass the cache can run. Made at `prepare` for its settings:
 * Capsaicin's in both shading modes (the insert, `PopulateCells`,
 * `UpdateTiles`, `ResolveCells`); `GenerateReservoirs` only per hit; GI-1.2's
 * three with `useMultibounce` (register D42; `PopulateMultibounceCells` from
 * every hit per hit, from the lit cells on a budget); lever 1's four on a
 * budget.
 */
type PopulationPassName = keyof typeof POPULATION_BINDINGS | CellShadingPassName;
type PopulationPasses = Readonly<Partial<Record<PopulationPassName, PopulationPass>>>;

/** What each pass binds: Capsaicin's passes' sets and lever 1's. */
const PASS_BINDINGS: Readonly<Record<PopulationPassName, (typeof POPULATION_BINDINGS)[keyof typeof POPULATION_BINDINGS] | (typeof CELL_SHADING_BINDINGS)[CellShadingPassName]>> = {
    ...POPULATION_BINDINGS,
    ...CELL_SHADING_BINDINGS,
};

/** Lever 1's own resources (`cellShading: 'budget'`): the age-class histogram and the scan that ranks it. */
interface CellShadingResources {
    readonly classes: GPUBuffer;
    readonly scan: GpuPrefixScan;
    /** GPU bytes lever 1 adds besides its per-ray lists: the per-cell words, the classes, the scan, the indirect arguments. */
    readonly bytes: number;
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
    readonly population: PopulationPasses;
    /** Lever 1's histogram and scan, with `cellShading: 'budget'`; null per hit. */
    readonly cellShading: CellShadingResources | null;
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
    /** One per pass that was made. */
    readonly groups: Readonly<Partial<Record<PopulationPassName, GPUBindGroup>>>;
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
    /** Whether the last `encodeFrame` ran the multibounce, or null before the first. */
    private lastUseMultibounce: boolean | null = null;

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

    /** Inserts refused this frame because their bucket was full (register D23): a word of the cache words, cleared by `ClearCounters`. */
    get bucketOverflowCount(): GpuWord {
        return { buffer: this.require().buffers.words, offset: HASH_GRID_COUNTER.bucketOverflowCount * WORD_BYTES };
    }

    /**
     * The packed tile list of the frame `encodeFrame` last recorded (the
     * tiles alive after its purge, plus those its passes add): its count, and
     * the word of the cache words where its tile indices start.
     */
    get packedTiles(): { readonly count: GpuWord; readonly firstWord: number } {
        const { buffers, layout } = this.require();
        if (this.lastFramePingPong === null) throw new Error('@pmndrs/ddgi: the world cache has no packed tile list before its first encodeFrame');
        const first = this.lastFramePingPong === 0;
        const countWord = first ? HASH_GRID_COUNTER.packedTileCount0 : HASH_GRID_COUNTER.packedTileCount1;
        return {
            count: { buffer: buffers.words, offset: countWord * WORD_BYTES },
            firstWord: first ? layout.words.packedTileIndexBuffer0 : layout.words.packedTileIndexBuffer1,
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
        const cellShadingListBytes = prepared.cellShading && this.rayData ? (this.rayData.layout.bytes - hashGridCacheRayListLayout(this.rayData.layout.maxRayCount, capacity.numCells, prepared.layout.multibounce).bytes) : 0;
        return {
            buckets: capacity.numBuckets,
            'tile slots per bucket': capacity.numTilesPerBucket,
            tiles: capacity.numTiles,
            'cells per tile': capacity.numCellsPerTile,
            cells: capacity.numCells,
            'per-ray lists (bytes)': this.rayData?.listWords.size ?? 0,
            // 1 when prepared with useMultibounce: the indirect values and their scratch exist (register D42).
            'multibounce arrays': prepared.layout.multibounce ? 1 : 0,
            // Lever 1 (RC-L1): 1 when prepared with cellShading 'budget'; its memory, 0 per hit.
            'cell shading budget': prepared.cellShading ? 1 : 0,
            'cell shading (bytes)': (prepared.cellShading?.bytes ?? 0) + cellShadingListBytes,
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
        return device.createBindGroup({ label, layout, entries: groupEntries(buffers, HASH_GRID_CACHE_ALL_BUFFERS) });
    }

    readonly lookup: WorldCacheLookup = {
        code: (group: number) => this.wgsl(group),
        entries: HASH_GRID_CACHE_GROUP_ENTRIES,
        createBindGroup: (layout: GPUBindGroupLayout) => this.createBindGroup(layout, 'radiance-cache/hash-grid lookup'),
    };

    /**
     * Whether `encodeFrame` runs GI-1.2's multibounce: the option, if the
     * cache was prepared with it (only then do the indirect values exist and
     * the kernels read them; register D42). Null before `prepare`.
     */
    get multibounceActive(): boolean | null {
        const prepared = this.prepared;
        return prepared ? prepared.layout.multibounce && this.options.useMultibounce : null;
    }

    /** Whether the cache was prepared with `useMultibounce` (it holds the indirect values), or null before `prepare`. */
    get preparedWithMultibounce(): boolean | null {
        return this.prepared?.layout.multibounce ?? null;
    }

    /** How the cells get their light, as prepared (lever 1; `cellShading` takes effect at `prepare`), or null before. */
    get preparedCellShading(): HashGridCacheOptions['cellShading'] | null {
        const prepared = this.prepared;
        return prepared ? (prepared.cellShading ? 'budget' : 'perHit') : null;
    }

    /**
     * Lever 1's counters for the latest frame, words of the cache words (null
     * per hit): the cells in use (a hit offered itself), the cells lit, the
     * shadow rays their lighting traced, the cells in use that were stale, and
     * the samples written to the lit cells.
     */
    get cellShadingCounters(): {
        readonly usedCells: GpuWord;
        readonly litCells: GpuWord;
        readonly shadowRays: GpuWord;
        readonly staleCells: GpuWord;
        readonly samples: GpuWord;
        readonly ageClampedCells: GpuWord;
    } | null {
        const prepared = this.require();
        if (!prepared.cellShading) return null;
        const word = (counter: number): GpuWord => ({ buffer: prepared.buffers.words, offset: counter * WORD_BYTES });
        return {
            usedCells: word(HASH_GRID_COUNTER.usedCellCount),
            litCells: word(HASH_GRID_COUNTER.litCellCount),
            shadowRays: word(HASH_GRID_COUNTER.cellShadowRayCount),
            staleCells: word(HASH_GRID_COUNTER.staleCellCount),
            samples: word(HASH_GRID_COUNTER.cellSampleCount),
            ageClampedCells: word(HASH_GRID_COUNTER.ageClampedCellCount),
        };
    }

    async prepare(context: WorldCacheContext): Promise<void> {
        const { device, kernels, compute, shade } = context;
        const capacity = hashGridCacheCapacity(this.options);
        // Capsaicin's USE_MULTI_BOUNCE: without it the indirect arrays aren't made (gi1.cpp:497-552)
        // and the kernels are compiled without them (gi1.cpp:1428-1431).
        const multibounce = this.options.useMultibounce;
        // Lever 1: read at prepare, as its words, buffers and kernels are made here.
        const budget = this.options.cellShading === 'budget';
        if (budget) checkCellShadingOptions(this.options);
        else if (this.options.cellShading !== 'perHit') throw new RangeError(`@pmndrs/ddgi: cellShading must be 'perHit' or 'budget', got ${String(this.options.cellShading)}`);
        const layout = hashGridCacheBufferLayout(capacity, multibounce, budget);
        // Before anything is made: a device that can't hold the cache gets the limit's name (register D14).
        checkHashGridCacheLimits(layout, device.limits);
        const tilesPerUpdateGroup = updateTilesPerWorkgroup(this.options.hashGridCacheTileCellRatio);
        this.dispose();

        // Made zeroed (WebGPU clears new buffers), which is the empty cache: every hash slot 0
        // (Capsaicin clears the hash buffer and the update scratch at creation, gi1.cpp:478, 530).
        const storageUsage = BUFFER_USAGE.storage | BUFFER_USAGE.copySrc | BUFFER_USAGE.copyDst;
        device.pushErrorScope('out-of-memory');
        const buffers: HashGridCacheBuffers = {
            words: device.createBuffer({ label: 'radiance-cache/hash-grid cache words', size: layout.words.bytes, usage: storageUsage }),
            cellValues: device.createBuffer({ label: 'radiance-cache/hash-grid cell values', size: layout.cellValues.bytes, usage: storageUsage }),
            constants: device.createBuffer({ label: 'radiance-cache/hash-grid constants', size: HASH_GRID_CONSTANTS_BYTES, usage: BUFFER_USAGE.uniform | BUFFER_USAGE.copyDst }),
        };
        const outOfMemory = await device.popErrorScope();
        if (outOfMemory) {
            for (const buffer of Object.values(buffers)) buffer.destroy();
            throw new Error(`@pmndrs/ddgi: the GPU ran out of memory making the world cache's ${layout.totalBytes} bytes: ${outOfMemory.message}`);
        }

        const cacheGroup = (set: HashGridCacheBindingSet): GPUBindGroupLayoutDescriptor => ({ label: 'radiance-cache/hash-grid group', entries: hashGridCacheGroupEntries(set) });
        const rayDataGroup = (set: RayDataBindingSet): GPUBindGroupLayoutDescriptor => ({ label: 'radiance-cache/hash-grid ray data', entries: rayDataGroupEntries(set) });
        const entryPoint = (name: PopulationPassName): string => (name in CELL_SHADING_ENTRY_POINT ? CELL_SHADING_ENTRY_POINT[name as CellShadingPassName] : POPULATION_ENTRY_POINT[name as keyof typeof POPULATION_ENTRY_POINT]);
        const word = (counter: number): GpuWord => ({ buffer: buffers.words, offset: counter * WORD_BYTES });
        /**
         * A population kernel: the hash grid in group 1 and the ray data in group 2, each with the
         * buffers the pass binds (POPULATION_BINDINGS), and the platform's hit shading in group 0
         * when it shades. The kernel builder counts them against maxStorageBuffersPerShaderStage.
         */
        const populationKernel = (label: string, name: PopulationPassName, code: string, shades: boolean): Promise<ComputeKernel> =>
            kernels.computeKernel({
                label: `radiance-cache/${label}`,
                modules: shades ? [shade.module] : [],
                code,
                entryPoint: entryPoint(name),
                ownGroups: [cacheGroup(PASS_BINDINGS[name].cache), rayDataGroup(PASS_BINDINGS[name].rayData)],
            });
        const indirectArgs = (label: string, counter: number, itemsPerWorkgroup: number): Promise<GpuIndirectArgs> =>
            compute.createIndirectArgs({ label: `radiance-cache/${label}`, count: word(counter), itemsPerWorkgroup });

        // GI-1.2's three passes and their arguments, only with multibounce (register D42).
        const multibounceWork = multibounce
            ? Promise.all([
                  // On a budget, the second bounce leaves from the lit cells' kept hits (lever 1).
                  populationKernel('PopulateMultibounceCells', 'populateMultibounceCells', populateMultibounceCellsWgsl(layout, budget ? 'litCells' : 'everyHit'), true),
                  populationKernel('GenerateMultibounceReservoirs', 'generateMultibounceReservoirs', generateMultibounceReservoirsWgsl(layout), true),
                  populationKernel('UpdateMultibounceCells', 'updateMultibounceCells', updateMultibounceCellsWgsl(layout), false),
                  // generateDispatch(radiance_cache_visibility_count_buffer0_, ...) (gi1.cpp:2389); on a budget, the
                  // short list of the lit cells' picks and the bypassing hits.
                  indirectArgs('PopulateMultibounceCells args', budget ? HASH_GRID_COUNTER.multibounceSourceCount : HASH_GRID_COUNTER.visibilityCount0, POPULATION_WORKGROUP_SIZE.populateMultibounceCells),
                  // generateDispatch(radiance_cache_visibility_count_buffer1_, ...) (gi1.cpp:2430, 2511)
                  indirectArgs('GenerateMultibounceReservoirs args', HASH_GRID_COUNTER.visibilityCount1, POPULATION_WORKGROUP_SIZE.generateMultibounceReservoirs),
                  indirectArgs('UpdateMultibounceCells args', HASH_GRID_COUNTER.visibilityCount1, POPULATION_WORKGROUP_SIZE.updateMultibounceCells),
              ] as const)
            : Promise.resolve(null);
        const mainWork = Promise.all([
            kernels.computeKernel({
                label: 'radiance-cache/ClearCounters (hash grid)',
                modules: [],
                code: clearCountersWgsl(layout),
                entryPoint: HASH_GRID_ENTRY_POINT.clearCounters,
                ownGroups: [cacheGroup(HASH_GRID_CACHE_WORDS_ONLY)],
            }),
            kernels.computeKernel({
                label: 'radiance-cache/PurgeTiles',
                modules: [compute.indirectModule],
                code: purgeTilesWgsl(layout),
                entryPoint: HASH_GRID_ENTRY_POINT.purgeTiles,
                ownGroups: [cacheGroup(HASH_GRID_CACHE_WORDS_ONLY)],
            }),
            // With buffer_ping_pong 0, last frame's count is PackedTileCountBuffer1, and the other
            // way round (hash_grid_cache.hlsl:75-76; gi1.cpp:2241-2244).
            indirectArgs('PurgeTiles (ping-pong 0)', HASH_GRID_COUNTER.packedTileCount1, PURGE_TILES_WORKGROUP_SIZE),
            indirectArgs('PurgeTiles (ping-pong 1)', HASH_GRID_COUNTER.packedTileCount0, PURGE_TILES_WORKGROUP_SIZE),
            populationKernel('PopulateScreenProbes (cache half)', 'populateScreenProbes', populateScreenProbesWgsl(layout), false),
            populationKernel('PopulateCells', 'populateCells', populateCellsWgsl(layout), true),
            populationKernel('UpdateTiles', 'updateTiles', updateTilesWgsl(layout), false),
            populationKernel('ResolveCells', 'resolveCells', resolveCellsWgsl(layout), false),
            // Capsaicin dispatches PopulateScreenProbes over every possible ray; here it is its own
            // pass after the screen cache's trace, so it runs on the probe-ray count.
            indirectArgs('PopulateScreenProbes (cache half) args', HASH_GRID_COUNTER.probeRayCount, POPULATION_WORKGROUP_SIZE.populateScreenProbes),
            // generateDispatch(radiance_cache_visibility_ray_count_buffer_, ...) (gi1.cpp:2494-2499)
            indirectArgs('PopulateCells args', HASH_GRID_COUNTER.visibilityRayCount, POPULATION_WORKGROUP_SIZE.populateCells),
            // GenerateUpdateTilesDispatch (gi1.comp:2129-2140): ceil(update tiles / num_tiles_by_group) groups.
            indirectArgs('GenerateUpdateTilesDispatch', HASH_GRID_COUNTER.updateTileCount, tilesPerUpdateGroup),
            // generateDispatch(radiance_cache_resolve_count_buffer_, ...) (gi1.cpp:2523-2527)
            indirectArgs('ResolveCells args', HASH_GRID_COUNTER.resolveCount, POPULATION_WORKGROUP_SIZE.resolveCells),
        ]);
        // Capsaicin's light draw per hit, or lever 1's four kernels in its place.
        const perHitWork = budget
            ? Promise.resolve(null)
            : Promise.all([
                  populationKernel('GenerateReservoirs', 'generateReservoirs', generateReservoirsWgsl(layout), true),
                  // generateDispatch(radiance_cache_visibility_count_buffer0_, ...) (gi1.cpp:2423-2426)
                  indirectArgs('GenerateReservoirs args', HASH_GRID_COUNTER.visibilityCount0, POPULATION_WORKGROUP_SIZE.generateReservoirs),
              ] as const);
        const classes = budget
            ? device.createBuffer({ label: 'radiance-cache/hash-grid cell shading classes', size: CELL_SHADING_CLASS_WORDS * WORD_BYTES, usage: storageUsage })
            : null;
        const budgetWork = classes
            ? Promise.all([
                  populationKernel('CellShadingPick', 'cellShadingPick', cellShadingPickWgsl(layout), true),
                  populationKernel('CellShadingRank', 'cellShadingRank', cellShadingRankWgsl(layout), false),
                  populationKernel('CellShadingChoose', 'cellShadingChoose', cellShadingChooseWgsl(layout), false),
                  populationKernel('CellShadingLight', 'cellShadingLight', cellShadingLightWgsl(layout), true),
                  indirectArgs('CellShadingPick args', HASH_GRID_COUNTER.visibilityCount0, CELL_SHADING_WORKGROUP_SIZE.cellShadingPick),
                  indirectArgs('CellShadingRank args', HASH_GRID_COUNTER.visibilityCount0, CELL_SHADING_WORKGROUP_SIZE.cellShadingRank),
                  indirectArgs('CellShadingChoose args', HASH_GRID_COUNTER.usedCellCount, CELL_SHADING_WORKGROUP_SIZE.cellShadingChoose),
                  indirectArgs('CellShadingLight args', HASH_GRID_COUNTER.litCellCount, CELL_SHADING_WORKGROUP_SIZE.cellShadingLight),
                  // The age classes' first ranks: the platform's scan over the histogram, in place.
                  compute.createPrefixScan({ label: 'radiance-cache/CellShadingRank scan', input: classes, capacity: CELL_SHADING_CLASSES }),
              ] as const)
            : Promise.resolve(null);
        // Awaited together, so a refusal in any is the one error prepare throws.
        const [[clearCounters, purgeTiles, purgeArgs0, purgeArgs1, ...population], multibounceMade, perHitMade, budgetMade] = await Promise.all([mainWork, multibounceWork, perHitWork, budgetWork]);
        const kernelCount = 4;
        const [populateScreenProbes, populateCells, updateTiles, resolveCells] = population.slice(0, kernelCount) as ComputeKernel[];
        const [populateArgs, cellArgs, updateArgs, resolveArgs] = population.slice(kernelCount) as GpuIndirectArgs[];
        const pass = (name: PopulationPassName, kernel: ComputeKernel, args: GpuIndirectArgs): PopulationPass => ({
            kernel,
            args,
            cacheGroup: device.createBindGroup({
                label: `${kernel.pipeline.label} hash grid`,
                layout: kernel.ownLayouts[0],
                entries: groupEntries(buffers, PASS_BINDINGS[name].cache),
            }),
        });
        const multibouncePasses: PopulationPasses = multibounceMade
            ? {
                  populateMultibounceCells: pass('populateMultibounceCells', multibounceMade[0], multibounceMade[3]),
                  generateMultibounceReservoirs: pass('generateMultibounceReservoirs', multibounceMade[1], multibounceMade[4]),
                  updateMultibounceCells: pass('updateMultibounceCells', multibounceMade[2], multibounceMade[5]),
              }
            : {};
        const perHitPasses: PopulationPasses = perHitMade ? { generateReservoirs: pass('generateReservoirs', perHitMade[0], perHitMade[1]) } : {};
        const budgetPasses: PopulationPasses = budgetMade
            ? {
                  cellShadingPick: pass('cellShadingPick', budgetMade[0], budgetMade[4]),
                  cellShadingRank: pass('cellShadingRank', budgetMade[1], budgetMade[5]),
                  cellShadingChoose: pass('cellShadingChoose', budgetMade[2], budgetMade[6]),
                  cellShadingLight: pass('cellShadingLight', budgetMade[3], budgetMade[7]),
              }
            : {};
        const populationPasses: PopulationPasses = {
            populateScreenProbes: pass('populateScreenProbes', populateScreenProbes, populateArgs),
            populateCells: pass('populateCells', populateCells, cellArgs),
            updateTiles: pass('updateTiles', updateTiles, updateArgs),
            resolveCells: pass('resolveCells', resolveCells, resolveArgs),
            ...perHitPasses,
            ...multibouncePasses,
            ...budgetPasses,
        };
        const rayListsUniform = device.createBuffer({ label: 'radiance-cache/hash-grid ray lists', size: RAY_LISTS_UNIFORM_BYTES, usage: BUFFER_USAGE.uniform | BUFFER_USAGE.copyDst });

        const purgeArgs = [purgeArgs0, purgeArgs1] as const;
        const argsBytes = [...purgeArgs, ...populationPassList(populationPasses).map((populationPass) => populationPass.args)].reduce((sum, args) => sum + args.memoryBytes, 0);
        let cellShading: CellShadingResources | null = null;
        if (classes && budgetMade) {
            const scan = budgetMade[8];
            const budgetArgsBytes = budgetMade.slice(4, 8).reduce((sum, args) => sum + (args as GpuIndirectArgs).memoryBytes, 0);
            // The per-cell words are in the cache words: what the layout gains with them.
            const cellWordsBytes = layout.words.bytes - hashGridCacheBufferLayout(capacity, multibounce, false).words.bytes;
            cellShading = { classes, scan, bytes: cellWordsBytes + classes.size + scan.memoryBytes + budgetArgsBytes };
        }
        const memoryBytes = layout.totalBytes + HASH_GRID_CONSTANTS_BYTES + RAY_LISTS_UNIFORM_BYTES + argsBytes + (cellShading ? cellShading.classes.size + cellShading.scan.memoryBytes : 0);
        this.prepared = {
            device,
            capacity,
            layout,
            buffers,
            clearCounters,
            purgeTiles,
            clearCountersGroup: device.createBindGroup({ label: 'radiance-cache/ClearCounters (hash grid)', layout: clearCounters.ownLayouts[0], entries: groupEntries(buffers, HASH_GRID_CACHE_WORDS_ONLY) }),
            purgeTilesGroup: device.createBindGroup({ label: 'radiance-cache/PurgeTiles', layout: purgeTiles.ownLayouts[0], entries: groupEntries(buffers, HASH_GRID_CACHE_WORDS_ONLY) }),
            purgeArgs,
            population: populationPasses,
            cellShading,
            rayListsUniform,
            timer: context.timer,
            memoryBytes,
        };
        this.bufferPingPong = 0;
        this.lastFramePingPong = null;
        this.lastUseMultibounce = null;
    }

    /**
     * Empties the cache: the counters and the per-tile words (every hash
     * slot, decay stamp, both packed lists) back to zero, as `prepare` leaves
     * them (Capsaicin clears the hash buffer when it makes it, gi1.cpp:478).
     * The cell values and the update scratch after them stay: a tile's finest
     * cells are cleared when it is inserted again (`HashGridCache_AddNewTile`),
     * its coarser levels are rewritten in the same frame (`UpdateTiles`), and
     * the direct scratch is zero between frames (`UpdateTiles` clears what it
     * reads, gi1.comp:2213-2224). The indirect scratch can hold the second
     * bounce `UpdateMultibounceCells` added after `UpdateTiles` (the one-frame
     * lag); it stays too, as Capsaicin's reset clears only the hash buffer and
     * the packed tile counts (`clearHashGridCache`, gi1.cpp:3284-3291).
     */
    reset(encoder: GPUCommandEncoder): void {
        const { buffers, layout } = this.require();
        encoder.clearBuffer(buffers.words, 0, layout.words.updateCellValueBuffer * WORD_BYTES);
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
        encoder.copyBufferToBuffer(rays.count.buffer, rays.count.offset, prepared.buffers.words, HASH_GRID_COUNTER.probeRayCount * WORD_BYTES, WORD_BYTES);

        // The multibounce runs if the option is on and the cache was prepared with it (the indirect
        // values exist only then; turning the option on takes effect at the next prepare, D42).
        // Capsaicin empties the cache when gi1_use_multibounce changes, and makes the indirect
        // buffers (scratch cleared) when it comes on (gi1.cpp:1740-1745, 535-547): here, when
        // the option goes off and on again between two prepares, the scratch is cleared.
        const useMultibounce = this.multibounceActive === true;
        if (this.lastUseMultibounce !== null && this.lastUseMultibounce !== useMultibounce) {
            this.reset(encoder);
            const indirectScratch = prepared.layout.words.updateCellValueIndirectBuffer;
            if (indirectScratch !== null) {
                const indirectScratchBytes = UPDATE_SCRATCH_WORDS_PER_CELL * prepared.capacity.numCells * WORD_BYTES;
                encoder.clearBuffer(prepared.buffers.words, indirectScratch * WORD_BYTES, indirectScratchBytes);
            }
        }
        this.lastUseMultibounce = useMultibounce;

        const purge = this.beginPass(encoder, prepared, 'PurgeRadianceCache');
        prepared.purgeArgs[this.bufferPingPong].encode(purge);
        prepared.clearCounters.bind(purge);
        purge.setBindGroup(HASH_GRID_KERNEL_GROUP, prepared.clearCountersGroup);
        purge.dispatchWorkgroups(1);
        prepared.purgeTiles.bind(purge);
        purge.setBindGroup(HASH_GRID_KERNEL_GROUP, prepared.purgeTilesGroup);
        purge.dispatchWorkgroupsIndirect(prepared.purgeArgs[this.bufferPingPong].buffer, 0);
        purge.end();

        // GI1::render's order (gi1.cpp:2335-2528); the multibounce passes only with gi1_use_multibounce.
        this.encodePopulationPass(encoder, prepared, rayData, 'populateScreenProbes', 'PopulateScreenProbesCells');
        if (prepared.cellShading) {
            this.encodeCellShading(encoder, prepared, prepared.cellShading, rayData);
            // The second bounce from the cells just lit (lever 1, step 5), then its lights as Capsaicin draws them.
            if (useMultibounce) this.encodePopulationPass(encoder, prepared, rayData, 'populateMultibounceCells', 'PopulateMultibounceCells');
            if (useMultibounce) this.encodePopulationPass(encoder, prepared, rayData, 'generateMultibounceReservoirs', 'GenerateMultibounceReservoirs');
        } else {
            if (useMultibounce) this.encodePopulationPass(encoder, prepared, rayData, 'populateMultibounceCells', 'PopulateMultibounceCells');
            this.encodePopulationPass(encoder, prepared, rayData, 'generateReservoirs', 'GenerateReservoirs');
            if (useMultibounce) this.encodePopulationPass(encoder, prepared, rayData, 'generateMultibounceReservoirs', 'GenerateMultibounceReservoirs');
        }
        this.encodePopulationPass(encoder, prepared, rayData, 'populateCells', 'PopulateRadianceCache');
        this.encodePopulationPass(encoder, prepared, rayData, 'updateTiles', 'UpdateRadianceCache');
        // "Resolve bounce 1 cells into first bounce cells using last frame" (gi1.cpp:2505): after
        // UpdateTiles, so the next frame's UpdateTiles takes these samples.
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
        prepared.cellShading?.classes.destroy();
        prepared.cellShading?.scan.dispose();
        prepared.rayListsUniform.destroy();
        for (const args of prepared.purgeArgs) args.dispose();
        for (const populationPass of populationPassList(prepared.population)) populationPass.args.dispose();
        this.prepared = null;
    }

    /**
     * Lever 1's four kernels, each its own timed section: the hits offer
     * themselves (and the bypass cases draw their own light), the cells in use
     * are counted by age class and the histogram scanned into first ranks,
     * the first `cellShadingBudget` ranks are chosen, and those cells lit.
     */
    private encodeCellShading(encoder: GPUCommandEncoder, prepared: Prepared, cellShading: CellShadingResources, rayData: RayData): void {
        // The histogram and the cursors start at zero each frame (2 KB).
        encoder.clearBuffer(cellShading.classes);
        this.encodePopulationPass(encoder, prepared, rayData, 'cellShadingPick', 'CellShadingPick');
        const rank = this.beginPass(encoder, prepared, 'CellShadingRank');
        this.encodePopulationDispatch(rank, prepared, rayData, 'cellShadingRank');
        // Every class: 0 (new), then by age, oldest first; the order the cells are taken in.
        cellShading.scan.encode(rank, CELL_SHADING_CLASSES);
        rank.end();
        this.encodePopulationPass(encoder, prepared, rayData, 'cellShadingChoose', 'CellShadingChoose');
        this.encodePopulationPass(encoder, prepared, rayData, 'cellShadingLight', 'CellShadingLight');
    }

    /** One of Capsaicin's timed sections as a compute pass: its indirect arguments from the GPU count, then the kernel. */
    private encodePopulationPass(encoder: GPUCommandEncoder, prepared: Prepared, rayData: RayData, name: PopulationPassName, section: string): void {
        const pass = this.beginPass(encoder, prepared, section);
        this.encodePopulationDispatch(pass, prepared, rayData, name);
        pass.end();
    }

    private encodePopulationDispatch(pass: GPUComputePassEncoder, prepared: Prepared, rayData: RayData, name: PopulationPassName): void {
        const populationPass = prepared.population[name];
        const group = rayData.groups[name];
        if (!populationPass || !group) throw new Error(`@pmndrs/ddgi: the world cache was prepared without its ${name} pass (useMultibounce or cellShading changed since prepare)`);
        populationPass.args.encode(pass);
        populationPass.kernel.bind(pass);
        pass.setBindGroup(HASH_GRID_KERNEL_GROUP, populationPass.cacheGroup);
        pass.setBindGroup(RAY_DATA_GROUP, group);
        pass.dispatchWorkgroupsIndirect(populationPass.args.buffer, 0);
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
        const layout = hashGridCacheRayListLayout(rays.capacity, prepared.capacity.numCells, prepared.layout.multibounce, prepared.layout.cellShading);
        const listWords = device.createBuffer({ label: 'radiance-cache/hash-grid per-ray lists', size: layout.bytes, usage: BUFFER_USAGE.storage | BUFFER_USAGE.copySrc });
        device.queue.writeBuffer(
            prepared.rayListsUniform,
            0,
            new Uint32Array([
                layout.maxRayCount,
                layout.maxCombinedRayCount,
                layout.updateTileBuffer,
                layout.visibilityBuffer,
                layout.visibilityCellBuffer,
                layout.visibilityQueryBuffer,
                layout.visibilityRayBuffer,
                layout.resolveBuffer,
                layout.reservoirBuffer,
                layout.multibounceInfoBuffer,
                layout.visibilityBypassBuffer,
                layout.usedCellBuffer,
                layout.chosenCellBuffer,
                layout.multibounceSourceBuffer,
            ]),
        );
        /** The ray-data group's entries for the buffers a pass binds (POPULATION_BINDINGS). */
        const entries = (set: RayDataBindingSet): GPUBindGroupEntry[] => {
            const bound: GPUBindGroupEntry[] = [{ binding: RAY_DATA_BINDING.rayLists, resource: { buffer: prepared.rayListsUniform } }];
            if (set.probeRays) bound.push({ binding: RAY_DATA_BINDING.probeRays, resource: { buffer: rays.buffer } });
            if (set.radiance) bound.push({ binding: RAY_DATA_BINDING.radiance, resource: rays.radiance });
            bound.push({ binding: RAY_DATA_BINDING.listWords, resource: { buffer: listWords } });
            if (set.cellShadingClasses) {
                const classes = prepared.cellShading?.classes;
                if (!classes) throw new Error('@pmndrs/ddgi: a cell-shading pass without its classes buffer');
                bound.push({ binding: RAY_DATA_BINDING.cellShadingClasses, resource: { buffer: classes } });
            }
            return bound;
        };
        const groups: Partial<Record<PopulationPassName, GPUBindGroup>> = {};
        for (const name of Object.keys(prepared.population) as PopulationPassName[]) {
            const populationPass = prepared.population[name];
            if (!populationPass) continue;
            groups[name] = device.createBindGroup({
                label: `${populationPass.kernel.pipeline.label} ray data`,
                layout: populationPass.kernel.ownLayouts[1],
                entries: entries(PASS_BINDINGS[name].rayData),
            });
        }
        const rayData: RayData = { rays, layout, listWords, groups };
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
        words[word.useMultibounce] = this.multibounceActive ? 1 : 0;
        words[word.environmentLightEnable] = options.environmentLightEnable ? 1 : 0;
        // Lever 1's settings, read every frame on a budget (the mode itself was read at prepare).
        if (prepared.cellShading) checkCellShadingOptions(options);
        words[word.cellShadingBudget] = options.cellShadingBudget;
        words[word.cellShadingSamples] = options.cellShadingSamples;
        words[word.cellShadingMaxAge] = options.cellShadingMaxAge;
        prepared.device.queue.writeBuffer(prepared.buffers.constants, 0, bytes);
    }

    private require(): Prepared {
        if (!this.prepared) throw new Error('@pmndrs/ddgi: the world cache is used before prepare()');
        return this.prepared;
    }
}

/** The passes that were made (the multibounce ones only with `useMultibounce` at `prepare`). */
function populationPassList(passes: PopulationPasses): PopulationPass[] {
    return Object.values(passes).filter((populationPass): populationPass is PopulationPass => populationPass !== undefined);
}

/** The hash grid's group entries for the buffers of `set` (layout: `hashGridCacheGroupEntries(set)`). */
function groupEntries(buffers: HashGridCacheBuffers, set: HashGridCacheBindingSet): GPUBindGroupEntry[] {
    const entries: GPUBindGroupEntry[] = [
        { binding: HASH_GRID_CACHE_BINDING.constants, resource: { buffer: buffers.constants } },
        { binding: HASH_GRID_CACHE_BINDING.words, resource: { buffer: buffers.words } },
    ];
    if (set.cellValues) entries.push({ binding: HASH_GRID_CACHE_BINDING.cellValues, resource: { buffer: buffers.cellValues } });
    return entries;
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
