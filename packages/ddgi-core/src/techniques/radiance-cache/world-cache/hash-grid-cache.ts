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
 * Experiment E10 (P6.3): with the second bounce from a technique stacked
 * under the radiance cache (`secondBounce: 'auto'` or `'fallback'`), the three
 * multibounce passes give way to one, `SampleBounceSourceCells`, before
 * `UpdateTiles` (`bounce-source.ts`).
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
    cellResetFrameWords,
    checkHashGridCacheLimits,
    DEFAULT_HASH_GRID_CACHE_OPTIONS,
    HASH_GRID_COUNTER,
    hashGridCacheBufferLayout,
    hashGridCacheCapacity,
    type HashGridCacheBufferLayout,
    type HashGridCacheCapacity,
    type HashGridCacheOptions,
} from './capacity.ts';
import { CELL_CHANGES_BINDING, CELL_SHADING_MAX_CHANGE_EVENTS, type CellShadingListLayout } from './cell-shading-common.ts';
import {
    CELL_SHADING_ENTRY_POINT,
    CELL_SHADING_UNIFORM_BYTES,
    CELL_SHADING_WORKGROUP_SIZE,
    claimCellsWgsl,
    prioritizeCellsWgsl,
    scatterCellsWgsl,
    shadeCellsWgsl,
    sharedBounceWgsl,
} from './cell-shading-wgsl.ts';
import {
    BOUNCE_SOURCE_GROUP,
    SAMPLE_BOUNCE_SOURCE_ENTRY_POINT,
    SAMPLE_BOUNCE_SOURCE_WORKGROUP_SIZE,
    sampleBounceSourceCellsWgsl,
    shadingModuleLayoutEntries,
    ShadingModuleBindings,
} from './bounce-source.ts';
import type { ShadingModule } from '../../../platform/contract/shading-module.ts';
import type { SecondBounceInUse } from './capacity.ts';
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

/** The two packed buffers and the constant block (register D6). */
export interface HashGridCacheBuffers {
    /**
     * The words: counters, then per tile hash slot, decay stamp, packed tile
     * lists 0 and 1 and update-listed frame; then the direct and indirect
     * update scratch, four words per cell (the indirect only in a cache
     * prepared with `useMultibounce`, register D67).
     * Offsets in `HashGridCacheBufferLayout` (`capacity.ts`).
     */
    readonly tileWords: GPUBuffer;
    /** Direct, then indirect values: `vec2u` per cell (the indirect only with `useMultibounce` at `prepare`, register D67). */
    readonly cellValues: GPUBuffer;
    /** `HashGridCacheConstants`, written every frame. */
    readonly constants: GPUBuffer;
}

/** The per-ray lists of the latest frame, for tests and debugging: the packed words and where each list starts. */
export interface HashGridCacheRayLists {
    readonly buffer: GPUBuffer;
    readonly layout: HashGridCacheRayListLayout;
}

/** What the hash grid hands a tail (P6.4's neural tail, register E11) inside its frame. */
export interface HashGridTailFrame {
    readonly frame: WorldCacheFrame;
    /** The probe rays of this screen, and the per-ray lists over them (with the tail's list). */
    readonly rays: WorldCacheProbeRays;
    readonly rayLists: HashGridCacheRayListLayout;
    readonly listWords: GPUBuffer;
    /** Entries for a bind group of the hash grid (`HASH_GRID_CACHE_GROUP_ENTRIES`' layout) and of the ray data (`RAY_DATA_GROUP_ENTRIES`'). */
    readonly cacheEntries: readonly GPUBindGroupEntry[];
    readonly rayDataEntries: readonly GPUBindGroupEntry[];
}

/**
 * Work that runs inside the hash grid's frame with its data (P6.4): after
 * `UpdateTiles` (the cells' values this frame) and before
 * `UpdateMultibounceCells` (which reads its answers when `active`), only
 * while GI-1.2's own second-bounce rays run (multibounce on, `secondBounce`
 * resolved to `'gi1-rays'` and not waiting on a fallback's pass). The grid
 * makes its per-ray lists with the tail's list (`neuralTailBuffer`) while it
 * has one.
 */
export interface HashGridTail {
    /** Read when the frame's constants are written: sets `use_neural_tail`. It must not change until `encode`. */
    readonly active: boolean;
    /** Before the frame's constants are written: anything that changes `active` this frame (a restart) happens here. */
    beginFrame(frame: WorldCacheFrame): void;
    encode(encoder: GPUCommandEncoder, frame: HashGridTailFrame): void;
}

/** The population and resolve passes: a kernel and its indirect arguments each. */
interface PopulationPass {
    readonly kernel: ComputeKernel;
    readonly args: GpuIndirectArgs;
    readonly cacheGroup: GPUBindGroup;
}

/** GI-1.2's second-bounce passes (RC3.3), made only when the cache is prepared with `useMultibounce` (register D67). */
interface MultibouncePasses {
    readonly populateMultibounceCells: PopulationPass;
    readonly generateMultibounceReservoirs: PopulationPass;
    readonly updateMultibounceCells: PopulationPass;
}

interface PopulationPasses extends Partial<MultibouncePasses> {
    readonly populateScreenProbes: PopulationPass;
    readonly generateReservoirs: PopulationPass;
    readonly populateCells: PopulationPass;
    readonly updateTiles: PopulationPass;
    readonly resolveCells: PopulationPass;
}

/** Lever E8's passes (cell-shading-wgsl.ts): kernels with their indirect arguments (none for the one-thread `PrioritizeCells`). */
interface CellShadingPasses {
    readonly claimFirstBounceCells: PopulationPass;
    readonly claimSecondBounceCells: PopulationPass;
    readonly prioritizeCells: Omit<PopulationPass, 'args'>;
    readonly scatterCells: PopulationPass;
    readonly shadeCells: PopulationPass;
    /** pmndrs/gi#54's shared-cell bounce: made only with `useMultibounce` at `prepare`, as it adds to the indirect scratch (register D67). */
    readonly sharedCellBounce?: PopulationPass;
    readonly shortRayBounce?: PopulationPass;
}

type PopulationPassName = keyof PopulationPasses;
type MultibouncePassName = keyof MultibouncePasses;
type CellShadingPassName = keyof CellShadingPasses;
type RayDataPassName = PopulationPassName | CellShadingPassName;
type SharedBouncePassName = 'sharedCellBounce' | 'shortRayBounce';
type RayDataGroups = Readonly<Record<Exclude<RayDataPassName, MultibouncePassName | SharedBouncePassName>, GPUBindGroup>> & Readonly<Partial<Record<MultibouncePassName | SharedBouncePassName, GPUBindGroup>>>;

/** Experiment E10's pass and the fallback's resources it binds in group 3. */
interface BounceSourcePass extends PopulationPass {
    readonly bindings: ShadingModuleBindings;
    /** The layout of its ray-data group (made per probe-ray buffers, as the other passes'). */
    readonly rayDataLayout: GPUBindGroupLayout;
}

/**
 * E10's pass for one fallback module at one `version`: built at `prepare`
 * when the setting asks for it and a world-space fallback is stacked, or on
 * the first frame that wants it (a new module or version included); rays run
 * until it is ready, and for good if it failed to compile.
 */
interface BounceSourceBuild {
    readonly module: ShadingModule;
    readonly version: number;
    pass: BounceSourcePass | null;
    failed: boolean;
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
    readonly cellShading: CellShadingPasses;
    /** The context's fallback (read when used: the platform's is a getter) and its warnings. */
    readonly context: WorldCacheContext;
    /** What E10's pass is built from: the group layouts and the first-bounce count its dispatch reads. */
    readonly cacheGroupDescriptor: GPUBindGroupLayoutDescriptor;
    readonly rayDataGroupDescriptor: GPUBindGroupLayoutDescriptor;
    readonly firstBounceCount: GpuWord;
    /** `HashGridCacheRayLists`: where each per-ray list starts. */
    readonly rayListsUniform: GPUBuffer;
    /** `HashGridCacheCellShading` (E8's budget and the change events, read by E8 and E9). */
    readonly cellShadingUniform: GPUBuffer;
    readonly timer: WorldCacheContext['timer'];
    readonly memoryBytes: number;
}

/** The per-ray lists for one screen's probe rays, and the ray-data bind groups over them. */
interface RayData {
    readonly rays: WorldCacheProbeRays;
    readonly layout: HashGridCacheRayListLayout;
    readonly listWords: GPUBuffer;
    /** One per pass that was made: the second-bounce passes' only with multibounce at `prepare` (register D67). */
    groups: RayDataGroups;
    /** The ray-data group's entries, for E10's pass, whose group is made when that pass is first used with these lists (and again when the entries change). */
    entries: readonly GPUBindGroupEntry[];
    bounceSourceGroup: { readonly pass: BounceSourcePass; readonly group: GPUBindGroup } | null;
    /** Whether the lists have room for second-bounce entries (GI-1.2's rays). */
    readonly secondBounceRays: boolean;
}

export class HashGridCache implements WorldCache {
    readonly id = 'hash-grid';
    /** Live: the capacity fields take effect at the next `prepare`, the rest every frame. */
    readonly options: HashGridCacheOptions;

    private prepared: Prepared | null = null;
    private rayData: RayData | null = null;
    /** Per-ray list words replaced while a frame copied E9's frames out of them: destroyed at the next frame, after the host's submit. */
    private retiredListWords: GPUBuffer[] = [];
    /** Capsaicin's `radiance_cache_hash_buffer_ping_pong_` (gi1.h:272): which packed list the next frame fills. */
    private bufferPingPong = 0;
    /** The value the last `encodeFrame` used, or null before the first. */
    private lastFramePingPong: number | null = null;
    /** The options whose change empties the cache, as the last `encodeFrame` read them; null before the first. */
    private lastClearOptions: HashGridCacheClearOptions | null = null;
    /** The second-bounce estimator the last `encodeFrame` ran (experiment E10); null before the first or with multibounce off. */
    private lastSecondBounce: SecondBounceInUse | null = null;
    /** E10's pass for the current fallback module, or null when none was wanted yet. */
    private bounceSource: BounceSourceBuild | null = null;
    /** P6.4's neural tail (register E11), set by the technique; null runs GI-1.2's two bounces. Applies only while GI-1.2's second-bounce rays run. */
    tail: HashGridTail | null = null;
    /** The tail was asked for while the second bounce came from a fallback, and that was said once. */
    private tailInactiveWarned = false;
    /**
     * `resolveSecondBounce` answered `'gi1-rays'` only because the fallback's pass is still
     * being built (`'auto'` or `'fallback'` with a field stacked): the tail waits, so it never
     * trains or answers for the frames before the second bounce settles (P6.4's review).
     */
    private secondBouncePending = false;
    /** `useMultibounce` was turned on after a `prepare` without it, and that was said once (register D67). */
    private multibounceNeedsPrepareWarned = false;

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

    /** Lever E8's lists of the latest screen (there while it lights cells on a budget), for tests: the per-ray list words and where each list starts in them. */
    get cellShadingLists(): { readonly buffer: GPUBuffer; readonly layout: CellShadingListLayout } | null {
        const rayData = this.rayData;
        return rayData?.layout.cellLists ? { buffer: rayData.listWords, layout: rayData.layout.cellLists } : null;
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

    /**
     * The second-bounce estimator the latest frame ran (experiment E10):
     * `'fallback'` when the setting allows it, a world-space fallback is
     * stacked and its pass is built; null before the first frame or with
     * multibounce off.
     */
    get secondBounceInUse(): SecondBounceInUse | null {
        return this.lastSecondBounce;
    }

    /** GPU bytes: the cache's buffers at its capacity, the per-ray lists of the latest screen, and E10's pass when built. */
    get memoryBytes(): number {
        const pass = this.bounceSource?.pass;
        return (
            (this.prepared?.memoryBytes ?? 0) +
            (this.rayData?.listWords.size ?? 0) +
            (pass ? pass.args.memoryBytes + pass.bindings.memoryBytes : 0)
        );
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
            // 1 when prepared with useMultibounce: the indirect values and their scratch exist (register D67).
            'multibounce arrays': prepared.layout.multibounce ? 1 : 0,
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

    /** Whether the cache was prepared with `useMultibounce`, so it holds the indirect values (register D67); null before `prepare`. */
    get preparedWithMultibounce(): boolean | null {
        return this.prepared?.layout.multibounce ?? null;
    }

    async prepare(context: WorldCacheContext): Promise<void> {
        const { device, kernels, compute, shade } = context;
        const capacity = hashGridCacheCapacity(this.options);
        // Capsaicin's USE_MULTI_BOUNCE: without it the indirect arrays aren't made (gi1.cpp:497-552)
        // and the kernels are compiled without them (gi1.cpp:1428-1431; register D67).
        const multibounce = this.options.useMultibounce;
        const layout = hashGridCacheBufferLayout(capacity, multibounce);
        // Before anything is made: a device that can't hold the cache gets the limit's name (register D14).
        checkHashGridCacheLimits(layout, device.limits);
        const tilesPerUpdateGroup = updateTilesPerWorkgroup(this.options.hashGridCacheTileCellRatio);
        this.dispose();

        // Made zeroed (WebGPU clears new buffers), which is the empty cache: every hash slot 0
        // (Capsaicin clears the hash buffer and the update scratch at creation, gi1.cpp:478, 530).
        const storageUsage = BUFFER_USAGE.storage | BUFFER_USAGE.copySrc | BUFFER_USAGE.copyDst;
        device.pushErrorScope('out-of-memory');
        const buffers: HashGridCacheBuffers = {
            tileWords: device.createBuffer({ label: 'radiance-cache/hash-grid words', size: layout.words.bytes, usage: storageUsage }),
            cellValues: device.createBuffer({ label: 'radiance-cache/hash-grid cell values', size: layout.cellValues.bytes, usage: storageUsage }),
            // copySrc: the GPU pages read the words back (P6.4's page checks `use_neural_tail` on a restart frame).
            constants: device.createBuffer({ label: 'radiance-cache/hash-grid constants', size: HASH_GRID_CONSTANTS_BYTES, usage: BUFFER_USAGE.uniform | BUFFER_USAGE.copyDst | BUFFER_USAGE.copySrc }),
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
        const indirectArgs = (label: string, counter: number, itemsPerWorkgroup: number): Promise<GpuIndirectArgs> =>
            compute.createIndirectArgs({ label: `radiance-cache/${label}`, count: word(counter), itemsPerWorkgroup });

        // GI-1.2's three second-bounce passes and their arguments, only with multibounce (register D67).
        const multibounceWork = multibounce
            ? Promise.all([
                  populationKernel('PopulateMultibounceCells', populateMultibounceCellsWgsl(layout), POPULATION_ENTRY_POINT.populateMultibounceCells, true),
                  populationKernel('GenerateMultibounceReservoirs', generateMultibounceReservoirsWgsl(layout), POPULATION_ENTRY_POINT.generateMultibounceReservoirs, true),
                  populationKernel('UpdateMultibounceCells', updateMultibounceCellsWgsl(layout), POPULATION_ENTRY_POINT.updateMultibounceCells, false),
                  // generateDispatch(radiance_cache_visibility_count_buffer0_, ...) (gi1.cpp:2385-2390)
                  indirectArgs('PopulateMultibounceCells args', HASH_GRID_COUNTER.visibilityCount0, POPULATION_WORKGROUP_SIZE.populateMultibounceCells),
                  // generateDispatch(radiance_cache_visibility_count_buffer1_, ...) (gi1.cpp:2431-2432, 2512-2513)
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
        // Awaited together, so a refusal in either is the one error prepare throws.
        const [[clearCounters, purgeTiles, purgeArgs0, purgeArgs1, ...population], multibounceMade] = await Promise.all([mainWork, multibounceWork]);
        // Lever E8 (compiled always, run only with cellShading 'budgeted').
        const claimCode = claimCellsWgsl(layout);
        const [claimFirstKernel, claimSecondKernel, prioritizeKernel, scatterKernel, shadeKernel, claimFirstArgs, claimSecondArgs, scatterArgs, shadeArgs] = await Promise.all([
            populationKernel('ClaimCells (first bounce)', claimCode, CELL_SHADING_ENTRY_POINT.claimFirstBounceCells, false),
            populationKernel('ClaimCells (second bounce)', claimCode, CELL_SHADING_ENTRY_POINT.claimSecondBounceCells, false),
            populationKernel('PrioritizeCells', prioritizeCellsWgsl(layout), CELL_SHADING_ENTRY_POINT.prioritizeCells, false),
            populationKernel('ScatterCells', scatterCellsWgsl(layout), CELL_SHADING_ENTRY_POINT.scatterCells, false),
            populationKernel('ShadeCells', shadeCellsWgsl(layout), CELL_SHADING_ENTRY_POINT.shadeCells, true),
            indirectArgs('ClaimCells (first bounce) args', HASH_GRID_COUNTER.visibilityCount0, CELL_SHADING_WORKGROUP_SIZE.claimCells),
            indirectArgs('ClaimCells (second bounce) args', HASH_GRID_COUNTER.visibilityCount1, CELL_SHADING_WORKGROUP_SIZE.claimCells),
            indirectArgs('ScatterCells args', HASH_GRID_COUNTER.cellCandidateCount, CELL_SHADING_WORKGROUP_SIZE.scatterCells),
            indirectArgs('ShadeCells args', HASH_GRID_COUNTER.cellShadedCount, CELL_SHADING_WORKGROUP_SIZE.shadeCells),
        ]);
        // pmndrs/gi#54's shared-cell bounce adds to the indirect scratch, so it needs the multibounce's arrays (register D67).
        const sharedCode = multibounce ? sharedBounceWgsl(layout) : undefined;
        const sharedMade = sharedCode
            ? await Promise.all([
                  populationKernel('SharedCellBounce', sharedCode, 'SharedCellBounce', true),
                  populationKernel('ShortRayBounce', sharedCode, 'ShortRayBounce', true),
                  indirectArgs('ShortRayBounce args', HASH_GRID_COUNTER.resolveCount, 32),
              ])
            : undefined;
        const [populateScreenProbes, generateReservoirs, populateCells, updateTiles, resolveCells] = population.slice(0, 5) as ComputeKernel[];
        const [populateArgs, reservoirArgs, cellArgs, updateArgs, resolveArgs] = population.slice(5) as GpuIndirectArgs[];
        const pass = (kernel: ComputeKernel, args: GpuIndirectArgs): PopulationPass => ({
            kernel,
            args,
            cacheGroup: device.createBindGroup({ label: `${kernel.pipeline.label} hash grid`, layout: kernel.ownLayouts[0], entries: groupEntries(buffers) }),
        });
        const multibouncePasses: Partial<MultibouncePasses> = multibounceMade
            ? {
                  populateMultibounceCells: pass(multibounceMade[0], multibounceMade[3]),
                  generateMultibounceReservoirs: pass(multibounceMade[1], multibounceMade[4]),
                  updateMultibounceCells: pass(multibounceMade[2], multibounceMade[5]),
              }
            : {};
        const populationPasses: PopulationPasses = {
            populateScreenProbes: pass(populateScreenProbes, populateArgs),
            generateReservoirs: pass(generateReservoirs, reservoirArgs),
            populateCells: pass(populateCells, cellArgs),
            updateTiles: pass(updateTiles, updateArgs),
            resolveCells: pass(resolveCells, resolveArgs),
            ...multibouncePasses,
        };
        const rayListsUniform = device.createBuffer({ label: 'radiance-cache/hash-grid ray lists', size: RAY_LISTS_UNIFORM_BYTES, usage: BUFFER_USAGE.uniform | BUFFER_USAGE.copyDst });
        const cellShadingUniform = device.createBuffer({ label: 'radiance-cache/cell shading', size: CELL_SHADING_UNIFORM_BYTES, usage: BUFFER_USAGE.uniform | BUFFER_USAGE.copyDst });
        const cellShading: CellShadingPasses = {
            claimFirstBounceCells: pass(claimFirstKernel, claimFirstArgs),
            claimSecondBounceCells: pass(claimSecondKernel, claimSecondArgs),
            prioritizeCells: {
                kernel: prioritizeKernel,
                cacheGroup: device.createBindGroup({ label: `${prioritizeKernel.pipeline.label} hash grid`, layout: prioritizeKernel.ownLayouts[0], entries: groupEntries(buffers) }),
            },
            scatterCells: pass(scatterKernel, scatterArgs),
            shadeCells: pass(shadeKernel, shadeArgs),
            ...(sharedMade ? { sharedCellBounce: pass(sharedMade[0], shadeArgs), shortRayBounce: pass(sharedMade[1], sharedMade[2]) } : {}),
        };

        const purgeArgs = [purgeArgs0, purgeArgs1] as const;
        const allArgs = [...purgeArgs, ...populationPassList(populationPasses).map((populationPass) => populationPass.args), claimFirstArgs, claimSecondArgs, scatterArgs, shadeArgs, ...(sharedMade ? [sharedMade[2]] : [])];
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
            context,
            cacheGroupDescriptor: cacheGroup,
            rayDataGroupDescriptor: rayDataGroup,
            firstBounceCount: word(HASH_GRID_COUNTER.visibilityCount0),
            rayListsUniform,
            cellShadingUniform,
            timer: context.timer,
            memoryBytes,
        };
        this.bufferPingPong = 0;
        this.lastFramePingPong = null;
        this.lastClearOptions = null;
        this.lastSecondBounce = null;
        this.multibounceNeedsPrepareWarned = false;
        // E10: build the pass now when the setting can use it and a world-space fallback is stacked
        // (only then: 'gi1-rays' compiles nothing), so the first frame has it.
        // E10 adds to the indirect scratch, so it needs the multibounce's arrays (register D67).
        const module = (this.options.secondBounce === 'auto' || this.options.secondBounce === 'fallback') && multibounce ? worldSpaceFallback(context) : undefined;
        if (module) await this.buildBounceSource(this.prepared, module);
    }

    /**
     * E10's pass for `module` (`compileBounceSource`). A failed compile is
     * kept as failed, with a warning, and the cache traces rays; a pass that
     * finishes after the module or the cache moved on is thrown away.
     */
    private async buildBounceSource(prepared: Prepared, module: ShadingModule): Promise<void> {
        disposeBounceSource(this.bounceSource);
        const build: BounceSourceBuild = { module, version: module.version, pass: null, failed: false };
        this.bounceSource = build;
        try {
            const pass = await compileBounceSource(prepared, module);
            if (this.bounceSource === build && this.prepared === prepared) build.pass = pass;
            else disposeBounceSource({ ...build, pass });
        } catch (error) {
            build.failed = true;
            const message = error instanceof Error ? error.message : String(error);
            prepared.context.warn?.(`@pmndrs/gi: the radiance cache could not build its pass reading ${module.techniqueId}'s field as the second bounce (${message}); tracing GI-1.2's second-bounce rays`);
        }
    }

    /**
     * Which estimator fills the indirect values this frame (experiment E10).
     * `'gi1-rays'` asks for rays. `'auto'` and `'fallback'` read the stacked
     * technique when its shading module answers in world space and the pass
     * for it is built; a new module or `version` starts a new build (rays until
     * it is ready). `'fallback'` warns when it can't have it; `'auto'` doesn't.
     */
    private resolveSecondBounce(prepared: Prepared): SecondBounceInUse {
        this.secondBouncePending = false;
        const asked = this.options.secondBounce;
        if (asked === 'gi1-rays' || asked === 'shared-cell') return asked;
        const warnAsked = (message: string): void => {
            if (asked === 'fallback') prepared.context.warn?.(message);
        };
        const module = prepared.context.fallback?.shading;
        if (!module) {
            warnAsked("@pmndrs/gi: the radiance cache's secondBounce 'fallback' needs a technique stacked under it (gi.setFallback(radianceCache, ddgi)); tracing GI-1.2's second-bounce rays instead");
            return 'gi1-rays';
        }
        if (module.provides.worldSpaceIrradiance !== true) {
            warnAsked(`@pmndrs/gi: the radiance cache's secondBounce 'fallback' needs a world-space field (provides.worldSpaceIrradiance), and ${module.techniqueId}'s shading module is not one; tracing GI-1.2's second-bounce rays instead`);
            return 'gi1-rays';
        }
        const build = this.bounceSource;
        if (!build || build.module !== module || build.version !== module.version) {
            if (build?.pass) prepared.context.warn?.(`@pmndrs/gi: the technique under the radiance cache changed (${module.techniqueId}, shading version ${module.version}); rebuilding the pass that reads it, GI-1.2's rays until it is ready`);
            void this.buildBounceSource(prepared, module);
            this.secondBouncePending = true;
            return 'gi1-rays';
        }
        if (build.pass) return 'fallback';
        this.secondBouncePending = !build.failed;
        return 'gi1-rays';
    }

    /**
     * Empties the cache: the tile words (every hash slot, decay stamp, both
     * packed lists and the counters; the words' first section) back to zero, as `prepare` leaves them
     * (Capsaicin clears the hash buffer when it makes it, gi1.cpp:478, and in
     * `clearHashGridCache`, gi1.cpp:3286-3291). The cell values and scratch
     * stay: a tile's finest cells are cleared when it is inserted again
     * (`HashGridCache_AddNewTile`), its coarser levels are rewritten in the
     * same frame (`UpdateTiles`), and the direct scratch is zero between
     * frames (`UpdateTiles` clears what it reads, gi1.comp:2213-2224). The
     * indirect scratch is not: `UpdateMultibounceCells` fills it after
     * `UpdateTiles` for the next frame, so what the last frame left there is
     * blended into whichever new tile takes those cells, once. Capsaicin's
     * clear leaves it too. The cells' last-reset frames (E9, in the per-ray
     * lists) stay too.
     */
    reset(encoder: GPUCommandEncoder): void {
        const prepared = this.require();
        encoder.clearBuffer(prepared.buffers.tileWords, 0, prepared.layout.tileWords.bytes);
    }

    /**
     * Writes the frame's constants and records the cache's work for the
     * frame, in Capsaicin's order (see the file's comment): purge, insert the
     * probe rays' hits, light one sample per hit, update the tiles, resolve
     * into `rays.radiance`.
     */
    encodeFrame(encoder: GPUCommandEncoder, frame: WorldCacheFrame, rays: WorldCacheProbeRays): void {
        const prepared = this.require();
        // The multibounce runs if the option is on and the cache was prepared with it (register D67).
        const multibounce = this.multibounceThisFrame(prepared);
        // E10's estimator first: it decides the lists' size and the indirect cap in the constants.
        const secondBounce: SecondBounceInUse = multibounce ? this.resolveSecondBounce(prepared) : 'gi1-rays';
        // E9's frames are in the lists while E8 or E9 uses them (both claim a cell's reset through them).
        // The shared-cell second bounce lights cells on the budget whatever cellShading says.
        const budgetedCells = hashGridCellsBudgeted(this.options, secondBounce);
        const resetFrames = budgetedCells || this.options.cellChangeReset === true;
        const rayData = this.rayDataFor(encoder, prepared, rays, multibounce && secondBounce === 'gi1-rays', budgetedCells, resetFrames);
        // P6.4: the tail runs with GI-1.2's own rays only, once the second bounce has settled; its
        // restart (if any) comes before the constants, which read whether it is active.
        const tail = hashGridTailRuns(multibounce, secondBounce, this.secondBouncePending) ? this.tail : null;
        tail?.beginFrame(frame);
        this.writeConstants(prepared, frame, secondBounce, tail?.active ?? false, multibounce);

        // Capsaicin empties the cache when the cell size, the smallest cell size or multibounce
        // changes (needs_hash_grid_clear, gi1.cpp:1740-1745, cleared at 1786-1789): cells keyed at
        // the old sizes would otherwise live on beside the new ones until they decay. Its other two
        // reasons are the debug view's mip level and propagation (we have no such options), and its
        // first frame (our buffers start empty from prepare). Turning multibounce on or off also
        // remakes its indirect buffers, the scratch cleared (gi1.cpp:497-552).
        // Before the count below: emptying the cache zeroes the tile words it is copied into.
        // Experiment E10: switching the second-bounce estimator empties the cache as switching
        // multibounce does, so neither estimator's values linger under the other.
        const clearOptions = hashGridCacheClearOptions(this.options, secondBounce, multibounce);
        const last = this.lastClearOptions;
        if (last && !sameClearOptions(last, clearOptions)) {
            this.reset(encoder);
            const scratch = prepared.layout.updateScratch;
            const changedEstimator = last.useMultibounce !== clearOptions.useMultibounce || last.secondBounce !== clearOptions.secondBounce;
            if (changedEstimator && scratch.updateCellValueIndirectBuffer !== null) {
                const indirectStart = scratch.updateCellValueIndirectBuffer * WORD_BYTES;
                encoder.clearBuffer(prepared.buffers.tileWords, indirectStart, scratch.bytes / 2);
            }
        }
        this.lastClearOptions = clearOptions;
        const useMultibounce = clearOptions.useMultibounce;
        const budgeted = this.writeCellShading(encoder, prepared, rayData, frame.changes, secondBounce);
        this.lastSecondBounce = useMultibounce ? secondBounce : null;
        // GI-1.2's second-bounce rays, or experiment E10's lookup into the bounce source.
        const traceSecondBounce = useMultibounce && secondBounce === 'gi1-rays';
        const bounceSource = useMultibounce && secondBounce === 'fallback' ? (this.bounceSource?.pass ?? null) : null;

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
        if (traceSecondBounce) this.encodePopulationPass(encoder, prepared, rayData, 'populateMultibounceCells', 'PopulateMultibounceCells');
        this.encodePopulationPass(encoder, prepared, rayData, 'generateReservoirs', 'GenerateReservoirs');
        if (traceSecondBounce) this.encodePopulationPass(encoder, prepared, rayData, 'generateMultibounceReservoirs', 'GenerateMultibounceReservoirs');
        this.encodePopulationPass(encoder, prepared, rayData, 'populateCells', 'PopulateRadianceCache');
        if (budgeted) this.encodeCellShading(encoder, prepared, rayData, traceSecondBounce, secondBounce === 'shared-cell');
        // E10: after GenerateReservoirs (the first-bounce bypass flags) and PopulateCells (its
        // additions to bypassing probe rays), before UpdateTiles, which blends its samples this frame.
        if (bounceSource) this.encodeBounceSourcePass(encoder, prepared, bounceSource, rayData);
        this.encodePopulationPass(encoder, prepared, rayData, 'updateTiles', 'UpdateRadianceCache');
        // P6.4: the neural tail gathers its records and answers the second hits here, before
        // UpdateMultibounceCells reads them. It extends GI-1.2's second-bounce rays only: with the
        // second bounce from a fallback (E10) the fallback's field already carries the later bounces.
        if (traceSecondBounce && tail) {
            tail.encode(encoder, { frame, rays, rayLists: rayData.layout, listWords: rayData.listWords, cacheEntries: groupEntries(prepared.buffers), rayDataEntries: rayData.entries });
        } else if (useMultibounce && this.tail && secondBounce === 'fallback' && !this.tailInactiveWarned) {
            this.tailInactiveWarned = true;
            prepared.context.warn?.("@pmndrs/gi: the radiance cache's neuralCache 'tail' is off while its second bounce comes from the technique stacked under it (secondBounce 'auto' or 'fallback'), whose field already carries the later bounces; set secondBounce: 'gi1-rays' to use the tail");
        } else if (useMultibounce && this.tail && secondBounce === 'shared-cell' && !this.tailInactiveWarned) {
            this.tailInactiveWarned = true;
            prepared.context.warn?.("@pmndrs/gi: the radiance cache's neuralCache 'tail' is off with secondBounce 'shared-cell', which keeps no second-bounce entries for it to extend; set secondBounce: 'gi1-rays' to use the tail");
        }
        if (traceSecondBounce) this.encodePopulationPass(encoder, prepared, rayData, 'updateMultibounceCells', 'UpdateMultibounceCells');
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
        for (const populationPass of populationPassList(prepared.population)) populationPass.args.dispose();
        const cellShading = prepared.cellShading;
        prepared.cellShadingUniform.destroy();
        for (const cellPass of [cellShading.claimFirstBounceCells, cellShading.claimSecondBounceCells, cellShading.scatterCells, cellShading.shadeCells, cellShading.shortRayBounce]) cellPass?.args.dispose();
        disposeBounceSource(this.bounceSource);
        this.bounceSource = null;
        this.prepared = null;
    }

    /**
     * Lever E8 and experiment E9: writes this frame's way of lighting cells
     * into the ray lists' uniform (every population pass reads it) and, when
     * either is on, the budget and the change events into their uniform;
     * records clearing E8's per-frame counters. The cells' last-reset frames
     * and E8's lists are sections of the per-ray list words (`rayDataFor`
     * makes them with the lists while they are used). Returns whether the
     * cells are lit on the budget.
     */
    private writeCellShading(encoder: GPUCommandEncoder, prepared: Prepared, rayData: RayData, changes: WorldCacheChanges | undefined, secondBounce: SecondBounceInUse): boolean {
        const { device, buffers } = prepared;
        const options = this.options;
        const budgeted = hashGridCellsBudgeted(options, secondBounce);
        if (!budgeted && options.cellShading !== 'per-hit') throw new RangeError(`@pmndrs/gi: cellShading is 'per-hit' or 'budgeted', got ${String(options.cellShading)}`);
        const changeReset = !budgeted && options.cellChangeReset === true;
        const history = options.cellChangeResetHistory ?? 0;
        if (!Number.isInteger(history) || history < 0 || history > 15) throw new RangeError(`@pmndrs/gi: cellChangeResetHistory is a whole number of samples from 0 to 15, got ${history}`);
        const budget = budgeted ? options.cellShadingBudget : 0;
        if (!Number.isInteger(budget) || budget < 0 || budget > 0xffffffff) throw new RangeError(`@pmndrs/gi: cellShadingBudget is a whole number of cells, 0 or more, got ${budget}`);
        const flags = (budgeted ? CELL_SHADING_FLAG.budgeted : 0) | (changeReset ? CELL_SHADING_FLAG.changeReset | (history << CELL_SHADING_FLAG.historyShift) : 0);
        device.queue.writeBuffer(prepared.rayListsUniform, RAY_LISTS_CELL_SHADING_WORD * WORD_BYTES, new Uint32Array([flags]));
        if (!budgeted && !changeReset) return false;

        device.queue.writeBuffer(prepared.cellShadingUniform, 0, cellShadingWords(budget, changes));
        const cellLists = rayData.layout.cellLists;
        if (!budgeted) return false;
        if (!cellLists) throw new Error('@pmndrs/gi: the per-ray lists were made without lever E8\'s lists');
        // The candidate and lit counts and the priority bins' counts start at zero (the turn period is kept).
        encoder.clearBuffer(buffers.tileWords, HASH_GRID_COUNTER.cellCandidateCount * WORD_BYTES, 2 * WORD_BYTES);
        encoder.clearBuffer(rayData.listWords, cellLists.bins * WORD_BYTES, 3 * CELL_SHADING_BINS * WORD_BYTES);
        return true;
    }

    /** Lever E8's passes (cell-shading-wgsl.ts): claim, prioritise and scatter, light. */
    private encodeCellShading(encoder: GPUCommandEncoder, prepared: Prepared, rayData: RayData, useMultibounce: boolean, sharedBounce: boolean): void {
        const cellShading = prepared.cellShading;
        const indirect = (pass: GPUComputePassEncoder, name: Exclude<CellShadingPassName, 'prioritizeCells'>): void => {
            const cellPass = cellShading[name];
            const group = rayData.groups[name];
            if (!cellPass || !group) throw new Error(`@pmndrs/gi: the world cache was prepared without ${name} (useMultibounce off at prepare)`);
            cellPass.args.encode(pass);
            cellPass.kernel.bind(pass);
            pass.setBindGroup(HASH_GRID_KERNEL_GROUP, cellPass.cacheGroup);
            pass.setBindGroup(RAY_DATA_GROUP, group);
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
        prioritize.dispatchWorkgroups(1);
        indirect(prioritize, 'scatterCells');
        prioritize.end();

        if (sharedBounce) {
            const bounce = this.beginPass(encoder, prepared, 'SharedCellBounce');
            indirect(bounce, 'sharedCellBounce');
            indirect(bounce, 'shortRayBounce');
            bounce.end();
        }
        const shade = this.beginPass(encoder, prepared, 'ShadeCells');
        indirect(shade, 'shadeCells');
        shade.end();
    }

    /** One of Capsaicin's timed sections as a compute pass: its indirect arguments from the GPU count, then the kernel. */
    private encodePopulationPass(encoder: GPUCommandEncoder, prepared: Prepared, rayData: RayData, name: PopulationPassName, section: string): void {
        const populationPass = prepared.population[name];
        const group = rayData.groups[name];
        if (!populationPass || !group) throw new Error(`@pmndrs/gi: the world cache was prepared without ${section} (useMultibounce off at prepare)`);
        const pass = this.beginPass(encoder, prepared, section);
        populationPass.args.encode(pass);
        populationPass.kernel.bind(pass);
        pass.setBindGroup(HASH_GRID_KERNEL_GROUP, populationPass.cacheGroup);
        pass.setBindGroup(RAY_DATA_GROUP, group);
        pass.dispatchWorkgroupsIndirect(populationPass.args.buffer, 0);
        pass.end();
    }

    /** Experiment E10's `SampleBounceSourceCells`, with the fallback's resources in group 3. */
    private encodeBounceSourcePass(encoder: GPUCommandEncoder, prepared: Prepared, bounceSource: BounceSourcePass, rayData: RayData): void {
        if (rayData.bounceSourceGroup?.pass !== bounceSource) {
            const group = prepared.device.createBindGroup({ label: 'radiance-cache/SampleBounceSourceCells ray data', layout: bounceSource.rayDataLayout, entries: rayData.entries });
            rayData.bounceSourceGroup = { pass: bounceSource, group };
        }
        const rayDataGroup = rayData.bounceSourceGroup.group;
        const bounceGroup = bounceSource.bindings.bindGroup();
        const pass = this.beginPass(encoder, prepared, 'SampleBounceSourceCells');
        bounceSource.args.encode(pass);
        bounceSource.kernel.bind(pass);
        pass.setBindGroup(HASH_GRID_KERNEL_GROUP, bounceSource.cacheGroup);
        pass.setBindGroup(RAY_DATA_GROUP, rayDataGroup);
        pass.setBindGroup(BOUNCE_SOURCE_GROUP, bounceGroup);
        pass.dispatchWorkgroupsIndirect(bounceSource.args.buffer, 0);
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
    private rayDataFor(encoder: GPUCommandEncoder, prepared: Prepared, rays: WorldCacheProbeRays, secondBounceRays: boolean, cellShadingLists: boolean, resetFrames: boolean): RayData {
        for (const buffer of this.retiredListWords) buffer.destroy();
        this.retiredListWords = [];
        const current = this.rayData;
        // E10: without second-bounce rays the lists are about a third the size (register E10); made again on a switch.
        // P6.4: with a tail the lists carry its answers (`neuralTailBuffer`), made again when one is set.
        // E8: while cells are lit on a budget the lists end with E8's (register D6), made again on a switch.
        // E9: while E8 or E9 is used the lists end with the cells' last-reset frames, made again on a switch.
        const withTail = this.tail !== null && secondBounceRays;
        if (
            current &&
            sameRays(current.rays, rays) &&
            current.secondBounceRays === secondBounceRays &&
            (current.layout.neuralTailBuffer !== 0) === withTail &&
            (current.layout.cellLists !== null) === cellShadingLists &&
            (current.layout.resetFrames !== null) === resetFrames
        ) {
            return current;
        }
        const { device } = prepared;
        const layout = hashGridCacheRayListLayout(rays.capacity, prepared.capacity.numCells, secondBounceRays, withTail, cellShadingLists, resetFrames ? cellResetFrameWords(prepared.capacity) : 0);
        // The lists grow with the screen: a device that can't bind them gets GILimitError before anything is made (D14).
        checkHashGridRayListLimits(layout, device.limits);
        // copyDst: E8's bins are cleared each budgeted frame (encoder.clearBuffer).
        const listWords = device.createBuffer({ label: 'radiance-cache/hash-grid per-ray lists', size: layout.bytes, usage: BUFFER_USAGE.storage | BUFFER_USAGE.copySrc | BUFFER_USAGE.copyDst });
        // E9's frames belong to the cache, not the screen: they move to the new lists (a new
        // section starts at zero, as the cache's first frames did). The old lists are destroyed
        // next frame, after the host has submitted this copy.
        const oldFrames = current?.layout.resetFrames;
        if (current && oldFrames && layout.resetFrames) {
            encoder.copyBufferToBuffer(current.listWords, oldFrames.firstWord * WORD_BYTES, listWords, layout.resetFrames.firstWord * WORD_BYTES, layout.resetFrames.words * WORD_BYTES);
            this.retiredListWords.push(current.listWords);
            this.rayData = null;
        } else {
            this.releaseRayData();
        }
        const cellLists = layout.cellLists;
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
                // E8's lists, at the end of the list words while it is used (0: none).
                cellLists?.candidates ?? 0,
                cellLists?.lit ?? 0,
                0, // cell_shading: written every frame (writeCellShading)
                cellLists?.bins ?? 0,
                layout.neuralTailBuffer,
                // E9's frames, at the end while E8 or E9 is used (0: none; nothing reads them then).
                layout.resetFrames?.firstWord ?? 0,
            ]),
        );
        const rayData: RayData = {
            rays,
            layout,
            listWords,
            groups: {} as RayData['groups'],
            entries: [],
            bounceSourceGroup: null,
            secondBounceRays,
        };
        rayData.groups = this.rayDataGroups(prepared, rayData);
        this.rayData = rayData;
        return rayData;
    }

    /** The ray-data groups of every population pass that was made, over this screen's lists and the cache's change data. */
    private rayDataGroups(prepared: Prepared, rayData: RayData): RayDataGroups {
        const { device } = prepared;
        const rays = rayData.rays;
        const entries: GPUBindGroupEntry[] = [
            { binding: RAY_DATA_BINDING.rayLists, resource: { buffer: prepared.rayListsUniform } },
            { binding: RAY_DATA_BINDING.probeRays, resource: { buffer: rays.buffer } },
            { binding: RAY_DATA_BINDING.radiance, resource: rays.radiance },
            { binding: RAY_DATA_BINDING.listWords, resource: { buffer: rayData.listWords } },
            { binding: CELL_CHANGES_BINDING.uniform, resource: { buffer: prepared.cellShadingUniform } },
        ];
        // E10's pass makes its own group over these entries; it is made again with the new ones.
        rayData.entries = entries;
        rayData.bounceSourceGroup = null;
        const groupFor = (kernel: ComputeKernel): GPUBindGroup => device.createBindGroup({ label: `${kernel.pipeline.label} ray data`, layout: kernel.ownLayouts[1], entries });
        const population = prepared.population;
        const cellShading = prepared.cellShading;
        // The second-bounce passes' groups only where the passes were made (register D67).
        const optionalGroupFor = (populationPass: PopulationPass | undefined): GPUBindGroup | undefined => (populationPass ? groupFor(populationPass.kernel) : undefined);
        return {
            populateScreenProbes: groupFor(population.populateScreenProbes.kernel),
            populateMultibounceCells: optionalGroupFor(population.populateMultibounceCells),
            generateReservoirs: groupFor(population.generateReservoirs.kernel),
            generateMultibounceReservoirs: optionalGroupFor(population.generateMultibounceReservoirs),
            populateCells: groupFor(population.populateCells.kernel),
            updateTiles: groupFor(population.updateTiles.kernel),
            updateMultibounceCells: optionalGroupFor(population.updateMultibounceCells),
            resolveCells: groupFor(population.resolveCells.kernel),
            claimFirstBounceCells: groupFor(cellShading.claimFirstBounceCells.kernel),
            claimSecondBounceCells: groupFor(cellShading.claimSecondBounceCells.kernel),
            prioritizeCells: groupFor(cellShading.prioritizeCells.kernel),
            scatterCells: groupFor(cellShading.scatterCells.kernel),
            shadeCells: groupFor(cellShading.shadeCells.kernel),
            sharedCellBounce: optionalGroupFor(cellShading.sharedCellBounce),
            shortRayBounce: optionalGroupFor(cellShading.shortRayBounce),
        };
    }

    private releaseRayData(): void {
        this.rayData?.listWords.destroy();
        this.rayData = null;
        for (const buffer of this.retiredListWords) buffer.destroy();
        this.retiredListWords = [];
    }

    /** `HashGridCacheConstants` for this frame (gi1.cpp:1852-1893); `tailActive`: P6.4's tail adds its answers this frame (`use_neural_tail`). */
    private writeConstants(prepared: Prepared, frame: WorldCacheFrame, secondBounce: SecondBounceInUse, tailActive: boolean, multibounce: boolean): void {
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
        // E10: the fallback's value has no ray noise, so its own, shorter running mean (register E10).
        floats[word.maxMultibounceSampleCount] = secondBounce === 'fallback' ? options.fallbackIndirectSampleCount : options.hashGridCacheMaxMultibounceSampleCount;
        words[word.useDirectLighting] = options.useDirectLighting ? 1 : 0;
        words[word.useMultibounce] = multibounce ? 1 : 0;
        // RC3.4's sky light (register D62), left out in E10's fallback mode: the fallback's field
        // already holds the sky that reaches a hit (radiance-cache-ddgi-bounce's sky section).
        words[word.environmentLightEnable] = hashGridCacheSamplesSky(options, secondBounce) ? 1 : 0;
        words[word.useNeuralTail] = tailActive ? 1 : 0;
        prepared.device.queue.writeBuffer(prepared.buffers.constants, 0, bytes);
    }

    /**
     * Whether this frame runs GI-1.2's multibounce: the option, if the cache
     * was prepared with it (only then do the indirect values exist and the
     * kernels read them; register D67). Turned on after a `prepare` without
     * it, the cache runs one bounce until the host prepares again, and says
     * so once.
     */
    private multibounceThisFrame(prepared: Prepared): boolean {
        if (!this.options.useMultibounce) return false;
        if (prepared.layout.multibounce) return true;
        if (!this.multibounceNeedsPrepareWarned) {
            this.multibounceNeedsPrepareWarned = true;
            prepared.context.warn?.('@pmndrs/gi: useMultibounce was turned on after the radiance cache was prepared without it; it takes effect at the next GIPlatform.prepare() (until then the world cache runs one bounce)');
        }
        return false;
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
    /** Experiment E10: the estimator that ran (not the one asked for), so the cache empties when it really changes. */
    readonly secondBounce: SecondBounceInUse;
}

/**
 * Whether the cache's own light choice takes the sky (RC3.4's
 * `environment_light_enable`, `GI_DIRECT_SKY_LIGHT_BIT`): as its option says,
 * except with the second bounce from a fallback, whose field already holds the
 * sky reaching a hit (register E10). Emissive surfaces as cache lights (D21's
 * way back) will need the same rule.
 */
export function hashGridCacheSamplesSky(options: Pick<HashGridCacheOptions, 'environmentLightEnable'>, secondBounce: SecondBounceInUse): boolean {
    return options.environmentLightEnable && secondBounce !== 'fallback';
}

/**
 * Whether a tail (P6.4) runs this frame, and so may set `use_neural_tail`:
 * with GI-1.2's own second-bounce rays only (multibounce on, the second
 * bounce resolved to `'gi1-rays'`), and not while a fallback's pass is still
 * being built (`pending`: `'auto'` would switch to it within a few frames, and
 * the tail would have trained on the frames before).
 */
export function hashGridTailRuns(useMultibounce: boolean, secondBounce: SecondBounceInUse, pending: boolean): boolean {
    return useMultibounce && secondBounce === 'gi1-rays' && !pending;
}

/**
 * Whether cells are lit on the budget this frame (lever E8): `cellShading:
 * 'budgeted'`, or the shared-cell second bounce (pmndrs/gi#54), which traces
 * its one ray from each lit cell's sample hit. `secondBounce` is the estimator
 * that runs: with multibounce off it is never `'shared-cell'`.
 */
export function hashGridCellsBudgeted(options: Pick<HashGridCacheOptions, 'cellShading'>, secondBounce: SecondBounceInUse): boolean {
    return options.cellShading === 'budgeted' || secondBounce === 'shared-cell';
}

/** `useMultibounce`: whether the frame runs it (the option, if the cache was prepared with it; register D67). */
export function hashGridCacheClearOptions(options: HashGridCacheOptions, secondBounce: SecondBounceInUse = 'gi1-rays', useMultibounce = options.useMultibounce): HashGridCacheClearOptions {
    return { cellSize: options.hashGridCacheCellSize, minCellSize: options.hashGridCacheMinCellSize, useMultibounce, secondBounce };
}

/** The passes that were made (the second-bounce ones only with `useMultibounce` at `prepare`). */
function populationPassList(passes: PopulationPasses): PopulationPass[] {
    return Object.values(passes).filter((populationPass): populationPass is PopulationPass => populationPass !== undefined);
}

/** The estimator only matters with multibounce on: with it off, neither runs. */
export function sameClearOptions(a: HashGridCacheClearOptions, b: HashGridCacheClearOptions): boolean {
    return a.cellSize === b.cellSize && a.minCellSize === b.minCellSize && a.useMultibounce === b.useMultibounce && (!a.useMultibounce || a.secondBounce === b.secondBounce);
}

/** The fallback's shading module, when one is stacked and answers in world space (E10). */
function worldSpaceFallback(context: WorldCacheContext): ShadingModule | undefined {
    const module = context.fallback?.shading;
    return module?.provides.worldSpaceIrradiance === true ? module : undefined;
}

/** Experiment E10's pass for `module`: its kernel includes the module's globals in group 3. Rejects on a compile error. */
async function compileBounceSource(prepared: Prepared, module: ShadingModule): Promise<BounceSourcePass> {
    const { device, kernels, compute, shade } = prepared.context;
    const bounceGroup: GPUBindGroupLayoutDescriptor = { label: `radiance-cache/bounce source (${module.techniqueId})`, entries: shadingModuleLayoutEntries(module) };
    const [kernel, args] = await Promise.all([
        kernels.computeKernel({
            label: 'radiance-cache/SampleBounceSourceCells',
            modules: [shade.module],
            code: sampleBounceSourceCellsWgsl(prepared.layout, module),
            entryPoint: SAMPLE_BOUNCE_SOURCE_ENTRY_POINT,
            ownGroups: [prepared.cacheGroupDescriptor, prepared.rayDataGroupDescriptor, bounceGroup],
        }),
        compute.createIndirectArgs({ label: 'radiance-cache/SampleBounceSourceCells args', count: prepared.firstBounceCount, itemsPerWorkgroup: SAMPLE_BOUNCE_SOURCE_WORKGROUP_SIZE }),
    ]);
    return {
        kernel,
        args,
        cacheGroup: device.createBindGroup({ label: 'radiance-cache/SampleBounceSourceCells hash grid', layout: kernel.ownLayouts[0], entries: groupEntries(prepared.buffers) }),
        rayDataLayout: kernel.ownLayouts[1],
        bindings: new ShadingModuleBindings(device, module, kernel.ownLayouts[2]),
    };
}

function disposeBounceSource(build: BounceSourceBuild | null): void {
    build?.pass?.args.dispose();
    build?.pass?.bindings.dispose();
}

/**
 * `HashGridCacheCellShading` (cell-shading-common.ts): E8's budget and the
 * change events, each with its box (a change with no bounds as the whole
 * world) and its current epoch plus one. At most
 * `CELL_SHADING_MAX_CHANGE_EVENTS`; the technique's tracker keeps fewer.
 */
export function cellShadingWords(budget: number, changes: WorldCacheChanges | undefined): ArrayBuffer {
    const bytes = new ArrayBuffer(CELL_SHADING_UNIFORM_BYTES);
    const words = new Uint32Array(bytes);
    const floats = new Float32Array(bytes);
    const events = changes?.events ?? [];
    if (events.length > CELL_SHADING_MAX_CHANGE_EVENTS) throw new RangeError(`@pmndrs/gi: the world cache takes at most ${CELL_SHADING_MAX_CHANGE_EVENTS} change events a frame, got ${events.length}`);
    words[0] = budget;
    words[1] = events.length;
    const everywhere = 3e38;
    events.forEach((event, index) => {
        const base = 4 + index * 8;
        const box = event.box;
        floats.set(box ? [box.min[0], box.min[1], box.min[2]] : [-everywhere, -everywhere, -everywhere], base);
        words[base + 3] = (event.epoch + 1) >>> 0;
        floats.set(box ? [box.max[0], box.max[1], box.max[2]] : [everywhere, everywhere, everywhere], base + 4);
    });
    return bytes;
}

function groupEntries(buffers: HashGridCacheBuffers): GPUBindGroupEntry[] {
    return [
        { binding: HASH_GRID_CACHE_BINDING.constants, resource: { buffer: buffers.constants } },
        { binding: HASH_GRID_CACHE_BINDING.tileWords, resource: { buffer: buffers.tileWords } },
        { binding: HASH_GRID_CACHE_BINDING.cellValues, resource: { buffer: buffers.cellValues } },
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
