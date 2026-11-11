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
    UPDATE_SCRATCH_WORDS_PER_CELL,
} from './capacity.ts';
import { hashGridCacheCellSizeConstant, verticalFieldOfView } from './hash-grid-cpu.ts';
import {
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

/** GI-1.2's second-bounce passes (RC3.3), made only when the cache is prepared with `useMultibounce` (register D42). */
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
        return {
            buckets: capacity.numBuckets,
            'tile slots per bucket': capacity.numTilesPerBucket,
            tiles: capacity.numTiles,
            'cells per tile': capacity.numCellsPerTile,
            cells: capacity.numCells,
            'per-ray lists (bytes)': this.rayData?.listWords.size ?? 0,
            // 1 when prepared with useMultibounce: the indirect values and their scratch exist (register D42).
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

    async prepare(context: WorldCacheContext): Promise<void> {
        const { device, kernels, compute, shade } = context;
        const capacity = hashGridCacheCapacity(this.options);
        // Capsaicin's USE_MULTI_BOUNCE: without it the indirect arrays aren't made (gi1.cpp:497-552)
        // and the kernels are compiled without them (gi1.cpp:1428-1431).
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
                entryPoint: POPULATION_ENTRY_POINT[name],
                ownGroups: [cacheGroup(POPULATION_BINDINGS[name].cache), rayDataGroup(POPULATION_BINDINGS[name].rayData)],
            });
        const indirectArgs = (label: string, counter: number, itemsPerWorkgroup: number): Promise<GpuIndirectArgs> =>
            compute.createIndirectArgs({ label: `radiance-cache/${label}`, count: word(counter), itemsPerWorkgroup });

        // GI-1.2's three passes and their arguments, only with multibounce (register D42).
        const multibounceWork = multibounce
            ? Promise.all([
                  populationKernel('PopulateMultibounceCells', 'populateMultibounceCells', populateMultibounceCellsWgsl(layout), true),
                  populationKernel('GenerateMultibounceReservoirs', 'generateMultibounceReservoirs', generateMultibounceReservoirsWgsl(layout), true),
                  populationKernel('UpdateMultibounceCells', 'updateMultibounceCells', updateMultibounceCellsWgsl(layout), false),
                  // generateDispatch(radiance_cache_visibility_count_buffer0_, ...) (gi1.cpp:2389)
                  indirectArgs('PopulateMultibounceCells args', HASH_GRID_COUNTER.visibilityCount0, POPULATION_WORKGROUP_SIZE.populateMultibounceCells),
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
            populationKernel('GenerateReservoirs', 'generateReservoirs', generateReservoirsWgsl(layout), true),
            populationKernel('PopulateCells', 'populateCells', populateCellsWgsl(layout), true),
            populationKernel('UpdateTiles', 'updateTiles', updateTilesWgsl(layout), false),
            populationKernel('ResolveCells', 'resolveCells', resolveCellsWgsl(layout), false),
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
        const kernelCount = 5;
        const [populateScreenProbes, generateReservoirs, populateCells, updateTiles, resolveCells] = population.slice(0, kernelCount) as ComputeKernel[];
        const [populateArgs, reservoirArgs, cellArgs, updateArgs, resolveArgs] = population.slice(kernelCount) as GpuIndirectArgs[];
        const pass = (name: PopulationPassName, kernel: ComputeKernel, args: GpuIndirectArgs): PopulationPass => ({
            kernel,
            args,
            cacheGroup: device.createBindGroup({
                label: `${kernel.pipeline.label} hash grid`,
                layout: kernel.ownLayouts[0],
                entries: groupEntries(buffers, POPULATION_BINDINGS[name].cache),
            }),
        });
        const multibouncePasses: Partial<MultibouncePasses> = multibounceMade
            ? {
                  populateMultibounceCells: pass('populateMultibounceCells', multibounceMade[0], multibounceMade[3]),
                  generateMultibounceReservoirs: pass('generateMultibounceReservoirs', multibounceMade[1], multibounceMade[4]),
                  updateMultibounceCells: pass('updateMultibounceCells', multibounceMade[2], multibounceMade[5]),
              }
            : {};
        const populationPasses: PopulationPasses = {
            populateScreenProbes: pass('populateScreenProbes', populateScreenProbes, populateArgs),
            generateReservoirs: pass('generateReservoirs', generateReservoirs, reservoirArgs),
            populateCells: pass('populateCells', populateCells, cellArgs),
            updateTiles: pass('updateTiles', updateTiles, updateArgs),
            resolveCells: pass('resolveCells', resolveCells, resolveArgs),
            ...multibouncePasses,
        };
        const rayListsUniform = device.createBuffer({ label: 'radiance-cache/hash-grid ray lists', size: RAY_LISTS_UNIFORM_BYTES, usage: BUFFER_USAGE.uniform | BUFFER_USAGE.copyDst });

        const purgeArgs = [purgeArgs0, purgeArgs1] as const;
        const argsBytes = [...purgeArgs, ...populationPassList(populationPasses).map((populationPass) => populationPass.args)].reduce((sum, args) => sum + args.memoryBytes, 0);
        const memoryBytes = layout.totalBytes + HASH_GRID_CONSTANTS_BYTES + RAY_LISTS_UNIFORM_BYTES + argsBytes;
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
        if (useMultibounce) this.encodePopulationPass(encoder, prepared, rayData, 'populateMultibounceCells', 'PopulateMultibounceCells');
        this.encodePopulationPass(encoder, prepared, rayData, 'generateReservoirs', 'GenerateReservoirs');
        if (useMultibounce) this.encodePopulationPass(encoder, prepared, rayData, 'generateMultibounceReservoirs', 'GenerateMultibounceReservoirs');
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
        prepared.rayListsUniform.destroy();
        for (const args of prepared.purgeArgs) args.dispose();
        for (const populationPass of populationPassList(prepared.population)) populationPass.args.dispose();
        this.prepared = null;
    }

    /** One of Capsaicin's timed sections as a compute pass: its indirect arguments from the GPU count, then the kernel. */
    private encodePopulationPass(encoder: GPUCommandEncoder, prepared: Prepared, rayData: RayData, name: PopulationPassName, section: string): void {
        const populationPass = prepared.population[name];
        const group = rayData.groups[name];
        if (!populationPass || !group) throw new Error(`@pmndrs/ddgi: the world cache was prepared without ${section} (useMultibounce off)`);
        const pass = this.beginPass(encoder, prepared, section);
        populationPass.args.encode(pass);
        populationPass.kernel.bind(pass);
        pass.setBindGroup(HASH_GRID_KERNEL_GROUP, populationPass.cacheGroup);
        pass.setBindGroup(RAY_DATA_GROUP, group);
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
        const layout = hashGridCacheRayListLayout(rays.capacity, prepared.capacity.numCells, prepared.layout.multibounce);
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
            ]),
        );
        /** The ray-data group's entries for the buffers a pass binds (POPULATION_BINDINGS). */
        const entries = (set: RayDataBindingSet): GPUBindGroupEntry[] => {
            const bound: GPUBindGroupEntry[] = [{ binding: RAY_DATA_BINDING.rayLists, resource: { buffer: prepared.rayListsUniform } }];
            if (set.probeRays) bound.push({ binding: RAY_DATA_BINDING.probeRays, resource: { buffer: rays.buffer } });
            if (set.radiance) bound.push({ binding: RAY_DATA_BINDING.radiance, resource: rays.radiance });
            bound.push({ binding: RAY_DATA_BINDING.listWords, resource: { buffer: listWords } });
            return bound;
        };
        const groups: Partial<Record<PopulationPassName, GPUBindGroup>> = {};
        for (const name of Object.keys(prepared.population) as PopulationPassName[]) {
            const populationPass = prepared.population[name];
            if (!populationPass) continue;
            groups[name] = device.createBindGroup({
                label: `${populationPass.kernel.pipeline.label} ray data`,
                layout: populationPass.kernel.ownLayouts[1],
                entries: entries(POPULATION_BINDINGS[name].rayData),
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
