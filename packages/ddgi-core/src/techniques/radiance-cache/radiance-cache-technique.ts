/**
 * The radiance cache: a faithful WebGPU port of AMD's GI-1.0/1.2 as Capsaicin
 * v1.3 builds it (docs/design/radiance-cache.md). The screen probes, traced
 * every frame (RC2.1) and kept over frames by reprojection and the off-screen
 * probe cache (RC4.1), their hits lit through the world cache (RC3.2, RC3.3),
 * then projected to SH and interpolated per pixel into irradiance the shading
 * module reads by screen position (RC2.2, screen-irradiance.ts), with their
 * debug views.
 *
 * Per frame, in `encodeScreen` (after the host's `encode` and its depth and
 * normal pass), Capsaicin's SSGI technique, then the passes it runs for the
 * screen probes (`GI1::render`, gi1.cpp:2250-2636), in its order:
 *
 * 0. `SSGI` (ssgi.comp:132; ssgi.ts): occlusion, bent normal and near-field
 *    light from the host's depth, normals and last frame's lit colour
 *    (Capsaicin's renderer runs the SSGI technique before GI-1,
 *    gi1_renderer.cpp:60-61; RC4.2);
 * 1. `ClearCounters` (gi1.comp:163);
 * 2. `ReprojectScreenProbes` (gi1.comp:292), then the probe cache's lookup:
 *    `CountScreenProbes` (gi1.comp:501), the platform's scan and
 *    `ScatterScreenProbes` (gi1.comp:541), dispatched on the listed count
 *    (gi1.cpp:2256-2290; wgsl/screen-probe-history.ts);
 * 3. `SpawnScreenProbes` (gi1.comp:555), the platform's stream compaction
 *    (Capsaicin's `gfxCommandScanSum`, gi1.cpp:2305) and `CompactScreenProbes`
 *    (gi1.comp:581), dispatched indirectly on the kept count, then
 *    `PatchScreenProbes` (gi1.comp:615) on the empty-tile count;
 * 4. `SampleScreenProbes` (gi1.comp:646) and `PopulateScreenProbesMain`
 *    (gi1.comp:1160);
 * 5. the world cache (`world-cache/`): Capsaicin's hash grid inserts the
 *    hits' cells, lights one light sample per hit, updates its tiles and
 *    resolves its radiance into the probe rays (gi1.cpp:2238-2253 and
 *    2335-2528, here after the trace: `encodeWorldCache`);
 * 6. `BlendScreenProbes` (gi1.comp:1167) into the probe atlas, the mask and
 *    the probe cache, and the atlas copied into last frame's (gi1.cpp:2533-2542);
 * 7. the platform's scan of the LRU flags and `ReorderScreenProbes`
 *    (gi1.comp:1334, gi1.cpp:2544-2556);
 * 8. `FilterProbeMask` (gi1.comp:249), `FilterScreenProbes` twice
 *    (gi1.comp:1357; RC4.2), `ProjectScreenProbes` (gi1.comp:1446) and
 *    `InterpolateScreenProbes` (gi1.comp:1515), the last with SSGI's outputs:
 *    screen-irradiance.ts (gi1.cpp:2561-2636); then the image-space GI
 *    denoiser, `ReprojectGI` and `FilterGI` ×2 (gi1.cpp:2882-2927;
 *    gi-denoiser.ts, `@pmndrs/denoiser-gi1`, RC4.3), whose output the shading
 *    module reads;
 * 9. the history kept for the next frame: the filtered atlas (which
 *    Capsaicin's flip makes the next frame's previous one, gi1.cpp:2606-2610,
 *    3097), the mask, the SH, the host's depth and normals, and the camera
 *    (Capsaicin swaps its buffers and its framework copies the depth and
 *    normals; registers D45, D49).
 *
 * Each of Capsaicin's timed sections is a compute pass of its own, timed as
 * `radiance-cache/<section>` (timed-pass.ts, RC5.1), so `GIStats.passes`
 * lines up with native Capsaicin's per-pass CSV.
 *
 * Capsaicin dispatches the per-ray passes over every possible ray
 * (`max_ray_count`, gi1.cpp:2323-2542) with an early return past the probe
 * count, and so do we: they are not sized on the GPU in Capsaicin.
 *
 * Resources that depend on the screen's size are made when the size changes,
 * as Capsaicin's `ensureMemoryIsAllocated` does (gi1.cpp:113); that makes
 * buffers and textures, never pipelines, and starts the history afresh (both
 * probe masks cleared, the LRU filled, gi1.cpp:146-180, 335-349). Everything
 * the spawn tiles and the platform's scans need is made in `prepare`, for the
 * largest screen the device can draw.
 */

import type { GI1Denoiser } from '@pmndrs/denoiser-gi1';
import type { GpuCompaction, GpuIndirectArgs, GpuPrefixScan } from '../../platform/contract/compute.ts';
import { GILimitError } from '../../platform/contract/errors.ts';
import type { FrameCamera, FrameScreen } from '../../platform/contract/frame.ts';
import type { RayScratch } from '../../platform/contract/shade.ts';
import type { DebugView } from '../../platform/contract/stats.ts';
import type { GITechnique, TechniqueContext, TechniqueFrame, TechniqueRequirements, TechniqueScreenFrame } from '../../platform/contract/technique.ts';
import type { GpuTimerScope } from '../../platform/contract/timing.ts';
import type { ComputeKernel, WgslModule } from '../../platform/contract/wgsl.ts';
import { cameraRelativeInverseViewProjection, cameraRelativeViewProjection } from './camera.ts';
import { largestProbeTileCount, probeCacheLayout, type ProbeCacheLayout } from './probe-cache.ts';
import { ScreenIrradiance, screenProbeCellSize } from './screen-irradiance.ts';
import { DEFAULT_RADIANCE_CACHE_SETTINGS, PROBE_SPAWN_TILE_SIZE, screenProbeLayout, type RadianceCacheSettings, type ScreenProbeLayout } from './settings.ts';
import { RadianceCacheShadingModule } from './shading.ts';
import { ScreenSpaceGI } from './ssgi.ts';
import { createGIDenoiser, encodeGIDenoiser } from './gi-denoiser.ts';
import { beginTimedPass, KEEP_HISTORY_SECTION, POPULATE_TRACE_SECTION } from './timed-pass.ts';
import { HashGridCache } from './world-cache/hash-grid-cache.ts';
import type { WorldCache, WorldCacheProbeRays } from './world-cache/world-cache.ts';
import { DEBUG_ENTRY_POINT, DEBUG_GROUP, DEBUG_UNIFORM_BYTES, DEBUG_UNIFORM_OFFSETS, DEBUG_WGSL, RADIANCE_CACHE_DEBUG_VIEW, type RadianceCacheDebugViewId } from './wgsl/debug.ts';
import { HISTORY_ENTRY_POINT, HISTORY_KERNELS, HISTORY_WORKGROUP_SIZE, PREVIOUS_DEPTH_FORMAT, PREVIOUS_NORMAL_FORMAT } from './wgsl/screen-probe-history.ts';
import {
    BLEND_GROUP,
    BLEND_WGSL,
    CLEAR_COUNTERS_GROUP,
    CLEAR_COUNTERS_WGSL,
    CLEAR_PROBE_MASK_GROUP,
    CLEAR_PROBE_MASK_WGSL,
    COMPACT_GROUP,
    COMPACT_WGSL,
    counterByteOffset,
    COUNTERS_BYTES,
    ENTRY_POINT,
    FRAME_UNIFORM_BYTES,
    FRAME_UNIFORM_OFFSETS,
    POPULATE_GROUP,
    populateWgsl,
    PROBE_RAY_BYTES,
    PROBE_RAYS_HEADER_BYTES,
    PROBE_SPAWN_SEEDS_OFFSET,
    SAMPLE_GROUP,
    SAMPLE_WGSL,
    SPAWN_GROUP,
    SPAWN_WGSL,
    WORKGROUP_SIZE,
} from './wgsl/screen-probes.ts';

const BUFFER_USAGE = { storage: 0x80, copySrc: 0x04, copyDst: 0x08, uniform: 0x40 } as const;
const TEXTURE_USAGE = { copySrc: 0x01, copyDst: 0x02, textureBinding: 0x04, storageBinding: 0x08 } as const;
const WORD_BYTES = 4;
/** Bytes of an `rgba16float` texel (the probe atlases). */
const HALF4_TEXEL_BYTES = 8;
/** Bytes of an `rgba32uint` texel (the probe cache's index). */
const UINT4_TEXEL_BYTES = 16;

/** The technique's id: its passes are timed as `radiance-cache/<pass>`. */
const TECHNIQUE_ID = 'radiance-cache';

/** The GPU resources of the probes for one screen size: what tests and debug views read. */
export interface RadianceCacheProbes {
    readonly layout: ScreenProbeLayout;
    /** The frame index the latest `encodeScreen` spawned with (the Halton jitter's index). */
    readonly frameIndex: number;
    /** `g_ScreenProbes_ProbeBuffer`: `rgba16float`, the atlas (8 × 8 cells per probe tile): radiance, hit distance. */
    readonly probeAtlas: GPUTexture;
    /** `g_ScreenProbes_ProbeMaskBuffer`: `r32uint` with a mip chain; level 0 holds each tile's packed seed or 0xffffffff. */
    readonly probeMask: GPUTexture;
    /** One `u32` per spawn tile: 1 when the tile spawned a probe (`g_ScreenProbes_ProbeSpawnScanBuffer`). */
    readonly spawnFlags: GPUBuffer;
    /** One packed seed per spawn tile (`g_ScreenProbes_PreviousProbeSpawnBuffer`); written for spawned tiles only. */
    readonly spawnSeeds: GPUBuffer;
    /** The compacted (and patched) probes: word 0 the count, then one packed seed each from byte `PROBE_SPAWN_SEEDS_OFFSET`. */
    readonly probeSpawn: GPUBuffer;
    /**
     * The probe rays (`RcProbeRays`, wgsl/screen-probes.ts): the ray count in word 0, then from byte
     * `PROBE_RAYS_HEADER_BYTES` one `PROBE_RAY_BYTES` record per ray: origin, flags, direction as three
     * halves (`g_ScreenProbes_ProbeSpawnSampleBuffer`) and the hit record.
     */
    readonly probeRays: GPUBuffer;
    /** The ray results (radiance and hit distance as four halves): `rayScratch.rayResults`. */
    readonly rayScratch: RayScratch;
    /** Each probe tile's nine SH coefficients (`packHalf4`), in a 3 × 3 block of this `rg32uint` texture (register D26). */
    readonly probeSH: GPUTexture;
    /** The interpolated irradiance, screen-sized `rgba16float`: rgb E, alpha Capsaicin's denoiser hint. */
    readonly irradiance: GPUTexture;
    /** SSGI's outputs (ssgi.ts), screen-sized `rgba16float`: rgb the bent normal as 0.5 n + 0.5 and a the occlusion; rgb the near-field light. */
    readonly occlusionAndBentNormal: GPUTexture;
    readonly nearField: GPUTexture;
}

/** The screen probes' history and the off-screen probe cache (RC4.1): what tests read. */
export interface RadianceCacheProbeHistory {
    /** `RcCounters` (`COUNTER_NAMES`): this frame's empty, override, LRU, MRU and list-element counts. */
    readonly counters: GPUBuffer;
    /**
     * `g_ScreenProbes_PreviousProbeBuffer`: after a frame, the copy of the filtered atlas the next
     * frame reprojects from (`COPY_DST`: a test may write a history of its own into it).
     */
    readonly previousProbeAtlas: GPUTexture;
    /** Level 0 of last frame's probe mask, as the next frame reads it. */
    readonly previousProbeMask: GPUTexture;
    /** The host's depth (`r32float`) and normals (`rgba16float`, signed) as the next frame reads them. */
    readonly previousDepth: GPUTexture;
    readonly previousNormal: GPUTexture;
    /** `g_ScreenProbes_ProbeCachedTileBuffer` and `g_ScreenProbes_ProbeCachedTileIndexBuffer` (`rgba32uint`, register D47). */
    readonly cachedTile: GPUTexture;
    readonly cachedTileIndex: GPUTexture;
    /** The packed words (probe-cache.ts) and their layout. */
    readonly probeCacheWords: GPUBuffer;
    readonly probeCacheLayout: ProbeCacheLayout;
    /** The LRU half the next frame reads as its own (`probeCacheLayout.lru[half]`); the other is the one the latest frame read. */
    readonly nextLRUHalf: 0 | 1;
    /** `ProbeCachedTileLRUFlagBuffer`, `ProbeCachedTileLRUIndexBuffer`, `ProbeCachedTileListCountBuffer`, `ProbeCachedTileListIndexBuffer`. */
    readonly lruFlags: GPUBuffer;
    readonly lruIndex: GPUBuffer;
    readonly listCounts: GPUBuffer;
    readonly listIndex: GPUBuffer;
}

/** Kernels compiled in `prepare`. */
interface Kernels {
    readonly clearCounters: ComputeKernel;
    readonly clearProbeMask: ComputeKernel;
    readonly initCachedTileLRU: ComputeKernel;
    readonly reproject: ComputeKernel;
    readonly count: ComputeKernel;
    readonly scatter: ComputeKernel;
    readonly spawn: ComputeKernel;
    readonly compact: ComputeKernel;
    readonly patch: ComputeKernel;
    readonly sample: ComputeKernel;
    readonly populate: ComputeKernel;
    readonly blend: ComputeKernel;
    readonly reorder: ComputeKernel;
    readonly copyPreviousScreen: ComputeKernel;
    readonly debug: ComputeKernel;
}

/** What `prepare` makes: everything that doesn't depend on the screen's size. */
interface Prepared {
    readonly kernels: Kernels;
    /** Spawn tiles the buffers below hold: enough for the largest screen the device can draw. */
    readonly spawnTileCapacity: number;
    /** Probe tiles the probe cache's scan buffers hold, for the same screen. */
    readonly probeTileCapacity: number;
    readonly frameUniforms: GPUBuffer;
    readonly counters: GPUBuffer;
    readonly spawnFlags: GPUBuffer;
    readonly spawnSeeds: GPUBuffer;
    readonly probeSpawn: GPUBuffer;
    readonly compaction: GpuCompaction;
    readonly compactArgs: GpuIndirectArgs;
    /** The probe cache's scanned arrays (probe-cache.ts): flags and their scan, list counts and theirs. */
    readonly lruFlags: GPUBuffer;
    readonly lruIndex: GPUBuffer;
    readonly listCounts: GPUBuffer;
    readonly listIndex: GPUBuffer;
    readonly lruScan: GpuPrefixScan;
    readonly listScan: GpuPrefixScan;
    /** ScatterScreenProbes' dispatch on the listed count, PatchScreenProbes' on the empty tiles. */
    readonly scatterArgs: GpuIndirectArgs;
    readonly patchArgs: GpuIndirectArgs;
    readonly clearCountersGroup: GPUBindGroup;
    readonly debugUniforms: GPUBuffer;
    readonly bytes: number;
}

/** What one screen size needs. */
interface ScreenResources {
    readonly layout: ScreenProbeLayout;
    readonly probeCache: ProbeCacheLayout;
    readonly probeAtlas: GPUTexture;
    readonly previousProbeAtlas: GPUTexture;
    readonly evictedProbeAtlas: GPUTexture;
    readonly probeMask: GPUTexture;
    readonly probeMaskLevel0: GPUTextureView;
    readonly previousProbeMask: GPUTexture;
    readonly cachedTile: GPUTexture;
    readonly cachedTileIndex: GPUTexture;
    readonly previousDepth: GPUTexture;
    readonly previousNormal: GPUTexture;
    readonly probeCacheWords: GPUBuffer;
    readonly probeSH: GPUTexture;
    readonly previousProbeSH: GPUTexture;
    readonly probeRays: GPUBuffer;
    readonly rayScratch: RayScratch;
    /** ClearProbeMask over every level of the mask, then last frame's. */
    readonly clearProbeMaskGroups: readonly { readonly group: GPUBindGroup; readonly size: readonly [number, number] }[];
    readonly initCachedTileLRUGroup: GPUBindGroup;
    readonly countGroup: GPUBindGroup;
    readonly scatterGroup: GPUBindGroup;
    readonly compactGroup: GPUBindGroup;
    readonly reorderGroup: GPUBindGroup;
    readonly bytes: number;
}

/** Bind groups over the host's screen textures, rebuilt when the host hands over other textures. */
interface ScreenInputGroups {
    readonly depth: GPUTexture;
    readonly normal: GPUTexture;
    readonly velocity: GPUTexture;
    /** `FrameScreen.normalEncoding` was `unorm` (the SH debug view decodes the normals as the probes do). */
    readonly normalUnorm: boolean;
    /** `FrameScreen.reversedDepth` (the SH debug view tells sky pixels by the far plane's depth, as the probes do). */
    readonly reversedDepth: boolean;
    readonly resources: ScreenResources;
    readonly reproject: GPUBindGroup;
    readonly spawn: GPUBindGroup;
    readonly patch: GPUBindGroup;
    readonly sample: GPUBindGroup;
    readonly populate: GPUBindGroup;
    readonly blend: GPUBindGroup;
    readonly copyPreviousScreen: GPUBindGroup;
}

export class RadianceCacheTechnique implements GITechnique<RadianceCacheSettings> {
    readonly id = TECHNIQUE_ID;
    /**
     * Rays through the platform's tracer; the host's depth, normals, velocity and last frame's lit
     * colour every frame (`encodeScreen`): reprojection reads the velocity, as Capsaicin's does
     * (gi1.comp:326), and SSGI the lit colour, as Capsaicin's reads `PrevCombinedIllumination`
     * (ssgi.cpp:149; register D56).
     */
    readonly requirements: TechniqueRequirements = { queries: ['rays'], screen: { velocity: true, previousLighting: true } };
    readonly settings: RadianceCacheSettings;
    /** Reads the interpolated irradiance by screen position (shading.ts). */
    readonly shading = new RadianceCacheShadingModule({ shadingTextures: () => this.shadingTextures() });
    /**
     * Drawn per pixel through the shading module's `giDebugColor` by screen
     * position; {@link encodeDebugView} draws the same views engine-free.
     */
    readonly debugViews: readonly DebugView[] = [
        {
            kind: 'screen',
            id: RADIANCE_CACHE_DEBUG_VIEW.probeAtlas,
            label: 'Probe atlas',
            description: "Each probe's 8 × 8 cells of incoming light, drawn in its own 8 × 8 pixel tile; tiles without a probe are a dark checkerboard.",
        },
        { kind: 'screen', id: RADIANCE_CACHE_DEBUG_VIEW.probeMask, label: 'Probe mask', description: 'Which 8 × 8 tiles hold a probe: a colour per probe, its seed pixel white.' },
        {
            kind: 'screen',
            id: RADIANCE_CACHE_DEBUG_VIEW.sphericalHarmonics,
            label: 'Probe SH',
            description: "The light of each tile's probe as its nine SH coefficients hold it, along the pixel's normal, as a white surface would reflect it.",
        },
        // Capsaicin's SSGI views (ssgi_debug.comp, ssgi.cpp:85-90).
        { kind: 'screen', id: RADIANCE_CACHE_DEBUG_VIEW.occlusion, label: 'SSGI occlusion', description: 'How open each pixel is to the sky of its own hemisphere, from nearby depth: white open, black closed in.' },
        { kind: 'screen', id: RADIANCE_CACHE_DEBUG_VIEW.bentNormal, label: 'SSGI bent normal', description: 'The mean open direction at each pixel, as a colour (x red, y green, z blue, 0.5 for 0).' },
    ];
    readonly msPerUnit: number | undefined = undefined;

    private context: TechniqueContext | null = null;
    private prepared: Prepared | null = null;
    /** Capsaicin's hash grid behind the world-cache slot (RC3.2), made in `prepare`. */
    private cache: HashGridCache | null = null;
    /** `resetHistory` empties the cache at the next `encodeScreen` (it has no encoder of its own). */
    private worldCacheResetPending = false;
    /** The screen probes' history starts afresh at the next `encodeScreen` (new resources, or `resetHistory`). */
    private screenHistoryResetPending = true;
    private screenResources: ScreenResources | null = null;
    private inputGroups: ScreenInputGroups | null = null;
    /** The probe filter, SH projection and per-pixel interpolation (RC2.2, RC4.2): made in `prepare`. */
    private irradiance: ScreenIrradiance | null = null;
    /** Capsaicin's SSGI technique (RC4.2): made in `prepare`. */
    private ssgi: ScreenSpaceGI | null = null;
    /** GI-1's image-space denoiser (RC4.3): made in `prepare`. */
    private giDenoiser: GI1Denoiser | null = null;
    /** The latest frame's denoised irradiance, which the shading module reads; null when `useGIDenoiser` is off. */
    private denoisedIrradiance: GPUTexture | null = null;
    private latestFrameIndex = 0;
    /** The LRU half this frame reads (Capsaicin's `probe_buffer_index_` for the LRU buffers, gi1.cpp:2119, 2141). */
    private lruHalf: 0 | 1 = 0;
    /** The camera of the latest frame: the next frame's previous camera (register D49). */
    private previousCamera: FrameCamera | null = null;
    private readonly frameWords = new ArrayBuffer(FRAME_UNIFORM_BYTES);
    private disposed = false;

    constructor(settings: Partial<RadianceCacheSettings> = {}) {
        this.settings = { ...DEFAULT_RADIANCE_CACHE_SETTINGS, ...settings };
    }

    attach(context: TechniqueContext): void {
        this.context = context;
    }

    async prepare(): Promise<void> {
        const context = this.context;
        if (!context) throw new Error('@pmndrs/ddgi: attach the radiance cache (gi.addTechnique) before prepare');
        if (this.disposed) throw new Error('@pmndrs/ddgi: this radiance cache was disposed');
        const device = context.device;

        // The world cache first: a device that can't hold it gets GILimitError before anything else is made.
        const cache = await this.prepareWorldCache(context);

        // gi/direct first: the populate kernel includes its side of the ray scratch (the probe
        // rays' radiance slots; since RC3.2 nothing goes on its work list).
        await context.shade.direct.prepare();
        const workList = context.shade.direct.defaultWorkList;
        const kernel = (label: string, code: string, entryPoint: string, group: GPUBindGroupLayoutDescriptor, modules: readonly WgslModule[] = []) =>
            context.kernels.computeKernel({ label: `${TECHNIQUE_ID}/${label}`, modules, code, entryPoint, ownGroups: [group] });
        const history = Object.fromEntries(
            await Promise.all(
                HISTORY_KERNELS.map(async (spec) => [spec.entryPoint, await kernel(spec.entryPoint, spec.code, spec.entryPoint, spec.group, spec.indirect ? [context.compute.indirectModule] : [])] as const),
            ),
        ) as Record<(typeof HISTORY_KERNELS)[number]['entryPoint'], ComputeKernel>;
        const [clearCounters, clearProbeMask, spawn, compact, sample, populate, blend, debug] = await Promise.all([
            kernel('ClearCounters', CLEAR_COUNTERS_WGSL, ENTRY_POINT.clearCounters, CLEAR_COUNTERS_GROUP),
            kernel('ClearProbeMask', CLEAR_PROBE_MASK_WGSL, ENTRY_POINT.clearProbeMask, CLEAR_PROBE_MASK_GROUP),
            kernel('SpawnScreenProbes', SPAWN_WGSL, ENTRY_POINT.spawn, SPAWN_GROUP),
            // The platform's GiIndirectArgs and giIndirectItemIndex (code only, nothing bound).
            kernel('CompactScreenProbes', COMPACT_WGSL, ENTRY_POINT.compact, COMPACT_GROUP, [context.compute.indirectModule]),
            kernel('SampleScreenProbes', SAMPLE_WGSL, ENTRY_POINT.sample, SAMPLE_GROUP),
            // Hit shading brings the scene copy and the rays in as its dependencies.
            kernel(`PopulateScreenProbes (${workList})`, populateWgsl(context.shade.direct.traceWgsl('rcRayScratch', workList)), ENTRY_POINT.populate, POPULATE_GROUP, [context.shade.module]),
            kernel('BlendScreenProbes', BLEND_WGSL, ENTRY_POINT.blend, BLEND_GROUP),
            kernel('debug-view', DEBUG_WGSL, DEBUG_ENTRY_POINT, DEBUG_GROUP),
        ]);
        const kernels: Kernels = {
            clearCounters,
            clearProbeMask,
            initCachedTileLRU: history[HISTORY_ENTRY_POINT.initCachedTileLRU],
            reproject: history[HISTORY_ENTRY_POINT.reproject],
            count: history[HISTORY_ENTRY_POINT.count],
            scatter: history[HISTORY_ENTRY_POINT.scatter],
            spawn,
            compact,
            patch: history[HISTORY_ENTRY_POINT.patch],
            sample,
            populate,
            blend,
            reorder: history[HISTORY_ENTRY_POINT.reorder],
            copyPreviousScreen: history[HISTORY_ENTRY_POINT.copyPreviousScreen],
            debug,
        };
        const [irradiance, ssgi, giDenoiser] = await Promise.all([ScreenIrradiance.create(context), ScreenSpaceGI.create(context), createGIDenoiser(context)]);

        // The spawn and probe tiles of the largest screen the device can draw: the platform's scans
        // and compaction take their buffers when they are made (asynchronously), so a resize never
        // waits for new ones (register D54).
        const largestScreen = device.limits.maxTextureDimension2D;
        const spawnTileCapacity = Math.ceil(largestScreen / PROBE_SPAWN_TILE_SIZE) ** 2;
        const probeTileCapacity = largestProbeTileCount(largestScreen);
        const storage = (label: string, bytes: number, extraUsage = 0): GPUBuffer =>
            device.createBuffer({ label: `${TECHNIQUE_ID} ${label}`, size: bytes, usage: BUFFER_USAGE.storage | BUFFER_USAGE.copySrc | extraUsage });
        const spawnFlags = storage('spawn flags', spawnTileCapacity * WORD_BYTES);
        const spawnSeeds = storage('spawn seeds', spawnTileCapacity * WORD_BYTES);
        const probeSpawn = storage('probe spawn', PROBE_SPAWN_SEEDS_OFFSET + spawnTileCapacity * WORD_BYTES, BUFFER_USAGE.copyDst);
        const counters = storage('counters', COUNTERS_BYTES);
        // ProbeCachedTileLRUFlagBuffer and ProbeCachedTileLRUIndexBuffer hold one more than the slots (gi1.cpp:307, 313).
        const lruFlags = storage('probe cache LRU flags', (probeTileCapacity + 1) * WORD_BYTES, BUFFER_USAGE.copyDst);
        const lruIndex = storage('probe cache LRU index', (probeTileCapacity + 1) * WORD_BYTES);
        const listCounts = storage('probe cache list counts', probeTileCapacity * WORD_BYTES, BUFFER_USAGE.copyDst);
        const listIndex = storage('probe cache list index', probeTileCapacity * WORD_BYTES);
        const frameUniforms = device.createBuffer({ label: `${TECHNIQUE_ID} frame`, size: FRAME_UNIFORM_BYTES, usage: BUFFER_USAGE.uniform | BUFFER_USAGE.copyDst });
        const debugUniforms = device.createBuffer({ label: `${TECHNIQUE_ID} debug view`, size: DEBUG_UNIFORM_BYTES, usage: BUFFER_USAGE.uniform | BUFFER_USAGE.copyDst });
        const compaction = await context.compute.createCompaction({ label: `${TECHNIQUE_ID} spawn compaction`, flags: spawnFlags, capacity: spawnTileCapacity });
        const compactArgs = await context.compute.createIndirectArgs({
            label: `${TECHNIQUE_ID} CompactScreenProbes args`,
            count: compaction.count,
            itemsPerWorkgroup: WORKGROUP_SIZE.spawn,
        });
        const lruScan = await context.compute.createPrefixScan({ label: `${TECHNIQUE_ID} probe cache LRU scan`, input: lruFlags, output: lruIndex, capacity: probeTileCapacity + 1 });
        const listScan = await context.compute.createPrefixScan({ label: `${TECHNIQUE_ID} probe cache list scan`, input: listCounts, output: listIndex, capacity: probeTileCapacity });
        const scatterArgs = await context.compute.createIndirectArgs({
            label: `${TECHNIQUE_ID} ScatterScreenProbes args`,
            count: { buffer: counters, offset: counterByteOffset('cachedTileListElementCount') },
            itemsPerWorkgroup: HISTORY_WORKGROUP_SIZE.list,
        });
        const patchArgs = await context.compute.createIndirectArgs({
            label: `${TECHNIQUE_ID} PatchScreenProbes args`,
            count: { buffer: counters, offset: counterByteOffset('emptyTileCount') },
            itemsPerWorkgroup: HISTORY_WORKGROUP_SIZE.list,
        });

        this.releaseScreen();
        this.releasePrepared();
        this.cache = cache;
        this.irradiance = irradiance;
        this.ssgi = ssgi;
        this.giDenoiser = giDenoiser;
        this.screenHistoryResetPending = true;
        this.prepared = {
            kernels,
            spawnTileCapacity,
            probeTileCapacity,
            frameUniforms,
            counters,
            spawnFlags,
            spawnSeeds,
            probeSpawn,
            compaction,
            compactArgs,
            lruFlags,
            lruIndex,
            listCounts,
            listIndex,
            lruScan,
            listScan,
            scatterArgs,
            patchArgs,
            clearCountersGroup: device.createBindGroup({ label: `${TECHNIQUE_ID}/ClearCounters`, layout: clearCounters.ownLayouts[0], entries: [{ binding: 0, resource: { buffer: counters } }] }),
            debugUniforms,
            bytes:
                spawnFlags.size +
                spawnSeeds.size +
                probeSpawn.size +
                counters.size +
                lruFlags.size +
                lruIndex.size +
                listCounts.size +
                listIndex.size +
                frameUniforms.size +
                debugUniforms.size +
                lruScan.memoryBytes +
                listScan.memoryBytes +
                scatterArgs.memoryBytes +
                patchArgs.memoryBytes,
        };
    }

    /** No world-space work here: the world cache runs in `encodeScreen`, on the probe rays (Capsaicin's order). */
    encode(encoder: GPUCommandEncoder, frame: TechniqueFrame): number {
        void encoder;
        void frame;
        return 0;
    }

    encodeScreen(encoder: GPUCommandEncoder, frame: TechniqueScreenFrame): void {
        const context = this.context;
        const prepared = this.prepared;
        if (!context || !prepared) throw new Error('@pmndrs/ddgi: prepare the radiance cache before encodeScreen');
        const screen = frame.screen;
        const resources = this.resourcesFor(context, prepared, screen);
        const groups = this.inputGroupsFor(context.device, prepared, resources, screen);
        const layout = resources.layout;
        if (layout.maxProbeSpawnCount > prepared.spawnTileCapacity) {
            throw new GILimitError('maxTextureDimension2D', Math.max(screen.width, screen.height), context.device.limits.maxTextureDimension2D);
        }
        this.latestFrameIndex = frame.frame >>> 0;
        // A frame with no history reads its own camera as the previous one: nothing is found anyway.
        const historyReset = this.screenHistoryResetPending;
        if (historyReset) this.previousCamera = null;
        this.writeFrameUniforms(context.device, prepared, resources, frame);
        const { kernels } = prepared;
        const timer = context.timer;
        const maxPerDimension = context.device.limits.maxComputeWorkgroupsPerDimension;
        const slots = resources.probeCache.slots;

        // 0: Capsaicin's SSGI technique, before GI-1 (gi1_renderer.cpp:60-61).
        this.encodeSSGI(encoder, frame, prepared);

        if (historyReset) {
            this.encodeHistoryReset(encoder, prepared, resources);
            this.screenHistoryResetPending = false;
        }
        // gfxCommandClearBuffer(probe_cached_tile_list_count_buffer_) before the count (gi1.cpp:2283).
        encoder.clearBuffer(prepared.listCounts, 0, slots * WORD_BYTES);

        // 1-3: a fresh frame, the reprojection and the cache's lookup, the spawn, its compaction, the patch.
        // ReprojectScreenProbes over every probe tile (gi1.cpp:2256-2271), after the screen probes'
        // ClearCounters (Capsaicin clears every counter in one kernel in PurgeRadianceCache's section).
        const reprojectPass = beginTimedPass(encoder, timer, 'ReprojectScreenProbes');
        kernels.clearCounters.bind(reprojectPass);
        reprojectPass.setBindGroup(1, prepared.clearCountersGroup);
        reprojectPass.dispatchWorkgroups(1);
        kernels.reproject.bind(reprojectPass);
        reprojectPass.setBindGroup(1, groups.reproject);
        reprojectPass.dispatchWorkgroups(layout.probeCount[0], layout.probeCount[1]);
        reprojectPass.end();
        // LookupScreenProbes (gi1.cpp:2274-2291): count, scan, scatter.
        const lookupPass = beginTimedPass(encoder, timer, 'LookupScreenProbes');
        kernels.count.bind(lookupPass);
        lookupPass.setBindGroup(1, resources.countGroup);
        lookupPass.dispatchWorkgroups(Math.ceil(slots / HISTORY_WORKGROUP_SIZE.list));
        prepared.listScan.encode(lookupPass, slots);
        prepared.scatterArgs.encode(lookupPass);
        kernels.scatter.bind(lookupPass);
        lookupPass.setBindGroup(1, resources.scatterGroup);
        lookupPass.dispatchWorkgroupsIndirect(prepared.scatterArgs.buffer, 0);
        lookupPass.end();
        // SpawnScreenProbes, the scan and CompactScreenProbes (gi1.cpp:2294-2310).
        const spawnPass = beginTimedPass(encoder, timer, 'SpawnScreenProbes');
        kernels.spawn.bind(spawnPass);
        spawnPass.setBindGroup(1, groups.spawn);
        spawnPass.dispatchWorkgroups(Math.ceil(layout.maxProbeSpawnCount / WORKGROUP_SIZE.spawn));
        prepared.compaction.encode(spawnPass, layout.maxProbeSpawnCount);
        prepared.compactArgs.encode(spawnPass);
        kernels.compact.bind(spawnPass);
        spawnPass.setBindGroup(1, resources.compactGroup);
        spawnPass.dispatchWorkgroupsIndirect(prepared.compactArgs.buffer, 0);
        spawnPass.end();
        // PatchScreenProbes on the empty tiles (gi1.cpp:2313-2320); it writes seeds, not the count.
        const patchPass = beginTimedPass(encoder, timer, 'PatchScreenProbes');
        prepared.patchArgs.encode(patchPass);
        kernels.patch.bind(patchPass);
        patchPass.setBindGroup(1, groups.patch);
        patchPass.dispatchWorkgroupsIndirect(prepared.patchArgs.buffer, 0);
        patchPass.end();
        // The probe count in front of the list, where the per-ray kernels read it.
        encoder.copyBufferToBuffer(prepared.compaction.count.buffer, prepared.compaction.count.offset, prepared.probeSpawn, 0, WORD_BYTES);

        // 4: a direction per ray (SampleScreenProbes, gi1.cpp:2322-2330), then the rays: the trace
        // half of Capsaicin's PopulateScreenProbesCells section, whose cache half (the hits' cells)
        // the world cache times under the section's own name.
        const samplePass = beginTimedPass(encoder, timer, 'SampleScreenProbes');
        kernels.sample.bind(samplePass);
        samplePass.setBindGroup(1, groups.sample);
        dispatchSplit(samplePass, layout.maxRayCount / WORKGROUP_SIZE.probe, maxPerDimension);
        samplePass.end();
        const tracePass = beginTimedPass(encoder, timer, POPULATE_TRACE_SECTION);
        kernels.populate.bind(tracePass);
        tracePass.setBindGroup(1, groups.populate);
        dispatchSplit(tracePass, layout.maxRayCount / WORKGROUP_SIZE.populate, maxPerDimension);
        tracePass.end();

        // 5: the world cache lights the rays' hits.
        this.encodeWorldCache(encoder, frame, resources);

        // 6: the rays into the probe atlas, the mask and the probe cache; then the atlas into last
        // frame's (gfxCommandCopyTexture, gi1.cpp:2540-2541): FilterScreenProbes' second pass reads
        // the reprojected probes' tiles there, which its first pass doesn't write.
        const blendPass = beginTimedPass(encoder, timer, 'BlendScreenProbes');
        kernels.blend.bind(blendPass);
        blendPass.setBindGroup(1, groups.blend);
        dispatchSplit(blendPass, layout.maxRayCount / WORKGROUP_SIZE.probe, maxPerDimension);
        blendPass.end();
        copyWholeTexture(encoder, resources.probeAtlas, resources.previousProbeAtlas);

        // 7: re-order the cached probes so the least-recently used entries are evicted first (gi1.cpp:2544-2556).
        const reorderPass = beginTimedPass(encoder, timer, 'ReorderScreenProbes');
        prepared.lruScan.encode(reorderPass, slots + 1);
        kernels.reorder.bind(reorderPass);
        reorderPass.setBindGroup(1, resources.reorderGroup);
        reorderPass.dispatchWorkgroups(Math.ceil(slots / HISTORY_WORKGROUP_SIZE.list));
        reorderPass.end();

        // 8: the mask's mips, the probe filter, the SH and the irradiance per pixel (RC2.2, RC4.2),
        // after the blend as in Capsaicin (gi1.cpp:2561-2636).
        this.encodeIrradiance(encoder, frame, prepared, resources);
        // 8b: the image-space GI denoiser, before the resolve (RC4.3, gi-denoiser.ts).
        this.encodeDenoiser(encoder, frame);

        // 9: what the next frame reprojects from.
        this.encodeHistoryKeep(encoder, prepared, resources, groups, timer);
        this.lruHalf = this.lruHalf === 0 ? 1 : 0;
        this.previousCamera = frame.camera;
    }

    /**
     * The history from nothing: both probe masks cleared on every level
     * (gi1.cpp:153-178), the LRU filled (`InitCachedTileLRU`, gi1.cpp:336-343)
     * and the LRU flags zeroed, so the scan's last flag is 0 whatever screen
     * size ran before (the buffer is made for the largest).
     */
    private encodeHistoryReset(encoder: GPUCommandEncoder, prepared: Prepared, resources: ScreenResources): void {
        encoder.clearBuffer(prepared.lruFlags);
        const pass = encoder.beginComputePass({ label: `${TECHNIQUE_ID}/history-reset` });
        prepared.kernels.clearProbeMask.bind(pass);
        for (const level of resources.clearProbeMaskGroups) {
            pass.setBindGroup(1, level.group);
            pass.dispatchWorkgroups(Math.ceil(level.size[0] / HISTORY_WORKGROUP_SIZE.tile), Math.ceil(level.size[1] / HISTORY_WORKGROUP_SIZE.tile));
        }
        prepared.kernels.initCachedTileLRU.bind(pass);
        pass.setBindGroup(1, resources.initCachedTileLRUGroup);
        pass.dispatchWorkgroups(Math.ceil(resources.probeCache.slots / HISTORY_WORKGROUP_SIZE.list));
        pass.end();
    }

    /**
     * Keeps this frame's filtered atlas, probe mask (level 0), SH, depth and
     * normals as the next frame's "previous" ones. Capsaicin swaps its two
     * atlases, probe masks and SH buffers each frame instead (gi1.cpp:3097):
     * the filter's second pass writes `probe_buffers_[probe_buffer_index_]`
     * (gi1.cpp:2606-2607), which the flip makes the next frame's previous
     * atlas, so its history is the filtered one; its framework copies the
     * depth and normals (register D45, D49).
     */
    private encodeHistoryKeep(encoder: GPUCommandEncoder, prepared: Prepared, resources: ScreenResources, groups: ScreenInputGroups, timer: GpuTimerScope): void {
        const { width, height } = resources.layout;
        copyWholeTexture(encoder, resources.probeAtlas, resources.previousProbeAtlas);
        encoder.copyTextureToTexture({ texture: resources.probeMask, mipLevel: 0 }, { texture: resources.previousProbeMask }, [resources.layout.probeCount[0], resources.layout.probeCount[1]]);
        this.irradiance?.encodeHistoryCopy(encoder);
        const pass = beginTimedPass(encoder, timer, KEEP_HISTORY_SECTION);
        prepared.kernels.copyPreviousScreen.bind(pass);
        pass.setBindGroup(1, groups.copyPreviousScreen);
        pass.dispatchWorkgroups(Math.ceil(width / HISTORY_WORKGROUP_SIZE.tile), Math.ceil(height / HISTORY_WORKGROUP_SIZE.tile));
        pass.end();
    }

    /** Capsaicin's SSGI technique (ssgi.ts) on this frame's screen, with the frame's uniforms. */
    private encodeSSGI(encoder: GPUCommandEncoder, frame: TechniqueScreenFrame, prepared: Prepared): void {
        const ssgi = this.ssgi;
        if (!ssgi) throw new Error('@pmndrs/ddgi: prepare the radiance cache before encodeScreen');
        const screen = frame.screen;
        const previousLighting = screen.previousLighting;
        if (!previousLighting) throw new Error("@pmndrs/ddgi: the radiance cache's SSGI reads FrameScreen.previousLighting, which this screen has not");
        checkScreenTexture(previousLighting, 'previousLighting');
        ssgi.encode(
            encoder,
            { width: screen.width, height: screen.height, frameUniforms: prepared.frameUniforms, depth: screen.depth, normal: screen.normal, previousLighting },
            frame.camera,
            this.settings,
        );
    }

    /** The passes after the blend (screen-irradiance.ts), on this frame's probes, screen and SSGI. */
    private encodeIrradiance(encoder: GPUCommandEncoder, frame: TechniqueScreenFrame, prepared: Prepared, resources: ScreenResources): void {
        const irradiance = this.irradiance;
        const ssgi = this.ssgi;
        if (!irradiance || !ssgi) throw new Error('@pmndrs/ddgi: prepare the radiance cache before encodeScreen');
        const ssgiOutputs = ssgi.outputsFor(frame.screen.width, frame.screen.height);
        irradiance.encode(
            encoder,
            {
                layout: resources.layout,
                frameUniforms: prepared.frameUniforms,
                probeSpawn: prepared.probeSpawn,
                probeAtlas: resources.probeAtlas,
                previousProbeAtlas: resources.previousProbeAtlas,
                probeMask: resources.probeMask,
                depth: frame.screen.depth,
                normal: frame.screen.normal,
                occlusionAndBentNormal: ssgiOutputs.occlusionAndBentNormal,
                nearField: ssgiOutputs.nearField,
            },
            frame.camera,
        );
    }

    /**
     * GI-1's denoiser on this frame's interpolated irradiance, with the
     * previous frame's camera (this frame's after a history reset).
     */
    private encodeDenoiser(encoder: GPUCommandEncoder, frame: TechniqueScreenFrame): void {
        const interpolated = this.irradiance?.irradiance;
        const denoiser = this.giDenoiser;
        if (!this.settings.useGIDenoiser || !denoiser || !interpolated) {
            this.denoisedIrradiance = null;
            return;
        }
        // Turned back on: its history is from frames ago, under another camera.
        if (this.denoisedIrradiance === null) denoiser.resetHistory();
        this.denoisedIrradiance = encodeGIDenoiser(denoiser, encoder, frame.screen, frame.camera, this.previousCamera ?? frame.camera, interpolated);
    }

    /** What the shading module binds: the latest frame's textures, or 1 × 1 placeholders before the first. */
    private shadingTextures(): { irradiance: GPUTexture; probeAtlas: GPUTexture; probeMask: GPUTexture; probeSH: GPUTexture; occlusionAndBentNormal: GPUTexture } {
        const irradiance = this.irradiance;
        if (!irradiance) throw new Error('@pmndrs/ddgi: the radiance cache has no shading textures before prepare');
        const placeholders = irradiance.placeholders;
        return {
            irradiance: this.denoisedIrradiance ?? irradiance.irradiance ?? placeholders.irradiance,
            probeAtlas: this.screenResources?.probeAtlas ?? placeholders.probeAtlas,
            probeMask: this.screenResources?.probeMask ?? placeholders.probeMask,
            probeSH: irradiance.probeSH ?? placeholders.probeSH,
            // rgba16float like the irradiance: its placeholder (zeros) serves both.
            occlusionAndBentNormal: this.ssgi?.occlusionAndBentNormal ?? placeholders.irradiance,
        };
    }

    /**
     * Draws a debug view (wgsl/debug.ts) of the latest frame's probes into
     * `target`: an `rgba8unorm` texture with `STORAGE_BINDING` usage, at least
     * the screen's size. Engine-free: the host shows the image as it likes.
     * `exposure` multiplies radiance before tone mapping (the atlas and SH views).
     */
    encodeDebugView(encoder: GPUCommandEncoder, view: RadianceCacheDebugViewId, target: GPUTexture, exposure = 1): void {
        const context = this.context;
        const prepared = this.prepared;
        const resources = this.screenResources;
        const inputs = this.inputGroups;
        const probeSH = this.irradiance?.probeSH;
        const occlusionAndBentNormal = this.ssgi?.occlusionAndBentNormal;
        if (!context || !prepared || !resources || !inputs || !probeSH || !occlusionAndBentNormal) throw new Error('@pmndrs/ddgi: the radiance cache has no probes to show before its first encodeScreen');
        const { width, height } = resources.layout;
        if (target.width < width || target.height < height) throw new RangeError(`@pmndrs/ddgi: the debug view needs a target of at least ${width} × ${height}`);
        const words = new ArrayBuffer(DEBUG_UNIFORM_BYTES);
        const uints = new Uint32Array(words);
        const offsets = DEBUG_UNIFORM_OFFSETS;
        uints.set([width, height], offsets.size / WORD_BYTES);
        uints[offsets.view / WORD_BYTES] = view;
        new Float32Array(words)[offsets.exposure / WORD_BYTES] = exposure;
        uints[offsets.normalUnorm / WORD_BYTES] = inputs.normalUnorm ? 1 : 0;
        uints[offsets.reversedDepth / WORD_BYTES] = inputs.reversedDepth ? 1 : 0;
        context.device.queue.writeBuffer(prepared.debugUniforms, 0, words);
        const group = context.device.createBindGroup({
            label: `${TECHNIQUE_ID}/debug-view`,
            layout: prepared.kernels.debug.ownLayouts[0],
            entries: [
                { binding: 0, resource: { buffer: prepared.debugUniforms } },
                { binding: 1, resource: resources.probeAtlas.createView() },
                // The whole mip chain: the SH view finds the nearest probe up it (FindClosestProbe).
                { binding: 2, resource: resources.probeMask.createView({ baseMipLevel: 0, mipLevelCount: resources.layout.probeMaskMipCount }) },
                { binding: 3, resource: target.createView({ baseMipLevel: 0, mipLevelCount: 1 }) },
                { binding: 4, resource: probeSH.createView() },
                { binding: 5, resource: screenTextureView(inputs.normal) },
                { binding: 6, resource: screenTextureView(inputs.depth) },
                { binding: 7, resource: occlusionAndBentNormal.createView() },
            ],
        });
        const pass = encoder.beginComputePass({ label: `${TECHNIQUE_ID}/debug-view` });
        prepared.kernels.debug.bind(pass);
        pass.setBindGroup(1, group);
        pass.dispatchWorkgroups(Math.ceil(width / 8), Math.ceil(height / 8));
        pass.end();
    }

    /** The latest screen's probe resources, for debug views and tests; null before the first `encodeScreen`. */
    get probes(): RadianceCacheProbes | null {
        const prepared = this.prepared;
        const resources = this.screenResources;
        const irradiance = this.irradiance;
        const ssgi = this.ssgi;
        if (!prepared || !resources || !irradiance?.probeSH || !irradiance.irradiance || !ssgi?.occlusionAndBentNormal || !ssgi.nearField) return null;
        return {
            layout: resources.layout,
            frameIndex: this.latestFrameIndex,
            probeAtlas: resources.probeAtlas,
            probeMask: resources.probeMask,
            spawnFlags: prepared.spawnFlags,
            spawnSeeds: prepared.spawnSeeds,
            probeSpawn: prepared.probeSpawn,
            probeRays: resources.probeRays,
            rayScratch: resources.rayScratch,
            probeSH: irradiance.probeSH,
            irradiance: irradiance.irradiance,
            occlusionAndBentNormal: ssgi.occlusionAndBentNormal,
            nearField: ssgi.nearField,
        };
    }

    /** The screen probes' history and the probe cache after the latest `encodeScreen`, for tests; null before it. */
    get probeHistory(): RadianceCacheProbeHistory | null {
        const prepared = this.prepared;
        const resources = this.screenResources;
        if (!prepared || !resources) return null;
        return {
            counters: prepared.counters,
            previousProbeAtlas: resources.previousProbeAtlas,
            previousProbeMask: resources.previousProbeMask,
            previousDepth: resources.previousDepth,
            previousNormal: resources.previousNormal,
            cachedTile: resources.cachedTile,
            cachedTileIndex: resources.cachedTileIndex,
            probeCacheWords: resources.probeCacheWords,
            probeCacheLayout: resources.probeCache,
            nextLRUHalf: this.lruHalf,
            lruFlags: prepared.lruFlags,
            lruIndex: prepared.lruIndex,
            listCounts: prepared.listCounts,
            listIndex: prepared.listIndex,
        };
    }

    /** GI-1's image-space denoiser once prepared (its output and history), for tests; null before `prepare`. */
    get denoiser(): GI1Denoiser | null {
        return this.giDenoiser;
    }

    /** The world cache (Capsaicin's hash grid) once prepared, for tests and debug views; null before `prepare`. */
    get worldCache(): HashGridCache | null {
        return this.cache;
    }

    get counters(): Readonly<Record<string, number>> {
        const layout = this.screenResources?.layout;
        // What the CPU knows without a readback: the spawned count stays on the GPU. The world
        // cache adds its capacity and memory (its per-frame counts stay on the GPU too).
        const screen: Record<string, number> = layout ? { 'spawn tiles': layout.maxProbeSpawnCount, 'probe rays at most': layout.maxRayCount } : {};
        return { ...screen, ...worldCacheCounters(this.cache) };
    }

    /** GPU bytes: the screen probes' buffers, the SH and irradiance (RC2.2), SSGI's outputs (RC4.2), the GI denoiser's (RC4.3) and the world cache's (about 1 GiB at Capsaicin's default capacity). */
    get memoryBytes(): number {
        const prepared = this.prepared;
        if (!prepared) return 0;
        const screenProbes = prepared.bytes + prepared.compaction.memoryBytes + prepared.compactArgs.memoryBytes + (this.screenResources?.bytes ?? 0);
        return screenProbes + (this.irradiance?.memoryBytes ?? 0) + (this.ssgi?.memoryBytes ?? 0) + (this.giDenoiser?.memoryBytes ?? 0) + (this.cache?.memoryBytes ?? 0);
    }

    /**
     * Forgets the history at the next `encodeScreen`: the world cache's cells,
     * and the screen probes' (both probe masks cleared and the probe cache
     * emptied, as when Capsaicin makes them), so that frame finds no previous
     * probe anywhere.
     */
    resetHistory(): void {
        this.worldCacheResetPending = true;
        this.resetScreenProbes();
    }

    /**
     * Forgets only the screen probes' history at the next `encodeScreen`
     * (a camera cut): the world cache keeps its cells.
     */
    resetScreenProbes(): void {
        this.screenHistoryResetPending = true;
        // The denoiser's history is the screen's too (Capsaicin's starts empty when it is made).
        this.giDenoiser?.resetHistory();
    }

    dispose(): void {
        this.releaseScreen();
        this.releasePrepared();
        this.cache?.dispose();
        this.cache = null;
        this.disposed = true;
    }

    /**
     * Makes the world cache at the technique's settings (they carry the hash
     * grid's options under the same names) and prepares it: Capsaicin's
     * default capacity, about 1 GiB, or `GILimitError` naming the limit on a
     * device that can't hold it (register D14).
     */
    private async prepareWorldCache(context: TechniqueContext): Promise<HashGridCache> {
        const cache = this.cache ?? new HashGridCache(this.settings);
        await cache.prepare(context);
        this.worldCacheResetPending = false;
        return cache;
    }

    /**
     * The world cache's work for this frame, after the probe rays are traced
     * and before they are blended: the slot's `encodeFrame` with the rays and
     * their radiance slots (docs/design/radiance-cache.md, "The world-cache
     * interface"), and first, after `resetHistory`, its `reset`.
     */
    private encodeWorldCache(encoder: GPUCommandEncoder, frame: TechniqueScreenFrame, resources: ScreenResources): void {
        const cache: WorldCache | null = this.cache;
        if (!cache) throw new Error('@pmndrs/ddgi: prepare the radiance cache before encodeScreen');
        if (this.worldCacheResetPending) {
            cache.reset(encoder);
            this.worldCacheResetPending = false;
        }
        const rays: WorldCacheProbeRays = {
            buffer: resources.probeRays,
            count: { buffer: resources.probeRays, offset: 0 },
            capacity: resources.layout.maxRayCount,
            radiance: resources.rayScratch.rayResults,
        };
        cache.encodeFrame(encoder, { frame: frame.frame, camera: frame.camera, width: resources.layout.width, height: resources.layout.height }, rays);
    }

    private resourcesFor(context: TechniqueContext, prepared: Prepared, screen: FrameScreen): ScreenResources {
        const current = this.screenResources;
        if (current && current.layout.width === screen.width && current.layout.height === screen.height) return current;
        const device = context.device;
        const irradiance = this.irradiance;
        if (!irradiance) throw new Error('@pmndrs/ddgi: prepare the radiance cache before encodeScreen');
        const layout = screenProbeLayout(screen.width, screen.height);
        this.releaseScreen();
        const atlas = (label: string, usage: number): GPUTexture =>
            device.createTexture({ label: `${TECHNIQUE_ID} ${label}`, size: [layout.atlasSize[0], layout.atlasSize[1]], format: 'rgba16float', usage });
        // g_ScreenProbes_ProbeBuffer (gi1.cpp:131-144): read by the filter, the projection and the
        // debug view, copied into last frame's after the blend and the filter.
        const probeAtlas = atlas('probe atlas', TEXTURE_USAGE.storageBinding | TEXTURE_USAGE.textureBinding | TEXTURE_USAGE.copySrc);
        // The other of Capsaicin's two (register D45): last frame's atlas, then SampleScreenProbes'
        // history for the blend. COPY_DST also lets a test write a history of its own.
        const previousProbeAtlas = atlas('previous probe atlas', TEXTURE_USAGE.storageBinding | TEXTURE_USAGE.textureBinding | TEXTURE_USAGE.copySrc | TEXTURE_USAGE.copyDst);
        // The replaced probes' radiance from SampleScreenProbes to BlendScreenProbes (register D46).
        const evictedProbeAtlas = atlas('evicted probe atlas', TEXTURE_USAGE.storageBinding | TEXTURE_USAGE.textureBinding);
        // g_ScreenProbes_ProbeCachedTileBuffer (gi1.cpp:289-292).
        const cachedTile = atlas('probe cache atlas', TEXTURE_USAGE.storageBinding | TEXTURE_USAGE.textureBinding | TEXTURE_USAGE.copySrc);
        // g_ScreenProbes_ProbeMaskBuffer (gi1.cpp:146-180): with its mip chain, which RC2.2's FilterProbeMask fills.
        const probeMask = device.createTexture({
            label: `${TECHNIQUE_ID} probe mask`,
            size: [layout.probeCount[0], layout.probeCount[1]],
            format: 'r32uint',
            mipLevelCount: layout.probeMaskMipCount,
            usage: TEXTURE_USAGE.storageBinding | TEXTURE_USAGE.textureBinding | TEXTURE_USAGE.copySrc,
        });
        const probeMaskLevel0 = probeMask.createView({ label: `${TECHNIQUE_ID} probe mask level 0`, baseMipLevel: 0, mipLevelCount: 1 });
        // Last frame's level 0: the only level ReprojectScreenProbes reads (register D45).
        const previousProbeMask = device.createTexture({
            label: `${TECHNIQUE_ID} previous probe mask`,
            size: [layout.probeCount[0], layout.probeCount[1]],
            format: 'r32uint',
            usage: TEXTURE_USAGE.storageBinding | TEXTURE_USAGE.textureBinding | TEXTURE_USAGE.copySrc | TEXTURE_USAGE.copyDst,
        });
        // g_ScreenProbes_ProbeCachedTileIndexBuffer (gi1.cpp:294-296), as integer bits (register D47).
        const cachedTileIndex = device.createTexture({
            label: `${TECHNIQUE_ID} probe cache index`,
            size: [layout.probeCount[0], layout.probeCount[1]],
            format: 'rgba32uint',
            usage: TEXTURE_USAGE.storageBinding | TEXTURE_USAGE.textureBinding | TEXTURE_USAGE.copySrc,
        });
        // Last frame's depth and normals (register D49).
        const screenTexture = (label: string, format: GPUTextureFormat): GPUTexture =>
            device.createTexture({ label: `${TECHNIQUE_ID} ${label}`, size: [layout.width, layout.height], format, usage: TEXTURE_USAGE.storageBinding | TEXTURE_USAGE.textureBinding | TEXTURE_USAGE.copySrc });
        const previousDepth = screenTexture('previous depth', PREVIOUS_DEPTH_FORMAT);
        const previousNormal = screenTexture('previous normal', PREVIOUS_NORMAL_FORMAT);
        const probeCache = probeCacheLayout(layout);
        const probeCacheWords = device.createBuffer({ label: `${TECHNIQUE_ID} probe cache words`, size: probeCache.bytes, usage: BUFFER_USAGE.storage | BUFFER_USAGE.copySrc });
        const { probeSH, previousProbeSH } = irradiance.shTextures(layout);
        const probeRays = device.createBuffer({
            label: `${TECHNIQUE_ID} probe rays`,
            size: PROBE_RAYS_HEADER_BYTES + layout.maxRayCount * PROBE_RAY_BYTES,
            usage: BUFFER_USAGE.storage | BUFFER_USAGE.copySrc,
        });
        // The rays' radiance slots. Nothing goes on gi/direct's work list since RC3.2 (the world
        // cache lights the hits), so it has no room.
        const rayScratch = context.shade.direct.createScratch({ label: `${TECHNIQUE_ID} probe rays`, rays: layout.maxRayCount, workItems: 0 });

        const { kernels } = prepared;
        const uniforms = { buffer: prepared.frameUniforms };
        const words = { buffer: probeCacheWords };
        const group = (label: string, kernel: ComputeKernel, entries: GPUBindingResource[]): GPUBindGroup =>
            device.createBindGroup({ label: `${TECHNIQUE_ID}/${label}`, layout: kernel.ownLayouts[0], entries: entries.map((resource, binding) => ({ binding, resource })) });
        const clearProbeMaskGroups: { group: GPUBindGroup; size: readonly [number, number] }[] = [];
        for (let level = 0; level < layout.probeMaskMipCount; level++) {
            clearProbeMaskGroups.push({
                group: group(`ClearProbeMask level ${level}`, kernels.clearProbeMask, [probeMask.createView({ baseMipLevel: level, mipLevelCount: 1 })]),
                size: [Math.max(layout.probeCount[0] >> level, 1), Math.max(layout.probeCount[1] >> level, 1)],
            });
        }
        clearProbeMaskGroups.push({ group: group('ClearProbeMask previous', kernels.clearProbeMask, [previousProbeMask.createView()]), size: layout.probeCount });

        const atlasBytes = layout.atlasSize[0] * layout.atlasSize[1] * HALF4_TEXEL_BYTES;
        const maskBytes = mipChainTexels(layout.probeCount[0], layout.probeCount[1], layout.probeMaskMipCount) * WORD_BYTES;
        const tileTexels = layout.probeCount[0] * layout.probeCount[1];
        const resources: ScreenResources = {
            layout,
            probeCache,
            probeAtlas,
            previousProbeAtlas,
            evictedProbeAtlas,
            probeMask,
            probeMaskLevel0,
            previousProbeMask,
            cachedTile,
            cachedTileIndex,
            previousDepth,
            previousNormal,
            probeCacheWords,
            probeSH,
            previousProbeSH,
            probeRays,
            rayScratch,
            clearProbeMaskGroups,
            initCachedTileLRUGroup: group('InitCachedTileLRU', kernels.initCachedTileLRU, [uniforms, words]),
            countGroup: group('CountScreenProbes', kernels.count, [uniforms, cachedTileIndex.createView(), words, { buffer: prepared.counters }, { buffer: prepared.listCounts }, { buffer: prepared.lruFlags }]),
            scatterGroup: group('ScatterScreenProbes', kernels.scatter, [uniforms, { buffer: prepared.scatterArgs.buffer }, words, { buffer: prepared.listIndex }]),
            compactGroup: group('CompactScreenProbes', kernels.compact, [
                uniforms,
                { buffer: prepared.compactArgs.buffer },
                { buffer: prepared.compaction.indices },
                { buffer: prepared.spawnSeeds },
                { buffer: prepared.probeSpawn },
                probeMaskLevel0,
                { buffer: prepared.counters },
                words,
            ]),
            reorderGroup: group('ReorderScreenProbes', kernels.reorder, [uniforms, words, { buffer: prepared.lruFlags }, { buffer: prepared.lruIndex }]),
            bytes:
                4 * atlasBytes +
                maskBytes +
                tileTexels * WORD_BYTES +
                tileTexels * UINT4_TEXEL_BYTES +
                layout.width * layout.height * (WORD_BYTES + HALF4_TEXEL_BYTES) +
                probeCacheWords.size +
                probeRays.size +
                rayScratch.bytes,
        };
        this.screenResources = resources;
        // New textures: the history starts afresh, as Capsaicin's on allocation.
        this.screenHistoryResetPending = true;
        this.lruHalf = 0;
        return resources;
    }

    private inputGroupsFor(device: GPUDevice, prepared: Prepared, resources: ScreenResources, screen: FrameScreen): ScreenInputGroups {
        const current = this.inputGroups;
        const normalUnorm = screen.normalEncoding === 'unorm';
        const reversedDepth = screen.reversedDepth === true;
        const velocityTexture = screen.velocity;
        if (!velocityTexture) throw new Error('@pmndrs/ddgi: the radiance cache reprojects its probes with FrameScreen.velocity, which this screen has not');
        if (
            current &&
            current.resources === resources &&
            current.depth === screen.depth &&
            current.normal === screen.normal &&
            current.velocity === velocityTexture &&
            current.normalUnorm === normalUnorm &&
            current.reversedDepth === reversedDepth
        ) {
            return current;
        }
        checkScreenTexture(screen.depth, 'depth');
        checkScreenTexture(screen.normal, 'normal');
        checkScreenTexture(velocityTexture, 'velocity');
        const depth = screenTextureView(screen.depth);
        const normal = screenTextureView(screen.normal);
        const velocity = screenTextureView(velocityTexture);
        const uniforms = { buffer: prepared.frameUniforms };
        const counters = { buffer: prepared.counters };
        const words = { buffer: resources.probeCacheWords };
        const { kernels } = prepared;
        const group = (label: string, kernel: ComputeKernel, entries: GPUBindingResource[]): GPUBindGroup =>
            device.createBindGroup({ label: `${TECHNIQUE_ID}/${label}`, layout: kernel.ownLayouts[0], entries: entries.map((resource, binding) => ({ binding, resource })) });
        const groups: ScreenInputGroups = {
            depth: screen.depth,
            normal: screen.normal,
            velocity: velocityTexture,
            normalUnorm,
            reversedDepth,
            resources,
            reproject: group('ReprojectScreenProbes', kernels.reproject, [
                uniforms,
                depth,
                normal,
                velocity,
                resources.previousDepth.createView(),
                resources.previousNormal.createView(),
                resources.previousProbeMask.createView(),
                resources.previousProbeAtlas.createView(),
                resources.previousProbeSH.createView(),
                resources.probeAtlas.createView(),
                resources.probeMaskLevel0,
                resources.probeSH.createView(),
                counters,
                words,
            ]),
            spawn: group('SpawnScreenProbes', kernels.spawn, [uniforms, depth, normal, { buffer: prepared.spawnSeeds }, { buffer: prepared.spawnFlags }]),
            patch: group('PatchScreenProbes', kernels.patch, [uniforms, depth, { buffer: prepared.patchArgs.buffer }, counters, words, { buffer: prepared.probeSpawn }]),
            sample: group('SampleScreenProbes', kernels.sample, [
                uniforms,
                depth,
                normal,
                resources.probeMaskLevel0,
                resources.probeAtlas.createView(),
                resources.cachedTile.createView(),
                resources.cachedTileIndex.createView(),
                { buffer: prepared.probeSpawn },
                { buffer: resources.probeRays },
                counters,
                words,
                { buffer: prepared.listCounts },
                { buffer: prepared.listIndex },
                { buffer: prepared.lruFlags },
                resources.previousProbeAtlas.createView(),
                resources.evictedProbeAtlas.createView(),
            ]),
            populate: group('PopulateScreenProbes', kernels.populate, [
                uniforms,
                depth,
                normal,
                { buffer: prepared.probeSpawn },
                { buffer: resources.probeRays },
                { buffer: resources.rayScratch.buffer },
            ]),
            blend: group('BlendScreenProbes', kernels.blend, [
                uniforms,
                normal,
                depth,
                resources.previousProbeAtlas.createView(),
                resources.evictedProbeAtlas.createView(),
                { buffer: prepared.probeSpawn },
                { buffer: resources.probeRays },
                resources.rayScratch.rayResults,
                counters,
                words,
                { buffer: prepared.lruFlags },
                resources.probeAtlas.createView(),
                resources.probeMaskLevel0,
                resources.cachedTile.createView(),
                resources.cachedTileIndex.createView(),
            ]),
            copyPreviousScreen: group('CopyPreviousScreen', kernels.copyPreviousScreen, [uniforms, depth, normal, resources.previousDepth.createView(), resources.previousNormal.createView()]),
        };
        this.inputGroups = groups;
        return groups;
    }

    private writeFrameUniforms(device: GPUDevice, prepared: Prepared, resources: ScreenResources, frame: TechniqueScreenFrame): void {
        const { layout, probeCache } = resources;
        const words = this.frameWords;
        const floats = new Float32Array(words);
        const uints = new Uint32Array(words);
        // RcFrame's fields by name (FRAME_UNIFORM_OFFSETS are bytes; the arrays index words).
        const word = (field: keyof typeof FRAME_UNIFORM_OFFSETS): number => FRAME_UNIFORM_OFFSETS[field] / WORD_BYTES;
        // With the camera's jitter when the host drew its depth with one (FrameCamera.jitter, camera.ts).
        const previousCamera = this.previousCamera ?? frame.camera;
        floats.set(cameraRelativeInverseViewProjection(frame.camera), word('inverseViewProjection'));
        floats.set(cameraRelativeInverseViewProjection(previousCamera), word('previousInverseViewProjection'));
        floats.set(cameraRelativeViewProjection(frame.camera), word('viewProjection'));
        floats.set(frame.camera.position, word('eye'));
        floats.set(previousCamera.position, word('previousEye'));
        floats[word('cellSize')] = screenProbeCellSize(frame.camera, layout.width, layout.height);
        uints[word('frameIndex')] = this.latestFrameIndex;
        uints.set([layout.width, layout.height], word('bufferDimensions'));
        uints.set([layout.probeCount[0], layout.probeCount[1]], word('probeCount'));
        uints[word('maxProbeSpawnCount')] = layout.maxProbeSpawnCount;
        uints[word('normalUnorm')] = frame.screen.normalEncoding === 'unorm' ? 1 : 0;
        uints[word('reversedDepth')] = frame.screen.reversedDepth === true ? 1 : 0;
        uints[word('useDirectLighting')] = this.settings.useDirectLighting ? 1 : 0;
        uints[word('cachedTileLRUOffset')] = probeCache.lru[this.lruHalf];
        uints[word('previousCachedTileLRUOffset')] = probeCache.lru[this.lruHalf === 0 ? 1 : 0];
        uints[word('cachedTileMRUOffset')] = probeCache.mru;
        uints[word('cachedTileListOffset')] = probeCache.list;
        uints[word('cachedTileListElementOffset')] = probeCache.listElement;
        uints[word('emptyTileOffset')] = probeCache.emptyTile;
        uints[word('overrideTileOffset')] = probeCache.overrideTile;
        uints[word('spawnProbeOffset')] = probeCache.spawnProbe;
        device.queue.writeBuffer(prepared.frameUniforms, 0, words);
    }

    private releaseScreen(): void {
        const resources = this.screenResources;
        if (!resources) return;
        for (const texture of [
            resources.probeAtlas,
            resources.previousProbeAtlas,
            resources.evictedProbeAtlas,
            resources.probeMask,
            resources.previousProbeMask,
            resources.cachedTile,
            resources.cachedTileIndex,
            resources.previousDepth,
            resources.previousNormal,
        ]) {
            texture.destroy();
        }
        resources.probeCacheWords.destroy();
        resources.probeRays.destroy();
        resources.rayScratch.destroy();
        this.screenResources = null;
        this.inputGroups = null;
    }

    private releasePrepared(): void {
        const prepared = this.prepared;
        if (!prepared) return;
        const buffers = [prepared.frameUniforms, prepared.counters, prepared.spawnFlags, prepared.spawnSeeds, prepared.probeSpawn, prepared.debugUniforms];
        for (const buffer of [...buffers, prepared.lruFlags, prepared.lruIndex, prepared.listCounts, prepared.listIndex]) buffer.destroy();
        for (const utility of [prepared.compaction, prepared.compactArgs, prepared.lruScan, prepared.listScan, prepared.scatterArgs, prepared.patchArgs]) utility.dispose();
        this.irradiance?.dispose();
        this.irradiance = null;
        this.ssgi?.dispose();
        this.ssgi = null;
        this.giDenoiser?.dispose();
        this.giDenoiser = null;
        this.denoisedIrradiance = null;
        this.prepared = null;
    }
}

/**
 * `workgroups` workgroups of a 1D kernel as `[x, y]`, so neither passes the
 * device's limit; the kernels count workgroups in rows (`rcWorkgroupIndex`),
 * and those past the last probe return early.
 */
function dispatchSplit(pass: GPUComputePassEncoder, workgroups: number, maxPerDimension: number): void {
    const x = Math.max(1, Math.min(workgroups, maxPerDimension));
    pass.dispatchWorkgroups(x, Math.ceil(workgroups / x));
}

/** A whole single-level texture into another of the same size and format. */
function copyWholeTexture(encoder: GPUCommandEncoder, source: GPUTexture, destination: GPUTexture): void {
    encoder.copyTextureToTexture({ texture: source }, { texture: destination }, [source.width, source.height]);
}

function mipChainTexels(width: number, height: number, levels: number): number {
    let texels = 0;
    for (let level = 0; level < levels; level++) texels += Math.max(1, width >> level) * Math.max(1, height >> level);
    return texels;
}

function checkScreenTexture(texture: GPUTexture, name: string): void {
    if ((texture.usage & TEXTURE_USAGE.textureBinding) === 0) {
        throw new Error(`@pmndrs/ddgi: the radiance cache reads FrameScreen.${name} in compute, so it needs TEXTURE_BINDING usage`);
    }
}

/** A depth texture is read through its depth aspect (as `unfilterable-float`); anything else whole. */
function screenTextureView(texture: GPUTexture): GPUTextureView {
    const isDepth = texture.format.startsWith('depth');
    return texture.createView({ aspect: isDepth ? 'depth-only' : 'all', baseMipLevel: 0, mipLevelCount: 1 });
}

/** The world cache's figures for the technique's counters, each prefixed so they read apart from the screen probes'. */
function worldCacheCounters(cache: WorldCache | null): Record<string, number> {
    const counters: Record<string, number> = {};
    if (!cache) return counters;
    for (const [name, value] of Object.entries(cache.stats)) counters[`world cache ${name}`] = value;
    return counters;
}
