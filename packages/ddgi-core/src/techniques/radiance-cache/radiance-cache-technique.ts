/**
 * The radiance cache: a faithful WebGPU port of AMD's GI-1.0/1.2 as Capsaicin
 * v1.3 builds it (docs/design/radiance-cache.md). The screen probes,
 * traced and blended into the probe atlas every frame (RC2.1), then
 * projected to SH and interpolated per pixel into irradiance the shading
 * module reads by screen position (RC2.2, screen-irradiance.ts), with their
 * debug views. The world cache (phase 3, RC3.2) lights the probe rays' hits;
 * the probes carry over from frame to frame (RC4.1, screen-probe-history.ts).
 *
 * Per frame, in `encodeScreen` (after the host's `encode` and its depth and
 * normal pass), the passes Capsaicin runs for the screen probes
 * (`GI1::render`, gi1.cpp:2250-2557):
 *
 * 1. `ClearCounters` (gi1.comp:163);
 * 2. `ReprojectScreenProbes` (gi1.comp:292), then the off-screen cache's
 *    lookup: `CountScreenProbes` (gi1.comp:502), the platform's scan and
 *    `ScatterScreenProbes` (gi1.comp:542);
 * 3. `SpawnScreenProbes` (gi1.comp:555), the platform's stream compaction
 *    (Capsaicin's `gfxCommandScanSum`, gi1.cpp:2305) and `CompactScreenProbes`
 *    (gi1.comp:581), dispatched indirectly on the kept count, then
 *    `PatchScreenProbes` (gi1.comp:616) on the empty tiles;
 * 4. `SampleScreenProbes` (gi1.comp:646) and `PopulateScreenProbesMain`
 *    (gi1.comp:1160);
 * 5. the world cache (`world-cache/`, RC3.2): Capsaicin's hash grid inserts
 *    the hits' cells, lights one light sample per hit, updates its tiles and
 *    resolves its radiance into the probe rays (gi1.cpp:2238-2253 and
 *    2335-2528, here after the trace: `encodeWorldCache`);
 * 6. `BlendScreenProbes` (gi1.comp:1167) into the probe atlas, the mask and
 *    the cache, then `ReorderScreenProbes` (gi1.comp:1334) of the cache's LRU;
 * 7. `FilterProbeMask` (gi1.comp:249), `ProjectScreenProbes` (gi1.comp:1446)
 *    and `InterpolateScreenProbes` (gi1.comp:1515): screen-irradiance.ts (RC2.2);
 * 8. the screen's depth and normals copied for the next frame's reprojection.
 *
 * Capsaicin dispatches the per-ray passes over every possible ray
 * (`max_ray_count`, gi1.cpp:2323-2542) with an early return past the probe
 * count, and so do we: they are not sized on the GPU in Capsaicin.
 *
 * Resources that depend on the screen's size are made when the size changes,
 * as Capsaicin's `ensureMemoryIsAllocated` does (gi1.cpp:113); that makes
 * buffers and textures, never pipelines. Everything the spawn tiles need is
 * made in `prepare`, for the largest screen the device can draw.
 */

import type { GpuCompaction, GpuIndirectArgs } from '../../platform/contract/compute.ts';
import { GILimitError } from '../../platform/contract/errors.ts';
import type { FrameScreen } from '../../platform/contract/frame.ts';
import type { RayScratch } from '../../platform/contract/shade.ts';
import type { DebugView } from '../../platform/contract/stats.ts';
import type { GITechnique, TechniqueContext, TechniqueFrame, TechniqueRequirements, TechniqueScreenFrame } from '../../platform/contract/technique.ts';
import type { ComputeKernel, WgslModule } from '../../platform/contract/wgsl.ts';
import { cameraRelativeInverseViewProjection, cameraRelativeViewProjection } from './camera.ts';
import { ScreenIrradiance, screenProbeCellSize } from './screen-irradiance.ts';
import { historyChangesWords, mergeHistoryChanges, NO_HISTORY_CHANGES, ScreenProbeHistory, type HistoryChanges, type HistoryResources } from './screen-probe-history.ts';
import { DEFAULT_RADIANCE_CACHE_SETTINGS, PROBE_SPAWN_TILE_SIZE, screenProbeLayout, type RadianceCacheSettings, type ScreenProbeLayout } from './settings.ts';
import { RadianceCacheShadingModule } from './shading.ts';
import { HashGridCache } from './world-cache/hash-grid-cache.ts';
import { SharcCache } from './world-cache/sharc/sharc-cache.ts';
import type { WorldCache, WorldCacheProbeRays } from './world-cache/world-cache.ts';
import { DEBUG_ENTRY_POINT, DEBUG_GROUP, DEBUG_UNIFORM_BYTES, DEBUG_UNIFORM_OFFSETS, DEBUG_WGSL, RADIANCE_CACHE_DEBUG_VIEW, type RadianceCacheDebugViewId } from './wgsl/debug.ts';
import {
    BLEND_GROUP,
    BLEND_WGSL,
    CLEAR_COUNTERS_GROUP,
    CLEAR_COUNTERS_WGSL,
    COMPACT_GROUP,
    COMPACT_WGSL,
    COUNTERS_BYTES,
    ENTRY_POINT,
    FRAME_UNIFORM_BYTES,
    FRAME_UNIFORM_OFFSETS,
    HISTORY_CHANGES_BYTES,
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
const TEXTURE_USAGE = { copySrc: 0x01, textureBinding: 0x04, storageBinding: 0x08 } as const;
const WORD_BYTES = 4;

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
    /** The compacted probes: word 0 the count, then one packed seed each from byte `PROBE_SPAWN_SEEDS_OFFSET`. */
    readonly probeSpawn: GPUBuffer;
    /**
     * The probe rays (`RcProbeRays`, wgsl/screen-probes.ts): the ray count in word 0, then from byte
     * `PROBE_RAYS_HEADER_BYTES` one `PROBE_RAY_BYTES` record per ray: origin, flags, direction as three
     * halves (`g_ScreenProbes_ProbeSpawnSampleBuffer`) and the hit record. Shaped for phase 3's world cache.
     */
    readonly probeRays: GPUBuffer;
    /** The ray results (radiance and hit distance as four halves): `rayScratch.rayResults`. */
    readonly rayScratch: RayScratch;
    /** Each probe tile's nine SH coefficients (`packHalf4`), in a 3 × 3 block of this `rg32uint` texture (register D26). */
    readonly probeSH: GPUTexture;
    /** The interpolated irradiance, screen-sized `rgba16float`: rgb E, alpha Capsaicin's denoiser hint. */
    readonly irradiance: GPUTexture;
    /** Capsaicin's append counters (`COUNTER_NAMES`' order), one word each. */
    readonly counters: GPUBuffer;
    /** The history's buffers (RC4.1): last frame's probes, the off-screen cache, the empty and override tiles. */
    readonly history: HistoryResources;
    /** The cache's per-tile list counts and the LRU flags (the scans' inputs, sized for the largest screen). */
    readonly cachedTileListCount: GPUBuffer;
    readonly cachedTileLRUFlags: GPUBuffer;
}

/** Kernels compiled in `prepare`. */
interface Kernels {
    readonly clearCounters: ComputeKernel;
    readonly spawn: ComputeKernel;
    readonly compact: ComputeKernel;
    readonly sample: ComputeKernel;
    readonly populate: ComputeKernel;
    readonly blend: ComputeKernel;
    readonly debug: ComputeKernel;
}

/** What `prepare` makes: everything that doesn't depend on the screen's size. */
interface Prepared {
    readonly kernels: Kernels;
    /** Spawn tiles the buffers below hold: enough for the largest screen the device can draw. */
    readonly spawnTileCapacity: number;
    readonly frameUniforms: GPUBuffer;
    readonly counters: GPUBuffer;
    readonly spawnFlags: GPUBuffer;
    readonly spawnSeeds: GPUBuffer;
    readonly probeSpawn: GPUBuffer;
    readonly compaction: GpuCompaction;
    readonly compactArgs: GpuIndirectArgs;
    readonly clearCountersGroup: GPUBindGroup;
    readonly debugUniforms: GPUBuffer;
    /** `RcHistoryChanges`: this frame's changed boxes for the blend's running mean (D46). */
    readonly historyChanges: GPUBuffer;
    /** The probes' history: reprojection, the off-screen cache, the patch, the reorder (RC4.1). */
    readonly history: ScreenProbeHistory;
    readonly bytes: number;
}

/** What one screen size needs. */
interface ScreenResources {
    readonly layout: ScreenProbeLayout;
    readonly probeAtlas: GPUTexture;
    readonly probeMask: GPUTexture;
    readonly probeMaskLevel0: GPUTextureView;
    readonly probeRays: GPUBuffer;
    readonly rayScratch: RayScratch;
    readonly history: HistoryResources;
    readonly probeSH: GPUTexture;
    readonly compactGroup: GPUBindGroup;
    readonly bytes: number;
}

/** The previous frame's camera, for the reprojection (`g_PreviousViewProjectionInverse`, `g_PreviousEye`). */
interface PreviousCamera {
    readonly inverseViewProjection: number[];
    readonly eye: readonly [number, number, number];
}

/** Bind groups over the host's screen textures, rebuilt when the host hands over other textures. */
interface ScreenInputGroups {
    readonly depth: GPUTexture;
    readonly normal: GPUTexture;
    /** `FrameScreen.normalEncoding` was `unorm` (the SH debug view decodes the normals as the probes do). */
    readonly normalUnorm: boolean;
    readonly resources: ScreenResources;
    readonly spawn: GPUBindGroup;
    readonly sample: GPUBindGroup;
    readonly populate: GPUBindGroup;
    readonly blend: GPUBindGroup;
}

export class RadianceCacheTechnique implements GITechnique<RadianceCacheSettings> {
    readonly id = TECHNIQUE_ID;
    /**
     * Rays through the platform's tracer; the host's depth, normals and
     * motion every frame (`encodeScreen`): the probes are reprojected by
     * `FrameScreen.velocity` (RC4.1).
     */
    readonly requirements: TechniqueRequirements = { queries: ['rays'], screen: { velocity: true } };
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
            description: "Each probe's 8 × 8 cells of incoming light, drawn in its own 8 × 8 pixel tile; tiles without a probe this frame are a dark checkerboard.",
        },
        { kind: 'screen', id: RADIANCE_CACHE_DEBUG_VIEW.probeMask, label: 'Probe mask', description: 'Which 8 × 8 tiles hold a probe this frame: a colour per probe, its seed pixel white.' },
        {
            kind: 'screen',
            id: RADIANCE_CACHE_DEBUG_VIEW.sphericalHarmonics,
            label: 'Probe SH',
            description: "The light of each tile's probe as its nine SH coefficients hold it, along the pixel's normal, as a white surface would reflect it.",
        },
    ];
    readonly msPerUnit: number | undefined = undefined;

    private context: TechniqueContext | null = null;
    private prepared: Prepared | null = null;
    /** The world-cache slot's backend, made in `prepare`: Capsaicin's hash grid (RC3.2) or, with `worldCache: 'sharc'`, the SHaRC cache (P6.1). */
    private cache: WorldCache | null = null;
    /** `resetHistory` empties the cache at the next `encodeScreen` (it has no encoder of its own). */
    private worldCacheResetPending = false;
    private screenResources: ScreenResources | null = null;
    private inputGroups: ScreenInputGroups | null = null;
    /** SH projection and per-pixel interpolation (RC2.2): made in `prepare`. */
    private irradiance: ScreenIrradiance | null = null;
    private latestFrameIndex = 0;
    /** The camera of the latest `encodeScreen`, which the next one reprojects from. */
    private previousCamera: PreviousCamera | null = null;
    private readonly frameWords = new ArrayBuffer(FRAME_UNIFORM_BYTES);
    /** Scene changes since the last screen stage, for the running means (D46). */
    private pendingChanges: HistoryChanges = NO_HISTORY_CHANGES;
    private disposed = false;

    constructor(settings: Partial<RadianceCacheSettings> = {}) {
        // The SHaRC overrides are copied too, so changing them never touches the frozen defaults.
        this.settings = { ...DEFAULT_RADIANCE_CACHE_SETTINGS, ...settings, sharc: { ...settings.sharc } };
    }

    attach(context: TechniqueContext): void {
        this.context = context;
    }

    async prepare(): Promise<void> {
        const context = this.context;
        if (!context) throw new Error('@pmndrs/gi: attach the radiance cache (gi.addTechnique) before prepare');
        if (this.disposed) throw new Error('@pmndrs/gi: this radiance cache was disposed');
        const device = context.device;

        // The world cache first: a device that can't hold it gets GILimitError before anything else is made.
        const cache = await this.prepareWorldCache(context);

        // gi/direct first: the populate kernel includes its side of the ray scratch (the probe
        // rays' radiance slots; since RC3.2 nothing goes on its work list).
        await context.shade.direct.prepare();
        const workList = context.shade.direct.defaultWorkList;
        const kernel = (label: string, code: string, entryPoint: string, group: GPUBindGroupLayoutDescriptor, modules: readonly WgslModule[] = []) =>
            context.kernels.computeKernel({ label: `${TECHNIQUE_ID}/${label}`, modules, code, entryPoint, ownGroups: [group] });
        const [clearCounters, spawn, compact, sample, populate, blend, debug] = await Promise.all([
            kernel('ClearCounters', CLEAR_COUNTERS_WGSL, ENTRY_POINT.clearCounters, CLEAR_COUNTERS_GROUP),
            kernel('SpawnScreenProbes', SPAWN_WGSL, ENTRY_POINT.spawn, SPAWN_GROUP),
            // The platform's GiIndirectArgs and giIndirectItemIndex (code only, nothing bound).
            kernel('CompactScreenProbes', COMPACT_WGSL, ENTRY_POINT.compact, COMPACT_GROUP, [context.compute.indirectModule]),
            kernel('SampleScreenProbes', SAMPLE_WGSL, ENTRY_POINT.sample, SAMPLE_GROUP),
            // Hit shading brings the scene copy and the rays in as its dependencies.
            kernel(`PopulateScreenProbes (${workList})`, populateWgsl(context.shade.direct.traceWgsl('rcRayScratch', workList)), ENTRY_POINT.populate, POPULATE_GROUP, [context.shade.module]),
            kernel('BlendScreenProbes', BLEND_WGSL, ENTRY_POINT.blend, BLEND_GROUP),
            kernel('debug-view', DEBUG_WGSL, DEBUG_ENTRY_POINT, DEBUG_GROUP),
        ]);
        const kernels: Kernels = { clearCounters, spawn, compact, sample, populate, blend, debug };
        const irradiance = await ScreenIrradiance.create(context);

        // The spawn tiles of the largest screen the device can draw: a few MB at most, so a
        // resize never has to wait for new utilities (their creation is asynchronous).
        const largestScreen = device.limits.maxTextureDimension2D;
        const spawnTileCapacity = Math.ceil(largestScreen / PROBE_SPAWN_TILE_SIZE) ** 2;
        const storage = (label: string, bytes: number, extraUsage = 0): GPUBuffer =>
            device.createBuffer({ label: `${TECHNIQUE_ID} ${label}`, size: bytes, usage: BUFFER_USAGE.storage | BUFFER_USAGE.copySrc | extraUsage });
        const spawnFlags = storage('spawn flags', spawnTileCapacity * WORD_BYTES);
        const spawnSeeds = storage('spawn seeds', spawnTileCapacity * WORD_BYTES);
        const probeSpawn = storage('probe spawn', PROBE_SPAWN_SEEDS_OFFSET + spawnTileCapacity * WORD_BYTES, BUFFER_USAGE.copyDst);
        const counters = storage('counters', COUNTERS_BYTES);
        const frameUniforms = device.createBuffer({ label: `${TECHNIQUE_ID} frame`, size: FRAME_UNIFORM_BYTES, usage: BUFFER_USAGE.uniform | BUFFER_USAGE.copyDst });
        const debugUniforms = device.createBuffer({ label: `${TECHNIQUE_ID} debug view`, size: DEBUG_UNIFORM_BYTES, usage: BUFFER_USAGE.uniform | BUFFER_USAGE.copyDst });
        const historyChanges = device.createBuffer({ label: `${TECHNIQUE_ID} history changes`, size: HISTORY_CHANGES_BYTES, usage: BUFFER_USAGE.uniform | BUFFER_USAGE.copyDst });
        const compaction = await context.compute.createCompaction({ label: `${TECHNIQUE_ID} spawn compaction`, flags: spawnFlags, capacity: spawnTileCapacity });
        const compactArgs = await context.compute.createIndirectArgs({
            label: `${TECHNIQUE_ID} CompactScreenProbes args`,
            count: compaction.count,
            itemsPerWorkgroup: WORKGROUP_SIZE.spawn,
        });
        const history = await ScreenProbeHistory.create(context, counters);

        this.releaseScreen();
        this.releasePrepared();
        this.irradiance = irradiance;
        this.cache = cache;
        this.previousCamera = null;
        this.prepared = {
            kernels,
            spawnTileCapacity,
            frameUniforms,
            counters,
            spawnFlags,
            spawnSeeds,
            probeSpawn,
            compaction,
            compactArgs,
            clearCountersGroup: device.createBindGroup({ label: `${TECHNIQUE_ID}/ClearCounters`, layout: clearCounters.ownLayouts[0], entries: [{ binding: 0, resource: { buffer: counters } }] }),
            debugUniforms,
            historyChanges,
            history,
            bytes: spawnFlags.size + spawnSeeds.size + probeSpawn.size + counters.size + frameUniforms.size + debugUniforms.size,
        };
    }

    /**
     * No world-space work: the world cache runs in `encodeScreen`, where the
     * probe rays are (Capsaicin updates it from the screen probes' hits), so
     * every unit is left unused. The frame's scene changes are kept for the
     * screen stage, whose frame carries none: where they are, the probe
     * cells' running means start over (D46).
     */
    encode(encoder: GPUCommandEncoder, frame: TechniqueFrame): number {
        void encoder;
        this.pendingChanges = mergeHistoryChanges(this.pendingChanges, frame.changes);
        return 0;
    }

    encodeScreen(encoder: GPUCommandEncoder, frame: TechniqueScreenFrame): void {
        const context = this.context;
        const prepared = this.prepared;
        if (!context || !prepared) throw new Error('@pmndrs/gi: prepare the radiance cache before encodeScreen');
        const screen = frame.screen;
        const resources = this.resourcesFor(context, prepared, screen);
        const groups = this.inputGroupsFor(context.device, prepared, resources, screen);
        const layout = resources.layout;
        if (layout.maxProbeSpawnCount > prepared.spawnTileCapacity) {
            throw new GILimitError('maxTextureDimension2D', Math.max(screen.width, screen.height), context.device.limits.maxTextureDimension2D);
        }
        this.latestFrameIndex = frame.frame >>> 0;
        this.writeFrameUniforms(context.device, prepared, layout, frame);
        context.device.queue.writeBuffer(prepared.historyChanges, 0, historyChangesWords(this.pendingChanges));
        this.pendingChanges = NO_HISTORY_CHANGES;
        const { kernels, history } = prepared;
        const timer = context.timer;
        const maxPerDimension = context.device.limits.maxComputeWorkgroupsPerDimension;

        // 1: a fresh frame (ClearCounters, gi1.cpp:2250-2251).
        const clearPass = encoder.beginComputePass({ label: `${TECHNIQUE_ID}/clear-counters` });
        kernels.clearCounters.bind(clearPass);
        clearPass.setBindGroup(1, prepared.clearCountersGroup);
        clearPass.dispatchWorkgroups(1);
        clearPass.end();

        // 2: last frame's probes moved to this frame, and the cache's tiles on the screen (RC4.1).
        history.encodeReprojection(encoder, {
            frameUniforms: prepared.frameUniforms,
            counters: prepared.counters,
            screen,
            probeAtlas: resources.probeAtlas,
            probeMask: resources.probeMask,
            probeSH: resources.probeSH,
            probeSpawn: prepared.probeSpawn,
        });

        // 3: the spawn and its compaction.
        const spawnPass = encoder.beginComputePass({ label: `${TECHNIQUE_ID}/spawn`, timestampWrites: timer.computePass(`${TECHNIQUE_ID}/spawn`) });
        kernels.spawn.bind(spawnPass);
        spawnPass.setBindGroup(1, groups.spawn);
        spawnPass.dispatchWorkgroups(Math.ceil(layout.maxProbeSpawnCount / WORKGROUP_SIZE.spawn));
        prepared.compaction.encode(spawnPass, layout.maxProbeSpawnCount);
        prepared.compactArgs.encode(spawnPass);
        kernels.compact.bind(spawnPass);
        spawnPass.setBindGroup(1, resources.compactGroup);
        spawnPass.dispatchWorkgroupsIndirect(prepared.compactArgs.buffer, 0);
        spawnPass.end();
        // The probe count in front of the list, where the per-ray kernels read it.
        encoder.copyBufferToBuffer(prepared.compaction.count.buffer, prepared.compaction.count.offset, prepared.probeSpawn, 0, WORD_BYTES);

        // 3-4: the empty tiles take rays from tiles that have history, then a direction per ray, then the rays.
        const tracePass = encoder.beginComputePass({ label: `${TECHNIQUE_ID}/sample-populate`, timestampWrites: timer.computePass(`${TECHNIQUE_ID}/sample-populate`) });
        history.encodePatch(tracePass);
        kernels.sample.bind(tracePass);
        tracePass.setBindGroup(1, groups.sample);
        dispatchSplit(tracePass, layout.maxRayCount / WORKGROUP_SIZE.probe, maxPerDimension);
        kernels.populate.bind(tracePass);
        tracePass.setBindGroup(1, groups.populate);
        dispatchSplit(tracePass, layout.maxRayCount / WORKGROUP_SIZE.populate, maxPerDimension);
        tracePass.end();

        // 5: the world cache lights the rays' hits.
        this.encodeWorldCache(encoder, frame, resources);

        // 6: the rays into the probe atlas, the mask and the off-screen cache.
        const blendPass = encoder.beginComputePass({ label: `${TECHNIQUE_ID}/blend`, timestampWrites: timer.computePass(`${TECHNIQUE_ID}/blend`) });
        kernels.blend.bind(blendPass);
        blendPass.setBindGroup(1, groups.blend);
        dispatchSplit(blendPass, layout.maxRayCount / WORKGROUP_SIZE.probe, maxPerDimension);
        blendPass.end();

        // 7: the mask's mips, the SH and the irradiance per pixel (RC2.2).
        this.encodeIrradiance(encoder, frame, prepared, resources);

        // 6 and 8: the cache's LRU order for the next frame, and this frame's depth and normals kept.
        history.encodeReorder(encoder, screen);
        this.previousCamera = { inverseViewProjection: cameraRelativeInverseViewProjection(frame.camera), eye: [frame.camera.position[0], frame.camera.position[1], frame.camera.position[2]] };
    }

    /** The passes after the blend (screen-irradiance.ts), on this frame's probes and screen. */
    private encodeIrradiance(encoder: GPUCommandEncoder, frame: TechniqueScreenFrame, prepared: Prepared, resources: ScreenResources): void {
        const irradiance = this.irradiance;
        if (!irradiance) throw new Error('@pmndrs/gi: prepare the radiance cache before encodeScreen');
        irradiance.encode(
            encoder,
            {
                layout: resources.layout,
                frameUniforms: prepared.frameUniforms,
                probeSpawn: prepared.probeSpawn,
                probeAtlas: resources.probeAtlas,
                probeMask: resources.probeMask,
                depth: frame.screen.depth,
                normal: frame.screen.normal,
            },
            frame.camera,
            this.settings.interpolationJitter,
        );
    }

    /** What the shading module binds: the latest frame's textures, or 1 × 1 placeholders before the first. */
    private shadingTextures(): { irradiance: GPUTexture; probeAtlas: GPUTexture; probeMask: GPUTexture; probeSH: GPUTexture } {
        const irradiance = this.irradiance;
        if (!irradiance) throw new Error('@pmndrs/gi: the radiance cache has no shading textures before prepare');
        const placeholders = irradiance.placeholders;
        return {
            irradiance: irradiance.irradiance ?? placeholders.irradiance,
            probeAtlas: this.screenResources?.probeAtlas ?? placeholders.probeAtlas,
            probeMask: this.screenResources?.probeMask ?? placeholders.probeMask,
            probeSH: irradiance.probeSH ?? placeholders.probeSH,
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
        if (!context || !prepared || !resources || !inputs || !probeSH) throw new Error('@pmndrs/gi: the radiance cache has no probes to show before its first encodeScreen');
        const { width, height } = resources.layout;
        if (target.width < width || target.height < height) throw new RangeError(`@pmndrs/gi: the debug view needs a target of at least ${width} × ${height}`);
        const words = new ArrayBuffer(DEBUG_UNIFORM_BYTES);
        const uints = new Uint32Array(words);
        const offsets = DEBUG_UNIFORM_OFFSETS;
        uints.set([width, height], offsets.size / WORD_BYTES);
        uints[offsets.view / WORD_BYTES] = view;
        new Float32Array(words)[offsets.exposure / WORD_BYTES] = exposure;
        uints[offsets.normalUnorm / WORD_BYTES] = inputs.normalUnorm ? 1 : 0;
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
        if (!prepared || !resources || !irradiance?.probeSH || !irradiance.irradiance) return null;
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
            counters: prepared.counters,
            history: resources.history,
            cachedTileListCount: resources.history.scanBuffers.cachedTileListCount,
            cachedTileLRUFlags: resources.history.scanBuffers.cachedTileLRUFlags,
        };
    }

    /**
     * Capsaicin's hash grid once prepared, for tests and debug views; null
     * before `prepare` or when another backend fills the slot (P6.1: see
     * {@link worldCacheBackend}).
     */
    get worldCache(): HashGridCache | null {
        return this.cache instanceof HashGridCache ? this.cache : null;
    }

    /** Whichever backend fills the world-cache slot (`settings.worldCache`) once prepared; null before `prepare`. */
    get worldCacheBackend(): WorldCache | null {
        return this.cache;
    }

    get counters(): Readonly<Record<string, number>> {
        const layout = this.screenResources?.layout;
        // What the CPU knows without a readback: the spawned count stays on the GPU. The world
        // cache adds its capacity and memory (its per-frame counts stay on the GPU too).
        const screen: Record<string, number> = layout ? { 'spawn tiles': layout.maxProbeSpawnCount, 'probe rays at most': layout.maxRayCount } : {};
        return { ...screen, ...worldCacheCounters(this.cache) };
    }

    /** GPU bytes: the screen probes' buffers and the world cache's (about 1 GiB at Capsaicin's default capacity). */
    get memoryBytes(): number {
        const prepared = this.prepared;
        if (!prepared) return 0;
        const screenProbes = prepared.bytes + prepared.compaction.memoryBytes + prepared.compactArgs.memoryBytes + prepared.history.memoryBytes + (this.screenResources?.bytes ?? 0);
        const screenIrradiance = this.irradiance?.memoryBytes ?? 0;
        return screenProbes + screenIrradiance + (this.cache?.memoryBytes ?? 0);
    }

    /**
     * Forgets everything kept between frames, at the next `encodeScreen`: the
     * world cache's cells and the screen probes' history.
     */
    resetHistory(): void {
        this.worldCacheResetPending = true;
        this.resetScreenHistory();
    }

    /**
     * Forgets the screen probes' history at the next `encodeScreen` (a cut):
     * no probe is reprojected and the off-screen cache is emptied, so the
     * frame runs as the first one does. The world cache keeps its cells.
     */
    resetScreenHistory(): void {
        this.prepared?.history.reset();
        this.previousCamera = null;
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
     * device that can't hold it (register D14). With `worldCache: 'sharc'`,
     * the SHaRC cache at its settings (`settings.sharc`, P6.1) instead.
     */
    private async prepareWorldCache(context: TechniqueContext): Promise<WorldCache> {
        // The backend the settings name (P6.1); a change of backend disposes the other one.
        const wanted = this.settings.worldCache === 'sharc' ? 'sharc' : 'hash-grid';
        if (this.cache && this.cache.id !== wanted) {
            this.cache.dispose();
            this.cache = null;
        }
        const cache = this.cache ?? (wanted === 'sharc' ? new SharcCache(this.settings) : new HashGridCache(this.settings));
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
        if (!cache) throw new Error('@pmndrs/gi: prepare the radiance cache before encodeScreen');
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
        const layout = screenProbeLayout(screen.width, screen.height);
        this.releaseScreen();
        // g_ScreenProbes_ProbeBuffer (gi1.cpp:131-144): read by RC2.2's projection, SampleScreenProbes' taps and
        // the debug view; copied by tests and, as last frame's, by the history.
        const probeAtlas = device.createTexture({
            label: `${TECHNIQUE_ID} probe atlas`,
            size: [layout.atlasSize[0], layout.atlasSize[1]],
            format: 'rgba16float',
            usage: TEXTURE_USAGE.storageBinding | TEXTURE_USAGE.textureBinding | TEXTURE_USAGE.copySrc,
        });
        // g_ScreenProbes_ProbeMaskBuffer (gi1.cpp:146-180): with its mip chain, which RC2.2's FilterProbeMask fills.
        const probeMask = device.createTexture({
            label: `${TECHNIQUE_ID} probe mask`,
            size: [layout.probeCount[0], layout.probeCount[1]],
            format: 'r32uint',
            mipLevelCount: layout.probeMaskMipCount,
            usage: TEXTURE_USAGE.storageBinding | TEXTURE_USAGE.textureBinding | TEXTURE_USAGE.copySrc,
        });
        const probeMaskLevel0 = probeMask.createView({ label: `${TECHNIQUE_ID} probe mask level 0`, baseMipLevel: 0, mipLevelCount: 1 });
        const probeRays = device.createBuffer({
            label: `${TECHNIQUE_ID} probe rays`,
            size: PROBE_RAYS_HEADER_BYTES + layout.maxRayCount * PROBE_RAY_BYTES,
            usage: BUFFER_USAGE.storage | BUFFER_USAGE.copySrc,
        });
        // The rays' radiance slots. Nothing goes on gi/direct's work list since RC3.2 (the world
        // cache lights the hits), so it has no room.
        const rayScratch = context.shade.direct.createScratch({ label: `${TECHNIQUE_ID} probe rays`, rays: layout.maxRayCount, workItems: 0 });
        const maskBytes = mipChainTexels(layout.probeCount[0], layout.probeCount[1], layout.probeMaskMipCount) * WORD_BYTES;
        // The history of this size (empty: the first frame runs without one), and the SH the
        // reprojection copies into.
        const history = prepared.history.resourcesFor(layout);
        const probeSH = this.irradiance?.probeSHFor(layout);
        if (!probeSH) throw new Error('@pmndrs/gi: prepare the radiance cache before encodeScreen');
        const resources: ScreenResources = {
            layout,
            probeAtlas,
            probeMask,
            probeMaskLevel0,
            probeRays,
            rayScratch,
            history,
            probeSH,
            compactGroup: device.createBindGroup({
                label: `${TECHNIQUE_ID}/CompactScreenProbes`,
                layout: prepared.kernels.compact.ownLayouts[0],
                entries: [
                    { binding: 0, resource: { buffer: prepared.compactArgs.buffer } },
                    { binding: 1, resource: { buffer: prepared.compaction.indices } },
                    { binding: 2, resource: { buffer: prepared.spawnSeeds } },
                    { binding: 3, resource: { buffer: prepared.probeSpawn } },
                    { binding: 4, resource: probeMaskLevel0 },
                    { binding: 5, resource: { buffer: prepared.counters } },
                    { binding: 6, resource: { buffer: history.overrideTiles } },
                ],
            }),
            bytes: layout.atlasSize[0] * layout.atlasSize[1] * 8 + maskBytes + probeRays.size + rayScratch.bytes,
        };
        this.screenResources = resources;
        return resources;
    }

    private inputGroupsFor(device: GPUDevice, prepared: Prepared, resources: ScreenResources, screen: FrameScreen): ScreenInputGroups {
        const current = this.inputGroups;
        const normalUnorm = screen.normalEncoding === 'unorm';
        if (current && current.resources === resources && current.depth === screen.depth && current.normal === screen.normal && current.normalUnorm === normalUnorm) return current;
        checkScreenTexture(screen.depth, 'depth');
        checkScreenTexture(screen.normal, 'normal');
        const depth = screenTextureView(screen.depth);
        const normal = screenTextureView(screen.normal);
        const uniforms = { buffer: prepared.frameUniforms };
        const { kernels } = prepared;
        const { history } = resources;
        const { scanBuffers } = history;
        const counters = { buffer: prepared.counters };
        const group = (label: string, kernel: ComputeKernel, entries: GPUBindingResource[]): GPUBindGroup =>
            device.createBindGroup({ label: `${TECHNIQUE_ID}/${label}`, layout: kernel.ownLayouts[0], entries: entries.map((resource, binding) => ({ binding, resource })) });
        const groups: ScreenInputGroups = {
            depth: screen.depth,
            normal: screen.normal,
            normalUnorm,
            resources,
            spawn: group('SpawnScreenProbes', kernels.spawn, [uniforms, depth, normal, { buffer: prepared.spawnSeeds }, { buffer: prepared.spawnFlags }]),
            sample: group('SampleScreenProbes', kernels.sample, [
                uniforms,
                normal,
                { buffer: prepared.probeSpawn },
                { buffer: resources.probeRays },
                depth,
                resources.probeAtlas.createView(),
                resources.probeMaskLevel0,
                history.previousProbeAtlas.createView(),
                { buffer: scanBuffers.cachedTileListCount },
                { buffer: scanBuffers.cachedTileListIndex },
                { buffer: history.cachedTileList },
                { buffer: history.cachedTileLRU },
                { buffer: history.cachedTileIndex },
                history.cachedTileAtlas.createView(),
                { buffer: scanBuffers.cachedTileLRUFlags },
                counters,
                { buffer: history.spawnProbe },
                { buffer: history.evictedRadiance },
                { buffer: history.cellCounts },
                { buffer: history.previousCellCounts },
                { buffer: history.cachedCellCounts },
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
                { buffer: prepared.probeSpawn },
                { buffer: resources.probeRays },
                resources.rayScratch.rayResults,
                resources.probeAtlas.createView(),
                resources.probeMaskLevel0,
                depth,
                history.previousProbeAtlas.createView(),
                { buffer: history.spawnProbe },
                { buffer: history.evictedRadiance },
                counters,
                { buffer: history.cachedTileLRU },
                { buffer: scanBuffers.cachedTileLRUFlags },
                { buffer: history.cachedTileMRU },
                { buffer: history.cachedTileIndex },
                history.cachedTileAtlas.createView(),
                { buffer: prepared.historyChanges },
                { buffer: history.cellCounts },
                { buffer: history.previousCellCounts },
                { buffer: history.cachedCellCounts },
            ]),
        };
        this.inputGroups = groups;
        return groups;
    }

    private writeFrameUniforms(device: GPUDevice, prepared: Prepared, layout: ScreenProbeLayout, frame: TechniqueScreenFrame): void {
        const words = this.frameWords;
        const floats = new Float32Array(words);
        const uints = new Uint32Array(words);
        // RcFrame's fields by name (FRAME_UNIFORM_OFFSETS are bytes; the arrays index words).
        const word = (field: keyof typeof FRAME_UNIFORM_OFFSETS): number => FRAME_UNIFORM_OFFSETS[field] / WORD_BYTES;
        // With the camera's jitter when the host drew its depth with one (FrameCamera.jitter, camera.ts).
        floats.set(cameraRelativeInverseViewProjection(frame.camera), word('inverseViewProjection'));
        floats.set(frame.camera.position, word('eye'));
        uints[word('frameIndex')] = this.latestFrameIndex;
        uints.set([layout.width, layout.height], word('bufferDimensions'));
        uints.set([layout.probeCount[0], layout.probeCount[1]], word('probeCount'));
        uints[word('maxProbeSpawnCount')] = layout.maxProbeSpawnCount;
        uints[word('normalUnorm')] = frame.screen.normalEncoding === 'unorm' ? 1 : 0;
        uints[word('reversedDepth')] = frame.screen.reversedDepth === true ? 1 : 0;
        uints[word('useDirectLighting')] = this.settings.useDirectLighting ? 1 : 0;
        // The camera the history was made with; with none (the first frame), this one.
        const previous = this.previousCamera;
        floats.set(previous?.inverseViewProjection ?? cameraRelativeInverseViewProjection(frame.camera), word('previousInverseViewProjection'));
        floats.set(cameraRelativeViewProjection(frame.camera), word('viewProjection'));
        floats.set(previous?.eye ?? frame.camera.position, word('previousEye'));
        floats[word('cellSize')] = screenProbeCellSize(frame.camera, layout.width, layout.height);
        device.queue.writeBuffer(prepared.frameUniforms, 0, words);
    }

    private releaseScreen(): void {
        const resources = this.screenResources;
        if (!resources) return;
        resources.probeAtlas.destroy();
        resources.probeMask.destroy();
        resources.probeRays.destroy();
        resources.rayScratch.destroy();
        this.screenResources = null;
        this.inputGroups = null;
    }

    private releasePrepared(): void {
        const prepared = this.prepared;
        if (!prepared) return;
        for (const buffer of [prepared.frameUniforms, prepared.counters, prepared.spawnFlags, prepared.spawnSeeds, prepared.probeSpawn, prepared.debugUniforms, prepared.historyChanges]) buffer.destroy();
        prepared.compaction.dispose();
        prepared.compactArgs.dispose();
        prepared.history.dispose();
        this.irradiance?.dispose();
        this.irradiance = null;
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

function mipChainTexels(width: number, height: number, levels: number): number {
    let texels = 0;
    for (let level = 0; level < levels; level++) texels += Math.max(1, width >> level) * Math.max(1, height >> level);
    return texels;
}

function checkScreenTexture(texture: GPUTexture, name: string): void {
    if ((texture.usage & TEXTURE_USAGE.textureBinding) === 0) {
        throw new Error(`@pmndrs/gi: the radiance cache reads FrameScreen.${name} in compute, so it needs TEXTURE_BINDING usage`);
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
