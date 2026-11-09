/**
 * The radiance cache: a faithful WebGPU port of AMD's GI-1.0/1.2 as Capsaicin
 * v1.3 builds it (docs/design/radiance-cache.md). This is phase 2's first
 * half (RC2.1): the screen probes, traced and blended into the probe atlas
 * every frame, with their debug views. The image (SH, interpolation, the
 * shading module) is RC2.2; the world cache is phase 3; history is phase 4.
 *
 * Per frame, in `encodeScreen` (after the host's `encode` and its depth and
 * normal pass), the passes Capsaicin runs for the screen probes
 * (`GI1::render`, gi1.cpp:2250-2542), as far as phase 2 has them:
 *
 * 1. `ClearCounters` (gi1.comp:163), and `ClearProbeMask` (gi1.comp:243), which
 *    stands in for reprojection's clearing until phase 4;
 * 2. `SpawnScreenProbes` (gi1.comp:555), the platform's stream compaction
 *    (Capsaicin's `gfxCommandScanSum`, gi1.cpp:2305) and `CompactScreenProbes`
 *    (gi1.comp:581), dispatched indirectly on the kept count;
 * 3. `SampleScreenProbes` (gi1.comp:646) and `PopulateScreenProbesMain`
 *    (gi1.comp:1160), then the platform's `gi/direct` for the hits' direct
 *    light (phase 2's staging of Capsaicin's cache-bypass path);
 * 4. `BlendScreenProbes` (gi1.comp:1167) into the probe atlas and the mask.
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
import { cameraRelativeInverseViewProjection } from './camera.ts';
import { DEFAULT_RADIANCE_CACHE_SETTINGS, PROBE_SPAWN_TILE_SIZE, screenProbeLayout, type RadianceCacheSettings, type ScreenProbeLayout } from './settings.ts';
import { RadianceCacheShadingModule } from './shading.ts';
import { DEBUG_ENTRY_POINT, DEBUG_GROUP, DEBUG_UNIFORM_BYTES, DEBUG_WGSL, type RadianceCacheDebugViewId } from './wgsl/debug.ts';
import {
    ALL_LIGHT_GROUPS,
    BLEND_GROUP,
    BLEND_WGSL,
    CLEAR_COUNTERS_GROUP,
    CLEAR_COUNTERS_WGSL,
    CLEAR_PROBE_MASK_GROUP,
    CLEAR_PROBE_MASK_WGSL,
    COMPACT_GROUP,
    COMPACT_WGSL,
    COUNTERS_BYTES,
    ENTRY_POINT,
    FRAME_UNIFORM_BYTES,
    POPULATE_GROUP,
    populateWgsl,
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
/** Bytes of one packed sample direction (`uint2`, three halves). */
const SAMPLE_BYTES = 8;

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
    /** One `vec2u` per ray: its direction as three halves (`g_ScreenProbes_ProbeSpawnSampleBuffer`). */
    readonly samples: GPUBuffer;
    /** The ray results (radiance and hit distance as four halves): `rayScratch.rayResults`. */
    readonly rayScratch: RayScratch;
}

/** Kernels compiled in `prepare`. */
interface Kernels {
    readonly clearCounters: ComputeKernel;
    readonly clearProbeMask: ComputeKernel;
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
    readonly compactGroup: GPUBindGroup;
    readonly debugUniforms: GPUBuffer;
    readonly bytes: number;
}

/** What one screen size needs. */
interface ScreenResources {
    readonly layout: ScreenProbeLayout;
    readonly probeAtlas: GPUTexture;
    readonly probeMask: GPUTexture;
    readonly probeMaskLevel0: GPUTextureView;
    readonly samples: GPUBuffer;
    readonly rayScratch: RayScratch;
    readonly clearProbeMaskGroup: GPUBindGroup;
    readonly bytes: number;
}

/** Bind groups over the host's screen textures, rebuilt when the host hands over other textures. */
interface ScreenInputGroups {
    readonly depth: GPUTexture;
    readonly normal: GPUTexture;
    readonly resources: ScreenResources;
    readonly spawn: GPUBindGroup;
    readonly sample: GPUBindGroup;
    readonly populate: GPUBindGroup;
    readonly blend: GPUBindGroup;
}

export class RadianceCacheTechnique implements GITechnique<RadianceCacheSettings> {
    readonly id = TECHNIQUE_ID;
    /** Rays through the platform's tracer; the host's depth and normals every frame (`encodeScreen`). */
    readonly requirements: TechniqueRequirements = { queries: ['rays'], screen: true };
    readonly settings: RadianceCacheSettings;
    readonly shading = new RadianceCacheShadingModule();
    /**
     * Empty in RC2.1: the contract's debug views are drawn through the shading
     * module (`giDebugColor`), which has no screen position to read the atlas
     * by. The atlas and mask views are drawn engine-free by
     * {@link encodeDebugView}; the viewer's entry for them comes with RC2.2.
     */
    readonly debugViews: readonly DebugView[] = [];
    readonly msPerUnit: number | undefined = undefined;

    private context: TechniqueContext | null = null;
    private prepared: Prepared | null = null;
    private screenResources: ScreenResources | null = null;
    private inputGroups: ScreenInputGroups | null = null;
    private latestFrameIndex = 0;
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

        // gi/direct first: the populate kernel includes its side of the ray scratch.
        await context.shade.direct.prepare();
        const workList = context.shade.direct.defaultWorkList;
        const kernel = (label: string, code: string, entryPoint: string, group: GPUBindGroupLayoutDescriptor, modules: readonly WgslModule[] = []) =>
            context.kernels.computeKernel({ label: `${TECHNIQUE_ID}/${label}`, modules, code, entryPoint, ownGroups: [group] });
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
        const kernels: Kernels = { clearCounters, clearProbeMask, spawn, compact, sample, populate, blend, debug };

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
        const compaction = await context.compute.createCompaction({ label: `${TECHNIQUE_ID} spawn compaction`, flags: spawnFlags, capacity: spawnTileCapacity });
        const compactArgs = await context.compute.createIndirectArgs({
            label: `${TECHNIQUE_ID} CompactScreenProbes args`,
            count: compaction.count,
            itemsPerWorkgroup: WORKGROUP_SIZE.spawn,
        });

        this.releaseScreen();
        this.releasePrepared();
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
            compactGroup: device.createBindGroup({
                label: `${TECHNIQUE_ID}/CompactScreenProbes`,
                layout: compact.ownLayouts[0],
                entries: [
                    { binding: 0, resource: { buffer: compactArgs.buffer } },
                    { binding: 1, resource: { buffer: compaction.indices } },
                    { binding: 2, resource: { buffer: spawnSeeds } },
                    { binding: 3, resource: { buffer: probeSpawn } },
                ],
            }),
            debugUniforms,
            bytes: spawnFlags.size + spawnSeeds.size + probeSpawn.size + counters.size + frameUniforms.size + debugUniforms.size,
        };
    }

    /** No world-space work in phase 2 (the world cache is phase 3): every unit is left unused. */
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
        this.writeFrameUniforms(context.device, prepared, layout, frame);
        const { kernels } = prepared;
        const timer = context.timer;
        const maxPerDimension = context.device.limits.maxComputeWorkgroupsPerDimension;

        // 1-2: a fresh frame, the spawn, its compaction.
        const spawnPass = encoder.beginComputePass({ label: `${TECHNIQUE_ID}/spawn`, timestampWrites: timer.computePass(`${TECHNIQUE_ID}/spawn`) });
        kernels.clearCounters.bind(spawnPass);
        spawnPass.setBindGroup(1, prepared.clearCountersGroup);
        spawnPass.dispatchWorkgroups(1);
        kernels.clearProbeMask.bind(spawnPass);
        spawnPass.setBindGroup(1, resources.clearProbeMaskGroup);
        spawnPass.dispatchWorkgroups(Math.ceil(layout.probeCount[0] / WORKGROUP_SIZE.clearProbeMask), Math.ceil(layout.probeCount[1] / WORKGROUP_SIZE.clearProbeMask));
        kernels.spawn.bind(spawnPass);
        spawnPass.setBindGroup(1, groups.spawn);
        spawnPass.dispatchWorkgroups(Math.ceil(layout.maxProbeSpawnCount / WORKGROUP_SIZE.spawn));
        prepared.compaction.encode(spawnPass, layout.maxProbeSpawnCount);
        prepared.compactArgs.encode(spawnPass);
        kernels.compact.bind(spawnPass);
        spawnPass.setBindGroup(1, prepared.compactGroup);
        spawnPass.dispatchWorkgroupsIndirect(prepared.compactArgs.buffer, 0);
        spawnPass.end();
        // The probe count in front of the list, where the per-ray kernels read it.
        encoder.copyBufferToBuffer(prepared.compaction.count.buffer, prepared.compaction.count.offset, prepared.probeSpawn, 0, WORD_BYTES);

        // 3: a direction per ray, then the rays.
        const tracePass = encoder.beginComputePass({ label: `${TECHNIQUE_ID}/sample-populate`, timestampWrites: timer.computePass(`${TECHNIQUE_ID}/sample-populate`) });
        kernels.sample.bind(tracePass);
        tracePass.setBindGroup(1, groups.sample);
        dispatchSplit(tracePass, layout.maxRayCount / WORKGROUP_SIZE.probe, maxPerDimension);
        kernels.populate.bind(tracePass);
        tracePass.setBindGroup(1, groups.populate);
        dispatchSplit(tracePass, layout.maxRayCount / WORKGROUP_SIZE.populate, maxPerDimension);
        tracePass.end();
        context.shade.direct.encode(encoder, resources.rayScratch, { frame: frame.frame, groupMask: ALL_LIGHT_GROUPS, owner: TECHNIQUE_ID, timer });

        // 4: the rays into the probe atlas and the mask.
        const blendPass = encoder.beginComputePass({ label: `${TECHNIQUE_ID}/blend`, timestampWrites: timer.computePass(`${TECHNIQUE_ID}/blend`) });
        kernels.blend.bind(blendPass);
        blendPass.setBindGroup(1, groups.blend);
        dispatchSplit(blendPass, layout.maxRayCount / WORKGROUP_SIZE.probe, maxPerDimension);
        blendPass.end();
    }

    /**
     * Draws a debug view (wgsl/debug.ts) of the latest frame's probes into
     * `target`: an `rgba8unorm` texture with `STORAGE_BINDING` usage, at least
     * the screen's size. Engine-free: the host shows the image as it likes.
     * `exposure` multiplies radiance before tone mapping (the atlas view).
     */
    encodeDebugView(encoder: GPUCommandEncoder, view: RadianceCacheDebugViewId, target: GPUTexture, exposure = 1): void {
        const context = this.context;
        const prepared = this.prepared;
        const resources = this.screenResources;
        if (!context || !prepared || !resources) throw new Error('@pmndrs/ddgi: the radiance cache has no probes to show before its first encodeScreen');
        const { width, height } = resources.layout;
        if (target.width < width || target.height < height) throw new RangeError(`@pmndrs/ddgi: the debug view needs a target of at least ${width} × ${height}`);
        const words = new ArrayBuffer(DEBUG_UNIFORM_BYTES);
        new Uint32Array(words, 0, 3).set([width, height, view]);
        new Float32Array(words, 12, 1)[0] = exposure;
        context.device.queue.writeBuffer(prepared.debugUniforms, 0, words);
        const group = context.device.createBindGroup({
            label: `${TECHNIQUE_ID}/debug-view`,
            layout: prepared.kernels.debug.ownLayouts[0],
            entries: [
                { binding: 0, resource: { buffer: prepared.debugUniforms } },
                { binding: 1, resource: resources.probeAtlas.createView() },
                { binding: 2, resource: resources.probeMask.createView({ baseMipLevel: 0, mipLevelCount: 1 }) },
                { binding: 3, resource: target.createView({ baseMipLevel: 0, mipLevelCount: 1 }) },
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
        if (!prepared || !resources) return null;
        return {
            layout: resources.layout,
            frameIndex: this.latestFrameIndex,
            probeAtlas: resources.probeAtlas,
            probeMask: resources.probeMask,
            spawnFlags: prepared.spawnFlags,
            spawnSeeds: prepared.spawnSeeds,
            probeSpawn: prepared.probeSpawn,
            samples: resources.samples,
            rayScratch: resources.rayScratch,
        };
    }

    get counters(): Readonly<Record<string, number>> {
        const layout = this.screenResources?.layout;
        if (!layout) return {};
        // What the CPU knows without a readback: the spawned count stays on the GPU.
        return { 'spawn tiles': layout.maxProbeSpawnCount, 'probe rays at most': layout.maxRayCount };
    }

    get memoryBytes(): number {
        const prepared = this.prepared;
        if (!prepared) return 0;
        return prepared.bytes + prepared.compaction.memoryBytes + prepared.compactArgs.memoryBytes + (this.screenResources?.bytes ?? 0);
    }

    /** Phase 2 keeps nothing between frames: every probe is traced afresh. */
    resetHistory(): void {}

    dispose(): void {
        this.releaseScreen();
        this.releasePrepared();
        this.disposed = true;
    }

    private resourcesFor(context: TechniqueContext, prepared: Prepared, screen: FrameScreen): ScreenResources {
        const current = this.screenResources;
        if (current && current.layout.width === screen.width && current.layout.height === screen.height) return current;
        const device = context.device;
        const layout = screenProbeLayout(screen.width, screen.height);
        this.releaseScreen();
        // g_ScreenProbes_ProbeBuffer (gi1.cpp:131-144): read by RC2.2's projection and the debug view, copied by tests.
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
        const samples = device.createBuffer({
            label: `${TECHNIQUE_ID} probe ray directions`,
            size: layout.maxRayCount * SAMPLE_BYTES,
            usage: BUFFER_USAGE.storage | BUFFER_USAGE.copySrc,
        });
        // Every ray may hit a surface that needs direct light, so the work list has room for all of them.
        const rayScratch = context.shade.direct.createScratch({ label: `${TECHNIQUE_ID} probe rays`, rays: layout.maxRayCount, workItems: layout.maxRayCount });
        const maskBytes = mipChainTexels(layout.probeCount[0], layout.probeCount[1], layout.probeMaskMipCount) * WORD_BYTES;
        const resources: ScreenResources = {
            layout,
            probeAtlas,
            probeMask,
            probeMaskLevel0,
            samples,
            rayScratch,
            clearProbeMaskGroup: device.createBindGroup({
                label: `${TECHNIQUE_ID}/ClearProbeMask`,
                layout: prepared.kernels.clearProbeMask.ownLayouts[0],
                entries: [
                    { binding: 0, resource: { buffer: prepared.frameUniforms } },
                    { binding: 1, resource: probeMaskLevel0 },
                ],
            }),
            bytes: layout.atlasSize[0] * layout.atlasSize[1] * 8 + maskBytes + samples.size + rayScratch.bytes,
        };
        this.screenResources = resources;
        return resources;
    }

    private inputGroupsFor(device: GPUDevice, prepared: Prepared, resources: ScreenResources, screen: FrameScreen): ScreenInputGroups {
        const current = this.inputGroups;
        if (current && current.resources === resources && current.depth === screen.depth && current.normal === screen.normal) return current;
        checkScreenTexture(screen.depth, 'depth');
        checkScreenTexture(screen.normal, 'normal');
        const depth = screenTextureView(screen.depth);
        const normal = screenTextureView(screen.normal);
        const uniforms = { buffer: prepared.frameUniforms };
        const { kernels } = prepared;
        const group = (label: string, kernel: ComputeKernel, entries: GPUBindingResource[]): GPUBindGroup =>
            device.createBindGroup({ label: `${TECHNIQUE_ID}/${label}`, layout: kernel.ownLayouts[0], entries: entries.map((resource, binding) => ({ binding, resource })) });
        const groups: ScreenInputGroups = {
            depth: screen.depth,
            normal: screen.normal,
            resources,
            spawn: group('SpawnScreenProbes', kernels.spawn, [uniforms, depth, normal, { buffer: prepared.spawnSeeds }, { buffer: prepared.spawnFlags }]),
            sample: group('SampleScreenProbes', kernels.sample, [uniforms, normal, { buffer: prepared.probeSpawn }, { buffer: resources.samples }]),
            populate: group('PopulateScreenProbes', kernels.populate, [
                uniforms,
                depth,
                normal,
                { buffer: prepared.probeSpawn },
                { buffer: resources.samples },
                { buffer: resources.rayScratch.buffer },
            ]),
            blend: group('BlendScreenProbes', kernels.blend, [
                uniforms,
                normal,
                { buffer: prepared.probeSpawn },
                { buffer: resources.samples },
                resources.rayScratch.rayResults,
                resources.probeAtlas.createView(),
                resources.probeMaskLevel0,
            ]),
        };
        this.inputGroups = groups;
        return groups;
    }

    private writeFrameUniforms(device: GPUDevice, prepared: Prepared, layout: ScreenProbeLayout, frame: TechniqueScreenFrame): void {
        const words = this.frameWords;
        const floats = new Float32Array(words);
        const uints = new Uint32Array(words);
        floats.set(cameraRelativeInverseViewProjection(frame.camera), 0);
        floats.set(frame.camera.position, 16);
        uints[19] = this.latestFrameIndex;
        uints.set([layout.width, layout.height, layout.probeCount[0], layout.probeCount[1], layout.maxProbeSpawnCount], 20);
        uints[25] = frame.screen.normalEncoding === 'unorm' ? 1 : 0;
        uints[26] = frame.screen.reversedDepth === true ? 1 : 0;
        uints[27] = this.settings.useDirectLighting ? 1 : 0;
        device.queue.writeBuffer(prepared.frameUniforms, 0, words);
    }

    private releaseScreen(): void {
        const resources = this.screenResources;
        if (!resources) return;
        resources.probeAtlas.destroy();
        resources.probeMask.destroy();
        resources.samples.destroy();
        resources.rayScratch.destroy();
        this.screenResources = null;
        this.inputGroups = null;
    }

    private releasePrepared(): void {
        const prepared = this.prepared;
        if (!prepared) return;
        for (const buffer of [prepared.frameUniforms, prepared.counters, prepared.spawnFlags, prepared.spawnSeeds, prepared.probeSpawn, prepared.debugUniforms]) buffer.destroy();
        prepared.compaction.dispose();
        prepared.compactArgs.dispose();
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
        throw new Error(`@pmndrs/ddgi: the radiance cache reads FrameScreen.${name} in compute, so it needs TEXTURE_BINDING usage`);
    }
}

/** A depth texture is read through its depth aspect (as `unfilterable-float`); anything else whole. */
function screenTextureView(texture: GPUTexture): GPUTextureView {
    const isDepth = texture.format.startsWith('depth');
    return texture.createView({ aspect: isDepth ? 'depth-only' : 'all', baseMipLevel: 0, mipLevelCount: 1 });
}
