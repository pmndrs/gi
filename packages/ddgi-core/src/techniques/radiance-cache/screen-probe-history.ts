/**
 * The screen probes' history (RC4.1): what carries the probes from one frame
 * to the next, as Capsaicin's `GI1::render` runs it (gi1.cpp:2256-2557).
 *
 * - **Last frame's probes:** the atlas, the mask and the SH as the frame left
 *   them (Capsaicin swaps two of each, `probe_buffer_index_`; we copy the
 *   current ones at the start of the frame, register D44), and the host's
 *   depth and normals through the platform's `GpuScreenHistory`.
 * - **The off-screen probe cache:** an atlas of cached probe tiles, each
 *   tile's position and normal, the LRU list (Capsaicin's two lists swap; we
 *   copy the reordered one back, D44) with its flags, MRU order and scans, and
 *   the per-tile lists `CountScreenProbes` and `ScatterScreenProbes` build.
 * - **Adaptive sampling:** the empty-tile and override-tile lists.
 *
 * The kernels are in wgsl/temporal-probes.ts. Sampling and blending read and
 * write this history too; their bind groups (in the technique) take the
 * buffers from {@link HistoryResources}.
 *
 * Everything here is sized per screen when the screen's size changes; the
 * platform's two scans are made once (in `prepare`, on placeholder buffers)
 * and pointed at each size's buffers with `rebind` (register D44).
 */

import type { GpuIndirectArgs, GpuPrefixScan, GpuScreenHistory, GpuWord, ScreenHistoryTextures } from '../../platform/contract/compute.ts';
import type { FrameScreen } from '../../platform/contract/frame.ts';
import type { Aabb } from '../../platform/contract/math.ts';
import type { SceneChanges } from '../../platform/contract/scene-copy.ts';
import type { TechniqueContext } from '../../platform/contract/technique.ts';
import type { ComputeKernel } from '../../platform/contract/wgsl.ts';
import type { ScreenProbeLayout } from './settings.ts';
import { SH_BLOCK_SIZE } from './wgsl/spherical-harmonics.ts';
import { CLEAR_PROBE_MASK_GROUP, HISTORY_CHANGES_BYTES, MAX_HISTORY_CHANGE_BOXES, CLEAR_PROBE_MASK_WGSL, COUNTER, ENTRY_POINT, PROBE_SPAWN_PROBE_BYTES, WORKGROUP_SIZE } from './wgsl/screen-probes.ts';
import {
    COUNT_GROUP,
    COUNT_WGSL,
    INIT_CACHED_TILE_LRU_GROUP,
    INIT_CACHED_TILE_LRU_WGSL,
    PATCH_GROUP,
    PATCH_WGSL,
    REORDER_GROUP,
    REORDER_WGSL,
    REPROJECT_GROUP,
    REPROJECT_WGSL,
    SCATTER_GROUP,
    SCATTER_WGSL,
    TEMPORAL_ENTRY_POINT,
    TEMPORAL_WORKGROUP,
} from './wgsl/temporal-probes.ts';

const BUFFER_USAGE = { storage: 0x80, copySrc: 0x04, copyDst: 0x08 } as const;
const TEXTURE_USAGE = { copySrc: 0x01, copyDst: 0x02, textureBinding: 0x04, storageBinding: 0x08 } as const;
const WORD_BYTES = 4;
const TECHNIQUE_ID = 'radiance-cache';

interface HistoryKernels {
    readonly initCachedTileLRU: ComputeKernel;
    readonly clearProbeMask: ComputeKernel;
    readonly reproject: ComputeKernel;
    readonly count: ComputeKernel;
    readonly scatter: ComputeKernel;
    readonly patch: ComputeKernel;
    readonly reorder: ComputeKernel;
}

/** The history buffers sized per screen; tests read them through the technique's `probes.history`. */
export interface HistoryResources {
    readonly layout: ScreenProbeLayout;
    /** Probe tiles of this screen: the LRU lists' length (`max_probe_count`, gi1.cpp:303). */
    readonly probeTileCount: number;
    /** g_ScreenProbes_PreviousProbeBuffer: last frame's atlas, then each spawned probe's previous radiance (SampleScreenProbes). */
    readonly previousProbeAtlas: GPUTexture;
    /** g_ScreenProbes_PreviousProbeMaskBuffer: last frame's mask, level 0 only. */
    readonly previousProbeMask: GPUTexture;
    /** g_ScreenProbes_PreviousProbeSHBuffer, as the SH texture's layout (register D26). */
    readonly previousProbeSH: GPUTexture;
    /** g_ScreenProbes_ProbeCachedTileBuffer: the cached tiles' cells, an atlas of the screen's probe tiles. */
    readonly cachedTileAtlas: GPUTexture;
    /** g_ScreenProbes_ProbeCachedTileIndexBuffer: per slot, the position (f32 bits) and packed normal (`vec4u`). */
    readonly cachedTileIndex: GPUBuffer;
    /** g_ScreenProbes_ProbeCachedTileLRUBuffer: LRU order → slot, bit 31 for an unused entry. */
    readonly cachedTileLRU: GPUBuffer;
    /** g_ScreenProbes_PreviousProbeCachedTileLRUBuffer: where ReorderScreenProbes writes next frame's order. */
    readonly nextCachedTileLRU: GPUBuffer;
    /** g_ScreenProbes_ProbeCachedTileMRUBuffer: per LRU entry used this frame, its order of use. */
    readonly cachedTileMRU: GPUBuffer;
    /** g_ScreenProbes_ProbeCachedTileListBuffer, ...ListElementBuffer: the cached tiles listed per probe tile. */
    readonly cachedTileList: GPUBuffer;
    readonly cachedTileListElements: GPUBuffer;
    /** g_ScreenProbes_EmptyTileBuffer, g_ScreenProbes_OverrideTileBuffer. */
    readonly emptyTiles: GPUBuffer;
    readonly overrideTiles: GPUBuffer;
    /** g_ScreenProbes_ProbeSpawnProbeBuffer: per spawned probe, the evicted and updated LRU entries and the replaced probe's seed. */
    readonly spawnProbe: GPUBuffer;
    /** The replaced probes' cells, a packed radiance per ray, staged by Sample for Blend (register D43). */
    readonly evictedRadiance: GPUBuffer;
    /** Per ray slot: the rays its cell expected this frame under ray guiding (f32; register D62). */
    readonly expectedRays: GPUBuffer;
    /**
     * The history sample count of each atlas texel (D46): this frame's, last
     * frame's (then each spawned probe's previous count, from Sample), and the
     * cached tiles'. One word a texel, the atlas's row order.
     */
    readonly cellCounts: GPUBuffer;
    readonly previousCellCounts: GPUBuffer;
    readonly cachedCellCounts: GPUBuffer;
    /** The buffers the platform's scans read and write for this size. */
    readonly scanBuffers: HistoryScanBuffers;
    readonly bytes: number;
}

/** The buffers the platform's scans are bound to, sized to the screen's probe tiles. */
export interface HistoryScanBuffers {
    /** g_ScreenProbes_ProbeCachedTileListCountBuffer and ...ListIndexBuffer (its exclusive scan). */
    readonly cachedTileListCount: GPUBuffer;
    readonly cachedTileListIndex: GPUBuffer;
    /** g_ScreenProbes_ProbeCachedTileLRUFlagBuffer and ...LRUIndexBuffer (its scan), one word more than the tiles. */
    readonly cachedTileLRUFlags: GPUBuffer;
    readonly cachedTileLRUIndex: GPUBuffer;
}

/** What the frame's history passes read from the technique. */
export interface HistoryFrameInputs {
    readonly frameUniforms: GPUBuffer;
    readonly counters: GPUBuffer;
    readonly screen: FrameScreen;
    readonly probeAtlas: GPUTexture;
    readonly probeMask: GPUTexture;
    readonly probeSH: GPUTexture;
    /** The compacted probe list (`RcProbeSpawn`), which the patch rewrites. */
    readonly probeSpawn: GPUBuffer;
}

export class ScreenProbeHistory {
    private sized: HistoryResources | null = null;
    /** The sized resources hold last frame's probes (a frame ran since they were made, and no reset). */
    private sizedValid = false;
    private resetPending = false;
    private groups: { readonly key: readonly unknown[]; readonly value: FrameGroups } | null = null;

    private constructor(
        private readonly context: TechniqueContext,
        private readonly kernels: HistoryKernels,
        private readonly placeholders: readonly GPUBuffer[],
        private readonly listScan: GpuPrefixScan,
        private readonly lruScan: GpuPrefixScan,
        private readonly scatterArgs: GpuIndirectArgs,
        private readonly patchArgs: GpuIndirectArgs,
        readonly screenHistory: GpuScreenHistory,
    ) {}

    /** Compiles the kernels and makes the scans, the indirect arguments and the screen history. */
    static async create(context: TechniqueContext, counters: GPUBuffer): Promise<ScreenProbeHistory> {
        const kernel = (label: string, code: string, entryPoint: string, group: GPUBindGroupLayoutDescriptor, withIndirect = false) =>
            context.kernels.computeKernel({ label: `${TECHNIQUE_ID}/${label}`, modules: withIndirect ? [context.compute.indirectModule] : [], code, entryPoint, ownGroups: [group] });
        const entry = TEMPORAL_ENTRY_POINT;
        const [initCachedTileLRU, clearProbeMask, reproject, count, scatter, patch, reorder] = await Promise.all([
            kernel(entry.initCachedTileLRU, INIT_CACHED_TILE_LRU_WGSL, entry.initCachedTileLRU, INIT_CACHED_TILE_LRU_GROUP),
            kernel('ClearProbeMask (previous)', CLEAR_PROBE_MASK_WGSL, ENTRY_POINT.clearProbeMask, CLEAR_PROBE_MASK_GROUP),
            kernel(entry.reproject, REPROJECT_WGSL, entry.reproject, REPROJECT_GROUP),
            kernel(entry.count, COUNT_WGSL, entry.count, COUNT_GROUP),
            kernel(entry.scatter, SCATTER_WGSL, entry.scatter, SCATTER_GROUP, true),
            kernel(entry.patch, PATCH_WGSL, entry.patch, PATCH_GROUP, true),
            kernel(entry.reorder, REORDER_WGSL, entry.reorder, REORDER_GROUP),
        ]);
        const device = context.device;
        // The scans are made on placeholders (their kernels compile now, asynchronously); each screen
        // size rebinds them to its own buffers, synchronously (resourcesFor).
        const placeholder = (label: string): GPUBuffer => device.createBuffer({ label: `${TECHNIQUE_ID} ${label} (placeholder)`, size: 16, usage: BUFFER_USAGE.storage });
        const placeholders = [placeholder('scan input'), placeholder('scan output'), placeholder('scan input'), placeholder('scan output')];
        const counterWord = (name: keyof typeof COUNTER): GpuWord => ({ buffer: counters, offset: COUNTER[name] * WORD_BYTES });
        const [listScan, lruScan, scatterArgs, patchArgs, screenHistory] = await Promise.all([
            context.compute.createPrefixScan({ label: `${TECHNIQUE_ID} cached tile lists`, input: placeholders[0], output: placeholders[1], capacity: 1 }),
            context.compute.createPrefixScan({ label: `${TECHNIQUE_ID} cached tile LRU`, input: placeholders[2], output: placeholders[3], capacity: 1 }),
            context.compute.createIndirectArgs({ label: `${TECHNIQUE_ID} ScatterScreenProbes args`, count: counterWord('cachedTileListElementCount'), itemsPerWorkgroup: TEMPORAL_WORKGROUP.list }),
            context.compute.createIndirectArgs({ label: `${TECHNIQUE_ID} PatchScreenProbes args`, count: counterWord('emptyTileCount'), itemsPerWorkgroup: TEMPORAL_WORKGROUP.list }),
            context.compute.createScreenHistory({ label: TECHNIQUE_ID }),
        ]);
        const kernels: HistoryKernels = { initCachedTileLRU, clearProbeMask, reproject, count, scatter, patch, reorder };
        return new ScreenProbeHistory(context, kernels, placeholders, listScan, lruScan, scatterArgs, patchArgs, screenHistory);
    }

    /** The resources of the latest screen size, or null before the first frame. */
    get resources(): HistoryResources | null {
        return this.sized;
    }

    /** The next frame starts without history: no reprojection, an empty probe cache. */
    reset(): void {
        this.resetPending = true;
    }

    get memoryBytes(): number {
        const utilities = this.listScan.memoryBytes + this.lruScan.memoryBytes + this.scatterArgs.memoryBytes + this.patchArgs.memoryBytes;
        return utilities + this.screenHistory.memoryBytes + (this.sized?.bytes ?? 0);
    }

    /** The resources for `layout`, made (empty, no history) when the size changes. */
    resourcesFor(layout: ScreenProbeLayout): HistoryResources {
        const current = this.sized;
        if (current && current.layout.width === layout.width && current.layout.height === layout.height) return current;
        this.releaseSized();
        const device = this.context.device;
        const probeTileCount = layout.probeCount[0] * layout.probeCount[1];
        const atlasTexels = layout.atlasSize[0] * layout.atlasSize[1];
        const texture = (label: string, size: readonly [number, number], format: GPUTextureFormat, usage: number): GPUTexture =>
            device.createTexture({ label: `${TECHNIQUE_ID} ${label}`, size: [size[0], size[1]], format, usage });
        const buffer = (label: string, bytes: number): GPUBuffer =>
            device.createBuffer({ label: `${TECHNIQUE_ID} ${label}`, size: Math.max(bytes, 16), usage: BUFFER_USAGE.storage | BUFFER_USAGE.copySrc | BUFFER_USAGE.copyDst });
        const shSize: [number, number] = [layout.probeCount[0] * SH_BLOCK_SIZE, layout.probeCount[1] * SH_BLOCK_SIZE];
        const sampledStorage = TEXTURE_USAGE.textureBinding | TEXTURE_USAGE.storageBinding | TEXTURE_USAGE.copySrc;
        const resources: HistoryResources = {
            layout,
            probeTileCount,
            previousProbeAtlas: texture('previous probe atlas', layout.atlasSize, 'rgba16float', sampledStorage | TEXTURE_USAGE.copyDst),
            previousProbeMask: texture('previous probe mask', layout.probeCount, 'r32uint', sampledStorage | TEXTURE_USAGE.copyDst),
            previousProbeSH: texture('previous probe SH', shSize, 'rg32uint', TEXTURE_USAGE.textureBinding | TEXTURE_USAGE.copySrc | TEXTURE_USAGE.copyDst),
            cachedTileAtlas: texture('cached tile atlas', layout.atlasSize, 'rgba16float', sampledStorage),
            cachedTileIndex: buffer('cached tile index', probeTileCount * 16),
            cachedTileLRU: buffer('cached tile LRU', probeTileCount * WORD_BYTES),
            nextCachedTileLRU: buffer('next cached tile LRU', probeTileCount * WORD_BYTES),
            cachedTileMRU: buffer('cached tile MRU', probeTileCount * WORD_BYTES),
            cachedTileList: buffer('cached tile list', probeTileCount * WORD_BYTES),
            cachedTileListElements: buffer('cached tile list elements', probeTileCount * 16),
            emptyTiles: buffer('empty tiles', probeTileCount * WORD_BYTES),
            overrideTiles: buffer('override tiles', layout.maxProbeSpawnCount * WORD_BYTES),
            spawnProbe: buffer('spawn probe', layout.maxProbeSpawnCount * PROBE_SPAWN_PROBE_BYTES),
            evictedRadiance: buffer('evicted probe radiance', layout.maxRayCount * 8),
            expectedRays: buffer('expected rays per cell', layout.maxRayCount * 4),
            cellCounts: buffer('cell counts', atlasTexels * WORD_BYTES),
            previousCellCounts: buffer('previous cell counts', atlasTexels * WORD_BYTES),
            cachedCellCounts: buffer('cached cell counts', atlasTexels * WORD_BYTES),
            scanBuffers: {
                cachedTileListCount: buffer('cached tile list count', probeTileCount * WORD_BYTES),
                cachedTileListIndex: buffer('cached tile list index', probeTileCount * WORD_BYTES),
                // One word more than the tiles: the scan's last word counts the unused entries.
                cachedTileLRUFlags: buffer('cached tile LRU flags', (probeTileCount + 1) * WORD_BYTES),
                cachedTileLRUIndex: buffer('cached tile LRU index', (probeTileCount + 1) * WORD_BYTES),
            },
            bytes: 0,
        };
        const { scanBuffers } = resources;
        this.listScan.rebind({ input: scanBuffers.cachedTileListCount, output: scanBuffers.cachedTileListIndex, capacity: probeTileCount });
        this.lruScan.rebind({ input: scanBuffers.cachedTileLRUFlags, output: scanBuffers.cachedTileLRUIndex, capacity: probeTileCount + 1 });
        const textureBytes = layout.atlasSize[0] * layout.atlasSize[1] * 8 * 2 + probeTileCount * WORD_BYTES + shSize[0] * shSize[1] * 8;
        const bufferBytes = [
            resources.cachedTileIndex,
            resources.cachedTileLRU,
            resources.nextCachedTileLRU,
            resources.cachedTileMRU,
            resources.cachedTileList,
            resources.cachedTileListElements,
            resources.emptyTiles,
            resources.overrideTiles,
            resources.spawnProbe,
            resources.evictedRadiance,
            resources.expectedRays,
            resources.cellCounts,
            resources.previousCellCounts,
            resources.cachedCellCounts,
            ...Object.values(scanBuffers),
        ].reduce((sum, item) => sum + item.size, 0);
        this.sized = { ...resources, bytes: textureBytes + bufferBytes };
        this.sizedValid = false;
        return this.sized;
    }

    /** The previous frame's depth and normals for this screen (the platform's copy). */
    screenTexturesFor(layout: ScreenProbeLayout): ScreenHistoryTextures {
        return this.screenHistory.texturesFor(layout.width, layout.height);
    }

    /**
     * The frame's first history passes, before the spawn: last frame's probes
     * taken aside (or, with no history, an empty mask and cache), then
     * `ReprojectScreenProbes` and the cache lookup (`CountScreenProbes`, the
     * scan, `ScatterScreenProbes`). The counters must be cleared already.
     */
    encodeReprojection(encoder: GPUCommandEncoder, inputs: HistoryFrameInputs): void {
        const resources = this.sized;
        if (!resources) throw new Error('@pmndrs/gi: the radiance cache history has no resources for this screen');
        const { layout } = resources;
        const screenTextures = this.screenTexturesFor(layout);
        const hasHistory = this.sizedValid && screenTextures.valid && !this.resetPending;
        this.resetPending = false;
        const groups = this.groupsFor(resources, inputs, screenTextures);
        const timer = this.context.timer;
        const maxPerDimension = this.context.device.limits.maxComputeWorkgroupsPerDimension;
        const tiles = resources.probeTileCount;

        // Last frame's probes, aside: Capsaicin swaps its two buffers (probe_buffer_index_), we copy (D44).
        if (hasHistory) {
            encoder.copyTextureToTexture({ texture: inputs.probeAtlas }, { texture: resources.previousProbeAtlas }, [layout.atlasSize[0], layout.atlasSize[1]]);
            encoder.copyTextureToTexture({ texture: inputs.probeMask, mipLevel: 0 }, { texture: resources.previousProbeMask }, [layout.probeCount[0], layout.probeCount[1]]);
            encoder.copyTextureToTexture({ texture: inputs.probeSH }, { texture: resources.previousProbeSH }, [layout.probeCount[0] * SH_BLOCK_SIZE, layout.probeCount[1] * SH_BLOCK_SIZE]);
            encoder.copyBufferToBuffer(resources.cellCounts, 0, resources.previousCellCounts, 0, resources.cellCounts.size);
        }
        // gfxCommandClearBuffer of the per-tile list counts (gi1.cpp:2283).
        encoder.clearBuffer(resources.scanBuffers.cachedTileListCount, 0, tiles * WORD_BYTES);

        const pass = encoder.beginComputePass({ label: `${TECHNIQUE_ID}/reproject`, timestampWrites: timer.computePass(`${TECHNIQUE_ID}/reproject`) });
        if (!hasHistory) {
            // No previous probe anywhere (ClearProbeMask, gi1.comp:242-246, on last frame's mask), and an
            // empty cache (InitCachedTileLRU, gi1.cpp:337-344).
            this.kernels.clearProbeMask.bind(pass);
            pass.setBindGroup(1, groups.clearPreviousMask);
            pass.dispatchWorkgroups(Math.ceil(layout.probeCount[0] / WORKGROUP_SIZE.clearProbeMask), Math.ceil(layout.probeCount[1] / WORKGROUP_SIZE.clearProbeMask));
            this.kernels.initCachedTileLRU.bind(pass);
            pass.setBindGroup(1, groups.initCachedTileLRU);
            dispatchSplit(pass, Math.ceil(tiles / TEMPORAL_WORKGROUP.list), maxPerDimension);
        }
        // ReprojectScreenProbes, one probe tile per workgroup (gi1.cpp:2256-2272).
        this.kernels.reproject.bind(pass);
        pass.setBindGroup(1, groups.reproject);
        pass.dispatchWorkgroups(layout.probeCount[0], layout.probeCount[1]);
        // LookupScreenProbes (gi1.cpp:2274-2292).
        this.kernels.count.bind(pass);
        pass.setBindGroup(1, groups.count);
        dispatchSplit(pass, Math.ceil(tiles / TEMPORAL_WORKGROUP.list), maxPerDimension);
        this.listScan.encode(pass, tiles);
        this.scatterArgs.encode(pass);
        this.kernels.scatter.bind(pass);
        pass.setBindGroup(1, groups.scatter);
        pass.dispatchWorkgroupsIndirect(this.scatterArgs.buffer, 0);
        pass.end();
    }

    /** `PatchScreenProbes` (gi1.cpp:2311-2320), after the compaction, on the probe list. */
    encodePatch(pass: GPUComputePassEncoder): void {
        const groups = this.groups?.value;
        if (!groups) throw new Error('@pmndrs/gi: encodeReprojection runs before the patch');
        this.patchArgs.encode(pass);
        this.kernels.patch.bind(pass);
        pass.setBindGroup(1, groups.patch);
        pass.dispatchWorkgroupsIndirect(this.patchArgs.buffer, 0);
    }

    /**
     * After the blend: `ReorderScreenProbes` (gi1.cpp:2544-2557) with the scan
     * of the LRU flags, the new order copied back (D44); and the screen's depth
     * and normals copied for the next frame. The history is valid from here.
     */
    encodeReorder(encoder: GPUCommandEncoder, screen: FrameScreen): void {
        const resources = this.sized;
        const groups = this.groups?.value;
        if (!resources || !groups) throw new Error('@pmndrs/gi: encodeReprojection runs before the reorder');
        const tiles = resources.probeTileCount;
        const timer = this.context.timer;
        const pass = encoder.beginComputePass({ label: `${TECHNIQUE_ID}/reorder`, timestampWrites: timer.computePass(`${TECHNIQUE_ID}/reorder`) });
        // gfxCommandScanSum over the flags: max_probe_count + 1 entries, so the last word counts the unused ones.
        this.lruScan.encode(pass, tiles + 1);
        this.kernels.reorder.bind(pass);
        pass.setBindGroup(1, groups.reorder);
        dispatchSplit(pass, Math.ceil(tiles / TEMPORAL_WORKGROUP.list), this.context.device.limits.maxComputeWorkgroupsPerDimension);
        pass.end();
        encoder.copyBufferToBuffer(resources.nextCachedTileLRU, 0, resources.cachedTileLRU, 0, tiles * WORD_BYTES);
        this.screenHistory.encode(encoder, screen);
        this.sizedValid = true;
    }

    dispose(): void {
        this.releaseSized();
        for (const buffer of this.placeholders) buffer.destroy();
        this.listScan.dispose();
        this.lruScan.dispose();
        this.scatterArgs.dispose();
        this.patchArgs.dispose();
        this.screenHistory.dispose();
    }

    private groupsFor(resources: HistoryResources, inputs: HistoryFrameInputs, screenTextures: ScreenHistoryTextures): FrameGroups {
        const screen = inputs.screen;
        const velocity = screen.velocity;
        if (!velocity) throw new Error('@pmndrs/gi: the radiance cache reprojects its probes by FrameScreen.velocity, which this screen lacks');
        const key = [resources, inputs.frameUniforms, inputs.counters, screen.depth, screen.normal, velocity, inputs.probeAtlas, inputs.probeMask, inputs.probeSH, inputs.probeSpawn, screenTextures.depth, screenTextures.normal];
        const current = this.groups;
        if (current && current.key.length === key.length && current.key.every((value, index) => value === key[index])) return current.value;
        const device = this.context.device;
        const { kernels } = this;
        const { scanBuffers } = resources;
        const group = (label: string, kernel: ComputeKernel, resourceList: GPUBindingResource[]): GPUBindGroup =>
            device.createBindGroup({ label: `${TECHNIQUE_ID}/${label}`, layout: kernel.ownLayouts[0], entries: resourceList.map((resource, binding) => ({ binding, resource })) });
        const uniforms = { buffer: inputs.frameUniforms };
        const counters = { buffer: inputs.counters };
        const level0 = (texture: GPUTexture): GPUTextureView => texture.createView({ baseMipLevel: 0, mipLevelCount: 1 });
        const value: FrameGroups = {
            clearPreviousMask: group('ClearProbeMask (previous)', kernels.clearProbeMask, [uniforms, level0(resources.previousProbeMask)]),
            initCachedTileLRU: group(TEMPORAL_ENTRY_POINT.initCachedTileLRU, kernels.initCachedTileLRU, [uniforms, { buffer: resources.cachedTileLRU }]),
            reproject: group(TEMPORAL_ENTRY_POINT.reproject, kernels.reproject, [
                uniforms,
                screenTextureView(screen.depth),
                screenTextureView(screen.normal),
                screenTextureView(velocity),
                screenTextures.depth.createView(),
                screenTextures.normal.createView(),
                resources.previousProbeMask.createView(),
                resources.previousProbeAtlas.createView(),
                inputs.probeAtlas.createView(),
                level0(inputs.probeMask),
                resources.previousProbeSH.createView(),
                inputs.probeSH.createView(),
                { buffer: resources.emptyTiles },
                counters,
                { buffer: resources.previousCellCounts },
                { buffer: resources.cellCounts },
            ]),
            count: group(TEMPORAL_ENTRY_POINT.count, kernels.count, [
                uniforms,
                { buffer: resources.cachedTileLRU },
                { buffer: resources.cachedTileIndex },
                { buffer: scanBuffers.cachedTileListCount },
                { buffer: resources.cachedTileListElements },
                { buffer: scanBuffers.cachedTileLRUFlags },
                counters,
            ]),
            scatter: group(TEMPORAL_ENTRY_POINT.scatter, kernels.scatter, [
                { buffer: this.scatterArgs.buffer },
                { buffer: resources.cachedTileListElements },
                { buffer: scanBuffers.cachedTileListIndex },
                { buffer: resources.cachedTileList },
            ]),
            patch: group(TEMPORAL_ENTRY_POINT.patch, kernels.patch, [
                uniforms,
                screenTextureView(screen.depth),
                { buffer: this.patchArgs.buffer },
                counters,
                { buffer: resources.emptyTiles },
                { buffer: resources.overrideTiles },
                { buffer: inputs.probeSpawn },
            ]),
            reorder: group(TEMPORAL_ENTRY_POINT.reorder, kernels.reorder, [
                uniforms,
                { buffer: resources.cachedTileLRU },
                { buffer: scanBuffers.cachedTileLRUFlags },
                { buffer: scanBuffers.cachedTileLRUIndex },
                { buffer: resources.cachedTileMRU },
                { buffer: resources.nextCachedTileLRU },
            ]),
        };
        this.groups = { key, value };
        return value;
    }

    private releaseSized(): void {
        const sized = this.sized;
        if (!sized) return;
        for (const texture of [sized.previousProbeAtlas, sized.previousProbeMask, sized.previousProbeSH, sized.cachedTileAtlas]) texture.destroy();
        for (const buffer of [
            sized.cachedTileIndex,
            sized.cachedTileLRU,
            sized.nextCachedTileLRU,
            sized.cachedTileMRU,
            sized.cachedTileList,
            sized.cachedTileListElements,
            sized.emptyTiles,
            sized.overrideTiles,
            sized.spawnProbe,
            sized.evictedRadiance,
            sized.expectedRays,
            sized.cellCounts,
            sized.previousCellCounts,
            sized.cachedCellCounts,
            ...Object.values(sized.scanBuffers),
        ]) {
            buffer.destroy();
        }
        this.sized = null;
        this.sizedValid = false;
        this.groups = null;
    }
}

interface FrameGroups {
    readonly clearPreviousMask: GPUBindGroup;
    readonly initCachedTileLRU: GPUBindGroup;
    readonly reproject: GPUBindGroup;
    readonly count: GPUBindGroup;
    readonly scatter: GPUBindGroup;
    readonly patch: GPUBindGroup;
    readonly reorder: GPUBindGroup;
}

/**
 * `workgroups` workgroups of a 1D kernel as `[x, y]`, so neither passes the
 * device's limit; the kernels count workgroups in rows (`rcLinearIndex`).
 */
function dispatchSplit(pass: GPUComputePassEncoder, workgroups: number, maxPerDimension: number): void {
    const x = Math.max(1, Math.min(workgroups, maxPerDimension));
    pass.dispatchWorkgroups(x, Math.ceil(workgroups / x));
}

/** A depth texture is read through its depth aspect (as `unfilterable-float`); anything else whole. */
function screenTextureView(texture: GPUTexture): GPUTextureView {
    const isDepth = texture.format.startsWith('depth');
    return texture.createView({ aspect: isDepth ? 'depth-only' : 'all', baseMipLevel: 0, mipLevelCount: 1 });
}

/** Where this frame's scene changes are, for the running means (D46): boxes, or everything. */
export interface HistoryChanges {
    readonly everything: boolean;
    readonly boxes: readonly Aabb[];
}

export const NO_HISTORY_CHANGES: HistoryChanges = Object.freeze({ everything: false, boxes: [] });

/**
 * Adds a frame's `SceneChanges` to those not yet given to a screen stage:
 * every dirty region's box (lights with their reach, geometry, materials);
 * a change with no bounds (`global`: the sun, the environment, too many
 * lights) or more boxes than the blend takes counts as everything.
 */
export function mergeHistoryChanges(pending: HistoryChanges, changes: SceneChanges): HistoryChanges {
    if (pending.everything) return pending;
    if (changes.global.length > 0) return { everything: true, boxes: [] };
    if (changes.regions.length === 0) return pending;
    const boxes = [...pending.boxes, ...changes.regions.map((region) => region.bounds)];
    if (boxes.length > MAX_HISTORY_CHANGE_BOXES) return { everything: true, boxes: [] };
    return { everything: false, boxes };
}

/** `RcHistoryChanges` (wgsl/screen-probes.ts) for `changes`. */
export function historyChangesWords(changes: HistoryChanges): ArrayBuffer {
    const words = new ArrayBuffer(HISTORY_CHANGES_BYTES);
    const uints = new Uint32Array(words);
    const floats = new Float32Array(words);
    uints[0] = changes.everything ? 0 : changes.boxes.length;
    uints[1] = changes.everything ? 1 : 0;
    changes.boxes.forEach((box, index) => {
        floats.set([box.min[0], box.min[1], box.min[2], 0], 4 + index * 8);
        floats.set([box.max[0], box.max[1], box.max[2], 0], 8 + index * 8);
    });
    return words;
}
