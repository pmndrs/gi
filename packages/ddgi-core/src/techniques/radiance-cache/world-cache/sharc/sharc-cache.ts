/**
 * A SHaRC-style spatial hash cache behind the radiance cache's world-cache
 * slot (`worldCache: 'sharc'`; phase 6 experiment P6.1, register E1-E7).
 * Capsaicin's hash grid (`HashGridCache`) stays the default.
 *
 * Ported from the pmndrs/pathtracer project's hash-grid radiance cache
 * (`bench/nrc/hash-grid.ts` at dd771c8, MIT; design in its
 * `docs/design/nrc-hash-cache.md`), itself written from the published
 * descriptions of SHaRC and of Binder et al. 2019 (no NVIDIA SDK code; see
 * `sharc-wgsl.ts` and THIRD_PARTY.md).
 *
 * Its per-frame work (`encodeFrame`), three passes:
 *
 * 1. `SharcUpdate`, per probe ray: the hit's emission into the ray, a short
 *    update path from the hit (its light sample, `bounceRays` bounces, the
 *    cache as the tail), each vertex's estimate into its cell;
 * 2. `SharcResolve`, per slot: the running means, ages and evictions;
 * 3. `SharcLookup`, per probe ray: the cell's radiance (or the ray's own
 *    estimate when it is shorter than its cell) into the ray's radiance slot.
 *
 * The per-ray passes are sized on the GPU from the probe-ray count, through
 * the platform's indirect-dispatch arguments.
 */

import type { GpuIndirectArgs, GpuWord } from '../../../../platform/contract/compute.ts';
import { GILimitError } from '../../../../platform/contract/errors.ts';
import type { ComputeKernel } from '../../../../platform/contract/wgsl.ts';
import { verticalFieldOfView } from '../hash-grid-cpu.ts';
import type { WorldCache, WorldCacheContext, WorldCacheFrame, WorldCacheLookup, WorldCacheProbeRays } from '../world-cache.ts';
import { sharcCellSizePerMetre } from './sharc-cpu.ts';
import { resolveSharcSettings, type SharcSettings } from './sharc-settings.ts';
import {
    SHARC_CACHE_BINDING,
    SHARC_CACHE_GROUP,
    SHARC_CACHE_GROUP_ENTRIES,
    SHARC_COUNTER,
    SHARC_COUNTER_WORDS,
    SHARC_ENTRY_POINT,
    SHARC_LOOKUP_GROUP_ENTRIES,
    SHARC_PARAMS_BYTES,
    SHARC_PARAMS_WORD,
    SHARC_RAY_BINDING,
    SHARC_RAY_GROUP,
    SHARC_RAY_GROUP_ENTRIES,
    SHARC_RAY_SAMPLE_BYTES,
    SHARC_RESOLVE_WORKGROUP_SIZE,
    SHARC_SLOT_WORDS,
    SHARC_WORKGROUP_SIZE,
    sharcLookupFunctionWgsl,
    sharcLookupWgsl,
    sharcResolveWgsl,
    sharcUpdateWgsl,
    type SharcCompileOptions,
} from './sharc-wgsl.ts';

/** `GPUBufferUsage` bits (the global doesn't exist in Node, where the tests import this file). */
const BUFFER_USAGE = { copySrc: 0x4, copyDst: 0x8, uniform: 0x40, storage: 0x80 } as const;
const WORD_BYTES = 4;
/** Timed as `radiance-cache/<pass>`, beside the hash grid's sections. */
const TIMER_PREFIX = 'radiance-cache';

/** What the cache reads from its owner's settings, live: the technique passes its own settings object. */
export interface SharcCacheOptions {
    /** Overrides of the defaults (`DEFAULT_SHARC_SETTINGS`). */
    readonly sharc?: Partial<SharcSettings>;
    /** `RadianceCacheSettings.useDirectLighting`: emissive hits add their emission to the probe ray. */
    readonly useDirectLighting: boolean;
}

/** The cache's buffers, for tests and debug readbacks. */
export interface SharcCacheBuffers {
    /** `SharcParams`, written every frame. */
    readonly params: GPUBuffer;
    /** Per slot: the cell's checksum (0 = empty). */
    readonly keys: GPUBuffer;
    /** The counters (`SHARC_COUNTER`, `SHARC_COUNTER_WORDS` words), then `SHARC_SLOT_WORDS` words per slot. */
    readonly accum: GPUBuffer;
    /** Per slot: the resolved radiance and its sample weight (vec4f). */
    readonly resolved: GPUBuffer;
}

interface Prepared {
    readonly device: GPUDevice;
    readonly settings: SharcSettings;
    readonly compile: SharcCompileOptions;
    readonly entries: number;
    readonly buffers: SharcCacheBuffers;
    readonly update: ComputeKernel;
    readonly resolve: ComputeKernel;
    readonly lookup: ComputeKernel;
    readonly updateCacheGroup: GPUBindGroup;
    readonly resolveCacheGroup: GPUBindGroup;
    readonly lookupCacheGroup: GPUBindGroup;
    readonly rayArgs: GpuIndirectArgs;
    readonly timer: WorldCacheContext['timer'];
    readonly memoryBytes: number;
}

interface RayData {
    readonly rays: WorldCacheProbeRays;
    /** Two vec4 words per probe ray: its own estimate and where it went (`SHARC_RAY_SAMPLE_BYTES`). */
    readonly raySamples: GPUBuffer;
    readonly updateGroup: GPUBindGroup;
    readonly lookupGroup: GPUBindGroup;
}

export class SharcCache implements WorldCache {
    readonly id = 'sharc';
    /** Read live: `sharc` every frame (its table size, fallback levels and pyramid at `prepare`). */
    readonly options: SharcCacheOptions;

    private prepared: Prepared | null = null;
    private rayData: RayData | null = null;
    private readonly paramWords = new ArrayBuffer(SHARC_PARAMS_BYTES);

    constructor(options: SharcCacheOptions = { useDirectLighting: true }) {
        this.options = options;
    }

    /** The settings `prepare` built the cache with (live fields may have changed since). */
    get settings(): SharcSettings {
        return this.require().settings;
    }

    /** Slots in the table. */
    get entries(): number {
        return this.require().entries;
    }

    get buffers(): SharcCacheBuffers {
        return this.require().buffers;
    }

    /** The latest screen's per-ray words (two `vec4u` a ray), or null before the first `encodeFrame`. */
    get raySamples(): GPUBuffer | null {
        return this.rayData?.raySamples ?? null;
    }

    /** One of the counter words (`SHARC_COUNTER`), cleared at the start of each frame (but the ray count). */
    counter(name: keyof typeof SHARC_COUNTER): GpuWord {
        return { buffer: this.require().buffers.accum, offset: SHARC_COUNTER[name] * WORD_BYTES };
    }

    get memoryBytes(): number {
        return (this.prepared?.memoryBytes ?? 0) + (this.rayData?.raySamples.size ?? 0);
    }

    get stats(): Readonly<Record<string, number>> {
        const prepared = this.prepared;
        if (!prepared) return {};
        return {
            slots: prepared.entries,
            'per-ray samples (bytes)': this.rayData?.raySamples.size ?? 0,
            'GPU memory (bytes)': this.memoryBytes,
        };
    }

    readonly lookup: WorldCacheLookup = {
        code: (group: number) => sharcLookupFunctionWgsl(group, this.require().compile),
        entries: SHARC_LOOKUP_GROUP_ENTRIES,
        createBindGroup: (layout: GPUBindGroupLayout) => {
            const { device, buffers } = this.require();
            return device.createBindGroup({
                label: 'radiance-cache/sharc lookup',
                layout,
                entries: [
                    { binding: SHARC_CACHE_BINDING.params, resource: { buffer: buffers.params } },
                    { binding: SHARC_CACHE_BINDING.keys, resource: { buffer: buffers.keys } },
                    { binding: SHARC_CACHE_BINDING.resolved, resource: { buffer: buffers.resolved } },
                ],
            });
        },
    };

    async prepare(context: WorldCacheContext): Promise<void> {
        const { device, kernels, compute, shade } = context;
        const settings = resolveSharcSettings(this.options.sharc);
        const entries = 2 ** settings.log2Entries;
        const bytes = {
            keys: entries * WORD_BYTES,
            accum: (SHARC_COUNTER_WORDS + SHARC_SLOT_WORDS * entries) * WORD_BYTES,
            resolved: entries * 16,
        };
        // Name the limit before anything is made (register D14's rule).
        const largest = Math.max(bytes.keys, bytes.accum, bytes.resolved);
        if (largest > device.limits.maxStorageBufferBindingSize) throw new GILimitError('maxStorageBufferBindingSize', largest, device.limits.maxStorageBufferBindingSize);
        if (largest > device.limits.maxBufferSize) throw new GILimitError('maxBufferSize', largest, device.limits.maxBufferSize);
        const compile: SharcCompileOptions = { fallbackLevels: settings.fallbackLevels, pyramid: settings.pyramid };
        this.dispose();

        const storageUsage = BUFFER_USAGE.storage | BUFFER_USAGE.copySrc | BUFFER_USAGE.copyDst;
        device.pushErrorScope('out-of-memory');
        // Made zeroed (WebGPU clears new buffers): every key 0, the empty table.
        const buffers: SharcCacheBuffers = {
            params: device.createBuffer({ label: 'radiance-cache/sharc params', size: SHARC_PARAMS_BYTES, usage: BUFFER_USAGE.uniform | BUFFER_USAGE.copyDst }),
            keys: device.createBuffer({ label: 'radiance-cache/sharc keys', size: bytes.keys, usage: storageUsage }),
            accum: device.createBuffer({ label: 'radiance-cache/sharc accumulators', size: bytes.accum, usage: storageUsage }),
            resolved: device.createBuffer({ label: 'radiance-cache/sharc resolved', size: bytes.resolved, usage: storageUsage }),
        };
        const outOfMemory = await device.popErrorScope();
        if (outOfMemory) {
            for (const buffer of Object.values(buffers)) buffer.destroy();
            throw new Error(`@pmndrs/gi: the GPU ran out of memory making the SHaRC cache's ${bytes.keys + bytes.accum + bytes.resolved} bytes: ${outOfMemory.message}`);
        }

        const cacheGroup = { label: 'radiance-cache/sharc group', entries: SHARC_CACHE_GROUP_ENTRIES };
        const rayGroup = { label: 'radiance-cache/sharc rays', entries: SHARC_RAY_GROUP_ENTRIES };
        const [update, resolve, lookup, rayArgs] = await Promise.all([
            kernels.computeKernel({ label: 'radiance-cache/SharcUpdate', modules: [shade.module], code: sharcUpdateWgsl(compile), entryPoint: SHARC_ENTRY_POINT.update, ownGroups: [cacheGroup, rayGroup] }),
            kernels.computeKernel({ label: 'radiance-cache/SharcResolve', modules: [], code: sharcResolveWgsl(compile), entryPoint: SHARC_ENTRY_POINT.resolve, ownGroups: [cacheGroup] }),
            kernels.computeKernel({ label: 'radiance-cache/SharcLookup', modules: [], code: sharcLookupWgsl(compile), entryPoint: SHARC_ENTRY_POINT.lookup, ownGroups: [cacheGroup, rayGroup] }),
            compute.createIndirectArgs({
                label: 'radiance-cache/sharc per-ray args',
                count: { buffer: buffers.accum, offset: SHARC_COUNTER.rayCount * WORD_BYTES },
                itemsPerWorkgroup: SHARC_WORKGROUP_SIZE,
            }),
        ]);
        const cacheEntries: GPUBindGroupEntry[] = [
            { binding: SHARC_CACHE_BINDING.params, resource: { buffer: buffers.params } },
            { binding: SHARC_CACHE_BINDING.keys, resource: { buffer: buffers.keys } },
            { binding: SHARC_CACHE_BINDING.accum, resource: { buffer: buffers.accum } },
            { binding: SHARC_CACHE_BINDING.resolved, resource: { buffer: buffers.resolved } },
        ];
        const group = (kernel: ComputeKernel): GPUBindGroup => device.createBindGroup({ label: `${kernel.pipeline.label} cache`, layout: kernel.ownLayouts[0], entries: cacheEntries });
        this.prepared = {
            device,
            settings,
            compile,
            entries,
            buffers,
            update,
            resolve,
            lookup,
            updateCacheGroup: group(update),
            resolveCacheGroup: group(resolve),
            lookupCacheGroup: group(lookup),
            rayArgs,
            timer: context.timer,
            memoryBytes: SHARC_PARAMS_BYTES + bytes.keys + bytes.accum + bytes.resolved + rayArgs.memoryBytes,
        };
    }

    /** Empties the table: keys, accumulators, counters and resolved values back to zero, as `prepare` leaves them. */
    reset(encoder: GPUCommandEncoder): void {
        const { buffers } = this.require();
        encoder.clearBuffer(buffers.keys);
        encoder.clearBuffer(buffers.accum);
        encoder.clearBuffer(buffers.resolved);
    }

    encodeFrame(encoder: GPUCommandEncoder, frame: WorldCacheFrame, rays: WorldCacheProbeRays): void {
        const prepared = this.require();
        const rayData = this.rayDataFor(prepared, rays);
        this.writeParams(prepared, frame);
        const { buffers } = prepared;

        // This frame's counters from zero; the ray count where the indirect arguments read it.
        encoder.clearBuffer(buffers.accum, WORD_BYTES, (SHARC_COUNTER_WORDS - 1) * WORD_BYTES);
        encoder.copyBufferToBuffer(rays.count.buffer, rays.count.offset, buffers.accum, SHARC_COUNTER.rayCount * WORD_BYTES, WORD_BYTES);

        const update = this.beginPass(encoder, prepared, 'SharcUpdate');
        prepared.rayArgs.encode(update);
        prepared.update.bind(update);
        update.setBindGroup(SHARC_CACHE_GROUP, prepared.updateCacheGroup);
        update.setBindGroup(SHARC_RAY_GROUP, rayData.updateGroup);
        update.dispatchWorkgroupsIndirect(prepared.rayArgs.buffer, 0);
        update.end();

        const resolve = this.beginPass(encoder, prepared, 'SharcResolve');
        prepared.resolve.bind(resolve);
        resolve.setBindGroup(SHARC_CACHE_GROUP, prepared.resolveCacheGroup);
        const groups = prepared.entries / SHARC_RESOLVE_WORKGROUP_SIZE;
        const maxPerDimension = prepared.device.limits.maxComputeWorkgroupsPerDimension;
        // Split in rows past the per-dimension limit (2^24 slots and more); the kernel counts in rows.
        const width = Math.min(groups, maxPerDimension);
        resolve.dispatchWorkgroups(width, Math.ceil(groups / width));
        resolve.end();

        const lookup = this.beginPass(encoder, prepared, 'SharcLookup');
        prepared.lookup.bind(lookup);
        lookup.setBindGroup(SHARC_CACHE_GROUP, prepared.lookupCacheGroup);
        lookup.setBindGroup(SHARC_RAY_GROUP, rayData.lookupGroup);
        lookup.dispatchWorkgroupsIndirect(prepared.rayArgs.buffer, 0);
        lookup.end();
    }

    dispose(): void {
        this.releaseRayData();
        const prepared = this.prepared;
        if (!prepared) return;
        for (const buffer of Object.values(prepared.buffers)) buffer.destroy();
        prepared.rayArgs.dispose();
        this.prepared = null;
    }

    private beginPass(encoder: GPUCommandEncoder, prepared: Prepared, section: string): GPUComputePassEncoder {
        const name = `${TIMER_PREFIX}/${section}`;
        return encoder.beginComputePass({ label: name, timestampWrites: prepared.timer.computePass(name) });
    }

    /** The per-ray buffer and the ray groups for these probe rays, made again when the screen's buffers change. */
    private rayDataFor(prepared: Prepared, rays: WorldCacheProbeRays): RayData {
        const current = this.rayData;
        if (current && sameRays(current.rays, rays)) return current;
        this.releaseRayData();
        const { device } = prepared;
        const bytes = Math.max(rays.capacity, 1) * SHARC_RAY_SAMPLE_BYTES;
        if (bytes > device.limits.maxStorageBufferBindingSize) throw new GILimitError('maxStorageBufferBindingSize', bytes, device.limits.maxStorageBufferBindingSize);
        const raySamples = device.createBuffer({ label: 'radiance-cache/sharc per-ray samples', size: bytes, usage: BUFFER_USAGE.storage | BUFFER_USAGE.copySrc });
        const entries: GPUBindGroupEntry[] = [
            { binding: SHARC_RAY_BINDING.probeRays, resource: { buffer: rays.buffer } },
            { binding: SHARC_RAY_BINDING.radiance, resource: rays.radiance },
            { binding: SHARC_RAY_BINDING.raySamples, resource: { buffer: raySamples } },
        ];
        const group = (kernel: ComputeKernel): GPUBindGroup => device.createBindGroup({ label: `${kernel.pipeline.label} rays`, layout: kernel.ownLayouts[1], entries });
        const rayData: RayData = { rays, raySamples, updateGroup: group(prepared.update), lookupGroup: group(prepared.lookup) };
        this.rayData = rayData;
        return rayData;
    }

    private releaseRayData(): void {
        this.rayData?.raySamples.destroy();
        this.rayData = null;
    }

    /** `SharcParams` for this frame, from the settings as they are now (the compiled ones aside). */
    private writeParams(prepared: Prepared, frame: WorldCacheFrame): void {
        const live = { ...resolveSharcSettings(this.options.sharc), log2Entries: prepared.settings.log2Entries };
        const floats = new Float32Array(this.paramWords);
        const words = new Uint32Array(this.paramWords);
        const word = SHARC_PARAMS_WORD;
        floats.set(frame.camera.position, word.camera);
        floats[word.cellSizePerMetre] = sharcCellSizePerMetre(verticalFieldOfView(frame.camera.projection), frame.height, live.cellPixels);
        floats[word.minCell] = live.minCell;
        floats[word.decay] = 1 - 1 / Math.max(1, live.historyFrames);
        floats[word.minSamples] = live.minSamples;
        words[word.frame] = frame.frame >>> 0;
        words[word.mask] = prepared.entries - 1;
        words[word.staleFrames] = live.staleFrames;
        words[word.normalBins] = live.normalBins ? 1 : 0;
        words[word.bounceRays] = live.bounceRays;
        words[word.cacheTail] = live.cacheTail ? 1 : 0;
        words[word.useDirectLighting] = this.options.useDirectLighting ? 1 : 0;
        words[word.skyOnMiss] = live.skyOnMiss ? 1 : 0;
        floats[word.updateFraction] = live.updateFraction;
        prepared.device.queue.writeBuffer(prepared.buffers.params, 0, this.paramWords);
    }

    private require(): Prepared {
        if (!this.prepared) throw new Error('@pmndrs/gi: the SHaRC world cache is used before prepare()');
        return this.prepared;
    }
}

/** The same probe-ray buffers as before (a new screen size or new buffers need a new per-ray buffer and groups). */
function sameRays(a: WorldCacheProbeRays, b: WorldCacheProbeRays): boolean {
    return (
        a.buffer === b.buffer &&
        a.capacity === b.capacity &&
        a.radiance.buffer === b.radiance.buffer &&
        (a.radiance.offset ?? 0) === (b.radiance.offset ?? 0) &&
        a.radiance.size === b.radiance.size
    );
}
