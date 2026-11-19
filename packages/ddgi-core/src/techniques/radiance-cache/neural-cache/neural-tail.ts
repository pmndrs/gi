/**
 * The neural tail (phase 6 experiment P6.4, register E11): pmndrs/pathtracer's
 * neural radiance cache network (`@pathtracer/mlp`, MIT, linked at a pinned
 * commit; THIRD_PARTY.md) as an enhancement of the GI-1.2 hash grid, not a
 * replacement (docs/design/radiance-cache.md, "Neural tail").
 *
 * The grid stays the near cache and keeps reacting as fast as it does: every
 * probe-ray hit is still lit, inserted and resolved by it. What the network
 * adds is the second bounce's tail. GI-1.2's multibounce gives a first-bounce
 * cell the light its second hit reflects from *direct* light only (two
 * bounces in all); with the tail, `UpdateMultibounceCells` adds the
 * network's answer for the light the second hit reflects from *indirect*
 * light. The network learns that quantity from the grid's own indirect values
 * (cell records) and the screen probes' irradiance (screen records), so the
 * cells' indirect values become a fixed point that carries every bounce in
 * expectation (the furnace: ρ (ρE/π) / (1 − ρ²) = 0.4244 where two bounces
 * read 0.3183).
 *
 * Per frame:
 *
 * 1. Inside the hash grid's frame (`encode`, the grid's `HashGridTail` slot,
 *    after `UpdateTiles`): the counts cleared; cell records and the queries
 *    gathered (`neural-tail-wgsl.ts`); inference over the second-bounce
 *    entries into the grid's `neural_tail_buffer` list, once the gate is open.
 * 2. After the screen's irradiance (`encodeTraining`, from the technique):
 *    screen records, then the training steps over the frame's set.
 *
 * Until the network has taken `warmupSteps` steps the gate is closed: the
 * grid runs as GI-1.2, and the network trains at `warmupStepsPerFrame`.
 */

import { DEFAULT_TRAIN, defaultForwardRecords, forwardWorkgroupBytes, inferTiledWorkgroupBytes, initWeights, layerOffset, MLP_SHAPES, MlpEngine, type MlpShape } from '@pathtracer/mlp';
import type { GpuIndirectArgs } from '../../../platform/contract/compute.ts';
import type { FrameCamera } from '../../../platform/contract/frame.ts';
import type { TechniqueContext } from '../../../platform/contract/technique.ts';
import type { ComputeKernel } from '../../../platform/contract/wgsl.ts';
import { cameraRelativeInverseViewProjection } from '../camera.ts';
import { HASH_GRID_COUNTER } from '../world-cache/capacity.ts';
import type { HashGridCache, HashGridTail, HashGridTailFrame } from '../world-cache/hash-grid-cache.ts';
import { RAY_DATA_GROUP_ENTRIES } from '../world-cache/hash-grid-population-wgsl.ts';
import { HASH_GRID_CACHE_GROUP_ENTRIES } from '../world-cache/hash-grid-wgsl.ts';
import { neuralTailBox, neuralTailGateOpen, neuralTailStepsThisFrame } from './neural-tail-cpu.ts';
import { neuralTailSettings, type NeuralTailSettings } from './neural-tail-settings.ts';
import {
    gatherCellRecordsWgsl,
    gatherQueriesWgsl,
    gatherScreenRecordsWgsl,
    NEURAL_TAIL_BINDING,
    NEURAL_TAIL_COUNT_WORD,
    NEURAL_TAIL_COUNTS_BYTES,
    NEURAL_TAIL_ENTRY_POINT,
    NEURAL_TAIL_GROUP_ENTRIES,
    NEURAL_TAIL_HASH_GRID_GROUP,
    NEURAL_TAIL_PARAMS_BYTES,
    NEURAL_TAIL_PARAMS_WORD,
    NEURAL_TAIL_QUERY_VEC4,
    NEURAL_TAIL_RECORD_VEC4,
    NEURAL_TAIL_SCREEN_GROUP,
    NEURAL_TAIL_SCREEN_GROUP_ENTRIES,
    NEURAL_TAIL_WORKGROUP,
    neuralTailInferHooks,
    neuralTailTrainHooks,
} from './neural-tail-wgsl.ts';

/** `GPUBufferUsage` bits (the global doesn't exist in Node, where the tests import this file). */
const BUFFER_USAGE = { copySrc: 0x4, copyDst: 0x8, uniform: 0x40, storage: 0x80 } as const;
const VEC4_BYTES = 16;
const WORD_BYTES = 4;
/** Timed as `radiance-cache/<section>`, beside the hash grid's sections. */
const TIMER_PREFIX = 'radiance-cache';
/** Inference records per call at most: more than any screen's second-bounce entries (the engine sizes nothing by it). */
const INFER_CAPACITY = 1 << 26;

/** The training set: room for the cell records and a full screen's records at the default stride, with slack. */
function setCapacity(settings: NeuralTailSettings): number {
    const screen = settings.screenStride > 0 ? Math.ceil(3840 / settings.screenStride) * Math.ceil(2160 / settings.screenStride) : 0;
    return Math.max(1024, Math.ceil((settings.cellRecords * 1.25 + Math.min(screen, 65536)) / 64) * 64);
}

/**
 * Records per workgroup of the tiled kernels: the pathtracer's default (64 at
 * f16, 48 at f32) or the largest multiple of 16 below it that fits the
 * device's workgroup memory. The platform's device keeps WebGPU's default
 * 16 KiB (`requiredLimits` asks for no more), where the compact network's
 * 64-record inference needs 20 KiB.
 */
function tiledRecords(shape: MlpShape, precision: 'f16' | 'f32', limit: number, bytes: (records: number) => number): number {
    for (let records = defaultForwardRecords(precision); records >= 16; records -= 16) {
        if (bytes(records) <= limit) return records;
    }
    throw new Error(`@pmndrs/gi: the neural tail's ${shape.hidden} × ${shape.width} network needs more workgroup memory than the device's ${limit} bytes`);
}

/** The screen the training reads: the irradiance the screen probes made this frame. */
export interface NeuralTailScreen {
    readonly irradiance: GPUTexture;
    readonly width: number;
    readonly height: number;
}

interface Prepared {
    readonly device: GPUDevice;
    readonly mlp: MlpEngine;
    readonly shape: MlpShape;
    readonly gatherCells: ComputeKernel;
    readonly gatherQueries: ComputeKernel;
    readonly gatherScreen: ComputeKernel;
    readonly cellArgs: GpuIndirectArgs;
    readonly queryArgs: GpuIndirectArgs;
    readonly params: GPUBuffer;
    readonly counts: GPUBuffer;
    readonly set: GPUBuffer;
    readonly setCapacity: number;
    readonly tileWords: GPUBuffer;
    readonly trainGroup: GPUBindGroup;
    readonly cellGroup: GPUBindGroup;
}

/** Bind groups over one screen's lists and queries. */
interface ScreenData {
    readonly listWords: GPUBuffer;
    readonly queries: GPUBuffer;
    readonly queryCapacity: number;
    readonly cellGroups: readonly [GPUBindGroup, GPUBindGroup];
    readonly queryGroups: readonly [GPUBindGroup, GPUBindGroup, GPUBindGroup];
    readonly inferGroup: GPUBindGroup;
}

export class NeuralTail implements HashGridTail {
    readonly settings: NeuralTailSettings;

    private prepared: Prepared | null = null;
    private screenData: ScreenData | null = null;
    private screenGroup: { readonly texture: GPUTexture; readonly group: GPUBindGroup } | null = null;
    private context: TechniqueContext | null = null;
    private trainedSteps = 0;
    /** The default box's centre: the camera where the network (re)started, taken at its first frame. */
    private boxCentre: readonly [number, number, number] | null = null;
    /** The hash grid ran the tail this frame (multibounce on), so training follows. */
    private gatheredThisFrame = false;
    private readonly paramWords = new ArrayBuffer(NEURAL_TAIL_PARAMS_BYTES);

    constructor(settings: Partial<NeuralTailSettings> = {}) {
        this.settings = neuralTailSettings(settings);
    }

    /** The gate is open: the network has trained `warmupSteps` steps, and the grid adds its answers. */
    get active(): boolean {
        return this.prepared !== null && neuralTailGateOpen(this.trainedSteps, this.settings.warmupSteps);
    }

    /** Training steps encoded since the network was (re)started. */
    get steps(): number {
        return this.trainedSteps;
    }

    /** The network, for tests and the timing page. */
    get network(): MlpEngine | null {
        return this.prepared?.mlp ?? null;
    }

    /** The training set's records and its count (word `NEURAL_TAIL_COUNT_WORD.records` of `counts`), for tests. */
    get buffers(): { readonly set: GPUBuffer; readonly counts: GPUBuffer; readonly setCapacity: number; readonly queries: GPUBuffer | null } | null {
        const prepared = this.prepared;
        return prepared ? { set: prepared.set, counts: prepared.counts, setCapacity: prepared.setCapacity, queries: this.screenData?.queries ?? null } : null;
    }

    get memoryBytes(): number {
        const prepared = this.prepared;
        if (!prepared) return 0;
        const own = prepared.params.size + prepared.counts.size + prepared.set.size + prepared.cellArgs.memoryBytes + prepared.queryArgs.memoryBytes;
        return own + prepared.mlp.bufferBytes + (this.screenData?.queries.size ?? 0);
    }

    get stats(): Readonly<Record<string, number>> {
        const prepared = this.prepared;
        if (!prepared) return {};
        return {
            'training steps': this.trainedSteps,
            'gate open': this.active ? 1 : 0,
            'network weights': prepared.mlp.weightCount,
            'GPU memory (bytes)': this.memoryBytes,
        };
    }

    /**
     * Compiles the network and the gather kernels and makes the training
     * set, after the grid's own `prepare` (its layout and counters). Starts
     * the network from fresh weights.
     */
    async prepare(context: TechniqueContext, grid: HashGridCache): Promise<void> {
        this.dispose();
        this.context = context;
        const { device, kernels, compute, shade } = context;
        const settings = this.settings;
        const shape = MLP_SHAPES[settings.network];
        const precision = settings.precision === 'auto' ? (device.features.has('shader-f16') ? 'f16' : 'f32') : settings.precision;
        const capacity = setCapacity(settings);
        const layout = grid.layout;
        const tileWords = grid.buffers.tileWords;
        const limit = device.limits.maxComputeWorkgroupStorageSize;
        const inferRecords = tiledRecords(shape, precision, limit, (records) => inferTiledWorkgroupBytes({ shape, precision }, { records }));
        const forwardRecords = tiledRecords(shape, precision, limit, (records) => forwardWorkgroupBytes({ shape, precision }, { records }));
        // The tiled weight gradient stages `chunk` records of deltas and activations (f32 vec4 rows each).
        const rowBytes = Math.max(shape.inputs, shape.width) * 4;
        const gradientChunk = 2 * 32 * rowBytes + 16 <= limit ? 32 : 16;
        const word = (counter: number) => ({ buffer: tileWords, offset: counter * WORD_BYTES });
        const ownGroups = [
            { label: 'radiance-cache/hash-grid group', entries: HASH_GRID_CACHE_GROUP_ENTRIES },
            { label: 'radiance-cache/hash-grid ray data', entries: RAY_DATA_GROUP_ENTRIES },
            { label: 'radiance-cache/neural tail', entries: NEURAL_TAIL_GROUP_ENTRIES },
        ];
        const [mlp, gatherCells, gatherQueries, gatherScreen, cellArgs, queryArgs] = await Promise.all([
            MlpEngine.create(device, {
                shape,
                precision,
                capacity: settings.batch,
                inferCapacity: INFER_CAPACITY,
                infer: 'tiled',
                inferOptions: { records: inferRecords },
                forwardOptions: { records: forwardRecords },
                gradientOptions: { chunk: gradientChunk },
                useEma: settings.ema !== 'off',
                train: { ...DEFAULT_TRAIN, learningRate: settings.learningRate, emaDecay: settings.ema === 'off' ? 0 : DEFAULT_TRAIN.emaDecay, emaWarmup: settings.ema === 'warmup' },
                trainHooks: neuralTailTrainHooks(settings.positionBands),
                inferHooks: neuralTailInferHooks(settings.positionBands),
            }),
            kernels.computeKernel({ label: 'radiance-cache/NeuralTail_GatherCellRecords', modules: [shade.module], code: gatherCellRecordsWgsl(layout), entryPoint: NEURAL_TAIL_ENTRY_POINT.gatherCellRecords, ownGroups }),
            kernels.computeKernel({ label: 'radiance-cache/NeuralTail_GatherQueries', modules: [shade.module], code: gatherQueriesWgsl(layout), entryPoint: NEURAL_TAIL_ENTRY_POINT.gatherQueries, ownGroups }),
            kernels.computeKernel({
                label: 'radiance-cache/NeuralTail_GatherScreenRecords',
                modules: [shade.module],
                code: gatherScreenRecordsWgsl(),
                entryPoint: NEURAL_TAIL_ENTRY_POINT.gatherScreenRecords,
                ownGroups: [{ label: 'radiance-cache/neural tail screen', entries: NEURAL_TAIL_SCREEN_GROUP_ENTRIES }],
            }),
            compute.createIndirectArgs({ label: 'radiance-cache/NeuralTail_GatherCellRecords args', count: word(HASH_GRID_COUNTER.visibilityCount0), itemsPerWorkgroup: NEURAL_TAIL_WORKGROUP }),
            compute.createIndirectArgs({ label: 'radiance-cache/NeuralTail_GatherQueries args', count: word(HASH_GRID_COUNTER.visibilityCount1), itemsPerWorkgroup: NEURAL_TAIL_WORKGROUP }),
        ]);
        const storage = BUFFER_USAGE.storage | BUFFER_USAGE.copySrc | BUFFER_USAGE.copyDst;
        const params = device.createBuffer({ label: 'radiance-cache/neural tail params', size: NEURAL_TAIL_PARAMS_BYTES, usage: BUFFER_USAGE.uniform | BUFFER_USAGE.copyDst });
        const counts = device.createBuffer({ label: 'radiance-cache/neural tail counts', size: NEURAL_TAIL_COUNTS_BYTES, usage: storage });
        const set = device.createBuffer({ label: 'radiance-cache/neural tail training set', size: capacity * NEURAL_TAIL_RECORD_VEC4 * VEC4_BYTES, usage: storage });
        const trainGroup = device.createBindGroup({
            label: 'radiance-cache/neural tail train hooks',
            layout: mlp.trainLayout,
            entries: [
                { binding: 0, resource: { buffer: set } },
                { binding: 1, resource: { buffer: counts } },
                { binding: 2, resource: { buffer: params } },
            ],
        });
        const cellGroup = device.createBindGroup({ label: 'radiance-cache/neural tail cell records', layout: gatherCells.ownLayouts[2], entries: neuralGroupEntries(params, set, counts) });
        this.prepared = { device, mlp, shape, gatherCells, gatherQueries, gatherScreen, cellArgs, queryArgs, params, counts, set, setCapacity: capacity, tileWords, trainGroup, cellGroup };
        this.restartNetwork();
    }

    /** Forgets what the network learned: fresh weights (a zero output layer, so it answers 0 at first) and the gate closed. */
    reset(): void {
        if (this.prepared) this.restartNetwork();
    }

    /** Inside the hash grid's frame (`HashGridTail`): gathers this frame's cell records and queries, then answers the queries once the gate is open. */
    encode(encoder: GPUCommandEncoder, tailFrame: HashGridTailFrame): void {
        const prepared = this.require();
        const context = this.context;
        if (!context) throw new Error('@pmndrs/gi: the neural tail is used before prepare()');
        const screenData = this.screenDataFor(prepared, tailFrame);
        this.writeParams(prepared, tailFrame, screenData.queryCapacity);
        encoder.clearBuffer(prepared.counts);

        const gather = encoder.beginComputePass({ label: `${TIMER_PREFIX}/NeuralTailGather`, timestampWrites: context.timer.computePass(`${TIMER_PREFIX}/NeuralTailGather`) });
        prepared.cellArgs.encode(gather);
        prepared.gatherCells.bind(gather);
        gather.setBindGroup(1, screenData.cellGroups[0]);
        gather.setBindGroup(2, screenData.cellGroups[1]);
        gather.setBindGroup(NEURAL_TAIL_HASH_GRID_GROUP, prepared.cellGroup);
        gather.dispatchWorkgroupsIndirect(prepared.cellArgs.buffer, 0);
        if (this.active) {
            prepared.queryArgs.encode(gather);
            prepared.gatherQueries.bind(gather);
            gather.setBindGroup(1, screenData.queryGroups[0]);
            gather.setBindGroup(2, screenData.queryGroups[1]);
            gather.setBindGroup(NEURAL_TAIL_HASH_GRID_GROUP, screenData.queryGroups[2]);
            gather.dispatchWorkgroupsIndirect(prepared.queryArgs.buffer, 0);
        }
        gather.end();

        if (this.active) {
            // Inference's record count is the second-bounce entries' (its record r is entry r).
            encoder.copyBufferToBuffer(prepared.tileWords, HASH_GRID_COUNTER.visibilityCount1 * WORD_BYTES, prepared.counts, NEURAL_TAIL_COUNT_WORD.queries * WORD_BYTES, WORD_BYTES);
            prepared.mlp.encodeInfer(encoder, prepared.counts, NEURAL_TAIL_COUNT_WORD.queries, screenData.inferGroup, context.timer.computePass(`${TIMER_PREFIX}/NeuralTailInference`));
        }
        this.gatheredThisFrame = true;
    }

    /**
     * After the screen's irradiance: the screen records, then this frame's
     * training steps over the set. Does nothing on a frame the hash grid
     * didn't run the tail (multibounce off).
     */
    encodeTraining(encoder: GPUCommandEncoder, screen: NeuralTailScreen): void {
        const prepared = this.require();
        const context = this.context;
        if (!context || !this.gatheredThisFrame) return;
        this.gatheredThisFrame = false;
        const settings = this.settings;
        if (settings.screenStride > 0) {
            const group = this.screenGroupFor(prepared, screen.irradiance);
            const pass = encoder.beginComputePass({ label: `${TIMER_PREFIX}/NeuralTailScreenRecords`, timestampWrites: context.timer.computePass(`${TIMER_PREFIX}/NeuralTailScreenRecords`) });
            prepared.gatherScreen.bind(pass);
            pass.setBindGroup(1, group);
            const blocks = [Math.ceil(screen.width / settings.screenStride), Math.ceil(screen.height / settings.screenStride)];
            pass.dispatchWorkgroups(Math.ceil(blocks[0] / NEURAL_TAIL_SCREEN_GROUP), Math.ceil(blocks[1] / NEURAL_TAIL_SCREEN_GROUP));
            pass.end();
        }
        const steps = neuralTailStepsThisFrame(this.trainedSteps, settings);
        for (let step = 0; step < steps; step++) {
            prepared.mlp.encodeTrainStep(encoder, prepared.counts, NEURAL_TAIL_COUNT_WORD.records, prepared.trainGroup, context.timer.computePass(`${TIMER_PREFIX}/NeuralTailTraining`));
        }
        this.trainedSteps += steps;
    }

    dispose(): void {
        const prepared = this.prepared;
        this.screenData?.queries.destroy();
        this.screenData = null;
        this.screenGroup = null;
        if (!prepared) return;
        prepared.mlp.destroy();
        for (const buffer of [prepared.params, prepared.counts, prepared.set]) buffer.destroy();
        prepared.cellArgs.dispose();
        prepared.queryArgs.dispose();
        this.prepared = null;
    }

    private restartNetwork(): void {
        const prepared = this.require();
        // The pathtracer's start (NrcCache.freshWeights, outputInit 'zero'): He-initialised layers and a zero
        // output layer, so a fresh network answers 0 and its first steps don't throw light around.
        const weights = initWeights(prepared.shape, this.settings.seed);
        weights.fill(0, layerOffset(prepared.shape, prepared.shape.hidden));
        prepared.mlp.uploadWeights(weights);
        this.trainedSteps = 0;
        this.gatheredThisFrame = false;
        this.boxCentre = null;
    }

    private writeParams(prepared: Prepared, tailFrame: HashGridTailFrame, queryCapacity: number): void {
        const words = new Uint32Array(this.paramWords);
        const floats = new Float32Array(this.paramWords);
        const at = NEURAL_TAIL_PARAMS_WORD;
        const position = tailFrame.frame.camera.position;
        this.boxCentre ??= [position[0], position[1], position[2]];
        const half = this.settings.boundsHalfSize;
        const centre = this.boxCentre;
        const box = neuralTailBox(this.settings.bounds ?? { min: [centre[0] - half, centre[1] - half, centre[2] - half], max: [centre[0] + half, centre[1] + half, centre[2] + half] });
        floats.set(box.lo, at.boxLo);
        floats.set(box.scale, at.boxScale);
        floats.set(screenRayMatrix(tailFrame.frame.camera), at.inverseViewProjection);
        floats.set(tailFrame.frame.camera.position, at.eye);
        words[at.frameIndex] = tailFrame.frame.frame >>> 0;
        words[at.setCapacity] = prepared.setCapacity;
        words[at.queryCapacity] = queryCapacity;
        words[at.cellRecords] = this.settings.cellRecords;
        words[at.tailList] = tailFrame.rayLists.neuralTailBuffer;
        words[at.screenStride] = this.settings.screenStride;
        words[at.screenWidth] = tailFrame.frame.width;
        words[at.screenHeight] = tailFrame.frame.height;
        prepared.device.queue.writeBuffer(prepared.params, 0, this.paramWords);
    }

    /** The queries and bind groups for this screen's per-ray lists, made when the lists change. */
    private screenDataFor(prepared: Prepared, tailFrame: HashGridTailFrame): ScreenData {
        const current = this.screenData;
        if (current && current.listWords === tailFrame.listWords && current.queryCapacity >= tailFrame.rays.capacity) return current;
        current?.queries.destroy();
        const { device } = prepared;
        const queryCapacity = tailFrame.rays.capacity;
        const queries = device.createBuffer({
            label: 'radiance-cache/neural tail queries',
            size: Math.max(queryCapacity, 1) * NEURAL_TAIL_QUERY_VEC4 * VEC4_BYTES,
            usage: BUFFER_USAGE.storage | BUFFER_USAGE.copySrc,
        });
        const group = (kernel: ComputeKernel, index: number, entries: readonly GPUBindGroupEntry[], label: string): GPUBindGroup =>
            device.createBindGroup({ label, layout: kernel.ownLayouts[index], entries: [...entries] });
        const data: ScreenData = {
            listWords: tailFrame.listWords,
            queries,
            queryCapacity,
            cellGroups: [
                group(prepared.gatherCells, 0, tailFrame.cacheEntries, 'radiance-cache/neural tail cells: hash grid'),
                group(prepared.gatherCells, 1, tailFrame.rayDataEntries, 'radiance-cache/neural tail cells: ray data'),
            ],
            queryGroups: [
                group(prepared.gatherQueries, 0, tailFrame.cacheEntries, 'radiance-cache/neural tail queries: hash grid'),
                group(prepared.gatherQueries, 1, tailFrame.rayDataEntries, 'radiance-cache/neural tail queries: ray data'),
                group(prepared.gatherQueries, 2, neuralGroupEntries(prepared.params, queries, prepared.counts), 'radiance-cache/neural tail queries'),
            ],
            inferGroup: device.createBindGroup({
                label: 'radiance-cache/neural tail infer hooks',
                layout: prepared.mlp.inferLayout,
                entries: [
                    { binding: 0, resource: { buffer: queries } },
                    { binding: 1, resource: { buffer: tailFrame.listWords } },
                    { binding: 2, resource: { buffer: prepared.params } },
                ],
            }),
        };
        this.screenData = data;
        return data;
    }

    private screenGroupFor(prepared: Prepared, irradiance: GPUTexture): GPUBindGroup {
        const current = this.screenGroup;
        if (current && current.texture === irradiance) return current.group;
        const group = prepared.device.createBindGroup({
            label: 'radiance-cache/neural tail screen records',
            layout: prepared.gatherScreen.ownLayouts[0],
            entries: [...neuralGroupEntries(prepared.params, prepared.set, prepared.counts), { binding: NEURAL_TAIL_BINDING.irradiance, resource: irradiance.createView() }],
        });
        this.screenGroup = { texture: irradiance, group };
        return group;
    }

    private require(): Prepared {
        if (!this.prepared) throw new Error('@pmndrs/gi: the neural tail is used before prepare()');
        return this.prepared;
    }
}

function neuralGroupEntries(params: GPUBuffer, records: GPUBuffer, counts: GPUBuffer): GPUBindGroupEntry[] {
    return [
        { binding: NEURAL_TAIL_BINDING.params, resource: { buffer: params } },
        { binding: NEURAL_TAIL_BINDING.records, resource: { buffer: records } },
        { binding: NEURAL_TAIL_BINDING.counts, resource: { buffer: counts } },
    ];
}

/** `inverse(P × V × T(eye))` with the frame's jitter (camera.ts): a pixel's ray direction from the eye. */
function screenRayMatrix(camera: FrameCamera): number[] {
    return cameraRelativeInverseViewProjection(camera);
}
