/**
 * The neural tail (phase 6 experiment P6.4, register E11): pmndrs/pathtracer's
 * neural radiance cache network (`@pathtracer/mlp`, MIT; Müller et al. 2021's
 * design as VkNRC builds it; THIRD_PARTY.md) as an enhancement of the GI-1.2
 * hash grid, not a replacement (docs/design/radiance-cache.md, "Neural tail").
 *
 * The grid stays the near cache and keeps reacting as fast as it does: every
 * probe-ray hit is still lit, inserted and resolved by it. What the network
 * adds is the second bounce's tail. GI-1.2's multibounce gives a first-bounce
 * cell the light its second hit reflects from *direct* light only (two
 * bounces in all); with the tail, `UpdateMultibounceCells` adds the
 * network's answer for the light the second hit reflects from *indirect*
 * light. The network learns that quantity from the grid's own indirect values
 * (cell records) and one-ray estimates of it on the surfaces the camera sees
 * (screen records), so the cells' indirect values become a fixed point that
 * carries every bounce in expectation (the furnace: ρ (ρE/π) / (1 − ρ²) =
 * 0.4244 where two bounces read 0.3183).
 *
 * Per frame, all inside the hash grid's frame (its `HashGridTail` slot, after
 * `UpdateTiles`, with GI-1.2's own second-bounce rays only): the counts
 * cleared; cell and screen records gathered (`neural-tail-wgsl.ts`); the
 * frame's training steps over them; then, once the gate is open, the queries
 * gathered and answered into the grid's `neural_tail_buffer` list.
 *
 * The gate opens once the GPU has taken `warmupSteps` training steps (steps
 * with no records don't count; the count is read back a frame or two late,
 * never waited on). Until then the grid runs as GI-1.2, and the network
 * trains at `warmupStepsPerFrame`, as it does again for `changeBoostFrames`
 * after a change with no bounds (the sun, the sky).
 *
 * The package is loaded at run time (`mlp-loader.ts`); the technique hands it
 * to `prepare`.
 */

import type { MlpEngine, MlpShape } from '@pathtracer/mlp';
import type { GpuIndirectArgs } from '../../../platform/contract/compute.ts';
import { GILimitError } from '../../../platform/contract/errors.ts';
import type { FrameCamera } from '../../../platform/contract/frame.ts';
import type { TechniqueContext } from '../../../platform/contract/technique.ts';
import type { ComputeKernel } from '../../../platform/contract/wgsl.ts';
import { cameraRelativeInverseViewProjection } from '../camera.ts';
import { HASH_GRID_COUNTER } from '../world-cache/capacity.ts';
import type { HashGridCache, HashGridTail, HashGridTailFrame } from '../world-cache/hash-grid-cache.ts';
import { RAY_DATA_GROUP_ENTRIES } from '../world-cache/hash-grid-population-wgsl.ts';
import { HASH_GRID_CACHE_GROUP_ENTRIES } from '../world-cache/hash-grid-wgsl.ts';
import type { MlpModule } from './mlp-loader.ts';
import { neuralTailBox, neuralTailGateOpen, neuralTailNeedsRecentre, neuralTailQueryCapacity, neuralTailSetCapacity, neuralTailStepsThisFrame } from './neural-tail-cpu.ts';
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
    NEURAL_TAIL_SCREEN_GRID_GROUP,
    NEURAL_TAIL_SCREEN_GROUP,
    NEURAL_TAIL_SCREEN_OWN_GROUP,
    NEURAL_TAIL_WORKGROUP,
    neuralTailInferHooks,
    neuralTailTrainHooks,
} from './neural-tail-wgsl.ts';

/** `GPUBufferUsage` and `GPUMapMode` bits (the globals don't exist in Node, where the tests import this file). */
const BUFFER_USAGE = { mapRead: 0x1, copySrc: 0x4, copyDst: 0x8, uniform: 0x40, storage: 0x80 } as const;
const MAP_MODE_READ = 0x1;
const VEC4_BYTES = 16;
const WORD_BYTES = 4;
/** Timed as `radiance-cache/<section>`, beside the hash grid's sections. */
const TIMER_PREFIX = 'radiance-cache';
/** Inference records per call at most (the engine sizes nothing by it; the query buffer bounds the work). */
const INFER_CAPACITY = 1 << 26;
/** The Adam step's word in `MlpEngine.trainState` (`NnState.step`). */
const TRAIN_STATE_STEP_BYTE = 4;

/**
 * Records per workgroup of the tiled kernels: the pathtracer's default (64 at
 * f16, 48 at f32) or the largest multiple of 16 below it that fits the
 * device's workgroup memory. `requiredLimits` asks for the adapter's maximum
 * (DECISIONS 2026-11-20; 32 KiB on the RTX 3060); a device at WebGPU's
 * default 16 KiB still runs the compact network, with smaller tiles. A
 * network that can't fit at all is a `GILimitError` naming the limit.
 */
function tiledRecords(mlp: MlpModule, shape: MlpShape, precision: 'f16' | 'f32', limit: number, bytes: (records: number) => number): number {
    for (let records = mlp.defaultForwardRecords(precision); records >= 16; records -= 16) {
        if (bytes(records) <= limit) return records;
    }
    throw new GILimitError('maxComputeWorkgroupStorageSize', bytes(16), limit);
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
    readonly stepReadback: GPUBuffer;
    readonly tileWords: GPUBuffer;
    readonly trainGroup: GPUBindGroup;
    readonly cellGroup: GPUBindGroup;
    readonly screenOwnGroup: GPUBindGroup;
    /** The grid's discard probability when the tail was made, for sizing the queries. */
    readonly discardProbability: () => number;
}

/** Bind groups over one screen's lists and queries. */
interface ScreenData {
    readonly listWords: GPUBuffer;
    readonly queries: GPUBuffer;
    readonly queryCapacity: number;
    readonly cellGroups: readonly [GPUBindGroup, GPUBindGroup];
    readonly queryGroups: readonly [GPUBindGroup, GPUBindGroup, GPUBindGroup];
    readonly screenGridGroup: GPUBindGroup;
    readonly inferGroup: GPUBindGroup;
}

export class NeuralTail implements HashGridTail {
    readonly settings: NeuralTailSettings;

    private prepared: Prepared | null = null;
    private screenData: ScreenData | null = null;
    private context: TechniqueContext | null = null;
    /** Training steps the GPU has taken since the network (re)started, as last read back. */
    private gpuSteps = 0;
    /** The step readback: `copied` once its copy is encoded, `mapping` once its map is asked for. */
    private readbackState: 'idle' | 'copied' | 'mapping' = 'idle';
    /** Bumped on every restart, so a readback from before it is dropped. */
    private generation = 0;
    /** Frames left at the warm-up training rate after a change with no bounds. */
    private boostFramesLeft = 0;
    /** The default box's centre: the camera where the network (re)started, taken at its first frame. */
    private boxCentre: readonly [number, number, number] | null = null;
    private readonly paramWords = new ArrayBuffer(NEURAL_TAIL_PARAMS_BYTES);

    constructor(settings: Partial<NeuralTailSettings> = {}) {
        this.settings = neuralTailSettings(settings);
    }

    /** The gate is open: the network has trained `warmupSteps` steps, and the grid adds its answers. */
    get active(): boolean {
        return this.prepared !== null && neuralTailGateOpen(this.gpuSteps, this.settings.warmupSteps);
    }

    /** Training steps the GPU has taken since the network was (re)started (read back a frame or two late). */
    get steps(): number {
        return this.gpuSteps;
    }

    /** The network, for tests and the timing page. */
    get network(): MlpEngine | null {
        return this.prepared?.mlp ?? null;
    }

    /** The training set's records and its count (word `NEURAL_TAIL_COUNT_WORD.records` of `counts`), and the queries, for tests. */
    get buffers(): { readonly set: GPUBuffer; readonly counts: GPUBuffer; readonly setCapacity: number; readonly queries: GPUBuffer | null; readonly queryCapacity: number } | null {
        const prepared = this.prepared;
        return prepared ? { set: prepared.set, counts: prepared.counts, setCapacity: prepared.setCapacity, queries: this.screenData?.queries ?? null, queryCapacity: this.screenData?.queryCapacity ?? 0 } : null;
    }

    /** GPU bytes the tail made: the network (weights, Adam, EMA, training scratch), the training set, the queries, small buffers. Its answers live in the grid's lists. */
    get memoryBytes(): number {
        const prepared = this.prepared;
        if (!prepared) return 0;
        const own = prepared.params.size + prepared.counts.size + prepared.set.size + prepared.stepReadback.size + prepared.cellArgs.memoryBytes + prepared.queryArgs.memoryBytes;
        return own + prepared.mlp.bufferBytes + (this.screenData?.queries.size ?? 0);
    }

    get stats(): Readonly<Record<string, number>> {
        const prepared = this.prepared;
        if (!prepared) return {};
        return {
            'training steps': this.gpuSteps,
            'gate open': this.active ? 1 : 0,
            'network weights': prepared.mlp.weightCount,
            'GPU memory (bytes)': this.memoryBytes,
        };
    }

    /**
     * Compiles the network and the gather kernels and makes the training
     * set, after the grid's own `prepare` (its layout and counters), with the
     * package `loadMlp` gave. Starts the network from fresh weights. Throws a
     * `GILimitError` when the device can't hold the network's tiles.
     */
    async prepare(context: TechniqueContext, grid: HashGridCache, mlp: MlpModule): Promise<void> {
        this.dispose();
        this.context = context;
        const { device, kernels, compute, shade } = context;
        const settings = this.settings;
        const shape = mlp.MLP_SHAPES[settings.network];
        const precision = settings.precision === 'auto' ? (device.features.has('shader-f16') ? 'f16' : 'f32') : settings.precision;
        const limit = device.limits.maxComputeWorkgroupStorageSize;
        const inferRecords = tiledRecords(mlp, shape, precision, limit, (records) => mlp.inferTiledWorkgroupBytes({ shape, precision }, { records }));
        const forwardRecords = tiledRecords(mlp, shape, precision, limit, (records) => mlp.forwardWorkgroupBytes({ shape, precision }, { records }));
        // The tiled weight gradient stages `chunk` records of deltas and activations (f32 vec4 rows each).
        const rowBytes = Math.max(shape.inputs, shape.width) * 4;
        const gradientChunk = 2 * 32 * rowBytes + 16 <= limit ? 32 : 16;
        const capacity = neuralTailSetCapacity(settings);
        const layout = grid.layout;
        const tileWords = grid.buffers.tileWords;
        const word = (counter: number) => ({ buffer: tileWords, offset: counter * WORD_BYTES });
        const cacheGroup = { label: 'radiance-cache/hash-grid group', entries: HASH_GRID_CACHE_GROUP_ENTRIES };
        const ownGroup = { label: 'radiance-cache/neural tail', entries: NEURAL_TAIL_GROUP_ENTRIES };
        const ownGroups = [cacheGroup, { label: 'radiance-cache/hash-grid ray data', entries: RAY_DATA_GROUP_ENTRIES }, ownGroup];
        const encoding = mlp.nrcEncodingWgsl(settings.positionBands);
        const [engine, gatherCells, gatherQueries, gatherScreen, cellArgs, queryArgs] = await Promise.all([
            mlp.MlpEngine.create(device, {
                shape,
                precision,
                capacity: settings.batch,
                inferCapacity: INFER_CAPACITY,
                infer: 'tiled',
                inferOptions: { records: inferRecords },
                forwardOptions: { records: forwardRecords },
                gradientOptions: { chunk: gradientChunk },
                useEma: settings.ema !== 'off',
                train: { ...mlp.DEFAULT_TRAIN, learningRate: settings.learningRate, emaWarmup: settings.ema === 'warmup' },
                trainHooks: neuralTailTrainHooks(encoding),
                inferHooks: neuralTailInferHooks(encoding),
            }),
            kernels.computeKernel({ label: 'radiance-cache/NeuralTail_GatherCellRecords', modules: [shade.module], code: gatherCellRecordsWgsl(layout), entryPoint: NEURAL_TAIL_ENTRY_POINT.gatherCellRecords, ownGroups }),
            kernels.computeKernel({ label: 'radiance-cache/NeuralTail_GatherQueries', modules: [shade.module], code: gatherQueriesWgsl(layout), entryPoint: NEURAL_TAIL_ENTRY_POINT.gatherQueries, ownGroups }),
            kernels.computeKernel({
                label: 'radiance-cache/NeuralTail_GatherScreenRecords',
                modules: [shade.module],
                code: gatherScreenRecordsWgsl(layout),
                entryPoint: NEURAL_TAIL_ENTRY_POINT.gatherScreenRecords,
                ownGroups: [cacheGroup, ownGroup],
            }),
            compute.createIndirectArgs({ label: 'radiance-cache/NeuralTail_GatherCellRecords args', count: word(HASH_GRID_COUNTER.visibilityCount0), itemsPerWorkgroup: NEURAL_TAIL_WORKGROUP }),
            compute.createIndirectArgs({ label: 'radiance-cache/NeuralTail_GatherQueries args', count: word(HASH_GRID_COUNTER.visibilityCount1), itemsPerWorkgroup: NEURAL_TAIL_WORKGROUP }),
        ]);
        const storage = BUFFER_USAGE.storage | BUFFER_USAGE.copySrc | BUFFER_USAGE.copyDst;
        const params = device.createBuffer({ label: 'radiance-cache/neural tail params', size: NEURAL_TAIL_PARAMS_BYTES, usage: BUFFER_USAGE.uniform | BUFFER_USAGE.copyDst });
        const counts = device.createBuffer({ label: 'radiance-cache/neural tail counts', size: NEURAL_TAIL_COUNTS_BYTES, usage: storage });
        const set = device.createBuffer({ label: 'radiance-cache/neural tail training set', size: capacity * NEURAL_TAIL_RECORD_VEC4 * VEC4_BYTES, usage: storage });
        const stepReadback = device.createBuffer({ label: 'radiance-cache/neural tail step readback', size: WORD_BYTES, usage: BUFFER_USAGE.mapRead | BUFFER_USAGE.copyDst });
        const trainGroup = device.createBindGroup({
            label: 'radiance-cache/neural tail train hooks',
            layout: engine.trainLayout,
            entries: [
                { binding: 0, resource: { buffer: set } },
                { binding: 1, resource: { buffer: counts } },
                { binding: 2, resource: { buffer: params } },
            ],
        });
        const cellGroup = device.createBindGroup({ label: 'radiance-cache/neural tail cell records', layout: gatherCells.ownLayouts[2], entries: neuralGroupEntries(params, set, counts) });
        const screenOwnGroup = device.createBindGroup({ label: 'radiance-cache/neural tail screen records', layout: gatherScreen.ownLayouts[1], entries: neuralGroupEntries(params, set, counts) });
        this.prepared = {
            device,
            mlp: engine,
            shape,
            gatherCells,
            gatherQueries,
            gatherScreen,
            cellArgs,
            queryArgs,
            params,
            counts,
            set,
            setCapacity: capacity,
            stepReadback,
            tileWords,
            trainGroup,
            cellGroup,
            screenOwnGroup,
            discardProbability: () => grid.options.hashGridCacheDiscardMultibounceRayProbability,
        };
        this.restartNetwork(initialWeights(mlp, shape, settings.seed));
        this.freshWeights = () => initialWeights(mlp, shape, settings.seed);
    }

    /** Forgets what the network learned: fresh weights (a zero output layer, so it answers 0 at first) and the gate closed. */
    reset(): void {
        if (this.prepared && this.freshWeights) this.restartNetwork(this.freshWeights());
    }

    /** Inside the hash grid's frame (`HashGridTail`): this frame's records, training, then the queries and their answers once the gate is open. */
    encode(encoder: GPUCommandEncoder, tailFrame: HashGridTailFrame): void {
        const prepared = this.require();
        const context = this.context;
        if (!context) throw new Error('@pmndrs/gi: the neural tail is used before prepare()');
        const settings = this.settings;
        const camera = tailFrame.frame.camera.position;
        // The default box follows the camera: centred again, and the network started over, when it leaves the middle half.
        if (!settings.bounds && this.boxCentre && neuralTailNeedsRecentre(camera, this.boxCentre, settings.boundsHalfSize)) this.reset();
        this.pollSteps(prepared);
        // A change with no bounds (the sun, the sky) starts its epoch: train at the warm-up rate for a while.
        const frameIndex = tailFrame.frame.frame;
        if (tailFrame.frame.changes?.events.some((event) => event.box === null && event.epoch === frameIndex)) this.boostFramesLeft = settings.changeBoostFrames;
        const screenData = this.screenDataFor(prepared, tailFrame);
        this.writeParams(prepared, tailFrame, screenData.queryCapacity);
        encoder.clearBuffer(prepared.counts);

        // The records: cells (indirect on the first-bounce count) and the screen.
        const gather = encoder.beginComputePass({ label: `${TIMER_PREFIX}/NeuralTailGather`, timestampWrites: context.timer.computePass(`${TIMER_PREFIX}/NeuralTailGather`) });
        prepared.cellArgs.encode(gather);
        prepared.gatherCells.bind(gather);
        gather.setBindGroup(1, screenData.cellGroups[0]);
        gather.setBindGroup(2, screenData.cellGroups[1]);
        gather.setBindGroup(NEURAL_TAIL_HASH_GRID_GROUP, prepared.cellGroup);
        gather.dispatchWorkgroupsIndirect(prepared.cellArgs.buffer, 0);
        if (settings.screenStride > 0 && settings.screenRecords > 0) {
            prepared.gatherScreen.bind(gather);
            gather.setBindGroup(NEURAL_TAIL_SCREEN_GRID_GROUP, screenData.screenGridGroup);
            gather.setBindGroup(NEURAL_TAIL_SCREEN_OWN_GROUP, prepared.screenOwnGroup);
            const blocks = [Math.ceil(tailFrame.frame.width / settings.screenStride), Math.ceil(tailFrame.frame.height / settings.screenStride)];
            gather.dispatchWorkgroups(Math.ceil(blocks[0] / NEURAL_TAIL_SCREEN_GROUP), Math.ceil(blocks[1] / NEURAL_TAIL_SCREEN_GROUP));
        }
        gather.end();

        // The frame's training steps (a step with no records is skipped on the GPU and not counted).
        const steps = neuralTailStepsThisFrame(this.gpuSteps, settings, this.boostFramesLeft > 0);
        for (let step = 0; step < steps; step++) {
            prepared.mlp.encodeTrainStep(encoder, prepared.counts, NEURAL_TAIL_COUNT_WORD.records, prepared.trainGroup, context.timer.computePass(`${TIMER_PREFIX}/NeuralTailTraining`));
        }
        if (this.boostFramesLeft > 0) this.boostFramesLeft--;
        if (this.readbackState === 'idle') {
            encoder.copyBufferToBuffer(prepared.mlp.trainState, TRAIN_STATE_STEP_BYTE, prepared.stepReadback, 0, WORD_BYTES);
            this.readbackState = 'copied';
        }

        if (this.active) {
            const answer = encoder.beginComputePass({ label: `${TIMER_PREFIX}/NeuralTailGather`, timestampWrites: context.timer.computePass(`${TIMER_PREFIX}/NeuralTailGather`) });
            prepared.queryArgs.encode(answer);
            prepared.gatherQueries.bind(answer);
            answer.setBindGroup(1, screenData.queryGroups[0]);
            answer.setBindGroup(2, screenData.queryGroups[1]);
            answer.setBindGroup(NEURAL_TAIL_HASH_GRID_GROUP, screenData.queryGroups[2]);
            answer.dispatchWorkgroupsIndirect(prepared.queryArgs.buffer, 0);
            answer.end();
            // Inference's record count is the second-bounce entries' (its record r is entry r; past the queries' room it stores nothing).
            encoder.copyBufferToBuffer(prepared.tileWords, HASH_GRID_COUNTER.visibilityCount1 * WORD_BYTES, prepared.counts, NEURAL_TAIL_COUNT_WORD.queries * WORD_BYTES, WORD_BYTES);
            prepared.mlp.encodeInfer(encoder, prepared.counts, NEURAL_TAIL_COUNT_WORD.queries, screenData.inferGroup, context.timer.computePass(`${TIMER_PREFIX}/NeuralTailInference`));
        }
    }

    dispose(): void {
        const prepared = this.prepared;
        this.screenData?.queries.destroy();
        this.screenData = null;
        this.generation++;
        this.readbackState = 'idle';
        if (!prepared) return;
        prepared.mlp.destroy();
        for (const buffer of [prepared.params, prepared.counts, prepared.set, prepared.stepReadback]) buffer.destroy();
        prepared.cellArgs.dispose();
        prepared.queryArgs.dispose();
        this.prepared = null;
    }

    private freshWeights: (() => Float32Array) | null = null;

    private restartNetwork(weights: Float32Array): void {
        const prepared = this.require();
        prepared.mlp.uploadWeights(weights);
        this.gpuSteps = 0;
        this.generation++;
        this.boostFramesLeft = 0;
        this.boxCentre = null;
    }

    /**
     * The step count, read back without waiting: the copy encoded last frame
     * is mapped now (its frame is submitted), and the value lands a frame or
     * two later. A restart in between drops it.
     */
    private pollSteps(prepared: Prepared): void {
        if (this.readbackState !== 'copied') return;
        this.readbackState = 'mapping';
        const generation = this.generation;
        const buffer = prepared.stepReadback;
        buffer.mapAsync(MAP_MODE_READ).then(
            () => {
                const steps = new Uint32Array(buffer.getMappedRange())[0] ?? 0;
                buffer.unmap();
                if (generation === this.generation) this.gpuSteps = steps;
                if (this.prepared?.stepReadback === buffer) this.readbackState = 'idle';
            },
            () => {
                // Destroyed or lost: nothing to read.
            },
        );
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
        words[at.screenRecords] = this.settings.screenRecords;
        prepared.device.queue.writeBuffer(prepared.params, 0, this.paramWords);
    }

    /** The queries and bind groups for this screen's per-ray lists, made when the lists change. */
    private screenDataFor(prepared: Prepared, tailFrame: HashGridTailFrame): ScreenData {
        const current = this.screenData;
        if (current && current.listWords === tailFrame.listWords) return current;
        current?.queries.destroy();
        const { device } = prepared;
        // The second-bounce entries a frame can have: (1 − p) of the probe rays in expectation, with room to spare.
        const queryCapacity = neuralTailQueryCapacity(tailFrame.rays.capacity, prepared.discardProbability());
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
            screenGridGroup: group(prepared.gatherScreen, 0, tailFrame.cacheEntries, 'radiance-cache/neural tail screen: hash grid'),
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

    private require(): Prepared {
        if (!this.prepared) throw new Error('@pmndrs/gi: the neural tail is used before prepare()');
        return this.prepared;
    }
}

/** He-initialised layers and a zero output layer (the pathtracer's `NrcCache.freshWeights`, outputInit 'zero'), so a fresh network answers 0. */
function initialWeights(mlp: MlpModule, shape: MlpShape, seed: number): Float32Array {
    const weights = mlp.initWeights(shape, seed);
    weights.fill(0, mlp.layerOffset(shape, shape.hidden));
    return weights;
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
