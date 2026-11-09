/**
 * Per-pass GPU timing with WebGPU timestamp queries.
 *
 * Per frame, the host does this:
 *
 *     timer.beginFrame();
 *     const pass = encoder.beginComputePass({ timestampWrites: timer.computePass('ddgi/trace') });
 *     ...
 *     timer.resolve(encoder);
 *     device.queue.submit([encoder.finish()]);
 *     timer.readback();
 *
 * Results arrive a few frames later through {@link GpuTimer.timings} and
 * {@link GpuTimer.takeSamples}. Timing is a debugging aid, so by default it
 * must never break rendering:
 *
 *  - Without the `timestamp-query` feature every call does nothing, and
 *    `computePass` / `renderPass` return `undefined`, which
 *    `beginComputePass({ timestampWrites })` accepts as "no timing".
 *  - Allocation failures are reported by WebGPU later, and meanwhile the
 *    device hands back an *invalid* query set. Using it in a command buffer
 *    would invalidate the whole frame. So the resources are created inside
 *    error scopes, and no timestamp work is attached until the scopes have
 *    come back clean (the first frame or so goes untimed).
 *  - If allocation or a readback fails, the timer switches itself off for
 *    good, logs one warning and clears its timings.
 *  - A frame that cannot be timed (every slot busy, too many passes) is left
 *    untimed and counted in {@link GpuTimer.counters}.
 *
 * A benchmark cannot accept silent gaps, so `strict` mode turns each of those
 * cases into an exception, and {@link GpuTimer.waitForFreeSlot} lets it wait
 * for a slot instead of dropping the frame.
 *
 * A frame may be submitted in several command buffers. Pass `completesFrame:
 * false` to `readback` for all but the last, and give them all the same frame
 * number: their passes are published together, once the last one is read back.
 */

import type { GpuTimerScope } from '../contract/timing';
import { TimestampRing, type TimestampSlot } from './timestamp-ring';
import { timestampsToPasses, totalByName, type TimedPass } from './timestamp-results';

/** Everything one timed submit produced. */
export interface GpuTimerSample {
    /** The frame number given to `beginFrame` (or a running count). */
    readonly frame: number;
    /** Order of submission; samples always come out in this order. */
    readonly sequence: number;
    /** Every timed pass in submission order, one entry per `computePass`/`renderPass` call. */
    readonly passes: readonly TimedPass[];
}

export interface GpuTimerOptions {
    /**
     * How many submits can be waiting for readback at once. A readback usually
     * takes 2 to 4 frames, so 8 leaves headroom for a slow driver.
     */
    slotCount?: number;
    /** Timed passes per submit. Passes beyond this are not timed. */
    maxPassesPerFrame?: number;
    /**
     * Benchmark mode: a missing slot, too many passes, missing support or a
     * failed timer throws instead of leaving work untimed.
     */
    strict?: boolean;
    /** Where the one-time warnings go. */
    warn?: (message: string) => void;
}

/**
 * - `unsupported`: the device has no `timestamp-query`.
 * - `pending`: resources were created, but their error scopes haven't resolved.
 * - `ready`: timing works.
 * - `failed`: something went wrong; the timer is off for good.
 */
export type GpuTimerStatus = 'unsupported' | 'pending' | 'ready' | 'failed';

/** How often work was left untimed. All zero in a healthy benchmark. */
export interface GpuTimerCounters {
    /** `beginFrame` found every slot waiting for readback. */
    framesWithoutSlot: number;
    /** `computePass` / `renderPass` calls beyond `maxPassesPerFrame`. */
    passesOverCapacity: number;
}

const DEFAULT_SLOT_COUNT = 8;
/**
 * 32 since several DDGI volumes (M3, T3.5): each volume runs DDGI's passes
 * (mark, schedule, trace with `gi/direct`, blend, write, propagate, census)
 * in turn, about 8 a volume, and two volumes with the platform's own passes
 * went past the 16 one volume needed. 2 × 32 × 8 = 512 timestamps a set.
 */
const DEFAULT_MAX_PASSES_PER_FRAME = 32;
/** Samples nobody collects with `takeSamples` are dropped, oldest first, past this many. */
const MAX_BUFFERED_SAMPLES = 256;
const BYTES_PER_TIMESTAMP = 8;

interface SlotResources {
    querySet: GPUQuerySet;
    /** Query results land here first (QUERY_RESOLVE buffers can't be mapped)... */
    resolveBuffer: GPUBuffer;
    /** ...then are copied here, which the CPU can map. */
    readBuffer: GPUBuffer;
    /** The in-flight `mapAsync` for this slot, so `drain` can wait for it. */
    pendingReadback: Promise<void> | null;
}

/** The submits of one frame that have been read back so far. */
interface FrameInProgress {
    frame: number;
    completed: boolean;
    newestSequence: number;
    submits: Array<{ sequence: number; passes: readonly TimedPass[] }>;
}

export class GpuTimer implements GpuTimerScope {
    /**
     * Settles when the allocation error scopes have resolved, whether the
     * timer came up healthy or switched itself off. Never rejects.
     */
    readonly ready: Promise<void>;

    private readonly device: GPUDevice;
    private readonly ring: TimestampRing;
    private readonly warn: (message: string) => void;
    private strict: boolean;
    private resources: SlotResources[] = [];
    private currentStatus: GpuTimerStatus;
    private failureDetail = '';

    /** The slot of the submit being encoded, between `beginFrame` and `readback`. */
    private activeSlot: TimestampSlot | null = null;
    private activeSlotResolved = false;

    private latestTimings = new Map<string, number>();
    private frameInProgress: FrameInProgress | null = null;
    private samples: GpuTimerSample[] = [];
    private readonly countersValue: GpuTimerCounters = { framesWithoutSlot: 0, passesOverCapacity: 0 };
    private warnedAboutPassOverflow = false;
    private disposed = false;

    constructor(device: GPUDevice, options: GpuTimerOptions = {}) {
        this.device = device;
        this.warn = options.warn ?? ((message) => console.warn(message));
        this.strict = options.strict ?? false;
        this.ring = new TimestampRing(
            options.slotCount ?? DEFAULT_SLOT_COUNT,
            options.maxPassesPerFrame ?? DEFAULT_MAX_PASSES_PER_FRAME,
        );

        if (!device.features.has('timestamp-query')) {
            this.currentStatus = 'unsupported';
            this.ready = Promise.resolve();
            return;
        }
        this.currentStatus = 'pending';

        device.pushErrorScope('out-of-memory');
        device.pushErrorScope('validation');
        let thrown: unknown = null;
        try {
            this.resources = this.ring.slots.map((slot) => this.createResources(slot.index));
        } catch (error) {
            thrown = error;
        }
        // Pop both scopes even if creation threw, or they would stay on the
        // device and swallow unrelated errors later.
        const validationError = device.popErrorScope();
        const outOfMemoryError = device.popErrorScope();
        this.ready = Promise.all([validationError, outOfMemoryError]).then(
            ([validation, outOfMemory]) => {
                const problem = thrown ?? validation ?? outOfMemory;
                if (problem) this.fail('allocating the query sets failed', problem);
                else if (this.currentStatus === 'pending') this.currentStatus = 'ready';
            },
            (error: unknown) => this.fail('the allocation error scope was rejected', error),
        );
    }

    get status(): GpuTimerStatus {
        return this.currentStatus;
    }

    /**
     * True while the timer can still produce timings. It stays true while the
     * allocation is being confirmed; timings just start a frame or two later.
     */
    get enabled(): boolean {
        return this.currentStatus === 'pending' || this.currentStatus === 'ready';
    }

    get counters(): Readonly<GpuTimerCounters> {
        return this.countersValue;
    }

    /** Turns benchmark mode on or off; see {@link GpuTimerOptions.strict}. */
    setStrict(strict: boolean): void {
        this.strict = strict;
    }

    /**
     * Starts timing one submit. If every slot is still waiting for a readback,
     * this one goes untimed (or throws in strict mode; call `waitForFreeSlot`
     * first to avoid both).
     *
     * @param frame the host's frame number. Submits that share it are
     * published together; it defaults to a fresh number per submit.
     */
    beginFrame(frame?: number): void {
        this.abandonActiveSlot();
        if (this.strict) this.throwIfNotUsableForBenchmark();
        if (this.currentStatus !== 'ready') return;
        const slot = this.ring.acquire(frame);
        if (!slot) {
            this.countersValue.framesWithoutSlot++;
            if (this.strict) {
                throw new Error('GPU timer: every timestamp slot is waiting for readback. Call waitForFreeSlot() first.');
            }
            return;
        }
        this.activeSlot = slot;
        this.activeSlotResolved = false;
    }

    /** Timestamp writes for a compute pass. `undefined` when this frame isn't timed. */
    computePass(name: string): GPUComputePassTimestampWrites | undefined {
        return this.reserveQueries(name);
    }

    /** Timestamp writes for a render pass. `undefined` when this frame isn't timed. */
    renderPass(name: string): GPURenderPassTimestampWrites | undefined {
        return this.reserveQueries(name);
    }

    /** Encodes copying the timestamps to a readable buffer. Call after the last timed pass, before `finish()`. */
    resolve(encoder: GPUCommandEncoder): void {
        const slot = this.activeSlot;
        if (!slot || slot.passNames.length === 0 || this.currentStatus !== 'ready') return;
        const queryCount = slot.passNames.length * 2;
        const { querySet, resolveBuffer, readBuffer } = this.resources[slot.index];
        encoder.resolveQuerySet(querySet, 0, queryCount, resolveBuffer, 0);
        encoder.copyBufferToBuffer(resolveBuffer, 0, readBuffer, 0, queryCount * BYTES_PER_TIMESTAMP);
        this.activeSlotResolved = true;
    }

    /**
     * Starts the asynchronous readback. Call right after `queue.submit`.
     *
     * @param completesFrame `false` for every submit of a split frame except
     * the last. Their passes are held back until the last one arrives, so
     * `timings` never shows half a frame.
     */
    readback(completesFrame = true): void {
        const slot = this.activeSlot;
        this.activeSlot = null;
        if (!slot) return;

        const queryCount = this.ring.submit(slot);
        if (queryCount === 0) return;
        if (!this.activeSlotResolved || this.currentStatus !== 'ready') {
            // `resolve` was skipped, so the buffer holds nothing worth reading.
            this.ring.release(slot);
            return;
        }

        const resources = this.resources[slot.index];
        const passNames = [...slot.passNames];
        const byteLength = queryCount * BYTES_PER_TIMESTAMP;
        const { frame, sequence } = slot;
        const wasStale = (): boolean => this.disposed || this.ring.isStale(slot);

        resources.pendingReadback = resources.readBuffer
            .mapAsync(GPUMapMode.READ, 0, byteLength)
            .then(() => {
                const timestamps = new BigUint64Array(resources.readBuffer.getMappedRange(0, byteLength).slice(0));
                resources.readBuffer.unmap();
                if (wasStale()) return;
                const passes = timestampsToPasses(passNames, timestamps);
                this.recordSample({ frame, sequence, passes }, completesFrame);
            })
            .catch((error: unknown) => {
                // dispose() destroys the buffers mid-map, which is expected.
                if (this.disposed) return;
                // A readback that fails once will keep failing, so stop timing.
                this.fail('reading the timestamps back failed', error);
            })
            .finally(() => {
                resources.pendingReadback = null;
                this.ring.release(slot);
            });
    }

    /**
     * Pass times of the newest completed frame whose results have arrived,
     * summed per pass name. Only holds the passes that frame ran, and is empty
     * once the timer has failed.
     */
    get timings(): ReadonlyMap<string, number> {
        return this.latestTimings;
    }

    /** Returns and clears the submits whose results arrived since the last call, oldest first. */
    takeSamples(): GpuTimerSample[] {
        const taken = this.samples.sort((a, b) => a.sequence - b.sequence);
        this.samples = [];
        return taken;
    }

    /** Resolves when a slot is free, so the next `beginFrame` will not drop (or, in strict mode, throw). */
    async waitForFreeSlot(): Promise<void> {
        await this.ready;
        while (this.enabled && this.ring.idleCount === 0) {
            const pending = this.pendingReadbacks();
            if (pending.length === 0) return; // nothing in flight, so waiting would never end
            await Promise.race(pending);
        }
    }

    /** Waits for submitted work and every outstanding readback. Handy in tests and benchmarks. */
    async drain(): Promise<void> {
        await this.ready;
        if (this.strict) this.throwIfNotUsableForBenchmark();
        if (!this.enabled) return;
        await this.device.queue.onSubmittedWorkDone();
        await Promise.all(this.pendingReadbacks());
        if (this.strict) this.throwIfNotUsableForBenchmark();
    }

    /** Forgets old results and ignores readbacks that are still in flight. */
    reset(): void {
        this.ring.invalidate();
        this.activeSlot = null;
        this.latestTimings = new Map();
        this.frameInProgress = null;
        this.samples = [];
    }

    dispose(): void {
        this.disposed = true;
        this.ring.invalidate();
        this.activeSlot = null;
        this.destroyResources();
    }

    private createResources(slotIndex: number): SlotResources {
        const queryCount = this.ring.queriesPerSlot;
        const byteLength = queryCount * BYTES_PER_TIMESTAMP;
        return {
            querySet: this.device.createQuerySet({
                label: `gpu-timer queries ${slotIndex}`,
                type: 'timestamp',
                count: queryCount,
            }),
            resolveBuffer: this.device.createBuffer({
                label: `gpu-timer resolve ${slotIndex}`,
                size: byteLength,
                usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
            }),
            readBuffer: this.device.createBuffer({
                label: `gpu-timer read ${slotIndex}`,
                size: byteLength,
                usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
            }),
            pendingReadback: null,
        };
    }

    private pendingReadbacks(): Promise<void>[] {
        return this.resources.flatMap((slotResources) =>
            slotResources.pendingReadback ? [slotResources.pendingReadback] : [],
        );
    }

    private reserveQueries(
        name: string,
    ): { querySet: GPUQuerySet; beginningOfPassWriteIndex: number; endOfPassWriteIndex: number } | undefined {
        const slot = this.activeSlot;
        if (!slot || this.currentStatus !== 'ready') return undefined;
        const indices = this.ring.addPass(slot, name);
        if (!indices) {
            this.noteOverflow(name);
            return undefined;
        }
        return {
            querySet: this.resources[slot.index].querySet,
            beginningOfPassWriteIndex: indices.beginIndex,
            endOfPassWriteIndex: indices.endIndex,
        };
    }

    private noteOverflow(name: string): void {
        this.countersValue.passesOverCapacity++;
        if (this.strict) {
            throw new Error(
                `GPU timer: pass "${name}" does not fit; maxPassesPerFrame is ${this.ring.maxPassesPerSlot}.`,
            );
        }
        if (!this.warnedAboutPassOverflow) {
            this.warnedAboutPassOverflow = true;
            this.warn(
                `GPU timing: more than ${this.ring.maxPassesPerSlot} passes in one submit; "${name}" and later ones ` +
                    'are not timed. Raise maxPassesPerFrame. This warning is shown once.',
            );
        }
    }

    /** A frame that was begun but never read back (the host skipped it) must not hold its slot forever. */
    private abandonActiveSlot(): void {
        if (this.activeSlot) this.ring.release(this.activeSlot);
        this.activeSlot = null;
    }

    private throwIfNotUsableForBenchmark(): void {
        if (this.currentStatus === 'unsupported') {
            throw new Error('GPU timer: strict mode needs the timestamp-query feature on the device.');
        }
        if (this.currentStatus === 'failed') {
            throw new Error(`GPU timer: timing failed (${this.failureDetail}); a benchmark cannot continue untimed.`);
        }
        if (this.currentStatus === 'pending') {
            throw new Error('GPU timer: not ready yet. Await timer.ready before timing in strict mode.');
        }
    }

    private recordSample(sample: GpuTimerSample, completesFrame: boolean): void {
        this.samples.push(sample);
        if (this.samples.length > MAX_BUFFERED_SAMPLES) this.samples.shift();

        // Merge the submits of one frame. A split frame's submits run different
        // passes, so replacing `timings` per submit would drop the early ones.
        let building = this.frameInProgress;
        if (building && building.frame !== sample.frame) {
            // A late arrival from an older frame must not replace newer timings.
            if (sample.sequence < building.newestSequence) return;
            building = null;
        }
        if (!building) {
            building = { frame: sample.frame, completed: false, newestSequence: -1, submits: [] };
            this.frameInProgress = building;
        }
        building.submits.push({ sequence: sample.sequence, passes: sample.passes });
        building.newestSequence = Math.max(building.newestSequence, sample.sequence);
        building.completed ||= completesFrame;

        // Readbacks can finish out of order, so rebuild in submission order.
        // After the completing submit has arrived, a straggler republishes.
        if (building.completed) {
            const ordered = [...building.submits].sort((a, b) => a.sequence - b.sequence);
            this.latestTimings = totalByName(ordered.flatMap((submit) => submit.passes));
        }
    }

    /** Switches timing off for good: one warning, no more timestamp work, no stale numbers, resources freed. */
    private fail(reason: string, error: unknown): void {
        if (this.currentStatus === 'failed' || this.disposed) return;
        this.currentStatus = 'failed';
        this.activeSlot = null;
        const detail =
            error && typeof (error as { message?: unknown }).message === 'string'
                ? (error as { message: string }).message.trim()
                : String(error);
        this.failureDetail = `${reason}: ${detail}`;
        this.warn(`GPU timing is off: ${reason} (${detail}). Rendering is not affected.`);
        // Frozen numbers would look like live ones in an overlay.
        this.latestTimings = new Map();
        this.frameInProgress = null;
        this.destroyResources();
    }

    private destroyResources(): void {
        // Work that was already submitted keeps its resources alive until it finishes.
        for (const slotResources of this.resources) {
            slotResources.querySet.destroy();
            slotResources.resolveBuffer.destroy();
            slotResources.readBuffer.destroy();
        }
        this.resources = [];
    }
}
