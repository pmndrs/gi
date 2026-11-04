/**
 * GpuTimer against a fake device: checks the lifecycle and failure handling
 * with no GPU. The real thing is exercised by the device page (tools/gpu).
 *
 * The fake models what matters for correctness: `resolveQuerySet` writes
 * values into the slot's resolve buffer, `copyBufferToBuffer` moves them to
 * the slot's read buffer, and only that buffer's `mapAsync` gives them back.
 * Every resolve writes different numbers (resolve number n gives its passes
 * durations of about n + 1 ms), so results read from the wrong slot, or
 * attributed to the wrong frame, show up as wrong milliseconds.
 */

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { GpuTimer, type GpuTimerOptions } from './gpu-timer';

const USAGE = { QUERY_RESOLVE: 1, COPY_SRC: 2, COPY_DST: 4, MAP_READ: 8 };

interface FakeBuffer {
    usage: number;
    destroyed: boolean;
    data: BigUint64Array | null;
    mapAsync(mode: number, offset: number, size: number): Promise<void>;
    getMappedRange(offset: number, size: number): ArrayBuffer;
    unmap(): void;
    destroy(): void;
}

interface DeferredMap {
    buffer: FakeBuffer;
    resolve(): void;
    reject(error: Error): void;
}

interface FakeGpuOptions {
    withTimestamps?: boolean;
    /** What the error scopes report. Validation is popped first, then out-of-memory. */
    validationError?: { message: string } | null;
    outOfMemoryError?: { message: string } | null;
    /** `createQuerySet` throws synchronously. */
    createQuerySetThrows?: Error;
    /** `mapAsync` waits until the test settles it, so completion order is under the test's control. */
    manualMaps?: boolean;
}

/** Expected duration of pass `passNumber` in the `resolveNumber`-th resolve, in milliseconds. */
function expectedMs(resolveNumber: number, passNumber = 0): number {
    return resolveNumber + 1 + passNumber * 0.001;
}

function createFakeGpu(options: FakeGpuOptions = {}) {
    const { withTimestamps = true, validationError = null, outOfMemoryError = null, manualMaps = false } = options;
    const buffers: FakeBuffer[] = [];
    const deferredMaps: DeferredMap[] = [];
    const state = { resolveCount: 0, failMaps: false };

    function createBuffer(descriptor: { usage: number }): FakeBuffer {
        const buffer: FakeBuffer = {
            usage: descriptor.usage,
            destroyed: false,
            data: null,
            mapAsync() {
                if (state.failMaps) return Promise.reject(new Error('map failed'));
                if (!manualMaps) {
                    return buffer.destroyed ? Promise.reject(new Error('buffer destroyed')) : Promise.resolve();
                }
                return new Promise<void>((resolve, reject) => deferredMaps.push({ buffer, resolve, reject }));
            },
            getMappedRange(offset, size) {
                if (buffer.destroyed || !buffer.data) throw new Error('cannot map a destroyed or empty buffer');
                return buffer.data.buffer.slice(offset, offset + size) as ArrayBuffer;
            },
            unmap() {},
            destroy() {
                buffer.destroyed = true;
            },
        };
        buffers.push(buffer);
        return buffer;
    }

    const device = {
        features: new Set(withTimestamps ? ['timestamp-query'] : []),
        pushErrorScope: vi.fn(),
        popErrorScope: vi.fn().mockResolvedValueOnce(validationError).mockResolvedValueOnce(outOfMemoryError),
        createQuerySet: vi.fn(() => {
            if (options.createQuerySetThrows) throw options.createQuerySetThrows;
            return { destroy: vi.fn() };
        }),
        createBuffer: vi.fn(createBuffer),
        queue: { onSubmittedWorkDone: () => Promise.resolve() },
    };

    const encoder = {
        resolveQuerySet(_querySet: unknown, _first: number, count: number, destination: FakeBuffer) {
            const resolveNumber = state.resolveCount++;
            const values = new BigUint64Array(count);
            for (let pass = 0; pass < count / 2; pass++) {
                const start = BigInt(pass) * 1_000_000_000n;
                const duration = BigInt(Math.round(expectedMs(resolveNumber, pass) * 1e6));
                values[pass * 2] = start;
                values[pass * 2 + 1] = start + duration;
            }
            destination.data = values;
        },
        copyBufferToBuffer(source: FakeBuffer, _sourceOffset: number, destination: FakeBuffer, _destinationOffset: number, size: number) {
            destination.data = (source.data as BigUint64Array).slice(0, size / 8);
        },
    };

    return {
        device: device as unknown as GPUDevice,
        fake: device,
        encoder: encoder as unknown as GPUCommandEncoder,
        buffers,
        deferredMaps,
        state,
    };
}

type FakeGpu = ReturnType<typeof createFakeGpu>;

function createTimer(gpu: FakeGpu, options: GpuTimerOptions = {}): GpuTimer {
    return new GpuTimer(gpu.device, { warn: () => {}, ...options });
}

interface FrameOptions {
    frame?: number;
    completesFrame?: boolean;
}

/** One submit with the given compute passes, the way a host would encode it. */
function runFrame(timer: GpuTimer, gpu: FakeGpu, passNames: string[], options: FrameOptions = {}): void {
    timer.beginFrame(options.frame);
    for (const name of passNames) timer.computePass(name);
    timer.resolve(gpu.encoder);
    timer.readback(options.completesFrame);
}

/** Lets promise callbacks run. */
const flush = (): Promise<void> => new Promise((resolve) => setTimeout(resolve, 0));

/** Completes the n-th outstanding readback (in the order they were started). */
async function completeMap(gpu: FakeGpu, index: number): Promise<void> {
    gpu.deferredMaps[index].resolve();
    await flush();
}

beforeEach(() => {
    vi.stubGlobal('GPUBufferUsage', USAGE);
    vi.stubGlobal('GPUMapMode', { READ: 1 });
});

afterEach(() => {
    vi.unstubAllGlobals();
});

describe('without timestamp-query', () => {
    it('does nothing and hands out no timestamp writes', async () => {
        const gpu = createFakeGpu({ withTimestamps: false });
        const timer = createTimer(gpu);
        await timer.ready;

        timer.beginFrame();
        expect(timer.status).toBe('unsupported');
        expect(timer.enabled).toBe(false);
        expect(timer.computePass('a')).toBeUndefined();
        expect(timer.renderPass('b')).toBeUndefined();
        timer.resolve(gpu.encoder);
        timer.readback();
        await timer.drain();
        await timer.waitForFreeSlot();

        expect(gpu.fake.createQuerySet).not.toHaveBeenCalled();
        expect(timer.timings.size).toBe(0);
        expect(timer.takeSamples()).toEqual([]);
    });

    it('throws in strict mode, because a benchmark must not run untimed', async () => {
        const gpu = createFakeGpu({ withTimestamps: false });
        const timer = createTimer(gpu, { strict: true });
        await timer.ready;
        expect(() => timer.beginFrame()).toThrow(/timestamp-query/);
    });
});

describe('allocation', () => {
    it('attaches nothing until the allocation scopes come back clean', async () => {
        const timer = createTimer(createFakeGpu());

        timer.beginFrame();
        expect(timer.status).toBe('pending');
        expect(timer.enabled).toBe(true);
        expect(timer.computePass('early')).toBeUndefined();

        await timer.ready;
        expect(timer.status).toBe('ready');
    });

    it.each([
        ['validation', { validationError: { message: 'bad query set' } }, 'bad query set'],
        ['out-of-memory', { outOfMemoryError: { message: 'no memory left' } }, 'no memory left'],
    ] as const)('disables itself with one warning on an %s error', async (_name, gpuOptions, expectedText) => {
        const warn = vi.fn();
        const gpu = createFakeGpu(gpuOptions);
        const timer = createTimer(gpu, { warn });
        await timer.ready;

        expect(timer.status).toBe('failed');
        expect(warn).toHaveBeenCalledTimes(1);
        expect(warn.mock.calls[0][0]).toContain(expectedText);
        expect(gpu.buffers.every((buffer) => buffer.destroyed)).toBe(true);
        timer.beginFrame();
        expect(timer.computePass('p')).toBeUndefined();
    });

    it('pops both error scopes and disables itself when creation throws', async () => {
        const warn = vi.fn();
        const gpu = createFakeGpu({ createQuerySetThrows: new Error('createQuerySet blew up') });
        const timer = createTimer(gpu, { warn });
        await timer.ready;

        expect(gpu.fake.pushErrorScope).toHaveBeenCalledTimes(2);
        expect(gpu.fake.popErrorScope).toHaveBeenCalledTimes(2);
        expect(timer.status).toBe('failed');
        expect(warn).toHaveBeenCalledTimes(1);
        expect(warn.mock.calls[0][0]).toContain('createQuerySet blew up');
    });
});

describe('timing frames', () => {
    it('times a frame end to end and reports milliseconds per pass name', async () => {
        const gpu = createFakeGpu();
        const timer = createTimer(gpu);
        await timer.ready;

        timer.beginFrame(7);
        expect(timer.computePass('ddgi/trace')).toMatchObject({ beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 });
        expect(timer.renderPass('ddgi/shade')).toMatchObject({ beginningOfPassWriteIndex: 2, endOfPassWriteIndex: 3 });
        timer.resolve(gpu.encoder);
        timer.readback();
        await timer.drain();

        expect(timer.timings.get('ddgi/trace')).toBeCloseTo(expectedMs(0, 0), 9);
        expect(timer.timings.get('ddgi/shade')).toBeCloseTo(expectedMs(0, 1), 9);
        const [sample] = timer.takeSamples();
        expect(sample.frame).toBe(7);
        expect(sample.passes.map((pass) => pass.name)).toEqual(['ddgi/trace', 'ddgi/shade']);
        expect(timer.takeSamples()).toEqual([]);
    });

    it('gives every frame its own numbers when slots are reused in a different order than they were filled', async () => {
        const gpu = createFakeGpu({ manualMaps: true });
        const timer = createTimer(gpu, { slotCount: 3 });
        await timer.ready;

        // Frames 0, 1, 2 fill slots 0, 1, 2. Complete frame 1 first, so slot 1 is reused for frame 3.
        for (let frame = 0; frame < 3; frame++) runFrame(timer, gpu, ['first', 'second']);
        await completeMap(gpu, 1);
        runFrame(timer, gpu, ['first', 'second']); // resolve number 3, in slot 1
        // Complete everything else in a scrambled order.
        await completeMap(gpu, 3);
        await completeMap(gpu, 2);
        await completeMap(gpu, 0);

        const samples = timer.takeSamples();
        expect(samples.map((sample) => sample.sequence)).toEqual([0, 1, 2, 3]);
        for (const sample of samples) {
            expect(sample.passes.map((pass) => pass.name)).toEqual(['first', 'second']);
            expect(sample.passes[0].gpuMs).toBeCloseTo(expectedMs(sample.sequence, 0), 9);
            expect(sample.passes[1].gpuMs).toBeCloseTo(expectedMs(sample.sequence, 1), 9);
        }
    });

    it('keeps the newest frame in timings when readbacks finish out of order', async () => {
        const gpu = createFakeGpu({ manualMaps: true });
        const timer = createTimer(gpu);
        await timer.ready;
        for (let frame = 0; frame < 3; frame++) runFrame(timer, gpu, ['p']);

        await completeMap(gpu, 2);
        expect(timer.timings.get('p')).toBeCloseTo(expectedMs(2), 9);
        await completeMap(gpu, 0);
        await completeMap(gpu, 1);

        // The older frames arrived late and must not replace the newer result.
        expect(timer.timings.get('p')).toBeCloseTo(expectedMs(2), 9);
        expect(timer.takeSamples().map((sample) => sample.sequence)).toEqual([0, 1, 2]);
    });

    it('sums passes that share a name within a frame', async () => {
        const gpu = createFakeGpu();
        const timer = createTimer(gpu);
        await timer.ready;
        runFrame(timer, gpu, ['cascade', 'cascade']);
        await timer.drain();
        expect(timer.timings.get('cascade')).toBeCloseTo(expectedMs(0, 0) + expectedMs(0, 1), 9);
    });

    it('does not hold a slot for a frame that was begun but never read back', async () => {
        const timer = createTimer(createFakeGpu(), { slotCount: 1 });
        await timer.ready;
        for (let frame = 0; frame < 3; frame++) {
            timer.beginFrame();
            expect(timer.computePass('p')).toBeDefined();
        }
    });
});

describe('split frames (two submits, one frame)', () => {
    it('publishes both submits together once the last one is read back', async () => {
        const gpu = createFakeGpu({ manualMaps: true });
        const timer = createTimer(gpu);
        await timer.ready;
        runFrame(timer, gpu, ['early'], { frame: 5, completesFrame: false });
        runFrame(timer, gpu, ['late'], { frame: 5, completesFrame: true });

        await completeMap(gpu, 0);
        expect(timer.timings.size).toBe(0); // half a frame is never shown
        await completeMap(gpu, 1);

        expect([...timer.timings.keys()].sort()).toEqual(['early', 'late']);
    });

    it('still includes the early submit when it is read back after the late one', async () => {
        const gpu = createFakeGpu({ manualMaps: true });
        const timer = createTimer(gpu);
        await timer.ready;
        runFrame(timer, gpu, ['early'], { frame: 5, completesFrame: false });
        runFrame(timer, gpu, ['late'], { frame: 5, completesFrame: true });

        await completeMap(gpu, 1);
        expect([...timer.timings.keys()]).toEqual(['late']);
        await completeMap(gpu, 0);

        expect([...timer.timings.keys()].sort()).toEqual(['early', 'late']);
    });

    it('does not carry a pass over from the previous frame', async () => {
        const gpu = createFakeGpu();
        const timer = createTimer(gpu);
        await timer.ready;
        runFrame(timer, gpu, ['only-in-frame-1'], { frame: 1 });
        await timer.drain();
        runFrame(timer, gpu, ['only-in-frame-2'], { frame: 2 });
        await timer.drain();
        expect([...timer.timings.keys()]).toEqual(['only-in-frame-2']);
    });
});

describe('a full ring', () => {
    it('leaves the frame untimed and counts it', async () => {
        const gpu = createFakeGpu({ manualMaps: true });
        const timer = createTimer(gpu, { slotCount: 2 });
        await timer.ready;
        runFrame(timer, gpu, ['p']);
        runFrame(timer, gpu, ['p']);

        timer.beginFrame();
        expect(timer.computePass('p')).toBeUndefined();
        expect(timer.counters.framesWithoutSlot).toBe(1);

        await completeMap(gpu, 0);
        timer.beginFrame();
        expect(timer.computePass('p')).toBeDefined();
    });

    it('throws in strict mode', async () => {
        const gpu = createFakeGpu({ manualMaps: true });
        const timer = createTimer(gpu, { slotCount: 1, strict: true });
        await timer.ready;
        runFrame(timer, gpu, ['p']);
        expect(() => timer.beginFrame()).toThrow(/waitForFreeSlot/);
        expect(timer.counters.framesWithoutSlot).toBe(1);
    });

    it('waitForFreeSlot waits for a readback, so strict mode never has to drop a frame', async () => {
        const gpu = createFakeGpu({ manualMaps: true });
        const timer = createTimer(gpu, { slotCount: 1, strict: true });
        await timer.ready;
        runFrame(timer, gpu, ['p']);

        let freed = false;
        const waiting = timer.waitForFreeSlot().then(() => {
            freed = true;
        });
        await flush();
        expect(freed).toBe(false);

        await completeMap(gpu, 0);
        await waiting;
        expect(freed).toBe(true);
        expect(() => timer.beginFrame()).not.toThrow();
    });
});

describe('too many passes', () => {
    it('counts the overflow and warns once', async () => {
        const warn = vi.fn();
        const gpu = createFakeGpu();
        const timer = createTimer(gpu, { maxPassesPerFrame: 2, warn });
        await timer.ready;

        timer.beginFrame();
        expect(timer.computePass('a')).toBeDefined();
        expect(timer.computePass('b')).toBeDefined();
        expect(timer.computePass('c')).toBeUndefined();
        expect(timer.computePass('d')).toBeUndefined();
        timer.beginFrame();
        timer.computePass('a');
        timer.computePass('b');
        expect(timer.renderPass('e')).toBeUndefined();

        expect(timer.counters.passesOverCapacity).toBe(3);
        expect(warn).toHaveBeenCalledTimes(1);
        expect(warn.mock.calls[0][0]).toContain('maxPassesPerFrame');
    });

    it('throws in strict mode', async () => {
        const timer = createTimer(createFakeGpu(), { maxPassesPerFrame: 1, strict: true });
        await timer.ready;
        timer.beginFrame();
        timer.computePass('a');
        expect(() => timer.computePass('b')).toThrow(/maxPassesPerFrame/);
    });
});

describe('failure while running', () => {
    it('disables itself with one warning and clears the timings, so an overlay cannot show frozen numbers', async () => {
        const warn = vi.fn();
        const gpu = createFakeGpu();
        const timer = createTimer(gpu, { warn });
        await timer.ready;

        runFrame(timer, gpu, ['p']);
        await timer.drain();
        expect(timer.timings.size).toBe(1);

        gpu.state.failMaps = true;
        runFrame(timer, gpu, ['p']);
        await timer.drain();

        expect(timer.status).toBe('failed');
        expect(timer.timings.size).toBe(0);
        expect(warn).toHaveBeenCalledTimes(1);
        expect(() => timer.beginFrame()).not.toThrow();
        expect(timer.computePass('p')).toBeUndefined();
    });

    it('throws in strict mode instead of carrying on untimed', async () => {
        const gpu = createFakeGpu();
        const timer = createTimer(gpu, { strict: true });
        await timer.ready;
        gpu.state.failMaps = true;
        runFrame(timer, gpu, ['p']);
        await expect(timer.drain()).rejects.toThrow(/timing failed/);
        expect(() => timer.beginFrame()).toThrow(/timing failed/);
    });
});

describe('reset and dispose', () => {
    it('ignores readbacks that were in flight when reset was called', async () => {
        const gpu = createFakeGpu();
        const timer = createTimer(gpu);
        await timer.ready;
        runFrame(timer, gpu, ['old']);
        timer.reset();
        await timer.drain();

        expect(timer.timings.size).toBe(0);
        expect(timer.takeSamples()).toEqual([]);
    });

    it('frees its buffers on dispose', async () => {
        const gpu = createFakeGpu();
        const timer = createTimer(gpu);
        await timer.ready;
        timer.dispose();
        expect(gpu.buffers.length).toBeGreaterThan(0);
        expect(gpu.buffers.every((buffer) => buffer.destroyed)).toBe(true);
    });

    it('stays quiet when a readback ends after dispose, whether it succeeds or is aborted', async () => {
        const warn = vi.fn();
        const gpu = createFakeGpu({ manualMaps: true });
        const timer = createTimer(gpu, { warn });
        await timer.ready;
        runFrame(timer, gpu, ['a']);
        runFrame(timer, gpu, ['b']);

        timer.dispose();
        gpu.deferredMaps[0].resolve();
        gpu.deferredMaps[1].reject(new Error('Buffer destroyed while mapping'));
        await flush();

        expect(warn).not.toHaveBeenCalled();
        expect(timer.timings.size).toBe(0);
        expect(timer.takeSamples()).toEqual([]);
    });
});
