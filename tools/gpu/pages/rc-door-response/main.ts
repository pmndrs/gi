import { assert, reportGpuTest, summarizeAdapter } from '../../client.ts';
import { EXAMPLES_SITE_BASE, type AdapterSummary } from '../../contract.ts';
import { waitForRailControl } from '../rc-rail/frame.ts';

interface ReceiverSample { luma: number; elapsedMs: number; doorOpenness: number }
interface DoorCapture { frame: number; renderedFrames: number; linearReceiver: ReceiverSample[] }
interface DoorControl {
    render(options: {
        frames: number; frame: number; advance: boolean; camera: 'receiver'; view: 'indirect';
        door: 'open' | 'sequence'; linearReceiver: true;
    }): Promise<DoorCapture>;
}
// Each arm receives the identical simulation ticks. Measure both transitions
// after the travel ends; a response that arrives during travel takes zero hold frames.
const warmFrames = 160;
const firstTick = 120;
const closedTick = 150;
const reopenTick = 390;
const openedTick = 420;
const finalTick = 659;
const holdWindow = 32;
const sustain = 8;
const limits = { minimumOpenLuma: 0.002, closedToOpen: 0.18, recoveredMin: 0.65, recoveredMax: 1.35, t90HoldFrames: 180 };
const progress = document.querySelector('output')!;

function median(samples: ReceiverSample[]) {
    const sorted = samples.map(sample => sample.luma).sort((a, b) => a - b);
    return (sorted[Math.floor((sorted.length - 1) / 2)] + sorted[Math.floor(sorted.length / 2)]) / 2;
}

function crossing(samples: ReceiverSample[], endOfTravel: number, lastTick: number, start: number, target: number, fraction: number) {
    const origin = endOfTravel - firstTick;
    const luminousChange = target - start;
    if (Math.abs(luminousChange) <= 1e-8) return null;
    for (let i = origin; i <= lastTick - firstTick - sustain + 1; i++) {
        let consecutive = true;
        for (let n = i; n < i + sustain; n++) {
            if ((samples[n].luma - start) / luminousChange < fraction) {
                consecutive = false;
                break;
            }
        }
        if (consecutive) return { tick: firstTick + i, framesAfterTravel: i - origin, msAfterTravel: samples[i].elapsedMs - samples[origin].elapsedMs };
    }
    return null;
}

function validateSamples(samples: ReceiverSample[], expectedCount: number) {
    assert(samples?.length === expectedCount, `expected ${expectedCount} scene-linear samples, got ${samples?.length ?? 0}`);
    let lastTime = -Infinity;
    for (const sample of samples) {
        assert(Number.isFinite(sample.luma) && sample.luma >= 0, 'linear receiver sample is negative or non-finite');
        assert(Number.isFinite(sample.elapsedMs) && sample.elapsedMs >= lastTime, 'sample wall times must increase');
        assert(sample.doorOpenness >= 0 && sample.doorOpenness <= 1, 'invalid shutter travel');
        lastTime = sample.elapsedMs;
    }
}

async function measure(backend: string) {
    const iframe = document.createElement('iframe');
    iframe.style.cssText = 'display:block;border:0;width:320px;height:180px';
    const url = new URL(`${EXAMPLES_SITE_BASE}11-radiance-cascades/`, location.origin);
    url.search = new URLSearchParams({ scene: 'door', camera: 'receiver', resolve: 'native', backend, verify: '1', stride: '2' }).toString();
    iframe.src = url.href;
    document.body.append(iframe);
    try {
        const control = await waitForRailControl(iframe) as unknown as DoorControl;
        const fixed = { camera: 'receiver', view: 'indirect', linearReceiver: true } as const;
        const warm = await control.render({ ...fixed, frame: firstTick - 1, frames: warmFrames, advance: false, door: 'open' });
        validateSamples(warm.linearReceiver, warmFrames);
        const length = finalTick - firstTick + 1;
        const run = await control.render({ ...fixed, frame: firstTick, frames: length, advance: true, door: 'sequence' });
        const samples = run.linearReceiver;
        validateSamples(samples, length);
        assert(run.frame === finalTick && run.renderedFrames - warm.renderedFrames === length, 'draw count or final simulation tick differs');
        const travel = (tick: number) => samples[tick - firstTick].doorOpenness;
        assert(travel(firstTick) === 1 && travel(closedTick) === 0 && travel(reopenTick) === 0 && travel(openedTick) === 1, 'shutter missed a travel endpoint');
        assert(Math.abs(travel(135) - 0.5) < 1e-6 && Math.abs(travel(405) - 0.5) < 1e-6, 'shutter did not translate through the half-open position');
        const open = median(warm.linearReceiver.slice(-holdWindow));
        const closed = median(samples.slice(reopenTick - firstTick - holdWindow, reopenTick - firstTick));
        const reopened = median(samples.slice(-holdWindow));
        const close50 = crossing(samples, closedTick, reopenTick - 1, open, closed, 0.5);
        const close90 = crossing(samples, closedTick, reopenTick - 1, open, closed, 0.9);
        const reopen50 = crossing(samples, openedTick, finalTick, closed, open, 0.5);
        const reopen90 = crossing(samples, openedTick, finalTick, closed, open, 0.9);
        const closedToOpen = open > 0 ? closed / open : null;
        const reopenedToOpen = open > 0 ? reopened / open : null;
        const failures = [];
        if (open <= limits.minimumOpenLuma) failures.push('insufficient open-state indirect signal');
        if (closedToOpen === null || closedToOpen > limits.closedToOpen) failures.push('excess light reaches the sealed receiver');
        if (reopenedToOpen === null || reopenedToOpen < limits.recoveredMin || reopenedToOpen > limits.recoveredMax) failures.push('reopened receiver misses its initial light level');
        if (!close90 || close90.framesAfterTravel > limits.t90HoldFrames) failures.push('closing response exceeds deadline');
        if (!reopen90 || reopen90.framesAfterTravel > limits.t90HoldFrames) failures.push('reopening response exceeds deadline');
        return {
            backend, open, closed, reopened, closedToOpen, reopenedToOpen,
            response: { close50, close90, reopen50, reopen90 }, failures,
            // Index i is simulation tick firstTick+i; keep timing and openness alongside the actual measured light.
            luma: samples.map(s => s.luma), elapsedMs: samples.map(s => s.elapsedMs), openness: samples.map(s => s.doorOpenness),
        };
    } finally {
        iframe.remove();
    }
}

const startTime = performance.now();
const arms: Awaited<ReturnType<typeof measure>>[] = [];
const data = { firstTick, closedTick, reopenTick, openedTick, finalTick, warmFrames, holdWindow, sustain, limits, arms };
let adapter: AdapterSummary | null = null;
try {
    const gpuAdapter = await navigator.gpu?.requestAdapter({ powerPreference: 'high-performance' });
    if (gpuAdapter) adapter = summarizeAdapter(gpuAdapter);
    for (const backend of ['three', 'interval']) {
        progress.textContent = `Measuring the shutter on ${backend}…`;
        arms.push(await measure(backend));
    }
    const failures = arms.flatMap(arm => arm.failures.map(failure => `${arm.backend}: ${failure}`));
    assert(failures.length === 0, failures.join(' | '));
    progress.remove();
    reportGpuTest({ status: 'pass', startTime, adapter, data });
} catch (error) {
    progress.remove();
    reportGpuTest({ status: 'fail', startTime, adapter, data, message: error instanceof Error ? error.message : String(error) });
}
