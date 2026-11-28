import { assert, reportGpuTest, summarizeAdapter } from '../../client.ts';
import { EXAMPLES_SITE_BASE, type AdapterSummary } from '../../contract.ts';
import { imageMean, waitForRailControl, type RailCapture } from '../rc-rail/frame.ts';

// Match the interactive demo's maximum drawing size. The small run measures
// the same poses and probe spacing with a quarter of the G-buffer pixels.
const small = import.meta.env.VITE_RC_BUDGET_SIZE === '480x270';
const width = small ? 480 : 960;
const height = small ? 270 : 540;
const warmupFrames = 48;
const measuredFrames = 60;
const receiverFrames = 64;
const startTick = 172;
const progress = document.querySelector('output')!;

function distribution(samples: number[]) {
    if (samples.length === 0) return { count: 0, median: null, p95: null };
    const sorted = [...samples].sort((a, b) => a - b);
    const mid = Math.floor(sorted.length / 2);
    const median = sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
    return { count: sorted.length, median, p95: sorted[Math.ceil(sorted.length * 0.95) - 1] };
}

function receiverError(sparse: RailCapture, dense: RailCapture) {
    assert(sparse.width === dense.width && sparse.height === dense.height, 'receiver capture dimensions differ');
    // Center third in each dimension, as in the rail receiver check. Values are displayed sRGB after tone mapping.
    const xInset = Math.floor(dense.width / 3);
    const yInset = Math.floor(dense.height / 3);
    let absolute = 0;
    let squared = 0;
    let samples = 0;
    for (let y = yInset; y < dense.height - yInset; y++) {
        for (let x = xInset; x < dense.width - xInset; x++) {
            const offset = (y * dense.width + x) * 4;
            for (let channel = 0; channel < 3; channel++) {
                const diff = (sparse.pixels[offset + channel] - dense.pixels[offset + channel]) / 255;
                absolute += Math.abs(diff);
                squared += diff * diff;
                samples++;
            }
        }
    }
    return { samples, meanAbsolute: absolute / samples, rootMeanSquare: Math.sqrt(squared / samples) };
}

async function runArm(stride: number) {
    // A new browser document per arm resets probe history, low-discrepancy sequence and GPU allocations.
    const iframe = document.createElement('iframe');
    iframe.style.cssText = `display:block;border:0;width:${width}px;height:${height}px`;
    const url = new URL(`${EXAMPLES_SITE_BASE}cascades/11-radiance-cascades/`, location.origin);
    url.searchParams.set('verify', '1');
    url.searchParams.set('stride', String(stride));
    iframe.src = url.href;
    document.body.append(iframe);
    try {
        const rail = await waitForRailControl(iframe);
        const warm = await rail.render({ frame: startTick, frames: warmupFrames, advance: true, camera: 'rail', view: 'lit', cue: 'sequence' });
        const measured = await rail.render({ frames: measuredFrames, advance: true, measure: true });
        assert(measured.frame === startTick + warmupFrames + measuredFrames - 1, 'timed window ended on the wrong tick');
        assert(measured.renderedFrames - warm.renderedFrames === measuredFrames, 'timed draw count differs from requested frames');
        const m = measured.measurement;
        assert(m && m.stride === stride && m.renderWidth === width && m.renderHeight === height, 'ray stride or internal resolution changed');
        const rayLanes = Math.ceil(width / stride) * Math.ceil(height / stride);
        const expectedScratch = Math.ceil(rayLanes * 1.25) * 32;
        assert(m.hitScratchBytes === expectedScratch, `hit scratch uses ${m.hitScratchBytes} bytes for ${rayLanes} ray lanes`);
        assert(m.frameIntervalsMs.length === measuredFrames - 1, 'incomplete frame-interval window');
        if (m.timestampSupported) {
            assert(m.gpuComputeMs.length === measuredFrames, `received ${m.gpuComputeMs.length}/${measuredFrames} distinct RC GPU timestamps`);
        }
        const receiver = await rail.render({ frame: 0, frames: receiverFrames, advance: false, camera: 'receiver', view: 'indirect', cue: 'blue' });
        // A bright average can hide black cells when the probe pools fill. Read counters
        // outside the timed window, after the same receiver exposure in both arms.
        const { occupancy } = await rail.render({ frames: 1, advance: false, occupancy: true });
        assert(occupancy && occupancy.capacities.length === 4 && occupancy.peakProbes.length === 4,
            'missing cascade occupancy');
        assert(occupancy.peakProbes.every((n, i) => n > 0 && n < occupancy.capacities[i]),
            `receiver probe pool saturated or empty: ${occupancy.peakProbes} / ${occupancy.capacities}`);
        const receiverMean = imageMean(receiver, true);
        assert(receiverMean.luma > 2 / 255, 'the fixed receiver has no usable indirect signal');
        return {
            receiver, stats: {
                stride, renderWidth: m.renderWidth, renderHeight: m.renderHeight, timestampSupported: m.timestampSupported,
                gpuComputeMs: distribution(m.gpuComputeMs), frameIntervalMs: distribution(m.frameIntervalsMs),
                probeBytes: m.probeBytes, hitScratchBytes: m.hitScratchBytes, occupancy, receiverMean,
            },
        };
    } finally {
        iframe.remove();
    }
}

const startTime = performance.now();
let adapter: AdapterSummary | null = null;
const pairs: Record<string, unknown>[] = [];
const data = {
    rail: { width, height, warmupFrames, measuredFrames, warmupStartTick: startTick, measuredTicks: [220, 279] },
    receiver: { pose: 'receiver', cue: 'blue', view: 'indirect', settleFrames: receiverFrames, sample: 'center third of 64×36 presented image', units: 'normalized display RGB', reference: 'stride 1 under the same replay and frame budget' },
    pairs,
};
try {
    const gpuAdapter = await navigator.gpu?.requestAdapter({ powerPreference: 'high-performance' });
    if (gpuAdapter) adapter = summarizeAdapter(gpuAdapter);
    // On thermally limited hardware each pair can run under its own cool-start lock.
    const onlyPair = import.meta.env.VITE_RC_BUDGET_PAIR;
    assert(onlyPair === undefined || ['1', '2', '3'].includes(onlyPair), 'VITE_RC_BUDGET_PAIR must be 1, 2 or 3');
    for (const [pair, order] of [[2, 1], [1, 2], [2, 1]].entries()) {
        if (onlyPair !== undefined && Number(onlyPair) !== pair + 1) continue;
        const arms = new Map<number, Awaited<ReturnType<typeof runArm>>>();
        for (const stride of order) {
            progress.textContent = `Pair ${pair + 1} / 3 · ray stride ${stride}`;
            arms.set(stride, await runArm(stride));
        }
        const sparse = arms.get(2)!;
        const dense = arms.get(1)!;
        const sparseMs = sparse.stats.gpuComputeMs.median;
        const denseMs = dense.stats.gpuComputeMs.median;
        pairs.push({
            pair: pair + 1, order, stride2: sparse.stats, stride1: dense.stats,
            medianComputeSavingMs: sparseMs !== null && denseMs !== null ? denseMs - sparseMs : null,
            receiverErrorVsStride1: receiverError(sparse.receiver, dense.receiver),
        });
    }
    progress.remove();
    reportGpuTest({ status: 'pass', adapter, startTime, data });
} catch (error) {
    progress.remove();
    reportGpuTest({ status: 'fail', adapter, startTime, data, message: error instanceof Error ? error.message : String(error) });
}
