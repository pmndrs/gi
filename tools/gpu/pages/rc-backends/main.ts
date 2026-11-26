import { assert, reportGpuTest, summarizeAdapter } from '../../client.ts';
import { EXAMPLES_SITE_BASE, type AdapterSummary } from '../../contract.ts';
import { imageMean, waitForRailControl, type RailCapture } from '../rc-rail/frame.ts';

type Backend = 'three' | 'interval';
type Views = Record<'lit' | 'direct' | 'indirect', RailCapture>;
// The full gallery preview increases GPU load on a vsync-paced host.
const preview = import.meta.env.VITE_RC_AB_SIZE === '960x540';
const width = preview ? 960 : 640, height = preview ? 540 : 360;
const warmFrames = 48, measuredFrames = 24;
const starts = [112, 288];
const startTime = performance.now();
const data: Record<string, unknown> = { width, height, warmFrames, measuredFrames, starts, resolve: 'native', stride: 2 };
let adapter: AdapterSummary | null = null;

function timing(values: number[]) {
    const sorted = values.filter(v => Number.isFinite(v) && v > 0).sort((a, b) => a - b);
    const n = sorted.length;
    if (!n) return { samples: 0, medianMs: null, p95Ms: null };
    return { samples: n, medianMs: (sorted[(n - 1) >> 1] + sorted[n >> 1]) / 2, p95Ms: sorted[Math.ceil(0.95 * n) - 1] };
}

function difference(a: RailCapture, b: RailCapture) {
    assert(a.frame === b.frame && a.width === b.width && a.height === b.height, 'image comparison does not have matching ticks and dimensions');
    const histogram = new Uint32Array(256);
    let sum = 0, squareSum = 0;
    for (let i = 0; i < a.pixels.length; i += 4) {
        for (let c = i; c < i + 3; c++) {
            const delta = Math.abs(a.pixels[c] - b.pixels[c]);
            sum += delta;
            squareSum += delta * delta;
            histogram[delta]++;
        }
    }
    const n = a.width * a.height * 3;
    let p95 = 0, count = histogram[0];
    while (count < 0.95 * n) count += histogram[++p95];
    return { rgbMae255: sum / n, rgbRmse255: Math.sqrt(squareSum / n), rgbP95Abs255: p95 };
}

async function captureBackend(backend: Backend): Promise<Views[]> {
    const frame = document.createElement('iframe');
    frame.style.cssText = `width:${width}px;height:${height}px;border:0;display:block;`;
    frame.src = new URL(`${EXAMPLES_SITE_BASE}11-radiance-cascades/?verify=1&scene=gallery&backend=${backend}&resolve=native&stride=2`, location.origin).href;
    document.body.append(frame);
    const captures: Views[] = [];
    const statistics: Record<string, unknown>[] = [];
    data[backend] = statistics;
    try {
        const control = await waitForRailControl(frame);
        const pose = { camera: 'rail', cue: 'sequence', captureSize: 'display' } as const;
        for (const firstTick of starts) {
            const warm = await control.render({ ...pose, frame: firstTick, frames: warmFrames, view: 'lit', advance: true });
            assert(warm.backend === backend && warm.reconstruction.mode === 'native', `requested ${backend}/native, got ${warm.backend}/${warm.reconstruction.mode}`);
            const lit = await control.render({ ...pose, frames: measuredFrames, view: 'lit', advance: true, measure: true });
            const lastTick = firstTick + warmFrames + measuredFrames - 1;
            assert(lit.frame === lastTick, `${backend} measured to tick ${lit.frame}, expected ${lastTick}`);
            assert(lit.reconstruction.updates - warm.reconstruction.updates === measuredFrames, `${backend} did not update once per measured draw`);
            assert(lit.measurement, `${backend} returned no timing data`);
            const m = lit.measurement;
            assert(m.renderWidth === width && m.renderHeight === height, `${backend} ran at ${m.renderWidth}×${m.renderHeight}`);
            const wholeFrame = timing(m.gpuFrameMs);
            if (m.timestampSupported) assert(wholeFrame.samples > 0, `${backend} returned no whole-frame GPU timestamps`);
            const direct = await control.render({ ...pose, frames: 1, view: 'direct', advance: false });
            const indirect = await control.render({ ...pose, frames: 1, view: 'indirect', advance: false });
            assert(lit.width === width && lit.height === height, 'A/B requires full-size presented pixels');
            const means = { lit: imageMean(lit).luma, direct: imageMean(direct).luma, indirect: imageMean(indirect).luma };
            const gained = difference(lit, direct);
            statistics.push({ tick: lastTick, timing: wholeFrame, rafIntervals: timing(m.frameIntervalsMs), displayLuma: means, litVsDirect: gained,
                timestampSupported: m.timestampSupported, stride: m.stride, probeBytes: m.probeBytes, hitScratchBytes: m.hitScratchBytes });
            assert(means.indirect > 1 / 255 && gained.rgbMae255 > 1, `${backend} shows no appreciable indirect illumination at tick ${lastTick}`);
            captures.push({ lit, direct, indirect });
        }
    } finally {
        const unloaded = new Promise(resolve => frame.addEventListener('load', resolve, { once: true }));
        frame.src = 'about:blank';
        await unloaded;
        frame.remove();
    }
    return captures;
}

try {
    const gpuAdapter = await navigator.gpu?.requestAdapter();
    if (gpuAdapter) adapter = summarizeAdapter(gpuAdapter);
    const reference = await captureBackend('three');
    const interval = await captureBackend('interval');
    const pairs = reference.map((r, i) => ({
        tick: r.lit.frame, direct: difference(r.direct, interval[i].direct),
        lit: difference(r.lit, interval[i].lit), indirect: difference(r.indirect, interval[i].indirect),
    }));
    data.presentedRgbDifferences = pairs;
    for (const pair of pairs) {
        assert(pair.direct.rgbMae255 <= 1 && pair.direct.rgbP95Abs255 <= 3,
            `direct-only inputs disagree at tick ${pair.tick}: MAE ${pair.direct.rgbMae255}, p95 ${pair.direct.rgbP95Abs255}`);
    }
    reportGpuTest({ status: 'pass', data, adapter, startTime });
} catch (error) {
    reportGpuTest({ status: 'fail', message: error instanceof Error ? error.message : String(error), data, adapter, startTime });
}
