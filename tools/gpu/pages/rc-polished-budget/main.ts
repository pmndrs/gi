import { assert, reportGpuTest, summarizeAdapter } from '../../client.ts';
import { EXAMPLES_SITE_BASE, type AdapterSummary } from '../../contract.ts';
import { waitForRailControl, type RailCapture, type RailControl } from '../rc-rail/frame.ts';

type State = 'on' | 'off';
type Capture = RailCapture & { reflection: { gathered: boolean; enabled: boolean; cameraPosition: number[] };
    measurement?: NonNullable<RailCapture['measurement']> & { threeTimestampPools: number } };
type Control = { render(options: Parameters<RailControl['render']>[0]): Promise<Capture> };
const width = 960, height = 540, startingTick = 80, warmFrames = 48, measuredFrames = 60;
const pair = Number(import.meta.env.VITE_RC_REFLECTION_PAIR ?? 1);
const order: State[] = pair === 2 ? ['off', 'on'] : ['on', 'off'];
const progress = document.querySelector('output')!;

function timing(samples: number[], expected: number) {
    assert(samples.length === expected && samples.every(t => Number.isFinite(t) && t > 0), 'incomplete timing window');
    const sorted = [...samples].sort((a,b) => a-b);
    const midpoint = Math.floor(sorted.length / 2);
    return { count: samples.length, medianMs: (sorted[(sorted.length - 1) >> 1] + sorted[midpoint]) / 2,
        p95Ms: sorted[Math.ceil(0.95 * sorted.length) - 1], samplesMs: samples };
}

function floorDelta(a: Capture, b: Capture) {
    // Central floor rectangle in the final lit image, after temporal reconstruction and display transfer.
    let rgb = 0, brighter = 0, count = 0;
    for (let y = Math.floor(height*0.63); y < Math.floor(height*0.98); y++) {
        for (let x = Math.floor(width*0.1); x < Math.floor(width*0.9); x++) {
            const i = 4 * (y * width + x);
            const dr = a.pixels[i] - b.pixels[i], dg = a.pixels[i+1] - b.pixels[i+1], db = a.pixels[i+2] - b.pixels[i+2];
            rgb += Math.abs(dr) + Math.abs(dg) + Math.abs(db);
            brighter += 0.2126 * dr + 0.7152 * dg + 0.0722 * db;
            count++;
        }
    }
    return { pixels: count, rgbMae255: rgb / (3*count), signedDisplayLuma255: brighter / count };
}

async function arm(state: State) {
    const iframe = document.createElement('iframe');
    iframe.style.cssText = 'width:960px;height:540px;border:0;display:block';
    const url = new URL(EXAMPLES_SITE_BASE + '11-radiance-cascades/', location.origin);
    url.search = new URLSearchParams({ verify:'1', backend:'interval', scene:'polished', resolve:'upscale', ratio:'1.5',
        outputSize:'preview', view:'lit', camera:'rail', roughness:'0.24', motion:'still', cue:'blue', stride:'2', specular:state }).toString();
    iframe.src = url.href;
    document.body.append(iframe);
    try {
        const control = await waitForRailControl(iframe) as Control;
        const warm = await control.render({ frame: startingTick, frames: warmFrames, advance: true });
        assert(warm.reconstruction.ready && warm.reconstruction.mode === 'upscale', 'temporal reconstruction did not warm up');
        const end = await control.render({ frames: measuredFrames, advance: true, measure: true, captureSize: 'display' });
        const m = end.measurement;
        assert(end.backend === 'interval' && m?.timestampSupported && m.stride === 2, 'native gallery/timestamp instrumentation unavailable');
        assert(m.threeTimestampPools === 0, 'native gallery allocated Three.js per-pass timestamp pools');
        assert(end.width === width && end.height === height && end.reconstruction.renderWidth === 640 && end.reconstruction.renderHeight === 360,
            'gallery display or internal size changed');
        assert(end.reflection.enabled === (state === 'on') && end.reflection.gathered === (state === 'on'), 'requested gather state was not applied');
        assert(end.frame === startingTick + warmFrames + measuredFrames - 1 && end.renderedFrames - warm.renderedFrames === measuredFrames &&
            end.reconstruction.updates - warm.reconstruction.updates === measuredFrames && end.reconstruction.historyResets === warm.reconstruction.historyResets,
            'timed rail or history mismatch: tick=' + end.frame + ', draws=' + (end.renderedFrames - warm.renderedFrames) +
            ', cascade updates=' + (end.reconstruction.updates - warm.reconstruction.updates) +
            ', resets=' + (end.reconstruction.historyResets - warm.reconstruction.historyResets));
        return { end, timing: { ...timing(m.gpuFrameMs, measuredFrames),
            animationIntervals: timing(m.frameIntervalsMs, measuredFrames - 1), cascadeUpdates: end.reconstruction.updates - warm.reconstruction.updates,
            threeTimestampPools: m.threeTimestampPools } };
    } finally {
        const unloaded = new Promise(resolve => iframe.addEventListener('load', resolve, { once: true }));
        iframe.src = 'about:blank';
        await unloaded;
        iframe.remove();
    }
}

const startTime = performance.now();
let adapter: AdapterSummary | null = null;
const data: Record<string, unknown> = { pair, order, resolution: [width,height], internal: [640,360], ratio: 1.5,
    warmFrames, measuredFrames, measuredTicks: [startingTick + warmFrames, startingTick + warmFrames + measuredFrames - 1],
    pacing: 'one draw per requestAnimationFrame; runner requests uncapped Chrome scheduling', scope: 'GPU queue span from before G-buffer to after temporal resolve and canvas blit' };
try {
    assert(Number.isInteger(pair) && pair >= 1 && pair <= 3, 'VITE_RC_REFLECTION_PAIR must be 1, 2 or 3');
    const gpu = await navigator.gpu?.requestAdapter({ powerPreference: 'high-performance' });
    if (gpu) adapter = summarizeAdapter(gpu);
    const captures = new Map<State, Capture>();
    for (const state of order) {
        progress.textContent = 'Pair ' + pair + ', reflections ' + state;
        const sample = await arm(state);
        captures.set(state, sample.end);
        data[state] = sample.timing;
    }
    const on = captures.get('on')!, off = captures.get('off')!;
    assert(on.reflection.cameraPosition.join() === off.reflection.cameraPosition.join(), 'final camera positions differ');
    const signal = floorDelta(on, off);
    data.floor = signal;
    assert(signal.rgbMae255 > 1 && signal.signedDisplayLuma255 > 0, 'matched lit floor has no visible reflection gain');
    progress.remove();
    reportGpuTest({ status: 'pass', adapter, startTime, data });
} catch (error) {
    progress.remove();
    reportGpuTest({ status: 'fail', adapter, startTime, data, message: error instanceof Error ? error.message : String(error) });
}
