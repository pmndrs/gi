import { assert, reportGpuTest, summarizeAdapter } from '../../client.ts';
import { EXAMPLES_SITE_BASE, type AdapterSummary } from '../../contract.ts';
import { imageMean, waitForRailControl, type RailCapture, type ReconstructionState } from '../rc-rail/frame.ts';

const startTime = performance.now();
const data: Record<string, unknown> = {};
let adapter: AdapterSummary | null = null;
const frame = document.createElement('iframe');
frame.style.cssText = 'width:481px;height:271px;border:0;display:block;';
frame.src = new URL(`${EXAMPLES_SITE_BASE}11-radiance-cascades/?verify=1&scene=gallery&backend=three&resolve=native&ratio=2`, location.origin).href;
document.body.append(frame);

function controlValue(id: string, value: string, event = 'change') {
    const input = frame.contentDocument?.querySelector<HTMLInputElement | HTMLSelectElement>(`#${id}`);
    assert(input, `missing #${id}`);
    input.value = value;
    assert(input.value === value, `#${id} rejected ${value}`);
    input.dispatchEvent(new Event(event, { bubbles: true }));
}

function checkSize(s: ReconstructionState, w: number, h: number, ratio: number) {
    assert(s.displayWidth === w && s.displayHeight === h, `display is ${s.displayWidth}×${s.displayHeight}, expected ${w}×${h}`);
    assert(s.renderWidth === Math.floor(w / ratio) && s.renderHeight === Math.floor(h / ratio),
        `G-buffer is ${s.renderWidth}×${s.renderHeight}, expected floor(${w}×${h}/${ratio})`);
}

/** Display-space MAE and edge MAE in 8-bit RGB code values against the warmed native TRAA image. */
function compareEdges(sample: RailCapture, reference: RailCapture) {
    const { width, height, pixels } = reference;
    const input = document.createElement('canvas');
    input.width = sample.width;
    input.height = sample.height;
    const source = input.getContext('2d')!;
    const rgb = source.createImageData(input.width, input.height);
    rgb.data.set(sample.pixels);
    source.putImageData(rgb, 0, 0);
    const output = document.createElement('canvas');
    output.width = width;
    output.height = height;
    const context = output.getContext('2d')!;
    // Canvas's low quality interpolation selects bilinear sampling for the raw half-size baseline.
    context.imageSmoothingEnabled = true;
    context.imageSmoothingQuality = 'low';
    context.drawImage(input, 0, 0, width, height);
    const presented = context.getImageData(0, 0, width, height).data;
    let total = 0, edgeTotal = 0, edges = 0;
    for (let y = 0; y < height - 1; y++) {
        for (let x = 0; x < width - 1; x++) {
            const i = 4 * (y * width + x);
            let delta = 0, contrast = 0;
            for (let c = i; c < i + 3; c++) {
                delta += Math.abs(presented[c] - pixels[c]) / 3;
                contrast = Math.max(contrast, Math.abs(pixels[c] - pixels[c + 4]), Math.abs(pixels[c] - pixels[c + width * 4]));
            }
            total += delta;
            if (contrast >= 12) { edgeTotal += delta; edges++; }
        }
    }
    assert(edges > 0, 'native TRAA reference contains no measurable edges');
    return { meanRgbError: total / ((width - 1) * (height - 1)), edgeRgbError: edgeTotal / edges, edges };
}

try {
    const gpuAdapter = await navigator.gpu?.requestAdapter();
    if (gpuAdapter) adapter = summarizeAdapter(gpuAdapter);
    const control = await waitForRailControl(frame);
    async function render(frames: number, camera: 'receiver' | 'rail' = 'receiver') {
        const capture = await control.render({ frames, camera, view: 'lit', cue: 'blue', frame: 160, advance: false });
        assert(capture.reconstruction, 'the gallery did not report reconstruction diagnostics');
        assert(capture.frame === 160, `fixed capture advanced to tick ${capture.frame}`);
        return capture;
    }
    const modes: Record<string, unknown> = {};
    data.modes = modes;
    for (const mode of ['native', 'taa', 'upscale'] as const) {
        controlValue('resolve', mode);
        let warm = await render(16);
        const deadline = performance.now() + 120_000;
        while (!warm.reconstruction.ready && performance.now() < deadline) warm = await render(2);
        assert(warm.reconstruction.mode === mode && warm.reconstruction.ready, `${mode} did not become ready`);
        const settled = await render(12);
        const a = settled.reconstruction;
        const next = await render(1);
        const b = next.reconstruction;
        const ticks = b.updates - warm.reconstruction.updates;
        const luma = imageMean(next, true).luma;
        modes[mode] = { ...b, firstJitter: [a.jitterX, a.jitterY], updatesIn13Draws: ticks, receiverLuma: luma };
        checkSize(b, 481, 271, mode === 'upscale' ? 2 : 1);
        assert(ticks === 13, `${mode} performed ${ticks} RC updates in 13 presentations`);
        assert(luma > 2 / 255, `${mode} presented a dark receiver (${luma})`);
        if (mode === 'native') {
            assert(a.jitterX === 0 && a.jitterY === 0 && b.jitterX === 0 && b.jitterY === 0, 'native G-buffer was jittered');
        } else {
            const change = Math.hypot(b.jitterX - a.jitterX, b.jitterY - a.jitterY);
            assert(change > 0.01 && change < 2, `${mode} G-buffer missed subpixel camera jitter (phase delta ${change})`);
            const cut = await render(1, 'rail');
            assert(cut.reconstruction.historyResets > b.historyResets, `${mode} kept presentation history across the camera cut`);
        }
    }
    frame.style.width = '319px';
    frame.style.height = '181px';
    controlValue('sharpness', '0.8', 'input');
    const resized = await render(24);
    checkSize(resized.reconstruction, 319, 181, 2);
    const resizedLuma = imageMean(resized, true).luma;
    data.resize = { ...resized.reconstruction, receiverLuma: resizedLuma };
    assert(resized.reconstruction.ready && resizedLuma > 2 / 255, 'resized upscaler did not present the lit receiver');
    controlValue('resolve', 'native');
    const last = await render(2);
    checkSize(last.reconstruction, 319, 181, 1);
    assert(last.reconstruction.jitterX === 0 && last.reconstruction.jitterY === 0, 'upscaler left camera jitter on native');

    // Isolate reconstruction from noisy indirect light. All captures use the same
    // fixed rail pose and aspect ratio; only render density and resolve change.
    frame.style.width = '480px';
    frame.style.height = '270px';
    const options = { view: 'direct', camera: 'rail', cue: 'amber', frame: 80, advance: false, captureSize: 'display' } as const;
    const images = new Map<string, RailCapture>();
    controlValue('sharpness', '0', 'input');
    for (const mode of ['native', 'taa', 'upscale'] as const) {
        controlValue('resolve', mode);
        await control.render({ ...options, frames: 8 });
        let capture = await control.render({ ...options, frames: 32 });
        const deadline = performance.now() + 120_000;
        while (!capture.reconstruction.ready && performance.now() < deadline) capture = await control.render({ ...options, frames: 4 });
        assert(capture.reconstruction.ready, `${mode} did not settle for AA comparison`);
        capture = await control.render({ ...options, frames: 32 });
        assert(capture.width === 480 && capture.height === 270, 'AA diagnostic needs full display readback');
        images.set(mode, capture);
    }
    controlValue('resolve', 'native');
    frame.style.width = '240px';
    frame.style.height = '135px';
    const low = await control.render({ ...options, frames: 16 });
    assert(low.width === 240 && low.height === 135, 'half-resolution baseline was not captured at source resolution');
    const reference = images.get('taa')!;
    data.stationaryDirectVsNativeTraa = {
        width: reference.width, height: reference.height, tick: reference.frame, sharpen: 0,
        nativeUnfiltered: compareEdges(images.get('native')!, reference),
        halfSizeBilinear: compareEdges(low, reference),
        halfSizeTemporalUpscaler: compareEdges(images.get('upscale')!, reference),
    };
    // Verify that the independent interval producer can supply the same depth
    // and velocity interface to the temporal upscaler at a non-integer ratio.
    const loaded = new Promise(resolve => frame.addEventListener('load', resolve, { once: true }));
    frame.style.width = '481px';
    frame.style.height = '271px';
    frame.src = new URL(`${EXAMPLES_SITE_BASE}11-radiance-cascades/?verify=1&scene=gallery&backend=interval&resolve=upscale&ratio=1.5`, location.origin).href;
    await loaded;
    const interval = await waitForRailControl(frame);
    const request = { frames: 8, camera: 'receiver', view: 'lit', cue: 'blue', advance: false } as const;
    let intervalImage = await interval.render(request);
    const deadline = performance.now() + 120_000;
    while (!intervalImage.reconstruction.ready && performance.now() < deadline) intervalImage = await interval.render(request);
    assert(intervalImage.backend === 'interval' && intervalImage.reconstruction.ready, 'interval backend never presented a ready temporal upscale');
    const settled = await interval.render({ ...request, frames: 24 });
    const intervalState = settled.reconstruction;
    const intervalLuma = imageMean(settled, true).luma;
    data.intervalUpscale = { ...intervalState, receiverLuma: intervalLuma };
    checkSize(intervalState, 481, 271, 1.5);
    assert(intervalState.updates - intervalImage.reconstruction.updates === 24, 'interval producer did not update once per upscaled draw');
    assert(intervalLuma > 2 / 255 && Math.hypot(intervalState.jitterX, intervalState.jitterY) > 0,
        'interval backend produced a dark or unjittered temporal upscale');
    reportGpuTest({ status: 'pass', data, adapter, startTime });
} catch (error) {
    reportGpuTest({ status: 'fail', message: error instanceof Error ? error.message : String(error), data, adapter, startTime });
}
