import { assert, reportGpuTest, summarizeAdapter } from '../../client.ts';
import { EXAMPLES_SITE_BASE, type AdapterSummary } from '../../contract.ts';
import { waitForRailControl, type RailCapture } from '../rc-rail/frame.ts';

const width = 960, height = 540, warmFrames = 48;
const starts = [31, 121, 211, 301];
const ticks = starts.flatMap(start => Array.from({ length: 6 }, (_, i) => start + i));
const startTime = performance.now();
const data: Record<string, unknown> = {
    display: [width, height], scene: 'polished', camera: 'rail', cue: 'blue', motion: 'still', roughness: 0.24,
    upscalerSharpness: 0.3, warmFrames, ticks, edgeThreshold: 12, lowerCenterRoi: [0.35, 0.72, 0.65, 0.96],
};
let adapter: AdapterSummary | null = null;

async function sequence(ratio: number, showCrop = true) {
    const mode = ratio === 1 ? 'taa' : 'upscale';
    const frame = document.createElement('iframe');
    frame.style.cssText = `width:${width}px;height:${height}px;display:block;border:0`;
    document.body.append(frame);
    const query = new URLSearchParams({ verify: '1', backend: 'interval', scene: 'polished', camera: 'rail', view: 'lit',
        cue: 'blue', motion: 'still', specular: 'on', roughness: '0.24', sharpness: '0.3', resolve: mode, ratio: String(ratio) });
    const loaded = new Promise(resolve => frame.addEventListener('load', resolve, { once: true }));
    frame.src = new URL(`${EXAMPLES_SITE_BASE}11-radiance-cascades/?${query}`, location.origin).href;
    await loaded;
    try {
        const control = await waitForRailControl(frame);
        const warm = await control.render({ frame: 0, frames: warmFrames, advance: false });
        assert(warm.reconstruction.ready, `${mode} was still compiling after the fixed warmup`);
        assert(warm.reconstruction.renderWidth === width / ratio && warm.reconstruction.renderHeight === height / ratio,
            `wrong internal dimensions for ${ratio}×`);
        const images: Uint8ClampedArray[] = [];
        let previous = 0;
        for (const tick of ticks) {
            const shot = await control.render({ frames: tick - previous, advance: true, captureSize: 'display' });
            assert(shot.frame === tick && shot.reconstruction.updates - warm.reconstruction.updates === tick,
                `cache or camera lost sync at ${ratio}× tick ${tick}`);
            assert(shot.width === width && shot.height === height && shot.reconstruction.ready, 'wrong display capture');
            assert(shot.reconstruction.historyResets === warm.reconstruction.historyResets, 'history reset during the camera rail');
            images.push(new Uint8ClampedArray(shot.pixels));
            if (showCrop && tick === 121) addCrop(shot, ratio);
            previous = tick;
        }
        return images;
    } finally {
        frame.remove();
    }
}

function addCrop(frame: RailCapture, ratio: number) {
    const canvas = document.createElement('canvas');
    canvas.width = 320;
    canvas.height = 320;
    const context = canvas.getContext('2d')!;
    const source = document.createElement('canvas');
    source.width = frame.width; source.height = frame.height;
    const sourceContext = source.getContext('2d')!;
    const rgba = sourceContext.createImageData(width, height);
    rgba.data.set(frame.pixels);
    sourceContext.putImageData(rgba, 0, 0);
    context.imageSmoothingEnabled = false;
    context.drawImage(source, 371, 203, 160, 160, 0, 0, 320, 320);
    const figure = document.createElement('figure');
    figure.style.cssText = 'margin:0';
    const caption = document.createElement('figcaption');
    caption.textContent = ratio === 1 ? 'Full resolution · TRAA' : `${ratio}× temporal upscale`;
    caption.style.cssText = 'padding:6px 8px';
    figure.append(canvas, caption);
    document.querySelector('#evidence')!.append(figure);
}

function accumulators() {
    return { full: 0, edge: 0, floor: 0, temporal: 0, floorTemporal: 0, pixels: 0, edges: 0, floorPixels: 0, pairs: 0 };
}

function measure(test: Uint8ClampedArray[], ref: Uint8ClampedArray[]) {
    const a = accumulators();
    const windows = [];
    for (let k = 0; k < ticks.length; k++) {
        const hasPrevious = k > 0 && ticks[k] === ticks[k - 1] + 1;
        let frameError = 0;
        for (let y = 0; y < height; y++) {
            const inFloorBand = y >= height * 0.72 && y < height * 0.96;
            for (let x = 0; x < width; x++) {
                const i = 4 * (y * width + x);
                const floor = inFloorBand && x >= width * 0.35 && x < width * 0.65;
                let error = 0, changeError = 0, gradient = 0;
                for (let c = 0; c < 3; c++) {
                    const j = i + c;
                    const signed = test[k][j] - ref[k][j];
                    error += Math.abs(signed) / 3;
                    if (hasPrevious) changeError += Math.abs(signed - (test[k - 1][j] - ref[k - 1][j])) / 3;
                    if (x + 1 < width) gradient = Math.max(gradient, Math.abs(ref[k][j] - ref[k][j + 4]));
                    if (y + 1 < height) gradient = Math.max(gradient, Math.abs(ref[k][j] - ref[k][j + 4 * width]));
                }
                frameError += error;
                if (gradient >= 12) { a.edge += error; a.edges++; }
                if (floor) { a.floor += error; a.floorTemporal += changeError; a.floorPixels++; }
                a.temporal += changeError;
            }
        }
        a.full += frameError;
        a.pixels += width * height;
        if (hasPrevious) a.pairs++;
        windows.push(frameError / (width * height));
    }
    return { allRgbMae255: a.full / a.pixels, referenceEdgeRgbMae255: a.edge / a.edges, floorRgbMae255: a.floor / a.floorPixels,
        adjacentChangeRgbMae255: a.temporal / (width * height * a.pairs), floorAdjacentChangeRgbMae255: a.floorTemporal / (a.floorPixels * a.pairs / ticks.length),
        edgeFraction: a.edges / a.pixels, consecutivePairs: a.pairs, frameRgbMae255: windows };
}

try {
    const gpu = await navigator.gpu?.requestAdapter();
    if (gpu) adapter = summarizeAdapter(gpu);
    const reference = await sequence(1);
    const repeat = measure(await sequence(1, false), reference);
    data.repeatReference = repeat;
    // A sub-code-value tolerance allows display quantization while identifying
    // mismatched camera, light or history before interpreting scale differences.
    assert(repeat.allRgbMae255 <= 0.5 && repeat.referenceEdgeRgbMae255 <= 1 && repeat.adjacentChangeRgbMae255 <= 1,
        'full-resolution TRAA repeat exceeded the 8-bit RGB repeatability tolerance');
    for (const ratio of [1.5, 2]) {
        const result = measure(await sequence(ratio), reference);
        assert(Number.isFinite(result.referenceEdgeRgbMae255) && result.edgeFraction > 0.005,
            'reference image contained insufficient visible edges');
        data[`ratio${ratio}`] = result;
    }
    reportGpuTest({ status: 'pass', data, adapter, startTime });
} catch (error) {
    reportGpuTest({ status: 'fail', message: error instanceof Error ? error.message : String(error), data, adapter, startTime });
}
