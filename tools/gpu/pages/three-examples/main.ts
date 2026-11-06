/**
 * T1.14: the three.js examples (`examples/three/`) load and render, as a
 * user runs them: each example's own `main()`, imported by the published
 * name (`@pmndrs/ddgi/three`, aliased to the source by the test server).
 *
 * Per example (simple, then composite):
 * - `main()` starts GI (`volume.start`, with the warm-up) and the animation loop;
 * - the render-time update runs GI frames with no per-frame call: the
 *   platform's frame count rises while three's animation loop draws;
 * - GI shows: the canvas with GI is brighter than with the volume's
 *   intensity at 0 (the GI light follows it), and the image isn't black.
 *   The composite example is the intended pipeline: through @pmndrs/upscaler
 *   (Performance), so pictures are read inside its animation loop, after
 *   enough frames for the upscaler's history to settle.
 * The runner's screenshot shows the composite example's last frame.
 */

import { drainDevice, reportGpuTest, summarizeAdapterInfo, watchDevice } from '../../client.ts';
import type { AdapterSummary } from '../../contract.ts';
import { main as simpleMain, type ExampleHandle } from '../../../../examples/three/src/simple.ts';
import { main as compositeMain } from '../../../../examples/three/src/composite.ts';

/** Frames of three's animation loop to let run before measuring. */
const LOOP_FRAMES = 40;
/** GI must add at least this share of the canvas's brightness for the check to say GI shows. */
const MIN_GI_SHARE = 0.05;

/** Mean luminance of what the WebGPU canvas shows now (read in the same task as the render that drew it). */
function canvasLuminance(canvas: HTMLCanvasElement): number {
    const copy = document.createElement('canvas');
    copy.width = canvas.width;
    copy.height = canvas.height;
    const context = copy.getContext('2d', { willReadFrequently: true })!;
    context.drawImage(canvas, 0, 0);
    const pixels = context.getImageData(0, 0, copy.width, copy.height).data;
    let sum = 0;
    for (let index = 0; index < pixels.length; index += 4) sum += 0.2126 * pixels[index] + 0.7152 * pixels[index + 1] + 0.0722 * pixels[index + 2];
    return sum / (pixels.length / 4) / 255;
}

/** The canvas's mean luminance, read right after the animation loop's next frame (the same task). */
function captureInLoop(handle: ExampleHandle): Promise<number> {
    return new Promise((resolve) => {
        handle.renderer.setAnimationLoop(() => {
            handle.render();
            handle.renderer.setAnimationLoop(handle.render);
            resolve(canvasLuminance(handle.renderer.domElement));
        });
    });
}

/** Waits until three's animation loop has drawn `frames` more frames. */
function loopFrames(frames: number): Promise<void> {
    return new Promise((resolve) => {
        let left = frames;
        const step = () => {
            if (--left <= 0) resolve();
            else requestAnimationFrame(step);
        };
        requestAnimationFrame(step);
    });
}

interface ExampleResult {
    platformFramesDuringLoop: number;
    luminanceWithGi: number;
    luminanceWithoutGi: number;
    giShare: number;
    deviceErrors: string[];
}

async function runExample(name: string, main: (container: HTMLElement) => Promise<ExampleHandle>, log: (line: string) => void): Promise<{ result: ExampleResult; handle: ExampleHandle; adapter: AdapterSummary }> {
    const container = document.createElement('div');
    container.id = name;
    document.body.append(container);
    const started = performance.now();
    const handle = await main(container);
    const device = (handle.renderer.backend as unknown as { device: GPUDevice }).device;
    const deviceErrors = watchDevice(device);
    log(`${name}: started in ${((performance.now() - started) / 1000).toFixed(1)} s`);
    const platform = handle.volume.platform!;
    const framesBefore = platform.stats.frame;
    await loopFrames(LOOP_FRAMES);
    const platformFramesDuringLoop = platform.stats.frame - framesBefore;
    // Read each picture inside the running animation loop: three only redraws its passes (and the
    // upscaler only advances its history) on a new frame of that loop.
    const luminanceWithGi = await captureInLoop(handle);
    handle.volume.intensity = 0;
    // Enough frames for the upscaler's history to settle on the new picture.
    await loopFrames(LOOP_FRAMES);
    const luminanceWithoutGi = await captureInLoop(handle);
    handle.volume.intensity = 1;
    await drainDevice(device);
    const result: ExampleResult = {
        platformFramesDuringLoop,
        luminanceWithGi,
        luminanceWithoutGi,
        giShare: 1 - luminanceWithoutGi / luminanceWithGi,
        // A copy: disposing the example later destroys its device, which is expected, not an error.
        deviceErrors: [...deviceErrors],
    };
    log(`${name}: ${platformFramesDuringLoop} GI frames in ${LOOP_FRAMES} animation frames; canvas luminance ${luminanceWithGi.toFixed(4)} with GI, ${luminanceWithoutGi.toFixed(4)} without (GI ${(result.giShare * 100).toFixed(1)}%)`);
    return { result, handle, adapter: summarizeAdapterInfo(device.adapterInfo, device.features) };
}

const startTime = performance.now();
const logLines: string[] = [];
const data: Record<string, unknown> = {};
let adapter: AdapterSummary | null = null;
const deviceErrors: string[] = [];
try {
    const simple = await runExample('simple', simpleMain, (line) => logLines.push(line));
    data.simple = simple.result;
    adapter = simple.adapter;
    deviceErrors.push(...simple.result.deviceErrors);
    // Free the first example's GPU work before the second (NOTES: release the device between pages).
    simple.handle.volume.dispose();
    simple.handle.renderer.dispose();
    simple.handle.renderer.domElement.remove();

    const composite = await runExample('composite', compositeMain, (line) => logLines.push(line));
    data.composite = composite.result;
    deviceErrors.push(...composite.result.deviceErrors);
    // Leave the composite running for the screenshot.
    composite.handle.render();

    const problems: string[] = [];
    for (const [name, result] of Object.entries({ simple: simple.result, composite: composite.result })) {
        if (result.platformFramesDuringLoop < LOOP_FRAMES / 2) problems.push(`${name}: only ${result.platformFramesDuringLoop} GI frames ran in ${LOOP_FRAMES} animation frames (the render-time update)`);
        if (!(result.luminanceWithGi > 0.01)) problems.push(`${name}: the canvas is black`);
        if (!(result.giShare >= MIN_GI_SHARE)) problems.push(`${name}: GI adds only ${(result.giShare * 100).toFixed(1)}% of the brightness`);
    }
    reportGpuTest({ status: problems.length === 0 ? 'pass' : 'fail', message: problems.join(' | ') || undefined, data, adapter, deviceErrors, startTime, logLines });
} catch (error) {
    const message = error instanceof Error ? `${error.message}\n${error.stack ?? ''}` : String(error);
    reportGpuTest({ status: 'fail', message, data, adapter, deviceErrors, startTime, logLines });
}
