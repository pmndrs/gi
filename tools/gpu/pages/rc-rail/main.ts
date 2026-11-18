import { assert, reportGpuTest, summarizeAdapter } from '../../client.ts';
import { EXAMPLES_SITE_BASE, type AdapterSummary } from '../../contract.ts';
import { imageMean, waitForRailControl, type RailCapture } from './frame.ts';

function meanPixelDifference(a: RailCapture, b: RailCapture): number {
    assert(a.width === b.width && a.height === b.height, 'presented image dimensions changed');
    let delta = 0;
    for (let i = 0; i < a.pixels.length; i += 4) {
        for (let channel = 0; channel < 3; channel++) delta += Math.abs(a.pixels[i + channel] - b.pixels[i + channel]);
    }
    return delta / (a.width * a.height * 3 * 255);
}

const startTime = performance.now();
const data: Record<string, unknown> = {};
let adapter: AdapterSummary | null = null;
const frame = document.createElement('iframe');
frame.style.cssText = 'width:480px;height:270px;border:0;display:block;';
frame.src = new URL(`${EXAMPLES_SITE_BASE}11-radiance-cascades/?verify=1`, location.origin).href;
document.body.append(frame);
try {
    const gpuAdapter = await navigator.gpu?.requestAdapter();
    if (gpuAdapter) adapter = summarizeAdapter(gpuAdapter);
    const control = await waitForRailControl(frame);
    const first = await control.render({ frame: 140, frames: 1, camera: 'rail', view: 'direct', cue: 'amber', advance: false });
    const moved = await control.render({ frames: 32, advance: true });
    const railDelta = meanPixelDifference(first, moved);
    data.rail = { startFrame: first.frame, endFrame: moved.frame, rgbMeanAbsoluteDifference: railDelta };
    assert(moved.frame - first.frame === 32, 'the rail did not advance one tick per frame');
    assert(railDelta > 2 / 255, `camera rail images barely changed (${railDelta})`);

    const fixed = { camera: 'receiver', frame: 0, advance: false } as const;
    const direct = await control.render({ ...fixed, frames: 1, cue: 'blue', view: 'direct' });
    const lit = await control.render({ ...fixed, frames: 64, cue: 'blue', view: 'lit' });
    const indirect = await control.render({ ...fixed, frames: 1, cue: 'blue', view: 'indirect' });
    const justOff = await control.render({ ...fixed, frames: 1, cue: 'off', view: 'indirect' });
    const settledOff = await control.render({ ...fixed, frames: 64, cue: 'off', view: 'indirect' });
    const amber = await control.render({ ...fixed, frames: 64, cue: 'amber', view: 'indirect' });
    const directMean = imageMean(direct, true);
    const litMean = imageMean(lit, true);
    const blueMean = imageMean(indirect, true);
    const offMean = imageMean(settledOff, true);
    const amberMean = imageMean(amber, true);
    data.receiver = {
        direct: directMean, lit: litMean, blueIndirect: blueMean, firstFrameOff: imageMean(justOff, true),
        settledOff: offMean, amberIndirect: amberMean, settleFrames: 64, presentedRgbRange: '0..1',
    };
    assert(litMean.luma - directMean.luma > 1 / 255, 'composite adds no measurable light to the receiver');
    assert(blueMean.luma > 2 / 255, 'the indirect-only receiver is dark');
    assert(blueMean.luma - offMean.luma > 1 / 255, 'the receiver kept its indirect light after both emitters went dark');
    assert(amberMean.luma - offMean.luma > 1 / 255, 'the receiver stayed dark after the amber emitters switched on');
    const colourShift = (amberMean.red - amberMean.blue) - (blueMean.red - blueMean.blue);
    assert(colourShift > 2 / 255, `switching blue to amber did not warm the receiver (${colourShift})`);
    // The hit buffer grows with ray lanes. Odd dimensions also exercise partial cells.
    // Shrinking should preserve the larger allocation and valid secondary lighting.
    const resizeCases = [[641, 359], [319, 181]] as const;
    const sizes = [];
    let capacity = 0;
    for (const [width, height] of resizeCases) {
        frame.style.width = `${width}px`;
        frame.style.height = `${height}px`;
        const capture = await control.render({ ...fixed, frames: 5, cue: 'amber', view: 'indirect', measure: true });
        const m = capture.measurement;
        assert(m && m.renderWidth === width && m.renderHeight === height, 'RC canvas missed a viewport resize');
        const rays = Math.ceil(width / m.stride) * Math.ceil(height / m.stride);
        if (rays > capacity) capacity = Math.ceil(1.25 * rays);
        assert(m.hitScratchBytes === 32 * capacity, 'unexpected hit scratch allocation after resize');
        const luma = imageMean(capture, true).luma;
        assert(luma > 2 / 255, 'receiver lost indirect light after resize');
        sizes.push({ width, height, scratchBytes: m.hitScratchBytes, luma });
    }
    data.resize = sizes;
    reportGpuTest({ status: 'pass', data, adapter, startTime });
} catch (error) {
    reportGpuTest({ status: 'fail', message: error instanceof Error ? error.message : String(error), data, adapter, startTime });
}
