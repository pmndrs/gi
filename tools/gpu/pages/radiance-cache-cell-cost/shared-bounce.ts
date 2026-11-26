/** GPU time is the sum of the technique's pass timestamps. Calibrate the
 * shared budget against budgeted GI-1.2 at 20k, then independently measure
 * both images after the same cumulative GPU work from a cold cache. */
import type { GpuTimer } from '../../../../packages/ddgi-core/src/platform/timing/gpu-timer.ts';
import type { RadianceCacheTechnique } from '../../../../packages/ddgi-core/src/techniques/radiance-cache/radiance-cache-technique.ts';
import { accuracyNumbers, rgbOfRgba, type RgbImage } from '../radiance-cache-accuracy/accuracy.ts';
import { halves, readTexture } from '../radiance-cache-probes/readback.ts';

export async function compareSharedBounce(
    device: GPUDevice, technique: RadianceCacheTechnique, timer: GpuTimer, frame: () => void,
    width: number, height: number, changeScene?: (changed: boolean) => void,
): Promise<Record<string, unknown>> {
    const sample = async (): Promise<number> => {
        frame();
        await timer.drain();
        const frames = timer.takeSamples();
        if (frames.length !== 1) throw new Error('expected one GPU timestamp sample');
        const ms = frames[0].passes.reduce((sum, pass) => sum + pass.gpuMs, 0);
        if (!(ms > 0)) throw new Error('no GPU time measured');
        return ms;
    };
    await timer.drain();
    timer.takeSamples();
    const batch = async (count: number): Promise<number> => {
        let sum = 0;
        for (let i = 0; i < count; i++) sum += await sample();
        return sum / count;
    };
    const image = async (): Promise<RgbImage> => {
        const texture = technique.probes?.denoisedIrradiance;
        if (!texture) throw new Error('missing denoised irradiance');
        const values = halves(await readTexture(device, texture, 8, width, height));
        if (!values.every(Number.isFinite)) throw new Error('nonfinite irradiance');
        return rgbOfRgba(values, width, height);
    };
    const reference = async (): Promise<RgbImage> => {
        technique.settings.cellShading = 'per-hit';
        technique.settings.secondBounce = 'gi1-rays';
        technique.resetHistory();
        await batch(96);
        const sum = new Float32Array(width * height * 3);
        for (let i = 0; i < 24; i++) {
            await sample();
            const current = await image();
            for (let j = 0; j < sum.length; j++) sum[j] += current.rgb[j] / 24;
        }
        if (!sum.some(value => value > 0)) throw new Error('black reference');
        return { width, height, rgb: sum };
    };
    const before = await reference();
    let after: RgbImage | undefined;
    if (changeScene) {
        changeScene(true);
        after = await reference();
        changeScene(false);
    }
    technique.settings.cellShading = 'budgeted';
    const baselineBudget = 20000;
    const calibration: { mode: string; budget: number; ms: number }[] = [];
    const calibrate = async (mode: 'gi1-rays' | 'shared-cell', budget: number): Promise<number> => {
        technique.settings.secondBounce = mode;
        technique.settings.cellShadingBudget = budget;
        technique.resetHistory();
        await batch(24);
        const ms = await batch(20);
        calibration.push({ mode, budget, ms });
        return ms;
    };
    const target = await calibrate('gi1-rays', baselineBudget);
    let bestBudget = baselineBudget;
    let gap = Infinity;
    let low = 0, high = 160000, budget = baselineBudget;
    for (let attempt = 0; attempt < 7; attempt++) {
        const ms = await calibrate('shared-cell', budget);
        if (Math.abs(target - ms) < gap) {
            bestBudget = budget;
            gap = Math.abs(target - ms);
        }
        if (gap <= 0.025 * target) break;
        if (ms > target) high = budget;
        else low = budget;
        budget = Math.round((low + high) / 2);
    }
    // Stop at the first complete frame reaching the target. The overshoot
    // and frame count stay visible with each error measurement.
    const exposure = async (targetMs: number, ref: RgbImage) => {
        let gpuMs = 0, frames = 0;
        while (gpuMs < targetMs) {
            gpuMs += await sample();
            frames++;
        }
        return { targetMs, gpuMs, frames, msPerFrame: gpuMs / frames, ...accuracyNumbers(ref, await image()) };
    };
    const measurements = [];
    // Reverse the order for the replicate to reduce clock/thermal order effects.
    for (const mode of ['gi1-rays', 'shared-cell', 'shared-cell', 'gi1-rays'] as const) {
        changeScene?.(false);
        technique.settings.secondBounce = mode;
        technique.settings.cellShadingBudget = mode === 'gi1-rays' ? baselineBudget : bestBudget;
        technique.resetHistory();
        const still = await exposure(64 * target, before);
        if (technique.worldCache?.secondBounceInUse !== mode) throw new Error('wrong second-bounce estimator');
        let changed;
        if (after && changeScene) {
            changeScene(true);
            changed = await exposure(12 * target, after);
        }
        measurements.push({ mode, budget: technique.settings.cellShadingBudget, still, changed });
    }
    return { calibration, measurements, reference: '96 warm frames plus a 24-frame mean of per-hit GI-1.2 denoised irradiance' };
}
