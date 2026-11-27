/**
 * P6.5's measurements of how the radiance cache's variants behave over time,
 * as plain functions of the numbers the page reads back (unit-tested in
 * metrics.test.ts). Measurement only: nothing here gates.
 *
 * - {@link stepResponse}: how fast the picture follows a change (the lights
 *   doubled, a box moved) and how much of the old picture is left a few
 *   frames later (ghosting), from 8 × 8 block means of the irradiance per
 *   frame.
 * - {@link settleFrame}: the frame from which a series stays near its final
 *   value (the warm-up from an empty cache).
 * - {@link flicker}: the frame-to-frame change per pixel of a still view.
 */

/** How a picture followed one change, from the frames after it. */
export interface StepResponse {
    /** Blocks whose settled value differs from before the change (the region measured). */
    readonly changedBlocks: number;
    /** The changed region's settled brightness over its brightness before the change. */
    readonly endOverBefore: number;
    /**
     * Frames after the change from which the region stays at least 50% (or 90%) of the way
     * from before to settled, past the run's own noise; null when it never does before the
     * settled window.
     */
    readonly framesTo50: number | null;
    readonly framesTo90: number | null;
    /** The share of the change still missing at frames 1, 4 and 16 after it (0: settled, 1: as before). */
    readonly residualAt1: number;
    readonly residualAt4: number;
    readonly residualAt16: number;
    /** How far the region's brightness went past settled, as a share of the step (0: never past). */
    readonly overshoot: number;
    /** The residual the settled frames themselves show against their mean: the run's noise. */
    readonly noiseFloor: number;
}

export interface StepOptions {
    /** The last frames, averaged into the settled value. */
    readonly settleFrames: number;
    /** A block counts as changed when settled and before differ by this share of the larger. */
    readonly minRelativeChange: number;
    /** ... and the larger is at least this share of the frame's mean (dark blocks are left out). */
    readonly darkShare: number;
}

export const DEFAULT_STEP_OPTIONS: StepOptions = { settleFrames: 30, minRelativeChange: 0.1, darkShare: 0.05 };

function meanOf(frames: readonly Float32Array[], from: number, to: number, block: number): number {
    let sum = 0;
    for (let frame = from; frame < to; frame++) sum += frames[frame][block];
    return sum / Math.max(1, to - from);
}

/**
 * `before`: each block's mean over the frames before the change. `after`: each block's mean per
 * frame, starting with the first frame drawn after the change. The last `settleFrames` frames
 * are the settled picture. Blocks `exclude` marks are left out (a moved box's own footprint,
 * whose pixels change surface at once, so only the light around it is measured).
 */
export function stepResponse(before: Float32Array, after: readonly Float32Array[], options: StepOptions = DEFAULT_STEP_OPTIONS, exclude?: Uint8Array): StepResponse | null {
    const frames = after.length;
    const settleFrom = frames - options.settleFrames;
    if (settleFrom < 16) throw new Error(`stepResponse: ${frames} frames after the change leave too few before the ${options.settleFrames} settled frames`);
    const blocks = before.length;
    const end = new Float32Array(blocks);
    let frameMean = 0;
    for (let block = 0; block < blocks; block++) {
        end[block] = meanOf(after, settleFrom, frames, block);
        frameMean += end[block];
    }
    frameMean /= Math.max(1, blocks);
    const changed: number[] = [];
    let sumBefore = 0;
    let sumEnd = 0;
    let stepSize = 0;
    for (let block = 0; block < blocks; block++) {
        if (exclude?.[block]) continue;
        const larger = Math.max(before[block], end[block]);
        if (!(larger > options.darkShare * frameMean)) continue;
        if (!(Math.abs(end[block] - before[block]) > options.minRelativeChange * larger)) continue;
        changed.push(block);
        sumBefore += before[block];
        sumEnd += end[block];
        stepSize += Math.abs(before[block] - end[block]);
    }
    if (changed.length === 0 || !(stepSize > 0)) return null;

    const residual = (frame: number): number => {
        let sum = 0;
        for (const block of changed) sum += Math.abs(after[frame][block] - end[block]);
        return sum / stepSize;
    };
    const residuals = Array.from({ length: frames }, (_, frame) => residual(frame));
    let noiseFloor = 0;
    for (let frame = settleFrom; frame < frames; frame++) noiseFloor += residuals[frame];
    noiseFloor /= options.settleFrames;

    // The first frame from which every later frame before the settled window stays under the limit.
    const framesTo = (share: number): number | null => {
        const limit = 1 - share + noiseFloor;
        let first: number | null = null;
        for (let frame = 0; frame < settleFrom; frame++) {
            if (residuals[frame] <= limit) first ??= frame + 1;
            else first = null;
        }
        return first;
    };

    let overshoot = 0;
    const signedStep = sumEnd - sumBefore;
    if (Math.abs(signedStep) > 0) {
        for (let frame = 0; frame < frames; frame++) {
            let sum = 0;
            for (const block of changed) sum += after[frame][block];
            overshoot = Math.max(overshoot, (sum - sumBefore) / signedStep - 1);
        }
    }
    const at = (frame: number): number => residuals[Math.min(frame, frames) - 1];
    return {
        changedBlocks: changed.length,
        endOverBefore: sumBefore > 0 ? sumEnd / sumBefore : Number.POSITIVE_INFINITY,
        framesTo50: framesTo(0.5),
        framesTo90: framesTo(0.9),
        residualAt1: at(1),
        residualAt4: at(4),
        residualAt16: at(16),
        overshoot,
        noiseFloor,
    };
}

/**
 * The first frame (1-based) from which `series` stays within `tolerance` (a share) of `target`,
 * or null when its last frame is still outside.
 */
export function settleFrame(series: readonly number[], target: number, tolerance: number): number | null {
    let first: number | null = null;
    for (let frame = 0; frame < series.length; frame++) {
        if (Math.abs(series[frame] - target) <= tolerance * Math.abs(target)) first ??= frame + 1;
        else first = null;
    }
    return first;
}

export interface Flicker {
    /** Mean over surface pixels and frame pairs of |E_t − E_t−1| / max(E_t−1, floor). */
    readonly meanPerPixel: number;
    /** Its 95th percentile. */
    readonly p95PerPixel: number;
    /** Mean over the pairs of the frame's whole change, |ΣE_t − ΣE_t−1| / ΣE_t−1 (the picture brightening and darkening). */
    readonly meanOfFrame: number;
    readonly pairs: number;
}

/** The share of the frame's mean below which a pixel's change is measured against that floor instead of itself. */
export const FLICKER_FLOOR_SHARE = 0.05;

/**
 * Frame-to-frame change of a still view's irradiance. `pairs`: consecutive frames' E (one value per
 * pixel, e.g. the mean of rgb); `surface`: 1 where the pixel shows a surface.
 */
export function flicker(pairs: readonly (readonly [Float32Array, Float32Array])[], surface: Uint8Array): Flicker {
    const changes: number[] = [];
    let meanOfFrame = 0;
    for (const [previous, current] of pairs) {
        let sumPrevious = 0;
        let sumCurrent = 0;
        let count = 0;
        for (let pixel = 0; pixel < surface.length; pixel++) {
            if (!surface[pixel]) continue;
            sumPrevious += previous[pixel];
            sumCurrent += current[pixel];
            count++;
        }
        const floor = FLICKER_FLOOR_SHARE * (sumPrevious / Math.max(1, count));
        // Every fourth surface pixel: the percentile over a 1080p frame's pairs needs no more.
        for (let pixel = 0; pixel < surface.length; pixel += 4) {
            if (!surface[pixel]) continue;
            changes.push(Math.abs(current[pixel] - previous[pixel]) / Math.max(previous[pixel], floor, 1e-12));
        }
        meanOfFrame += sumPrevious > 0 ? Math.abs(sumCurrent - sumPrevious) / sumPrevious : 0;
    }
    const sorted = Float64Array.from(changes).sort();
    let sum = 0;
    for (const change of changes) sum += change;
    return {
        meanPerPixel: changes.length > 0 ? sum / changes.length : 0,
        p95PerPixel: sorted.length > 0 ? sorted[Math.min(sorted.length - 1, Math.floor(0.95 * sorted.length))] : 0,
        meanOfFrame: pairs.length > 0 ? meanOfFrame / pairs.length : 0,
        pairs: pairs.length,
    };
}
