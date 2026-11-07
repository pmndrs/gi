/**
 * The response meter's numbers (docs/design/m2-lights.md §7): a picked part
 * of the picture, its mean in each frame after a step, and two marks:
 *
 * - **first change:** the first frame whose mean moved by more than
 *   CHANGE_SHARE from before the step, as a share of the larger of before
 *   and settled (so a lamp switched on from black still has a scale);
 * - **within 5%:** the first frame within TOLERANCE of settled, and the
 *   first frame from which every measured frame stays within it ("for
 *   good"; a curve that enters the band and leaves it again is not settled
 *   until it is back for good).
 *
 * The definitions are the T2.4b test page's (`tools/gpu/pages/ddgi-shared/
 * response.ts`, `stepResponse`), so the meter's CSV and that page can be
 * compared frame for frame (T2.8 check 2). Frame 1 is the step's own frame.
 *
 * No DOM and no engine, so it is unit-tested.
 */

/** A frame counts as changed once it moved by more than this share. */
export const CHANGE_SHARE = 0.01;
/** Within this share of settled counts as settled. */
export const TOLERANCE = 0.05;

export interface ResponseSample {
    /** 1 is the step's own frame. */
    frame: number;
    /** Milliseconds since the step's frame began. */
    timeMs: number;
    /** The picked region's mean (linear, GI only), averaged over its pixels and colour channels. */
    mean: number;
    /** DDGI's probes updated in that frame. */
    probesUpdated: number;
}

export interface StepMarks {
    readonly before: number;
    readonly settled: number;
    /** The step as a share of the larger of before and settled. */
    readonly stepShare: number;
    readonly firstChangedFrame: number | null;
    readonly enteredFrame: number | null;
    readonly settledFrame: number | null;
    /** It entered the band and left it again. */
    readonly leftBandAgain: boolean;
    /** How far past settled it went, in the step's direction, as a share of the step (0 if never). */
    readonly overshootShareOfStep: number;
}

/** The marks of a step from its curve (frames in order, frame 1 first). */
export function stepMarks(before: number, settled: number, samples: readonly ResponseSample[]): StepMarks {
    const scale = Math.max(Math.abs(before), Math.abs(settled));
    const changed = samples.find((sample) => scale > 0 && Math.abs(sample.mean - before) / scale > CHANGE_SHARE);
    const inBand = samples.map((sample) => settled !== 0 && Math.abs(sample.mean / settled - 1) <= TOLERANCE);
    const enteredIndex = inBand.indexOf(true);
    let settledIndex: number | null = null;
    for (let index = inBand.length - 1; index >= 0 && inBand[index]; index--) settledIndex = index;
    const direction = Math.sign(settled - before);
    const past = Math.max(0, ...samples.map((sample) => direction * (sample.mean - settled)));
    const step = Math.abs(settled - before);
    return {
        before,
        settled,
        stepShare: scale > 0 ? step / scale : 0,
        firstChangedFrame: changed?.frame ?? null,
        enteredFrame: enteredIndex >= 0 ? samples[enteredIndex].frame : null,
        settledFrame: settledIndex === null ? null : samples[settledIndex].frame,
        leftBandAgain: enteredIndex >= 0 && (settledIndex === null || settledIndex > enteredIndex),
        overshootShareOfStep: step > 0 ? past / step : 0,
    };
}

/** What a CSV says about how it was measured. */
export interface ResponseRunInfo {
    scene: string;
    camera: string;
    motion: string;
    /** The region in drawing-buffer pixels: x, y, width, height (top-left origin). */
    region: readonly [number, number, number, number];
    budgetMs: number;
    adapter: string;
    /** Frames at the settling budget and their budget, for the `settled` line. */
    settleFrames: number;
    settleBudgetMs: number;
}

function csvField(value: string): string {
    return /[",\n]/.test(value) ? `"${value.replaceAll('"', '""')}"` : value;
}

/**
 * The curve as CSV: comment lines (`#`) say what was measured and the marks,
 * then one row per frame: frame, ms since the step, the region's mean, its
 * change from before and its error against settled (both as shares), and
 * DDGI's probes updated that frame.
 */
export function responseCsv(info: ResponseRunInfo, marks: StepMarks, samples: readonly ResponseSample[]): string {
    const mark = (frame: number | null) => (frame === null ? 'never' : String(frame));
    const lines = [
        `# DDGI viewer response meter (GI only): ${csvField(info.scene)}, camera ${csvField(info.camera)}, motion ${csvField(info.motion)}`,
        `# region x,y,width,height (pixels, top left): ${info.region.join(',')}; budget ${info.budgetMs} ms; adapter ${csvField(info.adapter)}`,
        `# before ${marks.before}; settled ${marks.settled} (mean of the last frames after ${info.settleFrames} frames at ${info.settleBudgetMs} ms, a test mode); step ${(marks.stepShare * 100).toFixed(1)}%`,
        `# first frame changed by more than ${CHANGE_SHARE * 100}%: ${mark(marks.firstChangedFrame)}; first within ${TOLERANCE * 100}% of settled: ${mark(marks.enteredFrame)}; within it for good from: ${mark(marks.settledFrame)}${marks.leftBandAgain ? ' (left the band again)' : ''}; overshoot ${(marks.overshootShareOfStep * 100).toFixed(1)}% of the step`,
        'frame,ms,mean,change_from_before,error_from_settled,probes_updated',
    ];
    const scale = Math.max(Math.abs(marks.before), Math.abs(marks.settled));
    for (const sample of samples) {
        const change = scale > 0 ? (sample.mean - marks.before) / scale : 0;
        const error = marks.settled !== 0 ? sample.mean / marks.settled - 1 : 0;
        lines.push(`${sample.frame},${sample.timeMs.toFixed(2)},${sample.mean.toPrecision(7)},${change.toFixed(5)},${error.toFixed(5)},${sample.probesUpdated}`);
    }
    return `${lines.join('\n')}\n`;
}

/** Reads a meter CSV back: its marks line and rows (for tests and the comparison page). */
export function parseResponseCsv(text: string): { samples: ResponseSample[]; enteredFrame: number | null; settledFrame: number | null; firstChangedFrame: number | null } {
    const samples: ResponseSample[] = [];
    let enteredFrame: number | null = null;
    let settledFrame: number | null = null;
    let firstChangedFrame: number | null = null;
    const frameAfter = (line: string, label: string): number | null => {
        const match = new RegExp(`${label}: (never|\\d+)`).exec(line);
        return match && match[1] !== 'never' ? Number(match[1]) : null;
    };
    for (const line of text.split('\n')) {
        if (line.startsWith('# first frame changed')) {
            firstChangedFrame = frameAfter(line, `more than ${CHANGE_SHARE * 100}%`);
            enteredFrame = frameAfter(line, `within ${TOLERANCE * 100}% of settled`);
            settledFrame = frameAfter(line, 'for good from');
            continue;
        }
        if (line.startsWith('#') || line.startsWith('frame') || line.trim() === '') continue;
        const [frame, ms, mean, , , probes] = line.split(',');
        samples.push({ frame: Number(frame), timeMs: Number(ms), mean: Number(mean), probesUpdated: Number(probes) });
    }
    return { samples, enteredFrame, settledFrame, firstChangedFrame };
}
