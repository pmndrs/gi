/**
 * The response to a manifest light motion, from recordings of either side
 * (T2.7, T2.10), measured the same way for both:
 *
 * - native's: `play.ps1 -Motion <motion> -Record` (frames.csv: the GI-only
 *   mean of the view each frame, its luminance, at a fixed 1/60 s a frame);
 * - ours: `tools/gpu/pages/viewer-recording` saved by
 *   `save-viewer-recording.ts` (the same layout; `gi_mean`, the response
 *   meter's mean of red, green and blue, on the same fixed motion clock).
 *
 *   node tools/native/response.ts <recording folder> [...] [--settled <folder> ...] [--against <folder> ...]
 *                                 [--commit <ours: this repo's commit>] [--device <ours: the GPU>] [--append] [--label t2.7]
 *
 * **A step motion.** For each jump (each key after the first), the view's
 * mean each frame from the jump's own frame (frame 1) to the next jump is held
 * against "settled": the view converged with the light held at the new place,
 * from a settle recording passed with --settled (native: `play.ps1
 * -SettleKey <key>`; ours: the page's `VITE_DDGI_SETTLE_KEY`; both 3,600
 * frames held, then the mean of the next 61). Without one, the mean of the
 * hold's last SETTLED_FRAMES frames stands in, and the script reports how far
 * that end still drifts. Measures as T2.4b's `stepResponse`
 * (tools/gpu/pages/ddgi-shared/response.ts): the signed error against settled
 * each frame, the first frame inside 5%, the first frame from which every
 * frame stays inside 5%, the overshoot as a share of the step, and the move in
 * the jump's own frame.
 *
 * **A sweep** (a linear motion out and back along the same path, ending where
 * it starts) has no step, so two plain measures (`sweepResponse`):
 *
 * - **lag behind its own path:** at time t and at time T - t the lamp is at
 *   the same place, so a view that kept up exactly would read the same at
 *   both. A view that runs d frames behind reads, on the way back, what it
 *   read on the way out 2d frames later. The lag is half the shift that best
 *   lines the way back up with the way out (least squares, to a tenth of a
 *   frame). `out_back_gap` is the mean difference between the two at the same
 *   lamp place, as a share of the view's mean;
 * - **against the other side** (--against, paired with the recordings by
 *   order): each curve as a share of its own first frame (the lamp at the
 *   path's start, settled by the 600 frames before), the mean and largest
 *   difference between the two, and the shift that lines this side's curve
 *   up best with the other's (positive: this side later).
 *
 * With --append, rows go to bench/results/response.csv as
 * `response/<native|ours>/<motion>/[grid_<rule>/]run<n>/<jump or sweep>/<measure>`
 * (bench/results/README.md). Native's rows carry browser `native` and the
 * fork's commit (recording.json); ours the browser from recording.json and
 * --commit. Native is a reference, not a target (m2-lights.md §7, Dennis
 * 2026-11-05).
 */

import { appendFileSync, readFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';
import type { ManifestMotion } from '../../scenes/src/manifest.ts';
import { readFramesCsv, readManifest, type LoggedFrame } from './motion.ts';

const repoRoot = resolve(dirname(fileURLToPath(import.meta.url)), '..', '..');
const TOLERANCE = 0.05;
const SETTLED_FRAMES = 30;
const FRAME_RATE = 60;
const FRAME_SECONDS = 1 / FRAME_RATE;
/** The largest shift a sweep's lag search tries, in frames (2 s). */
const MAX_SHIFT_FRAMES = 120;

interface Recording {
    scene: string;
    motion: string;
    device: string;
    recorded: string;
    /** `viewer` for ours (tools/gpu/pages/viewer-recording); absent on native's. */
    source?: string;
    /** Native's: the fork's commit. */
    harnessCommit?: string;
    /** Ours: the browser's user agent. */
    browser?: string;
    /** -1 when the motion played, else the key the light was held at (a settle recording). */
    settleKey?: number;
    /** play.ps1 -GridRule; absent in native's first recordings, which used the harness's own cells grid. Ours: `ddgi`. */
    gridRule?: string;
    /** Ours: the real time between frames, which DDGI ages its histories by. */
    frameIntervalMs?: { mean: number; median: number; largest: number; over25: number };
}

/** Which side made a recording: rows go under response/<side>/. */
function sideOf(recording: Recording): 'ours' | 'native' {
    return recording.source === 'viewer' ? 'ours' : 'native';
}

/**
 * The column holding each frame's brightness: native's luminance of the GI-only view (as T2.7's rows), ours the meter's
 * mean of red, green and blue. Every measure is a share of the same side's own levels, so the two weightings differ
 * only as far as the light's colour mix in the view changes.
 */
function brightnessColumn(frames: readonly LoggedFrame[]): string {
    return 'gi_mean' in frames[0].values ? 'gi_mean' : 'gi_mean_luminance';
}

function brightness(frames: readonly LoggedFrame[]): number[] {
    const column = brightnessColumn(frames);
    return frames.map((frame) => frame.values[column]);
}

function readRecording(folder: string): { recording: Recording; motion: ManifestMotion; frames: LoggedFrame[] } {
    const recording = JSON.parse(readFileSync(join(folder, 'recording.json'), 'utf8')) as Recording;
    const motion = (readManifest(recording.scene).motions ?? []).find((candidate) => candidate.id === recording.motion);
    if (!motion) throw new Error(`${folder}: ${recording.scene} has no motion ${recording.motion}`);
    return { recording, motion, frames: readFramesCsv(join(folder, 'frames.csv')) };
}

interface JumpResponse {
    name: string;
    startFrame: number;
    frames: number;
    before: number;
    settled: number;
    stepShare: number;
    firstFrameChange: number;
    enteredFrame: number | null;
    settledFrame: number | null;
    overshootShareOfStep: number;
    /** How much the hold's last SETTLED_FRAMES frames still move: (last 10 mean − first 10 mean) / settled. */
    endDrift: number;
    /** The settled level came from a long-held recording (a settle recording), not the end of the hold. */
    settledFromLongHold: boolean;
    errors: number[];
}

function mean(values: readonly number[]): number {
    return values.reduce((sum, value) => sum + value, 0) / values.length;
}

/** The view's response to one jump: `means` from the jump's own frame to the frame before the next jump. */
function jumpResponse(name: string, startFrame: number, before: number, means: readonly number[], settledLevel?: number): JumpResponse {
    const tail = means.slice(-SETTLED_FRAMES);
    const settled = settledLevel ?? mean(tail);
    const errors = means.map((value) => value / settled - 1);
    const inBand = errors.map((error) => Math.abs(error) <= TOLERANCE);
    const entered = inBand.indexOf(true);
    let settledIndex: number | null = null;
    for (let index = inBand.length - 1; index >= 0 && inBand[index]; index--) settledIndex = index;
    const direction = Math.sign(settled - before);
    const past = Math.max(0, ...means.map((value) => direction * (value - settled)));
    const step = Math.abs(settled - before);
    return {
        name,
        startFrame,
        frames: means.length,
        before,
        settled,
        stepShare: settled / before - 1,
        firstFrameChange: means[0] / before - 1,
        enteredFrame: entered >= 0 ? entered + 1 : null,
        settledFrame: settledIndex === null ? null : settledIndex + 1,
        overshootShareOfStep: step > 0 ? past / step : 0,
        endDrift: (mean(tail.slice(-10)) - mean(tail.slice(0, 10))) / settled,
        settledFromLongHold: settledLevel !== undefined,
        errors,
    };
}

/**
 * The settled level of a light position, from a settle recording (native's `play.ps1 -SettleKey`, ours the page's
 * VITE_DDGI_SETTLE_KEY; both 3,600 frames held, then 61 recorded): the mean over its recorded frames.
 */
function settledLevel(folder: string): { side: string; position: string; level: number } {
    const { recording, motion, frames } = readRecording(folder);
    if (!(recording.settleKey !== undefined && recording.settleKey >= 0)) throw new Error(`${folder} is not a settle recording`);
    return { side: sideOf(recording), position: String(motion.keyframes[recording.settleKey].position), level: mean(brightness(frames)) };
}

/** The jumps of a step motion's recording: one per key time crossed after 0. */
function stepJumps(folder: string, motion: ManifestMotion, frames: readonly LoggedFrame[], settledByPosition: ReadonlyMap<string, number>): JumpResponse[] {
    if (motion.interpolation !== 'step') throw new Error(`${folder}: ${motion.id} is not a step motion`);
    const luminance = brightness(frames);
    const duration = motion.keyframes[motion.keyframes.length - 1].t;
    const lastTime = frames[frames.length - 1].seconds;

    // The times the lamp jumps: every key after the first, in every loop the recording covers
    const jumpTimes: { time: number; key: number }[] = [];
    for (let loop = 0; loop * duration < lastTime; loop++) {
        for (let key = 1; key < motion.keyframes.length; key++) {
            const time = loop * duration + motion.keyframes[key].t;
            if (time <= lastTime) jumpTimes.push({ time, key });
        }
    }
    // The first frame at or after each jump (the frame that shows it; a small epsilon for the float clock)
    const frameAt = (time: number) => frames.findIndex((frame) => frame.seconds >= time - 1e-4);
    const jumps: JumpResponse[] = [];
    let previousJump = 0;
    for (let index = 0; index < jumpTimes.length; index++) {
        const start = frameAt(jumpTimes[index].time);
        const end = index + 1 < jumpTimes.length ? frameAt(jumpTimes[index + 1].time) : -1;
        if (start <= 0 || end < 0) continue; // the last jump has no full hold after it
        // Before: the settled level of the hold that ends at this jump
        const before = mean(luminance.slice(Math.max(previousJump, start - SETTLED_FRAMES), start));
        const from = motion.keyframes[jumpTimes[index].key - 1].position;
        const to = motion.keyframes[jumpTimes[index].key].position;
        const name = `jump${index + 1}_x${from?.[0]}_to_x${to?.[0]}`;
        jumps.push(jumpResponse(name, start, before, luminance.slice(start, end), settledByPosition.get(String(to))));
        previousJump = start;
    }
    return jumps;
}

/**
 * The shift s (frames, to a tenth) that best lines `a` up with `b`, in the least-squares sense: a[i] ≈ b[i − s] over
 * the frames both have. Positive: `a` runs s frames behind `b`. Searched over whole frames, then refined with a
 * parabola through the best one and its neighbours.
 */
export function bestShift(a: readonly number[], b: readonly number[], maxShift = MAX_SHIFT_FRAMES): { shift: number; rmsDifference: number } {
    const cost = (shift: number): number => {
        let sum = 0;
        let count = 0;
        for (let index = Math.max(0, shift); index < a.length && index - shift < b.length; index++) {
            const difference = a[index] - b[index - shift];
            sum += difference * difference;
            count++;
        }
        // At least half the curve must overlap, so a large shift can't win by comparing two flat ends
        return count >= Math.min(a.length, b.length) / 2 ? sum / count : Number.POSITIVE_INFINITY;
    };
    let best = 0;
    for (let shift = -maxShift; shift <= maxShift; shift++) if (cost(shift) < cost(best)) best = shift;
    const [left, middle, right] = [cost(best - 1), cost(best), cost(best + 1)];
    const curvature = left - 2 * middle + right;
    const offset = Number.isFinite(curvature) && curvature > 0 ? (0.5 * (left - right)) / curvature : 0;
    return { shift: Math.round((best + offset) * 10) / 10, rmsDifference: Math.sqrt(middle) };
}

export interface SweepResponse {
    /** Frames in one pass out and back (the motion's length × 60), plus the closing frame. */
    frames: number;
    /** How far the view runs behind the lamp, in frames: half the shift that lines the way back up with the way out. */
    lagFrames: number;
    /** The mean difference between the view on the way out and on the way back at the same lamp place, as a share of the view's mean. */
    outBackGap: number;
    /** The view's lowest and highest, as shares of its first frame (the lamp at the path's start, settled). */
    lowestShare: number;
    highestShare: number;
    /** The curve as shares of its first frame, for --against. */
    relative: number[];
}

/**
 * A sweep out and back along the same path (the lamp at t and at T − t is in the same place). If the view followed
 * the lamp d frames late, view(t) = settled(t − d) and view(T − t) = settled(t + d): the way back is the way out
 * shifted by 2d. `values` is one pass, frame 0 to T.
 */
export function sweepResponse(values: readonly number[]): SweepResponse {
    const last = values.length - 1;
    const mirrored = values.map((_, index) => values[last - index]);
    const { shift } = bestShift(values, mirrored);
    const average = mean(values);
    const outBackGap = mean(values.map((value, index) => Math.abs(value - mirrored[index]))) / average;
    const relative = values.map((value) => value / values[0]);
    return {
        frames: values.length,
        lagFrames: Math.round((shift / 2) * 100) / 100,
        outBackGap,
        lowestShare: Math.min(...relative),
        highestShare: Math.max(...relative),
        relative,
    };
}

/** One pass of an out-and-back sweep from a recording: frames 0 to the motion's length. */
function sweepPass(folder: string, motion: ManifestMotion, frames: readonly LoggedFrame[]): number[] {
    if (motion.interpolation === 'step') throw new Error(`${folder}: ${motion.id} is a step motion`);
    const first = motion.keyframes[0].position;
    const final = motion.keyframes[motion.keyframes.length - 1].position;
    if (String(first) !== String(final)) throw new Error(`${folder}: ${motion.id} does not end where it starts (the lag measure needs an out-and-back path)`);
    const passFrames = Math.round(motion.keyframes[motion.keyframes.length - 1].t * FRAME_RATE) + 1;
    if (frames.length < passFrames) throw new Error(`${folder}: ${frames.length} frames, one pass needs ${passFrames}`);
    return brightness(frames).slice(0, passFrames);
}

function round(value: number, digits = 5): number {
    return Number(value.toPrecision(digits));
}

/** Ours: "Chrome <version>[ headless]" from the user agent, as the other response rows write it. */
function browserLabel(recording: Recording): string {
    if (sideOf(recording) === 'native') return 'native';
    const agent = recording.browser ?? '';
    const version = /(?:HeadlessChrome|Chrome)\/([\d.]+)/.exec(agent)?.[1];
    return version ? `Chrome ${version}${agent.includes('HeadlessChrome') ? ' headless' : ''}` : 'unknown';
}

const USAGE =
    'usage: node tools/native/response.ts <recording folder> [...] [--settled <settle recording> ...] ' +
    "[--against <the other side's recording> ...] [--commit <ours: this repo's commit>] [--device <ours: the GPU>] [--append] [--label text]";

function main(): void {
    const { values, positionals } = parseArgs({
        allowPositionals: true,
        options: {
            append: { type: 'boolean', default: false },
            label: { type: 'string', default: '' },
            settled: { type: 'string', multiple: true, default: [] },
            against: { type: 'string', multiple: true, default: [] },
            commit: { type: 'string', default: '' },
            // Ours: the GPU's name when the browser's adapter gives none (headless Chrome says only "nvidia ampere")
            device: { type: 'string', default: '' },
        },
    });
    if (positionals.length === 0) {
        console.log(USAGE);
        process.exitCode = 2;
        return;
    }
    // Settled levels from long-held recordings; without one, a jump's settled level is the end of its own hold
    const settledByPosition = new Map<string, number>();
    for (const folder of values.settled) {
        const { side, position, level } = settledLevel(resolve(folder));
        settledByPosition.set(position, level);
        console.log(`settled level (${side}) with the light at ${position}: ${level.toPrecision(6)} (${folder})`);
    }
    const rows: string[] = [];
    positionals.forEach((folder, runIndex) => {
        const { recording, motion, frames } = readRecording(resolve(folder));
        const side = sideOf(recording);
        if (side === 'ours' && !values.commit) throw new Error('ours: pass --commit (the commit the recording was made at)');
        const commit = side === 'ours' ? values.commit : (recording.harnessCommit ?? 'unknown');
        const runNumber = runIndex + 1;
        console.log(`run ${runNumber} (${side}): ${recording.scene} ${recording.motion} (${folder}), commit ${commit}, brightness ${brightnessColumn(frames)}`);
        const date = recording.recorded.slice(0, 10);
        const device = (side === 'ours' && values.device ? values.device : recording.device).replace(/^NVIDIA GeForce /, '').replace(/ GPU$/, '');
        // The probe grid (play.ps1 -GridRule) is in native's path for every grid but its own cells grid, whose rows came
        // first; ours always uses DDGI's own grid, so its path has none.
        const grid = recording.gridRule ?? 'cells';
        const run = `${side === 'native' && grid !== 'cells' ? `grid_${grid}/` : ''}run${runNumber}`;
        const runPrefix = `${date},${commit},${device},${browserLabel(recording)},${recording.scene},response/${side}/${recording.motion}/${run}`;
        // The run's conditions: the grid, the brightness column, and for ours the probes a frame and the real frame
        // interval DDGI aged its histories by
        const conditionRows = (prefix: string): void => {
            rows.push(`${prefix}/grid_rule,${grid}`);
            rows.push(`${prefix}/brightness,${brightnessColumn(frames)}`);
            if (side === 'ours') {
                const probes = frames.map((frame) => frame.values.probes_updated).sort((a, b) => a - b);
                rows.push(`${prefix}/probes_per_frame_median,${probes[Math.floor(probes.length / 2)]}`);
                if (recording.frameIntervalMs) rows.push(`${prefix}/frame_interval_mean_ms,${round(recording.frameIntervalMs.mean, 4)}`);
            }
            if (values.label) rows.push(`${prefix}/label,${values.label}`);
        };

        if (motion.interpolation === 'step') {
            for (const jump of stepJumps(folder, motion, frames, settledByPosition)) {
                const describe = jump.errors
                    .map((error, index) => ({ frame: index + 1, error }))
                    .filter(({ frame }) => frame <= 12 || frame % 12 === 0)
                    .map(({ frame, error }) => `${frame}: ${(error * 100).toFixed(1)}%`)
                    .join(', ');
                console.log(
                    `  ${jump.name} at frame ${jump.startFrame} (${(jump.startFrame * FRAME_SECONDS).toFixed(2)} s): step ${(jump.stepShare * 100).toFixed(1)}%, ` +
                        `own frame ${(jump.firstFrameChange * 100).toFixed(1)}%, entered 5% at frame ${jump.enteredFrame ?? 'never'}, settled from frame ${jump.settledFrame ?? 'never'}, ` +
                        `overshoot ${(jump.overshootShareOfStep * 100).toFixed(1)}% of the step, hold's end drifts ${(jump.endDrift * 100).toFixed(2)}%`,
                );
                console.log(`    error by frame: ${describe}`);
                const prefix = `${runPrefix}/${jump.name}`;
                conditionRows(prefix);
                rows.push(`${prefix}/step_size,${round(jump.stepShare)}`);
                rows.push(`${prefix}/first_frame_move,${round(jump.firstFrameChange)}`);
                rows.push(`${prefix}/entered_frame,${jump.enteredFrame ?? -1}`);
                rows.push(`${prefix}/settled_frame,${jump.settledFrame ?? -1}`);
                rows.push(`${prefix}/overshoot_share_of_step,${round(jump.overshootShareOfStep)}`);
                rows.push(`${prefix}/settled_end_drift,${round(jump.endDrift)}`);
                rows.push(`${prefix}/settled_from_long_hold,${jump.settledFromLongHold ? 1 : 0}`);
                jump.errors.forEach((error, index) => {
                    const frame = index + 1;
                    if (frame <= 30 || frame % 6 === 0) rows.push(`${prefix}/error_frame${frame},${round(error)}`);
                });
            }
            return;
        }

        const sweep = sweepResponse(sweepPass(folder, motion, frames));
        console.log(
            `  sweep: lag behind its own path ${sweep.lagFrames.toFixed(2)} frames (${(sweep.lagFrames * FRAME_SECONDS * 1000).toFixed(0)} ms), ` +
                `way out against way back ${(sweep.outBackGap * 100).toFixed(2)}% of the mean apart, ` +
                `lowest ${(sweep.lowestShare * 100).toFixed(1)}% and highest ${(sweep.highestShare * 100).toFixed(1)}% of the start`,
        );
        const prefix = `${runPrefix}/sweep`;
        conditionRows(prefix);
        rows.push(`${prefix}/lag_frames,${sweep.lagFrames}`);
        rows.push(`${prefix}/out_back_gap,${round(sweep.outBackGap)}`);
        rows.push(`${prefix}/lowest_share_of_start,${round(sweep.lowestShare)}`);
        rows.push(`${prefix}/highest_share_of_start,${round(sweep.highestShare)}`);
        // Every 12th frame of the curve (shares of its start), so the CSV keeps its shape
        sweep.relative.forEach((share, frame) => {
            if (frame % 12 === 0) rows.push(`${prefix}/share_of_start_frame${frame},${round(share)}`);
        });

        const otherFolder = values.against[runIndex];
        if (otherFolder) {
            const other = readRecording(resolve(otherFolder));
            const otherSide = sideOf(other.recording);
            const otherSweep = sweepResponse(sweepPass(otherFolder, other.motion, other.frames));
            const length = Math.min(sweep.relative.length, otherSweep.relative.length);
            const differences = sweep.relative.slice(0, length).map((share, frame) => Math.abs(share - otherSweep.relative[frame]));
            const { shift } = bestShift(sweep.relative, otherSweep.relative);
            console.log(
                `  against ${otherSide} (${otherFolder}): curves as shares of their starts differ by ${(mean(differences) * 100).toFixed(2)} points on average, ` +
                    `${(Math.max(...differences) * 100).toFixed(2)} at most; ${side} runs ${shift.toFixed(1)} frames behind ${otherSide}`,
            );
            const againstPrefix = `${prefix}/against_${otherSide}_run${runNumber}`;
            rows.push(`${againstPrefix}/mean_abs_difference_share_of_start,${round(mean(differences))}`);
            rows.push(`${againstPrefix}/largest_abs_difference_share_of_start,${round(Math.max(...differences))}`);
            rows.push(`${againstPrefix}/lag_frames,${shift}`);
        }
    });
    if (values.append) {
        appendFileSync(join(repoRoot, 'bench', 'results', 'response.csv'), `${rows.join('\n')}\n`);
        console.log(`appended ${rows.length} rows to bench/results/response.csv`);
    }
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) main();
