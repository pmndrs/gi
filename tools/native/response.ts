/**
 * The native harness's response to a manifest step motion (T2.7), from
 * recordings made by `play.ps1 -Motion <step> -Record` (frames.csv: the
 * GI-only mean of the view each frame, at a fixed 1/60 s a frame).
 *
 *   node tools/native/response.ts <recording folder> [<recording folder> ...] [--append] [--label t2.7]
 *
 * For each jump of the motion (each key after the first), the view's mean
 * each frame from the jump's own frame (frame 1) to the next jump is held
 * against "settled": the mean of the last SETTLED_FRAMES frames of that
 * hold. The harness traces and blends every probe every frame, so a 3 s
 * hold (180 frames at hysteresis 0.97) ends converged; the script reports
 * how far the hold's end still drifts so that can be checked. Measures as
 * T2.4b's `stepResponse` (tools/gpu/pages/ddgi-shared/response.ts): the
 * signed error against settled each frame, the first frame inside 5%, the
 * first frame from which every frame stays inside 5%, the overshoot as a
 * share of the step, and the move in the jump's own frame.
 *
 * With --append, rows go to bench/results/response.csv as
 * `response/native/<motion>/run<n>/<jump>/<measure>`, browser `native`,
 * commit the fork's (from recording.json). Native is a reference, not a
 * target (m2-lights.md §7, Dennis 2026-11-05).
 */

import { appendFileSync, readFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';
import { readFramesCsv, readManifest } from './motion.ts';

const repoRoot = resolve(dirname(fileURLToPath(import.meta.url)), '..', '..');
const TOLERANCE = 0.05;
const SETTLED_FRAMES = 30;
const FRAME_SECONDS = 1 / 60;

interface Recording {
    scene: string;
    motion: string;
    harnessCommit: string;
    device: string;
    recorded: string;
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
    errors: number[];
}

function mean(values: readonly number[]): number {
    return values.reduce((sum, value) => sum + value, 0) / values.length;
}

/** The view's response to one jump: `means` from the jump's own frame to the frame before the next jump. */
function jumpResponse(name: string, startFrame: number, before: number, means: readonly number[]): JumpResponse {
    const tail = means.slice(-SETTLED_FRAMES);
    const settled = mean(tail);
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
        errors,
    };
}

/** The jumps in a recording of a looping step motion: one per key time crossed after 0. */
function analyse(folder: string): { recording: Recording; jumps: JumpResponse[] } {
    const recording = JSON.parse(readFileSync(join(folder, 'recording.json'), 'utf8')) as Recording;
    const manifest = readManifest(recording.scene);
    const motion = (manifest.motions ?? []).find((candidate) => candidate.id === recording.motion);
    if (!motion || motion.interpolation !== 'step') throw new Error(`${folder}: ${recording.motion} is not a step motion of ${recording.scene}`);
    const frames = readFramesCsv(join(folder, 'frames.csv'));
    const luminance = frames.map((frame) => frame.values.gi_mean_luminance);
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
        jumps.push(jumpResponse(name, start, before, luminance.slice(start, end)));
        previousJump = start;
    }
    return { recording, jumps };
}

function round(value: number, digits = 5): number {
    return Number(value.toPrecision(digits));
}

function main(): void {
    const { values, positionals } = parseArgs({
        allowPositionals: true,
        options: { append: { type: 'boolean', default: false }, label: { type: 'string', default: '' } },
    });
    if (positionals.length === 0) {
        console.log('usage: node tools/native/response.ts <recording folder> [...] [--append] [--label text]');
        process.exitCode = 2;
        return;
    }
    const rows: string[] = [];
    positionals.forEach((folder, runIndex) => {
        const { recording, jumps } = analyse(resolve(folder));
        const run = runIndex + 1;
        console.log(`run ${run}: ${recording.scene} ${recording.motion} (${folder}), fork ${recording.harnessCommit}`);
        for (const jump of jumps) {
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
            const date = recording.recorded.slice(0, 10);
            const device = recording.device.replace(/^NVIDIA GeForce /, '').replace(/ GPU$/, '');
            const prefix = `${date},${recording.harnessCommit},${device},native,${recording.scene},response/native/${recording.motion}/run${run}/${jump.name}`;
            rows.push(`${prefix}/step_size,${round(jump.stepShare)}`);
            rows.push(`${prefix}/first_frame_move,${round(jump.firstFrameChange)}`);
            rows.push(`${prefix}/entered_frame,${jump.enteredFrame ?? -1}`);
            rows.push(`${prefix}/settled_frame,${jump.settledFrame ?? -1}`);
            rows.push(`${prefix}/overshoot_share_of_step,${round(jump.overshootShareOfStep)}`);
            rows.push(`${prefix}/settled_end_drift,${round(jump.endDrift)}`);
            jump.errors.forEach((error, index) => {
                const frame = index + 1;
                if (frame <= 30 || frame % 6 === 0) rows.push(`${prefix}/error_frame${frame},${round(error)}`);
            });
            if (values.label) rows.push(`${prefix}/label,${values.label}`);
        }
    });
    if (values.append) {
        appendFileSync(join(repoRoot, 'bench', 'results', 'response.csv'), `${rows.join('\n')}\n`);
        console.log(`appended ${rows.length} rows to bench/results/response.csv`);
    }
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) main();
