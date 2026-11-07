/**
 * The response meter (docs/design/m2-lights.md §7): how fast DDGI's picture
 * follows a light's step, read from the picture itself, in GI-only mode (so
 * three's own direct light, which changes at once, doesn't count).
 *
 * A measurement, frame by frame, with the camera still:
 *
 * 1. **settle before** (a test mode): the step motion's light at its first
 *    keyframe, DDGI's budget raised to SETTLE_BUDGET_MS for SETTLE_SECONDS,
 *    so the field has converged there;
 * 2. **steady before:** the user's budget again for STEADY_SECONDS, so every
 *    probe's history is back at its everyday length when the step comes, as
 *    the T2.4b page's 1,200 frames at the batch do; `before` is the mean of
 *    the last AVERAGED_FRAMES frames;
 * 3. **record:** the light jumps to its second keyframe (frame 1 is the
 *    jump's own frame: the light moves before GI's update in that frame) and
 *    RECORD_FRAMES frames are read at the user's budget;
 * 4. **settle after** (the test mode again): SETTLE_SECONDS at
 *    SETTLE_BUDGET_MS; `settled` is the mean of the last AVERAGED_FRAMES.
 *
 * Every frame of those that counts is read back from DDGI's own picture
 * (`DdgiTechnique.readRegionMean`, the picked region's mean over pixels and
 * colour channels, linear). The marks and the CSV are `stats/response-curve.ts`,
 * the T2.4b page's definitions.
 *
 * A camera move cancels it (the picture would change for another reason).
 */

import type { Light } from 'three';
import type { ManifestMotion } from '../../../../scenes/src/manifest';
import { responseCsv, stepMarks, type ResponseSample, type StepMarks } from '../stats/response-curve';
import type { DdgiTechnique } from '../techniques/ddgi-technique';
import type { MotionPlayer } from './motion-player';

/** The test mode's budget, GPU ms a frame (m2-lights.md §7: "the budget raised to 16 ms"). */
export const SETTLE_BUDGET_MS = 16;
/** How long the test mode runs before and after the step (m2-lights.md §7: "the mean after 4 s"). */
export const SETTLE_SECONDS = 4;
/** At the user's budget before the step, so the histories are at their everyday length. */
export const STEADY_SECONDS = 4;
/** Frames recorded after the step: 2 s at 60 Hz, as the T2.4b page measures. */
export const RECORD_FRAMES = 120;
/** `before` and `settled` are means over this many frames, to average out DDGI's frame-to-frame noise. */
export const AVERAGED_FRAMES = 8;

export type MeterPhase = 'idle' | 'settle-before' | 'steady-before' | 'record' | 'settle-after' | 'reading' | 'done' | 'failed';

/** A region of the drawing buffer, pixels from the top left. */
export interface MeterRegion {
    x: number;
    y: number;
    width: number;
    height: number;
}

export interface MeterResult {
    readonly samples: ResponseSample[];
    readonly marks: StepMarks;
    readonly csv: string;
    readonly motion: string;
}

export interface MeterContext {
    ddgi(): DdgiTechnique | null;
    readonly player: MotionPlayer;
    /** The whole drawing buffer, for a meter with no region picked. */
    drawingBufferSize(): { width: number; height: number };
    /** Tells GI at once that a light changed (the adapter would otherwise find a light past its first 64 a few frames late). */
    markChanged(light: Light): void;
    /** What the CSV says about the run. */
    describe(): { scene: string; camera: string; adapter: string };
}

export class ResponseMeter {
    /** The picked region, or null for the whole picture. */
    region: MeterRegion | null = null;
    private phaseValue: MeterPhase = 'idle';
    private phaseStartedAt = 0;
    private motion: ManifestMotion | null = null;
    private light: Light | null = null;
    /** Frames counted in the current phase. */
    private phaseFrames = 0;
    private stepStartedAt = 0;
    private beforeReadings: Array<Promise<number>> = [];
    private afterReadings: Array<Promise<number>> = [];
    private settleFrames = 0;
    private recorded: Array<{ frame: number; timeMs: number; probesUpdated: number; mean: Promise<number> }> = [];
    /** Means that have arrived, frame 1 first, for the live plot. */
    readonly live: Array<{ frame: number; mean: number }> = [];
    private resultValue: MeterResult | null = null;
    private failure: string | null = null;
    private usedRegion: MeterRegion = { x: 0, y: 0, width: 1, height: 1 };
    private userBudgetMs = 0;

    constructor(private readonly context: MeterContext) {}

    get phase(): MeterPhase {
        return this.phaseValue;
    }

    get running(): boolean {
        return this.phaseValue !== 'idle' && this.phaseValue !== 'done' && this.phaseValue !== 'failed';
    }

    get result(): MeterResult | null {
        return this.resultValue;
    }

    get failed(): string | null {
        return this.failure;
    }

    /** Plain words for where the measurement is. */
    describe(now: number): string {
        const seconds = (now - this.phaseStartedAt) / 1000;
        const left = (total: number) => `${Math.max(0, Math.ceil(total - seconds))} s left`;
        switch (this.phaseValue) {
            case 'idle':
                return 'Pick a region (or keep the whole picture), then measure a step.';
            case 'settle-before':
                return `Test mode, budget raised to ${SETTLE_BUDGET_MS} ms: letting the light settle before the step (${left(SETTLE_SECONDS)}).`;
            case 'steady-before':
                return `Back at your budget, so the probes are as usual when the step comes (${left(STEADY_SECONDS)}).`;
            case 'record':
                return `Recording the step: frame ${this.phaseFrames} of ${RECORD_FRAMES}.`;
            case 'settle-after':
                return `Test mode, budget raised to ${SETTLE_BUDGET_MS} ms: finding the settled picture (${left(SETTLE_SECONDS)}).`;
            case 'reading':
                return 'Reading the last frames back...';
            case 'done':
                return 'Done. Save the CSV, or measure again.';
            case 'failed':
                return `Stopped: ${this.failure}`;
        }
    }

    /** Starts measuring `motion`'s step (its first keyframe, then its second). */
    start(motion: ManifestMotion, now: number): void {
        const ddgi = this.context.ddgi();
        if (!ddgi || !ddgi.running) {
            this.fail('DDGI is not running yet; wait for it to warm up.');
            return;
        }
        this.cancel();
        this.context.player.stop();
        this.motion = motion;
        this.failure = null;
        this.resultValue = null;
        this.live.length = 0;
        this.recorded = [];
        this.beforeReadings = [];
        this.afterReadings = [];
        this.userBudgetMs = ddgi.currentSettings.budgetMs;
        const size = this.context.drawingBufferSize();
        this.usedRegion = this.region ?? { x: 0, y: 0, width: size.width, height: size.height };
        this.light = this.context.player.pose(motion, 0);
        if (this.light) this.context.markChanged(this.light);
        this.enter('settle-before', now);
        ddgi.setBudgetOverride(SETTLE_BUDGET_MS);
    }

    /** Stops a measurement and gives the user's budget back. The light stays where it is. */
    cancel(reason?: string): void {
        if (!this.running) return;
        this.context.ddgi()?.setBudgetOverride(null);
        if (reason) this.fail(reason);
        else this.enter('idle', 0);
    }

    /** Call once a frame before GI's update and the render: moves the light at the step, changes phases. */
    beforeFrame(now: number): void {
        if (!this.running) return;
        const ddgi = this.context.ddgi();
        if (!ddgi || !ddgi.running) {
            this.cancel('DDGI started over during the measurement.');
            return;
        }
        const seconds = (now - this.phaseStartedAt) / 1000;
        switch (this.phaseValue) {
            case 'settle-before':
                if (seconds >= SETTLE_SECONDS) {
                    ddgi.setBudgetOverride(null);
                    this.enter('steady-before', now);
                }
                break;
            case 'steady-before':
                if (seconds >= STEADY_SECONDS && this.beforeReadings.length >= AVERAGED_FRAMES) {
                    // The step: the light jumps to the motion's second keyframe before this frame's GI update.
                    const motion = this.motion!;
                    const light = this.context.player.pose(motion, motion.keyframes[1].t);
                    if (light) this.context.markChanged(light);
                    this.stepStartedAt = now;
                    this.enter('record', now);
                }
                break;
            case 'record':
                if (this.phaseFrames >= RECORD_FRAMES) {
                    ddgi.setBudgetOverride(SETTLE_BUDGET_MS);
                    this.settleFrames = 0;
                    this.enter('settle-after', now);
                }
                break;
            case 'settle-after':
                if (seconds >= SETTLE_SECONDS && this.afterReadings.length >= AVERAGED_FRAMES) {
                    ddgi.setBudgetOverride(null);
                    this.enter('reading', now);
                    void this.finish();
                }
                break;
            default:
                break;
        }
    }

    /** Call once a frame right after the render: reads the frame back when this phase needs it. */
    afterFrame(now: number): void {
        if (!this.running || this.phaseValue === 'reading') return;
        const ddgi = this.context.ddgi();
        if (!ddgi) return;
        this.phaseFrames++;
        const seconds = (now - this.phaseStartedAt) / 1000;
        const region = this.usedRegion;
        const read = () => {
            const reading = ddgi.readRegionMean(region.x, region.y, region.width, region.height);
            if (!reading) this.cancel('the upscaler is on; switch it off to measure.');
            return reading;
        };
        switch (this.phaseValue) {
            case 'steady-before':
                // The last frames before the step (the step comes once STEADY_SECONDS are up).
                if (seconds >= STEADY_SECONDS - 0.5) this.keepLast(this.beforeReadings, read());
                break;
            case 'record': {
                const reading = read();
                if (!reading) return;
                const frame = this.phaseFrames;
                const entry = { frame, timeMs: now - this.stepStartedAt, probesUpdated: ddgi.probesUpdated, mean: reading };
                this.recorded.push(entry);
                void reading.then((mean) => this.live.push({ frame, mean }));
                break;
            }
            case 'settle-after':
                this.settleFrames++;
                if (seconds >= SETTLE_SECONDS - 0.5) this.keepLast(this.afterReadings, read());
                break;
            default:
                break;
        }
    }

    private keepLast(readings: Array<Promise<number>>, reading: Promise<number> | null): void {
        if (!reading) return;
        readings.push(reading);
        if (readings.length > AVERAGED_FRAMES) readings.shift();
    }

    private async finish(): Promise<void> {
        try {
            const mean = (values: number[]) => values.reduce((sum, value) => sum + value, 0) / values.length;
            const before = mean(await Promise.all(this.beforeReadings));
            const settled = mean(await Promise.all(this.afterReadings));
            const samples: ResponseSample[] = [];
            for (const entry of this.recorded) samples.push({ frame: entry.frame, timeMs: entry.timeMs, probesUpdated: entry.probesUpdated, mean: await entry.mean });
            const marks = stepMarks(before, settled, samples);
            const described = this.context.describe();
            const region = this.usedRegion;
            const csv = responseCsv(
                {
                    ...described,
                    motion: this.motion?.id ?? '',
                    region: [region.x, region.y, region.width, region.height],
                    budgetMs: this.userBudgetMs,
                    settleFrames: this.settleFrames,
                    settleBudgetMs: SETTLE_BUDGET_MS,
                },
                marks,
                samples,
            );
            this.resultValue = { samples, marks, csv, motion: this.motion?.id ?? '' };
            this.enter('done', performance.now());
        } catch (error) {
            this.fail(error instanceof Error ? error.message : String(error));
        }
    }

    private enter(phase: MeterPhase, now: number): void {
        this.phaseValue = phase;
        this.phaseStartedAt = now;
        this.phaseFrames = 0;
    }

    private fail(reason: string): void {
        this.failure = reason;
        this.phaseValue = 'failed';
    }
}
