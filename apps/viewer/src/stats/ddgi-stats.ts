/**
 * The DDGI lines of the stats overlay, kept free of the DOM and of the
 * library so they can be tested. Each line carries the key of the help text
 * that explains it (`ui/help-text.ts`, listed in the panel). Frames are added
 * every frame; the lines are built only when the overlay redraws.
 *
 * Numbers that change every frame are smoothed (`SmoothedValue`), as the
 * frame time is: one that rewrites itself 60 times a second can't be read.
 */

import { SmoothedValue } from './format-stats.ts';

/**
 * How the platform chose DDGI's probes per frame (budget v2), from its
 * `ddgi budget` counters.
 */
export interface DdgiBudgetStats {
    /** Measuring the GPU right now (start-up, or after the budget or the scene changed). */
    calibrating: boolean;
    /** The probes the budget buys with the GPU's clocks up; null when nothing was measured (no GPU timestamps). */
    calibratedProbes: number | null;
    /** The fewest probes per frame (a full sweep within 4 s, at most twice the calibrated number); 0 when off. */
    floorProbes: number;
    /** Fewer than calibrated: frames were late while the GPU was busy. */
    steppedDown: boolean;
    /** The scene's shape changed since it was measured (the number stays until it is measured again). */
    stale: boolean;
    /** The measured cost: a fixed part and a part per probe; null until measured. */
    fixedMs: number | null;
    usPerProbe: number | null;
}

/** One DDGI frame as the overlay needs it (the shape of `DdgiTechnique.statsSnapshot()`). */
export interface DdgiFrameStats {
    probeCount: number;
    probesUpdated: number;
    /** Seconds until every probe has been updated once at this rate; null until known. */
    sweepSeconds?: number | null;
    /** Null before the platform reports how it chose the number. */
    budget?: DdgiBudgetStats | null;
    /** The update budget the user chose, GPU ms per frame. */
    budgetMs: number;
    /** GPU ms of DDGI's passes, by short name (`trace`, `blend`, `write`, `census`); empty without timestamps. */
    updatePasses: ReadonlyMap<string, number>;
    /** GPU ms of the platform's own passes (`top-level refit`); empty without timestamps. */
    platformPasses: ReadonlyMap<string, number>;
    /** The newest timings are still the warm-up's: keep the averages as they are (the maps are empty). */
    timingsPending?: boolean;
    memory: { sceneCopy: number; acceleration: number; techniques: Readonly<Record<string, number>>; total: number };
    syncCpuMs: number;
    encodeCpuMs: number;
    /** Probes per state, in state order (`active 812`); empty until the GPU census arrives. */
    states: ReadonlyArray<{ readonly label: string; readonly count: number }>;
}

export type DdgiStatsHelpKey = 'statsProbes' | 'statsBudget' | 'statsDdgiUpdate' | 'statsRefit' | 'statsMemory' | 'statsSync' | 'statsStates';

export interface StatsLine {
    text: string;
    help: DdgiStatsHelpKey;
}

function megabytes(bytes: number): string {
    return `${(bytes / 1e6).toFixed(bytes < 1e7 ? 1 : 0)} MB`;
}

function milliseconds(ms: number): string {
    return `${ms.toFixed(2)} ms`;
}

function seconds(value: number): string {
    return value < 10 ? `${value.toFixed(1)} s` : `${Math.round(value)} s`;
}

/** Why DDGI updates this many probes a frame (budget v2), and the cost it measured. */
function budgetLines(frame: DdgiFrameStats): StatsLine[] {
    const budget = frame.budget;
    if (!budget) return [];
    const probes = (count: number) => `${count.toLocaleString('en')} probe${count === 1 ? '' : 's'}`;
    const buys = (count: number) => `${frame.budgetMs.toFixed(2)} ms buys ${count.toLocaleString('en')} at full GPU speed`;
    let reason: string;
    if (budget.calibrating) {
        reason = 'measuring the GPU...';
    } else if (budget.calibratedProbes === null) {
        reason = `${probes(frame.probesUpdated)} (no GPU timestamps to measure with)`;
    } else if (budget.steppedDown) {
        reason = `${probes(frame.probesUpdated)}: slowed, frames are late with the GPU busy (${buys(budget.calibratedProbes)})`;
    } else if (budget.floorProbes > budget.calibratedProbes) {
        reason = `${probes(budget.floorProbes)}: the least for a full sweep in 4 s (${buys(budget.calibratedProbes)})`;
    } else {
        const floor = budget.floorProbes > 0 ? `, at least ${budget.floorProbes.toLocaleString('en')}` : '';
        reason = `${probes(budget.calibratedProbes)}: what ${frame.budgetMs.toFixed(2)} ms buys at full GPU speed${floor}`;
    }
    const lines: StatsLine[] = [{ text: `  per frame       ${reason}`, help: 'statsBudget' }];
    if (!budget.calibrating && budget.fixedMs !== null && budget.usPerProbe !== null) {
        const stale = budget.stale ? ' (the scene changed since; Start the probes over to measure again)' : '';
        lines.push({ text: `  measured cost   ${milliseconds(budget.fixedMs)} + ${budget.usPerProbe.toFixed(1)} µs per probe${stale}`, help: 'statsBudget' });
    }
    return lines;
}

/** Smooths each number over frames, and turns the result into the overlay's lines. */
export class DdgiStatsText {
    private readonly probesUpdated = new SmoothedValue();
    private readonly updatePasses = new Map<string, SmoothedValue>();
    private readonly platformPasses = new Map<string, SmoothedValue>();
    private readonly syncCpu = new SmoothedValue();
    private readonly encodeCpu = new SmoothedValue();
    private latest: DdgiFrameStats | null = null;

    /** Call once per frame drawn with DDGI on screen. */
    add(frame: DdgiFrameStats): void {
        this.latest = frame;
        this.probesUpdated.add(frame.probesUpdated);
        this.syncCpu.add(frame.syncCpuMs);
        this.encodeCpu.add(frame.encodeCpuMs);
        if (frame.timingsPending) return;
        smoothInto(this.updatePasses, frame.updatePasses);
        smoothInto(this.platformPasses, frame.platformPasses);
    }

    /** Forget everything (a new scene, or DDGI left the screen). */
    reset(): void {
        this.latest = null;
        this.probesUpdated.reset();
        this.syncCpu.reset();
        this.encodeCpu.reset();
        this.updatePasses.clear();
        this.platformPasses.clear();
    }

    lines(): StatsLine[] {
        const frame = this.latest;
        if (!frame) return [];
        const lines: StatsLine[] = [{ text: 'DDGI (ours):', help: 'statsProbes' }];

        const updated = Math.round(this.probesUpdated.value ?? frame.probesUpdated);
        const cycleFrames = updated > 0 ? Math.ceil(frame.probeCount / updated) : null;
        let cycle = '';
        if (frame.sweepSeconds !== undefined && frame.sweepSeconds !== null) cycle = `, every probe every ${seconds(frame.sweepSeconds)}`;
        else if (cycleFrames !== null) cycle = `, each every ${cycleFrames} frame${cycleFrames === 1 ? '' : 's'}`;
        lines.push({ text: `  probes updated  ${updated.toLocaleString('en')} of ${frame.probeCount.toLocaleString('en')}${cycle}`, help: 'statsProbes' });
        lines.push(...budgetLines(frame));

        const passes = [...this.updatePasses.entries()].map(([name, smoothed]) => [name, smoothed.value ?? 0] as const);
        if (passes.length > 0) {
            const total = passes.reduce((sum, [, ms]) => sum + ms, 0);
            const breakdown = passes.map(([name, ms]) => `${name} ${ms.toFixed(2)}`).join(', ');
            // The budget beside the number, so an overshoot shows.
            lines.push({ text: `  update (GPU)    ${milliseconds(total)} (budget ${frame.budgetMs.toFixed(2)})  ${breakdown}`, help: 'statsDdgiUpdate' });
        } else if (frame.timingsPending) {
            lines.push({ text: '  update (GPU)    measuring...', help: 'statsDdgiUpdate' });
        } else {
            lines.push({ text: '  update (GPU)    - (no GPU timestamps)', help: 'statsDdgiUpdate' });
        }
        for (const [name, smoothed] of this.platformPasses) {
            lines.push({ text: `  ${name} (GPU)  ${milliseconds(smoothed.value ?? 0)}`, help: 'statsRefit' });
        }
        // The platform refits its top-level tree only in frames where something moved.
        if (this.platformPasses.size === 0 && passes.length > 0) lines.push({ text: '  top-level refit (GPU)  none (nothing moved)', help: 'statsRefit' });

        const ddgiBytes = Object.values(frame.memory.techniques).reduce((sum, bytes) => sum + bytes, 0);
        lines.push({
            text: `  GPU memory      ${megabytes(frame.memory.total)}  (scene ${megabytes(frame.memory.sceneCopy)}, ray tracing ${megabytes(frame.memory.acceleration)}, probes ${megabytes(ddgiBytes)})`,
            help: 'statsMemory',
        });
        lines.push({
            text: `  adapter (CPU)   sync ${milliseconds(this.syncCpu.value ?? frame.syncCpuMs)}, record ${milliseconds(this.encodeCpu.value ?? frame.encodeCpuMs)}`,
            help: 'statsSync',
        });

        if (frame.states.length > 0) {
            lines.push({ text: `  ${frame.states.map((state) => `${state.label} ${state.count.toLocaleString('en')}`).join(', ')}`, help: 'statsStates' });
        }
        return lines;
    }
}

/** Adds this frame's value of each name; names missing this frame drop out. */
function smoothInto(target: Map<string, SmoothedValue>, values: ReadonlyMap<string, number>): void {
    // Deleting the entry being visited is safe in a Map iteration (and copies nothing).
    for (const name of target.keys()) if (!values.has(name)) target.delete(name);
    for (const [name, value] of values) {
        let smoothed = target.get(name);
        if (!smoothed) {
            smoothed = new SmoothedValue();
            target.set(name, smoothed);
        }
        smoothed.add(value);
    }
}
