/**
 * The DDGI lines of the stats overlay, kept free of the DOM and of the
 * library so they can be tested. Each line carries the key of its hover help
 * (`ui/help-text.ts`).
 *
 * Numbers that change every frame are smoothed (`SmoothedValue`), as the
 * frame time is: one that rewrites itself 60 times a second can't be read.
 */

import { SmoothedValue } from './format-stats.ts';

/** One DDGI frame as the overlay needs it (a plain copy of `DdgiTechnique.statsSnapshot()`). */
export interface DdgiFrameStats {
    probeCount: number;
    probesUpdated: number;
    /** GPU ms of DDGI's update passes, by short name (`trace`, `blend`, `write`); empty without timestamps. */
    updatePasses: ReadonlyMap<string, number>;
    /** GPU ms of the platform's own passes (`top-level refit`); empty without timestamps. */
    platformPasses: ReadonlyMap<string, number>;
    /** The newest timings are still the warm-up's: keep the averages as they are (the maps are empty). */
    timingsPending?: boolean;
    memory: { sceneCopy: number; acceleration: number; techniques: Readonly<Record<string, number>>; total: number };
    syncCpuMs: number;
    encodeCpuMs: number;
    /** Probes per state, e.g. `{ 'probes active': 812 }`; empty until the GPU census arrives. */
    states: Readonly<Record<string, number>>;
}

export type DdgiStatsHelpKey = 'statsProbes' | 'statsDdgiUpdate' | 'statsRefit' | 'statsMemory' | 'statsSync' | 'statsStates';

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
        const cycle = cycleFrames === null ? '' : `, each every ${cycleFrames} frame${cycleFrames === 1 ? '' : 's'}`;
        lines.push({ text: `  probes updated  ${updated.toLocaleString('en')} of ${frame.probeCount.toLocaleString('en')}${cycle}`, help: 'statsProbes' });

        const passes = [...this.updatePasses.entries()].map(([name, smoothed]) => [name, smoothed.value ?? 0] as const);
        if (passes.length > 0) {
            const total = passes.reduce((sum, [, ms]) => sum + ms, 0);
            const breakdown = passes.map(([name, ms]) => `${name} ${ms.toFixed(2)}`).join(', ');
            lines.push({ text: `  update (GPU)    ${milliseconds(total)}  (${breakdown})`, help: 'statsDdgiUpdate' });
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

        const states = Object.entries(frame.states);
        if (states.length > 0) {
            lines.push({ text: `  ${states.map(([name, count]) => `${name.replace(/^probes /, '')} ${count.toLocaleString('en')}`).join(', ')}`, help: 'statsStates' });
        }
        return lines;
    }
}

/** Adds this frame's value of each name; names missing this frame drop out. */
function smoothInto(target: Map<string, SmoothedValue>, values: ReadonlyMap<string, number>): void {
    for (const name of [...target.keys()]) if (!values.has(name)) target.delete(name);
    for (const [name, value] of values) {
        let smoothed = target.get(name);
        if (!smoothed) {
            smoothed = new SmoothedValue();
            target.set(name, smoothed);
        }
        smoothed.add(value);
    }
}
