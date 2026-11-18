/**
 * The stats line at the bottom of each example: how many probes GI has, what
 * a GI frame costs on the GPU, the memory it holds, and how much light the
 * probes hold. The same numbers go to `window.ddgiExample` (status.ts).
 *
 * Everything comes from the library's public API: `volume.platform.stats`
 * for frames, GPU time and memory, and `exportIrradianceSH` for the light in
 * the probes. That export is one small GPU readback, so it runs only every
 * few seconds, never every frame.
 */

import { type GIPlatform } from '@pmndrs/gi/core';
import { DDGITechnique, exportIrradianceSH, PROBE_STATE } from '@pmndrs/gi/ddgi';
import type { DDGIVolume } from '@pmndrs/gi/three';
import { status } from './status.ts';

const TEXT_INTERVAL_MS = 500;
const GI_MEAN_INTERVAL_MS = 2000;

/**
 * The mean, over all directions, of irradiance from a probe's L2 SH, per
 * unit of its first coefficient: π × Y₀₀ (three's `getShIrradianceAt`
 * weights the constant term by this; the other terms average to zero).
 */
const SH_CONSTANT_TO_MEAN_IRRADIANCE = Math.PI * 0.282095;

/** Rec. 709 luminance of linear RGB. */
function luminance(red: number, green: number, blue: number): number {
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue;
}

/** The mean irradiance of the probes GI uses (state `active`), as luminance, or null before any is active. */
async function measureGIMean(technique: DDGITechnique): Promise<number | null> {
    const exported = await exportIrradianceSH(technique, { fillInactiveProbes: false });
    const floatsPerProbe = exported.coefficients.length / exported.probeStates.length;
    let sum = 0;
    let activeProbes = 0;
    for (let probe = 0; probe < exported.probeStates.length; probe++) {
        if (exported.probeStates[probe] !== PROBE_STATE.active) continue;
        // The constant term's red, green and blue come first in each probe's coefficients.
        const first = probe * floatsPerProbe;
        sum += luminance(exported.coefficients[first], exported.coefficients[first + 1], exported.coefficients[first + 2]) * SH_CONSTANT_TO_MEAN_IRRADIANCE;
        activeProbes++;
    }
    return activeProbes > 0 ? sum / activeProbes : null;
}

/** GPU milliseconds of the latest timed GI frame, or null without timestamp queries. */
function giMilliseconds(platform: GIPlatform): number | null {
    const passes = platform.stats.passes.filter((pass) => pass.gpuMs !== undefined);
    if (passes.length === 0) return null;
    return passes.reduce((total, pass) => total + (pass.gpuMs ?? 0), 0);
}

export interface StatsOptions {
    /** Extra text for the end of the line (what the example is showing right now). */
    readonly note?: () => string;
}

/**
 * Shows the stats line for `volume` and keeps it current. Pass a function
 * instead when the example swaps volumes (the presets example): the line
 * follows whichever volume it returns.
 */
export function showStats(volume: DDGIVolume | (() => DDGIVolume), options: StatsOptions = {}): void {
    const currentVolume = typeof volume === 'function' ? volume : () => volume;
    const line = document.createElement('div');
    line.id = 'stats';
    line.style.cssText =
        'position:fixed;left:0;bottom:0;padding:5px 10px;z-index:10;background:rgba(8,10,14,0.75);' +
        'color:#cfd8e3;font:12px/1.5 ui-monospace,Menlo,Consolas,monospace;pointer-events:none;';
    document.body.append(line);

    // Frames run on platforms shown before (a new start counts from 0 again).
    let framesBefore = 0;
    let shownPlatform: GIPlatform | null = null;
    const update = (): void => {
        const platform = currentVolume().platform;
        if (platform !== shownPlatform) {
            if (shownPlatform) framesBefore += shownPlatform.stats.frame;
            shownPlatform = platform;
        }
        if (!platform) {
            line.textContent = 'GI starting…';
            return;
        }
        const stats = platform.stats;
        status.frames = framesBefore + stats.frame;
        const technique = platform.outputTechnique;
        const probes = technique instanceof DDGITechnique ? (technique.field?.grid.probeCount ?? 0) : 0;
        const milliseconds = giMilliseconds(platform);
        const parts = [
            `GI: ${probes.toLocaleString('en')} probes`,
            milliseconds === null ? 'GPU time not measured' : `${milliseconds.toFixed(2)} ms GPU`,
            `${(stats.memory.total / 2 ** 20).toFixed(0)} MB`,
            status.giMean === null ? 'mean irradiance …' : `mean irradiance ${status.giMean.toFixed(3)}`,
            `frame ${status.frames.toLocaleString('en')}`,
        ];
        const note = options.note?.();
        if (note) parts.push(note);
        line.textContent = parts.join(' · ');
    };
    setInterval(update, TEXT_INTERVAL_MS);
    update();

    let measuring = false;
    setInterval(() => {
        const technique = currentVolume().platform?.outputTechnique;
        if (measuring || !(technique instanceof DDGITechnique) || !technique.field) return;
        measuring = true;
        measureGIMean(technique)
            .then((mean) => {
                status.giMean = mean;
            })
            .finally(() => {
                measuring = false;
            });
    }, GI_MEAN_INTERVAL_MS);
}
