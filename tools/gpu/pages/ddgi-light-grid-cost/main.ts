/**
 * T2.3 checks 2 and 3 (docs/design/m2-lights.md, "T2.3 Many-light
 * sampling"), timed: what DDGI's update costs per updated probe, and what
 * the light grid's build costs, on a scene loaded the way the viewer loads
 * it (`startViewerScene`).
 *
 * - **Per updated probe:** fixed batches of SMALL_BATCH and LARGE_BATCH
 *   probes, FRAMES frames each, GPU-bound (two frames in flight). Each
 *   frame's update is the sum of every `ddgi/*` pass; the cost per probe is
 *   the difference of the two batches' medians over the probes between them,
 *   so the fixed part (scheduler, census, the grid) drops out. Check 2
 *   compares `sponza-candles` against `sponza` lit by its sun: run the page
 *   once per scene (VITE_DDGI_SCENES=sponza, then sponza-candles) and divide.
 * - **K:** the batches run once per value in VITE_DDGI_K (default 1,2,4;
 *   `maxShadowedLightsPerHit`), for the noise-against-cost record. On
 *   `sponza` (a sun only) K changes nothing.
 * - **The grid's build** (check 3, ≤ 0.1 ms): `gi/light-grid`'s median over
 *   the same frames. `sponza` has no grid (one sun).
 *
 * Kept short on purpose: the RTX 3060 Laptop throttles after about 20 s of
 * load, and a timing counts only under the clock rule (start below 70 °C,
 * under 5% of samples throttled).
 *
 * Manual (needs the Sponza and candle downloads):
 *   node tools/reference/bench-lock.ts run --what "T2.3 cost" -- pnpm test:gpu --gpu nvidia --filter ddgi-light-grid-cost
 *   with VITE_DDGI_SCENES=sponza-candles (default) or sponza; VITE_DDGI_K=1,2,4;
 *   VITE_DDGI_GRID_CELLS=4096 (the grid's cell cap, for recording cost against cells).
 */

import { loadAllManifests } from '../../../../apps/viewer/src/scene/manifests.ts';
import { LIGHT_GRID_PASS } from '../../../../packages/ddgi-core/src/platform/shade/light-grid.ts';
import { assert } from '../../client.ts';
import { startViewerScene, yieldToCallbacks, type ViewerScene } from '../ddgi-shared/viewer-scene.ts';
import { runThreePage } from '../three-shared/harness.ts';

const env = import.meta.env as Record<string, unknown>;
const SCENE_ID = typeof env.VITE_DDGI_SCENES === 'string' && env.VITE_DDGI_SCENES !== '' ? env.VITE_DDGI_SCENES.split(',')[0].trim() : 'sponza-candles';
/** The light grid's cell cap (`lightGridMaxCells`); the default when unset. */
const GRID_CELLS = typeof env.VITE_DDGI_GRID_CELLS === 'string' && env.VITE_DDGI_GRID_CELLS !== '' ? Number(env.VITE_DDGI_GRID_CELLS) : undefined;
const K_VALUES = typeof env.VITE_DDGI_K === 'string' && env.VITE_DDGI_K !== '' ? env.VITE_DDGI_K.split(',').map(Number) : [1, 2, 4];

const SMALL_BATCH = 64;
const LARGE_BATCH = 512;
const FRAMES = 60;
/** Frames before timing: probes get classified and lists settle. */
const SETTLE_FRAMES = 20;
const FRAMES_IN_FLIGHT = 2;
const UPDATE_PASS_PREFIX = 'ddgi/';

/** Every pass timing seen, by platform frame (the platform keeps only the newest timed frame's). */
class FrameTimings {
    readonly byFrame = new Map<number, Map<string, number>>();
    note(scene: ViewerScene): void {
        for (const pass of scene.platform.stats.passes) {
            if (pass.gpuMs === undefined || !(pass.gpuMs >= 0)) continue;
            let passes = this.byFrame.get(pass.frame);
            if (!passes) {
                passes = new Map();
                this.byFrame.set(pass.frame, passes);
            }
            passes.set(pass.name, pass.gpuMs);
        }
    }
    /** Per frame in [first, last] that was timed: the sum of the passes `select` picks. */
    sums(first: number, last: number, select: (name: string) => boolean): number[] {
        const sums: number[] = [];
        for (let frame = first; frame <= last; frame++) {
            const passes = this.byFrame.get(frame);
            if (!passes) continue;
            let sum = 0;
            let any = false;
            for (const [name, ms] of passes) {
                if (!select(name)) continue;
                sum += ms;
                any = true;
            }
            if (any) sums.push(sum);
        }
        return sums;
    }
}

function median(values: readonly number[]): number {
    const sorted = [...values].sort((a, b) => a - b);
    return sorted.length === 0 ? Number.NaN : sorted[Math.floor(sorted.length / 2)];
}
const round = (value: number, digits = 4) => Number(value.toFixed(digits));

await runThreePage(async ({ device, data, log: logLine }) => {
    const log = (line: string): void => {
        logLine(line);
        console.log(line);
    };
    const manifest = loadAllManifests().find((entry) => entry.id === SCENE_ID);
    assert(manifest, `no manifest ${SCENE_ID}`);
    data.scene = SCENE_ID;
    const scene = await startViewerScene(device, manifest, { gpuTiming: true, label: SCENE_ID });
    try {
        const timings = new FrameTimings();
        const pending: Promise<void>[] = [];
        const run = async (frames: number, units: number): Promise<{ first: number; last: number; hitsLit: number[] }> => {
            const first = scene.platform.stats.frame + 1;
            // The census's per-frame count of hits gi/direct lit (it arrives a frame or two late).
            const hitsLit: number[] = [];
            for (let index = 0; index < frames; index++) {
                scene.frame({ units }, 1 / 60);
                timings.note(scene);
                const lit = scene.ddgi.counters['hits lit by gi/direct'];
                if (index >= FRAMES_IN_FLIGHT + 2 && lit !== undefined && lit > 0) hitsLit.push(lit);
                pending.push(device.queue.onSubmittedWorkDone());
                if (pending.length >= FRAMES_IN_FLIGHT) await pending.shift();
                await yieldToCallbacks();
            }
            await Promise.all(pending.splice(0));
            // Timings arrive a few frames late: frames with no work let the last ones land.
            for (let extra = 0; extra < 6; extra++) {
                scene.frame({ paused: true }, 0);
                timings.note(scene);
                await device.queue.onSubmittedWorkDone();
                await yieldToCallbacks();
            }
            return { first, last: first + frames - 1, hitsLit };
        };

        if (GRID_CELLS !== undefined) scene.platform.shade.settings.lightGridMaxCells = GRID_CELLS;
        data.lightGridMaxCells = scene.platform.shade.settings.lightGridMaxCells;
        await run(SETTLE_FRAMES, LARGE_BATCH);
        const byK: Record<string, unknown> = {};
        for (const k of K_VALUES) {
            scene.platform.shade.settings.lightGridShadowedLightsPerHit = k;
            const small = await run(FRAMES, SMALL_BATCH);
            const large = await run(FRAMES, LARGE_BATCH);
            const update = (range: { first: number; last: number }) => timings.sums(range.first, range.last, (name) => name.startsWith(UPDATE_PASS_PREFIX));
            const pass = (range: { first: number; last: number }, wanted: string) => timings.sums(range.first, range.last, (name) => name === wanted);
            const smallMs = median(update(small));
            const largeMs = median(update(large));
            const usPerProbe = ((largeMs - smallMs) / (LARGE_BATCH - SMALL_BATCH)) * 1000;
            const grid = [...pass(small, LIGHT_GRID_PASS), ...pass(large, LIGHT_GRID_PASS)];
            const entry = {
                usPerProbe: round(usPerProbe, 3),
                updateMs: { [SMALL_BATCH]: round(smallMs), [LARGE_BATCH]: round(largeMs) },
                traceMs: { [SMALL_BATCH]: round(median(pass(small, 'ddgi/trace'))), [LARGE_BATCH]: round(median(pass(large, 'ddgi/trace'))) },
                directMs: { [SMALL_BATCH]: round(median(pass(small, 'ddgi/gi-direct'))), [LARGE_BATCH]: round(median(pass(large, 'ddgi/gi-direct'))) },
                blendMs: { [SMALL_BATCH]: round(median(pass(small, 'ddgi/blend'))), [LARGE_BATCH]: round(median(pass(large, 'ddgi/blend'))) },
                lightGridMs: grid.length > 0 ? round(median(grid)) : null,
                lightGridFrames: grid.length,
                timedFrames: update(small).length + update(large).length,
                hitsLitPerProbe: round(median(large.hitsLit) / LARGE_BATCH, 2),
            };
            byK[`K${k}`] = entry;
            log(
                `${SCENE_ID}, K = ${k}: ${entry.usPerProbe} µs per updated probe (update ${entry.updateMs[SMALL_BATCH]} ms at ${SMALL_BATCH}, ${entry.updateMs[LARGE_BATCH]} ms at ${LARGE_BATCH}; ` +
                    `trace ${entry.traceMs[LARGE_BATCH]}, gi-direct ${entry.directMs[LARGE_BATCH]}, blend ${entry.blendMs[LARGE_BATCH]} at ${LARGE_BATCH}); ` +
                    `gi/light-grid ${entry.lightGridMs ?? 'not built'} ms over ${entry.lightGridFrames} frames; ${entry.hitsLitPerProbe} hits lit a probe`,
            );
        }
        data.byK = byK;
        data.lights = scene.platform.sceneCopy.counts.lights;
        data.localLights = scene.platform.sceneCopy.localLightCount;
    } finally {
        scene.dispose();
    }
});
