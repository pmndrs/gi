/**
 * T5.1 (docs/design/m5-coverage.md §1): what each quality preset costs and
 * what it looks like, on Cornell and two-room. Numbers are recorded, not
 * gated (DECISIONS 2026-11-10, "Feature coverage before tuning"); the
 * presets' correctness checks are `ddgi-furnace`, `ddgi-leak` and
 * `platform-recompiles` run with `VITE_DDGI_PRESET`.
 *
 * Each scene, each preset, on a platform of its own made with the preset's
 * texture store size and DDGI with the preset's rays and tile sizes, on the
 * manifest's volume (a host that sets its spacing; what the preset's own
 * spacing would give is recorded beside it):
 *
 * 1. CONVERGE_FRAMES frames, every probe every frame, so the field is close
 *    to settled;
 * 2. TIMED_FRAMES more, GPU-bound, every probe every frame, timed: the
 *    update's GPU milliseconds per frame (every `ddgi/` pass but the census,
 *    the median frame), and per 100 probes;
 * 3. the engine-free test view's indirect mean at the scene's view, and how
 *    far it is from high's (the look's cost, as one number);
 * 4. `gi.stats.memory`: DDGI's own bytes, the texture store's, the total.
 *
 * Fails only if a preset doesn't run, its picture is empty or not a number,
 * or the platform compiled anything after `prepare` (recompiles must stay 0
 * within a preset).
 */

import { assert, runGpuTest } from '../../client.ts';
import { DDGI_PRESET_NAMES, ddgiPreset, type DDGIPresetName } from '../../../../packages/ddgi-core/src/techniques/ddgi/presets.ts';
import { effectiveRaysPerProbe, probeGrid } from '../../../../packages/ddgi-core/src/techniques/ddgi/settings.ts';
import { createDdgiScene, frameInput, imageMean, releaseDeviceOnPageHide, runFrames, TestView, type DdgiScene } from '../ddgi-shared/harness.ts';
import { everyProbeEveryFrame } from '../ddgi-shared/page-preset.ts';

const env = import.meta.env as Record<string, unknown>;

/** Frames before the reading: 10 s of DDGI's clock at 60 Hz, every probe every frame. Close to settled, not settled (the furnace needs 1800). */
const CONVERGE_FRAMES = Number(env.VITE_DDGI_FRAMES) > 0 ? Math.round(Number(env.VITE_DDGI_FRAMES)) : 600;
/** Timed frames for the update's cost. */
const TIMED_FRAMES = 60;
/** Frames queued at once while timing, so the GPU never waits for the CPU. */
const FRAMES_IN_FLIGHT = 2;
const WIDTH = 320;
const HEIGHT = 180;

interface PresetCase {
    readonly scene: string;
    /** The bookmark the test view reads. */
    readonly camera: string;
}

const CASES: readonly PresetCase[] = [
    { scene: 'cornell-box', camera: 'front' },
    { scene: 'two-room', camera: 'lit-room' },
];

const MEGABYTE = 1_000_000;

function megabytes(bytes: number): number {
    return Number((bytes / MEGABYTE).toFixed(3));
}

function median(values: readonly number[]): number | null {
    if (values.length === 0) return null;
    const sorted = [...values].sort((a, b) => a - b);
    const middle = Math.floor(sorted.length / 2);
    return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
}

/** Lets timestamp readbacks land between frames. */
function nextTask(): Promise<void> {
    return new Promise((resolve) => setTimeout(resolve, 0));
}

/** The update's passes: DDGI's own, but the census, which runs whatever the batch. */
function isUpdatePass(name: string): boolean {
    return name.startsWith('ddgi/') && name !== 'ddgi/census';
}

/** `frames` frames of every probe, GPU-bound; each timed frame's update ms (its `ddgi/` passes summed). */
async function timedUpdateMs(device: GPUDevice, scene: DdgiScene, units: number, frames: number): Promise<number[]> {
    const byFrame = new Map<number, number>();
    const collect = (): void => {
        const passes = scene.platform.stats.passes.filter((pass) => isUpdatePass(pass.name) && pass.gpuMs !== undefined && pass.gpuMs >= 0);
        if (passes.length === 0) return;
        const frame = passes[0].frame;
        if (byFrame.has(frame)) return;
        byFrame.set(
            frame,
            passes.filter((pass) => pass.frame === frame).reduce((sum, pass) => sum + (pass.gpuMs ?? 0), 0),
        );
    };
    const pending: Promise<void>[] = [];
    for (let index = 0; index < frames; index++) {
        const encoder = device.createCommandEncoder({ label: 'ddgi-presets frame' });
        scene.platform.encode(encoder, frameInput(units));
        device.queue.submit([encoder.finish()]);
        pending.push(device.queue.onSubmittedWorkDone());
        if (pending.length >= FRAMES_IN_FLIGHT) await pending.shift();
        await nextTask();
        collect();
    }
    await Promise.all(pending);
    // Paused frames collect the last timed frames' readbacks.
    for (let extra = 0; extra < 4; extra++) {
        const encoder = device.createCommandEncoder({ label: 'ddgi-presets readback frame' });
        scene.platform.encode(encoder, frameInput({ paused: true }));
        device.queue.submit([encoder.finish()]);
        await device.queue.onSubmittedWorkDone();
        await nextTask();
        collect();
    }
    return [...byFrame.values()];
}

interface PresetResult {
    readonly probes: number;
    /** Probes the preset's own spacing would give on the same box (a host that leaves the spacing to the preset). */
    readonly probesAtPresetSpacing: number;
    readonly raysPerProbe: number;
    readonly raysPerFrame: number;
    readonly distanceTexels: number;
    readonly irradianceTexels: number;
    readonly textureLayerSize: number;
    readonly budgetGpuMs: number;
    readonly ddgiMemoryMB: number;
    readonly textureStoreMB: number;
    readonly totalMemoryMB: number;
    /** Median GPU ms of an update of every probe; null without timestamps. */
    readonly updateMs: number | null;
    readonly updateMsPer100Probes: number | null;
    readonly timedFrames: number;
    readonly indirectMean: number;
    /** The indirect mean against high's on the same scene and view. */
    readonly meanAgainstHigh?: number;
    readonly recompiles: number;
}

async function runPreset(device: GPUDevice, presetCase: PresetCase, name: DDGIPresetName, log: (line: string) => void): Promise<PresetResult> {
    // Every probe every frame, so the default batch cap, not low's 1,024 (page-preset.ts).
    const preset = everyProbeEveryFrame(ddgiPreset(name));
    const scene = await createDdgiScene(device, presetCase.scene, { ...preset.ddgi }, { gpuTiming: true, textureLayerSize: preset.textureLayerSize });
    try {
        const field = scene.ddgi.field;
        assert(field, `${presetCase.scene} (${name}): no field after prepare`);
        const probes = field.grid.probeCount;
        await runFrames(device, scene.platform, CONVERGE_FRAMES, probes);
        const timed = await timedUpdateMs(device, scene, probes, TIMED_FRAMES);
        const updateMs = median(timed);
        // Read before the test view, whose own kernel the counter rightly counts (a page's tool, not DDGI's).
        const recompiles = scene.platform.stats.recompiles;
        assert(recompiles === 0, `${presetCase.scene} (${name}): ${recompiles} pipelines were compiled after prepare`);

        const camera = scene.manifest.cameras.find((bookmark) => bookmark.name === presetCase.camera);
        assert(camera, `${presetCase.scene} has no camera '${presetCase.camera}'`);
        const view = await TestView.create(device, scene.platform);
        const indirectMean = imageMean((await view.render(field, camera, WIDTH, HEIGHT)).indirect);
        assert(Number.isFinite(indirectMean) && indirectMean > 0, `${presetCase.scene} (${name}): the test view's indirect mean is ${indirectMean}`);

        const { memory } = scene.platform.stats;
        const ddgiBytes = memory.techniques[scene.ddgi.id] ?? 0;
        const volume = scene.manifest.volume;
        const raysPerProbe = effectiveRaysPerProbe(preset.ddgi.raysPerProbe);
        const result: PresetResult = {
            probes,
            probesAtPresetSpacing: probeGrid({ center: volume.center, size: volume.size, spacing: preset.probeSpacing }).probeCount,
            raysPerProbe,
            raysPerFrame: probes * raysPerProbe,
            distanceTexels: preset.ddgi.distanceTexels,
            irradianceTexels: preset.ddgi.irradianceTexels,
            textureLayerSize: preset.textureLayerSize,
            budgetGpuMs: preset.budget.gpuMs,
            ddgiMemoryMB: megabytes(ddgiBytes),
            textureStoreMB: megabytes(memory.textures ?? 0),
            totalMemoryMB: megabytes(memory.total),
            updateMs: updateMs === null ? null : Number(updateMs.toFixed(4)),
            updateMsPer100Probes: updateMs === null ? null : Number(((updateMs / probes) * 100).toFixed(4)),
            timedFrames: timed.length,
            indirectMean,
            recompiles,
        };
        log(
            `${presetCase.scene} ${name}: ${probes} probes x ${raysPerProbe} rays, distance ${preset.ddgi.distanceTexels}², store ${preset.textureLayerSize}²; ` +
                `DDGI ${result.ddgiMemoryMB} MB, textures ${result.textureStoreMB} MB, total ${result.totalMemoryMB} MB; ` +
                `update ${updateMs === null ? 'untimed' : `${updateMs.toFixed(3)} ms`}; indirect mean ${indirectMean.toFixed(5)}`,
        );
        return result;
    } finally {
        scene.platform.dispose();
    }
}

await runGpuTest(
    async ({ device, log: logLine }) => {
        releaseDeviceOnPageHide(device);
        const log = (line: string): void => {
            logLine(line);
            console.log(line);
        };
        const results: Record<string, Record<string, PresetResult>> = {};
        for (const presetCase of CASES) {
            const byPreset: Record<string, PresetResult> = {};
            for (const name of DDGI_PRESET_NAMES) byPreset[name] = await runPreset(device, presetCase, name, log);
            const high = byPreset.high.indirectMean;
            for (const name of DDGI_PRESET_NAMES) {
                const against = byPreset[name].indirectMean / high - 1;
                byPreset[name] = { ...byPreset[name], meanAgainstHigh: Number(against.toFixed(5)) };
            }
            log(
                `${presetCase.scene}: indirect mean against high: ` +
                    DDGI_PRESET_NAMES.map((name) => `${name} ${((byPreset[name].meanAgainstHigh ?? 0) * 100).toFixed(2)}%`).join(', '),
            );
            results[presetCase.scene] = byPreset;
        }
        return { convergeFrames: CONVERGE_FRAMES, timedFrames: TIMED_FRAMES, view: `${WIDTH}x${HEIGHT}`, results };
    },
    { optionalFeatures: ['texture-formats-tier1', 'subgroups', 'timestamp-query'] },
);
