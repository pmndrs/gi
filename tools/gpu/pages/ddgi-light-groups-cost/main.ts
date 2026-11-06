/**
 * T2.5 checks 2 and 3, the cost (docs/design/m2-lights.md §5, "Cost
 * estimate"): DDGI's update per updated probe with light groups off, 2 and 4
 * groups, on two-room and Sponza through the three adapter (`loadScene`, as
 * the viewer loads them). A timing page: run it alone under the bench lock,
 * which judges the clock (VALID / INVALID).
 *
 * - Every config runs on the same scene and platform: the technique's
 *   `lightGroups` is set and `ddgi.prepare()` runs again (kernels come from
 *   the device's cache, the scene copy and trees stay), then WARM_SWEEPS
 *   passes over every probe (so classification has switched probes off as
 *   it will), then fixed batches of BATCHES probes, FRAMES_PER_BATCH frames
 *   each, GPU-bound (two frames in flight, as Budget v2 measures).
 * - The configs run in ROUNDS rounds, each in a turned order (off, 2, 4 then
 *   4, 2, off), because the second of two back-to-back timings on this
 *   power-limited laptop runs a few percent slow (NOTES, B3).
 * - The update is every `ddgi/` pass the platform counts against DDGI's
 *   budget (mark, schedule, trace, gi-direct, blend, write, propagate,
 *   compose), not the census. Per config and batch: the median over every
 *   timed frame of every round, and that per probe.
 * - The scene's own lights stay in group 0; three more point lamps are added
 *   in groups 1, 2 and 3 (shadowed in GI), for every config alike, so with 4
 *   groups every layer has a light of its own and gi/direct writes into each.
 *
 * Check 3: with 4 groups the update per probe is at most 1.4× groups off,
 * at the 512-probe batch (near Budget v2's calibrated batch on Sponza). The
 * page asserts it; it counts only from a VALID run. Check 2 (groups off
 * within 3% of `m2`) compares this page's groups-off numbers with a run of
 * the same page on `m2`'s code (`VITE_DDGI_LIGHT_GROUPS=0`), which it
 * supports: with 0 alone it never touches `lightGroups`.
 *
 * Also reported: DDGI's GPU memory per config.
 *
 * Choose with environment variables:
 *   VITE_DDGI_SCENES=two-room,sponza     (default two-room,sponza)
 *   VITE_DDGI_LIGHT_GROUPS=0,2,4         (default 0,2,4; 0 is groups off)
 *
 * Manual (Sponza needs the download), under the bench lock:
 *   node tools/reference/bench-lock.ts run --what "T2.5 cost" -- pnpm test:gpu --filter ddgi-light-groups-cost --gpu nvidia --json <file>
 */

import { PointLight } from 'three/webgpu';
import { assert } from '../../client.ts';
import { loadAllManifests } from '../../../../apps/viewer/src/scene/manifests.ts';
import { startViewerScene, yieldToCallbacks, type ViewerScene } from '../ddgi-shared/viewer-scene.ts';
import { runThreePage } from '../three-shared/harness.ts';

const env = import.meta.env as Record<string, unknown>;

function listFromEnv(value: unknown, fallback: readonly string[]): string[] {
    return typeof value === 'string' && value.trim() !== '' ? value.split(',').map((item) => item.trim()) : [...fallback];
}

const SCENES = listFromEnv(env.VITE_DDGI_SCENES, ['two-room', 'sponza']);
/** Light-group counts to time; 0 is groups off. */
const CONFIGS = listFromEnv(env.VITE_DDGI_LIGHT_GROUPS, ['0', '2', '4']).map(Number);
const BATCHES = [256, 512] as const;
/** The batch check 3 is judged at. */
const JUDGED_BATCH = 512;
const FRAMES_PER_BATCH = 60;
const ROUNDS = 2;
const WARM_SWEEPS = 3;
/** Check 3's bar: 4 groups at most this times groups off, per updated probe. */
const FOUR_GROUPS_BAR = 1.4;
/** Frames kept queued, so the GPU never waits for the CPU between them. */
const FRAMES_IN_FLIGHT = 2;

/** Three lamps in groups 1-3 per scene (metres, three's units: candela). */
const EXTRA_LAMPS: Readonly<Record<string, readonly { position: [number, number, number]; color: number; intensity: number; distance: number }[]>> = {
    'two-room': [
        { position: [6.2, 2.45, 2.0], color: 0xbfd9ff, intensity: 8, distance: 14 },
        { position: [1.0, 1.0, 1.0], color: 0xffd0a0, intensity: 3, distance: 10 },
        { position: [7.5, 1.0, 3.0], color: 0xd0ffd0, intensity: 3, distance: 10 },
    ],
    sponza: [
        { position: [3, 1.5, 0.5], color: 0xffd0a0, intensity: 3, distance: 8 },
        { position: [-3, 1.5, 0.5], color: 0xbfd9ff, intensity: 3, distance: 8 },
        { position: [0, 4, -0.5], color: 0xd0ffd0, intensity: 3, distance: 8 },
    ],
};

/** The passes DDGI's budget counts, the census aside. */
function isUpdatePass(name: string): boolean {
    return name.startsWith('ddgi/') && name !== 'ddgi/census';
}

function median(values: readonly number[]): number {
    const sorted = [...values].sort((a, b) => a - b);
    const middle = Math.floor(sorted.length / 2);
    return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
}

/** Runs `frames` frames of `units` probes GPU-bound; returns each timed frame's update ms (every update pass summed). */
async function timedFrames(scene: ViewerScene, units: number, frames: number): Promise<number[]> {
    const byFrame = new Map<number, number>();
    const pending: Promise<void>[] = [];
    const collect = (): void => {
        const passes = scene.platform.stats.passes.filter((pass) => isUpdatePass(pass.name) && pass.gpuMs !== undefined && pass.gpuMs >= 0);
        if (passes.length === 0) return;
        // A frame's passes arrive together: the newest timed frame only.
        const frame = passes[0].frame;
        byFrame.set(frame, passes.filter((pass) => pass.frame === frame).reduce((sum, pass) => sum + (pass.gpuMs ?? 0), 0));
    };
    for (let index = 0; index < frames; index++) {
        scene.frame({ units }, 0);
        pending.push(scene.device.queue.onSubmittedWorkDone());
        if (pending.length >= FRAMES_IN_FLIGHT) await pending.shift();
        await yieldToCallbacks();
        collect();
    }
    await Promise.all(pending);
    for (let extra = 0; extra < 4; extra++) {
        scene.frame({ paused: true }, 0);
        await scene.device.queue.onSubmittedWorkDone();
        await yieldToCallbacks();
        collect();
    }
    return [...byFrame.values()];
}

/** Light groups for a config: off (null) or `count` groups at weight 1. Written as plain data, so the page also runs on code without light groups (count 0 only). */
function lightGroupsFor(count: number): unknown {
    if (count === 0) return null;
    return { count, weights: [1, 2, 3, 4].map(() => [1, 1, 1]), environmentWeight: [1, 1, 1] };
}

await runThreePage(async ({ device, data, log: logLine }) => {
    const log = (line: string): void => {
        logLine(line);
        console.log(line);
    };
    assert(device.features.has('timestamp-query'), 'timing needs the timestamp-query feature');
    const manifests = loadAllManifests();
    const results = data;
    Object.assign(results, { batches: BATCHES, framesPerBatch: FRAMES_PER_BATCH, rounds: ROUNDS, configs: CONFIGS });
    const failures: string[] = [];
    for (const sceneId of SCENES) {
        const manifest = manifests.find((candidate) => candidate.id === sceneId);
        assert(manifest, `no manifest '${sceneId}'`);
        const scene = await startViewerScene(device, manifest, {
            gpuTiming: true,
            label: `light groups cost ${sceneId}`,
            prepareScene: (loaded) => {
                (EXTRA_LAMPS[sceneId] ?? []).forEach((lamp, index) => {
                    const light = new PointLight(lamp.color, lamp.intensity, lamp.distance);
                    light.position.set(...lamp.position);
                    light.userData.giLightGroup = index + 1;
                    // Shadowed in GI, with no shadow map in three.
                    light.userData.giCastsShadows = true;
                    loaded.scene.add(light);
                });
            },
        });
        const settings = scene.ddgi.settings as unknown as Record<string, unknown>;
        const times = new Map<string, number[]>();
        const memory: Record<string, number> = {};
        for (let round = 0; round < ROUNDS; round++) {
            const order = round % 2 === 0 ? CONFIGS : [...CONFIGS].reverse();
            for (const groups of order) {
                if (groups !== 0 || settings.lightGroups !== undefined) settings.lightGroups = lightGroupsFor(groups);
                await scene.ddgi.prepare();
                const warmFrames = Math.ceil((WARM_SWEEPS * scene.probeCount) / scene.ddgi.settings.maxProbesPerFrame);
                await timedFrames(scene, scene.ddgi.settings.maxProbesPerFrame, warmFrames);
                memory[String(groups)] = scene.ddgi.memoryBytes;
                for (const batch of BATCHES) {
                    const key = `${groups}/${batch}`;
                    times.set(key, [...(times.get(key) ?? []), ...(await timedFrames(scene, batch, FRAMES_PER_BATCH))]);
                }
            }
        }
        scene.dispose();

        const perConfig: Record<string, unknown> = {};
        for (const groups of CONFIGS) {
            const byBatch: Record<string, unknown> = {};
            for (const batch of BATCHES) {
                const frames = times.get(`${groups}/${batch}`) ?? [];
                assert(frames.length >= (FRAMES_PER_BATCH * ROUNDS) / 3, `${sceneId}: only ${frames.length} timed frames for ${groups} groups at ${batch}`);
                const ms = median(frames);
                const offMs = median(times.get(`0/${batch}`) ?? [NaN]);
                byBatch[batch] = { updateMs: ms, usPerProbe: (ms * 1000) / batch, againstGroupsOff: ms / offMs, timedFrames: frames.length };
                log(`${sceneId} ${groups === 0 ? 'groups off' : `${groups} groups`} at ${batch}: update ${ms.toFixed(3)} ms, ${((ms * 1000) / batch).toFixed(3)} µs a probe, ${(ms / offMs).toFixed(3)}× groups off (${frames.length} frames)`);
            }
            perConfig[groups === 0 ? 'off' : `${groups} groups`] = { batches: byBatch, memoryBytes: memory[String(groups)] };
        }
        if (CONFIGS.includes(4) && CONFIGS.includes(0)) {
            const ratio = median(times.get(`4/${JUDGED_BATCH}`) ?? [NaN]) / median(times.get(`0/${JUDGED_BATCH}`) ?? [NaN]);
            if (!(ratio <= FOUR_GROUPS_BAR)) failures.push(`${sceneId}: 4 groups cost ${ratio.toFixed(3)}× groups off per probe at ${JUDGED_BATCH} (bar ${FOUR_GROUPS_BAR})`);
        }
        log(`${sceneId} memory: ${CONFIGS.map((groups) => `${groups === 0 ? 'off' : `${groups} groups`} ${(memory[String(groups)] / 1e6).toFixed(2)} MB`).join(', ')}`);
        results[sceneId] = { probes: scene.probeCount, configs: perConfig };
    }
    assert(failures.length === 0, failures.join('; '));
});
