/**
 * T2.5 checks 2 and 3, the cost (docs/design/m2-lights.md §5, "Cost
 * estimate"): DDGI's update per updated probe with light groups off, 2 and 4
 * groups, on two-room and Sponza through the three adapter (`loadScene`, as
 * the viewer loads them). Also T2.6b's check 3: the sun layer (`sun`, groups
 * off: 2 layers) against everything off, and the 6-layer cap (`4+sun`). A
 * timing page: run it alone under the bench lock, which judges the clock
 * (VALID / INVALID).
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
 * page asserts it; it counts only from a VALID run. T2.6b's check 3: the sun
 * layer at most 1.10× everything off at that batch (§6.5's estimate, +8-10%
 * per probe update), on scenes with a sun. Check 2 (groups off
 * within 3% of `m2`) compares this page's groups-off numbers with a run of
 * the same page on `m2`'s code (`VITE_DDGI_LIGHT_GROUPS=0`), which it
 * supports: with 0 alone it never touches `lightGroups`.
 *
 * Also reported: DDGI's GPU memory per config.
 *
 * Choose with environment variables:
 *   VITE_DDGI_SCENES=two-room,sponza     (default two-room,sponza)
 *   VITE_DDGI_LIGHT_GROUPS=0,2,4         (default 0,2,4; 0 is groups off; `sun` the sun layer
 *                                         with groups off, `4+sun` 4 groups and the sun layer)
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
/** Configs to time: a light-group count (0 is groups off), with `+sun` the sun layer too; `sun` alone is the sun layer with groups off. */
const CONFIGS = listFromEnv(env.VITE_DDGI_LIGHT_GROUPS, ['0', '2', '4']);
const BATCHES = [256, 512] as const;
/** The batch check 3 is judged at. */
const JUDGED_BATCH = 512;
const FRAMES_PER_BATCH = 60;
const ROUNDS = 2;
const WARM_SWEEPS = 3;
/** Check 3's bar: 4 groups at most this times groups off, per updated probe. */
const FOUR_GROUPS_BAR = 1.4;
/** T2.6b check 3's bar: the sun layer at most this times everything off (m2-lights.md §6.5: about +8-10%). */
const SUN_LAYER_BAR = 1.1;
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

/**
 * Runs `frames` frames of `units` probes GPU-bound; returns each timed frame's
 * update ms (every update pass summed). Each pass's own ms per timed frame
 * go into `byPass` when given (a breakdown, reported).
 */
async function timedFrames(scene: ViewerScene, units: number, frames: number, byPass?: Map<string, number[]>): Promise<number[]> {
    const byFrame = new Map<number, number>();
    const pending: Promise<void>[] = [];
    const collect = (): void => {
        const passes = scene.platform.stats.passes.filter((pass) => isUpdatePass(pass.name) && pass.gpuMs !== undefined && pass.gpuMs >= 0);
        if (passes.length === 0) return;
        // A frame's passes arrive together: the newest timed frame only.
        const frame = passes[0].frame;
        if (byFrame.has(frame)) return;
        const ofFrame = passes.filter((pass) => pass.frame === frame);
        byFrame.set(frame, ofFrame.reduce((sum, pass) => sum + (pass.gpuMs ?? 0), 0));
        // Passes that run more than once a frame (rounds) are summed.
        const perName = new Map<string, number>();
        for (const pass of ofFrame) perName.set(pass.name, (perName.get(pass.name) ?? 0) + (pass.gpuMs ?? 0));
        for (const [name, ms] of perName) byPass?.set(name, [...(byPass.get(name) ?? []), ms]);
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

/** A config's light-group count and sun layer: `2`, `sun` (0 groups and the sun), `4+sun`. */
function parseConfig(config: string): { groups: number; sun: boolean } {
    const sun = config === 'sun' || config.endsWith('+sun');
    const groups = config === 'sun' ? 0 : Number(config.replace('+sun', ''));
    return { groups, sun };
}

function configLabel(config: string): string {
    const { groups, sun } = parseConfig(config);
    const groupLabel = groups === 0 ? 'groups off' : `${groups} groups`;
    return sun ? `${groupLabel}, sun layer` : groupLabel;
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
        const passTimes = new Map<string, Map<string, number[]>>();
        const memory: Record<string, number> = {};
        for (let round = 0; round < ROUNDS; round++) {
            const order = round % 2 === 0 ? CONFIGS : [...CONFIGS].reverse();
            for (const config of order) {
                const { groups, sun } = parseConfig(config);
                if (groups !== 0 || settings.lightGroups !== undefined) settings.lightGroups = lightGroupsFor(groups);
                // Only touched with the sun layer asked for (or already there), so the page also runs on code without it.
                if (sun || settings.sunLayer !== undefined) settings.sunLayer = sun;
                await scene.ddgi.prepare();
                const warmFrames = Math.ceil((WARM_SWEEPS * scene.probeCount) / scene.ddgi.settings.maxProbesPerFrame);
                await timedFrames(scene, scene.ddgi.settings.maxProbesPerFrame, warmFrames);
                memory[config] = scene.ddgi.memoryBytes;
                for (const batch of BATCHES) {
                    const key = `${config}/${batch}`;
                    if (!passTimes.has(key)) passTimes.set(key, new Map());
                    times.set(key, [...(times.get(key) ?? []), ...(await timedFrames(scene, batch, FRAMES_PER_BATCH, passTimes.get(key)))]);
                }
            }
        }
        scene.dispose();

        const perConfig: Record<string, unknown> = {};
        for (const config of CONFIGS) {
            const byBatch: Record<string, unknown> = {};
            for (const batch of BATCHES) {
                const frames = times.get(`${config}/${batch}`) ?? [];
                assert(frames.length >= (FRAMES_PER_BATCH * ROUNDS) / 3, `${sceneId}: only ${frames.length} timed frames for ${configLabel(config)} at ${batch}`);
                const ms = median(frames);
                const offMs = median(times.get(`0/${batch}`) ?? [NaN]);
                const passes = Object.fromEntries([...(passTimes.get(`${config}/${batch}`) ?? new Map<string, number[]>())].map(([name, values]) => [name, median(values)]));
                byBatch[batch] = { updateMs: ms, usPerProbe: (ms * 1000) / batch, againstGroupsOff: ms / offMs, timedFrames: frames.length, passes };
                log(`${sceneId} ${configLabel(config)} at ${batch}: update ${ms.toFixed(3)} ms, ${((ms * 1000) / batch).toFixed(3)} µs a probe, ${(ms / offMs).toFixed(3)}× groups off (${frames.length} frames)`);
            }
            perConfig[config === '0' ? 'off' : configLabel(config)] = { batches: byBatch, memoryBytes: memory[config] };
        }
        const judged = (config: string): number => median(times.get(`${config}/${JUDGED_BATCH}`) ?? [NaN]) / median(times.get(`0/${JUDGED_BATCH}`) ?? [NaN]);
        if (CONFIGS.includes('4') && CONFIGS.includes('0')) {
            const ratio = judged('4');
            if (!(ratio <= FOUR_GROUPS_BAR)) failures.push(`${sceneId}: 4 groups cost ${ratio.toFixed(3)}× groups off per probe at ${JUDGED_BATCH} (bar ${FOUR_GROUPS_BAR})`);
        }
        if (CONFIGS.includes('sun') && CONFIGS.includes('0')) {
            const ratio = judged('sun');
            log(`${sceneId}: the sun layer costs ${ratio.toFixed(3)}× everything off per probe at ${JUDGED_BATCH} (bar ${SUN_LAYER_BAR})`);
            if (!(ratio <= SUN_LAYER_BAR)) failures.push(`${sceneId}: the sun layer costs ${ratio.toFixed(3)}× everything off per probe at ${JUDGED_BATCH} (bar ${SUN_LAYER_BAR})`);
        }
        log(`${sceneId} memory: ${CONFIGS.map((config) => `${config === '0' ? 'off' : configLabel(config)} ${(memory[config] / 1e6).toFixed(2)} MB`).join(', ')}`);
        results[sceneId] = { probes: scene.probeCount, configs: perConfig };
    }
    assert(failures.length === 0, failures.join('; '));
});
