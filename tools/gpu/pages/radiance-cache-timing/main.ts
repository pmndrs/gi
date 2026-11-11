/**
 * RC5.1: the radiance cache's GPU time per pass, through the platform's own
 * timer (`stats.passes`), shaped like `ddgi-timing`: per-pass medians over
 * warmed frames, named and grouped as native Capsaicin's per-pass CSV
 * (sections.ts), so phase 5 can set them beside native's groups.
 *
 * Per scene (`cornell-box`, `sponza`, `bistro-interior`, `bistro-exterior`,
 * `two-room`), at 1280 × 720 from the manifest's first bookmark (native's
 * runs used the same camera, `-Camera 1`): the viewer's loader, `ThreeGI`
 * and the radiance cache as the output technique, drawn the way a three user
 * builds it (the scene pass with the screen inputs, the screen stage attached
 * with `encodeBefore`, the composite on the radiance cache's GI light). Then,
 * per `useMultibounce` setting (on, Capsaicin's default, then off; the cache
 * empties when it changes): WARMUP_FRAMES untimed frames, then frames until
 * MEASURED_FRAMES have reported their passes (the timestamps arrive a few
 * frames late). Each frame waits for the GPU, so frames don't overlap and a
 * pass's time is its own.
 *
 * Reported per setting: each pass's median, each Capsaicin section's (the
 * median of its passes' per-frame sum), each group's sum of section medians,
 * native's diffuse subtotal, and the median of the per-frame total of every
 * radiance-cache pass. The platform's own passes (`platform/...`) are listed
 * apart. Not timed by the platform: the host's raster (three's scene pass,
 * its screen inputs and the composite, which is where Capsaicin's
 * `ResolveGI1` lives here), and the technique's buffer and texture copies
 * between passes.
 *
 * Checks: every section the setting runs is timed in at least half the
 * measured frames, with a median above 0 ms (before RC5.1 the platform read
 * every screen-stage pass as 0); every radiance-cache pass maps to a section;
 * no device error (the harness).
 *
 * Switches, for hand use: `VITE_RC_TIMING_SCENES=sponza,two-room` picks
 * scenes, `VITE_RC_TIMING_MULTIBOUNCE=on|off|both` (default both) the
 * multibounce settings, `VITE_RC_TIMING_CELL_SHADING=perHit|budget|alternate`
 * (default perHit) lever 1's mode (RC-L1; `alternate` runs perHit, budget,
 * perHit, budget for each multibounce setting, so the M5's drift with heat
 * falls on both modes alike; each change prepares the platform again, as
 * the mode is read at prepare; the budget at its defaults, and its counters
 * read back after the timed frames), `VITE_RC_TIMING_BUDGET` the budget's
 * `cellShadingBudget` (default the setting's default, 20,000), `VITE_DDGI_MAX_TEXTURE_SIZE` the model textures'
 * largest side (default 1024: native Capsaicin's runs used 1k textures, and
 * GI reads only their averages).
 *
 * M5 numbers are hints, labelled M5; the RTX 3060 under the bench lock is the
 * record (DECISIONS 2026-11-11). Through the lock:
 * `node tools/reference/bench-lock.ts run --no-cool-wait --what "rc5.1 timing" -- pnpm test:gpu --filter radiance-cache-timing --json <file>`
 * (add `--gpu nvidia` on Windows), then `node tools/gpu/pages/radiance-cache-timing/to-csv.ts <file> --append`.
 */

import { mrt, normalView, output, packNormalToRGB, pass } from 'three/tsl';
import { NodeMaterial, QuadMesh, type Mesh, type Node, type Object3D, type WebGPURenderer } from 'three/webgpu';
import type { EnvironmentDescriptor } from '../../../../packages/ddgi-core/src/platform/contract/scene.ts';
import { GIPlatformImpl } from '../../../../packages/ddgi-core/src/platform/gi-platform.ts';
import { RadianceCacheTechnique } from '../../../../packages/ddgi-core/src/techniques/radiance-cache/radiance-cache-technique.ts';
import type { CellShading } from '../../../../packages/ddgi-core/src/techniques/radiance-cache/world-cache/capacity.ts';
import { ddgi, giAlbedo, GI_ALBEDO_OUTPUT, GI_NORMAL_OUTPUT } from '../../../../packages/ddgi-three/src/composite.ts';
import { screenInputs } from '../../../../packages/ddgi-three/src/screen-inputs.ts';
import { ThreeGI } from '../../../../packages/ddgi-three/src/three-gi.ts';
import { loadScene } from '../../../../apps/viewer/src/scene/load-scene.ts';
import { platformSkyEnvironment } from '../../../../scenes/src/sky/sky-image.ts';
import type { SceneManifest } from '../../../../scenes/src/manifest.ts';
import { assert } from '../../client.ts';
import { advanceThreeFrame, findManifest, shotCamera } from '../three-composite-shared/composite-check.ts';
import { runThreePage, type ThreePageContext } from '../three-shared/harness.ts';
import { CAPSAICIN_GROUPS, DIFFUSE_GROUPS, groupOf, SECTIONS, sectionOf, type CapsaicinGroup, type SceneTiming, type SettingTiming } from './sections.ts';

const SIZE = { width: 1280, height: 720 } as const;
const DEFAULT_SCENES = ['cornell-box', 'sponza', 'bistro-interior', 'bistro-exterior', 'two-room'] as const;
/** Untimed frames per setting first: the world cache and the screen probes' history fill, the clock settles. */
const WARMUP_FRAMES = 300;
const MEASURED_FRAMES = 120;
/** Give up after this many frames per measured one (timestamps missing from frames). */
const MAX_FRAMES_PER_MEASURED = 3;
const DEFAULT_MAX_TEXTURE_SIZE = 1024;

const env = import.meta.env as Record<string, unknown>;
const SCENES = typeof env.VITE_RC_TIMING_SCENES === 'string' && env.VITE_RC_TIMING_SCENES.trim() !== '' ? env.VITE_RC_TIMING_SCENES.split(',').map((id) => id.trim()) : [...DEFAULT_SCENES];
const MULTIBOUNCE_SETTINGS = multibounceSettings(env.VITE_RC_TIMING_MULTIBOUNCE);
const CELL_SHADING_SEQUENCE = cellShadingSequence(env.VITE_RC_TIMING_CELL_SHADING);
const CELL_SHADING_BUDGET = typeof env.VITE_RC_TIMING_BUDGET === 'string' && env.VITE_RC_TIMING_BUDGET.trim() !== '' ? Number(env.VITE_RC_TIMING_BUDGET) : undefined;
const MAX_TEXTURE_SIZE = typeof env.VITE_DDGI_MAX_TEXTURE_SIZE === 'string' ? Number(env.VITE_DDGI_MAX_TEXTURE_SIZE) : DEFAULT_MAX_TEXTURE_SIZE;

function multibounceSettings(value: unknown): boolean[] {
    switch (value ?? 'both') {
        case 'both':
            return [true, false];
        case 'on':
            return [true];
        case 'off':
            return [false];
        default:
            throw new Error(`VITE_RC_TIMING_MULTIBOUNCE must be on, off or both, got "${String(value)}"`);
    }
}

/** Lever 1's modes to time in turn for each multibounce setting (RC-L1). */
function cellShadingSequence(value: unknown): CellShading[] {
    switch (value ?? 'perHit') {
        case 'perHit':
            return ['perHit'];
        case 'budget':
            return ['budget'];
        case 'alternate':
            return ['perHit', 'budget', 'perHit', 'budget'];
        default:
            throw new Error(`VITE_RC_TIMING_CELL_SHADING must be perHit, budget or alternate, got "${String(value)}"`);
    }
}

function median(values: readonly number[]): number {
    if (values.length === 0) return Number.NaN;
    const sorted = [...values].sort((a, b) => a - b);
    const middle = Math.floor(sorted.length / 2);
    return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
}

const round = (ms: number): number => Math.round(ms * 10000) / 10000;

/** The manifest's environment as the platform takes it (the viewer's radiance-cache technique does the same). */
function environmentFor(manifest: SceneManifest): EnvironmentDescriptor {
    const environment = manifest.environment;
    switch (environment.type) {
        case 'none':
            return { type: 'none' };
        case 'color':
            return { type: 'color', color: environment.color, intensity: environment.intensity };
        case 'procedural':
            return platformSkyEnvironment(manifest);
    }
}

/** Draws `node` to the canvas as a new three frame (the passes it reads render first). */
function drawFrame(renderer: WebGPURenderer, quad: QuadMesh): void {
    advanceThreeFrame(renderer);
    renderer.setRenderTarget(null);
    quad.render(renderer);
}

async function timeScene(context: ThreePageContext, sceneId: string): Promise<SceneTiming> {
    const { device, renderer, log } = context;
    const manifest = findManifest(sceneId);
    const loaded = await loadScene(manifest, () => {}, { maxTextureSize: MAX_TEXTURE_SIZE });
    // The sky dome is for viewing; GI takes the sky from the platform environment.
    const content = new Set<Object3D>();
    loaded.content.traverse((object) => content.add(object));
    loaded.scene.traverse((object) => {
        if ((object as Mesh).isMesh && !content.has(object)) object.userData.giContributes = false;
    });
    const camera = shotCamera(manifest, SIZE.width, SIZE.height);

    const platform = new GIPlatformImpl(device, { label: `radiance-cache-timing ${sceneId}`, gpuTiming: true });
    const technique = platform.addTechnique(
        new RadianceCacheTechnique({ useMultibounce: MULTIBOUNCE_SETTINGS[0], cellShading: CELL_SHADING_SEQUENCE[0], ...(CELL_SHADING_BUDGET === undefined ? {} : { cellShadingBudget: CELL_SHADING_BUDGET }) }),
    );
    platform.setOutput(technique);
    const preparing = platform.prepare();
    // Chrome sends the compile requests when this JavaScript task ends (ThreeGI's file comment).
    await new Promise((resolve) => setTimeout(resolve, 0));
    // The radiance cache has no world-space budget: all its work is the screen stage.
    const threeGI = new ThreeGI(platform, loaded.scene, { preparing, budget: { units: 0 }, environment: environmentFor(manifest) });
    threeGI.update(camera, 0);
    await preparing;
    // Time on the final mesh trees: tinybvh's replace the GPU's first ones as they arrive.
    threeGI.update(camera, 0);
    await platform.rays.whenBuilt();

    const scenePass = pass(loaded.scene, camera);
    const stage = screenInputs(scenePass, threeGI);
    scenePass.setMRT(mrt({ output, [GI_ALBEDO_OUTPUT]: giAlbedo, [GI_NORMAL_OUTPUT]: packNormalToRGB(normalView), ...stage.outputs }));
    const composite = stage.encodeBefore(ddgi(scenePass, threeGI, { occlusion: false }) as unknown as Node);
    const material = new NodeMaterial();
    material.fragmentNode = composite;
    material.name = 'radiance-cache-timing picture';
    const quad = new QuadMesh(material);

    const frame = async (): Promise<void> => {
        threeGI.update(camera, 1 / 60);
        drawFrame(renderer, quad);
        await device.queue.onSubmittedWorkDone();
        // Let the readback's mapAsync callbacks run before the next encode collects them.
        await new Promise((resolve) => setTimeout(resolve, 0));
    };

    try {
        const settings: SettingTiming[] = [];
        const runs = MULTIBOUNCE_SETTINGS.flatMap((useMultibounce) => CELL_SHADING_SEQUENCE.map((cellShading, index) => ({ useMultibounce, cellShading, round: Math.floor(index / 2) + 1 })));
        for (const { useMultibounce, cellShading, round } of runs) {
            technique.settings.useMultibounce = useMultibounce;
            // Lever 1's mode is read at prepare (its buffers and kernels); the multibounce turns on only at prepare too.
            if (technique.worldCache?.preparedCellShading !== cellShading || (useMultibounce && technique.worldCache?.preparedWithMultibounce === false)) {
                technique.settings.cellShading = cellShading;
                await platform.prepare();
            }
            for (let warm = 0; warm < WARMUP_FRAMES; warm++) await frame();
            assert(!platform.rays.building, `${sceneId}: mesh trees still building after the warm-up`);

            // Each pass's ms by frame; the frame number keys them, as stats.passes holds only the newest timed frame.
            const byPass = new Map<string, Map<number, number>>();
            const timedFrames = new Set<number>();
            const firstMeasuredFrame = platform.stats.frame + 1;
            for (let index = 0; timedFrames.size < MEASURED_FRAMES && index < MEASURED_FRAMES * MAX_FRAMES_PER_MEASURED; index++) {
                await frame();
                for (const timed of platform.stats.passes) {
                    if (timed.frame < firstMeasuredFrame || timed.gpuMs === undefined || timed.gpuMs < 0) continue;
                    const frames = byPass.get(timed.name) ?? new Map<number, number>();
                    // A name timed twice in a frame is added up, as the timer's own totals are.
                    frames.set(timed.frame, (frames.get(timed.frame) ?? 0) + timed.gpuMs);
                    byPass.set(timed.name, frames);
                    timedFrames.add(timed.frame);
                }
            }
            const cellShadingCounters = await readCellShadingCounters(device, technique);
            settings.push(summarise(sceneId, { useMultibounce, cellShading, round }, byPass, timedFrames, { ...technique.counters, ...cellShadingCounters }));
            const latest = settings[settings.length - 1];
            log(
                `${sceneId}, multibounce ${useMultibounce ? 'on' : 'off'}, ${cellShading} (round ${round}), ${latest.framesTimed} frames: ` +
                    CAPSAICIN_GROUPS.map((group) => `${group} ${latest.groups[group].toFixed(3)}`).join(', ') +
                    `; diffuse subtotal ${latest.diffuseSubtotalMs.toFixed(3)} ms; all passes ${latest.radianceCacheFrameMs.toFixed(3)} ms a frame`,
            );
            log(`${sceneId}, multibounce ${useMultibounce ? 'on' : 'off'}, ${cellShading} (round ${round}), sections: ${Object.entries(latest.sections).map(([section, ms]) => `${section} ${ms.toFixed(3)}`).join(', ')}`);
        }
        return { triangles: loaded.stats.triangles, maxTextureSize: MAX_TEXTURE_SIZE, settings };
    } finally {
        material.dispose();
        stage.dispose();
        scenePass.dispose();
        threeGI.dispose();
        platform.dispose();
        await device.queue.onSubmittedWorkDone();
    }
}

/** Lever 1's counters for the latest frame (cells in use, lit, their shadow rays), read back; none per hit. */
async function readCellShadingCounters(device: GPUDevice, technique: RadianceCacheTechnique): Promise<Record<string, number>> {
    const words = technique.worldCache?.cellShadingCounters;
    if (!words) return {};
    const readback = device.createBuffer({ label: 'radiance-cache-timing counters', size: 64 * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const encoder = device.createCommandEncoder({ label: 'radiance-cache-timing counters' });
    encoder.copyBufferToBuffer(words.usedCells.buffer, 0, readback, 0, 64 * 4);
    device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    const counters = new Uint32Array(readback.getMappedRange().slice(0));
    readback.unmap();
    readback.destroy();
    const at = (word: { offset: number }): number => counters[word.offset / 4];
    return { 'cell shading cells in use': at(words.usedCells), 'cell shading cells lit': at(words.litCells), 'cell shading shadow rays': at(words.shadowRays) };
}

/** Medians per pass, section and group, with the checks (the file comment). */
function summarise(
    sceneId: string,
    setting: { readonly useMultibounce: boolean; readonly cellShading: CellShading; readonly round: number },
    byPass: ReadonlyMap<string, ReadonlyMap<number, number>>,
    timedFrames: ReadonlySet<number>,
    counters: Record<string, number>,
): SettingTiming {
    const { useMultibounce, cellShading } = setting;
    const label = `${sceneId}, multibounce ${useMultibounce ? 'on' : 'off'}, ${cellShading} (round ${setting.round})`;
    assert(timedFrames.size >= MEASURED_FRAMES / 2, `${label}: only ${timedFrames.size} timed frames`);
    const passes: Record<string, number> = {};
    const platformPasses: Record<string, number> = {};
    // Each section's ms per frame: its passes added up (the trace and the cache half of PopulateScreenProbesCells).
    const sectionFrames = new Map<string, Map<number, number>>();
    const radianceCacheFrames = new Map<number, number>();
    for (const [name, frames] of byPass) {
        const ms = round(median([...frames.values()]));
        const section = sectionOf(name);
        if (section === null) {
            platformPasses[name] = ms;
            continue;
        }
        assert(groupOf(section) !== undefined, `${label}: the pass ${name} has no Capsaicin section in sections.ts`);
        passes[name] = ms;
        const perFrame = sectionFrames.get(section) ?? new Map<number, number>();
        for (const [frame, value] of frames) {
            perFrame.set(frame, (perFrame.get(frame) ?? 0) + value);
            radianceCacheFrames.set(frame, (radianceCacheFrames.get(frame) ?? 0) + value);
        }
        sectionFrames.set(section, perFrame);
    }

    const sections: Record<string, number> = {};
    const groups = Object.fromEntries(CAPSAICIN_GROUPS.map((group) => [group, 0])) as Record<CapsaicinGroup, number>;
    for (const { section, group, multibounceOnly, cellShadingOnly } of SECTIONS) {
        const frames = sectionFrames.get(section);
        if ((multibounceOnly && !useMultibounce) || (cellShadingOnly !== undefined && cellShadingOnly !== cellShading)) {
            assert(frames === undefined, `${label}: ${section} ran with the multibounce ${useMultibounce ? 'on' : 'off'} and cellShading ${cellShading}`);
            continue;
        }
        assert(frames !== undefined && frames.size >= timedFrames.size / 2, `${label}: ${section} was timed in ${frames?.size ?? 0} of ${timedFrames.size} frames`);
        const ms = round(median([...frames.values()]));
        assert(ms > 0, `${label}: ${section} reads ${ms} ms`);
        sections[section] = ms;
        groups[group] = round(groups[group] + ms);
    }
    return {
        useMultibounce,
        cellShading,
        round: setting.round,
        framesTimed: timedFrames.size,
        passes,
        sections,
        groups,
        diffuseSubtotalMs: round(DIFFUSE_GROUPS.reduce((sum, group) => sum + groups[group], 0)),
        radianceCacheFrameMs: round(median([...radianceCacheFrames.values()])),
        platformPasses,
        counters,
    };
}

await runThreePage(
    async (context) => {
        assert(context.device.features.has('timestamp-query'), 'timing needs the timestamp-query feature');
        context.log(`scenes ${SCENES.join(', ')}; multibounce ${MULTIBOUNCE_SETTINGS.map((on) => (on ? 'on' : 'off')).join(', ')}; cell shading ${CELL_SHADING_SEQUENCE.join(', ')}; textures at most ${MAX_TEXTURE_SIZE} px; ${WARMUP_FRAMES} warm-up and ${MEASURED_FRAMES} timed frames per setting`);
        for (const sceneId of SCENES) context.data[sceneId] = await timeScene(context, sceneId);
    },
    SIZE,
);
