/**
 * RC4.4: the radiance cache's accuracy against the Cycles references, at the
 * references' stored cameras and size (1920 × 1080). A measurement, not a
 * gate: it fails only on device errors, NaN or infinite values, a reference
 * read upside down, or no view measured. Phase 5 reuses it; the README
 * section "Radiance cache accuracy" (tools/reference/README.md) says how.
 *
 * The question it answers: how far is the radiance cache's GI from the path
 * tracer, and which part of it is responsible. Judged against Cycles (the
 * path tracer), never against another setting of the cache (DECISIONS
 * 2026-11-19).
 *
 * **What is rendered**, per scene, the way a three user builds it (the
 * viewer's radiance-cache entry, `apps/viewer/src/techniques/radiance-cache-technique.ts`):
 * the platform with the radiance cache as its output, `ThreeGI`, a `pass()`
 * with `mrt({ output, giAlbedo, normal, ...screenInputs outputs })`, and the
 * composite `ddgi(pass, threeGI, { occlusion: false })` drawn every frame.
 * As M4's textured accuracy page does: the scene's own materials (GI reads
 * the textures), point and spot lights without a range window (as Cycles
 * lights them), the sky dome left out of GI (GI takes the sky from the
 * platform environment), Bistro's three textures capped at 1024 (the 6 GB
 * RTX; GI's own store is unchanged).
 *
 * **What is compared** (accuracy.ts: bias, 8 × 8 mean error, 8 × 8 p95 error):
 * - **GI term** (`giAlbedo × E / π`, what the composite adds) against the
 *   render's GI composite (`<prefix>-gi.exr`), on E averaged over
 *   `MEASURE_FRAMES` frames after `WARM_FRAMES` (the bias, with the per-frame
 *   noise averaged out), and on the last frame alone (what one frame shows);
 * - **composite** (the pass's own colour + the GI term) against
 *   `<prefix>-full.exr`: information only, since it also carries three's
 *   direct light (shadow maps) and its sky dome;
 * - **world cache alone**: the cache looked up at each pixel's visible
 *   surface (gpu.ts) against what it holds in Cycles' terms, the scene
 *   lights' direct light (`<prefix>-lights-direct.exr`; the cache's hits get
 *   no sky and no emitter light, register D33 and D21) plus, with
 *   multibounce, the bounces (`<prefix>-indirect.exr`), over the pixels
 *   where the cache has a cell (its coverage is reported).
 *
 * **Settings measured** (each from an emptied cache and history, the same
 * frames): `default` (Capsaicin's defaults with our register changes),
 * `history off` (the screen probes' history dropped before every frame, so
 * each frame runs as the first; the world cache keeps its cells),
 * `multibounce off` (GI-1.0's single bounce through the cache) and
 * `cells 16 px` (the world cache's cells half as wide on screen, to see how
 * much of the error is the cache's resolution). The differences between
 * them say what each part contributes.
 *
 * **Experiment E10 settings** (P6.3, picked by name only: not in the default
 * list): `fallback <units>` stacks a DDGI volume (the manifest's) under the
 * radiance cache and reads the second bounce from it (`secondBounce:
 * 'fallback'`, the cells' indirect cap 4), DDGI updating 256 or 512 probes a
 * frame; `fallback cap16 <units>` the same at Capsaicin's cap of 16;
 * `fallback 512 no env` sets DDGI's environment layer (the sky and emissive
 * surfaces, and their bounces) to 0, so its difference from `fallback 512` is
 * the light the cache's own lights lack (registers D21, D33); `fallback 512
 * history off` drops the screen probes' history as `history off` does. DDGI
 * is attached when one of these is picked, stacked under the cache only for
 * them (the other settings run with nothing stacked, so `default`, whose
 * `secondBounce` is `'auto'`, traces rays as it does for a host without
 * DDGI), and its field is reset before each of them (its warm-up then runs
 * again), so no setting starts from another's field. `gi1-rays` names
 * GI-1.2's rays explicitly (the faithful baseline), whatever is stacked.
 *
 * **Picking scenes, settings and frame counts** (vite env, as other pages):
 * `VITE_RC_ACCURACY_SCENES=two-room,sponza`, `VITE_RC_ACCURACY_CONFIGS=default,history off`,
 * `VITE_RC_ACCURACY_WARM_FRAMES`, `VITE_RC_ACCURACY_MEASURE_FRAMES`.
 *
 * **RC3.6 settings** (more than two bounces, picked by name only):
 * `cache feedback` (register E12: GI-1.2's rays, each second hit bringing its
 * cell's indirect value too), `temporal feedback` (D68: probe rays whose hit
 * was on last frame's screen read last frame's lit image), `cache + temporal
 * feedback`, and `fallback 256 temporal feedback` (DDGI stacked, 256 probes a
 * frame, with temporal feedback). With any temporal-feedback setting picked,
 * the technique is made with `useTemporalFeedback` on so it asks three for the
 * lit colour and albedo (the other settings turn it off again).
 * `VITE_RC_ACCURACY_NEURAL=tail` makes the technique with P6.4's neural tail
 * for the whole run (read at prepare, as the backend is); run it with
 * `gi1-rays`, the only second bounce the tail extends.
 * `VITE_RC_COMPARE` sets compare toggles (RC4.5, radiance-cache-probes/compare-env.ts)
 * under every configuration's own settings, for example RC4.1's blend weight:
 * `VITE_RC_COMPARE=rayGuidingCorrection=false`.
 * Scenes without a reference are listed and skipped. The references are
 * `bench/references/<scene>/<bookmark>/` (gitignored; a junction to the
 * checkout that holds them works).
 *
 * Through the lock:
 * `node tools/reference/bench-lock.ts run --no-cool-wait --what "rc accuracy" -- pnpm test:gpu --filter radiance-cache-accuracy --gpu nvidia --out bench/screenshots/rc-accuracy --json <file>`.
 * The numbers are in the report's data (`--json`) and printed to the page's
 * log; the screenshot is a contact sheet per view (reference, the GI term
 * with and without history, the default's error).
 */

import { FloatType, NodeMaterial, PerspectiveCamera, QuadMesh, RenderTarget, type Mesh, type Node, type Object3D } from 'three/webgpu';
import { mrt, normalView, output, packNormalToRGB, pass, uv, vec4 } from 'three/tsl';
import { GIPlatformImpl } from '../../../../packages/core/src/platform/gi-platform.ts';
import { DDGITechnique } from '../../../../packages/core/src/techniques/ddgi/ddgi-technique.ts';
import { RadianceCacheTechnique } from '../../../../packages/cache/src/radiance-cache-technique.ts';
import { setCompareValues } from '../../../../packages/cache/src/compare-toggles.ts';
import { DEFAULT_RADIANCE_CACHE_SETTINGS, type RadianceCacheSettings, type WorldCacheKind } from '../../../../packages/cache/src/settings.ts';
import { ddgi, giAlbedo, GI_ALBEDO_OUTPUT, GI_NORMAL_OUTPUT } from '../../../../packages/three/src/composite.ts';
import { screenInputs } from '../../../../packages/three/src/screen-inputs.ts';
import { ThreeGI } from '../../../../packages/three/src/three-gi.ts';
import { loadScene } from '../../../../apps/viewer/src/scene/load-scene.ts';
import { platformSkyEnvironment } from '../../../../scenes/src/sky/sky-image.ts';
import { assert } from '../../client.ts';
import { PAGE_COMPARE_VALUES } from '../radiance-cache-probes/compare-env.ts';
import { halves, readTexture } from '../radiance-cache-probes/readback.ts';
import { runThreePage, type ThreePageContext } from '../three-shared/harness.ts';
import { advanceThreeFrame, findManifest, HdrReader, SHOT_FAR, SHOT_NEAR } from '../three-composite-shared/composite-check.ts';
import { accuracyNumbers, addImages, giTerm, type AccuracyNumbers, type RgbImage } from './accuracy.ts';
import { IrradianceAccumulator, WorldCacheProbe } from './gpu.ts';
import { findReferenceView, listReferenceFiles, loadReferencePart, type ReferenceView } from './references.ts';

const env = import.meta.env as Record<string, unknown>;
const listFromEnv = (name: string): string[] | null => (typeof env[name] === 'string' && env[name] ? (env[name] as string).split(',').map((item) => item.trim()).filter(Boolean) : null);
const numberFromEnv = (name: string, fallback: number): number => (typeof env[name] === 'string' && env[name] ? Number(env[name]) : fallback);

/** The scenes with Cycles references, in order: point-light and emissive Cornell, two-room, Sponza, Bistro. */
const DEFAULT_SCENES = ['cornell-box-point', 'cornell-box', 'two-room', 'sponza', 'bistro-interior', 'bistro-exterior'];
const SCENES = listFromEnv('VITE_RC_ACCURACY_SCENES') ?? DEFAULT_SCENES;
/** T2.17's cap on three's own textures where they don't fit the 6 GB RTX (as m4-textured-accuracy). */
const TEXTURE_CAPS: Record<string, number> = { 'bistro-exterior': 1024, 'bistro-interior': 1024 };
/**
 * Frames from an emptied cache before measuring: the world cache's cells fill to their 16-sample cap and
 * the probes' running means to 8 well within this (RC4.1's stability page converges in 60 + 30).
 */
const WARM_FRAMES = numberFromEnv('VITE_RC_ACCURACY_WARM_FRAMES', 90);
/** Frames whose irradiance is averaged: the per-frame noise mostly averages out, the bias stays. */
const MEASURE_FRAMES = numberFromEnv('VITE_RC_ACCURACY_MEASURE_FRAMES', 60);

/** A setting of the radiance cache to measure: its changes from the defaults, and whether the probes keep history. */
interface Configuration {
    readonly name: string;
    readonly settings: Partial<RadianceCacheSettings>;
    readonly history: boolean;
    /** E10: DDGI stacked under the cache, updating this many probes a frame; absent for none. */
    readonly ddgiUnits?: number;
    /** E10: DDGI's environment layer (sky, emitters) on; default true. */
    readonly ddgiEnvironment?: boolean;
}

const ALL_CONFIGURATIONS: readonly Configuration[] = [
    { name: 'default', settings: {}, history: true },
    // GI-1.2's second-bounce rays by name: the faithful baseline, whatever 'auto' picks.
    { name: 'gi1-rays', settings: { secondBounce: 'gi1-rays' }, history: true },
    { name: 'history off', settings: {}, history: false },
    { name: 'multibounce off', settings: { useMultibounce: false }, history: true },
    { name: 'cells 16 px', settings: { hashGridCacheCellSize: 16 }, history: true },
];
/** Experiment E10 (P6.3): the second bounce from a DDGI volume; run only when named. */
const E10_CONFIGURATIONS: readonly Configuration[] = [
    ...[256, 512].map((units) => ({ name: `fallback ${units}`, settings: { secondBounce: 'fallback' as const }, history: true, ddgiUnits: units })),
    ...[256, 512].map((units) => ({
        name: `fallback cap16 ${units}`,
        settings: { secondBounce: 'fallback' as const, fallbackIndirectSampleCount: 16 },
        history: true,
        ddgiUnits: units,
    })),
    { name: 'fallback 512 no env', settings: { secondBounce: 'fallback' }, history: true, ddgiUnits: 512, ddgiEnvironment: false },
    // With the screen probes' history off, which RC4.4 found too bright on Sponza.
    { name: 'fallback 512 history off', settings: { secondBounce: 'fallback' }, history: false, ddgiUnits: 512 },
];
/** RC3.6: more than two bounces (registers E12, D68); run only when named. */
const RC3_6_CONFIGURATIONS: readonly Configuration[] = [
    { name: 'cache feedback', settings: { secondBounce: 'gi1-rays', cacheFeedback: true }, history: true },
    { name: 'temporal feedback', settings: { secondBounce: 'gi1-rays', useTemporalFeedback: true }, history: true },
    { name: 'cache + temporal feedback', settings: { secondBounce: 'gi1-rays', cacheFeedback: true, useTemporalFeedback: true }, history: true },
    { name: 'fallback 256 temporal feedback', settings: { secondBounce: 'fallback', useTemporalFeedback: true }, history: true, ddgiUnits: 256 },
];
const configurationNames = listFromEnv('VITE_RC_ACCURACY_CONFIGS');
/**
 * `VITE_RC_ACCURACY_WORLD_CACHE=sharc`: the world-cache backend for the whole run (P6.1); the backend
 * is made at prepare, so it is a run's choice, not a configuration's. The settings tied to
 * Capsaicin's hash grid (multibounce, cell size) don't apply to the SHaRC cache.
 */
const WORLD_CACHE: WorldCacheKind = env.VITE_RC_ACCURACY_WORLD_CACHE === 'sharc' ? 'sharc' : 'gi1-hash-grid';
/** `VITE_RC_ACCURACY_NEURAL=tail`: P6.4's neural tail for the whole run (made at prepare). */
const NEURAL_CACHE = env.VITE_RC_ACCURACY_NEURAL === 'tail' ? 'tail' : 'off';
/** The run's settings under every configuration's own: the backend and the tail, made at prepare. */
const RUN_SETTINGS: Partial<RadianceCacheSettings> = { worldCache: WORLD_CACHE, neuralCache: NEURAL_CACHE };
const CONFIGURATIONS = configurationNames
    ? [...ALL_CONFIGURATIONS, ...E10_CONFIGURATIONS, ...RC3_6_CONFIGURATIONS].filter((configuration) => configurationNames.includes(configuration.name))
    : ALL_CONFIGURATIONS;
/** RC3.6: a temporal-feedback setting needs the lit colour and albedo, asked for when the technique is made. */
const NEEDS_LIT_IMAGE = CONFIGURATIONS.some((configuration) => configuration.settings.useTemporalFeedback === true);
const NEEDS_DDGI = CONFIGURATIONS.some((configuration) => configuration.ddgiUnits !== undefined);

/** Contact-sheet thumbnails, pixels across: four fit the runner's window (about 1,258 CSS pixels, NOTES). */
const THUMBNAIL_WIDTH = 300;
/** The sheet's exposure puts the reference GI's mean here before tone mapping. */
const SHEET_KEY = 0.35;

interface WorldCacheNumbers extends AccuracyNumbers {
    /** Share of surface pixels where the cache has a cell. */
    readonly coverage: number;
    /** What it was compared with. */
    readonly against: 'lights-direct' | 'lights-direct + indirect';
}

interface ConfigurationResult {
    /** GI term from E averaged over the measured frames, against `<prefix>-gi.exr`. */
    readonly gi: AccuracyNumbers;
    /** GI term of the last frame alone. */
    readonly giOneFrame: AccuracyNumbers;
    /** The pass's colour + the averaged GI term, against `<prefix>-full.exr` (information). */
    readonly composite: AccuracyNumbers;
    readonly worldCache: WorldCacheNumbers;
    readonly meanGi: number;
}

interface ViewResult {
    readonly reference: string;
    readonly referenceSamples: number;
    /** The reference's own noise (its json), per pixel and over 8 × 8 blocks, when measured. */
    readonly referenceNoise: { readonly pixel: number; readonly block8: number } | null;
    readonly sameManifest: boolean | null;
    readonly configurations: Record<string, ConfigurationResult>;
    /** The default's 8 × 8 mean error against the reference read the other way up (must be worse: the rows are the right way up). */
    readonly upsideDownMeanError8: number | null;
    /**
     * The check on the setup itself: three's picture without GI (its lights, shadow maps, emission
     * and sky dome) against Cycles' full image minus its GI composite (the scene lights' direct light,
     * emitters and the sky seen directly). Close means camera, units and orientation line up, so the
     * GI numbers are about GI.
     */
    readonly threeDirect: AccuracyNumbers | null;
}

/** Linear HDR rgb to display bytes at a thumbnail's size: exposure, Reinhard, sRGB-ish. */
function thumbnail(image: RgbImage, exposure: number): ImageData {
    const scale = image.width / THUMBNAIL_WIDTH;
    const height = Math.round(image.height / scale);
    const out = new ImageData(THUMBNAIL_WIDTH, height);
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < THUMBNAIL_WIDTH; x++) {
            const source = (Math.min(image.height - 1, Math.floor((y + 0.5) * scale)) * image.width + Math.min(image.width - 1, Math.floor((x + 0.5) * scale))) * 3;
            const target = (y * THUMBNAIL_WIDTH + x) * 4;
            for (let channel = 0; channel < 3; channel++) {
                const value = Math.max(0, image.rgb[source + channel]) * exposure;
                out.data[target + channel] = Math.round(255 * Math.pow(value / (1 + value), 1 / 2.2));
            }
            out.data[target + 3] = 255;
        }
    }
    return out;
}

/** The signed error per 8 × 8 block as a picture: red where the test is brighter, blue darker, full at ±50%. */
function errorThumbnail(reference: RgbImage, test: RgbImage): ImageData {
    const scale = reference.width / THUMBNAIL_WIDTH;
    const height = Math.round(reference.height / scale);
    const out = new ImageData(THUMBNAIL_WIDTH, height);
    let referenceMean = 0;
    for (let index = 0; index < reference.rgb.length; index++) referenceMean += reference.rgb[index];
    referenceMean /= reference.rgb.length;
    const radius = Math.max(1, Math.round(scale / 2));
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < THUMBNAIL_WIDTH; x++) {
            const cx = Math.floor((x + 0.5) * scale);
            const cy = Math.floor((y + 0.5) * scale);
            let r = 0;
            let t = 0;
            for (let sy = Math.max(0, cy - radius); sy < Math.min(reference.height, cy + radius); sy++) {
                for (let sx = Math.max(0, cx - radius); sx < Math.min(reference.width, cx + radius); sx++) {
                    const pixel = (sy * reference.width + sx) * 3;
                    r += reference.rgb[pixel] + reference.rgb[pixel + 1] + reference.rgb[pixel + 2];
                    t += test.rgb[pixel] + test.rgb[pixel + 1] + test.rgb[pixel + 2];
                }
            }
            const error = Math.max(-1, Math.min(1, (t - r) / (r + 0.05 * referenceMean * 3 * (2 * radius) ** 2 + 1e-12) / 0.5));
            const target = (y * THUMBNAIL_WIDTH + x) * 4;
            out.data[target] = Math.round(255 * Math.max(0, error));
            out.data[target + 1] = Math.round(64 * (1 - Math.abs(error)));
            out.data[target + 2] = Math.round(255 * Math.max(0, -error));
            out.data[target + 3] = 255;
        }
    }
    return out;
}

function addThumbnail(row: HTMLElement, caption: string, image: ImageData): void {
    const cell = document.createElement('div');
    const label = document.createElement('div');
    label.textContent = caption;
    label.style.font = '11px sans-serif';
    label.style.color = '#ddd';
    label.style.width = `${THUMBNAIL_WIDTH}px`;
    const canvas = document.createElement('canvas');
    canvas.width = image.width;
    canvas.height = image.height;
    canvas.style.display = 'block';
    canvas.getContext('2d')?.putImageData(image, 0, 0);
    cell.append(label, canvas);
    row.append(cell);
}

function upsideDown(image: RgbImage): RgbImage {
    const rgb = new Float32Array(image.rgb.length);
    const row = image.width * 3;
    for (let y = 0; y < image.height; y++) rgb.set(image.rgb.subarray(y * row, (y + 1) * row), (image.height - 1 - y) * row);
    return { width: image.width, height: image.height, rgb };
}

function countNonFinite(values: Float32Array): number {
    let count = 0;
    for (const value of values) if (!Number.isFinite(value)) count++;
    return count;
}

const percent = (value: number): string => `${value >= 0 ? '+' : ''}${(value * 100).toFixed(1)}%`;
const formatNumbers = (numbers: AccuracyNumbers): string => `bias ${percent(numbers.bias)}, 8x8 mean ${(numbers.meanError8 * 100).toFixed(1)}%, p95 ${(numbers.p95Error8 * 100).toFixed(0)}%`;

async function measureScene(context: ThreePageContext, sceneId: string, views: readonly ReferenceView[], sheet: HTMLElement): Promise<Record<string, ViewResult>> {
    const { device, renderer, log } = context;
    const { width, height } = views[0].shot.camera;
    for (const view of views) assert(view.shot.camera.width === width && view.shot.camera.height === height, `${sceneId}/${view.camera}: the references differ in size`);
    renderer.setSize(width, height, false);

    const manifest = findManifest(sceneId);
    const maxTextureSize = TEXTURE_CAPS[sceneId];
    const loaded = await loadScene(manifest, () => {}, maxTextureSize === undefined ? {} : { maxTextureSize });
    const scene = loaded.scene;
    // Cycles lights point and spot lights with no range window (M1's `ignoreLightRange`).
    scene.traverse((object) => {
        const light = object as Object3D & { isPointLight?: boolean; isSpotLight?: boolean; distance?: number };
        if (light.isPointLight || light.isSpotLight) light.distance = 0;
    });
    // The sky dome is for viewing; GI takes the sky from the platform environment.
    const content = new Set<Object3D>();
    loaded.content.traverse((object) => content.add(object));
    scene.traverse((object) => {
        if ((object as Mesh).isMesh && !content.has(object)) object.userData.giContributes = false;
    });

    const camera = new PerspectiveCamera(views[0].shot.camera.fovYDeg, width / height, SHOT_NEAR, SHOT_FAR);
    const placeCamera = (view: ReferenceView): void => {
        const shot = view.shot.camera;
        camera.fov = shot.fovYDeg;
        camera.position.set(...shot.position);
        camera.lookAt(...shot.target);
        camera.updateProjectionMatrix();
        camera.updateMatrixWorld(true);
    };
    placeCamera(views[0]);

    const platform = new GIPlatformImpl(device, { label: `radiance-cache-accuracy ${sceneId}` });
    const technique = platform.addTechnique(new RadianceCacheTechnique({ ...RUN_SETTINGS, useTemporalFeedback: NEEDS_LIT_IMAGE }));
    platform.setOutput(technique);
    // E10: DDGI stacked under the cache, with one light group so its environment layer can be weighed.
    const ddgiTechnique = NEEDS_DDGI
        ? platform.addTechnique(
              new DDGITechnique({
                  volume: { center: manifest.volume.center, size: manifest.volume.size, spacing: manifest.volume.probeSpacing },
                  lightGroups: { count: 1, weights: [[1, 1, 1]], environmentWeight: [1, 1, 1] },
              }),
          )
        : null;
    // Stacked for prepare, so the cache builds its pass for DDGI's module then; each setting stacks or unstacks it.
    if (ddgiTechnique) platform.setFallback(technique, ddgiTechnique);
    await platform.prepare();
    const environment = manifest.environment.type === 'procedural' ? platformSkyEnvironment(manifest) : null;
    // No world-space budget: all of the radiance cache's work is its screen stage.
    const threeGI = new ThreeGI(platform, scene, { budget: { units: 0 }, environment });
    threeGI.update(camera);
    await platform.rays.whenBuilt();
    threeGI.update(camera);

    const scenePass = pass(scene, camera);
    const stage = screenInputs(scenePass, threeGI);
    scenePass.setMRT(mrt({ output, [GI_ALBEDO_OUTPUT]: giAlbedo, [GI_NORMAL_OUTPUT]: packNormalToRGB(normalView), ...stage.outputs }));
    const composite = stage.encodeBefore(ddgi(scenePass, threeGI, { occlusion: false }) as unknown as Node);
    const frameMaterial = new NodeMaterial();
    frameMaterial.fragmentNode = composite;
    frameMaterial.name = 'radiance-cache-accuracy frame';
    const frameQuad = new QuadMesh(frameMaterial);
    const frameTarget = new RenderTarget(width, height, { type: FloatType, depthBuffer: false });
    const reader = new HdrReader(renderer, width, height);
    const accumulator = new IrradianceAccumulator(device, width, height);
    const worldCache = technique.worldCacheBackend;
    assert(worldCache?.lookup, `${sceneId}: the radiance cache has no world-cache lookup`);
    const worldCacheProbe = await WorldCacheProbe.create(device, worldCache.lookup);

    /** One frame as a user's would run: the GI frame, then the composite (scene pass, screen stage, composite). */
    const drawFrame = (): void => {
        threeGI.update(camera, 1 / 60);
        advanceThreeFrame(renderer);
        renderer.setRenderTarget(frameTarget);
        frameQuad.render(renderer);
        renderer.setRenderTarget(null);
    };
    const irradianceTexture = (): GPUTexture => {
        const probes = technique.probes;
        assert(probes, `${sceneId}: the radiance cache has no probes after a frame`);
        return probes.irradiance;
    };

    const results: Record<string, ViewResult> = {};
    try {
        for (const view of views) {
            placeCamera(view);
            const reference = {
                gi: await loadReferencePart(view, 'gi'),
                full: await loadReferencePart(view, 'full'),
                lightsDirect: await loadReferencePart(view, 'lightsDirect'),
                indirect: await loadReferencePart(view, 'indirect'),
            };
            const lightsAndBounces = addImages(reference.lightsDirect, reference.indirect);
            let referenceGiMean = 0;
            for (const value of reference.gi.rgb) referenceGiMean += value;
            referenceGiMean /= reference.gi.rgb.length;
            const exposure = SHEET_KEY / Math.max(referenceGiMean, 1e-6);

            let albedo: Float32Array | null = null;
            let passColor: Float32Array | null = null;
            let depth: Float32Array<ArrayBuffer> | null = null;
            const configurations: Record<string, ConfigurationResult> = {};
            const pictures: Record<string, RgbImage> = {};
            for (const configuration of CONFIGURATIONS) {
                Object.assign(technique.settings, DEFAULT_RADIANCE_CACHE_SETTINGS, RUN_SETTINGS);
                setCompareValues(technique.settings, PAGE_COMPARE_VALUES);
                Object.assign(technique.settings, configuration.settings);
                threeGI.budget = { units: configuration.ddgiUnits ?? 0 };
                if (ddgiTechnique) {
                    // DDGI under the cache only for E10's settings, from a fresh field each time.
                    const stacked = configuration.ddgiUnits !== undefined;
                    platform.setFallback(technique, stacked ? ddgiTechnique : undefined);
                    if (stacked) ddgiTechnique.resetHistory();
                }
                const environmentWeight = configuration.ddgiEnvironment === false ? 0 : 1;
                ddgiTechnique?.setEnvironmentWeight([environmentWeight, environmentWeight, environmentWeight]);
                // An emptied world cache and probe history: each setting starts as a camera cut does.
                technique.resetHistory();
                const step = (): void => {
                    if (!configuration.history) technique.resetScreenHistory();
                    drawFrame();
                };
                for (let frame = 0; frame < WARM_FRAMES; frame++) {
                    step();
                    if (frame % 4 === 3) await device.queue.onSubmittedWorkDone();
                }
                const asked = configuration.settings.useMultibounce === false ? null : (configuration.settings.secondBounce ?? 'gi1-rays');
                // The estimator is the hash grid's (SHaRC's cache has its own bounces).
                if (WORLD_CACHE === 'gi1-hash-grid') assert(technique.worldCache?.secondBounceInUse === asked, `${sceneId}/${view.camera}, ${configuration.name}: the cache ran ${technique.worldCache?.secondBounceInUse}, not ${asked}`);
                accumulator.clear();
                for (let frame = 0; frame < MEASURE_FRAMES; frame++) {
                    step();
                    accumulator.add(irradianceTexture());
                    if (frame % 4 === 3) await device.queue.onSubmittedWorkDone();
                }
                await device.queue.onSubmittedWorkDone();
                const meanIrradiance = await accumulator.mean();
                const lastIrradiance = halves(await readTexture(device, irradianceTexture(), 8, width, height));
                // The scene pass as the last frame drew it: the same for every setting of a still view.
                if (!albedo || !passColor || !depth) {
                    albedo = await reader.read(scenePass.getTextureNode(GI_ALBEDO_OUTPUT) as unknown as Node, false);
                    passColor = await reader.read(scenePass.getTextureNode('output') as unknown as Node, false);
                    const depthRgba = await reader.read(vec4(scenePass.getTextureNode('depth').sample(uv()).r, 0, 0, 1) as unknown as Node, false);
                    depth = new Float32Array(width * height);
                    for (let pixel = 0; pixel < depth.length; pixel++) depth[pixel] = depthRgba[pixel * 4];
                }
                const nonFinite = countNonFinite(meanIrradiance) + countNonFinite(lastIrradiance);
                assert(nonFinite === 0, `${sceneId}/${view.camera}, ${configuration.name}: ${nonFinite} non-finite irradiance values`);

                const gi = giTerm(albedo, meanIrradiance, width, height);
                const giOneFrame = giTerm(albedo, lastIrradiance, width, height);
                const compositeImage: RgbImage = { width, height, rgb: new Float32Array(width * height * 3) };
                for (let pixel = 0; pixel < width * height; pixel++) {
                    for (let channel = 0; channel < 3; channel++) compositeImage.rgb[pixel * 3 + channel] = passColor[pixel * 4 + channel] + gi.rgb[pixel * 3 + channel];
                }

                // The world cache on its own, where it has a cell for the visible surface.
                const cacheRadiance = await worldCacheProbe.look(depth, width, height, {
                    inverseViewProjection: camera.matrixWorld.clone().multiply(camera.projectionMatrixInverse).elements,
                    eye: [camera.position.x, camera.position.y, camera.position.z],
                });
                const covered = new Uint8Array(width * height);
                let surfaces = 0;
                let coveredCount = 0;
                const cacheImage: RgbImage = { width, height, rgb: new Float32Array(width * height * 3) };
                for (let pixel = 0; pixel < width * height; pixel++) {
                    if (!(depth[pixel] < 1)) continue;
                    surfaces++;
                    if (cacheRadiance[pixel * 4 + 3] > 0) {
                        covered[pixel] = 1;
                        coveredCount++;
                    }
                    for (let channel = 0; channel < 3; channel++) cacheImage.rgb[pixel * 3 + channel] = cacheRadiance[pixel * 4 + channel];
                }
                const multibounce = technique.settings.useMultibounce;
                const worldCacheNumbers: WorldCacheNumbers = {
                    ...accuracyNumbers(multibounce ? lightsAndBounces : reference.lightsDirect, cacheImage, covered),
                    coverage: surfaces > 0 ? coveredCount / surfaces : 0,
                    against: multibounce ? 'lights-direct + indirect' : 'lights-direct',
                };

                let meanGi = 0;
                for (const value of gi.rgb) meanGi += value;
                const result: ConfigurationResult = {
                    gi: accuracyNumbers(reference.gi, gi),
                    giOneFrame: accuracyNumbers(reference.gi, giOneFrame),
                    composite: accuracyNumbers(reference.full, compositeImage),
                    worldCache: worldCacheNumbers,
                    meanGi: meanGi / gi.rgb.length,
                };
                configurations[configuration.name] = result;
                pictures[configuration.name] = gi;
                log(
                    `${sceneId}/${view.camera} ${configuration.name}: GI ${formatNumbers(result.gi)}; one frame ${formatNumbers(result.giOneFrame)}; ` +
                        `composite ${formatNumbers(result.composite)}; world cache (${(worldCacheNumbers.coverage * 100).toFixed(0)}% covered, vs ${worldCacheNumbers.against}) ${formatNumbers(worldCacheNumbers)}`,
                );
            }
            Object.assign(technique.settings, DEFAULT_RADIANCE_CACHE_SETTINGS, { worldCache: WORLD_CACHE });

            const defaultPicture = pictures['default'] ?? Object.values(pictures)[0];
            let threeDirect: AccuracyNumbers | null = null;
            if (passColor) {
                const cyclesDirect: RgbImage = { width, height, rgb: new Float32Array(width * height * 3) };
                const threeColor: RgbImage = { width, height, rgb: new Float32Array(width * height * 3) };
                for (let pixel = 0; pixel < width * height; pixel++) {
                    for (let channel = 0; channel < 3; channel++) {
                        const index = pixel * 3 + channel;
                        cyclesDirect.rgb[index] = reference.full.rgb[index] - reference.gi.rgb[index];
                        threeColor.rgb[index] = passColor[pixel * 4 + channel];
                    }
                }
                threeDirect = accuracyNumbers(cyclesDirect, threeColor);
                log(`${sceneId}/${view.camera}: three without GI vs Cycles full minus GI: ${formatNumbers(threeDirect)}`);
            }
            const upsideDownMeanError8 = defaultPicture ? accuracyNumbers(upsideDown(reference.gi), defaultPicture).meanError8 : null;
            const firstResult = Object.values(configurations)[0];
            assert(
                upsideDownMeanError8 === null || !firstResult || upsideDownMeanError8 > firstResult.gi.meanError8,
                `${sceneId}/${view.camera}: the GI term agrees better with the reference upside down (${upsideDownMeanError8}); a row-order mistake`,
            );
            const noise = view.shot.render.noise?.gi ?? null;
            results[view.camera] = {
                reference: `${view.prefix}-gi.exr`,
                referenceSamples: view.info.samplesPerPixel,
                referenceNoise: noise ? { pixel: noise.pixel, block8: noise.block8 } : null,
                sameManifest: view.sameManifest,
                configurations,
                upsideDownMeanError8,
                threeDirect,
            };

            const row = document.createElement('div');
            row.style.display = 'flex';
            row.style.gap = '4px';
            sheet.append(row);
            addThumbnail(row, `${sceneId}/${view.camera}: Cycles GI (${view.prefix})`, thumbnail(reference.gi, exposure));
            for (const name of ['default', 'history off']) {
                const picture = pictures[name];
                const numbers = configurations[name]?.gi;
                if (picture && numbers) addThumbnail(row, `radiance cache, ${name}: bias ${percent(numbers.bias)}`, thumbnail(picture, exposure));
            }
            if (defaultPicture) addThumbnail(row, 'default vs Cycles: red brighter, blue darker (full at 50%)', errorThumbnail(reference.gi, defaultPicture));
        }
    } finally {
        accumulator.dispose();
        frameTarget.dispose();
        frameMaterial.dispose();
        reader.dispose();
        stage.dispose();
        threeGI.dispose();
        platform.dispose();
        await device.queue.onSubmittedWorkDone();
    }
    return results;
}

await runThreePage(
    async (context) => {
        const { renderer, data, log } = context;
        renderer.shadowMap.enabled = true;
        renderer.domElement.style.display = 'none';
        document.body.style.background = '#222';
        document.body.style.margin = '0';
        const sheet = document.createElement('div');
        sheet.style.display = 'flex';
        sheet.style.flexDirection = 'column';
        sheet.style.gap = '4px';
        document.body.append(sheet);
        data.settings = { worldCache: WORLD_CACHE, compare: PAGE_COMPARE_VALUES, warmFrames: WARM_FRAMES, measureFrames: MEASURE_FRAMES, configurations: CONFIGURATIONS.map((configuration) => configuration.name) };
        const scenes: Record<string, Record<string, ViewResult>> = {};
        const skipped: string[] = [];
        data.scenes = scenes;
        data.skipped = skipped;
        for (const sceneId of SCENES) {
            const manifest = findManifest(sceneId);
            const files = await listReferenceFiles(sceneId);
            const views: ReferenceView[] = [];
            for (const bookmark of manifest.cameras) {
                const view = await findReferenceView(files, sceneId, bookmark.name);
                if (view) views.push(view);
            }
            if (views.length === 0) {
                skipped.push(sceneId);
                log(`${sceneId}: no Cycles GI reference with a stored shot under bench/references/${sceneId}/; skipped`);
                continue;
            }
            scenes[sceneId] = await measureScene(context, sceneId, views, sheet);
        }
        assert(Object.keys(scenes).length > 0, `no scene had a reference (skipped: ${skipped.join(', ')}); link bench/references`);
    },
    { width: 1920, height: 1080 },
);
