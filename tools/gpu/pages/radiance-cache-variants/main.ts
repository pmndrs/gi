/**
 * P6.5: the radiance cache's world-cache variants side by side, so Dennis can
 * pick the default (DECISIONS 2026-11-23). A measurement, not a gate: it
 * fails only on device errors, non-finite values or no scene measured. The
 * table and the write-up are docs/design/radiance-cache-variants.md;
 * `node tools/reference/rc-variants-table.ts` turns this page's `--json`
 * report (and the timing page's) into that table.
 *
 * **The variants** are the compare panel's presets
 * (`RADIANCE_CACHE_VARIANT_PRESETS`, compare-toggles.ts): the GI-1.2 grid with
 * its own second-bounce rays; the grid over DDGI (what ships); SHaRC alone and
 * over DDGI; budgeted cells; the neural tail. Each is built from nothing the
 * way the viewer builds it (apps/viewer/src/techniques/radiance-cache-technique.ts):
 * the platform, the radiance cache with the preset's values, a DDGI volume
 * (the manifest's) stacked under it at 256 probes a frame unless the preset
 * runs the cache alone, `ThreeGI`, a `pass()` with the screen inputs, and the
 * composite drawn every frame. The scene is set up as the accuracy page sets
 * it up (radiance-cache-accuracy: the references' cameras and size, 1920 ×
 * 1080; point lights without a range window; the sky dome out of GI; Bistro's
 * textures capped at 1024).
 *
 * **What is measured** per scene with a Cycles reference, per variant, on the
 * irradiance the composite shows (the denoiser's output; the raw interpolation
 * too for the accuracy, to line up with RC4.4's numbers):
 *
 * 1. warm-up: from an empty cache, the frame from which the frame's mean E
 *    stays within 5% of its measured mean (`WARM_FRAMES` frames, then);
 * 2. accuracy: the GI term (albedo × E / π) of E averaged over
 *    `MEASURE_FRAMES` frames against `<prefix>-gi.exr`: bias, 8 × 8 mean and
 *    p95 error (radiance-cache-accuracy/accuracy.ts); and of the last frame
 *    alone; and the share of surface pixels where the world cache has a cell;
 * 3. flicker when still: `FLICKER_PAIRS` pairs of consecutive frames, per
 *    surface pixel |E_t − E_t−1| / max(E_t−1, 5% of the mean), mean and p95,
 *    and the whole frame's change (metrics.ts `flicker`);
 * 4. the lights double: every three light's intensity × 2 (the sun where the
 *    scene has one; Cornell's emissive ceiling: its emissive intensity × 2),
 *    then `AFTER_FRAMES` frames of 8 × 8 block means: frames to 50% and 90%
 *    of the step, the share left at 1, 4 and 16 frames, the overshoot
 *    (metrics.ts `stepResponse`); the lights are put back after;
 * 5. a box moves: a box (12% of the volume's smallest side, light grey) is
 *    put on the surface below the screen's centre, settles for
 *    `BOX_SETTLE_FRAMES`, then moves sideways by 1.5 times its size; the same
 *    step numbers over the blocks the move changed, leaving out the blocks
 *    the box itself covers before or after (they change surface at once), so
 *    what is measured is the light around it: its old shadow lit again (the
 *    ghost) and its new shadow; the box is taken out after;
 * 6. memory: the radiance cache's GPU bytes and DDGI's, and what ran
 *    (the second bounce the grid used, whether the neural tail ran, warnings).
 *
 * GPU time is on the timing page (radiance-cache-variants-timing), which needs
 * a VALID clock verdict; this page's numbers don't depend on the clock.
 *
 * **Picking** (vite env): `VITE_RC_VARIANTS=gi12-rays,sharc` (preset ids),
 * `VITE_RC_VARIANT_SCENES=two-room,cornell-box-point`, `VITE_RC_VARIANT_WARM_FRAMES`,
 * `VITE_RC_VARIANT_MEASURE_FRAMES`, `VITE_RC_VARIANT_AFTER_FRAMES`; `VITE_RC_COMPARE`
 * applies over every variant (radiance-cache-probes/compare-env.ts). Scenes
 * without a reference are listed and skipped.
 *
 * Through the lock:
 * `node tools/reference/bench-lock.ts run --no-cool-wait --what "p6.5 variants" -- pnpm test:gpu --filter radiance-cache-variants --gpu nvidia --json bench/results/rc-variants.json`.
 */

import { BoxGeometry, FloatType, Mesh, MeshStandardMaterial, NodeMaterial, PerspectiveCamera, QuadMesh, Raycaster, RenderTarget, Vector2, Vector3, type Light, type Material, type Node, type Object3D } from 'three/webgpu';
import { mrt, normalView, output, packNormalToRGB, pass, uv, vec4 } from 'three/tsl';
import { GIPlatformImpl } from '../../../../packages/ddgi-core/src/platform/gi-platform.ts';
import { DDGITechnique } from '../../../../packages/ddgi-core/src/techniques/ddgi/ddgi-technique.ts';
import { RadianceCacheTechnique } from '../../../../packages/ddgi-core/src/techniques/radiance-cache/radiance-cache-technique.ts';
import { hostCompareValue, RADIANCE_CACHE_VARIANT_PRESETS, setCompareValues, type CompareValues, type RadianceCacheVariantPreset } from '../../../../packages/ddgi-core/src/techniques/radiance-cache/compare-toggles.ts';
import { ddgi, giAlbedo, GI_ALBEDO_OUTPUT, GI_NORMAL_OUTPUT } from '../../../../packages/ddgi-three/src/composite.ts';
import { screenInputs } from '../../../../packages/ddgi-three/src/screen-inputs.ts';
import { ThreeGI } from '../../../../packages/ddgi-three/src/three-gi.ts';
import { loadScene } from '../../../../apps/viewer/src/scene/load-scene.ts';
import { STACKED_DDGI_PROBES_PER_FRAME } from '../../../../apps/viewer/src/techniques/radiance-cache-technique.ts';
import type { SceneManifest } from '../../../../scenes/src/manifest.ts';
import { platformSkyEnvironment } from '../../../../scenes/src/sky/sky-image.ts';
import { assert } from '../../client.ts';
import { accuracyNumbers, giTerm, type AccuracyNumbers, type RgbImage } from '../radiance-cache-accuracy/accuracy.ts';
import { IrradianceAccumulator, WorldCacheProbe } from '../radiance-cache-accuracy/gpu.ts';
import { findReferenceView, listReferenceFiles, loadReferencePart, type ReferenceView } from '../radiance-cache-accuracy/references.ts';
import { PAGE_COMPARE_VALUES } from '../radiance-cache-probes/compare-env.ts';
import { halves, readTexture } from '../radiance-cache-probes/readback.ts';
import { runThreePage, type ThreePageContext } from '../three-shared/harness.ts';
import { advanceThreeFrame, findManifest, HdrReader, SHOT_FAR, SHOT_NEAR } from '../three-composite-shared/composite-check.ts';
import { BlockMeans } from './gpu.ts';
import { flicker, settleFrame, stepResponse, type Flicker, type StepResponse } from './metrics.ts';

const env = import.meta.env as Record<string, unknown>;
const listFromEnv = (name: string): string[] | null => (typeof env[name] === 'string' && env[name] ? (env[name] as string).split(',').map((item) => item.trim()).filter(Boolean) : null);
const numberFromEnv = (name: string, fallback: number): number => (typeof env[name] === 'string' && env[name] ? Number(env[name]) : fallback);

/** The scenes with Cycles references: the accuracy page's list (the point-light Cornell box first), and the furnished room. */
const DEFAULT_SCENES = ['cornell-box-point', 'cornell-box', 'two-room', 'sponza', 'bistro-interior', 'bistro-exterior', 'furnished-room'];
const SCENES = listFromEnv('VITE_RC_VARIANT_SCENES') ?? DEFAULT_SCENES;
const variantIds = listFromEnv('VITE_RC_VARIANTS');
const VARIANTS = variantIds ? RADIANCE_CACHE_VARIANT_PRESETS.filter((preset) => variantIds.includes(preset.id)) : RADIANCE_CACHE_VARIANT_PRESETS;
/** T2.17's cap on three's own textures where they don't fit the 6 GB RTX (as the accuracy page). */
const TEXTURE_CAPS: Record<string, number> = { 'bistro-exterior': 1024, 'bistro-interior': 1024 };
/**
 * Frames from an empty cache before measuring. Longer than the accuracy page's 90: DDGI under the
 * cache updates 256 probes a frame, so Sponza's field (about 17,500 probes) takes about 70 frames
 * for one round and Bistro exterior's about 100; the viewer's user waits that long too.
 */
const WARM_FRAMES = numberFromEnv('VITE_RC_VARIANT_WARM_FRAMES', 240);
const MEASURE_FRAMES = numberFromEnv('VITE_RC_VARIANT_MEASURE_FRAMES', 60);
/** Pairs of consecutive frames read whole for the flicker. */
const FLICKER_PAIRS = 6;
/** Frames before a change whose block means are "before". */
const BEFORE_FRAMES = 10;
/** Frames after a change; the last 30 of them are the settled picture. */
const AFTER_FRAMES = numberFromEnv('VITE_RC_VARIANT_AFTER_FRAMES', 150);
/** Frames for the doubled lights' change to settle back, and for a new box to settle in, before the box moves. */
const RESTORE_FRAMES = 90;
const BOX_SETTLE_FRAMES = 150;
/** The box's side as a share of the volume's smallest side; it moves sideways by `BOX_MOVE` sides. */
const BOX_SIDE_SHARE = 0.12;
const BOX_MOVE = 1.5;
/** Warm-up: the frame mean within this share of the measured mean. */
const WARM_TOLERANCE = 0.05;

interface StepResult {
    /** What changed (which lights, or where the box went). */
    readonly what: string;
    readonly response: StepResponse | null;
}

interface VariantResult {
    readonly preset: string;
    readonly values: CompareValues;
    readonly ddgiStacked: boolean;
    /** What the grid's second bounce came from (null for SHaRC, or multibounce off). */
    readonly secondBounceInUse: string | null;
    readonly neuralTailRan: boolean;
    readonly worldCache: string;
    readonly warmupFrames: number | null;
    /** The frame's mean E per frame from the empty cache (every 4th frame, to keep the report small). */
    readonly warmupCurve: number[];
    readonly gi: AccuracyNumbers;
    readonly giRaw: AccuracyNumbers;
    readonly giOneFrame: AccuracyNumbers;
    /** Share of surface pixels where the world cache has a cell. */
    readonly coverage: number;
    readonly flicker: Flicker;
    readonly lightsDouble: StepResult;
    readonly boxMoves: StepResult;
    readonly memoryMiB: { readonly radianceCache: number; readonly ddgi: number; readonly total: number };
    readonly warnings: string[];
    readonly seconds: number;
}

const percent = (value: number): string => `${value >= 0 ? '+' : ''}${(value * 100).toFixed(1)}%`;
const formatNumbers = (numbers: AccuracyNumbers): string => `bias ${percent(numbers.bias)}, 8x8 mean ${(numbers.meanError8 * 100).toFixed(1)}%, p95 ${(numbers.p95Error8 * 100).toFixed(0)}%`;
const formatStep = (step: StepResult): string =>
    step.response
        ? `${step.what}: 50% in ${step.response.framesTo50 ?? '>'} frames, 90% in ${step.response.framesTo90 ?? '>'}, left at 4 frames ${(step.response.residualAt4 * 100).toFixed(0)}%, at 16 ${(step.response.residualAt16 * 100).toFixed(0)}%, overshoot ${(step.response.overshoot * 100).toFixed(0)}% (x${step.response.endOverBefore.toFixed(2)}, ${step.response.changedBlocks} blocks)`
        : `${step.what}: nothing changed`;

/** Mean of rgb per pixel from an rgba float image. */
function meanRgb(rgba: Float32Array): Float32Array {
    const out = new Float32Array(rgba.length / 4);
    for (let pixel = 0; pixel < out.length; pixel++) out[pixel] = (rgba[pixel * 4] + rgba[pixel * 4 + 1] + rgba[pixel * 4 + 2]) / 3;
    return out;
}

function meanOfBlocks(blocks: Float32Array): number {
    let sum = 0;
    for (const value of blocks) sum += value;
    return sum / Math.max(1, blocks.length);
}

function countNonFinite(values: Float32Array): number {
    let count = 0;
    for (const value of values) if (!Number.isFinite(value)) count++;
    return count;
}

/** The scene's three lights GI takes (not ambient, hemisphere or rect-area lights: GI lights the emissive Cornell box from its ceiling's emissive surface). */
function sceneLights(scene: Object3D): Light[] {
    const lights: Light[] = [];
    scene.traverse((object) => {
        // ThreeGI's own GILight carries GI into three's lighting; it is not a scene light.
        const light = object as Light & { isAmbientLight?: boolean; isHemisphereLight?: boolean; isLightProbe?: boolean; isGILight?: boolean; isRectAreaLight?: boolean };
        if (light.isLight && !light.isAmbientLight && !light.isHemisphereLight && !light.isLightProbe && !light.isGILight && !light.isRectAreaLight && light.intensity > 0) lights.push(light);
    });
    return lights;
}

/** Materials of the content that emit light (Cornell's ceiling), for a scene with no three light to double. */
function emissiveMaterials(content: Object3D): MeshStandardMaterial[] {
    const found = new Set<MeshStandardMaterial>();
    content.traverse((object) => {
        const mesh = object as Mesh;
        if (!mesh.isMesh) return;
        for (const material of Array.isArray(mesh.material) ? mesh.material : [mesh.material]) {
            const standard = material as MeshStandardMaterial;
            if (standard.emissive && standard.emissiveIntensity > 0 && standard.emissive.r + standard.emissive.g + standard.emissive.b > 0) found.add(standard);
        }
    });
    return [...found];
}

/** Multiplies the scene's light by `factor` (its three lights, or with none its emitters); says what it changed. */
function scaleLights(scene: Object3D, content: Object3D, factor: number): string {
    const lights = sceneLights(scene);
    if (lights.length > 0) {
        for (const light of lights) light.intensity *= factor;
        return lights.map((light) => light.type).join(', ');
    }
    const emitters = emissiveMaterials(content);
    for (const material of emitters) {
        material.emissiveIntensity *= factor;
        material.needsUpdate = true;
    }
    return emitters.length > 0 ? `${emitters.length} emissive material(s)` : 'nothing (no lights or emitters)';
}

interface BoxPlacement {
    readonly start: Vector3;
    readonly end: Vector3;
    readonly side: number;
}

/** Where the box goes: on the surface seen a little below the screen's centre, then sideways along the screen. */
function placeBox(manifest: SceneManifest, camera: PerspectiveCamera, content: Object3D): BoxPlacement | null {
    const side = BOX_SIDE_SHARE * Math.min(...manifest.volume.size);
    const raycaster = new Raycaster();
    raycaster.setFromCamera(new Vector2(0, -0.3), camera);
    const hit = raycaster.intersectObject(content, true).find((candidate) => candidate.face);
    if (!hit?.face) return null;
    const normal = hit.face.normal.clone().transformDirection(hit.object.matrixWorld).normalize();
    if (normal.dot(raycaster.ray.direction) > 0) normal.negate();
    const start = hit.point.clone().addScaledVector(normal, side * 0.51);
    const right = new Vector3().setFromMatrixColumn(camera.matrixWorld, 0);
    right.addScaledVector(normal, -right.dot(normal)).normalize();
    return { start, end: start.clone().addScaledVector(right, BOX_MOVE * side), side };
}

interface LoadedScene {
    readonly id: string;
    readonly manifest: SceneManifest;
    readonly scene: Object3D & { isScene: true };
    readonly content: Object3D;
    readonly camera: PerspectiveCamera;
    readonly view: ReferenceView;
    readonly referenceGi: RgbImage;
}

async function measureVariant(context: ThreePageContext, loaded: LoadedScene, preset: RadianceCacheVariantPreset): Promise<VariantResult> {
    const { device, renderer, log } = context;
    const { manifest, scene, content, camera } = loaded;
    const { width, height } = loaded.view.shot.camera;
    const started = performance.now();
    const values: CompareValues = { ...preset.values, ...PAGE_COMPARE_VALUES };
    const stacked = !hostCompareValue(values, 'host.cacheAlone');

    // The variant's warnings (the neural tail off, a fallback missing): what a user would see in the console.
    const warnings: string[] = [];
    const consoleWarn = console.warn;
    console.warn = (...parts: unknown[]) => {
        const text = parts.map(String).join(' ');
        if (!warnings.includes(text)) warnings.push(text);
        consoleWarn.apply(console, parts);
    };

    const platform = new GIPlatformImpl(device, { label: `radiance-cache-variants ${loaded.id} ${preset.id}` });
    const technique = platform.addTechnique(new RadianceCacheTechnique());
    setCompareValues(technique.settings, values);
    platform.setOutput(technique);
    const ddgiTechnique = stacked ? platform.addTechnique(new DDGITechnique({ volume: { center: manifest.volume.center, size: manifest.volume.size, spacing: manifest.volume.probeSpacing } })) : null;
    if (ddgiTechnique) platform.setFallback(technique, ddgiTechnique);
    await platform.prepare();
    const environment = manifest.environment.type === 'procedural' ? platformSkyEnvironment(manifest) : null;
    const threeGI = new ThreeGI(platform, scene as never, { budget: { units: stacked ? STACKED_DDGI_PROBES_PER_FRAME : 0 }, environment });
    threeGI.update(camera);
    await platform.rays.whenBuilt();
    threeGI.update(camera);

    const scenePass = pass(scene as never, camera);
    const stage = screenInputs(scenePass, threeGI);
    scenePass.setMRT(mrt({ output, [GI_ALBEDO_OUTPUT]: giAlbedo, [GI_NORMAL_OUTPUT]: packNormalToRGB(normalView), ...stage.outputs }));
    const composite = stage.encodeBefore(ddgi(scenePass, threeGI, { occlusion: false }) as unknown as Node);
    const frameMaterial = new NodeMaterial();
    frameMaterial.fragmentNode = composite;
    frameMaterial.name = 'radiance-cache-variants frame';
    const frameQuad = new QuadMesh(frameMaterial);
    const frameTarget = new RenderTarget(width, height, { type: FloatType, depthBuffer: false });
    const reader = new HdrReader(renderer, width, height);
    const shownSums = new IrradianceAccumulator(device, width, height);
    const rawSums = new IrradianceAccumulator(device, width, height);
    const blockMeans = new BlockMeans(device, width, height);
    const box = new Mesh(new BoxGeometry(1, 1, 1), new MeshStandardMaterial({ color: 0xb0b0b0, roughness: 1, metalness: 0 }));
    box.name = 'p6.5 moving box';

    const drawFrame = (): void => {
        threeGI.update(camera, 1 / 60);
        advanceThreeFrame(renderer);
        renderer.setRenderTarget(frameTarget);
        frameQuad.render(renderer);
        renderer.setRenderTarget(null);
    };
    const probes = () => {
        const current = technique.probes;
        assert(current, `${loaded.id} ${preset.id}: the radiance cache has no probes after a frame`);
        return current;
    };
    /** The irradiance the composite reads: the denoiser's output when it runs. */
    const shown = (): GPUTexture => probes().denoisedIrradiance ?? probes().irradiance;
    const run = async (frames: number): Promise<void> => {
        for (let frame = 0; frame < frames; frame++) {
            drawFrame();
            if (frame % 4 === 3) await device.queue.onSubmittedWorkDone();
        }
        await device.queue.onSubmittedWorkDone();
    };
    const blockFrames = async (frames: number): Promise<Float32Array[]> => {
        const out: Float32Array[] = [];
        for (let frame = 0; frame < frames; frame++) {
            drawFrame();
            out.push(await blockMeans.read(shown()));
        }
        return out;
    };
    const averageBlocks = (frames: readonly Float32Array[]): Float32Array => {
        const mean = new Float32Array(frames[0].length);
        for (const frame of frames) for (let block = 0; block < mean.length; block++) mean[block] += frame[block] / frames.length;
        return mean;
    };
    const readShown = async (): Promise<Float32Array> => meanRgb(halves(await readTexture(device, shown(), 8, width, height)));
    /** The scene pass's depth as the last frame drew it. */
    const readDepth = async (): Promise<Float32Array<ArrayBuffer>> => {
        const rgba = await reader.read(vec4(scenePass.getTextureNode('depth').sample(uv()).r, 0, 0, 1) as unknown as Node, false);
        const out = new Float32Array(width * height);
        for (let pixel = 0; pixel < out.length; pixel++) out[pixel] = rgba[pixel * 4];
        return out;
    };

    try {
        // 1. Warm-up from an empty cache (and DDGI's field from nothing).
        technique.resetHistory();
        const warmMeans: number[] = [];
        for (let frame = 0; frame < WARM_FRAMES; frame++) {
            drawFrame();
            warmMeans.push(meanOfBlocks(await blockMeans.read(shown())));
        }

        // 2. Accuracy over the measured frames.
        shownSums.clear();
        rawSums.clear();
        const measureMeans: number[] = [];
        for (let frame = 0; frame < MEASURE_FRAMES; frame++) {
            drawFrame();
            shownSums.add(shown());
            rawSums.add(probes().irradiance);
            if (frame % 4 === 3) measureMeans.push(meanOfBlocks(await blockMeans.read(shown())));
        }
        await device.queue.onSubmittedWorkDone();
        const settledMean = measureMeans.reduce((sum, value) => sum + value, 0) / Math.max(1, measureMeans.length);
        const meanShown = await shownSums.mean();
        const meanRaw = await rawSums.mean();
        const lastShown = halves(await readTexture(device, shown(), 8, width, height));
        const nonFinite = countNonFinite(meanShown) + countNonFinite(meanRaw) + countNonFinite(lastShown);
        assert(nonFinite === 0, `${loaded.id} ${preset.id}: ${nonFinite} non-finite irradiance values`);
        const albedo = await reader.read(scenePass.getTextureNode(GI_ALBEDO_OUTPUT) as unknown as Node, false);
        const depth = await readDepth();
        const surface = new Uint8Array(width * height);
        for (let pixel = 0; pixel < depth.length; pixel++) surface[pixel] = depth[pixel] < 1 ? 1 : 0;
        const gi = accuracyNumbers(loaded.referenceGi, giTerm(albedo, meanShown, width, height));
        const giRaw = accuracyNumbers(loaded.referenceGi, giTerm(albedo, meanRaw, width, height));
        const giOneFrame = accuracyNumbers(loaded.referenceGi, giTerm(albedo, lastShown, width, height));

        const backend = technique.worldCacheBackend;
        let coverage = 0;
        if (backend?.lookup) {
            const worldCacheProbe = await WorldCacheProbe.create(device, backend.lookup);
            const cacheRadiance = await worldCacheProbe.look(depth, width, height, {
                inverseViewProjection: camera.matrixWorld.clone().multiply(camera.projectionMatrixInverse).elements,
                eye: [camera.position.x, camera.position.y, camera.position.z],
            });
            let surfaces = 0;
            let covered = 0;
            for (let pixel = 0; pixel < surface.length; pixel++) {
                if (!surface[pixel]) continue;
                surfaces++;
                if (cacheRadiance[pixel * 4 + 3] > 0) covered++;
            }
            coverage = surfaces > 0 ? covered / surfaces : 0;
        }

        // 3. Flicker when still: consecutive frames read whole.
        const pairs: [Float32Array, Float32Array][] = [];
        for (let pair = 0; pair < FLICKER_PAIRS; pair++) {
            drawFrame();
            const previous = await readShown();
            drawFrame();
            pairs.push([previous, await readShown()]);
            await run(3);
        }
        const still = flicker(pairs, surface);

        // 4. The lights double.
        const beforeLights = averageBlocks(await blockFrames(BEFORE_FRAMES));
        const doubled = scaleLights(scene, content, 2);
        const afterLights = await blockFrames(AFTER_FRAMES);
        scaleLights(scene, content, 0.5);
        const lightsDouble: StepResult = { what: `x2 ${doubled}`, response: stepResponse(beforeLights, afterLights) };
        await run(RESTORE_FRAMES);

        // 5. A box moves.
        let boxMoves: StepResult = { what: 'no surface below the centre for the box', response: null };
        const placement = placeBox(manifest, camera, content);
        if (placement) {
            box.scale.setScalar(placement.side);
            box.position.copy(placement.start);
            scene.add(box);
            box.updateMatrixWorld(true);
            drawFrame();
            await platform.rays.whenBuilt();
            await run(BOX_SETTLE_FRAMES);
            const beforeBox = averageBlocks(await blockFrames(BEFORE_FRAMES));
            const footprint = new Uint8Array(blockMeans.blocksX * blockMeans.blocksY);
            const markFootprint = (boxDepth: Float32Array): void => {
                for (let pixel = 0; pixel < boxDepth.length; pixel++) {
                    if (Math.abs(boxDepth[pixel] - depth[pixel]) > 1e-6) footprint[Math.floor(pixel / width / 8) * blockMeans.blocksX + Math.floor((pixel % width) / 8)] = 1;
                }
            };
            markFootprint(await readDepth());
            box.position.copy(placement.end);
            box.updateMatrixWorld(true);
            const afterBox = await blockFrames(AFTER_FRAMES);
            markFootprint(await readDepth());
            let footprintBlocks = 0;
            for (const marked of footprint) footprintBlocks += marked;
            boxMoves = {
                what: `a ${placement.side.toFixed(2)} m box moved ${(BOX_MOVE * placement.side).toFixed(2)} m (its own ${footprintBlocks} blocks left out)`,
                response: stepResponse(beforeBox, afterBox, undefined, footprint),
            };
        }

        const radianceCacheBytes = technique.memoryBytes;
        const ddgiBytes = ddgiTechnique?.memoryBytes ?? 0;
        const result: VariantResult = {
            preset: preset.id,
            values,
            ddgiStacked: stacked,
            secondBounceInUse: technique.worldCache?.secondBounceInUse ?? null,
            neuralTailRan: technique.neuralTail !== null,
            worldCache: technique.settings.worldCache,
            warmupFrames: settleFrame(warmMeans, settledMean, WARM_TOLERANCE),
            warmupCurve: warmMeans.filter((_, frame) => frame % 4 === 3).map((value) => Number(value.toPrecision(4))),
            gi,
            giRaw,
            giOneFrame,
            coverage,
            flicker: still,
            lightsDouble,
            boxMoves,
            memoryMiB: { radianceCache: radianceCacheBytes / 2 ** 20, ddgi: ddgiBytes / 2 ** 20, total: (radianceCacheBytes + ddgiBytes) / 2 ** 20 },
            warnings: [...warnings],
            seconds: (performance.now() - started) / 1000,
        };
        log(
            `${loaded.id}/${loaded.view.camera} ${preset.id}: GI ${formatNumbers(gi)} (raw ${formatNumbers(giRaw)}; one frame ${formatNumbers(giOneFrame)}); ` +
                `warm-up ${result.warmupFrames ?? '>'} frames; coverage ${(coverage * 100).toFixed(0)}%; flicker ${(still.meanPerPixel * 100).toFixed(2)}% per pixel (p95 ${(still.p95PerPixel * 100).toFixed(1)}%), ` +
                `${(still.meanOfFrame * 100).toFixed(3)}% of the frame; ${formatStep(lightsDouble)}; ${formatStep(boxMoves)}; ` +
                `memory ${result.memoryMiB.total.toFixed(0)} MiB (DDGI ${result.memoryMiB.ddgi.toFixed(0)}); second bounce ${result.secondBounceInUse}; neural tail ${result.neuralTailRan ? 'ran' : 'off'}` +
                (warnings.length > 0 ? `; warnings: ${warnings.join(' | ')}` : ''),
        );
        return result;
    } finally {
        console.warn = consoleWarn;
        scene.remove(box);
        box.geometry.dispose();
        (box.material as Material).dispose();
        blockMeans.dispose();
        shownSums.dispose();
        rawSums.dispose();
        frameTarget.dispose();
        frameMaterial.dispose();
        reader.dispose();
        stage.dispose();
        threeGI.dispose();
        platform.dispose();
        await device.queue.onSubmittedWorkDone();
    }
}

async function loadForView(context: ThreePageContext, sceneId: string, view: ReferenceView): Promise<LoadedScene> {
    const { width, height } = view.shot.camera;
    context.renderer.setSize(width, height, false);
    const manifest = findManifest(sceneId);
    const maxTextureSize = TEXTURE_CAPS[sceneId];
    const loaded = await loadScene(manifest, () => {}, maxTextureSize === undefined ? {} : { maxTextureSize });
    const scene = loaded.scene;
    // As the accuracy page: Cycles lights point and spot lights with no range window; the sky dome is for viewing.
    scene.traverse((object) => {
        const light = object as Object3D & { isPointLight?: boolean; isSpotLight?: boolean; distance?: number };
        if (light.isPointLight || light.isSpotLight) light.distance = 0;
    });
    const content = new Set<Object3D>();
    loaded.content.traverse((object) => content.add(object));
    scene.traverse((object) => {
        if ((object as Mesh).isMesh && !content.has(object)) object.userData.giContributes = false;
    });
    const shot = view.shot.camera;
    const camera = new PerspectiveCamera(shot.fovYDeg, width / height, SHOT_NEAR, SHOT_FAR);
    camera.position.set(...shot.position);
    camera.lookAt(...shot.target);
    camera.updateProjectionMatrix();
    camera.updateMatrixWorld(true);
    scene.updateMatrixWorld(true);
    return { id: sceneId, manifest, scene: scene as LoadedScene['scene'], content: loaded.content, camera, view, referenceGi: await loadReferencePart(view, 'gi') };
}

await runThreePage(
    async (context) => {
        const { renderer, data, log } = context;
        renderer.shadowMap.enabled = true;
        renderer.domElement.style.display = 'none';
        data.settings = {
            variants: VARIANTS.map((preset) => ({ id: preset.id, label: preset.label, values: preset.values, needs: preset.needs })),
            compare: PAGE_COMPARE_VALUES,
            warmFrames: WARM_FRAMES,
            measureFrames: MEASURE_FRAMES,
            afterFrames: AFTER_FRAMES,
            flickerPairs: FLICKER_PAIRS,
            stackedDdgiProbesPerFrame: STACKED_DDGI_PROBES_PER_FRAME,
            shaderF16: context.device.features.has('shader-f16'),
        };
        const scenes: Record<string, { view: string; reference: string; sameManifest: boolean | null; variants: Record<string, VariantResult> }> = {};
        const skipped: string[] = [];
        data.scenes = scenes;
        data.skipped = skipped;
        for (const sceneId of SCENES) {
            const manifest = findManifest(sceneId);
            const files = await listReferenceFiles(sceneId);
            let view: ReferenceView | null = null;
            // One view per scene: the first bookmark with a reference.
            for (const bookmark of manifest.cameras) {
                view = await findReferenceView(files, sceneId, bookmark.name);
                if (view) break;
            }
            if (!view) {
                skipped.push(sceneId);
                log(`${sceneId}: no Cycles reference with a stored shot under bench/references/${sceneId}/; skipped`);
                continue;
            }
            const loaded = await loadForView(context, sceneId, view);
            const variants: Record<string, VariantResult> = {};
            scenes[sceneId] = { view: view.camera, reference: `${view.prefix}-gi.exr`, sameManifest: view.sameManifest, variants };
            for (const preset of VARIANTS) variants[preset.id] = await measureVariant(context, loaded, preset);
        }
        assert(Object.keys(scenes).length > 0, `no scene had a reference (skipped: ${skipped.join(', ')}); link bench/references`);
    },
    { width: 1920, height: 1080 },
);
