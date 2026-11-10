/**
 * RC2.2: the radiance cache's first image through the three adapter, beside
 * DDGI's, on `cornell-box`, `two-room` and `sponza` (with curtains) from each
 * scene's first bookmark at 1280 × 720, single bounce: the probe rays' hits
 * are lit through the world cache (RC3.2), which holds one bounce until
 * GI-1.2's multi-bounce cells (RC3.3).
 *
 * One platform per scene, the viewer's loader and `ThreeGI` (the shared
 * composite harness), with DDGI as the output technique (warmed for 120
 * frames, as the composite pages do) and the radiance cache attached beside
 * it. DDGI there runs with `bounceEnergy: 0`: its probe rays return the
 * hit's emission and `albedo / π ×` direct light and the sky on a miss, never
 * the field's own light, so it holds one bounce as the radiance cache does
 * before RC3.3. A second platform per scene runs DDGI with its default (all
 * bounces) for the contact sheet only. Two pictures, each the way a three
 * user builds it:
 * - **DDGI:** `pass()` with `mrt({ output, giAlbedo, normal })` and the
 *   composite (`DDGICompositeNode`) on the GI light;
 * - **radiance cache:** the same pass with the screen inputs' outputs too
 *   (`screenInputs`), the screen stage attached with `encodeBefore`, and the
 *   composite on a GI light carrying the radiance cache's shading module, so
 *   three adds `giAlbedo × E / π` with E read by `screenUV`.
 * `ThreeGI.update` runs before each picture (the frame's `encode`, then the
 * stage's `encodeScreen`).
 *
 * Checks, by readback: the radiance cache's picture and its irradiance image
 * hold no negative, infinite or NaN value; its irradiance is not black on
 * surfaces (some light arrives); the screen stage ran once per picture; no
 * device error (the harness).
 *
 * Reported, not checked: the mean luminance of the GI term alone (the
 * composite's picture minus its pass's own colour, which holds no GI on
 * composited surfaces) over surface pixels, for one-bounce DDGI and the
 * radiance cache, and their ratio: a like-for-like brightness comparison.
 * And at a sample of the probes, each technique's E against a path-traced
 * one-bounce reference at the same points, with the D17 self-shadowing
 * check at grazing angles (one-bounce-reference.ts).
 *
 * The contact sheet the runner's screenshot captures: per scene, DDGI (all
 * bounces), DDGI (one bounce), the radiance cache, the radiance cache's
 * E / π (what a white surface would reflect) and its SH debug view, at the
 * same exposure (from one-bounce DDGI's mean).
 *
 * Correctness only (DECISIONS 2026-11-09). Through the lock:
 * `node tools/reference/bench-lock.ts run --no-cool-wait --what "rc2.2 three" -- pnpm test:gpu --filter radiance-cache-three --gpu nvidia --out bench/screenshots/rc2.2`.
 */

import type { Node } from 'three/webgpu';
import { isBackgroundDepth, mrt, normalView, output, packNormalToRGB, pass, uv, vec4 } from 'three/tsl';
import { RadianceCacheTechnique } from '../../../../packages/ddgi-core/src/techniques/radiance-cache/radiance-cache-technique.ts';
import { DEFAULT_RADIANCE_CACHE_SETTINGS } from '../../../../packages/ddgi-core/src/techniques/radiance-cache/settings.ts';
import { RADIANCE_CACHE_DEBUG_VIEW } from '../../../../packages/ddgi-core/src/techniques/radiance-cache/wgsl/debug.ts';
import { DDGICompositeNode, giAlbedo, GI_ALBEDO_OUTPUT, GI_NORMAL_OUTPUT } from '../../../../packages/ddgi-three/src/composite.ts';
import { GILight } from '../../../../packages/ddgi-three/src/gi-light.ts';
import { screenInputs } from '../../../../packages/ddgi-three/src/screen-inputs.ts';
import { assert } from '../../client.ts';
import { halves, readTexture } from '../radiance-cache-probes/readback.ts';
import { compareAtProbes, type ProbeComparison } from './one-bounce-reference.ts';
import { runThreePage, type ThreePageContext } from '../three-shared/harness.ts';
import { HdrReader, loadCompositeScene, luminanceAt, type CompositeScene } from '../three-composite-shared/composite-check.ts';

const SIZE = { width: 1280, height: 720 } as const;
const SCENES = ['cornell-box', 'two-room', 'sponza'] as const;
/** DDGI's warm-up, as the composite pages (three-composite, three-composite-sponza). */
const GI_FRAMES = 120;
const PROBES_PER_FRAME = 4096;
/**
 * Frames of the radiance cache before the picture that counts. The screen
 * probes keep no history (phase 4), but the world cache does: each cell is a
 * running mean of up to `hashGridCacheMaxSampleCount` frames (16, gi1.h:63),
 * so the picture is taken once its cells have a full history, and the
 * comparisons are against the cache's converged value, not its first frames.
 */
const WARM_FRAMES = DEFAULT_RADIANCE_CACHE_SETTINGS.hashGridCacheMaxSampleCount;
/** Contact sheet thumbnails, CSS pixels. */
const THUMBNAIL_WIDTH = 250;
/** The sheet's exposure puts one-bounce DDGI's mean surface luminance here before tone mapping. */
const SHEET_KEY = 0.4;

interface SceneResult {
    readonly ddgiMeanLuminance: number;
    readonly radianceCacheMeanLuminance: number;
    /** The GI term's mean luminance over surface pixels: picture minus the pass's own colour. */
    readonly ddgiOneBounceGiLuminance: number;
    readonly radianceCacheGiLuminance: number;
    /** Radiance cache over one-bounce DDGI, GI term only. */
    readonly giLuminanceRatio: number;
    readonly ddgiAllBouncesGiLuminance: number;
    /** At a sample of the probes, each estimate of E against a path-traced one-bounce reference (one-bounce-reference.ts). */
    readonly atProbes: ProbeComparison;
    readonly meanIrradiance: number;
    readonly surfacePixels: number;
    readonly blackSurfacePixels: number;
    readonly litBackgroundPixels: number;
    readonly pictureNonFinite: number;
    readonly pictureNegative: number;
    readonly irradianceNonFinite: number;
    readonly irradianceNegative: number;
    readonly backupHints: number;
    readonly screenStages: number;
    readonly probes: number;
}

/**
 * Linear HDR to display bytes: exposure, Reinhard, then an approximate sRGB
 * curve. Both three's render-target readback (WebGPU) and the technique's
 * textures start at the top row.
 */
function toDisplay(rgba: Float32Array, exposure: number): ImageData {
    const image = new ImageData(SIZE.width, SIZE.height);
    for (let y = 0; y < SIZE.height; y++) {
        for (let x = 0; x < SIZE.width; x++) {
            const source = (y * SIZE.width + x) * 4;
            const target = source;
            for (let channel = 0; channel < 3; channel++) {
                const value = Math.max(0, rgba[source + channel]) * exposure;
                image.data[target + channel] = Math.round(255 * Math.pow(value / (1 + value), 1 / 2.2));
            }
            image.data[target + 3] = 255;
        }
    }
    return image;
}

function addThumbnail(row: HTMLElement, caption: string, image: ImageData): void {
    const cell = document.createElement('div');
    const label = document.createElement('div');
    label.textContent = caption;
    label.style.font = '11px sans-serif';
    label.style.color = '#ddd';
    const canvas = document.createElement('canvas');
    canvas.width = image.width;
    canvas.height = image.height;
    canvas.style.width = `${THUMBNAIL_WIDTH}px`;
    canvas.style.display = 'block';
    canvas.getContext('2d')?.putImageData(image, 0, 0);
    cell.append(label, canvas);
    row.append(cell);
}

/** A composite's picture and its pass's own colour (no GI on composited surfaces), from the same frame. */
interface PictureAndPass {
    readonly picture: Float32Array;
    readonly passColour: Float32Array;
}

/** The scene pass and DDGI's composite on it, as a three user builds it. */
function ddgiPicture(scene: CompositeScene): { readonly composite: Node; readonly passColour: Node; dispose(): void } {
    const ddgiPass = pass(scene.scene, scene.camera);
    ddgiPass.setMRT(mrt({ output, [GI_ALBEDO_OUTPUT]: giAlbedo, [GI_NORMAL_OUTPUT]: packNormalToRGB(normalView) }));
    const composite = new DDGICompositeNode(ddgiPass, scene.threeGI, { occlusion: false }) as unknown as Node;
    return { composite, passColour: ddgiPass.getTextureNode('output') as unknown as Node, dispose: () => ddgiPass.dispose() };
}

/** Mean luminance of picture − pass colour over the pixels `surfaces` marks (its red is 1 on a surface). */
function meanGiLuminance(images: PictureAndPass, surfaces: Float32Array): number {
    let sum = 0;
    let count = 0;
    for (let pixel = 0; pixel < SIZE.width * SIZE.height; pixel++) {
        if (surfaces[pixel * 4] < 0.5) continue;
        sum += luminanceAt(images.picture, pixel) - luminanceAt(images.passColour, pixel);
        count++;
    }
    return sum / Math.max(count, 1);
}

/** DDGI with its default bounces, on a platform of its own: the contact sheet's first picture. */
async function renderAllBouncesDdgi(context: ThreePageContext, sceneId: string): Promise<PictureAndPass> {
    const { device, renderer, log } = context;
    const scene = await loadCompositeScene(device, sceneId, { ...SIZE, giFrames: GI_FRAMES, probesPerFrame: PROBES_PER_FRAME, log });
    scene.threeGI.budget = { units: 0 };
    const reader = new HdrReader(renderer, SIZE.width, SIZE.height);
    const ddgi = ddgiPicture(scene);
    try {
        let picture: Float32Array = new Float32Array(0);
        for (let frame = 0; frame <= WARM_FRAMES; frame++) {
            scene.threeGI.update(scene.camera, 1 / 60);
            picture = await reader.read(ddgi.composite);
        }
        return { picture, passColour: await reader.read(ddgi.passColour, false) };
    } finally {
        ddgi.dispose();
        reader.dispose();
        scene.threeGI.dispose();
        scene.platform.dispose();
        await device.queue.onSubmittedWorkDone();
    }
}

async function renderScene(context: ThreePageContext, sceneId: string, sheet: HTMLElement): Promise<SceneResult> {
    const { device, renderer, log } = context;
    const allBounces = await renderAllBouncesDdgi(context, sceneId);
    const radianceCache = new RadianceCacheTechnique();
    // One bounce in DDGI (file comment): like for like with the radiance cache before RC3.3.
    const scene = await loadCompositeScene(device, sceneId, {
        ...SIZE,
        giFrames: GI_FRAMES,
        probesPerFrame: PROBES_PER_FRAME,
        log,
        techniques: [radianceCache],
        ddgiSettings: { bounceEnergy: 0 },
    });
    // DDGI keeps its warmed field; frames from here update no probes (units 0; the floor may still run a few).
    scene.threeGI.budget = { units: 0 };
    const reader = new HdrReader(renderer, SIZE.width, SIZE.height);
    const ddgiGraph = ddgiPicture(scene);
    try {
        // The radiance cache: the screen inputs on the same kind of pass, its module's GI light for the composite.
        const rcPass = pass(scene.scene, scene.camera);
        const stage = screenInputs(rcPass, scene.threeGI);
        rcPass.setMRT(mrt({ output, [GI_ALBEDO_OUTPUT]: giAlbedo, [GI_NORMAL_OUTPUT]: packNormalToRGB(normalView), ...stage.outputs }));
        const rcLight = new GILight(radianceCache.shading);
        const rcComposite = stage.encodeBefore(new DDGICompositeNode(rcPass, { light: rcLight }, { occlusion: false }) as unknown as Node);
        // 1 where the pass drew a surface, 0 where its depth is the background (three's own test).
        const surfaceNode = vec4(isBackgroundDepth(rcPass.getTextureNode('depth').sample(uv()).r).select(0, 1), 0, 0, 1) as unknown as Node;

        const frame = async (node: Node): Promise<Float32Array> => {
            scene.threeGI.update(scene.camera, 1 / 60);
            return reader.read(node);
        };
        for (let warm = 0; warm < WARM_FRAMES; warm++) await frame(ddgiGraph.composite);
        const ddgi = await frame(ddgiGraph.composite);
        const ddgiPassColour = await reader.read(ddgiGraph.passColour, false);
        const stagesBefore = stage.stages;
        for (let warm = 0; warm < WARM_FRAMES; warm++) await frame(rcComposite);
        const picture = await frame(rcComposite);
        const rcPassColour = await reader.read(rcPass.getTextureNode('output') as unknown as Node, false);
        const screenStages = stage.stages - stagesBefore;
        assert(screenStages === WARM_FRAMES + 1, `${sceneId}: ${screenStages} screen stages for ${WARM_FRAMES + 1} pictures`);

        const probes = radianceCache.probes;
        if (!probes) throw new Error(`${sceneId}: the radiance cache has no probes after its screen stages`);
        const irradiance = halves(await readTexture(device, probes.irradiance, 8, SIZE.width, SIZE.height));
        const probeCount = new Uint32Array(await readProbeCount(device, probes.probeSpawn))[0];

        // Surfaces, from the radiance cache's pass as it last drew (no new frame).
        const surfaces = await reader.read(surfaceNode, false);

        let pictureNonFinite = 0;
        let pictureNegative = 0;
        let irradianceNonFinite = 0;
        let irradianceNegative = 0;
        let surfacePixels = 0;
        let blackSurfacePixels = 0;
        let litBackgroundPixels = 0;
        let backupHints = 0;
        let ddgiLuminance = 0;
        let rcLuminance = 0;
        let irradianceSum = 0;
        for (let pixel = 0; pixel < SIZE.width * SIZE.height; pixel++) {
            for (let channel = 0; channel < 4; channel++) {
                const value = picture[pixel * 4 + channel];
                if (!Number.isFinite(value)) pictureNonFinite++;
                else if (channel < 3 && value < 0) pictureNegative++;
                const e = irradiance[pixel * 4 + channel];
                if (!Number.isFinite(e)) irradianceNonFinite++;
                else if (channel < 3 && e < 0) irradianceNegative++;
            }
            // Both images start at the top row (toDisplay). A sky pixel gets no irradiance (the stage reads the same depth).
            if (surfaces[pixel * 4] < 0.5) {
                if (irradiance[pixel * 4] + irradiance[pixel * 4 + 1] + irradiance[pixel * 4 + 2] > 0) litBackgroundPixels++;
                continue;
            }
            surfacePixels++;
            ddgiLuminance += luminanceAt(ddgi, pixel);
            rcLuminance += luminanceAt(picture, pixel);
            const e = (irradiance[pixel * 4] + irradiance[pixel * 4 + 1] + irradiance[pixel * 4 + 2]) / 3;
            irradianceSum += e;
            if (e === 0) blackSurfacePixels++;
            if (irradiance[pixel * 4 + 3] === 0) backupHints++;
        }
        const ddgiMeanLuminance = ddgiLuminance / Math.max(surfacePixels, 1);
        assert(pictureNonFinite === 0 && pictureNegative === 0, `${sceneId}: the radiance cache's picture has ${pictureNonFinite} values not finite and ${pictureNegative} negative`);
        assert(irradianceNonFinite === 0 && irradianceNegative === 0, `${sceneId}: the irradiance image has ${irradianceNonFinite} values not finite and ${irradianceNegative} negative`);
        assert(irradianceSum > 0, `${sceneId}: no irradiance on any surface`);
        // Also says both readbacks start at the same row: flipped, Sponza's sky would sit over lit floor.
        assert(litBackgroundPixels === 0, `${sceneId}: ${litBackgroundPixels} background pixels have irradiance`);

        const ddgiOneBounceGiLuminance = meanGiLuminance({ picture: ddgi, passColour: ddgiPassColour }, surfaces);
        const radianceCacheGiLuminance = meanGiLuminance({ picture, passColour: rcPassColour }, surfaces);
        const ddgiAllBouncesGiLuminance = meanGiLuminance(allBounces, surfaces);
        const screen = stage.screen;
        if (!screen) throw new Error(`${sceneId}: the screen stage has no FrameScreen`);
        const atProbes = await compareAtProbes(device, scene, probes, screen, irradiance, SIZE);
        log(
            `${sceneId}: at ${atProbes.sampledProbes} probes, E over the one-bounce reference: the probe rays ${atProbes.raysOverReference.toFixed(3)}, ` +
                `their SH ${atProbes.shOverReference.toFixed(3)}, the pixel ${atProbes.pixelOverReference.toFixed(3)}, DDGI ${atProbes.ddgiOverReference.toFixed(3)}; ` +
                `back-face rays ${(atProbes.backFaceRayShare * 100).toFixed(2)}%`,
        );
        for (const [name, group] of Object.entries(atProbes.byView)) {
            log(
                `${sceneId}: ${name} probes ${group.probes}: rays under 1 cm ${(group.shortHitShare * 100).toFixed(3)}%, back faces ${(group.backFaceShare * 100).toFixed(2)}%, ` +
                    `E over the reference ${group.raysOverReference.toFixed(3)}, over the reference lifted 2 cm ${group.raysOverLiftedReference.toFixed(3)}`,
            );
        }
        log(
            `${sceneId}: GI term mean luminance: radiance cache ${radianceCacheGiLuminance.toPrecision(4)}, DDGI one bounce ${ddgiOneBounceGiLuminance.toPrecision(4)} ` +
                `(ratio ${(radianceCacheGiLuminance / ddgiOneBounceGiLuminance).toFixed(3)}), DDGI all bounces ${ddgiAllBouncesGiLuminance.toPrecision(4)}`,
        );

        // The contact sheet's row: one exposure for the scene, from one-bounce DDGI's picture.
        const exposure = SHEET_KEY / Math.max(ddgiMeanLuminance, 1e-6);
        const row = document.createElement('div');
        row.style.display = 'flex';
        row.style.gap = '4px';
        sheet.append(row);
        addThumbnail(row, `${sceneId}: DDGI (all bounces)`, toDisplay(allBounces.picture, exposure));
        addThumbnail(row, `${sceneId}: DDGI (1 bounce)`, toDisplay(ddgi, exposure));
        addThumbnail(row, `${sceneId}: radiance cache (1 bounce)`, toDisplay(picture, exposure));
        const whiteSurface = Float32Array.from(irradiance, (value, index) => (index % 4 === 3 ? 1 : value / Math.PI));
        addThumbnail(row, `${sceneId}: radiance cache E / pi`, toDisplay(whiteSurface, exposure));
        const debugTarget = device.createTexture({ size: [SIZE.width, SIZE.height], format: 'rgba8unorm', usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.COPY_SRC });
        const encoder = device.createCommandEncoder();
        radianceCache.encodeDebugView(encoder, RADIANCE_CACHE_DEBUG_VIEW.sphericalHarmonics, debugTarget, exposure);
        device.queue.submit([encoder.finish()]);
        const debugPixels = await readTexture(device, debugTarget, 4, SIZE.width, SIZE.height);
        debugTarget.destroy();
        addThumbnail(row, `${sceneId}: probe SH view`, new ImageData(Uint8ClampedArray.from(debugPixels), SIZE.width, SIZE.height));

        rcLight.dispose();
        stage.dispose();
        rcPass.dispose();
        return {
            ddgiMeanLuminance,
            radianceCacheMeanLuminance: rcLuminance / Math.max(surfacePixels, 1),
            ddgiOneBounceGiLuminance,
            radianceCacheGiLuminance,
            giLuminanceRatio: radianceCacheGiLuminance / ddgiOneBounceGiLuminance,
            ddgiAllBouncesGiLuminance,
            atProbes,
            meanIrradiance: irradianceSum / Math.max(surfacePixels, 1),
            surfacePixels,
            blackSurfacePixels,
            litBackgroundPixels,
            pictureNonFinite,
            pictureNegative,
            irradianceNonFinite,
            irradianceNegative,
            backupHints,
            screenStages,
            probes: probeCount,
        };
    } finally {
        ddgiGraph.dispose();
        reader.dispose();
        scene.threeGI.dispose();
        scene.platform.dispose();
        await device.queue.onSubmittedWorkDone();
    }
}

/** The probe count in front of the compacted list (`RcProbeSpawn.count`). */
async function readProbeCount(device: GPUDevice, probeSpawn: GPUBuffer): Promise<ArrayBuffer> {
    const buffer = device.createBuffer({ size: 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const encoder = device.createCommandEncoder();
    encoder.copyBufferToBuffer(probeSpawn, 0, buffer, 0, 4);
    device.queue.submit([encoder.finish()]);
    await buffer.mapAsync(GPUMapMode.READ);
    const copy = buffer.getMappedRange().slice(0);
    buffer.unmap();
    buffer.destroy();
    return copy;
}

await runThreePage(
    async (context) => {
        const { renderer, data } = context;
        renderer.shadowMap.enabled = true;
        // The sheet is the screenshot; the renderer's own canvas isn't needed on screen.
        renderer.domElement.style.display = 'none';
        document.body.style.background = '#222';
        document.body.style.margin = '0';
        const sheet = document.createElement('div');
        sheet.style.display = 'flex';
        sheet.style.flexDirection = 'column';
        sheet.style.gap = '4px';
        document.body.append(sheet);
        const results: Record<string, SceneResult> = {};
        for (const sceneId of SCENES) {
            results[sceneId] = await renderScene(context, sceneId, sheet);
            data.scenes = results;
        }
    },
    SIZE,
);
