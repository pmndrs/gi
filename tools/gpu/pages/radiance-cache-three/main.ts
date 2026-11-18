/**
 * RC2.2: the radiance cache's first image through the three adapter, beside
 * DDGI's, on `cornell-box`, `two-room` and `sponza` (with curtains) from each
 * scene's first bookmark at 1280 × 720: screen probes lit through the world
 * cache (RC3.2), single bounce until RC3.3.
 *
 * One platform per scene, the viewer's loader and `ThreeGI` (the shared
 * composite harness), with DDGI as the output technique (warmed for 120
 * frames, as the composite pages do) and the radiance cache attached beside
 * it. Two pictures, each the way a three user builds it:
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
 * hold no negative, infinite or NaN value after its frames 1, 30 and 60 (the
 * world cache fills over frames); its irradiance is not black on surfaces
 * (some light arrives); the screen stage ran once per picture; no device
 * error (the harness). The sheet shows frame 60.
 *
 * And per pixel, at frame 60: the radiance cache's picture minus its pass's
 * own colour (GI off) equals `giAlbedo × E / π` with E the irradiance image's
 * texel at the same pixel, so a flipped or offset `screenUV`, a lost π or a
 * wrong weight fails (RC2.2 review).
 *
 * The contact sheet the runner's screenshot captures: per scene, DDGI, the
 * radiance cache, the radiance cache's E / π (what a white surface would
 * reflect) and its SH debug view, at the same exposure (from DDGI's mean).
 *
 * Correctness only (DECISIONS 2026-11-09). Through the lock:
 * `node tools/reference/bench-lock.ts run --no-cool-wait --what "rc2.2 three" -- pnpm test:gpu --filter radiance-cache-three --gpu nvidia --out bench/screenshots/rc2.2`.
 */

import type { Node } from 'three/webgpu';
import { isBackgroundDepth, mrt, normalView, output, packNormalToRGB, pass, uv, vec4 } from 'three/tsl';
import { RadianceCacheTechnique } from '../../../../packages/ddgi-core/src/techniques/radiance-cache/radiance-cache-technique.ts';
import { RADIANCE_CACHE_DEBUG_VIEW } from '../../../../packages/ddgi-core/src/techniques/radiance-cache/wgsl/debug.ts';
import { DDGICompositeNode, giAlbedo, GI_ALBEDO_OUTPUT, GI_NORMAL_OUTPUT } from '../../../../packages/ddgi-three/src/composite.ts';
import { GILight } from '../../../../packages/ddgi-three/src/gi-light.ts';
import { screenInputs } from '../../../../packages/ddgi-three/src/screen-inputs.ts';
import { assert } from '../../client.ts';
import { halves, readTexture } from '../radiance-cache-probes/readback.ts';
import { runThreePage, type ThreePageContext } from '../three-shared/harness.ts';
import { HdrReader, loadCompositeScene, luminanceAt } from '../three-composite-shared/composite-check.ts';

const SIZE = { width: 1280, height: 720 } as const;
const SCENES = ['cornell-box', 'two-room', 'sponza'] as const;
/** DDGI's warm-up, as the composite pages (three-composite, three-composite-sponza). */
const GI_FRAMES = 120;
const PROBES_PER_FRAME = 4096;
/** DDGI's pictures before the one that counts (its field is already warm). */
const WARM_FRAMES = 3;
/** The radiance cache's frames at which its picture and irradiance are scanned; the last is the sheet's. */
const SCAN_FRAMES = [1, 30, 60] as const;
/** The composite check: |(picture − pass colour) − giAlbedo × E / π| within this share of the expected value... */
const COMPOSITE_RELATIVE_TOLERANCE = 1e-3;
/** ...plus this share of the picture's value (f32 sums of half-float inputs). */
const COMPOSITE_PICTURE_TOLERANCE = 1e-4;
/** Pixels allowed past it (edge texels whose filtered sample differs). */
const COMPOSITE_OUTLIER_FRACTION = 0.001;
/** Contact sheet thumbnails, CSS pixels. */
const THUMBNAIL_WIDTH = 300;
/** The sheet's exposure puts DDGI's mean surface luminance here before tone mapping. */
const SHEET_KEY = 0.4;

interface SceneResult {
    readonly ddgiMeanLuminance: number;
    readonly radianceCacheMeanLuminance: number;
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
    /** The per-pixel composite check: pixels compared, those past the tolerance, the largest error relative to the expected GI. */
    readonly composite: { pixels: number; outliers: number; worstRelative: number; meanGi: number };
    /** NaN or infinite, and negative, values per scanned frame: picture, then irradiance. */
    readonly scans: readonly { frame: number; meanIrradiance: number; pictureNonFinite: number; pictureNegative: number; irradianceNonFinite: number; irradianceNegative: number }[];
}

/** NaN or infinite values (any channel) and negative ones (rgb) in an RGBA float image, and its mean rgb over every pixel. */
function countBad(values: Float32Array): { nonFinite: number; negative: number; meanRgb: number } {
    let nonFinite = 0;
    let negative = 0;
    let sum = 0;
    for (let index = 0; index < values.length; index++) {
        const value = values[index];
        if (!Number.isFinite(value)) nonFinite++;
        else if (index % 4 !== 3) {
            if (value < 0) negative++;
            sum += value;
        }
    }
    return { nonFinite, negative, meanRgb: sum / ((values.length / 4) * 3) };
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

async function renderScene(context: ThreePageContext, sceneId: string, sheet: HTMLElement): Promise<SceneResult> {
    const { device, renderer, log } = context;
    const radianceCache = new RadianceCacheTechnique();
    const scene = await loadCompositeScene(device, sceneId, { ...SIZE, giFrames: GI_FRAMES, probesPerFrame: PROBES_PER_FRAME, log, techniques: [radianceCache] });
    // DDGI keeps its warmed field; frames from here update no probes (units 0; the floor may still run a few).
    scene.threeGI.budget = { units: 0 };
    const reader = new HdrReader(renderer, SIZE.width, SIZE.height);
    try {
        // DDGI through the composite.
        const ddgiPass = pass(scene.scene, scene.camera);
        ddgiPass.setMRT(mrt({ output, [GI_ALBEDO_OUTPUT]: giAlbedo, [GI_NORMAL_OUTPUT]: packNormalToRGB(normalView) }));
        const ddgiComposite = new DDGICompositeNode(ddgiPass, scene.threeGI, { occlusion: false }) as unknown as Node;

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
        for (let warm = 0; warm < WARM_FRAMES; warm++) await frame(ddgiComposite);
        const ddgi = await frame(ddgiComposite);
        const stagesBefore = stage.stages;
        const scans: SceneResult['scans'][number][] = [];
        let picture: Float32Array = new Float32Array(0);
        let irradiance: Float32Array = new Float32Array(0);
        for (let rcFrame = 1; rcFrame <= SCAN_FRAMES[SCAN_FRAMES.length - 1]; rcFrame++) {
            picture = await frame(rcComposite);
            if (!(SCAN_FRAMES as readonly number[]).includes(rcFrame)) continue;
            const probes = radianceCache.probes;
            if (!probes) throw new Error(`${sceneId}: the radiance cache has no probes after its screen stages`);
            irradiance = halves(await readTexture(device, probes.irradiance, 8, SIZE.width, SIZE.height));
            const pictureBad = countBad(picture);
            const irradianceBad = countBad(irradiance);
            scans.push({ frame: rcFrame, meanIrradiance: irradianceBad.meanRgb, pictureNonFinite: pictureBad.nonFinite, pictureNegative: pictureBad.negative, irradianceNonFinite: irradianceBad.nonFinite, irradianceNegative: irradianceBad.negative });
            assert(pictureBad.nonFinite === 0 && pictureBad.negative === 0, `${sceneId} frame ${rcFrame}: the radiance cache's picture has ${pictureBad.nonFinite} values not finite and ${pictureBad.negative} negative`);
            assert(irradianceBad.nonFinite === 0 && irradianceBad.negative === 0, `${sceneId} frame ${rcFrame}: the irradiance image has ${irradianceBad.nonFinite} values not finite and ${irradianceBad.negative} negative`);
        }
        const screenStages = stage.stages - stagesBefore;
        const pictures = SCAN_FRAMES[SCAN_FRAMES.length - 1];
        assert(screenStages === pictures, `${sceneId}: ${screenStages} screen stages for ${pictures} pictures`);

        const probes = radianceCache.probes;
        if (!probes) throw new Error(`${sceneId}: the radiance cache has no probes after its screen stages`);
        const probeCount = new Uint32Array(await readProbeCount(device, probes.probeSpawn))[0];

        // Surfaces, from the radiance cache's pass as it last drew (no new frame).
        const surfaces = await reader.read(surfaceNode, false);
        // The same pass's own colour (GI off) and giAlbedo, as it last drew.
        const passColor = await reader.read(rcPass.getTextureNode('output') as unknown as Node, false);
        const passAlbedo = await reader.read(rcPass.getTextureNode(GI_ALBEDO_OUTPUT) as unknown as Node, false);
        let compositePixels = 0;
        let compositeOutliers = 0;
        let worstRelative = 0;
        let giSum = 0;
        for (let pixel = 0; pixel < SIZE.width * SIZE.height; pixel++) {
            if (surfaces[pixel * 4] < 0.5) continue;
            const weightSum = passAlbedo[pixel * 4] + passAlbedo[pixel * 4 + 1] + passAlbedo[pixel * 4 + 2];
            if (!(weightSum > 0)) continue;
            compositePixels++;
            let bad = false;
            for (let channel = 0; channel < 3; channel++) {
                const index = pixel * 4 + channel;
                const expected = (passAlbedo[index] * irradiance[index]) / Math.PI;
                const added = picture[index] - passColor[index];
                const error = Math.abs(added - expected);
                giSum += expected;
                if (error > COMPOSITE_RELATIVE_TOLERANCE * expected + COMPOSITE_PICTURE_TOLERANCE * Math.abs(picture[index]) + 1e-6) bad = true;
                if (expected > 1e-4) worstRelative = Math.max(worstRelative, error / expected);
            }
            if (bad) compositeOutliers++;
        }
        const compositeCheck = { pixels: compositePixels, outliers: compositeOutliers, worstRelative, meanGi: giSum / Math.max(3 * compositePixels, 1) };
        assert(compositePixels > (SIZE.width * SIZE.height) / 4, `${sceneId}: only ${compositePixels} pixels for the composite check`);
        assert(
            compositeOutliers <= COMPOSITE_OUTLIER_FRACTION * compositePixels,
            `${sceneId}: on ${compositeOutliers} of ${compositePixels} pixels the picture minus the pass colour isn't giAlbedo x E / pi of the same pixel`,
        );

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

        // The contact sheet's row: one exposure for the scene, from DDGI's picture.
        const exposure = SHEET_KEY / Math.max(ddgiMeanLuminance, 1e-6);
        const row = document.createElement('div');
        row.style.display = 'flex';
        row.style.gap = '4px';
        sheet.append(row);
        addThumbnail(row, `${sceneId}: DDGI`, toDisplay(ddgi, exposure));
        addThumbnail(row, `${sceneId}: radiance cache (1 bounce, frame ${pictures})`, toDisplay(picture, exposure));
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
        return {
            ddgiMeanLuminance,
            radianceCacheMeanLuminance: rcLuminance / Math.max(surfacePixels, 1),
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
            scans,
            composite: compositeCheck,
        };
    } finally {
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
