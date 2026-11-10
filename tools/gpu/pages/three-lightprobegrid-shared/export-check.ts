/**
 * Shared by the `three-lightprobegrid-export` (Cornell) and
 * `three-lightprobegrid-export-sponza` pages (T5.3,
 * docs/design/m5-coverage.md §3): the scene lit by the exported
 * `LightProbeGrid` against our live GI.
 *
 * 1. The scene loads the viewer's way, DDGI runs on the manifest's volume
 *    through `ThreeGI` for `giFrames` frames (every probe a few frames), and
 *    the field is frozen (`paused`), so both images shade the same field.
 * 2. `exportIrradianceSH` (one GPU projection, one readback), then
 *    `toLightProbeGrid`, then JSON text and back (`lightProbeGridToJSON`,
 *    `lightProbeGridFromJSON`): the round trip must give every coefficient
 *    back bit for bit, and the grid read back from the text is the one drawn.
 * 3. **The comparison.** For each camera bookmark, the GI-only view (direct
 *    lights hidden, every surface white Lambert, so a pixel is E / π; three's
 *    Lambert takes no diffuse light from `scene.environment`, so only the
 *    probe light counts): our GI light against the grid. Both lights stay in
 *    the scene and only their intensities change, so nothing recompiles
 *    between the two: GI at 1 claims three's probe slot at priority 1 above
 *    the grid's 0; GI at 0 gives the slot up and the grid lights alone.
 *    Bookmark 1 (the test view) must have its mean within
 *    {@link MEAN_TOLERANCE}; the others are reported. The difference is
 *    expected (L2, no visibility, no relocation) and is recorded, not tuned.
 * 4. **Light groups** (`lightGroups: true`): DDGI runs with one light group,
 *    so the export reads the composed atlas. Halving the group's weight and
 *    composing one frame must halve the exported coefficients (the export
 *    takes the composed field with the weights in force).
 * 5. **The screenshot:** at bookmark 1, the GI-only view (top) and the full
 *    look with direct light and the scene's materials (bottom), live GI on the
 *    left and the exported grid on the right, the same exposure on each row.
 *
 * Not a page itself (no index.html).
 */

import { ACESFilmicToneMapping, FloatType, MeshLambertMaterial, NoToneMapping, PerspectiveCamera, RenderTarget, type Mesh, type Object3D, type Scene } from 'three/webgpu';
import { assert } from '../../client.ts';
import { GIPlatformImpl } from '../../../../packages/ddgi-core/src/platform/gi-platform.ts';
import { DDGITechnique } from '../../../../packages/ddgi-core/src/techniques/ddgi/ddgi-technique.ts';
import { exportIrradianceSH, L2_SH_FLOATS_PER_PROBE, type IrradianceSHGrid } from '../../../../packages/ddgi-core/src/techniques/ddgi/irradiance-sh.ts';
import { PROBE_STATE } from '../../../../packages/ddgi-core/src/techniques/ddgi/layout.ts';
import { ThreeGI } from '../../../../packages/ddgi-three/src/three-gi.ts';
import { lightProbeGridFromJSON, lightProbeGridToJSON, toLightProbeGrid } from '../../../../packages/ddgi-three/src/light-probe-grid.ts';
import { loadAllManifests } from '../../../../apps/viewer/src/scene/manifests.ts';
import { loadScene } from '../../../../apps/viewer/src/scene/load-scene.ts';
import { platformSkyEnvironment } from '../../../../scenes/src/sky/sky-image.ts';
import type { CameraBookmark } from '../../../../scenes/src/manifest.ts';
import { runSteps, type ThreePageContext } from '../three-shared/harness.ts';

/** The done-when: the test view's mean through the exported grid within 15% of live GI's (docs/design/m5-coverage.md §3). */
export const MEAN_TOLERANCE = 0.15;

/** Each measured image and each screenshot panel; the page's canvas is two by two of them. */
// The runner's window is 1280 × 720, so two by two of these fit with the page's summary below.
export const VIEW_WIDTH = 480;
export const VIEW_HEIGHT = 270;
export const CANVAS_SIZE = { width: VIEW_WIDTH * 2, height: VIEW_HEIGHT * 2 } as const;

/** The halved weight of the light-group check, and how close the exported coefficients must follow it. */
const GROUP_WEIGHT = 0.5;
const GROUP_RATIO_TOLERANCE = 0.01;

/** The screenshot's rows are exposed so live GI's mean lands here after tone mapping's input. */
const SCREENSHOT_MEAN_TARGET = 0.25;

export interface ExportCheckOptions {
    readonly sceneId: string;
    /** GI frames before the field is frozen. */
    readonly giFrames: number;
    /** Probe updates per GI frame (a fixed `units` budget). */
    readonly probesPerFrame: number;
    /** Run DDGI with one light group and check the export follows its weight (file comment, 4). */
    readonly lightGroups: boolean;
}

function luminance(red: number, green: number, blue: number): number {
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue;
}

/** Mean luminance of an RGBA float image, and the summed absolute luminance difference from another. */
function imageStatistics(pixels: Float32Array, reference?: Float32Array): { mean: number; meanAbsoluteDifference: number } {
    let sum = 0;
    let absoluteDifference = 0;
    const count = pixels.length / 4;
    for (let pixel = 0; pixel < count; pixel++) {
        const value = luminance(pixels[pixel * 4], pixels[pixel * 4 + 1], pixels[pixel * 4 + 2]);
        sum += value;
        if (reference) absoluteDifference += Math.abs(value - luminance(reference[pixel * 4], reference[pixel * 4 + 1], reference[pixel * 4 + 2]));
    }
    return { mean: sum / count, meanAbsoluteDifference: absoluteDifference / count };
}

function bookmarkCamera(bookmark: CameraBookmark, far: number): PerspectiveCamera {
    const camera = new PerspectiveCamera(bookmark.fovYDeg, VIEW_WIDTH / VIEW_HEIGHT, 0.05, far);
    camera.position.set(...bookmark.position);
    camera.lookAt(...bookmark.target);
    camera.updateMatrixWorld(true);
    return camera;
}

/** Probes per state in an export, by label. */
function stateCounts(data: IrradianceSHGrid): Record<string, number> {
    const names: Record<number, string> = {
        [PROBE_STATE.active]: 'active',
        [PROBE_STATE.offInsideGeometry]: 'offInsideGeometry',
        [PROBE_STATE.offNoSurfaceNearby]: 'offNoSurfaceNearby',
        [PROBE_STATE.notUpdatedYet]: 'notUpdatedYet',
    };
    const counts: Record<string, number> = {};
    for (const state of data.probeStates) counts[names[state] ?? `state${state}`] = (counts[names[state] ?? `state${state}`] ?? 0) + 1;
    return counts;
}

/** The band-0 luminance summed over every probe: how bright the exported field is overall. */
function totalBandZero(data: IrradianceSHGrid): number {
    let sum = 0;
    for (let probe = 0; probe < data.coefficients.length / L2_SH_FLOATS_PER_PROBE; probe++) {
        const base = probe * L2_SH_FLOATS_PER_PROBE;
        sum += luminance(data.coefficients[base], data.coefficients[base + 1], data.coefficients[base + 2]);
    }
    return sum;
}

export async function runExportCheck(context: ThreePageContext, options: ExportCheckOptions): Promise<void> {
    const { device, renderer, data, log } = context;
    const manifest = loadAllManifests().find((candidate) => candidate.id === options.sceneId);
    assert(manifest, `no ${options.sceneId} manifest`);
    const loaded = await loadScene(manifest, () => {});
    const scene: Scene = loaded.scene;
    // The sky dome and other helpers are for viewing: out of GI, and hidden in the GI-only view.
    const content = new Set<Object3D>();
    loaded.content.traverse((object) => content.add(object));
    const notContent: Object3D[] = [];
    scene.traverse((object) => {
        if ((object as Mesh).isMesh && !content.has(object)) {
            object.userData.giContributes = false;
            notContent.push(object);
        }
    });

    // 1. DDGI, converged, then frozen.
    const platform = new GIPlatformImpl(device, { label: `three lightprobegrid export ${options.sceneId}` });
    const volume = manifest.volume;
    const technique = platform.addTechnique(
        new DDGITechnique({
            volume: { center: volume.center, size: volume.size, spacing: volume.probeSpacing },
            warmUpPasses: 0,
            lightGroups: options.lightGroups ? { count: 1, weights: [[1, 1, 1]], environmentWeight: [1, 1, 1] } : null,
        }),
    );
    platform.setOutput(technique);
    await platform.prepare();
    const environment = manifest.environment.type === 'procedural' ? platformSkyEnvironment(manifest) : null;
    const threeGI = new ThreeGI(platform, scene, { budget: { units: 0 }, environment });
    const firstCamera = bookmarkCamera(manifest.cameras[0], loaded.diagonal * 2);
    threeGI.update(firstCamera);
    await platform.rays.whenBuilt();
    threeGI.update(firstCamera);
    threeGI.budget = { units: options.probesPerFrame };
    await runSteps(device, options.giFrames, () => threeGI.update(firstCamera, 1 / 60));
    threeGI.budget = { paused: true };
    const field = technique.field;
    assert(field, 'no field after prepare');
    data.scene = options.sceneId;
    data.probes = field.grid.probeCount;
    data.giFrames = options.giFrames;
    data.probesPerFrame = options.probesPerFrame;
    data.irradianceLayers = field.layers.layers.length;

    // 2. The export, and its JSON round trip.
    const exportStart = performance.now();
    const exported = await exportIrradianceSH(technique);
    data.exportMs = performance.now() - exportStart;
    data.filledProbes = exported.filledProbes;
    data.probeStates = stateCounts(exported);
    data.composedLayers = exported.composedLayers;
    const built = toLightProbeGrid(exported);
    const jsonText = JSON.stringify(lightProbeGridToJSON(built));
    const grid = lightProbeGridFromJSON(jsonText);
    const builtBits = new Uint32Array(built.coefficients.buffer);
    const readBits = new Uint32Array(grid.coefficients.buffer);
    let differingFloats = 0;
    for (let float = 0; float < builtBits.length; float++) if (builtBits[float] !== readBits[float]) differingFloats++;
    data.jsonBytes = jsonText.length;
    data.jsonDifferingFloats = differingFloats;
    assert(readBits.length === builtBits.length && differingFloats === 0, `the JSON round trip changed ${differingFloats} of ${builtBits.length} coefficients`);
    assert(grid.position.equals(built.position) && grid.resolution.equals(built.resolution), 'the JSON round trip moved or resized the grid');
    log(
        `${options.sceneId}: ${field.grid.probeCount} probes, export ${(data.exportMs as number).toFixed(0)} ms, ${exported.filledProbes} inactive probes filled, ` +
            `${exported.composedLayers} layer(s) composed; JSON ${(jsonText.length / 1024).toFixed(0)} KiB, round trip exact`,
    );
    built.dispose();

    // Both probe lights in the scene; the intensities pick which one lights (file comment, 3).
    const giLight = threeGI.light;
    grid.intensity = 0;
    scene.add(grid);
    const useLive = (): void => {
        giLight.intensity = 1;
        grid.intensity = 0;
    };
    const useExported = (): void => {
        giLight.intensity = 0;
        grid.intensity = 1;
    };

    // Diffuse from scene.environment would count in one image and not the other (the GI light fades it where it covers); keep it out of both.
    scene.environment = null;
    // No background: the GI-only view must hold nothing but GI. The full look still shows Sponza's sky through its sky dome (a mesh).
    scene.background = null;
    const whiteLambert = new MeshLambertMaterial({ color: 0xffffff });
    const giOnly = (on: boolean): void => {
        for (const light of loaded.directLights) light.visible = !on;
        for (const object of notContent) object.visible = !on;
        scene.overrideMaterial = on ? whiteLambert : null;
    };

    // 3. The comparison, every bookmark.
    const target = new RenderTarget(VIEW_WIDTH, VIEW_HEIGHT, { type: FloatType, depthBuffer: true });
    const renderImage = async (camera: PerspectiveCamera): Promise<Float32Array> => {
        renderer.setRenderTarget(target);
        renderer.render(scene, camera);
        renderer.setRenderTarget(null);
        return (await renderer.readRenderTargetPixelsAsync(target, 0, 0, VIEW_WIDTH, VIEW_HEIGHT)) as Float32Array;
    };
    giOnly(true);
    const views: Record<string, unknown>[] = [];
    for (const [index, bookmark] of manifest.cameras.entries()) {
        const camera = bookmarkCamera(bookmark, loaded.diagonal * 2);
        useLive();
        await renderer.compileAsync(scene, camera);
        const live = await renderImage(camera);
        useExported();
        const fromGrid = await renderImage(camera);
        const liveStatistics = imageStatistics(live);
        const gridStatistics = imageStatistics(fromGrid, live);
        const meanDifference = gridStatistics.mean / liveStatistics.mean - 1;
        views.push({
            bookmark: index + 1,
            name: bookmark.name,
            liveMean: liveStatistics.mean,
            exportedMean: gridStatistics.mean,
            meanDifference,
            // Per-pixel: the mean absolute luminance difference over live GI's mean.
            pixelDifference: gridStatistics.meanAbsoluteDifference / liveStatistics.mean,
        });
        log(
            `${options.sceneId} bookmark ${index + 1} (${bookmark.name}), GI only: live ${liveStatistics.mean.toFixed(4)}, exported grid ${gridStatistics.mean.toFixed(4)} ` +
                `(${(meanDifference * 100).toFixed(1)}%; per pixel ${((gridStatistics.meanAbsoluteDifference / liveStatistics.mean) * 100).toFixed(1)}%)`,
        );
    }
    data.views = views;
    const testView = views[0] as { meanDifference: number };
    data.testViewMeanDifference = testView.meanDifference;

    // 5. The screenshot at bookmark 1: GI only on top, the full look below; live left, the grid right.
    const camera = bookmarkCamera(manifest.cameras[0], loaded.diagonal * 2);
    useLive();
    const giOnlyLiveMean = imageStatistics(await renderImage(camera)).mean;
    giOnly(false);
    await renderer.compileAsync(scene, camera);
    const fullLiveMean = imageStatistics(await renderImage(camera)).mean;
    renderer.toneMapping = ACESFilmicToneMapping;
    renderer.setRenderTarget(null);
    // Compile both rows for the canvas first, then draw all four panels in one task: an await between panels lets
    // the browser present the canvas, and the next panel lands on a fresh canvas texture (the first row went missing
    // on Sponza, whose compile takes real time).
    useLive();
    for (const gi of [true, false]) {
        giOnly(gi);
        await renderer.compileAsync(scene, camera);
    }
    // Opaque black where nothing is drawn, not the page behind the canvas. One clear for the whole canvas, then each
    // panel clears only its depth: a render pass's own clear covers the whole canvas whatever the scissor.
    renderer.setClearColor(0x000000, 1);
    renderer.clear();
    const autoClear = renderer.autoClear;
    renderer.autoClear = false;
    renderer.setScissorTest(true);
    const panel = (column: number, row: number): void => {
        renderer.setViewport(column * VIEW_WIDTH, row * VIEW_HEIGHT, VIEW_WIDTH, VIEW_HEIGHT);
        renderer.setScissor(column * VIEW_WIDTH, row * VIEW_HEIGHT, VIEW_WIDTH, VIEW_HEIGHT);
        renderer.clearDepth();
        renderer.render(scene, camera);
    };
    for (const [row, gi] of [[0, true], [1, false]] as const) {
        giOnly(gi);
        renderer.toneMappingExposure = SCREENSHOT_MEAN_TARGET / Math.max(gi ? giOnlyLiveMean : fullLiveMean, 1e-6);
        useLive();
        panel(0, row);
        useExported();
        panel(1, row);
    }
    renderer.setScissorTest(false);
    renderer.autoClear = autoClear;
    renderer.toneMapping = NoToneMapping;
    addLabels(['live GI, GI only', 'exported LightProbeGrid, GI only', 'live GI, full', 'exported LightProbeGrid, full']);

    // 4. Light groups: halve the group's weight, compose one frame (no probe updates), export again.
    if (options.lightGroups) {
        const before = totalBandZero(exported);
        technique.setGroupWeight(0, [GROUP_WEIGHT, GROUP_WEIGHT, GROUP_WEIGHT]);
        threeGI.budget = { units: 0 };
        threeGI.update(camera, 1 / 60);
        await device.queue.onSubmittedWorkDone();
        threeGI.budget = { paused: true };
        const halved = await exportIrradianceSH(technique);
        const ratio = totalBandZero(halved) / before;
        data.groupWeightRatio = ratio;
        log(`light group weight ${GROUP_WEIGHT}: exported band 0 at ${ratio.toFixed(4)} of before`);
        assert(Math.abs(ratio - GROUP_WEIGHT) <= GROUP_RATIO_TOLERANCE, `the export did not follow the light group's weight: band 0 at ${ratio.toFixed(4)} of before, expected ${GROUP_WEIGHT}`);
    }

    assert(
        Math.abs(testView.meanDifference) <= MEAN_TOLERANCE,
        `the exported grid's test-view mean is ${(testView.meanDifference * 100).toFixed(1)}% from live GI's (allowed ±${MEAN_TOLERANCE * 100}%)`,
    );
    target.dispose();
}

/** Captions over the four panels of the screenshot. */
function addLabels(captions: readonly string[]): void {
    const canvas = document.querySelector('canvas');
    if (!canvas) return;
    const box = canvas.getBoundingClientRect();
    captions.forEach((caption, index) => {
        const label = document.createElement('div');
        label.textContent = caption;
        const column = index % 2;
        const row = Math.floor(index / 2);
        Object.assign(label.style, {
            position: 'absolute',
            left: `${box.left + window.scrollX + (column * box.width) / 2 + 6}px`,
            top: `${box.top + window.scrollY + (row * box.height) / 2 + 4}px`,
            font: '13px sans-serif',
            color: '#fff',
            background: 'rgba(0, 0, 0, 0.55)',
            padding: '1px 5px',
        });
        document.body.append(label);
    });
}
