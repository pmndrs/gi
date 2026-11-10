/**
 * T1.7 engine neutrality: the same scene through the plain-WebGPU `hello`
 * example and through three.js with the adapter gives the same image.
 *
 * Per scene (the emissive Cornell and `cornell-box-point`, bookmark 1):
 *
 * - **hello:** the scene described straight from the manifest's data
 *   (`describeScene`), DDGI on the manifest's volume, `FRAMES` frames with
 *   every probe updated each frame, then hello's forward renderer.
 * - **three:** the same manifest built as three objects (`scenes/src/three-scene.ts`,
 *   the viewer's builder), mirrored into a second platform by the adapter's
 *   scene sync (`ThreeGI.update`), the same frames and budget (DDGI's rays
 *   are seeded by the frame number, so both fields see the same rays), then
 *   three renders with the adapter's GI light.
 *
 * What's compared, so that only GI and emission are in the picture:
 * - **Materials:** Lambert (`MeshLambertMaterial`), not the builder's
 *   `MeshStandardMaterial`: three's standard material takes the specular
 *   lobe's share out of indirect diffuse (NOTES: the references are pure
 *   Lambert), which hello doesn't model. Lambert is exactly
 *   `emission + albedo / π × irradiance`, as hello shades.
 * - **Emissive Cornell:** hello's `full` view against three's full image. The
 *   scene has no lights, so both are emission + albedo / π × GI.
 * - **Point Cornell:** hello's `gi` view (albedo / π × GI) against three with
 *   the point light hidden for the final render only, after the GI frames
 *   (its direct light and shadow map are each renderer's own, not GI's).
 *
 * Pass: the image means within `MEAN_TOLERANCE`, and at most
 * `MAX_DIFFERING_SHARE` of pixels off by more than `PIXEL_TOLERANCE`.
 *
 * Also checked: toggling the GI light's intensity to 0 and back makes no new
 * render pipelines, and at 0 the GI-only image goes black; and three's
 * Meshlet and Visibility-Buffer drawing pipelines give the standard
 * pipeline's image (means within `PIPELINE_MEAN_TOLERANCE`; all three with
 * front-sided materials, the only kind those pipelines draw).
 */

import { Box3, type Mesh, MeshLambertMaterial, PerspectiveCamera, RenderTarget, Scene, Vector3, type Light, type Material, type MeshStandardMaterial, type Object3D } from 'three/webgpu';
import { FloatType, FrontSide, type Side } from 'three';
import { MeshletDrawingPipeline } from 'three/addons/pipelines/MeshletDrawingPipeline.js';
import { VisibilityBufferDrawingPipeline } from 'three/addons/pipelines/VisibilityBufferDrawingPipeline.js';
import { assert } from '../../client.ts';
import { GIPlatformImpl } from '../../../../packages/ddgi-core/src/platform/gi-platform.ts';
import { DDGITechnique } from '../../../../packages/ddgi-core/src/techniques/ddgi/ddgi-technique.ts';
import { ThreeGI } from '../../../../packages/ddgi-three/src/three-gi.ts';
import { ForwardRenderer, type RenderView } from '../../../../examples/webgpu/hello/src/renderer.ts';
import { CAMERA_FAR, CAMERA_NEAR } from '../../../../examples/webgpu/hello/src/camera.ts';
import { loadManifest } from '../../../../examples/webgpu/hello/src/scene.ts';
import { buildProceduralParts, mergeScenes } from '../../../../scenes/src/build-scene.ts';
import { manifestLightToSceneLight, type CameraBookmark, type SceneManifest } from '../../../../scenes/src/manifest.ts';
import { lightToThree, proceduralSceneToThree } from '../../../../scenes/src/three-scene.ts';
import { imageMean, runFrames } from '../ddgi-shared/harness.ts';
import { createHelloScene, readRgba32FloatAsRgb } from '../ddgi-shading-shared/hello-scene.ts';
import { runSteps, runThreePage, type ThreePageContext } from '../three-shared/harness.ts';

const SCENES = ['cornell-box', 'cornell-box-point'] as const;
const CAMERA = 1;
const SIZE = 256;
/** GI frames per side, every probe each frame. Both sides run the same frames, so convergence doesn't matter for the comparison. */
const FRAMES = 200;

/** The two images' means may differ by this share. */
const MEAN_TOLERANCE = 0.005;
/** A pixel differs when it's off by more than this share of its value (or of the image mean, near black). */
const PIXEL_TOLERANCE = 0.02;
/** At most this share of pixels may differ. */
const MAX_DIFFERING_SHARE = 0.005;

/** The GPU-driven pipelines against the standard one: the same shading, rasterized another way. */
const PIPELINE_MEAN_TOLERANCE = 0.005;

interface PipelineProbe {
    readonly previous: unknown;
    /** Objects the pipeline took for its GPU-driven path; 0 would mean it fell back to the standard path. */
    gpuDrivenObjects: number;
}

/** Installs a GPU-driven drawing pipeline, counting the objects it takes (as the T0.8 spike does). */
function installDrawingPipeline(renderer: ThreePageContext['renderer'], name: 'meshlet' | 'visibility'): PipelineProbe {
    const host = renderer as unknown as { drawingPipeline: unknown };
    const probe: PipelineProbe = { previous: host.drawingPipeline, gpuDrivenObjects: 0 };
    const pipeline = (name === 'meshlet' ? new MeshletDrawingPipeline() : new VisibilityBufferDrawingPipeline()) as unknown as { _isMeshlet: (...args: unknown[]) => boolean };
    const original = pipeline._isMeshlet.bind(pipeline);
    pipeline._isMeshlet = (...args) => {
        const taken = original(...args);
        if (taken) probe.gpuDrivenObjects++;
        return taken;
    };
    host.drawingPipeline = pipeline;
    return probe;
}

/** Sets every material's side (they recompile), returning what to restore. */
function setSide(root: Object3D, sideToUse: Side): Map<Material, Side> {
    const previous = new Map<Material, Side>();
    root.traverse((object) => {
        const material = (object as Mesh).material as Material | undefined;
        if (!material || previous.has(material)) return;
        previous.set(material, material.side);
        material.side = sideToUse;
        material.needsUpdate = true;
    });
    return previous;
}

function restoreSides(previous: Map<Material, Side>): void {
    for (const [material, side] of previous) {
        material.side = side;
        material.needsUpdate = true;
    }
}

function restoreDrawingPipeline(renderer: ThreePageContext['renderer'], probe: PipelineProbe): void {
    (renderer as unknown as { drawingPipeline: unknown }).drawingPipeline = probe.previous;
}

interface Comparison {
    meanHello: number;
    meanThree: number;
    meanDifference: number;
    differingShare: number;
    worstDifference: number;
}

function compare(three: Float32Array, hello: Float32Array): Comparison {
    const meanHello = imageMean(hello);
    const meanThree = imageMean(three);
    const floor = PIXEL_TOLERANCE * meanHello * 3;
    let differing = 0;
    let worst = 0;
    const pixels = hello.length / 3;
    for (let pixel = 0; pixel < pixels; pixel++) {
        const a = three[pixel * 3] + three[pixel * 3 + 1] + three[pixel * 3 + 2];
        const b = hello[pixel * 3] + hello[pixel * 3 + 1] + hello[pixel * 3 + 2];
        const difference = Math.abs(a - b);
        worst = Math.max(worst, difference / Math.max(b, floor));
        if (difference > Math.max(PIXEL_TOLERANCE * b, floor)) differing++;
    }
    return { meanHello, meanThree, meanDifference: meanThree / meanHello - 1, differingShare: differing / pixels, worstDifference: worst };
}

/** The builder's standard materials as Lambert ones with the same colour, emission and side (see the file comment). */
function useLambert(root: Object3D): void {
    const converted = new Map<Material, Material>();
    root.traverse((object) => {
        const mesh = object as Mesh;
        if (!mesh.isMesh) return;
        const standard = mesh.material as MeshStandardMaterial;
        let lambert = converted.get(standard);
        if (!lambert) {
            lambert = new MeshLambertMaterial({ color: standard.color, emissive: standard.emissive, emissiveIntensity: standard.emissiveIntensity, side: standard.side, name: standard.name });
            converted.set(standard, lambert);
        }
        mesh.material = lambert;
    });
}

interface ThreeSide {
    scene: Scene;
    camera: PerspectiveCamera;
    threeGI: ThreeGI;
    platform: GIPlatformImpl;
    lights: Light[];
}

async function buildThreeSide(context: ThreePageContext, manifest: SceneManifest, bookmark: CameraBookmark): Promise<ThreeSide> {
    const { device } = context;
    const scene = new Scene();
    const content = proceduralSceneToThree(mergeScenes(buildProceduralParts(manifest)), manifest.id);
    useLambert(content);
    scene.add(content);
    content.updateMatrixWorld(true);
    const bounds = new Box3().setFromObject(content);
    const center = bounds.getCenter(new Vector3());
    const fitting = { center: [center.x, center.y, center.z] as [number, number, number], radius: bounds.getSize(new Vector3()).length() / 2 };
    const lights = manifest.lights.map((light) => lightToThree(manifestLightToSceneLight(light), fitting) as Light);
    for (const light of lights) scene.add(light);

    const camera = new PerspectiveCamera(bookmark.fovYDeg, 1, CAMERA_NEAR, CAMERA_FAR);
    camera.position.set(...bookmark.position);
    camera.lookAt(...bookmark.target);
    camera.updateMatrixWorld();

    const platform = new GIPlatformImpl(device, { label: `three ${manifest.id}` });
    const ddgi = platform.addTechnique(new DDGITechnique({ volume: { center: manifest.volume.center, size: manifest.volume.size, spacing: manifest.volume.probeSpacing }, warmUpPasses: 0 }));
    platform.setOutput(ddgi);
    await platform.prepare();
    const threeGI = new ThreeGI(platform, scene, { budget: { units: 0 } });
    // As createHelloScene: a frame with no probe updates places the scene, then the mesh trees arrive.
    threeGI.update(camera, 1 / 60);
    await platform.rays.whenBuilt();
    threeGI.update(camera, 1 / 60);
    const probeCount = ddgi.field!.grid.probeCount;
    threeGI.budget = { units: probeCount };
    await runSteps(device, FRAMES, () => threeGI.update(camera, 1 / 60));
    return { scene, camera, threeGI, platform, lights };
}

async function renderThree(context: ThreePageContext, side: ThreeSide): Promise<Float32Array> {
    const { renderer } = context;
    const target = new RenderTarget(SIZE, SIZE, { type: FloatType, depthBuffer: true });
    // Pipelines compile in the background and objects are skipped until theirs is ready (NOTES): compile first.
    await renderer.compileAsync(side.scene, side.camera);
    renderer.setRenderTarget(target);
    renderer.render(side.scene, side.camera);
    renderer.setRenderTarget(null);
    const rgba = (await renderer.readRenderTargetPixelsAsync(target, 0, 0, SIZE, SIZE)) as Float32Array;
    target.dispose();
    const rgb = new Float32Array(SIZE * SIZE * 3);
    for (let pixel = 0; pixel < SIZE * SIZE; pixel++) rgb.set(rgba.subarray(pixel * 4, pixel * 4 + 3), pixel * 3);
    return rgb;
}

async function renderHello(device: GPUDevice, sceneId: string, bookmark: CameraBookmark, view: RenderView): Promise<Float32Array> {
    const { platform, ddgi, scene } = await createHelloScene(device, sceneId);
    await runFrames(device, platform, FRAMES, ddgi.field!.grid.probeCount);
    const renderer = new ForwardRenderer(device, scene, { shading: platform.shading, hdrFormat: 'rgba32float' });
    renderer.resize(SIZE, SIZE);
    const encoder = device.createCommandEncoder({ label: 'hello render' });
    renderer.render(encoder, renderer.camera(bookmark), view);
    device.queue.submit([encoder.finish()]);
    const image = await readRgba32FloatAsRgb(device, renderer.hdrTexture!);
    renderer.destroy();
    platform.dispose();
    return image;
}

await runThreePage(async (context) => {
    const { device, data, log, pipelines } = context;
    const results: Record<string, unknown> = {};
    data.scenes = results;
    data.materials = 'Lambert (MeshLambertMaterial) on the three side';
    for (const sceneId of SCENES) {
        const manifest = loadManifest(sceneId);
        const bookmark = manifest.cameras[CAMERA - 1];
        const giOnly = manifest.lights.length > 0;

        const helloImage = await renderHello(device, sceneId, bookmark, giOnly ? { mode: 'gi' } : { mode: 'full' });
        const side = await buildThreeSide(context, manifest, bookmark);
        // GI-only on the point Cornell: hide the light for the picture; GI was computed with it on.
        for (const light of side.lights) light.visible = !giOnly;
        const threeImage = await renderThree(context, side);
        const comparison = compare(threeImage, helloImage);
        log(
            `${sceneId} (${giOnly ? 'GI only' : 'emission + GI'}): means hello ${comparison.meanHello.toFixed(5)}, three ${comparison.meanThree.toFixed(5)} ` +
                `(${(comparison.meanDifference * 100).toFixed(3)}%), ${(comparison.differingShare * 100).toFixed(3)}% of pixels off by more than ${PIXEL_TOLERANCE * 100}%`,
        );

        // Toggling with intensity: no new pipelines, and the GI goes away.
        const before = { ...pipelines };
        side.threeGI.light.intensity = 0;
        const off = await renderThree(context, side);
        side.threeGI.light.intensity = 1;
        const backOn = await renderThree(context, side);
        const toggle = { renderPipelinesMade: pipelines.render - before.render, meanOff: imageMean(off), meanBackOn: imageMean(backOn) };

        // The same picture under three's GPU-driven drawing pipelines (r187 #35191), which clone
        // materials and overwrite contextNode; the GI light must work there too (DECISIONS).
        // They draw only FrontSide materials, so all three renders here are front-sided.
        const pipelineChecks: Record<string, unknown> = {};
        pipelineChecks.frontSidedStandardMean = 0;
        const sides = setSide(side.scene, FrontSide);
        const standardFrontSided = imageMean(await renderThree(context, side));
        pipelineChecks.frontSidedStandardMean = standardFrontSided;
        assert(standardFrontSided > 0, `${sceneId}: the front-sided image is black`);
        for (const name of ['meshlet', 'visibility'] as const) {
            const probe = installDrawingPipeline(context.renderer, name);
            const image = await renderThree(context, side);
            restoreDrawingPipeline(context.renderer, probe);
            const againstStandard = imageMean(image) / standardFrontSided - 1;
            pipelineChecks[name] = { gpuDrivenObjects: probe.gpuDrivenObjects, meanAgainstStandard: againstStandard };
            assert(probe.gpuDrivenObjects > 0, `${sceneId}: the ${name} pipeline never took an object, so this proves nothing about it`);
            assert(Math.abs(againstStandard) <= PIPELINE_MEAN_TOLERANCE, `${sceneId}: under the ${name} pipeline the image mean differs from the standard pipeline's by ${(againstStandard * 100).toFixed(2)}%`);
        }
        restoreSides(sides);

        results[sceneId] = { view: giOnly ? 'GI only (hello gi view; three with the point light hidden)' : 'emission + GI (hello full view)', ...comparison, toggle, pipelines: pipelineChecks, adapterSyncCounts: side.threeGI.sync.counts };
        side.threeGI.dispose();
        side.platform.dispose();

        assert(comparison.meanHello > 0, `${sceneId}: hello's image is black`);
        assert(Math.abs(comparison.meanDifference) <= MEAN_TOLERANCE, `${sceneId}: the means differ by ${(comparison.meanDifference * 100).toFixed(3)}%, more than ${MEAN_TOLERANCE * 100}%`);
        assert(comparison.differingShare <= MAX_DIFFERING_SHARE, `${sceneId}: ${(comparison.differingShare * 100).toFixed(3)}% of pixels differ, more than ${MAX_DIFFERING_SHARE * 100}%`);
        assert(toggle.renderPipelinesMade === 0, `${sceneId}: toggling the GI light's intensity made ${toggle.renderPipelinesMade} render pipelines`);
        assert(Math.abs(toggle.meanBackOn / comparison.meanThree - 1) < 1e-4, `${sceneId}: the image changed after turning GI off and on`);
        if (giOnly) assert(toggle.meanOff < 1e-6, `${sceneId}: with the GI light at intensity 0 the GI-only image still reads ${toggle.meanOff}`);
        else assert(toggle.meanOff < comparison.meanThree, `${sceneId}: turning GI off didn't darken the image`);
    }
}, SIZE);
