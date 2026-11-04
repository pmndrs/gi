/**
 * One drawing pipeline, end to end: our device, our compute-written textures,
 * our light node, a pixel readback, and the recompile counts.
 *
 * Used by three pages (standard, meshlet, visibility). The only difference
 * between them is which `renderer.drawingPipeline` is installed.
 */

import { WebGPURenderer } from 'three/webgpu';
import { MeshletDrawingPipeline } from 'three/addons/pipelines/MeshletDrawingPipeline.js';
import { VisibilityBufferDrawingPipeline } from 'three/addons/pipelines/VisibilityBufferDrawingPipeline.js';
import { assert } from '../../client.ts';
import { diffCounts, runSpike, snapshotCounts, type CreationCounts, type SpikeContext } from './harness.ts';
import { checkEnvironmentDiffuse } from './environment-check.ts';
import { FIRST_VALUES, GiTextures, SECOND_VALUES, expectedIrradiance, wrapForThree } from './gi-textures.ts';
import { GiVolumeLight } from './gi-volume-light.ts';
import {
    VOLUME_CENTER,
    VOLUME_SIZE,
    assertRgbClose,
    createTestScene,
    lambertRadiance,
    readCentreAndCorner,
    renderToTarget,
    type TestScene,
} from './test-scene.ts';

export type DrawingPipelineName = 'standard' | 'meshlet' | 'visibility';

/** How many frames to render before reading back. The GPU-driven pipelines cull over a few frames. */
const SETTLE_FRAMES = 4;

interface PipelineProbe {
    /** Times the pipeline claimed an object for its GPU-driven path (0 for the standard pipeline). */
    gpuDrivenObjects: number;
}

/**
 * Installs the requested drawing pipeline. For the GPU-driven ones it also
 * wraps the pipeline's private `_isMeshlet` check to count how many objects it
 * took, so the page can prove the plane really went through that pipeline and
 * did not silently fall back to the standard path.
 */
function installDrawingPipeline(renderer: WebGPURenderer, name: DrawingPipelineName): PipelineProbe {
    const probe: PipelineProbe = { gpuDrivenObjects: 0 };
    if (name === 'standard') return probe;

    const pipeline = name === 'meshlet' ? new MeshletDrawingPipeline() : new VisibilityBufferDrawingPipeline();
    const original = (pipeline as unknown as { _isMeshlet: (...args: unknown[]) => boolean })._isMeshlet.bind(pipeline);
    (pipeline as unknown as { _isMeshlet: (...args: unknown[]) => boolean })._isMeshlet = (...args) => {
        const taken = original(...args);
        if (taken) probe.gpuDrivenObjects++;
        return taken;
    };
    (renderer as unknown as { drawingPipeline: unknown }).drawingPipeline = pipeline;
    return probe;
}

async function renderFrames(renderer: WebGPURenderer, test: TestScene, frames = SETTLE_FRAMES): Promise<void> {
    for (let frame = 0; frame < frames; frame++) renderToTarget(renderer, test);
}

export interface ToggleResult {
    name: string;
    created: CreationCounts;
    centre: number[];
}

export async function runPipelineSpike(name: DrawingPipelineName): Promise<void> {
    await runSpike(async (context: SpikeContext) => {
        const { device, counts, data, log } = context;
        data.pipeline = name;

        // (a) our device, handed to three.
        const renderer = new WebGPURenderer({ device, antialias: false });
        renderer.setPixelRatio(1);
        renderer.setSize(256, 256, false);
        document.body.append(renderer.domElement);
        await renderer.init();
        assert(
            (renderer as unknown as { backend: { device: GPUDevice } }).backend.device === device,
            'three did not adopt the device we passed in',
        );
        data.threeUsesOurDevice = true;

        const probe = installDrawingPipeline(renderer, name);

        // (b) compute-written textures wrapped for three, and the light.
        const textures = new GiTextures(device);
        textures.write(FIRST_VALUES);
        const wrapped = wrapForThree(textures);
        const light = new GiVolumeLight(wrapped, { center: VOLUME_CENTER, size: VOLUME_SIZE });

        const test = createTestScene();
        test.scene.add(light);

        if (name !== 'standard') await renderer.compileAsync(test.scene, test.camera);

        // --- Done-when 1: pixel readback shows the expected added irradiance. ---
        await renderFrames(renderer, test);
        const first = await readCentreAndCorner(renderer, test.target);
        const expectedFirst = lambertRadiance(expectedIrradiance(FIRST_VALUES));
        const checks = [
            assertRgbClose('centre (inside the volume)', first.centre.mean, expectedFirst),
            assertRgbClose('corner (outside the volume)', first.corner.mean, [0, 0, 0]),
        ];
        assert(first.centre.spread < 1e-3, `centre block is not flat (spread ${first.centre.spread})`);
        data.irradianceE = expectedIrradiance(FIRST_VALUES);
        data.expectedRadiance = expectedFirst;
        data.readbackCentre = first.centre.mean;
        data.readbackCorner = first.corner.mean;
        log(`centre ${first.centre.mean.map((v) => v.toFixed(4)).join(' ')} expected ${expectedFirst.map((v) => v.toFixed(4)).join(' ')}`);

        if (name !== 'standard') {
            assert(
                probe.gpuDrivenObjects > 0,
                `the ${name} pipeline never took the plane (fell back to the standard path), so this run proves nothing about it`,
            );
        }

        // A later compute write must reach the next frame with no recompile.
        const beforeRewrite = snapshotCounts(counts);
        textures.write(SECOND_VALUES);
        await renderFrames(renderer, test);
        const second = await readCentreAndCorner(renderer, test.target);
        checks.push(
            assertRgbClose(
                'centre after a second compute write',
                second.centre.mean,
                lambertRadiance(expectedIrradiance(SECOND_VALUES)),
            ),
        );
        data.createdByComputeRewrite = diffCounts(counts, beforeRewrite);
        data.checks = checks;

        // --- Done-when 2: removing the light vs setting intensity 0. ---
        const toggles: ToggleResult[] = [];
        const toggle = async (toggleName: string, action: () => void, expectedCentre: number[]): Promise<void> => {
            const before = snapshotCounts(counts);
            action();
            await renderFrames(renderer, test, 2);
            const created = diffCounts(counts, before);
            const reading = await readCentreAndCorner(renderer, test.target);
            assertRgbClose(`after "${toggleName}"`, reading.centre.mean, expectedCentre);
            toggles.push({ name: toggleName, created, centre: reading.centre.mean });
        };
        const secondRadiance = lambertRadiance(expectedIrradiance(SECOND_VALUES));
        await toggle('intensity 0', () => (light.intensity = 0), [0, 0, 0]);
        await toggle('intensity back to 1', () => (light.intensity = 1), secondRadiance);
        await toggle('remove light from scene', () => test.scene.remove(light), [0, 0, 0]);
        await toggle('add light back', () => test.scene.add(light), secondRadiance);
        data.toggles = toggles;

        // --- scene.environment with env diffuse suppressed, in this drawing pipeline. ---
        test.scene.remove(light);
        data.environment = await checkEnvironmentDiffuse(renderer, wrapped);
        test.scene.add(light);

        // Screenshot: the same scene on the canvas.
        renderer.render(test.scene, test.camera);
        await device.queue.onSubmittedWorkDone();

        data.gpuDrivenObjects = probe.gpuDrivenObjects;
        textures.destroy();
    });
}
