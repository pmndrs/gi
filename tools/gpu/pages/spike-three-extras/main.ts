/**
 * T0.8, the sub-questions that are not about the drawing pipeline.
 *
 *  1. Coexistence with a real `LightProbeGrid`: where both cover a fragment,
 *     one wins by priority and the two are never summed.
 *  2. `scene.environment` set, and the diffuse part switched off where our GI
 *     covers (r187 has no switch; our light node patches the lighting model).
 *  3. A `CubeCamera` capture of the scene, which is what a reflection-probe
 *     capture does (r187 has no reflection probes), contains our GI.
 *
 * All use the standard drawing pipeline; the other pipelines are covered by
 * the `spike-three-standard/meshlet/visibility` pages.
 */

import { CubeCamera, CubeRenderTarget, Color, FloatType, Scene, WebGPURenderer, type RenderTarget } from 'three/webgpu';
import { LightProbeGrid } from 'three/addons/lighting/LightProbeGrid.js';
import { assert } from '../../client.ts';
import { checkEnvironmentDiffuse } from '../spike-three-lib/environment-check.ts';
import { runSpike, type SpikeContext } from '../spike-three-lib/harness.ts';
import { FIRST_VALUES, GiTextures, expectedIrradiance, wrapForThree } from '../spike-three-lib/gi-textures.ts';
import { GiVolumeLight } from '../spike-three-lib/gi-volume-light.ts';
import {
    VOLUME_CENTER,
    VOLUME_SIZE,
    assertRgbClose,
    createTestScene,
    formatRgb,
    lambertRadiance,
    readBlock,
    readCentreAndCorner,
    renderToTarget,
    withinTolerance,
    type TestScene,
} from '../spike-three-lib/test-scene.ts';

const FRAMES = 3;

async function readCentre(renderer: WebGPURenderer, test: TestScene): Promise<number[]> {
    for (let frame = 0; frame < FRAMES; frame++) renderToTarget(renderer, test);
    return (await readCentreAndCorner(renderer, test.target)).centre.mean;
}

function sum(a: readonly number[], b: readonly number[]): number[] {
    return a.map((value, channel) => value + b[channel]);
}

function closeTo(actual: readonly number[], expected: readonly number[]): boolean {
    return actual.every((value, channel) => withinTolerance(value, expected[channel]));
}

await runSpike(async (context: SpikeContext) => {
    const { device, data } = context;

    const renderer = new WebGPURenderer({ device, antialias: false });
    renderer.setPixelRatio(1);
    renderer.setSize(256, 256, false);
    document.body.append(renderer.domElement);
    await renderer.init();

    const textures = new GiTextures(device);
    textures.write(FIRST_VALUES);
    const wrapped = wrapForThree(textures);
    const expectedOurs = lambertRadiance(expectedIrradiance(FIRST_VALUES));

    // ---------------------------------------------------------------------
    // 1. Coexistence with a real LightProbeGrid.
    // ---------------------------------------------------------------------
    // Bake a grid that sees only a flat grey background, so its irradiance is a
    // known, flat value (E = pi * L) that differs from ours in every channel.
    const backgroundGrey = 0.5;
    const bakeScene = new Scene();
    bakeScene.background = new Color(backgroundGrey, backgroundGrey, backgroundGrey);
    const grid = new LightProbeGrid(VOLUME_SIZE.x, VOLUME_SIZE.y, VOLUME_SIZE.z, 2, 2, 2);
    grid.position.copy(VOLUME_CENTER);
    grid.bake(renderer, bakeScene, { cubemapSize: 8 });

    const coexistence = createTestScene();
    const ours = new GiVolumeLight(wrapped, { center: VOLUME_CENTER, size: VOLUME_SIZE });
    coexistence.scene.add(ours, grid);

    // Reference readings, each light alone.
    ours.visible = false;
    const gridOnly = await readCentre(renderer, coexistence);
    ours.visible = true;
    grid.visible = false;
    const oursOnly = await readCentre(renderer, coexistence);
    grid.visible = true;
    assertRgbClose('our light alone', oursOnly, expectedOurs);
    assert(
        gridOnly.every((value) => value > 0.05),
        `the baked LightProbeGrid alone gave ${formatRgb(gridOnly)}; the bake saw nothing, so the test cannot tell winners apart`,
    );
    assert(!closeTo(gridOnly, oursOnly), 'the grid and our light give the same value; the test cannot tell them apart');
    const summed = sum(gridOnly, oursOnly);

    const readings: Record<string, unknown> = {
        gridAlone: gridOnly,
        oursAlone: oursOnly,
        sumIfAdded: summed,
    };

    ours.priority = 1;
    grid.priority = 0;
    const oursHigher = await readCentre(renderer, coexistence);
    assert(closeTo(oursHigher, oursOnly), `ours priority 1 vs grid 0: read ${formatRgb(oursHigher)}, expected ours alone ${formatRgb(oursOnly)}`);
    assert(!closeTo(oursHigher, summed), 'ours-higher reading equals the sum: the two were added');

    ours.priority = 0;
    grid.priority = 1;
    const gridHigher = await readCentre(renderer, coexistence);
    assert(closeTo(gridHigher, gridOnly), `grid priority 1 vs ours 0: read ${formatRgb(gridHigher)}, expected grid alone ${formatRgb(gridOnly)}`);
    assert(!closeTo(gridHigher, summed), 'grid-higher reading equals the sum: the two were added');

    ours.priority = 0;
    grid.priority = 0;
    const tie = await readCentre(renderer, coexistence);
    const tieWinner = closeTo(tie, oursOnly) ? 'ours' : closeTo(tie, gridOnly) ? 'grid' : 'neither';
    assert(tieWinner !== 'neither', `equal priorities gave ${formatRgb(tie)}, which is neither light alone (summed would be ${formatRgb(summed)})`);
    assert(!closeTo(tie, summed), 'equal priorities summed the two lights');

    readings.oursHigherPriority = oursHigher;
    readings.gridHigherPriority = gridHigher;
    readings.equalPriority = tie;
    readings.equalPriorityWinner = tieWinner;
    readings.oursCreatedBeforeGrid = ours.id < grid.id;
    data.coexistence = readings;

    // ---------------------------------------------------------------------
    // 2. scene.environment set, env diffuse off (workaround), Standard material.
    // ---------------------------------------------------------------------
    data.environment = await checkEnvironmentDiffuse(renderer, wrapped);

    // ---------------------------------------------------------------------
    // 3. A CubeCamera capture (what a reflection-probe capture does).
    // ---------------------------------------------------------------------
    const capture = createTestScene();
    const captureLight = new GiVolumeLight(wrapped, { center: VOLUME_CENTER, size: VOLUME_SIZE });
    capture.scene.add(captureLight);
    const cubeTarget = new CubeRenderTarget(32, { type: FloatType });
    const cubeCamera = new CubeCamera(0.1, 50, cubeTarget);
    cubeCamera.position.set(0, 3, 0);
    capture.scene.add(cubeCamera);

    const downFace = 3; // -Y
    const upFace = 2; // +Y: the empty sky above the plane
    const captureAndRead = async (faceIndex: number): Promise<number[]> => {
        for (let frame = 0; frame < FRAMES; frame++) cubeCamera.update(renderer, capture.scene);
        return (await readBlock(renderer, cubeTarget as unknown as RenderTarget, 16, 16, 4, faceIndex)).mean;
    };
    const captureDown = await captureAndRead(downFace);
    const captureUp = await captureAndRead(upFace);
    assertRgbClose('cube capture, -Y face (plane lit by our GI)', captureDown, expectedOurs);
    assertRgbClose('cube capture, +Y face (empty sky)', captureUp, [0, 0, 0]);

    captureLight.visible = false;
    const captureDownWithoutLight = await captureAndRead(downFace);
    assertRgbClose('cube capture without our light', captureDownWithoutLight, [0, 0, 0]);
    captureLight.visible = true;

    data.cubeCapture = {
        downFace: captureDown,
        upFace: captureUp,
        downFaceWithoutLight: captureDownWithoutLight,
        expected: expectedOurs,
    };

    // Screenshot: the coexistence scene (the first test's view).
    renderer.render(coexistence.scene, coexistence.camera);
    await device.queue.onSubmittedWorkDone();
    textures.destroy();
});
