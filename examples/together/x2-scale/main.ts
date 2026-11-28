/**
 * Scale: how each method covers a big world. A generated town about 200 m
 * across (the `block-world` scene, from a fixed seed), flown through.
 *
 * - **DDGI with several volumes** (`DDGISettings.volumes`): one fixed volume
 *   over the whole town at a coarse spacing, and one or two windows of
 *   32 × 8 × 32 probes that follow the camera (`follow: 'camera'`), each
 *   finer and at a higher `priority`, so the probes are densest where you
 *   are. Where volumes overlap the higher priority answers, and the next one
 *   fills in across its fade band.
 * - **The radiance cache with SHaRC** (`worldCache: 'sharc'`): no volume at
 *   all. Its cells are hashed from their position, so they reach anywhere,
 *   and a cell's size grows with its distance from the camera.
 *
 * The volumes and the world cache are read at `prepare`, so a change of
 * method, count or spacing makes the platform again (as C3 does), while the
 * scene, the camera and three's renderer stay.
 */

import * as THREE from 'three/webgpu';
import GUI from 'three/addons/libs/lil-gui.module.min.js';
import { mrt, normalView, output, packNormalToRGB, pass } from 'three/tsl';
import { createGI } from '@pmndrs/gi/core';
import { DDGITechnique, type DDGIVolumeSettings } from '@pmndrs/gi/ddgi';
import { RadianceCacheTechnique } from '@pmndrs/gi/radiance-cache';
import { ddgi, giAlbedo, GI_ALBEDO_OUTPUT, GI_NORMAL_OUTPUT, screenInputs, ThreeGI } from '@pmndrs/gi/three';
import { createRendererAndDevice } from '../../shared/renderer.ts';
import { loadScene } from '../../shared/scene.ts';
import { FlyCamera } from './fly.ts';
import { VolumeOutlines } from './outlines.ts';
import { showReadouts, type Running } from './readouts.ts';
import './x2.css';

// The screen inputs (the cache) need a single-sampled pass.
const { renderer, device } = await createRendererAndDevice({ antialias: false });
const { scene, camera, controls } = loadScene('block-world', renderer);
controls.dispose(); // the fly camera instead of orbiting
const fly = new FlyCamera(camera, renderer.domElement);

const settings = { method: 'ddgi' as 'ddgi' | 'sharc', volumes: 2, near: 1.5, far: 8, showVolumes: true, tour: true };

/** The DDGI volumes: the whole town at `far`, then camera-following windows, finer and nearer, at higher priority. */
function townVolumes(count: number, near: number, far: number): DDGIVolumeSettings[] {
    const town: DDGIVolumeSettings = { center: [0, 16, 0], size: [192, 32, 192], spacing: far }; // the ground to the tallest roof
    const window = (spacing: number, priority: number): DDGIVolumeSettings => ({
        center: camera.position.toArray(),
        size: [31 * spacing, 7 * spacing, 31 * spacing], // 32 × 8 × 32 probes
        spacing,
        follow: 'camera',
        priority,
    });
    const middle = Math.round(Math.sqrt(near * far) * 2) / 2; // halfway between, on a log scale
    return [town, ...(count >= 3 ? [window(middle, 1)] : []), ...(count >= 2 ? [window(near, 2)] : [])];
}

/** One method's platform and the pipeline that draws it. */
interface Rig extends Running {
    readonly threeGI: ThreeGI;
    readonly pipeline: THREE.RenderPipeline;
    dispose(): void;
}

async function makeRig(): Promise<Rig> {
    const gi = createGI(device, { gpuTiming: device.features.has('timestamp-query') });
    const technique =
        settings.method === 'ddgi'
            ? new DDGITechnique({ volumes: townVolumes(settings.volumes, settings.near, settings.far) })
            : new RadianceCacheTechnique({ worldCache: 'sharc' });
    gi.addTechnique(technique);
    gi.setOutput(technique);
    const preparing = gi.prepare();
    const threeGI = new ThreeGI(gi, scene, { preparing }); // DDGI's probe updates inside 2 ms; the cache ignores the budget
    await preparing;

    const pipeline = new THREE.RenderPipeline(renderer);
    const scenePass = pass(scene, camera);
    const screen = technique instanceof RadianceCacheTechnique ? screenInputs(scenePass, threeGI) : null;
    scenePass.setMRT(mrt({ output, [GI_ALBEDO_OUTPUT]: giAlbedo, [GI_NORMAL_OUTPUT]: packNormalToRGB(normalView), ...screen?.outputs }));
    const composite = ddgi(scenePass, threeGI);
    pipeline.outputNode = screen ? screen.encodeBefore(composite) : composite;

    const probes = technique instanceof DDGITechnique ? (technique.field?.volumes ?? []).map((volume) => `${volume.grid.probeCount.toLocaleString('en')} at ${volume.grid.spacing[0]} m`) : [];
    return {
        label: technique instanceof DDGITechnique ? `DDGI · ${settings.volumes} volume${settings.volumes > 1 ? 's' : ''}, ${settings.volumes > 1 ? `${settings.near} m to ` : ''}${settings.far} m` : 'Radiance cache · SHaRC',
        detail: technique instanceof DDGITechnique ? `probes: ${probes.join(', ')}` : 'no volume: 1M slots, cells sized by distance',
        gi,
        technique,
        threeGI,
        pipeline,
        dispose() {
            pipeline.dispose();
            screen?.dispose();
            scenePass.dispose();
            threeGI.dispose();
            gi.dispose();
        },
    };
}

let rig: Rig | null = null;
let switching = Promise.resolve();
const rebuild = () =>
    (switching = switching.then(async () => {
        rig?.dispose();
        rig = null;
        rig = await makeRig();
    }));
rebuild();

const gui = new GUI({ title: 'Scale' });
gui.add(settings, 'method', { 'DDGI: volumes': 'ddgi', 'Radiance cache: SHaRC': 'sharc' }).onChange(() => {
    volumesFolder.show(settings.method === 'ddgi');
    cellsFolder.show(settings.method === 'sharc');
    rebuild();
});
const rebuildDDGI = () => settings.method === 'ddgi' && rebuild();
const volumesFolder = gui.addFolder('DDGI volumes');
volumesFolder.add(settings, 'volumes', { '1: the town': 1, '2: + near window': 2, '3: + middle window': 3 }).name('count').onChange(rebuildDDGI);
volumesFolder.add(settings, 'near', [1, 1.5, 2, 3]).name('near spacing (m)').onChange(rebuildDDGI);
volumesFolder.add(settings, 'far', [4, 8, 12]).name('town spacing (m)').onChange(rebuildDDGI);
volumesFolder.add(settings, 'showVolumes').name('show volumes');
const cellsFolder = gui.addFolder('Radiance cache');
const showCells = cellsFolder.add({ showCells: false }, 'showCells').name('show cells: not in the library yet').disable();
showCells.domElement.title = 'The cache has no view of its cells in world space yet, so their size by distance cannot be shown (a TASKS row).';
cellsFolder.hide();
gui.add(settings, 'tour')
    .name('tour the streets')
    .listen() // a key or a drag ends the tour
    .onChange((on: boolean) => (on ? fly.tour() : (fly.touring = false)));
gui.add(fly, 'speed', 2, 30, 1).name('fly speed (m/s)');
const jumps = {
    above: () => fly.jump({ x: -60, y: 55, z: 90 }, { x: 0, y: 0, z: 0 }),
    // The 8 m yard of seed 1's lot 3-5, from its shaded corner up to the sunlit wings.
    courtyard: () => fly.jump({ x: -9, y: 1.7, z: 39.5 }, { x: -15, y: 5, z: 33 }),
};
gui.add(jumps, 'above').name('jump above the town');
gui.add(jumps, 'courtyard').name('jump into a courtyard');

const outlines = new VolumeOutlines(scene);
showReadouts(device, () => rig, gui.domElement); // GPU ms per pass and per method, memory, frame time

const timer = new THREE.Timer();
renderer.setAnimationLoop(() => {
    timer.update();
    fly.update(Math.min(timer.getDelta(), 0.1));
    settings.tour = fly.touring;
    if (!rig) {
        renderer.render(scene, camera); // direct light only while the platform is made again
        return;
    }
    outlines.update(rig.technique instanceof DDGITechnique ? rig.technique : null, settings.showVolumes);
    rig.threeGI.update(camera); // the frame's GI work; the cache's screen stage runs inside the pipeline
    rig.pipeline.render();
});
