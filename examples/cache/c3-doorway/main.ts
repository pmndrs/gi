/**
 * Light through a doorway: where the radiance cache's second bounce comes
 * from decides how much light reaches a closed room. Two rooms joined by a
 * doorway; a lamp lights the near room, and the far room, where the camera
 * stands, has no light of its own.
 *
 * Three sources for the world cache's second bounce, switched live:
 * - DDGI underneath (the default): a DDGI volume stacked under the cache
 *   (`gi.setFallback`), so each second-bounce hit reads DDGI's field, which
 *   carries every later bounce (`secondBounce: 'auto'`).
 * - GI-1.2's own rays (`secondBounce: 'gi1-rays'`): two bounces at most.
 *   The volume stays stacked but unread, so the switch is one live setting.
 * - SHaRC (`worldCache: 'sharc'`): a different world cache that carries
 *   every bounce itself. The world cache is read at `prepare`, so this
 *   switch makes the platform again, without DDGI (SHaRC doesn't read it).
 *
 * The door swings on its hinge, as in ddgi/04: nothing is called, the
 * adapter sees it move. The panel bottom right is the far room's GI
 * (far-room.ts).
 */

import * as THREE from 'three/webgpu';
import GUI from 'three/addons/libs/lil-gui.module.min.js';
import { mrt, normalView, output, packNormalToRGB, pass } from 'three/tsl';
import { createGI, type GIPlatform } from '@pmndrs/gi/core';
import { DDGITechnique } from '@pmndrs/gi/ddgi';
import { RadianceCacheTechnique } from '@pmndrs/gi/radiance-cache';
import { ddgi, giAlbedo, GI_ALBEDO_OUTPUT, GI_NORMAL_OUTPUT, screenInputs, ThreeGI } from '@pmndrs/gi/three';
import { createRendererAndDevice } from '../../shared/renderer.ts';
import { loadScene } from '../../shared/scene.ts';
import { showFarRoom, type Bounce } from './far-room.ts';

// The screen inputs need a single-sampled pass.
const { renderer, device } = await createRendererAndDevice({ antialias: false });
// The closed-door scene, its door swung open: the two-room scene with a door that moves. The camera stands in the far room.
const { scene, camera, controls, probes } = loadScene('closed-door', renderer, { cameraBookmark: 2 });
controls.enabled = false; // the reading is over this view

const door = scene.getObjectByName('door-panel');
if (!door) throw new Error('The closed-door scene has no door-panel.');
const hinge = new THREE.Group();
hinge.position.set(4.08, 0, 1.546);
scene.add(hinge);
hinge.attach(door);
const setDoor = (open: boolean) => (hinge.rotation.y = open ? -Math.PI / 2 : 0);

/** The cache, its platform and the pipeline that draws it. */
interface Rig {
    readonly gi: GIPlatform;
    readonly cache: RadianceCacheTechnique;
    readonly threeGI: ThreeGI;
    readonly pipeline: THREE.RenderPipeline;
    dispose(): void;
}

async function makeRig(sharc: boolean, bounce: Bounce): Promise<Rig> {
    const gi = createGI(device);
    const cache = gi.addTechnique(new RadianceCacheTechnique(sharc ? { worldCache: 'sharc' } : { secondBounce: bounce === 'gi1-rays' ? 'gi1-rays' : 'auto' }));
    gi.setOutput(cache);
    let under: DDGITechnique | null = null;
    if (!sharc) {
        // Both rooms, a probe every 0.5 m (the scene's suggested box).
        under = gi.addTechnique(new DDGITechnique({ volume: probes }));
        gi.setFallback(cache, under); // before prepare: the cache builds its DDGI lookup then
    }
    const preparing = gi.prepare();
    const threeGI = new ThreeGI(gi, scene, { preparing, budget: { units: 256 } }); // DDGI's probes a frame
    await preparing;
    // DDGI warms up first, so the cache's later bounces don't start from a black field.
    while (under?.warmingUp) {
        threeGI.update(camera, 0);
        await device.queue.onSubmittedWorkDone();
    }

    const pipeline = new THREE.RenderPipeline(renderer);
    const scenePass = pass(scene, camera);
    const screen = screenInputs(scenePass, threeGI);
    scenePass.setMRT(mrt({ output, [GI_ALBEDO_OUTPUT]: giAlbedo, [GI_NORMAL_OUTPUT]: packNormalToRGB(normalView), ...screen.outputs }));
    pipeline.outputNode = screen.encodeBefore(ddgi(scenePass, threeGI));
    return {
        gi,
        cache,
        threeGI,
        pipeline,
        dispose() {
            pipeline.dispose();
            screen.dispose();
            scenePass.dispose();
            threeGI.dispose();
            gi.dispose();
        },
    };
}

const settings = { bounce: 'ddgi' as Bounce, open: true };
let rig: Rig | null = await makeRig(false, settings.bounce);
const farRoom = showFarRoom(device, camera, () => rig?.cache ?? null, () => rig?.gi ?? null, settings);

async function switchBounce(): Promise<void> {
    const bounce = settings.bounce;
    const sharc = bounce === 'sharc';
    if (rig && sharc === (rig.cache.settings.worldCache === 'sharc')) {
        rig.cache.settings.secondBounce = bounce === 'gi1-rays' ? 'gi1-rays' : 'auto'; // live; it empties the world cache
    } else {
        rig?.dispose();
        rig = null;
        rig = await makeRig(sharc, bounce);
    }
    farRoom.restart();
}

let switching = Promise.resolve();
const gui = new GUI({ title: 'Far room' });
gui.add(settings, 'bounce', { 'DDGI underneath': 'ddgi', "GI-1.2's own rays": 'gi1-rays', SHaRC: 'sharc' })
    .name('second bounce')
    .onChange(() => (switching = switching.then(switchBounce))); // one switch at a time
gui.add(settings, 'open')
    .name('door open')
    .onChange((open: boolean) => {
        setDoor(open);
        farRoom.restart();
    });
setDoor(settings.open);

renderer.setAnimationLoop(() => {
    if (!rig) {
        renderer.render(scene, camera); // direct light only while the SHaRC switch makes the platform again
        return;
    }
    rig.threeGI.update(camera); // the frame's GI work; the screen stage runs inside the pipeline
    rig.pipeline.render();
});
