/**
 * Quality presets: one word for GI's quality. `low`, `medium` and `high`
 * each set the rays per probe, the size of each probe's tiles, the GPU
 * budget per frame and the size of GI's copy of the textures. A volume made
 * without probe counts also takes the preset's spacing: 2 m apart for `low`,
 * 1.5 m for `medium`, 1 m for `high`.
 *
 * The preset sizes GI's buffers when it starts, so changing it means
 * `dispose()` and a new start. Here each change makes a new volume.
 *
 * `recommendPreset(adapter.info, adapter.limits)` from `@pmndrs/ddgi/core`
 * suggests one for the GPU (`low` on integrated and mobile GPUs); this page
 * opens on its advice.
 */

import GUI from 'three/addons/libs/lil-gui.module.min.js';
import { DDGI_PRESET_NAMES, recommendPreset, type DDGIPresetName } from '@pmndrs/ddgi/core';
import { GIVolume } from '@pmndrs/ddgi/three';
import { createRenderer } from '../shared/renderer.ts';
import { loadScene } from '../shared/scene.ts';
import { showStats } from '../shared/stats.ts';

const renderer = await createRenderer();
const { scene, camera, probes } = loadScene('cornell-box-point', renderer);

/** A volume over the scene's probe box, made without probe counts so the preset's spacing applies. */
function makeVolume(): GIVolume {
    const volume = new GIVolume(probes.size[0], probes.size[1], probes.size[2]);
    volume.position.set(...probes.center);
    scene.add(volume);
    return volume;
}

const adapter = await navigator.gpu.requestAdapter();
const settings: { preset: DDGIPresetName } = { preset: adapter ? recommendPreset(adapter.info, adapter.limits) : 'high' };
let volume = makeVolume();
await volume.start(renderer, scene, { preset: settings.preset });

let switching = false;
const gui = new GUI({ title: 'Quality' });
gui.add(settings, 'preset', [...DDGI_PRESET_NAMES]).onChange(async (preset: DDGIPresetName) => {
    if (switching) return;
    switching = true;
    gui.controllers[0].disable();
    volume.dispose();
    scene.remove(volume);
    volume = makeVolume();
    await volume.start(renderer, scene, { preset });
    gui.controllers[0].enable();
    switching = false;
});

showStats(() => volume, { note: () => (switching ? `starting ${settings.preset}…` : `preset ${settings.preset}`) });
renderer.setAnimationLoop(() => renderer.render(scene, camera));
