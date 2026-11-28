/**
 * Seeing what the probes do: DDGI's debug outputs, drawn in three.js
 * (debug-views.ts).
 *
 * - **probes**: a dot at each probe, where it really sits (probes inside
 *   walls move out), in its state's colour: green active, red inside
 *   geometry, blue with no surface nearby, yellow not updated yet.
 * - **view** (or `?view=`): the picture, or a screen debug view: GI alone on white
 *   surfaces, how much of each surface GI covers (green full, red where the
 *   renderer's own light takes over), or the states of the probes each
 *   point reads.
 * - **atlas**: the irradiance atlas, the light each probe has gathered, one
 *   picture per row of probes, bottom right.
 */

import * as THREE from 'three/webgpu';
import GUI from 'three/addons/libs/lil-gui.module.min.js';
import { DDGI_DEBUG_VIEW, DDGITechnique } from '@pmndrs/gi/ddgi';
import { DDGIVolume } from '@pmndrs/gi/three';
import { createRenderer } from '../../shared/renderer.ts';
import { loadScene } from '../../shared/scene.ts';
import { showStats } from '../../shared/stats.ts';
import { IrradianceAtlas, ProbeDots, ScreenDebugMaterial, ShadingResources } from './debug-views.ts';

const renderer = await createRenderer();
const { scene, camera, probes } = loadScene('two-room', renderer, { cameraBookmark: 0, exposureEv: -1 });

const volume = DDGIVolume.fromSettings(probes);
scene.add(volume);
await volume.start(renderer, scene);

// The debug outputs come from the DDGI technique the volume started, and its shading module.
const technique = volume.platform?.outputTechnique;
if (!(technique instanceof DDGITechnique)) throw new Error('The volume has no DDGI technique.');
const resources = new ShadingResources(technique.shading);
const screenView = new ScreenDebugMaterial(resources);
const atlas = new IrradianceAtlas(technique, resources);
const pointsView = technique.debugViews.find((view) => view.kind === 'points');
if (pointsView?.kind !== 'points') throw new Error('DDGI has no points view.');
const dots = new ProbeDots(pointsView, 0.035); // 3.5 cm spheres; the probes are 0.5 m apart
scene.add(dots.mesh);

// The picture, or a screen debug view by its id. `?view=gi-only` (or another name) opens on one.
const VIEWS = { picture: -1, 'gi-only': DDGI_DEBUG_VIEW.giOnly, coverage: DDGI_DEBUG_VIEW.coverage, 'probe-states': DDGI_DEBUG_VIEW.probeStates };
const requested = new URLSearchParams(location.search).get('view');
const settings = { view: requested !== null && requested in VIEWS ? requested : 'picture', probes: true, atlas: true };
const gui = new GUI({ title: 'Debug' });
gui.add(settings, 'view', Object.keys(VIEWS));
gui.add(settings, 'probes');
gui.add(settings, 'atlas');

const size = new THREE.Vector2();
renderer.autoClear = false;
showStats(volume, { note: () => settings.view });
renderer.setAnimationLoop(() => {
    resources.refresh();
    const view = VIEWS[settings.view as keyof typeof VIEWS];
    screenView.view = view;
    scene.overrideMaterial = view < 0 ? null : screenView.material;
    dots.mesh.visible = settings.probes;
    // GI updates itself when three draws something lit; a screen view draws nothing lit, so update it here.
    if (view >= 0) volume.threeGI?.update(camera);

    renderer.clear();
    renderer.render(scene, camera);
    if (settings.atlas) {
        renderer.getSize(size);
        atlas.layout(size.x, size.y);
        renderer.clearDepth();
        renderer.render(atlas.scene, atlas.camera);
    }
});
