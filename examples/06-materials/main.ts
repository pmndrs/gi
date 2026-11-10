/**
 * GI's rays see each material as three draws it: its colour and textures
 * where a ray hits, see-through glass, and cut-out holes. Three scenes, each
 * from three's ordinary materials (`MeshStandardMaterial` with a map and
 * `alphaTest`, `MeshPhysicalMaterial` with `transmission`); the adapter needs
 * nothing extra.
 *
 * - alpha-checker: a lit room and a dark room share a wall that is a
 *   checkerboard of holes (`alphaTest`, exactly half open). The dark room's
 *   light comes only through the holes.
 * - glass-wall: the shared wall is one mesh with two materials, plaster and
 *   a clear glass pane. Light reaches the dark room only through the pane.
 * - uv-board: eight panels in the sun, each sampling its texture another way
 *   (repeat, mirror, clamp, offset, rotation, the second uv set...). Their
 *   colours bounce onto the ground as the textures say.
 *
 * Switching scenes reloads the page with `?scene=`.
 */

import GUI from 'three/addons/libs/lil-gui.module.min.js';
import { GIVolume } from '@pmndrs/ddgi/three';
import { createRenderer } from '../shared/renderer.ts';
import { loadScene, type ExampleSceneId } from '../shared/scene.ts';
import { showStats } from '../shared/stats.ts';

const SCENES = {
    'alpha-checker': { id: 'alpha-checker', cameraBookmark: 1 },
    'glass-wall': { id: 'glass-wall', cameraBookmark: 1 },
    'uv-board': { id: 'uv-board', cameraBookmark: 1 },
} as const satisfies Record<string, { id: ExampleSceneId; cameraBookmark: number }>;
type SceneName = keyof typeof SCENES;

const requested = new URLSearchParams(location.search).get('scene');
const sceneName: SceneName = requested !== null && requested in SCENES ? (requested as SceneName) : 'alpha-checker';
const choice = SCENES[sceneName];

const renderer = await createRenderer();
const { scene, camera, probes, title } = loadScene(choice.id, renderer, { cameraBookmark: choice.cameraBookmark });

const volume = GIVolume.fromDDGIVolume(probes);
scene.add(volume);
await volume.start(renderer, scene);

const gui = new GUI({ title: 'Materials' });
gui.add({ scene: sceneName as string }, 'scene', Object.keys(SCENES)).onChange((name: string) => {
    location.search = `?scene=${name}`;
});

showStats(volume, { note: () => title });
renderer.setAnimationLoop(() => renderer.render(scene, camera));
