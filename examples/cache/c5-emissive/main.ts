/**
 * C5 · Light from surfaces. A room with no light objects at all: two light
 * strips under the ceiling and a screen are its only light. The radiance
 * cache, on its default settings, lights the room from them, because a
 * screen probe's ray that hits an emissive surface takes its emission.
 *
 * - The emitters are ordinary three materials: `emissive`,
 *   `emissiveIntensity` and `emissiveMap`. GI reads all three, so editing
 *   them is all it takes; `markChanged` makes GI see the edit at once.
 * - The screen's two pictures are made in code (`DataTexture`), and GI's
 *   rays read the texel they hit.
 * - "GI only" shows what the composite adds, `giAlbedo × E / π`: the
 *   picture minus three's own colour, which here is just the emitters.
 */

import * as THREE from 'three/webgpu';
import { max, mix, mrt, normalView, output, packNormalToRGB, pass, uniform, vec3, vec4 } from 'three/tsl';
import GUI from 'three/addons/libs/lil-gui.module.min.js';
import { createGI } from '@pmndrs/gi/core';
import { RadianceCacheTechnique } from '@pmndrs/gi/radiance-cache';
import { ddgi, giAlbedo, GI_ALBEDO_OUTPUT, GI_NORMAL_OUTPUT, screenInputs, ThreeGI } from '@pmndrs/gi/three';
import { EMISSIVE_ROOM_MATERIALS } from '../../../scenes/src/procedural/emissive-room.ts';
import { createRendererAndDevice } from '../../shared/renderer.ts';
import { loadScene } from '../../shared/scene.ts';
import { showCacheStats } from '../../shared/cache-stats.ts';
import { showPassTiming } from '../../shared/pass-timing.ts';

// The screen inputs need a single-sampled pass.
const { renderer, device } = await createRendererAndDevice({ antialias: false });
const { scene, camera } = loadScene('emissive-room', renderer);

// The cache on its defaults: GI-1.2's hash grid, with its own second-bounce rays.
const gi = createGI(device, { gpuTiming: device.features.has('timestamp-query') });
const cache = gi.addTechnique(new RadianceCacheTechnique());
gi.setOutput(cache);
const preparing = gi.prepare();
const threeGI = new ThreeGI(gi, scene, { preparing });
await preparing;

const pipeline = new THREE.RenderPipeline(renderer);
const scenePass = pass(scene, camera);
const screen = screenInputs(scenePass, threeGI);
scenePass.setMRT(mrt({ output, [GI_ALBEDO_OUTPUT]: giAlbedo, [GI_NORMAL_OUTPUT]: packNormalToRGB(normalView), ...screen.outputs }));
const composite: THREE.Node = ddgi(scenePass, threeGI);
const fullPicture = vec4(composite as THREE.Node<'vec4'>);
// GI only: the composite minus three's own colour (the emitters themselves; there are no lights).
const giOnly = vec4(max(fullPicture.rgb.sub(scenePass.getTextureNode('output').rgb), vec3(0)), 1);
const showGIOnly = uniform(0);
pipeline.outputNode = screen.encodeBefore(mix(fullPicture, giOnly, showGIOnly));

// The emitters: the strips share one material, the screen has its own.
const meshesWith = (name: string) => {
    const meshes: THREE.Mesh[] = [];
    scene.traverse((object) => {
        if (object instanceof THREE.Mesh && (object.material as THREE.Material).name === name) meshes.push(object);
    });
    return meshes;
};
const strips = meshesWith(EMISSIVE_ROOM_MATERIALS.strip);
const screens = meshesWith(EMISSIVE_ROOM_MATERIALS.screen);
const stripMaterial = strips[0].material as THREE.MeshStandardMaterial;
const screenMaterial = screens[0].material as THREE.MeshStandardMaterial;

/** A picture for the screen, made in code: `paint(u, v)` gives an sRGB colour, 0 to 255 per channel. */
function screenPicture(name: string, paint: (u: number, v: number) => readonly [number, number, number]): THREE.DataTexture {
    const width = 192;
    const height = 108;
    const data = new Uint8Array(width * height * 4);
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            const [r, g, b] = paint((x + 0.5) / width, (y + 0.5) / height);
            data.set([r, g, b, 255], (y * width + x) * 4);
        }
    }
    const texture = new THREE.DataTexture(data, width, height);
    texture.name = name;
    texture.colorSpace = THREE.SRGBColorSpace;
    texture.needsUpdate = true;
    return texture;
}
const BARS = [[235, 235, 235], [235, 235, 16], [16, 235, 235], [16, 235, 16], [235, 16, 235], [235, 16, 16], [16, 16, 235]] as const;
const pictures = {
    'colour bars': screenPicture('colour bars', (u) => BARS[Math.min(BARS.length - 1, Math.floor(u * BARS.length))]),
    'warm and cool halves': screenPicture('warm and cool halves', (u) => (u < 0.5 ? [255, 120, 20] : [30, 90, 255])),
};

// The panel edits the materials; markChanged has GI see each edit at once.
const settings = {
    stripColor: `#${stripMaterial.emissive.getHexString()}`,
    brightness: stripMaterial.emissiveIntensity,
    picture: 'colour bars' as string,
    view: 'full picture',
};
const changed = (meshes: THREE.Mesh[]) => meshes.forEach((mesh) => threeGI.markChanged(mesh));
const showPicture = (name: string) => {
    const picture = pictures[name as keyof typeof pictures] ?? null; // null: the screen is off
    screenMaterial.emissive.setScalar(picture ? 1 : 0);
    screenMaterial.emissiveMap = picture;
    screenMaterial.needsUpdate = true;
    changed(screens);
};
showPicture(settings.picture);

const gui = new GUI({ title: 'Emitters' });
const stripsFolder = gui.addFolder('Light strips');
stripsFolder.addColor(settings, 'stripColor').name('colour').onChange((color: string) => {
    stripMaterial.emissive.set(color);
    changed(strips);
});
stripsFolder.add(settings, 'brightness', 0, 30, 0.1).onChange((value: number) => {
    stripMaterial.emissiveIntensity = value;
    changed(strips);
});
gui.add(settings, 'picture', [...Object.keys(pictures), 'off']).name('screen').onChange(showPicture);
gui.add(settings, 'view', ['full picture', 'GI only']).onChange((view: string) => (showGIOnly.value = view === 'GI only' ? 1 : 0));

showCacheStats(device, gi, cache, () => 'no lights: emitters only');
showPassTiming(gi);
renderer.setAnimationLoop(() => {
    threeGI.update(camera); // the frame's GI work; the screen stage runs inside the pipeline
    pipeline.render();
});
