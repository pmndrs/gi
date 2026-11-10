/**
 * The repo's procedural test scenes as three.js scenes, for the examples.
 *
 * Each scene is described by a manifest in scenes/manifests (geometry
 * builders, lights, camera bookmarks, the suggested probe box), and its
 * geometry is made in code by scenes/src/procedural, so nothing is
 * downloaded and everything here is ours to share (MIT). A manifest's model
 * parts (downloaded glTF files, such as the courtyard's tree) are left out.
 */

import * as THREE from 'three/webgpu';
import { SkyMesh } from 'three/addons/objects/SkyMesh.js';
import type { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import type { DDGIVolumeSettings } from '@pmndrs/gi/ddgi';
import { buildProceduralParts, mergeScenes } from '../../scenes/src/build-scene.ts';
import { manifestLightToSceneLight, type SceneManifest } from '../../scenes/src/manifest.ts';
import { lightToThree, proceduralSceneToThree } from '../../scenes/src/three-scene.ts';
import { orbitCamera } from './renderer.ts';

const manifests = import.meta.glob('../../scenes/manifests/*.json', { eager: true, import: 'default' }) as Record<string, SceneManifest>;

/** The procedural scenes the examples use. */
export type ExampleSceneId = 'cornell-box-point' | 'two-room' | 'closed-door' | 'cabinet-row' | 'courtyard-tree' | 'alpha-checker' | 'glass-wall' | 'uv-board';

export interface LoadSceneOptions {
    /** Which of the manifest's camera bookmarks to open on, 0-based. Default 0. */
    readonly cameraBookmark?: number;
    /** The picture's exposure in stops (+1 is twice as bright). Default the manifest's, or 0. */
    readonly exposureEv?: number;
}

export interface ExampleScene {
    readonly scene: THREE.Scene;
    readonly camera: THREE.PerspectiveCamera;
    readonly controls: OrbitControls;
    /** The manifest's lights, in its order, each named by its label. */
    readonly lights: THREE.Light[];
    /** three's `SkyMesh` for a scene under the sky, or null indoors. */
    readonly sky: SkyMesh | null;
    /** The probe box the scene suggests: pass it to `DDGIVolume.fromDDGIVolume`. */
    readonly probes: DDGIVolumeSettings;
    readonly title: string;
}

function findManifest(id: ExampleSceneId): SceneManifest {
    const manifest = Object.values(manifests).find((candidate) => candidate.id === id);
    if (!manifest) throw new Error(`No scene manifest "${id}" in scenes/manifests.`);
    return manifest;
}

/**
 * Builds a scene: its geometry, its lights, its sky (three's `SkyMesh`
 * toward the manifest's sun, which GI reads too), its camera at a bookmark
 * with orbit controls, and the picture's exposure from the manifest.
 */
export function loadScene(id: ExampleSceneId, renderer: THREE.WebGPURenderer, options: LoadSceneOptions = {}): ExampleScene {
    const manifest = findManifest(id);
    const scene = new THREE.Scene();
    // A black world unless the manifest says otherwise. GI reads a background colour as a uniform sky.
    scene.background = new THREE.Color(0, 0, 0);

    const content = proceduralSceneToThree(mergeScenes(buildProceduralParts(manifest)), manifest.id);
    scene.add(content);
    content.updateMatrixWorld(true);

    // Directional lights are placed, and their shadows fitted, around the geometry.
    const bounds = new THREE.Box3().setFromObject(content);
    const center = bounds.getCenter(new THREE.Vector3());
    const fitting = { center: center.toArray(), radius: bounds.getSize(new THREE.Vector3()).length() / 2 };
    const lights = manifest.lights.map((light) => lightToThree(manifestLightToSceneLight(light), fitting));
    for (const light of lights) scene.add(light);

    let sky: SkyMesh | null = null;
    const environment = manifest.environment;
    if (environment.type === 'color') {
        const scale = environment.intensity ?? 1;
        scene.background = new THREE.Color().setRGB(environment.color[0] * scale, environment.color[1] * scale, environment.color[2] * scale);
    } else if (environment.type === 'procedural') {
        sky = new SkyMesh();
        sky.scale.setScalar(1000);
        sky.turbidity.value = environment.turbidity ?? 3;
        sky.sunPosition.value.set(...environment.sunDirection).normalize();
        scene.add(sky);
    }

    const bookmark = manifest.cameras[options.cameraBookmark ?? 0] ?? manifest.cameras[0];
    const camera = new THREE.PerspectiveCamera(bookmark.fovYDeg, window.innerWidth / window.innerHeight, 0.05, 2000);
    camera.position.set(...bookmark.position);
    const controls = orbitCamera(renderer, camera, new THREE.Vector3(...bookmark.target));

    renderer.toneMappingExposure = 2 ** (options.exposureEv ?? manifest.exposureEv ?? 0);
    return {
        scene,
        camera,
        controls,
        lights,
        sky,
        probes: { center: manifest.volume.center, size: manifest.volume.size, spacing: manifest.volume.probeSpacing },
        title: manifest.title,
    };
}
