/**
 * Builds a three.js scene from a manifest: the model (procedural or a
 * downloaded glTF), the manifest's direct lights, and the manifest's sky.
 *
 * This reuses the scenes package's helpers (`proceduralSceneToThree`,
 * `lightToThree`, `emissiveQuadLights`, `applyModelTransform`). The only
 * loading code written here is the glTF part with progress, because the
 * check page's loader reports nothing while a 270 MB model downloads.
 *
 * Nothing is added beyond what the manifest says: no fill light, no ambient,
 * no environment map. Every technique sees this same scene (DECISIONS: a fair
 * wipe), so whatever light a technique gets from the sky, it gets from the
 * manifest's sky: the shared sky file (`scene/sky.ts`). See `addSky` for how
 * each technique takes it.
 */

import {
    Box3,
    Color,
    Group,
    Scene,
    Vector3,
    type DataTexture,
    type Light,
    type Mesh,
    type MeshStandardMaterial,
    type Object3D,
    type RectAreaLight,
} from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { RectAreaLightTexturesLib } from 'three/addons/lights/RectAreaLightTexturesLib.js';
import { RectAreaLightNode } from 'three/webgpu';
import { manifestLightToSceneLight, type ManifestEnvironment, type SceneManifest } from '../../../../scenes/src/manifest';
import { buildProcedural } from '../../../../scenes/src/procedural';
import { computeSceneStats, type SceneStats } from '../../../../scenes/src/stats';
import { applyModelTransform, emissiveQuadLights, lightToThree, proceduralSceneToThree } from '../../../../scenes/src/three-scene';
import { LAYER_DIRECT_LIGHTS, LAYER_VIEW_GEOMETRY } from './layers';
import { findModelUrl } from './model-url';
import { createSkyDome, loadSkyTexture } from './sky';

export interface LoadProgress {
    /** Plain words for what is happening now. */
    stage: string;
    /** 0 to 1 when known. */
    fraction: number | null;
}

export interface LoadedScene {
    manifest: SceneManifest;
    scene: Scene;
    /** The model only, without lights. */
    content: Group;
    /** The manifest's lights and the emissive-quad area lights. */
    directLights: Light[];
    bounds: Box3;
    /** The shared sky file as a three.js texture, for a procedural environment; null otherwise. */
    sky: DataTexture | null;
    /** Length of the bounds' diagonal in metres; sets camera speed and clipping. */
    diagonal: number;
    stats: SceneStats;
}

/** Lets the browser paint between heavy steps, so progress text actually shows. */
export function nextFrame(): Promise<void> {
    return new Promise((resolve) => requestAnimationFrame(() => resolve()));
}

/** Black: the manifest's "no environment", as in the Cycles references. */
const NO_SKY = new Color(0, 0, 0);

/**
 * Puts the manifest's sky into the scene. Every technique draws the same sky; how much
 * light each one takes from it is part of what the comparison shows:
 *  - `LightProbeGrid` sees it in its cube-face captures, so the probes carry sky light.
 *  - VXGI has no sky term: its voxelizer skips the sky mesh, and it injects direct lights only.
 * Surfaces get no light from the sky directly (no `scene.environment`); see `scene/sky.ts`.
 *
 * - `none`: a black world.
 * - `color`: a uniform background colour, scaled by `intensity`.
 * - `procedural`: the shared sky file `scenes/sky/<scene id>.exr` (a calibrated Preetham
 *   sky, black below the horizon), on a sky dome. It has no sun disc: the sun is the
 *   manifest's directional light, and must not be counted twice. Cycles, the platform
 *   and this viewer all read that one file, so the skies match. `intensity` is already
 *   in the file.
 *
 * Returns the sky texture for a procedural sky, null otherwise.
 */
async function addSky(scene: Scene, manifest: SceneManifest): Promise<DataTexture | null> {
    const environment: ManifestEnvironment = manifest.environment;
    switch (environment.type) {
        case 'none':
            scene.background = NO_SKY.clone();
            return null;
        case 'color': {
            const scale = environment.intensity ?? 1;
            scene.background = new Color().setRGB(environment.color[0] * scale, environment.color[1] * scale, environment.color[2] * scale);
            return null;
        }
        case 'procedural': {
            scene.background = NO_SKY.clone();
            const sky = await loadSkyTexture(manifest.id);
            const dome = createSkyDome(sky);
            // Layer 0 for the probe bake; the viewing layer for the cameras.
            dome.layers.enable(LAYER_VIEW_GEOMETRY);
            scene.add(dome);
            return sky;
        }
    }
}

async function loadGltf(url: string, label: string, report: (progress: LoadProgress) => void): Promise<Group> {
    const loader = new GLTFLoader();
    const gltf = await loader.loadAsync(url, (event) => {
        const megabytes = (bytes: number) => (bytes / 1e6).toFixed(0);
        if (event.lengthComputable && event.total > 0) {
            report({
                stage: `Downloading ${label} (${megabytes(event.loaded)} of ${megabytes(event.total)} MB)`,
                fraction: (event.loaded / event.total) * 0.7,
            });
        } else {
            report({ stage: `Downloading ${label} (${megabytes(event.loaded)} MB)`, fraction: null });
        }
    });
    report({ stage: `Preparing ${label}`, fraction: 0.8 });
    await nextFrame();
    // Lights embedded in the file (Sponza has some) would double up the manifest's lights.
    const embedded: Object3D[] = [];
    gltf.scene.traverse((object) => {
        if ((object as Light).isLight) embedded.push(object);
    });
    for (const light of embedded) light.removeFromParent();
    return gltf.scene;
}

/** Shadows on, a little texture sharpening at grazing angles, and the viewing layer. */
function prepareMeshes(root: Object3D): void {
    root.traverse((object) => {
        object.layers.enable(LAYER_VIEW_GEOMETRY);
        const mesh = object as Mesh;
        if (!mesh.isMesh) return;
        mesh.castShadow = true;
        mesh.receiveShadow = true;
        for (const material of Array.isArray(mesh.material) ? mesh.material : [mesh.material]) {
            const map = (material as MeshStandardMaterial).map;
            if (map) map.anisotropy = 8;
        }
    });
}

/** Frees the GPU memory of a scene's geometry, materials and textures. Large scenes would otherwise pile up when switching. */
export function disposeScene(loaded: LoadedScene): void {
    loaded.sky?.dispose();
    loaded.scene.traverse((object) => {
        const mesh = object as Mesh;
        if (!mesh.isMesh) return;
        mesh.geometry.dispose();
        for (const material of Array.isArray(mesh.material) ? mesh.material : [mesh.material]) {
            for (const value of Object.values(material)) {
                if ((value as { isTexture?: boolean } | null)?.isTexture) (value as { dispose(): void }).dispose();
            }
            material.dispose();
        }
    });
}

export async function loadScene(manifest: SceneManifest, report: (progress: LoadProgress) => void): Promise<LoadedScene> {
    const scene = new Scene();
    const sky = await addSky(scene, manifest);

    const content = new Group();
    content.name = manifest.id;
    const emitters: RectAreaLight[] = [];
    for (const part of manifest.parts) {
        if (part.kind === 'procedural') {
            report({ stage: `Building ${manifest.title}`, fraction: null });
            const built = buildProcedural(part.builder, part.options);
            content.add(proceduralSceneToThree(built, part.builder));
            emitters.push(...emissiveQuadLights(built));
        } else {
            const url = await findModelUrl(part.asset, part.path);
            const model = await loadGltf(url, manifest.title, report);
            const holder = new Group();
            holder.name = `${part.asset}/${part.path}`;
            holder.add(model);
            applyModelTransform(holder, part.transform);
            content.add(holder);
        }
    }
    prepareMeshes(content);
    scene.add(content);
    content.updateMatrixWorld(true);

    const bounds = new Box3().setFromObject(content);
    const center = bounds.isEmpty() ? new Vector3() : bounds.getCenter(new Vector3());
    const diagonal = bounds.isEmpty() ? 20 : bounds.getSize(new Vector3()).length();

    // Directional lights get a shadow camera that fits a sphere around the model.
    const fitting = { center: [center.x, center.y, center.z] as [number, number, number], radius: diagonal / 2 };
    const directLights: Light[] = [];
    for (const manifestLight of manifest.lights) {
        const light = lightToThree(manifestLightToSceneLight(manifestLight), fitting);
        light.layers.enable(LAYER_DIRECT_LIGHTS);
        directLights.push(light);
        scene.add(light);
        const target = (light as unknown as { target?: Object3D }).target;
        if (target) scene.add(target);
    }
    if (emitters.length > 0) {
        RectAreaLightNode.setLTC(RectAreaLightTexturesLib.init());
        for (const emitter of emitters) {
            emitter.layers.enable(LAYER_DIRECT_LIGHTS);
            directLights.push(emitter);
            scene.add(emitter);
        }
    }

    return { manifest, scene, content, directLights, bounds, sky, diagonal, stats: computeSceneStats(content) };
}
