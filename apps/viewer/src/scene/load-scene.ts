/**
 * Builds a three.js scene from a manifest: every part (procedural, or a
 * downloaded glTF; Sponza is the base package plus add-ons, each its own
 * part in one shared coordinate frame; or a light set, `sponza-candles`'
 * candles), the manifest's direct lights, and the manifest's sky.
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
    type Texture,
} from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { RectAreaLightTexturesLib } from 'three/addons/lights/RectAreaLightTexturesLib.js';
import { RectAreaLightNode } from 'three/webgpu';
import { manifestLightToSceneLight, sceneSettingValues, type ManifestEnvironment, type SceneManifest } from '../../../../scenes/src/manifest';
import { buildProcedural } from '../../../../scenes/src/procedural';
import { computeSceneStats, type SceneStats } from '../../../../scenes/src/stats';
import { applyModelTransform, emissiveQuadLights, lightToThree, loadLightSet, proceduralSceneToThree, type LoadedLightSet } from '../../../../scenes/src/three-scene';
import { LAYER_DIRECT_LIGHTS, LAYER_VIEW_GEOMETRY } from './layers';
import { compressModelTextures, type TextureCompression, type TextureCompressionReport } from './compress-textures';
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
    /** The manifest's lights, the emissive-quad area lights and a light set's lights. */
    directLights: Light[];
    /**
     * Manifest lights at intensity 0, made but not added to the scene: off
     * until a motion plays them (Sponza's arcade lamp). Left out because an
     * off light still costs: three r187 has no intensity test before it
     * renders a shadowed light's shadow map (six passes for a point light),
     * and in GI one local light makes hit shading loop over local lights on
     * every hit. Add one to `scene` to light it.
     */
    offLights: Light[];
    /**
     * The manifest's own lights, in manifest order (`manifest.lights[i]` is
     * `manifestLights[i]`), on or off: what the viewer's light controls and
     * motions work on. Each is named by its manifest label.
     */
    manifestLights: Light[];
    /**
     * The viewer's light markers and gizmo (lighting/light-editor.ts): a scene
     * of their own, drawn over the picture after the main one, so GI's scene
     * sync (which walks `scene`) and the bakes never see them.
     */
    helpers: Scene;
    /** The sky dome of a procedural sky (scene/sky.ts), whose picture time of day replaces; null otherwise. */
    skyDome: Mesh | null;
    /** The scene's settings as loaded (`sceneSettingValues`): `sponza-candles`' candle count, say. */
    settings: Record<string, number>;
    /** Each `light-set` part as loaded: its copies and lights. */
    lightSets: LoadedLightSet[];
    bounds: Box3;
    /** The shared sky file as a three.js texture, for a procedural environment; null otherwise. */
    sky: DataTexture | null;
    /** Length of the bounds' diagonal in metres; sets camera speed and clipping. */
    diagonal: number;
    stats: SceneStats;
    /** What texture compression did, one report per model part (empty when nothing was asked for). */
    textureCompression: TextureCompressionReport[];
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
 * Returns the sky texture and its dome for a procedural sky, nulls otherwise.
 */
async function addSky(scene: Scene, manifest: SceneManifest): Promise<{ sky: DataTexture | null; dome: Mesh | null }> {
    const environment: ManifestEnvironment = manifest.environment;
    switch (environment.type) {
        case 'none':
            scene.background = NO_SKY.clone();
            return { sky: null, dome: null };
        case 'color': {
            const scale = environment.intensity ?? 1;
            scene.background = new Color().setRGB(environment.color[0] * scale, environment.color[1] * scale, environment.color[2] * scale);
            return { sky: null, dome: null };
        }
        case 'procedural': {
            scene.background = NO_SKY.clone();
            const sky = await loadSkyTexture(manifest.id);
            const dome = createSkyDome(sky);
            // Layer 0 for the probe bake; the viewing layer for the cameras.
            dome.layers.enable(LAYER_VIEW_GEOMETRY);
            scene.add(dome);
            return { sky, dome };
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
    const objects: Object3D[] = [];
    loaded.scene.traverse((object) => objects.push(object));
    loaded.helpers.traverse((object) => objects.push(object));
    for (const object of objects) {
        const mesh = object as Mesh;
        if (!mesh.isMesh) continue;
        mesh.geometry.dispose();
        for (const material of Array.isArray(mesh.material) ? mesh.material : [mesh.material]) {
            for (const value of Object.values(material)) {
                if ((value as { isTexture?: boolean } | null)?.isTexture) (value as { dispose(): void }).dispose();
            }
            material.dispose();
        }
    }
}

export interface LoadSceneOptions {
    /** Scene settings by name, as a link gives them (`?candles=4096`); the manifest's defaults otherwise. */
    readonly settings?: Readonly<Record<string, unknown>>;
    /**
     * The largest texture side, in pixels, a model's images keep; bigger ones are scaled down
     * at load (`limitTextureSize`). Unset keeps every image as the file has it.
     */
    readonly maxTextureSize?: number;
    /**
     * How model textures are kept on the GPU (scene/compress-textures.ts). Default `'uncompressed'`.
     * Compression runs on `device`, which must be three's.
     */
    readonly textureCompression?: TextureCompression;
    /** The device three draws with: needed when `textureCompression` compresses. */
    readonly device?: GPUDevice;
}

/**
 * Scales down every image in `root`'s materials whose larger side is over `maxSize`
 * (halving until it fits, so mipmaps stay powers of two where they were).
 *
 * Why: three keeps every texture uncompressed (RGBA8 with mipmaps, 4/3 × 4 bytes a pixel).
 * Bistro exterior's 224 images of 2048 × 2048 then take 5.1 GB of GPU memory, which with
 * DDGI's 0.6 GB is more than a 6 GB RTX 3060 Laptop has. The driver then moves memory between
 * the GPU and system memory every frame the GI update runs, and a frame takes about 0.5 s
 * instead of 0.05 s (T2.17, NOTES). At 1024 the images take a quarter of that.
 * Returns how many images were scaled down.
 */
export async function limitTextureSize(root: Object3D, maxSize: number): Promise<number> {
    const textures = new Set<Texture>();
    root.traverse((object) => {
        const mesh = object as Mesh;
        if (!mesh.isMesh) return;
        for (const material of Array.isArray(mesh.material) ? mesh.material : [mesh.material]) {
            for (const value of Object.values(material)) {
                if ((value as Texture | null)?.isTexture) textures.add(value as Texture);
            }
        }
    });
    // GLTFLoader shares one decoded image between textures that use the same file: resize it once.
    const resizedImages = new Map<ImageBitmap | HTMLImageElement, ImageBitmap>();
    for (const texture of textures) {
        const image = texture.image as unknown;
        // Only decoded images can be resized here; data textures and canvases are left as they are.
        if (!(image instanceof ImageBitmap || image instanceof HTMLImageElement)) continue;
        let resized = resizedImages.get(image);
        if (!resized) {
            let width = image.width;
            let height = image.height;
            while (Math.max(width, height) > maxSize) {
                width = Math.max(1, Math.floor(width / 2));
                height = Math.max(1, Math.floor(height / 2));
            }
            if (width === image.width && height === image.height) continue;
            // 'none' for alpha and colour space: the pixels stay the file's values (sRGB stays encoded,
            // alpha is not premultiplied), as GLTFLoader decoded them.
            resized = await createImageBitmap(image, { resizeWidth: width, resizeHeight: height, resizeQuality: 'high', premultiplyAlpha: 'none', colorSpaceConversion: 'none' });
            resizedImages.set(image, resized);
        }
        texture.image = resized;
        texture.needsUpdate = true;
    }
    // The full-size images are no longer used by any texture.
    for (const image of resizedImages.keys()) if (image instanceof ImageBitmap) image.close();
    return resizedImages.size;
}

export async function loadScene(manifest: SceneManifest, report: (progress: LoadProgress) => void, options: LoadSceneOptions = {}): Promise<LoadedScene> {
    const scene = new Scene();
    const { sky, dome: skyDome } = await addSky(scene, manifest);
    const settings = sceneSettingValues(manifest, options.settings);
    const lightSets: LoadedLightSet[] = [];
    const textureCompression: TextureCompressionReport[] = [];
    const compression = options.textureCompression ?? 'uncompressed';
    // Compression encodes on three's own device, so the compressed blocks never cross devices.
    let compressionDevice: GPUDevice | null = null;
    if (compression !== 'uncompressed') {
        if (!options.device) throw new Error(`loadScene: textureCompression '${compression}' needs three's device (options.device)`);
        compressionDevice = options.device;
    }

    const content = new Group();
    content.name = manifest.id;
    const emitters: RectAreaLight[] = [];
    const modelPartCount = manifest.parts.filter((part) => part.kind === 'model').length;
    let modelPartNumber = 0;
    for (const part of manifest.parts) {
        if (part.kind === 'procedural') {
            report({ stage: `Building ${manifest.title}`, fraction: null });
            const built = buildProcedural(part.builder, part.options);
            content.add(proceduralSceneToThree(built, part.builder));
            emitters.push(...emissiveQuadLights(built));
        } else if (part.kind === 'light-set') {
            report({ stage: `Loading ${settings[part.countSetting]} ${part.countSetting} (${part.asset})`, fraction: null });
            const loaded = await loadLightSet(part, settings[part.countSetting], findModelUrl, new GLTFLoader());
            lightSets.push(loaded);
            content.add(loaded.group);
        } else {
            // Scenes made of several downloads (Sponza and its add-ons) say which one is loading.
            modelPartNumber++;
            const label = modelPartCount > 1 ? `${manifest.title}, part ${modelPartNumber} of ${modelPartCount} (${part.asset})` : manifest.title;
            const url = await findModelUrl(part.asset, part.path);
            const model = await loadGltf(url, label, report);
            if (options.maxTextureSize !== undefined) await limitTextureSize(model, options.maxTextureSize);
            if (compressionDevice) {
                const compressing = (done: number, total: number) => report({ stage: `Compressing ${label}'s textures (${done} of ${total})`, fraction: 0.8 + 0.1 * (done / Math.max(1, total)) });
                textureCompression.push(await compressModelTextures(model, compressionDevice, compression, compressing));
            }
            const holder = new Group();
            holder.name = `${part.asset}/${part.path}`;
            holder.add(model);
            applyModelTransform(holder, part.transform);
            content.add(holder);
        }
    }
    prepareMeshes(content);
    // A light set's copies don't cast three's shadows (loadLightSet says why); prepareMeshes turned them on for everything.
    for (const set of lightSets) set.group.traverse((object) => (object.castShadow = false));
    scene.add(content);
    content.updateMatrixWorld(true);

    const bounds = new Box3().setFromObject(content);
    const center = bounds.isEmpty() ? new Vector3() : bounds.getCenter(new Vector3());
    const diagonal = bounds.isEmpty() ? 20 : bounds.getSize(new Vector3()).length();

    // Directional lights get a shadow camera that fits a sphere around the model.
    const fitting = { center: [center.x, center.y, center.z] as [number, number, number], radius: diagonal / 2 };
    const directLights: Light[] = [];
    const offLights: Light[] = [];
    const manifestLights: Light[] = [];
    for (const manifestLight of manifest.lights) {
        const light = lightToThree(manifestLightToSceneLight(manifestLight), fitting);
        light.layers.enable(LAYER_DIRECT_LIGHTS);
        manifestLights.push(light);
        if (manifestLight.intensity === 0) {
            offLights.push(light);
            continue;
        }
        directLights.push(light);
        scene.add(light);
        const target = (light as unknown as { target?: Object3D }).target;
        if (target) scene.add(target);
    }
    for (const set of lightSets) {
        for (const light of set.lights) {
            light.layers.enable(LAYER_DIRECT_LIGHTS);
            directLights.push(light);
            scene.add(light);
        }
    }
    if (emitters.length > 0) {
        RectAreaLightNode.setLTC(RectAreaLightTexturesLib.init());
        for (const emitter of emitters) {
            emitter.layers.enable(LAYER_DIRECT_LIGHTS);
            directLights.push(emitter);
            scene.add(emitter);
        }
    }

    const helpers = new Scene();
    helpers.name = `${manifest.id} viewer helpers`;
    return { manifest, scene, content, directLights, offLights, manifestLights, helpers, skyDome, settings, lightSets, bounds, sky, diagonal, stats: computeSceneStats(content), textureCompression };
}
