/**
 * The scene check page: loads one manifest scene in three.js (WebGPU), lights
 * it with the manifest's direct lights only, and reports what it found.
 *
 *   ?scene=<manifest id>&camera=<n or name>
 *
 * `camera` is 1-based (bookmark 1 is the first in the manifest) or a bookmark
 * name. It defaults to bookmark 1. `&cam=px,py,pz,tx,ty,tz,fov` overrides it
 * with a free camera, for finding good bookmarks.
 *
 * A 1 m reference cube is added so scale can be judged by eye. Neither it nor
 * the lights count in the stats.
 *
 * `window.__sceneCheck` is what automated runs read: `ready` flips to true once
 * a frame has been drawn and the GPU finished it (also when loading failed, so
 * a runner never waits forever), `errors` lists every problem, and `stats`
 * counts the scene's triangles, meshes, materials and textures.
 */

import {
    Box3,
    BoxGeometry,
    Color,
    Group,
    Mesh,
    MeshStandardMaterial,
    PerspectiveCamera,
    type RectAreaLight,
    Scene,
    ACESFilmicToneMapping,
    Vector3,
    type Object3D,
} from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { RectAreaLightTexturesLib } from 'three/addons/lights/RectAreaLightTexturesLib.js';
import { RectAreaLightNode, WebGPURenderer } from 'three/webgpu';
import {
    manifestLightToSceneLight,
    sceneSettingValues,
    validateManifest,
    type CameraBookmark,
    type ManifestEnvironment,
    type SceneManifest,
} from './manifest';
import { buildProcedural, proceduralBuilders } from './procedural';
import { computeSceneStats, type SceneStats } from './stats';
import { applyModelTransform, emissiveQuadLights, lightToThree, loadLightSet, proceduralSceneToThree } from './three-scene';

/** Absolute paths of the downloads folders, injected by `scenes/vite.config.ts`. */
declare const __DOWNLOADS_DIRS__: string[];

export interface SceneCheckState {
    /** True once a frame has been drawn (or loading failed). */
    ready: boolean;
    /** Every problem seen: load failures, console errors, uncaptured GPU errors. */
    errors: string[];
    stats: SceneStats;
    /** What lights the picture, in words, for example "1 point light". The page adds nothing else: no fill, no ambient. */
    lighting: string;
    scene: string | null;
    camera: string | null;
}

declare global {
    interface Window {
        __sceneCheck: SceneCheckState;
    }
}

const manifestModules = import.meta.glob('../manifests/*.json', { eager: true, import: 'default' }) as Record<string, unknown>;

function loadManifests(): SceneManifest[] {
    const manifests: SceneManifest[] = [];
    for (const [file, value] of Object.entries(manifestModules)) {
        const result = validateManifest(value, { knownBuilders: Object.keys(proceduralBuilders) });
        if (!result.ok) throw new Error(`${file} is not a valid manifest: ${result.errors.join('; ')}`);
        manifests.push(value as SceneManifest);
    }
    return manifests.sort((a, b) => a.id.localeCompare(b.id));
}

function pickCamera(manifest: SceneManifest, request: string | null, state: SceneCheckState, free: string | null): CameraBookmark {
    if (free !== null) {
        // Exploring: `&cam=px,py,pz,tx,ty,tz,fov` replaces the bookmarks with a free camera.
        const [px, py, pz, tx, ty, tz, fov] = free.split(',').map(Number);
        if ([px, py, pz, tx, ty, tz, fov].every(Number.isFinite)) {
            return { name: 'free', position: [px, py, pz], target: [tx, ty, tz], fovYDeg: fov };
        }
        state.errors.push(`cam must be px,py,pz,tx,ty,tz,fov, got "${free}"`);
    }
    if (request === null) return manifest.cameras[0];
    const byName = manifest.cameras.find((camera) => camera.name === request);
    if (byName) return byName;
    const oneBased = Number(request);
    if (Number.isInteger(oneBased) && oneBased >= 1 && oneBased <= manifest.cameras.length) {
        return manifest.cameras[oneBased - 1];
    }
    state.errors.push(`camera "${request}" does not exist; use 1 to ${manifest.cameras.length} or one of: ${manifest.cameras.map((c) => c.name).join(', ')}`);
    return manifest.cameras[0];
}

/** Sky colour for the background only. Nothing here lights the scene. */
function backgroundFor(environment: ManifestEnvironment): Color {
    switch (environment.type) {
        case 'color': {
            const scale = environment.intensity ?? 1;
            return new Color().setRGB(environment.color[0] * scale, environment.color[1] * scale, environment.color[2] * scale);
        }
        case 'procedural':
            return new Color().setRGB(0.25, 0.4, 0.7);
        case 'none':
            return new Color().setRGB(0.01, 0.01, 0.012);
    }
}

/**
 * Finds a model file in the downloads folders (first match wins). Vite serves
 * any file under `server.fs.allow` at /@fs/<absolute path>, but answers a
 * missing file with its HTML page, which would otherwise fail later as a
 * confusing JSON error, so each candidate is checked by its content type.
 */
async function findModelUrl(asset: string, path: string): Promise<string> {
    for (const directory of __DOWNLOADS_DIRS__) {
        const url = `/@fs/${directory.replaceAll('\\', '/')}/${asset}/${path}`;
        const response = await fetch(url, { method: 'HEAD' });
        const isHtml = (response.headers.get('content-type') ?? '').includes('text/html');
        if (response.ok && !isHtml) return url;
    }
    throw new Error(`model file ${asset}/${path} not found under assets/downloads; run "pnpm assets:fetch --only ${asset}"`);
}

async function loadParts(manifest: SceneManifest, loader: GLTFLoader): Promise<{ content: Group; emitters: RectAreaLight[] }> {
    const content = new Group();
    content.name = manifest.id;
    const emitters: RectAreaLight[] = [];
    for (const part of manifest.parts) {
        if (part.kind === 'procedural') {
            const built = buildProcedural(part.builder, part.options);
            content.add(proceduralSceneToThree(built, part.builder));
            emitters.push(...emissiveQuadLights(built));
        } else if (part.kind === 'light-set') {
            // The candles, at the scene's default count; their lights go with them (shadowless in three, loadLightSet says why).
            const set = await loadLightSet(part, sceneSettingValues(manifest)[part.countSetting], findModelUrl, loader);
            set.group.add(...set.lights);
            content.add(set.group);
        } else {
            const gltf = await loader.loadAsync(await findModelUrl(part.asset, part.path));
            // GLTFLoader leaves shadows off; without these the lights would shine through everything.
            gltf.scene.traverse((object) => {
                const mesh = object as Mesh;
                if (mesh.isMesh) {
                    mesh.castShadow = true;
                    mesh.receiveShadow = true;
                }
            });
            const holder = new Group();
            holder.name = `${part.asset}/${part.path}`;
            holder.add(gltf.scene);
            applyModelTransform(holder, part.transform);
            content.add(holder);
        }
    }
    return { content, emitters };
}

/** Plain words for what lights the picture, shown on the page and reported in `window.__sceneCheck`. */
function describeLighting(manifest: SceneManifest, emissiveQuadCount: number): string {
    const parts: string[] = [];
    const countByType = new Map<string, number>();
    for (const light of manifest.lights) if (light.intensity > 0) countByType.set(light.type, (countByType.get(light.type) ?? 0) + 1);
    for (const [type, count] of countByType) parts.push(`${count} ${type} light${count > 1 ? 's' : ''} (direct, with shadows)`);
    if (emissiveQuadCount > 0) parts.push(`${emissiveQuadCount} emissive quad${emissiveQuadCount > 1 ? 's' : ''} as area light${emissiveQuadCount > 1 ? 's' : ''} (direct, no shadows)`);
    if (parts.length === 0) return 'none: only surface emission shows (nothing is lit by anything)';
    return `${parts.join(' + ')}; no bounce light, no fill, no ambient`;
}

/** A sphere around everything, for placing the directional light's shadow camera. */
function boundingSphere(object: Object3D): { center: [number, number, number]; radius: number } {
    object.updateMatrixWorld(true);
    const box = new Box3().setFromObject(object);
    if (box.isEmpty()) return { center: [0, 0, 0], radius: 10 };
    const center = box.getCenter(new Vector3());
    return { center: [center.x, center.y, center.z], radius: box.getSize(new Vector3()).length() / 2 };
}

function describe(state: SceneCheckState): string {
    const { stats } = state;
    const lines = [
        `scene: ${state.scene ?? '(none)'}   camera: ${state.camera ?? '-'}`,
        `triangles ${stats.triangles}   meshes ${stats.meshes}   materials ${stats.materials}   textures ${stats.textures}`,
        `lighting: ${state.lighting}`,
        state.ready ? (state.errors.length === 0 ? 'ready, no errors' : `ready, ${state.errors.length} error(s)`) : 'loading...',
        ...state.errors.map((error) => `ERROR: ${error}`),
    ];
    return lines.join('\n');
}

export async function runCheckPage(canvas: HTMLCanvasElement, status: HTMLElement): Promise<void> {
    const state: SceneCheckState = {
        ready: false,
        errors: [],
        stats: { triangles: 0, meshes: 0, materials: 0, textures: 0 },
        lighting: 'not set up yet',
        scene: null,
        camera: null,
    };
    window.__sceneCheck = state;
    const refresh = () => {
        status.textContent = describe(state);
    };

    // Collect everything that goes wrong, wherever it is reported.
    window.addEventListener('error', (event) => state.errors.push(`page error: ${event.message}`));
    window.addEventListener('unhandledrejection', (event) => state.errors.push(`unhandled rejection: ${String(event.reason)}`));
    const originalConsoleError = console.error.bind(console);
    console.error = (...args: unknown[]) => {
        state.errors.push(`console.error: ${args.map(String).join(' ')}`);
        originalConsoleError(...args);
    };

    try {
        const params = new URLSearchParams(window.location.search);
        const manifests = loadManifests();
        const sceneId = params.get('scene');
        const manifest = manifests.find((candidate) => candidate.id === sceneId);
        if (!manifest) {
            throw new Error(`scene "${sceneId ?? ''}" not found. Use ?scene=<id>; ids: ${manifests.map((m) => m.id).join(', ')}`);
        }
        state.scene = manifest.id;
        const bookmark = pickCamera(manifest, params.get('camera'), state, params.get('cam'));
        state.camera = bookmark.name;
        if (manifest.status === 'awaiting-asset') {
            state.errors.push('this scene is a placeholder (status "awaiting-asset"); its cameras are rough guesses');
        }
        refresh();

        if (!navigator.gpu) throw new Error('WebGPU is not available in this browser');
        const renderer = new WebGPURenderer({ canvas, antialias: true });
        renderer.setPixelRatio(1);
        renderer.setSize(window.innerWidth, window.innerHeight, false);
        renderer.toneMapping = ACESFilmicToneMapping;
        renderer.toneMappingExposure = 1;
        renderer.shadowMap.enabled = true;
        await renderer.init();

        // three creates the device here, so listen on it without replacing three's own handler.
        const device = (renderer.backend as unknown as { device?: GPUDevice }).device;
        device?.addEventListener('uncapturederror', (event) => {
            state.errors.push(`GPU error: ${(event as GPUUncapturedErrorEvent).error.message}`);
        });

        const scene = new Scene();
        scene.background = backgroundFor(manifest.environment);

        // Lights come from the manifest (procedural builders report the same ones,
        // and a unit test keeps them equal). Emissive quads in procedural parts
        // also light the scene, as area lights. Nothing else is added: no fill,
        // no ambient. What the picture shows is exactly what `lighting` says.
        const { content, emitters } = await loadParts(manifest, new GLTFLoader());
        scene.add(content);
        state.stats = computeSceneStats(content);

        // A light at intensity 0 is off until a motion plays it (Sponza's arcade lamp): left out, as the viewer leaves it out.
        const manifestLights = manifest.lights.filter((light) => light.intensity > 0).map(manifestLightToSceneLight);
        const fitting = boundingSphere(content);
        for (const light of manifestLights) {
            const threeLight = lightToThree(light, fitting);
            scene.add(threeLight);
            if ('target' in threeLight) scene.add((threeLight as unknown as { target: Object3D }).target);
        }
        if (emitters.length > 0) {
            RectAreaLightNode.setLTC(RectAreaLightTexturesLib.init());
            for (const emitter of emitters) scene.add(emitter);
        }
        state.lighting = describeLighting(manifest, emitters.length);

        // The 1 m reference cube, standing on its own position from the manifest.
        const cubePosition = manifest.referenceCube?.position ?? bookmark.target;
        const cube = new Mesh(
            new BoxGeometry(1, 1, 1),
            new MeshStandardMaterial({ name: 'reference-cube-1m', color: new Color().setRGB(0.35, 0.5, 0.8), roughness: 1, metalness: 0 }),
        );
        cube.name = 'reference-cube-1m';
        cube.position.set(...cubePosition);
        cube.castShadow = true;
        cube.receiveShadow = true;
        scene.add(cube);

        const camera = new PerspectiveCamera(bookmark.fovYDeg, window.innerWidth / window.innerHeight, 0.05, 1000);
        camera.position.set(...bookmark.position);
        camera.lookAt(...bookmark.target);

        // Compile first, then two frames: the second is the clean one.
        await renderer.compileAsync(scene, camera);
        renderer.render(scene, camera);
        renderer.render(scene, camera);
        await device?.queue.onSubmittedWorkDone();

        window.addEventListener('resize', () => {
            renderer.setSize(window.innerWidth, window.innerHeight, false);
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.render(scene, camera);
        });
    } catch (error) {
        state.errors.push(error instanceof Error ? error.message : String(error));
    }
    state.ready = true;
    refresh();
}
