/**
 * Finds a benchmark scene's geometry: reads the scene manifest, downloads the
 * model file from `assets/downloads` (or the older `assets/downloads.bak`),
 * and returns meshes, instances, the probe volume and the sun direction.
 *
 * Browser only: it uses `fetch` against the test site's dev server, which can
 * serve any file inside the repo at `/@fs/<absolute path>`.
 */

import type { ModelPart, SceneManifest } from '../../../../scenes/src/manifest.ts';
import repositoryPackageUrl from '../../../../package.json?url';
import { BENCH_SCENE_SUBSETS, FALLBACK_TOWARD_SUN } from './config.ts';
import { fetchGltfGeometry, modelTransformMatrix, type BenchInstance, type BenchMeshGeometry } from './gltf-geometry.ts';
import { normalizeVector, worldBounds, type Box, type Vec3 } from './rays.ts';

/** Folders searched for downloaded models, in order. The first one that has the file wins. */
const DOWNLOAD_FOLDERS = ['assets/downloads', 'assets/downloads.bak'];

/** vite answers `?url` for a file outside its root with `/@fs/<absolute path>`, which tells us where the repo is. */
const REPOSITORY_URL = repositoryPackageUrl.replace(/\/package\.json$/, '');

const manifestModules = import.meta.glob('../../../../scenes/manifests/*.json', { eager: true, import: 'default' }) as Record<
    string,
    SceneManifest
>;

export function findManifest(id: string): SceneManifest {
    const manifest = Object.values(manifestModules).find((candidate) => candidate.id === id);
    if (!manifest) {
        const known = Object.values(manifestModules)
            .map((candidate) => candidate.id)
            .sort()
            .join(', ');
        throw new Error(`no scene manifest with id "${id}" (known: ${known})`);
    }
    return manifest;
}

/**
 * Which model files exist, known without asking the server. Probing with
 * `fetch` would log a 404 for every folder that lacks the file, and the test
 * runner fails a page on any logged error. The glob is lazy and never loads
 * the files; only its keys (paths relative to this file) are used.
 */
const availableModelFiles = Object.keys(
    import.meta.glob(['../../../../assets/downloads/**/*.{glb,gltf}', '../../../../assets/downloads.bak/**/*.{glb,gltf}']),
).map((key) => key.replace(/^(\.\.\/)+/, ''));

function findModelUrl(asset: string, path: string): string {
    const tried = DOWNLOAD_FOLDERS.map((folder) => `${folder}/${asset}/${path}`);
    const found = tried.find((candidate) => availableModelFiles.includes(candidate));
    if (!found) {
        throw new Error(`model file not found (looked in ${tried.join(', ')}). Run "pnpm assets:fetch --only ${asset}" first.`);
    }
    return `${REPOSITORY_URL}/${found}`;
}

export interface BenchScene {
    id: string;
    meshes: BenchMeshGeometry[];
    /** The tracer's instance index is the position in this array. */
    instances: BenchInstance[];
    /** Triangles stored once per mesh. */
    uniqueTriangles: number;
    /** Triangles counting every instance. */
    instancedTriangles: number;
    /** Box around all geometry. */
    geometryBounds: Box;
    /** The box the probe grid fills. */
    probeBox: Box;
    probeBoxSource: 'manifest' | 'scene';
    /** Unit vector from a surface toward the sun. */
    towardSun: Vec3;
    sunSource: 'manifest' | 'fallback';
    skippedPrimitives: number;
    /** Where the model came from (for the record). */
    modelFiles: string[];
    loadMs: number;
}

function toSun(manifest: SceneManifest): { towardSun: Vec3; source: BenchScene['sunSource'] } {
    const sun = manifest.lights.find((light) => light.type === 'directional');
    if (sun?.type === 'directional') {
        const travel = normalizeVector(sun.direction);
        return { towardSun: [-travel[0], -travel[1], -travel[2]], source: 'manifest' };
    }
    return { towardSun: normalizeVector(FALLBACK_TOWARD_SUN), source: 'fallback' };
}

export function volumeBox(volume: SceneManifest['volume']): Box {
    return {
        min: [0, 1, 2].map((axis) => volume.center[axis] - volume.size[axis] / 2) as Vec3,
        max: [0, 1, 2].map((axis) => volume.center[axis] + volume.size[axis] / 2) as Vec3,
    };
}

/**
 * Downloads and flattens a manifest scene, or a subset of one
 * (`BENCH_SCENE_SUBSETS`: only the listed assets, with the manifest's volume
 * and sun). `volume` picks which box the probes fill.
 */
export async function loadBenchScene(id: string, volume: 'manifest' | 'scene'): Promise<BenchScene> {
    const start = performance.now();
    const subset = BENCH_SCENE_SUBSETS[id];
    const manifest = findManifest(subset?.manifest ?? id);
    const meshes: BenchMeshGeometry[] = [];
    const instances: BenchInstance[] = [];
    const modelFiles: string[] = [];
    let skippedPrimitives = 0;

    const parts = subset ? manifest.parts.filter((part) => part.kind === 'model' && subset.assets.includes(part.asset)) : manifest.parts;
    if (subset && parts.length !== subset.assets.length) {
        throw new Error(`scene "${id}" needs the parts ${subset.assets.join(', ')} of "${subset.manifest}", and the manifest doesn't have them all`);
    }
    for (const part of parts) {
        if (part.kind !== 'model') throw new Error(`scene "${id}" has a procedural part; the benchmark only loads model files`);
        const modelPart: ModelPart = part;
        const url = findModelUrl(modelPart.asset, modelPart.path);
        modelFiles.push(url.slice(REPOSITORY_URL.length + 1));
        const geometry = await fetchGltfGeometry(url, modelTransformMatrix(modelPart.transform));
        const meshOffset = meshes.length;
        meshes.push(...geometry.meshes);
        for (const instance of geometry.instances) {
            instances.push({ meshIndex: instance.meshIndex + meshOffset, objectToWorld: instance.objectToWorld });
        }
        skippedPrimitives += geometry.skippedPrimitives;
    }
    if (instances.length === 0) throw new Error(`scene "${id}" has no triangles`);

    const geometryBounds = worldBounds(meshes, instances);
    const uniqueTriangles = meshes.reduce((total, mesh) => total + mesh.indices.length / 3, 0);
    const instancedTriangles = instances.reduce((total, instance) => total + meshes[instance.meshIndex].indices.length / 3, 0);
    const sun = toSun(manifest);
    return {
        id,
        meshes,
        instances,
        uniqueTriangles,
        instancedTriangles,
        geometryBounds,
        probeBox: volume === 'manifest' ? volumeBox(manifest.volume) : geometryBounds,
        probeBoxSource: volume,
        towardSun: sun.towardSun,
        sunSource: sun.source,
        skippedPrimitives,
        modelFiles,
        loadMs: performance.now() - start,
    };
}
