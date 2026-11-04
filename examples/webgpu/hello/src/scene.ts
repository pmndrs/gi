/**
 * The test scenes for the hello example: a manifest's procedural parts built
 * by the repo's engine-free builders (`scenes/src`), described to the GI
 * platform through its scene API, and kept as plain data for the renderer.
 *
 * Only type imports from the library here, so GPU test pages can reuse this
 * file with the core loaded straight from source.
 */

import type { EnvironmentDescriptor, LightDescriptor, SceneApi } from '@pmndrs/ddgi/core';
import { buildProceduralParts, mergeScenes } from '../../../../scenes/src/build-scene.ts';
import { isProceduralOnly, type CameraBookmark, type ManifestLight, type SceneManifest } from '../../../../scenes/src/manifest.ts';
import type { ProceduralScene } from '../../../../scenes/src/scene-data.ts';
import { platformSkyEnvironment } from '../../../../scenes/src/sky/sky-image.ts';

export type { CameraBookmark, SceneManifest, ProceduralScene };

const manifests = import.meta.glob('../../../../scenes/manifests/*.json', { eager: true, import: 'default' }) as Record<string, SceneManifest>;

/** The point light the hello renderer lights directly (with a shadow map). */
export interface RendererPointLight {
    readonly position: readonly [number, number, number];
    /** Linear colour times intensity (candela). */
    readonly intensity: readonly [number, number, number];
    /** Metres; 0 means no range window (three's `distance = 0`). */
    readonly range: number;
}

/**
 * Why a scene can't run here, or null if it can. The hello renderer draws
 * procedural geometry with constant-colour materials and lights at most one
 * point light itself (directional and spot lights would need their own
 * shadow maps; the viewer has them).
 */
export function unsupportedReason(manifest: SceneManifest): string | null {
    if (!isProceduralOnly(manifest)) return 'it loads model files; the hello example only builds procedural scenes';
    const pointLights = manifest.lights.filter((light) => light.type === 'point').length;
    if (pointLights !== manifest.lights.length) return 'it has directional or spot lights; the hello renderer lights point lights only';
    if (pointLights > 1) return 'it has more than one point light; the hello renderer shadows one';
    return null;
}

export function loadManifest(id: string): SceneManifest {
    const manifest = Object.entries(manifests).find(([path]) => path.endsWith(`/${id}.json`))?.[1];
    if (!manifest) throw new Error(`No scene called "${id}". Try one of: ${supportedSceneIds().join(', ')}.`);
    const reason = unsupportedReason(manifest);
    if (reason) throw new Error(`The hello example can't show "${id}": ${reason}. Try one of: ${supportedSceneIds().join(', ')}.`);
    return manifest;
}

export function supportedSceneIds(): string[] {
    return Object.values(manifests)
        .filter((manifest) => unsupportedReason(manifest) === null)
        .map((manifest) => manifest.id)
        .sort();
}

export interface SceneOptions {
    /**
     * Light point lights with no range window (`range: 0`), as Cycles does.
     * The engine-free test view uses this for its reference comparisons.
     */
    readonly ignoreLightRange?: boolean;
}

function toPlatformLight(light: ManifestLight, options: SceneOptions): LightDescriptor {
    if (light.type !== 'point') throw new Error(`unsupported light type ${light.type}`);
    return {
        type: 'point',
        color: light.color,
        intensity: light.intensity,
        castsShadows: light.castsShadows ?? true,
        label: light.label,
        position: light.position,
        range: options.ignoreLightRange ? 0 : light.range,
    };
}

function toPlatformEnvironment(manifest: SceneManifest): EnvironmentDescriptor {
    const environment = manifest.environment;
    if (environment.type === 'none') return { type: 'none' };
    if (environment.type === 'color') return { type: 'color', color: environment.color, intensity: environment.intensity };
    // The calibrated sky, the same file the viewer and Cycles read.
    return platformSkyEnvironment(manifest);
}

export interface DescribedScene {
    readonly manifest: SceneManifest;
    /** Meshes, materials and instances, for the renderer. */
    readonly data: ProceduralScene;
    readonly pointLight: RendererPointLight | null;
    /**
     * Irradiance a renderer adds where GI coverage is low: a uniform colour
     * environment of radiance L gives E = π L. Zero for no environment, and
     * for a sky (the hello renderer has no sky lighting of its own).
     */
    readonly fallbackIrradiance: readonly [number, number, number];
}

/** Describes a manifest scene to the platform: meshes, materials, instances, lights and environment. */
export function describeScene(scene: SceneApi, manifest: SceneManifest, options: SceneOptions = {}): DescribedScene {
    const data = mergeScenes(buildProceduralParts(manifest));
    const meshes = data.meshes.map((mesh) => scene.createMesh({ positions: mesh.positions, indices: mesh.indices, normals: mesh.normals, uvs: mesh.uvs, label: mesh.name }));
    const materials = data.materials.map((material) =>
        scene.createMaterial({ albedo: material.albedo, emissive: material.emissive, doubleSided: material.doubleSided, label: material.name }),
    );
    for (const instance of data.instances) {
        scene.createInstance({ mesh: meshes[instance.mesh], materials: [materials[instance.material]], transform: instance.transform, label: instance.name });
    }
    for (const light of manifest.lights) scene.createLight(toPlatformLight(light, options));
    scene.setEnvironment(toPlatformEnvironment(manifest));

    const light = manifest.lights.find((candidate) => candidate.type === 'point');
    const pointLight: RendererPointLight | null =
        light && light.type === 'point'
            ? {
                  position: light.position,
                  intensity: [light.color[0] * light.intensity, light.color[1] * light.intensity, light.color[2] * light.intensity],
                  range: options.ignoreLightRange ? 0 : light.range,
              }
            : null;
    const environment = manifest.environment;
    // A uniform environment of radiance L gives irradiance E = pi L on any surface.
    const environmentRadianceScale = environment.type === 'color' ? (environment.intensity ?? 1) : 0;
    const fallbackIrradiance: [number, number, number] =
        environment.type === 'color'
            ? [Math.PI * environment.color[0] * environmentRadianceScale, Math.PI * environment.color[1] * environmentRadianceScale, Math.PI * environment.color[2] * environmentRadianceScale]
            : [0, 0, 0];
    return { manifest, data, pointLight, fallbackIrradiance };
}
