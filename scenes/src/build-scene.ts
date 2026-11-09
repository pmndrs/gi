/**
 * Turns a manifest's procedural parts into scene data. Browser-safe: no file
 * access, so both the exporter (Node) and the check page can use it.
 */

import type { SceneManifest } from './manifest';
import { buildProcedural } from './procedural';
import type { ProceduralScene, SceneTextureData } from './scene-data';

/** Concatenates scenes, shifting the mesh, material and texture indices of later ones. */
export function mergeScenes(scenes: readonly ProceduralScene[]): ProceduralScene {
    const meshes: ProceduralScene['meshes'][number][] = [];
    const materials: ProceduralScene['materials'][number][] = [];
    const instances: ProceduralScene['instances'][number][] = [];
    const lights: ProceduralScene['lights'][number][] = [];
    const textures: SceneTextureData[] = [];

    for (const scene of scenes) {
        const meshOffset = meshes.length;
        const materialOffset = materials.length;
        const textureOffset = textures.length;
        meshes.push(...scene.meshes);
        textures.push(...(scene.textures ?? []));
        for (const material of scene.materials) {
            materials.push(material.albedoMap ? { ...material, albedoMap: { ...material.albedoMap, texture: material.albedoMap.texture + textureOffset } } : material);
        }
        for (const instance of scene.instances) {
            instances.push({
                ...instance,
                mesh: instance.mesh + meshOffset,
                material: instance.material + materialOffset,
                ...(instance.materials ? { materials: instance.materials.map((material) => material + materialOffset) } : {}),
            });
        }
        lights.push(...scene.lights);
    }
    return { meshes, materials, instances, lights, ...(textures.length > 0 ? { textures } : {}) };
}

/** Builds every procedural part of a manifest, one scene per part, in order. */
export function buildProceduralParts(manifest: SceneManifest): ProceduralScene[] {
    return manifest.parts.flatMap((part) => (part.kind === 'procedural' ? [buildProcedural(part.builder, part.options)] : []));
}
