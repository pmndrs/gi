/**
 * Turns a manifest's procedural parts into scene data. Browser-safe: no file
 * access, so both the exporter (Node) and the check page can use it.
 */

import type { SceneManifest } from './manifest';
import { buildProcedural } from './procedural';
import type { ProceduralScene } from './scene-data';

/** Concatenates scenes, shifting the mesh and material indices of later ones. */
export function mergeScenes(scenes: readonly ProceduralScene[]): ProceduralScene {
    const meshes: ProceduralScene['meshes'][number][] = [];
    const materials: ProceduralScene['materials'][number][] = [];
    const instances: ProceduralScene['instances'][number][] = [];
    const lights: ProceduralScene['lights'][number][] = [];

    for (const scene of scenes) {
        const meshOffset = meshes.length;
        const materialOffset = materials.length;
        meshes.push(...scene.meshes);
        materials.push(...scene.materials);
        for (const instance of scene.instances) {
            instances.push({
                ...instance,
                mesh: instance.mesh + meshOffset,
                material: instance.material + materialOffset,
            });
        }
        lights.push(...scene.lights);
    }
    return { meshes, materials, instances, lights };
}

/** Builds every procedural part of a manifest, one scene per part, in order. */
export function buildProceduralParts(manifest: SceneManifest): ProceduralScene[] {
    return manifest.parts.flatMap((part) => (part.kind === 'procedural' ? [buildProcedural(part.builder, part.options)] : []));
}
