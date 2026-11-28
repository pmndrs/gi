/**
 * Shared by the `three-m4-*` pages (M4, T4.3): a procedural manifest scene
 * built as a three.js scene, mirrored into a platform by the three adapter's
 * own `SceneSync`, with DDGI on the manifest's volume. So what the pages
 * measure went through the adapter's material and texture conversions
 * (convert.ts), not through hand-made platform descriptors.
 *
 * No renderer is needed: `SceneSync` is the adapter's mirroring on its own,
 * and the frames call `platform.encode` as any host that drives the platform
 * itself does (docs/api/three.md, `SceneSync`).
 *
 * Not a page itself (no index.html).
 */

import { BufferAttribute, BufferGeometry, Mesh, MeshStandardMaterial, PointLight, Scene, type Material } from 'three';
import { GIPlatformImpl } from '../../../../packages/core/src/platform/gi-platform.ts';
import { DDGITechnique } from '../../../../packages/core/src/techniques/ddgi/ddgi-technique.ts';
import { SceneSync } from '../../../../packages/three/src/scene-sync.ts';
import { buildProceduralParts, mergeScenes } from '../../../../scenes/src/build-scene.ts';
import type { SceneManifest } from '../../../../scenes/src/manifest.ts';
import type { ProceduralScene, SceneMaterialData, SceneMeshData } from '../../../../scenes/src/scene-data.ts';
import { assert } from '../../client.ts';
import { frameInput, loadManifest } from '../ddgi-shared/harness.ts';

/** A procedural mesh as a three geometry: positions, normals, uv and the index. */
export function geometryOf(mesh: SceneMeshData): BufferGeometry {
    const geometry = new BufferGeometry();
    geometry.setAttribute('position', new BufferAttribute(mesh.positions, 3));
    geometry.setAttribute('normal', new BufferAttribute(mesh.normals, 3));
    geometry.setAttribute('uv', new BufferAttribute(mesh.uvs, 2));
    geometry.setIndex(new BufferAttribute(mesh.indices, 1));
    geometry.name = mesh.name;
    return geometry;
}

/**
 * A procedural material as the three material a host would make: a rough,
 * non-metal Standard material with the albedo as its colour and the emission
 * as its emissive colour (linear, three's working space).
 */
export function materialOf(data: SceneMaterialData): MeshStandardMaterial {
    const material = new MeshStandardMaterial({ name: data.name, roughness: 1, metalness: 0 });
    material.color.setRGB(data.albedo[0], data.albedo[1], data.albedo[2]);
    material.emissive.setRGB(data.emissive[0], data.emissive[1], data.emissive[2]);
    if (!data.doubleSided) material.userData.giSingleSided = true;
    return material;
}

export interface ThreeProceduralScene {
    readonly manifest: SceneManifest;
    readonly data: ProceduralScene;
    readonly scene: Scene;
    /** The three material of each procedural material, by index. */
    readonly materials: readonly MeshStandardMaterial[];
}

/**
 * The manifest's procedural scene as a three scene: a mesh per instance
 * (instances named in `skip` left out), sharing geometries and materials as
 * the data does, and the manifest's point lights as three point lights
 * (no range window, shadows on), as the GI pages light them.
 */
export function threeProceduralScene(manifestId: string, skip: readonly string[] = []): ThreeProceduralScene {
    const manifest = loadManifest(manifestId);
    const data = mergeScenes(buildProceduralParts(manifest));
    const scene = new Scene();
    const geometries = data.meshes.map(geometryOf);
    const materials = data.materials.map(materialOf);
    for (const instance of data.instances) {
        if (skip.includes(instance.name)) continue;
        const mesh = new Mesh(geometries[instance.mesh], materials[instance.material] as Material);
        mesh.name = instance.name;
        mesh.matrixAutoUpdate = false;
        mesh.matrix.fromArray(instance.transform);
        scene.add(mesh);
    }
    for (const light of manifest.lights) {
        assert(light.type === 'point', `three-m4 pages light point lights only; ${manifestId} has a ${light.type} light`);
        const lamp = new PointLight(undefined, light.intensity, 0, 2);
        lamp.color.setRGB(light.color[0], light.color[1], light.color[2]);
        lamp.position.set(...light.position);
        lamp.castShadow = true;
        scene.add(lamp);
    }
    scene.updateMatrixWorld(true);
    return { manifest, data, scene, materials };
}

export interface ThreeGiRun {
    readonly platform: GIPlatformImpl;
    readonly ddgi: DDGITechnique;
    readonly sync: SceneSync;
    /** One frame: the adapter sends what changed, then the platform encodes with `units` probes. */
    frame(units: number): void;
    /** `count` frames, waiting for the GPU every few. */
    frames(count: number, units: number): Promise<void>;
    dispose(): void;
}

/**
 * A platform with DDGI on the manifest's volume, fed from `scene` by the
 * adapter's SceneSync, prepared, its trees built.
 */
export async function startThreeGi(device: GPUDevice, scene: Scene, manifest: SceneManifest, label: string, warn: (message: string) => void): Promise<ThreeGiRun> {
    const platform = new GIPlatformImpl(device, { label });
    const ddgi = platform.addTechnique(new DDGITechnique({ warmUpPasses: 0, volume: { center: manifest.volume.center, size: manifest.volume.size, spacing: manifest.volume.probeSpacing } }));
    platform.setOutput(ddgi);
    await platform.prepare();
    const sync = new SceneSync(platform.scene, { device, warn });
    const frame = (units: number): void => {
        sync.update(scene);
        const encoder = device.createCommandEncoder({ label: `${label} frame` });
        platform.encode(encoder, frameInput(units));
        device.queue.submit([encoder.finish()]);
    };
    frame(0);
    await platform.rays.whenBuilt();
    frame(0);
    assert(!platform.rays.building, 'the platform still reports trees building');
    return {
        platform,
        ddgi,
        sync,
        frame,
        frames: async (count, units) => {
            for (let index = 0; index < count; index++) {
                frame(units);
                if (index % 8 === 7) await device.queue.onSubmittedWorkDone();
            }
            await device.queue.onSubmittedWorkDone();
        },
        dispose: () => {
            sync.dispose();
            platform.dispose();
        },
    };
}
