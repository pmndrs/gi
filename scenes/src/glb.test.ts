import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import type { Mesh, Object3D } from 'three';
import { describe, expect, it } from 'vitest';
import { buildProceduralParts, mergeScenes } from './build-scene';
import { quaternionFromLightDirection, readGlbChunks, writeGlb } from './glb';
import { loadManifests } from './export';
import { isProceduralOnly } from './manifest';
import { triangleCount } from './procedural/geometry';
import { computeSceneStats } from './stats';

async function parseWithThree(glb: Uint8Array): Promise<Object3D> {
    const copy = glb.buffer.slice(glb.byteOffset, glb.byteOffset + glb.byteLength) as ArrayBuffer;
    const gltf = await new GLTFLoader().parseAsync(copy, '');
    return gltf.scene;
}

describe('quaternionFromLightDirection', () => {
    function rotate(q: number[], v: number[]): number[] {
        // v' = v + 2w(q x v) + 2 q x (q x v)
        const [x, y, z, w] = q;
        const cross1 = [y * v[2] - z * v[1], z * v[0] - x * v[2], x * v[1] - y * v[0]];
        const cross2 = [y * cross1[2] - z * cross1[1], z * cross1[0] - x * cross1[2], x * cross1[1] - y * cross1[0]];
        return v.map((component, i) => component + 2 * w * cross1[i] + 2 * cross2[i]);
    }

    it('sends glTF light axis -Z onto the requested direction', () => {
        for (const direction of [[0, -1, 0], [1, 0, 0], [0, 0, 1], [0, 0, -1], [-0.35, -0.85, -0.4]] as const) {
            const length = Math.hypot(...direction);
            const rotated = rotate(quaternionFromLightDirection(direction), [0, 0, -1]);
            rotated.forEach((component, axis) => expect(component).toBeCloseTo(direction[axis] / length, 6));
        }
    });
});

describe('exported procedural scenes', async () => {
    const manifests = (await loadManifests()).filter(isProceduralOnly);

    it('covers the six procedural scenes', () => {
        expect(manifests.map((m) => m.id).sort()).toEqual([
            'cabinet-row',
            'closed-door',
            'cornell-box',
            'cornell-box-point',
            'furnace-room',
            'thin-wall',
            'two-room',
        ]);
    });

    for (const manifest of manifests) {
        describe(manifest.id, () => {
            const scene = mergeScenes(buildProceduralParts(manifest));
            const glb = writeGlb(scene, { name: manifest.title, extras: { manifestId: manifest.id } });
            const { json, binary } = readGlbChunks(glb);

            it('has a well-formed container: 4-byte aligned, buffer views inside the binary chunk', () => {
                expect(glb.byteLength % 4).toBe(0);
                expect(json.buffers[0].byteLength).toBe(binary.length);
                for (const view of json.bufferViews as Array<{ byteOffset: number; byteLength: number }>) {
                    expect(view.byteOffset % 4).toBe(0);
                    expect(view.byteOffset + view.byteLength).toBeLessThanOrEqual(binary.length);
                }
            });

            it('has accessor counts that match the source meshes', () => {
                const accessors = json.accessors as Array<{ count: number; type: string; min?: number[]; max?: number[] }>;
                const positions = accessors.filter((a) => a.type === 'VEC3' && a.min);
                expect(positions).toHaveLength(scene.meshes.length);
                positions.forEach((accessor, index) => {
                    expect(accessor.count).toBe(scene.meshes[index].positions.length / 3);
                    expect(accessor.min).toHaveLength(3);
                });
                expect(json.nodes.filter((node) => 'mesh' in node)).toHaveLength(scene.instances.length);
                expect(json.materials).toHaveLength(scene.materials.length);
                expect(json.extensions ? Object.keys(json.extensions) : []).toEqual(scene.lights.length > 0 ? ['KHR_lights_punctual'] : []);
            });

            it('loads in three.js with the same triangle, mesh and material counts as the data', async () => {
                const root = await parseWithThree(glb);
                const stats = computeSceneStats(root);
                expect(stats.triangles).toBe(triangleCount(scene));
                expect(stats.meshes).toBe(scene.instances.length);
                // A glTF material per scene material that is used by at least one instance.
                const usedMaterials = new Set(scene.instances.map((instance) => instance.material));
                expect(stats.materials).toBe(usedMaterials.size);
                expect(stats.textures).toBe(0);
            });

            it('keeps instance transforms and emission through the round trip', async () => {
                const root = await parseWithThree(glb);
                root.updateMatrixWorld(true);
                const meshes: Mesh[] = [];
                root.traverse((object) => {
                    if ((object as Mesh).isMesh) meshes.push(object as Mesh);
                });
                // Node order is preserved, so mesh i is instance i.
                meshes.forEach((mesh, index) => {
                    scene.instances[index].transform.forEach((value, element) => {
                        expect(mesh.matrixWorld.elements[element]).toBeCloseTo(value, 4);
                    });
                });
                const emissiveInstance = scene.instances.findIndex((i) => scene.materials[i.material].emissive.some((c) => c > 0));
                if (emissiveInstance >= 0) {
                    const expected = Math.max(...scene.materials[scene.instances[emissiveInstance].material].emissive);
                    const material = meshes[emissiveInstance].material as unknown as { emissive: { r: number; g: number; b: number }; emissiveIntensity: number };
                    const radiance = Math.max(material.emissive.r, material.emissive.g, material.emissive.b) * material.emissiveIntensity;
                    expect(radiance).toBeCloseTo(expected, 4);
                }
            });
        });
    }
});
