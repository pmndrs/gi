import { inflateSync } from 'node:zlib';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import type { Mesh, Object3D } from 'three';
import { describe, expect, it } from 'vitest';
import { buildProceduralParts, mergeScenes } from './build-scene';
import { quaternionFromLightDirection, readGlbChunks, writeGlb } from './glb';
import { loadManifests } from './export';
import { isProceduralOnly } from './manifest';
import { crc32 } from './png';
import { triangleCount } from './procedural/geometry';
import { buildAlphaChecker, buildGlassWall } from './procedural/see-through-walls';
import { buildTexturedFurnace } from './procedural/textured-furnace';
import { buildUvBoard, UV_BOARD_PANELS } from './procedural/uv-board';
import type { ProceduralScene } from './scene-data';
import { computeSceneStats } from './stats';

/**
 * The scene without its textures. three's GLTFLoader can't decode images in
 * Node (it needs `self` and an image decoder), so the round trip through
 * three runs on this copy; the textures themselves are checked from the
 * file's JSON and PNG bytes below.
 */
function withoutTextures(scene: ProceduralScene): ProceduralScene {
    return { ...scene, textures: undefined, materials: scene.materials.map((material) => ({ ...material, albedoMap: undefined })) };
}

/** three makes one Mesh per glTF primitive: one per instance, or one per group of a grouped mesh. */
function threeMeshesPerInstance(scene: ProceduralScene): number[] {
    return scene.instances.map((instance) => scene.meshes[instance.mesh].groups?.length ?? 1);
}

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

    it('covers the procedural scenes', () => {
        expect(manifests.map((m) => m.id).sort()).toEqual([
            'alpha-checker',
            'cabinet-row',
            'closed-door',
            'cornell-box',
            'cornell-box-point',
            'dark-room',
            'furnace-room',
            'glass-wall',
            'open-ground',
            'textured-furnace',
            'thin-wall',
            'two-room',
            'uv-board',
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

            const untexturedGlb = writeGlb(withoutTextures(scene), { name: manifest.title });

            it('loads in three.js with the same triangle, mesh and material counts as the data', async () => {
                const root = await parseWithThree(untexturedGlb);
                const stats = computeSceneStats(root);
                expect(stats.triangles).toBe(triangleCount(scene));
                expect(stats.meshes).toBe(threeMeshesPerInstance(scene).reduce((total, count) => total + count, 0));
                // A glTF material per scene material that is used by at least one instance.
                const usedMaterials = new Set(scene.instances.flatMap((instance) => instance.materials ?? [instance.material]));
                expect(stats.materials).toBe(usedMaterials.size);
                expect(stats.textures).toBe(0);
            });

            it('keeps instance transforms and emission through the round trip', async () => {
                const root = await parseWithThree(untexturedGlb);
                root.updateMatrixWorld(true);
                const meshes: Mesh[] = [];
                root.traverse((object) => {
                    if ((object as Mesh).isMesh) meshes.push(object as Mesh);
                });
                // Node order is preserved: each instance's meshes (one per primitive) follow the previous instance's.
                const firstMeshOfInstance: number[] = [];
                let nextMesh = 0;
                for (const count of threeMeshesPerInstance(scene)) {
                    firstMeshOfInstance.push(nextMesh);
                    nextMesh += count;
                }
                scene.instances.forEach((instance, index) => {
                    instance.transform.forEach((value, element) => {
                        expect(meshes[firstMeshOfInstance[index]].matrixWorld.elements[element]).toBeCloseTo(value, 4);
                    });
                });
                const emissiveInstance = scene.instances.findIndex((i) => scene.materials[i.material].emissive.some((c) => c > 0));
                if (emissiveInstance >= 0) {
                    const expected = Math.max(...scene.materials[scene.instances[emissiveInstance].material].emissive);
                    const material = meshes[firstMeshOfInstance[emissiveInstance]].material as unknown as { emissive: { r: number; g: number; b: number }; emissiveIntensity: number };
                    const radiance = Math.max(material.emissive.r, material.emissive.g, material.emissive.b) * material.emissiveIntensity;
                    expect(radiance).toBeCloseTo(expected, 4);
                }
            });
        });
    }
});

/** Decodes one of our PNGs (8-bit RGBA, stored or compressed IDAT, filter 0 on every row) back to its texels. */
function decodePng(png: Uint8Array): { width: number; height: number; rgba: Uint8Array } {
    const view = new DataView(png.buffer, png.byteOffset, png.byteLength);
    expect(Array.from(png.subarray(0, 8))).toEqual([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]);
    let cursor = 8;
    let width = 0;
    let height = 0;
    const idat: Uint8Array[] = [];
    while (cursor < png.length) {
        const length = view.getUint32(cursor, false);
        const type = String.fromCharCode(...png.subarray(cursor + 4, cursor + 8));
        const data = png.subarray(cursor + 8, cursor + 8 + length);
        expect(view.getUint32(cursor + 8 + length, false), `${type} CRC`).toBe(crc32(png.subarray(cursor + 4, cursor + 8 + length)));
        if (type === 'IHDR') {
            width = new DataView(data.buffer, data.byteOffset).getUint32(0, false);
            height = new DataView(data.buffer, data.byteOffset).getUint32(4, false);
            expect([data[8], data[9], data[12]]).toEqual([8, 6, 0]);
        } else if (type === 'IDAT') idat.push(data);
        cursor += 12 + length;
    }
    const raw = inflateSync(Buffer.concat(idat));
    const rgba = new Uint8Array(width * height * 4);
    for (let row = 0; row < height; row++) {
        expect(raw[row * (width * 4 + 1)]).toBe(0);
        rgba.set(raw.subarray(row * (width * 4 + 1) + 1, (row + 1) * (width * 4 + 1)), row * width * 4);
    }
    return { width, height, rgba };
}

describe('textured and multi-material scenes in the GLB', () => {
    it('embeds every texture as a PNG that decodes to the same texels', () => {
        const scene = buildUvBoard();
        const { json, binary } = readGlbChunks(writeGlb(scene, { name: 'uv-board' }));
        expect(json.images).toHaveLength(scene.textures!.length);
        json.images!.forEach((image, index) => {
            expect(image.mimeType).toBe('image/png');
            const view = json.bufferViews[image.bufferView as number] as { byteOffset: number; byteLength: number };
            const decoded = decodePng(binary.subarray(view.byteOffset, view.byteOffset + view.byteLength));
            const texture = scene.textures![index];
            expect([decoded.width, decoded.height]).toEqual([texture.width, texture.height]);
            expect(Buffer.from(decoded.rgba).equals(Buffer.from(texture.data))).toBe(true);
        });
    });

    it("writes each panel's uv mapping as wrap modes, texCoord and KHR_texture_transform", () => {
        const scene = buildUvBoard();
        const { json } = readGlbChunks(writeGlb(scene, { name: 'uv-board' }));
        const wrapCode = { repeat: 10497, clamp: 33071, mirror: 33648 };
        expect(json.extensionsUsed).toContain('KHR_texture_transform');
        for (const panel of UV_BOARD_PANELS) {
            const material = json.materials.find((candidate) => candidate.name === `panel-${panel.name}`)!;
            const info = (material.pbrMetallicRoughness as { baseColorTexture: { index: number; texCoord?: number; extensions?: Record<string, { offset: number[]; rotation: number; scale: number[] }> } }).baseColorTexture;
            const sampler = json.samplers![json.textures![info.index].sampler as number];
            expect([sampler.wrapS, sampler.wrapT], panel.name).toEqual([wrapCode[panel.uv.wrapU ?? 'repeat'], wrapCode[panel.uv.wrapV ?? 'repeat']]);
            expect(info.texCoord ?? 0, panel.name).toBe(panel.uv.channel ?? 0);
            const transform = info.extensions?.KHR_texture_transform;
            if (panel.uv.offset || panel.uv.repeat || panel.uv.rotation) {
                expect(transform, panel.name).toEqual({ offset: [...(panel.uv.offset ?? [0, 0])], rotation: panel.uv.rotation ?? 0, scale: [...(panel.uv.repeat ?? [1, 1])] });
            } else {
                expect(transform, panel.name).toBeUndefined();
            }
        }
        // The uv-set-1 panel's mesh carries a second uv set.
        const withSecondSet = json.meshes.filter((mesh) => (mesh.primitives as Array<{ attributes: Record<string, number> }>)[0].attributes.TEXCOORD_1 !== undefined);
        expect(withSecondSet).toHaveLength(1);
    });

    it('marks the alpha checker MASK at 0.5 and the textured furnace with its checker', () => {
        const alpha = readGlbChunks(writeGlb(buildAlphaChecker(), { name: 'alpha-checker' })).json;
        const cutout = alpha.materials.find((material) => material.name === 'alpha-checker-wall')!;
        expect([cutout.alphaMode, cutout.alphaCutoff, cutout.doubleSided]).toEqual(['MASK', 0.5, true]);
        const furnace = readGlbChunks(writeGlb(buildTexturedFurnace(), { name: 'textured-furnace' })).json;
        expect(furnace.images).toHaveLength(1);
        expect(furnace.materials[0].pbrMetallicRoughness).toHaveProperty('baseColorTexture');
    });

    it('writes the glass wall as one mesh with an opaque and a transmissive primitive', () => {
        const scene = buildGlassWall();
        const { json } = readGlbChunks(writeGlb(scene, { name: 'glass-wall' }));
        const wallMesh = json.meshes.find((mesh) => (mesh.primitives as unknown[]).length === 2)!;
        const [wallPrimitive, panePrimitive] = wallMesh.primitives as Array<{ material: number; indices: number }>;
        expect(json.materials[wallPrimitive.material].name).toBe('partition-wall');
        const glass = json.materials[panePrimitive.material];
        expect(glass.name).toBe('glass-pane');
        expect(glass.extensions).toEqual({ KHR_materials_transmission: { transmissionFactor: 1 } });
        expect(json.extensionsUsed).toContain('KHR_materials_transmission');
        // 4 boxes of wall and 1 box of pane.
        const indexCount = (primitive: { indices: number }) => (json.accessors[primitive.indices] as { count: number }).count;
        expect([indexCount(wallPrimitive), indexCount(panePrimitive)]).toEqual([4 * 36, 36]);
    });
});
