/**
 * Hands procedural scene data to the platform's scene API, everything the
 * data says: textures (as `PixelData`, the scene's own pixels), meshes with
 * both uv sets and their material groups, materials with their maps, uv
 * mappings, alpha and metalness, and instances with a material per slot.
 *
 * Used by the GPU test pages (`tools/gpu/pages/ddgi-shared/harness.ts`), so a
 * page sees the alpha checker's holes and the glass wall's pane as the
 * viewer and the references do. Nothing is dropped silently: a field this
 * file doesn't know how to pass throws, naming it, so a new field in
 * scene-data.ts can't be lost on the way to GI.
 *
 * Engine-free; the contract is imported for types only.
 */

import type { InstanceHandle, MaterialHandle, MeshHandle, TextureHandle } from '../../packages/ddgi-core/src/platform/contract/handles';
import type { MaterialDescriptor, SceneApi, UvMapping } from '../../packages/ddgi-core/src/platform/contract/scene';
import { uvMappingMatrix } from './procedural/textures';
import type { ProceduralScene, SceneTextureReference } from './scene-data';

/** The platform handles of a described scene, in the data's order (instances by name). */
export interface PlatformSceneHandles {
    readonly textures: readonly TextureHandle[];
    readonly meshes: readonly MeshHandle[];
    readonly materials: readonly MaterialHandle[];
    readonly instances: ReadonlyMap<string, InstanceHandle>;
}

/** The fields of each kind this file passes on. Anything else throws (see the file comment). */
const MESH_FIELDS = new Set(['name', 'positions', 'normals', 'uvs', 'uvs1', 'indices', 'groups']);
const MATERIAL_FIELDS = new Set([
    'name',
    'albedo',
    'albedoMap',
    'emissive',
    'alphaCutoff',
    'alphaMode',
    'opacity',
    'metalness',
    'metalnessMap',
    'transmission',
    'blocksLight',
    'doubleSided',
]);
const INSTANCE_FIELDS = new Set(['name', 'mesh', 'material', 'materials', 'transform']);
const TEXTURE_FIELDS = new Set(['name', 'width', 'height', 'data', 'srgb']);

function requireKnownFields(kind: string, name: string, value: object, known: ReadonlySet<string>): void {
    for (const [field, fieldValue] of Object.entries(value)) {
        if (fieldValue !== undefined && !known.has(field)) {
            throw new Error(`describing ${kind} '${name}' for the platform: its field '${field}' isn't passed on (scenes/src/platform-scene.ts)`);
        }
    }
}

/**
 * A scene texture's mapping in the contract's terms: the uv set, the 3×2
 * matrix three would build (`uvMappingMatrix`) and the wrap per axis. Scene
 * textures keep row 0 at v = 0, which is the contract's convention too, so
 * no flip is added.
 */
export function platformUvMapping(reference: SceneTextureReference): UvMapping {
    const uv = reference.uv ?? {};
    return {
        channel: uv.channel ?? 0,
        matrix: uvMappingMatrix(uv),
        wrapU: uv.wrapU ?? 'repeat',
        wrapV: uv.wrapV ?? 'repeat',
    };
}

/** Describes `data` through `scene`. Throws on a field it can't pass, or a reference out of range. */
export function describeProceduralScene(scene: SceneApi, data: ProceduralScene): PlatformSceneHandles {
    const textures = (data.textures ?? []).map((texture) => {
        requireKnownFields('texture', texture.name, texture, TEXTURE_FIELDS);
        return scene.createTexture({ source: { data: texture.data, width: texture.width, height: texture.height }, srgb: texture.srgb, label: texture.name });
    });
    const textureHandle = (owner: string, field: string, index: number): TextureHandle => {
        const handle = textures[index];
        if (handle === undefined) throw new Error(`material '${owner}': ${field}.texture ${index} is not one of the scene's ${textures.length} textures`);
        return handle;
    };

    const meshes = data.meshes.map((mesh) => {
        requireKnownFields('mesh', mesh.name, mesh, MESH_FIELDS);
        return scene.createMesh({
            positions: mesh.positions,
            indices: mesh.indices,
            normals: mesh.normals,
            uvs: mesh.uvs,
            ...(mesh.uvs1 ? { uvs1: mesh.uvs1 } : {}),
            ...(mesh.groups ? { groups: mesh.groups.map((group) => ({ start: group.start, count: group.count, materialSlot: group.materialSlot })) } : {}),
            label: mesh.name,
        });
    });

    const materials = data.materials.map((material) => {
        requireKnownFields('material', material.name, material, MATERIAL_FIELDS);
        const transmission = material.transmission ?? 0;
        if (transmission > 0 && material.blocksLight === true) {
            throw new Error(`material '${material.name}': transmission ${transmission} with blocksLight true; GI takes glass as letting light through`);
        }
        const descriptor: MaterialDescriptor = {
            albedo: material.albedo,
            emissive: material.emissive,
            doubleSided: material.doubleSided,
            label: material.name,
            ...(material.albedoMap
                ? { albedoTexture: textureHandle(material.name, 'albedoMap', material.albedoMap.texture), albedoUv: platformUvMapping(material.albedoMap) }
                : {}),
            ...(material.alphaCutoff !== undefined ? { alphaCutoff: material.alphaCutoff } : {}),
            ...(material.alphaMode !== undefined ? { alphaMode: material.alphaMode } : {}),
            ...(material.opacity !== undefined ? { opacity: material.opacity } : {}),
            // Scene data follows glTF: a metalness map with no factor means factor 1 (glTF's
            // metallicFactor default), as glb.ts and three-scene.ts also take it.
            ...(material.metalness !== undefined || material.metalnessMap ? { metalness: material.metalness ?? 1 } : {}),
            ...(material.metalnessMap
                ? {
                      metalnessTexture: {
                          texture: textureHandle(material.name, 'metalnessMap', material.metalnessMap.texture),
                          channel: material.metalnessMap.channel,
                          uv: platformUvMapping(material.metalnessMap),
                      },
                  }
                : {}),
            // The contract has no transmission: glass is a surface that doesn't block light, as the three adapter maps it.
            ...(material.blocksLight !== undefined ? { blocksLight: material.blocksLight } : transmission > 0 ? { blocksLight: false } : {}),
        };
        return scene.createMaterial(descriptor);
    });

    const instances = new Map<string, InstanceHandle>();
    for (const instance of data.instances) {
        requireKnownFields('instance', instance.name, instance, INSTANCE_FIELDS);
        const mesh = data.meshes[instance.mesh];
        const slots = instance.materials ?? [instance.material];
        const slotsNeeded = Math.max(0, ...(mesh.groups ?? []).map((group) => group.materialSlot)) + 1;
        if (slots.length < slotsNeeded) {
            throw new Error(`instance '${instance.name}': its mesh '${mesh.name}' uses ${slotsNeeded} material slots and 'materials' gives ${slots.length}`);
        }
        instances.set(
            instance.name,
            scene.createInstance({ mesh: meshes[instance.mesh], materials: slots.map((slot) => materials[slot]), transform: instance.transform, label: instance.name }),
        );
    }
    return { textures, meshes, materials, instances };
}
