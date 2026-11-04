/**
 * Counts what a three.js scene contains, for the contact sheet and the
 * per-scene numbers in PROGRESS.
 *
 * Only type imports from three: the checks use three's `isMesh`-style flags,
 * so this module loads without three.
 */

import type { BufferGeometry, InstancedMesh, Material, Mesh, Object3D, Texture } from 'three';

export interface SceneStats {
    /** Triangles drawn, counting every instance of an instanced mesh. */
    triangles: number;
    /** Mesh objects (an instanced mesh counts once). */
    meshes: number;
    /** Distinct material objects. */
    materials: number;
    /** Distinct texture objects referenced by those materials. */
    textures: number;
}

function geometryTriangles(geometry: BufferGeometry): number {
    const indexCount = geometry.index?.count;
    if (indexCount !== undefined) return indexCount / 3;
    return (geometry.getAttribute('position')?.count ?? 0) / 3;
}

function isTexture(value: unknown): value is Texture {
    return typeof value === 'object' && value !== null && (value as { isTexture?: boolean }).isTexture === true;
}

export function computeSceneStats(root: Object3D): SceneStats {
    let triangles = 0;
    let meshes = 0;
    const materials = new Set<Material>();
    const textures = new Set<Texture>();

    root.traverse((object) => {
        const mesh = object as Mesh;
        if (!mesh.isMesh) return;
        meshes++;
        const instances = (object as InstancedMesh).isInstancedMesh ? (object as InstancedMesh).count : 1;
        triangles += geometryTriangles(mesh.geometry) * instances;

        for (const material of Array.isArray(mesh.material) ? mesh.material : [mesh.material]) {
            materials.add(material);
            for (const value of Object.values(material)) {
                if (isTexture(value)) textures.add(value);
            }
        }
    });

    return { triangles, meshes, materials: materials.size, textures: textures.size };
}
