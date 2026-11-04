import {
    BoxGeometry,
    BufferGeometry,
    Float32BufferAttribute,
    Group,
    InstancedMesh,
    Mesh,
    MeshStandardMaterial,
    Texture,
} from 'three';
import { describe, expect, it } from 'vitest';
import { computeSceneStats } from './stats';

describe('computeSceneStats', () => {
    it('counts an empty scene as zero', () => {
        expect(computeSceneStats(new Group())).toEqual({ triangles: 0, meshes: 0, materials: 0, textures: 0 });
    });

    it('counts indexed and non-indexed triangles, shared materials once, and instanced meshes per instance', () => {
        const root = new Group();
        const shared = new MeshStandardMaterial();
        const map = new Texture();
        const textured = new MeshStandardMaterial({ map, emissiveMap: map });

        root.add(new Mesh(new BoxGeometry(), shared)); // 12 triangles, indexed
        root.add(new Mesh(new BoxGeometry(), shared));

        const loose = new BufferGeometry();
        loose.setAttribute('position', new Float32BufferAttribute(new Float32Array(9 * 3), 3)); // 9 vertices, 3 triangles
        root.add(new Mesh(loose, textured));

        root.add(new InstancedMesh(new BoxGeometry(), shared, 5)); // 5 * 12

        expect(computeSceneStats(root)).toEqual({
            triangles: 12 + 12 + 9 / 3 + 5 * 12,
            meshes: 4,
            materials: 2,
            textures: 1, // the same Texture object used twice counts once
        });
    });

    it('counts every material of a multi-material mesh', () => {
        const mesh = new Mesh(new BoxGeometry(), [new MeshStandardMaterial(), new MeshStandardMaterial()]);
        expect(computeSceneStats(mesh).materials).toBe(2);
    });
});
