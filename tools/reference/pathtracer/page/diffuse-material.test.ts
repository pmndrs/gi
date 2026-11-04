import { BoxGeometry, DoubleSide, Group, Mesh, MeshPhysicalMaterial, MeshStandardMaterial, Texture, Vector2 } from 'three/webgpu';
import { describe, expect, it } from 'vitest';
import { diffuseOnlyMaterial, makeDiffuseOnly } from './diffuse-material.ts';

describe('diffuseOnlyMaterial', () => {
    it('keeps colour, textures, emission, alpha and side, and drops every non-diffuse lobe', () => {
        const baseColorTexture = new Texture();
        const emissiveTexture = new Texture();
        const alphaTexture = new Texture();
        const normalTexture = new Texture();
        const metalRoughTexture = new Texture();
        const source = new MeshStandardMaterial({
            name: 'ivy',
            color: 0x336699,
            map: baseColorTexture,
            emissive: 0xff8800,
            emissiveIntensity: 3,
            emissiveMap: emissiveTexture,
            alphaMap: alphaTexture,
            alphaTest: 0.5,
            transparent: true,
            opacity: 0.75,
            normalMap: normalTexture,
            normalScale: new Vector2(0.5, -0.5),
            side: DoubleSide,
            vertexColors: true,
            metalness: 1,
            roughness: 0.1,
            metalnessMap: metalRoughTexture,
            roughnessMap: metalRoughTexture,
        });

        const result = diffuseOnlyMaterial(source);

        expect(result.isMeshPhysicalMaterial).toBe(true);
        expect(result.name).toBe('ivy');
        expect(result.color.getHex()).toBe(0x336699);
        expect(result.map).toBe(baseColorTexture);
        expect(result.emissive.getHex()).toBe(0xff8800);
        expect(result.emissiveIntensity).toBe(3);
        expect(result.emissiveMap).toBe(emissiveTexture);
        expect(result.alphaMap).toBe(alphaTexture);
        expect(result.alphaTest).toBe(0.5);
        expect(result.transparent).toBe(true);
        expect(result.opacity).toBe(0.75);
        expect(result.normalMap).toBe(normalTexture);
        expect(result.normalScale.toArray()).toEqual([0.5, -0.5]);
        expect(result.side).toBe(DoubleSide);
        expect(result.vertexColors).toBe(true);

        // Pure Lambert: no metal, no specular, no maps that would bring either back.
        expect(result.metalness).toBe(0);
        expect(result.roughness).toBe(1);
        expect(result.specularIntensity).toBe(0);
        expect(result.metalnessMap).toBeNull();
        expect(result.roughnessMap).toBeNull();
        expect(result.clearcoat).toBe(0);
        expect(result.sheen).toBe(0);
        expect(result.transmission).toBe(0);
        expect(result.iridescence).toBe(0);
    });
});

describe('makeDiffuseOnly', () => {
    it('converts every mesh, a shared material once, and lists the emitters', () => {
        const shared = new MeshStandardMaterial({ color: 0x808080 });
        const lamp = new MeshStandardMaterial({ emissive: 0xffffff, emissiveIntensity: 17 });
        const darkEmissive = new MeshStandardMaterial({ emissive: 0x000000 });
        const geometry = new BoxGeometry();
        const root = new Group();
        const wallA = new Mesh(geometry, shared);
        const wallB = new Mesh(geometry, shared);
        const multi = new Mesh(geometry, [lamp, darkEmissive]);
        root.add(wallA, wallB, multi);

        const emitters = makeDiffuseOnly(root);

        expect(wallA.material).toBe(wallB.material);
        expect(wallA.material).toBeInstanceOf(MeshPhysicalMaterial);
        expect((wallA.material as MeshPhysicalMaterial).specularIntensity).toBe(0);
        const parts = multi.material as MeshPhysicalMaterial[];
        expect(parts).toHaveLength(2);
        expect(emitters).toEqual([parts[0]]);
        expect(emitters[0].emissiveIntensity).toBe(17);
    });
});
