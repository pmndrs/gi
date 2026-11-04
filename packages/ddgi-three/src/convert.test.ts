import {
    BoxGeometry,
    BufferAttribute,
    BufferGeometry,
    ClampToEdgeWrapping,
    Color,
    DataTexture,
    DirectionalLight,
    DoubleSide,
    FrontSide,
    InterleavedBuffer,
    InterleavedBufferAttribute,
    MeshBasicMaterial,
    MeshLambertMaterial,
    MeshPhysicalMaterial,
    MeshStandardMaterial,
    MirroredRepeatWrapping,
    PointLight,
    RepeatWrapping,
    SpotLight,
    Texture,
} from 'three';
import { describe, expect, it } from 'vitest';
import type { TextureHandle } from '@pmndrs/ddgi-core';
import { giAlbedo, lightDescriptor, materialDescriptor, MaterialSnapshot, meshDescriptor, readFloatAttribute, uvMapping } from './convert.ts';

const HANDLE = 7 as TextureHandle;
const noTextures = () => undefined;

function expectClose(actual: readonly number[] | undefined, expected: readonly number[], digits = 6): void {
    expect(actual).toBeDefined();
    expect(actual!.length).toBe(expected.length);
    actual!.forEach((value, index) => expect(value).toBeCloseTo(expected[index], digits));
}

describe('metalness is folded into GI albedo', () => {
    it('scales the base colour by 1 - metalness', () => {
        const material = new MeshStandardMaterial({ color: new Color().setRGB(0.8, 0.6, 0.4), metalness: 0.25 });
        expectClose(giAlbedo(material), [0.6, 0.45, 0.3]);
        expectClose(materialDescriptor(material, noTextures).albedo, [0.6, 0.45, 0.3]);
    });

    it('gives a full metal no diffuse albedo', () => {
        const material = new MeshPhysicalMaterial({ color: 0xffffff, metalness: 1 });
        expectClose(materialDescriptor(material, noTextures).albedo, [0, 0, 0]);
    });

    it('leaves materials without metalness alone', () => {
        const lambert = new MeshLambertMaterial({ color: new Color().setRGB(0.5, 0.25, 0.125) });
        expectClose(materialDescriptor(lambert, noTextures).albedo, [0.5, 0.25, 0.125]);
        const basic = new MeshBasicMaterial({ color: new Color().setRGB(0.3, 0.3, 0.3) });
        expectClose(materialDescriptor(basic, noTextures).albedo, [0.3, 0.3, 0.3]);
    });

    it('clamps metalness outside 0..1', () => {
        const material = new MeshStandardMaterial({ color: 0xffffff });
        material.metalness = 1.5;
        expectClose(giAlbedo(material), [0, 0, 0]);
    });
});

describe('the rest of the material', () => {
    it('passes emissive and its intensity separately', () => {
        const material = new MeshStandardMaterial({ emissive: new Color().setRGB(1, 0.5, 0.25), emissiveIntensity: 12 });
        const descriptor = materialDescriptor(material, noTextures);
        expectClose(descriptor.emissive, [1, 0.5, 0.25]);
        expect(descriptor.emissiveIntensity).toBe(12);
    });

    it('turns alphaTest into alphaCutoff', () => {
        const material = new MeshStandardMaterial({ alphaTest: 0.3 });
        expect(materialDescriptor(material, noTextures).alphaCutoff).toBeCloseTo(0.3);
        expect(materialDescriptor(new MeshStandardMaterial(), noTextures).alphaCutoff).toBeUndefined();
    });

    it('lets light through glass, mostly transparent and invisible materials', () => {
        expect(materialDescriptor(new MeshPhysicalMaterial({ transmission: 1 }), noTextures).blocksLight).toBe(false);
        expect(materialDescriptor(new MeshStandardMaterial({ transparent: true, opacity: 0.2 }), noTextures).blocksLight).toBe(false);
        expect(materialDescriptor(new MeshStandardMaterial({ visible: false }), noTextures).blocksLight).toBe(false);
        expect(materialDescriptor(new MeshStandardMaterial({ transparent: true, opacity: 0.9 }), noTextures).blocksLight).toBe(true);
        expect(materialDescriptor(new MeshStandardMaterial(), noTextures).blocksLight).toBe(true);
    });

    it('cuts blended textured materials at half alpha', () => {
        const material = new MeshStandardMaterial({ transparent: true, map: new Texture() });
        expect(materialDescriptor(material, () => HANDLE).alphaCutoff).toBe(0.5);
    });

    it('maps side to doubleSided', () => {
        expect(materialDescriptor(new MeshStandardMaterial({ side: DoubleSide }), noTextures).doubleSided).toBe(true);
        expect(materialDescriptor(new MeshStandardMaterial({ side: FrontSide }), noTextures).doubleSided).toBe(false);
    });
});

describe('UV mappings are passed through', () => {
    it('passes texture.matrix as the contract\'s 3x2', () => {
        const texture = new Texture();
        texture.repeat.set(2, 3);
        texture.offset.set(0.25, 0.5);
        const mapping = uvMapping(texture);
        expectClose(mapping.matrix, [2, 0, 0, 3, 0.25, 0.5]);
    });

    it('matches three\'s own matrix with rotation and centre', () => {
        const texture = new Texture();
        texture.repeat.set(4, 0.5);
        texture.offset.set(-0.1, 0.3);
        texture.rotation = 0.7;
        texture.center.set(0.5, 0.25);
        const mapping = uvMapping(texture);
        const e = texture.matrix.elements;
        expectClose(mapping.matrix, [e[0], e[1], e[3], e[4], e[6], e[7]]);
        // A point maps the same way through both: uv' = M (u, v, 1).
        const [u, v] = [0.3, 0.8];
        const viaContract = [mapping.matrix![0] * u + mapping.matrix![2] * v + mapping.matrix![4], mapping.matrix![1] * u + mapping.matrix![3] * v + mapping.matrix![5]];
        const viaThree = [e[0] * u + e[3] * v + e[6], e[1] * u + e[4] * v + e[7]];
        expectClose(viaContract, viaThree);
    });

    it('uses a matrix three doesn\'t update as it is', () => {
        const texture = new Texture();
        texture.matrixAutoUpdate = false;
        texture.matrix.set(1, 2, 3, 4, 5, 6, 0, 0, 1);
        expectClose(uvMapping(texture).matrix, [1, 4, 2, 5, 3, 6]);
    });

    it('passes the uv channel and the wraps', () => {
        const texture = new Texture();
        texture.channel = 1;
        texture.wrapS = MirroredRepeatWrapping;
        texture.wrapT = ClampToEdgeWrapping;
        const mapping = uvMapping(texture);
        expect(mapping.channel).toBe(1);
        expect(mapping.wrapU).toBe('mirror');
        expect(mapping.wrapV).toBe('clamp');
        texture.wrapS = RepeatWrapping;
        expect(uvMapping(texture).wrapU).toBe('repeat');
    });

    it('falls back to channel 0 for channels the platform lacks', () => {
        const texture = new Texture();
        texture.channel = 2;
        const mapping = uvMapping(texture);
        expect(mapping.channel).toBe(0);
        expect(mapping.channelSupported).toBe(false);
    });

    it('puts the mapping and handle on the material', () => {
        const map = new Texture();
        map.repeat.set(2, 2);
        map.channel = 1;
        const emissiveMap = new Texture();
        emissiveMap.offset.set(0.5, 0);
        const material = new MeshStandardMaterial({ map, emissiveMap });
        const descriptor = materialDescriptor(material, (texture) => (texture === map ? HANDLE : ((HANDLE + 1) as TextureHandle)));
        expect(descriptor.albedoTexture).toBe(HANDLE);
        expect(descriptor.albedoUv?.channel).toBe(1);
        expectClose(descriptor.albedoUv?.matrix, [2, 0, 0, 2, 0, 0]);
        expect(descriptor.emissiveTexture).toBe(HANDLE + 1);
        expectClose(descriptor.emissiveUv?.matrix, [1, 0, 0, 1, 0.5, 0]);
    });

    it('leaves the mapping out when the texture has no handle yet', () => {
        const material = new MeshStandardMaterial({ map: new Texture() });
        const descriptor = materialDescriptor(material, noTextures);
        expect(descriptor.albedoTexture).toBeUndefined();
        expect(descriptor.albedoUv).toBeUndefined();
    });
});

describe('vertex normals are passed through', () => {
    it('passes the normal attribute', () => {
        const geometry = new BoxGeometry(1, 2, 3);
        const descriptor = meshDescriptor(geometry)!;
        expect(descriptor.normals).toEqual(geometry.getAttribute('normal').array);
        expect(descriptor.positions).toEqual(geometry.getAttribute('position').array);
        expect(descriptor.uvs).toEqual(geometry.getAttribute('uv').array);
        expect(descriptor.indices).toEqual(geometry.index!.array);
    });

    it('de-interleaves normals', () => {
        // position xyz, normal xyz per vertex, in one buffer.
        const values = new Float32Array([0, 0, 0, 0, 1, 0, 1, 0, 0, 0.6, 0.8, 0, 0, 0, 1, 0, 0, -1]);
        const buffer = new InterleavedBuffer(values, 6);
        const geometry = new BufferGeometry();
        geometry.setAttribute('position', new InterleavedBufferAttribute(buffer, 3, 0));
        geometry.setAttribute('normal', new InterleavedBufferAttribute(buffer, 3, 3));
        const descriptor = meshDescriptor(geometry)!;
        expectClose(Array.from(descriptor.normals!), [0, 1, 0, 0.6, 0.8, 0, 0, 0, -1]);
        expectClose(Array.from(descriptor.positions), [0, 0, 0, 1, 0, 0, 0, 0, 1]);
    });

    it('undoes normalized integer normals', () => {
        const attribute = new BufferAttribute(new Int8Array([0, 127, 0, -127, 0, 0]), 3, true);
        expectClose(Array.from(readFloatAttribute(attribute, 3)), [0, 1, 0, -1, 0, 0], 5);
    });

    it('leaves normals out when the geometry has none', () => {
        const geometry = new BufferGeometry();
        geometry.setAttribute('position', new BufferAttribute(new Float32Array(9), 3));
        expect(meshDescriptor(geometry)!.normals).toBeUndefined();
    });
});

describe('meshes', () => {
    it('makes up an index for non-indexed geometry', () => {
        const geometry = new BufferGeometry();
        geometry.setAttribute('position', new BufferAttribute(new Float32Array(18), 3));
        const descriptor = meshDescriptor(geometry)!;
        expect(Array.from(descriptor.indices)).toEqual([0, 1, 2, 3, 4, 5]);
        expect(descriptor.indices).toBeInstanceOf(Uint16Array);
    });

    it('passes groups as material slots, clipped to whole triangles', () => {
        const geometry = new BoxGeometry();
        const descriptor = meshDescriptor(geometry)!;
        expect(descriptor.groups).toHaveLength(6);
        expect(descriptor.groups![5]).toEqual({ start: 30, count: 6, materialSlot: 5 });
        geometry.clearGroups();
        geometry.addGroup(0, Infinity, 2);
        expect(meshDescriptor(geometry)!.groups).toEqual([{ start: 0, count: 36, materialSlot: 2 }]);
    });

    it('passes uv1 only when asked', () => {
        const geometry = new BoxGeometry();
        geometry.setAttribute('uv1', geometry.getAttribute('uv').clone());
        expect(meshDescriptor(geometry)!.uvs1).toBeUndefined();
        expect(meshDescriptor(geometry, { includeUv1: true })!.uvs1).toEqual(geometry.getAttribute('uv1').array);
    });

    it('refuses a geometry without positions', () => {
        expect(meshDescriptor(new BufferGeometry())).toBeNull();
    });
});

describe('lights', () => {
    it('sends a point light with distance 0 as range 0 (no limit)', () => {
        const light = new PointLight(0xffffff, 15, 0, 2);
        light.position.set(1, 2, 3);
        light.updateMatrixWorld();
        const descriptor = lightDescriptor(light)!;
        expect(descriptor.type).toBe('point');
        if (descriptor.type !== 'point') return;
        expect(descriptor.range).toBe(0);
        expect(descriptor.position).toEqual([1, 2, 3]);
        expect(descriptor.intensity).toBe(15);
    });

    it('passes a point light\'s distance as its range', () => {
        const light = new PointLight(0xffffff, 1, 7.5);
        light.updateMatrixWorld();
        const descriptor = lightDescriptor(light)!;
        expect(descriptor.type === 'point' && descriptor.range).toBe(7.5);
    });

    it('points a directional light from its position to its target', () => {
        const light = new DirectionalLight(new Color().setRGB(1, 0.5, 0.25), 3);
        light.position.set(0, 10, 0);
        light.target.position.set(10, 0, 0);
        light.updateMatrixWorld();
        light.target.updateMatrixWorld();
        const descriptor = lightDescriptor(light)!;
        expect(descriptor.type).toBe('directional');
        if (descriptor.type !== 'directional') return;
        expectClose(descriptor.direction, [Math.SQRT1_2, -Math.SQRT1_2, 0]);
        expectClose(descriptor.color, [1, 0.5, 0.25]);
        expect(descriptor.intensity).toBe(3);
    });

    it('maps a spot light\'s angle and penumbra to outer and inner cones', () => {
        const light = new SpotLight(0xffffff, 100, 0, Math.PI / 4, 0.5, 2);
        light.position.set(0, 3, 0);
        light.target.position.set(0, 0, 0);
        light.updateMatrixWorld();
        light.target.updateMatrixWorld();
        const descriptor = lightDescriptor(light)!;
        expect(descriptor.type).toBe('spot');
        if (descriptor.type !== 'spot') return;
        expect(descriptor.outerConeAngle).toBeCloseTo(Math.PI / 4);
        expect(descriptor.innerConeAngle).toBeCloseTo(Math.PI / 8);
        expect(descriptor.range).toBe(0);
        expectClose(descriptor.direction, [0, -1, 0]);
    });

    it('takes the light group from userData and scales hidden lights to 0', () => {
        const light = new PointLight(0xffffff, 4);
        light.userData.giLightGroup = 2;
        light.updateMatrixWorld();
        expect(lightDescriptor(light)!.group).toBe(2);
        expect(lightDescriptor(light, 0)!.intensity).toBe(0);
        light.userData.giLightGroup = 9;
        expect(lightDescriptor(light)!.group).toBe(0);
    });
});

describe('material snapshots', () => {
    it('see colour, metalness and texture edits that three\'s version misses', () => {
        const map = new DataTexture(new Uint8Array(4), 1, 1);
        const material = new MeshStandardMaterial({ color: 0xff0000, map });
        const before = new MaterialSnapshot();
        const after = new MaterialSnapshot();
        before.read(material);
        after.read(material);
        expect(after.equals(before)).toBe(true);
        material.color.setRGB(0, 1, 0);
        after.read(material);
        expect(after.equals(before)).toBe(false);
        before.read(material);
        material.metalness = 0.5;
        after.read(material);
        expect(after.equals(before)).toBe(false);
        before.read(material);
        map.offset.set(0.5, 0);
        after.read(material);
        expect(after.equals(before)).toBe(false);
        before.read(material);
        material.map = null;
        after.read(material);
        expect(after.equals(before)).toBe(false);
    });
});
