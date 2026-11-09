/**
 * M4's textured test scenes (T4.4): the numbers each one is built to have.
 */

import { Texture } from 'three';
import { describe, expect, it } from 'vitest';
import { proceduralSceneToThree } from '../three-scene';
import type { Mesh, MeshPhysicalMaterial, MeshStandardMaterial } from 'three';
import { FURNACE_ALBEDO } from './furnace-room';
import { ALPHA_CHECKER_CUTOFF, buildAlphaChecker, buildGlassWall, GLASS_WINDOW_Y_M, GLASS_WINDOW_Z_M, SEE_THROUGH_ROOM_HEIGHT_M, SEE_THROUGH_ROOM_SIZE_M } from './see-through-walls';
import { buildTexturedFurnace, TEXTURED_FURNACE_FACTOR } from './textured-furnace';
import {
    createFurnaceChecker,
    FURNACE_CHECKER_DARK_SRGB,
    FURNACE_CHECKER_LIGHT_SRGB,
    linearToSrgbByte,
    srgbByteToLinear,
    textureMeanLinear,
    uvMappingMatrix,
} from './textures';
import { buildUvBoard, UV_BOARD_PANELS } from './uv-board';

describe('sRGB helpers', () => {
    it('decode the end points exactly and round-trip every byte', () => {
        expect(srgbByteToLinear(0)).toBe(0);
        expect(srgbByteToLinear(255)).toBe(1);
        for (let byte = 0; byte < 256; byte++) expect(linearToSrgbByte(srgbByteToLinear(byte))).toBe(byte);
    });
});

describe('textured-furnace', () => {
    const scene = buildTexturedFurnace();
    const texture = scene.textures![0];

    it("has a checkerboard whose mean albedo is 0.5 in linear light, with the material's factor", () => {
        const [red, green, blue, alpha] = textureMeanLinear(texture);
        // The texture alone: as close to 0.5 as two 8-bit sRGB levels get (0.016% above).
        for (const channel of [red, green, blue]) expect(Math.abs(channel - 0.5)).toBeLessThan(2e-4);
        expect(alpha).toBe(1);
        // With the material's colour factor: 0.5 to float precision.
        const material = scene.materials[0];
        expect(material.albedoMap?.texture).toBe(0);
        for (let channel = 0; channel < 3; channel++) expect(material.albedo[channel] * textureMeanLinear(texture)[channel]).toBeCloseTo(FURNACE_ALBEDO, 12);
        expect(TEXTURED_FURNACE_FACTOR).toBeLessThanOrEqual(1);
    });

    it('is half dark squares and half light ones, about 0.26 and 0.74', () => {
        let dark = 0;
        let light = 0;
        for (let texel = 0; texel < texture.width * texture.height; texel++) {
            const value = texture.data[texel * 4];
            if (value === FURNACE_CHECKER_DARK_SRGB) dark++;
            else if (value === FURNACE_CHECKER_LIGHT_SRGB) light++;
        }
        expect(dark).toBe(light);
        expect(dark + light).toBe(texture.width * texture.height);
        expect(srgbByteToLinear(FURNACE_CHECKER_DARK_SRGB) * TEXTURED_FURNACE_FACTOR).toBeCloseTo(0.262, 3);
        expect(srgbByteToLinear(FURNACE_CHECKER_LIGHT_SRGB) * TEXTURED_FURNACE_FACTOR).toBeCloseTo(0.738, 3);
    });

    it('is the flat furnace in every other respect: same geometry, same emission, every instance textured', () => {
        expect(scene.materials).toHaveLength(1);
        expect(scene.materials[0].emissive).toEqual([0.25, 0.25, 0.25]);
        expect(scene.lights).toEqual([]);
        expect(Buffer.from(createFurnaceChecker().data).equals(Buffer.from(texture.data))).toBe(true);
    });
});

describe('alpha-checker', () => {
    const scene = buildAlphaChecker();
    const wall = scene.instances.find((instance) => instance.name === 'alpha-checker-wall')!;
    const material = scene.materials[wall.material];
    const texture = scene.textures![material.albedoMap!.texture];

    it('is a cutout with exactly half its texels open', () => {
        expect(material.alphaCutoff).toBe(ALPHA_CHECKER_CUTOFF);
        expect(material.doubleSided).toBe(true);
        let open = 0;
        let partial = 0;
        for (let texel = 0; texel < texture.width * texture.height; texel++) {
            const alpha = texture.data[texel * 4 + 3];
            if (alpha !== 0 && alpha !== 255) partial++;
            if (alpha / 255 < ALPHA_CHECKER_CUTOFF) open++;
        }
        expect(partial).toBe(0);
        expect(open / (texture.width * texture.height)).toBe(0.5);
    });

    it("maps the room's opening (not the whole quad) onto the texture's 0..1", () => {
        const m = uvMappingMatrix(material.albedoMap!.uv);
        const embed = 0.1;
        const quadWidth = SEE_THROUGH_ROOM_SIZE_M + 2 * embed;
        const quadHeight = SEE_THROUGH_ROOM_HEIGHT_M + 2 * embed;
        // The quad's own uv at the opening's corners, then the mapping.
        const mapped = (u: number, v: number) => [m[0] * u + m[2] * v + m[4], m[1] * u + m[3] * v + m[5]];
        const [u0, v0] = mapped(embed / quadWidth, embed / quadHeight);
        const [u1, v1] = mapped(1 - embed / quadWidth, 1 - embed / quadHeight);
        expect(u0).toBeCloseTo(0, 12);
        expect(v0).toBeCloseTo(0, 12);
        expect(u1).toBeCloseTo(1, 12);
        expect(v1).toBeCloseTo(1, 12);
    });

    it('lights room A only', () => {
        expect(scene.lights).toHaveLength(1);
        const lamp = scene.lights[0];
        expect(lamp.type).toBe('point');
        if (lamp.type === 'point') expect(lamp.position[0]).toBeLessThan(SEE_THROUGH_ROOM_SIZE_M);
    });
});

describe('glass-wall', () => {
    const scene = buildGlassWall();
    const wall = scene.instances.find((instance) => instance.name === 'glass-wall')!;
    const mesh = scene.meshes[wall.mesh];

    it('is one mesh with two material slots: the opaque wall and the glass pane', () => {
        expect(mesh.groups).toHaveLength(2);
        expect(wall.materials).toHaveLength(2);
        expect(wall.material).toBe(wall.materials![0]);
        const [wallMaterial, glassMaterial] = wall.materials!.map((index) => scene.materials[index]);
        expect(wallMaterial.transmission ?? 0).toBe(0);
        expect(wallMaterial.blocksLight ?? true).toBe(true);
        expect(glassMaterial.transmission).toBe(1);
        expect(glassMaterial.blocksLight).toBe(false);
        // The groups cover every triangle once, in order.
        const [first, second] = mesh.groups!;
        expect(first.start).toBe(0);
        expect(second.start).toBe(first.count);
        expect(first.count + second.count).toBe(mesh.indices.length);
    });

    it('puts the pane over the window and nowhere else', () => {
        const pane = mesh.groups!.find((group) => group.materialSlot === 1)!;
        let min = [Infinity, Infinity, Infinity];
        let max = [-Infinity, -Infinity, -Infinity];
        for (let index = pane.start; index < pane.start + pane.count; index++) {
            const vertex = mesh.indices[index];
            const position = [mesh.positions[vertex * 3], mesh.positions[vertex * 3 + 1], mesh.positions[vertex * 3 + 2]];
            min = min.map((value, axis) => Math.min(value, position[axis]));
            max = max.map((value, axis) => Math.max(value, position[axis]));
        }
        expect(min[1]).toBeCloseTo(GLASS_WINDOW_Y_M[0] - 0.01, 6);
        expect(max[1]).toBeCloseTo(GLASS_WINDOW_Y_M[1] + 0.01, 6);
        expect(min[2]).toBeCloseTo(GLASS_WINDOW_Z_M[0] - 0.01, 6);
        expect(max[2]).toBeCloseTo(GLASS_WINDOW_Z_M[1] + 0.01, 6);
    });

    it('becomes one three Mesh with a material per slot, the pane transmissive', () => {
        const root = proceduralSceneToThree(scene, 'glass-wall');
        const threeWall = root.children.find((child) => child.name === 'glass-wall') as Mesh;
        const materials = threeWall.material as Array<MeshStandardMaterial | MeshPhysicalMaterial>;
        expect(materials).toHaveLength(2);
        expect((materials[1] as MeshPhysicalMaterial).transmission).toBe(1);
        expect(threeWall.geometry.groups.map((group) => group.materialIndex)).toEqual([0, 1]);
    });
});

describe('uv-board', () => {
    const scene = buildUvBoard();

    it('has eight panels, each with its own texture and mapping', () => {
        expect(UV_BOARD_PANELS).toHaveLength(8);
        const panels = scene.instances.filter((instance) => instance.name.startsWith('panel-'));
        expect(panels).toHaveLength(8);
        const textures = new Set(panels.map((panel) => scene.materials[panel.material].albedoMap!.texture));
        expect(textures.size).toBe(8);
        const wraps = new Set(UV_BOARD_PANELS.flatMap((panel) => [panel.uv.wrapU ?? 'repeat', panel.uv.wrapV ?? 'repeat']));
        expect([...wraps].sort()).toEqual(['clamp', 'mirror', 'repeat']);
        expect(UV_BOARD_PANELS.some((panel) => panel.uv.channel === 1)).toBe(true);
        expect(UV_BOARD_PANELS.some((panel) => (panel.uv.rotation ?? 0) !== 0)).toBe(true);
        expect(UV_BOARD_PANELS.some((panel) => panel.uv.offset !== undefined)).toBe(true);
    });

    it('gives the uv-set-1 panel a second uv set that differs from the first', () => {
        const panel = scene.instances.find((instance) => instance.name === 'panel-uv-set-1')!;
        const mesh = scene.meshes[panel.mesh];
        expect(mesh.uvs1).toBeDefined();
        expect(Array.from(mesh.uvs1!)).not.toEqual(Array.from(mesh.uvs));
    });

    it("computes every panel's uv matrix as three's Texture.updateMatrix does", () => {
        for (const panel of UV_BOARD_PANELS) {
            const texture = new Texture();
            texture.offset.set(...(panel.uv.offset ?? [0, 0]));
            texture.repeat.set(...(panel.uv.repeat ?? [1, 1]));
            texture.rotation = panel.uv.rotation ?? 0;
            texture.center.set(0, 0);
            texture.updateMatrix();
            const e = texture.matrix.elements;
            const ours = uvMappingMatrix(panel.uv);
            [e[0], e[1], e[3], e[4], e[6], e[7]].forEach((value, index) => expect(ours[index], `${panel.name} [${index}]`).toBeCloseTo(value, 12));
        }
    });

    it('refuses a rotated mapping with an uneven repeat, where three and glTF disagree', () => {
        expect(() => uvMappingMatrix({ rotation: 0.3, repeat: [2, 1] })).toThrow(/same repeat/);
    });

    it('gives the three textures the mapping, sRGB colour and uv channel', () => {
        const root = proceduralSceneToThree(scene, 'uv-board');
        const panel = root.children.find((child) => child.name === 'panel-uv-set-1') as Mesh;
        const map = (panel.material as MeshStandardMaterial).map!;
        expect(map.channel).toBe(1);
        expect(map.colorSpace).toBe('srgb');
        expect(panel.geometry.getAttribute('uv1')).toBeDefined();
        const mirror = (root.children.find((child) => child.name === 'panel-mirror') as Mesh).material as MeshStandardMaterial;
        expect(mirror.map!.repeat.toArray()).toEqual([3, 2]);
    });
});
