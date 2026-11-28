/**
 * The procedural scenes reach the platform's scene API with everything their
 * data says (platform-scene.ts), checked against a scene API that records
 * what it is given.
 */

import { describe, expect, it } from 'vitest';
import type { InstanceDescriptor, MaterialDescriptor, MeshDescriptor, SceneApi, TextureDescriptor } from '../../packages/core/src/platform/contract/scene';
import { describeProceduralScene } from './platform-scene';
import { buildAlphaChecker, buildGlassWall } from './procedural/see-through-walls';
import { buildTexturedFurnace } from './procedural/textured-furnace';
import { uvMappingMatrix } from './procedural/textures';
import { buildUvBoard, UV_BOARD_PANELS } from './procedural/uv-board';
import type { ProceduralScene } from './scene-data';

interface Recorded {
    textures: TextureDescriptor[];
    meshes: MeshDescriptor[];
    materials: MaterialDescriptor[];
    instances: InstanceDescriptor[];
}

/** A scene API whose handles are the index of what was created, as `{ index }`. */
function recordingSceneApi(): { api: SceneApi; recorded: Recorded } {
    const recorded: Recorded = { textures: [], meshes: [], materials: [], instances: [] };
    const create =
        <T>(list: T[]) =>
        (descriptor: T) => {
            list.push(descriptor);
            return { index: list.length - 1 } as never;
        };
    const api = {
        createTexture: create(recorded.textures),
        createMesh: create(recorded.meshes),
        createMaterial: create(recorded.materials),
        createInstance: create(recorded.instances),
    } as unknown as SceneApi;
    return { api, recorded };
}

const indexOf = (handle: unknown) => (handle as { index: number }).index;

describe('describeProceduralScene', () => {
    it('passes the alpha checker as a textured cutout, its pixels as PixelData', () => {
        const scene = buildAlphaChecker();
        const { api, recorded } = recordingSceneApi();
        describeProceduralScene(api, scene);
        expect(recorded.textures).toHaveLength(1);
        expect(recorded.textures[0].srgb).toBe(true);
        expect((recorded.textures[0].source as { data: Uint8Array }).data).toBe(scene.textures![0].data);
        const wall = recorded.materials.find((material) => material.label === 'alpha-checker-wall')!;
        expect(wall.alphaCutoff).toBe(0.5);
        expect(indexOf(wall.albedoTexture)).toBe(0);
        expect(wall.albedoUv?.matrix).toEqual(uvMappingMatrix(scene.materials.find((material) => material.name === 'alpha-checker-wall')!.albedoMap!.uv));
    });

    it('passes the glass wall as one mesh with two groups and a material per slot, the pane not blocking light', () => {
        const scene = buildGlassWall();
        const { api, recorded } = recordingSceneApi();
        describeProceduralScene(api, scene);
        const wallMesh = recorded.meshes.find((mesh) => mesh.label === 'glass-wall')!;
        expect(wallMesh.groups).toHaveLength(2);
        const instance = recorded.instances.find((candidate) => candidate.label === 'glass-wall')!;
        const [wallMaterial, glass] = instance.materials.map((handle) => recorded.materials[indexOf(handle)]);
        expect(wallMaterial.blocksLight).toBeUndefined();
        expect(glass.label).toBe('glass-pane');
        expect(glass.blocksLight).toBe(false);
    });

    it("passes the uv board's second uv set and every panel's mapping", () => {
        const scene = buildUvBoard();
        const { api, recorded } = recordingSceneApi();
        describeProceduralScene(api, scene);
        expect(recorded.meshes.filter((mesh) => mesh.uvs1 !== undefined)).toHaveLength(1);
        for (const panel of UV_BOARD_PANELS) {
            const material = recorded.materials.find((candidate) => candidate.label === `panel-${panel.name}`)!;
            expect(material.albedoUv, panel.name).toEqual({
                channel: panel.uv.channel ?? 0,
                matrix: uvMappingMatrix(panel.uv),
                wrapU: panel.uv.wrapU ?? 'repeat',
                wrapV: panel.uv.wrapV ?? 'repeat',
            });
        }
    });

    it('passes alpha mode, opacity, metalness and a metalness map', () => {
        const base = buildTexturedFurnace();
        const scene: ProceduralScene = {
            ...base,
            materials: [{ ...base.materials[0], alphaMode: 'stochastic', opacity: 0.4, metalness: 0.3, metalnessMap: { texture: 0, channel: 'b', uv: { repeat: [2, 2] } } }],
        };
        const { api, recorded } = recordingSceneApi();
        describeProceduralScene(api, scene);
        const material = recorded.materials[0];
        expect([material.alphaMode, material.opacity, material.metalness]).toEqual(['stochastic', 0.4, 0.3]);
        expect(material.metalnessTexture?.channel).toBe('b');
        expect(material.metalnessTexture?.uv?.matrix).toEqual([2, -0, 0, 2, 0, 0]);
    });

    it('takes a metalness map with no metalness as metalness 1, as glTF, the GLB and three do', () => {
        const base = buildTexturedFurnace();
        const scene: ProceduralScene = { ...base, materials: [{ ...base.materials[0], metalnessMap: { texture: 0, channel: 'b' } }] };
        const { api, recorded } = recordingSceneApi();
        describeProceduralScene(api, scene);
        expect(recorded.materials[0].metalness).toBe(1);
        const plain = recordingSceneApi();
        describeProceduralScene(plain.api, base);
        expect(plain.recorded.materials[0].metalness).toBeUndefined();
    });

    it('throws, naming the field, on data it would otherwise drop', () => {
        const base = buildTexturedFurnace();
        const withUnknown = { ...base, materials: [{ ...base.materials[0], clearcoat: 1 } as never] };
        expect(() => describeProceduralScene(recordingSceneApi().api, withUnknown)).toThrow(/field 'clearcoat'/);
        const glass = buildGlassWall();
        const missingSlot = { ...glass, instances: glass.instances.map((instance) => (instance.materials ? { ...instance, materials: [instance.material] } : instance)) };
        expect(() => describeProceduralScene(recordingSceneApi().api, missingSlot)).toThrow(/uses 2 material slots/);
    });
});
