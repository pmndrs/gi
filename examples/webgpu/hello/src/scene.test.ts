/**
 * The hello example describes its scenes to the platform the way an adapter
 * should: every material slot, every texture, every material idea GI uses
 * (docs/guides/adapter.md). Checked against `describeProceduralScene`
 * (scenes/src/platform-scene.ts), the repository's own full description that
 * the GPU test pages use, which throws on any scene field it doesn't pass on.
 */

import { describe, expect, it } from 'vitest';
import type { InstanceDescriptor, MaterialDescriptor, MeshDescriptor, SceneApi, TextureDescriptor } from '../../../../packages/ddgi-core/src/platform/contract/scene';
import { buildProceduralParts, mergeScenes } from '../../../../scenes/src/build-scene.ts';
import { describeProceduralScene } from '../../../../scenes/src/platform-scene.ts';
import { describeScene, loadManifest, supportedSceneIds } from './scene.ts';

interface Recorded {
    textures: TextureDescriptor[];
    meshes: MeshDescriptor[];
    materials: MaterialDescriptor[];
    instances: InstanceDescriptor[];
}

/** A scene API whose handles are the index of what was created, and which ignores lights and the environment. */
function recordingSceneApi(): { api: SceneApi; recorded: Recorded } {
    const recorded: Recorded = { textures: [], meshes: [], materials: [], instances: [] };
    const create =
        <T>(list: T[]) =>
        (descriptor: T) => {
            list.push(descriptor);
            return list.length - 1;
        };
    const api = {
        createTexture: create(recorded.textures),
        createMesh: create(recorded.meshes),
        createMaterial: create(recorded.materials),
        createInstance: create(recorded.instances),
        createLight: () => 0,
        setEnvironment: () => {},
    } as unknown as SceneApi;
    return { api, recorded };
}

/** A material with the contract's defaults filled in, so an omitted field and its default compare equal. */
function withMaterialDefaults(material: MaterialDescriptor): MaterialDescriptor {
    return { metalness: 0, blocksLight: true, doubleSided: true, ...material };
}

describe("the hello example's scene description", () => {
    it('can show the textured and multi-material scenes', () => {
        expect(supportedSceneIds()).toEqual(expect.arrayContaining(['alpha-checker', 'glass-wall', 'textured-furnace', 'cornell-box']));
    });

    for (const sceneId of supportedSceneIds()) {
        it(`passes ${sceneId} as the full description does`, () => {
            const manifest = loadManifest(sceneId);
            const hello = recordingSceneApi();
            describeScene(hello.api, manifest);
            const full = recordingSceneApi();
            describeProceduralScene(full.api, mergeScenes(buildProceduralParts(manifest)));

            expect(hello.recorded.textures).toEqual(full.recorded.textures);
            expect(hello.recorded.meshes).toEqual(full.recorded.meshes);
            expect(hello.recorded.materials.map(withMaterialDefaults)).toEqual(full.recorded.materials.map(withMaterialDefaults));
            expect(hello.recorded.instances).toEqual(full.recorded.instances);
        });
    }

    it('gives the glass wall two material slots and the textured furnace its textures', () => {
        const glass = recordingSceneApi();
        describeScene(glass.api, loadManifest('glass-wall'));
        expect(glass.recorded.instances.some((instance) => instance.materials.length === 2)).toBe(true);
        expect(glass.recorded.materials.some((material) => material.blocksLight === false)).toBe(true);

        const furnace = recordingSceneApi();
        describeScene(furnace.api, loadManifest('textured-furnace'));
        expect(furnace.recorded.textures.length).toBeGreaterThan(0);
        expect(furnace.recorded.materials.some((material) => material.albedoTexture !== undefined)).toBe(true);
    });

    it("draws a textured material with its texture's mean colour and a metal with no diffuse colour", () => {
        const described = describeScene(recordingSceneApi().api, loadManifest('textured-furnace'));
        described.data.materials.forEach((material, index) => {
            const drawn = described.rendererMaterials[index];
            for (let channel = 0; channel < 3; channel++) expect(drawn.albedo[channel]).toBeLessThanOrEqual(material.albedo[channel] + 1e-9);
            if (!material.albedoMap && !material.metalness && !material.metalnessMap) expect(drawn.albedo).toEqual(material.albedo);
        });
    });
});
