import { BoxGeometry, Color, Mesh, MeshStandardMaterial, PerspectiveCamera, PointLight, Scene, Texture } from 'three';
import { describe, expect, it } from 'vitest';
import type { FrameInput, GIPlatform, SceneApi, ShadingModule } from '@pmndrs/ddgi-core';
import { GIVolume } from './gi-volume.ts';
import { ThreeGI } from './three-gi.ts';

/** A shading module whose shape can be changed, like a technique switching its code. */
function fakeShading(): ShadingModule & { version: number } {
    return {
        techniqueId: 'fake',
        code: 'fn giIrradiance() -> vec3f { return vec3f(0.0); }',
        parameters: [],
        provides: { radiance: false },
        globalsCode: () => '',
        writeUniforms: () => {},
        version: 1,
    };
}

/** Just enough of a platform for ThreeGI: a scene API that accepts everything, and a device that records submits. */
function fakePlatform(shading: ShadingModule) {
    const frames: FrameInput[] = [];
    let submits = 0;
    let handle = 1;
    const scene = new Proxy({}, { get: () => () => handle++ }) as unknown as SceneApi;
    const device = {
        createCommandEncoder: () => ({ finish: () => ({}) }),
        queue: { submit: () => submits++ },
    } as unknown as GPUDevice;
    const gi = {
        isReady: true,
        device,
        scene,
        get shading() {
            return current;
        },
        encode: (_encoder: GPUCommandEncoder, frame: FrameInput) => frames.push(frame),
    } as unknown as GIPlatform;
    let current = shading;
    return {
        gi,
        frames,
        submits: () => submits,
        setShading: (next: ShadingModule) => (current = next),
    };
}

describe('ThreeGI', () => {
    it('copies the scene in while the platform prepares, and adds its light at the first update once it is ready (T2.14)', () => {
        const platform = fakePlatform(fakeShading());
        (platform.gi as { isReady: boolean }).isReady = false;
        const scene = new Scene();
        scene.add(new Mesh(new BoxGeometry(), new MeshStandardMaterial()));
        const threeGI = new ThreeGI(platform.gi, scene);
        // The light wraps the technique's textures, which don't exist yet: a render meanwhile draws without GI.
        expect(scene.children).not.toContain(threeGI.light);
        const camera = new PerspectiveCamera();
        threeGI.update(camera, 0);
        expect(threeGI.sync.counts.instances).toBe(1);
        expect(platform.frames).toHaveLength(0);
        expect(platform.submits()).toBe(0);
        expect(scene.children).not.toContain(threeGI.light);

        (platform.gi as { isReady: boolean }).isReady = true;
        threeGI.update(camera, 0);
        expect(scene.children).toContain(threeGI.light);
        expect(platform.frames).toHaveLength(1);
        expect(platform.submits()).toBe(1);
    });

    it('adds its light, encodes once per update with the camera, and submits before three renders', () => {
        const platform = fakePlatform(fakeShading());
        const scene = new Scene();
        scene.add(new Mesh(new BoxGeometry(), new MeshStandardMaterial()));
        const threeGI = new ThreeGI(platform.gi, scene, { budget: { units: 7 } });
        expect(scene.children).toContain(threeGI.light);
        const camera = new PerspectiveCamera();
        camera.position.set(1, 2, 3);
        threeGI.update(camera, 0.5);
        expect(platform.frames).toHaveLength(1);
        expect(platform.submits()).toBe(1);
        expect(platform.frames[0].camera.position).toEqual([1, 2, 3]);
        expect(platform.frames[0].budget).toEqual({ units: 7 });
        expect(platform.frames[0].deltaTime).toBe(0.5);
        expect(threeGI.sync.counts.instances).toBe(1);
    });

    it('replaces its light when the shading module changes shape, keeping its settings', () => {
        const shading = fakeShading();
        const platform = fakePlatform(shading);
        const scene = new Scene();
        const threeGI = new ThreeGI(platform.gi, scene, { light: { priority: 3 } });
        const camera = new PerspectiveCamera();
        threeGI.update(camera);
        const first = threeGI.light;
        first.intensity = 0.25;

        // Same module, same version: nothing to do.
        threeGI.update(camera);
        expect(threeGI.light).toBe(first);

        // A new version: three caches shaders by light id, so a new light (new id) carries the new code.
        shading.version = 2;
        threeGI.update(camera);
        const second = threeGI.light;
        expect(second).not.toBe(first);
        expect(second.id).not.toBe(first.id);
        expect(second.shading).toBe(shading);
        expect(second.intensity).toBe(0.25);
        expect(second.priority).toBe(3);
        expect(first.parent).toBeNull();
        expect(scene.children).toContain(second);

        // Another technique's module.
        const other = fakeShading();
        platform.setShading(other);
        threeGI.update(camera);
        expect(threeGI.light).not.toBe(second);
        expect(threeGI.light.shading).toBe(other);
        expect(scene.children.filter((child) => (child as { isGILight?: boolean }).isGILight)).toHaveLength(1);
    });

    it('has the GI light follow the intensity and priority of a GIVolume', () => {
        const platform = fakePlatform(fakeShading());
        const scene = new Scene();
        const volume = new GIVolume(4, 4, 4);
        scene.add(volume);
        const threeGI = new ThreeGI(platform.gi, scene, { volume });
        volume.intensity = 0.5;
        volume.priority = 7;
        threeGI.update(new PerspectiveCamera());
        expect(threeGI.light.intensity).toBe(0.5);
        expect(threeGI.light.priority).toBe(7);
        expect(threeGI.sync.counts.lights).toBe(0);
    });

    it('computes coverage for a material with its own envMap, without scene.environment', () => {
        const platform = fakePlatform(fakeShading());
        const scene = new Scene();
        scene.add(new Mesh(new BoxGeometry(), new MeshStandardMaterial({ envMap: new Texture() })));
        const threeGI = new ThreeGI(platform.gi, scene);
        threeGI.update(new PerspectiveCamera());
        expect(threeGI.light.computeCoverage).toBe(true);
    });

    it('turns coverage on only when something needs it', () => {
        const platform = fakePlatform(fakeShading());
        const scene = new Scene();
        const threeGI = new ThreeGI(platform.gi, scene);
        const camera = new PerspectiveCamera();
        threeGI.update(camera);
        expect(threeGI.light.computeCoverage).toBe(false);
        scene.environment = new Texture();
        threeGI.update(camera);
        expect(threeGI.light.computeCoverage).toBe(true);
    });

    it('counts a LightProbeGrid for coverage only when the camera sees its layer', () => {
        const platform = fakePlatform(fakeShading());
        const scene = new Scene();
        const grid = Object.assign(new PointLight(), { isLightProbeGrid: true });
        grid.layers.set(3);
        scene.add(grid);
        const threeGI = new ThreeGI(platform.gi, scene);
        const camera = new PerspectiveCamera();
        threeGI.update(camera);
        expect(threeGI.light.computeCoverage).toBe(false);
        camera.layers.enable(3);
        threeGI.update(camera);
        expect(threeGI.light.computeCoverage).toBe(true);
        // The GI light itself lights every layer.
        expect(threeGI.light.layers.test(camera.layers)).toBe(true);
    });

    it("sets light group weights on the output technique, from three's colours in linear space times an intensity", () => {
        const platform = fakePlatform(fakeShading());
        const calls: Array<[string, number | null, readonly number[]]> = [];
        const technique = {
            id: 'ddgi',
            setGroupWeight: (group: number, weight: readonly number[]) => calls.push(['group', group, weight]),
            setEnvironmentWeight: (weight: readonly number[]) => calls.push(['environment', null, weight]),
        };
        Object.assign(platform.gi, { outputTechnique: technique });
        const threeGI = new ThreeGI(platform.gi, new Scene());
        threeGI.setGroupWeight(1, 'white', 0.25);
        // '#808080' is sRGB: three keeps colours linear, so 0.5 encoded is about 0.216 linear.
        threeGI.setGroupWeight(0, new Color('#808080'));
        threeGI.setEnvironmentWeight(0xff0000, 2);
        expect(calls[0]).toEqual(['group', 1, [0.25, 0.25, 0.25]]);
        expect(calls[1][2][0]).toBeCloseTo(0.2158605, 6);
        expect(calls[2]).toEqual(['environment', null, [2, 0, 0]]);
        // A technique without light groups is refused with a message that says so.
        Object.assign(platform.gi, { outputTechnique: { id: 'voxel' } });
        expect(() => threeGI.setGroupWeight(0, 'white')).toThrow(/voxel.*light group weights/);
    });

    it('removes everything it added on dispose', () => {
        const platform = fakePlatform(fakeShading());
        const scene = new Scene();
        scene.add(new Mesh(new BoxGeometry(), new MeshStandardMaterial()));
        const threeGI = new ThreeGI(platform.gi, scene);
        threeGI.update(new PerspectiveCamera());
        threeGI.dispose();
        expect(scene.children).toHaveLength(1);
        expect(threeGI.sync.counts.instances).toBe(0);
    });
});
