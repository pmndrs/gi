import { BoxGeometry, Color, Matrix4, Mesh, MeshStandardMaterial, OrthographicCamera, PerspectiveCamera, PointLight, Scene, Texture, Vector3, type Camera } from 'three';
import { describe, expect, it } from 'vitest';
import type { FrameInput, GIPlatform, SceneApi, ShadingModule } from '@pmndrs/ddgi-core';
import { DDGIVolume } from './ddgi-volume.ts';
import { jitterPixelsToNdc, ThreeGI, upscalerJitter } from './three-gi.ts';

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
    it('refuses a platform that is not prepared and has no prepare promise', () => {
        const platform = fakePlatform(fakeShading());
        (platform.gi as { isReady: boolean }).isReady = false;
        expect(() => new ThreeGI(platform.gi, new Scene())).toThrow(/prepare/);
    });

    it('copies the scene in while the platform prepares, and adds its light once it is ready (T2.14)', async () => {
        const platform = fakePlatform(fakeShading());
        (platform.gi as { isReady: boolean }).isReady = false;
        let prepared: () => void = () => {};
        const preparing = new Promise<void>((resolve) => (prepared = resolve));
        const scene = new Scene();
        scene.add(new Mesh(new BoxGeometry(), new MeshStandardMaterial()));
        const threeGI = new ThreeGI(platform.gi, scene, { preparing });
        // The light wraps the technique's textures, which don't exist yet: a render meanwhile draws without GI.
        expect(scene.children).not.toContain(threeGI.light);
        const camera = new PerspectiveCamera();
        threeGI.update(camera, 0);
        expect(threeGI.sync.counts.instances).toBe(1);
        expect(platform.frames).toHaveLength(0);
        expect(platform.submits()).toBe(0);
        expect(scene.children).not.toContain(threeGI.light);

        (platform.gi as { isReady: boolean }).isReady = true;
        prepared();
        await preparing;
        expect(scene.children).toContain(threeGI.light);
        threeGI.update(camera, 0);
        expect(platform.frames).toHaveLength(1);
        expect(platform.submits()).toBe(1);
    });

    it('gives GI to a host that only renders (updateOnRender), with no update call of its own', async () => {
        const platform = fakePlatform(fakeShading());
        (platform.gi as { isReady: boolean }).isReady = false;
        let prepared: () => void = () => {};
        const preparing = new Promise<void>((resolve) => (prepared = resolve));
        const scene = new Scene();
        scene.add(new Mesh(new BoxGeometry(), new MeshStandardMaterial()));
        const threeGI = new ThreeGI(platform.gi, scene, { preparing, updateOnRender: true });
        expect(scene.children).not.toContain(threeGI.light);

        (platform.gi as { isReady: boolean }).isReady = true;
        prepared();
        await preparing;
        // The light is in the scene, so three's next render runs its render-time update.
        expect(scene.children).toContain(threeGI.light);
        threeGI.light.runRenderTimeUpdate(new PerspectiveCamera());
        expect(platform.frames).toHaveLength(1);
        expect(threeGI.sync.counts.instances).toBe(1);
    });

    it("throws prepare's failure at the next update and warns once", async () => {
        const platform = fakePlatform(fakeShading());
        (platform.gi as { isReady: boolean }).isReady = false;
        const warnings: string[] = [];
        const preparing = Promise.reject(new Error('kernel failed to compile'));
        const threeGI = new ThreeGI(platform.gi, new Scene(), { preparing, warn: (message) => warnings.push(message) });
        await preparing.catch(() => {});
        await Promise.resolve();
        expect(() => threeGI.update(new PerspectiveCamera(), 0)).toThrow(/kernel failed to compile/);
        expect(warnings.some((message) => message.includes('kernel failed to compile'))).toBe(true);
        expect(platform.frames).toHaveLength(0);
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

    it('has the GI light follow the intensity and priority of a DDGIVolume', () => {
        const platform = fakePlatform(fakeShading());
        const scene = new Scene();
        const volume = new DDGIVolume(4, 4, 4);
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

    it("hands the platform the host's unjittered projection and its jitter (FrameCamera.jitter, RC2.2)", () => {
        const platform = fakePlatform(fakeShading());
        const threeGI = new ThreeGI(platform.gi, new Scene());
        const camera = new PerspectiveCamera(60, 16 / 9, 0.1, 100);
        camera.position.set(1, 2, 3);
        camera.updateMatrixWorld();
        threeGI.update(camera, 0);
        // No jitter source: no jitter, and the camera's own matrices.
        expect(platform.frames[0].camera.jitter).toBeUndefined();
        expect(Array.from(platform.frames[0].camera.projection)).toEqual(camera.projectionMatrix.elements);
        expect(threeGI.encodedFrames).toBe(1);

        // An upscaler's frame: beginFrame keeps the unjittered projection, then jitters the camera.
        const unjittered = camera.projectionMatrix.clone();
        const upscaler = { jitter: { x: 0.25, y: -0.125 }, unjitteredProjectionMatrix: unjittered };
        upscalerViewOffset(camera, upscaler.jitter.x, upscaler.jitter.y, 1280, 720);
        threeGI.jitter = upscalerJitter(upscaler, 1280, 720);
        threeGI.update(camera, 0);
        expect(platform.frames[1].camera.jitter).toEqual([-0.5 / 1280, -0.25 / 720]);
        // Not the camera's (jittered) projection: the jitter would count twice.
        expect(Array.from(platform.frames[1].camera.projection)).toEqual(unjittered.elements);
        expect(Array.from(platform.frames[1].camera.projection)).not.toEqual(camera.projectionMatrix.elements);
        expect(platform.frames[1].camera.position).toEqual([1, 2, 3]);
        expect(threeGI.encodedFrames).toBe(2);
    });

    // The convention against three's own setViewOffset, as @pmndrs/upscaler's beginFrame applies it
    // (dist/index.js, the view offset helper), not against our own formula.
    for (const kind of ['perspective', 'orthographic'] as const) {
        it(`matches three's setViewOffset on a ${kind} camera: jitterPixelsToNdc moves projected points as the upscaler's jitter does`, () => {
            const width = 1280;
            const height = 720;
            const camera = kind === 'perspective' ? new PerspectiveCamera(55, width / height, 0.1, 100) : new OrthographicCamera(-8, 8, 4.5, -4.5, 0.1, 100);
            camera.position.set(0.5, 1, 6);
            camera.lookAt(0, 0.3, 0);
            camera.updateMatrixWorld();
            camera.updateProjectionMatrix();
            const unjittered = camera.projectionMatrix.clone();
            const points = [new Vector3(0, 0, 0), new Vector3(1.5, -0.7, 2), new Vector3(-2, 1.2, -3)];
            for (const [x, y] of [[0.4, 0.3], [-0.25, 0.5], [0.5, -0.5]] as const) {
                upscalerViewOffset(camera, x, y, width, height);
                const jitter = jitterPixelsToNdc({ x, y }, width, height);
                // Three moves each point on screen by the NDC jitter...
                for (const point of points) {
                    const before = point.clone().applyMatrix4(camera.matrixWorldInverse).applyMatrix4(unjittered);
                    const after = point.clone().applyMatrix4(camera.matrixWorldInverse).applyMatrix4(camera.projectionMatrix);
                    expect(after.x - before.x).toBeCloseTo(jitter[0], 9);
                    expect(after.y - before.y).toBeCloseTo(jitter[1], 9);
                }
                // ...and its jittered projection is FrameCamera's translate(jitter) x projection.
                const translate = new Matrix4().makeTranslation(jitter[0], jitter[1], 0);
                const expected = translate.multiply(unjittered).elements;
                camera.projectionMatrix.elements.forEach((value, index) => expect(value).toBeCloseTo(expected[index], 9));
                camera.clearViewOffset();
            }
        });
    }

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

/** @pmndrs/upscaler's beginFrame on a camera with no view offset: three's view offset by the jitter in render pixels. */
function upscalerViewOffset(camera: Camera & { setViewOffset(fullWidth: number, fullHeight: number, x: number, y: number, width: number, height: number): void }, x: number, y: number, width: number, height: number): void {
    camera.setViewOffset(width, height, x, y, width, height);
}
