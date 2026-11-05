import { PerspectiveCamera, Scene } from 'three';
import { builtinAOContext, float, mrt, normalView, output, packNormalToRGB, pass } from 'three/tsl';
import { describe, expect, it } from 'vitest';
import type { GIPlatform, SceneApi, ShadingModule } from '@pmndrs/ddgi-core';
import { ddgi, giAlbedo } from './composite.ts';
import { GILight } from './gi-light.ts';
import { GIVolume } from './gi-volume.ts';
import { ThreeGI } from './three-gi.ts';

function fakeShading(): ShadingModule {
    return { techniqueId: 'fake', code: '', parameters: [], provides: { radiance: false }, globalsCode: () => '', writeUniforms: () => {}, version: 1 };
}

/** Just enough of a platform for ThreeGI: counts encodes. */
function fakePlatform() {
    let encodes = 0;
    let handle = 1;
    const gi = {
        isReady: true,
        device: { createCommandEncoder: () => ({ finish: () => ({}) }), queue: { submit: () => {} } },
        scene: new Proxy({}, { get: () => () => handle++ }) as unknown as SceneApi,
        shading: fakeShading(),
        encode: () => encodes++,
    } as unknown as GIPlatform;
    return { gi, encodes: () => encodes };
}

/** Lets queued microtasks run: the end of a JavaScript task, as far as the guard can tell. */
const endOfTask = () => new Promise((resolve) => setTimeout(resolve, 0));

describe('the render-time update (GILight.runRenderTimeUpdate)', () => {
    it('runs once per JavaScript task, however many lit renders the task draws', async () => {
        const light = new GILight(fakeShading());
        const cameras: unknown[] = [];
        light.onRender = (camera) => cameras.push(camera);
        const camera = new PerspectiveCamera();
        light.runRenderTimeUpdate(camera);
        light.runRenderTimeUpdate(new PerspectiveCamera());
        expect(cameras).toEqual([camera]);
        await endOfTask();
        light.runRenderTimeUpdate(camera);
        expect(cameras).toHaveLength(2);
    });

    it('does nothing without a hook or a camera', () => {
        const light = new GILight(fakeShading());
        expect(() => light.runRenderTimeUpdate(new PerspectiveCamera())).not.toThrow();
        let runs = 0;
        light.onRender = () => runs++;
        light.runRenderTimeUpdate(null);
        expect(runs).toBe(0);
    });

    it('reports an error once and stops, so the render carries on', async () => {
        const light = new GILight(fakeShading());
        const errors: unknown[] = [];
        let runs = 0;
        light.onRender = () => {
            runs++;
            throw new Error('boom');
        };
        light.onRenderError = (error) => errors.push(error);
        expect(() => light.runRenderTimeUpdate(new PerspectiveCamera())).not.toThrow();
        await endOfTask();
        light.runRenderTimeUpdate(new PerspectiveCamera());
        expect(runs).toBe(1);
        expect(errors).toHaveLength(1);
        expect(light.renderUpdateFailed).toBe(true);
    });
});

describe('ThreeGI with updateOnRender', () => {
    it('runs a GI frame from its light, with the render camera', () => {
        const platform = fakePlatform();
        const threeGI = new ThreeGI(platform.gi, new Scene(), { updateOnRender: true });
        threeGI.light.runRenderTimeUpdate(new PerspectiveCamera());
        expect(platform.encodes()).toBe(1);
    });

    it('can be switched on and off; off, the light has no hook', () => {
        const platform = fakePlatform();
        const threeGI = new ThreeGI(platform.gi, new Scene());
        expect(threeGI.light.onRender).toBeNull();
        threeGI.updateOnRender = true;
        expect(threeGI.light.onRender).not.toBeNull();
        threeGI.updateOnRender = false;
        expect(threeGI.light.onRender).toBeNull();
    });
});

describe('GIVolume before start', () => {
    it('has no light for the composite yet, and says what to do', () => {
        const volume = new GIVolume(6, 5, 6, 6, 6, 6);
        expect(volume.isStarted).toBe(false);
        expect(volume.threeGI).toBeNull();
        expect(() => volume.light).toThrow(/volume\.start/);
        expect(() => volume.recalibrate()).not.toThrow();
    });
});

describe('the composite node, ddgi(scenePass, volume)', () => {
    function compositePass() {
        const scenePass = pass(new Scene(), new PerspectiveCamera());
        scenePass.setMRT(mrt({ output, giAlbedo, normal: packNormalToRGB(normalView) }));
        return scenePass;
    }

    it('runs its own GTAO on the GI term by default, at 1 m', () => {
        const node = ddgi(compositePass(), new GIVolume());
        expect(node.occlusionNode).not.toBeNull();
        expect(node.occlusionNode!.radius.value).toBe(1);
    });

    it('takes occlusion settings, or none', () => {
        const half = ddgi(compositePass(), new GIVolume(), { occlusion: { radius: 0.5, resolutionScale: 0.5, samples: 8 } });
        expect(half.occlusionNode!.radius.value).toBe(0.5);
        expect(half.occlusionNode!.resolutionScale).toBe(0.5);
        expect(half.occlusionNode!.samples.value).toBe(8);
        expect(ddgi(compositePass(), new GIVolume(), { occlusion: false }).occlusionNode).toBeNull();
    });

    it("uses the pipeline's AO instead of making a second GTAO pass", () => {
        const source = { getTextureNode: () => compositePass().getTextureNode() };
        expect(ddgi(compositePass(), new GIVolume(), { occlusion: { source } }).occlusionNode).toBeNull();
    });

    it('adds no second occlusion when the pass already applies AO to its materials', () => {
        const scenePass = compositePass();
        scenePass.contextNode = builtinAOContext(float(0.5));
        expect(ddgi(scenePass, new GIVolume()).occlusionNode).toBeNull();
    });
});
