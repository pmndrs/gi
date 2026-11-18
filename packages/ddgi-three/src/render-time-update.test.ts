import { PerspectiveCamera, Scene } from 'three';
import { builtinAOContext, float, mrt, normalView, output, packNormalToRGB, pass } from 'three/tsl';
import { describe, expect, it, vi } from 'vitest';
import { GILimitError, type GIPlatform, type SceneApi, type ShadingModule } from '@pmndrs/ddgi-core';
import { flowContextData, hasCompositeMark } from './composite-mark.ts';
import { ddgi, giAlbedo } from './composite.ts';
import { GILight } from './gi-light.ts';
import { ThreeGILimitError } from './volume-start.ts';
import { DDGIVolume } from './ddgi-volume.ts';
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

    it("with three's animation loop, runs once per frame id, even across tasks (awaits between renders)", async () => {
        const light = new GILight(fakeShading());
        let runs = 0;
        light.onRender = () => runs++;
        const camera = new PerspectiveCamera();
        light.runRenderTimeUpdate(camera, 7);
        await endOfTask();
        light.runRenderTimeUpdate(camera, 7);
        expect(runs).toBe(1);
        light.runRenderTimeUpdate(camera, 8);
        expect(runs).toBe(2);
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

describe('DDGIVolume before start', () => {
    it('has no light for the composite yet, and says what to do', () => {
        const volume = new DDGIVolume(6, 5, 6, 6, 6, 6);
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

    it('by default runs no occlusion, on a full-resolution pass and on an upscaled (reduced) one (DECISIONS 2026-11-18)', () => {
        expect(ddgi(compositePass(), new DDGIVolume()).occlusionNode).toBeNull();
        const reduced = compositePass();
        reduced.setResolutionScale(0.5);
        expect(ddgi(reduced, new DDGIVolume()).occlusionNode).toBeNull();
    });

    it('makes no occlusion at the build either, whatever the pass scale is by then', () => {
        const scenePass = compositePass();
        const node = ddgi(scenePass, { light: new GILight(fakeShading()) });
        // The pass is reduced after ddgi(), as a host adding an upscaler later would.
        scenePass.setResolutionScale(0.5);
        node.setup({} as Parameters<typeof node.setup>[0]);
        expect(node.occlusionNode).toBeNull();
    });

    it('makes the 1 m GTAO a host opts in to, at the resolution it asks for', () => {
        const reduced = compositePass();
        reduced.setResolutionScale(0.5);
        const node = ddgi(reduced, new DDGIVolume(), { occlusion: { resolutionScale: 0.5 } });
        expect(node.occlusionNode).not.toBeNull();
        expect(node.occlusionNode!.radius.value).toBe(1);
        expect(node.occlusionNode!.resolutionScale).toBe(0.5);
    });

    it("sizes its GTAO from the scene pass (not the drawing buffer), with GTAO's own scale relative to the pass", () => {
        const scenePass = compositePass();
        const node = ddgi(scenePass, new DDGIVolume(), { occlusion: { resolutionScale: 0.5 } });
        const gtao = node.occlusionNode!;
        const aoSize = () => {
            const image = (gtao.getTextureNode() as unknown as { value: { image: { width: number; height: number } } }).value.image;
            return [image.width, image.height];
        };
        // GTAO's updateBefore passes the drawing buffer's size; under an upscaler the pass draws at a fraction of it.
        scenePass.setResolutionScale(0.5);
        gtao.setSize(2560, 1440);
        expect(aoSize()).toEqual([640, 360]);
        scenePass.setResolutionScale(1 / 1.5);
        gtao.setSize(2560, 1440);
        expect(aoSize()).toEqual([854, 480]);
        scenePass.setResolutionScale(1);
        gtao.setSize(2560, 1440);
        expect(aoSize()).toEqual([1280, 720]);
    });

    it('takes occlusion settings, or none', () => {
        const half = ddgi(compositePass(), new DDGIVolume(), { occlusion: { radius: 0.5, resolutionScale: 0.5, samples: 8 } });
        // An explicit choice stands on a full-resolution pass too.
        expect(half.occlusionNode!.radius.value).toBe(0.5);
        expect(half.occlusionNode!.resolutionScale).toBe(0.5);
        expect(half.occlusionNode!.samples.value).toBe(8);
        expect(ddgi(compositePass(), new DDGIVolume(), { occlusion: false }).occlusionNode).toBeNull();
    });

    it("uses the pipeline's AO instead of making a second GTAO pass", () => {
        const source = { getTextureNode: () => compositePass().getTextureNode() };
        expect(ddgi(compositePass(), new DDGIVolume(), { occlusion: { source } }).occlusionNode).toBeNull();
    });

    it('merges its mark back when the pass gets a new context after ddgi(), keeping that context', () => {
        const scenePass = compositePass();
        const node = ddgi(scenePass, { light: new GILight(fakeShading()) }, { occlusion: false });
        const aoContext = builtinAOContext(float(0.5));
        scenePass.contextNode = aoContext;
        const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
        node.update();
        const data = flowContextData(scenePass.contextNode!);
        expect(hasCompositeMark(data)).toBe(true);
        expect(data.getAO).toBe(flowContextData(aoContext).getAO);
        expect(warn).toHaveBeenCalledTimes(1);
        warn.mockRestore();
    });

    it("warns when an occlusion source meets a pass that already applies AO (GI would get AO twice)", () => {
        const scenePass = compositePass();
        scenePass.contextNode = builtinAOContext(float(0.5));
        const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
        ddgi(scenePass, new DDGIVolume(), { occlusion: { source: { getTextureNode: () => compositePass().getTextureNode() } } });
        expect(warn).toHaveBeenCalledWith(expect.stringMatching(/AO twice/));
        warn.mockRestore();
    });

    it('adds no second occlusion by default when the pass already applies AO to its materials', () => {
        const scenePass = compositePass();
        scenePass.setResolutionScale(0.5);
        scenePass.contextNode = builtinAOContext(float(0.5));
        expect(ddgi(scenePass, new DDGIVolume()).occlusionNode).toBeNull();
    });
});

describe('ThreeGILimitError', () => {
    it('is a GILimitError with the limit, and gives the fix for three', () => {
        const error = new ThreeGILimitError(new GILimitError('maxStorageBufferBindingSize', 300_000_000, 134_217_728));
        expect(error).toBeInstanceOf(GILimitError);
        expect(error.code).toBe('GI_LIMIT');
        expect(error.limit).toBe('maxStorageBufferBindingSize');
        expect(error.needed).toBe(300_000_000);
        expect(error.message).toMatch(/maxStorageBufferBindingSize/);
        expect(error.message).toMatch(/createGIRenderer/);
        expect(error.message).toMatch(/@pmndrs\/gi\/core/);
    });
});
