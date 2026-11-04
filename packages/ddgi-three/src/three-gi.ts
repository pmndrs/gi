/**
 * `ThreeGI`: the GI platform driven from a three.js scene. It owns the
 * scene sync (scene-sync.ts) and the GI light (gi-light.ts), and runs one GI
 * frame per call to `update`.
 *
 * ```ts
 * const { renderer, device } = await createGIRenderer();
 * const gi = createGI(device);
 * gi.setOutput(gi.addTechnique(new DDGITechnique({ volume: { center, size, spacing } })));
 * await gi.prepare();
 * const threeGI = new ThreeGI(gi, scene);
 * renderer.setAnimationLoop((time) => {
 *     threeGI.update(camera, deltaSeconds);
 *     renderer.render(scene, camera);
 * });
 * ```
 *
 * **Command order in r187.** three makes and submits its own command
 * encoders, one or more per `render()` (WebGPUBackend `beginRender` /
 * `finishRender`), and offers no hook to record into them. So `update`
 * records the GI frame into an encoder of its own and submits it right away.
 * Queue submissions run in order, so the GI work of frame N always completes
 * before anything three submits afterwards, including frame N's
 * `renderer.render`, which samples the result. `queue.writeBuffer` calls the
 * platform makes inside `encode` land before that submit (NOTES, WebGPU
 * ordering). That is one extra submit per frame, the same kind three makes
 * for every render pass and compute call.
 *
 * Call `update` before `renderer.render`, once per frame. It reads the
 * world matrices three computed in the last `render`, so a moved object
 * reaches GI one frame later, which DDGI's blending over many frames hides.
 * Updating them here as well would make three do it twice per frame: 1 ms on
 * Bistro interior (2,062 meshes) on this laptop. `updateMatrixWorld: true`
 * does it anyway, for hosts that want no lag. The first `update` always
 * updates them, so GI never starts from matrices three hasn't computed yet.
 */

import type { Camera, OrthographicCamera, PerspectiveCamera, Scene } from 'three';
import type { EnvironmentDescriptor, FrameBudget, GIPlatform, ShadingModule } from '@pmndrs/ddgi-core';
import type { PmndrsSkyLike } from './environment.ts';
import { GILight, type GILightOptions } from './gi-light.ts';
import { SceneSync, type SyncFrameReport } from './scene-sync.ts';

export interface ThreeGIOptions {
    /** GPU work allowed per frame for GI updates. Default `{ gpuMs: 2 }`. A live object: change it any time. */
    readonly budget?: FrameBudget;
    /** Settings for the GI light added to the scene. */
    readonly light?: GILightOptions;
    /** An environment for GI rays instead of the scene's (environment.ts). */
    readonly environment?: EnvironmentDescriptor | null;
    /** A pmndrs/sky `Sky` to take GI's sky from (environment.ts). */
    readonly sky?: PmndrsSkyLike | null;
    /** Where warnings go (each distinct message once). Default `console.warn`. */
    readonly warn?: (message: string) => void;
    /** Update the scene's world matrices in every `update` (see the file comment). Default false. */
    readonly updateMatrixWorld?: boolean;
}

/** CPU time of the last `update`, milliseconds (`performance.now()`, so steps of up to 0.1 ms in some browsers). */
export interface ThreeGITimings {
    /** Updating world matrices (three's own work, done again by `render`): only with `updateMatrixWorld`, and on the first update. */
    matrixWorldMs: number;
    /** The scene sync: walking the scene and sending changes. This is the adapter's own cost. */
    syncMs: number;
    /** `gi.encode` and the submit: the platform's CPU work. */
    encodeMs: number;
}

const DEFAULT_BUDGET: FrameBudget = { gpuMs: 2 };

export class ThreeGI {
    readonly gi: GIPlatform;
    readonly scene: Scene;
    /**
     * The light that brings GI into three's materials. Toggle GI with
     * `light.intensity`. A new light replaces it when the platform's shading
     * module changes shape (see `followShadingModule`), so read it from here
     * rather than keeping it.
     */
    get light(): GILight {
        return this.currentLight;
    }
    readonly sync: SceneSync;
    budget: FrameBudget;
    readonly timings: ThreeGITimings = { matrixWorldMs: 0, syncMs: 0, encodeMs: 0 };
    /** What the last update sent. */
    get report(): SyncFrameReport {
        return this.sync.report;
    }

    private currentLight: GILight;
    private shadingSeen: { module: ShadingModule; version: number };
    private previousTime = -1;
    private readonly alwaysUpdateMatrixWorld: boolean;
    private updates = 0;

    /** The platform must be prepared (`await gi.prepare()`), because the light wraps the technique's textures. */
    constructor(gi: GIPlatform, scene: Scene, options: ThreeGIOptions = {}) {
        if (!gi.isReady) throw new Error('@pmndrs/ddgi/three: prepare the platform first (await gi.prepare()), then make ThreeGI.');
        this.gi = gi;
        this.scene = scene;
        this.budget = options.budget ?? DEFAULT_BUDGET;
        this.alwaysUpdateMatrixWorld = options.updateMatrixWorld ?? false;
        this.sync = new SceneSync(gi.scene, { device: gi.device, warn: options.warn });
        this.sync.environment = options.environment ?? null;
        this.sync.sky = options.sky ?? null;
        const shading = gi.shading;
        this.shadingSeen = { module: shading, version: shading.version };
        this.currentLight = new GILight(shading, options.light);
        this.currentLight.name = 'GI';
        scene.add(this.currentLight);
    }

    /** An environment for GI rays instead of the scene's, or null to read the scene's. */
    set environment(environment: EnvironmentDescriptor | null) {
        this.sync.environment = environment;
    }

    get environment(): EnvironmentDescriptor | null {
        return this.sync.environment;
    }

    /** A pmndrs/sky `Sky` to take GI's sky from, or null. */
    set sky(sky: PmndrsSkyLike | null) {
        this.sync.sky = sky;
    }

    get sky(): PmndrsSkyLike | null {
        return this.sync.sky;
    }

    /**
     * One GI frame: world matrices, the scene sync, then the platform's
     * `encode`, submitted before three renders (see the file comment).
     * `deltaTime` in seconds; omitted, it's measured between calls.
     */
    update(camera: Camera, deltaTime?: number): void {
        const start = performance.now();
        if ((this.alwaysUpdateMatrixWorld || this.updates === 0) && this.scene.matrixWorldAutoUpdate) this.scene.updateMatrixWorld();
        // The camera alone is cheap, and GI's update order follows it.
        if (camera.parent === null && camera.matrixWorldAutoUpdate) camera.updateMatrixWorld();
        this.updates++;
        const afterMatrices = performance.now();

        this.followShadingModule();
        // Coverage is worth its cost only for the environment-diffuse workaround or to share the probe slot.
        this.light.computeCoverage = (this.light.replaceEnvironmentDiffuse && this.scene.environment !== null) || this.sync.otherProbeLights > 0;
        this.sync.update(this.scene);
        const afterSync = performance.now();

        const seconds = deltaTime ?? (this.previousTime < 0 ? 0 : (start - this.previousTime) / 1000);
        this.previousTime = start;
        const device = this.gi.device;
        const encoder = device.createCommandEncoder({ label: 'ddgi-three frame' });
        this.gi.encode(encoder, { deltaTime: seconds, budget: this.budget, camera: frameCamera(camera) });
        device.queue.submit([encoder.finish()]);
        const end = performance.now();

        this.timings.matrixWorldMs = afterMatrices - start;
        this.timings.syncMs = afterSync - afterMatrices;
        this.timings.encodeMs = end - afterSync;
    }

    /** Removes the light and everything the sync created. The platform itself is the host's to dispose. */
    dispose(): void {
        this.light.removeFromParent();
        this.light.dispose();
        this.sync.dispose();
    }

    /**
     * When the platform's output module changes shape (a new `version`, or
     * another technique), the light's shader code must be rebuilt. three
     * caches compiled shaders by material properties and light ids, so
     * `material.needsUpdate` alone would reuse the old shader: the light is
     * replaced by a new one (a new id), with the same settings, which
     * recompiles every lit material once. Texture reallocation alone doesn't
     * count; the light re-reads those each render.
     */
    private followShadingModule(): void {
        const shading = this.gi.shading;
        if (shading === this.shadingSeen.module && shading.version === this.shadingSeen.version) return;
        this.shadingSeen = { module: shading, version: shading.version };
        const old = this.currentLight;
        const next = new GILight(shading, { intensity: old.intensity, priority: old.priority, replaceEnvironmentDiffuse: old.replaceEnvironmentDiffuse });
        next.name = old.name;
        (old.parent ?? this.scene).add(next);
        old.removeFromParent();
        old.dispose();
        this.currentLight = next;
    }
}

/** The platform's plain camera from a three camera (matrices current). */
function frameCamera(camera: Camera): { position: [number, number, number]; view: ArrayLike<number>; projection: ArrayLike<number>; near: number; far: number } {
    const e = camera.matrixWorld.elements;
    const withPlanes = camera as PerspectiveCamera | OrthographicCamera;
    return {
        position: [e[12], e[13], e[14]],
        view: camera.matrixWorldInverse.elements,
        projection: camera.projectionMatrix.elements,
        near: typeof withPlanes.near === 'number' ? withPlanes.near : 0.1,
        far: typeof withPlanes.far === 'number' ? withPlanes.far : 1000,
    };
}
