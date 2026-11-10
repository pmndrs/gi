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
 * **Copying the scene while kernels compile** (T2.14). `ThreeGI` may also be
 * made while the platform is still preparing, if it is given `prepare`'s
 * promise (`preparing`). Until that resolves, `update` only copies the scene
 * into the platform (the scene sync's CPU work), and the GI light waits
 * outside the scene, so a render meanwhile draws without GI. When it
 * resolves the light joins the scene, so a host that only renders
 * (`updateOnRender`) gets GI too; if it rejects, the next `update` throws
 * its error. Without `preparing` the platform must already be ready, so a
 * host that forgot `prepare` hears so at once. `DDGIVolume.start` does it
 * this way:
 *
 * ```ts
 * const preparing = gi.prepare();
 * // Chrome sends the compile requests when this JavaScript task ends.
 * await new Promise((resolve) => setTimeout(resolve, 0));
 * const threeGI = new ThreeGI(gi, scene, { preparing });
 * threeGI.update(camera, 0); // the scene copy, while the kernels compile
 * await preparing;
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
 * With `updateOnRender` (what `DDGIVolume.start` uses), there is no per-frame
 * call: the GI light's node runs `update` from inside three's render, once
 * per frame, with that render's camera (gi-light.ts, "The render-time
 * update", lists what that means). Otherwise call `update` before
 * `renderer.render`, once per frame. It reads the
 * world matrices three computed in the last `render`, so a moved object
 * reaches GI one frame later, which DDGI's blending over many frames hides.
 * Updating them here as well would make three do it twice per frame: 1 ms on
 * Bistro interior (2,062 meshes) on this laptop. `updateMatrixWorld: true`
 * does it anyway, for hosts that want no lag. The first `update` always
 * updates them, so GI never starts from matrices three hasn't computed yet.
 */

import { Color, type Camera, type ColorRepresentation, type Object3D, type OrthographicCamera, type PerspectiveCamera, type Scene } from 'three';
import type { EnvironmentDescriptor, FrameBudget, GIPlatform, ShadingModule } from '@pmndrs/ddgi-core';
import type { PmndrsSkyLike } from './environment.ts';
import { GILight, type GILightOptions } from './gi-light.ts';
import type { DDGIVolume } from './ddgi-volume.ts';
import { SceneSync, type SyncFrameReport } from './scene-sync.ts';
import type { WebGPURenderer } from 'three/webgpu';

export interface ThreeGIOptions {
    /** GPU work allowed per frame for GI updates. Default `{ gpuMs: 2 }`. A live object: change it any time. */
    readonly budget?: FrameBudget;
    /** Settings for the GI light added to the scene. */
    readonly light?: GILightOptions;
    /** A LightProbeGrid-shaped volume (ddgi-volume.ts) whose `intensity` and `priority` the GI light follows each frame. */
    readonly volume?: DDGIVolume | null;
    /** An environment for GI rays instead of the scene's (environment.ts). */
    readonly environment?: EnvironmentDescriptor | null;
    /** A pmndrs/sky `Sky` to take GI's sky from (environment.ts). */
    readonly sky?: PmndrsSkyLike | null;
    /** Where warnings go (each distinct message once). Default `console.warn`. */
    readonly warn?: (message: string) => void;
    /** Update the scene's world matrices in every `update` (see the file comment). Default false. */
    readonly updateMatrixWorld?: boolean;
    /** Objects the scene sync's walk looks at per frame (scene-sync.ts, `DEFAULT_OBJECTS_PER_FRAME`). */
    readonly objectsPerFrame?: number;
    /** Geometries, and separately materials, checked for edits per frame (scene-sync.ts, `DEFAULT_EDIT_CHECKS_PER_FRAME`). */
    readonly editChecksPerFrame?: number;
    /** Lights past the first 64 checked per frame, round robin (scene-sync.ts, `DEFAULT_LIGHTS_PER_FRAME`). */
    readonly lightsPerFrame?: number;
    /** Run `update` from the GI light's render-time update instead of a call per frame (see the file comment). Default false. */
    readonly updateOnRender?: boolean;
    /**
     * The promise of the platform's `prepare()` still running: `ThreeGI` may
     * then be made before it resolves, and copies the scene in meanwhile
     * (file comment, T2.14). Without it the platform must be ready.
     */
    readonly preparing?: Promise<void>;
    /**
     * The renderer that draws the scene (on the platform's device). Given,
     * skinned meshes that contribute to GI (`userData.giContributes`) are
     * posed on the GPU by three's own skinning, and no position reaches the
     * CPU (gpu-skinning.ts). Without it they are posed on the CPU.
     */
    readonly renderer?: WebGPURenderer;
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
    /** The volume the GI light follows, if any (see the option). */
    volume: DDGIVolume | null;
    readonly timings: ThreeGITimings = { matrixWorldMs: 0, syncMs: 0, encodeMs: 0 };
    /** What the last update sent. */
    get report(): SyncFrameReport {
        return this.sync.report;
    }

    private currentLight: GILight;
    private shadingSeen: { module: ShadingModule; version: number };
    private previousTime = -1;
    private readonly alwaysUpdateMatrixWorld: boolean;
    private renderTimeUpdate: boolean;
    private readonly warnOnce: (message: string) => void;
    private updates = 0;
    private disposed = false;
    /**
     * The GI light is in the scene. False while the platform is still preparing:
     * the light wraps the technique's textures, which exist only after `prepare`.
     */
    private lightInScene = false;
    /** `prepare` failed (the `preparing` option): the next `update` throws this. */
    private prepareFailure: { readonly error: unknown } | null = null;

    /**
     * The platform needs its output technique (`gi.setOutput`). It must be
     * prepared, or still preparing with its promise passed as `preparing`:
     * `update` then copies the scene in, and the GI light joins the scene when
     * `prepare` resolves (file comment).
     */
    constructor(gi: GIPlatform, scene: Scene, options: ThreeGIOptions = {}) {
        if (!gi.isReady && !options.preparing) {
            throw new Error('@pmndrs/gi/three: prepare the platform first (await gi.prepare()), or pass its promise (`preparing: gi.prepare()`) to copy the scene in while it compiles.');
        }
        this.gi = gi;
        this.scene = scene;
        this.budget = options.budget ?? DEFAULT_BUDGET;
        this.volume = options.volume ?? null;
        this.alwaysUpdateMatrixWorld = options.updateMatrixWorld ?? false;
        this.renderTimeUpdate = options.updateOnRender ?? false;
        this.warnOnce = options.warn ?? ((message) => console.warn(`@pmndrs/gi/three: ${message}`));
        this.sync = new SceneSync(gi.scene, { device: gi.device, renderer: options.renderer, warn: options.warn, objectsPerFrame: options.objectsPerFrame, editChecksPerFrame: options.editChecksPerFrame, lightsPerFrame: options.lightsPerFrame });
        this.sync.environment = options.environment ?? null;
        this.sync.sky = options.sky ?? null;
        const shading = gi.shading;
        this.shadingSeen = { module: shading, version: shading.version };
        this.currentLight = new GILight(shading, options.light);
        this.currentLight.name = 'GI';
        this.connectRenderTimeUpdate(this.currentLight);
        // three only lights with lights on the camera's layers; GI is for every camera.
        this.currentLight.layers.enableAll();
        if (gi.isReady) {
            this.addLightToScene();
        } else if (options.preparing) {
            // The light joins as soon as the platform is ready, not at the next `update`: a host
            // that only renders (`updateOnRender`) never calls `update`, and the light's own
            // render-time update can't run while the light is outside the scene.
            options.preparing.then(
                () => {
                    if (this.gi.isReady && !this.disposed) this.addLightToScene();
                },
                (error: unknown) => {
                    this.prepareFailure = { error };
                    this.warnOnce(`the platform's prepare failed, so GI won't start: ${error instanceof Error ? error.message : String(error)}`);
                },
            );
        }
    }

    /** Puts the GI light into the scene, once (the platform is ready, so its textures exist). */
    private addLightToScene(): void {
        if (this.lightInScene) return;
        this.scene.add(this.currentLight);
        this.lightInScene = true;
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

        if (this.prepareFailure) throw this.prepareFailure.error;
        if (!this.lightInScene && !this.gi.isReady) {
            // The platform is still preparing (file comment): copy the scene in while its
            // kernels compile, and nothing else. There is no GI frame to encode yet.
            this.sync.update(this.scene, camera.layers);
            const afterCopy = performance.now();
            this.timings.matrixWorldMs = afterMatrices - start;
            this.timings.syncMs = afterCopy - afterMatrices;
            this.timings.encodeMs = 0;
            return;
        }
        // Usually done when `preparing` resolved; here too in case this update runs first.
        this.addLightToScene();
        this.followShadingModule();
        if (this.volume) {
            this.light.intensity = this.volume.intensity;
            this.light.priority = this.volume.priority;
        }
        this.sync.update(this.scene, camera.layers);
        // Coverage is worth its cost only for the environment-diffuse workaround or to share the probe slot.
        // Without it coverage reads 1 everywhere, and the workaround would remove environment diffuse outside the volume too.
        const environmentLit = this.scene.environment !== null || this.sync.materialsWithEnvMap > 0;
        this.light.computeCoverage = (this.light.replaceEnvironmentDiffuse && environmentLit) || this.sync.otherProbeLights > 0;
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

    /**
     * Says that `object` (and everything under it) was added, moved, edited
     * or removed, so the next `update` picks it up at once. Without it,
     * changes are found by the scene sync's walk within a few frames
     * (scene-sync.ts states how many).
     */
    markChanged(object: Object3D): void {
        this.sync.markChanged(object);
    }

    /**
     * Light groups (T2.5; docs/api/three.md, "Light groups"): dims or
     * recolours every light of group `group` (`userData.giLightGroup`, 0..3)
     * in GI at once, with no update lag and nothing traced again. `color` is
     * any three colour (a `Color`, `'#ffcc88'`, `0xffcc88`), read in three's
     * linear working colour space, and `intensity` scales it, as a three
     * light's colour and intensity do: `setGroupWeight(1, 'white', 0.25)` is
     * group 1 at a quarter. It multiplies the lights' own colour and intensity
     * (1, white, is the lights as they are), so it changes GI only: dim
     * three's own lights the same way for the direct light. The output
     * technique must have light groups on (DDGI: `settings.lightGroups`).
     */
    setGroupWeight(group: number, color: Color | ColorRepresentation, intensity = 1): void {
        const technique = this.gi.outputTechnique;
        if (!technique?.setGroupWeight) throw new Error(`@pmndrs/gi/three: the GI technique '${technique?.id ?? 'none'}' has no light group weights`);
        technique.setGroupWeight(group, linearWeight(color, intensity));
    }

    /** The same for the sky and emissive surfaces: GI's environment layer. */
    setEnvironmentWeight(color: Color | ColorRepresentation, intensity = 1): void {
        const technique = this.gi.outputTechnique;
        if (!technique?.setEnvironmentWeight) throw new Error(`@pmndrs/gi/three: the GI technique '${technique?.id ?? 'none'}' has no light group weights`);
        technique.setEnvironmentWeight(linearWeight(color, intensity));
    }

    /** Removes the light and everything the sync created. The platform itself is the host's to dispose. */
    dispose(): void {
        this.disposed = true;
        this.light.removeFromParent();
        this.light.dispose();
        this.sync.dispose();
    }

    /** Run `update` from the GI light's render-time update (see the file comment). Can be switched at any time. */
    get updateOnRender(): boolean {
        return this.renderTimeUpdate;
    }

    set updateOnRender(enabled: boolean) {
        this.renderTimeUpdate = enabled;
        this.connectRenderTimeUpdate(this.currentLight);
    }

    /** With `updateOnRender`, the light runs `update` from its render-time update (see the file comment). */
    private connectRenderTimeUpdate(light: GILight): void {
        if (!this.renderTimeUpdate) {
            light.onRender = null;
            return;
        }
        light.renderUpdateFailed = false;
        light.onRender = (camera) => this.update(camera);
        light.onRenderError = (error) => this.warnOnce(`the GI frame failed during a render, so GI stopped updating: ${error instanceof Error ? error.message : String(error)}`);
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
        next.layers.mask = old.layers.mask;
        this.connectRenderTimeUpdate(next);
        (old.parent ?? this.scene).add(next);
        old.removeFromParent();
        old.dispose();
        this.currentLight = next;
    }
}

/** One scratch colour for the conversion, so a weight change allocates nothing. */
const weightColor = new Color();

/** A three colour times an intensity, as the platform's linear RGB (three keeps `Color` in its linear working space). */
function linearWeight(color: Color | ColorRepresentation, intensity: number): [number, number, number] {
    weightColor.set(color);
    return [weightColor.r * intensity, weightColor.g * intensity, weightColor.b * intensity];
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
