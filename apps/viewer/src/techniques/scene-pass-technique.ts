/**
 * Shared base for techniques whose picture is a plain scene render: "no GI"
 * and the `LightProbeGrid`. The GI there is a light inside the scene (the
 * probe grid), so no extra pass is needed.
 *
 * "GI only" and "GI off" are done with camera layers (see `scene/layers.ts`):
 * the camera simply doesn't see the direct lights, or the probe grid.
 */

import { HalfFloatType, Vector2, type PerspectiveCamera } from 'three';
import { float, texture, uniform, vec4 } from 'three/tsl';
import { RenderTarget, type Node } from 'three/webgpu';
import { upscaleScene, QualityMode } from '@pmndrs/upscaler';
import { applyViewLayers, type GiMode } from '../scene/layers';
import type { GiTechnique, TechniqueContext, TechniqueId, UpscaleQuality, UpscaleSettings } from './types';
import { copyViewerCamera, createViewCamera } from './view-camera';

const UPSCALER_QUALITY: Record<UpscaleQuality, QualityMode> = {
    'native-aa': QualityMode.NativeAA,
    quality: QualityMode.Quality,
    balanced: QualityMode.Balanced,
    performance: QualityMode.Performance,
    'ultra-performance': QualityMode.UltraPerformance,
};

/**
 * Hardware multisampling of the scene target. These techniques have no temporal
 * anti-aliasing of their own (VXGI has TRAA), so without MSAA their edges would
 * crawl next to VXGI's in the wipe. The benchmark measures each technique's GI
 * cost against the same pipeline with GI off, so this choice does not leak into
 * the GI numbers.
 */
const SCENE_MSAA_SAMPLES = 4;

const drawingBufferSize = new Vector2();

function createSceneTarget(): RenderTarget {
    return new RenderTarget(1, 1, { type: HalfFloatType, samples: SCENE_MSAA_SAMPLES });
}

export abstract class ScenePassTechnique implements GiTechnique {
    abstract readonly id: TechniqueId;
    abstract readonly label: string;
    readonly camera: PerspectiveCamera = createViewCamera();
    readonly supportsUpscale = true;

    protected giMode: GiMode = 'on';
    private upscale: UpscaleSettings | null = null;
    private node: Node;
    /** 1 shows the picture; 0 blacks it out (used when "GI only" is asked of a technique with no GI). */
    private readonly visibility = uniform(1);
    /**
     * Where the scene is drawn: with a plain `renderer.render()` in `renderBeforePipeline`,
     * not by a `pass()` node inside the render pipeline. When this viewer first drew the
     * scene with a `pass()` node, the probe grid's light came out almost black, while the
     * same scene drawn this way was lit (found while building T0.5, three r187). The cause
     * was not pinned down; it looks like the stale atlas binding that `recreateTarget`
     * works around. `pass()` is still used when the upscaler is on, because
     * `upscaleScene` owns its own pass; it is built after the bake has started.
     */
    private target = createSceneTarget();

    protected constructor(protected readonly context: TechniqueContext) {
        this.node = this.buildNode();
    }

    /** Layers beyond the model that carry this technique's GI (for example the probe grid). */
    protected abstract giLayers(): readonly number[];
    /** True when "GI only" has nothing to show, so the picture goes black. */
    protected abstract hasNoGi(): boolean;

    get outputNode(): Node {
        return this.node;
    }

    abstract readonly ready: boolean;
    abstract readonly progress: number;
    abstract readonly prepareGeneration: number;
    abstract describe(): string;
    abstract encodePrepareStep(): boolean;
    abstract restartPrepare(): void;
    abstract notePrepareFinished(generation: number, wallMs: number): void;
    abstract describeSettings(): Record<string, number | string>;
    abstract warmUp(): Promise<void>;
    abstract dispose(): void;

    protected disposeTarget(): void {
        this.target.dispose();
    }

    /**
     * The scene target `renderBeforePipeline` draws into (resolved, linear HDR), for views that
     * look at the picture itself. Replaced by `recreateTarget`, which also rebuilds `outputNode`.
     */
    protected get sceneTarget(): RenderTarget {
        return this.target;
    }

    /** True while the upscaler draws the picture in its own pass: the scene target is then unused. */
    protected get upscaling(): boolean {
        return this.upscale !== null;
    }

    /**
     * Draws into a fresh render target from now on, which makes three build new pipelines
     * and bind groups for the scene. Needed once the probe grid's atlas has been rendered
     * to for the first time: until then every long-lived pipeline reads it as empty.
     */
    protected recreateTarget(): void {
        this.target.dispose();
        this.target = createSceneTarget();
        this.node = this.buildNode();
    }

    setGiMode(mode: GiMode): void {
        this.giMode = mode;
        this.applyLayers();
        this.visibility.value = mode === 'only' && this.hasNoGi() ? 0 : 1;
    }

    setUpscale(settings: UpscaleSettings | null): void {
        this.upscale = settings;
        this.node = this.buildNode();
    }

    syncCamera(viewerCamera: PerspectiveCamera): void {
        copyViewerCamera(viewerCamera, this.camera);
    }

    renderBeforePipeline(): void {
        if (this.upscale) return;
        const { renderer, loaded } = this.context;
        renderer.getDrawingBufferSize(drawingBufferSize);
        if (this.target.width !== drawingBufferSize.width || this.target.height !== drawingBufferSize.height) {
            this.target.setSize(drawingBufferSize.width, drawingBufferSize.height);
        }
        const previousTarget = renderer.getRenderTarget();
        renderer.setRenderTarget(this.target);
        renderer.render(loaded.scene, this.camera);
        // The light markers and gizmo over the picture, from a scene GI never sees (LoadedScene.helpers).
        if (loaded.helpers.children.some((child) => child.visible)) {
            const autoClear = renderer.autoClear;
            renderer.autoClear = false;
            renderer.render(loaded.helpers, this.camera);
            renderer.autoClear = autoClear;
        }
        renderer.setRenderTarget(previousTarget);
    }

    protected applyLayers(): void {
        applyViewLayers(this.camera.layers, this.giMode, this.giLayers());
    }

    private buildNode(): Node {
        this.applyLayers();
        const scene = this.context.loaded.scene;
        const picture: Node = this.upscale
            ? (upscaleScene(scene, this.camera, { quality: UPSCALER_QUALITY[this.upscale.quality] }) as unknown as Node)
            : (texture(this.target.texture) as unknown as Node);
        // Colour only: alpha stays 1 so the sky shows and the wipe line draws over it.
        const rgb = (picture as unknown as { rgb: { mul(factor: unknown): Node } }).rgb;
        return vec4(rgb.mul(this.visibility) as unknown as Node<'vec3'>, float(1));
    }
}
