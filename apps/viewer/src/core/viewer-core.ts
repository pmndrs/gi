/**
 * The viewer without any page around it: renderer, current scene, the
 * techniques, and the render pipeline that shows one of them or a
 * split-screen wipe between two. `ViewerApp` puts a UI on it; the baseline
 * benchmark drives it directly.
 */

import {
    ACESFilmicToneMapping,
    AgXToneMapping,
    CineonToneMapping,
    LinearToneMapping,
    NeutralToneMapping,
    NoToneMapping,
    PerspectiveCamera,
    ReinhardToneMapping,
    type ToneMapping,
} from 'three';
import { float, renderOutput, screenUV, select, uniform, vec4 } from 'three/tsl';
import { RenderPipeline, type Node } from 'three/webgpu';
import type { SceneManifest } from '../../../../scenes/src/manifest';
import { createRenderer, type RendererBundle } from '../render/create-renderer';
import { setTimestampTracking } from '../render/gpu-time';
import type { GiMode } from '../scene/layers';
import { disposeScene, loadScene, type LoadedScene, type LoadProgress } from '../scene/load-scene';
import { DdgiTechnique, DEFAULT_DDGI_VIEWER_SETTINGS, type DdgiViewerSettings } from '../techniques/ddgi-technique';
import { DEFAULT_LPG_SETTINGS, LightProbeGridTechnique, type LightProbeGridSettings } from '../techniques/light-probe-grid-technique';
import { NoGiTechnique } from '../techniques/no-gi-technique';
import { ReferenceImageTechnique } from '../techniques/reference-technique';
import { TECHNIQUE_LABELS, type GiTechnique, type TechniqueContext, type TechniqueId, type UpscaleSettings } from '../techniques/types';
import { DEFAULT_VXGI_SETTINGS, VxgiTechnique, type VxgiSettings } from '../techniques/vxgi-technique';

export type ToneMappingName = 'aces' | 'agx' | 'neutral' | 'reinhard' | 'cineon' | 'linear' | 'none';

export const TONE_MAPPERS: Record<ToneMappingName, { label: string; value: ToneMapping }> = {
    aces: { label: 'ACES filmic', value: ACESFilmicToneMapping },
    agx: { label: 'AgX', value: AgXToneMapping },
    neutral: { label: 'Neutral (Khronos)', value: NeutralToneMapping },
    reinhard: { label: 'Reinhard', value: ReinhardToneMapping },
    cineon: { label: 'Cineon', value: CineonToneMapping },
    linear: { label: 'Linear (clips highlights)', value: LinearToneMapping },
    none: { label: 'None (raw)', value: NoToneMapping },
};

/** What is on screen: one technique, or two with a wipe between them. */
export interface ViewLayout {
    /** The technique shown on the left (or everywhere when `right` is null). */
    left: TechniqueId;
    /** When set, the right side of the wipe. */
    right: TechniqueId | null;
    giMode: GiMode;
    /** Where the wipe line sits, 0 (left edge) to 1 (right edge). */
    split: number;
}

export const DEFAULT_LAYOUT: ViewLayout = { left: 'ddgi', right: null, giMode: 'on', split: 0.5 };

/** Width of the wipe line, in screen pixels. */
const WIPE_LINE_PIXELS = 2;

export class ViewerCore {
    readonly renderer: RendererBundle['renderer'];
    readonly device: GPUDevice;
    readonly adapterInfo: GPUAdapterInfo;
    readonly hasTimestamps: boolean;
    readonly gpuErrors: string[];
    /** The camera the user moves. Techniques copy it each frame. */
    readonly camera = new PerspectiveCamera(60, 1, 0.05, 1000);

    /** Called every animation frame, from three's animation loop. The viewer page draws from it. */
    onTick: ((time: number) => void) | null = null;

    loaded: LoadedScene | null = null;
    layout: ViewLayout = { ...DEFAULT_LAYOUT };

    private readonly pipeline: RenderPipeline;
    private readonly techniques = new Map<TechniqueId, GiTechnique>();
    private upscale: UpscaleSettings | null = null;
    private appliedUpscaleKey = 'off';
    private probeSettings: LightProbeGridSettings = DEFAULT_LPG_SETTINGS;
    private vxgiSettings: VxgiSettings = DEFAULT_VXGI_SETTINGS;
    private ddgiSettings: DdgiViewerSettings = DEFAULT_DDGI_VIEWER_SETTINGS;
    /** The camera bookmark the reference image belongs to (the last one chosen). */
    private referenceBookmark = 0;
    private readonly frameWaiters: Array<(startedAt: number) => void> = [];
    /** When the current set-up of each technique began, by technique, with the generation it belongs to. */
    private readonly prepareStarts = new Map<GiTechnique, { generation: number; startedAt: number }>();
    /** What each technique on screen had when the pipeline was built: its picture and its overlay key. */
    private readonly shownNodes = new Map<TechniqueId, { picture: Node; overlayKey: string | undefined }>();
    private readonly splitUniform = uniform(0.5);
    private readonly lineWidthUniform = uniform(0.002);
    private width = 1;
    private height = 1;

    private constructor(bundle: RendererBundle) {
        this.renderer = bundle.renderer;
        this.device = bundle.device;
        this.adapterInfo = bundle.adapterInfo;
        this.hasTimestamps = bundle.hasTimestamps;
        this.gpuErrors = bundle.gpuErrors;
        this.pipeline = new RenderPipeline(this.renderer);
        // The output transform (tone mapping, exposure, output colour) is applied per technique in
        // `rebuildOutput`, so debug overlays can be drawn after it, in display space.
        this.pipeline.outputColorTransform = false;
        this.pipeline.outputNode = vec4(0, 0, 0, 1);
        this.setToneMapping('aces');
        this.setExposureEv(0);
        // three advances its frame counter only inside its own animation loop. Without it frame-type
        // nodes (render passes, VXGI's temporal cycle, TRAA jitter) update once and then stop.
        this.renderer.setAnimationLoop((time: number) => this.tick(time));
    }

    private tick(time: number): void {
        this.onTick?.(time);
        if (this.frameWaiters.length === 0) return;
        const startedAt = performance.now();
        this.renderFrame();
        for (const waiter of this.frameWaiters.splice(0)) waiter(startedAt);
    }

    /**
     * Draws one frame from inside the animation loop and resolves once it has been recorded,
     * with the `performance.now()` at which drawing began. For code with no page loop of its
     * own, such as the benchmark.
     */
    renderThroughLoop(): Promise<number> {
        return new Promise((resolve) => this.frameWaiters.push(resolve));
    }

    static async create(canvas: HTMLCanvasElement): Promise<ViewerCore> {
        return new ViewerCore(await createRenderer(canvas));
    }

    // -- scene -------------------------------------------------------------

    /** Throws away the old scene and its techniques, loads `manifest`, and shows its first technique. */
    async loadScene(manifest: SceneManifest, report: (progress: LoadProgress) => void): Promise<LoadedScene> {
        this.unloadScene();
        const loaded = await loadScene(manifest, report);
        this.loaded = loaded;
        const first = manifest.cameras[0];
        this.camera.fov = first.fovYDeg;
        this.camera.near = Math.max(0.02, loaded.diagonal / 20000);
        this.camera.far = Math.max(200, loaded.diagonal * 4);
        this.camera.updateProjectionMatrix();
        this.rebuildOutput();
        return loaded;
    }

    unloadScene(): void {
        for (const technique of this.techniques.values()) technique.dispose();
        this.techniques.clear();
        this.referenceBookmark = 0;
        this.prepareStarts.clear();
        if (this.loaded) disposeScene(this.loaded);
        this.loaded = null;
        this.pipeline.outputNode = vec4(0, 0, 0, 1);
        this.pipeline.needsUpdate = true;
    }

    // -- techniques --------------------------------------------------------

    /** The technique, created (not yet prepared) the first time it is asked for. */
    getTechnique(id: TechniqueId): GiTechnique {
        const existing = this.techniques.get(id);
        if (existing) return existing;
        if (!this.loaded) throw new Error('No scene is loaded.');
        const context: TechniqueContext = { renderer: this.renderer, device: this.device, loaded: this.loaded, hasTimestamps: this.hasTimestamps };
        const technique = this.createTechnique(id, context);
        if (this.upscaleAvailable && this.upscale) technique.setUpscale(this.upscale);
        this.techniques.set(id, technique);
        return technique;
    }

    private createTechnique(id: TechniqueId, context: TechniqueContext): GiTechnique {
        switch (id) {
            case 'ddgi':
                return new DdgiTechnique(context, this.ddgiSettings);
            case 'lpg':
                return new LightProbeGridTechnique(context, this.probeSettings);
            case 'vxgi':
                return new VxgiTechnique(context, this.vxgiSettings);
            case 'cycles': {
                const reference = new ReferenceImageTechnique(context);
                reference.setBookmark(this.referenceBookmark);
                return reference;
            }
            case 'none':
                return new NoGiTechnique(context);
        }
    }

    /** DDGI, if it has been created for this scene. */
    get ddgi(): DdgiTechnique | null {
        return (this.techniques.get('ddgi') as DdgiTechnique | undefined) ?? null;
    }

    /** The reference image, if it has been created for this scene. */
    get reference(): ReferenceImageTechnique | null {
        return (this.techniques.get('cycles') as ReferenceImageTechnique | undefined) ?? null;
    }

    /** Techniques currently on screen. */
    activeTechniques(): GiTechnique[] {
        if (!this.loaded) return [];
        const ids = [this.layout.left, ...(this.layout.right ? [this.layout.right] : [])];
        return ids.map((id) => this.getTechnique(id));
    }

    /**
     * Applies to the probe grid now if it exists, and to the one created later. The grid bakes
     * again only when a setting that changes the bake changed. Returns whether it did.
     */
    configureProbeGrid(settings: LightProbeGridSettings): boolean {
        this.probeSettings = { ...settings };
        return (this.techniques.get('lpg') as LightProbeGridTechnique | undefined)?.configure(settings) ?? false;
    }

    configureVxgi(settings: VxgiSettings): void {
        this.vxgiSettings = settings;
        (this.techniques.get('vxgi') as VxgiTechnique | undefined)?.configure(settings);
    }

    /** Applies to DDGI now if it exists, and to the one created later. Display settings only: nothing recompiles. */
    configureDdgi(settings: DdgiViewerSettings): void {
        this.ddgiSettings = { ...settings };
        this.ddgi?.configure(settings);
    }

    /** Bakes the probe grid again with its current settings. */
    rebake(): void {
        this.techniques.get('lpg')?.restartPrepare();
    }

    /** Forgets DDGI's field and warms it up again. */
    restartDdgi(): void {
        this.ddgi?.restartPrepare();
    }

    /** Which camera bookmark's reference image to show (0-based). */
    setReferenceBookmark(index: number): void {
        this.referenceBookmark = index;
        this.reference?.setBookmark(index);
    }

    /** Whether a technique on screen still has set-up to do. */
    get preparing(): boolean {
        return this.activeTechniques().some((technique) => !technique.ready);
    }

    /**
     * Records set-up steps of what is on screen for about `budgetMs` of CPU time,
     * then waits once for the GPU to finish them.
     *
     * The steps themselves never wait: one wait per batch keeps the GPU queue
     * short and the page responsive without a read-back after every step. A
     * technique whose set-up finishes in this batch is told how long it took,
     * tagged with the generation it was started under, so a restart during
     * the wait can't be reported as finished.
     */
    async pumpPrepare(budgetMs: number): Promise<void> {
        const startedAt = performance.now();
        const waiting = this.activeTechniques().filter((technique) => !technique.ready);
        for (const technique of waiting) {
            const record = this.prepareStarts.get(technique);
            if (!record || record.generation !== technique.prepareGeneration) {
                this.prepareStarts.set(technique, { generation: technique.prepareGeneration, startedAt });
            }
            const maxSteps = technique.prepareStepsPerBatch ?? Number.POSITIVE_INFINITY;
            for (let step = 0; step < maxSteps && !technique.ready && performance.now() - startedAt < budgetMs; step++) technique.encodePrepareStep();
        }
        if (waiting.length === 0) return;
        await this.device.queue.onSubmittedWorkDone();
        const finishedAt = performance.now();
        for (const technique of waiting) {
            const record = this.prepareStarts.get(technique);
            if (!technique.ready || !record || record.generation !== technique.prepareGeneration) continue;
            technique.notePrepareFinished(record.generation, finishedAt - record.startedAt);
            this.prepareStarts.delete(technique);
        }
    }

    /**
     * Turns three's GPU pass timing on or off (only on where the device supports it).
     * The benchmark turns it off for wall-clock runs that must not read anything back.
     */
    setTimestampTracking(enabled: boolean): void {
        setTimestampTracking(this.renderer, enabled && this.hasTimestamps);
    }

    /** One plain-language line per technique that is on screen. */
    statusLines(): string[] {
        return this.activeTechniques().map((technique) => `${TECHNIQUE_LABELS[technique.id]}: ${technique.describe()}`);
    }

    // -- layout and pipeline -----------------------------------------------

    setLayout(change: Partial<ViewLayout>): void {
        const next = { ...this.layout, ...change };
        if (next.right === next.left) next.right = null;
        this.layout = next;
        this.splitUniform.value = next.split;
        if (this.loaded) this.rebuildOutput();
    }

    /** Moves only the wipe line; no shader work. */
    setSplit(split: number): void {
        this.layout.split = split;
        this.splitUniform.value = split;
    }

    setUpscale(settings: UpscaleSettings | null): void {
        this.upscale = settings;
        if (this.loaded) this.rebuildOutput();
    }

    /**
     * The upscaler is skipped while VXGI is on screen: `upscaleScene` can't wrap VXGI's passes,
     * and its own scene pass would share lights with VXGI's in the same frame.
     */
    get upscaleAvailable(): boolean {
        return this.layout.left !== 'vxgi' && this.layout.right !== 'vxgi';
    }

    private applyUpscale(): void {
        const wanted = this.upscaleAvailable ? this.upscale : null;
        const key = wanted ? wanted.quality : 'off';
        if (key === this.appliedUpscaleKey) return;
        this.appliedUpscaleKey = key;
        for (const technique of this.techniques.values()) technique.setUpscale(wanted);
    }

    private rebuildOutput(): void {
        const { left, right, giMode } = this.layout;
        // Techniques are created first so they exist when the upscaler setting is applied to them.
        this.getTechnique(left);
        if (right) this.getTechnique(right);
        this.applyUpscale();
        const leftTechnique = this.getTechnique(left);
        leftTechnique.setGiMode(giMode);
        let output = this.displayNode(leftTechnique);
        if (right) {
            const rightTechnique = this.getTechnique(right);
            rightTechnique.setGiMode(giMode);
            const distanceFromLine = screenUV.x.sub(this.splitUniform).abs();
            const onLine = distanceFromLine.lessThan(this.lineWidthUniform);
            const side = select(screenUV.x.lessThan(this.splitUniform), output, this.displayNode(rightTechnique));
            output = select(onLine, vec4(1, 1, 1, float(1)), side);
        }
        this.pipeline.outputNode = output;
        this.pipeline.needsUpdate = true;
        this.shownNodes.clear();
        this.shownNodes.set(left, { picture: leftTechnique.outputNode, overlayKey: leftTechnique.overlayKey });
        if (right) {
            const rightTechnique = this.getTechnique(right);
            this.shownNodes.set(right, { picture: rightTechnique.outputNode, overlayKey: rightTechnique.overlayKey });
        }
    }

    /**
     * A technique's picture as the screen shows it: its linear HDR picture through the renderer's
     * exposure, tone mapping and output colour (`renderOutput`, what the pipeline would otherwise
     * apply to the whole frame), then its debug overlay, if any, in display space.
     */
    private displayNode(technique: GiTechnique): Node {
        const display = renderOutput(technique.outputNode) as unknown as Node;
        return technique.overlay ? technique.overlay(display) : display;
    }

    // -- display -----------------------------------------------------------

    setToneMapping(name: ToneMappingName): void {
        this.renderer.toneMapping = TONE_MAPPERS[name].value;
        this.pipeline.needsUpdate = true;
    }

    /** Exposure in photographic stops: 0 EV is 1.0, +1 EV doubles the brightness. */
    setExposureEv(ev: number): void {
        this.renderer.toneMappingExposure = 2 ** ev;
    }

    resize(width: number, height: number): void {
        this.width = Math.max(1, Math.floor(width));
        this.height = Math.max(1, Math.floor(height));
        this.renderer.setSize(this.width, this.height, false);
        this.camera.aspect = this.width / this.height;
        this.camera.updateProjectionMatrix();
        this.lineWidthUniform.value = WIPE_LINE_PIXELS / 2 / this.width;
    }

    // -- frame -------------------------------------------------------------

    /** Copies the camera onto the active techniques and draws one frame. */
    renderFrame(): void {
        if (!this.loaded) return;
        const active = this.activeTechniques();
        // A technique that rebuilt its graph (upscaler toggled, render target recreated) or changed its overlay needs the pipeline rebuilt.
        const changed = (technique: GiTechnique): boolean => {
            const shown = this.shownNodes.get(technique.id);
            return !shown || shown.picture !== technique.outputNode || shown.overlayKey !== technique.overlayKey;
        };
        if (active.some(changed)) this.rebuildOutput();
        for (const technique of active) technique.syncCamera(this.camera);
        for (const technique of active) technique.renderBeforePipeline?.();
        this.pipeline.render();
    }

    /**
     * Frees everything, including the device: the viewer created it, so the viewer
     * destroys it. Waiting for garbage collection instead left a big scene's GPU memory
     * held after the page was left, and the next page (Bistro interior after Bistro
     * exterior) baked several times slower.
     */
    dispose(): void {
        this.renderer.setAnimationLoop(null);
        this.unloadScene();
        this.renderer.dispose();
        this.device.destroy();
    }
}
