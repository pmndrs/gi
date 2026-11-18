/**
 * The radiance cache (AMD GI-1.0/1.2, phase 2: single bounce, no history),
 * through the three.js adapter, the way a three user builds a screen-space
 * technique (docs/api/three.md):
 *
 *   pass(scene, camera) with mrt({ output, giAlbedo, normal, ...screenInputs.outputs })
 *   -> the screen stage (`screenInputs(...).encodeBefore`): the platform's
 *      `encodeScreen` on the pass's depth and normals
 *   -> the composite (`ddgi(pass, threeGI)`): the pass's colour plus
 *      `giAlbedo × E / π`, E read by screen position from the technique's
 *      interpolated irradiance.
 *
 * Each frame `ThreeGI.update` runs before the pipeline (the frame's `encode`,
 * then the stage's `encodeScreen` with that camera). GI off draws the same
 * scene pass without the stage and the composite, so no GI work runs; GI only
 * hides the direct lights from the camera, as for DDGI. Its GI light (for
 * surfaces the composite doesn't take, such as transparent ones) lives on its
 * own layer, so the other techniques in a wipe don't see it.
 *
 * The debug views are the technique's own (`debugViews`: probe atlas, probe
 * mask, probe SH), drawn by its shading module's `giDebugColor` by screen
 * position, as the pass's override material. The pass draws before this
 * frame's screen stage, so a debug view shows the previous frame's probes.
 *
 * Not yet: the upscaler (the composite and the screen stage would need the
 * upscaler's jitter, `ThreeGI.jitter`, and its pass; RC2.2 leaves the
 * radiance cache at full resolution), stats in the overlay, the benchmark.
 */

import { mrt, normalView, output, packNormalToRGB, pass } from 'three/tsl';
import type { Node, PassNode } from 'three/webgpu';
import { platformSkyEnvironment } from '../../../../scenes/src/sky/sky-image';
import { DdgiDebugMaterial } from '../ddgi/debug-material';
import {
    createGI,
    ddgi,
    giAlbedo,
    GI_ALBEDO_OUTPUT,
    GI_NORMAL_OUTPUT,
    screenInputs,
    ThreeGI,
    type EnvironmentDescriptor,
    type GIPlatform,
    type ScreenInputsNode,
} from '../ddgi/library';
// Not in the published package (internal until validated, docs/design/radiance-cache.md): by path.
import { RADIANCE_CACHE_DEBUG_VIEW, RadianceCacheTechnique } from '../../../../packages/ddgi-core/src/techniques/radiance-cache/index.ts';
import { applyViewLayers, LAYER_RADIANCE_CACHE, type GiMode } from '../scene/layers';
import type { GiTechnique, TechniqueContext, TechniqueId, UpscaleSettings } from './types';
import { copyViewerCamera, createViewCamera } from './view-camera';

/** The radiance cache's debug views by the name the viewer (and its links) use. */
export const RADIANCE_CACHE_SCREEN_VIEWS = {
    'probe-atlas': RADIANCE_CACHE_DEBUG_VIEW.probeAtlas,
    'probe-mask': RADIANCE_CACHE_DEBUG_VIEW.probeMask,
    'probe-sh': RADIANCE_CACHE_DEBUG_VIEW.sphericalHarmonics,
} as const;

export type RadianceCacheScreenView = keyof typeof RADIANCE_CACHE_SCREEN_VIEWS;

export interface RadianceCacheViewerSettings {
    /** A debug view instead of the normal picture, or null. */
    screenView: RadianceCacheScreenView | null;
}

export const DEFAULT_RADIANCE_CACHE_VIEWER_SETTINGS: RadianceCacheViewerSettings = { screenView: null };

type Phase = 'preparing' | 'ready' | 'failed';

/** The pipeline with GI: the scene pass, its screen stage and the composite. */
interface GiGraph {
    readonly scenePass: PassNode;
    readonly stage: ScreenInputsNode;
    readonly output: Node;
}

export class RadianceCacheViewerTechnique implements GiTechnique {
    readonly id: TechniqueId = 'rc';
    readonly label = 'Radiance cache (AMD GI-1.0 port)';
    readonly camera = createViewCamera();
    readonly supportsUpscale = false;

    private giMode: GiMode = 'on';
    private settings: RadianceCacheViewerSettings;
    private platform: GIPlatform | null = null;
    private technique: RadianceCacheTechnique | null = null;
    private threeGI: ThreeGI | null = null;
    private phase: Phase = 'preparing';
    private failure: string | null = null;
    private disposed = false;
    /** The scene pass alone: GI off, and the picture while the platform prepares. */
    private readonly plainPass: PassNode;
    private giGraph: GiGraph | null = null;
    private debugMaterial: DdgiDebugMaterial | null = null;
    private readonly warnings = new Set<string>();

    constructor(
        private readonly context: TechniqueContext,
        settings: RadianceCacheViewerSettings = DEFAULT_RADIANCE_CACHE_VIEWER_SETTINGS,
    ) {
        this.settings = { ...settings };
        this.applyLayers();
        this.plainPass = pass(context.loaded.scene, this.camera);
        void this.start().catch((error: unknown) => this.fail(error));
    }

    /**
     * The platform with the radiance cache as its output, prepared while the
     * scene is copied in (`ThreeGI`'s `preparing`), then the pipeline with GI.
     */
    private async start(): Promise<void> {
        const { device, loaded, hasTimestamps } = this.context;
        const platform = createGI(device, { gpuTiming: hasTimestamps, label: `viewer radiance cache ${loaded.manifest.id}` });
        this.platform = platform;
        const technique = platform.addTechnique(new RadianceCacheTechnique());
        this.technique = technique;
        platform.setOutput(technique);
        const preparing = platform.prepare();
        // Chrome sends the compile requests when this JavaScript task ends (ThreeGI's file comment).
        await new Promise((resolve) => setTimeout(resolve, 0));
        if (this.disposed) return;
        // The radiance cache has no world-space budget in phase 2: its work is the screen stage.
        const threeGI = new ThreeGI(platform, loaded.scene, { preparing, budget: { units: 0 }, environment: environmentFor(loaded.manifest), warn: (message) => this.warn(message) });
        this.threeGI = threeGI;
        // Only this technique's camera sees its GI light; the adapter keeps the layers when it replaces the light.
        threeGI.light.layers.set(LAYER_RADIANCE_CACHE);
        threeGI.update(this.camera, 0);
        await preparing;
        if (this.disposed) return;
        this.giGraph = this.buildGiGraph(threeGI);
        this.debugMaterial = new DdgiDebugMaterial(platform.shading);
        this.applyDebugView();
        this.phase = 'ready';
    }

    private buildGiGraph(threeGI: ThreeGI): GiGraph {
        const scenePass = pass(this.context.loaded.scene, this.camera);
        const stage = screenInputs(scenePass, threeGI);
        scenePass.setMRT(mrt({ output, [GI_ALBEDO_OUTPUT]: giAlbedo, [GI_NORMAL_OUTPUT]: packNormalToRGB(normalView), ...stage.outputs }));
        // No short-range occlusion: the pass is full resolution (the composite's default there too).
        const composite = ddgi(scenePass, threeGI, { occlusion: false });
        return { scenePass, stage, output: stage.encodeBefore(composite as unknown as Node) };
    }

    private fail(error: unknown): void {
        if (this.disposed) return;
        this.phase = 'failed';
        this.failure = error instanceof Error ? error.message : String(error);
        console.error(`The radiance cache could not start: ${this.failure}`);
    }

    private warn(message: string): void {
        if (this.warnings.has(message) || message.includes('RectAreaLight is not mirrored into GI')) return;
        this.warnings.add(message);
        console.warn(`Radiance cache: ${message}`);
    }

    get outputNode(): Node {
        const graph = this.giGraph;
        if (!graph || this.giMode === 'off') return this.plainPass as unknown as Node;
        return graph.output;
    }

    get ready(): boolean {
        return this.phase !== 'preparing';
    }

    get progress(): number {
        return this.ready ? 1 : 0;
    }

    get prepareGeneration(): number {
        return 0;
    }

    get failed(): string | null {
        return this.failure;
    }

    setGiMode(mode: GiMode): void {
        this.giMode = mode;
        this.applyLayers();
    }

    /** Not upscaled yet (file comment). */
    setUpscale(_settings: UpscaleSettings | null): void {}

    syncCamera(viewerCamera: ReturnType<typeof createViewCamera>): void {
        copyViewerCamera(viewerCamera, this.camera);
    }

    configure(settings: RadianceCacheViewerSettings): void {
        this.settings = { ...settings };
        this.applyDebugView();
    }

    get currentSettings(): Readonly<RadianceCacheViewerSettings> {
        return this.settings;
    }

    /** The frame's `encode` (the camera the screen stage rebuilds positions with), before the pipeline draws. */
    renderBeforePipeline(): void {
        const threeGI = this.threeGI;
        if (!threeGI || this.phase !== 'ready' || this.giMode === 'off') return;
        threeGI.update(this.camera);
        if (this.settings.screenView !== null) this.debugMaterial?.refresh();
    }

    describe(): string {
        switch (this.phase) {
            case 'preparing':
                return 'Starting: compiling the GPU programs and copying the scene';
            case 'failed':
                return `Could not start: ${this.failure}`;
            case 'ready': {
                const tiles = this.technique?.counters['spawn tiles'] ?? 0;
                return `Screen probes: up to ${tiles.toLocaleString('en')} a frame (one per 16 × 16 pixels), 64 rays each; one bounce, no history yet (phase 2)`;
            }
        }
    }

    encodePrepareStep(): boolean {
        return this.ready;
    }

    restartPrepare(): void {}

    notePrepareFinished(): void {}

    describeSettings(): Record<string, number | string> {
        return { probe_tile_px: 16, rays_per_probe: 64, bounces: 1 };
    }

    async warmUp(): Promise<void> {
        await this.context.device.queue.onSubmittedWorkDone();
    }

    dispose(): void {
        this.disposed = true;
        this.debugMaterial?.dispose();
        this.giGraph?.stage.dispose();
        this.giGraph?.scenePass.dispose();
        this.plainPass.dispose();
        this.threeGI?.dispose();
        this.platform?.dispose();
    }

    /** The debug view as the GI pass's override material (file comment), or none. */
    private applyDebugView(): void {
        const graph = this.giGraph;
        const material = this.debugMaterial;
        if (!graph || !material) return;
        const view = this.settings.screenView;
        if (view === null) {
            graph.scenePass.overrideMaterial = null;
            return;
        }
        material.view = RADIANCE_CACHE_SCREEN_VIEWS[view];
        graph.scenePass.overrideMaterial = material.material;
    }

    private applyLayers(): void {
        applyViewLayers(this.camera.layers, this.giMode, [LAYER_RADIANCE_CACHE]);
    }
}

/** The manifest's environment as the platform takes it (the same sky DDGI's rays see; ddgi-technique.ts). */
function environmentFor(manifest: TechniqueContext['loaded']['manifest']): EnvironmentDescriptor {
    const environment = manifest.environment;
    switch (environment.type) {
        case 'none':
            return { type: 'none' };
        case 'color':
            return { type: 'color', color: environment.color, intensity: environment.intensity };
        case 'procedural':
            return platformSkyEnvironment(manifest);
    }
}
