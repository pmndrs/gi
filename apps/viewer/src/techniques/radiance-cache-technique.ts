/**
 * The radiance cache (AMD GI-1.0/1.2: screen probes with history and the world cache),
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
 * The compare toggles (RC4.5, `settings.compare`) go into the core
 * technique's settings, live: it reads them every frame and starts its
 * history over where a rule changed. `changeReadout` measures how much the
 * GI changes per frame (rc-change-meter.ts) for the panel's readout.
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
import { applyCompareValues, type CompareValues } from '../../../../packages/ddgi-core/src/techniques/radiance-cache/compare-toggles.ts';
import { RcChangeMeter } from './rc-change-meter';
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
    /** The compare toggles that differ from ours (RC4.5; compare-toggles.ts), applied live. */
    compare: CompareValues;
}

export const DEFAULT_RADIANCE_CACHE_VIEWER_SETTINGS: RadianceCacheViewerSettings = { screenView: null, compare: {} };

/** The compare section's readout: the GI's change per frame and the frames since a toggle changed. */
export interface RadianceCacheChangeReadout {
    /** Mean |ΔE| over mean E from the previous frame, in percent (null until measured). */
    meanChangePercent: number | null;
    framesSinceChange: number;
}

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
    /** The GI's change per frame, for the compare readout (made when the technique is ready). */
    private changeMeter: RcChangeMeter | null = null;
    /** Frames drawn with GI since a compare toggle last changed. */
    private framesSinceCompareChange = 0;

    constructor(
        private readonly context: TechniqueContext,
        settings: RadianceCacheViewerSettings = DEFAULT_RADIANCE_CACHE_VIEWER_SETTINGS,
    ) {
        this.settings = { ...settings, compare: { ...settings.compare } };
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
        const technique = new RadianceCacheTechnique();
        applyCompareValues(technique.settings, this.settings.compare);
        platform.addTechnique(technique);
        this.technique = technique;
        platform.setOutput(technique);
        const preparing = platform.prepare();
        // Chrome sends the compile requests when this JavaScript task ends (ThreeGI's file comment).
        await new Promise((resolve) => setTimeout(resolve, 0));
        if (this.disposed) return;
        // The radiance cache has no world-space budget: all of its work, the world cache's
        // included, is the screen stage.
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
        this.changeMeter = new RcChangeMeter(device);
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

    /** The GI pipeline's scene pass and composite once started (for checks that read them back), or null. */
    get giNodes(): { readonly scenePass: PassNode; readonly output: Node } | null {
        const graph = this.giGraph;
        return graph ? { scenePass: graph.scenePass, output: graph.output } : null;
    }

    /** The core technique once started (its probes and irradiance, for checks and readbacks), or null. */
    get radianceCache(): RadianceCacheTechnique | null {
        return this.technique;
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
        const compareChanged = JSON.stringify(settings.compare) !== JSON.stringify(this.settings.compare);
        this.settings = { ...settings, compare: { ...settings.compare } };
        this.applyDebugView();
        if (compareChanged) {
            // Live: the core technique reads its settings every frame (RC4.5).
            if (this.technique) applyCompareValues(this.technique.settings, this.settings.compare);
            this.framesSinceCompareChange = 0;
        }
    }

    /** The GI's change per frame and the frames since a toggle changed; null before the technique is ready. */
    get changeReadout(): RadianceCacheChangeReadout | null {
        if (this.phase !== 'ready' || !this.changeMeter) return null;
        return { meanChangePercent: this.changeMeter.meanChangePercent, framesSinceChange: this.framesSinceCompareChange };
    }

    get currentSettings(): Readonly<RadianceCacheViewerSettings> {
        return this.settings;
    }

    /** The frame's `encode` (the camera the screen stage rebuilds positions with), before the pipeline draws. */
    renderBeforePipeline(): void {
        const threeGI = this.threeGI;
        if (!threeGI || this.phase !== 'ready' || this.giMode === 'off') {
            this.changeMeter?.restart();
            return;
        }
        // The previous frame's GI against the one before, before this frame's runs (rc-change-meter.ts).
        const irradiance = this.technique?.probes?.irradiance;
        if (irradiance) this.changeMeter?.measure(irradiance);
        this.framesSinceCompareChange++;
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
                return `Screen probes: up to ${tiles.toLocaleString('en')} a frame (one per 16 × 16 pixels), 64 rays each, kept from frame to frame, lit through the world cache (hash grid, with GI-1.2's second bounce)`;
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
        this.changeMeter?.dispose();
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
