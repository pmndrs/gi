/**
 * The viewer page: wires the panel, the camera, the stats overlay and the
 * link in the address bar to a `ViewerCore`.
 *
 * `window.__viewer` is what automated runs read (the contact sheet): `ready`
 * becomes true once the scene is drawn and the techniques on screen have
 * finished their set-up, and `errors` lists every problem seen.
 */

import { BoxGeometry, Color, Mesh, MeshStandardMaterial, Vector2 } from 'three';
import { defaultExposureEv, motionTarget, type CameraBookmark, type SceneManifest } from '../../../scenes/src/manifest';
import { CameraRig } from './controls/camera-rig';
import { TONE_MAPPERS, ViewerCore, type ToneMappingName, type ViewLayout } from './core/viewer-core';
import { LightEditor, type LightSettings } from './lighting/light-editor';
import type { CameraControl } from './lighting/motion-drivers';
import { MotionPlayer } from './lighting/motion-player';
import { ResponseMeter } from './lighting/response-meter';
import { TimeOfDay } from './lighting/time-of-day';
import { shotBookmark } from './scene/reference-files';
import { AddedCabinets } from './scene/added-cabinets';
import { findScenario, sceneAddsCabinets, type Scenario } from './scene/scenarios';
import { newestFrame, resolveGpuWindow } from './render/gpu-time';
import { findBookmark, loadAllManifests } from './scene/manifests';
import { LAYER_VIEW_GEOMETRY } from './scene/layers';
import { nextFrame, type LoadedScene, type LoadProgress } from './scene/load-scene';
import { PROBE_STATE_LABELS } from './ddgi/library';
import { halfwaySamples, historyColor, HISTORY_COLORS } from './ddgi/history-colors';
import { PROBE_STATE_COLORS } from './ddgi/probe-spheres';
import { DdgiStatsText, StatsOverlay } from './stats';
import { WHITE_WEIGHT, type DdgiViewerSettings, type GroupWeight, type LightGroupCount } from './techniques/ddgi-technique';
import { DEFAULT_LPG_SETTINGS } from './techniques/light-probe-grid-technique';
import { DEFAULT_VXGI_SETTINGS } from './techniques/vxgi-technique';
import { TECHNIQUE_LABELS, type TechniqueId } from './techniques/types';
import { HELP } from './ui/help-text';
import { LightingControls, type LightingHandlers } from './ui/lighting-controls';
import { Panel, type DdgiPanelSettings, type PanelState, type ProbeLegendEntry } from './ui/panel';
import { RegionPicker } from './ui/region-picker';
import { ScenarioControls, type ScenarioHandlers } from './ui/scenario-controls';
import { DEFAULT_URL_STATE, parseUrlState, presetBudgetMs, serializeUrlState, type GiChoice, type ViewerUrlState } from './ui/url-state';

export interface ViewerAutomationState {
    ready: boolean;
    errors: string[];
    scene: string | null;
    camera: string | null;
    /** Milliseconds from the start of loading until the scene was ready to draw. */
    loadMs: number;
    /** Milliseconds the techniques on screen took to finish their set-up, or null while it runs. */
    prepareMs: number | null;
    frames: number;
    adapter: string;
    /** DDGI's start-up, milliseconds from when it was created (null until reached, or when DDGI isn't used). */
    ddgiStartup: { preparedMs: number | null; firstGiMs: number | null; warmedMs: number | null } | null;
    /** The moving-geometry scenario chosen in the panel, or null. */
    scenario: string | null;
    /** The motion playing and how far into it, or null. */
    motion: { id: string; seconds: number } | null;
    /** GPU programs the GI library compiled while ready (should stay 0), or null without DDGI. */
    recompiles: number | null;
    /**
     * Saved bakes (the panel's "Save the probes to a file" and "Load probes
     * from a file"): the last file saved (its name and size), the last one
     * loaded (stale or not, and which parts of the scene changed), the bakes
     * the running DDGI has loaded, and the last error. Null fields until then.
     */
    bake: {
        /** `ms`: wall time of the save (the readback and the scene fingerprint) or of the load call. */
        saved: { file: string; bytes: number; ms: number } | null;
        loaded: { file: string; stale: boolean; changedParts: readonly string[]; ms: number } | null;
        bakesLoaded: number;
        error: string | null;
    };
}

declare global {
    interface Window {
        __viewer: ViewerAutomationState;
    }
}

/** How long the camera must rest before the link in the address bar follows it. */
const URL_UPDATE_DELAY_MS = 400;
/**
 * CPU time per animation frame given to recording set-up steps (probe baking). The steps
 * never wait for the GPU; there is one wait per batch (see `ViewerCore.pumpPrepare`).
 */
const PREPARE_SLICE_MS = 40;
/**
 * While a set-up runs, the view is redrawn at most every this many milliseconds, and
 * never more than `VIEW_SHARE_WHILE_PREPARING` of the time. A big scene's view costs
 * about as much CPU as a bake step (Bistro interior: about 180 ms for a view, 340 ms for
 * 16 probes), so redrawing on every animation frame would slow the bake down several
 * times over, for a picture that is still changing anyway.
 */
const VIEW_INTERVAL_WHILE_PREPARING_MS = 250;
const VIEW_SHARE_WHILE_PREPARING = 0.2;
/** The longest the view goes without a redraw during a set-up (a view that compiled shaders looks expensive once). */
const VIEW_INTERVAL_WHILE_PREPARING_MAX_MS = 2000;
/** Frames drawn after set-up finishes before an automated run is told the picture is ready. */
const SETTLED_FRAMES = 4;
/** How often the lighting controls (motion time, meter curve, a played day's sliders) are redrawn. */
const LIGHTING_UI_INTERVAL_MS = 100;
/** How close (in pixels) a press must be to the wipe line to grab it. */
const WIPE_GRAB_PIXELS = 10;

function requireElement<T extends HTMLElement>(id: string): T {
    const element = document.getElementById(id);
    if (!element) throw new Error(`index.html is missing #${id}`);
    return element as T;
}

/** Turns what the panel shows into what the core should draw. */
export function layoutFromState(state: ViewerUrlState): ViewLayout {
    if (state.gi === 'off') return { left: 'none', right: null, giMode: 'on', split: state.split };
    if (state.wipe) return { left: state.wipe[0], right: state.wipe[1], giMode: state.gi, split: state.split };
    return { left: state.technique, right: null, giMode: state.gi, split: state.split };
}

export class ViewerApp {
    private readonly canvas = requireElement<HTMLCanvasElement>('view');
    private readonly loadingBox = requireElement<HTMLElement>('loading');
    private readonly statusPill = requireElement<HTMLElement>('status-pill');
    private readonly showPanelButton = requireElement<HTMLButtonElement>('show-panel');
    private readonly leftLabel = document.createElement('div');
    private readonly rightLabel = document.createElement('div');

    private readonly manifests = loadAllManifests();
    private readonly automation: ViewerAutomationState = {
        ready: false,
        errors: [],
        scene: null,
        camera: null,
        loadMs: 0,
        prepareMs: null,
        frames: 0,
        adapter: '',
        ddgiStartup: null,
        scenario: null,
        motion: null,
        recompiles: null,
        bake: { saved: null, loaded: null, bakesLoaded: 0, error: null },
    };
    /** What the last bake save or load did, for the panel's status lines (plain words); null before any. */
    private bakeMessage: string | null = null;

    private core!: ViewerCore;
    private rig!: CameraRig;
    private panel!: Panel;
    private lightingControls!: LightingControls;
    private scenarioControls!: ScenarioControls;
    /** The cabinet scenario's added cabinets, on the scene that offers them; null elsewhere. */
    private cabinets: AddedCabinets | null = null;
    private regionPicker!: RegionPicker;
    /** Per scene: the light editor and gizmo, the motions, time of day and the response meter (made when a scene opens). */
    private lighting: { editor: LightEditor; player: MotionPlayer; timeOfDay: TimeOfDay; meter: ResponseMeter } | null = null;
    /** DDGI's light group weights (index = group) and the environment's, as the sliders set them. */
    private groupWeights: GroupWeight[] = [];
    private environmentWeight: GroupWeight = WHITE_WEIGHT;
    /** The GI choice to go back to after a response measurement, which runs in GI only; null when none runs. */
    private giBeforeMeter: GiChoice | null = null;
    private lastLightingUiAt = 0;
    private stats!: StatsOverlay;
    private readonly ddgiStats = new DdgiStatsText();

    private url: ViewerUrlState = { ...DEFAULT_URL_STATE };
    private manifest: SceneManifest | null = null;
    private bookmarkIndex: number | null = 0;
    private bookmarkPoseKey = '';
    private referenceCube: Mesh | null = null;
    private referenceCubeVisible = false;
    private probeSettings = { bounces: DEFAULT_LPG_SETTINGS.bounces, maxProbes: DEFAULT_LPG_SETTINGS.maxProbes };
    private vxgiSettings = { bounces: DEFAULT_VXGI_SETTINGS.bounces };
    private statsVisible = true;
    private loadingScene = false;
    private pumping = false;
    private settledFrames = 0;
    private resolvingTimestamps = false;
    /** When the set-up now running began; null when nothing has been set up since the last report. */
    private prepareStartedAt: number | null = null;
    private urlTimer: number | undefined;
    private lastFrameAt = performance.now();
    private lastViewRenderAt = 0;
    /** When the previous drawn frame began; the overlay's frame time is the time between drawn frames. */
    private lastViewStartedAt: number | null = null;
    /** CPU milliseconds the last view took to record, which sets how often it is redrawn during a set-up. */
    private lastViewCostMs = 0;
    private lastStatusAt = 0;
    private draggingWipe = false;
    private wasPreparing = false;
    /**
     * Set when the Cycles reference comes on screen or its camera view changes: once its
     * image has loaded, the camera moves to the shot it was rendered from
     * (`alignWithReference`). `display` also takes the shot's exposure and tone mapping
     * (only when Compare is started, so a later change by the user sticks).
     */
    private pendingShot: { display: boolean } | null = null;
    /** The aspect the picture is letterboxed to (the reference's), or null for the whole canvas area. */
    private letterboxAspect: number | null = null;

    static async start(): Promise<ViewerApp> {
        const app = new ViewerApp();
        await app.init();
        return app;
    }

    private constructor() {
        window.__viewer = this.automation;
        // Collect every problem, wherever it is reported, for the automated checks.
        window.addEventListener('error', (event) => this.automation.errors.push(`page error: ${event.message}`));
        window.addEventListener('unhandledrejection', (event) => this.automation.errors.push(`unhandled rejection: ${String(event.reason)}`));
        const originalConsoleError = console.error.bind(console);
        console.error = (...args: unknown[]) => {
            this.automation.errors.push(`console.error: ${args.map(String).join(' ')}`);
            originalConsoleError(...args);
        };
    }

    private async init(): Promise<void> {
        try {
            this.url = parseUrlState(window.location.search);
            this.core = await ViewerCore.create(this.canvas);
            const info = this.core.adapterInfo;
            this.automation.adapter = `${info.vendor} ${info.architecture} ${info.description}`.trim();
            this.fitCanvas();

            this.rig = new CameraRig(this.core.camera, this.canvas);
            this.rig.setMode(this.url.nav);
            this.rig.onPoseChanged(() => this.onCameraMoved());
            // Each GPU sample is one whole frame (see updateTimestamps); the overlay smooths them.
            this.stats = new StatsOverlay(document.body, 'GPU per frame, averaged over recent frames:');
            this.stats.setExtraLines(() => this.ddgiStats.lines());
            this.lightingControls = new LightingControls(this.lightingHandlers());
            this.scenarioControls = new ScenarioControls(this.scenarioHandlers());
            this.panel = new Panel(document.body, this.manifests, this.panelHandlers(), this.lightingControls.sections, [this.scenarioControls.section]);
            this.regionPicker = new RegionPicker(this.canvas);
            this.buildWipeLabels();
            this.installPageEvents();
        } catch (error) {
            this.fail(error);
            return;
        }

        // A link naming only a scenario opens its scene, camera view and motion.
        const linked = findScenario(this.url.scenario);
        if (!linked) this.url.scenario = null;
        else if (this.url.scene === null) this.url = { ...this.url, ...this.scenarioLink(linked) };
        const manifest = this.manifests.find((candidate) => candidate.id === this.url.scene) ?? this.manifests.find((m) => m.id === 'cornell-box') ?? this.manifests[0];
        await this.openScene(manifest, this.url.camera);
        // three's animation loop, not a bare requestAnimationFrame: it advances the frame counter that
        // frame-type nodes (the render passes, VXGI's temporal cycle, TRAA's jitter) update on.
        this.core.onTick = (now) => this.frame(now);
    }

    // -- scenes ------------------------------------------------------------

    private async openScene(manifest: SceneManifest, cameraRequest: string | null): Promise<void> {
        const startedAt = performance.now();
        this.loadingScene = true;
        this.automation.ready = false;
        this.settledFrames = 0;
        this.automation.scene = manifest.id;
        this.automation.prepareMs = null;
        this.prepareStartedAt = null;
        this.showLoading(`Loading ${manifest.title}`, { stage: 'Starting', fraction: null });
        await nextFrame();
        this.disposeLighting();
        // The new scene's DDGI starts with the link's volumes (the camera-following one), not the last scene's.
        this.core.setDdgiSettingsForNextScene(this.ddgiSettings());
        try {
            const loaded = await this.core.loadScene(manifest, (progress) => this.showLoading(`Loading ${manifest.title}`, progress), {
                settings: this.url.sceneSettings,
                clusteredLightReachMetres: this.url.lightReach ?? undefined,
                textureCompression: this.url.textures,
            });
            const notCompressed = loaded.textureCompression.find((part) => part.unavailable);
            if (notCompressed) console.warn(`Textures were loaded uncompressed: ${notCompressed.unavailable}`);
            this.manifest = manifest;
            this.url.scene = manifest.id;
            this.rig.setSceneScale(loaded.diagonal);
            this.addReferenceCube(manifest);

            const found = findBookmark(manifest, cameraRequest);
            const free = this.url.freeCamera;
            if (free) {
                this.rig.goTo({ name: 'free', position: [free[0], free[1], free[2]], target: [free[3], free[4], free[5]], fovYDeg: free[6] });
                this.bookmarkIndex = null;
            } else {
                const bookmark = (found ?? findBookmark(manifest, null)!).bookmark;
                this.bookmarkIndex = found ? found.index : 0;
                this.goToBookmark(bookmark);
            }
            this.automation.camera = free ? 'free' : manifest.cameras[this.bookmarkIndex ?? 0].name;
            this.core.setReferenceBookmark(this.bookmarkIndex ?? 0);
            // A link that opens on the reference wipe at a camera view lines up with the reference's shot.
            this.pendingShot = free ? null : { display: false };
            this.ddgiStats.reset();
            this.rig.update(0);
            this.bookmarkPoseKey = JSON.stringify(this.rig.pose());

            this.createLighting(loaded);
            this.showLoading(`Loading ${manifest.title}`, { stage: 'Compiling shaders', fraction: 0.9 });
            await nextFrame();
            this.applyState();
            this.startLightingFromLink();
            await this.core.renderer.compileAsync(loaded.scene, this.core.getTechnique(this.core.layout.left).camera);

            this.automation.loadMs = performance.now() - startedAt;
            this.prepareStartedAt = performance.now();
            this.loadingBox.classList.add('done');
        } catch (error) {
            this.fail(error);
        } finally {
            this.loadingScene = false;
        }
    }

    private goToBookmark(bookmark: CameraBookmark): void {
        this.rig.goTo(bookmark);
        this.core.camera.fov = bookmark.fovYDeg;
        this.core.camera.updateProjectionMatrix();
    }

    private addReferenceCube(manifest: SceneManifest): void {
        const loaded = this.core.loaded!;
        const position = manifest.referenceCube?.position ?? manifest.cameras[0].target;
        const cube = new Mesh(
            new BoxGeometry(1, 1, 1),
            new MeshStandardMaterial({ name: 'reference-cube-1m', color: new Color().setRGB(0.35, 0.5, 0.8), roughness: 1, metalness: 0 }),
        );
        cube.name = 'reference-cube-1m';
        cube.position.set(...position);
        cube.castShadow = true;
        cube.receiveShadow = true;
        // Viewing cameras only: the probe bake and the voxelizer (layer 0) should not see it.
        cube.layers.set(LAYER_VIEW_GEOMETRY);
        cube.visible = this.referenceCubeVisible;
        loaded.scene.add(cube);
        this.referenceCube = cube;
    }

    // -- state -> everything -----------------------------------------------

    /**
     * Pushes `this.url` and the settings into the core, the panel and the link. Display
     * settings (exposure, tone mapping, GI mode, wipe, upscaler) never re-bake; the probe
     * grid bakes again only if its own bake settings changed (`configure`).
     */
    private applyState(): void {
        this.core.configureProbeGrid({ ...DEFAULT_LPG_SETTINGS, ...this.probeSettings });
        this.core.configureVxgi({ ...DEFAULT_VXGI_SETTINGS, ...this.vxgiSettings });
        this.core.configureDdgi(this.ddgiSettings());
        this.core.setToneMapping(this.url.tone as ToneMappingName);
        this.core.setExposureEv(this.exposureEv());
        this.core.setUpscale(this.url.upscale ? { quality: this.url.upscale } : null);
        this.core.setLayout(layoutFromState(this.url));
        // The picture changed: an automated run waits a few frames before its screenshot.
        this.automation.ready = false;
        this.settledFrames = 0;
        this.refreshPanel();
        this.updateWipeLabels();
        this.scheduleUrlUpdate();
    }

    /**
     * "Save the probes to a file": DDGI's probes as a `.ddgibake` file, which
     * the browser downloads, named after the scene.
     */
    private async saveBake(): Promise<void> {
        const ddgi = this.core.ddgi;
        if (!ddgi || !this.manifest) {
            this.bakeMessage = 'Saving needs DDGI on screen.';
            return;
        }
        try {
            const startedAt = performance.now();
            const bytes = await ddgi.saveBake();
            const ms = performance.now() - startedAt;
            const file = `${this.manifest.id}.ddgibake`;
            const link = document.createElement('a');
            link.href = URL.createObjectURL(new Blob([bytes], { type: 'application/octet-stream' }));
            link.download = file;
            link.click();
            // The download has its own copy once it starts.
            setTimeout(() => URL.revokeObjectURL(link.href), 60_000);
            this.bakeMessage = `Saved the probes to ${file} (${(bytes.byteLength / 1e6).toFixed(1)} MB).`;
            this.automation.bake.saved = { file, bytes: bytes.byteLength, ms };
            this.automation.bake.error = null;
        } catch (error) {
            this.bakeFailed(`Couldn't save the probes: ${error instanceof Error ? error.message : String(error)}`);
        }
    }

    /**
     * "Load probes from a file" (or a dropped file): the probes from a
     * `.ddgibake` file in place of DDGI's own, from the next frame. A file of
     * a changed scene loads as a starting point; one that doesn't fit is
     * refused, and the status says why.
     */
    private async loadBake(file: File): Promise<void> {
        const ddgi = this.core.ddgi;
        if (!ddgi) {
            this.bakeFailed('Loading probes needs DDGI on screen.');
            return;
        }
        try {
            const contents = await file.arrayBuffer();
            const startedAt = performance.now();
            const loaded = ddgi.loadBake(contents);
            const ms = performance.now() - startedAt;
            const saved = new Date(loaded.savedAt).toLocaleString('en-GB');
            this.bakeMessage = loaded.stale
                ? `Loaded probes from ${file.name} (saved ${saved}). The scene changed since (${loaded.changedParts.join(', ')}), so they are updating from there.`
                : `Loaded probes from ${file.name} (saved ${saved}).`;
            this.automation.bake.loaded = { file: file.name, stale: loaded.stale, changedParts: loaded.changedParts, ms };
            this.automation.bake.error = null;
        } catch (error) {
            this.bakeFailed(`Couldn't load ${file.name}: ${error instanceof Error ? error.message.replace(/^@pmndrs\/gi: /, '') : String(error)}`);
        }
    }

    /** A save or load that failed: said in the status, and kept for automated runs (not a console error: it is the user's file). */
    private bakeFailed(message: string): void {
        this.bakeMessage = message;
        this.automation.bake.error = message;
        console.warn(message);
    }

    /** DDGI's settings from the link and the lighting controls. */
    private ddgiSettings(): DdgiViewerSettings {
        return {
            preset: this.url.preset,
            budgetMs: this.url.budgetMs,
            warmUp: this.url.warmUp,
            showProbes: this.url.probes,
            probeColor: this.url.probeColor,
            screenView: this.url.debug,
            atlas: this.url.atlas,
            atlasFill: this.url.atlasFill,
            atlasExposureEv: this.url.atlasEv,
            changeFullScalePercent: this.url.change,
            giTextures: this.url.giTextures,
            lightGroups: this.url.lightGroups,
            // Time of day turns DDGI's sun layer on (DECISIONS 2026-11-08).
            sunLayer: this.url.timeOfDay !== null && (this.lighting?.timeOfDay.available ?? false),
            cameraVolume: this.url.followCamera,
            groupWeights: this.groupWeights,
            environmentWeight: this.environmentWeight,
        };
    }

    /** The scene's own exposure (manifest `exposureEv`), unless the user or the link chose one. */
    private sceneExposureEv(): number {
        return this.manifest ? defaultExposureEv(this.manifest) : 0;
    }

    private exposureEv(): number {
        return this.url.ev ?? this.sceneExposureEv();
    }

    private refreshPanel(): void {
        const state: PanelState = {
            sceneId: this.manifest?.id ?? null,
            bookmarkIndex: this.bookmarkIndex,
            navigation: this.rig.mode,
            flySpeed: this.rig.flySpeed,
            fieldOfView: this.core.camera.fov,
            exposureEv: this.exposureEv(),
            sceneExposureEv: this.sceneExposureEv(),
            toneMapping: this.url.tone as ToneMappingName,
            referenceCube: this.referenceCubeVisible,
            gi: this.url.gi,
            technique: this.url.technique,
            ddgi: {
                preset: this.url.preset,
                budgetMs: this.url.budgetMs,
                followCamera: this.url.followCamera,
                probes: this.url.probes,
                probeColor: this.url.probeColor,
                debug: this.url.debug,
                atlas: this.url.atlas,
                atlasFill: this.url.atlasFill,
                atlasEv: this.url.atlasEv,
                change: this.url.change,
                giTextures: this.url.giTextures,
            },
            probeBounces: this.probeSettings.bounces,
            maxProbes: this.probeSettings.maxProbes,
            vxgiBounces: this.vxgiSettings.bounces,
            wipe: this.url.wipe ? { left: this.url.wipe[0], right: this.url.wipe[1] } : null,
            upscale: this.url.upscale,
            upscaleAvailable: this.core.upscaleAvailable,
            stats: this.statsVisible,
        };
        if (this.manifest) this.panel.setScene(this.manifest);
        this.panel.setState(state);
        this.refreshLightingControls();
    }

    private panelHandlers() {
        const change = (update: Partial<ViewerUrlState>) => {
            this.url = { ...this.url, ...update };
            this.applyState();
        };
        return {
            onScene: (id: string) => {
                const manifest = this.manifests.find((candidate) => candidate.id === id);
                if (!manifest) return;
                // Each scene opens at its own exposure, with its own sun, no motion playing and no scenario.
                // The camera-following volume goes too: it was the scenario's, or chosen for the last scene.
                this.url = { ...this.url, camera: null, freeCamera: null, ev: null, timeOfDay: null, motion: null, scenario: null, followCamera: false };
                void this.openScene(manifest, null);
            },
            onBookmark: (index: number) => {
                if (!this.manifest) return;
                this.bookmarkIndex = index;
                this.goToBookmark(this.manifest.cameras[index]);
                this.rig.update(0);
                this.bookmarkPoseKey = JSON.stringify(this.rig.pose());
                this.url = { ...this.url, camera: String(index + 1), freeCamera: null };
                this.automation.camera = this.manifest.cameras[index].name;
                this.core.setReferenceBookmark(index);
                this.pendingShot = { display: false };
                this.refreshPanel();
                this.scheduleUrlUpdate();
            },
            onCopyLink: () => {
                this.writeUrl();
                void navigator.clipboard?.writeText(window.location.href);
            },
            onNavigation: (mode: 'orbit' | 'fly') => {
                this.rig.setMode(mode);
                change({ nav: mode });
            },
            onFlySpeed: (metresPerSecond: number) => {
                this.rig.flySpeed = metresPerSecond;
            },
            onFieldOfView: (degrees: number) => {
                this.core.camera.fov = degrees;
                this.core.camera.updateProjectionMatrix();
            },
            onExposure: (ev: number | null) => change({ ev }),
            onToneMapping: (name: ToneMappingName) => change({ tone: name }),
            onReferenceCube: (visible: boolean) => {
                this.referenceCubeVisible = visible;
                if (this.referenceCube) this.referenceCube.visible = visible;
            },
            onGi: (choice: ViewerUrlState['gi']) => change({ gi: choice }),
            onTechnique: (id: ViewerUrlState['technique']) => change({ technique: id }),
            onDdgiSettings: (settings: DdgiPanelSettings) =>
                change({
                    preset: settings.preset,
                    // A newly picked preset brings its own budget; otherwise the slider's.
                    budgetMs: settings.preset !== this.url.preset ? presetBudgetMs(settings.preset) : settings.budgetMs,
                    followCamera: settings.followCamera,
                    probes: settings.probes,
                    probeColor: settings.probeColor,
                    debug: settings.debug,
                    atlas: settings.atlas,
                    atlasFill: settings.atlasFill,
                    atlasEv: settings.atlasEv,
                    change: settings.change,
                    giTextures: settings.giTextures,
                }),
            onDdgiRestart: () => this.core.restartDdgi(),
            onDdgiSaveBake: () => void this.saveBake(),
            onDdgiLoadBake: (file: File) => void this.loadBake(file),
            onCompareWithReference: () => {
                // Go to the reference's own shot: camera, aspect, exposure and tone mapping.
                this.pendingShot = { display: true };
                change({ wipe: ['ddgi', 'cycles'], gi: this.url.gi === 'off' ? 'on' : this.url.gi });
            },
            onProbeSettings: (settings: { bounces: number; maxProbes: number }) => {
                this.probeSettings = settings;
                change({});
            },
            onRebake: () => this.core.rebake(),
            onVxgiSettings: (settings: { bounces: number }) => {
                this.vxgiSettings = settings;
                change({});
            },
            onWipe: (enabled: boolean, left: TechniqueId, right: TechniqueId) => {
                if (!enabled) return change({ wipe: null });
                // Two different techniques are needed for a wipe.
                const wipe: [TechniqueId, TechniqueId] = left === right ? [left, left === 'cycles' ? 'ddgi' : 'cycles'] : [left, right];
                const referenceWasShown = this.url.wipe?.includes('cycles') ?? false;
                if (wipe.includes('cycles') && !referenceWasShown) this.pendingShot = { display: true };
                change({ wipe, gi: this.url.gi === 'off' ? 'on' : this.url.gi });
            },
            onUpscale: (quality: ViewerUrlState['upscale']) => change({ upscale: quality }),
            onStats: (visible: boolean) => {
                this.statsVisible = visible;
                this.stats.element.style.display = visible ? '' : 'none';
            },
            onPanelHidden: (hidden: boolean) => this.setPanelHidden(hidden),
        };
    }

    // -- the link ----------------------------------------------------------

    private onCameraMoved(): void {
        if (!this.manifest || this.loadingScene) return;
        // A camera motion (the fly-through) moves it every frame; the link keeps the motion instead.
        if (this.cameraMotionPlaying()) return;
        // The meter reads a still picture; a camera move changes it for another reason.
        if (this.lighting?.meter.running) this.lighting.meter.cancel('the camera moved. Keep it still while measuring.');
        const pose = this.rig.pose();
        if (JSON.stringify(pose) !== this.bookmarkPoseKey) {
            this.bookmarkIndex = null;
            this.url.freeCamera = [...pose.position, ...pose.target, pose.fovYDeg];
        }
        this.scheduleUrlUpdate();
    }

    private scheduleUrlUpdate(): void {
        window.clearTimeout(this.urlTimer);
        this.urlTimer = window.setTimeout(() => this.writeUrl(), URL_UPDATE_DELAY_MS);
    }

    private writeUrl(): void {
        const query = serializeUrlState(this.url);
        window.history.replaceState(null, '', query ? `?${query}` : window.location.pathname);
    }

    // -- split-screen wipe -------------------------------------------------

    private buildWipeLabels(): void {
        for (const [label, side] of [[this.leftLabel, 'left'], [this.rightLabel, 'right']] as const) {
            label.className = `wipe-label ${side}`;
            document.body.append(label);
        }
        this.showPanelButton.addEventListener('click', () => this.setPanelHidden(false));
    }

    /** The canvas's left edge and width in page pixels: the picture sits left of the panel. */
    private canvasBox(): { left: number; width: number } {
        const rect = this.canvas.getBoundingClientRect();
        return { left: rect.left, width: Math.max(1, rect.width) };
    }

    private updateWipeLabels(): void {
        const wipe = this.url.gi !== 'off' ? this.url.wipe : null;
        this.leftLabel.style.display = wipe ? '' : 'none';
        this.rightLabel.style.display = wipe ? '' : 'none';
        if (!wipe) return;
        this.leftLabel.textContent = this.wipeLabel(wipe[0]);
        this.rightLabel.textContent = this.wipeLabel(wipe[1]);
        const box = this.canvasBox();
        const x = `${box.left + this.url.split * box.width}px`;
        this.leftLabel.style.left = x;
        this.rightLabel.style.left = x;
        this.canvas.title = HELP.wipeLine;
    }

    /** A side's name over the picture; the reference also says when this view has none, so the grey side explains itself. */
    private wipeLabel(id: TechniqueId): string {
        const reference = this.core.reference;
        if (id === 'cycles' && reference?.ready && !reference.available) return `${TECHNIQUE_LABELS.cycles}: none for this camera view (see the panel)`;
        return TECHNIQUE_LABELS[id];
    }

    private wipeActive(): boolean {
        return this.url.gi !== 'off' && this.url.wipe !== null;
    }

    private nearWipeLine(clientX: number): boolean {
        const box = this.canvasBox();
        return this.wipeActive() && Math.abs(clientX - (box.left + this.url.split * box.width)) <= WIPE_GRAB_PIXELS;
    }

    // -- page events -------------------------------------------------------

    private setPanelHidden(hidden: boolean): void {
        this.panel.setHidden(hidden);
        // The canvas takes the panel's room (see index.html); the observer below resizes the renderer.
        document.body.classList.toggle('panel-hidden', hidden);
        this.showPanelButton.classList.toggle('visible', hidden);
        this.applyLetterbox();
        this.updateWipeLabels();
    }

    // -- lining up with the Cycles reference -------------------------------

    /**
     * While the Cycles reference is on screen: letterboxes the picture to the
     * reference's stored aspect, and, when asked (`pendingShot`), moves the
     * camera to the shot it was rendered from once its image has loaded, so the
     * wipe lines up pixel for pixel. The camera's near and far planes stay the
     * viewer's: they only clip, and the reference's 0.01 to 2000 m would cost
     * depth precision in the raster passes.
     */
    private alignWithReference(): void {
        const reference = this.core.reference;
        const onScreen = reference !== null && this.url.gi !== 'off' && this.core.activeTechniques().includes(reference);
        const shot = onScreen ? reference.shot : null;
        this.setLetterbox(shot ? shot.camera.width / shot.camera.height : null);
        // A request made while the reference isn't shown is dropped, so it can't move the camera later.
        if (!onScreen) this.pendingShot = null;
        if (!this.pendingShot || !onScreen || !reference.ready) return;
        const pending = this.pendingShot;
        this.pendingShot = null;
        // No image, or an old one without a stored shot: nothing to line up with.
        if (!shot || !this.manifest) return;
        const bookmark = shotBookmark(shot);
        this.goToBookmark(bookmark);
        this.rig.update(0);
        this.bookmarkPoseKey = JSON.stringify(this.rig.pose());
        const index = this.manifest.cameras.findIndex((camera) => camera.name === shot.camera.bookmark);
        if (index >= 0) {
            this.bookmarkIndex = index;
            this.url = { ...this.url, camera: String(index + 1), freeCamera: null };
            this.automation.camera = shot.camera.bookmark;
        }
        if (pending.display) {
            const tone = shot.display.toneMapping in TONE_MAPPERS ? (shot.display.toneMapping as ToneMappingName) : this.url.tone;
            // The scene's own exposure is the default; only a different one goes into the link.
            const ev = shot.display.exposureEv === this.sceneExposureEv() ? null : shot.display.exposureEv;
            this.url = { ...this.url, tone, ev };
            this.applyState();
        } else {
            this.refreshPanel();
            this.scheduleUrlUpdate();
        }
    }

    private setLetterbox(aspect: number | null): void {
        if (aspect === this.letterboxAspect) return;
        this.letterboxAspect = aspect;
        this.applyLetterbox();
    }

    /**
     * Fits the canvas to `letterboxAspect` inside the area index.html gives it,
     * centred, with the page background as the bars. The ResizeObserver then
     * resizes the renderer, so the camera's aspect is the reference's.
     */
    private applyLetterbox(): void {
        const style = this.canvas.style;
        style.width = '';
        style.height = '';
        style.left = '';
        style.top = '';
        const aspect = this.letterboxAspect;
        if (aspect === null) return;
        const area = this.canvas.getBoundingClientRect();
        let width = area.width;
        let height = area.height;
        if (width / height > aspect) width = height * aspect;
        else height = width / aspect;
        style.width = `${width}px`;
        style.height = `${height}px`;
        style.left = `${area.left + (area.width - width) / 2}px`;
        style.top = `${area.top + (area.height - height) / 2}px`;
    }

    /** Sizes the renderer to the canvas's laid-out size. */
    private fitCanvas(): void {
        const box = this.canvas.getBoundingClientRect();
        this.core.resize(box.width, box.height);
    }

    private installPageEvents(): void {
        // Give the GPU memory back as soon as the page is left (see ViewerCore.dispose).
        window.addEventListener('pagehide', () => this.core.dispose());
        new ResizeObserver(() => {
            this.fitCanvas();
            this.updateWipeLabels();
        }).observe(this.canvas);
        // A letterboxed canvas has a fixed size, so the observer above wouldn't see the window change.
        window.addEventListener('resize', () => this.applyLetterbox());
        // A .ddgibake file dropped onto the page loads as the panel's "Load probes from a file" does.
        window.addEventListener('dragover', (event) => {
            if (event.dataTransfer?.types.includes('Files')) event.preventDefault();
        });
        window.addEventListener('drop', (event) => {
            const file = event.dataTransfer?.files[0];
            if (!file || !file.name.endsWith('.ddgibake')) return;
            event.preventDefault();
            void this.loadBake(file);
        });
        window.addEventListener('keydown', (event) => {
            const typing = event.target instanceof HTMLElement && ['INPUT', 'SELECT', 'TEXTAREA'].includes(event.target.tagName);
            if (event.code === 'KeyH' && !typing && !event.ctrlKey && !event.metaKey) this.setPanelHidden(!this.panel.hidden);
        });

        // The wipe line: grab it before the camera controls see the press.
        window.addEventListener(
            'pointerdown',
            (event) => {
                if (event.target !== this.canvas || !this.nearWipeLine(event.clientX)) return;
                this.draggingWipe = true;
                this.rig.enabled = false;
                this.canvas.setPointerCapture(event.pointerId);
                event.stopImmediatePropagation();
            },
            { capture: true },
        );
        window.addEventListener('pointermove', (event) => {
            if (this.draggingWipe) {
                const box = this.canvasBox();
                this.url.split = Math.min(0.98, Math.max(0.02, (event.clientX - box.left) / box.width));
                this.core.setSplit(this.url.split);
                this.updateWipeLabels();
            } else if (event.target === this.canvas) {
                this.canvas.style.cursor = this.nearWipeLine(event.clientX) ? 'col-resize' : '';
            }
        });
        window.addEventListener('pointerup', () => {
            if (!this.draggingWipe) return;
            this.draggingWipe = false;
            this.rig.enabled = true;
            this.scheduleUrlUpdate();
        });
    }

    // -- the frame loop ----------------------------------------------------

    private frame(now: number): void {
        const deltaMs = now - this.lastFrameAt;
        this.lastFrameAt = now;
        try {
            this.rig.update(deltaMs / 1000);
            this.beforeLightingFrame(now);
            const preparing = this.core.preparing;
            // Set-up frames are drawn a few times a second on purpose; don't let them into the frame-time average.
            if (preparing !== this.wasPreparing) {
                this.wasPreparing = preparing;
                this.stats.resetFrameTime();
                this.lastViewStartedAt = null;
            }
            // While a set-up runs, most of the time goes to it (see VIEW_INTERVAL_WHILE_PREPARING_MS).
            const viewInterval = Math.min(
                VIEW_INTERVAL_WHILE_PREPARING_MAX_MS,
                Math.max(VIEW_INTERVAL_WHILE_PREPARING_MS, this.lastViewCostMs / VIEW_SHARE_WHILE_PREPARING),
            );
            if (!preparing || performance.now() - this.lastViewRenderAt >= viewInterval) {
                const viewStartedAt = performance.now();
                this.core.renderFrame();
                this.lighting?.meter.afterFrame(now);
                this.lastViewRenderAt = performance.now();
                this.lastViewCostMs = this.lastViewRenderAt - viewStartedAt;
                this.automation.frames++;
                if (this.lastViewStartedAt !== null) this.stats.recordFrame(viewStartedAt - this.lastViewStartedAt);
                this.lastViewStartedAt = viewStartedAt;
                this.updateTimestamps(preparing);
                this.updateDdgiStats();
            }
            this.pumpPreparation(preparing, now);
            this.updateStatus(now);
            this.updateLightingUi(now);
        } catch (error) {
            console.error(error);
            this.core.onTick = null;
            this.fail(error);
        }
    }

    /** Set-up (probe baking) runs between frames, so the picture stays live while it works. */
    private pumpPreparation(preparing: boolean, now: number): void {
        if (this.pumping || this.loadingScene || !this.core.loaded) return;
        if (!preparing) {
            if (this.automation.prepareMs === null) this.automation.prepareMs = now - (this.prepareStartedAt ?? now);
            this.prepareStartedAt = null;
            // A few clean frames after the set-up, then an automated run may take its picture.
            if (!this.automation.ready && ++this.settledFrames >= SETTLED_FRAMES) this.automation.ready = true;
            return;
        }
        // A set-up began (a new technique on screen, a re-bake): time it from here.
        if (this.prepareStartedAt === null) this.prepareStartedAt = now;
        this.automation.prepareMs = null;
        this.automation.ready = false;
        this.settledFrames = 0;
        this.pumping = true;
        void this.core
            .pumpPrepare(PREPARE_SLICE_MS)
            .catch((error) => {
                console.error(error);
                this.fail(error);
            })
            .finally(() => {
                this.pumping = false;
            });
    }

    /**
     * GPU time of the newest frame drawn, from three's timestamp queries. Several frames
     * may have been drawn since the last read; only the newest is shown, and the overlay
     * averages those per-frame values. Skipped during a set-up: its passes carry the
     * frame id of the frame before them and would be counted as drawing.
     */
    private updateTimestamps(preparing: boolean): void {
        if (!this.core.hasTimestamps || this.resolvingTimestamps || !this.statsVisible) return;
        this.resolvingTimestamps = true;
        void resolveGpuWindow(this.core.renderer)
            .then((window) => {
                const frame = preparing ? null : newestFrame(window);
                const timings = new Map<string, number>();
                if (frame && frame.invalidPasses === 0) {
                    if (frame.renderMs > 0) timings.set('render passes', frame.renderMs);
                    if (frame.computeMs > 0) timings.set('compute passes', frame.computeMs);
                }
                this.stats.setPassTimings(timings);
            })
            .catch(() => this.stats.setPassTimings(new Map()))
            .finally(() => {
                this.resolvingTimestamps = false;
            });
    }

    /** DDGI's lines in the stats box, while it is on screen and running. */
    private updateDdgiStats(): void {
        const ddgi = this.core.ddgi;
        const shown = ddgi !== null && this.core.activeTechniques().includes(ddgi) && this.url.gi !== 'off';
        // Warm-up frames update thousands of probes each; they would swamp the averages of normal frames.
        const snapshot = shown && this.statsVisible && ddgi.running ? ddgi.statsSnapshot() : null;
        // The overlay reads the lines itself when it redraws (a few times a second), not every frame.
        if (snapshot) this.ddgiStats.add(snapshot);
        else this.ddgiStats.reset();
    }

    /**
     * How far down the picture the stats box reaches, as a share of the picture's height (0 while it
     * is hidden). DDGI's probe atlas stays below it when it fills the picture.
     */
    private statsBoxCoverShare(): number {
        if (!this.statsVisible) return 0;
        const box = this.stats.element.getBoundingClientRect();
        const picture = this.canvas.getBoundingClientRect();
        if (box.height === 0 || picture.height === 0) return 0;
        return (box.bottom - picture.top) / picture.height;
    }

    /**
     * The probe legend. By state: each state's colour and, once the GPU has counted them, how many
     * probes are in it. By history: the colour scale from a few updates to the full count. By its own
     * light: the state colours again, which only the rims show.
     */
    private probeLegend(): ProbeLegendEntry[] {
        const ddgi = this.core.ddgi;
        if (this.url.probeColor === 'history') {
            const maxSamples = ddgi?.historyMaxSamples ?? 0;
            const halfway = halfwaySamples(maxSamples);
            return [
                { color: historyColor(1, maxSamples), label: '1 update: new, or rarely updated (noisy)', count: null },
                { color: historyColor(halfway, maxSamples), label: `about ${Math.round(halfway)} updates`, count: null },
                { color: historyColor(maxSamples, maxSamples), label: `${maxSamples} updates: settled, the most kept`, count: null },
                { color: HISTORY_COLORS.notActive, label: 'not active (off, or not updated yet)', count: null },
            ];
        }
        // By its own light, the ball shows the probe's light and only the rim its state (RTXGI's probe view).
        const rim = this.url.probeColor === 'irradiance' ? 'rim: ' : '';
        return Object.entries(PROBE_STATE_LABELS).map(([state, label]) => ({
            color: PROBE_STATE_COLORS[Number(state)],
            label: `${rim}${label}`,
            count: ddgi?.probeStateCount(Number(state)) ?? null,
        }));
    }

    /** What the panel says about the reference image while it is on screen: what it is, how to make it, or how to line it up. */
    private referenceNote(): string | null {
        const reference = this.core.reference;
        if (!reference || !this.core.activeTechniques().includes(reference)) return null;
        const lines = [reference.describe()];
        if (reference.available && !reference.isAligned(this.core.camera)) {
            lines.push(`It lines up only from the camera it was rendered from: press Compare again, or camera view ${reference.bookmarkNumber} under Scene, to go back.`);
        }
        return lines.join(' ');
    }

    private updateStatus(now: number): void {
        if (now - this.lastStatusAt < 200 || !this.core.loaded) return;
        this.lastStatusAt = now;
        const lines = this.core.statusLines();
        if (this.bakeMessage) lines.push(this.bakeMessage);
        this.automation.bake.bakesLoaded = this.core.ddgi?.bakesLoaded ?? 0;
        const { stats } = this.core.loaded;
        lines.push(`${stats.triangles.toLocaleString('en')} triangles, ${stats.meshes} meshes`);
        this.panel.setStatus(lines);
        this.automation.ddgiStartup = this.core.ddgi ? { ...this.core.ddgi.startup } : null;
        this.automation.recompiles = this.core.ddgi ? this.core.ddgi.recompiles : null;
        const player = this.lighting?.player;
        this.automation.motion = player?.playing ? { id: player.playing.id, seconds: player.seconds(now) } : null;
        this.automation.scenario = this.url.scenario;
        this.core.ddgi?.setOverlayTopInset(this.statsBoxCoverShare());
        if (this.url.probes) this.panel.setProbeLegend(this.probeLegend());
        this.alignWithReference();
        this.panel.setReferenceNote(this.referenceNote());
        // The reference loads in the background; its label follows.
        this.updateWipeLabels();
        const baking = this.core.activeTechniques().find((technique) => !technique.ready);
        const meter = this.lighting?.meter;
        // The meter's test mode says so on the picture too, so it is never mistaken for the normal picture.
        if (meter?.running) this.statusPill.textContent = `Response meter: ${meter.describe(now)}`;
        else this.statusPill.textContent = baking ? `${TECHNIQUE_LABELS[baking.id]}: ${baking.describe()}` : '';
    }

    // -- lights, motions, time of day, light groups, the response meter -----

    /** The scene's light editor, motion player, time of day and response meter. */
    private createLighting(loaded: LoadedScene): void {
        const editor = new LightEditor(loaded, this.core.camera, this.canvas, {
            onEdited: () => {
                const selected = editor.selected;
                if (selected) this.core.ddgi?.markChanged(selected.light);
                this.refreshLightingControls();
            },
            onDragging: (dragging) => {
                this.rig.enabled = !dragging;
            },
            onSelected: () => this.refreshLightingControls(),
        });
        const player = new MotionPlayer(loaded, { camera: this.cameraControl() });
        player.windStrength = this.url.windStrength;
        const timeOfDay = new TimeOfDay(loaded);
        const meter = new ResponseMeter({
            ddgi: () => this.core.ddgi,
            player,
            drawingBufferSize: () => {
                const size = this.core.renderer.getDrawingBufferSize(new Vector2());
                return { width: size.x, height: size.y };
            },
            markChanged: (light) => this.core.ddgi?.markChanged(light),
            describe: () => ({ scene: loaded.manifest.id, camera: this.automation.camera ?? 'free', adapter: this.automation.adapter }),
        });
        this.lighting = { editor, player, timeOfDay, meter };
        this.groupWeights = [];
        this.environmentWeight = WHITE_WEIGHT;
        if (!timeOfDay.available) this.url.timeOfDay = null;
        const fit = timeOfDay.fit;
        this.lightingControls.setScene(
            editor.lights.map((entry) => ({ label: entry.label, kind: entry.kind })),
            player.motions,
            { available: timeOfDay.available, hour: this.url.timeOfDay?.hour ?? fit?.hour ?? 12, dayOfYear: this.url.timeOfDay?.dayOfYear ?? fit?.dayOfYear ?? 172 },
            loaded.diagonal,
        );
        this.regionPicker.clear();
        this.lightingControls.setMeterRegion('Watching the whole picture.');
        this.cabinets = sceneAddsCabinets(loaded.manifest.id) ? new AddedCabinets(loaded.content) : null;
        this.scenarioControls.setSceneControls({ wind: player.hasWind, cabinets: this.cabinets !== null });
        this.scenarioControls.setWindStrength(player.windStrength);
    }

    /** What the link asks for once DDGI exists: time of day, and a motion playing. */
    private startLightingFromLink(): void {
        const lighting = this.lighting;
        if (!lighting) return;
        const now = performance.now();
        const link = this.url.timeOfDay;
        if (link && lighting.timeOfDay.available) {
            lighting.timeOfDay.hour = link.hour;
            lighting.timeOfDay.dayOfYear = link.dayOfYear;
            this.core.ddgi?.setEnvironment(lighting.timeOfDay.setEnabled(true, now));
        }
        const motion = this.url.motion;
        if (motion && lighting.player.motions.some((candidate) => candidate.id === motion)) this.playMotion(motion);
        else this.url.motion = null;
        this.refreshLightingControls();
    }

    private disposeLighting(): void {
        const lighting = this.lighting;
        if (!lighting) return;
        lighting.meter.cancel();
        lighting.player.dispose();
        this.cabinets?.dispose();
        this.cabinets = null;
        lighting.editor.dispose();
        lighting.timeOfDay.dispose();
        this.lighting = null;
        this.restoreAfterMeter();
    }

    /** Before GI's update and the render: motions, a played day, the meter's step, the light markers. */
    private beforeLightingFrame(now: number): void {
        const lighting = this.lighting;
        if (!lighting || this.loadingScene) return;
        lighting.player.update(now);
        const environment = lighting.timeOfDay.update(now);
        if (environment !== undefined) {
            this.core.ddgi?.setEnvironment(environment);
            this.url.timeOfDay = { hour: lighting.timeOfDay.hour, dayOfYear: lighting.timeOfDay.dayOfYear };
        }
        lighting.meter.beforeFrame(now);
        lighting.editor.update();
    }

    private playMotion(motionId: string | null): void {
        const lighting = this.lighting;
        if (!lighting) return;
        if (lighting.meter.running) lighting.meter.cancel('a motion was started.');
        if (motionId === null) {
            lighting.player.stop();
        } else {
            lighting.player.play(motionId, performance.now());
            const motion = lighting.player.playing;
            const moved = motion ? lighting.player.changedBy(motion) : null;
            if (moved) this.core.ddgi?.markChanged(moved);
        }
        this.url.motion = lighting.player.playing?.id ?? null;
        this.scheduleUrlUpdate();
        this.refreshLightingControls();
    }

    /** Turns time of day on or off: the sun light, GI's sky and the sky picture follow; DDGI starts over with or without its sun layer. */
    private setTimeOfDay(enabled: boolean): void {
        const lighting = this.lighting;
        if (!lighting || !lighting.timeOfDay.available) return;
        const environment = lighting.timeOfDay.setEnabled(enabled, performance.now());
        this.url.timeOfDay = enabled ? { hour: lighting.timeOfDay.hour, dayOfYear: lighting.timeOfDay.dayOfYear } : null;
        // A new sun layer setting starts a new DDGI (ViewerCore.configureDdgi); the environment goes to whichever DDGI there is.
        this.applyState();
        this.core.ddgi?.setEnvironment(environment);
    }

    private lightingHandlers(): LightingHandlers {
        const selected = () => this.lighting?.editor.selected ?? null;
        return {
            onSelectLight: (index) => {
                this.lighting?.editor.select(index);
                this.refreshLightingControls();
            },
            onLightSettings: (settings: LightSettings) => {
                const entry = selected();
                if (!entry || !this.lighting) return;
                this.core.ddgi?.markChanged(this.lighting.editor.apply(entry, settings));
            },
            onPutBack: () => {
                const entry = selected();
                if (!entry || !this.lighting) return;
                this.core.ddgi?.markChanged(this.lighting.editor.putBack(entry));
                this.refreshLightingControls();
            },
            onGizmoMode: (mode) => {
                this.lighting?.editor.setGizmoMode(mode);
                this.refreshLightingControls();
            },
            onPlayMotion: (motionId) => this.playMotion(motionId),
            onTimeOfDay: (enabled) => this.setTimeOfDay(enabled),
            onTime: (hour, dayOfYear) => {
                const timeOfDay = this.lighting?.timeOfDay;
                if (!timeOfDay) return;
                timeOfDay.stopPlaying();
                const environment = timeOfDay.setTime(hour, dayOfYear, performance.now());
                // Moving a slider turns time of day on (and DDGI's sun layer with it).
                if (!timeOfDay.enabled) this.setTimeOfDay(true);
                else this.core.ddgi?.setEnvironment(environment);
                this.url.timeOfDay = { hour, dayOfYear };
                this.scheduleUrlUpdate();
            },
            onPlayDay: () => {
                const timeOfDay = this.lighting?.timeOfDay;
                if (!timeOfDay) return;
                if (timeOfDay.playing) {
                    timeOfDay.stopPlaying();
                } else {
                    if (!timeOfDay.enabled) this.setTimeOfDay(true);
                    timeOfDay.playDay(performance.now());
                }
                this.refreshLightingControls();
            },
            onLightGroups: (count: LightGroupCount) => {
                this.url.lightGroups = count;
                this.applyState();
            },
            onGroupWeight: (key, weight) => {
                if (key === 'environment') this.environmentWeight = weight;
                else this.groupWeights[key] = weight;
                // Weights are live in DDGI: no new DDGI, nothing traced again.
                this.core.configureDdgi(this.ddgiSettings());
            },
            onPickRegion: () => {
                this.regionPicker.pick((region) => {
                    this.lightingControls.setMeterRegion(region ? 'Watching the picked rectangle (dashed blue).' : 'Watching the whole picture.');
                });
            },
            onWholePicture: () => {
                this.regionPicker.clear();
                this.lightingControls.setMeterRegion('Watching the whole picture.');
            },
            onMeasureStep: (motionId, jump) => this.startMeasurement(motionId, jump),
            onCancelMeasure: () => {
                this.lighting?.meter.cancel();
                this.restoreAfterMeter();
            },
            onSaveCsv: () => {
                const result = this.lighting?.meter.result;
                if (!result || !this.manifest) return;
                const link = document.createElement('a');
                link.href = URL.createObjectURL(new Blob([result.csv], { type: 'text/csv' }));
                link.download = `response-${this.manifest.id}-${result.motion}.csv`;
                link.click();
                URL.revokeObjectURL(link.href);
            },
        };
    }

    /** Starts the response meter on a step motion: in GI only, without the light markers, at the picked region. */
    private startMeasurement(motionId: string, jump: number): void {
        const lighting = this.lighting;
        if (!lighting) return;
        const motion = lighting.player.motions.find((candidate) => candidate.id === motionId);
        if (!motion) return;
        const ddgi = this.core.ddgi;
        if (!ddgi || !this.core.activeTechniques().includes(ddgi)) {
            this.lightingControls.setMeterStatus("The meter reads DDGI's picture: choose DDGI as the technique first.", false);
            return;
        }
        if (this.url.upscale) {
            this.lightingControls.setMeterStatus('Switch the upscaler off to measure (the meter reads the picture before it).', false);
            return;
        }
        if (lighting.player.playing) this.playMotion(null);
        if (lighting.timeOfDay.playing) lighting.timeOfDay.stopPlaying();
        // GI only, so three's own direct light, which changes at once, doesn't count (m2-lights.md §7).
        if (this.giBeforeMeter === null) this.giBeforeMeter = this.url.gi;
        if (this.url.gi !== 'only') {
            this.url.gi = 'only';
            this.applyState();
        }
        lighting.editor.setHelpersVisible(false);
        const size = this.core.renderer.getDrawingBufferSize(new Vector2());
        lighting.meter.region = this.regionPicker.pixels(size.x, size.y);
        lighting.meter.start(motion, performance.now(), jump);
        this.refreshLightingControls();
    }

    /** The GI choice and the light markers as they were before a measurement. */
    private restoreAfterMeter(): void {
        this.lighting?.editor.setHelpersVisible(true);
        const gi = this.giBeforeMeter;
        this.giBeforeMeter = null;
        if (gi !== null && gi !== this.url.gi) {
            this.url.gi = gi;
            this.applyState();
        }
    }

    /**
     * The lighting controls as things are now: with `everything`, the chosen light's sliders
     * and the group weights too (after a choice, a gizmo move or a scene change). The
     * periodic refresh leaves those alone, so a slider being dragged isn't written over.
     */
    private refreshLightingControls(everything = true): void {
        const lighting = this.lighting;
        const controls = this.lightingControls;
        if (!lighting || !controls) return;
        const { editor, player, timeOfDay, meter } = lighting;
        if (everything) {
            const entry = editor.selected;
            const index = entry ? editor.lights.indexOf(entry) : null;
            controls.setLight(index, entry ? { kind: entry.kind, settings: editor.settings(entry) } : null, entry ? (entry.original.intensity > 0 ? entry.original.intensity : entry.onIntensity) : 1);
            controls.setGizmoMode(editor.gizmoMode);
            const ddgiShown = this.url.gi !== 'off' && (this.url.wipe ? this.url.wipe.includes('ddgi') : this.url.technique === 'ddgi');
            controls.setGroups(this.url.lightGroups, this.groupWeights, this.environmentWeight, ddgiShown);
        }
        const now = performance.now();
        const playing = player.playing;
        controls.setMotionPlaying(playing?.id ?? null, playing ? `Playing ${playing.label}: ${player.seconds(now).toFixed(1)} s` : '');
        controls.setTimeOfDayEnabled(timeOfDay.enabled, timeOfDay.playing);
        if (timeOfDay.playing) controls.setTime(timeOfDay.hour, timeOfDay.dayOfYear);
        controls.setMeterStatus(meter.describe(now), meter.running);
        controls.setMeterResult(meter.result, meter.live);
        this.refreshScenarioControls(now);
    }

    // -- moving things: the scenarios (T3.8) ---------------------------------

    /** What a scenario sets in the link: its scene, camera view, motion and volumes. */
    private scenarioLink(scenario: Scenario): Partial<ViewerUrlState> {
        return {
            scenario: scenario.id,
            scene: scenario.scene,
            camera: scenario.camera,
            freeCamera: null,
            motion: scenario.motion,
            followCamera: scenario.followCamera,
        };
    }

    /** The viewer's camera for camera motions: the rig, so orbiting carries on from where the motion leaves it. */
    private cameraControl(): CameraControl {
        return {
            pose: () => {
                const pose = this.rig.pose();
                return { position: pose.position, target: pose.target };
            },
            place: (position, target) => this.rig.goTo({ name: 'motion', position, target, fovYDeg: this.core.camera.fov }),
        };
    }

    private cameraMotionPlaying(): boolean {
        const playing = this.lighting?.player.playing;
        return playing !== null && playing !== undefined && motionTarget(playing)?.kind === 'camera';
    }

    /**
     * Opens a scenario: its scene (if another is open, or its probe volumes
     * differ), its camera view and its motion, playing.
     */
    private chooseScenario(scenario: Scenario | null): void {
        if (!scenario) {
            this.url.scenario = null;
            this.scheduleUrlUpdate();
            this.refreshScenarioControls(performance.now());
            return;
        }
        const manifest = this.manifests.find((candidate) => candidate.id === scenario.scene);
        if (!manifest) return;
        const sameScene = this.manifest?.id === scenario.scene && !this.loadingScene;
        this.url = { ...this.url, ...this.scenarioLink(scenario) };
        if (!sameScene) {
            // A new scene opens at its own exposure and sun; the motion starts once it is up (`startLightingFromLink`).
            this.url = { ...this.url, ev: null, timeOfDay: null };
            void this.openScene(manifest, scenario.camera);
            return;
        }
        const index = manifest.cameras.findIndex((camera) => camera.name === scenario.camera);
        if (index >= 0) this.panelHandlers().onBookmark(index);
        // The camera-following volume changes start a new DDGI (`configureDdgi`).
        this.applyState();
        this.playMotion(scenario.motion);
    }

    private scenarioHandlers(): ScenarioHandlers {
        return {
            onScenario: (id) => this.chooseScenario(findScenario(id)),
            onPlayScenario: (play) => {
                const scenario = findScenario(this.url.scenario);
                if (!scenario) return;
                this.playMotion(play ? scenario.motion : null);
            },
            onMeasureScenario: () => {
                const scenario = findScenario(this.url.scenario);
                if (!scenario?.stepMotion) return;
                // The step's first jump: the door shutting, the cabinet moving out, the tree leaning.
                this.lightingControls.setMeterStep(scenario.stepMotion, 0);
                this.startMeasurement(scenario.stepMotion, 0);
            },
            onWindStrength: (strength) => {
                if (this.lighting) this.lighting.player.windStrength = strength;
                this.url.windStrength = strength;
                this.scheduleUrlUpdate();
            },
            onAddCabinet: () => {
                this.cabinets?.add();
                this.refreshScenarioControls(performance.now());
            },
            onCopyCabinet: () => {
                this.cabinets?.copy();
                this.refreshScenarioControls(performance.now());
            },
            onClearCabinets: () => {
                this.cabinets?.clear();
                this.refreshScenarioControls(performance.now());
            },
        };
    }

    /** The scenario section as things are now: the chosen scenario, whether its motion plays, the added cabinets. */
    private refreshScenarioControls(now: number): void {
        const controls = this.scenarioControls;
        const player = this.lighting?.player;
        if (!controls) return;
        const scenario = findScenario(this.url.scenario);
        const here = scenario !== null && player !== undefined && this.manifest?.id === scenario.scene && player.motions.some((motion) => motion.id === scenario.motion);
        controls.setScenario(scenario, here);
        const playing = here && player.playing?.id === scenario.motion;
        let status = '';
        const sceneTitle = this.manifests.find((manifest) => manifest.id === scenario?.scene)?.title ?? scenario?.scene;
        if (scenario && !here) status = this.loadingScene ? 'Opening the scene...' : `Choose it again to open ${sceneTitle}.`;
        else if (playing) status = `Playing: ${player.playing!.label}, ${player.seconds(now).toFixed(1)} s.`;
        else if (scenario) status = 'Stopped. Press Play to watch it again.';
        controls.setPlaying(playing, status);
        if (this.cabinets) controls.setCabinetCount(this.cabinets.count, this.cabinets.shapes);
    }

    /** A few times a second: the motion's time, a played day's sliders, the meter's progress and curve; and after a measurement, the GI choice back. */
    private updateLightingUi(now: number): void {
        const lighting = this.lighting;
        if (!lighting || now - this.lastLightingUiAt < LIGHTING_UI_INTERVAL_MS) return;
        this.lastLightingUiAt = now;
        if (!lighting.meter.running && this.giBeforeMeter !== null) this.restoreAfterMeter();
        // A played day writes its time into the link when it stops.
        if (lighting.timeOfDay.playing) this.scheduleUrlUpdate();
        this.regionPicker.refresh();
        this.refreshLightingControls(false);
    }

    // -- loading and failure -----------------------------------------------

    private showLoading(title: string, progress: LoadProgress): void {
        this.loadingBox.classList.remove('done', 'failed');
        const box = this.loadingBox;
        box.querySelector('.title')!.textContent = title;
        box.querySelector('.stage')!.textContent = progress.stage;
        const bar = box.querySelector('progress') as HTMLProgressElement;
        if (progress.fraction === null) bar.removeAttribute('value');
        else bar.value = progress.fraction;
    }

    private fail(error: unknown): void {
        const message = error instanceof Error ? error.message : String(error);
        this.automation.errors.push(message);
        console.error(message);
        this.loadingBox.classList.remove('done');
        this.loadingBox.classList.add('failed');
        this.loadingBox.querySelector('.title')!.textContent = 'Something went wrong';
        this.loadingBox.querySelector('.stage')!.textContent = message;
    }
}
