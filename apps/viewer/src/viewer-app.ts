/**
 * The viewer page: wires the panel, the camera, the stats overlay and the
 * link in the address bar to a `ViewerCore`.
 *
 * `window.__viewer` is what automated runs read (the contact sheet): `ready`
 * becomes true once the scene is drawn and the techniques on screen have
 * finished their set-up, and `errors` lists every problem seen.
 */

import { BoxGeometry, Color, Mesh, MeshStandardMaterial } from 'three';
import type { CameraBookmark, SceneManifest } from '../../../scenes/src/manifest';
import { CameraRig } from './controls/camera-rig';
import { ViewerCore, type ToneMappingName, type ViewLayout } from './core/viewer-core';
import { newestFrame, resolveGpuWindow } from './render/gpu-time';
import { findBookmark, loadAllManifests } from './scene/manifests';
import { LAYER_VIEW_GEOMETRY } from './scene/layers';
import { nextFrame, type LoadProgress } from './scene/load-scene';
import { StatsOverlay } from './stats';
import { DEFAULT_LPG_SETTINGS } from './techniques/light-probe-grid-technique';
import { DEFAULT_VXGI_SETTINGS } from './techniques/vxgi-technique';
import { TECHNIQUE_LABELS, type TechniqueId } from './techniques/types';
import { HELP } from './ui/help-text';
import { Panel, type PanelState } from './ui/panel';
import { DEFAULT_URL_STATE, parseUrlState, serializeUrlState, type ViewerUrlState } from './ui/url-state';

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
    };

    private core!: ViewerCore;
    private rig!: CameraRig;
    private panel!: Panel;
    private stats!: StatsOverlay;

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
            this.panel = new Panel(document.body, this.manifests, this.panelHandlers());
            this.buildWipeLabels();
            this.installPageEvents();
        } catch (error) {
            this.fail(error);
            return;
        }

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
        try {
            const loaded = await this.core.loadScene(manifest, (progress) => this.showLoading(`Loading ${manifest.title}`, progress));
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
            this.rig.update(0);
            this.bookmarkPoseKey = JSON.stringify(this.rig.pose());

            this.showLoading(`Loading ${manifest.title}`, { stage: 'Compiling shaders', fraction: 0.9 });
            await nextFrame();
            this.applyState();
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
        this.core.setToneMapping(this.url.tone as ToneMappingName);
        this.core.setExposureEv(this.url.ev);
        this.core.setUpscale(this.url.upscale ? { quality: this.url.upscale } : null);
        this.core.setLayout(layoutFromState(this.url));
        // The picture changed: an automated run waits a few frames before its screenshot.
        this.automation.ready = false;
        this.settledFrames = 0;
        this.refreshPanel();
        this.updateWipeLabels();
        this.scheduleUrlUpdate();
    }

    private refreshPanel(): void {
        const state: PanelState = {
            sceneId: this.manifest?.id ?? null,
            bookmarkIndex: this.bookmarkIndex,
            navigation: this.rig.mode,
            flySpeed: this.rig.flySpeed,
            fieldOfView: this.core.camera.fov,
            exposureEv: this.url.ev,
            toneMapping: this.url.tone as ToneMappingName,
            referenceCube: this.referenceCubeVisible,
            gi: this.url.gi,
            technique: this.url.technique,
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
                this.url = { ...this.url, camera: null, freeCamera: null };
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
            onExposure: (ev: number) => change({ ev }),
            onToneMapping: (name: ToneMappingName) => change({ tone: name }),
            onReferenceCube: (visible: boolean) => {
                this.referenceCubeVisible = visible;
                if (this.referenceCube) this.referenceCube.visible = visible;
            },
            onGi: (choice: ViewerUrlState['gi']) => change({ gi: choice }),
            onTechnique: (id: ViewerUrlState['technique']) => change({ technique: id }),
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
                const wipe: [TechniqueId, TechniqueId] = left === right ? [left, left === 'none' ? 'lpg' : 'none'] : [left, right];
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
        this.leftLabel.textContent = TECHNIQUE_LABELS[wipe[0]];
        this.rightLabel.textContent = TECHNIQUE_LABELS[wipe[1]];
        const box = this.canvasBox();
        const x = `${box.left + this.url.split * box.width}px`;
        this.leftLabel.style.left = x;
        this.rightLabel.style.left = x;
        this.canvas.title = HELP.wipeLine;
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
        this.updateWipeLabels();
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
            const preparing = this.core.preparing;
            // While a set-up runs, most of the time goes to it (see VIEW_INTERVAL_WHILE_PREPARING_MS).
            const viewInterval = Math.min(
                VIEW_INTERVAL_WHILE_PREPARING_MAX_MS,
                Math.max(VIEW_INTERVAL_WHILE_PREPARING_MS, this.lastViewCostMs / VIEW_SHARE_WHILE_PREPARING),
            );
            if (!preparing || performance.now() - this.lastViewRenderAt >= viewInterval) {
                const viewStartedAt = performance.now();
                this.core.renderFrame();
                this.lastViewRenderAt = performance.now();
                this.lastViewCostMs = this.lastViewRenderAt - viewStartedAt;
                this.automation.frames++;
                if (this.lastViewStartedAt !== null) this.stats.recordFrame(viewStartedAt - this.lastViewStartedAt);
                this.lastViewStartedAt = viewStartedAt;
                this.updateTimestamps(preparing);
            }
            this.pumpPreparation(preparing, now);
            this.updateStatus(now);
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

    private updateStatus(now: number): void {
        if (now - this.lastStatusAt < 200 || !this.core.loaded) return;
        this.lastStatusAt = now;
        const lines = this.core.statusLines();
        const { stats } = this.core.loaded;
        lines.push(`${stats.triangles.toLocaleString('en')} triangles, ${stats.meshes} meshes`);
        this.panel.setStatus(lines);
        const baking = this.core.activeTechniques().find((technique) => !technique.ready);
        this.statusPill.textContent = baking ? `${TECHNIQUE_LABELS[baking.id]}: ${baking.describe()}` : '';
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
