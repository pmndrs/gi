/**
 * Our DDGI, through the three.js adapter (`@pmndrs/gi/three`).
 *
 * The scene is drawn like the probe grid's: a plain `renderer.render` into a
 * multisampled target. GI reaches three's materials through the adapter's
 * GI light, which lives on DDGI's own layer, so only DDGI's camera sees it
 * and the other techniques in a wipe are untouched. Each frame, before that
 * render, `ThreeGI.update` mirrors scene changes and submits the GI update
 * within the frame budget (2 ms by default).
 *
 * Start-up goes through the library's own `DDGIVolume.start` (T2.4b), the
 * simple path any three user takes, so the viewer shows what they get. In
 * the order the viewer shows it:
 *  1. **Preparing:** the platform compiles its kernels (`gi.prepare()`).
 *  2. **Building:** the first update copies the scene into the platform, and
 *     the mesh trees for ray tracing are built in a worker.
 *  3. **Warming up:** the DDGI technique's own warm-up (its `warmUpPasses`,
 *     16 passes over every probe, stopping after 8 s once 2 are done):
 *     what a game does behind its loading screen, because at 2 ms a frame
 *     filling Sponza's 19,425 probes from black would take a minute.
 *     Then the platform's calibration (budget v2): frames with the
 *     millisecond budget, GPU-bound (two in flight, so the GPU never waits)
 *     and not paced by the display. `start` runs those frames and resolves
 *     when the platform reports it calibrated and confirmed. "Start the
 *     probes over" runs them again (`DDGIVolume.restart`).
 *  4. **Ready:** the calibrated number of probes every frame, and the field
 *     keeps converging live. The viewer then runs the GI frames itself
 *     (`updateOnRender` off), so "GI off" means no GI work at all.
 *
 * Debug views (T1.12), for seeing where noise and error come from:
 *  - the probe spheres, coloured by state or by history (how many updates
 *    each probe's light is averaged over; `ProbeHistoryTexture`);
 *  - the screen views (`DDGI_SCREEN_VIEWS`: GI only, coverage, probes near
 *    this point), drawn through `scene.overrideMaterial`;
 *  - a probe atlas over the picture (`AtlasOverlay`);
 *  - the change per frame (`FrameChangeView`), which works on whatever DDGI
 *    draws: the normal picture, GI only, or a screen view.
 * Direct light against GI is the viewer's GI off / on / GI only choice.
 */

import { Color, type Object3D } from 'three';
import type { Node } from 'three/webgpu';
import type { SceneManifest } from '../../../../scenes/src/manifest';
import { platformSkyEnvironment } from '../../../../scenes/src/sky/sky-image';
import { AtlasOverlay, type AtlasChoice } from '../ddgi/atlas-overlay';
import { DdgiDebugMaterial } from '../ddgi/debug-material';
import { FrameChangeView } from '../ddgi/frame-change';
import {
    DDGI_DEBUG_VIEW,
    DDGIVolume,
    PROBE_STATE_LABELS,
    type DDGIPresetName,
    type DDGITechnique,
    type DDGIVolumeSettings,
    type LoadedField,
    type DebugView,
    type EnvironmentDescriptor,
    type FrameBudget,
    type GIPlatform,
    type StartedGI,
    type ThreeGI,
} from '../ddgi/library';
import { ProbeHistoryTexture } from '../ddgi/probe-history';
import { AlbedoAtHitsView, albedoViewPlatform } from '../ddgi/albedo-at-hits';
import { ProbeSpheres, type ProbeColorBy } from '../ddgi/probe-spheres';
import type { DdgiBudgetStats, DdgiFrameStats } from '../stats/ddgi-stats';
import { LAYER_DDGI, LAYER_DDGI_PROBES } from '../scene/layers';
import { ScenePassTechnique } from './scene-pass-technique';
import type { TechniqueContext } from './types';

/** DDGI's screen debug views by the name the viewer (and its links) use. */
export const DDGI_SCREEN_VIEWS = {
    'gi-only': DDGI_DEBUG_VIEW.giOnly,
    coverage: DDGI_DEBUG_VIEW.coverage,
    'probe-states': DDGI_DEBUG_VIEW.probeStates,
} as const;

/**
 * A screen debug view: one of DDGI's own (`DDGI_SCREEN_VIEWS`, drawn through
 * `giDebugColor`), or the viewer's `gi-albedo` (the albedo GI's rays read
 * where they hit, traced by the platform: ddgi/albedo-at-hits.ts).
 */
export type DdgiScreenView = keyof typeof DDGI_SCREEN_VIEWS | 'gi-albedo';

function isCoreScreenView(view: DdgiScreenView | null): view is keyof typeof DDGI_SCREEN_VIEWS {
    return view !== null && view in DDGI_SCREEN_VIEWS;
}

/** A light group's weight in GI (T2.5): a linear RGB colour times an intensity. */
export interface GroupWeight {
    color: [number, number, number];
    intensity: number;
}

/** Light groups the viewer offers: off, or 2 or 4 groups (each a layer of its own; m2-lights.md §5). */
export type LightGroupCount = 0 | 2 | 4;

export interface DdgiViewerSettings {
    /**
     * The quality preset (T5.1, `ddgiPreset`): its rays per probe, tile
     * sizes and texture store. The scene's manifest sets the probe spacing,
     * and `budgetMs` the budget (the panel moves it to the preset's when a
     * preset is picked). Changing it starts DDGI over: rays and tile sizes
     * size DDGI's resources when it starts (`layerKey`).
     */
    preset: DDGIPresetName;
    /** GPU milliseconds per frame for probe updates. */
    budgetMs: number;
    /** The technique's warm-up before GI shows (off: the probes fill in live from black). Read when DDGI starts. */
    warmUp: boolean;
    /**
     * Light groups with a layer of their own, 0 (off) to 4 (T2.5). Changing it
     * starts DDGI over: the layers are made when it starts (`layerKey`).
     */
    lightGroups: LightGroupCount;
    /**
     * The sun's own layer (T2.6b): the sun's colour and brightness then change
     * GI at once, with nothing traced again. Time of day turns it on (DECISIONS
     * 2026-11-08). Changing it starts DDGI over, as `lightGroups` does.
     */
    sunLayer: boolean;
    /**
     * A second, finer volume of probes that follows the camera (T3.6,
     * `CAMERA_VOLUME`), in front of the manifest's fixed one, which fills in
     * beyond it. For big scenes, where the fixed volume's probes are metres
     * apart (Bistro exterior: 4 m). Changing it starts DDGI over.
     */
    cameraVolume: boolean;
    /** Each group's weight (index = group), live; missing ones are white at 1. */
    groupWeights: GroupWeight[];
    /** The environment layer's weight (the sky and glowing surfaces), live. */
    environmentWeight: GroupWeight;
    /** Draw every probe as a small sphere. */
    showProbes: boolean;
    /** What the spheres' colour shows: the probe's state, or how many updates its light is averaged over. */
    probeColor: ProbeColorBy;
    /** A screen debug view instead of the normal picture, or null. */
    screenView: DdgiScreenView | null;
    /** A probe atlas drawn over the picture, or null. */
    atlas: AtlasChoice | null;
    /** The atlas fills the view instead of sitting in the bottom-right corner. */
    atlasFill: boolean;
    /** Extra exposure for the irradiance atlas, in stops, on top of the viewer's. */
    atlasExposureEv: number;
    /** The change-per-frame view, full colour at this change in percent; null shows the picture itself. */
    changeFullScalePercent: number | null;
    /**
     * GI's rays read material textures where they hit (the platform's
     * hit-shading setting `sampleTextures`). Off, each texture counts as its
     * mean colour, as the adapter's averages did before M4: for comparing what
     * the textures change. Turning it over is one uniform write (nothing is
     * compiled); the field takes a few seconds to follow.
     */
    giTextures: boolean;
}

export const WHITE_WEIGHT: GroupWeight = { color: [1, 1, 1], intensity: 1 };

export const DEFAULT_DDGI_VIEWER_SETTINGS: DdgiViewerSettings = {
    preset: 'high',
    budgetMs: 2,
    warmUp: true,
    lightGroups: 0,
    sunLayer: false,
    cameraVolume: false,
    groupWeights: [],
    environmentWeight: WHITE_WEIGHT,
    showProbes: false,
    probeColor: 'state',
    screenView: null,
    atlas: null,
    atlasFill: false,
    atlasExposureEv: 0,
    changeFullScalePercent: null,
    giTextures: true,
};

/** The settings that shape DDGI's layers, volumes and tiles, which only a new start can change (`ViewerCore.configureDdgi`). */
export function layerKey(settings: Pick<DdgiViewerSettings, 'preset' | 'lightGroups' | 'sunLayer' | 'cameraVolume'>): string {
    return `preset ${settings.preset}, groups ${settings.lightGroups}, sun layer ${settings.sunLayer ? 'on' : 'off'}, camera volume ${settings.cameraVolume ? 'on' : 'off'}`;
}

/**
 * The camera-following volume (T3.6, m3-moving.md §4.2): 32 × 12 × 32 probes
 * 1.5 m apart (a window 46.5 m wide and 16.5 m high), Dennis's pick of
 * 2026-11-05, as `ddgi-camera-volume` checks it on Bistro's fly-through.
 * Priority 1, so it answers before the manifest's volume wherever it covers
 * the point, and fades into it over its last 10 spacings (DECISIONS
 * 2026-11-10).
 */
export const CAMERA_VOLUME = { probes: [32, 12, 32] as const, spacing: 1.5, priority: 1 };

/**
 * The volumes DDGI starts with: the manifest's, and with `cameraVolume` the
 * camera-following one first, its window starting around `start` (the
 * scene's first camera view; it moves to wherever the camera is from the
 * first warm-up frame on, since the warm-up runs with the viewer's camera).
 */
export function ddgiVolumes(manifest: Pick<SceneManifest, 'volume'>, cameraVolume: boolean, start: readonly [number, number, number]): DDGIVolumeSettings[] {
    const fixed: DDGIVolumeSettings = { center: manifest.volume.center, size: manifest.volume.size, spacing: manifest.volume.probeSpacing };
    if (!cameraVolume) return [fixed];
    const { probes, spacing, priority } = CAMERA_VOLUME;
    // n probes span n - 1 spacings.
    const size: [number, number, number] = [(probes[0] - 1) * spacing, (probes[1] - 1) * spacing, (probes[2] - 1) * spacing];
    return [{ center: [start[0], start[1], start[2]], size, spacing, follow: 'camera', priority }, fixed];
}

/** Set-up steps the viewer records per batch (`pumpPrepare`); `start` runs the real frames itself, so a step only looks at its progress. */
const PREPARE_STEPS_PER_BATCH = 1;

/** The adapter's warning about area lights. The viewer's emissive quads carry emissive materials, which GI does see, so it is expected. */
const EXPECTED_WARNING = 'RectAreaLight is not mirrored into GI';

type Phase = 'preparing' | 'building' | 'warming' | 'calibrating' | 'ready' | 'failed';

/**
 * What the stats overlay shows for DDGI, as of the latest frame. One object per technique,
 * refilled each frame (`statsSnapshot`), so drawing stats allocates nothing of its own.
 */
export type DdgiStatsSnapshot = DdgiFrameStats & {
    updatePasses: Map<string, number>;
    platformPasses: Map<string, number>;
    timingsPending: boolean;
    states: Array<{ label: string; count: number }>;
};

/**
 * Where each probe state's count is in `DDGITechnique.counters`: the core names those
 * counters "probes " + the state's label (`PROBE_STATE_LABELS`) and exports no constant for
 * them, so this table is the one place in the viewer that relies on that. A count that
 * isn't there reads as unknown, never as a guess.
 */
const STATE_COUNTERS: ReadonlyArray<{ readonly state: number; readonly label: string; readonly counter: string }> = Object.entries(PROBE_STATE_LABELS).map(
    ([state, label]) => ({ state: Number(state), label, counter: `probes ${label}` }),
);

/** The counter DDGI reports its probes-per-frame under. */
const PROBES_UPDATED_COUNTER = 'probes updated';
/** The counter DDGI reports how long a pass over every probe takes under (seconds). */
const SWEEP_COUNTER = 'seconds per full sweep';
/** The technique's counter of full passes its warm-up has made. */
const WARM_UP_PASSES_COUNTER = 'warm-up passes';

/** Start-up times, milliseconds from when the technique was created. Null until reached. */
export interface DdgiStartupTimes {
    preparedMs: number | null;
    /** The scene copied and its ray-tracing trees built: the first frame with real GI (seen at the viewer's next frame). */
    firstGiMs: number | null;
    warmedMs: number | null;
}

export class DdgiTechnique extends ScenePassTechnique {
    readonly id = 'ddgi';
    readonly label = 'DDGI (ours)';
    readonly prepareStepsPerBatch = PREPARE_STEPS_PER_BATCH;
    readonly startup: DdgiStartupTimes = { preparedMs: null, firstGiMs: null, warmedMs: null };

    /** The library's volume on the manifest's probe grid; `start` makes the platform, DDGI and ThreeGI on it. */
    private readonly volume: DDGIVolume;
    /** What `volume.start` made, from the moment it exists (before the warm-up). */
    private gi: StartedGI | null = null;
    private readonly createdAt = performance.now();
    /** Resolves once the platform is prepared and ThreeGI exists (the benchmark's `warmUp`). */
    private readonly created: Promise<void>;
    /** `volume.start` or `volume.restart` while it runs; null when the probes are warm. */
    private starting: Promise<void> | null;
    private spheres: ProbeSpheres | null = null;
    private probeHistory: ProbeHistoryTexture | null = null;
    private debugMaterial: DdgiDebugMaterial | null = null;
    private atlasOverlay: AtlasOverlay | null = null;
    /** The `gi-albedo` screen view, made when the platform can trace it (ddgi/albedo-at-hits.ts). */
    private albedoView: AlbedoAtHitsView | null = null;
    private readonly frameChange = new FrameChangeView();
    /** The share of the view's height at the top that the stats box covers; the atlas stays below it. */
    private overlayTopInsetShare = 0;
    private settings: DdgiViewerSettings;
    private budget: FrameBudget;
    /** The layer settings this DDGI started with (`layerKey`): changing them needs a new DDGI. */
    readonly layerKey: string;
    /** GI's environment set by the viewer (time of day), or null for the manifest's. */
    private environmentOverride: EnvironmentDescriptor | null = null;
    /** The response meter's test-mode budget (GPU ms), over the user's; null when not measuring. */
    private budgetOverrideMs: number | null = null;
    private failure: string | null = null;
    private generation = 0;
    private finishedWallMs: number | null = null;
    /** The platform frame the start-up ended on: GPU timings from it or before are start-up work, not normal frames. */
    private warmupEndFrame = Number.POSITIVE_INFINITY;
    private disposed = false;
    private readonly warnings = new Set<string>();
    private readonly snapshot: DdgiStatsSnapshot = {
        probeCount: 0,
        probesUpdated: 0,
        sweepSeconds: null,
        budget: null,
        budgetMs: 0,
        updatePasses: new Map(),
        platformPasses: new Map(),
        timingsPending: false,
        memory: { sceneCopy: 0, acceleration: 0, techniques: {}, total: 0 },
        syncCpuMs: 0,
        encodeCpuMs: 0,
        recompiles: 0,
        states: [],
    };

    constructor(context: TechniqueContext, settings: DdgiViewerSettings = DEFAULT_DDGI_VIEWER_SETTINGS) {
        super(context);
        this.settings = { ...settings };
        this.budget = { gpuMs: settings.budgetMs };
        this.layerKey = layerKey(settings);
        // The base class chose layers before the settings existed; choose again with them.
        this.applyLayers();
        const { manifest } = context.loaded;
        this.volume = DDGIVolume.fromSettings({ center: manifest.volume.center, size: manifest.volume.size, spacing: manifest.volume.probeSpacing });
        // Only DDGI's camera sees the volume (a light in three's terms), as its GI light.
        this.volume.layers.set(LAYER_DDGI);
        let created: () => void = () => {};
        this.created = new Promise<void>((resolve) => (created = resolve));
        this.starting = this.start(created).catch((error: unknown) => this.fail(error));
    }

    // -- set-up ------------------------------------------------------------

    private async start(created: () => void): Promise<void> {
        const { loaded, renderer } = this.context;
        await this.volume.start(renderer, loaded.scene, {
            // Rays, tile sizes and the texture store; the manifest's spacing and the panel's budget stay.
            preset: this.settings.preset,
            budget: this.budget,
            ...(this.settings.warmUp ? {} : { warmUp: false }),
            ddgi: {
                // The weights are set once the technique exists (`applyGroupWeights`); they are live.
                lightGroups: this.settings.lightGroups > 0 ? { count: this.settings.lightGroups, weights: [], environmentWeight: [1, 1, 1] } : null,
                sunLayer: this.settings.sunLayer,
                // Several volumes only with the camera's; one volume stays exactly M2's path (the DDGIVolume's own).
                ...(this.settings.cameraVolume ? { volumes: ddgiVolumes(loaded.manifest, true, loaded.manifest.cameras[0].position) } : {}),
            },
            environment: this.environmentOverride ?? environmentFor(loaded.manifest),
            // The warm-up runs with the viewer's camera (synced each frame, `syncCamera`), so the camera-following
            // volume warms up where the camera is: with the default (a camera at the volume's centre) its window
            // jumped to the viewer's camera on the first frame after the warm-up, and warmed up again (NOTES, T3.8).
            camera: this.camera,
            warn: (message) => this.warn(message),
            label: `viewer ${loaded.manifest.id}`,
            onCreated: (started) => {
                this.attach(started);
                created();
            },
        });
        if (this.disposed) return;
        this.finishStartUp();
    }

    /** The platform, DDGI and ThreeGI exist (prepared, before the warm-up): the debug views can be made. */
    private attach(started: StartedGI): void {
        if (this.disposed) return;
        this.gi = started;
        this.startup.preparedMs = performance.now() - this.createdAt;
        const { loaded } = this.context;
        // The adapter puts its light on every layer; here only DDGI's camera may see it, so the other
        // techniques in a wipe (and the probe grid's bake cameras on layer 0) get no DDGI light.
        // The adapter keeps the layers when it replaces the light.
        started.threeGI.light.layers.set(LAYER_DDGI);

        const points = this.pointsView;
        if (points) {
            const history = new ProbeHistoryTexture(this.context.device);
            this.probeHistory = history;
            // The history texture has the points texture's shape: probe i at texel (i mod width, i div width).
            this.spheres = new ProbeSpheres(points, loaded.manifest.volume.probeSpacing, {
                texture: () => history.texture(points.texture().width, points.texture().height),
                maxSamples: () => started.technique.settings.hysteresisMaxSamples,
            });
            loaded.scene.add(this.spheres.mesh);
        }
        this.debugMaterial = new DdgiDebugMaterial(started.platform.shading);
        this.atlasOverlay = new AtlasOverlay(started.platform.shading);
        const internals = albedoViewPlatform(started.platform);
        this.albedoView = internals ? new AlbedoAtHitsView(this.context.device, internals) : null;
        this.applyDisplaySettings();
        this.applyGroupWeights();
        // An environment set after `start` was called (time of day, on a new DDGI) reaches GI here.
        if (this.environmentOverride) started.threeGI.environment = this.environmentOverride;
    }

    /** The group and environment weights into GI (live: a new weight shows in the next frame, nothing is traced again). */
    private applyGroupWeights(): void {
        const threeGI = this.threeGI;
        const groups = this.startedGroups;
        if (!threeGI || groups === 0) return;
        for (let group = 0; group < groups; group++) {
            const weight = this.settings.groupWeights[group] ?? WHITE_WEIGHT;
            threeGI.setGroupWeight(group, new Color().setRGB(...weight.color), weight.intensity);
        }
        const environment = this.settings.environmentWeight;
        threeGI.setEnvironmentWeight(new Color().setRGB(...environment.color), environment.intensity);
    }

    /** The light groups this DDGI started with (the settings may already ask for others). */
    private get startedGroups(): number {
        return this.ddgi?.settings.lightGroups?.count ?? 0;
    }

    /**
     * GI's environment instead of the manifest's (time of day moves the sun
     * and the sky), or null to go back to it. The adapter sends a changed
     * environment as a scene change: the sky is baked again on the GPU and
     * the probes it changes are marked.
     */
    /** GI's environment set by the viewer (time of day), or null for the manifest's. */
    get environment(): EnvironmentDescriptor | null {
        return this.environmentOverride;
    }

    setEnvironment(environment: EnvironmentDescriptor | null): void {
        this.environmentOverride = environment;
        const threeGI = this.threeGI;
        if (threeGI) threeGI.environment = environment ?? environmentFor(this.context.loaded.manifest);
    }

    /**
     * The response meter's test mode: this budget (GPU ms per frame) instead
     * of the user's while it finds the settled picture; null goes back.
     */
    setBudgetOverride(gpuMs: number | null): void {
        this.budgetOverrideMs = gpuMs;
        this.budget = { gpuMs: gpuMs ?? this.settings.budgetMs };
        if (this.threeGI && this.phase === 'ready') this.threeGI.budget = this.budget;
    }

    /**
     * Tells GI that `object` changed, so the next update sends it (the adapter
     * would otherwise find a light past its first 64, or one just added, a few
     * frames later). Nothing happens before GI exists.
     */
    markChanged(object: Object3D): void {
        this.threeGI?.markChanged(object);
    }

    /** Whether this DDGI runs with its sun layer (its technique's `sun layer` counter, T2.6b). */
    get sunLayerOn(): boolean {
        return this.ddgi?.counters['sun layer'] === 1;
    }

    /** Probes DDGI updated in its latest frame. */
    get probesUpdated(): number {
        return this.ddgi?.counters[PROBES_UPDATED_COUNTER] ?? 0;
    }

    /** Compute pipelines the platform made while ready (`gi.stats.recompiles`); it should stay 0. */
    get recompiles(): number {
        return this.platform?.stats.recompiles ?? 0;
    }

    /**
     * The mean of DDGI's picture (linear, before exposure and tone mapping)
     * over a region of the drawing buffer, read back from the frame just
     * drawn: the response meter's reading. Averaged over pixels and the three
     * colour channels, as the T2.4b page averages its test view. Null while
     * the upscaler draws the picture (the scene target isn't drawn then).
     * x, y from the top left, in drawing-buffer pixels.
     */
    readRegionMean(x: number, y: number, width: number, height: number): Promise<number> | null {
        if (this.upscaling) return null;
        const target = this.sceneTarget;
        const left = Math.max(0, Math.min(target.width - 1, Math.floor(x)));
        const top = Math.max(0, Math.min(target.height - 1, Math.floor(y)));
        const right = Math.max(left + 1, Math.min(target.width, Math.floor(x + width)));
        const bottom = Math.max(top + 1, Math.min(target.height, Math.floor(y + height)));
        return this.context.renderer.readRenderTargetPixelsAsync(target, left, top, right - left, bottom - top).then((pixels) => halfRgbaMean(pixels as Uint16Array));
    }

    /** `start` or `restart` resolved: the viewer runs the GI frames from here (`updateOnRender` off). */
    private finishStartUp(): void {
        const gi = this.gi;
        if (!gi) return;
        gi.threeGI.updateOnRender = false;
        gi.threeGI.budget = this.budget;
        this.starting = null;
        this.warmupEndFrame = gi.platform.stats.frame;
        this.startup.warmedMs ??= performance.now() - this.createdAt;
    }

    private fail(error: unknown): void {
        if (this.disposed) return;
        this.starting = null;
        this.failure = error instanceof Error ? error.message : String(error);
        console.error(`DDGI could not start: ${this.failure}`);
    }

    private warn(message: string): void {
        if (message.includes(EXPECTED_WARNING) || this.warnings.has(message)) return;
        this.warnings.add(message);
        console.warn(`DDGI: ${message}`);
    }

    /** The library's DDGI technique, once `start` made it. */
    private get ddgi(): DDGITechnique | null {
        return this.gi?.technique ?? null;
    }

    private get platform(): GIPlatform | null {
        return this.gi?.platform ?? null;
    }

    private get threeGI(): ThreeGI | null {
        return this.gi?.threeGI ?? null;
    }

    /** Where start-up is, from what `start` made and the technique's warm-up counters. */
    private get phase(): Phase {
        if (this.failure !== null) return 'failed';
        if (this.starting === null) return 'ready';
        const ddgi = this.ddgi;
        if (!ddgi) return 'preparing';
        if (ddgi.warmingUp) return (ddgi.counters[WARM_UP_PASSES_COUNTER] ?? 0) > 0 ? 'warming' : 'building';
        return 'calibrating';
    }

    get ready(): boolean {
        const phase = this.phase;
        return phase === 'ready' || phase === 'failed';
    }

    /** Past its warm-up: the budget applies, and per-frame numbers mean what they say. */
    get running(): boolean {
        return this.phase === 'ready';
    }

    get failed(): string | null {
        return this.failure;
    }

    get prepareGeneration(): number {
        return this.generation;
    }

    get progress(): number {
        switch (this.phase) {
            case 'preparing':
                return 0;
            case 'building':
                return 0.05;
            case 'warming':
                return 0.1 + 0.85 * Math.min(1, this.warmupPasses / this.warmUpPassesWanted);
            case 'calibrating':
                return 0.95;
            default:
                return 1;
        }
    }

    private get probeCount(): number {
        return this.ddgi?.field?.grid.probeCount ?? 0;
    }

    /** Full passes over every probe the warm-up made (fractional while a pass is under way). */
    get warmupPasses(): number {
        return this.ddgi?.counters[WARM_UP_PASSES_COUNTER] ?? 0;
    }

    /** The passes the technique's warm-up makes at most (its `warmUpPasses`). */
    private get warmUpPassesWanted(): number {
        return Math.max(1, this.ddgi?.settings.warmUpPasses ?? 1);
    }

    /**
     * One set-up step. `volume.start` (or `restart`) runs the warm-up's and the calibration's
     * frames itself, so this only notes how far it got (the first frame with real GI).
     */
    encodePrepareStep(): boolean {
        if (this.startup.firstGiMs === null && this.warmupPasses > 0) this.startup.firstGiMs = performance.now() - this.createdAt;
        return this.ready;
    }

    /**
     * Saves the probes as a `.ddgibake` file's bytes (`DDGIVolume.saveBake`):
     * one GPU readback. Only once DDGI is running (warm), so the file holds
     * converged probes.
     */
    saveBake(): Promise<Uint8Array<ArrayBuffer>> {
        if (this.phase !== 'ready') return Promise.reject(new Error('DDGI is still starting; save the probes once they are warm.'));
        return this.volume.saveBake();
    }

    /**
     * Loads a `.ddgibake` file into the running probes (`DDGIVolume.loadBake`):
     * from the next frame the picture shows the loaded probes, which keep
     * updating. Throws (with a message that says why) for a file that isn't
     * a bake or doesn't fit this scene's probes; a bake of a changed scene
     * loads as stale.
     */
    loadBake(bytes: ArrayBuffer): LoadedField {
        if (this.phase !== 'ready') throw new Error('DDGI is still starting; load the probes once it is running.');
        return this.volume.loadBake(bytes);
    }

    /** Saved bakes loaded into this DDGI (its technique's counter), for automated runs. */
    get bakesLoaded(): number {
        return this.ddgi?.counters['saved fields loaded'] ?? 0;
    }

    /** "Start the probes over": forget the field and the calibration, and warm up again (`DDGIVolume.restart`). */
    restartPrepare(): void {
        this.generation++;
        this.finishedWallMs = null;
        if (this.phase !== 'ready') return;
        this.warmupEndFrame = Number.POSITIVE_INFINITY;
        this.starting = this.volume
            .restart({ budget: this.budget, camera: this.camera })
            .then(() => {
                if (!this.disposed) this.finishStartUp();
            })
            .catch((error: unknown) => this.fail(error));
    }

    notePrepareFinished(generation: number, wallMs: number): void {
        if (generation === this.generation && this.phase === 'ready') this.finishedWallMs = wallMs;
    }

    /** Waits until the platform is prepared and DDGI exists (for the benchmark); the warm-up runs on from there. */
    async warmUp(): Promise<void> {
        await Promise.race([this.created, this.starting ?? Promise.resolve()]);
    }

    // -- settings ----------------------------------------------------------

    configure(settings: DdgiViewerSettings): void {
        const probesChanged = settings.showProbes !== this.settings.showProbes;
        this.settings = { ...settings };
        // A new object: ThreeGI reads its `budget` field every update.
        this.budget = { gpuMs: this.budgetOverrideMs ?? settings.budgetMs };
        // While `start` runs it owns the frames, and it takes the budget it was given; later frames take this one.
        if (this.threeGI && this.phase === 'ready') this.threeGI.budget = this.budget;
        this.applyDisplaySettings();
        this.applyGroupWeights();
        if (probesChanged) this.applyLayers();
    }

    get currentSettings(): Readonly<DdgiViewerSettings> {
        return this.settings;
    }

    /** Settings that are uniforms: no shader is rebuilt. (The atlas choice and the change view change `overlayKey`.) */
    private applyDisplaySettings(): void {
        if (this.debugMaterial && isCoreScreenView(this.settings.screenView)) this.debugMaterial.view = DDGI_SCREEN_VIEWS[this.settings.screenView];
        // GI's textures on or off: the platform's host-facing hit-shading setting (DECISIONS 2026-11-18).
        const hitShading = this.platform?.hitShadingSettings;
        if (hitShading && hitShading.sampleTextures !== this.settings.giTextures) hitShading.sampleTextures = this.settings.giTextures;
        if (this.spheres) this.spheres.colorBy = this.settings.probeColor;
        if (this.settings.changeFullScalePercent !== null) this.frameChange.fullScalePercent = this.settings.changeFullScalePercent;
    }

    // -- the debug overlays --------------------------------------------------

    /** DDGI's `points` debug view (the probe positions and states). */
    private get pointsView(): Extract<DebugView, { kind: 'points' }> | undefined {
        return this.ddgi?.debugViews.find((view): view is Extract<DebugView, { kind: 'points' }> => view.kind === 'points');
    }

    /**
     * Whether the change per frame is on screen. It compares DDGI's own scene target, which the
     * upscaler doesn't draw into, so it is off while the upscaler is on.
     */
    private get showsChange(): boolean {
        // `settings` is still unset while the base class's constructor runs.
        const settings = this.settings as DdgiViewerSettings | undefined;
        if (!settings || settings.changeFullScalePercent === null) return false;
        return !this.upscaling && this.giMode !== 'off';
    }

    /** The atlas shown over the picture, once the platform has textures to show. */
    private get shownAtlas(): AtlasChoice | null {
        const settings = this.settings as DdgiViewerSettings | undefined;
        if (!settings?.atlas || !this.atlasOverlay || this.giMode === 'off') return null;
        return settings.atlas;
    }

    /** Whether the `gi-albedo` view replaces the picture. */
    private get showsAlbedo(): boolean {
        const settings = this.settings as DdgiViewerSettings | undefined;
        return settings?.screenView === 'gi-albedo' && this.albedoView !== null && this.phase !== 'failed';
    }

    /** What `overlay` draws; the viewer rebuilds its pipeline when this changes. */
    get overlayKey(): string {
        return `${this.showsAlbedo ? 'albedo' : 'shaded'}|${this.shownAtlas ?? 'no atlas'}|${this.showsChange ? 'change' : 'picture'}`;
    }

    /**
     * The debug overlays, drawn over `display` (DDGI's picture after tone mapping and output colour),
     * so their colours are exactly what reaches the screen: the change per frame, then the atlas on top.
     */
    overlay(display: Node): Node {
        let node = display;
        if (this.showsAlbedo && this.albedoView) node = this.albedoView.overlay(node);
        if (this.showsChange) node = this.frameChange.overlay(node, this.sceneTarget);
        const atlas = this.shownAtlas;
        if (atlas && this.atlasOverlay) node = this.atlasOverlay.overlay(node, atlas);
        return node;
    }

    /** The share of the view's height, from the top, that the stats box covers (0 when it is hidden). */
    setOverlayTopInset(share: number): void {
        this.overlayTopInsetShare = Number.isFinite(share) ? Math.min(1, Math.max(0, share)) : 0;
    }

    /** Once per frame, after DDGI's update and before anything is drawn: the debug views' inputs. */
    private refreshDebugViews(): void {
        const field = this.ddgi?.field ?? null;
        const points = this.pointsView;
        if (field && points && this.probeHistory && this.settings.showProbes && this.settings.probeColor === 'history') {
            this.probeHistory.update(field, points.texture().width, points.texture().height);
        }
        this.spheres?.refresh();
        if (field && this.atlasOverlay && this.shownAtlas) {
            this.atlasOverlay.refresh(this.context.renderer, field, {
                fill: this.settings.atlasFill,
                exposureStops: this.settings.atlasExposureEv,
                topInsetShare: this.overlayTopInsetShare,
            });
        }
        // Keep the last frame's picture before this frame draws over it.
        if (this.showsChange) this.frameChange.capturePrevious(this.context.renderer, this.sceneTarget, this.camera);
        else this.frameChange.forget();
    }

    protected giLayers(): readonly number[] {
        // `settings` is still unset while the base class's constructor runs.
        const settings = this.settings as DdgiViewerSettings | undefined;
        return settings?.showProbes ? [LAYER_DDGI, LAYER_DDGI_PROBES] : [LAYER_DDGI];
    }

    protected hasNoGi(): boolean {
        return false;
    }

    // -- the frame ---------------------------------------------------------

    renderBeforePipeline(): void {
        const threeGI = this.threeGI;
        // During warm-up the set-up steps do the updates; GI off means no GI work at all.
        if (threeGI && this.phase === 'ready' && this.giMode !== 'off') {
            threeGI.update(this.camera);
        }
        this.refreshDebugViews();
        if (this.showsAlbedo) this.albedoView?.render(this.context.renderer, this.camera);
        const scene = this.context.loaded.scene;
        const override = isCoreScreenView(this.settings.screenView) && this.debugMaterial && this.giMode !== 'off' && this.phase !== 'failed' ? this.debugMaterial : null;
        override?.refresh();
        const renderer = this.context.renderer;
        const previousOverride = scene.overrideMaterial;
        const previousClearAlpha = renderer.getClearAlpha();
        if (override) scene.overrideMaterial = override.material;
        // Alpha 1 where nothing is drawn: in DDGI's picture only the probe spheres write alpha 0,
        // which is how the change view finds them (frame-change.ts). The renderer itself clears to 0.
        renderer.setClearAlpha(1);
        try {
            super.renderBeforePipeline();
        } finally {
            scene.overrideMaterial = previousOverride;
            renderer.setClearAlpha(previousClearAlpha);
        }
    }

    // -- reporting ---------------------------------------------------------

    describe(): string {
        const field = this.ddgi?.field ?? null;
        const grid = field ? `${field.grid.probeCount.toLocaleString('en')} probes (${field.grid.counts.join(' x ')}), ${this.context.loaded.manifest.volume.probeSpacing} m apart` : '';
        switch (this.phase) {
            case 'preparing':
                return 'Starting: compiling the GPU programs';
            case 'building':
                return `Starting: copying the scene for ray tracing; ${grid}`;
            case 'warming':
                return `Warming up ${grid}: pass ${Math.min(this.warmUpPassesWanted, Math.floor(this.warmupPasses) + 1)} of ${this.warmUpPassesWanted}`;
            case 'calibrating':
                return `Warming up ${grid}: measuring how many probes fit in ${this.settings.budgetMs} ms`;
            case 'failed':
                return `Could not start: ${this.failure}`;
            case 'ready': {
                const updated = this.ddgi?.counters[PROBES_UPDATED_COUNTER] ?? 0;
                const warm = this.finishedWallMs === null ? '' : `; warmed up in ${(this.finishedWallMs / 1000).toFixed(1)} s`;
                const rays = this.ddgi?.settings.raysPerProbe ?? 0;
                return `${grid}; ${this.settings.preset} preset, ${rays} rays each; ${updated.toLocaleString('en')} updated per frame (budget ${this.settings.budgetMs} ms)${warm}`;
            }
        }
    }

    describeSettings(): Record<string, number | string> {
        const ddgi = this.ddgi;
        const field = ddgi?.field ?? null;
        return {
            probes: field?.grid.probeCount ?? 0,
            probe_grid: field ? field.grid.counts.join('x') : '',
            probe_spacing_m: this.context.loaded.manifest.volume.probeSpacing,
            preset: this.settings.preset,
            rays_per_probe: ddgi?.settings.raysPerProbe ?? 0,
            distance_texels: ddgi?.settings.distanceTexels ?? 0,
            budget_ms: this.settings.budgetMs,
            warmup_passes: ddgi?.settings.warmUpPasses ?? 0,
        };
    }

    /** The latest frame's numbers for the stats overlay, or null before the platform runs. The same object every call. */
    statsSnapshot(): DdgiStatsSnapshot | null {
        const gi = this.gi;
        if (!gi) return null;
        const { threeGI, platform, technique } = gi;
        const snapshot = this.snapshot;
        const stats = platform.stats;
        snapshot.updatePasses.clear();
        snapshot.platformPasses.clear();
        // Timings arrive a few frames late; right after the warm-up they still describe warm-up frames.
        snapshot.timingsPending = false;
        for (const pass of stats.passes) if (pass.frame <= this.warmupEndFrame) snapshot.timingsPending = true;
        if (!snapshot.timingsPending) {
            for (const pass of stats.passes) {
                if (pass.gpuMs === undefined || !(pass.gpuMs >= 0)) continue;
                // `ddgi/trace` is DDGI's; anything else (`platform/top-level refit`) is the platform's.
                const slash = pass.name.indexOf('/');
                const isDdgi = slash >= 0 && pass.name.startsWith('ddgi/');
                const target = isDdgi ? snapshot.updatePasses : snapshot.platformPasses;
                const name = slash < 0 ? pass.name : pass.name.slice(slash + 1);
                target.set(name, (target.get(name) ?? 0) + pass.gpuMs);
            }
        }
        const counters = technique.counters;
        snapshot.probeCount = this.probeCount;
        snapshot.probesUpdated = counters[PROBES_UPDATED_COUNTER] ?? 0;
        snapshot.sweepSeconds = counters[SWEEP_COUNTER] ?? null;
        snapshot.budget = budgetStats(stats.counters[`${technique.id} budget`], snapshot.budget);
        snapshot.budgetMs = this.settings.budgetMs;
        snapshot.memory = stats.memory;
        snapshot.recompiles = stats.recompiles;
        snapshot.syncCpuMs = threeGI.timings.syncMs;
        snapshot.encodeCpuMs = threeGI.timings.encodeMs;
        // The census needs GPU timing and arrives late: until then there are no state counts.
        const counted = STATE_COUNTERS.every((entry) => counters[entry.counter] !== undefined);
        snapshot.states.length = counted ? STATE_COUNTERS.length : 0;
        if (counted) {
            STATE_COUNTERS.forEach((entry, index) => {
                const row = (snapshot.states[index] ??= { label: entry.label, count: 0 });
                row.count = counters[entry.counter];
            });
        }
        return snapshot;
    }

    /** The most updates a probe's history keeps (`hysteresisMaxSamples`): the "settled" end of the history colours. */
    get historyMaxSamples(): number {
        return this.ddgi?.settings.hysteresisMaxSamples ?? 1;
    }

    /** How many probes are in `state` (a `PROBE_STATE` value), from the GPU census; null until it has counted. */
    probeStateCount(state: number): number | null {
        const entry = STATE_COUNTERS.find((candidate) => candidate.state === state);
        const count = entry ? this.ddgi?.counters[entry.counter] : undefined;
        return count ?? null;
    }

    /**
     * GPU ms of the platform's newest timed frame (every pass it encoded: DDGI's update and the
     * top-level refit), or null without timestamps. The platform records into its own encoder,
     * which three's timestamp queries don't see, so the benchmark adds this to three's frame.
     * It is one or two frames behind (timestamps are read back late); on a still view with a
     * fixed budget the same work runs every frame.
     */
    platformGpuMs(): number | null {
        const passes = this.platform?.stats.passes ?? [];
        if (passes.length === 0) return null;
        let total = 0;
        for (const pass of passes) {
            // Still a warm-up frame's timing, or no valid timing: nothing to report yet.
            if (pass.frame <= this.warmupEndFrame || pass.gpuMs === undefined || !(pass.gpuMs >= 0)) return null;
            total += pass.gpuMs;
        }
        return total;
    }

    dispose(): void {
        this.disposed = true;
        this.spheres?.dispose();
        this.probeHistory?.dispose();
        this.debugMaterial?.dispose();
        this.albedoView?.dispose();
        this.frameChange.dispose();
        // The GI light, ThreeGI and the platform `start` made. While `start` still runs the volume
        // doesn't hold them yet: free them here, and the start's next frame fails and stops it.
        if (this.volume.isStarted) {
            this.volume.dispose();
        } else {
            this.gi?.threeGI.dispose();
            this.gi?.platform.dispose();
        }
        this.volume.removeFromParent();
        this.disposeTarget();
    }
}

/**
 * The platform's budget counters (`ddgi budget`, budget v2) as the stats overlay reads them,
 * refilling `previous` when there is one. Null when the platform hasn't reported yet.
 */
function budgetStats(counters: Readonly<Record<string, number>> | undefined, previous: DdgiBudgetStats | null | undefined): DdgiBudgetStats | null {
    if (!counters) return null;
    const stats = previous ?? { calibrating: false, calibratedProbes: null, floorProbes: 0, steppedDown: false, stale: false, fixedMs: null, usPerProbe: null };
    stats.calibrating = counters.calibrating === 1;
    stats.stale = counters.stale === 1;
    stats.calibratedProbes = counters['calibrated units'] ?? null;
    stats.floorProbes = counters['floor units'] ?? 0;
    stats.steppedDown = counters['stepped down'] === 1;
    stats.fixedMs = counters['calibration fixed ms'] ?? null;
    const msPerProbe = counters['calibration ms per unit'];
    stats.usPerProbe = msPerProbe === undefined ? null : msPerProbe * 1000;
    return stats;
}

/** Half floats to floats, by table: the response meter reads a region every frame. */
let halfTable: Float32Array | null = null;

function halfToFloatTable(): Float32Array {
    if (halfTable) return halfTable;
    halfTable = new Float32Array(65536);
    for (let bits = 0; bits < 65536; bits++) {
        const sign = bits & 0x8000 ? -1 : 1;
        const exponent = (bits >> 10) & 0x1f;
        const mantissa = bits & 0x3ff;
        let value: number;
        if (exponent === 0) value = mantissa * 2 ** -24;
        else if (exponent === 31) value = mantissa === 0 ? Infinity : NaN;
        else value = (1 + mantissa / 1024) * 2 ** (exponent - 15);
        halfTable[bits] = sign * value;
    }
    return halfTable;
}

/** The mean of the RGB channels of RGBA half-float pixels (alpha left out). */
function halfRgbaMean(pixels: Uint16Array): number {
    const table = halfToFloatTable();
    let sum = 0;
    let count = 0;
    for (let index = 0; index + 3 < pixels.length; index += 4) {
        sum += table[pixels[index]] + table[pixels[index + 1]] + table[pixels[index + 2]];
        count += 3;
    }
    return count > 0 ? sum / count : 0;
}

/**
 * The manifest's environment as the platform takes it. The viewer draws a procedural sky
 * from the shared sky file; `platformSkyEnvironment` makes the platform build exactly
 * that file's pixels (with `sunIlluminance` from the manifest's sun), so DDGI's rays see
 * the sky the camera sees.
 */
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
