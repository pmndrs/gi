/**
 * Our DDGI, through the three.js adapter (`@pmndrs/ddgi/three`).
 *
 * The scene is drawn like the probe grid's: a plain `renderer.render` into a
 * multisampled target. GI reaches three's materials through the adapter's
 * GI light, which lives on DDGI's own layer, so only DDGI's camera sees it
 * and the other techniques in a wipe are untouched. Each frame, before that
 * render, `ThreeGI.update` mirrors scene changes and submits the GI update
 * within the frame budget (2 ms by default).
 *
 * Start-up, in the order the viewer shows it:
 *  1. **Preparing:** the platform compiles its kernels (`gi.prepare()`).
 *  2. **Building:** the first update copies the scene into the platform, and
 *     the mesh trees for ray tracing are built in a worker.
 *  3. **Warming up:** up to {@link WARMUP_PASSES} passes over every probe,
 *     as many probes per step as the technique allows, with a running
 *     average instead of the usual hysteresis (below). This is what a game
 *     would do behind its loading screen: at 2 ms a frame Sponza updates
 *     about 45 probes a frame (RTX 3060 at a locked 1200 MHz), so filling
 *     its 19,425 probes from black would take minutes. The warm-up stops
 *     early after {@link WARMUP_MAX_SECONDS} (at least
 *     {@link WARMUP_MIN_PASSES} passes), so a slow GPU isn't kept waiting.
 *  4. **Ready:** the budget applies, and the field keeps converging live.
 *
 * Why a running average during warm-up: pass k keeps k/(k+1) of the old
 * value, which makes the field the plain mean of every estimate so far, the
 * fastest way to average out ray noise. After 16 passes it is the mean of 16
 * estimates; the normal hysteresis (0.97) then carries on from there.
 */

import { platformSkyEnvironment } from '../../../../scenes/src/sky/sky-image';
import { DdgiDebugMaterial } from '../ddgi/debug-material';
import {
    createGI,
    DDGI_DEBUG_VIEW,
    DDGITechnique,
    ThreeGI,
    type DebugView,
    type EnvironmentDescriptor,
    type FrameBudget,
    type GIPlatform,
    type MemoryReport,
} from '../ddgi/library';
import { ProbeSpheres } from '../ddgi/probe-spheres';
import { LAYER_DDGI, LAYER_DDGI_PROBES } from '../scene/layers';
import { ScenePassTechnique } from './scene-pass-technique';
import type { TechniqueContext } from './types';

/** DDGI's screen debug views by the name the viewer (and its links) use. */
export const DDGI_SCREEN_VIEWS = {
    'gi-only': DDGI_DEBUG_VIEW.giOnly,
    coverage: DDGI_DEBUG_VIEW.coverage,
    'probe-states': DDGI_DEBUG_VIEW.probeStates,
} as const;

export type DdgiScreenView = keyof typeof DDGI_SCREEN_VIEWS;

export interface DdgiViewerSettings {
    /** GPU milliseconds per frame for probe updates. */
    budgetMs: number;
    /** Draw every probe as a small sphere coloured by its state. */
    showProbes: boolean;
    /** A screen debug view instead of the normal picture, or null. */
    screenView: DdgiScreenView | null;
}

export const DEFAULT_DDGI_VIEWER_SETTINGS: DdgiViewerSettings = { budgetMs: 2, showProbes: false, screenView: null };

/** Full passes over every probe before the frame budget applies (see the file comment). */
export const WARMUP_PASSES = 16;

/** The warm-up ends after this long even if it hasn't made all its passes (the Intel UHD on Sponza)... */
const WARMUP_MAX_SECONDS = 8;
/** ...but not before every probe has been updated this many times, so no probe is left black. */
const WARMUP_MIN_PASSES = 2;

/** Warm-up steps recorded per batch before the viewer waits for the GPU. Each can be a few thousand probes. */
const WARMUP_STEPS_PER_BATCH = 4;

/** The adapter's warning about area lights. The viewer's emissive quads carry emissive materials, which GI does see, so it is expected. */
const EXPECTED_WARNING = 'RectAreaLight is not mirrored into GI';

type Phase = 'preparing' | 'building' | 'warming' | 'ready' | 'failed';

/** What the stats overlay shows for DDGI, as of the latest frame. */
export interface DdgiStatsSnapshot {
    probeCount: number;
    probesUpdated: number;
    /** GPU ms of DDGI's update passes (`ddgi/*`), by pass name without the prefix; empty without timestamps. */
    updatePasses: Map<string, number>;
    /** GPU ms of the platform's own per-frame work (`platform/*`, the top-level tree refit), by pass name. */
    platformPasses: Map<string, number>;
    /** True while the newest GPU timings still belong to the warm-up (the pass maps are then empty). */
    timingsPending: boolean;
    memory: MemoryReport;
    /** The adapter's CPU time in the last update: the scene sync, then the platform's encode and submit. */
    syncCpuMs: number;
    encodeCpuMs: number;
    /** Probes per state from the GPU census (needs timestamps); empty until the first arrives. */
    states: Record<string, number>;
}

/** Start-up times, milliseconds from when the technique was created. Null until reached. */
export interface DdgiStartupTimes {
    preparedMs: number | null;
    /** The scene copied and its ray-tracing trees built: the first frame with real GI. */
    firstGiMs: number | null;
    warmedMs: number | null;
}

/** The platform's ray-tracing trees, as far as the viewer needs them (see `waitForTrees`). */
interface TreesBuiltSignal {
    rays?: { whenBuilt?: () => Promise<void> };
}

export class DdgiTechnique extends ScenePassTechnique {
    readonly id = 'ddgi';
    readonly label = 'DDGI (ours)';
    readonly prepareStepsPerBatch = WARMUP_STEPS_PER_BATCH;
    readonly startup: DdgiStartupTimes = { preparedMs: null, firstGiMs: null, warmedMs: null };

    private readonly platform: GIPlatform;
    private readonly ddgi: DDGITechnique;
    private readonly targetHysteresis: number;
    private readonly createdAt = performance.now();
    private readonly started: Promise<void>;
    private threeGI: ThreeGI | null = null;
    private spheres: ProbeSpheres | null = null;
    private debugMaterial: DdgiDebugMaterial | null = null;
    private settings: DdgiViewerSettings;
    private budget: FrameBudget;
    private phase: Phase = 'preparing';
    private failure: string | null = null;
    private generation = 0;
    private warmupProbeUpdates = 0;
    /** When the first warm-up step ran (null until then): the time limit counts warm-up work, not time spent off screen. */
    private warmupStartedAt: number | null = null;
    private finishedWallMs: number | null = null;
    /** The platform frame the warm-up ended on: GPU timings from it or before are warm-up work, not normal frames. */
    private warmupEndFrame = 0;
    private disposed = false;
    private readonly warnings = new Set<string>();

    constructor(context: TechniqueContext, settings: DdgiViewerSettings = DEFAULT_DDGI_VIEWER_SETTINGS) {
        super(context);
        this.settings = { ...settings };
        this.budget = { gpuMs: settings.budgetMs };
        // The base class chose layers before the settings existed; choose again with them.
        this.applyLayers();
        const { manifest } = context.loaded;
        // GPU timing feeds the millisecond budget (the platform turns ms into probes from measured costs) and the stats.
        this.platform = createGI(context.device, { gpuTiming: context.hasTimestamps, label: `viewer ${manifest.id}` });
        this.ddgi = this.platform.addTechnique(
            new DDGITechnique({ volume: { center: manifest.volume.center, size: manifest.volume.size, spacing: manifest.volume.probeSpacing } }),
        );
        this.platform.setOutput(this.ddgi);
        this.targetHysteresis = this.ddgi.settings.hysteresis;
        this.started = this.start().catch((error: unknown) => this.fail(error));
    }

    // -- set-up ------------------------------------------------------------

    private async start(): Promise<void> {
        await this.platform.prepare();
        if (this.disposed) return;
        this.startup.preparedMs = performance.now() - this.createdAt;
        this.phase = 'building';

        const { loaded } = this.context;
        this.threeGI = new ThreeGI(this.platform, loaded.scene, {
            budget: this.budget,
            environment: environmentFor(loaded.manifest),
            warn: (message) => this.warn(message),
        });
        this.placeLight();

        const points = this.ddgi.debugViews.find((view): view is Extract<DebugView, { kind: 'points' }> => view.kind === 'points');
        if (points) {
            this.spheres = new ProbeSpheres(points, loaded.manifest.volume.probeSpacing);
            loaded.scene.add(this.spheres.mesh);
        }
        this.debugMaterial = new DdgiDebugMaterial(this.platform.shading);
        this.applyScreenView();

        // The first update copies the scene in and starts the tree builds; no probe work until they exist,
        // or the first passes would see through every wall to the sky.
        this.threeGI.update(this.camera, 0);
        this.pauseOneFrame();
        await this.waitForTrees();
        if (this.disposed) return;
        // Finished trees reach the GPU in the platform's next encode.
        this.pauseOneFrame();
        this.startup.firstGiMs = performance.now() - this.createdAt;
        this.beginWarmup();
    }

    /**
     * Waits until the platform has built every mesh tree and the top level. The contract has no
     * public signal for that yet (reported for the platform), so this reads the platform object's
     * `rays.whenBuilt()` where it exists, and otherwise goes on at once.
     */
    private async waitForTrees(): Promise<void> {
        const whenBuilt = (this.platform as unknown as TreesBuiltSignal).rays?.whenBuilt;
        if (whenBuilt) await whenBuilt.call((this.platform as unknown as TreesBuiltSignal).rays);
    }

    /** One update with no probe work: scene changes and finished trees reach the GPU. */
    private pauseOneFrame(): void {
        const threeGI = this.threeGI!;
        threeGI.budget = { paused: true };
        threeGI.update(this.camera, 0);
        threeGI.budget = this.budget;
        this.placeLight();
    }

    private beginWarmup(): void {
        this.phase = 'warming';
        this.warmupProbeUpdates = 0;
        this.warmupStartedAt = null;
        this.warmupEndFrame = Number.POSITIVE_INFINITY;
    }

    private fail(error: unknown): void {
        if (this.disposed) return;
        this.phase = 'failed';
        this.failure = error instanceof Error ? error.message : String(error);
        console.error(`DDGI could not start: ${this.failure}`);
    }

    private warn(message: string): void {
        if (message.includes(EXPECTED_WARNING) || this.warnings.has(message)) return;
        this.warnings.add(message);
        console.warn(`DDGI: ${message}`);
    }

    get ready(): boolean {
        return this.phase === 'ready' || this.phase === 'failed';
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
                return 0.1 + 0.9 * Math.min(1, this.warmupPasses / WARMUP_PASSES);
            default:
                return 1;
        }
    }

    private get probeCount(): number {
        return this.ddgi.field?.grid.probeCount ?? 0;
    }

    /** Full passes over every probe the warm-up made (fractional while a pass is under way). */
    get warmupPasses(): number {
        return this.probeCount > 0 ? this.warmupProbeUpdates / this.probeCount : 0;
    }

    /** One warm-up step: as many probes as the technique takes in a frame, with the running-average hysteresis. */
    encodePrepareStep(): boolean {
        if (this.phase !== 'warming' || !this.threeGI) return this.ready;
        this.warmupStartedAt ??= performance.now();
        const completedPasses = Math.floor(this.warmupPasses);
        this.ddgi.settings.hysteresis = Math.min(this.targetHysteresis, completedPasses / (completedPasses + 1));
        this.threeGI.budget = { units: Math.min(this.probeCount, this.ddgi.settings.maxProbesPerFrame) };
        this.threeGI.update(this.camera, 0);
        this.threeGI.budget = this.budget;
        this.placeLight();
        this.warmupProbeUpdates += this.ddgi.counters['probes updated'] ?? 0;
        const outOfTime = performance.now() - this.warmupStartedAt > WARMUP_MAX_SECONDS * 1000 && this.warmupPasses >= WARMUP_MIN_PASSES;
        if (this.warmupPasses >= WARMUP_PASSES || outOfTime) {
            this.ddgi.settings.hysteresis = this.targetHysteresis;
            this.phase = 'ready';
            this.warmupEndFrame = this.platform.stats.frame;
            this.startup.warmedMs ??= performance.now() - this.createdAt;
        }
        return this.ready;
    }

    /** "Start the probes over": forget the field and warm up again. */
    restartPrepare(): void {
        this.generation++;
        this.finishedWallMs = null;
        if (this.phase !== 'warming' && this.phase !== 'ready') return;
        this.platform.resetHistory();
        this.beginWarmup();
    }

    notePrepareFinished(generation: number, wallMs: number): void {
        if (generation === this.generation && this.phase === 'ready') this.finishedWallMs = wallMs;
    }

    /** Waits until the platform is prepared and the trees are built (for the benchmark). */
    async warmUp(): Promise<void> {
        await this.started;
    }

    // -- settings ----------------------------------------------------------

    configure(settings: DdgiViewerSettings): void {
        const probesChanged = settings.showProbes !== this.settings.showProbes;
        this.settings = { ...settings };
        // A new object: ThreeGI reads its `budget` field every update.
        this.budget = { gpuMs: settings.budgetMs };
        if (this.threeGI) this.threeGI.budget = this.budget;
        this.applyScreenView();
        if (probesChanged) this.applyLayers();
    }

    get currentSettings(): Readonly<DdgiViewerSettings> {
        return this.settings;
    }

    private applyScreenView(): void {
        if (this.debugMaterial && this.settings.screenView) this.debugMaterial.view = DDGI_SCREEN_VIEWS[this.settings.screenView];
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

    /**
     * Keeps the GI light on DDGI's layer, and its coverage off. The adapter replaces the light
     * when the shading module changes shape, and a new light starts on layer 0, which the probe
     * grid's bake cameras would see. Coverage only matters where the light shares three's
     * probe slot or replaces `scene.environment`'s diffuse light; in the viewer it never
     * meets the probe grid (other layers) and there is no `scene.environment`, so the
     * adapter's guess (it counts every LightProbeGrid in the scene) would only cost time.
     */
    private placeLight(): void {
        const light = this.threeGI?.light;
        if (!light) return;
        if (light.layers.mask !== 1 << LAYER_DDGI) light.layers.set(LAYER_DDGI);
        light.computeCoverage = false;
    }

    renderBeforePipeline(): void {
        const threeGI = this.threeGI;
        // During warm-up the set-up steps do the updates; GI off means no GI work at all.
        if (threeGI && this.phase === 'ready' && this.giMode !== 'off') {
            threeGI.update(this.camera);
            this.placeLight();
        }
        this.spheres?.refresh();
        const scene = this.context.loaded.scene;
        const override = this.settings.screenView && this.debugMaterial && this.giMode !== 'off' && this.phase !== 'failed' ? this.debugMaterial : null;
        if (!override) {
            super.renderBeforePipeline();
            return;
        }
        override.refresh();
        const previous = scene.overrideMaterial;
        scene.overrideMaterial = override.material;
        try {
            super.renderBeforePipeline();
        } finally {
            scene.overrideMaterial = previous;
        }
    }

    // -- reporting ---------------------------------------------------------

    describe(): string {
        const field = this.ddgi.field;
        const grid = field ? `${field.grid.probeCount.toLocaleString('en')} probes (${field.grid.counts.join(' x ')}), ${this.context.loaded.manifest.volume.probeSpacing} m apart` : '';
        switch (this.phase) {
            case 'preparing':
                return 'Starting: compiling the GPU programs';
            case 'building':
                return `Starting: copying the scene for ray tracing; ${grid}`;
            case 'warming':
                return `Warming up ${grid}: pass ${Math.min(WARMUP_PASSES, Math.floor(this.warmupPasses) + 1)} of ${WARMUP_PASSES}`;
            case 'failed':
                return `Could not start: ${this.failure}`;
            case 'ready': {
                const updated = this.ddgi.counters['probes updated'] ?? 0;
                const warm = this.finishedWallMs === null ? '' : `; warmed up in ${(this.finishedWallMs / 1000).toFixed(1)} s`;
                return `${grid}; ${updated.toLocaleString('en')} updated last frame within ${this.settings.budgetMs} ms${warm}`;
            }
        }
    }

    describeSettings(): Record<string, number | string> {
        const field = this.ddgi.field;
        return {
            probes: field?.grid.probeCount ?? 0,
            probe_grid: field ? field.grid.counts.join('x') : '',
            probe_spacing_m: this.context.loaded.manifest.volume.probeSpacing,
            rays_per_probe: this.ddgi.settings.raysPerProbe,
            budget_ms: this.settings.budgetMs,
            warmup_passes: WARMUP_PASSES,
        };
    }

    /** The latest frame's numbers for the stats overlay, or null before the platform runs. */
    statsSnapshot(): DdgiStatsSnapshot | null {
        if (!this.threeGI) return null;
        const stats = this.platform.stats;
        const updatePasses = new Map<string, number>();
        const platformPasses = new Map<string, number>();
        // Timings arrive a few frames late; right after the warm-up they still describe warm-up frames.
        const timingsPending = stats.passes.some((pass) => pass.frame <= this.warmupEndFrame);
        for (const pass of timingsPending ? [] : stats.passes) {
            if (pass.gpuMs === undefined || !(pass.gpuMs >= 0)) continue;
            const [owner, name] = splitPassName(pass.name);
            const target = owner === 'ddgi' ? updatePasses : platformPasses;
            target.set(name, (target.get(name) ?? 0) + pass.gpuMs);
        }
        const counters = this.ddgi.counters;
        const states: Record<string, number> = {};
        for (const [name, value] of Object.entries(counters)) if (name.startsWith('probes ') && name !== 'probes updated') states[name] = value;
        return {
            probeCount: this.probeCount,
            probesUpdated: counters['probes updated'] ?? 0,
            updatePasses,
            platformPasses,
            timingsPending,
            memory: stats.memory,
            syncCpuMs: this.threeGI.timings.syncMs,
            encodeCpuMs: this.threeGI.timings.encodeMs,
            states,
        };
    }

    /**
     * GPU ms of the platform's newest timed frame (every pass it encoded: DDGI's update and the
     * top-level refit), or null without timestamps. The platform records into its own encoder,
     * which three's timestamp queries don't see, so the benchmark adds this to three's frame.
     * It is one or two frames behind (timestamps are read back late); on a still view with a
     * fixed budget the same work runs every frame.
     */
    platformGpuMs(): number | null {
        const passes = this.platform.stats.passes;
        if (passes.length === 0 || passes.some((pass) => pass.frame <= this.warmupEndFrame)) return null;
        let total = 0;
        for (const pass of passes) {
            if (pass.gpuMs === undefined || !(pass.gpuMs >= 0)) return null;
            total += pass.gpuMs;
        }
        return total;
    }

    dispose(): void {
        this.disposed = true;
        this.threeGI?.dispose();
        this.spheres?.dispose();
        this.debugMaterial?.dispose();
        this.platform.dispose();
        this.disposeTarget();
    }
}

/** `ddgi/trace` becomes `['ddgi', 'trace']`; a name with no slash belongs to the platform. */
function splitPassName(name: string): [string, string] {
    const slash = name.indexOf('/');
    return slash < 0 ? ['platform', name] : [name.slice(0, slash), name.slice(slash + 1)];
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
