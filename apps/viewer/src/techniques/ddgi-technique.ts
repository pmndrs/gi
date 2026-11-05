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
 *     Then the platform's calibration (budget v2): frames with the
 *     millisecond budget, GPU-bound (two in flight, so the GPU never waits)
 *     and not paced by the display (at 60 Hz, or with every frame waited
 *     for, the GPU is partly idle and its clocks wander). It ends when the platform reports it calibrated and
 *     confirmed (about 30-160 burst frames, then 300 confirming frames), or
 *     after {@link CALIBRATION_MAX_STEPS}. "Start
 *     the probes over" runs it again (for after the scene changed).
 *  4. **Ready:** the calibrated number of probes every frame, and the field
 *     keeps converging live.
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
    PROBE_STATE_LABELS,
    ThreeGI,
    type DebugView,
    type EnvironmentDescriptor,
    type FrameBudget,
    type GIPlatform,
} from '../ddgi/library';
import { ProbeSpheres } from '../ddgi/probe-spheres';
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

/** The frame budget while the scene's trees are still being built: no probe updates at all. */
const PAUSED: FrameBudget = { paused: true };

/** Warm-up steps recorded per batch before the viewer waits for the GPU. Each can be a few thousand probes. */
const WARMUP_STEPS_PER_BATCH = 4;

/**
 * The calibration frames at the end of the warm-up stop after this many even
 * if the platform is still measuring: the longest burst seen on the test
 * scenes (about 160 frames) plus the 300 confirming frames, with room to
 * spare. The burst or its confirmation then finishes in normal frames.
 */
const CALIBRATION_MAX_STEPS = 600;

/** Calibration frames queued at once: the GPU never waits for the CPU between them (see the file comment). */
const CALIBRATION_FRAMES_IN_FLIGHT = 2;

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
    /** The calibration's own loop of waited frames, while it runs. */
    private calibrationRun: Promise<void> | null = null;
    /** When the first warm-up step ran (null until then): the time limit counts warm-up work, not time spent off screen. */
    private warmupStartedAt: number | null = null;
    private finishedWallMs: number | null = null;
    /** The platform frame the warm-up ended on: GPU timings from it or before are warm-up work, not normal frames. */
    private warmupEndFrame = 0;
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
        states: [],
    };

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
            // No probe work until the ray-tracing trees exist: rays would see through every wall
            // of a half-built scene to the sky. The warm-up sets the real budget.
            budget: PAUSED,
            environment: environmentFor(loaded.manifest),
            warn: (message) => this.warn(message),
        });
        // The adapter puts its light on every layer; here only DDGI's camera may see it, so the other
        // techniques in a wipe (and the probe grid's bake cameras on layer 0) get no DDGI light.
        // The adapter keeps the layers when it replaces the light.
        this.threeGI.light.layers.set(LAYER_DDGI);

        const points = this.ddgi.debugViews.find((view): view is Extract<DebugView, { kind: 'points' }> => view.kind === 'points');
        if (points) {
            this.spheres = new ProbeSpheres(points, loaded.manifest.volume.probeSpacing);
            loaded.scene.add(this.spheres.mesh);
        }
        this.debugMaterial = new DdgiDebugMaterial(this.platform.shading);
        this.applyScreenView();

        // The first update copies the scene in and starts the tree builds (still paused).
        this.threeGI.update(this.camera, 0);
        await this.waitForTrees();
        if (this.disposed) return;
        // Finished trees reach the GPU in the platform's next encode, still with no probe work.
        this.threeGI.update(this.camera, 0);
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

    private beginWarmup(): void {
        this.phase = 'warming';
        this.calibrationRun = null;
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
                return 0.1 + 0.85 * Math.min(1, this.warmupPasses / WARMUP_PASSES);
            case 'calibrating':
                return 0.95;
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

    /**
     * One warm-up step: as many probes as the technique takes in a frame, with the
     * running-average hysteresis; then the calibration steps (see the file comment).
     */
    encodePrepareStep(): boolean {
        if (!this.threeGI) return this.ready;
        if (this.phase === 'calibrating') return this.encodeCalibrationStep(this.threeGI);
        if (this.phase !== 'warming') return this.ready;
        this.warmupStartedAt ??= performance.now();
        const completedPasses = Math.floor(this.warmupPasses);
        this.ddgi.settings.hysteresis = Math.min(this.targetHysteresis, completedPasses / (completedPasses + 1));
        this.threeGI.budget = { units: Math.min(this.probeCount, this.ddgi.settings.maxProbesPerFrame) };
        this.threeGI.update(this.camera, 0);
        this.threeGI.budget = this.budget;
        this.warmupProbeUpdates += this.ddgi.counters[PROBES_UPDATED_COUNTER] ?? 0;
        const outOfTime = performance.now() - this.warmupStartedAt > WARMUP_MAX_SECONDS * 1000 && this.warmupPasses >= WARMUP_MIN_PASSES;
        if (this.warmupPasses >= WARMUP_PASSES || outOfTime) {
            this.ddgi.settings.hysteresis = this.targetHysteresis;
            this.phase = 'calibrating';
        }
        return this.ready;
    }

    /** While the platform measures the GPU: starts the calibration loop once; later calls wait for it. */
    private encodeCalibrationStep(threeGI: ThreeGI): boolean {
        this.calibrationRun ??= this.runCalibration(threeGI, this.generation).catch((error: unknown) => this.fail(error));
        return this.ready;
    }

    /**
     * Frames with the millisecond budget, back to back with two in flight, until the platform has calibrated
     * (see the file comment). Stops early if the warm-up was restarted meanwhile.
     */
    private async runCalibration(threeGI: ThreeGI, generation: number): Promise<void> {
        threeGI.budget = this.budget;
        const queued: Promise<void>[] = [];
        for (let step = 0; step < CALIBRATION_MAX_STEPS; step++) {
            threeGI.update(this.camera, 0);
            // No report (paused, no technique) or calibrated and confirmed: done. Without GPU
            // timestamps the platform never calibrates, and says so at once.
            const budget = this.platform.stats.counters[this.budgetCounterKey];
            if (budget?.calibrating !== 1 && budget?.confirming !== 1) break;
            // Two frames in flight: wait for the one before this, never for this one.
            queued.push(this.platform.device.queue.onSubmittedWorkDone());
            if (queued.length >= CALIBRATION_FRAMES_IN_FLIGHT) await queued.shift();
            // Lets the timestamp read-backs land before the next frame collects them.
            await new Promise((resolve) => setTimeout(resolve, 0));
            if (this.disposed || generation !== this.generation || this.phase !== 'calibrating') return;
        }
        this.phase = 'ready';
        this.warmupEndFrame = this.platform.stats.frame;
        this.startup.warmedMs ??= performance.now() - this.createdAt;
    }

    /** Where the platform reports DDGI's budget in `stats.counters` (budget v2). */
    private get budgetCounterKey(): string {
        return `${this.ddgi.id} budget`;
    }

    /** "Start the probes over": forget the field and warm up again. */
    restartPrepare(): void {
        this.generation++;
        this.finishedWallMs = null;
        if (this.phase !== 'warming' && this.phase !== 'calibrating' && this.phase !== 'ready') return;
        this.platform.resetHistory();
        // Measure the GPU again too: after the scene changed, the old calibration is stale.
        this.platform.recalibrate();
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
        // Before the warm-up the platform stays paused (see `start`); the warm-up takes the new budget.
        if (this.threeGI && (this.phase === 'warming' || this.phase === 'calibrating' || this.phase === 'ready')) this.threeGI.budget = this.budget;
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

    renderBeforePipeline(): void {
        const threeGI = this.threeGI;
        // During warm-up the set-up steps do the updates; GI off means no GI work at all.
        if (threeGI && this.phase === 'ready' && this.giMode !== 'off') {
            threeGI.update(this.camera);
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
            case 'calibrating':
                return `Warming up ${grid}: measuring how many probes fit in ${this.settings.budgetMs} ms`;
            case 'failed':
                return `Could not start: ${this.failure}`;
            case 'ready': {
                const updated = this.ddgi.counters[PROBES_UPDATED_COUNTER] ?? 0;
                const warm = this.finishedWallMs === null ? '' : `; warmed up in ${(this.finishedWallMs / 1000).toFixed(1)} s`;
                return `${grid}; ${updated.toLocaleString('en')} updated per frame (budget ${this.settings.budgetMs} ms)${warm}`;
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

    /** The latest frame's numbers for the stats overlay, or null before the platform runs. The same object every call. */
    statsSnapshot(): DdgiStatsSnapshot | null {
        const threeGI = this.threeGI;
        if (!threeGI) return null;
        const snapshot = this.snapshot;
        const stats = this.platform.stats;
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
        const counters = this.ddgi.counters;
        snapshot.probeCount = this.probeCount;
        snapshot.probesUpdated = counters[PROBES_UPDATED_COUNTER] ?? 0;
        snapshot.sweepSeconds = counters[SWEEP_COUNTER] ?? null;
        snapshot.budget = budgetStats(stats.counters[this.budgetCounterKey], snapshot.budget);
        snapshot.budgetMs = this.settings.budgetMs;
        snapshot.memory = stats.memory;
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

    /** How many probes are in `state` (a `PROBE_STATE` value), from the GPU census; null until it has counted. */
    probeStateCount(state: number): number | null {
        const entry = STATE_COUNTERS.find((candidate) => candidate.state === state);
        const count = entry ? this.ddgi.counters[entry.counter] : undefined;
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
        const passes = this.platform.stats.passes;
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
        this.threeGI?.dispose();
        this.spheres?.dispose();
        this.debugMaterial?.dispose();
        this.platform.dispose();
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
