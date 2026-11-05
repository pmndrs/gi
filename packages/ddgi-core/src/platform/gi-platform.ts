/**
 * The platform object a host talks to: `createGI(device, options)`.
 *
 * It owns the scene copy (`PlatformScene`), the `rays` query backend over it
 * (`PlatformRayQueries`) and a `KernelBuilder`, and runs the attached
 * techniques. Each frame, `encode` records, in this order and into the host's
 * encoder:
 *
 * 1. the scene copy's uploads (`PlatformScene.encode`);
 * 2. the tracer acting on them: removed meshes, new mesh trees from the
 *    worker, and the top-level rebuild (CPU for now; B1a moves it to the GPU);
 * 2b. hit shading: its settings, and the sky's projection when the
 *    environment changed (`shade/hit-shading.ts`, T1.4);
 * 3. each technique's `encode`, fallbacks before the techniques stacked on them.
 *
 * The host owns the device. The platform watches `device.lost`; afterwards
 * `isReady` is false and calls throw `GINotReadyError('device-lost')`, and the
 * host makes a new platform on its new device.
 */

import { GINotReadyError } from './contract/errors.ts';
import type { InstanceHandle, LightHandle, MaterialHandle, MeshHandle, TextureHandle } from './contract/handles.ts';
import type { Aabb, Mat4 } from './contract/math.ts';
import type { FrameBudget, FrameInput, FrameScreen } from './contract/frame.ts';
import type { GIPlatform, GIPlatformOptions } from './contract/platform.ts';
import type { QueryKind, SceneQueries } from './contract/queries.ts';
import type {
    DeformedPositions,
    EnvironmentDescriptor,
    InstanceDescriptor,
    LightChanges,
    LightDescriptor,
    MaterialDescriptor,
    MeshDescriptor,
    SceneApi,
    TextureDescriptor,
} from './contract/scene.ts';
import type { SceneChanges } from './contract/scene-copy.ts';
import type { BounceSource } from './contract/shade.ts';
import type { ShadingModule } from './contract/shading-module.ts';
import type { GIStats, MemoryReport, PassTiming } from './contract/stats.ts';
import type { GITechnique, TechniqueContext, TechniqueFrame } from './contract/technique.ts';
import type { GpuTimerScope } from './contract/timing.ts';
import { PlatformKernelBuilder } from './kernel-builder.ts';
import { PlatformScene } from './scene/platform-scene.ts';
import { checkComputeLimits } from './device-requirements.ts';
import { PlatformHitShading } from './shade/hit-shading.ts';
import { GpuTimer } from './timing/gpu-timer.ts';
import { PlatformRayQueries } from './trace/ray-queries.ts';


/**
 * Work units a technique gets when there's no way to use `budget.gpuMs` (no
 * GPU timings yet, or none at all without `timestamp-query`) and the budget
 * names no units: DDGI's default probe count per frame on Tier A
 * (docs/design/ddgi.md).
 */
export const DEFAULT_UNITS_PER_FRAME = 512;

/**
 * A technique's GPU cost for a frame of `units` work units, as the budget
 * sees it: `fixedMs + msPerUnit × units`.
 *
 * Why a fixed part (T1.9a): a small batch doesn't fill the GPU, so its time
 * is mostly the latency of the slowest workgroup, not the sum of the work.
 * DDGI on Sponza (RTX 3060) costs about the same 1.6 ms for 32 probes as for
 * 128, then about 6 µs per probe more. A cost proportional to the units reads
 * small batches as expensive per unit and hands out too few, and after a
 * warm-up of big batches it hands out far too many.
 */
export interface UnitCostModel {
    readonly fixedMs: number;
    readonly msPerUnit: number;
}

/**
 * The fixed part is the median over this many recent timed frames (half a
 * second at 60 Hz), so up to 15 slow frames in 31 change nothing.
 *
 * Why a median (T1.9a): GPU timestamps time the GPU's wall clock, so another
 * program's GPU work that the driver slices into ours lands inside our
 * passes. On this laptop that added about 2.5 ms (RTX 3060) or 1.3 ms (Intel
 * UHD) to the trace pass every 50 ms, a third of the frames at 60 Hz, and
 * smaller overlaps to many more. A running mean held the *mean* at the
 * budget, so the typical frame ran well under it; on Sponza it shrank the
 * batch to 14 probes. A low quantile instead of the median overspent the
 * budget by 5-30% (measured), and 15 frames were too few to ride out bursts
 * of outside work.
 */
export const RECENT_FRAMES = 31;

/**
 * When the budget is in milliseconds two frames running, a technique's units
 * shrink by at most this share per frame. A burst of outside GPU work can
 * still push the measured median over the budget for a few frames; without
 * the limit the batch then fell to 1 probe (Sponza, Bistro), which makes the
 * time no shorter (most of it is fixed) but stops GI from converging.
 */
export const MAX_UNITS_DROP_PER_FRAME = 0.5;

/** Each batch-size bucket keeps this many of its latest frames; the slope is fitted through their medians. */
const BUCKET_FRAMES = 9;

/**
 * The fitted cost per unit stays at least this share of the lowest average
 * cost per unit measured (the biggest batches'). A nearly flat fit (all
 * latency, no throughput) would make every small timing error a huge jump in
 * units.
 */
const MIN_SLOPE_SHARE = 0.25;

/**
 * Batch sizes must span at least this factor before a slope is fitted. Nearby
 * sizes (a budget's batch wandering by a few units) give a slope that is all
 * noise.
 */
const MIN_FIT_SPAN = 4;

/** Frames of handed-out units remembered while their timings are on the way (they arrive a few frames late). */
const REMEMBERED_FRAMES = 32;

/**
 * Units each technique may spend this frame, in encode order (FrameBudget):
 * - with `gpuMs` and a cost model for every technique, the time is shared
 *   evenly between the techniques (docs/api/platform.md, gap 6: a real policy
 *   waits for a second technique), and each share buys the units its model
 *   says fit: `(share − fixedMs) / msPerUnit`, at least 1 while the share is
 *   above 0 (a share below the fixed cost can't be met by any batch; one unit
 *   keeps the technique working and measured, instead of stopping it for
 *   good). Capped by `units` when that is set too;
 * - otherwise `units`, or {@link DEFAULT_UNITS_PER_FRAME}.
 */
export function unitsForFrame(budget: FrameBudget, costs: readonly (UnitCostModel | undefined)[]): number[] {
    const fallback = budget.units ?? DEFAULT_UNITS_PER_FRAME;
    const gpuMs = budget.gpuMs;
    const costsKnown = costs.every(
        (cost) => cost !== undefined && cost.msPerUnit > 0 && Number.isFinite(cost.msPerUnit) && cost.fixedMs >= 0 && Number.isFinite(cost.fixedMs),
    );
    if (gpuMs === undefined || !(gpuMs >= 0) || !costsKnown) return costs.map(() => fallback);
    const share = gpuMs / Math.max(1, costs.length);
    return costs.map((cost) => {
        const model = cost as UnitCostModel;
        const units = share > 0 ? Math.max(1, Math.floor((share - model.fixedMs) / model.msPerUnit)) : 0;
        return budget.units === undefined ? units : Math.min(units, budget.units);
    });
}

/**
 * This frame's units for one technique after the drop limit
 * ({@link MAX_UNITS_DROP_PER_FRAME}): at least that share of the previous
 * frame's, when the previous frame was budgeted in milliseconds too
 * (`previous` defined). A zero hand-out (a zero budget) stays zero.
 */
export function limitUnitsDrop(units: number, previous: number | undefined): number {
    if (previous === undefined || units === 0) return units;
    return Math.max(units, Math.floor(previous * MAX_UNITS_DROP_PER_FRAME));
}

/** One timed frame: the units a technique used and the GPU ms its passes took. */
interface CostSample {
    readonly units: number;
    readonly ms: number;
}

interface TechniqueCost {
    /** The latest {@link RECENT_FRAMES} frames, oldest first. */
    readonly recent: CostSample[];
    /** The latest frames per batch size (bucket = floor(log2(units))): the spread the slope is fitted from. */
    readonly buckets: Map<number, CostSample[]>;
}

/**
 * Measures each technique's GPU cost from the platform's timer, as a
 * {@link UnitCostModel}: the passes named `<technique id>/...` in a frame
 * against the units the technique used that frame (what its `encode`
 * returned, or all it was handed).
 *
 * - **The slope** (cost per unit) is a least-squares line through the
 *   medians of each batch size's latest frames (sizes grouped by powers of
 *   two). The warm-up's big batches and the budget's small ones give it
 *   spread. It is kept at or above {@link MIN_SLOPE_SHARE} of the cheapest
 *   average cost per unit measured, and at or below the recent frames'
 *   average, so the fixed part is never negative.
 * - **The fixed part** is the median, over the latest {@link RECENT_FRAMES}
 *   frames, of each frame's ms minus slope × its units, at least 0. A median,
 *   so frames slowed by other programs' GPU work count little (see
 *   {@link RECENT_FRAMES}); and per frame, so a batch that changes from frame
 *   to frame doesn't blur it.
 * - Without spread (one batch size so far) there is no slope, and the model
 *   is proportional: the median of the recent frames' ms per unit, no fixed
 *   part, as before T1.9a.
 *
 * With this model the budget converges on the batch whose measured time is
 * the budget, by steps along the fitted slope. A proportional model has the
 * same end point in theory, but crawls there when most of the time is fixed,
 * and its running mean followed the outside spikes.
 *
 * Reset when the work changes shape: on `prepare`, when a technique is added
 * or removed, and on `resetHistory`.
 */
export class UnitCostMeter {
    private readonly handedOut = new Map<number, Map<string, number>>();
    private readonly costs = new Map<string, TechniqueCost>();

    /** Records the units a technique used in a frame. */
    handOut(frame: number, techniqueId: string, units: number): void {
        let frameUnits = this.handedOut.get(frame);
        if (!frameUnits) {
            frameUnits = new Map();
            this.handedOut.set(frame, frameUnits);
            for (const old of this.handedOut.keys()) if (old <= frame - REMEMBERED_FRAMES) this.handedOut.delete(old);
        }
        frameUnits.set(techniqueId, units);
    }

    /** Takes one frame's pass timings into the estimates. */
    measure(frame: number, passes: readonly { readonly name: string; readonly gpuMs: number }[]): void {
        const frameUnits = this.handedOut.get(frame);
        if (!frameUnits) return;
        for (const [techniqueId, units] of frameUnits) {
            if (!(units > 0)) continue;
            const prefix = `${techniqueId}/`;
            const timed = passes.filter((pass) => pass.name.startsWith(prefix) && pass.gpuMs >= 0);
            if (timed.length === 0) continue;
            const ms = timed.reduce((sum, pass) => sum + pass.gpuMs, 0);
            this.add(techniqueId, { units, ms });
        }
        this.handedOut.delete(frame);
    }

    /** The technique's cost model, or undefined before its first timed frame. */
    model(techniqueId: string): UnitCostModel | undefined {
        const cost = this.costs.get(techniqueId);
        if (!cost) return undefined;
        const points = [...cost.buckets.values()].map((samples) => ({
            units: median(samples.map((sample) => sample.units)),
            ms: median(samples.map((sample) => sample.ms)),
        }));
        const fitted = fittedSlope(points);
        const recentAverage = median(cost.recent.map((sample) => sample.ms / sample.units));
        if (fitted === undefined) return { fixedMs: 0, msPerUnit: recentAverage };
        const cheapest = Math.min(...points.map((point) => point.ms / point.units));
        // At most the recent average: where big batches cost more per unit than small ones
        // (the Intel UHD after a warm-up of 1,728-probe batches), a steeper line would need
        // a negative fixed part, and clamping that to 0 made the model overestimate the
        // batch in use by 40% and hold the budget at 1.4 ms (T1.9a).
        const slope = Math.min(recentAverage, Math.max(MIN_SLOPE_SHARE * cheapest, fitted));
        const fixedMs = median(cost.recent.map((sample) => sample.ms - slope * sample.units));
        return { fixedMs: Math.max(0, fixedMs), msPerUnit: slope };
    }

    reset(): void {
        this.handedOut.clear();
        this.costs.clear();
    }

    private add(techniqueId: string, sample: CostSample): void {
        let cost = this.costs.get(techniqueId);
        if (!cost) {
            cost = { recent: [], buckets: new Map() };
            this.costs.set(techniqueId, cost);
        }
        pushLimited(cost.recent, sample, RECENT_FRAMES);
        const bucket = Math.floor(Math.log2(sample.units));
        let samples = cost.buckets.get(bucket);
        if (!samples) {
            samples = [];
            cost.buckets.set(bucket, samples);
        }
        pushLimited(samples, sample, BUCKET_FRAMES);
    }
}

function pushLimited<T>(list: T[], item: T, limit: number): void {
    list.push(item);
    if (list.length > limit) list.shift();
}

function median(values: readonly number[]): number {
    const sorted = [...values].sort((a, b) => a - b);
    const middle = Math.floor(sorted.length / 2);
    return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
}

/** Least-squares slope of ms against units through the points; undefined without enough spread in units. */
function fittedSlope(points: readonly CostSample[]): number | undefined {
    if (points.length < 2) return undefined;
    const smallest = Math.min(...points.map((point) => point.units));
    const largest = Math.max(...points.map((point) => point.units));
    if (largest < MIN_FIT_SPAN * smallest) return undefined;
    const meanUnits = points.reduce((sum, point) => sum + point.units, 0) / points.length;
    const meanMs = points.reduce((sum, point) => sum + point.ms, 0) / points.length;
    let covariance = 0;
    let variance = 0;
    for (const point of points) {
        covariance += (point.units - meanUnits) * (point.ms - meanMs);
        variance += (point.units - meanUnits) ** 2;
    }
    return variance > 0 ? covariance / variance : undefined;
}

/** No scene changes: the screen stage runs after `encode`, which already reported them. */
const NO_CHANGES: SceneChanges = { regions: [], global: [], movedReceivers: 0 };

interface AttachedTechnique {
    readonly technique: GITechnique;
    fallback: GITechnique | undefined;
}

export function createGI(device: GPUDevice, options: GIPlatformOptions = {}): GIPlatform {
    return new GIPlatformImpl(device, options);
}

export class GIPlatformImpl implements GIPlatform {
    readonly device: GPUDevice;
    readonly scene: SceneApi;
    /** The scene copy itself, for platform code and tests (hosts see `scene`, techniques see `SceneCopy`). */
    readonly sceneCopy: PlatformScene;
    readonly rays: PlatformRayQueries;
    readonly kernels: PlatformKernelBuilder;

    private readonly label: string;
    private readonly timer: GpuTimer | null;
    private readonly queries: SceneQueries;
    /** Hit shading and the sky (T1.4): `giSurfaceAt`, `giDirectIrradiance`, `giSkyRadiance`, `giSkyIrradiance`. */
    readonly shade: PlatformHitShading;
    private readonly attached: AttachedTechnique[] = [];
    private output: GITechnique | undefined;
    private prepared = false;
    private lost = false;
    private disposed = false;
    private frame = 0;
    private lastEncodeCpuMs = 0;
    /** Set after `encode` resolved timestamps, so the next `encode` starts their readback (the host has submitted by then). */
    private timerAwaitingReadback = false;
    private latestPasses: PassTiming[] = [];
    /** Each technique's measured GPU cost (fixed + per unit), for turning `budget.gpuMs` into units. */
    private readonly unitCosts = new UnitCostMeter();
    /** Units handed out last frame, for techniques whose last frame had a millisecond budget (the drop limit). */
    private readonly lastMsBudgetUnits = new Map<GITechnique, number>();
    private readonly warnedMessages = new Set<string>();
    /** The `uncapturederror` listener, kept so `dispose` can remove it. */
    private readonly onUncapturedError: (event: Event) => void;

    constructor(device: GPUDevice, options: GIPlatformOptions = {}) {
        this.device = device;
        this.label = options.label ?? 'gi';
        this.sceneCopy = new PlatformScene(device, { label: this.label });
        this.scene = new GuardedScene(this.sceneCopy, () => this.checkUsable());
        this.rays = new PlatformRayQueries(device, this.sceneCopy, {
            label: this.label,
            buildIn: options.buildIn,
            // The top level reads its refit stats back only in stats mode: hosts are promised no readbacks.
            diagnostics: options.gpuTiming === true,
            warn: (message) => this.warnOnce(message),
        });
        this.kernels = new PlatformKernelBuilder(device, this.label);
        this.timer = options.gpuTiming ? new GpuTimer(device, { warn: (message) => this.warnOnce(message) }) : null;
        const rays = this.rays;
        this.queries = {
            rays,
            has: (kind: QueryKind) => kind === 'rays',
        };
        this.shade = new PlatformHitShading(device, this.sceneCopy.module, this.rays.module, {
            label: this.label,
            warn: (message) => this.warnOnce(message),
        });

        void device.lost.then(() => {
            this.lost = true;
            this.prepared = false;
        });

        // The platform's own GPU errors as warnings. A listener, never an assignment to
        // `onuncapturederror`: the host (three.js) owns that slot (SPEC, "GPU device").
        // The event also carries the host's errors, so only messages naming one of the
        // platform's GPU objects are forwarded: their labels start with this platform's
        // label, or with an attached technique's id and a slash ("ddgi/trace").
        const ownObject = new RegExp(`"${escapeRegExp(this.label)}[ /]`);
        this.onUncapturedError = (event: Event) => {
            const message = (event as GPUUncapturedErrorEvent).error?.message ?? '';
            const ours = ownObject.test(message) || this.attached.some(({ technique }) => message.includes(`"${technique.id}/`));
            if (ours) this.warnOnce(`@pmndrs/ddgi: GPU error in the platform's work: ${message}`);
        };
        device.addEventListener('uncapturederror', this.onUncapturedError);
    }

    get isReady(): boolean {
        return this.prepared && !this.lost && !this.disposed;
    }

    // -----------------------------------------------------------------------
    // Techniques
    // -----------------------------------------------------------------------

    addTechnique<T extends GITechnique>(technique: T): T {
        this.checkUsable();
        this.unitCosts.reset();
        this.lastMsBudgetUnits.clear();
        if (this.find(technique)) throw new Error(`@pmndrs/ddgi: technique '${technique.id}' is already attached`);
        for (const kind of technique.requirements.queries) {
            if (!this.queries.has(kind)) throw new Error(`@pmndrs/ddgi: technique '${technique.id}' needs the '${kind}' query backend, which this platform doesn't have`);
        }
        const entry: AttachedTechnique = { technique, fallback: undefined };
        this.attached.push(entry);
        technique.attach(this.contextFor(entry));
        // A new technique needs its resources and kernels: encode waits for the next prepare().
        this.prepared = false;
        return technique;
    }

    /** Detaches and disposes the technique (a technique is attached to one platform for its whole life). */
    removeTechnique(technique: GITechnique): void {
        this.unitCosts.reset();
        this.lastMsBudgetUnits.clear();
        const entry = this.find(technique);
        if (!entry) return;
        if (this.attached.some((other) => other.fallback === technique)) {
            throw new Error(`@pmndrs/ddgi: technique '${technique.id}' is another technique's fallback; call setFallback(…, undefined) first`);
        }
        this.attached.splice(this.attached.indexOf(entry), 1);
        if (this.output === technique) this.output = undefined;
        technique.dispose();
    }

    setOutput(technique: GITechnique): void {
        if (!this.find(technique)) throw new Error(`@pmndrs/ddgi: technique '${technique.id}' is not attached; call addTechnique first`);
        this.output = technique;
    }

    setFallback(technique: GITechnique, fallback: GITechnique | undefined): void {
        const entry = this.find(technique);
        if (!entry) throw new Error(`@pmndrs/ddgi: technique '${technique.id}' is not attached`);
        if (fallback !== undefined) {
            if (!this.find(fallback)) throw new Error(`@pmndrs/ddgi: fallback '${fallback.id}' is not attached`);
            // Walk the chain below the new fallback: reaching `technique` again would be a loop.
            for (let below: GITechnique | undefined = fallback; below; below = this.find(below)?.fallback) {
                if (below === technique) throw new Error(`@pmndrs/ddgi: making '${fallback.id}' the fallback of '${technique.id}' would make a loop`);
            }
        }
        entry.fallback = fallback;
    }

    // -----------------------------------------------------------------------
    // Lifecycle
    // -----------------------------------------------------------------------

    async prepare(): Promise<void> {
        // New resources and kernels: old costs per unit no longer apply.
        this.unitCosts.reset();
        this.lastMsBudgetUnits.clear();
        this.checkUsable();
        checkComputeLimits(this.device.limits);
        // The platform's kernels and the techniques' compile side by side: every module's
        // code exists from construction, so nothing here waits on another's prepare. One
        // after the other, a cold start spent about 2.5 s compiling on the RTX 3060
        // (the tracer's kernels, then the sky projection's 0.7 s, then DDGI's trace
        // kernel's 1.1 s; T1.9a). Techniques still prepare in encode order among themselves.
        const platformReady = Promise.all([this.rays.prepare(this.kernels), this.shade.prepare()]);
        const techniquesReady = (async () => {
            for (const technique of this.encodeOrder()) await technique.prepare();
        })();
        await Promise.all([platformReady, techniquesReady]);
        this.checkUsable();
        this.prepared = true;
    }

    encode(encoder: GPUCommandEncoder, frame: FrameInput): void {
        this.checkUsable();
        if (!this.prepared) throw new GINotReadyError('preparing');
        const start = performance.now();
        this.frame++;

        if (this.timer) {
            // The previous frame's encoder has been submitted by now, so its timestamps can be read back.
            if (this.timerAwaitingReadback) this.timer.readback();
            // Set again only once this frame's timestamps are resolved. If anything below throws,
            // the frame's slot is never read back; the next beginFrame abandons and frees it.
            this.timerAwaitingReadback = false;
            this.collectPasses(this.timer);
            this.timer.beginFrame(this.frame);
        }

        const upload = this.sceneCopy.encode(encoder);
        this.rays.update(encoder, upload, this.timer ?? undefined);
        // Settings, and the sky's projection when the environment changed.
        this.shade.encode(encoder, this.sceneCopy.environment, this.sceneCopy.environmentVersion);

        if (!frame.budget.paused) {
            const techniques = this.encodeOrder();
            const units = unitsForFrame(frame.budget, techniques.map((technique) => this.costModel(technique)));
            const msBudget = frame.budget.gpuMs !== undefined;
            techniques.forEach((technique, index) => {
                if (!msBudget) {
                    this.lastMsBudgetUnits.delete(technique);
                    return;
                }
                units[index] = limitUnitsDrop(units[index], this.lastMsBudgetUnits.get(technique));
                this.lastMsBudgetUnits.set(technique, units[index]);
            });
            techniques.forEach((technique, index) => {
                const techniqueFrame: TechniqueFrame = {
                    frame: this.frame,
                    deltaTime: frame.deltaTime,
                    camera: frame.camera,
                    changes: upload.changes,
                    units: units[index],
                };
                const used = technique.encode(encoder, techniqueFrame);
                // Cost per unit is measured against the units actually used: a budget
                // bigger than the work would otherwise make units look cheaper and cheaper.
                if (this.timer) this.unitCosts.handOut(this.frame, technique.id, typeof used === 'number' ? Math.min(used, units[index]) : units[index]);
            });
        }

        if (this.timer) {
            this.timer.resolve(encoder);
            this.timerAwaitingReadback = true;
        }
        this.lastEncodeCpuMs = performance.now() - start;
    }

    encodeScreen(encoder: GPUCommandEncoder, screen: FrameScreen): void {
        this.checkUsable();
        if (!this.prepared) throw new GINotReadyError('preparing');
        for (const technique of this.encodeOrder()) {
            if (!technique.encodeScreen) continue;
            technique.encodeScreen(encoder, {
                frame: this.frame,
                deltaTime: 0,
                camera: IDENTITY_CAMERA,
                changes: NO_CHANGES,
                units: 0,
                screen,
            });
        }
    }

    get shading(): ShadingModule {
        if (!this.output) throw new Error('@pmndrs/ddgi: no output technique; call setOutput(technique) first');
        return this.output.shading;
    }

    get stats(): GIStats {
        const techniques: Record<string, number> = {};
        const counters: Record<string, Readonly<Record<string, number>>> = {};
        for (const { technique } of this.attached) {
            techniques[technique.id] = technique.memoryBytes;
            counters[technique.id] = technique.counters;
        }
        const topLevel = this.rays.topLevelCounters;
        if (topLevel) counters['platform top level'] = topLevel;
        const sceneCopy = this.sceneCopy.memoryBytes;
        const acceleration = this.rays.memoryBytes;
        const memory: MemoryReport = {
            sceneCopy,
            acceleration,
            techniques,
            total: sceneCopy + acceleration + Object.values(techniques).reduce((sum, bytes) => sum + bytes, 0),
        };
        return { frame: this.frame, passes: this.latestPasses, memory, counters, encodeCpuMs: this.lastEncodeCpuMs };
    }

    resetHistory(): void {
        this.unitCosts.reset();
        this.lastMsBudgetUnits.clear();
        for (const { technique } of this.attached) technique.resetHistory();
    }

    dispose(): void {
        if (this.disposed) return;
        this.disposed = true;
        this.prepared = false;
        this.device.removeEventListener('uncapturederror', this.onUncapturedError);
        for (const { technique } of this.attached.splice(0)) technique.dispose();
        this.output = undefined;
        this.shade.dispose();
        this.rays.dispose();
        this.sceneCopy.dispose();
        this.timer?.dispose();
    }

    // -----------------------------------------------------------------------
    // Internals
    // -----------------------------------------------------------------------

    private checkUsable(): void {
        if (this.lost) throw new GINotReadyError('device-lost');
        if (this.disposed) throw new Error('@pmndrs/ddgi: this platform was disposed');
    }

    private find(technique: GITechnique): AttachedTechnique | undefined {
        return this.attached.find((entry) => entry.technique === technique);
    }

    /** Fallbacks first, so a stacked technique reads its fallback's result from this frame. */
    private encodeOrder(): GITechnique[] {
        const ordered: GITechnique[] = [];
        const visit = (entry: AttachedTechnique): void => {
            if (ordered.includes(entry.technique)) return;
            const below = entry.fallback ? this.find(entry.fallback) : undefined;
            if (below) visit(below);
            ordered.push(entry.technique);
        };
        for (const entry of this.attached) visit(entry);
        return ordered;
    }

    private contextFor(entry: AttachedTechnique): TechniqueContext {
        const warn = (message: string): void => this.warnOnce(message);
        const timer: GpuTimerScope = this.timer ?? NO_TIMER;
        return {
            device: this.device,
            scene: this.sceneCopy,
            queries: this.queries,
            shade: this.shade,
            kernels: this.kernels,
            // A getter: setFallback may be called after the technique was attached.
            get fallback(): BounceSource | undefined {
                const fallback = entry.fallback;
                // The fallback's shading module is read when used, so a rebuilt module is never stale.
                return fallback ? { get shading() { return fallback.shading; } } : undefined;
            },
            timer,
            warn,
        };
    }

    /** A technique's own estimate (cost per unit, no fixed part) if it has one, otherwise the platform's measurement. */
    costModel(technique: GITechnique): UnitCostModel | undefined {
        if (technique.msPerUnit !== undefined) return { fixedMs: 0, msPerUnit: technique.msPerUnit };
        return this.unitCosts.model(technique.id);
    }

    private collectPasses(timer: GpuTimer): void {
        const samples = timer.takeSamples();
        for (const sample of samples) this.unitCosts.measure(sample.frame, sample.passes);
        const newest = samples[samples.length - 1];
        if (!newest) return;
        this.latestPasses = newest.passes.map((pass) => ({ name: pass.name, gpuMs: pass.gpuMs, frame: newest.frame }));
    }

    private warnOnce(message: string): void {
        if (this.warnedMessages.has(message)) return;
        this.warnedMessages.add(message);
        console.warn(message);
    }
}

function escapeRegExp(text: string): string {
    return text.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

/** Timestamps off: every pass goes untimed, which `beginComputePass({ timestampWrites })` accepts. */
const NO_TIMER: GpuTimerScope = { computePass: () => undefined };

/** `encodeScreen` has no camera of its own; screen stages take what they need from `screen`. */
const IDENTITY_CAMERA = {
    position: [0, 0, 0],
    view: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
    projection: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
    near: 0,
    far: 0,
} as const;

/**
 * The host's view of the scene: every call goes to the scene copy, after
 * checking that the platform is still usable (not disposed, device not lost).
 */
class GuardedScene implements SceneApi {
    private readonly target: PlatformScene;
    private readonly check: () => void;

    constructor(target: PlatformScene, check: () => void) {
        this.target = target;
        this.check = check;
    }

    createMesh(descriptor: MeshDescriptor): MeshHandle {
        this.check();
        return this.target.createMesh(descriptor);
    }
    updateMesh(mesh: MeshHandle, descriptor: MeshDescriptor): void {
        this.check();
        this.target.updateMesh(mesh, descriptor);
    }
    setDeformedPositions(mesh: MeshHandle, positions: DeformedPositions): void {
        this.check();
        this.target.setDeformedPositions(mesh, positions);
    }
    destroyMesh(mesh: MeshHandle): void {
        this.check();
        this.target.destroyMesh(mesh);
    }
    createTexture(descriptor: TextureDescriptor): TextureHandle {
        this.check();
        return this.target.createTexture(descriptor);
    }
    destroyTexture(texture: TextureHandle): void {
        this.check();
        this.target.destroyTexture(texture);
    }
    createMaterial(descriptor: MaterialDescriptor): MaterialHandle {
        this.check();
        return this.target.createMaterial(descriptor);
    }
    updateMaterial(material: MaterialHandle, changes: Partial<MaterialDescriptor>): void {
        this.check();
        this.target.updateMaterial(material, changes);
    }
    destroyMaterial(material: MaterialHandle): void {
        this.check();
        this.target.destroyMaterial(material);
    }
    createInstance(descriptor: InstanceDescriptor): InstanceHandle {
        this.check();
        return this.target.createInstance(descriptor);
    }
    setTransform(instance: InstanceHandle, transform: Mat4): void {
        this.check();
        this.target.setTransform(instance, transform);
    }
    setVisible(instance: InstanceHandle, visible: boolean): void {
        this.check();
        this.target.setVisible(instance, visible);
    }
    setMaterials(instance: InstanceHandle, materials: readonly MaterialHandle[]): void {
        this.check();
        this.target.setMaterials(instance, materials);
    }
    destroyInstance(instance: InstanceHandle): void {
        this.check();
        this.target.destroyInstance(instance);
    }
    createLight(descriptor: LightDescriptor): LightHandle {
        this.check();
        return this.target.createLight(descriptor);
    }
    updateLight(light: LightHandle, changes: LightChanges): void {
        this.check();
        this.target.updateLight(light, changes);
    }
    destroyLight(light: LightHandle): void {
        this.check();
        this.target.destroyLight(light);
    }
    setEnvironment(environment: EnvironmentDescriptor): void {
        this.check();
        this.target.setEnvironment(environment);
    }
    get bounds(): Aabb {
        return this.target.bounds;
    }
}
