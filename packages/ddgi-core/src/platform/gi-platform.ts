/**
 * The platform object a host talks to: `createGI(device, options)`.
 *
 * It owns the scene copy (`PlatformScene`), the `rays` query backend over it
 * (`PlatformRayQueries`) and a `KernelBuilder`, and runs the attached
 * techniques. Each frame, `encode` records, in this order and into the host's
 * encoder:
 *
 * 1. the scene copy's uploads (`PlatformScene.encode`), and the texture
 *    store's new layers (`gi/texture-store`, textures/texture-store.ts);
 * 2. the tracer acting on them: removed meshes, new mesh trees (the GPU's
 *    first trees at load, then tinybvh's from the worker), and the top
 *    level's refit or rebuild;
 * 2b. hit shading: its settings, and the sky's projection when the
 *    environment changed (`shade/hit-shading.ts`, T1.4);
 * 3. each technique's `encode`, fallbacks before the techniques stacked on them,
 *    with the work units its budget gives it this frame (budget v2:
 *    calibrated after `prepare` with the GPU loaded, `work-budget.ts`).
 *
 * The host owns the device. The platform watches `device.lost`; afterwards
 * `isReady` is false and calls throw `GINotReadyError('device-lost')`, and the
 * host makes a new platform on its new device.
 */

import { GINotReadyError } from './contract/errors.ts';
import type { InstanceHandle, LightHandle, MaterialHandle, MeshHandle, TextureHandle } from './contract/handles.ts';
import type { Aabb, Mat4 } from './contract/math.ts';
import type { FrameInput, FrameScreen } from './contract/frame.ts';
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
import { countComputePipelines, type CompileCounter } from './compile-counter.ts';
import { PlatformKernelBuilder } from './kernel-builder.ts';
import { mergeSceneChanges } from './scene/change-log.ts';
import { PlatformScene } from './scene/platform-scene.ts';
import { checkComputeLimits } from './device-requirements.ts';
import { PlatformHitShading } from './shade/hit-shading.ts';
import { PlatformTextureStore } from './textures/texture-store.ts';
import { GpuTimer } from './timing/gpu-timer.ts';
import { DeformingTrees } from './trace/deforming-trees.ts';
import { PlatformRayQueries, type MeshTreeOptions } from './trace/ray-queries.ts';
import { UnitCostMeter, WorkBudget, type BudgetReport, type Calibration, type CalibrationMemory, type UnitCostModel } from './work-budget.ts';


// Budget v2 lives in work-budget.ts; these names were exported from here before it.
export { DEFAULT_UNITS_PER_FRAME, type UnitCostModel } from './work-budget.ts';

/** No scene changes: the screen stage runs after `encode`, which already reported them. */
const NO_CHANGES: SceneChanges = { regions: [], global: [], receivers: [], lights: [], lightsOverflowed: false };

/**
 * The last calibration accepted per device, technique and scene, for the
 * calibration guard (work-budget.ts, CALIBRATION `guardShare`). Per device
 * and not per platform: a page or host that makes a new platform for the
 * same scene on the same device (M1's flicker check did, and got one bad
 * calibration) still has the previous one to compare with.
 */
const calibrationsByDevice = new WeakMap<GPUDevice, Map<string, Calibration>>();

interface AttachedTechnique {
    readonly technique: GITechnique;
    fallback: GITechnique | undefined;
}

/** The implementation's options: the host's, plus what the repository's own tools need. */
export interface GIPlatformImplOptions extends GIPlatformOptions {
    /**
     * Compile `rays.encodeBatch`'s kernels in `prepare` even when no attached
     * technique asks for them (`requirements.rayBatches`). For tools that
     * call `rays.encodeBatch` themselves (the trace oracle and benchmarks);
     * hosts can't reach `rays`, so the host contract has no such option.
     */
    readonly rayBatches?: boolean;
    /**
     * How mesh trees are made (T2.12): by default the scene's first trees come
     * from the GPU (H-PLOC) on a device with subgroups, and tinybvh's replace
     * them as they arrive. Tools turn the GPU off (M1's path), hold the swap
     * to check a scene mid-swap, or refuse deeper GPU trees.
     */
    readonly meshTrees?: MeshTreeOptions;
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
    /** The `rayBatches` option: compile the batch kernels whatever the techniques ask. */
    private readonly rayBatchesForTools: boolean;
    /** Counts compute pipelines made while the platform is ready (`stats.recompiles`); its device view is what the platform's own code and techniques use. */
    private readonly compiles: CompileCounter;
    private readonly timer: GpuTimer | null;
    private readonly queries: SceneQueries;
    /** Hit shading and the sky (T1.4): `giSurfaceAt`, `giDirectIrradiance`, `giSkyRadiance`, `giSkyIrradiance`. */
    readonly shade: PlatformHitShading;
    /** The texture store (M4): the scene copy makes its layers, hit shading samples them. */
    readonly textures: PlatformTextureStore;
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
    /** Pairs each frame's GPU timings with the units each technique used (budget v2's calibration reads them). */
    private readonly unitCosts = new UnitCostMeter();
    /** Each technique's units per frame: calibration, floor, saturation (work-budget.ts). */
    private readonly budgets = new Map<GITechnique, WorkBudget>();
    /**
     * A technique on this platform has finished a calibration: the host is past its start-up,
     * so a budget made from now on (a technique added at runtime) runs its burst without filler.
     */
    private calibratedOnce = false;
    /**
     * Changes from frames the techniques didn't see (the budget was paused),
     * handed to them with the next frame they do see. Why: a light changed
     * while paused would otherwise never reach DDGI's change estimate, and
     * its probes would keep their full history.
     */
    private unseenChanges: SceneChanges | null = null;
    private readonly warnedMessages = new Set<string>();
    /** The `uncapturederror` listener, kept so `dispose` can remove it. */
    private readonly onUncapturedError: (event: Event) => void;

    constructor(device: GPUDevice, options: GIPlatformImplOptions = {}) {
        this.device = device;
        this.label = options.label ?? 'gi';
        this.rayBatchesForTools = options.rayBatches === true;
        this.compiles = countComputePipelines(device, () => this.isReady);
        // The platform's own GPU work goes through the counting view; the host keeps its device as it is.
        const ownDevice = this.compiles.device;
        this.textures = new PlatformTextureStore(ownDevice, { label: this.label, warn: (message) => this.warnOnce(message) });
        this.sceneCopy = new PlatformScene(ownDevice, {
            label: this.label,
            textureStore: this.textures,
            warn: (message) => this.warnOnce(message),
            // Deforming meshes' trees are refit on the GPU, which needs subgroups; without them they stay in their rest pose.
            posesTraced: DeformingTrees.isSupported(device),
        });
        this.scene = new GuardedScene(this.sceneCopy, () => this.checkUsable());
        this.rays = new PlatformRayQueries(ownDevice, this.sceneCopy, {
            label: this.label,
            buildIn: options.buildIn,
            // The top level reads its refit stats back only in stats mode: hosts are promised no readbacks frame
            // after frame (the only one is the first mesh trees' stats, once at load; DECISIONS 2026-11-08).
            diagnostics: options.gpuTiming === true,
            warn: (message) => this.warnOnce(message),
            meshTrees: options.meshTrees,
        });
        this.kernels = new PlatformKernelBuilder(ownDevice, this.label);
        this.timer = options.gpuTiming ? new GpuTimer(ownDevice, { warn: (message) => this.warnOnce(message) }) : null;
        const rays = this.rays;
        this.queries = {
            rays,
            has: (kind: QueryKind) => kind === 'rays',
        };
        this.shade = new PlatformHitShading(ownDevice, this.kernels, this.sceneCopy.module, this.rays.module, this.textures, {
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
        this.forgetBudgets();
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
        this.forgetBudgets();
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

    get outputTechnique(): GITechnique | undefined {
        return this.output;
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
        this.forgetBudgets();
        this.checkUsable();
        checkComputeLimits(this.device.limits);
        // The platform's kernels and the techniques' compile side by side: every module's
        // code exists from construction, so nothing here waits on another's prepare. One
        // after the other, a cold start spent about 2.5 s compiling on the RTX 3060
        // (the tracer's kernels, then the sky projection's 0.7 s, then DDGI's trace
        // kernel's 1.1 s; T1.9a). Techniques still prepare in encode order among themselves.
        //
        // Only what this configuration uses is compiled (T2.14), and all of it here, so
        // nothing compiles on first use while ready (`stats.recompiles`). The batch kernels
        // behind `rays.encodeBatch` are two more compiles of the whole tracer that DDGI
        // never runs (it traces inline): only when a technique asks for them.
        //
        // Every compile is asked for before the first `await` below (the first technique's
        // included), so a host may copy its scene in while they compile: call `prepare()`,
        // let the task end (Chrome sends WebGPU calls to its GPU process when the
        // JavaScript task ends; NOTES), then describe the scene, then await.
        //
        // The mesh packing kernel is asked for first (T2.13): it is small, so it is ready
        // early, and from this call on the scene stages meshes' raw arrays for it, so a host
        // copying its scene in now uploads it while the rest compiles.
        const packingReady = Promise.all([this.sceneCopy.prepareGpuPacking(), this.sceneCopy.prepareGpuPoses()]);
        const rayBatches = this.rayBatchesForTools || this.attached.some(({ technique }) => technique.requirements.rayBatches === true);
        const platformReady = Promise.all([packingReady, this.textures.prepare(), this.shade.prepare(), this.rays.prepare(this.kernels, { batches: rayBatches })]);
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

        // A deforming mesh's pose goes into the scene copy here, and the tracer refits its tree from it in
        // the same encoder (trace/deforming-trees.ts): rays see each pose in the frame it is applied, and a
        // pose never waits for a tree build.
        const upload = this.sceneCopy.encode(encoder, this.timer ?? undefined);
        this.rays.update(encoder, upload, this.timer ?? undefined);
        // Settings, the sky's bake when the environment changed, and the light grid when there are many
        // point and spot lights. No technique traces in a paused frame, so the grid is not built then.
        const lights = frame.budget.paused ? undefined : { frame: this.frame, localLights: this.sceneCopy.localLightCount, bounds: this.sceneCopy.bounds };
        this.shade.encode(encoder, this.sceneCopy.environment, this.sceneCopy.environmentVersion, this.timer ?? undefined, lights);

        const techniques = this.encodeOrder();
        // A calibration no longer describes the scene once the geometry rays see changed: a geometry
        // change, or mesh trees still being built. It is only marked stale (the host re-calibrates
        // at its next prepare or with recalibrate()): a burst during play would make frames heavier.
        const sceneSettling = this.rays.building;
        if (sceneSettling || upload.changes.regions.some((region) => region.reason === 'geometry')) {
            for (const technique of techniques) this.budgetFor(technique).geometryChanged();
        }

        if (frame.budget.paused) {
            this.unseenChanges = this.unseenChanges ? mergeSceneChanges(this.unseenChanges, upload.changes) : upload.changes;
        } else {
            const changes = this.unseenChanges ? mergeSceneChanges(this.unseenChanges, upload.changes) : upload.changes;
            this.unseenChanges = null;
            const gpuMs = frame.budget.gpuMs;
            // The millisecond budget is shared evenly between techniques (docs/api/platform.md, gap 6).
            const shareMs = gpuMs === undefined ? undefined : gpuMs / Math.max(1, techniques.length);
            const timed = this.timer?.enabled === true;
            for (const technique of techniques) {
                if (technique.warmingUp === true) {
                    // Its own start-up work (GITechnique.warmingUp): no budget, and nothing for the
                    // calibration to measure, which starts at the first frame after it.
                    technique.encode(encoder, { frame: this.frame, deltaTime: frame.deltaTime, camera: frame.camera, changes, units: 0 });
                    continue;
                }
                const budget = this.budgetFor(technique);
                const units = budget.unitsFor({ frame: this.frame, deltaTime: frame.deltaTime, budget: frame.budget, shareMs, timed, sceneSettling });
                const techniqueFrame: TechniqueFrame = {
                    frame: this.frame,
                    deltaTime: frame.deltaTime,
                    camera: frame.camera,
                    changes,
                    units,
                };
                const reported = technique.encode(encoder, techniqueFrame);
                // Cost is measured against the units actually used: a hand-out bigger than the
                // work would otherwise make units look cheaper than they are.
                const used = typeof reported === 'number' ? Math.min(reported, units) : units;
                if (this.timer) this.unitCosts.handOut(this.frame, technique.id, used);
                budget.record(units, used);
                if (budget.report.calibration?.measured) this.calibratedOnce = true;
            }
            // The techniques have been told about the sky as it is now: the next sky change is measured from it
            // (shade/sky-change.ts). Recorded after their kernels, which read the sky as it was before.
            if (changes.global.includes('environment')) this.shade.recordSkySeen(encoder);
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
        counters['platform mesh trees'] = this.rays.meshTreeCounters;
        const deforming = this.rays.deformingTreeCounters;
        if (deforming) counters['platform deforming meshes'] = deforming;
        for (const [technique, budget] of this.budgets) counters[`${technique.id} budget`] = budgetCounters(budget.report);
        // The sky the rays see (radiance map, procedural source texture) and the hit-shading
        // uniforms are part of the scene as the platform holds it, so they count as scene copy;
        // `MemoryReport` has no bucket of its own for them, and the total must cover them.
        const sceneCopy = this.sceneCopy.memoryBytes + this.shade.memoryBytes;
        const acceleration = this.rays.memoryBytes;
        const textures = this.textures.memoryBytes;
        const memory: MemoryReport = {
            sceneCopy,
            acceleration,
            textures,
            techniques,
            total: sceneCopy + acceleration + textures + Object.values(techniques).reduce((sum, bytes) => sum + bytes, 0),
        };
        return { frame: this.frame, passes: this.latestPasses, memory, counters, encodeCpuMs: this.lastEncodeCpuMs, recompiles: this.compiles.counted };
    }

    resetHistory(): void {
        this.forgetBudgets();
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
            device: this.compiles.device,
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

    /** Measure every technique's batch again at the next millisecond frame (GIPlatform; budget v2). */
    recalibrate(): void {
        this.checkUsable();
        for (const budget of this.budgets.values()) budget.recalibrate();
    }

    /** What a technique's budget did in the latest frame (also in `stats.counters['<id> budget']`). */
    budgetReport(technique: GITechnique): BudgetReport | undefined {
        return this.budgets.get(technique)?.report;
    }

    /** The cost line a technique's batch was calibrated from: its own estimate (cost per unit, no fixed part) if it has one. */
    costModel(technique: GITechnique): UnitCostModel | undefined {
        if (technique.msPerUnit !== undefined) return { fixedMs: 0, msPerUnit: technique.msPerUnit };
        return this.budgets.get(technique)?.model;
    }

    private budgetFor(technique: GITechnique): WorkBudget {
        let budget = this.budgets.get(technique);
        if (!budget) {
            budget = new WorkBudget(technique, { fillerAllowed: !this.calibratedOnce, memory: this.calibrationMemory(technique), warn: (message) => this.warnOnce(message) });
            this.budgets.set(technique, budget);
        }
        return budget;
    }

    /**
     * The previous calibration of this technique on this device and scene. The
     * scene is known by its counts (meshes, instances, triangles) when the
     * burst ends: a different scene is a different cost.
     */
    private calibrationMemory(technique: GITechnique): CalibrationMemory {
        let byKey = calibrationsByDevice.get(this.device);
        if (!byKey) {
            byKey = new Map();
            calibrationsByDevice.set(this.device, byKey);
        }
        const memory = byKey;
        const key = (): string => {
            const counts = this.sceneCopy.counts;
            return `${technique.id}|${counts.meshes}|${counts.instances}|${counts.triangles}`;
        };
        return {
            previous: () => memory.get(key()),
            remember: (calibration) => memory.set(key(), calibration),
        };
    }

    /** New resources, kernels or techniques: what was measured no longer applies. */
    private forgetBudgets(): void {
        this.unitCosts.reset();
        this.budgets.clear();
    }

    private collectPasses(timer: GpuTimer): void {
        const samples = timer.takeSamples();
        for (const sample of samples) {
            for (const { techniqueId, sample: cost } of this.unitCosts.measure(sample.frame, sample.passes)) {
                for (const [technique, budget] of this.budgets) if (technique.id === techniqueId) budget.addSample(cost);
            }
        }
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

/** A budget report as `stats.counters` numbers (flags as 0 or 1, unknowns left out). */
function budgetCounters(report: BudgetReport): Record<string, number> {
    const counters: Record<string, number> = {
        'units this frame': report.units,
        calibrating: report.state === 'calibrating' ? 1 : 0,
        // A millisecond budget without a calibration: no timings, or the burst gave up.
        uncalibrated: report.state === 'not calibrated' ? 1 : 0,
        'floor units': report.floorUnits,
        'stepped down': report.steppedDown ? 1 : 0,
        'frames late': report.framesLate ? 1 : 0,
        stale: report.stale ? 1 : 0,
        // 1 after a burst until its batch has been timed over back-to-back frames (or for good, if the host runs none).
        confirming: report.confirming ? 1 : 0,
    };
    if (report.ceilingUnits !== undefined) counters['ceiling units'] = report.ceilingUnits;
    if (report.calibratedUnits !== undefined) counters['calibrated units'] = report.calibratedUnits;
    if (report.framesPerSecond !== undefined) counters['frames per second'] = report.framesPerSecond;
    const calibration = report.calibration;
    if (calibration?.model) {
        counters['calibration fixed ms'] = calibration.model.fixedMs;
        counters['calibration ms per unit'] = calibration.model.msPerUnit;
    }
    if (calibration) {
        // How the calibration converged: its frames, the spread of the timings its batch came
        // from, and the filler batches it ran to keep the GPU loaded. The GPU's clock itself
        // can't be read from WebGPU.
        counters['calibration frames'] = calibration.frames;
        counters['calibration spread'] = calibration.spread;
        counters['calibration filler frames'] = calibration.fillerFrames;
        counters['calibration filler ms'] = calibration.fillerMs;
        if (calibration.confirmFactor !== undefined) counters['calibration confirm factor'] = calibration.confirmFactor;
        // The calibration guard measured again: the first burst's batch.
        if (calibration.remeasuredFrom !== undefined) counters['calibration re-measured from'] = calibration.remeasuredFrom;
    }
    return counters;
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
    setDeformedPositions(mesh: MeshHandle, positions: Float32Array | DeformedPositions): void {
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
    setTransforms(instances: Uint32Array, matrices: Float32Array): void {
        this.check();
        this.target.setTransforms(instances, matrices);
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
