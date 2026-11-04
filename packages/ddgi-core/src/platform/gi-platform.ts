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
 * Each new measurement of a technique's cost per unit moves its estimate this
 * share of the way, so one slow or fast frame doesn't swing the budget.
 */
export const MS_PER_UNIT_SMOOTHING = 0.1;

/** Frames of handed-out units remembered while their timings are on the way (they arrive a few frames late). */
const REMEMBERED_FRAMES = 32;

/**
 * Units each technique may spend this frame, in encode order (FrameBudget):
 * - with `gpuMs` and a cost per unit for every technique, the time is shared
 *   evenly between the techniques and each share divided by that technique's
 *   cost (docs/api/platform.md, gap 6: a real policy waits for a second
 *   technique), capped by `units` when that is set too;
 * - otherwise `units`, or {@link DEFAULT_UNITS_PER_FRAME}.
 */
export function unitsForFrame(budget: FrameBudget, msPerUnit: readonly (number | undefined)[]): number[] {
    const fallback = budget.units ?? DEFAULT_UNITS_PER_FRAME;
    const gpuMs = budget.gpuMs;
    const costsKnown = msPerUnit.every((cost) => cost !== undefined && cost > 0 && Number.isFinite(cost));
    if (gpuMs === undefined || !(gpuMs >= 0) || !costsKnown) return msPerUnit.map(() => fallback);
    const share = gpuMs / Math.max(1, msPerUnit.length);
    return msPerUnit.map((cost) => {
        const units = Math.floor(share / (cost as number));
        return budget.units === undefined ? units : Math.min(units, budget.units);
    });
}

/**
 * Measures each technique's GPU cost per unit from the platform's timer: the
 * passes named `<technique id>/...` in a frame, divided by the units the
 * technique used that frame (what its `encode` returned, or all it was
 * handed), smoothed over frames. Reset when the work changes shape: on
 * `prepare`, when a technique is added or removed, and on `resetHistory`.
 */
export class UnitCostMeter {
    private readonly handedOut = new Map<number, Map<string, number>>();
    private readonly estimates = new Map<string, number>();

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
            const msPerUnit = timed.reduce((sum, pass) => sum + pass.gpuMs, 0) / units;
            const previous = this.estimates.get(techniqueId);
            this.estimates.set(techniqueId, previous === undefined ? msPerUnit : previous + (msPerUnit - previous) * MS_PER_UNIT_SMOOTHING);
        }
        this.handedOut.delete(frame);
    }

    /** The smoothed estimate, or undefined before the first timed frame. */
    msPerUnit(techniqueId: string): number | undefined {
        return this.estimates.get(techniqueId);
    }

    reset(): void {
        this.handedOut.clear();
        this.estimates.clear();
    }
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
    /** Each technique's measured GPU cost per unit, for turning `budget.gpuMs` into units. */
    private readonly unitCosts = new UnitCostMeter();
    private readonly warnedMessages = new Set<string>();

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
        this.checkUsable();
        checkComputeLimits(this.device.limits);
        await this.rays.prepare(this.kernels);
        await this.shade.prepare();
        for (const technique of this.encodeOrder()) await technique.prepare();
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
            const units = unitsForFrame(frame.budget, techniques.map((technique) => this.msPerUnit(technique)));
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
        for (const { technique } of this.attached) technique.resetHistory();
    }

    dispose(): void {
        if (this.disposed) return;
        this.disposed = true;
        this.prepared = false;
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

    /** A technique's own estimate if it has one, otherwise the platform's measurement. */
    msPerUnit(technique: GITechnique): number | undefined {
        return technique.msPerUnit ?? this.unitCosts.msPerUnit(technique.id);
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
