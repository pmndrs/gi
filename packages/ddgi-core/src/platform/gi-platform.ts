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
import type { BounceSource, HitShading } from './contract/shade.ts';
import type { ShadingModule } from './contract/shading-module.ts';
import type { GIStats, MemoryReport, PassTiming } from './contract/stats.ts';
import type { GITechnique, TechniqueContext, TechniqueFrame } from './contract/technique.ts';
import type { GpuTimerScope } from './contract/timing.ts';
import type { WgslModule } from './contract/wgsl.ts';
import { PlatformKernelBuilder } from './kernel-builder.ts';
import { PlatformScene } from './scene/platform-scene.ts';
import { GpuTimer } from './timing/gpu-timer.ts';
import { PlatformRayQueries } from './trace/ray-queries.ts';


/**
 * Work units a technique gets when the frame's budget names none.
 *
 * This is the seam T1.5 fills in: there the platform turns `budget.gpuMs` into
 * units with each technique's `msPerUnit`, shared between techniques. Until
 * then every technique gets `budget.units`, or this many (DDGI's default probe
 * count per frame on Tier A, docs/design/ddgi.md).
 */
export const DEFAULT_UNITS_PER_FRAME = 512;

/** Units each technique may spend this frame, in encode order. */
export function unitsForFrame(budget: FrameBudget, techniques: readonly GITechnique[]): number[] {
    // T1.5: convert budget.gpuMs with technique.msPerUnit and split between techniques.
    const units = budget.units ?? DEFAULT_UNITS_PER_FRAME;
    return techniques.map(() => units);
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
    private readonly shade: HitShading;
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
    private readonly warnedMessages = new Set<string>();

    constructor(device: GPUDevice, options: GIPlatformOptions = {}) {
        this.device = device;
        this.label = options.label ?? 'gi';
        this.sceneCopy = new PlatformScene(device, { label: this.label });
        this.scene = new GuardedScene(this.sceneCopy, () => this.checkUsable());
        this.rays = new PlatformRayQueries(device, this.sceneCopy, {
            label: this.label,
            buildIn: options.buildIn,
        });
        this.kernels = new PlatformKernelBuilder(device, this.label);
        this.timer = options.gpuTiming ? new GpuTimer(device, { warn: (message) => this.warnOnce(message) }) : null;
        const rays = this.rays;
        this.queries = {
            rays,
            has: (kind: QueryKind) => kind === 'rays',
        };
        this.shade = createHitShadingPlaceholder(this.sceneCopy.module, this.rays.module);

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
        this.checkUsable();
        await this.rays.prepare(this.kernels);
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
            this.collectPasses(this.timer);
            this.timer.beginFrame(this.frame);
        }

        const upload = this.sceneCopy.encode(encoder);
        this.rays.update(encoder, upload);

        if (!frame.budget.paused) {
            const techniques = this.encodeOrder();
            const units = unitsForFrame(frame.budget, techniques);
            techniques.forEach((technique, index) => {
                const techniqueFrame: TechniqueFrame = {
                    frame: this.frame,
                    deltaTime: frame.deltaTime,
                    camera: frame.camera,
                    changes: upload.changes,
                    units: units[index],
                };
                technique.encode(encoder, techniqueFrame);
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
        for (const { technique } of this.attached) technique.resetHistory();
    }

    dispose(): void {
        if (this.disposed) return;
        this.disposed = true;
        this.prepared = false;
        for (const { technique } of this.attached.splice(0)) technique.dispose();
        this.output = undefined;
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

    private collectPasses(timer: GpuTimer): void {
        const samples = timer.takeSamples();
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
 * Hit shading arrives in T1.4 (`giSurfaceAt`, `giDirectIrradiance`,
 * `giSkyRadiance`). Until then techniques get a module with the right name and
 * dependencies but no functions, so a kernel that calls them fails to compile
 * with the shader compiler's "unresolved function" message.
 */
function createHitShadingPlaceholder(sceneModule: WgslModule, raysModule: WgslModule): HitShading {
    return {
        module: {
            name: 'gi-shade',
            dependencies: [sceneModule, raysModule],
            code: () => '// gi-shade: hit shading is not implemented yet (T1.4).\n',
            entries: [],
            resources: () => [],
            version: 0,
            storageBufferCount: 0,
        },
        settings: { maxShadowedLightsPerHit: 4, bounceAlbedoCap: 1 },
    };
}

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
