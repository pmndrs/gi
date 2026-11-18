/**
 * The GI light: how three's materials read the platform's GI.
 *
 * A three `Light` subclass with a registered `AnalyticLightNode`, the pattern
 * three's own `LightProbeGrid` uses (r187 `Light.registerNode`; DECISIONS
 * 2026-11-04: never `material.contextNode`, which the Meshlet and
 * Visibility-Buffer pipelines overwrite). Adding it to a scene lights every
 * lit node material, with no material changes. The T0.8 spike proved the
 * shape under the Standard, Meshlet and Visibility-Buffer pipelines
 * (tools/gpu/pages/spike-three-lib).
 *
 * Per fragment the node calls the active technique's shading module
 * (`gi.shading`) through one `wgslFn`:
 *
 * - **Irradiance** E from `giIrradiance` goes into three's light-probe slot
 *   (`context.lightProbeIrradiance`, with `lightProbePriority`), which
 *   `LightsNode` adds to `irradiance` once after all lights. three then
 *   applies albedo / π (NOTES: `irradiance` is E). Where several probe-like
 *   lights cover a fragment (a `LightProbeGrid`, say), the highest priority
 *   wins instead of both adding up. The default priority, 1, is distinct from
 *   `LightProbeGrid`'s default 0, because ties are broken by an unreliable
 *   id order (NOTES).
 * - **Coverage** from `giCoverage` decides where the light claims that slot
 *   (coverage above 0), and drives the environment-diffuse workaround below.
 *   It costs as much as the irradiance (another pass over the probes), so it
 *   is computed only when `computeCoverage` is on; `ThreeGI` turns it off
 *   when nothing needs it (no `scene.environment` and no other probe lights).
 *   With it off, coverage reads as 1 everywhere.
 *
 * The module's resources reach three as `ExternalTexture`s (re-read from
 * `texture()` each render and re-wrapped when the GPU texture changes) and
 * `renderGroup` `vec4` uniforms copied from `value()`; samplers are fed from
 * their texture's node, with that texture's filter and wrap set from the
 * module's sampler description (NOTES, `wgslFn` resources).
 *
 * Toggle it with `intensity` (0 = off). Adding or removing a light recompiles
 * every lit material (NOTES: `LightsNode.customCacheKey`); intensity is a
 * uniform. At intensity 0 it gives up the probe slot and stops suppressing
 * the environment.
 *
 * **The render-time update** (T1.14): with `onRender` set (`ThreeGI` sets it
 * for `updateOnRender`, which `DDGIVolume.start` uses), the node runs the GI
 * frame from its own render-time update, as three's `LightProbeGridNode`
 * updates itself, so the host has no per-frame call. In r187 a light node's
 * `RENDER` update runs inside the render pass's encoder lifetime: after
 * `backend.beginRender` created the pass's encoder and before `finishRender`
 * submits it (NOTES). The GI frame submits its own encoder there, so the
 * queue runs it before the render that samples it (checked on the GPU, the
 * three-composite page). Then the node reads the module's values, so this
 * render sees this frame's GI. What follows from running inside the render:
 * - **once per frame:** a frame may hold several lit renders (cube
 *   captures, reflections, several passes), and the update runs once. When
 *   `renderer.setAnimationLoop` drives the frames, "a frame" is three's own
 *   frame id (`NodeFrame.frameId`, which only that loop advances, NOTES).
 *   Otherwise it is one JavaScript task (a `requestAnimationFrame`
 *   callback): a host that drives frames itself and awaits between two
 *   renders of one frame (`await renderer.renderAsync()` twice, say) gets a
 *   GI frame per render. Use `setAnimationLoop`, or render synchronously;
 * - **the camera** is the first lit render's camera that frame. With cube
 *   captures or a reflection pass drawn before the main view, GI's update
 *   order follows that camera for the frame;
 * - **no lit render, no GI frame:** a frame that draws no lit object with
 *   the light (everything culled, GI's layer hidden) runs no GI work, so the
 *   field doesn't converge meanwhile;
 * - **errors** in the GI frame are reported once through `onRenderError` and
 *   stop the render-time update; three's render carries on without it.
 *
 * **Environment diffuse counted twice** (NOTES): r187 adds
 * `scene.environment`'s diffuse light (`iblIrradiance`) separately from
 * probe GI and has no switch (PR #35017 adds one). With
 * `replaceEnvironmentDiffuse` on (the default), the node wraps the lighting
 * model's `indirect` step to scale `iblIrradiance` by
 * 1 − coverage × saturate(intensity) first: `LightingModel.start` builds all
 * lights before calling `indirect`, so coverage is final by then.
 * - Standard and Physical materials: env diffuse fades out where GI covers;
 *   env specular stays. It also scales the env terms of the multi-scatter
 *   and sheen compensation inside the volume, which read `iblIrradiance` too.
 * - Lambert and Phong materials: unaffected, and nothing to fix: their
 *   lighting model never adds diffuse light from `scene.environment` (it only
 *   uses the environment as a reflection map), so GI is not doubled there.
 * - Basic materials are unlit and ignore lights, GI included.
 *
 * **In a composited pass** (composite.ts, T1.14): when the pass carries the
 * composite's mark and the surface is one the composite lights
 * (`compositeTakesGI`, composite-mark.ts), the node evaluates no irradiance.
 * It still claims the light-probe slot where GI covers, with zero, so a
 * lower-priority probe light doesn't add its light under the composite's,
 * and the environment-diffuse workaround runs as above. Coverage is then the
 * only per-fragment cost, and only when `computeCoverage` is on. Every other
 * surface, and every unmarked pass, builds exactly as before.
 *
 * The composite must add GI exactly where GI won the slot: a probe light of
 * higher priority (a `LightProbeGrid`, say) may take it after GI claimed it,
 * and then that light's irradiance is shaded per fragment. Once every light
 * has run (the lighting model's `indirect` step), the node writes
 * {@link GI_COMPOSITE_WON}: 1 where the slot still holds GI's claim (GI
 * claimed it, the slot's priority is GI's, and its irradiance is still
 * GI's zero), else 0. `giAlbedo` multiplies its weight by it, so where
 * another light won, the composite adds nothing. On an exact tie GI keeps
 * the slot only if the later light wrote zero irradiance too.
 */

import { AnalyticLightNode, Light, NodeUpdateType, Vector4, type Camera, type ExternalTexture, type Node, type NodeBuilder, type NodeFrame } from 'three/webgpu';
import { If, INFINITY, cameraPosition, float, normalWorld, positionWorld, property, renderGroup, screenUV, select, texture, texture3D, uniform, vec3, wgsl, wgslFn } from 'three/tsl';
import type { ShadingModule, ShadingParameter } from '@pmndrs/ddgi-core';
import { compositeTakesGI } from './composite-mark.ts';
import { wrapGpuTexture } from './external-texture.ts';

/** Where several probe-like lights cover a fragment, the highest priority wins. `LightProbeGrid` defaults to 0. */
export const GI_LIGHT_PRIORITY = 1;

/**
 * In a composited build: 1 where GI won the light-probe slot once every
 * light has run, else 0 (file comment). Written by the GI light's node at
 * the lighting model's `indirect` step; read by `giAlbedo` (composite.ts).
 * A shader-wide variable, so never written means 0: no GI light, no
 * composited GI.
 */
export const GI_COMPOSITE_WON = property('float', 'ThreeGiCompositeWon');

export interface GILightOptions {
    /** Multiplies the GI. 0 turns it off without recompiling. Default 1. */
    readonly intensity?: number;
    /** Default {@link GI_LIGHT_PRIORITY}. */
    readonly priority?: number;
    /** Fade `scene.environment`'s diffuse light where GI covers (Standard and Physical materials). Default true. */
    readonly replaceEnvironmentDiffuse?: boolean;
}

type TextureParameter = Extract<ShadingParameter, { kind: 'texture' }>;
type SamplerParameter = Extract<ShadingParameter, { kind: 'sampler' }>;

export class GILight extends Light {
    readonly isGILight = true;
    /**
     * The module the node reads, fixed for the light's life. three caches
     * compiled shaders by a key built from material properties and light
     * ids, not by material version, so a module of a new shape only reaches
     * the shaders through a new light (a new id). `ThreeGI` swaps lights when
     * the platform's module changes shape.
     */
    readonly shading: ShadingModule;
    priority: number;
    /** Fixed for the light's life, for the same reason as `shading`. */
    readonly replaceEnvironmentDiffuse: boolean;
    /** Compute coverage per pixel (see the file comment). A uniform, so changing it is free. */
    computeCoverage = true;
    /**
     * The GI frame, run from the node's render-time update once per frame
     * with the render's camera (file comment, "The render-time update").
     * Null: the host calls `ThreeGI.update` itself.
     */
    onRender: ((camera: Camera) => void) | null = null;
    /** Where an error from `onRender` goes (once); the render-time update stops after it. Default `console.error`. */
    onRenderError: (error: unknown) => void = (error) => console.error('@pmndrs/gi/three: the GI frame failed during a render; GI stops updating.', error);
    /** True once `onRender` threw: the render-time update stays off. */
    renderUpdateFailed = false;
    /** Whether `onRender` already ran in the current JavaScript task (without an animation loop). */
    private ranThisTask = false;
    /** three's frame id `onRender` last ran in (with an animation loop). */
    private lastFrameId: number | null = null;

    constructor(shading: ShadingModule, options: GILightOptions = {}) {
        super(0xffffff, options.intensity ?? 1);
        (this as { type: string }).type = 'GILight';
        this.shading = shading;
        this.priority = options.priority ?? GI_LIGHT_PRIORITY;
        this.replaceEnvironmentDiffuse = options.replaceEnvironmentDiffuse ?? true;
    }

    /**
     * Runs `onRender` unless it already ran this frame (file comment, "once
     * per frame"). `frameId` is three's frame id when `setAnimationLoop`
     * drives the frames, and keys the guard; null otherwise, and then the
     * guard is the JavaScript task: a microtask clears the mark once the
     * task's synchronous renders are done.
     */
    runRenderTimeUpdate(camera: Camera | null, frameId: number | null = null): void {
        const update = this.onRender;
        if (!update || !camera || this.renderUpdateFailed) return;
        if (frameId !== null) {
            if (frameId === this.lastFrameId) return;
            this.lastFrameId = frameId;
        } else {
            if (this.ranThisTask) return;
            this.ranThisTask = true;
            queueMicrotask(() => {
                this.ranThisTask = false;
            });
        }
        try {
            update(camera);
        } catch (error) {
            this.renderUpdateFailed = true;
            this.onRenderError(error);
        }
    }
}

/** The parts of three's lighting context the probe slot and the workaround use. */
interface LightingContext {
    lightProbeIrradiance?: Node<'vec3'>;
    lightProbePriority?: Node<'float'>;
    /** Ours: this fragment's GI coverage × saturate(intensity), 0 where GI didn't claim the slot. */
    threeGiCoverage?: Node<'float'>;
    /** Ours, composited builds: 1 once GI claimed the slot (another light may take it later). */
    threeGiCompositeClaimed?: Node<'float'>;
    iblIrradiance?: Node<'vec3'>;
    lightingModel: { indirect(builder: NodeBuilder): void };
}

/** Lighting models whose `indirect` step is already wrapped (each material build makes its own model). */
const patchedLightingModels = new WeakSet<object>();

/** Name of the generated WGSL entry the node calls. */
const ENTRY_NAME = 'giThreeLighting';

/** WGSL type of a module parameter, as `wgslFn` parses it. */
export function wgslParameterType(parameter: ShadingParameter): string {
    if (parameter.kind === 'vec4f') return 'vec4f';
    if (parameter.kind === 'sampler') return 'sampler';
    const sampled = parameter.sampleType === 'uint' ? 'u32' : 'f32';
    const dimension = parameter.dimension === '2d-array' ? '2d_array' : parameter.dimension;
    return `texture_${dimension}<${sampled}>`;
}

/**
 * The one function the node calls: irradiance and (when asked) coverage in
 * one `vec4f`, from the module's own entry points, with every resource
 * passed through by the module's parameter names. Generated from the
 * parameter list, so it works for any technique's module.
 */
export function entryWgsl(shading: ShadingModule): string {
    const declarations = shading.parameters.map((parameter) => `    ${parameter.name}: ${wgslParameterType(parameter)},`).join('\n');
    const resources = shading.parameters.map((parameter) => parameter.name).join(', ');
    return /* wgsl */ `fn ${ENTRY_NAME}(
    position: vec3f,
    normal: vec3f,
    viewDirection: vec3f,
    screenUV: vec2f,
    computeCoverage: f32,
${declarations}
) -> vec4f {
    let irradiance = giIrradiance(position, normal, viewDirection, screenUV, ${resources});
    // Coverage costs another pass over the field, so it's skipped (read as 1) when nothing needs it.
    var coverage = 1.0;
    if (computeCoverage > 0.5) {
        coverage = giCoverage(position, normal, ${resources});
    }
    return vec4f(irradiance, coverage);
}`;
}

/** A module texture parameter wrapped for three, with its sampler's filter and wrap (external-texture.ts). */
function wrapTexture(gpuTexture: GPUTexture, parameter: TextureParameter, sampler: SamplerParameter | undefined): ExternalTexture {
    return wrapGpuTexture(gpuTexture, { dimension: parameter.dimension, sampleType: parameter.sampleType, filtering: sampler?.filtering, addressMode: sampler?.addressMode });
}

/** Name of the generated WGSL coverage-only entry, for composited surfaces. */
const COVERAGE_ENTRY_NAME = 'giThreeCoverage';

/**
 * Coverage alone, for a composited surface (file comment): the composite
 * adds the irradiance, so per fragment only the workaround and the probe
 * slot need coverage, and only when it's asked for (1 otherwise, as in
 * {@link entryWgsl}).
 */
export function coverageEntryWgsl(shading: ShadingModule): string {
    const declarations = shading.parameters.map((parameter) => `    ${parameter.name}: ${wgslParameterType(parameter)},`).join('\n');
    const resources = shading.parameters.map((parameter) => parameter.name).join(', ');
    return /* wgsl */ `fn ${COVERAGE_ENTRY_NAME}(
    position: vec3f,
    normal: vec3f,
    computeCoverage: f32,
${declarations}
) -> f32 {
    var coverage = 1.0;
    if (computeCoverage > 0.5) {
        coverage = giCoverage(position, normal, ${resources});
    }
    return coverage;
}`;
}

type EntryFunction = ReturnType<typeof wgslFn>;

interface SharedEntries {
    readonly version: number;
    /** The module's WGSL, one node, included by every entry so a shader declares it once. */
    readonly code: ReturnType<typeof wgsl>;
    readonly entry: EntryFunction;
    /** Made on first use: only composited builds call it. */
    coverageEntry: EntryFunction | null;
}

/**
 * One function node (and one copy of the module's WGSL) per module version,
 * shared by every GI light and the composite. Each code node is emitted once
 * per shader, so two lights with their own copies would declare the module's
 * constants and functions twice in one shader, which WGSL refuses.
 */
const entries = new WeakMap<ShadingModule, SharedEntries>();

function sharedEntries(shading: ShadingModule): SharedEntries {
    const cached = entries.get(shading);
    if (cached && cached.version === shading.version) return cached;
    const code = wgsl(shading.code);
    const shared: SharedEntries = { version: shading.version, code, entry: wgslFn(entryWgsl(shading), [code]), coverageEntry: null };
    entries.set(shading, shared);
    return shared;
}

/** The irradiance-and-coverage function of {@link entryWgsl} for this module version. */
export function sharedEntry(shading: ShadingModule): EntryFunction {
    return sharedEntries(shading).entry;
}

/** The coverage-only function of {@link coverageEntryWgsl} for this module version. */
function sharedCoverageEntry(shading: ShadingModule): EntryFunction {
    const shared = sharedEntries(shading);
    shared.coverageEntry ??= wgslFn(coverageEntryWgsl(shading), [shared.code]);
    return shared.coverageEntry;
}

type TextureNode = ReturnType<typeof texture>;
/** A `vec4` uniform in three's render group: set once per render, shared by every object. */
function vec4Uniform(value: Vector4) {
    return uniform(value).setGroup(renderGroup);
}
type Vec4Uniform = ReturnType<typeof vec4Uniform>;

/**
 * The three nodes for one module version's resources: its textures wrapped
 * as `ExternalTexture`s and its `vec4f` values as `renderGroup` uniforms.
 * Made by the GI light's node and by the composite (composite.ts), each for
 * its own shaders; `refresh` once per render keeps them current.
 */
export class ShadingModuleNodes {
    readonly module: ShadingModule;
    readonly version: number;
    /** Each filterable texture's sampler description, by texture name. */
    private readonly samplers = new Map<string, SamplerParameter>();
    private readonly textureNodes = new Map<string, TextureNode>();
    private readonly gpuTextures = new Map<string, GPUTexture>();
    private readonly uniforms = new Map<string, Vec4Uniform>();

    constructor(shading: ShadingModule) {
        this.module = shading;
        this.version = shading.version;
        for (const parameter of shading.parameters) if (parameter.kind === 'sampler') this.samplers.set(parameter.texture, parameter);
        for (const parameter of shading.parameters) {
            if (parameter.kind === 'texture') {
                const gpuTexture = parameter.texture();
                const wrapped = wrapTexture(gpuTexture, parameter, this.samplers.get(parameter.name));
                this.textureNodes.set(parameter.name, (parameter.dimension === '3d' ? texture3D(wrapped) : texture(wrapped)) as TextureNode);
                this.gpuTextures.set(parameter.name, gpuTexture);
            } else if (parameter.kind === 'vec4f') {
                this.uniforms.set(parameter.name, vec4Uniform(new Vector4().fromArray(parameter.value())));
            }
        }
    }

    /** Whether these nodes still fit the module (a new version may have another shape). */
    fits(shading: ShadingModule): boolean {
        return shading === this.module && shading.version === this.version;
    }

    /** The module's resources by parameter name, as the entry functions take them. */
    resources(): Record<string, Node> {
        const resources: Record<string, Node> = {};
        for (const parameter of this.module.parameters) {
            // A sampler parameter is fed from its texture's node, which carries the sampler (NOTES).
            const name = parameter.kind === 'sampler' ? parameter.texture : parameter.name;
            resources[parameter.name] = (parameter.kind === 'vec4f' ? this.uniforms.get(name) : this.textureNodes.get(name)) as unknown as Node;
        }
        return resources;
    }

    /** The module's current uniform values, and a new wrap for any texture the technique reallocated. */
    refresh(): void {
        for (const parameter of this.module.parameters) {
            if (parameter.kind === 'vec4f') {
                this.uniforms.get(parameter.name)!.value.fromArray(parameter.value());
            } else if (parameter.kind === 'texture') {
                // Reallocated by the technique (a new prepare): wrap the new texture; three rebinds when the node's value changes.
                const gpuTexture = parameter.texture();
                if (gpuTexture === this.gpuTextures.get(parameter.name)) continue;
                this.textureNodes.get(parameter.name)!.value = wrapTexture(gpuTexture, parameter, this.samplers.get(parameter.name));
                this.gpuTextures.set(parameter.name, gpuTexture);
            }
        }
    }
}

class GILightNode extends AnalyticLightNode<GILight> {
    static get type(): string {
        return 'GILightNode';
    }

    private built: ShadingModuleNodes | null = null;
    private readonly intensityUniform = uniform(1).setGroup(renderGroup);
    private readonly priorityUniform = uniform(GI_LIGHT_PRIORITY).setGroup(renderGroup);
    private readonly computeCoverageUniform = uniform(1).setGroup(renderGroup);

    constructor(light: GILight | null = null) {
        super(light);
        this.updateType = NodeUpdateType.RENDER;
    }

    /** Makes the texture and uniform nodes for the light's current module, unless they're current. */
    private buildModule(): ShadingModuleNodes {
        const shading = this.light!.shading;
        if (this.built && this.built.version === shading.version) return this.built;
        this.built = new ShadingModuleNodes(shading);
        return this.built;
    }

    /**
     * Once per render: the GI frame when the light runs it here (file
     * comment, "The render-time update"), then the module's current textures
     * and uniform values, and the light's own settings.
     */
    update(frame: NodeFrame): boolean | undefined {
        const light = this.light!;
        // three's frame id is a frame only while its animation loop runs (NOTES); otherwise the task is.
        const renderer = (frame as unknown as { renderer: { getAnimationLoop(): unknown } | null }).renderer;
        const loopRunning = renderer !== null && renderer.getAnimationLoop() !== null;
        light.runRenderTimeUpdate(frame.camera, loopRunning ? frame.frameId : null);
        this.intensityUniform.value = light.intensity;
        this.priorityUniform.value = light.priority;
        this.computeCoverageUniform.value = light.computeCoverage ? 1 : 0;
        const built = this.built;
        // A module that changed shape no longer fits these nodes; ThreeGI replaces this light with a new one.
        if (!built || built.version !== light.shading.version) return undefined;
        built.refresh();
        return undefined;
    }

    setup(builder: NodeBuilder): Node | undefined {
        const light = this.light!;
        const built = this.buildModule();
        const context = builder.context as unknown as LightingContext;

        // The light-probe slot, shared by every probe-like light of this build.
        context.lightProbeIrradiance ||= vec3().toVar('lightProbeIrradiance');
        context.lightProbePriority ||= INFINITY.negate().toVar('lightProbePriority');
        // TSL variable names share the shader's namespace with the module's WGSL, whose
        // names start with gi/ddgi (platform rule), so ours start with `threeGi`.
        context.threeGiCoverage ||= float(0).toVar('threeGiCoverage');

        if (compositeTakesGI(builder)) {
            this.setupComposited(context, built);
        } else {
            this.setupPerFragment(context, built);
        }

        if (light.replaceEnvironmentDiffuse) replaceEnvironmentDiffuse(context);
        return undefined;
    }

    /** GI per fragment: irradiance (and coverage) from the module, into the probe slot. */
    private setupPerFragment(context: LightingContext, built: ShadingModuleNodes): void {
        const light = this.light!;
        const entry = sharedEntry(built.module);
        // From the eye towards the point, as the shading module expects (ddgiSampleIrradiance's view bias).
        const viewDirection = positionWorld.sub(cameraPosition).normalize();
        const lighting = (
            entry({
                position: positionWorld,
                normal: normalWorld,
                viewDirection,
                screenUV,
                computeCoverage: this.computeCoverageUniform,
                ...built.resources(),
            }) as unknown as Node<'vec4'>
        ).toVar('threeGiLighting');
        const coverage = lighting.w;
        const claims = this.priorityUniform
            .greaterThanEqual(context.lightProbePriority!)
            .and(coverage.greaterThan(0))
            .and(this.intensityUniform.greaterThan(0));

        If(claims, () => {
            context.lightProbeIrradiance!.assign(lighting.xyz.mul(this.intensityUniform));
            context.lightProbePriority!.assign(this.priorityUniform);
            // Only a light that replaces the environment's diffuse light says how much to remove.
            if (light.replaceEnvironmentDiffuse) context.threeGiCoverage!.assign(coverage.mul(this.intensityUniform.saturate()));
        });
    }

    /**
     * A composited surface (file comment): the composite adds the irradiance
     * after the pass, so the slot is claimed with zero where GI covers, and
     * only coverage is evaluated here (when asked for).
     */
    private setupComposited(context: LightingContext, built: ShadingModuleNodes): void {
        const light = this.light!;
        const coverage = (
            sharedCoverageEntry(built.module)({
                position: positionWorld,
                normal: normalWorld,
                computeCoverage: this.computeCoverageUniform,
                ...built.resources(),
            }) as unknown as Node<'float'>
        ).toVar('threeGiCompositeCoverage');
        const claims = this.priorityUniform
            .greaterThanEqual(context.lightProbePriority!)
            .and(coverage.greaterThan(0))
            .and(this.intensityUniform.greaterThan(0));

        context.threeGiCompositeClaimed ||= float(0).toVar('threeGiCompositeClaimed');
        If(claims, () => {
            context.lightProbeIrradiance!.assign(vec3(0));
            context.lightProbePriority!.assign(this.priorityUniform);
            context.threeGiCompositeClaimed!.assign(1);
            if (light.replaceEnvironmentDiffuse) context.threeGiCoverage!.assign(coverage.mul(this.intensityUniform.saturate()));
        });
        recordCompositeWinner(context, this.priorityUniform);
    }
}

/** Lighting models whose `indirect` step already records the composite's winner. */
const winnerRecordingModels = new WeakSet<object>();

/**
 * Wraps this build's lighting model's `indirect` step, which runs once every
 * light has had its turn at the probe slot, to write {@link GI_COMPOSITE_WON}
 * (file comment). Once per lighting model.
 */
function recordCompositeWinner(context: LightingContext, priority: Node<'float'>): void {
    const model = context.lightingModel;
    if (!model || winnerRecordingModels.has(model)) return;
    winnerRecordingModels.add(model);
    const originalIndirect = model.indirect.bind(model);
    model.indirect = (builder: NodeBuilder) => {
        const current = builder.context as unknown as LightingContext;
        const claimed = current.threeGiCompositeClaimed;
        const slot = current.lightProbeIrradiance;
        const slotPriority = current.lightProbePriority;
        if (claimed && slot && slotPriority) {
            // Still GI's claim: GI's priority and GI's zero. A light that took the slot after GI changed one of them.
            const stillGi = claimed.greaterThan(0.5).and(slotPriority.equal(priority)).and(slot.dot(slot).equal(0));
            GI_COMPOSITE_WON.assign(select(stillGi, float(1), float(0)));
        }
        originalIndirect(builder);
    };
}

/**
 * Wraps this build's lighting model's `indirect` step so it first scales the
 * environment's diffuse input by 1 − GI coverage (file comment). Once per
 * lighting model.
 */
function replaceEnvironmentDiffuse(context: LightingContext): void {
    const model = context.lightingModel;
    if (!model || patchedLightingModels.has(model)) return;
    patchedLightingModels.add(model);
    const originalIndirect = model.indirect.bind(model);
    model.indirect = (builder: NodeBuilder) => {
        const current = builder.context as unknown as LightingContext;
        if (current.iblIrradiance && current.threeGiCoverage) current.iblIrradiance.mulAssign(float(1).sub(current.threeGiCoverage));
        originalIndirect(builder);
    };
}

// `registerNode` stores the node class on the prototype; LightsNode makes one node per light from it.
GILight.registerNode(GILightNode);
