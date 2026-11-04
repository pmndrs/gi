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
 */

import { AnalyticLightNode, Light, NodeUpdateType, Vector4, type ExternalTexture, type Node, type NodeBuilder } from 'three/webgpu';
import { If, INFINITY, cameraPosition, float, normalWorld, positionWorld, renderGroup, screenUV, texture, texture3D, uniform, vec3, wgsl, wgslFn } from 'three/tsl';
import type { ShadingModule, ShadingParameter } from '@pmndrs/ddgi-core';
import { wrapGpuTexture } from './external-texture.ts';

/** Where several probe-like lights cover a fragment, the highest priority wins. `LightProbeGrid` defaults to 0. */
export const GI_LIGHT_PRIORITY = 1;

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

    constructor(shading: ShadingModule, options: GILightOptions = {}) {
        super(0xffffff, options.intensity ?? 1);
        (this as { type: string }).type = 'GILight';
        this.shading = shading;
        this.priority = options.priority ?? GI_LIGHT_PRIORITY;
        this.replaceEnvironmentDiffuse = options.replaceEnvironmentDiffuse ?? true;
    }
}

/** The parts of three's lighting context the probe slot and the workaround use. */
interface LightingContext {
    lightProbeIrradiance?: Node<'vec3'>;
    lightProbePriority?: Node<'float'>;
    /** Ours: this fragment's GI coverage × saturate(intensity), 0 where GI didn't claim the slot. */
    threeGiCoverage?: Node<'float'>;
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

type EntryFunction = ReturnType<typeof wgslFn>;

/**
 * One function node (and one copy of the module's WGSL) per module version,
 * shared by every GI light. Each code node is emitted once per shader, so two
 * lights with their own copies would declare the module's constants and
 * functions twice in one shader, which WGSL refuses.
 */
const entries = new WeakMap<ShadingModule, { readonly version: number; readonly entry: EntryFunction }>();

function sharedEntry(shading: ShadingModule): EntryFunction {
    const cached = entries.get(shading);
    if (cached && cached.version === shading.version) return cached.entry;
    const entry = wgslFn(entryWgsl(shading), [wgsl(shading.code)]);
    entries.set(shading, { version: shading.version, entry });
    return entry;
}

type TextureNode = ReturnType<typeof texture>;
/** A `vec4` uniform in three's render group: set once per render, shared by every object. */
function vec4Uniform(value: Vector4) {
    return uniform(value).setGroup(renderGroup);
}
type Vec4Uniform = ReturnType<typeof vec4Uniform>;

/** The nodes made for one module version, shared by every material that builds the light. */
interface BuiltModule {
    readonly module: ShadingModule;
    readonly version: number;
    readonly entry: EntryFunction;
    /** Each filterable texture's sampler description, by texture name. */
    readonly samplers: Map<string, SamplerParameter>;
    readonly textureNodes: Map<string, TextureNode>;
    readonly gpuTextures: Map<string, GPUTexture>;
    readonly uniforms: Map<string, Vec4Uniform>;
}

class GILightNode extends AnalyticLightNode<GILight> {
    static get type(): string {
        return 'GILightNode';
    }

    private built: BuiltModule | null = null;
    private readonly intensityUniform = uniform(1).setGroup(renderGroup);
    private readonly priorityUniform = uniform(GI_LIGHT_PRIORITY).setGroup(renderGroup);
    private readonly computeCoverageUniform = uniform(1).setGroup(renderGroup);

    constructor(light: GILight | null = null) {
        super(light);
        this.updateType = NodeUpdateType.RENDER;
    }

    /** Makes the texture, uniform and function nodes for the light's current module, unless they're current. */
    private buildModule(): BuiltModule {
        const shading = this.light!.shading;
        if (this.built && this.built.version === shading.version) return this.built;
        const samplers = new Map<string, SamplerParameter>();
        for (const parameter of shading.parameters) if (parameter.kind === 'sampler') samplers.set(parameter.texture, parameter);
        const textureNodes = new Map<string, TextureNode>();
        const gpuTextures = new Map<string, GPUTexture>();
        const uniforms = new Map<string, Vec4Uniform>();
        for (const parameter of shading.parameters) {
            if (parameter.kind === 'texture') {
                const gpuTexture = parameter.texture();
                const wrapped = wrapTexture(gpuTexture, parameter, samplers.get(parameter.name));
                textureNodes.set(parameter.name, (parameter.dimension === '3d' ? texture3D(wrapped) : texture(wrapped)) as TextureNode);
                gpuTextures.set(parameter.name, gpuTexture);
            } else if (parameter.kind === 'vec4f') {
                uniforms.set(parameter.name, vec4Uniform(new Vector4().fromArray(parameter.value())));
            }
        }
        const entry = sharedEntry(shading);
        this.built = { module: shading, version: shading.version, entry, samplers, textureNodes, gpuTextures, uniforms };
        return this.built;
    }

    /** Once per render: the module's current textures and uniform values, and the light's own settings. */
    update(): boolean | undefined {
        const light = this.light!;
        this.intensityUniform.value = light.intensity;
        this.priorityUniform.value = light.priority;
        this.computeCoverageUniform.value = light.computeCoverage ? 1 : 0;
        const built = this.built;
        // A module that changed shape no longer fits these nodes; ThreeGI replaces this light with a new one.
        if (!built || built.version !== light.shading.version) return undefined;
        for (const parameter of built.module.parameters) {
            if (parameter.kind === 'vec4f') {
                built.uniforms.get(parameter.name)!.value.fromArray(parameter.value());
            } else if (parameter.kind === 'texture') {
                // Reallocated by the technique (a new prepare): wrap the new texture; three rebinds when the node's value changes.
                const gpuTexture = parameter.texture();
                if (gpuTexture === built.gpuTextures.get(parameter.name)) continue;
                built.textureNodes.get(parameter.name)!.value = wrapTexture(gpuTexture, parameter, built.samplers.get(parameter.name));
                built.gpuTextures.set(parameter.name, gpuTexture);
            }
        }
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

        const resources: Record<string, Node> = {};
        for (const parameter of built.module.parameters) {
            // A sampler parameter is fed from its texture's node, which carries the sampler (NOTES).
            const name = parameter.kind === 'sampler' ? parameter.texture : parameter.name;
            resources[parameter.name] = (parameter.kind === 'vec4f' ? built.uniforms.get(name) : built.textureNodes.get(name)) as unknown as Node;
        }
        // From the eye towards the point, as the shading module expects (ddgiSampleIrradiance's view bias).
        const viewDirection = positionWorld.sub(cameraPosition).normalize();
        const lighting = (
            built.entry({
                position: positionWorld,
                normal: normalWorld,
                viewDirection,
                screenUV,
                computeCoverage: this.computeCoverageUniform,
                ...resources,
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

        if (light.replaceEnvironmentDiffuse) replaceEnvironmentDiffuse(context);
        return undefined;
    }
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
