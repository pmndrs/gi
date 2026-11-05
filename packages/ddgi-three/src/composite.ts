/**
 * GI as an additive composite on three's post-processing pipeline (T1.14,
 * `docs/design/three-composite.md`): the fast path for dense scenes, where
 * GI shaded per fragment runs many times per pixel (Sponza's sub-pixel
 * triangles: +12 ms at 1440p). Built like three r187's SSGI example:
 *
 * ```ts
 * await volume.start(renderer, scene);                 // the simple path (gi-volume.ts)
 * const pipeline = new RenderPipeline(renderer);
 * const scenePass = pass(scene, camera);
 * scenePass.setMRT(mrt({ output, giAlbedo, normal: packNormalToRGB(normalView) }));
 * pipeline.outputNode = ddgi(scenePass, volume);
 * ```
 *
 * `ddgi` and `giAlbedo` are the names Dennis confirmed (DECISIONS, 607c1f0).
 *
 * **Why it equals per-fragment GI.** three's materials take indirect diffuse
 * light as `E × weight / π`, linear in E (r187
 * `PhysicalLightingModel.indirectDiffuse`, then `ambientOcclusion`). The
 * scene pass writes `weight` per pixel (`giAlbedo`), the GI light adds no
 * indirect diffuse on those surfaces (gi-light.ts, the composite mark), and
 * the composite evaluates E once per pixel at the same point with the same
 * normal and adds `weight × E / π` to the pass's linear HDR colour, before
 * the pipeline's tone mapping. What differs: the position comes back from
 * the depth buffer, the normal and the weight go through the pass's
 * render-target format, and at an edge a pixel holds one surface's normal
 * (no MSAA in the pass) where per-fragment shading has one per sample.
 *
 * **Which surfaces** (composite-mark.ts, `compositeTakesGI`): opaque
 * surfaces whose lighting model `giAlbedo` reproduces. Transparent and other
 * surfaces keep GI per fragment and write a weight of 0.
 *
 * **Transparent surfaces in front of composited ones.** A transparent
 * surface writes weight 0 with its own alpha, and the composite gives the
 * `giAlbedo` target the material's blending, so what's behind keeps the
 * share of its weight that its colour keeps (`1 − α` for normal blending).
 * The normal and depth targets hold the transparent surface's own, though
 * (they don't blend), so that share of GI is evaluated at the transparent
 * surface rather than behind it. Measured in the three-composite page.
 *
 * Fog: the composite adds GI after the material's fog, so a fogged scene
 * gets un-fogged GI. Neither test scene has fog.
 *
 * **Short-range occlusion on GI** (DECISIONS 2026-11-05, curtained Sponza,
 * option A): probes 1 m apart can't follow light that changes inside small
 * folds (Sponza's curtains read +23% against Cycles without it). By default
 * the composite runs three's own GTAO (`ao()`, GTAONode) on the pass's depth
 * and normal, with a 1 m radius, and multiplies the GI term by it, and only
 * the GI term: direct light and the environment's light are untouched. The
 * option `occlusion` sets the radius, resolution and samples, turns it off
 * (`false`), or takes an AO texture the pipeline already makes (`{ source }`,
 * a GTAONode on this pass's depth and normal) instead of running a second
 * GTAO. If the pass already applies AO to its materials (a context with
 * `getAO`, three's `builtinAOContext`), `giAlbedo` carries that AO already,
 * so the default is then no second occlusion.
 */

import { BlendMode, MaterialBlending, Node, NodeUpdateType, PhysicalLightingModel, type NodeBuilder, type PassNode, type TextureNode } from 'three/webgpu';
import type GTAONode from 'three/addons/tsl/display/GTAONode.js';
import { ao } from 'three/addons/tsl/display/GTAONode.js';
import {
    DFGLUT,
    Fn,
    If,
    ambientOcclusion,
    diffuseColor,
    diffuseContribution,
    getViewPosition,
    isBackgroundDepth,
    normalView,
    positionViewDirection,
    roughness,
    sample,
    specularColor,
    specularF90,
    uniform,
    unpackRGBToNormal,
    uv,
    vec3,
    vec4,
} from 'three/tsl';
import { compositeTakesGI, compositedModel, flowContextData, markCompositePass } from './composite-mark.ts';
import { ShadingModuleNodes, sharedEntry, type GILight } from './gi-light.ts';

/** The MRT output name the composite reads the GI weight from. */
export const GI_ALBEDO_OUTPUT = 'giAlbedo';
/** The MRT output name the composite reads the normal from: `packNormalToRGB(normalView)`, as three's SSGI and GTAO examples write it. */
export const GI_NORMAL_OUTPUT = 'normal';

type TslNode = ReturnType<typeof vec3>;

/** Whether this render lights the material with a GI light (camera layers and visibility decide, in `LightsNode`). */
function rendersGILight(builder: NodeBuilder): boolean {
    const lightsNode = (builder as unknown as { lightsNode: { getLights(): ReadonlyArray<{ isGILight?: boolean }> } | null }).lightsNode;
    return lightsNode?.getLights().some((light) => light.isGILight === true) ?? false;
}

/**
 * `PhysicalLightingModel`'s weight on indirect diffuse light:
 * `diffuseContribution × (1 − F_ss − F_ms)`, the share the specular lobe
 * leaves to the diffuse layer (r187 `indirectDiffuse`, with the dielectric
 * scattering terms `start` computes). Built with three's own
 * `computeMultiscattering` and `DFGLUT` on the material's own properties, so
 * it follows three if the formula changes. Without iridescence, which routes
 * a material away from the composite (composite-mark.ts), the dielectric F0
 * is `specularColor`.
 */
function physicalIndirectDiffuseWeight(): TslNode {
    const dotNV = normalView.dot(positionViewDirection).clamp();
    // computeMultiscattering reads only `this.dfg`, which `start` sets the same way.
    const scattering = { dfg: DFGLUT({ roughness, dotNV }) };
    const singleScattering = vec3().toVar('threeGiSingleScattering');
    const multiScattering = vec3().toVar('threeGiMultiScattering');
    // The typings (0.187) list an older signature; r187's takes (singleScatter, multiScatter, specularF90, f0, iridescenceF0).
    const computeMultiscattering = PhysicalLightingModel.prototype.computeMultiscattering as unknown as (
        this: object,
        singleScatter: unknown,
        multiScatter: unknown,
        specularF90: unknown,
        f0: unknown,
    ) => void;
    computeMultiscattering.call(scattering, singleScattering, multiScattering, specularF90, specularColor);
    return diffuseContribution.mul(singleScattering.add(multiScattering).oneMinus()) as unknown as TslNode;
}

/**
 * The `giAlbedo` MRT node: per pixel, the weight three's lighting puts on
 * indirect diffuse light (rgb), for the composite to multiply by E / π.
 *
 * - Composited surfaces (composite-mark.ts): `diffuseContribution ×
 *   (1 − F_ss − F_ms)` for Standard and Physical materials, `diffuseColor`
 *   for Lambert and Phong, times the material's ambient occlusion (its AO
 *   map, and a pass context's AO such as GTAO's `builtinAOContext`) as three
 *   applies it to indirect diffuse. Alpha 1.
 * - Every other surface: 0, because it shades GI itself or gets none. A
 *   transparent surface writes its alpha, so the material's blending dims
 *   what's behind it (file comment).
 *
 * Use one instance, {@link giAlbedo}, like three's `diffuseColor`.
 */
export class GIAlbedoNode extends Node {
    static get type(): string {
        return 'GIAlbedoNode';
    }

    readonly isGIAlbedoNode = true;

    constructor() {
        super('vec4');
    }

    setup(builder: NodeBuilder): Node {
        const material = builder.material;
        if (!compositeTakesGI(builder) || !rendersGILight(builder)) {
            const alpha = material?.transparent ? diffuseColor.a : 1;
            return vec4(0, 0, 0, alpha) as unknown as Node;
        }
        const weight = compositedModel(material) === 'physical' ? physicalIndirectDiffuseWeight() : (diffuseColor.rgb as unknown as TslNode);
        // `ambientOcclusion` holds the material's AO only when `setupAmbientOcclusion` found some; otherwise it's never written.
        const hasAmbientOcclusion = (builder.context as { ambientOcclusion?: unknown }).ambientOcclusion !== undefined;
        const occluded = hasAmbientOcclusion ? weight.mul(ambientOcclusion) : weight;
        return vec4(occluded, 1) as unknown as Node;
    }
}

/** The weight of indirect diffuse light per pixel, for `mrt({ ..., giAlbedo })`. */
export const giAlbedo = new GIAlbedoNode();

/**
 * Where the composite finds the GI light: a started `GIVolume`, or a
 * `ThreeGI` (its light changes when the shading module changes shape, so
 * it's read each render).
 */
export interface CompositeGISource {
    readonly light: GILight;
}

/**
 * The `giAlbedo` target's blending and clear: a transparent surface dims the
 * weight behind it as its blending dims the colour (file comment), and
 * pixels with no surface keep weight 0 whatever the background colour (three
 * clears every MRT target to the scene background otherwise).
 */
function configureWeightOutput(outputs: NonNullable<ReturnType<PassNode['getMRT']>>): void {
    outputs.setBlendMode(GI_ALBEDO_OUTPUT, new BlendMode(MaterialBlending));
    outputs.setClearColor(GI_ALBEDO_OUTPUT, 0x000000, 0);
}

/** The GTAO radius on the GI term, metres (view space): DECISIONS 2026-11-05, curtained Sponza option A. */
export const DEFAULT_GI_OCCLUSION_RADIUS = 1;

/** The composite's own GTAO on the GI term. Anything left out keeps three's GTAO default, except the radius. */
export interface GIOcclusionSettings {
    /** Metres, in view space. Default {@link DEFAULT_GI_OCCLUSION_RADIUS}. */
    readonly radius?: number;
    /** GTAO's resolution against the pass's (three's `resolutionScale`; 0.5 is half). Default 1. */
    readonly resolutionScale?: number;
    /** Samples per pixel (three's default 16). */
    readonly samples?: number;
}

/**
 * An AO texture the pipeline already makes, one value per pass pixel in red,
 * from this pass's depth and normal: three's GTAONode fits.
 */
export interface GIOcclusionSource {
    getTextureNode(): TextureNode;
}

export interface DDGICompositeOptions {
    /**
     * Short-range occlusion on the GI term (file comment). Settings for the
     * composite's own GTAO (the default: a 1 m radius), `false` for none, or
     * `{ source }` to use an AO texture the pipeline already makes.
     */
    readonly occlusion?: false | GIOcclusionSettings | { readonly source: GIOcclusionSource };
}

/** 1 / π: three's Lambert BRDF factor (`BRDF_Lambert`), which turns irradiance into reflected radiance. */
const RECIPROCAL_PI = 1 / Math.PI;

/**
 * Adds GI to a scene pass's colour: `output + giAlbedo × E / π` per pixel,
 * E from the GI light's shading module at the pixel's world position (from
 * depth) and normal. Shaped like three's SSGI and GTAO nodes, but it needs
 * no render target of its own: it's a pure function of the pass textures,
 * so the pipeline's own full-screen quad evaluates it.
 *
 * Marks the pass at construction (composite-mark.ts), keeping any context
 * the pass already has. The pass's MRT must have `giAlbedo` and `normal`
 * outputs (file comment); `setup` says so if it doesn't.
 *
 * The shader is built for the shading module of the light at build time. If
 * the module changes shape (`ThreeGI` then swaps its light), set
 * `pipeline.needsUpdate = true` to rebuild it; until then the composite
 * keeps the old module's last values and warns once.
 */
export class DDGICompositeNode extends Node {
    static get type(): string {
        return 'DDGICompositeNode';
    }

    readonly scenePass: PassNode;
    readonly source: CompositeGISource;
    /**
     * The composite's own GTAO on the GI term, when it makes one: change its
     * `radius.value` or `samples.value` live. Null when occlusion is off or
     * comes from the pipeline (`occlusion: { source }`).
     */
    readonly occlusionNode: GTAONode | null;
    /** Where the occlusion is read from, or null for none. */
    private readonly occlusionSource: GIOcclusionSource | null;
    private moduleNodes: ShadingModuleNodes | null = null;
    private warnedStale = false;
    private readonly intensity = uniform(1);
    private readonly computeCoverage = uniform(1);

    constructor(scenePass: PassNode, source: CompositeGISource, options: DDGICompositeOptions = {}) {
        super('vec4');
        this.scenePass = scenePass;
        this.source = source;
        const occlusion = options.occlusion ?? defaultOcclusion(scenePass);
        if (occlusion === false) {
            this.occlusionNode = null;
            this.occlusionSource = null;
        } else if ('source' in occlusion) {
            this.occlusionNode = null;
            this.occlusionSource = occlusion.source;
        } else {
            this.occlusionNode = makeOcclusion(scenePass, occlusion);
            this.occlusionSource = this.occlusionNode;
        }
        // Each render of the pipeline's quad: the module's values and the light's settings.
        this.updateType = NodeUpdateType.RENDER;
        markCompositePass(scenePass as unknown as Parameters<typeof markCompositePass>[0]);
        // Before the pass compiles, if its MRT is already set (the usual order); `setup` checks again.
        const outputs = scenePass.getMRT();
        if (outputs?.has(GI_ALBEDO_OUTPUT)) configureWeightOutput(outputs);
    }

    update(): boolean | undefined {
        const light = this.source.light;
        this.intensity.value = light.intensity;
        this.computeCoverage.value = light.computeCoverage ? 1 : 0;
        const nodes = this.moduleNodes;
        if (!nodes) return undefined;
        if (nodes.fits(light.shading)) {
            nodes.refresh();
        } else if (!this.warnedStale) {
            this.warnedStale = true;
            console.warn('@pmndrs/ddgi/three: the GI shading module changed shape; set pipeline.needsUpdate = true to rebuild the GI composite.');
        }
        return undefined;
    }

    setup(builder: NodeBuilder): Node {
        void builder;
        const scenePass = this.scenePass;
        const outputs = scenePass.getMRT();
        if (!outputs || !outputs.has(GI_ALBEDO_OUTPUT) || !outputs.has(GI_NORMAL_OUTPUT)) {
            throw new Error(
                `@pmndrs/ddgi/three: the GI composite needs the scene pass to write '${GI_ALBEDO_OUTPUT}' and '${GI_NORMAL_OUTPUT}': ` +
                    `scenePass.setMRT(mrt({ output, ${GI_ALBEDO_OUTPUT}: giAlbedo, ${GI_NORMAL_OUTPUT}: packNormalToRGB(normalView) })).`,
            );
        }
        configureWeightOutput(outputs);

        const light = this.source.light;
        this.intensity.value = light.intensity;
        this.computeCoverage.value = light.computeCoverage ? 1 : 0;
        const moduleNodes = new ShadingModuleNodes(light.shading);
        this.moduleNodes = moduleNodes;
        this.warnedStale = false;
        const entry = sharedEntry(light.shading);

        const camera = scenePass.camera;
        // By reference: three's uniforms read the matrices each render, after the pass has updated them.
        const projectionMatrixInverse = uniform(camera.projectionMatrixInverse);
        const cameraWorldMatrix = uniform(camera.matrixWorld);
        const colorTexture = scenePass.getTextureNode('output');
        const weightTexture = scenePass.getTextureNode(GI_ALBEDO_OUTPUT);
        const normalTexture = scenePass.getTextureNode(GI_NORMAL_OUTPUT);
        const depthTexture = scenePass.getTextureNode('depth');
        const occlusionTexture = this.occlusionSource?.getTextureNode() ?? null;

        const composite = Fn(() => {
            const screen = uv();
            const sceneColor = colorTexture.sample(screen).toVar('threeGiSceneColor');
            const result = vec4(sceneColor).toVar('threeGiComposited');
            const weight = weightTexture.sample(screen).rgb.toVar('threeGiWeight');
            const depth = depthTexture.sample(screen).r;
            // Background pixels and surfaces that shade GI themselves (weight 0) skip the probe lookup.
            const lit = isBackgroundDepth(depth).not().and(weight.r.add(weight.g).add(weight.b).greaterThan(0));
            If(lit, () => {
                const viewPosition = getViewPosition(screen, depth, projectionMatrixInverse);
                const worldPosition = cameraWorldMatrix.mul(vec4(viewPosition, 1)).xyz.toVar('threeGiPosition');
                // The pass stores the view-space shading normal; the camera's world matrix turns it into world space.
                const viewNormal = unpackRGBToNormal(normalTexture.sample(screen).rgb);
                const worldNormal = cameraWorldMatrix.mul(vec4(viewNormal, 0)).xyz.normalize();
                const eye = cameraWorldMatrix.mul(vec4(0, 0, 0, 1)).xyz;
                // From the eye towards the point, as the shading module expects (gi-light.ts does the same).
                const viewDirection = worldPosition.sub(eye).normalize();
                const lighting = (
                    entry({
                        position: worldPosition,
                        normal: worldNormal,
                        viewDirection,
                        screenUV: screen,
                        computeCoverage: this.computeCoverage,
                        ...moduleNodes.resources(),
                    }) as unknown as ReturnType<typeof vec4>
                ).toVar('threeGiLighting');
                // The per-fragment path adds GI only where its coverage claims the probe slot.
                If(lighting.w.greaterThan(0), () => {
                    let irradiance = lighting.xyz.mul(this.intensity);
                    // Short-range occlusion on the GI term only (file comment).
                    if (occlusionTexture) irradiance = irradiance.mul(occlusionTexture.sample(screen).r);
                    const indirectDiffuse = weight.mul(irradiance).mul(RECIPROCAL_PI);
                    result.assign(vec4(sceneColor.rgb.add(indirectDiffuse), sceneColor.a));
                });
            });
            return result;
        });
        return composite() as unknown as Node;
    }
}

/**
 * The default occlusion: the composite's own GTAO at 1 m, unless the pass
 * already applies AO to its materials (a context with `getAO`), which
 * `giAlbedo` then carries (file comment).
 */
function defaultOcclusion(scenePass: PassNode): false | GIOcclusionSettings {
    const context = scenePass.contextNode;
    const passAppliesAO = context !== null && typeof flowContextData(context).getAO === 'function';
    return passAppliesAO ? false : {};
}

/** three's GTAO on the pass's depth and its packed view-space normal (the `normal` output). */
function makeOcclusion(scenePass: PassNode, settings: GIOcclusionSettings): GTAONode {
    const normalTexture = scenePass.getTextureNode(GI_NORMAL_OUTPUT);
    const viewNormal = sample((coordinate) => unpackRGBToNormal(normalTexture.sample(coordinate).rgb));
    const node = ao(scenePass.getTextureNode('depth'), viewNormal, scenePass.camera);
    node.radius.value = settings.radius ?? DEFAULT_GI_OCCLUSION_RADIUS;
    if (settings.resolutionScale !== undefined) node.resolutionScale = settings.resolutionScale;
    if (settings.samples !== undefined) node.samples.value = settings.samples;
    return node;
}

/**
 * GI on a scene pass, as a node for `pipeline.outputNode`, like three's
 * `ssgi(...)` and `ao(...)`: the pass's colour plus `giAlbedo × E / π`
 * (file comment). `source` is the started `GIVolume` (or a `ThreeGI`): the
 * composite reads its GI light. Options: `occlusion` (GTAO on the GI term,
 * 1 m by default).
 */
export function ddgi(scenePass: PassNode, source: CompositeGISource, options: DDGICompositeOptions = {}): DDGICompositeNode {
    return new DDGICompositeNode(scenePass, source, options);
}
