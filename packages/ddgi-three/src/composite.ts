/**
 * GI as an additive composite on three's post-processing pipeline (T1.14,
 * `docs/design/three-composite.md`): the fast path for dense scenes, where
 * GI shaded per fragment runs many times per pixel (Sponza's sub-pixel
 * triangles: +12 ms at 1440p). Built like three r187's SSGI example:
 *
 * ```ts
 * await volume.start(renderer, scene);                 // the simple path (ddgi-volume.ts)
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
 * **Short-range occlusion on GI, opt-in** (DECISIONS 2026-11-18, which
 * supersedes 2026-11-06's default): probes 1 m apart can't follow light that
 * changes inside small folds (Sponza's curtains read +23% against Cycles
 * without it). The composite can run three's own GTAO (`ao()`, GTAONode) on
 * the pass's depth and normal, with a 1 m radius, and multiply the GI term
 * by it, and only the GI term: direct light and the environment's light are
 * untouched. Its size follows the pass, and its `resolutionScale` is
 * relative to the pass.
 *
 * **The intended pipeline is `@pmndrs/upscaler`** (DECISIONS 2026-11-06,
 * the T1.14 default path): the scene pass and the composite at the
 * upscaler's render resolution, the upscaler bringing the result to the
 * display.
 *
 * **By default there is no occlusion, on any pass.** Until 2026-11-18 a
 * reduced (upscaled) pass got GTAO at half its resolution by default. With
 * M4's textures that darkened the scenes with little to occlude (against
 * Cycles: the textured furnace about 9% dark, −0.9% without; the furnished
 * room 14-17% dark, 5-6% without) to help the one it was made for (Sponza
 * +8-12% with it, +22-26% without, now a tuning-pass item). Hosts with
 * fold-heavy scenes opt in. Under the upscaler the measured setting is
 * `occlusion: { resolutionScale: 0.5 }`: 0.73 ms for showing GI on Sponza
 * at 2560 × 1440 (Performance), against 0.41 ms without (RTX 3060 laptop,
 * T1.14). On a full-resolution pass GTAO costs several times the
 * composite's 1.3 ms (8.0 ms at full resolution, 3.2 ms at half).
 *
 * The option `occlusion`: settings for the composite's GTAO, `false` (the
 * same as leaving it out), or an AO texture the pipeline already makes
 * (`{ source }`, a GTAONode on this pass's depth and normal) instead of
 * running a second GTAO. If the pass already applies AO to its materials (a
 * context with `getAO`, three's `builtinAOContext`), `giAlbedo` carries that
 * AO already, so adding the composite's occlusion as well darkens GI twice.
 */

import { BlendMode, MaterialBlending, Node, NodeUpdateType, PhysicalLightingModel, type Camera, type NodeBuilder, type PassNode, type TextureNode } from 'three/webgpu';
import GTAONode from 'three/addons/tsl/display/GTAONode.js';
import {
    DFGLUT,
    Fn,
    If,
    ambientOcclusion,
    diffuseColor,
    diffuseContribution,
    getViewPosition,
    isBackgroundDepth,
    nodeObject,
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
import { compositeTakesGI, compositedModel, flowContextData, hasCompositeMark, markCompositePass, type MarkablePass } from './composite-mark.ts';
import { GI_COMPOSITE_WON, ShadingModuleNodes, sharedEntry, type GILight } from './gi-light.ts';

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
        const lightingWeight = compositedModel(material) === 'physical' ? physicalIndirectDiffuseWeight() : (diffuseColor.rgb as unknown as TslNode);
        // Zero where another probe light won the slot: that light is shaded per fragment (gi-light.ts, GI_COMPOSITE_WON).
        const weight = lightingWeight.mul(GI_COMPOSITE_WON) as unknown as TslNode;
        // `ambientOcclusion` holds the material's AO only when `setupAmbientOcclusion` found some; otherwise it's never written.
        const hasAmbientOcclusion = (builder.context as { ambientOcclusion?: unknown }).ambientOcclusion !== undefined;
        const occluded = hasAmbientOcclusion ? weight.mul(ambientOcclusion) : weight;
        return vec4(occluded, 1) as unknown as Node;
    }
}

/** The weight of indirect diffuse light per pixel, for `mrt({ ..., giAlbedo })`. */
export const giAlbedo = new GIAlbedoNode();

/**
 * Where the composite finds the GI light: a started `DDGIVolume`, or a
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
    /** GTAO's resolution against the scene pass's (0.5 is half the pass's pixels, whatever the pass's own scale). Default 1. */
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
     * Short-range occlusion on the GI term (file comment). None by default
     * (DECISIONS 2026-11-18). Settings for the composite's own GTAO (a 1 m
     * radius unless set), `false` for none, or `{ source }` to use an AO
     * texture the pipeline already makes.
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
 * the pass already has. If `scenePass.contextNode` is replaced later, the
 * mark would be lost and GI counted twice (per fragment and composited), so
 * the node checks at build time and on every render, merges the mark into
 * the new context, and warns once. A replacement made after the pipeline's
 * first frame still shows for one frame: the pass draws before the
 * composite's update sees it. Set the pass's context before `ddgi()`. The pass's MRT must have `giAlbedo` and `normal`
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
     * `radius.value` or `samples.value` live. Null when occlusion is off
     * (the default) or comes from the pipeline (`occlusion: { source }`).
     */
    get occlusionNode(): GTAONode | null {
        return this.ownOcclusion;
    }

    private readonly ownOcclusion: GTAONode | null = null;
    /** Where the occlusion is read from, or null for none. */
    private readonly occlusionSource: GIOcclusionSource | null = null;
    private moduleNodes: ShadingModuleNodes | null = null;
    private warnedStale = false;
    /** The pass context the mark was last checked on. */
    private markedContext: PassNode['contextNode'] = null;
    private warnedLostMark = false;
    private readonly intensity = uniform(1);
    private readonly computeCoverage = uniform(1);

    constructor(scenePass: PassNode, source: CompositeGISource, options: DDGICompositeOptions = {}) {
        super('vec4');
        this.scenePass = scenePass;
        this.source = source;
        // Leaving `occlusion` out is the same as `false` (the file comment).
        const occlusion = options.occlusion ?? false;
        if (occlusion !== false) {
            if ('source' in occlusion) {
                this.occlusionSource = occlusion.source;
            } else {
                this.ownOcclusion = makeOcclusion(scenePass, occlusion);
                this.occlusionSource = this.ownOcclusion;
            }
        }
        // Each render of the pipeline's quad: the module's values and the light's settings.
        this.updateType = NodeUpdateType.RENDER;
        markCompositePass(scenePass as unknown as MarkablePass);
        this.markedContext = scenePass.contextNode;
        if (occlusion !== false && 'source' in occlusion && passAppliesAO(scenePass)) {
            console.warn(
                '@pmndrs/gi/three: ddgi() got an occlusion source, and its pass already applies AO to its materials (builtinAOContext), which giAlbedo carries: GI gets AO twice. Use one of them.',
            );
        }
        // Before the pass compiles, if its MRT is already set (the usual order); `setup` checks again.
        const outputs = scenePass.getMRT();
        if (outputs?.has(GI_ALBEDO_OUTPUT)) configureWeightOutput(outputs);
    }

    /** Merges the mark back into the pass's context if the context was replaced (the class comment). */
    private keepMark(): void {
        const scenePass = this.scenePass;
        if (scenePass.contextNode === this.markedContext) return;
        if (!scenePass.contextNode || !hasCompositeMark(flowContextData(scenePass.contextNode))) {
            markCompositePass(scenePass as unknown as MarkablePass);
            if (!this.warnedLostMark) {
                this.warnedLostMark = true;
                console.warn('@pmndrs/gi/three: the scene pass got a new contextNode after ddgi(); the GI composite merged its mark into it again. Set the pass context before calling ddgi().');
            }
        }
        this.markedContext = scenePass.contextNode;
    }

    update(): boolean | undefined {
        this.keepMark();
        const light = this.source.light;
        this.intensity.value = light.intensity;
        this.computeCoverage.value = light.computeCoverage ? 1 : 0;
        const nodes = this.moduleNodes;
        if (!nodes) return undefined;
        if (nodes.fits(light.shading)) {
            nodes.refresh();
        } else if (!this.warnedStale) {
            this.warnedStale = true;
            console.warn('@pmndrs/gi/three: the GI shading module changed shape; set pipeline.needsUpdate = true to rebuild the GI composite.');
        }
        return undefined;
    }

    setup(builder: NodeBuilder): Node {
        void builder;
        this.keepMark();
        const scenePass = this.scenePass;
        const outputs = scenePass.getMRT();
        if (!outputs || !outputs.has(GI_ALBEDO_OUTPUT) || !outputs.has(GI_NORMAL_OUTPUT)) {
            throw new Error(
                `@pmndrs/gi/three: the GI composite needs the scene pass to write '${GI_ALBEDO_OUTPUT}' and '${GI_NORMAL_OUTPUT}': ` +
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

/** Whether the pass's context applies AO to its materials (three's `builtinAOContext`), which `giAlbedo` then carries. */
function passAppliesAO(scenePass: PassNode): boolean {
    const context = scenePass.contextNode;
    return context !== null && typeof flowContextData(context).getAO === 'function';
}

/**
 * three's GTAO, sized from the scene pass instead of the drawing buffer.
 * GTAONode sizes itself each frame from `renderer.getDrawingBufferSize()`
 * (its `updateBefore`), times its own `resolutionScale`. A pass drawn at a
 * reduced size (`scenePass.setResolutionScale`, as under an upscaler) would
 * then get AO at display resolution, for depth and normals that only have
 * the pass's. Here the size is the pass's (drawing buffer × the pass's scale),
 * and GTAO's own `resolutionScale` is relative to that: 0.5 is always half the
 * pass's pixels, whatever the pass's scale.
 */
class PassSizedGTAONode extends GTAONode {
    constructor(
        depthNode: Node,
        normalNode: Node,
        camera: Camera,
        private readonly scenePass: PassNode,
    ) {
        super(depthNode, normalNode, camera);
    }

    setSize(width: number, height: number): void {
        const passScale = this.scenePass.getResolutionScale();
        super.setSize(Math.max(1, Math.round(width * passScale)), Math.max(1, Math.round(height * passScale)));
    }
}

/** three's GTAO on the pass's depth and its packed view-space normal (the `normal` output), sized from the pass. */
function makeOcclusion(scenePass: PassNode, settings: GIOcclusionSettings): GTAONode {
    const normalTexture = scenePass.getTextureNode(GI_NORMAL_OUTPUT);
    const viewNormal = sample((coordinate) => unpackRGBToNormal(normalTexture.sample(coordinate).rgb));
    const node = new PassSizedGTAONode(nodeObject(scenePass.getTextureNode('depth')) as unknown as Node, nodeObject(viewNormal) as unknown as Node, scenePass.camera, scenePass);
    node.radius.value = settings.radius ?? DEFAULT_GI_OCCLUSION_RADIUS;
    if (settings.resolutionScale !== undefined) node.resolutionScale = settings.resolutionScale;
    if (settings.samples !== undefined) node.samples.value = settings.samples;
    return node;
}

/**
 * GI on a scene pass, as a node for `pipeline.outputNode`, like three's
 * `ssgi(...)` and `ao(...)`: the pass's colour plus `giAlbedo × E / π`
 * (file comment). `source` is the started `DDGIVolume` (or a `ThreeGI`): the
 * composite reads its GI light. Options: `occlusion` (GTAO on the GI term,
 * off by default, a 1 m radius when on).
 */
export function ddgi(scenePass: PassNode, source: CompositeGISource, options: DDGICompositeOptions = {}): DDGICompositeNode {
    return new DDGICompositeNode(scenePass, source, options);
}
