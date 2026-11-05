/**
 * GI as an additive composite on three's post-processing pipeline (T1.14,
 * `docs/design/three-composite.md`): the fast path for dense scenes, where
 * GI shaded per fragment runs many times per pixel (Sponza's sub-pixel
 * triangles: +12 ms at 1440p). Built like three r187's SSGI example:
 *
 * ```ts
 * const pipeline = new RenderPipeline(renderer);
 * const scenePass = pass(scene, camera);
 * scenePass.setMRT(mrt({ output, giAlbedo, normal: packNormalToRGB(normalView) }));
 * pipeline.outputNode = ddgiComposite(scenePass, threeGI);
 * ```
 *
 * The names (`giAlbedo`, `ddgiComposite`) are working names until Dennis
 * confirms them, so nothing here is exported from the package yet.
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
 */

import { BlendMode, MaterialBlending, Node, NodeUpdateType, PhysicalLightingModel, type NodeBuilder, type PassNode } from 'three/webgpu';
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
    specularColor,
    specularF90,
    uniform,
    unpackRGBToNormal,
    uv,
    vec3,
    vec4,
} from 'three/tsl';
import { compositeTakesGI, compositedModel, markCompositePass } from './composite-mark.ts';
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

/** Where the composite finds the GI light: `ThreeGI` (its light changes when the shading module changes shape). */
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
    private moduleNodes: ShadingModuleNodes | null = null;
    private warnedStale = false;
    private readonly intensity = uniform(1);
    private readonly computeCoverage = uniform(1);

    constructor(scenePass: PassNode, source: CompositeGISource) {
        super('vec4');
        this.scenePass = scenePass;
        this.source = source;
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
                    const indirectDiffuse = weight.mul(lighting.xyz).mul(this.intensity.mul(RECIPROCAL_PI));
                    result.assign(vec4(sceneColor.rgb.add(indirectDiffuse), sceneColor.a));
                });
            });
            return result;
        });
        return composite() as unknown as Node;
    }
}

/**
 * The composite as a node for `pipeline.outputNode`, like three's `ssgi(...)`.
 * Working name, not exported from the package until the names are confirmed.
 */
export function ddgiComposite(scenePass: PassNode, source: CompositeGISource): DDGICompositeNode {
    return new DDGICompositeNode(scenePass, source);
}
