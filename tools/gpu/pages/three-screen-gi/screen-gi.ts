/**
 * The pieces of a "GI once per pixel" path through three's forward renderer,
 * built from outside the adapter so their cost can be measured before anyone
 * changes `packages/ddgi-three` (the frame breakdown's proposal in DECISIONS).
 *
 * 1. Depth and normals per pixel, from one of:
 *    - the main pass itself, with three's MRT (`mrt({ output, normal })`):
 *      `NORMAL_DEPTH_NODE` written to a second colour target;
 *    - a separate prepass (`NormalDepthMaterial` on every opaque mesh), which
 *      re-rasterises the whole scene.
 *    Both store the world normal in rgb and the view depth (metres along the
 *    view axis) in alpha of an rgba16f target. A real implementation would
 *    read the depth buffer; half floats keep about 1 cm at 20 m, plenty for
 *    a cost test (DDGI's own biases are 10 cm).
 * 2. `ScreenGIPass`: a full-screen pass that rebuilds each pixel's world
 *    position from that depth and calls the technique's shading module
 *    (`giIrradiance`, and `giCoverage` when the adapter's light computes it)
 *    once per pixel, through one `wgslFn` built the adapter's way
 *    (`entryWgsl`), into an rgba16f irradiance target (E in rgb, coverage in
 *    alpha).
 * 3. `ScreenGILight`: a light whose node reads that target by `screenUV`
 *    instead of evaluating the probes, and fills three's light-probe slot
 *    exactly as the adapter's `GILight` does (priority, coverage, and the
 *    environment-diffuse fade).
 */

import { Light, Matrix4, TangentSpaceNormalMap, Vector2, Vector3, Vector4, type Material, type NormalMapTypes, type Texture } from 'three';
import { AnalyticLightNode, MeshBasicNodeMaterial, NodeMaterial, QuadMesh, type ExternalTexture, type Node, type NodeBuilder, type WebGPURenderer } from 'three/webgpu';
import {
    Fn,
    If,
    INFINITY,
    float,
    materialNormal,
    normalWorld,
    positionView,
    renderGroup,
    screenUV,
    texture,
    texture3D,
    uniform,
    vec2,
    vec3,
    vec4,
    wgsl,
    wgslFn,
} from 'three/tsl';
import { entryWgsl, wrapGpuTexture, type ShadingModule } from '../../../../apps/viewer/src/ddgi/library';

type ShadingParameter = ShadingModule['parameters'][number];
type TextureParameter = Extract<ShadingParameter, { kind: 'texture' }>;
type SamplerParameter = Extract<ShadingParameter, { kind: 'sampler' }>;

/**
 * What the depth-and-normal target holds per pixel: the shading normal in world space (the
 * one three's lights see, normal maps included) and the view depth. `positionView.z` is
 * negative in front of the camera.
 */
export const NORMAL_DEPTH_NODE = vec4(normalWorld, positionView.z.negate());

// -- the prepass material ----------------------------------------------------------------------------

/**
 * A depth-and-normal prepass material. A basic material (no lighting) whose normal comes
 * from the material's normal map like a lit material's does: `MeshBasicNodeMaterial`
 * normally uses the geometry normal, so `setupNormal` is the base class's again.
 */
class NormalDepthMaterial extends MeshBasicNodeMaterial {
    setupNormal(): Node<'vec3'> {
        return materialNormal as unknown as Node<'vec3'>;
    }
}

/** The prepass stand-in for one source material: same face culling, cut-out and normal map. */
export function normalDepthMaterial(source: Material): MeshBasicNodeMaterial {
    const material = new NormalDepthMaterial();
    material.name = `normal and depth (${source.name})`;
    material.side = source.side;
    const lit = source as Material & { normalMap?: Texture | null; normalScale?: Vector2; normalMapType?: NormalMapTypes; map?: Texture | null };
    if (lit.normalMap) {
        // MeshBasicMaterial has no normal map of its own; MaterialNode reads these properties by name.
        const target = material as unknown as { normalMap: Texture; normalScale: Vector2; normalMapType: NormalMapTypes };
        target.normalMap = lit.normalMap;
        target.normalScale = lit.normalScale ?? new Vector2(1, 1);
        target.normalMapType = lit.normalMapType ?? TangentSpaceNormalMap;
    }
    if (source.alphaTest > 0) {
        material.alphaTest = source.alphaTest;
        if (lit.map) material.opacityNode = texture(lit.map).a.mul(source.opacity);
        else material.opacity = source.opacity;
    }
    material.outputNode = NORMAL_DEPTH_NODE;
    return material;
}

// -- the screen pass -----------------------------------------------------------------------------------

/** A module texture parameter wrapped for three, with its sampler's filter and wrap (as `GILight` does). */
function wrapTexture(gpuTexture: GPUTexture, parameter: TextureParameter, sampler: SamplerParameter | undefined): ExternalTexture {
    return wrapGpuTexture(gpuTexture, { dimension: parameter.dimension, sampleType: parameter.sampleType, filtering: sampler?.filtering, addressMode: sampler?.addressMode });
}

/**
 * The module's resources as three nodes, the way `GILightNode.buildModule` makes them. The
 * page freezes the field, so the textures never reallocate; `refresh` copies the uniforms.
 */
class ModuleResources {
    readonly nodes: Record<string, Node> = {};
    private readonly uniforms = new Map<string, { value: Vector4 }>();

    constructor(private readonly shading: ShadingModule) {
        const samplers = new Map<string, SamplerParameter>();
        for (const parameter of shading.parameters) if (parameter.kind === 'sampler') samplers.set(parameter.texture, parameter);
        const textureNodes = new Map<string, Node>();
        for (const parameter of shading.parameters) {
            if (parameter.kind === 'texture') {
                const wrapped = wrapTexture(parameter.texture(), parameter, samplers.get(parameter.name));
                textureNodes.set(parameter.name, (parameter.dimension === '3d' ? texture3D(wrapped) : texture(wrapped)) as unknown as Node);
            } else if (parameter.kind === 'vec4f') {
                const node = uniform(new Vector4().fromArray(parameter.value())).setGroup(renderGroup);
                this.uniforms.set(parameter.name, node as unknown as { value: Vector4 });
                textureNodes.set(parameter.name, node as unknown as Node);
            }
        }
        for (const parameter of shading.parameters) {
            // A sampler parameter is fed from its texture's node, which carries the sampler (NOTES).
            const source = parameter.kind === 'sampler' ? parameter.texture : parameter.name;
            this.nodes[parameter.name] = textureNodes.get(source)!;
        }
    }

    refresh(): void {
        for (const parameter of this.shading.parameters) {
            if (parameter.kind === 'vec4f') this.uniforms.get(parameter.name)!.value.fromArray(parameter.value());
        }
    }
}

/** Farther than this (metres) is not a surface: below the half-float maximum, far beyond any scene. */
const MAX_SURFACE_DEPTH = 60_000;

/**
 * Evaluates GI once per pixel from a depth-and-normal texture. One instance per input
 * texture, so switching inputs never rebinds a live pipeline.
 */
export class ScreenGIPass {
    private readonly quad: QuadMesh;
    private readonly resources: ModuleResources;
    private readonly inverseProjection = uniform(new Matrix4());
    private readonly cameraWorld = uniform(new Matrix4());
    private readonly eye = uniform(new Vector3());
    private readonly computeCoverage = uniform(1);

    constructor(shading: ShadingModule, normalDepth: Texture) {
        this.resources = new ModuleResources(shading);
        // Its own copy of the module: this shader contains nothing else that includes it.
        const entry = wgslFn(entryWgsl(shading), [wgsl(shading.code)]);
        const input = texture(normalDepth, screenUV);
        const shade = Fn(() => {
            const sample = input.toVar('screenGiInput');
            const result = vec4(0).toVar('screenGiResult');
            // Depth 0 is the cleared background; the sky dome (225 km away) overflows half floats to
            // infinity. Neither is a surface, so neither gets GI.
            If(sample.w.greaterThan(0).and(sample.w.lessThan(MAX_SURFACE_DEPTH)), () => {
                // screenUV runs top-left to bottom-right; NDC y points up.
                const ndc = vec2(screenUV.x.mul(2).sub(1), float(1).sub(screenUV.y.mul(2)));
                const onFarPlane = this.inverseProjection.mul(vec4(ndc, 1, 1));
                const viewRay = onFarPlane.xyz.div(onFarPlane.w);
                // Scale the ray so its view-axis length is the stored depth.
                const viewPosition = viewRay.mul(sample.w.div(viewRay.z.negate()));
                const worldPosition = this.cameraWorld.mul(vec4(viewPosition, 1)).xyz.toVar('screenGiPosition');
                const viewDirection = worldPosition.sub(this.eye).normalize();
                const lighting = entry({
                    position: worldPosition,
                    normal: sample.xyz.normalize(),
                    viewDirection,
                    screenUV,
                    computeCoverage: this.computeCoverage,
                    ...this.resources.nodes,
                }) as unknown as Node<'vec4'>;
                result.assign(lighting);
            });
            return result;
        });
        const material = new NodeMaterial();
        material.name = 'screen GI';
        material.fragmentNode = shade();
        this.quad = new QuadMesh(material);
    }

    /** Draws into whatever target the renderer has set. */
    render(renderer: WebGPURenderer, camera: { projectionMatrixInverse: Matrix4; matrixWorld: Matrix4 }, computeCoverage: boolean): void {
        this.resources.refresh();
        this.inverseProjection.value.copy(camera.projectionMatrixInverse);
        this.cameraWorld.value.copy(camera.matrixWorld);
        this.eye.value.setFromMatrixPosition(camera.matrixWorld);
        this.computeCoverage.value = computeCoverage ? 1 : 0;
        this.quad.render(renderer);
    }
}

// -- the light that reads it -------------------------------------------------------------------------

/** The same priority as the adapter's light. */
const SCREEN_GI_PRIORITY = 1;

export class ScreenGILight extends Light {
    readonly isScreenGILight = true;

    constructor(readonly irradiance: Texture) {
        super(0xffffff, 1);
        (this as { type: string }).type = 'ScreenGILight';
    }
}

interface LightingContext {
    lightProbeIrradiance?: Node<'vec3'>;
    lightProbePriority?: Node<'float'>;
    screenGiCoverage?: Node<'float'>;
    iblIrradiance?: Node<'vec3'>;
    lightingModel: { indirect(builder: NodeBuilder): void };
}

const patchedLightingModels = new WeakSet<object>();

class ScreenGILightNode extends AnalyticLightNode<ScreenGILight> {
    static get type(): string {
        return 'ScreenGILightNode';
    }

    private readonly priority = uniform(SCREEN_GI_PRIORITY).setGroup(renderGroup);

    setup(builder: NodeBuilder): Node | undefined {
        const light = this.light!;
        const context = builder.context as unknown as LightingContext;
        context.lightProbeIrradiance ||= vec3().toVar('lightProbeIrradiance');
        context.lightProbePriority ||= INFINITY.negate().toVar('lightProbePriority');
        context.screenGiCoverage ||= float(0).toVar('screenGiCoverage');
        // The pixel's GI, evaluated once by the screen pass: E in rgb, coverage in alpha.
        const lighting = (texture(light.irradiance, screenUV) as unknown as Node<'vec4'>).toVar('screenGiLighting');
        const claims = this.priority.greaterThanEqual(context.lightProbePriority!).and(lighting.w.greaterThan(0));
        If(claims, () => {
            context.lightProbeIrradiance!.assign(lighting.xyz);
            context.lightProbePriority!.assign(this.priority);
            context.screenGiCoverage!.assign(lighting.w);
        });
        // The adapter's environment-diffuse fade (gi-light.ts, `replaceEnvironmentDiffuse`), so the pictures match.
        const model = context.lightingModel;
        if (model && !patchedLightingModels.has(model)) {
            patchedLightingModels.add(model);
            const originalIndirect = model.indirect.bind(model);
            model.indirect = (indirectBuilder: NodeBuilder) => {
                const current = indirectBuilder.context as unknown as LightingContext;
                if (current.iblIrradiance && current.screenGiCoverage) current.iblIrradiance.mulAssign(float(1).sub(current.screenGiCoverage));
                originalIndirect(indirectBuilder);
            };
        }
        return undefined;
    }
}

ScreenGILight.registerNode(ScreenGILightNode);
