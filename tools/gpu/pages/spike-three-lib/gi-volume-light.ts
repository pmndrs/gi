/**
 * A three.js `Light` whose registered node adds GI irradiance from our own
 * textures. It is the shape the real adapter will have, written the way three's
 * own `LightProbeGrid` does it (examples/jsm/tsl/lighting/LightProbeGridNode.js):
 *
 * - The class extends `Light` and calls `registerNode`, so adding it to a scene
 *   lights every lit node material with no material changes.
 * - The node is an `AnalyticLightNode` with `updateType = RENDER`; its uniforms
 *   live in `renderGroup` and are refreshed in `update()` once per render call.
 * - It writes the light-probe slot (`builder.context.lightProbeIrradiance` and
 *   `lightProbePriority`), so where two probe-like lights cover a fragment only
 *   the one with the highest priority counts, instead of both adding up.
 *
 * The irradiance itself comes from a parameter-style WGSL function shaped like
 * the platform contract's `giIrradiance` (docs/api/platform.md): the world
 * position, normal, view direction and screen UV, then the resources in a fixed
 * order (textures, sampler, vec4f uniforms). Calling it through `wgslFn` is the
 * main thing this spike tests.
 */

import { AnalyticLightNode, Box3, Light, NodeUpdateType, Vector3, Vector4 } from 'three/webgpu';
import type { Node, NodeBuilder } from 'three/webgpu';
import {
    If,
    INFINITY,
    cameraPosition,
    float,
    normalWorld,
    positionWorld,
    renderGroup,
    screenUV,
    texture,
    texture3D,
    uniform,
    vec3,
    wgslFn,
} from 'three/tsl';
import { ATLAS_LAYER_USED, type WrappedGiTextures } from './gi-textures.ts';

/**
 * The contract-shaped function. `irradianceAtlas`, `volumeTexture`, `probeTable`,
 * `linearSampler` and the three `vec4f` values are the "resources", in the order
 * a `ShadingModule.parameters` list would give them.
 *
 * What it computes (so the test can know the answer):
 *   E = atlas[layer].rgb * table[0].x + volume.rgb * table[0].y
 * where `layer` is `settings.x`. Position selects texels inside the volume;
 * every texel of a texture holds the same value, so the result is constant.
 */
export const GI_IRRADIANCE_WGSL = /* wgsl */ `
fn giIrradiance(
    position: vec3f,
    normal: vec3f,
    viewDirection: vec3f,
    screenUV: vec2f,
    irradianceAtlas: texture_2d_array<f32>,
    volumeTexture: texture_3d<f32>,
    probeTable: texture_2d<f32>,
    linearSampler: sampler,
    volumeMin: vec4f,
    volumeSize: vec4f,
    settings: vec4f,
) -> vec3f {
    let uvw = (position - volumeMin.xyz) / volumeSize.xyz;
    let layer = i32(settings.x);
    let weights = textureLoad(probeTable, vec2i(0, 0), 0);
    let fromAtlas = textureSampleLevel(irradianceAtlas, linearSampler, uvw.xz, layer, 0.0).rgb;
    let fromVolume = textureSampleLevel(volumeTexture, linearSampler, uvw, 0.0).rgb;
    // viewDirection and screenUV are part of the contract signature; this
    // technique ignores them. Multiplying by a zero-valued term keeps them
    // "used" without changing the result, so a WGSL optimiser cannot hide a
    // binding mistake in how they are passed.
    let ignored = 0.0 * (dot(viewDirection, normal) + screenUV.x);
    return fromAtlas * weights.x + fromVolume * weights.y + vec3f(ignored);
}`;

/** The numbers the light node mirrors from the CPU each render. */
export interface GiVolumeLightOptions {
    /** World-space centre of the volume. */
    center: Vector3;
    /** Full size of the volume along each axis. */
    size: Vector3;
    intensity?: number;
    priority?: number;
    /**
     * Zero three's environment diffuse (`iblIrradiance`) inside the volume, so
     * `scene.environment` stops adding diffuse on top of our GI. r187 has no
     * switch for this (that is PR #35017), so the node patches the lighting
     * model's `indirect` step. See `suppressEnvironmentDiffuse` below.
     */
    suppressEnvironmentDiffuse?: boolean;
}

export class GiVolumeLight extends Light {
    readonly isGiVolumeLight = true;
    readonly boundingBox = new Box3();
    /** Where volumes overlap, the highest priority wins. Same meaning as `LightProbeGrid.priority`. */
    priority: number;
    suppressEnvironmentDiffuse: boolean;
    /** The wrapped GPU textures. Replace `textures.atlas` etc. through {@link setTextures} on reallocation. */
    textures: WrappedGiTextures;

    constructor(textures: WrappedGiTextures, options: GiVolumeLightOptions) {
        super(0xffffff, options.intensity ?? 1);
        this.textures = textures;
        this.priority = options.priority ?? 0;
        this.suppressEnvironmentDiffuse = options.suppressEnvironmentDiffuse ?? false;
        this.position.copy(options.center);
        this.boundingBox.setFromCenterAndSize(options.center, options.size);
    }
}

// `registerNode` stores the node class on the Light prototype; LightsNode makes
// one node per light from it (LightsNode.setupLightsNode).
GiVolumeLight.registerNode(createNodeClass());

/** The parts of `builder.context` the light-probe slot and the workaround use. */
interface LightingContext {
    lightProbeIrradiance?: Node<'vec3'>;
    lightProbePriority?: Node<'float'>;
    giCoverage?: Node<'float'>;
    iblIrradiance: Node<'vec3'>;
    lightingModel: { indirect(builder: NodeBuilder): void };
}

/** Lighting models whose `indirect` step we already wrapped, so we wrap each once. */
const patchedLightingModels = new WeakSet<object>();

function createNodeClass() {
    const giIrradiance = wgslFn(GI_IRRADIANCE_WGSL);

    return class GiVolumeLightNode extends AnalyticLightNode<GiVolumeLight> {
        static get type() {
            return 'GiVolumeLightNode';
        }

        private readonly volumeMin = uniform(new Vector4()).setGroup(renderGroup);
        private readonly volumeSize = uniform(new Vector4()).setGroup(renderGroup);
        private readonly settings = uniform(new Vector4()).setGroup(renderGroup);
        private readonly intensityUniform = uniform(1).setGroup(renderGroup);
        private readonly priorityUniform = uniform(0).setGroup(renderGroup);

        // One texture node per resource, created once and shared by every
        // material that builds this light. Each reads the light's *current*
        // ExternalTexture through `.value`, which update() refreshes.
        private readonly atlasNode;
        private readonly volumeNode;
        private readonly tableNode;

        constructor(light: GiVolumeLight | null = null) {
            super(light);
            this.updateType = NodeUpdateType.RENDER;
            const textures = light!.textures;
            this.atlasNode = texture(textures.atlas);
            this.volumeNode = texture3D(textures.volume);
            this.tableNode = texture(textures.table);
        }

        update(): boolean | undefined {
            const light = this.light!;
            const min = light.boundingBox.min;
            const size = light.boundingBox.getSize(new Vector3());
            this.volumeMin.value.set(min.x, min.y, min.z, 0);
            this.volumeSize.value.set(size.x, size.y, size.z, 0);
            this.settings.value.set(ATLAS_LAYER_USED, 0, 0, 0);
            this.intensityUniform.value = light.intensity;
            this.priorityUniform.value = light.priority;

            // Reallocation path: a new ExternalTexture replaces the old one. The
            // bind group is rebuilt by three when the node's value changes.
            if (this.atlasNode.value !== light.textures.atlas) this.atlasNode.value = light.textures.atlas;
            if (this.volumeNode.value !== light.textures.volume) this.volumeNode.value = light.textures.volume;
            if (this.tableNode.value !== light.textures.table) this.tableNode.value = light.textures.table;
            return undefined;
        }

        setup(builder: NodeBuilder): Node | undefined {
            const light = this.light!;
            const context = builder.context as unknown as LightingContext;

            // Inside the volume's box? The same test LightProbeGridNode makes.
            const min = this.volumeMin.xyz;
            const max = this.volumeMin.xyz.add(this.volumeSize.xyz);
            const inside = positionWorld
                .greaterThanEqual(min)
                .all()
                .and(positionWorld.lessThanEqual(max).all());

            // The light-probe slot. Created once per material build and shared by
            // every probe-like light; LightsNode adds it to `irradiance` after
            // all lights have run.
            context.lightProbeIrradiance ||= vec3().toVar('lightProbeIrradiance');
            context.lightProbePriority ||= INFINITY.negate().toVar('lightProbePriority');
            context.giCoverage ||= float(0).toVar('giCoverage');

            const prioritized = this.priorityUniform.greaterThanEqual(context.lightProbePriority);

            If(inside.and(prioritized), () => {
                const viewDirection = cameraPosition.sub(positionWorld).normalize();

                // The call under test: textures, a sampler and vec4f uniforms
                // passed to a parameter-style WGSL function by name.
                const irradiance = (giIrradiance({
                    position: positionWorld,
                    normal: normalWorld,
                    viewDirection,
                    screenUV,
                    irradianceAtlas: this.atlasNode,
                    volumeTexture: this.volumeNode,
                    probeTable: this.tableNode,
                    // A sampler parameter is fed from a texture node: the
                    // builder turns it into the node's `<name>_sampler` binding.
                    linearSampler: this.atlasNode,
                    volumeMin: this.volumeMin,
                    volumeSize: this.volumeSize,
                    settings: this.settings,
                }) as unknown as Node<'vec3'>).mul(this.intensityUniform);

                context.lightProbeIrradiance!.assign(irradiance.toConst());
                context.lightProbePriority!.assign(this.priorityUniform);
                context.giCoverage!.assign(1);
            });

            if (light.suppressEnvironmentDiffuse) this.suppressEnvironmentDiffuse(context);
            return undefined;
        }

        /**
         * r187 has no way to turn off `scene.environment`'s diffuse light (PR
         * #35017 adds `material.envMapDiffuseEnabled`). Workaround that needs no
         * material changes: wrap the `indirect` step of this build's lighting
         * model so it first scales `iblIrradiance` (the environment's diffuse
         * input) by `1 - coverage`.
         *
         * `LightingModel.start` builds all lights and then calls `this.indirect`,
         * so by then our coverage is final, whatever order the lights and the
         * environment node were built in.
         *
         * Cost: `iblIrradiance` also feeds the multi-scatter specular
         * compensation and sheen in the Physical model, so those lose their
         * environment term inside the volume. Lambert and Phong materials do not
         * read `iblIrradiance`, so they are unaffected.
         */
        private suppressEnvironmentDiffuse(context: LightingContext): void {
            const model = context.lightingModel;
            if (patchedLightingModels.has(model)) return;
            patchedLightingModels.add(model);
            const originalIndirect = model.indirect.bind(model);
            model.indirect = (builder: NodeBuilder) => {
                const current = builder.context as unknown as LightingContext;
                current.iblIrradiance.mulAssign(float(1).sub(current.giCoverage!));
                originalIndirect(builder);
            };
        }
    };
}
