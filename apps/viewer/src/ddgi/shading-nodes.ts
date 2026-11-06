/**
 * The DDGI shading module's resources as three.js nodes, for the viewer's
 * own debug drawing: the screen debug views (`giDebugColor`).
 *
 * The adapter's `GILight` does the same for `giIrradiance`; this follows its
 * pattern (NOTES, three.js "Wrapping our textures" and "`wgslFn` resources"):
 *  - a same-device `GPUTexture` becomes an `ExternalTexture` through the
 *    adapter's `wrapGpuTexture`, which sets the size, array flag and sample
 *    type three can't infer;
 *  - a sampler parameter is fed from its texture's node, which carries the
 *    filter and wrap settings;
 *  - `vec4f` parameters become uniforms copied from `value()` once per frame;
 *  - a reallocated texture is wrapped again and assigned to the node's value.
 */

import { Vector4, type Node } from 'three/webgpu';
import { texture, texture3D, uniform } from 'three/tsl';
import { wrapGpuTexture, type ShadingModule, type ShadingParameter } from './library';

type SamplerParameter = Extract<ShadingParameter, { kind: 'sampler' }>;
type TextureParameter = Extract<ShadingParameter, { kind: 'texture' }>;

type TextureNode = ReturnType<typeof texture>;
/** What this file needs of a `uniform(new Vector4())` node. */
type Vec4Uniform = { value: Vector4 };

/** One node per module parameter, kept current by {@link ShadingNodes.refresh}. */
export class ShadingNodes {
    /** By parameter name: the node to pass for it to a `wgslFn` call. */
    readonly byName: Record<string, Node> = {};

    private readonly textures = new Map<string, { node: TextureNode; gpuTexture: GPUTexture; parameter: TextureParameter; sampler: SamplerParameter | undefined }>();
    private readonly uniforms = new Map<string, Vec4Uniform>();

    constructor(private readonly module: ShadingModule) {
        const samplers = new Map<string, SamplerParameter>();
        for (const parameter of module.parameters) if (parameter.kind === 'sampler') samplers.set(parameter.texture, parameter);
        for (const parameter of module.parameters) {
            if (parameter.kind === 'texture') {
                const sampler = samplers.get(parameter.name);
                const gpuTexture = parameter.texture();
                const wrapped = wrapGpuTexture(gpuTexture, { ...parameter, filtering: sampler?.filtering, addressMode: sampler?.addressMode });
                const node = (parameter.dimension === '3d' ? texture3D(wrapped) : texture(wrapped)) as TextureNode;
                this.textures.set(parameter.name, { node, gpuTexture, parameter, sampler });
            } else if (parameter.kind === 'vec4f') {
                this.uniforms.set(parameter.name, uniform(new Vector4().fromArray(parameter.value())) as unknown as Vec4Uniform);
            }
        }
        for (const parameter of module.parameters) {
            // A sampler parameter is fed from its texture's node, which carries the sampler (NOTES).
            if (parameter.kind === 'sampler') this.byName[parameter.name] = this.textures.get(parameter.texture)!.node as unknown as Node;
            else if (parameter.kind === 'texture') this.byName[parameter.name] = this.textures.get(parameter.name)!.node as unknown as Node;
            else this.byName[parameter.name] = this.uniforms.get(parameter.name)! as unknown as Node;
        }
    }

    /** Copies the module's current uniform values, and re-wraps a texture the technique reallocated. Once per frame. */
    refresh(): void {
        for (const parameter of this.module.parameters) {
            if (parameter.kind === 'vec4f') {
                this.uniforms.get(parameter.name)!.value.fromArray(parameter.value());
            } else if (parameter.kind === 'texture') {
                const entry = this.textures.get(parameter.name)!;
                const gpuTexture = parameter.texture();
                if (gpuTexture === entry.gpuTexture) continue;
                entry.node.value = wrapGpuTexture(gpuTexture, { ...parameter, filtering: entry.sampler?.filtering, addressMode: entry.sampler?.addressMode });
                entry.gpuTexture = gpuTexture;
            }
        }
    }
}
