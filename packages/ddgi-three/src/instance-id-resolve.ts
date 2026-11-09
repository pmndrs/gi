/**
 * Instance ids from the scene pass's blendable encoding to `FrameScreen`'s
 * `r32uint` (RC1.1 review).
 *
 * Why an encoding at all: a blended (transparent) surface must leave the
 * screen channels as the opaque surface under it (screen-inputs.ts), and the
 * only per-material way to leave an MRT target unchanged in three r187 is
 * blending (a write mask applies to every target, the `output` colour too).
 * Integer targets can't blend, so the pass writes the id + 1 as three bytes
 * in an `rgba8unorm` target (red the lowest byte), alpha 1 for an opaque
 * surface and 0 for a blended one, and this small pass turns it into the
 * `r32uint` texture the contract promises. 8-bit unorm stores k / 255
 * exactly and blending by 1 and 0 keeps it exact, so the ids survive.
 * Ids up to 2^24 − 1 (16.7 million instances).
 */

/** The largest id (instance index + 1) three bytes hold. */
export const MAX_ENCODED_INSTANCE_ID = 0xffffff;

const WORKGROUP_SIZE = 8;

const RESOLVE_KERNEL = /* wgsl */ `
@group(0) @binding(0) var encodedIds: texture_2d<f32>;
@group(0) @binding(1) var instanceIds: texture_storage_2d<r32uint, write>;

@compute @workgroup_size(${WORKGROUP_SIZE}, ${WORKGROUP_SIZE})
fn main(@builtin(global_invocation_id) id: vec3u) {
    let size = textureDimensions(instanceIds);
    if (id.x >= size.x || id.y >= size.y) {
        return;
    }
    // Each byte was stored as k / 255; round back to k.
    let bytes = vec3u(round(textureLoad(encodedIds, id.xy, 0).rgb * 255.0));
    textureStore(instanceIds, id.xy, vec4u(bytes.x | (bytes.y << 8u) | (bytes.z << 16u), 0u, 0u, 0u));
}
`;

export class InstanceIdResolve {
    private readonly device: GPUDevice;
    private pipeline: GPUComputePipeline | null = null;
    private target: GPUTexture | null = null;

    constructor(device: GPUDevice) {
        this.device = device;
    }

    /** Records the conversion of `encoded` (the pass's `giInstance` target) and returns the `r32uint` texture it writes, the same size. */
    encode(encoder: GPUCommandEncoder, encoded: GPUTexture): GPUTexture {
        const device = this.device;
        // Compiled on the first frame that has ids; one small kernel on the host's device.
        this.pipeline ??= device.createComputePipeline({
            label: 'ddgi-three instance ids',
            layout: 'auto',
            compute: { module: device.createShaderModule({ label: 'ddgi-three instance ids', code: RESOLVE_KERNEL }), entryPoint: 'main' },
        });
        const target = this.targetFor(encoded.width, encoded.height);
        const bindGroup = device.createBindGroup({
            label: 'ddgi-three instance ids',
            layout: this.pipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: encoded.createView() },
                { binding: 1, resource: target.createView() },
            ],
        });
        const pass = encoder.beginComputePass({ label: 'ddgi-three instance ids' });
        pass.setPipeline(this.pipeline);
        pass.setBindGroup(0, bindGroup);
        pass.dispatchWorkgroups(Math.ceil(encoded.width / WORKGROUP_SIZE), Math.ceil(encoded.height / WORKGROUP_SIZE));
        pass.end();
        return target;
    }

    dispose(): void {
        this.target?.destroy();
        this.target = null;
    }

    private targetFor(width: number, height: number): GPUTexture {
        const current = this.target;
        if (current && current.width === width && current.height === height) return current;
        current?.destroy();
        this.target = this.device.createTexture({
            label: 'ddgi-three instance ids',
            size: { width, height },
            format: 'r32uint',
            // Written here, read by techniques and the debug view, copied by checks.
            usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC,
        });
        return this.target;
    }
}
