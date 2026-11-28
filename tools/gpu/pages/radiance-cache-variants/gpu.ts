/**
 * P6.5's GPU helper: the mean of an irradiance texture's rgb over each 8 × 8
 * block, so the page can follow a change frame by frame with a small read
 * (a 1920 × 1080 frame is 240 × 135 blocks, 130 KB) instead of the whole
 * texture. Measurement only, never a shading path.
 */

const BLOCK = 8;

const BLOCK_MEANS_WGSL = /* wgsl */ `
@group(0) @binding(0) var irradiance: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> blocks: array<f32>;
// width, height, blocks across, unused
@group(0) @binding(2) var<uniform> size: vec4u;

var<workgroup> partial: array<f32, ${BLOCK * BLOCK}>;

@compute @workgroup_size(${BLOCK}, ${BLOCK})
fn blockMeans(@builtin(global_invocation_id) id: vec3u, @builtin(workgroup_id) group: vec3u, @builtin(local_invocation_index) lane: u32) {
    var value = 0.0;
    if (id.x < size.x && id.y < size.y) {
        let e = textureLoad(irradiance, vec2i(id.xy), 0).rgb;
        value = (e.r + e.g + e.b) / 3.0;
    }
    partial[lane] = value;
    workgroupBarrier();
    for (var stride = ${(BLOCK * BLOCK) / 2}u; stride > 0u; stride = stride >> 1u) {
        if (lane < stride) {
            partial[lane] += partial[lane + stride];
        }
        workgroupBarrier();
    }
    if (lane == 0u) {
        blocks[group.y * size.z + group.x] = partial[0] / ${BLOCK * BLOCK}.0;
    }
}
`;

/** Block means of one screen size's irradiance. */
export class BlockMeans {
    readonly blocksX: number;
    readonly blocksY: number;
    private readonly pipeline: GPUComputePipeline;
    private readonly blocks: GPUBuffer;
    private readonly size: GPUBuffer;

    constructor(
        private readonly device: GPUDevice,
        readonly width: number,
        readonly height: number,
    ) {
        this.blocksX = Math.ceil(width / BLOCK);
        this.blocksY = Math.ceil(height / BLOCK);
        this.pipeline = device.createComputePipeline({
            label: 'radiance-cache-variants block means',
            layout: 'auto',
            compute: { module: device.createShaderModule({ label: 'radiance-cache-variants block means', code: BLOCK_MEANS_WGSL }), entryPoint: 'blockMeans' },
        });
        this.blocks = device.createBuffer({ label: 'radiance-cache-variants blocks', size: this.blocksX * this.blocksY * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
        this.size = device.createBuffer({ label: 'radiance-cache-variants size', size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        device.queue.writeBuffer(this.size, 0, new Uint32Array([width, height, this.blocksX, 0]));
    }

    /** This frame's block means (submitted after the frame's own work, so it reads that frame). */
    async read(irradiance: GPUTexture): Promise<Float32Array> {
        if (irradiance.width !== this.width || irradiance.height !== this.height) throw new Error(`the irradiance is ${irradiance.width} x ${irradiance.height}, the block means ${this.width} x ${this.height}`);
        const group = this.device.createBindGroup({
            label: 'radiance-cache-variants block means',
            layout: this.pipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: irradiance.createView() },
                { binding: 1, resource: { buffer: this.blocks } },
                { binding: 2, resource: { buffer: this.size } },
            ],
        });
        const bytes = this.blocksX * this.blocksY * 4;
        const staging = this.device.createBuffer({ size: bytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
        const encoder = this.device.createCommandEncoder({ label: 'radiance-cache-variants block means' });
        const pass = encoder.beginComputePass({ label: 'radiance-cache-variants block means' });
        pass.setPipeline(this.pipeline);
        pass.setBindGroup(0, group);
        pass.dispatchWorkgroups(this.blocksX, this.blocksY);
        pass.end();
        encoder.copyBufferToBuffer(this.blocks, 0, staging, 0, bytes);
        this.device.queue.submit([encoder.finish()]);
        await staging.mapAsync(GPUMapMode.READ);
        const values = new Float32Array(staging.getMappedRange().slice(0));
        staging.unmap();
        staging.destroy();
        return values;
    }

    dispose(): void {
        this.blocks.destroy();
        this.size.destroy();
    }
}
