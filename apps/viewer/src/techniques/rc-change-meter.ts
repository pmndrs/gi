/**
 * How much the radiance cache's GI term changes from one frame to the next,
 * for the compare section's readout (RC4.5): the GI flicker with a still
 * camera, and how long the picture takes to settle after a toggle.
 *
 * Each frame, before the radiance cache runs again, a small compute pass reads
 * its interpolated irradiance E (the previous frame's) on a sparse grid (one
 * pixel in every `GRID_STEP` × `GRID_STEP`), compares each sample with the
 * value it kept from the frame before, and keeps the new one. Each workgroup
 * sums |E_now − E_before| and E_before; the partial sums are read back
 * without waiting (a ring of staging buffers; a frame whose buffer is still
 * busy is skipped) and added up on the CPU:
 *
 *   change = Σ |E_now − E_before| / Σ E_before
 *
 * the mean change relative to the picture's mean GI, as DDGI's flicker pages
 * measure it (`tools/gpu/pages/ddgi-flicker`), so dark corners don't read as
 * huge changes. E is the mean of its three channels.
 *
 * Plain WebGPU on the viewer's device, submitted as its own small command
 * buffer: about 58,000 samples and a 7 KB readback at 1280 × 720.
 */

/** One sample in this many pixels each way. */
const GRID_STEP = 4;
/** Samples per workgroup (one thread each). */
const WORKGROUP_SIZE = 64;
/** Readings averaged for the shown value. */
const AVERAGED_READINGS = 8;
/** Staging buffers in flight at most. */
const STAGING_BUFFERS = 3;

const CHANGE_WGSL = /* wgsl */ `
@group(0) @binding(0) var irradiance: texture_2d<f32>;
// The sample grid's E from the frame before.
@group(0) @binding(1) var<storage, read_write> previous: array<f32>;
// Per workgroup: sum |E_now - E_before|, sum E_before.
@group(0) @binding(2) var<storage, read_write> partials: array<vec2f>;
// x, y: the grid's size; z: the grid step in pixels; w: 1 when 'previous' holds the frame before.
@group(0) @binding(3) var<uniform> shape: vec4u;

var<workgroup> sums: array<vec2f, ${WORKGROUP_SIZE}>;

@compute @workgroup_size(${WORKGROUP_SIZE})
fn measureChange(@builtin(global_invocation_id) id: vec3u, @builtin(local_invocation_index) local: u32, @builtin(workgroup_id) group: vec3u) {
    let index = id.x;
    var sum = vec2f(0.0);
    if (index < shape.x * shape.y) {
        let pixel = vec2u(index % shape.x, index / shape.x) * shape.z;
        let e = dot(textureLoad(irradiance, pixel, 0).rgb, vec3f(1.0 / 3.0));
        let now = select(0.0, e, e == e && e >= 0.0 && e < 1e30); // NaN, negative or huge: left out
        if (shape.w != 0u) {
            let before = previous[index];
            sum = vec2f(abs(now - before), before);
        }
        previous[index] = now;
    }
    sums[local] = sum;
    workgroupBarrier();
    for (var stride = ${WORKGROUP_SIZE / 2}u; stride > 0u; stride >>= 1u) {
        if (local < stride) {
            sums[local] += sums[local + stride];
        }
        workgroupBarrier();
    }
    if (local == 0u) {
        partials[group.x] = sums[0];
    }
}
`;

interface Sized {
    readonly texture: GPUTexture;
    readonly gridWidth: number;
    readonly gridHeight: number;
    readonly workgroups: number;
    readonly previous: GPUBuffer;
    readonly partials: GPUBuffer;
    readonly bindGroup: GPUBindGroup;
}

export class RcChangeMeter {
    private readonly pipeline: GPUComputePipeline;
    private readonly shape: GPUBuffer;
    private sized: Sized | null = null;
    /** False until `previous` holds a frame. */
    private hasPrevious = false;
    /** Every staging buffer made for this size, and those not waiting on a readback. */
    private readonly staging: GPUBuffer[] = [];
    private readonly idle: GPUBuffer[] = [];
    private readonly readings: number[] = [];
    private disposed = false;

    constructor(private readonly device: GPUDevice) {
        const module = device.createShaderModule({ label: 'viewer rc change meter', code: CHANGE_WGSL });
        this.pipeline = device.createComputePipeline({ label: 'viewer rc change meter', layout: 'auto', compute: { module, entryPoint: 'measureChange' } });
        this.shape = device.createBuffer({ label: 'viewer rc change meter shape', size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    }

    /** The mean of the last few readings, in percent; null before the first. */
    get meanChangePercent(): number | null {
        if (this.readings.length === 0) return null;
        return (100 * this.readings.reduce((sum, value) => sum + value, 0)) / this.readings.length;
    }

    /** Forgets the readings (a cut: the camera jumped, GI was off). The next frame only keeps its values. */
    restart(): void {
        this.hasPrevious = false;
        this.readings.length = 0;
    }

    /** Measures `irradiance` (the radiance cache's latest frame) against the frame before. */
    measure(irradiance: GPUTexture): void {
        if (this.disposed) return;
        const sized = this.sizedFor(irradiance);
        this.device.queue.writeBuffer(this.shape, 0, new Uint32Array([sized.gridWidth, sized.gridHeight, GRID_STEP, this.hasPrevious ? 1 : 0]));
        const encoder = this.device.createCommandEncoder({ label: 'viewer rc change meter' });
        const pass = encoder.beginComputePass({ label: 'viewer rc change meter' });
        pass.setPipeline(this.pipeline);
        pass.setBindGroup(0, sized.bindGroup);
        pass.dispatchWorkgroups(sized.workgroups);
        pass.end();
        const staging = this.hasPrevious ? this.stagingBuffer(sized.partials.size) : null;
        if (staging) encoder.copyBufferToBuffer(sized.partials, 0, staging, 0, sized.partials.size);
        this.device.queue.submit([encoder.finish()]);
        this.hasPrevious = true;
        if (staging) void this.read(staging, sized);
    }

    dispose(): void {
        this.disposed = true;
        this.sized?.previous.destroy();
        this.sized?.partials.destroy();
        this.shape.destroy();
        this.releaseStaging();
    }

    private async read(staging: GPUBuffer, sized: Sized): Promise<void> {
        try {
            await staging.mapAsync(GPUMapMode.READ);
        } catch {
            return; // the device went away
        }
        if (this.disposed || !this.staging.includes(staging)) return;
        const partials = new Float32Array(staging.getMappedRange().slice(0));
        staging.unmap();
        this.idle.push(staging);
        let change = 0;
        let total = 0;
        for (let index = 0; index < partials.length; index += 2) {
            change += partials[index];
            total += partials[index + 1];
        }
        if (!(total > 0) || this.sized !== sized) return;
        this.readings.push(change / total);
        if (this.readings.length > AVERAGED_READINGS) this.readings.shift();
    }

    /** An idle staging buffer, or a new one while fewer than STAGING_BUFFERS exist; null when all are busy. */
    private stagingBuffer(size: number): GPUBuffer | null {
        const idle = this.idle.pop();
        if (idle) return idle;
        if (this.staging.length >= STAGING_BUFFERS) return null;
        const buffer = this.device.createBuffer({ label: 'viewer rc change meter readback', size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
        this.staging.push(buffer);
        return buffer;
    }

    /** Destroys the staging buffers (a pending readback then fails and is dropped). */
    private releaseStaging(): void {
        for (const buffer of this.staging) buffer.destroy();
        this.staging.length = 0;
        this.idle.length = 0;
    }

    private sizedFor(texture: GPUTexture): Sized {
        const current = this.sized;
        if (current && current.texture === texture) return current;
        current?.previous.destroy();
        current?.partials.destroy();
        this.releaseStaging();
        this.restart();
        const gridWidth = Math.ceil(texture.width / GRID_STEP);
        const gridHeight = Math.ceil(texture.height / GRID_STEP);
        const workgroups = Math.ceil((gridWidth * gridHeight) / WORKGROUP_SIZE);
        const previous = this.device.createBuffer({ label: 'viewer rc change meter previous', size: gridWidth * gridHeight * 4, usage: GPUBufferUsage.STORAGE });
        const partials = this.device.createBuffer({ label: 'viewer rc change meter partials', size: workgroups * 8, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
        const bindGroup = this.device.createBindGroup({
            label: 'viewer rc change meter',
            layout: this.pipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: texture.createView() },
                { binding: 1, resource: { buffer: previous } },
                { binding: 2, resource: { buffer: partials } },
                { binding: 3, resource: { buffer: this.shape } },
            ],
        });
        this.sized = { texture, gridWidth, gridHeight, workgroups, previous, partials, bindGroup };
        return this.sized;
    }
}
