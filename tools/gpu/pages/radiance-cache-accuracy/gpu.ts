/**
 * The radiance-cache accuracy page's two small GPU helpers. Measurement
 * only, never a shading path.
 *
 * - {@link IrradianceAccumulator}: adds the radiance cache's interpolated
 *   irradiance (`probes.irradiance`, `rgba16float`) into an f32 sum per pixel
 *   after each frame, so the page averages many frames without reading each
 *   one back. The GI term's mean is then albedo × mean E / π (the albedo of
 *   a still view doesn't change).
 * - {@link WorldCacheProbe}: looks the world cache up at every pixel's
 *   visible surface, through the cache's own WGSL lookup
 *   (`WorldCache.lookup`, `worldCacheLookup`), with the view ray as the
 *   direction a probe ray would arrive along. The result is the radiance the
 *   cache holds for that surface (its direct light from the scene's lights
 *   plus, with multibounce, its bounced light), and w = 1 where the cache
 *   has a cell there, 0 where it doesn't.
 */

import type { WorldCacheLookup } from '../../../../packages/ddgi-core/src/techniques/radiance-cache/world-cache/world-cache.ts';

const WORKGROUP = 8;

const ACCUMULATE_WGSL = /* wgsl */ `
@group(0) @binding(0) var irradiance: texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> sums: array<vec4f>;
@group(0) @binding(2) var<uniform> size: vec4u;

@compute @workgroup_size(${WORKGROUP}, ${WORKGROUP})
fn accumulate(@builtin(global_invocation_id) id: vec3u) {
    if (id.x >= size.x || id.y >= size.y) {
        return;
    }
    let index = id.y * size.x + id.x;
    sums[index] += textureLoad(irradiance, vec2i(id.xy), 0);
}
`;

/** Reads a buffer back as floats. */
async function readFloats(device: GPUDevice, source: GPUBuffer, bytes: number): Promise<Float32Array> {
    const staging = device.createBuffer({ size: bytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const encoder = device.createCommandEncoder({ label: 'radiance-cache-accuracy readback' });
    encoder.copyBufferToBuffer(source, 0, staging, 0, bytes);
    device.queue.submit([encoder.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const floats = new Float32Array(staging.getMappedRange().slice(0));
    staging.unmap();
    staging.destroy();
    return floats;
}

/** Sums the irradiance texture over frames; one per screen size. */
export class IrradianceAccumulator {
    private readonly pipeline: GPUComputePipeline;
    private readonly sums: GPUBuffer;
    private readonly size: GPUBuffer;
    private group: GPUBindGroup | null = null;
    private groupTexture: GPUTexture | null = null;
    /** Frames added since the last `clear`. */
    frames = 0;

    constructor(
        private readonly device: GPUDevice,
        readonly width: number,
        readonly height: number,
    ) {
        this.pipeline = device.createComputePipeline({
            label: 'radiance-cache-accuracy accumulate',
            layout: 'auto',
            compute: { module: device.createShaderModule({ label: 'radiance-cache-accuracy accumulate', code: ACCUMULATE_WGSL }), entryPoint: 'accumulate' },
        });
        this.sums = device.createBuffer({ label: 'radiance-cache-accuracy irradiance sums', size: width * height * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
        this.size = device.createBuffer({ label: 'radiance-cache-accuracy size', size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        device.queue.writeBuffer(this.size, 0, new Uint32Array([width, height, 0, 0]));
    }

    clear(): void {
        const encoder = this.device.createCommandEncoder({ label: 'radiance-cache-accuracy clear sums' });
        encoder.clearBuffer(this.sums);
        this.device.queue.submit([encoder.finish()]);
        this.frames = 0;
    }

    /** Adds this frame's irradiance (submitted after the frame's own work, so it reads that frame). */
    add(irradiance: GPUTexture): void {
        if (irradiance.width !== this.width || irradiance.height !== this.height) {
            throw new Error(`the irradiance is ${irradiance.width} x ${irradiance.height}, the accumulator ${this.width} x ${this.height}`);
        }
        if (this.groupTexture !== irradiance) {
            this.group = this.device.createBindGroup({
                label: 'radiance-cache-accuracy accumulate',
                layout: this.pipeline.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: irradiance.createView() },
                    { binding: 1, resource: { buffer: this.sums } },
                    { binding: 2, resource: { buffer: this.size } },
                ],
            });
            this.groupTexture = irradiance;
        }
        const encoder = this.device.createCommandEncoder({ label: 'radiance-cache-accuracy accumulate' });
        const pass = encoder.beginComputePass({ label: 'radiance-cache-accuracy accumulate' });
        pass.setPipeline(this.pipeline);
        pass.setBindGroup(0, this.group);
        pass.dispatchWorkgroups(Math.ceil(this.width / WORKGROUP), Math.ceil(this.height / WORKGROUP));
        pass.end();
        this.device.queue.submit([encoder.finish()]);
        this.frames++;
    }

    /** The mean irradiance per pixel since `clear`, rgba, top row first. */
    async mean(): Promise<Float32Array> {
        if (this.frames === 0) throw new Error('no frames accumulated');
        const sums = await readFloats(this.device, this.sums, this.width * this.height * 16);
        for (let index = 0; index < sums.length; index++) sums[index] /= this.frames;
        return sums;
    }

    dispose(): void {
        this.sums.destroy();
        this.size.destroy();
    }
}

/** The view the lookup unprojects with: three's camera, WebGPU's 0..1 depth. */
export interface LookupCamera {
    /** Column-major `camera.matrixWorld × camera.projectionMatrixInverse` (NDC to world). */
    readonly inverseViewProjection: ArrayLike<number>;
    readonly eye: readonly [number, number, number];
}

const LOOKUP_GROUP = 1;

function lookupWgsl(lookup: WorldCacheLookup): string {
    return /* wgsl */ `
${lookup.code(0)}

struct LookupView {
    inverse_view_projection: mat4x4f,
    eye: vec4f,
    size: vec4u,
}

@group(${LOOKUP_GROUP}) @binding(0) var<uniform> lookup_view: LookupView;
@group(${LOOKUP_GROUP}) @binding(1) var<storage, read> lookup_depth: array<f32>;
@group(${LOOKUP_GROUP}) @binding(2) var<storage, read_write> lookup_radiance: array<vec4f>;

@compute @workgroup_size(64)
fn lookupVisibleSurfaces(@builtin(global_invocation_id) id: vec3u) {
    let width = lookup_view.size.x;
    let height = lookup_view.size.y;
    let index = id.x;
    if (index >= width * height) {
        return;
    }
    let depth = lookup_depth[index];
    if (!(depth < 1.0)) {
        lookup_radiance[index] = vec4f(0.0);   // the background: no surface
        return;
    }
    let x = f32(index % width);
    let y = f32(index / width);
    // Row 0 is the top of the picture; NDC y is up.
    let ndc = vec4f((x + 0.5) / f32(width) * 2.0 - 1.0, 1.0 - (y + 0.5) / f32(height) * 2.0, depth, 1.0);
    let world = lookup_view.inverse_view_projection * ndc;
    let position = world.xyz / world.w;
    let direction = normalize(position - lookup_view.eye.xyz);
    lookup_radiance[index] = worldCacheLookup(position, -direction, direction);
}
`;
}

/** Looks the world cache up at each pixel's visible surface. */
export class WorldCacheProbe {
    private readonly pipeline: GPUComputePipeline;
    private readonly cacheGroup: GPUBindGroup;
    private readonly viewLayout: GPUBindGroupLayout;

    private constructor(
        private readonly device: GPUDevice,
        pipeline: GPUComputePipeline,
        cacheGroup: GPUBindGroup,
        viewLayout: GPUBindGroupLayout,
    ) {
        this.pipeline = pipeline;
        this.cacheGroup = cacheGroup;
        this.viewLayout = viewLayout;
    }

    static async create(device: GPUDevice, lookup: WorldCacheLookup): Promise<WorldCacheProbe> {
        const cacheLayout = device.createBindGroupLayout({ label: 'radiance-cache-accuracy world cache', entries: [...lookup.entries] });
        const viewLayout = device.createBindGroupLayout({
            label: 'radiance-cache-accuracy lookup view',
            entries: [
                { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
                { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            ],
        });
        const module = device.createShaderModule({ label: 'radiance-cache-accuracy world-cache lookup', code: lookupWgsl(lookup) });
        const pipeline = await device.createComputePipelineAsync({
            label: 'radiance-cache-accuracy world-cache lookup',
            layout: device.createPipelineLayout({ bindGroupLayouts: [cacheLayout, viewLayout] }),
            compute: { module, entryPoint: 'lookupVisibleSurfaces' },
        });
        return new WorldCacheProbe(device, pipeline, lookup.createBindGroup(cacheLayout), viewLayout);
    }

    /**
     * The cache's radiance at each pixel's surface (rgba, top row first; w 1 where it has a cell).
     * `depth` is the scene pass's depth per pixel, top row first, 1 for the background.
     */
    async look(depth: Float32Array<ArrayBuffer>, width: number, height: number, camera: LookupCamera): Promise<Float32Array> {
        const device = this.device;
        const viewWords = new ArrayBuffer(96);
        new Float32Array(viewWords, 0, 16).set(Array.from(camera.inverseViewProjection));
        new Float32Array(viewWords, 64, 4).set([camera.eye[0], camera.eye[1], camera.eye[2], 1]);
        new Uint32Array(viewWords, 80, 4).set([width, height, 0, 0]);
        const view = device.createBuffer({ size: viewWords.byteLength, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
        device.queue.writeBuffer(view, 0, viewWords);
        const depthBuffer = device.createBuffer({ size: depth.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
        device.queue.writeBuffer(depthBuffer, 0, depth);
        const radianceBytes = width * height * 16;
        const radiance = device.createBuffer({ size: radianceBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
        const group = device.createBindGroup({
            layout: this.viewLayout,
            entries: [
                { binding: 0, resource: { buffer: view } },
                { binding: 1, resource: { buffer: depthBuffer } },
                { binding: 2, resource: { buffer: radiance } },
            ],
        });
        const encoder = device.createCommandEncoder({ label: 'radiance-cache-accuracy world-cache lookup' });
        const pass = encoder.beginComputePass({ label: 'radiance-cache-accuracy world-cache lookup' });
        pass.setPipeline(this.pipeline);
        pass.setBindGroup(0, this.cacheGroup);
        pass.setBindGroup(LOOKUP_GROUP, group);
        const workgroups = Math.ceil((width * height) / 64);
        if (workgroups > device.limits.maxComputeWorkgroupsPerDimension) throw new RangeError(`the lookup needs ${workgroups} workgroups in one dimension`);
        pass.dispatchWorkgroups(workgroups);
        pass.end();
        device.queue.submit([encoder.finish()]);
        const result = await readFloats(device, radiance, radianceBytes);
        view.destroy();
        depthBuffer.destroy();
        radiance.destroy();
        return result;
    }
}
