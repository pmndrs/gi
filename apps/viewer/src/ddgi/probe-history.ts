/**
 * Each DDGI probe's history count n (adaptive hysteresis: the updates its
 * light is averaged over) as a texture, so the probe spheres can colour by it.
 *
 * DDGI keeps the history in a storage buffer (`DDGIField.probeHistory`) that
 * only its blend kernel uses. three can't bind an outside buffer to a vertex
 * shader (NOTES, three.js "External buffers"), so this small compute pass
 * copies n into an `r32float` texture laid out like the `points` debug view:
 * probe i at texel (i mod width, i div width). It reads the buffer by the
 * layout the core exports (`PROBE_HISTORY_WORD`), so the viewer never
 * restates the struct.
 *
 * Plain WebGPU on the viewer's device, submitted after DDGI's update and
 * before three draws, so the spheres show this frame's counts. It runs only
 * while the spheres are coloured by history.
 */

import { PROBE_HISTORY_WORD, PROBE_HISTORY_WORDS, type DDGIField } from './library';

/** Threads per workgroup; one per probe. */
const WORKGROUP_SIZE = 64;

const COPY_SAMPLES_WGSL = /* wgsl */ `
// The history buffer read as plain words: PROBE_HISTORY_WORDS per probe.
@group(0) @binding(0) var<storage, read> probeHistory: array<u32>;
@group(0) @binding(1) var samplesOut: texture_storage_2d<r32float, write>;
// x: probe count, y: the texture's width.
@group(0) @binding(2) var<uniform> shape: vec4u;

const WORDS_PER_PROBE: u32 = ${PROBE_HISTORY_WORDS}u;
const SAMPLES_WORD: u32 = ${PROBE_HISTORY_WORD.samples}u;

@compute @workgroup_size(${WORKGROUP_SIZE})
fn copySamples(@builtin(global_invocation_id) id: vec3u) {
    let probe = id.x;
    if (probe >= shape.x) {
        return;
    }
    let samples = bitcast<f32>(probeHistory[probe * WORDS_PER_PROBE + SAMPLES_WORD]);
    let width = shape.y;
    textureStore(samplesOut, vec2u(probe % width, probe / width), vec4f(samples, 0.0, 0.0, 1.0));
}
`;

export class ProbeHistoryTexture {
    private readonly pipeline: GPUComputePipeline;
    private readonly shape: GPUBuffer;
    private target: GPUTexture | null = null;
    private bindGroup: GPUBindGroup | null = null;
    /** The history buffer the bind group reads; a new `prepare` replaces it. */
    private source: GPUBuffer | null = null;
    private probeCount = 0;

    constructor(private readonly device: GPUDevice) {
        const module = device.createShaderModule({ label: 'viewer probe history copy', code: COPY_SAMPLES_WGSL });
        this.pipeline = device.createComputePipeline({ label: 'viewer probe history copy', layout: 'auto', compute: { module, entryPoint: 'copySamples' } });
        this.shape = device.createBuffer({ label: 'viewer probe history shape', size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    }

    /**
     * The texture, sized `width` × `height` (the points texture's size). Made on first use,
     * remade when the size changes; it starts at zero (no history).
     */
    texture(width: number, height: number): GPUTexture {
        const current = this.target;
        if (current && current.width === width && current.height === height) return current;
        current?.destroy();
        this.target = this.device.createTexture({
            label: 'viewer probe history samples',
            size: { width, height },
            format: 'r32float',
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.STORAGE_BINDING,
        });
        this.bindGroup = null;
        return this.target;
    }

    /** Copies every probe's n into the texture. Submits its own small command buffer. */
    update(field: DDGIField, width: number, height: number): void {
        const target = this.texture(width, height);
        const probeCount = field.grid.probeCount;
        if (!this.bindGroup || this.source !== field.probeHistory || this.probeCount !== probeCount) {
            this.source = field.probeHistory;
            this.probeCount = probeCount;
            this.device.queue.writeBuffer(this.shape, 0, new Uint32Array([probeCount, width, 0, 0]));
            this.bindGroup = this.device.createBindGroup({
                label: 'viewer probe history copy',
                layout: this.pipeline.getBindGroupLayout(0),
                entries: [
                    { binding: 0, resource: { buffer: field.probeHistory } },
                    { binding: 1, resource: target.createView() },
                    { binding: 2, resource: { buffer: this.shape } },
                ],
            });
        }
        const encoder = this.device.createCommandEncoder({ label: 'viewer probe history copy' });
        const pass = encoder.beginComputePass({ label: 'viewer probe history copy' });
        pass.setPipeline(this.pipeline);
        pass.setBindGroup(0, this.bindGroup);
        pass.dispatchWorkgroups(Math.ceil(probeCount / WORKGROUP_SIZE));
        pass.end();
        this.device.queue.submit([encoder.finish()]);
    }

    dispose(): void {
        this.target?.destroy();
        this.shape.destroy();
    }
}
