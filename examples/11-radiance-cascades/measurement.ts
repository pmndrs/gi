import * as THREE from 'three/webgpu';

/**
 * Whole-queue timestamps include the raster, cascade and presentation submissions.
 *
 * Each frame is bracketed by two tiny compute passes submitted on their own. Some GPUs drop the
 * timestamp of a pass that does no work, so each marker dispatches one empty workgroup. Apple
 * GPUs also run independent submissions side by side: an end marker that shares no resource with
 * the frame starts as soon as the frame starts. The end marker therefore copies one texel of the
 * frame's final output and reads it, which makes it wait until that output has been written.
 */
export function frameTimestamps(device: GPUDevice, count: number, frameOutput: () => GPUTexture) {
    const queries = device.createQuerySet({ type: 'timestamp', count: count * 2 });
    const size = count * 16;
    const result = device.createBuffer({ size, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
    const read = device.createBuffer({ size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    // One texel of the presented frame. Texture-to-buffer copies need 256-byte rows.
    const outputTexel = device.createBuffer({ size: 256, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE });
    const marker = device.createComputePipeline({ layout: 'auto', compute: { entryPoint: 'main', module: device.createShaderModule({ code: /* wgsl */`
        @group(0) @binding(0) var<storage, read_write> texel: array<u32>;
        @compute @workgroup_size(1) fn main() { texel[1] = texel[0]; }` }) } });
    const markerBindings = device.createBindGroup({ layout: marker.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: outputTexel } }] });
    function mark(queryIndex: number, afterOutput: boolean) {
        const encoder = device.createCommandEncoder();
        if (afterOutput) encoder.copyTextureToBuffer({ texture: frameOutput() }, { buffer: outputTexel, bytesPerRow: 256 }, [1, 1]);
        const pass = encoder.beginComputePass({ timestampWrites: { querySet: queries, beginningOfPassWriteIndex: queryIndex } });
        pass.setPipeline(marker);
        pass.setBindGroup(0, markerBindings);
        pass.dispatchWorkgroups(1);
        pass.end();
        device.queue.submit([encoder.finish()]);
    }
    return {
        /** Call before submitting the frame's work. */
        begin(frameIndex: number) { mark(2 * frameIndex, false); },
        /** Call after submitting the frame's work and before the browser presents it. */
        end(frameIndex: number) { mark(2 * frameIndex + 1, true); },
        async finish() {
            try {
                const encoder = device.createCommandEncoder();
                encoder.resolveQuerySet(queries,0,count*2,result,0);
                encoder.copyBufferToBuffer(result,0,read,0,size);
                device.queue.submit([encoder.finish()]);
                await read.mapAsync(GPUMapMode.READ);
                const times = new BigUint64Array(read.getMappedRange());
                return Array.from({length:count},(_,i) => Number(times[2*i+1]-times[2*i])/1e6);
            } finally { read.destroy(); result.destroy(); queries.destroy(); outputTexel.destroy(); }
        },
    };
}

/** Sample the center of the receiver in linear RGB before any display transfer or temporal resolve. */
export function receiverMeter(renderer: THREE.WebGPURenderer) {
    const target = new THREE.RenderTarget(48, 27, { type: THREE.HalfFloatType, depthBuffer: false });
    const material = new THREE.NodeMaterial();
    material.toneMapped = false;
    const quad = new THREE.QuadMesh(material);
    return {
        async sample(composite: THREE.Node) {
            if (material.fragmentNode !== composite) { material.fragmentNode = composite; material.needsUpdate = true; }
            const old = renderer.getRenderTarget(), mrt = renderer.getMRT();
            try {
                renderer.setMRT(null); renderer.setRenderTarget(target);
                quad.render(renderer);
            } finally { renderer.setRenderTarget(old); renderer.setMRT(mrt); }
            const data = await renderer.readRenderTargetPixelsAsync(target,16,9,16,9);
            const words = new Uint16Array(data.buffer,data.byteOffset,data.byteLength / 2);
            // Three returns tightly packed rows after stripping WebGPU copy padding.
            const stride = Math.floor(words.length / 9);
            const half = THREE.DataUtils.fromHalfFloat;
            let sum = 0;
            for (let y=0; y<9; y++) for (let x=0; x<16; x++) {
                const i = y*stride+x*4;
                sum += 0.2126*half(words[i])+0.7152*half(words[i+1])+0.0722*half(words[i+2]);
            }
            return sum / 144;
        },
        dispose() { material.dispose(); target.dispose(); },
    };
}
