import * as THREE from 'three/webgpu';

/** Whole-queue timestamps include the raster, cascade and presentation submissions. */
export function frameTimestamps(device: GPUDevice, count: number) {
    const queries = device.createQuerySet({ type: 'timestamp', count: count * 2 });
    const size = count * 16;
    const result = device.createBuffer({ size, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
    const read = device.createBuffer({ size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    return {
        mark(index: number) {
            const encoder = device.createCommandEncoder();
            encoder.beginComputePass({ timestampWrites: { querySet: queries, beginningOfPassWriteIndex: index } }).end();
            device.queue.submit([encoder.finish()]);
        },
        async finish() {
            try {
                const encoder = device.createCommandEncoder();
                encoder.resolveQuerySet(queries,0,count*2,result,0);
                encoder.copyBufferToBuffer(result,0,read,0,size);
                device.queue.submit([encoder.finish()]);
                await read.mapAsync(GPUMapMode.READ);
                const times = new BigUint64Array(read.getMappedRange());
                return Array.from({length:count},(_,i) => Number(times[2*i+1]-times[2*i])/1e6);
            } finally { read.destroy(); result.destroy(); queries.destroy(); }
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
