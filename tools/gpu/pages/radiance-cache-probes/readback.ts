/**
 * Readbacks the radiance-cache pages share (radiance-cache-probes,
 * radiance-cache-image). Not a page itself.
 */

import { halfBitsToFloat } from '../../../../packages/cache/src/capsaicin-math.ts';

/** A texture's mip level as bytes, rows packed (the copy pads them to 256 bytes). */
export async function readTexture(device: GPUDevice, texture: GPUTexture, bytesPerTexel: number, width: number, height: number, mipLevel = 0): Promise<Uint8Array> {
    const rowBytes = width * bytesPerTexel;
    const paddedRow = Math.ceil(rowBytes / 256) * 256;
    const buffer = device.createBuffer({ size: paddedRow * height, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const encoder = device.createCommandEncoder();
    encoder.copyTextureToBuffer({ texture, mipLevel }, { buffer, bytesPerRow: paddedRow }, [width, height]);
    device.queue.submit([encoder.finish()]);
    await buffer.mapAsync(GPUMapMode.READ);
    const mapped = new Uint8Array(buffer.getMappedRange());
    const out = new Uint8Array(rowBytes * height);
    for (let row = 0; row < height; row++) out.set(mapped.subarray(row * paddedRow, row * paddedRow + rowBytes), row * rowBytes);
    buffer.unmap();
    buffer.destroy();
    return out;
}

/** Bytes of half floats as numbers. */
export function halves(bytes: Uint8Array, byteOffset = 0): Float32Array {
    const words = new Uint16Array(bytes.buffer, bytes.byteOffset + byteOffset, (bytes.byteLength - byteOffset) / 2);
    return Float32Array.from(words, (bits) => halfBitsToFloat(bits));
}
