/**
 * Read-backs for the `three-screen-inputs` pages: three's own MRT targets
 * through `renderer.readRenderTargetPixelsAsync` (three's path, not ours),
 * and the core debug view's texture.
 *
 * Not a page itself (no index.html).
 */

import type { PassNode, WebGPURenderer } from 'three/webgpu';

/** Half-float bits to a number. */
export function halfToFloat(bits: number): number {
    const sign = bits & 0x8000 ? -1 : 1;
    const exponent = (bits >> 10) & 0x1f;
    const mantissa = bits & 0x3ff;
    if (exponent === 0) return sign * mantissa * 2 ** -24;
    if (exponent === 31) return mantissa === 0 ? sign * Infinity : NaN;
    return sign * (1 + mantissa / 1024) * 2 ** (exponent - 15);
}

/** A whole target as numbers, `channels` per pixel, rows from the top (WebGPU's texel order). */
export interface TargetImage {
    readonly width: number;
    readonly height: number;
    readonly channels: number;
    readonly values: Float64Array | Uint32Array;
    at(x: number, y: number, channel: number): number;
}

function image(width: number, height: number, channels: number, values: Float64Array | Uint32Array): TargetImage {
    return { width, height, channels, values, at: (x, y, channel) => values[(y * width + x) * channels + channel] };
}

/**
 * One of a pass's MRT targets read back by three: half floats and 8-bit
 * unorm as numbers (unorm / 255), r32uint as integers.
 */
export async function readPassTarget(renderer: WebGPURenderer, scenePass: PassNode, name: string): Promise<TargetImage> {
    const target = scenePass.renderTarget;
    // By name, not `getTexture`, which would make a target that was never drawn.
    const index = target.textures.findIndex((texture) => texture.name === name);
    if (index < 0) throw new Error(`the pass has no '${name}' target`);
    const { width, height } = target;
    const pixels = (await renderer.readRenderTargetPixelsAsync(target, 0, 0, width, height, index)) as ArrayLike<number> & { BYTES_PER_ELEMENT: number };
    const count = width * height;
    const channels = Math.round(pixels.length / count);
    if (pixels instanceof Uint32Array) return image(width, height, channels, new Uint32Array(pixels));
    const values = new Float64Array(pixels.length);
    if (pixels instanceof Uint16Array) for (let index = 0; index < pixels.length; index++) values[index] = halfToFloat(pixels[index]);
    else if (pixels instanceof Uint8Array) for (let index = 0; index < pixels.length; index++) values[index] = pixels[index] / 255;
    else for (let index = 0; index < pixels.length; index++) values[index] = pixels[index];
    return image(width, height, channels, values);
}

/**
 * The pass's instance-id target read back by three and decoded the way the
 * adapter encodes it (three bytes, red the lowest; instance-id-resolve.ts),
 * one id per pixel.
 */
export async function readInstanceIds(renderer: WebGPURenderer, scenePass: PassNode, name: string): Promise<TargetImage> {
    const encoded = await readPassTarget(renderer, scenePass, name);
    const ids = new Uint32Array(encoded.width * encoded.height);
    for (let pixel = 0; pixel < ids.length; pixel++) {
        const byte = (channel: number) => Math.round(encoded.values[pixel * encoded.channels + channel] * 255);
        ids[pixel] = byte(0) | (byte(1) << 8) | (byte(2) << 16);
    }
    return image(encoded.width, encoded.height, 1, ids);
}

/** An `rgba8unorm` texture (the debug view) read back as bytes, rows unpadded. */
export async function readRgba8(device: GPUDevice, texture: GPUTexture): Promise<Uint8ClampedArray<ArrayBuffer>> {
    const { width, height } = texture;
    const bytesPerRow = Math.ceil((width * 4) / 256) * 256;
    const buffer = device.createBuffer({ size: bytesPerRow * height, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const encoder = device.createCommandEncoder();
    encoder.copyTextureToBuffer({ texture }, { buffer, bytesPerRow }, { width, height });
    device.queue.submit([encoder.finish()]);
    await buffer.mapAsync(GPUMapMode.READ);
    const padded = new Uint8Array(buffer.getMappedRange());
    const pixels = new Uint8ClampedArray(width * height * 4);
    for (let row = 0; row < height; row++) pixels.set(padded.subarray(row * bytesPerRow, row * bytesPerRow + width * 4), row * width * 4);
    buffer.destroy();
    return pixels;
}
