/**
 * Shared by the texture-store GPU pages (M4, T4.1): test images, a tiny BC7
 * encoder, reading a store layer back, and a CPU sampler that reads the store
 * the way hit shading's sampler does (trilinear, sRGB-decoded, repeating).
 *
 * Not a page itself (no index.html).
 */

import { STORE_LAYER_SIZE, STORE_MIP_LEVELS } from '../../../../packages/core/src/platform/textures/texture-store-wgsl.ts';

// ---------------------------------------------------------------------------
// Colour
// ---------------------------------------------------------------------------

/** The sRGB decoding curve (IEC 61966-2-1), one channel in 0..1. */
export function srgbToLinear(encoded: number): number {
    return encoded <= 0.04045 ? encoded / 12.92 : Math.pow((encoded + 0.055) / 1.055, 2.4);
}

export function linearToSrgb(linear: number): number {
    return linear <= 0.0031308 ? linear * 12.92 : 1.055 * Math.pow(linear, 1 / 2.4) - 0.055;
}

// ---------------------------------------------------------------------------
// Test images
// ---------------------------------------------------------------------------

/** A square RGBA8 image, top row first. */
export interface TestImage {
    readonly size: number;
    readonly pixels: Uint8Array<ArrayBuffer>;
}

/**
 * The colour of one 4×4 block of the sources test image. Every channel is
 * even, so a BC7 mode 6 block with p-bits 0 holds it exactly (see
 * {@link bc7SolidBlock}); the blocks differ from their neighbours in every
 * channel, so a wrong level, flip or offset shows.
 */
export function blockColour(blockX: number, blockY: number): [number, number, number, number] {
    const even = (value: number) => value & 0xfe;
    return [even(blockX * 4), even(blockY * 4), even(((blockX ^ blockY) * 37) & 0xff), even(128 + ((blockX + 3 * blockY) & 63) * 2)];
}

/** A size × size image of 4×4 blocks of `colour(blockX, blockY)`. */
export function blockImage(size: number, colour: (blockX: number, blockY: number) => readonly number[]): TestImage {
    const pixels = new Uint8Array(size * size * 4);
    for (let y = 0; y < size; y++) {
        for (let x = 0; x < size; x++) {
            pixels.set(colour(x >> 2, y >> 2), (y * size + x) * 4);
        }
    }
    return { size, pixels };
}

// ---------------------------------------------------------------------------
// BC7
// ---------------------------------------------------------------------------

/**
 * One BC7 block of a single colour, in mode 6 (one subset, RGBA endpoints of
 * 7 bits plus a shared p-bit each, 4-bit indices). Both endpoints are the
 * colour and every index is 0, so the block decodes to exactly the colour:
 * a weight-0 interpolation returns endpoint 0, (64·e0 + 32) >> 6 = e0.
 * Every channel must be even (p-bit 0).
 *
 * Bit layout (little-endian, bit 0 first; Khronos Data Format spec, BC7):
 * mode 6 is 6 zero bits then a 1; then R0 R1 G0 G1 B0 B1 A0 A1 (7 bits
 * each), P0, P1, and 63 index bits (the anchor index has 3 bits).
 */
export function bc7SolidBlock(colour: readonly number[]): Uint8Array {
    for (const channel of colour) if (channel & 1) throw new Error('bc7SolidBlock needs even channel values');
    const block = new Uint8Array(16);
    let bit = 0;
    const write = (value: number, bits: number) => {
        for (let index = 0; index < bits; index++, bit++) {
            if ((value >> index) & 1) block[bit >> 3] |= 1 << (bit & 7);
        }
    };
    write(1 << 6, 7);
    for (const channel of colour) {
        // Endpoint 0 and endpoint 1 of this channel: the colour's top 7 bits.
        write(channel >> 1, 7);
        write(channel >> 1, 7);
    }
    write(0, 2);
    // The indices stay 0: the block is all endpoint 0.
    return block;
}

/** A BC7 level of an image whose 4×4 blocks are each one colour (as {@link blockImage} makes them). */
export function bc7Level(image: TestImage): Uint8Array<ArrayBuffer> {
    const blocks = image.size / 4;
    const data = new Uint8Array(blocks * blocks * 16);
    for (let blockY = 0; blockY < blocks; blockY++) {
        for (let blockX = 0; blockX < blocks; blockX++) {
            const at = (blockY * 4 * image.size + blockX * 4) * 4;
            data.set(bc7SolidBlock(Array.from(image.pixels.subarray(at, at + 4))), (blockY * blocks + blockX) * 16);
        }
    }
    return data;
}

// ---------------------------------------------------------------------------
// Reading the store back
// ---------------------------------------------------------------------------

/** Every level of one layer of the store, as raw sRGB-encoded RGBA8 bytes (rows unpadded). */
export type LayerLevels = Uint8Array[];

export async function readLayer(device: GPUDevice, store: GPUTexture, layer: number): Promise<LayerLevels> {
    const levels: { offset: number; rowBytes: number; size: number }[] = [];
    let total = 0;
    // The store's own size and mips, so a smaller store (a quality preset's) reads back too.
    for (let level = 0; level < store.mipLevelCount; level++) {
        const size = store.width >> level;
        const rowBytes = Math.ceil((size * 4) / 256) * 256;
        levels.push({ offset: total, rowBytes, size });
        total += rowBytes * size;
    }
    const buffer = device.createBuffer({ label: 'texture store readback', size: total, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const encoder = device.createCommandEncoder({ label: 'texture store readback' });
    levels.forEach(({ offset, rowBytes, size }, level) => {
        encoder.copyTextureToBuffer(
            { texture: store, mipLevel: level, origin: { x: 0, y: 0, z: layer } },
            { buffer, offset, bytesPerRow: rowBytes, rowsPerImage: size },
            { width: size, height: size, depthOrArrayLayers: 1 },
        );
    });
    device.queue.submit([encoder.finish()]);
    await buffer.mapAsync(GPUMapMode.READ);
    const mapped = new Uint8Array(buffer.getMappedRange());
    const result = levels.map(({ offset, rowBytes, size }) => {
        const level = new Uint8Array(size * size * 4);
        for (let row = 0; row < size; row++) level.set(mapped.subarray(offset + row * rowBytes, offset + row * rowBytes + size * 4), row * size * 4);
        return level;
    });
    buffer.unmap();
    buffer.destroy();
    return result;
}

/** The largest difference between two byte arrays, and where. */
export function largestByteDifference(a: Uint8Array, b: Uint8Array): { difference: number; index: number } {
    let difference = 0;
    let index = -1;
    for (let at = 0; at < a.length; at++) {
        const delta = Math.abs(a[at] - b[at]);
        if (delta > difference) {
            difference = delta;
            index = at;
        }
    }
    return { difference, index };
}

// ---------------------------------------------------------------------------
// A CPU stand-in for hit shading's texture sampler
// ---------------------------------------------------------------------------

/**
 * Samples a layer as `textureSampleLevel` with hit shading's sampler does:
 * texels decoded from sRGB, bilinear within a level with repeat addressing,
 * and a linear blend between the two levels around `lod`.
 */
export class LayerSampler {
    private readonly linear: Float32Array[];

    constructor(levels: LayerLevels) {
        this.linear = levels.map((bytes) => {
            const values = new Float32Array(bytes.length);
            for (let index = 0; index < bytes.length; index++) {
                // Alpha (every fourth byte) is stored linear.
                values[index] = index % 4 === 3 ? bytes[index] / 255 : srgbToLinear(bytes[index] / 255);
            }
            return values;
        });
    }

    sample(u: number, v: number, lod: number): [number, number, number, number] {
        const clamped = Math.min(Math.max(lod, 0), STORE_MIP_LEVELS - 1);
        const lower = Math.floor(clamped);
        const upper = Math.min(lower + 1, STORE_MIP_LEVELS - 1);
        const blend = clamped - lower;
        const a = this.bilinear(lower, u, v);
        if (blend === 0) return a;
        const b = this.bilinear(upper, u, v);
        return [0, 1, 2, 3].map((channel) => a[channel] * (1 - blend) + b[channel] * blend) as [number, number, number, number];
    }

    private bilinear(level: number, u: number, v: number): [number, number, number, number] {
        const size = STORE_LAYER_SIZE >> level;
        const x = u * size - 0.5;
        const y = v * size - 0.5;
        const x0 = Math.floor(x);
        const y0 = Math.floor(y);
        const fx = x - x0;
        const fy = y - y0;
        const wrap = (value: number) => ((value % size) + size) % size;
        const texel = (tx: number, ty: number) => (wrap(ty) * size + wrap(tx)) * 4;
        const values = this.linear[level];
        const result: [number, number, number, number] = [0, 0, 0, 0];
        const corners: [number, number, number][] = [
            [x0, y0, (1 - fx) * (1 - fy)],
            [x0 + 1, y0, fx * (1 - fy)],
            [x0, y0 + 1, (1 - fx) * fy],
            [x0 + 1, y0 + 1, fx * fy],
        ];
        for (const [tx, ty, weight] of corners) {
            const at = texel(tx, ty);
            for (let channel = 0; channel < 4; channel++) result[channel] += values[at + channel] * weight;
        }
        return result;
    }
}
