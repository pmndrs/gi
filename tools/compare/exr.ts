/**
 * A small OpenEXR reader and writer, enough for reference images.
 *
 * Reads: single-part scanline files with no compression, ZIPS (1 line per
 * block) or ZIP (16 lines per block), channels of HALF or FLOAT. That is what
 * Blender's File Output node (ZIP, 32-bit) and our own writer produce. Other
 * compressions (PIZ, DWA, ...) and tiled or deep files are rejected with a
 * message that says what was found, rather than misread.
 *
 * Writes: uncompressed 32-bit float RGB, scanline. Big, but simple, and every
 * EXR reader understands it.
 *
 * Pixels are returned as interleaved RGB floats, row 0 at the top (EXR's own
 * order). Alpha is dropped. A grey image with only a `Y` channel is expanded
 * to RGB.
 */

import { inflateSync } from 'node:zlib';

export interface FloatImage {
    width: number;
    height: number;
    /** Interleaved R, G, B; length width * height * 3. */
    rgb: Float32Array;
}

const EXR_MAGIC = 20000630;
const PIXEL_TYPE_HALF = 1;
const PIXEL_TYPE_FLOAT = 2;
const COMPRESSION_NONE = 0;
const COMPRESSION_ZIPS = 2;
const COMPRESSION_ZIP = 3;
const COMPRESSION_NAMES = ['NONE', 'RLE', 'ZIPS', 'ZIP', 'PIZ', 'PXR24', 'B44', 'B44A', 'DWAA', 'DWAB'];

interface ChannelInfo {
    name: string;
    pixelType: number;
    bytesPerSample: number;
}

function halfToFloat(half: number): number {
    const sign = (half & 0x8000) !== 0 ? -1 : 1;
    const exponent = (half >> 10) & 0x1f;
    const fraction = half & 0x3ff;
    if (exponent === 0) return sign * 2 ** -14 * (fraction / 1024);
    if (exponent === 31) return fraction === 0 ? sign * Infinity : NaN;
    return sign * 2 ** (exponent - 15) * (1 + fraction / 1024);
}

class HeaderReader {
    private offset = 0;
    private readonly bytes: Uint8Array;
    private readonly view: DataView;

    constructor(bytes: Uint8Array) {
        this.bytes = bytes;
        this.view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    }

    get position(): number {
        return this.offset;
    }

    int32(): number {
        const value = this.view.getInt32(this.offset, true);
        this.offset += 4;
        return value;
    }

    uint8(): number {
        return this.bytes[this.offset++];
    }

    skip(count: number): void {
        this.offset += count;
    }

    /** A null-terminated string; returns '' for a lone terminator. */
    cString(): string {
        const start = this.offset;
        while (this.bytes[this.offset] !== 0) this.offset++;
        const text = new TextDecoder().decode(this.bytes.subarray(start, this.offset));
        this.offset++;
        return text;
    }
}

export function readExr(file: Uint8Array): FloatImage {
    const view = new DataView(file.buffer, file.byteOffset, file.byteLength);
    if (file.byteLength < 8 || view.getInt32(0, true) !== EXR_MAGIC) throw new Error('not an OpenEXR file (bad magic number)');
    const versionFlags = view.getInt32(4, true);
    if (versionFlags & 0x200) throw new Error('tiled OpenEXR files are not supported');
    if (versionFlags & 0x800 || versionFlags & 0x1000 || versionFlags & 0x2000) {
        throw new Error('deep or multi-part OpenEXR files are not supported');
    }

    const header = new HeaderReader(file);
    header.skip(8);
    let channels: ChannelInfo[] = [];
    let compression = -1;
    let dataWindow: [number, number, number, number] | null = null;

    for (;;) {
        const name = header.cString();
        if (name === '') break;
        header.cString(); // attribute type
        const size = header.int32();
        const valueStart = header.position;
        if (name === 'channels') {
            channels = [];
            for (;;) {
                const channelName = header.cString();
                if (channelName === '') break;
                const pixelType = header.int32();
                header.skip(4); // pLinear + 3 reserved bytes
                const xSampling = header.int32();
                const ySampling = header.int32();
                if (xSampling !== 1 || ySampling !== 1) throw new Error(`channel ${channelName} is sub-sampled, which is not supported`);
                if (pixelType !== PIXEL_TYPE_HALF && pixelType !== PIXEL_TYPE_FLOAT) {
                    throw new Error(`channel ${channelName} has pixel type ${pixelType}; only HALF and FLOAT are supported`);
                }
                channels.push({ name: channelName, pixelType, bytesPerSample: pixelType === PIXEL_TYPE_HALF ? 2 : 4 });
            }
        } else if (name === 'compression') {
            compression = file[valueStart];
        } else if (name === 'dataWindow') {
            dataWindow = [0, 1, 2, 3].map((index) => view.getInt32(valueStart + index * 4, true)) as [number, number, number, number];
        }
        header.skip(valueStart + size - header.position);
    }

    if (!dataWindow || channels.length === 0 || compression < 0) throw new Error('OpenEXR header is missing channels, compression or dataWindow');
    if (compression !== COMPRESSION_NONE && compression !== COMPRESSION_ZIPS && compression !== COMPRESSION_ZIP) {
        throw new Error(`OpenEXR compression ${COMPRESSION_NAMES[compression] ?? compression} is not supported; re-save with NONE or ZIP`);
    }

    const [xMin, yMin, xMax, yMax] = dataWindow;
    const width = xMax - xMin + 1;
    const height = yMax - yMin + 1;
    const linesPerBlock = compression === COMPRESSION_ZIP ? 16 : 1;
    const blockCount = Math.ceil(height / linesPerBlock);
    const tableStart = header.position;
    const bytesPerPixelAllChannels = channels.reduce((total, channel) => total + channel.bytesPerSample, 0);

    const pick = (...names: string[]): number => channels.findIndex((channel) => names.includes(channel.name.split('.').pop() ?? channel.name));
    let redIndex = pick('R');
    let greenIndex = pick('G');
    let blueIndex = pick('B');
    if (redIndex < 0 || greenIndex < 0 || blueIndex < 0) {
        const grey = pick('Y');
        if (grey < 0) throw new Error(`OpenEXR has no R, G, B or Y channel (found ${channels.map((c) => c.name).join(', ')})`);
        redIndex = greenIndex = blueIndex = grey;
    }

    // Byte offset of each channel inside one scanline, since a scanline stores whole channels one after another.
    const channelOffsets: number[] = [];
    let running = 0;
    for (const channel of channels) {
        channelOffsets.push(running);
        running += width * channel.bytesPerSample;
    }

    const rgb = new Float32Array(width * height * 3);
    const wantedChannels = [redIndex, greenIndex, blueIndex];

    for (let block = 0; block < blockCount; block++) {
        const blockOffset = Number(view.getBigUint64(tableStart + block * 8, true));
        const firstLine = view.getInt32(blockOffset, true) - yMin;
        const storedSize = view.getInt32(blockOffset + 4, true);
        let payload = file.subarray(blockOffset + 8, blockOffset + 8 + storedSize);
        const lines = Math.min(linesPerBlock, height - firstLine);
        const expectedSize = lines * width * bytesPerPixelAllChannels;
        if (compression !== COMPRESSION_NONE && storedSize < expectedSize) payload = decodeZip(payload, expectedSize);
        const lineView = new DataView(payload.buffer, payload.byteOffset, payload.byteLength);

        for (let line = 0; line < lines; line++) {
            const lineStart = line * width * bytesPerPixelAllChannels;
            const y = firstLine + line;
            for (let slot = 0; slot < 3; slot++) {
                const channel = channels[wantedChannels[slot]];
                const channelStart = lineStart + channelOffsets[wantedChannels[slot]];
                for (let x = 0; x < width; x++) {
                    const sampleOffset = channelStart + x * channel.bytesPerSample;
                    rgb[(y * width + x) * 3 + slot] =
                        channel.pixelType === PIXEL_TYPE_FLOAT ? lineView.getFloat32(sampleOffset, true) : halfToFloat(lineView.getUint16(sampleOffset, true));
                }
            }
        }
    }
    return { width, height, rgb };
}

/**
 * EXR's ZIP codec: deflate, over bytes that were first split into two halves
 * (even-indexed bytes, then odd-indexed) and delta-coded with an offset of 128.
 */
function decodeZip(compressed: Uint8Array, expectedSize: number): Uint8Array {
    const delta = new Uint8Array(inflateSync(compressed));
    if (delta.length !== expectedSize) throw new Error(`ZIP block inflated to ${delta.length} bytes, expected ${expectedSize}`);
    for (let index = 1; index < delta.length; index++) {
        delta[index] = (delta[index - 1] + delta[index] - 128) & 0xff;
    }
    const out = new Uint8Array(expectedSize);
    const half = Math.ceil(expectedSize / 2);
    for (let index = 0; index < half; index++) {
        out[2 * index] = delta[index];
        if (2 * index + 1 < expectedSize) out[2 * index + 1] = delta[half + index];
    }
    return out;
}

/** Writes an uncompressed 32-bit float RGB scanline EXR. */
export function writeExr(image: FloatImage): Uint8Array {
    const { width, height, rgb } = image;
    const text = new TextEncoder();
    const parts: number[] = [];
    const pushString = (value: string): void => {
        parts.push(...text.encode(value), 0);
    };
    const pushInt32 = (value: number): void => {
        const bytes = new Uint8Array(4);
        new DataView(bytes.buffer).setInt32(0, value, true);
        parts.push(...bytes);
    };
    const pushAttribute = (name: string, type: string, payload: number[]): void => {
        pushString(name);
        pushString(type);
        pushInt32(payload.length);
        parts.push(...payload);
    };

    pushInt32(EXR_MAGIC);
    pushInt32(2); // version 2, no flags

    const channelPayload: number[] = [];
    for (const name of ['B', 'G', 'R']) {
        // Channels must be sorted by name; each is FLOAT, pLinear 0, sampling 1x1.
        channelPayload.push(...text.encode(name), 0, PIXEL_TYPE_FLOAT, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0);
    }
    channelPayload.push(0);
    pushAttribute('channels', 'chlist', channelPayload);
    pushAttribute('compression', 'compression', [COMPRESSION_NONE]);

    const box = new Uint8Array(16);
    const boxView = new DataView(box.buffer);
    boxView.setInt32(8, width - 1, true);
    boxView.setInt32(12, height - 1, true);
    pushAttribute('dataWindow', 'box2i', [...box]);
    pushAttribute('displayWindow', 'box2i', [...box]);
    pushAttribute('lineOrder', 'lineOrder', [0]);
    const one = new Uint8Array(4);
    new DataView(one.buffer).setFloat32(0, 1, true);
    pushAttribute('pixelAspectRatio', 'float', [...one]);
    pushAttribute('screenWindowCenter', 'v2f', [0, 0, 0, 0, 0, 0, 0, 0]);
    pushAttribute('screenWindowWidth', 'float', [...one]);
    parts.push(0); // end of header

    const headerBytes = Uint8Array.from(parts);
    const lineBytes = width * 3 * 4;
    const blockBytes = 8 + lineBytes;
    const tableBytes = height * 8;
    const out = new Uint8Array(headerBytes.length + tableBytes + height * blockBytes);
    const view = new DataView(out.buffer);
    out.set(headerBytes, 0);

    let blockStart = headerBytes.length + tableBytes;
    for (let y = 0; y < height; y++) {
        view.setBigUint64(headerBytes.length + y * 8, BigInt(blockStart), true);
        view.setInt32(blockStart, y, true);
        view.setInt32(blockStart + 4, lineBytes, true);
        // Stored channel order is B, G, R, one whole row each.
        for (let channel = 0; channel < 3; channel++) {
            const source = 2 - channel;
            for (let x = 0; x < width; x++) {
                view.setFloat32(blockStart + 8 + (channel * width + x) * 4, rgb[(y * width + x) * 3 + source], true);
            }
        }
        blockStart += blockBytes;
    }
    return out;
}
