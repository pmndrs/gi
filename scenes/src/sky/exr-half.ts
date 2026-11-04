/**
 * A small OpenEXR codec for the sky files: scanline, single part, channels
 * B, G, R as 16-bit floats ("half"), ZIP compression (16 lines per block) or
 * none. No engine, no Node APIs: it runs in the browser and in Node, so the
 * viewer, the platform's host and the generator all read the file the same way.
 *
 * Why OpenEXR half and not Radiance .hdr (RGBE): the three readers we need
 * disagree on RGBE. three.js's HDRLoader decodes a mantissa m as m / 255,
 * stb_image as m / 256 and Blender as (m + 0.5) / 256, so the same file reads
 * up to 0.4% apart, which is most of the ±0.5% reference tolerance. A half
 * float has one meaning everywhere (IEEE 754 binary16, about 0.05% steps), and
 * three.js (EXRLoader), Blender and `tools/compare` all read it. Half floats
 * also go straight into an `rgba16float` GPU texture with no conversion.
 *
 * Format reference: "OpenEXR File Layout" and "Technical Introduction"
 * (openexr.com). ZIP blocks hold the block's raw scanlines, byte-reordered
 * (even-indexed bytes first, then odd), delta-encoded byte by byte (each byte
 * minus the previous plus 128), then zlib-compressed. A block that would not
 * get smaller is stored raw, and the reader tells the two apart by size.
 *
 * Compression itself is passed in (`deflate`, `inflate`), so this module needs
 * neither `node:zlib` nor a bundled inflater: the generator passes
 * `zlib.deflateSync`, readers use the platform's `DecompressionStream`.
 */

export const EXR_MAGIC = 20000630;
const PIXEL_TYPE_HALF = 1;
export const COMPRESSION_NONE = 0;
export const COMPRESSION_ZIP = 3;
/** OpenEXR's ZIP compression packs 16 scanlines per block. */
const ZIP_LINES_PER_BLOCK = 16;
/** Channels are stored in alphabetical order. */
const CHANNEL_NAMES = ['B', 'G', 'R'] as const;
const BYTES_PER_HALF = 2;

/** An RGB image of half floats, row 0 at the top. */
export interface HalfRgbImage {
    width: number;
    height: number;
    /** IEEE 754 binary16 bit patterns, interleaved R, G, B; length width * height * 3. */
    halves: Uint16Array;
}

export type Deflate = (bytes: Uint8Array) => Uint8Array;
export type Inflate = (bytes: Uint8Array) => Uint8Array | Promise<Uint8Array>;

// ---------------------------------------------------------------------------
// Half floats
// ---------------------------------------------------------------------------

const float32Scratch = new Float32Array(1);
const uint32Scratch = new Uint32Array(float32Scratch.buffer);

/**
 * float32 to binary16 with round-to-nearest-even, the IEEE default, so a
 * value is never biased up or down on average. Values above the largest half
 * (65504) are an error rather than silently becoming infinity: a sky that
 * bright means a units mistake upstream.
 */
export function floatToHalf(value: number): number {
    if (Number.isNaN(value)) throw new Error('cannot store NaN in the sky file');
    float32Scratch[0] = value;
    const bits = uint32Scratch[0];
    const sign = (bits >>> 16) & 0x8000;
    const exponent = (bits >>> 23) & 0xff;
    const mantissa = bits & 0x7fffff;

    if (exponent === 0xff) throw new Error(`cannot store ${value} in the sky file`);
    // Re-bias the exponent from float32 (127) to half (15).
    const halfExponent = exponent - 127 + 15;
    if (halfExponent >= 0x1f) throw new Error(`${value} is larger than the largest half float (65504)`);

    if (halfExponent <= 0) {
        // Subnormal half (or zero): shift the mantissa, with its implicit 1, into place.
        if (halfExponent < -10) return sign;
        const fullMantissa = mantissa | 0x800000;
        const shift = 14 - halfExponent;
        const halfMantissa = fullMantissa >>> shift;
        const remainder = fullMantissa & ((1 << shift) - 1);
        const halfway = 1 << (shift - 1);
        const roundUp = remainder > halfway || (remainder === halfway && (halfMantissa & 1) === 1);
        return sign | (halfMantissa + (roundUp ? 1 : 0));
    }

    const halfMantissa = mantissa >>> 13;
    const remainder = mantissa & 0x1fff;
    const roundUp = remainder > 0x1000 || (remainder === 0x1000 && (halfMantissa & 1) === 1);
    // A carry out of the mantissa correctly bumps the exponent.
    const rounded = ((halfExponent << 10) | halfMantissa) + (roundUp ? 1 : 0);
    if (rounded >= 0x7c00) throw new Error(`${value} is larger than the largest half float (65504)`);
    return sign | rounded;
}

export function halfToFloat(half: number): number {
    const sign = (half & 0x8000) !== 0 ? -1 : 1;
    const exponent = (half >> 10) & 0x1f;
    const fraction = half & 0x3ff;
    if (exponent === 0) return sign * 2 ** -14 * (fraction / 1024);
    if (exponent === 31) return fraction === 0 ? sign * Infinity : NaN;
    return sign * 2 ** (exponent - 15) * (1 + fraction / 1024);
}

// ---------------------------------------------------------------------------
// Writing
// ---------------------------------------------------------------------------

class ByteWriter {
    private bytes = new Uint8Array(1024);
    private view = new DataView(this.bytes.buffer);
    length = 0;

    private reserve(count: number): void {
        if (this.length + count <= this.bytes.length) return;
        const grown = new Uint8Array(Math.max(this.bytes.length * 2, this.length + count));
        grown.set(this.bytes.subarray(0, this.length));
        this.bytes = grown;
        this.view = new DataView(grown.buffer);
    }

    uint8(value: number): void {
        this.reserve(1);
        this.bytes[this.length++] = value;
    }

    int32(value: number): void {
        this.reserve(4);
        this.view.setInt32(this.length, value, true);
        this.length += 4;
    }

    float32(value: number): void {
        this.reserve(4);
        this.view.setFloat32(this.length, value, true);
        this.length += 4;
    }

    uint64(value: number): void {
        this.reserve(8);
        this.view.setBigUint64(this.length, BigInt(value), true);
        this.length += 8;
    }

    cString(text: string): void {
        for (let i = 0; i < text.length; i++) this.uint8(text.charCodeAt(i));
        this.uint8(0);
    }

    raw(data: Uint8Array): void {
        this.reserve(data.length);
        this.bytes.set(data, this.length);
        this.length += data.length;
    }

    /** Writes an attribute: name, type name, byte size, then the value written by `body`. */
    attribute(name: string, type: string, body: (writer: ByteWriter) => void): void {
        const value = new ByteWriter();
        body(value);
        this.cString(name);
        this.cString(type);
        this.int32(value.length);
        this.raw(value.result());
    }

    result(): Uint8Array {
        return this.bytes.slice(0, this.length);
    }
}

/** One scanline block in the file's raw layout: for each line, all B halves, then all G, then all R. */
function packBlock(image: HalfRgbImage, firstLine: number, lineCount: number): Uint8Array {
    const { width } = image;
    const block = new Uint8Array(lineCount * CHANNEL_NAMES.length * width * BYTES_PER_HALF);
    const view = new DataView(block.buffer);
    let offset = 0;
    for (let line = firstLine; line < firstLine + lineCount; line++) {
        for (const channel of CHANNEL_NAMES) {
            // Interleaved storage is R, G, B, so B is component 2.
            const component = channel === 'R' ? 0 : channel === 'G' ? 1 : 2;
            for (let x = 0; x < width; x++) {
                view.setUint16(offset, image.halves[(line * width + x) * 3 + component], true);
                offset += BYTES_PER_HALF;
            }
        }
    }
    return block;
}

/** OpenEXR's ZIP pre-pass: split even and odd bytes, then byte-wise deltas. */
function zipPrepare(raw: Uint8Array): Uint8Array {
    const reordered = new Uint8Array(raw.length);
    const secondHalf = (raw.length + 1) >> 1;
    for (let i = 0; i < raw.length; i++) {
        const target = (i & 1) === 0 ? i >> 1 : secondHalf + (i >> 1);
        reordered[target] = raw[i];
    }
    const predicted = new Uint8Array(raw.length);
    predicted[0] = reordered[0];
    for (let i = 1; i < raw.length; i++) predicted[i] = (reordered[i] - reordered[i - 1] + 128) & 0xff;
    return predicted;
}

/** The inverse of {@link zipPrepare}. */
function zipRestore(prepared: Uint8Array): Uint8Array {
    const reordered = new Uint8Array(prepared.length);
    reordered[0] = prepared[0];
    for (let i = 1; i < prepared.length; i++) reordered[i] = (reordered[i - 1] + prepared[i] - 128) & 0xff;
    const raw = new Uint8Array(prepared.length);
    const secondHalf = (prepared.length + 1) >> 1;
    for (let i = 0; i < prepared.length; i++) {
        raw[i] = (i & 1) === 0 ? reordered[i >> 1] : reordered[secondHalf + (i >> 1)];
    }
    return raw;
}

/**
 * Encodes an image. With `deflate` the file is ZIP-compressed (smaller files to
 * commit); without it, uncompressed.
 */
export function encodeHalfExr(image: HalfRgbImage, deflate?: Deflate): Uint8Array {
    const { width, height } = image;
    if (image.halves.length !== width * height * 3) throw new Error('halves must hold width * height * 3 values');
    const compression = deflate ? COMPRESSION_ZIP : COMPRESSION_NONE;
    const linesPerBlock = deflate ? ZIP_LINES_PER_BLOCK : 1;

    const header = new ByteWriter();
    header.int32(EXR_MAGIC);
    header.int32(2); // version 2, single-part scanline, no flags
    header.attribute('channels', 'chlist', (w) => {
        for (const name of CHANNEL_NAMES) {
            w.cString(name);
            w.int32(PIXEL_TYPE_HALF);
            w.uint8(0); // pLinear
            w.uint8(0); // reserved
            w.uint8(0);
            w.uint8(0);
            w.int32(1); // x sampling
            w.int32(1); // y sampling
        }
        w.uint8(0);
    });
    header.attribute('compression', 'compression', (w) => w.uint8(compression));
    const box = (w: ByteWriter) => {
        w.int32(0);
        w.int32(0);
        w.int32(width - 1);
        w.int32(height - 1);
    };
    header.attribute('dataWindow', 'box2i', box);
    header.attribute('displayWindow', 'box2i', box);
    header.attribute('lineOrder', 'lineOrder', (w) => w.uint8(0)); // increasing y
    header.attribute('pixelAspectRatio', 'float', (w) => w.float32(1));
    header.attribute('screenWindowCenter', 'v2f', (w) => {
        w.float32(0);
        w.float32(0);
    });
    header.attribute('screenWindowWidth', 'float', (w) => w.float32(1));
    header.uint8(0); // end of header

    const blocks: Uint8Array[] = [];
    for (let firstLine = 0; firstLine < height; firstLine += linesPerBlock) {
        const lineCount = Math.min(linesPerBlock, height - firstLine);
        const raw = packBlock(image, firstLine, lineCount);
        let data = raw;
        if (deflate) {
            const compressed = deflate(zipPrepare(raw));
            if (compressed.length < raw.length) data = compressed;
        }
        const chunk = new ByteWriter();
        chunk.int32(firstLine);
        chunk.int32(data.length);
        chunk.raw(data);
        blocks.push(chunk.result());
    }

    const offsetTableBytes = blocks.length * 8;
    const file = new ByteWriter();
    file.raw(header.result());
    let offset = header.length + offsetTableBytes;
    for (const block of blocks) {
        file.uint64(offset);
        offset += block.length;
    }
    for (const block of blocks) file.raw(block);
    return file.result();
}

// ---------------------------------------------------------------------------
// Reading
// ---------------------------------------------------------------------------

/** zlib inflate with the platform's `DecompressionStream` (browsers, Node 18+). */
export async function inflateWithDecompressionStream(bytes: Uint8Array): Promise<Uint8Array> {
    const stream = new Blob([bytes as BlobPart]).stream().pipeThrough(new DecompressionStream('deflate'));
    return new Uint8Array(await new Response(stream).arrayBuffer());
}

interface ParsedHeader {
    width: number;
    height: number;
    compression: number;
    /** dataWindow.yMin: chunks store absolute line numbers, so row = line - yMin. */
    firstLineY: number;
    /** Byte offset of the first offset-table entry. */
    end: number;
}

function parseHeader(file: Uint8Array): ParsedHeader {
    const view = new DataView(file.buffer, file.byteOffset, file.byteLength);
    if (file.length < 8 || view.getInt32(0, true) !== EXR_MAGIC) throw new Error('not an OpenEXR file');
    // Version field flags: 0x200 tiled, 0x1000 deep data, 0x2000 multi-part.
    const versionFlags = view.getInt32(4, true);
    if ((versionFlags & 0x200) !== 0 || (versionFlags & 0x1000) !== 0 || (versionFlags & 0x2000) !== 0) {
        throw new Error('only single-part scanline OpenEXR files are supported (this one is tiled, deep or multi-part)');
    }
    let offset = 8;
    const cString = (): string => {
        let text = '';
        while (offset < file.length && file[offset] !== 0) text += String.fromCharCode(file[offset++]);
        offset++;
        return text;
    };

    let channels: { name: string; type: number }[] | null = null;
    let compression = -1;
    let dataWindow: number[] | null = null;
    for (;;) {
        const name = cString();
        if (name === '') break;
        cString(); // type name
        const size = view.getInt32(offset, true);
        offset += 4;
        const valueStart = offset;
        if (name === 'channels') {
            channels = [];
            let channelOffset = valueStart;
            while (file[channelOffset] !== 0) {
                let channelName = '';
                while (file[channelOffset] !== 0) channelName += String.fromCharCode(file[channelOffset++]);
                channelOffset++;
                channels.push({ name: channelName, type: view.getInt32(channelOffset, true) });
                channelOffset += 16;
            }
        } else if (name === 'compression') {
            compression = file[valueStart];
        } else if (name === 'dataWindow') {
            dataWindow = [0, 1, 2, 3].map((i) => view.getInt32(valueStart + i * 4, true));
        }
        offset = valueStart + size;
    }

    const names = channels?.map((channel) => channel.name).join(',');
    if (names !== CHANNEL_NAMES.join(',') || !channels?.every((channel) => channel.type === PIXEL_TYPE_HALF)) {
        throw new Error(`the sky file must have exactly the channels B, G, R as half floats (found ${names ?? 'none'})`);
    }
    if (compression !== COMPRESSION_NONE && compression !== COMPRESSION_ZIP) {
        throw new Error(`unsupported OpenEXR compression ${compression} (the sky reader handles none and ZIP)`);
    }
    if (!dataWindow) throw new Error('the OpenEXR header has no dataWindow');
    return { width: dataWindow[2] - dataWindow[0] + 1, height: dataWindow[3] - dataWindow[1] + 1, compression, firstLineY: dataWindow[1], end: offset };
}

/**
 * Decodes a sky file written by {@link encodeHalfExr} (or any single-part
 * scanline EXR with half B, G, R and no or ZIP compression).
 */
export async function decodeHalfExr(file: Uint8Array, inflate: Inflate = inflateWithDecompressionStream): Promise<HalfRgbImage> {
    const header = parseHeader(file);
    const { width, height, compression } = header;
    const view = new DataView(file.buffer, file.byteOffset, file.byteLength);
    const linesPerBlock = compression === COMPRESSION_ZIP ? ZIP_LINES_PER_BLOCK : 1;
    const blockCount = Math.ceil(height / linesPerBlock);
    const halves = new Uint16Array(width * height * 3);
    const lineBytes = CHANNEL_NAMES.length * width * BYTES_PER_HALF;

    for (let block = 0; block < blockCount; block++) {
        const chunkOffset = Number(view.getBigUint64(header.end + block * 8, true));
        // The chunk names its first scanline in data-window coordinates, which need not start at 0.
        const firstLine = view.getInt32(chunkOffset, true) - header.firstLineY;
        const dataSize = view.getInt32(chunkOffset + 4, true);
        if (firstLine < 0 || firstLine >= height) throw new Error(`block ${block} starts at line ${firstLine + header.firstLineY}, outside the data window`);
        const lineCount = Math.min(linesPerBlock, height - firstLine);
        const rawSize = lineCount * lineBytes;
        let data = file.subarray(chunkOffset + 8, chunkOffset + 8 + dataSize);
        if (dataSize < rawSize) data = zipRestore(await inflate(data));
        if (data.length !== rawSize) throw new Error(`block at line ${firstLine} holds ${data.length} bytes, expected ${rawSize}`);

        const blockView = new DataView(data.buffer, data.byteOffset, data.byteLength);
        for (let line = 0; line < lineCount; line++) {
            for (let channelIndex = 0; channelIndex < CHANNEL_NAMES.length; channelIndex++) {
                const channel = CHANNEL_NAMES[channelIndex];
                const component = channel === 'R' ? 0 : channel === 'G' ? 1 : 2;
                const channelStart = line * lineBytes + channelIndex * width * BYTES_PER_HALF;
                for (let x = 0; x < width; x++) {
                    halves[((firstLine + line) * width + x) * 3 + component] = blockView.getUint16(channelStart + x * BYTES_PER_HALF, true);
                }
            }
        }
    }
    return { width, height, halves };
}

/** Decoded halves as float32 RGB, row 0 at the top. */
export function halfImageToFloats(image: HalfRgbImage): Float32Array {
    const floats = new Float32Array(image.halves.length);
    for (let i = 0; i < floats.length; i++) floats[i] = halfToFloat(image.halves[i]);
    return floats;
}
