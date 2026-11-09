/**
 * A minimal PNG writer for the procedural textures, so the GLB exporter can
 * embed them (glTF images are PNG or JPEG). No dependencies and no Node
 * APIs, so it runs anywhere the exporter does.
 *
 * The image data is stored without compression ("stored" deflate blocks):
 * the textures are small (a 256² RGBA texture is 257 kB this way), and a
 * stored stream needs no compressor, only the zlib framing and checksums.
 * Every PNG reader accepts it.
 *
 * Format: 8-bit RGBA (colour type 6), no interlacing, filter type 0 (None) on
 * every row. Row 0 of the data is the image's top row, which is v = 0 in glTF.
 */

const PNG_SIGNATURE = [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a];
const COLOUR_TYPE_RGBA = 6;
const BIT_DEPTH = 8;
/** The largest payload of one stored deflate block. */
const STORED_BLOCK_MAX = 0xffff;

/** CRC-32 (ISO 3309, polynomial 0xedb88320), as PNG chunks use. */
const CRC_TABLE = (() => {
    const table = new Uint32Array(256);
    for (let n = 0; n < 256; n++) {
        let c = n;
        for (let bit = 0; bit < 8; bit++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
        table[n] = c >>> 0;
    }
    return table;
})();

export function crc32(bytes: Uint8Array): number {
    let crc = 0xffffffff;
    for (const byte of bytes) crc = CRC_TABLE[(crc ^ byte) & 0xff] ^ (crc >>> 8);
    return (crc ^ 0xffffffff) >>> 0;
}

/** Adler-32, the checksum at the end of a zlib stream. */
function adler32(bytes: Uint8Array): number {
    const modulus = 65521;
    let a = 1;
    let b = 0;
    for (const byte of bytes) {
        a = (a + byte) % modulus;
        b = (b + a) % modulus;
    }
    return ((b << 16) | a) >>> 0;
}

/** A zlib stream (RFC 1950) of stored deflate blocks (RFC 1951 §3.2.4) holding `data` as it is. */
function zlibStored(data: Uint8Array): Uint8Array {
    const blocks = Math.max(1, Math.ceil(data.length / STORED_BLOCK_MAX));
    const out = new Uint8Array(2 + blocks * 5 + data.length + 4);
    // CMF 0x78: deflate, 32 kB window. FLG 0x01: no dictionary, and (0x78 * 256 + 0x01) % 31 == 0.
    out[0] = 0x78;
    out[1] = 0x01;
    let cursor = 2;
    for (let block = 0; block < blocks; block++) {
        const start = block * STORED_BLOCK_MAX;
        const length = Math.min(STORED_BLOCK_MAX, data.length - start);
        // Block header: BFINAL in bit 0, BTYPE 00 (stored); the rest of the byte is padding.
        out[cursor++] = block === blocks - 1 ? 1 : 0;
        out[cursor++] = length & 0xff;
        out[cursor++] = length >>> 8;
        out[cursor++] = ~length & 0xff;
        out[cursor++] = (~length >>> 8) & 0xff;
        out.set(data.subarray(start, start + length), cursor);
        cursor += length;
    }
    new DataView(out.buffer).setUint32(cursor, adler32(data), false);
    return out;
}

function chunk(type: string, data: Uint8Array): Uint8Array {
    const out = new Uint8Array(12 + data.length);
    const view = new DataView(out.buffer);
    view.setUint32(0, data.length, false);
    for (let i = 0; i < 4; i++) out[4 + i] = type.charCodeAt(i);
    out.set(data, 8);
    view.setUint32(8 + data.length, crc32(out.subarray(4, 8 + data.length)), false);
    return out;
}

/** Encodes `width * height` RGBA texels (row 0 at the top) as a PNG file. */
export function encodePng(width: number, height: number, rgba: Uint8Array): Uint8Array {
    if (rgba.length !== width * height * 4) throw new Error(`encodePng: ${rgba.length} bytes is not ${width} x ${height} RGBA`);

    const header = new Uint8Array(13);
    const headerView = new DataView(header.buffer);
    headerView.setUint32(0, width, false);
    headerView.setUint32(4, height, false);
    header[8] = BIT_DEPTH;
    header[9] = COLOUR_TYPE_RGBA;
    // header[10..12]: compression 0 (deflate), filter method 0, interlace 0 (none).

    // Each row starts with its filter type byte, 0 = None.
    const rowBytes = width * 4;
    const filtered = new Uint8Array(height * (rowBytes + 1));
    for (let row = 0; row < height; row++) {
        filtered.set(rgba.subarray(row * rowBytes, (row + 1) * rowBytes), row * (rowBytes + 1) + 1);
    }

    const parts = [Uint8Array.from(PNG_SIGNATURE), chunk('IHDR', header), chunk('IDAT', zlibStored(filtered)), chunk('IEND', new Uint8Array(0))];
    const out = new Uint8Array(parts.reduce((total, part) => total + part.length, 0));
    let cursor = 0;
    for (const part of parts) {
        out.set(part, cursor);
        cursor += part.length;
    }
    return out;
}
