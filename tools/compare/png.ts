/**
 * A tiny PNG encoder (8-bit RGB, no interlace) and the colour helpers the
 * comparison images need. Node has zlib, so no dependency is needed.
 */

import { crc32, deflateSync, inflateSync } from 'node:zlib';

export function encodePng(width: number, height: number, rgb8: Uint8Array): Uint8Array {
    if (rgb8.length !== width * height * 3) throw new Error('encodePng: pixel array does not match width x height x 3');
    const rowBytes = width * 3;
    // Each scanline starts with a filter-type byte; 0 means "no filter".
    const raw = new Uint8Array(height * (rowBytes + 1));
    for (let y = 0; y < height; y++) {
        raw.set(rgb8.subarray(y * rowBytes, (y + 1) * rowBytes), y * (rowBytes + 1) + 1);
    }

    const chunk = (type: string, data: Uint8Array): Uint8Array => {
        const out = new Uint8Array(12 + data.length);
        const view = new DataView(out.buffer);
        view.setUint32(0, data.length);
        for (let index = 0; index < 4; index++) out[4 + index] = type.charCodeAt(index);
        out.set(data, 8);
        view.setUint32(8 + data.length, crc32(out.subarray(4, 8 + data.length)));
        return out;
    };

    const header = new Uint8Array(13);
    const headerView = new DataView(header.buffer);
    headerView.setUint32(0, width);
    headerView.setUint32(4, height);
    header[8] = 8; // bit depth
    header[9] = 2; // colour type: RGB

    const pieces = [
        Uint8Array.of(0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a),
        chunk('IHDR', header),
        chunk('IDAT', deflateSync(raw)),
        chunk('IEND', new Uint8Array(0)),
    ];
    const out = new Uint8Array(pieces.reduce((total, piece) => total + piece.length, 0));
    let cursor = 0;
    for (const piece of pieces) {
        out.set(piece, cursor);
        cursor += piece.length;
    }
    return out;
}

/** Minimal PNG decoder for the 8-bit RGB / RGBA, non-interlaced files FLIP writes. */
export function decodePng(file: Uint8Array): { width: number; height: number; rgb8: Uint8Array } {
    const view = new DataView(file.buffer, file.byteOffset, file.byteLength);
    let cursor = 8;
    let width = 0;
    let height = 0;
    let channels = 0;
    const idat: Uint8Array[] = [];
    while (cursor < file.length) {
        const length = view.getUint32(cursor);
        const type = String.fromCharCode(...file.subarray(cursor + 4, cursor + 8));
        const data = file.subarray(cursor + 8, cursor + 8 + length);
        if (type === 'IHDR') {
            width = view.getUint32(cursor + 8);
            height = view.getUint32(cursor + 12);
            const bitDepth = data[8];
            const colourType = data[9];
            if (bitDepth !== 8 || (colourType !== 2 && colourType !== 6) || data[12] !== 0) {
                throw new Error(`decodePng: only 8-bit RGB/RGBA non-interlaced PNGs are supported (got depth ${bitDepth}, type ${colourType})`);
            }
            channels = colourType === 2 ? 3 : 4;
        } else if (type === 'IDAT') {
            idat.push(data);
        }
        cursor += 12 + length;
    }
    const compressed = new Uint8Array(idat.reduce((total, part) => total + part.length, 0));
    let offset = 0;
    for (const part of idat) {
        compressed.set(part, offset);
        offset += part.length;
    }
    const raw = new Uint8Array(inflateSync(compressed));
    const stride = width * channels;
    const rgb8 = new Uint8Array(width * height * 3);
    let previous = new Uint8Array(stride);
    for (let y = 0; y < height; y++) {
        const filter = raw[y * (stride + 1)];
        const line = raw.slice(y * (stride + 1) + 1, (y + 1) * (stride + 1));
        for (let index = 0; index < stride; index++) {
            const left = index >= channels ? line[index - channels] : 0;
            const up = previous[index];
            const upLeft = index >= channels ? previous[index - channels] : 0;
            let predictor = 0;
            if (filter === 1) predictor = left;
            else if (filter === 2) predictor = up;
            else if (filter === 3) predictor = (left + up) >> 1;
            else if (filter === 4) predictor = paeth(left, up, upLeft);
            line[index] = (line[index] + predictor) & 0xff;
        }
        for (let x = 0; x < width; x++) {
            for (let channel = 0; channel < 3; channel++) rgb8[(y * width + x) * 3 + channel] = line[x * channels + channel];
        }
        previous = line;
    }
    return { width, height, rgb8 };
}

function paeth(left: number, up: number, upLeft: number): number {
    const estimate = left + up - upLeft;
    const distanceLeft = Math.abs(estimate - left);
    const distanceUp = Math.abs(estimate - up);
    const distanceUpLeft = Math.abs(estimate - upLeft);
    if (distanceLeft <= distanceUp && distanceLeft <= distanceUpLeft) return left;
    return distanceUp <= distanceUpLeft ? up : upLeft;
}

/** Linear radiance to an 8-bit sRGB value, clamped. */
export function linearToSrgb8(value: number): number {
    const clamped = Number.isFinite(value) ? Math.min(Math.max(value, 0), 1) : 0;
    const encoded = clamped <= 0.0031308 ? clamped * 12.92 : 1.055 * clamped ** (1 / 2.4) - 0.055;
    return Math.round(encoded * 255);
}

/** An exposure-scaled, clamped sRGB version of a linear HDR image, for looking at. */
export function toDisplayRgb8(rgb: Float32Array, exposureStops: number): Uint8Array {
    const scale = 2 ** exposureStops;
    const out = new Uint8Array(rgb.length);
    for (let index = 0; index < rgb.length; index++) out[index] = linearToSrgb8(rgb[index] * scale);
    return out;
}

/** Control points of the "magma" colour map (matplotlib), 0 = black, 1 = pale yellow. */
const MAGMA_STOPS: ReadonlyArray<readonly [number, number, number]> = [
    [0, 0, 4],
    [28, 16, 68],
    [79, 18, 123],
    [129, 37, 129],
    [181, 54, 122],
    [229, 80, 100],
    [251, 135, 97],
    [254, 194, 135],
    [252, 253, 191],
];

/** Maps 0..1 to a magma colour; values outside are clamped. */
export function magma(value: number): [number, number, number] {
    const clamped = Math.min(Math.max(Number.isFinite(value) ? value : 1, 0), 1);
    const position = clamped * (MAGMA_STOPS.length - 1);
    const lower = Math.min(Math.floor(position), MAGMA_STOPS.length - 2);
    const fraction = position - lower;
    const a = MAGMA_STOPS[lower];
    const b = MAGMA_STOPS[lower + 1];
    return [a[0] + (b[0] - a[0]) * fraction, a[1] + (b[1] - a[1]) * fraction, a[2] + (b[2] - a[2]) * fraction];
}
