import { describe, expect, it } from 'vitest';
import { spawnSync } from 'node:child_process';
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { deflateSync } from 'node:zlib';
import { readExr, writeExr, type FloatImage } from './exr.ts';
import { parseFlipOutput } from './flip.ts';
import { clampImage, compareImages, downsample, findFinitePixels, maskImage, percentile, relativeErrorMap, scaleImage, zeroNonFinitePixels } from './metrics.ts';
import { decodePng, encodePng, linearToSrgb8, magma } from './png.ts';

/** A small image with a wide range of values, including some dark pixels. */
function testImage(width = 8, height = 6): FloatImage {
    const rgb = new Float32Array(width * height * 3);
    for (let index = 0; index < rgb.length; index++) {
        rgb[index] = 0.001 + ((index * 37) % 101) / 10; // 0.001 .. 10.1
    }
    return { width, height, rgb };
}

describe('compareImages', () => {
    it('gives zero error for an image against itself', () => {
        const image = testImage();
        const metrics = compareImages(image.rgb, image.rgb);
        expect(metrics.meanRelativeError).toBe(0);
        expect(metrics.bias).toBe(0);
        expect(metrics.rms).toBe(0);
        expect(metrics.relativeRms).toBe(0);
        expect(metrics.nonFiniteValues).toBe(0);
    });

    it('reports exactly 10 percent for a copy brightened by 10 percent, however dark some pixels are', () => {
        const image = testImage();
        const metrics = compareImages(image.rgb, scaleImage(image.rgb, 1.1));
        expect(metrics.meanRelativeError).toBeCloseTo(0.1, 6);
        expect(metrics.bias).toBeCloseTo(0.1, 6);
        // For a uniform scale k the error is (k - 1) * R, so RMS / mean R = (k - 1) * sqrt(mean R^2) / mean R.
        const meanSquare = image.rgb.reduce((total, value) => total + value * value, 0) / image.rgb.length;
        expect(metrics.rms).toBeCloseTo(0.1 * Math.sqrt(meanSquare), 4);
    });

    it('keeps the sign in bias but not in the mean relative error', () => {
        const image = testImage();
        const darker = compareImages(image.rgb, scaleImage(image.rgb, 0.8));
        expect(darker.meanRelativeError).toBeCloseTo(0.2, 6);
        expect(darker.bias).toBeCloseTo(-0.2, 6);
    });

    it('skips non-finite values and counts them', () => {
        const image = testImage();
        const broken = Float32Array.from(image.rgb);
        broken[4] = NaN;
        broken[9] = Infinity;
        const metrics = compareImages(image.rgb, broken);
        expect(metrics.nonFiniteValues).toBe(2);
        expect(metrics.meanRelativeError).toBe(0);
    });

    it('refuses images of different sizes', () => {
        expect(() => compareImages(new Float32Array(12), new Float32Array(15))).toThrow(/differ in size/);
    });
});

describe('relativeErrorMap', () => {
    it('is zero for equal images and about 0.1 where a lit pixel is 10 percent off', () => {
        const reference = new Float32Array([1, 1, 1, 0, 0, 0]);
        expect(Array.from(relativeErrorMap(reference, reference))).toEqual([0, 0]);
        const brighter = relativeErrorMap(reference, scaleImage(reference, 1.1));
        // The floor (5 percent of the mean luminance) keeps this a little under 0.1.
        expect(brighter[0]).toBeGreaterThan(0.08);
        expect(brighter[0]).toBeLessThan(0.1);
        expect(brighter[1]).toBe(0);
    });
});

describe('clampImage and downsample', () => {
    it('clamps values above the limit and leaves the rest', () => {
        expect(Array.from(clampImage(new Float32Array([0.5, 1, 17]), 1))).toEqual([0.5, 1, 1]);
    });

    it('averages blocks, including a smaller edge block', () => {
        // 3 x 1 image, factor 2: one block of pixels 0 and 1, one edge block of pixel 2.
        const out = downsample(new Float32Array([1, 1, 1, 3, 3, 3, 10, 10, 10]), 3, 1, 2);
        expect(out.width).toBe(2);
        expect(out.height).toBe(1);
        expect(Array.from(out.rgb)).toEqual([2, 2, 2, 10, 10, 10]);
    });
});

describe('NaN and infinite pixels', () => {
    it('counts bad pixels per image and in either, a whole pixel at a time', () => {
        // 4 pixels: reference pixel 1 has a NaN (two channels), test pixel 1 and 2 have infinities.
        const reference = new Float32Array([1, 1, 1, NaN, NaN, 1, 2, 2, 2, 3, 3, 3]);
        const test = new Float32Array([1, 1, 1, 1, Infinity, 1, -Infinity, 2, 2, 3, 3, 3]);
        const finite = findFinitePixels(reference, test);
        expect(finite.referenceBad).toBe(1);
        expect(finite.testBad).toBe(2);
        expect(finite.eitherBad).toBe(2);
        expect(Array.from(finite.usable)).toEqual([1, 0, 0, 1]);
        // Masked in both images, so the remaining pixels compare exactly.
        const metrics = compareImages(maskImage(reference, finite.usable), maskImage(test, finite.usable));
        expect(metrics.meanRelativeError).toBe(0);
        expect(metrics.meanReference).toBe(2);
    });

    it('averages only the finite pixels of a block and leaves out a block with none', () => {
        // 4 x 2 image, factor 2: left block has one NaN pixel, right block is all bad.
        const rgb = new Float32Array([
            1, 1, 1, NaN, 0, 0, NaN, NaN, NaN, Infinity, 0, 0,
            3, 3, 3, 5, 5, 5, NaN, 1, 1, 1, 1, Infinity,
        ]);
        const out = downsample(rgb, 4, 2, 2);
        expect(out.emptyBlocks).toBe(1);
        expect(Array.from(out.rgb.subarray(0, 3))).toEqual([3, 3, 3]);
        expect(Array.from(out.rgb.subarray(3, 6)).every(Number.isNaN)).toBe(true);
    });

    it('uses the pair mask, so both images skip the same pixels', () => {
        // Pixel 0 is bad only in the test; the reference must skip it too, or the block averages differ.
        const reference = new Float32Array([100, 100, 100, 1, 1, 1]);
        const test = new Float32Array([NaN, 0, 0, 1, 1, 1]);
        const { usable } = findFinitePixels(reference, test);
        const reducedReference = downsample(reference, 2, 1, 2, usable);
        const reducedTest = downsample(test, 2, 1, 2, usable);
        expect(Array.from(reducedReference.rgb)).toEqual([1, 1, 1]);
        expect(compareImages(reducedReference.rgb, reducedTest.rgb).meanRelativeError).toBe(0);
    });

    it('clampImage leaves NaN and infinities alone instead of turning +Inf into the limit', () => {
        const out = clampImage(new Float32Array([0.5, 17, Infinity, -Infinity, NaN]), 1);
        expect(Array.from(out.subarray(0, 4))).toEqual([0.5, 1, Infinity, -Infinity]);
        expect(Number.isNaN(out[4])).toBe(true);
    });

    it('gives FLIP zeros in both images where either is not finite, and counts them', () => {
        const reference = new Float32Array([1, 2, 3, 4, 5, 6]);
        const test = new Float32Array([1, 2, 3, NaN, 5, 6]);
        const cleaned = zeroNonFinitePixels(reference, test);
        expect(cleaned.zeroedPixels).toBe(1);
        expect(Array.from(cleaned.reference)).toEqual([1, 2, 3, 0, 0, 0]);
        expect(Array.from(cleaned.test)).toEqual([1, 2, 3, 0, 0, 0]);
        // The inputs are not changed.
        expect(reference[3]).toBe(4);
    });

    it('leaves bad pixels out of the p95 error instead of counting them as full error', () => {
        const reference = new Float32Array([1, 1, 1, 1, 1, 1, NaN, 1, 1]);
        const map = relativeErrorMap(reference, reference);
        expect(Number.isNaN(map[2])).toBe(true);
        expect(percentile(map, 95)).toBe(0);
    });

    it('the compare tool reports the true count and finite numbers', () => {
        const folder = mkdtempSync(join(tmpdir(), 'ddgi-compare-test-'));
        try {
            const width = 16;
            const height = 8;
            const reference = new Float32Array(width * height * 3).fill(0.5);
            const test = scaleImage(reference, 1.1);
            // Three bad pixels: two in the test, one in the reference; one 8 x 8 block keeps finite pixels.
            test[0] = NaN;
            test[3 * 5 + 1] = Infinity;
            reference[3 * 20] = -Infinity;
            writeFileSync(join(folder, 'reference.exr'), writeExr({ width, height, rgb: reference }));
            writeFileSync(join(folder, 'test.exr'), writeExr({ width, height, rgb: test }));
            const result = spawnSync(
                process.execPath,
                [join(__dirname, 'index.ts'), join(folder, 'reference.exr'), join(folder, 'test.exr'), '--no-flip', '--no-images', '--json', '--downsample', '8'],
                { encoding: 'utf8' },
            );
            expect(result.stderr).toBe('');
            const report = JSON.parse(result.stdout);
            expect(report.nonFinite).toEqual({ referencePixels: 1, testPixels: 2, eitherPixels: 3, emptyBlocks: 0 });
            expect(report.metrics.meanRelativeError).toBeCloseTo(0.1, 6);
            expect(report.metrics.nonFiniteValues).toBe(0);
            expect(report.clamped.metrics.meanRelativeError).toBeCloseTo(0.1, 6);
        } finally {
            rmSync(folder, { recursive: true, force: true });
        }
    }, 30_000);
});

describe('percentile', () => {
    it('interpolates between ranks and ignores non-finite values', () => {
        expect(percentile([1, 2, 3, 4, 5], 0)).toBe(1);
        expect(percentile([1, 2, 3, 4, 5], 50)).toBe(3);
        expect(percentile([1, 2, 3, 4, 5], 100)).toBe(5);
        expect(percentile([0, 10], 95)).toBeCloseTo(9.5, 10);
        expect(percentile([1, NaN, 3, Infinity], 50)).toBe(2);
        expect(percentile([], 95)).toBe(0);
    });
});

describe('EXR round trip', () => {
    it('writes and reads back float RGB exactly', () => {
        const image = testImage(13, 7);
        const read = readExr(writeExr(image));
        expect(read.width).toBe(13);
        expect(read.height).toBe(7);
        expect(Array.from(read.rgb)).toEqual(Array.from(image.rgb));
    });

    it('reads a ZIP-compressed file (the codec Blender writes), shrinking and stored-raw blocks alike', () => {
        // A smooth image compresses; a noisy one does not and is stored raw.
        const smooth: FloatImage = { width: 40, height: 20, rgb: new Float32Array(40 * 20 * 3).fill(0.5) };
        for (const image of [smooth, testImage(5, 3)]) {
            const zipped = rewriteAsZip(writeExr(image), image);
            expect(Array.from(readExr(zipped).rgb)).toEqual(Array.from(image.rgb));
        }
    });

    it('rejects a file that is not EXR', () => {
        expect(() => readExr(new Uint8Array(16))).toThrow(/not an OpenEXR/);
    });
});

describe('PNG helpers', () => {
    it('round trips pixels through the encoder and decoder', () => {
        const pixels = Uint8Array.from({ length: 4 * 3 * 3 }, (_, index) => (index * 7) % 256);
        const decoded = decodePng(encodePng(4, 3, pixels));
        expect(decoded.width).toBe(4);
        expect(decoded.height).toBe(3);
        expect(Array.from(decoded.rgb8)).toEqual(Array.from(pixels));
    });

    it('converts linear values to sRGB bytes', () => {
        expect(linearToSrgb8(0)).toBe(0);
        expect(linearToSrgb8(1)).toBe(255);
        expect(linearToSrgb8(5)).toBe(255);
        expect(linearToSrgb8(0.18)).toBe(118);
    });

    it('maps magma from black to pale yellow', () => {
        expect(magma(0)).toEqual([0, 0, 4]);
        expect(magma(1)).toEqual([252, 253, 191]);
        expect(magma(-3)).toEqual(magma(0));
    });
});

describe('parseFlipOutput', () => {
    it('reads the mean and the exposure range from FLIP text output', () => {
        const text = ['FLIP v1.7.', '\tMean: 0.047195', '\tWeighted median: 0.04', '\tStart exposure: 1.98', '\tStop exposure: 2.0833', '\tNumber of exposures: 2'].join('\n');
        const parsed = parseFlipOutput(text);
        expect(parsed.mean).toBeCloseTo(0.047195, 6);
        expect(parsed.startExposure).toBeCloseTo(1.98, 6);
        expect(parsed.numExposures).toBe(2);
    });

    it('fails loudly when there is no mean', () => {
        expect(() => parseFlipOutput('nothing useful')).toThrow(/Mean/);
    });
});

/**
 * Re-encodes an uncompressed float EXR as ZIP. This is the encoder side of
 * EXR's codec, used only to test the reader: bytes are split into even and odd
 * halves, delta-coded with +128, then deflated.
 */
function rewriteAsZip(uncompressed: Uint8Array, image: FloatImage): Uint8Array {
    const view = new DataView(uncompressed.buffer, uncompressed.byteOffset, uncompressed.byteLength);
    const lineBytes = image.width * 3 * 4;
    const firstOffset = Number(view.getBigUint64(findTableStart(uncompressed), true));
    const headerLength = findTableStart(uncompressed);
    const blockCount = Math.ceil(image.height / 16);

    const blocks: Uint8Array[] = [];
    for (let block = 0; block < blockCount; block++) {
        const firstLine = block * 16;
        const lines = Math.min(16, image.height - firstLine);
        const raw = new Uint8Array(lines * lineBytes);
        for (let line = 0; line < lines; line++) {
            const start = firstOffset + (firstLine + line) * (8 + lineBytes) + 8;
            raw.set(uncompressed.subarray(start, start + lineBytes), line * lineBytes);
        }
        const split = new Uint8Array(raw.length);
        const half = Math.ceil(raw.length / 2);
        for (let index = 0; index < raw.length; index++) {
            split[index % 2 === 0 ? index / 2 : half + (index - 1) / 2] = raw[index];
        }
        const delta = new Uint8Array(split.length);
        delta[0] = split[0];
        for (let index = 1; index < split.length; index++) delta[index] = (split[index] - split[index - 1] + 128 + 256) & 0xff;
        // As in real EXR files, a block that does not shrink is stored raw.
        const deflated = deflateSync(delta);
        const compressed = deflated.length < raw.length ? deflated : raw;
        const out = new Uint8Array(8 + compressed.length);
        const outView = new DataView(out.buffer);
        outView.setInt32(0, firstLine, true);
        outView.setInt32(4, compressed.length, true);
        out.set(compressed, 8);
        blocks.push(out);
    }

    // Header with the compression attribute switched to ZIP (3), then the table, then the blocks.
    const header = Uint8Array.from(uncompressed.subarray(0, headerLength));
    const marker = new TextEncoder().encode('compression\0compression\0');
    const at = indexOfBytes(header, marker) + marker.length + 4; // skip the int32 size
    header[at] = 3;
    const tableBytes = blockCount * 8;
    const out = new Uint8Array(headerLength + tableBytes + blocks.reduce((total, block) => total + block.length, 0));
    out.set(header, 0);
    const outView = new DataView(out.buffer);
    let cursor = headerLength + tableBytes;
    blocks.forEach((block, index) => {
        outView.setBigUint64(headerLength + index * 8, BigInt(cursor), true);
        out.set(block, cursor);
        cursor += block.length;
    });
    return out;
}

/** The scanline offset table starts right after the header, whose last byte is the 0 that ends the attribute list. */
function findTableStart(file: Uint8Array): number {
    const view = new DataView(file.buffer, file.byteOffset, file.byteLength);
    let cursor = 8;
    for (;;) {
        if (file[cursor] === 0) return cursor + 1;
        while (file[cursor] !== 0) cursor++;
        cursor++;
        while (file[cursor] !== 0) cursor++;
        cursor++;
        cursor += 4 + view.getInt32(cursor, true);
    }
}

function indexOfBytes(haystack: Uint8Array, needle: Uint8Array): number {
    outer: for (let start = 0; start <= haystack.length - needle.length; start++) {
        for (let index = 0; index < needle.length; index++) if (haystack[start + index] !== needle[index]) continue outer;
        return start;
    }
    return -1;
}
