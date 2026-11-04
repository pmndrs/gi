/**
 * Compares two linear HDR images (OpenEXR) and prints how different they are.
 *
 *   node tools/compare/index.ts reference.exr test.exr [options]
 *
 * The first image is the reference (ground truth); the second is the image
 * being judged. Prints FLIP, mean relative error and RMS, and writes
 * heatmap PNGs.
 */

import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { basename, dirname, join, parse, resolve } from 'node:path';
import { parseArgs } from 'node:util';
import { readExr, type FloatImage } from './exr.ts';
import { runFlip, type FlipOptions, type FlipResult } from './flip.ts';
import { clampImage, compareImages, downsample, percentile, relativeErrorMap, type ImageMetrics } from './metrics.ts';
import { decodePng, encodePng, magma, toDisplayRgb8 } from './png.ts';

const HELP = `usage: node tools/compare/index.ts <reference.exr> <test.exr> [options]

Compares two linear HDR images of the same size. The FIRST image is the
reference (ground truth); relative numbers are relative to it.

Numbers (sums run over every pixel and every colour channel):
  FLIP                 NVIDIA FLIP in HDR mode (HDR-FLIP), mean over pixels, 0 = identical.
                       Perceptual: it models how visible the difference is.
  mean relative error  sum |test - ref| / sum ref. The average absolute error as a
                       fraction of the average reference value, so a test image
                       10% too bright everywhere gives 0.100. (Not a per-pixel mean
                       of |t-r|/r: that blows up on black pixels.)
  bias                 sum (test - ref) / sum ref. Positive: the test is brighter.
  RMS                  sqrt(mean (test - ref)^2), in image units; "relative RMS" is
                       RMS divided by the mean reference value.
  p95 error            95th percentile of the per-pixel relative luminance error
                       |T - R| / (R + 5% of the mean R) over the compared image. With
                       --downsample n each pixel is an n x n block, so this is the
                       "95th-percentile block error": the worst few percent of the
                       picture, which the averages hide.
  clamped line         The same numbers again with every value limited to 1 first
                       (--also-clamp <x> changes the limit), so a very bright lamp
                       (17 in Cornell) does not dominate. Skipped if --clamp is given.

FLIP HDR settings: tone mapper ACES; start/stop exposure and number of exposures
are chosen by FLIP from the reference image unless given below, and are printed;
pixels per degree 67 (4K, 0.7 m wide, seen from 0.7 m) unless --ppd is given. With
--downsample n each compared pixel covers n times the viewing angle, so the pixels per
degree are divided by n (67 becomes 67/n) and FLIP judges the same view, only blurred.

options:
  --out <prefix>        Write <prefix>-heatmap.png (relative error), <prefix>-flip.png
                        (FLIP's error map) and <prefix>-side-by-side.png
                        (reference | test | FLIP). Default: next to the test image.
  --no-images           Do not write PNGs.
  --no-flip             Skip FLIP (no Python needed).
  --clamp <x>           Limit every pixel value to at most x in both images before
                        everything else (before block averaging too).
  --also-clamp <x>      Limit for the extra clamped line (default 1).
  --downsample <n>      Average n x n pixel blocks in both images first, then compare.
                        Cuts Monte Carlo noise by about n so lighting differences
                        stand out (FLIP and the PNGs use the reduced images too).
  --json                Print the results as JSON instead of text.
  --max-error <x>       Heatmap saturates at this relative error (default 0.5).
  --exposure <stops>    Exposure for the viewing PNGs (default 0).
  --tone-mapper <name>  ACES (default), HABLE or REINHARD.
  --ppd <n>             FLIP pixels per degree.
  --start-exposure <c>  FLIP start exposure in stops.
  --stop-exposure <c>   FLIP stop exposure in stops.
  --num-exposures <n>   FLIP number of exposures.
  --python <exe>        Python for FLIP (default: python).
`;

interface CompareReport {
    reference: string;
    test: string;
    width: number;
    height: number;
    downsample: number;
    metrics: ImageMetrics;
    p95Error: number;
    /** The same numbers with values limited to `limit`; null when --clamp was given. */
    clamped: { limit: number; metrics: ImageMetrics; p95Error: number } | null;
    flip: Omit<FlipResult, 'errorMapPng'> | null;
}

async function main(): Promise<number> {
    const { values, positionals } = parseArgs({
        allowPositionals: true,
        options: {
            out: { type: 'string' },
            'no-images': { type: 'boolean' },
            'no-flip': { type: 'boolean' },
            downsample: { type: 'string' },
            clamp: { type: 'string' },
            'also-clamp': { type: 'string' },
            json: { type: 'boolean' },
            'max-error': { type: 'string' },
            exposure: { type: 'string' },
            'tone-mapper': { type: 'string' },
            ppd: { type: 'string' },
            'start-exposure': { type: 'string' },
            'stop-exposure': { type: 'string' },
            'num-exposures': { type: 'string' },
            python: { type: 'string' },
            help: { type: 'boolean', short: 'h' },
        },
    });
    if (values.help || positionals.length !== 2) {
        console.log(HELP);
        return values.help ? 0 : 1;
    }

    const [referencePath, testPath] = positionals.map((path) => resolve(path));
    const originalReference = readExr(new Uint8Array(await readFile(referencePath)));
    const originalTest = readExr(new Uint8Array(await readFile(testPath)));
    if (originalReference.width !== originalTest.width || originalReference.height !== originalTest.height) {
        throw new Error(
            `size mismatch: ${basename(referencePath)} is ${originalReference.width}x${originalReference.height}, ${basename(testPath)} is ${originalTest.width}x${originalTest.height}`,
        );
    }
    const blockSize = Number(values.downsample ?? 1);
    if (!Number.isInteger(blockSize) || blockSize < 1) throw new Error('--downsample must be a whole number of 1 or more');

    /** Clamp first, so a bright pixel cannot leak into its block average, then average blocks. */
    const prepare = (limit: number | null): { reference: FloatImage; test: FloatImage } => {
        let reference = originalReference;
        let test = originalTest;
        if (limit !== null) {
            reference = { ...reference, rgb: clampImage(reference.rgb, limit) };
            test = { ...test, rgb: clampImage(test.rgb, limit) };
        }
        if (blockSize > 1) {
            reference = downsample(reference.rgb, reference.width, reference.height, blockSize);
            test = downsample(test.rgb, test.width, test.height, blockSize);
        }
        return { reference, test };
    };
    const measure = (images: { reference: FloatImage; test: FloatImage }): { metrics: ImageMetrics; p95Error: number } => ({
        metrics: compareImages(images.reference.rgb, images.test.rgb),
        p95Error: percentile(relativeErrorMap(images.reference.rgb, images.test.rgb), 95),
    });

    let mainClamp: number | null = null;
    if (values.clamp !== undefined) {
        mainClamp = Number(values.clamp);
        if (!(mainClamp > 0)) throw new Error('--clamp must be a number above 0');
    }
    const { reference, test } = prepare(mainClamp);
    const { metrics, p95Error } = measure({ reference, test });

    let clamped: CompareReport['clamped'] = null;
    if (mainClamp === null) {
        const limit = Number(values['also-clamp'] ?? 1);
        if (!(limit > 0)) throw new Error('--also-clamp must be a number above 0');
        clamped = { limit, ...measure(prepare(limit)) };
    }

    let flip: FlipResult | null = null;
    if (!values['no-flip']) {
        const toneMapper = (values['tone-mapper'] ?? 'ACES').toUpperCase();
        if (toneMapper !== 'ACES' && toneMapper !== 'HABLE' && toneMapper !== 'REINHARD') throw new Error(`unknown tone mapper "${toneMapper}"`);
        const options: FlipOptions = { toneMapper, python: values.python ?? 'python' };
        // FLIP's default is 67 pixels per degree. Block averaging makes each pixel n times wider in
        // viewing angle, so divide by n to keep FLIP looking at the same view.
        const basePpd = values.ppd ? Number(values.ppd) : blockSize > 1 ? 67 : null;
        if (basePpd !== null) options.pixelsPerDegree = basePpd / blockSize;
        if (values['start-exposure']) options.startExposure = Number(values['start-exposure']);
        if (values['stop-exposure']) options.stopExposure = Number(values['stop-exposure']);
        if (values['num-exposures']) options.numExposures = Number(values['num-exposures']);
        flip = await runFlip(reference, test, options);
    }

    const written: string[] = [];
    if (!values['no-images']) {
        const prefix = values.out ? resolve(values.out) : join(dirname(testPath), `${parse(testPath).name}-vs-${parse(referencePath).name}`);
        await mkdir(dirname(prefix), { recursive: true });
        written.push(...(await writeImages(prefix, reference, test, flip, Number(values['max-error'] ?? 0.5), Number(values.exposure ?? 0))));
    }

    const report: CompareReport = {
        reference: referencePath,
        test: testPath,
        width: reference.width,
        height: reference.height,
        downsample: blockSize,
        metrics,
        p95Error,
        clamped,
        flip: flip ? withoutErrorMap(flip) : null,
    };
    if (values.json) {
        console.log(JSON.stringify({ ...report, images: written }, null, 2));
    } else {
        printReport(report, written);
    }
    return 0;
}

function withoutErrorMap(result: FlipResult): Omit<FlipResult, 'errorMapPng'> {
    const { errorMapPng: _errorMapPng, ...numbers } = result;
    return numbers;
}

async function writeImages(
    prefix: string,
    reference: { width: number; height: number; rgb: Float32Array },
    test: { width: number; height: number; rgb: Float32Array },
    flip: FlipResult | null,
    maxError: number,
    exposureStops: number,
): Promise<string[]> {
    const { width, height } = reference;
    const errors = relativeErrorMap(reference.rgb, test.rgb);
    const heat = new Uint8Array(width * height * 3);
    for (let pixel = 0; pixel < errors.length; pixel++) {
        const [r, g, b] = magma(errors[pixel] / maxError);
        heat[pixel * 3] = r;
        heat[pixel * 3 + 1] = g;
        heat[pixel * 3 + 2] = b;
    }
    const heatmapPath = `${prefix}-heatmap.png`;
    await writeFile(heatmapPath, encodePng(width, height, heat));
    const written = [heatmapPath];

    const referenceDisplay = toDisplayRgb8(reference.rgb, exposureStops);
    const testDisplay = toDisplayRgb8(test.rgb, exposureStops);
    let flipDisplay: Uint8Array = heat;
    if (flip) {
        const flipPath = `${prefix}-flip.png`;
        await writeFile(flipPath, Buffer.from(flip.errorMapPng));
        written.push(flipPath);
        const decoded = decodePng(flip.errorMapPng);
        if (decoded.width === width && decoded.height === height) flipDisplay = decoded.rgb8;
    }

    const sideBySide = new Uint8Array(width * 3 * height * 3);
    for (let y = 0; y < height; y++) {
        const row = y * width * 3;
        sideBySide.set(referenceDisplay.subarray(row, row + width * 3), y * width * 9);
        sideBySide.set(testDisplay.subarray(row, row + width * 3), y * width * 9 + width * 3);
        sideBySide.set(flipDisplay.subarray(row, row + width * 3), y * width * 9 + width * 6);
    }
    const sidePath = `${prefix}-side-by-side.png`;
    await writeFile(sidePath, encodePng(width * 3, height, sideBySide));
    written.push(sidePath);
    return written;
}

function printReport(report: CompareReport, written: string[]): void {
    const { metrics, flip } = report;
    const percent = (value: number): string => `${(value * 100).toFixed(3)}%`;
    console.log(`reference: ${report.reference} (${report.width}x${report.height}${report.downsample > 1 ? `, after ${report.downsample}x${report.downsample} block averaging` : ''})`);
    console.log(`test:      ${report.test}`);
    console.log(`mean value: reference ${metrics.meanReference.toPrecision(5)}, test ${metrics.meanTest.toPrecision(5)}`);
    if (flip) {
        const exposures =
            flip.startExposure !== null && flip.stopExposure !== null
                ? `, exposures ${flip.startExposure} to ${flip.stopExposure} (${flip.numExposures} steps)`
                : '';
        console.log(`FLIP (HDR${exposures}${flip.pixelsPerDegree !== null ? `, ${flip.pixelsPerDegree} ppd` : ''}): mean ${flip.mean.toFixed(6)}`);
    }
    console.log(`mean relative error: ${metrics.meanRelativeError.toFixed(6)}  (${percent(metrics.meanRelativeError)})`);
    console.log(`bias:                ${metrics.bias >= 0 ? '+' : ''}${metrics.bias.toFixed(6)}  (${metrics.bias > 0 ? 'test brighter' : metrics.bias < 0 ? 'test darker' : 'no overall difference'})`);
    console.log(`RMS:                 ${metrics.rms.toPrecision(5)}  (relative to mean reference: ${percent(metrics.relativeRms)})`);
    const unit = report.downsample > 1 ? `${report.downsample}x${report.downsample} block` : 'pixel';
    console.log(`p95 ${unit} error:`.padEnd(21) + ` ${percent(report.p95Error)}  (95th percentile of the per-${unit} relative luminance error)`);
    if (report.clamped) {
        const c = report.clamped;
        console.log(`values limited to ${c.limit}: mean relative error ${percent(c.metrics.meanRelativeError)}, bias ${c.metrics.bias >= 0 ? '+' : ''}${percent(c.metrics.bias)}, relative RMS ${percent(c.metrics.relativeRms)}, p95 ${unit} error ${percent(c.p95Error)}`);
    }
    if (metrics.nonFiniteValues > 0) console.log(`warning: ${metrics.nonFiniteValues} NaN/infinite values were skipped`);
    for (const path of written) console.log(`wrote ${path}`);
}

main().then(
    (code) => process.exit(code),
    (error) => {
        console.error(`error: ${error instanceof Error ? error.message : error}`);
        process.exit(1);
    },
);
