/**
 * Measures a Cycles reference's own Monte Carlo noise and stores it in its
 * json (`noise`, and `shot.render.noise`, which the viewer shows next to the
 * picture).
 *
 *   # a single render, against an independent render of the same shot (another seed, any sample count)
 *   node tools/reference/cycles/noise.ts --dir bench/references/two-room/lit-room --other-dir <folder of the second render>
 *   # the same, when the second render sits in the same folder with a suffix
 *   node tools/reference/cycles/noise.ts --dir bench/references/cornell-box-point/front --other-suffix=-seed1
 *   # a reference averaged from seed parts (combine.ts): its first two parts are compared
 *   node tools/reference/cycles/noise.ts --dir bench/references/two-room/dark-room-leak-view
 *
 * The number is the mean relative error (tools/compare: sum |A - B| / sum A)
 * the reference would show against the fully converged image, per pixel and
 * after 8 x 8 block averaging, for the GI composite and the full image.
 *
 * How it is derived from two independent renders A (the reference, N
 * samples) and B (M samples): their noise is independent and shrinks as
 * 1 / sqrt(samples), so the difference d between them has
 * d^2 = e_A^2 + e_B^2 = e_A^2 (1 + N / M), and e_A = d / sqrt(1 + N / M).
 * For a reference averaged from k parts of N / k samples, two parts give a
 * part's noise d / sqrt(2), and the average's is that / sqrt(k). (The mean
 * absolute error scales like the standard deviation for Gaussian noise, which
 * per-pixel Monte Carlo noise at these sample counts is close to.)
 */

import { readFile, writeFile } from 'node:fs/promises';
import { join, resolve } from 'node:path';
import { parseArgs } from 'node:util';
import { readExr } from '../../compare/exr.ts';
import { compareImages, downsample } from '../../compare/metrics.ts';

/** One image's noise: mean relative error against the converged image. */
export interface ImageNoise {
    readonly pixel: number;
    readonly block8: number;
}

/** The noise of a reference with `samples` samples, from its difference `difference` to an independent render with `otherSamples`. */
export function noiseFromIndependentRender(difference: number, samples: number, otherSamples: number): number {
    return difference / Math.sqrt(1 + samples / otherSamples);
}

/** The noise of the average of `parts` equal parts, from the difference between two of them. */
export function noiseFromParts(difference: number, parts: number): number {
    return difference / Math.sqrt(2) / Math.sqrt(parts);
}

const BLOCK = 8;

/** Mean relative error between two EXRs, per pixel and over 8 x 8 blocks. */
async function difference(referencePath: string, otherPath: string): Promise<ImageNoise> {
    const reference = readExr(await readFile(referencePath));
    const other = readExr(await readFile(otherPath));
    const block = (image: typeof reference) => downsample(image.rgb, image.width, image.height, BLOCK).rgb;
    return {
        pixel: compareImages(reference.rgb, other.rgb).meanRelativeError,
        block8: compareImages(block(reference), block(other)).meanRelativeError,
    };
}

interface ReferenceMeta {
    samplesPerPixel: number;
    combinedFrom?: { parts: string[]; samplesEach: number };
    noise?: unknown;
    shot: { render: Record<string, unknown> };
}

async function main(): Promise<void> {
    const { values } = parseArgs({
        options: {
            dir: { type: 'string' },
            prefix: { type: 'string', default: 'cycles' },
            'other-dir': { type: 'string' },
            'other-suffix': { type: 'string' },
        },
    });
    if (!values.dir) throw new Error('usage: noise.ts --dir <reference folder> [--prefix cycles] [--other-dir <folder> | --other-suffix=-seed1]');
    const directory = resolve(values.dir);
    const prefix = values.prefix ?? 'cycles';
    const metaPath = join(directory, `${prefix}.json`);
    const meta = JSON.parse(await readFile(metaPath, 'utf8')) as ReferenceMeta;
    const image = (folder: string, kind: string, suffix = '') => join(folder, `${prefix}-${kind}${suffix}.exr`);

    let scale: (value: number) => number;
    let pair: (kind: string) => [string, string];
    let method: string;
    if (meta.combinedFrom) {
        const [first, second] = meta.combinedFrom.parts;
        const parts = meta.combinedFrom.parts.length;
        scale = (value) => noiseFromParts(value, parts);
        pair = (kind) => [image(directory, kind, first), image(directory, kind, second)];
        method = `from parts ${first} and ${second} of ${parts} (${meta.combinedFrom.samplesEach} samples each)`;
    } else {
        const otherDirectory = values['other-dir'] ? resolve(values['other-dir']) : directory;
        const otherSuffix = values['other-suffix'] ?? '';
        if (!values['other-dir'] && !otherSuffix) throw new Error('a single render needs an independent second render: --other-dir or --other-suffix');
        const otherMeta = JSON.parse(await readFile(join(otherDirectory, `${prefix}${otherSuffix}.json`), 'utf8')) as ReferenceMeta & { seed: number };
        scale = (value) => noiseFromIndependentRender(value, meta.samplesPerPixel, otherMeta.samplesPerPixel);
        pair = (kind) => [image(directory, kind), image(otherDirectory, kind, otherSuffix)];
        method = `against an independent render (seed ${otherMeta.seed}, ${otherMeta.samplesPerPixel} samples)`;
    }

    const measure = async (kind: string): Promise<ImageNoise> => {
        const raw = await difference(...pair(kind));
        return { pixel: scale(raw.pixel), block8: scale(raw.block8) };
    };
    const noise = { gi: await measure('gi'), full: await measure('full'), method };
    meta.noise = noise;
    meta.shot.render.noise = noise;
    await writeFile(metaPath, JSON.stringify(meta, null, 2));
    const percent = (value: number): string => `${(value * 100).toFixed(2)}%`;
    console.log(
        `${directory}: GI ${percent(noise.gi.pixel)} per pixel, ${percent(noise.gi.block8)} 8x8; ` +
            `full ${percent(noise.full.pixel)} per pixel, ${percent(noise.full.block8)} 8x8 (${method})`,
    );
}

if (process.argv[1] && resolve(process.argv[1]) === resolve(import.meta.filename)) {
    main().catch((error) => {
        console.error(error instanceof Error ? error.message : error);
        process.exit(1);
    });
}
