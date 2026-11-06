/**
 * Averages several renders of the same shot made with different seeds into
 * one reference, so a long render can be split into short bench-lock holds
 * (one shot per hold; other lanes time between them).
 *
 *   node tools/reference/cycles/run.ts --scene sponza --camera 1 --samples 2048 --seed 0 --name-suffix=-part0 ...
 *   node tools/reference/cycles/run.ts --scene sponza --camera 1 --samples 2048 --seed 1 --name-suffix=-part1 ...
 *   node tools/reference/cycles/combine.ts --dir bench/references/sponza/colonnade-from-the-floor \
 *        --prefix cycles-untextured --parts=-part0,-part1
 *
 * Writes `<prefix>-{full,indirect,direct,lights-direct,gi}.exr` and `<prefix>.json`.
 *
 * Why averaging is exact: each part is an unbiased Monte Carlo estimate of the
 * same pixel values with independent random numbers (a different seed), so
 * the mean of N parts of S samples each is the same estimator as one render of
 * N × S samples. Every part must have the same shot (camera, size, scene,
 * material table), sample count, filter and device, or the run stops.
 * Denoised viewing copies are not averaged; `--denoised-from -part0` keeps
 * that part's denoised copy (rendered with --denoise) as the reference's
 * viewing copy, with its stored check.
 *
 * Two parts also give the reference's noise: compare them with tools/compare
 * (`--downsample 8`); the difference of two halves, divided by 2, is the
 * noise of their average.
 */

import { readFile, writeFile } from 'node:fs/promises';
import { join, resolve } from 'node:path';
import { parseArgs } from 'node:util';
import { readExr, writeExr, type FloatImage } from '../../compare/exr.ts';

/** The images every render writes (render.py and run.ts), by kind. */
const KINDS = ['full', 'indirect', 'direct', 'lights-direct', 'gi'] as const;

/** Fields of the shot that must agree between parts for their average to be one reference. */
function shotKey(meta: Record<string, unknown>): string {
    const shot = meta.shot as { camera: unknown; scene: unknown; display: unknown; render: { pixelFilter: unknown; device: unknown; samplesPerPixel: unknown } } | undefined;
    if (!shot) throw new Error('a part has no stored shot (rendered before T1.13)');
    return JSON.stringify([shot.camera, shot.scene, shot.display, shot.render.pixelFilter, shot.render.device, shot.render.samplesPerPixel]);
}

function average(images: readonly FloatImage[]): FloatImage {
    const [first] = images;
    for (const image of images) {
        if (image.width !== first.width || image.height !== first.height) throw new Error('the parts differ in size');
    }
    const rgb = new Float32Array(first.rgb.length);
    for (const image of images) for (let index = 0; index < rgb.length; index++) rgb[index] += image.rgb[index];
    for (let index = 0; index < rgb.length; index++) rgb[index] /= images.length;
    return { width: first.width, height: first.height, rgb };
}

async function main(): Promise<void> {
    const { values } = parseArgs({
        options: {
            dir: { type: 'string' },
            prefix: { type: 'string', default: 'cycles' },
            parts: { type: 'string' },
            // A part rendered with --denoise whose denoised copy becomes the reference's viewing copy.
            'denoised-from': { type: 'string' },
        },
    });
    if (!values.dir || !values.parts) throw new Error('usage: combine.ts --dir <reference folder> [--prefix cycles] --parts=-part0,-part1');
    const directory = resolve(values.dir);
    const prefix = values.prefix ?? 'cycles';
    const parts = values.parts.split(',').filter(Boolean);
    if (parts.length < 2) throw new Error('give at least two parts');

    const metas = await Promise.all(parts.map(async (part) => JSON.parse(await readFile(join(directory, `${prefix}${part}.json`), 'utf8')) as Record<string, unknown>));
    const key = shotKey(metas[0]);
    metas.forEach((meta, index) => {
        if (shotKey(meta) !== key) throw new Error(`part ${parts[index]} has a different shot, sample count, filter or device than ${parts[0]}`);
    });
    const seeds = metas.map((meta) => meta.seed as number);
    if (new Set(seeds).size !== seeds.length) throw new Error(`the parts must use different seeds; got ${seeds.join(', ')}`);

    const files: string[] = [];
    for (const kind of KINDS) {
        const images = await Promise.all(parts.map(async (part) => readExr(await readFile(join(directory, `${prefix}-${kind}${part}.exr`)))));
        const name = `${prefix}-${kind}.exr`;
        await writeFile(join(directory, name), writeExr(average(images)));
        files.push(name);
    }

    const samplesEach = metas[0].samplesPerPixel as number;
    const sum = (field: string): number => Math.round(metas.reduce((total, meta) => total + (meta[field] as number), 0) * 100) / 100;
    const combined = structuredClone(metas[0]) as Record<string, unknown> & { shot: { render: Record<string, unknown> } };
    combined.samplesPerPixel = samplesEach * parts.length;
    combined.seed = seeds;
    combined.renderSeconds = sum('renderSeconds');
    combined.lightsOnlySeconds = sum('lightsOnlySeconds');
    combined.files = files;
    combined.jsonFile = `${prefix}.json`;
    combined.denoisedFile = null;
    combined.denoiseCheck = undefined;
    combined.combinedFrom = { parts, samplesEach, note: 'the mean of independent renders with different seeds (tools/reference/cycles/combine.ts)' };
    combined.finishedAt = metas.map((meta) => meta.finishedAt as string).sort().at(-1);
    combined.shot.render.samplesPerPixel = combined.samplesPerPixel;
    combined.shot.render.seed = seeds;
    combined.shot.render.renderSeconds = combined.renderSeconds;
    combined.shot.render.denoised = null;
    combined.shot.render.combinedFrom = parts;
    const denoisedFrom = values['denoised-from'];
    if (denoisedFrom) {
        // The viewing copy is one part's denoised image (fewer samples than the average; it is only looked at).
        const index = parts.indexOf(denoisedFrom);
        if (index < 0) throw new Error(`--denoised-from ${denoisedFrom} is not one of the parts`);
        const part = metas[index] as { denoisedFile?: string | null; denoiseCheck?: unknown; shot: { render: { denoised: unknown } } };
        if (!part.denoisedFile) throw new Error(`part ${denoisedFrom} has no denoised copy (render it with --denoise)`);
        const name = `${prefix}-full-denoised.exr`;
        await writeFile(join(directory, name), await readFile(join(directory, part.denoisedFile)));
        combined.denoisedFile = name;
        combined.denoiseCheck = part.denoiseCheck;
        combined.shot.render.denoised = { ...(part.shot.render.denoised as object), file: name, fromPart: denoisedFrom, samplesPerPixel: samplesEach };
    }
    await writeFile(join(directory, `${prefix}.json`), JSON.stringify(combined, null, 2));
    console.log(`wrote ${files.join(', ')} and ${prefix}.json: ${parts.length} parts × ${samplesEach} samples = ${combined.samplesPerPixel} (seeds ${seeds.join(', ')})`);
}

main().catch((error) => {
    console.error(error instanceof Error ? error.message : error);
    process.exit(1);
});
