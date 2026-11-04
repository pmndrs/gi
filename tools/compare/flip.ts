/**
 * NVIDIA FLIP, through its Python package (`pip install flip-evaluator`).
 *
 * Why Python for this one part: FLIP's reference implementation is the
 * `flip-evaluator` wheel (the CLI is `python -m flip_evaluator.flip_python_api`).
 * There is no JavaScript port, and re-implementing a published perceptual
 * metric here would be a bigger risk than calling it. Everything else in
 * tools/compare (EXR reading, the other metrics, PNGs) is Node, so the tool
 * works, minus FLIP, on a machine without Python (`--no-flip`).
 *
 * HDR mode: when both inputs are .exr, FLIP runs HDR-FLIP. It tone-maps the
 * images at a range of exposures (start exposure to stop exposure, N steps),
 * runs LDR-FLIP on each, and keeps the per-pixel maximum. Unless told
 * otherwise it picks the exposure range from the reference image. We pass the
 * options through and report what it chose.
 */

import { spawn } from 'node:child_process';
import { mkdtemp, readdir, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { writeExr, type FloatImage } from './exr.ts';

export interface FlipOptions {
    /** 'ACES' (FLIP's default), 'HABLE' or 'REINHARD'. */
    toneMapper: 'ACES' | 'HABLE' | 'REINHARD';
    /** Pixels per degree. FLIP's default of 67 means a 4K, 0.7 m wide display seen from 0.7 m. */
    pixelsPerDegree?: number;
    /** Exposure range in stops; omitted means FLIP chooses from the reference. */
    startExposure?: number;
    stopExposure?: number;
    numExposures?: number;
    /** Python executable. */
    python: string;
}

export interface FlipResult {
    /** Mean FLIP error, 0 (identical) to 1. */
    mean: number;
    weightedMedian: number | null;
    startExposure: number | null;
    stopExposure: number | null;
    numExposures: number | null;
    pixelsPerDegree: number | null;
    /** FLIP's own error map, a magma-coloured PNG. */
    errorMapPng: Uint8Array;
}

/** Pulls the numbers out of FLIP's text output. Exported for testing. */
export function parseFlipOutput(text: string): Omit<FlipResult, 'errorMapPng'> {
    const number = (pattern: RegExp): number | null => {
        const match = text.match(pattern);
        return match ? Number(match[1]) : null;
    };
    const mean = number(/Mean:\s*([-+0-9.eE]+)/);
    if (mean === null || Number.isNaN(mean)) throw new Error(`could not find "Mean:" in FLIP's output:\n${text}`);
    return {
        mean,
        weightedMedian: number(/Weighted median:\s*([-+0-9.eE]+)/),
        startExposure: number(/Start exposure:\s*([-+0-9.eE]+)/),
        stopExposure: number(/Stop exposure:\s*([-+0-9.eE]+)/),
        numExposures: number(/Number of exposures:\s*([-+0-9.eE]+)/),
        pixelsPerDegree: number(/(?:PPD|Pixels per degree)[^0-9-]*([-+0-9.eE]+)/i),
    };
}

function run(command: string, args: string[], cwd: string): Promise<{ code: number | null; output: string }> {
    return new Promise((resolve, reject) => {
        const child = spawn(command, args, { cwd, stdio: ['ignore', 'pipe', 'pipe'] });
        let output = '';
        child.stdout.on('data', (chunk: Buffer) => (output += chunk.toString()));
        child.stderr.on('data', (chunk: Buffer) => (output += chunk.toString()));
        child.on('error', reject);
        child.on('close', (code) => resolve({ code, output }));
    });
}

export async function runFlip(reference: FloatImage, test: FloatImage, options: FlipOptions): Promise<FlipResult> {
    const folder = await mkdtemp(join(tmpdir(), 'ddgi-flip-'));
    try {
        await writeFile(join(folder, 'reference.exr'), writeExr(reference));
        await writeFile(join(folder, 'test.exr'), writeExr(test));
        const args = [
            // -E: ignore PYTHON* environment variables. (Not -I: that also hides the per-user
            // site-packages where flip-evaluator lives.) The working directory is a fresh temp
            // folder holding only the images, so nothing in the repo can shadow a module.
            '-E',
            '-m',
            'flip_evaluator.flip_python_api',
            '-r',
            'reference.exr',
            '-t',
            'test.exr',
            '-v',
            '2',
            '-b',
            'flip',
            '-d',
            'out',
            '-tm',
            options.toneMapper,
            '-nexm', // no exposure map: only the error map is written
        ];
        if (options.pixelsPerDegree !== undefined) args.push('-ppd', String(options.pixelsPerDegree));
        if (options.startExposure !== undefined) args.push('-cstart', String(options.startExposure));
        if (options.stopExposure !== undefined) args.push('-cstop', String(options.stopExposure));
        if (options.numExposures !== undefined) args.push('-n', String(options.numExposures));

        let result: { code: number | null; output: string };
        try {
            result = await run(options.python, args, folder);
        } catch (error) {
            throw new Error(`could not start Python ("${options.python}"): ${(error as Error).message}. Use --no-flip to skip FLIP.`);
        }
        if (result.code !== 0) {
            throw new Error(
                `FLIP failed (exit ${result.code}). Is the flip-evaluator package installed for this Python? (pip install flip-evaluator)\n${result.output}`,
            );
        }
        const parsed = parseFlipOutput(result.output);
        const outputFiles = await readdir(join(folder, 'out'));
        const errorMap = outputFiles.find((name) => name.endsWith('.png'));
        if (!errorMap) throw new Error(`FLIP wrote no error map; output folder had: ${outputFiles.join(', ')}`);
        return { ...parsed, errorMapPng: new Uint8Array(await readFile(join(folder, 'out', errorMap))) };
    } finally {
        await rm(folder, { recursive: true, force: true });
    }
}
