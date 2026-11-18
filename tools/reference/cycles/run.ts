/**
 * Runs the Cycles reference renderer (render.py) in Blender 4.5.
 *
 *   node tools/reference/cycles/run.ts --scene cornell-box [--camera 1 | --from-shot <cycles.json>] --samples 4096
 *        [--width 1920] [--height 1080] [--seed 0] [--name-suffix -seed1]
 *        [--device gpu|cpu] [--threads N] [--materials-json <GI material table> | --textured]
 *        [--pixel-filter blackman-harris|gaussian|box] [--filter-width 1.5] [--denoise]
 *   node tools/reference/cycles/run.ts --calibrate     known-answer checks of the unit conversions
 *
 * Output: bench/references/<scene>/<camera name>/cycles-{full,indirect,direct,lights-direct}.exr
 * and cycles.json, plus one line appended to bench/references/render-log.csv.
 * Then it checks the lights-only render where the answer is known and adds
 * the parts up into `cycles-gi.exr`, the quantity DDGI's test view shows
 * (gi-composite.ts). A failed check stops with exit code 1 and writes no GI
 * image. With `--materials-json` (render.py: every material gets the table's
 * constant GI albedo and emission) the files are cycles-untextured-* and the
 * log's tool column says `cycles-untextured`; with `--textured` (the glTF's own
 * materials as M4's GI sees them, render.py) they are cycles-textured-* and
 * `cycles-textured`. When the images are written, their sha256s go into the
 * json as `outputSha256`, beside render.py's record of the source files.
 *
 * Every render stores its camera, size, display settings, scene hashes,
 * commit and render settings as `shot` in its json (render.py; README
 * "Stored camera"), which the viewer's Compare and the DDGI test pages use.
 * `--denoise` adds a denoised viewing copy, `cycles-full-denoised.exr`, and
 * records its bias against the raw image (`denoiseCheck`); the raw EXRs are
 * what accuracy is measured on.
 *
 * A GPU render (the default, OptiX) holds the GPU bench lock
 * (`$TEMP/ddgi-gpu-bench.lock`) for the whole render. A CPU render
 * (`--device cpu --threads 12`) holds only the CPU-heavy lock
 * (`$TEMP/ddgi-cpu-heavy.lock`) and leaves 4 of the 16 threads free so GPU
 * benchmarks can still submit. Ctrl-C or SIGTERM kills Blender and releases
 * the lock. See tools/reference/README.md, "Bench lock".
 */

import { spawn, type ChildProcess } from 'node:child_process';
import { createHash } from 'node:crypto';
import { appendFile, mkdir, readFile, rm, stat, writeFile } from 'node:fs/promises';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { findBlender } from '../../assets/fetch.ts';
import { readExr, writeExr } from '../../compare/exr.ts';
import { installShutdownHandlers, registerCleanup } from '../../gpu/cleanup.ts';
import { withBenchLock } from '../bench-lock.ts';
import { compareImages, downsample } from '../../compare/metrics.ts';
import { benchLockFor, renderDevice } from './device.ts';
import { checkLightsOnlyImage, composeGiImage, type LightsOnlyExpectation } from './gi-composite.ts';

const repoRoot = resolve(dirname(fileURLToPath(import.meta.url)), '..', '..', '..');
const renderScript = join(repoRoot, 'tools', 'reference', 'cycles', 'render.py');
const calibrateScript = join(repoRoot, 'tools', 'reference', 'cycles', 'calibrate.py');
const renderLog = join(repoRoot, 'bench', 'references', 'render-log.csv');

/** Stops Blender and its helper processes; used when this script is interrupted. */
function killBlender(child: ChildProcess): Promise<void> {
    if (child.pid === undefined || child.exitCode !== null) return Promise.resolve();
    return new Promise((resolvePromise) => {
        if (process.platform === 'win32') {
            // /T includes child processes; plain kill() would leave them running.
            const taskkill = spawn('taskkill', ['/PID', String(child.pid), '/T', '/F'], { windowsHide: true, stdio: 'ignore' });
            taskkill.on('error', () => resolvePromise());
            taskkill.on('exit', () => resolvePromise());
        } else {
            child.kill('SIGKILL');
            resolvePromise();
        }
    });
}

/**
 * Blender 4.5.0 is pinned: the compositor and glTF importer APIs used here differ
 * in 5.x, and every reference so far was rendered with 4.5.0. `findBlender`
 * matches all three parts of the version.
 */
const BLENDER_VERSION = '4.5.0';

async function main(): Promise<void> {
    const forwarded = process.argv.slice(2);
    const calibrate = forwarded.includes('--calibrate');
    if (forwarded.length === 0 || forwarded.includes('--help')) {
        console.log('usage: node tools/reference/cycles/run.ts --scene <id> [--camera 1 | --from-shot cycles.json] --samples N [--width 1920] [--height 1080] [--seed N] [--name-suffix S] [--device gpu|cpu] [--threads N] [--materials-json file | --textured] [--pixel-filter blackman-harris|gaussian|box] [--filter-width 1.5] [--denoise] [--exposure-ev N] [--tone-mapping aces]');
        process.exit(forwarded.length === 0 ? 1 : 0);
    }
    // render.py's default is the GPU (OptiX); `--device cpu` and `--device=cpu` both mean the CPU.
    const device = renderDevice(forwarded);
    const blender = await findBlender(BLENDER_VERSION);
    console.log(`Blender ${blender.version} at ${blender.path}, device ${device}`);

    const args = [
        '--background',
        // Skips the user profile and its add-ons: it makes the render repeatable and the exit quick.
        '--factory-startup',
        '--python-exit-code',
        '1',
        '--python',
        calibrate ? calibrateScript : renderScript,
        '--',
        ...forwarded,
    ];

    const render = (): Promise<string | null> =>
        new Promise((resolvePromise, reject) => {
            const child = spawn(blender.path, args, { stdio: ['ignore', 'pipe', 'inherit'] });
            const releaseKill = registerCleanup(() => killBlender(child));
            let result: string | null = null;
            let pending = '';
            let lastProgress = 0;
            child.stdout.on('data', (chunk: Buffer) => {
                pending += chunk.toString();
                const lines = pending.split(/\r?\n/);
                pending = lines.pop() ?? '';
                for (const line of lines) {
                    if (line.startsWith('REFERENCE_RESULT ')) {
                        result = line.slice('REFERENCE_RESULT '.length);
                    } else if (line.startsWith('Fra:') || line.startsWith('Time:') || line.startsWith('Saved:')) {
                        // Blender prints a line per update; show one every 20 s.
                        if (Date.now() - lastProgress > 20_000) {
                            lastProgress = Date.now();
                            console.log(line.replace(/^Fra:\d+ Mem:[^|]*\| /, ''));
                        }
                    } else if (line.trim()) {
                        console.log(line);
                    }
                }
            });
            child.on('error', (error) => {
                void releaseKill();
                reject(error);
            });
            child.on('close', (code) => {
                void releaseKill();
                if (code === 0) resolvePromise(result ?? '');
                else reject(new Error(`Blender exited with code ${code}`));
            });
        });

    const scene = forwarded[forwarded.indexOf('--scene') + 1] ?? '?';
    // GPU renders hold the GPU bench lock for the whole render; CPU renders hold only the CPU-heavy
    // lock (lead's rule, 2026-11-05: a long OptiX render keeps the card hot and blocks every GPU
    // timing lane, so references move to the CPU with `--threads 12`, leaving cores for GPU
    // benchmarks to submit, while CPU-timed benchmarks still get the CPU to themselves).
    const result = await withBenchLock(render, { lock: benchLockFor(device), what: calibrate ? 'cycles calibrate' : `cycles ${scene}` });
    if (calibrate) return;
    if (!result) throw new Error('Blender finished without reporting a result');
    const meta = JSON.parse(result);

    await mkdir(dirname(renderLog), { recursive: true });
    const exists = await stat(renderLog).then(() => true, () => false);
    if (!exists) {
        await writeFile(renderLog, 'finishedAt,tool,scene,camera,width,height,samples,seed,device,threads,loadSeconds,renderSeconds,triangles\n');
    }
    await appendFile(
        renderLog,
        `${meta.finishedAt},${logToolName(meta)},${meta.scene},${meta.camera.name},${meta.width},${meta.height},${meta.samplesPerPixel},${meta.seed},${meta.device},${meta.threads},${meta.loadSeconds},${meta.renderSeconds},${meta.triangles}\n`,
    );
    meta.files.push(await writeGiImage(meta.outDir, meta.giComposite));
    const metaPath = join(meta.outDir, meta.jsonFile);
    const stored = JSON.parse(await readFile(metaPath, 'utf8'));
    if (meta.denoisedFile) {
        // The viewing copy's bias against the raw image it was made from, stored with the render.
        meta.denoiseCheck = await checkDenoised(meta.outDir, meta.files[0], meta.denoisedFile);
        stored.denoiseCheck = meta.denoiseCheck;
        stored.shot.render.denoised.check = meta.denoiseCheck;
    }
    // Every image this render wrote, with its checksum, so a copy or a later reader can tell it is the same file.
    stored.files = meta.files;
    stored.outputSha256 = Object.fromEntries(await Promise.all(meta.files.map(async (file: string) => [file, await fileSha256(join(meta.outDir, file))])));
    await writeFile(metaPath, JSON.stringify(stored, null, 2));
    console.log(`done: ${meta.files.join(', ')} in ${meta.renderSeconds} s (${meta.samplesPerPixel} samples, ${meta.width}x${meta.height}, ${meta.device})`);
}

/** The render log's tool column: which material mode made the render (render.py's `mode`). */
function logToolName(meta: { mode?: string; giMaterials?: unknown }): string {
    if (meta.mode === 'textured') return 'cycles-textured';
    return meta.giMaterials ? 'cycles-untextured' : 'cycles';
}

async function fileSha256(path: string): Promise<string> {
    return createHash('sha256').update(await readFile(path)).digest('hex');
}

/**
 * How far the denoised viewing copy is from the raw full image: overall bias
 * (sum against sum) and the mean relative error of 8 x 8 blocks. Denoised
 * images are for viewing; accuracy numbers come from the raw EXRs, and this
 * says how much a denoised image would move them.
 */
async function checkDenoised(outDir: string, rawName: string, denoisedName: string): Promise<{ bias: number; blockError: number; blockSize: number }> {
    const raw = readExr(await readFile(join(outDir, rawName)));
    const denoised = readExr(await readFile(join(outDir, denoisedName)));
    const blockSize = 8;
    const pixels = compareImages(raw.rgb, denoised.rgb);
    const blocks = compareImages(downsample(raw.rgb, raw.width, raw.height, blockSize).rgb, downsample(denoised.rgb, denoised.width, denoised.height, blockSize).rgb);
    console.log(`denoised viewing copy against the raw image: bias ${(pixels.bias * 100).toFixed(3)}%, ${blockSize}x${blockSize} block error ${(blocks.meanRelativeError * 100).toFixed(3)}%`);
    return { bias: pixels.bias, blockError: blocks.meanRelativeError, blockSize };
}

/** render.py's recipe for the DDGI-comparable image: the sum of `add` minus `subtract` (all EXRs in `outDir`). */
interface GiComposite {
    readonly output: string;
    readonly add: readonly string[];
    readonly subtract: string;
    /** The one of `add` that is the normal direct image, which the lights-only image is checked against. */
    readonly direct: string;
    readonly lightsOnlyExpected: LightsOnlyExpectation;
}

/**
 * Writes the render's GI image: Cycles' indirect + direct minus the lights'
 * direct light (render.py, `render_with_gi_parts`), which is what DDGI's field
 * holds (sky and emission seen straight from a surface, and every bounce).
 * First checks the lights-only image (gi-composite.ts) and throws, writing
 * nothing, if it isn't what it must be. Returns the file name.
 */
async function writeGiImage(outDir: string, recipe: GiComposite): Promise<string> {
    const read = async (name: string) => readExr(await readFile(join(outDir, name)));
    const add = await Promise.all(recipe.add.map(read));
    const lightsDirect = await read(recipe.subtract);
    const direct = add[recipe.add.indexOf(recipe.direct)];
    if (!direct) throw new Error(`the GI recipe's direct image ${recipe.direct} is not one of ${recipe.add.join(', ')}`);
    const check = checkLightsOnlyImage(recipe.lightsOnlyExpected, direct, lightsDirect);
    console.log(check.summary);
    if (!check.passed) {
        // A GI image left from an earlier render would now be mistaken for this one's.
        await rm(join(outDir, recipe.output), { force: true });
        throw new Error(`the lights-only render is not lights-only (or lost the lights' own light); ${recipe.output} not written.\n  ${check.summary}`);
    }
    await writeFile(join(outDir, recipe.output), writeExr(composeGiImage(add, lightsDirect)));
    return recipe.output;
}

installShutdownHandlers();
main().catch((error) => {
    console.error(error instanceof Error ? error.message : error);
    process.exit(1);
});
