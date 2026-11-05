/**
 * Runs the Cycles reference renderer (render.py) in Blender 4.5.
 *
 *   node tools/reference/cycles/run.ts --scene cornell-box [--camera 1] [--samples 1024]
 *        [--width 512] [--height 512] [--seed 0] [--name-suffix -seed1]
 *        [--device cpu|gpu] [--threads N] [--materials-json <GI material table>]
 *   node tools/reference/cycles/run.ts --calibrate     known-answer checks of the unit conversions
 *
 * Output: bench/references/<scene>/<camera name>/cycles-{full,indirect,direct,lights-direct}.exr
 * and cycles.json, plus one line appended to bench/references/render-log.csv.
 * Then it checks the lights-only render where the answer is known and adds
 * the parts up into `cycles-gi.exr`, the quantity DDGI's test view shows
 * (gi-composite.ts). A failed check stops with exit code 1 and writes no GI
 * image. With `--materials-json` (render.py: every material gets the table's
 * constant GI albedo and emission) the files are cycles-untextured-* and the
 * log's tool column says `cycles-untextured`.
 *
 * Every render holds the shared bench lock (`$TEMP/ddgi-gpu-bench.lock`), CPU
 * renders included: 16 busy threads skew other agents' build-time and CPU-side
 * numbers just as a GPU job skews GPU numbers. One machine-heavy job at a time.
 * Ctrl-C or SIGTERM kills Blender and releases the lock. The default device is
 * the CPU; `--device gpu` uses OptiX. See tools/reference/README.md.
 */

import { spawn, type ChildProcess } from 'node:child_process';
import { appendFile, mkdir, readFile, rm, stat, writeFile } from 'node:fs/promises';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { findBlender } from '../../assets/fetch.ts';
import { readExr, writeExr } from '../../compare/exr.ts';
import { installShutdownHandlers, registerCleanup } from '../../gpu/cleanup.ts';
import { withBenchLock } from '../bench-lock.ts';
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
        console.log('usage: node tools/reference/cycles/run.ts --scene <id> [--camera 1] [--samples N] [--width W] [--height H] [--seed N] [--name-suffix S] [--device cpu|gpu] [--threads N] [--materials-json file]');
        process.exit(forwarded.length === 0 ? 1 : 0);
    }
    const device = forwarded[forwarded.indexOf('--device') + 1] === 'gpu' ? 'gpu' : 'cpu';
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
    const result = await withBenchLock(render, { what: calibrate ? 'cycles calibrate' : `cycles ${scene}` });
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
        `${meta.finishedAt},${meta.giMaterials ? 'cycles-untextured' : 'cycles'},${meta.scene},${meta.camera.name},${meta.width},${meta.height},${meta.samplesPerPixel},${meta.seed},${meta.device},${meta.threads},${meta.loadSeconds},${meta.renderSeconds},${meta.triangles}\n`,
    );
    meta.files.push(await writeGiImage(meta.outDir, meta.giComposite));
    console.log(`done: ${meta.files.join(', ')} in ${meta.renderSeconds} s (${meta.samplesPerPixel} samples, ${meta.width}x${meta.height}, ${meta.device})`);
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
