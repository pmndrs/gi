/**
 * M4, T4.5: runs the `m4-textured-accuracy` page (see its main.ts) for each
 * scene with textured Cycles references and compares GI's indirect light
 * with them.
 *
 *   node tools/gpu/pages/m4-textured-accuracy/run.ts [--scenes a,b,...] [--gpu nvidia|intel]
 *        [--references <dir>] [--out <dir>] [--compare-only] [--timeout-minutes 30]
 *
 * For each scene, every camera folder under `<references>/<scene>/` with a
 * `cycles-textured-gi.exr` is a view: its json's stored shot gives the
 * camera and size. One page run per scene (DDGI converges once, then each
 * view is read back), under the GPU bench lock (minutes of full GPU load).
 * Bistro (both) loads three's textures with T2.17's cap at 1024 (they don't
 * fit the 6 GB RTX uncompressed on `m4`); GI's own store is 256² a layer
 * either way, so GI's answer doesn't change.
 *
 * Writes `<out>/<scene>/<camera>/ddgi-textured-gi.exr` (the GI term in the
 * reference's row order), comparison PNGs beside it, and
 * `<out>/m4-textured-accuracy-<gpu>.json` with every row. `--compare-only`
 * compares the EXRs already written (no GPU).
 *
 * What is compared, as M1's HD accuracy (DECISIONS 2026-11-06; T1.14's
 * `three-composite-ao/run.ts --upscale performance --resolution-scale 0.5`,
 * then `tools/compare --downsample 8`):
 * - mean error: sum(GI) / sum(Cycles) − 1 over every pixel and channel
 *   (SPEC's bar ±10%);
 * - FLIP on 8 × 8 block means (HDR-FLIP, 67 / 8 pixels per degree; bar
 *   0.15), with the 8 × 8 block error's mean and p95 beside it;
 * - the reference's own noise (its json's `noise.gi`, T4.4), when measured.
 */

import { spawnSync } from 'node:child_process';
import { existsSync } from 'node:fs';
import { mkdir, readdir, readFile, writeFile } from 'node:fs/promises';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';
import { createServer } from 'vite';
import { readExr, writeExr } from '../../../compare/exr.ts';
import { withBenchLock } from '../../../reference/bench-lock.ts';
import { installShutdownHandlers, registerCleanup, runCleanups } from '../../cleanup.ts';
import type { GpuChoice } from '../../contract.ts';
import { SITE_ROOT } from '../../server.ts';
import { BrowserSession } from '../../session.ts';
import type { AccuracyView, TexturedAccuracyJob } from './main.ts';

const here = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(here, '..', '..', '..', '..');

/** The scenes T4.5 measures, in order (sponza-full only once its reference has landed). */
const DEFAULT_SCENES = ['textured-furnace', 'uv-board', 'alpha-checker', 'glass-wall', 'sponza', 'furnished-room', 'courtyard-tree', 'bistro-exterior', 'bistro-interior', 'sponza-full'];

/** T2.17's opt-in cap on three's own textures, for the scenes that need it on the 6 GB RTX. */
const TEXTURE_CAPS: Record<string, number> = { 'bistro-exterior': 1024, 'bistro-interior': 1024 };

const REFERENCE_PREFIX = 'cycles-textured';
const REFERENCE_IMAGE = `${REFERENCE_PREFIX}-gi.exr`;
const TEST_IMAGE = 'ddgi-textured-gi.exr';

/** SPEC's accuracy bar (mean error) and the FLIP bar (DECISIONS 2026-11-06). */
const MEAN_TOLERANCE = 0.1;
const FLIP_LIMIT = 0.15;
const BLOCK = 8;

interface ReferenceMeta {
    readonly width: number;
    readonly height: number;
    readonly samplesPerPixel: number;
    readonly shot?: { readonly camera: { readonly bookmark: string; readonly position: [number, number, number]; readonly target: [number, number, number]; readonly fovYDeg: number; readonly width: number; readonly height: number } };
    readonly noise?: { readonly gi: { readonly pixel: number; readonly block8: number } };
}

interface ViewReference {
    readonly camera: string;
    readonly directory: string;
    readonly meta: ReferenceMeta;
}

interface Row {
    readonly scene: string;
    readonly camera: string;
    readonly meanError: number;
    readonly flip8: number | null;
    readonly blockError8: number;
    readonly blockErrorP95: number;
    readonly referenceNoise: { readonly pixel: number; readonly block8: number } | null;
    readonly referenceSamples: number;
    readonly textureStoreMB: number | null;
    readonly probes: number | null;
    readonly withinMean: boolean;
    readonly withinFlip: boolean | null;
}

function decodeFloats(base64: string): Float32Array {
    const bytes = Buffer.from(base64, 'base64');
    return new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4).slice();
}

function flipRows(rgb: Float32Array, width: number, height: number): Float32Array {
    const flipped = new Float32Array(rgb.length);
    for (let y = 0; y < height; y++) flipped.set(rgb.subarray(y * width * 3, (y + 1) * width * 3), (height - 1 - y) * width * 3);
    return flipped;
}

/** sum |block(test) − block(reference)| / sum block(reference), over 8 × 8 blocks: used only to tell the read-back's row order. */
function blockError(test: Float32Array, reference: Float32Array, width: number, height: number): number {
    let absolute = 0;
    let total = 0;
    for (let by = 0; by + BLOCK <= height; by += BLOCK) {
        for (let bx = 0; bx + BLOCK <= width; bx += BLOCK) {
            let t = 0;
            let r = 0;
            for (let y = by; y < by + BLOCK; y++) {
                for (let x = bx; x < bx + BLOCK; x++) {
                    const pixel = (y * width + x) * 3;
                    t += test[pixel] + test[pixel + 1] + test[pixel + 2];
                    r += reference[pixel] + reference[pixel + 1] + reference[pixel + 2];
                }
            }
            absolute += Math.abs(t - r);
            total += r;
        }
    }
    return absolute / total;
}

/** The camera folders of a scene that hold a textured GI reference. */
async function viewReferences(referencesRoot: string, scene: string): Promise<ViewReference[]> {
    const sceneDirectory = join(referencesRoot, scene);
    if (!existsSync(sceneDirectory)) return [];
    const views: ViewReference[] = [];
    for (const entry of (await readdir(sceneDirectory, { withFileTypes: true })).filter((candidate) => candidate.isDirectory()).sort((a, b) => a.name.localeCompare(b.name))) {
        const directory = join(sceneDirectory, entry.name);
        if (!existsSync(join(directory, REFERENCE_IMAGE)) || !existsSync(join(directory, `${REFERENCE_PREFIX}.json`))) continue;
        const meta = JSON.parse(await readFile(join(directory, `${REFERENCE_PREFIX}.json`), 'utf8')) as ReferenceMeta;
        if (!meta.shot) throw new Error(`${directory}: the reference has no stored shot`);
        views.push({ camera: entry.name, directory, meta });
    }
    return views;
}

/** Runs the page once in its own vite server and Chrome, with the job injected; returns the page's report data. */
async function runPage(job: Partial<TexturedAccuracyJob>, gpu: GpuChoice, timeoutMs: number, logDir: string): Promise<Record<string, unknown>> {
    const pagePath = '/pages/m4-textured-accuracy/job.html';
    const server = await createServer({
        root: SITE_ROOT,
        configFile: false,
        logLevel: 'warn',
        appType: 'mpa',
        optimizeDeps: { noDiscovery: true, include: [] },
        server: { host: '127.0.0.1', port: 5295, strictPort: false, hmr: false, watch: null, fs: { allow: [repoRoot] } },
        plugins: [
            {
                name: 'm4-textured-accuracy-job',
                configureServer(devServer) {
                    devServer.middlewares.use(pagePath, (_request, response) => {
                        response.setHeader('content-type', 'text/html');
                        response.end(
                            `<!doctype html><html><head><meta charset="utf-8" /><title>m4-textured-accuracy job</title></head><body>` +
                                `<script>window.m4TexturedAccuracyJob = ${JSON.stringify(job)};</script>` +
                                `<script type="module" src="/pages/m4-textured-accuracy/main.ts"></script></body></html>`,
                        );
                    });
                    // Chrome asks for one; a 404 in the log fails the page.
                    devServer.middlewares.use('/favicon.ico', (_request, response) => {
                        response.statusCode = 204;
                        response.end();
                    });
                },
            },
        ],
    });
    registerCleanup(() => server.close());
    await server.listen();
    const origin = server.resolvedUrls?.local[0]?.replace(/\/$/, '');
    if (!origin) throw new Error('vite reported no address');
    let session: BrowserSession | null = null;
    try {
        session = await BrowserSession.launch({ serverOrigin: origin, chromePath: null, gpu, withoutGpu: false });
        const result = await session.runPage(
            { name: 'm4-textured-accuracy-job', urlPath: pagePath, description: 'M4 textured accuracy', expectFailure: false, expectMessage: null, manual: false, timeoutMs },
            { timeoutMs, expectFailure: false, screenshotDir: logDir, gpuLabel: gpu },
        );
        if (result.report?.message) console.log(`  page: ${result.report.message}`);
        if (!result.verdict.acceptable || !result.report) throw new Error(`the page failed:\n  ${result.verdict.reasons.join('\n  ')}`);
        return result.report.data as Record<string, unknown>;
    } finally {
        await (session as BrowserSession | null)?.close();
        await server.close();
    }
}

interface CompareJson {
    readonly metrics: { readonly meanRelativeError: number; readonly bias: number };
    readonly p95Error: number;
    readonly flip: { readonly mean: number } | null;
}

/** tools/compare on 8 × 8 blocks, as M1's FLIP was taken; writes its PNGs next to the test image. */
function compareBlocks(referencePath: string, testPath: string, outPrefix: string): CompareJson {
    const result = spawnSync(process.execPath, [join(repoRoot, 'tools', 'compare', 'index.ts'), referencePath, testPath, '--downsample', String(BLOCK), '--json', '--out', outPrefix], {
        encoding: 'utf8',
        maxBuffer: 64 * 2 ** 20,
    });
    if (result.status !== 0) throw new Error(`tools/compare failed (${result.status}):\n${result.stderr}\n${result.stdout}`);
    return JSON.parse(result.stdout) as CompareJson;
}

async function main(): Promise<void> {
    const { values } = parseArgs({
        options: {
            scenes: { type: 'string' },
            gpu: { type: 'string', default: 'nvidia' },
            references: { type: 'string', default: join(repoRoot, 'bench', 'references') },
            out: { type: 'string', default: join(repoRoot, 'bench', 'references') },
            'compare-only': { type: 'boolean', default: false },
            'timeout-minutes': { type: 'string', default: '30' },
        },
    });
    if (values.gpu !== 'nvidia' && values.gpu !== 'intel') throw new Error('--gpu must be nvidia or intel');
    const gpu: GpuChoice = values.gpu;
    const referencesRoot = resolve(values.references);
    const outRoot = resolve(values.out);
    const scenes = values.scenes ? values.scenes.split(',').map((scene) => scene.trim()).filter(Boolean) : DEFAULT_SCENES;
    const timeoutMs = Number(values['timeout-minutes']) * 60_000;
    const summaryPath = join(outRoot, `m4-textured-accuracy-${gpu}.json`);
    const earlier = existsSync(summaryPath) ? (JSON.parse(await readFile(summaryPath, 'utf8')) as { rows: Row[]; scenes: Record<string, unknown> }) : { rows: [], scenes: {} };
    const rows: Row[] = earlier.rows.filter((row) => !scenes.includes(row.scene));
    const sceneRecords: Record<string, unknown> = { ...earlier.scenes };

    for (const scene of scenes) {
        const views = await viewReferences(referencesRoot, scene);
        if (views.length === 0) {
            console.log(`${scene}: no textured reference under ${join(referencesRoot, scene)}; skipped`);
            continue;
        }
        const { width, height } = views[0].meta;
        for (const view of views) {
            if (view.meta.width !== width || view.meta.height !== height) throw new Error(`${scene}/${view.camera}: ${view.meta.width} x ${view.meta.height}, the others ${width} x ${height}`);
        }
        const outDirectory = (camera: string) => join(outRoot, scene, camera);
        let record: { probeCount: number; textureStoreBytes: number; recompiles: number; recompilesAfter: number; maxTextureSize: number | null } | null = null;
        if (!values['compare-only']) {
            const accuracyViews: AccuracyView[] = views.map((view) => {
                const camera = view.meta.shot!.camera;
                return { name: view.camera, position: camera.position, target: camera.target, fovYDeg: camera.fovYDeg };
            });
            const maxTextureSize = TEXTURE_CAPS[scene] ?? null;
            const job: Partial<TexturedAccuracyJob> = { scene, views: accuracyViews, width, height, maxTextureSize };
            console.log(`${scene}: ${views.map((view) => view.camera).join(', ')} on ${gpu}${maxTextureSize ? `, three's textures capped at ${maxTextureSize}` : ''}`);
            const data = await withBenchLock(() => runPage(job, gpu, timeoutMs, join(outRoot, 'logs')), { what: `m4-textured-accuracy ${scene} on ${gpu}` });
            const images = data.giTermBase64 as Record<string, string>;
            for (const view of views) {
                let test = decodeFloats(images[view.camera]);
                const reference = readExr(await readFile(join(view.directory, REFERENCE_IMAGE))).rgb;
                // The read-back's row order: keep the one that agrees with the reference.
                if (blockError(flipRows(test, width, height), reference, width, height) < blockError(test, reference, width, height)) test = flipRows(test, width, height);
                await mkdir(outDirectory(view.camera), { recursive: true });
                await writeFile(join(outDirectory(view.camera), TEST_IMAGE), writeExr({ width, height, rgb: test }));
            }
            record = {
                probeCount: data.probeCount as number,
                textureStoreBytes: data.textureStoreBytes as number,
                recompiles: data.recompiles as number,
                recompilesAfter: data.recompilesAfter as number,
                maxTextureSize,
            };
            sceneRecords[scene] = { ...record, finishedAt: new Date().toISOString() };
            console.log(`  ${record.probeCount} probes, texture store ${(record.textureStoreBytes / 2 ** 20).toFixed(1)} MB, recompiles ${record.recompiles} after the warm-up, ${record.recompilesAfter} after the views`);
        } else {
            record = (sceneRecords[scene] as typeof record) ?? null;
        }

        for (const view of views) {
            const testPath = join(outDirectory(view.camera), TEST_IMAGE);
            if (!existsSync(testPath)) throw new Error(`${testPath} is missing: run without --compare-only first`);
            const referencePath = join(view.directory, REFERENCE_IMAGE);
            const test = readExr(await readFile(testPath)).rgb;
            const reference = readExr(await readFile(referencePath)).rgb;
            let testSum = 0;
            let referenceSum = 0;
            for (let index = 0; index < test.length; index++) {
                testSum += test[index];
                referenceSum += reference[index];
            }
            const compared = compareBlocks(referencePath, testPath, join(outDirectory(view.camera), `compare-${REFERENCE_PREFIX}-vs-ddgi`));
            const meanError = testSum / referenceSum - 1;
            const flip8 = compared.flip?.mean ?? null;
            rows.push({
                scene,
                camera: view.camera,
                meanError,
                flip8,
                blockError8: compared.metrics.meanRelativeError,
                blockErrorP95: compared.p95Error,
                referenceNoise: view.meta.noise?.gi ?? null,
                referenceSamples: view.meta.samplesPerPixel,
                textureStoreMB: record ? record.textureStoreBytes / 2 ** 20 : null,
                probes: record ? record.probeCount : null,
                withinMean: Math.abs(meanError) <= MEAN_TOLERANCE,
                withinFlip: flip8 === null ? null : flip8 <= FLIP_LIMIT,
            });
        }
        await mkdir(outRoot, { recursive: true });
        await writeFile(summaryPath, JSON.stringify({ gpu, rows, scenes: sceneRecords, writtenAt: new Date().toISOString() }, null, 2));
    }

    const percent = (value: number, digits = 2) => `${value >= 0 ? '+' : ''}${(value * 100).toFixed(digits)}%`;
    console.log(`\nGI term against ${REFERENCE_IMAGE} on ${gpu} (mean error bar ±${MEAN_TOLERANCE * 100}%, FLIP 8x8 bar ${FLIP_LIMIT})`);
    console.log('scene              camera                         mean error  FLIP 8x8  block err  p95 block  ref noise (px / 8x8)  store MB  probes');
    for (const row of rows) {
        const noise = row.referenceNoise ? `${(row.referenceNoise.pixel * 100).toFixed(1)}% / ${(row.referenceNoise.block8 * 100).toFixed(2)}%` : 'not measured';
        console.log(
            `${row.scene.padEnd(18)} ${row.camera.padEnd(30)} ${percent(row.meanError).padStart(9)}${row.withinMean ? ' ' : '!'} ${(row.flip8 === null ? '-' : row.flip8.toFixed(3)).padStart(8)}${row.withinFlip === false ? '!' : ' '} ` +
                `${(row.blockError8 * 100).toFixed(1).padStart(8)}% ${(row.blockErrorP95 * 100).toFixed(0).padStart(8)}%  ${noise.padEnd(20)} ${(row.textureStoreMB === null ? '-' : row.textureStoreMB.toFixed(1)).padStart(8)}  ${row.probes ?? '-'}`,
        );
    }
    console.log(`\nwritten: ${summaryPath}`);
}

installShutdownHandlers();
main().then(
    async () => {
        await runCleanups();
        process.exit(0);
    },
    async (error) => {
        console.error(error instanceof Error ? error.message : error);
        await runCleanups();
        process.exit(1);
    },
);
