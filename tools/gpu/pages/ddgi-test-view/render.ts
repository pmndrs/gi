/**
 * Renders DDGI's engine-free test view to EXR and compares it with the Cycles
 * references (T1.5a done-when 2).
 *
 *   node tools/gpu/pages/ddgi-test-view/render.ts [--scene cornell-box-point] [--camera 1]
 *        [--frames 600] [--width 512] [--height 512] [--gpu nvidia|intel] [--keep-light-range]
 *        [--suffix -run2] [--settings '{"viewBias": 0}']
 *
 * Writes `bench/references/<scene>/<camera>/ddgi-indirect<suffix>.exr` and
 * `ddgi-full<suffix>.exr` (gitignored, next to the references), then runs
 * `tools/compare` against `cycles-indirect.exr` and `cycles-full.exr` per
 * pixel and with 8 × 8 blocks, writing heatmaps there too. Exits non-zero if
 * the indirect image's mean is more than ±10% from the reference's (the M1
 * accuracy target), unless the scene has emissive surfaces (then indirect
 * isn't comparable; the full image is the one to read).
 *
 * It serves the test pages with its own vite server, so it can add one route:
 * `job.html`, which sets `window.ddgiTestViewJob` and loads the same
 * `main.ts` that `pnpm test:gpu` runs.
 */

import { spawnSync } from 'node:child_process';
import { existsSync } from 'node:fs';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';
import { createServer } from 'vite';
import { readExr, writeExr } from '../../../compare/exr.ts';
import { installShutdownHandlers, registerCleanup, runCleanups } from '../../cleanup.ts';
import type { GpuChoice } from '../../contract.ts';
import { SITE_ROOT } from '../../server.ts';
import { BrowserSession } from '../../session.ts';
import type { TestViewJob } from './main.ts';

const here = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(here, '..', '..', '..', '..');
const MEAN_TOLERANCE = 0.1;

function decodeFloats(base64: string): Float32Array {
    const bytes = Buffer.from(base64, 'base64');
    return new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4).slice();
}

function mean(values: Float32Array): number {
    let sum = 0;
    for (const value of values) sum += value;
    return sum / values.length;
}

async function main(): Promise<void> {
    const { values } = parseArgs({
        options: {
            scene: { type: 'string', default: 'cornell-box-point' },
            camera: { type: 'string', default: '1' },
            frames: { type: 'string', default: '600' },
            width: { type: 'string', default: '512' },
            height: { type: 'string', default: '512' },
            gpu: { type: 'string', default: 'nvidia' },
            'keep-light-range': { type: 'boolean' },
            suffix: { type: 'string', default: '' },
            settings: { type: 'string', default: '{}' },
        },
    });
    if (values.gpu !== 'nvidia' && values.gpu !== 'intel') throw new Error('--gpu must be nvidia or intel');
    const gpu: GpuChoice = values.gpu;
    const manifest = JSON.parse(await readFile(join(repoRoot, 'scenes', 'manifests', `${values.scene}.json`), 'utf8')) as { id: string; cameras: { name: string }[] };
    const camera = Number(values.camera);
    const cameraName = manifest.cameras[camera - 1]?.name;
    if (!cameraName) throw new Error(`${values.scene} has no camera ${values.camera}`);
    const outDir = join(repoRoot, 'bench', 'references', manifest.id, cameraName);
    const referenceIndirect = join(outDir, 'cycles-indirect.exr');
    const referenceFull = join(outDir, 'cycles-full.exr');
    const referenceIndirectMean = existsSync(referenceIndirect) ? mean(readExr(await readFile(referenceIndirect)).rgb) : null;

    const job: TestViewJob = {
        scene: manifest.id,
        camera,
        width: Number(values.width),
        height: Number(values.height),
        frames: Number(values.frames),
        ignoreLightRange: !values['keep-light-range'],
        exportImages: true,
        referenceIndirectMean: null,
        settings: JSON.parse(values.settings ?? '{}') as TestViewJob['settings'],
    };

    const server = await createServer({
        root: SITE_ROOT,
        configFile: false,
        logLevel: 'warn',
        appType: 'mpa',
        optimizeDeps: { noDiscovery: true, include: [] },
        server: { host: '127.0.0.1', port: 5292, strictPort: false, hmr: false, watch: null, fs: { allow: [repoRoot] } },
        plugins: [
            {
                name: 'ddgi-test-view-job',
                configureServer(devServer) {
                    devServer.middlewares.use('/pages/ddgi-test-view/job.html', (_request, response) => {
                        response.setHeader('content-type', 'text/html');
                        response.end(
                            `<!doctype html><html><head><meta charset="utf-8" /><title>ddgi-test-view job</title></head><body>` +
                                `<script>window.ddgiTestViewJob = ${JSON.stringify(job)};</script>` +
                                `<script type="module" src="/pages/ddgi-test-view/main.ts"></script></body></html>`,
                        );
                    });
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
    let result;
    try {
        session = await BrowserSession.launch({ serverOrigin: origin, chromePath: null, gpu, withoutGpu: false });
        console.log(`test view: ${job.scene} / ${cameraName}, ${job.frames} frames, ${job.width}x${job.height}, light range ${job.ignoreLightRange ? 'off (as Cycles)' : 'on'}, on ${gpu}`);
        const timeoutMs = 10 * 60_000;
        result = await session.runPage(
            { name: 'ddgi-test-view-job', urlPath: '/pages/ddgi-test-view/job.html', description: 'DDGI test view', expectFailure: false, expectMessage: null, manual: false, timeoutMs },
            { timeoutMs, expectFailure: false, screenshotDir: join(outDir, 'logs'), gpuLabel: gpu },
        );
    } finally {
        await (session as BrowserSession | null)?.close();
    }
    if (!result.verdict.acceptable || !result.report) throw new Error(`the page failed:\n  ${result.verdict.reasons.join('\n  ')}`);
    const data = result.report.data as Record<string, unknown>;
    const image = (key: string) => ({ width: job.width, height: job.height, rgb: decodeFloats(String(data[key])) });
    const indirect = image('indirectBase64');
    const full = image('fullBase64');

    await mkdir(outDir, { recursive: true });
    const suffix = values.suffix ?? '';
    const indirectFile = join(outDir, `ddgi-indirect${suffix}.exr`);
    const fullFile = join(outDir, `ddgi-full${suffix}.exr`);
    await writeFile(indirectFile, writeExr(indirect));
    await writeFile(fullFile, writeExr(full));
    const summary = { ...data, indirectBase64: undefined, fullBase64: undefined, adapter: result.report.adapter };
    await writeFile(join(outDir, `ddgi${suffix}.json`), JSON.stringify(summary, null, 2));
    console.log(`wrote ${indirectFile} and ${fullFile}`);
    console.log(`adapter: ${result.report.adapter?.vendor} / ${result.report.adapter?.architecture}; converged in ${String(data.convergeMs)} ms; distance format ${String(data.distanceFormat)}`);

    const compare = (reference: string, test: string, label: string, extra: string[]): void => {
        if (!existsSync(reference)) {
            console.log(`no ${reference}; skipping the ${label} comparison`);
            return;
        }
        console.log(`\n== ${label} ==`);
        const run = spawnSync(process.execPath, [join(repoRoot, 'tools', 'compare', 'index.ts'), reference, test, ...extra], { encoding: 'utf8', cwd: repoRoot });
        process.stdout.write(run.stdout);
        process.stderr.write(run.stderr);
    };
    const prefix = (name: string): string => join(outDir, `compare-cycles-vs-ddgi-${name}${suffix}`);
    compare(referenceIndirect, indirectFile, 'indirect, per pixel', ['--out', prefix('indirect')]);
    compare(referenceIndirect, indirectFile, 'indirect, 8x8 blocks', ['--downsample', '8', '--out', prefix('indirect-blocks8')]);
    compare(referenceFull, fullFile, 'full, per pixel (not gating)', ['--out', prefix('full')]);
    compare(referenceFull, fullFile, 'full, 8x8 blocks (not gating)', ['--downsample', '8', '--out', prefix('full-blocks8')]);

    if (data.emissiveSurfaces === true) {
        console.log(
            "\nthe scene has emissive surfaces: DDGI counts their light as GI and Cycles' indirect pass does not, " +
                'so the indirect numbers above are not comparable and do not gate (DECISIONS 2026-11-04)',
        );
    } else if (referenceIndirectMean !== null) {
        const bias = mean(indirect.rgb) / referenceIndirectMean - 1;
        console.log(`\nindirect mean ${mean(indirect.rgb).toFixed(6)} against Cycles ${referenceIndirectMean.toFixed(6)}: ${(bias * 100).toFixed(2)}% (target ±${MEAN_TOLERANCE * 100}%)`);
        if (Math.abs(bias) > MEAN_TOLERANCE) throw new Error('the indirect mean is outside the M1 accuracy target');
    }
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
