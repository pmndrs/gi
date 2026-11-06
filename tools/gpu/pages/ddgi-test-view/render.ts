/**
 * Renders DDGI's engine-free test view to EXR and compares it with the Cycles
 * references (T1.5a done-when 2).
 *
 *   node tools/gpu/pages/ddgi-test-view/render.ts [--scene cornell-box-point] [--camera 1]
 *        [--frames 600] [--width W] [--height H] [--gpu nvidia|intel] [--keep-light-range]
 *        [--volume manifest|in-air] [--suffix -run2] [--settings '{"viewBias": 0}']
 *
 * Camera and size: those the Cycles reference was rendered with (`shot` in
 * its cycles.json, T1.13: 1920 x 1080), so DDGI and Cycles show exactly the
 * same view. `--width`/`--height` override the size only. A reference
 * rendered before shots were stored falls back to the bookmark at 1920 x 1080.
 *
 * `--volume in-air` (Cornell only) uses `CORNELL_POINT_IN_AIR_VOLUME`, T1.5a's
 * measurement, as an extra report: its files get `-in-air` and it never
 * gates. The gate is the manifest's own volume (DECISIONS 2026-11-05).
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
import { CORNELL_POINT_IN_AIR_VOLUME } from '../ddgi-shared/volumes.ts';
import { DEFAULT_REFERENCE_SIZE, readReferenceShot, referenceSize, shotView } from '../../../reference/shot.ts';

const here = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(here, '..', '..', '..', '..');
const MEAN_TOLERANCE = 0.1;

function decodeFloats(base64: string): Float32Array {
    const bytes = Buffer.from(base64, 'base64');
    return new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4).slice();
}

/**
 * `--suffix -air`: parseArgs refuses an option value that starts with a dash,
 * so such a value is joined to its flag (`--suffix=-air`) first.
 */
function joinDashValues(args: readonly string[], flag: string): string[] {
    const joined: string[] = [];
    for (let index = 0; index < args.length; index++) {
        const next = args[index + 1];
        if (args[index] === flag && next !== undefined && next.startsWith('-')) {
            joined.push(`${flag}=${next}`);
            index++;
        } else {
            joined.push(args[index]);
        }
    }
    return joined;
}

function mean(values: Float32Array): number {
    let sum = 0;
    for (const value of values) sum += value;
    return sum / values.length;
}

async function main(): Promise<void> {
    const { values } = parseArgs({
        args: joinDashValues(process.argv.slice(2), '--suffix'),
        options: {
            volume: { type: 'string', default: 'manifest' },
            scene: { type: 'string', default: 'cornell-box-point' },
            camera: { type: 'string', default: '1' },
            frames: { type: 'string', default: '600' },
            width: { type: 'string' },
            height: { type: 'string' },
            gpu: { type: 'string', default: 'nvidia' },
            'keep-light-range': { type: 'boolean' },
            suffix: { type: 'string', default: '' },
            settings: { type: 'string', default: '{}' },
        },
    });
    if (values.gpu !== 'nvidia' && values.gpu !== 'intel') throw new Error('--gpu must be nvidia or intel');
    if (values.volume !== 'manifest' && values.volume !== 'in-air') throw new Error('--volume must be manifest or in-air');
    const inAir = values.volume === 'in-air';
    if (inAir && values.scene !== 'cornell-box-point') throw new Error('--volume in-air exists for cornell-box-point only');
    const gpu: GpuChoice = values.gpu;
    const manifest = JSON.parse(await readFile(join(repoRoot, 'scenes', 'manifests', `${values.scene}.json`), 'utf8')) as { id: string; cameras: { name: string }[] };
    const camera = Number(values.camera);
    const cameraName = manifest.cameras[camera - 1]?.name;
    if (!cameraName) throw new Error(`${values.scene} has no camera ${values.camera}`);
    const outDir = join(repoRoot, 'bench', 'references', manifest.id, cameraName);
    const referenceIndirect = join(outDir, 'cycles-indirect.exr');
    const referenceFull = join(outDir, 'cycles-full.exr');
    const referenceIndirectMean = existsSync(referenceIndirect) ? mean(readExr(await readFile(referenceIndirect)).rgb) : null;
    // The reference's own camera and size (T1.13), so the images line up pixel for pixel.
    const shot = await readReferenceShot(repoRoot, outDir, 'cycles.json');
    // A reference from before T1.13 has no stored camera: the bookmark, at that reference's size.
    const size = shot?.camera ?? (await referenceSize(outDir, 'cycles.json')) ?? DEFAULT_REFERENCE_SIZE;
    if (!shot) console.warn(`no stored camera in ${join(outDir, 'cycles.json')} (rendered before T1.13, or missing): using the bookmark at ${size.width} x ${size.height}`);
    const width = Number(values.width ?? size.width);
    const height = Number(values.height ?? size.height);

    const job: TestViewJob = {
        scene: manifest.id,
        camera,
        ...(shot ? { view: shotView(shot) } : {}),
        width,
        height,
        frames: Number(values.frames),
        ignoreLightRange: !values['keep-light-range'],
        exportImages: true,
        referenceIndirectMean: null,
        settings: {
            ...(JSON.parse(values.settings ?? '{}') as TestViewJob['settings']),
            ...(inAir ? { volume: CORNELL_POINT_IN_AIR_VOLUME } : {}),
        },
        // One volume per run here; the page's own in-air extra is for pnpm test:gpu.
        extraVolumes: [],
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
    const suffix = `${inAir ? '-in-air' : ''}${values.suffix ?? ''}`;
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

    if (inAir) {
        const value = mean(indirect.rgb);
        const versus = referenceIndirectMean === null ? '' : ` against Cycles ${referenceIndirectMean.toFixed(6)}: ${((value / referenceIndirectMean - 1) * 100).toFixed(2)}%`;
        console.log(`
in-air volume, an extra report that never gates: indirect mean ${value.toFixed(6)}${versus}`);
    } else if (data.emissiveSurfaces === true) {
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
