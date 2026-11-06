/**
 * Runs the `three-composite-ao` page (see its main.ts) and compares.
 *
 *   node tools/gpu/pages/three-composite-ao/run.ts --scene sponza|sponza-base|cornell-box-point
 *        [--gpu nvidia|intel] [--radii 0.1,0.25,0.5] [--samples 16] [--resolution-scale 1] [--suffix=-run2]
 *        [--ao-kind gtao|gtao-temporal|ssao] [--upscale quality|performance]
 *        [--reference cycles-untextured-gi-c23.exr] [--table gi-materials-cp1.json]
 *   node tools/gpu/pages/three-composite-ao/run.ts --mode timing --timing-part gtao|rays [--gpu nvidia]
 *        (time it under `node tools/reference/bench-lock.ts run --what ... -- node ...run.ts --mode timing`)
 *
 * Match mode writes `three-ao-gi<suffix>.exr` (the composite's GI term),
 * `three-ao-weight<suffix>.exr` and `three-ao-gtao-<radius><suffix>.exr`
 * next to the scene's Cycles reference, then prints, for no AO and for each
 * GTAO radius (GI term × AO, so the AO touches GI only): the bias against
 * the reference overall and split into curtain and other pixels (by the
 * weight, which is the GI albedo with the page's Lambert materials, matched
 * to the GI material table), the 8 × 8 block error, and the per-pixel
 * relative RMS. Timing mode prints the page's numbers and writes them to
 * `three-composite-ao-timing<suffix>.json` in bench/references/sponza.
 */

import { existsSync } from 'node:fs';
import { readFile, writeFile } from 'node:fs/promises';
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
import type { CompositeAoJob, CompositeAoScene } from './main.ts';

const here = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(here, '..', '..', '..', '..');

/** Where each scene's reference lives, its default reference and size, and its GI material table (for the curtain split). */
const SCENES: Record<CompositeAoScene, { directory: string; reference: string; width: number; height: number; table: string | null }> = {
    sponza: { directory: 'sponza/colonnade-from-the-floor', reference: 'cycles-untextured-gi-c23.exr', width: 640, height: 360, table: 'gi-materials-cp1.json' },
    'sponza-base': { directory: 'sponza/colonnade-from-the-floor', reference: 'cycles-untextured-gi-cbase23.exr', width: 640, height: 360, table: 'gi-materials-cp1.json' },
    'cornell-box-point': { directory: 'cornell-box-point/front', reference: 'cycles-gi.exr', width: 1920, height: 1080, table: null },
};

function decodeFloats(base64: string): Float32Array {
    const bytes = Buffer.from(base64, 'base64');
    return new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4).slice();
}

async function runPage(job: Partial<CompositeAoJob>, gpu: GpuChoice, timeoutMs: number): Promise<Record<string, unknown>> {
    const pagePath = '/pages/three-composite-ao/job.html';
    const server = await createServer({
        root: SITE_ROOT,
        configFile: false,
        logLevel: 'warn',
        appType: 'mpa',
        optimizeDeps: { noDiscovery: true, include: [] },
        server: { host: '127.0.0.1', port: 5294, strictPort: false, hmr: false, watch: null, fs: { allow: [repoRoot] } },
        plugins: [
            {
                name: 'three-composite-ao-job',
                configureServer(devServer) {
                    devServer.middlewares.use(pagePath, (_request, response) => {
                        response.setHeader('content-type', 'text/html');
                        response.end(
                            `<!doctype html><html><head><meta charset="utf-8" /><title>three-composite-ao job</title></head><body>` +
                                `<script>window.threeCompositeAoJob = ${JSON.stringify(job)};</script>` +
                                `<script type="module" src="/pages/three-composite-ao/main.ts"></script></body></html>`,
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
            { name: 'three-composite-ao-job', urlPath: pagePath, description: 'three composite AO', expectFailure: false, expectMessage: null, manual: false, timeoutMs },
            { timeoutMs, expectFailure: false, screenshotDir: join(repoRoot, 'bench', 'references', 'logs'), gpuLabel: gpu },
        );
        if (result.report?.message) console.log(`  page: ${result.report.message}`);
        if (!result.verdict.acceptable || !result.report) throw new Error(`the page failed:\n  ${result.verdict.reasons.join('\n  ')}`);
        return result.report.data as Record<string, unknown>;
    } finally {
        await (session as BrowserSession | null)?.close();
        await server.close();
    }
}

/** A GI albedo matches a table row when every channel agrees within this share. */
const ALBEDO_MATCH_TOLERANCE = 1e-3;

interface Stats {
    readonly bias: number;
    readonly curtains: number | null;
    readonly other: number | null;
    readonly blockError: number;
    readonly pixelRelativeRms: number;
}

/** Sum ratios (test / reference − 1) overall and by class, 8 × 8 block mean relative error, per-pixel relative RMS over surface pixels. */
function compare(test: Float32Array, reference: Float32Array, curtain: Uint8Array | null, surface: Uint8Array, width: number, height: number): Stats {
    let testSum = 0;
    let referenceSum = 0;
    const classSums = [0, 0, 0, 0]; // curtain test, curtain reference, other test, other reference
    let squared = 0;
    let referencePixelSum = 0;
    let surfacePixels = 0;
    for (let pixel = 0; pixel < width * height; pixel++) {
        const t = test[pixel * 3] + test[pixel * 3 + 1] + test[pixel * 3 + 2];
        const r = reference[pixel * 3] + reference[pixel * 3 + 1] + reference[pixel * 3 + 2];
        testSum += t;
        referenceSum += r;
        if (curtain) {
            const offset = curtain[pixel] ? 0 : 2;
            classSums[offset] += t;
            classSums[offset + 1] += r;
        }
        if (surface[pixel]) {
            squared += (t - r) * (t - r);
            referencePixelSum += r;
            surfacePixels++;
        }
    }
    let blockAbs = 0;
    let blockReference = 0;
    for (let by = 0; by + 8 <= height; by += 8) {
        for (let bx = 0; bx + 8 <= width; bx += 8) {
            let t = 0;
            let r = 0;
            for (let y = by; y < by + 8; y++) {
                for (let x = bx; x < bx + 8; x++) {
                    const pixel = y * width + x;
                    t += test[pixel * 3] + test[pixel * 3 + 1] + test[pixel * 3 + 2];
                    r += reference[pixel * 3] + reference[pixel * 3 + 1] + reference[pixel * 3 + 2];
                }
            }
            blockAbs += Math.abs(t - r);
            blockReference += r;
        }
    }
    return {
        bias: testSum / referenceSum - 1,
        curtains: curtain && classSums[1] > 0 ? classSums[0] / classSums[1] - 1 : null,
        other: curtain && classSums[3] > 0 ? classSums[2] / classSums[3] - 1 : null,
        blockError: blockAbs / blockReference,
        pixelRelativeRms: Math.sqrt(squared / surfacePixels) / (referencePixelSum / surfacePixels),
    };
}

function flipRows(rgb: Float32Array, width: number, height: number): Float32Array {
    const flipped = new Float32Array(rgb.length);
    for (let y = 0; y < height; y++) flipped.set(rgb.subarray(y * width * 3, (y + 1) * width * 3), (height - 1 - y) * width * 3);
    return flipped;
}

async function main(): Promise<void> {
    const { values } = parseArgs({
        options: {
            mode: { type: 'string', default: 'match' },
            scene: { type: 'string', default: 'sponza' },
            gpu: { type: 'string', default: 'nvidia' },
            radii: { type: 'string', default: '0.1,0.25,0.5' },
            samples: { type: 'string', default: '16' },
            'resolution-scale': { type: 'string', default: '1' },
            // The T1.14 cost study's variants (the page's job): the occlusion kind, the upscaler.
            'ao-kind': { type: 'string', default: 'gtao' },
            upscale: { type: 'string' },
            suffix: { type: 'string', default: '' },
            reference: { type: 'string' },
            table: { type: 'string' },
            'timeout-minutes': { type: 'string', default: '30' },
            // Match: DDGI passes at the normal hysteresis after the warm-up (the page's default when absent).
            passes: { type: 'string' },
            // Timing: 'gtao' (three's pipeline) or 'rays' (the ray-traced term), each its own run from a cool GPU.
            'timing-part': { type: 'string', default: 'gtao' },
        },
    });
    if (values.gpu !== 'nvidia' && values.gpu !== 'intel') throw new Error('--gpu must be nvidia or intel');
    const gpu: GpuChoice = values.gpu;
    const sceneId = values.scene as CompositeAoScene;
    const setup = SCENES[sceneId];
    if (!setup) throw new Error(`--scene must be one of ${Object.keys(SCENES).join(', ')}`);
    const suffix = values.suffix ?? '';
    const timeoutMs = Number(values['timeout-minutes']) * 60_000;
    const outDir = join(repoRoot, 'bench', 'references', setup.directory);

    if (values.mode === 'timing') {
        const timingPart = values['timing-part'] === 'rays' ? 'rays' : 'gtao';
        // Shading cost doesn't depend on how converged the field is: a short warm-up heats the GPU less.
        const timingJob: Partial<CompositeAoJob> = { mode: 'timing', timingPart, scene: sceneId, samples: Number(values.samples), warmupPasses: 2, passes: 0 };
        const data = await withBenchLock(() => runPage(timingJob, gpu, timeoutMs), { what: `three-composite-ao timing (${timingPart}) on ${gpu}` });
        const gtao = data.gtao as { fullMs: number; halfMs: number } | undefined;
        const summary = { gtao: data.gtao, rayTerm: data.rayTerm, job: data.job, finishedAt: new Date().toISOString() };
        console.log(JSON.stringify({ gtao: gtao ? { fullMs: gtao.fullMs, halfMs: gtao.halfMs } : undefined, rayTerm: data.rayTerm }, null, 2));
        await writeFile(join(outDir, `three-composite-ao-timing-${timingPart}-${gpu}${suffix}.json`), JSON.stringify(summary, null, 2));
        return;
    }

    const radii = (values.radii ?? '').split(',').filter(Boolean).map(Number);
    const job: Partial<CompositeAoJob> = { mode: 'match', scene: sceneId, width: setup.width, height: setup.height, radii, samples: Number(values.samples), resolutionScale: Number(values['resolution-scale']),
        aoKind: values['ao-kind'] as CompositeAoJob['aoKind'],
        upscale: values.upscale === 'quality' || values.upscale === 'performance' ? values.upscale : null,
        ...(values.passes ? { passes: Number(values.passes) } : {}) };
    const data = await withBenchLock(() => runPage(job, gpu, timeoutMs), { what: `three-composite-ao match ${sceneId} on ${gpu}` });
    const { width, height } = setup;
    const image = (base64: unknown) => decodeFloats(String(base64));
    const giTerm = image(data.giTermBase64);
    const weight = image(data.weightBase64);
    const aoImages = Object.fromEntries(Object.entries(data.aoBase64 as Record<string, string>).map(([radius, base64]) => [radius, image(base64)]));
    await writeFile(join(outDir, `three-ao-gi${suffix}.exr`), writeExr({ width, height, rgb: giTerm }));
    await writeFile(join(outDir, `three-ao-weight${suffix}.exr`), writeExr({ width, height, rgb: weight }));
    for (const [radius, ao] of Object.entries(aoImages)) await writeFile(join(outDir, `three-ao-gtao-${radius}${suffix}.exr`), writeExr({ width, height, rgb: ao }));

    let reference = readExr(await readFile(join(outDir, values.reference ?? setup.reference))).rgb;
    // Pixel classes from the weight (the GI albedo with Lambert materials).
    const surface = new Uint8Array(width * height);
    let curtain: Uint8Array | null = null;
    const tableName = values.table ?? setup.table;
    const table = tableName && existsSync(join(outDir, tableName)) ? (JSON.parse(await readFile(join(outDir, tableName), 'utf8')) as { materials: { name: string; albedo: number[] }[] }) : null;
    if (table) curtain = new Uint8Array(width * height);
    for (let pixel = 0; pixel < width * height; pixel++) {
        const albedo = [weight[pixel * 3], weight[pixel * 3 + 1], weight[pixel * 3 + 2]];
        if (albedo.every((value) => value <= 0)) continue;
        surface[pixel] = 1;
        if (curtain && table) {
            const row = table.materials.find((material) => material.albedo.every((value, channel) => Math.abs(value - albedo[channel]) <= ALBEDO_MATCH_TOLERANCE * Math.max(value, 1e-2)));
            if (row?.name.startsWith('curtain')) curtain[pixel] = 1;
        }
    }
    // The readback's row order: take the one that agrees with the reference (printed).
    const asIs = compare(giTerm, reference, curtain, surface, width, height);
    const flipped = compare(giTerm, flipRows(reference, width, height), curtain, surface, width, height);
    if (flipped.blockError < asIs.blockError) {
        reference = flipRows(reference, width, height);
        console.log('row order: the readback is bottom-up; the reference was flipped to match');
    }
    const percent = (value: number | null) => (value === null ? '-' : `${value >= 0 ? '+' : ''}${(value * 100).toFixed(2)}%`);
    console.log(`\n${sceneId} on ${gpu}, ${width} x ${height}, ${values['ao-kind']} ${values.samples} samples at resolution scale ${values['resolution-scale']}, against ${values.reference ?? setup.reference}`);
    console.log('variant          bias      curtains  other     8x8 block error  per-pixel rel. RMS');
    const rows: Record<string, Stats> = {};
    const report = (name: string, test: Float32Array) => {
        const stats = compare(test, reference, curtain, surface, width, height);
        rows[name] = stats;
        console.log(`${name.padEnd(16)} ${percent(stats.bias).padEnd(9)} ${percent(stats.curtains).padEnd(9)} ${percent(stats.other).padEnd(9)} ${(stats.blockError * 100).toFixed(2).padStart(6)}%          ${(stats.pixelRelativeRms * 100).toFixed(1)}%`);
    };
    report('no AO', giTerm);
    for (const [radius, ao] of Object.entries(aoImages)) {
        const occluded = new Float32Array(giTerm.length);
        // GTAO's r channel (replicated into g and b by the page) multiplies the GI term only.
        for (let index = 0; index < giTerm.length; index++) occluded[index] = giTerm[index] * ao[index - (index % 3)];
        report(`${values['ao-kind'] === 'gtao' ? 'GTAO' : values['ao-kind']} ${radius} m`, occluded);
    }
    // Through the upscaler: the GI term at the output size, with the occlusion already inside.
    for (const [name, base64] of Object.entries((data.upscaledBase64 ?? {}) as Record<string, string>)) report(name, image(base64));
    await writeFile(join(outDir, `three-ao-${sceneId}-${gpu}${suffix}.json`), JSON.stringify({ job, rows, finishedAt: new Date().toISOString() }, null, 2));
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
