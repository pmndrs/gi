/**
 * Sponza accuracy (T1.9): runs the `ddgi-sponza-accuracy` page, writes its
 * images and the GI material table, and compares DDGI with the untextured
 * Cycles reference. Also prints one accuracy table for the three M1 scenes.
 *
 *   node tools/gpu/pages/ddgi-sponza-accuracy/run.ts [--gpu nvidia|intel] [--camera 1]
 *        [--width W] [--height H] [--suffix -run2] [--reference-suffix -base] [--job '{"blockPasses": 32}']
 *        [--timeout-minutes 40]
 *   node tools/gpu/pages/ddgi-sponza-accuracy/run.ts --compare-only [--suffix ...] [--reference-suffix ...]
 *   node tools/gpu/pages/ddgi-sponza-accuracy/run.ts --table [--cornell-suffix -t19] [--two-room-suffix -t19] [--suffix ...]
 *
 * Writes to `bench/references/sponza/<camera>/` (gitignored):
 * - `gi-materials<suffix>.json`: per material (model part and glTF name), the GI
 *   albedo and emission the three adapter sent to the platform. This is what
 *   `render.py --materials-json` renders Cycles with:
 *     node tools/reference/cycles/run.ts --scene sponza --camera 1 \
 *          --samples 8192 --device gpu --materials-json bench/references/sponza/colonnade-from-the-floor/gi-materials.json
 *   which writes `cycles-untextured-{full,indirect,gi}.exr` next to it. The
 *   table also lists the model parts loaded (render.py loads only those) and,
 *   per material, the meshes the adapter left out of GI as see-through
 *   (render.py removes those faces). A reference rendered with
 *   `--name-suffix S` (e.g. the base package, `--job '{"parts":["sponza-intel"]}'`)
 *   is compared with `--reference-suffix S`.
 * - `ddgi-untextured-{indirect,full}<suffix>.exr` and `ddgi-untextured<suffix>.json`;
 * - with a `pathTrace` job, `pt-untextured-indirect<suffix>.exr`: the same
 *   quantity from the page's path tracer over the platform's own scene;
 * - comparison PNGs (`compare-cycles-untextured-vs-ddgi-*`).
 *
 * What is compared: DDGI's test view "indirect" is albedo / π × the field,
 * and the field holds everything but the analytic lights: bounce light, and
 * also sky light (and emission) reaching the surface straight. Cycles'
 * Diffuse Indirect pass counts that sky light as direct. So every scene is
 * judged against the GI composite, `cycles-gi.exr` or
 * `cycles-untextured-gi.exr` (render.py `render_with_gi_parts`: indirect +
 * direct − the lights' direct; DECISIONS "What 'indirect light' means against
 * Cycles in a scene with a sky"). Cornell (point) and two-room have no sky and
 * no emitter, so there the composite equals the indirect pass (run.ts checks
 * that the two direct images match); Sponza's open courtyard makes them differ
 * by about 2×, and the Cycles indirect pass is reported alongside.
 *
 * The GPU run holds the bench lock (it is minutes of full GPU load), unless
 * started under `bench-lock.ts run`.
 *
 * `--table` compares the three scenes' existing images (no GPU), each against
 * its GI composite: Cornell (point light, `front`) and two-room (`lit-room`)
 * from `ddgi-test-view/render.ts` against `cycles-gi.exr`, and Sponza from
 * this script against `cycles-untextured-gi.exr`. Bias is per pixel (the
 * sums); block error, p95 block error and FLIP use 8 × 8 blocks (NOTES:
 * comparing noisy renders), and FLIP is also given per pixel.
 */

import { spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { existsSync } from 'node:fs';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { dirname, join, relative, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';
import { createServer } from 'vite';
import { readExr, writeExr } from '../../../compare/exr.ts';
import { withBenchLock } from '../../../reference/bench-lock.ts';
import { DEFAULT_REFERENCE_SIZE, readReferenceShot, referenceSize, shotView } from '../../../reference/shot.ts';
import { installShutdownHandlers, registerCleanup, runCleanups } from '../../cleanup.ts';
import type { GpuChoice } from '../../contract.ts';
import { SITE_ROOT } from '../../server.ts';
import { BrowserSession } from '../../session.ts';
import type { SponzaAccuracyJob } from './main.ts';

const here = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(here, '..', '..', '..', '..');
const referencesRoot = join(repoRoot, 'bench', 'references');

/** SPEC's M1 accuracy target: indirect light's mean within ±10%. */
const MEAN_TOLERANCE = 0.1;

function decodeFloats(base64: string): Float32Array {
    const bytes = Buffer.from(base64, 'base64');
    return new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4).slice();
}

/** parseArgs refuses an option value that starts with a dash (`--suffix -run2`), so such values are joined to their flag first. */
function joinDashValues(args: readonly string[], flags: readonly string[]): string[] {
    const joined: string[] = [];
    for (let index = 0; index < args.length; index++) {
        const next = args[index + 1];
        if (flags.includes(args[index]) && next !== undefined && next.startsWith('-')) {
            joined.push(`${args[index]}=${next}`);
            index++;
        } else {
            joined.push(args[index]);
        }
    }
    return joined;
}

interface Manifest {
    readonly id: string;
    readonly cameras: readonly { readonly name: string }[];
}

async function cameraName(scene: string, camera: number): Promise<string> {
    const manifest = JSON.parse(await readFile(join(repoRoot, 'scenes', 'manifests', `${scene}.json`), 'utf8')) as Manifest;
    const name = manifest.cameras[camera - 1]?.name;
    if (!name) throw new Error(`${scene} has no camera ${camera}`);
    return name;
}

/** Runs the page once in its own vite server and Chrome, with the job injected; returns the page's report data. */
async function runPage(job: Partial<SponzaAccuracyJob>, gpu: GpuChoice, logDir: string, timeoutMs: number): Promise<{ data: Record<string, unknown>; adapter: unknown }> {
    const pagePath = '/pages/ddgi-sponza-accuracy/job.html';
    const server = await createServer({
        root: SITE_ROOT,
        configFile: false,
        logLevel: 'warn',
        appType: 'mpa',
        optimizeDeps: { noDiscovery: true, include: [] },
        server: { host: '127.0.0.1', port: 5293, strictPort: false, hmr: false, watch: null, fs: { allow: [repoRoot] } },
        plugins: [
            {
                name: 'ddgi-sponza-accuracy-job',
                configureServer(devServer) {
                    devServer.middlewares.use(pagePath, (_request, response) => {
                        response.setHeader('content-type', 'text/html');
                        response.end(
                            `<!doctype html><html><head><meta charset="utf-8" /><title>ddgi-sponza-accuracy job</title></head><body>` +
                                `<script>window.ddgiSponzaAccuracyJob = ${JSON.stringify(job)};</script>` +
                                `<script type="module" src="/pages/ddgi-sponza-accuracy/main.ts"></script></body></html>`,
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
    try {
        session = await BrowserSession.launch({ serverOrigin: origin, chromePath: null, gpu, withoutGpu: false });
        const result = await session.runPage(
            { name: 'ddgi-sponza-accuracy-job', urlPath: pagePath, description: 'DDGI Sponza accuracy', expectFailure: false, expectMessage: null, manual: false, timeoutMs },
            { timeoutMs, expectFailure: false, screenshotDir: logDir, gpuLabel: gpu },
        );
        if (!result.verdict.acceptable || !result.report) throw new Error(`the page failed:\n  ${result.verdict.reasons.join('\n  ')}`);
        return { data: result.report.data as Record<string, unknown>, adapter: result.report.adapter };
    } finally {
        await (session as BrowserSession | null)?.close();
        await server.close();
    }
}

/** `tools/compare --json` on two EXRs. */
interface CompareNumbers {
    readonly metrics: { readonly meanReference: number; readonly meanTest: number; readonly meanRelativeError: number; readonly bias: number; readonly relativeRms: number };
    readonly p95Error: number;
    readonly flip: { readonly mean: number } | null;
}

function compareJson(reference: string, test: string, extra: readonly string[]): CompareNumbers {
    const run = spawnSync(process.execPath, [join(repoRoot, 'tools', 'compare', 'index.ts'), reference, test, '--json', ...extra], { encoding: 'utf8', cwd: repoRoot });
    if (run.status !== 0) throw new Error(`compare failed on ${test}:\n${run.stderr}`);
    return JSON.parse(run.stdout) as CompareNumbers;
}

function comparePrinted(reference: string, test: string, label: string, extra: readonly string[]): void {
    console.log(`\n== ${label} ==`);
    const run = spawnSync(process.execPath, [join(repoRoot, 'tools', 'compare', 'index.ts'), reference, test, ...extra], { encoding: 'utf8', cwd: repoRoot });
    process.stdout.write(run.stdout);
    process.stderr.write(run.stderr);
}

function sha256(text: string | Buffer): string {
    return createHash('sha256').update(text).digest('hex');
}

/**
 * Says whether the Cycles reference was rendered from this exact material
 * table (render.py records the table's sha256). A reference rendered from an
 * older table compares different materials.
 */
async function checkReferenceTable(outDir: string, tablePath: string, referenceSuffix: string): Promise<void> {
    const metaPath = join(outDir, `cycles-untextured${referenceSuffix}.json`);
    if (!existsSync(metaPath) || !existsSync(tablePath)) return;
    const meta = JSON.parse(await readFile(metaPath, 'utf8')) as { giMaterials?: { sha256?: string } };
    const tableHash = sha256(await readFile(tablePath));
    if (meta.giMaterials?.sha256 === tableHash) console.log(`the Cycles reference was rendered from this material table (${relative(repoRoot, tablePath)})`);
    else console.warn(`warning: the Cycles reference was rendered from a different material table than ${relative(repoRoot, tablePath)}; render it again`);
}

async function compareSponza(outDir: string, suffix: string, referenceSuffix: string): Promise<void> {
    const referenceGi = join(outDir, `cycles-untextured-gi${referenceSuffix}.exr`);
    const referenceIndirect = join(outDir, `cycles-untextured-indirect${referenceSuffix}.exr`);
    const referenceFull = join(outDir, `cycles-untextured-full${referenceSuffix}.exr`);
    const indirect = join(outDir, `ddgi-untextured-indirect${suffix}.exr`);
    const full = join(outDir, `ddgi-untextured-full${suffix}.exr`);
    if (!existsSync(referenceGi)) {
        console.log(`
no ${relative(repoRoot, referenceGi)} yet; render it with tools/reference/cycles/run.ts --materials-json (see this file's header)`);
        return;
    }
    await checkReferenceTable(outDir, join(outDir, `gi-materials${suffix}.json`), referenceSuffix);
    const prefix = (name: string): string => join(outDir, `compare-cycles-untextured-vs-ddgi-${name}${suffix}`);
    comparePrinted(referenceGi, indirect, 'GI (DDGI indirect against Cycles indirect + sky and emission direct), per pixel', ['--out', prefix('gi')]);
    comparePrinted(referenceGi, indirect, 'GI, 8x8 blocks', ['--downsample', '8', '--out', prefix('gi-blocks8')]);
    // Sky pixels are 0 in the test view and the sky in Cycles' full image: the full numbers are for looking at, not judging.
    comparePrinted(referenceFull, full, 'full, 8x8 blocks (not gating; sky pixels are black in the test view)', ['--downsample', '8', '--out', prefix('full-blocks8')]);
    const gi = compareJson(referenceGi, indirect, ['--no-images', '--no-flip']);
    const pathTraced = join(outDir, `pt-untextured-indirect${suffix}.exr`);
    const cyclesIndirect = existsSync(referenceIndirect) ? compareJson(referenceIndirect, indirect, ['--no-images', '--no-flip']) : null;
    const signed = (value: number): string => `${value >= 0 ? '+' : ''}${(value * 100).toFixed(2)}%`;
    console.log(`
Sponza: DDGI indirect mean ${gi.metrics.meanTest.toFixed(6)} against untextured Cycles GI ${gi.metrics.meanReference.toFixed(6)}: ${signed(gi.metrics.bias)} (target ±${MEAN_TOLERANCE * 100}%)`);
    if (cyclesIndirect) {
        console.log(`(against Cycles' own indirect pass, ${cyclesIndirect.metrics.meanReference.toFixed(6)}, which leaves out sky light reaching a surface straight: ${signed(cyclesIndirect.metrics.bias)})`);
    }
    const albedoFile = join(outDir, `ddgi-untextured-albedo${suffix}.exr`);
    const tableFile = join(outDir, `gi-materials${suffix}.json`);
    if (existsSync(albedoFile) && existsSync(tableFile)) {
        await printMaterialBreakdown({ albedoFile, tableFile, ddgi: indirect, cycles: referenceGi, pathTraced: existsSync(pathTraced) ? pathTraced : null });
    }
    if (existsSync(pathTraced)) {
        // The path tracer shares DDGI's scene exactly, so it splits the gap in two:
        // how far our scene and shading are from Cycles, and how far DDGI is from them.
        const tracedAgainstCycles = compareJson(referenceGi, pathTraced, ['--no-images', '--no-flip']);
        const ddgiAgainstTraced = compareJson(pathTraced, indirect, ['--no-images', '--no-flip']);
        const ddgiAgainstTracedBlocks = compareJson(pathTraced, indirect, ['--no-images', '--no-flip', '--downsample', '8']);
        console.log(`path traced (our scene and shading) against Cycles GI: ${signed(tracedAgainstCycles.metrics.bias)}`);
        console.log(`DDGI against path traced: ${signed(ddgiAgainstTraced.metrics.bias)}, 8x8 block error ${(ddgiAgainstTracedBlocks.metrics.meanRelativeError * 100).toFixed(2)}%`);
    }
}

/** A GI albedo matches a table row when every channel agrees within this share (the shader rounds to f32). */
const ALBEDO_MATCH_TOLERANCE = 1e-4;

/**
 * DDGI against Cycles (and the path tracer), split by the material each
 * pixel shows. The material comes from the test view's albedo image, matched
 * to the GI material table (Sponza's albedos are all different, except the
 * curtain rods, which share `metal_door`'s). Pixels whose albedo matches no
 * row (the sky) are left out.
 */
async function printMaterialBreakdown(files: { albedoFile: string; tableFile: string; ddgi: string; cycles: string; pathTraced: string | null }): Promise<void> {
    const table = JSON.parse(await readFile(files.tableFile, 'utf8')) as { materials: { name: string; albedo: [number, number, number] }[] };
    const albedo = readExr(await readFile(files.albedoFile)).rgb;
    const ddgi = readExr(await readFile(files.ddgi)).rgb;
    const cycles = readExr(await readFile(files.cycles)).rgb;
    const pathTraced = files.pathTraced ? readExr(await readFile(files.pathTraced)).rgb : null;
    const sums = new Map<string, { pixels: number; ddgi: number; cycles: number; pathTraced: number }>();
    const pixels = albedo.length / 3;
    for (let pixel = 0; pixel < pixels; pixel++) {
        const pixelAlbedo = [albedo[pixel * 3], albedo[pixel * 3 + 1], albedo[pixel * 3 + 2]];
        if (pixelAlbedo.every((value) => value === 0)) continue;
        const row = table.materials.find((material) => material.albedo.every((value, channel) => Math.abs(value - pixelAlbedo[channel]) <= ALBEDO_MATCH_TOLERANCE * Math.max(value, 1e-3)));
        const name = row?.name.startsWith('curtain') ? 'curtains (all three)' : 'everything else';
        for (const key of [name, row ? `  ${row.name}` : '  (no match)']) {
            const entry = sums.get(key) ?? { pixels: 0, ddgi: 0, cycles: 0, pathTraced: 0 };
            entry.pixels++;
            for (let channel = 0; channel < 3; channel++) {
                const index = pixel * 3 + channel;
                entry.ddgi += ddgi[index];
                entry.cycles += cycles[index];
                entry.pathTraced += pathTraced ? pathTraced[index] : 0;
            }
            sums.set(key, entry);
        }
    }
    const signed = (value: number): string => `${value >= 0 ? '+' : ''}${(value * 100).toFixed(1)}%`;
    console.log('\nby material (pixel share; DDGI against Cycles GI; path traced against Cycles; DDGI against path traced; Cycles share of the image sum):');
    const cyclesTotal = [...sums.entries()].filter(([key]) => !key.startsWith(' ')).reduce((total, [, entry]) => total + entry.cycles, 0);
    const rows = [...sums.entries()].sort((a, b) => b[1].cycles - a[1].cycles);
    for (const [key, entry] of [...rows.filter(([key]) => !key.startsWith(' ')), ...rows.filter(([key]) => key.startsWith(' ')).slice(0, 12)]) {
        const traced = pathTraced ? `${signed(entry.pathTraced / entry.cycles - 1)}, ${signed(entry.ddgi / entry.pathTraced - 1)}` : 'no path-traced image';
        console.log(`${key.padEnd(28)} ${((entry.pixels / pixels) * 100).toFixed(1).padStart(5)}%  ${signed(entry.ddgi / entry.cycles - 1).padStart(7)}  ${traced}  ${((entry.cycles / cyclesTotal) * 100).toFixed(1)}%`);
    }
}

interface TableRow {
    readonly scene: string;
    readonly camera: string;
    readonly reference: string;
    readonly test: string;
}

/**
 * One accuracy table for the three M1 scenes, from images already on disk.
 * Every row uses the same reference definition, the GI composite (see the
 * file header); a missing `cycles-gi.exr` is rendered with
 * tools/reference/cycles/run.ts at the reference's size and samples.
 */
async function printTable(suffixes: { readonly cornell: string; readonly twoRoom: string; readonly sponza: string }): Promise<void> {
    const rows: TableRow[] = [
        { scene: 'cornell-box-point', camera: 'front', reference: 'cycles-gi.exr', test: `ddgi-indirect${suffixes.cornell}.exr` },
        { scene: 'two-room', camera: 'lit-room', reference: 'cycles-gi.exr', test: `ddgi-indirect${suffixes.twoRoom}.exr` },
        { scene: 'sponza', camera: 'colonnade-from-the-floor', reference: 'cycles-untextured-gi.exr', test: `ddgi-untextured-indirect${suffixes.sponza}.exr` },
    ];
    const lines = [
        '| scene / camera | Cycles reference (GI composite) | DDGI image | GI bias | 8x8 block mean rel. error | p95 block error | FLIP 8x8 (8.4 ppd) | FLIP per pixel (67 ppd) |',
        '| --- | --- | --- | --- | --- | --- | --- | --- |',
    ];
    const percent = (value: number): string => `${(value * 100).toFixed(2)}%`;
    for (const row of rows) {
        const directory = join(referencesRoot, row.scene, row.camera);
        const reference = join(directory, row.reference);
        const test = join(directory, row.test);
        if (!existsSync(reference) || !existsSync(test)) {
            lines.push(`| ${row.scene} / ${row.camera} | ${row.reference} | ${row.test} | missing | | | | |`);
            continue;
        }
        const pixels = compareJson(reference, test, ['--no-images']);
        const blocks = compareJson(reference, test, ['--no-images', '--downsample', '8']);
        const flip = (numbers: CompareNumbers): string => (numbers.flip ? numbers.flip.mean.toFixed(4) : 'n/a');
        lines.push(
            `| ${row.scene} / ${row.camera} | ${row.reference} | ${row.test} | ${pixels.metrics.bias >= 0 ? '+' : ''}${percent(pixels.metrics.bias)} | ${percent(blocks.metrics.meanRelativeError)} | ${percent(blocks.p95Error)} | ${flip(blocks)} | ${flip(pixels)} |`,
        );
    }
    console.log(lines.join('\n'));
}

async function main(): Promise<void> {
    const { values } = parseArgs({
        args: joinDashValues(process.argv.slice(2), ['--suffix', '--reference-suffix', '--cornell-suffix', '--two-room-suffix']),
        options: {
            camera: { type: 'string', default: '1' },
            // Default: the untextured reference's stored size (T1.13).
            width: { type: 'string' },
            height: { type: 'string' },
            gpu: { type: 'string', default: 'nvidia' },
            suffix: { type: 'string', default: '' },
            'reference-suffix': { type: 'string', default: '' },
            job: { type: 'string', default: '{}' },
            'compare-only': { type: 'boolean' },
            // How long the page may run before it counts as failed (a long job, or a hung shader compile).
            'timeout-minutes': { type: 'string', default: '40' },
            table: { type: 'boolean' },
            'cornell-suffix': { type: 'string', default: '' },
            'two-room-suffix': { type: 'string', default: '' },
        },
    });
    const suffix = values.suffix ?? '';
    if (values.table) {
        await printTable({ cornell: values['cornell-suffix'] ?? '', twoRoom: values['two-room-suffix'] ?? '', sponza: suffix });
        return;
    }
    const camera = Number(values.camera);
    const outDir = join(referencesRoot, 'sponza', await cameraName('sponza', camera));
    if (values['compare-only']) {
        await compareSponza(outDir, suffix, values['reference-suffix'] ?? '');
        return;
    }
    if (values.gpu !== 'nvidia' && values.gpu !== 'intel') throw new Error('--gpu must be nvidia or intel');
    const gpu: GpuChoice = values.gpu;

    // The untextured reference's own camera and size (T1.13), so the images line up pixel for pixel.
    const referenceJson = `cycles-untextured${values['reference-suffix'] ?? ''}.json`;
    const shot = await readReferenceShot(repoRoot, outDir, referenceJson);
    // A reference from before T1.13 has no stored camera: the bookmark, at that reference's size.
    const size = shot?.camera ?? (await referenceSize(outDir, referenceJson)) ?? DEFAULT_REFERENCE_SIZE;
    if (!shot) console.warn(`no stored camera in ${referenceJson} (rendered before T1.13, or missing): using the bookmark at ${size.width} x ${size.height}`);
    const width = Number(values.width ?? size.width);
    const height = Number(values.height ?? size.height);
    const job: Partial<SponzaAccuracyJob> = {
        ...(JSON.parse(values.job ?? '{}') as Partial<SponzaAccuracyJob>),
        camera,
        ...(shot ? { view: shotView(shot) } : {}),
        width,
        height,
        exportImages: true,
    };
    console.log(`sponza accuracy: camera ${camera}, ${job.width}x${job.height}, on ${gpu}`);
    const { data, adapter } = await withBenchLock(() => runPage(job, gpu, join(outDir, 'logs'), Number(values['timeout-minutes']) * 60_000), { what: `ddgi-sponza-accuracy on ${gpu}` });

    await mkdir(outDir, { recursive: true });
    const image = (key: string) => ({ width, height, rgb: decodeFloats(String(data[key])) });
    const indirectFile = join(outDir, `ddgi-untextured-indirect${suffix}.exr`);
    const fullFile = join(outDir, `ddgi-untextured-full${suffix}.exr`);
    await writeFile(indirectFile, writeExr(image('indirectBase64')));
    await writeFile(fullFile, writeExr(image('fullBase64')));
    if (data.pathTracedBase64) await writeFile(join(outDir, `pt-untextured-indirect${suffix}.exr`), writeExr(image('pathTracedBase64')));
    if (data.albedoBase64) await writeFile(join(outDir, `ddgi-untextured-albedo${suffix}.exr`), writeExr(image('albedoBase64')));
    if (data.positionBase64) await writeFile(join(outDir, `ddgi-untextured-position${suffix}.exr`), writeExr(image('positionBase64')));
    // Diagnostic lookups: one EXR per image (`-r<rays>` for each short-range ray count).
    const diagnosticImages = (data.diagnosticImages ?? {}) as Record<string, string>;
    const lookup = String((data.job as SponzaAccuracyJob).diagnosticLookup);
    for (const [key, base64] of Object.entries(diagnosticImages)) {
        const name = key === 'image' ? `ddgi-untextured-indirect-${lookup}${suffix}.exr` : `ddgi-untextured-indirect-${lookup}-${key}${suffix}.exr`;
        await writeFile(join(outDir, name), writeExr({ width, height, rgb: decodeFloats(base64) }));
    }
    const tableFile = join(outDir, `gi-materials${suffix}.json`);
    await writeFile(tableFile, `${JSON.stringify(data.giMaterials, null, 2)}\n`);
    const summary = { ...data, indirectBase64: undefined, fullBase64: undefined, pathTracedBase64: undefined, albedoBase64: undefined, positionBase64: undefined, diagnosticImages: undefined, giMaterials: undefined, adapter, finishedAt: new Date().toISOString() };
    await writeFile(join(outDir, `ddgi-untextured${suffix}.json`), JSON.stringify(summary, null, 2));
    console.log(`wrote ${relative(repoRoot, indirectFile)}, ${relative(repoRoot, fullFile)} and ${relative(repoRoot, tableFile)}`);
    console.log(`${String(data.passes)} passes in ${String(data.convergeMs)} ms; block means ${JSON.stringify(data.blockMeans)}`);
    await compareSponza(outDir, suffix, values['reference-suffix'] ?? '');
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
