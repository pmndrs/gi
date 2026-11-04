/**
 * Path-traced reference with three-gpu-pathtracer, in headless Chrome.
 *
 *   node tools/reference/pathtracer/run.ts --scene cornell-box [--camera 1] [--samples 4096]
 *        [--width 512] [--height 512] [--bounces 33]  [--gpu nvidia|intel]
 *        [--denoise] [--ignore-light-range] [--name-suffix=-run2] [--out-dir dir]
 *
 * Output: bench/references/<scene>/<camera name>/pathtracer-{full,indirect}.exr
 * and pathtracer.json (settings, adapter, times). Indirect-only is
 * full minus the one-bounce image: see page/main.ts.
 *
 * Reuses the GPU test runner's Chrome launcher (tools/gpu) but serves its own
 * page. The whole run holds the shared GPU bench lock (see bench-lock.ts):
 * a long trace on the GPU would otherwise distort other agents' timings.
 */

import { existsSync } from 'node:fs';
import { appendFile, mkdir, readFile, stat, writeFile } from 'node:fs/promises';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';
import { createServer } from 'vite';
import { writeExr } from '../../compare/exr.ts';
import { installShutdownHandlers, registerCleanup, runCleanups } from '../../gpu/cleanup.ts';
import type { GpuChoice } from '../../gpu/contract.ts';
import { BrowserSession } from '../../gpu/session.ts';
import { withBenchLock } from '../bench-lock.ts';

const here = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(here, '..', '..', '..');
const pageRoot = join(here, 'page');
const renderLog = join(repoRoot, 'bench', 'references', 'render-log.csv');
/**
 * Where `@pmndrs/denoiser` is built from source: it is not published to npm.
 * `node_modules` is gitignored. See tools/reference/README.md for the steps.
 */
const denoiserClone = join(repoRoot, 'node_modules', '.cache', 'pmndrs-denoiser');
const denoiserBundle = join(denoiserClone, 'packages', 'denoiser', 'dist', 'index.mjs');
const denoiserWeights = join(denoiserClone, 'packages', 'denoiser', 'tzas');

function decodeFloats(base64: string): Float32Array {
    const bytes = Buffer.from(base64, 'base64');
    return new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4).slice();
}

async function main(): Promise<void> {
    const { values } = parseArgs({
        options: {
            scene: { type: 'string' },
            camera: { type: 'string', default: '1' },
            samples: { type: 'string', default: '4096' },
            width: { type: 'string', default: '512' },
            height: { type: 'string', default: '512' },
            bounces: { type: 'string', default: '33' },
            'direct-bounces-lights': { type: 'string', default: '1' },
            'direct-bounces-emission': { type: 'string', default: '2' },
            gpu: { type: 'string', default: 'nvidia' },
            'frame-budget': { type: 'string', default: '4000000' },
            'timeout-minutes': { type: 'string', default: '60' },
            'name-suffix': { type: 'string', default: '' },
            'out-dir': { type: 'string' },
            denoise: { type: 'boolean' },
            'ignore-light-range': { type: 'boolean' },
            help: { type: 'boolean' },
        },
    });
    if (values.help || !values.scene) {
        console.log('usage: node tools/reference/pathtracer/run.ts --scene <id> [--camera 1] [--samples N] [--width W] [--height H] [--bounces N] [--gpu nvidia|intel]');
        process.exit(values.help ? 0 : 1);
    }
    if (values.gpu !== 'nvidia' && values.gpu !== 'intel') throw new Error('--gpu must be nvidia or intel');
    const gpu: GpuChoice = values.gpu;

    const manifestText = await readFile(join(repoRoot, 'scenes', 'manifests', `${values.scene}.json`), 'utf8');
    const manifest = JSON.parse(manifestText) as { id: string; cameras: Array<{ name: string }> };
    const cameraIndex = /^\d+$/.test(values.camera) ? Number(values.camera) - 1 : manifest.cameras.findIndex((camera) => camera.name === values.camera);
    const cameraName = manifest.cameras[cameraIndex]?.name;
    if (!cameraName) throw new Error(`unknown camera ${values.camera}`);

    const job = {
        scene: manifest.id,
        camera: values.camera,
        samples: Number(values.samples),
        width: Number(values.width),
        height: Number(values.height),
        bounces: Number(values.bounces),
        directBouncesLights: Number(values['direct-bounces-lights']),
        directBouncesEmission: Number(values['direct-bounces-emission']),
        frameBudget: Number(values['frame-budget']),
        ignoreLightRange: values['ignore-light-range'] ?? false,
        denoiser: values.denoise
            ? { moduleUrl: `/@fs/${denoiserBundle.replaceAll('\\', '/')}`, weightsUrl: '/denoiser-weights' }
            : null,
    };
    if (values.denoise && !existsSync(denoiserBundle)) {
        throw new Error(`--denoise needs @pmndrs/denoiser built at ${denoiserBundle}; see tools/reference/README.md`);
    }

    const server = await createServer({
        root: pageRoot,
        configFile: false,
        logLevel: 'warn',
        appType: 'mpa',
        optimizeDeps: { noDiscovery: true, include: [] },
        server: {
            host: '127.0.0.1',
            port: 5291,
            strictPort: false,
            hmr: false,
            watch: null,
            // The page imports scenes/src and tools/gpu/client.ts from outside its own folder.
            fs: { allow: [repoRoot] },
        },
        plugins: [
            {
                name: 'reference-job',
                configureServer(devServer) {
                    devServer.middlewares.use('/job.json', (_request, response) => {
                        response.setHeader('content-type', 'application/json');
                        response.end(JSON.stringify(job));
                    });
                    devServer.middlewares.use('/manifest', (request, response) => {
                        readFile(join(repoRoot, 'scenes', 'manifests', (request.url ?? '').replace(/^\//, '').replace(/[^a-z0-9.-]/gi, '')))
                            .then((text) => {
                                response.setHeader('content-type', 'application/json');
                                response.end(text);
                            })
                            .catch(() => {
                                response.statusCode = 404;
                                response.end();
                            });
                    });
                    // The OIDN network weights (.tza files) from the denoiser clone, so nothing is fetched from a CDN.
                    devServer.middlewares.use('/denoiser-weights', (request, response) => {
                        readFile(join(denoiserWeights, (request.url ?? '').replace(/^\//, '').replace(/[^a-z0-9._-]/gi, '')))
                            .then((bytes) => {
                                response.setHeader('content-type', 'application/octet-stream');
                                response.end(bytes);
                            })
                            .catch(() => {
                                response.statusCode = 404;
                                response.end();
                            });
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

    const outDir = resolve(values['out-dir'] ?? join(repoRoot, 'bench', 'references', manifest.id, cameraName));
    const suffix = values['name-suffix'] ?? '';

    // Only the GPU work holds the bench lock. Closing Chrome can take minutes on a busy
    // machine, and the lock should not stay held while it does.
    let session: BrowserSession | null = null;
    let result;
    try {
        result = await withBenchLock(async () => {
            session = await BrowserSession.launch({ serverOrigin: origin, chromePath: null, gpu, withoutGpu: false });
            console.log(`rendering ${job.scene} / ${cameraName}: ${job.samples} samples, ${job.width}x${job.height}, ${job.bounces} bounces, on ${gpu}`);
            const timeoutMs = Number(values['timeout-minutes']) * 60_000;
            return session.runPage(
                { name: 'pathtracer', urlPath: '/index.html', description: 'path-traced reference', expectFailure: false, expectMessage: null, manual: false, timeoutMs },
                { timeoutMs, expectFailure: false, screenshotDir: join(outDir, 'logs'), gpuLabel: gpu },
            );
        });
    } finally {
        await (session as BrowserSession | null)?.close();
    }

    if (!result.verdict.acceptable || !result.report) {
        throw new Error(`the page failed:\n  ${result.verdict.reasons.join('\n  ')}`);
    }
    const data = result.report.data as Record<string, unknown>;
    const full = decodeFloats(String(data.fullBase64));
    const direct = decodeFloats(String(data.directBase64));
    const denoised = data.denoisedBase64 ? decodeFloats(String(data.denoisedBase64)) : null;
    const indirect = full.map((value, index) => value - direct[index]);

    await mkdir(outDir, { recursive: true });
    const image = (rgb: Float32Array) => ({ width: job.width, height: job.height, rgb });
    await writeFile(join(outDir, `pathtracer-full${suffix}.exr`), writeExr(image(full)));
    await writeFile(join(outDir, `pathtracer-indirect${suffix}.exr`), writeExr(image(indirect)));

    if (denoised) await writeFile(join(outDir, `pathtracer-full-denoised${suffix}.exr`), writeExr(image(denoised)));

    const meta = {
        tool: 'tools/reference/pathtracer/run.ts',
        library: 'three-gpu-pathtracer 0.0.28 (WebGPU), three 0.187.0',
        scene: manifest.id,
        camera: cameraName,
        width: job.width,
        height: job.height,
        samplesPerPixel: job.samples,
        maxBounces: job.bounces,
        directParts: data.directParts,
        clampDirect: 0,
        clampIndirect: 0,
        ignoreLightRange: job.ignoreLightRange,
        denoising: denoised ? `@pmndrs/denoiser (OIDN ${String(data.denoiserModel)}, WGSL, color only), ${Number(data.denoiseSeconds).toFixed(1)} s; pathtracer-full.exr itself is not denoised` : false,
        materials: 'diffuse only (specularIntensity 0)',
        indirectDefinition: `full (${job.bounces} bounces) minus the direct image (${(data.directParts as string[]).join(' + ')}; lights and emission rendered separately and added)`,
        gpuRequested: gpu,
        adapter: result.report.adapter,
        fullSeconds: Number(Number(data.fullSeconds).toFixed(2)),
        directSeconds: Number(Number(data.directSeconds).toFixed(2)),
        // Local time, like the Cycles rows (shifted by the UTC offset before ISO formatting).
        finishedAt: new Date(Date.now() - new Date().getTimezoneOffset() * 60_000).toISOString().slice(0, 19),
    };
    await writeFile(join(outDir, `pathtracer${suffix}.json`), JSON.stringify(meta, null, 2));

    await mkdir(dirname(renderLog), { recursive: true });
    if (!(await stat(renderLog).then(() => true, () => false))) {
        await writeFile(renderLog, 'finishedAt,tool,scene,camera,width,height,samples,seed,device,threads,loadSeconds,renderSeconds,triangles\n');
    }
    await appendFile(renderLog, `${meta.finishedAt},pathtracer,${meta.scene},${meta.camera},${meta.width},${meta.height},${meta.samplesPerPixel},,${gpu},,,${meta.fullSeconds + meta.directSeconds},\n`);
    console.log(`done: full ${meta.fullSeconds} s + direct ${meta.directSeconds} s, wrote ${outDir}`);
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
