/**
 * Renders a scene the way the viewer builds it, lit by the shared sky through
 * three's environment lighting, to a linear float EXR next to the references,
 * so `tools/compare` can put it against the Cycles render.
 *
 *   node tools/sky/viewer-render.ts --scene open-ground [--camera 1] [--width 640] [--height 360]
 *        [--material diffuse|standard] [--gpu nvidia|intel]
 *
 * Output: bench/references/<scene>/<camera name>/viewer-<material>.exr and
 * viewer-<material>.json. It is a correctness render (one still frame), so it
 * does not take the GPU bench lock.
 */

import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';
import { createServer } from 'vite';
import { writeExr } from '../compare/exr.ts';
import { installShutdownHandlers, registerCleanup, runCleanups } from '../gpu/cleanup.ts';
import type { GpuChoice } from '../gpu/contract.ts';
import { BrowserSession } from '../gpu/session.ts';

const here = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(here, '..', '..');
const pageRoot = join(here, 'viewer-page');

async function main(): Promise<void> {
    const { values } = parseArgs({
        options: {
            scene: { type: 'string', default: 'open-ground' },
            camera: { type: 'string', default: '1' },
            width: { type: 'string', default: '640' },
            height: { type: 'string', default: '360' },
            material: { type: 'string', default: 'diffuse' },
            gpu: { type: 'string', default: 'nvidia' },
            'out-dir': { type: 'string' },
        },
    });
    if (values.gpu !== 'nvidia' && values.gpu !== 'intel') throw new Error('--gpu must be nvidia or intel');
    if (values.material !== 'diffuse' && values.material !== 'standard') throw new Error('--material must be diffuse or standard');
    const gpu: GpuChoice = values.gpu;

    const manifest = JSON.parse(await readFile(join(repoRoot, 'scenes', 'manifests', `${values.scene}.json`), 'utf8')) as {
        id: string;
        cameras: Array<{ name: string }>;
    };
    const cameraIndex = /^\d+$/.test(values.camera) ? Number(values.camera) - 1 : manifest.cameras.findIndex((camera) => camera.name === values.camera);
    const cameraName = manifest.cameras[cameraIndex]?.name;
    if (!cameraName) throw new Error(`unknown camera ${values.camera}`);
    const job = { scene: manifest.id, camera: values.camera, width: Number(values.width), height: Number(values.height), material: values.material };

    const server = await createServer({
        root: pageRoot,
        configFile: false,
        logLevel: 'warn',
        appType: 'mpa',
        optimizeDeps: { noDiscovery: true, include: [] },
        server: {
            host: '127.0.0.1',
            port: 5293,
            strictPort: false,
            hmr: false,
            watch: null,
            // The page imports the viewer, scenes/src and the sky files from outside its folder.
            fs: { allow: [repoRoot] },
        },
        plugins: [
            {
                name: 'sky-render-job',
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
    const session = await BrowserSession.launch({ serverOrigin: origin, chromePath: null, gpu, withoutGpu: false });
    registerCleanup(() => session.close());
    console.log(`rendering ${job.scene} / ${cameraName} (${job.material} materials, sky as three environment light), ${job.width}x${job.height}, on ${gpu}`);
    const result = await session.runPage(
        { name: 'sky-viewer-render', urlPath: '/index.html', description: 'viewer render lit by the shared sky', expectFailure: false, expectMessage: null, manual: false, timeoutMs: 120_000 },
        { timeoutMs: 120_000, expectFailure: false, screenshotDir: join(outDir, 'logs'), gpuLabel: gpu },
    );
    if (!result.verdict.acceptable || !result.report) throw new Error(`the page failed:\n  ${result.verdict.reasons.join('\n  ')}`);

    const data = result.report.data as { meanLuminance: number; rgbBase64: string };
    const bytes = Buffer.from(data.rgbBase64, 'base64');
    const rgb = new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4).slice();
    await mkdir(outDir, { recursive: true });
    const name = `viewer-${job.material}`;
    await writeFile(join(outDir, `${name}.exr`), writeExr({ width: job.width, height: job.height, rgb }));
    await writeFile(
        join(outDir, `${name}.json`),
        JSON.stringify(
            {
                tool: 'tools/sky/viewer-render.ts',
                library: 'three 0.187.0 (WebGPURenderer), apps/viewer loadScene, sky as scene.environment (PMREM)',
                ...job,
                cameraName,
                meanLuminance: data.meanLuminance,
                adapter: result.report.adapter,
            },
            null,
            2,
        ),
    );
    console.log(`mean luminance ${data.meanLuminance.toPrecision(6)}; wrote ${join(outDir, `${name}.exr`)}`);
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
