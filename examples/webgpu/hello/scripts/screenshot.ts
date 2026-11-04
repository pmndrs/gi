/// <reference types="node" />
/**
 * Screenshots of the real hello example: starts its vite dev server, opens
 * it in headless Chrome on the GPU you ask for, waits until DDGI's field has
 * settled (`window.helloState.ready`: every probe updated at least 100
 * times), and saves a PNG per view. Fails on any browser or WebGPU error.
 *
 *   node examples/webgpu/hello/scripts/screenshot.ts
 *     [--gpu nvidia|intel]                   default nvidia
 *     [--scene cornell-box]                  default cornell-box
 *     [--camera 1]
 *     [--views full,gi,coverage,probes]      default full
 *     [--out <dir>]                          default bench/screenshots/t1.6 (gitignored)
 *
 * It reuses the GPU test runner's Chrome launcher and DevTools client.
 */

import { mkdtempSync } from 'node:fs';
import { mkdir, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';
import { createServer } from 'vite';
import { CdpClient } from '../../../../tools/gpu/cdp.ts';
import { discoverGpuLuid, findBrowser, removeProfile, startChrome } from '../../../../tools/gpu/chrome.ts';
import { installShutdownHandlers, registerCleanup, runCleanups } from '../../../../tools/gpu/cleanup.ts';

const helloDirectory = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const repoRoot = resolve(helloDirectory, '..', '..', '..');
const VIEWPORT = { width: 1280, height: 720 };
/** Until every probe is updated 100 times; the Intel UHD takes a few minutes at 1280 x 720. */
const READY_TIMEOUT_MS = 600_000;

interface HelloState {
    ready: boolean;
    updatesPerProbe: number;
    frames: number;
    error: string | null;
    adapter: string;
}

async function main(): Promise<number> {
    installShutdownHandlers();
    const { values } = parseArgs({
        args: process.argv.slice(2),
        options: {
            gpu: { type: 'string', default: 'nvidia' },
            scene: { type: 'string', default: 'cornell-box' },
            camera: { type: 'string', default: '1' },
            views: { type: 'string', default: 'full' },
            out: { type: 'string', default: join(repoRoot, 'bench', 'screenshots', 't1.6') },
        },
    });
    const gpu = values.gpu;
    if (gpu !== 'nvidia' && gpu !== 'intel') throw new Error('--gpu must be nvidia or intel');
    const out = resolve(values.out);

    const server = await createServer({ configFile: join(helloDirectory, 'vite.config.ts'), server: { port: 0 }, logLevel: 'warn' });
    const closeServer = registerCleanup(() => server.close());
    await server.listen();
    const address = server.httpServer?.address();
    if (!address || typeof address === 'string') throw new Error('The vite server did not report a port.');
    const origin = `http://localhost:${address.port}`;

    const luid = gpu === 'intel' ? await discoverGpuLuid('intel') : null;
    const profileDir = mkdtempSync(join(tmpdir(), 'ddgi-hello-chrome-'));
    const chrome = startChrome(findBrowser(null), { profileDir, luid, withoutGpu: false });
    let cdp: CdpClient | null = null;
    const closeChrome = registerCleanup(async () => {
        cdp?.close();
        await chrome.stop();
        await removeProfile(profileDir);
    });

    let failures = 0;
    try {
        await chrome.ready();
        const targets = (await (await fetch(`${chrome.devtoolsOrigin}/json/list`)).json()) as Array<{ type: string; webSocketDebuggerUrl: string }>;
        const target = targets.find((candidate) => candidate.type === 'page');
        if (!target) throw new Error('Chrome started without a page target.');
        cdp = await CdpClient.connect(target.webSocketDebuggerUrl);
        await Promise.all([cdp.send('Page.enable'), cdp.send('Runtime.enable'), cdp.send('Log.enable')]);
        await cdp.send('Emulation.setDeviceMetricsOverride', { ...VIEWPORT, deviceScaleFactor: 1, mobile: false });

        let browserErrors: string[] = [];
        cdp.on('Log.entryAdded', (params) => {
            const entry = params.entry as { level: string; text: string };
            if (entry.level === 'error' || /WGSL|WebGPU/i.test(entry.text)) browserErrors.push(`log: ${entry.text}`);
        });
        cdp.on('Runtime.consoleAPICalled', (params) => {
            if (params.type !== 'error') return;
            const args = (params.args as Array<{ value?: unknown; description?: string }>) ?? [];
            browserErrors.push(`console: ${args.map((arg) => (arg.value !== undefined ? String(arg.value) : (arg.description ?? ''))).join(' ')}`);
        });
        cdp.on('Runtime.exceptionThrown', (params) => {
            const details = params.exceptionDetails as { text: string; exception?: { description?: string } };
            browserErrors.push(`exception: ${details.exception?.description ?? details.text}`);
        });

        await mkdir(out, { recursive: true });
        for (const view of values.views.split(',').filter(Boolean)) {
            browserErrors = [];
            await cdp.send('Page.navigate', { url: 'about:blank' });
            await cdp.send('Page.navigate', { url: `${origin}/?scene=${values.scene}&camera=${values.camera}&view=${view}` });
            let state: HelloState | null = null;
            const deadline = Date.now() + READY_TIMEOUT_MS;
            while (Date.now() < deadline) {
                await new Promise((wait) => setTimeout(wait, 500));
                state = await cdp.evaluate<HelloState | null>('window.helloState ?? null');
                if (state?.ready || state?.error) break;
            }
            const file = join(out, `${gpu}-${values.scene}-camera${values.camera}-${view}.png`);
            const { data } = await cdp.send<{ data: string }>('Page.captureScreenshot', { format: 'png' }, 60_000);
            await writeFile(file, Buffer.from(data, 'base64'));
            const problems = [...browserErrors, ...(state?.error ? [state.error] : []), ...(state?.ready ? [] : ['never became ready'])];
            console.log(`${view}: ${state?.adapter ?? '?'}, ${state?.frames ?? 0} frames, ${state?.updatesPerProbe ?? 0} updates per probe -> ${file}`);
            for (const problem of problems) console.log(`    ERROR: ${problem}`);
            if (problems.length > 0) failures++;
        }
    } finally {
        await closeChrome();
        await closeServer();
    }
    return failures > 0 ? 1 : 0;
}

main().then(
    async (code) => {
        await runCleanups();
        process.exit(code);
    },
    async (error: unknown) => {
        console.error(error instanceof Error ? error.message : error);
        await runCleanups();
        process.exit(1);
    },
);
