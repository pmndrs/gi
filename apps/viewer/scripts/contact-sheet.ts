/// <reference types="node" />
/**
 * Viewer contact sheet: opens the real viewer page in headless Chrome for
 * each scene (and each technique you ask for), waits until the page says it
 * is ready, saves a screenshot, and lists every console error and GPU error
 * the browser reported.
 *
 *   node apps/viewer/scripts/contact-sheet.ts
 *     [--gpu nvidia|intel]        which GPU Chrome gets (default nvidia)
 *     [--only a,b]                only these scene ids
 *     [--camera 1]                bookmark number (default 1)
 *     [--query "technique=vxgi&gi=only"]   extra URL parameters for every shot
 *     [--out <dir>]               default bench/screenshots/viewer-sheet (gitignored)
 *     [--timeout <seconds>]       per scene, default 360
 *     [--eval "<js>"]            run this in the page once ready and print the value
 *
 * It reuses the GPU test runner's Chrome launcher and DevTools client. The
 * exit code is 1 when any page reported an error or never became ready.
 */

import { mkdtempSync } from 'node:fs';
import { mkdir, readdir, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createServer } from 'vite';
import { CdpClient } from '../../../tools/gpu/cdp.ts';
import { discoverGpuLuid, findBrowser, removeProfile, startChrome } from '../../../tools/gpu/chrome.ts';
import { installShutdownHandlers, registerCleanup, runCleanups } from '../../../tools/gpu/cleanup.ts';

const viewerDirectory = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const repoRoot = resolve(viewerDirectory, '..', '..');
const VIEWPORT = { width: 1280, height: 720 };

interface Options {
    gpu: 'nvidia' | 'intel';
    only: string[];
    camera: string;
    query: string;
    out: string;
    timeoutSeconds: number;
    /** A JavaScript expression run in the page once it is ready; its value is printed. For debugging. */
    evaluate: string | null;
}

function parseOptions(argv: string[]): Options {
    const options: Options = {
        gpu: 'nvidia',
        only: [],
        camera: '1',
        query: '',
        out: join(repoRoot, 'bench', 'screenshots', 'viewer-sheet'),
        timeoutSeconds: 360,
        evaluate: null,
    };
    for (let i = 0; i < argv.length; i++) {
        const flag = argv[i];
        const value = (): string => {
            const next = argv[++i];
            if (next === undefined) throw new Error(`${flag} needs a value`);
            return next;
        };
        if (flag === '--gpu') {
            const gpu = value();
            if (gpu !== 'nvidia' && gpu !== 'intel') throw new Error('--gpu must be nvidia or intel');
            options.gpu = gpu;
        } else if (flag === '--only') options.only = value().split(',').filter(Boolean);
        else if (flag === '--camera') options.camera = value();
        else if (flag === '--query') options.query = value().replace(/^[?&]/, '');
        else if (flag === '--out') options.out = resolve(value());
        else if (flag === '--timeout') options.timeoutSeconds = Number(value());
        else if (flag === '--eval') options.evaluate = value();
        else if (flag === '--') continue;
        else throw new Error(`Unknown argument "${flag}"`);
    }
    return options;
}

interface ViewerState {
    ready: boolean;
    errors: string[];
    scene: string | null;
    camera: string | null;
    loadMs: number;
    prepareMs: number | null;
    frames: number;
    adapter: string;
}

interface Shot {
    scene: string;
    file: string;
    state: ViewerState | null;
    browserErrors: string[];
    problem: string | null;
    seconds: number;
}

function escapeHtml(text: string): string {
    return text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

function indexHtml(shots: Shot[], adapter: string, query: string): string {
    const cells = shots
        .map((shot) => {
            const problems = [...(shot.state?.errors ?? []), ...shot.browserErrors, ...(shot.problem ? [shot.problem] : [])];
            const timing = shot.state ? `load ${(shot.state.loadMs / 1000).toFixed(1)} s, set-up ${shot.state.prepareMs === null ? '-' : `${(shot.state.prepareMs / 1000).toFixed(1)} s`}` : 'no data';
            return `<figure class="${problems.length ? 'bad' : 'ok'}">
  <a href="${escapeHtml(shot.file)}"><img src="${escapeHtml(shot.file)}" loading="lazy" width="640" height="360"></a>
  <figcaption><b>${escapeHtml(shot.scene)}</b> ${escapeHtml(shot.state?.camera ?? '')}<br>${escapeHtml(timing)}${problems.map((p) => `<br><span class="err">${escapeHtml(p)}</span>`).join('')}</figcaption>
</figure>`;
        })
        .join('\n');
    return `<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Viewer contact sheet</title>
<style>
body { font: 13px/1.4 system-ui, sans-serif; background: #181818; color: #ddd; margin: 12px; }
.grid { display: grid; grid-template-columns: repeat(auto-fill, 640px); gap: 12px; }
figure { margin: 0; } figure.bad img { outline: 3px solid #c33; } img { display: block; background: #000; }
figcaption { padding: 4px 0; } .err { color: #f88; }
</style></head><body>
<h1>Viewer contact sheet</h1>
<p>The real viewer page, default technique unless the query says otherwise (${escapeHtml(query || 'no extra query')}). Adapter: ${escapeHtml(adapter)}. Generated ${new Date().toISOString()}.</p>
<div class="grid">
${cells}
</div></body></html>
`;
}

async function sceneIds(): Promise<string[]> {
    const directory = join(repoRoot, 'scenes', 'manifests');
    return (await readdir(directory)).filter((file) => file.endsWith('.json')).map((file) => file.slice(0, -'.json'.length)).sort();
}

async function main(): Promise<number> {
    installShutdownHandlers();
    const options = parseOptions(process.argv.slice(2));

    const server = await createServer({ configFile: join(viewerDirectory, 'vite.config.ts'), server: { port: 0 }, logLevel: 'warn' });
    const closeServer = registerCleanup(() => server.close());
    await server.listen();
    const address = server.httpServer?.address();
    if (!address || typeof address === 'string') throw new Error('The vite server did not report a port.');
    const origin = `http://localhost:${address.port}`;

    const luid = options.gpu === 'intel' ? await discoverGpuLuid('intel') : null;
    const profileDir = mkdtempSync(join(tmpdir(), 'ddgi-viewer-sheet-chrome-'));
    const chrome = startChrome(findBrowser(null), { profileDir, luid, withoutGpu: false });
    let cdp: CdpClient | null = null;
    const closeChrome = registerCleanup(async () => {
        cdp?.close();
        await chrome.stop();
        await removeProfile(profileDir);
    });

    try {
        await chrome.ready();
        const targets = (await (await fetch(`${chrome.devtoolsOrigin}/json/list`)).json()) as Array<{ type: string; webSocketDebuggerUrl: string }>;
        const target = targets.find((candidate) => candidate.type === 'page');
        if (!target) throw new Error('Chrome started without a page target.');
        cdp = await CdpClient.connect(target.webSocketDebuggerUrl);
        await Promise.all([cdp.send('Page.enable'), cdp.send('Runtime.enable'), cdp.send('Log.enable')]);
        await cdp.send('Emulation.setDeviceMetricsOverride', { ...VIEWPORT, deviceScaleFactor: 1, mobile: false });

        // Everything the browser calls an error, whoever logged it.
        let browserErrors: string[] = [];
        cdp.on('Log.entryAdded', (params) => {
            const entry = params.entry as { level: string; text: string };
            if (entry.level === 'error') browserErrors.push(`log: ${entry.text}`);
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

        await cdp.send('Page.navigate', { url: `${origin}/` });
        await new Promise((resolveWait) => setTimeout(resolveWait, 800));
        const adapter = await cdp.evaluate<string>(`(async () => {
            const adapter = await navigator.gpu?.requestAdapter();
            if (!adapter) return 'NO WEBGPU ADAPTER';
            const info = adapter.info;
            return info.vendor + ' ' + info.architecture + (info.description ? ' (' + info.description + ')' : '');
        })()`);
        console.log(`adapter: ${adapter}`);
        if (adapter === 'NO WEBGPU ADAPTER') throw new Error('WebGPU is not available in this Chrome.');
        if (!adapter.toLowerCase().includes(options.gpu)) throw new Error(`Asked for ${options.gpu}, but Chrome gave "${adapter}".`);

        const known = await sceneIds();
        const ids = options.only.length > 0 ? options.only : known;
        for (const id of ids) if (!known.includes(id)) throw new Error(`Unknown scene id "${id}". Known: ${known.join(', ')}`);

        await mkdir(options.out, { recursive: true });
        const shots: Shot[] = [];
        for (const id of ids) {
            const started = Date.now();
            const file = `${id}.png`;
            let state: ViewerState | null = null;
            let problem: string | null = null;
            browserErrors = [];
            try {
                await cdp.send('Page.navigate', { url: 'about:blank' });
                const query = `scene=${id}&camera=${options.camera}${options.query ? `&${options.query}` : ''}`;
                await cdp.send('Page.navigate', { url: `${origin}/?${query}` });
                const deadline = Date.now() + options.timeoutSeconds * 1000;
                while (Date.now() < deadline) {
                    await new Promise((resolveWait) => setTimeout(resolveWait, 500));
                    state = await cdp.evaluate<ViewerState | null>('window.__viewer ?? null');
                    if (state?.ready) break;
                    // A load that failed outright never becomes ready; stop waiting.
                    if (state && state.errors.length > 0 && state.prepareMs === null && state.frames === 0) break;
                }
                if (!state?.ready) problem = `not ready after ${options.timeoutSeconds} s`;
                if (options.evaluate) console.log(`eval: ${JSON.stringify(await cdp.evaluate(options.evaluate))}`);
                const { data } = await cdp.send<{ data: string }>('Page.captureScreenshot', { format: 'png' }, 60_000);
                await writeFile(join(options.out, file), Buffer.from(data, 'base64'));
                // Give late GPU errors a moment to reach the log before closing the books on this scene.
                await new Promise((resolveWait) => setTimeout(resolveWait, 300));
                state = await cdp.evaluate<ViewerState | null>('window.__viewer ?? null');
            } catch (error) {
                problem = error instanceof Error ? error.message : String(error);
            }
            const seconds = (Date.now() - started) / 1000;
            shots.push({ scene: id, file, state, browserErrors: [...browserErrors], problem, seconds });
            const errorCount = (state?.errors.length ?? 0) + browserErrors.length + (problem ? 1 : 0);
            console.log(`${id}: ${errorCount} error(s), load ${((state?.loadMs ?? 0) / 1000).toFixed(1)} s, set-up ${state?.prepareMs === null || state === null ? '-' : `${(state.prepareMs / 1000).toFixed(1)} s`}, ${seconds.toFixed(1)} s total`);
            for (const message of [...(state?.errors ?? []), ...browserErrors, ...(problem ? [problem] : [])]) console.log(`    ERROR: ${message}`);
        }

        await writeFile(join(options.out, 'index.html'), indexHtml(shots, adapter, options.query));
        console.log(`\n${shots.length} screenshot(s) and index.html in ${options.out}`);
        return shots.some((shot) => shot.problem || (shot.state?.errors.length ?? 0) > 0 || shot.browserErrors.length > 0) ? 1 : 0;
    } finally {
        await closeChrome();
        await closeServer();
    }
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

