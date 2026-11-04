/// <reference types="node" />
/**
 * Contact sheet: opens the scene check page in headless Chrome for every scene
 * and every camera bookmark, saves a screenshot of each, and writes one
 * index.html that shows them all in a grid with the scene's numbers.
 *
 *   pnpm --filter @pmndrs/ddgi-scenes contact-sheet --
 *     [--gpu nvidia|intel]     which GPU Chrome gets (default nvidia)
 *     [--only a,b]             only these scene ids (heavy scenes, like sponza-full, only when named here)
 *     [--camera 1]             only this bookmark number
 *     [--cam px,py,pz,tx,ty,tz,fov]  a one-off free camera instead of the bookmarks (for finding good views)
 *     [--downloads <dir>[;<dir>]  read downloaded models from these folders instead of assets/downloads
 *     [--out <dir>]            default bench/screenshots/contact-sheet (gitignored)
 *
 * It reuses the GPU test runner's Chrome launcher and DevTools client in
 * tools/gpu rather than carrying its own. The check page is served by a vite
 * dev server started here.
 *
 * The exit code is 1 when any page reported an error or never became ready.
 */

import { mkdtempSync } from 'node:fs';
import { mkdir, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { delimiter, dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createServer } from 'vite';
import { CdpClient } from '../../tools/gpu/cdp.ts';
import { discoverGpuLuid, findBrowser, removeProfile, startChrome } from '../../tools/gpu/chrome.ts';
import { installShutdownHandlers, registerCleanup, runCleanups } from '../../tools/gpu/cleanup.ts';
import { loadManifests } from '../src/export';
import { isHeavyScene } from '../src/manifest';

const scenesDirectory = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const repoRoot = resolve(scenesDirectory, '..');

/** A big model can take a while to download into the page and compile. */
const PAGE_READY_TIMEOUT_MS = 240_000;
const VIEWPORT = { width: 1280, height: 720 };

interface Options {
    gpu: 'nvidia' | 'intel';
    only: string[];
    camera: number | null;
    freeCamera: string | null;
    downloads: string | null;
    out: string;
}

function parseOptions(argv: string[]): Options {
    const options: Options = {
        gpu: 'nvidia',
        only: [],
        camera: null,
        freeCamera: null,
        downloads: null,
        out: join(repoRoot, 'bench', 'screenshots', 'contact-sheet'),
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
        else if (flag === '--camera') options.camera = Number(value());
        else if (flag === '--cam') options.freeCamera = value();
        else if (flag === '--downloads') options.downloads = value().split(delimiter).map((directory) => resolve(directory)).join(delimiter);
        else if (flag === '--out') options.out = resolve(value());
        else if (flag === '--') continue;
        else throw new Error(`Unknown argument "${flag}"`);
    }
    return options;
}

interface PageState {
    ready: boolean;
    errors: string[];
    stats: { triangles: number; meshes: number; materials: number; textures: number };
    lighting: string;
    scene: string | null;
    camera: string | null;
}

interface Shot {
    scene: string;
    cameraNumber: number;
    cameraName: string;
    file: string;
    state: PageState | null;
    problem: string | null;
    seconds: number;
}

function escapeHtml(text: string): string {
    return text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

function indexHtml(shots: Shot[], adapter: string): string {
    const cells = shots
        .map((shot) => {
            const state = shot.state;
            const problems = [...(state?.errors ?? []), ...(shot.problem ? [shot.problem] : [])];
            const stats = state ? `${state.stats.triangles.toLocaleString('en')} tris, ${state.stats.meshes} meshes, ${state.stats.materials} materials, ${state.stats.textures} textures` : 'no data';
            return `<figure class="${problems.length ? 'bad' : 'ok'}">
  <a href="${escapeHtml(shot.file)}"><img src="${escapeHtml(shot.file)}" loading="lazy" width="640" height="360"></a>
  <figcaption><b>${escapeHtml(shot.scene)}</b> #${shot.cameraNumber} ${escapeHtml(shot.cameraName)}<br>${escapeHtml(stats)}<br>${escapeHtml(state?.lighting ?? '')}${problems.map((p) => `<br><span class="err">${escapeHtml(p)}</span>`).join('')}</figcaption>
</figure>`;
        })
        .join('\n');
    return `<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Scene contact sheet</title>
<style>
body { font: 13px/1.4 system-ui, sans-serif; background: #181818; color: #ddd; margin: 12px; }
.grid { display: grid; grid-template-columns: repeat(auto-fill, 640px); gap: 12px; }
figure { margin: 0; } figure.bad img { outline: 3px solid #c33; } img { display: block; background: #000; }
figcaption { padding: 4px 0; } .err { color: #f88; }
</style></head><body>
<h1>Scene contact sheet</h1>
<p>Direct light only (manifest lights, plus emissive quads as area lights); no bounce light. Adapter: ${escapeHtml(adapter)}. Generated ${new Date().toISOString()}.</p>
<div class="grid">
${cells}
</div></body></html>
`;
}

async function main(): Promise<number> {
    installShutdownHandlers();
    const options = parseOptions(process.argv.slice(2));
    if (options.downloads) process.env.DDGI_DOWNLOADS_DIR = options.downloads;

    // The vite dev server for the check page.
    const server = await createServer({ configFile: join(scenesDirectory, 'vite.config.ts'), server: { port: 0 }, logLevel: 'warn' });
    const closeServer = registerCleanup(() => server.close());
    await server.listen();
    const address = server.httpServer?.address();
    if (!address || typeof address === 'string') throw new Error('The vite server did not report a port.');
    const origin = `http://localhost:${address.port}`;

    // Chrome, pinned to the GPU asked for.
    const luid = options.gpu === 'intel' ? await discoverGpuLuid('intel') : null;
    const profileDir = mkdtempSync(join(tmpdir(), 'ddgi-sheet-chrome-'));
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
        await Promise.all([cdp.send('Page.enable'), cdp.send('Runtime.enable')]);
        await cdp.send('Emulation.setDeviceMetricsOverride', { ...VIEWPORT, deviceScaleFactor: 1, mobile: false });

        // Say which GPU this is, so a sheet is never mistaken for another device's.
        await cdp.send('Page.navigate', { url: `${origin}/` });
        await new Promise((resolveWait) => setTimeout(resolveWait, 500));
        const adapter = await cdp.evaluate<string>(`(async () => {
            const adapter = await navigator.gpu?.requestAdapter();
            if (!adapter) return 'NO WEBGPU ADAPTER';
            const info = adapter.info;
            return info.vendor + ' ' + info.architecture + (info.description ? ' (' + info.description + ')' : '');
        })()`);
        console.log(`adapter: ${adapter}`);
        if (adapter === 'NO WEBGPU ADAPTER') throw new Error('WebGPU is not available in this Chrome.');
        if (!adapter.toLowerCase().includes(options.gpu)) {
            throw new Error(`Asked for ${options.gpu}, but Chrome gave "${adapter}".`);
        }

        // Which scenes and bookmarks exist: read the manifests the same way the page does.
        // Without --only: every scene except the heavy ones, which load only when named.
        const manifests = (await loadManifests()).filter((manifest) =>
            options.only.length === 0 ? !isHeavyScene(manifest) : options.only.includes(manifest.id),
        );
        if (options.only.length > 0 && manifests.length !== options.only.length) {
            throw new Error(`Unknown scene id in --only. Known ids: ${(await loadManifests()).map((m) => m.id).join(', ')}`);
        }

        await mkdir(options.out, { recursive: true });
        const shots: Shot[] = [];
        for (const manifest of manifests) {
            for (const [index, camera] of manifest.cameras.entries()) {
                const cameraNumber = index + 1;
                if (options.freeCamera !== null && index > 0) continue;
                if (options.camera !== null && options.camera !== cameraNumber) continue;
                const started = Date.now();
                const file = options.freeCamera !== null ? `${manifest.id}-free.png` : `${manifest.id}-${cameraNumber}.png`;
                const freeParameter = options.freeCamera !== null ? `&cam=${options.freeCamera}` : '';
                let state: PageState | null = null;
                let problem: string | null = null;
                try {
                    await cdp.send('Page.navigate', { url: 'about:blank' });
                    await cdp.send('Page.navigate', { url: `${origin}/?scene=${manifest.id}&camera=${cameraNumber}${freeParameter}` });
                    const deadline = Date.now() + PAGE_READY_TIMEOUT_MS;
                    while (Date.now() < deadline) {
                        await new Promise((resolveWait) => setTimeout(resolveWait, 250));
                        state = await cdp.evaluate<PageState | null>('window.__sceneCheck ?? null');
                        if (state?.ready) break;
                    }
                    if (!state?.ready) problem = `not ready after ${PAGE_READY_TIMEOUT_MS / 1000} s`;
                    const { data } = await cdp.send<{ data: string }>('Page.captureScreenshot', { format: 'png' }, 60_000);
                    await writeFile(join(options.out, file), Buffer.from(data, 'base64'));
                } catch (error) {
                    problem = error instanceof Error ? error.message : String(error);
                }
                const seconds = (Date.now() - started) / 1000;
                shots.push({ scene: manifest.id, cameraNumber, cameraName: camera.name, file, state, problem, seconds });
                const stats = state?.stats;
                const errorCount = (state?.errors.length ?? 0) + (problem ? 1 : 0);
                console.log(
                    `${manifest.id} #${cameraNumber} ${camera.name}: ${stats ? `${stats.triangles} tris, ${stats.meshes} meshes, ${stats.materials} materials, ${stats.textures} textures` : 'no data'}, ${errorCount} error(s), ${seconds.toFixed(1)} s`,
                );
                for (const message of [...(state?.errors ?? []), ...(problem ? [problem] : [])]) console.log(`    ERROR: ${message}`);
            }
        }

        await writeFile(join(options.out, 'index.html'), indexHtml(shots, adapter));
        console.log(`\n${shots.length} screenshot(s) and index.html in ${options.out}`);
        return shots.some((shot) => shot.problem || (shot.state?.errors.length ?? 0) > 0) ? 1 : 0;
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
