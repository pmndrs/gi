/// <reference types="node" />
/**
 * The quality preset picker, headless (T5.1): opens the real viewer page in
 * headless Chrome on each scene (`?scene=<id>`), waits until DDGI is warm,
 * takes a screenshot, then picks each other preset in the panel's "Quality
 * preset" list as a person would, waits until DDGI has started over and is
 * warm again, and takes another. Each time it checks that the panel's status
 * line names the preset, that the link says `preset=` (absent for high) and
 * that the budget slider moved to the preset's budget.
 *
 * It fails (exit code 1) on any console or GPU error, a page that never got
 * ready, a status line or link that doesn't follow the picker, or a
 * recompile counter above 0. It judges nothing else: the pictures are for
 * looking at.
 *
 *   node apps/viewer/scripts/preset-shots.ts
 *     [--gpu nvidia|intel]   which GPU Chrome gets (default: whatever Chrome picks; the Mac has one)
 *     [--scenes sponza,bistro-exterior]   default sponza,bistro-exterior
 *     [--out <dir>]          default bench/screenshots/presets (gitignored)
 *     [--timeout <seconds>]  per preset to get ready, default 600 (Bistro loads slowly)
 *
 * Run it under the bench lock: node tools/reference/bench-lock.ts run --no-cool-wait --what "T5.1 presets" -- node apps/viewer/scripts/preset-shots.ts
 */

import { mkdtempSync } from 'node:fs';
import { mkdir, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createServer } from 'vite';
import { CdpClient } from '../../../tools/gpu/cdp.ts';
import { discoverGpuLuid, findBrowser, removeProfile, startChrome } from '../../../tools/gpu/chrome.ts';
import { installShutdownHandlers, registerCleanup, runCleanups } from '../../../tools/gpu/cleanup.ts';
import { ddgiPreset, type DDGIPresetName } from '../../../packages/core/src/techniques/ddgi/presets.ts';

const viewerDirectory = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const repoRoot = resolve(viewerDirectory, '..', '..');
/** Wide enough that the panel's preset list and status line are readable beside the picture. */
const VIEWPORT = { width: 1600, height: 900 };
/** The order the picker is walked in: the page opens on high (the default), then the cheaper two. */
const PICKS: readonly DDGIPresetName[] = ['medium', 'low', 'high'];

interface Options {
    gpu: 'nvidia' | 'intel' | null;
    scenes: string[];
    out: string;
    timeoutSeconds: number;
}

function parseOptions(argv: string[]): Options {
    const options: Options = { gpu: null, scenes: ['sponza', 'bistro-exterior'], out: join(repoRoot, 'bench', 'screenshots', 'presets'), timeoutSeconds: 600 };
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
        } else if (flag === '--scenes') options.scenes = value().split(',').filter(Boolean);
        else if (flag === '--out') options.out = resolve(value());
        else if (flag === '--timeout') options.timeoutSeconds = Number(value());
        else if (flag === '--') continue;
        else throw new Error(`Unknown argument "${flag}"`);
    }
    return options;
}

/** What the page reports (`window.__viewer`, viewer-app.ts). */
interface ViewerState {
    ready: boolean;
    errors: string[];
    scene: string | null;
    frames: number;
    recompiles: number | null;
}

/** What the panel shows about DDGI. */
interface PanelReading {
    status: string;
    preset: string;
    budget: string;
    link: string;
}

const sleep = (ms: number) => new Promise((resolveWait) => setTimeout(resolveWait, ms));

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
    const profileDir = mkdtempSync(join(tmpdir(), 'ddgi-viewer-presets-chrome-'));
    const chrome = startChrome(findBrowser(null), { profileDir, luid, withoutGpu: false });
    let cdp: CdpClient | null = null;
    const closeChrome = registerCleanup(async () => {
        cdp?.close();
        await chrome.stop();
        await removeProfile(profileDir);
    });

    const problems: string[] = [];
    try {
        await chrome.ready();
        const targets = (await (await fetch(`${chrome.devtoolsOrigin}/json/list`)).json()) as Array<{ type: string; webSocketDebuggerUrl: string }>;
        const target = targets.find((candidate) => candidate.type === 'page');
        if (!target) throw new Error('Chrome started without a page target.');
        const client = await CdpClient.connect(target.webSocketDebuggerUrl);
        cdp = client;
        await Promise.all([client.send('Page.enable'), client.send('Runtime.enable'), client.send('Log.enable')]);
        await client.send('Emulation.setDeviceMetricsOverride', { ...VIEWPORT, deviceScaleFactor: 1, mobile: false });

        let browserErrors: string[] = [];
        client.on('Log.entryAdded', (params) => {
            const entry = params.entry as { level: string; text: string };
            if (entry.level === 'error') browserErrors.push(`log: ${entry.text}`);
        });
        client.on('Runtime.exceptionThrown', (params) => {
            const details = params.exceptionDetails as { text: string; exception?: { description?: string } };
            browserErrors.push(`exception: ${details.exception?.description ?? details.text}`);
        });

        await client.send('Page.navigate', { url: `${origin}/` });
        await sleep(800);
        const adapter = await client.evaluate<string>(`(async () => {
            const adapter = await navigator.gpu?.requestAdapter();
            if (!adapter) return 'NO WEBGPU ADAPTER';
            const info = adapter.info;
            return info.vendor + ' ' + info.architecture + (info.description ? ' (' + info.description + ')' : '');
        })()`);
        console.log(`adapter: ${adapter}`);
        if (options.gpu && !adapter.toLowerCase().includes(options.gpu)) throw new Error(`Asked for ${options.gpu}, but Chrome gave "${adapter}".`);

        await mkdir(options.out, { recursive: true });
        const state = () => client.evaluate<ViewerState | null>('window.__viewer ?? null');
        const panel = () =>
            client.evaluate<PanelReading>(`(() => {
                const select = [...document.querySelectorAll('select')].find((candidate) => [...candidate.options].some((option) => option.value === 'medium'));
                const budget = [...document.querySelectorAll('.label')].find((label) => (label.textContent ?? '').startsWith('Update budget per frame'));
                return { status: document.querySelector('.status')?.textContent ?? '', preset: select?.value ?? '', budget: budget?.textContent ?? '', link: location.search };
            })()`);
        const screenshot = async (file: string) => {
            // The panel scrolled to the picker, so each picture shows the preset chosen beside it.
            await client.evaluate(`(() => {
                const select = [...document.querySelectorAll('select')].find((candidate) => [...candidate.options].some((option) => option.value === 'medium'));
                select?.scrollIntoView({ block: 'center' });
            })()`);
            await sleep(200);
            const { data } = await client.send<{ data: string }>('Page.captureScreenshot', { format: 'png' }, 60_000);
            await writeFile(join(options.out, file), Buffer.from(data, 'base64'));
        };
        /** Waits for `ready` and for DDGI's status line to name `preset` (a new DDGI after a pick). */
        const waitUntilWarm = async (preset: DDGIPresetName, started: number): Promise<ViewerState | null> => {
            const deadline = Date.now() + options.timeoutSeconds * 1000;
            let current: ViewerState | null = null;
            while (Date.now() < deadline) {
                await sleep(500);
                current = await state();
                if (current && current.errors.length > 0) break;
                if (current?.ready && (await panel()).status.includes(`${preset} preset`)) return current;
            }
            throw new Error(`${preset}: not warm after ${((Date.now() - started) / 1000).toFixed(0)} s (${(await panel()).status})`);
        };

        for (const scene of options.scenes) {
            browserErrors = [];
            const sceneProblems: string[] = [];
            try {
                const opened = Date.now();
                await client.send('Page.navigate', { url: 'about:blank' });
                await client.send('Page.navigate', { url: `${origin}/?scene=${scene}` });
                await waitUntilWarm('high', opened);
                let reading = await panel();
                console.log(`${scene} high: warm after ${((Date.now() - opened) / 1000).toFixed(1)} s; ${reading.status}`);
                await screenshot(`${scene}-high-start.png`);
                for (const preset of PICKS) {
                    const picked = Date.now();
                    await client.evaluate(`(() => {
                        const select = [...document.querySelectorAll('select')].find((candidate) => [...candidate.options].some((option) => option.value === 'medium'));
                        select.value = ${JSON.stringify(preset)};
                        select.dispatchEvent(new Event('change'));
                    })()`);
                    const current = await waitUntilWarm(preset, picked);
                    reading = await panel();
                    const budgetMs = ddgiPreset(preset).budget.gpuMs;
                    console.log(`${scene} ${preset}: warm after ${((Date.now() - picked) / 1000).toFixed(1)} s; ${reading.status}; link ${reading.link}; ${reading.budget}; recompiles ${current?.recompiles ?? '-'}`);
                    const linkNames = reading.link.includes(`preset=${preset}`);
                    if (preset === 'high' ? reading.link.includes('preset=') : !linkNames) sceneProblems.push(`${preset}: the link is "${reading.link}"`);
                    if (!reading.budget.includes(`${budgetMs.toFixed(2)} ms`)) sceneProblems.push(`${preset}: the budget reads "${reading.budget}", not the preset's ${budgetMs} ms`);
                    if ((current?.recompiles ?? 0) > 0) sceneProblems.push(`${preset}: recompiles ${current?.recompiles}`);
                    await screenshot(`${scene}-${preset}.png`);
                }
                const last = await state();
                for (const error of [...(last?.errors ?? []), ...browserErrors]) sceneProblems.push(error);
            } catch (error) {
                sceneProblems.push(error instanceof Error ? error.message : String(error));
            }
            for (const problem of sceneProblems) console.log(`    PROBLEM (${scene}): ${problem}`);
            problems.push(...sceneProblems.map((problem) => `${scene}: ${problem}`));
        }
        console.log(`\nScreenshots in ${options.out}`);
        return problems.length > 0 ? 1 : 0;
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
