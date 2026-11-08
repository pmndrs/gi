/// <reference types="node" />
/**
 * Opens the native compare page (compare.html, T2.8 check 3) in headless
 * Chrome on the viewer's own dev server, waits until both recordings have
 * loaded, plays them for a moment, and saves a screenshot. It fails on any
 * browser error, on a page that never loads its pair, and on sides that
 * don't show the same moment of the motion.
 *
 *   node apps/viewer/scripts/compare-shot.ts --motion two-room-light-sweep [--native <name>] [--ours <name>]
 *     [--seconds 2] (how long to play before the screenshot)  [--out <png>]  [--gpu nvidia|intel]
 *
 * The recordings are the dev server's (vite.config.ts: native/build/recordings
 * and bench/recordings/viewer, or DDGI_NATIVE_RECORDINGS / DDGI_VIEWER_RECORDINGS).
 */

import { mkdtempSync } from 'node:fs';
import { mkdir, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';
import { createServer } from 'vite';
import { CdpClient } from '../../../tools/gpu/cdp.ts';
import { discoverGpuLuid, findBrowser, removeProfile, startChrome } from '../../../tools/gpu/chrome.ts';
import { installShutdownHandlers, registerCleanup, runCleanups } from '../../../tools/gpu/cleanup.ts';

const viewerDirectory = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const repoRoot = resolve(viewerDirectory, '..', '..');
const VIEWPORT = { width: 1600, height: 1100 };
const LOAD_TIMEOUT_MS = 120_000;

interface PageState {
    status: string;
    time: string;
    readouts: string[];
    images: string[];
    playing: string;
}

const READ_STATE = `(() => ({
    status: document.getElementById('status')?.textContent ?? '',
    time: document.getElementById('time')?.textContent ?? '',
    readouts: [...document.querySelectorAll('.side .readout')].map((element) => element.textContent),
    images: [...document.querySelectorAll('.side img')].map((element) => element.getAttribute('src') ?? ''),
    playing: [...document.querySelectorAll('button')].map((element) => element.textContent).join(','),
}))()`;

async function main(): Promise<number> {
    installShutdownHandlers();
    const { values } = parseArgs({
        options: {
            motion: { type: 'string', default: 'two-room-light-sweep' },
            native: { type: 'string' },
            ours: { type: 'string' },
            seconds: { type: 'string', default: '2' },
            out: { type: 'string' },
            gpu: { type: 'string', default: 'nvidia' },
        },
    });
    const out = resolve(values.out ?? join(repoRoot, 'bench', 'screenshots', 'compare', `${values.motion}.png`));

    const server = await createServer({ configFile: join(viewerDirectory, 'vite.config.ts'), server: { port: 0 }, logLevel: 'warn' });
    registerCleanup(() => server.close());
    await server.listen();
    const address = server.httpServer?.address();
    if (!address || typeof address === 'string') throw new Error('The vite server did not report a port.');
    const origin = `http://localhost:${address.port}`;

    const luid = values.gpu === 'intel' ? await discoverGpuLuid('intel') : null;
    const profileDir = mkdtempSync(join(tmpdir(), 'ddgi-compare-chrome-'));
    const chrome = startChrome(findBrowser(null), { profileDir, luid, withoutGpu: false });
    let cdp: CdpClient | null = null;
    registerCleanup(async () => {
        cdp?.close();
        await chrome.stop();
        await removeProfile(profileDir);
    });

    await chrome.ready();
    const targets = (await (await fetch(`${chrome.devtoolsOrigin}/json/list`)).json()) as Array<{ type: string; webSocketDebuggerUrl: string }>;
    const target = targets.find((candidate) => candidate.type === 'page');
    if (!target) throw new Error('Chrome started without a page target.');
    cdp = await CdpClient.connect(target.webSocketDebuggerUrl);
    await Promise.all([cdp.send('Page.enable'), cdp.send('Runtime.enable'), cdp.send('Log.enable')]);
    await cdp.send('Emulation.setDeviceMetricsOverride', { ...VIEWPORT, deviceScaleFactor: 1, mobile: false });
    const browserErrors: string[] = [];
    cdp.on('Log.entryAdded', (params) => {
        const entry = params.entry as { level: string; text: string };
        if (entry.level === 'error') browserErrors.push(`log: ${entry.text}`);
    });
    cdp.on('Runtime.exceptionThrown', (params) => {
        const details = params.exceptionDetails as { text: string; exception?: { description?: string } };
        browserErrors.push(`exception: ${details.exception?.description ?? details.text}`);
    });

    const query = new URLSearchParams({ motion: values.motion! });
    if (values.native) query.set('native', values.native);
    if (values.ours) query.set('ours', values.ours);
    const url = `${origin}/compare.html?${query.toString()}`;
    console.log(`page: ${url}`);
    await cdp.send('Page.navigate', { url });

    const problems: string[] = [];
    const startedAt = Date.now();
    let state: PageState = await cdp.evaluate<PageState>(READ_STATE);
    while (!state.status.startsWith('Loaded') && Date.now() - startedAt < LOAD_TIMEOUT_MS) {
        if (/^Could not|^No motion/.test(state.status)) break;
        await new Promise((wait) => setTimeout(wait, 500));
        state = await cdp.evaluate<PageState>(READ_STATE);
    }
    console.log(`status: ${state.status}`);
    if (!state.status.startsWith('Loaded')) problems.push(`the pair never loaded: ${state.status}`);
    else {
        // Play from the start for a moment, then pause, and read both sides.
        await cdp.evaluate(`(() => { const scrub = document.getElementById('scrub'); scrub.value = '0'; scrub.dispatchEvent(new Event('input')); [...document.querySelectorAll('button')].find((b) => b.textContent === 'Play').click(); })()`);
        await new Promise((wait) => setTimeout(wait, Number(values.seconds) * 1000));
        await cdp.evaluate(`[...document.querySelectorAll('button')].find((b) => b.textContent === 'Pause')?.click()`);
        await new Promise((wait) => setTimeout(wait, 300));
        state = await cdp.evaluate<PageState>(READ_STATE);
        console.log(`time: ${state.time}`);
        for (const readout of state.readouts) console.log(`  ${readout}`);
        // In sync: both sides at the same frame of the motion.
        const frames = state.readouts.map((readout) => /^frame (\d+)/.exec(readout)?.[1]);
        if (frames.length !== 2 || frames[0] === undefined || frames[0] !== frames[1]) problems.push(`the sides are not at the same frame: ${frames.join(' and ')}`);
        if (state.images.some((source) => source === '')) problems.push('a side shows no picture');
    }
    const { data } = await cdp.send<{ data: string }>('Page.captureScreenshot', { format: 'png' }, 60_000);
    await mkdir(dirname(out), { recursive: true });
    await writeFile(out, Buffer.from(data, 'base64'));
    console.log(`screenshot: ${out}`);
    for (const error of browserErrors) problems.push(error);
    for (const problem of problems) console.log(`PROBLEM: ${problem}`);
    return problems.length === 0 ? 0 : 1;
}

main()
    .then(async (code) => {
        await runCleanups();
        process.exit(code);
    })
    .catch(async (error: unknown) => {
        console.error(error);
        await runCleanups();
        process.exit(1);
    });
