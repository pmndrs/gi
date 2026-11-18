/// <reference types="node" />
/**
 * Screenshots of every page of the built examples site, for looking at:
 * serves examples/dist under /gi/ with the GPU test server (as
 * the `examples-site` GPU page does), opens each page in headless Chrome
 * at 1280 × 720, waits until it reports GI (`window.ddgiExample`, or the
 * hello example's `window.helloState`) and a few seconds more, and saves a
 * PNG. The RC rail uses its deterministic capture controls for a warm rail
 * pose and three matching views of the receiver. Prints browser errors.
 *
 *   PAGES_BASE=/gi/ pnpm examples:build
 *   node tools/examples/screenshots.ts [--out <dir>] [--filter <path substring>]
 *   node tools/examples/screenshots.ts --filter 11-radiance-cascades
 *   Default output: bench/screenshots/t5.6 (gitignored).
 *
 * The GPU page is the check; this is for eyes. Run it under the bench lock.
 */

import { mkdtempSync } from 'node:fs';
import { mkdir, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';
import { CdpClient } from '../gpu/cdp.ts';
import { findBrowser, removeProfile, startChrome } from '../gpu/chrome.ts';
import { installShutdownHandlers, registerCleanup, runCleanups } from '../gpu/cleanup.ts';
import { EXAMPLES_SITE_BASE, EXAMPLES_SITE_FILES } from '../gpu/contract.ts';
import { startTestSite } from '../gpu/server.ts';

const repoRoot = resolve(dirname(fileURLToPath(import.meta.url)), '..', '..');
const VIEWPORT = { width: 1280, height: 720 };
const READY_TIMEOUT_MS = 120_000;
/** After the page reports GI, let it run this long before the picture: settled probes, a few seconds of motion. */
const SETTLE_MS = 4000;
/** Pages worth a second picture with other settings. */
const EXTRA_PAGES = ['06-materials/?scene=glass-wall', '06-materials/?scene=uv-board', '10-debug/?view=gi-only', '10-debug/?view=coverage', '10-debug/?view=probe-states', 'webgpu/hello/?view=gi'];

interface PageState {
    frames: number;
    giMean?: number | null;
    error: string | null;
}

async function saveScreenshot(cdp: CdpClient, file: string): Promise<void> {
    const { data } = await cdp.send<{ data: string }>('Page.captureScreenshot', { format: 'png' }, 60_000);
    await writeFile(file, Buffer.from(data, 'base64'));
}

async function captureRail(cdp: CdpClient, out: string): Promise<void> {
    const deadline = Date.now() + READY_TIMEOUT_MS;
    while (!(await cdp.evaluate<boolean>('Boolean(window.__rcRailVerify)'))) {
        if (Date.now() > deadline) throw new Error(`Split RC capture unavailable: ${await cdp.evaluate("document.getElementById('status')?.textContent")}`);
        await new Promise((wait) => setTimeout(wait, 250));
    }
    await cdp.evaluate("document.getElementById('panel').open = false");
    const shots = [
        { camera: 'rail', cue: 'amber', view: 'lit', frame: 100, frames: 64 },
        { camera: 'receiver', cue: 'blue', view: 'lit', frame: 0, frames: 64 },
        { camera: 'receiver', cue: 'blue', view: 'indirect', frame: 0, frames: 1 },
        { camera: 'receiver', cue: 'blue', view: 'direct', frame: 0, frames: 1 },
    ];
    for (const shot of shots) {
        const options = JSON.stringify({ ...shot, advance: false, occupancy: true });
        const capture = await cdp.send<{
            result: { value: { renderedFrames: number; occupancy: { peakProbes: number[]; capacities: number[]; probeBytes: number } } };
            exceptionDetails?: { text: string; exception?: { description?: string } };
        }>('Runtime.evaluate', {
            expression: `window.__rcRailVerify.render(${options}).then(({renderedFrames, occupancy}) => ({renderedFrames, occupancy}))`,
            awaitPromise: true,
            returnByValue: true,
        }, READY_TIMEOUT_MS);
        if (capture.exceptionDetails) throw new Error(capture.exceptionDetails.exception?.description ?? capture.exceptionDetails.text);
        const file = join(out, `11-radiance-cascades-${shot.camera}-${shot.cue}-${shot.view}.png`);
        await saveScreenshot(cdp, file);
        const { renderedFrames, occupancy } = capture.result.value;
        const slots = occupancy.peakProbes.map((count, n) => `${count}/${occupancy.capacities[n]}`).join(', ');
        console.log(`Split RC: ${renderedFrames} completed draws, ${shot.camera} pose ${shot.frame}, peak probes [${slots}], ${occupancy.probeBytes} probe bytes -> ${file}`);
    }
}

async function main(): Promise<number> {
    installShutdownHandlers();
    const { values } = parseArgs({ args: process.argv.slice(2), options: {
        out: { type: 'string', default: join(repoRoot, 'bench', 'screenshots', 't5.6') },
        filter: { type: 'string', default: '' },
    } });
    const out = resolve(values.out);

    const site = await startTestSite();
    const closeSite = registerCleanup(() => site.close());
    const files = (await (await fetch(`${site.origin}${EXAMPLES_SITE_FILES}`)).json()) as Array<{ path: string }>;
    const pages = files
        .map((file) => file.path)
        .filter((path) => path.endsWith('index.html') && path !== 'index.html')
        .map((path) => path.replace(/index\.html$/, ''))
        .sort();
    if (pages.length === 0) throw new Error(`examples/dist is empty: run PAGES_BASE=${EXAMPLES_SITE_BASE} pnpm examples:build first.`);
    const selected = [...pages, ...EXTRA_PAGES].filter((page) => page.includes(values.filter));
    if (selected.length === 0) throw new Error(`No example matched --filter "${values.filter}".`);

    const profileDir = mkdtempSync(join(tmpdir(), 'ddgi-examples-chrome-'));
    const chrome = startChrome(findBrowser(null), { profileDir, luid: null, withoutGpu: false });
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
        for (const page of selected) {
            browserErrors = [];
            await cdp.send('Page.navigate', { url: 'about:blank' });
            const url = new URL(page, `${site.origin}${EXAMPLES_SITE_BASE}`);
            const isRail = page === '11-radiance-cascades/';
            if (isRail) url.searchParams.set('verify', '1');
            await cdp.send('Page.navigate', { url: url.href });
            if (isRail) {
                await captureRail(cdp, out);
                for (const problem of browserErrors) console.log(`    ERROR: ${problem}`);
                if (browserErrors.length > 0) failures++;
                continue;
            }
            let state: PageState | null = null;
            const deadline = Date.now() + READY_TIMEOUT_MS;
            while (Date.now() < deadline) {
                await new Promise((wait) => setTimeout(wait, 250));
                state = await cdp.evaluate<PageState | null>('window.ddgiExample ?? window.helloState ?? null');
                if (state?.error || (state && state.frames >= 60 && state.giMean !== null)) break;
            }
            await new Promise((wait) => setTimeout(wait, SETTLE_MS));
            state = await cdp.evaluate<PageState | null>('window.ddgiExample ?? window.helloState ?? null');
            const file = join(out, `${page.replace(/\/\?/, '-').replace(/[/=?]+/g, '-').replace(/-$/, '')}.png`);
            await saveScreenshot(cdp, file);
            const problems = [...browserErrors, ...(state?.error ? [state.error] : []), ...(state ? [] : ['reported nothing'])];
            console.log(`${page}: ${state?.frames ?? 0} GI frames, GI mean ${state?.giMean?.toFixed(4) ?? '-'} -> ${file}`);
            for (const problem of problems) console.log(`    ERROR: ${problem}`);
            if (problems.length > 0) failures++;
        }
    } finally {
        await closeChrome();
        await closeSite();
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
