/// <reference types="node" />
/**
 * The viewer's saved-bake round trip, headless (T5.2): the panel's "Save the
 * probes to a file" and "Load probes from a file", pressed as a person
 * would, on the real viewer page.
 *
 * 1. `?scene=<scene>&gi=only`: DDGI warms up as usual. Screenshot
 *    (`1-saved.png`), then "Save the probes to a file"; the browser's
 *    download lands in the output folder.
 * 2. `?scene=<scene>&gi=only&warmup=0`: a fresh page whose probes start from
 *    black and fill in live. Screenshot (`2-before-load.png`), then the file
 *    goes into the panel's file picker; screenshot after a second
 *    (`3-after-load.png`). The loaded picture should look like the saved one.
 * 3. `?scene=<scene>&follow=camera`: DDGI with the camera-following volume
 *    (another layout). The same file is refused, and the panel's status says
 *    why (`4-refused.png`).
 *
 * Prints each picture's mean brightness over the middle of the canvas (8-bit
 * sRGB after tone mapping, the GI-only view, so the GI alone): recorded, not
 * judged. Fails (exit code 1) on a console or GPU error, a save or load that
 * didn't happen, a load that wasn't refused in step 3, or a recompile
 * counter above 0.
 *
 *   node apps/viewer/scripts/bake-round-trip.ts [--scene sponza] [--gpu <vendor>] [--out <dir>] [--timeout <seconds>]
 *
 * `--gpu` checks the adapter's vendor contains the word (`apple`, `nvidia`,
 * `intel`); without it any adapter is taken. Default output folder:
 * bench/screenshots/bake-round-trip (gitignored). Run it under the bench
 * lock: node tools/reference/bench-lock.ts run --no-cool-wait --what "T5.2 viewer bake" -- node apps/viewer/scripts/bake-round-trip.ts
 */

import { existsSync, mkdtempSync } from 'node:fs';
import { mkdir, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createServer } from 'vite';
import { CdpClient } from '../../../tools/gpu/cdp.ts';
import { findBrowser, removeProfile, startChrome } from '../../../tools/gpu/chrome.ts';
import { installShutdownHandlers, registerCleanup, runCleanups } from '../../../tools/gpu/cleanup.ts';
import { decodePng } from '../../../tools/compare/png.ts';

const viewerDirectory = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const repoRoot = resolve(viewerDirectory, '..', '..');
const VIEWPORT = { width: 1600, height: 900 };

interface Options {
    scene: string;
    gpu: string | null;
    out: string;
    timeoutSeconds: number;
}

function parseOptions(argv: string[]): Options {
    const options: Options = { scene: 'sponza', gpu: null, out: join(repoRoot, 'bench', 'screenshots', 'bake-round-trip'), timeoutSeconds: 600 };
    for (let i = 0; i < argv.length; i++) {
        const flag = argv[i];
        const value = (): string => {
            const next = argv[++i];
            if (next === undefined) throw new Error(`${flag} needs a value`);
            return next;
        };
        if (flag === '--scene') options.scene = value();
        else if (flag === '--gpu') options.gpu = value().toLowerCase();
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
    bake: {
        saved: { file: string; bytes: number; ms: number } | null;
        loaded: { file: string; stale: boolean; changedParts: string[]; ms: number } | null;
        bakesLoaded: number;
        error: string | null;
    };
}

const sleep = (ms: number) => new Promise((resolveWait) => setTimeout(resolveWait, ms));

async function main(): Promise<number> {
    installShutdownHandlers();
    const options = parseOptions(process.argv.slice(2));
    await mkdir(options.out, { recursive: true });
    const bakeFile = join(options.out, `${options.scene}.ddgibake`);
    await rm(bakeFile, { force: true });

    const server = await createServer({ configFile: join(viewerDirectory, 'vite.config.ts'), server: { port: 0 }, logLevel: 'warn' });
    const closeServer = registerCleanup(() => server.close());
    await server.listen();
    const address = server.httpServer?.address();
    if (!address || typeof address === 'string') throw new Error('The vite server did not report a port.');
    const origin = `http://localhost:${address.port}`;

    const profileDir = mkdtempSync(join(tmpdir(), 'ddgi-viewer-bake-chrome-'));
    const chrome = startChrome(findBrowser(null), { profileDir, luid: null, withoutGpu: false });
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
        await Promise.all([client.send('Page.enable'), client.send('Runtime.enable'), client.send('Log.enable'), client.send('DOM.enable')]);
        await client.send('Emulation.setDeviceMetricsOverride', { ...VIEWPORT, deviceScaleFactor: 1, mobile: false });
        // The panel's save is a browser download: let it land in the output folder.
        await client.send('Browser.setDownloadBehavior', { behavior: 'allow', downloadPath: options.out });

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
        if (adapter === 'NO WEBGPU ADAPTER') throw new Error('Chrome has no WebGPU adapter.');
        if (options.gpu && !adapter.toLowerCase().includes(options.gpu)) throw new Error(`Asked for ${options.gpu}, but Chrome gave "${adapter}".`);

        const state = () => client.evaluate<ViewerState | null>('window.__viewer ?? null');
        const open = async (query: string): Promise<ViewerState> => {
            browserErrors = [];
            const started = Date.now();
            await client.send('Page.navigate', { url: 'about:blank' });
            await client.send('Page.navigate', { url: `${origin}/?${query}` });
            const deadline = started + options.timeoutSeconds * 1000;
            let current: ViewerState | null = null;
            while (Date.now() < deadline) {
                await sleep(500);
                current = await state();
                if (current?.ready) break;
                if (current && current.errors.length > 0 && current.frames === 0) break;
            }
            if (!current?.ready) throw new Error(`?${query}: not ready after ${((Date.now() - started) / 1000).toFixed(0)} s (${current?.errors.join('; ') ?? 'no state'})`);
            console.log(`?${query}: ready after ${((Date.now() - started) / 1000).toFixed(1)} s`);
            return current;
        };
        /** Saves a screenshot and returns its mean over the middle of the canvas (0-255). */
        const screenshot = async (file: string): Promise<number> => {
            const { data } = await client.send<{ data: string }>('Page.captureScreenshot', { format: 'png' }, 60_000);
            const bytes = Buffer.from(data, 'base64');
            await writeFile(join(options.out, file), bytes);
            const rect = await client.evaluate<{ x: number; y: number; width: number; height: number }>(`(() => {
                const box = document.querySelector('canvas').getBoundingClientRect();
                return { x: box.x, y: box.y, width: box.width, height: box.height };
            })()`);
            const image = decodePng(new Uint8Array(bytes));
            // The middle half of the canvas: clear of the panel, the stats box and the status pill.
            const x0 = Math.floor(rect.x + rect.width * 0.25);
            const x1 = Math.floor(rect.x + rect.width * 0.75);
            const y0 = Math.floor(rect.y + rect.height * 0.3);
            const y1 = Math.floor(rect.y + rect.height * 0.8);
            let sum = 0;
            let count = 0;
            for (let y = y0; y < y1; y++) {
                for (let x = x0; x < x1; x++) {
                    const at = (y * image.width + x) * 3;
                    sum += image.rgb8[at] + image.rgb8[at + 1] + image.rgb8[at + 2];
                    count += 3;
                }
            }
            return sum / Math.max(1, count);
        };
        const press = (text: string) =>
            client.evaluate<boolean>(`(() => {
                const button = [...document.querySelectorAll('button')].find((candidate) => candidate.textContent === ${JSON.stringify(text)} && !candidate.disabled);
                button?.click();
                return Boolean(button);
            })()`);
        /** Puts the bake file into the panel's file picker, as choosing it in the dialog would. */
        const chooseBakeFile = async (): Promise<void> => {
            const { root } = await client.send<{ root: { nodeId: number } }>('DOM.getDocument', { depth: 1 });
            const { nodeId } = await client.send<{ nodeId: number }>('DOM.querySelector', { nodeId: root.nodeId, selector: 'input[type=file][accept=".ddgibake"]' });
            if (!nodeId) throw new Error('the panel has no .ddgibake file picker');
            await client.send('DOM.setFileInputFiles', { nodeId, files: [bakeFile] });
        };
        const waitFor = async (what: string, test: (current: ViewerState) => boolean, seconds = 60): Promise<ViewerState> => {
            const deadline = Date.now() + seconds * 1000;
            for (;;) {
                const current = await state();
                if (current && test(current)) return current;
                if (Date.now() > deadline) throw new Error(`${what} didn't happen within ${seconds} s (bake state ${JSON.stringify(current?.bake)})`);
                await sleep(250);
            }
        };
        const noteErrors = (step: string, current: ViewerState | null) => {
            for (const error of [...(current?.errors ?? []), ...browserErrors]) problems.push(`${step}: ${error}`);
            if ((current?.recompiles ?? 0) > 0) problems.push(`${step}: recompiles ${current?.recompiles} (should stay 0)`);
        };

        // 1. Warm up as usual, then save.
        await open(`scene=${options.scene}&gi=only`);
        await sleep(2000);
        const savedMean = await screenshot('1-saved.png');
        if (!(await press('Save the probes to a file'))) throw new Error('no "Save the probes to a file" button');
        const afterSave = await waitFor('the save', (current) => current.bake.saved !== null || current.bake.error !== null);
        if (afterSave.bake.error) throw new Error(afterSave.bake.error);
        const downloadDeadline = Date.now() + 30_000;
        while (!existsSync(bakeFile) && Date.now() < downloadDeadline) await sleep(250);
        if (!existsSync(bakeFile)) throw new Error(`the download didn't reach ${bakeFile}`);
        const fileBytes = (await readFile(bakeFile)).byteLength;
        console.log(`saved: ${afterSave.bake.saved?.file}, ${fileBytes} bytes on disk (the page said ${afterSave.bake.saved?.bytes}) in ${afterSave.bake.saved?.ms.toFixed(1)} ms; picture mean ${savedMean.toFixed(2)}`);
        if (fileBytes !== afterSave.bake.saved?.bytes) problems.push(`the downloaded file is ${fileBytes} bytes, the page saved ${afterSave.bake.saved?.bytes}`);
        noteErrors('save', afterSave);

        // 2. A fresh page without the warm-up, then load.
        await open(`scene=${options.scene}&gi=only&warmup=0`);
        const beforeMean = await screenshot('2-before-load.png');
        await chooseBakeFile();
        const afterLoad = await waitFor('the load', (current) => current.bake.loaded !== null || current.bake.error !== null);
        if (afterLoad.bake.error) throw new Error(afterLoad.bake.error);
        await sleep(1000);
        const loadedMean = await screenshot('3-after-load.png');
        const loaded = await state();
        console.log(
            `loaded in ${afterLoad.bake.loaded?.ms.toFixed(1)} ms: stale ${afterLoad.bake.loaded?.stale} (${afterLoad.bake.loaded?.changedParts.join(', ') || 'nothing changed'}), bakes loaded ${loaded?.bake.bakesLoaded}; ` +
                `picture mean before the load ${beforeMean.toFixed(2)}, after ${loadedMean.toFixed(2)} (saved page ${savedMean.toFixed(2)}); recompiles ${loaded?.recompiles}`,
        );
        if (afterLoad.bake.loaded?.stale) problems.push(`the same scene loaded as stale (${afterLoad.bake.loaded.changedParts.join(', ')})`);
        if ((loaded?.bake.bakesLoaded ?? 0) !== 1) problems.push(`the technique counted ${loaded?.bake.bakesLoaded} bakes loaded, not 1`);
        noteErrors('load', loaded);

        // 3. Another layout refuses the file, with a reason.
        await open(`scene=${options.scene}&follow=camera`);
        await chooseBakeFile();
        const refused = await waitFor('the refusal', (current) => current.bake.loaded !== null || current.bake.error !== null);
        await sleep(500);
        await screenshot('4-refused.png');
        console.log(`with the camera volume: ${refused.bake.error ?? 'loaded (it should have been refused)'}`);
        if (!refused.bake.error) problems.push('a bake of another layout was loaded, not refused');
        // The refusal is a console warning, not an error; anything else counts.
        noteErrors('refusal', await state());

        await writeFile(
            join(options.out, 'summary.json'),
            `${JSON.stringify({ adapter, scene: options.scene, fileBytes, saveMs: afterSave.bake.saved?.ms, loadMs: afterLoad.bake.loaded?.ms, savedMean, beforeMean, loadedMean, refusal: refused.bake.error, problems }, null, 2)}\n`,
        );
        console.log(`\nScreenshots, the bake and summary.json in ${options.out}`);
    } catch (error) {
        problems.push(error instanceof Error ? error.message : String(error));
    } finally {
        await closeChrome();
        await closeServer();
    }
    for (const problem of problems) console.log(`PROBLEM: ${problem}`);
    return problems.length > 0 ? 1 : 0;
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
