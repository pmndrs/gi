/// <reference types="node" />
/**
 * The moving-geometry scenarios, headless (T3.8 check 3): opens the real
 * viewer page in headless Chrome for each scenario (`?scenario=<id>`, the
 * link Dennis would use), waits until DDGI is warm, starts the scenario
 * again from the panel's list as a person would, and takes screenshots at
 * set moments of its motion. On the cabinet row it also presses "Add a box"
 * and "Copy a box". With `--measure` it then presses the scenario's
 * "Measure a step" and waits for the response meter's curve.
 *
 * It fails (exit code 1) on any console or GPU error, a page that never got
 * ready, a motion that isn't playing, or a recompile counter above 0. It
 * judges nothing else: the pictures are for looking at.
 *
 *   node apps/viewer/scripts/scenario-shots.ts
 *     [--gpu nvidia|intel]   which GPU Chrome gets (default nvidia)
 *     [--only door,furniture]
 *     [--measure]            also run the response meter on each scenario's step
 *     [--sample]             print the stats box's frame time, probes updated and DDGI's GPU time once a second
 *                            while the motion plays (recorded, not judged)
 *     [--out <dir>]          default bench/screenshots/scenarios (gitignored)
 *     [--timeout <seconds>]  per scenario to get ready, default 600 (Bistro loads slowly)
 *
 * Run it under the bench lock: node tools/reference/bench-lock.ts run --no-cool-wait --what "T3.8 scenarios" -- node apps/viewer/scripts/scenario-shots.ts
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
import { SCENARIOS } from '../src/scene/scenarios.ts';

const viewerDirectory = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const repoRoot = resolve(viewerDirectory, '..', '..');
/** Wide enough that the panel's "Moving things" section and the stats box are readable beside the picture. */
const VIEWPORT = { width: 1600, height: 900 };

/** Seconds into each scenario's motion at which a screenshot is taken (the motions are in its manifest). */
const SHOT_SECONDS: Record<string, readonly number[]> = {
    // Cabinet slide: at rest, slid 1 m out (held 5-9 s), back.
    furniture: [3.5, 8.5, 11],
    // Door: open (held to 3 s), shut 4 s in and 1 s later (the leak should be gone), open again.
    door: [2.5, 4.2, 5.5, 8.5, 11.5],
    // Cabinet door: shut, open (held 4.5-8.5 s), shut again.
    cabinets: [2.5, 7.5, 11],
    // Wind gusts: the lean at 9 s (0.9) and the swing back at 11 s and 15 s.
    'wind-tree': [1, 9, 11, 15],
    // Fly-through: the start, mid first leg, after each turn, far down the street, and on the way back.
    'fly-through': [0.8, 2.5, 5, 9, 13.5, 20, 28],
};

interface Options {
    gpu: 'nvidia' | 'intel';
    only: string[];
    measure: boolean;
    sample: boolean;
    out: string;
    timeoutSeconds: number;
}

function parseOptions(argv: string[]): Options {
    const options: Options = { gpu: 'nvidia', only: [], measure: false, sample: false, out: join(repoRoot, 'bench', 'screenshots', 'scenarios'), timeoutSeconds: 600 };
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
        else if (flag === '--measure') options.measure = true;
        else if (flag === '--sample') options.sample = true;
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
    camera: string | null;
    frames: number;
    adapter: string;
    scenario: string | null;
    motion: { id: string; seconds: number } | null;
    recompiles: number | null;
}

interface ScenarioRun {
    id: string;
    shots: Array<{ file: string; seconds: number; recompiles: number | null }>;
    problems: string[];
    meter: string | null;
}

const sleep = (ms: number) => new Promise((resolveWait) => setTimeout(resolveWait, ms));

function escapeHtml(text: string): string {
    return text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

function indexHtml(runs: ScenarioRun[], adapter: string): string {
    const sections = runs
        .map((run) => {
            const shots = run.shots
                .map((shot) => `<figure><a href="${escapeHtml(shot.file)}"><img src="${escapeHtml(shot.file)}" width="640" loading="lazy"></a><figcaption>${shot.seconds.toFixed(1)} s into the motion, recompiles ${shot.recompiles ?? '-'}</figcaption></figure>`)
                .join('\n');
            const problems = run.problems.map((problem) => `<p class="err">${escapeHtml(problem)}</p>`).join('');
            const meter = run.meter ? `<p>Meter: ${escapeHtml(run.meter)}</p>` : '';
            return `<h2>${escapeHtml(run.id)}</h2>${problems}${meter}<div class="grid">${shots}</div>`;
        })
        .join('\n');
    return `<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Viewer scenarios</title>
<style>body { font: 13px/1.4 system-ui, sans-serif; background: #181818; color: #ddd; margin: 12px; } .grid { display: grid; grid-template-columns: repeat(auto-fill, 640px); gap: 12px; } figure { margin: 0; } .err { color: #f88; }</style>
</head><body><h1>Viewer scenarios (T3.8)</h1><p>Adapter: ${escapeHtml(adapter)}. Generated ${new Date().toISOString()}.</p>${sections}</body></html>\n`;
}

async function main(): Promise<number> {
    installShutdownHandlers();
    const options = parseOptions(process.argv.slice(2));
    const scenarios = options.only.length > 0 ? SCENARIOS.filter((scenario) => options.only.includes(scenario.id)) : SCENARIOS;
    for (const id of options.only) if (!SCENARIOS.some((scenario) => scenario.id === id)) throw new Error(`Unknown scenario "${id}". Known: ${SCENARIOS.map((scenario) => scenario.id).join(', ')}`);

    const server = await createServer({ configFile: join(viewerDirectory, 'vite.config.ts'), server: { port: 0 }, logLevel: 'warn' });
    const closeServer = registerCleanup(() => server.close());
    await server.listen();
    const address = server.httpServer?.address();
    if (!address || typeof address === 'string') throw new Error('The vite server did not report a port.');
    const origin = `http://localhost:${address.port}`;

    const luid = options.gpu === 'intel' ? await discoverGpuLuid('intel') : null;
    const profileDir = mkdtempSync(join(tmpdir(), 'ddgi-viewer-scenarios-chrome-'));
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
        if (!adapter.toLowerCase().includes(options.gpu)) throw new Error(`Asked for ${options.gpu}, but Chrome gave "${adapter}".`);

        await mkdir(options.out, { recursive: true });
        const state = () => client.evaluate<ViewerState | null>('window.__viewer ?? null');
        const screenshot = async (file: string) => {
            const { data } = await client.send<{ data: string }>('Page.captureScreenshot', { format: 'png' }, 60_000);
            await writeFile(join(options.out, file), Buffer.from(data, 'base64'));
        };
        /** The stats box's frame, probes-updated and DDGI GPU lines, squeezed onto one line. */
        const statsLine = () =>
            client.evaluate<string>(`(() => {
                const rows = [...document.querySelectorAll('.stats-overlay div')].map((row) => (row.textContent ?? '').trim());
                const pick = (name) => (rows.find((row) => row.startsWith(name)) ?? '').replace(/ +/g, ' ').slice(0, 90);
                return [pick('frame:'), pick('probes updated'), pick('update (GPU)')].join(' | ');
            })()`);
        /** Presses a panel button by its text, as a person would. */
        const press = (text: string) =>
            client.evaluate<boolean>(`(() => {
                const button = [...document.querySelectorAll('button')].find((candidate) => candidate.textContent === ${JSON.stringify(text)} && !candidate.disabled && candidate.offsetParent !== null);
                button?.click();
                return Boolean(button);
            })()`);

        const runs: ScenarioRun[] = [];
        for (const scenario of scenarios) {
            const run: ScenarioRun = { id: scenario.id, shots: [], problems: [], meter: null };
            runs.push(run);
            browserErrors = [];
            const started = Date.now();
            try {
                await client.send('Page.navigate', { url: 'about:blank' });
                await client.send('Page.navigate', { url: `${origin}/?scenario=${scenario.id}` });
                let current: ViewerState | null = null;
                const deadline = Date.now() + options.timeoutSeconds * 1000;
                while (Date.now() < deadline) {
                    await sleep(500);
                    current = await state();
                    if (current?.ready) break;
                    if (current && current.errors.length > 0 && current.frames === 0) break;
                }
                if (!current?.ready) throw new Error(`not ready after ${((Date.now() - started) / 1000).toFixed(0)} s`);
                console.log(`${scenario.id}: ready after ${((Date.now() - started) / 1000).toFixed(1)} s on ${current.scene} (${current.camera}); motion ${current.motion?.id ?? 'none'}`);
                if (current.motion?.id !== scenario.motion) run.problems.push(`the link should have started ${scenario.motion}, but ${current.motion?.id ?? 'nothing'} plays`);

                // From the start: choose the scenario again in the panel's list.
                await client.evaluate(`(() => {
                    const select = [...document.querySelectorAll('select')].find((candidate) => [...candidate.options].some((option) => option.value === ${JSON.stringify(scenario.id)}));
                    select.value = '';
                    select.dispatchEvent(new Event('change'));
                    select.value = ${JSON.stringify(scenario.id)};
                    select.dispatchEvent(new Event('change'));
                })()`);
                // The page reports the motion's time a few times a second: wait until it shows the restart.
                const restartDeadline = Date.now() + 30_000;
                for (current = await state(); (current?.motion?.seconds ?? Infinity) > 1 && Date.now() < restartDeadline; current = await state()) await sleep(50);
                if ((current?.motion?.seconds ?? Infinity) > 1) run.problems.push('choosing the scenario again did not start its motion over');
                let sampledAt = 0;
                for (const [index, seconds] of (SHOT_SECONDS[scenario.id] ?? [2]).entries()) {
                    const shotDeadline = Date.now() + 60_000;
                    for (current = await state(); (current?.motion?.seconds ?? 0) < seconds && Date.now() < shotDeadline; current = await state()) {
                        if (options.sample && Date.now() - sampledAt >= 1000) {
                            sampledAt = Date.now();
                            console.log(`    ${(current?.motion?.seconds ?? 0).toFixed(1)} s: ${await statsLine()}`);
                        }
                        await sleep(30);
                    }
                    if (current?.motion?.id !== scenario.motion) run.problems.push(`at ${seconds} s ${scenario.motion} wasn't playing (${current?.motion?.id ?? 'nothing'})`);
                    const file = `${scenario.id}-${index + 1}.png`;
                    await screenshot(file);
                    run.shots.push({ file, seconds: current?.motion?.seconds ?? 0, recompiles: current?.recompiles ?? null });
                }

                if (scenario.addsCabinets) {
                    // Ten new shapes, then twenty copies of the last one.
                    for (let count = 0; count < 10; count++) if (!(await press('Add a box'))) run.problems.push('no "Add a box" button');
                    for (let count = 0; count < 20; count++) if (!(await press('Copy a box'))) run.problems.push('no "Copy a box" button');
                    await sleep(3000);
                    const file = `${scenario.id}-added.png`;
                    await screenshot(file);
                    current = await state();
                    run.shots.push({ file, seconds: current?.motion?.seconds ?? 0, recompiles: current?.recompiles ?? null });
                }

                if (options.measure && scenario.stepMotion) {
                    if (!(await press('Measure a step'))) run.problems.push('no "Measure a step" button');
                    const meterDeadline = Date.now() + 120_000;
                    let meter = '';
                    while (Date.now() < meterDeadline) {
                        await sleep(1000);
                        // The meter's marks line once it is done ("Step 12% of the picture's brightness..."), or its "Stopped:" line.
                        meter = await client.evaluate<string>(`(() => {
                            const texts = [...document.querySelectorAll('p.small')].map((p) => p.textContent ?? '');
                            return texts.find((text) => text.startsWith('Step ')) ?? texts.find((text) => text.startsWith('Stopped:')) ?? '';
                        })()`);
                        if (meter !== '') break;
                    }
                    run.meter = meter || 'no result within 2 minutes';
                    if (!meter.startsWith('Step ')) run.problems.push(`the meter didn't finish: ${run.meter}`);
                    const file = `${scenario.id}-meter.png`;
                    await screenshot(file);
                    current = await state();
                    run.shots.push({ file, seconds: current?.motion?.seconds ?? 0, recompiles: current?.recompiles ?? null });
                }

                await sleep(300);
                current = await state();
                const recompiles = current?.recompiles ?? null;
                if (recompiles === null) run.problems.push('no recompile counter (DDGI not on screen?)');
                else if (recompiles > 0) run.problems.push(`recompiles: ${recompiles} (should stay 0)`);
                for (const error of [...(current?.errors ?? []), ...browserErrors]) run.problems.push(error);
            } catch (error) {
                run.problems.push(error instanceof Error ? error.message : String(error));
            }
            const last = run.shots[run.shots.length - 1];
            console.log(`${scenario.id}: ${run.shots.length} screenshot(s), recompiles ${last?.recompiles ?? '-'}, ${run.problems.length} problem(s), ${((Date.now() - started) / 1000).toFixed(1)} s${run.meter ? `; meter: ${run.meter}` : ''}`);
            for (const problem of run.problems) console.log(`    PROBLEM: ${problem}`);
        }

        await writeFile(join(options.out, 'index.html'), indexHtml(runs, adapter));
        console.log(`\nScreenshots and index.html in ${options.out}`);
        return runs.some((run) => run.problems.length > 0) ? 1 : 0;
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
