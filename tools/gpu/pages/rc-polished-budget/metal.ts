/** Apple Silicon reproduction of the existing RC fixtures. Run from the GI repository with Node 24. */
import assert from 'node:assert/strict';
import { execFile, spawn } from 'node:child_process';
import { createHash } from 'node:crypto';
import { mkdir, mkdtemp, readFile, stat, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { basename, dirname, join, resolve } from 'node:path';
import { setTimeout as sleep } from 'node:timers/promises';
import { parseArgs, promisify } from 'node:util';
import { CdpClient } from '../../cdp.ts';
import { findBrowser, removeProfile, startChrome } from '../../chrome.ts';
import { installShutdownHandlers, registerCleanup } from '../../cleanup.ts';
import type { RunSummary } from '../../contract.ts';
import { withBenchLock } from '../../../reference/bench-lock.ts';
import { appleMetalDevices, budgetPage, checkQuality, compareWindows, correctnessPages, idleWindow, pagesPassed, qualityPage,
    readPair, utilizationCounters, type Cadence } from './metal-evidence.ts';

const root = resolve(import.meta.dirname, '../../../..');
const execute = promisify(execFile);
const { values } = parseArgs({ options: { out: { type: 'string' }, chrome: { type: 'string' }, help: { type: 'boolean' } } });
const usage = 'node tools/gpu/pages/rc-polished-budget/metal.ts [--chrome /path/to/Chrome] [--out new-directory]';
const save = (file: string, value: unknown) => writeFile(file, JSON.stringify(value, null, 2) + '\n');

async function output(file: string, args: string[] = [], timeout = 30000) {
    const { stdout } = await execute(file, args, { cwd: root, encoding: 'utf8', timeout, maxBuffer: 16 * 1024 * 1024 });
    return stdout.trim();
}

async function optionalOutput(file: string, args: string[]) {
    try { return await output(file, args, 2500); } catch { return null; }
}

async function telemetry() {
    const startedAt = new Date().toISOString();
    const [thermal, ioreg] = await Promise.all([
        optionalOutput('/usr/bin/osascript', ['-l', 'JavaScript', '-e',
            'ObjC.import("Foundation"); Number($.NSProcessInfo.processInfo.thermalState)']),
        optionalOutput('/usr/sbin/ioreg', ['-r', '-c', 'IOAccelerator', '-d', '1', '-l', '-w', '0']),
    ]);
    let counters = utilizationCounters(ioreg ?? '');
    if (!counters.length) {
        counters = utilizationCounters(await optionalOutput('/usr/sbin/ioreg', ['-r', '-c', 'AGXAccelerator', '-d', '1', '-l', '-w', '0']) ?? '');
    }
    const levels = ['nominal', 'fair', 'serious', 'critical'];
    return { startedAt, endedAt: new Date().toISOString(), thermalPressure: thermal && /^[0-3]$/.test(thermal) ? levels[Number(thermal)] : null,
        utilizationPercent: counters.length ? counters : null };
}
type Reading = Awaited<ReturnType<typeof telemetry>>;

async function waitForIdle() {
    const started = Date.now();
    const attempts = [];
    for (;;) {
        const readings: Reading[] = [];
        for (let i = 0; i < 5; i++) {
            if (i > 0) await sleep(400);
            readings.push(await telemetry());
            if (Date.now() - started >= 90000) break;
        }
        const assessment = idleWindow(readings);
        attempts.push({ readings, ...assessment });
        const elapsedMs = Date.now() - started;
        if (assessment.verdict !== 'busy-or-elevated' || elapsedMs >= 90000) {
            return { ...assessment, attempts, elapsedMs, waitExhausted: elapsedMs >= 90000 && assessment.verdict !== 'observed-idle-and-nominal' };
        }
        console.log('Waiting for lower Apple GPU utilization or thermal pressure:', assessment);
        await sleep(3000);
    }
}

/** Capture Chrome's GPU-process report using the same executable and flags as the page runner. */
async function chromeGpu(executable: string) {
    const profileDir = await mkdtemp(join(tmpdir(), 'rc-metal-identify-'));
    const chrome = startChrome(executable, { profileDir, luid: null, withoutGpu: false });
    let cdp: CdpClient | undefined;
    const close = registerCleanup(async () => { cdp?.close(); await chrome.stop(); await removeProfile(profileDir); });
    try {
        await chrome.ready();
        const response = await fetch(chrome.devtoolsOrigin + '/json/version', { signal: AbortSignal.timeout(10000) });
        const { webSocketDebuggerUrl } = await response.json() as { webSocketDebuggerUrl: string };
        cdp = await CdpClient.connect(webSocketDebuggerUrl);
        const info = await cdp.send<{ gpu: { devices: Array<{ vendorId: number; deviceString: string; vendorString: string }>;
            auxAttributes: Record<string, string>; featureStatus: Record<string, string> } }>('SystemInfo.getInfo');
        const gpu = info.gpu;
        const version = await cdp.send<{ product: string; revision: string }>('Browser.getVersion');
        return { product: version.product, revision: version.revision, devices: gpu.devices,
            glRenderer: gpu.auxAttributes.glRenderer, glImplementation: gpu.auxAttributes.glImplementationParts,
            featureStatus: gpu.featureStatus };
    } finally { await close(); }
}

async function runNode(args: string[], env = process.env) {
    const child = spawn(process.execPath, args, { cwd: root, stdio: 'inherit', env });
    const ended = new Promise<void>((done, fail) => {
        child.once('error', fail);
        child.once('close', (code, signal) => code === 0 ? done() : fail(new Error(args[0] + ' exited ' + (signal ?? code))));
    });
    const cleanup = registerCleanup(async () => {
        if (child.exitCode === null && child.signalCode === null) child.kill('SIGTERM');
        await ended.catch(() => {});
    });
    try { await ended; } finally { await cleanup(); }
}

async function suite(out: string) {
    const executable = findBrowser(values.chrome ?? null);
    const profile = JSON.parse(await output('/usr/sbin/system_profiler', ['SPDisplaysDataType', '-json']));
    const host = { startedAt: new Date().toISOString(), node: process.version, arch: process.arch,
        macOS: await output('/usr/bin/sw_vers'), machine: await output('/usr/sbin/sysctl', ['-n', 'hw.model']),
        arm64: await output('/usr/sbin/sysctl', ['-n', 'hw.optional.arm64']), gpu: appleMetalDevices(profile),
        power: await optionalOutput('/usr/bin/pmset', ['-g', 'ps']), chrome: await chromeGpu(executable),
        gitHead: await output('git', ['rev-parse', 'HEAD']), sourceChanges: await output('git',
            ['status', '--short', '--', 'examples', 'tools/gpu', 'tools/reference', 'packages', 'pnpm-lock.yaml', 'package.json']),
        lockfileSha256: createHash('sha256').update(await readFile(join(root, 'pnpm-lock.yaml'))).digest('hex') };
    await save(join(out, 'host.json'), host);
    assert.equal(host.arm64, '1', 'This runner requires Apple Silicon');
    assert(host.chrome.devices.some(gpu => gpu.vendorId === 0x106b) && /Metal/i.test(host.chrome.glRenderer ?? '') &&
        !/swiftshader|software/i.test(host.chrome.glRenderer), 'Chrome GPU process did not identify an Apple GPU with a Metal renderer');

    await runNode(['node_modules/vite/bin/vite.js', 'build', '--config', 'examples/vite.config.ts'], { ...process.env, PAGES_BASE: '/gi/' });
    assert((await readFile(join(root, 'examples/dist/cascades/11-radiance-cascades/index.html'), 'utf8')).includes('src="/gi/assets/'),
        'gallery build is missing the /gi/ base');
    async function fixture(name: string, filters: string[], pair?: number, cadence: Cadence = 'normal') {
        const reportDir = join(out, name);
        await mkdir(reportDir);
        const file = join(reportDir, 'run.json');
        let harnessError: unknown;
        try {
            await runNode(['tools/gpu/run.ts', '--chrome', executable, '--filter', filters.join(','), '--out', reportDir, '--json', file],
                { ...process.env, VITE_RC_REFLECTION_PAIR: pair === undefined ? '' : String(pair), VITE_RC_REFLECTION_CADENCE: cadence });
        } catch (error) { harnessError = error; }
        const raw = await readFile(file, 'utf8').catch(error => { throw harnessError ?? error; });
        const report = JSON.parse(raw) as RunSummary;
        // The general harness records absolute screenshot paths. Keep the exported evidence movable as a directory.
        for (const page of report.pages) {
            if (page.screenshot) page.screenshot = basename(page.screenshot);
        }
        await save(file, report);
        if (harnessError) throw harnessError;
        for (const page of report.pages) {
            assert(page.screenshot, page.name + ' has no retained screenshot');
            assert((await stat(join(reportDir, page.screenshot))).size > 0, page.screenshot + ' is empty');
        }
        return report;
    }
    const correctness = pagesPassed(await fixture('correctness', correctnessPages), correctnessPages).map(p => p.name);
    const quality = checkQuality(await fixture('quality', [qualityPage]));
    async function measureCadence(cadence: Cadence) {
        const prefix = cadence === 'normal' ? 'pair-' : 'uncapped-pair-';
        const pairs = [];
        const conditions: Array<{ preflight: ReturnType<typeof idleWindow> & { elapsedMs: number; waitExhausted: boolean };
            before: Reading; after: Reading; sampleCount: number;
            thermalPressureObserved: string[]; samplesWithUtilization: number }> = [];
        for (const pair of [1, 2, 3]) {
            const name = prefix + pair;
            const { attempts, ...preflight } = await waitForIdle();
            const before = attempts.at(-1)!.readings.at(-1)!;
            await save(join(out, name + '-preflight.json'), { ...preflight, attempts });
            const during: Reading[] = [];
            let sampling: Promise<void> | undefined;
            const interval = setInterval(() => {
                if (!sampling) sampling = telemetry().then(reading => { during.push(reading); }).finally(() => { sampling = undefined; });
            }, 3000);
            let pairData;
            try {
                pairData = readPair(await fixture(name, [budgetPage], pair, cadence), pair, cadence);
            } finally {
                clearInterval(interval);
                await sampling;
                const after = await telemetry();
                await save(join(out, name + '-telemetry.json'), { before, during, after });
                const readings = [before, ...during, after];
                conditions.push({ preflight, before, after, sampleCount: readings.length,
                    thermalPressureObserved: [...new Set(readings.map(sample => sample.thermalPressure).filter(value => value !== null))],
                    samplesWithUtilization: readings.filter(sample => sample.utilizationPercent !== null).length });
            }
            pairs.push(pairData);
        }
        const matched = compareWindows(pairs.flatMap(pair => pair.on.samplesMs), pairs.flatMap(pair => pair.off.samplesMs));
        const sustained = cadence === 'uncapped' ? compareWindows(pairs.flatMap(pair => pair.on.sustained!.samplesMs),
            pairs.flatMap(pair => pair.off.sustained!.samplesMs)) : undefined;
        return { cadence, endedAt: new Date().toISOString(),
            allPreflightsObservedIdleAndNominal: conditions.every(c => c.preflight.verdict === 'observed-idle-and-nominal'),
            thermalQualification: 'unassessed', on: matched.on, off: matched.off, pooledMedianDeltaMs: matched.medianDeltaMs,
            pooledP95DeltaMs: matched.p95DeltaMs, sustained, pairs: pairs.map((pair, index) => {
                const name = prefix + pair.pair;
                return { pair: pair.pair, order: pair.order, ...compareWindows(pair.on.samplesMs, pair.off.samplesMs),
                    sustained: cadence === 'uncapped' ? compareWindows(pair.on.sustained!.samplesMs, pair.off.sustained!.samplesMs) : undefined,
                    floor: pair.floor, report: name + '/run.json', conditions: conditions[index],
                    preflight: name + '-preflight.json', telemetry: name + '-telemetry.json' };
            }) };
    }
    const normal = await measureCadence('normal');
    await save(join(out, 'summary.json'), { ok: true, host, correctness, quality, ...normal });
    console.log('Normal-cadence GPU milliseconds:', { on: normal.on, off: normal.off });
    const uncapped = await measureCadence('uncapped');
    await save(join(out, 'uncapped-summary.json'), { ok: true, host, ...uncapped });
    console.log('Uncapped sustained GPU milliseconds:', uncapped.sustained);
    console.log('Completed both cadences. Evidence:', out);
}

if (values.help) console.log(usage);
else if (process.platform !== 'darwin') { console.error('Apple Silicon macOS is required. ' + usage); process.exitCode = 2; }
else {
    installShutdownHandlers();
    const out = resolve(values.out ?? join(root, 'bench/screenshots/rc-metal-' + new Date().toISOString().replace(/[:.]/g, '-')));
    await mkdir(dirname(out), { recursive: true });
    await mkdir(out); // Fail on an existing return bundle rather than mixing two machines or revisions.
    try {
        const reservation = { what: 'Apple Metal RC correctness, polished quality and six timing pairs', command: usage };
        // Both platforms reserve GPU first, then CPU, to keep queue submission clear of other benchmarks.
        await withBenchLock(() => withBenchLock(() => suite(out), { ...reservation, lock: 'cpu' }), reservation);
    } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        await save(join(out, 'failure.json'), { ok: false, at: new Date().toISOString(), message });
        console.error(message, '\nPartial evidence:', out);
        process.exitCode = 1;
    }
}
