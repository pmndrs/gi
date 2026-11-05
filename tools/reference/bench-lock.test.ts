import { existsSync } from 'node:fs';
import { mkdir, mkdtemp, readFile, rm, utimes, writeFile } from 'node:fs/promises';
import { hostname, tmpdir } from 'node:os';
import { basename, join } from 'node:path';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import {
    CLOCK_FILE_ENV,
    clockLowWarning,
    clockNoteRows,
    isClockLow,
    LOW_CLOCK_MHZ,
    parseClockLine,
    readClockSummary,
    startHotWarning,
    summarizeClockSamples,
} from '../gpu/clock-note.ts';
import {
    acquireBenchLock,
    benchLockCli,
    benchLockPath,
    BenchLockTimeoutError,
    cpuLockPath,
    defaultLockPath,
    judgeLock,
    LOCK_HELD_ENV,
    lockHeldByParent,
    OWNER_FILE,
    readLockState,
    releaseBenchLock,
    parseHeldPids,
    startClockSampler,
    systemProbe,
    waitForCoolGpu,
    withBenchLock,
    type BenchLockCliDeps,
    type BenchLockOwner,
    type ProcessProbe,
} from './bench-lock.ts';

let folder: string;
let lockPath: string;

beforeEach(async () => {
    folder = await mkdtemp(join(tmpdir(), 'ddgi-bench-lock-test-'));
    lockPath = join(folder, 'test.lock');
});

afterEach(async () => {
    await rm(folder, { recursive: true, force: true });
});

/** A pretend OS: `processes` maps a pid to its start time; anything else is dead. */
function fakeProbe(processes: Record<number, string>): ProcessProbe {
    return {
        isAlive: (pid) => pid in processes,
        startTime: async (pid) => processes[pid] ?? null,
    };
}

function foreignOwner(overrides: Partial<BenchLockOwner> = {}): BenchLockOwner {
    return {
        pid: 4242,
        processStart: 'started-at-noon',
        host: hostname(),
        startedAt: '2026-11-04T12:00:00.000Z',
        what: 'someone else',
        command: 'node something.ts',
        ...overrides,
    };
}

async function placeLock(owner: BenchLockOwner | null): Promise<void> {
    await mkdir(lockPath);
    if (owner) await writeFile(join(lockPath, OWNER_FILE), JSON.stringify(owner));
}

async function ownerOnDisk(): Promise<BenchLockOwner> {
    return JSON.parse(await readFile(join(lockPath, OWNER_FILE), 'utf8')) as BenchLockOwner;
}

const quiet = { log: () => undefined, pollMs: 5, reportEveryMs: 0 };

describe('bench lock', () => {
    it('takes a free lock, writes who holds it, and releases it', async () => {
        const handle = await acquireBenchLock({ ...quiet, path: lockPath, what: 'unit test', probe: fakeProbe({ [process.pid]: 'me' }) });
        const owner = await ownerOnDisk();
        expect(owner).toEqual(handle.owner);
        expect(owner.pid).toBe(process.pid);
        expect(owner.processStart).toBe('me');
        expect(owner.host).toBe(hostname());
        expect(owner.what).toBe('unit test');
        expect(Number.isNaN(Date.parse(owner.startedAt))).toBe(false);
        expect(await handle.release()).toBe('released');
        expect(existsSync(lockPath)).toBe(false);
    });

    it('never takes a lock whose owner is running, and gives up at the timeout', async () => {
        const owner = foreignOwner();
        await placeLock(owner);
        const messages: string[] = [];
        const attempt = acquireBenchLock({
            ...quiet,
            log: (message) => messages.push(message),
            path: lockPath,
            timeoutMs: 60,
            probe: fakeProbe({ 4242: 'started-at-noon' }),
        });
        await expect(attempt).rejects.toBeInstanceOf(BenchLockTimeoutError);
        expect(await ownerOnDisk()).toEqual(owner);
        // The waiting message says who holds it.
        expect(messages.some((message) => message.includes('pid 4242') && message.includes('someone else'))).toBe(true);
    });

    it('takes over a lock whose owner process is gone', async () => {
        await placeLock(foreignOwner());
        const handle = await acquireBenchLock({ ...quiet, path: lockPath, timeoutMs: 1000, probe: fakeProbe({ [process.pid]: 'me' }) });
        expect((await ownerOnDisk()).pid).toBe(process.pid);
        await handle.release();
    });

    it('takes over a lock whose pid now belongs to a different process', async () => {
        await placeLock(foreignOwner());
        const probe = fakeProbe({ 4242: 'started-later', [process.pid]: 'me' });
        expect((await judgeLock(await readLockState(lockPath), { probe })).stale).toBe(true);
        const handle = await acquireBenchLock({ ...quiet, path: lockPath, timeoutMs: 1000, probe });
        expect((await ownerOnDisk()).pid).toBe(process.pid);
        await handle.release();
    });

    it('keeps a live owner when its start time cannot be read', async () => {
        await placeLock(foreignOwner());
        const probe: ProcessProbe = { isAlive: () => true, startTime: async () => null };
        expect((await judgeLock(await readLockState(lockPath), { probe })).stale).toBe(false);
    });

    it('takes over an old folder with no owner file, but waits on a young one', async () => {
        await placeLock(null);
        const probe = fakeProbe({ [process.pid]: 'me' });
        await expect(acquireBenchLock({ ...quiet, path: lockPath, timeoutMs: 60, ownerlessStaleMs: 5 * 60_000, probe })).rejects.toBeInstanceOf(
            BenchLockTimeoutError,
        );
        expect(existsSync(join(lockPath, OWNER_FILE))).toBe(false);

        const tenMinutesAgo = new Date(Date.now() - 10 * 60_000);
        await utimes(lockPath, tenMinutesAgo, tenMinutesAgo);
        const handle = await acquireBenchLock({ ...quiet, path: lockPath, timeoutMs: 1000, ownerlessStaleMs: 5 * 60_000, probe });
        expect((await ownerOnDisk()).pid).toBe(process.pid);
        await handle.release();
    });

    it('release refuses a lock that someone else owns', async () => {
        const handle = await acquireBenchLock({ ...quiet, path: lockPath, probe: fakeProbe({ [process.pid]: 'me' }) });
        // Someone replaced the lock (this should never happen to a live owner, but release must not trust it).
        await rm(lockPath, { recursive: true });
        const other = foreignOwner();
        await placeLock(other);
        expect(await handle.release()).toBe('not-ours');
        expect(await ownerOnDisk()).toEqual(other);

        expect(await releaseBenchLock((owner) => owner.pid === process.pid, lockPath)).toBe('not-ours');
        expect(existsSync(lockPath)).toBe(true);
    });

    it('release never removes a folder without an owner file', async () => {
        await placeLock(null);
        expect(await releaseBenchLock(() => true, lockPath)).toBe('not-ours');
        expect(existsSync(lockPath)).toBe(true);
    });
});

describe('bench lock command line', () => {
    it('acquires for a given pid, refuses a release by another pid, then releases', async () => {
        const silence = { log: console.log, error: console.error };
        console.log = () => undefined;
        console.error = () => undefined;
        try {
            expect(await benchLockCli(['status', '--path', lockPath])).toBe(0);
            expect(await benchLockCli(['acquire', '--pid', String(process.pid), '--what', 'cli test', '--path', lockPath])).toBe(0);
            expect((await ownerOnDisk()).what).toBe('cli test');
            expect(await benchLockCli(['status', '--path', lockPath])).toBe(3);
            expect(await benchLockCli(['release', '--pid', String(process.pid + 1), '--path', lockPath])).toBe(2);
            expect(existsSync(lockPath)).toBe(true);
            expect(await benchLockCli(['release', '--pid', String(process.pid), '--path', lockPath])).toBe(0);
            expect(existsSync(lockPath)).toBe(false);
        } finally {
            console.log = silence.log;
            console.error = silence.error;
        }
    }, 30_000);

    it('refuses acquire without --pid (its own pid dies at once)', async () => {
        await expect(benchLockCli(['acquire', '--path', lockPath])).rejects.toThrow(/--pid/);
    });
});

describe('a lock held by the parent (LOCK_HELD_ENV)', () => {
    const saved = process.env[LOCK_HELD_ENV];
    afterEach(() => {
        if (saved === undefined) delete process.env[LOCK_HELD_ENV];
        else process.env[LOCK_HELD_ENV] = saved;
    });

    it("runs the work under the parent's hold without waiting, and leaves the lock to the parent", async () => {
        const parent = foreignOwner({ what: 'bench-lock run' });
        await placeLock(parent);
        process.env[LOCK_HELD_ENV] = String(parent.pid);
        const probe = fakeProbe({ 4242: 'started-at-noon', [process.pid]: 'me' });
        // Without the parent check this would wait on pid 4242 and time out.
        const seen = await withBenchLock(async ({ ownerPid }) => ownerPid, { ...quiet, path: lockPath, timeoutMs: 60, probe });
        expect(seen).toBe(4242);
        expect(await ownerOnDisk()).toEqual(parent);
    });

    it('ignores the variable when that pid does not own the lock, or is not running', async () => {
        const probe = fakeProbe({ 4242: 'started-at-noon', [process.pid]: 'me' });
        process.env[LOCK_HELD_ENV] = '4242';
        // Free lock: the variable is leftover, so the lock is taken normally (and released after).
        expect(await lockHeldByParent({ path: lockPath, probe })).toBeNull();
        const seen = await withBenchLock(async ({ ownerPid }) => {
            expect((await ownerOnDisk()).pid).toBe(process.pid);
            return ownerPid;
        }, { ...quiet, path: lockPath, probe });
        expect(seen).toBe(process.pid);
        expect(existsSync(lockPath)).toBe(false);

        // Someone else owns it: no skipping.
        await placeLock(foreignOwner({ pid: 5555 }));
        expect(await lockHeldByParent({ path: lockPath, probe: fakeProbe({ 5555: 'x' }) })).toBeNull();
        await expect(withBenchLock(async () => 0, { ...quiet, path: lockPath, timeoutMs: 60, probe: fakeProbe({ 5555: 'started-at-noon' }) })).rejects.toBeInstanceOf(
            BenchLockTimeoutError,
        );
        await rm(lockPath, { recursive: true });

        // The named pid owns it on paper but is gone.
        await placeLock(foreignOwner());
        expect(await lockHeldByParent({ path: lockPath, probe: fakeProbe({}) })).toBeNull();
    });

    it('run gives its command the owner pid, and a nested run under it does not deadlock', async () => {
        delete process.env[LOCK_HELD_ENV];
        const out = join(folder, 'env.txt');
        const script = join(folder, 'print-env.mjs');
        await writeFile(script, `import { writeFileSync } from 'node:fs';
writeFileSync(${JSON.stringify(out)}, process.env.${LOCK_HELD_ENV} ?? 'unset');
`);
        const silence = { log: console.log, error: console.error };
        console.log = () => undefined;
        console.error = () => undefined;
        try {
            expect(await benchLockCli(['run', '--what', 'outer', '--path', lockPath, '--', 'node', script])).toBe(0);
            expect(await readFile(out, 'utf8')).toBe(String(process.pid));
            expect(existsSync(lockPath)).toBe(false);

            // Nested: hold the lock as this process, mark it as the parent's, and run again.
            const handle = await acquireBenchLock({ ...quiet, path: lockPath });
            process.env[LOCK_HELD_ENV] = String(process.pid);
            expect(await benchLockCli(['run', '--what', 'inner', '--timeout-minutes', '0.05', '--path', lockPath, '--', 'node', script])).toBe(0);
            expect(await readFile(out, 'utf8')).toBe(String(process.pid));
            // The inner run did not release the outer hold.
            expect((await ownerOnDisk()).pid).toBe(process.pid);
            expect(await handle.release()).toBe('released');
        } finally {
            console.log = silence.log;
            console.error = silence.error;
        }
    }, 60_000);
});

describe('system probe', () => {
    it('sees this process alive with a stable start time, and a dead pid as dead', async () => {
        expect(systemProbe.isAlive(process.pid)).toBe(true);
        const first = await systemProbe.startTime(process.pid);
        expect(first).not.toBeNull();
        expect(await systemProbe.startTime(process.pid)).toBe(first);
        // Pids are multiples of 4 on Windows and small elsewhere; this one is not in use.
        expect(systemProbe.isAlive(2 ** 30 + 1)).toBe(false);
    }, 30_000);
});

describe('GPU clock sampling', () => {
    it('parses the nvidia-smi line and ignores anything else', () => {
        expect(parseClockLine('1215, 6001, 41\r\n')).toEqual({ smMhz: 1215, memMhz: 6001, utilizationPercent: 41 });
        expect(parseClockLine('1215, 6001, [N/A]')).toBeNull();
        expect(parseClockLine('')).toBeNull();
        // With temperature and the throttle-reason mask (hex), as the real query returns them.
        expect(parseClockLine('585, 7151, 93, 89, 0x0000000000000020')).toEqual({ smMhz: 585, memMhz: 7151, utilizationPercent: 93, temperatureC: 89, throttleReasons: 0x20 });
        expect(parseClockLine('585, 7151, 93, 89')).toBeNull();
    });

    it('summarizes min / median / max, and takes the headline from samples under load', () => {
        const idle = { smMhz: 210, memMhz: 810, utilizationPercent: 2 };
        const busy = (smMhz: number) => ({ smMhz, memMhz: 6001, utilizationPercent: 98 });
        const summary = summarizeClockSamples([idle, idle, busy(1800), busy(2100), busy(1950), busy(2100)])!;
        expect(summary.samples).toBe(6);
        expect(summary.loadedSamples).toBe(4);
        expect(summary.smMhz).toEqual({ min: 210, median: 1875, max: 2100 });
        expect(summary.smMhzLoaded).toEqual({ min: 1800, median: 2025, max: 2100 });
        expect(clockNoteRows(summary).find((row) => row.metric === 'note/sm_clock_mhz')?.value).toBe(2025);
        // Too few loaded samples to trust: the headline falls back to every sample.
        const mostlyIdle = summarizeClockSamples([idle, idle, idle, busy(2100)])!;
        expect(clockNoteRows(mostlyIdle).find((row) => row.metric === 'note/sm_clock_mhz')?.value).toBe(210);
        expect(summarizeClockSamples([])).toBeNull();
    });

    describe('the low-clock flag', () => {
        const idle = { smMhz: 210, memMhz: 810, utilizationPercent: 2 };
        const busy = (smMhz: number, utilizationPercent = 98) => ({ smMhz, memMhz: 6001, utilizationPercent });
        const noteNames = (samples: Parameters<typeof summarizeClockSamples>[0]) => clockNoteRows(summarizeClockSamples(samples)!).map((row) => row.metric);

        it('flags a loaded median under 1,500 MHz, and writes note/clock_low,1', () => {
            expect(LOW_CLOCK_MHZ).toBe(1500);
            const low = summarizeClockSamples([busy(590), busy(600), busy(1400), busy(580), busy(610)])!;
            expect(isClockLow(low)).toBe(true);
            expect(clockNoteRows(low).find((row) => row.metric === 'note/clock_low')?.value).toBe(1);
            expect(clockLowWarning(low)).toContain('WARNING');
            expect(clockLowWarning(low)).toContain('600 MHz');
        });

        it('does not flag a healthy run, and the limit itself is healthy', () => {
            const healthy = summarizeClockSamples([busy(1600), busy(1650), busy(1500), busy(1700)])!;
            expect(isClockLow(healthy)).toBe(false);
            expect(clockLowWarning(healthy)).toBeNull();
            expect(noteNames([busy(1500), busy(1500), busy(1500)])).not.toContain('note/clock_low');
            expect(noteNames([busy(1499), busy(1499), busy(1499)])).toContain('note/clock_low');
        });

        it('counts only samples at 70% use or more: idle gaps do not make a healthy run low', () => {
            const withGaps = summarizeClockSamples([idle, idle, idle, idle, idle, busy(1700), busy(1650), busy(1600), busy(1700), busy(1500, 69)])!;
            expect(withGaps.smMhz.median).toBeLessThan(LOW_CLOCK_MHZ);
            expect(withGaps.loadedSamples).toBe(4);
            expect(isClockLow(withGaps)).toBe(false);
            // Busy samples below the threshold are not "under load" either, however low their clock.
            expect(isClockLow(summarizeClockSamples([busy(1700), busy(1700), busy(1700), busy(300, 69), busy(300, 69), busy(300, 69)])!)).toBe(false);
        });

        it('gives no verdict from fewer than three loaded samples (all-sample fallback would read low)', () => {
            const mostlyIdle = summarizeClockSamples([idle, idle, idle, busy(900), busy(900)])!;
            expect(clockNoteRows(mostlyIdle).find((row) => row.metric === 'note/sm_clock_mhz')?.value).toBe(210);
            expect(isClockLow(mostlyIdle)).toBe(false);
            expect(noteNames([idle, idle, idle])).not.toContain('note/clock_low');
        });

        it('says why when nvidia-smi named a reason: heat or the power limit', () => {
            const hot = (smMhz: number, throttleReasons: number) => ({ smMhz, memMhz: 7151, utilizationPercent: 95, temperatureC: 88, throttleReasons });
            const summary = summarizeClockSamples([hot(500, 0x20), hot(550, 0x20), hot(600, 0x20), hot(700, 0x4)])!;
            expect(summary.thermalLimitedShare).toBe(0.75);
            expect(summary.powerCappedShare).toBe(0.25);
            expect(summary.temperatureCLoaded?.median).toBe(88);
            const warning = clockLowWarning(summary)!;
            expect(warning).toContain('thermal throttling in 75%');
            expect(warning).toContain('power limit in 25%');
            expect(warning).toContain('88 C');
            // Three-field samples carry no reason: the warning is still given, without a cause.
            expect(clockLowWarning(summarizeClockSamples([busy(500), busy(500), busy(500)])!)).not.toContain('nvidia-smi: thermal');
        });

        it('bench-lock run prints the warning at the end of a low-clock run', async () => {
            const file = join(folder, 'low-clock.json');
            const printed: string[] = [];
            const original = { log: console.log, error: console.error };
            console.log = () => undefined;
            console.error = (message: string) => void printed.push(String(message));
            try {
                delete process.env[LOCK_HELD_ENV];
                const code = await benchLockCli(['run', '--path', lockPath, '--clock-file', file, '--', 'node', '-e', 'setTimeout(() => {}, 150)'], { readClock: async () => '600, 6001, 95', clockIntervalMs: 10 });
                expect(code).toBe(0);
            } finally {
                console.log = original.log;
                console.error = original.error;
            }
            expect(printed.some((line) => line.includes('WARNING') && line.includes('600 MHz'))).toBe(true);
        });
    });

    it('samples while running, keeps a readable file, and does nothing without nvidia-smi', async () => {
        const file = join(folder, 'clock.json');
        let calls = 0;
        const sampler = await startClockSampler({ file, intervalMs: 5, read: async () => `${1000 + 100 * calls++}, 6001, 90` });
        expect(sampler).not.toBeNull();
        await new Promise((done) => setTimeout(done, 60));
        // Readable while still running.
        expect(readClockSummary(file)?.samples).toBeGreaterThanOrEqual(2);
        const summary = (await sampler!.stop())!;
        expect(summary.smMhz.min).toBe(1000);
        expect(summary.smMhz.max).toBeGreaterThan(1000);
        expect(readClockSummary(file)?.samples).toBe(summary.samples);

        expect(await startClockSampler({ file: join(folder, 'none.json'), read: async () => null })).toBeNull();
        expect(existsSync(join(folder, 'none.json'))).toBe(false);
    });

    it('run writes the clock file, prints the range and passes the file to its command', async () => {
        const file = join(folder, 'run-clock.json');
        const out = join(folder, 'clock-env.txt');
        const script = join(folder, 'print-clock-env.mjs');
        await writeFile(script, `import { writeFileSync } from 'node:fs';
writeFileSync(${JSON.stringify(out)}, process.env.${CLOCK_FILE_ENV} ?? 'unset');
`);
        const silence = { log: console.log, error: console.error };
        const printed: string[] = [];
        console.log = () => undefined;
        console.error = (message: string) => void printed.push(String(message));
        try {
            delete process.env[LOCK_HELD_ENV];
            const code = await benchLockCli(['run', '--path', lockPath, '--clock-file', file, '--', 'node', script], { readClock: async () => '1500, 6001, 80', clockIntervalMs: 10 });
            expect(code).toBe(0);
        } finally {
            console.log = silence.log;
            console.error = silence.error;
        }
        expect(await readFile(out, 'utf8')).toBe(file);
        expect(readClockSummary(file)?.smMhz).toEqual({ min: 1500, median: 1500, max: 1500 });
        expect(printed.some((line) => line.includes('GPU clock') && line.includes('median 1500'))).toBe(true);

        // No nvidia-smi: the command runs, no file is written and the variable is not set.
        await rm(file);
        console.error = () => undefined;
        try {
            expect(await benchLockCli(['run', '--path', lockPath, '--clock-file', file, '--', 'node', script], { readClock: async () => null })).toBe(0);
        } finally {
            console.error = silence.error;
        }
        expect(await readFile(out, 'utf8')).toBe('unset');
        expect(existsSync(file)).toBe(false);
    }, 60_000);
});

/** Runs the command line with console output captured; `printed` is stderr, where `run` logs. */
async function runCliCaptured(args: string[], deps: BenchLockCliDeps = {}): Promise<{ code: number; printed: string[]; stdout: string[] }> {
    const original = { log: console.log, error: console.error };
    const printed: string[] = [];
    const stdout: string[] = [];
    console.log = (message: string) => void stdout.push(String(message));
    console.error = (message: string) => void printed.push(String(message));
    try {
        return { code: await benchLockCli(args, deps), printed, stdout };
    } finally {
        console.log = original.log;
        console.error = original.error;
    }
}

/**
 * A pretend nvidia-smi: one five-field line per call, with the temperatures in turn (the last one
 * repeats). Records when each temperature was read.
 */
function fakeSmi(temperatures: number[]): { read: () => Promise<string>; reads: Array<{ temperatureC: number; at: number }> } {
    const reads: Array<{ temperatureC: number; at: number }> = [];
    return {
        reads,
        read: async () => {
            const temperatureC = temperatures[Math.min(reads.length, temperatures.length - 1)];
            reads.push({ temperatureC, at: Date.now() });
            return `1500, 6001, 90, ${temperatureC}, 0x0000000000000000`;
        },
    };
}

let probeScripts = 0;

/** A script that writes the time it started, which lock folders exist and LOCK_HELD_ENV to `out`. */
async function writeProbeScript(out: string, paths: Record<string, string>): Promise<string> {
    const script = join(folder, `probe-${probeScripts++}.mjs`);
    await writeFile(
        script,
        `import { existsSync, writeFileSync } from 'node:fs';
const paths = ${JSON.stringify(paths)};
const exists = Object.fromEntries(Object.entries(paths).map(([name, path]) => [name, existsSync(path)]));
writeFileSync(${JSON.stringify(out)}, JSON.stringify({ startedAt: Date.now(), exists, held: process.env.${LOCK_HELD_ENV} ?? null }));
`,
    );
    return script;
}

describe('cool-down wait in bench-lock run', () => {
    beforeEach(() => {
        delete process.env[LOCK_HELD_ENV];
    });

    it('waits until the GPU is below 70 C, prints the temperature at the start and when cool, and records it', async () => {
        const smi = fakeSmi([80, 76, 72, 69]);
        const file = join(folder, 'cool.json');
        const out = join(folder, 'cool-out.json');
        const script = await writeProbeScript(out, {});
        const { code, printed } = await runCliCaptured(['run', '--path', lockPath, '--clock-file', file, '--', 'node', script], {
            readClock: smi.read,
            clockIntervalMs: 10,
            coolPollMs: 5,
        });
        expect(code).toBe(0);
        expect(smi.reads.slice(0, 4).map((read) => read.temperatureC)).toEqual([80, 76, 72, 69]);
        // The command started only after the reading under 70 C.
        const started = JSON.parse(await readFile(out, 'utf8')) as { startedAt: number };
        expect(started.startedAt).toBeGreaterThanOrEqual(smi.reads[3].at);
        // One line when it starts waiting and one when it is cool; nothing per poll.
        const temperatureLines = printed.filter((line) => line.startsWith('GPU at'));
        expect(temperatureLines).toHaveLength(2);
        expect(temperatureLines[0]).toContain('80 C: waiting until it is below 70 C');
        expect(temperatureLines[1]).toMatch(/^GPU at 69 C after .*: starting$/);
        const summary = readClockSummary(file)!;
        expect(summary.start).toMatchObject({ temperatureC: 69, coolBelowC: 70, capped: false });
        const rows = clockNoteRows(summary);
        expect(rows.find((row) => row.metric === 'note/start_temp_c')?.value).toBe(69);
        expect(rows.map((row) => row.metric)).not.toContain('note/start_hot');
        expect(existsSync(lockPath)).toBe(false);
    }, 30_000);

    it('starts at once when the GPU is already cool, and --cool-below changes the limit', async () => {
        const cool = await runCliCaptured(['run', '--path', lockPath, '--', 'node', '-e', '0'], { readClock: fakeSmi([55]).read, coolPollMs: 5 });
        expect(cool.code).toBe(0);
        expect(cool.printed).toContain('GPU at 55 C, below 70 C: starting');

        const smi = fakeSmi([66, 64]);
        const lower = await runCliCaptured(['run', '--path', lockPath, '--cool-below', '65', '--', 'node', '-e', '0'], { readClock: smi.read, coolPollMs: 5 });
        expect(lower.code).toBe(0);
        expect(smi.reads.slice(0, 2).map((read) => read.temperatureC)).toEqual([66, 64]);
        expect(lower.printed.some((line) => line.includes('below 65 C'))).toBe(true);
    }, 30_000);

    it('runs anyway at the time cap, with a warning, and the rows flag the hot start', async () => {
        const file = join(folder, 'capped.json');
        const { code, printed } = await runCliCaptured(
            ['run', '--path', lockPath, '--clock-file', file, '--cool-wait-max-minutes', '0.001', '--', 'node', '-e', '0'],
            { readClock: fakeSmi([80]).read, clockIntervalMs: 10, coolPollMs: 5 },
        );
        expect(code).toBe(0);
        expect(printed.some((line) => line.startsWith('WARNING: the GPU is still at 80 C') && line.includes('starting anyway'))).toBe(true);
        const summary = readClockSummary(file)!;
        expect(summary.start).toMatchObject({ temperatureC: 80, coolBelowC: 70, capped: true });
        expect(clockNoteRows(summary).find((row) => row.metric === 'note/start_hot')?.value).toBe(1);
        expect(startHotWarning(summary)).toContain('gave up');
    }, 30_000);

    it('--no-cool-wait starts a hot GPU at once, but still records the start temperature', async () => {
        const file = join(folder, 'no-wait.json');
        // With the default 15-minute cap, a wait here would time the test out.
        const { code, printed } = await runCliCaptured(['run', '--no-cool-wait', '--path', lockPath, '--clock-file', file, '--', 'node', '-e', '0'], {
            readClock: fakeSmi([85]).read,
            clockIntervalMs: 10,
        });
        expect(code).toBe(0);
        expect(printed).toContain('GPU at 85 C (no cool-down wait)');
        expect(printed.some((line) => line.includes('waiting until'))).toBe(false);
        const summary = readClockSummary(file)!;
        expect(summary.start).toMatchObject({ temperatureC: 85, coolBelowC: null, capped: false });
        expect(clockNoteRows(summary).find((row) => row.metric === 'note/start_hot')?.value).toBe(1);
    }, 30_000);

    it('skips the wait without nvidia-smi, or without a temperature in its line', async () => {
        const messages: string[] = [];
        expect(await waitForCoolGpu({ read: async () => null, belowC: 70, log: (message) => messages.push(message) })).toBeNull();
        expect(messages).toEqual(['no GPU temperature from nvidia-smi: no cool-down wait']);
        expect(await waitForCoolGpu({ read: async () => '1500, 6001, 90', belowC: 70, log: () => undefined })).toBeNull();
    });
});

describe('the CPU lock', () => {
    let gpuPath: string;
    let cpuPath: string;
    beforeEach(() => {
        delete process.env[LOCK_HELD_ENV];
        gpuPath = join(folder, 'gpu.lock');
        cpuPath = join(folder, 'cpu.lock');
    });

    it('is its own folder next to the GPU lock', () => {
        expect(basename(benchLockPath)).toBe('ddgi-gpu-bench.lock');
        expect(basename(cpuLockPath)).toBe('ddgi-cpu-heavy.lock');
        expect(defaultLockPath('gpu')).toBe(benchLockPath);
        expect(defaultLockPath('cpu')).toBe(cpuLockPath);
    });

    it('run --lock cpu takes only the CPU lock, while the GPU lock is held, and never waits for a hot GPU', async () => {
        const gpuHolder = await acquireBenchLock({ ...quiet, path: gpuPath });
        const out = join(folder, 'cpu-out.json');
        const script = await writeProbeScript(out, { gpu: gpuPath, cpu: cpuPath });
        const { code, printed } = await runCliCaptured(
            ['run', '--lock', 'cpu', '--gpu-path', gpuPath, '--cpu-path', cpuPath, '--timeout-minutes', '0.5', '--clock-file', join(folder, 'cpu-clock.json'), '--', 'node', script],
            { readClock: fakeSmi([90]).read, clockIntervalMs: 10 },
        );
        expect(code).toBe(0);
        const seen = JSON.parse(await readFile(out, 'utf8')) as { exists: Record<string, boolean>; held: string };
        expect(seen.exists).toEqual({ gpu: true, cpu: true });
        expect(seen.held).toBe(String(process.pid));
        expect(printed.some((line) => line.includes('waiting until'))).toBe(false);
        // The CPU lock is released; the GPU lock is still the other holder's.
        expect(existsSync(cpuPath)).toBe(false);
        expect(await gpuHolder.release()).toBe('released');
    }, 30_000);

    it('run --cpu-timed takes the GPU lock first, then waits for the CPU lock, and holds both while the command runs', async () => {
        const cpuHolder = await acquireBenchLock({ ...quiet, path: cpuPath, what: 'cpu render' });
        const out = join(folder, 'both-out.json');
        const script = await writeProbeScript(out, { gpu: gpuPath, cpu: cpuPath });
        const running = runCliCaptured(['run', '--cpu-timed', '--gpu-path', gpuPath, '--cpu-path', cpuPath, '--timeout-minutes', '0.5', '--', 'node', script], {
            readClock: async () => null,
        });
        // While the CPU render holds its lock: the GPU lock is already ours, and the command has not started.
        const deadline = Date.now() + 10_000;
        while (!existsSync(join(gpuPath, OWNER_FILE)) && Date.now() < deadline) await new Promise((done) => setTimeout(done, 20));
        expect((JSON.parse(await readFile(join(gpuPath, OWNER_FILE), 'utf8')) as BenchLockOwner).pid).toBe(process.pid);
        await new Promise((done) => setTimeout(done, 300));
        expect(existsSync(out)).toBe(false);

        expect(await cpuHolder.release()).toBe('released');
        const { code } = await running;
        expect(code).toBe(0);
        const seen = JSON.parse(await readFile(out, 'utf8')) as { exists: Record<string, boolean>; held: string };
        expect(seen.exists).toEqual({ gpu: true, cpu: true });
        expect(seen.held).toBe(String(process.pid));
        expect(existsSync(gpuPath)).toBe(false);
        expect(existsSync(cpuPath)).toBe(false);
    }, 30_000);

    it('run refuses --cpu-timed with --lock cpu', async () => {
        await expect(benchLockCli(['run', '--cpu-timed', '--lock', 'cpu', '--cpu-path', cpuPath, '--', 'node', '-e', '0'])).rejects.toThrow(/--cpu-timed/);
    });

    it('status shows both locks; the exit code is for the chosen lock', async () => {
        const cpuHolder = await acquireBenchLock({ ...quiet, path: cpuPath, what: 'cpu render' });
        try {
            const gpuStatus = await runCliCaptured(['status', '--gpu-path', gpuPath, '--cpu-path', cpuPath]);
            expect(gpuStatus.code).toBe(0);
            expect(gpuStatus.stdout.some((line) => line.startsWith(`GPU lock ${gpuPath}: free`))).toBe(true);
            expect(gpuStatus.stdout.some((line) => line.startsWith(`CPU lock ${cpuPath}: held by pid ${process.pid}`) && line.includes('cpu render'))).toBe(true);
            expect((await runCliCaptured(['status', '--lock', 'cpu', '--gpu-path', gpuPath, '--cpu-path', cpuPath])).code).toBe(3);

            const json = await runCliCaptured(['status', '--json', '--gpu-path', gpuPath, '--cpu-path', cpuPath]);
            const parsed = JSON.parse(json.stdout.join('\n')) as { path: string; kind: string; locks: Record<string, { path: string; kind: string }> };
            expect(parsed.path).toBe(gpuPath);
            expect(parsed.kind).toBe('free');
            expect(parsed.locks.gpu.kind).toBe('free');
            expect(parsed.locks.cpu).toMatchObject({ path: cpuPath, kind: 'owned' });
        } finally {
            await cpuHolder.release();
        }
    }, 30_000);

    it('withBenchLock takes the lock it is given, and a parent hold can name two pids', async () => {
        const seen = await withBenchLock(async () => existsSync(join(cpuPath, OWNER_FILE)), { ...quiet, lock: 'cpu', path: cpuPath });
        expect(seen).toBe(true);
        expect(existsSync(cpuPath)).toBe(false);

        expect(parseHeldPids('1234,5678')).toEqual([1234, 5678]);
        expect(parseHeldPids('x, 0, 42')).toEqual([42]);
        await mkdir(cpuPath);
        await writeFile(join(cpuPath, OWNER_FILE), JSON.stringify(foreignOwner()));
        const probe = fakeProbe({ 4242: 'started-at-noon' });
        expect((await lockHeldByParent({ path: cpuPath, probe, env: { [LOCK_HELD_ENV]: '1111,4242' } }))?.pid).toBe(4242);
        expect(await lockHeldByParent({ path: cpuPath, probe, env: { [LOCK_HELD_ENV]: '1111' } })).toBeNull();
    });
});
