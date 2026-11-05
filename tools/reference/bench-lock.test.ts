import { existsSync } from 'node:fs';
import { mkdir, mkdtemp, readFile, rm, utimes, writeFile } from 'node:fs/promises';
import { hostname, tmpdir } from 'node:os';
import { join } from 'node:path';
import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { CLOCK_FILE_ENV, clockLowWarning, clockNoteRows, isClockLow, LOW_CLOCK_MHZ, parseClockLine, readClockSummary, summarizeClockSamples } from '../gpu/clock-note.ts';
import {
    acquireBenchLock,
    benchLockCli,
    BenchLockTimeoutError,
    judgeLock,
    LOCK_HELD_ENV,
    lockHeldByParent,
    OWNER_FILE,
    readLockState,
    releaseBenchLock,
    startClockSampler,
    systemProbe,
    withBenchLock,
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
