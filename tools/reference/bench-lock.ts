/**
 * The shared bench lock (see docs/NOTES.md, "One GPU benchmark at a time"): a
 * folder `$TEMP/ddgi-gpu-bench.lock`. Anything that loads the machine for
 * minutes holds it: a GPU timing run, a native harness run, a Cycles render (16
 * CPU threads skew other agents' build-time and CPU-side numbers just as a GPU
 * job skews GPU numbers) and a long path-tracer run.
 *
 * The protocol (also in tools/reference/README.md, "Bench lock"):
 *
 *   take       `mkdir` the folder. mkdir fails if it exists, so exactly one
 *              process wins; that is the atomic step.
 *   claim      write `owner.json` inside it straight away (written to a temp
 *              name and renamed, so a reader never sees half a file): pid,
 *              process start time, host, ISO start time, a short "what" label
 *              and the command.
 *   wait       while someone else holds it, poll, and say once a minute who.
 *   stale      the lock is taken over ONLY when its owner is provably gone:
 *              the owner pid is dead, or that pid now belongs to a different
 *              process (its start time differs: Windows reuses pids quickly).
 *              A live owner is never touched, however long it holds the lock.
 *              A folder with no owner file is taken over only once it is older
 *              than `ownerlessStaleMs` (see OWNERLESS_STALE_MS for why that is
 *              long today).
 *   release    remove the folder, but only if owner.json is ours.
 *
 * In-process: `withBenchLock(work, { what })`. It releases when the work ends,
 * fails, or the process is interrupted (Ctrl-C, SIGTERM): the release is on the
 * tools/gpu cleanup list, which `installShutdownHandlers()` runs on a signal.
 *
 * Nested holders: `run` gives its command LOCK_HELD_ENV=<owner pid>. `withBenchLock` (and
 * tools/native's scripts) check it, and if that pid owns the lock they run under the parent's
 * hold instead of waiting for a lock that will not be released until they finish.
 *
 * GPU clock: while `run`'s command runs, it samples nvidia-smi about once a second (if
 * nvidia-smi exists), prints min / median / max SM and memory clock at the end, and keeps the
 * summary in a JSON file (`--clock-file`, default `<lock folder>.clock.json`), rewritten after
 * every sample. The command gets the file's path in DDGI_CLOCK_FILE, so the to-csv scripts and
 * tools/native/bench.ps1 add `note/sm_clock_mhz` rows (see tools/gpu/clock-note.ts). If the median
 * SM clock over the samples under load (70% GPU use or more) is below 1,500 MHz, the end of the run
 * prints a warning and the to-csv rows get `note/clock_low,1`.
 *
 * From a shell (agents, the native PowerShell launcher), the same protocol:
 *
 *   node tools/reference/bench-lock.ts status
 *   node tools/reference/bench-lock.ts run --what "trace bench" -- pnpm test:gpu --filter trace-bench
 *   node tools/reference/bench-lock.ts acquire --pid $PID --what "native cache build"
 *   node tools/reference/bench-lock.ts release --pid $PID
 */

import { execFile, spawn } from 'node:child_process';
import { mkdir, readFile, rename, rm, stat, writeFile } from 'node:fs/promises';
import { hostname, tmpdir } from 'node:os';
import { join } from 'node:path';
import { parseArgs } from 'node:util';
import { installShutdownHandlers, registerCleanup } from '../gpu/cleanup.ts';
import {
    CLOCK_FILE_ENV,
    clockLowWarning,
    describeClockSummary,
    NVIDIA_SMI_CLOCK_ARGS,
    parseClockLine,
    summarizeClockSamples,
    type ClockSample,
    type ClockSummary,
} from '../gpu/clock-note.ts';

export const benchLockPath = join(process.env.TEMP ?? process.env.TMPDIR ?? tmpdir(), 'ddgi-gpu-bench.lock');

/** The file inside the lock folder that says who holds it. */
export const OWNER_FILE = 'owner.json';

/**
 * `run` sets this in the command's environment to the pid that owns the lock. A tool started
 * under `run` that would take the lock itself (`withBenchLock`, tools/native's PowerShell
 * scripts) would otherwise wait forever on its own parent. When the variable is set AND that pid
 * really owns the lock right now, such a tool uses the parent's hold instead of taking it again.
 */
export const LOCK_HELD_ENV = 'DDGI_BENCH_LOCK_HELD';

/** Who holds the lock. Written by the taker, read by everyone who waits. */
export interface BenchLockOwner {
    pid: number;
    /**
     * When the owner process started, as reported by the OS (see `systemProbe`),
     * or null if it could not be read. Together with the pid it identifies the
     * process even after Windows hands the same pid to a new one.
     */
    processStart: string | null;
    host: string;
    /** ISO time the lock was taken. */
    startedAt: string;
    /** A short label for humans: "cycles sponza", "native cache build". */
    what: string;
    command: string;
}

/** What the lock needs to know about other processes. Injected in tests. */
export interface ProcessProbe {
    isAlive(pid: number): boolean;
    /** An opaque start-time string for a running process, or null if unknown. */
    startTime(pid: number): Promise<string | null>;
}

export type LockState =
    | { kind: 'free' }
    | { kind: 'owned'; owner: BenchLockOwner; ownerText: string; folderAgeMs: number }
    | { kind: 'ownerless'; folderAgeMs: number; folderMtimeMs: number };

export interface Verdict {
    stale: boolean;
    /** Plain-language reason, printed when the lock is taken over or while waiting. */
    reason: string;
}

/**
 * How old a lock folder WITHOUT an owner file must be before it counts as
 * abandoned. With every holder writing owner.json, a missing file only means a
 * crash between mkdir and the write (milliseconds), and a few minutes would do.
 * But some holders still take the lock with a bare `mkdir` and write no owner
 * file: tools/native's `Use-BenchLock` (PowerShell) and the manual protocol in
 * docs/NOTES.md. They hold it for whole native runs, up to an hour or more,
 * and they are alive. Taking their lock after a few minutes would break the one
 * rule that matters (never take a live owner's lock), so the default is longer
 * than any such run. Lower it (option or `--ownerless-stale-minutes`) once every
 * holder writes owner.json.
 */
export const OWNERLESS_STALE_MS = 3 * 60 * 60_000;

/**
 * Waiters poll often: some holders (tools/native's scripts) release and re-take the
 * lock between consecutive runs, and a slow poll keeps missing that gap. A poll is
 * one mkdir; the expensive part of judging a holder (the start-time lookup, a
 * PowerShell call on Windows) is cached for START_TIME_CACHE_MS.
 */
const POLL_MS = 1_000;
const START_TIME_CACHE_MS = 30_000;
const REPORT_EVERY_MS = 60_000;

export interface BenchLockOptions {
    /** Lock folder; default `benchLockPath`. */
    path?: string;
    what?: string;
    command?: string;
    /** The process that owns the lock; default this one. The CLI's `acquire --pid` sets it. */
    pid?: number;
    /** Give up after this long (throws BenchLockTimeoutError); default: wait forever. */
    timeoutMs?: number | null;
    ownerlessStaleMs?: number;
    pollMs?: number;
    reportEveryMs?: number;
    log?: (message: string) => void;
    probe?: ProcessProbe;
}

export class BenchLockTimeoutError extends Error {}

export interface BenchLockHandle {
    owner: BenchLockOwner;
    /** Removes the lock if it is still ours. Returns what happened. */
    release(): Promise<ReleaseResult>;
}

export type ReleaseResult = 'released' | 'already-free' | 'not-ours';

// ---------------------------------------------------------------------------
// Process probe
// ---------------------------------------------------------------------------

function run(file: string, args: string[]): Promise<string | null> {
    return new Promise((resolvePromise) => {
        execFile(file, args, { windowsHide: true, timeout: 15_000 }, (error, stdout) => {
            resolvePromise(error ? null : stdout.trim() || null);
        });
    });
}

/** The real OS. Start times: Windows via PowerShell, Linux via /proc, others via `ps`. */
export const systemProbe: ProcessProbe = {
    isAlive(pid) {
        try {
            // Signal 0 checks for existence without sending anything (Windows included).
            process.kill(pid, 0);
            return true;
        } catch (error) {
            // EPERM: it exists, we just may not signal it.
            return (error as NodeJS.ErrnoException).code === 'EPERM';
        }
    },
    async startTime(pid) {
        if (!Number.isInteger(pid) || pid <= 0) return null;
        if (process.platform === 'win32') {
            // UTC round-trip format, to 100 ns: two processes never share it.
            const script = `$p = Get-Process -Id ${pid} -ErrorAction SilentlyContinue; if ($p) { $p.StartTime.ToUniversalTime().ToString('o') }`;
            return run('powershell.exe', ['-NoProfile', '-NonInteractive', '-Command', script]);
        }
        if (process.platform === 'linux') {
            try {
                // Field 22 (starttime, clock ticks since boot). The name field can hold spaces
                // and parentheses, so count fields after its closing parenthesis.
                const text = await readFile(`/proc/${pid}/stat`, 'utf8');
                const fields = text.slice(text.lastIndexOf(')') + 2).split(' ');
                return fields[19] ?? null;
            } catch {
                return null;
            }
        }
        return run('ps', ['-o', 'lstart=', '-p', String(pid)]);
    },
};

/**
 * The same probe, but a pid's start time is looked up at most once per
 * START_TIME_CACHE_MS. A pid reused within that time is still caught on the next
 * lookup; liveness is checked on every call.
 */
function cachingStartTimes(probe: ProcessProbe): ProcessProbe {
    const cache = new Map<number, { value: string | null; at: number }>();
    return {
        isAlive: (pid) => probe.isAlive(pid),
        async startTime(pid) {
            const cached = cache.get(pid);
            if (cached && Date.now() - cached.at < START_TIME_CACHE_MS) return cached.value;
            const value = await probe.startTime(pid);
            cache.set(pid, { value, at: Date.now() });
            return value;
        },
    };
}

// ---------------------------------------------------------------------------
// Reading and judging the lock
// ---------------------------------------------------------------------------

export async function readLockState(path: string = benchLockPath): Promise<LockState> {
    let folder;
    try {
        folder = await stat(path);
    } catch (error) {
        if ((error as NodeJS.ErrnoException).code === 'ENOENT') return { kind: 'free' };
        throw error;
    }
    const folderAgeMs = Date.now() - folder.mtimeMs;
    let ownerText: string;
    try {
        ownerText = await readFile(join(path, OWNER_FILE), 'utf8');
    } catch {
        return { kind: 'ownerless', folderAgeMs, folderMtimeMs: folder.mtimeMs };
    }
    const owner = parseOwner(ownerText);
    if (!owner) return { kind: 'ownerless', folderAgeMs, folderMtimeMs: folder.mtimeMs };
    return { kind: 'owned', owner, ownerText, folderAgeMs };
}

function parseOwner(text: string): BenchLockOwner | null {
    try {
        const value = JSON.parse(text) as Partial<BenchLockOwner>;
        if (typeof value.pid !== 'number' || typeof value.host !== 'string') return null;
        return {
            pid: value.pid,
            processStart: typeof value.processStart === 'string' ? value.processStart : null,
            host: value.host,
            startedAt: String(value.startedAt ?? ''),
            what: String(value.what ?? ''),
            command: String(value.command ?? ''),
        };
    } catch {
        return null;
    }
}

/** Is the holder provably gone? Never says yes for a live owner. */
export async function judgeLock(
    state: LockState,
    options: { probe?: ProcessProbe; ownerlessStaleMs?: number; host?: string } = {},
): Promise<Verdict> {
    const probe = options.probe ?? systemProbe;
    const ownerlessStaleMs = options.ownerlessStaleMs ?? OWNERLESS_STALE_MS;
    if (state.kind === 'free') return { stale: false, reason: 'free' };
    if (state.kind === 'ownerless') {
        const age = formatMinutes(state.folderAgeMs);
        if (state.folderAgeMs > ownerlessStaleMs) {
            return { stale: true, reason: `no owner file and the folder is ${age} old (over ${formatMinutes(ownerlessStaleMs)})` };
        }
        return {
            stale: false,
            reason: `probably a plain mkdir (tools/native, or a manual hold), which writes no owner file; taken over only after ${formatMinutes(ownerlessStaleMs)}`,
        };
    }
    const { owner } = state;
    if (owner.host !== (options.host ?? hostname())) {
        // The folder is in local TEMP, so this should not happen; if it does, the pid means nothing here.
        return { stale: false, reason: `owned by a process on another host (${owner.host}); cannot check it, so it is kept` };
    }
    if (!probe.isAlive(owner.pid)) return { stale: true, reason: `owner pid ${owner.pid} is no longer running` };
    if (owner.processStart !== null) {
        const current = await probe.startTime(owner.pid);
        if (current !== null && current !== owner.processStart) {
            return { stale: true, reason: `pid ${owner.pid} now belongs to a different process (started ${current}, owner started ${owner.processStart})` };
        }
    }
    return { stale: false, reason: `owner pid ${owner.pid} is running` };
}

export function describeLock(state: LockState): string {
    if (state.kind === 'free') return 'free';
    if (state.kind === 'ownerless') return `held, no owner file, folder ${formatMinutes(state.folderAgeMs)} old`;
    const { owner } = state;
    const label = owner.what ? `"${owner.what}"` : 'no label';
    return `held by pid ${owner.pid} on ${owner.host} since ${owner.startedAt} (${formatMinutes(state.folderAgeMs)}), ${label}: ${owner.command}`;
}

function formatMinutes(ms: number): string {
    return ms < 60_000 ? `${Math.max(0, Math.round(ms / 1000))} s` : `${Math.round(ms / 60_000)} min`;
}

// ---------------------------------------------------------------------------
// Taking and releasing
// ---------------------------------------------------------------------------

/**
 * Takes over a stale lock without racing another waiter that judged it stale at
 * the same time: the folder is first RENAMED to a unique name (atomic, so only
 * one waiter gets it), then checked to be the same lock that was judged. If a
 * newer lock was grabbed by mistake (someone took over and re-took it in
 * between), it is put back.
 */
async function removeStaleLock(path: string, judged: LockState): Promise<boolean> {
    const aside = `${path}.stale-${process.pid}-${Date.now()}`;
    try {
        await rename(path, aside);
    } catch {
        return false; // someone else moved or released it first
    }
    const moved = await readLockState(aside);
    const sameLock =
        (judged.kind === 'owned' && moved.kind === 'owned' && moved.ownerText === judged.ownerText) ||
        (judged.kind === 'ownerless' && moved.kind === 'ownerless' && moved.folderMtimeMs === judged.folderMtimeMs);
    if (!sameLock) {
        await rename(aside, path).catch(() => undefined);
        return false;
    }
    await rm(aside, { recursive: true, force: true });
    return true;
}

async function writeOwner(path: string, owner: BenchLockOwner): Promise<void> {
    const temporary = join(path, `${OWNER_FILE}.${process.pid}.tmp`);
    await writeFile(temporary, `${JSON.stringify(owner, null, 2)}\n`);
    await rename(temporary, join(path, OWNER_FILE));
}

function sameOwner(a: BenchLockOwner, b: BenchLockOwner): boolean {
    return a.pid === b.pid && a.host === b.host && a.startedAt === b.startedAt;
}

/** Removes the lock if `isOurs` accepts its owner file. A lock without an owner file is never ours. */
export async function releaseBenchLock(isOurs: (owner: BenchLockOwner) => boolean, path: string = benchLockPath): Promise<ReleaseResult> {
    const state = await readLockState(path);
    if (state.kind === 'free') return 'already-free';
    if (state.kind !== 'owned' || !isOurs(state.owner)) return 'not-ours';
    await rm(path, { recursive: true, force: true });
    return 'released';
}

export async function acquireBenchLock(options: BenchLockOptions = {}): Promise<BenchLockHandle> {
    const path = options.path ?? benchLockPath;
    const log = options.log ?? console.log;
    const probe = cachingStartTimes(options.probe ?? systemProbe);
    const pollMs = options.pollMs ?? POLL_MS;
    const reportEveryMs = options.reportEveryMs ?? REPORT_EVERY_MS;
    const ownerPid = options.pid ?? process.pid;
    // Looked up before taking the lock (on Windows it starts PowerShell, about half a
    // second), so the folder is without its owner file for milliseconds only.
    const ownerProcessStart = await probe.startTime(ownerPid);
    const startedWaiting = Date.now();
    let lastReport = -Infinity;

    for (;;) {
        try {
            // mkdir without `recursive` fails if the folder exists: that is the atomic "take" step.
            await mkdir(path);
        } catch (error) {
            if ((error as NodeJS.ErrnoException).code !== 'EEXIST') throw error;
            const state = await readLockState(path);
            if (state.kind === 'free') continue; // released between our mkdir and the read
            const verdict = await judgeLock(state, { probe, ownerlessStaleMs: options.ownerlessStaleMs });
            if (verdict.stale && (await removeStaleLock(path, state))) {
                log(`bench lock: removed a stale lock (${verdict.reason}; it was ${describeLock(state)})`);
                continue;
            }
            // Not stale, or the stale folder could not be moved (another waiter got there
            // first, or Windows refuses while a file in it is open): wait and look again.
            const waited = Date.now() - startedWaiting;
            if (Date.now() - lastReport >= reportEveryMs) {
                lastReport = Date.now();
                log(`bench lock (${path}) is ${describeLock(state)}; ${verdict.reason}. Waiting${waited > 0 ? ` (${formatMinutes(waited)} so far)` : ''}...`);
            }
            if (options.timeoutMs != null && waited >= options.timeoutMs) {
                throw new BenchLockTimeoutError(`gave up after waiting ${formatMinutes(waited)} for the bench lock, which is ${describeLock(state)}`);
            }
            await new Promise((resolvePromise) => setTimeout(resolvePromise, pollMs));
            continue;
        }

        const owner: BenchLockOwner = {
            pid: ownerPid,
            processStart: ownerProcessStart,
            host: hostname(),
            startedAt: new Date().toISOString(),
            what: options.what ?? '',
            command: options.command ?? process.argv.join(' '),
        };
        try {
            await writeOwner(path, owner);
        } catch (error) {
            await rm(path, { recursive: true, force: true });
            throw error;
        }
        log(`took the bench lock (${path})${owner.what ? ` for "${owner.what}"` : ''}`);
        return {
            owner,
            release: () => releaseBenchLock((found) => sameOwner(found, owner), path),
        };
    }
}

/**
 * The owner of the lock if a parent process holds it for us: `LOCK_HELD_ENV` names a pid, and the
 * lock at `path` is owned by that pid on this host and the pid is running. Anything else (no
 * variable, a different owner, a free lock, a dead pid) returns null, so a stale or leaked
 * variable never lets a process skip the lock.
 */
export async function lockHeldByParent(
    options: { path?: string; probe?: ProcessProbe; env?: NodeJS.ProcessEnv } = {},
): Promise<BenchLockOwner | null> {
    const text = (options.env ?? process.env)[LOCK_HELD_ENV];
    if (!text) return null;
    const pid = Number(text);
    if (!(Number.isInteger(pid) && pid > 0)) return null;
    const state = await readLockState(options.path ?? benchLockPath);
    if (state.kind !== 'owned' || state.owner.pid !== pid || state.owner.host !== hostname()) return null;
    return (options.probe ?? systemProbe).isAlive(pid) ? state.owner : null;
}

/**
 * Holds the lock while `work` runs. The second argument may be a log function
 * (the old signature) or options. `work` gets the pid that owns the lock (this process, or the
 * parent that holds it for us), so it can pass it on to a child in `LOCK_HELD_ENV`.
 */
export async function withBenchLock<T>(
    work: (lock: { ownerPid: number }) => Promise<T>,
    options: BenchLockOptions | ((message: string) => void) = {},
): Promise<T> {
    const settings: BenchLockOptions = typeof options === 'function' ? { log: options } : options;
    const log = settings.log ?? console.log;
    const parent = await lockHeldByParent({ path: settings.path, probe: settings.probe });
    if (parent) {
        // Started under `bench-lock run` (or another holder that set LOCK_HELD_ENV): the lock is
        // already ours through the parent, which also releases it. Taking it again would wait forever.
        log(`the bench lock is already held for us by pid ${parent.pid} ("${parent.what}"); not taking it again`);
        return work({ ownerPid: parent.pid });
    }
    const handle = await acquireBenchLock(settings);
    // Run-once: whichever of the normal path and the shutdown handler gets there first releases it.
    const release = registerCleanup(async () => {
        const result = await handle.release();
        if (result === 'not-ours') log('warning: the bench lock no longer has our owner file; left it alone');
    });
    try {
        return await work({ ownerPid: handle.owner.pid });
    } finally {
        await release();
        log('released the bench lock');
    }
}

// ---------------------------------------------------------------------------
// GPU clock sampling
// ---------------------------------------------------------------------------

const CLOCK_SAMPLE_MS = 1_000;

/** Returns nvidia-smi's clock line, or null if nvidia-smi is missing or fails. Injected in tests. */
export type ClockReader = () => Promise<string | null>;

export const readNvidiaSmiClocks: ClockReader = () => run('nvidia-smi', NVIDIA_SMI_CLOCK_ARGS);

export interface ClockSettings {
    file: string;
    read: ClockReader;
    intervalMs?: number;
}

export interface ClockSampler {
    /** Stops sampling, writes the final file and returns the summary (null if nothing was sampled). */
    stop(): Promise<ClockSummary | null>;
}

/**
 * Samples the GPU clock every `intervalMs` until `stop()`, rewriting `file` after each sample
 * (to a temp name and renamed, so a reader never sees half a file). Returns null if the reader
 * gives nothing on the first try: no nvidia-smi means no clock rows, not an error.
 */
export async function startClockSampler(settings: ClockSettings): Promise<ClockSampler | null> {
    const samples: ClockSample[] = [];
    const take = async (): Promise<void> => {
        const text = await settings.read();
        const sample = text === null ? null : parseClockLine(text);
        if (sample) samples.push(sample);
    };
    await take();
    if (samples.length === 0) return null;

    const writeSummary = async (): Promise<ClockSummary | null> => {
        const summary = summarizeClockSamples(samples);
        if (!summary) return null;
        const temporary = `${settings.file}.${process.pid}.tmp`;
        // A reader holding the old file open can make the rename fail on Windows; the next sample retries.
        await writeFile(temporary, `${JSON.stringify(summary, null, 2)}\n`)
            .then(() => rename(temporary, settings.file))
            .catch(() => undefined);
        return summary;
    };
    await writeSummary();

    let busy: Promise<void> = Promise.resolve();
    const timer = setInterval(() => {
        // One sample at a time; a slow nvidia-smi call delays the next one rather than piling up.
        busy = busy.then(async () => {
            await take();
            await writeSummary();
        });
    }, settings.intervalMs ?? CLOCK_SAMPLE_MS);
    return {
        async stop() {
            clearInterval(timer);
            await busy;
            return writeSummary();
        },
    };
}

// ---------------------------------------------------------------------------
// Command line
// ---------------------------------------------------------------------------

const CLI_HELP = `usage: node tools/reference/bench-lock.ts <command> [options]

The shared bench lock ($TEMP/ddgi-gpu-bench.lock): one machine-heavy job at a time.

commands:
  status                      Who holds it, and whether that holder is still running.
                              Exit code 0 free, 3 held, 4 held but stale.
  run --what <label> -- <command...>
                              Wait for the lock, run the command holding it, release it
                              (also on failure or Ctrl-C). Exits with the command's code.
                              The command gets ${LOCK_HELD_ENV}=<owner pid>; tools that take
                              the lock themselves (withBenchLock, tools/native) see it and use
                              this hold instead of waiting on it.
  acquire --pid <pid> --what <label>
                              Wait for the lock and take it for process <pid> (a shell
                              script passes its own pid, e.g. PowerShell $PID). The lock
                              stays until "release", or until that process is gone, after
                              which the next waiter takes it over.
  release --pid <pid>         Remove the lock if <pid> on this host owns it. Refuses (exit
                              code 2) if someone else holds it.

options:
  --timeout-minutes <n>       acquire/run: give up after n minutes (exit code 5).
  --ownerless-stale-minutes <n>
                              A lock folder with no owner file counts as abandoned after
                              this long (default ${OWNERLESS_STALE_MS / 60_000}).
  --clock-file <file>         run: where the GPU clock summary goes (default <lock folder>.clock.json).
                              While the command runs, nvidia-smi is sampled about once a second (if
                              it exists); min / median / max SM and memory clock print at the end.
                              The command gets ${CLOCK_FILE_ENV}=<file>.
  --command <text>            acquire: the command to record (default: this one).
  --path <folder>             Another lock folder (tests).
  --json                      status: print JSON (with "heldByParent": the owner pid when
                              ${LOCK_HELD_ENV} names the pid that holds the lock, else null).
`;

function quoteForShell(argument: string): string {
    return /^[\w@%+=:,./\\-]+$/.test(argument) ? argument : `"${argument.replace(/(["\\])/g, '\\$1')}"`;
}

/** Runs a command under the lock; returns its exit code. */
async function runUnderLock(command: string[], settings: BenchLockOptions, clock: ClockSettings): Promise<number> {
    return withBenchLock(async ({ ownerPid }) => {
        const sampler = await startClockSampler(clock);
        try {
            return await runCommand(command, ownerPid, sampler ? clock.file : null);
        } finally {
            const summary = await sampler?.stop();
            if (summary) {
                const log = settings.log ?? console.log;
                log(describeClockSummary(summary));
                const warning = clockLowWarning(summary);
                if (warning) log(warning);
            }
        }
    }, settings);
}

function runCommand(command: string[], ownerPid: number, clockFile: string | null): Promise<number> {
    return new Promise<number>((resolvePromise, reject) => {
        // One command string through the shell, so `pnpm` (a .cmd on Windows) resolves.
        // LOCK_HELD_ENV tells a command that takes the lock itself that we hold it for it.
        const child = spawn(command.map(quoteForShell).join(' '), {
            stdio: 'inherit',
            shell: true,
            env: { ...process.env, [LOCK_HELD_ENV]: String(ownerPid), ...(clockFile ? { [CLOCK_FILE_ENV]: clockFile } : {}) },
        });
        const releaseKill = registerCleanup(
            () =>
                new Promise<void>((done) => {
                    if (child.pid === undefined || child.exitCode !== null) return done();
                    if (process.platform === 'win32') {
                        const taskkill = spawn('taskkill', ['/PID', String(child.pid), '/T', '/F'], { windowsHide: true, stdio: 'ignore' });
                        taskkill.on('exit', () => done());
                        taskkill.on('error', () => done());
                    } else {
                        child.kill('SIGTERM');
                        done();
                    }
                }),
        );
        child.on('error', (error) => {
            void releaseKill();
            reject(error);
        });
        child.on('close', (code) => {
            void releaseKill();
            resolvePromise(code ?? 1);
        });
    });
}

/** Test hooks for the command line: another clock reader or sampling interval. */
export interface BenchLockCliDeps {
    readClock?: ClockReader;
    clockIntervalMs?: number;
}

export async function benchLockCli(argv: string[], deps: BenchLockCliDeps = {}): Promise<number> {
    const separator = argv.indexOf('--');
    const ownArgs = separator >= 0 ? argv.slice(0, separator) : argv;
    const command = separator >= 0 ? argv.slice(separator + 1) : [];
    const { values, positionals } = parseArgs({
        args: ownArgs,
        allowPositionals: true,
        options: {
            pid: { type: 'string' },
            what: { type: 'string' },
            command: { type: 'string' },
            path: { type: 'string' },
            'timeout-minutes': { type: 'string' },
            'ownerless-stale-minutes': { type: 'string' },
            'clock-file': { type: 'string' },
            json: { type: 'boolean' },
            help: { type: 'boolean', short: 'h' },
        },
    });
    const action = positionals[0];
    if (values.help || !action) {
        console.log(CLI_HELP);
        return values.help ? 0 : 1;
    }
    const path = values.path ?? benchLockPath;
    const minutes = (text: string | undefined, name: string): number | undefined => {
        if (text === undefined) return undefined;
        const value = Number(text);
        if (!(value >= 0)) throw new Error(`--${name} must be a number of minutes`);
        return value * 60_000;
    };
    const ownerlessStaleMs = minutes(values['ownerless-stale-minutes'], 'ownerless-stale-minutes');
    const timeoutMs = minutes(values['timeout-minutes'], 'timeout-minutes') ?? null;
    const pid = values.pid === undefined ? undefined : Number(values.pid);
    if (pid !== undefined && !(Number.isInteger(pid) && pid > 0)) throw new Error('--pid must be a process id');
    // Waiting messages go to stderr, so a script can read stdout cleanly.
    const log = (message: string) => console.error(message);

    switch (action) {
        case 'status': {
            const state = await readLockState(path);
            const verdict = await judgeLock(state, { ownerlessStaleMs });
            if (values.json) {
                const parent = await lockHeldByParent({ path });
                console.log(JSON.stringify({ path, ...state, stale: verdict.stale, reason: verdict.reason, heldByParent: parent?.pid ?? null }, null, 2));
            }
            else console.log(`${path}: ${describeLock(state)}${state.kind === 'free' ? '' : `\n${verdict.stale ? 'STALE' : 'live'}: ${verdict.reason}`}`);
            return state.kind === 'free' ? 0 : verdict.stale ? 4 : 3;
        }
        case 'acquire': {
            if (pid === undefined) {
                throw new Error('acquire needs --pid <the process that will hold the lock>: this command exits at once, so its own pid would look dead');
            }
            const handle = await acquireBenchLock({
                path,
                pid,
                what: values.what,
                command: values.command ?? `${process.argv.slice(1).join(' ')}`,
                timeoutMs,
                ownerlessStaleMs,
                log,
            });
            console.log(JSON.stringify(handle.owner));
            return 0;
        }
        case 'release': {
            if (pid === undefined) throw new Error('release needs --pid <the pid given to acquire>');
            const host = hostname();
            const result = await releaseBenchLock((owner) => owner.pid === pid && owner.host === host, path);
            if (result === 'not-ours') {
                console.error(`not released: ${describeLock(await readLockState(path))}, not pid ${pid}`);
                return 2;
            }
            console.log(result === 'released' ? 'released' : 'already free');
            return 0;
        }
        case 'run': {
            if (command.length === 0) throw new Error('run needs a command after --');
            installShutdownHandlers();
            return runUnderLock(
                command,
                { path, what: values.what ?? command.join(' '), command: command.join(' '), timeoutMs, ownerlessStaleMs, log },
                { file: values['clock-file'] ?? `${path}.clock.json`, read: deps.readClock ?? readNvidiaSmiClocks, intervalMs: deps.clockIntervalMs },
            );
        }
        default:
            throw new Error(`unknown command "${action}"; see --help`);
    }
}

if (import.meta.main) {
    benchLockCli(process.argv.slice(2)).then(
        (code) => process.exit(code),
        (error) => {
            console.error(`error: ${error instanceof Error ? error.message : error}`);
            process.exit(error instanceof BenchLockTimeoutError ? 5 : 1);
        },
    );
}
