/**
 * The shared GPU benchmark lock (see docs/NOTES.md, "One GPU benchmark at a
 * time"): a folder `$TEMP/ddgi-gpu-bench.lock`. Anything that loads the machine
 * for minutes holds it: a Cycles render (16 CPU threads skew other agents'
 * build-time and CPU-side numbers just as a GPU job skews GPU numbers) and a
 * long path-tracer run. Take it with `withBenchLock`, which waits while someone
 * else has it and releases it when the work ends, fails, or the process is
 * interrupted (Ctrl-C, SIGTERM): the release is registered with the tools/gpu
 * cleanup list, which `installShutdownHandlers()` runs on a signal. A stale lock
 * would block every agent, so callers must call `installShutdownHandlers()`.
 */

import { mkdir, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { registerCleanup } from '../gpu/cleanup.ts';

export const benchLockPath = join(process.env.TEMP ?? process.env.TMPDIR ?? tmpdir(), 'ddgi-gpu-bench.lock');

const POLL_MS = 5_000;

export async function withBenchLock<T>(work: () => Promise<T>, log: (message: string) => void = console.log): Promise<T> {
    let announcedWaiting = false;
    for (;;) {
        try {
            // mkdir without `recursive` fails if the folder exists: that is the atomic "take" step.
            await mkdir(benchLockPath);
            break;
        } catch (error) {
            if ((error as NodeJS.ErrnoException).code !== 'EEXIST') throw error;
            if (!announcedWaiting) {
                log(`bench lock is held (${benchLockPath}); waiting for it...`);
                announcedWaiting = true;
            }
            await new Promise((resolve) => setTimeout(resolve, POLL_MS));
        }
    }
    log(`took the bench lock (${benchLockPath})`);
    // Run-once: whichever of the normal path and the shutdown handler gets there first releases it.
    const release = registerCleanup(() => rm(benchLockPath, { recursive: true, force: true }));
    try {
        return await work();
    } finally {
        await release();
        log('released the bench lock');
    }
}
