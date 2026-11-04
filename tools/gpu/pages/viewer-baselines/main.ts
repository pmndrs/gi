/**
 * The viewer's baseline benchmark as a GPU test page. See
 * `apps/viewer/src/bench/baselines.ts` for what is measured and how.
 *
 * The runner opens pages with fixed URL parameters, so what to measure is
 * chosen with environment variables set before `pnpm test:gpu` (vite exposes
 * `VITE_*` variables to the page):
 *
 *   (nothing)                      two small scenes, all three techniques: a quick check that it works
 *   VITE_DDGI_BASELINES=full       every scene
 *   VITE_DDGI_SCENES=sponza,...    only these scenes
 *   VITE_DDGI_TECHNIQUES=none,lpg  only these techniques
 *   VITE_DDGI_ROUNDS=5             frame rounds (GI off and on, alternating)
 *   VITE_DDGI_BAKE_RUNS=3          repeats of each bake
 *
 * The page fails when the device reports an error or is lost, when anything
 * is logged as an error, or when any measurement failed (a technique that
 * could not run, or a number that came out negative or not finite). The
 * data is reported either way, so a failed run can still be read.
 *
 * A timing run must hold the bench lock: wrap it in
 * `node tools/reference/bench-lock.ts run --what "T0.5 viewer baselines" -- pnpm test:gpu ...`. Turn the JSON
 * into CSV rows and a table with `node tools/gpu/pages/viewer-baselines/to-csv.ts <file.json>`.
 */

import { reportGpuTest } from '../../client.ts';
import { DEFAULT_BASELINE_OPTIONS, failedMeasurements, runBaselines } from '../../../../apps/viewer/src/bench/baselines.ts';
import type { TechniqueId } from '../../../../apps/viewer/src/techniques/types.ts';

const QUICK_SCENES = ['cornell-box', 'two-room'];

function listFromEnv(value: unknown): string[] | null {
    return typeof value === 'string' && value.trim() !== '' ? value.split(',').map((item) => item.trim()) : null;
}

function countFromEnv(value: unknown, fallback: number): number {
    const parsed = typeof value === 'string' ? Number(value) : NaN;
    return Number.isInteger(parsed) && parsed > 0 ? parsed : fallback;
}

const startTime = performance.now();
const logElement = document.getElementById('log') as HTMLElement;
const logLines: string[] = [];
const log = (message: string): void => {
    logLines.push(message);
    logElement.textContent = logLines.join('\n');
};

// Everything the page itself sees as an error. The runner also watches the browser log;
// collecting here as well puts the messages into the report next to the numbers.
const pageErrors: string[] = [];
window.addEventListener('error', (event) => pageErrors.push(`page error: ${event.message}`));
window.addEventListener('unhandledrejection', (event) => pageErrors.push(`unhandled rejection: ${String(event.reason)}`));
const originalConsoleError = console.error.bind(console);
console.error = (...args: unknown[]) => {
    pageErrors.push(`console.error: ${args.map(String).join(' ')}`);
    originalConsoleError(...args);
};

const env = import.meta.env as Record<string, unknown>;
const requestedScenes = listFromEnv(env.VITE_DDGI_SCENES);
const full = env.VITE_DDGI_BASELINES === 'full';
const techniques = (listFromEnv(env.VITE_DDGI_TECHNIQUES) ?? DEFAULT_BASELINE_OPTIONS.techniques) as TechniqueId[];

try {
    const result = await runBaselines(document.getElementById('view') as HTMLCanvasElement, {
        ...DEFAULT_BASELINE_OPTIONS,
        scenes: requestedScenes ?? (full ? null : QUICK_SCENES),
        techniques,
        frameRounds: countFromEnv(env.VITE_DDGI_ROUNDS, DEFAULT_BASELINE_OPTIONS.frameRounds),
        bakeRuns: countFromEnv(env.VITE_DDGI_BAKE_RUNS, DEFAULT_BASELINE_OPTIONS.bakeRuns),
        log,
    });
    const failures = failedMeasurements(result);
    // GPU errors reach the console too (create-renderer logs them); count each problem once.
    const consoleOnly = pageErrors.filter((message) => !result.gpuErrors.some((gpuError) => message.includes(gpuError)));
    const problems = [...consoleOnly, ...failures];
    reportGpuTest({
        status: problems.length === 0 ? 'pass' : 'fail',
        message: problems.length === 0 ? undefined : `${problems.length} problem(s): ${problems.join(' | ')}`,
        data: { ...result, pageErrors, failedMeasurements: failures } as unknown as Record<string, unknown>,
        // Device errors and device loss fail the page in reportGpuTest.
        deviceErrors: result.gpuErrors,
        startTime,
        adapter: { ...result.adapter, isFallbackAdapter: false },
        logLines: [...logLines, `${result.rows.length} rows, ${failures.length} failed measurement(s), ${pageErrors.length} page error(s)`],
    });
} catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    reportGpuTest({ status: 'fail', message: [message, ...pageErrors].join(' | '), startTime, logLines });
}
