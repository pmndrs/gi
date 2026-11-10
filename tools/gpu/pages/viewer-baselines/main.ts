/**
 * The viewer's baseline benchmark as a GPU test page. See
 * `apps/viewer/src/bench/baselines.ts` for what is measured and how.
 *
 * The runner opens pages with fixed URL parameters, so what to measure is
 * chosen with environment variables set before `pnpm test:gpu` (vite exposes
 * `VITE_*` variables to the page):
 *
 *   (nothing)                      two small scenes, every technique (none, lpg, vxgi, ddgi): a quick check that it works
 *   VITE_DDGI_BASELINES=full       every scene except heavy ones (sponza-full; name it in VITE_DDGI_SCENES)
 *   VITE_DDGI_SCENES=sponza,...    only these scenes
 *   VITE_DDGI_TECHNIQUES=none,ddgi only these techniques
 *   VITE_DDGI_ROUNDS=5             frame rounds (GI off and on, alternating)
 *   VITE_DDGI_FRAMES=20            frames timed per round and GI mode
 *   VITE_DDGI_BAKE_RUNS=3          repeats of each bake
 *   VITE_DDGI_WIDTH=2560 VITE_DDGI_HEIGHT=1440   the view size (default 1280x720)
 *   VITE_DDGI_MAX_TEXTURE_SIZE=1024   scale model textures down to this side at load (default: as the files have them).
 *                                  Bistro exterior's textures take 5.1 GB as loaded, more than a 6 GB GPU holds with DDGI (T2.17).
 *
 * DDGI's frame cost for T1.9 (GI on against off: GPU ms, frame p50 and p95):
 *   VITE_DDGI_SCENES=cornell-box,two-room,sponza VITE_DDGI_TECHNIQUES=none,ddgi pnpm test:gpu --filter viewer-baselines
 *
 * Where DDGI's GI-on frame time goes, pass by pass (`breakdown.ts`; default scenes Cornell, two-room, Sponza;
 * VITE_DDGI_ROUNDS and VITE_DDGI_FRAMES default to 3 and 15 there; VITE_DDGI_VARIANTS and VITE_DDGI_BURST are its own):
 *   VITE_DDGI_BREAKDOWN=1 pnpm test:gpu --filter viewer-baselines
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
import { DEFAULT_BASELINE_OPTIONS, failedMeasurements, runBaselines, type BaselineOptions } from '../../../../apps/viewer/src/bench/baselines.ts';
import { runBreakdown, VARIANTS, type VariantId } from './breakdown.ts';

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
const techniques = (listFromEnv(env.VITE_DDGI_TECHNIQUES) ?? DEFAULT_BASELINE_OPTIONS.techniques) as BaselineOptions['techniques'];

const BREAKDOWN_SCENES = ['cornell-box', 'two-room', 'sponza'];

async function breakdown(): Promise<void> {
    const result = await runBreakdown(document.getElementById('view') as HTMLCanvasElement, {
        scenes: requestedScenes ?? BREAKDOWN_SCENES,
        variants: (listFromEnv(env.VITE_DDGI_VARIANTS) ?? VARIANTS).filter((variant): variant is VariantId => (VARIANTS as readonly string[]).includes(variant)),
        burst: countFromEnv(env.VITE_DDGI_BURST, 1),
        rounds: countFromEnv(env.VITE_DDGI_ROUNDS, 3),
        sampleFrames: countFromEnv(env.VITE_DDGI_FRAMES, 15),
        width: countFromEnv(env.VITE_DDGI_WIDTH, DEFAULT_BASELINE_OPTIONS.width),
        height: countFromEnv(env.VITE_DDGI_HEIGHT, DEFAULT_BASELINE_OPTIONS.height),
        log,
    });
    const failures = result.scenes.filter((scene) => scene.error).map((scene) => `${scene.scene}: ${scene.error}`);
    const consoleOnly = pageErrors.filter((message) => !result.gpuErrors.some((gpuError) => message.includes(gpuError)));
    const problems = [...consoleOnly, ...failures];
    reportGpuTest({
        status: problems.length === 0 ? 'pass' : 'fail',
        message: problems.length === 0 ? undefined : `${problems.length} problem(s): ${problems.join(' | ')}`,
        data: { ...result, pageErrors } as unknown as Record<string, unknown>,
        deviceErrors: result.gpuErrors,
        startTime,
        adapter: { ...result.adapter, isFallbackAdapter: false },
        logLines,
    });
}

async function baselines(): Promise<void> {
    const result = await runBaselines(document.getElementById('view') as HTMLCanvasElement, {
        ...DEFAULT_BASELINE_OPTIONS,
        scenes: requestedScenes ?? (full ? null : QUICK_SCENES),
        techniques,
        frameRounds: countFromEnv(env.VITE_DDGI_ROUNDS, DEFAULT_BASELINE_OPTIONS.frameRounds),
        sampleFrames: countFromEnv(env.VITE_DDGI_FRAMES, DEFAULT_BASELINE_OPTIONS.sampleFrames),
        bakeRuns: countFromEnv(env.VITE_DDGI_BAKE_RUNS, DEFAULT_BASELINE_OPTIONS.bakeRuns),
        width: countFromEnv(env.VITE_DDGI_WIDTH, DEFAULT_BASELINE_OPTIONS.width),
        height: countFromEnv(env.VITE_DDGI_HEIGHT, DEFAULT_BASELINE_OPTIONS.height),
        maxTextureSize: env.VITE_DDGI_MAX_TEXTURE_SIZE === undefined ? null : countFromEnv(env.VITE_DDGI_MAX_TEXTURE_SIZE, 1024),
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
}

try {
    if (env.VITE_DDGI_BREAKDOWN === '1') await breakdown();
    else await baselines();
} catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    reportGpuTest({ status: 'fail', message: [message, ...pageErrors].join(' | '), startTime, logLines });
}
