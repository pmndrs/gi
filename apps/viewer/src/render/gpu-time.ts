/**
 * GPU time from three's own timestamp tracking.
 *
 * three records a begin/end timestamp around every render and compute pass
 * it submits, into one query pool per pass type. `resolveTimestampsAsync(type)`
 * reads the pool back and empties it. The durations are the GPU's busy time
 * inside passes: gaps between passes are not counted, which makes them
 * steadier than CPU frame time, and is what "GPU ms" means in this viewer.
 *
 * Our own `GpuTimer` (ddgi-core) times passes we encode ourselves. Passes
 * that three encodes, like everything in these baselines, are timed here.
 *
 * Why three's own return value is not used: it is the total of the passes
 * that carry the *last frame id* in the window. The frame id (`info.frame`)
 * changes only in three's animation loop (`Animation.js:87`), once per tick.
 * A window can hold several ticks (the overlay resolves less often than it
 * draws), or work done outside a tick (a probe-bake step run between frames
 * carries the id of the tick before it). So this file keeps three's per-pass
 * durations and groups them by frame id itself: callers ask either for the
 * whole window or for the newest frame in it.
 */

import type { WebGPURenderer } from 'three/webgpu';

interface QueryPoolShape {
    currentQueryIndex: number;
    maxQueries: number;
    /** Duration in milliseconds of every pass resolved by the last `resolveQueriesAsync`, keyed `<kind>:<calls>:<context id>:f<frame>`. */
    timestamps: Map<string, number>;
}

interface TimedBackend {
    trackTimestamp: boolean;
    /** three creates each pool on first use; until then the slot holds null. */
    timestampQueryPool?: Partial<Record<'render' | 'compute', QueryPoolShape | null>>;
}

export interface GpuPassTotals {
    renderMs: number;
    computeMs: number;
    /** How many passes the totals add up. Zero means nothing was timed. */
    passCount: number;
    /**
     * Passes whose timestamps were impossible (end before start, or over a minute) and were
     * left out of the totals. The Intel UHD does this now and then; one such pass made a bake
     * report -3,546 seconds. A window with any of them cannot be trusted as a total.
     */
    invalidPasses: number;
}

export interface GpuWindow {
    /** Everything submitted since the previous resolve. */
    total: GpuPassTotals;
    /** The same passes split by three's frame id. */
    byFrame: Map<number, GpuPassTotals>;
}

/** A single pass longer than this is a timestamp glitch, not work. */
const MAX_PLAUSIBLE_PASS_MS = 60_000;

const FRAME_ID_PATTERN = /:f(\d+)$/;

function emptyTotals(): GpuPassTotals {
    return { renderMs: 0, computeMs: 0, passCount: 0, invalidPasses: 0 };
}

function addPass(totals: GpuPassTotals, kind: 'render' | 'compute', durationMs: number): void {
    totals.passCount++;
    if (!Number.isFinite(durationMs) || durationMs < 0 || durationMs > MAX_PLAUSIBLE_PASS_MS) {
        totals.invalidPasses++;
        return;
    }
    if (kind === 'render') totals.renderMs += durationMs;
    else totals.computeMs += durationMs;
}

function backendOf(renderer: WebGPURenderer): TimedBackend {
    return renderer.backend as unknown as TimedBackend;
}

/**
 * Reads and clears the GPU time of everything submitted since the last call.
 * A pool with no new passes would hand back its previous durations, so those
 * pools are skipped.
 */
export async function resolveGpuWindow(renderer: WebGPURenderer): Promise<GpuWindow> {
    const backend = backendOf(renderer);
    const window: GpuWindow = { total: emptyTotals(), byFrame: new Map() };
    if (!backend.trackTimestamp) return window;
    const pools = backend.timestampQueryPool ?? {};
    for (const kind of ['render', 'compute'] as const) {
        const pool = pools[kind];
        if (!pool || pool.currentQueryIndex === 0) continue;
        await renderer.resolveTimestampsAsync(kind);
        for (const [uid, durationMs] of pool.timestamps) {
            addPass(window.total, kind, durationMs);
            const frameId = Number(FRAME_ID_PATTERN.exec(uid)?.[1] ?? -1);
            let frame = window.byFrame.get(frameId);
            if (!frame) {
                frame = emptyTotals();
                window.byFrame.set(frameId, frame);
            }
            addPass(frame, kind, durationMs);
        }
    }
    return window;
}

/** The whole window since the last resolve, as one total. */
export async function resolveGpuTotals(renderer: WebGPURenderer): Promise<GpuPassTotals> {
    return (await resolveGpuWindow(renderer)).total;
}

/** The newest frame in a window, or null when the window is empty. */
export function newestFrame(window: GpuWindow): GpuPassTotals | null {
    if (window.byFrame.size === 0) return null;
    return window.byFrame.get(Math.max(...window.byFrame.keys())) ?? null;
}

/**
 * Turns three's pass timing on or off. Off, passes carry no timestamp writes at all,
 * which the benchmark uses for wall-clock bake runs that must not read anything back.
 */
export function setTimestampTracking(renderer: WebGPURenderer, enabled: boolean): void {
    backendOf(renderer).trackTimestamp = enabled;
}

/**
 * True when the query pools are more than half full. Past full, three resolves on its own
 * and the passes in that window are lost to us, so long runs resolve before that.
 */
export function timestampPoolsFillingUp(renderer: WebGPURenderer): boolean {
    const pools = backendOf(renderer).timestampQueryPool ?? {};
    return Object.values(pools).some((pool) => pool != null && pool.currentQueryIndex > pool.maxQueries / 2);
}
