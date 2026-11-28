/**
 * Shared by the T2.M1 pages (rigid moving objects, docs/design/m3-moving.md
 * §5): each probe's record and state read back, how much a probe's light
 * really changed between two converged fields, the marks against those
 * changes (the far-set check, m3-moving.md risk 4), and a per-pixel error.
 *
 * Not a page itself (no index.html).
 */

import type { DDGIField } from '../../../../packages/core/src/techniques/ddgi/ddgi-technique.ts';
import { PROBE_HISTORY_FLAGS_MASK, PROBE_HISTORY_UPDATES, PROBE_HISTORY_WORD, PROBE_HISTORY_WORDS, PROBE_STATE } from '../../../../packages/core/src/techniques/ddgi/layout.ts';
import { halfToFloat, readBuffer, readProbeData } from './harness.ts';

type Vec3 = [number, number, number];

/** Every probe's record and state, in probe-index order. */
export interface ProbeSnapshot {
    readonly state: number[];
    /** Mean irradiance per probe (history words 6-7, written by every blend). */
    readonly mean: Vec3[];
    readonly pendingChange: number[];
    /** k: blended updates since the probe was new, reset or back on. */
    readonly updates: number[];
    readonly owed: number[];
    readonly flags: number[];
    readonly moverMask: number[];
    /** The frame until which the probe is awake (T3.4, woken air probes); 0 when it isn't. */
    readonly wakeUntil: number[];
    /** Where the probe sits (grid point plus relocation). */
    readonly position: Vec3[];
}

export async function probeSnapshot(device: GPUDevice, field: DDGIField): Promise<ProbeSnapshot> {
    const probeCount = field.grid.probeCount;
    const words = new Uint32Array(await readBuffer(device, field.probeHistory, probeCount * PROBE_HISTORY_WORDS * 4));
    const floats = new Float32Array(words.buffer);
    const probes = await readProbeData(device, field);
    const [nx, , nz] = field.grid.counts;
    const snapshot: ProbeSnapshot = { state: [], mean: [], pendingChange: [], updates: [], owed: [], flags: [], moverMask: [], wakeUntil: [], position: [] };
    for (let probe = 0; probe < probeCount; probe++) {
        const base = probe * PROBE_HISTORY_WORDS;
        const redGreen = words[base + PROBE_HISTORY_WORD.meanRedGreen];
        const blueNoise = words[base + PROBE_HISTORY_WORD.meanBlueNoise];
        const updates = words[base + PROBE_HISTORY_WORD.updates];
        snapshot.state.push(probes[probe].state);
        snapshot.mean.push([halfToFloat(redGreen & 0xffff), halfToFloat(redGreen >>> 16), halfToFloat(blueNoise & 0xffff)]);
        snapshot.pendingChange.push(floats[base + PROBE_HISTORY_WORD.pendingChange]);
        snapshot.updates.push(updates & PROBE_HISTORY_UPDATES.countMask);
        snapshot.owed.push((updates >>> PROBE_HISTORY_UPDATES.owedShift) & PROBE_HISTORY_UPDATES.owedMask);
        snapshot.flags.push(words[base + PROBE_HISTORY_WORD.scheduleAndFlags] & PROBE_HISTORY_FLAGS_MASK);
        snapshot.moverMask.push(words[base + PROBE_HISTORY_WORD.moverMask]);
        snapshot.wakeUntil.push(words[base + PROBE_HISTORY_WORD.wakeUntil]);
        const grid: Vec3 = [probe % nx, Math.floor(probe / (nx * nz)), Math.floor(probe / nx) % nz];
        const offset = probes[probe].offset;
        snapshot.position.push([0, 1, 2].map((axis) => field.grid.origin[axis] + (grid[axis] + offset[axis]) * field.grid.spacing[axis]) as Vec3);
    }
    return snapshot;
}

/** The largest channel's relative change of a probe's mean (T2.4a's measure, ddgi-change-step). */
export function relativeChange(before: Vec3, after: Vec3): number {
    let largest = 0;
    for (let channel = 0; channel < 3; channel++) largest = Math.max(largest, Math.abs(after[channel] - before[channel]) / Math.max(before[channel], 1e-6));
    return largest;
}

/** A probe "really changed" by more than this: the 5% tolerance (T2.4a's check 4, T2.M1 item 8). */
export const TRUE_CHANGE = 0.05;

export interface MarkRecall {
    /** Active probes (before and after) whose light really changed by more than TRUE_CHANGE. */
    readonly changed: number;
    /** Probes whose marks added up to more than the tolerance (so their history was cut). */
    readonly marked: number;
    /** Changed and marked. */
    readonly caught: number;
    readonly recall: number;
    readonly precision: number;
    /** The same split by where the probe is: near the object's boxes (within `nearSpacings`) or farther. */
    readonly near: { readonly changed: number; readonly caught: number };
    readonly far: { readonly changed: number; readonly caught: number; readonly marked: number };
    /** The probes that changed but weren't marked, for a closer look. */
    readonly missed: readonly number[];
}

/**
 * The marks (`pendingChange` read before any blend took them in) against the
 * probes that really changed between two converged fields, over probes
 * active in both. `isNear` says which probes are in the near set.
 */
export function markRecall(marks: ProbeSnapshot, before: ProbeSnapshot, after: ProbeSnapshot, tolerance: number, isNear: (probe: number) => boolean): MarkRecall {
    let changed = 0;
    let marked = 0;
    let caught = 0;
    const near = { changed: 0, caught: 0 };
    const far = { changed: 0, caught: 0, marked: 0 };
    const missed: number[] = [];
    for (let probe = 0; probe < marks.pendingChange.length; probe++) {
        if (before.state[probe] !== PROBE_STATE.active || after.state[probe] !== PROBE_STATE.active) continue;
        const reallyChanged = relativeChange(before.mean[probe], after.mean[probe]) > TRUE_CHANGE;
        const isMarked = marks.pendingChange[probe] > tolerance;
        const side = isNear(probe) ? near : far;
        if (reallyChanged) {
            changed++;
            side.changed++;
        }
        if (isMarked) {
            marked++;
            if (side === far) far.marked++;
        }
        if (reallyChanged && isMarked) {
            caught++;
            side.caught++;
        }
        if (reallyChanged && !isMarked) missed.push(probe);
    }
    return { changed, marked, caught, recall: changed > 0 ? caught / changed : 1, precision: marked > 0 ? caught / marked : 1, near, far, missed };
}

/**
 * The missed probes by how far they are from the nearest of some boxes, in
 * spacings (the largest axis's gap over that axis's spacing): bands up to
 * the near set's 1.5, 4, the far set's 8, and beyond it.
 */
export function missedByDistance(probes: readonly number[], positions: readonly Vec3[], boxes: readonly { min: readonly number[]; max: readonly number[] }[], spacing: readonly number[]): Record<string, number> {
    const bands: Record<string, number> = { 'within 1.5': 0, '1.5-4': 0, '4-8': 0, 'beyond 8': 0 };
    for (const probe of probes) {
        let nearest = Infinity;
        for (const box of boxes) {
            let gap = 0;
            for (let axis = 0; axis < 3; axis++) {
                const outside = Math.max(box.min[axis] - positions[probe][axis], positions[probe][axis] - box.max[axis], 0);
                gap = Math.max(gap, outside / spacing[axis]);
            }
            nearest = Math.min(nearest, gap);
        }
        const band = nearest <= 1.5 ? 'within 1.5' : nearest <= 4 ? '1.5-4' : nearest <= 8 ? '4-8' : 'beyond 8';
        bands[band]++;
    }
    return bands;
}

/** The mean over pixels of |image − reference|, over the mean of the reference (T2.4b's per-pixel error). */
export function perPixelError(image: Float32Array, reference: Float32Array): number {
    let difference = 0;
    let sum = 0;
    for (let index = 0; index < image.length; index++) {
        difference += Math.abs(image[index] - reference[index]);
        sum += reference[index];
    }
    return sum > 0 ? difference / sum : 0;
}

/** Whether a point lies within `margin` of an axis-aligned box. */
export function nearBox(point: readonly number[], min: readonly number[], max: readonly number[], margin: readonly number[]): boolean {
    for (let axis = 0; axis < 3; axis++) {
        if (point[axis] < min[axis] - margin[axis] || point[axis] > max[axis] + margin[axis]) return false;
    }
    return true;
}
