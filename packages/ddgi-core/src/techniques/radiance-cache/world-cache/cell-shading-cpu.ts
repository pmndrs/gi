/**
 * CPU twins of lever E8's bookkeeping (`cell-shading-wgsl.ts`, cells lit on a
 * budget), for the unit tests and the GPU page's predictions: the priority
 * class a claim gives a cell, `PrioritizeCells`' counting sort and turn
 * period, `ScatterCells`' slots, and a model of one cell's value over frames,
 * per hit and budgeted, built on `UpdateTiles`' blend (`hash-grid-cpu.ts`).
 */

import { CELL_SHADING_SUB_BINS } from './capacity.ts';
import { CELL_PRIORITY_CLASS } from './cell-shading-wgsl.ts';
import { updateTilesAccumulate, type CellValue } from './hash-grid-cpu.ts';

/** The class `ClaimCells` gives a cell (lower is lit first). */
export function cellPriorityClass(cell: { readonly count: number; readonly changed: boolean; readonly turn: boolean }, maxSampleCount: number): number {
    if (cell.changed || cell.count <= 0) return CELL_PRIORITY_CLASS.newOrChanged;
    if (cell.count < maxSampleCount) return CELL_PRIORITY_CLASS.converging;
    return cell.turn ? CELL_PRIORITY_CLASS.turn : CELL_PRIORITY_CLASS.rest;
}

/** What `PrioritizeCells` writes: each bin's first slot, the cells lit this frame, and the next frame's turn period. */
export interface PrioritizedCells {
    readonly firstSlots: readonly number[];
    readonly shaded: number;
    readonly period: number;
}

/** `PrioritizeCells`: an exclusive prefix sum of the bins' counts; min(candidates, budget); the settled cells over what the urgent ones leave. */
export function prioritizeCells(binCounts: readonly number[], budget: number, subBins = CELL_SHADING_SUB_BINS): PrioritizedCells {
    const firstSlots: number[] = [];
    let slot = 0;
    let urgent = 0;
    binCounts.forEach((count, bin) => {
        firstSlots.push(slot);
        slot += count;
        if (bin < CELL_PRIORITY_CLASS.turn * subBins) urgent += count;
    });
    const candidates = slot;
    const settled = candidates - urgent;
    const left = Math.max(budget - Math.min(budget, urgent), 1);
    return { firstSlots, shaded: Math.min(candidates, budget), period: Math.max(Math.ceil(settled / left), 1) };
}

/**
 * `ScatterCells`, the candidates taken in the order given (on the GPU, the
 * order the atomics happen to run): each takes its bin's next slot, and is lit
 * when that slot is under the frame's lit count. Returns each candidate's slot
 * and whether it is lit.
 */
export function scatterCells(candidateBins: readonly number[], prioritized: PrioritizedCells): { readonly slot: number; readonly lit: boolean }[] {
    const next = [...prioritized.firstSlots];
    return candidateBins.map((bin) => {
        const slot = next[bin]++;
        return { slot, lit: slot < prioritized.shaded };
    });
}

/** `HashGridCache_KeepOneSample`: a value's history cut to one sample, its mean kept (a value with at most one sample stays). */
export function keepOneSample(value: CellValue): CellValue {
    if (!(value[3] > 1)) return value;
    return [value[0] / value[3], value[1] / value[3], value[2] / value[3], 1];
}

/** One cell's hits in a frame of the model: the light one light sample of each hit would give (a sample of the hit's light). */
export type HitSampler = (frame: number) => readonly number[];

/** What `simulateCell` reports: the cell's mean (rgb / count) after each frame. */
export interface CellHistory {
    readonly perHit: readonly number[];
    readonly budgeted: readonly number[];
}

/**
 * One cell over `frames` frames, in one channel: per hit, every hit's sample
 * goes into the frame's scratch (their sum and count) and `UpdateTiles`
 * blends it in; budgeted, on a frame the cell is lit (`lit(frame)`), one hit
 * drawn uniformly (`pick`, a number in [0, 1)) gives the frame's one sample,
 * and on other frames the value is kept. `UpdateTiles`' blend in both, with
 * the cap `maxCount`. A frame with no hits leaves both alone.
 */
export function simulateCell(options: {
    readonly frames: number;
    readonly hits: HitSampler;
    readonly lit: (frame: number) => boolean;
    readonly pick: (frame: number) => number;
    readonly maxCount: number;
}): CellHistory {
    let perHit: CellValue = [0, 0, 0, 0];
    let budgeted: CellValue = [0, 0, 0, 0];
    const perHitMeans: number[] = [];
    const budgetedMeans: number[] = [];
    for (let frame = 0; frame < options.frames; frame++) {
        const samples = options.hits(frame);
        if (samples.length > 0) {
            const sum = samples.reduce((total, value) => total + value, 0);
            perHit = updateTilesAccumulate(perHit, [sum, 0, 0, samples.length], options.maxCount);
            if (options.lit(frame)) {
                const drawn = samples[Math.min(Math.floor(options.pick(frame) * samples.length), samples.length - 1)];
                budgeted = updateTilesAccumulate(budgeted, [drawn, 0, 0, 1], options.maxCount);
            }
        }
        perHitMeans.push(perHit[0] / Math.max(perHit[3], 1));
        budgetedMeans.push(budgeted[0] / Math.max(budgeted[3], 1));
    }
    return { perHit: perHitMeans, budgeted: budgetedMeans };
}
