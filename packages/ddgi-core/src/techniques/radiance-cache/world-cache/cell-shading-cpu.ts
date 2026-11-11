/**
 * CPU twins of lever 1's choice (`cellShading: 'budget'`, RC-L1;
 * `hash-grid-cell-shading-wgsl.ts`): the key a hit offers its cell, a cell's
 * age class, and which cells a frame lights. The unit tests and the
 * `radiance-cache-cell-shading` GPU page check the GPU against them.
 */

import { pcgHash } from './hash-grid-cpu.ts';

/** `HashGridCache_CellShadingKey`: the hit's index XORed with a hash of the frame, then lowbias32. Different for every index in a frame. */
export function cellShadingKey(visibilityIndex: number, frameIndex: number): number {
    let x = (visibilityIndex ^ pcgHash(frameIndex >>> 0)) >>> 0;
    x = (x ^ (x >>> 16)) >>> 0;
    x = Math.imul(x, 0x7feb352d) >>> 0;
    x = (x ^ (x >>> 15)) >>> 0;
    x = Math.imul(x, 0x846ca68b) >>> 0;
    x = (x ^ (x >>> 16)) >>> 0;
    return x;
}

/** The oldest age the ranking tells apart: classes 1 (this age or more) to 255 (lit last frame). */
export const CELL_SHADING_OLDEST_AGE = 255;

/** `HashGridCache_CellShadingAge`: frames since lit, at least 1 (`lastLit` is the frame plus one; wraps with the frame counter). */
export function cellShadingAge(lastLit: number, frameIndex: number): number {
    return Math.max((frameIndex + 1 - lastLit) >>> 0, 1);
}

/**
 * `HashGridCache_CellShadingClass`: 0 for never lit (`lastLit` 0), then by
 * age, oldest first: 1 for 255 frames or more, 255 for lit last frame.
 */
export function cellShadingClass(lastLit: number, frameIndex: number): number {
    if (lastLit === 0) return 0;
    return 1 + (CELL_SHADING_OLDEST_AGE - Math.min(cellShadingAge(lastLit, frameIndex), CELL_SHADING_OLDEST_AGE));
}

/** `HashGridCache_CellShadingIsStale`: never lit, or lit `maxAge` frames ago or more. */
export function cellShadingIsStale(lastLit: number, frameIndex: number, maxAge: number): boolean {
    return lastLit === 0 || cellShadingAge(lastLit, frameIndex) >= maxAge;
}

/** What `chooseCells` says about one frame's cells in use. */
export interface CellChoice {
    /** Cells lit: min(budget, cells in use). */
    readonly litCount: number;
    /** The class the budget runs out in (the last class with a lit cell), or -1 when nothing is lit. */
    readonly cutoffClass: number;
    /** Cells in use per class. */
    readonly histogram: readonly number[];
    /** Of the cutoff class, how many cells are lit (the rest of it waits). */
    readonly litInCutoffClass: number;
}

/**
 * Which cells the budget lights, given each cell in use's class: every cell
 * of the classes before the cutoff, and `litInCutoffClass` of the cutoff
 * class (which ones depends on the GPU's atomic order, so a check takes the
 * class, not the cell).
 */
export function chooseCells(classes: readonly number[], classCount: number, budget: number): CellChoice {
    const histogram = new Array<number>(classCount).fill(0);
    for (const value of classes) histogram[value]++;
    const litCount = Math.min(budget, classes.length);
    let remaining = litCount;
    let cutoffClass = -1;
    let litInCutoffClass = 0;
    for (let index = 0; index < classCount && remaining > 0; index++) {
        cutoffClass = index;
        litInCutoffClass = Math.min(remaining, histogram[index]);
        remaining -= litInCutoffClass;
    }
    return { litCount, cutoffClass, histogram, litInCutoffClass };
}
