/**
 * Lever 1's age watch (`cellShading: 'budget'`, RC-L1): the ranking tells
 * cells' ages apart up to 255 frames, so a budget can go round at most 255 ×
 * `cellShadingBudget` cells in use. Past that, the oldest cells share one rank
 * and some can wait indefinitely. The hash grid counts the cells in use last
 * lit 255 frames ago or more (`ageClampedCellCount`); this reads it back
 * every so often, without stalling (the copy is recorded in one frame's
 * encoder and mapped once the next frame starts, after the host submitted
 * it), and warns once when it isn't 0. No minimum budget is enforced.
 */

import type { GpuWord } from '../../platform/contract/compute.ts';

/** `GPUBufferUsage` and `GPUMapMode` bits (the globals don't exist in Node, where the tests import this file). */
const BUFFER_USAGE = { mapRead: 0x1, copyDst: 0x8 } as const;
const MAP_MODE_READ = 0x1;

/** Frames between readings. */
const READ_EVERY_FRAMES = 60;

export class CellShadingAgeWatch {
    /** The latest reading: cells in use last lit 255 frames ago or more, or null before the first. */
    agedCells: number | null = null;
    private staging: GPUBuffer | null = null;
    private copied = false;
    private mapping = false;
    private warned = false;
    private framesSinceRead = READ_EVERY_FRAMES;

    constructor(
        private readonly device: GPUDevice,
        private readonly warn: (message: string) => void,
    ) {}

    /**
     * Call once per frame after the world cache's work is recorded into
     * `encoder`, with the counter word and the budget (for the message).
     */
    record(encoder: GPUCommandEncoder, counter: GpuWord, budget: number): void {
        // Last frame's copy has been submitted by now (encodeScreen's contract): map it.
        if (this.copied && !this.mapping && this.staging) {
            const staging = this.staging;
            this.mapping = true;
            this.copied = false;
            staging.mapAsync(MAP_MODE_READ).then(
                () => {
                    const value = new Uint32Array(staging.getMappedRange())[0];
                    staging.unmap();
                    this.mapping = false;
                    this.agedCells = value;
                    if (value > 0 && !this.warned) {
                        this.warned = true;
                        this.warn(
                            `@pmndrs/ddgi: cellShadingBudget ${budget} is too small to go round the radiance cache's cells in use: ${value} were last lit 255 frames ago or more, ` +
                                'where the ranking stops telling ages apart, so some may wait indefinitely. Raise the budget to at least the cells in use / 255.',
                        );
                    }
                },
                () => {
                    // The device went away or the buffer was destroyed: stop reading.
                    this.mapping = false;
                },
            );
        }
        this.framesSinceRead++;
        if (this.framesSinceRead < READ_EVERY_FRAMES || this.mapping || this.copied) return;
        this.framesSinceRead = 0;
        this.staging ??= this.device.createBuffer({ label: 'radiance-cache/cell shading age watch', size: 4, usage: BUFFER_USAGE.mapRead | BUFFER_USAGE.copyDst });
        encoder.copyBufferToBuffer(counter.buffer, counter.offset, this.staging, 0, 4);
        this.copied = true;
    }

    dispose(): void {
        this.staging?.destroy();
        this.staging = null;
    }
}
