/**
 * The neural tail's training-step count, read back from the GPU without
 * waiting (P6.4, register E11): the gate opens on the steps the GPU took, not
 * the steps encoded (a step with no records is skipped there). One small
 * buffer: a copy of the Adam step is encoded into it once a frame while it is
 * free, mapped at the next frame (its frame is submitted by then), and the
 * value lands a frame or two after the copy.
 *
 * A restart (fresh weights: `reset`, the box centred again) bumps the
 * generation, and a value is kept only if its copy was encoded in the
 * current generation: a copy of the old network's count, mapped after the
 * restart, would otherwise open the gate on fresh weights (P6.4's review).
 *
 * GPU-free (a structural buffer type), so the unit tests drive it.
 */

/** The parts of `GPUBuffer` the readback uses. */
export interface StepReadbackBuffer {
    mapAsync(mode: number): Promise<void>;
    getMappedRange(): ArrayBuffer;
    unmap(): void;
}

/** `GPUMapMode.READ` (the global doesn't exist in Node, where the tests run). */
const MAP_MODE_READ = 0x1;

export class StepReadback {
    /** Training steps the GPU took in the current generation, as last read back. */
    steps = 0;
    /** `copied` once a copy is encoded, `mapping` once its map is asked for. */
    private state: 'idle' | 'copied' | 'mapping' = 'idle';
    private generation = 0;
    /** The generation the pending copy was encoded in. */
    private copyGeneration = 0;

    constructor(private readonly buffer: StepReadbackBuffer) {}

    /** The network started over: steps from 0, and any copy taken before now is dropped when it lands. */
    restart(): void {
        this.generation++;
        this.steps = 0;
    }

    /** Whether a copy may be encoded into the buffer this frame (it isn't mapped or waiting). */
    get wantsCopy(): boolean {
        return this.state === 'idle';
    }

    /** A copy of the step count into the buffer has been encoded (call after encoding it). */
    copied(): void {
        this.state = 'copied';
        this.copyGeneration = this.generation;
    }

    /** At a frame's start: maps the copy encoded last frame; its value is kept if no restart came since the copy. */
    poll(): void {
        if (this.state !== 'copied') return;
        this.state = 'mapping';
        const generation = this.copyGeneration;
        const buffer = this.buffer;
        buffer.mapAsync(MAP_MODE_READ).then(
            () => {
                const steps = new Uint32Array(buffer.getMappedRange())[0] ?? 0;
                buffer.unmap();
                if (generation === this.generation) this.steps = steps;
                this.state = 'idle';
            },
            () => {
                // Destroyed or lost: nothing to read, and no copy is taken again.
            },
        );
    }
}
