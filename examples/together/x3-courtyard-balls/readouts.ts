/**
 * The page's own readings, for whichever technique runs now:
 *
 * - **the picture's mean brightness**, every frame, for the timeline: a few
 *   rows of the canvas as it was drawn, read back. It is the same reading
 *   for every technique, and it includes three's direct light and the
 *   balls' own glow, so a throw shows as a step and GI's catching up as the
 *   slope after it;
 * - **the GI mean** for `window.ddgiExample` (shared/status.ts), every two
 *   seconds: DDGI's active probes (`exportIrradianceSH`), or the
 *   irradiance the cache gives the composite (`cache.probes`), as the
 *   shared stats lines measure them;
 * - **the stats line**: the technique, GPU ms (the pass list's total),
 *   memory, the balls and the frame time.
 */

import { halfToFloat, type GIPlatform } from '@pmndrs/gi/core';
import { DDGITechnique, exportIrradianceSH, PROBE_STATE } from '@pmndrs/gi/ddgi';
import type { RadianceCacheTechnique } from '@pmndrs/gi/radiance-cache';
import { status } from '../../shared/status.ts';

/** Rec. 709 luminance. */
const luminance = (red: number, green: number, blue: number): number => 0.2126 * red + 0.7152 * green + 0.0722 * blue;
/** Mean irradiance over all directions per unit of a probe's first SH coefficient (shared/stats.ts). */
const SH_CONSTANT_TO_MEAN_IRRADIANCE = Math.PI * 0.282095;

/** DDGI: the mean irradiance of its active probes, or null before any is active. */
async function ddgiMean(technique: DDGITechnique): Promise<number | null> {
    const exported = await exportIrradianceSH(technique, { fillInactiveProbes: false });
    const perProbe = exported.coefficients.length / exported.probeStates.length;
    let sum = 0;
    let active = 0;
    exported.probeStates.forEach((state, probe) => {
        if (state !== PROBE_STATE.active) return;
        const first = probe * perProbe;
        sum += luminance(exported.coefficients[first], exported.coefficients[first + 1], exported.coefficients[first + 2]) * SH_CONSTANT_TO_MEAN_IRRADIANCE;
        active++;
    });
    return active > 0 ? sum / active : null;
}

/** The cache: the mean irradiance the composite reads, over the pixels that have any. */
async function cacheMean(device: GPUDevice, cache: RadianceCacheTechnique): Promise<number | null> {
    const texture = cache.probes?.denoisedIrradiance ?? cache.probes?.irradiance;
    if (!texture) return null;
    const { width, height } = texture;
    const bytesPerRow = Math.ceil((width * 8) / 256) * 256;
    const buffer = device.createBuffer({ label: 'x3: irradiance readback', size: bytesPerRow * height, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const encoder = device.createCommandEncoder();
    encoder.copyTextureToBuffer({ texture }, { buffer, bytesPerRow }, [width, height]);
    device.queue.submit([encoder.finish()]);
    try {
        await buffer.mapAsync(GPUMapMode.READ);
        const halves = new Uint16Array(buffer.getMappedRange());
        let sum = 0;
        let lit = 0;
        for (let y = 0; y < height; y++) {
            for (let x = 0; x < width; x++) {
                const first = (y * bytesPerRow) / 2 + x * 4;
                const value = luminance(halfToFloat(halves[first]), halfToFloat(halves[first + 1]), halfToFloat(halves[first + 2]));
                if (value > 0) {
                    sum += value;
                    lit++;
                }
            }
        }
        return lit > 0 ? sum / lit : 0;
    } finally {
        buffer.destroy();
    }
}

/** What the readouts need from the technique running now. */
export interface Running {
    readonly label: string;
    readonly platform: GIPlatform;
    /** The cache, when the technique is one of its variants. */
    readonly cache: RadianceCacheTechnique | null;
}

/** Measures the GI mean into `status.giMean` now. Resolves when done, so a technique switch can wait for it. */
export async function measureGIMean(device: GPUDevice, running: Running): Promise<void> {
    const output = running.platform.outputTechnique;
    const mean = running.cache ? await cacheMean(device, running.cache) : output instanceof DDGITechnique && output.field ? await ddgiMean(output) : null;
    if (mean !== null) status.giMean = mean;
}

/** Rows of the canvas read back each frame for the picture's mean. */
const ROWS = 12;
/** Readback buffers in flight at most; a frame with none free is a gap in the chart. */
const BUFFERS = 6;

/**
 * Reads the mean brightness of the canvas as drawn this frame (display
 * values, 0 to 1), from `ROWS` rows. Call right after the frame's render, in
 * the same task: the canvas's texture is the frame's until the task ends.
 */
export class PictureMean {
    private readonly context: GPUCanvasContext;
    private readonly pool: Array<{ buffer: GPUBuffer; busy: boolean; bytesPerRow: number }> = [];

    constructor(
        private readonly device: GPUDevice,
        canvas: HTMLCanvasElement,
    ) {
        const context = canvas.getContext('webgpu');
        if (!context) throw new Error('x3: the canvas has no WebGPU context.');
        this.context = context;
    }

    /** This frame's mean, or null when no buffer was free (or the format isn't one this reads). */
    read(): Promise<number | null> {
        const texture = this.context.getCurrentTexture();
        const bytesPerTexel = texture.format === 'rgba16float' ? 8 : 4;
        const bgra = texture.format.startsWith('bgra');
        const bytesPerRow = Math.ceil((texture.width * bytesPerTexel) / 256) * 256;
        let slot = this.pool.find((entry) => !entry.busy && entry.bytesPerRow === bytesPerRow);
        if (!slot && this.pool.length < BUFFERS) {
            slot = { buffer: this.device.createBuffer({ label: 'x3: picture mean', size: bytesPerRow * ROWS, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ }), busy: false, bytesPerRow };
            this.pool.push(slot);
        }
        if (!slot) {
            // A resize left buffers of the old size: drop the idle ones.
            for (const entry of this.pool.filter((candidate) => !candidate.busy)) {
                entry.buffer.destroy();
                this.pool.splice(this.pool.indexOf(entry), 1);
            }
            return Promise.resolve(null);
        }
        const { width, height } = texture;
        const encoder = this.device.createCommandEncoder();
        for (let row = 0; row < ROWS; row++) {
            const y = Math.floor(((row + 0.5) / ROWS) * height);
            encoder.copyTextureToBuffer({ texture, origin: [0, y] }, { buffer: slot.buffer, offset: row * bytesPerRow, bytesPerRow }, [width, 1]);
        }
        this.device.queue.submit([encoder.finish()]);
        const taken = slot;
        taken.busy = true;
        return taken.buffer.mapAsync(GPUMapMode.READ).then(
            () => {
                let sum = 0;
                if (bytesPerTexel === 8) {
                    const halves = new Uint16Array(taken.buffer.getMappedRange());
                    for (let row = 0; row < ROWS; row++) {
                        for (let x = 0; x < width; x++) {
                            const first = (row * bytesPerRow) / 2 + x * 4;
                            sum += luminance(halfToFloat(halves[first]), halfToFloat(halves[first + 1]), halfToFloat(halves[first + 2]));
                        }
                    }
                } else {
                    const bytes = new Uint8Array(taken.buffer.getMappedRange());
                    for (let row = 0; row < ROWS; row++) {
                        for (let x = 0; x < width; x++) {
                            const first = row * bytesPerRow + x * 4;
                            const [red, blue] = bgra ? [bytes[first + 2], bytes[first]] : [bytes[first], bytes[first + 2]];
                            sum += luminance(red, bytes[first + 1], blue) / 255;
                        }
                    }
                }
                taken.buffer.unmap();
                taken.busy = false;
                return sum / (ROWS * width);
            },
            () => {
                taken.busy = false;
                return null;
            },
        );
    }
}

/** The stats line at the bottom: what runs, its cost, the balls, and this page's frame time. */
export function showStatsLine(read: () => { running: Running | null; gpuMs: number | null; balls: number; moving: number; frameMs: number; place: string; pixels: string }): void {
    const line = document.createElement('div');
    line.id = 'stats';
    line.style.cssText =
        'position:fixed;left:0;bottom:0;padding:5px 10px;z-index:10;background:rgba(8,10,14,0.75);' +
        'color:#cfd8e3;font:12px/1.5 ui-monospace,Menlo,Consolas,monospace;pointer-events:none;';
    document.body.append(line);
    let framesBefore = 0;
    let shown: GIPlatform | null = null;
    setInterval(() => {
        const { running, gpuMs, balls, moving, frameMs, place, pixels } = read();
        if (running?.platform !== shown) {
            if (shown) framesBefore += shown.stats.frame;
            shown = running?.platform ?? null;
        }
        if (running) status.frames = framesBefore + running.platform.stats.frame;
        const parts = [
            running ? `GI: ${running.label}` : 'GI starting…',
            gpuMs === null ? 'GPU time not measured' : `${gpuMs.toFixed(2)} ms GPU`,
            running ? `${(running.platform.stats.memory.total / 2 ** 20).toFixed(0)} MB` : '',
            `${balls} balls (${moving} moving)`,
            `${place}`,
            `${pixels} px`,
            `frame ${frameMs.toFixed(1)} ms`,
        ];
        line.textContent = parts.filter(Boolean).join(' · ');
    }, 500);
}
