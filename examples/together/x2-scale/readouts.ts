/**
 * X2's live readings: for each method, as you fly, its GPU time a frame (the
 * shared pass list's mean over 30 frames), the memory it holds, and the
 * frame time. A method keeps its last reading after you switch away, so the
 * two sit side by side. These are this page's own measurements on your GPU,
 * not numbers from docs/.
 *
 * The stats line at the bottom and `window.ddgiExample` (status.ts) carry
 * the frame count and the mean irradiance GI gives, as on every page: for
 * DDGI the mean over the nearest volume's probes (`exportIrradianceSH`), for
 * the cache the mean of the irradiance its composite reads. Both are small
 * GPU readbacks, so they run every two seconds.
 */

import { halfToFloat, type GIPlatform } from '@pmndrs/gi/core';
import { DDGITechnique, exportIrradianceSH, PROBE_STATE } from '@pmndrs/gi/ddgi';
import type { RadianceCacheTechnique } from '@pmndrs/gi/radiance-cache';
import { showPassTiming } from '../../shared/pass-timing.ts';
import { status } from '../../shared/status.ts';

/** What the page is running now. */
export interface Running {
    /** The row it reads into, e.g. "DDGI · 2 volumes". */
    readonly label: string;
    /** A line of detail: probes, spacings, or the cache's table. */
    readonly detail: string;
    readonly gi: GIPlatform;
    readonly technique: DDGITechnique | RadianceCacheTechnique;
}

const luminance = (red: number, green: number, blue: number): number => 0.2126 * red + 0.7152 * green + 0.0722 * blue;
const mib = (bytes: number): string => `${Math.round(bytes / 2 ** 20).toLocaleString('en')} MiB`;

/** DDGI: the mean irradiance of the nearest volume's probes in use (stats.ts's measure). */
async function ddgiMean(technique: DDGITechnique): Promise<number | null> {
    const exported = await exportIrradianceSH(technique, { fillInactiveProbes: false });
    const floats = exported.coefficients.length / exported.probeStates.length;
    let sum = 0;
    let used = 0;
    for (let probe = 0; probe < exported.probeStates.length; probe++) {
        if (exported.probeStates[probe] !== PROBE_STATE.active) continue;
        const first = probe * floats; // the constant term's red, green and blue; π Y₀₀ turns it into mean irradiance
        sum += luminance(exported.coefficients[first], exported.coefficients[first + 1], exported.coefficients[first + 2]) * Math.PI * 0.282095;
        used++;
    }
    return used > 0 ? sum / used : null;
}

/** The cache: the mean of the irradiance its composite reads, over the pixels that have any (cache-stats.ts's measure). */
async function cacheMean(device: GPUDevice, cache: RadianceCacheTechnique): Promise<number | null> {
    const texture = cache.probes?.denoisedIrradiance ?? cache.probes?.irradiance;
    if (!texture) return null;
    const bytesPerRow = Math.ceil((texture.width * 8) / 256) * 256; // rgba16float
    const buffer = device.createBuffer({ label: 'x2: irradiance readback', size: bytesPerRow * texture.height, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const encoder = device.createCommandEncoder();
    encoder.copyTextureToBuffer({ texture }, { buffer, bytesPerRow }, [texture.width, texture.height]);
    device.queue.submit([encoder.finish()]);
    try {
        await buffer.mapAsync(GPUMapMode.READ);
        const halves = new Uint16Array(buffer.getMappedRange());
        let sum = 0;
        let lit = 0;
        for (let y = 0; y < texture.height; y++) {
            for (let x = 0; x < texture.width; x++) {
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

/**
 * Shows the readings for whatever `running` returns (null while the platform
 * is being made again), in a column on the right under `above` (the panel),
 * with the shared pass list under them.
 */
export function showReadouts(device: GPUDevice, running: () => Running | null, above: HTMLElement): void {
    const column = document.createElement('div');
    column.className = 'x2-column';
    const box = document.createElement('section');
    box.className = 'x2-readouts floating-panel';
    const passes = document.createElement('div');
    passes.className = 'floating-panel';
    column.append(box, passes);
    document.body.append(column);
    new ResizeObserver(() => (column.style.top = `${above.getBoundingClientRect().bottom + 8}px`)).observe(above);
    const timing = showPassTiming(() => running()?.gi ?? null, { container: passes });

    const line = document.createElement('div');
    line.id = 'stats';
    line.className = 'x2-stats';
    document.body.append(line);

    // The frame time, from the browser's own frames: what flying feels like.
    const frameMs: number[] = [];
    let last = performance.now();
    const tick = (now: number) => {
        frameMs.push(now - last);
        if (frameMs.length > 60) frameMs.shift();
        last = now;
        requestAnimationFrame(tick);
    };
    requestAnimationFrame(tick);

    const readings = new Map<string, { gpuMs: number | null; memory: number; detail: string }>();
    let framesBefore = 0;
    let shown: GIPlatform | null = null;
    const update = () => {
        const now = running();
        const meanFrame = frameMs.length ? frameMs.reduce((a, b) => a + b, 0) / frameMs.length : 0;
        if (now && now.gi !== shown) {
            if (shown) framesBefore += shown.stats.frame;
            shown = now.gi;
        }
        if (now) {
            status.frames = framesBefore + now.gi.stats.frame;
            const memory = now.gi.stats.memory.techniques[now.technique.id] ?? 0;
            // The pass list's mean needs a few timed frames of this platform before it is its own.
            const gpuMs = timing.rows.length > 0 ? timing.totalMs : null;
            // The latest last; the four most recent kept.
            readings.delete(now.label);
            readings.set(now.label, { gpuMs, memory, detail: now.detail });
            for (const label of [...readings.keys()].slice(0, -4)) readings.delete(label);
        }
        box.replaceChildren();
        const title = document.createElement('div');
        title.className = 'x2-readouts-title';
        title.textContent = 'Per method, as you fly (this page, your GPU)';
        box.append(title);
        for (const [label, reading] of readings) {
            const row = document.createElement('div');
            row.className = now?.label === label ? 'x2-reading current' : 'x2-reading';
            const name = document.createElement('div');
            name.textContent = label;
            const numbers = document.createElement('div');
            numbers.className = 'x2-reading-numbers';
            numbers.textContent = `${reading.gpuMs === null ? '… ms' : `${reading.gpuMs.toFixed(2)} ms`} GPU · ${mib(reading.memory)}`;
            const detail = document.createElement('div');
            detail.className = 'x2-reading-detail';
            detail.textContent = reading.detail;
            row.append(name, numbers, detail);
            box.append(row);
        }
        line.textContent = [
            now ? now.label : 'making the platform…',
            `frame ${meanFrame.toFixed(1)} ms`,
            now ? `GI memory ${mib(now.gi.stats.memory.total)}` : '',
            status.giMean === null ? 'mean irradiance …' : `mean irradiance ${status.giMean.toFixed(3)}`,
            `GI frame ${status.frames.toLocaleString('en')}`,
        ]
            .filter(Boolean)
            .join(' · ');
    };
    setInterval(update, 500);
    update();

    let measuring = false;
    setInterval(() => {
        const now = running();
        if (measuring || !now) return;
        measuring = true;
        const { technique } = now;
        (technique instanceof DDGITechnique ? ddgiMean(technique) : cacheMean(device, technique))
            .then((mean) => {
                if (mean !== null) status.giMean = mean;
            })
            .catch(() => {}) // a readback cut short by a switch; the next one reads the new platform
            .finally(() => (measuring = false));
    }, 2000);
}
