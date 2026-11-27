/**
 * The stats line for the radiance-cache example: the screen probes, the
 * memory GI holds, how much light the cache gives the screen, and where the
 * world cache's second bounce comes from. The same numbers go to
 * `window.ddgiExample` (status.ts), as the DDGI examples' stats line
 * (stats.ts) does.
 *
 * Everything comes from the public API: `gi.stats` for frames and memory,
 * and `cache.probes` for the irradiance image the composite reads. The mean
 * is one GPU readback, so it runs only every few seconds. There is no GPU
 * time: from three.js, `gi.stats.passes` doesn't hold the cache's
 * screen-stage passes (they run inside three's render, after the platform's
 * timestamps were resolved), so their sum would leave out most of its cost.
 */

import { halfToFloat, type GIPlatform } from '@pmndrs/gi/core';
import type { RadianceCacheTechnique } from '@pmndrs/gi/radiance-cache';
import { status } from './status.ts';

const TEXT_INTERVAL_MS = 500;
const GI_MEAN_INTERVAL_MS = 2000;
/** Bytes per texel of the cache's `rgba16float` irradiance. */
const TEXEL_BYTES = 8;

/** Rec. 709 luminance of linear RGB. */
function luminance(red: number, green: number, blue: number): number {
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue;
}

/**
 * The mean irradiance the composite reads (the denoiser's output, or the
 * interpolated irradiance with the denoiser off), as luminance, over the
 * pixels that have any: the sky has none. Null before the first frame.
 */
async function measureGIMean(device: GPUDevice, cache: RadianceCacheTechnique): Promise<number | null> {
    const probes = cache.probes;
    const texture = probes?.denoisedIrradiance ?? probes?.irradiance;
    if (!texture) return null;
    const { width, height } = texture;
    const bytesPerRow = Math.ceil((width * TEXEL_BYTES) / 256) * 256;
    const buffer = device.createBuffer({ label: 'example: irradiance readback', size: bytesPerRow * height, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
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

/** Shows the stats line for the radiance cache on `gi` and keeps it current. `note` adds text at the end. */
export function showCacheStats(device: GPUDevice, gi: GIPlatform, cache: RadianceCacheTechnique, note?: () => string): void {
    const line = document.createElement('div');
    line.id = 'stats';
    line.style.cssText =
        'position:fixed;left:0;bottom:0;padding:5px 10px;z-index:10;background:rgba(8,10,14,0.75);' +
        'color:#cfd8e3;font:12px/1.5 ui-monospace,Menlo,Consolas,monospace;pointer-events:none;';
    document.body.append(line);

    const update = (): void => {
        const stats = gi.stats;
        status.frames = stats.frame;
        const layout = cache.probes?.layout;
        const parts = [
            layout ? `GI: up to ${layout.maxProbeSpawnCount.toLocaleString('en')} screen probes a frame` : 'GI starting…',
            `${(stats.memory.total / 2 ** 20).toFixed(0)} MB`,
            status.giMean === null ? 'mean irradiance …' : `mean irradiance ${status.giMean.toFixed(3)}`,
            `frame ${status.frames.toLocaleString('en')}`,
        ];
        const extra = note?.();
        if (extra) parts.push(extra);
        line.textContent = parts.join(' · ');
    };
    setInterval(update, TEXT_INTERVAL_MS);
    update();

    let measuring = false;
    setInterval(() => {
        if (measuring) return;
        measuring = true;
        measureGIMean(device, cache)
            .then((mean) => {
                status.giMean = mean;
            })
            .finally(() => {
                measuring = false;
            });
    }, GI_MEAN_INTERVAL_MS);
}
