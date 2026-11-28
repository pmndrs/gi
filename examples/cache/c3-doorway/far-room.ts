/**
 * The far room's GI, as a number and a bar for each second-bounce source:
 * the mean irradiance the composite reads (the cache's denoised E, as
 * luminance) over the pixels that show the far room. Pixels seen through the
 * doorway show the lit room (or the door), so they are left out: a pixel is
 * one of them when its view ray crosses the partition's far-room face inside
 * the doorway. That holds while the camera is in the far room, which is why
 * main.ts keeps it still.
 *
 * Each source keeps its last reading per door state, so the bars compare
 * them after you switch. The reading is one GPU readback every half second.
 * It goes to `window.ddgiExample` too (status.ts), with the frame count.
 */

import type * as THREE from 'three/webgpu';
import { halfToFloat, type GIPlatform } from '@pmndrs/gi/core';
import type { RadianceCacheTechnique } from '@pmndrs/gi/radiance-cache';
import { DOORWAY_HEIGHT_M, DOORWAY_Z0_M, DOORWAY_Z1_M, PARTITION_THICKNESS_M, ROOM_SIZE_M } from '../../../scenes/src/procedural/two-rooms.ts';
import { floatingDock } from '../../shared/dock.ts';
import { status } from '../../shared/status.ts';
import './far-room.css';

/** Where the world cache's second bounce comes from. */
export type Bounce = 'ddgi' | 'gi1-rays' | 'sharc';

const SOURCES: ReadonlyArray<{ readonly bounce: Bounce; readonly label: string }> = [
    { bounce: 'ddgi', label: 'DDGI underneath' },
    { bounce: 'gi1-rays', label: "GI-1.2's own rays" },
    { bounce: 'sharc', label: 'SHaRC' },
];

/** The partition's face on the far room's side. */
const FAR_FACE_X_M = ROOM_SIZE_M + PARTITION_THICKNESS_M;
const READ_INTERVAL_MS = 500;
/** Bytes per texel of the cache's `rgba16float` irradiance. */
const TEXEL_BYTES = 8;

/** The mean luminance of `texture` over the far room's pixels, as `camera` sees them. Null before the first frame. */
async function measureFarRoom(device: GPUDevice, texture: GPUTexture, camera: THREE.PerspectiveCamera): Promise<number | null> {
    const { width, height } = texture;
    const bytesPerRow = Math.ceil((width * TEXEL_BYTES) / 256) * 256;
    const buffer = device.createBuffer({ label: 'C3: far-room readback', size: bytesPerRow * height, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const encoder = device.createCommandEncoder();
    encoder.copyTextureToBuffer({ texture }, { buffer, bytesPerRow }, [width, height]);
    device.queue.submit([encoder.finish()]);
    // The view ray of NDC (x, y): forward + x·right + y·up, scaled by the field of view.
    const m = camera.matrixWorld.elements;
    const tanY = Math.tan((camera.fov * Math.PI) / 360);
    const tanX = tanY * camera.aspect;
    const [ox, oy, oz] = [m[12], m[13], m[14]];
    try {
        await buffer.mapAsync(GPUMapMode.READ);
        const halves = new Uint16Array(buffer.getMappedRange());
        let sum = 0;
        let count = 0;
        for (let y = 0; y < height; y += 2) {
            const ny = (1 - (2 * (y + 0.5)) / height) * tanY;
            for (let x = 0; x < width; x += 2) {
                const nx = ((2 * (x + 0.5)) / width - 1) * tanX;
                const dx = -m[8] + nx * m[0] + ny * m[4];
                if (dx < 0) {
                    // Where the ray crosses the far room's side of the partition: inside the doorway, it sees the lit room.
                    const t = (FAR_FACE_X_M - ox) / dx;
                    const hitY = oy + t * (-m[9] + nx * m[1] + ny * m[5]);
                    const hitZ = oz + t * (-m[10] + nx * m[2] + ny * m[6]);
                    if (t > 0 && hitY < DOORWAY_HEIGHT_M && hitZ > DOORWAY_Z0_M && hitZ < DOORWAY_Z1_M) continue;
                }
                const first = (y * bytesPerRow) / 2 + x * 4;
                const value = 0.2126 * halfToFloat(halves[first]) + 0.7152 * halfToFloat(halves[first + 1]) + 0.0722 * halfToFloat(halves[first + 2]);
                if (!Number.isFinite(value)) continue;
                sum += value;
                count++;
            }
        }
        return count > 0 ? sum / count : null;
    } finally {
        buffer.destroy();
    }
}

export interface FarRoom {
    /** Call after a change (source or door): the next readings belong to the new state. */
    restart(): void;
}

/** Shows the far room's readings, bottom right, and keeps them current. */
export function showFarRoom(
    device: GPUDevice,
    camera: THREE.PerspectiveCamera,
    cache: () => RadianceCacheTechnique | null,
    gi: () => GIPlatform | null,
    state: { readonly bounce: Bounce; readonly open: boolean },
): FarRoom {
    const panel = document.createElement('div');
    panel.className = 'far-room floating-panel';
    const header = document.createElement('div');
    header.className = 'far-room-header';
    const title = document.createElement('strong');
    const frames = document.createElement('span');
    header.append(title, frames);
    const rows = SOURCES.map(({ bounce, label }) => {
        const row = document.createElement('div');
        row.className = 'far-room-row';
        const name = document.createElement('span');
        name.textContent = label;
        const track = document.createElement('span');
        track.className = 'far-room-track';
        const bar = document.createElement('span');
        bar.className = 'far-room-bar';
        track.append(bar);
        const value = document.createElement('span');
        value.className = 'far-room-value';
        row.append(name, track, value);
        panel.append(row);
        return { bounce, row, bar, value };
    });
    const note = document.createElement('div');
    note.className = 'far-room-note';
    note.textContent = "This page's own reading: the mean GI irradiance over the far room's pixels, the doorway left out. The cache takes a few seconds to settle after a change; % is against DDGI's reading.";
    panel.prepend(header);
    panel.append(note);
    floatingDock().append(panel);

    /** The last reading of each source, per door state. */
    const readings = new Map<string, number>();
    const key = (bounce: Bounce, open: boolean) => `${bounce}/${open ? 'open' : 'shut'}`;
    let generation = 0;
    let startFrame = 0;

    const draw = (): void => {
        const platform = gi();
        status.frames = platform?.stats.frame ?? status.frames;
        title.textContent = `Far room's GI, door ${state.open ? 'open' : 'shut'}`;
        frames.textContent = platform ? `${Math.max(0, platform.stats.frame - startFrame)} frames since the change · ${(platform.stats.memory.total / 2 ** 20).toFixed(0)} MB` : 'starting…';
        const values = rows.map(({ bounce }) => readings.get(key(bounce, state.open)));
        const scale = Math.max(...values.map((value) => value ?? 0)) || 1;
        const ddgiValue = values[0];
        for (const [index, { bounce, row, bar, value }] of rows.entries()) {
            const reading = values[index];
            row.classList.toggle('active', bounce === state.bounce);
            bar.style.width = `${(100 * (reading ?? 0)) / scale}%`;
            const against = reading !== undefined && ddgiValue && bounce !== 'ddgi' ? ` (${reading >= ddgiValue ? '+' : '−'}${Math.abs(100 * (reading / ddgiValue - 1)).toFixed(0)}%)` : '';
            value.textContent = reading === undefined ? 'switch to read' : `${reading.toFixed(3)}${against}`;
        }
    };

    let reading = false;
    setInterval(() => {
        const texture = cache()?.probes?.denoisedIrradiance;
        if (reading || !texture) return draw();
        reading = true;
        const asked = generation;
        const at = key(state.bounce, state.open);
        measureFarRoom(device, texture, camera)
            .then((mean) => {
                if (mean === null || asked !== generation) return;
                readings.set(at, mean);
                status.giMean = mean;
            })
            .finally(() => {
                reading = false;
                draw();
            });
    }, READ_INTERVAL_MS);
    draw();

    return {
        restart() {
            generation++;
            startFrame = gi()?.stats.frame ?? 0;
            draw();
        },
    };
}
