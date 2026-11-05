/**
 * T1.9 memory per scene: what the library owns on the GPU, per scene at the
 * manifest's own volume. Each scene is started as the viewer starts it (files
 * loaded, scene copied, mesh trees built, a few frames run), then
 * `gi.stats.memory` is read: scene copy, acceleration (trees), each
 * technique's own (DDGI: probe textures, the ray buffer, update lists), and
 * the total.
 *
 * `memory.total` leaves out one thing the core owns, the shading module's
 * sky (radiance map, and the procedural sky's source texture); this page
 * reads it separately (`platform.shade.memoryBytes`) and reports
 * "everything" = total + that. SPEC's target (64 MB for a single
 * furnished-room volume, everything the core owns) is checked against it.
 * These are byte counts of the resources the core created, not driver
 * numbers; they do not depend on the GPU, so one run per GPU is plenty.
 *
 *   VITE_DDGI_SCENES=furnished-room,sponza   (default: the six T1.9 scenes)
 *
 * Run under the bench lock (the GPU is busy while Sponza loads), then turn
 * the JSON into rows with to-csv.ts.
 */

import { assert } from '../../client.ts';
import { loadAllManifests } from '../../../../apps/viewer/src/scene/manifests.ts';
import { runThreePage } from '../three-shared/harness.ts';
import { startViewerScene } from '../ddgi-shared/viewer-scene.ts';
import { fullBrowserVersion } from '../trace-bench/run.ts';

const env = import.meta.env as Record<string, unknown>;
const DEFAULT_SCENES = ['cornell-box', 'two-room', 'furnished-room', 'sponza', 'bistro-interior', 'bistro-exterior'];
const SCENES = typeof env.VITE_DDGI_SCENES === 'string' && env.VITE_DDGI_SCENES.trim() !== '' ? env.VITE_DDGI_SCENES.split(',').map((item) => item.trim()) : DEFAULT_SCENES;

/** SPEC: 64 MB for everything the core owns, for a single furnished-room volume. */
const FURNISHED_ROOM_LIMIT_BYTES = 64 * 1_000_000;
/** A few frames so lazily made resources exist before they are counted. */
const FRAMES = 4;

function megabytes(bytes: number): number {
    return Number((bytes / 1_000_000).toFixed(3));
}

await runThreePage(async ({ device, data, log }) => {
    data.userAgent = navigator.userAgent;
    data.browserVersion = await fullBrowserVersion();
    const manifests = loadAllManifests();
    const scenes: Record<string, unknown> = {};
    data.scenes = scenes;
    const failures: string[] = [];
    for (const id of SCENES) {
        const manifest = manifests.find((candidate) => candidate.id === id);
        assert(manifest, `no manifest '${id}'`);
        const scene = await startViewerScene(device, manifest, { gpuTiming: true, label: `memory ${id}` });
        try {
            for (let index = 0; index < FRAMES; index++) {
                scene.frame({ units: Math.min(scene.probeCount, 256) });
                await device.queue.onSubmittedWorkDone();
            }
            const { memory } = scene.platform.stats;
            const ddgiBytes = Object.values(memory.techniques).reduce((sum, bytes) => sum + bytes, 0);
            const shadeBytes = scene.platform.shade.memoryBytes;
            const everything = memory.total + shadeBytes;
            const field = scene.ddgi.field;
            const result = {
                probes: scene.probeCount,
                probe_counts: field ? field.grid.counts.join('x') : '',
                spacing_m: manifest.volume.probeSpacing,
                rays_per_probe: scene.ddgi.settings.raysPerProbe,
                scene_copy_bytes: memory.sceneCopy,
                acceleration_bytes: memory.acceleration,
                ddgi_bytes: ddgiBytes,
                total_bytes: memory.total,
                shading_sky_bytes: shadeBytes,
                everything_bytes: everything,
                scene_copy_mb: megabytes(memory.sceneCopy),
                acceleration_mb: megabytes(memory.acceleration),
                ddgi_mb: megabytes(ddgiBytes),
                total_mb: megabytes(memory.total),
                shading_sky_mb: megabytes(shadeBytes),
                everything_mb: megabytes(everything),
                triangles: scene.loaded.stats.triangles,
            };
            scenes[id] = result;
            log(
                `${id}: ${result.probes} probes (${result.probe_counts} at ${result.spacing_m} m); scene ${result.scene_copy_mb} MB, acceleration ${result.acceleration_mb} MB, ` +
                    `DDGI ${result.ddgi_mb} MB, total ${result.total_mb} MB, sky not in the total ${result.shading_sky_mb} MB, everything ${result.everything_mb} MB`,
            );
            if (id === 'furnished-room' && everything > FURNISHED_ROOM_LIMIT_BYTES) {
                failures.push(`furnished-room: everything the core owns is ${megabytes(everything)} MB, over the 64 MB target (total without the sky ${megabytes(memory.total)} MB)`);
            }
        } finally {
            scene.dispose();
            await device.queue.onSubmittedWorkDone();
        }
    }
    // After the numbers are in the report, so a failed run can still be read.
    assert(failures.length === 0, failures.join('; '));
});
