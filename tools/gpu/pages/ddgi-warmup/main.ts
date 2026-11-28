/**
 * T2.4b check 6: the warm-up runs in the technique and is on by default, so
 * a host that does nothing gets the same first frames as the viewer's own
 * warm-up gave before (apps/viewer before T2.4b: no probe work until the
 * mesh trees were built, then 16 passes of `maxProbesPerFrame` probes with
 * no time between them for the history, then the budget's calibration).
 *
 * Per scene, two platforms on the same scene:
 * - **the technique's warm-up:** default settings; the host runs every frame
 *   with `{ gpuMs: 2 }` and a 60 Hz `deltaTime` from the first, trees still
 *   building, and does nothing else;
 * - **the old viewer's warm-up**, replicated from outside with the
 *   technique's warm-up off (`warmUpPasses: 0`): frames at 0 probes until
 *   the trees are built, then frames of `min(probes, maxProbesPerFrame)`
 *   with `deltaTime` 0 until 16 passes are made.
 * The two fields must be the same (the test view's GI image at the first
 * bookmark, pixel for pixel, and the irradiance atlas's mean), the warm-up
 * must take the same frames, and the first frame after it must start the
 * budget's calibration (the warm-up's frames weren't measured).
 *
 * `pnpm test:gpu --filter ddgi-warmup --gpu nvidia` (or intel).
 */

import { assert, runGpuTest } from '../../client.ts';
import { GIPlatformImpl } from '../../../../packages/core/src/platform/gi-platform.ts';
import { DDGITechnique } from '../../../../packages/core/src/techniques/ddgi/ddgi-technique.ts';
import { DEFAULT_WARM_UP_PASSES } from '../../../../packages/core/src/techniques/ddgi/settings.ts';
import { describeScene, frameInput, halfToFloat, loadManifest, readTextureBytes, releaseDeviceOnPageHide, runFrame, TestView } from '../ddgi-shared/harness.ts';

const SCENES = ['cornell-box', 'two-room'];
const WIDTH = 320;
const HEIGHT = 180;

async function platformFor(device: GPUDevice, sceneId: string, warmUpPasses: number | undefined) {
    const manifest = loadManifest(sceneId);
    // Both runs trace the GPU's trees, which nothing replaces (#26), so the two fields see the same trees.
    const platform = new GIPlatformImpl(device, { label: `warm-up ${sceneId}`, gpuTiming: true });
    describeScene(platform, manifest);
    const volume = { center: manifest.volume.center, size: manifest.volume.size, spacing: manifest.volume.probeSpacing };
    const ddgi = platform.addTechnique(new DDGITechnique(warmUpPasses === undefined ? { volume } : { volume, warmUpPasses }));
    platform.setOutput(ddgi);
    await platform.prepare();
    return { manifest, platform, ddgi };
}

async function atlasMean(device: GPUDevice, texture: GPUTexture): Promise<number> {
    const halves = new Uint16Array((await readTextureBytes(device, texture, 8)).buffer);
    let sum = 0;
    for (let texel = 0; texel < halves.length; texel += 4) sum += halfToFloat(halves[texel]) + halfToFloat(halves[texel + 1]) + halfToFloat(halves[texel + 2]);
    return sum / ((halves.length / 4) * 3);
}

await runGpuTest(
    async ({ device, log: logLine }) => {
        releaseDeviceOnPageHide(device);
        const log = (line: string): void => {
            logLine(line);
            console.log(line);
        };
        const results: Record<string, unknown> = {};
        const failures: string[] = [];
        for (const sceneId of SCENES) {
            // The technique's warm-up: a host that only runs frames.
            const host = await platformFor(device, sceneId, undefined);
            let hostFrames = 0;
            let warmFrames = 0;
            do {
                const encoder = device.createCommandEncoder();
                host.platform.encode(encoder, { ...frameInput({ gpuMs: 2 }), deltaTime: 1 / 60 });
                device.queue.submit([encoder.finish()]);
                hostFrames++;
                if ((host.ddgi.counters['warm-up passes'] ?? 0) > 0) warmFrames++;
                if (hostFrames % 4 === 0) await device.queue.onSubmittedWorkDone();
                // Trees are built in a worker: let it finish while the host keeps running frames.
                if (host.platform.rays.building) await new Promise((resolve) => setTimeout(resolve, 5));
                assert(hostFrames < 20_000, `${sceneId}: the warm-up never ended`);
            } while (host.ddgi.warmingUp);
            await device.queue.onSubmittedWorkDone();
            // The next frame starts the calibration.
            runFrame(device, host.platform, { gpuMs: 2 });
            hostFrames++;
            const budgetState = host.platform.budgetReport(host.ddgi)?.state;
            const hostPasses = host.ddgi.counters['warm-up passes'] ?? 0;

            // The old viewer's warm-up, from outside. It waits as many frames as the host's
            // technique waited for the trees, so both warm up on the same frame numbers (rays are
            // seeded by the frame), and the fields can be compared value for value.
            const viewer = await platformFor(device, sceneId, 0);
            // The meshes reach the ray queries (and start their trees) at the first frame, so `whenBuilt` is awaited
            // only after it: awaited before any frame it resolves at once, and when the host's trees took a single
            // frame (waitFrames 1) the replica then warmed up on trees still building (an atlas of 0, or too dark;
            // T3.5's review). The next frame's update places the built trees before the technique traces, as
            // the host's first warming frame does.
            const waitFrames = hostFrames - 1 - warmFrames;
            assert(waitFrames >= 1, `${sceneId}: the host warmed up in its first frame, before any tree could be built`);
            for (let frame = 0; frame < waitFrames; frame++) runFrame(device, viewer.platform, { paused: true });
            await viewer.platform.rays.whenBuilt();
            const field = viewer.ddgi.field;
            assert(field, 'no field');
            const probeCount = field.grid.probeCount;
            const stepUnits = Math.min(probeCount, viewer.ddgi.settings.maxProbesPerFrame);
            let viewerUpdates = 0;
            let viewerFrames = 0;
            while (viewerUpdates / probeCount < DEFAULT_WARM_UP_PASSES) {
                const encoder = device.createCommandEncoder();
                viewer.platform.encode(encoder, { ...frameInput(stepUnits), deltaTime: 0 });
                device.queue.submit([encoder.finish()]);
                viewerUpdates += viewer.ddgi.counters['probes updated'] ?? 0;
                viewerFrames++;
                if (viewerFrames % 4 === 0) await device.queue.onSubmittedWorkDone();
            }
            await device.queue.onSubmittedWorkDone();

            // The same field? The host's frame after the warm-up updated probes too, so compare the
            // fields as the warm-up left them: run the viewer's one more frame at the host's budget too.
            runFrame(device, viewer.platform, { gpuMs: 2 });
            await device.queue.onSubmittedWorkDone();
            const views = await Promise.all([host, viewer].map(async (run) => {
                const runField = run.ddgi.field!;
                const view = await TestView.create(device, run.platform);
                const image = (await view.render(runField, run.manifest.cameras[0], WIDTH, HEIGHT)).indirect;
                return { image, atlas: await atlasMean(device, runField.irradiance) };
            }));
            host.platform.dispose();
            viewer.platform.dispose();
            let differentPixels = 0;
            let largest = 0;
            for (let index = 0; index < views[0].image.length; index++) {
                const difference = Math.abs(views[0].image[index] - views[1].image[index]);
                if (difference > 0) differentPixels++;
                largest = Math.max(largest, difference);
            }
            results[sceneId] = { probes: probeCount, hostFrames, warmFrames, hostPasses, viewerFrames, viewerPasses: viewerUpdates / probeCount, budgetStateAfter: budgetState, atlasHost: views[0].atlas, atlasViewer: views[1].atlas, differentPixelValues: differentPixels, largestDifference: largest };
            log(
                `${sceneId}: the technique's warm-up took ${warmFrames} frames (${hostPasses.toFixed(2)} passes; ${hostFrames} host frames, trees included), the old viewer's ${viewerFrames} (${(viewerUpdates / probeCount).toFixed(2)} passes); ` +
                    `the budget after it: ${budgetState}; atlas means ${views[0].atlas.toPrecision(7)} and ${views[1].atlas.toPrecision(7)}; ${differentPixels} pixel values differ (largest ${largest.toExponential(2)})`,
            );
            if (warmFrames !== viewerFrames) failures.push(`${sceneId}: the warm-up took ${warmFrames} frames, the old viewer's ${viewerFrames}`);
            if (budgetState !== 'calibrating') failures.push(`${sceneId}: the first frame after the warm-up left the budget '${budgetState}', not calibrating`);
            if (differentPixels > 0) failures.push(`${sceneId}: the fields differ (${differentPixels} pixel values, largest ${largest.toExponential(2)})`);
        }
        assert(failures.length === 0, failures.join('; '));
        return results;
    },
    { optionalFeatures: ['texture-formats-tier1', 'subgroups', 'timestamp-query'] },
);
