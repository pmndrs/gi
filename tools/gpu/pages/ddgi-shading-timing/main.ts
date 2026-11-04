/**
 * T1.6 done-when: what GI sampling costs a renderer at 1440p.
 *
 * The hello example's forward pass at 2560 × 1440 on the point-light Cornell
 * box, timed by GPU timestamps in two versions that differ only in where
 * irradiance comes from:
 * - `module`: the DDGI shading module (`giIrradianceAt`, via `globalsCode`);
 * - `flat`: a constant in its place.
 * GI sampling cost = median(module) − median(flat), per camera bookmark.
 * The two passes alternate inside each submit (BatchTimer), so clocks and
 * heat drift hit both alike. The field is converged first (every probe for FRAMES
 * frames), so probe states are what a running scene has.
 *
 * Manual, under the bench lock with an idle GPU; run it three times per GPU
 * and turn the runner's JSON into `bench/results/shading.csv` with to-csv.ts
 * (run1..3 plus the median rows).
 */

import { assert, runGpuTest } from '../../client.ts';
import { ForwardRenderer, type RenderView } from '../../../../examples/webgpu/hello/src/renderer.ts';
import { releaseDeviceOnPageHide, runFrames } from '../ddgi-shared/harness.ts';
import { createHelloScene } from '../ddgi-shading-shared/hello-scene.ts';
import { fullBrowserVersion } from '../trace-bench/run.ts';

const SCENE = 'cornell-box-point';
/** Bookmark 1 looks into the open box; bookmark 2 is inside it, so every pixel is a surface. */
const CAMERAS = [1, 2] as const;
const WIDTH = 2560;
const HEIGHT = 1440;
const FRAMES = 600;
/** Passes of each version per submit (BatchTimer). */
const PASSES_PER_BATCH = 20;
const WARMUP_BATCHES = 3;
const MEASURED_BATCHES = 10;
/** The flat version's constant: about Cornell's mean irradiance, so the pixels aren't black. */
const FLAT_IRRADIANCE: readonly [number, number, number] = [0.5, 0.4, 0.2];
const VIEW: RenderView = { mode: 'full' };
/** The done-when bar on the RTX 3060 (TASKS.md, T1.6). */
const BUDGET_MS = 1.0;

/** Timestamp ticks are nanoseconds. */
const NANOSECONDS_PER_MILLISECOND = 1e6;

function percentile(values: readonly number[], share: number): number {
    const sorted = [...values].sort((a, b) => a - b);
    const position = (sorted.length - 1) * share;
    const below = Math.floor(position);
    const above = Math.ceil(position);
    return sorted[below] + (sorted[above] - sorted[below]) * (position - below);
}

/**
 * Times batches of forward passes by timestamps. A batch is one submit of
 * PASSES_PER_BATCH passes of each version, interleaved, so the GPU stays busy
 * the whole time: timing one pass per submit leaves it idle between
 * readbacks, and an idle laptop GPU drops its clocks.
 */
class BatchTimer {
    private readonly querySet: GPUQuerySet;
    private readonly resolve: GPUBuffer;
    private readonly readback: GPUBuffer;
    private readonly queries: number;

    constructor(
        private readonly device: GPUDevice,
        private readonly passesPerVersion: number,
    ) {
        // Two versions, two timestamps (begin, end) per pass.
        this.queries = passesPerVersion * 2 * 2;
        this.querySet = device.createQuerySet({ type: 'timestamp', count: this.queries });
        this.resolve = device.createBuffer({ size: this.queries * 8, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC });
        this.readback = device.createBuffer({ size: this.queries * 8, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    }

    /** Milliseconds per pass of each version, in submission order. */
    async time(
        recordA: (encoder: GPUCommandEncoder, timestampWrites: GPURenderPassTimestampWrites) => void,
        recordB: (encoder: GPUCommandEncoder, timestampWrites: GPURenderPassTimestampWrites) => void,
    ): Promise<{ a: number[]; b: number[] }> {
        const encoder = this.device.createCommandEncoder({ label: 'shading timing' });
        for (let pass = 0; pass < this.passesPerVersion; pass++) {
            const base = pass * 4;
            recordA(encoder, { querySet: this.querySet, beginningOfPassWriteIndex: base, endOfPassWriteIndex: base + 1 });
            recordB(encoder, { querySet: this.querySet, beginningOfPassWriteIndex: base + 2, endOfPassWriteIndex: base + 3 });
        }
        encoder.resolveQuerySet(this.querySet, 0, this.queries, this.resolve, 0);
        encoder.copyBufferToBuffer(this.resolve, 0, this.readback, 0, this.queries * 8);
        this.device.queue.submit([encoder.finish()]);
        await this.readback.mapAsync(GPUMapMode.READ);
        const stamps = new BigInt64Array(this.readback.getMappedRange().slice(0));
        this.readback.unmap();
        const a: number[] = [];
        const b: number[] = [];
        for (let pass = 0; pass < this.passesPerVersion; pass++) {
            const base = pass * 4;
            a.push(Number(stamps[base + 1] - stamps[base]) / NANOSECONDS_PER_MILLISECOND);
            b.push(Number(stamps[base + 3] - stamps[base + 2]) / NANOSECONDS_PER_MILLISECOND);
        }
        return { a, b };
    }

    destroy(): void {
        this.querySet.destroy();
        this.resolve.destroy();
        this.readback.destroy();
    }
}

await runGpuTest(
    async ({ device, log }) => {
        releaseDeviceOnPageHide(device);
        assert(device.features.has('timestamp-query'), 'timing needs the timestamp-query feature');
        const { platform, ddgi, scene } = await createHelloScene(device, SCENE, {});
        const timer = new BatchTimer(device, PASSES_PER_BATCH);
        try {
            const field = ddgi.field;
            assert(field, 'DDGI has no field after prepare');
            await runFrames(device, platform, FRAMES, field.grid.probeCount);

            const withGi = new ForwardRenderer(device, scene, { shading: platform.shading });
            const flat = new ForwardRenderer(device, scene, { shading: null, flatIrradiance: FLAT_IRRADIANCE });
            withGi.resize(WIDTH, HEIGHT);
            flat.resize(WIDTH, HEIGHT);

            const results: Record<string, unknown> = {};
            let worstGiMs = 0;
            for (const cameraIndex of CAMERAS) {
                const camera = scene.manifest.cameras[cameraIndex - 1];
                const moduleMs: number[] = [];
                const flatMs: number[] = [];
                // The renderers write their frame uniforms once per batch; every pass in it sees the same camera.
                const recordWithGi = (encoder: GPUCommandEncoder, timestampWrites: GPURenderPassTimestampWrites): void => withGi.render(encoder, withGi.camera(camera), VIEW, timestampWrites);
                const recordFlat = (encoder: GPUCommandEncoder, timestampWrites: GPURenderPassTimestampWrites): void => flat.render(encoder, flat.camera(camera), VIEW, timestampWrites);
                for (let batch = 0; batch < WARMUP_BATCHES + MEASURED_BATCHES; batch++) {
                    const times = await timer.time(recordWithGi, recordFlat);
                    if (batch < WARMUP_BATCHES) continue;
                    // Intel UHD sometimes returns an end before its start (NOTES); drop those.
                    moduleMs.push(...times.a.filter((ms) => ms > 0));
                    flatMs.push(...times.b.filter((ms) => ms > 0));
                }
                const measuredPasses = MEASURED_BATCHES * PASSES_PER_BATCH;
                assert(moduleMs.length >= measuredPasses / 2 && flatMs.length >= measuredPasses / 2, 'too many invalid timestamps');
                const moduleMedian = percentile(moduleMs, 0.5);
                const flatMedian = percentile(flatMs, 0.5);
                const giMs = moduleMedian - flatMedian;
                worstGiMs = Math.max(worstGiMs, giMs);
                log(`${camera.name}: forward pass ${moduleMedian.toFixed(3)} ms with GI, ${flatMedian.toFixed(3)} ms with a constant; GI sampling ${giMs.toFixed(3)} ms`);
                results[camera.name] = {
                    moduleMsMedian: moduleMedian,
                    moduleMsP10: percentile(moduleMs, 0.1),
                    moduleMsP90: percentile(moduleMs, 0.9),
                    flatMsMedian: flatMedian,
                    flatMsP10: percentile(flatMs, 0.1),
                    flatMsP90: percentile(flatMs, 0.9),
                    giSamplingMs: giMs,
                    passes: moduleMs.length,
                };
            }
            withGi.destroy();
            flat.destroy();
            return {
                benchmark: 'ddgi-shading-timing',
                scene: SCENE,
                width: WIDTH,
                height: HEIGHT,
                probes: field.grid.counts,
                budgetMs: BUDGET_MS,
                worstGiSamplingMs: worstGiMs,
                cameras: results,
                userAgent: navigator.userAgent,
                browserVersion: await fullBrowserVersion(),
            };
        } finally {
            timer.destroy();
            platform.dispose();
        }
    },
    { optionalFeatures: ['texture-formats-tier1', 'subgroups', 'timestamp-query'] },
);
