/**
 * T2.M1 item 1 (docs/design/m3-moving.md §1, T3.1 check 1): 10,000
 * transforms through `setTransforms` cost ≤ 2 ms of CPU. B1a measured
 * 10,000 `setTransform` calls at 18-20 ms, about 2 µs each, all bookkeeping.
 *
 * A platform with no technique and 10,000 instances of one small sphere at
 * random places (the B1a bench's `synthetic-10k` layout). Each frame every
 * instance moves a small step: the matrices are made first (the host's
 * work, not timed), then one `gi.scene.setTransforms` call is timed, then the
 * frame is encoded and submitted (the scene copy's upload and the GPU top
 * level's refit; `stats.encodeCpuMs` reported beside it, not part of the
 * target). Three sets of frames, taking turns frame by frame:
 * - **translating** (the gate): every instance slides, as in B1a's bench,
 *   so the number compares with its 18-20 ms;
 * - **turning:** every instance also turns a little, so the scene copy
 *   works out each normal matrix again (reported, not judged);
 * - **one call each:** the translating moves through 10,000 `setTransform`
 *   calls, for scale.
 *
 * `performance.now()` steps in 0.1 ms with jitter (NOTES), well below the
 * target. Judged on the median of TIMED_FRAMES frames.
 *
 * A timing run takes both locks (CPU-timed):
 *   node tools/reference/bench-lock.ts run --cpu-timed --what "T2.M1 setTransforms" -- pnpm test:gpu --gpu nvidia --filter scene-set-transforms
 */

import type { InstanceHandle } from '../../../../packages/ddgi-core/src/platform/contract/handles.ts';
import { GIPlatformImpl } from '../../../../packages/ddgi-core/src/platform/gi-platform.ts';
import { createRandom, randomAffine } from '../../../../packages/ddgi-core/src/platform/trace/oracle-fixtures.ts';
import { assert, runGpuTest } from '../../client.ts';
import { frameInput } from '../ddgi-shared/harness.ts';
import { median } from '../trace-bench/statistics.ts';
import { makeSphereMesh } from '../trace-bench/synthetic.ts';

const INSTANCES = 10_000;
const EXTENT = 100;
/** Each instance moves up to this far per axis per frame. */
const STEP = 0.25;
const WARMUP_FRAMES = 10;
/** Timed frames of each set (sliding, turning, one call each). */
const TIMED_FRAMES = 40;
/** The turning set's turn per frame. */
const TURN_RADIANS = 0.01;
/** The done-when's bar (TASKS, T2.M1 item 1). */
const TARGET_MS = 2;

function percentile(values: readonly number[], share: number): number {
    const sorted = [...values].sort((a, b) => a - b);
    return sorted[Math.min(sorted.length - 1, Math.floor(share * sorted.length))];
}

await runGpuTest(
    async ({ device, log }) => {
        const platform = new GIPlatformImpl(device, { label: 'set-transforms' });
        try {
            await platform.prepare();
            const scene = platform.scene;
            const random = createRandom(0x10_000);
            const sphere = makeSphereMesh();
            assert(sphere.indices, 'the sphere mesh is indexed');
            const mesh = scene.createMesh({ positions: sphere.positions, indices: sphere.indices });
            const material = scene.createMaterial({ albedo: [0.8, 0.8, 0.8] });
            const matrices = new Float32Array(INSTANCES * 16);
            const handles = new Uint32Array(INSTANCES);
            for (let index = 0; index < INSTANCES; index++) {
                const matrix = randomAffine(random, { extent: EXTENT, mirror: index % 4 === 0 });
                matrices.set(matrix, index * 16);
                handles[index] = scene.createInstance({ mesh, materials: [material], transform: matrix });
            }
            const frame = async (): Promise<number> => {
                const encoder = device.createCommandEncoder({ label: 'set-transforms frame' });
                platform.encode(encoder, frameInput({ paused: true }));
                device.queue.submit([encoder.finish()]);
                await device.queue.onSubmittedWorkDone();
                return platform.stats.encodeCpuMs;
            };
            await frame();
            for (let waited = 0; platform.rays.building; waited++) {
                assert(waited < 200, 'the trees still build after 200 frames');
                await platform.rays.whenBuilt();
                await frame();
            }

            const slide = (): void => {
                for (let index = 0; index < INSTANCES; index++) {
                    for (let axis = 0; axis < 3; axis++) matrices[index * 16 + 12 + axis] += (random() * 2 - 1) * STEP;
                }
            };
            // A small turn about y for every instance, on top of the slide.
            const turn = (): void => {
                const cos = Math.cos(TURN_RADIANS);
                const sin = Math.sin(TURN_RADIANS);
                for (let index = 0; index < INSTANCES; index++) {
                    const base = index * 16;
                    for (let column = 0; column < 3; column++) {
                        const x = matrices[base + column * 4];
                        const z = matrices[base + column * 4 + 2];
                        matrices[base + column * 4] = cos * x + sin * z;
                        matrices[base + column * 4 + 2] = -sin * x + cos * z;
                    }
                }
            };
            const slidingMs: number[] = [];
            const turningMs: number[] = [];
            const singleMs: number[] = [];
            const encodeMs: number[] = [];
            const one = new Array<number>(16);
            for (let index = 0; index < WARMUP_FRAMES + 3 * TIMED_FRAMES; index++) {
                slide();
                const set = index % 3;
                if (set === 1) turn();
                const start = performance.now();
                if (set === 2) {
                    // The same moves one call each, for scale (B1a's number).
                    for (let instance = 0; instance < INSTANCES; instance++) {
                        for (let value = 0; value < 16; value++) one[value] = matrices[instance * 16 + value];
                        scene.setTransform(handles[instance] as InstanceHandle, one);
                    }
                } else {
                    scene.setTransforms(handles, matrices);
                }
                const elapsed = performance.now() - start;
                const encoded = await frame();
                if (index < WARMUP_FRAMES) continue;
                [slidingMs, turningMs, singleMs][set].push(elapsed);
                if (set === 0) encodeMs.push(encoded);
            }
            const result = {
                instances: INSTANCES,
                setTransformsMsMedian: median(slidingMs),
                setTransformsMsP95: percentile(slidingMs, 0.95),
                setTransformsMsMax: Math.max(...slidingMs),
                setTransformsTurningMsMedian: median(turningMs),
                setTransformsTurningMsP95: percentile(turningMs, 0.95),
                setTransformLoopMsMedian: median(singleMs),
                encodeCpuMsMedian: median(encodeMs),
                timedFrames: slidingMs.length,
                userAgent: navigator.userAgent,
            };
            log(
                `setTransforms x${INSTANCES} sliding: median ${result.setTransformsMsMedian.toFixed(3)} ms (p95 ${result.setTransformsMsP95.toFixed(3)}, max ${result.setTransformsMsMax.toFixed(3)}; bar ${TARGET_MS} ms); ` +
                    `turning too: median ${result.setTransformsTurningMsMedian.toFixed(3)} ms; ` +
                    `${INSTANCES} setTransform calls ${result.setTransformLoopMsMedian.toFixed(2)} ms; the frame's encode ${result.encodeCpuMsMedian.toFixed(2)} ms`,
            );
            assert(result.setTransformsMsMedian <= TARGET_MS, `setTransforms of ${INSTANCES} took ${result.setTransformsMsMedian.toFixed(3)} ms (median), over ${TARGET_MS} ms`);
            return result;
        } finally {
            platform.dispose();
        }
    },
    { requiredFeatures: [], optionalFeatures: ['subgroups'] },
);
