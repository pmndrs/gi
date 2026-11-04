/**
 * T1.5a: the distance atlas's declared fallback.
 *
 * DDGI stores distances as `rg16float` when the device has
 * `texture-formats-tier1` (which makes `rg16float` a storage format), and as
 * `rgba16float` otherwise. This page runs both on the same GPU: device A asks
 * for the feature, device B (a second device from a fresh adapter) does not.
 * The same scene and settings then must give the same field: the distance
 * moments texel for texel, and `ddgiSampleIrradiance` at surface points.
 * Both formats hold the same half floats and the per-frame rotation is
 * seeded by the frame number, so they agree to rounding; nothing requires
 * bit-identical results.
 *
 * Fails if the adapter lacks `texture-formats-tier1` (then the compact path
 * can't be tested here), if either device picks the wrong format, or if the
 * two disagree.
 */

import { assert, drainDevice, runGpuTest, watchDevice } from '../../client.ts';
import { CORNELL_POINT_IN_AIR_VOLUME, createDdgiScene, FieldProbe, runFrames, type SurfacePoint } from '../ddgi-shared/harness.ts';
import type { DDGIField } from '../../../../packages/ddgi-core/src/techniques/ddgi/ddgi-technique.ts';

const FRAMES = 200;
/** Largest allowed difference, relative to the larger value (with a small floor for values near 0). */
const AGREEMENT = 1e-3;

/** Points spread over the Cornell box's walls, floor and ceiling. */
function surfacePoints(): SurfacePoint[] {
    const points: SurfacePoint[] = [];
    const size = [5.55, 5.488, 5.592];
    for (let axis = 0; axis < 3; axis++) {
        for (const end of [0, 1]) {
            const [a, b] = [0, 1, 2].filter((other) => other !== axis);
            for (let i = 0; i < 6; i++) {
                for (let j = 0; j < 6; j++) {
                    const position: [number, number, number] = [0, 0, 0];
                    position[axis] = end * size[axis];
                    position[a] = ((i + 0.5) / 6) * size[a];
                    position[b] = ((j + 0.5) / 6) * size[b];
                    const normal: [number, number, number] = [0, 0, 0];
                    normal[axis] = end === 0 ? 1 : -1;
                    points.push({ position, normal, viewDirection: [-normal[0], -normal[1], -normal[2]] });
                }
            }
        }
    }
    return points;
}

/** The distance atlas's first two channels, as floats, read back with 256-byte row padding. */
async function readDistanceMoments(device: GPUDevice, field: DDGIField): Promise<Float32Array> {
    const texture = field.distance;
    const bytesPerTexel = field.layout.distanceFormat === 'rg16float' ? 4 : 8;
    const rowBytes = Math.ceil((texture.width * bytesPerTexel) / 256) * 256;
    const layers = texture.depthOrArrayLayers;
    const buffer = device.createBuffer({ size: rowBytes * texture.height * layers, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const encoder = device.createCommandEncoder();
    encoder.copyTextureToBuffer({ texture }, { buffer, bytesPerRow: rowBytes, rowsPerImage: texture.height }, { width: texture.width, height: texture.height, depthOrArrayLayers: layers });
    device.queue.submit([encoder.finish()]);
    await buffer.mapAsync(GPUMapMode.READ);
    const halves = new Uint16Array(buffer.getMappedRange().slice(0));
    buffer.unmap();
    buffer.destroy();
    const moments = new Float32Array(texture.width * texture.height * layers * 2);
    const halvesPerTexel = bytesPerTexel / 2;
    let out = 0;
    for (let layer = 0; layer < layers; layer++) {
        for (let row = 0; row < texture.height; row++) {
            const rowStart = ((layer * texture.height + row) * rowBytes) / 2;
            for (let column = 0; column < texture.width; column++) {
                moments[out++] = halfToFloat(halves[rowStart + column * halvesPerTexel]);
                moments[out++] = halfToFloat(halves[rowStart + column * halvesPerTexel + 1]);
            }
        }
    }
    return moments;
}

function halfToFloat(half: number): number {
    const sign = half & 0x8000 ? -1 : 1;
    const exponent = (half >> 10) & 0x1f;
    const fraction = half & 0x3ff;
    if (exponent === 0) return sign * 2 ** -14 * (fraction / 1024);
    if (exponent === 31) return fraction === 0 ? sign * Infinity : NaN;
    return sign * 2 ** (exponent - 15) * (1 + fraction / 1024);
}

interface FormatRun {
    format: string;
    moments: Float32Array;
    irradiance: [number, number, number][];
}

async function runOn(device: GPUDevice, points: readonly SurfacePoint[]): Promise<FormatRun> {
    const scene = await createDdgiScene(device, 'cornell-box-point', { volume: CORNELL_POINT_IN_AIR_VOLUME }, { ignoreLightRange: true });
    try {
        const field = scene.ddgi.field;
        assert(field, 'no field after prepare');
        await runFrames(device, scene.platform, FRAMES, field.grid.probeCount);
        const moments = await readDistanceMoments(device, field);
        const irradiance = await (await FieldProbe.create(device, scene.platform)).sample(field, points);
        return { format: field.layout.distanceFormat, moments, irradiance };
    } finally {
        scene.platform.dispose();
    }
}

function largestDifference(a: ArrayLike<number>, b: ArrayLike<number>, floor: number): number {
    let worst = 0;
    for (let index = 0; index < a.length; index++) {
        const difference = Math.abs(a[index] - b[index]) / Math.max(Math.abs(a[index]), Math.abs(b[index]), floor);
        worst = Math.max(worst, difference);
    }
    return worst;
}

await runGpuTest(
    async ({ adapter, device, log }) => {
        assert(adapter.features.has('texture-formats-tier1'), 'this adapter has no texture-formats-tier1, so the rg16float path cannot be tested here');
        assert(device.features.has('texture-formats-tier1'), 'device A was created without texture-formats-tier1');
        const points = surfacePoints();
        const compact = await runOn(device, points);

        // Device B: a fresh adapter (an adapter gives out one device), no texture-formats-tier1.
        const fallbackAdapter = await navigator.gpu.requestAdapter();
        assert(fallbackAdapter, 'no second adapter');
        const fallbackDevice = await fallbackAdapter.requestDevice({ requiredFeatures: device.features.has('subgroups') ? ['subgroups'] : [] });
        const fallbackErrors = watchDevice(fallbackDevice);
        assert(!fallbackDevice.features.has('texture-formats-tier1'), 'device B has texture-formats-tier1 although it was not requested');
        const fallback = await runOn(fallbackDevice, points);
        await drainDevice(fallbackDevice);
        fallbackDevice.destroy();
        assert(fallbackErrors.length === 0, `device B reported errors: ${fallbackErrors.join(' | ')}`);

        assert(compact.format === 'rg16float', `with the feature, distances use ${compact.format}`);
        assert(fallback.format === 'rgba16float', `without the feature, distances use ${fallback.format}`);
        assert(compact.moments.length === fallback.moments.length, 'the atlases differ in size');
        let nonZero = 0;
        for (const value of compact.moments) if (value > 0) nonZero++;
        assert(nonZero > compact.moments.length / 2, `only ${nonZero} of ${compact.moments.length} distance values were written`);

        const momentsDifference = largestDifference(compact.moments, fallback.moments, 1e-3);
        const irradianceDifference = largestDifference(compact.irradiance.flat(), fallback.irradiance.flat(), 1e-3);
        log(`distance moments: largest difference ${momentsDifference.toExponential(2)}; irradiance at ${points.length} points: ${irradianceDifference.toExponential(2)}`);
        assert(momentsDifference <= AGREEMENT, `the distance moments differ by up to ${momentsDifference} between the formats`);
        assert(irradianceDifference <= AGREEMENT, `the sampled irradiance differs by up to ${irradianceDifference} between the formats`);
        return {
            frames: FRAMES,
            formats: [compact.format, fallback.format],
            distanceValues: compact.moments.length,
            largestMomentDifference: momentsDifference,
            largestIrradianceDifference: irradianceDifference,
            points: points.length,
        };
    },
    { optionalFeatures: ['texture-formats-tier1', 'subgroups'] },
);
