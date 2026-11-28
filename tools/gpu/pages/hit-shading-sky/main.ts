/**
 * T1.4: the platform's sky on a real GPU, through the hit-shading module a
 * technique compiles (`giSkyRadiance`, `giSkyIrradianceAt`), and the sky
 * projection itself (`PlatformSky`).
 *
 * 1. **Open ground against the reference.** The shared sky file
 *    (`scenes/sky/open-ground.exr`) as an equirect texture: `giSkyIrradiance`
 *    on flat ground must match the file's own integral (the sidecar's
 *    `measured`, the analytic answer Cycles matched within 0.03% in T1.1)
 *    within the ±0.5% reference tolerance, per channel.
 * 2. **`procedural` is the same sky,** and its brightness numbers work:
 *    by default the sky of a sun of illuminance 1; `sunIlluminance` 2 and
 *    `intensity` 2 each double it.
 *    **The same sky:** The open-ground and Sponza manifests
 *    through `platformSkyEnvironment`: the platform builds the sky with the
 *    sky files' code; the ground irradiance must match the sidecars within
 *    ±0.5%, and the open-ground SH must equal the texture path's.
 * 3. **Tilted surfaces**, against a brute-force CPU integral of the file
 *    (within 2% of the ground's irradiance), and **radiance on a miss**
 *    against a bilinear lookup of the file (2%).
 * 4. **The equirect layout of the contract** (three.js's `equirectUV`): a
 *    texture whose texels hold their own direction, (d + 1) / 2, read back
 *    along known directions, also turned by `rotationY` and scaled by
 *    `intensity`, in rgba16float and (unfilterable) rgba32float. Its
 *    irradiance has an exact answer, pi / 2 + (pi / 3) n, which the SH must hit.
 *    The same for a **cube** texture, filled with WebGPU's cube-face table.
 * 5. **`color` and `none`.**
 * 6. **Re-projection only on change**, counted, and again when the same
 *    environment object is set again after its texture changed.
 * 7. **Subgroups and the fallback** reduce the same sky to the same SH,
 *    and the fallback matches the CPU projection (`spherical-harmonics.ts`)
 *    of the file's radiance map; how far the map's SH are from the file's
 *    is reported.
 * 8. **The GPU's procedural sky is the CPU's** (T2.6a check 1). For skies
 *    from a 2-degree sun (made at 5) to 89 degrees and turbidity 1.7 to 10,
 *    the radiance map the bake writes, read at every texel centre, against
 *    the CPU sky (`buildCalibratedSky`, the sky files' code) box filtered to
 *    the map's texels as the resample filters a 512 x 256 source: each
 *    channel's mean within 0.1%, every texel's channels within 1% (a channel
 *    the gamut clamp made 0 on the CPU: against the texel's brightest),
 *    black below the horizon; the radiance and irradiance SH against the
 *    CPU projection of that sky's map (since T2.6a the SH integrate the
 *    map, within 5e-4 of the largest coefficient: the GPU stores the map in
 *    halves, DECISIONS 2026-11-06), with subgroups and with the fallback,
 *    which agree within 1e-4; the ground's irradiance against the
 *    CPU sky's own (reported; ±0.5% is gated on the open-ground and Sponza
 *    skies, section 2).
 * 9. **When a procedural sky re-bakes** (m2-lights.md §6.2): not for the
 *    same numbers again or a sun turned 0.03 degrees; for a turn of more
 *    than 0.05 degrees since the last bake, and for any change of
 *    turbidity, intensity or sunIlluminance.
 */

import { GIPlatformImpl } from '../../../../packages/core/src/platform/gi-platform.ts';
import { buildCalibratedSky, DEFAULT_TURBIDITY } from '../../../../packages/core/src/platform/sky/calibrated-sky.ts';
import { liftSunAboveMinimum } from '../../../../packages/core/src/platform/sky/procedural-sky.ts';
import { MAX_TURBIDITY, MIN_TURBIDITY } from '../../../../packages/core/src/platform/sky/preetham.ts';
import { floatToHalf, halfImageToTextureRgba, halfToFloat } from '../../../../packages/core/src/platform/sky/half-float.ts';
import { PlatformSky, RADIANCE_MAP_HEIGHT, RADIANCE_MAP_WIDTH } from '../../../../packages/core/src/platform/sky/platform-sky.ts';
import { RADIANCE_SH_COUNT, SKY_HARMONICS_LAYOUT } from '../../../../packages/core/src/platform/sky/projection-wgsl.ts';
import {
    bruteForceIrradiance,
    convolveWithCosineLobe,
    projectEquirectRadiance,
    SH_COEFFICIENT_COUNT,
} from '../../../../packages/core/src/platform/sky/spherical-harmonics.ts';
import { sampleEquirectBilinear } from '../../../../packages/core/src/platform/sky/equirect.ts';
import type { EnvironmentDescriptor } from '../../../../packages/core/src/platform/contract/scene.ts';
import type { SceneManifest } from '../../../../scenes/src/manifest.ts';
import { loadSkyFile, type SkySidecar } from '../../../../scenes/src/sky/sky-file.ts';
import { platformSkyEnvironment } from '../../../../scenes/src/sky/sky-image.ts';
import { assert, runGpuTest } from '../../client.ts';
import { frame, normalize, readBuffer, ready, relativeError, ShadeKernel, type Vec3 } from '../hit-shading-shared/harness.ts';

/** The reference tolerance (DECISIONS: two renderers agree within ±0.5% on brightness). */
const REFERENCE_TOLERANCE = 0.005;
/** Tilted surfaces against brute force, relative to the ground's irradiance (band-4 truncation). */
const TILTED_TOLERANCE = 0.02;
/** Radiance on a miss against the file's own bilinear lookup (the map is half the file's resolution). */
const MISS_RADIANCE_TOLERANCE = 0.02;
/** Direction-coded textures: absolute, on values in 0..1 (bilinear and resampling error is about 1e-3). */
const CODED_RADIANCE_TOLERANCE = 0.01;
/** Their irradiance against the exact answer. */
const CODED_IRRADIANCE_TOLERANCE = 0.003;
/** Subgroup against fallback SH: relative to the largest coefficient. */
const SH_AGREEMENT_TOLERANCE = 1e-4;
/**
 * GPU SH against the CPU projection of the same map, and the procedural sky
 * against the sky file: relative to the largest coefficient. Looser than the
 * reductions' agreement because the GPU integrates the map it stored in
 * halves, and the RTX 3060 stores halves by truncation (-0.035% on average,
 * NOTES); a procedural sky is calibrated after that store, a texture isn't
 * (DECISIONS 2026-11-06).
 */
const MAP_SH_TOLERANCE = 5e-4;
/** T2.6a check 1: the GPU's procedural map against the CPU sky, each channel's mean and every texel's channels. */
const PROCEDURAL_MEAN_TOLERANCE = 0.001;
const PROCEDURAL_TEXEL_TOLERANCE = 0.01;

const urls = (pattern: Record<string, unknown>): string => Object.values(pattern)[0] as string;
const openGroundSkyUrl = urls(import.meta.glob('../../../../scenes/sky/open-ground.exr', { query: '?url', import: 'default', eager: true }));
const openGroundSidecarUrl = urls(import.meta.glob('../../../../scenes/sky/open-ground.json', { query: '?url', import: 'default', eager: true }));
const sponzaSidecarUrl = urls(import.meta.glob('../../../../scenes/sky/sponza.json', { query: '?url', import: 'default', eager: true }));
const openGroundManifestUrl = urls(import.meta.glob('../../../../scenes/manifests/open-ground.json', { query: '?url', import: 'default', eager: true }));
const sponzaManifestUrl = urls(import.meta.glob('../../../../scenes/manifests/sponza.json', { query: '?url', import: 'default', eager: true }));

async function fetchJson<T>(url: string): Promise<T> {
    const response = await fetch(url);
    if (!response.ok) throw new Error(`could not load ${url}: HTTP ${response.status}`);
    return (await response.json()) as T;
}

/** Radiance and irradiance along each direction, through the hit-shading module. */
const SKY_KERNEL = /* wgsl */ `
@group(1) @binding(0) var<storage, read> inputs: array<vec4f>;
@group(1) @binding(1) var<storage, read_write> outputs: array<vec4f>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let index = id.x;
    if (index >= arrayLength(&inputs)) {
        return;
    }
    let direction = normalize(inputs[index].xyz);
    outputs[2u * index] = vec4f(giSkyRadiance(direction), 0.0);
    outputs[2u * index + 1u] = vec4f(giSkyIrradianceAt(direction), 0.0);
}
`;

interface SkySamples {
    radiance: Vec3[];
    irradiance: Vec3[];
}

async function sampleSky(kernel: ShadeKernel, directions: readonly Vec3[]): Promise<SkySamples> {
    const inputs = new Float32Array(directions.length * 4);
    directions.forEach((direction, index) => inputs.set(direction, index * 4));
    const out = new Float32Array(await kernel.run(inputs, directions.length * 32, directions.length));
    return {
        radiance: directions.map((_, index) => [out[index * 8], out[index * 8 + 1], out[index * 8 + 2]]),
        irradiance: directions.map((_, index) => [out[index * 8 + 4], out[index * 8 + 5], out[index * 8 + 6]]),
    };
}

/** Directions spread over the sphere, away from the exact poles. */
function sphereDirections(count: number): Vec3[] {
    const directions: Vec3[] = [];
    const golden = Math.PI * (3 - Math.sqrt(5));
    for (let index = 0; index < count; index++) {
        const y = 0.97 - (1.94 * index) / (count - 1);
        const radius = Math.sqrt(1 - y * y);
        directions.push([radius * Math.cos(index * golden), y, radius * Math.sin(index * golden)]);
    }
    return directions;
}

/** R_y(-angle) d: where the unrotated source is read for world direction d (the contract's rotationY). */
function unrotate(direction: Vec3, angle: number): Vec3 {
    const c = Math.cos(angle);
    const s = Math.sin(angle);
    return [c * direction[0] - s * direction[2], direction[1], s * direction[0] + c * direction[2]];
}

/** three.js's equirectDirection for texture coordinates (v = 0 at the first row in memory, the nadir). */
function textureEquirectDirection(u: number, v: number): Vec3 {
    const azimuth = (u - 0.5) * 2 * Math.PI;
    const latitude = (v - 0.5) * Math.PI;
    return [Math.cos(latitude) * Math.cos(azimuth), Math.sin(latitude), Math.cos(latitude) * Math.sin(azimuth)];
}

/**
 * Direction of the centre of texel (s, t) of cube face `face`, from WebGPU's
 * cube addressing (the spec's face-selection table, as in Vulkan and D3D):
 * written out here independently of the platform's WGSL, so the hardware
 * lookup checks both.
 */
function cubeTexelDirection(face: number, column: number, row: number, size: number): Vec3 {
    const sc = ((column + 0.5) / size) * 2 - 1;
    const tc = ((row + 0.5) / size) * 2 - 1;
    const byFace: Vec3[] = [
        [1, -tc, -sc],
        [-1, -tc, sc],
        [sc, 1, tc],
        [sc, -1, -tc],
        [sc, -tc, 1],
        [-sc, -tc, -1],
    ];
    return normalize(byFace[face]);
}

const coded = (direction: Vec3): Vec3 => [(direction[0] + 1) / 2, (direction[1] + 1) / 2, (direction[2] + 1) / 2];

function codedEquirectTexture(device: GPUDevice, width: number, height: number, format: 'rgba16float' | 'rgba32float'): GPUTexture {
    const texture = device.createTexture({ label: `direction-coded equirect (${format})`, size: { width, height }, format, usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
    const values = new Float32Array(width * height * 4);
    for (let row = 0; row < height; row++) {
        for (let column = 0; column < width; column++) {
            const colour = coded(textureEquirectDirection((column + 0.5) / width, (row + 0.5) / height));
            values.set([...colour, 1], (row * width + column) * 4);
        }
    }
    if (format === 'rgba32float') {
        device.queue.writeTexture({ texture }, values, { bytesPerRow: width * 16 }, { width, height });
    } else {
        const halves = Uint16Array.from(values, (value) => floatToHalf(value));
        device.queue.writeTexture({ texture }, halves, { bytesPerRow: width * 8 }, { width, height });
    }
    return texture;
}

/**
 * A cube whose texels hold their own direction. With `threeConvention`, each
 * texel holds (-x, y, z) instead: what three.js r187 shows there, since it
 * looks every cube texture up at (-d.x, d.y, d.z) on WebGPU
 * (nodes/accessors/CubeTextureNode.js, setupUV).
 */
function codedCubeTexture(device: GPUDevice, size: number, threeConvention = false): GPUTexture {
    const texture = device.createTexture({ label: 'direction-coded cube', size: { width: size, height: size, depthOrArrayLayers: 6 }, format: 'rgba16float', usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST });
    for (let face = 0; face < 6; face++) {
        const halves = new Uint16Array(size * size * 4);
        for (let row = 0; row < size; row++) {
            for (let column = 0; column < size; column++) {
                const direction = cubeTexelDirection(face, column, row, size);
                const colour = coded(threeConvention ? [-direction[0], direction[1], direction[2]] : direction);
                halves.set([floatToHalf(colour[0]), floatToHalf(colour[1]), floatToHalf(colour[2]), floatToHalf(1)], (row * size + column) * 4);
            }
        }
        device.queue.writeTexture({ texture, origin: { x: 0, y: 0, z: face } }, halves, { bytesPerRow: size * 8 }, { width: size, height: size });
    }
    return texture;
}

/** Versions for the stand-alone skies: each call is a new environment. */
let standaloneVersion = 0;

/** The sky's harmonics as the bake writes them: irradiance SH (18 vec4f) and radiance SH bands 0-2 (9 vec4f). */
interface Harmonics {
    irradiance: Float32Array;
    radiance: Float32Array;
}

/** Bakes one environment with a stand-alone sky and reads the harmonics back. */
async function projectStandalone(device: GPUDevice, sky: PlatformSky, environment: EnvironmentDescriptor): Promise<Harmonics> {
    const bytes = SKY_HARMONICS_LAYOUT.byteLength;
    const output = device.createBuffer({ label: 'standalone sky harmonics', size: bytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const encoder = device.createCommandEncoder({ label: 'standalone sky bake' });
    const baked = sky.encode(encoder, environment, ++standaloneVersion, { buffer: output });
    assert(baked, 'the stand-alone sky did not bake a new environment');
    device.queue.submit([encoder.finish()]);
    const result = new Float32Array(await readBuffer(device, output, bytes));
    output.destroy();
    return {
        irradiance: result.slice(SKY_HARMONICS_LAYOUT.irradiance / 4, SKY_HARMONICS_LAYOUT.irradiance / 4 + SH_COEFFICIENT_COUNT * 4),
        radiance: result.slice(SKY_HARMONICS_LAYOUT.radiance / 4, SKY_HARMONICS_LAYOUT.radiance / 4 + RADIANCE_SH_COUNT * 4),
    };
}

function largestDifference(a: ArrayLike<number>, b: ArrayLike<number>, count = SH_COEFFICIENT_COUNT): { difference: number; scale: number } {
    let difference = 0;
    let scale = 0;
    for (let index = 0; index < count; index++) {
        for (let channel = 0; channel < 3; channel++) {
            difference = Math.max(difference, Math.abs(a[index * 4 + channel] - b[index * 4 + channel]));
            scale = Math.max(scale, Math.abs(b[index * 4 + channel]));
        }
    }
    return { difference, scale };
}

const percent = (value: number): string => `${(value * 100).toFixed(3)}%`;

type ProceduralEnvironment = Extract<EnvironmentDescriptor, { type: 'procedural' }>;

/** A unit direction toward a sun at `elevation` degrees, `azimuth` degrees from +X toward +Z. */
function sunAt(elevation: number, azimuth: number): Vec3 {
    const e = (elevation * Math.PI) / 180;
    const a = (azimuth * Math.PI) / 180;
    return [Math.cos(e) * Math.cos(a), Math.sin(e), Math.cos(e) * Math.sin(a)];
}

/** The CPU sky the platform's procedural bake must match: the sky files' builder, with the platform's clamps. */
function cpuSkyFor(environment: ProceduralEnvironment): ReturnType<typeof buildCalibratedSky> {
    return buildCalibratedSky({
        sunDirection: liftSunAboveMinimum(environment.sunDirection),
        turbidity: Math.min(MAX_TURBIDITY, Math.max(MIN_TURBIDITY, environment.turbidity ?? DEFAULT_TURBIDITY)),
        sunIlluminance: environment.sunIlluminance ?? 1,
        intensity: environment.intensity ?? 1,
    });
}

/**
 * The CPU sky (image order, 512 x 256) box filtered to the radiance map's
 * texels in texture order (row 0 the nadir), as the bake's resample filters
 * a 512 x 256 source: the 2 x 2 source texels in each map texel, weighted by
 * cos(latitude) at their centres. RGB per texel.
 */
function cpuRadianceMap(image: { width: number; height: number; halves: Uint16Array }): Float64Array {
    const { width, height, halves } = image;
    const across = width / RADIANCE_MAP_WIDTH;
    const down = height / RADIANCE_MAP_HEIGHT;
    const map = new Float64Array(RADIANCE_MAP_WIDTH * RADIANCE_MAP_HEIGHT * 3);
    for (let row = 0; row < RADIANCE_MAP_HEIGHT; row++) {
        for (let column = 0; column < RADIANCE_MAP_WIDTH; column++) {
            let weightSum = 0;
            const sum = [0, 0, 0];
            for (let j = 0; j < down; j++) {
                // The source row in texture order, its latitude at the centre, and its row in the image.
                const sourceRow = row * down + j;
                const weight = Math.cos(((sourceRow + 0.5) / height - 0.5) * Math.PI);
                const imageRow = height - 1 - sourceRow;
                for (let i = 0; i < across; i++) {
                    const base = (imageRow * width + column * across + i) * 3;
                    for (let channel = 0; channel < 3; channel++) sum[channel] += halfToFloat(halves[base + channel]) * weight;
                    weightSum += weight;
                }
            }
            for (let channel = 0; channel < 3; channel++) map[(row * RADIANCE_MAP_WIDTH + column) * 3 + channel] = sum[channel] / weightSum;
        }
    }
    return map;
}

/** Every radiance-map texel's centre as a direction, texel by texel in texture order. */
function radianceMapTexelDirections(): Vec3[] {
    const directions: Vec3[] = [];
    for (let row = 0; row < RADIANCE_MAP_HEIGHT; row++) {
        for (let column = 0; column < RADIANCE_MAP_WIDTH; column++) directions.push(textureEquirectDirection((column + 0.5) / RADIANCE_MAP_WIDTH, (row + 0.5) / RADIANCE_MAP_HEIGHT));
    }
    return directions;
}

interface MapComparison {
    /** Per channel: the sum of the GPU's texels over the CPU's, minus 1. */
    meanError: Vec3;
    /** The worst texel: its largest relative error per channel (a channel black on the CPU: over the texel's brightest). */
    worstTexelError: number;
    worstTexel: { column: number; row: number; gpu: Vec3; cpu: Vec3 };
    /** The mean of every lit texel's error, the same measure. */
    meanTexelError: number;
    /** Texels black on the CPU (below the horizon) that aren't black on the GPU. */
    litBelowHorizon: number;
}

function compareMaps(gpu: readonly Vec3[], cpu: Float64Array): MapComparison {
    const gpuSum = [0, 0, 0];
    const cpuSum = [0, 0, 0];
    let worst = 0;
    let worstIndex = 0;
    let errorSum = 0;
    let lit = 0;
    let litBelowHorizon = 0;
    gpu.forEach((texel, index) => {
        const reference: Vec3 = [cpu[index * 3], cpu[index * 3 + 1], cpu[index * 3 + 2]];
        const brightest = Math.max(...reference);
        for (let channel = 0; channel < 3; channel++) {
            gpuSum[channel] += texel[channel];
            cpuSum[channel] += reference[channel];
        }
        if (brightest === 0) {
            if (texel.some((value) => value !== 0)) litBelowHorizon++;
            return;
        }
        // Each channel against itself; a channel the gamut clamp took to 0 on the CPU against the texel's brightest.
        const error = Math.max(...texel.map((value, channel) => Math.abs(value - reference[channel]) / (reference[channel] > 0 ? reference[channel] : brightest)));
        errorSum += error;
        lit++;
        if (error > worst) {
            worst = error;
            worstIndex = index;
        }
    });
    const reference: Vec3 = [cpu[worstIndex * 3], cpu[worstIndex * 3 + 1], cpu[worstIndex * 3 + 2]];
    return {
        meanError: [gpuSum[0] / cpuSum[0] - 1, gpuSum[1] / cpuSum[1] - 1, gpuSum[2] / cpuSum[2] - 1],
        worstTexelError: worst,
        worstTexel: { column: worstIndex % RADIANCE_MAP_WIDTH, row: Math.floor(worstIndex / RADIANCE_MAP_WIDTH), gpu: gpu[worstIndex], cpu: reference },
        meanTexelError: errorSum / Math.max(1, lit),
        litBelowHorizon,
    };
}

/** A radiance map's SH (radiance coefficients), the map in texture order (row 0 the nadir), projected as the GPU does: its own texels. */
function projectMap(map: Float64Array): Float64Array {
    const imageOrder = new Float64Array(map.length);
    const rowLength = RADIANCE_MAP_WIDTH * 3;
    for (let row = 0; row < RADIANCE_MAP_HEIGHT; row++) imageOrder.set(map.subarray(row * rowLength, (row + 1) * rowLength), (RADIANCE_MAP_HEIGHT - 1 - row) * rowLength);
    return projectEquirectRadiance(imageOrder, RADIANCE_MAP_WIDTH, RADIANCE_MAP_HEIGHT);
}

/** The SH of an image's radiance map (irradiance and radiance), as vec4f rows like the GPU's. */
function cpuHarmonics(image: { width: number; height: number; halves: Uint16Array }): Harmonics {
    const radiance = projectMap(cpuRadianceMap(image));
    const irradiance = convolveWithCosineLobe(radiance);
    const rows = (coefficients: Float64Array, count: number): Float32Array => {
        const out = new Float32Array(count * 4);
        for (let index = 0; index < count; index++) out.set(coefficients.subarray(index * 3, index * 3 + 3), index * 4);
        return out;
    };
    return { irradiance: rows(irradiance, SH_COEFFICIENT_COUNT), radiance: rows(radiance, RADIANCE_SH_COUNT) };
}

/** T2.6a's skies: low, middle and high suns, clear to hazy, and the brightness numbers. */
const PROCEDURAL_SKIES: ReadonlyArray<{ name: string; environment: ProceduralEnvironment }> = [
    { name: 'sun 2 degrees (made at 5), turbidity 2', environment: { type: 'procedural', sunDirection: sunAt(2, 40), turbidity: 2 } },
    { name: 'sun 12 degrees, turbidity 3', environment: { type: 'procedural', sunDirection: sunAt(12, 200), turbidity: 3 } },
    { name: 'sun 30 degrees, turbidity 6, sunIlluminance 3', environment: { type: 'procedural', sunDirection: sunAt(30, -75), turbidity: 6, sunIlluminance: 3 } },
    { name: 'sun 60 degrees, turbidity 10, intensity 0.5', environment: { type: 'procedural', sunDirection: sunAt(60, 130), turbidity: 10, intensity: 0.5 } },
    { name: 'sun 89 degrees, turbidity 1.7', environment: { type: 'procedural', sunDirection: sunAt(89, 10), turbidity: 1.7 } },
];

await runGpuTest(
    async ({ device, log }) => {
        const results: Record<string, unknown> = {};
        const skyImage = await loadSkyFile(openGroundSkyUrl);
        const openGroundSidecar = await fetchJson<SkySidecar>(openGroundSidecarUrl);
        const sponzaSidecar = await fetchJson<SkySidecar>(sponzaSidecarUrl);
        const openGroundManifest = await fetchJson<SceneManifest>(openGroundManifestUrl);
        const sponzaManifest = await fetchJson<SceneManifest>(sponzaManifestUrl);
        const skyFloats = Float64Array.from(skyImage.halves, (half) => halfToFloat(half));

        // The sky file as the platform's texture: its own rows flipped into the contract's order.
        const skyTexture = device.createTexture({
            label: 'open-ground sky',
            size: { width: skyImage.width, height: skyImage.height },
            format: 'rgba16float',
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
        });
        device.queue.writeTexture({ texture: skyTexture }, halfImageToTextureRgba(skyImage) as Uint16Array<ArrayBuffer>, { bytesPerRow: skyImage.width * 8 }, { width: skyImage.width, height: skyImage.height });

        const platform = new GIPlatformImpl(device, { label: 'hit-shading-sky' });
        try {
            await ready(device, platform);
            const kernel = await ShadeKernel.create(device, platform, 'sky samples', SKY_KERNEL);
            const sky = platform.shade.sky;
            const setEnvironment = (environment: EnvironmentDescriptor): void => {
                platform.scene.setEnvironment(environment);
                frame(device, platform);
            };
            const up: Vec3 = [0, 1, 0];

            // 1. Open ground: the sky file as a texture, against its own integral.
            setEnvironment({ type: 'texture', texture: skyTexture, layout: 'equirect' });
            const tilted: Vec3[] = [up, normalize([1, 1, 0]), [1, 0, 0], normalize([-0.5, 0, 0.866]), normalize([0.3, -0.2, -0.9]), normalize([0, -1, 0.001])];
            const fileSamples = await sampleSky(kernel, tilted);
            const ground = fileSamples.irradiance[0];
            const reference = openGroundSidecar.measured.skyHorizontalIrradianceRgb;
            const groundErrors = ground.map((value, channel) => value / reference[channel] - 1);
            results.openGroundGroundIrradiance = ground;
            results.openGroundReference = reference;
            results.openGroundError = groundErrors.map(percent);
            log(`open ground, texture: ${ground.map((v) => v.toFixed(5)).join(' ')} against ${reference.join(' ')}: ${groundErrors.map(percent).join(' / ')}`);
            for (const [channel, error] of groundErrors.entries()) {
                assert(Math.abs(error) <= REFERENCE_TOLERANCE, `open ground: channel ${'RGB'[channel]} sky irradiance is off by ${percent(error)} (limit ±0.5%)`);
            }

            // 3. Tilted surfaces against brute force, and radiance on misses against the file.
            const luminance = (rgb: readonly number[]): number => 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2];
            const groundLuminance = luminance(reference);
            const tiltedErrors = tilted.map((normal, index) => {
                const exact = bruteForceIrradiance(skyFloats, skyImage.width, skyImage.height, normal);
                return (luminance(fileSamples.irradiance[index]) - luminance(exact)) / groundLuminance;
            });
            results.tiltedErrorsOfGround = tiltedErrors.map(percent);
            log(`tilted normals (luminance error / ground irradiance): ${tiltedErrors.map(percent).join(', ')}`);
            tiltedErrors.forEach((error, index) => assert(Math.abs(error) <= TILTED_TOLERANCE, `tilted normal ${tilted[index].map((v) => v.toFixed(2))}: ${percent(error)} of the ground's irradiance (limit 2%)`));

            const missDirections = sphereDirections(64).filter((direction) => direction[1] > 0.05);
            const missSamples = await sampleSky(kernel, missDirections);
            let worstMiss = 0;
            missDirections.forEach((direction, index) => {
                const expected = sampleEquirectBilinear(skyFloats, skyImage.width, skyImage.height, direction);
                worstMiss = Math.max(worstMiss, relativeError(luminance(missSamples.radiance[index]), luminance(expected), 1e-6));
            });
            results.missRadianceWorstError = percent(worstMiss);
            log(`radiance on a miss, ${missDirections.length} directions above the horizon: worst ${percent(worstMiss)}`);
            assert(worstMiss <= MISS_RADIANCE_TOLERANCE, `radiance on a miss is off by up to ${percent(worstMiss)} (limit 2%)`);

            // 2. procedural: the same calibrated sky, built by the platform.
            setEnvironment(platformSkyEnvironment(openGroundManifest));
            const proceduralGround = (await sampleSky(kernel, [up])).irradiance[0];
            const proceduralErrors = proceduralGround.map((value, channel) => value / reference[channel] - 1);
            results.openGroundProceduralError = proceduralErrors.map(percent);
            log(`open ground, procedural: ${proceduralErrors.map(percent).join(' / ')}`);
            proceduralErrors.forEach((error, channel) => assert(Math.abs(error) <= REFERENCE_TOLERANCE, `open ground procedural: channel ${'RGB'[channel]} off by ${percent(error)}`));

            setEnvironment(platformSkyEnvironment(sponzaManifest));
            const sponzaGround = (await sampleSky(kernel, [up])).irradiance[0];
            const sponzaReference = sponzaSidecar.measured.skyHorizontalIrradianceRgb;
            const sponzaErrors = sponzaGround.map((value, channel) => value / sponzaReference[channel] - 1);
            results.sponzaProceduralError = sponzaErrors.map(percent);
            log(`sponza, procedural (sun from its directional light): ${sponzaErrors.map(percent).join(' / ')}`);
            sponzaErrors.forEach((error, channel) => assert(Math.abs(error) <= REFERENCE_TOLERANCE, `sponza procedural: channel ${'RGB'[channel]} off by ${percent(error)}`));

            // procedural's two brightness numbers: by default the sky of a sun of illuminance 1;
            // sunIlluminance 2 and intensity 2 each double it (radiance on a miss and irradiance).
            const sunDirection = openGroundManifest.environment.type === 'procedural' ? openGroundManifest.environment.sunDirection : up;
            const unitSun = buildCalibratedSky({ sunDirection, turbidity: 3, sunIlluminance: 1 });
            const brightnessProbes: Vec3[] = [up, ...missDirections.slice(0, 12)];
            setEnvironment({ type: 'procedural', sunDirection, turbidity: 3 });
            const defaults = await sampleSky(kernel, brightnessProbes);
            const defaultErrors = defaults.irradiance[0].map((value, channel) => value / unitSun.measured.rgb[channel] - 1);
            assert(defaultErrors.every((error) => Math.abs(error) <= REFERENCE_TOLERANCE), `procedural defaults: ground irradiance off the sky of a sun of 1 by ${defaultErrors.map(percent)}`);
            const doubling: Record<string, number> = {};
            for (const [name, extra] of [['sunIlluminance 2', { sunIlluminance: 2 }], ['intensity 2', { intensity: 2 }]] as const) {
                setEnvironment({ type: 'procedural', sunDirection, turbidity: 3, ...extra });
                const doubled = await sampleSky(kernel, brightnessProbes);
                let worst = 0;
                brightnessProbes.forEach((_, index) => {
                    for (let channel = 0; channel < 3; channel++) {
                        worst = Math.max(worst, relativeError(doubled.radiance[index][channel], 2 * defaults.radiance[index][channel], 1e-6));
                        worst = Math.max(worst, relativeError(doubled.irradiance[index][channel], 2 * defaults.irradiance[index][channel], 1e-6));
                    }
                });
                doubling[name] = worst;
                assert(worst <= 1e-3, `procedural ${name} should double the sky; off by ${percent(worst)}`);
            }
            results.proceduralDefaultsError = defaultErrors.map(percent);
            results.proceduralDoubling = doubling;
            log(`procedural defaults: ground ${defaultErrors.map(percent).join(' / ')} against a sun of 1; sunIlluminance 2 and intensity 2 double it (worst ${Object.values(doubling).map((v) => v.toExponential(1)).join(', ')})`);

            // 4. The equirect layout, with direction-coded textures.
            const probes = sphereDirections(40);
            for (const variant of [
                { format: 'rgba16float' as const, rotationY: 0, intensity: 1 },
                { format: 'rgba16float' as const, rotationY: 0.7, intensity: 2 },
                // rgba32float can't be filtered, so it is read with nearest samples. Unrotated, every
                // projection cell lands on a texel centre and that is exact; turned, nearest samples
                // shift the sky by up to half a source texel (0.7 degrees here), which is what they do.
                { format: 'rgba32float' as const, rotationY: 0, intensity: 0.5 },
            ]) {
                const texture = codedEquirectTexture(device, 256, 128, variant.format);
                setEnvironment({ type: 'texture', texture, layout: 'equirect', rotationY: variant.rotationY, intensity: variant.intensity });
                const samples = await sampleSky(kernel, probes);
                let worstRadiance = 0;
                let worstIrradiance = 0;
                probes.forEach((direction, index) => {
                    const source = unrotate(direction, variant.rotationY);
                    const expectedRadiance = coded(source).map((value) => value * variant.intensity);
                    const expectedIrradiance = source.map((value) => (Math.PI / 2 + (Math.PI / 3) * value) * variant.intensity);
                    for (let channel = 0; channel < 3; channel++) {
                        worstRadiance = Math.max(worstRadiance, Math.abs(samples.radiance[index][channel] - expectedRadiance[channel]) / variant.intensity);
                        worstIrradiance = Math.max(worstIrradiance, relativeError(samples.irradiance[index][channel], expectedIrradiance[channel], 1));
                    }
                });
                const name = `equirect ${variant.format}, rotationY ${variant.rotationY}, intensity ${variant.intensity}`;
                results[name] = { worstRadiance, worstIrradiance };
                log(`${name}: radiance worst ${worstRadiance.toExponential(2)}, irradiance worst ${percent(worstIrradiance)}`);
                assert(worstRadiance <= CODED_RADIANCE_TOLERANCE, `${name}: radiance off by ${worstRadiance} (the layout or rotation is wrong)`);
                assert(worstIrradiance <= CODED_IRRADIANCE_TOLERANCE, `${name}: irradiance off by ${percent(worstIrradiance)}`);
                texture.destroy();
            }

            const cube = codedCubeTexture(device, 64);
            setEnvironment({ type: 'texture', texture: cube, layout: 'cube' });
            const cubeSamples = await sampleSky(kernel, probes);
            let cubeRadiance = 0;
            let cubeIrradiance = 0;
            probes.forEach((direction, index) => {
                for (let channel = 0; channel < 3; channel++) {
                    cubeRadiance = Math.max(cubeRadiance, Math.abs(cubeSamples.radiance[index][channel] - coded(direction)[channel]));
                    cubeIrradiance = Math.max(cubeIrradiance, relativeError(cubeSamples.irradiance[index][channel], Math.PI / 2 + (Math.PI / 3) * direction[channel], 1));
                }
            });
            results.cube = { worstRadiance: cubeRadiance, worstIrradiance: cubeIrradiance };
            log(`cube: radiance worst ${cubeRadiance.toExponential(2)}, irradiance worst ${percent(cubeIrradiance)}`);
            assert(cubeRadiance <= CODED_RADIANCE_TOLERANCE, `cube: radiance off by ${cubeRadiance}`);
            assert(cubeIrradiance <= CODED_IRRADIANCE_TOLERANCE, `cube: irradiance off by ${percent(cubeIrradiance)}`);
            cube.destroy();

            // A cube made for three.js, passed straight in, comes out mirrored in x (the contract notes this
            // for the three adapter): along d the platform sees what three shows along (-d.x, d.y, d.z).
            const threeCube = codedCubeTexture(device, 64, true);
            setEnvironment({ type: 'texture', texture: threeCube, layout: 'cube' });
            const threeCubeSamples = await sampleSky(kernel, probes);
            let mirrored = 0;
            probes.forEach((direction, index) => {
                const seenByThree = coded([-direction[0], direction[1], direction[2]]);
                for (let channel = 0; channel < 3; channel++) mirrored = Math.max(mirrored, Math.abs(threeCubeSamples.radiance[index][channel] - seenByThree[channel]));
            });
            results.threeConventionCubeMirrored = mirrored;
            log(`a three-convention cube passed straight in reads as three's x-mirror (worst ${mirrored.toExponential(2)})`);
            assert(mirrored <= CODED_RADIANCE_TOLERANCE, `a three-convention cube should read as the x-mirror of what three shows (off by ${mirrored})`);
            threeCube.destroy();

            // 5. color and none.
            setEnvironment({ type: 'color', color: [0.2, 0.5, 1], intensity: 3 });
            const colourSamples = await sampleSky(kernel, probes);
            const colour: Vec3 = [0.6, 1.5, 3];
            let worstColour = 0;
            probes.forEach((_, index) => {
                for (let channel = 0; channel < 3; channel++) {
                    worstColour = Math.max(worstColour, relativeError(colourSamples.radiance[index][channel], colour[channel], 1e-6));
                    worstColour = Math.max(worstColour, relativeError(colourSamples.irradiance[index][channel], Math.PI * colour[channel], 1e-6));
                }
            });
            results.colorWorst = worstColour;
            assert(worstColour <= 1e-3, `color environment off by ${percent(worstColour)}`);
            setEnvironment({ type: 'none' });
            const noneSamples = await sampleSky(kernel, probes);
            assert(noneSamples.radiance.every((rgb) => rgb.every((value) => value === 0)) && noneSamples.irradiance.every((rgb) => rgb.every((value) => value === 0)), "'none' must be black");

            // 6. Projection only when the environment changes.
            const before = sky.projections;
            for (let index = 0; index < 3; index++) frame(device, platform);
            platform.scene.createLight({ type: 'point', color: [1, 1, 1], intensity: 1, position: [0, 1, 0], range: 0 });
            frame(device, platform);
            assert(sky.projections === before, `the sky was projected ${sky.projections - before} times without an environment change`);
            setEnvironment({ type: 'color', color: [1, 1, 1] });
            assert(sky.projections === before + 1, 'a new environment must be projected exactly once');
            // The same object again, after its texture's contents changed, must be projected again (contract/scene.ts).
            const changing = codedEquirectTexture(device, 64, 32, 'rgba16float');
            const changingEnvironment: EnvironmentDescriptor = { type: 'texture', texture: changing, layout: 'equirect' };
            setEnvironment(changingEnvironment);
            const constantHalves = new Uint16Array(64 * 32 * 4).fill(floatToHalf(0.25));
            device.queue.writeTexture({ texture: changing }, constantHalves, { bytesPerRow: 64 * 8 }, { width: 64, height: 32 });
            setEnvironment(changingEnvironment);
            assert(sky.projections === before + 3, `setEnvironment with the same object must re-project (projections ${sky.projections - before}, expected 3)`);
            const refreshed = await sampleSky(kernel, probes);
            const worstRefreshed = Math.max(...refreshed.radiance.flat().map((value) => Math.abs(value - 0.25)));
            assert(worstRefreshed <= 1e-3, `after the same object was set again, the sky still shows the old contents (off by ${worstRefreshed})`);
            changing.destroy();
            results.projectionsCounted = true;

            // 8. T2.6a: the GPU's procedural sky against the CPU sky.
            const texelDirections = radianceMapTexelDirections();
            const proceduralResults: Record<string, unknown> = {};
            const skies = [{ name: 'open ground (manifest)', environment: platformSkyEnvironment(openGroundManifest) as ProceduralEnvironment }, ...PROCEDURAL_SKIES];
            for (const { name, environment } of skies) {
                setEnvironment(environment);
                const samples = await sampleSky(kernel, [up, ...texelDirections]);
                const cpuSky = cpuSkyFor(environment);
                const comparison = compareMaps(samples.radiance.slice(1), cpuRadianceMap(cpuSky.image));
                const groundError = samples.irradiance[0].map((value, channel) => value / cpuSky.measured.rgb[channel] - 1);
                proceduralResults[name] = { ...comparison, groundIrradianceError: groundError.map(percent) };
                log(
                    `procedural ${name}: map mean ${comparison.meanError.map(percent).join(' / ')}, texels mean ${percent(comparison.meanTexelError)}, worst ${percent(comparison.worstTexelError)} ` +
                        `(texel ${comparison.worstTexel.column}, ${comparison.worstTexel.row}); ground irradiance ${groundError.map(percent).join(' / ')} against the CPU sky's own`,
                );
                comparison.meanError.forEach((error, channel) => assert(Math.abs(error) <= PROCEDURAL_MEAN_TOLERANCE, `procedural ${name}: channel ${'RGB'[channel]}'s mean is off the CPU sky by ${percent(error)} (limit 0.1%)`));
                assert(
                    comparison.worstTexelError <= PROCEDURAL_TEXEL_TOLERANCE,
                    `procedural ${name}: texel ${comparison.worstTexel.column}, ${comparison.worstTexel.row} is off the CPU sky by ${percent(comparison.worstTexelError)} in a channel (limit 1%)`,
                );
                assert(comparison.litBelowHorizon === 0, `procedural ${name}: ${comparison.litBelowHorizon} texels below the horizon aren't black`);
            }
            results.procedural = proceduralResults;

            // 9. When a procedural sky re-bakes.
            const baseSky: ProceduralEnvironment = { type: 'procedural', sunDirection: sunAt(30, 20), turbidity: 3 };
            setEnvironment(baseSky);
            const later: ProceduralEnvironment = { ...baseSky, sunDirection: sunAt(30.06, 20), turbidity: 3.01, intensity: 0.99, sunIlluminance: 1.01 };
            const rebakeCases: Array<[string, ProceduralEnvironment, boolean]> = [
                ['the same numbers again', { ...baseSky }, false],
                ['the sun 0.03 degrees higher', { ...baseSky, sunDirection: sunAt(30.03, 20) }, false],
                ['the sun 0.06 degrees higher than the last bake', { ...baseSky, sunDirection: sunAt(30.06, 20) }, true],
                ['turbidity 3.01', { ...baseSky, sunDirection: sunAt(30.06, 20), turbidity: 3.01 }, true],
                ['intensity 0.99', { ...baseSky, sunDirection: sunAt(30.06, 20), turbidity: 3.01, intensity: 0.99 }, true],
                ['sunIlluminance 1.01', later, true],
                ['the sun 0.04 degrees along', { ...later, sunDirection: sunAt(30.06, 20.04) }, false],
                ['0.04 degrees more (0.08 from the last bake)', { ...later, sunDirection: sunAt(30.06, 20.08) }, true],
            ];
            for (const [what, environment, expected] of rebakeCases) {
                const before = sky.projections;
                setEnvironment(environment);
                const baked = sky.projections > before;
                assert(baked === expected, `procedural re-bake rule: ${what} ${baked ? 'baked' : 'did not bake'} (expected ${expected ? 'a bake' : 'none'})`);
            }
            results.rebakeRule = rebakeCases.map(([what, , expected]) => `${what}: ${expected ? 'bakes' : 'no bake'}`);
            log(`procedural re-bake rule: ${rebakeCases.map(([what, , expected]) => `${what} ${expected ? 'bakes' : 'does not'}`).join('; ')}`);
        } finally {
            platform.dispose();
        }

        // 7b. Subgroups against the declared fallback, on the same sky.
        const hasSubgroups = device.features.has('subgroups');
        results.subgroups = hasSubgroups;
        const fallback = new PlatformSky(device, { label: 'fallback', reduction: 'workgroup' });
        await fallback.prepare();
        const skyEnvironment: EnvironmentDescriptor = { type: 'texture', texture: skyTexture, layout: 'equirect' };
        const fallbackHarmonics = await projectStandalone(device, fallback, skyEnvironment);
        const fallbackSh = fallbackHarmonics.irradiance;
        // Since T2.6a the SH integrate the radiance map: against the CPU projection of the file's map.
        const cpuShVec4 = cpuHarmonics(skyImage).irradiance;
        const fallbackVsCpu = largestDifference(fallbackSh, cpuShVec4);
        results.shFallbackVsCpu = fallbackVsCpu.difference / fallbackVsCpu.scale;
        // What integrating the 256 x 128 map instead of the 512 x 256 file changes (reported).
        const fileSh = convolveWithCosineLobe(projectEquirectRadiance(skyFloats, skyImage.width, skyImage.height));
        const fileShVec4 = new Float32Array(SH_COEFFICIENT_COUNT * 4);
        for (let index = 0; index < SH_COEFFICIENT_COUNT; index++) fileShVec4.set(fileSh.subarray(index * 3, index * 3 + 3), index * 4);
        const mapVsFile = largestDifference(cpuShVec4, fileShVec4);
        results.shMapVsFile = mapVsFile.difference / mapVsFile.scale;
        log(
            `SH, workgroup fallback against the CPU projection of the map: largest difference ${(fallbackVsCpu.difference / fallbackVsCpu.scale).toExponential(2)} of the largest coefficient; ` +
                `the map's SH against the file's (both on the CPU): ${(mapVsFile.difference / mapVsFile.scale).toExponential(2)}`,
        );
        assert(fallbackVsCpu.difference <= MAP_SH_TOLERANCE * fallbackVsCpu.scale, 'the fallback SH reduction disagrees with the CPU projection');
        // 2 (continued). The procedural open-ground sky is the sky file, coefficient for coefficient.
        const proceduralSh = (await projectStandalone(device, fallback, platformSkyEnvironment(openGroundManifest))).irradiance;
        const proceduralVsTexture = largestDifference(proceduralSh, fallbackSh);
        results.shProceduralVsTexture = proceduralVsTexture.difference / proceduralVsTexture.scale;
        log(`SH, procedural open ground against the sky file: largest difference ${(proceduralVsTexture.difference / proceduralVsTexture.scale).toExponential(2)}`);
        assert(proceduralVsTexture.difference <= MAP_SH_TOLERANCE * proceduralVsTexture.scale, 'the procedural open-ground sky differs from the sky file');
        // 8 (continued). The procedural skies' SH against the CPU projection of the CPU sky, radiance and irradiance.
        const proceduralShAgreement: Record<string, { irradiance: number; radiance: number }> = {};
        const fallbackProceduralSh = new Map<string, Harmonics>();
        for (const { name, environment } of PROCEDURAL_SKIES) {
            const gpu = await projectStandalone(device, fallback, environment);
            fallbackProceduralSh.set(name, gpu);
            const cpu = cpuHarmonics(cpuSkyFor(environment).image);
            const irradianceAgreement = largestDifference(gpu.irradiance, cpu.irradiance);
            const radianceAgreement = largestDifference(gpu.radiance, cpu.radiance, RADIANCE_SH_COUNT);
            proceduralShAgreement[name] = { irradiance: irradianceAgreement.difference / irradianceAgreement.scale, radiance: radianceAgreement.difference / radianceAgreement.scale };
            assert(irradianceAgreement.difference <= MAP_SH_TOLERANCE * irradianceAgreement.scale, `procedural ${name}: the irradiance SH differs from the CPU sky's by ${proceduralShAgreement[name].irradiance.toExponential(2)} of the largest coefficient`);
            assert(radianceAgreement.difference <= MAP_SH_TOLERANCE * radianceAgreement.scale, `procedural ${name}: the radiance SH differs from the CPU sky's by ${proceduralShAgreement[name].radiance.toExponential(2)} of the largest coefficient`);
        }
        results.proceduralShVsCpu = proceduralShAgreement;
        log(`SH, procedural skies against the CPU projection of the CPU sky (largest difference over the largest coefficient, irradiance / radiance): ${Object.entries(proceduralShAgreement).map(([name, value]) => `${name} ${value.irradiance.toExponential(1)} / ${value.radiance.toExponential(1)}`).join('; ')}`);
        if (hasSubgroups) {
            const withSubgroups = new PlatformSky(device, { label: 'subgroups', reduction: 'subgroups' });
            await withSubgroups.prepare();
            const subgroupSh = (await projectStandalone(device, withSubgroups, skyEnvironment)).irradiance;
            const agreement = largestDifference(subgroupSh, fallbackSh);
            results.shSubgroupsVsFallback = agreement.difference / agreement.scale;
            log(`SH, subgroups against the workgroup fallback: largest difference ${(agreement.difference / agreement.scale).toExponential(2)} of the largest coefficient`);
            assert(agreement.difference <= SH_AGREEMENT_TOLERANCE * agreement.scale, 'subgroups and the fallback reduce the sky to different SH');
            // The procedural skies too: radiance and irradiance (the GPU's own Preetham, and its calibration sum).
            let worstProcedural = 0;
            for (const { name, environment } of PROCEDURAL_SKIES) {
                const withSubgroupsSh = await projectStandalone(device, withSubgroups, environment);
                const withFallbackSh = fallbackProceduralSh.get(name)!;
                const irradiance = largestDifference(withSubgroupsSh.irradiance, withFallbackSh.irradiance);
                const radiance = largestDifference(withSubgroupsSh.radiance, withFallbackSh.radiance, RADIANCE_SH_COUNT);
                worstProcedural = Math.max(worstProcedural, irradiance.difference / irradiance.scale, radiance.difference / radiance.scale);
            }
            results.shSubgroupsVsFallbackProcedural = worstProcedural;
            log(`SH, procedural skies, subgroups against the fallback: largest difference ${worstProcedural.toExponential(2)} of the largest coefficient`);
            assert(worstProcedural <= SH_AGREEMENT_TOLERANCE, 'subgroups and the fallback bake the procedural skies to different SH');
            withSubgroups.dispose();
        } else {
            log('this device has no subgroups: only the fallback ran');
        }
        fallback.dispose();
        skyTexture.destroy();
        return results;
    },
    { optionalFeatures: ['subgroups'] },
);
