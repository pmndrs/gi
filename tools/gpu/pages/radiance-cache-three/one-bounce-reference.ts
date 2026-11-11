/**
 * The like-for-like brightness check's diagnostic (RC2.2): at a sample of the
 * radiance cache's probes, the irradiance E four ways against a path-traced
 * one-bounce reference at the same point and normal. Not a page itself.
 *
 * - **Reference:** cosine-sampled rays from the probe's own ray origin (the
 *   pixel's position pushed off the surface, as its probe rays start), each
 *   hit shaded with the one bounce the world cache holds before RC3.3's
 *   multi-bounce cells: an emissive front face its emission, any other hit
 *   (back faces too, as Capsaicin) `albedo / π ×` (`giDirectIrradiance`, the
 *   platform's direct light with its shadow ray, from the same candidates the
 *   cache's `GenerateReservoirs` draws its one light from, plus the sky:
 *   one `giSkyLightSample` direction and its shadow ray, as the cache's hits
 *   get the sky since RC6.1, register D33), the sky on a miss. Before RC6.1
 *   neither the cache nor this reference had the sky at hits. The cache stores that light per cell, averaged over the cell's hits
 *   and up to 16 frames and read back filtered, so the probes' rays measure
 *   the cache's discretisation against this exact per-point value. Over the
 *   platform's own tracer and hit shading, so it shares both techniques'
 *   scene and lights exactly (the ddgi-irradiance-check page's path tracer,
 *   one bounce). DDGI's rays differ in one way: a back face is black there
 *   (`backFaceRayShare` says how many probe rays that touches).
 * - **The probe's rays:** `2π / 64 × Σ L cos θ` over its 64 rays (one per
 *   equal-area cell), from the ray results: the Monte Carlo estimate before
 *   any SH. Says whether the tracing and the lighting at the hits are right.
 * - **The probe's SH** along its normal (`ProjectScreenProbes`' output,
 *   evaluated as `InterpolateScreenProbes` does): the rays' estimate times the
 *   projection's 2.1% (register D25, DECISIONS 2026-11-10).
 * - **The interpolated pixel** at the probe's seed: what the composite reads.
 * - **DDGI's field** (`ddgiSampleIrradiance`) at the same point and normal.
 *
 * Each is reported as a ratio of sums over the sampled probes (the mean of
 * the rgb channels), so dim probes weigh less than bright ones, as in the
 * picture's mean.
 */

import { shIrradiance, unpackHalf3, unpackSeed, type Direction } from '../../../../packages/ddgi-core/src/techniques/radiance-cache/capsaicin-math.ts';
import type { RadianceCacheProbes } from '../../../../packages/ddgi-core/src/techniques/radiance-cache/radiance-cache-technique.ts';
import { PROBE_RAY_BYTES, PROBE_RAYS_HEADER_BYTES, PROBE_SPAWN_SEEDS_OFFSET } from '../../../../packages/ddgi-core/src/techniques/radiance-cache/wgsl/screen-probes.ts';
import { SH_BLOCK_SIZE, SH_COEFFICIENTS } from '../../../../packages/ddgi-core/src/techniques/radiance-cache/wgsl/spherical-harmonics.ts';
import { PROBE_SIZE } from '../../../../packages/ddgi-core/src/techniques/radiance-cache/settings.ts';
import { RAY_RESULT_BYTES, RAY_SCRATCH_HEADER_BYTES } from '../../../../packages/ddgi-core/src/platform/shade/direct.ts';
import type { FrameScreen } from '../../../../packages/ddgi-core/src/platform/contract/frame.ts';
import { FieldProbe, readBuffer, type SurfacePoint } from '../ddgi-shared/harness.ts';
import type { CompositeScene } from '../three-composite-shared/composite-check.ts';
import { halves, readTexture } from '../radiance-cache-probes/readback.ts';

/** Rays per reference point. */
const REFERENCE_PATHS = 4096;
/** Every this-many-th probe is sampled. */
const PROBE_STRIDE = 3;
const RAYS_PER_PROBE = PROBE_SIZE * PROBE_SIZE;
const CELL_SOLID_ANGLE = (2 * Math.PI) / RAYS_PER_PROBE;

const REFERENCE_WGSL = /* wgsl */ `
struct RefPoint { position: vec4f, normal: vec4f }
@group(1) @binding(0) var<storage, read> refPoints: array<RefPoint>;
@group(1) @binding(1) var<storage, read_write> refResults: array<vec4f>;
@group(1) @binding(2) var<uniform> refSettings: vec4u; // paths per point, seed

fn refHash(value: u32) -> u32 {
    let state = value * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}
var<private> refState: u32;
fn refRandom() -> f32 {
    refState = refHash(refState);
    return f32(refState >> 8u) * (1.0 / 16777216.0);
}
fn refCosineDirection(normal: vec3f) -> vec3f {
    let u = refRandom();
    let v = refRandom();
    let radius = sqrt(u);
    let angle = 6.28318530718 * v;
    let helper = select(vec3f(1.0, 0.0, 0.0), vec3f(0.0, 1.0, 0.0), abs(normal.x) > 0.9);
    let tangent = normalize(cross(helper, normal));
    let bitangent = cross(normal, tangent);
    return normalize(tangent * radius * cos(angle) + bitangent * radius * sin(angle) + normal * sqrt(max(0.0, 1.0 - u)));
}

// One workgroup per point, 64 threads sharing its rays. One bounce, the light the world cache holds per hit.
@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) group: vec3u, @builtin(local_invocation_index) thread: u32) {
    let pointIndex = group.x;
    if (pointIndex >= arrayLength(&refPoints)) {
        return;
    }
    let point = refPoints[pointIndex];
    refState = refHash(pointIndex * 1000003u + thread * 7919u + refSettings.y);
    var sum = vec3f(0.0);
    let paths = refSettings.x / 64u;
    for (var path = 0u; path < paths; path++) {
        let direction = refCosineDirection(point.normal.xyz);
        let ray = GiRay(point.position.xyz, 0.0, direction, 1.0e27);
        let hit = giTraceClosest(ray);
        if (hit.t < 0.0) {
            sum += giSkyRadiance(direction);
            continue;
        }
        // As the world cache lights a hit (and Capsaicin, gi1.comp:2427-2452): an emissive front face gives its
        // emission alone; any other hit, a back face too (giSurfaceAt turns its normal to the ray's
        // side), reflects its direct light.
        let surface = giSurfaceAt(ray, hit);
        if (surface.frontFace && dot(surface.emission, surface.emission) > 0.0) {
            sum += surface.emission;
        } else {
            // The sky as a light (RC6.1): one uniform direction over the hemisphere, kept if nothing blocks it.
            let sky = giSkyLightSample(surface, vec2f(refRandom(), refRandom()));
            let skyIrradiance = select(sky.irradiance, vec3f(0.0), giTraceAny(sky.shadowRay));
            sum += surface.albedo * 0.318309886 * (giDirectIrradiance(surface, 0xfu, vec2f(refRandom(), refRandom())) + skyIrradiance);
        }
    }
    refResults[pointIndex * 64u + thread] = vec4f(sum, f32(paths));
}
`;

export interface ProbeComparison {
    readonly sampledProbes: number;
    /** Σ E over the sampled probes, mean of rgb, for each estimate. */
    readonly referenceE: number;
    readonly raysOverReference: number;
    readonly shOverReference: number;
    readonly pixelOverReference: number;
    readonly ddgiOverReference: number;
    /** Share of the sampled probes' rays that hit a back face (lit by the probes and the reference, black to DDGI's rays). */
    readonly backFaceRayShare: number;
    /**
     * The D17 check (self-shadowing at grazing angles): the sampled probes
     * split by how the camera sees their surface, grazing (the normal within
     * {@link GRAZING_COSINE} of the view's perpendicular) or not. Per group:
     * how many probes, the share of their rays that hit something closer
     * than {@link SHORT_HIT_METRES} (a ray caught by its own surface, from an
     * origin rebuilt from depth or a shading normal tilted off the triangle,
     * ends that close), the share hitting a back face, and the rays' E over
     * the reference twice: from the same origin, and from the origin lifted
     * {@link LIFT_METRES} along the normal (out of reach of its own surface;
     * a probe darkened by self-shadowing reads well under this one).
     */
    readonly byView: Record<'grazing' | 'facing', ViewGroup>;
}

export interface ViewGroup {
    readonly probes: number;
    readonly shortHitShare: number;
    readonly backFaceShare: number;
    readonly raysOverReference: number;
    readonly raysOverLiftedReference: number;
}

/** A probe's surface is grazing when |n · v| is below this (about 76 degrees off the view). */
const GRAZING_COSINE = 0.25;
/** A ray that hits closer than this counts as caught by its own surface. */
const SHORT_HIT_METRES = 0.01;
/** The lifted reference's origin, along the normal from the probe's own. */
const LIFT_METRES = 0.02;

const meanOf = (rgb: readonly number[]): number => (rgb[0] + rgb[1] + rgb[2]) / 3;

/** One-bounce irradiance at each point by path tracing (file comment). */
async function referenceIrradiance(device: GPUDevice, scene: CompositeScene, points: readonly SurfacePoint[]): Promise<number[]> {
    const kernel = await scene.platform.kernels.computeKernel({
        label: 'radiance-cache one-bounce reference',
        modules: [scene.platform.shade.module],
        code: REFERENCE_WGSL,
        entryPoint: 'main',
        ownGroups: [
            {
                entries: [
                    { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                    { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                    { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
                ],
            },
        ],
    });
    const inputs = new Float32Array(points.length * 8);
    points.forEach((point, index) => {
        inputs.set(point.position, index * 8);
        inputs.set(point.normal, index * 8 + 4);
    });
    const input = device.createBuffer({ size: inputs.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(input, 0, inputs);
    const outputBytes = points.length * 64 * 16;
    const output = device.createBuffer({ size: outputBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const settings = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(settings, 0, new Uint32Array([REFERENCE_PATHS, 9781, 0, 0]));
    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    kernel.bind(pass);
    pass.setBindGroup(
        1,
        device.createBindGroup({
            layout: kernel.ownLayouts[0],
            entries: [
                { binding: 0, resource: { buffer: input } },
                { binding: 1, resource: { buffer: output } },
                { binding: 2, resource: { buffer: settings } },
            ],
        }),
    );
    pass.dispatchWorkgroups(points.length);
    pass.end();
    device.queue.submit([encoder.finish()]);
    const values = new Float32Array(await readBuffer(device, output, outputBytes));
    input.destroy();
    output.destroy();
    settings.destroy();
    return points.map((_, index) => {
        const total = [0, 0, 0];
        let paths = 0;
        for (let thread = 0; thread < 64; thread++) {
            const base = (index * 64 + thread) * 4;
            for (let channel = 0; channel < 3; channel++) total[channel] += values[base + channel];
            paths += values[base + 3];
        }
        // Cosine-weighted sampling: E = π × the mean incoming radiance.
        return meanOf(total.map((value) => (Math.PI * value) / paths));
    });
}

/**
 * The comparison at every {@link PROBE_STRIDE}-th probe of the latest screen
 * stage. `irradiance` is the interpolated image (rgba per pixel, row 0 at
 * the top); `screen` the stage's `FrameScreen` (for the probes' normals).
 */
export async function compareAtProbes(
    device: GPUDevice,
    scene: CompositeScene,
    probes: RadianceCacheProbes,
    screen: FrameScreen,
    irradiance: Float32Array,
    size: { readonly width: number; readonly height: number },
): Promise<ProbeComparison> {
    const field = scene.ddgi.field;
    if (!field) throw new Error('DDGI has no field');
    if (screen.normal.format !== 'rgba16float' || screen.normalEncoding !== 'snorm-float') throw new Error(`the diagnostic reads signed rgba16float normals, not ${screen.normal.format} ${screen.normalEncoding}`);
    const spawn = new Uint32Array(await readBuffer(device, probes.probeSpawn, PROBE_SPAWN_SEEDS_OFFSET + probes.layout.maxProbeSpawnCount * 4));
    const probeCount = spawn[0];
    const seeds = spawn.subarray(PROBE_SPAWN_SEEDS_OFFSET / 4, PROBE_SPAWN_SEEDS_OFFSET / 4 + probeCount);
    const rayCount = probeCount * RAYS_PER_PROBE;
    const rayBytes = new DataView(await readBuffer(device, probes.probeRays, PROBE_RAYS_HEADER_BYTES + rayCount * PROBE_RAY_BYTES));
    const resultBytes = new Uint8Array(await readBuffer(device, probes.rayScratch.buffer, RAY_SCRATCH_HEADER_BYTES + rayCount * RAY_RESULT_BYTES));
    const results = halves(resultBytes, RAY_SCRATCH_HEADER_BYTES);
    const normals = halves(await readTexture(device, screen.normal, 8, size.width, size.height));
    const shWidth = probes.layout.probeCount[0] * SH_BLOCK_SIZE;
    const probeSH = new Uint32Array((await readTexture(device, probes.probeSH, 8, shWidth, probes.layout.probeCount[1] * SH_BLOCK_SIZE)).buffer);
    const eye = scene.camera.position;

    const points: SurfacePoint[] = [];
    const raysE: number[] = [];
    const grazing: boolean[] = [];
    const shortHits: number[] = [];
    const backFaceHits: number[] = [];
    let shSum = 0;
    let pixelSum = 0;
    for (let probe = 0; probe < probeCount; probe += PROBE_STRIDE) {
        const [x, y] = unpackSeed(seeds[probe]);
        const pixel = y * size.width + x;
        const normal = normalize([normals[pixel * 4], normals[pixel * 4 + 1], normals[pixel * 4 + 2]]);
        const firstRay = PROBE_RAYS_HEADER_BYTES + probe * RAYS_PER_PROBE * PROBE_RAY_BYTES;
        const origin: Direction = [rayBytes.getFloat32(firstRay, true), rayBytes.getFloat32(firstRay + 4, true), rayBytes.getFloat32(firstRay + 8, true)];
        // The rays' own estimate of E: 2π / 64 × Σ L cos θ (one ray per equal-area cell).
        let estimate = 0;
        let short = 0;
        let backFaces = 0;
        for (let ray = 0; ray < RAYS_PER_PROBE; ray++) {
            const record = firstRay + ray * PROBE_RAY_BYTES;
            const direction = unpackHalf3([rayBytes.getUint32(record + 16, true), rayBytes.getUint32(record + 20, true)]);
            const hitT = rayBytes.getFloat32(record + 24, true);
            if (hitT >= 0 && hitT < SHORT_HIT_METRES) short++;
            if (hitT >= 0 && rayBytes.getUint32(record + 36, true) === 0) backFaces++;
            const result = (probe * RAYS_PER_PROBE + ray) * 4;
            const cosine = Math.max(0, direction[0] * normal[0] + direction[1] * normal[1] + direction[2] * normal[2]);
            estimate += CELL_SOLID_ANGLE * cosine * meanOf([results[result], results[result + 1], results[result + 2]]);
        }
        raysE.push(estimate);
        shortHits.push(short);
        backFaceHits.push(backFaces);
        shSum += shIrradiance(probeCoefficients(probeSH, shWidth, [Math.floor(x / PROBE_SIZE), Math.floor(y / PROBE_SIZE)]), normal);
        pixelSum += meanOf([irradiance[pixel * 4], irradiance[pixel * 4 + 1], irradiance[pixel * 4 + 2]]);
        const view = normalize([origin[0] - eye.x, origin[1] - eye.y, origin[2] - eye.z]);
        grazing.push(Math.abs(view[0] * normal[0] + view[1] * normal[1] + view[2] * normal[2]) < GRAZING_COSINE);
        points.push({ position: origin, normal, viewDirection: view });
    }
    const lifted = points.map((point) => ({ ...point, position: [0, 1, 2].map((axis) => point.position[axis] + LIFT_METRES * point.normal[axis]) as [number, number, number] }));
    const reference = await referenceIrradiance(device, scene, points);
    const liftedReference = await referenceIrradiance(device, scene, lifted);
    const ddgi = await (await FieldProbe.create(device, scene.platform)).sample(field, points);
    const sum = (values: readonly number[], keep: (index: number) => boolean = () => true): number => values.reduce((total, value, index) => (keep(index) ? total + value : total), 0);
    const referenceSum = sum(reference);
    const group = (isGrazing: boolean): ViewGroup => {
        const keep = (index: number): boolean => grazing[index] === isGrazing;
        const probes = grazing.filter((value) => value === isGrazing).length;
        const rays = Math.max(probes * RAYS_PER_PROBE, 1);
        return {
            probes,
            shortHitShare: sum(shortHits, keep) / rays,
            backFaceShare: sum(backFaceHits, keep) / rays,
            raysOverReference: sum(raysE, keep) / sum(reference, keep),
            raysOverLiftedReference: sum(raysE, keep) / sum(liftedReference, keep),
        };
    };
    return {
        sampledProbes: points.length,
        referenceE: referenceSum / Math.max(points.length, 1),
        raysOverReference: sum(raysE) / referenceSum,
        shOverReference: shSum / referenceSum,
        pixelOverReference: pixelSum / referenceSum,
        ddgiOverReference: sum(ddgi.map(meanOf)) / referenceSum,
        backFaceRayShare: sum(backFaceHits) / Math.max(points.length * RAYS_PER_PROBE, 1),
        byView: { grazing: group(true), facing: group(false) },
    };
}

function normalize(vector: readonly number[]): Direction {
    const length = Math.hypot(vector[0], vector[1], vector[2]) || 1;
    return [vector[0] / length, vector[1] / length, vector[2] / length];
}

/** A probe tile's nine coefficients (rgb averaged) from the SH texture's 3 × 3 block (register D26). */
function probeCoefficients(probeSH: Uint32Array, shWidth: number, tile: readonly [number, number]): number[] {
    const coefficients: number[] = [];
    for (let index = 0; index < SH_COEFFICIENTS; index++) {
        const texel = (tile[1] * SH_BLOCK_SIZE + Math.floor(index / SH_BLOCK_SIZE)) * shWidth + tile[0] * SH_BLOCK_SIZE + (index % SH_BLOCK_SIZE);
        const words = new Uint32Array([probeSH[texel * 2], probeSH[texel * 2 + 1]]);
        coefficients.push(meanOf(Array.from(halves(new Uint8Array(words.buffer)))));
    }
    return coefficients;
}
