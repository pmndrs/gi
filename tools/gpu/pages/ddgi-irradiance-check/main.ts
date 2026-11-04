/**
 * Diagnostic (T1.5a): DDGI's field against a path-traced irradiance at the
 * same surface points, on the point-light Cornell box.
 *
 * The reference is a small GPU path tracer over the platform's own tracer and
 * hit shading (cosine-sampled bounces, direct light by `giDirectIrradiance`
 * at every vertex), so it shares DDGI's scene and lights exactly. It answers
 * "is the field itself off, and where" without image-space effects.
 *
 * Reports, per surface (floor, ceiling, back wall, red and green walls):
 * mean field E / reference E, for the full field and for one bounce
 * (DDGI with bounceEnergy 0 against one-bounce paths).
 */

import { assert, runGpuTest } from '../../client.ts';
import { CORNELL_POINT_IN_AIR_VOLUME, createDdgiScene, FieldProbe, readBuffer, runFrames, type SurfacePoint } from '../ddgi-shared/harness.ts';
import type { GIPlatformImpl } from '../../../../packages/ddgi-core/src/platform/gi-platform.ts';

const FRAMES = 400;
const PATHS_PER_POINT = 16384;
const MAX_BOUNCES = 24;

const PATH_WGSL = /* wgsl */ `
struct RefPoint { position: vec4f, normal: vec4f }
@group(1) @binding(0) var<storage, read> refPoints: array<RefPoint>;
@group(1) @binding(1) var<storage, read_write> refResults: array<vec4f>;
@group(1) @binding(2) var<uniform> refSettings: vec4u; // paths, bounces, seed

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

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) group: vec3u, @builtin(local_invocation_index) thread: u32) {
    // One workgroup per point, 64 threads sharing its paths.
    let pointIndex = group.x;
    if (pointIndex >= arrayLength(&refPoints)) {
        return;
    }
    let point = refPoints[pointIndex];
    refState = refHash(pointIndex * 1000003u + thread * 7919u + refSettings.z);
    var sum = vec3f(0.0);
    let paths = refSettings.x / 64u;
    for (var path = 0u; path < paths; path++) {
        var origin = point.position.xyz + point.normal.xyz * 1.0e-3;
        var direction = refCosineDirection(point.normal.xyz);
        var throughput = vec3f(1.0);
        for (var bounce = 0u; bounce < refSettings.y; bounce++) {
            let ray = GiRay(origin, 0.0, direction, 1.0e27);
            let hit = giTraceClosest(ray);
            if (hit.t < 0.0) {
                sum += throughput * giSkyRadiance(direction);
                break;
            }
            if (!hit.frontFace) {
                break;
            }
            let surface = giSurfaceAt(ray, hit);
            sum += throughput * (surface.emission + surface.albedo * 0.318309886 * giDirectIrradiance(surface, 0xfu, vec2f(refRandom(), refRandom())));
            throughput *= surface.albedo;
            origin = surface.position + surface.normal * 1.0e-3;
            direction = refCosineDirection(surface.normal);
        }
    }
    refResults[pointIndex * 64u + thread] = vec4f(sum, f32(paths));
}
`;

async function pathTracedIrradiance(device: GPUDevice, platform: GIPlatformImpl, points: readonly SurfacePoint[], bounces: number): Promise<number[][]> {
    const kernel = await platform.kernels.computeKernel({
        label: 'ddgi irradiance reference',
        modules: [platform.shade.module],
        code: PATH_WGSL,
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
    device.queue.writeBuffer(settings, 0, new Uint32Array([PATHS_PER_POINT, bounces, 12345, 0]));
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
    return points.map((_, index) => {
        const total = [0, 0, 0];
        let paths = 0;
        for (let thread = 0; thread < 64; thread++) {
            const base = (index * 64 + thread) * 4;
            for (let channel = 0; channel < 3; channel++) total[channel] += values[base + channel];
            paths += values[base + 3];
        }
        // Cosine-weighted sampling: E = π × mean incoming radiance.
        return total.map((value) => (Math.PI * value) / paths);
    });
}

const W = 5.55;
const H = 5.488;
const D = 5.592;

function gridPoints(): { name: string; points: SurfacePoint[] }[] {
    const surfaces: { name: string; points: SurfacePoint[] }[] = [];
    const steps = 8;
    const make = (name: string, at: (a: number, b: number) => [number, number, number], normal: [number, number, number], skip?: (p: [number, number, number]) => boolean): void => {
        const points: SurfacePoint[] = [];
        for (let i = 0; i < steps; i++) {
            for (let j = 0; j < steps; j++) {
                const position = at((i + 0.5) / steps, (j + 0.5) / steps);
                if (skip?.(position)) continue;
                points.push({ position, normal, viewDirection: [-normal[0], -normal[1], -normal[2]] });
            }
        }
        surfaces.push({ name, points });
    };
    // Floor points away from the blocks' footprints (conservative circles).
    const nearBlock = (p: [number, number, number]): boolean => Math.hypot(p[0] - 1.865, p[2] - 2.08) < 1.3 || Math.hypot(p[0] - 3.7, p[2] - 3.9) < 1.3;
    make('floor', (a, b) => [a * W, 0, b * D], [0, 1, 0], nearBlock);
    make('ceiling', (a, b) => [a * W, H, b * D], [0, -1, 0]);
    make('back wall', (a, b) => [a * W, b * H, 0], [0, 0, 1]);
    make('red wall', (a, b) => [0, b * H, a * D], [1, 0, 0]);
    make('green wall', (a, b) => [W, b * H, a * D], [-1, 0, 0]);
    return surfaces;
}

/** Sum of all channels over each surface's points. */
function perSurfaceSums(surfaces: readonly { name: string; points: SurfacePoint[] }[], values: readonly number[][]): Map<string, number> {
    const sums = new Map<string, number>();
    let start = 0;
    for (const surface of surfaces) {
        let sum = 0;
        for (let index = start; index < start + surface.points.length; index++) sum += values[index][0] + values[index][1] + values[index][2];
        sums.set(surface.name, sum);
        start += surface.points.length;
    }
    return sums;
}

await runGpuTest(
    async ({ device, log }) => {
        const results: Record<string, unknown> = {};
        const surfaces = gridPoints();
        const allPoints = surfaces.flatMap((surface) => surface.points);
        const volumes = { manifest: undefined, 'in air': CORNELL_POINT_IN_AIR_VOLUME } as const;
        for (const [volumeName, volume] of Object.entries(volumes)) {
            for (const [label, bounceEnergy, bounces] of [
                ['all bounces', 1, MAX_BOUNCES],
                ['one bounce', 0, 1],
            ] as const) {
                const scene = await createDdgiScene(device, 'cornell-box-point', { bounceEnergy, ...(volume ? { volume } : {}) }, { ignoreLightRange: true });
                const field = scene.ddgi.field;
                assert(field, 'no field');
                await runFrames(device, scene.platform, FRAMES, field.grid.probeCount);
                const fieldSums = perSurfaceSums(surfaces, await (await FieldProbe.create(device, scene.platform)).sample(field, allPoints));
                const reference = await pathTracedIrradiance(device, scene.platform, allPoints, bounces);
                const referenceSums = perSurfaceSums(surfaces, reference);
                const ratios: Record<string, number> = {};
                for (const surface of surfaces) ratios[surface.name] = fieldSums.get(surface.name)! / referenceSums.get(surface.name)!;
                results[`${volumeName}, ${label}: field / path traced`] = ratios;
                log(`${volumeName}, ${label}: ${Object.entries(ratios).map(([name, ratio]) => `${name} ${ratio.toFixed(3)}`).join(', ')}`);

                // How fast the true irradiance changes off each surface (once per bounce count; it doesn't depend on DDGI).
                if (volumeName === 'manifest') {
                    const gradients: Record<string, Record<string, number>> = {};
                    for (const lift of [0.1, 0.25, 0.5]) {
                        const lifted = allPoints.map((point) => ({
                            ...point,
                            position: [0, 1, 2].map((axis) => point.position[axis] + point.normal[axis] * lift) as [number, number, number],
                        }));
                        const liftedSums = perSurfaceSums(surfaces, await pathTracedIrradiance(device, scene.platform, lifted, bounces));
                        for (const surface of surfaces) {
                            gradients[surface.name] ??= {};
                            gradients[surface.name][`${lift} m`] = liftedSums.get(surface.name)! / referenceSums.get(surface.name)!;
                        }
                    }
                    results[`${label}: path-traced E off the surface / on it`] = gradients;
                }
                scene.platform.dispose();
            }
        }
        return results;
    },
    { optionalFeatures: ['texture-formats-tier1', 'subgroups'] },
);
