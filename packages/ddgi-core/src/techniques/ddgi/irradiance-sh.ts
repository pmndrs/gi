/**
 * The field as L2 spherical harmonics per probe (T5.3,
 * docs/design/m5-coverage.md §3): what the `LightProbeGrid` export is built
 * from, and engine-free, so any engine's SH probe grid can take it.
 *
 * `exportIrradianceSH(technique)` projects every probe of one volume on the
 * GPU (`ddgi/export-sh`, wgsl/irradiance-sh.ts) and reads the result back
 * once. It is an explicit host call, like saving a bake, so its readback is
 * allowed (SPEC's no-readback rule is about the frame loop).
 *
 * **The convention is three's** (`SphericalHarmonics3`, `LightProbe`,
 * `LightProbeGrid`): 9 coefficients a probe in the order constant, y, z, x,
 * xy, yz, 3z² − 1, xz, x² − y², each RGB, holding *radiance* SH. three's
 * `getShIrradianceAt(n, sh)` evaluates Σ A_l c_lm Y_lm(n), the irradiance of
 * that radiance (Ramamoorthi and Hanrahan 2001; A_0 = π, A_1 = 2π/3,
 * A_2 = π/4). Our probes store irradiance E, so the export projects E and
 * divides each band by A_l; three's evaluation then gives back E, truncated
 * to bands 0-2. The unit test (packages/ddgi-three, `irradiance-sh.test.ts`)
 * checks it through three's own formula.
 *
 * **What is the field here:** the irradiance renderers sample. With light
 * groups or a sun layer on, that is the composed atlas, every layer times
 * its weight as of the last frame the technique encoded. A weight set since
 * then shows only after the next frame.
 *
 * **What is lost** (the fallback's known cost): bands above 2 (DDGI's tiles
 * hold more detail than L2), visibility (an SH grid has no distances, so its
 * light leaks through walls the Chebyshev test would stop), and relocation
 * (the coefficients are placed at the lattice points, not where relocated
 * probes sit).
 *
 * **Probes live GI doesn't read** (off inside geometry, off with no surface
 * nearby, not updated yet) have weight 0 in live sampling, but a grid
 * without visibility blends every probe. Their tiles hold old or no light,
 * so by default each gets the mean of its nearest active neighbours
 * ({@link fillInactiveProbes}), which is closest to what live GI shows near
 * them.
 */

import { PROBE_STATE } from './layout.ts';
import type { DDGITechnique } from './ddgi-technique.ts';
import {
    EXPORT_JOB_BYTES,
    EXPORT_RESULT_FLOATS,
    EXPORT_RESULT_STATE_FLOAT,
    EXPORT_SH_BINDING,
    EXPORT_SH_ENTRY_POINT,
    EXPORT_SH_WGSL,
    EXPORT_SH_WORKGROUP_SIZE,
} from './wgsl/irradiance-sh.ts';

/** Coefficients per probe and colour channel: bands 0, 1 and 2. */
export const L2_SH_COEFFICIENTS = 9;

/** Floats per probe in {@link IrradianceSHGrid.coefficients}: 9 coefficients × RGB. */
export const L2_SH_FLOATS_PER_PROBE = 27;

/**
 * Directions each probe's tile is read in. An 8 × 8 tile (the default
 * `irradianceTexels`) has 64 texels, so this is about 16 reads a texel: the
 * bilinear surface between texels is followed closely, and the projection of
 * an L2 function is good to a few parts in 10,000 (the unit test).
 */
export const EXPORT_SH_DIRECTIONS = 1024;

/** What {@link exportIrradianceSH} gives: one volume's probes as L2 radiance SH. */
export interface IrradianceSHGrid {
    /** Probes per axis. */
    readonly counts: readonly [number, number, number];
    /** World position of probe (0, 0, 0): for a camera-following volume, its window's first probe at export. */
    readonly origin: readonly [number, number, number];
    /** Metres between probes, per axis. */
    readonly spacing: readonly [number, number, number];
    /**
     * {@link L2_SH_FLOATS_PER_PROBE} floats per probe, probe index
     * x + y·nx + z·nx·ny, then coefficient, then r, g, b: radiance SH in
     * three's convention (the file comment).
     */
    readonly coefficients: Float32Array;
    /** Each probe's state at export (layout.ts, PROBE_STATE), in the same order. */
    readonly probeStates: Uint8Array;
    /** Probes that weren't active and were given their active neighbours' mean (0 with `fillInactiveProbes: false`). */
    readonly filledProbes: number;
    /** Irradiance layers composed into the exported field (light groups, the sun layer); 1 without them. */
    readonly composedLayers: number;
}

export interface ExportIrradianceSHOptions {
    /** Which volume, as an index into `technique.field.volumes` (highest priority first). Default 0. */
    readonly volume?: number;
    /** Give probes live GI doesn't read their active neighbours' mean (the file comment). Default true. */
    readonly fillInactiveProbes?: boolean;
}

/**
 * Directions spread evenly over the sphere (a Fibonacci spiral), each with
 * an equal share of 4π steradians: `count` × (x, y, z, weight). Even spacing
 * matters more than exactness here: the function is a bilinear surface over
 * the tile's texels, not a polynomial.
 */
export function shProjectionDirections(count: number = EXPORT_SH_DIRECTIONS): Float32Array<ArrayBuffer> {
    const directions = new Float32Array(count * 4);
    const goldenAngle = Math.PI * (3 - Math.sqrt(5));
    const weight = (4 * Math.PI) / count;
    for (let index = 0; index < count; index++) {
        // Equal-area bands in z, so equal weights are right.
        const z = 1 - (2 * index + 1) / count;
        const radius = Math.sqrt(Math.max(0, 1 - z * z));
        const azimuth = index * goldenAngle;
        directions[index * 4] = radius * Math.cos(azimuth);
        directions[index * 4 + 1] = radius * Math.sin(azimuth);
        directions[index * 4 + 2] = z;
        directions[index * 4 + 3] = weight;
    }
    return directions;
}

/** The 9 basis functions at a unit direction, in three's order and normalisation (the WGSL's `exportShBasis`). */
export function l2ShBasis(x: number, y: number, z: number): [number, number, number, number, number, number, number, number, number] {
    return [
        0.28209479177387814,
        0.4886025119029199 * y,
        0.4886025119029199 * z,
        0.4886025119029199 * x,
        1.0925484305920792 * x * y,
        1.0925484305920792 * y * z,
        0.31539156525252005 * (3 * z * z - 1),
        1.0925484305920792 * x * z,
        0.5462742152960396 * (x * x - y * y),
    ];
}

/** The clamped-cosine constant A_l of coefficient k's band (π, 2π/3, π/4). */
export function l2CosineLobe(coefficient: number): number {
    if (coefficient === 0) return Math.PI;
    if (coefficient < 4) return (2 * Math.PI) / 3;
    return Math.PI / 4;
}

/**
 * The CPU twin of `ddgi/export-sh` for one probe: projects an irradiance
 * function E(direction) over `directions` and returns three's radiance
 * coefficients (27 floats, coefficient then RGB). For tests, and for engines
 * that hold irradiance on the CPU.
 */
export function projectIrradianceToSH(
    irradianceAt: (x: number, y: number, z: number) => readonly [number, number, number],
    directions: Float32Array = shProjectionDirections(),
): Float64Array {
    const coefficients = new Float64Array(L2_SH_FLOATS_PER_PROBE);
    for (let index = 0; index < directions.length / 4; index++) {
        const x = directions[index * 4];
        const y = directions[index * 4 + 1];
        const z = directions[index * 4 + 2];
        const weight = directions[index * 4 + 3];
        const irradiance = irradianceAt(x, y, z);
        const basis = l2ShBasis(x, y, z);
        for (let coefficient = 0; coefficient < L2_SH_COEFFICIENTS; coefficient++) {
            for (let channel = 0; channel < 3; channel++) coefficients[coefficient * 3 + channel] += irradiance[channel] * basis[coefficient] * weight;
        }
    }
    for (let coefficient = 0; coefficient < L2_SH_COEFFICIENTS; coefficient++) {
        for (let channel = 0; channel < 3; channel++) coefficients[coefficient * 3 + channel] /= l2CosineLobe(coefficient);
    }
    return coefficients;
}

/**
 * Gives every probe whose state isn't active the mean coefficients of its
 * already-known neighbours (the 26 around it), ring by ring outwards from
 * the active probes, so a probe deep in a wall takes the nearest light
 * rather than black. Each ring reads only probes known before it started,
 * so the result doesn't depend on visiting order. Changes `coefficients` in
 * place; returns how many probes it filled. With no active probe at all,
 * nothing is filled.
 */
export function fillInactiveProbes(coefficients: Float32Array, probeStates: Uint8Array, counts: readonly [number, number, number]): number {
    const [nx, ny, nz] = counts;
    const probeCount = nx * ny * nz;
    const known = new Uint8Array(probeCount);
    for (let probe = 0; probe < probeCount; probe++) known[probe] = probeStates[probe] === PROBE_STATE.active ? 1 : 0;
    let filled = 0;
    const sum = new Float64Array(L2_SH_FLOATS_PER_PROBE);
    for (;;) {
        const ring: number[] = [];
        for (let z = 0; z < nz; z++) {
            for (let y = 0; y < ny; y++) {
                for (let x = 0; x < nx; x++) {
                    const probe = x + y * nx + z * nx * ny;
                    if (known[probe]) continue;
                    sum.fill(0);
                    let neighbours = 0;
                    for (let dz = -1; dz <= 1; dz++) {
                        for (let dy = -1; dy <= 1; dy++) {
                            for (let dx = -1; dx <= 1; dx++) {
                                const ox = x + dx;
                                const oy = y + dy;
                                const oz = z + dz;
                                if (ox < 0 || oy < 0 || oz < 0 || ox >= nx || oy >= ny || oz >= nz) continue;
                                const neighbour = ox + oy * nx + oz * nx * ny;
                                // 1 = known before this ring; 2 = filled in this ring (not read until the next).
                                if (known[neighbour] !== 1) continue;
                                for (let float = 0; float < L2_SH_FLOATS_PER_PROBE; float++) sum[float] += coefficients[neighbour * L2_SH_FLOATS_PER_PROBE + float];
                                neighbours++;
                            }
                        }
                    }
                    if (neighbours === 0) continue;
                    for (let float = 0; float < L2_SH_FLOATS_PER_PROBE; float++) coefficients[probe * L2_SH_FLOATS_PER_PROBE + float] = sum[float] / neighbours;
                    known[probe] = 2;
                    ring.push(probe);
                }
            }
        }
        if (ring.length === 0) return filled;
        for (const probe of ring) known[probe] = 1;
        filled += ring.length;
    }
}

// WebGPU flag values, written out so this file needs no DOM globals (the tests run in Node).
const BUFFER_USAGE = { mapRead: 0x1, copySrc: 0x4, copyDst: 0x8, uniform: 0x40, storage: 0x80 } as const;
const SHADER_STAGE_COMPUTE = 0x4;
const MAP_MODE_READ = 0x1;

interface ExportPipeline {
    readonly layout: GPUBindGroupLayout;
    readonly pipeline: GPUComputePipeline;
    readonly directions: GPUBuffer;
}

/** One pipeline per device, compiled at the first export: the export is rare, and repeated ones shouldn't recompile. */
const pipelines = new WeakMap<GPUDevice, Promise<ExportPipeline>>();

function exportPipeline(device: GPUDevice): Promise<ExportPipeline> {
    let pending = pipelines.get(device);
    if (!pending) {
        pending = (async (): Promise<ExportPipeline> => {
            const layout = device.createBindGroupLayout({
                label: 'ddgi export-sh',
                entries: [
                    { binding: EXPORT_SH_BINDING.irradianceAtlas, visibility: SHADER_STAGE_COMPUTE, texture: { sampleType: 'float', viewDimension: '2d-array' } },
                    { binding: EXPORT_SH_BINDING.irradianceSampler, visibility: SHADER_STAGE_COMPUTE, sampler: { type: 'filtering' } },
                    { binding: EXPORT_SH_BINDING.probeData, visibility: SHADER_STAGE_COMPUTE, texture: { sampleType: 'unfilterable-float', viewDimension: '2d-array' } },
                    { binding: EXPORT_SH_BINDING.volumeTable, visibility: SHADER_STAGE_COMPUTE, texture: { sampleType: 'unfilterable-float', viewDimension: '2d' } },
                    { binding: EXPORT_SH_BINDING.job, visibility: SHADER_STAGE_COMPUTE, buffer: { type: 'uniform' } },
                    { binding: EXPORT_SH_BINDING.directions, visibility: SHADER_STAGE_COMPUTE, buffer: { type: 'read-only-storage' } },
                    { binding: EXPORT_SH_BINDING.results, visibility: SHADER_STAGE_COMPUTE, buffer: { type: 'storage' } },
                ],
            });
            const module = device.createShaderModule({ label: 'ddgi export-sh', code: EXPORT_SH_WGSL });
            const pipeline = await device.createComputePipelineAsync({
                label: 'ddgi export-sh',
                layout: device.createPipelineLayout({ label: 'ddgi export-sh', bindGroupLayouts: [layout] }),
                compute: { module, entryPoint: EXPORT_SH_ENTRY_POINT },
            });
            const directionData = shProjectionDirections();
            const directions = device.createBuffer({ label: 'ddgi export-sh directions', size: directionData.byteLength, usage: BUFFER_USAGE.storage | BUFFER_USAGE.copyDst });
            device.queue.writeBuffer(directions, 0, directionData);
            return { layout, pipeline, directions };
        })();
        pipelines.set(device, pending);
    }
    return pending;
}

/**
 * One volume of a prepared DDGI technique as L2 radiance SH per probe (the
 * file comment): one GPU dispatch and one readback. Reads the field as the
 * frames submitted so far left it.
 */
export async function exportIrradianceSH(technique: DDGITechnique, options: ExportIrradianceSHOptions = {}): Promise<IrradianceSHGrid> {
    const field = technique.field;
    const device = technique.device;
    if (!field || !device) throw new Error('@pmndrs/gi: exportIrradianceSH needs a prepared DDGI technique (await gi.prepare() first)');
    const volumeIndex = options.volume ?? 0;
    const volume = field.volumes[volumeIndex];
    if (!volume) throw new Error(`@pmndrs/gi: exportIrradianceSH: there is no volume ${volumeIndex} (the field has ${field.volumes.length})`);

    // The volume's live sampling parameters: origin (its window's, for a camera-following volume), spacing, counts.
    const parameters = volume.sampleParameters;
    const origin: [number, number, number] = [parameters[0], parameters[1], parameters[2]];
    const spacing: [number, number, number] = [parameters[4], parameters[5], parameters[6]];
    const counts: [number, number, number] = [volume.grid.counts[0], volume.grid.counts[1], volume.grid.counts[2]];
    const probeCount = counts[0] * counts[1] * counts[2];
    const workgroups = Math.ceil(probeCount / EXPORT_SH_WORKGROUP_SIZE);
    if (workgroups > device.limits.maxComputeWorkgroupsPerDimension) {
        throw new Error(`@pmndrs/gi: exportIrradianceSH: ${probeCount} probes is more than one dispatch covers on this device`);
    }

    const { layout, pipeline, directions } = await exportPipeline(device);
    const resultBytes = probeCount * EXPORT_RESULT_FLOATS * 4;
    const job = device.createBuffer({ label: 'ddgi export-sh job', size: EXPORT_JOB_BYTES, usage: BUFFER_USAGE.uniform | BUFFER_USAGE.copyDst });
    const results = device.createBuffer({ label: 'ddgi export-sh results', size: resultBytes, usage: BUFFER_USAGE.storage | BUFFER_USAGE.copySrc });
    const readback = device.createBuffer({ label: 'ddgi export-sh readback', size: resultBytes, usage: BUFFER_USAGE.mapRead | BUFFER_USAGE.copyDst });
    try {
        device.queue.writeBuffer(job, 0, new Uint32Array([volumeIndex, probeCount, directions.size / 16, 0]));
        // The renderers' sampler: bilinear, clamped (the atlas's tiles carry their own borders).
        const sampler = device.createSampler({ label: 'ddgi export-sh', magFilter: 'linear', minFilter: 'linear', addressModeU: 'clamp-to-edge', addressModeV: 'clamp-to-edge' });
        const bindGroup = device.createBindGroup({
            label: 'ddgi export-sh',
            layout,
            entries: [
                { binding: EXPORT_SH_BINDING.irradianceAtlas, resource: field.irradiance.createView({ dimension: '2d-array' }) },
                { binding: EXPORT_SH_BINDING.irradianceSampler, resource: sampler },
                { binding: EXPORT_SH_BINDING.probeData, resource: field.probeData.createView({ dimension: '2d-array' }) },
                { binding: EXPORT_SH_BINDING.volumeTable, resource: field.volumeTable.createView({ dimension: '2d' }) },
                { binding: EXPORT_SH_BINDING.job, resource: { buffer: job } },
                { binding: EXPORT_SH_BINDING.directions, resource: { buffer: directions } },
                { binding: EXPORT_SH_BINDING.results, resource: { buffer: results } },
            ],
        });
        const encoder = device.createCommandEncoder({ label: 'ddgi export-sh' });
        const pass = encoder.beginComputePass({ label: 'ddgi export-sh' });
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, bindGroup);
        pass.dispatchWorkgroups(workgroups);
        pass.end();
        encoder.copyBufferToBuffer(results, 0, readback, 0, resultBytes);
        device.queue.submit([encoder.finish()]);
        await readback.mapAsync(MAP_MODE_READ);
        const floats = new Float32Array(readback.getMappedRange());
        const coefficients = new Float32Array(probeCount * L2_SH_FLOATS_PER_PROBE);
        const probeStates = new Uint8Array(probeCount);
        for (let probe = 0; probe < probeCount; probe++) {
            const source = probe * EXPORT_RESULT_FLOATS;
            coefficients.set(floats.subarray(source, source + L2_SH_FLOATS_PER_PROBE), probe * L2_SH_FLOATS_PER_PROBE);
            probeStates[probe] = Math.round(floats[source + EXPORT_RESULT_STATE_FLOAT]);
        }
        readback.unmap();
        const filledProbes = (options.fillInactiveProbes ?? true) ? fillInactiveProbes(coefficients, probeStates, counts) : 0;
        return { counts, origin, spacing, coefficients, probeStates, filledProbes, composedLayers: field.layers.layers.length };
    } finally {
        job.destroy();
        results.destroy();
        readback.destroy();
    }
}
