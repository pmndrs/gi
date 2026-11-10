/**
 * `ddgi/export-sh` (T5.3, docs/design/m5-coverage.md §3): projects each
 * probe's irradiance tile onto L2 spherical harmonics, for the
 * `LightProbeGrid` export (irradiance-sh.ts).
 *
 * One thread per probe. Each thread reads its probe's tile in a fixed set of
 * directions (a Fibonacci sphere with equal solid-angle weights, uploaded by
 * the CPU so the CPU twin in irradiance-sh.ts uses exactly the same points),
 * through the same octahedral and atlas functions renderers use
 * (`DDGI_SAMPLE_WGSL`), so the function projected is the one live GI reads
 * from that probe: bilinear between texels, borders included.
 *
 * What it writes per probe, {@link EXPORT_RESULT_FLOATS} floats:
 * - 27 floats, the 9 coefficients' r, g, b in three's order (constant, y, z,
 *   x, xy, yz, 3z² − 1, xz, x² − y²). They are *radiance* coefficients, as
 *   three's `LightProbe` and `LightProbeGrid` store them: the projection of
 *   the irradiance E divided by the band's clamped-cosine constant A_l
 *   (Ramamoorthi and Hanrahan 2001: A_0 = π, A_1 = 2π/3, A_2 = π/4). three's
 *   `getShIrradianceAt` multiplies by A_l again, so it gives back E;
 * - the probe's state (layout.ts, PROBE_STATE) as a float, so the CPU can fill
 *   in probes live GI doesn't read.
 *
 * The grid is walked in window probes, x fastest, then y, then z (the order of
 * the export's coefficients); a camera-following volume's data is read from
 * its wrapping slot (`ddgiProbeSlot`), so the export is the window as it is.
 *
 * Bindings (group 0, no platform module): the irradiance atlas renderers
 * sample (the composed one with light groups on), its filtering sampler, the
 * probe data, the volume table, the job's uniforms, the directions (storage,
 * read) and the results (storage, read-write).
 */

import { DDGI_SAMPLE_WGSL } from './sample.ts';

/** Threads per workgroup: one probe each. */
export const EXPORT_SH_WORKGROUP_SIZE = 64;

/** Floats the kernel writes per probe: 9 coefficients × RGB, then the state. */
export const EXPORT_RESULT_FLOATS = 28;

/** Where the state sits among a probe's results. */
export const EXPORT_RESULT_STATE_FLOAT = 27;

/** The job's uniform buffer: volume index, probe count, direction count, a spare word. */
export const EXPORT_JOB_BYTES = 16;

export const EXPORT_SH_ENTRY_POINT = 'ddgiExportProbeSH';

export const EXPORT_SH_BINDING = {
    irradianceAtlas: 0,
    irradianceSampler: 1,
    probeData: 2,
    volumeTable: 3,
    job: 4,
    directions: 5,
    results: 6,
} as const;

export const EXPORT_SH_WGSL = /* wgsl */ `
${DDGI_SAMPLE_WGSL}

// ---- ddgi/export-sh: L2 SH per probe (wgsl/irradiance-sh.ts) ----

struct ExportJob {
    volumeIndex: u32,      // the volume's row in the volume table (priority order)
    probeCount: u32,       // probes in its window
    directionCount: u32,   // entries of 'directions'
    spare: u32,
}

@group(0) @binding(${EXPORT_SH_BINDING.irradianceAtlas}) var irradianceAtlas: texture_2d_array<f32>;
@group(0) @binding(${EXPORT_SH_BINDING.irradianceSampler}) var irradianceSampler: sampler;
@group(0) @binding(${EXPORT_SH_BINDING.probeData}) var probeData: texture_2d_array<f32>;
@group(0) @binding(${EXPORT_SH_BINDING.volumeTable}) var volumeTable: texture_2d<f32>;
@group(0) @binding(${EXPORT_SH_BINDING.job}) var<uniform> job: ExportJob;
// xyz a unit direction, w its share of the sphere's 4π steradians.
@group(0) @binding(${EXPORT_SH_BINDING.directions}) var<storage, read> directions: array<vec4f>;
@group(0) @binding(${EXPORT_SH_BINDING.results}) var<storage, read_write> results: array<f32>;

const EXPORT_RESULT_FLOATS: u32 = ${EXPORT_RESULT_FLOATS}u;
const EXPORT_RESULT_STATE_FLOAT: u32 = ${EXPORT_RESULT_STATE_FLOAT}u;

// The real SH basis of bands 0-2 at a unit direction, in three's order and
// with three's normalisation (SphericalHarmonics3, LightProbeGrid's bake).
fn exportShBasis(d: vec3f) -> array<f32, 9> {
    return array<f32, 9>(
        0.28209479177387814,
        0.4886025119029199 * d.y,
        0.4886025119029199 * d.z,
        0.4886025119029199 * d.x,
        1.0925484305920792 * d.x * d.y,
        1.0925484305920792 * d.y * d.z,
        0.31539156525252005 * (3.0 * d.z * d.z - 1.0),
        1.0925484305920792 * d.x * d.z,
        0.5462742152960396 * (d.x * d.x - d.y * d.y),
    );
}

// 1 / A_l for coefficient k: projecting irradiance gives A_l times the
// radiance coefficient, and three stores radiance coefficients.
fn exportInverseCosineLobe(coefficient: u32) -> f32 {
    if (coefficient == 0u) {
        return 1.0 / DDGI_PI;
    }
    if (coefficient < 4u) {
        return 3.0 / (2.0 * DDGI_PI);
    }
    return 4.0 / DDGI_PI;
}

@compute @workgroup_size(${EXPORT_SH_WORKGROUP_SIZE})
fn ${EXPORT_SH_ENTRY_POINT}(@builtin(global_invocation_id) invocation: vec3u) {
    let probe = invocation.x;
    if (probe >= job.probeCount) {
        return;
    }
    let volume = ddgiTableVolume(volumeTable, job.volumeIndex);
    let counts = volume.counts;
    let index = i32(probe);
    let windowProbe = vec3i(index % counts.x, (index / counts.x) % counts.y, index / (counts.x * counts.y));
    let slot = ddgiProbeSlot(volume, windowProbe);
    let state = ddgiReadProbeData(probeData, volume, slot).state;
    let atlasScale = ddgiAtlasScale(volume, volume.irradianceTexels);
    let layer = ddgiAtlasLayer(volume, slot);

    var sums: array<vec3f, 9>;
    for (var sampleIndex = 0u; sampleIndex < job.directionCount; sampleIndex++) {
        let direction = directions[sampleIndex];
        // The same lookup ddgiAccumulateVolume makes for a surface facing this direction.
        let uv = ddgiTileUv(ddgiTileTexel(direction.xyz, volume.irradianceTexels), slot, atlasScale);
        let irradiance = textureSampleLevel(irradianceAtlas, irradianceSampler, uv, layer, 0.0).rgb;
        var basis = exportShBasis(direction.xyz);
        for (var coefficient = 0u; coefficient < 9u; coefficient++) {
            sums[coefficient] += irradiance * (basis[coefficient] * direction.w);
        }
    }

    let base = probe * EXPORT_RESULT_FLOATS;
    for (var coefficient = 0u; coefficient < 9u; coefficient++) {
        let radiance = sums[coefficient] * exportInverseCosineLobe(coefficient);
        results[base + coefficient * 3u] = radiance.r;
        results[base + coefficient * 3u + 1u] = radiance.g;
        results[base + coefficient * 3u + 2u] = radiance.b;
    }
    results[base + EXPORT_RESULT_STATE_FLOAT] = f32(state);
}
`;
