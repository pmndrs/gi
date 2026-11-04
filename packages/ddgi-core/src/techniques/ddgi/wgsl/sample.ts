/**
 * `ddgiSampleIrradiance`: reading the DDGI field at a surface point
 * (docs/design/ddgi.md, "Sampling the field").
 *
 * One function, two users: the update kernel calls it for bounce light, and
 * the shading module (T1.6) wraps it for renderers, so there is exactly one
 * sampling code path. That shapes it the way the shading-module contract
 * needs:
 *
 * - every resource is a parameter (no `@group/@binding` here), so three's
 *   `wgslFn` and a plain WebGPU renderer can each declare them their own way;
 * - it uses `textureSampleLevel(…, 0)` and `textureLoad` only, which work in
 *   compute and fragment stages alike (`textureSample` is fragment-only);
 * - the volume arrives as four `vec4f`s, the first four of the update
 *   kernel's uniforms (layout.ts, `UNIFORM_VEC4`).
 *
 * ```wgsl
 * fn ddgiSampleIrradiance(
 *     position: vec3f, normal: vec3f, viewDirection: vec3f,
 *     irradianceAtlas: texture_2d_array<f32>, irradianceSampler: sampler,
 *     distanceAtlas: texture_2d_array<f32>, distanceSampler: sampler,
 *     probeData: texture_2d_array<f32>,
 *     volumeOrigin: vec4f, volumeSpacing: vec4f, volumeCounts: vec4f, volumeTexels: vec4f,
 * ) -> vec3f   // irradiance E (not E / π)
 * ```
 *
 * `viewDirection` is the direction the point is seen *along* (from the eye or
 * the ray origin towards the point), so `-viewDirection` points back at the
 * viewer.
 */

import { PROBE_STATE, TILE_BORDER } from '../layout.ts';

export const DDGI_SAMPLE_WGSL = /* wgsl */ `
// ---- DDGI field sampling (techniques/ddgi) ---------------------------------------------

const DDGI_PI: f32 = 3.14159265358979;
const DDGI_STATE_ACTIVE: u32 = ${PROBE_STATE.active}u;
const DDGI_STATE_NOT_UPDATED_YET: u32 = ${PROBE_STATE.notUpdatedYet}u;
const DDGI_TILE_BORDER: f32 = ${TILE_BORDER}.0;
// A probe facing straight away from the surface still keeps this much
// direction weight (Majercik et al. 2019), so a point no probe faces is not
// left with no answer at all.
const DDGI_DIRECTION_WEIGHT_FLOOR: f32 = 0.2;
// Smallest visibility factor an occluded probe keeps, after cubing.
const DDGI_CHEBYSHEV_FLOOR: f32 = 0.05;
// Weights below this are crushed by (w / threshold)^2, so small weights fade
// fast instead of adding a faint wrong colour (Majercik et al. 2021).
const DDGI_CRUSH_THRESHOLD: f32 = 0.2;
// Keeps the direction to a probe defined when the sample point sits on it.
const DDGI_MIN_PROBE_DISTANCE: f32 = 1.0e-4;
// Keeps the final division defined when no probe has any weight (then the sum is 0 too).
const DDGI_MIN_WEIGHT_SUM: f32 = 1.0e-30;

fn ddgiSignNotZero(value: vec2f) -> vec2f {
    return select(vec2f(-1.0), vec2f(1.0), value >= vec2f(0.0));
}

// Octahedral map (Cigolle et al. 2014): unit direction to [-1, 1]^2. The
// upper hemisphere (z >= 0) is the inner diamond; the lower one is folded
// out into the corners.
fn ddgiOctEncode(direction: vec3f) -> vec2f {
    let projected = direction.xy / (abs(direction.x) + abs(direction.y) + abs(direction.z));
    if (direction.z >= 0.0) {
        return projected;
    }
    return (1.0 - abs(projected.yx)) * ddgiSignNotZero(projected);
}

fn ddgiOctDecode(coordinates: vec2f) -> vec3f {
    let z = 1.0 - abs(coordinates.x) - abs(coordinates.y);
    var xy = coordinates;
    if (z < 0.0) {
        xy = (1.0 - abs(coordinates.yx)) * ddgiSignNotZero(coordinates);
    }
    return normalize(vec3f(xy, z));
}

// The volume, unpacked from the four vec4f parameters a renderer passes
// (engines bind small uniforms as vec4s; the fields get names here).
struct DdgiVolume {
    origin: vec3f,           // world position of probe (0, 0, 0)
    normalBias: f32,         // metres along the surface normal
    spacing: vec3f,          // metres between probes, per axis
    viewBias: f32,           // metres back towards the viewer
    counts: vec3i,           // probes per axis
    irradianceTexels: f32,   // interior texels per side of an irradiance tile
    distanceTexels: f32,     // interior texels per side of a distance tile
}

fn ddgiVolume(volumeOrigin: vec4f, volumeSpacing: vec4f, volumeCounts: vec4f, volumeTexels: vec4f) -> DdgiVolume {
    return DdgiVolume(volumeOrigin.xyz, volumeOrigin.w, volumeSpacing.xyz, volumeSpacing.w, vec3i(volumeCounts.xyz), volumeTexels.x, volumeTexels.y);
}

// One probe's texel of the probe-data atlas: rgb the relocation offset in
// spacings, alpha the state (layout.ts, PROBE_STATE).
struct DdgiProbeData {
    offset: vec3f,
    state: u32,
}

fn ddgiReadProbeData(probeData: texture_2d_array<f32>, probe: vec3i) -> DdgiProbeData {
    let texel = textureLoad(probeData, probe.xz, probe.y, 0);
    return DdgiProbeData(texel.rgb, u32(round(texel.a)));
}

// Normalized atlas coordinates of a direction inside a probe's tile. Tiles
// are interiorTexels + 2 border texels wide; probe (x, z) owns tile (x, z) of
// layer y. Interior texel centres map to the octahedral square's texel
// centres, so a bilinear lookup near an edge blends into the border, which
// holds the octahedral neighbour.
fn ddgiAtlasUv(direction: vec3f, probe: vec3i, interiorTexels: f32, counts: vec3i) -> vec2f {
    let tile = interiorTexels + 2.0 * DDGI_TILE_BORDER;
    let inTile = (ddgiOctEncode(direction) * 0.5 + 0.5) * interiorTexels + DDGI_TILE_BORDER;
    let texel = vec2f(probe.xz) * tile + inTile;
    return texel / (vec2f(counts.xz) * tile);
}

// Irradiance E arriving at a surface point from the field: the 8 probes
// around the point, weighted by trilinear position, by how much they face
// the surface, and by whether they can see the point (Chebyshev test on the
// distance moments). Only active probes count.
fn ddgiSampleIrradiance(
    position: vec3f,
    normal: vec3f,
    viewDirection: vec3f,
    irradianceAtlas: texture_2d_array<f32>,
    irradianceSampler: sampler,
    distanceAtlas: texture_2d_array<f32>,
    distanceSampler: sampler,
    probeData: texture_2d_array<f32>,
    // The volume as four vec4f uniforms; ddgiVolume() names their fields.
    volumeOrigin: vec4f,
    volumeSpacing: vec4f,
    volumeCounts: vec4f,
    volumeTexels: vec4f,
) -> vec3f {
    let volume = ddgiVolume(volumeOrigin, volumeSpacing, volumeCounts, volumeTexels);
    // Bias the point off the surface and back towards the viewer, so the
    // visibility test isn't decided by the surface the point lies on.
    let samplePoint = position + normal * volume.normalBias - viewDirection * volume.viewBias;
    let spacing = volume.spacing;
    let counts = volume.counts;
    // The cell whose 8 corner probes surround the point, clamped to the grid
    // (outside the volume the nearest cell answers; coverage fades it, T1.6).
    let gridPosition = (samplePoint - volume.origin) / spacing;
    let baseProbe = clamp(vec3i(floor(gridPosition)), vec3i(0), max(counts - 2, vec3i(0)));
    let alpha = clamp(gridPosition - vec3f(baseProbe), vec3f(0.0), vec3f(1.0));

    var irradianceSum = vec3f(0.0);
    var weightSum = 0.0;
    for (var corner = 0u; corner < 8u; corner++) {
        let step = vec3i(vec3u(corner, corner >> 1u, corner >> 2u) & vec3u(1u));
        let probe = min(baseProbe + step, counts - 1);
        let data = ddgiReadProbeData(probeData, probe);
        if (data.state != DDGI_STATE_ACTIVE) {
            continue;
        }
        // The probe's relocated position (offsets are stored in spacings).
        let probePosition = volume.origin + (vec3f(probe) + data.offset) * spacing;
        let toProbe = probePosition - samplePoint;
        let probeDistance = max(length(toProbe), DDGI_MIN_PROBE_DISTANCE);
        let directionToProbe = toProbe / probeDistance;

        // Probes behind the surface count for little, never zero. Measured
        // from the surface point itself, not the biased one (Majercik et al.
        // 2019's "true direction to probe"): from the biased point, probes in
        // the surface's own plane look like they are in front of it, which
        // gave the probe layer farther from a wall more weight and read the
        // Cornell walls about 1% brighter.
        let surfaceToProbe = probePosition - position;
        let trueDirectionToProbe = surfaceToProbe / max(length(surfaceToProbe), DDGI_MIN_PROBE_DISTANCE);
        let facing = (dot(trueDirectionToProbe, normal) + 1.0) * 0.5;
        var weight = facing * facing + DDGI_DIRECTION_WEIGHT_FLOOR;

        // Visibility: the probe's mean distance (and mean squared distance)
        // to the nearest surface in the point's direction. Farther than the
        // mean means probably occluded; Chebyshev's inequality bounds how
        // likely the point is visible anyway (variance shadow maps).
        let moments = textureSampleLevel(distanceAtlas, distanceSampler, ddgiAtlasUv(-directionToProbe, probe, volume.distanceTexels, counts), probe.y, 0.0).xy;
        if (probeDistance > moments.x) {
            let variance = abs(moments.y - moments.x * moments.x);
            let excess = probeDistance - moments.x;
            let chebyshev = variance / (variance + excess * excess);
            weight *= max(chebyshev * chebyshev * chebyshev, DDGI_CHEBYSHEV_FLOOR);
        }

        if (weight < DDGI_CRUSH_THRESHOLD) {
            weight *= (weight * weight) / (DDGI_CRUSH_THRESHOLD * DDGI_CRUSH_THRESHOLD);
        }

        let trilinear = mix(1.0 - alpha, alpha, vec3f(step));
        weight *= trilinear.x * trilinear.y * trilinear.z;

        let irradiance = textureSampleLevel(irradianceAtlas, irradianceSampler, ddgiAtlasUv(normal, probe, volume.irradianceTexels, counts), probe.y, 0.0).rgb;
        irradianceSum += weight * irradiance;
        weightSum += weight;
    }
    return irradianceSum / max(weightSum, DDGI_MIN_WEIGHT_SUM);
}
`;
