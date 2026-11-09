/**
 * `ddgiSampleIrradiance`: reading the DDGI field at a surface point
 * (docs/design/ddgi.md, "Sampling the field").
 *
 * One function, two users. The update kernel calls it for bounce light, and
 * the shading module (wgsl/shading.ts) wraps it for renderers as
 * `giIrradiance`, so there is exactly one sampling code path.
 *
 * The probe loop itself is `ddgiAccumulateField`. It returns the weighted
 * sum and the summed weight; `ddgiSampleIrradiance` divides one by the
 * other, and coverage (`giCoverage`) reads the summed weight. Each corner
 * probe's weight is `ddgiCornerWeight`, which the update's bounce lookup over
 * several light-group layers (wgsl/trace.ts) shares, so the weights are
 * worked out by one piece of code wherever the field is read.
 *
 * Being shared with renderers shapes it the way the shading-module contract
 * needs:
 *
 * - every resource is a parameter (no `@group/@binding` here), so three's
 *   `wgslFn` and a plain WebGPU renderer can each declare them their own way;
 * - it uses `textureSampleLevel(…, 0)` and `textureLoad` only, which work in
 *   compute and fragment stages alike (`textureSample` is fragment-only);
 * - one volume arrives as four `vec4f`s, the first four of the update
 *   kernel's uniforms (layout.ts, `UNIFORM_WORD`);
 * - several volumes (m3-moving.md §4.1) arrive as the volume table, a small
 *   `rgba32float` texture whose row v holds volume v's four `vec4f`s, rows in
 *   priority order (volumes.ts). `ddgiSampleVolumes` blends them. It is what
 *   renderers (`giIrradiance`) and the update's bounce light read, so one
 *   volume goes through the same code as several.
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
import { BLEND_COMPLETE_WEIGHT, COVERAGE_FADE_SPACINGS, COVERAGE_FULL_MARGIN_SPACINGS, COVERAGE_FULL_WEIGHT, MAX_BLENDED_VOLUMES } from '../volumes.ts';

export const DDGI_SAMPLE_WGSL = /* wgsl */ `
// ---- DDGI field sampling (techniques/ddgi) ---------------------------------------------

const DDGI_PI: f32 = 3.14159265358979;
const DDGI_STATE_ACTIVE: u32 = ${PROBE_STATE.active}u;
const DDGI_STATE_OFF_INSIDE_GEOMETRY: u32 = ${PROBE_STATE.offInsideGeometry}u;
const DDGI_STATE_OFF_NO_SURFACE_NEARBY: u32 = ${PROBE_STATE.offNoSurfaceNearby}u;
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

// One volume, unpacked from its four vec4f parameters (engines bind small
// uniforms as vec4s; the fields get names here).
//
// Every volume shares one set of atlases (m3-moving.md §4.1): volume v owns
// the array layers [layerBase, layerBase + ny), probe (x, z) owning tile
// (x, z) in them. A layer is as large as the largest volume's footprint,
// atlasTiles tiles across and down, so a smaller volume leaves part of its
// layers unused. With one volume, layerBase is 0 and atlasTiles is (nx, nz).
struct DdgiVolume {
    origin: vec3f,           // world position of probe (0, 0, 0)
    normalBias: f32,         // metres along the surface normal
    spacing: vec3f,          // metres between probes, per axis
    viewBias: f32,           // metres back towards the viewer
    counts: vec3i,           // probes per axis
    layerBase: i32,          // the atlas layer of the volume's probe row y = 0
    irradianceTexels: f32,   // interior texels per side of an irradiance tile
    distanceTexels: f32,     // interior texels per side of a distance tile
    atlasTiles: vec2f,       // tiles across (x) and down (z) one atlas layer, the same for every volume
}

fn ddgiVolume(volumeOrigin: vec4f, volumeSpacing: vec4f, volumeCounts: vec4f, volumeTexels: vec4f) -> DdgiVolume {
    return DdgiVolume(
        volumeOrigin.xyz, volumeOrigin.w,
        volumeSpacing.xyz, volumeSpacing.w,
        vec3i(volumeCounts.xyz), i32(volumeCounts.w),
        volumeTexels.x, volumeTexels.y, volumeTexels.zw,
    );
}

// The atlas layer that holds a volume's probe row probe.y.
fn ddgiAtlasLayer(volume: DdgiVolume, probe: vec3i) -> i32 {
    return volume.layerBase + probe.y;
}

// One probe's texel of the probe-data atlas: rgb the relocation offset in
// spacings, alpha the state (layout.ts, PROBE_STATE).
struct DdgiProbeData {
    offset: vec3f,
    state: u32,
}

fn ddgiReadProbeData(probeData: texture_2d_array<f32>, volume: DdgiVolume, probe: vec3i) -> DdgiProbeData {
    let texel = textureLoad(probeData, probe.xz, ddgiAtlasLayer(volume, probe), 0);
    return DdgiProbeData(texel.rgb, u32(round(texel.a)));
}

// Atlas coordinates. Tiles are interiorTexels + 2 border texels wide; probe
// (x, z) owns tile (x, z) of its row's layer (ddgiAtlasLayer). Interior texel
// centres map to the octahedral square's texel centres, so a bilinear lookup
// near an edge blends into the border, which holds the octahedral neighbour.
//
// The tile size and the atlas's reciprocal size are the same for every probe,
// so they are worked out once per sample (ddgiAtlasScale), not once per probe.
struct DdgiAtlasScale {
    tile: f32,                 // texels per tile side, border included
    inverseSize: vec2f,        // 1 / (atlas width, height) in texels
}

fn ddgiAtlasScale(volume: DdgiVolume, interiorTexels: f32) -> DdgiAtlasScale {
    let tile = interiorTexels + 2.0 * DDGI_TILE_BORDER;
    return DdgiAtlasScale(tile, 1.0 / (volume.atlasTiles * tile));
}

// Where a direction lands inside any probe's tile, in texels from the tile's corner.
fn ddgiTileTexel(direction: vec3f, interiorTexels: f32) -> vec2f {
    return (ddgiOctEncode(direction) * 0.5 + 0.5) * interiorTexels + DDGI_TILE_BORDER;
}

// That spot in one probe's tile, as normalized atlas coordinates.
fn ddgiTileUv(inTile: vec2f, probe: vec3i, atlas: DdgiAtlasScale) -> vec2f {
    return (vec2f(probe.xz) * atlas.tile + inTile) * atlas.inverseSize;
}

// The point the field is read at: off the surface along its normal and
// back towards the viewer, so the visibility test isn't decided by the
// surface the point lies on.
fn ddgiBiasedPoint(volume: DdgiVolume, position: vec3f, normal: vec3f, viewDirection: vec3f) -> vec3f {
    return position + normal * volume.normalBias - viewDirection * volume.viewBias;
}

// The cell whose 8 corner probes surround a point, and the point's place
// in it (0..1 per axis). Clamped to the grid: outside the volume the nearest
// cell answers, and coverage fades it (giCoverage).
struct DdgiCage {
    baseProbe: vec3i,
    alpha: vec3f,
}

fn ddgiCage(volume: DdgiVolume, samplePoint: vec3f) -> DdgiCage {
    let gridPosition = (samplePoint - volume.origin) / volume.spacing;
    let baseProbe = clamp(vec3i(floor(gridPosition)), vec3i(0), max(volume.counts - 2, vec3i(0)));
    let alpha = clamp(gridPosition - vec3f(baseProbe), vec3f(0.0), vec3f(1.0));
    return DdgiCage(baseProbe, alpha);
}

// Corner 0..7 of a cage as a 0/1 step per axis (bit 0: x, bit 1: y, bit 2: z).
fn ddgiCageStep(corner: u32) -> vec3i {
    return vec3i(vec3u(corner, corner >> 1u, corner >> 2u) & vec3u(1u));
}

fn ddgiTrilinearWeight(cage: DdgiCage, step: vec3i) -> f32 {
    let trilinear = mix(1.0 - cage.alpha, cage.alpha, vec3f(step));
    return trilinear.x * trilinear.y * trilinear.z;
}

// The field's weighted sum at a point, before normalising: sum of w * E, and
// sum of w. ddgiSampleIrradiance divides one by the other; coverage
// (giCoverage) reads the weight sum, which is 0 where no active probe counts.
struct DdgiFieldSum {
    weightedIrradiance: vec3f,
    weight: f32,
}

// One corner probe of a sample point's cage, and how much it counts: by
// trilinear position, by how much it faces the surface, and by whether it can
// see the point (Chebyshev test on the distance moments). 'counts' is false
// for a probe that isn't active, which adds nothing (no irradiance fetch).
struct DdgiCornerWeight {
    probe: vec3i,
    weight: f32,
    counts: bool,
}

fn ddgiCornerWeight(
    corner: u32,
    volume: DdgiVolume,
    cage: DdgiCage,
    samplePoint: vec3f,
    position: vec3f,
    normal: vec3f,
    distanceAtlas: texture_2d_array<f32>,
    distanceSampler: sampler,
    probeData: texture_2d_array<f32>,
    distanceScale: DdgiAtlasScale,
) -> DdgiCornerWeight {
    let step = ddgiCageStep(corner);
    let probe = min(cage.baseProbe + step, volume.counts - 1);
    let data = ddgiReadProbeData(probeData, volume, probe);
    if (data.state != DDGI_STATE_ACTIVE) {
        return DdgiCornerWeight(probe, 0.0, false);
    }
    // The probe's relocated position (offsets are stored in spacings).
    let probePosition = volume.origin + (vec3f(probe) + data.offset) * volume.spacing;
    let toProbe = probePosition - samplePoint;
    // One inverse square root gives both the distance and the direction
    // (capped, so a probe at the point itself keeps a defined direction).
    let squaredDistance = dot(toProbe, toProbe);
    let inverseProbeDistance = min(inverseSqrt(squaredDistance), 1.0 / DDGI_MIN_PROBE_DISTANCE);
    let probeDistance = max(squaredDistance * inverseProbeDistance, DDGI_MIN_PROBE_DISTANCE);
    let directionToProbe = toProbe * inverseProbeDistance;

    // Probes behind the surface count for little, never zero. Measured
    // from the surface point itself, not the biased one (Majercik et al.
    // 2019's "true direction to probe"): from the biased point, probes in
    // the surface's own plane look like they are in front of it, which
    // gave the probe layer farther from a wall more weight and read the
    // Cornell walls about 1% brighter.
    let surfaceToProbe = probePosition - position;
    let trueDirectionToProbe = surfaceToProbe * min(inverseSqrt(dot(surfaceToProbe, surfaceToProbe)), 1.0 / DDGI_MIN_PROBE_DISTANCE);
    let facing = (dot(trueDirectionToProbe, normal) + 1.0) * 0.5;
    var weight = facing * facing + DDGI_DIRECTION_WEIGHT_FLOOR;

    // Visibility: the probe's mean distance (and mean squared distance)
    // to the nearest surface in the point's direction. Farther than the
    // mean means probably occluded; Chebyshev's inequality bounds how
    // likely the point is visible anyway (variance shadow maps).
    let distanceUv = ddgiTileUv(ddgiTileTexel(-directionToProbe, volume.distanceTexels), probe, distanceScale);
    let moments = textureSampleLevel(distanceAtlas, distanceSampler, distanceUv, ddgiAtlasLayer(volume, probe), 0.0).xy;
    if (probeDistance > moments.x) {
        let variance = abs(moments.y - moments.x * moments.x);
        let excess = probeDistance - moments.x;
        let chebyshev = variance / (variance + excess * excess);
        weight *= max(chebyshev * chebyshev * chebyshev, DDGI_CHEBYSHEV_FLOOR);
    }

    if (weight < DDGI_CRUSH_THRESHOLD) {
        weight *= (weight * weight) * (1.0 / (DDGI_CRUSH_THRESHOLD * DDGI_CRUSH_THRESHOLD));
    }

    weight *= ddgiTrilinearWeight(cage, step);
    return DdgiCornerWeight(probe, weight, true);
}

// The 8 probes of one volume around the point, each weighted by
// ddgiCornerWeight. Only active probes count.
fn ddgiAccumulateVolume(
    volume: DdgiVolume,
    position: vec3f,
    normal: vec3f,
    viewDirection: vec3f,
    irradianceAtlas: texture_2d_array<f32>,
    irradianceSampler: sampler,
    distanceAtlas: texture_2d_array<f32>,
    distanceSampler: sampler,
    probeData: texture_2d_array<f32>,
) -> DdgiFieldSum {
    let samplePoint = ddgiBiasedPoint(volume, position, normal, viewDirection);
    let cage = ddgiCage(volume, samplePoint);

    // Worked out once, not per probe. Why this matters: sampling is
    // arithmetic-heavy as well as fetch-heavy, and T1.6 measured these
    // hoists plus inverseSqrt (ddgiCornerWeight) at 16% of a renderer's GI
    // cost at 1440p.
    let irradianceScale = ddgiAtlasScale(volume, volume.irradianceTexels);
    let distanceScale = ddgiAtlasScale(volume, volume.distanceTexels);
    // Irradiance is read in the normal's direction from every probe, so its
    // place inside a tile is the same for all 8.
    let normalInTile = ddgiTileTexel(normal, volume.irradianceTexels);

    var irradianceSum = vec3f(0.0);
    var weightSum = 0.0;
    for (var corner = 0u; corner < 8u; corner++) {
        let weighted = ddgiCornerWeight(corner, volume, cage, samplePoint, position, normal, distanceAtlas, distanceSampler, probeData, distanceScale);
        if (!weighted.counts) {
            continue;
        }
        let uv = ddgiTileUv(normalInTile, weighted.probe, irradianceScale);
        let irradiance = textureSampleLevel(irradianceAtlas, irradianceSampler, uv, ddgiAtlasLayer(volume, weighted.probe), 0.0).rgb;
        irradianceSum += weighted.weight * irradiance;
        weightSum += weighted.weight;
    }
    return DdgiFieldSum(irradianceSum, weightSum);
}

// ddgiAccumulateVolume with the volume as its four vec4f parameters.
fn ddgiAccumulateField(
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
) -> DdgiFieldSum {
    return ddgiAccumulateVolume(
        ddgiVolume(volumeOrigin, volumeSpacing, volumeCounts, volumeTexels),
        position, normal, viewDirection,
        irradianceAtlas, irradianceSampler, distanceAtlas, distanceSampler, probeData,
    );
}

// Irradiance E arriving at a surface point from one volume's field: the
// weighted mean of ddgiAccumulateField, or 0 where no probe has any weight.
// Renderers and the update read the field through ddgiSampleVolumes, which
// blends this over the volumes; test tools that look at one volume call it.
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
    let field = ddgiAccumulateField(
        position, normal, viewDirection,
        irradianceAtlas, irradianceSampler, distanceAtlas, distanceSampler, probeData,
        volumeOrigin, volumeSpacing, volumeCounts, volumeTexels,
    );
    return field.weightedIrradiance / max(field.weight, DDGI_MIN_WEIGHT_SUM);
}

// ---- Several volumes (m3-moving.md §4.1) ----
//
// Volume v's coverage at a point is its edge fade (1 out to half a spacing
// past its outermost probes, 0 a further half spacing out) times
// saturate(sum of its probe weights / DDGI_COVERAGE_FULL_WEIGHT), so a point
// none of its probes can see counts as not covered (DECISIONS 2026-11-05).
// Volumes are visited in priority order (the volume table's rows; the CPU
// sorts them, volumes.ts). Each adds its own normalised irradiance E_v with
// weight
//     w_v = coverage_v * (1 - W),
// W being the weight gathered so far, and the visit stops once W reaches
// DDGI_BLEND_COMPLETE or DDGI_MAX_BLENDED_VOLUMES volumes were read. Inside a
// higher-priority volume that is one volume's sampling, as before; only in
// its fade band are two read. The blend's irradiance is (sum of w_v E_v) / W
// and its coverage W, so a renderer's mix(environment, irradiance, coverage)
// gives sum of w_v E_v + (1 - W) environment: RTXGI's priority blend.

// A volume whose probes' weights sum to this counts as fully seeing a point,
// and coverage stays full this many spacings past the outermost probes, then
// fades over the next DDGI_COVERAGE_FADE (volumes.ts says why each).
const DDGI_COVERAGE_FULL_WEIGHT: f32 = ${COVERAGE_FULL_WEIGHT};
const DDGI_COVERAGE_FULL_MARGIN: f32 = ${COVERAGE_FULL_MARGIN_SPACINGS};
const DDGI_COVERAGE_FADE: f32 = ${COVERAGE_FADE_SPACINGS};
// The visit stops once the gathered weight reaches this...
const DDGI_BLEND_COMPLETE: f32 = ${BLEND_COMPLETE_WEIGHT};
// ...or after this many volumes, so no point costs more than two volumes' sampling.
const DDGI_MAX_BLENDED_VOLUMES: u32 = ${MAX_BLENDED_VOLUMES}u;

// 1 within half a spacing of a volume's outermost probes, falling to 0 over
// the next half spacing. Cheap, so the blend tests it before reading probes.
fn ddgiVolumeEdgeCoverage(volume: DdgiVolume, position: vec3f) -> f32 {
    let firstProbe = volume.origin;
    let lastProbe = volume.origin + vec3f(volume.counts - 1) * volume.spacing;
    // Distance outside the probe box per axis, in spacings (0 inside).
    let outside = max(max(firstProbe - position, position - lastProbe), vec3f(0.0)) / volume.spacing;
    let farthest = max(max(outside.x, outside.y), outside.z);
    return saturate(1.0 - (farthest - DDGI_COVERAGE_FULL_MARGIN) / DDGI_COVERAGE_FADE);
}

// How many volumes the table holds (one row each).
fn ddgiVolumeCount(volumeTable: texture_2d<f32>) -> u32 {
    return textureDimensions(volumeTable).y;
}

// Volume 'index' of the table: its row holds the four vec4f parameters in
// order (VOLUME_TABLE_TEXELS in volumes.ts).
fn ddgiTableVolume(volumeTable: texture_2d<f32>, index: u32) -> DdgiVolume {
    let row = i32(index);
    return ddgiVolume(
        textureLoad(volumeTable, vec2i(0, row), 0),
        textureLoad(volumeTable, vec2i(1, row), 0),
        textureLoad(volumeTable, vec2i(2, row), 0),
        textureLoad(volumeTable, vec2i(3, row), 0),
    );
}

// Whether the blend reads a volume. One whose edge coverage is 0 adds
// nothing and is skipped before any probe is read. The last (lowest
// priority) volume is read anyway when none before it was: a point outside
// every volume then gets that volume's nearest cell, as a single volume
// always gave it. So bounce light from hits outside the volumes keeps M2's
// value, and renderers fade such points out by their coverage of 0.
fn ddgiBlendReads(edgeCoverage: f32, index: u32, count: u32, volumesRead: u32) -> bool {
    return edgeCoverage > 0.0 || (index + 1u == count && volumesRead == 0u);
}

// A volume's share of the blend: its coverage, times what the volumes before it left.
fn ddgiBlendShare(edgeCoverage: f32, fieldWeight: f32, gatheredWeight: f32) -> f32 {
    return edgeCoverage * saturate(fieldWeight / DDGI_COVERAGE_FULL_WEIGHT) * (1.0 - gatheredWeight);
}

// Whether the blend has read enough volumes.
fn ddgiBlendDone(gatheredWeight: f32, volumesRead: u32) -> bool {
    return gatheredWeight >= DDGI_BLEND_COMPLETE || volumesRead >= DDGI_MAX_BLENDED_VOLUMES;
}

// The blend's irradiance: the volumes' weighted mean; the first volume read's
// own value when only one was read or nothing has weight. The first case keeps
// a single volume's answer exactly as ddgiSampleIrradiance gives it (no
// w * E / w rounding), so a one-volume field reads as it did before volumes.
fn ddgiBlendIrradiance(blended: vec3f, gatheredWeight: f32, firstIrradiance: vec3f, volumesRead: u32) -> vec3f {
    if (volumesRead <= 1u || gatheredWeight <= 0.0) {
        return firstIrradiance;
    }
    return blended / gatheredWeight;
}

// The blend of every volume at a point: irradiance E (not E / pi) and coverage, 0..1.
struct DdgiVolumesSample {
    irradiance: vec3f,
    coverage: f32,
}

// Every volume's field at a surface point, blended by priority (the rule
// above). What renderers read (giIrradiance, giCoverage), and what the
// update's bounce light reads, so light bounces between volumes.
fn ddgiSampleVolumes(
    position: vec3f,
    normal: vec3f,
    viewDirection: vec3f,
    irradianceAtlas: texture_2d_array<f32>,
    irradianceSampler: sampler,
    distanceAtlas: texture_2d_array<f32>,
    distanceSampler: sampler,
    probeData: texture_2d_array<f32>,
    volumeTable: texture_2d<f32>,
) -> DdgiVolumesSample {
    let count = ddgiVolumeCount(volumeTable);
    var blended = vec3f(0.0);
    var gatheredWeight = 0.0;
    var firstIrradiance = vec3f(0.0);
    var volumesRead = 0u;
    for (var index = 0u; index < count; index++) {
        let volume = ddgiTableVolume(volumeTable, index);
        let edgeCoverage = ddgiVolumeEdgeCoverage(volume, position);
        if (!ddgiBlendReads(edgeCoverage, index, count, volumesRead)) {
            continue;
        }
        let field = ddgiAccumulateVolume(volume, position, normal, viewDirection, irradianceAtlas, irradianceSampler, distanceAtlas, distanceSampler, probeData);
        let irradiance = field.weightedIrradiance / max(field.weight, DDGI_MIN_WEIGHT_SUM);
        if (volumesRead == 0u) {
            firstIrradiance = irradiance;
        }
        let share = ddgiBlendShare(edgeCoverage, field.weight, gatheredWeight);
        blended += share * irradiance;
        gatheredWeight += share;
        volumesRead++;
        if (ddgiBlendDone(gatheredWeight, volumesRead)) {
            break;
        }
    }
    return DdgiVolumesSample(ddgiBlendIrradiance(blended, gatheredWeight, firstIrradiance, volumesRead), gatheredWeight);
}
`;
