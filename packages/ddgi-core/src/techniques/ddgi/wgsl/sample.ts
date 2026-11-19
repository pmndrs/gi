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
 *   `rgba32float` texture whose row v holds volume v's four `vec4f`s, its
 *   window and its valid extent, rows in priority order (volumes.ts).
 *   `ddgiSampleVolumes` blends them. It is what renderers (`giIrradiance`)
 *   and the update's bounce light read, so one volume goes through the same
 *   code as several;
 * - a camera-following volume (m3-moving.md §4.2, scroll.ts) keeps each
 *   probe's data in a wrapping slot: the cage is found in window probes, and
 *   each corner's data is read from its slot (`ddgiProbeSlot`). For a fixed
 *   volume the window is 0 and a slot is its window probe: the same code,
 *   not a second path. The four-`vec4f` entry points (`ddgiSampleIrradiance`,
 *   `ddgiAccumulateField`) describe a fixed volume; read a camera-following
 *   one through the table. They also have no live `samplingWeightFloors`
 *   (DV-06): they use its default, where the table has the setting.
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
import { BLEND_COMPLETE_WEIGHT, COVERAGE_FADE_SPACINGS, COVERAGE_FULL_MARGIN_SPACINGS, COVERAGE_FULL_WEIGHT, MAX_BLENDED_VOLUMES, VOLUME_TABLE_TEXEL } from '../volumes.ts';
import { DEFAULT_DDGI_SETTINGS } from '../settings.ts';

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
// RTXGI's floors on the sampling weights (DV-06, a volume's weightFloors;
// RTXGI's Irradiance.hlsl, DDGIGetVolumeIrradiance). Each axis of a corner
// probe's trilinear weight is at least this, so a point on a probe plane
// still gives the far plane's probes a tiny weight: when the near plane's
// probes are all off, the far ones answer instead of nobody.
const DDGI_TRILINEAR_AXIS_FLOOR: f32 = 0.001;
// ...and a probe's weight before the crush is at least this ("avoid a weight
// of zero"). With the direction and Chebyshev floors above the weight is
// already at least 0.2 x 0.05 = 0.01, so this only matters to a NaN.
const DDGI_WEIGHT_FLOOR: f32 = 1.0e-6;
// The four-vec4f entry points (ddgiVolume) have no volume table to say, so they use the setting's default.
const DDGI_DEFAULT_WEIGHT_FLOORS: bool = ${DEFAULT_DDGI_SETTINGS.samplingWeightFloors};
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
// the array layers [layerBase, layerBase + ny), slot (x, z) owning tile
// (x, z) in them. A layer is as large as the largest volume's footprint,
// atlasTiles tiles across and down, so a smaller volume leaves part of its
// layers unused. With one volume, layerBase is 0 and atlasTiles is (nx, nz).
//
// A camera-following volume (m3-moving.md §4.2, scroll.ts) is a window of
// probes over a lattice fixed in world space. origin is its window's first
// probe, so positions in window probes work as for a fixed volume; window
// is where that probe is on the lattice, and slotShift the slot it lives in
// (window mod counts), so window probe w's data is in slot (w + slotShift)
// mod counts. validMin and validMax are where its coverage fades, in lattice
// units, and fade how wide its fade band is on each axis, in spacings. A
// fixed volume has window 0, slotShift 0, the valid extent of all its probes
// and a fade of half a spacing.
struct DdgiVolume {
    origin: vec3f,           // world position of window probe (0, 0, 0)
    normalBias: f32,         // metres along the surface normal
    spacing: vec3f,          // metres between probes, per axis
    viewBias: f32,           // metres back towards the viewer
    counts: vec3i,           // probes per axis
    layerBase: i32,          // the atlas layer of the volume's slot row y = 0
    irradianceTexels: f32,   // interior texels per side of an irradiance tile
    distanceTexels: f32,     // interior texels per side of a distance tile
    atlasTiles: vec2f,       // tiles across (x) and down (z) one atlas layer, the same for every volume
    window: vec3i,           // lattice coordinates of window probe (0, 0, 0)
    slotShift: vec3i,        // the slot window probe (0, 0, 0) lives in: window mod counts, per axis
    validMin: vec3f,         // the valid extent's low sides, lattice units
    validMax: vec3f,         // its high sides
    fade: vec3f,             // the fade band's width per axis, spacings
    weightFloors: bool,      // RTXGI's floors on the sampling weights (DDGI_TRILINEAR_AXIS_FLOOR, DDGI_WEIGHT_FLOOR)
}

// value mod divisor in 0..divisor - 1, also for negative values (WGSL's % keeps the dividend's sign).
fn ddgiPositiveModulo(value: vec3i, divisor: vec3i) -> vec3i {
    return ((value % divisor) + divisor) % divisor;
}

// A fixed volume from its four vec4f parameters: window 0, every probe valid,
// the usual fade, and the sampling weights as the settings' default
// (samplingWeightFloors); the volume table carries the live setting.
fn ddgiVolume(volumeOrigin: vec4f, volumeSpacing: vec4f, volumeCounts: vec4f, volumeTexels: vec4f) -> DdgiVolume {
    let counts = vec3i(volumeCounts.xyz);
    return DdgiVolume(
        volumeOrigin.xyz, volumeOrigin.w,
        volumeSpacing.xyz, volumeSpacing.w,
        counts, i32(volumeCounts.w),
        volumeTexels.x, volumeTexels.y, volumeTexels.zw,
        vec3i(0), vec3i(0), vec3f(0.0), vec3f(counts - 1), vec3f(DDGI_COVERAGE_FADE),
        DDGI_DEFAULT_WEIGHT_FLOORS,
    );
}

// The same volume with its window at lattice 'window', the given valid extent and fade.
fn ddgiVolumeInWindow(volume: DdgiVolume, window: vec3i, validMin: vec3f, validMax: vec3f, fade: vec3f) -> DdgiVolume {
    var placed = volume;
    placed.window = window;
    placed.slotShift = ddgiPositiveModulo(window, volume.counts);
    placed.validMin = validMin;
    placed.validMax = validMax;
    placed.fade = fade;
    return placed;
}

// The slot that holds window probe 'windowProbe' (both 0..counts - 1 per
// axis). Both terms are under counts, so one conditional subtraction is the
// modulo: cheaper than an integer %, and this runs once per probe sample.
fn ddgiProbeSlot(volume: DdgiVolume, windowProbe: vec3i) -> vec3i {
    let slot = windowProbe + volume.slotShift;
    return select(slot, slot - volume.counts, slot >= volume.counts);
}

// The window probe a slot holds (the inverse of ddgiProbeSlot).
fn ddgiWindowProbe(volume: DdgiVolume, slot: vec3i) -> vec3i {
    let windowProbe = slot - volume.slotShift;
    return select(windowProbe, windowProbe + volume.counts, windowProbe < vec3i(0));
}

// The atlas layer that holds a volume's slot row slot.y.
fn ddgiAtlasLayer(volume: DdgiVolume, slot: vec3i) -> i32 {
    return volume.layerBase + slot.y;
}

// One probe's texel of the probe-data atlas: rgb the relocation offset in
// spacings, alpha the state (layout.ts, PROBE_STATE).
struct DdgiProbeData {
    offset: vec3f,
    state: u32,
}

// The probe data of the probe in 'slot' (its window probe for a fixed volume).
fn ddgiReadProbeData(probeData: texture_2d_array<f32>, volume: DdgiVolume, slot: vec3i) -> DdgiProbeData {
    let texel = textureLoad(probeData, slot.xz, ddgiAtlasLayer(volume, slot), 0);
    return DdgiProbeData(texel.rgb, u32(round(texel.a)));
}

// Atlas coordinates. Tiles are interiorTexels + 2 border texels wide; slot
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

// That spot in the tile of the probe in 'slot', as normalized atlas coordinates.
fn ddgiTileUv(inTile: vec2f, slot: vec3i, atlas: DdgiAtlasScale) -> vec2f {
    return (vec2f(slot.xz) * atlas.tile + inTile) * atlas.inverseSize;
}

// The point the field is read at: off the surface along its normal and
// back towards the viewer, so the visibility test isn't decided by the
// surface the point lies on.
fn ddgiBiasedPoint(volume: DdgiVolume, position: vec3f, normal: vec3f, viewDirection: vec3f) -> vec3f {
    return position + normal * volume.normalBias - viewDirection * volume.viewBias;
}

// The cell whose 8 corner probes surround a point, in window probes, and the
// point's place in it (0..1 per axis). Clamped to the window: outside the
// volume the nearest cell answers, and coverage fades it (giCoverage).
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

// The trilinear weight a corner probe's sampling weight uses: with the
// volume's weightFloors, each axis is at least DDGI_TRILINEAR_AXIS_FLOOR, as
// RTXGI's max(0.001, lerp(1 - alpha, alpha, offset)).
fn ddgiSamplingTrilinearWeight(volume: DdgiVolume, cage: DdgiCage, step: vec3i) -> f32 {
    var trilinear = mix(1.0 - cage.alpha, cage.alpha, vec3f(step));
    if (volume.weightFloors) {
        trilinear = max(trilinear, vec3f(DDGI_TRILINEAR_AXIS_FLOOR));
    }
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
// for a probe that isn't active, which adds nothing (no irradiance fetch):
// so a probe still warming (state 3, a camera-following volume's entering
// planes among them) is never used. 'slot' is where its tiles are.
struct DdgiCornerWeight {
    slot: vec3i,
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
    let windowProbe = min(cage.baseProbe + step, volume.counts - 1);
    let slot = ddgiProbeSlot(volume, windowProbe);
    let data = ddgiReadProbeData(probeData, volume, slot);
    if (data.state != DDGI_STATE_ACTIVE) {
        return DdgiCornerWeight(slot, 0.0, false);
    }
    // The probe's relocated position (offsets are stored in spacings).
    let probePosition = volume.origin + (vec3f(windowProbe) + data.offset) * volume.spacing;
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
    let distanceUv = ddgiTileUv(ddgiTileTexel(-directionToProbe, volume.distanceTexels), slot, distanceScale);
    let moments = textureSampleLevel(distanceAtlas, distanceSampler, distanceUv, ddgiAtlasLayer(volume, slot), 0.0).xy;
    if (probeDistance > moments.x) {
        let variance = abs(moments.y - moments.x * moments.x);
        let excess = probeDistance - moments.x;
        let chebyshev = variance / (variance + excess * excess);
        weight *= max(chebyshev * chebyshev * chebyshev, DDGI_CHEBYSHEV_FLOOR);
    }

    if (volume.weightFloors) {
        weight = max(weight, DDGI_WEIGHT_FLOOR);
    }
    if (weight < DDGI_CRUSH_THRESHOLD) {
        weight *= (weight * weight) * (1.0 / (DDGI_CRUSH_THRESHOLD * DDGI_CRUSH_THRESHOLD));
    }

    weight *= ddgiSamplingTrilinearWeight(volume, cage, step);
    return DdgiCornerWeight(slot, weight, true);
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
        let uv = ddgiTileUv(normalInTile, weighted.slot, irradianceScale);
        let irradiance = textureSampleLevel(irradianceAtlas, irradianceSampler, uv, ddgiAtlasLayer(volume, weighted.slot), 0.0).rgb;
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
// past its valid extent, 0 a further half spacing out; for a fixed volume
// the valid extent is its outermost probes) times
// saturate(sum of its probe weights / DDGI_COVERAGE_FULL_WEIGHT), so a point
// none of its probes can see counts as not covered (DECISIONS 2026-11-05).
// Volumes are visited in priority order (the volume table's rows; the CPU
// sorts them, volumes.ts). Each adds its own normalised irradiance E_v with
// weight
//     w_v = coverage_v * (1 - W),
// W being the weight gathered so far, and the visit stops once W reaches
// DDGI_BLEND_COMPLETE or DDGI_MAX_BLENDED_VOLUMES volumes have added weight
// (a volume that covers the point but whose probes all have weight 0 there
// adds nothing and doesn't count, so with three nested volumes the third
// still fills in behind it). Inside a
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
// ...or once this many volumes have added weight: in a fade band two volumes' sampling, not more.
const DDGI_MAX_BLENDED_VOLUMES: u32 = ${MAX_BLENDED_VOLUMES}u;

// How far a point lies outside a volume's valid extent on each axis, in
// metres (0 inside). The extent is in lattice units, and the window's first
// probe (lattice 'window') sits at origin.
fn ddgiVolumeOutside(volume: DdgiVolume, position: vec3f) -> vec3f {
    let firstProbe = volume.origin + (volume.validMin - vec3f(volume.window)) * volume.spacing;
    let lastProbe = volume.origin + (volume.validMax - vec3f(volume.window)) * volume.spacing;
    return max(max(firstProbe - position, position - lastProbe), vec3f(0.0));
}

// 1 within half a spacing of a volume's valid extent, falling to 0 over the
// next 'fade' spacings on each axis (half a spacing for a fixed volume), the
// lowest over the axes. Cheap, so the blend tests it before reading probes.
fn ddgiVolumeEdgeCoverage(volume: DdgiVolume, position: vec3f) -> f32 {
    // Distance outside per axis, in spacings.
    let outside = ddgiVolumeOutside(volume, position) / volume.spacing;
    let perAxis = saturate(1.0 - (outside - DDGI_COVERAGE_FULL_MARGIN) / volume.fade);
    return min(min(perAxis.x, perAxis.y), perAxis.z);
}

// How far a point is from a volume's valid extent, in metres (the largest
// axis): which volume is nearest to a point outside them all.
fn ddgiVolumeDistance(volume: DdgiVolume, position: vec3f) -> f32 {
    let outside = ddgiVolumeOutside(volume, position);
    return max(max(outside.x, outside.y), outside.z);
}

// How many volumes the table holds (one row each).
fn ddgiVolumeCount(volumeTable: texture_2d<f32>) -> u32 {
    return textureDimensions(volumeTable).y;
}

// The texels of a table row (VOLUME_TABLE_TEXEL in volumes.ts).
const DDGI_TABLE_WINDOW_TEXEL: i32 = ${VOLUME_TABLE_TEXEL.window};
const DDGI_TABLE_FADE_TEXEL: i32 = ${VOLUME_TABLE_TEXEL.fade};
const DDGI_TABLE_VALID_MIN_TEXEL: i32 = ${VOLUME_TABLE_TEXEL.validMin};
const DDGI_TABLE_VALID_MAX_TEXEL: i32 = ${VOLUME_TABLE_TEXEL.validMax};

// Volume 'index' of the table: its row holds the four vec4f parameters, then
// its window and its valid extent.
fn ddgiTableVolume(volumeTable: texture_2d<f32>, index: u32) -> DdgiVolume {
    let row = i32(index);
    let volume = ddgiVolume(
        textureLoad(volumeTable, vec2i(${VOLUME_TABLE_TEXEL.sampling}, row), 0),
        textureLoad(volumeTable, vec2i(${VOLUME_TABLE_TEXEL.sampling + 1}, row), 0),
        textureLoad(volumeTable, vec2i(${VOLUME_TABLE_TEXEL.sampling + 2}, row), 0),
        textureLoad(volumeTable, vec2i(${VOLUME_TABLE_TEXEL.sampling + 3}, row), 0),
    );
    // Whole numbers stored as floats; round, so no conversion can land one below.
    let window = vec3i(round(textureLoad(volumeTable, vec2i(DDGI_TABLE_WINDOW_TEXEL, row), 0).xyz));
    let validMin = textureLoad(volumeTable, vec2i(DDGI_TABLE_VALID_MIN_TEXEL, row), 0).xyz;
    let validMax = textureLoad(volumeTable, vec2i(DDGI_TABLE_VALID_MAX_TEXEL, row), 0).xyz;
    let fadeAndFloors = textureLoad(volumeTable, vec2i(DDGI_TABLE_FADE_TEXEL, row), 0);
    var placed = ddgiVolumeInWindow(volume, window, validMin, validMax, fadeAndFloors.xyz);
    placed.weightFloors = fadeAndFloors.w > 0.5;
    return placed;
}

// Whether the blend reads a volume: one whose edge coverage is 0 adds
// nothing and is skipped before any probe is read.
fn ddgiBlendReads(edgeCoverage: f32) -> bool {
    return edgeCoverage > 0.0;
}

// A point outside every volume (no volume read) still gets an answer, though
// renderers fade it out by its coverage of 0: the nearest volume's nearest
// cell, as a single volume always gave it, so bounce light from hits outside
// the volumes keeps M2's value. The nearest one, not the lowest priority, so
// volumes side by side each answer on their own side (the camera-following
// volume next to a fixed one); on a tie the later (lower priority) one, as
// the lowest priority answered before.
struct DdgiNearestVolume {
    index: u32,
    distance: f32,
}

fn ddgiNearerVolume(nearest: DdgiNearestVolume, index: u32, volume: DdgiVolume, position: vec3f) -> DdgiNearestVolume {
    let distance = ddgiVolumeDistance(volume, position);
    if (distance <= nearest.distance) {
        return DdgiNearestVolume(index, distance);
    }
    return nearest;
}

// Farther than any point: the start of the search for the nearest volume.
const DDGI_FARTHEST_DISTANCE: f32 = 3.0e38;

// A volume's share of the blend: its coverage, times what the volumes before it left.
fn ddgiBlendShare(edgeCoverage: f32, fieldWeight: f32, gatheredWeight: f32) -> f32 {
    return edgeCoverage * saturate(fieldWeight / DDGI_COVERAGE_FULL_WEIGHT) * (1.0 - gatheredWeight);
}

// Whether the blend has gathered enough: the weight, or the volumes that added some.
fn ddgiBlendDone(gatheredWeight: f32, contributingVolumes: u32) -> bool {
    return gatheredWeight >= DDGI_BLEND_COMPLETE || contributingVolumes >= DDGI_MAX_BLENDED_VOLUMES;
}

// The blend's irradiance: the volumes' weighted mean when two or more added
// weight; the one volume's own value when only one did; the first volume
// read's value when none did. The middle case keeps a single volume's answer
// exactly as ddgiSampleIrradiance gives it (no w * E / w rounding), so a
// one-volume field reads as it did before volumes.
fn ddgiBlendIrradiance(blended: vec3f, gatheredWeight: f32, firstRead: vec3f, firstContributing: vec3f, contributingVolumes: u32) -> vec3f {
    if (contributingVolumes == 0u) {
        return firstRead;
    }
    if (contributingVolumes == 1u) {
        return firstContributing;
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
    var firstRead = vec3f(0.0);
    var firstContributing = vec3f(0.0);
    var volumesRead = 0u;
    var contributingVolumes = 0u;
    var nearest = DdgiNearestVolume(0u, DDGI_FARTHEST_DISTANCE);
    for (var index = 0u; index < count; index++) {
        let volume = ddgiTableVolume(volumeTable, index);
        let edgeCoverage = ddgiVolumeEdgeCoverage(volume, position);
        if (!ddgiBlendReads(edgeCoverage)) {
            nearest = ddgiNearerVolume(nearest, index, volume, position);
            continue;
        }
        let field = ddgiAccumulateVolume(volume, position, normal, viewDirection, irradianceAtlas, irradianceSampler, distanceAtlas, distanceSampler, probeData);
        let irradiance = field.weightedIrradiance / max(field.weight, DDGI_MIN_WEIGHT_SUM);
        if (volumesRead == 0u) {
            firstRead = irradiance;
        }
        volumesRead++;
        let share = ddgiBlendShare(edgeCoverage, field.weight, gatheredWeight);
        if (share > 0.0) {
            if (contributingVolumes == 0u) {
                firstContributing = irradiance;
            }
            contributingVolumes++;
            blended += share * irradiance;
            gatheredWeight += share;
        }
        if (ddgiBlendDone(gatheredWeight, contributingVolumes)) {
            break;
        }
    }
    if (volumesRead == 0u && count > 0u) {
        // Outside every volume: the nearest one's nearest cell, with no weight (ddgiNearerVolume).
        let volume = ddgiTableVolume(volumeTable, nearest.index);
        let field = ddgiAccumulateVolume(volume, position, normal, viewDirection, irradianceAtlas, irradianceSampler, distanceAtlas, distanceSampler, probeData);
        firstRead = field.weightedIrradiance / max(field.weight, DDGI_MIN_WEIGHT_SUM);
    }
    let irradiance = ddgiBlendIrradiance(blended, gatheredWeight, firstRead, firstContributing, contributingVolumes);
    return DdgiVolumesSample(irradiance, gatheredWeight);
}
`;
