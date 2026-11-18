/**
 * The WGSL of DDGI's shading module (T1.6): the contract's entry points
 * (`platform/contract/shading-module.ts`) on top of the one sampling function.
 *
 * ```wgsl
 * fn giIrradiance(position: vec3f, normal: vec3f, viewDirection: vec3f, screenUV: vec2f, <resources>) -> vec3f
 * fn giCoverage(position: vec3f, normal: vec3f, <resources>) -> f32
 * fn giDebugColor(position: vec3f, normal: vec3f, view: u32, screenUV: vec2f, <resources>) -> vec3f
 * ```
 *
 * `<resources>` are the module's parameters in order
 * (`DDGIShadingModule.parameters` in shading.ts, named by
 * `DDGI_SHADING_PARAMETER_NAMES`), and the WGSL parameter names are the parameter
 * names, so an engine that matches inputs by name (three's `wgslFn`) can call
 * through directly:
 *
 * ```wgsl
 * ddgiIrradiance: texture_2d_array<f32>, ddgiIrradianceSampler: sampler,
 * ddgiDistance: texture_2d_array<f32>, ddgiDistanceSampler: sampler,
 * ddgiProbeData: texture_2d_array<f32>, ddgiVolumeTable: texture_2d<f32>
 * ```
 *
 * `giIrradiance` and `giCoverage` are `ddgiSampleVolumes`, the blend of
 * every volume (m3-moving.md §4.1) that the update kernel also uses for
 * bounce light, so renderers and the update share one code path; one volume
 * goes through it as several do. No `giRadiance`: the module says
 * `provides.radiance = false`.
 */

import { DDGI_SAMPLE_WGSL } from './sample.ts';

// The coverage numbers live with the volume blend now (volumes.ts); exported here too, where checks found them first.
export { COVERAGE_FADE_SPACINGS, COVERAGE_FULL_MARGIN_SPACINGS, COVERAGE_FULL_WEIGHT } from '../volumes.ts';

/** Ids of DDGI's screen debug views, the `view` argument of `giDebugColor` (id 0 is the `points` view). */
export const DDGI_DEBUG_VIEW = {
    points: 0,
    giOnly: 1,
    coverage: 2,
    probeStates: 3,
} as const;

/**
 * The debug views' colours (linear rgb), chosen to read apart at a glance.
 * Exported so checks can work out what a view should show.
 */
export const DDGI_DEBUG_COLORS = {
    fullCoverage: [0.1, 0.75, 0.2],
    noCoverage: [0.85, 0.1, 0.1],
    stateActive: [0.1, 0.75, 0.2],
    stateInsideGeometry: [0.85, 0.1, 0.1],
    stateNoSurfaceNearby: [0.15, 0.35, 0.95],
    stateNotUpdatedYet: [0.95, 0.8, 0.1],
    // An id the module doesn't know: magenta, so a wrong id is obvious.
    unknownView: [1, 0, 1],
} as const satisfies Record<string, readonly [number, number, number]>;

function wgslColor(color: readonly [number, number, number]): string {
    return `vec3f(${color.map((channel) => channel.toFixed(4)).join(', ')})`;
}

export const DDGI_SHADING_WGSL = /* wgsl */ `
${DDGI_SAMPLE_WGSL}
// ---- DDGI shading module (techniques/ddgi/shading.ts) -----------------------------------

const DDGI_DEBUG_VIEW_GI_ONLY: u32 = ${DDGI_DEBUG_VIEW.giOnly}u;
const DDGI_DEBUG_VIEW_COVERAGE: u32 = ${DDGI_DEBUG_VIEW.coverage}u;
const DDGI_DEBUG_VIEW_PROBE_STATES: u32 = ${DDGI_DEBUG_VIEW.probeStates}u;

// Debug colours (DDGI_DEBUG_COLORS in wgsl/shading.ts).
const DDGI_COLOR_FULL_COVERAGE: vec3f = ${wgslColor(DDGI_DEBUG_COLORS.fullCoverage)};
const DDGI_COLOR_NO_COVERAGE: vec3f = ${wgslColor(DDGI_DEBUG_COLORS.noCoverage)};
const DDGI_COLOR_STATE_ACTIVE: vec3f = ${wgslColor(DDGI_DEBUG_COLORS.stateActive)};
const DDGI_COLOR_STATE_INSIDE_GEOMETRY: vec3f = ${wgslColor(DDGI_DEBUG_COLORS.stateInsideGeometry)};
const DDGI_COLOR_STATE_NO_SURFACE: vec3f = ${wgslColor(DDGI_DEBUG_COLORS.stateNoSurfaceNearby)};
const DDGI_COLOR_STATE_NOT_UPDATED: vec3f = ${wgslColor(DDGI_DEBUG_COLORS.stateNotUpdatedYet)};
const DDGI_COLOR_UNKNOWN_VIEW: vec3f = ${wgslColor(DDGI_DEBUG_COLORS.unknownView)};

// giCoverage and giDebugColor get no view direction (the contract gives
// none), so they look at the surface head-on: the view bias then pushes the
// point straight off the surface, along with the normal bias.
fn ddgiHeadOnViewDirection(normal: vec3f) -> vec3f {
    return -normal;
}

// The state colour of one probe, for the probe-states debug view.
fn ddgiStateColor(state: u32) -> vec3f {
    switch state {
        case DDGI_STATE_ACTIVE: { return DDGI_COLOR_STATE_ACTIVE; }
        case DDGI_STATE_OFF_INSIDE_GEOMETRY: { return DDGI_COLOR_STATE_INSIDE_GEOMETRY; }
        case DDGI_STATE_OFF_NO_SURFACE_NEARBY: { return DDGI_COLOR_STATE_NO_SURFACE; }
        default: { return DDGI_COLOR_STATE_NOT_UPDATED; }
    }
}

// The states of the 8 probes the field would blend at this point, mixed by
// their trilinear weights (no direction or visibility weighting, so every
// probe of the cage shows).
fn ddgiProbeStatesColor(volume: DdgiVolume, samplePoint: vec3f, probeData: texture_2d_array<f32>) -> vec3f {
    let cage = ddgiCage(volume, samplePoint);
    var color = vec3f(0.0);
    for (var corner = 0u; corner < 8u; corner++) {
        let step = ddgiCageStep(corner);
        let slot = ddgiProbeSlot(volume, min(cage.baseProbe + step, volume.counts - 1));
        color += ddgiTrilinearWeight(cage, step) * ddgiStateColor(ddgiReadProbeData(probeData, volume, slot).state);
    }
    return color;
}

// The volume whose probes the probe-states view shows at a point: the first
// in priority order that covers it at all, else the nearest (the one a point
// outside every volume reads, ddgiNearerVolume).
fn ddgiShownVolume(volumeTable: texture_2d<f32>, position: vec3f) -> DdgiVolume {
    let count = ddgiVolumeCount(volumeTable);
    var nearest = DdgiNearestVolume(0u, DDGI_FARTHEST_DISTANCE);
    for (var index = 0u; index < count; index++) {
        let volume = ddgiTableVolume(volumeTable, index);
        if (ddgiVolumeEdgeCoverage(volume, position) > 0.0) {
            return volume;
        }
        nearest = ddgiNearerVolume(nearest, index, volume, position);
    }
    return ddgiTableVolume(volumeTable, nearest.index);
}

// Irradiance E at a surface point (the renderer applies albedo / pi), every
// volume blended by priority. screenUV is for screen-space techniques; DDGI
// doesn't use it.
fn giIrradiance(
    position: vec3f,
    normal: vec3f,
    viewDirection: vec3f,
    screenUV: vec2f,
    ddgiIrradiance: texture_2d_array<f32>,
    ddgiIrradianceSampler: sampler,
    ddgiDistance: texture_2d_array<f32>,
    ddgiDistanceSampler: sampler,
    ddgiProbeData: texture_2d_array<f32>,
    ddgiVolumeTable: texture_2d<f32>,
) -> vec3f {
    return ddgiSampleVolumes(
        position, normal, viewDirection,
        ddgiIrradiance, ddgiIrradianceSampler, ddgiDistance, ddgiDistanceSampler, ddgiProbeData, ddgiVolumeTable,
    ).irradiance;
}

// How much of the answer DDGI is sure of, 0..1: the weight the volume blend
// gathered (design doc, "Coverage"; m3-moving.md §4.1). With one volume, its
// edge fade times whether any active probe can see the point. Renderers fade
// to their own environment light where it is low.
fn giCoverage(
    position: vec3f,
    normal: vec3f,
    ddgiIrradiance: texture_2d_array<f32>,
    ddgiIrradianceSampler: sampler,
    ddgiDistance: texture_2d_array<f32>,
    ddgiDistanceSampler: sampler,
    ddgiProbeData: texture_2d_array<f32>,
    ddgiVolumeTable: texture_2d<f32>,
) -> f32 {
    return ddgiSampleVolumes(
        position, normal, ddgiHeadOnViewDirection(normal),
        ddgiIrradiance, ddgiIrradianceSampler, ddgiDistance, ddgiDistanceSampler, ddgiProbeData, ddgiVolumeTable,
    ).coverage;
}

// A colour for one of DDGI's screen debug views (DDGI_DEBUG_VIEW in shading.ts). DDGI's views
// are drawn from the field at the point; the screen position (the contract's, for screen-space
// techniques) is not needed.
fn giDebugColor(
    position: vec3f,
    normal: vec3f,
    view: u32,
    screenUV: vec2f,
    ddgiIrradiance: texture_2d_array<f32>,
    ddgiIrradianceSampler: sampler,
    ddgiDistance: texture_2d_array<f32>,
    ddgiDistanceSampler: sampler,
    ddgiProbeData: texture_2d_array<f32>,
    ddgiVolumeTable: texture_2d<f32>,
) -> vec3f {
    let viewDirection = ddgiHeadOnViewDirection(normal);
    switch view {
        // The indirect light alone, on a white matte surface: E / pi.
        case DDGI_DEBUG_VIEW_GI_ONLY: {
            let irradiance = giIrradiance(
                position, normal, viewDirection, vec2f(0.0),
                ddgiIrradiance, ddgiIrradianceSampler, ddgiDistance, ddgiDistanceSampler, ddgiProbeData, ddgiVolumeTable,
            );
            return irradiance / DDGI_PI;
        }
        case DDGI_DEBUG_VIEW_COVERAGE: {
            let coverage = giCoverage(
                position, normal,
                ddgiIrradiance, ddgiIrradianceSampler, ddgiDistance, ddgiDistanceSampler, ddgiProbeData, ddgiVolumeTable,
            );
            return mix(DDGI_COLOR_NO_COVERAGE, DDGI_COLOR_FULL_COVERAGE, coverage);
        }
        case DDGI_DEBUG_VIEW_PROBE_STATES: {
            let volume = ddgiShownVolume(ddgiVolumeTable, position);
            return ddgiProbeStatesColor(volume, ddgiBiasedPoint(volume, position, normal, viewDirection), ddgiProbeData);
        }
        default: {
            return DDGI_COLOR_UNKNOWN_VIEW;
        }
    }
}
`;
