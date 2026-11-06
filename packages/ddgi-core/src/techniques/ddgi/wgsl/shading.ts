/**
 * The WGSL of DDGI's shading module (T1.6): the contract's entry points
 * (`platform/contract/shading-module.ts`) on top of the one sampling function.
 *
 * ```wgsl
 * fn giIrradiance(position: vec3f, normal: vec3f, viewDirection: vec3f, screenUV: vec2f, <resources>) -> vec3f
 * fn giCoverage(position: vec3f, normal: vec3f, <resources>) -> f32
 * fn giDebugColor(position: vec3f, normal: vec3f, view: u32, <resources>) -> vec3f
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
 * ddgiProbeData: texture_2d_array<f32>,
 * ddgiVolumeOrigin: vec4f, ddgiVolumeSpacing: vec4f, ddgiVolumeCounts: vec4f, ddgiVolumeTexels: vec4f
 * ```
 *
 * `giIrradiance` is `ddgiSampleIrradiance` unchanged, the same function the
 * update kernel uses for bounce light, so renderers and the update share one
 * code path. No `giRadiance`: the module says `provides.radiance = false`.
 */

import { DDGI_SAMPLE_WGSL } from './sample.ts';

/** Ids of DDGI's screen debug views, the `view` argument of `giDebugColor` (id 0 is the `points` view). */
export const DDGI_DEBUG_VIEW = {
    points: 0,
    giOnly: 1,
    coverage: 2,
    probeStates: 3,
} as const;

/**
 * Coverage: how far past the outermost probes the field still counts as
 * fully covering a point, and over how much farther it then fades to 0, both
 * in probe spacings.
 *
 * Each probe answers for the half cell around it, so the volume the grid
 * covers is the probe box grown by half a spacing; the settings box always
 * lies inside it (probe counts are `floor(size / spacing) + 1`, centred).
 * Past that, sampling extrapolates the outermost cell, so coverage fades out
 * over the next half cell. A room whose walls sit just outside the outermost
 * probes (Cornell: 3 cm) keeps full coverage on its walls.
 */
export const COVERAGE_FULL_MARGIN_SPACINGS = 0.5;
export const COVERAGE_FADE_SPACINGS = 0.5;

/**
 * Coverage: the summed probe weight (trilinear × direction × visibility,
 * after the crush) at which a point counts as fully seen.
 *
 * Why this value: a probe that sees the point keeps at least the direction
 * floor (0.2) times its trilinear share, while an occluded one keeps at most
 * 1.2 × 0.05 = 0.06 before the crush and about 0.005 after it. So a sum of
 * 0.02 needs a visible probe with at least a 10% trilinear share, and 8
 * occluded probes alone stay below it.
 */
export const COVERAGE_FULL_WEIGHT = 0.02;

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
const DDGI_COVERAGE_FULL_MARGIN: f32 = ${COVERAGE_FULL_MARGIN_SPACINGS};
const DDGI_COVERAGE_FADE: f32 = ${COVERAGE_FADE_SPACINGS};
const DDGI_COVERAGE_FULL_WEIGHT: f32 = ${COVERAGE_FULL_WEIGHT};

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

// 1 within half a spacing of the outermost probes, falling to 0 over the
// next half spacing (COVERAGE_FULL_MARGIN_SPACINGS in shading.ts says why).
fn ddgiVolumeEdgeCoverage(volume: DdgiVolume, position: vec3f) -> f32 {
    let firstProbe = volume.origin;
    let lastProbe = volume.origin + vec3f(volume.counts - 1) * volume.spacing;
    // Distance outside the probe box per axis, in spacings (0 inside).
    let outside = max(max(firstProbe - position, position - lastProbe), vec3f(0.0)) / volume.spacing;
    let farthest = max(max(outside.x, outside.y), outside.z);
    return saturate(1.0 - (farthest - DDGI_COVERAGE_FULL_MARGIN) / DDGI_COVERAGE_FADE);
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
        let probe = min(cage.baseProbe + step, volume.counts - 1);
        color += ddgiTrilinearWeight(cage, step) * ddgiStateColor(ddgiReadProbeData(probeData, probe).state);
    }
    return color;
}

// Irradiance E at a surface point (the renderer applies albedo / pi).
// screenUV is for screen-space techniques; DDGI doesn't use it.
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
    ddgiVolumeOrigin: vec4f,
    ddgiVolumeSpacing: vec4f,
    ddgiVolumeCounts: vec4f,
    ddgiVolumeTexels: vec4f,
) -> vec3f {
    return ddgiSampleIrradiance(
        position, normal, viewDirection,
        ddgiIrradiance, ddgiIrradianceSampler, ddgiDistance, ddgiDistanceSampler, ddgiProbeData,
        ddgiVolumeOrigin, ddgiVolumeSpacing, ddgiVolumeCounts, ddgiVolumeTexels,
    );
}

// How much of the answer DDGI is sure of, 0..1: the volume's edge fade times
// whether any active probe can see the point (design doc, "Coverage").
// Renderers fade to their own environment light where it is low.
fn giCoverage(
    position: vec3f,
    normal: vec3f,
    ddgiIrradiance: texture_2d_array<f32>,
    ddgiIrradianceSampler: sampler,
    ddgiDistance: texture_2d_array<f32>,
    ddgiDistanceSampler: sampler,
    ddgiProbeData: texture_2d_array<f32>,
    ddgiVolumeOrigin: vec4f,
    ddgiVolumeSpacing: vec4f,
    ddgiVolumeCounts: vec4f,
    ddgiVolumeTexels: vec4f,
) -> f32 {
    let volume = ddgiVolume(ddgiVolumeOrigin, ddgiVolumeSpacing, ddgiVolumeCounts, ddgiVolumeTexels);
    let field = ddgiAccumulateField(
        position, normal, ddgiHeadOnViewDirection(normal),
        ddgiIrradiance, ddgiIrradianceSampler, ddgiDistance, ddgiDistanceSampler, ddgiProbeData,
        ddgiVolumeOrigin, ddgiVolumeSpacing, ddgiVolumeCounts, ddgiVolumeTexels,
    );
    let seen = saturate(field.weight / DDGI_COVERAGE_FULL_WEIGHT);
    return ddgiVolumeEdgeCoverage(volume, position) * seen;
}

// A colour for one of DDGI's screen debug views (DDGI_DEBUG_VIEW in shading.ts).
fn giDebugColor(
    position: vec3f,
    normal: vec3f,
    view: u32,
    ddgiIrradiance: texture_2d_array<f32>,
    ddgiIrradianceSampler: sampler,
    ddgiDistance: texture_2d_array<f32>,
    ddgiDistanceSampler: sampler,
    ddgiProbeData: texture_2d_array<f32>,
    ddgiVolumeOrigin: vec4f,
    ddgiVolumeSpacing: vec4f,
    ddgiVolumeCounts: vec4f,
    ddgiVolumeTexels: vec4f,
) -> vec3f {
    let viewDirection = ddgiHeadOnViewDirection(normal);
    switch view {
        // The indirect light alone, on a white matte surface: E / pi.
        case DDGI_DEBUG_VIEW_GI_ONLY: {
            let irradiance = ddgiSampleIrradiance(
                position, normal, viewDirection,
                ddgiIrradiance, ddgiIrradianceSampler, ddgiDistance, ddgiDistanceSampler, ddgiProbeData,
                ddgiVolumeOrigin, ddgiVolumeSpacing, ddgiVolumeCounts, ddgiVolumeTexels,
            );
            return irradiance / DDGI_PI;
        }
        case DDGI_DEBUG_VIEW_COVERAGE: {
            let coverage = giCoverage(
                position, normal,
                ddgiIrradiance, ddgiIrradianceSampler, ddgiDistance, ddgiDistanceSampler, ddgiProbeData,
                ddgiVolumeOrigin, ddgiVolumeSpacing, ddgiVolumeCounts, ddgiVolumeTexels,
            );
            return mix(DDGI_COLOR_NO_COVERAGE, DDGI_COLOR_FULL_COVERAGE, coverage);
        }
        case DDGI_DEBUG_VIEW_PROBE_STATES: {
            let volume = ddgiVolume(ddgiVolumeOrigin, ddgiVolumeSpacing, ddgiVolumeCounts, ddgiVolumeTexels);
            return ddgiProbeStatesColor(volume, ddgiBiasedPoint(volume, position, normal, viewDirection), ddgiProbeData);
        }
        default: {
            return DDGI_COLOR_UNKNOWN_VIEW;
        }
    }
}
`;
