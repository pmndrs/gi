/**
 * How point and spot lights fall off, as a WGSL module of its own with no
 * bindings: three.js r187's inverse-square falloff with its range window, and
 * its spot cone.
 *
 * Hit shading depends on it (its direct light), and so can any technique
 * that needs a light's unshadowed irradiance at a point without binding the
 * scene, the tracer and the sky: DDGI's change estimate (`ddgi/mark`) works
 * out how much a light's change moved the light at each probe with exactly
 * the falloff that shades the probes' rays (docs/design/m2-lights.md §4.2).
 *
 * ```wgsl
 * fn giShadeDistanceAttenuation(lightDistance: f32, range: f32) -> f32
 * fn giShadeSpotAttenuation(cosine: f32, cosOuterCone: f32, cosInnerCone: f32) -> f32
 * ```
 */

import type { WgslModule } from '../contract/wgsl.ts';

export const LIGHT_FALLOFF_WGSL = /* wgsl */ `
// ---- Light falloff (platform/shade) -----------------------------------------------------

// three.js r187's distance falloff for point and spot lights, with decay 2
// (the platform's inverse-square units): src/nodes/lighting/LightUtils.js,
// getDistanceAttenuation, which cites Frostbite ("Moving Frostbite to PBR",
// Lagarde and de Rousiers 2014, p. 32, eq. 26):
//   falloff = 1 / max(d^2, 0.01), times (clamp(1 - (d / range)^4, 0, 1))^2 when range > 0.
// range = 0 means no window (three's distance = 0).
fn giShadeDistanceAttenuation(lightDistance: f32, range: f32) -> f32 {
    let inverseSquare = 1.0 / max(lightDistance * lightDistance, 0.01);
    if (range <= 0.0) {
        return inverseSquare;
    }
    let ratio = lightDistance / range;
    let window = clamp(1.0 - ratio * ratio * ratio * ratio, 0.0, 1.0);
    return inverseSquare * window * window;
}

// three.js r187's spot cone: src/nodes/lighting/SpotLightNode.js,
// getSpotAttenuation = smoothstep(cos(angle), cos(angle * (1 - penumbra)), cosine),
// where three's angle is the outer cone and angle * (1 - penumbra) the inner one.
// smoothstep is undefined when the edges meet (no penumbra), so that case is a hard edge.
fn giShadeSpotAttenuation(cosine: f32, cosOuterCone: f32, cosInnerCone: f32) -> f32 {
    if (cosInnerCone <= cosOuterCone) {
        return select(0.0, 1.0, cosine > cosOuterCone);
    }
    return smoothstep(cosOuterCone, cosInnerCone, cosine);
}
`;

/** The module: code only, nothing bound. One object, so the kernel builder includes it once however many modules depend on it. */
export const LIGHT_FALLOFF_MODULE: WgslModule = {
    name: 'gi-light-falloff',
    dependencies: [],
    code: () => LIGHT_FALLOFF_WGSL,
    entries: [],
    resources: () => [],
    version: 0,
    storageBufferCount: 0,
};
