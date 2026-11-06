/**
 * How the sky changed, as a WGSL module with two uniform bindings and no
 * storage buffers: the sky's radiance SH (bands 0-2) as it is now, and as it
 * was when the techniques were last handed an environment change. A
 * technique compares the two in a kernel to size a sky change without
 * reading anything back to the CPU: DDGI's change estimate (`ddgi/mark`,
 * docs/design/m2-lights.md §4.2) does, and the instant sky's nudge (T2.6b,
 * §6.3) will.
 *
 * ```wgsl
 * fn giSkyRadianceShNow(index: u32) -> vec3f     // coefficient `index` (0..8) now
 * fn giSkyRadianceShBefore(index: u32) -> vec3f  // the same, when the techniques last saw a sky change
 * ```
 *
 * The platform keeps "before" itself: at the end of every frame whose
 * `SceneChanges` hand the techniques an environment change, it copies "now"
 * into it (`PlatformHitShading.recordSkySeen`). So changes made while the
 * budget is paused, which reach the techniques together in the next frame,
 * are measured together too, and a `setEnvironment` that changed nothing (or
 * turned the sun too little to re-bake) measures as no change.
 */

import { RADIANCE_SH_COUNT } from '../sky/projection-wgsl.ts';
import type { BindingSlot, ModuleBindingEntry } from '../contract/wgsl.ts';

const SHADER_STAGE_COMPUTE = 0x4;

/** Two uniform buffers: the radiance SH now, and before. */
export const SKY_CHANGE_MODULE_ENTRIES: readonly ModuleBindingEntry[] = [
    { visibility: SHADER_STAGE_COMPUTE, buffer: { type: 'uniform' } },
    { visibility: SHADER_STAGE_COMPUTE, buffer: { type: 'uniform' } },
];

/** Bytes of one binding: the radiance SH, one vec4f a coefficient. */
export const SKY_RADIANCE_SH_BYTES = RADIANCE_SH_COUNT * 16;

export function skyChangeModuleWgsl(slot: BindingSlot): string {
    return /* wgsl */ `
// ---- How the sky changed (platform/shade) --------------------------------------------------

struct GiSkyRadianceSh {
    // Bands 0-2, rgb per coefficient, in spherical-harmonics.ts's order; w unused.
    coefficients: array<vec4f, ${RADIANCE_SH_COUNT}>,
}

@group(${slot.group}) @binding(${slot.firstBinding}) var<uniform> giSkyRadianceNow: GiSkyRadianceSh;
@group(${slot.group}) @binding(${slot.firstBinding + 1}) var<uniform> giSkyRadianceBefore: GiSkyRadianceSh;

fn giSkyRadianceShNow(index: u32) -> vec3f {
    return giSkyRadianceNow.coefficients[index].xyz;
}

fn giSkyRadianceShBefore(index: u32) -> vec3f {
    return giSkyRadianceBefore.coefficients[index].xyz;
}
`;
}
