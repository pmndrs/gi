/**
 * The radiance cache's shading module, in the shape every technique
 * publishes (`platform/contract/shading-module.ts`).
 *
 * **Phase 2's first half (RC2.1) has no image yet:** the probes are traced
 * and blended, but the SH projection and the per-pixel interpolation that
 * turn them into irradiance come with RC2.2. Until then the module answers
 * no irradiance and no coverage, so a renderer keeps its own environment
 * light (coverage 0 means "fade to yours"), and binds nothing.
 */

import type { ShadingModule, ShadingParameter } from '../../platform/contract/shading-module.ts';

const SHADING_WGSL = /* wgsl */ `
// ---- radiance cache shading (RC2.1: nothing to read yet; RC2.2 interpolates the probes) ----
fn giIrradiance(position: vec3f, normal: vec3f, viewDirection: vec3f, screenUV: vec2f) -> vec3f {
    return vec3f(0.0);
}

fn giCoverage(position: vec3f, normal: vec3f) -> f32 {
    return 0.0;
}

fn giDebugColor(position: vec3f, normal: vec3f, view: u32) -> vec3f {
    return vec3f(0.0);
}
`;

const WRAPPERS_WGSL = /* wgsl */ `
fn giIrradianceAt(position: vec3f, normal: vec3f, viewDirection: vec3f, screenUV: vec2f) -> vec3f {
    return giIrradiance(position, normal, viewDirection, screenUV);
}

fn giCoverageAt(position: vec3f, normal: vec3f) -> f32 {
    return giCoverage(position, normal);
}

fn giDebugColorAt(position: vec3f, normal: vec3f, view: u32) -> vec3f {
    return giDebugColor(position, normal, view);
}
`;

export class RadianceCacheShadingModule implements ShadingModule {
    readonly techniqueId = 'radiance-cache';
    readonly code = SHADING_WGSL;
    readonly parameters: readonly ShadingParameter[] = [];
    readonly provides = { radiance: false } as const;
    readonly version = 1;

    /** `code` and the parameterless wrappers. No resources, so nothing is bound at `group`. */
    globalsCode(group: number): string {
        return `${this.code}\n// ---- radiance cache shading globals (globalsCode(${group})): no resources ----\n${WRAPPERS_WGSL}`;
    }

    /** No `vec4f` parameters: nothing to write. */
    writeUniforms(target: Float32Array): void {
        void target;
    }
}
