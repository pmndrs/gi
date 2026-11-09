/**
 * The radiance cache's shading module, in the shape every technique
 * publishes (`platform/contract/shading-module.ts`).
 *
 * - **`giIrradiance`** reads the irradiance image `InterpolateScreenProbes`
 *   wrote this frame (screen-irradiance.ts) at the pixel `screenUV` names:
 *   irradiance E, which the renderer turns into reflected light with
 *   `albedo / π` (three's composite does; Capsaicin's `ResolveGI1`,
 *   gi1.frag:76-150, does the same with `evaluateLambert`). Position, normal
 *   and view direction are not needed: the screen stage worked them out from
 *   the host's depth and normals. The image is the screen's own size, so a
 *   pixel reads its own texel.
 * - **`giCoverage`** is 1: the cache answers for every pixel on screen.
 * - **`giDebugColor`** draws the debug views by `screenUV`
 *   ({@link RADIANCE_CACHE_DEBUG_VIEW}): the probe atlas, the probe mask, and
 *   the SH (the irradiance of the pixel's own probe tile's SH along the
 *   pixel's normal, divided by π: what a white diffuse surface would reflect).
 *
 * Resources: four textures, all read with `textureLoad` (no samplers):
 * the irradiance image, the probe atlas, the probe mask (`r32uint`) and the
 * SH texture (`rg32uint`, register D26). Each is a 1 × 1 placeholder until
 * the first screen stage, and a new texture after a resize (hosts re-read
 * `texture()` each frame). `version` doesn't change: the code and the
 * parameters' shape are fixed.
 */

import type { ShadingModule, ShadingParameter } from '../../platform/contract/shading-module.ts';
import { CAPSAICIN_WGSL } from './wgsl/capsaicin.ts';
import { RADIANCE_CACHE_DEBUG_VIEW } from './wgsl/debug.ts';
import { shIrradianceWgsl, SPHERICAL_HARMONICS_WGSL } from './wgsl/spherical-harmonics.ts';

/** The textures the module reads, current as of the latest screen stage (placeholders before it). */
export interface RadianceCacheShadingSource {
    shadingTextures(): { readonly irradiance: GPUTexture; readonly probeAtlas: GPUTexture; readonly probeMask: GPUTexture; readonly probeSH: GPUTexture };
}

/** The parameter names, which are also the WGSL parameter names below. */
export const RADIANCE_CACHE_SHADING_PARAMETER_NAMES = {
    irradiance: 'rcIrradiance',
    probeAtlas: 'rcProbeAtlas',
    probeMask: 'rcProbeMask',
    probeSH: 'rcProbeSH',
} as const;

const NAMES = RADIANCE_CACHE_SHADING_PARAMETER_NAMES;

/** Every entry point's resource parameters, in the order of `parameters`. */
const RESOURCE_PARAMETERS = `${NAMES.irradiance}: texture_2d<f32>, ${NAMES.probeAtlas}: texture_2d<f32>, ${NAMES.probeMask}: texture_2d<u32>, ${NAMES.probeSH}: texture_2d<u32>`;
const RESOURCE_ARGUMENTS = `${NAMES.irradiance}, ${NAMES.probeAtlas}, ${NAMES.probeMask}, ${NAMES.probeSH}`;

const SHADING_WGSL = /* wgsl */ `
// ---- radiance cache shading (RC2.2): the interpolated irradiance by screen position ----
${CAPSAICIN_WGSL}
${SPHERICAL_HARMONICS_WGSL}
${shIrradianceWgsl('rcShadingSHIrradiance', 'probeSH')}

const RC_DEBUG_VIEW_PROBE_ATLAS: u32 = ${RADIANCE_CACHE_DEBUG_VIEW.probeAtlas}u;
const RC_DEBUG_VIEW_PROBE_MASK: u32 = ${RADIANCE_CACHE_DEBUG_VIEW.probeMask}u;
const RC_DEBUG_VIEW_SH: u32 = ${RADIANCE_CACHE_DEBUG_VIEW.sphericalHarmonics}u;
const RC_SHADING_INV_PI: f32 = 0.318309886183791;

// The texel of a size × size image under screenUV (0..1, top left 0), kept inside it.
fn rcShadingPixel(screenUV: vec2f, size: vec2u) -> vec2u {
    return min(vec2u(max(screenUV, vec2f(0.0)) * vec2f(size)), size - 1u);
}

fn giIrradiance(position: vec3f, normal: vec3f, viewDirection: vec3f, screenUV: vec2f, ${RESOURCE_PARAMETERS}) -> vec3f {
    let size = textureDimensions(${NAMES.irradiance});
    return textureLoad(${NAMES.irradiance}, rcShadingPixel(screenUV, size), 0).rgb;
}

fn giCoverage(position: vec3f, normal: vec3f, ${RESOURCE_PARAMETERS}) -> f32 {
    return 1.0;
}

// A bright colour of its own for each seed, so neighbouring probes tell apart (wgsl/debug.ts).
fn rcShadingSeedColour(packedSeed: u32) -> vec3f {
    let hash = rcPcgHash(packedSeed);
    return vec3f(f32(hash & 0xffu), f32((hash >> 8u) & 0xffu), f32((hash >> 16u) & 0xffu)) / 255.0 * 0.7 + 0.3;
}

fn giDebugColor(position: vec3f, normal: vec3f, view: u32, screenUV: vec2f, ${RESOURCE_PARAMETERS}) -> vec3f {
    let pixel = rcShadingPixel(screenUV, textureDimensions(${NAMES.irradiance}));
    let maskSize = textureDimensions(${NAMES.probeMask}, 0);
    let tile = min(pixel / RC_PROBE_SIZE, maskSize - 1u);
    let packedSeed = textureLoad(${NAMES.probeMask}, tile, 0).x;
    if (packedSeed == RC_INVALID_ID) {
        // No probe in this tile this frame: a dark checkerboard by tile.
        return select(vec3f(0.02), vec3f(0.06), ((tile.x + tile.y) & 1u) == 0u);
    }
    if (view == RC_DEBUG_VIEW_PROBE_ATLAS) {
        // The atlas is the screen in whole probe tiles: each pixel shows the cell at its place.
        return textureLoad(${NAMES.probeAtlas}, pixel, 0).rgb;
    }
    if (view == RC_DEBUG_VIEW_SH) {
        return rcShadingSHIrradiance(normalize(normal), 1.0, tile, ${NAMES.probeSH}) * RC_SHADING_INV_PI;
    }
    // The probe mask: the tile's colour, its seed pixel white.
    return select(rcShadingSeedColour(packedSeed), vec3f(1.0), all(rcUnpackSeed(packedSeed) == pixel));
}
`;

/** The entry points and the arguments each takes before the resources, for `globalsCode`'s wrappers. */
const ENTRY_POINTS: readonly { readonly name: string; readonly arguments: string; readonly call: string; readonly returns: string }[] = [
    { name: 'giIrradiance', arguments: 'position: vec3f, normal: vec3f, viewDirection: vec3f, screenUV: vec2f', call: 'position, normal, viewDirection, screenUV', returns: 'vec3f' },
    { name: 'giCoverage', arguments: 'position: vec3f, normal: vec3f', call: 'position, normal', returns: 'f32' },
    { name: 'giDebugColor', arguments: 'position: vec3f, normal: vec3f, view: u32, screenUV: vec2f', call: 'position, normal, view, screenUV', returns: 'vec3f' },
];

export class RadianceCacheShadingModule implements ShadingModule {
    readonly techniqueId = 'radiance-cache';
    readonly code = SHADING_WGSL;
    readonly parameters: readonly ShadingParameter[];
    readonly provides = { radiance: false } as const;
    readonly version = 1;

    constructor(source: RadianceCacheShadingSource) {
        const texture = (name: string, sampleType: 'float' | 'uint', pick: (textures: ReturnType<RadianceCacheShadingSource['shadingTextures']>) => GPUTexture): ShadingParameter => ({
            kind: 'texture',
            name,
            dimension: '2d',
            sampleType,
            texture: () => pick(source.shadingTextures()),
        });
        this.parameters = [
            // rgba16float, read with textureLoad.
            texture(NAMES.irradiance, 'float', (textures) => textures.irradiance),
            texture(NAMES.probeAtlas, 'float', (textures) => textures.probeAtlas),
            texture(NAMES.probeMask, 'uint', (textures) => textures.probeMask),
            texture(NAMES.probeSH, 'uint', (textures) => textures.probeSH),
        ];
    }

    /**
     * `code`, the four textures as globals at `group` (bindings 0 to 3, in
     * parameter order; no uniforms), then the wrappers
     * `giIrradianceAt(position, normal, viewDirection, screenUV)`,
     * `giCoverageAt(position, normal)` and
     * `giDebugColorAt(position, normal, view, screenUV)`.
     */
    globalsCode(group: number): string {
        const lines = [this.code, `// ---- radiance cache shading globals (globalsCode(${group})) ----`];
        this.parameters.forEach((parameter, binding) => {
            if (parameter.kind !== 'texture') return;
            lines.push(`@group(${group}) @binding(${binding}) var ${parameter.name}: texture_2d<${parameter.sampleType === 'uint' ? 'u32' : 'f32'}>;`);
        });
        for (const entry of ENTRY_POINTS) {
            lines.push(`fn ${entry.name}At(${entry.arguments}) -> ${entry.returns} {`, `    return ${entry.name}(${entry.call}, ${RESOURCE_ARGUMENTS});`, '}');
        }
        return `${lines.join('\n')}\n`;
    }

    /** No `vec4f` parameters: nothing to write. */
    writeUniforms(target: Float32Array): void {
        void target;
    }
}
