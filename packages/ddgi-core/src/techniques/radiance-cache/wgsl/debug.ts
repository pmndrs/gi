/**
 * The radiance cache's debug views (engine-free): a kernel that draws the
 * probe atlas, the probe mask or the probes' SH into a screen-sized
 * `rgba8unorm` image the host can show however it likes (a canvas, a texture
 * in its engine). The shading module's `giDebugColor` draws the same views
 * per pixel for engines (shading.ts).
 *
 * Capsaicin draws its probes with a graphics pass (`DebugScreenProbes`,
 * gi1.vert / gi1.frag); this is our own view of the same data, not a port.
 *
 * - **Probe atlas:** each pixel shows the atlas texel at the same place (the
 *   atlas is the screen rounded up to whole 8 × 8 tiles, one probe's 64 cells
 *   per tile), tone-mapped (`x / (1 + x)` after `exposure`) and gamma-encoded.
 *   Tiles with no probe this frame are a dark checkerboard.
 * - **Probe mask:** a tile with a probe gets a colour of its own (from its
 *   seed) and its seed pixel is white; a tile without one is black.
 * - **SH:** the irradiance of the pixel's probe tile's SH along the pixel's
 *   normal, divided by π (what a white diffuse surface would reflect),
 *   tone-mapped like the atlas.
 */

import { CAPSAICIN_WGSL } from './capsaicin.ts';
import { shIrradianceWgsl, SPHERICAL_HARMONICS_WGSL } from './spherical-harmonics.ts';

/** The views `RadianceCacheTechnique.encodeDebugView` draws. */
export const RADIANCE_CACHE_DEBUG_VIEW = {
    probeAtlas: 0,
    probeMask: 1,
    /** The irradiance from each probe's SH along the pixel's normal (RC2.2). */
    sphericalHarmonics: 2,
} as const;

export type RadianceCacheDebugViewId = (typeof RADIANCE_CACHE_DEBUG_VIEW)[keyof typeof RADIANCE_CACHE_DEBUG_VIEW];

/** Bytes of `RcDebug`. */
export const DEBUG_UNIFORM_BYTES = 32;

/** Byte offsets of `RcDebug`'s fields (checked against the struct by a unit test). */
export const DEBUG_UNIFORM_OFFSETS = { size: 0, view: 8, exposure: 12, normalUnorm: 16 } as const;

export const DEBUG_ENTRY_POINT = 'radianceCacheDebugView';

const COMPUTE = 0x4;

export const DEBUG_GROUP: GPUBindGroupLayoutDescriptor = {
    label: 'radiance-cache/debug-view group 1',
    entries: [
        { binding: 0, visibility: COMPUTE, buffer: { type: 'uniform', minBindingSize: DEBUG_UNIFORM_BYTES } },
        { binding: 1, visibility: COMPUTE, texture: { sampleType: 'unfilterable-float', viewDimension: '2d' } },
        { binding: 2, visibility: COMPUTE, texture: { sampleType: 'uint', viewDimension: '2d' } },
        { binding: 3, visibility: COMPUTE, storageTexture: { access: 'write-only', format: 'rgba8unorm', viewDimension: '2d' } },
        { binding: 4, visibility: COMPUTE, texture: { sampleType: 'uint', viewDimension: '2d' } },
        { binding: 5, visibility: COMPUTE, texture: { sampleType: 'unfilterable-float', viewDimension: '2d' } },
    ],
};

export const DEBUG_WGSL = /* wgsl */ `
${CAPSAICIN_WGSL}
${SPHERICAL_HARMONICS_WGSL}

struct RcDebug {
    size: vec2u,     // the image (the screen) in pixels
    view: u32,       // RADIANCE_CACHE_DEBUG_VIEW
    exposure: f32,   // multiplies radiance before tone mapping
    normalUnorm: u32, // 1: the screen's normals are 0..1 encoded (RcFrame.normalUnorm)
}
@group(1) @binding(0) var<uniform> rcDebug: RcDebug;
@group(1) @binding(1) var rcProbeBuffer: texture_2d<f32>;
@group(1) @binding(2) var rcProbeMask: texture_2d<u32>;
@group(1) @binding(3) var rcDebugImage: texture_storage_2d<rgba8unorm, write>;
// The SH texture (register D26) and the screen's normals (FrameScreen.normal).
@group(1) @binding(4) var rcProbeSH: texture_2d<u32>;
@group(1) @binding(5) var rcNormal: texture_2d<f32>;

const RC_DEBUG_PROBE_ATLAS: u32 = ${RADIANCE_CACHE_DEBUG_VIEW.probeAtlas}u;
const RC_DEBUG_SH: u32 = ${RADIANCE_CACHE_DEBUG_VIEW.sphericalHarmonics}u;
const RC_DEBUG_INV_PI: f32 = 0.318309886183791;

${shIrradianceWgsl('rcDebugSHIrradiance', 'probeSH')}

// A display colour: Reinhard tone mapping, then an approximate sRGB curve.
fn rcDebugDisplay(radiance: vec3f) -> vec3f {
    let mapped = max(radiance, vec3f(0.0)) * rcDebug.exposure;
    return pow(mapped / (1.0 + mapped), vec3f(1.0 / 2.2));
}

// A bright colour of its own for each seed, so neighbouring probes tell apart.
fn rcDebugSeedColour(packedSeed: u32) -> vec3f {
    let hash = rcPcgHash(packedSeed);
    return vec3f(f32(hash & 0xffu), f32((hash >> 8u) & 0xffu), f32((hash >> 16u) & 0xffu)) / 255.0 * 0.7 + 0.3;
}

@compute @workgroup_size(8, 8)
fn ${DEBUG_ENTRY_POINT}(@builtin(global_invocation_id) id: vec3u) {
    if (any(id.xy >= rcDebug.size)) {
        return;
    }
    let tile = id.xy / RC_PROBE_SIZE;
    let packedSeed = textureLoad(rcProbeMask, tile, 0).x;
    var colour: vec3f;
    if (packedSeed == RC_INVALID_ID) {
        // No probe in this tile this frame: a dark checkerboard by tile.
        colour = select(vec3f(0.02), vec3f(0.06), ((tile.x + tile.y) & 1u) == 0u);
    } else if (rcDebug.view == RC_DEBUG_PROBE_ATLAS) {
        colour = rcDebugDisplay(textureLoad(rcProbeBuffer, id.xy, 0).rgb);
    } else if (rcDebug.view == RC_DEBUG_SH) {
        let stored = textureLoad(rcNormal, id.xy, 0).xyz;
        let normal = normalize(select(stored, 2.0 * stored - 1.0, rcDebug.normalUnorm != 0u));
        colour = rcDebugDisplay(rcDebugSHIrradiance(normal, 1.0, tile, rcProbeSH) * RC_DEBUG_INV_PI);
    } else {
        colour = select(rcDebugSeedColour(packedSeed), vec3f(1.0), all(rcUnpackSeed(packedSeed) == id.xy));
    }
    textureStore(rcDebugImage, id.xy, vec4f(colour, 1.0));
}
`;
