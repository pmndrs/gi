/**
 * The `ddgi/write` kernel (docs/design/ddgi.md, "The write kernel"): copies
 * each updated probe's scratch slot into the atlases, fills the tile borders,
 * and writes the probe-data texel. One workgroup per updated probe.
 *
 * It exists because the update kernel samples the atlases (bounce light and
 * the old values), and a dispatch can't both sample a texture and store into
 * it; core WebGPU has read-write storage textures only for 32-bit
 * single-channel formats.
 *
 * Bindings (group 1; it uses no platform module): uniforms, the update list
 * and the scratch buffer (2 storage buffers), and the three atlases as
 * write-only storage textures.
 */

import type { DistanceFormat, ScratchLayout } from '../layout.ts';
import { PROBE_STATE, TILE_BORDER } from '../layout.ts';
import { DDGI_UNIFORMS_WGSL } from './update.ts';

export const WRITE_WORKGROUP_SIZE = 64;

export interface WriteKernelShape {
    readonly irradianceTexels: number;
    readonly distanceTexels: number;
    readonly scratch: ScratchLayout;
    readonly distanceFormat: DistanceFormat;
}

export function writeKernelWgsl(shape: WriteKernelShape): string {
    const { irradianceTexels, distanceTexels, scratch, distanceFormat } = shape;
    return /* wgsl */ `
${DDGI_UNIFORMS_WGSL}

// ---- DDGI write kernel (techniques/ddgi) -----------------------------------------------

const DDGI_WRITE_WORKGROUP_SIZE: u32 = ${WRITE_WORKGROUP_SIZE}u;
const DDGI_TILE_BORDER: u32 = ${TILE_BORDER}u;
const DDGI_IRRADIANCE_TEXELS: u32 = ${irradianceTexels}u;
const DDGI_DISTANCE_TEXELS: u32 = ${distanceTexels}u;
const DDGI_SCRATCH_STRIDE: u32 = ${scratch.strideWords}u;
const DDGI_SCRATCH_IRRADIANCE: u32 = ${scratch.irradianceOffset}u;
const DDGI_SCRATCH_DISTANCE: u32 = ${scratch.distanceOffset}u;
const DDGI_SCRATCH_PROBE_DATA: u32 = ${scratch.probeDataOffset}u;
const DDGI_STATE_OFF_INSIDE_GEOMETRY: u32 = ${PROBE_STATE.offInsideGeometry}u;
const DDGI_STATE_OFF_NO_SURFACE_NEARBY: u32 = ${PROBE_STATE.offNoSurfaceNearby}u;

@group(1) @binding(0) var<uniform> ddgi: DdgiUniforms;
@group(1) @binding(1) var<storage, read> ddgiUpdateList: array<u32>;
@group(1) @binding(2) var<storage, read> ddgiScratch: array<u32>;
@group(1) @binding(3) var ddgiIrradianceOut: texture_storage_2d_array<rgba16float, write>;
@group(1) @binding(4) var ddgiDistanceOut: texture_storage_2d_array<${distanceFormat}, write>;
@group(1) @binding(5) var ddgiProbeDataOut: texture_storage_2d_array<rgba16float, write>;

// The interior texel a tile texel copies. Tile texels run 0..interior+1 with
// the border at both ends. Each edge of an octahedral map mirrors about its
// midpoint and all four corners are one direction, so a border texel copies
// the interior texel mirrored across its edge, and a border corner copies the
// diagonally opposite interior corner (octahedral.ts has the same on the CPU).
fn ddgiBorderSource(tileTexel: vec2u, interior: u32) -> vec2u {
    let last = i32(interior) - 1;
    let inside = vec2i(tileTexel) - i32(DDGI_TILE_BORDER);
    let low = inside < vec2i(0);
    let high = inside > vec2i(last);
    let onBorder = low | high;
    let clamped = vec2u(clamp(inside, vec2i(0), vec2i(last)));
    if (all(onBorder)) {
        return select(vec2u(0u), vec2u(u32(last)), low);
    }
    if (onBorder.y) {
        return vec2u(u32(last) - clamped.x, clamped.y);
    }
    if (onBorder.x) {
        return vec2u(clamped.x, u32(last) - clamped.y);
    }
    return clamped;
}

@compute @workgroup_size(DDGI_WRITE_WORKGROUP_SIZE)
fn ddgiWrite(
    @builtin(workgroup_id) group: vec3u,
    @builtin(num_workgroups) groups: vec3u,
    @builtin(local_invocation_index) thread: u32,
) {
    let slot = ddgiUpdateSlot(group, groups);
    if (slot >= ddgi.counts.y) {
        return;
    }
    let probe = ddgiProbeCoordinates(ddgiUpdateList[slot], vec3u(ddgi.volumeCounts.xyz));
    let slotBase = slot * DDGI_SCRATCH_STRIDE;

    let offsetXy = unpack2x16float(ddgiScratch[slotBase + DDGI_SCRATCH_PROBE_DATA]);
    let offsetZAndState = unpack2x16float(ddgiScratch[slotBase + DDGI_SCRATCH_PROBE_DATA + 1u]);
    let state = u32(round(offsetZAndState.y));
    if (thread == 0u) {
        textureStore(ddgiProbeDataOut, probe.xz, probe.y, vec4f(offsetXy, offsetZAndState));
    }
    // Off probes were not blended this update: their tiles keep what they had.
    if (state == DDGI_STATE_OFF_INSIDE_GEOMETRY || state == DDGI_STATE_OFF_NO_SURFACE_NEARBY) {
        return;
    }

    let irradianceTile = DDGI_IRRADIANCE_TEXELS + 2u * DDGI_TILE_BORDER;
    for (var index = thread; index < irradianceTile * irradianceTile; index += DDGI_WRITE_WORKGROUP_SIZE) {
        let tileTexel = vec2u(index % irradianceTile, index / irradianceTile);
        let source = ddgiBorderSource(tileTexel, DDGI_IRRADIANCE_TEXELS);
        let word = slotBase + DDGI_SCRATCH_IRRADIANCE + 2u * (source.y * DDGI_IRRADIANCE_TEXELS + source.x);
        let rg = unpack2x16float(ddgiScratch[word]);
        let b = unpack2x16float(ddgiScratch[word + 1u]).x;
        textureStore(ddgiIrradianceOut, probe.xz * i32(irradianceTile) + vec2i(tileTexel), probe.y, vec4f(rg, b, 1.0));
    }

    let distanceTile = DDGI_DISTANCE_TEXELS + 2u * DDGI_TILE_BORDER;
    for (var index = thread; index < distanceTile * distanceTile; index += DDGI_WRITE_WORKGROUP_SIZE) {
        let tileTexel = vec2u(index % distanceTile, index / distanceTile);
        let source = ddgiBorderSource(tileTexel, DDGI_DISTANCE_TEXELS);
        let moments = unpack2x16float(ddgiScratch[slotBase + DDGI_SCRATCH_DISTANCE + source.y * DDGI_DISTANCE_TEXELS + source.x]);
        // rg16float keeps xy; the rgba16float fallback ignores zw.
        textureStore(ddgiDistanceOut, probe.xz * i32(distanceTile) + vec2i(tileTexel), probe.y, vec4f(moments, 0.0, 0.0));
    }
}
`;
}
