/**
 * `ddgi/reset-probes` (DV-12, docs/divergence.md; the CPU half is
 * probe-reset.ts): `DDGITechnique.resetProbes` on the GPU. One thread per
 * probe of one volume, a dispatch per volume, in one timed pass before the
 * volumes' own passes.
 *
 * Each probe's probe-data texel holds its relocation offset (xyz) and its
 * state (w) together, and the CPU knows neither, so the kernel reads the
 * texel, changes what was asked for and keeps the rest:
 *
 * - **offsets:** the offset goes back to 0, so the probe sits at its grid
 *   point again; relocation moves it at its next updates as it did the first
 *   time;
 * - **states:** a probe that classification switched off (state 1 or 2)
 *   becomes state 3, "not updated yet": exactly what `ddgiDecide` in
 *   wgsl/blend.ts does when an off probe comes back on. Its next update
 *   traces all its rays, classification decides again, and if it stays on
 *   it starts its irradiance from nothing and is used in shading after two
 *   blended updates. Active probes (0) and probes not updated yet (3) keep
 *   their state: classification looks at them again at every update anyway.
 *   *Why not "every probe active", as RTXGI's reset:* an off probe's tiles
 *   hold whatever it had before it went off (black for one that was never
 *   on), and a state-3 probe hasn't been blended yet; shading would read both
 *   at once. Through state 3 nothing the renderers see changes in the frame
 *   of the reset.
 *
 * Irradiance, distances and the probe's history record are not touched.
 *
 * The `points` debug view is written too (the probe at its grid point plus
 * its offset, and its state), as the write kernel does after an update, so
 * the view shows the reset in the same frame.
 *
 * *Why a copy:* a pass can't read and write the `rgba16float` probe-data
 * texture in one dispatch in core WebGPU (read-write storage for that format
 * needs `texture-formats-tier2`), so the technique first copies the texture
 * and the kernel reads the copy.
 */

import { PROBE_STATE } from '../layout.ts';

export const PROBE_RESET_ENTRY_POINT = 'ddgiResetProbes';

/** Threads per workgroup. */
export const PROBE_RESET_WORKGROUP_SIZE = 64;

/**
 * Where each field of `DdgiProbeReset` sits, in 32-bit words (the WGSL
 * struct below; a unit test checks them under WGSL's layout rules).
 */
export const PROBE_RESET_UNIFORM_WORD = {
    origin: 0,
    spacing: 4,
    counts: 8,
    window: 12,
    layerBase: 16,
    probeBase: 17,
    resetOffsets: 18,
    resetStates: 19,
} as const;
export const PROBE_RESET_UNIFORM_WORDS = 20;
export const PROBE_RESET_UNIFORM_BYTES = PROBE_RESET_UNIFORM_WORDS * 4;

export const DDGI_PROBE_RESET_WGSL = /* wgsl */ `
// ---- ddgi/reset-probes (packages/ddgi/src/wgsl/probe-reset.ts) -------------------------------------

struct DdgiProbeReset {
    origin: vec4f,         // xyz: world position of the window's first probe (a fixed volume's first probe)
    spacing: vec4f,        // xyz: metres between probes
    counts: vec4i,         // xyz: probes per axis
    window: vec4i,         // xyz: lattice coordinates of the window's first probe (0 for a fixed volume; scroll.ts)
    layerBase: i32,        // the volume's first layer in the shared atlases
    probeBase: u32,        // its first point in the points debug view
    resetOffsets: u32,     // 1: every offset goes back to 0
    resetStates: u32,      // 1: every probe classification switched off comes back on, through state 3
}

const DDGI_PROBE_RESET_WORKGROUP_SIZE: u32 = ${PROBE_RESET_WORKGROUP_SIZE}u;
const DDGI_RESET_STATE_OFF_INSIDE_GEOMETRY: u32 = ${PROBE_STATE.offInsideGeometry}u;
const DDGI_RESET_STATE_OFF_NO_SURFACE_NEARBY: u32 = ${PROBE_STATE.offNoSurfaceNearby}u;
const DDGI_RESET_STATE_NOT_UPDATED_YET: u32 = ${PROBE_STATE.notUpdatedYet}u;

@group(1) @binding(0) var<uniform> ddgiProbeReset: DdgiProbeReset;
// The probe data as it was before this pass (a copy: see the file comment).
@group(1) @binding(1) var ddgiProbeDataBefore: texture_2d_array<f32>;
@group(1) @binding(2) var ddgiProbeDataOut: texture_storage_2d_array<rgba16float, write>;
@group(1) @binding(3) var ddgiPointsOut: texture_storage_2d<rgba32float, write>;

// value mod divisor in 0..divisor - 1, also for negative values.
fn ddgiProbeResetModulo(value: vec3i, divisor: vec3i) -> vec3i {
    return ((value % divisor) + divisor) % divisor;
}

@compute @workgroup_size(DDGI_PROBE_RESET_WORKGROUP_SIZE)
fn ${PROBE_RESET_ENTRY_POINT}(
    @builtin(workgroup_id) group: vec3u,
    @builtin(num_workgroups) groups: vec3u,
    @builtin(local_invocation_index) thread: u32,
) {
    let counts = ddgiProbeReset.counts.xyz;
    // Dispatches past 65,535 workgroups use two dimensions.
    let probeIndex = (group.x + group.y * groups.x) * DDGI_PROBE_RESET_WORKGROUP_SIZE + thread;
    if (probeIndex >= u32(counts.x * counts.y * counts.z)) {
        return;
    }
    // Slot (x, y, z) of index x + nx * (z + nz * y): its texel is (x, z) in the layer of its row y (volumes.ts).
    let index = i32(probeIndex);
    let slot = vec3i(index % counts.x, index / (counts.x * counts.z), (index / counts.x) % counts.z);
    let layer = ddgiProbeReset.layerBase + slot.y;
    let before = textureLoad(ddgiProbeDataBefore, slot.xz, layer, 0);

    let offset = select(before.xyz, vec3f(0.0), ddgiProbeReset.resetOffsets != 0u);
    var state = u32(round(before.w));
    let off = state == DDGI_RESET_STATE_OFF_INSIDE_GEOMETRY || state == DDGI_RESET_STATE_OFF_NO_SURFACE_NEARBY;
    if (ddgiProbeReset.resetStates != 0u && off) {
        state = DDGI_RESET_STATE_NOT_UPDATED_YET;
    }
    textureStore(ddgiProbeDataOut, slot.xz, layer, vec4f(offset, f32(state)));

    // The points debug view: the window probe this slot holds, at its grid point plus its offset.
    let windowProbe = ddgiProbeResetModulo(slot - ddgiProbeReset.window.xyz, counts);
    let position = ddgiProbeReset.origin.xyz + (vec3f(windowProbe) + offset) * ddgiProbeReset.spacing.xyz;
    let pointIndex = ddgiProbeReset.probeBase + probeIndex;
    let pointsWidth = textureDimensions(ddgiPointsOut).x;
    textureStore(ddgiPointsOut, vec2u(pointIndex % pointsWidth, pointIndex / pointsWidth), vec4f(position, f32(state)));
}
`;
