/**
 * The `ddgi/trace` kernel (docs/design/ddgi.md, "Revision after T1.5a"):
 * one thread per ray of every probe in this frame's update list. It traces
 * inline with `giTraceClosest`, shades the hit, and writes each ray's
 * radiance and signed distance to the ray results for `ddgi/blend`.
 *
 * Why its own kernel: tracing, shading and blending in one 256-thread
 * workgroup was register-bound (RTX 3060 2.48 ms against 1.22 ms split, at
 * 512 probes × 256 rays; DECISIONS 2026-11-05).
 *
 * - Fixed rays (the first 32) need only the hit distance and whether it was a
 *   front face (for phase B); they are not shaded.
 * - Blended rays: a miss sees the sky; a back face is black with a shortened
 *   distance; a front face gets emission plus albedo / π × (direct light +
 *   bounce light from the current field).
 *
 * Two versions, chosen by `DDGITechniqueOptions.directLight`:
 *
 * - **deferred** (the default since T2.2, {@link deferredTraceKernelWgsl}): a
 *   front face's direct light is left to the platform's `gi/direct`. The
 *   kernel stores emission + albedo / π × bounce and, if a light can reach
 *   the hit, appends its surface record to the ray scratch's work list;
 *   `gi/direct` adds the direct light before `ddgi/blend` reads the ray
 *   (docs/design/m2-lights.md §2). It traces one round of at most
 *   {@link DIRECT_ROUND_PROBES} probes per dispatch, so the work list needs
 *   room for one round only.
 * - **inline** (M1's, {@link DDGI_TRACE_WGSL}): the kernel traces the shadow
 *   rays itself with `giDirectIrradiance`. Kept for comparison. Since T2.2 it
 *   draws a ray's light-choice and soft-shadow numbers with `giDirectRandom`
 *   (the ray's index and the frame), as `gi/direct` does, so the two paths'
 *   fields can be compared texel by texel (`ddgi-direct`).
 *
 * Uses the platform's group 0 (scene copy, rays, hit shading) and its own
 * group 1. Storage buffers: tracer 3 + scene copy 2 + update list + the ray
 * scratch (deferred) or its ray results (inline) = 7.
 */

import { DDGI_COMMON_WGSL } from './common.ts';

export const TRACE_WORKGROUP_SIZE = 64;

/**
 * The light groups whose lights reach DDGI's probes: every group, until
 * light groups (T2.5) give each its own layer. The one definition: the inline
 * kernel's WGSL and the deferred path's `gi/direct` both take it from here.
 */
export const DDGI_LIGHT_GROUP_MASK = 0xf;

/**
 * Probes per deferred trace round. The work list needs room for one round's
 * blended rays (1,024 × 224 × 32 bytes = 7.3 MB at the default 256 rays a
 * probe), where a whole 4,096-probe warm-up batch would need 29 MB
 * (docs/design/m2-lights.md §2, "Memory"). Batches at Budget v2's calibrated
 * sizes on Sponza are one round.
 */
export const DIRECT_ROUND_PROBES = 1024;

/** Bytes of `DdgiTraceRound`, the deferred kernel's per-round uniform. */
export const TRACE_ROUND_BYTES = 16;

/** Constants both versions share. */
const TRACE_CONSTANTS_WGSL = /* wgsl */ `
const DDGI_TRACE_WORKGROUP_SIZE: u32 = ${TRACE_WORKGROUP_SIZE}u;
// The light groups that reach the probes (DDGI_LIGHT_GROUP_MASK in trace.ts).
const DDGI_ALL_LIGHT_GROUPS: u32 = ${DDGI_LIGHT_GROUP_MASK}u;
// A back-face hit is stored at this share of its distance, negative to mark
// it a back face. Pulling the mean distance short in that direction makes the
// Chebyshev test treat it as blocked: the "double-sided distance" leak fix
// (NOTES, 84x).
const DDGI_BACK_FACE_DISTANCE_SHARE: f32 = 0.2;
// The distance stored for a miss: farther than any hit worth telling apart,
// and still a finite half float (the largest is 65,504). The blend clamps
// distances anyway; phase B takes a miss as the most open direction.
const DDGI_MISS_DISTANCE: f32 = 60000.0;
`;

export const DDGI_TRACE_WGSL = /* wgsl */ `
${DDGI_COMMON_WGSL}

// ---- DDGI trace kernel, direct light inline (techniques/ddgi) ---------------------------
${TRACE_CONSTANTS_WGSL}

@group(1) @binding(0) var<uniform> ddgi: DdgiUniforms;
@group(1) @binding(1) var<storage, read> ddgiUpdateList: array<u32>;
@group(1) @binding(2) var<storage, read_write> ddgiRays: array<vec2u>;
@group(1) @binding(3) var ddgiIrradianceAtlas: texture_2d_array<f32>;
@group(1) @binding(4) var ddgiDistanceAtlas: texture_2d_array<f32>;
@group(1) @binding(5) var ddgiProbeData: texture_2d_array<f32>;
// Linear, clamp to edge: tiles have borders, so filtering never needs to wrap.
@group(1) @binding(6) var ddgiAtlasSampler: sampler;

// What radiance travels back along a blended ray that hit a front face:
// emission + albedo / pi × (direct light + bounce light).
//
// The bounce is read before the direct light on purpose. The direct light
// traces a shadow ray (an any-hit traversal loop, the most expensive part of
// this kernel in sunlit scenes), and whatever is still needed afterwards stays
// in registers through that loop. Folding the bounce in first leaves only the
// running sum and albedo / pi alive there. Measured (T1.9a, RTX 3060, the same
// probes): 512 probes on Sponza 17-30% faster, two-room 11%, Cornell 20%.
fn ddgiShadeRay(ray: GiRay, hit: GiHit, random: vec2f) -> vec3f {
    let surface = giSurfaceAt(ray, hit);
    // Bounce light from the current field: the field already holds every
    // earlier bounce, so this gives unlimited bounces over updates.
    let bounce = ddgiSampleIrradiance(
        surface.position, surface.normal, ray.direction,
        ddgiIrradianceAtlas, ddgiAtlasSampler, ddgiDistanceAtlas, ddgiAtlasSampler, ddgiProbeData,
        ddgi.volumeOrigin, ddgi.volumeSpacing, ddgi.volumeCounts, ddgi.volumeTexels,
    );
    let albedoOverPi = surface.albedo * (1.0 / DDGI_PI);
    let emittedAndBounced = surface.emission + albedoOverPi * (ddgi.bounceEnergy * bounce);
    let direct = giDirectIrradiance(surface, DDGI_ALL_LIGHT_GROUPS, random);
    return emittedAndBounced + albedoOverPi * direct;
}

// Workgroup (x, y, z): x walks a probe's rays 64 at a time; y and z give the
// update-list slot (z only past 65,535 probes).
@compute @workgroup_size(DDGI_TRACE_WORKGROUP_SIZE)
fn ddgiTrace(
    @builtin(workgroup_id) group: vec3u,
    @builtin(num_workgroups) groups: vec3u,
    @builtin(local_invocation_index) thread: u32,
) {
    let slot = group.y + group.z * groups.y;
    if (slot >= ddgi.updateCount) {
        return;
    }
    let rayIndex = group.x * DDGI_TRACE_WORKGROUP_SIZE + thread;
    let probeIndex = ddgiUpdateList[slot];
    let probe = ddgiProbeCoordinates(probeIndex, ddgiProbeCount());
    let probeData = ddgiReadProbeData(ddgiProbeData, probe);
    if (rayIndex >= ddgiTracedRays(probeData.state)) {
        return;
    }
    let rayRecord = slot * ddgi.raysPerProbe + rayIndex;

    let isFixed = rayIndex < DDGI_FIXED_RAYS;
    let direction = ddgiRayDirection(rayIndex);
    let ray = GiRay(ddgiProbePosition(probe, probeData.offset), 0.0, direction, ddgi.maxRayDistance);
    let hit = giTraceClosest(ray);

    var radiance = vec3f(0.0);
    var signedDistance: f32;
    if (hit.t < 0.0) {
        signedDistance = DDGI_MISS_DISTANCE;
        if (!isFixed) {
            radiance = giSkyRadiance(direction);
        }
    } else if (!hit.frontFace) {
        // Back faces are black and short (see DDGI_BACK_FACE_DISTANCE_SHARE).
        signedDistance = -DDGI_BACK_FACE_DISTANCE_SHARE * hit.t;
    } else {
        signedDistance = hit.t;
        if (!isFixed) {
            // gi/direct's random numbers for this ray (giDirectRandom), so this kernel and the
            // deferred one light the same ray alike and their fields can be compared texel by texel.
            radiance = ddgiShadeRay(ray, hit, giDirectRandom(rayRecord, ddgi.frame));
        }
    }
    ddgiRays[rayRecord] = vec2u(pack2x16float(radiance.rg), pack2x16float(vec2f(radiance.b, signedDistance)));
}
`;

/**
 * The deferred version (see the file comment), around the platform's side
 * of the ray scratch (`HitShading.direct.traceWgsl('ddgiRayScratch', ...)`).
 *
 * The body is shaped for the work list's append: no thread returns before
 * `giDeferDirect`, which every thread calls in uniform control flow. The one
 * early return left (a workgroup past its round) depends only on the
 * workgroup and uniforms, so it is uniform for the whole workgroup.
 */
export function deferredTraceKernelWgsl(directScratchWgsl: string): string {
    return /* wgsl */ `
${DDGI_COMMON_WGSL}

// ---- DDGI trace kernel, direct light deferred to gi/direct (techniques/ddgi) ------------
${TRACE_CONSTANTS_WGSL}

// This dispatch's round: update-list slots firstSlot .. firstSlot + slotCount.
struct DdgiTraceRound {
    firstSlot: u32,
    slotCount: u32,
    reserved0: u32,
    reserved1: u32,
}

@group(1) @binding(0) var<uniform> ddgi: DdgiUniforms;
@group(1) @binding(1) var<storage, read> ddgiUpdateList: array<u32>;
@group(1) @binding(2) var<storage, read_write> ddgiRayScratch: GiRayScratch;
@group(1) @binding(3) var ddgiIrradianceAtlas: texture_2d_array<f32>;
@group(1) @binding(4) var ddgiDistanceAtlas: texture_2d_array<f32>;
@group(1) @binding(5) var ddgiProbeData: texture_2d_array<f32>;
// Linear, clamp to edge: tiles have borders, so filtering never needs to wrap.
@group(1) @binding(6) var ddgiAtlasSampler: sampler;
// One 256-byte slot per round, chosen with a dynamic offset.
@group(1) @binding(7) var<uniform> ddgiTraceRound: DdgiTraceRound;

// The platform's side of the ray scratch: giStoreRayResult, giNeedsDirect, giSurfaceRecord, giDeferDirect.
${directScratchWgsl}

// Emission + albedo / pi × bounce light from the current field. The direct
// part, albedo / pi × direct light, is added by gi/direct.
fn ddgiEmittedAndBounced(ray: GiRay, surface: GiSurface) -> vec3f {
    // Bounce light from the current field: the field already holds every
    // earlier bounce, so this gives unlimited bounces over updates.
    let bounce = ddgiSampleIrradiance(
        surface.position, surface.normal, ray.direction,
        ddgiIrradianceAtlas, ddgiAtlasSampler, ddgiDistanceAtlas, ddgiAtlasSampler, ddgiProbeData,
        ddgi.volumeOrigin, ddgi.volumeSpacing, ddgi.volumeCounts, ddgi.volumeTexels,
    );
    return surface.emission + surface.albedo * (1.0 / DDGI_PI) * (ddgi.bounceEnergy * bounce);
}

// Workgroup (x, y): x walks a probe's rays 64 at a time; y is the probe's
// place in this round (a round has at most DIRECT_ROUND_PROBES probes).
@compute @workgroup_size(DDGI_TRACE_WORKGROUP_SIZE)
fn ddgiTrace(
    @builtin(workgroup_id) group: vec3u,
    @builtin(local_invocation_index) thread: u32,
) {
    let slot = ddgiTraceRound.firstSlot + group.y;
    if (group.y >= ddgiTraceRound.slotCount || slot >= ddgi.updateCount) {
        return;
    }
    let rayIndex = group.x * DDGI_TRACE_WORKGROUP_SIZE + thread;
    let probeIndex = ddgiUpdateList[slot];
    let probe = ddgiProbeCoordinates(probeIndex, ddgiProbeCount());
    let probeData = ddgiReadProbeData(ddgiProbeData, probe);
    let rayRecord = slot * ddgi.raysPerProbe + rayIndex;

    var deferred = false;
    var record = GiSurfaceRecord();
    if (rayIndex < ddgiTracedRays(probeData.state)) {
        let isFixed = rayIndex < DDGI_FIXED_RAYS;
        let direction = ddgiRayDirection(rayIndex);
        let ray = GiRay(ddgiProbePosition(probe, probeData.offset), 0.0, direction, ddgi.maxRayDistance);
        let hit = giTraceClosest(ray);

        var radiance = vec3f(0.0);
        var signedDistance: f32;
        if (hit.t < 0.0) {
            signedDistance = DDGI_MISS_DISTANCE;
            if (!isFixed) {
                radiance = giSkyRadiance(direction);
            }
        } else if (!hit.frontFace) {
            // Back faces are black and short (see DDGI_BACK_FACE_DISTANCE_SHARE).
            signedDistance = -DDGI_BACK_FACE_DISTANCE_SHARE * hit.t;
        } else {
            signedDistance = hit.t;
            if (!isFixed) {
                let surface = giSurfaceAt(ray, hit);
                radiance = ddgiEmittedAndBounced(ray, surface);
                // Only hits a light can reach go on the list (half of Sponza's face away from the sun).
                deferred = giNeedsDirect(surface);
                if (deferred) {
                    record = giSurfaceRecord(surface);
                }
            }
        }
        giStoreRayResult(rayRecord, radiance, signedDistance);
    }
    giDeferDirect(deferred, record, rayRecord);
}
`;
}
