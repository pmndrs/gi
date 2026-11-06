/**
 * The `ddgi/trace` kernel (docs/design/ddgi.md, "Revision after T1.5a"):
 * one thread per ray of every probe in this frame's update list. It traces
 * inline with `giTraceClosest`, shades the hit, and writes each ray's
 * radiance and signed distance to the ray buffer for `ddgi/blend`.
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
 * Uses the platform's group 0 (scene copy, rays, hit shading) and its own
 * group 1. Storage buffers: tracer 3 + scene copy 2 + update list + rays = 7.
 */

import { DDGI_COMMON_WGSL } from './common.ts';

export const TRACE_WORKGROUP_SIZE = 64;

export const DDGI_TRACE_WGSL = /* wgsl */ `
${DDGI_COMMON_WGSL}

// ---- DDGI trace kernel (techniques/ddgi) -----------------------------------------------

const DDGI_TRACE_WORKGROUP_SIZE: u32 = ${TRACE_WORKGROUP_SIZE}u;
// Light groups are M2: every group's lights reach every probe for now.
const DDGI_ALL_LIGHT_GROUPS: u32 = 0xfu;
// A back-face hit is stored at this share of its distance, negative to mark
// it a back face. Pulling the mean distance short in that direction makes the
// Chebyshev test treat it as blocked: the "double-sided distance" leak fix
// (NOTES, 84x).
const DDGI_BACK_FACE_DISTANCE_SHARE: f32 = 0.2;
// The distance stored for a miss: farther than any hit worth telling apart,
// and still a finite half float (the largest is 65,504). The blend clamps
// distances anyway; phase B takes a miss as the most open direction.
const DDGI_MISS_DISTANCE: f32 = 60000.0;

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
            let random = vec2f(ddgiRandom(probeIndex, rayIndex, ddgi.frame), ddgiRandom(probeIndex ^ 0x9e3779b9u, rayIndex, ddgi.frame));
            radiance = ddgiShadeRay(ray, hit, random);
        }
    }
    ddgiRays[slot * ddgi.raysPerProbe + rayIndex] = vec2u(pack2x16float(radiance.rg), pack2x16float(vec2f(radiance.b, signedDistance)));
}
`;
