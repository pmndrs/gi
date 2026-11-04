/**
 * The `ddgi/update` kernel (docs/design/ddgi.md, "The update kernel"): trace,
 * shade and blend in one kernel. One workgroup per probe in this frame's
 * update list; rays never touch device memory.
 *
 * - **Phase A, trace and shade.** Thread t handles rays t, t + 256, ... The
 *   first 32 rays point in fixed directions and only record their distance
 *   and facing (for phase B). The rest are rotated by this frame's rotation,
 *   shaded with the platform's hit shading plus the current field for bounce
 *   light, and kept in workgroup memory.
 * - **Phase B, classification and relocation**, is T1.5b. It goes between the
 *   two barriers marked below; until then every probe keeps its offset and
 *   counts as active.
 * - **Phase C, blend.** One thread per atlas texel loops over the blended
 *   rays, then blends with the old value (hysteresis) and writes the result
 *   to the scratch buffer. The write kernel copies it into the atlases: this
 *   kernel samples the atlases, and WebGPU can't sample a texture and store
 *   into it in the same dispatch.
 *
 * Uses the platform's group 0 (scene copy, rays, hit shading) and its own
 * group 1. Storage buffers: tracer 3 + scene copy 2 + update list + scratch = 7.
 */

import { FIXED_RAYS, MAX_RAYS_PER_PROBE } from '../settings.ts';
import { PROBE_STATE, type ScratchLayout } from '../layout.ts';
import { DDGI_SAMPLE_WGSL } from './sample.ts';

/** The default `maxComputeInvocationsPerWorkgroup`; also the most rays a thread loop strides by. */
export const UPDATE_WORKGROUP_SIZE = 256;

/** `DdgiUniforms`, shared by both kernels; its layout is `UNIFORM_VEC4` in layout.ts. */
export const DDGI_UNIFORMS_WGSL = /* wgsl */ `
struct DdgiUniforms {
    volumeOrigin: vec4f,    // xyz: probe (0, 0, 0); w: normal bias
    volumeSpacing: vec4f,   // xyz: spacing; w: view bias
    volumeCounts: vec4f,    // xyz: probes per axis
    volumeTexels: vec4f,    // x: irradiance interior texels; y: distance interior texels
    rayRotation: mat3x3f,   // this frame's rotation of the blended rays (columns padded to vec4)
    blend: vec4f,           // hysteresis, irradianceChangeThreshold, maxBrightnessChange, distanceSharpness
    rays: vec4f,            // maxRayDistance, distance clamp, bounceEnergy, unused
    counts: vec4u,          // raysPerProbe, probes in the update list, frame, unused
}

// Probe (x, y, z) of index p = x + nx * (z + nz * y).
fn ddgiProbeCoordinates(index: u32, counts: vec3u) -> vec3i {
    let x = index % counts.x;
    let z = (index / counts.x) % counts.z;
    let y = index / (counts.x * counts.z);
    return vec3i(vec3u(x, y, z));
}

// The workgroup's slot in the update list; dispatches past 65,535 workgroups use two dimensions.
fn ddgiUpdateSlot(group: vec3u, groups: vec3u) -> u32 {
    return group.x + group.y * groups.x;
}
`;

export interface UpdateKernelShape {
    readonly irradianceTexels: number;
    readonly distanceTexels: number;
    readonly scratch: ScratchLayout;
}

export function updateKernelWgsl(shape: UpdateKernelShape): string {
    const { irradianceTexels, distanceTexels, scratch } = shape;
    return /* wgsl */ `
${DDGI_SAMPLE_WGSL}
${DDGI_UNIFORMS_WGSL}

// ---- DDGI update kernel (techniques/ddgi) ----------------------------------------------

const DDGI_WORKGROUP_SIZE: u32 = ${UPDATE_WORKGROUP_SIZE}u;
const DDGI_MAX_RAYS: u32 = ${MAX_RAYS_PER_PROBE}u;
// The first rays of every probe: fixed directions, never rotated or blended;
// they feed classification and relocation (phase B, T1.5b).
const DDGI_FIXED_RAYS: u32 = ${FIXED_RAYS}u;
const DDGI_IRRADIANCE_TEXELS: u32 = ${irradianceTexels}u;
const DDGI_DISTANCE_TEXELS: u32 = ${distanceTexels}u;
const DDGI_IRRADIANCE_TEXEL_COUNT: u32 = DDGI_IRRADIANCE_TEXELS * DDGI_IRRADIANCE_TEXELS;
const DDGI_DISTANCE_TEXEL_COUNT: u32 = DDGI_DISTANCE_TEXELS * DDGI_DISTANCE_TEXELS;
const DDGI_SCRATCH_STRIDE: u32 = ${scratch.strideWords}u;
const DDGI_SCRATCH_IRRADIANCE: u32 = ${scratch.irradianceOffset}u;
const DDGI_SCRATCH_DISTANCE: u32 = ${scratch.distanceOffset}u;
const DDGI_SCRATCH_PROBE_DATA: u32 = ${scratch.probeDataOffset}u;
const DDGI_STATE_OFF_INSIDE_GEOMETRY: u32 = ${PROBE_STATE.offInsideGeometry}u;
const DDGI_STATE_OFF_NO_SURFACE_NEARBY: u32 = ${PROBE_STATE.offNoSurfaceNearby}u;
// Light groups are M2: every group's lights reach every probe for now.
const DDGI_ALL_LIGHT_GROUPS: u32 = 0xfu;
// A back-face hit is stored at this share of its distance (negative in
// workgroup memory to mark it a back face). Pulling the mean distance short
// in that direction makes the Chebyshev test treat it as blocked: the
// "double-sided distance" leak fix (NOTES, 84x).
const DDGI_BACK_FACE_DISTANCE_SHARE: f32 = 0.2;
// A texel whose irradiance jumped by more than the change threshold gets its
// hysteresis multiplied by this, so a light switch shows up quickly.
const DDGI_BIG_CHANGE_HYSTERESIS_SCALE: f32 = 0.75;
// Smallest normal half-float: the firefly clamp's floor, so a texel at 0 can
// still grow (by at most maxBrightnessChange times this per update).
const DDGI_SMALLEST_NORMAL_HALF: f32 = 6.103515625e-5;
// 1 / golden ratio, for spherical Fibonacci points.
const DDGI_GOLDEN_FRACTION: f32 = 0.618033988749895;
const DDGI_TWO_PI: f32 = 6.28318530717959;

@group(1) @binding(0) var<uniform> ddgi: DdgiUniforms;
@group(1) @binding(1) var<storage, read> ddgiUpdateList: array<u32>;
@group(1) @binding(2) var<storage, read_write> ddgiScratch: array<u32>;
@group(1) @binding(3) var ddgiIrradianceAtlas: texture_2d_array<f32>;
@group(1) @binding(4) var ddgiDistanceAtlas: texture_2d_array<f32>;
@group(1) @binding(5) var ddgiProbeData: texture_2d_array<f32>;
// Linear, clamp to edge: tiles have borders, so filtering never needs to wrap.
@group(1) @binding(6) var ddgiAtlasSampler: sampler;

// Per ray: its direction (xyz) and signed hit distance (w, negative = back
// face). Stored decoded: the blend reads every ray once per texel, and
// decoding a packed direction each time made the update 27% slower (RTX 3060).
var<workgroup> ddgiRayDirectionDistance: array<vec4f, DDGI_MAX_RAYS>;
// Per ray: rgb radiance as half floats (rg, then b), the atlases' own
// precision. With the array above, 12 KB of the 16 KB default.
var<workgroup> ddgiRayRadiance: array<vec2u, DDGI_MAX_RAYS>;

// Point i of n spherical Fibonacci points (Keinert et al. 2015): equal-area
// rings in z, golden-angle steps around it. Evenly spread for any n.
fn ddgiSphericalFibonacci(index: u32, count: u32) -> vec3f {
    let i = f32(index);
    let phi = DDGI_TWO_PI * fract(i * DDGI_GOLDEN_FRACTION);
    let cosTheta = 1.0 - (2.0 * i + 1.0) / f32(count);
    let sinTheta = sqrt(saturate(1.0 - cosTheta * cosTheta));
    return vec3f(cos(phi) * sinTheta, sin(phi) * sinTheta, cosTheta);
}

// PCG hash (Jarzynski and Olano 2020), for decorrelated random numbers.
fn ddgiHash(value: u32) -> u32 {
    let state = value * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

fn ddgiRandom(a: u32, b: u32, c: u32) -> f32 {
    let bits = ddgiHash(a ^ ddgiHash(b ^ ddgiHash(c)));
    // The top 24 bits as a float in [0, 1).
    return f32(bits >> 8u) * (1.0 / 16777216.0);
}

// Rounds a value to a neighbouring half-float at random, in proportion to how
// close it is to each (stochastic rounding), so it is exact when packed.
// Why: with hysteresis 0.97 an update moves a texel by 3% of the gap to its
// estimate. Rounded to the nearest half, a small gap moves it by less than
// half a step, so it never moves, and the field stalls short of its answer:
// the furnace room settled 4.6% low that way (T1.5a), against 0.6% with
// this. Rounded at random, the expected value moves by exactly the blended
// amount.
fn ddgiStochasticHalf(value: f32, random: f32) -> f32 {
    // Half-floats have 10 fraction bits, so neighbours are 2^(e - 10) apart for
    // a value in [2^e, 2^(e+1)); below the smallest normal the step is fixed.
    // frexp gives e exactly (magnitude = fraction * 2^exponent, fraction in [0.5, 1)).
    let magnitude = max(abs(value), DDGI_SMALLEST_NORMAL_HALF);
    let step = exp2(f32(frexp(magnitude).exp - 1) - 10.0);
    let steps = value / step;
    let below = floor(steps);
    return (below + select(0.0, 1.0, random < steps - below)) * step;
}

fn ddgiPackHalves(a: f32, b: f32, seed: u32) -> u32 {
    return pack2x16float(vec2f(ddgiStochasticHalf(a, ddgiRandom(seed, 1u, ddgi.counts.z)), ddgiStochasticHalf(b, ddgiRandom(seed, 2u, ddgi.counts.z))));
}

// Direction at the centre of interior texel 'texel' of a tile 'texels' wide.
fn ddgiTexelDirection(texel: vec2u, texels: u32) -> vec3f {
    return ddgiOctDecode((vec2f(texel) + 0.5) / f32(texels) * 2.0 - 1.0);
}

// Atlas texel (with layer) of interior texel 'texel' of a probe's tile.
fn ddgiAtlasTexel(probe: vec3i, texel: vec2u, texels: u32) -> vec2i {
    return probe.xz * i32(texels + 2u * u32(DDGI_TILE_BORDER)) + i32(DDGI_TILE_BORDER) + vec2i(texel);
}

// Shades one blended ray: what radiance travels back along it to the probe.
fn ddgiShadeRay(ray: GiRay, hit: GiHit, random: vec2f) -> vec3f {
    let surface = giSurfaceAt(ray, hit);
    let direct = giDirectIrradiance(surface, DDGI_ALL_LIGHT_GROUPS, random);
    // Bounce light from the current field: the field already holds every
    // earlier bounce, so this gives unlimited bounces over updates.
    let bounce = ddgiSampleIrradiance(
        surface.position, surface.normal, ray.direction,
        ddgiIrradianceAtlas, ddgiAtlasSampler, ddgiDistanceAtlas, ddgiAtlasSampler, ddgiProbeData,
        ddgi.volumeOrigin, ddgi.volumeSpacing, ddgi.volumeCounts, ddgi.volumeTexels,
    );
    return surface.emission + surface.albedo * (1.0 / DDGI_PI) * (direct + ddgi.rays.z * bounce);
}

// Hysteresis for one probe: none for a probe's first update, so the field
// shows partial GI at once instead of fading in from black.
fn ddgiProbeHysteresis(state: u32) -> f32 {
    return select(ddgi.blend.x, 0.0, state == DDGI_STATE_NOT_UPDATED_YET);
}

fn ddgiBlendIrradianceTexel(slot: u32, probeIndex: u32, probe: vec3i, state: u32, texelIndex: u32, blendedRays: u32) {
    let texel = vec2u(texelIndex % DDGI_IRRADIANCE_TEXELS, texelIndex / DDGI_IRRADIANCE_TEXELS);
    let normal = ddgiTexelDirection(texel, DDGI_IRRADIANCE_TEXELS);
    var radianceSum = vec3f(0.0);
    var weightSum = 0.0;
    for (var blended = 0u; blended < blendedRays; blended++) {
        let ray = DDGI_FIXED_RAYS + blended;
        let weight = max(0.0, dot(normal, ddgiRayDirectionDistance[ray].xyz));
        let packed = ddgiRayRadiance[ray];
        radianceSum += weight * vec3f(unpack2x16float(packed.x), unpack2x16float(packed.y).x);
        weightSum += weight;
    }
    // With rays spread evenly over the sphere, the cosine-weighted mean of the
    // radiance over the hemisphere is E / pi; the field stores E.
    let estimate = DDGI_PI * radianceSum / max(weightSum, 1.0e-6);

    let old = textureLoad(ddgiIrradianceAtlas, ddgiAtlasTexel(probe, texel, DDGI_IRRADIANCE_TEXELS), probe.y, 0).rgb;
    var hysteresis = ddgiProbeHysteresis(state);
    if (any(abs(estimate - old) > ddgi.blend.y * old)) {
        hysteresis *= DDGI_BIG_CHANGE_HYSTERESIS_SCALE;
    }
    var blendedValue = mix(estimate, old, hysteresis);
    if (state != DDGI_STATE_NOT_UPDATED_YET) {
        // Firefly clamp: one update may change a channel by at most
        // maxBrightnessChange times its old value, so one lucky ray can't
        // flash the probe.
        let limit = ddgi.blend.z * max(old, vec3f(DDGI_SMALLEST_NORMAL_HALF));
        blendedValue = old + clamp(blendedValue - old, -limit, limit);
    }

    let base = slot * DDGI_SCRATCH_STRIDE + DDGI_SCRATCH_IRRADIANCE + 2u * texelIndex;
    let seed = probeIndex * DDGI_IRRADIANCE_TEXEL_COUNT + texelIndex;
    ddgiScratch[base] = ddgiPackHalves(blendedValue.r, blendedValue.g, seed);
    ddgiScratch[base + 1u] = ddgiPackHalves(blendedValue.b, 0.0, seed ^ 0x5bd1e995u);
}

fn ddgiBlendDistanceTexel(slot: u32, probeIndex: u32, probe: vec3i, state: u32, texelIndex: u32, blendedRays: u32) {
    let texel = vec2u(texelIndex % DDGI_DISTANCE_TEXELS, texelIndex / DDGI_DISTANCE_TEXELS);
    let normal = ddgiTexelDirection(texel, DDGI_DISTANCE_TEXELS);
    let clampDistance = ddgi.rays.y;
    var moments = vec2f(0.0);
    var weightSum = 0.0;
    for (var blended = 0u; blended < blendedRays; blended++) {
        let ray = DDGI_FIXED_RAYS + blended;
        let directionDistance = ddgiRayDirectionDistance[ray];
        // A sharp cosine power: distances are about one direction, not a hemisphere.
        let weight = pow(max(0.0, dot(normal, directionDistance.xyz)), ddgi.blend.w);
        let distance = min(abs(directionDistance.w), clampDistance);
        moments += weight * vec2f(distance, distance * distance);
        weightSum += weight;
    }
    let old = textureLoad(ddgiDistanceAtlas, ddgiAtlasTexel(probe, texel, DDGI_DISTANCE_TEXELS), probe.y, 0).xy;
    // A texel no ray came near this update keeps its old value.
    let estimate = select(old, moments / weightSum, weightSum > 1.0e-6);
    let blendedValue = mix(estimate, old, ddgiProbeHysteresis(state));

    let base = slot * DDGI_SCRATCH_STRIDE + DDGI_SCRATCH_DISTANCE + texelIndex;
    let seed = (probeIndex * DDGI_DISTANCE_TEXEL_COUNT + texelIndex) ^ 0x68e31da4u;
    ddgiScratch[base] = ddgiPackHalves(blendedValue.x, blendedValue.y, seed);
}

@compute @workgroup_size(DDGI_WORKGROUP_SIZE)
fn ddgiUpdate(
    @builtin(workgroup_id) group: vec3u,
    @builtin(num_workgroups) groups: vec3u,
    @builtin(local_invocation_index) thread: u32,
) {
    let slot = ddgiUpdateSlot(group, groups);
    if (slot >= ddgi.counts.y) {
        return;
    }
    let probeIndex = ddgiUpdateList[slot];
    let counts = vec3u(ddgi.volumeCounts.xyz);
    let probe = ddgiProbeCoordinates(probeIndex, counts);
    let probeData = textureLoad(ddgiProbeData, probe.xz, probe.y, 0);
    let state = u32(round(probeData.w));
    let offset = probeData.xyz;
    let origin = ddgi.volumeOrigin.xyz + (vec3f(probe) + offset) * ddgi.volumeSpacing.xyz;

    let raysPerProbe = ddgi.counts.x;
    let blendedRays = raysPerProbe - DDGI_FIXED_RAYS;
    let isOff = state == DDGI_STATE_OFF_INSIDE_GEOMETRY || state == DDGI_STATE_OFF_NO_SURFACE_NEARBY;
    // Off probes trace only their fixed rays, so they can come back on (T1.5b).
    let tracedRays = select(raysPerProbe, DDGI_FIXED_RAYS, isOff);

    // ---- Phase A: trace and shade ----
    for (var rayIndex = thread; rayIndex < tracedRays; rayIndex += DDGI_WORKGROUP_SIZE) {
        let isFixed = rayIndex < DDGI_FIXED_RAYS;
        var direction: vec3f;
        if (isFixed) {
            direction = ddgiSphericalFibonacci(rayIndex, DDGI_FIXED_RAYS);
        } else {
            direction = ddgi.rayRotation * ddgiSphericalFibonacci(rayIndex - DDGI_FIXED_RAYS, blendedRays);
        }
        let ray = GiRay(origin, 0.0, direction, ddgi.rays.x);
        let hit = giTraceClosest(ray);

        var radiance = vec3f(0.0);
        var signedDistance: f32;
        if (hit.t < 0.0) {
            // A miss sees the sky, at the clamp distance.
            signedDistance = ddgi.rays.y;
            if (!isFixed) {
                radiance = giSkyRadiance(direction);
            }
        } else if (!hit.frontFace) {
            // Back faces are black and short (see DDGI_BACK_FACE_DISTANCE_SHARE).
            signedDistance = -DDGI_BACK_FACE_DISTANCE_SHARE * hit.t;
        } else {
            signedDistance = hit.t;
            // Fixed rays need only the distance and facing; they are never blended.
            if (!isFixed) {
                let random = vec2f(ddgiRandom(probeIndex, rayIndex, ddgi.counts.z), ddgiRandom(probeIndex ^ 0x9e3779b9u, rayIndex, ddgi.counts.z));
                radiance = ddgiShadeRay(ray, hit, random);
            }
        }
        ddgiRayDirectionDistance[rayIndex] = vec4f(direction, signedDistance);
        ddgiRayRadiance[rayIndex] = vec2u(pack2x16float(radiance.rg), pack2x16float(vec2f(radiance.b, 0.0)));
    }
    workgroupBarrier();

    // ---- Phase B: classification and relocation (T1.5b) ----
    // Reductions over the fixed rays go here. Until then the probe keeps its
    // offset, and a blended update makes it active.
    let newOffset = offset;
    var newState = state;
    if (!isOff) {
        newState = DDGI_STATE_ACTIVE;
    }

    // ---- Phase C: blend (probes that traced all their rays) ----
    if (!isOff) {
        for (var texelIndex = thread; texelIndex < DDGI_IRRADIANCE_TEXEL_COUNT + DDGI_DISTANCE_TEXEL_COUNT; texelIndex += DDGI_WORKGROUP_SIZE) {
            if (texelIndex < DDGI_IRRADIANCE_TEXEL_COUNT) {
                ddgiBlendIrradianceTexel(slot, probeIndex, probe, state, texelIndex, blendedRays);
            } else {
                ddgiBlendDistanceTexel(slot, probeIndex, probe, state, texelIndex - DDGI_IRRADIANCE_TEXEL_COUNT, blendedRays);
            }
        }
    }
    if (thread == 0u) {
        // Offsets and states are small exact values in half floats; no rounding noise wanted.
        let base = slot * DDGI_SCRATCH_STRIDE + DDGI_SCRATCH_PROBE_DATA;
        ddgiScratch[base] = pack2x16float(newOffset.xy);
        ddgiScratch[base + 1u] = pack2x16float(vec2f(newOffset.z, f32(newState)));
    }
}
`;
}
