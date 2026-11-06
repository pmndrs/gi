/**
 * WGSL shared by DDGI's kernels (`ddgi/trace`, `ddgi/blend`, `ddgi/write`,
 * `ddgi/census`): the sampling function and its constants, the uniform
 * block, probe indexing, ray directions and random numbers. Every kernel
 * includes it once; none of it binds anything.
 */

import { FIXED_RAYS, MAX_RAYS_PER_PROBE } from '../settings.ts';
import { DDGI_SAMPLE_WGSL } from './sample.ts';

/** `DdgiUniforms`; its byte layout is `UNIFORM_WORD` in layout.ts (checked by a unit test). */
export const DDGI_COMMON_WGSL = /* wgsl */ `
${DDGI_SAMPLE_WGSL}
// ---- DDGI common (techniques/ddgi) -----------------------------------------------------

struct DdgiUniforms {
    // The four vec4f parameters of ddgiSampleIrradiance, exactly as a renderer
    // passes them; read them through ddgiUniformVolume() (named fields).
    volumeOrigin: vec4f,
    volumeSpacing: vec4f,
    volumeCounts: vec4f,
    volumeTexels: vec4f,
    // This frame's rotation of the blended rays.
    rayRotation: mat3x3f,
    // Blending (phase C).
    hysteresis: f32,
    irradianceChangeThreshold: f32,
    maxBrightnessChange: f32,
    distanceSharpness: f32,
    // Rays.
    maxRayDistance: f32,
    distanceClamp: f32,       // distances in the distance atlas never exceed this
    bounceEnergy: f32,
    // Classification and relocation (phase B).
    minFrontFaceDistance: f32,
    relocation: u32,          // 0 or 1
    classification: u32,      // 0 or 1
    // This frame.
    raysPerProbe: u32,
    updateCount: u32,         // probes in the update list
    frame: u32,
    // Hysteresis that knows the time between a probe's updates (phase C;
    // docs/design/ddgi.md). The clock is the technique's own, in seconds.
    adaptiveHysteresis: u32,  // 0: every update keeps 'hysteresis'
    clockSeconds: f32,
    inverseMemorySeconds: f32, // 1 / tau; 0 never forgets
    maxHistorySamples: f32,   // n_max
    updatesPerSample: f32,    // a: n is at most (blended updates) / a during the ramp; 1 = a plain mean
    rampUpdates: f32,         // K: the ramp lasts a probe's first K blended updates
    // Change-driven history (docs/design/m2-lights.md §4.1, §4.2): ddgi/mark
    // and the cut in ddgi/blend.
    changeDrivenHistory: u32, // 0: no cut, the M1 rule alone
    changeTolerance: f32,     // epsilon: what a changed probe's history is cut for
    changeUpdates: f32,       // K: updates a changed probe is expected to get soon
    lightChangeScale: f32,    // the share of a light's change at a probe the probe is taken to see
}

const DDGI_MAX_RAYS: u32 = ${MAX_RAYS_PER_PROBE}u;
// The first rays of every probe: fixed directions, never rotated or blended;
// they feed classification and relocation (phase B in ddgi/blend).
const DDGI_FIXED_RAYS: u32 = ${FIXED_RAYS}u;
// 1 / golden ratio, for spherical Fibonacci points.
const DDGI_GOLDEN_FRACTION: f32 = 0.618033988749895;
const DDGI_TWO_PI: f32 = 6.28318530717959;

fn ddgiUniformVolume() -> DdgiVolume {
    return ddgiVolume(ddgi.volumeOrigin, ddgi.volumeSpacing, ddgi.volumeCounts, ddgi.volumeTexels);
}

fn ddgiProbeCount() -> vec3u {
    return vec3u(ddgiUniformVolume().counts);
}

// Probe (x, y, z) of index p = x + nx * (z + nz * y).
fn ddgiProbeCoordinates(index: u32, counts: vec3u) -> vec3i {
    let x = index % counts.x;
    let z = (index / counts.x) % counts.z;
    let y = index / (counts.x * counts.z);
    return vec3i(vec3u(x, y, z));
}

// A probe's world position: its grid point plus its relocation offset (stored in spacings).
fn ddgiProbePosition(probe: vec3i, offset: vec3f) -> vec3f {
    let volume = ddgiUniformVolume();
    return volume.origin + (vec3f(probe) + offset) * volume.spacing;
}

// Off probes trace only their fixed rays, so they can come back on when the scene changes.
fn ddgiIsOff(state: u32) -> bool {
    return state == DDGI_STATE_OFF_INSIDE_GEOMETRY || state == DDGI_STATE_OFF_NO_SURFACE_NEARBY;
}

fn ddgiTracedRays(state: u32) -> u32 {
    return select(ddgi.raysPerProbe, DDGI_FIXED_RAYS, ddgiIsOff(state));
}

// Point i of n spherical Fibonacci points (Keinert et al. 2015): equal-area
// rings in z, golden-angle steps around it. Evenly spread for any n.
fn ddgiSphericalFibonacci(index: u32, count: u32) -> vec3f {
    let i = f32(index);
    let phi = DDGI_TWO_PI * fract(i * DDGI_GOLDEN_FRACTION);
    let cosTheta = 1.0 - (2.0 * i + 1.0) / f32(count);
    let sinTheta = sqrt(saturate(1.0 - cosTheta * cosTheta));
    return vec3f(cos(phi) * sinTheta, sin(phi) * sinTheta, cosTheta);
}

// Ray 'index' of a probe: the fixed rays never rotate; the blended ones turn
// with this frame's rotation, so successive updates sample new directions.
fn ddgiRayDirection(index: u32) -> vec3f {
    if (index < DDGI_FIXED_RAYS) {
        return ddgiSphericalFibonacci(index, DDGI_FIXED_RAYS);
    }
    return ddgi.rayRotation * ddgiSphericalFibonacci(index - DDGI_FIXED_RAYS, ddgi.raysPerProbe - DDGI_FIXED_RAYS);
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
`;
