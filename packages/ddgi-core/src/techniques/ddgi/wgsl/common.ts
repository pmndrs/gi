/**
 * WGSL shared by DDGI's kernels (`ddgi/trace`, `ddgi/blend`, `ddgi/write`,
 * `ddgi/census`, `ddgi/mark`, `ddgi/schedule-*`): the sampling function and
 * its constants, the uniform block, probe indexing, the camera weight, ray
 * directions and random numbers. Every kernel
 * includes it once; none of it binds anything.
 */

import { FIXED_RAYS, MAX_RAYS_PER_PROBE, OUT_OF_VIEW_WEIGHT } from '../settings.ts';
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
    // The GPU scheduler (docs/design/m2-lights.md §4.3, §4.6, §6.4).
    feedback: u32,            // 0: the feedback rule is off
    priorityCount: u32,       // P: at most this many probes of the list go to probes owed updates
    scrambleStride: u32,      // the round robin's order: order[i] = (i * stride) mod probe count
    farOwedUpdates: f32,      // updates far probes owe after a global change this frame (K when the rule is off)
    cameraFalloffDistance: f32, // the camera weight's falloff: 4 spacings
    cameraPosition: vec4f,    // xyz: the camera; w: how far outside the frustum still counts as in view
    // The view's planes, xyz pointing inwards and normalised, w the offset:
    // a point p is inside when dot(xyz, p) + w >= 0 for all six.
    frustumLeft: vec4f,
    frustumRight: vec4f,
    frustumBottom: vec4f,
    frustumTop: vec4f,
    frustumNear: vec4f,
    frustumFar: vec4f,
    // The share of a probe's light taken to come back through the field (rho), for the cut (ddgi/blend).
    changeBounceShare: f32,
    // Key probes and propagation (T2.4c, docs/design/m2-lights.md §4.5).
    keyProbes: u32,           // 1: changes marked this frame take the key-probe path (the setting, on a volume too big for every probe's K updates)
    keyProbeBoost: f32,       // a marked key's urgency is its change times this
    keyResidualShare: f32,    // a fine probe waiting for its keys: urgency share, and the least of its change its own update is cut for
    keyOwedUpdates: f32,      // updates a key away from the view's core owes
    fineOwedUpdates: f32,     // updates a fine probe waiting for its keys owes
    propagation: u32,         // 1: keys taking in a change pass it to their fine neighbours (ddgi/blend, ddgi/propagate)
    holdFineProbes: u32,      // the test mode (settings.holdFineProbes): fine probes waiting for their keys are not blended
    // The sun layer (T2.6b): the sun's colour x intensity (rgb; w unused), what
    // one unit of its white-reference layer stands for. ddgi/blend weighs that
    // layer by it in the probe's mean, so the mean is the probe's real light.
    sunLight: vec4f,
    // Several volumes (m3-moving.md §4.1, volumes.ts): each is updated with
    // uniforms of its own, and its probes' points are listed from here on in
    // the points debug view (the volumes' probes one after another).
    probeBase: u32,
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

// The atlas layer of one of this volume's probes: every volume shares the
// atlases, this one from its layer base on (volumes.ts).
fn ddgiProbeLayer(probe: vec3i) -> i32 {
    return ddgiAtlasLayer(ddgiUniformVolume(), probe);
}

// Tiles across one atlas layer, the same for every volume. Irradiance layer L
// (light groups) has its tiles L times this many tiles to the right.
fn ddgiAtlasTilesAcross() -> i32 {
    return i32(ddgiUniformVolume().atlasTiles.x);
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

// ---- The camera weight (docs/design/m2-lights.md §4.3) ----
// SPEC's response target is judged on the image, and nearby, visible probes
// are what the image shows, so the scheduler serves them first (Lumen's
// priority does the same). Used by ddgi/schedule-* and by ddgi/mark (which
// probes count as near after a global change).

// A probe outside the view counts this much less.
const DDGI_OUT_OF_VIEW_WEIGHT: f32 = ${OUT_OF_VIEW_WEIGHT};

// Inside the view's six planes, or within the margin (cameraPosition.w) of
// them: a probe up to a cell's diagonal outside still lights visible points.
fn ddgiInView(position: vec3f) -> bool {
    let margin = -ddgi.cameraPosition.w;
    let point = vec4f(position, 1.0);
    return dot(ddgi.frustumLeft, point) >= margin && dot(ddgi.frustumRight, point) >= margin
        && dot(ddgi.frustumBottom, point) >= margin && dot(ddgi.frustumTop, point) >= margin
        && dot(ddgi.frustumNear, point) >= margin && dot(ddgi.frustumFar, point) >= margin;
}

// 1 / (1 + (d / (4 spacings))^2), times DDGI_OUT_OF_VIEW_WEIGHT outside the
// view. Taken at the probe's grid point: relocation moves it less than half a
// spacing, which the falloff hardly notices.
fn ddgiCameraWeight(probe: vec3i) -> f32 {
    let position = ddgiProbePosition(probe, vec3f(0.0));
    let scaled = length(position - ddgi.cameraPosition.xyz) / ddgi.cameraFalloffDistance;
    let weight = 1.0 / (1.0 + scaled * scaled);
    return select(weight * DDGI_OUT_OF_VIEW_WEIGHT, weight, ddgiInView(position));
}

// ---- Key probes (T2.4c, docs/design/m2-lights.md §4.5) ----
// A key probe's grid index is even or the last one on every axis: about one
// probe in 8, the coarse level of a virtual two-level grid at twice the
// spacing. (isKeyProbe in layout.ts is the CPU twin.)

fn ddgiIsKeyAxis(index: i32, count: i32) -> bool {
    return index % 2 == 0 || index == count - 1;
}

fn ddgiIsKeyProbe(probe: vec3i) -> bool {
    let counts = vec3i(ddgiProbeCount());
    return ddgiIsKeyAxis(probe.x, counts.x) && ddgiIsKeyAxis(probe.y, counts.y) && ddgiIsKeyAxis(probe.z, counts.z);
}

// Key neighbour j (0..7) of a fine probe as an offset in grid steps, and
// whether j is one of its neighbours (keyNeighbourOffset in layout.ts): on an
// axis where the probe's index is a key index the offset is 0 and that
// axis's bit of j must be 0; otherwise the bit chooses the key one step
// below (0) or above (1). Every key neighbour has the same trilinear weight:
// on the key lattice a fine probe sits halfway between its keys on each axis
// where it isn't a key.
struct DdgiKeyNeighbour {
    offset: vec3i,
    exists: bool,
}

fn ddgiKeyNeighbourOffset(neighbour: u32, probe: vec3i) -> DdgiKeyNeighbour {
    let counts = vec3i(ddgiProbeCount());
    var offset = vec3i(0);
    var exists = true;
    for (var axis = 0u; axis < 3u; axis++) {
        let above = ((neighbour >> axis) & 1u) != 0u;
        if (ddgiIsKeyAxis(probe[axis], counts[axis])) {
            exists = exists && !above;
        } else {
            offset[axis] = select(-1, 1, above);
        }
    }
    return DdgiKeyNeighbour(offset, exists);
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
