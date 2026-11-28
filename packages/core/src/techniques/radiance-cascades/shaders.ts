/**
 * Surface-launched Split RC for a bounded volume. Each ray deposits one
 * observation per interval it can see. Unknown (a closer hit) never writes
 * black into a farther interval. Each cascade doubles directional resolution
 * along each cube face and approximately halves spatial resolution per axis.
 */
export const nativeRcShader = /* wgsl */ `
struct Params {
    lower: vec4f,
    inverseSpan: vec4f,
    screen: vec4u,          // width, height, pixel stride, ray count
    distances: vec4f,      // first endpoint, final endpoint, history retention, origin bias
    grid: array<vec4u, 3>, // dimensions and first directional bin
}
struct Ray { originMin: vec4f, directionMax: vec4f }
struct Batch {
    r: atomic<u32>, g: atomic<u32>, b: atomic<u32>,
    transparent: atomic<u32>, count: atomic<u32>,
}
struct History { value: vec4f } // w = 1 + mean transmittance, zero means unobserved
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage,read_write> clock: array<u32>;
@group(0) @binding(2) var<storage,read_write> rays: array<Ray>;
@group(0) @binding(3) var<storage,read_write> hits: array<vec4f>;
@group(0) @binding(4) var<storage,read_write> batch: array<Batch>;
@group(0) @binding(5) var<storage,read_write> history: array<History>;
@group(0) @binding(6) var<storage,read_write> seen: array<u32>;
@group(0) @binding(7) var<storage,read_write> merged: array<vec4f>;
@group(0) @binding(8) var<storage,read_write> ambientCube: array<vec4f>;
@group(0) @binding(9) var positionTexture: texture_2d<f32>;
@group(0) @binding(10) var normalTexture: texture_2d<f32>;
@group(0) @binding(11) var irradianceTexture: texture_storage_2d<rgba16float,write>;
const PI = 3.14159265359;
const FIXED = 2048.0;
const MAX_OBSERVATIONS = 4096u;

fn hash32(key: u32) -> u32 {
    var x = key;
    x ^= x >> 16u; x *= 0x7feb352du; x ^= x >> 15u; x *= 0x846ca68bu;
    return x ^ (x >> 16u);
}
fn unitFloat(x: u32) -> f32 { return f32(x >> 8u) / 16777216.0; }
fn inVolume(uvw: vec3f) -> bool { return all(uvw >= vec3f(0)) && all(uvw < vec3f(1)); }
fn normalizedPoint(pt: vec3f) -> vec3f { return (pt - p.lower.xyz) * p.inverseSpan.xyz; }
fn cellIndex(cell: vec3u, dims: vec3u) -> u32 { return cell.x + dims.x * (cell.y + cell.z * dims.y); }
fn cellPosition(index: u32, dims: vec3u) -> vec3u {
    return vec3u(index % dims.x, (index / dims.x) % dims.y, index / (dims.x * dims.y));
}
fn directionIndex(dir: vec3f, side: u32) -> u32 {
    let a = abs(dir);
    var face: u32; var uv: vec2f;
    if (a.x >= a.y && a.x >= a.z) {
        face = select(1u, 0u, dir.x >= 0); uv = dir.yz / a.x;
    } else if (a.y >= a.z) {
        face = select(3u, 2u, dir.y >= 0); uv = dir.xz / a.y;
    } else {
        face = select(5u, 4u, dir.z >= 0); uv = dir.xy / a.z;
    }
    let ij = min(vec2u((uv * 0.5 + 0.5) * f32(side)), vec2u(side - 1u));
    return face * side * side + ij.y * side + ij.x;
}
// xyz unit direction, w cube-map quadrature weight (solid-angle Jacobian).
fn binDirection(index: u32, side: u32) -> vec4f {
    let uv = (vec2f(f32(index % side), f32((index / side) % side)) + 0.5) * (2.0 / f32(side)) - 1.0;
    let face = index / (side * side);
    let s = select(1.0, -1.0, (face & 1u) != 0u);
    var local = vec3f(s, uv.x, uv.y);
    if (face >= 4u) { local = vec3f(uv, s); }
    else if (face >= 2u) { local = vec3f(uv.x, s, uv.y); }
    let inv = inverseSqrt(dot(local, local));
    return vec4f(local * inv, inv * inv * inv);
}

@compute @workgroup_size(1) fn begin() { clock[0] += 1u; }

@compute @workgroup_size(64) fn generate(@builtin(global_invocation_id) id: vec3u) {
    let lane = id.x;
    if (lane >= p.screen.w) { return; }
    let stride = p.screen.z;
    let columns = (p.screen.x + stride - 1u) / stride;
    let block = vec2u(lane % columns, lane / columns) * stride;
    let hash = hash32(lane);
    let phase = (clock[0] + hash) % (stride * stride);
    let pixel = min(block + vec2u(phase % stride, phase / stride), p.screen.xy - 1u);
    let surface = textureLoad(positionTexture, vec2i(pixel), 0);
    let normal = textureLoad(normalTexture, vec2i(pixel), 0).xyz;
    if (surface.w <= 0 || !inVolume(normalizedPoint(surface.xyz)) || dot(normal, normal) < 0.1) {
        rays[lane] = Ray(vec4f(0), vec4f(0, 1, 0, 0));
        return;
    }
    let seed = vec2f(unitFloat(hash), unitFloat(hash32(hash)));
    let xi = fract(seed + f32(clock[0] & 65535u) * vec2f(0.754877666, 0.569840291));
    let y = 2.0 * xi.x - 1.0;
    let r = sqrt(max(0.0, 1.0 - y * y));
    let phi = 2.0 * PI * xi.y;
    let sphere = vec3f(r * cos(phi), y, r * sin(phi));
    let n = normalize(normal);
    let dir = select(-sphere, sphere, dot(sphere, n) >= 0);
    let origin = surface.xyz + n * p.distances.w;
    rays[lane] = Ray(vec4f(origin, 0.0001), vec4f(dir, p.distances.y));
}

@compute @workgroup_size(64) fn deposit(@builtin(global_invocation_id) id: vec3u) {
    let lane = id.x;
    if (lane >= p.screen.w) { return; }
    let ray = rays[lane];
    if (ray.directionMax.w <= 0) { return; }
    let coord = clamp(normalizedPoint(ray.originMin.xyz), vec3f(0), vec3f(0.9999999));
    let hit = hits[lane];
    let missed = hit.w < 0;
    var start = 0.0;
    for (var level = 0u; level < 3u; level++) {
        var end = p.distances.x;
        if (level == 1u) { end *= 4.0; }
        if (level == 2u) { end = p.distances.y; }
        if (!missed && hit.w < start) { break; }
        let terminal = level == 2u;
        let transparent = !terminal && (missed || hit.w >= end);
        let grid = p.grid[level];
        let side = 2u << level;
        let index = grid.w + cellIndex(vec3u(coord * vec3f(grid.xyz)), grid.xyz) * (6u * side * side)
                         + directionIndex(ray.directionMax.xyz, side);
        // A bounded accepted count and 64 HDR ceiling guarantee the unsigned
        // fixed-point sum cannot overflow even if all pixels share one bin.
        if (atomicAdd(&batch[index].count, 1u) < MAX_OBSERVATIONS) {
            if (transparent) {
                atomicAdd(&batch[index].transparent, 1u);
            } else {
                let rgb = vec3u(round(clamp(hit.rgb, vec3f(0), vec3f(64)) * FIXED));
                atomicAdd(&batch[index].r, rgb.r);
                atomicAdd(&batch[index].g, rgb.g);
                atomicAdd(&batch[index].b, rgb.b);
            }
        }
        start = end;
    }
}

@compute @workgroup_size(64) fn resolveHistory(@builtin(global_invocation_id) id: vec3u) {
    let index = id.x;
    if (index >= arrayLength(&history)) { return; }
    let count = min(atomicLoad(&batch[index].count), MAX_OBSERVATIONS);
    if (count == 0u) {
        let previous = seen[index];
        if (previous != 0u && clock[0] - previous > 120u) {
            history[index].value = vec4f(0);
            seen[index] = 0u;
        }
        return;
    }
    let inv = 1.0 / f32(count);
    let rgb = vec3f(f32(atomicLoad(&batch[index].r)), f32(atomicLoad(&batch[index].g)), f32(atomicLoad(&batch[index].b)));
    let observation = vec4f(rgb * (inv / FIXED), 1.0 + f32(atomicLoad(&batch[index].transparent)) * inv);
    let old = history[index].value;
    let retention = select(0.0, p.distances.z, old.w > 0.0);
    history[index].value = mix(observation, old, retention);
    seen[index] = clock[0];
}

fn mergeLevel(index: u32, level: u32) {
    let grid = p.grid[level];
    let side = 2u << level;
    let count = 6u * side * side;
    let cell = index / count;
    if (cell >= grid.x * grid.y * grid.z) { return; }
    let dst = grid.w + index;
    let local = history[dst].value;
    // An unobserved interval has no visibility to its parent. A fully opaque
    // interval cannot see its parent either. Avoid the four parent reads.
    if (local.w == 0.0) { merged[dst] = vec4f(0); return; }
    if (local.w == 1.0) { merged[dst] = vec4f(local.rgb, 1); return; }
    var far = vec4f(0);
    if (level < 2u) {
        let parent = p.grid[level + 1u];
        let uvw = (vec3f(cellPosition(cell, grid.xyz)) + 0.5) / vec3f(grid.xyz);
        let parentCell = cellIndex(vec3u(uvw * vec3f(parent.xyz)), parent.xyz);
        let angular = index % count;
        let face = angular / (side * side);
        let childSide = 2u * side;
        let base = parent.w + parentCell * (count * 4u) + face * childSide * childSide
                   + 2u * ((angular / side) % side) * childSide + 2u * (angular % side);
        for (var y = 0u; y < 2u; y++) { for (var x = 0u; x < 2u; x++) {
            let v = merged[base + y * childSide + x];
            if (v.w == 0.0) { continue; }
            // Every child is a quarter of the face-plane area; weight by its solid angle.
            let weight = binDirection((angular % (side * side) / side * 2u + y) * childSide
                 + (angular % side * 2u + x), childSide).w * v.w;
            far += vec4f(v.rgb * weight, weight);
        } }
        if (far.w > 0.0) { far = vec4f(far.rgb / far.w, 1.0); }
    }
    let trans = clamp(local.w - 1.0, 0.0, 1.0);
    merged[dst] = vec4f(local.rgb + trans * far.rgb, select(0.0, 1.0, trans < 0.99999 || far.w > 0.0));
}
@compute @workgroup_size(64) fn mergeFar(@builtin(global_invocation_id) id: vec3u) { mergeLevel(id.x, 2u); }
@compute @workgroup_size(64) fn mergeMiddle(@builtin(global_invocation_id) id: vec3u) { mergeLevel(id.x, 1u); }
@compute @workgroup_size(64) fn mergeNear(@builtin(global_invocation_id) id: vec3u) { mergeLevel(id.x, 0u); }

// Project once per world cell for six diffuse normals. The full-screen gather
// interpolates these irradiances, avoiding a per-pixel directional integral.
@compute @workgroup_size(64) fn project(@builtin(global_invocation_id) id: vec3u) {
    let cell = id.x;
    let grid = p.grid[0];
    if (cell >= grid.x * grid.y * grid.z) { return; }
    var sums: array<vec4f, 6>;
    for (var bin = 0u; bin < 24u; bin++) {
        let v = merged[cell * 24u + bin];
        if (v.w == 0.0) { continue; }
        let dir = binDirection(bin, 2u);
        for (var axis = 0u; axis < 3u; axis++) {
            let i = 2u * axis + select(1u, 0u, dir[axis] >= 0.0);
            let weight = abs(dir[axis]) * dir.w * v.w;
            sums[i] += vec4f(v.rgb * weight, weight);
        }
    }
    for (var i = 0u; i < 6u; i++) {
        let s = sums[i];
        ambientCube[cell * 6u + i] = vec4f(PI * s.rgb / max(s.w, 1e-6), select(0.0, 1.0, s.w > 0.0));
    }
}

@compute @workgroup_size(8, 8) fn gather(@builtin(global_invocation_id) id: vec3u) {
    if (any(id.xy >= p.screen.xy)) { return; }
    let xy = vec2i(id.xy);
    let surface = textureLoad(positionTexture, xy, 0);
    let n = textureLoad(normalTexture, xy, 0).xyz;
    let coord = normalizedPoint(surface.xyz);
    var sum = vec4f(0);
    if (surface.w > 0.0 && inVolume(coord) && dot(n,n) >= 0.1) {
        let dims = p.grid[0].xyz;
        let center = coord * vec3f(dims) - 0.5;
        let lower = vec3i(floor(center));
        let f = fract(center);
        for (var corner = 0u; corner < 8u; corner++) {
            let delta = vec3u(corner & 1u, (corner >> 1u) & 1u, corner >> 2u);
            let cell = cellIndex(vec3u(clamp(lower + vec3i(delta), vec3i(0), vec3i(dims) - 1)), dims);
            let w = select(vec3f(1) - f, f, delta != vec3u(0));
            for (var axis = 0u; axis < 3u; axis++) {
                if (n[axis] == 0.0) { continue; }
                let i = 2u * axis + select(1u, 0u, n[axis] >= 0.0);
                let l = ambientCube[cell * 6u + i];
                let blend = w.x * w.y * w.z * n[axis] * n[axis] * l.w;
                sum += vec4f(l.rgb * blend, blend);
            }
        }
    }
    textureStore(irradianceTexture, xy, vec4f(sum.rgb / max(sum.w, 1e-6), select(0.0, 1.0, sum.w > 0.0)));
}
`;
