/** Compiled only when roughSpecular is enabled, appended to the native cache module. */
export const nativeRcSpecularShader = /* wgsl */ `
// Perspective camera world position (w=1), or orthographic surface-to-camera direction (w=0).
@group(1) @binding(0) var<uniform> specularView: vec4f;
@group(1) @binding(1) var specularOutput: texture_storage_2d<rgba16float,write>;

// Nine taps in a finite reflected cone. The outer ring keeps polished
// distant illumination at the angular scale of the final 8×8 cube faces.
const disk = array<vec2f, 9>(
    vec2f(0), vec2f(1,0), vec2f(-1,0), vec2f(0,1), vec2f(0,-1),
    vec2f(0.707107,0.707107), vec2f(-0.707107,0.707107),
    vec2f(0.707107,-0.707107), vec2f(-0.707107,-0.707107));

// A cube texel lives at its direction's center. Reproject out-of-face
// neighbors so the four-tap reconstruction extends over cube edges.
fn angularAddress(face: u32, ij: vec2i, side: u32) -> u32 {
    if (all(ij >= vec2i(0)) && all(ij < vec2i(i32(side)))) {
        return face * side * side + u32(ij.y) * side + u32(ij.x);
    }
    let uv = (vec2f(ij) + 0.5) * (2.0 / f32(side)) - 1.0;
    let s = select(1.0, -1.0, (face & 1u) != 0u);
    var direction = vec3f(uv, s);
    if (face < 2u) { direction = vec3f(s, uv); }
    else if (face < 4u) { direction = vec3f(uv.x, s, uv.y); }
    return directionIndex(direction, side);
}

// Middle history stores 1 + transmittance in alpha. The outer merged
// interval uses alpha as validity. Unknown neighbors contribute no weight.
fn filteredInterval(base: u32, side: u32, face: u32, uv: vec2f) -> vec4f {
    let texel = uv * f32(side) - 0.5;
    let ij = vec2i(floor(texel));
    let f = fract(texel);
    var sum = vec4f(0);
    var known = 0.0;
    for (var i = 0u; i < 4u; i++) {
        let offset = vec2u(i & 1u, i >> 1u);
        let w = select(vec2f(1) - f, f, offset != vec2u(0));
        let addr = base + angularAddress(face, ij + vec2i(offset), side);
        var obs: vec4f;
        if (side == 4u) { obs = history[addr].value; } else { obs = merged[addr]; }
        if (obs.w > 0.0) {
            sum += (w.x * w.y) * obs;
            known += w.x * w.y;
        }
    }
    return sum / max(known, 1e-8);
}

@compute @workgroup_size(8, 8) fn roughReflection(@builtin(global_invocation_id) id: vec3u) {
    if (any(id.xy >= p.screen.xy)) { return; }
    let xy = vec2i(id.xy);
    let surface = textureLoad(positionTexture, xy, 0);
    let material = textureLoad(normalTexture, xy, 0);
    let uvw = normalizedPoint(surface.xyz);
    let toView = specularView.xyz - surface.xyz * specularView.w;
    if (surface.w <= 0.0 || !inVolume(uvw) || dot(material.xyz, material.xyz) < 0.1 ||
        dot(toView, toView) < 1e-10 || dot(material.xyz, toView) <= 0.0) {
        textureStore(specularOutput, xy, vec4f(0));
        return;
    }
    let n = normalize(material.xyz);
    let axis = reflect(-normalize(toView), n);
    let roughness = clamp(material.w, 0.0, 1.0);
    // A normalized spherical Gaussian around the reflected view direction.
    // Limit its peak sharpness: 2×2 texels per cube face cannot resolve a
    // mirror. For opaque near hits the minimum half-power radius is acos(5/6),
    // about 34 degrees. Roughness one integrates the cosine-weighted hemisphere.
    let sharpness = 6.0 * (1.0 - roughness * roughness);
    let dims = p.grid[0].xyz;
    let center = uvw * vec3f(dims) - 0.5;
    let lower = vec3i(floor(center));
    let f = fract(center);
    var bases: array<u32, 8>;
    var weights: array<f32, 8>;
    for (var corner = 0u; corner < 8u; corner++) {
        let delta = vec3u(corner & 1u, (corner >> 1u) & 1u, corner >> 2u);
        let cell = vec3u(clamp(lower + vec3i(delta), vec3i(0), vec3i(dims) - 1));
        let w = select(vec3f(1) - f, f, delta != vec3u(0));
        bases[corner] = cellIndex(cell, dims) * 24u;
        weights[corner] = w.x * w.y * w.z;
    }
    var sum = vec4f(0);
    var expectedWeight = 0.0;
    for (var bin = 0u; bin < 24u; bin++) {
        let direction = binDirection(bin, 2u);
        let cosine = dot(n, direction.xyz);
        if (cosine <= 0.0) { continue; }
        let angular = cosine * direction.w * exp2(sharpness * (dot(axis, direction.xyz) - 1.0));
        expectedWeight += angular;
        for (var corner = 0u; corner < 8u; corner++) {
            let spatial = weights[corner];
            if (spatial <= 0.0) { continue; }
            let observation = merged[bases[corner] + bin];
            let weight = angular * spatial * observation.w;
            sum += vec4f(observation.rgb * weight, weight);
        }
    }
    // Preserve constant incident radiance as the cone widens. Carry the
    // observed share separately so a host can blend with another estimate.
    var radiance = sum.rgb / max(sum.w, 1e-8);
    var support = clamp(sum.w / max(expectedWeight, 1e-8), 0, 1);
    if (roughness < 0.65 && support > 0.0) {
        let middle = p.grid[1];
        let outer = p.grid[2];
        let middleBase = middle.w + 96u * cellIndex(vec3u(uvw * vec3f(middle.xyz)), middle.xyz);
        let outerBase = outer.w + 384u * cellIndex(vec3u(uvw * vec3f(outer.xyz)), outer.xyz);
        let helper = select(vec3f(0,1,0), vec3f(1,0,0), abs(axis.y) > 0.9);
        let tangent = normalize(cross(helper, axis));
        let bitangent = cross(axis, tangent);
        let radius = 0.28 + 0.8 * roughness * roughness;
        var raw = vec4f(0);
        var transparentWeight = 0.0;
        var coneWeight = 0.0;
        for (var tap = 0u; tap < 9u; tap++) {
            let sampleDir = normalize(axis + radius * (disk[tap].x * tangent + disk[tap].y * bitangent));
            let w = max(dot(n, sampleDir), 0.0) * select(1.0, 2.0, tap == 0u);
            if (w <= 0.0) { continue; }
            coneWeight += w;
            let bin = directionIndex(sampleDir, 2u);
            var near = vec4f(0);
            var known = 0.0;
            for (var corner = 0u; corner < 8u; corner++) {
                let h = history[bases[corner] + bin].value;
                if (h.w > 0.0) { near += vec4f(h.rgb, h.w - 1.0) * weights[corner]; known += weights[corner]; }
            }
            if (known <= 0.0) { continue; }
            near /= known;
            var value = near.rgb;
            if (near.w > 0.0) {
                let face = bin / 4u;
                var uv = sampleDir.xy / max(abs(sampleDir.z), 1e-8);
                if (face < 2u) { uv = sampleDir.yz / abs(sampleDir.x); }
                else if (face < 4u) { uv = sampleDir.xz / abs(sampleDir.y); }
                uv = uv * 0.5 + 0.5;
                let mid = filteredInterval(middleBase, 4u, face, uv);
                if (mid.w <= 0.0) { continue; }
                var farther = mid.rgb;
                if (mid.w > 1.0) {
                    let far = filteredInterval(outerBase, 8u, face, uv);
                    if (far.w <= 0.0) { continue; }
                    farther += (mid.w - 1.0) * far.rgb;
                }
                value += near.w * farther;
            }
            raw += vec4f(value, 1.0) * (w * known);
            transparentWeight += w * known * near.w;
        }
        if (raw.w > 0.0) {
            // Near opaque radiance has only 24 directional bins. Give the
            // narrower query weight only where the near interval sees farther
            // and the distant intervals have actual observations.
            let rawSupport = clamp(raw.w / max(coneWeight, 1e-8), 0.0, 1.0);
            let confidence = min(1.0, rawSupport / max(support, 1e-8));
            let blend = (1.0 - smoothstep(0.25, 0.65, roughness)) * confidence
                        * clamp(transparentWeight / raw.w, 0.0, 1.0);
            radiance = mix(radiance, raw.rgb / raw.w, blend);
            support = mix(support, rawSupport, blend);
        }
    }
    textureStore(specularOutput, xy, vec4f(radiance, support));
}
`;
