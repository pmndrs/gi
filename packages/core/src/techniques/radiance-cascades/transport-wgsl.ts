export const RC_TRANSPORT_WGSL = /* wgsl */ `
struct Ray { origin: vec3f, lo: f32, direction: vec3f, hi: f32 }
struct Material { albedo: vec3f, side: f32, emission: vec3f, pad: f32 }
struct Light { position: vec3f, cutoff: f32, intensity: vec3f, decay: f32 }
struct Frame { sky: vec3f, lightCount: u32, seed: u32, bias: f32, pad: vec2u }
@group(0) @binding(0) var<storage,read> rays: array<Ray>;
@group(0) @binding(1) var<storage,read> hits: array<u32>;
@group(0) @binding(2) var<storage,read> materials: array<Material>;
@group(0) @binding(3) var<storage,read> lights: array<Light>;
@group(0) @binding(4) var<storage,read_write> result: array<vec4f>;
@group(0) @binding(5) var<storage,read_write> shadows: array<Ray>;
@group(0) @binding(6) var<storage,read_write> direct: array<vec4f>;
@group(0) @binding(7) var<uniform> frame: Frame;
@group(0) @binding(8) var<storage,read> visibility: array<u32>;

fn random(index: u32) -> f32 {
    var n = index * 747796405u + 2891336453u;
    n = ((n >> ((n >> 28u) + 4u)) ^ n) * 277803737u;
    return f32(((n >> 22u) ^ n) >> 8u) / 16777216.0;
}

@compute @workgroup_size(64)
fn shade(@builtin(global_invocation_id) invocation: vec3u, @builtin(num_workgroups) groups: vec3u) {
    let i = invocation.x + invocation.y * groups.x * 64u;
    if (i >= arrayLength(&result)) { return; }
    // RtIntervalTracer full hit ABI is ten words; geometric normals face the authored winding.
    let base = i * 10u;
    let t = bitcast<f32>(hits[base]);
    direct[i] = vec4f(0);
    shadows[i] = Ray(vec3f(0), 0.0, vec3f(0, 0, 1), 0.0);
    if (t < 0.0) { result[i] = vec4f(frame.sky, -1); return; }
    let material = materials[hits[base + 6u]];
    let front = hits[base + 3u] != 0u;
    if ((!front && material.side == 0.0) || (front && material.side == 1.0)) {
        result[i] = vec4f(0, 0, 0, t);
        return;
    }
    result[i] = vec4f(material.emission, t);
    let normal = vec3f(bitcast<f32>(hits[base+7u]), bitcast<f32>(hits[base+8u]), bitcast<f32>(hits[base+9u])) * select(-1.0, 1.0, front);
    let point = rays[i].origin + rays[i].direction * t;
    var total = 0.0;
    var sampleWeight = 0.0;
    var sampleValue = vec3f(0);
    var samplePosition = vec3f(0);
    // One visibility ray. Reservoir sampling gives each local light probability
    // proportional to its unoccluded diffuse contribution at this hit.
    for (var j = 0u; j < frame.lightCount; j++) {
        let light = lights[j];
        let delta = light.position - point;
        let distance = length(delta);
        let cosine = max(dot(normal, delta / max(distance, 1e-6)), 0.0);
        var attenuation = 1.0 / max(pow(distance, light.decay), 0.01);
        if (light.cutoff > 0.0) {
            let falloff = clamp(1.0 - pow(distance / light.cutoff, 4.0), 0.0, 1.0);
            attenuation *= falloff * falloff;
        }
        let value = material.albedo * light.intensity * (cosine * attenuation / 3.14159265);
        let weight = dot(value, vec3f(0.2126, 0.7152, 0.0722));
        total += weight;
        if (weight > 0.0 && random(i ^ (frame.seed * 31337u) ^ (j * 104729u)) * total < weight) {
            sampleWeight = weight;
            sampleValue = value;
            samplePosition = light.position;
        }
    }
    if (sampleWeight > 0.0) {
        let origin = point + normal * frame.bias;
        let delta = samplePosition - origin;
        let lengthToLight = length(delta);
        if (lengthToLight > frame.bias) {
            shadows[i] = Ray(origin, 0.0, delta / lengthToLight, lengthToLight - frame.bias);
            direct[i] = vec4f(sampleValue * (total / sampleWeight), 0);
        }
    }
}

@compute @workgroup_size(64)
fn resolve(@builtin(global_invocation_id) invocation: vec3u, @builtin(num_workgroups) groups: vec3u) {
    let i = invocation.x + invocation.y * groups.x * 64u;
    if (i < arrayLength(&result) && visibility[i] != 0u) { result[i] += direct[i]; }
}
`;
