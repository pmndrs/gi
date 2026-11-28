/**
 * What the accessor checks read back, and how it is laid out in words.
 *
 * The GPU readback page runs a kernel that calls every WGSL accessor and
 * writes the answers in this layout; the unit tests run a CPU reading of the
 * same buffers (scene-reader.ts) that writes the same layout. Both are then
 * decoded here and compared with values worked out from the scene's inputs
 * (fixture.ts). Test-only.
 */

/** One accessor query: an instance slot, a triangle, a point on it, and a uv set. */
export interface InstanceQuery {
    readonly instance: number;
    readonly triangle: number;
    readonly barycentrics: readonly [number, number];
    readonly channel: 0 | 1;
}

/** Words per query in the input buffer: instance, triangle, channel, 0, barycentrics (2 floats), 0, 0. */
export const QUERY_INPUT_WORDS = 8;

export interface UvMappingResult {
    /** mat3x2 columns: m0..m5. */
    readonly transform: number[];
    readonly channel: number;
    readonly wrapU: number;
    readonly wrapV: number;
}

export interface InstanceQueryResult {
    readonly triangleCount: number;
    readonly contributes: boolean;
    readonly mesh: number;
    readonly materialAt: number;
    /** Column-major 4x4. */
    readonly transform: number[];
    /** Column-major 3x3. */
    readonly normalMatrix: number[];
    readonly bounds: number[];
    /** World-space corners, 0 then 1 then 2. */
    readonly corners: number[];
    readonly geometricNormal: number[];
    readonly normalAt: number[];
    readonly uv: number[];
    readonly vertices: number[];
}

export interface MaterialResult {
    readonly albedo: number[];
    readonly alphaCutoff: number;
    readonly emission: number[];
    readonly blocksLight: boolean;
    readonly doubleSided: boolean;
    readonly albedoTexture: number;
    readonly albedoUv: UvMappingResult;
    readonly emissiveTexture: number;
    readonly emissiveUv: UvMappingResult;
}

export interface LightResult {
    readonly kind: number;
    readonly position: number[];
    readonly direction: number[];
    readonly range: number;
    readonly intensity: number[];
    readonly castsShadows: boolean;
    readonly cosInnerCone: number;
    readonly cosOuterCone: number;
    readonly radius: number;
    readonly group: number;
}

export interface EnvironmentResult {
    readonly kind: number;
    readonly radiance: number[];
    readonly intensity: number;
    readonly sunDirection: number[];
    readonly turbidity: number;
    readonly rotationY: number;
}

export interface SceneReadback {
    readonly instanceCount: number;
    readonly lightCount: number;
    readonly directionalLightCount: number;
    readonly environment: EnvironmentResult;
    readonly queries: InstanceQueryResult[];
    readonly materials: MaterialResult[];
    readonly lights: LightResult[];
}

/** Word offsets in the output buffer. */
export const GLOBAL_WORDS = 16;
export const GLOBAL_WORD = {
    instanceCount: 0,
    lightCount: 1,
    directionalLightCount: 2,
    environmentKind: 3,
    radiance: 4,
    intensity: 7,
    sunDirection: 8,
    turbidity: 11,
    rotationY: 12,
} as const;

export const QUERY_WORDS = 64;
export const QUERY_WORD = {
    triangleCount: 0,
    contributes: 1,
    mesh: 2,
    materialAt: 3,
    transform: 4,
    normalMatrix: 20,
    bounds: 29,
    corners: 35,
    geometricNormal: 44,
    normalAt: 47,
    uv: 50,
    vertices: 52,
} as const;

export const MATERIAL_WORDS = 32;
export const MATERIAL_RESULT_WORD = {
    albedo: 0,
    alphaCutoff: 3,
    emission: 4,
    blocksLight: 7,
    doubleSided: 8,
    albedoTexture: 9,
    albedoUv: 10,
    emissiveTexture: 19,
    emissiveUv: 20,
} as const;
/** A uv mapping result: six floats, then channel, wrapU, wrapV. */
export const UV_RESULT_WORDS = 9;

export const LIGHT_WORDS = 16;
export const LIGHT_RESULT_WORD = {
    kind: 0,
    position: 1,
    direction: 4,
    range: 7,
    intensity: 8,
    castsShadows: 11,
    cosInnerCone: 12,
    cosOuterCone: 13,
    radius: 14,
    group: 15,
} as const;

/** Where each part starts in the output buffer, in words. */
export function outputLayout(counts: { queries: number; materials: number; lights: number }) {
    const queries = GLOBAL_WORDS;
    const materials = queries + counts.queries * QUERY_WORDS;
    const lights = materials + counts.materials * MATERIAL_WORDS;
    return { queries, materials, lights, totalWords: lights + counts.lights * LIGHT_WORDS };
}

export function packQueries(queries: readonly InstanceQuery[]): Uint32Array<ArrayBuffer> {
    const words = new Uint32Array(Math.max(1, queries.length) * QUERY_INPUT_WORDS);
    const floats = new Float32Array(words.buffer);
    queries.forEach((query, index) => {
        const base = index * QUERY_INPUT_WORDS;
        words[base] = query.instance;
        words[base + 1] = query.triangle;
        words[base + 2] = query.channel;
        floats[base + 4] = query.barycentrics[0];
        floats[base + 5] = query.barycentrics[1];
    });
    return words;
}

function floatsAt(floats: Float32Array, start: number, count: number): number[] {
    return Array.from(floats.subarray(start, start + count));
}

function decodeUv(words: Uint32Array, floats: Float32Array, start: number): UvMappingResult {
    return { transform: floatsAt(floats, start, 6), channel: words[start + 6], wrapU: words[start + 7], wrapV: words[start + 8] };
}

/** Turns the output words into results. */
export function decodeReadback(output: ArrayBuffer, counts: { queries: number; materials: number; lights: number }): SceneReadback {
    const words = new Uint32Array(output);
    const floats = new Float32Array(output);
    const layout = outputLayout(counts);
    const queries: InstanceQueryResult[] = [];
    for (let index = 0; index < counts.queries; index++) {
        const base = layout.queries + index * QUERY_WORDS;
        queries.push({
            triangleCount: words[base + QUERY_WORD.triangleCount],
            contributes: words[base + QUERY_WORD.contributes] !== 0,
            mesh: words[base + QUERY_WORD.mesh],
            materialAt: words[base + QUERY_WORD.materialAt],
            transform: floatsAt(floats, base + QUERY_WORD.transform, 16),
            normalMatrix: floatsAt(floats, base + QUERY_WORD.normalMatrix, 9),
            bounds: floatsAt(floats, base + QUERY_WORD.bounds, 6),
            corners: floatsAt(floats, base + QUERY_WORD.corners, 9),
            geometricNormal: floatsAt(floats, base + QUERY_WORD.geometricNormal, 3),
            normalAt: floatsAt(floats, base + QUERY_WORD.normalAt, 3),
            uv: floatsAt(floats, base + QUERY_WORD.uv, 2),
            vertices: Array.from(words.subarray(base + QUERY_WORD.vertices, base + QUERY_WORD.vertices + 3)),
        });
    }
    const materials: MaterialResult[] = [];
    for (let index = 0; index < counts.materials; index++) {
        const base = layout.materials + index * MATERIAL_WORDS;
        materials.push({
            albedo: floatsAt(floats, base + MATERIAL_RESULT_WORD.albedo, 3),
            alphaCutoff: floats[base + MATERIAL_RESULT_WORD.alphaCutoff],
            emission: floatsAt(floats, base + MATERIAL_RESULT_WORD.emission, 3),
            blocksLight: words[base + MATERIAL_RESULT_WORD.blocksLight] !== 0,
            doubleSided: words[base + MATERIAL_RESULT_WORD.doubleSided] !== 0,
            albedoTexture: words[base + MATERIAL_RESULT_WORD.albedoTexture],
            albedoUv: decodeUv(words, floats, base + MATERIAL_RESULT_WORD.albedoUv),
            emissiveTexture: words[base + MATERIAL_RESULT_WORD.emissiveTexture],
            emissiveUv: decodeUv(words, floats, base + MATERIAL_RESULT_WORD.emissiveUv),
        });
    }
    const lights: LightResult[] = [];
    for (let index = 0; index < counts.lights; index++) {
        const base = layout.lights + index * LIGHT_WORDS;
        lights.push({
            kind: words[base + LIGHT_RESULT_WORD.kind],
            position: floatsAt(floats, base + LIGHT_RESULT_WORD.position, 3),
            direction: floatsAt(floats, base + LIGHT_RESULT_WORD.direction, 3),
            range: floats[base + LIGHT_RESULT_WORD.range],
            intensity: floatsAt(floats, base + LIGHT_RESULT_WORD.intensity, 3),
            castsShadows: words[base + LIGHT_RESULT_WORD.castsShadows] !== 0,
            cosInnerCone: floats[base + LIGHT_RESULT_WORD.cosInnerCone],
            cosOuterCone: floats[base + LIGHT_RESULT_WORD.cosOuterCone],
            radius: floats[base + LIGHT_RESULT_WORD.radius],
            group: words[base + LIGHT_RESULT_WORD.group],
        });
    }
    return {
        instanceCount: words[GLOBAL_WORD.instanceCount],
        lightCount: words[GLOBAL_WORD.lightCount],
        directionalLightCount: words[GLOBAL_WORD.directionalLightCount],
        environment: {
            kind: words[GLOBAL_WORD.environmentKind],
            radiance: floatsAt(floats, GLOBAL_WORD.radiance, 3),
            intensity: floats[GLOBAL_WORD.intensity],
            sunDirection: floatsAt(floats, GLOBAL_WORD.sunDirection, 3),
            turbidity: floats[GLOBAL_WORD.turbidity],
            rotationY: floats[GLOBAL_WORD.rotationY],
        },
        queries,
        materials,
        lights,
    };
}

/**
 * The WGSL kernel that calls every accessor and writes this layout. Group 0
 * holds the scene module (bindings 0 and 1); group 1 the queries, the
 * material indices and the output. The light count comes from the scene.
 */
export function accessorKernelWgsl(sceneModuleCode: string): string {
    const q = QUERY_WORD;
    const m = MATERIAL_RESULT_WORD;
    const l = LIGHT_RESULT_WORD;
    const g = GLOBAL_WORD;
    return /* wgsl */ `${sceneModuleCode}

struct Counts {
    queries: u32,
    materials: u32,
    lights: u32,
    queriesBase: u32,
    materialsBase: u32,
    lightsBase: u32,
}

@group(1) @binding(0) var<uniform> counts: Counts;
// Queries (${QUERY_INPUT_WORDS} words each), then one material index per material check.
@group(1) @binding(1) var<storage, read> inputs: array<u32>;
@group(1) @binding(2) var<storage, read_write> outputs: array<u32>;

fn putFloat(word: u32, value: f32) {
    outputs[word] = bitcast<u32>(value);
}

fn putVec3(word: u32, value: vec3f) {
    putFloat(word, value.x);
    putFloat(word + 1u, value.y);
    putFloat(word + 2u, value.z);
}

fn putUv(word: u32, mapping: GiUvMapping) {
    putFloat(word, mapping.transform[0].x);
    putFloat(word + 1u, mapping.transform[0].y);
    putFloat(word + 2u, mapping.transform[1].x);
    putFloat(word + 3u, mapping.transform[1].y);
    putFloat(word + 4u, mapping.transform[2].x);
    putFloat(word + 5u, mapping.transform[2].y);
    outputs[word + 6u] = mapping.channel;
    outputs[word + 7u] = mapping.wrapU;
    outputs[word + 8u] = mapping.wrapV;
}

fn runQuery(index: u32) {
    let inputBase = index * ${QUERY_INPUT_WORDS}u;
    let instance = inputs[inputBase];
    let triangle = inputs[inputBase + 1u];
    let channel = inputs[inputBase + 2u];
    let barycentrics = vec2f(bitcast<f32>(inputs[inputBase + 4u]), bitcast<f32>(inputs[inputBase + 5u]));
    let outputBase = counts.queriesBase + index * ${QUERY_WORDS}u;

    let triangleCount = giInstanceTriangleCount(instance);
    outputs[outputBase + ${q.triangleCount}u] = triangleCount;
    outputs[outputBase + ${q.contributes}u] = select(0u, 1u, giInstanceContributesToGI(instance));
    if (triangleCount == 0u) {
        return;   // a free slot: nothing else is defined
    }
    outputs[outputBase + ${q.mesh}u] = giInstanceMesh(instance);
    outputs[outputBase + ${q.materialAt}u] = giMaterialAt(instance, triangle);
    let transform = giInstanceTransform(instance);
    for (var column = 0u; column < 4u; column += 1u) {
        for (var row = 0u; row < 4u; row += 1u) {
            putFloat(outputBase + ${q.transform}u + column * 4u + row, transform[column][row]);
        }
    }
    let normalMatrix = giInstanceNormalMatrix(instance);
    for (var column = 0u; column < 3u; column += 1u) {
        putVec3(outputBase + ${q.normalMatrix}u + column * 3u, normalMatrix[column]);
    }
    let bounds = giInstanceBounds(instance);
    putVec3(outputBase + ${q.bounds}u, bounds.min);
    putVec3(outputBase + ${q.bounds}u + 3u, bounds.max);
    let corners = giInstanceTriangle(instance, triangle);
    putVec3(outputBase + ${q.corners}u, corners.vertex0);
    putVec3(outputBase + ${q.corners}u + 3u, corners.vertex1);
    putVec3(outputBase + ${q.corners}u + 6u, corners.vertex2);
    putVec3(outputBase + ${q.geometricNormal}u, giInstanceGeometricNormal(instance, triangle));
    putVec3(outputBase + ${q.normalAt}u, giInstanceNormalAt(instance, triangle, barycentrics));
    let uv = giInstanceUvAt(instance, triangle, barycentrics, channel);
    putFloat(outputBase + ${q.uv}u, uv.x);
    putFloat(outputBase + ${q.uv}u + 1u, uv.y);
    let vertices = giMeshTriangleVertices(giInstanceMesh(instance), triangle);
    outputs[outputBase + ${q.vertices}u] = vertices.x;
    outputs[outputBase + ${q.vertices}u + 1u] = vertices.y;
    outputs[outputBase + ${q.vertices}u + 2u] = vertices.z;
}

fn runMaterial(index: u32) {
    let material = giMaterial(inputs[counts.queries * ${QUERY_INPUT_WORDS}u + index]);
    let outputBase = counts.materialsBase + index * ${MATERIAL_WORDS}u;
    putVec3(outputBase + ${m.albedo}u, material.albedo);
    putFloat(outputBase + ${m.alphaCutoff}u, material.alphaCutoff);
    putVec3(outputBase + ${m.emission}u, material.emission);
    outputs[outputBase + ${m.blocksLight}u] = select(0u, 1u, material.blocksLight);
    outputs[outputBase + ${m.doubleSided}u] = select(0u, 1u, material.doubleSided);
    outputs[outputBase + ${m.albedoTexture}u] = material.albedoTexture;
    putUv(outputBase + ${m.albedoUv}u, material.albedoUv);
    outputs[outputBase + ${m.emissiveTexture}u] = material.emissiveTexture;
    putUv(outputBase + ${m.emissiveUv}u, material.emissiveUv);
}

fn runLight(index: u32) {
    let light = giLight(index);
    let outputBase = counts.lightsBase + index * ${LIGHT_WORDS}u;
    outputs[outputBase + ${l.kind}u] = light.kind;
    putVec3(outputBase + ${l.position}u, light.position);
    putVec3(outputBase + ${l.direction}u, light.direction);
    putFloat(outputBase + ${l.range}u, light.range);
    putVec3(outputBase + ${l.intensity}u, light.intensity);
    outputs[outputBase + ${l.castsShadows}u] = select(0u, 1u, light.castsShadows);
    putFloat(outputBase + ${l.cosInnerCone}u, light.cosInnerCone);
    putFloat(outputBase + ${l.cosOuterCone}u, light.cosOuterCone);
    putFloat(outputBase + ${l.radius}u, light.radius);
    outputs[outputBase + ${l.group}u] = light.group;
}

fn runGlobals() {
    outputs[${g.instanceCount}u] = giInstanceCount();
    outputs[${g.lightCount}u] = giLightCount();
    outputs[${g.directionalLightCount}u] = giDirectionalLightCount();
    let environment = giEnvironment();
    outputs[${g.environmentKind}u] = environment.kind;
    putVec3(${g.radiance}u, environment.radiance);
    putFloat(${g.intensity}u, environment.intensity);
    putVec3(${g.sunDirection}u, environment.sunDirection);
    putFloat(${g.turbidity}u, environment.turbidity);
    putFloat(${g.rotationY}u, environment.rotationY);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let index = id.x;
    if (index == 0u) {
        runGlobals();
    }
    if (index < counts.queries) {
        runQuery(index);
    }
    if (index < counts.materials) {
        runMaterial(index);
    }
    if (index < counts.lights) {
        runLight(index);
    }
}
`;
}
