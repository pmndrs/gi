/**
 * Scene copy (T1.2) on a real GPU, on a default-limits device:
 *
 * 1. **Accessors against the inputs.** A kernel calls every WGSL accessor of
 *    the scene module for a few points on every instance slot, every material
 *    and every light, and the answers are compared with values worked out
 *    from the scene's inputs (packages/ddgi-core/src/platform/scene/testing/fixture.ts).
 *    Three states: the basic scene, after edits, and after churn that
 *    reallocates every buffer.
 * 2. **Nothing recompiles.** Shader modules and pipelines are counted on the
 *    device; creating and destroying meshes, instances, materials and lights
 *    (with reallocations) must not add any. Only bind groups are rebuilt,
 *    when `module.version` changes.
 * 3. **A DDGI-shaped kernel fits the binding budget.** The tracer's inline
 *    module (3 storage buffers), the scene copy (2) and 3 technique storage
 *    buffers, 8 in all, in one compute kernel on a device with the default
 *    limit of 8. It traces rays, reads the hit's surface through the scene
 *    copy, casts a shadow ray and counts hits with an atomic, and the results
 *    are checked (the hit point lies on the triangle the scene copy returns,
 *    the front face agrees with the geometric normal, the albedo is the
 *    material the inputs give that triangle).
 */

import type { BindingSlot, ModuleBindingEntry, WgslModule } from '../../../../packages/ddgi-core/src/platform/contract/wgsl.ts';
import { PlatformScene } from '../../../../packages/ddgi-core/src/platform/scene/platform-scene.ts';
import {
    buildBasicScene,
    checkReadback,
    churnScene,
    editBasicScene,
    expectedMaterialAt,
    instanceAtSlot,
    makeQueries,
    materialIndices,
    SceneModel,
} from '../../../../packages/ddgi-core/src/platform/scene/testing/fixture.ts';
import {
    accessorKernelWgsl,
    decodeReadback,
    outputLayout,
    packQueries,
} from '../../../../packages/ddgi-core/src/platform/scene/testing/readback.ts';
import {
    assembleTraceScene,
    createCpuTracer,
    createTraceModule,
    packRays,
    uploadTraceScene,
    type TraceRay,
} from '../../../../packages/ddgi-core/src/platform/trace/index.ts';
import { buildMeshBvhWithTinyBvh } from '../../../../packages/ddgi-core/src/platform/trace/mesh-bvh-topology.ts';
import { createTinyBvhBuilder } from '../../../../packages/bvh/src/index.ts';
import { createRandom, randomUnitVector } from '../../../../packages/ddgi-core/src/platform/trace/oracle-fixtures.ts';
import { assert, runGpuTest } from '../../client.ts';

// Mesh trees come from tinybvh (WASM), the only mesh builder since B0.
const tinyBvh = await createTinyBvhBuilder();

/** Default `maxStorageBuffersPerShaderStage`: the budget the DDGI-shaped kernel must fit. */
const DEFAULT_STORAGE_BUFFERS = 8;

/** Small starting tables, so the churn reallocates every buffer more than once. */
const SMALL_CAPACITIES = { instances: 4, meshes: 4, materials: 4, lights: 2, materialListWords: 8, vertexWords: 1024 };

interface CreationCounts {
    shaderModules: number;
    pipelines: number;
    bindGroups: number;
}

/** Wraps the device's create calls so the page can show which ones scene changes cause. */
function countCreations(device: GPUDevice): CreationCounts {
    const counts: CreationCounts = { shaderModules: 0, pipelines: 0, bindGroups: 0 };
    const wrap = <Name extends 'createShaderModule' | 'createComputePipeline' | 'createComputePipelineAsync' | 'createRenderPipeline' | 'createRenderPipelineAsync' | 'createBindGroup'>(
        name: Name,
        counter: keyof CreationCounts,
    ) => {
        const original = device[name].bind(device) as (...args: unknown[]) => unknown;
        (device as unknown as Record<string, unknown>)[name] = (...args: unknown[]) => {
            counts[counter]++;
            return original(...args);
        };
    };
    wrap('createShaderModule', 'shaderModules');
    wrap('createComputePipeline', 'pipelines');
    wrap('createComputePipelineAsync', 'pipelines');
    wrap('createRenderPipeline', 'pipelines');
    wrap('createRenderPipelineAsync', 'pipelines');
    wrap('createBindGroup', 'bindGroups');
    return counts;
}

/**
 * What a kernel builder does with platform modules: every module in group 0,
 * bindings numbered one after another in module order.
 */
interface ComposedModules {
    code: string;
    layout: GPUBindGroupLayout;
    storageBuffers: number;
    bindGroup(): GPUBindGroup;
}

function composeModules(device: GPUDevice, modules: readonly WgslModule[], label: string): ComposedModules {
    let nextBinding = 0;
    const placed = modules.map((module) => {
        const slot: BindingSlot = { group: 0, firstBinding: nextBinding };
        nextBinding += module.entries.length;
        return { module, slot };
    });
    const entries: GPUBindGroupLayoutEntry[] = placed.flatMap(({ module, slot }) =>
        module.entries.map((entry: ModuleBindingEntry, index) => ({ ...entry, binding: slot.firstBinding + index })),
    );
    const layout = device.createBindGroupLayout({ label: `${label} group 0`, entries });
    let cached: { group: GPUBindGroup; versions: string } | null = null;
    return {
        code: placed.map(({ module, slot }) => module.code(slot)).join('\n'),
        layout,
        storageBuffers: modules.reduce((sum, module) => sum + module.storageBufferCount, 0),
        bindGroup() {
            // Rebuilt only when a module reallocated: the "compare version each frame" rule.
            const versions = modules.map((module) => module.version).join(',');
            if (!cached || cached.versions !== versions) {
                const resources = placed.flatMap(({ module, slot }) =>
                    module.resources().map((resource, index) => ({ binding: slot.firstBinding + index, resource })),
                );
                cached = { group: device.createBindGroup({ label: `${label} group 0`, layout, entries: resources }), versions };
            }
            return cached.group;
        },
    };
}

async function readBuffer(device: GPUDevice, source: GPUBuffer, size: number): Promise<ArrayBuffer> {
    const readback = device.createBuffer({ label: 'scene-copy readback', size, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
    const encoder = device.createCommandEncoder({ label: 'scene-copy readback' });
    encoder.copyBufferToBuffer(source, 0, readback, 0, size);
    device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    const copy = readback.getMappedRange().slice(0);
    readback.unmap();
    readback.destroy();
    return copy;
}

// ---------------------------------------------------------------------------
// 1 and 2: accessors, and nothing recompiles
// ---------------------------------------------------------------------------

interface AccessorKernel {
    pipeline: GPUComputePipeline;
    sceneGroup: ComposedModules;
    ownLayout: GPUBindGroupLayout;
}

async function createAccessorKernel(device: GPUDevice, scene: PlatformScene): Promise<AccessorKernel> {
    const sceneGroup = composeModules(device, [scene.module], 'accessors');
    const ownLayout = device.createBindGroupLayout({
        label: 'accessors group 1',
        entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
            { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        ],
    });
    const pipeline = await device.createComputePipelineAsync({
        label: 'scene accessors',
        layout: device.createPipelineLayout({ bindGroupLayouts: [sceneGroup.layout, ownLayout] }),
        compute: { module: device.createShaderModule({ label: 'scene accessors', code: accessorKernelWgsl(sceneGroup.code) }), entryPoint: 'main' },
    });
    return { pipeline, sceneGroup, ownLayout };
}

/** Uploads the scene's pending changes and checks every accessor. Returns the failures. */
async function checkAccessors(device: GPUDevice, kernel: AccessorKernel, model: SceneModel): Promise<{ failures: string[]; queries: number }> {
    const queries = makeQueries(model);
    const materials = materialIndices(model);
    const counts = { queries: queries.length, materials: materials.length, lights: model.lights.size };
    const layout = outputLayout(counts);

    const queryWords = packQueries(queries);
    const inputWords = new Uint32Array(queryWords.length + Math.max(1, materials.length));
    inputWords.set(queryWords);
    inputWords.set(materials, queries.length * 8);
    const inputs = device.createBuffer({ label: 'accessor inputs', size: inputWords.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(inputs, 0, inputWords);
    const uniforms = device.createBuffer({ label: 'accessor counts', size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(uniforms, 0, new Uint32Array([counts.queries, counts.materials, counts.lights, layout.queries, layout.materials, layout.lights, 0, 0]));
    const outputBytes = layout.totalWords * 4;
    const outputs = device.createBuffer({ label: 'accessor outputs', size: outputBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });

    // As the platform would: scene uploads first, in the same encoder, then the technique's pass.
    const encoder = device.createCommandEncoder({ label: 'accessors' });
    model.scene.encode(encoder);
    const pass = encoder.beginComputePass({ label: 'accessors' });
    pass.setPipeline(kernel.pipeline);
    pass.setBindGroup(0, kernel.sceneGroup.bindGroup());
    pass.setBindGroup(
        1,
        device.createBindGroup({
            label: 'accessors group 1',
            layout: kernel.ownLayout,
            entries: [
                { binding: 0, resource: { buffer: uniforms } },
                { binding: 1, resource: { buffer: inputs } },
                { binding: 2, resource: { buffer: outputs } },
            ],
        }),
    );
    pass.dispatchWorkgroups(Math.ceil(Math.max(counts.queries, counts.materials, counts.lights, 1) / 64));
    pass.end();
    device.queue.submit([encoder.finish()]);

    const readback = decodeReadback(await readBuffer(device, outputs, outputBytes), counts);
    inputs.destroy();
    uniforms.destroy();
    outputs.destroy();
    return { failures: checkReadback(model, queries, materials, readback), queries: queries.length };
}

// ---------------------------------------------------------------------------
// 3: the DDGI-shaped kernel
// ---------------------------------------------------------------------------

/** Words written per ray by the DDGI-shaped kernel. */
const RAY_RESULT_WORDS = 12;

/**
 * The technique's part of the kernel: group 1 is its 3 storage buffers.
 * Rays in, one result per ray out, and a hit counter (an atomic, as an
 * accumulation pass would use).
 */
const DDGI_SHAPED_WGSL = /* wgsl */ `
@group(1) @binding(0) var<storage, read> rays: array<GiRay>;
@group(1) @binding(1) var<storage, read_write> results: array<u32>;
@group(1) @binding(2) var<storage, read_write> hitCounter: array<atomic<u32>>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3u) {
    let index = id.x;
    if (index >= arrayLength(&rays)) {
        return;
    }
    let ray = rays[index];
    let base = index * ${RAY_RESULT_WORDS}u;
    let hit = giTraceClosest(ray);
    results[base] = bitcast<u32>(hit.t);
    if (hit.t < 0.0) {
        return;
    }
    atomicAdd(&hitCounter[0], 1u);

    // The surface at the hit, read through the scene copy.
    let position = ray.origin + ray.direction * hit.t;
    let corners = giInstanceTriangle(hit.instance, hit.triangle);
    let onTriangle = corners.vertex0 * (1.0 - hit.barycentrics.x - hit.barycentrics.y) +
        corners.vertex1 * hit.barycentrics.x + corners.vertex2 * hit.barycentrics.y;
    let geometricNormal = giInstanceGeometricNormal(hit.instance, hit.triangle);
    let facing = dot(ray.direction, geometricNormal);
    let material = giMaterial(giMaterialAt(hit.instance, hit.triangle));
    let shadingNormal = giInstanceNormalAt(hit.instance, hit.triangle, hit.barycentrics);

    // A shadow ray toward the first directional light, from just off the surface.
    var blocked = 2u;   // 2 = no directional light
    if (giDirectionalLightCount() > 0u) {
        let light = giLight(0u);
        let side = select(1.0, -1.0, facing > 0.0);
        let shadowOrigin = position + geometricNormal * side * 1.0e-3;
        blocked = select(0u, 1u, giTraceAny(GiRay(shadowOrigin, 0.0, -light.direction, 1.0e30)));
    }

    results[base + 1u] = hit.instance;
    results[base + 2u] = hit.triangle;
    results[base + 3u] = select(0u, 1u, hit.frontFace);
    results[base + 4u] = bitcast<u32>(distance(position, onTriangle));
    results[base + 5u] = bitcast<u32>(facing);
    results[base + 6u] = bitcast<u32>(material.albedo.x);
    results[base + 7u] = bitcast<u32>(material.albedo.y);
    results[base + 8u] = bitcast<u32>(material.albedo.z);
    results[base + 9u] = blocked;
    results[base + 10u] = bitcast<u32>(length(shadingNormal));
}
`;

/** Rays from around the scene: half aimed at triangle centres of traceable instances, half random. */
function makeRays(model: SceneModel, count: number): TraceRay[] {
    const random = createRandom(21);
    const targets: [number, number, number][] = [];
    for (const instance of model.instances.values()) {
        if (!instance.visible || !instance.contributesToGi) continue;
        const mesh = model.meshes.get(instance.mesh)!;
        const m = instance.transform;
        for (let triangle = 0; triangle < Math.min(64, mesh.indices.length / 3); triangle++) {
            const centre = [0, 0, 0];
            for (let corner = 0; corner < 3; corner++) {
                const vertex = mesh.indices[triangle * 3 + corner];
                for (let axis = 0; axis < 3; axis++) centre[axis] += mesh.positions[vertex * 3 + axis] / 3;
            }
            targets.push([0, 1, 2].map((row) => m[row] * centre[0] + m[4 + row] * centre[1] + m[8 + row] * centre[2] + m[12 + row]) as [number, number, number]);
        }
    }
    const rays: TraceRay[] = [];
    for (let index = 0; index < count; index++) {
        const origin: [number, number, number] = [random() * 30 - 15, random() * 15 - 5, random() * 30 - 15];
        let direction = randomUnitVector(random);
        if (index % 2 === 0 && targets.length > 0) {
            const target = targets[Math.floor(random() * targets.length)];
            const toward = [target[0] - origin[0], target[1] - origin[1], target[2] - origin[2]];
            const length = Math.hypot(toward[0], toward[1], toward[2]);
            direction = [toward[0] / length, toward[1] / length, toward[2] / length];
        }
        // Round to f32 so the CPU walk sees exactly the GPU's rays.
        const toFloat32 = (v: readonly number[]): [number, number, number] => [Math.fround(v[0]), Math.fround(v[1]), Math.fround(v[2])];
        rays.push({ origin: toFloat32(origin), direction: toFloat32(direction), tMin: 0, tMax: 1e30 });
    }
    return rays;
}

async function runDdgiShapedKernel(device: GPUDevice, model: SceneModel): Promise<Record<string, unknown>> {
    const scene = model.scene;
    // The tracer's buffers, built from the same inputs: GiHit.instance must be the scene copy's slot.
    const meshHandles = [...model.meshes.keys()];
    const meshIndex = new Map(meshHandles.map((handle, index) => [handle, index]));
    const traceData = assembleTraceScene(
        meshHandles.map((handle) => {
            const mesh = model.meshes.get(handle)!;
            return buildMeshBvhWithTinyBvh({ positions: mesh.positions, indices: Uint32Array.from(mesh.indices) }, tinyBvh);
        }),
        [...model.instances].map(([handle, instance]) => ({
            meshIndex: meshIndex.get(instance.mesh)!,
            instanceIndex: scene.instanceSlot(handle),
            objectToWorld: instance.transform,
            contributesToGi: instance.contributesToGi,
            visible: instance.visible,
        })),
    );
    const traceBuffers = uploadTraceScene(device, traceData, 'scene-copy trace');
    const traceModule = createTraceModule(() => traceBuffers);

    const platform = composeModules(device, [traceModule, scene.module], 'ddgi-shaped');
    const ownStorageBuffers = 3;
    const totalStorage = platform.storageBuffers + ownStorageBuffers;
    assert(traceModule.storageBufferCount === 3, `the trace module binds ${traceModule.storageBufferCount} storage buffers, expected 3`);
    assert(scene.module.storageBufferCount <= 2, `the scene copy binds ${scene.module.storageBufferCount} storage buffers, the budget is 2`);
    assert(
        totalStorage <= device.limits.maxStorageBuffersPerShaderStage,
        `the kernel needs ${totalStorage} storage buffers, the device allows ${device.limits.maxStorageBuffersPerShaderStage}`,
    );

    const ownLayout = device.createBindGroupLayout({
        label: 'ddgi-shaped group 1',
        entries: [
            { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
            { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
            { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
        ],
    });
    // Explicit layouts, so all 8 storage buffers count against the limit (an 'auto' layout drops unused ones).
    const pipeline = await device.createComputePipelineAsync({
        label: 'ddgi-shaped',
        layout: device.createPipelineLayout({ bindGroupLayouts: [platform.layout, ownLayout] }),
        compute: { module: device.createShaderModule({ label: 'ddgi-shaped', code: `${platform.code}\n${DDGI_SHAPED_WGSL}` }), entryPoint: 'main' },
    });

    const rays = makeRays(model, 4096);
    const packedRays = packRays(rays) as Float32Array<ArrayBuffer>;
    const rayBuffer = device.createBuffer({ label: 'ddgi-shaped rays', size: packedRays.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(rayBuffer, 0, packedRays);
    const resultBytes = rays.length * RAY_RESULT_WORDS * 4;
    const results = device.createBuffer({ label: 'ddgi-shaped results', size: resultBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const counter = device.createBuffer({ label: 'ddgi-shaped counter', size: 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });

    const encoder = device.createCommandEncoder({ label: 'ddgi-shaped' });
    scene.encode(encoder);
    const pass = encoder.beginComputePass({ label: 'ddgi-shaped' });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, platform.bindGroup());
    pass.setBindGroup(
        1,
        device.createBindGroup({
            label: 'ddgi-shaped group 1',
            layout: ownLayout,
            entries: [
                { binding: 0, resource: { buffer: rayBuffer } },
                { binding: 1, resource: { buffer: results } },
                { binding: 2, resource: { buffer: counter } },
            ],
        }),
    );
    pass.dispatchWorkgroups(Math.ceil(rays.length / 64));
    pass.end();
    device.queue.submit([encoder.finish()]);

    const resultData = await readBuffer(device, results, resultBytes);
    const hitCount = new Uint32Array(await readBuffer(device, counter, 16))[0];
    rayBuffer.destroy();
    results.destroy();
    counter.destroy();
    traceBuffers.destroy();

    // Check every hit against the inputs and the CPU walk.
    const words = new Uint32Array(resultData);
    const floats = new Float32Array(resultData);
    const cpu = createCpuTracer(traceData);
    const failures: string[] = [];
    let hits = 0;
    let shadowed = 0;
    let cpuDisagreements = 0;
    let worstOffTriangle = 0;
    for (let index = 0; index < rays.length; index++) {
        const base = index * RAY_RESULT_WORDS;
        const t = floats[base];
        const cpuHit = cpu.closest(rays[index]);
        if ((t >= 0) !== (cpuHit !== null) || (cpuHit && cpuHit.instanceIndex !== words[base + 1])) cpuDisagreements++;
        if (!(t >= 0)) continue;
        hits++;
        const slot = words[base + 1];
        const triangle = words[base + 2];
        const frontFace = words[base + 3] === 1;
        const offTriangle = floats[base + 4];
        const facing = floats[base + 5];
        worstOffTriangle = Math.max(worstOffTriangle, offTriangle / Math.max(1, t));
        if (words[base + 9] === 1) shadowed++;
        const handle = instanceAtSlot(model, slot);
        if (handle === undefined) {
            failures.push(`ray ${index} hit slot ${slot}, which holds no instance`);
            continue;
        }
        const instance = model.instances.get(handle)!;
        if (!instance.visible || !instance.contributesToGi) failures.push(`ray ${index} hit instance slot ${slot}, which rays must not see`);
        if (offTriangle > 1e-4 * Math.max(1, t)) failures.push(`ray ${index}: the hit point is ${offTriangle} from the scene copy's triangle ${triangle} of slot ${slot}`);
        // frontFace is true exactly when the ray runs against the geometric normal (skip grazing hits).
        if (Math.abs(facing) > 1e-4 && frontFace !== facing < 0) failures.push(`ray ${index}: frontFace ${frontFace} but dot(direction, geometric normal) = ${facing}`);
        const material = model.materials.get([...model.materials.keys()].find((h) => scene.materialSlot(h) === expectedMaterialAt(model, handle, triangle))!)!;
        const albedo = material.albedo ?? [1, 1, 1];
        for (let channel = 0; channel < 3; channel++) {
            if (Math.abs(floats[base + 6 + channel] - albedo[channel]) > 1e-6) {
                failures.push(`ray ${index}: albedo [${Array.from(floats.subarray(base + 6, base + 9))}], expected [${albedo}]`);
                break;
            }
        }
        if (Math.abs(floats[base + 10] - 1) > 1e-4) failures.push(`ray ${index}: shading normal has length ${floats[base + 10]}`);
    }
    if (hitCount !== hits) failures.push(`the atomic counted ${hitCount} hits, the results hold ${hits}`);
    if (hits < rays.length / 4) failures.push(`only ${hits} of ${rays.length} rays hit; the test rays are aimed badly`);
    // Rays that graze an edge can go either way between f32 and f64 walks; anything more is a real disagreement.
    if (cpuDisagreements > rays.length / 500) failures.push(`${cpuDisagreements} rays disagree with the CPU walk`);
    assert(failures.length === 0, `DDGI-shaped kernel: ${failures.length} problem(s): ${failures.slice(0, 8).join(' | ')}`);
    return {
        storageBuffers: { trace: traceModule.storageBufferCount, sceneCopy: scene.module.storageBufferCount, technique: ownStorageBuffers, total: totalStorage, limit: device.limits.maxStorageBuffersPerShaderStage },
        rays: rays.length,
        hits,
        shadowed,
        cpuDisagreements,
        worstOffTriangleRelative: worstOffTriangle,
    };
}

// ---------------------------------------------------------------------------

await runGpuTest(async ({ device, log }) => {
    // runGpuTest asks for no limits, so this is a default device: the binding budget is the real one.
    assert(
        device.limits.maxStorageBuffersPerShaderStage === DEFAULT_STORAGE_BUFFERS,
        `expected a default-limits device (8 storage buffers per stage), got ${device.limits.maxStorageBuffersPerShaderStage}`,
    );
    const created = countCreations(device);
    const scene = new PlatformScene(device, { label: 'scene-copy', initialCapacities: SMALL_CAPACITIES });
    const model = new SceneModel(scene);
    const kernel = await createAccessorKernel(device, scene);
    const compiledAfterSetup = { shaderModules: created.shaderModules, pipelines: created.pipelines };

    const states: Record<string, unknown>[] = [];
    const check = async (state: string) => {
        const versionBefore = scene.module.version;
        const { failures, queries } = await checkAccessors(device, kernel, model);
        assert(failures.length === 0, `${state}: ${failures.length} accessor mismatch(es): ${failures.slice(0, 8).join(' | ')}`);
        states.push({ state, queries, version: scene.module.version, reallocated: scene.module.version !== versionBefore, ...scene.counts, memoryBytes: scene.memoryBytes });
        log(`${state}: ${queries} queries match the inputs (version ${scene.module.version}, ${scene.memoryBytes} bytes)`);
    };

    const basic = buildBasicScene(model);
    await check('basic scene');
    editBasicScene(model, basic);
    await check('after edits');
    let churnUploads = 0;
    churnScene(model, 4, 11, () => {
        // Upload mid-churn too, so reallocations happen with meshes already on the GPU.
        const encoder = device.createCommandEncoder({ label: 'churn upload' });
        scene.encode(encoder);
        device.queue.submit([encoder.finish()]);
        churnUploads++;
    });
    await check('after churn');

    const compiledAfterChanges = { shaderModules: created.shaderModules, pipelines: created.pipelines };
    const bindGroupsAfterChanges = created.bindGroups;
    assert(
        compiledAfterChanges.shaderModules === compiledAfterSetup.shaderModules && compiledAfterChanges.pipelines === compiledAfterSetup.pipelines,
        `scene changes created shader modules or pipelines: ${JSON.stringify(compiledAfterSetup)} before, ${JSON.stringify(compiledAfterChanges)} after`,
    );
    const versions = states.map((state) => state.version as number);
    assert(versions[versions.length - 1] > versions[0], 'the churn should have reallocated a buffer (module.version unchanged)');
    log(`create/destroy: ${compiledAfterChanges.pipelines} pipeline(s) and ${compiledAfterChanges.shaderModules} shader module(s) before and after; ${bindGroupsAfterChanges} bind groups`);

    const ddgi = await runDdgiShapedKernel(device, model);
    log(`DDGI-shaped kernel: ${JSON.stringify(ddgi.storageBuffers)} storage buffers, ${ddgi.hits} hits of ${ddgi.rays}`);

    const memory = scene.memory;
    scene.dispose();
    return {
        states,
        churnUploads,
        compiled: { beforeSceneChanges: compiledAfterSetup, afterSceneChanges: compiledAfterChanges },
        bindGroupsDuringSceneChanges: bindGroupsAfterChanges,
        memory,
        ddgiShapedKernel: ddgi,
    };
});
