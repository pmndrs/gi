import {
    assembleTraceScene, buildTopLevel, createTinyBvhBuilder, uploadTraceScene,
    type Mat4, type MeshGeometry, type TraceSceneInstance, type TraceInstance,
} from '@pmndrs/rc-bvh';
import { buildCaps, dispatchGrid, RtIntervalTracer, sceneFormatOf } from '@pmndrs/raytracer';
import { RC_TRANSPORT_WGSL } from './transport-wgsl.ts';

export type RcRGB = readonly [number, number, number];
export interface RcSurfaceMaterial {
    albedo: RcRGB;
    emission: RcRGB;
    /** Opaque triangles. A hidden face occludes with zero outgoing radiance. */
    side?: 'front' | 'back' | 'double';
}
export interface RcPointLight {
    position: RcRGB;
    /** Linear RGB radiant intensity. */
    intensity: RcRGB;
    distance?: number;
    decay?: number;
}
export interface RcTransportScene {
    meshes: readonly MeshGeometry[];
    instances: readonly TraceSceneInstance[];
    materials: readonly RcSurfaceMaterial[];
    lights: readonly RcPointLight[];
    sky?: RcRGB;
    /** World-space normal offset for visibility rays. */
    shadowBias?: number;
}
export interface RcRayJob {
    /** 32 B per ray: origin.xyz, tMin, unit direction.xyz, tMax. */
    rays: GPUBuffer;
    /** 16 B per result: linear outgoing RGB and first-hit distance. Miss distance is -1. */
    results: GPUBuffer;
    count: number;
}
export type RcTransport = Awaited<ReturnType<typeof createRcTransport>>;

/** Opaque diffuse transport on the caller's GPUDevice. All ray, hit and shadow traffic remains GPU resident. */
export async function createRcTransport(device: GPUDevice, input: RcTransportScene) {
    if (!input.meshes.length || !input.instances.length || !input.materials.length) {
        throw new Error('RC transport requires geometry, instances and materials.');
    }
    // Keep BLAS immutable. A rigid transform only changes the small top-level tree.
    const builder = await createTinyBvhBuilder();
    const meshes = input.meshes.map(mesh => builder.buildLayout(mesh, { triangleLeaves: 'records' }).layout);
    // Use material zero when neither a triangle nor its instance specifies one.
    const placed = input.instances.map(instance => ({ ...instance, material: instance.material ?? 0 }));
    const packed = assembleTraceScene(meshes, placed);
    const instances: TraceInstance[] = placed.map(instance => ({
        ...instance, objectToWorld: Float32Array.from(instance.objectToWorld),
        meshBounds: meshes[instance.meshIndex].bounds,
        nodeOffset: packed.meshes[instance.meshIndex].nodeOffset,
        triangleOffset: packed.meshes[instance.meshIndex].triangleOffset,
    }));
    const caps = buildCaps({ features: device.features, limits: device.limits as unknown as Record<string, unknown> });
    const options = { ...sceneFormatOf(packed), source: 'rays' as const, convention: 'open-low' as const };
    const module = device.createShaderModule({ label: 'RC diffuse transport', code: RC_TRANSPORT_WGSL });
    const [shade, resolve] = await Promise.all(['shade', 'resolve'].map(entryPoint => device.createComputePipelineAsync({
        label: 'RC transport ' + entryPoint, layout: 'auto', compute: { module, entryPoint },
    }))).catch(async error => {
        const diagnostics = (await module.getCompilationInfo()).messages.filter(item => item.type === 'error');
        if (diagnostics.length) throw new Error(diagnostics.map(item => `RC transport WGSL ${item.lineNum}:${item.linePos} ${item.message}`).join('\n'));
        throw error;
    });
    const closest = await RtIntervalTracer.create(device, caps, { ...options, output: 'hit', label: 'RC interval closest' });
    let visibility: RtIntervalTracer;
    try {
        visibility = await RtIntervalTracer.create(device, caps, { ...options, output: 'visible', label: 'RC interval visibility' });
    } catch (error) {
        closest.destroy();
        throw error;
    }
    const uploaded = uploadTraceScene(device, packed, 'RC scene');
    let scene = uploaded;
    const materialCount = input.materials.length;
    const makeBuffer = (label: string, size: number, usage = GPUBufferUsage.STORAGE) =>
        device.createBuffer({ label: 'RC ' + label, size, usage });
    const materialBuffer = makeBuffer('materials', input.materials.length * 32, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    let lightCapacity = Math.max(1, input.lights.length);
    let lightBuffer = makeBuffer('point lights', lightCapacity * 32, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    const frameBuffer = makeBuffer('transport frame', 32, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
    const frameData = new ArrayBuffer(32);
    const settings = new Float32Array(frameData), words = new Uint32Array(frameData);
    settings.set(input.sky ?? [0, 0, 0]);
    settings[5] = input.shadowBias ?? 0.001;
    let topDirty = false;
    const retired: GPUBuffer[] = [];
    let capacity = 0;
    let hits: GPUBuffer, shadows: GPUBuffer, direct: GPUBuffer, vis: GPUBuffer;
    type Bindings = { count: number; shade: GPUBindGroup; resolve: GPUBindGroup };
    let bindings = new WeakMap<GPUBuffer, WeakMap<GPUBuffer, Bindings>>();
    function setMaterials(materials: readonly RcSurfaceMaterial[]) {
        if (materials.length !== materialCount) throw new Error('Recreate RC transport when the material table size changes.');
        const data = new Float32Array(materials.length * 8);
        materials.forEach((m, i) => {
            data.set(m.albedo, 8 * i);
            data[8 * i + 3] = m.side === 'back' ? 1 : m.side === 'double' ? 2 : 0;
            data.set(m.emission, 8 * i + 4);
        });
        device.queue.writeBuffer(materialBuffer, 0, data);
    }
    function setLights(lights: readonly RcPointLight[]) {
        if (lights.length > lightCapacity) {
            retired.push(lightBuffer);
            lightCapacity = lights.length;
            lightBuffer = makeBuffer('point lights', lightCapacity * 32, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
            bindings = new WeakMap();
        }
        words[3] = lights.length;
        if (!lights.length) return;
        const data = new Float32Array(lights.length * 8);
        lights.forEach((l, i) => {
            data.set(l.position, i * 8);
            data[i * 8 + 3] = l.distance ?? 0;
            data.set(l.intensity, i * 8 + 4);
            data[i * 8 + 7] = l.decay ?? 2;
        });
        device.queue.writeBuffer(lightBuffer, 0, data);
    }
    function ensureScratch(count: number) {
        if (count <= capacity) return;
        if (capacity) retired.push(hits, shadows, direct, vis);
        capacity = count;
        hits = makeBuffer('full hit records', count * 40);
        shadows = makeBuffer('shadow rays', count * 32);
        direct = makeBuffer('light samples', count * 16);
        vis = makeBuffer('visibility', count * 4);
        bindings = new WeakMap();
    }
    function groups(job: RcRayJob) {
        let outputs = bindings.get(job.rays);
        if (!outputs) { outputs = new WeakMap(); bindings.set(job.rays, outputs); }
        const cached = outputs.get(job.results);
        if (cached?.count === job.count) return cached;
        const entry = (binding: number, buffer: GPUBuffer, size = buffer.size): GPUBindGroupEntry => ({ binding, resource: { buffer, size } });
        // The result binding also bounds the dispatch's final partial workgroup.
        const output = entry(4, job.results, job.count * 16);
        const next = { count: job.count,
            shade: device.createBindGroup({ layout: shade.getBindGroupLayout(0), entries: [
                entry(0, job.rays, job.count * 32), entry(1, hits), entry(2, materialBuffer), entry(3, lightBuffer),
                output, entry(5, shadows), entry(6, direct), entry(7, frameBuffer),
            ] }),
            resolve: device.createBindGroup({ layout: resolve.getBindGroupLayout(0), entries: [output, entry(6, direct), entry(8, vis)] }),
        };
        outputs.set(job.results, next);
        return next;
    }
    setMaterials(input.materials);
    setLights(input.lights);
    return {
        setMaterials, setLights,
        /** Index into the supplied instances array. Apply before beginFrame. */
        setTransform(index: number, matrix: Mat4) {
            const target = instances[index];
            if (!target) throw new RangeError('Unknown RC instance ' + index);
            if (matrix.length !== 16) throw new RangeError('An RC transform requires a column-major 4×4 matrix.');
            (target.objectToWorld as Float32Array).set(matrix);
            topDirty = true;
        },
        /** Call once after submitting the preceding frame; all jobs for this frame share the uniform ring. */
        beginFrame() {
            for (const buffer of retired.splice(0)) buffer.destroy();
            if (topDirty) {
                const top = buildTopLevel(instances);
                if (top.buffer.byteLength > scene.topLevel.size) {
                    // Changing instance placement can change SAH topology even with an unchanged BLAS.
                    if (scene !== uploaded) scene.topLevel.destroy();
                    const topLevel = makeBuffer('moving TLAS', top.buffer.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
                    scene = { ...uploaded, topLevel };
                }
                device.queue.writeBuffer(scene.topLevel, 0, top.buffer);
                topDirty = false;
            }
            words[4]++;
            device.queue.writeBuffer(frameBuffer, 0, frameData);
            closest.beginFrame();
            visibility.beginFrame();
        },
        encode(encoder: GPUCommandEncoder, job: RcRayJob) {
            if (!Number.isSafeInteger(job.count) || job.count < 0 || job.count * 32 > job.rays.size || job.count * 16 > job.results.size) throw new RangeError('RC ray job exceeds its buffer capacity.');
            if (job.count === 0) return;
            ensureScratch(job.count);
            const bind = groups(job);
            const dispatch = dispatchGrid(job.count, 64, device.limits.maxComputeWorkgroupsPerDimension);
            const run = (pipeline: GPUComputePipeline, group: GPUBindGroup) => {
                const pass = encoder.beginComputePass({ label: pipeline.label });
                pass.setPipeline(pipeline);
                pass.setBindGroup(0, group);
                pass.dispatchWorkgroups(...dispatch);
                pass.end();
            };
            closest.encode(encoder, scene, { source: job.rays, results: hits, count: job.count });
            run(shade, bind.shade);
            if (words[3]) {
                visibility.encode(encoder, scene, { source: shadows, results: vis, count: job.count });
                run(resolve, bind.resolve);
            }
        },
        get scratchBytes() { return capacity * (40 + 32 + 16 + 4); },
        dispose() {
            closest.destroy();
            visibility.destroy();
            if (scene !== uploaded) scene.topLevel.destroy();
            uploaded.destroy();
            for (const b of [frameBuffer, materialBuffer, lightBuffer, hits, shadows, direct, vis, ...retired]) b?.destroy();
        },
    };
}
