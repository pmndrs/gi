/**
 * The GPU mesh packing (T2.13) in Node: the fake device runs the kernel's
 * CPU twin (testing/pack-mesh-cpu.ts) at submit, and what lands in the vertex
 * buffer is compared with the CPU packer, the reference: everything bit for
 * bit except octahedral normals, which may differ by one step (1/32767) per
 * component. The GPU page `scene-gpu-packing` checks the real WGSL the same
 * way on procedural scenes and Sponza.
 */

import { describe, expect, it } from 'vitest';
import type { MeshDescriptor } from '../contract/scene.ts';
import { GINotReadyError } from '../contract/errors.ts';
import { createRandom, randomUnitVector } from '../trace/oracle-fixtures.ts';
import { packDispatchSize } from './gpu-mesh-packing.ts';
import { NONE } from './layout.ts';
import { packMesh } from './mesh-packing.ts';
import { encodeOctahedral } from './octahedral.ts';
import { PACK_MESH_ENTRY_POINT, PACK_MESH_WORKGROUP_SIZE } from './pack-mesh-wgsl.ts';
import { PlatformScene, type SceneUpload, type UploadedMesh } from './platform-scene.ts';
import { createFakeGpu, type FakeBinding, type FakeBuffer, type FakeGpu } from './testing/fake-gpu.ts';
import { buildBasicScene, checkReadback, makeQueries, materialIndices, SceneModel } from './testing/fixture.ts';
import { packOctahedralF32, runPackMeshDispatch } from './testing/pack-mesh-cpu.ts';
import { SceneReader } from './testing/scene-reader.ts';

function wordsOf(binding: FakeBinding): Uint32Array {
    return new Uint32Array(binding.buffer.bytes.buffer, binding.offset, binding.size / 4);
}

/** A fake device that runs `gi/pack-mesh` on the CPU. */
function packingGpu(limits = {}): FakeGpu {
    return createFakeGpu(limits, {
        kernels: {
            [PACK_MESH_ENTRY_POINT]: (bindings, workgroupsX, workgroupsY) => {
                runPackMeshDispatch(wordsOf(bindings.get(0)!), wordsOf(bindings.get(1)!), wordsOf(bindings.get(2)!), workgroupsX, workgroupsY);
            },
        },
    });
}

function encodeAndSubmit(gpu: FakeGpu, scene: PlatformScene): SceneUpload {
    const encoder = gpu.createEncoder();
    const upload = scene.encode(encoder);
    gpu.submit(encoder);
    return upload;
}

function vertexWords(scene: PlatformScene): Uint32Array {
    const buffer = (scene.module.resources()[0] as GPUBufferBinding).buffer as unknown as FakeBuffer;
    return buffer.words;
}

/** The signed snorm16 halves of an octahedral word. */
function halves(word: number): [number, number] {
    const toSigned = (bits: number) => (bits >= 0x8000 ? bits - 0x10000 : bits);
    return [toSigned(word & 0xffff), toSigned(word >>> 16)];
}

/** Steps between two octahedral words: the larger difference of their two components. */
function octahedralSteps(a: number, b: number): number {
    const [au, av] = halves(a);
    const [bu, bv] = halves(b);
    return Math.max(Math.abs(au - bu), Math.abs(av - bv));
}

/**
 * Compares a GPU-packed mesh with the CPU packer's words: every word equal,
 * except normals within one step. Returns what differs.
 */
function compareWithCpuPacker(descriptor: MeshDescriptor, uploaded: UploadedMesh, vertices: Uint32Array): string[] {
    const reference = packMesh(descriptor);
    const failures: string[] = [];
    const normals = reference.offsets.normals;
    for (let word = 0; word < reference.words.length; word++) {
        const actual = vertices[uploaded.vertexOffset + word];
        const expected = reference.words[word];
        const isNormal = normals !== NONE && word >= normals && word < normals + reference.vertexCount;
        if (isNormal ? octahedralSteps(actual, expected) > 1 : actual !== expected) {
            failures.push(`word ${word}: ${actual.toString(16)} against ${expected.toString(16)}`);
            if (failures.length > 5) break;
        }
    }
    return failures;
}

function randomNormals(seed: number, count: number): Float32Array {
    const random = createRandom(seed);
    const normals = new Float32Array(count * 3);
    for (let vertex = 0; vertex < count; vertex++) normals.set(randomUnitVector(random), vertex * 3);
    return normals;
}

/** A mesh of `vertexCount` vertices and `triangleCount` random triangles, with every optional part. */
function testMesh(seed: number, vertexCount: number, triangleCount: number, indexType: 'u16' | 'u32', groups = true): MeshDescriptor {
    const random = createRandom(seed);
    const positions = Float32Array.from({ length: vertexCount * 3 }, () => random() * 10 - 5);
    const indexValues = Array.from({ length: triangleCount * 3 }, () => Math.floor(random() * Math.min(vertexCount, 0x10000)));
    return {
        positions,
        indices: indexType === 'u16' ? Uint16Array.from(indexValues) : Uint32Array.from(indexValues),
        normals: randomNormals(seed + 1, vertexCount),
        uvs: Float32Array.from({ length: vertexCount * 2 }, () => random()),
        uvs1: Float32Array.from({ length: vertexCount * 2 }, () => random()),
        groups: groups ? [{ start: 0, count: Math.floor(triangleCount / 2) * 3, materialSlot: 1 }, { start: Math.floor(triangleCount / 2) * 3, count: 3, materialSlot: 2 }] : undefined,
    };
}

async function gpuPackingScene(gpu: FakeGpu): Promise<PlatformScene> {
    const scene = new PlatformScene(gpu.device);
    await scene.prepareGpuPacking();
    return scene;
}

describe('the kernel twin', () => {
    it('encodes normals within one step of the CPU encoder, edge cases included', () => {
        const random = createRandom(7);
        const bits = new Uint32Array(3);
        const floats = new Float32Array(bits.buffer);
        const cases: number[][] = [
            [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1],
            [0, 0, 0], [-0, -0, -0], [0, -0, -1], [1e30, 1e30, -1e30], [3e38, 3e38, 3e38],
            [NaN, 0, 1], [Infinity, 0, 0], [0.5, -0.5, -1e-9], [1e-3, 0, -1],
        ];
        for (let sample = 0; sample < 100000; sample++) cases.push(randomUnitVector(random));
        let worst = 0;
        for (const vector of cases) {
            floats.set(vector);
            const gpu = packOctahedralF32(bits[0], bits[1], bits[2]);
            const cpu = encodeOctahedral(floats[0], floats[1], floats[2]);
            worst = Math.max(worst, octahedralSteps(gpu, cpu));
        }
        expect(worst).toBeLessThanOrEqual(1);
    });

    it('splits large dispatches over two dimensions without losing threads', () => {
        for (const items of [1, 255, 256, 257, 65535 * 256, 65535 * 256 + 1, 28_000_000]) {
            const { x, y, threadsPerRow } = packDispatchSize(items);
            expect(x).toBeLessThanOrEqual(65535);
            expect(threadsPerRow).toBe(x * PACK_MESH_WORKGROUP_SIZE);
            expect(x * y * PACK_MESH_WORKGROUP_SIZE).toBeGreaterThanOrEqual(items);
        }
    });
});

describe('meshes packed on the GPU', () => {
    const meshes: Record<string, MeshDescriptor> = {
        '16-bit indices given as 16-bit, odd index count': testMesh(1, 300, 101, 'u16'),
        '16-bit indices given as 32-bit (narrowed)': testMesh(2, 500, 120, 'u32'),
        '16-bit indices given as 32-bit, odd index count (narrowed)': testMesh(3, 40, 7, 'u32'),
        '32-bit indices given as 32-bit': testMesh(4, 70000, 50, 'u32', false),
        '32-bit indices given as 16-bit (widened)': testMesh(5, 70000, 33, 'u16'),
        'positions and indices only': { positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]), indices: new Uint16Array([0, 1, 2]) },
    };

    for (const [name, descriptor] of Object.entries(meshes)) {
        it(`matches the CPU packer: ${name}`, async () => {
            const gpu = packingGpu();
            const scene = await gpuPackingScene(gpu);
            scene.createMesh(descriptor);
            const upload = encodeAndSubmit(gpu, scene);
            expect(upload.meshesUploaded).toHaveLength(1);
            expect(compareWithCpuPacker(descriptor, upload.meshesUploaded[0], vertexWords(scene))).toEqual([]);
        });
    }

    it('packs many meshes in one encode, across several staging buffers and a vertex buffer that grows', async () => {
        // A 4 MiB binding limit: staging buffers of at most 4 MiB, and these meshes are 4.4 MB raw, so they take two.
        const gpu = packingGpu({ maxStorageBufferBindingSize: 4 * 1024 * 1024 });
        const scene = await gpuPackingScene(gpu);
        const descriptors = Array.from({ length: 40 }, (_, index) => testMesh(100 + index, 2000 + index * 37, 900 + index, index % 2 === 0 ? 'u16' : 'u32'));
        const handles = descriptors.map((descriptor) => scene.createMesh(descriptor));
        const upload = encodeAndSubmit(gpu, scene);
        const vertices = vertexWords(scene);
        for (const uploaded of upload.meshesUploaded) {
            const descriptor = descriptors[handles.indexOf(uploaded.handle)];
            expect(compareWithCpuPacker(descriptor, uploaded, vertices)).toEqual([]);
        }
        expect(upload.bytesWritten).toBeGreaterThan(0);
        expect(gpu.buffers.filter((buffer) => buffer.label.endsWith('mesh staging'))).toHaveLength(2);
    });

    it('hands tree builders copies of positions and indices in the host types', async () => {
        const gpu = packingGpu();
        const scene = await gpuPackingScene(gpu);
        const descriptor = testMesh(9, 100, 20, 'u16');
        scene.createMesh(descriptor);
        // The host reuses its arrays straight after the call: neither the GPU nor the copies may see it.
        const positionsBefore = descriptor.positions.slice();
        const indicesBefore = descriptor.indices.slice();
        descriptor.positions.fill(7);
        descriptor.indices.fill(0);
        const [uploaded] = encodeAndSubmit(gpu, scene).meshesUploaded;
        expect(uploaded.geometry.indices).toBeInstanceOf(Uint16Array);
        expect(Array.from(uploaded.geometry.positions)).toEqual(Array.from(positionsBefore));
        expect(Array.from(uploaded.geometry.indices)).toEqual(Array.from(indicesBefore));
        const restored = { ...descriptor, positions: positionsBefore, indices: indicesBefore };
        expect(compareWithCpuPacker(restored, uploaded, vertexWords(scene))).toEqual([]);
    });

    it('reads back every accessor of the basic scene', async () => {
        const gpu = packingGpu();
        const scene = await gpuPackingScene(gpu);
        const model = new SceneModel(scene);
        buildBasicScene(model);
        encodeAndSubmit(gpu, scene);
        const [vertices, tables] = scene.module.resources().map((resource) => (resource as GPUBufferBinding).buffer as unknown as FakeBuffer);
        const reader = new SceneReader(vertices.bytes, tables.bytes);
        const queries = makeQueries(model);
        const materials = materialIndices(model);
        expect(checkReadback(model, queries, materials, reader.readAll(queries, materials))).toEqual([]);
    });

    it('frees staging buffers one encode after packing, and forgets meshes replaced before their upload', async () => {
        const gpu = packingGpu();
        const scene = await gpuPackingScene(gpu);
        const kept = scene.createMesh(testMesh(20, 1000, 300, 'u16'));
        const replaced = scene.createMesh(testMesh(21, 1000, 300, 'u32'));
        const destroyed = scene.createMesh(testMesh(22, 1000, 300, 'u16'));
        const replacement = testMesh(23, 500, 100, 'u32');
        scene.updateMesh(replaced, replacement);
        scene.destroyMesh(destroyed);
        expect(scene.memory.stagingBuffers).toBeGreaterThan(0);

        const upload = encodeAndSubmit(gpu, scene);
        expect(upload.meshesUploaded.map((mesh) => mesh.handle).sort()).toEqual([kept, replaced].sort());
        const replacedUpload = upload.meshesUploaded.find((mesh) => mesh.handle === replaced)!;
        expect(compareWithCpuPacker(replacement, replacedUpload, vertexWords(scene))).toEqual([]);

        const staging = gpu.buffers.filter((buffer) => buffer.label.endsWith('mesh staging'));
        expect(staging.length).toBeGreaterThan(0);
        expect(staging.every((buffer) => !buffer.destroyed)).toBe(true);
        encodeAndSubmit(gpu, scene);
        expect(staging.every((buffer) => buffer.destroyed)).toBe(true);
        expect(scene.memory.stagingBuffers).toBe(0);
    });

    it('refuses to encode staged meshes before the kernel is compiled, and keeps them queued', async () => {
        const gpu = packingGpu();
        const scene = new PlatformScene(gpu.device);
        const compiled = scene.prepareGpuPacking();
        const descriptor = testMesh(30, 200, 50, 'u16');
        scene.createMesh(descriptor);
        expect(() => encodeAndSubmit(gpu, scene)).toThrow(GINotReadyError);
        await compiled;
        const [uploaded] = encodeAndSubmit(gpu, scene).meshesUploaded;
        expect(compareWithCpuPacker(descriptor, uploaded, vertexWords(scene))).toEqual([]);
    });

    it('packs on the CPU when a mesh is too large to stage, and before prepareGpuPacking', async () => {
        // 112 KiB per binding: the large mesh's raw arrays (121 KB) don't fit a staging buffer, its packed words (97 KB) fit the vertex buffer.
        const gpu = packingGpu({ maxStorageBufferBindingSize: 112 * 1024 });
        const scene = new PlatformScene(gpu.device, { initialCapacities: { vertexWords: 1024 } });
        const early = testMesh(40, 100, 30, 'u16');
        scene.createMesh(early);
        await scene.prepareGpuPacking();
        const large = testMesh(41, 3000, 100, 'u32', false);
        scene.createMesh(large);
        const upload = encodeAndSubmit(gpu, scene);
        expect(gpu.buffers.some((buffer) => buffer.label.endsWith('mesh staging'))).toBe(false);
        expect(upload.meshesUploaded).toHaveLength(2);
        const vertices = vertexWords(scene);
        const descriptors = [early, large];
        upload.meshesUploaded.forEach((uploaded, index) => expect(compareWithCpuPacker(descriptors[index], uploaded, vertices)).toEqual([]));
    });
});
