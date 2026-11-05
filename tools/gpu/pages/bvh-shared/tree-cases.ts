/**
 * Meshes every GPU builder's test page builds (bvh-lbvh, bvh-hploc): edge cases
 * (one, two and three triangles, tile boundaries), a grid with 16-bit indices,
 * all-equal triangles, a flat mesh, flags, a large soup, and meshes already in
 * GPU buffers (interleaved vertices; 16-bit indices at an odd offset).
 */

import type { GpuLbvhGeometry } from '../../../../packages/bvh/src/gpu-lbvh.ts';
import type { TinyBvhInput } from '../../../../packages/bvh/src/tinybvh.ts';

/** Deterministic random numbers in [0, 1). */
export function randomSource(seed: number): () => number {
    let state = seed >>> 0;
    return () => {
        state = (state + 0x6d2b79f5) >>> 0;
        let t = state;
        t = Math.imul(t ^ (t >>> 15), t | 1);
        t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
        return ((t ^ (t >>> 14)) >>> 0) / 2 ** 32;
    };
}

export function randomSoup(count: number, seed: number, spread = 20): Float32Array {
    const random = randomSource(seed);
    const positions = new Float32Array(count * 9);
    for (let triangle = 0; triangle < count; triangle++) {
        const center = [random() * spread - spread / 2, random() * spread - spread / 2, random() * spread - spread / 2];
        for (let corner = 0; corner < 3; corner++) {
            for (let axis = 0; axis < 3; axis++) positions[triangle * 9 + corner * 3 + axis] = center[axis] + random() - 0.5;
        }
    }
    return positions;
}

/** A wavy grid of `size` x `size` quads, indexed with 16-bit indices when they fit. */
export function grid(size: number): TinyBvhInput {
    const positions = new Float32Array((size + 1) * (size + 1) * 3);
    for (let z = 0; z <= size; z++) {
        for (let x = 0; x <= size; x++) positions.set([x * 0.1, Math.sin(x * 0.3) * Math.cos(z * 0.2), z * 0.1], (z * (size + 1) + x) * 3);
    }
    const vertexCount = (size + 1) * (size + 1);
    const indices = vertexCount <= 0xffff ? new Uint16Array(size * size * 6) : new Uint32Array(size * size * 6);
    for (let z = 0; z < size; z++) {
        for (let x = 0; x < size; x++) {
            const corner = z * (size + 1) + x;
            indices.set([corner, corner + 1, corner + size + 1, corner + 1, corner + size + 2, corner + size + 1], (z * size + x) * 6);
        }
    }
    return { positions, indices };
}

export function storageBuffer(device: GPUDevice, label: string, data: Uint32Array | Uint16Array | Float32Array | number): GPUBuffer {
    const size = Math.max(16, typeof data === 'number' ? data : data.byteLength);
    const buffer = device.createBuffer({ label, size, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    if (typeof data !== 'number') device.queue.writeBuffer(buffer, 0, data as Uint32Array<ArrayBuffer>);
    return buffer;
}

export interface TreeCase {
    name: string;
    /** What the checker compares with (CPU arrays). */
    geometry: TinyBvhInput;
    /** What the GPU builds from; the CPU arrays unless the case puts the mesh on the GPU first. */
    input?: GpuLbvhGeometry;
}

export function treeCases(device: GPUDevice): { cases: TreeCase[]; owned: GPUBuffer[] } {
    const owned: GPUBuffer[] = [];
    const cases: TreeCase[] = [
        { name: 'one triangle', geometry: { positions: randomSoup(1, 1) } },
        { name: 'two triangles', geometry: { positions: randomSoup(2, 2) } },
        { name: 'three triangles', geometry: { positions: randomSoup(3, 3) } },
        { name: 'one sort tile', geometry: { positions: randomSoup(2048, 4) } },
        { name: 'one past a tile', geometry: { positions: randomSoup(2049, 5) } },
        { name: 'grid, 16-bit indices', geometry: grid(150) },
        {
            name: 'all triangles equal',
            geometry: { positions: Float32Array.from({ length: 3000 * 9 }, (_, index) => [0, 0, 0, 1, 0, 0, 0, 1, 0][index % 9]) },
        },
        { name: 'flat (no height)', geometry: { positions: randomSoup(4000, 6).map((value, index) => (index % 3 === 1 ? 0 : value)) } },
        { name: 'flags', geometry: { positions: randomSoup(500, 7), triangleFlags: Uint32Array.from({ length: 500 }, (_, index) => index % 3) } },
        { name: 'random soup 300k', geometry: { positions: randomSoup(300_000, 8, 200) } },
        // Nine 16-bit indices: an odd count, padded to whole words on upload.
        { name: 'odd 16-bit index list', geometry: { positions: randomSoup(3, 9), indices: Uint16Array.from({ length: 9 }, (_, index) => 8 - index) } },
    ];

    // The same grid again, but from GPU buffers: interleaved vertices (position, then 3 unused floats) and 32-bit indices.
    const source = grid(120);
    const vertexCount = source.positions.length / 3;
    const interleaved = new Float32Array(vertexCount * 6);
    for (let vertex = 0; vertex < vertexCount; vertex++) interleaved.set(source.positions.subarray(vertex * 3, vertex * 3 + 3), vertex * 6 + 1);
    const indices = Uint32Array.from(source.indices as Uint16Array);
    const positionBuffer = storageBuffer(device, 'interleaved vertices', interleaved);
    const indexBuffer = storageBuffer(device, 'indices', indices);
    owned.push(positionBuffer, indexBuffer);
    cases.push({
        name: 'GPU buffers, interleaved',
        geometry: { positions: source.positions, indices },
        input: { positionBuffer, vertexCount, positionStrideFloats: 6, positionOffsetFloats: 1, indexBuffer, indexFormat: 'uint32', triangleCount: indices.length / 3 },
    });
    // A 16-bit index buffer whose mesh starts at an odd index: one stray index first.
    const shared = grid(40);
    const sharedIndices = shared.indices as Uint16Array;
    const offsetIndices = new Uint16Array(sharedIndices.length + 2);
    offsetIndices[0] = 0xffff;
    offsetIndices.set(sharedIndices, 1);
    const sharedPositions = storageBuffer(device, 'positions', shared.positions);
    const oddIndexBuffer = storageBuffer(device, 'indices at an odd offset', offsetIndices);
    owned.push(sharedPositions, oddIndexBuffer);
    cases.push({
        name: 'GPU buffers, 16-bit indices at an odd offset',
        geometry: shared,
        input: {
            positionBuffer: sharedPositions,
            vertexCount: shared.positions.length / 3,
            indexBuffer: oddIndexBuffer,
            indexFormat: 'uint16',
            indexOffset: 1,
            triangleCount: sharedIndices.length / 3,
        },
    });
    return { cases, owned };
}
