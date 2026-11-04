/**
 * Turns a `MeshDescriptor` into the exact words the vertex buffer holds for
 * that mesh, after checking it.
 *
 * Packing happens when the mesh is created, so the host may reuse or free its
 * arrays straight after the call, and an invalid mesh throws at the call that
 * made it rather than frames later.
 *
 * Word layout of one mesh (offsets relative to the mesh's first word; see layout.ts):
 *
 *     indices        3 per triangle, 16-bit (two per word) when the mesh has at most 65,536 vertices
 *     positions      3 floats per vertex
 *     normals        1 word per vertex (octahedral), when given
 *     uvs (set 0)    2 floats per vertex, when given
 *     uvs (set 1)    2 floats per vertex, when given
 *     materialSlots  16-bit slot per triangle (two per word), when the triangles use more than one slot
 */

import type { MeshDescriptor } from '../contract/scene.ts';
import {
    MAX_MATERIAL_SLOTS,
    MAX_VERTICES_FOR_16_BIT_INDICES,
    NONE,
    NORMAL_WORDS_PER_VERTEX,
    POSITION_WORDS_PER_VERTEX,
    UV_WORDS_PER_VERTEX,
} from './layout.ts';
import { encodeOctahedral } from './octahedral.ts';

/** Where each part of a packed mesh starts, in words from the mesh's first word. NONE when absent. */
export interface PackedMeshOffsets {
    readonly indices: number;
    readonly positions: number;
    readonly normals: number;
    readonly uvs0: number;
    readonly uvs1: number;
    readonly materialSlots: number;
}

export interface PackedMesh {
    /** The words to upload, starting at the mesh's offset in the vertex buffer. */
    readonly words: Uint32Array<ArrayBuffer>;
    readonly offsets: PackedMeshOffsets;
    readonly triangleCount: number;
    readonly vertexCount: number;
    readonly indices16: boolean;
    /** The slot every triangle uses, when they all use the same one; otherwise NONE. */
    readonly constantMaterialSlot: number;
    /** Materials an instance must supply: the highest slot used, plus one. */
    readonly materialSlotCount: number;
    /** Object-space box `[minX, minY, minZ, maxX, maxY, maxZ]`. */
    readonly bounds: Float32Array;
    readonly deforming: boolean;
}

function describe(descriptor: MeshDescriptor): string {
    return descriptor.label ? `mesh "${descriptor.label}"` : 'mesh';
}

function checkPerVertex(name: string, array: Float32Array | undefined, components: number, vertexCount: number, label: string): void {
    if (array === undefined) return;
    if (array.length !== vertexCount * components) {
        throw new RangeError(`${label}: ${name} has ${array.length} floats, expected ${components} per vertex (${vertexCount * components})`);
    }
}

/** Per-triangle material slots from the groups. Triangles no group covers use slot 0. */
function triangleSlots(descriptor: MeshDescriptor, triangleCount: number, label: string): Uint16Array | null {
    const groups = descriptor.groups;
    if (!groups || groups.length === 0) return null;
    const slots = new Uint16Array(triangleCount);
    const indexCount = descriptor.indices.length;
    for (const group of groups) {
        const { start, count, materialSlot } = group;
        if (!Number.isInteger(start) || !Number.isInteger(count) || start < 0 || count < 0 || start % 3 !== 0 || count % 3 !== 0) {
            throw new RangeError(`${label}: group start and count must be non-negative multiples of 3 (got ${start}, ${count})`);
        }
        if (start + count > indexCount) {
            throw new RangeError(`${label}: group [${start}, ${start + count}) runs past the ${indexCount} indices`);
        }
        if (!Number.isInteger(materialSlot) || materialSlot < 0 || materialSlot >= MAX_MATERIAL_SLOTS) {
            throw new RangeError(`${label}: material slot ${materialSlot} must be an integer in [0, ${MAX_MATERIAL_SLOTS})`);
        }
        // Later groups win where groups overlap, as a renderer drawing them in order would show.
        slots.fill(materialSlot, start / 3, (start + count) / 3);
    }
    return slots;
}

export function packMesh(descriptor: MeshDescriptor): PackedMesh {
    const label = describe(descriptor);
    const { positions, indices, normals, uvs, uvs1 } = descriptor;

    if (positions.length % 3 !== 0) throw new RangeError(`${label}: positions must hold three floats per vertex`);
    const vertexCount = positions.length / 3;
    if (indices.length % 3 !== 0) throw new RangeError(`${label}: indices must be a multiple of 3 (a triangle list)`);
    const triangleCount = indices.length / 3;
    if (triangleCount === 0) throw new RangeError(`${label}: a mesh needs at least one triangle`);
    checkPerVertex('normals', normals, 3, vertexCount, label);
    checkPerVertex('uvs', uvs, 2, vertexCount, label);
    checkPerVertex('uvs1', uvs1, 2, vertexCount, label);

    for (let index = 0; index < indices.length; index++) {
        if (indices[index] >= vertexCount) {
            throw new RangeError(`${label}: index ${index} is vertex ${indices[index]}, but there are ${vertexCount} vertices`);
        }
    }

    const bounds = new Float32Array([Infinity, Infinity, Infinity, -Infinity, -Infinity, -Infinity]);
    for (let vertex = 0; vertex < vertexCount; vertex++) {
        for (let axis = 0; axis < 3; axis++) {
            const value = positions[vertex * 3 + axis];
            if (!Number.isFinite(value)) throw new RangeError(`${label}: vertex ${vertex} has a non-finite position`);
            if (value < bounds[axis]) bounds[axis] = value;
            if (value > bounds[axis + 3]) bounds[axis + 3] = value;
        }
    }

    const slots = triangleSlots(descriptor, triangleCount, label);
    let constantMaterialSlot = 0;
    let materialSlotCount = 1;
    if (slots) {
        constantMaterialSlot = slots[0];
        let highest = 0;
        for (let triangle = 0; triangle < triangleCount; triangle++) {
            if (slots[triangle] !== constantMaterialSlot) constantMaterialSlot = NONE;
            if (slots[triangle] > highest) highest = slots[triangle];
        }
        // An instance supplies one material per slot the groups name, even a slot
        // that an overlapping group hides completely.
        for (const group of descriptor.groups ?? []) highest = Math.max(highest, group.materialSlot);
        materialSlotCount = highest + 1;
    }

    // Lay out the parts one after another.
    const indices16 = vertexCount <= MAX_VERTICES_FOR_16_BIT_INDICES;
    let cursor = 0;
    const take = (words: number): number => {
        const offset = cursor;
        cursor += words;
        return offset;
    };
    const offsets: PackedMeshOffsets = {
        indices: take(indices16 ? Math.ceil(indices.length / 2) : indices.length),
        positions: take(vertexCount * POSITION_WORDS_PER_VERTEX),
        normals: normals ? take(vertexCount * NORMAL_WORDS_PER_VERTEX) : NONE,
        uvs0: uvs ? take(vertexCount * UV_WORDS_PER_VERTEX) : NONE,
        uvs1: uvs1 ? take(vertexCount * UV_WORDS_PER_VERTEX) : NONE,
        materialSlots: constantMaterialSlot === NONE ? take(Math.ceil(triangleCount / 2)) : NONE,
    };

    const words = new Uint32Array(cursor);
    const floats = new Float32Array(words.buffer);

    if (indices16) {
        // Index i sits in word i / 2: even indices in the low half, odd ones in the high half.
        for (let index = 0; index < indices.length; index++) {
            const word = offsets.indices + (index >> 1);
            words[word] |= (indices[index] & 0xffff) << ((index & 1) * 16);
        }
    } else {
        words.set(indices, offsets.indices);
    }
    floats.set(positions, offsets.positions);
    if (normals) {
        for (let vertex = 0; vertex < vertexCount; vertex++) {
            words[offsets.normals + vertex] = encodeOctahedral(normals[vertex * 3], normals[vertex * 3 + 1], normals[vertex * 3 + 2]);
        }
    }
    if (uvs) floats.set(uvs, offsets.uvs0);
    if (uvs1) floats.set(uvs1, offsets.uvs1);
    if (slots && offsets.materialSlots !== NONE) {
        for (let triangle = 0; triangle < triangleCount; triangle++) {
            words[offsets.materialSlots + (triangle >> 1)] |= slots[triangle] << ((triangle & 1) * 16);
        }
    }

    return {
        words,
        offsets,
        triangleCount,
        vertexCount,
        indices16,
        constantMaterialSlot,
        materialSlotCount,
        bounds,
        deforming: descriptor.deforming === true,
    };
}

/** Reads triangle `triangle`'s three vertex indices back out of packed words (tests, and the tracer's BVH build). */
export function packedTriangleVertices(mesh: PackedMesh, triangle: number): [number, number, number] {
    const result: [number, number, number] = [0, 0, 0];
    for (let corner = 0; corner < 3; corner++) {
        const index = triangle * 3 + corner;
        result[corner] = mesh.indices16
            ? (mesh.words[mesh.offsets.indices + (index >> 1)] >>> ((index & 1) * 16)) & 0xffff
            : mesh.words[mesh.offsets.indices + index];
    }
    return result;
}

/** The mesh's positions as floats (a view of the packed words, no copy). */
export function packedPositions(mesh: PackedMesh): Float32Array {
    return new Float32Array(mesh.words.buffer, mesh.words.byteOffset + mesh.offsets.positions * 4, mesh.vertexCount * 3);
}

/** All vertex indices as 32-bit values, in triangle order (a copy). For building BVHs. */
export function packedIndices(mesh: PackedMesh): Uint32Array {
    if (!mesh.indices16) return mesh.words.slice(mesh.offsets.indices, mesh.offsets.indices + mesh.triangleCount * 3);
    const result = new Uint32Array(mesh.triangleCount * 3);
    for (let index = 0; index < result.length; index++) {
        result[index] = (mesh.words[mesh.offsets.indices + (index >> 1)] >>> ((index & 1) * 16)) & 0xffff;
    }
    return result;
}
