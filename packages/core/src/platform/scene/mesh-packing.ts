/**
 * The layout of one mesh in the vertex buffer, and the CPU packer that fills
 * it: the reference for the GPU packer (`gpu-mesh-packing.ts`, T2.13) and its
 * fallback where the GPU path can't run.
 *
 * Every mesh is checked and laid out on the CPU when it is created
 * ({@link planMesh}), so an invalid mesh throws at the call that made it, not
 * frames later. Since T2.13 the words themselves are usually made on the GPU
 * from the host's raw arrays; {@link packMesh} makes the same words on the CPU.
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

/** Where a mesh's parts go in the vertex buffer, and what the CPU learned from checking it. */
export interface MeshLayout {
    readonly offsets: PackedMeshOffsets;
    /** Words the whole mesh occupies. */
    readonly wordCount: number;
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

/** A mesh as the CPU packs it: its layout and the words to upload, starting at the mesh's offset in the vertex buffer. */
export interface PackedMesh extends MeshLayout {
    readonly words: Uint32Array<ArrayBuffer>;
}

/** {@link planMesh}'s result: the layout, and the material-slot words, which the CPU always makes (they come from `groups`, not from an array). */
export interface MeshPlan {
    readonly layout: MeshLayout;
    /** The packed per-triangle slots (`layout.offsets.materialSlots`'s words), or null when every triangle uses one slot. */
    readonly materialSlotWords: Uint32Array<ArrayBuffer> | null;
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

/**
 * The highest index, in one tight pass. Where it is past the last vertex, a
 * second pass finds the first bad index for the message: the common case
 * (every index valid) costs one comparison per index and no branch that
 * throws (Sponza: 17 M indices on every load).
 */
function checkIndices(indices: Uint16Array | Uint32Array, vertexCount: number, label: string): void {
    // A 16-bit index can't be past the last vertex of a mesh with 65,536 or more.
    if (indices instanceof Uint16Array && vertexCount >= 0x10000) return;
    let highest = 0;
    for (let index = 0; index < indices.length; index++) {
        const vertex = indices[index];
        if (vertex > highest) highest = vertex;
    }
    if (highest < vertexCount) return;
    for (let index = 0; index < indices.length; index++) {
        if (indices[index] >= vertexCount) {
            throw new RangeError(`${label}: index ${index} is vertex ${indices[index]}, but there are ${vertexCount} vertices`);
        }
    }
}

/**
 * The object-space box `[minX, minY, minZ, maxX, maxY, maxZ]`, checking every
 * coordinate is finite. `value * 0` is 0 for a finite value and NaN for an
 * infinite or NaN one, so one sum says whether any coordinate was bad without
 * a test per value; a second pass finds the vertex for the message.
 */
function checkedBounds(positions: Float32Array, label: string): Float32Array {
    let minX = Infinity;
    let minY = Infinity;
    let minZ = Infinity;
    let maxX = -Infinity;
    let maxY = -Infinity;
    let maxZ = -Infinity;
    let finiteCheck = 0;
    for (let at = 0; at < positions.length; at += 3) {
        const x = positions[at];
        const y = positions[at + 1];
        const z = positions[at + 2];
        finiteCheck += x * 0 + y * 0 + z * 0;
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
        if (z < minZ) minZ = z;
        if (z > maxZ) maxZ = z;
    }
    if (finiteCheck !== 0) {
        for (let at = 0; at < positions.length; at++) {
            if (!Number.isFinite(positions[at])) throw new RangeError(`${label}: vertex ${Math.floor(at / 3)} has a non-finite position`);
        }
    }
    return new Float32Array([minX, minY, minZ, maxX, maxY, maxZ]);
}

/**
 * Checks a mesh and lays it out: everything about it except the words of
 * its arrays. Throws a `RangeError` that says what is wrong for an invalid
 * mesh. Reads the arrays (the index check and the box) but keeps none of them.
 */
export function planMesh(descriptor: MeshDescriptor): MeshPlan {
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
    checkIndices(indices, vertexCount, label);
    const bounds = checkedBounds(positions, label);

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

    let materialSlotWords: Uint32Array<ArrayBuffer> | null = null;
    if (slots && offsets.materialSlots !== NONE) {
        materialSlotWords = new Uint32Array(Math.ceil(triangleCount / 2));
        for (let triangle = 0; triangle < triangleCount; triangle++) {
            materialSlotWords[triangle >> 1] |= slots[triangle] << ((triangle & 1) * 16);
        }
    }

    const layout: MeshLayout = {
        offsets,
        wordCount: cursor,
        triangleCount,
        vertexCount,
        indices16,
        constantMaterialSlot,
        materialSlotCount,
        bounds,
        deforming: descriptor.deforming === true,
    };
    return { layout, materialSlotWords };
}

/**
 * The words of a planned mesh, made on the CPU: the reference the GPU packer
 * must match (positions and indices bit for bit, normals within one
 * octahedral step), and its fallback.
 */
export function packMeshWords(descriptor: MeshDescriptor, plan: MeshPlan): Uint32Array<ArrayBuffer> {
    const { layout, materialSlotWords } = plan;
    const { offsets, vertexCount, indices16 } = layout;
    const { positions, indices, normals, uvs, uvs1 } = descriptor;
    const words = new Uint32Array(layout.wordCount);
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
    if (materialSlotWords) words.set(materialSlotWords, offsets.materialSlots);
    return words;
}

/** Checks, lays out and packs a mesh on the CPU (see {@link planMesh} and {@link packMeshWords}). */
export function packMesh(descriptor: MeshDescriptor): PackedMesh {
    const plan = planMesh(descriptor);
    return { ...plan.layout, words: packMeshWords(descriptor, plan) };
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
