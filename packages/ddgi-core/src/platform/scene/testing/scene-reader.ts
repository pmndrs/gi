/**
 * A CPU reading of the two scene buffers, written to mirror the WGSL
 * accessors in wgsl.ts step by step. The unit tests run it over the bytes a
 * fake device received, so the CPU-side layout is checked in Node; the GPU
 * readback page checks the WGSL itself the same way. Test-only.
 */

import {
    HEADER_WORD,
    INSTANCE_FLAG,
    INSTANCE_RECORD_VEC4S,
    INSTANCE_WORD,
    LIGHT_FLAG,
    LIGHT_RECORD_VEC4S,
    LIGHT_WORD,
    MATERIAL_FLAG,
    MATERIAL_RECORD_VEC4S,
    MATERIAL_WORD,
    MESH_FLAG,
    MESH_RECORD_VEC4S,
    MESH_WORD,
    NONE,
    UV_PACK_SHIFT,
} from '../layout.ts';
import { decodeOctahedral } from '../octahedral.ts';
import type { EnvironmentResult, InstanceQuery, InstanceQueryResult, LightResult, MaterialResult, SceneReadback, UvMappingResult } from './readback.ts';

type Vec3 = [number, number, number];

function cross(a: Vec3, b: Vec3): Vec3 {
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}

function subtract(a: Vec3, b: Vec3): Vec3 {
    return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

function unit(v: Vec3): Vec3 {
    const length = Math.hypot(v[0], v[1], v[2]);
    return [v[0] / length, v[1] / length, v[2] / length];
}

export class SceneReader {
    private readonly tableWords: Uint32Array;
    private readonly tableFloats: Float32Array;
    private readonly vertexWords: Uint32Array;
    private readonly vertexFloats: Float32Array;

    constructor(vertices: Uint8Array, tables: Uint8Array) {
        this.vertexWords = new Uint32Array(vertices.buffer, vertices.byteOffset, vertices.byteLength / 4);
        this.vertexFloats = new Float32Array(vertices.buffer, vertices.byteOffset, vertices.byteLength / 4);
        this.tableWords = new Uint32Array(tables.buffer, tables.byteOffset, tables.byteLength / 4);
        this.tableFloats = new Float32Array(tables.buffer, tables.byteOffset, tables.byteLength / 4);
    }

    instanceCount(): number {
        return this.tableWords[HEADER_WORD.instanceSlots];
    }

    private instanceWord(instance: number): number {
        return (this.tableWords[HEADER_WORD.instancesBase] + instance * INSTANCE_RECORD_VEC4S) * 4;
    }

    private meshWord(mesh: number): number {
        return (this.tableWords[HEADER_WORD.meshesBase] + mesh * MESH_RECORD_VEC4S) * 4;
    }

    private instanceFlags(instance: number): number {
        return this.tableWords[this.instanceWord(instance) + INSTANCE_WORD.flags];
    }

    instanceMesh(instance: number): number {
        return this.tableWords[this.instanceWord(instance) + INSTANCE_WORD.mesh];
    }

    triangleCount(instance: number): number {
        if (instance >= this.instanceCount() || (this.instanceFlags(instance) & INSTANCE_FLAG.live) === 0) return 0;
        return this.tableWords[this.meshWord(this.instanceMesh(instance)) + MESH_WORD.triangleCount];
    }

    contributes(instance: number): boolean {
        if (instance >= this.instanceCount()) return false;
        const mask = INSTANCE_FLAG.live | INSTANCE_FLAG.contributesToGi | INSTANCE_FLAG.visible | INSTANCE_FLAG.singular;
        return (this.instanceFlags(instance) & mask) === (INSTANCE_FLAG.live | INSTANCE_FLAG.contributesToGi | INSTANCE_FLAG.visible);
    }

    /** Column-major 4x4. */
    transform(instance: number): number[] {
        const base = this.instanceWord(instance);
        const rows = [INSTANCE_WORD.objectToWorldRow0, INSTANCE_WORD.objectToWorldRow1, INSTANCE_WORD.objectToWorldRow2];
        const matrix = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1];
        rows.forEach((rowWord, row) => {
            for (let column = 0; column < 4; column++) matrix[column * 4 + row] = this.tableFloats[base + rowWord + column];
        });
        return matrix;
    }

    /** Column-major 3x3: the stored rows of inverse(L) are its columns. */
    normalMatrix(instance: number): number[] {
        const base = this.instanceWord(instance);
        return [INSTANCE_WORD.normalRow0, INSTANCE_WORD.normalRow1, INSTANCE_WORD.normalRow2].flatMap((word) =>
            Array.from(this.tableFloats.subarray(base + word, base + word + 3)),
        );
    }

    bounds(instance: number): number[] {
        const base = this.instanceWord(instance);
        return [
            ...this.tableFloats.subarray(base + INSTANCE_WORD.boundsMin, base + INSTANCE_WORD.boundsMin + 3),
            ...this.tableFloats.subarray(base + INSTANCE_WORD.boundsMax, base + INSTANCE_WORD.boundsMax + 3),
        ];
    }

    triangleVertices(mesh: number, triangle: number): [number, number, number] {
        const record = this.meshWord(mesh);
        const indices = this.tableWords[record + MESH_WORD.indices];
        const flags = this.tableWords[record + MESH_WORD.flags];
        const first = 3 * triangle;
        if (flags & MESH_FLAG.indices16) {
            const word = indices + (first >> 1);
            const a = this.vertexWords[word];
            const b = this.vertexWords[word + 1];
            return first & 1 ? [a >>> 16, b & 0xffff, b >>> 16] : [a & 0xffff, a >>> 16, b & 0xffff];
        }
        return [this.vertexWords[indices + first], this.vertexWords[indices + first + 1], this.vertexWords[indices + first + 2]];
    }

    private position(mesh: number, vertex: number): Vec3 {
        const word = this.tableWords[this.meshWord(mesh) + MESH_WORD.positions] + 3 * vertex;
        return [this.vertexFloats[word], this.vertexFloats[word + 1], this.vertexFloats[word + 2]];
    }

    private toWorld(instance: number, point: Vec3): Vec3 {
        const m = this.transform(instance);
        return [0, 1, 2].map((row) => m[row] * point[0] + m[4 + row] * point[1] + m[8 + row] * point[2] + m[12 + row]) as Vec3;
    }

    corners(instance: number, triangle: number): [Vec3, Vec3, Vec3] {
        const mesh = this.instanceMesh(instance);
        const vertices = this.triangleVertices(mesh, triangle);
        return vertices.map((vertex) => this.toWorld(instance, this.position(mesh, vertex))) as [Vec3, Vec3, Vec3];
    }

    geometricNormal(instance: number, triangle: number): Vec3 {
        const [a, b, c] = this.corners(instance, triangle);
        return unit(cross(subtract(b, a), subtract(c, a)));
    }

    normalAt(instance: number, triangle: number, barycentrics: readonly [number, number]): Vec3 {
        const mesh = this.instanceMesh(instance);
        const normals = this.tableWords[this.meshWord(mesh) + MESH_WORD.normals];
        if (normals === NONE) return this.geometricNormal(instance, triangle);
        const weights = [1 - barycentrics[0] - barycentrics[1], barycentrics[0], barycentrics[1]];
        const object: Vec3 = [0, 0, 0];
        this.triangleVertices(mesh, triangle).forEach((vertex, corner) => {
            const decoded = decodeOctahedral(this.vertexWords[normals + vertex]);
            for (let axis = 0; axis < 3; axis++) object[axis] += decoded[axis] * weights[corner];
        });
        const n = this.normalMatrix(instance);
        const world: Vec3 = [0, 1, 2].map((row) => n[row] * object[0] + n[3 + row] * object[1] + n[6 + row] * object[2]) as Vec3;
        return unit(world);
    }

    uvAt(instance: number, triangle: number, barycentrics: readonly [number, number], channel: 0 | 1): [number, number] {
        const mesh = this.instanceMesh(instance);
        const uvs = this.tableWords[this.meshWord(mesh) + (channel === 1 ? MESH_WORD.uvs1 : MESH_WORD.uvs0)];
        if (uvs === NONE) return [0, 0];
        const weights = [1 - barycentrics[0] - barycentrics[1], barycentrics[0], barycentrics[1]];
        const result: [number, number] = [0, 0];
        this.triangleVertices(mesh, triangle).forEach((vertex, corner) => {
            result[0] += this.vertexFloats[uvs + 2 * vertex] * weights[corner];
            result[1] += this.vertexFloats[uvs + 2 * vertex + 1] * weights[corner];
        });
        return result;
    }

    materialAt(instance: number, triangle: number): number {
        const mesh = this.instanceMesh(instance);
        const record = this.meshWord(mesh);
        const slots = this.tableWords[record + MESH_WORD.materialSlots];
        const slot =
            slots === NONE
                ? this.tableWords[record + MESH_WORD.constantMaterialSlot]
                : (this.vertexWords[slots + (triangle >> 1)] >>> ((triangle & 1) * 16)) & 0xffff;
        const base = this.instanceWord(instance);
        const listStart = this.tableWords[base + INSTANCE_WORD.materialListStart];
        const listCount = this.tableWords[base + INSTANCE_WORD.materialListCount];
        const clamped = Math.min(slot, Math.max(listCount, 1) - 1);
        return this.tableWords[this.tableWords[HEADER_WORD.materialListsBase] * 4 + listStart + clamped];
    }

    private uvMapping(matrixWord: number, translationWord: number, packedWord: number): UvMappingResult {
        const packed = this.tableWords[packedWord];
        return {
            transform: [...this.tableFloats.subarray(matrixWord, matrixWord + 4), ...this.tableFloats.subarray(translationWord, translationWord + 2)],
            channel: (packed >>> UV_PACK_SHIFT.channel) & 0xf,
            wrapU: (packed >>> UV_PACK_SHIFT.wrapU) & 0xf,
            wrapV: (packed >>> UV_PACK_SHIFT.wrapV) & 0xf,
        };
    }

    material(material: number): MaterialResult {
        const base = (this.tableWords[HEADER_WORD.materialsBase] + material * MATERIAL_RECORD_VEC4S) * 4;
        const flags = this.tableWords[base + MATERIAL_WORD.flags];
        return {
            albedo: Array.from(this.tableFloats.subarray(base + MATERIAL_WORD.albedo, base + MATERIAL_WORD.albedo + 3)),
            alphaCutoff: this.tableFloats[base + MATERIAL_WORD.alphaCutoff],
            emission: Array.from(this.tableFloats.subarray(base + MATERIAL_WORD.emission, base + MATERIAL_WORD.emission + 3)),
            blocksLight: (flags & MATERIAL_FLAG.blocksLight) !== 0,
            doubleSided: (flags & MATERIAL_FLAG.doubleSided) !== 0,
            albedoTexture: this.tableWords[base + MATERIAL_WORD.albedoTexture],
            albedoUv: this.uvMapping(base + MATERIAL_WORD.albedoUvMatrix, base + MATERIAL_WORD.albedoUvTranslation, base + MATERIAL_WORD.albedoUvPacked),
            emissiveTexture: this.tableWords[base + MATERIAL_WORD.emissiveTexture],
            emissiveUv: this.uvMapping(base + MATERIAL_WORD.emissiveUvMatrix, base + MATERIAL_WORD.emissiveUvTranslation, base + MATERIAL_WORD.emissiveUvPacked),
        };
    }

    light(light: number): LightResult {
        const base = (this.tableWords[HEADER_WORD.lightsBase] + light * LIGHT_RECORD_VEC4S) * 4;
        const vector = (word: number) => Array.from(this.tableFloats.subarray(base + word, base + word + 3));
        return {
            kind: this.tableWords[base + LIGHT_WORD.type],
            position: vector(LIGHT_WORD.position),
            direction: vector(LIGHT_WORD.direction),
            range: this.tableFloats[base + LIGHT_WORD.range],
            intensity: vector(LIGHT_WORD.intensity),
            castsShadows: (this.tableWords[base + LIGHT_WORD.flags] & LIGHT_FLAG.castsShadows) !== 0,
            cosInnerCone: this.tableFloats[base + LIGHT_WORD.cosInnerCone],
            cosOuterCone: this.tableFloats[base + LIGHT_WORD.cosOuterCone],
            radius: this.tableFloats[base + LIGHT_WORD.radius],
            group: this.tableWords[base + LIGHT_WORD.group],
        };
    }

    environment(): EnvironmentResult {
        const floats = this.tableFloats;
        return {
            kind: this.tableWords[HEADER_WORD.environmentType],
            radiance: Array.from(floats.subarray(HEADER_WORD.environmentColor, HEADER_WORD.environmentColor + 3)),
            intensity: floats[HEADER_WORD.environmentIntensity],
            sunDirection: Array.from(floats.subarray(HEADER_WORD.environmentSunDirection, HEADER_WORD.environmentSunDirection + 3)),
            turbidity: floats[HEADER_WORD.environmentTurbidity],
            rotationY: floats[HEADER_WORD.environmentRotationY],
        };
    }

    query(query: InstanceQuery): InstanceQueryResult {
        const triangleCount = this.triangleCount(query.instance);
        const contributes = this.contributes(query.instance);
        if (triangleCount === 0) {
            return {
                triangleCount, contributes, mesh: 0, materialAt: 0, transform: [], normalMatrix: [], bounds: [],
                corners: [], geometricNormal: [], normalAt: [], uv: [], vertices: [],
            };
        }
        const mesh = this.instanceMesh(query.instance);
        return {
            triangleCount,
            contributes,
            mesh,
            materialAt: this.materialAt(query.instance, query.triangle),
            transform: this.transform(query.instance),
            normalMatrix: this.normalMatrix(query.instance),
            bounds: this.bounds(query.instance),
            corners: this.corners(query.instance, query.triangle).flat(),
            geometricNormal: this.geometricNormal(query.instance, query.triangle),
            normalAt: this.normalAt(query.instance, query.triangle, query.barycentrics),
            uv: this.uvAt(query.instance, query.triangle, query.barycentrics, query.channel),
            vertices: this.triangleVertices(mesh, query.triangle),
        };
    }

    readAll(queries: readonly InstanceQuery[], materials: readonly number[]): SceneReadback {
        const lightCount = this.tableWords[HEADER_WORD.lightCount];
        return {
            instanceCount: this.instanceCount(),
            lightCount,
            directionalLightCount: this.tableWords[HEADER_WORD.directionalLightCount],
            environment: this.environment(),
            queries: queries.map((query) => this.query(query)),
            materials: materials.map((material) => this.material(material)),
            lights: Array.from({ length: lightCount }, (_, index) => this.light(index)),
        };
    }
}
