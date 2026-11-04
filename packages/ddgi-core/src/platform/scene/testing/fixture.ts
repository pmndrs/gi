/**
 * Test scenes for the scene copy, and the answers every accessor must give,
 * worked out from the inputs alone (never from the buffers).
 *
 * `SceneModel` wraps a `PlatformScene` and remembers what was asked of it.
 * `checkReadback` compares a readback (from the GPU kernel, or the CPU
 * reader in the unit tests) with that memory. Test-only.
 */

import { DEFAULT_TURBIDITY } from '../../sky/calibrated-sky.ts';
import type { InstanceHandle, LightHandle, MaterialHandle, MeshHandle, TextureHandle } from '../../contract/handles.ts';
import type {
    EnvironmentDescriptor,
    InstanceDescriptor,
    LightChanges,
    LightDescriptor,
    MaterialDescriptor,
    MeshDescriptor,
    UvMapping,
} from '../../contract/scene.ts';
import { createRandom, type Random } from '../../trace/oracle-fixtures.ts';
import { ENVIRONMENT_TYPE, LIGHT_TYPE, NONE, UV_WRAP } from '../layout.ts';
import type { PlatformScene } from '../platform-scene.ts';
import type { InstanceQuery, InstanceQueryResult, MaterialResult, SceneReadback, UvMappingResult } from './readback.ts';

type Vec3 = [number, number, number];

interface ModelInstance {
    mesh: MeshHandle;
    materials: MaterialHandle[];
    transform: number[];
    contributesToGi: boolean;
    visible: boolean;
}

/** A PlatformScene plus a record of everything asked of it. */
export class SceneModel {
    readonly scene: PlatformScene;
    readonly meshes = new Map<MeshHandle, MeshDescriptor>();
    readonly materials = new Map<MaterialHandle, MaterialDescriptor>();
    readonly textures = new Set<TextureHandle>();
    readonly instances = new Map<InstanceHandle, ModelInstance>();
    readonly lights = new Map<LightHandle, LightDescriptor>();
    environment: EnvironmentDescriptor = { type: 'none' };

    constructor(scene: PlatformScene) {
        this.scene = scene;
    }

    createMesh(descriptor: MeshDescriptor): MeshHandle {
        const handle = this.scene.createMesh(descriptor);
        this.meshes.set(handle, descriptor);
        return handle;
    }

    updateMesh(handle: MeshHandle, descriptor: MeshDescriptor): void {
        this.scene.updateMesh(handle, descriptor);
        this.meshes.set(handle, descriptor);
    }

    destroyMesh(handle: MeshHandle): void {
        this.scene.destroyMesh(handle);
        this.meshes.delete(handle);
    }

    createTexture(): TextureHandle {
        const handle = this.scene.createTexture({ source: { data: new Uint8Array([255, 255, 255, 255]), width: 1, height: 1 }, srgb: true });
        this.textures.add(handle);
        return handle;
    }

    createMaterial(descriptor: MaterialDescriptor): MaterialHandle {
        const handle = this.scene.createMaterial(descriptor);
        this.materials.set(handle, { ...descriptor });
        return handle;
    }

    updateMaterial(handle: MaterialHandle, changes: Partial<MaterialDescriptor>): void {
        this.scene.updateMaterial(handle, changes);
        this.materials.set(handle, { ...this.materials.get(handle), ...changes });
    }

    destroyMaterial(handle: MaterialHandle): void {
        this.scene.destroyMaterial(handle);
        this.materials.delete(handle);
    }

    createInstance(descriptor: InstanceDescriptor): InstanceHandle {
        const handle = this.scene.createInstance(descriptor);
        this.instances.set(handle, {
            mesh: descriptor.mesh,
            materials: [...descriptor.materials],
            transform: Array.from(descriptor.transform),
            contributesToGi: descriptor.contributesToGI !== false,
            visible: descriptor.visible !== false,
        });
        return handle;
    }

    setTransform(handle: InstanceHandle, transform: number[]): void {
        this.scene.setTransform(handle, transform);
        this.instance(handle).transform = [...transform];
    }

    setVisible(handle: InstanceHandle, visible: boolean): void {
        this.scene.setVisible(handle, visible);
        this.instance(handle).visible = visible;
    }

    setMaterials(handle: InstanceHandle, materials: MaterialHandle[]): void {
        this.scene.setMaterials(handle, materials);
        this.instance(handle).materials = [...materials];
    }

    destroyInstance(handle: InstanceHandle): void {
        this.scene.destroyInstance(handle);
        this.instances.delete(handle);
    }

    createLight(descriptor: LightDescriptor): LightHandle {
        const handle = this.scene.createLight(descriptor);
        this.lights.set(handle, descriptor);
        return handle;
    }

    updateLight(handle: LightHandle, changes: LightChanges): void {
        this.scene.updateLight(handle, changes);
        this.lights.set(handle, { ...this.lights.get(handle), ...changes } as LightDescriptor);
    }

    destroyLight(handle: LightHandle): void {
        this.scene.destroyLight(handle);
        this.lights.delete(handle);
    }

    setEnvironment(environment: EnvironmentDescriptor): void {
        this.scene.setEnvironment(environment);
        this.environment = environment;
    }

    private instance(handle: InstanceHandle): ModelInstance {
        const instance = this.instances.get(handle);
        if (!instance) throw new Error(`model has no instance ${handle}`);
        return instance;
    }
}

// ---------------------------------------------------------------------------
// Building scenes
// ---------------------------------------------------------------------------

const IDENTITY = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];

/** Translation * rotation (about a unit axis) * scale, column-major. */
export function composeTransform(translation: Vec3, axis: Vec3, angle: number, scale: Vec3): number[] {
    const [x, y, z] = axis;
    const c = Math.cos(angle);
    const s = Math.sin(angle);
    const t = 1 - c;
    const rotation = [
        t * x * x + c, t * x * y + s * z, t * x * z - s * y,
        t * x * y - s * z, t * y * y + c, t * y * z + s * x,
        t * x * z + s * y, t * y * z - s * x, t * z * z + c,
    ];
    return [
        rotation[0] * scale[0], rotation[1] * scale[0], rotation[2] * scale[0], 0,
        rotation[3] * scale[1], rotation[4] * scale[1], rotation[5] * scale[1], 0,
        rotation[6] * scale[2], rotation[7] * scale[2], rotation[8] * scale[2], 0,
        translation[0], translation[1], translation[2], 1,
    ];
}

function randomAxis(random: Random): Vec3 {
    const v: Vec3 = [random() - 0.5, random() - 0.5, random() - 0.5];
    const length = Math.hypot(...v) || 1;
    return [v[0] / length, v[1] / length, v[2] / length];
}

function randomFloats(random: Random, count: number, low: number, high: number): Float32Array {
    return Float32Array.from({ length: count }, () => low + (high - low) * random());
}

/** Unit normals that lean around one direction, so interpolating three never cancels out. */
function leaningNormals(random: Random, count: number, around: Vec3): Float32Array {
    const normals = new Float32Array(count * 3);
    for (let vertex = 0; vertex < count; vertex++) {
        const n: Vec3 = [around[0] + (random() - 0.5) * 0.8, around[1] + (random() - 0.5) * 0.8, around[2] + (random() - 0.5) * 0.8];
        const length = Math.hypot(...n);
        normals.set([n[0] / length, n[1] / length, n[2] / length], vertex * 3);
    }
    return normals;
}

/** A unit cube with corner normals, two uv sets and two material groups (16-bit indices). */
export function cubeMesh(random: Random): MeshDescriptor {
    const positions = new Float32Array([
        -0.5, -0.5, -0.5, 0.5, -0.5, -0.5, 0.5, 0.5, -0.5, -0.5, 0.5, -0.5,
        -0.5, -0.5, 0.5, 0.5, -0.5, 0.5, 0.5, 0.5, 0.5, -0.5, 0.5, 0.5,
    ]);
    const indices = new Uint32Array([
        4, 5, 6, 4, 6, 7, // +z
        1, 0, 3, 1, 3, 2, // -z
        5, 1, 2, 5, 2, 6, // +x
        0, 4, 7, 0, 7, 3, // -x
        7, 6, 2, 7, 2, 3, // +y
        0, 1, 5, 0, 5, 4, // -y
    ]);
    // Corner normals point out of the corner, as a smoothed cube's would.
    const normals = new Float32Array(positions.length);
    for (let vertex = 0; vertex < 8; vertex++) {
        const p = positions.subarray(vertex * 3, vertex * 3 + 3);
        const length = Math.hypot(p[0], p[1], p[2]);
        normals.set([p[0] / length, p[1] / length, p[2] / length], vertex * 3);
    }
    return {
        positions,
        indices,
        normals,
        uvs: randomFloats(random, 16, -2, 3),
        uvs1: randomFloats(random, 16, 0, 1),
        groups: [
            { start: 0, count: 12, materialSlot: 0 },
            { start: 12, count: 24, materialSlot: 1 },
        ],
        label: 'cube',
    };
}

/** More than 65,536 vertices, so indices are stored as 32 bits. No normals or uvs. */
export function wideMesh(random: Random): MeshDescriptor {
    const vertexCount = 70000;
    const positions = randomFloats(random, vertexCount * 3, -3, 3);
    const triangleCount = 41;
    const indices = new Uint32Array(triangleCount * 3);
    for (let index = 0; index < indices.length; index++) {
        // Mostly vertices above 65,535, so a 16-bit mix-up would show.
        indices[index] = index % 5 === 0 ? Math.floor(random() * 65536) : 65536 + Math.floor(random() * (vertexCount - 65536));
    }
    return { positions, indices, label: 'wide' };
}

/** Seven triangles, normals and one uv set; groups leave triangle 3 to slot 0. */
export function fanMesh(random: Random): MeshDescriptor {
    const vertexCount = 9;
    const positions = new Float32Array(vertexCount * 3);
    positions.set([0, 0, 0], 0);
    for (let vertex = 1; vertex < vertexCount; vertex++) {
        const angle = ((vertex - 1) / (vertexCount - 1)) * Math.PI * 1.75;
        positions.set([Math.cos(angle), Math.sin(angle), 0.2 * random()], vertex * 3);
    }
    const indices = new Uint16Array(7 * 3);
    for (let triangle = 0; triangle < 7; triangle++) indices.set([0, triangle + 1, triangle + 2], triangle * 3);
    return {
        positions,
        indices,
        normals: leaningNormals(random, vertexCount, [0, 0, 1]),
        uvs: randomFloats(random, vertexCount * 2, 0, 1),
        groups: [
            { start: 0, count: 9, materialSlot: 1 },
            { start: 12, count: 9, materialSlot: 2 },
        ],
        label: 'fan',
    };
}

/** One triangle, only the second uv set, every triangle in slot 2 (stored as a constant). */
export function singleTriangleMesh(random: Random): MeshDescriptor {
    return {
        positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]),
        indices: new Uint16Array([0, 1, 2]),
        uvs1: randomFloats(random, 6, -1, 1),
        groups: [{ start: 0, count: 3, materialSlot: 2 }],
        label: 'single',
    };
}

/** Small random triangle soup with normals and uvs, for churn. */
export function soupMesh(random: Random, triangleCount: number): MeshDescriptor {
    const vertexCount = triangleCount * 3;
    const center: Vec3 = [random() * 10 - 5, random() * 4, random() * 10 - 5];
    const positions = new Float32Array(vertexCount * 3);
    for (let vertex = 0; vertex < vertexCount; vertex++) {
        positions.set([center[0] + random() - 0.5, center[1] + random() - 0.5, center[2] + random() - 0.5], vertex * 3);
    }
    const indices = Uint32Array.from({ length: vertexCount }, (_, index) => index);
    return {
        positions,
        indices,
        normals: leaningNormals(random, vertexCount, randomAxis(random)),
        uvs: randomFloats(random, vertexCount * 2, 0, 4),
        label: 'soup',
    };
}

export interface BasicScene {
    readonly meshes: { cube: MeshHandle; wide: MeshHandle; fan: MeshHandle; single: MeshHandle };
    readonly materials: MaterialHandle[];
    readonly instances: InstanceHandle[];
    readonly lights: LightHandle[];
}

/**
 * Every kind of thing the scene copy stores: 16- and 32-bit indices, odd and
 * even triangle counts, meshes with and without normals and each uv set,
 * constant and per-triangle material slots, plain, rotated, non-uniformly
 * scaled, mirrored, hidden, non-contributing and singular instances, every
 * light type, textures with uv mappings, and a procedural sky.
 */
export function buildBasicScene(model: SceneModel, seed = 1): BasicScene {
    const random = createRandom(seed);
    const cube = model.createMesh(cubeMesh(random));
    const wide = model.createMesh(wideMesh(random));
    const fan = model.createMesh(fanMesh(random));
    const single = model.createMesh(singleTriangleMesh(random));

    const albedoTexture = model.createTexture();
    const emissiveTexture = model.createTexture();
    const turn = 0.3;
    const materials = [
        model.createMaterial({ label: 'default' }),
        model.createMaterial({ albedo: [0.2, 0.5, 0.8], emissive: [1, 0.5, 0], emissiveIntensity: 3, doubleSided: false, label: 'glowing' }),
        model.createMaterial({
            albedo: [0.9, 0.1, 0.1],
            blocksLight: false,
            alphaCutoff: 0.5,
            albedoTexture,
            albedoUv: { channel: 1, matrix: [2, 0, 0, 3, 0.5, 0.25], wrapU: 'mirror', wrapV: 'clamp' },
            label: 'glass',
        }),
        model.createMaterial({
            emissiveTexture,
            emissiveUv: { matrix: [Math.cos(turn), Math.sin(turn), -Math.sin(turn), Math.cos(turn), 0.1, 0.2] },
            label: 'sign',
        }),
        model.createMaterial({ albedo: [0.05, 0.05, 0.05], label: 'dark' }),
    ];
    const [m0, m1, m2, m3, m4] = materials;

    const instances = [
        model.createInstance({ mesh: cube, materials: [m0, m1], transform: IDENTITY, label: 'cube' }),
        model.createInstance({ mesh: cube, materials: [m2, m3], transform: composeTransform([3, 1, -2], randomAxis(random), 0.7, [2, 0.5, 1.5]), label: 'cube turned' }),
        model.createInstance({ mesh: fan, materials: [m0, m1, m2], transform: composeTransform([-4, 0, 1], [0, 1, 0], 1.1, [-1.5, 1, 2]), label: 'fan mirrored' }),
        model.createInstance({ mesh: wide, materials: [m4], transform: composeTransform([0, 5, 0], [1, 0, 0], 0.2, [1, 1, 1]), label: 'wide' }),
        model.createInstance({ mesh: single, materials: [m0, m1, m3], transform: composeTransform([1, 2, 3], [0, 0, 1], 0.5, [3, 3, 3]), visible: false, label: 'hidden' }),
        model.createInstance({ mesh: cube, materials: [m1, m1], transform: composeTransform([0, -3, 0], [0, 1, 0], 0, [1, 1, 1]), contributesToGI: false, label: 'receiver' }),
        model.createInstance({ mesh: cube, materials: [m0, m1], transform: composeTransform([5, 5, 5], [0, 1, 0], 0, [1, 0, 1]), label: 'flattened (singular)' }),
        model.createInstance({ mesh: fan, materials: [m3, m4, m0], transform: composeTransform([2, -1, 4], randomAxis(random), 2.1, [0.7, 1.3, 0.9]), label: 'fan turned' }),
    ];

    const lights = [
        model.createLight({ type: 'directional', color: [1, 0.9, 0.8], intensity: 3, direction: [0.3, -1, 0.2], label: 'sun' }),
        model.createLight({ type: 'point', color: [1, 1, 1], intensity: 20, position: [0, 3, 0], range: 0, group: 1, label: 'unlimited' }),
        model.createLight({
            type: 'spot', color: [0.5, 0.6, 1], intensity: 50, position: [2, 4, 2], direction: [0, -1, 0.1], range: 10,
            innerConeAngle: 0.3, outerConeAngle: 0.6, radius: 0.05, group: 2, label: 'spot',
        }),
        model.createLight({ type: 'directional', color: [0.2, 0.2, 0.3], intensity: 0.5, direction: [-0.5, -1, 0], angularRadius: 0.01, castsShadows: false, label: 'moon' }),
        model.createLight({ type: 'point', color: [1, 0.4, 0.2], intensity: 5, position: [-2, 1, 3], range: 5, radius: 0.1, castsShadows: false, group: 3, label: 'candle' }),
    ];
    model.setEnvironment({ type: 'procedural', sunDirection: [0.3, 1, 0.2], turbidity: 4, intensity: 2 });

    return { meshes: { cube, wide, fan, single }, materials, instances, lights };
}

/** Edits after the first upload: moves, hides, swaps materials, reshapes a mesh, removes a light. */
export function editBasicScene(model: SceneModel, basic: BasicScene): void {
    const random = createRandom(7);
    model.setTransform(basic.instances[0], composeTransform([0.5, 0.5, 0.5], [0, 1, 0], 0.4, [1, 2, 1]));
    model.setVisible(basic.instances[4], true);
    model.setMaterials(basic.instances[3], [basic.materials[1]]);
    model.updateMaterial(basic.materials[4], { albedo: [0.3, 0.3, 0.1], doubleSided: false });
    model.updateMesh(basic.meshes.single, { ...singleTriangleMesh(random), uvs: randomFloats(random, 6, 0, 1) });
    model.destroyLight(basic.lights[0]);
    model.updateLight(basic.lights[2], { intensity: 70, group: 0 });
    model.setEnvironment({ type: 'color', color: [0.4, 0.5, 0.6], intensity: 1.5 });
}

/**
 * Many creates and destroys with the tables kept small, so every buffer is
 * reallocated at least once. Afterwards the basic scene's objects are still
 * there, plus whatever survived.
 */
export function churnScene(model: SceneModel, rounds: number, seed = 11, afterRound?: () => void): void {
    const random = createRandom(seed);
    const pick = <T>(items: readonly T[]): T => items[Math.floor(random() * items.length)];
    for (let round = 0; round < rounds; round++) {
        const meshes = Array.from({ length: 20 }, () => model.createMesh(soupMesh(random, 5 + Math.floor(random() * 60))));
        const materials = Array.from({ length: 12 }, () =>
            model.createMaterial({ albedo: [random(), random(), random()], emissive: [random(), 0, 0], emissiveIntensity: random() * 4 }),
        );
        const instances = Array.from({ length: 40 }, () =>
            model.createInstance({
                mesh: pick(meshes),
                // Lists of 1 to 3 materials: soups use slot 0 only, extra entries are allowed.
                materials: Array.from({ length: 1 + Math.floor(random() * 3) }, () => pick(materials)),
                transform: composeTransform([random() * 20 - 10, random() * 5, random() * 20 - 10], randomAxis(random), random() * 6, [
                    0.5 + random(), 0.5 + random(), 0.5 + random(),
                ]),
                visible: random() > 0.1,
                contributesToGI: random() > 0.1,
            }),
        );
        const lights = Array.from({ length: 10 }, (_, index) =>
            model.createLight(
                index % 3 === 0
                    ? { type: 'directional', color: [random(), random(), random()], intensity: random() * 4, direction: randomAxis(random), group: index % 4 }
                    : { type: 'point', color: [1, 1, 1], intensity: random() * 40, position: [random() * 10, random() * 4, random() * 10], range: index % 2 ? 0 : 4 + random() * 6, group: index % 4 },
            ),
        );
        // Edits.
        for (const instance of instances.slice(0, 10)) model.setTransform(instance, composeTransform([random() * 10, 0, random() * 10], [0, 1, 0], random(), [1, 1, 1]));
        model.setMaterials(instances[0], [pick(materials), pick(materials)]);
        model.updateMesh(meshes[1], soupMesh(random, 90));
        model.updateMaterial(materials[2], { albedo: [0.5, 0.5, 0.5] });
        afterRound?.();
        // Remove about half, instances before what they use.
        const keptInstances = new Set(instances.filter(() => random() > 0.5));
        for (const instance of instances) if (!keptInstances.has(instance)) model.destroyInstance(instance);
        const used = (handle: MeshHandle | MaterialHandle, kind: 'mesh' | 'material') =>
            [...keptInstances].some((instance) => {
                const entry = model.instances.get(instance);
                return entry !== undefined && (kind === 'mesh' ? entry.mesh === handle : entry.materials.includes(handle as MaterialHandle));
            });
        for (const mesh of meshes) if (!used(mesh, 'mesh') && random() > 0.3) model.destroyMesh(mesh);
        for (const material of materials) if (!used(material, 'material') && random() > 0.3) model.destroyMaterial(material);
        for (const light of lights) if (random() > 0.5) model.destroyLight(light);
    }
}

// ---------------------------------------------------------------------------
// Queries
// ---------------------------------------------------------------------------

/** A few points on a few triangles of every instance slot, free slots included. */
export function makeQueries(model: SceneModel, seed = 3): InstanceQuery[] {
    const random = createRandom(seed);
    const queries: InstanceQuery[] = [];
    const slotCount = Math.max(0, ...[...model.instances.keys()].map((handle) => model.scene.instanceSlot(handle) + 1)) + 2;
    const slotToInstance = new Map([...model.instances.keys()].map((handle) => [model.scene.instanceSlot(handle), handle]));
    for (let slot = 0; slot < slotCount; slot++) {
        const handle = slotToInstance.get(slot);
        const triangles = handle === undefined ? 1 : model.meshes.get(model.instances.get(handle)!.mesh)!.indices.length / 3;
        const picks = new Set([0, triangles - 1, Math.floor(triangles / 2), Math.floor(random() * triangles)]);
        for (const triangle of picks) {
            const u = random();
            const v = random() * (1 - u);
            queries.push({ instance: slot, triangle, barycentrics: [u, v], channel: queries.length % 2 === 0 ? 0 : 1 });
        }
    }
    return queries;
}

/** Material indices to read back: every live material. */
export function materialIndices(model: SceneModel): number[] {
    return [...model.materials.keys()].map((handle) => model.scene.materialSlot(handle));
}

// ---------------------------------------------------------------------------
// Expected values and comparison
// ---------------------------------------------------------------------------

function close(actual: number, expected: number, tolerance: number): boolean {
    return Math.abs(actual - expected) <= tolerance * Math.max(1, Math.abs(expected));
}

function compareArrays(failures: string[], what: string, actual: readonly number[], expected: readonly number[], tolerance: number): void {
    if (actual.length !== expected.length) {
        failures.push(`${what}: got ${actual.length} values, expected ${expected.length}`);
        return;
    }
    for (let index = 0; index < expected.length; index++) {
        if (!close(actual[index], expected[index], tolerance)) {
            failures.push(`${what}: got [${actual.map((value) => value.toPrecision(6))}], expected [${expected.map((value) => value.toPrecision(6))}]`);
            return;
        }
    }
}

function transformPoint(m: readonly number[], p: ArrayLike<number>): Vec3 {
    return [0, 1, 2].map((row) => m[row] * p[0] + m[4 + row] * p[1] + m[8 + row] * p[2] + m[12 + row]) as Vec3;
}

/** transpose(inverse(L)) of the linear part, column-major 3x3, or null when singular. */
function normalMatrixOf(m: readonly number[]): number[] | null {
    // L[row][column] = m[column * 4 + row]
    const a = m[0], b = m[4], c = m[8];
    const d = m[1], e = m[5], f = m[9];
    const g = m[2], h = m[6], i = m[10];
    const determinant = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
    if (Math.abs(determinant) < 1e-12) return null;
    // inverse(L)[row][column] = cofactor(column, row) / det; its transpose is the cofactor matrix / det.
    const cofactor = [
        [e * i - f * h, -(d * i - f * g), d * h - e * g],
        [-(b * i - c * h), a * i - c * g, -(a * h - b * g)],
        [b * f - c * e, -(a * f - c * d), a * e - b * d],
    ];
    // N = transpose(inverse(L)) = cofactor / det, so N[row][column] = cofactor[row][column] / det.
    const result = new Array<number>(9);
    for (let column = 0; column < 3; column++) for (let row = 0; row < 3; row++) result[column * 3 + row] = cofactor[row][column] / determinant;
    return result;
}

function unit(v: readonly number[]): number[] {
    const length = Math.hypot(v[0], v[1], v[2]);
    return [v[0] / length, v[1] / length, v[2] / length];
}

function crossOf(a: readonly number[], b: readonly number[]): number[] {
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}

/** The slot the groups give a triangle: later groups win, uncovered triangles use slot 0. */
function triangleSlot(mesh: MeshDescriptor, triangle: number): number {
    let slot = 0;
    for (const group of mesh.groups ?? []) {
        if (triangle * 3 >= group.start && triangle * 3 < group.start + group.count) slot = group.materialSlot;
    }
    return slot;
}

/** The material index `giMaterialAt` must return for a triangle of an instance, from the inputs. */
export function expectedMaterialAt(model: SceneModel, handle: InstanceHandle, triangle: number): number {
    const instance = model.instances.get(handle)!;
    const mesh = model.meshes.get(instance.mesh)!;
    const slot = Math.min(triangleSlot(mesh, triangle), instance.materials.length - 1);
    return model.scene.materialSlot(instance.materials[slot]);
}

/** The live instance in a slot, if any. */
export function instanceAtSlot(model: SceneModel, slot: number): InstanceHandle | undefined {
    return [...model.instances.keys()].find((candidate) => model.scene.instanceSlot(candidate) === slot);
}

function checkQuery(failures: string[], model: SceneModel, query: InstanceQuery, actual: InstanceQueryResult): void {
    const label = `instance ${query.instance} triangle ${query.triangle}`;
    const handle = instanceAtSlot(model, query.instance);
    if (handle === undefined) {
        if (actual.triangleCount !== 0) failures.push(`${label}: free slot reports ${actual.triangleCount} triangles`);
        if (actual.contributes) failures.push(`${label}: free slot reports it contributes`);
        return;
    }
    const instance = model.instances.get(handle)!;
    const mesh = model.meshes.get(instance.mesh)!;
    const m = instance.transform;
    const normalMatrix = normalMatrixOf(m);

    if (actual.triangleCount !== mesh.indices.length / 3) failures.push(`${label}: triangleCount ${actual.triangleCount}, expected ${mesh.indices.length / 3}`);
    const contributes = instance.visible && instance.contributesToGi && normalMatrix !== null;
    if (actual.contributes !== contributes) failures.push(`${label}: contributes ${actual.contributes}, expected ${contributes}`);
    if (actual.mesh !== model.scene.meshSlot(instance.mesh)) failures.push(`${label}: mesh ${actual.mesh}, expected ${model.scene.meshSlot(instance.mesh)}`);
    compareArrays(failures, `${label} transform`, actual.transform, m, 1e-6);

    const vertices = [0, 1, 2].map((corner) => mesh.indices[query.triangle * 3 + corner]);
    compareArrays(failures, `${label} vertices`, actual.vertices, vertices, 0);

    const expectedMaterial = expectedMaterialAt(model, handle, query.triangle);
    if (actual.materialAt !== expectedMaterial) failures.push(`${label}: materialAt ${actual.materialAt}, expected ${expectedMaterial}`);

    // Bounds: the box around the mesh's object-space box once transformed (as
    // the tracer's top level has it), at most slightly padded.
    const objectBox = [Infinity, Infinity, Infinity, -Infinity, -Infinity, -Infinity];
    for (let vertex = 0; vertex < mesh.positions.length / 3; vertex++) {
        for (let axis = 0; axis < 3; axis++) {
            objectBox[axis] = Math.min(objectBox[axis], mesh.positions[vertex * 3 + axis]);
            objectBox[axis + 3] = Math.max(objectBox[axis + 3], mesh.positions[vertex * 3 + axis]);
        }
    }
    const worldBox = [Infinity, Infinity, Infinity, -Infinity, -Infinity, -Infinity];
    for (let corner = 0; corner < 8; corner++) {
        const point = [0, 1, 2].map((axis) => objectBox[corner & (1 << axis) ? axis + 3 : axis]);
        const world = transformPoint(m, point);
        for (let axis = 0; axis < 3; axis++) {
            worldBox[axis] = Math.min(worldBox[axis], world[axis]);
            worldBox[axis + 3] = Math.max(worldBox[axis + 3], world[axis]);
        }
    }
    compareArrays(failures, `${label} bounds`, actual.bounds, worldBox, 1e-5);

    const corners = vertices.map((vertex) => transformPoint(m, mesh.positions.subarray(vertex * 3, vertex * 3 + 3)));
    compareArrays(failures, `${label} corners`, actual.corners, corners.flat(), 1e-5);

    const uvSet = query.channel === 1 ? mesh.uvs1 : mesh.uvs;
    const weights = [1 - query.barycentrics[0] - query.barycentrics[1], query.barycentrics[0], query.barycentrics[1]];
    const uv = [0, 0];
    if (uvSet) {
        vertices.forEach((vertex, corner) => {
            uv[0] += uvSet[vertex * 2] * weights[corner];
            uv[1] += uvSet[vertex * 2 + 1] * weights[corner];
        });
    }
    compareArrays(failures, `${label} uv${query.channel}`, actual.uv, uv, 1e-5);

    if (normalMatrix === null) return; // a flattened instance has no normals to check
    compareArrays(failures, `${label} normal matrix`, actual.normalMatrix, normalMatrix, 1e-5);
    const winding = unit(crossOf(
        [corners[1][0] - corners[0][0], corners[1][1] - corners[0][1], corners[1][2] - corners[0][2]],
        [corners[2][0] - corners[0][0], corners[2][1] - corners[0][1], corners[2][2] - corners[0][2]],
    ));
    // The authored front side (DECISIONS 2026-11-04): a mirror reverses the world-space winding,
    // so its winding normal is turned back. Mirrored means a negative determinant of the linear part.
    const mirrored = m[0] * (m[5] * m[10] - m[9] * m[6]) - m[4] * (m[1] * m[10] - m[9] * m[2]) + m[8] * (m[1] * m[6] - m[5] * m[2]) < 0;
    const geometric = mirrored ? [-winding[0], -winding[1], -winding[2]] : winding;
    compareArrays(failures, `${label} geometric normal`, actual.geometricNormal, geometric, 1e-4);

    let shading = geometric;
    if (mesh.normals) {
        const object = [0, 0, 0];
        vertices.forEach((vertex, corner) => {
            for (let axis = 0; axis < 3; axis++) object[axis] += mesh.normals![vertex * 3 + axis] * weights[corner];
        });
        shading = unit([0, 1, 2].map((row) => normalMatrix[row] * object[0] + normalMatrix[3 + row] * object[1] + normalMatrix[6 + row] * object[2]));
    }
    // Octahedral storage moves a normal by about 1e-4 radians at most.
    const cosine = actual.normalAt[0] * shading[0] + actual.normalAt[1] * shading[1] + actual.normalAt[2] * shading[2];
    if (!(cosine > 1 - 1e-6)) failures.push(`${label}: normalAt [${actual.normalAt}] is ${Math.acos(Math.min(1, cosine))} rad from [${shading}]`);
}

function expectedUv(mapping: UvMapping | undefined): UvMappingResult {
    return {
        transform: [...(mapping?.matrix ?? [1, 0, 0, 1, 0, 0])],
        channel: mapping?.channel ?? 0,
        wrapU: UV_WRAP[mapping?.wrapU ?? 'repeat'],
        wrapV: UV_WRAP[mapping?.wrapV ?? 'repeat'],
    };
}

function checkMaterial(failures: string[], descriptor: MaterialDescriptor, actual: MaterialResult, label: string): void {
    compareArrays(failures, `${label} albedo`, actual.albedo, [...(descriptor.albedo ?? [1, 1, 1])], 1e-6);
    const intensity = descriptor.emissiveIntensity ?? 1;
    compareArrays(failures, `${label} emission`, actual.emission, (descriptor.emissive ?? [0, 0, 0]).map((value) => value * intensity), 1e-6);
    if (!close(actual.alphaCutoff, descriptor.alphaCutoff ?? 0, 1e-6)) failures.push(`${label}: alphaCutoff ${actual.alphaCutoff}`);
    if (actual.blocksLight !== (descriptor.blocksLight !== false)) failures.push(`${label}: blocksLight ${actual.blocksLight}`);
    if (actual.doubleSided !== (descriptor.doubleSided !== false)) failures.push(`${label}: doubleSided ${actual.doubleSided}`);
    if ((actual.albedoTexture !== NONE) !== (descriptor.albedoTexture !== undefined)) failures.push(`${label}: albedoTexture ${actual.albedoTexture}`);
    if ((actual.emissiveTexture !== NONE) !== (descriptor.emissiveTexture !== undefined)) failures.push(`${label}: emissiveTexture ${actual.emissiveTexture}`);
    for (const [name, mapping, result] of [['albedoUv', descriptor.albedoUv, actual.albedoUv], ['emissiveUv', descriptor.emissiveUv, actual.emissiveUv]] as const) {
        const expected = expectedUv(mapping);
        compareArrays(failures, `${label} ${name}`, result.transform, expected.transform, 1e-6);
        if (result.channel !== expected.channel || result.wrapU !== expected.wrapU || result.wrapV !== expected.wrapV) {
            failures.push(`${label} ${name}: channel/wrap ${result.channel}/${result.wrapU}/${result.wrapV}, expected ${expected.channel}/${expected.wrapU}/${expected.wrapV}`);
        }
    }
}

function checkLights(failures: string[], model: SceneModel, readback: SceneReadback): void {
    const lights = [...model.lights];
    const directional = lights.filter(([, light]) => light.type === 'directional').length;
    if (readback.lightCount !== lights.length) failures.push(`lightCount ${readback.lightCount}, expected ${lights.length}`);
    if (readback.directionalLightCount !== directional) failures.push(`directionalLightCount ${readback.directionalLightCount}, expected ${directional}`);
    for (const [handle, light] of lights) {
        const index = model.scene.lightIndex(handle);
        const actual = readback.lights[index];
        const label = `light ${index} (${light.label ?? light.type})`;
        if (!actual) {
            failures.push(`${label}: not read back`);
            continue;
        }
        if ((light.type === 'directional') !== index < directional) failures.push(`${label}: directional lights must come first`);
        if (actual.kind !== LIGHT_TYPE[light.type]) failures.push(`${label}: kind ${actual.kind}`);
        compareArrays(failures, `${label} intensity`, actual.intensity, light.color.map((value) => value * light.intensity), 1e-6);
        if (actual.group !== (light.group ?? 0)) failures.push(`${label}: group ${actual.group}`);
        if (actual.castsShadows !== (light.castsShadows !== false)) failures.push(`${label}: castsShadows ${actual.castsShadows}`);
        if (light.type !== 'point') compareArrays(failures, `${label} direction`, actual.direction, unit(light.direction), 1e-6);
        if (light.type === 'directional') {
            if (!close(actual.radius, light.angularRadius ?? 0, 1e-6)) failures.push(`${label}: angular radius ${actual.radius}`);
            continue;
        }
        compareArrays(failures, `${label} position`, actual.position, [...light.position], 1e-6);
        if (!close(actual.range, light.range, 1e-6)) failures.push(`${label}: range ${actual.range}`);
        if (!close(actual.radius, light.radius ?? 0, 1e-6)) failures.push(`${label}: radius ${actual.radius}`);
        const cosInner = light.type === 'spot' ? Math.cos(light.innerConeAngle) : -1;
        const cosOuter = light.type === 'spot' ? Math.cos(light.outerConeAngle) : -1;
        if (!close(actual.cosInnerCone, cosInner, 1e-6) || !close(actual.cosOuterCone, cosOuter, 1e-6)) {
            failures.push(`${label}: cone cosines ${actual.cosInnerCone}, ${actual.cosOuterCone}`);
        }
    }
}

function checkEnvironment(failures: string[], environment: EnvironmentDescriptor, readback: SceneReadback): void {
    const actual = readback.environment;
    if (actual.kind !== ENVIRONMENT_TYPE[environment.type]) failures.push(`environment kind ${actual.kind}, expected ${environment.type}`);
    if (environment.type === 'color') {
        const intensity = environment.intensity ?? 1;
        compareArrays(failures, 'environment radiance', actual.radiance, environment.color.map((value) => value * intensity), 1e-6);
    } else if (environment.type === 'procedural') {
        compareArrays(failures, 'sun direction', actual.sunDirection, unit(environment.sunDirection), 1e-6);
        if (!close(actual.turbidity, environment.turbidity ?? DEFAULT_TURBIDITY, 1e-6)) failures.push(`turbidity ${actual.turbidity}`);
        if (!close(actual.intensity, environment.intensity ?? 1, 1e-6)) failures.push(`environment intensity ${actual.intensity}`);
    }
}

/** Every difference between a readback and what the inputs say, as readable lines. Empty means it all matches. */
export function checkReadback(model: SceneModel, queries: readonly InstanceQuery[], materials: readonly number[], readback: SceneReadback): string[] {
    const failures: string[] = [];
    const slotCount = Math.max(0, ...[...model.instances.keys()].map((handle) => model.scene.instanceSlot(handle) + 1));
    if (readback.instanceCount !== slotCount) failures.push(`instanceCount ${readback.instanceCount}, expected ${slotCount}`);
    queries.forEach((query, index) => checkQuery(failures, model, query, readback.queries[index]));
    const materialBySlot = new Map([...model.materials].map(([handle, descriptor]) => [model.scene.materialSlot(handle), descriptor]));
    materials.forEach((slot, index) => checkMaterial(failures, materialBySlot.get(slot)!, readback.materials[index], `material ${slot}`));
    checkLights(failures, model, readback);
    checkEnvironment(failures, model.environment, readback);
    return failures;
}
