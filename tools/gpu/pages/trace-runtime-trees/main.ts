/**
 * T3.2 (docs/design/m3-moving.md §2.2 and "Draft done-whens"): mesh trees for
 * meshes added or replaced while the scene runs. Their trees are built on the
 * GPU (H-PLOC) in the frame they are uploaded and placed after a small stats
 * readback; nothing replaces them later (#26: tinybvh is only the fallback).
 *
 * Per scene (the cabinet row; the furnished room too when it is downloaded),
 * two platforms in turn, each traced through `rays.encodeBatch`:
 *
 * - `tinybvh-only` (`gpuFirst: false`): the fallback, M1's path, the
 *   reference for memory and trace speed. Run first;
 * - `gpu` (the defaults, what hosts get): every mesh on its GPU tree.
 *
 * Steps, each with a reference worked out on the CPU before the change:
 *
 * 1. **add** (done-when item 2): a new hollow cabinet (a new mesh and a new
 *    instance) next to the scene's furniture. Frames run at a 60 Hz pace;
 *    after each, CHECK_RAYS rays aimed at the cabinet are traced and checked
 *    against the oracle. The cabinet must pass within SEEN_WITHIN_FRAMES
 *    frames, counting the frame that uploads it as 1, and keep passing.
 * 2. **replace** (item 1): `updateMesh` on that cabinet's mesh, which now has
 *    an open front with a shelf, with its instance unchanged. The same bar.
 * 3. **freed** (item 3, since #26 about the replaced tree rather than a
 *    swap): within FREED_WITHIN_FRAMES frames of the frame the replace was
 *    first seen, the tracer's triangles in use must equal `tinybvh-only`'s in
 *    the same scene (every builder writes one record per triangle, so nothing
 *    of the cabinet's old tree is left) and the GPU build's own buffers must
 *    be gone. Nodes in use are recorded (the builders' trees differ).
 * 4. **sheet** (recorded, not judged): a 100,352-triangle bumpy sheet added,
 *    where the worker's build is long enough for the GPU's tree to matter:
 *    frames until it passes the oracle, its memory judged as in 3, and the
 *    trace speed of rays aimed at it on the GPU tree against `tinybvh-only`
 *    (TASKS M3: timing is recorded, not gated).
 *
 * Recompiles must stay 0 throughout. Each check's rays are also compared with
 * the scene before the change, and must disagree on at least
 * MIN_CHANGED_SHARE of them, so a check can fail.
 *
 * Aimed rays stop just past their target (PAST_TARGET), so only geometry near
 * the change can be hit: the brute-force reference then needs only the
 * triangles whose boxes come near it (the furnished room has 270k triangles).
 *
 * `VITE_RUNTIME_TREE_SCENES=a,b` picks scenes. Not a timing page: run it under
 * the bench lock anyway (`node tools/reference/bench-lock.ts run --no-cool-wait
 * --what "T3.2" -- pnpm test:gpu --gpu nvidia --filter trace-runtime-trees`).
 */

import type { FrameInput } from '../../../../packages/core/src/platform/contract/frame.ts';
import type { MaterialHandle, MeshHandle } from '../../../../packages/core/src/platform/contract/handles.ts';
import { GIPlatformImpl } from '../../../../packages/core/src/platform/gi-platform.ts';
import { HIT_RECORD_BYTES, packRays, RAY_BYTES, type MeshGeometry, type TraceRay, type TraceSceneInstance } from '../../../../packages/core/src/platform/trace/index.ts';
import type { TracedHit } from '../../../../packages/core/src/platform/trace/oracle-compare.ts';
import { createRandom, randomUnitVector, tessellatedQuad, toFloat32Ray, type Random } from '../../../../packages/core/src/platform/trace/oracle-fixtures.ts';
import type { AccelerationMemory, MeshTreeCounts, MeshTreeOptions } from '../../../../packages/core/src/platform/trace/ray-queries.ts';
import { buildProcedural } from '../../../../scenes/src/procedural/index.ts';
import { assert, runGpuTest } from '../../client.ts';
import {
    buildReferenceScene,
    traceAnyReference,
    traceClosestReference,
    type ReferenceScene,
} from '../../../../packages/core/src/platform/trace/reference-tracer.ts';
import { boxMesh, unpackRay, type OracleCase } from '../trace-oracle/cases.ts';
import { evaluateCase, type ReferenceAnswers } from '../trace-oracle/check.ts';
import { readBuffer } from '../tlas-oracle/top-level-check.ts';
import { IDENTITY_MATRIX } from '../trace-bench/gltf-geometry.ts';
import { findManifest, loadBenchScene } from '../trace-bench/scene-source.ts';

const env = import.meta.env as Record<string, unknown>;
/** Whether the furnished room's model is on disk; the glob is lazy, so nothing is loaded to find out. */
const FURNISHED_ROOM_DOWNLOADED = Object.keys(import.meta.glob('../../../../assets/downloads/furnished-room/*.glb')).length > 0;
const SCENES =
    typeof env.VITE_RUNTIME_TREE_SCENES === 'string' && env.VITE_RUNTIME_TREE_SCENES.trim() !== ''
        ? env.VITE_RUNTIME_TREE_SCENES.split(',').map((id) => id.trim())
        : ['cabinet-row', ...(FURNISHED_ROOM_DOWNLOADED ? ['furnished-room'] : [])];

const SEED = 0x7e32;
/** A host drawing at 60 Hz. */
const FRAME_MS = 1000 / 60;
/** The done-whens' bars. */
const SEEN_WITHIN_FRAMES = 4;
const FREED_WITHIN_FRAMES = 2;
/** Frames given to a change before the page gives up on it (1.5 s), and frames it must keep passing after it first did. */
const MAX_FRAMES = 90;
const STAY_FRAMES = 6;
const CHECK_RAYS = 2000;
/** The CPU reference yields to the browser after this many rays. */
const REFERENCE_YIELD_RAYS = 50;
const SPEED_RAYS = 1 << 18;
const SPEED_REPEATS = 15;
/** Aimed rays start this far from their target (metres) and meet its triangle no flatter than MIN_AIM_COSINE (see scene-deformed-positions). */
const AIM_DISTANCE: readonly [number, number] = [0.4, 1.5];
const MIN_AIM_COSINE = 0.1;
/** Aimed rays end this far past their target, as a share of the distance to it. */
const PAST_TARGET = 1.05;
/**
 * Rays may disagree with the reference this often, if the reference marks
 * them ambiguous. Higher than the oracle's random cases (1%): every ray here
 * is aimed at a cabinet of 18 mm boards, so many land within the reference's
 * edge tolerance of an edge two faces share, a real tie (the first run read
 * 1.25% with every tree settled). A tracer with a hole still fails on the
 * mismatches, which are never excused.
 */
const AMBIGUOUS_SHARE = 0.03;
/** A check's rays must answer differently before and after its change on at least this share. */
const MIN_CHANGED_SHARE = 0.1;
/** The boards of the cabinets, 18 mm as in the cabinet row. */
const PANEL = 0.018;
/** The sheet: 2 × SHEET_CELLS² triangles over a square of 2 × SHEET_HALF metres, bumps of ± SHEET_BUMP. */
const SHEET_CELLS = 224;
const SHEET_HALF = 0.8;
const SHEET_BUMP = 0.15;
const HIT_RECORD_WORDS = HIT_RECORD_BYTES / 4;
/** Reported for a hit on a slot that holds none of the page's instances, so the checks fail it. */
const UNKNOWN_SLOT = 0xffffffff;

const IDLE_FRAME: FrameInput = {
    deltaTime: FRAME_MS / 1000,
    budget: { units: 0 },
    camera: { position: [0, 0, 0], view: IDENTITY_MATRIX, projection: IDENTITY_MATRIX, near: 0.1, far: 100 },
};

type Vec3 = [number, number, number];
type Variant = 'tinybvh-only' | 'gpu';
const VARIANTS: readonly Variant[] = ['tinybvh-only', 'gpu'];
const MESH_TREES: Record<Variant, MeshTreeOptions> = {
    'tinybvh-only': { gpuFirst: false },
    gpu: {},
};

// ---------------------------------------------------------------------------
// Geometry
// ---------------------------------------------------------------------------

/** Boxes (min and max corners) merged into one mesh. */
function mergeBoxes(boxes: readonly (readonly [Vec3, Vec3])[]): MeshGeometry {
    const positions: number[] = [];
    const indices: number[] = [];
    for (const [min, max] of boxes) {
        const box = boxMesh([(max[0] - min[0]) / 2, (max[1] - min[1]) / 2, (max[2] - min[2]) / 2]);
        const firstVertex = positions.length / 3;
        for (let at = 0; at < box.positions.length; at += 3) {
            positions.push(box.positions[at] + (min[0] + max[0]) / 2, box.positions[at + 1] + (min[1] + max[1]) / 2, box.positions[at + 2] + (min[2] + max[2]) / 2);
        }
        for (const index of box.indices!) indices.push(index + firstVertex);
    }
    return { positions: new Float32Array(positions), indices: new Uint32Array(indices) };
}

/**
 * A hollow cabinet of 18 mm boards as one mesh, its outer box centred on the
 * origin with the front at +z: with a door, or open with a shelf halfway up.
 * Boards meet without overlapping, so no two faces lie on top of each other
 * where a ray can reach them (that would make ties the oracle can't decide).
 */
function hollowCabinet(size: Vec3, front: 'door' | 'open-with-shelf'): MeshGeometry {
    const [w, h, d] = [size[0] / 2, size[1] / 2, size[2] / 2];
    // The carcass stops behind the door when there is one.
    const carcassFront = front === 'door' ? d - PANEL : d;
    const boards: [Vec3, Vec3][] = [
        [[-w, -h, -d], [w, -h + PANEL, carcassFront]], // bottom
        [[-w, h - PANEL, -d], [w, h, carcassFront]], // top
        [[-w, -h + PANEL, -d], [-w + PANEL, h - PANEL, carcassFront]], // left side
        [[w - PANEL, -h + PANEL, -d], [w, h - PANEL, carcassFront]], // right side
        [[-w + PANEL, -h + PANEL, -d], [w - PANEL, h - PANEL, -d + PANEL]], // back
    ];
    if (front === 'door') boards.push([[-w, -h, d - PANEL], [w, h, d]]);
    else boards.push(cabinetShelf(size));
    return mergeBoxes(boards);
}

/** The open cabinet's shelf, halfway up, set back 2 cm from the front. */
function cabinetShelf(size: Vec3): [Vec3, Vec3] {
    const [w, d] = [size[0] / 2, size[2] / 2];
    return [[-w + PANEL, -PANEL / 2, -d + PANEL], [w - PANEL, PANEL / 2, d - 0.02]];
}

/** A square sheet in its own xy plane with smooth bumps in z: big enough that tinybvh's build takes a while. */
function bumpySheet(): MeshGeometry {
    const sheet = tessellatedQuad(SHEET_CELLS, SHEET_HALF);
    for (let at = 0; at < sheet.positions.length; at += 3) {
        sheet.positions[at + 2] = SHEET_BUMP * Math.sin(7 * sheet.positions[at]) * Math.cos(5 * sheet.positions[at + 1]);
    }
    return sheet;
}

function translation(point: Vec3): number[] {
    return [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, point[0], point[1], point[2], 1];
}

/** Lies a mesh's xy plane flat (its z becomes world y), centred at `point`. Column-major. */
function lyingFlatAt(point: Vec3): number[] {
    return [1, 0, 0, 0, 0, 0, -1, 0, 0, 1, 0, 0, point[0], point[1], point[2], 1];
}

function transformPoint(matrix: ArrayLike<number>, point: readonly number[]): Vec3 {
    return [0, 1, 2].map((row) => matrix[row] * point[0] + matrix[4 + row] * point[1] + matrix[8 + row] * point[2] + matrix[12 + row]) as Vec3;
}

/** World box (min xyz, max xyz) of a mesh placed by `objectToWorld`: its local box's corners, transformed. */
function worldBox(mesh: MeshGeometry, objectToWorld: ArrayLike<number>): number[] {
    const local = [Infinity, Infinity, Infinity, -Infinity, -Infinity, -Infinity];
    for (let at = 0; at < mesh.positions.length; at += 3) {
        for (let axis = 0; axis < 3; axis++) {
            local[axis] = Math.min(local[axis], mesh.positions[at + axis]);
            local[axis + 3] = Math.max(local[axis + 3], mesh.positions[at + axis]);
        }
    }
    const box = [Infinity, Infinity, Infinity, -Infinity, -Infinity, -Infinity];
    for (let corner = 0; corner < 8; corner++) {
        const point = transformPoint(objectToWorld, [corner & 1 ? local[3] : local[0], corner & 2 ? local[4] : local[1], corner & 4 ? local[5] : local[2]]);
        for (let axis = 0; axis < 3; axis++) {
            box[axis] = Math.min(box[axis], point[axis]);
            box[axis + 3] = Math.max(box[axis + 3], point[axis]);
        }
    }
    return box;
}

function boxesOverlap(a: readonly number[], b: readonly number[]): boolean {
    return [0, 1, 2].every((axis) => a[axis] <= b[axis + 3] && b[axis] <= a[axis + 3]);
}

// ---------------------------------------------------------------------------
// Scenes and checks
// ---------------------------------------------------------------------------

interface PlacedMesh {
    readonly meshIndex: number;
    readonly objectToWorld: number[];
}

interface SceneGeometry {
    readonly id: string;
    /** The scene's meshes, then the closed cabinet, the open cabinet and the sheet. */
    readonly meshes: MeshGeometry[];
    readonly instances: PlacedMesh[];
    readonly triangles: number;
    readonly cabinetAt: number[];
    readonly sheetAt: number[];
    readonly closedCabinet: number;
    readonly openCabinet: number;
    /** The open cabinet's shelf alone, in the cabinet's own space: what the replace check's rays aim at. */
    readonly shelf: MeshGeometry;
    readonly sheet: number;
}

/**
 * Where each scene's new cabinet stands, by its box's centre, and its size.
 * Cabinet row: the next base unit along the run (x 3.05-3.65 m), against the
 * back wall (z = 0), as tall as the others with their kick space. Furnished
 * room: where `ddgi-move-cabinet` stands its cabinet, on the floor against
 * the north wall (from the scene's world box).
 */
const CABINETS: Record<string, { readonly size: Vec3; readonly centre: (bounds: { min: Vec3; max: Vec3 }) => Vec3 }> = {
    'cabinet-row': { size: [0.6, 0.82, 0.58], centre: () => [3.35, 0.411, 0.295] },
    'furnished-room': { size: [0.9, 0.85, 0.45], centre: (bounds) => [-1.0, bounds.min[1] + 0.425 + 0.002, bounds.min[2] + 0.225 + 0.03] },
};

async function loadSceneGeometry(id: string): Promise<SceneGeometry> {
    const manifest = findManifest(id);
    const cabinet = CABINETS[id];
    assert(cabinet, `no cabinet place for scene '${id}' (known: ${Object.keys(CABINETS).join(', ')})`);
    const meshes: MeshGeometry[] = [];
    const instances: PlacedMesh[] = [];
    if (manifest.parts.every((part) => part.kind === 'procedural')) {
        for (const part of manifest.parts) {
            if (part.kind !== 'procedural') continue;
            const data = buildProcedural(part.builder, part.options);
            const offset = meshes.length;
            meshes.push(...data.meshes.map((mesh) => ({ positions: mesh.positions, indices: mesh.indices })));
            instances.push(...data.instances.map((instance) => ({ meshIndex: instance.mesh + offset, objectToWorld: [...instance.transform] })));
        }
    } else {
        const bench = await loadBenchScene(id, 'manifest');
        meshes.push(...bench.meshes);
        instances.push(...bench.instances.map((instance) => ({ meshIndex: instance.meshIndex, objectToWorld: [...instance.objectToWorld] })));
    }
    const bounds = { min: [Infinity, Infinity, Infinity] as Vec3, max: [-Infinity, -Infinity, -Infinity] as Vec3 };
    for (const instance of instances) {
        const box = worldBox(meshes[instance.meshIndex], instance.objectToWorld);
        for (let axis = 0; axis < 3; axis++) {
            bounds.min[axis] = Math.min(bounds.min[axis], box[axis]);
            bounds.max[axis] = Math.max(bounds.max[axis], box[axis + 3]);
        }
    }
    const triangles = instances.reduce((sum, instance) => sum + (meshes[instance.meshIndex].indices?.length ?? 0) / 3, 0);
    const closedCabinet = meshes.push(hollowCabinet(cabinet.size, 'door')) - 1;
    const openCabinet = meshes.push(hollowCabinet(cabinet.size, 'open-with-shelf')) - 1;
    const sheet = meshes.push(bumpySheet()) - 1;
    const volumeCentre = manifest.volume.center as Vec3;
    const shelf = mergeBoxes([cabinetShelf(cabinet.size)]);
    return { id, meshes, instances, triangles, cabinetAt: translation(cabinet.centre(bounds)), sheetAt: lyingFlatAt(volumeCentre), closedCabinet, openCabinet, shelf, sheet };
}

type StepName = 'add' | 'replace' | 'sheet';

/** One change, checked by rays aimed at what it puts in the scene. */
interface Check {
    readonly step: StepName;
    readonly rays: Float32Array;
    /** The scene after the change, with only the instances near the rays (see the file comment). */
    readonly oracleCase: OracleCase;
    readonly reference: ReferenceAnswers;
    /** Share of the rays whose closest hit differs from the scene before the change. */
    readonly changedShare: number;
}

/** The scene's instances (case ids 0..n-1) plus the page's: the cabinet is id n, the sheet n + 1. */
function sceneState(scene: SceneGeometry, cabinetMesh: number | null, withSheet: boolean): TraceSceneInstance[] {
    const instances: TraceSceneInstance[] = scene.instances.map((instance, id) => ({ meshIndex: instance.meshIndex, instanceIndex: id, objectToWorld: instance.objectToWorld }));
    if (cabinetMesh !== null) instances.push({ meshIndex: cabinetMesh, instanceIndex: scene.instances.length, objectToWorld: scene.cabinetAt });
    if (withSheet) instances.push({ meshIndex: scene.sheet, instanceIndex: scene.instances.length + 1, objectToWorld: scene.sheetAt });
    return instances;
}

/**
 * `count` rays aimed at random points on `target`'s triangles (placed by
 * `objectToWorld`), from AIM_DISTANCE away in a direction no flatter than
 * MIN_AIM_COSINE to the triangle, each ending PAST_TARGET of the way past its
 * target (or never, with `endless`, for timing).
 */
function aimedRays(random: Random, target: MeshGeometry, objectToWorld: ArrayLike<number>, count: number, endless = false): TraceRay[] {
    const indices = target.indices!;
    const triangleCount = indices.length / 3;
    const rays: TraceRay[] = [];
    for (let index = 0; index < count; index++) {
        const triangle = Math.floor(random() * triangleCount);
        let u = random();
        let v = random();
        if (u + v > 1) {
            u = 1 - u;
            v = 1 - v;
        }
        const corner = (which: number): Vec3 => {
            const vertex = indices[triangle * 3 + which];
            return transformPoint(objectToWorld, [target.positions[vertex * 3], target.positions[vertex * 3 + 1], target.positions[vertex * 3 + 2]]);
        };
        const [a, b, c] = [corner(0), corner(1), corner(2)];
        const point = [0, 1, 2].map((axis) => a[axis] + u * (b[axis] - a[axis]) + v * (c[axis] - a[axis]));
        const edge1 = [0, 1, 2].map((axis) => b[axis] - a[axis]);
        const edge2 = [0, 1, 2].map((axis) => c[axis] - a[axis]);
        const normal = [edge1[1] * edge2[2] - edge1[2] * edge2[1], edge1[2] * edge2[0] - edge1[0] * edge2[2], edge1[0] * edge2[1] - edge1[1] * edge2[0]];
        const normalLength = Math.hypot(normal[0], normal[1], normal[2]);
        let away = randomUnitVector(random);
        while (Math.abs(away[0] * normal[0] + away[1] * normal[1] + away[2] * normal[2]) < MIN_AIM_COSINE * normalLength) away = randomUnitVector(random);
        const distance = AIM_DISTANCE[0] + random() * (AIM_DISTANCE[1] - AIM_DISTANCE[0]);
        const origin = [0, 1, 2].map((axis) => point[axis] + away[axis] * distance);
        rays.push(toFloat32Ray(origin, [-away[0], -away[1], -away[2]], 0, endless ? 1e6 : distance * PAST_TARGET));
    }
    return rays;
}

/**
 * The brute-force reference over only the triangles whose world box meets
 * `region`, keeping each triangle's own index. Exact for rays that stay
 * inside the region (see the file comment). Hands control back to the
 * browser every REFERENCE_YIELD_RAYS rays: the test runner gives up on a
 * page that is busy for 30 s, and the furnished room has 270k triangles.
 */
async function nearbyReferenceAnswers(oracleCase: OracleCase, region: readonly number[]): Promise<ReferenceAnswers> {
    const full = buildReferenceScene(oracleCase.meshes, oracleCase.instances);
    const kept: number[] = [];
    for (let triangle = 0; triangle < full.triangleCount; triangle++) {
        const base = triangle * 9;
        let overlaps = true;
        for (let axis = 0; axis < 3 && overlaps; axis++) {
            // Corners a, a + edge1, a + edge2.
            const a = full.geometry[base + axis];
            const b = a + full.geometry[base + 3 + axis];
            const c = a + full.geometry[base + 6 + axis];
            overlaps = Math.max(a, b, c) >= region[axis] && Math.min(a, b, c) <= region[axis + 3];
        }
        if (overlaps) kept.push(triangle);
    }
    const scene: ReferenceScene = {
        triangleCount: kept.length,
        geometry: new Float64Array(kept.length * 9),
        instanceIndices: new Uint32Array(kept.length),
        triangleIndices: new Uint32Array(kept.length),
        mirrored: new Uint8Array(kept.length),
        // M4's material walk: keep each kept triangle's material and corner uvs (6 numbers per uv set).
        materials: kept.map((triangle) => full.materials[triangle]),
        uvs: [subsetUvs(full.uvs[0], kept), subsetUvs(full.uvs[1], kept)],
    };
    kept.forEach((triangle, slot) => {
        scene.geometry.set(full.geometry.subarray(triangle * 9, triangle * 9 + 9), slot * 9);
        scene.instanceIndices[slot] = full.instanceIndices[triangle];
        scene.triangleIndices[slot] = full.triangleIndices[triangle];
        scene.mirrored[slot] = full.mirrored[triangle];
    });
    const rayCount = oracleCase.rays.length / 8;
    const closest: ReferenceAnswers['closest'][number][] = [];
    const any: ReferenceAnswers['any'][number][] = [];
    for (let index = 0; index < rayCount; index++) {
        const ray = unpackRay(oracleCase.rays, index);
        closest.push(traceClosestReference(scene, ray));
        any.push(traceAnyReference(scene, ray));
        if (index % REFERENCE_YIELD_RAYS === REFERENCE_YIELD_RAYS - 1) await new Promise((resolve) => setTimeout(resolve, 0));
    }
    return { closest, any };
}

/** Whether two closest-hit answers differ: one hits and the other doesn't, or another instance, or another distance. */
function closestDiffers(before: ReferenceAnswers['closest'][number], after: ReferenceAnswers['closest'][number]): boolean {
    if (!before || !after) return before !== after;
    return before.instanceIndex !== after.instanceIndex || Math.abs(before.t - after.t) > 1e-4 * Math.max(1, after.t);
}

/**
 * A check of one change: rays aimed at instance `targetId` after it (or at
 * `aimAt` placed as that instance, a part of its mesh), the reference after
 * the change, and how many rays answer differently before it.
 */
async function makeCheck(
    random: Random,
    scene: SceneGeometry,
    step: StepName,
    before: TraceSceneInstance[],
    after: TraceSceneInstance[],
    targetId: number,
    aimAt?: MeshGeometry,
): Promise<Check> {
    const target = after.find((instance) => instance.instanceIndex === targetId)!;
    const targetMesh = aimAt ?? scene.meshes[target.meshIndex];
    const rays = packRays(aimedRays(random, targetMesh, target.objectToWorld, CHECK_RAYS));
    // Only instances whose boxes come within reach of the rays: they all start within AIM_DISTANCE of the target and stop just past it.
    const reach = AIM_DISTANCE[1] * PAST_TARGET + 0.05;
    const targetBox = worldBox(targetMesh, target.objectToWorld);
    const region = targetBox.map((value, index) => (index < 3 ? value - reach : value + reach));
    const near = (instances: TraceSceneInstance[]) => instances.filter((instance) => boxesOverlap(worldBox(scene.meshes[instance.meshIndex], instance.objectToWorld), region));
    const caseOf = (instances: TraceSceneInstance[], name: string): OracleCase => ({
        name,
        description: `${scene.id}: ${step}`,
        meshes: scene.meshes,
        instances: near(instances),
        rays,
        maxAmbiguousShare: { closest: AMBIGUOUS_SHARE, any: AMBIGUOUS_SHARE },
    });
    const oracleCase = caseOf(after, `${scene.id} ${step}`);
    const reference = await nearbyReferenceAnswers(oracleCase, region);
    const referenceBefore = await nearbyReferenceAnswers(caseOf(before, `${scene.id} before ${step}`), region);
    const changed = reference.closest.filter((hit, index) => closestDiffers(referenceBefore.closest[index], hit)).length;
    return { step, rays, oracleCase, reference, changedShare: changed / reference.closest.length };
}

// ---------------------------------------------------------------------------
// A platform and its frames
// ---------------------------------------------------------------------------

interface Run {
    readonly variant: Variant;
    readonly device: GPUDevice;
    readonly platform: GIPlatformImpl;
    readonly material: MaterialHandle;
    /** Case id by the scene copy's instance slot. */
    readonly caseIdBySlot: Map<number, number>;
    frame(): void;
}

async function startRun(device: GPUDevice, scene: SceneGeometry, variant: Variant): Promise<Run> {
    const platform = new GIPlatformImpl(device, { label: `runtime trees ${scene.id} ${variant}`, rayBatches: true, meshTrees: MESH_TREES[variant] });
    const material = platform.scene.createMaterial({ albedo: [0.8, 0.8, 0.8] });
    const caseIdBySlot = new Map<number, number>();
    // The scene's own meshes come before the page's extras (the cabinets and the sheet).
    const sceneMeshes: MeshHandle[] = scene.meshes.slice(0, scene.closedCabinet).map((mesh) => platform.scene.createMesh({ positions: mesh.positions, indices: mesh.indices! }));
    scene.instances.forEach((instance, id) => {
        const handle = platform.scene.createInstance({ mesh: sceneMeshes[instance.meshIndex], materials: [material], transform: instance.objectToWorld });
        caseIdBySlot.set(platform.sceneCopy.instanceSlot(handle), id);
    });
    await platform.prepare();
    const run: Run = {
        variant,
        device,
        platform,
        material,
        caseIdBySlot,
        frame: () => {
            const encoder = device.createCommandEncoder({ label: 'runtime trees frame' });
            platform.encode(encoder, IDLE_FRAME);
            device.queue.submit([encoder.finish()]);
        },
    };
    run.frame();
    await settle(run);
    return run;
}

/** Frames until no mesh waits for a tree and no tinybvh build runs. */
async function settle(run: Run): Promise<void> {
    const rays = run.platform.rays;
    for (let frame = 0; ; frame++) {
        assert(frame < 200, `${run.variant}: the mesh trees still build after 200 frames`);
        await rays.whenBuilt();
        run.frame();
        await run.device.queue.onSubmittedWorkDone();
        if (!rays.building && rays.meshTreeCounts.tinybvhBuilding === 0) break;
    }
    // One more, for a top-level shape that arrived with the last trees.
    run.frame();
    await run.device.queue.onSubmittedWorkDone();
}

async function waitUntil(time: number): Promise<void> {
    const wait = time - performance.now();
    if (wait > 0) await new Promise((resolve) => setTimeout(resolve, wait));
}

/** Traces both modes through the platform; hits carry case ids in place of scene slots. */
async function trace(run: Run, packedRays: Float32Array): Promise<{ closest: (TracedHit | null)[]; any: boolean[] }> {
    const { device, platform } = run;
    const count = packedRays.length / 8;
    const rays = device.createBuffer({ label: 'runtime trees rays', size: count * RAY_BYTES, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(rays, 0, packedRays as Float32Array<ArrayBuffer>);
    const hits = device.createBuffer({ label: 'runtime trees hits', size: count * HIT_RECORD_BYTES, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const blocked = device.createBuffer({ label: 'runtime trees blocked', size: count * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const encoder = device.createCommandEncoder({ label: 'runtime trees trace' });
    platform.rays.encodeBatch(encoder, { mode: 'closest', rays, hits, count });
    platform.rays.encodeBatch(encoder, { mode: 'any', rays, hits: blocked, count });
    device.queue.submit([encoder.finish()]);
    const hitData = await readBuffer(device, hits, count * HIT_RECORD_BYTES);
    const blockedWords = new Uint32Array(await readBuffer(device, blocked, count * 4));
    for (const buffer of [rays, hits, blocked]) buffer.destroy();
    const floats = new Float32Array(hitData);
    const words = new Uint32Array(hitData);
    const closest: (TracedHit | null)[] = [];
    const any: boolean[] = [];
    for (let index = 0; index < count; index++) {
        const base = index * HIT_RECORD_WORDS;
        const t = floats[base];
        closest.push(
            t >= 0
                ? {
                      hit: true,
                      t,
                      instanceIndex: run.caseIdBySlot.get(words[base + 1]) ?? UNKNOWN_SLOT,
                      triangleIndex: words[base + 2],
                      frontFace: words[base + 3] !== 0,
                      u: floats[base + 4],
                      v: floats[base + 5],
                  }
                : null,
        );
        any.push(blockedWords[index] !== 0);
    }
    return { closest, any };
}

/** The oracle's failures for a check traced now (empty when it passes). */
async function checkNow(run: Run, check: Check): Promise<string[]> {
    const answers = await trace(run, check.rays);
    const lookup = { closest: (index: number) => answers.closest[index], any: (index: number) => answers.any[index] };
    return evaluateCase(check.oracleCase, lookup, check.reference).failures;
}

interface SeenResult {
    /** The first frame after the change whose rays passed the oracle (the uploading frame is 1), or null. */
    readonly firstSeenFrame: number | null;
    /** Milliseconds from the change to that frame's answers. */
    readonly firstSeenMs: number | null;
    /** The mesh trees when it was first seen. */
    readonly treesWhenSeen: MeshTreeCounts | null;
    /** Frames after it was first seen whose rays failed the oracle. */
    readonly failedAfterSeen: number[];
    /** The tracer's memory in the frame it was first seen and the next FREED_WITHIN_FRAMES frames. */
    readonly memoryAfterSeen: MemoryNow[];
    /** The last frame's failures, when it was never seen. */
    readonly lastFailures: string[];
}

/** Makes the change, then frames at a 60 Hz pace, tracing the check after each, until it has passed for STAY_FRAMES frames. */
async function framesUntilSeen(run: Run, check: Check, change: () => void): Promise<SeenResult> {
    const changedAt = performance.now();
    change();
    let firstSeenFrame: number | null = null;
    let firstSeenMs: number | null = null;
    let treesWhenSeen: MeshTreeCounts | null = null;
    let lastFailures: string[] = [];
    const failedAfterSeen: number[] = [];
    const memoryAfterSeen: MemoryNow[] = [];
    for (let frame = 1; frame <= MAX_FRAMES; frame++) {
        const frameStart = performance.now();
        run.frame();
        const failures = await checkNow(run, check);
        if (failures.length === 0 && firstSeenFrame === null) memoryAfterSeen.push(memoryNow(run));
        else if (firstSeenFrame !== null && frame <= firstSeenFrame + FREED_WITHIN_FRAMES) memoryAfterSeen.push(memoryNow(run));
        if (failures.length === 0 && firstSeenFrame === null) {
            firstSeenFrame = frame;
            firstSeenMs = performance.now() - changedAt;
            treesWhenSeen = run.platform.rays.meshTreeCounts;
        } else if (failures.length > 0) {
            lastFailures = failures;
            if (firstSeenFrame !== null) failedAfterSeen.push(frame);
        }
        if (firstSeenFrame !== null && frame >= firstSeenFrame + STAY_FRAMES) break;
        await waitUntil(frameStart + FRAME_MS);
    }
    return { firstSeenFrame, firstSeenMs, treesWhenSeen, failedAfterSeen, memoryAfterSeen, lastFailures: firstSeenFrame === null ? lastFailures : [] };
}

interface MemoryNow {
    readonly nodesInUse: number;
    readonly trianglesInUse: number;
    readonly gpuBuildOutputs: number;
    readonly onGpuTrees: number;
    readonly onTinybvhTrees: number;
}

function memoryNow(run: Run): MemoryNow {
    const memory: AccelerationMemory = run.platform.rays.memory;
    const trees = run.platform.rays.meshTreeCounts;
    return {
        nodesInUse: memory.nodesInUse,
        trianglesInUse: memory.trianglesInUse,
        gpuBuildOutputs: memory.gpuBuildOutputs,
        onGpuTrees: trees.onGpuTrees,
        onTinybvhTrees: trees.onTinybvhTrees,
    };
}

/**
 * Frames after the one the change was first seen in until nothing of the
 * mesh's old tree is left and the GPU build's buffers are gone (0 = that
 * frame itself), or null if not within FREED_WITHIN_FRAMES (see the file
 * comment, step 3).
 */
function freedAfterFrames(seen: SeenResult, tinybvhOnly: MemoryNow): number | null {
    const index = seen.memoryAfterSeen.findIndex((now) => now.gpuBuildOutputs === 0 && now.trianglesInUse === tinybvhOnly.trianglesInUse);
    return index < 0 ? null : index;
}

function median(values: readonly number[]): number {
    const sorted = [...values].sort((a, b) => a - b);
    const middle = Math.floor(sorted.length / 2);
    return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
}

/** Median GPU ms of SPEED_REPEATS batches of each mode (timestamps when the device has them, else the wall clock to queue idle). */
async function timeTrace(run: Run, packedRays: Float32Array): Promise<{ closestMs: number; anyMs: number; clock: 'timestamps' | 'wall' }> {
    const { device, platform } = run;
    const count = packedRays.length / 8;
    const rays = device.createBuffer({ label: 'runtime trees speed rays', size: count * RAY_BYTES, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(rays, 0, packedRays as Float32Array<ArrayBuffer>);
    const hits = device.createBuffer({ label: 'runtime trees speed hits', size: count * HIT_RECORD_BYTES, usage: GPUBufferUsage.STORAGE });
    const timed = device.features.has('timestamp-query');
    const querySet = timed ? device.createQuerySet({ label: 'runtime trees speed', type: 'timestamp', count: 2 }) : null;
    const resolved = timed ? device.createBuffer({ label: 'runtime trees speed times', size: 16, usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC }) : null;
    const measure = async (mode: 'closest' | 'any'): Promise<number> => {
        const samples: number[] = [];
        for (let repeat = 0; repeat < SPEED_REPEATS; repeat++) {
            const encoder = device.createCommandEncoder({ label: `runtime trees speed ${mode}` });
            const started = performance.now();
            platform.rays.encodeBatch(encoder, { mode, rays, hits, count }, querySet ? { querySet, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 } : undefined);
            if (querySet && resolved) encoder.resolveQuerySet(querySet, 0, 2, resolved, 0);
            device.queue.submit([encoder.finish()]);
            if (resolved) {
                const ticks = new BigUint64Array(await readBuffer(device, resolved, 16));
                samples.push(Number(ticks[1] - ticks[0]) / 1e6);
            } else {
                await device.queue.onSubmittedWorkDone();
                samples.push(performance.now() - started);
            }
        }
        return median(samples);
    };
    const closestMs = await measure('closest');
    const anyMs = await measure('any');
    for (const buffer of [rays, hits, resolved]) buffer?.destroy();
    querySet?.destroy();
    return { closestMs, anyMs, clock: timed ? 'timestamps' : 'wall' };
}

function round(value: number | null, digits = 3): number | null {
    return value === null || !Number.isFinite(value) ? value : Number(value.toFixed(digits));
}

// ---------------------------------------------------------------------------
// The page
// ---------------------------------------------------------------------------

await runGpuTest(
    async ({ device, log: logOnPage }) => {
        // Also to the console, which the runner prints: the page's own summary is lost when it fails.
        const log = (line: string): void => {
            logOnPage(line);
            console.log(line);
        };
        const failures: string[] = [];
        const report: Record<string, unknown> = { scenes: SCENES, frameMs: round(FRAME_MS), seenWithinFrames: SEEN_WITHIN_FRAMES, freedWithinFrames: FREED_WITHIN_FRAMES };
        if (!FURNISHED_ROOM_DOWNLOADED) log('the furnished room is not downloaded, so only the cabinet row runs');
        assert(device.features.has('subgroups'), 'this device has no subgroups, so there are no GPU trees to check (ddgi-trees-no-subgroups and the unit tests cover that path)');

        for (const sceneId of SCENES) {
            const random = createRandom(SEED);
            const scene = await loadSceneGeometry(sceneId);
            const cabinetId = scene.instances.length;
            const sheetId = cabinetId + 1;
            const checks: Record<StepName, Check> = {
                add: await makeCheck(random, scene, 'add', sceneState(scene, null, false), sceneState(scene, scene.closedCabinet, false), cabinetId),
                // Aimed at the shelf: most of the cabinet's faces are the same before and after, so rays aimed at all of it would test little.
                replace: await makeCheck(random, scene, 'replace', sceneState(scene, scene.closedCabinet, false), sceneState(scene, scene.openCabinet, false), cabinetId, scene.shelf),
                sheet: await makeCheck(random, scene, 'sheet', sceneState(scene, scene.openCabinet, false), sceneState(scene, scene.openCabinet, true), sheetId),
            };
            for (const check of Object.values(checks)) {
                if (check.changedShare < MIN_CHANGED_SHARE) {
                    failures.push(`${sceneId} ${check.step}: only ${(check.changedShare * 100).toFixed(1)}% of the rays answer differently after the change, so the check tests little`);
                }
            }
            const speedRays = packRays(aimedRays(random, scene.meshes[scene.sheet], scene.sheetAt, SPEED_RAYS, true));
            log(
                `${sceneId}: ${scene.triangles} triangles in ${scene.instances.length} instances; rays changed by add ${(checks.add.changedShare * 100).toFixed(0)}%, ` +
                    `replace ${(checks.replace.changedShare * 100).toFixed(0)}%, sheet ${(checks.sheet.changedShare * 100).toFixed(0)}%`,
            );

            const sceneReport: Record<string, unknown> = { triangles: scene.triangles, instances: scene.instances.length, changedShare: Object.fromEntries(Object.values(checks).map((check) => [check.step, round(check.changedShare)])) };
            const tinybvhOnlyMemory: Partial<Record<'replace' | 'sheet', MemoryNow>> = {};
            let tinybvhOnlySpeed: { closestMs: number; anyMs: number } | null = null;

            for (const variant of VARIANTS) {
                const run = await startRun(device, scene, variant);
                const { platform } = run;
                const recompilesBefore = platform.stats.recompiles;
                const label = `${sceneId} ${variant}`;
                const variantReport: Record<string, unknown> = {};
                const gated = variant !== 'tinybvh-only';
                // Scene meshes the GPU refused (too deep, say) are on fallback trees from the start; the page's meshes must not be.
                const fallbackAtStart = platform.rays.meshTreeCounts.onTinybvhTrees;

                const judgeSeen = (step: StepName, seen: SeenResult): void => {
                    const trees = seen.treesWhenSeen;
                    log(
                        `${label} ${step}: ` +
                            (seen.firstSeenFrame === null
                                ? `never passed in ${MAX_FRAMES} frames: ${seen.lastFailures.slice(0, 2).join(' | ')}`
                                : `passed from frame ${seen.firstSeenFrame} (${round(seen.firstSeenMs, 1)} ms), on ${trees?.onGpuTrees} GPU / ${trees?.onTinybvhTrees} tinybvh trees` +
                                  (seen.failedAfterSeen.length > 0 ? `; failed again in frames ${seen.failedAfterSeen.join(', ')}` : '')),
                    );
                    variantReport[step] = { ...seen, firstSeenMs: round(seen.firstSeenMs, 1) };
                    if (seen.failedAfterSeen.length > 0) failures.push(`${label} ${step}: passed from frame ${seen.firstSeenFrame}, then failed in frames ${seen.failedAfterSeen.join(', ')}`);
                    if (step === 'sheet' || !gated) return;
                    if (seen.firstSeenFrame === null || seen.firstSeenFrame > SEEN_WITHIN_FRAMES) {
                        failures.push(`${label} ${step}: rays saw the change in frame ${seen.firstSeenFrame ?? `> ${MAX_FRAMES}`} (bar ${SEEN_WITHIN_FRAMES})`);
                    }
                    // The changed mesh is on its GPU tree (no fallback tree beyond the scene's own), so passing proves the GPU tree.
                    if ((trees?.onTinybvhTrees ?? 0) > fallbackAtStart) {
                        failures.push(`${label} ${step}: ${trees?.onTinybvhTrees} meshes were on tinybvh (fallback) trees when the rays first saw the change, ${fallbackAtStart} at the start`);
                    }
                };
                const judgeFreed = (step: 'replace' | 'sheet', seen: SeenResult): void => {
                    const reference = tinybvhOnlyMemory[step];
                    assert(reference, `${label} ${step}: no tinybvh-only memory to compare with`);
                    const frames = freedAfterFrames(seen, reference);
                    variantReport[`${step}Memory`] = { freedAfterFrames: frames, afterSeen: seen.memoryAfterSeen, tinybvhOnly: reference };
                    const last = seen.memoryAfterSeen.at(-1);
                    log(
                        `${label} ${step} memory: ${frames === null ? `NOT FREED within ${FREED_WITHIN_FRAMES} frames` : `freed ${frames} frames after the frame it was seen`}; ` +
                            `nodes in use ${last?.nodesInUse} (tinybvh-only ${reference.nodesInUse}), triangles ${last?.trianglesInUse} (${reference.trianglesInUse})`,
                    );
                    if (frames === null) {
                        failures.push(`${label} ${step}: the old tree or the GPU build's buffers were not freed within ${FREED_WITHIN_FRAMES} frames (${JSON.stringify(seen.memoryAfterSeen)} against tinybvh-only ${JSON.stringify(reference)})`);
                    }
                };

                // The meshes the page adds, made inside the changes below.
                const added: { cabinet?: MeshHandle; sheet?: MeshHandle } = {};

                // 1. add: a new mesh and a new instance.
                judgeSeen(
                    'add',
                    await framesUntilSeen(run, checks.add, () => {
                        const closed = scene.meshes[scene.closedCabinet];
                        added.cabinet = platform.scene.createMesh({ positions: closed.positions, indices: closed.indices!, label: 'new cabinet' });
                        const handle = platform.scene.createInstance({ mesh: added.cabinet, materials: [run.material], transform: scene.cabinetAt });
                        run.caseIdBySlot.set(platform.sceneCopy.instanceSlot(handle), cabinetId);
                    }),
                );
                const cabinet = added.cabinet;
                assert(cabinet, 'the cabinet was not made');
                await settle(run);

                // 2. replace: the same mesh, a new shape, the instance unchanged.
                const replaced = await framesUntilSeen(run, checks.replace, () => {
                    const open = scene.meshes[scene.openCabinet];
                    platform.scene.updateMesh(cabinet, { positions: open.positions, indices: open.indices!, label: 'new cabinet, opened' });
                });
                judgeSeen('replace', replaced);

                // 3. freed: nothing of the cabinet's old tree stays, nor the GPU build's buffers.
                await settle(run);
                if (variant === 'tinybvh-only') tinybvhOnlyMemory.replace = memoryNow(run);
                else judgeFreed('replace', replaced);
                const afterReplace = await checkNow(run, checks.replace);
                if (afterReplace.length > 0) failures.push(`${label} replace: the oracle fails once the trees settled: ${afterReplace.slice(0, 2).join(' | ')}`);

                // 4. sheet: a big mesh added, its memory, and trace speed (recorded).
                const sheetSeen = await framesUntilSeen(run, checks.sheet, () => {
                        const sheet = scene.meshes[scene.sheet];
                        added.sheet = platform.scene.createMesh({ positions: sheet.positions, indices: sheet.indices!, label: 'bumpy sheet' });
                        const handle = platform.scene.createInstance({ mesh: added.sheet, materials: [run.material], transform: scene.sheetAt });
                    run.caseIdBySlot.set(platform.sceneCopy.instanceSlot(handle), sheetId);
                });
                judgeSeen('sheet', sheetSeen);
                assert(added.sheet, 'the sheet was not made');
                await settle(run);
                if (variant === 'tinybvh-only') {
                    tinybvhOnlyMemory.sheet = memoryNow(run);
                    tinybvhOnlySpeed = await timeTrace(run, speedRays);
                    variantReport.speed = tinybvhOnlySpeed;
                } else {
                    judgeFreed('sheet', sheetSeen);
                    const onGpuTree = await timeTrace(run, speedRays);
                    const ratio = (ms: number, against: number | undefined) => (against === undefined ? null : round(ms / against));
                    variantReport.speed = {
                        onGpuTree,
                        gpuTreeOverTinybvh: { closest: ratio(onGpuTree.closestMs, tinybvhOnlySpeed?.closestMs), any: ratio(onGpuTree.anyMs, tinybvhOnlySpeed?.anyMs) },
                    };
                    log(`${label} sheet speed (ms, closest / any): GPU tree ${round(onGpuTree.closestMs)} / ${round(onGpuTree.anyMs)}, tinybvh only ${round(tinybvhOnlySpeed?.closestMs ?? NaN)} / ${round(tinybvhOnlySpeed?.anyMs ?? NaN)}`);
                }

                const recompiles = platform.stats.recompiles - recompilesBefore;
                variantReport.recompiles = recompiles;
                variantReport.meshTrees = platform.rays.meshTreeCounts;
                if (recompiles !== 0) failures.push(`${label}: ${recompiles} compute pipelines were made while meshes came and changed`);
                sceneReport[variant] = variantReport;
                platform.dispose();
            }
            report[sceneId] = sceneReport;
        }
        assert(failures.length === 0, failures.slice(0, 10).join('\n'));
        return report;
    },
    { optionalFeatures: ['subgroups', 'timestamp-query'] },
);

/** The corner uvs of the kept triangles (6 numbers each), or null when the set is absent. */
function subsetUvs(uvs: Float64Array | null, kept: readonly number[]): Float64Array | null {
    if (!uvs) return null;
    const subset = new Float64Array(kept.length * 6);
    kept.forEach((triangle, slot) => subset.set(uvs.subarray(triangle * 6, triangle * 6 + 6), slot * 6));
    return subset;
}
