/**
 * Deforming meshes reach rays through the GPU refit (T3.3 items 1, 2 and 4;
 * T3.1 item 2 before it). docs/design/m3-moving.md §3, "Draft done-whens".
 *
 * The scene: a tessellated sheet created `deforming: true` with a
 * `deformBounds` that holds every pose, placed twice (once plainly, once
 * mirrored and non-uniformly scaled), plus a third placement that only
 * receives GI (rays must never hit it), and some rigid boxes and a triangle
 * soup around them.
 *
 * Each check traces random rays (half of them aimed at points on the sheet's
 * current surface) through the platform and compares them with the CPU
 * reference for the scene in that pose; the same answers against the flat
 * rest pose must disagree on many rays (the control). **Hit points too:** for
 * every closest hit on the sheet, the hit point rebuilt from the scene copy
 * (its positions, the hit's triangle and barycentrics, as hit shading's
 * `giInstanceTriangle` does) must be the ray's point at `t`, and the normal
 * hit shading reads (`giInstanceNormalAt`) must be the posed triangle's own.
 *
 * - **poses:** 8 waves, each traced in the frame right after it is set.
 * - **stream:** a new pose every frame for 60 frames, each traced in its own
 *   frame. No tree builds may start except a drift rebuild (the refit
 *   replaced T3.1's per-pose builds; a wave this far from the flat sheet the
 *   tree was built for can hold its drift long enough for one).
 * - **gpu forms:** a pose in a GPU buffer (vec4 stride, an offset) and in an
 *   rgba32float texture, each traced the frame after.
 * - **held bend** (first, from the flat rest tree with no drift running): a
 *   strong fold held for 150 frames: its refit drifts, so one drift rebuild
 *   (H-PLOC) must run, not before the drift has held 60 frames, and only one;
 *   the fold is traced once the rebuilt tree is in.
 * - **cost** (recorded, not judged; TASKS M3): the `platform/mesh refit`
 *   pass for a 100,000-triangle sheet posed every frame.
 * - **no subgroups:** on a device without subgroups, poses are ignored with
 *   one warning, and rays see the rest pose.
 *
 * Recompiles must stay 0. Nothing is timed against a bar.
 */

import type { FrameInput } from '../../../../packages/core/src/platform/contract/frame.ts';
import type { InstanceHandle, MeshHandle } from '../../../../packages/core/src/platform/contract/handles.ts';
import type { DeformedPositions } from '../../../../packages/core/src/platform/contract/scene.ts';
import { GIPlatformImpl } from '../../../../packages/core/src/platform/gi-platform.ts';
import { SceneReader } from '../../../../packages/core/src/platform/scene/testing/scene-reader.ts';
import { HIT_RECORD_BYTES, packRays, RAY_BYTES, unpackRay, type MeshGeometry, type TraceRay, type TraceSceneInstance } from '../../../../packages/core/src/platform/trace/index.ts';
import { DRIFT_HOLD_FRAMES } from '../../../../packages/core/src/platform/trace/deforming-trees.ts';
import { type TracedHit } from '../../../../packages/core/src/platform/trace/oracle-compare.ts';
import { createRandom, randomAffine, randomRays, randomTriangleSoup, randomUnitVector, tessellatedQuad, toFloat32Ray, type Random } from '../../../../packages/core/src/platform/trace/oracle-fixtures.ts';
import { assert, runGpuTest } from '../../client.ts';
import { boxMesh, transformPoint, type OracleCase } from '../trace-oracle/cases.ts';
import { computeReferenceAnswersYielding, evaluateCase } from '../trace-oracle/check.ts';
import { readBuffer } from '../tlas-oracle/top-level-check.ts';
import { IDENTITY_MATRIX } from '../trace-bench/gltf-geometry.ts';

const SEED = 0xdef0;
/** The sheet: CELLS x CELLS squares, two triangles each, from -HALF to HALF in x and y. */
const CELLS = 24;
const HALF = 1.5;
/** The wave's height (z) and the sideways push (x), in the sheet's own units. */
const WAVE_HEIGHT = 0.4;
const SIDEWAYS_PUSH = 0.1;
/** The held bend: a fold this high, well past the wave, so the rest-pose tree drifts. */
const BEND_HEIGHT = 1.0;
/** Frames the bend is held: past DRIFT_HOLD_FRAMES, with room for the rebuild to arrive and a second one not to start. */
const BEND_FRAMES = 150;
const POSES = 8;
const STREAM_FRAMES = 60;
const RAYS = 4000;
const STREAM_RAYS = 1000;
/** The cost check's sheet: 224 x 224 cells, 100,352 triangles. */
const COST_CELLS = 224;
const COST_FRAMES = 60;
/**
 * Aimed rays meet their target triangle at least this steeply (the cosine
 * between the ray and the triangle's normal). Why: a ray aimed from a
 * uniformly random direction grazes its target far more often than the
 * oracle's random rays do, and at a graze the f32 tracer's distance error is
 * multiplied by 1 / cosine, past the oracle's t tolerance (NOTES).
 */
const MIN_AIM_COSINE = 0.1;
/** Random rays may disagree with the reference this often, if the reference marks them ambiguous (as the oracle's random cases). */
const AMBIGUOUS_SHARE = 0.01;
/** Against the flat rest pose, at least this share of the rays must disagree, or the poses don't test anything. */
const MIN_REST_DISAGREEMENT = 0.1;
/**
 * A hit point rebuilt from the scene copy may differ from the ray's point at
 * `t` by this much (metres): f32 rounding of the corners, the barycentrics
 * and t, at distances of a few metres. A hit on the previous pose's sheet
 * would be off by centimetres.
 */
const HIT_POINT_TOLERANCE = 5e-4;
/** The shading normal must be the posed triangle's own: cosine at least this. */
const NORMAL_COSINE = 0.9999;
const HIT_RECORD_WORDS = HIT_RECORD_BYTES / 4;
const NO_CASE_ID = 0xffffffff;
/** rgba32float rows of 64 texels: 1,024 bytes, a multiple of the copy's 256. */
const TEXTURE_WIDTH = 64;

const IDLE_FRAME: FrameInput = {
    deltaTime: 0,
    budget: { units: 0 },
    camera: { position: [0, 0, 0], view: IDENTITY_MATRIX, projection: IDENTITY_MATRIX, near: 0.1, far: 100 },
};

/** The sheet at wave phase `phase`: z = height x sin(2x + phase) x cos(1.5y + phase / 2), x pushed by sin(y + phase). */
function sheetPose(rest: Float32Array, phase: number, height = WAVE_HEIGHT): Float32Array {
    const pose = new Float32Array(rest.length);
    for (let at = 0; at < rest.length; at += 3) {
        const x = rest[at];
        const y = rest[at + 1];
        pose[at] = x + SIDEWAYS_PUSH * Math.sin(y + phase);
        pose[at + 1] = y;
        pose[at + 2] = height * Math.sin(2 * x + phase) * Math.cos(1.5 * y + phase / 2);
    }
    return pose;
}

/** A fold that stays: z = BEND_HEIGHT x sin(4x) x cos(3y), a crumple the flat sheet's tree describes badly. */
function bendPose(rest: Float32Array): Float32Array {
    const pose = new Float32Array(rest.length);
    for (let at = 0; at < rest.length; at += 3) {
        pose[at] = rest[at];
        pose[at + 1] = rest[at + 1];
        pose[at + 2] = BEND_HEIGHT * Math.sin(4 * rest[at]) * Math.cos(3 * rest[at + 1]);
    }
    return pose;
}

interface CaseInstance {
    readonly id: number;
    readonly meshIndex: number;
    readonly objectToWorld: number[];
    readonly handle: InstanceHandle;
}

/**
 * Half random rays through the scene, half aimed at random points on the
 * sheet's surface in this pose (in each traced placement), from 0.5-4 m away
 * in a random direction no flatter than MIN_AIM_COSINE to the triangle.
 */
function makeRays(random: Random, sheet: MeshGeometry, placements: readonly number[][], reach: number, count: number): TraceRay[] {
    const rays = randomRays(random, count / 2, [-reach, -reach, -reach, reach, reach, reach], 0);
    const indices = sheet.indices!;
    const triangleCount = indices.length / 3;
    for (let index = 0; index < count / 2; index++) {
        const triangle = Math.floor(random() * triangleCount);
        let u = random();
        let v = random();
        if (u + v > 1) {
            u = 1 - u;
            v = 1 - v;
        }
        const corner = (which: number): [number, number, number] => {
            const vertex = indices[triangle * 3 + which];
            return [sheet.positions[vertex * 3], sheet.positions[vertex * 3 + 1], sheet.positions[vertex * 3 + 2]];
        };
        const placement = placements[index % placements.length];
        const [a, b, c] = [corner(0), corner(1), corner(2)].map((point) => transformPoint(placement, point));
        const target = [0, 1, 2].map((axis) => a[axis] + u * (b[axis] - a[axis]) + v * (c[axis] - a[axis]));
        const normal = cross(sub(b, a), sub(c, a));
        const normalLength = Math.hypot(normal[0], normal[1], normal[2]);
        let away = randomUnitVector(random);
        while (Math.abs(away[0] * normal[0] + away[1] * normal[1] + away[2] * normal[2]) < MIN_AIM_COSINE * normalLength) away = randomUnitVector(random);
        const distance = 0.5 + random() * 3.5;
        const origin = [0, 1, 2].map((axis) => target[axis] + away[axis] * distance);
        rays.push(toFloat32Ray(origin, [-away[0], -away[1], -away[2]], 0, 1e6));
    }
    return rays;
}

function sub(a: readonly number[], b: readonly number[]): number[] {
    return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

function cross(a: readonly number[], b: readonly number[]): number[] {
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}

interface Traced {
    readonly closest: (TracedHit | null)[];
    /** The scene slot each closest hit reports (`GiHit.instance`), or -1 for a miss. */
    readonly slots: number[];
    readonly any: boolean[];
}

/** Traces both modes through the platform and returns the answers with case ids in place of scene slots. */
async function trace(device: GPUDevice, platform: GIPlatformImpl, packedRays: Float32Array, bySlot: ReadonlyMap<number, CaseInstance>): Promise<Traced> {
    const count = packedRays.length / 8;
    const rays = device.createBuffer({ label: 'deformed rays', size: count * RAY_BYTES, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(rays, 0, packedRays as Float32Array<ArrayBuffer>);
    const hits = device.createBuffer({ label: 'deformed hits', size: count * HIT_RECORD_BYTES, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const blocked = device.createBuffer({ label: 'deformed blocked', size: count * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const encoder = device.createCommandEncoder({ label: 'deformed trace' });
    platform.rays.encodeBatch(encoder, { mode: 'closest', rays, hits, count });
    platform.rays.encodeBatch(encoder, { mode: 'any', rays, hits: blocked, count });
    device.queue.submit([encoder.finish()]);
    const hitData = await readBuffer(device, hits, count * HIT_RECORD_BYTES);
    const blockedWords = new Uint32Array(await readBuffer(device, blocked, count * 4));
    rays.destroy();
    hits.destroy();
    blocked.destroy();
    const floats = new Float32Array(hitData);
    const words = new Uint32Array(hitData);
    const closest: (TracedHit | null)[] = [];
    const slots: number[] = [];
    const any: boolean[] = [];
    for (let index = 0; index < count; index++) {
        const base = index * HIT_RECORD_WORDS;
        const t = floats[base];
        const hit = t >= 0;
        closest.push(
            hit
                ? {
                      hit: true,
                      t,
                      instanceIndex: bySlot.get(words[base + 1])?.id ?? NO_CASE_ID,
                      triangleIndex: words[base + 2],
                      frontFace: words[base + 3] !== 0,
                      u: floats[base + 4],
                      v: floats[base + 5],
                  }
                : null,
        );
        slots.push(hit ? words[base + 1] : -1);
        any.push(blockedWords[index] !== 0);
    }
    return { closest, slots, any };
}

/** The scene copy as hit shading reads it, from a readback of its two buffers. */
async function readSceneCopy(device: GPUDevice, platform: GIPlatformImpl): Promise<SceneReader> {
    const [vertices, tables] = platform.sceneCopy.module.resources().map((resource) => (resource as GPUBufferBinding).buffer);
    return new SceneReader(new Uint8Array(await readBuffer(device, vertices, vertices.size)), new Uint8Array(await readBuffer(device, tables, tables.size)));
}

/** The pose as an rgba32float texture (vertex i at texel (i mod width, i / width)), the form three's TSL compute writes. */
function poseTexture(device: GPUDevice, pose: Float32Array): GPUTexture {
    const vertexCount = pose.length / 3;
    const rows = Math.ceil(vertexCount / TEXTURE_WIDTH);
    const texels = new Float32Array(TEXTURE_WIDTH * rows * 4);
    for (let vertex = 0; vertex < vertexCount; vertex++) texels.set(pose.subarray(vertex * 3, vertex * 3 + 3), vertex * 4);
    const texture = device.createTexture({ label: 'pose texture', size: [TEXTURE_WIDTH, rows], format: 'rgba32float', usage: GPUTextureUsage.COPY_SRC | GPUTextureUsage.COPY_DST });
    device.queue.writeTexture({ texture }, texels, { bytesPerRow: TEXTURE_WIDTH * 16, rowsPerImage: rows }, [TEXTURE_WIDTH, rows]);
    return texture;
}

/** The pose in a storage buffer, 4 floats a vertex after a 16-byte header: a stride and offset other than tight xyz. */
function poseBuffer(device: GPUDevice, pose: Float32Array): GPUBuffer {
    const vertexCount = pose.length / 3;
    const floats = new Float32Array(4 + vertexCount * 4);
    for (let vertex = 0; vertex < vertexCount; vertex++) floats.set(pose.subarray(vertex * 3, vertex * 3 + 3), 4 + vertex * 4);
    const buffer = device.createBuffer({ label: 'pose buffer', size: floats.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(buffer, 0, floats);
    return buffer;
}

function median(values: readonly number[]): number {
    const sorted = [...values].sort((a, b) => a - b);
    return sorted.length === 0 ? Number.NaN : sorted[Math.floor(sorted.length / 2)];
}

/** Records the refit pass for a 100,000-triangle sheet posed every frame (not judged). */
async function measureRefitCost(device: GPUDevice): Promise<{ triangles: number; medianMs: number; samples: number }> {
    const sheet = tessellatedQuad(COST_CELLS, HALF);
    const platform = new GIPlatformImpl(device, { label: 'refit cost', gpuTiming: true });
    try {
        const material = platform.scene.createMaterial({ albedo: [1, 1, 1] });
        const mesh = platform.scene.createMesh({
            positions: sheet.positions,
            indices: sheet.indices!,
            deforming: true,
            deformBounds: { min: [-HALF - SIDEWAYS_PUSH, -HALF, -WAVE_HEIGHT], max: [HALF + SIDEWAYS_PUSH, HALF, WAVE_HEIGHT] },
        });
        platform.scene.createInstance({ mesh, materials: [material], transform: IDENTITY_MATRIX.slice() });
        await platform.prepare();
        const frame = async (): Promise<void> => {
            const encoder = device.createCommandEncoder({ label: 'refit cost frame' });
            platform.encode(encoder, IDLE_FRAME);
            device.queue.submit([encoder.finish()]);
            await device.queue.onSubmittedWorkDone();
        };
        for (let frames = 0; platform.rays.building || frames < 2; frames++) {
            assert(frames < 500, 'the cost sheet has no tree after 500 frames');
            await platform.rays.whenBuilt();
            await frame();
        }
        const samples = new Map<number, number>();
        for (let index = 0; index < COST_FRAMES; index++) {
            platform.scene.setDeformedPositions(mesh, sheetPose(sheet.positions, index * 0.2));
            await frame();
            for (const pass of platform.stats.passes) if (pass.name === 'platform/mesh refit' && pass.gpuMs !== undefined) samples.set(pass.frame, pass.gpuMs);
        }
        for (let index = 0; index < 4; index++) {
            await frame();
            for (const pass of platform.stats.passes) if (pass.name === 'platform/mesh refit' && pass.gpuMs !== undefined) samples.set(pass.frame, pass.gpuMs);
        }
        return { triangles: sheet.indices!.length / 3, medianMs: median([...samples.values()]), samples: samples.size };
    } finally {
        platform.dispose();
    }
}

/** On a device without subgroups: a pose is ignored with one warning, and rays see the rest pose. */
async function checkWithoutSubgroups(): Promise<{ warnings: number; restMismatches: number; poseDisagreement: number }> {
    const adapter = await navigator.gpu.requestAdapter();
    assert(adapter, 'no second adapter for the no-subgroups check');
    // No features asked for: the device has no subgroups.
    const device = await adapter.requestDevice();
    assert(!device.features.has('subgroups'), 'the second device has subgroups; this check needs one without');
    const warnings: string[] = [];
    const consoleWarn = console.warn;
    console.warn = (...parts: unknown[]) => {
        warnings.push(parts.map(String).join(' '));
        consoleWarn(...parts);
    };
    const sheet = tessellatedQuad(CELLS, HALF);
    const platform = new GIPlatformImpl(device, { label: 'no subgroups', rayBatches: true });
    try {
        const material = platform.scene.createMaterial({ albedo: [1, 1, 1] });
        const mesh = platform.scene.createMesh({
            positions: sheet.positions,
            indices: sheet.indices!,
            deforming: true,
            deformBounds: { min: [-HALF - SIDEWAYS_PUSH, -HALF, -WAVE_HEIGHT], max: [HALF + SIDEWAYS_PUSH, HALF, WAVE_HEIGHT] },
        });
        const handle = platform.scene.createInstance({ mesh, materials: [material], transform: IDENTITY_MATRIX.slice() });
        await platform.prepare();
        const frame = (): void => {
            const encoder = device.createCommandEncoder({ label: 'no subgroups frame' });
            platform.encode(encoder, IDLE_FRAME);
            device.queue.submit([encoder.finish()]);
        };
        for (let frames = 0; platform.rays.building || frames < 2; frames++) {
            assert(frames < 500, 'no tree without subgroups after 500 frames');
            await platform.rays.whenBuilt();
            frame();
            await device.queue.onSubmittedWorkDone();
        }
        const pose = sheetPose(sheet.positions, 1.3);
        for (let index = 0; index < 5; index++) {
            platform.scene.setDeformedPositions(mesh, sheetPose(sheet.positions, 1.3 + index * 0.1));
            frame();
        }
        platform.scene.setDeformedPositions(mesh, pose);
        frame();
        await device.queue.onSubmittedWorkDone();
        const random = createRandom(SEED + 7);
        const bySlot = new Map([[platform.sceneCopy.instanceSlot(handle), { id: 0, meshIndex: 0, objectToWorld: IDENTITY_MATRIX.slice(), handle }] as const]);
        const rays = packRays(makeRays(random, { positions: pose, indices: sheet.indices }, [IDENTITY_MATRIX.slice()], 4, RAYS));
        const answers = await trace(device, platform, rays, bySlot);
        const lookup = { closest: (index: number) => answers.closest[index], any: (index: number) => answers.any[index] };
        const instances: TraceSceneInstance[] = [{ meshIndex: 0, instanceIndex: 0, objectToWorld: IDENTITY_MATRIX.slice(), visible: true }];
        const restCase: OracleCase = { name: 'rest', description: 'no subgroups', meshes: [sheet], instances, rays, maxAmbiguousShare: { closest: AMBIGUOUS_SHARE, any: AMBIGUOUS_SHARE } };
        const posedCase: OracleCase = { ...restCase, name: 'posed', meshes: [{ positions: pose, indices: sheet.indices }] };
        const againstRest = evaluateCase(restCase, lookup, await computeReferenceAnswersYielding(restCase));
        const againstPose = evaluateCase(posedCase, lookup, await computeReferenceAnswersYielding(posedCase));
        const restWarnings = warnings.filter((message) => message.includes('rest pose')).length;
        return { warnings: restWarnings, restMismatches: againstRest.closest.mismatches + againstRest.any.mismatches, poseDisagreement: againstPose.closest.mismatches / againstPose.rays };
    } finally {
        console.warn = consoleWarn;
        platform.dispose();
        device.destroy();
    }
}

await runGpuTest(
    async ({ device, log }) => {
        const random = createRandom(SEED);
        const sheet = tessellatedQuad(CELLS, HALF);
        const restPositions = sheet.positions;
        const meshes: MeshGeometry[] = [sheet, boxMesh([0.6, 0.3, 0.2]), randomTriangleSoup(random, 60, 1, 0.4)];
        const SHEET = 0;
        const platform = new GIPlatformImpl(device, { label: 'scene-deformed-positions', rayBatches: true });
        const failures: string[] = [];
        try {
            const material = platform.scene.createMaterial({ albedo: [1, 1, 1] });
            // Normals given, as an adapter would: a deforming mesh's are dropped, and hits use the posed triangle's own.
            const sheetNormals = new Float32Array(restPositions.length).map((_, index) => (index % 3 === 2 ? 1 : 0));
            const meshHandles: MeshHandle[] = meshes.map((mesh, index) =>
                platform.scene.createMesh({
                    positions: mesh.positions,
                    indices: mesh.indices ?? Uint32Array.from({ length: mesh.positions.length / 3 }, (_, vertex) => vertex),
                    ...(index === SHEET
                        ? {
                              normals: sheetNormals,
                              deforming: true,
                              deformBounds: { min: [-HALF - SIDEWAYS_PUSH, -HALF, -BEND_HEIGHT], max: [HALF + SIDEWAYS_PUSH, HALF, BEND_HEIGHT] },
                              label: 'waving sheet',
                          }
                        : {}),
                }),
            );
            const traced: CaseInstance[] = [];
            const place = (meshIndex: number, objectToWorld: number[]): void => {
                const handle = platform.scene.createInstance({ mesh: meshHandles[meshIndex], materials: [material], transform: objectToWorld });
                traced.push({ id: traced.length, meshIndex, objectToWorld, handle });
            };
            const sheetPlacements = [IDENTITY_MATRIX.slice(), randomAffine(random, { extent: 3, mirror: true })];
            for (const placement of sheetPlacements) place(SHEET, placement);
            for (let index = 0; index < 6; index++) place(1 + (index % 2), randomAffine(random, { extent: 5 }));
            // A receiver: deformed with the others, never hit by rays.
            platform.scene.createInstance({ mesh: meshHandles[SHEET], materials: [material], transform: randomAffine(random, { extent: 3 }), contributesToGI: false });

            await platform.prepare();
            const frame = (): void => {
                const encoder = device.createCommandEncoder({ label: 'scene-deformed-positions frame' });
                platform.encode(encoder, IDLE_FRAME);
                device.queue.submit([encoder.finish()]);
            };
            // The scene's first trees (the sheet's from the GPU with its layout read back, or tinybvh's).
            for (let frames = 0; platform.rays.building || frames < 2; frames++) {
                assert(frames < 500, 'the mesh trees still build after 500 frames');
                await platform.rays.whenBuilt();
                frame();
                await device.queue.onSubmittedWorkDone();
            }

            const bySlot = new Map(traced.map((instance) => [platform.sceneCopy.instanceSlot(instance.handle), instance] as const));
            const sheetSlots = new Set(traced.filter((instance) => instance.meshIndex === SHEET).map((instance) => platform.sceneCopy.instanceSlot(instance.handle)));
            const caseInstances: TraceSceneInstance[] = traced.map((instance) => ({ meshIndex: instance.meshIndex, instanceIndex: instance.id, objectToWorld: instance.objectToWorld, visible: true }));
            const reach = 7;
            const restSheet: MeshGeometry = { positions: restPositions, indices: sheet.indices };

            /** Traces the scene with the sheet in `pose`: the oracle, the rest-pose control, and hit points and normals from the scene copy. */
            const check = async (label: string, pose: Float32Array, rayCount: number) => {
                const posed: MeshGeometry = { positions: pose, indices: sheet.indices };
                const caseMeshes = meshes.map((mesh, index) => (index === SHEET ? posed : mesh));
                const rays = packRays(makeRays(random, posed, sheetPlacements, reach, rayCount));
                const oracleCase: OracleCase = {
                    name: label,
                    description: 'a deforming sheet',
                    meshes: caseMeshes,
                    instances: caseInstances,
                    rays,
                    maxAmbiguousShare: { closest: AMBIGUOUS_SHARE, any: AMBIGUOUS_SHARE },
                };
                const answers = await trace(device, platform, rays, bySlot);
                const lookup = { closest: (index: number) => answers.closest[index], any: (index: number) => answers.any[index] };
                const evaluated = evaluateCase(oracleCase, lookup, await computeReferenceAnswersYielding(oracleCase));
                for (const failure of evaluated.failures) failures.push(`${label}: ${failure}`);
                const restCase: OracleCase = { ...oracleCase, name: `${label} (rest pose)`, meshes: meshes.map((mesh, index) => (index === SHEET ? restSheet : mesh)) };
                const againstRest = evaluateCase(restCase, lookup, await computeReferenceAnswersYielding(restCase));
                const restDisagreement = againstRest.closest.mismatches / againstRest.rays;
                if (restDisagreement < MIN_REST_DISAGREEMENT) failures.push(`${label}: only ${(restDisagreement * 100).toFixed(1)}% of the rays disagree with the flat sheet, so the pose tests nothing`);

                // Hit points and normals as hit shading rebuilds them from the scene copy.
                const reader = await readSceneCopy(device, platform);
                let sheetHits = 0;
                let largestPointError = 0;
                let smallestNormalCosine = 1;
                answers.closest.forEach((hit, index) => {
                    const slot = answers.slots[index];
                    if (!hit || !sheetSlots.has(slot)) return;
                    sheetHits++;
                    const ray = unpackRay(rays, index);
                    const [a, b, c] = reader.corners(slot, hit.triangleIndex);
                    const rebuilt = [0, 1, 2].map((axis) => a[axis] + hit.u * (b[axis] - a[axis]) + hit.v * (c[axis] - a[axis]));
                    const atT = [0, 1, 2].map((axis) => ray.origin[axis] + hit.t * ray.direction[axis]);
                    largestPointError = Math.max(largestPointError, Math.hypot(rebuilt[0] - atT[0], rebuilt[1] - atT[1], rebuilt[2] - atT[2]));
                    const shading = reader.normalAt(slot, hit.triangleIndex, [hit.u, hit.v]);
                    const geometric = reader.geometricNormal(slot, hit.triangleIndex);
                    smallestNormalCosine = Math.min(smallestNormalCosine, shading[0] * geometric[0] + shading[1] * geometric[1] + shading[2] * geometric[2]);
                });
                if (largestPointError > HIT_POINT_TOLERANCE) failures.push(`${label}: a hit point rebuilt from the scene copy is ${largestPointError.toExponential(2)} m from the ray's point (tolerance ${HIT_POINT_TOLERANCE})`);
                if (smallestNormalCosine < NORMAL_COSINE) failures.push(`${label}: a shading normal is off the posed triangle's own (cosine ${smallestNormalCosine.toFixed(5)})`);
                if (sheetHits < rayCount / 8) failures.push(`${label}: only ${sheetHits} rays hit the sheet`);
                return {
                    rays: evaluated.rays,
                    mismatches: evaluated.closest.mismatches + evaluated.any.mismatches,
                    ambiguous: evaluated.closest.ambiguous + evaluated.any.ambiguous,
                    restDisagreement,
                    sheetHits,
                    largestPointError,
                };
            };
            const treeCounts = () => platform.rays.meshTreeCounts;

            // held bend, first: the tree is still the flat rest pose's and no drift has started, so the
            // rebuild's timing is the bend's alone (later phases wave far from whatever tree they have).
            const rebuildsBefore = platform.rays.deformingTreeCounts?.driftRebuilds ?? 0;
            const gpuTreesBefore = treeCounts().gpuTreesBuilt;
            const bend = bendPose(restPositions);
            let rebuiltAtFrame = -1;
            for (let index = 0; index < BEND_FRAMES; index++) {
                // The same bend, given again every frame, as a host's animation holding a pose would.
                platform.scene.setDeformedPositions(meshHandles[SHEET], bend);
                frame();
                await device.queue.onSubmittedWorkDone();
                if (rebuiltAtFrame < 0 && treeCounts().gpuTreesBuilt > gpuTreesBefore) rebuiltAtFrame = index;
            }
            const driftRebuilds = (platform.rays.deformingTreeCounts?.driftRebuilds ?? 0) - rebuildsBefore;
            if (driftRebuilds !== 1) failures.push(`the held bend started ${driftRebuilds} drift rebuilds in ${BEND_FRAMES} frames (expected exactly 1)`);
            if (rebuiltAtFrame < 0) failures.push('the drift rebuild never arrived');
            else if (rebuiltAtFrame < DRIFT_HOLD_FRAMES) failures.push(`the drift rebuild arrived at frame ${rebuiltAtFrame}, before the drift had held ${DRIFT_HOLD_FRAMES} frames`);
            const bendResult = await check('held bend after its rebuild', bend, RAYS);
            log(`held bend: ${driftRebuilds} drift rebuild, its tree in at frame ${rebuiltAtFrame} of ${BEND_FRAMES}; then ${bendResult.mismatches} mismatches`);

            // poses: each traced in the frame right after it is set.
            const poseResults = [];
            for (let index = 0; index < POSES; index++) {
                const pose = sheetPose(restPositions, 0.7 + index * 0.9);
                platform.scene.setDeformedPositions(meshHandles[SHEET], pose);
                frame();
                const result = await check(`pose ${index}`, pose, RAYS);
                poseResults.push(result);
                log(
                    `pose ${index}: ${result.mismatches} mismatches, ${result.ambiguous} ambiguous of ${result.rays * 2} answers; ${result.sheetHits} sheet hits, hit points within ${result.largestPointError.toExponential(1)} m; ` +
                        `${(result.restDisagreement * 100).toFixed(1)}% disagree with the flat sheet`,
                );
            }

            // stream: a new pose every frame for 60 frames, each traced in its own frame; no tree builds start but drift rebuilds.
            const before = treeCounts();
            const driftRebuildsBefore = platform.rays.deformingTreeCounts?.driftRebuilds ?? 0;
            let streamMismatches = 0;
            let streamLargestPointError = 0;
            for (let index = 0; index < STREAM_FRAMES; index++) {
                const pose = sheetPose(restPositions, 10 + index * 0.25);
                platform.scene.setDeformedPositions(meshHandles[SHEET], pose);
                frame();
                const result = await check(`stream frame ${index}`, pose, STREAM_RAYS);
                streamMismatches += result.mismatches;
                streamLargestPointError = Math.max(streamLargestPointError, result.largestPointError);
            }
            const after = treeCounts();
            const streamDriftRebuilds = (platform.rays.deformingTreeCounts?.driftRebuilds ?? 0) - driftRebuildsBefore;
            const buildsDuringStream = after.gpuTreesBuilt - before.gpuTreesBuilt - streamDriftRebuilds + after.tinybvhBuilding;
            if (buildsDuringStream > 0) failures.push(`${buildsDuringStream} tree builds besides drift rebuilds ran while the sheet was posed every frame (the refit should need none)`);
            if (streamDriftRebuilds > 1) failures.push(`${streamDriftRebuilds} drift rebuilds in ${STREAM_FRAMES} frames (at most one a second)`);
            log(`stream: ${STREAM_FRAMES} poses, each traced in its own frame: ${streamMismatches} mismatches in all, hit points within ${streamLargestPointError.toExponential(1)} m; other tree builds ${buildsDuringStream}, drift rebuilds ${streamDriftRebuilds}`);

            // gpu forms: a buffer with a vec4 stride and an offset, then a texture.
            const bufferPose = sheetPose(restPositions, 3.1);
            const buffer = poseBuffer(device, bufferPose);
            const fromBuffer: DeformedPositions = { buffer, offset: 16, stride: 16 };
            platform.scene.setDeformedPositions(meshHandles[SHEET], fromBuffer);
            frame();
            const bufferResult = await check('gpu buffer pose', bufferPose, RAYS);
            buffer.destroy();
            const texturePose = sheetPose(restPositions, 4.7);
            const texture = poseTexture(device, texturePose);
            platform.scene.setDeformedPositions(meshHandles[SHEET], { texture });
            frame();
            const textureResult = await check('gpu texture pose', texturePose, RAYS);
            texture.destroy();
            log(`gpu forms: buffer ${bufferResult.mismatches} mismatches (${bufferResult.sheetHits} sheet hits), texture ${textureResult.mismatches} mismatches (${textureResult.sheetHits} sheet hits)`);

            const recompiles = platform.stats.recompiles;
            if (recompiles !== 0) failures.push(`${recompiles} recompiles`);
            const counters = platform.stats.counters['platform deforming meshes'];

            const cost = await measureRefitCost(device);
            log(`cost (not judged): platform/mesh refit for ${cost.triangles} triangles, median ${cost.medianMs.toFixed(3)} ms over ${cost.samples} frames`);

            const noSubgroups = await checkWithoutSubgroups();
            log(`no subgroups: ${noSubgroups.warnings} warning(s); against the rest pose ${noSubgroups.restMismatches} mismatches; ${(noSubgroups.poseDisagreement * 100).toFixed(1)}% disagree with the pose given`);
            if (noSubgroups.warnings !== 1) failures.push(`without subgroups, ${noSubgroups.warnings} rest-pose warnings (expected 1)`);
            if (noSubgroups.restMismatches !== 0) failures.push(`without subgroups, rays don't see the rest pose (${noSubgroups.restMismatches} mismatches)`);
            if (noSubgroups.poseDisagreement < MIN_REST_DISAGREEMENT) failures.push('without subgroups, the ignored pose is too close to rest to test anything');

            assert(failures.length === 0, failures.slice(0, 8).join('\n'));
            return {
                poses: poseResults,
                stream: { frames: STREAM_FRAMES, mismatches: streamMismatches, largestPointError: streamLargestPointError, treeBuilds: buildsDuringStream, driftRebuilds: streamDriftRebuilds },
                gpuForms: { buffer: bufferResult, texture: textureResult },
                heldBend: { driftRebuilds, rebuiltAtFrame, ...bendResult },
                recompiles,
                counters,
                cost,
                noSubgroups,
                triangles: sheet.indices!.length / 3,
            };
        } finally {
            platform.dispose();
        }
    },
    { optionalFeatures: ['subgroups', 'timestamp-query'] },
);
