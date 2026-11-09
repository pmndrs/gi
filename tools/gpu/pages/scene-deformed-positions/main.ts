/**
 * T3.1 item 2 (docs/design/m3-moving.md §1 and "Draft done-whens"):
 * `setDeformedPositions` passes the oracle on a deforming test mesh. The
 * positions go in through the scene API; rays must hit the deformed surface.
 *
 * The scene: a tessellated sheet created `deforming: true` with a
 * `deformBounds` that holds every pose, placed twice (once plainly, once
 * mirrored and non-uniformly scaled), plus a third placement that only
 * receives GI (rays must never hit it), and some rigid boxes and a triangle
 * soup around them.
 *
 * **poses:** POSES times, the sheet gets a new wave (z = a sine of x and y,
 * x pushed sideways a little) through `setDeformedPositions`. After its tree
 * is ready, random rays (half of them aimed at points on the sheet's new
 * surface, from a little way off) are traced through the platform and
 * compared with the CPU reference for the scene in that pose. As a check
 * that the test can fail, the same answers are also compared with the
 * reference for the flat rest pose: they must disagree on many rays.
 *
 * **stream:** a host that deforms every frame. STREAM_FRAMES frames each
 * give a new pose without waiting for anything. Trees must keep arriving
 * (the scene holds new poses while a tree is building, so each pose doesn't
 * restart the build before it ends), and once the stream stops and the last
 * tree is in, the rays must see the last pose given.
 *
 * Prototype level (TASKS M3): the pose reaches rays by a tree rebuild per
 * pose; T3.3 replaces it with the GPU refit. Nothing here is timed.
 */

import type { FrameInput } from '../../../../packages/ddgi-core/src/platform/contract/frame.ts';
import type { InstanceHandle } from '../../../../packages/ddgi-core/src/platform/contract/handles.ts';
import { GIPlatformImpl } from '../../../../packages/ddgi-core/src/platform/gi-platform.ts';
import { HIT_RECORD_BYTES, packRays, RAY_BYTES, type MeshGeometry, type TraceRay, type TraceSceneInstance } from '../../../../packages/ddgi-core/src/platform/trace/index.ts';
import { type TracedHit } from '../../../../packages/ddgi-core/src/platform/trace/oracle-compare.ts';
import { createRandom, randomAffine, randomRays, randomTriangleSoup, randomUnitVector, tessellatedQuad, toFloat32Ray, type Random } from '../../../../packages/ddgi-core/src/platform/trace/oracle-fixtures.ts';
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
const POSES = 8;
const STREAM_FRAMES = 40;
/** Trees that must arrive during the stream: a few, so a host deforming every frame isn't starved. */
const MIN_STREAM_TREES = 3;
const RAYS = 4000;
/**
 * Aimed rays meet their target triangle at least this steeply (the cosine
 * between the ray and the triangle's normal). Why: a ray aimed from a
 * uniformly random direction grazes its target far more often than the
 * oracle's random rays do (those hit surfaces in proportion to how much they
 * face them), and at a graze the f32 tracer's distance error is multiplied
 * by 1 / cosine, past the oracle's t tolerance (the first run: 2 of 8 poses
 * had one such ray, t off by 2e-4). Grazing precision is trace-oracle's
 * business; this page is about which surface the rays see.
 */
const MIN_AIM_COSINE = 0.1;
/** Random rays may disagree with the reference this often, if the reference marks them ambiguous (as the oracle's random cases). */
const AMBIGUOUS_SHARE = 0.01;
/** Against the flat rest pose, at least this share of the rays must disagree, or the poses don't test anything. */
const MIN_REST_DISAGREEMENT = 0.1;
const HIT_RECORD_WORDS = HIT_RECORD_BYTES / 4;
const NO_CASE_ID = 0xffffffff;

const IDLE_FRAME: FrameInput = {
    deltaTime: 0,
    budget: { units: 0 },
    camera: { position: [0, 0, 0], view: IDENTITY_MATRIX, projection: IDENTITY_MATRIX, near: 0.1, far: 100 },
};

/** The sheet at wave phase `phase`: z = height x sin(2x + phase) x cos(1.5y + phase / 2), x pushed by sin(y + phase). */
function sheetPose(rest: Float32Array, phase: number): Float32Array {
    const pose = new Float32Array(rest.length);
    for (let at = 0; at < rest.length; at += 3) {
        const x = rest[at];
        const y = rest[at + 1];
        pose[at] = x + SIDEWAYS_PUSH * Math.sin(y + phase);
        pose[at + 1] = y;
        pose[at + 2] = WAVE_HEIGHT * Math.sin(2 * x + phase) * Math.cos(1.5 * y + phase / 2);
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
 * in a random direction no flatter than MIN_AIM_COSINE to the triangle, so
 * most of them land on the deformed surface.
 */
function makeRays(random: Random, sheet: MeshGeometry, placements: readonly number[][], reach: number): TraceRay[] {
    const rays = randomRays(random, RAYS / 2, [-reach, -reach, -reach, reach, reach, reach], 0);
    const indices = sheet.indices!;
    const triangleCount = indices.length / 3;
    for (let index = 0; index < RAYS / 2; index++) {
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
        // The triangle's world normal, for how steeply a direction meets it.
        const edge1 = [0, 1, 2].map((axis) => b[axis] - a[axis]);
        const edge2 = [0, 1, 2].map((axis) => c[axis] - a[axis]);
        const normal = [edge1[1] * edge2[2] - edge1[2] * edge2[1], edge1[2] * edge2[0] - edge1[0] * edge2[2], edge1[0] * edge2[1] - edge1[1] * edge2[0]];
        const normalLength = Math.hypot(normal[0], normal[1], normal[2]);
        let away = randomUnitVector(random);
        while (Math.abs(away[0] * normal[0] + away[1] * normal[1] + away[2] * normal[2]) < MIN_AIM_COSINE * normalLength) away = randomUnitVector(random);
        const distance = 0.5 + random() * 3.5;
        const origin = [0, 1, 2].map((axis) => target[axis] + away[axis] * distance);
        rays.push(toFloat32Ray(origin, [-away[0], -away[1], -away[2]], 0, 1e6));
    }
    return rays;
}

/** Traces both modes through the platform and returns the answers with case ids in place of scene slots. */
async function trace(device: GPUDevice, platform: GIPlatformImpl, packedRays: Float32Array, bySlot: ReadonlyMap<number, CaseInstance>): Promise<{ closest: (TracedHit | null)[]; any: boolean[] }> {
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
    const any: boolean[] = [];
    for (let index = 0; index < count; index++) {
        const base = index * HIT_RECORD_WORDS;
        const t = floats[base];
        closest.push(
            t >= 0
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
        any.push(blockedWords[index] !== 0);
    }
    return { closest, any };
}

await runGpuTest(
    async ({ device, log }) => {
        const random = createRandom(SEED);
        const sheet = tessellatedQuad(CELLS, HALF);
        const restPositions = sheet.positions;
        const meshes: MeshGeometry[] = [sheet, boxMesh([0.6, 0.3, 0.2]), randomTriangleSoup(random, 60, 1, 0.4)];
        const SHEET = 0;
        const platform = new GIPlatformImpl(device, { label: 'scene-deformed-positions', rayBatches: true });
        try {
            const material = platform.scene.createMaterial({ albedo: [1, 1, 1] });
            const meshHandles = meshes.map((mesh, index) =>
                platform.scene.createMesh({
                    positions: mesh.positions,
                    indices: mesh.indices ?? Uint32Array.from({ length: mesh.positions.length / 3 }, (_, vertex) => vertex),
                    ...(index === SHEET
                        ? {
                              deforming: true,
                              deformBounds: { min: [-HALF - SIDEWAYS_PUSH, -HALF, -WAVE_HEIGHT], max: [HALF + SIDEWAYS_PUSH, HALF, WAVE_HEIGHT] },
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
            const settle = async (): Promise<number> => {
                // A pose given now is applied in the next frame, and its tree arrives some frames later.
                let frames = 0;
                for (; platform.rays.building || frames < 2; frames++) {
                    assert(frames < 500, 'the mesh trees still build after 500 frames');
                    await platform.rays.whenBuilt();
                    frame();
                    await device.queue.onSubmittedWorkDone();
                }
                return frames;
            };
            frame();
            await settle();

            const bySlot = new Map(traced.map((instance) => [platform.sceneCopy.instanceSlot(instance.handle), instance] as const));
            const caseInstances: TraceSceneInstance[] = traced.map((instance) => ({ meshIndex: instance.meshIndex, instanceIndex: instance.id, objectToWorld: instance.objectToWorld, visible: true }));
            const reach = 7;
            const failures: string[] = [];
            const restSheet: MeshGeometry = { positions: restPositions, indices: sheet.indices };
            /** Traces the scene with the sheet in `pose`, checks it against the reference, and returns the counts. */
            const check = async (label: string, pose: Float32Array): Promise<{ rays: number; mismatches: number; ambiguous: number; restDisagreement: number; sheetHits: number }> => {
                const posed: MeshGeometry = { positions: pose, indices: sheet.indices };
                const caseMeshes = meshes.map((mesh, index) => (index === SHEET ? posed : mesh));
                const rays = packRays(makeRays(random, posed, sheetPlacements, reach));
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
                // The control: the same answers against the flat sheet must disagree a lot.
                const restCase: OracleCase = { ...oracleCase, name: `${label} (rest pose)`, meshes: meshes.map((mesh, index) => (index === SHEET ? restSheet : mesh)) };
                const againstRest = evaluateCase(restCase, lookup, await computeReferenceAnswersYielding(restCase));
                const sheetIds = new Set(traced.filter((instance) => instance.meshIndex === SHEET).map((instance) => instance.id));
                const sheetHits = answers.closest.filter((hit) => hit !== null && sheetIds.has(hit.instanceIndex)).length;
                return {
                    rays: evaluated.rays,
                    mismatches: evaluated.closest.mismatches + evaluated.any.mismatches,
                    ambiguous: evaluated.closest.ambiguous + evaluated.any.ambiguous,
                    restDisagreement: againstRest.closest.mismatches / againstRest.rays,
                    sheetHits,
                };
            };

            // poses: one at a time, each traced once its tree is in.
            const poseResults = [];
            for (let index = 0; index < POSES; index++) {
                const pose = sheetPose(restPositions, 0.7 + index * 0.9);
                platform.scene.setDeformedPositions(meshHandles[SHEET], pose);
                const framesToTree = await settle();
                const result = await check(`pose ${index}`, pose);
                poseResults.push({ ...result, framesToTree });
                if (result.restDisagreement < MIN_REST_DISAGREEMENT) {
                    failures.push(`pose ${index}: only ${(result.restDisagreement * 100).toFixed(1)}% of the rays disagree with the flat sheet, so the pose tests nothing`);
                }
                log(
                    `pose ${index}: ${result.mismatches} mismatches, ${result.ambiguous} ambiguous of ${result.rays * 2} answers; ${result.sheetHits} rays hit the sheet; ` +
                        `${(result.restDisagreement * 100).toFixed(1)}% disagree with the flat sheet; tree in after ${framesToTree} frames`,
                );
            }

            // stream: a new pose every frame, never waiting for a tree.
            // Trees placed are counted from `rays.building` read right after each frame's encode
            // (before any await lets a worker's tree come back): a frame that ends with nothing
            // waiting after one that did placed the tree it waited for. A host starved of trees
            // (each pose restarting the build before its tree is placed) ends every frame waiting.
            let treesArrived = 0;
            let waitingAfterLastFrame = platform.rays.building;
            let lastPose = restPositions;
            for (let index = 0; index < STREAM_FRAMES; index++) {
                lastPose = sheetPose(restPositions, 10 + index * 0.25);
                platform.scene.setDeformedPositions(meshHandles[SHEET], lastPose);
                frame();
                const waitingNow = platform.rays.building;
                if (waitingAfterLastFrame && !waitingNow) treesArrived++;
                waitingAfterLastFrame = waitingNow;
                await device.queue.onSubmittedWorkDone();
            }
            const framesAfterStream = await settle();
            const streamResult = await check('after the stream', lastPose);
            if (treesArrived < MIN_STREAM_TREES) failures.push(`only ${treesArrived} trees arrived in ${STREAM_FRAMES} frames of new poses (at least ${MIN_STREAM_TREES} expected)`);
            log(`stream: ${treesArrived} trees arrived in ${STREAM_FRAMES} frames; the last pose: ${streamResult.mismatches} mismatches, ${streamResult.sheetHits} sheet hits, tree in ${framesAfterStream} frames after`);

            assert(failures.length === 0, failures.slice(0, 8).join('\n'));
            return {
                poses: poseResults,
                stream: { frames: STREAM_FRAMES, treesArrived, framesAfterStream, ...streamResult },
                triangles: sheet.indices!.length / 3,
            };
        } finally {
            platform.dispose();
        }
    },
    { optionalFeatures: ['subgroups'] },
);
