/**
 * B1a oracle: the top level with instances moving every frame.
 *
 * One scripted sequence (fixed seed) of 32 frames over 40 instances of three
 * meshes, a third of them mirrored, all non-uniformly scaled. Every frame
 * every instance moves (small steps; every sixth frame a third of them jump
 * across the scene, so the refit tree's quality decays and rebuilds start),
 * and some frames add events: hide one, squash one to a singular transform,
 * show and restore them, destroy one and create one (the slot is reused),
 * create two more, destroy another.
 *
 * After every frame's moves, with no waiting, 3000 random rays are traced
 * through the platform (`rays.encodeBatch`) and compared with the CPU
 * reference for the scene as it is now. Only after events that add an
 * instance does the page wait for the new top-level shape: new instances
 * appear when their shape arrives, by design. The top-level buffer is also
 * read back and checked node by node: every inner box is exactly the union of
 * its child's two boxes, every instance box contains the instance's exact
 * world box (and isn't much looser), hidden and singular instances can't be
 * hit, and every record's world-to-object matrix is the inverse.
 *
 * The sequence runs twice: on a device with `subgroups` (the GPU refit) and
 * on a device without (the declared fallback: the CPU top level). Both must
 * pass, and they must agree on every ray the reference is sure about.
 */

import type { FrameInput } from '../../../../packages/ddgi-core/src/platform/contract/frame.ts';
import type { InstanceHandle, MeshHandle } from '../../../../packages/ddgi-core/src/platform/contract/handles.ts';
import { GIPlatformImpl } from '../../../../packages/ddgi-core/src/platform/gi-platform.ts';
import {
    HIT_RECORD_BYTES,
    invertAffine,
    packRays,
    RAY_BYTES,
    type MeshGeometry,
    type TraceSceneInstance,
} from '../../../../packages/ddgi-core/src/platform/trace/index.ts';
import { type TracedHit } from '../../../../packages/ddgi-core/src/platform/trace/oracle-compare.ts';
import { createRandom, randomAffine, randomRays, randomTriangleSoup, tessellatedQuad, type Random } from '../../../../packages/ddgi-core/src/platform/trace/oracle-fixtures.ts';
import { assert, runGpuTest, watchDevice } from '../../client.ts';
import { boxMesh, type OracleCase } from '../trace-oracle/cases.ts';
import { computeReferenceAnswersYielding, evaluateCase, type ReferenceAnswers } from '../trace-oracle/check.ts';
import { IDENTITY_MATRIX } from '../trace-bench/gltf-geometry.ts';
import { checkTopLevel, objectBoxOf, readBuffer, type CheckedInstance } from './top-level-check.ts';

const SEED = 0xb1a;
const FRAMES = 32;
const START_INSTANCES = 40;
const RAYS_PER_FRAME = 3000;
const EXTENT = 8;
/** Random rays may disagree with the reference this often, if the reference marks them ambiguous (as the oracle's random cases). */
const AMBIGUOUS_SHARE = 0.01;
const HIT_RECORD_WORDS = HIT_RECORD_BYTES / 4;
const NO_CASE_ID = 0xffffffff;

const IDLE_FRAME: FrameInput = {
    deltaTime: 0,
    budget: { units: 0 },
    camera: { position: [0, 0, 0], view: IDENTITY_MATRIX, projection: IDENTITY_MATRIX, near: 0.1, far: 100 },
};

interface CaseInstance {
    readonly id: number;
    readonly meshIndex: number;
    objectToWorld: number[];
    visible: boolean;
    handle: InstanceHandle;
}

/** One frame's answers, by ray, in case ids. */
interface FrameAnswers {
    closest: (TracedHit | null)[];
    any: boolean[];
}

interface RunResult {
    path: string;
    frames: FrameAnswers[];
    references: ReferenceAnswers[];
    failures: string[];
    rays: number;
    mismatches: number;
    ambiguous: number;
    nodeChecks: number;
    counters: Record<string, number> | null;
}

function meshes(random: Random): MeshGeometry[] {
    return [boxMesh([1, 0.4, 0.2]), randomTriangleSoup(random, 80, 1.2, 0.5), tessellatedQuad(3, 1)];
}

/** Moves an instance a small step: translation, plus a small change of one axis's scale (kept non-uniform). */
function nudge(random: Random, matrix: number[]): number[] {
    const next = matrix.slice();
    for (let axis = 0; axis < 3; axis++) next[12 + axis] += (random() * 2 - 1) * 0.4;
    const stretch = 1 + (random() * 2 - 1) * 0.1;
    for (let row = 0; row < 3; row++) next[row] *= stretch;
    return next;
}

function singular(matrix: number[]): number[] {
    const next = matrix.slice();
    for (const index of [0, 1, 2, 4, 5, 6, 8, 9, 10]) next[index] = 0;
    return next;
}

/** Runs the whole scripted sequence on one device. */
async function runSequence(device: GPUDevice, expectedPath: 'gpu-refit' | 'cpu', log: (line: string) => void): Promise<RunResult> {
    const random = createRandom(SEED);
    const caseMeshes = meshes(random);
    const objectBoxes = caseMeshes.map((mesh) => objectBoxOf(mesh.positions));
    const platform = new GIPlatformImpl(device, { label: `tlas-oracle ${expectedPath}` });
    const result: RunResult = { path: expectedPath, frames: [], references: [], failures: [], rays: 0, mismatches: 0, ambiguous: 0, nodeChecks: 0, counters: null };
    try {
        assert(platform.rays.topLevelPath === expectedPath, `expected the ${expectedPath} top level, got ${platform.rays.topLevelPath}`);
        const material = platform.scene.createMaterial({ albedo: [1, 1, 1] });
        const meshHandles: MeshHandle[] = caseMeshes.map((mesh) =>
            platform.scene.createMesh({ positions: mesh.positions, indices: mesh.indices ?? Uint32Array.from({ length: mesh.positions.length / 3 }, (_, index) => index) }),
        );
        const instances: CaseInstance[] = [];
        let nextId = 0;
        const create = (meshIndex: number, mirror: boolean): void => {
            const objectToWorld = randomAffine(random, { extent: EXTENT, mirror });
            const handle = platform.scene.createInstance({ mesh: meshHandles[meshIndex], materials: [material], transform: objectToWorld });
            instances.push({ id: nextId++, meshIndex, objectToWorld, visible: true, handle });
        };
        const byId = (id: number): CaseInstance => {
            const found = instances.find((instance) => instance.id === id);
            assert(found, `no instance ${id}`);
            return found;
        };
        const destroy = (id: number): void => {
            const instance = byId(id);
            platform.scene.destroyInstance(instance.handle);
            instances.splice(instances.indexOf(instance), 1);
        };
        for (let index = 0; index < START_INSTANCES; index++) create(index % caseMeshes.length, index % 3 === 0);

        await platform.prepare();
        const frame = (): void => {
            const encoder = device.createCommandEncoder({ label: 'tlas-oracle frame' });
            platform.encode(encoder, IDLE_FRAME);
            device.queue.submit([encoder.finish()]);
        };
        const settle = async (): Promise<void> => {
            // New instances appear once their top-level shape (and new meshes once their tree) arrives.
            for (let waited = 0; platform.rays.building || waited === 0; waited++) {
                assert(waited < 500, 'the top level still builds after 500 frames');
                await platform.rays.whenBuilt();
                frame();
                await device.queue.onSubmittedWorkDone();
            }
        };
        frame();
        await settle();

        for (let frameIndex = 0; frameIndex < FRAMES; frameIndex++) {
            // Moves, every frame, for everything.
            const jump = frameIndex > 0 && frameIndex % 6 === 0;
            for (const instance of instances) {
                if (!instance.visible) continue;
                const mirrored = (invertAffine(instance.objectToWorld)?.determinant ?? 1) < 0 || instance.id % 3 === 0;
                instance.objectToWorld = jump && instance.id % 3 === frameIndex % 3 ? randomAffine(random, { extent: EXTENT, mirror: mirrored }) : nudge(random, instance.objectToWorld);
                platform.scene.setTransform(instance.handle, instance.objectToWorld);
            }
            // Events.
            let adds = false;
            if (frameIndex === 8) {
                byId(3).visible = false;
                platform.scene.setVisible(byId(3).handle, false);
                byId(4).objectToWorld = singular(byId(4).objectToWorld);
                platform.scene.setTransform(byId(4).handle, byId(4).objectToWorld);
            } else if (frameIndex === 12) {
                byId(3).visible = true;
                platform.scene.setVisible(byId(3).handle, true);
                byId(4).objectToWorld = randomAffine(random, { extent: EXTENT });
                platform.scene.setTransform(byId(4).handle, byId(4).objectToWorld);
            } else if (frameIndex === 16) {
                destroy(5);
                create(2, true);
                adds = true;
            } else if (frameIndex === 20) {
                create(0, false);
                create(1, true);
                adds = true;
            } else if (frameIndex === 24) {
                destroy(6);
            }
            frame();
            // Shown and hidden instances change at once; created ones appear with their new shape.
            if (adds) await settle();

            const bySlot = new Map(instances.map((instance) => [platform.sceneCopy.instanceSlot(instance.handle), instance] as const));
            const caseInstances: TraceSceneInstance[] = instances.map((instance) => ({
                meshIndex: instance.meshIndex,
                instanceIndex: instance.id,
                objectToWorld: instance.objectToWorld,
                visible: instance.visible,
            }));
            const rays = randomRays(random, RAYS_PER_FRAME, [-EXTENT - 2, -EXTENT - 2, -EXTENT - 2, EXTENT + 2, EXTENT + 2, EXTENT + 2], 0, {
                randomInterval: frameIndex % 2 === 1,
            });
            const oracleCase: OracleCase = {
                name: `frame ${frameIndex}`,
                description: 'moving instances',
                meshes: caseMeshes,
                instances: caseInstances,
                rays: packRays(rays),
                maxAmbiguousShare: { closest: AMBIGUOUS_SHARE, any: AMBIGUOUS_SHARE },
            };
            const answers = await trace(device, platform, oracleCase.rays, bySlot);
            const reference = await computeReferenceAnswersYielding(oracleCase);
            const evaluated = evaluateCase(oracleCase, { closest: (index) => answers.closest[index], any: (index) => answers.any[index] }, reference);
            result.frames.push(answers);
            result.references.push(reference);
            result.rays += evaluated.rays * 2;
            result.mismatches += evaluated.closest.mismatches + evaluated.any.mismatches;
            result.ambiguous += evaluated.closest.ambiguous + evaluated.any.ambiguous;
            for (const failure of evaluated.failures) result.failures.push(`${expectedPath} frame ${frameIndex}: ${failure}`);

            const checked = new Map<number, CheckedInstance>(
                [...bySlot].map(([slot, instance]) => [slot, { ...instance, objectBox: objectBoxes[instance.meshIndex] }]),
            );
            const tree = await checkTopLevel(device, (platform.rays.module.resources()[2] as GPUBufferBinding).buffer, checked);
            result.nodeChecks += tree.checked;
            for (const failure of tree.failures) result.failures.push(`${expectedPath} frame ${frameIndex}: ${failure}`);
            if (frameIndex % 8 === 0) log(`${expectedPath} frame ${frameIndex}: ${instances.length} instances, ${evaluated.closest.mismatches + evaluated.any.mismatches} mismatches, ${tree.checked} slots checked`);
        }

        const top = platform.rays.gpuTopLevel;
        if (top) {
            // Let the last stats readback land, then check the counters.
            for (let extra = 0; extra < 4; extra++) {
                frame();
                await device.queue.onSubmittedWorkDone();
            }
            result.counters = { ...top.counters, latestRelativeCost: top.latestRelativeCost };
            if (top.counters.refitIncomplete !== 0) result.failures.push(`gpu-refit: ${top.counters.refitIncomplete} refit box reads gave up waiting`);
            if (top.counters.refits < FRAMES) result.failures.push(`gpu-refit: only ${top.counters.refits} refits for ${FRAMES} moving frames`);
            if (!(top.latestRelativeCost > 0)) result.failures.push('gpu-refit: no refit stats were read back');
        }
        return result;
    } finally {
        platform.dispose();
    }
}

/** Traces both modes and returns the answers with case ids in place of scene slots. */
async function trace(device: GPUDevice, platform: GIPlatformImpl, packedRays: Float32Array, bySlot: ReadonlyMap<number, CaseInstance>): Promise<FrameAnswers> {
    const count = packedRays.length / 8;
    const rays = device.createBuffer({ label: 'tlas-oracle rays', size: count * RAY_BYTES, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(rays, 0, packedRays as Float32Array<ArrayBuffer>);
    const hits = device.createBuffer({ label: 'tlas-oracle hits', size: count * HIT_RECORD_BYTES, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const blocked = device.createBuffer({ label: 'tlas-oracle blocked', size: count * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    const encoder = device.createCommandEncoder({ label: 'tlas-oracle trace' });
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

/** Rays where the GPU refit and the CPU fallback disagree although neither answer is excused as ambiguous. */
function compareRuns(gpu: RunResult, cpu: RunResult): { compared: number; disagreements: number; examples: string[] } {
    let compared = 0;
    let disagreements = 0;
    const examples: string[] = [];
    gpu.frames.forEach((gpuFrame, frameIndex) => {
        const cpuFrame = cpu.frames[frameIndex];
        const reference = gpu.references[frameIndex];
        gpuFrame.closest.forEach((gpuHit, ray) => {
            compared++;
            const cpuHit = cpuFrame.closest[ray];
            const sameClosest = (gpuHit === null) === (cpuHit === null) && (gpuHit === null || (gpuHit.instanceIndex === cpuHit?.instanceIndex && gpuHit.triangleIndex === cpuHit.triangleIndex));
            const sameAny = gpuFrame.any[ray] === cpuFrame.any[ray];
            if (sameClosest && sameAny) return;
            const excused = reference.closest[ray]?.ambiguous === true || reference.any[ray].ambiguous;
            if (excused) return;
            disagreements++;
            if (examples.length < 5) examples.push(`frame ${frameIndex} ray ${ray}: gpu ${JSON.stringify(gpuHit)} / ${gpuFrame.any[ray]}, cpu ${JSON.stringify(cpuHit)} / ${cpuFrame.any[ray]}`);
        });
    });
    return { compared, disagreements, examples };
}

await runGpuTest(
    async ({ device, log }) => {
        assert(device.features.has('subgroups'), 'this adapter has no subgroups, so the GPU refit cannot be tested here');
        const gpu = await runSequence(device, 'gpu-refit', log);

        // The declared fallback: the same GPU, a device without subgroups (an adapter makes one device, so a new one).
        const plainAdapter = await navigator.gpu.requestAdapter();
        assert(plainAdapter, 'no second adapter');
        const plainDevice = await plainAdapter.requestDevice({});
        const plainErrors = watchDevice(plainDevice);
        let cpu: RunResult;
        try {
            cpu = await runSequence(plainDevice, 'cpu', log);
            await plainDevice.queue.onSubmittedWorkDone();
        } finally {
            plainDevice.destroy();
        }
        const agreement = compareRuns(gpu, cpu);
        log(`gpu-refit vs cpu: ${agreement.disagreements} disagreements in ${agreement.compared} rays`);

        const failures = [...gpu.failures, ...cpu.failures, ...plainErrors.map((error) => `device without subgroups: ${error}`)];
        if (agreement.disagreements > 0) failures.push(`gpu-refit and cpu disagree on ${agreement.disagreements} rays: ${agreement.examples.join(' | ')}`);
        assert(failures.length === 0, failures.slice(0, 12).join('\n'));
        const summary = (run: RunResult) => ({ rays: run.rays, mismatches: run.mismatches, ambiguous: run.ambiguous, slotsChecked: run.nodeChecks, counters: run.counters });
        return { frames: FRAMES, gpuRefit: summary(gpu), cpuFallback: summary(cpu), agreement: { compared: agreement.compared, disagreements: agreement.disagreements } };
    },
    { optionalFeatures: ['subgroups'] },
);
