/**
 * B1a oracle: the top level with instances moving every frame.
 *
 * **scripted**: one sequence (fixed seed) of 32 frames over 40 instances of
 * three meshes, a third of them mirrored, all non-uniformly scaled. Every
 * frame every instance moves (small steps; every sixth frame a third of them
 * jump across the scene, so the tree's quality decays and rebuilds start),
 * and some frames add events: hide one, squash one to a singular transform,
 * show and restore them, destroy one and create one (the slot is reused),
 * create two more, destroy another. It runs twice: on a device with
 * `subgroups` (the GPU refit) and on a device without (the declared
 * fallback: the CPU top level). Both must pass, and they must agree on every
 * ray the reference is sure about.
 *
 * **many** (GPU refit only): 2,000 small instances moving every frame for 8
 * frames, so climbing threads in 32 workgroups pass boxes to each other all
 * the time.
 *
 * In both, after every frame's moves, with no waiting, random rays are traced
 * through the platform (`rays.encodeBatch`) and compared with the CPU
 * reference for the scene as it is now. Only after events that add an
 * instance does the page wait for the new shape: new instances appear when
 * their shape arrives, by design. The top-level buffer is also read back and
 * checked node by node (top-level-check.ts).
 *
 * **repair** (GPU refit only): the refit's safety path. A leaf slot listed
 * twice makes one climb wait for a box no thread writes, so it gives up; the
 * repair pass must still leave exactly the CPU refit's boxes.
 *
 * The platform runs with `gpuTiming`, so the refit's stats are read back:
 * climbs that gave up are counted (and reported here), and the refit pass
 * must show in `stats.passes` and its counters in `stats.counters`.
 */

import { GpuBvhRefit } from '../../../../packages/bvh/src/gpu-refit.ts';
import { PRIMITIVE_BOX_FLOATS, refitOnCpu } from '../../../../packages/bvh/src/refit-cpu.ts';
import type { FrameInput } from '../../../../packages/ddgi-core/src/platform/contract/frame.ts';
import type { InstanceHandle, MeshHandle } from '../../../../packages/ddgi-core/src/platform/contract/handles.ts';
import { GIPlatformImpl } from '../../../../packages/ddgi-core/src/platform/gi-platform.ts';
import {
    BVH_NODE_BYTES,
    HIT_RECORD_BYTES,
    invertAffine,
    packRays,
    RAY_BYTES,
    type MeshGeometry,
    type TraceSceneInstance,
} from '../../../../packages/ddgi-core/src/platform/trace/index.ts';
import { type TracedHit } from '../../../../packages/ddgi-core/src/platform/trace/oracle-compare.ts';
import { createRandom, randomAffine, randomRays, randomTriangleSoup, tessellatedQuad, type Random } from '../../../../packages/ddgi-core/src/platform/trace/oracle-fixtures.ts';
import { buildTopLevelShape } from '../../../../packages/ddgi-core/src/platform/trace/top-level-shape.ts';
import { transformBounds } from '../../../../packages/ddgi-core/src/platform/trace/top-level.ts';
import { assert, runGpuTest, watchDevice } from '../../client.ts';
import { boxMesh, type OracleCase } from '../trace-oracle/cases.ts';
import { computeReferenceAnswersYielding, evaluateCase, type ReferenceAnswers } from '../trace-oracle/check.ts';
import { IDENTITY_MATRIX } from '../trace-bench/gltf-geometry.ts';
import { checkTopLevel, objectBoxOf, readBuffer, type CheckedInstance } from './top-level-check.ts';

const SEED = 0xb1a;
/** Random rays may disagree with the reference this often, if the reference marks them ambiguous (as the oracle's random cases). */
const AMBIGUOUS_SHARE = 0.01;
const HIT_RECORD_WORDS = HIT_RECORD_BYTES / 4;
const NO_CASE_ID = 0xffffffff;
const REFIT_PASS = 'platform/top-level refit';

const IDLE_FRAME: FrameInput = {
    deltaTime: 0,
    budget: { units: 0 },
    camera: { position: [0, 0, 0], view: IDENTITY_MATRIX, projection: IDENTITY_MATRIX, near: 0.1, far: 100 },
};

/** One moving-instances sequence. */
interface SequencePlan {
    readonly name: string;
    readonly startInstances: number;
    readonly frames: number;
    readonly raysPerFrame: number;
    /** Instances sit in a cube of this half-size. */
    readonly extent: number;
    readonly meshes: (random: Random) => MeshGeometry[];
    /** The scripted hide / singular / show / create / destroy events. */
    readonly events: boolean;
}

const SCRIPTED: SequencePlan = {
    name: 'scripted',
    startInstances: 40,
    frames: 32,
    raysPerFrame: 3000,
    extent: 8,
    meshes: (random) => [boxMesh([1, 0.4, 0.2]), randomTriangleSoup(random, 80, 1.2, 0.5), tessellatedQuad(3, 1)],
    events: true,
};

const MANY: SequencePlan = {
    name: 'many',
    startInstances: 2000,
    frames: 8,
    raysPerFrame: 2000,
    extent: 30,
    meshes: () => [boxMesh([0.6, 0.3, 0.2])],
    events: false,
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
    name: string;
    path: string;
    frames: FrameAnswers[];
    references: ReferenceAnswers[];
    failures: string[];
    rays: number;
    mismatches: number;
    ambiguous: number;
    nodeChecks: number;
    counters: Readonly<Record<string, number>> | null;
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

/** Runs one sequence on one device. */
async function runSequence(device: GPUDevice, plan: SequencePlan, expectedPath: 'gpu-refit' | 'cpu', log: (line: string) => void): Promise<RunResult> {
    const random = createRandom(SEED);
    const caseMeshes = plan.meshes(random);
    const objectBoxes = caseMeshes.map((mesh) => objectBoxOf(mesh.positions));
    const label = `${plan.name} ${expectedPath}`;
    const platform = new GIPlatformImpl(device, { label: `tlas-oracle ${label}`, gpuTiming: true });
    const result: RunResult = { name: plan.name, path: expectedPath, frames: [], references: [], failures: [], rays: 0, mismatches: 0, ambiguous: 0, nodeChecks: 0, counters: null };
    try {
        assert(platform.rays.topLevelPath === expectedPath, `expected the ${expectedPath} top level, got ${platform.rays.topLevelPath}`);
        const material = platform.scene.createMaterial({ albedo: [1, 1, 1] });
        const meshHandles: MeshHandle[] = caseMeshes.map((mesh) =>
            platform.scene.createMesh({ positions: mesh.positions, indices: mesh.indices ?? Uint32Array.from({ length: mesh.positions.length / 3 }, (_, index) => index) }),
        );
        const instances: CaseInstance[] = [];
        let nextId = 0;
        const create = (meshIndex: number, mirror: boolean): void => {
            const objectToWorld = randomAffine(random, { extent: plan.extent, mirror });
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
        for (let index = 0; index < plan.startInstances; index++) create(index % caseMeshes.length, index % 3 === 0);

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

        for (let frameIndex = 0; frameIndex < plan.frames; frameIndex++) {
            // Moves, every frame, for everything.
            const jump = frameIndex > 0 && frameIndex % (plan.events ? 6 : 3) === 0;
            for (const instance of instances) {
                if (!instance.visible) continue;
                const mirrored = (invertAffine(instance.objectToWorld)?.determinant ?? 1) < 0 || instance.id % 3 === 0;
                instance.objectToWorld =
                    jump && instance.id % 3 === frameIndex % 3 ? randomAffine(random, { extent: plan.extent, mirror: mirrored }) : nudge(random, instance.objectToWorld);
                platform.scene.setTransform(instance.handle, instance.objectToWorld);
            }
            let adds = false;
            if (plan.events) {
                if (frameIndex === 8) {
                    byId(3).visible = false;
                    platform.scene.setVisible(byId(3).handle, false);
                    byId(4).objectToWorld = singular(byId(4).objectToWorld);
                    platform.scene.setTransform(byId(4).handle, byId(4).objectToWorld);
                } else if (frameIndex === 12) {
                    byId(3).visible = true;
                    platform.scene.setVisible(byId(3).handle, true);
                    byId(4).objectToWorld = randomAffine(random, { extent: plan.extent });
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
            const reach = plan.extent + 2;
            const rays = randomRays(random, plan.raysPerFrame, [-reach, -reach, -reach, reach, reach, reach], 0, { randomInterval: frameIndex % 2 === 1 });
            const oracleCase: OracleCase = {
                name: `${label} frame ${frameIndex}`,
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
            for (const failure of evaluated.failures) result.failures.push(`${label} frame ${frameIndex}: ${failure}`);

            const checked = new Map<number, CheckedInstance>([...bySlot].map(([slot, instance]) => [slot, { ...instance, objectBox: objectBoxes[instance.meshIndex] }]));
            const tree = await checkTopLevel(device, (platform.rays.module.resources()[2] as GPUBufferBinding).buffer, checked);
            result.nodeChecks += tree.checked;
            for (const failure of tree.failures) result.failures.push(`${label} frame ${frameIndex}: ${failure}`);
            if (frameIndex % 8 === 0) log(`${label} frame ${frameIndex}: ${instances.length} instances, ${evaluated.closest.mismatches + evaluated.any.mismatches} mismatches, ${tree.checked} slots checked`);
        }

        if (platform.rays.gpuTopLevel) {
            // Let the last stats and timestamps come back, then check what the platform reports.
            for (let extra = 0; extra < 6; extra++) {
                frame();
                await device.queue.onSubmittedWorkDone();
            }
            const counters = platform.stats.counters['platform top level'];
            result.counters = counters ?? null;
            if (!counters) result.failures.push(`${label}: no "platform top level" counters in stats.counters`);
            else if (counters['refit passes'] < plan.frames) result.failures.push(`${label}: only ${counters['refit passes']} refits for ${plan.frames} moving frames`);
            if (counters && counters['refit reads that gave up (diagnostics)'] > 0) {
                log(`${label}: ${counters['refit reads that gave up (diagnostics)']} refit reads gave up and were repaired`);
            }
            if (!(platform.rays.gpuTopLevel.latestRelativeCost > 0)) result.failures.push(`${label}: no refit quality was reported`);
            if (device.features.has('timestamp-query') && !platform.stats.passes.some((pass) => pass.name === REFIT_PASS)) {
                result.failures.push(`${label}: the refit pass is missing from stats.passes`);
            }
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

/**
 * The repair path: a 500-instance tree whose leaf-slot list names one leaf
 * twice and leaves out its sibling. The sibling's box is then never written
 * by the climb, the thread waiting for it gives up, and the repair pass must
 * refit the whole tree to exactly what the CPU refit gives.
 */
async function checkRepair(device: GPUDevice): Promise<{ failures: string[]; gaveUp: number; repairs: number }> {
    const random = createRandom(SEED + 1);
    const count = 500;
    const objectToWorld = new Float64Array(count * 16);
    const meshBounds = new Float32Array(count * 6);
    for (let index = 0; index < count; index++) {
        objectToWorld.set(randomAffine(random, { extent: 40, mirror: index % 2 === 0 }), index * 16);
        meshBounds.set([-1, -0.5, -0.25, 1, 0.5, 0.25], index * 6);
    }
    const shape = buildTopLevelShape({
        slots: Uint32Array.from({ length: count }, (_, index) => index),
        visible: new Uint8Array(count).fill(1),
        objectToWorld: objectToWorld.slice(),
        meshBounds: meshBounds.slice(),
    });
    // Each record's box, a little moved from the build so the refit has work to do.
    const primitiveBoxes = new Float32Array(count * PRIMITIVE_BOX_FLOATS);
    const box = new Float32Array(6);
    for (let record = 0; record < count; record++) {
        const input = shape.recordSlots[record];
        const moved = objectToWorld.slice(input * 16, input * 16 + 16);
        moved[12] += 3;
        transformBounds(meshBounds.subarray(input * 6, input * 6 + 6), moved, box, 0);
        primitiveBoxes.set(box.subarray(0, 3), record * PRIMITIVE_BOX_FLOATS);
        primitiveBoxes.set(box.subarray(3, 6), record * PRIMITIVE_BOX_FLOATS + 4);
    }
    // Find a node whose two children are both leaves, and list its left leaf twice instead of both.
    const leafSlots = shape.leafSlots.slice();
    const position = leafSlots.findIndex((slot, index) => (slot & 1) === 0 && leafSlots[index + 1] === slot + 1);
    assert(position >= 0, 'no node with two leaf children');
    leafSlots[position + 1] = leafSlots[position];

    const nodeBytes = shape.nodeCount * BVH_NODE_BYTES;
    const tree = device.createBuffer({ label: 'repair tree', size: nodeBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC });
    device.queue.writeBuffer(tree, 0, shape.nodes, 0, nodeBytes);
    const boxes = device.createBuffer({ label: 'repair boxes', size: primitiveBoxes.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(boxes, 0, primitiveBoxes);
    const refit = new GpuBvhRefit(device, { label: 'repair check', diagnostics: true });
    try {
        await refit.prepare();
        refit.setShape({ nodeWordOffset: 0, nodeCount: shape.nodeCount, parentSlots: shape.parentSlots, leafSlots, generation: 1 });
        const encoder = device.createCommandEncoder({ label: 'repair check' });
        const pass = encoder.beginComputePass({ label: 'repair check' });
        refit.encode(pass, tree, boxes);
        pass.end();
        refit.copyStats(encoder);
        device.queue.submit([encoder.finish()]);
        await device.queue.onSubmittedWorkDone();
        refit.pollStats();
        for (let waited = 0; refit.latestStats === null; waited++) {
            assert(waited < 200, 'the repair check stats never came back');
            await new Promise((resolve) => setTimeout(resolve, 5));
        }
        const stats = refit.latestStats;

        const gpuWords = new Uint32Array(await readBuffer(device, tree, nodeBytes));
        const expected = new Uint32Array(shape.nodes.slice(0, nodeBytes));
        const parents = Uint32Array.from(shape.parentSlots, (slot) => (slot === 0xffffffff ? slot : slot >> 1));
        refitOnCpu(new Float32Array(expected.buffer), expected, 0, shape.nodeCount, parents, primitiveBoxes);
        const failures: string[] = [];
        let differing = 0;
        for (let word = 0; word < expected.length; word++) if (gpuWords[word] !== expected[word]) differing++;
        if (differing > 0) failures.push(`repair: ${differing} tree words differ from the CPU refit`);
        if (stats.incomplete === 0) failures.push('repair: the climb did not give up, so the repair path was not exercised');
        if (stats.repairs !== 1) failures.push(`repair: the repair pass ran ${stats.repairs} times, expected 1`);
        return { failures, gaveUp: stats.incomplete, repairs: stats.repairs };
    } finally {
        refit.dispose();
        tree.destroy();
        boxes.destroy();
    }
}

await runGpuTest(
    async ({ device, log }) => {
        assert(device.features.has('subgroups'), 'this adapter has no subgroups, so the GPU refit cannot be tested here');
        const gpu = await runSequence(device, SCRIPTED, 'gpu-refit', log);
        const many = await runSequence(device, MANY, 'gpu-refit', log);
        const repair = await checkRepair(device);
        log(`repair: ${repair.gaveUp} reads gave up, ${repair.repairs} repair pass`);

        // The declared fallback: the same GPU, a device without subgroups (an adapter makes one device, so a new one).
        const plainAdapter = await navigator.gpu.requestAdapter();
        assert(plainAdapter, 'no second adapter');
        const plainDevice = await plainAdapter.requestDevice({});
        const plainErrors = watchDevice(plainDevice);
        let cpu: RunResult;
        try {
            cpu = await runSequence(plainDevice, SCRIPTED, 'cpu', log);
            await plainDevice.queue.onSubmittedWorkDone();
        } finally {
            plainDevice.destroy();
        }
        const agreement = compareRuns(gpu, cpu);
        log(`gpu-refit vs cpu: ${agreement.disagreements} disagreements in ${agreement.compared} rays`);

        const failures = [...gpu.failures, ...many.failures, ...repair.failures, ...cpu.failures, ...plainErrors.map((error) => `device without subgroups: ${error}`)];
        if (agreement.disagreements > 0) failures.push(`gpu-refit and cpu disagree on ${agreement.disagreements} rays: ${agreement.examples.join(' | ')}`);
        assert(failures.length === 0, failures.slice(0, 12).join('\n'));
        const summary = (run: RunResult) => ({ rays: run.rays, mismatches: run.mismatches, ambiguous: run.ambiguous, slotsChecked: run.nodeChecks, counters: run.counters });
        return {
            scripted: { frames: SCRIPTED.frames, gpuRefit: summary(gpu), cpuFallback: summary(cpu), agreement: { compared: agreement.compared, disagreements: agreement.disagreements } },
            many: { instances: MANY.startInstances, frames: MANY.frames, ...summary(many) },
            repair: { gaveUp: repair.gaveUp, repairs: repair.repairs },
        };
    },
    { optionalFeatures: ['subgroups', 'timestamp-query'] },
);
