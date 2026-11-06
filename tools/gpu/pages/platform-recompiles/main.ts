/**
 * T2.M1 item 2 (docs/design/m3-moving.md §1, T3.1 check 4): a recompile
 * counter is in `gi.stats` (`recompiles`: compute pipelines the platform and
 * its techniques made while ready), and creating and destroying 1,000
 * instances in random order leaves it at 0.
 *
 * The cabinet row with DDGI updating every frame: over CHURN_FRAMES frames,
 * each frame creates a few instances (of the scene's meshes and of
 * NEW_MESHES meshes made along the way), destroys a few live ones, and moves
 * some (`setTransforms`), until INSTANCES have been created and every one of
 * them destroyed again. The instance table grows past its first capacity on
 * the way (reallocations, which must rebuild bind groups only). Then the
 * counter must still read 0, and as a control it must count a pipeline the
 * page makes through the platform's kernel builder after that.
 */

import { assert, runGpuTest } from '../../client.ts';
import type { InstanceHandle, MeshHandle } from '../../../../packages/ddgi-core/src/platform/contract/handles.ts';
import { createRandom } from '../../../../packages/ddgi-core/src/platform/trace/oracle-fixtures.ts';
import { createDdgiScene, releaseDeviceOnPageHide, runFrame } from '../ddgi-shared/harness.ts';

/** Instances created (and destroyed) over the run: the done-when's 1,000. */
const INSTANCES = 1000;
/** New meshes made among them, each destroyed once its last instance is gone. */
const NEW_MESHES = 20;
const CHURN_FRAMES = 200;
/** DDGI's probes a frame during the churn. */
const UNITS = 256;
/** The scene's instance table starts with room for 64 (DEFAULT_SCENE_CAPACITIES): the churn must outgrow it. */
const MIN_LIVE_PEAK = 128;

await runGpuTest(
    async ({ device, log }) => {
        releaseDeviceOnPageHide(device);
        const scene = await createDdgiScene(device, 'cabinet-row');
        const platform = scene.platform;
        const api = platform.scene;
        const random = createRandom(0x7e3);
        const boxMesh = scene.data.meshes[scene.data.instances[0].mesh];
        // One shared mesh for most instances (an instance per cabinet board would need the harness's mesh handles).
        const sharedMesh = api.createMesh({ positions: boxMesh.positions, indices: boxMesh.indices, label: 'shared churn box' });
        const material = scene.materials[0];
        // Read through a function: the counter changes behind TypeScript's back (a getter), so it mustn't be narrowed.
        const recompiles = (): number => platform.stats.recompiles;
        assert(recompiles() === 0, `the counter reads ${recompiles()} before the churn`);
        for (let frame = 0; frame < 16; frame++) runFrame(device, platform, UNITS);
        await device.queue.onSubmittedWorkDone();
        const before = recompiles();

        const live: { instance: InstanceHandle; mesh: MeshHandle }[] = [];
        const newMeshes = new Map<MeshHandle, number>();
        let created = 0;
        let destroyed = 0;
        let livePeak = 0;
        let moved = 0;
        let meshesMade = 0;
        const transform = (): number[] => {
            const size = 0.2 + random() * 0.4;
            return [size, 0, 0, 0, 0, size, 0, 0, 0, 0, size, 0, 0.5 + random() * 3, 0.2 + random() * 2, 0.5 + random() * 2, 1];
        };
        for (let frame = 0; frame < CHURN_FRAMES || live.length > 0; frame++) {
            const remainingFrames = Math.max(1, CHURN_FRAMES - frame);
            const toCreate = Math.min(INSTANCES - created, Math.ceil(((INSTANCES - created) / remainingFrames) * (0.5 + random())));
            for (let index = 0; index < toCreate; index++) {
                let mesh = sharedMesh;
                // About NEW_MESHES new meshes over the run (more likely early, so some live long).
                if (meshesMade < NEW_MESHES && random() < (2 * NEW_MESHES) / INSTANCES) {
                    mesh = api.createMesh({ positions: boxMesh.positions, indices: boxMesh.indices, label: `churn mesh ${meshesMade}` });
                    newMeshes.set(mesh, 0);
                    meshesMade++;
                } else if (newMeshes.size > 0 && random() < 0.3) {
                    const choices = [...newMeshes.keys()];
                    mesh = choices[Math.floor(random() * choices.length)];
                }
                if (newMeshes.has(mesh)) newMeshes.set(mesh, newMeshes.get(mesh)! + 1);
                live.push({ instance: api.createInstance({ mesh, materials: [material], transform: transform(), label: `churn ${created}` }), mesh });
                created++;
            }
            livePeak = Math.max(livePeak, live.length);
            // Destroy in random order: a few early on, everything once all are created.
            const toDestroy = created >= INSTANCES ? Math.min(live.length, 1 + Math.floor(random() * 40)) : Math.floor(random() * toCreate * 0.6);
            for (let index = 0; index < toDestroy && live.length > 0; index++) {
                const [gone] = live.splice(Math.floor(random() * live.length), 1);
                api.destroyInstance(gone.instance);
                destroyed++;
                const users = newMeshes.get(gone.mesh);
                if (users !== undefined) {
                    if (users <= 1) {
                        api.destroyMesh(gone.mesh);
                        newMeshes.delete(gone.mesh);
                    } else {
                        newMeshes.set(gone.mesh, users - 1);
                    }
                }
            }
            // Move a few of the live ones in one batch.
            const movers = live.filter(() => random() < 0.1);
            if (movers.length > 0) {
                const matrices = new Float32Array(movers.length * 16);
                movers.forEach((_, index) => matrices.set(transform(), index * 16));
                api.setTransforms(Uint32Array.from(movers.map((mover) => mover.instance)), matrices);
                moved += movers.length;
            }
            runFrame(device, platform, UNITS);
            if (frame % 8 === 7) await device.queue.onSubmittedWorkDone();
        }
        // Let the last trees and the top level catch up, then a few more frames.
        await platform.rays.whenBuilt();
        for (let frame = 0; frame < 8; frame++) runFrame(device, platform, UNITS);
        await device.queue.onSubmittedWorkDone();
        api.destroyMesh(sharedMesh);
        runFrame(device, platform, UNITS);
        await device.queue.onSubmittedWorkDone();

        const after = recompiles();
        assert(created === INSTANCES && destroyed === INSTANCES, `created ${created}, destroyed ${destroyed}`);
        assert(meshesMade >= NEW_MESHES / 2, `only ${meshesMade} new meshes were made`);
        assert(livePeak >= MIN_LIVE_PEAK, `only ${livePeak} instances were live at once; the table never had to grow`);
        assert(after === before && after === 0, `the counter went from ${before} to ${after} during the churn`);

        // The control: a pipeline made through the platform's kernel builder while it is ready counts.
        await platform.kernels.computeKernel({ label: 'recompile control', modules: [], code: '@compute @workgroup_size(1) fn main() {}', entryPoint: 'main' });
        const control = recompiles();
        assert(control === 1, `the control pipeline was counted as ${control}`);
        log(`${created} instances created and destroyed (${meshesMade} new meshes among them, at most ${livePeak} live, ${moved} moves); recompiles ${before} -> ${after}; the control kernel counts: ${control}`);
        platform.dispose();
        return { created, destroyed, livePeak, moved, newMeshes: meshesMade, recompilesBefore: before, recompilesAfter: after, controlCounted: control };
    },
    { optionalFeatures: ['texture-formats-tier1', 'subgroups'] },
);
