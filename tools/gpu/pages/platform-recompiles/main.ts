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
 *
 * T2.1 check 1 (docs/design/m2-lights.md §1, light buffer v2): then LIGHTS
 * lights of every type are created over LIGHT_FRAMES frames and destroyed
 * again in random order, with some edited on the way, still with DDGI
 * updating. The light table starts with room for 64 and doubles to 4,096 on
 * the way, and the counter must stay 0 through that too.
 */

import { assert, runGpuTest } from '../../client.ts';
import type { InstanceHandle, LightHandle, MeshHandle } from '../../../../packages/ddgi-core/src/platform/contract/handles.ts';
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
/** T2.1 check 1: lights created and destroyed, all live at once at the peak (the table doubles from 64 to 4,096). */
const LIGHTS = 4096;
const LIGHT_FRAMES = 64;
/** One light in this many is directional, one in this many a spot; the rest are points. */
const DIRECTIONAL_EVERY = 64;
const SPOT_EVERY = 3;

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

        // T2.1 check 1: 4,096 lights made (all live at once at the peak), edited, then destroyed in random order.
        const sceneLights = platform.sceneCopy.counts.lights;
        const lights: LightHandle[] = [];
        const lightKinds = { directional: 0, point: 0, spot: 0 };
        let lightsEdited = 0;
        const position = (): [number, number, number] => [0.2 + random() * 3.6, 0.2 + random() * 2.4, 0.2 + random() * 2.6];
        const color = (): [number, number, number] => [0.2 + random() * 0.8, 0.2 + random() * 0.8, 0.2 + random() * 0.8];
        const makeLight = (index: number): LightHandle => {
            // A few with no range (range 0), as three's default distance gives; the rest small boxes.
            const range = random() < 0.1 ? 0 : 0.5 + random() * 2;
            if (index % DIRECTIONAL_EVERY === 0) {
                lightKinds.directional++;
                return api.createLight({ type: 'directional', color: color(), intensity: 0.01, direction: [random() - 0.5, -1, random() - 0.5], label: `churn sun ${index}` });
            }
            if (index % SPOT_EVERY === 0) {
                lightKinds.spot++;
                return api.createLight({ type: 'spot', color: color(), intensity: 0.05, position: position(), direction: [0, -1, 0], range, innerConeAngle: 0.3, outerConeAngle: 0.6, label: `churn spot ${index}` });
            }
            lightKinds.point++;
            return api.createLight({ type: 'point', color: color(), intensity: 0.05, position: position(), range, label: `churn lamp ${index}` });
        };
        for (let frame = 0; frame < LIGHT_FRAMES; frame++) {
            const target = Math.round(((frame + 1) / LIGHT_FRAMES) * LIGHTS);
            while (lights.length < target) lights.push(makeLight(lights.length));
            runFrame(device, platform, UNITS);
            if (frame % 8 === 7) await device.queue.onSubmittedWorkDone();
        }
        const lightPeak = lights.length;
        const lightsAtPeak = platform.sceneCopy.counts.lights;
        let lightsDestroyed = 0;
        while (lights.length > 0) {
            const toDestroy = Math.min(lights.length, 1 + Math.floor(random() * 2 * (LIGHTS / LIGHT_FRAMES)));
            for (let index = 0; index < toDestroy; index++) {
                const [gone] = lights.splice(Math.floor(random() * lights.length), 1);
                api.destroyLight(gone);
                lightsDestroyed++;
            }
            // Edit a few of the ones left (their packed index moves as others go).
            for (let index = 0; index < 8 && lights.length > 0; index++) {
                api.updateLight(lights[Math.floor(random() * lights.length)], { intensity: random() * 0.05 });
                lightsEdited++;
            }
            runFrame(device, platform, UNITS);
            if (lightsDestroyed % 512 < toDestroy) await device.queue.onSubmittedWorkDone();
        }
        runFrame(device, platform, UNITS);
        await device.queue.onSubmittedWorkDone();
        const afterLights = recompiles();

        assert(created === INSTANCES && destroyed === INSTANCES, `created ${created}, destroyed ${destroyed}`);
        assert(meshesMade >= NEW_MESHES / 2, `only ${meshesMade} new meshes were made`);
        assert(livePeak >= MIN_LIVE_PEAK, `only ${livePeak} instances were live at once; the table never had to grow`);
        assert(after === before && after === 0, `the counter went from ${before} to ${after} during the churn`);
        assert(lightPeak === LIGHTS && lightsAtPeak === sceneLights + LIGHTS && lightsDestroyed === LIGHTS, `lights: ${lightPeak} made, ${lightsAtPeak} in the scene at the peak (${sceneLights} its own), ${lightsDestroyed} destroyed`);
        assert(platform.sceneCopy.counts.lights === sceneLights, `${platform.sceneCopy.counts.lights} lights left after the churn (the scene had ${sceneLights})`);
        assert(afterLights === 0, `the counter went from ${after} to ${afterLights} while ${LIGHTS} lights were made and destroyed`);

        // The control: a pipeline made through the platform's kernel builder while it is ready counts.
        await platform.kernels.computeKernel({ label: 'recompile control', modules: [], code: '@compute @workgroup_size(1) fn main() {}', entryPoint: 'main' });
        const control = recompiles();
        assert(control === 1, `the control pipeline was counted as ${control}`);
        log(`${created} instances created and destroyed (${meshesMade} new meshes among them, at most ${livePeak} live, ${moved} moves); recompiles ${before} -> ${after}`);
        log(
            `${LIGHTS} lights (${lightKinds.directional} directional, ${lightKinds.point} point, ${lightKinds.spot} spot) created, all live at once, ${lightsEdited} edits, ` +
                `destroyed in random order; recompiles ${after} -> ${afterLights}; the control kernel counts: ${control}`,
        );
        platform.dispose();
        return {
            created,
            destroyed,
            livePeak,
            moved,
            newMeshes: meshesMade,
            recompilesBefore: before,
            recompilesAfter: after,
            lights: { created: lightPeak, destroyed: lightsDestroyed, edited: lightsEdited, kinds: lightKinds, recompilesAfter: afterLights },
            controlCounted: control,
        };
    },
    { optionalFeatures: ['texture-formats-tier1', 'subgroups'] },
);
