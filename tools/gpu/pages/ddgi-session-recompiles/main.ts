/**
 * T2.14 check 2 (docs/design/m3-moving.md, "Startup lane"): the recompile
 * counter (`gi.stats.recompiles`, T2.M1: compute pipelines the platform and
 * its techniques made while ready) stays at 0 through a viewer session on
 * every scene. Every kernel a configuration needs is compiled in `prepare`,
 * so nothing a session does may compile one.
 *
 * Per scene, and per light-group setting (off, as the viewer runs today, and
 * on with 2 groups, so T2.5's layers and compose run), GI starts exactly as
 * the viewer starts it (`GIVolume.start`, since T2.14 with the scene copied
 * in while the kernels compile), then the page runs the GI frames itself as
 * the viewer does after start-up (`threeGI.update`, then three's render with
 * the GI light) through the steps of a session:
 *
 * - still frames;
 * - each light dimmed, recoloured, switched off and on, and moved over
 *   several frames (point and spot lights), then put back; a new point light
 *   added and later removed; MANY_LIGHTS point lights added and removed, so
 *   the light grid (`gi/light-grid`, T2.3: more point and spot lights than
 *   `exactLightLimit`) switches on and off mid-session;
 * - an object moved over several frames and back (T2.M1's rigid moves),
 *   then hidden and shown;
 * - the sky: a procedural sky's sun moving for 2 s (time of day: key probes,
 *   T2.4c, and T2.4d's continuing-change batch), and dimmed; a colour
 *   environment dimmed; a scene with none gets a colour one and loses it;
 * - with groups on, each group's weight and the environment's weight changed;
 * - what the viewer can turn on: the budget (1 ms, 4 ms, GI off, a fixed
 *   count), `recalibrate`, the probe debug view's texture, "start the probes
 *   over" (`GIVolume.restart`: the warm-up and the calibration again), and
 *   `resetHistory`.
 *
 * The page fails if the counter reads anything but 0 after any step, naming
 * the step. As a control it then compiles one kernel through the platform's
 * kernel builder, which must count 1. Each step also reports the frames whose
 * newest GPU timings include `gi/light-grid`, so a reader sees where the grid
 * ran. three's own pipelines made during the session are reported beside it (information: three compiles its materials
 * again when a light is added, say).
 *
 * Choose with environment variables (vite passes `VITE_*` to the page):
 *   VITE_DDGI_SCENES=cornell-box,two-room,sponza   (default cornell-box,two-room; `all` = every manifest)
 *   VITE_DDGI_GROUPS=off,on                          (default both)
 * Sponza, Bistro and the furnished room need the downloads.
 */

import { Color, Mesh, PerspectiveCamera, PointLight, RectAreaLight, SpotLight, Vector3, type Light, type Object3D } from 'three/webgpu';
import type { GIPlatformImpl } from '../../../../packages/ddgi-core/src/platform/gi-platform.ts';
import type { EnvironmentDescriptor } from '../../../../packages/ddgi-core/src/platform/contract/scene.ts';
import type { DDGITechnique } from '../../../../packages/ddgi-core/src/techniques/ddgi/ddgi-technique.ts';
import { GIVolume } from '../../../../packages/ddgi-three/src/gi-volume.ts';
import type { StartedGI } from '../../../../packages/ddgi-three/src/volume-start.ts';
import { loadAllManifests } from '../../../../apps/viewer/src/scene/manifests.ts';
import { disposeScene, loadScene, type LoadedScene } from '../../../../apps/viewer/src/scene/load-scene.ts';
import type { SceneManifest } from '../../../../scenes/src/manifest.ts';
import { platformSkyEnvironment } from '../../../../scenes/src/sky/sky-image.ts';
import { assert } from '../../client.ts';
import { runThreePage } from '../three-shared/harness.ts';

const env = import.meta.env as Record<string, unknown>;

function listFromEnv(value: unknown, fallback: readonly string[]): string[] {
    return typeof value === 'string' && value.trim() !== '' ? value.split(',').map((item) => item.trim()) : [...fallback];
}

const SCENES = listFromEnv(env.VITE_DDGI_SCENES, ['cornell-box', 'two-room']);
const GROUP_SETTINGS = listFromEnv(env.VITE_DDGI_GROUPS, ['off', 'on']).map((setting) => {
    if (setting !== 'off' && setting !== 'on') throw new Error(`VITE_DDGI_GROUPS: '${setting}' is neither off nor on`);
    return setting;
});

/** Light groups when on: 2 groups (lights alternate between them), so DDGI has 3 layers and composes. */
const GROUP_COUNT = 2;
/** The viewer's default budget. */
const BUDGET_MS = 2;
/** Frames after each change, so its marks, key probes and updates all run. */
const FRAMES_AFTER_CHANGE = 20;
/** Frames a continuing change (a light or an object moving, the sun) lasts. */
const MOVING_FRAMES = 30;
/** The sun's time of day: degrees of elevation it climbs over SUN_FRAMES frames. */
const SUN_DEGREES = 20;
const SUN_FRAMES = 120;
/** Point lights added at once: more than the platform's `exactLightLimit` (8), so the light grid runs. */
const MANY_LIGHTS = 12;
/** The light grid's pass name in `stats.passes` (light-grid.ts, `LIGHT_GRID_PASS`). */
const LIGHT_GRID_PASS = 'gi/light-grid';
/** Wait for the GPU every this many frames, and let the timestamp read-backs land. */
const FRAMES_PER_WAIT = 8;

/** The manifest's environment as the viewer gives it to the platform. */
function environmentFor(manifest: SceneManifest): EnvironmentDescriptor {
    const environment = manifest.environment;
    switch (environment.type) {
        case 'none':
            return { type: 'none' };
        case 'color':
            return { type: 'color', color: environment.color, intensity: environment.intensity };
        case 'procedural':
            return platformSkyEnvironment(manifest);
    }
}

function nextTask(): Promise<void> {
    return new Promise((resolve) => setTimeout(resolve, 0));
}

interface Session {
    readonly loaded: LoadedScene;
    readonly volume: GIVolume;
    readonly started: StartedGI;
    readonly platform: GIPlatformImpl;
    readonly technique: DDGITechnique;
    readonly camera: PerspectiveCamera;
    readonly device: GPUDevice;
}

/** What one step left on the counters. */
interface StepRecord {
    readonly step: string;
    readonly frames: number;
    readonly recompiles: number;
    readonly threeRenderPipelines: number;
    readonly threeComputePipelines: number;
    /** Frames whose newest GPU timings had a `gi/light-grid` pass (timings run a few frames behind). */
    readonly lightGridFrames: number;
}

await runThreePage(
    async ({ device, renderer, pipelines, data, log }) => {
        const manifests = loadAllManifests();
        const sceneIds = SCENES.length === 1 && SCENES[0] === 'all' ? manifests.map((manifest) => manifest.id) : SCENES;
        const results: Record<string, unknown> = {};
        data.scenes = results;
        const failures: string[] = [];

        for (const sceneId of sceneIds) {
            const manifest = manifests.find((candidate) => candidate.id === sceneId);
            assert(manifest, `no manifest '${sceneId}'`);
            for (const groups of GROUP_SETTINGS) {
                const key = `${sceneId}/groups ${groups}`;
                const session = await startSession(device, renderer, manifest, groups === 'on');
                const steps: StepRecord[] = [];
                const recompiles = (): number => session.platform.stats.recompiles;
                let renderBefore = pipelines.render;
                let computeBefore = pipelines.compute;
                let frameCount = 0;
                let lightGridFrames = 0;
                const frames = async (count: number, each?: (frame: number) => void): Promise<void> => {
                    for (let frame = 0; frame < count; frame++) {
                        each?.(frame);
                        session.started.threeGI.update(session.camera, 1 / 60);
                        renderer.render(session.loaded.scene, session.camera);
                        frameCount++;
                        if (session.platform.stats.passes.some((pass) => pass.name === LIGHT_GRID_PASS)) lightGridFrames++;
                        if (frameCount % FRAMES_PER_WAIT === 0) {
                            await device.queue.onSubmittedWorkDone();
                            await nextTask();
                        }
                    }
                };
                const step = async (name: string, run: () => Promise<void>): Promise<void> => {
                    const framesBefore = frameCount;
                    const gridFramesBefore = lightGridFrames;
                    await run();
                    await device.queue.onSubmittedWorkDone();
                    const record: StepRecord = {
                        step: name,
                        frames: frameCount - framesBefore,
                        recompiles: recompiles(),
                        threeRenderPipelines: pipelines.render - renderBefore,
                        threeComputePipelines: pipelines.compute - computeBefore,
                        lightGridFrames: lightGridFrames - gridFramesBefore,
                    };
                    renderBefore = pipelines.render;
                    computeBefore = pipelines.compute;
                    steps.push(record);
                    if (record.recompiles !== 0) failures.push(`${key}: the counter read ${record.recompiles} after '${name}'`);
                };

                await step('started (GIVolume.start: warm-up and calibration)', async () => {});
                await step('still', () => frames(FRAMES_AFTER_CHANGE));
                await runLightSteps(session, step, frames);
                await runObjectSteps(session, step, frames);
                await runSkySteps(session, step, frames);
                if (groups === 'on') await runGroupSteps(session, step, frames);
                await runViewerSteps(session, step, frames);

                const final = recompiles();
                log(`${key}: ${steps.length} steps, ${frameCount} frames, recompiles ${final}; three made ${steps.reduce((sum, record) => sum + record.threeRenderPipelines, 0)} render pipelines meanwhile`);
                // The control: a kernel made through the platform now counts, so the 0 above means what it says.
                await session.platform.kernels.computeKernel({ label: 'session control', modules: [], code: '@compute @workgroup_size(1) fn sessionControl() {}', entryPoint: 'sessionControl' });
                const control = recompiles() - final;
                if (control !== 1) failures.push(`${key}: the control kernel counted ${control}, not 1`);
                results[key] = { recompiles: final, control, frames: frameCount, steps };

                session.volume.dispose();
                disposeScene(session.loaded);
                await device.queue.onSubmittedWorkDone();
            }
        }
        assert(failures.length === 0, failures.join('; '));
    },
    { width: 320, height: 180 },
);

/** Loads the scene and starts GI as the viewer does (apps/viewer/src/techniques/ddgi-technique.ts). */
async function startSession(device: GPUDevice, renderer: Parameters<GIVolume['start']>[0], manifest: SceneManifest, groupsOn: boolean): Promise<Session> {
    const loaded = await loadScene(manifest, () => {});
    if (groupsOn) {
        // Lights alternate between the groups (the sync reads the group when it first sees a light).
        loaded.directLights.forEach((light, index) => (light.userData.giLightGroup = index % GROUP_COUNT));
    }
    const bookmark = manifest.cameras[0];
    const camera = new PerspectiveCamera(bookmark.fovYDeg, 16 / 9, 0.05, loaded.diagonal * 2);
    camera.position.set(...bookmark.position);
    camera.lookAt(...bookmark.target);
    camera.updateMatrixWorld();
    const volume = GIVolume.fromDDGIVolume({ center: manifest.volume.center, size: manifest.volume.size, spacing: manifest.volume.probeSpacing });
    let started: StartedGI | null = null;
    await volume.start(renderer, loaded.scene, {
        budget: { gpuMs: BUDGET_MS },
        environment: environmentFor(manifest),
        ddgi: groupsOn ? { lightGroups: { count: GROUP_COUNT, weights: [], environmentWeight: [1, 1, 1] } } : {},
        label: `session ${manifest.id}`,
        warn: () => {},
        onCreated: (created) => (started = created),
    });
    assert(started, 'GIVolume.start made no platform');
    const gi = started as StartedGI;
    // The viewer runs the GI frames itself after start-up.
    gi.threeGI.updateOnRender = false;
    return { loaded, volume, started: gi, platform: gi.platform as GIPlatformImpl, technique: gi.technique, camera, device };
}

type StepRunner = (name: string, run: () => Promise<void>) => Promise<void>;
type FrameRunner = (count: number, each?: (frame: number) => void) => Promise<void>;

/** Lights GI mirrors: three's area lights aren't (the adapter warns), so changing them changes nothing in GI. */
function mirroredLights(session: Session): Light[] {
    return session.loaded.directLights.filter((light) => !(light instanceof RectAreaLight));
}

async function runLightSteps(session: Session, step: StepRunner, frames: FrameRunner): Promise<void> {
    const { threeGI } = session.started;
    const lights = mirroredLights(session);
    const changed = (light: Light) => threeGI.markChanged(light);
    await step(`lights dimmed (${lights.length})`, async () => {
        for (const light of lights) {
            light.intensity *= 0.5;
            changed(light);
        }
        await frames(FRAMES_AFTER_CHANGE);
    });
    await step('lights recoloured', async () => {
        for (const light of lights) {
            light.userData.sessionColor = light.color.clone();
            light.color.multiply(new Color(1, 0.6, 0.3));
            changed(light);
        }
        await frames(FRAMES_AFTER_CHANGE);
    });
    await step('lights off', async () => {
        for (const light of lights) {
            light.userData.sessionIntensity = light.intensity;
            light.intensity = 0;
            changed(light);
        }
        await frames(FRAMES_AFTER_CHANGE);
    });
    await step('lights back on, as they were', async () => {
        for (const light of lights) {
            light.intensity = (light.userData.sessionIntensity as number) * 2;
            light.color.copy(light.userData.sessionColor as Color);
            changed(light);
        }
        await frames(FRAMES_AFTER_CHANGE);
    });
    const local = lights.filter((light) => light instanceof PointLight || light instanceof SpotLight);
    if (local.length > 0) {
        const offset = new Vector3(0.01, 0.005, 0).multiplyScalar(session.loaded.diagonal / MOVING_FRAMES);
        await step(`local lights moving (${local.length})`, () =>
            frames(MOVING_FRAMES, () => {
                for (const light of local) {
                    light.position.add(offset);
                    light.updateMatrixWorld();
                    changed(light);
                }
            }),
        );
        await step('local lights back', async () => {
            for (const light of local) {
                light.position.addScaledVector(offset, -MOVING_FRAMES);
                light.updateMatrixWorld();
                changed(light);
            }
            await frames(FRAMES_AFTER_CHANGE);
        });
    }
    const many = Array.from({ length: MANY_LIGHTS }, (_, index) => {
        const light = new PointLight(0xffd8b0, 0.5, session.loaded.diagonal * 0.25);
        const [x, y, z] = session.loaded.manifest.volume.center;
        const angle = (index / MANY_LIGHTS) * Math.PI * 2;
        light.position.set(x + Math.cos(angle) * 0.2 * session.loaded.diagonal, y, z + Math.sin(angle) * 0.2 * session.loaded.diagonal);
        return light;
    });
    await step(`${MANY_LIGHTS} point lights added (the light grid on)`, async () => {
        for (const light of many) {
            session.loaded.scene.add(light);
            changed(light);
        }
        await frames(FRAMES_AFTER_CHANGE);
    });
    await step(`the ${MANY_LIGHTS} lights removed`, async () => {
        for (const light of many) light.removeFromParent();
        threeGI.markChanged(session.loaded.scene);
        await frames(FRAMES_AFTER_CHANGE);
        for (const light of many) light.dispose();
    });
    const added = new PointLight(0xffeedd, 2, session.loaded.diagonal * 0.5);
    added.position.set(...session.loaded.manifest.volume.center);
    await step('a point light added', async () => {
        session.loaded.scene.add(added);
        changed(added);
        await frames(FRAMES_AFTER_CHANGE);
    });
    await step('the added light removed', async () => {
        added.removeFromParent();
        threeGI.markChanged(session.loaded.scene);
        await frames(FRAMES_AFTER_CHANGE);
        added.dispose();
    });
}

/** An object to move: the smallest mesh with a size of its own (not a wall the size of the scene). */
function movableObject(session: Session): Object3D | null {
    let best: { object: Mesh; size: number } | null = null;
    session.loaded.content.traverse((object) => {
        if (!(object instanceof Mesh) || !object.visible) return;
        object.geometry.computeBoundingSphere();
        const radius = object.geometry.boundingSphere?.radius ?? 0;
        const size = radius * object.getWorldScale(new Vector3()).length();
        if (size <= 0.02 * session.loaded.diagonal || size > 0.25 * session.loaded.diagonal) return;
        if (!best || size < best.size) best = { object, size };
    });
    return (best as { object: Mesh; size: number } | null)?.object ?? null;
}

async function runObjectSteps(session: Session, step: StepRunner, frames: FrameRunner): Promise<void> {
    const { threeGI } = session.started;
    const object = movableObject(session);
    if (!object) return;
    const offset = new Vector3(0.02 * session.loaded.diagonal, 0, 0).divideScalar(MOVING_FRAMES);
    await step(`an object moving ('${object.name || object.uuid}')`, () =>
        frames(MOVING_FRAMES, () => {
            object.position.add(offset);
            object.updateMatrixWorld();
            threeGI.markChanged(object);
        }),
    );
    await step('the object back', async () => {
        object.position.addScaledVector(offset, -MOVING_FRAMES);
        object.updateMatrixWorld();
        threeGI.markChanged(object);
        await frames(FRAMES_AFTER_CHANGE);
    });
    await step('the object hidden, then shown', async () => {
        object.visible = false;
        threeGI.markChanged(object);
        await frames(FRAMES_AFTER_CHANGE);
        object.visible = true;
        threeGI.markChanged(object);
        await frames(FRAMES_AFTER_CHANGE);
    });
}

async function runSkySteps(session: Session, step: StepRunner, frames: FrameRunner): Promise<void> {
    const { threeGI } = session.started;
    const original = threeGI.environment;
    if (original?.type === 'procedural') {
        const start = new Vector3(...(original.sunDirection ?? [0, 1, 0])).normalize();
        await step(`the sun moving (${SUN_DEGREES} degrees over ${SUN_FRAMES} frames)`, () =>
            frames(SUN_FRAMES, (frame) => {
                // Up toward the zenith, keeping the azimuth (the sun's direction is where it is).
                const elevation = Math.asin(Math.min(1, Math.max(-1, start.y))) + (((frame + 1) / SUN_FRAMES) * SUN_DEGREES * Math.PI) / 180;
                const horizontal = new Vector3(start.x, 0, start.z).normalize();
                const sun: [number, number, number] = [horizontal.x * Math.cos(elevation), Math.sin(elevation), horizontal.z * Math.cos(elevation)];
                threeGI.environment = { ...original, sunDirection: sun };
            }),
        );
        await step('the sky dimmed', async () => {
            threeGI.environment = { ...original, intensity: (original.intensity ?? 1) * 0.5 };
            await frames(FRAMES_AFTER_CHANGE);
        });
    } else if (original?.type === 'color') {
        await step('the environment dimmed', async () => {
            threeGI.environment = { ...original, intensity: (original.intensity ?? 1) * 0.5 };
            await frames(FRAMES_AFTER_CHANGE);
        });
    } else {
        await step('a colour environment added', async () => {
            threeGI.environment = { type: 'color', color: [0.4, 0.5, 0.7], intensity: 0.5 };
            await frames(FRAMES_AFTER_CHANGE);
        });
    }
    await step('the sky as it was', async () => {
        threeGI.environment = original;
        await frames(FRAMES_AFTER_CHANGE);
    });
}

async function runGroupSteps(session: Session, step: StepRunner, frames: FrameRunner): Promise<void> {
    const { threeGI } = session.started;
    await step('group weights changed', async () => {
        threeGI.setGroupWeight(0, 'orange', 0.5);
        await frames(FRAMES_AFTER_CHANGE);
        threeGI.setGroupWeight(1, 'white', 2);
        await frames(FRAMES_AFTER_CHANGE);
    });
    await step('the environment weight changed', async () => {
        threeGI.setEnvironmentWeight('skyblue', 0.25);
        await frames(FRAMES_AFTER_CHANGE);
    });
    await step('weights back to 1', async () => {
        threeGI.setGroupWeight(0, 'white', 1);
        threeGI.setGroupWeight(1, 'white', 1);
        threeGI.setEnvironmentWeight('white', 1);
        await frames(FRAMES_AFTER_CHANGE);
    });
}

async function runViewerSteps(session: Session, step: StepRunner, frames: FrameRunner): Promise<void> {
    const { threeGI } = session.started;
    await step('budget 1 ms, then 4 ms', async () => {
        threeGI.budget = { gpuMs: 1 };
        await frames(FRAMES_AFTER_CHANGE);
        threeGI.budget = { gpuMs: 4 };
        await frames(FRAMES_AFTER_CHANGE);
    });
    await step('GI off (paused), then on', async () => {
        threeGI.budget = { paused: true };
        await frames(FRAMES_AFTER_CHANGE);
        threeGI.budget = { gpuMs: BUDGET_MS };
        await frames(FRAMES_AFTER_CHANGE);
    });
    await step('a fixed count (64 probes)', async () => {
        threeGI.budget = { units: 64 };
        await frames(FRAMES_AFTER_CHANGE);
        threeGI.budget = { gpuMs: BUDGET_MS };
    });
    await step('recalibrate', async () => {
        session.platform.recalibrate();
        await frames(FRAMES_AFTER_CHANGE * 4);
    });
    await step("the probes' debug view (its texture)", async () => {
        for (const view of session.technique.debugViews) {
            if (view.kind === 'points') assert(view.texture() !== null, 'the points debug view has no texture');
        }
        await frames(FRAMES_AFTER_CHANGE);
    });
    await step('start the probes over (GIVolume.restart)', async () => {
        await session.volume.restart({ budget: { gpuMs: BUDGET_MS } });
        session.started.threeGI.updateOnRender = false;
        await frames(FRAMES_AFTER_CHANGE);
    });
    await step('resetHistory', async () => {
        session.platform.resetHistory();
        await frames(FRAMES_AFTER_CHANGE);
    });
}
