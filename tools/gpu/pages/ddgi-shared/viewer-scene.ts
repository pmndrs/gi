/**
 * A manifest scene started the way the viewer starts it (T1.9 measurements):
 * the files loaded with the viewer's `loadScene`, DDGI on the manifest's
 * volume through the three adapter, the mesh trees built, and optionally the
 * viewer's warm-up (apps/viewer/src/techniques/ddgi-technique.ts, DECISIONS
 * 2026-11-05). Used by the `ddgi-memory` and `ddgi-flicker` pages.
 *
 * Not a page itself (no index.html).
 */

import { PerspectiveCamera } from 'three/webgpu';
import { GIPlatformImpl } from '../../../../packages/ddgi-core/src/platform/gi-platform.ts';
import type { EnvironmentDescriptor } from '../../../../packages/ddgi-core/src/platform/contract/scene.ts';
import { DDGITechnique } from '../../../../packages/ddgi-core/src/techniques/ddgi/ddgi-technique.ts';
import { ThreeGI } from '../../../../packages/ddgi-three/src/three-gi.ts';
import { loadScene, type LoadedScene } from '../../../../apps/viewer/src/scene/load-scene.ts';
import type { SceneManifest } from '../../../../scenes/src/manifest.ts';
import { platformSkyEnvironment } from '../../../../scenes/src/sky/sky-image.ts';

/** The viewer's warm-up rule: at most 16 passes over every probe, stopping after 8 s once 2 passes are done. */
const WARMUP = { passes: 16, maxSeconds: 8, minPasses: 2, stepsPerBatch: 4 } as const;
const PROBES_UPDATED = 'probes updated';

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

export interface ViewerScene {
    readonly manifest: SceneManifest;
    readonly loaded: LoadedScene;
    readonly platform: GIPlatformImpl;
    readonly ddgi: DDGITechnique;
    readonly threeGI: ThreeGI;
    readonly camera: PerspectiveCamera;
    readonly probeCount: number;
    readonly device: GPUDevice;
    /**
     * One frame through the adapter with this budget. `deltaTime` (seconds,
     * default 1/60, a 60 Hz host) also advances DDGI's clock, which adaptive
     * hysteresis measures the time between a probe's updates on.
     */
    frame(budget: ThreeGI['budget'], deltaTime?: number): void;
    /** Hands a camera bookmark (1-based) to the adapter's camera. */
    lookFrom(bookmarkIndex: number): void;
    dispose(): void;
}

/** Loads the scene and creates DDGI with the trees built; the probes are black until frames run. */
export async function startViewerScene(device: GPUDevice, manifest: SceneManifest, options: { readonly gpuTiming: boolean; readonly label: string }): Promise<ViewerScene> {
    const loaded = await loadScene(manifest, () => {});
    const bookmark = manifest.cameras[0];
    const camera = new PerspectiveCamera(bookmark.fovYDeg, 1, 0.05, loaded.diagonal * 2);
    camera.position.set(...bookmark.position);
    camera.lookAt(...bookmark.target);
    camera.updateMatrixWorld();

    const platform = new GIPlatformImpl(device, { label: options.label, gpuTiming: options.gpuTiming });
    const volume = manifest.volume;
    const ddgi = platform.addTechnique(new DDGITechnique({ volume: { center: volume.center, size: volume.size, spacing: volume.probeSpacing } }));
    platform.setOutput(ddgi);
    await platform.prepare();
    const threeGI = new ThreeGI(platform, loaded.scene, { budget: { paused: true }, environment: environmentFor(manifest), warn: () => {} });
    threeGI.update(camera, 0);
    await platform.rays.whenBuilt();
    threeGI.update(camera, 0);
    await device.queue.onSubmittedWorkDone();
    const probeCount = ddgi.field?.grid.probeCount ?? 0;
    return {
        manifest,
        loaded,
        platform,
        ddgi,
        threeGI,
        camera,
        probeCount,
        device,
        frame(budget, deltaTime = 1 / 60) {
            threeGI.budget = budget;
            threeGI.update(camera, deltaTime);
        },
        lookFrom(bookmarkIndex) {
            const view = manifest.cameras[bookmarkIndex - 1];
            if (!view) throw new Error(`${manifest.id} has no camera ${bookmarkIndex}`);
            camera.position.set(...view.position);
            camera.lookAt(...view.target);
            camera.updateMatrixWorld();
        },
        dispose() {
            threeGI.dispose();
            platform.dispose();
        },
    };
}

/** Lets mapAsync callbacks (timestamps, census) run before the next encode collects them. */
export function yieldToCallbacks(): Promise<void> {
    return new Promise((resolve) => setTimeout(resolve, 0));
}

/**
 * The viewer's warm-up: steps of `maxProbesPerFrame` probes with a
 * running-average hysteresis, back to back with `deltaTime` 0 as the viewer
 * runs it (so adaptive hysteresis counts the steps with no time between them,
 * the same running average). Returns the passes made and the milliseconds taken.
 */
export async function warmUp(scene: ViewerScene): Promise<{ passes: number; ms: number }> {
    const { ddgi, probeCount, device } = scene;
    const targetHysteresis = ddgi.settings.hysteresis;
    const stepUnits = Math.min(probeCount, ddgi.settings.maxProbesPerFrame);
    let updates = 0;
    let steps = 0;
    const startedAt = performance.now();
    for (;;) {
        const completed = Math.floor(updates / probeCount);
        ddgi.settings.hysteresis = Math.min(targetHysteresis, completed / (completed + 1));
        scene.frame({ units: stepUnits }, 0);
        updates += ddgi.counters[PROBES_UPDATED] ?? 0;
        steps++;
        const done = updates / probeCount >= WARMUP.passes || (performance.now() - startedAt > WARMUP.maxSeconds * 1000 && updates / probeCount >= WARMUP.minPasses);
        if (done || steps % WARMUP.stepsPerBatch === 0) {
            await device.queue.onSubmittedWorkDone();
            await yieldToCallbacks();
        }
        if (done) break;
    }
    ddgi.settings.hysteresis = targetHysteresis;
    return { passes: updates / probeCount, ms: performance.now() - startedAt };
}
