/**
 * A manifest scene started the way the viewer starts it (T1.9 measurements):
 * the files loaded with the viewer's `loadScene`, DDGI on the manifest's
 * volume through the three adapter, the mesh trees built, and optionally the
 * warm-up, which since T2.4b is the technique's own (`warmUpPasses`; the
 * viewer runs it through `volume.start`). The technique is made with the
 * warm-up off, so a page's frames run at the budget it gives them until it
 * calls `warmUp`. Used by the `ddgi-memory`, `ddgi-flicker`, `ddgi-settle`
 * and `ddgi-batch-cost` pages.
 *
 * Not a page itself (no index.html).
 */

import { PerspectiveCamera } from 'three/webgpu';
import { GIPlatformImpl } from '../../../../packages/ddgi-core/src/platform/gi-platform.ts';
import type { EnvironmentDescriptor } from '../../../../packages/ddgi-core/src/platform/contract/scene.ts';
import { DDGITechnique } from '../../../../packages/ddgi-core/src/techniques/ddgi/ddgi-technique.ts';
import { DEFAULT_WARM_UP_PASSES } from '../../../../packages/ddgi-core/src/techniques/ddgi/settings.ts';
import { ThreeGI } from '../../../../packages/ddgi-three/src/three-gi.ts';
import { loadScene, type LoadedScene } from '../../../../apps/viewer/src/scene/load-scene.ts';
import type { SceneManifest } from '../../../../scenes/src/manifest.ts';
import { platformSkyEnvironment } from '../../../../scenes/src/sky/sky-image.ts';

/** Warm-up frames recorded before waiting for the GPU, as `volume.start` does. */
const WARM_UP_STEPS_PER_BATCH = 4;

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
export async function startViewerScene(
    device: GPUDevice,
    manifest: SceneManifest,
    options: { readonly gpuTiming: boolean; readonly label: string; readonly prepareScene?: (loaded: LoadedScene) => void },
): Promise<ViewerScene> {
    const loaded = await loadScene(manifest, () => {});
    // A page's own additions (a lamp to switch on later) go in before the adapter's first sync.
    options.prepareScene?.(loaded);
    const bookmark = manifest.cameras[0];
    const camera = new PerspectiveCamera(bookmark.fovYDeg, 1, 0.05, loaded.diagonal * 2);
    camera.position.set(...bookmark.position);
    camera.lookAt(...bookmark.target);
    camera.updateMatrixWorld();

    const platform = new GIPlatformImpl(device, { label: options.label, gpuTiming: options.gpuTiming });
    const volume = manifest.volume;
    const ddgi = platform.addTechnique(new DDGITechnique({ volume: { center: volume.center, size: volume.size, spacing: volume.probeSpacing }, warmUpPasses: 0 }));
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
 * The warm-up as the viewer runs it since T2.4b: the technique's own
 * (`warmUpPasses`, its default 16 passes of `maxProbesPerFrame` probes with
 * no time between them, stopping after 8 s once 2 are done), started on the
 * fresh field and run back to back in batches of a few frames, as
 * `volume.start` does. The frames' budget is ignored while it runs. Returns
 * the passes made and the milliseconds taken; the setting is back at 0 after.
 */
export async function warmUp(scene: ViewerScene): Promise<{ passes: number; ms: number }> {
    const { ddgi, device } = scene;
    ddgi.settings.warmUpPasses = DEFAULT_WARM_UP_PASSES;
    // A new warm-up on the field as it is (black after startViewerScene): resetHistory starts one.
    ddgi.resetHistory();
    const startedAt = performance.now();
    let steps = 0;
    try {
        while (ddgi.warmingUp) {
            scene.frame({ paused: false }, 0);
            steps++;
            if (steps % WARM_UP_STEPS_PER_BATCH === 0) {
                await device.queue.onSubmittedWorkDone();
                await yieldToCallbacks();
            }
        }
        await device.queue.onSubmittedWorkDone();
        await yieldToCallbacks();
    } finally {
        ddgi.settings.warmUpPasses = 0;
    }
    return { passes: ddgi.counters['warm-up passes'] ?? 0, ms: performance.now() - startedAt };
}
