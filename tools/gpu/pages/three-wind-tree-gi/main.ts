/**
 * T3.3 item 3 (TASKS M3, prototype level): the wind tree's sway reaches GI.
 * docs/design/m3-moving.md "Draft done-whens", T3.3.
 *
 * The scene is `courtyard-wind-tree` as the viewer loads it, through the
 * three adapter. The tree's three meshes (trunk, branches, leaves) are
 * marked `userData.giDeforming` with a `giDeformBounds` from the sway, and a
 * CPU sway driver (sway.ts) moves their vertices by the wind weight in
 * COLOR_0's alpha: the crown swings up to REACH metres along x, the foot not
 * at all. The adapter sends each pose to the platform, which refits the
 * trees on the GPU (T3.3).
 *
 * GI only (the platform's test view: albedo / π × field) is read on a ground
 * rectangle beside the trunk, on the +x side, which the crown shades more
 * when it leans that way.
 *
 * 1. **Extremes:** the tree held at sway +1 until GI settles, then at −1. The
 *    rectangle's mean GI must differ by at least EXTREME_DIFFERENCE.
 * 2. **Sway:** a sine of period PERIOD_S for SWAY_PERIODS periods at a fixed
 *    60 Hz step, the rectangle read every READ_EVERY frames. The lag at the
 *    largest cross-correlation of the two curves (the pose, signed by which
 *    extreme is brighter, and the rectangle's mean) must be at most
 *    MAX_LAG_S, and the correlation itself at least MIN_CORRELATION (or the
 *    GI doesn't follow the pose at all).
 *
 * The screenshot is three's frame at the two extremes and mid-sway, with the
 * rectangle outlined in the test view. Recompiles stay 0. Nothing is timed.
 *
 *   node tools/reference/bench-lock.ts run --no-cool-wait --what "T3.3 wind tree" -- pnpm test:gpu --gpu nvidia --filter three-wind-tree-gi
 */

import { PerspectiveCamera, type Mesh, type Object3D, type Scene, type WebGPURenderer } from 'three/webgpu';
import { assert } from '../../client.ts';
import { GIPlatformImpl } from '../../../../packages/ddgi-core/src/platform/gi-platform.ts';
import { DDGITechnique } from '../../../../packages/ddgi-core/src/techniques/ddgi/ddgi-technique.ts';
import type { EnvironmentDescriptor } from '../../../../packages/ddgi-core/src/platform/contract/scene.ts';
import { ThreeGI } from '../../../../packages/ddgi-three/src/three-gi.ts';
import { loadAllManifests } from '../../../../apps/viewer/src/scene/manifests.ts';
import { loadScene } from '../../../../apps/viewer/src/scene/load-scene.ts';
import { sceneLighting } from '../../../../apps/viewer/src/render/scene-lighting.ts';
import { platformSkyEnvironment } from '../../../../scenes/src/sky/sky-image.ts';
import type { CameraBookmark } from '../../../../scenes/src/manifest.ts';
import { TestView, type TestViewImages } from '../ddgi-shared/harness.ts';
import { runThreePage } from '../three-shared/harness.ts';
import { WindSway } from './sway.ts';

const WIDTH = 320;
const HEIGHT = 240;
/** Metres the crown (wind weight 1) moves at full sway: a strong wind for a 10 m tree. */
const REACH = 1.5;
const DIRECTION: [number, number, number] = [1, 0, 0];
/** The ground rectangle GI is read on (world x and z, metres), beside the trunk on the side the crown leans to at +1. */
const RECTANGLE = { minX: 0.6, maxX: 3.0, minZ: -1.2, maxZ: 1.2 };
/** Frames to settle at each extreme. */
const SETTLE_FRAMES = 600;
const BUDGET = { gpuMs: 4 };
/** The two extremes must differ in the rectangle's mean GI by at least this share. */
const EXTREME_DIFFERENCE = 0.02;
const PERIOD_S = 4;
const SWAY_PERIODS = 3;
const STEP_S = 1 / 60;
/** The rectangle is read every this many frames (a readback each; 15 a second is enough to find a lag within 0.5 s). */
const READ_EVERY = 4;
/** Lags tried, in reads: up to 1.5 s either way. */
const MAX_LAG_READS = Math.round(1.5 / (READ_EVERY * STEP_S));
const MAX_LAG_S = 0.5;
const MIN_CORRELATION = 0.5;
/** Low, beside the trunk, looking down at the rectangle from under the crown's lowest leaves. */
const VIEW: CameraBookmark = { name: 'the ground rectangle', position: [1.8, 2.2, 4.5], target: [1.8, 0, 0], fovYDeg: 60 };

interface Reading {
    readonly mean: number;
    readonly pixels: number;
}

function readRectangle(images: TestViewImages): Reading {
    let sum = 0;
    let pixels = 0;
    for (let pixel = 0; pixel < images.width * images.height; pixel++) {
        const x = images.position[pixel * 3];
        const y = images.position[pixel * 3 + 1];
        const z = images.position[pixel * 3 + 2];
        if (Math.abs(y) > 0.02 || x < RECTANGLE.minX || x > RECTANGLE.maxX || z < RECTANGLE.minZ || z > RECTANGLE.maxZ) continue;
        sum += (images.indirect[pixel * 3] + images.indirect[pixel * 3 + 1] + images.indirect[pixel * 3 + 2]) / 3;
        pixels++;
    }
    assert(pixels > 500, `the view sees only ${pixels} pixels of the ground rectangle`);
    return { mean: sum / pixels, pixels };
}

/** Pearson correlation of `a[i]` with `b[i + lag]`, over the indices both have. */
function correlationAtLag(a: readonly number[], b: readonly number[], lag: number): number {
    const pairs: [number, number][] = [];
    for (let index = 0; index < a.length; index++) {
        const other = index + lag;
        if (other >= 0 && other < b.length) pairs.push([a[index], b[other]]);
    }
    const meanA = pairs.reduce((sum, [value]) => sum + value, 0) / pairs.length;
    const meanB = pairs.reduce((sum, [, value]) => sum + value, 0) / pairs.length;
    let covariance = 0;
    let varianceA = 0;
    let varianceB = 0;
    for (const [valueA, valueB] of pairs) {
        covariance += (valueA - meanA) * (valueB - meanB);
        varianceA += (valueA - meanA) ** 2;
        varianceB += (valueB - meanB) ** 2;
    }
    return varianceA > 0 && varianceB > 0 ? covariance / Math.sqrt(varianceA * varianceB) : 0;
}

/** Renders three's frame and copies it in the same task (a WebGPU canvas reads blank after an await), labelled. */
function snapshot(renderer: WebGPURenderer, scene: Scene, camera: PerspectiveCamera, label: string, grid: HTMLElement): void {
    renderer.render(scene, camera);
    const figure = document.createElement('figure');
    figure.style.cssText = 'margin: 0; display: inline-block; font: 12px sans-serif; color: #ddd;';
    const copy = document.createElement('canvas');
    copy.width = WIDTH;
    copy.height = HEIGHT;
    copy.getContext('2d')!.drawImage(renderer.domElement, 0, 0, WIDTH, HEIGHT);
    const caption = document.createElement('figcaption');
    caption.textContent = label;
    figure.append(copy, caption);
    grid.append(figure);
}

await runThreePage(
    async ({ device, renderer, data, log }) => {
        document.body.style.background = '#222';
        const grid = document.createElement('div');
        grid.style.cssText = 'display: grid; grid-template-columns: repeat(3, auto); gap: 4px; justify-content: start;';
        document.body.prepend(grid);

        const manifest = loadAllManifests().find((candidate) => candidate.id === 'courtyard-wind-tree');
        assert(manifest, 'no manifest "courtyard-wind-tree"');
        const loaded = await loadScene(manifest, () => {});
        const scene = loaded.scene;
        renderer.lighting = sceneLighting(loaded.directLights, renderer.lighting);
        // The viewer's sky dome is a picture of the sky, not geometry.
        const content = new Set<Object3D>();
        loaded.content.traverse((object) => content.add(object));
        scene.traverse((object) => {
            if ((object as Mesh).isMesh && !content.has(object)) object.userData.giContributes = false;
        });

        // The tree's meshes: every mesh with a wind weight. Each sways, and tells GI it deforms within its sway box.
        const sways: WindSway[] = [];
        loaded.content.traverse((object) => {
            const mesh = object as Mesh;
            const color = mesh.isMesh ? mesh.geometry.getAttribute('color') : undefined;
            if (!color || color.itemSize !== 4) return;
            const sway = new WindSway(mesh, { reach: REACH, direction: DIRECTION });
            mesh.userData.giDeforming = true;
            mesh.userData.giDeformBounds = sway.swayBounds();
            sways.push(sway);
        });
        assert(sways.length >= 3, `found ${sways.length} meshes with a wind weight (expected the trunk, branches and leaves)`);
        data.treeMeshes = sways.map((sway) => ({ name: sway.mesh.name, vertices: sway.mesh.geometry.getAttribute('position').count, largestWeight: sway.largestWeight }));
        const pose = (s: number): void => {
            for (const sway of sways) sway.pose(s);
        };

        const camera = new PerspectiveCamera(manifest.cameras[2].fovYDeg, WIDTH / HEIGHT, 0.05, loaded.diagonal * 2);
        camera.position.set(...manifest.cameras[2].position);
        camera.lookAt(...manifest.cameras[2].target);
        camera.updateMatrixWorld();

        const platform = new GIPlatformImpl(device, { label: 'three wind tree' });
        const ddgi = platform.addTechnique(new DDGITechnique({ volume: { center: manifest.volume.center, size: manifest.volume.size, spacing: manifest.volume.probeSpacing }, warmUpPasses: 0 }));
        platform.setOutput(ddgi);
        await platform.prepare();
        const environment: EnvironmentDescriptor | null = manifest.environment.type === 'procedural' ? platformSkyEnvironment(manifest) : null;
        const warnings: string[] = [];
        const threeGI = new ThreeGI(platform, scene, { budget: BUDGET, environment, warn: (message) => warnings.push(message) });
        const view = await TestView.create(device, platform);
        const recompilesAtStart = platform.stats.recompiles;

        const frame = (): void => {
            threeGI.update(camera, STEP_S);
            renderer.render(scene, camera);
        };
        threeGI.update(camera, STEP_S);
        await platform.rays.whenBuilt();
        const settle = async (): Promise<void> => {
            for (let index = 0; index < SETTLE_FRAMES; index++) {
                frame();
                if (index % 8 === 7) await device.queue.onSubmittedWorkDone();
            }
            await platform.rays.whenBuilt();
            frame();
            await device.queue.onSubmittedWorkDone();
        };
        const readNow = async (): Promise<Reading> => readRectangle(await view.render(ddgi.field!, VIEW, WIDTH, HEIGHT));

        // 1. The extremes.
        pose(1);
        await settle();
        snapshot(renderer, scene, camera, 'sway +1 (held)', grid);
        const plus = await readNow();
        pose(-1);
        await settle();
        snapshot(renderer, scene, camera, 'sway -1 (held)', grid);
        const minus = await readNow();
        const extremeDifference = Math.abs(plus.mean - minus.mean) / ((plus.mean + minus.mean) / 2);
        log(`extremes: rectangle GI ${plus.mean.toFixed(5)} at +1, ${minus.mean.toFixed(5)} at -1 (${plus.pixels} px): ${(extremeDifference * 100).toFixed(2)}% apart`);

        // 2. The sway. The pose curve is signed so that it rises when the rectangle should brighten.
        const sign = plus.mean >= minus.mean ? 1 : -1;
        const framesPerPeriod = Math.round(PERIOD_S / STEP_S);
        const totalFrames = framesPerPeriod * SWAY_PERIODS;
        const poses: number[] = [];
        const means: number[] = [];
        // Start the sine at -1, where GI has settled, so the curve has no start-up jump.
        for (let index = 0; index < totalFrames; index++) {
            const s = -Math.cos((2 * Math.PI * index) / framesPerPeriod);
            pose(s);
            frame();
            if (index % READ_EVERY === 0) {
                await device.queue.onSubmittedWorkDone();
                poses.push(sign * s);
                means.push((await readNow()).mean);
            }
            if (index === Math.round(framesPerPeriod * 1.25)) snapshot(renderer, scene, camera, 'mid-sway (s = 0, moving)', grid);
        }
        let bestLag = 0;
        let bestCorrelation = -Infinity;
        for (let lag = -MAX_LAG_READS; lag <= MAX_LAG_READS; lag++) {
            const correlation = correlationAtLag(poses, means, lag);
            if (correlation > bestCorrelation) {
                bestCorrelation = correlation;
                bestLag = lag;
            }
        }
        const lagSeconds = bestLag * READ_EVERY * STEP_S;
        const meanLow = Math.min(...means);
        const meanHigh = Math.max(...means);
        log(`sway: ${SWAY_PERIODS} periods of ${PERIOD_S} s, ${means.length} reads; GI ${meanLow.toFixed(5)} to ${meanHigh.toFixed(5)}; best correlation ${bestCorrelation.toFixed(3)} at a lag of ${lagSeconds.toFixed(3)} s`);

        const recompiles = platform.stats.recompiles - recompilesAtStart;
        const deforming = platform.stats.counters['platform deforming meshes'];
        data.extremes = { plus, minus, difference: extremeDifference };
        data.sway = { periodS: PERIOD_S, periods: SWAY_PERIODS, reads: means.length, lagSeconds, correlation: bestCorrelation, poses, means };
        data.recompiles = recompiles;
        data.deformingCounters = deforming;
        data.warnings = warnings;
        threeGI.dispose();
        platform.dispose();
        renderer.domElement.style.display = 'none';

        const problems: string[] = [];
        if (!(extremeDifference >= EXTREME_DIFFERENCE)) problems.push(`the extremes differ by ${(extremeDifference * 100).toFixed(2)}% in the rectangle (at least ${EXTREME_DIFFERENCE * 100}%)`);
        if (!(bestCorrelation >= MIN_CORRELATION)) problems.push(`GI follows the sway with a correlation of only ${bestCorrelation.toFixed(3)} (at least ${MIN_CORRELATION})`);
        if (!(lagSeconds >= 0 && lagSeconds <= MAX_LAG_S)) problems.push(`GI lags the sway by ${lagSeconds.toFixed(3)} s (0 to ${MAX_LAG_S} s)`);
        if (recompiles !== 0) problems.push(`${recompiles} recompile(s)`);
        assert(problems.length === 0, problems.join('; '));
    },
    { width: WIDTH, height: HEIGHT },
);
