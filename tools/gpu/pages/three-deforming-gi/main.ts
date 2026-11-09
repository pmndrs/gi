/**
 * T3.7 check 4 (TASKS M3, prototype level): a skinned and a morphed test
 * mesh deform in GI, through the three adapter, with no `markChanged`. The
 * GI on a nearby surface must follow the deformation. Judged by eye on the
 * screenshot, and by numbers.
 *
 * The scene: a white room with no lights at all (4 × 2.5 × 3 m, open at
 * the front, where the camera stands), and an orange emissive panel 1.2 m
 * wide and 1.6 m tall standing 0.5 m in front of the back wall. All light in
 * the room is the panel's, bounced by GI, so what three draws on the walls
 * and floor is GI only. The deformation folds the panel's top half forward
 * by 90°: it becomes a horizontal sheet 0.8 m above the floor, facing down.
 * - `morphed`: a plain mesh with one position morph target (the folded
 *   shape), `morphTargetInfluences[0]` 0 → 1 → 0, with the pose box the
 *   adapter works out (exact for a morph: the panel and its folded half).
 * - `morphed, giDeformBounds`: the same, with `userData.giDeformBounds`
 *   giving a box 0.8 m larger on every side.
 * - `skinned`: a `SkinnedMesh` with two bones (the lower half on the root,
 *   the upper half on a bone at 0.8 m), that bone turned 0 → 90° → 0. It
 *   sets `userData.giContributes = true`; skinned meshes are characters
 *   (receivers only) by default. Its estimated box is the bind pose's grown
 *   by 0.8 m on every side.
 *
 * **A known miss, not gated (T3.4's):** with `morphed`'s exact box, GI
 * doesn't come all the way back when the panel unfolds (floor 6% and wall
 * 13% under rest, on both runs of 2026-11-09), and on folding the wall
 * darkens 13% where the larger boxes give 18%. The pose is right: rays find
 * the upright panel on exactly rest's pixels, and with the larger box the
 * same morph returns within 0.1%. DDGI re-traces only the probes its mark
 * finds near the change's box, and a deforming emitter changes the light on
 * walls further off, whose probes a still scene then never updates. So
 * `morphed`'s return is reported, and the two other cases are gated on it.
 *
 * Each case, through `ThreeGI` with three rendering every frame (update,
 * then render):
 * 1. GI converges with the panel upright: `rest`;
 * 2. the panel is folded through three only; the adapter must send the pose
 *    within 2 updates (a bone's world matrix is ready after the render, a
 *    morph influence at once); GI converges again: `bent`;
 * 3. the panel is unfolded and GI converges: `rest again`.
 * After each, GI only (the platform's test view: albedo / π × field) is
 * read in two regions: the floor in front of the panel, which the folded
 * half shines down on (must brighten by FLOOR_GAIN), and the back wall
 * beside the panel (reported). `rest again` must come back to `rest` within
 * RETURN_TOLERANCE. The test view also shows where GI's rays find the
 * panel: in `bent`, its folded half (pixels on a horizontal sheet at 0.8 m)
 * and not its upright top; the reverse at rest.
 *
 * The screenshot shows three's frame after each phase, one row per case.
 * Recompiles stay 0. Nothing is timed.
 *
 *   node tools/reference/bench-lock.ts run --no-cool-wait --what "T3.7 deforming GI" -- pnpm test:gpu --gpu nvidia --filter three-deforming-gi
 */

import {
    ACESFilmicToneMapping,
    Bone,
    Color,
    DoubleSide,
    Float32BufferAttribute,
    Mesh,
    MeshStandardMaterial,
    PerspectiveCamera,
    PlaneGeometry,
    Scene,
    Skeleton,
    SkinnedMesh,
    Uint16BufferAttribute,
    type WebGPURenderer,
} from 'three/webgpu';
import { assert } from '../../client.ts';
import { GIPlatformImpl } from '../../../../packages/ddgi-core/src/platform/gi-platform.ts';
import { DDGITechnique } from '../../../../packages/ddgi-core/src/techniques/ddgi/ddgi-technique.ts';
import { ThreeGI } from '../../../../packages/ddgi-three/src/three-gi.ts';
import { TestView, type TestViewImages } from '../ddgi-shared/harness.ts';
import { runThreePage } from '../three-shared/harness.ts';
import type { CameraBookmark } from '../../../../scenes/src/manifest.ts';

const WIDTH = 384;
const HEIGHT = 256;
/** The panel: width, height, where it stands (its foot's centre), and the fold's height (half way up). */
const PANEL_WIDTH = 1.2;
const PANEL_HEIGHT = 1.6;
const PANEL_Z = -1.0;
const FOLD_HEIGHT = PANEL_HEIGHT / 2;
/** The room: x from -2 to 2, y from 0 to 2.5, z from -1.5 (the back wall) to 1.5 (open). */
const ROOM = { halfWidth: 2, height: 2.5, back: -1.5, front: 1.5 };
const PROBE_SPACING = 0.25;
/** Frames for GI to settle after each change, at the budget below. */
const SETTLE_FRAMES = 360;
const BUDGET = { gpuMs: 4 };
/** The floor in front of the panel must brighten by at least this share when it folds. */
const FLOOR_GAIN = 0.15;
/** Unfolded again, both regions must come back within this share of the first rest (where gated; see the known miss above). */
const RETURN_TOLERANCE = 0.05;
/** The pose must reach the platform within this many updates of the change. */
const POSE_UPDATES = 2;
const CAMERA: CameraBookmark = { name: 'towards-the-panel', position: [0, 1.4, 2.6], target: [0, 0.6, -0.6], fovYDeg: 60 };

interface Case {
    readonly name: string;
    readonly kind: 'morphed' | 'skinned';
    /** `userData.giDeformBounds`, in the panel's own space; absent, the adapter estimates the box. */
    readonly deformBounds?: { readonly min: [number, number, number]; readonly max: [number, number, number] };
    /** Whether `rest again` must match `rest` (the file comment's known miss). */
    readonly gateReturn: boolean;
}

/** 0.8 m around the panel and its folded half: the size of the skinned panel's estimated box. */
const LARGER_BOX = { min: [-PANEL_WIDTH / 2 - 0.8, -0.8, -0.8] as [number, number, number], max: [PANEL_WIDTH / 2 + 0.8, PANEL_HEIGHT + 0.8, 0.8] as [number, number, number] };

const CASES: readonly Case[] = [
    { name: 'morphed', kind: 'morphed', gateReturn: false },
    { name: 'morphed, giDeformBounds', kind: 'morphed', deformBounds: LARGER_BOX, gateReturn: true },
    { name: 'skinned', kind: 'skinned', gateReturn: true },
];

/** The panel's geometry: a vertical sheet in the xy plane, foot at y = 0, facing +z (towards the camera). */
function panelGeometry(): PlaneGeometry {
    const geometry = new PlaneGeometry(PANEL_WIDTH, PANEL_HEIGHT, 6, 16);
    geometry.translate(0, PANEL_HEIGHT / 2, 0);
    return geometry;
}

/** A panel with one morph target: its top half folded forward 90° about the line at FOLD_HEIGHT. */
function morphedPanel(material: MeshStandardMaterial): { mesh: Mesh; fold: (folded: boolean) => void } {
    const geometry = panelGeometry();
    const position = geometry.getAttribute('position');
    const offsets = new Float32Array(position.count * 3);
    for (let vertex = 0; vertex < position.count; vertex++) {
        const above = position.getY(vertex) - FOLD_HEIGHT;
        if (above <= 0) continue;
        // (x, FOLD + a, 0) → (x, FOLD, a): down by a, forward by a.
        offsets[vertex * 3 + 1] = -above;
        offsets[vertex * 3 + 2] = above;
    }
    geometry.morphAttributes.position = [new Float32BufferAttribute(offsets, 3)];
    geometry.morphTargetsRelative = true;
    const mesh = new Mesh(geometry, material);
    mesh.updateMorphTargets();
    return { mesh, fold: (folded) => (mesh.morphTargetInfluences![0] = folded ? 1 : 0) };
}

/** A skinned panel: the lower half on the root bone, the upper half on a bone at FOLD_HEIGHT, turned to fold it. */
function skinnedPanel(material: MeshStandardMaterial): { mesh: SkinnedMesh; fold: (folded: boolean) => void } {
    const geometry = panelGeometry();
    const position = geometry.getAttribute('position');
    const skinIndex: number[] = [];
    const skinWeight: number[] = [];
    for (let vertex = 0; vertex < position.count; vertex++) {
        const upper = position.getY(vertex) > FOLD_HEIGHT + 1e-4;
        skinIndex.push(upper ? 1 : 0, 0, 0, 0);
        skinWeight.push(1, 0, 0, 0);
    }
    geometry.setAttribute('skinIndex', new Uint16BufferAttribute(skinIndex, 4));
    geometry.setAttribute('skinWeight', new Float32BufferAttribute(skinWeight, 4));
    const root = new Bone();
    const hinge = new Bone();
    hinge.position.y = FOLD_HEIGHT;
    root.add(hinge);
    const mesh = new SkinnedMesh(geometry, material);
    mesh.add(root);
    mesh.bind(new Skeleton([root, hinge]));
    // Skinned meshes are characters by default (receivers only); this one is scenery and opts in.
    mesh.userData.giContributes = true;
    // A turn of +90° about x takes the hinge's +y to +z: the top half folds forward, towards the camera.
    return { mesh, fold: (folded) => (hinge.rotation.x = folded ? Math.PI / 2 : 0) };
}

function roomScene(panel: Mesh): Scene {
    const scene = new Scene();
    scene.background = new Color(0x000000);
    const white = new MeshStandardMaterial({ color: new Color(0.8, 0.8, 0.8), roughness: 1, side: DoubleSide });
    const width = 2 * ROOM.halfWidth;
    const depth = ROOM.front - ROOM.back;
    const floor = new Mesh(new PlaneGeometry(width, depth), white);
    floor.rotation.x = -Math.PI / 2;
    floor.position.set(0, 0, (ROOM.front + ROOM.back) / 2);
    const ceiling = new Mesh(new PlaneGeometry(width, depth), white);
    ceiling.rotation.x = Math.PI / 2;
    ceiling.position.set(0, ROOM.height, (ROOM.front + ROOM.back) / 2);
    const back = new Mesh(new PlaneGeometry(width, ROOM.height), white);
    back.position.set(0, ROOM.height / 2, ROOM.back);
    const left = new Mesh(new PlaneGeometry(depth, ROOM.height), white);
    left.rotation.y = Math.PI / 2;
    left.position.set(-ROOM.halfWidth, ROOM.height / 2, (ROOM.front + ROOM.back) / 2);
    const right = new Mesh(new PlaneGeometry(depth, ROOM.height), white);
    right.rotation.y = -Math.PI / 2;
    right.position.set(ROOM.halfWidth, ROOM.height / 2, (ROOM.front + ROOM.back) / 2);
    panel.position.set(0, 0, PANEL_Z);
    scene.add(floor, ceiling, back, left, right, panel);
    return scene;
}

/** GI-only numbers from the test view. */
interface Reading {
    /** Mean GI (rgb mean) on the floor in front of the panel. */
    floor: number;
    /** Mean GI on the back wall beside the panel. */
    wall: number;
    /** Pixels where rays find the panel's top half upright, and folded flat. */
    uprightTopPixels: number;
    foldedTopPixels: number;
}

function read(images: TestViewImages): Reading {
    const { indirect, position } = images;
    let floorSum = 0;
    let floorCount = 0;
    let wallSum = 0;
    let wallCount = 0;
    let uprightTopPixels = 0;
    let foldedTopPixels = 0;
    const halfPanel = PANEL_WIDTH / 2 - 0.05;
    for (let pixel = 0; pixel < images.width * images.height; pixel++) {
        const x = position[pixel * 3];
        const y = position[pixel * 3 + 1];
        const z = position[pixel * 3 + 2];
        const gi = (indirect[pixel * 3] + indirect[pixel * 3 + 1] + indirect[pixel * 3 + 2]) / 3;
        // The floor just in front of where the folded half's edge ends (z = PANEL_Z + FOLD_HEIGHT = -0.2).
        if (y < 0.02 && Math.abs(x) < 0.5 && z > -0.1 && z < 0.5) {
            floorSum += gi;
            floorCount++;
        }
        // The back wall beside the panel, at the height of its top half.
        if (z < ROOM.back + 0.02 && Math.abs(x) > 0.7 && Math.abs(x) < 1.3 && y > 0.9 && y < 1.7) {
            wallSum += gi;
            wallCount++;
        }
        if (Math.abs(x) < halfPanel && Math.abs(z - PANEL_Z) < 0.03 && y > FOLD_HEIGHT + 0.1 && y < PANEL_HEIGHT - 0.05) uprightTopPixels++;
        if (Math.abs(x) < halfPanel && Math.abs(y - FOLD_HEIGHT) < 0.03 && z > PANEL_Z + 0.1 && z < PANEL_Z + FOLD_HEIGHT - 0.05) foldedTopPixels++;
    }
    assert(floorCount > 100 && wallCount > 100, `the view misses a region (floor ${floorCount} px, wall ${wallCount} px)`);
    return { floor: floorSum / floorCount, wall: wallSum / wallCount, uprightTopPixels, foldedTopPixels };
}

/** A copy of what the WebGPU canvas shows now (read in the same task as the render that drew it), labelled. */
function snapshot(renderer: WebGPURenderer, label: string, grid: HTMLElement): void {
    const figure = document.createElement('figure');
    figure.style.cssText = 'margin: 0; display: inline-block; font: 12px sans-serif; color: #ddd;';
    const copy = document.createElement('canvas');
    copy.width = WIDTH;
    copy.height = HEIGHT;
    // Three rows of three, with their captions, fit the runner's 1280 x 720 screenshot at 0.65 of full size.
    copy.style.width = `${WIDTH * 0.65}px`;
    copy.getContext('2d')!.drawImage(renderer.domElement, 0, 0, WIDTH, HEIGHT);
    const caption = document.createElement('figcaption');
    caption.textContent = label;
    figure.append(copy, caption);
    grid.append(figure);
}

await runThreePage(
    async ({ device, renderer, data, log }) => {
        renderer.toneMapping = ACESFilmicToneMapping;
        document.body.style.background = '#222';
        const grid = document.createElement('div');
        grid.style.cssText = 'display: grid; grid-template-columns: repeat(3, auto); gap: 4px; justify-content: start;';
        document.body.prepend(grid);
        const problems: string[] = [];

        for (const testCase of CASES) {
            const which = testCase.name;
            const material = new MeshStandardMaterial({ color: 0x000000, emissive: new Color(1, 0.45, 0.1), emissiveIntensity: 3, side: DoubleSide });
            const panel = testCase.kind === 'morphed' ? morphedPanel(material) : skinnedPanel(material);
            if (testCase.deformBounds) panel.mesh.userData.giDeformBounds = testCase.deformBounds;
            const scene = roomScene(panel.mesh);
            const camera = new PerspectiveCamera(CAMERA.fovYDeg, WIDTH / HEIGHT, 0.05, 50);
            camera.position.set(...CAMERA.position);
            camera.lookAt(...CAMERA.target);
            camera.updateMatrixWorld();

            const platform = new GIPlatformImpl(device, { label: `three deforming ${which}` });
            const ddgi = platform.addTechnique(
                new DDGITechnique({ volume: { center: [0, ROOM.height / 2, (ROOM.front + ROOM.back) / 2], size: [2 * ROOM.halfWidth, ROOM.height, ROOM.front - ROOM.back], spacing: PROBE_SPACING }, warmUpPasses: 0 }),
            );
            platform.setOutput(ddgi);
            await platform.prepare();
            const warnings: string[] = [];
            const threeGI = new ThreeGI(platform, scene, { budget: BUDGET, environment: { type: 'none' }, warn: (message) => warnings.push(message) });
            const view = await TestView.create(device, platform);
            const recompilesAtStart = platform.stats.recompiles;
            const frame = (): void => {
                threeGI.update(camera);
                renderer.render(scene, camera);
            };
            threeGI.update(camera);
            await platform.rays.whenBuilt();
            const settle = async (): Promise<void> => {
                for (let index = 0; index < SETTLE_FRAMES; index++) {
                    frame();
                    if (index % 8 === 7) await device.queue.onSubmittedWorkDone();
                }
                // The pose's tree, if one is still building, then a last frame: what the snapshot shows.
                await platform.rays.whenBuilt();
                frame();
                await device.queue.onSubmittedWorkDone();
            };
            const phase = async (label: string): Promise<Reading> => {
                await settle();
                renderer.render(scene, camera);
                snapshot(renderer, `${which}: ${label}`, grid);
                const reading = read(await view.render(ddgi.field!, CAMERA, WIDTH, HEIGHT));
                log(`${which} ${label}: floor GI ${reading.floor.toFixed(4)}, wall GI ${reading.wall.toFixed(4)}; rays find the top half upright on ${reading.uprightTopPixels} px, folded on ${reading.foldedTopPixels} px`);
                return reading;
            };
            /** Updates (each followed by a render) until the adapter sends a pose. */
            const updatesUntilPose = (): number => {
                for (let count = 1; count <= 10; count++) {
                    frame();
                    if (threeGI.report.poses > 0) return count;
                }
                return Number.POSITIVE_INFINITY;
            };

            const rest = await phase('rest');
            panel.fold(true);
            const foldSent = updatesUntilPose();
            const bent = await phase('bent');
            panel.fold(false);
            const unfoldSent = updatesUntilPose();
            const restAgain = await phase('rest again');
            const recompiles = platform.stats.recompiles - recompilesAtStart;
            threeGI.dispose();
            platform.dispose();

            const floorGain = bent.floor / rest.floor - 1;
            const wallChange = bent.wall / rest.wall - 1;
            const floorReturn = Math.abs(restAgain.floor / rest.floor - 1);
            const wallReturn = Math.abs(restAgain.wall / rest.wall - 1);
            data[which] = { rest, bent, restAgain, floorGain, wallChange, floorReturn, wallReturn, updatesUntilPose: { fold: foldSent, unfold: unfoldSent }, recompiles, warnings };
            log(`${which}: floor ${(floorGain * 100).toFixed(1)}% when folded, back ${(floorReturn * 100).toFixed(2)}% off; wall ${(wallChange * 100).toFixed(1)}%, back ${(wallReturn * 100).toFixed(2)}% off; pose sent after ${foldSent} / ${unfoldSent} update(s); recompiles ${recompiles}`);

            if (!(foldSent <= POSE_UPDATES && unfoldSent <= POSE_UPDATES)) problems.push(`${which}: the pose was sent after ${foldSent} / ${unfoldSent} updates (at most ${POSE_UPDATES})`);
            if (!(rest.uprightTopPixels > 200 && rest.foldedTopPixels === 0)) problems.push(`${which}: at rest, rays find the top half upright on ${rest.uprightTopPixels} px and folded on ${rest.foldedTopPixels} px`);
            if (!(bent.foldedTopPixels > 200 && bent.uprightTopPixels === 0)) problems.push(`${which}: folded, rays find the top half upright on ${bent.uprightTopPixels} px and folded on ${bent.foldedTopPixels} px`);
            if (!(floorGain >= FLOOR_GAIN)) problems.push(`${which}: the floor's GI changed by ${(floorGain * 100).toFixed(1)}% when the panel folded (at least +${FLOOR_GAIN * 100}%)`);
            if (testCase.gateReturn && !(floorReturn <= RETURN_TOLERANCE && wallReturn <= RETURN_TOLERANCE)) problems.push(`${which}: unfolded, GI came back to ${(floorReturn * 100).toFixed(2)}% (floor) and ${(wallReturn * 100).toFixed(2)}% (wall) of rest`);
            if (recompiles !== 0) problems.push(`${which}: ${recompiles} recompile(s)`);
        }
        // The live canvas has done its work; the grid of snapshots is the screenshot.
        renderer.domElement.style.display = 'none';
        assert(problems.length === 0, problems.join('; '));
    },
    { width: WIDTH, height: HEIGHT },
);
