/**
 * M4, T4.4 and T4.5: the alpha checker and the glass wall through the GPU test
 * harness, with the materials their scene data gives them
 * (`describeScene` → scenes/src/platform-scene.ts: the checker texture, its uv
 * mapping and cutoff; the wall mesh's two groups and the pane's
 * `blocksLight: false`).
 *
 * Both scenes are a lit room A and a dark room B sharing a wall
 * (scenes/src/procedural/see-through-walls.ts). Room B gets light only
 * through the shared wall. Each variant runs on a platform of its own, DDGI
 * updating every probe every frame, and reads E/π from the field at points
 * on each room's floor and ceiling, 1 m or more from the shared wall:
 *
 * alpha-checker: `checker` (as built, half the opening open), `open` (the
 * wall `blocksLight: false`), `closed` (the wall `alphaMode: 'opaque'`).
 * glass-wall: `glass` (as built), `pane-opaque` (the pane `blocksLight: true`).
 *
 * **Gated:**
 * - the order physics gives: room B gets more light through the open wall
 *   than through the checker, and more through the checker than through the
 *   closed wall; more through the glass than through the opaque pane;
 * - the leak behind the closed walls (the checker wall opaque, the pane
 *   opaque): room B against room A, M1's leak measure (DECISIONS
 *   2026-11-04), at most SPEC's 1% (T4.5).
 *
 * The leak needs the manifests' probe volume centred at x 4.25 (T4.5): at
 * x 4.0 a layer of probes sat inside the shared wall, and room B read 1.56%
 * (checker) and 1.47% (pane) of room A; moved half a spacing, 0.56% and
 * 0.10% (T4.4's measurement).
 *
 * **Reported, not gated:** the checker's and the glass's share of the open
 * wall's light, and the closed walls' share of it.
 *
 * `VITE_DDGI_FRAMES` replaces FRAMES.
 */

import { assert, runGpuTest } from '../../client.ts';
import type { MaterialDescriptor } from '../../../../packages/core/src/platform/contract/scene.ts';
import { SEE_THROUGH_ROOM_HEIGHT_M, SEE_THROUGH_ROOM_SIZE_M } from '../../../../scenes/src/procedural/see-through-walls.ts';
import { createDdgiScene, FieldProbe, loadManifest, releaseDeviceOnPageHide, runFrames, type SurfacePoint } from '../ddgi-shared/harness.ts';

const env = import.meta.env as Record<string, unknown>;
const FRAMES = Number(env.VITE_DDGI_FRAMES) > 0 ? Math.round(Number(env.VITE_DDGI_FRAMES)) : 900;
/** SPEC's leak bar: the dark room's light behind a solid wall, as a share of the lit room's. */
const LEAK_LIMIT = 0.01;
/** Points stay this far from each room's crate (room A: x 1.2, z 3.0; room B: x 6.2, z 2.4). */
const CRATE_CENTRES: readonly (readonly [number, number])[] = [
    [1.2, 3.0],
    [6.2, 2.4],
];
const CRATE_CLEARANCE_M = 0.75;
const POINT_STEP_M = 0.4;
/**
 * Room B's points start this far from the shared wall: two probe spacings, so
 * the probes they read sit in room B. Nearer, the field blends probes on both
 * sides of a thin wall (DDGI's known leak), which is not what the wall lets through.
 */
const WALL_CLEARANCE_M = 1.0;

type Vec3 = [number, number, number];

/**
 * A grid of points on one room's floor (facing up) and ceiling (facing down),
 * at least WALL_CLEARANCE_M from the shared wall and away from the crates.
 * Room A is x 0..4, room B x 4..8.
 */
function roomPoints(room: 'A' | 'B'): SurfacePoint[] {
    const points: SurfacePoint[] = [];
    const size = SEE_THROUGH_ROOM_SIZE_M;
    const [xStart, xEnd] = room === 'B' ? [size + WALL_CLEARANCE_M, 2 * size] : [POINT_STEP_M / 2, size - WALL_CLEARANCE_M + 1e-6];
    for (let x = xStart; x < xEnd; x += POINT_STEP_M) {
        for (let z = POINT_STEP_M / 2; z < size; z += POINT_STEP_M) {
            if (CRATE_CENTRES.some(([cx, cz]) => Math.hypot(x - cx, z - cz) < CRATE_CLEARANCE_M)) continue;
            for (const [y, normalY] of [[0, 1], [SEE_THROUGH_ROOM_HEIGHT_M, -1]] as const) {
                const normal: Vec3 = [0, normalY, 0];
                points.push({ position: [x, y, z], normal, viewDirection: [0, -normalY, 0] });
            }
        }
    }
    return points;
}

interface RoomMeans {
    /** Mean E/π over room A's points. */
    readonly roomA: number;
    /** Mean E/π over room B's points. */
    readonly roomB: number;
}

function meanEOverPi(irradiance: readonly (readonly number[])[]): number {
    let sum = 0;
    for (const rgb of irradiance) sum += (rgb[0] + rgb[1] + rgb[2]) / (3 * Math.PI);
    return sum / irradiance.length;
}

/** One variant: the scene, a change to one material (by name) or none; then both rooms' mean E/π. */
async function runVariant(
    device: GPUDevice,
    sceneId: string,
    label: string,
    materialName: string | null,
    change: Partial<MaterialDescriptor>,
    log: (line: string) => void,
): Promise<RoomMeans> {
    const manifest = loadManifest(sceneId);
    const volume = { center: manifest.volume.center, size: manifest.volume.size, spacing: manifest.volume.probeSpacing };
    const scene = await createDdgiScene(device, sceneId, { volume });
    const field = scene.ddgi.field;
    assert(field, 'DDGI has no field after prepare');
    if (materialName !== null) {
        const index = scene.data.materials.findIndex((material) => material.name === materialName);
        assert(index >= 0, `${sceneId} has no material '${materialName}'`);
        scene.platform.scene.updateMaterial(scene.materials[index], change);
    }
    const start = performance.now();
    await runFrames(device, scene.platform, FRAMES, field.grid.probeCount);
    const probe = await FieldProbe.create(device, scene.platform);
    const roomA = meanEOverPi(await probe.sample(field, roomPoints('A')));
    const roomB = meanEOverPi(await probe.sample(field, roomPoints('B')));
    log(`${sceneId} ${label}: mean E/pi room A ${roomA.toExponential(4)}, room B ${roomB.toExponential(4)}, ${FRAMES} frames in ${((performance.now() - start) / 1000).toFixed(1)} s`);
    scene.platform.dispose();
    return { roomA, roomB };
}

await runGpuTest(
    async ({ device, log }) => {
        releaseDeviceOnPageHide(device);
        const wall = 'alpha-checker-wall';
        const opaque: Partial<MaterialDescriptor> = { alphaMode: 'opaque' };
        const checker = await runVariant(device, 'alpha-checker', 'checker', null, {}, log);
        const open = await runVariant(device, 'alpha-checker', 'open', wall, { blocksLight: false }, log);
        const closed = await runVariant(device, 'alpha-checker', 'closed', wall, opaque, log);
        const glass = await runVariant(device, 'glass-wall', 'glass', null, {}, log);
        const paneOpaque = await runVariant(device, 'glass-wall', 'pane-opaque', 'glass-pane', { blocksLight: true }, log);

        const report = {
            checkerShareOfOpen: checker.roomB / open.roomB,
            glassShareOfOpenChecker: glass.roomB / open.roomB,
            // M1's leak measure: the dark room against the lit room, with the shared wall solid (gated below).
            closedLeakOfRoomA: closed.roomB / closed.roomA,
            paneOpaqueLeakOfRoomA: paneOpaque.roomB / paneOpaque.roomA,
            closedShareOfOpen: closed.roomB / open.roomB,
            paneOpaqueShareOfGlass: paneOpaque.roomB / glass.roomB,
        };
        const percent = (value: number) => `${(value * 100).toFixed(2)}%`;
        log(
            `checker ${percent(report.checkerShareOfOpen)} of open; glass ${percent(report.glassShareOfOpenChecker)} of the open checker wall; ` +
                `leak (room B / room A, limit ${percent(LEAK_LIMIT)}): closed checker ${percent(report.closedLeakOfRoomA)}, opaque pane ${percent(report.paneOpaqueLeakOfRoomA)}`,
        );

        // Gated: the order physics gives.
        assert(open.roomB > checker.roomB && checker.roomB > closed.roomB, `room B should get open > checker > closed; got ${open.roomB.toExponential(3)}, ${checker.roomB.toExponential(3)}, ${closed.roomB.toExponential(3)}`);
        assert(glass.roomB > paneOpaque.roomB, `room B should get more light through the glass (${glass.roomB.toExponential(3)}) than through the opaque pane (${paneOpaque.roomB.toExponential(3)})`);
        assert(report.closedLeakOfRoomA <= LEAK_LIMIT, `leak behind the closed checker wall: room B is ${percent(report.closedLeakOfRoomA)} of room A, over ${percent(LEAK_LIMIT)}`);
        assert(report.paneOpaqueLeakOfRoomA <= LEAK_LIMIT, `leak behind the opaque pane: room B is ${percent(report.paneOpaqueLeakOfRoomA)} of room A, over ${percent(LEAK_LIMIT)}`);
        return { frames: FRAMES, checker, open, closed, glass, paneOpaque, report };
    },
    { optionalFeatures: ['texture-formats-tier1', 'subgroups'] },
);
