/**
 * M4, T4.4: the alpha checker and the glass wall through the GPU test
 * harness, with the materials their scene data gives them
 * (`describeScene` → scenes/src/platform-scene.ts: the checker texture, its uv
 * mapping and cutoff; the wall mesh's two groups and the pane's
 * `blocksLight: false`).
 *
 * Both scenes are a lit room A and a dark room B sharing a wall
 * (scenes/src/procedural/see-through-walls.ts). Room B gets light only
 * through the shared wall, so its field says what the wall lets through. Each
 * variant runs on a platform of its own, DDGI updating every probe every
 * frame, and reads E/π from the field at points on room B's floor and ceiling:
 *
 * alpha-checker
 * - `checker`: as built, the cut-out checker (half the opening open);
 * - `open`: the wall material set to `blocksLight: false`, a fully open wall;
 * - `closed`: the wall material set to `alphaMode: 'opaque'`, a solid wall.
 *   Checks: checker / open is about 0.5 (0.40 to 0.60: light that crosses the
 *   wall more than once is halved each time, so a little under half is right;
 *   NOTES, T4.2's 48.9% on thin-wall), and closed / open under 2% (no leak).
 *
 * glass-wall
 * - `glass`: as built, the pane in slot 1 lets light through;
 * - `pane-opaque`: the pane material set to `blocksLight: true`.
 *   Checks: room B is lit through the glass (above 5% of the checker's open
 *   wall level), and with the pane opaque it gets under 2% of that: none comes
 *   through the wall around it.
 *
 * `VITE_DDGI_FRAMES` replaces FRAMES; `VITE_REPORT_ONLY=1` returns the numbers without checking them.
 */

import { assert, runGpuTest } from '../../client.ts';
import type { MaterialDescriptor } from '../../../../packages/ddgi-core/src/platform/contract/scene.ts';
import { SEE_THROUGH_ROOM_HEIGHT_M, SEE_THROUGH_ROOM_SIZE_M } from '../../../../scenes/src/procedural/see-through-walls.ts';
import { createDdgiScene, FieldProbe, releaseDeviceOnPageHide, runFrames, type SurfacePoint } from '../ddgi-shared/harness.ts';

const env = import.meta.env as Record<string, unknown>;
const FRAMES = Number(env.VITE_DDGI_FRAMES) > 0 ? Math.round(Number(env.VITE_DDGI_FRAMES)) : 900;
/** `VITE_REPORT_ONLY=1`: return every number and skip the checks, to see a failing run's values. */
const REPORT_ONLY = env.VITE_REPORT_ONLY === '1';
const CHECKER_SHARE_RANGE: readonly [number, number] = [0.4, 0.6];
const LEAK_LIMIT = 0.02;
const GLASS_LIT_SHARE = 0.05;
/** Room B's floor and ceiling points stay this far from its crate (centre x 6.2, z 2.4, 0.8 m wide, turned). */
const CRATE_CENTRE: readonly [number, number] = [6.2, 2.4];
const CRATE_CLEARANCE_M = 0.75;
const POINT_STEP_M = 0.4;

type Vec3 = [number, number, number];

/** A grid of points on room B's floor (facing up) and ceiling (facing down), away from the shared wall's plane and the crate. */
function roomBPoints(): SurfacePoint[] {
    const points: SurfacePoint[] = [];
    const size = SEE_THROUGH_ROOM_SIZE_M;
    for (let x = size + POINT_STEP_M; x < 2 * size; x += POINT_STEP_M) {
        for (let z = POINT_STEP_M / 2; z < size; z += POINT_STEP_M) {
            if (Math.hypot(x - CRATE_CENTRE[0], z - CRATE_CENTRE[1]) < CRATE_CLEARANCE_M) continue;
            for (const [y, normalY] of [[0, 1], [SEE_THROUGH_ROOM_HEIGHT_M, -1]] as const) {
                const normal: Vec3 = [0, normalY, 0];
                points.push({ position: [x, y, z], normal, viewDirection: [0, -normalY, 0] });
            }
        }
    }
    return points;
}

/** One variant: the scene, a change to one material (by name) or none, then the room B mean E/π. */
/** Each variant's texture state after its frames (built, waiting, refused), returned with the numbers. */
const textureStates: Record<string, string> = {};

async function runVariant(device: GPUDevice, sceneId: string, label: string, materialName: string | null, change: Partial<MaterialDescriptor>, log: (line: string) => void): Promise<number> {
    const scene = await createDdgiScene(device, sceneId);
    const field = scene.ddgi.field;
    assert(field, 'DDGI has no field after prepare');
    if (materialName !== null) {
        const index = scene.data.materials.findIndex((material) => material.name === materialName);
        assert(index >= 0, `${sceneId} has no material '${materialName}'`);
        scene.platform.scene.updateMaterial(scene.materials[index], change);
    }
    const start = performance.now();
    await runFrames(device, scene.platform, FRAMES, field.grid.probeCount);
    const points = roomBPoints();
    const irradiance = await (await FieldProbe.create(device, scene.platform)).sample(field, points);
    let sum = 0;
    for (const rgb of irradiance) sum += (rgb[0] + rgb[1] + rgb[2]) / (3 * Math.PI);
    const mean = sum / irradiance.length;
    const textures = scene.platform.sceneCopy.textureCounts();
    textureStates[`${sceneId} ${label}`] = `${textures.built} built, ${textures.pending} waiting, ${textures.failed} refused`;
    log(
        `${sceneId} ${label}: room B mean E/pi ${mean.toExponential(4)} over ${points.length} points, ${FRAMES} frames in ${((performance.now() - start) / 1000).toFixed(1)} s` +
            ` (textures built ${textures.built}, waiting ${textures.pending}, refused ${textures.failed})`,
    );
    scene.platform.dispose();
    return mean;
}

await runGpuTest(
    async ({ device, log }) => {
        releaseDeviceOnPageHide(device);
        const wall = 'alpha-checker-wall';
        const checker = await runVariant(device, 'alpha-checker', 'checker', null, {}, log);
        const open = await runVariant(device, 'alpha-checker', 'open', wall, { blocksLight: false }, log);
        const closed = await runVariant(device, 'alpha-checker', 'closed', wall, { alphaMode: 'opaque' }, log);
        const glass = await runVariant(device, 'glass-wall', 'glass', null, {}, log);
        const paneOpaque = await runVariant(device, 'glass-wall', 'pane-opaque', 'glass-pane', { blocksLight: true }, log);

        const checkerShare = checker / open;
        const closedShare = closed / open;
        const glassShare = glass / open;
        const paneOpaqueShare = paneOpaque / glass;
        log(`checker / open ${(checkerShare * 100).toFixed(1)}%, closed / open ${(closedShare * 100).toFixed(3)}%, glass / open checker wall ${(glassShare * 100).toFixed(1)}%, pane opaque / glass ${(paneOpaqueShare * 100).toFixed(3)}%`);
        const result = { frames: FRAMES, textureStates, checker, open, closed, glass, paneOpaque, checkerShare, closedShare, glassShare, paneOpaqueShare };
        if (REPORT_ONLY) return result;
        assert(checkerShare >= CHECKER_SHARE_RANGE[0] && checkerShare <= CHECKER_SHARE_RANGE[1], `the checker lets through ${(checkerShare * 100).toFixed(1)}% of the open wall's light (expected ${CHECKER_SHARE_RANGE[0] * 100}-${CHECKER_SHARE_RANGE[1] * 100}%)`);
        assert(closedShare < LEAK_LIMIT, `the opaque checker wall lets through ${(closedShare * 100).toFixed(2)}% of the open wall's light (limit ${LEAK_LIMIT * 100}%)`);
        assert(glassShare > GLASS_LIT_SHARE, `room B behind the glass gets only ${(glassShare * 100).toFixed(2)}% of the open checker wall's light`);
        assert(paneOpaqueShare < LEAK_LIMIT, `with the pane opaque, room B still gets ${(paneOpaqueShare * 100).toFixed(2)}% of the glass case's light (limit ${LEAK_LIMIT * 100}%)`);
        return result;
    },
    { optionalFeatures: ['texture-formats-tier1', 'subgroups'] },
);
