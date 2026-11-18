/**
 * T1.5a done-when 1: the furnace room converges to its analytic value.
 *
 * Every surface has albedo 0.5 and emits 0.25, so the radiance everywhere is
 * 0.25 / (1 - 0.5) = 0.5 and the irradiance on any surface is E = π × 0.5:
 * E/π = 0.5 (scenes/src/procedural/furnace-room.ts). The room has no lights
 * and no sky, so every bit of it comes through DDGI's own bounces: emission
 * seen by probe rays, then the field feeding back into itself.
 *
 * DDGI on the manifest's volume with default settings, every probe updated
 * every frame, for FRAMES frames. Then `ddgiSampleIrradiance` (the same
 * function the update kernel uses for bounce light) at a grid of points on
 * every wall, the floor and the ceiling, seen head-on. Passes when the mean
 * of |E/π − 0.5| / 0.5 over all points and channels is within 2%; the worst
 * point is reported. Runs on a default-limits device.
 *
 * For comparing hysteresis rules (T1.12), two optional variables:
 * `VITE_DDGI_HYSTERESIS` (one rule from ddgi-shared/hysteresis-modes.ts, for
 * example `fixed:0.97`) and `VITE_DDGI_FRAMES` (instead of FRAMES).
 *
 * Under a quality preset (T5.1): `VITE_DDGI_PRESET=low|medium|high` runs it
 * with that preset's rays, tile sizes and texture store (ddgi-shared/page-preset.ts).
 */

import { assert, runGpuTest } from '../../client.ts';
import { createDdgiScene, FieldProbe, runFrames, type SurfacePoint, releaseDeviceOnPageHide } from '../ddgi-shared/harness.ts';
import { parseHysteresisMode } from '../ddgi-shared/hysteresis-modes.ts';
import { pagePreset, presetReport } from '../ddgi-shared/page-preset.ts';
import { FURNACE_EXPECTED_RADIANCE, FURNACE_SIZE_M } from '../../../../scenes/src/procedural/furnace-room.ts';

const env = import.meta.env as Record<string, unknown>;

/**
 * Frames every probe is updated: 30 s of DDGI's clock at 60 Hz. Why 1800:
 * the adaptive hysteresis (T1.12, the default) climbs to about 0.995 when
 * every probe is updated every frame, so bounce light settles about 6x more
 * slowly than at the old fixed 0.97, for which 600 was enough. Measured mean
 * bias, RTX / Intel: adaptive 600 frames -1.95% / -2.05% (over the limit on
 * the Intel), 1800 frames -0.24% / -0.36%; fixed 0.97 at 600 -0.16% / -0.28%.
 */
const FRAMES = Number(env.VITE_DDGI_FRAMES) > 0 ? Math.round(Number(env.VITE_DDGI_FRAMES)) : 1800;
/** The hysteresis rule, when a run names one; otherwise the technique's defaults. */
/** The quality preset, when a run names one (T5.1). */
const PRESET = pagePreset();
const HYSTERESIS = typeof env.VITE_DDGI_HYSTERESIS === 'string' && env.VITE_DDGI_HYSTERESIS !== '' ? parseHysteresisMode(env.VITE_DDGI_HYSTERESIS) : null;
/** The check: the mean relative error, over every point and channel. */
const MEAN_TOLERANCE = 0.02;
/** Points per metre on each surface. */
const POINTS_PER_METRE = 4;
/** Instances that are part of the room's shell; everything else is a block that covers the surfaces it touches. */
const SHELL_PREFIX = 'furnace-';

type Vec3 = [number, number, number];

interface Box {
    min: Vec3;
    max: Vec3;
}

/** World-space boxes around the blocks inside the room, from the scene data (conservative for the rotated block). */
function blockBoxes(data: Awaited<ReturnType<typeof createDdgiScene>>['data']): Box[] {
    const boxes: Box[] = [];
    for (const instance of data.instances) {
        if (instance.name.startsWith(SHELL_PREFIX)) continue;
        const positions = data.meshes[instance.mesh].positions;
        const m = instance.transform;
        const box: Box = { min: [Infinity, Infinity, Infinity], max: [-Infinity, -Infinity, -Infinity] };
        for (let vertex = 0; vertex < positions.length; vertex += 3) {
            const [x, y, z] = [positions[vertex], positions[vertex + 1], positions[vertex + 2]];
            const world = [0, 1, 2].map((row) => m[row] * x + m[4 + row] * y + m[8 + row] * z + m[12 + row]);
            for (let axis = 0; axis < 3; axis++) {
                box.min[axis] = Math.min(box.min[axis], world[axis]);
                box.max[axis] = Math.max(box.max[axis], world[axis]);
            }
        }
        boxes.push(box);
    }
    return boxes;
}

/** A grid of points on the six inside faces of the room, leaving out points a block stands on or touches. */
function surfacePoints(blocks: readonly Box[]): SurfacePoint[] {
    const [sizeX, sizeY, sizeZ] = FURNACE_SIZE_M;
    const size: Vec3 = [sizeX, sizeY, sizeZ];
    const points: SurfacePoint[] = [];
    const covered = (point: Vec3): boolean => {
        const margin = 0.05;
        return blocks.some((box) => [0, 1, 2].every((axis) => point[axis] > box.min[axis] - margin && point[axis] < box.max[axis] + margin));
    };
    // Each face: the axis it is perpendicular to and which end of the room.
    for (let axis = 0; axis < 3; axis++) {
        for (const end of [0, 1]) {
            const [a, b] = [0, 1, 2].filter((other) => other !== axis);
            const stepsA = Math.round(size[a] * POINTS_PER_METRE);
            const stepsB = Math.round(size[b] * POINTS_PER_METRE);
            for (let i = 0; i < stepsA; i++) {
                for (let j = 0; j < stepsB; j++) {
                    const position: Vec3 = [0, 0, 0];
                    position[axis] = end * size[axis];
                    position[a] = ((i + 0.5) / stepsA) * size[a];
                    position[b] = ((j + 0.5) / stepsB) * size[b];
                    if (covered(position)) continue;
                    const normal: Vec3 = [0, 0, 0];
                    normal[axis] = end === 0 ? 1 : -1;
                    points.push({ position, normal, viewDirection: [-normal[0], -normal[1], -normal[2]] });
                }
            }
        }
    }
    return points;
}

await runGpuTest(
    async ({ device, log }) => {
        releaseDeviceOnPageHide(device);
        const scene = await createDdgiScene(device, 'furnace-room', { ...PRESET?.ddgi, ...HYSTERESIS?.settings }, { textureLayerSize: PRESET?.textureLayerSize });
        const field = scene.ddgi.field;
        assert(field, 'DDGI has no field after prepare');
        const probeCount = field.grid.probeCount;
        const start = performance.now();
        await runFrames(device, scene.platform, FRAMES, probeCount);
        const runMs = performance.now() - start;

        const points = surfacePoints(blockBoxes(scene.data));
        const probe = await FieldProbe.create(device, scene.platform);
        const irradiance = await probe.sample(field, points);
        const expected = FURNACE_EXPECTED_RADIANCE;

        let errorSum = 0;
        let valueSum = 0;
        let worst = { error: 0, value: 0, position: [0, 0, 0] as readonly number[] };
        const histogram = { under1: 0, under2: 0, under5: 0, over5: 0 };
        irradiance.forEach((rgb, index) => {
            for (const channel of rgb) {
                const value = channel / Math.PI;
                const error = Math.abs(value - expected) / expected;
                errorSum += error;
                valueSum += value;
                if (error > worst.error) worst = { error, value, position: points[index].position };
            }
            const pointError = Math.max(...rgb.map((channel) => Math.abs(channel / Math.PI - expected) / expected));
            if (pointError < 0.01) histogram.under1++;
            else if (pointError < 0.02) histogram.under2++;
            else if (pointError < 0.05) histogram.under5++;
            else histogram.over5++;
        });
        const samples = irradiance.length * 3;
        const meanError = errorSum / samples;
        const meanValue = valueSum / samples;
        log(`${points.length} points after ${FRAMES} frames of ${probeCount} probes: mean E/pi ${meanValue.toFixed(5)}, mean |error| ${(meanError * 100).toFixed(3)}%, worst ${(worst.error * 100).toFixed(2)}% at (${worst.position.map((v) => v.toFixed(2)).join(', ')})`);
        const counters = scene.ddgi.counters;
        scene.platform.dispose();

        assert(meanError <= MEAN_TOLERANCE, `mean |E/pi - 0.5| / 0.5 is ${(meanError * 100).toFixed(3)}%, over the ${MEAN_TOLERANCE * 100}% limit`);
        return {
            ...presetReport(PRESET),
            frames: FRAMES,
            hysteresis: HYSTERESIS?.name ?? 'default',
            probesPerFrame: probeCount,
            probes: field.grid.counts,
            distanceFormat: field.layout.distanceFormat,
            points: points.length,
            meanEOverPi: meanValue,
            meanRelativeError: meanError,
            meanBias: meanValue / expected - 1,
            worstRelativeError: worst.error,
            worstEOverPi: worst.value,
            worstPosition: worst.position,
            pointsByWorstChannelError: histogram,
            counters,
            runMs: Math.round(runMs),
        };
    },
    { optionalFeatures: ['texture-formats-tier1', 'subgroups'] },
);
