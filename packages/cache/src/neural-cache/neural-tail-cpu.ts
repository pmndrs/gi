/**
 * CPU twins of the neural tail's small rules (P6.4, register E11), for unit
 * tests and the GPU page's expectations: the direction angles the network
 * reads, the position it reads, the training schedule, and the furnace's
 * every-bounce answer.
 */

import type { Aabb, Vec3 } from '@pmndrs/gi-core/platform/contract/math';

/**
 * A unit direction as the network's two angles, each in [0, 1]: θ / π from
 * +y (0 straight up, 1 straight down) and φ / 2π around y, from −x
 * (atan2(z, x) / 2π + 1/2). The WGSL twin is `ntAngles`.
 */
export function neuralTailAngles(direction: Vec3): [number, number] {
    const [x, y, z] = direction;
    const theta = Math.acos(Math.min(Math.max(y, -1), 1)) / Math.PI;
    const phi = Math.atan2(z, x) / (2 * Math.PI) + 0.5;
    return [theta, phi];
}

/** The scene box as the network's position input reads it: `lo` and 2 / extent (an empty axis gets scale 1). */
export function neuralTailBox(bounds: Aabb): { lo: Vec3; scale: Vec3 } {
    const scale = (axis: 0 | 1 | 2): number => {
        const extent = bounds.max[axis] - bounds.min[axis];
        return extent > 0 && Number.isFinite(extent) ? 2 / extent : 1;
    };
    return { lo: [bounds.min[0], bounds.min[1], bounds.min[2]], scale: [scale(0), scale(1), scale(2)] };
}

/** A world position in [−1, 1]³ (clamped), as `ntPosition` reads it. */
export function neuralTailPosition(world: Vec3, box: { lo: Vec3; scale: Vec3 }): Vec3 {
    const axis = (index: 0 | 1 | 2): number => Math.min(Math.max((world[index] - box.lo[index]) * box.scale[index] - 1, -1), 1);
    return [axis(0), axis(1), axis(2)];
}

/** Training steps this frame: the warm-up rate until the gate opens and while `boosting` (after a change with no bounds), then the steady rate. */
export function neuralTailStepsThisFrame(trainedSteps: number, settings: { warmupSteps: number; warmupStepsPerFrame: number; stepsPerFrame: number }, boosting = false): number {
    if ((trainedSteps < settings.warmupSteps || boosting) && settings.warmupStepsPerFrame > 0) return Math.max(settings.warmupStepsPerFrame, settings.stepsPerFrame);
    return settings.stepsPerFrame;
}

/** The default box is centred again when the camera leaves its middle half on any axis. */
export function neuralTailNeedsRecentre(camera: Vec3 | readonly number[], centre: readonly number[], halfSize: number): boolean {
    return [0, 1, 2].some((axis) => Math.abs((camera[axis] ?? 0) - (centre[axis] ?? 0)) > halfSize / 2);
}

/**
 * The training set's records: room for the cell and screen records it takes
 * a frame (`cellRecords`, `screenRecords`, both random shares, so about
 * those counts at any screen size) with a quarter to spare; a record past it
 * is dropped.
 */
export function neuralTailSetCapacity(settings: { cellRecords: number; screenRecords: number; screenStride: number }): number {
    const screen = settings.screenStride > 0 ? settings.screenRecords : 0;
    return Math.max(1024, Math.ceil(((settings.cellRecords + screen) * 1.25) / 64) * 64);
}

/**
 * The queries a screen of `maxRayCount` probe rays can need: one per
 * second-bounce entry, (1 − p) of the first-bounce hits in expectation (p the
 * grid's discard probability), with half again to spare, at most one per ray.
 * An entry past it gets no answer that frame (counted out by the query kernel).
 */
export function neuralTailQueryCapacity(maxRayCount: number, discardProbability: number): number {
    const kept = Math.min(Math.max(1 - discardProbability, 0), 1);
    return Math.max(64, Math.min(maxRayCount, Math.ceil(maxRayCount * kept * 1.5 - 1e-6)));
}

/** The gate: the tail is read once the network has taken `warmupSteps` steps. */
export function neuralTailGateOpen(trainedSteps: number, warmupSteps: number): boolean {
    return trainedSteps >= warmupSteps;
}

/**
 * The furnace's ceiling (a floor facing up and a ceiling facing down, albedo
 * ρ, a sun of irradiance E on the floor only, `fill` the share of a bounce
 * ray from the ceiling that finds the floor): the light leaving the ceiling.
 *
 * - `second-bounce`: GI-1.2's estimate, ρ × fill × ρE/π (0.3183 at ρ 0.5,
 *   E 4, fill 1).
 * - `every-bounce`: L_c = ρ × fill × (ρE/π + ρ L_c), so L_c = ρ fill (ρE/π) /
 *   (1 − ρ² fill) (0.4244 at ρ 0.5, E 4, fill 1; 0.1819 at fill 1/2). The
 *   floor (or a strip) sees the whole ceiling, so its own indirect light is
 *   ρ L_c.
 */
export function furnaceCeiling(albedo: number, irradiance: number, mode: 'second-bounce' | 'every-bounce', fill = 1): number {
    const floor = (albedo * irradiance) / Math.PI;
    if (mode === 'second-bounce') return albedo * fill * floor;
    return (albedo * fill * floor) / (1 - albedo * albedo * fill);
}

/**
 * A full floor under a ceiling of strips (share `fill` of it, facing down),
 * albedo ρ, under a constant sky of radiance L and no sun (P6.4 with RC3.4's
 * sky light): the light leaving the strips. A ray up from the floor finds a
 * strip with probability `fill` at any angle (thin strips, a fine period) and
 * the sky otherwise; a strip sees only the floor. So L_f = ρ ((1 − fill) L +
 * fill L_c) and L_c = ρ L_f.
 *
 * - `second-bounce`: GI-1.2's estimate with the sky among the cache's lights,
 *   ρ × the floor's direct light ρ (1 − fill) L (0.125 at ρ 0.5, L 1, fill 1/2).
 * - `every-bounce`: ρ × ρ (1 − fill) L / (1 − ρ² fill) (0.1429 there).
 */
export function skyStripsCeiling(albedo: number, sky: number, mode: 'second-bounce' | 'every-bounce', fill = 0.5): number {
    const floorDirect = albedo * (1 - fill) * sky;
    if (mode === 'second-bounce') return albedo * floorDirect;
    return (albedo * floorDirect) / (1 - albedo * albedo * fill);
}
