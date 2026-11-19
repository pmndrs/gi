/**
 * CPU twins of the neural tail's small rules (P6.4, register E11), for unit
 * tests and the GPU page's expectations: the direction angles the network
 * reads, the position it reads, the training schedule, and the furnace's
 * every-bounce answer.
 */

import type { Aabb, Vec3 } from '../../../platform/contract/math.ts';

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

/** Training steps this frame: the warm-up rate until the gate opens, then the steady rate. */
export function neuralTailStepsThisFrame(trainedSteps: number, settings: { warmupSteps: number; warmupStepsPerFrame: number; stepsPerFrame: number }): number {
    if (trainedSteps < settings.warmupSteps && settings.warmupStepsPerFrame > 0) return settings.warmupStepsPerFrame;
    return settings.stepsPerFrame;
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
