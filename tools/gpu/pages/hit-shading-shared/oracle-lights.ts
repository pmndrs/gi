/**
 * Where the hit-shading pages put their lights in a trace-oracle scene
 * (T1.4's `hit-shading-shadows`, T2.2's `hit-shading-deferred`): a point
 * light outside the scene, a point light inside its first instance, and a
 * sun direction.
 *
 * Not a page itself (no index.html).
 */

import { transformPoint, type OracleCase } from '../trace-oracle/cases.ts';
import { normalize, type Vec3 } from './harness.ts';

export interface CaseLights {
    readonly outside: Vec3;
    readonly inside: Vec3;
}

export const LIGHT_NAMES = ['point light outside', 'point light inside the first instance', 'sun'] as const;
export const SUN_DIRECTION = normalize([-0.3, -1, -0.45]);

/** Lights placed from the scene's world bounds and its first instance. */
export function lightsFor(oracleCase: OracleCase): CaseLights {
    const min: Vec3 = [Infinity, Infinity, Infinity];
    const max: Vec3 = [-Infinity, -Infinity, -Infinity];
    for (const instance of oracleCase.instances) {
        const positions = oracleCase.meshes[instance.meshIndex].positions;
        for (let vertex = 0; vertex < positions.length; vertex += 3) {
            const point = transformPoint(Array.from(instance.objectToWorld), [positions[vertex], positions[vertex + 1], positions[vertex + 2]]);
            for (let axis = 0; axis < 3; axis++) {
                min[axis] = Math.min(min[axis], point[axis]);
                max[axis] = Math.max(max[axis], point[axis]);
            }
        }
    }
    const centre = min.map((value, axis) => (value + max[axis]) / 2) as Vec3;
    const extent = max.map((value, axis) => value - min[axis]) as Vec3;
    // Above and to one side, well clear of every surface.
    const outside: Vec3 = [centre[0] + 0.3 * extent[0] + 0.7, max[1] + 0.6 * extent[1] + 1.3, centre[2] + 0.2 * extent[2] + 0.9];
    // Near the first instance's own origin, nudged off any plane through it
    // (on thin-wall it sits between the two skins, 0.5 mm from each).
    const origin = transformPoint(Array.from(oracleCase.instances[0].objectToWorld), [0, 0, 0]);
    const inside: Vec3 = [origin[0] + 0.0005, origin[1] + 0.0131, origin[2] + 0.0213];
    return { outside, inside };
}
