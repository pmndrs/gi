/**
 * three.js r187's point and spot lights, written out on the CPU as the
 * reference the platform's hit shading must match, and the conversion from
 * three's light description to the platform's (what the three adapter, T1.7,
 * will do).
 *
 * Formulas, from three r187 (node_modules/three/src):
 * - `nodes/lighting/LightUtils.js`, `getDistanceAttenuation`:
 *   falloff = 1 / max(d^decay, 0.01), and when distance > 0 also
 *   times clamp(1 - (d / distance)^4, 0, 1)^2.
 * - `nodes/lighting/SpotLightNode.js`: coneCos = cos(angle),
 *   penumbraCos = cos(angle * (1 - penumbra)), and the cone factor is
 *   smoothstep(coneCos, penumbraCos, cos of the angle between the light's
 *   axis and the direction to the point).
 * - The irradiance a surface gets is color * intensity * falloff * cone *
 *   max(dot(N, L), 0) (`PhysicalLightingModel.direct`: dotNL times the light's colour).
 */

import type { LightDescriptor } from '../../../../packages/core/src/platform/contract/scene.ts';
import { normalize, type Vec3 } from '../hit-shading-shared/harness.ts';

export type ThreeLight =
    | { type: 'point'; color: Vec3; intensity: number; position: Vec3; distance: number; decay: 2 }
    | { type: 'spot'; color: Vec3; intensity: number; position: Vec3; target: Vec3; distance: number; decay: 2; angle: number; penumbra: number };

/** three's description to the platform's: `distance` is `range`, the outer cone is `angle`, the inner `angle * (1 - penumbra)`. */
export function toPlatformLight(light: ThreeLight): LightDescriptor {
    if (light.type === 'point') {
        return { type: 'point', color: light.color, intensity: light.intensity, position: light.position, range: light.distance };
    }
    const axis = normalize([light.target[0] - light.position[0], light.target[1] - light.position[1], light.target[2] - light.position[2]]);
    return {
        type: 'spot',
        color: light.color,
        intensity: light.intensity,
        position: light.position,
        direction: axis,
        range: light.distance,
        innerConeAngle: light.angle * (1 - light.penumbra),
        outerConeAngle: light.angle,
    };
}

function smoothstep(low: number, high: number, x: number): number {
    const t = Math.min(1, Math.max(0, (x - low) / (high - low)));
    return t * t * (3 - 2 * t);
}

/** Irradiance three.js gives a surface at `point` with unit normal `normal` (no shadows). */
export function threeLightIrradiance(light: ThreeLight, point: Vec3, normal: Vec3): Vec3 {
    const toLight: Vec3 = [light.position[0] - point[0], light.position[1] - point[1], light.position[2] - point[2]];
    const distance = Math.hypot(...toLight);
    const direction: Vec3 = [toLight[0] / distance, toLight[1] / distance, toLight[2] / distance];

    let falloff = 1 / Math.max(distance ** light.decay, 0.01);
    if (light.distance > 0) falloff *= Math.min(1, Math.max(0, 1 - (distance / light.distance) ** 4)) ** 2;

    let cone = 1;
    if (light.type === 'spot') {
        const axis = normalize([light.target[0] - light.position[0], light.target[1] - light.position[1], light.target[2] - light.position[2]]);
        const angleCos = -(direction[0] * axis[0] + direction[1] * axis[1] + direction[2] * axis[2]);
        const coneCos = Math.cos(light.angle);
        const penumbraCos = Math.cos(light.angle * (1 - light.penumbra));
        // With no penumbra three's smoothstep has equal edges: 0 below the edge, 1 above (a hard cone).
        cone = penumbraCos === coneCos ? (angleCos > coneCos ? 1 : 0) : smoothstep(coneCos, penumbraCos, angleCos);
    }

    const dotNL = Math.max(0, normal[0] * direction[0] + normal[1] * direction[1] + normal[2] * direction[2]);
    const scale = light.intensity * falloff * cone * dotNL;
    return [light.color[0] * scale, light.color[1] * scale, light.color[2] * scale];
}
