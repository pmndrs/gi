/**
 * Point and spot lights with no range window (range 0): plain inverse-square
 * falloff, as in Cycles. They must survive the GLB export and the three.js
 * conversion without turning into lights that reach nowhere.
 */

import type { PointLight, SpotLight } from 'three';
import { describe, expect, it } from 'vitest';
import { readGlbChunks, writeGlb } from './glb';
import { NO_RANGE_WINDOW } from './manifest';
import { SceneBuilder } from './procedural/geometry';
import type { SceneLightData } from './scene-data';
import { lightToThree } from './three-scene';

const point: SceneLightData = { type: 'point', color: [1, 1, 1], intensity: 10, position: [0, 2, 0], range: NO_RANGE_WINDOW };
const spot: SceneLightData = {
    type: 'spot',
    color: [1, 1, 1],
    intensity: 10,
    position: [0, 2, 0],
    direction: [0, -1, 0],
    range: NO_RANGE_WINDOW,
    innerConeAngle: 0.3,
    outerConeAngle: 0.5,
};

function lightsInGlb(lights: SceneLightData[]): Array<Record<string, unknown>> {
    const builder = new SceneBuilder();
    builder.addBox('floor', builder.addDiffuse('grey', [0.5, 0.5, 0.5]), [-1, -0.1, -1], [1, 0, 1]);
    for (const light of lights) builder.addLight(light);
    const { json } = readGlbChunks(writeGlb(builder.build(), { name: 'lights' }));
    const extension = (json.extensions as { KHR_lights_punctual: { lights: Array<Record<string, unknown>> } }).KHR_lights_punctual;
    return extension.lights;
}

describe('lights with no range window', () => {
    it('leave range out of the GLB, which KHR_lights_punctual reads as no limit (a range there must be above 0)', () => {
        const [pointOut, spotOut, windowed] = lightsInGlb([point, spot, { ...point, range: 12 }]);
        expect(pointOut).not.toHaveProperty('range');
        expect(spotOut).not.toHaveProperty('range');
        expect(windowed.range).toBe(12);
    });

    it("become three.js lights with distance 0 and a shadow camera that reaches past the geometry", () => {
        const fitting = { center: [3, 0, 4] as [number, number, number], radius: 5 };
        const threePoint = lightToThree(point, fitting) as PointLight;
        const threeSpot = lightToThree(spot, fitting) as SpotLight;
        for (const light of [threePoint, threeSpot]) {
            expect(light.distance).toBe(0);
            // From the light at (0, 2, 0) to the centre is sqrt(29) m; plus the radius.
            expect(light.shadow.camera.far).toBeCloseTo(Math.sqrt(29) + 5, 9);
        }
        const windowed = lightToThree({ ...point, range: 12 }, fitting) as PointLight;
        expect(windowed.distance).toBe(12);
        expect(windowed.shadow.camera.far).toBe(12);
    });
});
