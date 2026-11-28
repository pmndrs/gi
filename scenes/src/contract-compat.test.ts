/**
 * Compile-time check that procedural scene data can be handed to the
 * platform's scene contract as it is. There is no runtime import: a type-only
 * import is erased, so this does not couple `scenes` to `gi-core` at run time.
 *
 * If the contract's light or material shapes change, `pnpm typecheck` fails
 * here and the scene data shapes should follow.
 */

import { describe, expect, it } from 'vitest';
import type { LightDescriptor, MaterialDescriptor } from '../../packages/core/src/platform/contract/scene';
import type { SceneLightData, SceneMaterialData } from './scene-data';

const lightForPlatform = (light: SceneLightData): LightDescriptor => light;
const materialForPlatform = (material: SceneMaterialData): MaterialDescriptor => material;

describe('scene data matches the platform contract', () => {
    it('hands lights and materials over unchanged', () => {
        const light: SceneLightData = { type: 'point', color: [1, 1, 1], intensity: 12, position: [0, 2, 0], range: 10 };
        expect(lightForPlatform(light)).toBe(light);
        const material: SceneMaterialData = { name: 'wall', albedo: [0.7, 0.7, 0.7], emissive: [0, 0, 0], doubleSided: true };
        expect(materialForPlatform(material).albedo).toEqual([0.7, 0.7, 0.7]);
    });
});
