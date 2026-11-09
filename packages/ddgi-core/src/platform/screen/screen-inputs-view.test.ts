import { describe, expect, it } from 'vitest';
import { instanceDebugColor, instanceDebugHash } from './screen-inputs-view.ts';
import { mergeScreenNeeds, missingScreenChannels, screenNeedsOf } from './screen-needs.ts';

/** PCG hash in BigInt arithmetic, wrapped to 32 bits: an independent reference for the u32 version. */
function referenceHash(value: number): number {
    const mask = 0xffffffffn;
    const state = (BigInt(value) * 747796405n + 2891336453n) & mask;
    const word = (((state >> ((state >> 28n) + 4n)) ^ state) * 277803737n) & mask;
    return Number((word >> 22n) ^ word);
}

describe('the screen inputs view', () => {
    it('hashes instance ids as the kernel does (32-bit wrapping PCG)', () => {
        for (const id of [1, 2, 3, 255, 65_536, 1_000_003, 0xffffffff]) expect(instanceDebugHash(id)).toBe(referenceHash(id));
    });

    it('draws no instance as black and every instance in 0.25..1, different neighbours differently', () => {
        expect(instanceDebugColor(0)).toEqual([0, 0, 0]);
        const colours = new Set<string>();
        for (let id = 1; id <= 64; id++) {
            const colour = instanceDebugColor(id);
            for (const channel of colour) {
                expect(channel).toBeGreaterThanOrEqual(0.25);
                expect(channel).toBeLessThanOrEqual(1);
            }
            colours.add(colour.map((channel) => Math.round(channel * 255)).join(','));
        }
        expect(colours.size).toBe(64);
    });
});

describe('screen needs', () => {
    it('reads true as depth and normals only, and merges channels', () => {
        expect(screenNeedsOf({ queries: [] })).toEqual({ screen: false, velocity: false, albedo: false, instance: false });
        expect(screenNeedsOf({ queries: [], screen: true })).toEqual({ screen: true, velocity: false, albedo: false, instance: false });
        const merged = mergeScreenNeeds([{ queries: [], screen: { albedo: true } }, { queries: [], screen: { velocity: true } }, { queries: [] }]);
        expect(merged).toEqual({ screen: true, velocity: true, albedo: true, instance: false });
    });

    it('lists the channels a screen lacks', () => {
        const texture = {} as GPUTexture;
        const screen = { width: 1, height: 1, depth: texture, normal: texture, normalEncoding: 'unorm' as const, albedo: texture };
        expect(missingScreenChannels({ screen: true, velocity: true, albedo: true, instance: true }, screen)).toEqual(['velocity', 'instance']);
        expect(missingScreenChannels({ screen: true, velocity: false, albedo: true, instance: false }, screen)).toEqual([]);
    });
});
