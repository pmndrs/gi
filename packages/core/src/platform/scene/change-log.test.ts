import { describe, expect, it } from 'vitest';
import type { LightDescriptor, PointLightDescriptor } from '../contract/scene.ts';
import type { LightHandle } from '../contract/handles.ts';
import type { Aabb } from '../contract/math.ts';
import type { LightChange, SceneChanges } from '../contract/scene-copy.ts';
import { ChangeLog, influenceRadius, MAX_LISTED_LIGHT_CHANGES, MAX_RECEIVER_BOXES, mergeSceneChanges, referenceIrradiance, sameLight, type LightLookup } from './change-log.ts';

const lamp = (intensity: number, range = 0, color: [number, number, number] = [1, 1, 1]): PointLightDescriptor => ({ type: 'point', color, intensity, position: [0, 0, 0], range });
/** A unit box with its low corner at x = `x`. */
const box = (x: number): Aabb => ({ min: [x, 0, 0], max: [x + 1, 1, 1] });
const NO_LIGHTS: LightLookup = { current: () => undefined, all: () => [] };
const sun = (intensity: number): LightDescriptor => ({ type: 'directional', color: [1, 0.9, 0.8], intensity, direction: [0, -1, 0] });

describe('the influence radius (m2-lights.md §1)', () => {
    it('uses the sun as the reference, or the brightest lamp at 1 m without one', () => {
        expect(referenceIrradiance([lamp(12), lamp(3)])).toBe(12);
        expect(referenceIrradiance([lamp(12), sun(2)])).toBe(2);
        // A black sun is no reference.
        expect(referenceIrradiance([lamp(12), sun(0)])).toBe(12);
        expect(referenceIrradiance([])).toBe(0);
        // The strongest colour channel counts.
        expect(referenceIrradiance([lamp(10, 0, [0.2, 0.5, 0.1])])).toBe(5);
    });

    it('ends where the unwindowed irradiance falls to 1/1000 of the reference, capped by the range', () => {
        // The only lamp: I / d^2 = I / 1000 at d = sqrt(1000).
        expect(influenceRadius(lamp(12), 12)).toBeCloseTo(Math.sqrt(1000), 10);
        // A lamp a quarter as bright reaches half as far.
        expect(influenceRadius(lamp(3), 12)).toBeCloseTo(Math.sqrt(1000) / 2, 10);
        // Two-room's lamp has range 14, under its 31.6 m.
        expect(influenceRadius(lamp(12, 14), 12)).toBe(14);
        expect(influenceRadius(lamp(0), 12)).toBe(0);
        expect(influenceRadius(null, 12)).toBe(0);
        expect(influenceRadius(sun(3), 3)).toBe(Infinity);
    });

    it('compares light states field by field', () => {
        expect(sameLight(lamp(1), lamp(1))).toBe(true);
        expect(sameLight(lamp(1), lamp(2))).toBe(false);
        expect(sameLight(lamp(1), { ...lamp(1), castsShadows: false })).toBe(false);
        expect(sameLight(lamp(1), { ...lamp(1), position: [0, 0, 1e-7] })).toBe(false);
    });
});

describe('merging frames of changes (for frames a technique did not see)', () => {
    const none: SceneChanges = { regions: [], global: [], receivers: [], lights: [], lightsOverflowed: false };
    const change = (light: number, before: LightDescriptor | null, after: LightDescriptor | null): LightChange => ({
        light: light as LightHandle,
        index: 0,
        before,
        after,
        influenceRadius: { before: before ? 1 : 0, after: after ? 2 : 0 },
    });

    it("keeps the first frame's before and the last frame's after, and drops a light that came back", () => {
        const first = { ...none, lights: [change(1, lamp(1), lamp(2)), change(2, lamp(1), lamp(3))], global: ['sun' as const], receivers: [box(0)] };
        const second = { ...none, lights: [change(1, lamp(2), lamp(4)), change(2, lamp(3), lamp(1))], global: ['environment' as const], receivers: [box(1), box(2)] };
        const merged = mergeSceneChanges(first, second);
        expect(merged.lights).toHaveLength(1);
        expect(merged.lights[0].before?.intensity).toBe(1);
        expect(merged.lights[0].after?.intensity).toBe(4);
        expect(merged.global).toEqual(['sun', 'environment']);
        expect(merged.receivers.map((receiver) => receiver.min[0])).toEqual([0, 1, 2]);
        // Created in one frame, destroyed in the next: nothing.
        expect(mergeSceneChanges({ ...none, lights: [change(5, null, lamp(1))] }, { ...none, lights: [change(5, lamp(1), null)] }).lights).toEqual([]);
    });

    it('overflows like a single frame once the merged list is too long', () => {
        const many = (from: number) => ({ ...none, lights: Array.from({ length: MAX_LISTED_LIGHT_CHANGES }, (_, index) => change(from + index, lamp(1), lamp(2))) });
        const merged = mergeSceneChanges(many(0), many(MAX_LISTED_LIGHT_CHANGES));
        expect(merged.lightsOverflowed).toBe(true);
        expect(merged.lights).toEqual([]);
        expect(merged.global).toContain('lights');
    });
});

describe('receiver boxes (SceneChanges.receivers)', () => {
    it('lists each box, then clears', () => {
        const log = new ChangeLog();
        log.addReceiver([0, 0, 0, 1, 1, 1]);
        // From an offset into a larger array, as the scene logs its instance boxes.
        log.addReceiver([9, 9, 2, 3, 4, 5, 6, 7], 2);
        const changes = log.take(NO_LIGHTS);
        expect(changes.receivers).toEqual([
            { min: [0, 0, 0], max: [1, 1, 1] },
            { min: [2, 3, 4], max: [5, 6, 7] },
        ]);
        expect(changes.regions).toEqual([]);
        expect(log.take(NO_LIGHTS).receivers).toEqual([]);
    });

    it('merges into one box past MAX_RECEIVER_BOXES, in a frame and across frames', () => {
        const log = new ChangeLog();
        for (let index = 0; index <= MAX_RECEIVER_BOXES; index++) log.addReceiver([index, 0, 0, index + 1, 1, 1]);
        const merged = log.take(NO_LIGHTS).receivers;
        expect(merged).toEqual([{ min: [0, 0, 0], max: [MAX_RECEIVER_BOXES + 1, 1, 1] }]);
        // The next frame lists one by one again.
        log.addReceiver([0, 0, 0, 1, 1, 1]);
        expect(log.take(NO_LIGHTS).receivers).toHaveLength(1);

        const none: SceneChanges = { regions: [], global: [], receivers: [], lights: [], lightsOverflowed: false };
        const half = (from: number): SceneChanges => ({ ...none, receivers: Array.from({ length: MAX_RECEIVER_BOXES / 2 + 1 }, (_, index) => box(from + index)) });
        const acrossFrames = mergeSceneChanges(half(0), half(1000)).receivers;
        expect(acrossFrames).toEqual([{ min: [0, 0, 0], max: [1000 + MAX_RECEIVER_BOXES / 2 + 1, 1, 1] }]);
    });
});
