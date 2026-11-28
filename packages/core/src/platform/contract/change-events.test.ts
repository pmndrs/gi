import { describe, expect, it } from 'vitest';
import { ChangeEventTracker } from './change-events';
import type { InstanceHandle, LightHandle } from './handles';
import type { Aabb } from './math';
import type { DirtyRegion } from './scene-copy';

const box = (x: number, size = 1): Aabb => ({ min: [x, 0, 0], max: [x + size, size, size] });
const region = (bounds: Aabb, source?: number, reason: DirtyRegion['reason'] = 'geometry'): DirtyRegion =>
    ({ bounds, reason, source: source as InstanceHandle | LightHandle | undefined }) as DirtyRegion;
const none = { regions: [], global: [] } as const;

describe('change events', () => {
    it('resets a one-frame change once: its epoch is its frame, then stays', () => {
        const tracker = new ChangeEventTracker();
        tracker.update(10, { regions: [region(box(0), 1, 'light')], global: [] });
        expect(tracker.events(10).map((event) => event.epoch)).toEqual([10]);
        tracker.update(11, none);
        expect(tracker.events(11).map((event) => event.epoch)).toEqual([10]);
        expect(tracker.armedEvents(11)).toEqual([]);
        tracker.update(27, none);
        expect(tracker.events(27)).toEqual([]);
    });

    it('follows a moving object as one event with a swept box, re-armed every 8 frames and once when it stops', () => {
        const tracker = new ChangeEventTracker();
        const epochs: number[] = [];
        for (let frame = 1; frame <= 20; frame++) {
            tracker.update(frame, { regions: [region(box(frame * 0.1), 7)], global: [] });
            epochs.push(tracker.events(frame)[0].epoch);
        }
        expect(tracker.events(20)).toHaveLength(1);
        expect(new Set(epochs)).toEqual(new Set([1, 9, 17]));
        expect(tracker.events(20)[0].box?.min[0]).toBeCloseTo(0.1);
        expect(tracker.events(20)[0].box?.max[0]).toBeCloseTo(3);
        tracker.update(21, none);
        expect(tracker.armedEvents(21).map((event) => event.epoch)).toEqual([21]);
    });

    it('keeps a continuous change with no bounds as one event, never more than its re-arms', () => {
        const tracker = new ChangeEventTracker();
        let armed = 0;
        for (let frame = 1; frame <= 60; frame++) {
            tracker.update(frame, { regions: [], global: ['sun'] });
            armed += tracker.armedEvents(frame).length;
            expect(tracker.events(frame)).toHaveLength(1);
            expect(tracker.events(frame)[0].box).toBeNull();
        }
        expect(armed).toBe(8); // frames 1, 9, ..., 57
    });

    it('merges past its limit by the cheapest union instead of escalating to everything, and the sources keep extending it', () => {
        const tracker = new ChangeEventTracker({ maxEvents: 2 });
        for (let frame = 1; frame <= 30; frame++) tracker.update(frame, { regions: [region(box(0), 1), region(box(0.5), 2), region(box(100), 3)], global: [] });
        const events = tracker.events(30);
        expect(events).toHaveLength(2);
        expect(events.every((event) => event.box !== null)).toBe(true);
        // The two near boxes merged; the far one kept its own; neither restarts every frame.
        expect(events.map((event) => event.start).sort()).toEqual([1, 1]);
    });

    it('joins sourceless regions that overlap', () => {
        const tracker = new ChangeEventTracker();
        tracker.update(1, { regions: [region(box(0)), region(box(0.5)), region(box(9))], global: [] });
        expect(tracker.events(1)).toHaveLength(2);
    });
});
