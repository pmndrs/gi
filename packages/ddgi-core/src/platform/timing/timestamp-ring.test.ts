import { describe, expect, it } from 'vitest';
import { TimestampRing } from './timestamp-ring';

describe('TimestampRing', () => {
    it('rejects sizes that cannot work', () => {
        expect(() => new TimestampRing(0, 4)).toThrow(RangeError);
        expect(() => new TimestampRing(4, 0)).toThrow(RangeError);
    });

    it('needs two queries per pass', () => {
        expect(new TimestampRing(3, 5).queriesPerSlot).toBe(10);
    });

    it('hands out slots in order and numbers frames by default', () => {
        const ring = new TimestampRing(3, 4);
        const first = ring.acquire();
        const second = ring.acquire();
        expect([first?.index, second?.index]).toEqual([0, 1]);
        expect([first?.sequence, second?.sequence]).toEqual([0, 1]);
        expect(first?.frame).toBe(0);
        expect(ring.acquire(42)?.frame).toBe(42);
    });

    it('returns undefined when every slot is busy, instead of overwriting one', () => {
        const ring = new TimestampRing(2, 4);
        const first = ring.acquire();
        ring.acquire();
        expect(ring.acquire()).toBeUndefined();
        // A slot that is only encoding is just as busy as one waiting for readback.
        ring.release(first!);
        expect(ring.acquire()?.index).toBe(0);
    });

    it('gives each pass its own pair of consecutive query indices', () => {
        const ring = new TimestampRing(1, 3);
        const slot = ring.acquire()!;
        expect(ring.addPass(slot, 'a')).toEqual({ beginIndex: 0, endIndex: 1 });
        expect(ring.addPass(slot, 'b')).toEqual({ beginIndex: 2, endIndex: 3 });
        expect(ring.addPass(slot, 'c')).toEqual({ beginIndex: 4, endIndex: 5 });
        expect(slot.passNames).toEqual(['a', 'b', 'c']);
    });

    it('refuses passes beyond capacity and outside encoding', () => {
        const ring = new TimestampRing(1, 1);
        const slot = ring.acquire()!;
        expect(ring.addPass(slot, 'a')).toBeDefined();
        expect(ring.addPass(slot, 'overflow')).toBeUndefined();
        ring.submit(slot);
        expect(ring.addPass(slot, 'late')).toBeUndefined();
    });

    it('submit reports the query count and keeps the slot busy until release', () => {
        const ring = new TimestampRing(2, 4);
        const slot = ring.acquire()!;
        ring.addPass(slot, 'a');
        ring.addPass(slot, 'b');
        expect(ring.submit(slot)).toBe(4);
        expect(slot.state).toBe('pending');
        expect(ring.pendingCount).toBe(1);
        expect(ring.idleCount).toBe(1);
        ring.release(slot);
        expect(ring.idleCount).toBe(2);
    });

    it('frees a slot immediately when no pass used it', () => {
        const ring = new TimestampRing(1, 4);
        const slot = ring.acquire()!;
        expect(ring.submit(slot)).toBe(0);
        expect(slot.state).toBe('idle');
    });

    it('clears the pass names when a slot is reused', () => {
        const ring = new TimestampRing(1, 4);
        const slot = ring.acquire()!;
        ring.addPass(slot, 'old');
        ring.submit(slot);
        ring.release(slot);
        const again = ring.acquire()!;
        expect(again.passNames).toEqual([]);
        expect(again.sequence).toBe(1);
    });

    it('wraps around the ring and skips slots that are still pending', () => {
        const ring = new TimestampRing(3, 2);
        const order: number[] = [];
        for (let frame = 0; frame < 6; frame++) {
            const slot = ring.acquire()!;
            order.push(slot.index);
            ring.addPass(slot, 'p');
            ring.submit(slot);
            // Slot 1 never gets its readback; the ring must route around it.
            if (slot.index !== 1) ring.release(slot);
        }
        expect(order).toEqual([0, 1, 2, 0, 2, 0]);
    });

    it('marks old results stale after invalidate and frees slots being encoded', () => {
        const ring = new TimestampRing(2, 2);
        const pending = ring.acquire()!;
        ring.addPass(pending, 'a');
        ring.submit(pending);
        const encoding = ring.acquire()!;

        ring.invalidate();

        expect(ring.isStale(pending)).toBe(true);
        expect(encoding.state).toBe('idle');
        // The pending slot waits for its readback to end, then is reusable.
        expect(pending.state).toBe('pending');
        const fresh = ring.acquire()!;
        expect(ring.isStale(fresh)).toBe(false);
    });
});
