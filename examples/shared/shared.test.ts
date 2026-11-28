/**
 * The shared pieces' logic that needs no page: the explainer's inline text,
 * the timeline's settle reading, the pass list's means and descriptions,
 * and the wipe's clamp. The DOM and GPU parts are checked by the examples
 * site (tools/gpu/pages/examples-site).
 */

import { describe, expect, it } from 'vitest';
import { parseInline } from './explainer.ts';
import { framesToSettle } from './timeline.ts';
import { describePass, PassAverager } from './pass-timing.ts';
import { clampSplit } from './wipe.ts';

describe('parseInline', () => {
    it('splits plain text, code and links', () => {
        expect(parseInline('Use `DDGIVolume`, see [the guide](https://example.com/g).')).toEqual([
            { kind: 'text', text: 'Use ' },
            { kind: 'code', text: 'DDGIVolume' },
            { kind: 'text', text: ', see ' },
            { kind: 'link', text: 'the guide', href: 'https://example.com/g' },
            { kind: 'text', text: '.' },
        ]);
    });

    it('leaves text without markup alone', () => {
        expect(parseInline('No markup <b>here</b>.')).toEqual([{ kind: 'text', text: 'No markup <b>here</b>.' }]);
    });
});

describe('framesToSettle', () => {
    it('counts frames from the marker until the value stays near where it ends', () => {
        const values = [1, 1, 1, 2, 3, 3.5, 3.9, 4, 4, 4, 4, 4];
        // Marker at index 2; within 2% of 4 from index 7 on, five values held.
        expect(framesToSettle(values, 2, 0.02, 5)).toBe(5);
    });

    it('is null until the value has held for long enough', () => {
        expect(framesToSettle([1, 2, 3, 4, 4], 0, 0.02, 5)).toBeNull();
    });

    it('skips a gap before the value settles; a gap at the end, or no marker, reads null', () => {
        expect(framesToSettle([1, null, 4, 4, 4], 0, 0.02, 3)).toBe(2);
        expect(framesToSettle([1, 4, 4, null], 0, 0.02, 1)).toBeNull();
        expect(framesToSettle([1, 2], -1, 0.02, 1)).toBeNull();
    });
});

describe('PassAverager', () => {
    const frame = (number: number, passes: Array<[string, number]>) => passes.map(([name, gpuMs]) => ({ name, gpuMs, frame: number }));

    it('adds up a name within a frame and averages over frames, absent counting as 0', () => {
        const averager = new PassAverager(4);
        expect(averager.add(frame(1, [['ddgi/trace', 1], ['ddgi/trace', 0.5], ['ddgi/blend', 0.2]]))).toBe(true);
        expect(averager.add(frame(1, [['ddgi/trace', 9]]))).toBe(false);
        averager.add(frame(2, [['ddgi/trace', 0.5], ['ddgi/blend', 0.2], ['ddgi/propagate', 0.4]]));
        const rows = averager.rows();
        expect(rows.map((row) => row.name)).toEqual(['ddgi/trace', 'ddgi/blend', 'ddgi/propagate']);
        expect(rows[0].ms).toBeCloseTo(1, 6);
        expect(rows[1].ms).toBeCloseTo(0.2, 6);
        expect(rows[2].ms).toBeCloseTo(0.2, 6);
    });

    it('keeps only the last frames', () => {
        const averager = new PassAverager(2);
        averager.add(frame(1, [['a/x', 10]]));
        averager.add(frame(2, [['a/x', 1]]));
        averager.add(frame(3, [['a/x', 3]]));
        expect(averager.rows()[0].ms).toBeCloseTo(2, 6);
        expect(averager.frameCount).toBe(2);
    });

    it('ignores untimed passes', () => {
        const averager = new PassAverager();
        expect(averager.add([{ name: 'ddgi/trace', gpuMs: undefined, frame: 1 }])).toBe(false);
    });
});

describe('describePass', () => {
    it('knows the DDGI, platform and radiance-cache passes', () => {
        for (const name of ['ddgi/trace', 'ddgi/blend', 'gi/light-grid', 'radiance-cache/SpawnScreenProbes', 'radiance-cache/PopulateScreenProbesCells (trace)', 'radiance-cache/SharcLookup']) {
            expect(describePass(name), name).toBeTruthy();
        }
        expect(describePass('ddgi/gi-direct')).toMatch(/shadow ray/);
        expect(describePass('nobody/knows')).toBeUndefined();
        expect(describePass('nobody/knows', { 'nobody/knows': 'Now it does.' })).toBe('Now it does.');
    });
});

describe('clampSplit', () => {
    it('keeps the line inside the view', () => {
        expect(clampSplit(-1)).toBeCloseTo(0.02);
        expect(clampSplit(2)).toBeCloseTo(0.98);
        expect(clampSplit(0.3)).toBe(0.3);
        expect(clampSplit(Number.NaN)).toBe(0.5);
    });
});
