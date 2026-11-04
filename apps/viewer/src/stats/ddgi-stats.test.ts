import { describe, expect, it } from 'vitest';
import { DdgiStatsText, type DdgiFrameStats } from './ddgi-stats.ts';

function frame(overrides: Partial<DdgiFrameStats> = {}): DdgiFrameStats {
    return {
        probeCount: 19_425,
        probesUpdated: 800,
        budgetMs: 2,
        updatePasses: new Map([
            ['trace', 1.2],
            ['blend', 0.5],
            ['write', 0.1],
        ]),
        platformPasses: new Map([['top-level refit', 0.02]]),
        memory: { sceneCopy: 130_100_000, acceleration: 276_000_000, techniques: { ddgi: 9_500_000 }, total: 415_600_000 },
        syncCpuMs: 0.09,
        encodeCpuMs: 0.3,
        states: [],
        ...overrides,
    };
}

describe('DdgiStatsText', () => {
    it('is empty before the first frame', () => {
        expect(new DdgiStatsText().lines()).toEqual([]);
    });

    it('says what each number is, with the update split into its passes', () => {
        const text = new DdgiStatsText();
        text.add(frame());
        const lines = text.lines().map((line) => line.text);
        expect(lines).toContain('  probes updated  800 of 19,425, each every 25 frames');
        expect(lines).toContain('  update (GPU)    1.80 ms (budget 2.00)  trace 1.20, blend 0.50, write 0.10');
        expect(lines).toContain('  top-level refit (GPU)  0.02 ms');
        expect(lines).toContain('  GPU memory      416 MB  (scene 130 MB, ray tracing 276 MB, probes 9.5 MB)');
        expect(lines).toContain('  adapter (CPU)   sync 0.09 ms, record 0.30 ms');
    });

    it('gives every line a hover help key', () => {
        const text = new DdgiStatsText();
        text.add(frame({ states: [ { label: 'active', count: 700 }, { label: 'off: inside geometry', count: 100 } ] }));
        const lines = text.lines();
        expect(lines.every((line) => line.help.startsWith('stats'))).toBe(true);
        expect(lines.at(-1)?.text).toBe('  active 700, off: inside geometry 100');
    });

    it('says plainly when there are no GPU timings', () => {
        const text = new DdgiStatsText();
        text.add(frame({ updatePasses: new Map(), platformPasses: new Map() }));
        expect(text.lines().map((line) => line.text)).toContain('  update (GPU)    - (no GPU timestamps)');
    });

    it('keeps warm-up timings out of the averages', () => {
        const text = new DdgiStatsText();
        text.add(frame({ updatePasses: new Map(), platformPasses: new Map(), timingsPending: true }));
        expect(text.lines().map((line) => line.text)).toContain('  update (GPU)    measuring...');
        text.add(frame());
        expect(text.lines().map((line) => line.text)).toContain('  update (GPU)    1.80 ms (budget 2.00)  trace 1.20, blend 0.50, write 0.10');
    });

    it('says the top-level refit had nothing to do when nothing moved', () => {
        const text = new DdgiStatsText();
        text.add(frame({ platformPasses: new Map() }));
        expect(text.lines().map((line) => line.text)).toContain('  top-level refit (GPU)  none (nothing moved)');
    });

    it('shows the budget beside the update time, so an overshoot is plain', () => {
        const text = new DdgiStatsText();
        text.add(frame({ budgetMs: 2, updatePasses: new Map([['trace', 2.5], ['census', 0.4]]) }));
        expect(text.lines().map((line) => line.text)).toContain('  update (GPU)    2.90 ms (budget 2.00)  trace 2.50, census 0.40');
    });

    it('smooths the numbers over frames', () => {
        const text = new DdgiStatsText();
        text.add(frame({ probesUpdated: 1000 }));
        text.add(frame({ probesUpdated: 0 }));
        // 0.9 of the old average stays: 900.
        expect(text.lines()[1].text).toContain('900 of 19,425');
    });
});
