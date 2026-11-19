import { describe, expect, it } from 'vitest';
import { readFile } from 'node:fs/promises';
import { join } from 'node:path';
import type { RunSummary } from '../../contract.ts';
import { appleAdapter, appleMetalDevices, budgetPage, checkQuality, compareWindows, distribution, idleWindow, pagesPassed, qualityPage, readPair,
    utilizationCounters } from './metal-evidence.ts';

const adapter = { vendor: 'apple', architecture: 'apple-7', device: '', description: '', isFallbackAdapter: false, features: ['timestamp-query'] };
function report(name: string, data: Record<string, unknown>): RunSummary {
    return { schema: 1, startedAt: '', gpuRequested: null, adapter, ok: true, totalMs: 1, error: null, selfTests: [],
        pages: [{ name, data, outcome: 'pass', acceptable: true, durationMs: 1, reasons: [], adapter, screenshot: 'result.png' }] };
}
function timing(count: number) { return { count, samplesMs: Array.from({ length: count }, (_, i) => i + 1) }; }
const arm = { ...timing(60), animationIntervals: timing(59), cascadeUpdates: 60, threeTimestampPools: 0 };
const pair = { pair: 2, order: ['off', 'on'], resolution: [960, 540], internal: [640, 360], ratio: 1.5, warmFrames: 48,
    measuredFrames: 60, measuredTicks: [128, 187], on: arm, off: arm, floor: { pixels: 1, rgbMae255: 2, signedDisplayLuma255: 1 } };
const metrics = { consecutivePairs: 20, frameRgbMae255: Array(24).fill(1), allRgbMae255: 1,
    referenceEdgeRgbMae255: 2, floorRgbMae255: 1, adjacentChangeRgbMae255: 1, floorAdjacentChangeRgbMae255: 1, edgeFraction: 0.1 };
const quality = { display: [960, 540], ticks: [31,121,211,301].flatMap(start => [0,1,2,3,4,5].map(i => start+i)),
    repeatReference: metrics, 'ratio1.5': metrics, ratio2: metrics };

describe('Mac polished evidence', () => {
    it('requires Apple hardware and a real WebGPU timestamp adapter on each page', () => {
        const device = { sppci_model: 'Apple M5 Pro', spdisplays_mtlgpufamilysupport: 'spdisplays_metal4' };
        expect(appleMetalDevices({ SPDisplaysDataType: [device] })[0].model).toBe('Apple M5 Pro');
        expect(() => appleMetalDevices({ SPDisplaysDataType: [{ ...device, spdisplays_mtlgpufamilysupport: 'Not Supported' }] })).toThrow();
        for (const change of [{ vendor: 'nvidia' }, { isFallbackAdapter: true }, { description: 'SwiftShader' }, { features: [] }]) {
            expect(() => appleAdapter({ ...adapter, ...change })).toThrow();
        }
        const run = report('rc-transport', {});
        run.pages[0].adapter = null;
        expect(() => pagesPassed(run, ['rc-transport'])).toThrow();
        expect(() => pagesPassed(report('rc-transport', {}), ['rc-transport', 'rc-native-core'])).toThrow();
    });
    it('rejects wrong arm order, truncated samples or silent floor', () => {
        expect(readPair(report(budgetPage, pair), 2).order).toEqual(['off', 'on']);
        for (const change of [{ order: ['on','off'] }, { off: { ...arm, samplesMs: [0] } }, { on: { ...arm, cascadeUpdates: 59 } },
            { floor: { ...pair.floor, signedDisplayLuma255: 0 } }]) {
            expect(() => readPair(report(budgetPage, { ...pair, ...change }), 2)).toThrow();
        }
        expect(distribution([5, 4, 2, 3])).toEqual({ count: 4, medianMs: 3.5, p95Ms: 5 });
    });
    it('requires 480 contiguous updates and preserves the first measured window in each uncapped arm', () => {
        const sustained = { ...timing(480), animationIntervals: timing(472), cascadeUpdates: 480 };
        const extendedArm = { ...arm, sustained };
        const data = { ...pair, on: extendedArm, off: extendedArm, sustainedTicks: [128, 607] };
        expect(readPair(report(budgetPage, data), 2, 'uncapped').off.sustained?.count).toBe(480);
        expect(() => readPair(report(budgetPage, data), 2, 'normal')).toThrow();
        expect(() => readPair(report(budgetPage, pair), 2, 'uncapped')).toThrow();
        for (const change of [{ cascadeUpdates: 479 }, { count: 479 }, { samplesMs: sustained.samplesMs.slice(1) },
            { samplesMs: sustained.samplesMs.slice().reverse() }, { animationIntervals: timing(479) }]) {
            const test = { ...data, off: { ...extendedArm, sustained: { ...sustained, ...change } } };
            expect(() => readPair(report(budgetPage, test), 2, 'uncapped')).toThrow();
        }
        expect(() => readPair(report(budgetPage, { ...data, sustainedTicks: [128, 606] }), 2, 'uncapped')).toThrow();
    });
    // Optional replay of real pre-existing runner reports; the mock replaces only the adapter evidence in memory.
    const archive = process.env.RC_METAL_CAPTURE_REPLAY;
    it.runIf(archive)('replays captured Windows normal and uncapped arms without starting a GPU', async () => {
        for (const [folder, cadence] of [['run-02', 'normal'], ['uncapped-01', 'uncapped']] as const) {
            const pairs = [];
            for (const n of [1, 2, 3]) {
                const run: RunSummary = JSON.parse(await readFile(join(archive!, folder, 'pair-' + n + '.json'), 'utf8'));
                expect(run.adapter?.vendor).toBe('nvidia');
                run.adapter = adapter;
                for (const page of run.pages) page.adapter = adapter;
                pairs.push(readPair(run, n, cadence));
            }
            const expected = JSON.parse(await readFile(join(archive!, folder, 'summary.json'), 'utf8'));
            const matched = compareWindows(pairs.flatMap(pair => pair.on.samplesMs), pairs.flatMap(pair => pair.off.samplesMs));
            expect(matched.on).toEqual(expected.on);
            expect(matched.off).toEqual(expected.off);
            if (cadence === 'uncapped') {
                const extended = compareWindows(pairs.flatMap(pair => pair.on.sustained!.samplesMs), pairs.flatMap(pair => pair.off.sustained!.samplesMs));
                expect(extended.on).toEqual(expected.sustained.on);
                expect(extended.off).toEqual(expected.sustained.off);
                expect(extended.on.count).toBe(1440);
            }
        }
    });
    it('requires all 24 quality ticks, all ratios and 20 within-phase differences', () => {
        expect(checkQuality(report(qualityPage, quality))).toEqual(quality);
        for (const change of [{ ticks: quality.ticks.slice(1) }, { ratio2: { ...metrics, consecutivePairs: 23 } },
            { 'ratio1.5': { ...metrics, frameRgbMae255: [1] } }]) {
            expect(() => checkQuality(report(qualityPage, { ...quality, ...change }))).toThrow();
        }
    });
    it('only reports available IORegistry percentages in physical range', () => {
        expect(utilizationCounters(' | "PerformanceStatistics" = {"Device Utilization %"=2,"Renderer Utilization %"=100,"Ignored"=77}'))
            .toEqual([{ 'Device Utilization %': 2, 'Renderer Utilization %': 100 }]);
        expect(utilizationCounters(' "PerformanceStatistics" = {"Device Utilization %"=101}')).toEqual([]);
        expect(utilizationCounters('')).toEqual([]);
    });
    it('waits on busy or elevated observations, treating engine-only and missing samples as unknown', () => {
        const read = (thermalPressure: string | null, counters: Record<string, number> | null) =>
            ({ thermalPressure, utilizationPercent: counters ? [counters] : null });
        const idle = read('nominal', { 'Device Utilization %': 19 });
        expect(idleWindow(Array(5).fill(idle)).verdict).toBe('observed-idle-and-nominal');
        expect(idleWindow(Array(4).fill(idle)).verdict).toBe('unassessed');
        expect(idleWindow(Array(5).fill(read('nominal', { 'Renderer Utilization %': 1 }))).verdict).toBe('unassessed');
        expect(idleWindow(Array(5).fill(read(null, { 'Device Utilization %': 1 }))).verdict).toBe('unassessed');
        expect(idleWindow([idle, read(null, { 'Device Utilization %': 20 })]).verdict).toBe('busy-or-elevated');
        expect(idleWindow([read('serious', null)]).verdict).toBe('busy-or-elevated');
    });
});
