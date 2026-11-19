import assert from 'node:assert/strict';
import type { AdapterSummary, RunSummary } from '../../contract.ts';

export const correctnessPages = ['rc-transport', 'rc-native-core', 'rc-native-reflections', 'rc-door-response',
    'rc-backends', 'rc-reconstruction', 'rc-rail', 'rc-rough-motion'];
export const qualityPage = 'rc-polished-quality';
export const budgetPage = 'rc-polished-budget';

export function appleMetalDevices(profile: { SPDisplaysDataType?: Record<string, unknown>[] }) {
    const devices = (profile.SPDisplaysDataType ?? []).map(display => ({
        model: String(display.sppci_model ?? display._name ?? ''),
        metal: Object.fromEntries(Object.entries(display).filter(([key]) => /metal|mtl/i.test(key))),
    })).filter(({ model, metal }) => /^Apple\s/i.test(model) && Object.values(metal).some(value =>
        /metal|supported|spdisplays_yes/i.test(String(value)) && !/unsupported|not.?supported/i.test(String(value))));
    assert(devices.length > 0, 'system_profiler did not identify an Apple GPU with Metal support');
    return devices;
}

export function appleAdapter(adapter: AdapterSummary | null) {
    assert(adapter && /^(apple|0x106b)$/i.test(adapter.vendor) && adapter.isFallbackAdapter === false &&
        !/swiftshader|llvmpipe|software|fallback/i.test([adapter.architecture, adapter.device, adapter.description].join(' ')),
        'Chrome did not report a nonfallback Apple WebGPU adapter');
    assert(adapter.features.includes('timestamp-query'), 'Apple WebGPU adapter does not expose timestamp-query');
    return adapter;
}

export function pagesPassed(run: RunSummary, names: string[]) {
    assert(run.ok, run.error ?? 'GPU harness reported a failure');
    appleAdapter(run.adapter);
    assert.deepEqual(run.pages.map(page => page.name).sort(), [...names].sort(), 'unexpected GPU page set');
    for (const page of run.pages) {
        assert(page.acceptable && page.outcome === 'pass', page.name + ': ' + page.reasons.join(', '));
        appleAdapter(page.adapter);
    }
    return run.pages;
}

export function distribution(samples: number[]) {
    assert(samples.length && samples.every(n => Number.isFinite(n) && n > 0), 'invalid timing samples');
    const sorted = [...samples].sort((a, b) => a - b);
    return { count: sorted.length, medianMs: (sorted[(sorted.length - 1) >> 1] + sorted[sorted.length >> 1]) / 2,
        p95Ms: sorted[Math.ceil(0.95 * sorted.length) - 1] };
}

export type Cadence = 'normal' | 'uncapped';
type TimingWindow = { count: number; samplesMs: number[]; animationIntervals: { count: number; samplesMs: number[] };
    cascadeUpdates: number };
type Timing = TimingWindow & { threeTimestampPools: number; sustained?: TimingWindow };
type Pair = { pair: number; order: string[]; resolution: number[]; internal: number[]; ratio: number;
    warmFrames: number; measuredFrames: number; measuredTicks: number[]; sustainedTicks?: number[] | null; on: Timing; off: Timing;
    floor: { pixels: number; rgbMae255: number; signedDisplayLuma255: number } };

function checkWindow(timing: TimingWindow, frames: number, intervals: number) {
    assert.equal(timing.count, frames);
    assert.equal(distribution(timing.samplesMs).count, frames);
    assert.equal(timing.cascadeUpdates, frames);
    assert.equal(timing.animationIntervals.count, intervals);
    assert.equal(distribution(timing.animationIntervals.samplesMs).count, intervals);
}

export function readPair(report: RunSummary, pair: number, cadence: Cadence = 'normal') {
    const [page] = pagesPassed(report, [budgetPage]);
    const data = page.data as unknown as Pair;
    assert.equal(data.pair, pair);
    assert.deepEqual(data.order, pair === 2 ? ['off', 'on'] : ['on', 'off']);
    assert.deepEqual(data.resolution, [960, 540]);
    assert.deepEqual(data.internal, [640, 360]);
    assert.equal(data.ratio, 1.5);
    assert.equal(data.warmFrames, 48);
    assert.equal(data.measuredFrames, 60);
    assert.deepEqual(data.measuredTicks, [128, 187]);
    assert.deepEqual(data.sustainedTicks ?? null, cadence === 'uncapped' ? [128, 607] : null);
    for (const arm of [data.on, data.off]) {
        checkWindow(arm, 60, 59);
        assert.equal(arm.threeTimestampPools, 0);
        if (cadence === 'uncapped') {
            assert(arm.sustained, 'missing the extended timing window');
            checkWindow(arm.sustained, 480, 8 * 59);
            assert.deepEqual(arm.sustained.samplesMs.slice(0, 60), arm.samplesMs);
            assert.deepEqual(arm.sustained.animationIntervals.samplesMs.slice(0, 59), arm.animationIntervals.samplesMs);
        } else {
            assert.equal(arm.sustained, undefined, 'normal cadence unexpectedly included extended timings');
        }
    }
    assert(data.floor.pixels > 0 && data.floor.rgbMae255 > 1 && data.floor.signedDisplayLuma255 > 0,
        'reflection gather produced no measured floor gain');
    return data;
}

export function compareWindows(onSamples: number[], offSamples: number[]) {
    const on = distribution(onSamples), off = distribution(offSamples);
    return { on, off, medianDeltaMs: on.medianMs - off.medianMs, p95DeltaMs: on.p95Ms - off.p95Ms };
}

export function checkQuality(report: RunSummary) {
    const [page] = pagesPassed(report, [qualityPage]);
    const ticks = [31, 121, 211, 301].flatMap(start => Array.from({ length: 6 }, (_, i) => start + i));
    assert.deepEqual(page.data.ticks, ticks);
    assert.deepEqual(page.data.display, [960, 540]);
    for (const key of ['repeatReference', 'ratio1.5', 'ratio2']) {
        const values = page.data[key] as Record<string, number | number[]>;
        assert.equal(values.consecutivePairs, 20);
        assert(Array.isArray(values.frameRgbMae255) && values.frameRgbMae255.length === ticks.length &&
            values.frameRgbMae255.every(n => Number.isFinite(n) && n >= 0), 'incomplete phase results for ' + key);
        for (const name of ['allRgbMae255', 'referenceEdgeRgbMae255', 'floorRgbMae255', 'adjacentChangeRgbMae255',
            'floorAdjacentChangeRgbMae255', 'edgeFraction']) {
            assert(typeof values[name] === 'number' && Number.isFinite(values[name]) && values[name] >= 0,
                'invalid ' + key + '.' + name);
        }
    }
    return page.data;
}

/** Keep only utilization counters. IORegistry also contains user-irrelevant device identifiers and display data. */
export function utilizationCounters(ioreg: string) {
    return ioreg.split(/\r?\n/).filter(line => line.includes('"PerformanceStatistics"')).flatMap(line => {
        const counters: Record<string, number> = {};
        for (const match of line.matchAll(/"([^"]*Utilization[^"]*%)"\s*=\s*(\d+(?:\.\d+)?)/gi)) {
            const value = Number(match[2]);
            if (value >= 0 && value <= 100) counters[match[1]] = value;
        }
        return Object.keys(counters).length ? [counters] : [];
    });
}

export function idleWindow(readings: Array<{ thermalPressure: string | null; utilizationPercent: Record<string, number>[] | null }>) {
    // Tiler and renderer readings describe individual engines. Only the named whole-device counter can establish idle.
    const totals = readings.map(sample => sample.utilizationPercent?.flatMap(counter => Object.entries(counter)
        .filter(([name]) => /^Device Utilization %$/i.test(name)).map(([, n]) => n)) ?? []);
    const complete = readings.length === 5;
    const busy = totals.some(devices => devices.some(n => n >= 20));
    const elevated = readings.some(sample => sample.thermalPressure !== null && sample.thermalPressure !== 'nominal');
    const deviceUtilization = busy ? 'busy' : complete && totals.every(devices => devices.length > 0) ? 'below-20-percent' : 'unassessed';
    const systemPressure = elevated ? 'elevated' : complete && readings.every(sample => sample.thermalPressure === 'nominal') ? 'nominal' : 'unassessed';
    const verdict = busy || elevated ? 'busy-or-elevated' : deviceUtilization === 'below-20-percent' && systemPressure === 'nominal'
        ? 'observed-idle-and-nominal' : 'unassessed';
    return { deviceUtilization, systemPressure, verdict };
}
