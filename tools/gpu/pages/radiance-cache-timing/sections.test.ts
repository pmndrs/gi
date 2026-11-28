import { readdirSync, readFileSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { describe, expect, it } from 'vitest';
import { DEFAULT_RADIANCE_CACHE_SETTINGS } from '../../../../packages/cache/src/settings.ts';
import { SHARC_ENTRY_POINT } from '../../../../packages/cache/src/world-cache/sharc/sharc-wgsl.ts';
import { groupOf, SECTIONS, sectionOf, sectionRuns, type SceneTiming, type TimedSettings } from './sections.ts';
import { radianceCacheTimingRows } from './to-csv.ts';

const RADIANCE_CACHE_SOURCES = resolve(import.meta.dirname, '..', '..', '..', '..', 'packages', 'cache', 'src');

/** Every TypeScript source under the radiance cache, tests left out. */
function sources(directory: string): string[] {
    return readdirSync(directory, { withFileTypes: true }).flatMap((entry) => {
        const path = join(directory, entry.name);
        if (entry.isDirectory()) return sources(path);
        return entry.name.endsWith('.ts') && !entry.name.endsWith('.test.ts') ? [path] : [];
    });
}

/**
 * The section names written as literals where the technique starts a timed
 * pass: `beginTimedPass(encoder, timer, 'X')` (screen probes) and
 * `this.beginPass(encoder, prepared, 'X')` and
 * `this.encodePopulationPass(..., 'kernel', 'X')` (the world caches), plus
 * `${TIMER_PREFIX}/X` (the neural tail).
 */
function literalSections(): Set<string> {
    const found = new Set<string>();
    const patterns = [/beginTimedPass\(encoder, timer, '([^']+)'\)/g, /beginPass\(encoder, prepared, '([^']+)'\)/g, /encodePopulationPass\([^)]*, '[A-Za-z]+', '([^']+)'\)/g, /\$\{TIMER_PREFIX\}\/([A-Za-z]+)`/g];
    for (const file of sources(RADIANCE_CACHE_SOURCES)) {
        const text = readFileSync(file, 'utf8');
        for (const pattern of patterns) for (const match of text.matchAll(pattern)) found.add(match[1]);
    }
    return found;
}

const DEFAULTS: TimedSettings = { ...DEFAULT_RADIANCE_CACHE_SETTINGS };

describe('radiance-cache-timing sections', () => {
    it('has a section for every timed pass the technique names in its sources', () => {
        const named = literalSections();
        // A guard on the scan itself: the screen probes, the hash grid and the neural tail all matched.
        expect(named).toContain('SpawnScreenProbes');
        expect(named).toContain('ResolveCells');
        expect(named).toContain('NeuralTailGather');
        for (const section of [...named, ...Object.values(SHARC_ENTRY_POINT)]) expect(groupOf(section), section).toBeDefined();
    });

    it('lists each section once', () => {
        const names = SECTIONS.map((entry) => entry.section);
        expect(new Set(names).size).toBe(names.length);
    });

    it('maps pass names to sections, the probe trace onto PopulateScreenProbesCells', () => {
        expect(sectionOf('radiance-cache/PopulateScreenProbesCells (trace)')).toBe('PopulateScreenProbesCells');
        expect(sectionOf('radiance-cache/PopulateScreenProbesCells')).toBe('PopulateScreenProbesCells');
        expect(sectionOf('radiance-cache/ReprojectScreenProbes')).toBe('ReprojectScreenProbes');
        expect(sectionOf('platform/top-level refit')).toBeNull();
    });

    it('runs the second-bounce sections with multibounce on and the hash grid only', () => {
        expect(sectionRuns('PopulateMultibounceCells', DEFAULTS)).toBe(true);
        expect(sectionRuns('PopulateMultibounceCells', { ...DEFAULTS, useMultibounce: false })).toBe(false);
        expect(sectionRuns('PopulateMultibounceCells', { ...DEFAULTS, worldCache: 'sharc' })).toBe(false);
        expect(sectionRuns('SharcLookup', { ...DEFAULTS, worldCache: 'sharc' })).toBe(true);
        expect(sectionRuns('ResolveCells', { ...DEFAULTS, worldCache: 'sharc' })).toBe(false);
        expect(sectionRuns('ShadeCells', DEFAULTS)).toBe(false);
        expect(sectionRuns('ShadeCells', { ...DEFAULTS, cellShading: 'budgeted' })).toBe(true);
        expect(sectionRuns('SSGI', DEFAULTS)).toBe(DEFAULTS.ssgi);
        expect(sectionRuns('NoSuchSection', DEFAULTS)).toBe(false);
    });
});

describe('radiance-cache-timing CSV rows', () => {
    const timing: SceneTiming = {
        triangles: 1000,
        maxTextureSize: 1024,
        settings: [
            {
                name: 'mb_on',
                settings: DEFAULTS,
                framesTimed: 120,
                passes: { 'radiance-cache/PopulateScreenProbesCells (trace)': 1.5, 'radiance-cache/PopulateScreenProbesCells': 0.5, 'radiance-cache/ResolveCells': 0.25 },
                sections: { PopulateScreenProbesCells: 2, ResolveCells: 0.25 },
                groups: { 'probe tracing and cell population': 2, 'cache update and resolve': 0.25, 'screen probes': 0, 'denoise and resolve': 0, ssgi: 0, 'ours only': 0 },
                diffuseSubtotalMs: 2.25,
                radianceCacheFrameMs: 2.25,
                platformPasses: { 'platform/top-level refit': 0.01 },
                memoryBytes: 123,
            },
        ],
    };
    const rows = radianceCacheTimingRows({ sponza: timing });
    const metric = (name: string): number | undefined => rows.find((row) => row.metric === name)?.value;

    it('writes sections, groups, the subtotal, memory and the platform passes, with CSV-safe names', () => {
        expect(metric('rc/mb_on/section/PopulateScreenProbesCells/gpu_ms_median')).toBe(2);
        expect(metric('rc/mb_on/group/probe_tracing_and_cell_population/gpu_ms')).toBe(2);
        expect(metric('rc/mb_on/diffuse_subtotal_gpu_ms')).toBe(2.25);
        expect(metric('rc/mb_on/memory_bytes')).toBe(123);
        expect(metric('rc/mb_on/platform/platform/top-level_refit/gpu_ms_median')).toBe(0.01);
        for (const row of rows) expect(row.metric).not.toMatch(/[ ,()]/);
    });

    it('writes pass rows only for a section made of several passes', () => {
        expect(metric('rc/mb_on/pass/radiance-cache/PopulateScreenProbesCells_trace/gpu_ms_median')).toBe(1.5);
        expect(metric('rc/mb_on/pass/radiance-cache/PopulateScreenProbesCells/gpu_ms_median')).toBe(0.5);
        expect(rows.some((row) => row.metric.includes('pass/radiance-cache/ResolveCells'))).toBe(false);
    });
});
