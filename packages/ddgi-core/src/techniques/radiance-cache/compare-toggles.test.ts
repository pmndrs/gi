import { describe, expect, it, vi } from 'vitest';
import {
    applyCompareValues,
    capsaicinCompareValues,
    COMPARE_GROUPS,
    compareSettingValue,
    hostCompareValue,
    isHostToggle,
    compareValuesFromText,
    parseCompareValue,
    RADIANCE_CACHE_COMPARE_TOGGLES,
    readCompareValues,
    writeCompareValues,
    type CompareToggle,
    type CompareValues,
} from './compare-toggles.ts';
import { RadianceCacheTechnique, temporalRuleUniforms } from './radiance-cache-technique.ts';
import { DEFAULT_RADIANCE_CACHE_SETTINGS, PREPARE_SETTINGS, SCREEN_HISTORY_SETTINGS, type RadianceCacheSettings } from './settings.ts';
import { BLEND_WGSL, FRAME_UNIFORM_OFFSETS, SAMPLE_WGSL } from './wgsl/screen-probes.ts';
import { REPROJECT_WGSL } from './wgsl/temporal-probes.ts';

/** A value other than the default: Capsaicin's where it differs, else the other end or choice. */
function otherValue(entry: CompareToggle): boolean | number | string {
    if (entry.capsaicin !== undefined && entry.capsaicin !== entry.defaultValue) return entry.capsaicin;
    if (entry.kind === 'boolean') return !entry.defaultValue;
    if (entry.kind === 'number') return entry.defaultValue === entry.min ? entry.max : entry.min;
    return entry.choices.find((choice) => choice.value !== entry.defaultValue)!.value;
}

describe('the compare toggles (RC4.5)', () => {
    it('each names a setting, its default and a valid Capsaicin value, with a unique link name and help', () => {
        const urlNames = new Set<string>();
        for (const entry of RADIANCE_CACHE_COMPARE_TOGGLES) {
            const top = isHostToggle(entry) || entry.setting.startsWith('sharc.') ? null : (entry.setting as keyof RadianceCacheSettings);
            if (top) expect(top in DEFAULT_RADIANCE_CACHE_SETTINGS).toBe(true);
            expect(entry.defaultValue).toBe(compareSettingValue(DEFAULT_RADIANCE_CACHE_SETTINGS, entry.setting));
            expect(entry.defaultValue).toBeDefined();
            expect(COMPARE_GROUPS).toContain(entry.group);
            expect(entry.urlName).toMatch(/^[a-z0-9]+$/);
            expect(urlNames.has(entry.urlName)).toBe(false);
            urlNames.add(entry.urlName);
            expect(entry.help.length).toBeGreaterThan(20);
            expect(entry.help).toMatch(/Capsaicin/);
            expect(entry.resetsHistory).toBe(top !== null && SCREEN_HISTORY_SETTINGS.includes(top));
            expect(entry.restartsTechnique).toBe(isHostToggle(entry) || (top !== null && PREPARE_SETTINGS.includes(top)));
            // Capsaicin's value (where it has one) and the default are values the toggle can take.
            for (const value of [entry.capsaicin, entry.defaultValue]) {
                if (value === undefined) continue;
                if (entry.kind === 'boolean') expect(typeof value).toBe('boolean');
                if (entry.kind === 'choice') expect(entry.choices.map((choice) => choice.value)).toContain(value);
                if (entry.kind === 'number') {
                    expect(value).toBeGreaterThanOrEqual(entry.min);
                    expect(value).toBeLessThanOrEqual(entry.max);
                }
            }
        }
    });

    it("covers every temporal change RC4.1 made to Capsaicin, each with Capsaicin's value differing from ours", () => {
        const differ = RADIANCE_CACHE_COMPARE_TOGGLES.filter((entry) => entry.capsaicin !== entry.defaultValue).map((entry) => entry.setting);
        expect(differ).toEqual(expect.arrayContaining(['probeBlend', 'historySampleCap', 'emptyCellFill', 'backupFillsAsHistory', 'rayGuidingUniformShare', 'probeChangeReset']));
        expect(RADIANCE_CACHE_COMPARE_TOGGLES.map((entry) => entry.setting)).toEqual(expect.arrayContaining(['interpolationJitter', 'probeHistory']));
    });

    it("covers RC4.3's denoiser: on/off, its history clamp and clip, its change reset (D65, D66)", () => {
        const byName = new Map(RADIANCE_CACHE_COMPARE_TOGGLES.map((entry) => [entry.setting, entry]));
        expect(byName.get('denoiser')).toMatchObject({ group: 'denoiser', defaultValue: true, capsaicin: true, register: 'D65' });
        expect(byName.get('denoiserHistoryClamp')).toMatchObject({ group: 'denoiser', defaultValue: 'neighbourhood', capsaicin: 'none', register: 'D66' });
        expect(byName.get('denoiserHistoryClip')).toMatchObject({ group: 'denoiser', defaultValue: 2, register: 'D66' });
        expect(byName.get('denoiserHistoryClip')!.capsaicin).toBeUndefined();
        expect(byName.get('denoiserChangeReset')).toMatchObject({ group: 'denoiser', defaultValue: true, capsaicin: false, register: 'D66' });
        expect(capsaicinCompareValues()).toMatchObject({ denoiserHistoryClamp: 'none', denoiserChangeReset: false });
    });

    it("covers RC4.2's probe filter and its history, and the ray-guiding correction (D59, D64): E9's noise gate turns the first and last off by name", () => {
        const byName = new Map(RADIANCE_CACHE_COMPARE_TOGGLES.map((entry) => [entry.setting, entry]));
        expect(byName.get('probeFilter')).toMatchObject({ group: 'screen-probes', urlName: 'rcfilter', defaultValue: true, capsaicin: true, resetsHistory: false, restartsTechnique: false });
        expect(byName.get('probeFilterHistory')).toMatchObject({ group: 'screen-probes', defaultValue: false, capsaicin: true, register: 'D59' });
        expect(byName.get('rayGuidingCorrection')).toMatchObject({ group: 'screen-probes', urlName: 'rcguidefix', defaultValue: true, register: 'D64' });
        // radiance-cache-change-noise's calibration run (DECISIONS 2026-11-24).
        expect(compareValuesFromText('rcfilter=false,rcguidefix=false')).toEqual({ probeFilter: false, rayGuidingCorrection: false });
        const technique = new RadianceCacheTechnique();
        applyCompareValues(technique.settings, { probeFilter: false });
        expect(technique.settings.probeFilter).toBe(false);
    });

    it('round-trips every value through link parameters, writing only what differs from the defaults', () => {
        for (const entry of RADIANCE_CACHE_COMPARE_TOGGLES) {
            for (const value of [entry.capsaicin ?? entry.defaultValue, otherValue(entry)]) {
                const values = { [entry.setting]: value } as CompareValues;
                const params = new URLSearchParams();
                writeCompareValues(values, params);
                if (value === entry.defaultValue) {
                    expect(params.toString()).toBe('');
                    continue;
                }
                expect([...params.keys()]).toEqual([entry.urlName]);
                expect(readCompareValues(new URLSearchParams(params.toString()))).toEqual(values);
            }
        }
        const all = capsaicinCompareValues();
        const params = new URLSearchParams();
        writeCompareValues(all, params);
        expect(readCompareValues(new URLSearchParams(params.toString()))).toEqual(all);
        // A default in a link is dropped; an unknown value is ignored.
        expect(readCompareValues(new URLSearchParams('rcblend=running-mean&rcfill=sideways&rccap=abc'))).toEqual({});
    });

    it('clamps numbers to their range and step, and takes booleans as 1 and 0', () => {
        const cap = RADIANCE_CACHE_COMPARE_TOGGLES.find((entry) => entry.setting === 'historySampleCap')!;
        expect(parseCompareValue(cap, '500')).toBe(32);
        expect(parseCompareValue(cap, '3.4')).toBe(3);
        const share = RADIANCE_CACHE_COMPARE_TOGGLES.find((entry) => entry.setting === 'rayGuidingUniformShare')!;
        expect(parseCompareValue(share, '0.13')).toBe(0.125);
        const reset = RADIANCE_CACHE_COMPARE_TOGGLES.find((entry) => entry.setting === 'probeChangeReset')!;
        expect(parseCompareValue(reset, '0')).toBe(false);
        expect(parseCompareValue(reset, 'yes')).toBeUndefined();
    });

    it('reach the technique: its settings, and the RcFrame words the kernels read', () => {
        // A host choice stays out of the technique's settings and is read from the values.
        expect(hostCompareValue({}, 'host.cacheAlone')).toBe(false);
        expect(hostCompareValue(capsaicinCompareValues(), 'host.cacheAlone')).toBe(true);
        const untouched = new RadianceCacheTechnique();
        applyCompareValues(untouched.settings, { 'host.cacheAlone': true });
        expect(Object.keys(untouched.settings)).not.toContain('host.cacheAlone');
        for (const entry of RADIANCE_CACHE_COMPARE_TOGGLES.filter((candidate) => !isHostToggle(candidate))) {
            const technique = new RadianceCacheTechnique();
            applyCompareValues(technique.settings, { [entry.setting]: entry.capsaicin ?? otherValue(entry) } as CompareValues);
            expect(compareSettingValue(technique.settings, entry.setting)).toBe(entry.capsaicin ?? otherValue(entry));
            // Every other toggle at its default.
            for (const other of RADIANCE_CACHE_COMPARE_TOGGLES) if (other !== entry && !isHostToggle(other)) expect(compareSettingValue(technique.settings, other.setting)).toBe(other.defaultValue);
            applyCompareValues(technique.settings, {});
            expect(compareSettingValue(technique.settings, entry.setting)).toBe(entry.defaultValue);
        }
        const settings: RadianceCacheSettings = { ...DEFAULT_RADIANCE_CACHE_SETTINGS };
        expect(temporalRuleUniforms(settings)).toEqual({ historySampleCap: 8, rayGuidingUniformShare: 0.125, capsaicinHysteresis: 0, capsaicinEmptyCells: 0, fillHitDistance: -1, probeChangeReset: 1, rayGuidingCorrection: 1 });
        applyCompareValues(settings, capsaicinCompareValues());
        expect(temporalRuleUniforms(settings)).toEqual({ historySampleCap: 1, rayGuidingUniformShare: 0, capsaicinHysteresis: 1, capsaicinEmptyCells: 1, fillHitDistance: 65504, probeChangeReset: 0, rayGuidingCorrection: 1 });
        // Each RcFrame field has an offset, and the kernels read them (no compile-time constants left).
        for (const field of Object.keys(temporalRuleUniforms(settings))) expect(FRAME_UNIFORM_OFFSETS).toHaveProperty(field);
        expect(BLEND_WGSL).toContain('rcFrame.capsaicinHysteresis');
        expect(BLEND_WGSL).toContain('rcFrame.capsaicinEmptyCells');
        expect(BLEND_WGSL).toContain('rcFrame.historySampleCap');
        expect(BLEND_WGSL).toContain('rcFrame.probeChangeReset');
        expect(BLEND_WGSL).toContain('rcFrame.fillHitDistance');
        expect(BLEND_WGSL).toContain('rcFrame.rayGuidingCorrection');
        expect(REPROJECT_WGSL).toContain('rcFrame.capsaicinEmptyCells');
        expect(SAMPLE_WGSL).toContain('rcFrame.rayGuidingUniformShare');
    });

    it('start the screen history over when a history rule changes, and every frame with history off', () => {
        const technique = new RadianceCacheTechnique();
        const reset = vi.spyOn(technique, 'resetScreenHistory');
        const nextFrame = (): void => (technique as unknown as { resetHistoryForSettings(): void }).resetHistoryForSettings();
        nextFrame();
        nextFrame();
        expect(reset).not.toHaveBeenCalled();
        technique.settings.historySampleCap = 3;
        technique.settings.rayGuidingUniformShare = 0;
        nextFrame();
        expect(reset).not.toHaveBeenCalled();
        technique.settings.probeBlend = 'capsaicin-hysteresis';
        nextFrame();
        expect(reset).toHaveBeenCalledTimes(1);
        nextFrame();
        expect(reset).toHaveBeenCalledTimes(1);
        technique.settings.probeHistory = false;
        nextFrame();
        nextFrame();
        expect(reset).toHaveBeenCalledTimes(3);
    });

    it('read a GPU page configuration from text, by setting or link name, and refuse a typo', () => {
        expect(compareValuesFromText('probeBlend=capsaicin-hysteresis,rccap=1')).toEqual({ probeBlend: 'capsaicin-hysteresis', historySampleCap: 1 });
        expect(compareValuesFromText('probeChangeReset=false')).toEqual({ probeChangeReset: false });
        expect(compareValuesFromText('capsaicin')).toEqual(capsaicinCompareValues());
        expect(compareValuesFromText(undefined)).toEqual({});
        expect(() => compareValuesFromText('probeBlnd=capsaicin-hysteresis')).toThrow(/no compare toggle/);
        expect(() => compareValuesFromText('emptyCellFill=dark')).toThrow(/not a value/);
    });
});
