import { describe, expect, it } from 'vitest';
import { DEFAULT_URL_STATE, parseUrlState, serializeUrlState } from './url-state';
import { capsaicinCompareValues, RADIANCE_CACHE_COMPARE_TOGGLES } from '../../../../packages/ddgi-core/src/techniques/radiance-cache/compare-toggles';

describe('url state', () => {
    it("keeps each radiance-cache compare toggle (RC4.5) through a link, and doesn't take it for a scene setting", () => {
        for (const entry of RADIANCE_CACHE_COMPARE_TOGGLES) {
            if (entry.capsaicin === undefined || entry.capsaicin === entry.defaultValue) continue;
            const state = { ...DEFAULT_URL_STATE, technique: 'rc' as const, rcCompare: { [entry.setting]: entry.capsaicin } };
            const link = serializeUrlState(state);
            expect(link).toContain(`${entry.urlName}=`);
            const read = parseUrlState(`?${link}`);
            expect(read.rcCompare).toEqual(state.rcCompare);
            expect(read.sceneSettings).toEqual({});
        }
        const all = { ...DEFAULT_URL_STATE, rcCompare: capsaicinCompareValues() };
        expect(parseUrlState(`?${serializeUrlState(all)}`).rcCompare).toEqual(all.rcCompare);
        // Ours is the default: nothing written.
        expect(serializeUrlState({ ...DEFAULT_URL_STATE, rcCompare: { probeBlend: 'running-mean' } })).toBe('');
    });

    it('reads the documented parameters', () => {
        const state = parseUrlState('?scene=sponza&camera=2&technique=vxgi&gi=only&wipe=lpg,vxgi&split=0.25&ev=-1.5&tone=agx&nav=fly&upscale=quality');
        expect(state).toMatchObject({
            scene: 'sponza',
            camera: '2',
            technique: 'vxgi',
            gi: 'only',
            wipe: ['lpg', 'vxgi'],
            split: 0.25,
            ev: -1.5,
            tone: 'agx',
            nav: 'fly',
            upscale: 'quality',
        });
    });

    it('reads the quality preset, whose budget is the default budget', () => {
        expect(parseUrlState('').preset).toBe('high');
        const low = parseUrlState('?preset=low');
        expect(low).toMatchObject({ preset: 'low', budgetMs: 1 });
        expect(serializeUrlState(low)).toBe('preset=low');
        // A budget of its own wins, and is written only when it isn't the preset's.
        const lowFaster = parseUrlState('?preset=low&budget=3');
        expect(lowFaster.budgetMs).toBe(3);
        expect(parseUrlState(serializeUrlState(lowFaster))).toEqual(lowFaster);
        expect(serializeUrlState({ ...DEFAULT_URL_STATE, budgetMs: 2 })).toBe('');
        expect(parseUrlState('?preset=ultra').preset).toBe('high');
    });

    it('falls back to defaults for nonsense', () => {
        const state = parseUrlState('?technique=banana&gi=maybe&wipe=lpg,lpg&split=abc&ev=999&upscale=ludicrous&cam=1,2,3&budget=-4&debug=nope&probes=yes');
        expect(state.technique).toBe('ddgi');
        expect(state.gi).toBe('on');
        expect(state.wipe).toBeNull();
        expect(state.split).toBe(0.5);
        expect(state.ev).toBe(8);
        expect(state.upscale).toBeNull();
        expect(state.freeCamera).toBeNull();
        expect(state.budgetMs).toBe(0.25);
        expect(state.debug).toBeNull();
        expect(state.probes).toBe(false);
    });

    it('reads the DDGI settings and the reference in a wipe, and leaves the exposure to the scene unless given', () => {
        const state = parseUrlState('?technique=ddgi&wipe=ddgi,cycles&budget=1.5&probes=1&debug=coverage');
        expect(state).toMatchObject({ technique: 'ddgi', wipe: ['ddgi', 'cycles'], budgetMs: 1.5, probes: true, debug: 'coverage', ev: null });
        // The reference image is never the technique itself, only one side of a wipe.
        expect(parseUrlState('?technique=cycles').technique).toBe('ddgi');
        expect(parseUrlState(serializeUrlState(state))).toEqual(state);
        expect(serializeUrlState({ ...DEFAULT_URL_STATE, ev: 0 })).toBe('ev=0');
    });

    it('reads warmup=0 (DDGI starts without its warm-up) and round-trips it', () => {
        expect(parseUrlState('').warmUp).toBe(true);
        const state = parseUrlState('?scene=sponza&warmup=0');
        expect(state.warmUp).toBe(false);
        expect(serializeUrlState(state)).toBe('scene=sponza&warmup=0');
        expect(parseUrlState(serializeUrlState(state))).toEqual(state);
    });

    it("reads DDGI's debug views and round-trips them", () => {
        const state = parseUrlState('?probes=1&probecolor=history&atlas=irradiance&atlasfill=1&atlasev=2.5&change=0.25&gi=only');
        expect(state).toMatchObject({ probes: true, probeColor: 'history', atlas: 'irradiance', atlasFill: true, atlasEv: 2.5, change: 0.25, gi: 'only' });
        expect(parseUrlState(serializeUrlState(state))).toEqual(state);
        // The screen-input views (RC1.1) are debug views too.
        expect(parseUrlState('?debug=screen-instance').debug).toBe('screen-instance');
        // Only the offered scales; anything else is off. The atlas exposure is clamped to the slider.
        const nonsense = parseUrlState('?probecolor=rainbow&atlas=normals&atlasfill=yes&atlasev=99&change=3');
        expect(nonsense).toMatchObject({ probeColor: 'state', atlas: null, atlasFill: false, atlasEv: 10, change: null });
        // The atlas exposure lands on the slider's half-stop steps.
        expect(parseUrlState('?atlasev=1.3').atlasEv).toBe(1.5);
        expect(parseUrlState('?atlasev=-2.2').atlasEv).toBe(-2);
    });

    it("reads the GI albedo view and the textures switch (T4.4), and round-trips them", () => {
        const state = parseUrlState('?debug=gi-albedo&gitextures=0');
        expect(state).toMatchObject({ debug: 'gi-albedo', giTextures: false });
        expect(parseUrlState(serializeUrlState(state))).toEqual(state);
        // Textures are on unless a link says 0, and on is not written.
        expect(parseUrlState('?gitextures=1').giTextures).toBe(true);
        expect(serializeUrlState({ ...DEFAULT_URL_STATE, giTextures: true })).toBe('');
    });

    it('round-trips a state and keeps the default link empty', () => {
        expect(serializeUrlState(DEFAULT_URL_STATE)).toBe('');
        const state = parseUrlState('?scene=sponza&camera=1&wipe=lpg,vxgi&split=0.4');
        expect(parseUrlState(serializeUrlState(state))).toEqual(state);
    });

    it("keeps other names as scene settings and writes them back (sponza-candles' candles=4096)", () => {
        const state = parseUrlState('?scene=sponza-candles&candles=4096&gi=only');
        expect(state.sceneSettings).toEqual({ candles: '4096' });
        expect(state.gi).toBe('only');
        expect(serializeUrlState(state)).toContain('candles=4096');
        expect(parseUrlState(serializeUrlState(state))).toEqual(state);
    });

    it('reads and writes the texture compression (T3.12): all-compressed by default since M4, the other modes by name', () => {
        expect(parseUrlState('?scene=bistro-exterior').textures).toBe('all-compressed');
        expect(parseUrlState('?textures=uncompressed').textures).toBe('uncompressed');
        expect(parseUrlState('?textures=compressed').textures).toBe('compressed');
        expect(parseUrlState('?textures=zip').textures).toBe('all-compressed');
        expect(serializeUrlState({ ...DEFAULT_URL_STATE, scene: 'sponza' })).toBe('scene=sponza');
        expect(serializeUrlState({ ...DEFAULT_URL_STATE, textures: 'uncompressed' })).toBe('textures=uncompressed');
        // Not taken for a scene setting.
        expect(parseUrlState('?textures=uncompressed').sceneSettings).toEqual({});
    });

    it('reads the light settings (T2.8): light groups, time of day, a motion, the reach of many candles', () => {
        const state = parseUrlState('?scene=sponza&groups=4&tod=14.50,172&motion=light-sweep&reach=6');
        expect(state).toMatchObject({ lightGroups: 4, timeOfDay: { hour: 14.5, dayOfYear: 172 }, motion: 'light-sweep', lightReach: 6 });
        expect(state.sceneSettings).toEqual({});
        expect(parseUrlState(serializeUrlState(state))).toEqual(state);
        const nonsense = parseUrlState('?groups=3&tod=25,400,1&motion=../x&reach=-1');
        expect(nonsense).toMatchObject({ lightGroups: 0, timeOfDay: null, motion: null, lightReach: null });
        expect(parseUrlState('?tod=30,0').timeOfDay).toEqual({ hour: 24, dayOfYear: 1 });
    });

    it('prefers a free camera over a bookmark in the link', () => {
        const state = { ...DEFAULT_URL_STATE, scene: 'cornell-box', camera: '2', freeCamera: [1, 2, 3, 4, 5, 6, 60] as [number, number, number, number, number, number, number] };
        const link = serializeUrlState(state);
        expect(link).toContain('cam=1%2C2%2C3%2C4%2C5%2C6%2C60');
        expect(link).not.toContain('camera=');
    });

    it('reads and writes the moving-geometry scenario, the camera-following volume and the wind (T3.8)', () => {
        const state = parseUrlState('?scenario=fly-through&follow=camera&wind=80&motion=fly-through');
        expect(state).toMatchObject({ scenario: 'fly-through', followCamera: true, windStrength: 0.8, motion: 'fly-through' });
        // Scenario names aren't scene settings.
        expect(state.sceneSettings).toEqual({});
        expect(parseUrlState(serializeUrlState(state))).toEqual(state);
        // Defaults stay out of the link; nonsense falls back.
        expect(serializeUrlState(DEFAULT_URL_STATE)).toBe('');
        expect(parseUrlState('?follow=yes&wind=500&scenario=Not%20One')).toMatchObject({ followCamera: false, windStrength: 1, scenario: null });
    });

    it("keeps M3's scenario keys and M4's texture keys side by side (the m34 merge)", () => {
        const state = parseUrlState('?scenario=fly-through&follow=camera&wind=30&gitextures=0&textures=uncompressed');
        expect(state).toMatchObject({ scenario: 'fly-through', followCamera: true, windStrength: 0.3, giTextures: false, textures: 'uncompressed', sceneSettings: {} });
        expect(parseUrlState(serializeUrlState(state))).toEqual(state);
        // A scenario link alone keeps M4's defaults: GI reads textures, every texture compressed.
        expect(parseUrlState('?scenario=wind-tree')).toMatchObject({ giTextures: true, textures: 'all-compressed' });
    });
});
