import { describe, expect, it } from 'vitest';
import { DEFAULT_URL_STATE, parseUrlState, serializeUrlState } from './url-state';

describe('url state', () => {
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

    it("reads DDGI's debug views and round-trips them", () => {
        const state = parseUrlState('?probes=1&probecolor=history&atlas=irradiance&atlasfill=1&atlasev=2.5&change=0.25&gi=only');
        expect(state).toMatchObject({ probes: true, probeColor: 'history', atlas: 'irradiance', atlasFill: true, atlasEv: 2.5, change: 0.25, gi: 'only' });
        expect(parseUrlState(serializeUrlState(state))).toEqual(state);
        // Only the offered scales; anything else is off. The atlas exposure is clamped to the slider.
        const nonsense = parseUrlState('?probecolor=rainbow&atlas=normals&atlasfill=yes&atlasev=99&change=3');
        expect(nonsense).toMatchObject({ probeColor: 'state', atlas: null, atlasFill: false, atlasEv: 10, change: null });
        // The atlas exposure lands on the slider's half-stop steps.
        expect(parseUrlState('?atlasev=1.3').atlasEv).toBe(1.5);
        expect(parseUrlState('?atlasev=-2.2').atlasEv).toBe(-2);
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
});
