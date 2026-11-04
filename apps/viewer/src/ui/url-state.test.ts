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
        expect(state.ev).toBe(6);
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

    it('round-trips a state and keeps the default link empty', () => {
        expect(serializeUrlState(DEFAULT_URL_STATE)).toBe('');
        const state = parseUrlState('?scene=sponza&camera=1&wipe=lpg,vxgi&split=0.4');
        expect(parseUrlState(serializeUrlState(state))).toEqual(state);
    });

    it('prefers a free camera over a bookmark in the link', () => {
        const state = { ...DEFAULT_URL_STATE, scene: 'cornell-box', camera: '2', freeCamera: [1, 2, 3, 4, 5, 6, 60] as [number, number, number, number, number, number, number] };
        const link = serializeUrlState(state);
        expect(link).toContain('cam=1%2C2%2C3%2C4%2C5%2C6%2C60');
        expect(link).not.toContain('camera=');
    });
});
