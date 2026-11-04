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
        const state = parseUrlState('?technique=banana&gi=maybe&wipe=lpg,lpg&split=abc&ev=999&upscale=ludicrous&cam=1,2,3');
        expect(state.technique).toBe('lpg');
        expect(state.gi).toBe('on');
        expect(state.wipe).toBeNull();
        expect(state.split).toBe(0.5);
        expect(state.ev).toBe(6);
        expect(state.upscale).toBeNull();
        expect(state.freeCamera).toBeNull();
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
