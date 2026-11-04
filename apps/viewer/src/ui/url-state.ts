/**
 * The viewer's state as a link, so a URL reproduces a view:
 *
 *   ?scene=sponza&camera=2&technique=vxgi&gi=on&wipe=lpg,vxgi&split=0.4&ev=0.5&tone=agx
 *
 * - `scene`      manifest id
 * - `camera`     bookmark number (1-based) or name; `cam=px,py,pz,tx,ty,tz,fov` is a free camera that overrides it
 * - `technique`  `lpg` or `vxgi` (the GI technique shown when there is no wipe)
 * - `gi`         `off` (direct light only), `on` (default), or `only` (the indirect light alone)
 * - `wipe`       two techniques, `left,right`, each `none`, `lpg` or `vxgi`; turns the split screen on
 * - `split`      where the wipe line starts, 0 to 1
 * - `ev`, `tone` exposure in stops and the tone mapper
 * - `nav`        `orbit` or `fly`
 * - `upscale`    an upscaler quality preset; absent means off
 *
 * No engine imports, so it is unit-tested.
 */

import type { TechniqueId, UpscaleQuality } from '../techniques/types';

export type GiChoice = 'off' | 'on' | 'only';

export interface ViewerUrlState {
    scene: string | null;
    camera: string | null;
    freeCamera: [number, number, number, number, number, number, number] | null;
    /** The GI technique (never `none`: "off" is expressed by `gi`). */
    technique: Exclude<TechniqueId, 'none'>;
    gi: GiChoice;
    wipe: [TechniqueId, TechniqueId] | null;
    split: number;
    ev: number;
    tone: string;
    nav: 'orbit' | 'fly';
    upscale: UpscaleQuality | null;
}

export const DEFAULT_URL_STATE: ViewerUrlState = {
    scene: null,
    camera: null,
    freeCamera: null,
    technique: 'lpg',
    gi: 'on',
    wipe: null,
    split: 0.5,
    ev: 0,
    tone: 'aces',
    nav: 'orbit',
    upscale: null,
};

const TECHNIQUES: readonly TechniqueId[] = ['none', 'lpg', 'vxgi'];
const UPSCALE_QUALITIES: readonly UpscaleQuality[] = ['native-aa', 'quality', 'balanced', 'performance', 'ultra-performance'];

function asTechnique(value: string | undefined): TechniqueId | null {
    return TECHNIQUES.find((id) => id === value) ?? null;
}

function finiteNumber(text: string | null): number | null {
    if (text === null || text.trim() === '') return null;
    const value = Number(text);
    return Number.isFinite(value) ? value : null;
}

/** Reads a query string. Unknown or malformed values fall back to the defaults. */
export function parseUrlState(search: string): ViewerUrlState {
    const params = new URLSearchParams(search);
    const state: ViewerUrlState = { ...DEFAULT_URL_STATE };

    state.scene = params.get('scene');
    state.camera = params.get('camera');

    const free = params.get('cam')?.split(',').map(Number);
    if (free && free.length === 7 && free.every(Number.isFinite)) state.freeCamera = free as ViewerUrlState['freeCamera'];

    const technique = asTechnique(params.get('technique') ?? undefined);
    if (technique && technique !== 'none') state.technique = technique;

    const gi = params.get('gi');
    if (gi === 'off' || gi === 'on' || gi === 'only') state.gi = gi;

    const wipe = params.get('wipe')?.split(',');
    if (wipe && wipe.length === 2) {
        const left = asTechnique(wipe[0]);
        const right = asTechnique(wipe[1]);
        if (left && right && left !== right) state.wipe = [left, right];
    }

    const split = finiteNumber(params.get('split'));
    if (split !== null) state.split = Math.min(1, Math.max(0, split));

    const ev = finiteNumber(params.get('ev'));
    if (ev !== null) state.ev = Math.min(6, Math.max(-6, ev));

    state.tone = params.get('tone') ?? state.tone;
    if (params.get('nav') === 'fly') state.nav = 'fly';
    state.upscale = UPSCALE_QUALITIES.find((quality) => quality === params.get('upscale')) ?? null;
    return state;
}

/** Writes only what differs from the defaults, so links stay short. */
export function serializeUrlState(state: ViewerUrlState): string {
    const params = new URLSearchParams();
    if (state.scene) params.set('scene', state.scene);
    if (state.freeCamera) params.set('cam', state.freeCamera.join(','));
    else if (state.camera) params.set('camera', state.camera);
    if (state.technique !== DEFAULT_URL_STATE.technique) params.set('technique', state.technique);
    if (state.gi !== DEFAULT_URL_STATE.gi) params.set('gi', state.gi);
    if (state.wipe) {
        params.set('wipe', state.wipe.join(','));
        if (state.split !== DEFAULT_URL_STATE.split) params.set('split', state.split.toFixed(2));
    }
    if (state.ev !== 0) params.set('ev', String(state.ev));
    if (state.tone !== DEFAULT_URL_STATE.tone) params.set('tone', state.tone);
    if (state.nav !== 'orbit') params.set('nav', state.nav);
    if (state.upscale) params.set('upscale', state.upscale);
    return params.toString();
}
