/**
 * The viewer's state as a link, so a URL reproduces a view:
 *
 *   ?scene=sponza&camera=2&technique=vxgi&gi=on&wipe=lpg,vxgi&split=0.4&ev=0.5&tone=agx
 *
 * - `scene`      manifest id
 * - `camera`     bookmark number (1-based) or name; `cam=px,py,pz,tx,ty,tz,fov` is a free camera that overrides it
 * - `technique`  `ddgi` (default), `lpg` or `vxgi` (the GI technique shown when there is no wipe)
 * - `gi`         `off` (direct light only), `on` (default), or `only` (the indirect light alone)
 * - `wipe`       two techniques, `left,right`, each `none`, `ddgi`, `lpg`, `vxgi` or `cycles` (the
 *                reference image); turns the split screen on
 * - `split`      where the wipe line starts, 0 to 1
 * - `ev`, `tone` exposure in stops and the tone mapper; without `ev` the scene's own default (manifest `exposureEv`)
 * - `nav`        `orbit` or `fly`
 * - `upscale`    an upscaler quality preset; absent means off
 * - `budget`     DDGI's GPU milliseconds per frame for probe updates (default 2)
 * - `probes`     `1` draws DDGI's probes as spheres coloured by state
 * - `debug`      one of DDGI's screen debug views: `gi-only`, `coverage`, `probe-states`
 *
 * No engine imports, so it is unit-tested.
 */

import { MAX_DEFAULT_EXPOSURE_EV } from '../../../../scenes/src/manifest';
import type { GiTechniqueId, TechniqueId, UpscaleQuality } from '../techniques/types';

/** Exposure range in stops, for links and the slider: the same range a manifest's default may use. */
export const MAX_EXPOSURE_EV = MAX_DEFAULT_EXPOSURE_EV;

export type GiChoice = 'off' | 'on' | 'only';

/** DDGI's screen debug views by their link names (ids in `techniques/ddgi-technique.ts`). */
export type DdgiScreenViewName = 'gi-only' | 'coverage' | 'probe-states';
export const DDGI_SCREEN_VIEW_NAMES: readonly DdgiScreenViewName[] = ['gi-only', 'coverage', 'probe-states'];

/** DDGI's per-frame budget range, in GPU milliseconds (the slider's ends). */
export const MIN_BUDGET_MS = 0.25;
export const MAX_BUDGET_MS = 8;
export const DEFAULT_BUDGET_MS = 2;

export interface ViewerUrlState {
    scene: string | null;
    camera: string | null;
    freeCamera: [number, number, number, number, number, number, number] | null;
    /** The GI technique (never `none`: "off" is expressed by `gi`; never the reference image). */
    technique: GiTechniqueId;
    gi: GiChoice;
    wipe: [TechniqueId, TechniqueId] | null;
    split: number;
    /** Exposure in stops; null means the scene's own default (the manifest's `exposureEv`). */
    ev: number | null;
    tone: string;
    nav: 'orbit' | 'fly';
    upscale: UpscaleQuality | null;
    budgetMs: number;
    probes: boolean;
    debug: DdgiScreenViewName | null;
}

export const DEFAULT_URL_STATE: ViewerUrlState = {
    scene: null,
    camera: null,
    freeCamera: null,
    technique: 'ddgi',
    gi: 'on',
    wipe: null,
    split: 0.5,
    ev: null,
    tone: 'aces',
    nav: 'orbit',
    upscale: null,
    budgetMs: DEFAULT_BUDGET_MS,
    probes: false,
    debug: null,
};

const TECHNIQUES: readonly TechniqueId[] = ['none', 'ddgi', 'lpg', 'vxgi', 'cycles'];
const GI_TECHNIQUES: readonly GiTechniqueId[] = ['ddgi', 'lpg', 'vxgi'];
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

    const technique = GI_TECHNIQUES.find((id) => id === params.get('technique'));
    if (technique) state.technique = technique;

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
    if (ev !== null) state.ev = Math.min(MAX_EXPOSURE_EV, Math.max(-MAX_EXPOSURE_EV, ev));

    state.tone = params.get('tone') ?? state.tone;
    if (params.get('nav') === 'fly') state.nav = 'fly';
    state.upscale = UPSCALE_QUALITIES.find((quality) => quality === params.get('upscale')) ?? null;

    const budget = finiteNumber(params.get('budget'));
    if (budget !== null) state.budgetMs = Math.min(MAX_BUDGET_MS, Math.max(MIN_BUDGET_MS, budget));
    state.probes = params.get('probes') === '1';
    state.debug = DDGI_SCREEN_VIEW_NAMES.find((name) => name === params.get('debug')) ?? null;
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
    if (state.ev !== null) params.set('ev', String(state.ev));
    if (state.tone !== DEFAULT_URL_STATE.tone) params.set('tone', state.tone);
    if (state.nav !== 'orbit') params.set('nav', state.nav);
    if (state.upscale) params.set('upscale', state.upscale);
    if (state.budgetMs !== DEFAULT_BUDGET_MS) params.set('budget', String(state.budgetMs));
    if (state.probes) params.set('probes', '1');
    if (state.debug) params.set('debug', state.debug);
    return params.toString();
}
