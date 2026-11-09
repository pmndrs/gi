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
 * - `probecolor` `history` colours DDGI's probe spheres by how settled they are (default: by state)
 * - `atlas`      `irradiance` or `distance`: that probe atlas drawn over the picture
 * - `atlasfill`  `1`: the atlas fills the view instead of the bottom-right corner
 * - `atlasev`    extra exposure for the irradiance atlas, in stops
 * - `change`     DDGI's change-per-frame view, full colour at this many percent (`0.25`, `1` or `4`)
 * - `groups`     DDGI light groups with a layer each: `2` or `4` (absent: off)
 * - `tod`        time of day on, at `hour,day` (sun time 0 to 24, day of the year 1 to 365); it also
 *                turns DDGI's sun layer on
 * - `motion`     a light motion of the scene (`light-sweep`, `light-step`) playing
 * - `reach`      metres of reach for many shadowless lights with no range (`sponza-candles`' candles),
 *                so three can cull them; `0` keeps them unranged (exact and very slow). Read when the
 *                scene loads (render/scene-lighting.ts)
 * - anything else is kept as a scene setting by name, like `candles=4096` for `sponza-candles`
 *                (the manifest's `settings`; read when the scene loads, ignored by scenes without it)
 *
 * No engine imports, so it is unit-tested.
 */

import { MAX_DEFAULT_EXPOSURE_EV } from '../../../../scenes/src/manifest';
import type { GiTechniqueId, TechniqueId, UpscaleQuality } from '../techniques/types';

/** Exposure range in stops, for links and the slider: the same range a manifest's default may use. */
export const MAX_EXPOSURE_EV = MAX_DEFAULT_EXPOSURE_EV;

export type GiChoice = 'off' | 'on' | 'only';

/** DDGI's screen debug views by their link names (`techniques/ddgi-technique.ts`; `gi-albedo` is the viewer's own). */
export type DdgiScreenViewName = 'gi-only' | 'coverage' | 'probe-states' | 'gi-albedo';
export const DDGI_SCREEN_VIEW_NAMES: readonly DdgiScreenViewName[] = ['gi-only', 'coverage', 'probe-states', 'gi-albedo'];

export type ProbeColorName = 'state' | 'history';
export type AtlasName = 'irradiance' | 'distance';
const ATLAS_NAMES: readonly AtlasName[] = ['irradiance', 'distance'];

/**
 * DDGI's change-per-frame view (`ddgi/frame-change.ts`): full colour at 0.25%, 1% or 4% change
 * per frame. Settled Cornell changes about 0.05-0.09% a frame on average and 0.16-0.26% at the
 * 95th percentile (T1.12 phase 2b, RTX), so 0.25% shows its faintest flicker.
 */
export const CHANGE_FULL_SCALE_PERCENTS = [0.25, 1, 4] as const;
export type ChangeFullScalePercent = (typeof CHANGE_FULL_SCALE_PERCENTS)[number];

/** The irradiance atlas's own exposure range, in stops (the slider's ends), and the slider's step. */
export const MAX_ATLAS_EV = 10;
export const ATLAS_EV_STEP = 0.5;

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
    probeColor: ProbeColorName;
    debug: DdgiScreenViewName | null;
    atlas: AtlasName | null;
    atlasFill: boolean;
    atlasEv: number;
    /** The change-per-frame view's full-colour change, in percent; null is off. */
    change: ChangeFullScalePercent | null;
    /** GI's rays read material textures (`gitextures=0` in a link turns them off; `textures` is T3.12's compression choice). */
    giTextures: boolean;
    /** DDGI's light groups: 0 (off), 2 or 4. */
    lightGroups: 0 | 2 | 4;
    /** Time of day, when on: the hour (sun time) and the day of the year. */
    timeOfDay: { hour: number; dayOfYear: number } | null;
    /** The light motion playing (a manifest motion id), or null. */
    motion: string | null;
    /** Reach in metres for many unranged shadowless lights (render/scene-lighting.ts); null is the viewer's default. */
    lightReach: number | null;
    /** Scene settings by name (`candles=4096`), as written; the scene's manifest says which it reads. */
    sceneSettings: Record<string, string>;
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
    probeColor: 'state',
    debug: null,
    atlas: null,
    atlasFill: false,
    atlasEv: 0,
    change: null,
    giTextures: true,
    lightGroups: 0,
    timeOfDay: null,
    motion: null,
    lightReach: null,
    sceneSettings: {},
};

/** The link's own names; any other name is a scene setting. */
const URL_KEYS = new Set(['scene', 'camera', 'cam', 'technique', 'gi', 'wipe', 'split', 'ev', 'tone', 'nav', 'upscale', 'budget', 'probes', 'debug', 'probecolor', 'atlas', 'atlasfill', 'atlasev', 'change', 'gitextures', 'groups', 'tod', 'motion', 'reach']);

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
    const state: ViewerUrlState = { ...DEFAULT_URL_STATE, sceneSettings: {} };
    for (const [name, value] of params) if (!URL_KEYS.has(name)) state.sceneSettings[name] = value;

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
    if (params.get('probecolor') === 'history') state.probeColor = 'history';
    state.atlas = ATLAS_NAMES.find((name) => name === params.get('atlas')) ?? null;
    state.atlasFill = params.get('atlasfill') === '1';
    const atlasEv = finiteNumber(params.get('atlasev'));
    // On the slider's steps, so the link and the slider agree.
    if (atlasEv !== null) state.atlasEv = Math.min(MAX_ATLAS_EV, Math.max(-MAX_ATLAS_EV, Math.round(atlasEv / ATLAS_EV_STEP) * ATLAS_EV_STEP));
    const change = finiteNumber(params.get('change'));
    state.change = CHANGE_FULL_SCALE_PERCENTS.find((percent) => percent === change) ?? null;
    state.giTextures = params.get('gitextures') !== '0';
    const groups = finiteNumber(params.get('groups'));
    if (groups === 2 || groups === 4) state.lightGroups = groups;
    const timeOfDay = params.get('tod')?.split(',').map(Number);
    if (timeOfDay && timeOfDay.length === 2 && timeOfDay.every(Number.isFinite)) {
        state.timeOfDay = { hour: Math.min(24, Math.max(0, timeOfDay[0])), dayOfYear: Math.min(365, Math.max(1, Math.round(timeOfDay[1]))) };
    }
    const motion = params.get('motion');
    if (motion && /^[a-z0-9-]+$/.test(motion)) state.motion = motion;
    const reach = finiteNumber(params.get('reach'));
    if (reach !== null && reach >= 0) state.lightReach = reach;
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
    if (state.probeColor !== DEFAULT_URL_STATE.probeColor) params.set('probecolor', state.probeColor);
    if (state.atlas) params.set('atlas', state.atlas);
    if (state.atlasFill) params.set('atlasfill', '1');
    if (state.atlasEv !== DEFAULT_URL_STATE.atlasEv) params.set('atlasev', String(state.atlasEv));
    if (state.change !== null) params.set('change', String(state.change));
    if (!state.giTextures) params.set('gitextures', '0');
    if (state.lightGroups !== 0) params.set('groups', String(state.lightGroups));
    if (state.timeOfDay) params.set('tod', `${state.timeOfDay.hour.toFixed(2)},${state.timeOfDay.dayOfYear}`);
    if (state.motion) params.set('motion', state.motion);
    if (state.lightReach !== null) params.set('reach', String(state.lightReach));
    for (const [name, value] of Object.entries(state.sceneSettings)) if (!URL_KEYS.has(name)) params.set(name, value);
    return params.toString();
}
