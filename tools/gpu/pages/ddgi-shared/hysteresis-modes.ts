/**
 * The hysteresis rules the T1.12 measurement pages compare (ddgi-flicker,
 * ddgi-settle), named in an environment list:
 *
 *   fixed:0.97   every update keeps 0.97 of the old value (the rule before T1.12)
 *   tau:5        adaptive hysteresis with a 5 s memory time (hysteresisMemorySeconds)
 *   tau:5/200    the same with n_max = 200 (hysteresisMaxSamples)
 *   default      the technique's defaults
 *
 * Not a page itself (no index.html).
 */

import { DEFAULT_DDGI_SETTINGS, type DDGISettings } from '../../../../packages/ddgi-core/src/techniques/ddgi/settings.ts';

export type HysteresisSettings = Pick<DDGISettings, 'adaptiveHysteresis' | 'hysteresis' | 'hysteresisMemorySeconds' | 'hysteresisMaxSamples'>;

export interface HysteresisMode {
    /** As written in the list, for reports. */
    readonly name: string;
    readonly settings: HysteresisSettings;
}

const DEFAULTS: HysteresisSettings = {
    adaptiveHysteresis: DEFAULT_DDGI_SETTINGS.adaptiveHysteresis,
    hysteresis: DEFAULT_DDGI_SETTINGS.hysteresis,
    hysteresisMemorySeconds: DEFAULT_DDGI_SETTINGS.hysteresisMemorySeconds,
    hysteresisMaxSamples: DEFAULT_DDGI_SETTINGS.hysteresisMaxSamples,
};

/** The rule before T1.12, for converging a field fast and as the baseline. */
export const FIXED_097: HysteresisMode = { name: 'fixed:0.97', settings: { ...DEFAULTS, adaptiveHysteresis: false, hysteresis: 0.97 } };

export function parseHysteresisMode(text: string): HysteresisMode {
    const name = text.trim();
    if (name === 'default') return { name, settings: { ...DEFAULTS } };
    const [kind, value] = name.split(':');
    if (kind === 'fixed') {
        const hysteresis = Number(value);
        if (!(hysteresis >= 0 && hysteresis <= 1)) throw new Error(`bad hysteresis mode '${name}': fixed:<0..1>`);
        return { name, settings: { ...DEFAULTS, adaptiveHysteresis: false, hysteresis } };
    }
    if (kind === 'tau') {
        const [seconds, maxSamples] = (value ?? '').split('/').map(Number);
        if (!(seconds > 0)) throw new Error(`bad hysteresis mode '${name}': tau:<seconds>[/<n_max>]`);
        return {
            name,
            settings: { ...DEFAULTS, adaptiveHysteresis: true, hysteresisMemorySeconds: seconds, hysteresisMaxSamples: maxSamples > 0 ? maxSamples : DEFAULTS.hysteresisMaxSamples },
        };
    }
    throw new Error(`bad hysteresis mode '${name}': use fixed:<h>, tau:<seconds>[/<n_max>] or default`);
}

export function parseHysteresisModes(value: unknown, fallback: readonly string[]): HysteresisMode[] {
    const list = typeof value === 'string' && value.trim() !== '' ? value.split(',') : [...fallback];
    return list.map(parseHysteresisMode);
}

/** Puts a mode's settings on the live technique settings (the next encode reads them). */
export function applyHysteresisMode(settings: DDGISettings, mode: HysteresisMode): void {
    Object.assign(settings, mode.settings);
}
