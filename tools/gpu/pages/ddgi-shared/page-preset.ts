/**
 * A quality preset for a DDGI page run (T5.1, docs/design/m5-coverage.md §1):
 * `VITE_DDGI_PRESET=low pnpm test:gpu --filter ddgi-furnace` runs the page
 * with the preset's DDGI settings (rays, tile sizes) and its texture store
 * size. Pages that set their own budget in units (every probe every frame)
 * and the manifest's spacing keep both: the preset's budget and default
 * spacing are for hosts that don't choose them, and these pages do. For the
 * same reason they keep the default batch cap (`maxProbesPerFrame` 4,096):
 * `low`'s 1,024 (TU.1) would cap "every probe every frame" on two-room's
 * 1,071 probes. `ddgi-low-warmup` is the page for the preset's own batch.
 *
 * Not a page itself (no index.html).
 */

import { ddgiPreset, isDDGIPresetName, type DDGIPreset } from '../../../../packages/core/src/techniques/ddgi/presets.ts';
import { DEFAULT_DDGI_SETTINGS } from '../../../../packages/core/src/techniques/ddgi/settings.ts';

const env = import.meta.env as Record<string, unknown>;

/** The preset `VITE_DDGI_PRESET` names, or null (the defaults, which are `high`). An unknown name throws, so a typo can't pass as the defaults. */
export function pagePreset(): DDGIPreset | null {
    const name = env.VITE_DDGI_PRESET;
    if (name === undefined || name === '') return null;
    if (!isDDGIPresetName(name)) throw new Error(`VITE_DDGI_PRESET is "${String(name)}"; use low, medium or high`);
    return everyProbeEveryFrame(ddgiPreset(name));
}

/** The preset with the default batch cap, for pages that update every probe every frame (the file comment). */
export function everyProbeEveryFrame(preset: DDGIPreset): DDGIPreset {
    return { ...preset, ddgi: { ...preset.ddgi, maxProbesPerFrame: DEFAULT_DDGI_SETTINGS.maxProbesPerFrame } };
}

/** What a page reports about the preset it ran with. */
export function presetReport(preset: DDGIPreset | null): { readonly preset: string; readonly raysPerProbe: number; readonly irradianceTexels: number; readonly distanceTexels: number; readonly textureLayerSize: number } {
    const resolved = preset ?? ddgiPreset('high');
    return { preset: preset?.name ?? 'none (defaults)', ...resolved.ddgi, textureLayerSize: resolved.textureLayerSize };
}
