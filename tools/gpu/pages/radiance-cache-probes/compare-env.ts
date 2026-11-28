/**
 * RC4.5: a GPU page's radiance cache at the compare toggles named in
 * `VITE_RC_COMPARE` (setting or link names, comma-separated, or `capsaicin`
 * for every Capsaicin value; compare-toggles.ts `compareValuesFromText`), so
 * a page can be run once with Capsaicin's rules back:
 *
 *   VITE_RC_COMPARE=emptyCellFill=capsaicin node tools/reference/bench-lock.ts run --no-cool-wait -- pnpm test:gpu --filter radiance-cache-temporal
 *
 * Unset, the technique runs at its defaults. A name or value the registry
 * doesn't know throws, so a typo can't run the defaults unnoticed.
 */

import { compareValuesFromText, setCompareValues, type CompareValues } from '../../../../packages/cache/src/compare-toggles.ts';
import { RadianceCacheTechnique } from '../../../../packages/cache/src/radiance-cache-technique.ts';
import type { RadianceCacheSettings } from '../../../../packages/cache/src/settings.ts';

const env = import.meta.env as Record<string, unknown>;

/** The values this run sets (empty: the defaults). */
export const PAGE_COMPARE_VALUES: CompareValues = compareValuesFromText(typeof env.VITE_RC_COMPARE === 'string' ? env.VITE_RC_COMPARE : undefined);

/** A radiance cache at `base` (the page's own settings), with the run's compare values over them. */
export function pageRadianceCache(base: Partial<RadianceCacheSettings> = {}): RadianceCacheTechnique {
    const technique = new RadianceCacheTechnique(base);
    setCompareValues(technique.settings, PAGE_COMPARE_VALUES);
    return technique;
}
