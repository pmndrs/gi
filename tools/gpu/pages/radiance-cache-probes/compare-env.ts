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

import { applyCompareValues, compareValuesFromText, type CompareValues } from '../../../../packages/ddgi-core/src/techniques/radiance-cache/compare-toggles.ts';
import { RadianceCacheTechnique } from '../../../../packages/ddgi-core/src/techniques/radiance-cache/radiance-cache-technique.ts';

const env = import.meta.env as Record<string, unknown>;

/** The values this run sets (empty: the defaults). */
export const PAGE_COMPARE_VALUES: CompareValues = compareValuesFromText(typeof env.VITE_RC_COMPARE === 'string' ? env.VITE_RC_COMPARE : undefined);

/** A radiance cache at the run's compare values. */
export function pageRadianceCache(): RadianceCacheTechnique {
    const technique = new RadianceCacheTechnique();
    applyCompareValues(technique.settings, PAGE_COMPARE_VALUES);
    return technique;
}
