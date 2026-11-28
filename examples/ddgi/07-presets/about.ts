/** What 07 shows and how it works: the page's explainer panel (shared/explainer.ts). Plain data; main.ts holds the example. */

import { API_PAGES, docsLink, showExplainer } from '../../shared/explainer.ts';

const PRESETS = docsLink('api/platform.md#quality-presets', 'platform.md, "Quality presets"');

showExplainer({
    seeing: [
        'One word sets GI\'s quality: `low`, `medium` or `high`. Switch them in the panel and watch the probe count, the GPU time and the memory in the stats line.',
        'The page opens on the preset `recommendPreset` suggests for this GPU.',
    ],
    how: [
        "A preset sets the rays per probe, the size of each probe's tiles, the GPU budget per frame and the size of GI's copy of the textures.",
        "A volume made without probe counts, as here, also takes the preset's spacing: wider for `low`, closer for `high` ([platform.md](https://github.com/pmndrs/gi/blob/main/docs/api/platform.md#quality-presets)).",
        "The preset sizes GI's buffers when it starts, so a change means `dispose()` and a new start. Here each change makes a new volume.",
    ],
    tryThis: [
        'Switch `preset` to `low`: fewer probes, less memory and less GPU time. Light in the corners blurs a little.',
        'Switch to `high`: the most probes, and the sharpest light where walls meet.',
        'After each switch, read the stats line: probes, ms of GPU time, and MB.',
    ],
    tradeOffs: {
        pros: [{ text: 'At the scene\'s own spacing, Cornell at `low` holds 8.9 MB and updates in 0.65 ms; at `high`, 16.5 MB and 1.97 ms (M5 Pro, every probe every frame).', source: PRESETS }],
        cons: [
            { text: "Coarser distance tiles leak more: a thin wall's dark side gets 3.2% of the lit side's light at `low`, 0.6% at `high`.", source: PRESETS },
            { text: 'Fewer rays make each update noisier, so GI flickers a little more and settles a little later.', source: PRESETS },
        ],
    },
    api: API_PAGES.three,
    more: [PRESETS],
});
