/** What 03 shows and how it works: the page's explainer panel (shared/explainer.ts). Plain data; main.ts holds the example. */

import { API_PAGES, docsLink, showExplainer } from '../shared/explainer.ts';

const LIGHT_GROUPS = docsLink('api/three.md#light-groups', 'three.md, "Light groups"');

showExplainer({
    seeing: [
        'The Cornell box with one point lamp. The lamp sweeps across the box, and the panel on the right moves, dims, recolours or switches it.',
        'GI follows within a few frames, and the room never goes dark while it does.',
    ],
    how: [
        "Nothing is called when the light changes. The adapter checks the scene's lights every frame, so the next render sees the lamp's new position, colour and intensity.",
        "DDGI works out how much each probe's light changed, and owes those probes updates. They are traced again, nearest the camera first.",
        'Until a probe is traced again it keeps its old light, so nothing goes black in between.',
        'For a change that must show in the same frame, such as dimming a whole group of lights, use light groups.',
    ],
    tryThis: [
        'Untick `sweep across the box`, then untick `on`: the coloured bounce light on the boxes fades over a few frames.',
        'Tick `on` again and set `color` to blue: the bounce light turns blue as the probes catch up.',
        'Drag `intensity` up and down quickly: GI follows the direct light a few frames behind.',
    ],
    tradeOffs: {
        pros: [{ text: 'Light groups show a weight change in the same frame, with no lag and no noise.', source: LIGHT_GROUPS }],
        cons: [
            { text: "A changed light re-traces the probes it reaches, so GI follows over frames, not at once.", source: docsLink('design/ddgi.md#change-driven-history-m2-t24a-m2-lightsmd-41-42-as-built', 'design/ddgi.md, "Change-driven history"') },
            { text: 'Each light group costs a layer of the field: on Sponza about 25 MB and 5% of each probe update.', source: LIGHT_GROUPS },
        ],
    },
    api: API_PAGES.three,
});
