/** What 05 shows and how it works: the page's explainer panel (shared/explainer.ts). Plain data; main.ts holds the example. */

import { API_PAGES, docsLink, showExplainer } from '../shared/explainer.ts';

const SUN_LAYER = docsLink('api/three.md#the-sun-layer', 'three.md, "The sun layer"');

showExplainer({
    seeing: [
        'The sun crosses the sky over a walled courtyard, in a 30-second day. The sky and the sun light move together, and GI follows both.',
        'Near the horizon the sun turns warm and dim, and the bounce light turns with it.',
    ],
    how: [
        "Each frame moves three things: three's `DirectionalLight` (the sun), three's `SkyMesh`, and GI, which reads both. GI makes its own copy of the sky again whenever the sun has turned a little ([three.md](https://github.com/pmndrs/gi/blob/main/docs/api/three.md#time-of-day)).",
        'With `sunLayer: true` the sun has a layer of its own in GI. A change of its colour or brightness then shows in the same frame, with nothing traced again.',
        'A change of its direction moves the sunlit patches, so the probes are traced again, nearest the camera first.',
    ],
    tryThis: [
        'Untick `play` and drag `hour` towards 6: the sun turns warm and dim, and the bounce light follows in the same frame.',
        'Drag `hour` from 8 to 16: the sunlit patches move, and the bounce light under the walls follows over a few frames.',
        'Look at the shaded side of the column: its light is all sky and bounce light.',
    ],
    tradeOffs: {
        pros: [{ text: "The sun layer shows the sun's colour and brightness exactly, in the frame they change.", source: SUN_LAYER }],
        cons: [{ text: 'It costs a layer of the field: on Sponza, 58 MB of GI becomes 93 MB.', source: SUN_LAYER }],
    },
    api: API_PAGES.three,
    more: [docsLink('api/three.md#time-of-day', 'Time of day')],
});
