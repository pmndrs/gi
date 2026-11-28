/** What 10 shows and how it works: the page's explainer panel (shared/explainer.ts). Plain data; main.ts holds the example. */

import { API_PAGES, docsLink, showExplainer } from '../shared/explainer.ts';

showExplainer({
    seeing: [
        'What the probes are doing, in the two-room scene. A dot marks where each probe sits, in its state\'s colour, and the bottom right shows the light the probes have gathered.',
        'The panel switches the picture to GI alone, GI\'s coverage, or the probe states each point reads.',
    ],
    how: [
        'The core hands out debug data, and the page draws it in three.js. Everything stays on the GPU: three reads the same textures DDGI writes.',
        'Probe states: green active, red inside geometry, blue with no surface nearby, yellow not updated yet. Probes inside walls move out where they can.',
        'The atlas holds each probe\'s light as a small tile, one picture per row of probes ([design/ddgi.md](https://github.com/pmndrs/gi/blob/main/docs/design/ddgi.md#the-textures)).',
    ],
    tryThis: [
        'Set `view` to `gi-only`: every surface white, lit by GI alone.',
        'Set `view` to `coverage`: green where GI covers a surface fully, red where the renderer\'s own light takes over.',
        'Look at the probes along the wall between the rooms: their colours show which ones are inside it or moved out.',
    ],
    api: API_PAGES.three,
    more: [docsLink('design/ddgi.md#debug-data-and-counters', 'Debug data and counters')],
});
