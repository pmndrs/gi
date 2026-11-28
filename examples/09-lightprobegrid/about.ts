/** What 09 shows and how it works: the page's explainer panel (shared/explainer.ts). Plain data; main.ts holds the example. */

import { API_PAGES, docsLink, showExplainer } from '../shared/explainer.ts';

const EXPORT = docsLink('api/three.md#exporting-a-lightprobegrid', 'three.md, "Exporting a LightProbeGrid"');

showExplainer({
    seeing: [
        "GI exported as three's own `LightProbeGrid`, a static fallback for where DDGI can't run.",
        'Left of the line: live GI. Right: the exported grid. Move the mouse to wipe.',
    ],
    how: [
        '`volume.exportLightProbeGrid()` turns every probe into spherical harmonics on the GPU, and builds the grid from them. Its probes sit exactly where DDGI\'s do.',
        "Probes live GI doesn't read (inside walls, not updated yet) take the mean of their nearest active neighbours, because a grid without visibility blends every probe.",
        'The grid is a snapshot: export again after the light changes.',
    ],
    tryThis: [
        'Move the mouse across the picture to wipe between live GI and the grid.',
        'Look where walls meet: the grid has no distances, so light leaks between its probes more easily.',
        'Press `export again` after a change, or `download the JSON` to keep the grid.',
    ],
    tradeOffs: {
        pros: [{ text: "On the point-light Cornell box, the grid's mean is within 1% of live GI's at every view (2–4% per pixel).", source: EXPORT }],
        cons: [
            { text: 'It loses detail above L2, visibility and relocation. On Sponza it reads 8% darker at the colonnade.', source: EXPORT },
            { text: "It is a snapshot: it doesn't follow changes in the scene.", source: EXPORT },
        ],
    },
    api: API_PAGES.three,
});
