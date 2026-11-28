/** What 08 shows and how it works: the page's explainer panel (shared/explainer.ts). Plain data; main.ts holds the example. */

import { API_PAGES, docsLink, showExplainer } from '../shared/explainer.ts';

const BAKES = docsLink('api/three.md#ddgivolume', 'three.md, "Saved bakes"');

showExplainer({
    seeing: ['The two-room scene. The first visit warms the probes up and saves them in this browser; a reload starts from the saved probes, with no warm-up.'],
    how: [
        "`volume.saveBake()` copies the probes' light and histories out in one GPU readback: a `.ddgibake` file's bytes. This page keeps them in IndexedDB.",
        '`volume.start(renderer, scene, { bake })` loads them and skips the warm-up passes. It still waits for the ray-tracing trees and calibrates the budget.',
        'A bake records a fingerprint of its scene. If the scene changed since, it still loads, as a starting point, and every probe updates from it. A bake for another probe layout is refused with `SavedFieldError`.',
    ],
    tryThis: [
        'Wait for the line above to say a bake was saved, then press `reload the page`: GI is there at once.',
        'Compare the start times the line gives on the two visits.',
        'Press `forget the bake and reload` to warm up and save again.',
    ],
    tradeOffs: {
        pros: [{ text: 'A page opens with finished GI, and keeps updating from there.', source: BAKES }],
        cons: [{ text: 'A bake loads only into the same layout: the same probe grid and place, tile sizes, light groups and sun layer.', source: BAKES }],
    },
    api: API_PAGES.three,
    more: [docsLink('api/platform.md#saved-bakes-ddgi-t52', 'The .ddgibake file')],
});
