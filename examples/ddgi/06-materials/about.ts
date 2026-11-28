/** What 06 shows and how it works: the page's explainer panel (shared/explainer.ts). Plain data; main.ts holds the example. */

import { API_PAGES, docsLink, showExplainer } from '../../shared/explainer.ts';

const MATERIALS = docsLink('api/platform.md#materials-and-textures', 'platform.md, "Materials and textures"');

showExplainer({
    seeing: [
        "GI's rays see each material as three draws it. Pick a scene in the panel: a wall of cut-out holes, a glass pane, or eight textured panels in the sun.",
        "In the first two, the dark room's light comes only through the holes or the glass.",
    ],
    how: [
        "The adapter copies three's ordinary materials into GI: colour and textures, metalness, opacity and alpha cut-outs, and see-through glass. Nothing extra is set.",
        "Where a ray hits, it sees the material's colour times its texture there, less what is metal, its emission, and its alpha as the material says.",
        "GI keeps its own copy of each texture, with mips ([platform.md](https://github.com/pmndrs/gi/blob/main/docs/api/platform.md#materials-and-textures)). Its rays mostly read the coarse mips.",
    ],
    tryThis: [
        "In `alpha-checker`, look into the dark room: its light comes through the checkerboard of holes, half of the wall open.",
        'Switch `scene` to `glass-wall`: light reaches the dark room only through the pane.',
        "Switch to `uv-board`: each panel's colours bounce onto the ground as its texture mapping says.",
    ],
    tradeOffs: {
        pros: [{ text: "three's own materials, with no extra set-up.", source: docsLink('api/three.md#what-is-mirrored', 'three.md, "What is mirrored"') }],
        cons: [
            { text: "Each texture takes about 0.35 MB in GI's store at 256².", source: MATERIALS },
            { text: "Compressed textures need the format's device feature; without it those materials keep their flat colour in GI.", source: MATERIALS },
        ],
    },
    api: API_PAGES.three,
    more: [MATERIALS],
});
