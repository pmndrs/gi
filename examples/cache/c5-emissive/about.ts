/** What C5 shows and how it works: the page's explainer panel (shared/explainer.ts). Plain data; main.ts holds the example. */

import { API_PAGES, docsLink, showExplainer } from '../../shared/explainer.ts';

const API = 'https://github.com/pmndrs/gi/blob/main/docs/api/radiance-cache.md';
const VARIANTS = docsLink('design/radiance-cache-variants.md#summary', 'radiance-cache-variants.md, "Summary"');
const KNOWN_LIMITS = docsLink('api/radiance-cache.md#known-limits', 'radiance-cache.md, "Known limits"');

/** A room in section: a probe on the floor, one ray to the ceiling strip, one to the screen, one to a plain wall. */
const DIAGRAM = `<svg viewBox="0 0 320 150" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="A probe on the floor sends rays. Rays that hit the light strip or the screen take their emission and stop. A ray that hits a plain wall reads the wall's cache cell, which holds no emitter light.">
  <rect x="10" y="10" width="300" height="118" fill="none" stroke="currentColor" stroke-opacity="0.5"/>
  <rect x="120" y="10" width="70" height="5" fill="#ffc878"/>
  <rect x="304" y="44" width="6" height="46" fill="#7dd3fc"/>
  <rect x="10" y="50" width="12" height="12" fill="none" stroke="currentColor" stroke-dasharray="2 2"/>
  <g stroke="#ffc878" stroke-width="1.4">
    <line x1="150" y1="128" x2="155" y2="15"/>
    <line x1="150" y1="128" x2="304" y2="70"/>
  </g>
  <line x1="150" y1="128" x2="22" y2="56" stroke="currentColor" stroke-width="1.2" stroke-dasharray="4 3" stroke-opacity="0.8"/>
  <circle cx="150" cy="128" r="5" fill="#ffc878"/>
  <g fill="currentColor" font-family="system-ui, sans-serif" font-size="10">
    <text x="196" y="26">strip: its emission</text>
    <text x="298" y="104" text-anchor="end">screen: its emission</text>
    <text x="28" y="76">wall: its cell, empty</text>
    <text x="160" y="146">a screen probe</text>
  </g>
</svg>`;

showExplainer({
    seeing: [
        'A living room with no light objects at all. Two light strips under the ceiling and the screen on the far wall are its only light.',
        'Everything they light is GI: the radiance cache on its default settings. `GI only` hides the emitters themselves and leaves the light they put on the room.',
    ],
    how: [
        `The cache puts one screen probe on each 16 × 16 pixel tile of the surfaces you see, and each probe traces 64 rays ([radiance-cache.md](${API})).`,
        "A ray that hits the front of an emissive surface takes that surface's emission and stops there. That is the `useDirectLighting` setting, on by default. It costs no extra rays: the probe traces that ray anyway.",
        "The emission is the material's own, `emissive` × `emissiveIntensity` × `emissiveMap`, so changing the material is all it takes. The page calls `markChanged` on the meshes, so GI sees an edit in the next frame.",
        "A ray that hits a plain wall reads that wall's world-cache cell instead. The cache's own lights don't include emitters, so those cells hold none of the strips' or the screen's light. The room gets the emitters' light once, at the probes, and no further bounce of it.",
        'The GPU time panel shows what the cache costs per pass. Those are this page\'s own readings on your GPU.',
    ],
    diagram: DIAGRAM,
    diagramCaption: 'Rays that hit an emitter take its light. Rays that hit anything else read a cache cell, and the cells hold no emitter light.',
    tryThis: [
        'Set `view` to `GI only`: the strips and the screen go dark, and the light they cast on the room stays.',
        "Change the strips' `colour` to blue: the walls, floor and furniture turn blue within a few frames.",
        "Drag the strips' `brightness` to 0, so only the screen lights the room. Switch `screen` between `colour bars` and `warm and cool halves`: the room's light changes colour with the picture.",
        'Then set `screen` to `off`: the room goes black, because nothing else lights it.',
    ],
    tradeOffs: {
        pros: [
            { text: 'Any material can be a light: no light objects to place, and a texture on the emitter colours its light.' },
            { text: 'It needs no extra rays. A probe ray that hits an emitter stops there and reads no cache cell.', source: docsLink('api/radiance-cache.md#world-cache-gi-12s-hash-grid', 'radiance-cache.md, "World cache"') },
        ],
        cons: [
            {
                text: 'Emitters are not lights for the cache, so their light reaches only the probes whose rays hit them, never the cells. On the emissive Cornell box the cache alone reads −49.1% against Cycles. With a DDGI volume stacked under it, it reads −1.8%.',
                source: VARIANTS,
            },
            {
                text: "The neural tail can't fix this. It learns from the cells, which hold no emitter light, so it reads −49.1% on the emissive Cornell box too (register D21).",
                source: VARIANTS,
            },
            {
                text: "Small, very bright emitters are noisy, because a probe finds them only when its rays happen to hit them. On the still emissive Cornell box, the cache's irradiance before the denoiser changes 30% per pixel from frame to frame (mean). In two-room, lit by a lamp, it changes 4.6%.",
                source: docsLink('NOTES.md#radiance-cache-measured-lessons', 'NOTES.md, "How steady the radiance cache\'s irradiance is"'),
            },
        ],
    },
    api: API_PAGES.radianceCache,
    more: [KNOWN_LIMITS, VARIANTS],
});
