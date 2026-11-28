/** What C2 shows and how it works: the page's explainer panel (shared/explainer.ts). Plain data; main.ts holds the example. */

import { API_PAGES, docsLink, showExplainer } from '../../shared/explainer.ts';

const VARIANTS = 'design/radiance-cache-variants.md';
const P65_SUMMARY = docsLink(`${VARIANTS}#summary`, 'radiance-cache-variants.md, "Summary"');
const P65_TRADE_OFFS = docsLink(`${VARIANTS}#trade-offs`, 'radiance-cache-variants.md, "Trade-offs"');

/**
 * A box on a shelf against a wall, twice. Left: DDGI's probes on a grid, and a
 * point beside the box reading probes a spacing away. Right: a screen probe
 * on that point, its rays starting there, the box blocking some of them.
 */
const DIAGRAM = `<svg viewBox="0 0 320 150" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Left: a probe grid; a point beside a box reads probes that sit a spacing away from it. Right: a probe on the surface beside the box; the box blocks the rays on its side.">
  <g stroke="currentColor" stroke-opacity="0.6" fill="none">
    <line x1="8" y1="118" x2="152" y2="118"/>
    <line x1="168" y1="118" x2="312" y2="118"/>
    <rect x="58" y="84" width="34" height="34"/>
    <rect x="218" y="84" width="34" height="34"/>
  </g>
  <g fill="currentColor">
    <circle cx="30" cy="30" r="4"/><circle cx="75" cy="30" r="4"/><circle cx="120" cy="30" r="4"/>
    <circle cx="30" cy="70" r="4"/><circle cx="75" cy="70" r="4"/><circle cx="120" cy="70" r="4"/>
  </g>
  <g stroke="currentColor" stroke-dasharray="3 2" stroke-opacity="0.8">
    <line x1="100" y1="118" x2="75" y2="70"/><line x1="100" y1="118" x2="120" y2="70"/>
  </g>
  <circle cx="100" cy="118" r="4" fill="none" stroke="#7dd3fc" stroke-width="2"/>
  <g stroke="#ffc878" stroke-width="1.2">
    <line x1="260" y1="118" x2="252" y2="100"/><line x1="260" y1="118" x2="252" y2="90"/>
    <line x1="260" y1="118" x2="245" y2="20"/><line x1="260" y1="118" x2="280" y2="18"/>
    <line x1="260" y1="118" x2="310" y2="50"/><line x1="260" y1="118" x2="312" y2="96"/>
  </g>
  <circle cx="260" cy="118" r="5" fill="#ffc878"/>
  <g fill="currentColor" font-family="system-ui, sans-serif" font-size="10">
    <text x="80" y="140" text-anchor="middle">DDGI: probes on a grid</text>
    <text x="240" y="140" text-anchor="middle">cache: a probe on the surface</text>
  </g>
</svg>`;

showExplainer({
    seeing: [
        'A cabinet row with an open shelf on legs, boxes on it, under it and in front of it. Left of the line is the radiance cache; right of it is DDGI. Drag the line, or set `wipe position`.',
        'The two differ most where things meet: under the boxes, under the shelf board, between the legs and the wall. Those are the contacts.',
    ],
    how: [
        "DDGI keeps light in probes on a grid, here 0.25 m apart. A surface reads the probes around it and blends them. The objects here are 0.1 to 0.35 m across, about one spacing or less, so the light around them changes over less than the spacing. That change is averaged away: it blurs.",
        'The radiance cache puts its probes on the surfaces the camera sees, one per 16 × 16 pixel tile, each tracing 64 rays from the surface itself. A probe on the shelf beside a box starts its rays there, so the box blocks the rays on its side. Its detail follows the screen, not a grid.',
        "The probes' rays end in a world cache. Here that is SHaRC (`worldCache: 'sharc'`), which needs no volume, so the cache's side has no probe grid at all.",
        "Both pictures come from one scene pass. Its depth, normals and motion feed the cache, and each technique's `ddgi()` composite adds its own GI to the same direct light. The wipe picks a side per pixel, or shows |cache − DDGI| × 4.",
    ],
    diagram: DIAGRAM,
    diagramCaption: 'A point beside a box. DDGI reads probes a spacing away; the cache traces from the point itself.',
    tryThis: [
        'Set `DDGI probe spacing` to `0.5 m` and wait for the new probes to warm up (a few seconds). Drag the line over the red box: on DDGI\'s side its red spreads onto the side of the tall cabinet to the left, and the light under the shelf board turns patchy. The cache\'s side doesn\'t change: it has no grid.',
        'Tick `difference`. Bright means the two disagree. Edges and halos sit around the boxes and the shelf; a wide haze is the two techniques differing overall, which by itself doesn\'t say which is right.',
        'Set the spacing back to `0.25 m`: the halos shrink but don\'t go. Finer probes cost more memory and more probe updates.',
    ],
    tradeOffs: {
        pros: [
            {
                text: "P6.5 measured the cache's variants against Cycles in 8 × 8 pixel blocks. In the open scenes SHaRC's block error ties or beats the grid over DDGI's (the cache's hash grid with DDGI giving its later bounces), and its p95 is lower: Sponza 40% against 52%, Bistro exterior 38% against 47%. The reason given: DDGI's coarse field blurs the bounce at contacts.",
                source: P65_SUMMARY,
            },
            { text: 'No volume to place: the cache lights what the camera sees, with detail where it looks.', source: docsLink('guides/ddgi-or-radiance-cache.md#how-they-differ', 'the guide, "How they differ"') },
        ],
        cons: [
            {
                text: 'DDGI is the steadier choice: a fixed probe grid, which the cache\'s design doc calls "stable but blurry and slow to react".',
                source: docsLink('design/radiance-cache.md', 'design/radiance-cache.md'),
            },
            {
                text: "DDGI needs only the camera. The cache needs the scene pass's depth, normals and motion every frame, at full resolution, and doesn't run under `@pmndrs/upscaler` yet.",
                source: docsLink('guides/ddgi-or-radiance-cache.md#how-they-differ', 'the guide, "How they differ"'),
            },
            {
                text: "DDGI's field answers anywhere in the world it covers, on screen or off, so a surface seen by another camera or through glass gets its own light. The cache's result answers only for the pixel being shaded on this screen: a transparent surface gets the GI of the opaque surface behind it.",
                source: docsLink('api/radiance-cache.md', 'radiance-cache.md ("Getting started", "Settings", "Without three.js")'),
            },
            {
                text: "SHaRC has limits of its own: its bias is 2-5 points behind the grid over DDGI's (mean size 5.9%), and its coarse cells also blur the light near contacts. It holds 415 MiB at 1080p in all; DDGI's field held 12-72 MiB on P6.5's scenes.",
                source: P65_TRADE_OFFS,
            },
        ],
    },
    api: API_PAGES.radianceCache,
    more: [API_PAGES.three, docsLink(VARIANTS, 'P6.5: the cache variants compared')],
});
