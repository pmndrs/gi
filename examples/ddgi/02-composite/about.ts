/** What 02 shows and how it works: the page's explainer panel (shared/explainer.ts). Plain data; main.ts holds the example. */

import { API_PAGES, docsLink, showExplainer } from '../../shared/explainer.ts';

/** The pipeline: scene pass with its MRT, the composite, the upscaler, the screen. */
const DIAGRAM = `<svg viewBox="0 0 320 70" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="The scene pass writes colour, albedo, normal and velocity; ddgi() adds GI once per pixel; the upscaler brings it to the screen.">
  <g fill="none" stroke="currentColor" stroke-opacity="0.6">
    <rect x="4" y="14" width="86" height="42" rx="4"/><rect x="118" y="14" width="70" height="42" rx="4"/><rect x="216" y="14" width="100" height="42" rx="4"/>
  </g>
  <g stroke="#ffc878" stroke-width="1.5"><line x1="90" y1="35" x2="114" y2="35"/><line x1="188" y1="35" x2="212" y2="35"/></g>
  <g fill="#ffc878"><path d="M114 31 l6 4 -6 4z"/><path d="M212 31 l6 4 -6 4z"/></g>
  <g fill="currentColor" font-family="system-ui, sans-serif" font-size="10" text-anchor="middle">
    <text x="47" y="31">scene pass</text><text x="47" y="45" font-size="8.5">colour, albedo, normal</text>
    <text x="153" y="31">ddgi()</text><text x="153" y="45" font-size="8.5">GI once a pixel</text>
    <text x="266" y="31">upscaler</text><text x="266" y="45" font-size="8.5">half res → screen</text>
  </g>
</svg>`;

const DENSE = docsLink('api/three.md#dense-scenes-gi-once-per-pixel-through-the-upscaler', 'three.md, "Dense scenes"');

showExplainer({
    seeing: [
        "01's scene and volume, drawn through three's post-processing pipeline. `ddgi()` adds GI once per pixel, and `@pmndrs/upscaler` renders at half resolution and brings the picture up to the screen.",
        "It looks like 01. What changes is the cost in scenes with many small triangles: this is GI's intended pipeline for them.",
    ],
    how: [
        'Lit materials normally take GI per fragment. With many tiny triangles that runs many times for each pixel.',
        "Here the scene pass writes its colour, `giAlbedo`, the normal and three's `velocity` with `mrt()`. `ddgi()` then adds `giAlbedo × E / π` once per pixel, before tone mapping.",
        "The upscaler runs after it, in its Performance mode: half the display's size in each direction.",
    ],
    diagram: DIAGRAM,
    tryThis: [
        'Look at the edges of the boxes: they are smooth with no multisampling. The upscaler’s temporal pass is the anti-aliasing.',
        "Compare the stats line with 01's: GI's own work is the same. Only how it reaches the screen changed.",
        'Open the source: `pipeline.render()` replaces `renderer.render(scene, camera)`.',
    ],
    tradeOffs: {
        pros: [{ text: 'Showing GI costs 0.41 ms on Sponza at 1440p under the upscaler, against 3.4 ms per fragment (RTX 3060 laptop).', source: DENSE }],
        cons: [
            { text: 'No occlusion on the GI term by default. GTAO at half the render resolution is an opt-in: 0.73 ms on the same test.', source: DENSE },
            { text: "The upscaler's own cost is separate: 1.55 ms a frame in Performance mode, with or without GI.", source: DENSE },
        ],
    },
    api: API_PAGES.three,
});
