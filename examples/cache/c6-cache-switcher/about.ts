/** What C6 shows and how to read it: the page's explainer panel (shared/explainer.ts). Plain data; main.ts holds the example, cards.ts the variants' cards. */

import { API_PAGES, DOCS_ROOT, docsLink, showExplainer } from '../../shared/explainer.ts';

const API = `${DOCS_ROOT}api/radiance-cache.md`;
const VARIANTS = `${DOCS_ROOT}design/radiance-cache-variants.md`;

const DIAGRAM = `<svg viewBox="0 0 320 150" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Screen probes send rays; the world cache answers each ray with the light at its hit; the variants differ in where a cell's later bounces come from.">
  <g fill="none" stroke="currentColor" stroke-opacity="0.6">
    <rect x="6" y="22" width="92" height="46" rx="4"/>
    <rect x="120" y="22" width="92" height="46" rx="4"/>
    <rect x="234" y="6" width="80" height="138" rx="4" stroke="#86efac"/>
    <rect x="6" y="98" width="206" height="34" rx="4"/>
  </g>
  <g stroke="#ffc878" stroke-width="1.4" fill="#ffc878">
    <line x1="98" y1="38" x2="114" y2="38"/><path d="M 120 38 l -7 -3.5 v 7 z"/>
    <line x1="212" y1="45" x2="228" y2="45"/><path d="M 234 45 l -7 -3.5 v 7 z"/>
  </g>
  <g stroke="#7dd3fc" stroke-width="1.4" fill="#7dd3fc">
    <line x1="120" y1="56" x2="104" y2="56"/><path d="M 98 56 l 7 -3.5 v 7 z"/>
    <line x1="52" y1="68" x2="52" y2="92"/><path d="M 52 98 l -3.5 -7 h 7 z"/>
  </g>
  <g fill="currentColor" font-family="system-ui, sans-serif" font-size="10">
    <text x="52" y="40" text-anchor="middle" font-weight="600">screen probes</text>
    <text x="52" y="54" text-anchor="middle">on what you see</text>
    <text x="166" y="40" text-anchor="middle" font-weight="600">world cache</text>
    <text x="166" y="54" text-anchor="middle">light at the hits</text>
    <text x="106" y="31" font-size="8">rays</text>
    <text x="102" y="78" font-size="8">cells answer</text>
    <text x="274" y="24" text-anchor="middle" font-weight="600">later bounces</text>
    <text x="242" y="44">one more ray</text>
    <text x="242" y="62">DDGI's probes</text>
    <text x="242" y="80">SHaRC's paths</text>
    <text x="242" y="98">a network</text>
    <text x="242" y="116">feedback</text>
    <text x="109" y="113" text-anchor="middle">denoiser, then per pixel:</text>
    <text x="109" y="126" text-anchor="middle">albedo × E / π added to the picture</text>
  </g>
</svg>`;

showExplainer({
    seeing: [
        'Two rooms joined by a doorway. One lamp lights the left room; the right room has no light of its own, so all its light comes through the doorway.',
        "The radiance cache lights them, with one world cache at a time: pick it under `Variant`, top right. Its card says how it works, what it is good and bad at, what P6.5 measured and what it needs. Under the card is this page's own GPU time per pass.",
        '`Scene` swaps in a courtyard under the open sky.',
    ],
    how: [
        `The radiance cache puts a probe on the visible surface in each 16 × 16-pixel tile and traces 64 rays from it ([radiance-cache.md](${API})). Each ray needs the light arriving at its hit, which is the light that surface passes on. Tracing a whole path from every hit, every frame, would cost too much.`,
        'A world cache answers instead. It keeps the light at ray hits in cells laid out in world space, smaller near the camera, and keeps them from frame to frame. Every ray that lands in a cell reads it, so the work is shared across rays and frames.',
        "The variants differ in where a cell's later bounces come from. GI-1.2's grid traces one more ray, so it stops at two bounces. Over DDGI it reads DDGI's probes for all the rest. SHaRC follows short paths and ends them in its own cells. The others light cells on a budget, train a small network, or feed light back from the cells or from last frame's picture.",
        `Outdoors most light comes straight from the sun and sky, so a few bounces are enough. In a closed room light keeps bouncing, and a cache that stops early reads dark. So the variants agree in the open and part in the closed rooms ([radiance-cache-variants.md](${VARIANTS}#summary)).`,
        "Reading a card: its numbers are P6.5's, every variant run on ten views of seven scenes and compared with Cycles, the path tracer. Bias is how much darker (−) or brighter (+) the GI is than Cycles': the card gives its mean size over the ten views, and two-room's own. Frames to 90% count how long the picture takes to get 90% of the way to its new state after the lights double or a box moves. Memory is the radiance cache's at 1920 × 1080.",
        "Reading the timing: it is this page's own, on your GPU, the mean of the last 30 timed frames. Hover or tap a pass for what it does. `radiance-cache` is the screen probes, the world cache and the denoiser; `ddgi` is the volume under it. The world cache's passes change with the variant: `Sharc…` for SHaRC, `ClaimCells` to `ShadeCells` for budgeted cells, `SampleBounceSourceCells` where DDGI gives the bounce.",
        "`Overlay` shows the cache's own debug views, a frame behind: the probe atlas (each probe's 8 × 8 cells of incoming light, in its own 8 × 8-pixel tile), the probe mask (which tiles hold a probe this frame) and each probe's light as its spherical harmonics hold it.",
    ],
    diagram: DIAGRAM,
    diagramCaption: 'Every variant keeps the screen probes and the denoiser; only the world cache and its later bounces change.',
    tryThis: [
        `Pick \`GI-1.2 grid alone\` under \`Variant\`: the rooms go darker. P6.5 measured two-room's first view 44% too dark this way against Cycles ([radiance-cache-variants.md](${VARIANTS}#summary)).`,
        'Pick `SHaRC cache alone`: most of the light comes back with no DDGI volume, and the memory line drops by about 1 GiB.',
        'Switch `Scene` to the courtyard and step through the variants again: under the open sky they look much alike.',
        'Set `Overlay` to `Probe mask`: each colour is one probe, its white dot the pixel it sits on. Orbit, and the colours follow the surfaces.',
    ],
    tradeOffs: {
        pros: [
            { text: 'With DDGI under it, the GI-1.2 grid has the smallest bias on all ten views: 3.1% on average.', source: docsLink('design/radiance-cache-variants.md#recommendation', 'radiance-cache-variants.md, "Recommendation"') },
            { text: 'SHaRC needs no volume, stays within 2-5 points of that, and uses under a third of the memory.', source: docsLink('design/radiance-cache-variants.md#trade-offs', 'radiance-cache-variants.md, "Trade-offs"') },
        ],
        cons: [
            { text: "The GI-1.2 grid's world cache is about 1 GiB, Capsaicin's default capacity, whatever the scene; most of it sits unused on the test scenes.", source: docsLink('design/radiance-cache-variants.md#gpu-time', 'radiance-cache-variants.md, "GPU time"') },
            { text: 'Every variant takes over a second to get 90% of the way after the lights double. Most of that lag is in the screen probes and the world cache, the same for each.', source: docsLink('design/radiance-cache-variants.md#still-open', 'radiance-cache-variants.md, "Still open"') },
            { text: 'Most GPU times in the docs come from thermally throttled runs. Only five, on Bistro exterior, are VALID, and the cards quote only those.', source: docsLink('design/radiance-cache-variants.md#gpu-time', 'radiance-cache-variants.md, "GPU time"') },
        ],
    },
    api: API_PAGES.radianceCache,
    more: [docsLink('design/radiance-cache-variants.md', 'The variants compared (P6.5)')],
});
