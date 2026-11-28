/** What C1 shows and how it works: the page's explainer panel (shared/explainer.ts). Plain data; main.ts holds the example. */

import { API_PAGES, docsLink, showExplainer } from '../../shared/explainer.ts';

const VARIANTS_TRADE_OFFS = docsLink('design/radiance-cache-variants.md#trade-offs', 'radiance-cache-variants.md, "Trade-offs"');
const DEVICE_NEEDS = docsLink('api/radiance-cache.md#what-it-needs-from-the-device', 'radiance-cache.md, "What it needs from the device"');

/** Side view: the camera sees a surface; a probe there sends rays; cells sit only where rays land, larger with distance. */
const DIAGRAM = `<svg viewBox="0 0 320 150" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Side view. The camera looks at the ground. A screen probe on the ground sends rays; where they land, the world cache holds a cell, and the cells are larger the farther they are from the camera.">
  <line x1="10" y1="128" x2="310" y2="128" stroke="currentColor" stroke-opacity="0.6"/>
  <line x1="300" y1="30" x2="300" y2="128" stroke="currentColor" stroke-opacity="0.6"/>
  <path d="M30 46 l-16 -8 v16 z" fill="currentColor"/>
  <line x1="30" y1="46" x2="80" y2="128" stroke="currentColor" stroke-dasharray="3 2" stroke-opacity="0.7"/>
  <g stroke="#ffc878" stroke-width="1.2">
    <line x1="80" y1="128" x2="140" y2="128"/><line x1="80" y1="128" x2="300" y2="100"/>
    <line x1="80" y1="128" x2="300" y2="58"/><line x1="80" y1="128" x2="210" y2="128"/>
  </g>
  <g fill="#7dd3fc" fill-opacity="0.35" stroke="#7dd3fc">
    <rect x="136" y="120" width="8" height="8"/>
    <rect x="203" y="116" width="12" height="12"/>
    <rect x="291" y="92" width="16" height="16"/>
    <rect x="291" y="50" width="16" height="16"/>
  </g>
  <circle cx="80" cy="128" r="5" fill="#ffc878"/>
  <g fill="currentColor" font-family="system-ui, sans-serif" font-size="10">
    <text x="12" y="28">camera</text>
    <text x="60" y="146">screen probe</text>
    <text x="120" y="30">cells only where rays land,</text>
    <text x="120" y="42">larger farther from the camera</text>
  </g>
</svg>`;

showExplainer({
    seeing: [
        'A walled courtyard in the sun, lit by the radiance cache alone. There is no probe box: nothing to place or size.',
        'The light in the shade (the walls facing away from the sun, the ground under the gateway) is GI: sky light, and sunlight bounced off the paving and the walls.',
        "The page sets `worldCache: 'sharc'` itself. With no DDGI volume that is the world cache to use; making it the default when there is no volume is still to come (TASKS P6.7).",
    ],
    how: [
        'Screen probes. Each frame the cache puts one probe on the visible surface in every 16 × 16 pixel tile. Each probe traces 64 rays and keeps its light from frame to frame.',
        'The world cache. The rays end in cells of a hash table, keyed by position and by the way the surface faces. A cell exists only where a ray has landed, and one with no samples for 64 frames is dropped. A cell spans about 16 pixels of the screen and at least 2 cm, so cells grow coarser with distance from the camera.',
        "SHaRC lights its cells with short paths from the probes' hits, with the cache itself as the end of each path, so the cells carry every bounce.",
        "What the host provides: the screen's depth, normals and motion, every frame. `screenInputs` adds them to the scene pass's MRT, and `screen.encodeBefore` runs the cache's screen stage after the pass draws. Then `ddgi()` adds albedo × E / π to each pixel, E read from the cache's denoised irradiance at that pixel.",
    ],
    diagram: DIAGRAM,
    diagramCaption: 'Only the surfaces the rays reach get cells. Nothing is stored for empty space.',
    tryThis: [
        'Untick `GI`: the shade goes black, and only the sun and the sky remain. Tick it again: the light is back at once, because the cache kept running.',
        'Drag `sun height` down: the sunlit patch climbs the walls, and the bounce light in the shade follows it, a little behind.',
        'Turn `sun direction`: the sunlit wall changes, and the bounce light moves to the walls facing it.',
        "Orbit out over the walls: there is no box to resize. Surfaces that come into view get cells as the probes' rays reach them.",
    ],
    tradeOffs: {
        pros: [
            { text: 'No volume to place: it lights what the camera sees, wherever that is.', source: docsLink('api/radiance-cache.md', 'radiance-cache.md') },
            { text: 'SHaRC needs nothing from the host and carries every bounce. Its bias against the path tracer is 5.9% averaged over ten views, 2–5 points behind the grid over DDGI.', source: VARIANTS_TRADE_OFFS },
            { text: "In the open scenes its 8 × 8 block error ties or beats the grid over DDGI's.", source: VARIANTS_TRADE_OFFS },
            { text: "It holds less than a third of the GI-1.2 grid's memory: 415 MiB at 1080p in all.", source: VARIANTS_TRADE_OFFS },
        ],
        cons: [
            { text: 'The host renders an MRT pass with depth, normals and motion: up to 56 bytes a pixel. The device needs 10 storage buffers per shader stage; three\'s own device gives 8.', source: DEVICE_NEEDS },
            { text: 'It follows changes more slowly than the grid over DDGI: after the lights double, 90% of the change shows at a median of 83 frames, against 77.', source: VARIANTS_TRADE_OFFS },
            { text: 'Its frame costs more: 26.2 ms against 21.8 ms for the grid over DDGI on Bistro exterior (RTX 3060 laptop, 1280 × 720).', source: docsLink('design/radiance-cache-variants.md#gpu-time', 'radiance-cache-variants.md, "GPU time"') },
            { text: 'Indirect diffuse only, at full resolution only: not under the upscaler yet.', source: docsLink('api/radiance-cache.md#known-limits', 'radiance-cache.md, "Known limits"') },
        ],
    },
    api: API_PAGES.radianceCache,
    more: [docsLink('design/radiance-cache-variants.md#recommendation', 'Which world cache, and why')],
});
