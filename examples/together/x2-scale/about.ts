/** What X2 shows and how it works: the page's explainer panel (shared/explainer.ts). Plain data; main.ts holds the example. */

import { API_PAGES, docsLink, GUIDE, showExplainer } from '../../shared/explainer.ts';

const SEVERAL_VOLUMES = docsLink('design/m3-moving.md#41-several-volumes-in-one-technique', 'm3-moving.md §4.1, "Several volumes in one technique"');
const CAMERA_VOLUME = docsLink('design/m3-moving.md#42-the-camera-following-volume-scrolling', 'm3-moving.md §4.2, "The camera-following volume"');
const SHARC_SETTINGS = docsLink('api/radiance-cache.md#variants', 'radiance-cache.md, "Variants", the sharc row');
const VARIANT_MEMORY = docsLink('design/radiance-cache-variants.md#gpu-time', 'P6.5, radiance-cache-variants.md, "GPU time" (its memory table)');
const VARIANT_TRADE_OFFS = docsLink('design/radiance-cache-variants.md#trade-offs', 'P6.5, radiance-cache-variants.md, "Trade-offs"');
const PLANNED = docsLink('api/radiance-cache.md#variants-compared', 'radiance-cache.md, "Variants compared"');

/** A row of dots: `count` across, `rows` down, from (x, y), `step` apart. */
const dots = (x: number, y: number, count: number, rows: number, step: number, r: number): string =>
    Array.from({ length: count * rows }, (_, index) => `<circle cx="${x + (index % count) * step}" cy="${y + Math.floor(index / count) * step}" r="${r}"/>`).join('');

/** Top: DDGI's boxes of probes, dense near the camera. Bottom: SHaRC's cells, growing with distance. */
const DIAGRAM = `<svg viewBox="0 0 320 180" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Top: DDGI. A coarse grid of probes covers the whole town, and a small dense grid follows the camera. Bottom: SHaRC. Cells on the ground grow larger the farther they are from the camera.">
  <g font-family="system-ui, sans-serif" font-size="10" fill="currentColor">
    <text x="4" y="12">DDGI: boxes of probes</text>
    <text x="4" y="104">SHaRC: cells sized by distance</text>
  </g>
  <rect x="24" y="20" width="290" height="64" fill="none" stroke="#4cc9f0" stroke-width="1.2"/>
  <g fill="#4cc9f0">${dots(30, 26, 10, 3, 28, 2)}</g>
  <rect x="24" y="34" width="70" height="36" fill="none" stroke="#ffd166" stroke-width="1.2"/>
  <g fill="#ffd166">${dots(29, 39, 8, 4, 8.6, 1.3)}</g>
  <g fill="currentColor" font-family="system-ui, sans-serif" font-size="9" fill-opacity="0.8">
    <text x="100" y="64">near window: fine, follows you</text>
    <text x="190" y="96">the town: coarse, fixed</text>
  </g>
  <path d="M8 52 l12 -7 v14 z" fill="currentColor"/>
  <path d="M8 144 l12 -7 v14 z" fill="currentColor"/>
  <line x1="20" y1="170" x2="316" y2="170" stroke="currentColor" stroke-opacity="0.6" stroke-width="1.5"/>
  <g fill="none" stroke="#06d6a0" stroke-width="1.2">
    <rect x="30" y="166" width="4" height="4"/>
    <rect x="40" y="164" width="6" height="6"/>
    <rect x="54" y="161" width="9" height="9"/>
    <rect x="74" y="157" width="13" height="13"/>
    <rect x="102" y="150" width="20" height="20"/>
    <rect x="144" y="140" width="30" height="30"/>
    <rect x="204" y="125" width="45" height="45"/>
  </g>
</svg>`;

showExplainer({
    seeing: [
        'A town made from a seed: 64 lots of blocks, courtyards and plazas with 6 m streets, on a 200 m square of open ground under the sun and the sky. The camera tours four streets at 5 m/s until a key or a drag takes over.',
        "Two ways to light it, switched live: DDGI with several volumes, or the radiance cache with SHaRC and no volume at all. The column on the right shows each method's GPU time and memory as you fly. Those are this page's own readings on your GPU, not numbers from the docs.",
    ],
    how: [
        "DDGI keeps light in probes on a grid inside boxes you place. One coarse box over the whole town (blue, 8 m between probes by default) is cheap, but it can't hold light that changes within a 6 m street. So the page adds a window of 32 × 8 × 32 probes that follows the camera (`follow: 'camera'`, yellow, 1.5 m apart), and with three volumes a middle window between them (green). The nearer window has the higher `priority`.",
        `The probes themselves never move. When you go more than 0.75 of a spacing from the window's centre, the window moves by whole planes: the planes it leaves behind are reused for the planes entering ahead, which are updated first and fade in. Source: [${CAMERA_VOLUME.label}](${CAMERA_VOLUME.href}).`,
        `Where volumes overlap, the one with the higher priority answers, and the next fills in across the window's fade band, up to 10 spacings wide. A point reads at most 2 volumes, and one technique takes at most 8. Sources: [${SEVERAL_VOLUMES.label}](${SEVERAL_VOLUMES.href}) and §4.2.`,
        `Each probe costs memory wherever it is. M3's design estimated about 1.5 KB a probe, so 19 MB for a 32 × 12 × 32 window of 12,288 probes; the memory readout shows what the volumes cost here. Its updates come out of a GPU budget, 2 ms by default ([${GUIDE.label}](${GUIDE.href})): 132 probe updates a frame on Bistro exterior at 2 ms. A camera at 5 m/s through a 32 × 12 × 32 window at 1.5 m needs about 43 of them a frame for the planes entering. Source: [${CAMERA_VOLUME.label}](${CAMERA_VOLUME.href}).`,
        `The radiance cache keeps no probes in the world. Screen probes, one per 16 × 16 pixels ([${API_PAGES.radianceCache.label}](${API_PAGES.radianceCache.href})), trace rays, and each hit adds its light to a cell of SHaRC's hash table. A cell is named by its position, its size and which way its surface faces, hashed into 1M slots, so cells exist wherever rays land. There is no box to place and no edge to fly past.`,
        `SHaRC's levels of detail: a cell's size is set by its distance from the camera, so it covers about 16 pixels on screen (\`cellPixels\`), and it is never smaller than 2 cm (\`minCell\`). Near walls fall into small cells and far streets into big ones. A cell with fewer than 16 samples (\`minSamples\`) is read from a coarser level instead, up to 3 levels up (\`fallbackLevels\`), and each sample is also added to those coarser cells (\`pyramid\`). A cell no ray has reached for 64 frames is freed (\`staleFrames\`). Source: [${SHARC_SETTINGS.label}](${SHARC_SETTINGS.href}).`,
        "The cache can't draw its cells in the world yet, so this page can't show their size by distance: that is the greyed-out `show cells`.",
        "Which suits which world? DDGI suits a world you can cover with boxes, even a big one with a coarse box and a window that follows the camera, when you want memory you can predict and a GPU budget you set. Its probes also keep the light behind the camera. The cache suits a world with no shape to put a box around, or detail finer than a probe grid holds, on a device with the memory. Its cost follows the screen, not the size of the world.",
    ],
    diagram: DIAGRAM,
    diagramCaption: 'DDGI: a coarse box over everything and a fine window that follows the camera. SHaRC: cells grow with distance, so each covers about the same area on screen.',
    tryThis: [
        'Leave the tour running and watch `GPU time per pass`: `scroll` moves the window and stays small, and the DDGI total stays about the same wherever you go.',
        'Press `jump above the town`: the yellow near window rises with you, because it follows the camera on every axis, and the blue town volume alone lights the streets below.',
        'Set `count` to `1: the town`, then `3: + middle window`: the memory follows the probe count. Then set `town spacing` to 4 and see what a finer coarse box costs over the whole town.',
        "Pick `Radiance cache: SHaRC` and fly out past the edge of the town: nothing has to be placed or moved. Compare its memory and GPU time with DDGI's rows on the right.",
    ],
    tradeOffs: {
        pros: [
            { text: 'DDGI: memory follows the probe count, not the screen: 12–72 MiB on the scenes measured, from two-room to Bistro exterior.', source: VARIANT_MEMORY },
            { text: 'DDGI: its updates stay inside a budget. At 132 updates a frame, a 1.5 m window following a camera at 5 m/s takes about a third of it.', source: CAMERA_VOLUME },
            { text: 'SHaRC: nothing to place and no edge: cells are hashed from their position, and finer near the camera.', source: SHARC_SETTINGS },
            { text: "SHaRC: 83 MiB for the cache itself and 415 MiB at 1080p in all, less than a third of the memory of the cache's default world cache, the GI-1.2 grid.", source: VARIANT_TRADE_OFFS },
            { text: 'SHaRC: every bounce with no volume. In the open scenes measured its block error ties with or beats the grid over DDGI.', source: VARIANT_TRADE_OFFS },
        ],
        cons: [
            { text: 'DDGI: you place the volumes, and they must cover what the camera sees. A coarse volume blurs the bounce near contacts and leaks light through thin walls.', source: VARIANT_TRADE_OFFS },
            { text: 'DDGI: a window follows the camera on every axis, so flying up takes its probes off the ground. Probes entering it need 2 updates before they show.', source: CAMERA_VOLUME },
            { text: 'SHaRC: all of its work runs every frame, with no budget. On the big scenes measured its world cache was the costliest part, 14–19 ms (relative only: those timing runs were thermally limited).', source: VARIANT_TRADE_OFFS },
            { text: 'SHaRC: it lags after changes. 90% of a light change comes at a median of 83 frames, against 77 for the grid over DDGI.', source: VARIANT_TRADE_OFFS },
            { text: "The cache's default world cache, the GI-1.2 grid, holds 1,085 MiB (about 1.1 GiB) whatever the world.", source: VARIANT_MEMORY },
            { text: 'A smaller grid (P6.8), and SHaRC as the default when no volume is stacked (P6.7), are planned.', source: PLANNED },
        ],
    },
    api: API_PAGES.radianceCache,
    more: [API_PAGES.platform, docsLink('design/m3-moving.md#4-several-volumes-and-the-camera-following-volume', 'Several volumes and the camera-following volume (M3)'), docsLink('design/radiance-cache-variants.md', 'The world-cache variants compared (P6.5)')],
});
