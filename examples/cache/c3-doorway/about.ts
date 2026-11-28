/** What C3 shows and how it works: the page's explainer panel (shared/explainer.ts). Plain data; main.ts holds the example. */

import { API_PAGES, docsLink, showExplainer } from '../../shared/explainer.ts';

const VARIANTS = docsLink('design/radiance-cache-variants.md#summary', 'P6.5, radiance-cache-variants.md, "Summary"');
const VARIANT_TRADE_OFFS = docsLink('design/radiance-cache-variants.md#trade-offs', 'P6.5, radiance-cache-variants.md, "Trade-offs"');
const REGISTER_D68 = docsLink('design/radiance-cache.md#divergence-register', 'RC3.6, design/radiance-cache.md, register row D68');
const TWO_BOUNCES = docsLink('NOTES.md#radiance-cache-measured-lessons', 'NOTES.md, "Radiance cache (measured lessons)"');

/** One bounce: lamp → wall → floor → eye. */
const DIAGRAM = `<svg viewBox="0 0 320 150" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Light leaves a lamp and lands on a wall: direct light. The wall reflects part of it onto the floor: the first bounce. The floor reflects part of that into the eye.">
  <defs>
    <marker id="c3-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M0 0 L10 5 L0 10 z" fill="#ffc878"/>
    </marker>
  </defs>
  <line x1="10" y1="128" x2="290" y2="128" stroke="currentColor" stroke-opacity="0.6" stroke-width="2"/>
  <line x1="290" y1="10" x2="290" y2="128" stroke="currentColor" stroke-opacity="0.6" stroke-width="2"/>
  <circle cx="60" cy="30" r="9" fill="#ffc878"/>
  <g fill="none" stroke="#ffc878" stroke-width="1.6" marker-end="url(#c3-arrow)">
    <line x1="70" y1="34" x2="286" y2="62"/>
    <line x1="286" y1="66" x2="186" y2="124" stroke-opacity="0.75"/>
    <line x1="180" y1="124" x2="44" y2="100" stroke-opacity="0.5"/>
  </g>
  <g fill="none" stroke="#7dd3fc" stroke-width="2">
    <path d="M18 98 Q32 86 46 98 Q32 110 18 98 z"/>
    <circle cx="32" cy="98" r="3" fill="#7dd3fc"/>
  </g>
  <g fill="currentColor" font-family="system-ui, sans-serif" font-size="10">
    <text x="48" y="16">lamp</text>
    <text x="172" y="40">direct light</text>
    <text x="246" y="104" text-anchor="end">bounce 1</text>
    <text x="104" y="104">to the eye</text>
    <text x="284" y="22" text-anchor="end">wall</text>
    <text x="284" y="144" text-anchor="end">floor</text>
    <text x="14" y="124">eye</text>
  </g>
</svg>`;

showExplainer({
    seeing: [
        'Two rooms joined by a doorway. A lamp lights the near room. The far room, where the camera stands, has no light of its own: all its light comes through the doorway.',
        "The panel at the bottom right shows the far room's GI for each second-bounce source, as a number and a bar. The page measures them itself.",
    ],
    how: [
        'A bounce is light reflected off a surface. Lamp light that lands on a wall is direct light. The wall reflects part of it onto the floor, so the floor is lit by one bounce. Each bounce loses the light the surface absorbs.',
        "Little lamp light falls straight into the far room. Most of its light has bounced in the near room, come through the doorway, and bounced again off the far room's floor and walls. That takes several bounces.",
        "The cache's screen probes trace rays from the surfaces you see. Each ray ends in a world-cache cell. The cell holds the direct light where the ray landed, plus one more hit: the second bounce.",
        `In GI-1.2 that second hit reads only its own cell's direct light, so the cache holds two bounces at most. Light that needs a third bounce never arrives, and closed rooms come out dark. Source: [NOTES.md, "Radiance cache (measured lessons)"](${TWO_BOUNCES.href}).`,
        "With DDGI underneath, the second hit reads DDGI's irradiance instead of tracing a ray. DDGI's probes feed their own light back every frame, so its field holds every bounce, and the cache gets them all.",
        'SHaRC replaces the hash grid. It follows each probe hit with short paths that end in its own cache, so its cells carry every bounce without DDGI.',
    ],
    diagram: DIAGRAM,
    diagramCaption: 'One bounce: lamp light lands on the wall (direct light), the wall lights the floor (bounce 1), and the floor sends it to your eye.',
    tryThis: [
        "Let the `DDGI underneath` bar settle. Then pick `GI-1.2's own rays` under `second bounce`: the far room darkens, and its bar shrinks well below DDGI's.",
        'Pick `SHaRC`: the page makes the platform again, with only direct light for a moment. The far room then comes back close to DDGI\'s level, a little more slowly.',
        'Untick `door open`: the far room goes dark with every source, because no light can reach it. Tick it again and watch each source recover.',
        'Each bar keeps its last reading for each door state, so after a round of switching the three sit side by side.',
    ],
    tradeOffs: {
        pros: [
            { text: "DDGI underneath carries every bounce: two-room reads −1.5% against Cycles in RC3.6's run.", source: REGISTER_D68 },
            { text: 'SHaRC needs no volume and still carries every bounce: two-room reads −4.4% against Cycles. It holds 415 MiB at 1080p, against 1.4 GiB for the grid.', source: VARIANTS },
            { text: "GI-1.2's own rays need nothing from the host and warm up fastest: a median of 22 frames.", source: VARIANT_TRADE_OFFS },
        ],
        cons: [
            { text: "GI-1.2's own rays stop at two bounces: two-room reads −44% against Cycles, and the closed rooms measured read 25–62% dark.", source: VARIANTS },
            { text: 'DDGI underneath needs a volume around what the camera sees: outside it a hit gets no later bounce. DDGI adds 12–72 MiB and warms up more slowly (a median of 49 frames).', source: VARIANT_TRADE_OFFS },
            { text: 'SHaRC follows changes more slowly: 90% of a light change comes at a median of 83 frames, against 77 with DDGI underneath.', source: VARIANT_TRADE_OFFS },
            { text: "Don't stack DDGI under SHaRC: SHaRC doesn't read it, so the volume is pure cost. This page drops the volume when you pick SHaRC.", source: docsLink('api/radiance-cache.md#variants-compared', 'radiance-cache.md, "Variants compared"') },
        ],
    },
    api: API_PAGES.radianceCache,
    more: [docsLink('api/radiance-cache.md#stacking-ddgi-under-the-cache', 'Stacking DDGI under the cache'), docsLink('design/radiance-cache-variants.md', 'The world-cache variants compared (P6.5)')],
});
