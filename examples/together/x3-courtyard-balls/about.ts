/** What X3 shows and how it works: the page's explainer panel (shared/explainer.ts). Plain data; main.ts holds the example. */

import { API_PAGES, docsLink, showExplainer } from '../../shared/explainer.ts';

const HOW_LATE = docsLink('api/three.md#how-changes-are-found-and-how-late', 'three.md, "How changes are found"');
const DDGI_MOVES = docsLink('design/ddgi.md#moving-geometry-t2m1-m3-movingmd-1-and-5-rigid-objects-as-built', 'design/ddgi.md, "Moving geometry"');
const DEFAULTS = docsLink('api/radiance-cache.md#the-defaults-that-matter', 'radiance-cache.md, "The defaults that matter"');
const LIMITS = docsLink('api/radiance-cache.md#known-limits', 'radiance-cache.md, "Known limits"');
const SUMMARY = docsLink('design/radiance-cache-variants.md#summary', 'radiance-cache-variants.md, "Summary"');
const TRADE_OFFS = docsLink('design/radiance-cache-variants.md#trade-offs', 'radiance-cache-variants.md, "Trade-offs"');
const LAG = docsLink('design/radiance-cache-variants.md#where-the-lag-comes-from', 'radiance-cache-variants.md, "Where the lag comes from"');

// A ball rolls from where it was to where it lands: its old light and shadow fade at the old place, new light fills in at the new one.
const DIAGRAM = `<svg viewBox="0 0 300 120" width="300" height="120" role="img" aria-label="A glowing ball moves from one place to another. Where it was, its light fades over frames; where it lands, its light fills in over frames." fill="none" stroke="currentColor" font-size="10" font-family="system-ui, sans-serif">
  <line x1="10" y1="96" x2="290" y2="96" opacity="0.5"/>
  <circle cx="70" cy="76" r="20" stroke-dasharray="3 3" opacity="0.5"/>
  <circle cx="220" cy="76" r="20" fill="#ffc878" fill-opacity="0.35" stroke="#ffc878"/>
  <path d="M92 70 C130 40 170 40 198 66" stroke-dasharray="4 3" opacity="0.7"/>
  <path d="M194 60 L199 67 L190 68" opacity="0.7"/>
  <ellipse cx="70" cy="99" rx="38" ry="4" fill="#ffc878" fill-opacity="0.15" stroke="none"/>
  <ellipse cx="220" cy="99" rx="44" ry="5" fill="#ffc878" fill-opacity="0.4" stroke="none"/>
  <text x="22" y="20" fill="currentColor" stroke="none">old light: fades</text>
  <text x="22" y="32" fill="currentColor" stroke="none" opacity="0.7">(lag, or a ghost)</text>
  <text x="178" y="20" fill="currentColor" stroke="none">new light: fills in</text>
  <text x="178" y="32" fill="currentColor" stroke="none" opacity="0.7">over frames</text>
  <text x="18" y="114" fill="currentColor" stroke="none" opacity="0.8">GI on the floor, a few frames after the move</text>
</svg>`;

showExplainer({
    seeing: [
        "Sponza's atrium, the palace courtyard with two storeys of galleries round it. Balls 1 to 2 m across come in over a gallery's railing or the roof's edge, drop to the floor, bounce and roll; some roll on in under the ground-floor arcade at the ends. Half of them glow in a random colour, and some of those pulse or flicker. The rest have a random PBR material: colour, metalness and roughness. The physics is this page's own small sphere solver.",
        "Sponza is Crytek's model, from Khronos's glTF samples, fetched from GitHub when the page opens; the site doesn't ship it. Offline, the page uses the repo's small procedural courtyard instead, and says so bottom left.",
        'The chart, bottom right, is the mean brightness of the picture over the last 240 frames, with a marker at each throw. It is this page\'s own reading, for whichever technique runs. "Settled in N frames" counts the frames until the mean stays within 3% of where it ends up, for 20 frames. It includes the ball\'s flight and bounces, not only GI catching up.',
    ],
    how: [
        "Nothing special is done for GI. A ball is an ordinary three mesh: `scene.add` puts it in, and setting its position moves it. The adapter finds both in three's render. A glowing ball's `emissiveIntensity` changes every frame, and `threeGI.markChanged(ball)` has GI see each edit in its next update. A ball that has settled isn't touched, so GI sees it still.",
        'DDGI keeps light in a box of probes, 1 m apart here (0.5 m in the small courtyard). A moved ball marks the probes near it, and the farther ones that see it, for updates. Each update blends new rays into what the probe had, so the light catches up over a few frames. A glowing ball that moves marks every probe its light can reach, seen or not, because its light also arrives by bouncing.',
        'The radiance cache keeps light in probes on the visible surfaces and in a world cache of cells. Where the scene changed, a cell a ray hits drops its history once (`cellChangeReset`), and the screen probes and the denoiser shorten theirs near the change. Nothing in the cache moves with a ball: its light is learnt again where it lands, and what it left behind fades.',
        "A glowing ball's light needs rays that reach it. The GI-1.2 grid's cells have no emitter lights, so on its own it gets the balls' light only from probe rays that hit them. With DDGI under the grid, DDGI's rays see the balls and carry their light into every bounce. SHaRC's update paths take emission where they hit, so its cells carry it too.",
    ],
    diagram: DIAGRAM,
    diagramCaption: "A ball moves: where it was, its light and shadow fade; where it lands, its light fills in. How many frames each takes is the technique's lag. A sketch, not this page's data.",
    tryThis: [
        'Right-click the floor: a ball comes over the railing and lands near there. On the chart, the picture brightens when a glowing ball comes into view, then GI catches up after it. Read the settle count.',
        'Switch `GI` between `DDGI` and `Cache: GI-1.2 grid over DDGI`, and throw a glowing ball each time. Watch its coloured light arrive on the floor and the arches, and fade from where the ball rolls away.',
        'Untick `glowing balls glow`. The coloured light leaves the walls and floor over a few frames; each technique fades it at its own pace.',
        'Tick `freeze the balls` with a few pulsing ones in view. Nothing moves, but their light keeps changing, and GI follows it through `markChanged`.',
    ],
    tradeOffs: {
        pros: [
            { text: "A ball's first move is sent to GI 2 frames after it starts; an added ball in the next update, and rays hit it 2 frames later, once its tree is built.", source: HOW_LATE },
            {
                text: 'DDGI: a glowing object that moves updates every probe its light reaches. Without that, a room lit only by an emissive panel stayed 6% (floor) and 15% (wall) too dark; with it, within 0.5%.',
                source: DDGI_MOVES,
            },
            { text: "The cache's change reset: a cell follows a doubled sun in 2 frames, against 37 without it.", source: DEFAULTS },
            { text: "Of the cache's variants, the grid over DDGI clears a moved box's ghost soonest: 90% of the way in 74 frames (median), budgeted cells 75, SHaRC 85.", source: SUMMARY },
        ],
        cons: [
            { text: 'DDGI: after a 1 m slide, the pixels it changed were within 5% of settled from frame 15 to 29 (at 358 and 580 probes a frame).', source: DDGI_MOVES },
            { text: 'No cache entry moves with its object yet: the light near a moved object is learnt again where it lands.', source: LIMITS },
            { text: "The denoiser that keeps the cache's still picture calm adds about 25 frames to a moved box's ghost (two-room).", source: LAG },
            {
                text: "Emitters aren't lights for the cache's cells. On the emissive Cornell box the GI-1.2 grid alone reads −49.1% against Cycles, over DDGI −1.8%, and SHaRC −3.8%.",
                source: SUMMARY,
            },
            { text: 'Budgeted cells react more slowly: 4 frames after the lights double, 98% of the change is still missing (median), against 91% for the grid over DDGI.', source: TRADE_OFFS },
            { text: "Memory at 1080p: the cache holds 1,373 MiB with the GI-1.2 grid and 415 MiB with SHaRC, DDGI's not counted.", source: SUMMARY },
        ],
    },
    api: API_PAGES.three,
    more: [API_PAGES.radianceCache, docsLink('api/three.md#moving-adding-and-removing-objects', 'Moving, adding and removing objects'), SUMMARY],
});
