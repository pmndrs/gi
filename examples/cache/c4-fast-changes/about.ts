/** What C4 shows and how it works: the page's explainer panel (shared/explainer.ts). Plain data; main.ts holds the example. */

import { API_PAGES, docsLink, showExplainer } from '../../shared/explainer.ts';

const E9 = docsLink('design/radiance-cache.md#experiment-e9-per-hit-change-reset-p62b', 'design/radiance-cache.md, "Experiment E9"');
const TASKS_P62B = { label: 'TASKS.md, P6.2b', href: 'https://github.com/pmndrs/gi/blob/main/TASKS.md' };
const DEFAULTS = docsLink('api/radiance-cache.md#the-defaults-that-matter', 'radiance-cache.md, "The defaults that matter"');
const MEMORY = docsLink('api/radiance-cache.md#memory', 'radiance-cache.md, "Memory"');
const NOTES = docsLink('NOTES.md#radiance-cache-measured-lessons', 'NOTES.md, "Radiance cache (measured lessons)"');
const LIMITS = docsLink('api/radiance-cache.md#known-limits', 'radiance-cache.md, "Known limits"');

// One cell's light after a lamp is switched off at the marker: without the reset the old light fades over frames; with it, it drops at once.
const DIAGRAM = `<svg viewBox="0 0 260 112" width="260" height="112" role="img" aria-label="A cell's light after a switch, with and without the change reset" fill="none" stroke="currentColor" font-size="10" font-family="system-ui, sans-serif">
  <line x1="20" y1="90" x2="250" y2="90" opacity="0.5"/>
  <line x1="20" y1="10" x2="20" y2="90" opacity="0.5"/>
  <line x1="80" y1="10" x2="80" y2="90" stroke-dasharray="3 3" opacity="0.6"/>
  <text x="84" y="18" fill="currentColor" stroke="none">light off</text>
  <path d="M20 28 H80 C110 72 150 86 245 88" stroke-width="2" opacity="0.5"/>
  <path d="M80 28 V86 C90 88 110 88 245 88" stroke-width="2"/>
  <text x="140" y="66" fill="currentColor" stroke="none" opacity="0.7">no reset: old light fades</text>
  <text x="90" y="106" fill="currentColor" stroke="none">reset: follows at once</text>
</svg>`;

showExplainer({
    seeing: [
        'A closed room with no sky, lit only by a ceiling lamp and a flashlight on the camera. A red box stands on the floor. The radiance cache runs on its default settings.',
        'The chart, bottom right, is the mean GI light on screen over the last 120 frames, with a marker at each change. "Settled in N frames" is this page\'s own reading: the frames until the mean stays within 5% of where it ends up, for 10 frames.',
        'The ghost view shows how far the GI light is from its average over about the last 8 frames, as a share of it. Where GI has settled it is black: differences under 4%, the noise, stay black, and a quarter is white. Light still fading out, or still filling in, shows.',
    ],
    how: [
        "Each frame's rays are noisy, so every cache keeps history: it blends this frame's light into what it saw before. That keeps the picture calm.",
        'History is also lag. A world-cache cell blends each frame into up to 16 samples of history, so after a change the old light leaves a little at a time. "Frames to settle" is how long the picture takes to stop moving after a change.',
        'Ghosting is old light left behind. Drag the box, and its shadow and its red bounce light stay where it was for a while, then fade.',
        "The change reset (`cellChangeReset`, on by default) works from what the adapter sees change: a light switched, an object moved. A world-cache cell hit near a change drops its history once and takes that frame's light whole, so it follows at once.",
        'The screen probes and the denoiser keep history of their own, with resets of their own. They set the pace of the picture, so the picture follows slower than the cells.',
    ],
    diagram: DIAGRAM,
    diagramCaption: "One world-cache cell after a light is switched off. A sketch, not this page's data.",
    tryThis: [
        'Untick `room light`. The chart falls after the marker. Read the settle count under it.',
        "Untick `change resets`, then switch `room light` off and on again. Switching it on settles slower without the reset: this page's own reading on an RTX 3060 laptop, three runs each, was 85 to 89 frames with it and 126 to 134 without. Switching it off showed no clear difference: the cells follow at once, but the screen probes and the denoiser set the picture's pace. The cells' own numbers are under Trade-offs.",
        'Tick `ghost view`, then drag the red box somewhere else. Its old shadow and bounce light show for a moment. With `change resets` off they show for longer.',
        'Orbit the camera with `flashlight` on. The light it bounces follows the beam a few frames behind.',
    ],
    tradeOffs: {
        pros: [
            { text: 'A world-cache cell follows a doubled sun in 2 frames with the reset, against 37 without it.', source: TASKS_P62B },
            { text: 'A light switched off: 20 frames against 42, to within 5%.', source: E9 },
            { text: 'The picture follows sooner too: on two-room it is within 5% of settled after 60 frames with the reset, 108 without.', source: E9 },
        ],
        cons: [
            { text: 'Noise rises for a short while after a change: on Sponza from 3.40% to 4.08% frame to frame.', source: DEFAULTS },
            { text: "32 MiB for each cell's last-reset frame.", source: MEMORY },
            { text: 'The picture follows slower than the cells. With the reset, the cells follow a doubled sun in 2 frames, but 16 frames after a change the picture reaches only 0.86 to 0.90 of settled on Sponza, and 0.79 to 0.83 on two-room.', source: NOTES },
            { text: 'No cache entry moves with its object yet: the light near a moved object is learnt again where it lands.', source: LIMITS },
        ],
    },
    api: API_PAGES.radianceCache,
    more: [E9],
});
