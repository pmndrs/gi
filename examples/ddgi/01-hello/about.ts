/** What 01 shows and how it works: the page's explainer panel (shared/explainer.ts). Plain data; main.ts holds the example. */

import { API_PAGES, docsLink, showExplainer } from '../../shared/explainer.ts';

/** A room with probes on a grid: one probe's rays, and a surface point reading the probes around it. */
const DIAGRAM = `<svg viewBox="0 0 320 150" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="A grid of probes in a room. One probe sends rays to the walls; a point on the floor reads the probes around it.">
  <rect x="10" y="10" width="300" height="118" fill="none" stroke="currentColor" stroke-opacity="0.5"/>
  <g stroke="#ffc878" stroke-width="1.2">
    <line x1="100" y1="69" x2="10" y2="36"/><line x1="100" y1="69" x2="58" y2="10"/><line x1="100" y1="69" x2="150" y2="10"/>
    <line x1="100" y1="69" x2="10" y2="110"/><line x1="100" y1="69" x2="78" y2="128"/><line x1="100" y1="69" x2="170" y2="128"/>
  </g>
  <g stroke="currentColor" stroke-dasharray="3 2" stroke-opacity="0.8">
    <line x1="235" y1="128" x2="220" y2="104"/><line x1="235" y1="128" x2="280" y2="104"/>
  </g>
  <g fill="currentColor">
    <circle cx="40" cy="34" r="4"/><circle cx="100" cy="34" r="4"/><circle cx="160" cy="34" r="4"/><circle cx="220" cy="34" r="4"/><circle cx="280" cy="34" r="4"/>
    <circle cx="40" cy="69" r="4"/><circle cx="160" cy="69" r="4"/><circle cx="220" cy="69" r="4"/><circle cx="280" cy="69" r="4"/>
    <circle cx="40" cy="104" r="4"/><circle cx="100" cy="104" r="4"/><circle cx="160" cy="104" r="4"/><circle cx="220" cy="104" r="4"/><circle cx="280" cy="104" r="4"/>
  </g>
  <circle cx="100" cy="69" r="5" fill="#ffc878"/>
  <circle cx="235" cy="128" r="4" fill="none" stroke="#7dd3fc" stroke-width="2"/>
  <g fill="currentColor" font-family="system-ui, sans-serif" font-size="10">
    <text x="108" y="64">a probe and its rays</text>
    <text x="312" y="146" text-anchor="end">a surface reads the probes near it</text>
  </g>
</svg>`;

showExplainer({
    seeing: [
        "The scene of three's own probe-grid example: a room with a red and a green wall, two boxes, a sphere and one point light. Drag to orbit.",
        'The red and green walls bounce their colour onto the boxes and the ceiling. That bounce light is GI.',
    ],
    how: [
        "A `DDGIVolume` is a box of probes on a regular grid: here 6 × 6 × 6 probes in the 6 × 5 × 6 m room. It takes the place of three's `LightProbeGrid`, line for line.",
        'Each probe traces rays in every direction. It stores the light arriving from each direction, and how far it can see in each, so light leaks little through walls.',
        'Every frame a budget of probes traces again and blends the result into what it had. A surface reads the probes around it.',
        'After `start`, GI updates itself inside `renderer.render`: there is no per-frame call.',
    ],
    diagram: DIAGRAM,
    tryThis: [
        'Orbit to the red wall: the side of the tall box that faces it turns pink. That is light bounced off the wall.',
        'Look at the floor in the shadows of the boxes: it is not black. The lamp cannot see it, so its light there is all bounce light.',
        'Open the source (Source, in the bar at the top): the GI is four lines, the same shape as `LightProbeGrid`.',
    ],
    tradeOffs: {
        pros: [
            { text: 'The shortest set-up, and it runs on the widest range of GPUs.', source: docsLink('guides/ddgi-or-radiance-cache.md#which-to-use', 'the guide, "Which to use"') },
            { text: 'Its GPU work stays inside a budget you set: 2 ms a frame by default.', source: docsLink('api/three.md#ddgivolume', 'three.md, "DDGIVolume"') },
        ],
        cons: [
            { text: 'Detail is limited by the probe spacing: light that changes over less than the spacing (a doorway, a corner) blurs.', source: docsLink('api/platform.md#quality-presets', 'platform.md, "Quality presets"') },
            { text: 'Memory grows with the probes: 55 MB on Sponza with 19,425 probes.', source: docsLink('guides/ddgi-or-radiance-cache.md#how-they-differ', 'the guide, "How they differ"') },
        ],
    },
    api: API_PAGES.three,
    more: [docsLink('design/ddgi.md#in-one-paragraph', 'DDGI in one paragraph')],
});
