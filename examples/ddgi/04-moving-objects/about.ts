/** What 04 shows and how it works: the page's explainer panel (shared/explainer.ts). Plain data; main.ts holds the example. */

import { API_PAGES, docsLink, showExplainer } from '../../shared/explainer.ts';

const HOW_LATE = docsLink('api/three.md#how-changes-are-found-and-how-late', 'three.md, "How changes are found"');

showExplainer({
    seeing: [
        'Two rooms joined by a doorway. A lamp lights the near room; the far room has no light of its own.',
        'A door swings open and shut, and a blue box slides across the near room. With the door shut the far room goes dark; open, light spills through the doorway.',
    ],
    how: [
        "Nothing is called: move objects as you always would. three's render finds a moved mesh, and the next GI update sends it.",
        'DDGI marks the probes near what moved and traces them again. A move also wakes the probes in the empty air around it, so they light what passes through.',
        "For an edit three's render doesn't see, `volume.threeGI.markChanged(object)` makes it count at once. Nothing here needs it.",
    ],
    tryThis: [
        'Watch the far room as the door shuts: it goes dark over a few frames, then lights again when the door opens.',
        'Watch the floor and wall near the blue box as it slides: the light around it changes with it.',
        'Read the end of the stats line: it says whether the door is open or shut.',
    ],
    tradeOffs: {
        pros: [{ text: "A still object's first move reaches GI in 2 frames; with `markChanged`, in the next update.", source: HOW_LATE }],
        cons: [{ text: "By default a move reaches GI one frame late, which DDGI's blending over many frames hides.", source: HOW_LATE }],
    },
    api: API_PAGES.three,
    more: [docsLink('api/three.md#moving-adding-and-removing-objects', 'Moving, adding and removing objects')],
});
