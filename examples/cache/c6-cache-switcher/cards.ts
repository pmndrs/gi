/**
 * One card per world-cache variant (`RADIANCE_CACHE_VARIANT_PRESETS`): how it
 * works, its pros and cons, P6.5's measured numbers and what it needs from
 * the host. Plain data, written from docs/design/radiance-cache-variants.md
 * and docs/api/radiance-cache.md; every number is from there, with its
 * source beside it. main.ts shows the card of the variant it runs.
 */

import { docsLink, parseInline, type ExplainerLink, type TradeOff } from '../../shared/explainer.ts';

const VARIANTS_DOC = 'design/radiance-cache-variants.md';
const SUMMARY = docsLink(`${VARIANTS_DOC}#summary`, 'radiance-cache-variants.md, "Summary"');
const TRADE_OFFS = docsLink(`${VARIANTS_DOC}#trade-offs`, 'radiance-cache-variants.md, "Trade-offs"');
/** The five VALID timing runs: Bistro exterior, 1280 × 720, RTX 3060 Laptop. */
const VALID_TIMING = docsLink(`${VARIANTS_DOC}#gpu-time`, 'radiance-cache-variants.md, "GPU time", the VALID split runs');
const API_VARIANTS = docsLink('api/radiance-cache.md#variants', 'radiance-cache.md, "Variants"');
const API_STACKING = docsLink('api/radiance-cache.md#stacking-ddgi-under-the-cache', 'radiance-cache.md, "Stacking DDGI under the cache"');
const API_NEURAL = docsLink('api/radiance-cache.md#the-neural-tail-and-pathtracermlp', 'radiance-cache.md, "The neural tail"');
const RECOMMENDATION = docsLink(`${VARIANTS_DOC}#recommendation`, 'radiance-cache-variants.md, "Recommendation"');

/** P6.5's numbers for one variant (radiance-cache-variants.md, "Summary"). */
export interface MeasuredNumbers {
    /** The size of the GI bias against Cycles, averaged over the ten views. */
    readonly meanBias: string;
    /** two-room's own bias (its `lit-room` view), signed. */
    readonly twoRoom: string;
    /** Median frames until the picture is 90% of the way there after the lights double. */
    readonly lights90: number;
    /** The same after a box moves (the old box's ghost). */
    readonly box90: number;
    /** Median frames from an empty cache until the picture's mean stays within 5%. */
    readonly warmUp: number;
    /** The radiance cache's GPU memory at 1920 × 1080, DDGI's apart. */
    readonly memory: string;
}

export interface VariantCard {
    /** The preset's id. */
    readonly id: string;
    readonly title: string;
    /** Where it stands: what ships, opt-in, planned, or not a candidate. */
    readonly standing: string;
    /** How it works: two or three sentences. */
    readonly how: string;
    readonly diagram: string;
    readonly pros: readonly TradeOff[];
    readonly cons: readonly TradeOff[];
    readonly measured: MeasuredNumbers;
    /** What it needs from the host. */
    readonly needs: string;
    readonly needsSource: ExplainerLink;
}

// ---- The diagrams: one room in section, a screen probe on the floor, its ray to a cell on the ceiling ----

const ORANGE = '#ffc878';
const BLUE = '#7dd3fc';
const GREEN = '#86efac';

function cell(x: number, y: number, filled = true): string {
    return `<rect x="${x - 6}" y="${y - 4}" width="12" height="8" fill="${ORANGE}" fill-opacity="${filled ? 0.45 : 0}" stroke="${ORANGE}"/>`;
}

function text(x: number, y: number, words: string, anchor: 'start' | 'end' = 'start'): string {
    return `<text x="${x}" y="${y}" text-anchor="${anchor}">${words}</text>`;
}

/** DDGI's probes, a grid of dots in the right half of the room. */
function ddgiProbes(faded = false): string {
    const dots: string[] = [];
    for (let x = 170; x <= 270; x += 25) for (let y = 30; y <= 70; y += 20) dots.push(`<circle cx="${x}" cy="${y}" r="2.2"/>`);
    return `<g fill="${GREEN}" fill-opacity="${faded ? 0.3 : 0.85}">${dots.join('')}</g>`;
}

/** A second-bounce ray from the first cell to the right wall, and its cell there. */
const SECOND_RAY = `<line x1="130" y1="10" x2="290" y2="50" stroke="${ORANGE}" stroke-dasharray="4 3"/>${cell(286, 50)}`;

function sketch(label: string, caption: string, body: string): string {
    return `<svg viewBox="0 0 300 104" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="${label}">
  <rect x="6" y="6" width="288" height="80" fill="none" stroke="currentColor" stroke-opacity="0.4"/>
  ${body}
  <line x1="40" y1="86" x2="130" y2="10" stroke="${ORANGE}" stroke-width="1.4"/>
  ${cell(130, 10)}
  <circle cx="40" cy="86" r="4" fill="${BLUE}"/>
  <g fill="currentColor" font-family="system-ui, sans-serif" font-size="9">
    ${text(10, 99, 'screen probe')}${text(98, 30, 'its cell', 'end')}${text(294, 99, caption, 'end')}
  </g>
</svg>`;
}

const LAST_FRAME = `<rect x="214" y="58" width="72" height="24" fill="none" stroke="${BLUE}" stroke-dasharray="3 2"/>
  <line x1="134" y1="14" x2="214" y2="66" stroke="${BLUE}" stroke-dasharray="2 2"/>
  <g fill="currentColor" font-family="system-ui, sans-serif" font-size="8">${text(219, 68, "last frame's")}${text(219, 78, 'picture')}</g>`;

const FEEDBACK_ARROW = `<path d="M 282 46 Q 210 30 140 12" fill="none" stroke="${GREEN}" stroke-width="1.2"/><path d="M 140 12 l 7 -1 l -3 6 z" fill="${GREEN}"/>`;

const DIAGRAMS = {
    gridAlone: sketch('A probe ray hits a cell on the ceiling; one more ray goes to the wall, and there it stops.', 'one more ray, then nothing: two bounces', SECOND_RAY),
    gridOverDdgi: sketch(
        "A probe ray hits a cell; the cell reads DDGI's probes nearby for every later bounce.",
        "later bounces read from DDGI's probes",
        `${ddgiProbes()}<line x1="136" y1="12" x2="170" y2="30" stroke="${GREEN}" stroke-dasharray="3 2"/><line x1="136" y1="12" x2="195" y2="30" stroke="${GREEN}" stroke-dasharray="3 2"/>`,
    ),
    sharc: sketch(
        'A probe ray hits a cell; a short path of two more rays follows it and ends in another cell.',
        'a short path, ended by a cell',
        `<polyline points="130,10 290,46 220,86" fill="none" stroke="${ORANGE}" stroke-dasharray="4 3"/>${cell(286, 46, false)}${cell(220, 82)}`,
    ),
    sharcOverDdgi: sketch(
        "SHaRC's short path as before; DDGI's probes are there but nothing reads them.",
        "DDGI's probes: not read",
        `${ddgiProbes(true)}<polyline points="130,10 290,46 220,86" fill="none" stroke="${ORANGE}" stroke-dasharray="4 3"/>${cell(286, 46, false)}${cell(220, 82)}`,
    ),
    budgeted: sketch(
        'A row of cells along the ceiling: only a few are lit this frame, the rest wait their turn.',
        'a budget of cells lit a frame; the rest wait',
        [0, 1, 2, 3, 4, 5, 6, 7, 8].map((index) => cell(160 + index * 15, 10, index === 1 || index === 5)).join(''),
    ),
    neural: sketch(
        "A probe ray hits a cell; one more ray reaches the wall, and a small network answers the bounces after it.",
        'a small network answers the later bounces',
        `${SECOND_RAY}<g stroke="${GREEN}" stroke-opacity="0.6">${[0, 1, 2].map((a) => [0, 1, 2].map((b) => `<line x1="215" y1="${48 + a * 10}" x2="240" y2="${48 + b * 10}"/><line x1="240" y1="${48 + a * 10}" x2="265" y2="${48 + b * 10}"/>`).join('')).join('')}</g><g fill="${GREEN}">${[215, 240, 265].map((x) => [48, 58, 68].map((y) => `<circle cx="${x}" cy="${y}" r="2.2"/>`).join('')).join('')}</g><line x1="270" y1="58" x2="284" y2="52" stroke="${GREEN}"/>`,
    ),
    feedback: sketch(
        "A probe ray hits a cell; the second ray's cell sends back the light it already holds, one bounce more each frame.",
        "the second hit's cell brings its own bounces",
        `${SECOND_RAY}${FEEDBACK_ARROW}`,
    ),
    temporal: sketch("A probe ray's hit was on screen last frame, so it reads last frame's lit picture there.", 'on-screen hits read the picture', `${SECOND_RAY}${LAST_FRAME}`),
    feedbackTemporal: sketch("Both feedbacks: the second hit brings its cell's light, and on-screen hits read last frame's picture.", 'both feedbacks', `${SECOND_RAY}${FEEDBACK_ARROW}${LAST_FRAME}`),
    ddgiTemporal: sketch("DDGI's probes give the later bounces; on-screen hits read last frame's picture instead.", "DDGI under it, and last frame's picture", `${ddgiProbes()}${LAST_FRAME}`),
};

// ---- The cards ----

const NOTHING = 'Nothing beyond the radiance cache itself.';

export const VARIANT_CARDS: readonly VariantCard[] = [
    {
        id: 'gi12-rays',
        title: 'GI-1.2 grid alone',
        standing: "Capsaicin's own way; opt-in",
        how: "Each probe ray's hit finds its cell in a world-space hash grid, and lights it there: one light sample and its shadow ray per hit. Some hits trace one more ray, whose light goes into the first hit's cell. So the cells carry two bounces, and nothing after them.",
        diagram: DIAGRAMS.gridAlone,
        pros: [
            { text: 'It needs nothing from the host, and it warms up fast: a median of 22 frames.', source: TRADE_OFFS },
            { text: "It is the faithful baseline, Capsaicin's GI-1.2.", source: TRADE_OFFS },
        ],
        cons: [
            { text: "Two bounces, and the cache's own lights leave out emitters: closed rooms read 25-62% dark against Cycles.", source: TRADE_OFFS },
            { text: "It holds the grid's 1.4 GiB at 1080p.", source: TRADE_OFFS },
            { text: 'Dearest of the five timed validly: 29.28 ms against 21.81 ms over DDGI (Bistro exterior, 1280 × 720).', source: VALID_TIMING },
        ],
        measured: { meanBias: '27.5%', twoRoom: '−43.8%', lights90: 68, box90: 79, warmUp: 22, memory: '1,419 MiB' },
        needs: NOTHING,
        needsSource: SUMMARY,
    },
    {
        id: 'gi12-ddgi',
        title: 'GI-1.2 grid over DDGI',
        standing: 'What ships, when the host has a DDGI volume',
        how: "The same grid with a DDGI volume stacked under the cache (`gi.setFallback(cache, ddgi)`). A cell's second bounce reads DDGI's light at the hit instead of tracing a ray, and DDGI's field already holds every later bounce and the light of emissive surfaces. This page's volume updates 256 probes a frame.",
        diagram: DIAGRAMS.gridOverDdgi,
        pros: [
            { text: 'The smallest bias on every view: 3.1% on average, −0.9% on two-room.', source: SUMMARY },
            { text: 'Its ghost clears soonest after a box moves (median 74 frames to 90%), and its brightness is the calmest when still.', source: TRADE_OFFS },
            { text: 'On big scenes DDGI replaces the second-bounce rays, so it costs less than the grid alone: 21.81 ms against 29.28 ms (Bistro exterior, 1280 × 720).', source: VALID_TIMING },
        ],
        cons: [
            { text: 'The host must place a volume around the scene. DDGI adds 12-72 MiB, and the grid keeps its 1.4 GiB.', source: TRADE_OFFS },
            { text: "Slower to warm up than the grid alone (median 49 frames): until DDGI's field fills, there is no later bounce.", source: TRADE_OFFS },
            { text: "A hit outside every volume gets no bounce after the first. In open scenes DDGI's coarse field blurs the bounce at contacts.", source: TRADE_OFFS },
        ],
        measured: { meanBias: '3.1%', twoRoom: '−0.9%', lights90: 77, box90: 74, warmUp: 49, memory: '1,373 MiB (+12-72 MiB DDGI)' },
        needs: "A DDGI volume around what the camera sees, stacked with `gi.setFallback` before `prepare`. This page uses the scene's suggested probe box.",
        needsSource: API_STACKING,
    },
    {
        id: 'sharc',
        title: 'SHaRC cache alone',
        standing: 'Opt-in; planned as the default when no volume is stacked',
        how: "pmndrs/pathtracer's SHaRC-style hash cache takes the grid's place (`worldCache: 'sharc'`). From each probe hit it follows a short path of two bounce rays and ends it in its own cells, so the cells carry every bounce without DDGI. A quarter of the probes update a frame, and each cell keeps a 32-frame mean.",
        diagram: DIAGRAMS.sharc,
        pros: [
            { text: 'It needs nothing from the host and carries every bounce, emitters included: 5.9% bias on average, 2-5 points behind the grid over DDGI.', source: TRADE_OFFS },
            { text: "Under a third of the grid's memory: 415 MiB at 1080p, 83 MiB of it the cache.", source: TRADE_OFFS },
            { text: "In the open scenes its block error ties or beats the grid over DDGI's.", source: TRADE_OFFS },
        ],
        cons: [
            { text: 'It follows changes more slowly: a median 83 frames to 90% after the lights double, against 77 over DDGI.', source: TRADE_OFFS },
            { text: "Its paths are the dearest world cache on big scenes: 14.24 ms against the grid's 7.09 ms over DDGI (Bistro exterior, 1280 × 720).", source: VALID_TIMING },
            { text: 'Its coarse cells blur the light near contacts.', source: TRADE_OFFS },
        ],
        measured: { meanBias: '5.9%', twoRoom: '−4.4%', lights90: 83, box90: 85, warmUp: 45, memory: '415 MiB' },
        needs: NOTHING,
        needsSource: API_VARIANTS,
    },
    {
        id: 'sharc-ddgi',
        title: 'SHaRC over DDGI',
        standing: 'Not a candidate',
        how: "SHaRC with a DDGI volume still stacked under the cache. SHaRC doesn't read the volume: only the GI-1.2 grid takes its second bounce from it. So the picture is SHaRC's, and the volume is pure cost. It is listed because the viewer stacks DDGI whatever the world cache is.",
        diagram: DIAGRAMS.sharcOverDdgi,
        pros: [{ text: "The same picture as SHaRC alone, to the noise: 5.9% bias on average.", source: SUMMARY }],
        cons: [
            { text: 'DDGI costs its time for nothing: 28.72 ms against 26.21 ms for SHaRC alone (Bistro exterior, 1280 × 720).', source: VALID_TIMING },
            { text: "Don't stack DDGI under SHaRC: if SHaRC is used, leave the volume out.", source: RECOMMENDATION },
        ],
        measured: { meanBias: '5.9%', twoRoom: 'as SHaRC (−4.4%)', lights90: 83, box90: 86, warmUp: 45, memory: '415 MiB (+ DDGI)' },
        needs: 'A DDGI volume, which it never reads.',
        needsSource: SUMMARY,
    },
    {
        id: 'budgeted',
        title: 'Budgeted cells over DDGI',
        standing: 'Opt-in, for big scenes',
        how: "id Tech 8's way, on the grid over DDGI (`cellShading: 'budgeted'`). A hit only finds its cell. Then at most 20,000 cells a frame are lit, one light sample each, new and changed cells first: fewer shadow rays, and each cell is noisier and older.",
        diagram: DIAGRAMS.budgeted,
        pros: [
            { text: 'Its accuracy, flicker and memory match the grid over DDGI: 3.4% bias on average, −0.8% on two-room.', source: SUMMARY },
            { text: 'The cheapest variant on big scenes: 20.78 ms against 21.81 ms per hit (Bistro exterior, 1280 × 720).', source: VALID_TIMING },
        ],
        cons: [
            { text: 'It reacts more slowly to light changes: 4 frames after the lights double, a median 98% of the change is still missing, against 91% for the grid over DDGI.', source: TRADE_OFFS },
            { text: 'On small scenes it costs more than per hit, since sorting and claiming cells is a fixed cost.', source: TRADE_OFFS },
        ],
        measured: { meanBias: '3.4%', twoRoom: '−0.8%', lights90: 86, box90: 75, warmUp: 54, memory: '1,379 MiB (+ DDGI)' },
        needs: 'A DDGI volume, as for the grid over DDGI.',
        needsSource: API_VARIANTS,
    },
    {
        id: 'neural-tail',
        title: 'GI-1.2 grid with the neural tail',
        standing: 'Opt-in; needs the optional package',
        how: "The grid alone on its own rays, plus a small neural network (`neuralCache: 'tail'`). The network learns the cells' indirect light as the frames go by and answers the bounces past the second, so the cells carry every bounce without DDGI.",
        diagram: DIAGRAMS.neural,
        pros: [
            { text: 'It needs no DDGI volume, and in rooms lit by lights it closes most of the grid alone\'s gap: two-room −44% → −7%.', source: TRADE_OFFS },
        ],
        cons: [
            { text: "It can't light an emissive scene: the emissive Cornell box reads −49%, as the grid alone.", source: TRADE_OFFS },
            { text: 'It warms up slowest in the rooms lit by lights (138-214 frames), because the network trains first, and it pumps when still, as it trains.', source: TRADE_OFFS },
            { text: 'It is the most expensive variant: the network runs on top of the grid alone\'s rays.', source: TRADE_OFFS },
        ],
        measured: { meanBias: '11.3%', twoRoom: '−7.0%', lights90: 84, box90: 82, warmUp: 31, memory: '1,453 MiB' },
        needs: "The optional `@pathtracer/mlp` package, which isn't on npm and which this site doesn't bundle, and `shader-f16` for full speed (f32 works, slower). Without the package the tail turns off with a warning and the cache runs as the grid alone.",
        needsSource: API_NEURAL,
    },
    {
        id: 'gi12-rays-feedback',
        title: 'Grid alone, cache feedback',
        standing: 'Opt-in (RC3.6)',
        how: "The grid alone on its own rays (`cacheFeedback: true`). A second-bounce hit also brings the indirect light its own cell already holds, so each frame adds one more bounce, with no extra rays or memory.",
        diagram: DIAGRAMS.feedback,
        pros: [
            { text: "It needs nothing from the host and halves the grid alone's error in rooms lit by lamps: two-room −44% → −20%.", source: TRADE_OFFS },
            { text: 'Open scenes are unchanged.', source: TRADE_OFFS },
        ],
        cons: [
            { text: 'Closed rooms stay 20-40% dark, well behind SHaRC and DDGI, and an emitter-lit room as dark as the grid alone.', source: TRADE_OFFS },
            { text: 'It warms up slowly in the rooms (88-145 frames), since each frame adds one bounce.', source: TRADE_OFFS },
        ],
        measured: { meanBias: '17.6%', twoRoom: '−19.7%', lights90: 82, box90: 80, warmUp: 24, memory: '1,419 MiB' },
        needs: NOTHING,
        needsSource: API_VARIANTS,
    },
    {
        id: 'gi12-rays-temporal',
        title: "Grid alone, last frame's picture at hits",
        standing: 'Opt-in (RC3.6)',
        how: "The grid alone on its own rays (`useTemporalFeedback: true`). A probe ray whose hit was on screen last frame reads last frame's lit picture there instead of the cache, so it gets every bounce that picture shows.",
        diagram: DIAGRAMS.temporal,
        pros: [{ text: "It costs no rays, and in the open scenes it removes the grid alone's small darkening (Bistro exterior +0.5 to +1.5%).", source: TRADE_OFFS }],
        cons: [
            { text: "It over-brightens where last frame's picture shows lamps or emitters: Bistro interior +14 to +23%.", source: TRADE_OFFS },
            { text: 'It barely helps closed rooms whose lights are off screen: two-room −33%.', source: TRADE_OFFS },
            { text: "It feeds the picture's own errors back into the light.", source: TRADE_OFFS },
        ],
        measured: { meanBias: '19.2%', twoRoom: '−32.7%', lights90: 82, box90: 84, warmUp: 51, memory: '1,419 MiB' },
        needs: "The host's lit picture and albedo, which three's `screenInputs` gives. It is read when the technique is made, so changing it makes the technique again.",
        needsSource: TRADE_OFFS,
    },
    {
        id: 'gi12-rays-feedback-temporal',
        title: 'Grid alone, both feedbacks',
        standing: 'Opt-in (RC3.6)',
        how: "Both of RC3.6's feedbacks on the grid alone: second hits bring their cells' light, and probe rays landing on last frame's screen read its lit picture.",
        diagram: DIAGRAMS.feedbackTemporal,
        pros: [{ text: 'Its mean bias, 14.4%, is below either feedback alone (17.6% and 19.2%).', source: SUMMARY }],
        cons: [
            { text: 'It over-brightens Bistro interior (+22.0% and +14.8%) and leaves two-room 18.9% dark.', source: SUMMARY },
            { text: 'Its warm-up is the slowest median in the table: 61 frames.', source: SUMMARY },
        ],
        measured: { meanBias: '14.4%', twoRoom: '−18.9%', lights90: 86, box90: 81, warmUp: 61, memory: '1,419 MiB' },
        needs: "The host's lit picture and albedo, as for temporal feedback alone.",
        needsSource: TRADE_OFFS,
    },
    {
        id: 'gi12-ddgi-temporal',
        title: "Grid over DDGI, last frame's picture at hits",
        standing: 'Opt-in (RC3.6)',
        how: "What ships, plus temporal feedback: probe rays landing on last frame's screen read its lit picture instead of the cache. Cache feedback would add nothing here, as DDGI already carries every bounce.",
        diagram: DIAGRAMS.ddgiTemporal,
        pros: [{ text: 'On two-room it stays close to what ships: −2.3%.', source: SUMMARY }],
        cons: [
            { text: 'Over DDGI it is worse than DDGI alone on every view: 10.1% bias on average, against 3.1%.', source: TRADE_OFFS },
            { text: "It reads the emissive Cornell box 26% too bright, and it pumps on one Bistro exterior view.", source: TRADE_OFFS },
        ],
        measured: { meanBias: '10.1%', twoRoom: '−2.3%', lights90: 84, box90: 78, warmUp: 50, memory: '1,373 MiB (+ DDGI)' },
        needs: "A DDGI volume, and the host's lit picture and albedo.",
        needsSource: SUMMARY,
    },
];

/** The card for a preset id. */
export function variantCard(id: string): VariantCard | undefined {
    return VARIANT_CARDS.find((card) => card.id === id);
}

// ---- Drawing a card ----

function element<K extends keyof HTMLElementTagNameMap>(tag: K, className?: string, words?: string): HTMLElementTagNameMap[K] {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (words !== undefined) {
        for (const part of parseInline(words)) {
            if (part.kind === 'text') node.append(part.text);
            else if (part.kind === 'code') node.append(Object.assign(document.createElement('code'), { textContent: part.text }));
            else node.append(link({ label: part.text, href: part.href }));
        }
    }
    return node;
}

function link(target: ExplainerLink): HTMLAnchorElement {
    return Object.assign(document.createElement('a'), { href: target.href, textContent: target.label, target: '_blank', rel: 'noopener' });
}

function source(target: ExplainerLink): HTMLElement {
    const line = element('span', 'card-source', 'Source: ');
    line.append(link(target));
    return line;
}

/** The card as DOM: title, standing, how, diagram, measured numbers, pros and cons, needs. */
export function renderCard(card: VariantCard): HTMLElement {
    const root = element('article', 'variant-card');
    root.append(element('h3', undefined, card.title), element('p', 'card-standing', card.standing), element('p', undefined, card.how));

    const figure = element('figure', 'card-diagram');
    // The page's own SVG, never a visitor's: set as markup.
    figure.innerHTML = card.diagram;
    root.append(figure);

    const numbers = element('table', 'card-numbers');
    const rows: Array<[string, string]> = [
        ['Bias against Cycles, mean size over ten views', card.measured.meanBias],
        ['On two-room', card.measured.twoRoom],
        ['Frames to 90% after the lights double', String(card.measured.lights90)],
        ['Frames to 90% after a box moves', String(card.measured.box90)],
        ['Warm-up from empty, frames', String(card.measured.warmUp)],
        ['Memory at 1920 × 1080', card.measured.memory],
    ];
    for (const [label, value] of rows) {
        const row = element('tr');
        row.append(element('td', undefined, label), element('td', 'value', value));
        numbers.append(row);
    }
    const caption = element('caption', undefined, "P6.5's measurements, RTX 3060 Laptop. Frames are medians over the views; 60 frames is one second at 60 fps.");
    numbers.prepend(caption);
    root.append(numbers, source(SUMMARY));

    const table = element('table', 'card-tradeoffs');
    for (const [items, kind] of [
        [card.pros, 'pro'],
        [card.cons, 'con'],
    ] as const) {
        for (const item of items) {
            const row = element('tr', kind);
            const sign = element('td', 'sign', kind === 'pro' ? '+' : '−');
            sign.setAttribute('aria-label', kind === 'pro' ? 'For' : 'Against');
            const words = element('td', undefined, item.text);
            if (item.source) words.append(source(item.source));
            row.append(sign, words);
            table.append(row);
        }
    }
    root.append(table);

    const needs = element('p', 'card-needs');
    needs.append(element('strong', undefined, 'Needs from the host: '), element('span', undefined, card.needs), source(card.needsSource));
    root.append(needs);
    return root;
}
