/**
 * A small live chart of one value over the last N frames, with a marker at
 * each change (docs/plans/examples.md, C4 and X3): push a value every
 * frame, call `mark` when the page changes something (a light switched, a
 * ball thrown), and the chart shows how the value follows.
 *
 * It also reads "frames to settle" after the latest marker, by a rule of
 * the page's own (`framesToSettle`): the frames until the value stays
 * within a tolerance of where it ends up. That is this page's readout, not
 * a number from docs/, and the chart says so.
 *
 * Drawn as SVG, so the page's only canvas stays three's (the site check
 * reads the first canvas it finds).
 */

import './timeline.css';

export interface TimelineOptions {
    /** What the value is, shown above the chart. */
    readonly label: string;
    /** Frames the chart holds. Default 120. */
    readonly frames?: number;
    /** How a value is written. Default three significant digits. */
    readonly format?: (value: number) => string;
    /** The chart's lowest and highest value. Default 0 to the largest value shown, plus a tenth. */
    readonly range?: readonly [number, number];
    /** Relative tolerance for "frames to settle". Default 0.02 (2%). */
    readonly settleTolerance?: number;
    /** Frames the value must stay within the tolerance to count as settled. Default 10. */
    readonly settleHold?: number;
    /** Where the chart goes. Default a box of its own, bottom right, above the stats line. */
    readonly container?: HTMLElement;
}

export interface Timeline {
    readonly element: HTMLElement;
    /** Adds this frame's value (null for a frame with no reading: a gap). */
    push(value: number | null): void;
    /** Puts a marker at the latest frame, named `label` ("lamp off"). */
    mark(label?: string): void;
    /** Frames from the latest marker until the value settled, or null (no marker, or not settled yet). */
    readonly settleFrames: number | null;
    /** Empties the chart. */
    clear(): void;
    dispose(): void;
}

/**
 * Frames from `markIndex` until `values` settle: the first index from which
 * every later value is within `tolerance` (relative) of the last value, with
 * at least `hold` values after it. Null while that isn't so yet, or with a
 * null after the marker. Exported for its test.
 */
export function framesToSettle(values: readonly (number | null)[], markIndex: number, tolerance: number, hold: number): number | null {
    if (markIndex < 0 || markIndex >= values.length) return null;
    const final = values[values.length - 1];
    if (final === null) return null;
    const band = Math.abs(final) * tolerance;
    let first = values.length;
    for (let index = values.length - 1; index >= markIndex; index--) {
        const value = values[index];
        if (value === null || Math.abs(value - final) > band) break;
        first = index;
    }
    if (first >= values.length || values.length - first < hold) return null;
    return first - markIndex;
}

const SVG = 'http://www.w3.org/2000/svg';
const WIDTH = 240;
const HEIGHT = 64;
/** A marker's frames are kept for the settle reading for at most this many charts' lengths. */
const SETTLE_WINDOW = 10;

function svg<K extends keyof SVGElementTagNameMap>(tag: K, attributes: Record<string, string | number> = {}): SVGElementTagNameMap[K] {
    const node = document.createElementNS(SVG, tag);
    for (const [name, value] of Object.entries(attributes)) node.setAttribute(name, String(value));
    return node;
}

/** Shows a timeline chart. */
export function createTimeline(options: TimelineOptions): Timeline {
    const capacity = Math.max(2, Math.floor(options.frames ?? 120));
    const format = options.format ?? ((value: number) => value.toPrecision(3));
    const tolerance = options.settleTolerance ?? 0.02;
    const hold = options.settleHold ?? 10;

    const box = document.createElement('div');
    box.className = options.container ? 'timeline' : 'timeline timeline-floating';
    const header = document.createElement('div');
    header.className = 'timeline-header';
    const title = document.createElement('span');
    title.textContent = options.label;
    const reading = document.createElement('span');
    reading.className = 'timeline-reading';
    header.append(title, reading);

    const chart = svg('svg', { viewBox: `0 0 ${WIDTH} ${HEIGHT}`, preserveAspectRatio: 'none', role: 'img' });
    chart.setAttribute('aria-label', `${options.label} over the last ${capacity} frames`);
    const markers = svg('g', { class: 'timeline-markers' });
    const trace = svg('polyline', { class: 'timeline-trace', fill: 'none' });
    chart.append(markers, trace);
    // The markers' names are HTML over the chart: SVG text would stretch with its viewBox.
    const plot = document.createElement('div');
    plot.className = 'timeline-plot';
    const labels = document.createElement('div');
    labels.className = 'timeline-marker-labels';
    plot.append(chart, labels);

    const footer = document.createElement('div');
    footer.className = 'timeline-footer';
    const span = document.createElement('span');
    span.textContent = `last ${capacity} frames`;
    const settle = document.createElement('span');
    settle.title = "This page's own reading: frames from the latest marker until the value stays within " + `${Math.round(tolerance * 100)}% of where it ends up, for ${hold} frames.`;
    footer.append(span, settle);
    box.append(header, plot, footer);
    (options.container ?? document.body).append(box);

    // Every frame since the latest marker is kept for the settle reading; the chart shows the last `capacity`.
    let values: (number | null)[] = [];
    let marks: Array<{ index: number; label: string }> = [];
    let settleFrames: number | null = null;
    let dirty = false;

    const draw = (): void => {
        dirty = false;
        const start = Math.max(0, values.length - capacity);
        const shown = values.slice(start);
        const low = options.range?.[0] ?? 0;
        let high = options.range?.[1] ?? 0;
        if (!options.range) for (const value of shown) if (value !== null) high = Math.max(high, value * 1.1);
        if (!(high > low)) high = low + 1;
        const x = (index: number): number => ((index - start) / (capacity - 1)) * WIDTH;
        const y = (value: number): number => HEIGHT - ((Math.min(high, Math.max(low, value)) - low) / (high - low)) * HEIGHT;
        const points: string[] = [];
        shown.forEach((value, offset) => {
            if (value !== null) points.push(`${x(start + offset).toFixed(1)},${y(value).toFixed(1)}`);
        });
        trace.setAttribute('points', points.join(' '));

        markers.replaceChildren();
        labels.replaceChildren();
        for (const mark of marks) {
            if (mark.index < start) continue;
            const markX = x(mark.index);
            markers.append(svg('line', { x1: markX, x2: markX, y1: 0, y2: HEIGHT, class: 'timeline-marker' }));
            if (mark.label) {
                const text = document.createElement('span');
                text.textContent = mark.label;
                const percent = (markX / WIDTH) * 100;
                // Right of the marker, or left of it near the right edge.
                text.style.cssText = percent > 70 ? `right:${(100 - percent).toFixed(1)}%` : `left:${percent.toFixed(1)}%`;
                labels.append(text);
            }
        }
        const latest = values[values.length - 1];
        reading.textContent = latest === null || latest === undefined ? '…' : format(latest);
        const lastMark = marks[marks.length - 1];
        settle.textContent = lastMark ? (settleFrames === null ? 'settling…' : `settled in ${settleFrames} frames`) : '';
    };
    const schedule = (): void => {
        if (dirty) return;
        dirty = true;
        requestAnimationFrame(draw);
    };

    return {
        element: box,
        push(value: number | null): void {
            values.push(value !== null && Number.isFinite(value) ? value : null);
            // Keep the frames the chart shows, and the frames since the latest marker (at most `SETTLE_WINDOW` charts' worth).
            const lastMarkIndex = Math.max(marks[marks.length - 1]?.index ?? Infinity, values.length - SETTLE_WINDOW * capacity);
            const keepFrom = Math.min(values.length - capacity, lastMarkIndex);
            if (keepFrom > capacity) {
                values = values.slice(keepFrom);
                marks = marks.map((mark) => ({ ...mark, index: mark.index - keepFrom })).filter((mark) => mark.index >= 0);
            }
            const lastMark = marks[marks.length - 1];
            settleFrames = lastMark ? framesToSettle(values, lastMark.index, tolerance, hold) : null;
            schedule();
        },
        mark(label = ''): void {
            marks.push({ index: Math.max(0, values.length - 1), label });
            settleFrames = null;
            schedule();
        },
        get settleFrames() {
            return settleFrames;
        },
        clear(): void {
            values = [];
            marks = [];
            settleFrames = null;
            schedule();
        },
        dispose(): void {
            box.remove();
        },
    };
}
