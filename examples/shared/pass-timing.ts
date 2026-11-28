/**
 * GPU milliseconds per pass, from the platform's own timing
 * (`gi.stats.passes`, filled when the device has `timestamp-query` and the
 * platform was made with `gpuTiming`). Since #52 that includes the
 * radiance cache's screen-stage passes, which run after `encode`.
 *
 * Each pass gets one line on what it does (`describePass`), written from
 * docs/design/radiance-cache.md, docs/design/ddgi.md and
 * docs/design/m2-lights.md. Hover a pass, or tap it, to read it.
 *
 * Timings arrive a few frames late and vary frame to frame, so each row is
 * the mean over the last frames timed (30 by default), a pass that didn't
 * run in a frame counting as 0 there. These are this page's live readings
 * on this GPU, not numbers from docs/.
 */

import type { GIPlatform, PassTiming } from '@pmndrs/gi/core';
import './pass-timing.css';

/** One line per pass, by its full name in `stats.passes`. */
const PASS_DESCRIPTIONS: Readonly<Record<string, string>> = {
    // DDGI (docs/design/ddgi.md, docs/design/m2-lights.md).
    'ddgi/scroll': "Moves a camera-following volume's window, and resets the probes that enter it.",
    'ddgi/mark': "After a light, sky or object change, works out how much each probe's light changed and gives it updates to catch up.",
    'ddgi/schedule': "Picks this frame's probes on the GPU: the most urgent first, as many as the budget allows.",
    'ddgi/trace': "Traces each picked probe's rays and shades their hits, with last frame's probes giving the bounce light.",
    'ddgi/blend': "Blends each probe's new rays into its stored light and distances, keeping history; also moves probes out of walls and turns off useless ones.",
    'ddgi/write': 'Copies the updated probes into the irradiance and distance atlases the renderer reads.',
    'ddgi/propagate': 'Passes a change from a key probe to the fine probes around it, so far probes catch up without each tracing.',
    'ddgi/compose': 'Adds up the light-group and sun layers, by their weights, into the atlas the renderer reads.',
    'ddgi/census': 'Counts the probes in each state (active, off, not updated yet) for the stats; read back a few frames later.',
    'ddgi/reset-probes': 'Resets probe offsets and states after a reset call.',
    // The platform (docs/api/platform.md).
    'gi/light-grid': 'Builds the light grid: which lights matter in each cell, so a hit picks from those.',
    'gi/texture-store': "Copies new material textures into the texture array GI's rays read.",
    'gi/pack-mesh': "Copies new or changed meshes into GI's copy of the scene.",
    'gi/pose-copy': "Copies deforming meshes' new poses into GI's copy of the scene.",
    'platform/sky bake': 'Bakes the sky into the small map that rays read when they hit nothing.',
    'platform/top-level refit': 'Refits the tree over all objects after some moved, so rays find them where they are now.',
    'platform/mesh refit': "Refits a deforming mesh's own tree to its new pose.",
    // The radiance cache's screen stage (docs/design/radiance-cache.md, phases 2 and 4).
    'radiance-cache/SSGI': 'Screen-space occlusion and near-field light, from the depth and normals (off by default).',
    'radiance-cache/ClearCounters': "Clears the frame's counters.",
    'radiance-cache/ReprojectScreenProbes': "Moves last frame's screen probes to where their surfaces are now, by the motion vectors.",
    'radiance-cache/SpawnScreenProbes': 'Places one probe on the visible surface of each 16 × 16-pixel tile that needs one.',
    'radiance-cache/PatchScreenProbes': 'Fills tiles that reprojection left without a probe.',
    'radiance-cache/SampleScreenProbes': "Picks each probe's ray directions, more of them where last frame found more light.",
    'radiance-cache/PopulateScreenProbesCells (trace)': "Traces the probes' rays into the scene.",
    'radiance-cache/PopulateScreenProbesCells': "Finds, or makes, the world-cache cell at each probe ray's hit.",
    'radiance-cache/BlendScreenProbes': "Blends the rays' light into each probe, with its history.",
    'radiance-cache/ReorderScreenProbes': 'Keeps the cache of probes that left the screen, most recently used first.',
    'radiance-cache/FilterProbeMask': "Builds the probe mask's smaller levels, for the filter.",
    'radiance-cache/FilterScreenProbes': 'Smooths each probe with its neighbours on the same surface.',
    'radiance-cache/ProjectScreenProbes': 'Turns each probe into nine spherical-harmonic numbers per colour.',
    'radiance-cache/InterpolateScreenProbes': 'Gives every pixel its light from up to four nearby probes on its surface.',
    // The world cache: GI-1.2's hash grid (docs/design/radiance-cache.md, phase 3, levers E8 to E10).
    'radiance-cache/PurgeRadianceCache': "Ages the world cache's cells and frees those no ray has used for a while.",
    'radiance-cache/GenerateReservoirs': 'Picks one light for each hit in the world cache, for its shadow ray.',
    'radiance-cache/PopulateRadianceCache': "Traces those shadow rays and adds each hit's light to its cell.",
    'radiance-cache/UpdateRadianceCache': "Blends the frame's samples into each cell's running mean.",
    'radiance-cache/ResolveCells': "Adds each hit's cell light, filtered, to the probe ray that found it.",
    'radiance-cache/PopulateMultibounceCells': 'The second bounce: one more ray from some first hits, its hit put in a cell.',
    'radiance-cache/GenerateMultibounceReservoirs': 'The second bounce: picks a light for each second hit.',
    'radiance-cache/UpdateMultibounceCells': "The second bounce: adds the second hits' light to the first hits' cells.",
    'radiance-cache/SampleBounceSourceCells': "Reads DDGI's light at each hit as its later bounces, instead of tracing them (DDGI under the cache).",
    'radiance-cache/ClaimCells': 'Budgeted shading: each cell hit this frame picks one of its hits to light.',
    'radiance-cache/PrioritizeCells': 'Budgeted shading: orders the cells, new and changed first, and keeps as many as the budget allows.',
    'radiance-cache/SharedCellBounce': "Budgeted shading: one second-bounce ray per chosen cell, shared by the cell's hits.",
    'radiance-cache/ShadeCells': 'Budgeted shading: lights the chosen cells, one light and one shadow ray each.',
    // SHaRC, the other world cache (docs/design/radiance-cache.md, P6.1).
    'radiance-cache/SharcBegin': "SHaRC: starts an update path at each probe ray's hit.",
    'radiance-cache/SharcBounce1': "SHaRC: the path's first bounce ray.",
    'radiance-cache/SharcBounce2': "SHaRC: the path's second bounce ray.",
    'radiance-cache/SharcLight': "SHaRC: a light sample at each of the path's hits.",
    'radiance-cache/SharcCombine': 'SHaRC: adds up each path, the cache itself giving its tail, into its cells.',
    'radiance-cache/SharcResolve': "SHaRC: blends the frame's paths into each cell's mean.",
    'radiance-cache/SharcLookup': "SHaRC: reads each probe ray's light back from its cell.",
    // The neural tail (docs/design/radiance-cache.md, P6.4).
    'radiance-cache/NeuralTailGather': 'The neural tail: gathers the hits the network trains on or answers for.',
    'radiance-cache/NeuralTailTraining': 'The neural tail: one training step of the small network.',
    'radiance-cache/NeuralTailInference': "The neural tail: the network's answers for the later bounces.",
};

/** One line on what the pass named `name` does, or undefined for a pass this page doesn't know. */
export function describePass(name: string, extra?: Readonly<Record<string, string>>): string | undefined {
    const known = extra?.[name] ?? PASS_DESCRIPTIONS[name];
    if (known) return known;
    // Direct light at hits, run for whichever technique owns the rays: `<owner>/gi-direct`.
    if (name.endsWith('/gi-direct')) return "Direct light at each ray hit: one light sample and its shadow ray (the platform's hit shading).";
    return undefined;
}

/** One row of the list: a pass, its mean GPU time and its line. */
export interface PassRow {
    readonly name: string;
    /** Mean GPU milliseconds a frame over the frames averaged. */
    readonly ms: number;
    readonly description: string | undefined;
}

/**
 * Keeps the last `frames` timed frames of `stats.passes` and gives each
 * pass's mean. Passes that share a name in one frame (several volumes, two
 * rounds) are added up. Exported for its test.
 */
export class PassAverager {
    private readonly history: Array<{ frame: number; totals: Map<string, number> }> = [];
    /** Each pass's place in the GPU's order, from the latest frame it ran in. */
    private readonly order = new Map<string, number>();

    constructor(private readonly frames = 30) {}

    /** Adds a frame's passes; false if that frame was already added, or nothing was timed. */
    add(passes: readonly PassTiming[]): boolean {
        const timed = passes.filter((pass) => pass.gpuMs !== undefined);
        if (timed.length === 0) return false;
        const frame = timed[0].frame;
        if (this.history.some((entry) => entry.frame === frame)) return false;
        const totals = new Map<string, number>();
        timed.forEach((pass, index) => {
            // A name's place is where it first ran in its latest frame.
            if (!totals.has(pass.name)) this.order.set(pass.name, index);
            totals.set(pass.name, (totals.get(pass.name) ?? 0) + (pass.gpuMs ?? 0));
        });
        this.history.push({ frame, totals });
        if (this.history.length > this.frames) this.history.shift();
        return true;
    }

    /** Each pass seen in the frames kept, in GPU order, with its mean time. */
    rows(extra?: Readonly<Record<string, string>>): PassRow[] {
        const sums = new Map<string, number>();
        for (const { totals } of this.history) for (const [name, ms] of totals) sums.set(name, (sums.get(name) ?? 0) + ms);
        const count = Math.max(1, this.history.length);
        return [...sums]
            .map(([name, sum]) => ({ name, ms: sum / count, description: describePass(name, extra) }))
            .sort((a, b) => (this.order.get(a.name) ?? 0) - (this.order.get(b.name) ?? 0));
    }

    /** The frames averaged. */
    get frameCount(): number {
        return this.history.length;
    }

    clear(): void {
        this.history.length = 0;
        this.order.clear();
    }
}

export interface PassTimingOptions {
    /** The heading. Default "GPU time per pass". */
    readonly title?: string;
    /** Frames each mean covers. Default 30. */
    readonly frames?: number;
    /** Lines for passes this module doesn't know, or to say more about one, by full name. */
    readonly descriptions?: Readonly<Record<string, string>>;
    /** Where the list goes. Default a box of its own, bottom right, above the stats line. */
    readonly container?: HTMLElement;
}

export interface PassTimingPanel {
    readonly element: HTMLElement;
    /** The rows shown, newest. */
    readonly rows: readonly PassRow[];
    /** Their sum, ms a frame, or null before anything was timed. */
    readonly totalMs: number | null;
    dispose(): void;
}

const REFRESH_MS = 500;

/** The part of a name the list shows under its group: `ddgi/trace` → `trace`. */
function shortName(name: string): { group: string; pass: string } {
    const slash = name.indexOf('/');
    return slash < 0 ? { group: '', pass: name } : { group: name.slice(0, slash), pass: name.slice(slash + 1) };
}

/**
 * Shows GPU time per pass for `platform` (or for whichever platform the
 * function returns, for a page that swaps them) and keeps it current.
 */
export function showPassTiming(platform: GIPlatform | null | (() => GIPlatform | null), options: PassTimingOptions = {}): PassTimingPanel {
    const currentPlatform = typeof platform === 'function' ? platform : () => platform;
    const averager = new PassAverager(options.frames ?? 30);

    const box = document.createElement('section');
    box.className = options.container ? 'pass-timing' : 'pass-timing pass-timing-floating';
    const header = document.createElement('div');
    header.className = 'pass-timing-header';
    const title = document.createElement('span');
    title.textContent = options.title ?? 'GPU time per pass';
    const total = document.createElement('span');
    total.className = 'pass-timing-total';
    header.append(title, total);
    const list = document.createElement('div');
    list.className = 'pass-timing-list';
    const note = document.createElement('p');
    note.className = 'pass-timing-note';
    box.append(header, list, note);
    (options.container ?? document.body).append(box);

    const rowsByName = new Map<string, { element: HTMLDetailsElement; ms: HTMLElement; bar: HTMLElement }>();
    let rows: PassRow[] = [];
    let shownPlatform: GIPlatform | null = null;

    const makeRow = (row: PassRow): { element: HTMLDetailsElement; ms: HTMLElement; bar: HTMLElement } => {
        const element = document.createElement('details');
        element.className = 'pass';
        const summary = document.createElement('summary');
        const { group, pass } = shortName(row.name);
        const name = document.createElement('span');
        name.className = 'pass-name';
        name.textContent = pass;
        if (group) name.dataset.group = group;
        const ms = document.createElement('span');
        ms.className = 'pass-ms';
        const bar = document.createElement('span');
        bar.className = 'pass-bar';
        summary.append(name, ms, bar);
        summary.title = row.description ?? row.name;
        element.append(summary);
        const text = document.createElement('p');
        text.textContent = row.description ? `${row.description} (${row.name})` : `${row.name}: no description yet.`;
        element.append(text);
        return { element, ms, bar };
    };

    const render = (): void => {
        const gi = currentPlatform();
        if (gi !== shownPlatform) {
            shownPlatform = gi;
            averager.clear();
        }
        rows = averager.rows(options.descriptions);
        const sum = rows.reduce((total, row) => total + row.ms, 0);
        total.textContent = rows.length > 0 ? `${sum.toFixed(2)} ms` : '';
        if (!gi) note.textContent = 'GI starting…';
        else if (rows.length === 0) note.textContent = 'No GPU timings yet. They need the timestamp-query feature, which this browser or GPU may not offer.';
        else note.textContent = `Mean of the last ${averager.frameCount} timed frames, on this GPU. Hover or tap a pass for what it does.`;

        // Rows by group: a heading where the group changes.
        const wanted: HTMLElement[] = [];
        let lastGroup: string | null = null;
        for (const row of rows) {
            const { group } = shortName(row.name);
            if (group !== lastGroup) {
                const heading = document.createElement('div');
                heading.className = 'pass-group';
                heading.textContent = group || 'other';
                wanted.push(heading);
                lastGroup = group;
            }
            let entry = rowsByName.get(row.name);
            if (!entry) {
                entry = makeRow(row);
                rowsByName.set(row.name, entry);
            }
            entry.ms.textContent = row.ms.toFixed(3);
            entry.bar.style.width = `${sum > 0 ? Math.max(1, (row.ms / sum) * 100) : 0}%`;
            wanted.push(entry.element);
        }
        list.replaceChildren(...wanted);
    };

    let frameRequest = 0;
    const poll = (): void => {
        const gi = currentPlatform();
        if (gi) averager.add(gi.stats.passes);
        frameRequest = requestAnimationFrame(poll);
    };
    poll();
    const timer = setInterval(render, REFRESH_MS);
    render();

    return {
        element: box,
        get rows() {
            return rows;
        },
        get totalMs() {
            return rows.length > 0 ? rows.reduce((total, row) => total + row.ms, 0) : null;
        },
        dispose(): void {
            cancelAnimationFrame(frameRequest);
            clearInterval(timer);
            box.remove();
        },
    };
}
