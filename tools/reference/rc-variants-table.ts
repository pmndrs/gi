/**
 * P6.5: the radiance-cache variants table (docs/design/radiance-cache-variants.md)
 * from the GPU pages' `--json` reports, so the comparison can be run again and
 * the table rebuilt without copying numbers by hand.
 *
 *   node tools/reference/rc-variants-table.ts \
 *     --variants bench/results/rc-variants.json [--variants <more>.json ...] \
 *     [--timing bench/results/rc-variants-timing.json --clock bench/results/rc-variants-timing.clock.json]
 *
 * `--variants`: reports of `radiance-cache-variants` (accuracy, warm-up, flicker, lag, memory).
 * Several may be given (references arrive scene by scene): a later file's scene replaces an
 * earlier one's. `--timing`: a report of `radiance-cache-variants-timing`; `--clock`: the clock
 * file of that run (`bench-lock run --clock-file`), whose verdict (VALID, INVALID, NO VERDICT)
 * labels the GPU times. Without a clock file the times are labelled "no verdict".
 * Prints Markdown to stdout.
 */

import { readFileSync } from 'node:fs';
import { parseArgs } from 'node:util';
import { judgeTiming, readClockSummary } from '../gpu/clock-note.ts';

interface AccuracyNumbers {
    readonly bias: number;
    readonly meanError8: number;
    readonly p95Error8: number;
}

interface StepResponse {
    readonly changedBlocks: number;
    readonly framesTo50: number | null;
    readonly framesTo90: number | null;
    readonly residualAt4: number;
    readonly residualAt16: number;
    readonly overshoot: number;
    readonly endOverBefore: number;
}

interface VariantResult {
    readonly ddgiStacked: boolean;
    readonly secondBounceInUse: string | null;
    readonly neuralTailRan: boolean;
    readonly warmupFrames: number | null;
    readonly gi: AccuracyNumbers;
    readonly giOneFrame: AccuracyNumbers;
    readonly coverage: number;
    readonly flicker: { readonly meanPerPixel: number; readonly p95PerPixel: number; readonly meanOfFrame: number };
    readonly lightsDouble: { readonly what: string; readonly response: StepResponse | null };
    readonly boxMoves: { readonly what: string; readonly response: StepResponse | null };
    readonly memoryMiB: { readonly radianceCache: number; readonly ddgi: number; readonly total: number };
    readonly warnings: readonly string[];
}

interface SceneResult {
    readonly view: string;
    readonly variants: Record<string, VariantResult>;
}

interface TimingResult {
    readonly totalMsMedian: number;
    readonly radianceCacheMsMedian: number;
    readonly worldCacheMsMedian: number;
    readonly ddgiMsMedian: number;
    readonly neuralTailMsMedian?: number;
}

interface PageSummary {
    readonly name: string;
    readonly data: Record<string, unknown>;
    readonly adapter: { readonly description?: string; readonly vendor?: string; readonly architecture?: string } | null;
}

const { values } = parseArgs({
    options: {
        variants: { type: 'string', multiple: true },
        timing: { type: 'string' },
        clock: { type: 'string' },
    },
});

function pageData(path: string, page: string): PageSummary {
    const summary = JSON.parse(readFileSync(path, 'utf8')) as { pages: PageSummary[] };
    const found = summary.pages.find((entry) => entry.name === page);
    if (!found) throw new Error(`${path}: no ${page} page in the report`);
    return found;
}

const SCENE_ORDER = ['cornell-box-point', 'cornell-box', 'two-room', 'furnished-room', 'sponza', 'bistro-interior', 'bistro-exterior'];
const VARIANT_ORDER = ['gi12-rays', 'gi12-ddgi', 'sharc', 'sharc-ddgi', 'budgeted', 'neural-tail', 'gi12-rays-feedback', 'gi12-rays-temporal', 'gi12-rays-feedback-temporal', 'gi12-ddgi-temporal'];
/** By `order`, a `<scene>/<view>` key after its scene's first view. */
const sortBy = (order: readonly string[]) => (a: string, b: string) => {
    const rank = (key: string): number => (order.indexOf(key.split('/')[0]) + 1 || 99) * 2 + (key.includes('/') ? 1 : 0);
    return rank(a) - rank(b) || a.localeCompare(b);
};

const scenes = new Map<string, SceneResult>();
let adapter = '';
for (const path of values.variants ?? []) {
    const page = pageData(path, 'radiance-cache-variants');
    adapter = page.adapter ? [page.adapter.vendor, page.adapter.architecture, page.adapter.description].filter(Boolean).join(' ') : adapter;
    for (const [scene, result] of Object.entries((page.data.scenes ?? {}) as Record<string, SceneResult>)) scenes.set(scene, result);
}

const pct = (value: number, digits = 1): string => `${(value * 100).toFixed(digits)}%`;
const signed = (value: number): string => `${value >= 0 ? '+' : ''}${(value * 100).toFixed(1)}%`;
const frames = (value: number | null): string => (value === null ? '>' : String(value));
/** Fewer changed 8 × 8 blocks than this and the region is too small to read (a box hidden behind furniture). */
const MIN_CHANGED_BLOCKS = 20;
const step = (response: StepResponse | null): string =>
    !response ? 'n/a' : response.changedBlocks < MIN_CHANGED_BLOCKS ? `(${response.changedBlocks} blocks)` : `${frames(response.framesTo50)} / ${frames(response.framesTo90)} · ${pct(response.residualAt4, 0)}`;

const lines: string[] = [];
const sceneIds = [...scenes.keys()].sort(sortBy(SCENE_ORDER));
lines.push(`Measured on ${adapter || 'an unnamed adapter'}; scenes: ${sceneIds.join(', ') || 'none'}.`, '');
for (const sceneId of sceneIds) {
    const scene = scenes.get(sceneId)!;
    lines.push(`#### ${sceneId.split('/')[0]} (\`${scene.view}\`)`, '');
    lines.push('| Variant | GI bias | 8×8 mean | 8×8 p95 | one frame mean | cache coverage | warm-up (frames) | flicker per pixel (p95) | flicker of the frame | lights ×2: 50% / 90% frames · left at 4 | box moves: 50% / 90% frames · left at 4 | memory MiB (DDGI) |');
    lines.push('|---|---|---|---|---|---|---|---|---|---|---|---|');
    for (const variantId of Object.keys(scene.variants).sort(sortBy(VARIANT_ORDER))) {
        const result = scene.variants[variantId];
        const ran = variantId === 'neural-tail' && !result.neuralTailRan ? ' (tail off)' : '';
        lines.push(
            `| ${variantId}${ran} | ${signed(result.gi.bias)} | ${pct(result.gi.meanError8)} | ${pct(result.gi.p95Error8, 0)} | ${pct(result.giOneFrame.meanError8)} | ${pct(result.coverage, 0)} | ${frames(result.warmupFrames)} | ${pct(result.flicker.meanPerPixel, 2)} (${pct(result.flicker.p95PerPixel, 1)}) | ${pct(result.flicker.meanOfFrame, 3)} | ${step(result.lightsDouble.response)} | ${step(result.boxMoves.response)} | ${result.memoryMiB.total.toFixed(0)} (${result.memoryMiB.ddgi.toFixed(0)}) |`,
        );
    }
    const first = Object.values(scene.variants)[0];
    if (first) lines.push('', `Lights: ${first.lightsDouble.what}. Box: ${first.boxMoves.what}.`);
    lines.push('');
}

if (values.timing) {
    const page = pageData(values.timing, 'radiance-cache-variants-timing');
    const summary = values.clock ? readClockSummary(values.clock) : null;
    const verdict = summary ? judgeTiming(summary) : null;
    const label = verdict ? `${verdict.kind === 'valid' ? 'VALID' : verdict.kind === 'invalid' ? 'INVALID' : 'NO VERDICT'} (${verdict.reason})` : 'no verdict (no clock file)';
    const data = page.data as { width: number; height: number; scenes: Record<string, Record<string, TimingResult>> };
    lines.push(`#### GPU ms per frame at ${data.width}×${data.height}: radiance cache + DDGI (world cache; DDGI) — timing ${label}`, '');
    const timingScenes = Object.keys(data.scenes).sort(sortBy(SCENE_ORDER));
    lines.push(`| Variant | ${timingScenes.join(' | ')} |`);
    lines.push(`|---|${timingScenes.map(() => '---').join('|')}|`);
    const variantIds = [...new Set(timingScenes.flatMap((scene) => Object.keys(data.scenes[scene])))].sort(sortBy(VARIANT_ORDER));
    for (const variantId of variantIds) {
        const cells = timingScenes.map((scene) => {
            const result = data.scenes[scene][variantId];
            if (!result) return '';
            const neural = result.neuralTailMsMedian ? `; NRC ${result.neuralTailMsMedian.toFixed(2)}` : '';
            return `${result.totalMsMedian.toFixed(2)} (${result.worldCacheMsMedian.toFixed(2)}; ${result.ddgiMsMedian.toFixed(2)}${neural})`;
        });
        lines.push(`| ${variantId} | ${cells.join(' | ')} |`);
    }
    lines.push('');
}

console.log(lines.join('\n'));
