/**
 * P6.5: the radiance-cache variants' GPU timing as many short runs, so each stays inside the
 * clock rule (DECISIONS 2026-11-05: a VALID run starts below 70 C and throttles in under 5% of
 * its loaded samples). The whole set in one run (ten presets on seven scenes, 27 minutes)
 * throttled in 99.5% of its samples on the RTX 3060 Laptop (2026-11-28).
 *
 * Each run is one scene and two or three presets of `radiance-cache-variants-timing`, with fewer
 * warm-up and timed frames, through `bench-lock run` with its cool wait and its own clock file.
 * The runs are queued one after another, so other jobs can take the lock between them. A run
 * whose verdict isn't VALID is retried once at the end. Then:
 *
 *   node tools/reference/rc-variants-table.ts --valid-only --timing <dir>/<run>.json --clock <dir>/<run>.clock.json ...
 *
 * (the script prints that command with every run's files at the end).
 *
 *   node tools/reference/rc-variants-timing-split.ts [--out bench/results/incoming/rc-variants-timing-split]
 *     [--scenes a,b] [--warm 120] [--timed 60] [--dry-run]
 */

import { spawnSync } from 'node:child_process';
import { existsSync, mkdirSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { parseArgs } from 'node:util';
import { judgeTiming, readClockSummary } from '../gpu/clock-note.ts';

const ROOT = resolve(import.meta.dirname, '..', '..');

const { values } = parseArgs({
    options: {
        out: { type: 'string', default: 'bench/results/incoming/rc-variants-timing-split' },
        scenes: { type: 'string', default: 'cornell-box-point,cornell-box,two-room,furnished-room,sponza,bistro-interior,bistro-exterior' },
        warm: { type: 'string', default: '120' },
        timed: { type: 'string', default: '60' },
        'dry-run': { type: 'boolean' },
    },
});

/** The presets, two or three a run (RADIANCE_CACHE_VARIANT_PRESETS). */
const PRESET_GROUPS: readonly (readonly string[])[] = [
    ['gi12-rays', 'gi12-ddgi', 'sharc'],
    ['sharc-ddgi', 'budgeted'],
    ['neural-tail', 'gi12-rays-feedback'],
    ['gi12-rays-temporal', 'gi12-rays-feedback-temporal', 'gi12-ddgi-temporal'],
];

interface Run {
    readonly name: string;
    readonly scene: string;
    readonly presets: readonly string[];
    readonly json: string;
    readonly clock: string;
}

const outDir = resolve(ROOT, values.out);
mkdirSync(outDir, { recursive: true });
const runs: Run[] = [];
for (const scene of values.scenes.split(',').map((item) => item.trim()).filter(Boolean)) {
    for (const [index, presets] of PRESET_GROUPS.entries()) {
        const name = `${scene}-${index + 1}`;
        runs.push({ name, scene, presets, json: join(outDir, `${name}.json`), clock: join(outDir, `${name}.clock.json`) });
    }
}

function verdictOf(run: Run): string {
    const summary = existsSync(run.clock) ? readClockSummary(run.clock) : null;
    if (!summary || !existsSync(run.json)) return 'missing';
    return judgeTiming(summary).kind;
}

function execute(run: Run, attempt: string): string {
    const env = {
        ...process.env,
        VITE_RC_VARIANT_SCENES: run.scene,
        VITE_RC_VARIANTS: run.presets.join(','),
        VITE_RC_VARIANT_TIMING_WARM_FRAMES: values.warm,
        VITE_RC_VARIANT_TIMING_TIMED_FRAMES: values.timed,
    };
    const args = ['tools/reference/bench-lock.ts', 'run', '--what', `p6.5 variants timing ${run.name}${attempt}`, '--clock-file', run.clock, '--', 'pnpm', 'test:gpu', '--filter', 'radiance-cache-variants-timing', '--gpu', 'nvidia', '--json', run.json];
    console.log(`\n== ${run.name}${attempt}: ${run.scene}, ${run.presets.join(', ')}`);
    if (values['dry-run']) return 'dry run';
    spawnSync(process.execPath, args, { cwd: ROOT, env, stdio: 'inherit', shell: false });
    const verdict = verdictOf(run);
    console.log(`== ${run.name}${attempt}: ${verdict.toUpperCase()}`);
    return verdict;
}

const verdicts = new Map<string, string>();
for (const run of runs) verdicts.set(run.name, execute(run, ''));
for (const run of runs) if (verdicts.get(run.name) !== 'valid' && !values['dry-run']) verdicts.set(run.name, execute(run, ' (retry)'));

const valid = [...verdicts.values()].filter((verdict) => verdict === 'valid').length;
console.log(`\n${valid} of ${runs.length} runs VALID.`);
for (const run of runs) console.log(`  ${run.name}: ${verdicts.get(run.name)}`);
const files = runs.filter((run) => existsSync(run.json)).flatMap((run) => ['--timing', run.json, '--clock', run.clock]);
console.log(`\nTable: node tools/reference/rc-variants-table.ts --valid-only ${files.join(' ')}`);
