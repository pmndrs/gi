/// <reference types="node" />
/**
 * Runs trace-bench pages several times on each GPU and turns the runs into
 * CSV rows: one set per run plus the median of the runs (see to-csv.ts).
 *
 *   node tools/gpu/pages/trace-bench/repeat.ts --page trace-bench-sponza [--page ...] [options]
 *
 *   --page <name>       a trace-bench page folder (repeatable, or comma-separated)
 *   --runs <n>          runs per page and GPU (default 3)
 *   --gpu <list>        GPUs to run on, comma-separated (default "nvidia,intel")
 *   --json-dir <dir>    where the runner's JSON files go (default: a new folder under the OS temp folder)
 *   --append            add the rows to bench/results/trace.csv; without it they are printed
 *   --allow-dirty       measure even when the measured code differs from HEAD (rows say "-dirty")
 *
 * The commit is read once, before the first run, and must still be the same
 * after the last one. This script does not take the bench lock itself; wrap it:
 *
 *   node tools/reference/bench-lock.ts run --what "trace bench x3" -- node tools/gpu/pages/trace-bench/repeat.ts --page trace-bench-sponza --append
 */

import { spawnSync } from 'node:child_process';
import { mkdtempSync, mkdirSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { parseArgs } from 'node:util';
import { gitCommit } from './to-csv.ts';

const REPOSITORY_ROOT = resolve(import.meta.dirname, '..', '..', '..', '..');
const DEFAULT_RUNS = 3;

function splitList(values: readonly string[] | undefined): string[] {
    return (values ?? [])
        .flatMap((value) => value.split(','))
        .map((value) => value.trim())
        .filter((value) => value.length > 0);
}

function runNode(args: string[]): number {
    const result = spawnSync(process.execPath, args, { cwd: REPOSITORY_ROOT, stdio: 'inherit' });
    return result.status ?? 1;
}

function main(): void {
    const { values } = parseArgs({
        args: process.argv.slice(2),
        options: {
            page: { type: 'string', multiple: true },
            runs: { type: 'string' },
            gpu: { type: 'string' },
            'json-dir': { type: 'string' },
            append: { type: 'boolean' },
            'allow-dirty': { type: 'boolean' },
        },
    });
    const pages = splitList(values.page);
    const gpus = splitList(values.gpu ? [values.gpu] : ['nvidia,intel']);
    const runs = Number(values.runs ?? DEFAULT_RUNS);
    if (pages.length === 0 || !Number.isInteger(runs) || runs < 1) {
        console.error('Usage: node tools/gpu/pages/trace-bench/repeat.ts --page <name> [--page ...] [--runs 3] [--gpu nvidia,intel] [--json-dir dir] [--append] [--allow-dirty]');
        process.exit(2);
    }

    const commit = gitCommit();
    if (commit.endsWith('-dirty') && !values['allow-dirty']) {
        console.error(`the measured code (see MEASURED_PATHS in to-csv.ts) differs from HEAD (${commit}). Commit first, or pass --allow-dirty.`);
        process.exit(2);
    }
    const jsonDir = values['json-dir'] ? resolve(values['json-dir']) : mkdtempSync(join(tmpdir(), 'trace-bench-repeat-'));
    mkdirSync(jsonDir, { recursive: true });
    console.error(`commit ${commit}; ${runs} run(s) of ${pages.join(', ')} on ${gpus.join(', ')}; runner JSON in ${jsonDir}`);

    const files: string[] = [];
    const failures: string[] = [];
    for (const gpu of gpus) {
        for (const page of pages) {
            for (let run = 1; run <= runs; run++) {
                const json = join(jsonDir, `${page}-${gpu}-run${run}.json`);
                console.error(`\n== ${page} on ${gpu}, run ${run} of ${runs}`);
                const status = runNode(['tools/gpu/run.ts', '--gpu', gpu, '--filter', page, '--json', json]);
                if (status === 0) files.push(json);
                else failures.push(`${page} on ${gpu}, run ${run} (exit ${status})`);
            }
        }
    }

    if (gitCommit() !== commit) {
        console.error(`the measured code changed during the runs (was ${commit}, now ${gitCommit()}); no rows written`);
        process.exit(1);
    }
    if (failures.length > 0) {
        // A failed run would leave a median of fewer runs than the rows claim, so nothing is written.
        console.error(`failed: ${failures.join('; ')}. No rows written; the JSON files are in ${jsonDir}`);
        process.exit(1);
    }
    const status = runNode([
        'tools/gpu/pages/trace-bench/to-csv.ts',
        ...files,
        '--commit',
        commit,
        ...(values.append ? ['--append'] : []),
    ]);
    process.exit(status);
}

main();
