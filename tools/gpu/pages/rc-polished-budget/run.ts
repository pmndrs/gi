/** From the GI root: pnpm exec vite build --config examples/vite.config.ts --base /gi/
 * then: node tools/gpu/pages/rc-polished-budget/run.ts */
import { spawnSync } from 'node:child_process';
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { judgeTiming, type ClockSummary } from '../../clock-note.ts';
import { setTimeout as sleep } from 'node:timers/promises';
import { LOCK_HELD_ENV, withBenchLock } from '../../../reference/bench-lock.ts';

if (process.argv[2] === '--inside-lock') {
    const [, , , resultFile, idleFile] = process.argv;
    const idleWindows = [];
    // Measure *before* Chrome starts, so this job cannot produce this load.
    for (let attempt = 0; attempt < 3; attempt++) {
        const samples: number[] = [];
        for (let n = 0; n < 5; n++) {
            const sample = spawnSync('nvidia-smi', ['--query-gpu=utilization.gpu','--format=csv,noheader,nounits'], {encoding:'utf8'});
            const value = Number(sample.stdout?.trim().split(/\r?\n/)[0]);
            if (sample.status === 0 && Number.isFinite(value)) samples.push(value);
            if (n < 4) await sleep(400);
        }
        const quiet = samples.length === 5 && samples.every(n => n < 20);
        idleWindows.push({ samples, quiet });
        if (quiet || attempt === 2) break;
        await sleep(3000);
    }
    writeFileSync(idleFile, JSON.stringify(idleWindows,null,2) + '\n');
    console.log('RTX utilization before Chrome:', JSON.stringify(idleWindows));
    const child = spawnSync(process.execPath, ['tools/gpu/run.ts', '--gpu', 'nvidia', '--filter', 'rc-polished-budget', '--json', resultFile], {stdio:'inherit'});
    if (child.error) throw child.error;
    process.exit(child.status ?? 1);
}

const directory = resolve(process.argv[2] ?? 'bench/screenshots/rc-polished-budget');
const gallery = readFileSync('examples/dist/11-radiance-cascades/index.html', 'utf8');
if (!gallery.includes('src="/gi/assets/')) throw new Error('Build examples with --base /gi/ before running the GPU harness.');
mkdirSync(directory, { recursive: true });
const pairs: ReturnType<typeof measurePair>[] = [];
function measurePair(pair: number, ownerPid: number) {
    const resultFile = join(directory, 'pair-' + pair + '.json');
    const clockFile = join(directory, 'pair-' + pair + '-clocks.json');
    const idleFile = join(directory, 'pair-' + pair + '-idle.json');
    const child = spawnSync(process.execPath, [
        'tools/reference/bench-lock.ts', 'run', '--what', 'polished rail pair ' + pair,
        '--cool-below', '70', '--cool-wait-max-minutes', '2', '--clock-file', clockFile, '--',
        process.execPath, process.argv[1], '--inside-lock', resultFile, idleFile,
    ], { stdio:'inherit', env:{ ...process.env, [LOCK_HELD_ENV]:String(ownerPid), VITE_RC_REFLECTION_PAIR:String(pair) } });
    if (child.error) throw child.error;
    if (child.status !== 0) throw new Error('Polished rail pair ' + pair + ' failed (exit ' + child.status + ')');
    const result = JSON.parse(readFileSync(resultFile, 'utf8'));
    const clock = JSON.parse(readFileSync(clockFile, 'utf8')) as ClockSummary;
    const page = result.pages.find((p: {name: string}) => p.name === 'rc-polished-budget');
    if (!result.ok || !page?.acceptable || page.data.pair !== pair) throw new Error('Incomplete pair ' + pair);
    const { on, off, floor } = page.data;
    const idle = JSON.parse(readFileSync(idleFile, 'utf8')).at(-1);
    return { pair, order:page.data.order, resultFile, clockFile, idleFile, preflightQuiet:idle.quiet, preflightGpuUtilization:idle.samples, timing:judgeTiming(clock),
        startC:clock.start?.temperatureC, loadedSamples:clock.loadedSamples, thermalShare:clock.thermalLimitedShare,
        on, off, medianDeltaMs:on.medianMs-off.medianMs, p95DeltaMs:on.p95Ms-off.p95Ms, floor };
}
await withBenchLock(async ({ ownerPid }) => {
    for (const pair of [1, 2, 3]) pairs.push(measurePair(pair, ownerPid));
}, { what:'three matched polished rail pairs', command:'node tools/gpu/pages/rc-polished-budget/run.ts' });
// Pool the equal-size windows without averaging percentiles. Keep matched
// median deltas and per-pair thermal status alongside these descriptive values.
function distribution(samples: number[]) {
    const sorted = [...samples].sort((a,b) => a-b), n = sorted.length;
    return { count:n, medianMs:(sorted[(n-1)>>1] + sorted[n>>1])/2, p95Ms:sorted[Math.ceil(n*0.95)-1] };
}
const on = distribution(pairs.flatMap(p => p.on.samplesMs)), off = distribution(pairs.flatMap(p => p.off.samplesMs));
const summary = { allPairsQualified:pairs.every(p => p.preflightQuiet && p.timing.kind === 'valid'), on, off,
    pooledMedianDeltaMs:on.medianMs-off.medianMs, pooledP95DeltaMs:on.p95Ms-off.p95Ms,
    pairs:pairs.map(({on,off,...p}) => ({...p, on:{count:on.count,medianMs:on.medianMs,p95Ms:on.p95Ms},
        off:{count:off.count,medianMs:off.medianMs,p95Ms:off.p95Ms}})) };
writeFileSync(join(directory,'summary.json'), JSON.stringify(summary,null,2) + '\n');
console.log(JSON.stringify(summary,null,2));
