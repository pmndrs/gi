/**
 * B2: the GPU H-PLOC's Sponza build time, measured GPU-bound with the clocks
 * up (DECISIONS f3e3bba). Only the `ours-hploc` tracer runs: tinybvh's trees
 * are built in the worker (on the CPU, for the SAH reference) but not traced,
 * so the GPU's first heavy work is the build's loaded section. Trace speed
 * against tinybvh's trees comes from trace-bench-hploc-sponza.
 */
import { runTraceBenchPage } from '../trace-bench/run.ts';

await runTraceBenchPage({ scenes: ['sponza'], profile: 'full', query: 'tracer=ours-hploc' });
