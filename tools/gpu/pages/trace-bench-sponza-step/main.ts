/**
 * Sponza at the full ray count (about 2M closest-hit rays, the same rays as
 * trace-bench-sponza) but with 2 warm-ups and 7 timed dispatches instead of
 * 5 and 20. It is the quick before/after check for one optimisation step: the
 * same rays and both tracers in one run, so the ratio to three-mesh-bvh is
 * measured under the same clocks. Query parameters still override.
 */
import { runTraceBenchPage } from '../trace-bench/run.ts';

await runTraceBenchPage({ scenes: ['sponza'], profile: 'full', query: 'warmups=2&iterations=7' });
