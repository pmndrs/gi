/**
 * Quick B1 benchmark: the furnished room at 64 rays per probe, tinybvh's trees
 * and the GPU LBVH's. A few seconds. Not a measurement; it shows that the GPU
 * LBVH builds, its trees trace, and both agree on what the rays hit.
 */
import { runTraceBenchPage } from '../trace-bench/run.ts';

await runTraceBenchPage({ scenes: ['furnished-room'], profile: 'quick', query: 'tracer=ours-tinybvh,ours-lbvh' });
