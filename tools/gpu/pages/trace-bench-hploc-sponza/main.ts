/**
 * B2: full tracer benchmark of Sponza on the GPU H-PLOC's trees against
 * tinybvh's (B0), on the same rays: the GPU build time (timestamps, one batch
 * of every mesh), the trees' SAH cost and size, the match with tinybvh's
 * BuildHPLOC, and closest-hit and any-hit speed. Query parameters still
 * override (see ../trace-bench/config.ts).
 */
import { runTraceBenchPage } from '../trace-bench/run.ts';

await runTraceBenchPage({ scenes: ['sponza'], profile: 'full', query: 'tracer=ours-tinybvh,ours-hploc' });
