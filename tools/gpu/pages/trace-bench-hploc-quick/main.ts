/**
 * B2, quick: the GPU H-PLOC's trees of the furnished room, built, matched with
 * tinybvh's BuildHPLOC and traced against tinybvh's trees (a check that the
 * page and the tracer work, not a measurement).
 */
import { runTraceBenchPage } from '../trace-bench/run.ts';

await runTraceBenchPage({ scenes: ['furnished-room'], profile: 'quick', query: 'tracer=ours-tinybvh,ours-hploc' });
