/**
 * Tracer benchmark, selected by query parameters (see config.ts).
 *
 *   ?scene=sponza&tracer=ours-tinybvh,tmbvh&rays=16&targetRays=65536&profile=quick
 *
 * With no parameters it runs the quick profile on the furnished room: a few
 * seconds, enough to see that every tracer works and agrees. The full-size
 * runs are the trace-bench-<scene> pages next to this one.
 */
import { runTraceBenchPage } from './run.ts';

await runTraceBenchPage({ scenes: ['furnished-room'], profile: 'quick' });
