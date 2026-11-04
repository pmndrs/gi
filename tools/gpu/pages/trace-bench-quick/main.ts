/**
 * Quick tracer benchmark: the furnished room at 64 rays per probe. A few
 * seconds. Not a measurement; it shows that ours (on tinybvh and three-mesh-bvh trees, and through the platform API) and
 * three-mesh-bvh run on this GPU and agree on what the rays hit.
 */
import { runTraceBenchPage } from '../trace-bench/run.ts';

await runTraceBenchPage({ scenes: ['furnished-room'], profile: 'quick' });
