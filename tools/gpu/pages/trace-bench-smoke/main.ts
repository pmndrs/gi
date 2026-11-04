/**
 * Smoke run of the tracer benchmark on a small scene and on Sponza, with tiny
 * ray counts. It proves the whole path works on large geometry (loading, worker
 * build, three-mesh-bvh, 3.75 M triangles). The numbers are not results.
 */
import { runTraceBenchPage } from '../trace-bench/run.ts';

await runTraceBenchPage({
    scenes: ['furnished-room', 'sponza'],
    profile: 'quick',
    query: 'rays=16&targetRays=16384&warmups=1&iterations=3',
});
