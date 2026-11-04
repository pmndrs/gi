/** Full-size tracer benchmark of one scene. Query parameters still override (see ../trace-bench/config.ts). */
import { runTraceBenchPage } from '../trace-bench/run.ts';

await runTraceBenchPage({ scenes: ['sponza'], profile: 'full' });
