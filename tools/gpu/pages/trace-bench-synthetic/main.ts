/** Top-level rebuild time with 10,000 instances of a small mesh. CPU only; the GPU is not used for tracing. */
import { runTraceBenchPage } from '../trace-bench/run.ts';

await runTraceBenchPage({ scenes: ['synthetic-10k'], profile: 'full' });
