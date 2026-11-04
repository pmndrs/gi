/**
 * Full-size tracer benchmark of Sponza's base package alone (no curtains): the
 * scene the `sponza` rows measured before T1.10, so new rows compare with them.
 */
import { runTraceBenchPage } from '../trace-bench/run.ts';

await runTraceBenchPage({ scenes: ['sponza-base'], profile: 'full' });
