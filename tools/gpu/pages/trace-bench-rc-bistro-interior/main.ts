/**
 * Ray cost table (RC1.5) on bistro-interior: Capsaicin GI-1.2-shaped ray sets from the scene's first camera
 * bookmark at 1280x720, timed through the platform tracer and ours-tinybvh (see ../trace-bench/rc-rays.ts).
 * Query parameters and VITE_DDGI_TRACE_QUERY still override (see ../trace-bench/config.ts).
 */
import { runTraceBenchPage } from '../trace-bench/run.ts';

await runTraceBenchPage({ scenes: ['bistro-interior'], profile: 'full', query: 'raySet=rc&tracer=platform,ours-tinybvh' });
