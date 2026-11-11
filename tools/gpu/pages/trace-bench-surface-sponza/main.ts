/**
 * The surface-ray investigation (RC5.4, lever 20) on sponza: RC1.5's screen-probe rays against
 * probe-grid rays of the same size, and variants that change one thing at a time between them
 * (see ../trace-bench/surface-rays-bench.ts).
 */
import { runSurfaceRaysPage } from '../trace-bench/surface-rays-bench.ts';

await runSurfaceRaysPage('sponza');
