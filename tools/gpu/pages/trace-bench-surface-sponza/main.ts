/**
 * The surface-ray investigation (P6.6, from the Mac track's RC5.4) on sponza: RC1.5's screen-probe rays against
 * probe-grid rays of the same size, and variants that change one thing at a time between them
 * (see ../trace-bench/surface-rays-bench.ts).
 */
import { runSurfaceRaysPage } from '../trace-bench/surface-rays-bench.ts';

await runSurfaceRaysPage('sponza');
