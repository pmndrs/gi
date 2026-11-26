# Implementation and measurement plan

## Phase status

All five original phases pass under the web acceptance criteria agreed on November 19. Chandra and S5 are comparison tools for diagnosing artifacts. Browser frame cost, memory, temporal response and visible quality at the intended display resolution drive solver choices. [Decision](../DECISIONS.md#2026-11-19)

| Original phase | Status on the RC branch | Evidence or remaining work |
| --- | --- | --- |
| Reference rail | Passed | S5 Three.js port, fixed rail and lighting cues verified on NVIDIA. |
| Interval seam | Passed | GPU fixtures cover interval boundaries, misses, near occlusion and directional splits. |
| Shared tracer | Passed | The [transport fixture](../../tools/gpu/pages/rc-transport/) checks hit distances, misses, emission and analytical diffuse lighting before and after a blocker moves. The gallery passes the shutter-response and offscreen-light checks. The [Sponza comparison](chandra.md) records remaining visual differences for investigation when they affect the web demos. |
| Dynamic occlusion | Passed | Both backends darken and recover on the moving shutter. The response curves and independent emitter integral are in the [progress log](../../PROGRESS.md). |
| Reconstruction and hardware | Passed | Windows passed the 24-pose polished-rail quality check and three idle RTX reflection pairs. Normal gallery pacing gave pooled medians of 6.29 ms on and 3.61 ms off with 0–1% GPU use before Chrome. Three clock-qualified uncapped pairs gave 2.11 ms on and 1.76 ms off over the extended rail. Intel Gen12LP passed solver, transport and reflection correctness. On an Apple M5 Pro (macOS 26.6.2, Chrome 158) at `d8b9776`, all eight correctness fixtures, the quality check, six idle reflection pairs and the examples smoke test passed. The shutter reached 90% at 2 frames after closure and 12 after reopening. Polished-floor RGB error was 2.137/255 at 1.5× and 2.630/255 at 2×. Uncapped reflections cost 1.30 ms on and 1.14 ms off over 1,440 frames each. The [Apple Metal issue](https://github.com/pmndrs/gi/issues/43) holds the full results. |

The [upstream Chandra build](chandra.md) and three polished-floor experiments now run on Windows. See the [experiment list](../../research/radiance-cascades/reflections.md) for their individual checks.

| Stage | Work | Evidence to advance |
| --- | --- | --- |
| Reference rail | Pin the S5 solver and its notices. Build a small architectural passage with bounced color, a repeatable route and lighting cues. | The built example runs on a hardware adapter. Receiver pixels respond to the lamp. Record resource limits and timing scope. |
| Interval seam | Feed a known triangle fixture and explicit GPU ray records into the pathtracer dispatcher. Classify endpoints, misses, transparent intervals and the unobserved tail. | GPU records agree with analytical expectations at both sides of a seam. |
| Shared tracer | Keep the reference selectable. Implement a GI-owned surface-ray generator, closest-hit and hit shading, then deposit, merge and gather the observed intervals. | Check hit distances, visibility and outgoing light on known geometry. Exercise the native transport on the rail, shutter and Sponza. Use per-ray and image comparisons to investigate visible defects. |
| Dynamic occlusion | Replay a sealed side chamber with an off-screen constant emitter, a rigid sliding shutter, and a fixed receiver. Compare closure and reopening on both solvers. | Measure closed/open indirect luminance, sustained T50/T90 after travel ends, and recovery relative to the original open state. |
| Reconstruction and hardware | Compare native, reduced GI resolution and velocity/depth-aware upscaling. Evaluate a secondary NRC hook only if profiling identifies a bottleneck. | Alternate modes on the same adapter and cue, capture image pairs, then repeat on Apple Metal and Windows D3D12. |

## Reproduction

1. On a fresh checkout, run `pnpm install --frozen-lockfile` and `pnpm examples` from this GI checkout. Open the [architectural rail](http://localhost:5203/11-radiance-cascades/).
2. Replay the rail with `?stride=2` and `?stride=1`. The former schedules one primary ray per 2 × 2 cell and the latter one per pixel, before background rejection. Hold viewport, exposure, secondaries, probe capacity and camera clock constant. Switching to the direct view keeps GI computing, so use it for a lighting comparison.
3. Build with `PAGES_BASE=/gi/` and `pnpm examples:build`, then run `pnpm test:gpu --filter rc-rail`. In PowerShell set `$env:PAGES_BASE='/gi/'` before building. The [standalone interval experiment](../../experiments/rc-intervals/) has its own setup instructions.
4. Run the paired benchmark under the shared GPU lock with `node tools/reference/bench-lock.ts run --what rc-rail-budget -- node tools/gpu/run.ts --filter rc-rail-budget --json rail-budget.json`. This manual page alternates the order of stride 1 and stride 2 across three pairs, uses the demo’s 960 × 540 drawing size by default (set `VITE_RC_BUDGET_SIZE=480x270` for the smaller comparison) and measures the same 60 rail ticks in both arms. Receiver error uses normalized display RGB relative to stride 1. On a thermally limited adapter, set `VITE_RC_BUDGET_PAIR` to 1, 2 and 3 in turn and run each pair under its own cool-start lock. Keep all three clock verdicts.
5. For a performance decision take at least three alternating warmed runs at matched ticks. Record median and p95 frame times, cascade GPU timestamps, ray budget and allocated bytes. Compare post-cue images to stride 1 at the same pose. For radiometric error use a converged path-traced reference with the same materials and lights and compare linear HDR regions before tone mapping.
6. Keep startup, steady motion, a newly visible room and post-cue settling separate. Include indirect-only captures. For the shutter, normalize the fixed receiver to its measured open and closed states. A pass needs both light transfer through the open doorway and attenuation after closure.

The interactive gallery starts with the native solver and 1.5× temporal upscaling. For the native backend use `?backend=interval`. Select `?backend=three` for the port. Both support `scene=gallery|door`, `resolve=native|taa|upscale`, `ratio=1.5|2|3`, `sharpness=0..1` and `outputSize=preview|full`. The ratio is display width divided by render width. For a fixed receiver use `camera=receiver&view=indirect`. Use the manual GPU pages `rc-backends`, `rc-door-response` and `rc-reconstruction` for the controlled comparisons. `rc-native-core` verifies the solver contract on raw WebGPU.

For the glossy experiments, open the gallery with `scene=polished`, `scene=roughness` or `scene=reflection-motion`. Select the native interval backend and use `view=specular` to see the reflected contribution alone. Use `specular=off` for a matched composite, or `roughness=0.15` and `roughness=0.85` on a single-finish floor. The reflection-motion preset accepts `motion=left|right|still|sequence`. The GPU fixtures are `rc-native-reflections` and `rc-rough-motion`.

To compare reconstruction on the polished rail, run `pnpm test:gpu --gpu nvidia --filter rc-polished-quality`. The fixture replays the same 24 poses at 960 × 540 using the native interval solver throughout. It compares 1.5× and 2× temporal upscaling with full-resolution TRAA, repeats the full-resolution run to check reproducibility, and measures both spatial error and frame-to-frame changes in display RGB.

On NVIDIA run `node tools/gpu/pages/rc-polished-budget/run.ts` to collect three alternating on/off pairs with GPU and CPU reservations. Each arm replays the same 60 moving rail frames at 640 × 360 internal and 960 × 540 display. The timestamps span the G-buffer through the final canvas blit. The runner saves per-frame timestamps, temperature, clocks, pre-launch GPU utilization and a summary under `bench/screenshots/rc-polished-budget/`. For the independent sustained check run `node tools/gpu/pages/rc-polished-budget/run.ts <new-output-directory> --uncapped`. It replays the same first 60 ticks and continues to tick 607, capturing 480 frame spans per arm.

For the Apple Silicon run, use the [Metal handoff](metal.md). The portable kit includes the pinned tracer dependency. It builds the gallery, checks transport, shutter response and reconstruction, and runs the same three normal and three uncapped reflection pairs. Return the results archive to close the hardware phase.

Set `VITE_RC_AB_SIZE=960x540` to repeat the backend A/B at the capped preview size. The default is 640 × 360.

Reproduce the independent emitter-only numerical check with `node tools/gpu/pages/rc-door-response/emitter-oracle.ts`. It integrates direct transfer from the recessed emitting surface to the fixed receiver patch.

Branch results are in [PROGRESS](../../PROGRESS.md) and the [paired GPU measurements](../../bench/results/radiance-cascades.csv).
