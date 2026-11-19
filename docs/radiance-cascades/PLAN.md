# Implementation and measurement plan

## Phase status

| Original phase | Status on the RC branch | Evidence or remaining work |
| --- | --- | --- |
| Reference rail | Passed | S5 Three.js port, fixed rail and lighting cues verified on NVIDIA. |
| Interval seam | Passed | GPU fixtures cover interval boundaries, misses, near occlusion and directional splits. |
| Shared tracer | Partial | The gallery runs both solvers with identical direct images. Sponza and the upstream Chandra renderer are separate comparisons. Equivalent indirect brightness remains open. |
| Dynamic occlusion | Passed | Both backends darken and recover on the moving shutter. The response curves and independent emitter integral are in the [progress log](../../PROGRESS.md). |
| Reconstruction and hardware | Partial | Windows hardware checks cover temporal sampling, resize and edge error. Apple Metal validation and sustained comparative timing remain open. |

The upstream Chandra build and polished-floor experiments extend the diffuse project. See the [experiment list](../../research/radiance-cascades/reflections.md) for their individual checks.

| Stage | Work | Evidence to advance |
| --- | --- | --- |
| Reference rail | Pin the S5 solver and its notices. Build a small architectural passage with bounced color, a repeatable route and lighting cues. | The built example runs on a hardware adapter. Receiver pixels respond to the lamp. Record resource limits and timing scope. |
| Interval seam | Feed a known triangle fixture and explicit GPU ray records into the pathtracer dispatcher. Classify endpoints, misses, transparent intervals and the unobserved tail. | GPU records agree with analytical expectations at both sides of a seam. |
| Shared tracer | Keep the reference selectable. Implement a GI-owned surface-ray generator, closest-hit and hit shading, then deposit, merge and gather over the same ray directions and scene. | Compare hit distance, triangle/material identity, radiance and final irradiance in order. A/B on the rail at matched budgets, then on S5 Sponza. |
| Dynamic occlusion | Replay a sealed side chamber with an off-screen constant emitter, a rigid sliding shutter, and a fixed receiver. Compare closure and reopening on both solvers. | Measure closed/open indirect luminance, sustained T50/T90 after travel ends, and recovery relative to the original open state. |
| Reconstruction and hardware | Compare native, reduced GI resolution and velocity/depth-aware upscaling. Evaluate a secondary NRC hook only if profiling identifies a bottleneck. | Alternate modes on the same adapter and cue, capture image pairs, then repeat on Apple Metal and Windows D3D12. |

## Reproduction

1. Run `pnpm install --frozen-lockfile` and `pnpm examples` from this GI checkout. Open the [architectural rail](http://localhost:5203/11-radiance-cascades/).
2. Replay the rail with `?stride=2` and `?stride=1`. The former schedules one primary ray per 2 × 2 cell and the latter one per pixel, before background rejection. Hold viewport, exposure, secondaries, probe capacity and camera clock constant. Switching to the direct view keeps GI computing, so use it for a lighting comparison.
3. Build with `PAGES_BASE=/gi/` and `pnpm examples:build`, then run `pnpm test:gpu --filter rc-rail`. In PowerShell set `$env:PAGES_BASE='/gi/'` before building. See the [interval experiment](../../experiments/rc-intervals/) for its source-checkout requirement.
4. Run the paired benchmark under the shared GPU lock with `node tools/reference/bench-lock.ts run --what rc-rail-budget -- node tools/gpu/run.ts --filter rc-rail-budget --json rail-budget.json`. This manual page alternates the order of stride 1 and stride 2 across three pairs, uses the demo’s 960 × 540 drawing size by default (set `VITE_RC_BUDGET_SIZE=480x270` for the smaller comparison) and measures the same 60 rail ticks in both arms. Receiver error uses normalized display RGB relative to stride 1. On a thermally limited adapter, set `VITE_RC_BUDGET_PAIR` to 1, 2 and 3 in turn and run each pair under its own cool-start lock. Keep all three clock verdicts.
5. For a performance decision take at least three alternating warmed runs at matched ticks. Record median and p95 frame times, cascade GPU timestamps, ray budget and allocated bytes. Compare post-cue images to stride 1 at the same pose. For radiometric error use a converged path-traced reference with the same materials and lights and compare linear HDR regions before tone mapping.
6. Keep startup, steady motion, a newly visible room and post-cue settling separate. Include indirect-only captures. For the shutter, normalize the fixed receiver to its measured open and closed states. A pass needs both light transfer through the open doorway and attenuation after closure.

The interactive gallery starts with the native solver and 1.5× temporal upscaling. For the native backend use `?backend=interval`. Select `?backend=three` for the port. Both support `scene=gallery|door`, `resolve=native|taa|upscale`, `ratio=1.5|2|3`, `sharpness=0..1` and `outputSize=preview|full`. The ratio is display width divided by render width. For a fixed receiver use `camera=receiver&view=indirect`. Use the manual GPU pages `rc-backends`, `rc-door-response` and `rc-reconstruction` for the controlled comparisons. `rc-native-core` verifies the solver contract on raw WebGPU.

For the glossy experiments, open the gallery with `scene=polished`, `scene=roughness` or `scene=reflection-motion`. Select the native interval backend and use `view=specular` to see the reflected contribution alone. Use `specular=off` for a matched composite, or `roughness=0.15` and `roughness=0.85` on a single-finish floor. The reflection-motion preset accepts `motion=left|right|still|sequence`. The GPU fixtures are `rc-native-reflections` and `rc-rough-motion`.

Set `VITE_RC_AB_SIZE=960x540` to repeat the backend A/B at the capped preview size. The default is 640 × 360.

Reproduce the independent emitter-only numerical check with `node tools/gpu/pages/rc-door-response/emitter-oracle.ts`. It integrates direct transfer from the recessed emitting surface to the fixed receiver patch.

Branch results are in [PROGRESS](../../PROGRESS.md) and the [paired GPU measurements](../../bench/results/radiance-cascades.csv).
