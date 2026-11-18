# Implementation and measurement plan

| Stage | Work | Evidence to advance |
| --- | --- | --- |
| Reference rail | Pin the S5 solver and its notices. Build a small architectural passage with bounced color, a repeatable route and lighting cues. | The built example runs on a hardware adapter. Receiver pixels respond to the lamp. Record resource limits and timing scope. |
| Interval seam | Feed a known triangle fixture and explicit GPU ray records into the pathtracer dispatcher. Classify endpoints, misses, transparent intervals and the unobserved tail. | GPU records agree with analytical expectations at both sides of a seam. |
| Shared tracer | Keep the reference selectable. Implement a GI-owned surface-ray generator, closest-hit and hit shading, then deposit, merge and gather over the same ray directions and scene. | Compare hit distance, triangle/material identity, radiance and final irradiance in order. A/B on the rail at matched budgets, then on S5 Sponza. |
| Dynamic occlusion | Replay `scenes/manifests/closed-door.json` with updated instance transforms and BVH bounds. Refresh history near swept geometry. Extend to instanced balls after one door passes. | Measure dark-room to lit-room indirect ratio on closure, nonzero open response, and reopening recovery. Time traversal and refit separately. |
| Reconstruction and hardware | Compare native, reduced GI resolution and velocity/depth-aware upscaling. Evaluate a secondary NRC hook only if profiling identifies a bottleneck. | Alternate modes on the same adapter and cue, capture image pairs, then repeat on Apple Metal and Windows D3D12. |

## Reproduction

1. Run `pnpm install --frozen-lockfile` and `pnpm examples` from this GI checkout. Open the [architectural rail](http://localhost:5203/11-radiance-cascades/).
2. Replay the rail with `?stride=2` and `?stride=1`. The former schedules one primary ray per 2 × 2 cell and the latter one per pixel, before background rejection. Hold viewport, exposure, secondaries, probe capacity and camera clock constant. Switching to the direct view keeps GI computing, so use it for a lighting comparison.
3. Build with `PAGES_BASE=/gi/` and `pnpm examples:build`, then run `pnpm test:gpu --filter rc-rail`. In PowerShell set `$env:PAGES_BASE='/gi/'` before building. See the [interval experiment](../../experiments/rc-intervals/) for its source-checkout requirement.
4. Run the paired benchmark under the shared GPU lock with `node tools/reference/bench-lock.ts run --what rc-rail-budget -- node tools/gpu/run.ts --filter rc-rail-budget --json rail-budget.json`. This manual page alternates the order of stride 1 and stride 2 across three pairs, uses the demo’s 960 × 540 drawing size by default (set `VITE_RC_BUDGET_SIZE=480x270` for the smaller comparison) and measures the same 60 rail ticks in both arms. Receiver error uses normalized display RGB relative to stride 1.
5. For a performance decision take at least three alternating warmed runs at matched ticks. Record median and p95 frame times, cascade GPU timestamps, ray budget and allocated bytes. Compare post-cue images to stride 1 at the same pose. For radiometric error use a converged path-traced reference with the same materials and lights and compare linear HDR regions before tone mapping.
6. Keep startup, steady motion, a newly visible room and post-cue settling separate. Include indirect-only captures. For the door, reuse the existing suite's open/closed fixture and dark/lit normalization. A pass needs both light transfer through the open doorway and attenuation after closure.

Branch results are in [PROGRESS](../../PROGRESS.md).
