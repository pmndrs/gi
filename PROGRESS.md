# Progress log

Newest entries go at the top. One entry per work session on a task, under 15 lines. This is how agents hand work to each other across contexts, so write for someone with no memory of the conversation.

Template:

```
## YYYY-MM-DD · T0.x · agent (model)
Did: what changed, with file paths.
Verified: the command or check, and its result. Say which device and browser.
Not verified: anything not checked, and why.
Numbers: the measurements, or a link to the CSV rows.
Next: what's left or blocked, and who should take it.
```

Rules:

- "Verified" means you ran it and saw it pass. "It builds" is not "it works".
- Keep numbers here or in `bench/results/*.csv`. Don't create report files.
- If you learned something that will matter again, add it to `docs/NOTES.md` too.

---

## 2026-11-06 · T2.0 spike: deferred shadow-ray pass on Sponza · implementer (opus)
Did (spike code 06d5a18 on throwaway branch `m2-t2.0`, worktree `webgpu-ddgi-m2-t20`): `DDGITechniqueOptions.directLight: 'inline' | 'deferred'` switches the paths in one build (default inline, M1's kernel untouched). Deferred: `wgsl/trace-deferred.ts` writes a 24-byte surface record (position, octahedral shading and ray-side normals, 10:10:10 albedo) plus the ray index (28 B per item) into a compacted work list, one `atomicAdd` per subgroup with `subgroupExclusiveAdd` offsets, or per hit (`workList: 'atomics'`); a 1-thread kernel writes the indirect args; `gi/direct` (`platform/shade/direct.ts`) rebuilds the surface, calls the same `giDirectIrradiance` and adds albedo/π × direct into the ray's radiance. The kernel builder now hoists a kernel's leading `enable` lines above the modules. Pages: `ddgi-batch-cost` gets `VITE_DDGI_DIRECT`, `VITE_DDGI_WORK_LIST` and a `bound` section (GPU-bound curve, 16–512 probes, two frames in flight, per pass, with fit and a page-only shadow-ray count); new `ddgi-direct-spike` (field match) and `trace-bench-sponza-anyhit` (our tracers only).
Verified, timing (RTX 3060 Laptop, headless Chrome 156, Sponza, one VALID run per path under the clock rule: inline started 47 C, deferred 49 C, both 0% thermal samples; SM median 1,852 / 1,755 MHz, power-limited). Update GPU ms (all `ddgi/` passes but the census), GPU-bound:

| Probes | 16 | 32 | 64 | 128 | 256 | 512 | Fit (16–512) | At 2 ms: fit / curve / Budget v2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Inline (M1) | 0.674 | 0.883 | 1.141 | 1.389 | 1.762 | 2.831 | 0.764 ms + 4.07 µs/probe | 304 / 313 / **293** (1.87 ms) |
| Deferred | 0.548 | 0.651 | 0.694 | 0.793 | 1.055 | 1.723 | 0.528 ms + 2.28 µs/probe | 647 / 618 / **646** (2.02 ms) |

- **Ratio: 2.20× (Budget v2), 2.13× (fit), 1.97× (curve). Decision rule (≥ 1.25×): go ahead with T2.2.** Far from the threshold, so no second run.
- Both parts dropped: fixed −31%, per probe 0.56× (the design guessed 0.6×). Deferred at 512: trace 1.17, `gi/direct` 0.41, args 0.002, blend 0.14, write 0.008 ms. Inline trace at 512: 2.68 ms; its breakdown: full 2.52, without direct 1.10.
- **Shadow-ray rate:** Sponza gives 81 work items and 38.6 shadow rays per probe (half of the front-face hits face away from the sun). `gi/direct`: 48 Mrays/s at 512 probes (19.8k rays in 0.41 ms), 22 at 128. Inline, from its breakdown (full − without direct): 14 Mrays/s at 512, 6.6 at 128. The trace bench's any-hit rate in the same worktree: 245 Mrays/s (ours-tinybvh), 214 (platform), at 878k rays a dispatch. So deferred runs at about 1/4.4 of the bench rate (inline 1/15); a 20k-ray dispatch can't fill the GPU.

Verified, correctness (`ddgi-direct-spike`, `--no-cool-wait`, 300 frames of 4,096 probes or every probe; test-view GI image mean per bookmark and the irradiance atlas mean against inline): **pass on both GPUs.** RTX: Cornell (emissive, no lights) 0 exactly; Cornell point light −0.023% to −0.033%; Sponza −0.003% to +0.011%. Intel UHD: Cornell 0; Cornell point −0.026% to −0.036%; Sponza −0.002% to +0.012%. The subgroup and atomic work lists give identical fields (gi/direct's random numbers follow the ray index, not the list order). Also: `pnpm lint`, `typecheck`, `test` (947) pass; on the RTX `ddgi-*`, `hit-shading-*` and `trace-inline` (12 pages) pass with the spike in the build.
Not verified: the Intel's timing (correctness only); Sponza with many lights (one sun here, so no light choice and no soft-shadow noise); two-room and Bistro costs; a second run per path (the rule didn't need one).
Numbers: `timing.csv` `batch_cost/inline/...` and `batch_cost/deferred/...` (commit 06d5a18: `bound_batch<n>`, `bound_fit`, `shadow_rays`, `v2_budget2ms`, `startup`, inline `breakdown`; `note/timing_valid,1`); `trace.csv` 06d5a18 `ours-tinybvh/any` and `platform/any`.
Next (T2.2): append only hits with at least one light reaching their side (halves Sponza's list; cheap sun test in the trace); fold rays and work list into one ray scratch (the deferred trace binds 8 storage buffers, the default limit); size the list per blended ray and in 1,024-probe rounds (the spike's is 28 B per ray: 29 MB at a 4,096-probe batch); albedo above 1 (the spike clamps); `giDirectLightSamples`; pass attribution for a platform `gi/direct` (the spike times it as `ddgi/direct`).

## 2026-11-06 · T1.12 flicker against Dennis's threshold · implementer (opus)
Verified (RTX 3060, Chrome 156, `ddgi-flicker`, `--no-cool-wait`, default rule τ 5 s with the ramp, 1200 frames settled, one run each; threshold mean ≤ 0.1%, p95 ≤ 0.5%, DECISIONS e7a6e88): **all pass.** At `gpuMs:2` (mean / p95 per-pixel change a frame): Cornell 0.082% / 0.34%, two-room 0.0099% / 0.032%, Sponza 0.027% / 0.096%. At the calibrated batch as a fixed count (1574 / 707 / 268 probes): 0.088% / 0.36%, 0.0096% / 0.031%, 0.024% / 0.081%. Cornell has the least margin (12–18% under the mean bar). Rows: `bench/results/flicker.csv`, `note/flicker_threshold_check`. No code changed.

## 2026-11-06 · T1.9 M1 measurements against the SPEC targets · gpu-tester (sonnet)
Did: one VALID run per target, no code changes (HEAD 3701d8f). RTX 3060 Laptop (nvidia, ampere), headless Chrome 156, under the bench lock. Leaks, memory and the Intel suite are correctness runs (`--no-cool-wait`); startup runs were timing runs (VALID by the clock rule). Rows added: `memory.csv` (48), `timing.csv` (3 x 35, `batch_cost/startup/...`, one scene per run, so the compile is cold). The leak page has no to-csv; its numbers are below.
Reused, not re-run: update (T1.9b), shading (T1.14), accuracy (T1.13, T1.14), native matched counts ("T1.9 quiet-window measurements").

| SPEC row (RTX, 1440p) | Number | Valid | Verdict |
| --- | --- | --- | --- |
| Cost of showing GI (<= 1.0 ms) | 0.73 ms through the upscaler Performance preset with half-res GTAO (plus the upscaler's own 1.55 ms); native composite 1.26-1.41 ms (T1.14) | VALID (T1.14) | met on the default path; native composite misses 1.0 |
| Cost of updating GI (<= 2.0 ms) | 12/12 runs within +-10% of 2 ms (T1.9b). Probes at 2 ms: Cornell 1,574 / two-room 707 / Sponza 268 / Bistro exterior 132 | VALID (T1.9b) | met |
| Whole frame, p50/p95 GI on vs off (<= 3 ms p50 = update + shading, no p95 spikes) | No existing page measures update plus shading in one frame on all three scenes without new code. Sum of rows: 2.0 + 0.73 = 2.73 ms on the default path (4.28 ms with the upscaler's 1.55 ms, which is the user's upscale and not GI); native composite 2.0 + 1.26-1.41 = 3.3-3.4 ms. Measured p95 not available. Earlier unvalid quiet-window deltas (power-state inflated, 1440p): Cornell +2.41 (p95 +4.83), two-room +3.18 (+2.85), Sponza +6.18 (+5.43) | sum of VALID rows | not measured directly; sum met (2.73) on the default path without the upscaler's own cost; p95 not measured |
| Leaks (<= 1%) | two-room 0.83% (lit 0.715, dark 0.0658; Cycles 0.676 / 0.0602); thin-wall 0.59%; closed-door 0.060% | correctness | met (3000 frames, `--no-cool-wait`, the clock was throttled and doesn't matter here) |
| Accuracy (+-10% mean) | Cornell +5.74%, two-room +5.50% (T1.13, HD); Sponza +6.59% (T1.14 default path, HD) | T1.13/T1.14 | met |
| Memory (guide 64 MB) | MB scene copy / acceleration / DDGI / total (+ sky 0.26-1.31): Cornell 1.33 / 0.17 / 5.64 / 7.14; two-room 1.33 / 0.17 / 3.89 / 5.40; Sponza 115.4 / 469.9 / 36.1 / **621.4** (622.7 with sky). Furnished room 40.7 earlier | deterministic | reported; Sponza fits a 6 GB card with room |
| Startup (<= 2 s from handoff, cold compile) | handoff / file request, first GI drawn: Cornell 1.54 / 1.77 s; two-room 1.16 / 1.29 s; **Sponza 4.50 / 7.23 s** | VALID (start 52-68 C, 0% throttled) | met on Cornell and two-room; **Sponza missed**: files 2.7 s, mesh trees 2.5 s, scene copy 0.85 s, compile 0.8 s (known gap, B2 and M3 per DECISIONS) |
| Light change / object move response | M2 | | not in M1 |
| Stability | flicker numbers from the 2026-11-05 entry; the threshold by eye is still with Dennis | | not measured here |
| Engine neutrality | `three-neutrality` passes on both GPUs | pass | met |
| Adapter cost | not re-measured | | not measured |
| Native RTXGI at matched counts | earlier entry: trace/blend Cornell 0.68/0.40, two-room 0.53/0.35, Sponza 11.4/4.02 ms | | cited |

Intel UHD (gen-12lp), the whole suite, `--no-cool-wait`: first run 30 PASS, 7 FAIL. `trace-bench-hploc-quick` hit **DXGI_ERROR_DEVICE_HUNG** (device lost), and every page after it failed with "no usable GPU" (trace-bench-lbvh-quick, trace-bench-quick, trace-inline, trace-oracle, viewer-baselines, wgsl-error): a cascade, not seven failures. Re-running those seven alone: the six real pages all PASS (hploc 20 s, lbvh 13 s, quick 4.6 s, trace-inline 76 s, trace-oracle 71 s, viewer-baselines 76 s). So every correctness page passes on the Intel, but one run of the full suite does not finish clean: the hploc quick bench hung the Intel's device once (not reproduced alone; GPU timeout on a slow adapter, I did not investigate). `ddgi-leak` passes on the Intel (165 s).
**Flag:** the self-test page `wgsl-error` reports plain FAIL when run alone on both the RTX and the Intel, with the expected error text ("Error while parsing WGSL: ... cannot initialize 'let' of type 'f32'") present. It looks like the runner's expect-failure handling under `--filter`, not a GPU issue. Not investigated.
Also: the two-room startup run had a median SM clock of 382 MHz under load but passed the validity rule (5 loaded samples, 0% thermal); its 1.16 s may be slower than a hot-clock run.
Next: whole-frame p50/p95 on GI on vs off with the update included needs a page (the three composite pages hold the field frozen); stability threshold with Dennis; Sponza startup (files, trees, copy) is B2/M3.

Lead corrections after the Opus review (pass with notes):
- The VALID verdicts for startup come from bench-lock's end-of-run lines; the to-csv ran outside `bench-lock run`, so the `timing.csv` rows carry no `note/timing_valid` and can't show it themselves.
- `wgsl-error` failing under `--filter` is by design (`tools/gpu/run.ts:179-183` expects the failure only in a full run), not a runner bug.
- Accuracy: Sponza's +6.59% is the composite default path with GTAO; the raw field and the simple `GIVolume` per-fragment path read +23.07% (T1.13). FLIP: no threshold is set yet (SPEC asks for one in M1); HD values Cornell 0.104, two-room 0.069. Proposed to Dennis with the M1 note.
- Stability: the gate is T1.12's eye check by Dennis; current numbers (after T1.12 2a) are Cornell 0.095% and Sponza 0.027% mean change a frame, not the 2026-11-05 figures.
- Object-move response is M3 in SPEC, not M2. Adapter cost: T1.7 measured 0.09–0.14 ms still, 0.11–0.19 ms moving (not re-run). Neutrality: `three-neutrality` passes in this entry's RTX and Intel suite runs. The Bistro exterior baseline re-measure was skipped (Dennis 2026-11-06, lean T1.9).
- Whole frame with the measured Sponza update median (2.02 ms): 2.75 ms. The verdict holds.
- The Intel has now hung (`DXGI_ERROR_DEVICE_HUNG`) twice in two days during long back-to-back runs (T1.14's `three-examples`, this entry's `trace-bench-hploc-quick`). It's a pattern to look at, under Next.

## 2026-11-06 · T1.12 phase 2b review fixes: the change view's colours are exact · implementer (opus)
Did (uncommitted, on top of b002f42), from the Opus review:
1. **Change view in display space.** `viewer-core.ts` turns the pipeline's output transform off and applies `renderOutput()` to each technique's picture itself; a technique may then draw `overlay(display)` after tone mapping and output colour (`types.ts`: `overlay`, `overlayKey`). `frame-change.ts` paints there with a linear share, change / scale: full colour means a change of at least the scale, half colour half of it, under any tone mapper and exposure. The weight^2.2 is gone. The atlas overlay moved to display space too (on top of the change view; distance greys are no longer tone mapped).
2. **Probe spheres keep their colours** under the change view: they write alpha 0 (`NoBlending`, since three forces opaque alpha to 1), DDGI's pass clears alpha to 1, and the change view shows marked pixels unchanged.
3. **"Fill the picture"** places the atlas below the stats box (`viewer-app.ts` passes how far down the box reaches; 1 px per texel there on Cornell at 1280x720, 2 px with the stats hidden).
4. `atlasev` snaps to the slider's 0.5 steps (test); `probe-history.ts` uses `GPUTextureUsage`/`GPUBufferUsage`. Help text: what full colour, half colour and the speckle mean; spheres keep colours; fill stays below the stats.
Verified (RTX 3060, headless Chrome 156): share of pixels in exact full colour on screen (canvas drawn to a 2D canvas, 6 frames) against the share whose change reaches the scale (temporary readback of the two pictures, 8 frames, removed): GI only 0.25% ACES 5.1% / 4.8%, AgX 4.6% / 4.3%, Linear +1 EV 4.7% / 4.2%; full picture 0.25% 0.58% / 0.48%; GI only 1% 0.007% / 0.014%. At 0.25% GI only, 19% of pixels reach half colour or more and 60% change at all. Screenshots in `bench/screenshots/t1.12-debug-views/` (new 7, 8, 9, 10, 11, 13 change 1%, 14 change with history spheres, 15 wipe, 16 upscaler with spheres, 17 light probe grid), all 0 errors. lint, typecheck, `pnpm test` 944/944, `pnpm build:apps`.
Not verified: the Intel this round; the GPU suite not re-run (no core change since b002f42).
Next:
- **Dennis** (replaces the instructions in the entry below): Cornell box, camera still, GI only (indirect), "Change per frame: 0.25%". Full orange or blue = that pixel moved 0.25% or more since the last frame; faint scattered speckle = the picture's own precision, not flicker. Then 1%: what is still coloured moves by 1% a frame. Show the probes by "How settled" (they keep their colours in this view), and the irradiance atlas filled at +2 EV (hide the stats for a bigger atlas). Say whether the flicker is acceptable: that sets the threshold.
- The probe spheres look speckled through the upscaler; it is the upscaler's reconstruction (same with the old alpha), not this change.

## 2026-11-06 · T1.14 cleanups after the final Opus review · implementer (opus)
Did (uncommitted):
- `packages/ddgi-three/src/composite.ts`: the GI evaluation is back inline in the composite (no helper, no `Vec2Node`/`FloatNode` aliases, none of their casts); the stray blank lines are gone; a default GTAO replaced when the pass's scale crosses 1 is `dispose()`d; the `occlusionNode` comment says it is re-chosen at any build where the scale has crossed 1.
- `render-time-update.test.ts`: the default occlusion is tested through `setup()` (the build): it's made when the pass is reduced after `ddgi()`, then dropped and disposed when the pass is back at full resolution. This replaces the test of the private method.
- `docs/api/three.md`: the base package's −1.2% is against its 640×360 reference.
- `examples/three` is a workspace package (`@pmndrs/ddgi-example-three`: `@pmndrs/ddgi` workspace, `@pmndrs/upscaler` 0.5.0, `three` 0.187.0, `vite`), listed in `pnpm-workspace.yaml`. The root has `@pmndrs/upscaler` 0.5.0 as a devDependency for the GPU pages (`three-composite-shared/variants.ts` now imports it by name). The aliases into `apps/viewer/node_modules` are gone (examples vite config, `tools/gpu/server.ts`, tsconfig paths). `pnpm install`: the lockfile gained 19 lines, the two importer entries and nothing else (no new packages resolved).
Verified (RTX 3060, headless Chrome 156):
- lint, typecheck and `pnpm test` (944 passed);
- `pnpm build`, and `pnpm --filter @pmndrs/ddgi-example-three build`;
- `test:gpu --filter three-`: 9 of 9 (Cornell point 0.042% / 0.16%; examples GI 45.8% simple, 44.8% composite through the upscaler).
Not verified: the Intel this round (the lead ran all 9 on it before these cleanups).
FLIP on the default path (upscaler Performance, GTAO at half the render resolution; RTX; `three-composite-ao/run.ts --upscale performance --resolution-scale 0.5`, which now also writes each upscaled GI term as an EXR, then `tools/compare --downsample 8`, FLIP at 8 ppd), against the 1920×1080 references: **Sponza FLIP 8×8 0.283** (over the 0.15 bar; bias +6.59%, 8×8 block error mean 18.3%, p95 114%); **Cornell 0.124** (within it; bias +0.22%, block error mean 8.8%, p95 40.4%). FLIP's output has a mean and a weighted median but no p95 (Sponza 0.408, Cornell 0.410), so the p95 given is the 8×8 block error's.

## 2026-11-06 · T1.12 phase 2b: the viewer's debug views for the flicker eye check · implementer (opus)
Did (uncommitted; DDGI section of the viewer panel; hover help for each in `apps/viewer/src/ui/help-text.ts`; link keys in `url-state.ts` and the viewer README):
- **Probe atlas over the picture** (`src/ddgi/atlas-overlay.ts`): irradiance or distance, bottom-right corner or "Fill the picture", one block per layer of probes (lowest first, top left), sharp texels; "Atlas exposure" (stops) for irradiance. Reads the same textures three is lit from.
- **Probes coloured by history** ("Colour the probes by: How settled"): n, the updates each probe's light is averaged over, red (1) to yellow (~13) to green (200, the most kept), log scale (`history-colors.ts`); inactive probes grey. Core addition: `DDGIField.probeHistory` (the blend kernel's history buffer) and `PROBE_HISTORY_WORD`/`PROBE_HISTORY_WORDS` in `layout.ts`, with a unit test tying them to the WGSL struct; the viewer copies n into an r32float texture with a small compute pass (`probe-history.ts`).
- **Direct against indirect:** the existing GI radios, relabelled "Off (direct)", "On (both)", "GI only (indirect)", with clearer help.
- **Change per frame** (`frame-change.ts`): with the camera still, orange where a pixel got brighter since the last frame, blue where darker, over a dim grey picture; full colour at 0.25%, 1% or 4% (weight^2.2, so it reads linearly after sRGB). Works on top of GI only and the screen views; no colour while the camera moves; off under the upscaler.
- `scene-pass-technique.ts` gains `sceneTarget`/`upscaling` accessors; `ddgi-technique.ts` wraps `outputNode` with the overlays only while one is on.
Verified (RTX 3060, headless Chrome 156 via `apps/viewer/scripts/contact-sheet.ts`): every view on Cornell (`cornell-box`, camera 1), 0 errors; screenshots in `bench/screenshots/t1.12-debug-views/` (1-normal ... 12-panel-controls). A readback of the two pictures the change view compares (temporary, removed): settled Cornell changes 0.055% mean / 0.16% p95 per pixel a frame (own-brightness metric), GI only 0.086% / 0.26% (flicker-page metric 0.04% / 0.16%), in line with phase 2a; 18-25% of pixels change by exactly 0 (half-float target). lint, typecheck clean; `pnpm test` 943/944, the one failure `tools/assets/fetch.test.ts` (listed flaky under load) passes alone 104/104; `pnpm build:apps` builds. `test:gpu --filter ddgi-` under bench-lock (`--no-cool-wait`, "T1.12 2b"): RTX 8 of 8 pass, numbers unchanged (furnace −0.15%, leak 0.83%, test view +5.63%).
Not verified: the Intel UHD (the viewer views and the GPU suite); Sponza; an interactive session in a real window (headless screenshots only).
Next:
- **Dennis:** open the viewer on the Cornell box, keep the camera still, and try: GI only (indirect) plus "Change per frame: 0.25%" (where and how much it flickers); "Show the probes" coloured by "How settled" (green = averaging 200 updates); "Probe atlas: Irradiance", filled, +2 EV (squares that shimmer are noisy probes). Then say whether the flicker is acceptable: that sets the threshold.
- Phase 2a said the viewer's warm-up `hysteresis` override can go (unused under the adaptive rule); not done here, it's harmless.
- Possible later: a number for the change view in the stats box (mean / p95 per frame) would need a readback; not built.

## 2026-11-06 · T1.14 default locked: the upscaler's Performance preset with half-resolution GTAO · implementer (opus)
Did (uncommitted):
- **HD re-measure** (`three-composite-ao/run.ts`, whose `sponza` default is now the HD reference: `cycles-untextured-gi.exr` at 1920×1080 with `gi-materials.json`, `7680a2d0…`). The GI term through `@pmndrs/upscaler` at the stored camera, GTAO 1 m, 16 samples, at half the render resolution, RTX:
  - **Performance:** Sponza **+6.59%** (curtain pixels +15.7%); base package −1.19% (its 640×360 reference, the only one so far); Cornell +0.22% (1920×1080, 600 passes);
  - **Quality:** Sponza +6.06% (curtains +14.6%); base −1.53%; Cornell +0.21%;
  - without the upscaler, half-resolution GTAO reads +5.83% (curtains +13.6%) at HD.
  - **Chosen by the DECISIONS rule (Performance if Sponza is within ±10%): Performance.** No cost re-runs: the cost study's numbers stand (0.73 ms for showing GI; the upscaler's own 1.55 ms).
- `packages/ddgi-three/src/composite.ts`: **the default occlusion follows the pass.** On a reduced pass (`getResolutionScale() < 1`, the upscaler's path) it is GTAO at half the pass's resolution, 1 m, 16 samples. On a full-resolution pass it is none (native GTAO is opt-in, its cost in docs). The choice is made at construction and again at the build, so a scale set after `ddgi()` counts. `occlusionNode` is now a getter. No new exports, and the names are unchanged.
- `examples/three/composite.*`: the upscaled path (Performance: `rtt(ddgi(...), …, { type: HalfFloatType, resolutionScale })` into `upscale(...)`, `velocity` in the MRT, `antialias: false`). `@pmndrs/upscaler` resolves through aliases (examples vite config, `tools/gpu/server.ts`, tsconfig paths) to the viewer's copy, because `examples/three` isn't a workspace package.
- `docs/api/three.md`: the intended pipeline (the upscaler, with the composable wiring), a cost and accuracy table, the upscaler's own cost as a separate line, native GTAO opt-in with its cost, and the default following the pass.
- `tools/gpu/pages/three-examples`: pictures are read inside the running animation loop, because the upscaler only advances on a new frame.
- Tests: the default (GTAO at 0.5 on a reduced pass, none on a full one, re-chosen at the build).
Verified (headless Chrome 156, RTX 3060):
- lint, typecheck and `pnpm test` (942 passed);
- `test:gpu --filter three-`: 9 of 9, the Cornell match unchanged;
- `three-examples`: both examples render with 40 GI frames in 40 animation frames; GI is 45.7% of the canvas (simple) and 44.8% (composite, through the upscaler). The screenshot shows clean colour bleed, and a little speckle on the short box's top under the sphere (likely the half-resolution GTAO at a 400 px-tall render).
Not verified: the Intel for this round (`--filter three-` was asked on the RTX only); a 1440p display with a moving camera.
Usage bar, updated: the composite example is now 14 GI lines (the simple 5, plus the `three/tsl` and `@pmndrs/upscaler` imports, the render scale, `RenderPipeline`, `pass`, `setMRT`, `setResolutionScale`, `rtt`, `outputNode`); LightProbeGrid 5, ours simple 5. Names a user touches for the upscaled path: ours `GIVolume`, `ddgi`, `giAlbedo`; three's `RenderPipeline`, `pass`, `mrt`, `output`, `normalView`, `packNormalToRGB`, `velocity`, `rtt`, `HalfFloatType`; the upscaler's `upscale`, `QualityMode`, `getQualityModeRatio`.
Next:
- `examples/three` as a workspace package with its own dependencies (three, `@pmndrs/upscaler`), replacing the aliases;
- the viewer's warm-up switch-over to `volume.start` and a composite toggle, once `apps/viewer` is free;
- the base package's HD reference, to replace its 640×360 numbers.

## 2026-11-06 · BVH package docs · side session (opus)
Did (docs only, branch `bvh-docs`; no code changed): `packages/bvh/README.md` rewritten for users (what it is, the builder contract, which builder to use when, one short example per path, the measured numbers with their conditions, licences, status: GPU mesh builders not wired into ddgi-core until M2's fast first GI). The design text it had moved, lightly reordered, to `docs/design/bvh-builders.md`. New `docs/design/bvh-package-split.md`: an outline (not a decision) for splitting out `@pmndrs/bvh`. `packages/bvh/package.json` description: bundled into `@pmndrs/ddgi` for now, standalone later (still `private`). `bvh-framework.md` links the new docs.
Verified: `pnpm lint` and `pnpm typecheck` clean, `pnpm test` 912 passed (6 skipped). The README examples were checked against the source signatures by reading them, not run.
Next: Dennis reads the split outline. Its open question (move the ray walk into the package or not) is best decided after M2.

## 2026-11-06 · T1.14 after the cost study: half-resolution GI deleted, the composite's GTAO sized from the pass · implementer (opus)
Did (uncommitted; defaults and exports unchanged):
- `packages/ddgi-three/src/composite.ts`:
  - the internal half-resolution GI (`irradianceResolution`) is deleted. The study below has its numbers: it saved 0.23 ms and missed the match by 1% / 4% untextured and 4% / 20% with normal maps. NOTES has the reason;
  - the composite's own GTAO is now a `PassSizedGTAONode` (a GTAONode whose `setSize` scales the drawing-buffer size by the scene pass's `getResolutionScale()`), so under a reduced pass (the upscaler) it runs at the pass's size, and `occlusion.resolutionScale` is relative to the pass (0.5 = half the pass's pixels at any upscaler preset).
- The pages: the half-resolution options are removed. The upscaled variant now uses the composite's own GTAO instead of the page's scaled copy (variants.ts).
- Test: a unit test that GTAO's target is 640×360, 854×480 and 1280×720 for a 2560×1440 drawing buffer at pass scales 0.5, 1/1.5 and 1, with GTAO at 0.5.
Verified (headless Chrome 156):
- lint and typecheck clean; `pnpm test` 917 of 918;
  - the one failure is `tools/reference/bench-lock.test.ts` ("a nested run under it does not deadlock"), which timed out at 60 s while the RTX was warm and passed (37/37) once it was under 70 C. It runs a real `bench-lock run`, which waits for a cool GPU. For the bench-lock owner;
- `test:gpu --filter three-`: RTX 9 of 9, the composite's match unchanged (Cornell point 0.042% / 0.16%);
- Intel UHD: the run straight after the RTX suite lost the device in `three-examples`, during the simple example's `volume.start`, and the two pages after it found no adapter. The same happened once in the study above, also straight after the RTX run.  Run alone, the suite passed 9 of 9 (Cornell point 0.033% / 0.083%), and `three-examples` passed twice more. So it's intermittent, on the Intel only; the cause is unknown.
Next: the Intel device loss in `three-examples` after a long GPU session (2 of 6 Intel suite runs). Worth a look if it shows up again outside back-to-back RTX + Intel runs.

## 2026-11-06 · T1.14 cost study for Dennis: GTAO variants, half-resolution GI, the upscaler · implementer (opus)
Did (uncommitted; measurement only; defaults and exports unchanged):
- `packages/ddgi-three/src/composite.ts`: an internal setting, `DDGICompositeInternalOptions.irradianceResolution: 'half'`. It's on the constructor only, not in `DDGICompositeOptions` and not exported from the package. GI is evaluated per 2×2 block in a pass of its own, then upsampled with weights for bilinear position, depth (5%) and normal (cos⁸), with a full-resolution evaluation where nothing fits. The default path's GI evaluation moved into a shared helper.
- `tools/gpu/pages/three-composite-shared/variants.ts` (new, page-only):
  - the composite through `@pmndrs/upscaler` by its documented composable path: the pass at 1/ratio with `velocity` in the MRT, as `upscaleScene` builds it; the composite made a texture with `rtt(…, { resolutionScale })`; then `upscale(color, depth, velocity, camera, { quality })`. The viewer's toggle uses `upscaleScene`, which can't take a composite;
  - GTAO averaged over frames (`AccumulatedOcclusion`: GTAO's `useTemporalFiltering` plus a plain or exponential mean over 16 frames; static camera only, nothing reprojects the history);
  - three's SSAO.
- `three-composite-sponza`: `VITE_DDGI_SET` = `native` or `upscaled`, the upscaler's own compute time per variant (`gpuTiming`), and `VITE_DDGI_IRRADIANCE_RESOLUTION=half` for the match, with silhouette and interior pixels reported apart.
- `three-composite-ao` (page and run.ts): `--ao-kind gtao|gtao-temporal|ssao`, `--irradiance-resolution half`, `--upscale quality|performance`. The last gives the GI term at the output size through the upscaler's own pipeline (its jitter, 32 frames on a still camera), with GTAO inside.
Verified:
- lint, typecheck, `pnpm test` (917 passed);
- `test:gpu --filter three-` passes on the RTX 3060. On the Intel UHD the first run lost the device (`DXGI_ERROR_DEVICE_HUNG` in `three-examples`, then no adapter for the two pages after it); the re-run passed all 9 pages.
- The composite's match is unchanged on both GPUs (Cornell point 0.042% / 0.16% RTX, 0.033% / 0.083% Intel).
- All cost runs below are VALID under the clock rule (cool start, 0% thermal samples; bench-lock), GPU-bound (100% GPU use in all 31 samples), headless Chrome 156.
**The table for Dennis.** Sponza with curtains, RTX 3060 Laptop, frozen field. Cost = showing GI − the same path without GI, 2560×1440 output, raw ms, with the figure normalised to 1,500 MHz in brackets (SM 1,185–1,395 MHz; power-limited, not thermal). Accuracy is the GI term against untextured Cycles: Sponza at the 640×360 reference (overall / curtain pixels), the base package, and Cornell at 1920×1080 with 600 passes.

| Path (2560×1440 output) | Cost of showing GI | Sponza | curtains | base | Cornell |
| --- | --- | --- | --- | --- | --- |
| Composite, no occlusion | 1.26–1.41 (1.05–1.15) | +23.0% | +43.5% | +3.2% | +5.6% |
| + GTAO full res, 16 samples (default today) | 8.0–8.1 (6.5–6.6) | +4.4% | +10.7% | −2.6% | +0.5% |
| + GTAO half res | 3.2 (2.6) | +8.0% | +18.0% | −1.5% | +0.7% |
| + GTAO quarter res | 2.0 (1.66) | +7.9% | +17.5% | −1.4% | +0.65% |
| + GTAO half, 16-frame average (static camera only) | 3.2–3.3 (2.7) | +2.3% | +11.9% | −6.7% | −4.8% |
| + three's SSAO (half res, its blur) | 2.3–2.4 (1.9) | +14.8% | +31.0% | −0.7% | +1.0% |
| Half-resolution GI, no occlusion | 1.17–1.19 (0.96) | (with full GTAO: +4.3%) | | | |
| Forward per fragment (on the MRT pass − mrt) | 3.43 (2.86); 3.89 (3.25) over `off` | | | | |
| **Upscaler Quality (1.5×), composite** | **0.66 (0.61)** | +22.9% | +43.9% | +3.0% | +5.4% |
| Upscaler Quality + GTAO at render res | 2.9–3.1 (2.7–2.9) | +8.6% | +20.6% | −1.8% | −0.3% |
| Upscaler Quality + GTAO half of render res | 1.25 (1.16) | +9.1% | +21.1% | −1.4% | +0.2% |
| **Upscaler Performance (2×), composite** | **0.41 (0.38)** | +22.9% | +43.9% | +3.0% | +5.4% |
| Upscaler Performance + GTAO at render res | 1.59 (1.47) | +7.9% | +19.5% | −2.4% | −0.3% |
| **Upscaler Performance + GTAO half of render res** | **0.73 (0.68)** | +10.0% | +23.2% | −1.2% | +0.2% |

- **The upscaler itself:** 1.72 ms (Quality) and 1.55 ms (Performance) of compute per frame. The scene without GI costs 2.58 / 2.04 ms through it, against 2.96–3.05 ms native. On this scene, then, upscaling costs more than it saves (2.58 + 1.72 = 4.30 ms against 3.0). That cost is the same with and without GI, so it isn't in the deltas above.
- **Under the 1.0 ms bar:** the composite without occlusion through either preset (0.41–0.66 ms), and the Performance preset with GTAO at half of its render resolution (0.73 ms, Sponza +10.0%, right at the ±10% edge). The Quality preset with half-res GTAO is 1.25 ms, at +9.1%. Natively, nothing with occlusion is under 1 ms; quarter-res GTAO is the cheapest at 2.0 ms, +7.9%.
- **Accuracy caveat:** the Sponza reference is 640×360, so through the upscaler the GTAO behind these numbers ran at 160×90 to 427×240. At a 2560×1440 output it would run at 640×360 to 1707×960, finer than measured. The curtain error grows as GTAO gets coarser (+10.7% at full res, +17.5% at quarter, +23% at the coarsest here). A 1440p reference would tell how much of that is resolution.
- **Half-resolution GI:** it saves only 0.23 ms (1.39 → 1.17), against the hoped-for three quarters: most of the composite's 1.3 ms is reading the targets, not the probe lookup. It misses the composite's own match bar badly:
  - Sponza with the GI table: mean 0.99%, p95 4.1%; silhouettes (1.5% of pixels) 4.6% / 20%, interior 0.95% / 4.0%;
  - with Sponza's own materials: 4.0% / 20%. A 2×2 block shares one normal while normal maps vary per pixel.
  - The upscaler's Performance preset already shades GI at a quarter of the pixels with a temporal reconstruction. Recommendation: delete the internal half-res option (about 150 lines in composite.ts) rather than keep it.
- **GTAO averaged over frames** darkens more than one frame of GTAO (Sponza +2.3%, but Cornell −4.8%, base −6.7%). GTAO's temporal pattern is meant for TRAA, which three applies to the whole image, not to GI alone. It costs the same as half-res GTAO, so it isn't worth it.
- **SSAO** is cheap but too weak on the curtains (+31%).
- **GTAO's size:** it sizes itself from the renderer's drawing buffer, not the pass. Under the upscaler its `resolutionScale` has to include 1/ratio, which the page sets. The production composite's own GTAO doesn't yet follow a pass's `setResolutionScale` (Next).
Not verified:
- accuracy at 1440p (no 1440p Cycles reference);
- the upscaler with a moving camera (all still; the upscaler's history and our GTAO average behave differently in motion);
- the Intel for any of the new variants (correctness only; Intel has no performance target).
Next (lead / Dennis):
- choose the shipped path: for example, the upscaler (Performance or Quality) with half-res GTAO on GI, or no occlusion by default;
- if the upscaler is the intended pipeline, the composite's own GTAO should scale with the pass's `setResolutionScale`;
- delete the internal half-res GI option;
- a 1440p untextured Cycles reference for Sponza, to confirm the curtain numbers at the real resolution.

## 2026-11-06 · T1.12 phase 2a review fixes · implementer (opus)
Did (uncommitted, on top of 76ca4c9), from the Opus review of 14c2ed2 and 58b6dcc:
- **The clock wraps every 4096 s** (`CLOCK_WRAP_SECONDS`, settings.ts). The design doc's "millisecond precision for days" was wrong: an f32 `t` steps 0.49 ms below 2^12 s, 7.8 ms after a day and 62.5 ms after a week. The technique keeps its clock modulo 4096 s; `ddgi/blend` adds the period back when `T − t` is negative (that replaces the `max(0, …)` guard; a new `prepare` makes a fresh buffer anyway). It would misread only a probe left unblended for more than 68 min while active; active probes are blended every sweep, and off probes restart from state 3. Design doc "The clock" corrected.
- A NaN or negative `hysteresisRampUpdates` now falls back to the default (16), like every other setting; Infinity is still an endless ramp. Test updated.
- Comments: `hysteresis-modes.ts` (the default rule is `tau:5/200/8/16`; `tau:5/200/8/endless` names the endless ramp) and the `blend.ts` header (n, t and k).
Verified: lint, typecheck, DDGI unit tests 43. GPU: `test:gpu --filter ddgi-` passes on the RTX 3060 with unchanged numbers (furnace −0.15%, leak 0.83%, test view +5.63%), so the kernel compiles and behaves; no run reaches the wrap. Not run on the Intel this time.
Not verified: a run past 4096 s of clock (the wrap path itself only by reading it).
Next (other lanes; I didn't edit them):
- **Calibration runs with `deltaTime` 0** after the warm-up (the viewer, and `packages/ddgi-three/src/volume-start.ts:152`), so the clock stays frozen and probes keep counting up with no fade. My estimate:
  - Cornell: no effect, since it reaches n_max at 60 Hz anyway.
  - Sponza on the RTX: about 410 calibration frames at 291 probes is about 6 more updates each, so n goes from 2 to about 8 (h 0.89 instead of its 60 Hz steady 0.80). Once the clock runs, n fades back to about 4 within 4–5 updates (about 5 s at 60 Hz).
  - Upper bound: a 1,000-frame calibration with 1,024-probe burst batches would leave n near 50 (h 0.98) and take about 12 updates (13 s) to return.
  - So for a few seconds after loading, Sponza is a little quieter and up to 2× slower to follow a change. Nothing changes behind the loading screen, so I'd leave it. Passing the real elapsed time during calibration would remove it.
  - The real risk is a host that passes `deltaTime` 0 (or a constant tiny value) at runtime: the clock never moves, every probe climbs to n_max, and Sponza is back to minutes. Worth a check that ddgi-three's render-time update passes measured time.

## 2026-11-06 · T1.14: a GPU-bound cost page, the cost numbers, and the Opus review fixes · implementer (opus)
Did (uncommitted):
- **Cost page** (`three-composite-sponza`, `three-composite-shared/composite-check.ts`, `run.ts`). The lead's runs left the GPU about 20% busy. The cause was CPU, at 130–280 ms to record each frame:
  - two passes with the same targets but different lights (`off` and `forward`) made three rebuild their objects every frame;
  - the sun's shadow map, redrawn under each variant's pass context, cost 130 ms a frame.
  - The fix: each variant has its own pass and camera, and the GI light sits on a layer only the GI cameras see. `off` is the only single-target pass, and `forward` writes the same MRT as `mrt` but unmarked (so `forward − mrt` is GI per fragment on the same pass). The shadow map is drawn once; the scene is static, and it's the same in every variant.
  - Batches stay in flight: the CPU stays a batch ahead and reads batch k − 1's timestamps while batch k runs, with no other waits.
  - Each variant's CPU recording time is reported: about 2.5–3 ms now.
  - `run.ts` drops its own verdict and prints bench-lock's, plus each section's GPU use (and whether it's ≥ 90%). The two sections are labelled: "section 1 of 2: 2560 x 1440, the bar" and "section 2 of 2: 1920 x 1080, the composite pass at the reference size".
- **Review fixes:**
  1. *LightProbeGrid double count:* at the lighting model's `indirect` step (after every light), the GI light writes `GI_COMPOSITE_WON`: 1 only where the slot still holds GI's claim (GI's priority and GI's zero irradiance). `giAlbedo` multiplies by it.
  2. *A lost mark:* the composite checks the pass context at build and on every render, merges the mark back and warns once. A change after the first frame still shows for one frame (documented).
  3. *Reproducible WGSL check:* new page `three-unmarked-wgsl`. The unmarked Standard and Lambert WGSL is the same before and after a composite pipeline is built and drawn, and holds none of the composite's symbols. The comparison with the pre-T1.14 build (part 1) was a one-off and isn't repeated.
  4. *Checks:* three new cases on the Cornell page (`grid`, `ao`, `environment`) and four unit tests (the AO-twice warning, the mark re-merge, `ThreeGILimitError`, the frame-keyed guard). `ddgi()` now warns when `occlusion: { source }` meets a pass that already applies AO.
  5. *Render-time guard:* it keys on three's frame id when `setAnimationLoop` drives the frames, and falls back to the JavaScript task otherwise. The caveat (awaits between two renders of one frame without the loop) is documented.
  6. *Docs:* three.md (0.04–0.08%; `requiredLimits` from `@pmndrs/ddgi/core`), the same in the error message, TASKS (`ddgi`).
Verified (headless Chrome 156):
- RTX, Cornell page, per-pixel relative luminance difference against per fragment (mean / p95):
  - cornell-box-point 0.042% / 0.16%;
  - emissive Cornell 0.052% / 0.21%;
  - **grid 0.026% / 0.12%** (GI 18% of the light; the grid wins the left half);
  - **ao 0.041% / 0.15%**;
  - **environment 0.037% / 0.14%**.
- Intel UHD, the same cases: 0.033% / 0.083%, 0.045% / 0.14%, grid 0.019% / 0.058%, ao 0.032% / 0.078%, environment 0.030% / 0.078%.
- RTX: `three-unmarked-wgsl` passes (30,860 and 25,208 characters, the same lengths as the pre-T1.14 baseline).
- `test:gpu --filter three-` passes on both GPUs, 9 pages each (RTX 3060, Intel UHD gen-12lp), the new cases included. `fetch.test.ts` passes on its own.
- `pnpm test`: 916 of 917 passed. The failure is `tools/assets/fetch.test.ts`, already listed as flaky under load (TASKS, carried to M3), and run while the GPU suite was going.
Cost on the RTX, Sponza 2560×1440, GPU-bound (GPU use 99–100%), field frozen, median of per-cycle differences:
- **run 1** (10 s + 4 s, 2026-11-05 22:5x): **INVALID**, thermal throttling in 46% of loaded samples, SM 690–712 MHz.
  - composite (no GTAO) − off 1.84 ms; composite + GTAO (the default) − off 11.97; GTAO half 5.15; MRT 0.68; composite pass 1.33; GTAO full 10.2; GI per fragment on the same pass 6.51.
  - At 1920×1080: composite pass 0.85 ms, composite − off 1.32.
- **run 2** (6 s): **VALID** (63 C start, 0% thermal), SM 944–990 MHz.
  - composite − off 1.89; default path 11.47; GTAO half 5.90; composite pass 1.36.
  - Wide per-cycle spread (MAD about 3 ms): the clock rose from idle.
- **run 3** (6 s): **VALID** (68 C, 0% thermal), SM 600–810 MHz.
  - composite − off 1.72 (MAD 0.24); default 11.38; GTAO half 4.96; MRT 0.63; composite pass 1.28; GTAO full 9.72.
- **run 4** (6 s): **INVALID**, thermal throttling in 11.8%, SM 667–705 MHz.
  - composite − off 1.83; default 12.46; composite pass 1.37.
- **Spread:** composite − off is 1.72–1.89 ms across all four runs, within ±0.1 ms. The default path is 11.4–12.5 ms (±0.55).
- **Against the bar (≤ 1.0 ms):**
  - at these power-limited clocks (600–990 MHz) even the composite without GTAO is over the bar;
  - clock-normalised to 1,500 MHz it's 0.86–1.25 ms;
  - GTAO, which DECISIONS puts on by default, adds 9.7–10.8 ms at full resolution and 3.2–3.5 ms at half (about 5 and 1.7 ms normalised). The cheapest setting that keeps Sponza within ±10% (half, 16 samples) is still well over the bar.
Not verified: a VALID run at a clock near 1,500 MHz. The RTX is power-capped (60–76% of loaded samples) and throttles within about 10 s of GPU-bound load.
Next:
- **lead / Dennis:** GTAO on GI by default doesn't fit the 1.0 ms bar at any setting that keeps Sponza within ±10%. Choose between:
  - GTAO off by default (opt-in for cloth);
  - a cheaper occlusion term;
  - counting GTAO apart from the shading bar;
  - a quarter-resolution GTAO, whose accuracy is unmeasured.
- The viewer's warm-up switch-over to `volume.start` once `apps/viewer` is free.

## 2026-11-05 · T1.12 phase 2a follow-up: the warm-up's bounce light (the ramp) · implementer (opus)
Did (uncommitted, on top of 14c2ed2): the lead asked to fix Cornell coming out of the warm-up 25% low and needing 17.5 s to get within 1%, in the technique.
- Cause: the warm-up was a plain running average whose estimates read the field it was building. The bias shrinks like k^−(1−ρ); Cornell fits ρ ≈ 0.65 (the model matches a = 1–4 within 1–3 points).
- The ramp: for a probe's first K = 16 blended updates its count is capped at k/a with a = 8, then it grows by one per update. Every-frame updates get h = k/(k+a) during the ramp, which weights later estimates more. Settings `hysteresisUpdatesPerSample` (8) and `hysteresisRampUpdates` (16); the history is now n, t and k (12 bytes a probe). Files: `wgsl/blend.ts` (`ddgiRampCap`, `DdgiProbeHistory`), `wgsl/common.ts`, `layout.ts`, `settings.ts`, `ddgi-technique.ts`, `ddgi.test.ts`; pages `ddgi-shared/hysteresis-modes.ts` (`tau:5/200/<a>/<K>`), `ddgi-flicker` (`VITE_DDGI_FLICKER_AFTER_WARMUP`: measure right after loading); design doc ("The ramp"), NOTES.
Numbers (RTX 3060, τ 5 s; Cornell 1,574 probes a frame, Sponza 291; the error right after the warm-up is one image, ±1.5 points run to run):
- Cornell after the warm-up: a = 1 −26%, 2 −15.6%, 3 −10.5%, 4 −9.2%, 6 −9.0%, 8 −4.6%; **a = 8, K = 16: −7.4% / −6.3%**. Within 1%: 19 s, 5.5, 2, 1.5, 1, 0.5; **1 s**. After a restart: 16.5 s → **2 s**.
- Cornell per-pixel change the 240 frames after the warm-up: a = 1 0.30% (the image climbs 21%), a = 3 0.63%, a = 8 1.52%, **a = 8 K = 16 0.39%**. 400 frames later: 0.092% (still climbing), 0.109%, 0.283%, **0.089%**. So an endless a = 8 is out: it reaches n_max only after 1,600 updates.
- Settled flicker with the defaults: Cornell 0.094% / 0.38% (`gpuMs:2`), Sponza 0.025% / 0.091% (unchanged).
- Sponza with the defaults: +0.05% after the warm-up, within 1% 3.5 s after a restart (3 s before), sun halved within 5% / 1% in 12.5 / 20.5 s (13 / 22.5 s). Per-pixel error 5 s after a restart 5.0% against 3.6%: a little noisier for its first 16 updates.
- A fixed small cap (the lead's other idea) loses by the same model: a cap of 8 during the warm-up leaves Cornell about 23% low, a cap of 2 about 13%. Not measured.
- With a = 3: furnace at 600 frames −0.19% (settled), two-room leak at 600 0.74% (settled 0.81%), so the page counts (1800 / 3000) stay.
Verified: lint, typecheck; DDGI unit tests 43; the full `pnpm test` gave 912/917 while the Intel suite and a Cycles CPU render ran (timeouts in bench-lock, sky and trace-bench tests). Those three files alone: 88/88. `test:gpu --filter ddgi-` passes on the RTX (furnace −0.15%, leak 0.83%, test view +5.63%) and the Intel UHD (−0.28%, 0.83%, +5.57%). The test view went from +4.52% to +5.63%: its 600 frames now settle further (the T1.5 number was +5.7%).
Not verified: Intel settle and flicker with the ramp; the viewer by eye (phase 2b).
Next: the lead: the ramp's two thresholds for DECISIONS (in my report).

## 2026-11-05 · T1.13 HD references with stored cameras · implementer (opus)
Did:
- **References:** `tools/reference/cycles/render.py`, `blender_scene.py`, `run.ts`.
  - The defaults are now 1920×1080, Blackman-Harris 1.5 px and OptiX.
  - Every json stores a `shot`: camera, aspect, size, near and far, exposure and tone mapping, the manifest, sky and table hashes, commit, device, samples, seed and filter.
  - `--from-shot` re-renders a stored view.
  - `--denoise` writes `cycles-full-denoised.exr` for viewing; its bias against the raw image is stored.
  - New `combine.ts` averages seed parts into one reference; new `shot.ts` reads shots for the pages.
  - CPU renders take the CPU lock, GPU renders the GPU lock.
  - Pre-T1.13 files are kept as `cycles*-lowres.*`. README: "HD references" and "Stored camera".
- **Viewer** (`reference-files.ts`, `reference-technique.ts`, `viewer-app.ts`, `vite.config.ts`):
  - Compare moves the camera to the shot, letterboxes the picture to its aspect, and applies its exposure and tone mapping.
  - The panel warns when the manifest has changed, and says when the render is from an older commit.
  - GI-only shows `cycles-gi.exr`; the full view prefers the denoised copy.
- **Test pages:** `ddgi-test-view` and `ddgi-sponza-accuracy` render at the reference's stored camera and size. The test-view gate now runs at HD.
Verified (RTX 3060, headless Chrome 156):
- `pnpm test:gpu --filter ddgi-test-view` passes.
- Compare on point-light Cornell and Sponza: 16:9 letterbox; the green wall's edge sits in the same place in the all-DDGI and all-Cycles views to within 1 px; Sponza's exposure returned to +2. This alignment was checked by hand with a scratch script on the RTX only; no automated test guards it.
- Lint and tests pass; typecheck is clean except Budget v2's `work-budget.ts`.
Not verified: Intel.
Committed as 689b5f6, which also carries the curtain investigation's test pages, because they share `ddgi-shared/harness.ts`.
After the review of 689b5f6 (uncommitted):
- `noise.ts` stores each reference's measured noise; the viewer states it and the sample count next to the picture, and the README has the table. GI per pixel is over 3% on point-light Cornell views 2 and 3 (9–10%), the dark room (6.2%), the doorway (3.5%) and textured Sponza at 512 (14%); every reference is under 2% over 8×8 blocks. No re-renders.
- `combine.ts` also refuses a different commit or bounce count; unit tests cover every refusal.
- `--device=cpu` takes the CPU lock (`device.ts`, tested).
- `uncommittedChanges` counts only tools/reference, scenes and tools/sky.
Renders (seconds, and the GPU lock wait before each):
- **OptiX:** point-light Cornell front, 8192 samples × 2 seeds: 983 s (wait 383) and 975 s (wait 123). Two-room lit-room, 4096: 1351 s (wait 597). Dark room, 2 × 2048: 564 s (wait 935) and 432 s (wait 11). Untextured Sponza (GI table sha256 7680a2d0…, unchanged), 4 × 2048: 445, 470, 379 and 381 s (waits 1, 1, 92 and 99).
- **CPU, 12 threads (no GPU lock):**
  - Point-light Cornell cameras 2 and 3: 1559 and 1745 s.
  - Emissive Cornell cameras 2 and 3: 1770 and 1802 s.
  - Open ground: 1019 and 756 s. Through-the-doorway: 2475 s.
  - Textured Sponza at 512 samples: 2029 s. Emissive Cornell front, 2 × 1024: 1319 and 1303 s. Sponza upper gallery at 512: 2831 s.
  - Point-light Cornell camera 3 and emissive Cornell camera 2 include sleeps of 161 s and 59 s (`timingNote` in their jsons).
Numbers:
- **8×8-block noise:**
  - Point-light Cornell 8192: the two seeds differ by 0.53% (GI) and 0.24% (full).
  - Dark-room halves differ by 1.56%; emissive Cornell halves by 0.32% (GI) and 0.15% (full).
- **Denoised copy against raw:** bias −0.01% to −0.18%.
- **HD against old** (bias, block error, FLIP 8×8):
  - Cornell +5.74%, 6.46%, 0.104 against +5.68%, 6.38%, 0.179.
  - Two-room +5.50%, 5.91%, 0.069 against +5.63%, 5.98%, 0.073.
  - Sponza +23.07%, 27.66%, 0.324 against +23.03%, 26.71%, 0.326; today's code at 640×360 against the old reference reads +23.18%. Reference noise: two 2048 parts differ by 0.60% (8×8 blocks).
  - Today's code at 512² reproduces the old Cornell numbers exactly, so resolution moves nothing.
  - Emissive Cornell front (raw 2048; reference noise 0.16% GI, 0.07% full): DDGI against the GI composite +5.72%, block error 8.35%; full image +2.06%, block error 4.63% (not gating).
  - Cornell's p95 and FLIP fall only because 16:9 adds black pixels.
Next:
- Sponza's HD reference for T1.14's 1440p accuracy: `bench/references/sponza/colonnade-from-the-floor/cycles-untextured-gi.exr` (with `cycles-untextured.json`).
- `ddgi-leak`'s constants stay valid: the two-room HD means match the old ones within 0.12%.

## 2026-11-05 · T1.12 phase 2a: hysteresis that knows the time between a probe's updates · implementer (opus)
Did (uncommitted; the lead's design, docs/design/ddgi.md "Hysteresis that knows the time between updates"):
- Core: per probe, `n` (effective samples) and `t` (last update, s) in a new storage buffer that only `ddgi/blend` reads and writes (binding 8; 5 storage buffers). On an update: `n ← n·exp(−(T−t)/τ)`, `h = n/(n+1)`, then `n ← min(n+1, nMax)`. State 3 gives n = 0 (h = 0). The same h for distances; the big-change rule and the firefly clamp are unchanged.
- The clock: the sum of `deltaTime`, at most 0.25 s a frame (`MAX_CLOCK_STEP_SECONDS`: a hidden tab mustn't wipe every probe's history), reset at `prepare`.
- Settings: `adaptiveHysteresis` (true), `hysteresisMemorySeconds` τ (5, `ADAPTIVE_MEMORY_SECONDS`), `hysteresisMaxSamples` nMax (200, `ADAPTIVE_MAX_SAMPLES`); `hysteresis` (0.97) is now the fixed rule's. The viewer's and ddgi-three's warm-ups still write `hysteresis`; under the adaptive rule that is unused and harmless, because they run with `deltaTime` 0, which is the same running average.
- Files: `packages/ddgi-core/src/techniques/ddgi/` `settings.ts`, `layout.ts`, `ddgi-technique.ts`, `wgsl/common.ts`, `wgsl/blend.ts`, `ddgi.test.ts`. Pages: new `ddgi-settle` (manual); `ddgi-flicker` (`VITE_DDGI_FLICKER_HYSTERESIS`; it converges with the fixed 0.97 first); `ddgi-shared/hysteresis-modes.ts` (new), `viewer-scene.ts` (`frame(budget, deltaTime)`; the warm-up passes 0 like the viewer); `ddgi-furnace` and `ddgi-leak` (`VITE_DDGI_HYSTERESIS` and `VITE_DDGI_FRAMES`; **default frames 600 → 1800 and 3000**, below). Comment in `examples/webgpu/hello`. Design doc, NOTES (3 bullets), `bench/results/flicker.csv` (`note/t1_12_hysteresis_rules`).
Numbers (RTX 3060 Laptop, Chrome 156; DDGI's clock at 60 Hz; fixed 0.97 / τ 2 / **τ 5** / τ 10):
- Cornell flicker (`gpuMs:2`, every probe each frame), per-pixel change mean / p95: 0.56 / 2.30% · 0.155 / 0.64% · **0.095 / 0.39%** · 0.093 / 0.38%. Theory: 6.0× quieter at h 0.995, measured 5.9×. nMax 100: 0.185 / 0.76%.
- Sponza flicker (291 probes a frame): 0.0038 / 0.014% · 0.058 / 0.22% · **0.027 / 0.10%** · 0.0145 / 0.054%. Intel Cornell (90 a frame): 0.046% · – · **0.090%** · 0.048%.
- Sponza settle (`ddgi-settle`; batch 291 calibrated from a 62 C start, 1.1 s a sweep; the run's verdict was INVALID because it ran hot later, but only the batch is timing). After the warm-up the image mean is already within 0.6% for every rule (0 frames to 5% and 1%). Restart with no warm-up, to within 5% / 1%: 60 / 2,640 frames (1 / 44 s) · 30 / 720 (0.5 / 12 s) · **60 / 180 (1 / 3 s)** · 30 / 210 (0.5 / 3.5 s). Sun halved, no reset: 5,460 / 9,750 frames (91 / 162.5 s) · 330 frames (5.5 s) / noisy · **780 / 1,350 (13 / 22.5 s)** · 1,530 / 2,640 (25.5 / 44 s).
- Cornell settle (1,574 a frame): after the warm-up the image is 25% low. To within 5% / 1%: 2 / 4.5 s (first time; the 0.97 noise leaves the band until 49 s) · 3.5 / 10 s · **4.5 / 17.5 s** · 6 / 18.5 s.
- Accuracy, RTX / Intel: test view Cornell point against the HD Cycles reference +4.52% / +4.47% (600 frames, pass). Furnace at 600 frames: adaptive −1.95% / **−2.05% (fails 2%)**, fixed −0.16% / −0.28%; at 1800 frames −0.24% / −0.36%. Two-room leak at 600 frames: adaptive −0.22% / −0.21% (lit room 5.4% below settled), fixed 0.80% / 0.80%; at 1800 0.67% / 0.67%; at 3000 0.81% (RTX). So the settled accuracy is unchanged, and every-frame probes take 3–5× the frames to settle bounce light.
Verified: lint, typecheck, test 913. GPU suite: `test:gpu --filter ddgi-` passes on the RTX 3060 and the Intel UHD (8 pages each: furnace 1800 frames −0.24% / −0.36%, two-room leak 3000 frames 0.81% / 0.81%, test view +4.52% / +4.47%; the Intel leak page takes 183 s, so its timeout went 300 → 600 s). The manual pages `ddgi-flicker` and `ddgi-settle` ran on the RTX, and `ddgi-flicker` on the Intel, all passing.
Not verified: the viewer by eye (phase 2b); Sponza settle on the Intel; settle at frame rates other than 60 Hz (τ is in seconds, so adaptive settle times should hold; the fixed rule's scale with the frame rate); curtained Sponza.
Next:
- Lead: the proposed DECISIONS entry (τ 5 s, nMax 200; the accuracy pages' new frame counts) is in my report.
- Lead or T1.13: `ddgi-test-view` still converges 600 frames (I didn't touch it; T1.13 has it open). For a settled number under the adaptive rule it needs about 3000.
- Phase 2b: the viewer's warm-up `hysteresis` override can go; debug views; Dennis judges by eye.
- Possible later: Cornell's warm-up leaves it 25% low and the adaptive rule takes 17 s to settle it within 1%. Weighting later updates more during the running average (bounce estimates improve as the field fills) would shorten that. Not built: outside the lead's design.

## 2026-11-05 · T1.14 part 2: the confirmed names, volume.start, the render-time update, examples · implementer (opus)
Did (uncommitted):
- `packages/ddgi-three/src/index.ts` exports `ddgi` (renamed from `ddgiComposite`; it takes the volume), `giAlbedo`, the `GIVolume` additions, `ThreeGILimitError`, `DEFAULT_WARM_UP` and their option types.
- `volume-start.ts` (new) is what `GIVolume.start(renderer, scene, options?)` runs:
  - three's device, created if needed (`renderer.backend.device`, `renderer.init()`);
  - `createGI` with GPU timing, and DDGI from `volume.ddgiVolume()`, prepared;
  - `ThreeGI`;
  - the warm-up, copied from the viewer: paused until the trees exist, then 16 running-average passes (8 s cap, at least 2), then budget calibration;
  - then it switches on the render-time update.
  - A core `GILimitError` comes back as `ThreeGILimitError` (still a `GILimitError`), with the fix for three: `createGIRenderer()`, or `requiredLimits(adapter)` on the renderer.
- `gi-volume.ts`: `start`, `isStarted`, `threeGI`, `platform`, `light` (throws with "await volume.start" before), `recalibrate()` (`platform.recalibrate()`) and `dispose()`.
- `gi-light.ts`: the render-time update. The node's `update` first runs `light.onRender(camera)`, at most once per JavaScript task (a microtask clears the mark, so it doesn't need three's frame counter), then refreshes the module's values. An error is reported once and stops it. The file comment lists the caveats: the first lit render's camera, and no GI frame in a frame with nothing lit.
- `three-gi.ts`: `updateOnRender` (option and settable property), carried across light swaps.
- `packages/ddgi/vite.config.ts`: `three/addons/*` is external too (the composite imports three's GTAO).
- `examples/three/`: `simple.html` and `composite.html`, on three's `webgpu_lightprobes.html` scene (`src/cornell.ts`, copied), plus `index.html` and `vite.config.ts`. Run with `pnpm exec vite --config examples/three/vite.config.ts`. `tsconfig.json` includes `examples/three/src`.
- `tools/gpu/server.ts` aliases `@pmndrs/ddgi(/core|/three)` to source, as the examples' config does. New page `three-examples` runs both examples' own `main()`.
- `docs/api/three.md`: getting started, `GIVolume.start`, the render-time update, `ddgi()` and `giAlbedo`.
- Tests: `render-time-update.test.ts` (10 tests: the once-per-task guard, errors, `updateOnRender`, the volume before start, the composite's occlusion defaults).
Verified (headless Chrome 156):
- lint, typecheck, `pnpm build`, `pnpm test` (911 passed);
- RTX: the unmarked WGSL is still byte-identical to the part-1 HEAD baseline for Standard and Lambert, so the simple path's shader is unchanged;
- RTX, `three-examples`: each example ran 40 GI frames in 40 animation frames with no per-frame call. GI is 45% of the canvas brightness in simple and 44% in composite. Both use `antialias: true`, so the composite works with an MSAA pass.
- `test:gpu --filter three-` passes on both GPUs, 8 pages each (RTX 3060, Intel UHD gen-12lp): the T1.7 neutrality and environment pages, the composite page and the examples. On the Intel the examples give GI 44.5% (simple) and 43.3% (composite), 40 GI frames in 40 animation frames.
Usage bar against three's `webgpu_lightprobes.html` (GI lines):
- LightProbeGrid: 5 (import, `new`, `position.set`, `scene.add`, `bake`).
- Ours, simple: 5 (import, `new GIVolume`, `position.set`, `scene.add`, `await volume.start`). Nothing per frame; the probes update live, where LightProbeGrid needs a re-bake when the light changes.
- Ours, composite: 10, the 5 plus a `three/tsl` import, `new RenderPipeline`, `pass`, `setMRT(mrt({ output, giAlbedo, normal: packNormalToRGB(normalView) }))` and `outputNode = ddgi(scenePass, volume)`. `pipeline.render()` replaces `renderer.render`.
- Names a user touches: simple, only `GIVolume` (constructor, `position`, `start`). Composite adds `ddgi` and `giAlbedo`, plus three's own `RenderPipeline`, `pass`, `mrt`, `output`, `normalView` and `packNormalToRGB`.
Still harder than LightProbeGrid:
- **Large scenes need raised device limits** (`createGIRenderer()` or `requiredLimits(adapter)`). `start` throws `GILimitError` with the fix.
- **Composite:**
  - transparent objects in front of composited surfaces get the GI behind them evaluated at the transparent surface (−19% behind a 50% sphere in Cornell), so draw them after the composite;
  - the MRT output must be named `normal` and packed with `packNormalToRGB`;
  - `start` must finish before the pipeline's first render;
  - a shading module that changes shape needs `pipeline.needsUpdate = true`.
- **The render-time update:** it uses the first lit render's camera, and runs no GI frame in a frame with nothing lit.
- **Start-up time:** `start` takes seconds (up to 8 s of warm-up plus calibration). `warmUp: false` makes it fast, and GI then fills in live.
- **GTAO** (on by default in the composite): its cost is not yet timed under the clock rule.
Scout's list of differences from three's examples:
- LightProbeGrid's shape and its self-updating node: now matched (`start`, the render-time update).
- SSGI's additive composite: matched. We write `giAlbedo` instead of `diffuseColor`, because `diffuseColor` is about 4% brighter at normal incidence and would fail the match.
- GTAO/VXGI's `builtinGIContext`: not used. It needs GI before the pass (a prepass, +6.5 ms; DECISIONS).
- The display nodes' own render target: the composite needs none (it's a pure function of the pass textures); its GTAO has one, as three's does.
- Double counting: handled by the pass mark and the environment-diffuse workaround.
- Default limits: `GILimitError` with the fix.
Not verified: the cost (timing waits for the lead's go-ahead); a scene with `scene.environment` in the composite path; the examples by hand in a visible browser (only headless).
Next:
- the viewer still has its own warm-up (`apps/viewer/src/techniques/ddgi-technique.ts`). Switch it over to `GIVolume.start` (or a shared warm-up) once other agents are out of `apps/viewer`;
- the viewer's "GI composite" toggle;
- the RTX timing of `three-composite-sponza`, 3 runs under the clock rule.

## 2026-11-05 · T1.14 part 1b: GTAO on the GI term, and drift-proof cost timing · implementer (opus)
Did (uncommitted):
- `packages/ddgi-three/src/composite.ts`: option `occlusion` on the composite.
  - Default: three's own GTAO (`ao()`) on the pass's depth and packed normal, radius 1 m (`DEFAULT_GI_OCCLUSION_RADIUS`), multiplying the GI term only.
  - `{ radius, resolutionScale, samples }` sets it; `false` turns it off; `{ source }` takes a GTAO node the pipeline already has.
  - A pass whose context already applies AO (`builtinAOContext`) defaults to none, since `giAlbedo` already carries that AO.
  - `occlusionNode` exposes the composite's own GTAO node for live changes.
- `tools/gpu/pages/three-composite-shared/composite-check.ts`: `timeInterleaved` replaces `timeVariants`.
  - Variants are drawn frame by frame (A B C D A B C D …), and every pass of every frame is GPU-timestamped and tagged with its variant and cycle.
  - Each difference is the median of per-cycle differences between neighbouring frames, with p10–p90, MAD and the medians of each third (a drift check).
  - `CompositeRoutes` takes `occlusion`; it defaults to off, because the match checks compare with GI per fragment.
- `three-composite-sponza/main.ts` (rewritten):
  - variants off, mrt, composite, composite+gtao, composite+gtao-half and forward;
  - a 10 s section at 2560×1440, then 4 s at 1920×1080;
  - a 25 s idle after loading, then a 2 s ramp;
  - the page no longer judges the bar.
- `three-composite-sponza/run.ts` (new):
  - samples nvidia-smi every 200 ms with wall-clock stamps and runs the page;
  - for each section, gives the SM clock median, the clock rule's verdict (`judgeTiming`, with the temperature at the section's start) and every delta clock-normalised (ms × MHz ÷ 1,500).
- The curtain agent's `three-composite-ao` (uncommitted): its timing moved to `timeInterleaved`, and its match takes `--resolution-scale`.
Verified:
- `three-composite` passes on both GPUs after the change. The Cornell match, transparent check and render-time update results are unchanged.
- `run.ts` runs the cost page end to end on the Intel UHD (correctness only; no nvidia-smi there, so no verdict).
- Typecheck and lint are clean.
GTAO accuracy against untextured Cycles (RTX; `three-composite-ao/run.ts`; Sponza at the 640×360 reference, Cornell at 600 passes). Bias at radius 1 m:
- full resolution, 16 samples: Sponza +4.27% (curtains +10.5%), base package −2.68%, Cornell +0.58%;
- **half resolution, 16 samples: Sponza +7.99% (curtains +18.0%), base −1.47%, Cornell +0.73%**;
- 8 samples: Sponza +15.8% (full) / +17.0% (half), outside ±10%;
- 4 samples: Sponza +20.4% (full) / +20.7% (half), almost no effect.
- So the sample count matters far more than the resolution, and the cheapest setting within ±10% is half resolution at 16 samples.
- Caveat: at 640×360, half resolution is only 320×180, much coarser than half of 1440p. Its accuracy at 1440p is unmeasured until a 1440p reference exists.
Not verified: the cost on the RTX. I'm waiting for the lead's go-ahead on the fan-control reinstall. The clock rule's bench-lock update has landed (8939225). Command, 3 runs: `node tools/reference/bench-lock.ts run --what "three composite sponza" -- node tools/gpu/pages/three-composite-sponza/run.ts --gpu nvidia`. So whether the composite − off spread is now within ±0.2 ms is unknown.
Next: if composite + GTAO at full resolution is over 1.0 ms, propose half resolution at 16 samples as the default.

## 2026-11-05 · Bench lock: the clock rule for valid timing · implementer (opus)
Did (uncommitted): `tools/gpu/clock-note.ts`: `judgeTiming` (DECISIONS cca8ab9). A hot start (70 C or more) is INVALID. Otherwise: under 5 loaded samples, or no start temperature, is NO VERDICT; 5% or more thermal samples is INVALID; else VALID. There is no MHz bar. Rows: `note/timing_valid,1|0`, `note/timing_no_verdict,1`, `note/thermal_share`, `note/start_temp_c`; `note/clock_low` and `note/start_hot` are gone (start_hot was never written to a CSV). `tools/reference/bench-lock.ts`: sampling every 200 ms on a fixed beat (a slow nvidia-smi call skips beats, never queues); the summary records `sampleIntervalMs`; the end-of-run line is the verdict. Docs: tools/reference/README "Bench lock", bench/results/README (old and new notes, with the date), my NOTES bullet.
Verified: `npx vitest run tools/reference tools/gpu/clock` 48 passed (the three verdicts, the 5% and 5-sample edges, 200 ms rate, a slow reader). `pnpm lint` clean. Typecheck clean apart from other lanes' uncommitted `work-budget.ts` and `three-composite-*`. Real nvidia-smi on temp lock folders: 11 samples in about 2.2 s.
Re-judged the cool-start set (scratchpad clock JSONs, 1 Hz; CSVs untouched): Budget v2 b-1/2/3: Bistro valid ×3, Sponza valid in run 2 and no verdict in runs 1 and 3, Cornell and two-room no verdict ×3 (1–4 loaded samples). Re-runs: Bistro rr-1 and rr-3 valid, Sponza rr-3 valid, Cornell rr-3 no verdict. Composite c-1/2/3 invalid (thermal 22%, 17%, 14%); re-runs cr-1 and cr-3 valid (0%), cr-2 invalid (14%). Saturation: Cornell invalid (42%), Sponza invalid (31%).
Next: `tools/native/bench.ps1` builds its own clock rows and writes no verdict. The NOTES bullet on sustained load still names `note/clock_low`. The DECISIONS pass/fail entry still says ≥ 1,500 MHz (superseded by cca8ab9).

**Lead mistake (2026-11-05):** a lead script opened PROGRESS.md for writing before reading it in the same expression, which emptied the working-tree file and lost the curtain agent's uncommitted entry; the lead restored the file from 08bf097 and the agent rewrote its entry. Nothing committed was lost.

## 2026-11-05 · Curtained Sponza +23%: cause and measured fixes · implementer (opus)
Did (test tools only; no production change): `ddgi-sponza-accuracy` (census of probes near the curtains, a per-material split in run.ts, the experiment options `partsOutOfGI`, `singleSidedFrontSide` and `diagnosticLookup`/`diagnosticRays`, and `--timeout-minutes`); `ddgi-shared/harness.ts` (albedo and position images, and the diagnostic lookups `tracedVisibility` and `shortRangeOcclusion` in the test view); a new page `three-composite-ao` with its own run.ts (three's GTAO on the composite's GI term, and timing).
Cause (RTX and Intel agree within 0.1 point): the light changes inside the cloth folds faster than probes 1 m apart can follow. The data is fine: the curtains are single sheets, winding matches the normals on over 99% of their area, and the path tracer matches Cycles on curtain pixels (+0.3%). Curtain pixels read +43%, other pixels +14.5%. The ruled-out causes:
- treating cloth back faces as surfaces: +22.2%;
- exact visibility rays instead of Chebyshev: +23.7%;
- exact irradiance at DDGI's lookup point: curtain pixels already read +19% there.
Fixes against untextured Cycles (both GPUs):
- **Ray-traced short-range term** (rays out to 0.5 m; each hit takes away 1 − its albedo; GI only):
  - Sponza: 4 rays +1.6% (curtain pixels −2.9%), 8 rays +1.4%. Base package −1.3 / −1.4%.
  - Half resolution moves the mean by under 0.1 point. Per-pixel noise of the factor against 128 rays: 20% (4 rays), 12% (8).
  - **Cost on the RTX at 2560×1440: 4 rays 123 ms full / 34 ms half; 8 rays 263 / 72 ms** (tracer about 120 Mrays/s): not real-time as built.
- **three's GTAO on the composite's GI term** (16 samples; Sponza at the 640×360 shot, Cornell at 1920×1080):
  - Sponza by radius: 0.1 m +15.1%, 0.25 m +10.9%, 0.5 m +7.5%, 1 m +4.3%. Curtain pixels stay at +10.5 to +11%.
  - Base package −2.1 to −2.7%. Cornell point +5.7% without AO, +2.8% at 0.5 m, +0.6% at 1 m: never darker than Cycles.
  - **Cost on the RTX at 2560×1440: 5.6 ms full, 1.4 ms half resolution.**
- A finer 0.25 m volume over the curtains (composited offline): +5.5%.
Clock: every timing above is flagged low (SM median 731 MHz for GTAO, 1117 MHz for the rays; power limit 86–93%).
Cornell discrepancy (+0.5% on the three path against +5.7% in the test view): an unconverged field on my page (16 + 64 passes), not a bug. At 600 passes the two paths agree to 0.01% (0.049928 against 0.049934): +5.73% (RTX), +5.67% (Intel). The light's 20 m range window moves it only 0.15 point; the page now drops it (`ignoreLightRange`) and takes `--passes`. Sponza on the short schedule matched the settled test view.
Verified: lint and typecheck clean; `pnpm test:gpu --filter ddgi-,three-neutrality` passes on the RTX and Intel (two-room leak 0.80%, thin wall 0.59%, closed door 0.06%). No run overlapped the 20:05 or 20:14 sleeps.
Not verified: GTAO accuracy at 1440p (the Sponza reference is 640×360); any-hit or accumulated ray variants; a real two-volume implementation.
Next: Dennis picks the fix: GTAO 0.5–1 m on GI (cheap, curtains about +11%), a finer nested volume (M3 work), or a cheaper ray term (accumulated or baked).

## 2026-11-05 · Cool-start timing set: Budget v2, T1.14 composite, saturation · gpu-tester (sonnet), lead checked
Did: RTX 3060 Laptop, headless Chrome 156, every run under `bench-lock run` from 61–65 C, one scene per page run. No code changed. Runs checked against the two sleep windows (11:05–11:08 and 11:14–11:15 UTC): none overlapped. HEAD moved during the set (e068896 → 64c9404); the calibration ladder didn't change, the saturation code did. Rows: `timing.csv` (`note/budget_v2_fixes,1`, `note/cool_start,1`), new `three-composite.csv` and `saturation.csv` (converted to the 7-column format by the lead).
- **Budget v2, 2 ms, 3 runs (median):** Cornell 1,569 probes at 1.80 ms; two-room 710 at 1.90; Sponza 245 at 1.93 (228–291 across runs and re-runs); Bistro exterior 140 at 2.06 (132–183). The page's ±10% check passed in 14 of 16 runs: Cornell run 1 read 13% under (1,474 probes, 1.74 ms), Bistro re-run 3 read 11% over (183 probes, 2.22 ms, still `clock_low`). Clocks 1,234–1,725 MHz, power-limited, never thermal; 1–6 loaded clock samples a run.
- **T1.14 composite cost, Sponza 2560×1440, 6 runs:** composite − off 0.46–3.47 ms (median 1.46, re-runs 1.91); composite pass alone 0.66–1.16 ms; off alone 3.6–7.2 ms. Every run `clock_low` (855–1,215 MHz). **Not judgeable:** the deltas are smaller than the run-to-run swing of the frame itself.
- **Saturation check (new page section), one run each: fail on both scenes.** Cornell: stepped down at 6.75 s (allowed 4 s), lowest 1,004 (floor 8), not back to 0.85× in 40 s (1,309). Sponza: stepped down at 6.63 s, lowest 140 (floor 81), not recovered (183, needs 233). The synthetic load made frames only 17.9–18.2 ms against 16.67 under load, under the 10% "late" line (18.33), so the load is too light on the RTX. Clocks 1,567 / 1,620 MHz.
Not verified: the composite bar (needs a stable measurement, see Next); the saturation step-down with a load that really makes frames late on the RTX.
Next: size the saturation load from measured frame time; measure the composite as A/B deltas under a steadier clock (the clock-rule question with Dennis).

## 2026-11-05 · B3 CWBVH spike · side session (opus), Opus review
Did (experiment code on branch `bvh-b3`, head bb86f7c, code at 9c9a997; only these findings are merged into m1): `tbvh_write_cwbvh` in `packages/bvh/native/tinybvh-wasm.cpp` runs tinybvh's `BVH8_CWBVH` (80-byte nodes, our 48-byte triangles) and re-quantizes every child box from the triangles with a 1/1024-step margin; `buildCwbvh` and a checker in `packages/bvh/src/cwbvh-layout.ts`. An 8-wide WGSL walk (closest and any hit, watertight test, the tracer's ray and hit records), ported from tinybvh's `traverse_cwbvh.cl`, is experiment code in `tools/gpu/pages/cwbvh-shared/`; tracer `ours-cwbvh` on the trace bench (pages `trace-bench-cwbvh-sponza`, `-sponza-cwfirst`, `-bistro-interior`, `-quick`) and oracle variant `watertight-cwbvh`. No ddgi-core change.
Verified: `pnpm bvh:wasm --check` matches (154131 bytes, e6f071ac…); unit tests, typecheck, lint pass. Oracle at 9c9a997 passes on the RTX 3060 and the Intel UHD: CWBVH against the CPU reference 0 escapes and 0 mismatches over 2.09M rays; against the BVH2 walk 0 closest and 0 any-hit differences (185k / 292k ties within 1e-5, largest gap 2.0e-7). Bench agreement 100% on both scenes. Opus review: code sound; it asked for clocks-up rows and for the timing order to alternate (done: `?cwFirst=1`).
Numbers (RTX 3060, 9c9a997, median of 3, each set started at 62 C with m1's cool-down wait, CWBVH ÷ BVH2 on the same tinybvh trees and rays, closest / any):
- Sponza, BVH2 timed first: 108.7 / 113.6 = **0.957×**, any 0.974× (SM 1395 MHz). CWBVH timed first: 108.7 / 107.0 = **1.016×**, any 1.043× (SM 1264 MHz). The order is worth about 6%; across both, about 1.0×.
- Bistro interior: **0.821×**, any 0.807× (SM 1485 MHz).
- Every set is `clock_low`: from 62 C the card is held at 1264–1485 MHz by the power limit (38–67% of loaded samples), not by heat. I can't change power settings, so no set meets "clocks ≥ 1,500 MHz". Earlier sets (0ad7725, 646de83 at 1425 MHz; 2c9feb8 at 1530 MHz before the rule, rows not kept) gave Sponza 0.94–0.97× and Bistro 0.73–0.81×.
- CWBVH's speed doesn't follow the SM clock: Sponza 107–109 and Bistro 54 Mrays/s in every set from 1264 to 1530 MHz, while BVH2 moved 107–115 and 65–74. Higher clocks help BVH2, not CWBVH.
- Memory: Sponza nodes 42.5 MB against 194.1 MB (4.6× smaller), whole scene 318 MB against 470 MB; Bistro interior 7.9 MB against 30.9 MB, 58 MB against 81 MB. Page build 8.4 s against 3.8 s (Sponza), 1.6 s against 0.8 s.
- Exploratory single runs on Sponza (closest / any): tinybvh's nested walk 0.69× / 0.65×; float decode by bits 0.93× / 1.04×; one triangle per step 0.87× / 0.92×; nearer-first any-hit 0.94× / 0.97×. Kept: one loop over both levels, decode by conversion, a leaf's triangles as a group, farther-first any-hit.
**Verdict: dropped (Dennis, 2026-11-05: "drop it, merge the findings").** Closest-hit is about 1.0× BVH2 on Sponza against the 1.3× bar, and 0.82× on Bistro interior. Strictly, `clock_low` rows can't pass or fail; but CWBVH's speed is flat across clocks and BVH2's rises, so full clocks would move the ratio down, not up. Not profiled (WebGPU gives per-pass times only); likely reasons, from the variants: the walk is bound by dependent memory loads (flat under SM clock), not by box decoding (bits against conversion moved it 3–8%), and our BVH2 walk was already tuned (row vectors, 16-byte loads, farther-first any-hit). The paper's kernel relies on persistent threads with warp-wide work fetching, PTX byte permutes and a shared-memory stack, measured against a 2009-style baseline; WebGPU guarantees no forward progress for persistent threads. Bistro interior (1,186 small meshes) is worse again, probably because every mesh entry decodes a whole wide root; not checked.
Learned (NOTES, Tracing): quantized boxes need a margin (1.7% of edge-aimed rays slipped on the Intel UHD without one); keep a two-level walk in one loop; watertight tie gaps at shared edges; on a power-limited card the second of two tracers timed in one page runs about 6% slow.
Next: none. If memory ever matters more than speed (CWBVH is 32% smaller on Sponza), the code on `bvh-b3` is the starting point. The `trace.csv` rows with `ours-cwbvh/*` metrics are this experiment (commits 0ad7725, 646de83, 9c9a997 on `bvh-b3`; bench/results/README.md).
## 2026-11-05 · B2 · lead side session (opus); Opus review: pass with notes, fixed (the build bar waits on Dennis)
Did (branch `bvh-b2`, from `m1` at 0d9dcdb, `m1` merged in twice since; nothing in ddgi-core, ddgi-three or the viewer):
- `packages/bvh/src/gpu-hploc.ts` + `gpu-hploc-wgsl.ts`: GPU H-PLOC (Benthin et al. 2024), tinybvh's `BuildHPLOC` (MIT) ported, behind B1's contract and in its exact output layout. Reuses B1's batches, boxes, Morton codes, segmented sort, scans and readback; climbs the same radix tree. A node of more than 16 triangles (and the root) holds its children's clusters (at most 32) and merges mutual nearest neighbours (radius 8) down to 16 (the root: 1). The whole subgroup merges one node at a time, cluster c in lane c % size, slot c / size, with `ceil(32 / subgroupMinSize)` slots per lane (RTX 1, Intel 4) and shuffles by cluster number, so the tree is the same on any subgroup size (the RTX and the Intel UHD build identical trees). Handed-over words are atomics written once (stored cluster lists carry their writer's radix-tree level, since positions are reused); nodes are numbered without a counter (a merge takes the absorbed cluster's representative leaf); SAH leaf collapse decided as nodes are made; two scans number kept nodes and place their leaves' triangles.
- Options: `mortonCodes` (`'cubic-30'`, B1's, default; `'tinybvh-63'`, two sort words) and `tieBreak` (`'pairs'` default; `'tinybvh'`). tinybvh's own rule chains the tree on exact area ties (3000 identical triangles: depth 128, twice the tracer's stack), so ties pair up by default (NOTES).
- Oracle: tinybvh's `BuildHPLOC` exposed through the WASM wrapper (`builder: 'hploc'`; `pnpm bvh:wasm --check` reproduces the new .wasm), and `hploc-cpu.ts`, a CPU twin with the same options; with tinybvh's codes and ties it builds tinybvh's tree exactly (unit test). `tools/bvh/tree-quality.ts` compares builders on real scenes on the CPU.
- `createMeshLayoutBuilder` now picks H-PLOC by default (`prefer: 'gpu-lbvh'` for B1); pages `bvh-hploc`, `bvh-hploc-profile` (manual), `trace-bench-hploc-sponza` (manual), `trace-bench-hploc-build-sponza` (manual), `trace-bench-hploc-quick`, trace-oracle variant `watertight-hploc`; `ours-hploc` tracer (not in the default list). Trace bench: GPU builds are also timed back to back for about 10 s (`build_gpu_loaded_ms_*`, the pass/fail number; single builds let the GPU idle down: furnished room 24 ms single, 2.4 ms loaded). Small B1 changes: `lbvhBoxes` also reduces the mesh box, the radix sort allows odd pass counts, shared helpers exported, the bvh-lbvh page's meshes moved to `bvh-shared/`.
Verified (headless Chrome 156; RTX 3060 subgroups 32, Intel UHD 8-16): `pnpm lint`, `typecheck`, `test` (864), `build`; `layout-identity` (Sponza, both Bistros identical). On both GPUs: bvh-hploc (13 meshes x 3 option sets through the layout checker; default and tinybvh-codes trees match the CPU twin's node counts and its SAH cost to 4 decimals, except the 300k soup with tinybvh's codes at 0.997x (within the page's 1%; likely the GPU's f32 division at 21-bit cells); with tinybvh's codes and ties SAH within 1% of `BuildHPLOC`: 1.0000x on all but the 300k soup, 0.997x; 3000 identical triangles refused there, as tinybvh's own tree is 128 deep; a bad index refused, its neighbour kept), trace-oracle with `watertight-hploc` (0 mismatches, 0 of 2M shared-edge rays slipped), the fallback page, the quick benches (agreement 100%).
Numbers (RTX 3060 Laptop, `bench-lock run`, each run from 62-69 C; `bench/results/trace.csv` at 9221360, median of 3):
- Sponza (5,744,384 triangles, 132 meshes): SAH cost 1.032x tinybvh's default build (bar 1.15; LBVH 1.217x), 3,013,617 nodes, depth 37; with tinybvh's codes and ties 1.00001x the SAH cost of tinybvh's `BuildHPLOC` (bar ±1%); 0 meshes fell back.
- Build: 47.8 ms loaded (188 builds back to back over 9.6 s), 50.1 ms single. Building first (`trace-bench-hploc-build-sponza`, x3, not in the CSV so the two sets don't mix): 45.3 ms loaded (43.7, 48.0, 45.3), 47.7 single. Bar 50.
- Trace on the same rays: closest-hit 109.4 Mrays/s vs 115.1 on tinybvh's trees (0.950x), any-hit 202.4 vs 198.1 (1.022x); bar 0.9x. Agreement 100%.
Clock: every timing set is flagged `note/clock_low,1` (SM median 1,012-1,095 MHz under load, max 1,380-1,897; power limit in 63-67% of loaded samples, then heat). Even from a cool card this build never reached 1,500 MHz sustained, so by DECISIONS f3e3bba these rows can't pass or fail the 50 ms bar: **the build bar is not formally passed.** It is under 50 ms at the low clock (memory at its 7,151 MHz maximum), and the time should only fall as the SM clock rises, but the margin is thin (47.8 ms; loaded maxima 86-93 ms; loaded sections ran 9.1-9.8 s, a little under the rule's 10). Dennis's call: accept these rows, or re-time once the laptop holds 1,500 MHz under this load. The 0.95x trace ratio is from the same `clock_low` runs, but both tracers ran in each run, so the ratio is fairer than the milliseconds. Profile by step (single builds, slow clock, sort 8.7 ms): climb 26.6, sort 8.7, triangles 5.0, emit 2.6, boxes 1.4, clear 1.0, mark 0.9, rank 0.9.
Not done or not verified: no separate (non-hierarchical) PLOC builder, since H-PLOC's merge is PLOC and Dennis set H-PLOC as the target: a scope call for Dennis (the stage says "PLOC, then H-PLOC"). A subgroup narrower than the device reports is refused by the shader, but no test makes one. `createMeshLayoutBuilder` now defaults to H-PLOC (better trees, slower build than B1's); ddgi-core doesn't use it yet. Intel build speed (no bar), Bistro, and multi-batch builds at default limits are not timed. Not wired into ddgi-core (as B1). The M5 Pro is untested; if its browser doesn't report `subgroupMinSize`, the builder assumes 4-wide subgroups (8 slots per lane), which is correct but slower.
Review (Opus): pass with notes, no correctness bugs; re-ran lint, typecheck, build, test (893), bvh tests (40), and on both GPUs bvh-hploc, bvh-lbvh, the fallback page, trace-bench-hploc-quick and trace-oracle (all pass; trees identical on both GPUs). Notes and what was done: the build bar is not formally passed (wording fixed above, the call is Dennis's); the twin-match wording (fixed); a theoretical stale-write case in the cluster lists that would refuse a tree, not corrupt it (documented in the WGSL); a mesh of exactly 2^24 triangles would wrap the root's count (limit now 2^24 - 1); the PLOC scope change and the chooser's new default (raised here). An earlier trace-oracle run on both GPUs died with "DevTools connection closed" when the laptop slept; re-run after: pass on both.
Next: speed (the climb is most of the build: a two-round climb as in B1, fewer atomic record reads); a clean clocks-up Sponza number once the laptop holds 1,500 MHz; ddgi-core wiring with GPU-resident trees (M2/M3).

## 2026-11-05 · Bench lock: cool-down wait and a CPU lock · implementer (opus)
Did (uncommitted): `tools/reference/bench-lock.ts`: `run` waits after taking its lock until nvidia-smi reads the GPU below 70 C (polls every 5 s; `--cool-below`, `--cool-wait-max-minutes` default 15, then runs with a warning; `--no-cool-wait`; no nvidia-smi or `--lock cpu`: no wait). Second lock `$TEMP/ddgi-cpu-heavy.lock` (`--lock cpu`, `withBenchLock(work, { lock: 'cpu' })`); `run --cpu-timed` takes GPU then CPU; `status` shows both (exit code and top-level JSON stay the chosen lock's, so tools/native is unaffected; `locks` has both). `DDGI_BENCH_LOCK_HELD` may list two pids. `tools/gpu/clock-note.ts`: summary `start` (temperature, wait, capped), rows `note/start_temp_c` and `note/start_hot,1` (start at 70 C or more). Docs: header, tools/reference/README "Bench lock", NOTES bullet, bench/results/README.
Verified: `npx vitest run tools/reference tools/gpu/clock` 46 passed (11 new: wait, cap, `--cool-below`, `--no-cool-wait`, no nvidia-smi, CPU lock, `--cpu-timed` order and both held, status, two-pid parent). `pnpm lint` clean. Typecheck clean except `work-budget.ts` (T1.9b's uncommitted work). Smoke run against the real nvidia-smi on temp lock folders: read 62 C, `start` recorded.
Not verified: a real cool-down wait on a hot RTX (only with fake readings).
Next: `tools/reference/cycles/run.ts` (being edited by another lane: CPU renders take no lock) should pass `lock: 'cpu'` for `--device cpu`, and the README's "a separate CPU-heavy lock is planned" line should then point to it. `tools/native/bench.ps1` reads the clock file itself and doesn't write `note/start_temp_c`, and native timing doesn't go through `run`, so it doesn't wait for a cool GPU. The CPU-timed pages' docs (three adapter CPU cost, B0 worker builds) should say `run --cpu-timed`.

## 2026-11-05 · T1.9b Budget v2 review fixes · implementer (opus), lead checked
Did (working-tree files committed by the lead): `work-budget.ts` rewritten after the failed review; `gi-platform.ts`, the contract (`GITechnique.minUnitsPerFrame?`, `GIPlatform.recalibrate()`, `gpuMs` needs `gpuTiming`), DDGI imports only the contract, viewer warm-up and stats, `ddgi-batch-cost` page.
- Calibration: each batch size needs agreeing timings (last 3 within 5%, their median within 2% of the 3 before); a missed verify drops the point on the other side and re-measures; 300 back-to-back frames confirm the batch. The page and viewer run calibration and judged frames with **two frames in flight**: waiting for each frame left the RTX about 40% busy and its clocks wandered up to 2×, the real cause of the 1.7× reads in the review.
- No live bursts: a new `gpuMs` or cap is solved from the stored curve; a geometry change only marks the calibration stale (viewer: "Start the probes over" measures again); the first calibration waits for the mesh trees; filler only in the first calibration.
- Saturation: steps down only when frames are late and DDGI is at most 1.5× its calibrated cost; recovery waits 3 s on time, stops at 0.9× the first late batch, then probes up.
Verified: agent: lint, typecheck, `pnpm test` 867; RTX `ddgi-` (8) and three/spike (7) pages pass; Intel `ddgi-batch-cost` passes (cost reported). Lead: typecheck clean; `vitest packages/ddgi-core/src/platform apps/viewer/src/stats` 210 passed.
Timing (one cooled RTX run, 66–68 C at start, headless Chrome 156; **to re-run** as 3 runs under the new timing rule, DECISIONS): Cornell 1,728 probes (all) 1.84 ms; two-room 819, 1.99 ms; Sponza 276, 2.10 ms (`clock_low`, power limit); Bistro exterior 108, 2.03 ms. SM medians 1,567 / 1,665 / 1,477 / 1,590 MHz. Rows `note/budget_v2,1` in 88dbe45 are from the version the review failed.
Not verified: the 3-run timing set (paused for the fan-control reinstall); 30 / 120 Hz behaviour on the Intel.

## 2026-11-05 · T1.14 part 1: the composite (names pending) · implementer (opus)
Did (uncommitted; working names `giAlbedo`, `ddgiComposite`, not exported from `index.ts`; no `volume.start`, no `examples/three/`):
- `packages/ddgi-three/src/composite-mark.ts` (new): the pass mark through `pass.contextNode`, wrapping any existing context (`builtinAOContext` keeps working); `compositeTakesGI` is the one rule both sides use: marked pass, opaque, and a model `giAlbedo` reproduces (Standard; Physical without clearcoat, sheen, iridescence, transmission, dispersion or EON; Lambert; Phong). Everything else stays per fragment.
- `composite.ts` (new): `GIAlbedoNode` = `diffuseContribution × (1 − F_ss − F_ms) × AO`, built with three's own `PhysicalLightingModel.prototype.computeMultiscattering` and `DFGLUT` (`diffuseColor × AO` for Lambert and Phong; 0 elsewhere; transparent surfaces write their alpha, and the target gets the material's blending). `DDGICompositeNode` (extends `Node`, like SSGINode): depth to world position, `unpackRGBToNormal` normal, the light's `giIrradiance` once per pixel, `output + giAlbedo × E × intensity / π` in linear HDR before tone mapping; skips the background and weight-0 pixels; respects coverage as the per-fragment probe slot does.
- `gi-light.ts`: module nodes shared as `ShadingModuleNodes`; on composited surfaces the node evaluates no irradiance (it claims the probe slot with 0, coverage only when `computeCoverage`); the environment-diffuse workaround is unchanged.
- Pages: `three-composite` (Cornell, in the suite), `three-composite-sponza` (manual), shared code in `three-composite-shared/`; `three-shared/harness.ts` takes a width × height and renderer options.
Verified: lint, typecheck, `pnpm test` (867 passed; 8 new mark tests). The unmarked per-fragment WGSL is byte-identical to HEAD's for Standard and Lambert (dumped with `debug.getShaderAsync`, RTX). `pnpm test:gpu --filter three-` passes on both GPUs (7 pages each, RTX 3060 and Intel UHD gen-12lp). All on headless Chrome 156.
Numbers, per-pixel relative luminance difference against per fragment, opaque pixels, HDR, bookmark 1 at 1920×1080 (the stored shot), half-float targets; mean / p95 (bar 0.5% / 2%):
- RTX: Cornell point 0.043% / 0.16% (GI is 44% of the light); emissive Cornell 0.054% / 0.22% (35%); Sponza GI table 0.041% / 0.15%; Sponza textured 0.066% / 0.16% (GI 8–8.5% of the light at this bookmark).
- Intel UHD: 0.034% / 0.086%, 0.047% / 0.15%, Sponza GI table 0.055% / 0.18%, textured 0.080% / 0.23%.
- 8-bit `giAlbedo` and normal fail: Sponza GI table 0.62% / 1.89%, textured 0.69% / 2.09% (RTX).
- Transparent check (both GPUs): a `transparent: true` sphere at opacity 1 matches per fragment (RTX exactly 0; Intel mean 0.004%) and gets 70% of its light from GI. Behind an opacity-0.5 sphere the composite is −19% (RTX; Intel −18%): the transparent surface writes the pass's normal and depth, so GI for what's behind is evaluated at the sphere, facing the open front of the box.
- Render-time update (both GPUs): a light node with `updateType = RENDER` calling `ThreeGI.update` ran once per frame; its image after 4 GI frames equals the explicit update-then-render image exactly (0 difference; two explicit runs also 0), and differs from the explicit one a frame behind by mean 1.06%, p95 3.4%. Queue order: scene pass `beginRender`, shadow maps (nested), `submit ddgi-three frame`, then that pass's own submit. So yes, it lands before the render in r187.
Not verified: the cost. All runs so far were thermally throttled (bench-lock note: SM median 900–1,425 MHz under load, thermal throttling 22–65% of samples), and timing is paused for the fan-control reinstall. These are not valid numbers, but they show the shape (RTX, 2560×1440, frozen field): composite − off +1.76 ms (MRT +0.71, composite pass +1.15), forward − off +5.1 ms. The composite pass alone may exceed 1.0 ms. MSAA was not tested (the pass has no samples). Neither was a scene with `scene.environment`: there, composited surfaces still evaluate `giCoverage` per fragment.
Next:
- lead: re-run `three-composite-sponza` timing once the machine settles (`VITE_DDGI_SKIP=match`, short warm-up `VITE_DDGI_GI_FRAMES=8`; the page idles 25 s after loading, then ramps the clock 3 s, then about 10 s of interleaved rounds);
- if the bar fails, the levers are in core (cheaper or half-resolution irradiance in the composite) and need a decision;
- the transparent-behind error needs transparents drawn after the composite (an opaque pass, the composite, then a transparent pass);
- part 2 once Dennis confirms the names: export, `volume.start`, wiring the render-time update into `GILight`, `examples/three/`.

## 2026-11-05 · T1.9 Sponza accuracy check · implementer (opus), lead checked
Did (5434142): `SceneSync.sentMaterial(material)` returns the GI descriptor the platform got; page `ddgi-sponza-accuracy` (manual) with `run.ts` (writes `gi-materials.json`, EXRs and comparisons; `--table` prints the three scenes); `render.py --materials-json` (constant colours, alpha 1, no normal maps, checked before rendering) and a GI composite (DECISIONS: indirect + direct − lights-only direct). Convergence: 16 warm-up passes, then blocks of 32 full passes at hysteresis 0.97 until the mean moves < 0.5% between blocks (2–12 blocks); fixed 3,885 probes a frame.
Verified (agent): RTX 3060 and Intel UHD. Cycles: Blender 4.5, OptiX, 8,192 samples, 640×360, 321 s, under the lock; two seeds give 8×8-block noise 0.51% (per pixel 4.1%). Direct light agrees with Cycles to 0.02% (0.011139 against 0.011137), so the excess below is DDGI's.
Numbers (indirect, against Cycles; bias, 8×8 block error, p95 block, FLIP 8×8, FLIP per pixel):
- Cornell (point): +5.68%, 6.38%, 98.6%, 0.179, 0.197.
- two-room lit room: +5.63%, 5.98%, 12.0%, 0.073, 0.074.
- **Sponza (untextured): +26.29%, 29.88%, 162.8%, 0.331, 0.329. Fails the ±10% target.** Intel +26.25%. A long run (272 passes) +26.52%.
Not verified: the cause. The ratio map shows DDGI brighter on most shadowed surfaces, the curtains most.
Next: find Sponza's +26% (leaks through thin curtains, probe density, sun shadow rays, the decal `dirt_decal` with GI albedo 0 acting as a black opaque surface because glTF's default metalness is 1 and M1 ignores `blocksLight`); a README section for `--materials-json`; the test view's full image is black on sky pixels.

## 2026-11-05 · Why the RTX clocks stay low · implementer (opus), lead checked
Did: `bench-lock run` now samples temperature and nvidia-smi's throttle reasons along with the clocks. `note/clock_low,1` is written when the median SM clock over loaded samples (≥ 70% use, at least 3) is below 1,500 MHz, and the end-of-run warning names the cause (heat or the power limit). Files: `tools/gpu/clock-note.ts`, `tools/reference/bench-lock.ts` and `bench-lock.test.ts` (the lock part went into b009816 by the lead's mistake, the rest here), `bench/results/README.md`.
Verified (agent; RTX 3060 Laptop, headless Chrome 156): `three-screen-gi` on Sponza, 80 s from a cool 59 C, sampled every 200 ms:
- the clock ramps from P8 (210 MHz) to 1,400–2,000 MHz in about 1 s and holds for 15–20 s, at the 65 W power cap;
- at 85–87 C it drops, and from about 30 s it holds 450–600 MHz at 88–89 C, 58–60 W, 93–100% use;
- SW thermal slowdown in 197 of 395 samples; loaded median 555 MHz over 58 samples;
- short loads read fine (`trace-bench-sponza` 1,229–1,387 MHz).
Lead re-ran `vitest tools/reference tools/gpu/clock`: 35 passed.
Conclusion: the low clocks under sustained load are heat, not sampling and not submission gaps. Chrome flags unchanged.
Not verified: the laptop's fan mode (Armoury Crate Silent / Performance / Turbo), vents and stand (questions for Dennis); whether a cool-down plus short timed sections reproduces 1,500+ MHz medians for the T1.9 set.

## 2026-11-05 · T1.13 tooling · implementer (opus); HD renders running
Did (b009816, reference tooling only): every reference render's json holds a `shot` block (`ddgi-reference-shot/1`): camera position, target, up, vertical FOV, aspect, width, height, near 0.01 and far 2000; exposure and tone mapping; manifest id, path and sha256; sky and material-table hashes; commit and dirty flag; Blender, device, samples, seed, pixel filter, seconds; and whether a denoised copy exists, with its bias. `--from-shot` re-renders a stored view. Defaults: 1920×1080, Blackman-Harris 1.5 px, OptiX. `--denoise` adds a viewing-only denoised copy; the raw EXRs stay the accuracy source. The viewer's Compare (moves to the stored camera, letterboxes to its aspect, applies its exposure and tone mapping, flags another manifest or an older commit) and the DDGI pages at the reference's camera and size are done but not yet committed (those files are shared with agents still working).
Verified (agent, RTX, headless Chrome): the canvas letterboxed to exactly 16:9, and the green wall's edge lines up within 1 px in all-DDGI and all-Cycles views. Cost per sample at HD (both renders): Cornell point 0.085 s, emissive Cornell 0.12, two-room 0.24, Sponza untextured 0.26, textured 0.37. Noise: two seeds of Cornell point at 256 samples differ by 3.04% (GI) and 1.37% (full) after 8×8 blocks, scaling to 0.54% and 0.24% at 8192. The denoised copy's mean bias: −0.07% Cornell, −0.01% Sponza.
Plan (about 2¼ h of lock time, the lock taken per render so other timing can slot in): point-light Cornell 8192 × 2 seeds; the two-room views 4096; Sponza untextured 8192 and textured 2048 (denoised for viewing); emissive Cornell 4096; other bookmarks 1024 (denoised); then DDGI at HD and the accuracy table.
**Lead mistake:** b009816 also took the clock investigation's in-progress `tools/reference/bench-lock.ts` and its test, because the lead committed the folder. The snapshot is consistent (its 34 tests pass, typecheck clean); the agent carries on.

## 2026-11-05 · GI once per pixel: routes measured · implementer (opus)
Did: pages `three-vb-gi`, `three-screen-gi`, `three-screen-gi-shared` (measurement only; no adapter changes). RTX 3060, 2560×1440, Sponza with curtains, bookmark 1, 3 runs under the lock. **Clocks low** (SM median 592 MHz under load, Dennis's apps open), so the ratios count more than the ms.
Scene pass, GI off / on / added (medians, no shadow map):
- standard forward: 6.4 / 18.4 / +12.0 (another run 9.8 / 22.2 / +12.0);
- Visibility Buffer: 14.1 (10.5 is VB's own compute) / 20.3 / +6.2; with the fallback materials forced FrontSide 18.2 / 22.7 / +4.5;
- Meshlet: 5.9 / 18.1 / +11.8;
- **MRT route** (normals and depth from the main pass's extra outputs, GI once per pixel, materials read it by `screenUV`): MRT +1.1, screen pass 3.1, texture read about ±0.7: **+3.9** in total;
- prepass route: prepass 3.8 (6.2 at 4× MSAA) + screen pass 3.0: +6.5 (4×: +8.8).
The probe update (141 probes) adds 4.2 ms on top at these clocks. VB: 25 of Sponza's 50 materials fall back to forward (8 DoubleSide, which includes the curtains' 1.9 M triangles; 14 with vertex colours; 2 transparent decals), 2.33 M of 5.74 M triangles, about 70% of the VB frame's GI cost; VB also needs 23–33 s of CPU set-up. The MRT image differs on 3.6% of pixels and the prepass on 1.3%, at edges; mean brightness matches.
Qualitative: MSAA (MRT averages normal and depth at edges; a 1-sample prepass uses the centre surface); alpha-tested foliage (a prepass must run the alpha test; MRT gets it free); transparent objects and cube captures have no screen data and keep per-fragment GI. The MRT route reads last frame's GI unless the composite comes after the main pass (see DECISIONS).

## 2026-11-05 · Budget v2 · implementer (opus); Opus review: fail, fixes in progress
Did (88dbe45):
- `packages/ddgi-core/src/platform/work-budget.ts` (new) holds the whole budget:
  - a calibration burst: ramp the GPU up, bracket the budget with ×4 steps, alternate small batches with ~12 ms filler batches so a 60 Hz host keeps its clocks up; re-solve on a `gpuMs` change; give up after 400 frames without timings;
  - the saturation step-down: ×0.8 a second when frames are more than 10% late and DDGI is more than 10% over its share; ×1.1 a second back when on time;
  - the floor hook.
- `gi-platform.ts` re-calibrates on `prepare`, techniques added or removed, geometry (once still for 1 s; geometry-triggered at most every 10 s except the first), a `gpuMs` or cap change, and `recalibrate()`. Counters under 'ddgi budget'. T1.9a's frame-by-frame steering is gone.
- DDGI: `minProbesPerFrame: number | 'auto'` (a full sweep within 4 s, at most 2× the calibrated batch); counter 'seconds per full sweep'.
- `FrameBudget.targetFps` (platform.md Revisions).
- Viewer: the warm-up ends with the calibration; stats lines "per frame" and "measured cost" with help.
Verified (agent): lint, typecheck, test 848 (33 budget tests). RTX 3060, headless Chrome 156: ddgi-batch-cost on 4 scenes, the 8 `ddgi-` pages, the three and spike pages pass. Intel UHD: ddgi-batch-cost runs cleanly.
Numbers (`timing.csv` `batch_cost/v2_*`; RTX GPU-bound, back to back, 1 run): calibrated probes / ms (budget 2) / sweep at 60 Hz: Cornell 1,553 / 1.66 / 0.03 s; two-room 857 / 1.96 / 0.03 s; **Sponza 266 / 1.88 / 1.23 s** (39 probes at 60 Hz under T1.9a); Bistro exterior 112 / 1.89 / 4.0 s (the floor of 121 binds at 60 Hz). Calibration takes 20–52 frames (59–363 ms after the warm-up). At 30 / 60 / 120 Hz: one value per rate, no late frames, no step-downs. Intel: calibrated 90 / 92 / 58 probes cost 3.07 / 4.40 / 4.36 ms GPU-bound (it measured about half the sustained cost, probably short-lived turbo after the warm-up); reported, not judged (Intel is correctness-only).
Not verified: the saturation step-down on a GPU (unit tests only); the M5 Pro; a host with no warm-up on a GPU.
Note: one help-text hunk in this commit (the Compare camera) belongs to T1.13, whose code lands with T1.13.
**Lead mistake (again):** `gi-platform.ts`'s Budget v2 wiring landed in 0dde1f2 ("Frame breakdown"), because the lead committed the whole file for the breakdown's memory line while Budget v2 was editing it. So 0dde1f2 imports `./work-budget.ts`, which only exists from 88dbe45: **0dde1f2 alone doesn't build**, and `git show 88dbe45` hides the platform wiring. History isn't rewritten (other sessions may have branched from `m1`). Rule since: on any file an active agent may touch, the lead checks `git diff` hunk by hunk before committing, not only the path.
Review (Opus): **fail.** Re-run on the RTX: Cornell calibrated to 695 probes costing 1.14 ms (−43% of the budget), two-room 619 / 1.73 ms, against the claimed 1,553 / 1.66 and 857 / 1.96. The burst's cost line is about 1.7× the sustained curve, so 6 ramp frames don't reliably raise the clocks (the Intel overshoot is the same flaw the other way); the page only failed on going over. Also: filler batches and bursts during live use (not in the accepted design; at 120 Hz a filler exceeds the frame); a `gpuMs` slider drag re-ran the ramp each time; the "over its share" gate always passes at light load, so a CPU-bound late host would step down; possible ×0.8 / ×1.1 oscillation; `minUnitsPerFrame` found by name and `recalibrate()` outside the contract; new thresholds not in DECISIONS; stale design-doc sections; no TASKS row. Floor correct; no new readbacks. Being fixed per the lead's design calls: calibrate on sustained frames and verify; no live bursts; step down only when DDGI's time is near its full-clock cost (clocks up, GPU busy); hold after a step-down; contract additions.

## 2026-11-05 · T1.12 phase 1: Cornell flicker cause · implementer (opus)
Did: page `ddgi-flicker-causes` (manual, correctness only). No core or viewer edits.
Found (RTX 3060; the Intel gave the same; `cornell-box`, every probe updated each frame unless noted; image per-pixel change mean or p95; atlas rms per texel):
- base (h 0.97, 256 rays): 0.56% / 2.31%; atlas 0.91%;
- 32 / 128 / 512 / 876 probes a frame: 0.016 / 0.066 / 0.22 / 0.33% (fewer probes slow the wobble, not shrink it);
- h 0.98 / 0.99: 0.37 / 0.19%; 128 / 512 rays: 1.23 / 0.30%;
- a fixed rotation: 0.0046% (but the image reads 5.5% darker); also with a frozen frame number: 0.0016%;
- the firefly clamp off or 0.05: 0.56 / 0.55%; bounce off: the same absolute noise;
- point-light Cornell: 0.26% (σ 16%, the hot spot under the light).
Theory against 600 frames: per-frame change predicted 0.94%, measured 0.91%; spread over time 3.8% against 3.6%.
Cause and the chosen fix: DECISIONS. The red and green walls aren't special in the field; their pixels change about 15% more than white ones relative to their brightness, because only GI lights them.
Next (phase 2, after Budget v2 and the curtain fix leave those files): per-probe adaptive hysteresis, the viewer debug views, then Dennis judges by eye.

## 2026-11-05 · Frame breakdown, GI on against off · implementer (opus)
Did: viewer-baselines gets a breakdown mode (`breakdown.ts`), labelling three's passes by render context; `stats.memory` now includes the sky (`shade.memoryBytes`), with a test (the `ddgi-memory` page still adds it itself, so it double-counts: to fix there).
Found: **forward GI cost scales with fragment-shader runs, not visible pixels.** Sponza draws 5.7 M triangles; most are smaller than a pixel, and the GPU shades 2×2 blocks per triangle, so the GI function runs many times per pixel. A control material with no GI or textures costs 6.9× more on Sponza than Cornell on the RTX (13.6× on Intel); GI with one plain material shows nearly the same ratios (9×, 10.7×). A depth prepass doesn't help: Sponza shades only 1.98 fragments per pixel (Cornell 0.81, two-room 1.48). The update stayed at its budget (about 2.2 ms); warm-up, probe spheres, debug views and coverage were off. Textured materials made GI about 1.7× dearer on the RTX.
Scene pass, GI off → on (DDGI's own update listed separately):
- RTX 1440p: **Sponza 4.5 → 20.9 ms** (update 1.95; with prepass 21.5 + 3.3), two-room 1.2 → 3.65, Cornell 0.46 → 1.43.
- RTX 720p: Sponza 4.0 → 11.4 (update 2.2), two-room 0.16 → 0.64, Cornell 0.10 → 0.31.
- Intel 720p: Sponza 32.2 → 74.2 (+42; update 2.18), two-room 4.1 → 9.6, Cornell 4.8 → 12.3.
The RTX runs were GPU-bound (8 frames back to back), but Dennis's apps were open and the SM clock under load had a median of only 650–990 MHz, so the absolutes are noisy; the ratios held across runs.
Lever: shade GI once per pixel (a screen pass from depth and normals, which the contract's `encodeScreen` and `screenUV` already allow), or reduce geometry. Proposed to Dennis (DECISIONS).

## 2026-11-05 · Dennis reviewed the viewer · parent session relayed
His words: "it is not steady" (the flicker threshold). "The cornell box particularly the color sides is constantly moving. its like the lighting cant settle and is very noisy. I dont have debug views of the source channels or info to be able to provide more details. using the upscaler seems to settle this. Its not visible in either threejs version so its us." And: "640 x 360 is an insane size to judge ANYTHING against. Do a HD render and store that, that can be a baseline."
New M1 blockers (DECISIONS): Cornell's flicker; debug views of the source channels; HD Cycles references (1920+, anti-aliased, stored with their cameras, the viewer's Compare matching them). Earlier he wrote: "The viewer loads but quality has issues, they may be addressed later."

## 2026-11-05 · T1.9 memory, flicker and the Intel set · gpu-tester (sonnet), lead checked
Did (0db31d6): pages `ddgi-memory` and `ddgi-flicker` (new), `ddgi-shared/viewer-scene.ts`; width and height options on `ddgi-shading-timing`; rows in `memory.csv`, `flicker.csv`, `shading.csv`, `timing.csv`, `baselines.csv`. Every timing under the lock; Dennis's Chrome open, about 2% 3D load from other processes.
Numbers:
- **Memory (MB; scene copy / acceleration / DDGI / total, plus the sky the total leaves out):** Cornell 1.07 / 0.17 / 5.62 / 6.85 (+0.27); two-room 1.07 / 0.17 / 3.88 / 5.12; **furnished room (4,998 probes) 5.49 / 20.57 / 13.29 / 39.35, 40.66 with the sky: meets ≤ 64 MB**; Sponza 114.1 / 469.9 / 35.9 / 619.9; Bistro interior 26.6 / 82.0 / 35.5 / 144.1; Bistro exterior 88.5 / 220.4 / 50.7 / 359.6.
- **Flicker on a still scene (per-pixel change, mean / p95, median of 3):** at the 2 ms budget: RTX Cornell 0.31% / 1.35% (876 probes a frame), two-room 0.030% / 0.11%, Sponza 0.0016% / 0.0051%; Intel Cornell 0.033% / 0.155%. At a fixed 256 probes: RTX Cornell 0.12% / 0.63%. The image mean drifts less than 0.4%. Frames: `bench/screenshots/t1.9-flicker/`.
- **Intel UHD (median of 3):** GI sampling front / inside 720p 2.14 / 3.53 ms, 1080p 4.66 / 7.81 ms. Frame GI on minus off at 720p: Cornell +9.5 ms (15.5 / 5.8), two-room +6.4, **Sponza +43.2 ms (93.0 / 49.4)**. Per probe back to back: Cornell 11.1 µs + 0.34 ms, two-room 16.0 + 0.30, Sponza 22.4 + 0.37. Startup from handoff / from file request: Cornell 0.99 / 1.18 s, Sponza 5.2 / 7.5 s (settled 15.0 s; the 8 s warm-up cap hit at 15.4 passes).
Not measured: Intel memory (the same bytes), Intel flicker beyond Cornell, the Intel clock.
Open: Sponza's +43 ms on Intel is far more than GI sampling (about 3.5 ms) plus a 2 ms update; T1.9 has to explain it. `stats.memory.total` leaves out the sky (0.26–1.3 MB): fix.

## 2026-11-05 · Decals out of GI (adapter) · implementer (opus), lead checked
Did (90c54e2 and the next commit): an alpha-blended three material (glTF BLEND: `transparent` with `alphaTest` 0) maps to `blocksLight: false`. A `Mesh` whose materials all block no light is sent with `contributesToGI: false`, because M1's tracer doesn't honour `blocksLight` per triangle (M4). A mixed mesh stays in, with a one-time warning. `material.userData.giBlocksLight = true` keeps a blended material in GI. Also fixed a crash when a material was swapped away. `SceneSync.sentSeeThrough(mesh)`; the GI material table lists the loaded parts and the meshes left out; `render.py` loads only the listed parts and removes the left-out materials' faces (Sponza: `dirt_decal`, 8,140 faces), stopping if a material is only partly left out.
Verified (agent): lint, typecheck, test 837; three-neutrality and three-environment on both GPUs; all 8 `ddgi-` pages on the RTX (covering the investigation's harness change). Sponza against the new untextured Cycles reference (decal faces removed, 8,192 samples, OptiX, under the lock):
- **with curtains: RTX +23.03%, Intel +22.99%** (8×8 block error 26.7%), down from +26.3%;
- **base package: RTX +3.05%, Intel +3.02%** (block error 9.4%);
- the path tracer against Cycles: +0.16% and +0.14%; DDGI against the path tracer: +22.83% and +2.91%.
So the decals were about 3 points; the rest is the curtain limit. The canonical `cycles-untextured-*` references were overwritten; the base one is `cycles-untextured-*-base`; the `-seed1` files are stale.

## 2026-11-05 · T1.9 quiet-window measurements · gpu-tester (sonnet), lead checked
Did (453d213): with Dennis's GPU apps closed, about 57 min, all under the lock, medians of 3 (rows `note/quiet,1`; native label `t1.9-quiet`). Before every set the RTX read 0% use, 210 MHz, P8, 12 W, and nothing else ran on it (only the compositor and the Claude app, about 1.3% each, on the other adapter). SM clock under load about 1400–1500 MHz. HEAD 90c54e2 plus tool-page options (width and height, `--note`, per-pass steady medians).
Numbers (RTX 3060, headless Chrome 156):
- **GI sampling at 1440p:** front 0.548 ms, inside 0.923 ms (0.923–0.960); bar 1.0 on the inside view: met.
- **Startup, cold Chrome profile (the driver's shader cache was probably warm), first GI drawn,** from handoff / from file request: Cornell 1.44 / 1.63 s, two-room 1.40 / 1.56 s, Sponza 4.75 / 7.63 s. Target ≤ 2 s from handoff: met on Cornell and two-room, missed on Sponza.
- **Whole frame, GPU ms, GI on minus off, p50 (p95 delta):** 720p Cornell +2.56 (3.66), two-room +2.57 (3.01), Sponza +4.23 (3.58); 1440p Cornell +2.41 (4.83), two-room +3.18 (2.85), Sponza +6.18 (5.43). Target ≤ +3 ms p50. Inflated by the power-state effect below.
- **Fixed batch, update ms p50 / p95** at 60 Hz, then back to back: Cornell 2.04 / 2.19 and 1.07 / 2.05; two-room 2.04 / 2.23 and 1.10 / 2.07; Sponza 2.33 / 8.90 and 1.11 / 4.60; Bistro exterior 2.89 / 10.36 and 1.33 / 5.07.
- **The 2 ms budget:** Cornell 2.04 ms, 118 / 138 / 147 probes (sweep 0.2 s); two-room 2.03 ms, 33 / 39 / 44 (0.5 s); **Sponza 2.35 ms, 1 / 2 / 3 probes (sweep 162 s); Bistro exterior 2.49 ms, 1 / 3 / 4 (161 s).** The page's ±10% check failed on Sponza and Bistro (`note/page_failed`).
- **Sponza at a fixed 20 probes,** trace / blend / write ms: 60 Hz 2.47 / 0.13 / 0.01 (total 2.63, p95 7.05); back to back 1.37 / 0.05 / 0.01 (1.44, p95 4.81). Spikes every other frame (33.3 ms apart), not the outside 50 ms pattern.
- **Native, matched grids** (trace / blend ms): Cornell 0.68 / 0.40, two-room 0.53 / 0.35, Sponza 11.4 / 4.02 (slower than the 8.6–8.8 trace measured with Dennis's apps open).
- **Three's baselines on Bistro exterior:** still too noisy to quote (GI-cost coefficient of variation 1.1–2.0). VXGI frame 24.4 ms, LightProbeGrid 10.9 ms (`baselines.csv`).
**The finding:** on a quiet machine at 60 Hz with a light load, the RTX drops into its low-power state (P8, 210–560 MHz; median SM clock 240 MHz over the budget set). With Dennis's apps running, they kept the clocks up. So the 2 ms GPU-time budget bought 2–3 probes on Sponza where it bought 30–43 before, and much of the "fixed cost" chased in T1.9a is the power state. The whole-frame deltas and native's slower trace show the same effect. A budget in GPU milliseconds isn't stable under power management (proposal in DECISIONS).
Not measured: the Intel GPU in a quiet window.

## 2026-11-05 · Sponza +26% investigation · implementer (opus), lead checked the logic
Did (079e2d5): `ddgi-shared/path-tracer.ts` (a path tracer over the platform's tracer and hit shading) and `probe-values.ts`; the Sponza page gained `ddgi`, `parts`, `excludeMaterials`, `pathTrace` and `probeCheck` options and probe-state counts; `run.ts` writes and compares `pt-untextured-indirect*.exr`. No core code changed.
Verified (agent; RTX 3060 and Intel UHD): experiments, each one change (DDGI against Cycles or the path tracer):
- the path tracer against Cycles: +0.15% (black back faces instead of shaded: −0.17%). So the gap is DDGI's;
- **without the curtains (base package): +2.59%,** block error 9.95%; Intel +2.57%. With curtains, against the path tracer, +26.10%: curtain pixels 1.41×, other pixels 1.17×;
- probe spacing: 1 m +26.29%, 0.75 m +19.71%, 0.5 m +12.33% (block error 16.9%, FLIP 0.244);
- probes against the path tracer at their own positions: +0.47% (1 m), +0.32% (0.5 m). Exact irradiance at DDGI's biased lookup point: +11.5% (0.1 m biases), +6.9% (0.05 m);
- biases 0.05/0.05 +25.67%, 0.1/0 +25.21%, 0/0.1 +26.08%, 0.2/0.2 +28.01%; `minFrontFaceDistance` 0.1 m +25.09%, 0.3 m +26.53%;
- relocation off +33.40%; classification off +25.95%; both off +28.04%; back-face distance share 1.0 +25.80%; 30 distance texels +26.77%; 512 rays +26.37%; 10 irradiance texels +26.79%;
- `dirt_decal` left out: +22.83% (opaque black on both sides, so not the cause; it darkens the true GI by about 30%).
Conclusion: a probe-resolution limit on thin folded cloth, not a bug; no fix made. Sponza stays +26.29% / +26.25%. lint, typecheck, test 831.
Not verified: `test:gpu --filter ddgi-` after a one-word `export` change in `ddgi-shared/harness.ts` (paused for the quiet window; the lead runs it after); a Cycles render of the base package (the path tracer stood in).
Proposed (DECISIONS, waiting on Dennis): judge M1 Sponza accuracy on the base package, report the curtains as a known limit. Follow-ups: honour `blocksLight: false` so decals don't act as black sheets (with M4's alpha cutouts); a near-surface occlusion term (after M1).

## 2026-11-05 · Review of the sky-aware GI reference · reviewer (opus)
Asked by the parent session, so the reference isn't judged by the agent that wrote it. Verdict: **pass with notes** (all from the reviewer's own renders and code reading).
- DDGI's shaded GI term equals the composite's contents (DECISIONS has the details); the sun's direct light matches to 0.02% and Cycles full = lights-only direct + composite (0.012070 against 0.012069).
- Cornell (point) and two-room with the composite: unchanged (+5.68%; two-room +5.50% on OptiX against +5.63% on the CPU reference, an OptiX-against-CPU difference of 0.13 points, the same without `--materials-json`).
- `--materials-json`: all 49 Blender materials set, all 32 rows matched, the table hash recorded; Metallic, Transmission and specular forced to 0; `glass` and `dirt_decal` opaque black on both sides (consistent, not physical).
Fixed later the same day (implementer): the lights-only render swaps in a black world and zeroes every emitter (Principled, Emission and Background shaders, volume, node groups), refuses to render if anything still emits, and is checked by a new `--calibrate` case (lights-only 3.97888 against 3.97887 expected; the old switch-off would have left 3 of its 4 emitters on). Every render writes the composite; `run.ts --table` uses it for all three scenes (Cornell point and two-room re-rendered on the CPU, 2048 spp: lights-only equals direct exactly, so the composite equals indirect, pixel-identical). README and design-doc sections added. Table: Cornell +5.68%, two-room +5.63%, Sponza +23.03% (FLIP 8×8: 0.179, 0.073, 0.326).
Originally listed, to fix after the Sponza investigation:
- a check that the lights-only render really is lights-only (`render.py:259-265` only turns off Principled emission and the Background);
- `run.ts --table` using the composite for all three scenes;
- README and design-doc sections;
- DECISIONS saying when the definition was set (done).

## 2026-11-05 · B1 merged into m1 · M1 lead
Merged `refs/heads/bvh-b1` (48b2aed) into `m1`, Dennis approved (provisional answers). Conflicts only in append-only files (PROGRESS: B1's entry kept above; `bvh.csv` and `trace.csv`: both sides' rows kept). The LBVH is not wired into `ddgi-core` in M1 (M2 "fast first GI").
Verified (lead, on the merged tree): lint, typecheck, test 831 (58 files). On the RTX 3060 and Intel UHD: trace-oracle, trace-inline, tlas-oracle, bvh-lbvh, bvh-lbvh-fallback and bvh-blas-refit all pass.

## 2026-11-05 · B1 · lead side session (opus); Opus review: pass with notes, fixed; merged into m1 (Dennis approved); done
Did (branch `bvh-b1`, from `m1` at 6555a1c; nothing outside `packages/bvh`, its pages, the trace-bench and trace-oracle pages, and B1 rows):
- `packages/bvh/src/gpu-lbvh.ts` + `gpu-lbvh-wgsl.ts`: GPU LBVH in the tracer's layout (bytes the tracer already reads, so no tracer or DDGI change). Triangle boxes and centroid bounds (subgroup, workgroup, then one atomic per workgroup); 30-bit Morton codes in cubic cells over each mesh's own centroid box; a stable radix sort; Apetrei's bottom-up climb with HIPRT's key-plus-index tie-break (ported from HIPRT `LbvhBuilderKernels.h` / `MortonCode.h`, MIT, `licenses/hiprt.txt`, notice kept in the WGSL); SAH leaf collapse decided on the way up (leaves up to 8, costs 1 and 1); a scan numbers the kept nodes; triangles, nodes and parents written out. Batches: every mesh shares each dispatch (mesh table, tile table, segmented sort), output in three shared buffers with a node offset per mesh; `planGpuLbvhBatches` splits by the device's buffer limit. Handed-over words go through atomics with a bounded wait; a tree whose wait ran out, or with an out-of-range index, is refused (`problem`). Leaves are never handed over (positions say where they are). The climb runs in two rounds (the second an indirect dispatch).
- `gpu-radix-sort.ts` (5-bit digits, reduce-then-scan, no look-back), `gpu-scan.ts`, `gpu-workgroup-scan.ts`: workgroup scans number threads in subgroup order, so no subgroup size or lane layout is assumed.
- `gpu-mesh-refit.ts`: BLAS refit from an adapter's vertex buffer (any stride/offset, 16- or 32-bit indices), then the B1a refit; works on any builder's tree, inside shared scene buffers.
- `mesh-builder.ts` (`createMeshLayoutBuilder`): the Builder contract's chooser: GPU LBVH with subgroups, tinybvh (B0) without; single meshes that are too big or refused also go to tinybvh, and the result says which builder made each tree.
- `layout-check.ts` (any layout against its mesh, exact boxes; `treeSahCost`), `lbvh-cpu.ts` (the same LBVH on the CPU, for tests and cross-checks), `wgsl-names.ts` (reserved words, name clashes) with tests.
- Pages: `bvh-lbvh`, `bvh-blas-refit`, `bvh-lbvh-fallback`, `bvh-lbvh-profile` (manual), `trace-bench-lbvh-sponza` (manual) and `-quick`; trace-oracle got a `watertight-lbvh` variant; trace-bench got an `ours-lbvh` tracer (not in its default tracer list, so existing pages run as before) and its CSV rows.
Verified (agent, headless Chrome 156): lint, typecheck, test (all, 761 incl. 20 in packages/bvh), build. On the RTX 3060 (ampere, subgroups 32) and the Intel UHD (gen-12lp, subgroups 8-16): bvh-lbvh (scan of up to 3M, segmented sort of up to 1M keys against CPU results; 11 meshes in one batch through the layout checker, node counts equal to the CPU twin's), bvh-blas-refit (5 deforming frames, GPU LBVH and tinybvh trees, exact), bvh-lbvh-fallback (device without subgroups: tinybvh chosen, LBVH refuses to start), trace-bench-lbvh-quick, and trace-oracle with all four variants: 0 mismatches, 0 escapes, 0 of 2M shared-edge rays slipped for `watertight-lbvh` on both GPUs.
Numbers (median of 3, bench lock, RTX 3060 Laptop; `bench/results/trace.csv` and `bvh.csv` at 600bba0):
- Sponza (base + curtains, 5,744,384 triangles, 132 meshes): GPU build 31.2 ms (27.7-33.8; bar 50). Wall time of the first build with uploads and the stats readback 302 ms; tinybvh in the worker 3909 ms. LBVH trees: SAH cost 1.217x tinybvh's, 3,263,758 nodes (tinybvh 3,033,036), mesh depth 33 (28), 0 meshes fell back.
- Trace on the same rays: closest-hit 80.5 Mrays/s on LBVH trees vs 98.4 on tinybvh's (0.82x); any-hit 171.2 vs 212.7 (0.80x); agreement 100%.
- Top level, 10k moving instances (B1a's page, unchanged code): refit pass 0.091 ms (bar 0.5; B1a recorded 0.086).
- How the build got there (`bvh-lbvh-profile`, exploratory): per-mesh builds 97 ms (dispatch-bound: the largest mesh alone 2.3 ms; contended atomic loads of one mesh's bounds 23 ms) -> batched 42 -> 32. Last profile by step: climb 11.7, sort 5.8, triangles 3.5, emit 2.0, clear 1.8 ms.
Not verified: build speed on Intel (no bar there; not recorded). Bistro scenes not run with the LBVH. The LBVH is not wired into ddgi-core's scene (the platform still builds mesh trees with tinybvh in its worker): that is ddgi-core work, kept out of this task. B1 trees are read back for the tracer's CPU-side scene assembly; a GPU-resident path needs that wiring too.
Clock: not locked. `nvidia-smi` after the runs showed the SM clock's maximum at 2100 MHz (idle 1147 MHz). I did not sample it under load during the timed runs. The CSV rows are dated 2026-11-04 because the runner writes the UTC date.
Device limits: the 31.2 ms is for one batch, on a device asking for the adapter's largest buffers (trace-bench does; Sponza needs a 436 MB scratch buffer). With default limits (128 MB bindings) `planGpuLbvhBatches` splits Sponza into about 4 batches, and `createMeshLayoutBuilder` builds them one after another with two readbacks each. That path is tested for correctness, not timed.
Review (Opus): pass with notes. It re-ran lint, tsc, test (761), build, the bvh pages and trace-bench-lbvh-quick on both GPUs, and trace-oracle on both (watertight-lbvh 0 slipped). It checked the CSV rows against the claims, and broke trees by hand to confirm the checker catches them. Notes, and what was done:
- the checker had no test of its own: `layout-check.test.ts` now breaks trees seven ways (box one step too big, triangle twice, short leaf, moved corner, wrong parent, wrong mesh box, wrong depth) and each must be reported;
- untested inputs: bvh-lbvh now builds GPU buffers with 16-bit indices at an odd offset and an odd-length 16-bit list; the batch planner has unit tests (multi-batch builds on the GPU still aren't run);
- magic indices: node-info words, slot words, index formats and uniforms are named;
- GPU-buffer inputs too small for the stated counts were silently clamped: now refused when the batch is made;
- a failed stats readback leaked the output buffers: fixed;
- a comment for the zero-start of workgroup memory, and a stale method name in a comment;
- open: the chooser (`createMeshLayoutBuilder`) takes CPU arrays and returns CPU copies; the Builder contract's GPU-in, GPU-out path is `GpuLbvhBuilder` itself, and the platform doesn't use it yet (Next).
After the fixes: lint, tsc, test (packages/bvh 30), and the bvh pages pass on the RTX and the Intel UHD.
Next:
- ddgi-core: take GPU-built trees without a readback (the batch output is already laid out like the scene buffers), and use `GpuMeshRefit` for deforming meshes (M3), with a rebuild when `relativeCost` drifts.
- Speed: the climb (11.7 ms) is neither bound by the words it hands over nor fixed by the two rounds; profile it on its own. The sort could rank in shared memory and write runs per digit.
- Trace speed on LBVH trees (0.8x) is what B2 (PLOC, H-PLOC) is for.

## 2026-11-05 · T1.9a · implementer (opus); reviews: pass with notes, then controller fail; fixed; done
Did (9ba4c5c):
- The budget model is fixed + per-probe from medians, in gi-platform.ts (DECISIONS). `msPerUnit()` became `costModel()`. 20 unit tests, including a lagged, spiky simulation.
- `ddgi/trace` reads the bounce light before the shadow ray: 4–30% faster in A/B runs.
- `prepare` compiles platform and technique kernels in parallel: RTX cold compile 2.46–2.92 → 1.43–1.90 s.
- A new timing page `ddgi-batch-cost`, through the three adapter (startup, budget at 60 Hz, curve, steady, breakdown), with `to-csv.ts`.
Verified (agent): lint, typecheck, test 795; `test:gpu --filter ddgi-` passes on both GPUs with T1.5b's numbers unchanged (furnace −0.16% / −0.28%, Cornell +5.68% / +5.63%, two-room leak 0.797% / 0.795%).
Numbers (timing.csv `batch_cost/`; RTX, median of 3, under the lock, 60 Hz):
- Update ms at a 2 ms budget, and probes per frame: Cornell 2.03 / 995 (old meter 1.17 / 515), two-room 2.02 / 337 (1.15 / 197), Sponza 2.08 / 39 (1.45 / 14), Bistro exterior 2.09 / 23 (1.29 / 10). Intel (1 run): 1.98 / 139, 2.00 / 45, 1.95 / 16, 1.98 / 11.
- Cost curves and the Sponza breakdown: NOTES.
- Startup from technique creation, RTX (first GI / settled): Cornell 1.97 / 2.08 s (compile 1.86 s); Sponza 4.68 / 6.88 s (trees 3.32 s, scene copy 1.31 s; file loading 2.4 s before that); Bistro exterior 6.03 / 10.36 s (scene copy 4.07 s).
- p95: other programs' GPU work every 50 ms lands inside our passes (RTX +2.5 ms, Intel +1.3 ms). With a fixed batch and no overlap, 0 spikes.
Not verified: a quiet GPU (Dennis's apps were running); Intel cost curves.
Review (Opus): lint, typecheck, test 795 (two timing-based `bench-lock.test.ts` failures on a first run passed on re-run); all 8 `ddgi-` pages on both GPUs with T1.5b's numbers exactly, so moving the bounce read didn't change results. Budget re-run under the lock on the RTX: Cornell 2.092 ms median (896 probes), Sponza 2.142 ms (+7%, 27 probes, p05 1 probe). Within ±10%, not ±5%. Corrections:
- `timing.csv` holds **two** RTX sets with the same labels, both tagged `146ce4b-dirty` (not 9ba4c5c). The numbers above are the second set (rows 2470 on). The first (row 1809 on) had Bistro exterior at a 2.177 ms median, its run 3 at 2.254 ms (+12.7%), and probe medians of 949 / 395 / 45 / 30.
- **Startup is understated** against SPEC's "load to first visible GI": the figures above exclude file loading (Sponza 2.3 s, Bistro 6.6–7.8 s) and had warm compiles (Cornell compiled first on the same device). "First GI" was the first warm-up step finishing, not a frame shown. From load: Sponza about 7.0 s, Bistro exterior 12.6–14.4 s.
- The p95 attribution is circumstantial and recorded only in prose (DECISIONS now says "likely").
Fixes (be46b60): `units` applied last; steering by the median of recent same-size frames (×0.8–×1.25 per frame); buckets aged out after 1,800 frames and reset on geometry changes; the p95 evidence in the CSV (`batch_cost/steady/`: 27–30% of fixed-batch frames hit, +2.0–2.3 ms, exactly 50 ms apart); the page asserts the median within ±10%; startup from load with a cold compile, one scene per run, first GI drawn / settled: Cornell 2.34 / 2.45 s, two-room 2.33 / 2.41 s, Sponza 10.8 / 13.1 s (files 4.2 s, trees 3.3 s, scene copy 1.2 s, compile 1.2 s). **Even the small scenes miss SPEC's ≤ 2 s from load on a cold compile.**
Controller fix (after the re-review): steering keys on the units used, and the hand-out is at most 1.25× + 1 of what the technique could use last frame. Inside ±5% and at or under budget the batch grows ×1.1 every 6 frames, and backs off once over. Tests drive the real steering path (`BatchSteering`): 8 → 1 → 8 ms, a budget bigger than all the work, the reviewer's case, `maxProbesPerFrame`. A flat-curve simulation settles above 170 probes. lint, typecheck, test 809 (lead re-ran the platform tests: 190 pass); `ddgi-` pages on the RTX; on Intel the first run failed `distance-format` and every page after it, and the re-run passed all eight (the known Intel hang, NOTES).
RTX, 3 runs under the lock, judged from 15 s after warm-up (median ms; probes p05 / median / p95; full sweep):
- Cornell 2.04–2.07; ~440–550 / ~875–910 / ~1,500–1,600; 0.03 s.
- two-room 1.98–2.03; 110–253 / 294–340 / 460–566; 0.06 s.
- Sponza 2.02–2.06; 9–16 / 30–43 / 87–181; 7.5–10.8 s.
- Bistro exterior 1.98–2.02; 7–11 / 20–27 / 50–90; 18–24 s.
**Sponza's 60 Hz cost isn't flat after all:** fixed batches of 20 / 40 / 128 / 190 probes cost 1.55 / 1.91 / 2.81 / 2.91 ms, so about 40 probes is what 2 ms buys. The batch never settles to one size: it follows other programs' GPU load, 12 to 225 probes second to second.
Re-review (Opus): **fail.** Steering is keyed on the batch handed out, not the units used; when the budget exceeds the work the batch freezes and later budget cuts are ignored (simulation: 8 → 1 ms leaves 1.55 ms forever). And at 60 Hz Sponza's time is nearly flat from 20 to 190 probes, so the controller settles at about 20 (a 16 s sweep) when about 190 fit (about 2 s). Being fixed: key on units used; climb while inside the budget band.
Earlier, being fixed: the drop limit can exceed the `units` hard cap (contract bug); the batch oscillating down to 1 probe on Sponza and Bistro; size buckets that never expire; the p95 evidence exported to the CSV; the page asserting the median within ±10%; startup measured from load with a cold compile.
Gaps explained, not closed (DECISIONS): Sponza's update rate (shadow rays; M2) and large-scene startup (B2 and M3).
Follow-ups: compile `encodeBatch` only on first use; find why the sky SH projection takes 1 s to compile; the viewer pauses DDGI until the trees are built (being fixed); `docs/design/ddgi.md`'s budget bullet and a `batch_cost/` section in `bench/results/README.md`.

## 2026-11-05 · T1.8 · implementer (opus); Opus review: pass with notes, fixed; Dennis tried it; done
Did (8fe593e):
- DDGI is the viewer's default technique (`ThreeGI` on a `createGIRenderer` device; the manifest volume; `platformSkyEnvironment`; the sky dome excluded from GI).
- Warm-up of up to 16 passes, then a 2 ms budget slider.
- Probe spheres from the `points` texture, with a legend and counts; screen debug views (GI only, coverage, probes near this point); "Start the probes over".
- The Cycles reference as a wipe side, with a Compare button, aligned by field of view and aspect; it says when you're off its bookmark, and gives the render command when the image is missing.
- A stats overlay with hover help: probes per frame, update GPU ms (trace / blend / write), top-level refit, memory, adapter sync and encode CPU ms.
- Per-scene `exposureEv` in manifests (DECISIONS).
- viewer-baselines measures `ddgi` (start-up, GI on against off, frame GPU and wall p50/p95). Fixed a bench bug where loading a scene created the previous technique.
Verified (agent): lint, typecheck, test 779, build:apps. viewer-baselines passes on the RTX 3060 and Intel UHD (headless Chrome 156): the quick default, and Cornell, two-room and Sponza with `none,ddgi`. Contact sheet: every non-heavy scene, 0 errors on both GPUs. Screenshots in `bench/screenshots/t1.8/`. The lead looked at the Sponza reference wipe and Cornell's probe spheres.
Not verified: sponza-full; a headed browser by hand. Dennis tried it (2026-11-05): "The viewer loads but quality has issues, they may be addressed later." Usability is met; his quality notes go to T1.9 and sign-off (the parent session collects them). No timings recorded as results.
Found:
- **Sponza's GI is near black:** 25 of 28 materials have metalness factor 1 with a metalness texture, so folding in the factor alone gives GI albedo 0. The adapter must fold in the texture's average metalness (with T1.7's fixes).
- **The emissive Cornell counted its lamp twice:** the analytic area light plus DDGI's emitter light. DECISIONS: the viewer turns the area light off with DDGI on.
- **Sponza's probe update is slow:** at 2 ms, 59 probes a frame (trace 1.92 ms, about 32 µs per probe, against about 2 µs on Cornell). Each probe is updated every 330 frames. GPU memory 627 MB (scene copy 114 MB, ray tracing 470 MB, probes 43 MB). T1.9 investigates.
Review (Opus): lint, typecheck, test 779, build:apps; viewer-baselines passes on both GPUs (DDGI ran all 16 warm-up passes, no errors); the contact sheet on the Cornells and two-room has 0 errors. Confirmed: the probe spheres read the points texture in the vertex shader (no readback); the warm-up never blocks the CPU; the reference uses the same exposure and tone mapping, and its off-bookmark message is honest; GI on and off are measured the same way. Being fixed: the stats overlay blocks orbiting; tooltips are lost when rows are rebuilt; per-frame allocations; the update ms shown against its budget; the exposure slider range; a stale screenshot; the render command's resolution; the legend's string matching.
Fixes (1081258): the stats box lets drags through again (the help moved to a "What the stats mean" panel section); rows update in place; snapshot maps reused; the update line shows "x ms (budget y)" and names all four passes; exposure −8…+8 in both slider and validation; the render command uses the scene's reference size; legend counts from one table; the viewer uses T1.7's exports (`DDGI_DEBUG_COLORS`, `wrapGpuTexture`, `wgslParameterType`) and drops its coverage workaround. lint, typecheck, build:apps; viewer-baselines (quick) on both GPUs (agent). `tools/assets/fetch.test.ts` fails now and then under load and passes alone: a flaky test, follow-up.
For T1.9:
- **The update can overshoot its 2 ms budget:** two-room 2.92 ms and Bistro exterior 4.54 ms in the screenshots.
- **DDGI's frame p95 is 4.28 ms against a 1.10 ms median on Cornell** (RTX), while GI off's p95 matches its median.
Both need explaining before T1.9 quotes them.
- Copies and duck-typing to clean up: the viewer copies `DDGI_DEBUG_COLORS` and the `ExternalTexture` wrapping, and reaches `rays.whenBuilt`; the adapter turns coverage on for any `LightProbeGrid`, whatever its layer.

## 2026-11-05 · RTX re-time after the clock unlock · gpu-tester (sonnet), lead checked
Did (fbc9523): `bench-lock run` samples `nvidia-smi` (SM and memory clock, utilisation) about once a second and writes them out. Bench rows now carry `note/sm_clock_mhz` (the median over all samples, which reads low for some pages; README says so). Re-timed under the lock, 3 runs each, after Dennis reset the clock lock.
Clocks seen: under load the SM clock held about 1610–1660 MHz (min 780–1440, max 1730–2010), not 2100. nvidia-smi showed the power cap (about 72 W) and thermal slowdown (87 °C). Dennis's own apps keep the RTX at 16–28% when idle, with spikes near 50%.
T1.6 GI sampling at 2560×1440 (bar 1.0 ms on the inside view; median of 3 per set):

| set | front | inside |
| --- | --- | --- |
| 1 | 0.523 (0.507–0.599) | 1.101 (0.905–1.132) |
| 2 | 0.517 (0.494–0.545) | 0.882 (0.877–0.905) |
| 3 | 0.526 (0.494–0.536) | 0.910 (0.885–0.920) |

Nine runs: inside median 0.905 ms (0.877–1.132). Set 1 jumped and was repeated twice. Locked at 1200 MHz it read 0.62 / 1.07.
Spot checks of today's RTX numbers ("changed" = more than about 10%; new rows added, old rows kept):

| check | old (locked or unknown) | new | changed? |
| --- | --- | --- | --- |
| sponza-base closest, Mrays/s | 107.3 | 105.1 | no |
| sponza-base any | 168.3 | 196.0 | yes, +16% |
| furnished-room closest | 157.3 | 144.9 (128–166) | no |
| furnished-room any | 512.3 | 610.0 | yes, +19% |
| top-level refit, 10k, ms | 0.0859 | 0.0959 | +12%, near timer noise |
| DDGI Cornell, µs per probe | 2.31 | 1.91 | yes, −17% |
| DDGI two-room, µs per probe | 3.90 | 2.46 | yes, −37% |
| native Sponza trace, ms | 9.845 | 8.82 (8.61–10.39) | yes, −10% |
| native Sponza blend, ms | 3.48 | 2.87 | yes, −18% |
| native Sponza DDGI mean, ms | 16.20 | 16.44 | no |

Not clean: the first trace-bench set had runs 4–10× slower than the rest (Sponza 117, 23, 9 Mrays/s), marked `note/gpu_busy,1`. A second set had the same problem and wasn't appended. The cause wasn't found; other agents' correctness pages and Dennis's apps share the GPU. Native runs were no longer flat to 1% as the locked ones were.
Next: T1.9 measures with the clock note and on an idle machine where possible. Variance matters more than the clock now.

## 2026-11-05 · T1.7 · implementer (opus); review failed, fixed; re-review passed with notes; done
Did (d0e3525), in `packages/ddgi-three`:
- `GILight`: a `Light` subclass with a registered node. One `wgslFn` built from `gi.shading.parameters`, returning irradiance and coverage together (coverage computed only when an environment or `LightProbeGrid` needs it). `ExternalTexture`s re-read every render, `renderGroup` vec4s, three's light-probe slot at priority 1, given up at intensity 0. The env-diffuse workaround wraps the lighting model's `indirect` (Lambert and Phong get no env diffuse at all; measured 0).
- `SceneSync`: meshes, groups, normals, uv and uv1, metalness-folded materials, texture handles with `UvMapping`, `InstancedMesh`, visibility, lights, environment; the cube mirror and the environment upload.
- `ThreeGI` (`update(camera, dt)`, `light`, `timings`, `report`, `dispose`), `requestGIDevice()` / `createGIRenderer()`, `GI_USER_DATA`.
- three caches shaders by light id, so `ThreeGI` replaces the light when `shading.version` changes.
- Pages three-neutrality, three-environment, three-adapter-timing (manual; needs Sponza). `packages/ddgi/tsconfig.build.json` gets workspace paths (the d.ts build ran out of memory).
Verified (agent; headless Chrome 156, RTX 3060 and Intel UHD): three-neutrality and three-environment pass on both, and the spike-three pages still pass.
- Neutrality with Lambert materials: emissive Cornell full image and point Cornell GI only. Means within 1e-8, 0 pixels differ. The Meshlet and Visibility-Buffer pipelines equal the standard one (within 1e-10).
- Intensity toggle: 0 new render pipelines.
- three's background against our sky: worst 0.09%. Floor under the environment: GI plus the workaround reads 0.999× the environment alone; without the workaround, 1.98× (double-counted).
- lint, typecheck, test 745, build.
Numbers (CPU only; Sponza with curtains, 449 meshes, 600 frames): sync per frame, median still 0.2 ms (RTX session; means 0.19–0.21) and 0.2 ms (Intel session; 0.24–0.28); one object moving 0.1 / 0.2 ms; warm back-to-back 0.04–0.05 ms. Bistro interior (2,062 meshes): 1.2 ms per frame, over the 0.2 ms target. First sync on Sponza 220–300 ms. The browser timer steps 0.1 ms, so medians are coarse.
Not verified on a GPU: the module-version light swap (unit test only); sharing the slot with a real `LightProbeGrid`; spot and directional lights end to end; a real pmndrs/sky; `InstancedMesh`.
Fixes (52a7300):
- Change tracking does a fixed amount of work per frame: the scene is walked 128 objects a frame, picking up where it stopped; objects seen changing are checked every frame for 60 frames; 16 geometries and 16 materials are checked for edits per frame; lights every frame. `markChanged(object)` picks a change up in the next update.
- Every material is double-sided for GI unless `material.userData.giSingleSided` is set.
CPU per frame, means over 600 frames, still / one object moving (targets 0.2 / 0.5 ms):
- Sponza: RTX session 0.089 / 0.075 ms; Intel session 0.08–0.09 / 0.10 ms.
- Bistro interior (2,062 meshes): 0.15 / 0.17 ms; Intel 0.16–0.17 / 0.19 ms.
Detection latency is at most ceil(objects / 128) frames (Sponza 5, measured 2; Bistro 21, measured 18), or 1 frame with `markChanged`. A geometry edit can take about 130 frames on Bistro unless marked.
Review (Opus): lint, typecheck, build, test 749; three-neutrality, three-environment and the spike-three pages pass on both GPUs (neutrality reproduced: means within 1.3e-8). Sponza CPU reproduced (still 0.12, moving 0.13 ms), and so did the detection latencies. **Bistro interior still did not reproduce: 0.209 and 0.241 ms on two RTX runs, against the 0.15 claimed and the 0.2 target.** Being fixed:
- per-frame allocations and the Bistro cost;
- a re-parenting bug (an object moved mid-walk was destroyed while still in the scene);
- coverage when a material has its own `envMap`;
- documenting that camera layers aren't mirrored;
- the LightProbeGrid-shaped volume SPEC promises;
- the core listening for `uncapturederror` (SPEC);
- a docs/api/three.md page.
Fixes (a97e29a):
- per-frame allocations removed;
- the walk halved to 64 objects and 8 edit checks per frame. Latency of a static object's first unmarked move: 41 frames on Bistro (23 measured, about 0.7 s), 10 on Sponza (3 measured); 1 frame with `markChanged`;
- re-parenting fixed, with a failing-without test;
- coverage when a material has its own envMap;
- camera layers documented as not mirrored;
- texture-average colour folded into GI albedo and emission (until the M4 atlas);
- `GIVolume`, the LightProbeGrid-shaped volume (SPEC);
- `createGI` forwards `uncapturederror` (SPEC);
- `docs/api/three.md`.
RTX CPU, three runs, still / moving ms: Sponza 0.094/0.142, 0.063/0.103, 0.098/0.145; Bistro interior 0.117/0.179, 0.111/0.172, 0.153/0.181. The Intel-session runs (Sponza 0.259, Bistro 0.206) were invalid: another process held the RTX at 87–100% outside the lock. To re-time when idle.
lint, typecheck, test 779 and build pass; three-neutrality, three-environment and spike-three pass on both GPUs (agent).
Re-review (Opus): lint, typecheck, test 779 and build pass; three-neutrality, three-environment and spike-three pass on both GPUs (neutrality −3.4e-9 / 4.7e-9, 0 pixels differ). `GIVolume`, the `uncapturederror` listener and the re-parenting fix are correct. CPU not re-timed: the GPU never went idle (15–63% busy). Last round:
- fold texture metalness into GI albedo (Sponza's GI is near black without it, found in T1.8);
- weight the texture average by alpha, and average in linear space;
- a doc wording fix; a non-whole-spacing `GIVolume` test;
- `LightProbeGrid` layers for coverage;
- exports the viewer can use.
Last round (9d78f04): GI albedo = colour × average(albedo texel × (1 − metalness × metalnessMap.b)), texel by texel when both maps share a channel and uv transform (all 28 Sponza materials). Averages are taken in linear light and weighted by alpha; images over 256 px are first drawn at 256 by the browser, which filters in sRGB, as the comment now says. Sponza on the RTX (`three-sponza-gi`, manual): mean GI albedo 0.033 → 0.178; mean GI irradiance at bookmark 1 0.0185 → 0.042. Also: `LightProbeGrid` coverage only for grids on the camera's layers; exports for the viewer (`DDGI_DEBUG_COLORS`, `wrapGpuTexture`). lint, typecheck, test 786, build; three-neutrality and three-environment on both GPUs (agent).
**CPU target, re-timed by the lead** (RTX 3060, bench lock, 9d78f04, means over 600 frames; GPU at Dennis's usual ~18% background; SM about 1620–1770 MHz under load):
- Sponza with curtains (449 meshes): still 0.086 ms, one object moving 0.106 ms.
- Bistro interior (2,062 meshes): still 0.136 ms, moving 0.192 ms.
Targets ≤ 0.2 / ≤ 0.5 ms: met. Detection: Sponza 3 frames unmarked, Bistro 23, 1 with `markChanged`. The first sync is 0.9 s (Sponza) and 2.0 s (Bistro), including the texture averages. Not re-timed in an Intel-GPU session; the adapter's sync is CPU work.
Known gap, now handled: textured materials bounce with their base colour only (the core samples no textures until M4), so textured scenes read too bright. The adapter should fold in each texture's average colour.

## 2026-11-05 · T1.6 · implementer (opus); Opus review: pass with notes, fixed; re-timed at normal clocks; done
Did:
- DDGI's shading module (`techniques/ddgi/shading.ts`, `wgsl/shading.ts`): `giIrradiance` is `ddgiSampleIrradiance` unchanged; `giCoverage` (DECISIONS); `giDebugColor` with GI only, coverage and probes near this point. 3 textures, 2 samplers, 4 vec4f uniforms, `globalsCode` and `writeUniforms`. `version` 1. `provides.radiance` is false: a rough-only fallback is right only at roughness 1.
- The sampling loop moved into `ddgiAccumulateField` (bit-identical leak numbers). Per-sample hoisting plus `inverseSqrt` made it 16% cheaper, a rounding-level change; all T1.5 checks pass again on both GPUs (furnace −0.16% / −0.28%, Cornell +5.68% / +5.63%, two-room leak 0.797% / 0.795%).
- `examples/webgpu/hello`: a plain WebGPU forward renderer with a depth prepass, the point light with a cube shadow map, and `?scene`, `?camera`, `?view`, `?exposure`; plus `scripts/screenshot.ts`.
- Pages: `ddgi-shading-consistency`; `ddgi-shading-timing` (manual, with `to-csv.ts` → `bench/results/shading.csv`).
- Contract: a doc comment in `shading-module.ts` only (`globalsCode` includes `code`; binding order).
Verified (agent; RTX 3060 and Intel UHD, headless Chrome 156): every ddgi page passes on both. Raster against the ray-traced test view: means differ by +0.00014% / +0.00013%, and 1 pixel of 262,144 is off by more than 1%. The parameter path, as three would pass resources, equals the test view exactly. 99.9% of Cornell's surface pixels have coverage 1. lint, typecheck, test 668, `build:apps`, `build`.
Numbers (shading.csv, median of 3, under the lock): GI sampling at 2560×1440, RTX front view 0.62 ms, inside view 1.07 ms (bar 1.0 ms, judged on the inside view, DECISIONS); Intel 8.16 / 13.60 ms. Without a depth prepass the inside view cost 2.49 ms.
**Not valid as the final RTX number:** the RTX's SM clock was locked at 1200 MHz (max 2100) during every run. The lead confirmed it under load (82% use, still 1200 MHz) and asked Dennis to unlock it. Re-time after that.
Not verified: a compatibility-mode device; hello scenes other than the two Cornells; hello in a visible browser by hand.
Review (Opus): re-ran lint, typecheck, test 668, build and build:apps, and all 7 ddgi pages on both GPUs. Every number reproduces. The contract is met (no bindings in `code`, parameter order, `<texture>Sampler`, 3 textures, no storage buffers), and `giIrradiance` is the same function the trace kernel uses for bounce. Being fixed:
- coverage asserted, not just logged (fade band, past the margin, a point no probe sees);
- stronger debug-view checks;
- hello's `ready` to mean a settled field;
- `textureBindingViewDimension: '2d-array'` for compatibility mode;
- the back/forward cache;
- two comment nits.
The shading.csv rows carry `954d393-dirty` and a locked clock; they get replaced by the re-time.
Fixes (1864ecd, ceba56c): coverage asserted at four points (inside 1, fade band 0.5, past the margin 0, unseen 0); each debug view checked per channel against its own reference (at most 9 pixels differ, limit 1%); hello's `ready` = every probe updated 100 times; a back/forward-cache-restored page reloads.
Compatibility mode tested on a new page `ddgi-shading-compat`: raster GI equals the test view (0 pixels differ) on both GPUs. It needed `textureBindingViewDimension` on the atlases and on the sky placeholder, and `requiredLimits` asking for 256-thread workgroups (DECISIONS).
Next: re-time at normal clocks; spot-check today's earlier RTX timings; add the hello and shading pages to `MEASURED_PATHS`; a `shading.csv` note in `bench/results/README.md`.

## 2026-11-05 · T1.9 prep: native matched-count timings · native-runner (opus)
Did: `tools/native/bench.ps1` gains opt-in `-GridRule ddgi` (our probe grid), `-Runs` and `-Label`; the defaults are unchanged. Ran cornell-box, two-room and sponza (base + curtains) 3 times each: Vulkan, RTX 3060 Laptop, 256 rays, every probe updated every frame. 405 rows in native.csv (`label,t1.9-matched`, `grid_ddgi,1`).
Verified: every run updated every probe (1023 of 1024 frames). The spread over 3 runs was under 1% this session, with the runs interleaved in one lock hold. The GPU was idle at the start; the python process held a core.
Numbers (median of 3, ms; min–max below 1%):

| | cornell-box | two-room | sponza |
| --- | --- | --- | --- |
| Grid | 12×12×12 = 1,728 | 17×7×9 = 1,071 | 37×21×25 = 19,425 |
| Active / off | 1,037 / 691 | 844 / 227 | 7,947 / 11,478 |
| Trace | 1.055 | 0.825 | 9.845 |
| Blend | 0.429 | 0.338 | 3.482 |
| Relocation | 0.022 | 0.020 | 0.034 |
| Classification | 0.031 | 0.031 | 0.046 |
| DDGI total | 4.165 | 3.936 | 16.198 |
| Rays traced per frame | 287,584 | 223,328 | 2,401,728 |
| Mrays/s | 273 | 271 | 244 |

"DDGI total" includes about 2.7 ms of native's screen-space gather at 1920×1080.

Native settings that differ from ours (for T1.9 to explain differences; nothing changed):

| Setting | Native | Ours |
| --- | --- | --- |
| View bias | 0.3 × spacing | 0.1 × spacing |
| Min front-face distance | 0.3 × spacing | 0.02 × spacing |
| Big-change rule | on, 0.2 | off |
| Brightness clamp | 1.0 | 2.0 |
| Albedo cap on probe hits | 0.9 | 1.0 |
| Irradiance storage | R10G10B10A2, gamma 5 | rgba16float |
| Classification reach | own cell | 1.5 spacings |
| Max ray distance | 10,000 | unlimited |
| Probes per frame | all | budgeted (max 4,096) |

Hysteresis 0.97, normal bias 0.1 × spacing, texel counts 6 and 14 and the distance clamp match.
Not verified: image comparison; D3D12.
Next (T1.9): compare per probe updated, or raise our `maxProbesPerFrame` to cover Sponza's 19,425 probes for a whole-field comparison.

## 2026-11-05 · T1.5b · implementer (opus); Opus review: pass with notes, fixed; done
Did (9c6348a):
- The single kernel is replaced by `ddgi/trace` (64 threads, one ray each, inline `giTraceClosest`, a ray buffer that grows when needed), then `ddgi/blend` (phases B and C), then `ddgi/write`.
- Phase B: subgroups first, with a declared workgroup fallback. Relocation and classification per the design, with the tuning in DECISIONS. Offsets are used for ray origins and in sampling; off and not-yet-updated probes are skipped when sampling.
- The `points` debug view, and a GPU state census for the counters (read back a few frames late, like timings).
- The ms budget converted into probes in gi-platform.ts (`UnitCostMeter`).
- T1.5a review notes: named uniform fields and a `DdgiVolume` struct, with a layout test; render.ts gates on the manifest volume; a GPU test page for the stochastic rounding and the border rule.
Verified (agent; RTX 3060 and Intel UHD, headless Chrome 156, default limits), with the lagged big-change rule on:
- Furnace mean −0.15% / −0.28% (worst point 3.7% / 3.8%).
- Cornell on the manifest volume +4.09% / +4.03% (gate ±10%). In air +2.80% / +2.73%. RTX detail: 8×8-block FLIP 0.190, block mean error 6.7%, p95 block error 94% (the wall-slab front edges, where Cycles is about 0). Emissive Cornell full image +5.1%, FLIP 0.280.
- Two-room leak 0.783% / 0.781% (≤ 1%). Thin wall 0.48% / 0.48%; closed door 0.035% / 0.036%.
- Classification: everything the geometry demands; subgroups and the fallback gave identical states and offsets for all 5,256 probes.
- lint, typecheck, test 654.
Numbers (median of 3, under the lock, 512 probes; µs per probe, the probes 2 ms buys, and what a `gpuMs: 2` budget gave):
- RTX, Cornell: 2.31 µs; 866 probes; 932 probes in 1.999 ms.
- RTX, two-room: 3.90 µs; 513 probes; 514 probes in 1.997 ms.
- Intel, Cornell: 13.5 µs; 147 probes; 133 probes in 1.94 ms.
- Intel, two-room: 17.4 µs; 115 probes; 93 probes in 1.98 ms.
- After the split, Intel spends about half its update time in `ddgi/blend`.
Not verified: probes coming back on in a changing scene (M3); `resetHistory` and a second `prepare` on the GPU; dispatches past 65,535 probes; Sponza.
With the big-change rule off by default (c7fbbea; DECISIONS), re-run on both GPUs (RTX / Intel):
- furnace −0.16% / −0.28% (worst point 3.7% / 3.8%);
- Cornell on the manifest volume +5.67% / +5.61% (gate ±10%); in air +4.33% / +4.26%;
- two-room leak 0.800% / 0.797% (≤ 1%); thin wall 0.48%; closed door 0.039% / 0.040%;
- classification passes, subgroups equal to the fallback;
- test 655.
Lead: `ddgi-leak` re-run on the RTX passes. Timing wasn't re-run, since the rule doesn't change the per-frame work.
Review (Opus): every number reproduced on both GPUs; all 6 DDGI pages pass; subgroups equal to the fallback, with Intel exercising the cross-subgroup combine; the leak page's Cycles means match the EXRs. Being fixed:
- the state census read back every frame against the no-readback rule;
- the budget meter divided by units handed out instead of units used;
- the cabinet row depends on the GPU (79 probes off inside a solid on the RTX, 39 on Intel; one probe still in state 3 after 60 frames, which suggests it flips);
- two WGSL clarity items.
Fixes:
- The census runs only with `gpuTiming`, and its pass is timed; without it the per-state counters are absent.
- `encode` may return the units it used (technique-facing contract change, platform.md Revisions), and the budget meter divides by those. It resets on `prepare`, technique add/remove and `resetHistory`. `gpuMs: 2` gives 1.996 ms, with a new over-budget test.
- The cabinet row's GPU difference came from probes that never settled: they crept by tiny relocation moves, or hopped between overlapping solids. Moves under ¼ `minFrontFaceDistance` are now dropped, and a probe already off as inside doesn't hop again. A new settle check (120 frames, then a 20-frame window) fails any probe that keeps changing.
- A third change ("a move away from a face must head away from it") raised the leaks (two-room 0.80 → 0.967%). The agent measured each change on its own and dropped that one; the paper's direction rule is back. Moving straight away from the face instead gave two-room 1.49%.
- `DDGI_KEY_INDEX_BITS` derived from the fixed-ray count; the reliance on zero-initialised workgroup memory commented.
Final (agent, both GPUs; RTX / Intel):
- furnace −0.16% / −0.28%;
- Cornell on the manifest volume +5.68% / +5.63% (in air +4.33% / +4.26%);
- two-room leak 0.797% / 0.795%; thin wall 0.589% / 0.585%; closed door 0.062% / 0.063%;
- cabinet row: 251 / 250 probes off inside; 0 unsettled, 0 left in state 3, 0 subgroup mismatches.
Three leak runs per GPU gave identical numbers: rays are seeded by the frame number, so runs are deterministic. lint, typecheck, test 659. Lead re-ran ddgi-classification and ddgi-leak on the Intel UHD: both pass.
Known issue: the Intel UHD hangs (6 of 6 tries) when ddgi-distance-format runs its second device straight after ddgi-classification. It also hangs with 9c6348a's kernels, so it isn't this round's kernel changes, and the cause isn't found. To reproduce: put distance-format back to running device A before device B, then `node tools/gpu/run.ts --gpu intel --filter ddgi-classification --filter ddgi-distance-format`. Worked around by running the fallback device first. To look at in T1.9 (device release between pages, NOTES).
Worth knowing before the targets tighten: as DECISIONS defines it, the leak measure can't tell a leak from overall brightness. DDGI reads +5.7% (lit room) and +9.0% (dark room) over Cycles, and that general excess accounts for most of the 0.80%.

## 2026-11-05 · T1.5a · implementer (opus); Opus review: pass with notes; fixes folded into T1.5b
Did (e18ba01): `DDGITechnique` from `docs/design/ddgi.md`, in `packages/ddgi-core/src/techniques/ddgi/`:
- the volume and three `texture_2d_array` atlases with octahedral borders; distances in `rg16float`, falling back to `rgba16float`;
- the `ddgi/update` kernel (phases A and C, with a slot for phase B) and the `ddgi/write` kernel;
- the scrambled update order with a CPU cursor, the per-frame rotation, hysteresis with the change threshold and firefly clamp, state 3;
- one `ddgiSampleIrradiance`, plus counters, timer scopes and `memoryBytes`;
- pages ddgi-furnace, -test-view, -distance-format, -timing, -irradiance-check.
Three measured deviations from the design: stochastic rounding into halves (DECISIONS); decoded ray directions kept in workgroup memory (+21% speed); the direction weight taken from the unbiased surface point, as in the 2019 paper (about 1% better on Cornell walls).
Verified (agent; RTX 3060 ampere and Intel UHD gen-12lp, headless Chrome 156, default limits):
- Furnace: 600 frames, all 693 probes each frame; mean E/π 0.4968 (−0.64%) on the RTX and 0.4967 (−0.66%) on Intel, inside 2%. Worst point −6.2% / −6.4%, on the floor beside the pillar, where a probe sits on the pillar's face (T1.5b's relocation).
- Cornell (point light) indirect against Cycles, mean: +9.50% RTX / +9.43% Intel with every probe in the air (passes ±10%); +11.4% on the manifest's volume, with probes buried in the floor and ceiling (DECISIONS: T1.5b must pass it). In-air volume: FLIP 0.219, mean error 10.2%, p95 8×8 block error 102% (the wall-slab front edges, where Cycles is about 0).
- Emissive Cornell, full image (not gating): +3.8%, FLIP 0.279.
- The two distance formats give identical fields on both GPUs.
- lint, typecheck, test 641.
Numbers (median of 3, under the lock, 256 rays; µs per probe, update + write):
- Cornell: RTX 4.85 + 0.014 (411 probes in 2 ms); Intel 23.2 + 0.12 (85).
- Two-room: RTX 6.23 + 0.014 (320); Intel 25.7 + 0.14 (77).
- One kernel against the split, at 512 probes: RTX 2.48 against 1.22 ms; Intel 11.6 against 7.8 ms (DECISIONS: the split is adopted).
Not verified: `msPerUnit`, `resetHistory` and a second `prepare` on a GPU; dispatches over 65,535 probes; other ray counts; Sponza timing.
Corrections from the review (Opus; clean worktree at e18ba01, every number reproduced on both GPUs):
- "FLIP 0.219" is the 8×8-block FLIP; per pixel it is 0.232, with p95 pixel error 158%.
- The test count at e18ba01 is 637 (631 passed, 6 skipped); "641" included B1a's uncommitted tests.
- The in-air volume also changes the y spacing from 0.5 to 0.49 m, which shifts the auto biases and the distance clamp by 2%.
- **Main finding:** most of Cornell's excess came from the per-texel change-threshold rule, not probe placement. With the rule off: in-air +4.38% / +4.31%, FLIP 0.169; manifest volume +6.73% (RTX). The rule moves to the probe's mean (design doc, DECISIONS correction). Also being fixed in T1.5b: named uniform fields instead of `.w`/`.z` indices, consistent defaults in `render.ts`, and unit tests for the stochastic half rounding and the WGSL border rule.
Next (T1.5b): the trace/blend split; classification and relocation; ms-to-units budget conversion (the default of 512 probes is over 2 ms); bias tuning against the leak checks; Cornell on the manifest volume.

## 2026-11-05 · B1a · implementer (opus); Opus review and re-review passed with notes, all fixed; done
Did (see the B1a commit after b99773f):
- packages/bvh: `gpu-refit.ts`, a bottom-up refit with per-node arrival counters, boxes copied as 32-bit atomic words, no float atomics, and a subgroup pairing step for sibling leaves; `gpu-instance-box.ts`; `refit-cpu.ts`, the CPU reference, with a test.
- Core: `top-level-gpu.ts` (one pass gathers each instance's transform from the scene copy and writes its record and world box; then the refit). Shape builds in their own worker (`top-level-shape/async/worker.ts`). `ray-queries.ts` chooses the GPU refit, or the CPU top level without subgroups.
- `setTransform` never triggers a CPU rebuild. The traversal WGSL is unchanged.
- Pages `tlas-oracle` and `tlas-bench`. trace-oracle and trace-inline now request subgroups, so they test the GPU top level.
Verified (agent, headless Chrome 156): tlas-oracle passes on the RTX 3060 and the Intel UHD, with 32 frames of moving instances (mirrored, non-uniformly scaled, hidden, singular, created, destroyed). Per path: 192,000 rays, 0 mismatches. The GPU path and the no-subgroups fallback disagree on 0 rays, and no refit was incomplete. trace-inline and trace-oracle pass on both GPUs. lint, typecheck, test (637) and build pass.
Numbers (`bench/results/bvh.csv`, median of 3, bench lock, idle GPU; the rows say `b99773f-dirty` because the code was uncommitted): refit pass for 10k moving instances, RTX 0.084 ms (bar 0.5), Intel 0.437 ms (no bar); without the subgroup pairing 0.0855 / 0.476 ms. Furnished room on the RTX: 157.3 / 513.1 Mrays/s closest / any (T1.3 157.3 / 513.2). CPU: tracer update 0.4 ms (it scans for new instances), scene upload 1.2 ms, 10k `setTransform` calls 18–20 ms (the scene copy's own cost), shape build 54–57 ms in the worker.
Not verified: Intel closest-hit in the quick bench (too noisy).
Review (Opus): re-ran lint, typecheck, test, build, and tlas-oracle, trace-inline and trace-oracle on both GPUs (0 mismatches; 4 quality rebuilds each); bvh.csv matches (0.0837 / 0.4372 ms). The design holds: one writer per box slot, the clear in an earlier dispatch, waits only on siblings that have arrived. Being fixed:
- a refit that gives up was only counted, and could keep a box shrunk by NaN;
- the refit pass wasn't connected to `stats.passes` at all;
- the oracle never ray-checked more than 64 instances, so the cross-workgroup path was untested;
- a failed tree placement could leave records pointing at released ranges;
- the instance scan and snapshot every moving frame (a scene flag is being added);
- the refit's stats readback ran always, against the platform's "no readbacks" rule;
- a rebuild costs 3.0–3.7 ms of main-thread time to snapshot, not "well under a millisecond".
Fixes (4f85332):
- a repair pass on the GPU for refits that give up;
- the quality check in the worker (DECISIONS);
- the stats readback only with `gpuTiming`, with a test that there's no `mapAsync` by default;
- an `instanceSetChanged` scene flag, which cut the tracer's CPU update from 0.4 to 0.1 ms per frame;
- the mesh table written even when placing trees fails;
- the timer wired, so the refit shows in `stats.passes`;
- transform packing as a typed array: about 1.3 ms per check for 10k instances (2.6–4.4 ms cold), down from 3.0–3.7.
Re-review (Opus): lint, typecheck, test (641); tlas-oracle on the RTX 3060 and Intel UHD:
- the scripted case: 192,000 rays, 0 mismatches;
- 2,000 moving instances: 32,000 rays, 0 mismatches;
- the fallback agrees on every ray;
- a forced give-up (7 reads on the RTX, 10 on Intel) is repaired to match the CPU refit bit for bit.
The repair pass can't race the main refit: it reads a flag written by earlier dispatches. Refit re-timed (median of 3, under the lock): RTX 0.086 ms, Intel 0.43 ms.
Last fixes:
- a failed `swap` keeps the old shape, drops the arrival, throws once, and pauses quality rebuilds until instances or meshes change; the worker judges shapes by the generation the GPU uses (tests fail without the fix);
- the give-up warning is deduped inside `GpuTopLevel`.
The agent re-ran lint, typecheck, test (645) and tlas-oracle on both GPUs.
A refit that has to repair, forced, at 10k instances: RTX 8.43 ms, Intel 4.68 ms (median of 3). Most of it is the spin before giving up. Give-ups have never happened outside the forced test on either GPU, so this is a rare-spike cost, not a frame cost.


## 2026-11-05 · T1.4 · implementer (opus); Opus review: pass with notes, fixed; done
Did (72ebc0e):
- `platform/shade`: hit shading (`giSurfaceAt`, `giDirectIrradiance`, `giSkyRadiance`, `giSkyIrradiance`, `giSkyIrradianceAt`). Point and spot lights use three r187's falloff and cones. Shadow rays use `giTraceAny`, and lights beyond the cap are picked stochastically and rescaled. It binds 0 storage buffers.
- `platform/sky`: the environment projected by compute into SH (bands 0, 1, 2 and 4) plus a 256×128 radiance map; subgroups first, with a workgroup fallback; re-projected only on change.
- The Preetham and calibration code moved from `scenes/src/sky` into the core; the sky files are byte-identical.
- The contract states the equirect layout. One turbidity default, 2.
Verified (agent; RTX 3060 ampere and Intel UHD gen-12lp, headless Chrome 156, default limits):
- Sun on a plane matches E·cos θ, worst 7e-8.
- Open-ground sky irradiance against the sky file's own integral (what the page measures): +0.30 / −0.11 / −0.25%. Against Cycles, worked out through T1.1's 0.03% agreement between the file and Cycles: +0.33 / −0.09 / −0.25%. Both inside ±0.5%. L2 SH alone would be +0.74 / −0.33 / −0.87%.
- Shadow rays against the oracle on 9 scenes: 0 disagreements, 0 leaks.
- Point and spot lights against three's formulas: worst 1e-5.
- The stochastic choice: cap 2 within 0.21% (1.6 SE) of all lights.
- Subgroups against the fallback: 1.4e-7.
- Scene copy plus hit shading: 2 storage buffers.
- lint, typecheck, test 597.
Not verified: cube maps made by three; an environment change in the viewer; the sky's ~1.3 MB isn't in `stats.memory` yet.
Review (Opus): re-ran lint, typecheck, test, `generate.ts --check`, and every hit-shading page plus sky-orientation on both GPUs; the numbers match. hit-shading-sky was run in a clean worktree at 72ebc0e, because B1a's uncommitted shader broke it in the live tree. The reviewer confirmed the band-4 constants, the solid angles, three's falloff and cone formulas, and the unbiased rescale. Fixed:
- every `setEnvironment` bumps a counter, and the sky re-projects on the counter, so the same object passed again after its texture changed now re-projects (new GPU test);
- the contract states the cube convention, and that three r187 looks cubes up at (−x, y, z), so T1.7 must mirror them (test: a three-style cube reads back mirrored, worst 7e-4).
hit-shading-sky passes on both GPUs on the live tree (agent). Dennis chose a new `sunIlluminance` field (DECISIONS 2026-11-05), now built: sky radiance = calibrated sky × `sunIlluminance` × `intensity`. A unit test shows the platform's sky equals each of the 6 sky files half for half. hit-shading-sky passes on both GPUs: doubling either field doubles the radiance and irradiance within 2e-7, and open ground stays +0.30 / −0.11 / −0.25% (agent). Lead re-ran the sky and scene tests and `generate.ts --check`.
Next: `stats.memory` should include hit shading's ~1.3 MB; T1.7 handles three's light decay, `environmentRotation` and cube maps.
Lead mistake, fixed: 43fab16 (T1.3's fixes) also caught T1.4's in-progress `gi-platform.ts` wiring, because the agent edited it between the lead's diff check and the commit. So `origin/m1` at 43fab16 didn't build until 72ebc0e. Rule since: no commit of a file that two active agents may touch.

## 2026-11-05 · T1.3 · implementer (opus); Opus review: pass with notes, fixed; done
Did (168e632; timing rows d6ee376):
- The platform object: `createGI`, `requiredLimits`, `optionalFeatures`, `isSupported`, a `KernelBuilder`, the technique lifecycle, device loss, `dispose`, `stats.memory`.
- `RayQueries` over the scene copy: tinybvh trees built in the worker, a CPU top level, `GiHit.instance` = the scene-copy slot, `GILimitError` past one binding.
- Our CPU mesh builder is deleted (tinybvh only; `sah-builder` stays for the top level until B1a).
- Front faces follow the authored side, also on mirrored instances, with a new oracle case `mirrored-box`.
- Any-hit is its own walk, farther child first (DECISIONS).
- Oracle helpers are out of the public exports; layout sizes come from `@pmndrs/bvh/gpu-layout`.
- The runner records the commit at run start.
- Bench: a `platform` tracer and a `sponza-base` scene; `ours-wt` removed.
- Converting a ms budget into units is a stub for T1.5 (`budget.units`, default 512). Hit shading is a placeholder until T1.4.
Verified (agent): lint, typecheck, test (580), build, with no helper names in `dist`. trace-oracle (variants platform, tinybvh, tmbuild; 0 slipped of 2 M each; mirrored-box 0 mismatches) and scene-copy pass on the RTX 3060 (ampere) and Intel UHD (gen-12lp), headless Chrome 156, default limits. Lead: no measured code changed after the timing runs started, so the rows carry 168e632 (set by hand with `to-csv --commit`, overriding the new run-start record, because the runs were on the uncommitted tree).
Review (Opus): re-ran lint, typecheck, test, build, the oracle and scene-copy on both GPUs; the CSV medians match. The oracle page builds the platform object directly (`GIPlatformImpl`) rather than through `createGI` and a technique, so the inline path DDGI will use isn't on a GPU yet. That's being added.
Review fixes:
- **New GPU page `trace-inline`:** `createGI` → scene API → a test technique whose kernel (tracer 3 + scene copy 2 + own 3 = 8 of 8 storage buffers) calls `giTraceClosest` and `giTraceAny` inline, checked against the CPU reference. On both GPUs: 87,078 oracle rays, 0 mismatches, 0 escapes, 0 front-face disagreements, 0 of 400,000 shared-edge rays slipped. Ambiguous closest hits (3,439 RTX / 3,456 Intel) come only from the thin-box case, as on the oracle page. A 9th storage buffer now throws `GILimitError`. The lead re-ran it on the RTX: PASS.
- `KernelBuilder` counts the technique's own storage buffers (`ownGroups`; platform.md Revisions).
- The public JS exports are trimmed to the errors, `GpuTimer` and its helpers, `createGI`, `requiredLimits`, `optionalFeatures`, `isSupported` and the hit-record constants.
- The contract's `GiHitRecord` field order now matches the WGSL, with a layout test.
- Smaller fixes: the docs on limits and the design doc; the timer when `encode` throws; tests for `dispose`, `requiredLimits`, `isSupported` and `optionalFeatures`.
After the fixes: lint, typecheck, test (586) and build pass; trace-oracle, scene-copy and trace-bench-quick pass on both GPUs (agent).
Not verified: Bistro exterior's `GILimitError` on a real GPU (fake-GPU unit test only); `stats.passes`; real device loss.
Numbers (median of 3, idle GPU, under the lock; Mrays/s closest / any, tinybvh trees; B0b in brackets). The `platform` tracer is within 0.3% of these on the RTX; on Intel its any-hit reads up to 2.6% higher (sponza-base 19.82 against 19.31):
- RTX: sponza-base 107.3 / 168.3 (106.4 / 137.6); furnished room 157.3 / 513.2 (156.1 / 185.2); Bistro interior 55.6 / 66.9 (55.0 / 44.1); Bistro exterior 59.7 / 48.1 (59.2 / 43.9).
- Intel: 13.1 / 19.3 (13.6 / 16.4); 23.3 / 63.8 (23.0 / 28.0); 7.5 / 9.2 (7.4 / 5.8); 8.5 / 6.6 (8.7 / 6.0).
- Any-hit against three-mesh-bvh: RTX 1.22 / 1.03 / 0.93 / 1.05; Intel 1.91 / 1.66 / 1.44 / 1.69. Closest-hit within 4% of B0b, so the "within 10% of T0.10" check holds.
- Any-hit study (RTX), furnished room / Bistro interior: left child first 256 / 49.1; right first 413 / 58.6; farther first 512 / 66.8.
- Per shadow ray: furnished room 14 nodes, 1.3 instances, 3.7 triangles (97% blocked); Bistro interior 44 / 5.0 / 41 (50% blocked).
- Memory, sponza-base: triangle records 179.9 MB. Corners read from the scene copy: closest −12%, any −10% on Sponza; −24 to −30% on the furnished room and Bistro interior. A 16-byte record with vertex indices: −7 to −21%. Both GPUs allow `maxStorageBufferBindingSize` 2,147,483,644 and `maxBufferSize` 2,147,483,648.

## 2026-11-05 · T1.10 part 2 · implementer (opus); Opus review: pass with notes, fixed by the lead; done
Did (a5a1113):
- `tools/assets/fetch.ts` downloads from `cdrdv2.intel.com` with `%SystemRoot%\System32\curl.exe` (`--disable --location --fail`, no header or user-agent flags, arguments as an array with no shell). Other hosts keep Node fetch.
- `sponza` is now the base package plus the curtains. New `sponza-full`: base + curtains + ivy + trees, tagged `heavy`, `nativeCapable: false`, candles left out until Dennis decides.
- `isHeavyScene` keeps heavy scenes out of "every scene" runs unless named. The viewer's loading text names each part. `tools/bench/scene-copy-memory.ts` sums every part.
Lead fixes after review: `--disable` added to the curl call, so the user's `.curlrc` can't add headers. The Cycles metadata now records the scene's parts, its environment and the sky file's sha256. THIRD_PARTY's "Use" column and two NOTES bullets corrected.
Verified:
- Reviewer: `--verify-only` passes for all five Intel ids; 283 scene and asset tests pass; `generate.ts --check` OK; every add-on lies inside Sponza's volume; `sponza.png` shows the curtains at bookmark 1; nothing from the packages is committed.
- Lead: looked at `bench/screenshots/t1.10/sponza.png` (curtains in the arches); fetch tests 104 pass after the curl change.
- Agent only: a real forced download of the curtains from Intel through the tool (786,898,766 B, about 25 MB/s, 48 s including the Blender conversion), then `--verify-only` OK. Viewer on the RTX 3060, headless Chrome 156, 0 console errors.
- Cycles reference re-rendered at bookmark 1: 4096 spp, 640×360, diffuse-only check passed (49 BSDFs reset, 86 links removed).
Not verified: the Intel GPU in the viewer. The reviewer didn't reproduce the Cycles means.
Numbers:
- Triangles: base 3,747,018; `sponza` (with curtains) 5,744,384; `sponza-full` 12,833,689.
- Viewer load / set-up, one warm run each: base 2.0 / 5.6 s; `sponza` 2.5 / 7.4 s; `sponza-full` 4.9 / 18.9 s.
- Scene copy: `sponza` 130.1 MB (97% of the default binding); `sponza-full` 348.5 MB.
- Cycles: render 1,107 s. Full mean 0.0151 (RGB 0.01778 / 0.01480 / 0.01272); indirect mean 0.00107. Before (no curtains, shared sky, 256 spp): 0.0174 / 0.00218.
Watch out: **`sponza` rows from a5a1113 on include the curtains.** Earlier `trace.csv`, baseline and B0 rows are base-only; T1.3 adds a `sponza-base` bench scene for like-for-like comparisons.

## 2026-11-05 · T1.9 prep: native multi-part scenes · native-runner (opus)
Did: `tools/native/merge-gltf.mjs` merges a scene's model parts into one glTF in the native cache, with no fork change. It applies each part's transform through one wrapper node and points at the parts' own `.bin` and image files. `native-lib.ps1` sends multi-part scenes down this path. `bench.ps1` writes a `scene_parts` row.
Verified (agent, RTX 3060 Laptop, Vulkan, fork c7d0162):
- `play.ps1 -Scene sponza -Clean` shows the curtains at bookmark 1 (`bench/screenshots/native-sponza/`, gitignored).
- 25 generated inis are unchanged for the single-part scenes, and the cornell-box cache has the same sha256.
- One `bench.ps1 -Scene sponza` run: 43 rows.
Not verified: furnished-room in the harness; D3D12; repeat runs.
Numbers (1 run, so ±15–30% drift; Sponza with curtains, 17,760 probes, 256 rays, 6,454 active): trace 8.34, blend 2.94, lighting 2.53, DDGI total 14.00, gbuffer 5.61, frame 19.76 ms; 242 Mrays/s. CPU frame 150 ms (machine busy). The base-only T0.6 run had trace 7.94 and gbuffer 4.68.
Next: T1.9 takes 3+ runs.

## 2026-11-04 · B0b · implementer (opus); Opus review: pass with notes, fixed; done
Did: tinybvh's WASM writes our GPU layout directly (`tbvh_write_layout`, `buildLayout` in packages/bvh/src/tinybvh.ts, sizes in `gpu-layout.ts`), commit 4340e4e.
- Leaves are packed and boxed during the walk, then inner boxes joined in a backward pass. Boxes are recomputed from the triangles with WASM SIMD.
- Vertex widening and the index check moved into the WASM.
- The JS conversion is off the tinybvh path.
Verified (the reviewer re-ran these in a clean worktree at b10ad3f):
- `mesh-bvh-tinybvh-layout.test.ts` compares the WASM layout byte for byte with the JS conversion.
- `bvh:wasm --check` reproduces the WASM (SHA 656b99…); `verify-artifacts` passes.
- The oracle passes on the RTX 3060 (ampere) and the Intel UHD (gen-12lp), with 0 slipped rays on tinybvh trees.
Per-mesh identity: `node tools/bvh/layout-identity.ts` (lead re-ran it, 12 s) prints every mesh identical: `sponza` (now with curtains) 132 of 132, Bistro exterior 1,296 of 1,296, Bistro interior 1,186 of 1,186. Three deliberate C++ faults were each caught (agent).
Timing at 8e5373c (b10ad3f), idle RTX, all under the bench lock, 3 runs per GPU:
- Sponza in the browser worker: 1.89 / 2.27 / 1.70 s (RTX session) and 1.53 / 1.61 / 1.78 s (Intel session). That passes on the median (DECISIONS); 1 run in 6 went over 2 s. The stray python.exe held one core throughout (`note/cpu_core_busy`), and Windows Defender briefly took 2 cores twice.
- Bistro exterior 1.42 / 1.34 s; Bistro interior 0.50 / 0.56 s.
- In Node: 1.71 s against 2.23 s with the JS conversion (CPU busy with other agents).
- tinybvh trees against ours on the RTX: 0.99–1.10×.
- Against three-mesh-bvh (tinybvh trees), RTX closest / any: Sponza 1.22 / 1.00, Bistro exterior 0.83 / 0.96, Bistro interior 1.03 / 0.61, furnished room 1.04 / 0.37. Intel: Sponza 2.02 / 1.57, Bistro exterior 1.30 / 1.52, Bistro interior 1.57 / 0.91, furnished room 1.68 / 0.72.
- The rows were written with `to-csv --commit 8e5373c` from the 24 run JSONs, because docs-only commits landed mid-run and `repeat.ts` refused. The measured code paths are unchanged between 4340e4e and 8e5373c (the reviewer checked). T1.3's uncommitted trace files all date from after the runs, which fits a clean run but doesn't prove it. T1.10 briefly had `scenes/src/manifest.ts` edited mid-run and reverted it, outside the measured paths.
Review fixes:
- buffer sizes are computed in 64 bits, with a clear `STATUS_TOO_LARGE` error at 4 GB on wasm32;
- `static_assert`s on the node, triangle and vertex structs;
- NaN test cases;
- timing numbers removed from the README and comments;
- THIRD_PARTY lists the runtime calls the wrapper makes;
- the per-mesh script.
After the fixes, `bvh:wasm --check` matches (SHA ae5e0cf4…) and the oracle passes on both GPUs (agent).
Next: B0 is passed, so T1.3 deletes our CPU mesh builder. B1a (GPU top level) follows T1.3.

## 2026-11-04 · T1.1 · implementer (opus); Opus review failed, fixed, re-review passed; done
Did (e98d543, fixes 8e5373c):
- One sky file per manifest: `node tools/sky/generate.ts` writes `scenes/sky/<id>.exr` (half-float, about 50 kB, committed) plus a JSON sidecar. The model is Preetham 1999 from the paper, with no sun disc. The brightness uses the Ineichen–Perez clear-sky ratio (DECISIONS).
- Readers of the file: the viewer's sky dome (surfaces still get no environment light); Cycles' Environment Texture; native's flat equivalent (DECISIONS); and an engine-free platform loader (`scenes/src/sky/sky-texture.ts`, rgba16float).
- New `open-ground` scene. GPU page `sky-orientation`.
Verified (both reviews re-ran these):
- `generate.ts --check`: 5 skies, calibration error 0.000%. lint, typecheck and test pass (557).
- `sky-orientation`, 8 directions, worst error: RTX 3060 0.021%, Intel UHD 0.008% (headless Chrome 156). It uses the viewer's own loader and sky dome.
- Cycles `--calibrate`: 0.00–0.05%, including the sky on a floor (E/π).
- Open ground, bookmark 1, Cycles 1024 spp against the analytic answer: R −0.03%, G −0.01%, B −0.01%.
- Sponza bookmark 1, probe-grid technique: no longer blue (`bench/screenshots/t1.1-before|after/sponza.png`, lead looked). But the shadows are near black at 0 EV, and no number checks the probe grid's sky light against Cycles. The exposure question went to Dennis.
Not met as first written: the viewer leg. three's PMREM environment diffuse, which the viewer doesn't use, measured against Cycles:
- diffuse-only material: bias −0.65% (limit ±0.5%), mean relative error 0.650%, p95 8×8 block error 0.530%; per channel against analytic R −0.04%, G −0.61%, B −0.97%;
- standard material: +1.05%, mean relative error 1.046%, p95 block error 1.442%.
The check was redefined (DECISIONS); sky-lighting accuracy is now T1.4's.
Not verified: a native harness run (only the generated ini was checked); the viewer baselines weren't re-measured with the new sky.
Numbers:
- Sky-to-sun ratio on the ground (diffuse fraction of all daylight): sponza 0.123 (11.0%), courtyard-tree 0.132 (11.6%), bistro-exterior 0.137 (12.0%), furnished-room 0.152 (13.2%), open-ground 0.167 (14.3%).
- Sponza Cycles bookmark 1 means: old Nishita sky 0.0346 (indirect 0.00491); shared sky, 256 spp, 0.0174 (indirect 0.00218). The sun glow is 3.08× the opposite sky.
Next: T1.4 (the platform's sky irradiance within ±0.5%, the procedural type through the same code, the equirect layout in the contract); T1.10 re-renders Sponza's reference with the curtains.

## 2026-11-04 · T1.2 · implementer (opus); Opus review: pass with notes, fixed; done
Did: `PlatformScene` (SceneApi and SceneCopy) in `packages/ddgi-core/src/platform/scene/`, commit 73f99e6.
- Two storage buffers: vertex data, and one table buffer (header, instance, mesh, material and light records, per-instance material lists).
- Uploads with `queue.writeBuffer` inside `encode(encoder)`. Tables double and reallocate, and `module.version` bumps.
- Octahedral normals and 16-bit indices (DECISIONS). Also `SceneChanges`, bounds, memory and counts.
- GPU page `tools/gpu/pages/scene-copy`. Contract fix by the lead: `updateLight` takes `LightChanges` (acc4d32).
Verified (reviewer re-ran everything): lint, typecheck, test 545 pass. `test:gpu --filter scene-copy` passes on the RTX 3060 (ampere) and Intel UHD (gen-12lp), headless Chrome 156, default-limits device:
- every accessor matches the inputs in 3 states;
- 1 shader module and 1 pipeline before and after create/destroy churn;
- a DDGI-shaped kernel at 8 of 8 storage buffers: 0 of 4096 rays disagree with the CPU walk.
Not verified: GPU timing of uploads. Packing Sponza takes 698 ms on the CPU, in Node only.
Numbers (vertex buffer): Sponza 87.7 MB (104.1 with float normals, 110.2 with 32-bit indices); Bistro exterior 111.4 MB (134.8); Bistro interior 32.3 MB. Tables 0.1–0.8 MB.
Review fixes (lead re-checked):
- the scene internals are gone from the public build (none of their names appear in `packages/ddgi/dist`);
- the contract comment on normals is corrected;
- `node tools/bench/scene-copy-memory.ts` reproduces the numbers below;
- `testing/fixture.test.ts` plants 10 faults and the checker catches each one;
- `GI_UV_WRAP_*` is now generated from `UV_WRAP`.
After the fixes: lint, typecheck and test pass (556); the agent re-ran `scene-copy` on both GPUs.
Next (T1.3): `GiHit.instance` = scene-copy slot; act on `SceneUpload` (`meshesRemoved`, then `meshesUploaded`, then `instancesChanged`); positions are stored twice (tracer 180 MB plus scene copy 87.7 MB on Sponza), so measure whether the tracer can read the scene copy's positions; mirrored front faces (DECISIONS).

## 2026-11-04 · T1.10 part 1 · wrangler (haiku), lead checked
Did: Intel's four Sponza add-ons in `tools/assets/sources.json` (`sponza-intel-curtains`, `-ivy`, `-trees`, `-candles`; download-only), with 1k texture conversion for the first three (the base's script, Blender 4.5.0). Four rows in THIRD_PARTY.md.
Verified:
- Licence files byte-identical to the base's.
- The tool's download, checksum, extract, convert and output checks pass from scratch for all four, fed from a local server holding zips that `curl.exe` fetched from Intel.
- Lead: `--verify-only` on all four passes.
Not verified: a download straight from Intel through the tool. Intel rejects Node `fetch`, so part 2 switches the tool to `curl.exe` for Intel (DECISIONS).
Numbers: triangles: curtains 2.0 M, ivy 5.1 M, trees 2.0 M, candles 19.5 M (plus 10,000 point lights); base 3.7 M.
Next: part 2 (implementer, after T1.1): the `curl.exe` path; curtains into `sponza`; `sponza-full`; re-render Sponza's Cycles reference. Candles' size goes to Dennis.

## 2026-11-04 · T1.5 references · gpu-tester (sonnet), lead checked files
Did: `scenes/manifests/cornell-box-point.json`, the point-light Cornell (16a6099). Cycles renders at 2048 spp on the CPU, under the lock, into `bench/references/` (gitignored, so re-render them after a clean clone):
- `cornell-box-point/front`: seeds 0 and 1;
- `two-room/dark-room-leak-view`.
Verified: the agent ran `pnpm test` (483 pass) before the open-ground work landed. The lead saw the EXRs on disk. The viewer scene picker wasn't checked.
Numbers:
- Render times: Cornell point 269 s and 290 s (512²); two-room dark view 385 s (640×360).
- Mean indirect: Cornell point 0.0840; two-room lit-room 0.676; two-room dark room 0.0602, which is 8.9% of the lit room's, the light through the doorway.
- Cornell point, seed 0 against seed 1: indirect bias −0.006%, 8×8 block error 1.07%; full bias −0.003%, block error 0.48%. So one image's per-pixel indirect noise is about 10%.
Next: T1.5 uses these for its checks.

## 2026-11-04 · M1 start, T1.5 design · M1 lead (main session, opus)
Did:
- Branch `m1` from `main` at 799b050. The M1 board gets a Status column.
- First wave running in parallel on separate files: T1.1, T1.2, T1.10 part 1 (add-on downloads), B0b, and the Cycles references T1.5's checks need (point-light Cornell, two-room dark-room view).
- Wrote the T1.5 design, `docs/design/ddgi.md`. DECISIONS records how its Cornell accuracy and two-room leak checks are measured, and why `bounceEnergy` defaults to 1.
Verified: `pnpm lint && pnpm typecheck && pnpm test` pass on fresh `m1` (475 tests). The RTX was idle (1%); the `pnpm dev` process (46096) was gone. The stray python.exe (16968) uses one full CPU core; the parent session was told.
Next: review each task as it lands; T1.3 after T1.2.

## 2026-11-04 · T0.5, T0.12 · main session
Did: recorded Dennis's M0 sign-off. T0.5 and T0.12 marked done; M1 started.
Verified: Dennis, on the viewer: "the viewer runs". He had no other feedback and asked to start M1.
Not verified: a structured walk through every viewer control by Dennis. Any later feedback on it goes to T1.8 (viewer integration).
Next: merge PR #1 (`m0` → main); M1 from main.

## 2026-11-04 · B0 · implementer (opus), main session checked
Did:
- Emscripten pinned: emsdk 6.0.13 (commit 35659964) installs into the gitignored `tools/emsdk/` with `pnpm bvh:wasm`; `--check` rebuilds and compares. Dennis's JoltPhysics.js emsdk wasn't used.
- tinybvh built with `-msimd128 -mavx`, single-threaded, so `Build` runs its SIMD Bonsai builder. No scalar fallback: every browser with WebGPU has WASM SIMD.
- Licences: `packages/bvh/licenses/` (tinybvh, and the Emscripten runtime linked into the WASM), embedded in the module; `verify-artifacts.mjs` fails a bundle without them; THIRD_PARTY.md rows.
- Tree-to-layout conversion about 30% faster, output byte-identical. WGSL review nits fixed (345fb5d).
Verified (main session): lint, typecheck and test pass (475); `pnpm bvh:wasm --check` rebuilds the committed WASM byte-identical on Windows. Agent: oracle passes on RTX 3060 and Intel UHD after the WGSL cleanup; 3-run trace bench at 80cd2dc.
Numbers (median of 3):
- Sponza build in the browser worker: 2.67 s during the RTX runs, 2.19 s during the Intel runs (5.3 s before). In Node: tinybvh's build 1.5 s, its JS side 0.18 s, our layout conversion 0.65 s.
- Bonsai against tinybvh's other builders on Sponza: 1.5 s and tree cost 23.3, against binned SIMD 3.0 s / 24.2 and the scalar builder 5.5 s / 29.9.
- tinybvh trees ÷ our trees, closest-hit: Intel 1.04 / 1.03 / 1.07, RTX 0.92 / 0.99 / 1.05 (Sponza / Bistro exterior / Bistro interior).
Don't trust the RTX rows at 80cd2dc: another Chrome used about 20% of that GPU throughout, and every tracer, three-mesh-bvh included, lost about 35%. They carry a `note/gpu_busy,1` row. So "the WGSL cleanup cost nothing" isn't confirmed on the RTX.
B0 against its checks: oracle passes; trace speed passes on Intel, unclear on the RTX (re-run idle); Sponza ≤ 2 s fails by 0.2-0.7 s.
Next: B0b, the conversion inside the WASM (DECISIONS 2026-11-04); re-run the RTX trace bench on an idle GPU; M1's T1.3 switches the default mesh builder to tinybvh (`mesh-bvh-async.ts` still defaults to ours).

## 2026-11-04 · T0.10 review · reviewer (opus), main session triaged
Verdict: pass with notes. All four done-when checks hold. The benchmark is fair: the same rays, three-mesh-bvh built with SAH, the same warm-ups, and one compute pass timed on both sides. Sponza runs vary by under 2%.
Corrections to the part B and C entries:
- **The any-hit gap is wider than written.** The furnished room is 0.34× three-mesh-bvh on the RTX 3060 (168 against 499 Mrays/s, steady over 3 runs) and 0.65× on Intel. Part B called the 499 "not credible"; it holds up. Comparing our any-hit with our closest-hit says nothing, since they use different rays. Trees from tinybvh and three-mesh-bvh walk at the same speed as ours, so the cause is in the walk, the top level or the leaf size, not the triangle test. T1.3 measures it (per-ray node counts, three-mesh-bvh at leaf size 4).
- Intel Bistro exterior any-hit is noise: 2.87, 2.81 and 5.72 Mrays/s across runs.
- The shader that ships (after `af6e7f9`) wasn't timed. The B0 agent's final run times it.
Being fixed: the oracle must fail a corner escape and must be shown to catch faults (oracle agent); licence text, a reproducible WASM build and four code nits (B0 agent). Design doc updated (watertight layout and test).
Next (T1.3): `to-csv.ts` reads the commit when the CSV is written, not when the page ran; record it at run start.
Oracle fix (implementer, opus, debf057): every thin-box escape now fails, for closest-hit and any-hit; any-hit gets no ambiguous allowance there. 14 unit tests feed the checks a slipped edge ray, a corner escape, a wrong t, a false "clear" and a flipped front face, and each fails. Main session re-ran those tests (24 pass in the folder) and lint. The oracle still passes on both GPUs (agent, RTX 3060 and Intel UHD) with 0 escapes. T0.10 done.

## 2026-11-04 · T0.6 · native-runner built it; Opus reviewer failed it; native-runner (opus) fixed it; main session checked
Did: fork branch `m0-harness` of `native/rtxgi-ddgi` (c7d0162): device pick from `app.gpu` / `RTXGI_GPU`, `probeVariability.skipConverged=0`, Vulkan mesh buffer sized from the scene, `app.ignoreInput=1`, D3D12 build. `tools/native/` (play.ps1, bench.ps1, prepare-cache.ps1, native-lib.ps1, README): ini from our manifests, clean mode, scene caches, F4 benchmark to `bench/results/native.csv`, all under the shared GPU lock.
Review fixes: rows written per scene and one failed scene no longer loses the others; Ctrl-C closes the harness; the stale-cache delete really empties the folder; the 99% "probes updated every frame" check fails when its column is missing; the 9bd7f3c rows are marked invalid (`probe_update_every_frame,0`) and sweep rows carry `sweep,1`; `bench-lock run` hands its hold to a child (`DDGI_BENCH_LOCK_HELD`), so wrapping a native script no longer deadlocks.
Verified (main session): lock tests 16 pass, lint and typecheck clean. Agent, RTX 3060: done-when 4, default `two-rooms.ini` on upstream 059dd3c against the fork, 5 captures each: upstream against fork differs by 1.44% mean relative error, the same as upstream against itself (1.30%) and fork against itself (1.55%); the cause is the ini's variability threshold stopping updates on a random frame. Sponza re-run on c7d0162 under the lock.
Numbers (GPU ms, 256 rays, Vulkan / D3D12, build 1df1f97, every probe updated every frame):
| scene | probes (active) | trace | blend | DDGI total | frame |
| --- | --- | --- | --- | --- | --- |
| two-room | 918 (634) | 0.62 / 0.53 | 0.51 / 0.54 | 4.49 / 4.01 | 6.45 / 6.19 |
| cornell-box | 1728 (883) | 1.20 / 1.19 | 0.39 / 0.58 | 4.87 / 4.91 | 6.02 / 6.04 |
| sponza | 17760 (6457) | 9.34 / 9.85 | 3.57 / 5.33 | 16.32 / 17.77 | 21.57 / 22.92 |
| bistro-interior | 18720 (5059) | 14.44 / 13.94 | 3.28 / 3.95 | 21.89 / 21.33 | 37.11 / 31.08 |
| bistro-exterior | 26884 (5239) | 10.90 / 11.09 | 3.11 / 3.76 | 16.55 / 17.16 | 26.10 / 24.67 |
- Relocate, classify and variability are 0.02-0.13 ms each. Sponza Vulkan on c7d0162: trace 8.01, frame 18.36. The same config varies 15-30% between runs, so the Sponza row above is the slow end.
- Sponza sweeps (trace / blend ms): rays 32 0.87/0.58, 64 1.88/0.83, 128 3.89/1.46, 256 8.03/3.00, 512 18.66/9.09; probe spacing 2 m (2280 probes) 1.46/0.84, 1.5 m (5600) 3.72/1.53, 0.8 m (34500) 15.39/5.32.
Not verified: Ctrl-C live; the missing-column path triggered for real; D3D12 on c7d0162; the default-config check on D3D12.
Next: comparisons need 3+ runs per scene (the CSV has no run id; runs start at their `probes_x` row). Remove the upstream worktree (`native/build/upstream-src`) when no longer needed.

## 2026-11-04 · T0.10 (part C) · implementer (opus) built and measured, main session checked
Did:
- Watertight test without runtime vector indexing (per-ray rows built with `select()`), and nodes and triangles read as `vec4f` arrays (`platform/trace/wgsl.ts`).
- A faster SAH builder that makes byte-identical trees (`sah-builder.test.ts` checks it against a plain reference builder).
- `packages/bvh`: tinybvh 1.9.1 in WASM (B0), plus `buildMeshBvhFromTopology` so the tracer can take a prebuilt tree.
- A three-mesh-bvh-to-our-layout converter (`tmbvh-convert.ts`), for the experiment.
- Repeat runs (`repeat.ts`: `run<k>/` and `median/` rows), and a `-dirty` check scoped to the measured paths.
- Möller–Trumbore deleted after the final run (`af6e7f9`).
Verified: final 3-run medians at `16a2902`, Chrome 156, RTX 3060 and Intel UHD; I read the Sponza median rows in `trace.csv` and they match. Main session re-ran lint, typecheck and test (461 pass) and the oracle page on both GPUs after the removal.
Numbers (Sponza, closest-hit Mrays/s, median of 3): RTX 3060 ours 104.3, tinybvh trees 101.6, three-mesh-bvh 87.2 (1.20×); Intel UHD ours 12.8, three-mesh-bvh 6.5 (1.97×). Bistro exterior 0.83× and Bistro interior 0.99× on the RTX 3060. Sponza build: ours 8.7 s, tinybvh 5.3 s, three-mesh-bvh built then converted 7.4 s, three-mesh-bvh alone 5.4 s. Memory: ours 302 MB, three-mesh-bvh 153 MB. Everything else: `bench/results/trace.csv`.
What each step bought: removing the per-ray vector indexing took Sponza from 61 to 101 Mrays/s on the RTX 3060; `vec4f` loads gave +3% there and +23% on Intel. An unordered any-hit walk and a 32-entry stack bought nothing and were not kept.
Not verified: B0's ≤ 2 s (tinybvh runs its scalar builder: the only Emscripten here, 3.1.57, can't compile its SIMD builder); the Mac.
Watch: any-hit trails three-mesh-bvh on both Bistro scenes (interior 43 against 72 Mrays/s on the RTX 3060, 0.60×; 0.75-0.96× elsewhere), and on Bistro our any-hit is slower than our closest-hit. The bar was closest-hit, but DDGI's shadow rays are any-hit. T1.3 should look at it.
Next: Dennis decides on a newer Emscripten for B0 (DECISIONS 2026-11-04). The `median/` rows need a line in `bench/results/README.md` (that file is open in T0.6, so after it lands). Opus review of T0.10.

## 2026-11-04 · T0.5 (baselines) · gpu-tester (sonnet) measured, main session checked
Did: the full baseline run, all 11 scenes × probe grid (lpg), VXGI and no GI, on both GPUs. 1280×720, 5 rounds of 20 frames, 3 bake runs. Rows in `bench/results/baselines.csv` (803 per GPU, no `_problem` rows, no retries).
Verified (main session): row counts per GPU, spot-checked the Sponza rows against the agent's table. The viewer code didn't change between the commits the rows name (`70074de` NVIDIA, `d192f6d` Intel) and `43fe8cf`; `-dirty` comes from native and T0.10 files the viewer doesn't use.
Numbers (GI ms = frame with GI minus GI off, 95% interval; bake = wall time):
| scene | RTX 3060: lpg GI ms / bake | RTX 3060: VXGI GI ms / bake | Intel: lpg GI ms / bake | Intel: VXGI GI ms / bake |
| --- | --- | --- | --- | --- |
| sponza | 0.12 (below noise) / 2.3 s | 3.27 / 2.4 s | 3.13 / 15.7 s | 24.9 / 2.5 s |
| cornell-box | 0.04 / 0.18 s | 2.51 / 0.02 s | 0.95 (below noise) / 0.26 s | 16.3 / 0.08 s |
| bistro-interior | 0.25 (below noise) / 9.4 s | 3.46 / 0.7 s | 1.27 (below noise) / 8.7 s | 24.8 / 0.6 s |
- The probe grid costs almost nothing per frame and a lot to bake. VXGI costs 2.5-4 ms per frame on the RTX 3060 and 16-32 ms on Intel.
Don't trust:
- `vxgi_bake_gpu_pass_sum_ms` (1-71 ms against seconds of wall time). It misses VXGI's voxelising; use the wall time.
- Bistro exterior on the RTX 3060: GI-off frames scatter so much that GI-off came out slower than VXGI. Re-measure before quoting it.
- Sponza lpg on the RTX 3060: the GPU pass sum (2640 ms) is above the wall time (2323 ms), so the two clocks don't cover the same span. Fine for a baseline, not for a fine comparison.
Not verified: Dennis using it (done-when 4).
Next: Dennis tries the viewer; then T0.5 is done.

## 2026-11-04 · T0.5 · implementer (sonnet) built it; Opus reviewer failed it; implementer (opus) fixed it; main session checked
Did: `apps/viewer` on three r187.
- Viewing: a scene picker, orbit and fly, camera bookmarks, exposure and tone mapping, and the T0.3 overlay.
- GI: off, on or GI-only; the split wipe; three's LightProbeGrid and VXGI as baselines; an upscaler toggle.
- Hover help on every control, and the view saved in the URL.
- The runner page `tools/gpu/pages/viewer-baselines` and `bench/results/baselines.csv`. It's header-only for now: the old rows were measured with definitions the review threw out.
Review fixes:
- **Fair wipe:** both techniques get the same sky (from the manifest), no extra fill light, and GI strength 1.
- **Probe grid bake:** it re-bakes only when a bake setting changes, and a generation counter fixes the restart race.
- **"GI ms"** = frame with GI minus frame with GI off, through the same pipeline, with a 95% interval. A value whose interval includes 0 is flagged `below_noise`.
- **Bake time:** wall time (one fence at the end) and the sum of GPU pass times are reported separately.
- **Bad values:** the CSV never stores a negative or non-finite value without a flag, and the baselines page fails on device errors.
- **Speed:** Bistro interior gets ready in 22 s, down from 449 s.
- **Layout:** the canvas sits beside the panel.
Verified (I re-ran it): lint, typecheck and test pass (462). The agent reports the contact sheet 11/11 ready with 0 errors on the RTX 3060, headless Chrome 156. I looked at the Sponza wipe and the furnished-room screenshots.
Numbers (quick check, RTX 3060, Sponza): probe-grid GI 0.20 ms [0.02, 0.39], noisy; VXGI GI 3.27 ms [3.26, 3.27]; frame with no GI 2.04 ms. The full baseline table on both GPUs comes from the next gpu-tester run.
Finding: **the sky isn't calibrated.**
- The viewer's Preetham sky is about as bright as the sun, so the probe-grid half of Sponza turns blue.
- VXGI ignores the sky, so its shadows are black.
- Cycles uses a different sky model (Nishita).
- Fix in M1: one shared sky file read by every renderer, at a realistic sky-to-sun ratio.
Not verified:
- Intel for the fixes.
- A real headed browser by hand.
- Re-bake and the restart race at runtime (checked in code and unit tests only).
- Dennis's own use (done-when 4).

## 2026-11-04 · T0.10 (part B) · implementer (sonnet) built it, gpu-tester (sonnet) measured, main session (opus) interpreted
Did:
- Benchmark pages `tools/gpu/pages/trace-bench*` (one folder per scene, all `manual`) and `to-csv.ts`.
- `TraceBatch.timestampWrites` in `trace-gpu.ts`, so a batch can be timed.
- The full run on both GPUs: 5 pages × 2 GPUs, rows in `bench/results/trace.csv`.
Verified: all 10 runs pass, measured with GPU timestamps in Chrome 156 headless on nvidia/ampere (RTX 3060 Laptop) and intel/gen-12lp. Closest-hit results agree between ours-mt, ours-wt and three-mesh-bvh on 99.998–100% of rays, with no hit/miss mismatches. I spot-checked the Sponza rows in the CSV.
Numbers: Mrays/s closest-hit, ours Möller–Trumbore / ours watertight / three-mesh-bvh.
| scene | RTX 3060 | Intel UHD |
| --- | --- | --- |
| Sponza | 110.8 / 61.2 / 87.2 | 9.97 / 7.99 / 5.32 |
| Bistro exterior | 64.4 / 33.7 / 71.3 | 6.90 / 6.47 / 5.91 (thinned) |
| Bistro interior | 57.1 / 30.1 / 53.5 | 5.71 / 5.64 / 4.19 (thinned) |
| Furnished room | 114.2 / 60.7 / 113.6 | 17.8 / 16.5 / 11.7 |
Other costs:
- Build: ours is 3.7–4.6× slower than three-mesh-bvh (Sponza 28.1 s against 7.0 s).
- Memory: about 2× (Sponza 288 MB against 146 MB).
- Top-level rebuild: 1–10 ms for the real scenes, 121 ms (RTX 3060) and 86 ms (Intel) for 10,000 instances.
Verdict, by the rule set before measuring:
- The watertight test is the one we must ship, since Möller–Trumbore leaks (part A).
- On Sponza it is 0.70× three-mesh-bvh on the RTX 3060, below the 0.8× bar, and 1.50× on Intel.
- So part C optimises the tracer and measures again. See DECISIONS and `docs/design/trace.md`.
Not verified:
- Each page ran once; the spread is only within one run.
- Furnished-room numbers on the RTX 3060 are noisy (tiny dispatches); the three-mesh-bvh any-hit rate of 499 is not credible.
- The CSV has no "thinned" flag, and its rows carry three different commits.
Next: T0.10 part C.

## 2026-11-04 · T0.4 follow-up (furnished-room metal) · implementer (opus)
Did: the furnished room came out of Blender almost entirely metal, so it got no diffuse GI (found by T0.5).
- Cause: Blender 4.5's OBJ importer sets Metallic to the average of `Ka` whenever `illum` turns on reflection, and every material here is `illum 4` with mostly `Ka 1 1 1`. 12 of 15 materials came out metal.
- Fix: `obj-to-glb.py` resets every material to a plain dielectric. A new `"dielectric": true` output check in `sources.json` fails any metal material, or any primitive without a material, whatever the hashes say. The new GLB reconverts byte-identically.
Verified (I re-ran it): `assets:fetch --verify-only` "All outputs verified". I read the GLB JSON myself: 0 of 15 materials are metal.
Checked in passing: the Bistro GLBs have 14 metal materials each. They come from the FBX source and are mostly real metals (Vespa, bollards, chrome, cutlery), so I left them. The tree has none.
Next: re-take the T0.5 furnished-room screenshots and baselines with the new GLB.

## 2026-11-04 · T0.7 follow-up · implementer (opus), after an Opus re-review
Did: six fixes from the second review.
- **Matte references.** The Cycles material reset unlinks inputs before setting them, so references are pure Lambert. That includes Diffuse Roughness, which turns on Oren-Nayar in Blender 4.3+. A check stops the render if anything is left non-neutral.
- **Bench lock** (`tools/reference/bench-lock.ts`), rewritten with an owner file (pid, process start time, label, command). It takes over a lock only when its owner is dead. It has a CLI: `status`, `run -- cmd`, `acquire --pid`, `release --pid`.
- **Path tracer materials** keep their textures, alpha and normal maps.
- **Compare** masks NaN/Inf pixels before block averaging and before FLIP.
- **`--ppd` help** says that a user-given value is also divided by the block size.
- **A camera looking straight down** gets the same nudge as three's `lookAt`.
Verified (I re-ran it): `vitest tools/compare tools/reference` 37 pass. Lint and typecheck are clean. `bench-lock.ts status` reports the live T0.10 holder correctly.
The agent reports:
- `--calibrate` 7/7 within 0.05%, including a new linked-metal case;
- the Cornell path tracer at 1024 spp is +0.01% against Cycles.
Numbers: the Sponza 4096 spp re-render, diffuse-only, took 997 s. 76 texture links were removed. Full mean 0.0341 → 0.0349 (+2.4%); indirect 0.0042 → 0.0050 (+18%). The old "metal" texels had absorbed bounce light. Cornell, two-room and furnace have no texture links, so their references stand.
Deviation: a lock folder with no owner file counts as stale only after 3 hours, because older holders still use a bare `mkdir`. Lower it once every holder writes `owner.json`.

## 2026-11-04 · T0.4 follow-up · implementer (opus), after an Opus re-review
Did: seven fixes from the second review.
- **Blender GLBs:** checked per bufferView (hashes in `tools/assets/glb-views/<id>.json`). Only TANGENT views may vary. Indices, images, JSON and padding are pinned exactly, and Blender is pinned to 4.5.0.
- **Hang guard:** a real non-zero exit always fails. The guard's own kill carries a marker exit code.
- **Unpacking:** stale unpacked files are removed. A cached unpack is reused only when its file list matches. Inputs must be unchanged after a conversion.
- **Sponza license:** `CC-BY-4.0+intel-preface`, with an approval that points at its DECISIONS entry. The tool checks that the entry exists.
- **Sealing test:** it states honestly what it checks, fails when a name pattern matches nothing, and uses the builder's constants.
- **Light range:** manifests may now set `range: 0`, meaning no range window, as in Cycles.
Verified (I re-ran it): `vitest scenes tools/assets` 220 pass, `assets:fetch --verify-only` "All outputs verified", lint and typecheck clean.
Finding: Blender's glTF export varies between runs only in TANGENT data (4th-decimal rounding): 1 view of 27 on the tree, 1 of 11,036 on Bistro interior.
Not verified: Bistro exterior wasn't reconverted. It uses the same exporter as the interior.
Next:
- `tools/native/native-lib.ps1` passes `range` as the light radius, so `range: 0` would mean radius 0 there. No manifest uses 0 yet.
- The T0.11 `LightDescriptor` says range is needed for culling; `range: 0` should mean "no culling".

## 2026-11-04 · T0.4 · implementer + wrangler (sonnet), reviewed by reviewer (sonnet) and main session (opus)
Did: built the scene set and its tooling.
- `scenes/`: manifests for 11 scenes, a validator, procedural builders, a GLB writer, the check page, a contact sheet, a sealing test, and `sun-visibility.ts`.
- `tools/assets/fetch.ts` with `sources.json`: sha256 checks, a cache, a Blender 4.5 pin, `--verify-only`, and a hang guard.
- Convert scripts under `tools/assets/convert/`.
- License rows in `THIRD_PARTY.md`.
Verified (I re-ran it):
- `pnpm assets:fetch --verify-only` gives "All outputs verified" in 22 s. A clean fetch from scratch took 9 min 16 s.
- `vitest scenes tools/assets`: 195 tests pass. Lint is clean.
- I looked at the contact sheet (`bench/screenshots/contact-sheet/index.html`); every scene is lit and at the right scale.
Numbers (tris / meshes / materials / textures):
| scene | tris | meshes | materials | textures |
| --- | --- | --- | --- | --- |
| cornell | 86 | 8 | 4 | 0 |
| two-room | 216 | 18 | 8 | 0 |
| closed-door | 228 | 19 | 9 | 0 |
| furnace | 108 | 9 | 1 | 0 |
| thin-wall | 108 | 9 | 6 | 0 |
| cabinet-row | 744 | 62 | 6 | 0 |
| courtyard-tree | 2,062,607 | 13 | 6 | 10 |
| furnished-room | 269,764 | 21 | 15 | 2 |
| bistro-exterior | 2,829,226 | 1591 | 132 | 277 |
| bistro-interior | 1,043,077 | 2062 | 71 | 145 |
| sponza | 3,747,018 | 405 | 44 | 72 |
Deviations:
- Sponza is Intel's New Sponza with 1k textures, download-only (see DECISIONS). Its sun was raised to 71° so 62% of the open court floor is lit.
- Blender's glTF export is not bit-reproducible, so its three outputs are checked by the JSON-chunk sha256 plus exact sizes. See NOTES.
- The furnished-room model is scaled by 0.34.
Not verified: the Mac (deferred).
Next: the courtyard-tree has no exported GLB in `assets/generated`, which T0.6 needs. GLTFLoader leaves shadows off, so loaders must turn on cast/receive.

## 2026-11-04 · T0.7 · implementer (sonnet), reviewed by reviewer (sonnet) and main session (opus)
Did: three pieces, with settings, units and tolerance in `tools/reference/README.md`.
- Cycles references: `tools/reference/cycles`, Blender 4.5, CPU, `--factory-startup`, no denoise. Unit conversions are in `units.py`.
- A three-gpu-pathtracer reference: `tools/reference/pathtracer`. `--denoise` uses a gitignored build of pmndrs/denoiser.
- `tools/compare`: mean relative error, bias, RMS, p95 block error, a clamped line, and HDR-FLIP through the `flip-evaluator` Python package.
All renders, CPU ones included, take the bench lock and release it on exit.
Verified (I re-ran the compare checks):
- an image against itself gives 0;
- a +10% copy gives 0.100000;
- Cornell numbers as below;
- `--calibrate` (point, sun, spot, furnace 0.5/0.125) is within 0.05% of the exact answers;
- lint, typecheck and test pass (375). The pathtracer ran on the RTX 3060 Laptop in Chrome 156 headless.
Numbers: Cornell, Cycles 2048 spp against pathtracer 8192 spp, after 8×8 blocks.
| image | bias | block error | clamped | p95 | FLIP at 8 ppd |
| --- | --- | --- | --- | --- | --- |
| full | -0.002% | 0.37% | 0.71% | 2.3% | 0.041 |
| indirect | -0.014% | 1.43% | 1.43% | 3.8% | 0.056 |
Two-room bias: +0.11% full, +0.12% indirect. The tolerance is in DECISIONS. Times and sample counts:
| reference | samples | time |
| --- | --- | --- |
| Cycles Cornell | 2048 | 336 s |
| Cycles two-room | 2048 | 442 s |
| Cycles Sponza | 4096 | 1658 s, while other work ran |
| pathtracer Cornell | 8192 | 225 s |
| pathtracer two-room | 8192 | 288 s |
Deviations:
- Denoising is a single pass at the end, not progressive.
- No pathtracer reference for Sponza: its page can't load glTF yet.
- The tolerance comes from Cycles' seed-to-seed noise only; the path tracer's own noise wasn't measured.
Not verified: the Intel GPU; a real Ctrl-C releasing the lock (tested only after a normal run and a failed one); Cycles on the GPU.
Next:
- Done: the Sponza Cycles reference was re-rendered with the raised sun. 4096 spp took 1196 s; the mean is 0.034, up from 0.018; the noise is about 3% (full) and 11% (indirect), estimated from a 1024 spp seed-1 render. The floor band is lit.
- Cycles has no point-light range window, so on two-room three is about 12% dimmer than Cycles at 7 m.
- Remove the stray `C:\x\` folder (two small EXRs from a failed first run).

## 2026-11-04 · T0.10 (part A) · implementer (sonnet), reviewed by main session (opus)
Did: built the tracer from `docs/design/trace.md` in `packages/ddgi-core/src/platform/trace/`:
- the binned-SAH builder, in a worker plus an inline version;
- the CPU top-level builder;
- the WGSL two-level walk (`giTraceClosest`, `giTraceAny`) and a batch kernel;
- a CPU port of the walk and a brute-force reference tracer.
The GPU oracle page is `tools/gpu/pages/trace-oracle/`. Its cases are thin walls, mirrored and non-uniform transforms, instancing, and 2M rays aimed at shared edges.
Verified (I re-ran it): `pnpm test:gpu --filter trace-oracle` passes on nvidia/ampere and intel/gen-12lp at default limits, with the worker build, about 66 s per GPU. 46 unit tests pass. Every oracle case has 0 mismatches; the largest relative error in t was 9.3e-6.
Finding: Möller–Trumbore let rays slip at shared edges, 231,925 of 2M (11.6%) on NVIDIA and 60,147 (3.0%) on Intel. The watertight test let 0 through on both. Möller–Trumbore also leaked corner rays into a sealed thin box; watertight didn't.
Numbers: the trace module binds 3 storage buffers, the batch kernel 5. The stack depth needed is 19 at worst, of 64.
Deviations: both triangle tests are a build option for now; batch hits use `GiHitRecord` (the contract is updated); there's a top-level header and a `mirrored` flag. See the design doc's "What part A found".
Not verified: speed (part B), large limits, the Mac (deferred), watertight under other drivers' FMA fusing.
Next: part B times both triangle tests on Sponza, Bistro and the furnished room against three-mesh-bvh. `oracle-compare` and `reference-tracer` are test helpers and should leave the public exports in M1.

## 2026-11-04 · T0.9 · implementer (sonnet), reviewed by main session (opus)
Did: `tools/gpu/pages/spike-babylon/`. Babylon 9.33.2's WebGPU engine with a `MaterialPluginBase` plugin, built generically from a `ShadingModule`. It adds E/π·albedo from our 2D-array, 3D and `rgba32float` textures plus `vec4f` uniforms.
Verified (I re-ran it on both GPUs: `pnpm test:gpu --filter spike-babylon` passes on nvidia/ampere and intel/gen-12lp): readbacks match E/π within 0.2%, e.g. (0.1973, 0.1687, 0.1878) against (0.1974, 0.1687, 0.1878). With the plugin off, the reading is 0.
Answers:
- **Can Babylon run on our device? No.** `WebGPUEngine` always creates its own device. Our limits and features go through `deviceDescriptor` and arrive intact, and the core takes `engine._device`. `dispose()` destroys the device, and Babylon makes a new one after device loss.
- **Foreign textures:** yes, via `wrapWebGPUTexture` plus flags plus `updateWrappedWebGPUTexture`.
- **Plugin injection:** yes. The plugin can turn env diffuse off while keeping env specular.
- **Contract changes:** none required. The per-texture sampler rule from T0.8 also fits Babylon's `<texture>Sampler` auto-binding.
Not verified: the Mac (deferred), device-loss restore, the GLSL path, mipmapped wrapped textures.

## 2026-11-04 · T0.8 · implementer (sonnet), reviewed by main session (opus)
Did: spike pages `tools/gpu/pages/spike-three-{standard,meshlet,visibility,extras}` plus the shared `spike-three-lib/`.
Verified (I re-ran it on both GPUs: `pnpm test:gpu --filter spike-three`, 4 pages pass on nvidia/ampere and intel/gen-12lp). A white Lambert plane read (0.35810, 0.23873, 0.23873), which equals E/π for E = (1.125, 0.75, 0.75), in all three drawing pipelines. Tolerance was 1% + 0.001.
Answers:
- (a) Our device with raised limits, passed to `new WebGPURenderer({ device })`: **yes**.
- (b) A `Light` subclass with a registered node, sampling compute-written `ExternalTexture`s through a `wgslFn` shaped like `giIrradiance`: **yes**. 2D-array, 3D and `rgba32float` textures, samplers and `vec4f` parameters all work.
- **Priority slot with a real `LightProbeGrid`:** the winner's value only, never the sum. Equal priority goes to the later light.
- (c) Standard, Meshlet and Visibility-Buffer pipelines: **yes** on both GPUs.
- (d) Env diffuse off, env specular on: **yes, with a workaround**. The node wraps `lightingModel.indirect` and scales `iblIrradiance` by 1 − coverage. That removed 98.6% of env diffuse inside the volume and left the outside untouched. It depends on internals; PR #35017 would replace it.
- (e) GI inside a `CubeCamera` capture: **yes** (PR #35046 not tested).
- **Recompiles** (shader modules / render pipelines): intensity 0 costs 0/0. Removing the light costs 2/1 (Visibility-Buffer 4/2), and re-adding it costs the same again.
Not verified: the Meshlet pipeline inside a cube capture (it skips cube targets, per the source).
Contract revision: samplers belong to a texture (docs/api/platform.md, "Revisions").

## 2026-11-04 · T0.2 · implementer (sonnet), reviewed by main session (opus)
Did:
- `pnpm test:gpu [--gpu nvidia|intel] [--filter name] [--json file] [--timeout ms] [--no-webgpu]`, in `tools/gpu/`.
- It starts headless Chrome over the DevTools protocol (no Playwright) and serves `tools/gpu/pages/*` with vite.
- It collects `Log.entryAdded`, console output and exceptions, and takes a screenshot per page.
- `client.ts` gives pages `runGpuTest` and `reportGpuTest`. README explains the page contract.
- A reviewer pass found holes in Ctrl-C cleanup and a self-test that accepted any failure. Both are fixed.
Verified (Windows 11, headless Chrome 156; I re-ran each check):
- `--gpu nvidia` reports `vendor=nvidia architecture=ampere`, smoke passes.
- `--gpu intel` reports `vendor=intel architecture=gen-12lp`, smoke passes.
- `--filter wgsl-error` fails and prints the WGSL error. A full run requires that error text, not just any failure.
- `--no-webgpu` fails with "WebGPU is not available".
- `--json` writes per-page outcome, data and adapter.
- 54 runner unit tests pass.
- Ctrl-C was tested by the implementer at 20 delays on both GPUs: no Chrome or profile folder left behind.
Numbers: `--filter smoke` takes 2.1–2.5 s wall on NVIDIA and 6.1–6.7 s on Intel. About 2 s of the Intel time is the Windows lookup of the Intel adapter's ID.
Not verified: macOS and Linux. Cleanup after `taskkill` or a crash isn't possible on Windows (README says so).
Next: benchmark tasks use `--json` plus the GpuTimer strict mode.

## 2026-11-04 · T0.3 · implementer (sonnet), reviewed by main session (opus)
Did:
- `GpuTimer` in `packages/ddgi-core/src/platform/timing/`. It has a slot ring with `mapAsync` readback, allocation inside error scopes, and switches itself off with one warning on failure. Frames split over two submits keep both submits' passes. A strict mode for benchmarks throws instead of silently leaving passes untimed.
- A stats overlay in `apps/viewer/src/stats/`.
- A device page in `tools/gpu/pages/device/`, plus `pnpm device:serve`: HTTPS on the LAN with a self-signed certificate made by openssl, and results posted to `bench/results/incoming/`.
Verified (Windows 11, headless Chrome 156):
- 63 unit tests pass, including a fake device that returns different data per slot, so mislabelled readbacks would be caught.
- `pnpm test:gpu --filter device` passes on `nvidia/ampere` and `intel/gen-12lp`. The implementer ran 3 runs per GPU under the bench lock; I re-ran one per GPU.
Results:

| Device and browser | WebGPU | Timestamps | Timing overhead (CPU frame time, timed vs untimed) |
| --- | --- | --- | --- |
| Windows Chrome, RTX 3060 | yes | yes | within noise: median +0.225 ms, rounds -0.05 to +0.5 ms, so under ~6% |
| Windows Chrome, Intel UHD | yes | yes | within noise: median +0.25 ms on a 92 ms frame, so under ~0.7% |
| Mac Chrome / Mac Safari / iPad / older iOS | deferred (DECISIONS 2026-11-04) | | |

Numbers: per-pass GPU ms for compute and render are 3.59 / 1.76 on NVIDIA and 60.9 / 28.5 on Intel (a fixed workload). Rows are in `bench/results/timing.csv`.
Not verified: Apple devices (deferred); whether normal, non-headless Chrome rounds timestamps to 100 µs; Ctrl-C of `device:serve`.
Next: T0.5 wires the overlay into the viewer. Benchmarks (T0.10 part B) should use the timer's strict mode.

## 2026-11-04 · T0.1 review · main session (opus)
Did: reviewer pass (verdict: pass with notes) plus my own re-run. Fixed one finding: the boundary rule missed dynamic `import('three')`; added a `no-restricted-syntax` rule in `eslint.config.mjs` for `ImportExpression`.
Verified (Windows 11, no GPU): `pnpm lint && pnpm typecheck && pnpm test && pnpm build` exit 0; `export const e = import('three')` in ddgi-core now fails lint with the boundary message; `packages/ddgi/dist` has index/core/three `.js` + `.d.ts`. A fresh `git clone` of commit ecc503f, then `pnpm i --frozen-lockfile && pnpm lint && pnpm typecheck && pnpm test && pnpm build`: exit 0, dist has all three entries.
Not verified: no CI yet.
Next: T0.2, T0.4 dispatched. Marked T0.1 `done`.

## 2026-11-04 · T0.1 · implementer (sonnet)
Did: pnpm workspace mirroring `@pmndrs/upscaler` (vite 6, vitest 3, ESLint 9 flat + typescript-eslint, rollup-plugin-dts, TS 5.9 strict + `@webgpu/types`; three and `@types/three` pinned `0.187.0`, peer `~0.187.0`, optional on the umbrella).
- Root: `package.json`, `pnpm-workspace.yaml`, `tsconfig.json`, `eslint.config.mjs`, `vitest.config.ts`, `pnpm-lock.yaml`.
- `packages/ddgi` (umbrella, entries `.`, `./core`, `./three`, own vite/rollup/`verify-artifacts.mjs`), `packages/ddgi-core` (`src/platform/`, `src/techniques/ddgi/`, empty modules with indexes), `packages/ddgi-three`.
- `apps/viewer` and `examples/webgpu/hello`: minimal vite apps (placeholder `main.ts`). `tools/README.md`, `bench/results/README.md` (CSV columns).
- Boundary rule: `no-restricted-imports` on `packages/ddgi-core/**`; also `ddgi-three` may not import babylon/pixi/the umbrella. `.git/info/exclude` has `node_modules`.
Verified (Windows 11, Node 24.13.1, pnpm 10.23.0; no GPU involved):
- Done-when 1: deleted all `node_modules` and `dist`, then `pnpm i --frozen-lockfile` (exit 0) and `pnpm lint && pnpm typecheck && pnpm test && pnpm build` (exit 0). `pnpm test` reports "No test files found, exiting with code 0" (`passWithNoTests`; there is no code to test yet).
- Done-when 2: `import 'three';` in `packages/ddgi-core/src/platform/` failed `pnpm lint` with "'three' import is restricted ... ddgi-core must stay engine-free: do not import three, babylon, pixi or ddgi-three here. Put engine code in an adapter package  no-restricted-imports"; `import ... from '@pmndrs/ddgi-three'` failed the same way. Both probes reverted; lint is clean again.
- Done-when 3: `packages/ddgi/dist/` holds `index|core|three` `.js` and `.d.ts`; `verify-artifacts.mjs` also checks the `exports` targets exist and no `@pmndrs/ddgi-*` name leaks. `pnpm build:apps` builds both apps.
Not verified: a clone from git (nothing is committed yet; I deleted installs and `dist` in the working tree instead). The apps were built, not opened in a browser. Empty modules make vite and rollup warn "empty chunk".
Numbers: install 7.4 s cold-ish (first run), 4.3 s on the clean re-run with a warm store; `pnpm build` 4.2 s wall (vite 51 ms, dts 2.1 s); `pnpm build:apps` 1.5 s.
Next: T0.2 can start. When real code lands, drop `passWithNoTests`; add a `test:gpu` script in T0.2. No CI workflow yet (the upscaler has `.github/workflows/ci.yml`; a later task). Prettier is not used, as in the upscaler.

## 2026-11-04 · setup · main session (opus)

Did:
- Created the repo with the spec, the audit, the research notes and the native RTXGI submodule.
- Analyzed three.js r187, HomeFig's DDGI branch, `@pmndrs/upscaler` and `@pmndrs/denoiser`.
- Wrote `PLAN.md`, `TASKS.md`, `docs/NOTES.md`, `docs/DECISIONS.md`, `CLAUDE.md` and the subagent definitions in `.claude/agents/`.

Installed on the Windows laptop:
- RenderDoc 1.47.0, via winget;
- NVIDIA FLIP (`flip-evaluator` 1.7, via pip).

Already present: PIX 2603.25, Blender, Vulkan SDK 1.3.283, Nsight Systems 2022.4, Nsight Compute 2022.3.

Not done: Nsight Graphics, which needs an NVIDIA developer-account login. Dennis to install it from developer.nvidia.com/nsight-graphics.

Next: T0.1 (implementer). T0.11 can start in parallel in the main session.

## 2026-11-05 · Clock-rule timing runs: Budget v2 calibration and B2 Sponza build · measurer
Did: RTX 3060 Laptop (nvidia, ampere), headless Chrome 156.0.8078.80, each run under `bench-lock run` (cool start below 70 C, 200 ms sampling). No code changes. Rows in `bench/results/timing.csv` (720 rows, `note/set` 1-3, `note/timing_valid,1`, `note/thermal_share,0`, `note/clock_rule_job_budget_v2_calibration,1`). The notes were added by hand with `--note` because to-csv ran after the runs, outside `bench-lock run`, so the rows have no `note/sm_clock_mhz` or `note/start_temp_c`.
Budget v2 calibration (`ddgi-batch-cost`, `VITE_DDGI_SECTIONS=startup,budget`, `rates` not needed; budget 2 ms). All 12 runs VALID, 0% thermal samples. Update ms is the median of the timed frames at the calibrated batch; the page's check is within ±10% of 2 ms.
| scene | run | calibrated probes | update ms | vs budget | ±10% | start C | clock MHz |
|---|---|---|---|---|---|---|---|
| cornell-box | 1 / 2 / 3 | 1574 / 1821 / 1563 | 1.877 / 2.057 / 1.838 | -6.2 / +2.9 / -8.1% | pass | 61 / 66 / 65 | 1575 / 1552 / 1544 |
| two-room | 1 / 2 / 3 | 715 / 706 / 707 | 1.923 / 1.937 / 1.953 | -3.8 / -3.1 / -2.4% | pass | 62 | 1522 / 1507 / 1425 |
| sponza | 1 / 2 / 3 | 268 / 255 / 272 | 2.020 / 1.995 / 2.085 | +1.0 / -0.3 / +4.3% | pass | 62 | 1762 / 1687 / 1762 |
| bistro-exterior | 1 / 2 / 3 | 124 / 160 / 132 | 1.964 / 2.114 / 1.985 | -1.8 / +5.7 / -0.8% | pass | 63 / 63 / 69 | 1912 / 1729 / 1530 |
Medians of the 3 VALID runs: cornell 1574 probes, 1.877 ms; two-room 707, 1.937; sponza 268, 2.020; Bistro 132, 1.985. Spread of the calibrated probes: cornell 1563-1821 (16%), two-room 1.3%, Sponza 6.7%, Bistro 29% (124-160). Cornell run 2 calibrated 1821 probes, above the 1728 it has, so all probes fit (`all_probes_fit` 1) and the batch ran at 1728; runs 1 and 3 did not fit them all. Bistro's calibrated count varies most between runs, though every run is inside ±10%.
B2 Sponza H-PLOC build (`bvh-hploc-profile`, `pnpm test:gpu --gpu nvidia --filter bvh-hploc-profile`, Sponza, 132 meshes, the page's own median of 5 builds after one warm-up; the page is about 6 s, so it only gets 4-8 loaded samples). 5 attempts: runs 1 and 5 VALID, runs 2, 3 and 4 NO VERDICT (3-4 loaded samples, fewer than 5). The page's own timing is short and has no option to lengthen it (changing that is a code change).
| run | verdict | build ms | clock MHz (loaded median) |
|---|---|---|---|
| 1 | VALID (65 C, 8 loaded samples) | 46.899 | 1005 |
| 2 | NO VERDICT | 47.483 | 1301 |
| 3 | NO VERDICT | 48.013 | 735 |
| 4 | NO VERDICT | 46.592 | 382 |
| 5 | VALID (68 C, 5 loaded samples) | 46.289 | 967 |
Median of the 2 VALID runs: 46.6 ms (bar 50 ms): pass, but on only 2 VALID runs (the task asked for 3) and loaded clocks near 1,000 MHz with the power limit on in 100% of loaded samples. Climb 26.0-26.8 ms, sort about 8.6, triangles about 5.1. No B2 CSV rows (the profile page has no to-csv).
Flagged: B2 clocks were low (about 1,000 MHz) even on VALID runs; the rule has no MHz bar, so the verdict stands, but the number is a slow-clock one. Nothing looked wrong in the outputs (no NaNs, adapter was nvidia/ampere every time). A third VALID B2 run needs a longer page or a second pass in one run.
Next: a longer-running variant of the B2 profile page (repeat builds for 10 s) would give the verdict enough samples.
