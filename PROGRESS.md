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

## 2026-11-05 · T1.9 Sponza accuracy check · implementer (opus), lead checked
Did (5434142): `SceneSync.sentMaterial(material)` returns the GI descriptor the platform got; page `ddgi-sponza-accuracy` (manual) with `run.ts` (writes `gi-materials.json`, EXRs and comparisons; `--table` prints the three scenes); `render.py --materials-json` (constant colours, alpha 1, no normal maps, checked before rendering) and a GI composite (DECISIONS: indirect + direct − lights-only direct). Convergence: 16 warm-up passes, then blocks of 32 full passes at hysteresis 0.97 until the mean moves < 0.5% between blocks (2–12 blocks); fixed 3,885 probes a frame.
Verified (agent): RTX 3060 and Intel UHD. Cycles: Blender 4.5, OptiX, 8,192 samples, 640×360, 321 s, under the lock; two seeds give 8×8-block noise 0.51% (per pixel 4.1%). Direct light agrees with Cycles to 0.02% (0.011139 against 0.011137), so the excess below is DDGI's.
Numbers (indirect, against Cycles; bias, 8×8 block error, p95 block, FLIP 8×8, FLIP per pixel):
- Cornell (point): +5.68%, 6.38%, 98.6%, 0.179, 0.197.
- two-room lit room: +5.63%, 5.98%, 12.0%, 0.073, 0.074.
- **Sponza (untextured): +26.29%, 29.88%, 162.8%, 0.331, 0.329. Fails the ±10% target.** Intel +26.25%. A long run (272 passes) +26.52%.
Not verified: the cause. The ratio map shows DDGI brighter on most shadowed surfaces, the curtains most.
Next: find Sponza's +26% (leaks through thin curtains, probe density, sun shadow rays, the decal `dirt_decal` with GI albedo 0 acting as a black opaque surface because glTF's default metalness is 1 and M1 ignores `blocksLight`); a README section for `--materials-json`; the test view's full image is black on sky pixels.

## 2026-11-05 · Review of the sky-aware GI reference · reviewer (opus)
Asked by the parent session, so the reference isn't judged by the agent that wrote it. Verdict: **pass with notes** (all from the reviewer's own renders and code reading).
- DDGI's shaded GI term equals the composite's contents (DECISIONS has the details); the sun's direct light matches to 0.02% and Cycles full = lights-only direct + composite (0.012070 against 0.012069).
- Cornell (point) and two-room with the composite: unchanged (+5.68%; two-room +5.50% on OptiX against +5.63% on the CPU reference, an OptiX-against-CPU difference of 0.13 points, the same without `--materials-json`).
- `--materials-json`: all 49 Blender materials set, all 32 rows matched, the table hash recorded; Metallic, Transmission and specular forced to 0; `glass` and `dirt_decal` opaque black on both sides (consistent, not physical).
To fix (after the Sponza investigation, which is in those files now):
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
