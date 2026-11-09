# bench/results

One CSV per benchmark family, appended to (never rewritten). Numbers live here or in PROGRESS entries, not in report files.

Columns, in this order:

| column | meaning |
|---|---|
| `date` | ISO date of the run, `YYYY-MM-DD` |
| `commit` | short git hash of the code that was measured (add `-dirty` if uncommitted changes) |
| `device` | GPU, e.g. `RTX 3060 Laptop`, `Intel UHD` |
| `browser` | browser and version, or `native` for the native harness |
| `scene` | scene id from the test-scene manifest |
| `metric` | what was measured, with the unit in the name, e.g. `gpu_ms_probe_update` |
| `value` | the number |

## trace.csv

- **Repeated runs:** a 3-run measurement writes `run1/`, `run2/` and `run3/` rows plus `median/` rows (the median of the three). Quote the `median/` rows.
- **`probes/thinned,1`:** that scene ran with fewer rays (Intel Bistro), so compare its Mrays/s, not its ms.
- **`note/gpu_busy,1`:** another program was using the GPU during that scene's runs. The rows are kept (the file is never edited) but aren't valid timings. The RTX 3060 rows at `80cd2dc` are like that.
- **`note/cpu_core_busy,1`:** another process held one CPU core during the runs (the stray python.exe at `8e5373c`). GPU timings are valid. CPU-side numbers, such as the worker build time, may read a little high.
- **`scene_parts` (native.csv):** how many model parts the scene was merged from. Sponza rows without it are the base package only; from 2026-11-05 `sponza` is base plus curtains (2 parts).
- **`grid_ddgi,1` and `label,t1.9-matched` (native.csv):** the probe grid follows our DDGI rule (`floor(size / spacing) + 1` probes per axis at exactly the manifest spacing, centred), not the harness's default (`ceil(size / spacing)` cell centres). Compare these rows with our DDGI. Don't mix them with default-grid rows. The commit reads `c7d0162-dirty` because the `tools/native` option was uncommitted at the time; the fork itself is clean.
- **`note/sm_clock_mhz` (every CSV, NVIDIA rows only):** the RTX's SM clock while that run was going: the median over the nvidia-smi samples taken under load (70% GPU use or more). With fewer than three such samples the note is the median over ALL samples instead, idle seconds before and after included, so it reads low. That happens for short pages, for latency-bound ones (`tlas-bench`, `ddgi-timing`: nvidia-smi never showed 70%) and for the native harness (nvidia-smi shows only 35-45% while it runs minimized). Then `note/sm_clock_max_mhz` is the better sign of what the card reached. `node tools/reference/bench-lock.ts run` samples nvidia-smi: once a second until the clock rule (2026-11-05), every 200 ms since. Companion rows: `note/sm_clock_min_mhz`, `note/sm_clock_max_mhz` and `note/mem_clock_mhz` (same samples). The RTX 3060 Laptop's maximum is 2100 MHz. Short loads on a cool card read 1,400 to 2,000 MHz; a sustained GPU-bound load (more than about 20 s) heats it to the 87 C target and the clock falls to 450 to 600 MHz (see `note/clock_low`). The nvidia-smi values refresh only about every 0.6 s, and the first loaded second of a burst still shows the ramp from the idle clock. Since the clock rule the median is information only; `note/timing_valid` says whether the run counts. One note set belongs to one `bench-lock run`, which can cover several repeated runs, so the clock is for the whole group, not each run. Rows without the note were taken before 2026-11-05 or outside `bench-lock run`; `note/sm_clock_locked_mhz,1200` marks the clock locked by hand (NOTES). The converters read the summary file from `--clock-file`, or from DDGI_CLOCK_FILE, which `bench-lock run` sets for its command.
- **`note/clock_low,1` (every CSV, NVIDIA rows only; old rows only, replaced by `note/timing_valid` with the clock rule on 2026-11-05):** the run's SM clock was low, so its milliseconds are slow-clock numbers (the rows stay; the file is never edited). The rule, from `isClockLow` in `tools/gpu/clock-note.ts`: among the nvidia-smi samples under load (GPU use 70% or more), the median SM clock is **below 1,500 MHz**, with at least three such samples. A run with fewer than three loaded samples (short pages, latency-bound pages, the native harness) gets no verdict, so no flag, because its all-sample median includes idle seconds and would read low for no reason. `bench-lock run` also prints a `WARNING: the GPU clock was LOW under load` line at the end, with the share of loaded samples where nvidia-smi named thermal throttling or the power limit and the GPU temperature; the to-csv scripts print it too. Why 1,500: the RTX 3060 Laptop reaches 2,000 MHz or more within seconds of a cool start and holds 1,400 to 2,000 while cool, but under a sustained GPU-bound load it heats to its 87 C target in about 20 s and then settles at 450 to 600 MHz (measured 2026-11-05; see PROGRESS, "GPU clock under sustained load"). Many of those runs are 3 to 4 times slower than a cool card's. Compare flagged rows only with other flagged rows, or re-time after the GPU has cooled (under about 70 C) with a shorter timed section. Idle gaps do not cause a flag: they are below 70% use and are not counted.
- **`note/timing_valid,1|0`, `note/timing_no_verdict,1`, `note/thermal_share`, `note/start_temp_c` (every CSV, NVIDIA rows only; from 2026-11-05):** the clock rule for valid timing (DECISIONS 2026-11-05, `judgeTiming` in `tools/gpu/clock-note.ts`), which replaced `note/clock_low` that day. `bench-lock run` waits until the GPU is below 70 C (at most 15 minutes), then samples nvidia-smi every 200 ms. **Valid (`timing_valid,1`):** the run started below 70 C (`start_temp_c`) and fewer than 5% of its loaded samples (70% GPU use or more) show thermal throttling (`thermal_share`, 0 to 1). **Invalid (`timing_valid,0`):** a hot start, or 5% or more thermal samples. **No verdict (`timing_valid,0` plus `timing_no_verdict,1`):** fewer than 5 loaded samples, or no start temperature recorded. Only valid rows can pass or fail a target. There is no MHz bar: from a cool start the card is power-limited and its median lands anywhere from about 1,000 to 1,700 MHz, depending on the load. Rows written before the change keep their old notes (`clock_low`, `cool_start`); it landed during 2026-11-05, so that date has both kinds.
- **`ours-cwbvh/*` and `agreement/ours-tinybvh_vs_ours-cwbvh/*` (trace.csv): the B3 CWBVH experiment** (compressed 8-wide nodes, dropped 2026-11-05; PROGRESS, B3). Its commits (0ad7725, 646de83, 9c9a997) are on branch `bvh-b3`, not on m1, and so is the code that made the rows. The `ours-tinybvh/*` rows in the same sets are its BVH2 baseline. `ours-cwbvh/timed_first` says which of the pair was timed first; the second runs about 6% slow.
- **`three-composite.csv` and `saturation.csv` (from 2026-11-05):** one `note/set,<n>` per page run, with `note/cool_start,1` (started below 70 C), and for the composite `note/gpu_bound,1`, `note/width` / `note/height` and `note/gi_frames`. `note/rerun_of_low_clock,1` marks a run repeated because the first one was flagged `clock_low`; both are kept. Every composite row from 2026-11-05 is flagged `clock_low`: don't judge the 1.0 ms bar on them (PROGRESS, "Cool-start timing set").
- **`note/set,<n>` (shading.csv):**- **`note/set,<n>` (shading.csv):** the `bench-lock run` set (3 runs each) a group of rows belongs to, when a measurement was repeated because the first set's numbers jumped. All sets are kept.
- **The date column** is the UTC date the runner started, so a run in the early morning in Japan carries the day before.
- **The commit column** names the commit the CSV was written at, with `-dirty` only if a measured file had uncommitted changes.

## response.csv

From T2.4b (2026-11-06): how the GI-only view (and on Sponza the whole field) follows a light step at the calibrated batch, counted in frames (DDGI's clock 1/60 s a frame), from `ddgi-schedule-step` and `ddgi-schedule-sponza`. `response/<step>/units<n>/error_frame<k>` is the view's signed error against a field converged in the new state, at frame k after the step (frame 1 is the step's own frame; frames 1–30, then every 6th); `settled_frame` the first frame from which every measured frame is within 5% (-1: never); `overshoot_share_of_step` how far past settled the view went, as a share of the step. Sponza's `sun_turn30` rows add, every 6th frame, the field's light-weighted error (`weighted`, `weightedFar`), the error of its mean (`ofMean`) and the per-probe relative error (`mean`, `p95`, `near`, `far`), and `noise_floor/*`, the same measures for the field before the step against itself converged.

`response/native/<motion>/run<n>/<jump>/*` (T2.7, browser `native`, commit the fork's): the native harness playing the manifest motion (`play.ps1 -Motion light-step -Record`, `tools/native/response.ts`), every probe traced and blended every frame, the motion advanced a fixed 1/60 s a frame. The view is the whole bookmark view's GI-only mean (the harness's DDGI output, linear luminance). `<jump>` names the lamp's move (`jump1_x1_to_x6.2`: out of the viewed room). The measures are T2.4b's; `entered_frame` is the first frame inside 5%. "Settled" is a field converged with the lamp held at the new place for 3,600 frames (`play.ps1 -SettleKey`) when `settled_from_long_hold` is 1, otherwise the last 30 frames of the 3 s hold; `settled_end_drift` is how much those last 30 frames still move. The step holds only 3 s (180 frames), so a jump that never enters 5% reads `-1`. **Probe grid:**
- Rows under `response/native/<motion>/run<n>/` use the harness's own cells grid (two-room: 918 probes); they carry `grid_rule,cells`, a label row appended later.
- Rows under `response/native/<motion>/grid_ddgi/run<n>/` use our DDGI's grid (`play.ps1 -GridRule ddgi`, 1,071 probes). These are the rows to compare with ours.
- Commit `3d6cd8e-dirty` on the `grid_ddgi` rows: the fork's uncommitted change at the time touched only the texture sky's descriptor writes (350f9de). These runs use the flat sky, so it didn't change them.
- Don't set native's lamp-leaving jump beside our `lamp_halved` rows: a dimming is a different event (NOTES).

`response/ours/<motion>/run<n>/*` (T2.10, browser Chrome, commit this repo's): our viewer playing the same manifest motion (`tools/gpu/pages/viewer-recording`, saved by `save-viewer-recording.ts`), measured by the same `tools/native/response.ts`, so the step rows mean what native's do. Our DDGI's own grid at the viewer's default budget; `probes_per_frame_median` and `frame_interval_mean_ms` (the real time between frames, which our DDGI ages its histories by) say how it ran. `brightness` names the column: native's `gi_mean_luminance`, ours `gi_mean` (the meter's mean of red, green and blue). Re-measuring native's two-room step with the mean of red, green and blue gives the same frames. "Settled" is the same on both sides: the lamp held at the new place for 3,600 frames, then the mean of the next 61 (native `play.ps1 -SettleKey`, ours the page's `VITE_DDGI_SETTLE_KEY`).

**Sweep rows** (`.../run<n>/sweep/*`, both sides): the sweep goes out and back along the same path, so at time t and at the end minus t the lamp is in the same place.
- `lag_frames`: how far the view runs behind the lamp, in frames. It is half the shift that best lines the way back up with the way out (least squares).
- `out_back_gap`: the mean difference between the view on the way out and on the way back at the same lamp place, as a share of the view's mean.
- `lowest_share_of_start`, `highest_share_of_start`, `share_of_start_frame<k>` (every 12th frame): the curve as a share of its first frame, where the lamp sits settled at the path's start.
- `against_<side>_run<n>/*`: this run against the other side's run n, both as shares of their own starts. `mean_abs_difference_share_of_start` and `largest_...` are the mean and largest gap between the two curves; `lag_frames` is the shift that lines this side up best with the other (positive: this side later).

**Sponza's step: use the `lamp_lit` rows.**
- The rows to use are `response/native/light-step/grid_ddgi/lamp_lit/run<n>/*` and `response/ours/light-step/lamp_lit/run<n>/*`.
- Sponza's first T2.10 set (`.../grid_ddgi/run<n>/` and `response/ours/light-step/run<n>/` on `sponza`) is superseded. Each of those runs has a `superseded` row:
  - native's arcade lamp lit nothing then (`range: 0` became radius 0, NOTES);
  - our first runs were paired with those, and their pictures were at 0 EV. Their measures were read before exposure and stand on their own.
- `response.ts --set <name>` puts the set name in the path, so a re-recorded set doesn't share the old one's paths.

## native.csv

NVIDIA's RTXGI-DDGI test harness (our fork in `native/rtxgi-ddgi`), written by `tools/native/bench.ps1`. Metric names and what they mean: `tools/native/README.md`, section "Timing".

- **`commit`** is the **fork's** commit (the harness that was measured), not this repo's. `-dirty` means the fork or `tools/native/` had uncommitted changes (those scripts write the harness `.ini`, so they shape the run too). Rows written before that rule existed (all `9bd7f3c` and `1df1f97` rows) were also taken with `tools/native/` uncommitted, but only the imported sweep rows say `-dirty`.
- **`browser`** is `native` for the Vulkan build and `native-d3d12` for the D3D12 build.
- **One run** is the rows with the same `date, commit, device, browser, scene`. If a key has more than one run (the Sponza sweep rows under `1df1f97-dirty`), each run is a block of consecutive rows that starts at `probes_x`.

**Which rows are valid timings.** Keep a run only if it has `probe_update_every_frame,1` and **no** row `probe_update_every_frame,0` and no `sweep,1`:

- `probe_update_every_frame,0`: the probes were not updated every frame (upstream behaviour: a converged volume stops tracing), so `gpu_ms_probe_trace` and `gpu_ms_blend` are averages over mostly empty frames. All `9bd7f3c` runs are like that. Their marker rows were appended later, at the end of the file (the file is never edited), so for those match the key, not the block.
- `sweep,1`: a run with a non-default ray count (`rays_per_probe`) or probe spacing. Valid timings, but not the default configuration; compare them with each other, not with the default rows.

For example, default Vulkan runs with valid timings:

```powershell
$rows = Import-Csv bench/results/native.csv
$bad = $rows | Where-Object { ($_.metric -eq 'probe_update_every_frame' -and $_.value -eq '0') -or $_.metric -eq 'sweep' } |
    ForEach-Object { "$($_.date),$($_.commit),$($_.device),$($_.browser),$($_.scene)" }
$rows | Where-Object { $_.browser -eq 'native' -and "$($_.date),$($_.commit),$($_.device),$($_.browser),$($_.scene)" -notin $bad }
```

## timing.csv

- **`batch_cost/inline/...` and `batch_cost/deferred/...` (from 2026-11-06, commit 06d5a18 on the spike branch `m2-t2.0`):** T2.0's A/B of where DDGI computes direct light, M1's inline shadow rays or the deferred `gi/direct` pass. The path sits after `batch_cost/`; the rest of the name is the `ddgi-batch-cost` page's usual metric (`tools/gpu/pages/ddgi-batch-cost/to-csv.ts`). New with them: `bound_batch<n>/...` (the GPU-bound curve from 16 to 512 probes, each pass's median ms), `bound_fit/...` (fixed ms, µs per probe, the probes at the budget from the fit and from the curve) and, deferred only, `shadow_rays/batch<n>/...`. Each path is one run with its own `note/timing_valid`.

- **`t2.10/...` rows (2026-11-08, T2.10's M2 measurements):** pages with no to-csv of their own, flattened with `tools/gpu/flatten-to-csv.ts` (every number in the page's data, named by its path). `t2.10/adapter/...` is `three-adapter-timing`, `t2.10/set_transforms_in_frame/...` its `set-transforms` mode, `t2.10/key_probes_cost/...` `ddgi-key-probes-cost` (with `stepFirst60/...`, the 60 frames after Sponza's sun turn), `t2.10/light_grid_cost[_ramp]/...` `ddgi-light-grid-cost`. `batch_cost/ramp/...` rows are `ddgi-batch-cost`'s `VITE_DDGI_RAMP_SECONDS` re-calibration. Each run has its own `note/timing_valid`.
- **T2.17 rows (2026-11-08, Bistro exterior's GI-on frame):** in `baselines.csv`, two `viewer-baselines` blocks marked `note/t2.17_bistro_frame` (1440p, DDGI only, 5 rounds of 20 frames). One has textures as loaded, and one has `view/max_texture_px` 1024 (textures scaled down at load). In `timing.csv`, two `ddgi-batch-cost` blocks (start-up and budget) marked `note/t2.17_bistro_calibration`, one with `note/t2.17_max_texture_px`. Bistro's textures take 5.1 GB as loaded, more than the 6 GB RTX holds with DDGI (NOTES), so the as-loaded blocks show the overflow, not DDGI's cost. Their commit reads `9ed2950-dirty` or `9ed2950`: the texture cap wasn't committed yet.
- **`t2.18/key_probes_cost/...` rows (2026-11-08, T2.18):** `ddgi-key-probes-cost` with the second count after a single step, flattened the same way (one VALID run on 3c93a29). `stepFirst60/batch*` is the batch those frames listed, and `stepFirst60/firstFramesPasses/<frame>/<pass>` the first 20 frames pass by pass. In response.csv, `response/t2.18/key_probes_sponza/...` (the turn) and `response/t2.18/sky_sun_step/...` (the furnished room's sun step) are the same code's response runs.

## checks.csv

From T2.10 (2026-11-08): correctness checks with numbers (recompile counts, leaks, light groups, the still-scene checks, accuracy against the Cycles references), one block per run, metrics `t2.10/<page>/...`. The accuracy rows (`t2.10/accuracy/...`) were copied from `ddgi-test-view/render.ts` and `ddgi-sponza-accuracy/run.ts` output, which have no CSV route; `note/correctness_run,1` marks rows whose clock doesn't matter.

## memory.csv

`gi.stats.memory` per scene at the manifest's volume, from the `ddgi-memory` page (`tools/gpu/pages/ddgi-memory`, T1.9). Bytes unless the name ends in `_mb` (1 MB = 1,000,000 bytes, as the viewer shows them). Byte counts of what the core created, so they don't depend on the GPU.

- `memory/scene_copy_*`, `acceleration_*`, `ddgi_*`, `total_*`: the four numbers of `gi.stats.memory`.
- `memory/shading_sky_*`: the shading module's sky (radiance map and the procedural sky's source texture). **`gi.stats.memory.total` does not include it**; `memory/everything_*` is the total plus it, and is the number to compare with SPEC's 64 MB.

## flicker.csv

Frame-to-frame change of the GI image on a still, converged scene, from the `ddgi-flicker` page (T1.9). Definitions are at the top of `tools/gpu/pages/ddgi-flicker/main.ts`. The scene column is `<scene>/<camera bookmark>`; 3 runs give `run1/`..`run3/` and `median/` rows.

- Metrics are `flicker/<budget>/<name>`; the budget is `gpuMs2` (the viewer's 2 ms budget) or `units256` (a fixed 256 probes a frame).
- `rel_change_mean` and `rel_change_p95`: per pixel `|GI(t) - GI(t-1)| / mean(GI)` over pixels that show a surface; the mean and the 95th percentile over pixels, averaged over the measured frame pairs. `1` means 100%.
- `probes_per_frame_mean`: what the budget gave. With `gpuMs2` it varies with the GPU's speed that second, so compare flicker with this number beside it.
- `image_mean_drift`: the whole image's mean, last 10 frames against first 10: a check that the scene was converged.
- The GI image is the engine-free test view's indirect image (albedo / pi times the field), 640 x 360, so it holds no noise except the probes'.
