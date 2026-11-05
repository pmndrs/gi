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
- **`note/sm_clock_mhz` (every CSV, NVIDIA rows only):** the RTX's SM clock while that run was going: the median over the nvidia-smi samples taken under load (70% GPU use or more). With fewer than three such samples the note is the median over ALL samples instead, idle seconds before and after included, so it reads low. That happens for short pages, for latency-bound ones (`tlas-bench`, `ddgi-timing`: nvidia-smi never showed 70%) and for the native harness (nvidia-smi shows only 35-45% while it runs minimized). Then `note/sm_clock_max_mhz` is the better sign of what the card reached, sampled about once a second by `node tools/reference/bench-lock.ts run`. Companion rows: `note/sm_clock_min_mhz`, `note/sm_clock_max_mhz` and `note/mem_clock_mhz` (same samples). The RTX 3060 Laptop's maximum is 2100 MHz. Short loads on a cool card read 1,400 to 2,000 MHz; a sustained GPU-bound load (more than about 20 s) heats it to the 87 C target and the clock falls to 450 to 600 MHz (see `note/clock_low`). The nvidia-smi values refresh only about every 0.6 s, so the once-a-second sampling loses nothing, but the first loaded second of a burst still shows the ramp from the idle clock. One note set belongs to one `bench-lock run`, which can cover several repeated runs, so the clock is for the whole group, not each run. Rows without the note were taken before 2026-11-05 or outside `bench-lock run`; `note/sm_clock_locked_mhz,1200` marks the clock locked by hand (NOTES). The converters read the summary file from `--clock-file`, or from DDGI_CLOCK_FILE, which `bench-lock run` sets for its command.
- **`note/clock_low,1` (every CSV, NVIDIA rows only):** the run's SM clock was low, so its milliseconds are slow-clock numbers (the rows stay; the file is never edited). The rule, from `isClockLow` in `tools/gpu/clock-note.ts`: among the nvidia-smi samples under load (GPU use 70% or more), the median SM clock is **below 1,500 MHz**, with at least three such samples. A run with fewer than three loaded samples (short pages, latency-bound pages, the native harness) gets no verdict, so no flag, because its all-sample median includes idle seconds and would read low for no reason. `bench-lock run` also prints a `WARNING: the GPU clock was LOW under load` line at the end, with the share of loaded samples where nvidia-smi named thermal throttling or the power limit and the GPU temperature; the to-csv scripts print it too. Why 1,500: the RTX 3060 Laptop reaches 2,000 MHz or more within seconds of a cool start and holds 1,400 to 2,000 while cool, but under a sustained GPU-bound load it heats to its 87 C target in about 20 s and then settles at 450 to 600 MHz (measured 2026-11-05; see PROGRESS, "GPU clock under sustained load"). Many of those runs are 3 to 4 times slower than a cool card's. Compare flagged rows only with other flagged rows, or re-time after the GPU has cooled (under about 70 C) with a shorter timed section. Idle gaps do not cause a flag: they are below 70% use and are not counted.
- **`note/set,<n>` (shading.csv):** the `bench-lock run` set (3 runs each) a group of rows belongs to, when a measurement was repeated because the first set's numbers jumped. All sets are kept.
- **The date column** is the UTC date the runner started, so a run in the early morning in Japan carries the day before.
- **The commit column** names the commit the CSV was written at, with `-dirty` only if a measured file had uncommitted changes.

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
