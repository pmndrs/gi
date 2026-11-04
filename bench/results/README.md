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
