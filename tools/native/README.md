# tools/native: NVIDIA RTXGI-DDGI test harness as a comparison tool

The harness (`native/rtxgi-ddgi`, our fork on branch `m0-harness`) is a **comparison and timing tool only**. It is under NVIDIA's RTX SDK license. Nothing from it is copied into the library packages, and these scripts only write its `.ini` and start it.

Machine this was set up on: Windows 11, RTX 3060 Laptop GPU (the harness needs hardware ray tracing, so the Intel UHD is skipped automatically).

## Run a scene

From the repo root, in PowerShell 7:

```powershell
./tools/native/play.ps1 -Scene sponza -Clean              # Sponza at bookmark 1, DDGI on, no probe spheres / RTAO / dither
./tools/native/play.ps1 -Scene two-room -Camera 2         # bookmark 2 of the two-room scene, default look
./tools/native/play.ps1 -Scene sponza -Clean -Api d3d12   # the D3D12 build instead of Vulkan
./tools/native/play.ps1 -Scene cornell-box -Gpu 3060      # pick the GPU by part of its name
./tools/native/play.ps1 -Scene sponza -Clean -CaptureFrames 300   # no person needed: 300 frames, save a PNG, exit
```

- `-Scene` is a scene id from `scenes/manifests/` (`two-room`, `cornell-box`, `sponza`, `bistro-exterior`, `bistro-interior`, and the other procedural ones).
- `-Camera n` is the manifest bookmark number, **1 = the first bookmark** (default). The harness lists all bookmarks in its Camera panel.
- `-Clean` turns off probe spheres, ray traced ambient occlusion and dither. Use it whenever you compare images.
- `-Api vulkan` (default) or `d3d12`. `-Gpu <text>` or the environment variable `RTXGI_GPU` picks the GPU by part of its name, any case. With neither, the first GPU with ray tracing is used.
- `-Width`, `-Height` (default 1600x900), `-Rays` (default 256), `-NoVsync`.
- `-NoWait`: see "The GPU lock" below.

The script writes `native/build/runs/<scene>-<api>[-clean]/<scene>.ini` and runs the harness there, so `log.txt`, `device-selection.txt` and `screenshots/` are all in that folder. The window stays open until you close it.

### The GPU lock

`play.ps1` holds the shared GPU lock (`%TEMP%\ddgi-gpu-bench.lock`) for as long as the harness window is open, so no benchmark runs on the card while you look at a scene. If another job holds the lock, the script says which one and waits; it opens the scene by itself when that job is done. `-NoWait` exits instead of waiting. Closing the window gives the lock back; so does Ctrl-C in the console, which also closes the harness window it opened.

All native scripts use the same lock tool as the browser benchmarks (`node tools/reference/bench-lock.ts`, see `tools/reference/README.md`). It writes an owner file with the PowerShell process id, so `node tools/reference/bench-lock.ts status` shows who holds the lock, and a lock left behind by a closed console is taken over automatically.

The scripts also work under `node tools/reference/bench-lock.ts run --what "..." -- pwsh -File tools/native/bench.ps1 ...`: `run` sets `DDGI_BENCH_LOCK_HELD` to its own pid, and the scripts then use that hold instead of waiting for it. Every script closes the harness it started (by its process id, never by name) before it gives the lock back, also on Ctrl-C.

Keys in the harness: F2 dumps DDGI textures, F4 runs the 1024-frame benchmark, J toggles the performance overlay, R reloads shaders and config, M switches between DDGI and the path tracer, U hides the settings panel.

### What the manifest becomes

| Manifest | Harness `.ini` |
|---|---|
| `parts[0]` model, or the procedural scene's `assets/generated/<id>.glb` | `scene.path` / `scene.file` |
| `cameras[]` (position, target, fovYDeg) | `scene.cameras.N` as yaw, pitch, position, fov (vertical) |
| `lights[]` directional | type 0; `power` = intensity (lux), `direction` = travel direction |
| `lights[]` point | type 2; `power` = intensity (candela), `radius` = range |
| `lights[]` spot | type 1; angles in radians (outer = umbra, inner = penumbra) |
| `environment` | `none` is black; `color` is a flat colour; `procedural` is the shared sky file's flat equivalent (`flatEquivalentRadianceRgb` in `scenes/sky/<scene>.json`: the uniform radiance that lights a flat open ground like the file does). The harness's shaders take one constant sky radiance (`app.skyRadiance`) for every missed ray, not an image, so it can't read the sky file itself. Nor can it tell a ray going up from one going down without a shader change, so it also returns this colour below the horizon, where the file is black: in open scenes (Bistro exterior, the courtyard, open ground), rays that leave the model downwards pick up sky light that the other renderers don't have. M1 keeps this; a texture sky in the fork's miss shader is an M2 sky and time-of-day item |
| `volume` | `ddgi.volume.0`, see below |

Biases scale with the smallest probe spacing `s`: normal bias `0.1 s`, view bias `0.3 s`, relocation minimum front-face distance `0.3 s`. Everything else in the DDGI block is the upstream Sponza sample: relocation, classification and variability on, no scrolling, hysteresis 0.97, 8x8 irradiance and 16x16 distance texels per probe.

### Probe grid (the rule the browser side must copy)

Probes sit at the **centres of equal cells that cover the manifest volume exactly**, the same layout as three.js `LightProbeGrid`:

```
count[axis]   = ceil(size[axis] / probeSpacing)
spacing[axis] = size[axis] / count[axis]        (so it can be a little smaller than probeSpacing)
origin        = volume.center
```

### Scenes the harness cannot read directly

- **A `.glb` with textures** is refused by the harness ("GLB format not supported for scenes with texture data"). The Bistro files are like that. `play.ps1` splits them once with `glb-to-gltf.mjs` into `native/build/scenes/<id>/` (a `.gltf`, a `.bin` and the images). Nothing in `assets/` is touched. While doing that the script also fixes two more things the harness cannot handle in the Bistro files:
  - **WebP textures** (Blender 4.5 wrote `EXT_texture_webp`): converted to PNG with ImageMagick (`magick` must be on PATH). Without this the harness silently finds no textures and renders black.
  - **Normal maps without tangents**: the harness needs a `TANGENT` attribute to use a normal map and shades garbage (black and purple, NaN probes) without one. If a file has no tangents at all, the staged copy drops its normal maps. The current Bistro exports (2026-11-04, 14:00) have tangents, so their normal maps are kept.
- **Model transforms.** The harness cannot place a model, so a scene whose model part has a non-identity transform (`furnished-room`) is refused with a clear message. Scenes with more than one part (`courtyard-tree`) are refused too.
- **File names** must be letters, digits, `_` and `-` only (`NewSponza_Main_1k.gltf` is fine).
- The first load of a textured scene converts every texture to BC7 on the CPU, one thread, with no progress bar, and writes a cache to `native/build/cache/<id>/` (Sponza: about 2 minutes; Bistro: see the measured times below). The scripts do this first by themselves in a **minimized** window under the GPU lock (`play.ps1`, `bench.ps1` and `prepare-cache.ps1 -Scene a,b` all call the same step, one scene at a time), so an interactive window never has to. The harness now keeps its window answering Windows during this, but if you ever see "Not Responding", do not close it. Delete the cache folder if you change the scene file. Never run two Bistro loads at once: they use most of the 6 GB card.
- If `assets/downloads/<asset>/<file>` is missing, the script falls back to `assets/downloads.bak/` and says so.

## Timing: `bench.ps1`

```powershell
./tools/native/bench.ps1                                   # the default scene list, Vulkan, 256 rays, RTX 3060
./tools/native/bench.ps1 -Scene sponza -Api d3d12
./tools/native/bench.ps1 -Scene two-room -NoAppend         # print only, leave bench/results/native.csv alone
```

Per scene it writes a clean-mode config (1920x1080, vsync off, 256 rays per probe, grid from the manifest), **takes the shared GPU lock once for the whole run** (`%TEMP%\ddgi-gpu-bench.lock`; it waits while another job holds it, and gives it back at the end, also on an error or Ctrl-C), builds the scene cache if it is missing, waits 300 frames for the probes to settle, records the harness's own 1024-frame benchmark (the same code as F4: `app.autoBenchmark=1` in the `.ini` runs it by itself and exits), and appends that scene's rows to `bench/results/native.csv` as soon as the scene is done. A scene that fails is reported and skipped; the scenes before it stay in the CSV, the ones after it still run, and the script exits with code 1 at the end.

**Probes are updated every frame** in these runs (fork option `probeVariability.skipConverged=0`). Upstream stops tracing and blending a volume once it has converged; rows from before fork commit `5b8cfac` (commit `9bd7f3c` in the CSV) were taken that way, and in them only about 16 to 80 of the 1024 frames had any probe work, so their `probe_trace` and `blend` means are not per-frame costs. Do not use those rows. `-SkipConverged` reproduces the upstream behaviour if you need to show it.

CSV conventions (columns, and how to pick valid rows: `bench/results/README.md`, section "native.csv"): `browser` is `native` for Vulkan and `native-d3d12` for D3D12; `commit` is the fork's commit, with `-dirty` when the fork or `tools/native/` has uncommitted changes (the scripts write the `.ini`, so they shape the run too). Per run:

| metric | meaning |
|---|---|
| `probe_count`, `probes_x/y/z` | the grid |
| `probes_active`, `probes_inactive` | probe states at the end of the run (classification), counted from the probe-data texture the benchmark dumps |
| `rays_per_probe` | rays per active probe |
| `rays_dispatched_per_frame` | `probe_count x rays_per_probe`: ray generation threads launched |
| `rays_traced_per_frame` | `active x rays + inactive x min(32, rays)`: inactive probes trace only the 32 fixed rays that classification needs |
| `probe_update_every_frame`, `frames_probes_updated` | 1 for normal runs; how many of the 1024 frames really ran the probe trace (the script refuses to record a run below 99%, or when benchmarkGpu.csv has no "Probe Trace" column) |
| `sweep` | 1 when `-Rays` is not 256 or `-ProbeSpacing` is given; absent for default runs |
| `gpu_ms_<pass>`, `gpu_ms_<pass>_median` | the harness's GPU timestamps around that pass, mean and median over the 1024 frames |
| `cpu_ms_frame`, `cpu_ms_frame_median` | CPU time of one main-loop iteration (see below) |
| `mrays_per_s` | `rays_traced_per_frame / gpu_ms_probe_trace_median`, in millions per second |

What the passes are, in the harness's own timers: `frame` (whole command buffer), `gbuffer`, `ddgi` (all probe work plus the screen-space gather), and inside it `probe_trace`, `blend`, `relocate`, `classify`, `variability`, `lighting` (the screen-space gather that applies the probes to the image).

### What differs from the browser side

- **Hardware ray tracing** (Vulkan ray tracing pipeline, or DXR) against a real acceleration structure. The browser side traces in software in compute shaders. Native trace times are therefore not a target to hit, they are a reference point.
- **Recursion / albedo cap.** NVIDIA caps the albedo at **0.9** when a probe ray hit is lit with the volume's own previous irradiance (`ProbeTraceRGS.hlsl`), and at 0.9 per bounce in the path tracer. The browser side must use 0.9 too, or its indirect light is about 23% off (see docs/NOTES.md).
- One ray bounce per probe ray plus a lookup in the volume's own irradiance (that is how it gets more bounces over time). There is no recursion limit setting beyond that.
- Probes the classification marks inactive trace only 32 fixed rays and are not blended, so use `rays_traced_per_frame`, not `rays_dispatched_per_frame`, when you compare ray throughput.
- **CPU frame time is not GPU time.** On Sponza and Bistro the harness's main loop takes 35 to 90 ms per frame while the GPU works 20 to 36 ms of it (the harness process uses one CPU core at 100%; the NVIDIA 3D engine is busy about 25% of the time). Vsync is off and present and the wait for the GPU take under 0.2 ms, so the time is CPU work on the main thread that the harness's own CPU timers miss: each pass's `Update()` starts its CPU timer and `Execute()` restarts it, so all `Update()` time (including building the ImGui settings windows) is dropped from the per-pass CPU numbers. The frame rate of the harness is therefore CPU-bound; use the GPU numbers.

## Build

Built with: Visual Studio 2022 Community 17.10 (MSVC 14.40.33807), Windows SDK 10.0.22621.0, Vulkan SDK 1.3.283.0, CMake 3.24.0. Both targets are in one build tree:

```powershell
# one time: the harness's own submodules, and DXC / D3D12 Agility SDK in native/rtxgi-ddgi/external
git -C native/rtxgi-ddgi submodule update --init thirdparty/glfw thirdparty/tinygltf thirdparty/imgui
# (CMake downloads external/dxc and external/agilitysdk on first configure; they are git-ignored)

$env:CMAKE_TOOLCHAIN_FILE = $null   # a stale vcpkg toolchain variable breaks the configure
cmake -S native/rtxgi-ddgi -B native/build/rtxgi -G "Visual Studio 17 2022" -A x64 `
  -DRTXGI_API_D3D12_ENABLE=ON -DRTXGI_API_VULKAN_ENABLE=ON -DRTXGI_COORDINATE_SYSTEM="Right Hand, Y-Up" `
  -DRTXGI_API_D3D12_DXIL_PATH="C:/Program Files (x86)/Windows Kits/10/bin/10.0.22621.0/x64"
cmake --build native/build/rtxgi --config Release --target TestHarness-VK TestHarness-D3D12 --parallel 8
```

Executables: `native/build/rtxgi/samples/bin/vulkan/Release/TestHarness-VK.exe` and `.../d3d12/Release/TestHarness-D3D12.exe`. A full build takes a few minutes; run it in the background.

## What changed in the fork (all opt-in, default behaviour unchanged)

- `app.gpu=<text>` / env `RTXGI_GPU`: pick the GPU by name. It replaces the hard-coded "RTX 3060" check in `Vulkan.cpp`, and also works for D3D12.
- `scene.path` may be absolute; `scene.cacheDir` moves the scene cache; `scene.activeCamera` picks the starting camera.
- `app.autoBenchmark=1`, `app.benchmarkWarmupFrames=N`: run the F4 benchmark by itself and exit.
- `ddgi.volume.N.probeVariability.skipConverged=0`: keep tracing and blending a volume every frame after it has converged. Upstream (and the default) stops updating a volume once its measured variability is under the threshold. `bench.ps1` sets it, because otherwise almost every benchmark frame has no probe work in it.
- `app.ignoreInput=1`: no keyboard or mouse input. The scripts set it for every unattended run (benchmarks, `-CaptureFrames`, cache builds). GLFW gives a new window the keyboard focus even when it starts minimized; one test run switched to the path tracer during warm-up, most likely from a key typed into another window (`bench.ps1` caught it: no probe trace in any frame; a repeat of the same run did not switch).
- Vulkan: the `ByteAddrBuffer[]` descriptor array (an index and a vertex buffer per mesh, plus 4) is sized from the scene instead of a fixed 2048. Upstream it overflowed for scenes with more than 1022 meshes, which lost the GPU on our Bistro exports. Scenes up to 1022 meshes get exactly the old size.
- The log now says `Graphics API: ..., device: ...`.

### Checking that the default config is unchanged

Upstream (`059dd3c`) is checked out as a git worktree of the fork in `native/build/upstream-src` (its `thirdparty/` and `external/` copied from the fork) and built into `native/build/upstream` (Vulkan only). Both harnesses run the unmodified `samples/test-harness/config/two-rooms.ini` of their own tree, minimized, with `DDGI_CAPTURE_FRAMES=600` (600 frames, save the back buffer, exit), under the lock. Crop off the settings panel (x < 1400; it shows live frame times), convert to linear EXR (`magick in.png -alpha off -crop 1400x1080+0+0 +repage -colorspace RGB -depth 32 -define quantum:format=floating-point out.exr`) and compare with `node tools/compare/index.ts`.

Result on 2026-11-04 (fork `c7d0162`, RTX 3060 Laptop on both, picked by upstream's hard-coded "RTX 3060" filter and by the fork's "first GPU with ray tracing"): 5 captures per build. The direct sunlight is identical; only the indirect light differs, as smooth probe-shaped blobs, at most 4 of 255 in 8 bits. That is run-to-run noise. This ini has probe variability on with threshold 0.03, and the volume's variability sits at about 0.029, so the volume stops updating on whichever frame the noisy value first dips under the threshold. Upstream against upstream: mean relative error 1.30% on average (0.88 to 2.12%). Fork against fork: 1.55% (0.94 to 2.51%). Upstream against fork: 1.44% (0.93 to 2.53%). Mean brightness: the fork is 0.59% darker, with a standard deviation of about 0.9% per capture.

## Known problems

- **Fixed: Bistro lost the GPU on Vulkan** (`VkResult=-4`, device lost, first frame). The Vulkan harness has one descriptor array for the mesh index and vertex buffers, fixed at 2048 entries (4 + 2 per mesh). The Bistro exports have 1186 (interior) and 1296 (exterior) meshes, which need 2376 and 2596 entries, so the scene's descriptor write ran past the end of the array. The fork now sizes it from the scene (the log says so when it is larger than 2048). D3D12 sizes its heap from the scene already.
- **D3D12, Bistro exterior: an "Error!" box after the benchmark.** The 1024-frame benchmark completes and writes its CSVs; then the harness's own texture dump fails on the probe-variability-average texture. That texture is two-channel float (8 bytes per texel) but the dump passes it to WIC as four-channel float (16 bytes per texel) with a 256-byte row pitch, which only fits up to 16 texels wide; this volume's is 17. It is an upstream bug in a debug dump, not a GPU fault. `bench.ps1` records such a run (with a warning) only if the log shows the benchmark finished. The box is a normal Windows message box and may show on screen for a few seconds before the script closes the harness.
