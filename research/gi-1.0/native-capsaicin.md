# AMD Capsaicin as a native GI comparison tool

Date: 2026-11-09. Status summary:

| Stage | Status |
| --- | --- |
| Built | Yes. Release x64, 1 min 44 s build, 0 errors. |
| Launched | Yes, on the NVIDIA RTX 3060 Laptop GPU (confirmed). |
| Captured | Yes. One GI image and one reference path tracer image of the same Sponza view, plus per-pass GPU timing CSVs. |
| Our own glTF scenes (Sponza-with-curtains, Bistro, two-room) | Yes. All four load and render with manifest cameras and lights, GI-1.2 and the reference path tracer captured and timed for each. See "Our scenes in Capsaicin" at the end. |

## What it is

Capsaicin is AMD GPUOpen's research rendering framework (D3D12 and DXR 1.1 only, Windows only). It ships a viewer app called `scene_viewer` with two renderers:

- **GI-1.2** (internal name; the default renderer string in the viewer source is "GI-1.1"). This is the lineage of GI-1.0: two-level radiance caching with screen-space probes plus a world-space radiance cache, ReSTIR-style reservoir light sampling, and a denoised ray-traced reflection path. The tag we pinned is v1.3 and its renderer is GI-1.2, not literally "GI-1.0". The README links the GI-1.0 technical report and the GI-1.1 reflections paper. Anything we say about "GI-1.0" should say "GI-1.0 lineage as shipped in GI-1.2".
- **Reference Path Tracer**: a brute-force accumulating path tracer (inline ray queries, optional DXR 1.0 pipelines). Defaults: 30 bounces, Russian roulette after 2, 1 path per pixel per frame, accumulation on.

There is no other renderer. The core also contains SSGI, FSR upscaling, bloom, auto exposure, tone mapping and similar techniques that the two renderers compose.

## License

- Capsaicin: MIT, "Copyright 2023 Advanced Micro Devices, Inc." (verified in `LICENSE`).
- `third_party/gfx` (the D3D12 helper library, submodule): MIT, Copyright 2024 Guillaume Boisse (verified).
- Dependencies fetched at configure time: CLI11, yaml-cpp, nlohmann-json, meshoptimizer, AMD FidelityFX SDK v2.0.0, CapsaicinTestMedia v1.2, plus (inside gfx) D3D12MemoryAllocator, DirectX Agility SDK, DirectX Shader Compiler, WinPixEventRuntime, imgui, stb, cgltf, glm, tinyobjloader, tinyexr, KTX, Vulkan-Headers. Their licenses were not individually audited. From general knowledge they are MIT/BSD/Apache-style, with Microsoft's own terms for the Agility SDK and PIX runtime. Treat this as unverified before any redistribution. We only run these locally and do not redistribute anything.
- Test scenes: each scene folder carries its own license text (e.g. `sponza/SponzaLicense.txt`, `gas_station/GasStationLicense.txt`). Not reviewed.

## Pinned source

- Capsaicin: `914b91596cd119eda85fbc1d3c7ee6ac391b1452`, tag `v1.3` (also upstream master HEAD on 2026-11-09). Submodule at `native/capsaicin`, URL `https://github.com/GPUOpen-LibrariesAndSDKs/Capsaicin.git`, registered in `.gitmodules`.
- Its own submodule `third_party/gfx`: `c65d6172500864b68988cdf64846520ebbb42e6f`.
- Configure-time fetches (pinned by tag in Capsaicin's CMakeLists): FidelityFX SDK `v2.0.0`, CapsaicinTestMedia `v1.2`; gfx fetches D3D12MemoryAllocator v3.0.1, DXC, Agility SDK, etc.
- Not committed. The fetched content lands inside the submodule working tree (`third_party/FidelityFX-SDK`, `assets/CapsaicinTestMedia`, `third_party/gfx/third_party/*`); I added the first two plus `dump/`, `shader_pdb/` and `imgui.ini` to the submodule's local `info/exclude` so it reports clean. That exclude file lives in the main repo's `.git/modules` and does not travel with a fresh clone, so on a fresh clone the submodule will show untracked content after configure. This is harmless.

## Requirements and sizes

- Windows 10 20H2+ with a D3D12 Ultimate capable GPU (DXR 1.1). The README lists Windows 10 SDK 2004+ and **CMake 3.30 or newer**. It is not AMD-only: it uses standard DXR 1.1 and ran on the RTX 3060. FSR is built in, and the AMD frame-generation DLLs are shipped but not needed.
- Machine CMake is 3.24.0, too old. I downloaded the official portable `cmake-3.31.6-windows-x86_64.zip` (46 MB, from the Kitware GitHub release) into `native/build/tools/` and used it locally. Nothing installed globally, no drivers touched. Ninja is not installed, so the Visual Studio 17 2022 generator was used (README also documents that path).
- Sizes: source checkout incl. gfx 75 MB; test media 727 MB (git clone of CapsaicinTestMedia, no LFS); FidelityFX SDK 677 MB; build output 153 MB. Total well under the 5 GB stop line. No git LFS involved.
- Test media downloads happen automatically at configure time (`CAPSAICIN_DOWNLOAD_TEST_MEDIA=ON`). Needs network at configure.

## Exact commands

From the repo root (PowerShell). Run these with the stale vcpkg env removed (see problems):

```powershell
$root = (Get-Location).Path
$cm = "$root\native\build\tools\cmake-3.31.6-windows-x86_64\bin\cmake.exe"
Remove-Item Env:CMAKE_TOOLCHAIN_FILE, Env:VCPKG_ROOT -ErrorAction SilentlyContinue
$env:GIT_CONFIG_COUNT='1'; $env:GIT_CONFIG_KEY_0='core.longpaths'; $env:GIT_CONFIG_VALUE_0='true'
git submodule update --init --recursive native/capsaicin
& $cm -S native/capsaicin -B native/build/capsaicin -G "Visual Studio 17 2022" -A x64
& $cm --build native/build/capsaicin --config Release --parallel
```

Result: `native/build/capsaicin/bin/Release/scene_viewer.exe` (plus capsaicin.dll, DXC, Agility D3D12Core, FSR DLLs).

Launch via `tools/reference/capsaicin/run-capsaicin.ps1` (see below), or by hand from the Capsaicin source directory (scene paths are relative to the working directory):

```powershell
Set-Location native\references\capsaicin
..\..\build\capsaicin\bin\Release\scene_viewer.exe --width 1280 --height 720 --start-scene-index 4 --start-camera-index 0 --start-renderer-index 0 --benchmark-mode --benchmark-frames 120 --benchmark-capture-frames true --benchmark-capture-timings true --dump-folder <abs-path>/
```

## Build time

- Build: 1 min 43.6 s wall (Release, MSBuild parallel). Logs: `artifacts/native/capsaicin/build.log`.
- First configure: about 3 min 17 s, dominated by cloning FidelityFX SDK, the test media and the gfx dependencies. A warm re-configure takes 27 s. The saved `configure.log` is the final successful warm re-run; the failed first attempts were overwritten.

## What ran and on which adapter

Machine has the RTX 3060 Laptop and Intel UHD. The app enumerates with `EnumAdapterByGpuPreference(HIGH_PERFORMANCE)` and printed `Created Direct3D12 device 'NVIDIA GeForce RTX 3060 Laptop GPU'` (from `gfx.cpp`) when launched from a bash console. Independent confirmation during a 3000-frame run: Windows GPU Engine counters showed 99.7% 3D engine use on the 3060's LUID (0x00014256), 2.2 GB dedicated memory for the process, and `nvidia-smi` reported 100% utilisation. There is no command-line adapter switch; if Windows ever routes it to Intel, use Windows Graphics Settings. I did not change any setting.

Runs (all Sponza, 1280x720, camera index 0, env map Kiara Dawn, Release):

- GI-1.2, 120 frames: average GPU time about 7.1 ms for the GI-1.2 block over a 2900-frame steady sample (min 5.2, max 10.0), visibility buffer 1.6 ms, SSGI 0.45 ms, FSR 1.2 ms. Whole benchmark including startup took 15 s.
- Reference Path Tracer, 1000 accumulated frames: about 13.8 ms per frame (1 path per pixel), run took 12 s.

Note on the timings: the GI-1.2 numbers include the full pipeline at 720p with FSR in the chain (FSR is part of the default renderer). They are real GPU timestamps but come from a loaded laptop GPU while other apps held the GPU, so treat as indicative only.

## What the viewer offers

- Renderers (index): 0 = GI-1.2, 1 = Reference Path Tracer. Confirmed through the saved output names and CSV pass lists. (The `--list-*` flags print only to a parent console; they printed nothing when I invoked them through this tool, so the lists come from the source.)
- Bundled scenes (index): 0 Breakfast Room, 1 Flying World (Battle of the Trash God), 2 Gas Station, 3 Tropical Bedroom, 4 Sponza. The bundled Sponza has the curtains, same family as our scene.
- Environment maps (index): 0 None, 1 Photo Studio London Hall, 2 Kiara Dawn, 3 Nagoya Wall Path, 4 Spaichingen Hill, 5 Studio Small, 6 White, 7 Atmosphere (procedural sky).
- Arbitrary scenes: yes. `--load-external-scene <path>` overrides the built-in list; the code accepts `.gltf`, `.glb`, `.obj` and `.hdr` environment maps by drag-and-drop (Shift appends a scene). I verified `--load-external-scene` with an absolute path to the bundled Sponza.gltf through the launcher script. Our own scenes are covered in "Our scenes in Capsaicin" below. Unknown: KHR extension support, scene scale, and whether our emissive/light setup carries over. Check that when we try them.
- Command line (CLI11): `--width`, `--height`, `--render-scale`, `--full-screen`, `--display-hdr`, `--start-scene-index`, `--start-environment-map-index`, `--load-external-scene`, `--start-renderer-index`, `--start-camera-index`, `--user-camera-position x y z`, `--user-camera-lookat x y z`, `--start-playing`, `--render-options "name=value ..."`, `--save-as-jpeg`, `--dump-folder`, `--benchmark-mode`, `--benchmark-frames N`, `--benchmark-first-frame T [I]`, `--benchmark-suffix`, `--benchmark-capture-frames bool`, `--benchmark-capture-timings bool`, `--list-scenes|environments|renderers`, `--version`, `--help`. Booleans take an explicit `true`/`false` value. Benchmark mode runs unattended, fixed frame rate, and exits by itself.
- Interactive UI: scene/env/camera/renderer dropdowns, animation controls, every render option, a debug-view dropdown (shared textures and debug views) and a screenshot button. Keys: WASD/QE move, left-drag look, F5 reload shaders, F6 screenshot, Ctrl pause rendering. I did not drive the interactive UI by screenshot, only the benchmark path and the launcher.
- Per-pass GPU timings: yes. The UI shows a profiling tree. Benchmark mode writes a CSV with Average, Min, Max, Accumulated and per-frame columns for every pass (for example `GI-1.2 / UpdateRadianceCache`, `ReprojectScreenProbes`, `InterpolateScreenProbes`, `TraceReflections`). With `--benchmark-first-frame 100` the CSV covers frames 100 to N; the default only records the last frame.
- Image capture: yes. Benchmark mode writes the final frame (or a range) as 16-bit half-float OpenEXR (pre-tonemap linear values), or JPEG with `--save-as-jpeg`. The debug view selected in the UI is what gets saved, so radiance cache and other buffers can be dumped too. File names are `Scene_EnvMap_Camera_Renderer_suffix_frame_frametime.exr`.

## Captured evidence

All under `artifacts/native/capsaicin/` (git-ignored):

- `gi/Sponza_KiaraDawn_User_GI-1.2_gi_120_*.exr` and `.png`, plus `gi/*.csv`.
- `pt/Sponza_KiaraDawn_User_ReferencePathTracer_pt_1000_*.exr` and `.png`, plus `pt/*.csv`.
- `probe/*.csv`: 2900-frame GI-1.2 timing sample.
- `runs/script-test-gi/`: output of the launcher script with an external scene and an explicit user camera.
- `configure.log`, `build.log`.
The two PNGs are `magick` conversions of the EXRs for viewing only (sRGB conversion, no exposure). They show the same framing: the Sponza atrium looking down the nave from a low position, a dark scene lit by the dawn environment. GI-1.2 is visibly smoother and slightly softer (FSR), and the reference shows crisper floor reflections and a cleaner far wall. Exposure is automatic in both and the image is dim; auto exposure acts on the displayed image, not on the saved EXR.

## Problems hit

1. Machine CMake 3.24 is below the required 3.30. Solved with a local portable 3.31.6 (no global install).
2. `CMAKE_TOOLCHAIN_FILE` is set in the user environment and points at a missing vcpkg path (`C:\Users\denni\documents\github\vcpkg\...`). CMake aborted. The failed run also cached the value in the build directory, so the build directory must be deleted and the env var removed for the child process (the script above does this). Nothing was changed globally.
3. The long worktree path (about 100 characters before `native/`) makes `git clone` of D3D12MemoryAllocator fail with "Filename too long" for its docs HTML, even though Windows long paths are enabled in the registry. Git needs `core.longpaths`. I set it per process with `GIT_CONFIG_COUNT/KEY/VALUE` environment variables instead of changing git config. The leftover partial fetch was cleared. I accidentally deleted the tracked `third_party/gfx/third_party/AmdDxExt` folder while clearing the partial fetch; restored with `git checkout -- third_party` inside the gfx submodule.
4. `scene_viewer.exe` is a GUI-subsystem app. Console output works only when it attaches to the parent console, so `--help` and `--list-*` showed nothing through the tool shell. Not a real blocker.
5. Camera index 0 on Sponza is labelled "User" in the saved name, and "Main" when `--camera -1` with explicit position is used. Both are fine but name the camera in any comparison manifest.

## What a same-scene comparison with our WebGPU output needs

1. Same scene file. Use `--load-external-scene` with our Sponza-with-curtains glTF (and Bistro). Check that the file loads, that units match and that lights/emissives survive. Our WebGPU lighting rig and Capsaicin's lighting (environment map or atmosphere or emissive meshes) must be matched. Capsaicin has no point lights from glTF here unless the scene provides them, so define the same sky: set `--start-environment-map-index 6` (White) or 7 (Atmosphere) for a simple known case, or export our environment as a `.hdr` and add it to the list (needs a source change) or by drag-and-drop in the UI.
2. Same camera. Use `--user-camera-position` and `--user-camera-lookat` plus a matching vertical FOV. FOV is exposed in the UI but I did not find a command-line flag for it; verify it. Capsaicin's coordinate system is right-handed, Y up (same as glTF), so positions should transfer directly.
3. Same resolution and no upscaling. Run at the exact comparison resolution. The FSR technique (it also provides TAA) sits in the GI-1.2 chain. Its render options in the source are `taa_enable`, `fsr_version`, `fsr_sharpen_enable` and `fsr_sharpen_sharpness`, settable with `--render-options`. I did not test switching it off, and the pass cannot necessarily be bypassed completely. Use `--render-scale 1`.
4. Linear output. The saved EXR is pre-tonemap linear HDR, which is what we want for numeric comparison. Match exposure by comparing in linear space with a fixed scale, not the auto-exposed view. Compare against the reference path tracer EXR as ground truth with a large frame count (for example 4000 or more; 1000 frames at 1 spp was visibly clean but not noise-free).
5. Metrics. Capsaicin has an image metrics technique (`image_metrics_enable` and a benchmark suffix write a metrics file). Not tried. Otherwise load both EXRs in our own tooling.
6. Timing. Use `--benchmark-first-frame` for a warmed average and read the per-pass CSV. Remember the 7.1 ms figure is a loaded laptop GPU at 720p with FSR in the chain. For a fair cost comparison with our compute passes, use matched resolution and compare GI-1.2 pass subtotals, not the entire frame.
7. Dynamic scenes. The viewer supports animations (`--start-playing`), which fits the moving-objects goal. Not tested.

## Launcher

**GPU sharing:** `run-capsaicin.ps1` does not take webgpu-ddgi's bench lock on its own. Whenever webgpu-ddgi work may be on the GPU, launch it through the lock from the webgpu-ddgi checkout, e.g. `node tools/reference/bench-lock.ts run -- pwsh <path>\scripts\native\run-capsaicin.ps1 ...` (check the lock tool's exact argument syntax first), so native timings and browser timings never overlap.

`tools/reference/capsaicin/run-capsaicin.ps1`. Examples:

```powershell
# GI-1.2 on bundled Sponza, 120 frames, images + timings in artifacts/native/capsaicin/runs/<name>
.\scripts\native\run-capsaicin.ps1 -Renderer gi -RunName sponza-gi
# Reference path tracer, 2000 accumulated frames
.\scripts\native\run-capsaicin.ps1 -Renderer pt -Frames 2000 -RunName sponza-pt
# Our own glTF at an explicit camera
.\scripts\native\run-capsaicin.ps1 -ExternalScene C:\path\scene.gltf -Camera -1 -CameraPosition 0,2,0 -CameraLookAt 5,2,0
# Interactive window with the UI
.\scripts\native\run-capsaicin.ps1 -Interactive -Renderer gi
```

`-DebugView <name>` saves a debug view or AOV (for example `GlobalIllumination`) instead of the final colour, and `-ImageFirstFrame N` dumps every frame from N (0-based) to the end. `-DebugView` needs the local scene_viewer patch `tools/reference/capsaicin/debug-view.patch` (adds `--debug-view`; applied and built on this laptop on 2026-11-19, see "Constant-sky brightness" below).

It runs the exe from the Capsaicin source directory, refuses to reuse a run name, and records the command line, exit code and peak `nvidia-smi` utilisation in `run.json` as an NVIDIA sanity check (warns if 0%). It does not pick an adapter itself because the app has no such flag.

## Our scenes in Capsaicin

Date: 2026-11-09. All runs: RTX 3060 Laptop (6 GB), 1280x720, render scale 1, Release build, one instance at a time, laptop not cleared of background apps. Everything is under `artifacts/native/capsaicin/` (git-ignored): `scenes/` (the Capsaicin-ready scene files) and `runs/<name>/` (EXR, per-pass CSV, `run.json`, `preview.png`). The helpers are tracked in `tools/reference/capsaicin/`: `make_scenes.py` (builds the scene files), `summarize.py` (groups the CSV), `preview.py` (EXR to viewable PNG). Capsaicin loads exactly one scene file per `--load-external-scene` (the Shift-append exists only for drag and drop in the UI: `loadScene(path, append)` is never reached with `append` from the command line). So every scene is first rebuilt as one file by `make_scenes.py`.

### Status per scene

| Scene (manifest) | Loads | Wall time of the 300-frame GI run (incl. about 12 s start-up and shader compile) | Materials and textures | Lights | VRAM (GI run, whole-GPU delta from nvidia-smi) |
| --- | --- | --- | --- | --- | --- |
| Sponza with curtains (`sponza`, 1k textures, main + curtains merged) | Yes | 26 s | Correct: albedo, normal and metal/rough maps look right, curtain colours right, one blend material only (no cutouts in this variant) | Manifest sun only | +3.5 GB |
| Bistro interior (`bistro-interior`) | Yes, after a texture fix (below) | 24 s with 1k textures, 87 s with the original 2048 px | Correct after the fix. The 4 emissive materials act as area lights | Manifest 3 lamps | +3.5 GB (1k), +5.6 GB (2048, at the 6 GB limit) |
| Bistro exterior (`bistro-exterior`) | Yes, after the same fix | 50 s with 1k textures, 174 s with 2048 px | Correct after the fix, cutout foliage visible | Manifest sun | +4.9 GB (1k), +5.4 GB (2048, spills) |
| Two-room (`two-room`) | Yes | 14 s (this is the start-up floor) | Flat colours, correct | Manifest lamp | +2.1 GB (Capsaicin's own GI buffers at 720p) |

Sponza with ivy and trees (`sponza-full`) and the candles scene were not tried. `make_scenes.py` would need two more entries.

Problems found and how they were solved:

1. **Bistro textures are WebP only.** Both GLBs carry only `EXT_texture_webp` images. Capsaicin (cgltf and stb) ignores the extension and renders white albedo (the first run looked like a bleached grey scene). `make_scenes.py` re-encodes every WebP into the GLB (JPEG q95, PNG when there is alpha) and drops the extension.
2. **Bistro at 2048 px does not fit 6 GB.** 119 images of 2048x2048 expand to about 2.6 GB uncompressed. GI-1.2 then takes 120 ms (interior) and 233 ms (exterior) per frame, which is memory thrash, not GI cost. Timings below use `bistro-*-1k.glb` (textures capped to 1024 px). The full-resolution runs are kept as `runs/bistroin-fullres-gi` and `runs/bistroex-fullres-gi` (renamed after the run, so their `run.json` still shows the old run name). Our WebGPU Bistro uses the 2048 px textures, so a texture budget difference is part of any cost comparison.

### Exact commands

Prepare the scene files (reads from `C:\Users\denni\Documents\dev\webgpu-ddgi`, read-only):

```powershell
python scripts\native\capsaicin\make_scenes.py            # sponza, bistro-interior, bistro-exterior, two-room
python scripts\native\capsaicin\make_scenes.py bistro-interior-1k bistro-exterior-1k   # texture-capped copies for timing
```

Run (camera 1 is manifest camera 0 of each scene, camera 2 is manifest camera 1):

```powershell
$s = "artifacts\native\capsaicin\scenes"
# GI-1.2: 300 frames, timings averaged over frames 101-300
.\scripts\native\run-capsaicin.ps1 -Renderer gi -ExternalScene "$s\sponza-curtains.gltf"   -Camera 1 -EnvironmentMap 0 -Frames 300 -TimingFirstFrame 100 -RunName sponza-gi
.\scripts\native\run-capsaicin.ps1 -Renderer gi -ExternalScene "$s\bistro-interior-1k.glb" -Camera 1 -EnvironmentMap 0 -Frames 300 -TimingFirstFrame 100 -RunName bistroin-gi
.\scripts\native\run-capsaicin.ps1 -Renderer gi -ExternalScene "$s\bistro-exterior-1k.glb" -Camera 1 -EnvironmentMap 0 -Frames 300 -TimingFirstFrame 100 -RunName bistroex-gi
.\scripts\native\run-capsaicin.ps1 -Renderer gi -ExternalScene "$s\two-room.glb"           -Camera 1 -EnvironmentMap 0 -Frames 300 -TimingFirstFrame 100 -RunName tworoom-gi
# Reference path tracer: 2000 accumulated frames at 1 spp, timings from frame 501 (add -TimeoutSeconds 900 for Bistro)
.\scripts\native\run-capsaicin.ps1 -Renderer pt -ExternalScene "$s\sponza-curtains.gltf"   -Camera 1 -EnvironmentMap 0 -Frames 2000 -TimingFirstFrame 500 -RunName sponza-pt
# ... same pattern for bistro-interior-1k.glb, bistro-exterior-1k.glb and two-room.glb
python scripts\native\capsaicin\summarize.py artifacts\native\capsaicin\runs\sponza-gi   # grouped GI-1.2 table
python scripts\native\capsaicin\preview.py <exr> <png>                                   # viewing aid, prints linear luminance stats
```

The launcher was extended for this: `-RenderOptions` (passes `--render-options`), `run.json` now records wall time and the nvidia-smi peak memory delta, the poll interval is 0.5 s, and a comment explains camera 0 versus glTF cameras.

### Camera and FOV matching

- Capsaicin's camera list is: index 0 `User`, then every perspective camera of the loaded glTF in file order. The User camera is created in `capsaicin_internal_scene.cpp` with a fixed **90 degree vertical FOV** (near 0.1, far 10000). There is **no command-line FOV flag**. `--user-camera-position` and `--user-camera-lookat` only work with `--start-camera-index 0`, and they set position and look-at only. The FOV can be changed in the UI (whole degrees, 10 to 140) or through `SetSceneCameraFOV`, but not at launch.
- A glTF camera carries `yfov` exactly: `gfx_scene.cpp` copies it into `camera.fovY` in radians. When selected by index, the aspect ratio is overwritten with the back-buffer aspect (1280/720), so the vertical FOV is what is honoured, the same convention as our `fovYDeg`.
- Method used: `make_scenes.py` injects each manifest camera as a glTF camera node (translation = `position`, rotation from look-at with up = +Y, `yfov = radians(fovYDeg)`) and removes the cameras the source files carry. Select with `-Camera 1` (manifest camera 0) or `-Camera 2`. The saved file name carries the camera name (for example `..._colonnade-from-the-floor_GI-1.2_...`), which confirmed the right camera was active. Manifest FOVs used: 75 degrees (Sponza), 70 (Bistro interior), 65 (Bistro exterior), 70 (two-room). Near is 0.1 m and far 10000 m; our renderer's clip planes were not compared.
- Coordinates: both are right-handed, Y up, metres, and glTF scenes load without any scale, so manifest positions transfer unchanged. Checked visually on Sponza and two-room.
- GI-1.2 detail that depends on FOV: the screen-probe and cache cell sizes scale with `tan(fovY ...)` (`gi1.cpp`), so changing FOV changes the cache resolution. Always compare at the same FOV, which the injected cameras guarantee.
- Alternatives that were not needed: set our camera to 90 degrees, or set the FOV in the UI.

### Lighting differences versus our manifests

- Capsaicin imports glTF `KHR_lights_punctual` (directional, point, spot), treats emissive meshes as area lights, and has an environment map. It has no equivalent of our procedural sky: the "Atmosphere" environment (index 7) is a fixed procedural sky with its own time-based sun, not driven by a scene light, so our `sunDirection`/`turbidity` sky cannot be reproduced. All runs therefore use **environment 0 (None) and the manifest's punctual lights**. Sponza and Bistro exterior have the sun and no sky, so shadowed areas get only bounce light. That is much darker than our WebGPU scenes with a procedural sky. A sky-lit comparison needs a matching HDR sky loaded (the UI takes `.hdr` by drag and drop).
- Light units. Capsaicin divides glTF intensity by 683, multiplies point and spot lights by 4 pi, and shades with `W/d^2`. The scene builder converts so that the manifest `intensity` (assumed to follow the three.js convention, lux for the sun and candela for points; I did not verify this against the WebGPU code) gives the same irradiance: sun `intensity * 683`, point `intensity * 683 / (4 pi)`, same `range`. In absolute terms the path tracer looks plausible: two-room about 1 linear, Sponza floor in the sun about 0.1.
- The source files' own lights are removed. Intel Sponza carries 24 punctual lights, all with intensity 0. The `arcade-lamp` in `sponza.json` has intensity 0 and was skipped. Manifest `exposureEv` is not applied: the EXRs are linear and pre-exposure.
- Emissive materials (Bistro: 4 interior, 10 exterior) carry over as area lights and visibly light the street and bar. The Bistro `KHR_materials_specular` extension is read (the exporter sets 2.0 on many materials). Whether our renderer honours it was not checked.
- **GI-1.2 handles delta lights much worse than the path tracer.** The path tracer gives sharp direct light and shadows from the sun and lamps. GI-1.2 gets punctual lights only through its world-space radiance cache and reservoir light sampling (`gi1_use_direct_lighting` covers emissive meshes and the environment map, not punctual lights in the screen pass). Result: soft, blurred or missing shadows and much less light. Mean linear luminance, GI-1.2 versus path tracer: Sponza 0.00085 vs 0.016, Bistro interior 0.029 vs 0.197, Bistro exterior 0.0054 vs 0.046, two-room 0.83 vs 1.08. Side by side images: `runs/*-cmp.png` and `runs/*/preview.png`. The sun in Sponza with a white environment (env 6, `runs/sponza-white-*`) behaved the same way. A fair GI-1.2 reference for our DDGI therefore needs emissive or environment-map lighting, or the path tracer must be the only ground truth for direct light. I did not find a GI-1.2 option that changes this.

### GI-1.2 per-pass timings (ms, warmed averages over frames 101-300)

Groups (pass names are `GI-1.2 / <name>` in the CSV). **Probe tracing / cell population**: SampleScreenProbes, PopulateScreenProbesCells, PopulateMultibounceCells, PopulateRadianceCache, GenerateReservoirs, GenerateMultibounceReservoirs, CalculateLightSamplerBounds and Configuration, BuildLightSampler. **Cache update / resolve**: PurgeRadianceCache, UpdateRadianceCache, UpdateMultibounceCells, ResolveCells. **Screen probes**: Reproject, Lookup, Spawn, Patch, Blend, Reorder, FilterProbeMask, Filter, Project, InterpolateScreenProbes. **Denoise / resolve**: ReprojectGI, FilterGI, ResolveGI1, MarkFireflies, CleanupFireflies. **Reflections**: TraceReflections, the ResolveReflections Atrous passes, ReprojectReflections.

| Group | Sponza | Bistro interior (1k tex) | Bistro exterior (1k tex) | Two-room |
| --- | --- | --- | --- | --- |
| Probe tracing / cell population | 4.75 | 7.81 | 7.56 | 1.31 |
| Cache update / resolve | 0.74 | 0.67 | 0.80 | 0.65 |
| Screen probes | 1.38 | 1.18 | 1.31 | 1.51 |
| Denoise / resolve | 1.08 | 0.71 | 0.95 | 0.56 |
| **Diffuse subtotal (excluding reflections)** | **7.95** | **10.38** | **10.62** | **4.03** |
| Reflections | 0.69 | 0.31 | 0.44 | 0.17 |
| GI-1.2 block total | 8.64 | 10.71 | 11.07 | 4.21 |
| Visibility buffer / SSGI / FSR | 2.42 / 0.36 / 1.06 | 1.20 / 0.31 / 0.88 | 2.05 / 0.32 / 1.03 | 0.78 / 0.43 / 1.06 |
| Whole frame (top-level passes) | 12.8 | 13.4 | 14.8 | 6.8 |
| Reference path tracer, 1 spp per frame | 23.8 | 34.8 | 53.0 | 7.4 |

Bistro with the original 2048 px textures, for reference only (memory thrash, not valid cost): interior diffuse subtotal 115.7 ms (block 120.0), exterior 213.4 ms (block 232.9).

### Caveats

- Timings come from a 6 GB laptop GPU that also drives the display and was not clock-locked. Run-to-run noise is visible: two-room GI diffuse measured 3.44 ms in a first test run and 4.03 ms in the recorded run. Treat differences under about 15 percent as noise.
- The FSR pass (0.9 to 1.1 ms) is in every GI frame and the saved EXR comes after it. It was not switched off. `taa_enable` and the other FSR options can be set with `-RenderOptions`, effect not tested.
- Probe tracing is where Bistro costs extra (about 7.7 ms versus 4.75 ms in Sponza): hit shading grows with scene complexity and texture footprint. The 1k numbers assume the textures fit in VRAM; the 2048 px runs show what happens when they do not.
- VRAM figures are whole-GPU `memory.used` minus a pre-launch baseline, so any other process allocating during a run changes them. Path tracer runs gave inconsistent values (Sponza PT +1.5 GB, Sponza white-sky PT +4.7 GB), so only the GI figures and the two-room floor of about 2.1 GB are meaningful.
- The EXRs are linear and unexposed. Absolute values depend on the light unit conversion above. Normalise both sides to a common reference patch before numeric comparison. `preview.py` scales to a 90th-percentile target for viewing only.
- Path tracer images at 2000 frames and 1 spp are still slightly noisy in Bistro (30 bounces, no denoiser). Use more frames for ground truth.
- The scene files are derived copies. The merged Sponza glTF references the original 1k glTF `.bin` files and textures by relative path (it breaks if either tree moves). The Bistro GLBs are re-encoded copies (181 MB and 527 MB at full resolution). The helper scripts (`make_scenes.py`, `summarize.py`, `preview.py`) are tracked in `tools/reference/capsaicin/`; `make_scenes.py` still writes its scene files into the git-ignored `artifacts/native/capsaicin/scenes/`.
- Not done: the second camera of each scene (the scene files already contain it as `-Camera 2`), moving objects and lights (`--start-playing` with animations), image metrics, a sky matching our manifest.

## Constant-sky brightness of GI-1.2 diffuse GI (D25)

Date: 2026-11-19. Question: how bright does GI-1.2's screen-probe irradiance read under a constant sky, at frame 1 and at steady state? Review derivation: the SH projection (`gi1.comp` ProjectScreenProbes, weight 1/8 per hemi-octahedral cell at its centre, clamped-cosine evaluation) gives E = 1.300 πL without history; the steady state with history (sampling with replacement, empty cells filled by BlendScreenProbes' backup / empty count) was unknown. Our WebGPU port uses a solid-angle weight and reads 1.0005 πL.

### Method

- Scene `flat-plane.gltf` (`make_scenes.py flat-plane`): one 200 x 200 m upward plane, base colour 0.5 grey, metallic 0, roughness 1, no lights, glTF camera 1 at 5 m looking straight down, yfov 60 degrees. Environment 6 (White, `white.hdr`, a single RGBE pixel decoding to exactly 1.0). Nothing above the plane, so every probe ray misses and E = πL everywhere; the world-space cache stays empty and the GI is purely sky misses in the screen probes (`PopulateScreenProbesHandleMiss`, `gi1_use_direct_lighting` default true). 1280x720, render scale 1.
- What is measured: the `GlobalIllumination` AOV (GI-1.2's ResolveGI1 output, before Combine, FSR, bloom, vignette and tone mapping), dumped as linear EXR. The viewer cannot select a debug view from the command line, so I added `--debug-view` to `scene_viewer` (`tools/reference/capsaicin/debug-view.patch`, 2 hunks in `main_shared.cpp`, additive; the submodule working tree now carries it and the Release exe was rebuilt, 19 s).
- Two GI variants: defaults as shipped, and `gi1_disable_specular_materials=true`. The second compiles the specular layer out, so the AOV is exactly albedo/π · E and **E/(πL) = AOV / 0.5**. With the default material (F0 0.04 is fixed in `MakeMaterialBRDF`, roughness 1 > the 0.6 threshold) the AOV is E/π · (albedo · diffuse compensation + directional albedo), so it can only be compared with the path tracer of the same material.
- Ground truth: Reference Path Tracer, 1000 frames, `Color` (that renderer has no post chain), with the default material and with `reference_pt_disable_specular_materials=true`.
- Statistic: mean of the linear red channel (R = G = B here) over a centred crop, `region_mean.py`. 10% crop = the middle 128x72 pixels, 50% = the middle quarter.
- Commands: `python scripts\native\capsaicin\make_scenes.py flat-plane`, then from `C:\Users\denni\Documents\dev\gi`: `node tools/reference/bench-lock.ts run --no-cool-wait -- pwsh -File <ddgi>\scripts\native\capsaicin\run-d25-sky.ps1` (8 runs, runs `d25-*` in `artifacts/native/capsaicin/runs/`), plus `d25-gins-curve` (frames 1 to 200), `d25-ao` (OcclusionAndBentNormal), `d25-occ` (SSGI Occlusion view) with the same launcher flags. `python scripts\native\capsaicin\region_mean.py --ref 0.5 --crop 10 <exr>...`.

### Results (E/(πL); diffuse-only GI-1.2 AOV / 0.5)

| Crop | Frame 1 | Frame 16 | Frame 48 | Frame 300 | Frame 1000 | Path tracer, diffuse-only |
| --- | --- | --- | --- | --- | --- | --- |
| Centre 10% | **1.294** | | | | **0.874** | 1.007 |
| Centre 50% | 1.255 | 1.135 | 1.002 | 0.843 | 0.840 | 1.007 |
| Full frame | 1.178 | | | | 0.782 | 1.005 |

- Default material (as shipped), GI-1.2 AOV / path tracer AOV, centre 10%: frame 1 **1.240**, frame 1000 **0.877** (50% crop: 1.193 and 0.843). Same picture as diffuse-only.
- Time course (50% crop): 1.255, 1.246 (2), 1.219 (4), 1.196 (8), 1.135 (16), 1.052 (32), 1.002 (48), 0.968 (64), 0.917 (96), 0.885 (128), 0.854 (200), 0.843 (300), 0.840 (1000). Monotonic decay, converged by about frame 300, crosses 1.0 near frame 48.
- Path tracer check: 1.007 diffuse-only (0.50355 for an expected 0.5). The 0.7% excess is unexplained (the HDR pixel decodes to exactly 1.0; possibly the environment cubemap conversion). Dividing by the path tracer instead of 0.5 moves every GI ratio down by 0.7% (centre 10%: 1.285 frame 1, 0.867 steady).
- Offline replica of the frame-1 SH path (`tools/reference/capsaicin/sh_model.py`: 8x8 cells, L = 1, weight 1/8 at cell centres, Capsaicin's SH and clamped-cosine-cone code): 1.3000 for the true normal, matching the review. Averaged over the SSGI bent normals of the centre 10% crop it predicts 1.2985; measured 1.294.

### Reading

- Frame 1 confirms the review: about **1.29-1.30 πL** where the bent normal equals the normal.
- Steady state is **dark, not bright: about 0.87 πL** in the centre (0.84 over the middle quarter). History drives the value down over about 300 frames. The mechanism is not isolated here (candidates: sampling with replacement in SampleScreenProbes and the backup / empty-count fill in BlendScreenProbes); only the end value is measured.
- So GI-1.2 as shipped is neither energy-conserving at frame 1 (+30%) nor at steady state (-13%). Our port's 1.0005 πL is the physically correct value; matching Capsaicin would mean matching a value that changes over time.

### Caveats

- **Screen-space bent normals darken the edges.** GI-1.2 evaluates the probe SH with SSGI's bent normal (`HAS_OCCLUSION`). On this plane SSGI reports occlusion about 1 (mean 0.997) but a noisy bent normal whose tilt grows away from the screen centre: mean cos(tilt) 0.998 in the centre 10%, 0.958 over the middle quarter, 0.91 in a corner. The model above turns that into -2% at the 50% crop, and the corner reads 0.53/0.5 = 1.07 at frame 1 and 0.70 at steady state. This is an SSGI effect, separate from the SH weighting; the centre 10% numbers are the clean SH answer. Whether the SSGI tilt is an artefact of looking straight down at a plane was not checked.
- One scene, one view, one albedo, one environment. Steady state could depend on resolution, FOV (probe and cache cell sizes scale with FOV) and the spatial-temporal denoiser (ReprojectGI / FilterGI are part of the measured AOV).
- `gi1_disable_specular_materials` also removes GGX-lobe probe rays (`SampleScreenProbes`). The default-material run, compared against the path tracer, gives nearly the same ratios (1.240 and 0.877 centre), so that option does not change the answer.
- Only the GI-1.2 AOV is measured. The final displayed image adds Combine, FSR/TAA, bloom (`bloom_enable` default true) and vignette (`lens_vignette_enable` default true), none of which were looked at.
- The `scene_viewer` patch stays in the submodule working tree (shows as modified content); the patch file restores it on a fresh checkout with `git -C native/capsaicin apply ../../../tools/reference/capsaicin/debug-view.patch`.
