# DDGI viewer

The tool for looking at, and comparing, global illumination on our test scenes. Today it runs two baselines from three.js r187 (`LightProbeGrid` and VXGI); our own technique plugs in later.

## Start it

```
pnpm dev
```

Open http://localhost:5200/ in Chrome or Edge (WebGPU needed). It opens on the Cornell box. Pick another scene in the panel on the right; the big ones (Sponza, Bistro) take a few seconds to load and a progress bar shows how far along it is.

The picture sits to the left of the panel, never under it. Hover over any control and the box at the bottom of the panel explains it in plain words. Press `H` to hide the panel; the picture then takes the whole window.

## What you can do

- **Scene and camera views:** the scenes come from `scenes/manifests/`. Each has saved camera views (shared with the native RTXGI tool).
- **Move:** orbit (drag, scroll, right-drag) or fly (WASD, Q/E, drag to look, Shift for fast).
- **Light:** exposure in stops (0 is normal) and the tone mapper.
- **GI off / on / GI only:** off is direct light only; GI only hides the direct lights so you see just the bounced light.
- **Technique:** light probe grid (baked, with bounces) or VXGI (traced live).
- **Split screen:** compare two techniques on the same view. Drag the white line. Both sides get the same lights, the same sky from the manifest, the same exposure and tone mapping, and GI at full physical strength (1.0). No technique gets a fill light of its own. The probe grid takes sky light from its bake; VXGI has no sky term, so open scenes look darker on its side. That is how VXGI is, not a setting.
- **Baking:** the probe grid bakes again only when its own settings change (bounces, probe budget) or you press "Bake again". Exposure, tone mapping, GI mode, split screen and upscaler never re-bake. While it bakes, the picture updates about four times a second so the bake gets most of the time.
- **Upscaler:** optional `@pmndrs/upscaler`; works with No GI and the light probe grid.
- **Stats:** frame time, and the GPU time of one frame, averaged over the last few frames (top left).

## Links reproduce a view

The address bar follows what you do, so a link reopens the same view:

```
?scene=sponza&camera=1&wipe=lpg,vxgi&split=0.5
```

`scene`, `camera` (number or name), `technique` (`lpg` or `vxgi`), `gi` (`off`, `on`, `only`), `wipe` (two of `none`, `lpg`, `vxgi`), `split`, `ev`, `tone`, `nav` (`orbit` or `fly`), `upscale` (a quality preset), and `cam=px,py,pz,tx,ty,tz,fov` for a free camera. See `src/ui/url-state.ts`.

## Checks and measurements

- **Contact sheet** (headless Chrome, every scene, fails on any console or GPU error, saves screenshots to `bench/screenshots/viewer-sheet`):
  `pnpm --filter @pmndrs/ddgi-viewer contact-sheet -- [--gpu nvidia|intel] [--only a,b] [--camera 1] [--query "gi=only"] [--eval "<js>"]`
- **Baseline numbers** (bake time, frame cost, GI cost per scene and technique): the GPU test page `tools/gpu/pages/viewer-baselines`. Hold the bench lock for the whole run: wrap the command in `node tools/reference/bench-lock.ts run --what "viewer baselines" -- <command>` (`status` shows who holds it; never a bare `mkdir`). Definitions are at the top of `src/bench/baselines.ts`.
  ```
  VITE_DDGI_BASELINES=full node tools/reference/bench-lock.ts run --what "T0.5 viewer baselines" -- pnpm test:gpu --filter viewer-baselines --gpu nvidia --json run.json
  node tools/gpu/pages/viewer-baselines/to-csv.ts run.json --append
  ```
  Without `VITE_DDGI_BASELINES=full` it measures two small scenes only (a quick check that it works). `VITE_DDGI_SCENES=sponza` and `VITE_DDGI_TECHNIQUES=none,lpg` narrow a run; `VITE_DDGI_ROUNDS` and `VITE_DDGI_BAKE_RUNS` change the repeats.

  What the numbers mean, in short:
  - **Bake** is two numbers: `bake_wall_ms` (start to finish, one wait at the end, nothing read back) and `bake_gpu_pass_sum_ms` (the sum of the bake passes' GPU durations, from a second bake; passes may overlap on the GPU, so it can exceed the wall time). Shader compilation is kept out of both and recorded as `warmup_wall_ms`.
  - **GI ms** is the frame's GPU time with the technique minus its GPU time with GI off, through the same pipeline (same MSAA, same TRAA, same targets), as the mean over alternating rounds with a 95% confidence interval (`_ci_low`, `_ci_high`). The GI-off frame is recorded too (`gi_off_frame_gpu_ms`).
  - When that interval contains 0, the GI cost is smaller than the run can resolve. The value is written as measured, possibly a little negative, with `_below_noise` set to 1. That is a result, not a failure. Only an interval entirely below 0 is a failed measurement.
  - Every repeated number gets a `_cv` (spread divided by mean) and a `_noisy` flag (CV above 10%). A time that comes out negative, or any value that is not finite, is never written; it becomes a `_problem` row with the reason, and the page fails.

## Where things are

- `src/core/viewer-core.ts`: renderer, scene, techniques, render pipeline and the wipe. No page code.
- `src/techniques/`: the `GiTechnique` contract and the three implementations (no GI, light probe grid, VXGI).
- `src/scene/`: manifest loading, scene building (reuses `scenes/src`), model lookup, layers.
- `src/ui/`: panel, hover help text, link state.
- `src/bench/`: the baseline benchmark.
