# DDGI viewer

The tool for looking at, and comparing, global illumination on our test scenes. It runs our DDGI (through the three.js adapter, `@pmndrs/ddgi/three`), two baselines from three.js r187 (`LightProbeGrid` and VXGI), and shows the path-traced Cycles reference next to any of them.

## Start it

```
pnpm dev
```

Open http://localhost:5200/ in Chrome or Edge (WebGPU needed). It opens on the Cornell box with DDGI. Pick another scene in the panel on the right; the big ones (Sponza, Bistro) take a few seconds to load and a progress bar shows how far along it is. DDGI then warms its probes up (the pill at the bottom says how far); on Sponza that takes about 10 seconds.

The picture sits to the left of the panel, never under it. Hover over any control and the box at the bottom of the panel explains it in plain words. Press `H` to hide the panel; the picture then takes the whole window.

## What you can do

- **Scene and camera views:** the scenes come from `scenes/manifests/`. Each has saved camera views (shared with the native RTXGI tool).
- **Move:** orbit (drag, scroll, right-drag) or fly (WASD, Q/E, drag to look, Shift for fast).
- **Light:** exposure in stops and the tone mapper. Each scene opens at its own exposure (the manifest's `exposureEv`: Sponza +2, Bistro exterior +1, the rest 0); Reset goes back to it.
- **GI off / on / GI only:** off is direct light only; GI only hides the direct lights so you see just the bounced light.
- **Technique:** DDGI (ours, the default), light probe grid (baked, with bounces) or VXGI (traced live).
- **DDGI's settings:**
  - the update budget per frame (2 ms by default; the stats box shows how many probes that buys);
  - "Show the probes": every probe as a small ball coloured by its state (with a legend and counts), or by how settled it is: how many updates its light is averaged over (red few, green the most DDGI keeps, 200);
  - a debug view: GI only on white surfaces, coverage (where DDGI answers), or the probes near each point;
  - a probe atlas over the picture, in the corner or filling it: irradiance (one small square per probe, the light it gathered, with its own exposure) or distance;
  - change per frame (the flicker check): with the camera still, orange where a pixel got brighter since the last frame and blue where it got darker, full colour at 0.25%, 1% or 4%. It works on top of GI only and the debug views;
  - "Start the probes over": warm up again from black.
- **Lights** (panel, "Lights"): choose one of the scene's lights from the list, or click its marker in the picture (a small ball; a cone for a spotlight). Drag the arrows to move it ("Turn" shows rings to aim a spotlight). Sliders for on/off, brightness (each notch about twice as bright), colour, reach, a spotlight's cone and soft edge, shadows, and its GI light group. "Put this light back" restores the scene's own values. None of it compiles anything in GI: the stats box's "recompiles" line stays 0.
- **Light motions:** the scene's scripted motions (`light-sweep`: a lamp gliding at 1 m/s and back; `light-step`: the same lamp jumping every 3 s). Play and Stop; Stop puts the lamp back.
- **Time of day** (scenes with a sun and a sky): the hour and the day of the year move the sun; the sun light, GI's sky and the sky picture follow. "Play a 10-second day" runs sunrise to sunset. It opens with the sun where the scene has it. Turning it on (or moving a slider) turns DDGI's sun layer on, so the sun's brightness changes show in GI at once; DDGI starts over once for that. Below 5° the sky model stops, so the viewer fades the sun and sky out towards night.
- **Light groups (GI):** off, 2 or 4 groups (DDGI starts over), then a colour and a dimmer per group and for the sky and glowing surfaces. Instant in GI, nothing traced again; three's own direct light is unchanged (use GI only to see the bounced light alone).
- **Response meter:** pick a region (or keep the whole picture), choose a step motion and press "Measure a step". In GI only, it lets the light settle at a 16 ms budget (a test mode, said on the picture), waits 4 s at your budget, makes the jump, records 2 s, then finds the settled picture at 16 ms again. It plots the curve with the first frame that changed by more than 1% and the first frame within 5% of settled, and saves it as CSV. Keep the camera still. The definitions are the T2.4b test page's; `tools/gpu/pages/viewer-lighting` checks the two agree.
- **`sponza-candles`:** three draws its 1,024 shadowless candles with clustered lighting, each given a 3 m reach (three's `distance`; GI uses the same reach) so each cluster lists only nearby candles. Unranged, every pixel looped over every candle and the viewer drew 2 frames a second; `reach=0` in the link brings that back, `reach=6` widens it.
- **How DDGI starts:** it compiles its GPU programs, copies the scene and builds the ray-tracing trees, then warms up (up to 16 fast passes over every probe, at most 8 seconds of them) before the budget applies. The scene is drawn throughout.
- **Compare with the Cycles reference:** one button puts DDGI on the left and the path-traced Cycles picture of the current camera view on the right, at the same exposure and tone mapping. It lines up at the camera view it was rendered from (the panel says so when you move away). "GI only" shows Cycles' bounced-light image. A view without a reference says so, with the command that renders it. References live in `bench/references` (gitignored), so a built copy of the viewer has none.
- **Split screen:** compare two techniques on the same view. Drag the white line. Both sides get the same lights, the same sky from the manifest, the same exposure and tone mapping, and GI at full physical strength (1.0). No technique gets a fill light of its own. The probe grid takes sky light from its bake; VXGI has no sky term, so open scenes look darker on its side. That is how VXGI is, not a setting.
- **Baking:** the probe grid bakes again only when its own settings change (bounces, probe budget) or you press "Bake again". Exposure, tone mapping, GI mode, split screen and upscaler never re-bake. While it bakes, the picture updates about four times a second so the bake gets most of the time.
- **Upscaler:** optional `@pmndrs/upscaler`; works with No GI and the light probe grid.
- **Stats** (top left; hover over a line for what it means): frame time; three's GPU time per frame; and, with DDGI on screen, probes updated per frame, DDGI's update GPU time (trace, blend, write), the top-level tree refit, the library's GPU memory (scene copy, ray-tracing trees, probes), the adapter's CPU time per frame, and the GI library's recompiles (should stay 0).

## Links reproduce a view

The address bar follows what you do, so a link reopens the same view:

```
?scene=sponza&camera=1&wipe=lpg,vxgi&split=0.5
```

`scene`, `camera` (number or name), `motion` (a light motion playing), `tod=hour,day` (time of day on), `groups` (`2` or `4`), `reach` (metres for many unranged candles), `technique` (`ddgi`, `lpg` or `vxgi`), `gi` (`off`, `on`, `only`), `wipe` (two of `none`, `ddgi`, `lpg`, `vxgi`, `cycles`), `split`, `ev` (absent: the scene's own), `tone`, `nav` (`orbit` or `fly`), `upscale` (a quality preset), `budget` (DDGI's ms per frame), `probes=1`, `probecolor=history`, `debug` (`gi-only`, `coverage`, `probe-states`), `atlas` (`irradiance`, `distance`), `atlasfill=1`, `atlasev` (stops), `change` (`0.25`, `1`, `4`), and `cam=px,py,pz,tx,ty,tz,fov` for a free camera. See `src/ui/url-state.ts`.

## Checks and measurements

- **Contact sheet** (headless Chrome, every scene, fails on any console or GPU error, saves screenshots to `bench/screenshots/viewer-sheet`):
  `pnpm --filter @pmndrs/ddgi-viewer contact-sheet -- [--gpu nvidia|intel] [--only a,b] [--camera 1] [--query "gi=only"] [--eval "<js>"]`
- **Baseline numbers** (bake time, frame cost, GI cost per scene and technique): the GPU test page `tools/gpu/pages/viewer-baselines`. Hold the bench lock for the whole run: wrap the command in `node tools/reference/bench-lock.ts run --what "viewer baselines" -- <command>` (`status` shows who holds it; never a bare `mkdir`). Definitions are at the top of `src/bench/baselines.ts`.
  ```
  VITE_DDGI_BASELINES=full node tools/reference/bench-lock.ts run --what "T0.5 viewer baselines" -- pnpm test:gpu --filter viewer-baselines --gpu nvidia --json run.json
  node tools/gpu/pages/viewer-baselines/to-csv.ts run.json --append
  ```
  Without `VITE_DDGI_BASELINES=full` it measures two small scenes only (a quick check that it works). `VITE_DDGI_SCENES=sponza` and `VITE_DDGI_TECHNIQUES=none,lpg` narrow a run; `VITE_DDGI_ROUNDS`, `VITE_DDGI_FRAMES` and `VITE_DDGI_BAKE_RUNS` change the repeats.

  DDGI's frame cost (T1.9: GI on against off, GPU ms, frame p50 and p95, and its start-up), under the lock:
  ```
  VITE_DDGI_SCENES=cornell-box,two-room,sponza VITE_DDGI_TECHNIQUES=none,ddgi node tools/reference/bench-lock.ts run --what "DDGI frame cost" -- pnpm test:gpu --filter viewer-baselines --gpu nvidia --json ddgi.json
  ```

  What the numbers mean, in short:
  - **Bake** is two numbers: `bake_wall_ms` (start to finish, one wait at the end, nothing read back) and `bake_gpu_pass_sum_ms` (the sum of the bake passes' GPU durations, from a second bake; passes may overlap on the GPU, so it can exceed the wall time). Shader compilation is kept out of both and recorded as `warmup_wall_ms`.
  - **DDGI's start-up** replaces the bake: `prepared_wall_ms`, `first_gi_wall_ms` (scene copied, ray-tracing trees built) and `warmed_wall_ms`, each from when DDGI was created.
  - **GI ms** is the frame's GPU time with the technique minus its GPU time with GI off, through the same pipeline (same MSAA, same TRAA, same targets), as the mean over alternating rounds with a 95% confidence interval (`_ci_low`, `_ci_high`). The GI-off frame is recorded too (`gi_off_frame_gpu_ms`). For DDGI the frame includes the library's own passes (the probe update, the tree refit), which three's timestamps don't see.
  - **p95** (`frame_gpu_p95_ms`, `frame_wall_p95_ms`, and their `gi_off_` twins): the 95th-percentile frame of each round, median over rounds, to show spikes.
  - When that interval contains 0, the GI cost is smaller than the run can resolve. The value is written as measured, possibly a little negative, with `_below_noise` set to 1. That is a result, not a failure. Only an interval entirely below 0 is a failed measurement.
  - Every repeated number gets a `_cv` (spread divided by mean) and a `_noisy` flag (CV above 10%). A time that comes out negative, or any value that is not finite, is never written; it becomes a `_problem` row with the reason, and the page fails.

## Where things are

- `src/core/viewer-core.ts`: renderer, scene, techniques, render pipeline and the wipe. No page code.
- `src/techniques/`: the `GiTechnique` contract and its implementations (no GI, DDGI, light probe grid, VXGI, and the Cycles reference image).
- `src/ddgi/`: the library's entry points as the viewer reads them, and DDGI's debug drawing (probe spheres and their history colours, screen debug views, the atlas overlay, the change per frame).
- `src/scene/`: manifest loading, scene building (reuses `scenes/src`), model lookup, layers.
- `src/lighting/`: the light editor and gizmo, the motion player, time of day, the response meter.
- `src/ui/`: panel, the lighting controls, the meter's region picker, hover help text, link state.
- `src/bench/`: the baseline benchmark.
