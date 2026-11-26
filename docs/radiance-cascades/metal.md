# Apple Silicon reproduction

Run the existing split radiance cascade fixtures on an Apple GPU with a recent Chrome, Node 24 and pnpm 10.23.0. The Windows evidence uses NVIDIA hardware. An Apple measurement comes from the output directory of a completed Mac run.

The recorded Apple run used an M5 Pro at `d8b9776`, and its results are on [issue #43](https://github.com/pmndrs/gi/issues/43). The pinned test branch at `75b9c67` predates the frame-timing fix. On Apple GPUs, `rc-backends` fails there with "three returned no whole-frame GPU timestamps", so run from the RC branch instead. [Why](../NOTES.md#webgpu)

## Run on the Mac

The [Apple Metal test issue](https://github.com/pmndrs/gi/issues/43) has the example checks and the recorded results for PR #42. Start from a fresh parent directory with Git, Corepack, Node 24 and Chrome installed.

```sh
git clone --single-branch --branch codex/radiance-cascades https://github.com/pmndrs/gi.git gi-rc-metal
cd gi-rc-metal
node tools/linked-siblings.ts --fetch
corepack pnpm install --frozen-lockfile
node tools/gpu/pages/rc-polished-budget/metal.ts --out ../rc-metal-results
```

Follow the issue's built-site smoke test and archive instructions after the runner finishes.

For the portable handoff, unzip `gi-rc-metal.zip`, open Terminal in the extracted directory, and run:

```sh
bash run-metal.sh
```

The launcher verifies the bundled GI checkout, installs its locked dependencies, runs the fixtures and writes `rc-metal-results-<timestamp>.tar.gz` next to the launcher for return.

If you already received the earlier kit, finish the run with its included launcher and return its results archive as usual. Its `host.json` records the source revision and lockfile digest so the results can be accepted as a historical baseline for that build.

Fresh checkouts install the interval tracer, `@pmndrs/raytracer@0.1.0`, from the public npm registry. The lockfile also pins the public BVH, downsampler, upscaler and denoiser packages. The fetch command clones the optional neural-tail sibling next to the checkout, and the Split RC run works without it. [Published tracer](https://www.npmjs.com/package/@pmndrs/raytracer/v/0.1.0)

Choose a fresh output directory for each run. The launcher builds the gallery with the `/gi/` base, acquires the repository GPU and CPU bench locks in that order and uses the shared headless Chrome test harness. Close unrelated GPU workloads before measuring. The default browser lookup includes Chrome in `/Applications`. For another installation, append `--chrome "/path/to/Google Chrome.app/Contents/MacOS/Google Chrome"`. The terminal displays each fixture result as it completes.

## What the run records

- **Hardware.** `host.json` records the macOS version, machine model, Apple Silicon capability, Metal-capable GPU model from System Profiler, Chrome version and GPU process renderer, Node version, source revision, changed source paths and lockfile digest. Each fixture report must identify a nonfallback Apple WebGPU adapter with timestamp queries. Chrome's ANGLE renderer string identifies its graphics process. The page reports separately identify their WebGPU adapter.
- **Correctness.** `correctness/run.json` includes the transport, native cascades, native reflections, moving door, backend comparison, reconstruction, moving rail and rough-reflection motion pages. It contains their own assertions and readbacks. Their PNGs accompany the report. The separate Chandra Sponza exercise uses additional reference assets and its own reproduction recipe. [Chandra reference](./chandra.md)
- **Image quality.** `quality/run.json` and its screenshot compare full-resolution temporal antialiasing, its repeat, and 1.5Ã— and 2Ã— temporal upscaling on the 960 Ã— 540 polished rail. The four six-frame windows begin at ticks 31, 121, 211 and 301. Reports retain per-frame RGB errors, reference-edge and floor errors, and adjacent-frame errors for the 20 within-window frame pairs. Each arm must retain valid history throughout.
- **Matched timing.** Each of the three `pair-N/run.json` files contains both arms at 640 Ã— 360 internal and 960 Ã— 540 display. Arm order is on/off, off/on, on/off. Each arm warms up for 48 moving frames and measures ticks 128 through 187. The files retain all 60 whole-frame GPU queue timestamps, 59 animation intervals, cascade update count, and lit-floor difference. `summary.json` reports matched and pooled median/p95 differences, defined as on minus off. Pooling uses the 180 individual timestamps for each arm.
- **Auxiliary sustained timing.** After the normal run, three more pairs repeat the same arm order, scene, warmup, resolution and image comparison. The shared fixture disables Chromeâ€™s frame limiter and GPU vsync while continuing one draw per animation callback. Each `uncapped-pair-N/run.json` retains the first 60 samples and all 480 consecutive cascade updates per arm at ticks 128 through 607. The first 60 are part of the 480. The eight windows each record 59 animation intervals, for 472 per arm. `uncapped-summary.json` reports the matched first-window results and the sustained medians, p95 values and on-minus-off differences for each pair and for all 1,440 samples per arm. The auxiliary pairs use the same idle preflight and telemetry, with files prefixed `uncapped-pair-N`.
- **Idle preflight.** Each `pair-N-preflight.json` records five-sample windows before launching Chrome for that pair. An observed-idle-and-nominal result requires every sample to report whole-device GPU utilization below 20% and nominal system thermal pressure. A busy device or elevated pressure triggers another window after a three-second pause, with a 90-second wait budget plus the final sensor read. The run retains descriptive timing if the wait expires and flags that condition. A missing whole-device counter or pressure state yields an unassessed verdict if there is no observed busy reading or elevated pressure. Renderer and tiler activity are separate counters.
- **System conditions.** `pair-N-telemetry.json` records the final preflight sample, approximately three-second samples while each pair's Chrome runner is active, and a post-run sample. The Foundation thermal state is system-wide pressure, reported as nominal, fair, serious or critical. GPU utilization percentages come from exposed IORegistry accelerator counters. Unavailable readings are recorded as null. These observations span startup and both arms. Per-frame GPU clock and junction temperature remain unmeasured. The summary records thermal qualification as unassessed.

Keep the entire output directory together for the return handoff. The normal result is saved to `summary.json` after correctness, quality and the three normal pairs pass. The complete run also writes `uncapped-summary.json` after all three sustained pairs pass. On failure the command exits nonzero, writes `failure.json` and retains any reports produced before the error.
