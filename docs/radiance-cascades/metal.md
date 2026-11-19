# Apple Silicon reproduction

Run the existing split radiance cascade fixtures on an Apple GPU with a recent Chrome, Node 24 and pnpm 10.23.0. The Windows evidence uses NVIDIA hardware. An Apple measurement comes from the output directory of a completed Mac run.

## Run on the Mac

For the portable handoff, unzip `gi-rc-metal.zip`, open Terminal in the extracted directory, and run:

```sh
bash run-metal.sh
```

The launcher prepares a fresh checkout and dependency seed, runs the fixtures and writes `rc-metal-results-<timestamp>.tar.gz` next to the launcher for return.

To run from source, use a checkout of `codex/radiance-cascades`. If you received just the Git bundle, this command creates the checkout:

```sh
git clone -b codex/radiance-cascades gi-radiance-cascades.bundle gi-rc-metal
cd gi-rc-metal
```

```sh
pnpm install --frozen-lockfile
node tools/gpu/pages/rc-polished-budget/metal.ts --out ../rc-metal-results
```

The source install needs Git read access to the private `pmndrs/pathtracer` repository for the pinned `packages/raytracer` dependency at `2489a09ab1844a7effe6e845ecccd05876580c9f`. The portable handoff includes the private dependency seed. The BVH, downsampler and upscaler dependencies resolve from the registry through the lockfile. No sibling checkout is needed for these fixtures.

Choose a fresh output directory for each run. The launcher builds the gallery with the `/gi/` base, acquires the repository’s GPU bench lock and uses the shared headless Chrome test harness. Close unrelated GPU workloads before measuring. The default browser lookup includes Chrome in `/Applications`. For another installation, append `--chrome "/path/to/Google Chrome.app/Contents/MacOS/Google Chrome"`. The terminal displays each fixture result as it completes.

## What the run records

- **Hardware.** `host.json` records the macOS version, machine model, Apple Silicon capability, Metal-capable GPU model from System Profiler, Chrome version and GPU process renderer, Node version, source revision, changed source paths and lockfile digest. Each fixture report must identify a nonfallback Apple WebGPU adapter with timestamp queries. Chrome’s ANGLE renderer string identifies its graphics process. The page reports separately identify their WebGPU adapter.
- **Correctness.** `correctness/run.json` includes the transport, native cascades, native reflections, moving door, backend comparison, reconstruction, moving rail and rough-reflection motion pages. It contains their own assertions and readbacks. Their PNGs accompany the report. The separate Chandra Sponza exercise uses additional reference assets and its own reproduction recipe. [Chandra reference](./chandra.md)
- **Image quality.** `quality/run.json` and its screenshot compare full-resolution temporal antialiasing, its repeat, and 1.5× and 2× temporal upscaling on the 960 × 540 polished rail. The four six-frame windows begin at ticks 31, 121, 211 and 301. Reports retain per-frame RGB errors, reference-edge and floor errors, and adjacent-frame errors for the 20 within-window frame pairs. Each arm must retain valid history throughout.
- **Matched timing.** Each of the three `pair-N/run.json` files contains both arms at 640 × 360 internal and 960 × 540 display. Arm order is on/off, off/on, on/off. Each arm warms up for 48 moving frames and measures ticks 128 through 187. The files retain all 60 whole-frame GPU queue timestamps, 59 animation intervals, cascade update count, and lit-floor difference. `summary.json` reports matched and pooled median/p95 differences, defined as on minus off. Pooling uses the 180 individual timestamps for each arm.
- **Idle preflight.** Each `pair-N-preflight.json` records five-sample windows before launching Chrome for that pair. An observed-idle-and-nominal result requires every sample to report whole-device GPU utilization below 20% and nominal system thermal pressure. A busy device or elevated pressure triggers another window after a three-second pause, with a 90-second wait budget plus the final sensor read. The run retains descriptive timing if the wait expires and flags that condition. A missing whole-device counter or pressure state yields an unassessed verdict if there is no observed busy reading or elevated pressure. Renderer and tiler activity are separate counters.
- **System conditions.** `pair-N-telemetry.json` records the final preflight sample, approximately three-second samples while each pair's Chrome runner is active, and a post-run sample. The Foundation thermal state is system-wide pressure, reported as nominal, fair, serious or critical. GPU utilization percentages come from exposed IORegistry accelerator counters. Unavailable readings are recorded as null. These observations span startup and both arms. Per-frame GPU clock and junction temperature remain unmeasured. The summary records thermal qualification as unassessed.

Keep the entire output directory together for the return handoff. A completed run writes `summary.json` only after correctness, quality and all three pairs pass. On failure the command exits nonzero, writes `failure.json` and retains any reports produced before the error.
