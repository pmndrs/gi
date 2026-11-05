# Task board

Status values: `todo`, `active (agent, date)`, `review`, `done`, `blocked (why)`. One active task per agent. The main session moves tasks to `done` after review. How tasks work is in [PLAN.md](PLAN.md#what-done-means-for-a-task).

## Summary

| ID | Task | Agent | Size | Depends on | Status |
| --- | --- | --- | --- | --- | --- |
| T0.1 | Workspace | implementer | M | none | done |
| T0.2 | GPU runner | implementer | M | T0.1 | done |
| T0.3 | Timing and devices | implementer, then gpu-tester | M | T0.2 | done (Windows only; Apple devices deferred, see DECISIONS 2026-11-04) |
| T0.4 | Test scenes | wrangler (assets) + implementer (manifest, procedural scenes) | M | T0.1 | done (11 scenes; Sponza is Intel, download-only; Opus-review fixes in) |
| T0.5 | Viewer shell | implementer | M | T0.2, T0.3, T0.4 | done (baselines recorded on both GPUs; Dennis ran it 2026-11-04) |
| T0.6 | Native harness | native-runner | M | T0.4 | done (review fixes checked; default config matches upstream within noise; fork branch `m0-harness` at c7d0162) |
| T0.7 | References and compare | implementer | M | T0.4 | done (Cycles references for Cornell, two-room, Sponza; tolerance in DECISIONS; Opus-review fixes in, Sponza re-rendered diffuse-only) |
| T0.8 | Spike: three.js sharing | implementer | S | T0.2 | done (yes to all; env-diffuse needs a workaround) |
| T0.9 | Spike: Babylon | implementer | S | T0.2 | done (fits the contract; Babylon owns its device). Paused: three-only until v1 is validated |
| T0.10 | Spike: BVH | main session designs, implementer builds, gpu-tester measures | M | T0.2, T0.4 | done (our tracer chosen: watertight 1.20× three-mesh-bvh on Sponza on the RTX 3060, 1.97× on Intel; Opus review passed; the oracle can now fail on corner escapes. B0 continues in the BVH lane) |
| T0.11 | Platform API draft | main session | M | none (revise after T0.8–T0.10) | done (signed off by Dennis 2026-11-04, incl. UV mapping and vertex normals; revised after T0.10: binding budget, inline tracing by default, `range: 0` = no limit) |
| T0.12 | M0 note and M1 breakdown | main session | S | all M0 | done (note in [docs/milestones/M0.md](docs/milestones/M0.md); M1 on the board; signed off by Dennis 2026-11-04) |

## M0 tasks

### T0.1 Workspace

- **Goal:** a clean pnpm workspace in the same shape as `@pmndrs/upscaler`, so the sibling libraries feel alike.
- **Do:**
  - `pnpm` workspace with:
    - `packages/ddgi` (umbrella `@pmndrs/ddgi`, subpath exports `.`, `./core`, `./three`);
    - `packages/ddgi-core`;
    - `packages/ddgi-three`;
    - `apps/viewer`;
    - `examples/webgpu/hello`;
    - `tools/`;
    - `bench/results/`.
  - Inside `ddgi-core`: `src/platform/` and `src/techniques/ddgi/`, as empty modules with an index.
  - TypeScript strict with `@webgpu/types`; ESLint 9 + typescript-eslint; vitest; vite; rollup type bundling; three pinned to `0.187.x`.
  - An ESLint rule: `ddgi-core` may not import `three`, `@babylonjs/*`, `pixi.js` or anything from `ddgi-three`.
  - `.git/info/exclude` gets a slash-less `node_modules` entry (see NOTES).
- **Done when:**
  1. From a fresh clone, `pnpm i && pnpm lint && pnpm typecheck && pnpm test && pnpm build` exits 0.
  2. Adding `import 'three'` to a `ddgi-core` file makes `pnpm lint` fail with the boundary rule's message (verify, then revert).
  3. `pnpm build` emits `dist/` with `.`, `./core` and `./three` entries and their `.d.ts` files.
- **Show:** the command output in PROGRESS.
- **Record:** install and build times.

### T0.2 GPU runner

- **Goal:** one command that runs browser GPU checks on a chosen GPU of this laptop and fails loudly when something's wrong.
- **Do:**
  - `pnpm test:gpu [--gpu nvidia|intel] [--filter name]`.
  - It launches installed Chrome headless with WebGPU through the DevTools protocol (Node's native WebSocket, no Playwright), following the upscaler's approach.
  - It serves test pages with vite.
  - It collects WGSL validation errors (`Log.entryAdded`), console output and screenshots.
  - It reports the adapter vendor and architecture.
  - It fails if WebGPU is unavailable, if the wrong adapter is picked, if there are uncaught errors, or on a timeout.
  - It ships with one smoke test: a compute shader writes known values and the page reads them back.
- **Done when:**
  1. `pnpm test:gpu --gpu nvidia` passes and reports `nvidia`/`ampere`.
  2. `pnpm test:gpu --gpu intel` passes and reports `intel`.
  3. A test page with a deliberate WGSL error fails the run, with the error text in the output.
  4. A run with WebGPU disabled fails with a clear message, not a timeout.
- **Show:** the run output in PROGRESS.
- **Record:** run time per GPU.

### T0.3 Timing and devices

- **Goal:** know whether GPU timestamps work, and what they cost, on every test device.
- **Do:**
  - In `ddgi-core/src/platform/`, a `GpuTimer`:
    - timestamp queries with a query-set ring;
    - pass begin/end helpers;
    - results available a few frames later;
    - silently does nothing without `timestamp-query`.
  - A tiny stats overlay in the viewer package.
  - A device page that runs a fixed workload and reports adapter info, timestamp support and results.
  - The page is served on the LAN (`vite --host`) so the Mac, iPad and iOS device can open it.
- **Done when:**
  1. Unit tests cover the ring and result mapping (no GPU).
  2. The device page reports per-pass GPU ms on NVIDIA and Intel through T0.2.
  3. A table in PROGRESS lists, for Windows Chrome NVIDIA and Windows Chrome Intel: WebGPU yes/no, timestamps yes/no, and the overhead of timing vs not timing. The Apple rows (Mac Chrome, Mac Safari, iPad Safari, older iOS Safari) are deferred to a later milestone (DECISIONS 2026-11-04); the LAN page is ready for them.
- **Show:** the device page URL.
- **Record:** that table, plus a `bench/results/timing.csv` row per device.
- **Note:** the Apple devices need you to open the LAN URL. The gpu-tester prepares the page and the instructions, and you read the numbers back, or the page posts them to the dev server.

### T0.4 Test scenes

- **Goal:** a varied, honest test set that the viewer, native harness and references can all use.
- **Do:**
  - Define the scene manifest: id, title, how to build or load it (a code module or model files plus transforms), units, up axis, cameras (position, target, fov), lights, suggested volume bounds and spacing, and source and license for every asset.
  - **Built in code:** Cornell box, classic two-room (doorway), furnace room, thin wall, closed door, hollow 18 mm cabinet row.
  - **Downloaded by `tools/assets/fetch`,** never committed unless redistributable:
    - Sponza (Khronos glTF Sample Assets; and Intel's version if feasible);
    - Bistro (NVIDIA ORCA, interior and exterior, converted to glTF);
    - one furnished room and one courtyard with an alpha-cutout tree, from Sketchfab or Poly Haven, under CC0 or CC-BY.
  - Checksums are recorded.
  - Each scene gets 2–4 camera bookmarks on lit, useful views.
- **Done when:**
  1. `pnpm assets:fetch` downloads and verifies every external asset from a clean clone.
  2. Every scene loads in a minimal three.js page with no errors, and shows correct scale against a 1 m reference cube.
  3. Every asset's license and source URL is in `THIRD_PARTY.md` and its manifest.
  4. No asset that can't be redistributed is committed.
- **Show:** a contact sheet: one screenshot per scene bookmark, direct light only.
- **Record:** triangle, mesh, material and texture counts per scene, in PROGRESS.

### T0.5 Viewer shell

- **Goal:** the human-facing tool, before any of our GI exists.
- **Do:**
  - `apps/viewer` on three r187:
    - scene picker from manifests;
    - orbit and fly;
    - camera bookmarks;
    - exposure (default 0 EV) and tone mapping;
    - the stats overlay (T0.3);
    - GI off/on/only;
    - a split-screen wipe between two techniques.
  - A technique switcher with three's `LightProbeGrid` (with bounces) and VXGI as baselines.
  - An optional `@pmndrs/upscaler` toggle.
  - Plain-language hover help on every control.
- **Done when:**
  1. Every manifest scene opens on its first bookmark, visibly lit, with no console errors.
  2. The split wipe shows `LightProbeGrid` vs VXGI on Sponza.
  3. Bake time and frame cost for each baseline are recorded per scene on NVIDIA and Intel. (The Mac is deferred: DECISIONS 2026-11-04.)
  4. You can use it without instructions; your feedback goes in PROGRESS.
- **Show:** the viewer URL.
- **Record:** a baseline table (scene × technique × device: bake ms, frame ms, GI ms) in PROGRESS and `bench/results/baselines.csv`.

### T0.6 Native harness

- **Goal:** NVIDIA's native RTXGI as a usable comparison and timing tool on our scenes.
- **Do** (in the `native/rtxgi-ddgi` fork, on a new branch):
  - Replace the hard-coded "RTX 3060" device pick with an ini/env setting.
  - `tools/native/play.ps1 -Scene <id> [-Clean] [-Camera n]`, which writes the harness `.ini` from our manifest (glTF path, cameras, lights, volume).
  - Clean mode turns off probe spheres, RTAO and dither.
  - Bring up Sponza and Bistro natively.
  - Script the built-in F4 benchmark to produce CSVs.
  - Build the D3D12 target too.
- **Done when:**
  1. `play.ps1 -Scene sponza -Clean` opens Sponza at bookmark 1 with DDGI on.
  2. Per-pass GPU ms (probe trace, blend, relocate, classify, lighting) are recorded for two-rooms, Cornell, Sponza and Bistro at matched probe counts and rays.
  3. The D3D12 build runs Sponza.
  4. The default config renders the same as before the change (checked by eye plus one capture).
- **Show:** the launch commands in `tools/native/README.md`.
- **Record:** `bench/results/native.csv`, plus a summary table in PROGRESS.

### T0.7 References and compare

- **Goal:** ground truth on demand, and a single way to compare images.
- **Do:**
  - `tools/reference/cycles`: a Blender script that builds a manifest scene and camera and renders linear EXRs (full and indirect-only) at a set sample count with denoising off.
  - `tools/reference/pathtracer`: three-gpu-pathtracer, denoised progressively with `@pmndrs/denoiser`.
  - `tools/compare a.exr b.exr`, which prints FLIP, mean relative error and RMS, and writes a heatmap PNG.
- **Done when:**
  1. Cycles references exist for Cornell, two-rooms and Sponza at bookmark 1.
  2. Cornell's Cycles and pathtracer references agree within a stated tolerance, and the tolerance is recorded.
  3. `tools/compare` on an image against itself gives zero error, and against a 10%-brightened copy reports the right mean error.
- **Show:** the heatmaps.
- **Record:** render times and sample counts.

### T0.8 Spike: three.js sharing

- **Goal:** confirm at runtime what the r187 source suggests.
- **Do:** a throwaway example in which:
  - our code creates the device with raised limits and passes it to three;
  - a `Light` subclass with a registered `AnalyticLightNode` samples an `ExternalTexture`-wrapped `GPUTexture` our code writes by compute, and adds it to irradiance;
  - it is run under the Standard, Meshlet and Visibility-Buffer drawing pipelines;
  - it is checked with `scene.environment` set and env diffuse off, if PR #35017 or a workaround allows;
  - it is checked inside a reflection-probe capture, using PR #35046's branch if needed.
- **Done when:**
  1. A pixel readback shows the expected added irradiance in all three pipelines.
  2. Removing the light vs setting intensity 0: the recompile count is recorded for each.
  3. A written yes/no answer for each sub-question, with the workaround if any.
- **Show:** screenshots per pipeline.
- **Record:** answers in PROGRESS; durable facts in NOTES.

### T0.9 Spike: Babylon

- **Goal:** prove the contract isn't three-shaped.
- **Do:** Babylon's WebGPU engine running on a device we pass in, if Babylon allows (otherwise using its device). A material plugin injects a WGSL function that samples an external `GPUTexture` into indirect diffuse.
- **Done when:**
  1. A pixel readback shows the added irradiance.
  2. A written answer: what the contract needs to change for Babylon, if anything.
- **Show:** a screenshot.
- **Record:** answers in PROGRESS.

### T0.10 Spike: BVH

- **Goal:** decide between our own tracer and three-mesh-bvh's WebGPU path, with numbers.
- **Do:**
  - The main session writes the design: node layout, two levels, binned SAH, worker build, any-hit early-out.
  - The implementer builds:
    - the builder (in a worker);
    - WGSL closest-hit and any-hit kernels;
    - a ray-batch benchmark page;
    - a port of the old project's CPU triangle oracle cases (thin wall, mirrored and non-uniform transforms, instancing) as tests.
- **Done when:**
  1. All oracle cases pass on GPU on NVIDIA and Intel.
  2. Rays per second for closest-hit and any-hit (probe-style rays) on Sponza and Bistro, for ours vs the latest three-mesh-bvh, on NVIDIA and Intel. (The Mac is deferred: DECISIONS 2026-11-04.)
  3. BVH build time and memory per scene.
  4. A decision is written in DECISIONS.md.
- **Show:** the benchmark page.
- **Record:** `bench/results/trace.csv`.

### T0.11 Platform API draft

- **Goal:** the contract everything else builds on, checked against more than one GI technique.
- **Do:**
  - TypeScript interfaces for:
    - the host contract;
    - scene handles;
    - the frame budget;
    - scene queries;
    - hit shading;
    - the technique contract;
    - the shading module;
    - stats and debug data;
    - lifecycle (configure, prepare, encode, dispose, device loss).
  - Walk a voxel technique and a surfel technique through them on paper.
  - Revise after T0.8–T0.10.
- **Done when:**
  1. The interfaces compile.
  2. The voxel and surfel walk-throughs are written, with gaps fixed or listed.
  3. You've read it and agreed.
- **Show:** `docs/api/platform.md`.

### T0.12 M0 note and M1 breakdown

- **Done when:**
  1. A one-page M0 note covers what works, what doesn't, the baseline table and the spike answers.
  2. M1 tasks are on this board with done-when checks.
  3. You've signed off.
- **Already decided for the M1 breakdown** (collected during M0, so they aren't lost):
  - **Sponza add-ons as a main task** (Dennis: "make that a main task when we start m1"):
    - the curtains (Intel 726650) join the default `sponza` scene, for colour bleeding;
    - ivy, trees and candles (726656, 726662, 726676) go into a new, heavier `sponza-full` scene, for M4 alpha cutouts and M2 many lights;
    - per add-on: licence text, size, triangle count and checksum, all download-only;
    - done when the viewer's Sponza at bookmark 1 shows the curtains, the fetch verifies, and the Cycles reference is re-rendered diffuse-only.
    - Nothing is downloaded before M1 starts.
  - **GPU top-level refit** every frame, with occasional rebuilds (DECISIONS 2026-11-04).
  - **One shared sky file** read by the viewer, Cycles, the native harness and DDGI, at a realistic sky-to-sun ratio. Early in M1, because the Sponza targets compare against references (T0.5 finding).
  - **Three-adapter checks:**
    - metalness folded into GI albedo;
    - UV mappings passed through;
    - vertex normals passed through.
  - **No Babylon tasks** (DECISIONS 2026-11-04).
  - **The subgroups-first rule** for new compute kernels.
  - **The BVH runs as its own lane** (Dennis, 2026-11-04): B rows on the board next to M1. B0 must be done before M1 starts, and B1–B2 before M3.

## M1: core DDGI loop, static scene, two hosts

Status: **started 2026-11-04** (M0 signed off). Targets come from SPEC "How we'll judge it". Every task's numbers are on the RTX 3060 and the Intel UHD.

Rules for every M1 task:
- **Subgroups first.** A new compute kernel uses subgroups where they help, with one declared fallback for devices without them, and a test covers the fallback (DECISIONS 2026-11-04).
- **No Babylon work** (DECISIONS 2026-11-04).
- **Native comparisons take 3 or more runs per scene,** because native timings drift 15-30% between runs (NOTES).

| ID | Task | Agent | Size | Depends on | Status |
| --- | --- | --- | --- | --- | --- |
| T1.1 | Shared sky | implementer | S | none | done (Opus re-review passed; viewer check redefined, DECISIONS) |
| T1.2 | Platform scene copy | implementer | M | none | done (Opus review passed with notes, all fixed; both GPUs) |
| T1.3 | Tracer as a platform module | implementer | M | T1.2 (B0 as built) | done (Opus review passed with notes, fixed; any-hit 1.03-1.22× three-mesh-bvh on the RTX, except Bistro interior 0.93×) |
| T1.4 | Hit shading | implementer | M | T1.1, T1.3 | done (Opus review passed with notes, fixed; `sunIlluminance` added per Dennis) |
| T1.5a | DDGI field and update loop | main session designs, implementer builds | M | T1.4 | done (Opus review passed with notes; fixes carried into T1.5b) |
| T1.5b | DDGI classification and relocation | implementer | M | T1.5a | done (Opus review passed with notes, fixed; Cornell +5.7%, two-room leak 0.80%) |
| T1.6 | Shading module and plain-WebGPU example | implementer | M | T1.5 | done (Opus review passed with notes, fixed; GI sampling 1440p inside view 0.905 ms median of 9 against 1.0) |
| T1.7 | three.js adapter | implementer | M | T1.6 | done (Opus review failed, fixed, re-review passed; neutrality 1e-8; CPU 0.09–0.14 ms still, 0.11–0.19 moving, lead re-timed) |
| T1.8 | Viewer integration | implementer | M | T1.7 | done (Dennis: "The viewer loads but quality has issues, they may be addressed later"; usability met, quality notes go to T1.9 and sign-off) |
| T1.9a | DDGI performance pass (budget model, small-batch cost, p95, startup) | implementer | M | T1.6, T1.8 | done (budget median within ±5% on four scenes; Sponza 30–43 probes per 2 ms; gaps in DECISIONS) |
| T1.9 | M1 measurements against targets | gpu-tester, native-runner | M | T1.8, T1.9a | todo (native matched-count runs done early) |
| T1.10 | Sponza add-ons | wrangler + implementer | M | T1.1 (for the re-rendered reference) | done (Opus review passed with notes, fixed; candles waiting on Dennis) |
| T1.9b | Budget v2 (work-based, calibrated with the clocks up; DECISIONS 2026-11-05) | implementer | M | T1.9a | done (Opus reviews passed with notes, all fixed through 037476d; RTX 3-run timing under the clock rule: 12 of 12 VALID and within ±10% of 2 ms; Cornell 1,574 / two-room 707 / Sponza 268 / Bistro exterior 132 probes; saturation check passes on both GPUs) |
| T1.12 | Stability: Cornell flicker fixed, debug views of the source channels, threshold set by eye; hysteresis aware of the time between a probe's updates, and Sponza's settle time measured GPU-bound (frames and seconds) | implementer | M | T1.5, T1.8 | review (phase 2a: time-aware hysteresis; 2b debug views to do) |
| T1.13 | HD Cycles references with stored cameras; the viewer's Compare matches them | implementer, then gpu-tester | M | T1.8, T0.7 | active (2026-11-05; Dennis: "640 x 360 is an insane size") |
| T1.14 | GI composite for three (`docs/design/three-composite.md`): a drop-in TSL node on three's `pass()` and `mrt()`, with three's GTAO on the GI term by default (curtains, DECISIONS); `GIVolume` stays the simple path; usage bar against LightProbeGrid | main session designs, implementer builds | M | T1.7, T1.8 | review (part 2: names, volume.start, render-time update, examples; cost timing pending) |
| T1.11 | M1 note and M2 breakdown | main session | S | all M1 | todo |

### BVH lane (runs alongside; DECISIONS 2026-11-04)

| ID | Stage | Must be done before | Status |
| --- | --- | --- | --- |
| B0 | tinybvh WASM builder in a worker (`packages/bvh`) | M1 starts (M1 can start without the ≤ 2 s, DECISIONS 2026-11-04) | done with B0b (DECISIONS: the median of 3 worker runs ≤ 2 s; our CPU builder deleted in T1.3) |
| B0b | Write our GPU layout from inside the WASM (no JS conversion pass); re-run the RTX trace bench on an idle GPU | T1.3 ends | done (Opus review passed; B0 passed: Sponza worker median 1.89 / 1.61 s) |
| B1a | GPU top-level tree: refit every frame from instance transforms, rebuild now and then | M1 ends (after T1.3's review fixes, alongside T1.4; DECISIONS 2026-11-05) | done (Opus review and re-review passed; 10k instances: RTX 0.086 ms, Intel 0.43 ms) |
| B1 | GPU LBVH per mesh (subgroups first), BLAS refit | M3 | done (merged into m1, Dennis approved; not wired into ddgi-core until M2's fast first GI. Opus review passed with notes, fixed: Sponza GPU build 31.2 ms on the RTX; oracle passes on both GPUs; trace 0.82x / 0.80x of tinybvh trees; 10k top level 0.091 ms; no-subgroups fallback to tinybvh tested) |
| B2 | GPU PLOC, then H-PLOC | M3 | done: merged into m1 (Dennis: merge once reviewed; Opus review passed with notes, fixed; the 50 ms build bar passed on the re-time under the clock rule: 46.6 ms median of the VALID runs, every run under 50, DECISIONS). GPU H-PLOC behind the B1 contract; its trees match tinybvh's `BuildHPLOC` (SAH 1.00001x on Sponza); SAH 1.032x tinybvh's best; trace 0.95x closest / 1.02x any of B0's trees; oracle passes on the RTX and the Intel UHD. Sponza loaded build 47.8 ms (45.3 ms building first), but every timing row is `clock_low` (power-limited ~1,000-1,100 MHz even from a cool card), so the 50 ms bar is not formally passed: Dennis's call. No separate PLOC builder: H-PLOC's merge is the PLOC step, also Dennis's call) |
| B3 | Wide or compressed nodes, short-stack traversal | the performance review | done: dropped (Dennis, 2026-11-05). CWBVH closest-hit about 1.0× BVH2 on Sponza against the 1.3× bar, 0.82× on Bistro interior; oracle passed on both GPUs. Findings in PROGRESS and NOTES; code on branch `bvh-b3` |

Done-whens for the B stages are in `docs/design/bvh-framework.md`. B1a's: a 10k-instance top level costs ≤ 0.5 ms of GPU time, and the oracle passes with instances moving every frame.

### Draft done-whens

- **T1.1 Shared sky.**
  - One sky definition becomes one HDR sky file, made by a script from the manifest (sun direction, turbidity). Its brightness is set so the sky lights a flat open ground with a stated, realistic share of the sun's light.
  - The viewer, Cycles, the native harness and the platform's environment all read that same file.
  - Done when an open-ground test scene lit only by the sky renders within the reference tolerance in Cycles, the viewer shows the same sky file the right way round, and Sponza's probe-grid view is no longer blue. (Changed 2026-11-04, DECISIONS: the viewer never lights surfaces with three's environment, so sky-lighting accuracy is checked on the platform's own sky irradiance in T1.4.)
- **T1.2 Platform scene copy.**
  - `SceneApi` is implemented: meshes with optional normals and uvs, constant-colour materials (metalness already folded in by the adapter), instances, lights and environment, with changes queued and uploaded inside `encode`.
  - Binding budget (`docs/api/platform.md`, "After T0.10"): the scene copy and hit shading use at most 2 storage buffers together. Vertex attributes share one buffer; instances, materials and lights share one table buffer.
  - Done when unit tests and a GPU readback page check every accessor against the input, memory is reported, adding or removing things never recompiles shaders, and a DDGI-shaped test kernel (tracer + scene copy + 3 technique buffers) compiles on a default device.
- **T1.3 Tracer as a platform module.**
  - `RayQueries` (`giTraceClosest`, `giTraceAny`, `encodeBatch`) runs over the scene copy. Mesh trees come from tinybvh (B0 as built) by default, with our builder as the fallback; the top level from B1a once it lands.
  - The oracle's test helpers (`oracle-compare`, `reference-tracer`) leave the public exports.
  - Any-hit: on the same rays it trails three-mesh-bvh in every scene but Sponza (furnished room 0.34×, Bistro interior 0.60× on the RTX 3060). Measure nodes, instances and triangles per shadow ray, and three-mesh-bvh at leaf size 4, then fix it or record why not.
  - Record the commit when a benchmark page runs, not when its CSV is written (`to-csv.ts`).
  - Done when the oracle passes through the platform API on both GPUs, and speed is within 10% of the T0.10 numbers.
- **T1.4 Hit shading.**
  - `giSurfaceAt` (interpolated normals), `giDirectIrradiance` (sun and sky, shadow rays, light-group mask), and `giSkyRadiance` and `giSkyIrradiance` from the shared sky.
  - Done when a sun on a plane matches E·cos θ, sky irradiance matches Cycles within tolerance, and shadow rays agree with the oracle.
  - From the T1.1 review (DECISIONS 2026-11-04):
    - `giSkyIrradiance` on the open-ground scene meets the reference tolerance (±0.5%) against Cycles.
    - The `procedural` environment type uses the same Preetham and calibration code as the sky files: move it from `scenes/src/sky` into the core (scenes then imports it), with one turbidity default.
    - The contract states the `equirect` layout (which row is the zenith, where u = 0 points), matching three's `equirectUV`, so three's textures can be passed straight in.
- **T1.5 DDGI field and update loop.**
  - Main session design first: probe layout, octahedral irradiance and distance atlases, rays with a random rotation each update, hysteresis, unlimited bounce from the current field, a budget in probes per frame, and classification and relocation on the GPU.
  - Tracing is inline (trace, shade and accumulate in one kernel). If the update cost comes near its budget, time inline against batch once.
  - Split before starting, as PLAN asks of L tasks (2026-11-05). Design and measurements: `docs/design/ddgi.md`.
  - **T1.5a, field and update loop:** the volume, the atlases, the `ddgi/update` and `ddgi/write` kernels with every probe treated as active, the budget and update order, `ddgiSampleIrradiance`, and the engine-free test view. Done when:
    - the furnace room converges to its analytic value within 2%;
    - Cornell's indirect light matches Cycles within the M1 accuracy target (the point-light Cornell, DECISIONS).
  - **T1.5b, classification and relocation:** first the trace/blend split (DECISIONS 2026-11-05), then the fixed-ray reductions (subgroups first, plus the fallback and its test), relocation, the probe states and their effect on sampling, and converting the ms budget into units. Done when:
    - probes inside walls are classified off;
    - the two-room leak is ≤ 1% (DECISIONS defines it);
    - the furnace and Cornell checks of T1.5a still pass, Cornell now on the manifest's own volume (DECISIONS 2026-11-05).
- **T1.6 Shading module and plain-WebGPU example.**
  - `giIrradiance` (Chebyshev visibility, coverage) and `globalsCode`; `examples/webgpu/hello` renders Cornell with GI.
  - Done when GI sampling costs ≤ 1.0 ms at 1440p on the RTX 3060.
- **T1.7 three.js adapter.**
  - Sky: map a pmndrs/sky instance's `exposure` to `sunIlluminance`, otherwise take it from the sun `DirectionalLight` intensity; mirror three's cube maps in x; pass `environmentRotation` (DECISIONS 2026-11-05).
  - A `Light` subclass with a registered node. Scene sync covers meshes, materials, instances and lights, plus the env-diffuse workaround.
  - Checks, each with a test: metalness folded into GI albedo; UV mappings passed through; vertex normals passed through; a point light with `distance = 0` arrives as `range: 0`.
  - Done when:
    - the same scene through `hello` and through three matches within a small tolerance (engine neutrality);
    - adapter CPU time is ≤ 0.2 ms per frame for a still scene and ≤ 0.5 ms with one object moving.
  - No Babylon work.
- **T1.8 Viewer integration.**
  - A default exposure per scene in the manifest (Sponza about +2 EV), with values picked with our DDGI on (DECISIONS 2026-11-05).
  - DDGI as a technique in the viewer, with probe debug spheres coloured by state.
  - Stats: probes per frame, update ms, memory, adapter sync ms.
  - A wipe against the Cycles reference image.
  - Done when Dennis can use it without instructions.
- **T1.9 M1 measurements.**
  - Measure on Sponza (with its curtains, untextured materials), Cornell and two-rooms: GI cost, update cost, whole-frame p50/p95, accuracy against Cycles, leaks, memory and startup.
  - Native RTXGI numbers on the same scenes, side by side, at matched probe counts and rays, 3 runs each.
  - Re-measure the Bistro exterior baselines on the RTX 3060; T0.5's are too noisy to quote.
  - Set the stability threshold by eye with Dennis.
  - Done when the targets are met or a gap is explained and agreed, measured GPU-bound on the target hardware (RTX 3060), and the Intel UHD passes every correctness check (SPEC revised 2026-11-05: no Tier B mode decision in M1).
- **T1.14 GI composite for three** (DECISIONS 2026-11-05).
  - A TSL node built like three r187's SSGI and GTAO nodes: the user's post-processing pipeline renders `pass(scene, camera)` with `mrt({...})`, and `ddgi(...)` adds `albedo/π × E` into the HDR colour before tone mapping. No custom render path in the adapter.
  - In composite mode the GI light returns no indirect diffuse for opaque surfaces. Transparent objects and cube captures keep the per-fragment path. `GIVolume` alone, with no pipeline changes, keeps working as today.
  - `examples/three/` next to three's LightProbeGrid example: about as many setup lines. Report the line counts, every API a user touches, and anything still harder than LightProbeGrid.
  - Done when:
    - the composite matches the per-fragment path on Sponza and Cornell: per-pixel relative luminance difference mean ≤ 0.5%, p95 ≤ 2%, opaque pixels, HDR before tone mapping;
    - the cost of showing GI (MRT extra plus the composite) is ≤ 1.0 ms on Sponza at 1440p, GPU-bound on the RTX 3060;
    - the usage report is written, and the scout's list of differences from three's examples is either fixed or explained;
    - both GPUs pass, and the Opus review passes.
- **T1.10 Sponza add-ons.**
  - See the T0.12 notes above: the curtains join `sponza`; ivy, trees and candles make `sponza-full`.
  - Done when the viewer shows the curtains at bookmark 1, the fetch verifies, and the Cycles reference is re-rendered.

## Carried to M2 (collected during M1)

- `sponza-candles`: an opt-in scene with a stated subset of Intel's 10,000 candle lights, for the many-lights stress test (DECISIONS 2026-11-05).
- A texture sky in the native harness's miss shader (DECISIONS 2026-11-04).
- Area lights on the platform, so an emitter's direct light can be analytic and DDGI holds only its bounced light (DECISIONS 2026-11-05, the viewer entry).
- A public "acceleration structures built" signal on the platform (the viewer duck-types `rays.whenBuilt`).
- **Fast first GI** (Dennis, provisional): first GI from B1's GPU-built trees, swapping to tinybvh's when the worker finishes; plus the other startup levers: file loading, the scene copy off the main thread, the cold shader compile.
- Light-change response from `SceneChanges` (the big-change rule is off; DECISIONS 2026-11-05).
- A deferred shadow-ray pass in platform hit shading: shadow rays take 53% of Sponza's DDGI trace and run at 1/7 of the bench any-hit rate (T1.9a).
- `encodeBatch` kernels compiled only on first use; why the sky SH projection takes 1 s to compile.

## Carried to M3 (collected during M1)

- Probes in empty space are switched off by classification ("no surface nearby"; 347 of Cornell's 1,728). Moving objects in mid-air then get no GI. M3 (moving objects) needs those probes active near receivers, or a fallback.
- The three adapter's change-detection latency (up to 41 frames on Bistro without `markChanged`).
- The main-thread scene copy (1.3 s Sponza, 4.1 s Bistro exterior at load) moves off the main thread.
- A flaky `tools/assets/fetch.test.ts` under load.
- Nested finer DDGI volumes around thin cloth (Sponza's curtains: 0.25 m around them measured +5.5% offline, about 40k probes, 60 MB, about 1 ms of update): the quality route behind M1's GTAO fix (DECISIONS 2026-11-05).

## Later

M1–M5 tasks are added at the end of the previous milestone. See [PLAN.md](PLAN.md#later-milestones-outline).
