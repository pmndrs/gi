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
| T1.9 | M1 measurements against targets | gpu-tester, native-runner | M | T1.8, T1.9a | done (lean pass per Dennis 2026-11-06; Opus review passed with notes, recorded in PROGRESS; FLIP threshold and stability wait on Dennis, see the M1 note) |
| T1.10 | Sponza add-ons | wrangler + implementer | M | T1.1 (for the re-rendered reference) | done (Opus review passed with notes, fixed; candles waiting on Dennis) |
| T1.9b | Budget v2 (work-based, calibrated with the clocks up; DECISIONS 2026-11-05) | implementer | M | T1.9a | done (Opus reviews passed with notes, all fixed through 037476d; RTX 3-run timing under the clock rule: 12 of 12 VALID and within ±10% of 2 ms; Cornell 1,574 / two-room 707 / Sponza 268 / Bistro exterior 132 probes; saturation check passes on both GPUs) |
| T1.12 | Stability: Cornell flicker fixed, debug views of the source channels, threshold set by eye; hysteresis aware of the time between a probe's updates, and Sponza's settle time measured GPU-bound (frames and seconds) | implementer | M | T1.5, T1.8 | done (Opus reviews passed with notes, all fixed; Dennis's eye check: steady; threshold locked at mean ≤ 0.1%, p95 ≤ 0.5%: Cornell 0.082% / 0.34%, two-room 0.0099% / 0.032%, Sponza 0.027% / 0.096%) |
| T1.13 | HD Cycles references with stored cameras; the viewer's Compare matches them | implementer, then gpu-tester | M | T1.8, T0.7 | done (Opus review passed with notes, all fixed; every reference 1920×1080 with its stored camera; HD accuracy Cornell +5.74%, two-room +5.50%; noise per view shown in the viewer, some viewing-only bookmarks over 3% per pixel on GI, listed in the M1 note) |
| T1.14 | GI composite for three (`docs/design/three-composite.md`): a drop-in TSL node on three's `pass()` and `mrt()`, with three's GTAO on the GI term by default (curtains, DECISIONS); `GIVolume` stays the simple path; usage bar against LightProbeGrid | main session designs, implementer builds | M | T1.7, T1.8 | done (Opus reviews passed with notes, all fixed; default: upscaler Performance + half-resolution GTAO, 0.73 ms, Sponza +6.59% vs HD Cycles) |
| T1.11 | M1 note and M2 breakdown | main session | S | all M1 | done (M1 note in docs/milestones/M1.md; M2 breakdown on the board; Dennis merged PR #3, m1 → main, on 2026-11-06) |

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

## M2: dynamic updates

Status: **signed off by Dennis 2026-11-09** (his look by eye pending; PR `m2` → `main` open). Started 2026-11-06 on branch `m2` (from main at 3d682dc; M1 merged as PR #3). The design is on main: [docs/design/m2-lights.md](docs/design/m2-lights.md) (§4.1–4.8 are the dynamic-update design), [docs/design/m3-moving.md](docs/design/m3-moving.md) (moving objects; the startup lane T2.12–T2.14) and [research/ddgi-dynamic-updates.md](research/ddgi-dynamic-updates.md). Dennis approved key probes as the main path, rigid moving objects in M2 and the H-PLOC rebuild on drift (DECISIONS 2026-11-06).

**Order (Dennis, DECISIONS 2026-11-06):** dynamic updates first (moving objects, time of day); startup lower.
1. The dynamic-update spine: T2.4a → T2.4b → T2.4c. T2.4c's first check decides whether key probes stay the main path ("far probes owe 2" is the fallback).
2. Rigid moving objects (T2.M1). The T2.0 shadow spike runs beside these.
3. Lights and time of day: T2.1, T2.2, T2.5, T2.6a/b, T2.3, T2.7, T2.8.
4. Startup, lower: T2.12, T2.13, T2.14.
5. Close: T2.10, then T2.11.

**How M2 is run (Dennis, 2026-11-06):** features over timing, so each gate gets one VALID timing run under the clock rule (`tools/reference/bench-lock.ts`) and no re-time rounds unless a decision depends on the number. Targets are the RTX 3060 and up and the Apple M5 Pro, GPU-bound; the Intel UHD is correctness only. Every task gets an Opus review against its done-when before it is marked done.

**Done-whens:** M2's are in m2-lights.md "Draft done-whens" (T2.0–T2.10) and m3-moving.md "Startup lane" (T2.12–T2.14). They are the board's done-whens as written, with the changes listed under "M2 done-when changes" below. T2.M1's is below too.

| ID | Task | Agent | Size | Depends on | Status |
| --- | --- | --- | --- | --- | --- |
| T2.4a | Change-driven history on T1.12's per-probe record, and `ddgi/mark` (the change estimate); takes T2.1's change-log part (old and new light in `SceneChanges.lights`, influence radius) | implementer | M | T1.12 | done (Opus review passed with notes; still scenes match M1 exactly and mark 0 probes on both GPUs; two-room's 2 m lamp step within 5% from the first update, against 1.42 s under M1's rule; recall 100%; `lightChangeScale` 0.5, DECISIONS) |
| T2.4b | GPU scheduler with camera priority over Budget v2's fixed count, the feedback rule; far probes owe 2 as the fallback; the warm-up moves into the technique (the viewer switches to `volume.start`) | implementer | M | T2.4a | done (merged into `m2` 2026-11-06 after Opus review; check 2 a borderline miss, kept: DECISIONS 2026-11-06) |
| T2.4c | Key probes (1 in 8) with change propagation, the main path for global changes; first checked on Sponza's sun step and two-room's lamp | implementer | M | T2.4b | done (merged into `m2` 2026-11-06 after Opus review; check 5's "10-second day" was not a day, re-timed on a real one at 3.40 ms against 2.2: moved to T2.4d) |
| T2.M1 | Rigid moving objects, pulled forward from M3: the change feed for transforms (M3's T3.1, rigid part), geometry marks and the mover mask (T3.4, rigid part), three adapter change tracking for transforms | implementer | M | T2.4b (T2.4c for far sets) | done (merged into `m2` 2026-11-06 after Opus review; items 1 and 6 ruled in DECISIONS) |
| T2.4d | A global change's response in rooms and during a moving sun: near probes keep their updates while the room they bounce off takes in the change; the update's cost while changes continue (m2-lights.md §4.5's second calibrated count if needed) | implementer | M | T2.4c, T2.6a | done (merged into `m2` 2026-11-06 after Opus review; all three items met) |
| T2.0 | Spike: deferred shadow-ray pass on Sponza (decision rule: go ahead if Sponza fits ≥ 1.25× the probes at 2 ms) | implementer | S | M1 done | done (Opus review passed with notes; **go for T2.2**: Sponza fits 2.2× the probes at 2 ms on the RTX, 646 against 293 by Budget v2; field within 0.04% on both GPUs; spike code stays on branch `m2-t2.0`, 06d5a18) |
| T2.1 | Light buffer v2 (capacity, the adapter's light walk), manifest `motions`, `sponza-candles` (1,024 lights) | implementer + wrangler | M | T2.4a (change log) | done (merged into `m2` 2026-11-06 after Opus review; `sponza-candles` set-up takes 958 s, handed to T2.3) |
| T2.2 | Deferred direct light (`gi/direct`) in platform hit shading | implementer | M | T2.0 | done (merged into `m2` 2026-11-06 after Opus review; deferred is the default, inline behind `directLight: 'inline'`; Sponza 2.03× the probes at 2 ms, DECISIONS) |
| T2.5 | Light groups: layers, compose, weights, adapter API | implementer | M | T2.2, T2.4a | done (merged into `m2` 2026-11-07 after Opus review; at most 6 layers a volume) |
| T2.6a | Sky re-bake for time of day: GPU procedural source, two-level SH with radiance SH, the compile fix | implementer | M | T2.4b | done (merged into `m2` 2026-11-06 after Opus review; checks 1–3 pass; check 4, the furnished room's sun step, not met and moved to T2.4d) |
| T2.6b | The sun layer in white with T2.5's groups (its own route for directional lights in `gi/direct`). The instant sky (first-order transfer, nudge) was dropped by its decision check: 52.8% / 35.6% against 70% (DECISIONS 2026-11-07; code kept on branch `m2-t2.6b`, d26cb01, not merged) | implementer | S | T2.3, T2.5 | done (2026-11-08; checks 1-3 met, cost 1.055× VALID; `sunLayer` off by default; Sponza's turn with the layer on is at the edge of T2.4c's bar, to T2.8) |
| T2.3 | Many-light sampling: the light grid | implementer | M | T2.1, T2.2 | done (merged into `m2` 2026-11-07 after Opus review; check 2 a known miss at 2.2–2.5×; checks 3 and 4 owe a VALID run, DECISIONS) |
| T2.7 | Native harness: manifest motions, texture sky, animated sun | native-runner | M | T2.1 | done (2026-11-09; Opus review passed, fixes verified; fork branch `m2-t2.7` pushed by the lead on Dennis's answer, 350f9de; merged into `m2`) |
| T2.8 | Viewer: gizmos, sliders, groups, time of day, motions, a response meter, the native compare page | implementer | M | T2.4b, T2.5, T2.6b, T2.7 | done (2026-11-09; checks 1-3 met, Opus reviews passed; check 4, Dennis's look at the viewer and compare page, pending and not blocking (DECISIONS 2026-11-09)) |
| T2.12 | Fast first GI: GPU H-PLOC/LBVH trees at load, tinybvh swapped in per mesh (lower priority) | implementer | M | B1, B2 | done (2026-11-08; check 1 1.90 s median VALID against 2.0; swap −0.0019%; trace 0.98× after the swap; SPEC's no-submit wording amended for load, for Dennis) |
| T2.13 | Scene copy off the main thread (lower priority) | implementer | M | none | done (2026-11-08; GPU packing matches on both GPUs; main-thread bar missed, Sponza 816 ms median VALID vs 150: a known miss for Dennis, guards proposed in DECISIONS) |
| T2.14 | Compile only what's configured (`encodeBatch` on first use, the sky kernel's compile) (lower priority) | implementer | S | T2.6a | done (2026-11-08; check 2 met on 14 scenes; check 1, the cold Cornell start, owed to T2.10) |
| T2.16 | Re-render the noisy viewing-only reference bookmarks (M1 note), or mark them viewing-only in the viewer (lower priority) | wrangler, then gpu-tester | S | none | done (2026-11-08; six views re-rendered in parts on the RTX, GI at most 3.2% per pixel and under 0.5% over 8 × 8 blocks; lead review) |
| T2.17 | Bistro exterior's GI frame: 545 ms wall time with GI on against 19 ms of GPU time, and Budget v2 calibrating to 14-20 probes (T2.10) | implementer | S | T2.10 | done (2026-11-08; diagnosed, not fixed: GPU memory overflow on the 6 GB RTX 3060 Laptop, 5.1 GB of uncompressed textures in three plus DDGI's 0.6 GB; with textures capped at 1024 a GI frame takes 47 ms instead of 517. Opt-in cap `VITE_DDGI_MAX_TEXTURE_SIZE` for the pages, merged into m2-t2.10. The fix (manifest cap, or KTX2) is Dennis's choice) |
| T2.18 | Keep the update within budget in the frames after a big change (T2.10: Sponza's sun turn, 2.44 ms median over the next 60 frames); the second calibrated count (m2-lights §4.5) while a change is still being taken in | implementer | S | T2.10 | done (2026-11-08; Opus review, fixes in; the median of the 60 frames after Sponza's sun turn is 0.99× the same run's still frames (lead ruling, DECISIONS 2026-11-08), p95 1.05× reported; view within 5% at frame 15 (bar 15); merges into m2-t2.10 after the Intel runs) |
| T2.10 | M2 measurements against targets (one VALID run per gate, DECISIONS 2026-11-06), with the measured whole-frame p50/p95 | gpu-tester, native-runner | M | all above | done (2026-11-09; RTX at about 800 MHz accepted as a low-clock floor, Intel UHD correctness, response against native; Opus review's fixes in; eight known misses accepted by Dennis (DECISIONS 2026-11-09); the sweep's look by eye pending) |
| T2.11 | M2 note and M3 breakdown | main session | S | T2.10 | done (2026-11-09; M2 note in [docs/milestones/M2.md](docs/milestones/M2.md), Opus review's fixes in; the M3 draft breakdown below) |

M3 keeps deforming meshes and the BVH work: runtime mesh trees, `GpuMeshRefit` wired, and the H-PLOC rebuild when a refit drifts, several volumes, the camera-following volume, nested finer volumes around thin cloth (carried from M1), and the scenarios (`docs/design/m3-moving.md`, T3.0–T3.11).

### M2 done-when changes

These add to the design docs' draft done-whens. M1's open follow-ups (the M1 note, "What isn't done") are placed here too.

- **T2.4a** builds the change log's light part, which the design put in T2.1: `SceneChanges.lights` with each change's old and new light, the 1,024-a-frame cap, and the influence radius. T2.1 keeps the capacity, the adapter's light walk, `motions` and `sponza-candles`. T2.4a's step test drives the lamp from its test page; it doesn't need manifest motions.
- **T2.4b** adds five checks (the first two from M1's follow-ups, the last three from T2.4a's review):
  - *The calibration guard* (M1 follow-up: Budget v2 once gave Sponza 26 probes a frame instead of about 290): the calibration logs its samples, and a result below half of the previous calibration on the same device and scene is re-measured once before it's used. Test with a forced low sample.
  - *The viewer uses `volume.start`* (M1 follow-up): with the warm-up in the technique (check 6), the viewer's own copy of the warm-up is removed.
  - *A larger step, and the overshoot:* T2.4a's lamp step changed the view by only 6.9%, and the view overshot by 56% of the step (+3.9%), still 29% of it after 2 s. Check 1's steps must change the view by at least 25% (for example the lamp switched on or off), and the overshoot is reported. If the view leaves the 5% band again after entering it, that counts as not settled.
  - *The steady-change rule has a test:* a lamp moving steadily (the sweep) keeps the view within 5% of a field re-converged at the same moment, measured at 3 points of the sweep.
  - *A lamp inside sunlit Sponza:* switching a lamp on in Sponza marks the probes it changes by more than 5% (recall ≥ 95%), because its influence radius is set against the sun's irradiance (DECISIONS 2026-11-06, the `range: 0` entry).
- **T2.4c** takes these from T2.4b's review: it reports Sponza's sun-halved step beside the sun turn (T2.4b's check 2 was a borderline miss: 19, 22, ≤ 30, 34, 24 frames against 30), and its sweep asserts the per-pixel error ≤ 5%, not only the view mean (T2.4b's: 3.9–4.4%).
- **T2.1** bounds the mark's cost for many lights with no range (three's default `distance` is 0; 1,024 candles changing in a frame would be about 20M mark threads, DECISIONS 2026-11-06): a radius set against each probe's own light, or such a frame treated as global. It measures `ddgi/mark` with all of `sponza-candles` changing at once. It also measures a Sponza lamp step's response on the key-probe path (T2.4c's review: only its recall was checked; the step pages run at calibrated batches, where the key path is off).
- **T2.4d** (new, from T2.6a's check 4 and T2.4c's re-timed day). Done when, on the RTX at the default budget, GI only:
  1. T2.6a's check 4: on the furnished room with the lamp off, a 30° sun step (lowered, `turnedSun` after b4e1fad) by re-tracing alone (instant sky off) has the bookmark view within 5% of settled in ≤ 0.5 s at the calibrated count; the per-pixel error and the 3 s value are recorded. T2.6a measured 23–25% at 0.5 s and 10% at 3 s (key path forced on: 12–13% and 7–8%); the cause is the order of updates (PROGRESS, T2.6a);
  2. T2.4c's check 5 on a real day (`ddgi-key-probes-cost`, elevation 5°→75°→5°, azimuth 160°, 600 frames): the update stays within Budget v2's +10% margin, back to back. It measured 3.40 ms median against 2.2 (still 2.02). First find where the extra time goes, then fix it, or build §4.5's second calibrated count while changes continue;
  3. no regressions: two-room 9/17 and Cornell 5/8 frames; Sponza's true 30° turn with the view within 5% in ≤ 15 frames and the field in ≤ 120; still scenes mark 0; the leak check and flicker hold.
- **T2.M1, rigid moving objects.** Done when, at the default budget, GI only, on the RTX:
  1. 10,000 transforms through `setTransforms` cost ≤ 2 ms of CPU (from 18–20 ms), and a moved instance's change carries `moved`;
  2. a recompile counter is in `gi.stats`, and creating and destroying 1,000 instances in random order leaves it at 0;
  3. on the furnished room, a 1 m cabinet move is within 5% of settled in ≤ 1 s;
  4. on the closed-door scene with its door moving, the dark room's leak (DECISIONS' definition) is ≤ 1% within 1 s of the door closing and stays there;
  5. no probe is used in shading before 2 blended updates (m3-moving.md 5.2; test);
  6. a still scene's field and cost match T2.4c's (mean within 0.1%, cost within 3%);
  7. the three adapter sends moved objects in one `setTransforms` batch, `InstancedMesh` update ranges send only those instances (test), and its CPU cost stays ≤ 0.2 ms still and ≤ 0.5 ms with one mover (T1.7's pages);
  8. marked probes against probes whose true change was > 5% are recorded for the cabinet and the door (the far-set guess, m3-moving.md risk 4).
  - Not in T2.M1 (stays in M3): deforming meshes, `setDeformedPositions`, `receivers` and woken air probes, runtime mesh trees, the `updateMatrixWorld` override.
- **T2.2** takes these from T2.0's review:
  - only hits a light can reach go on the work list (half of Sponza's front-face hits face away from the sun);
  - the deferred trace was at 8 storage buffers, the default limit, so the ray results and the work list share one buffer before the light grid is added;
  - check 1 adds a per-texel comparison (mean and p95 of the per-texel difference), because a mean can hide errors that cancel. The spike's 0.03% offset is a small fixed bias from the surface record's rounding (10-bit albedo, 16-bit normals), not noise;
  - don't copy the spike's shortcuts: the `.replace()` struct edit (`direct.ts:230`), the COPY_SRC usage added for a test page and read through a private cast, the light-group mask written twice. The kernel builder's `splitLeadingDirectives` (WGSL `enable` must come first) is worth keeping;
  - check 3's target is 646 probes at 2 ms on Sponza from T2.0's run in a worktree; T2.2 re-measures both paths in its own environment and compares the ratio (2.2×), not the absolute count.
- **T2.3** makes `sponza-candles` usable (T2.1's measurement, RTX 3060, `ddgi-batch-cost` startup, VALID): ready from load in **958 s** against `sponza`'s 7.8 s. The stages: load files 420.6 s (1.8 s for `sponza`), warm-up 530.8 s for 2.1 passes (1.3 s for 16), calibration 2.6 s at 1 probe a frame (`gi/direct` with 1,024 lights costs about 7 ms at 16 probes). Done when, in addition to m2-lights.md's checks: `sponza-candles` is ready from load in ≤ 2× `sponza`'s time on the same machine, and the load stage is split in the stage log (DECISIONS 2026-11-06). If the load stage's cause isn't light sampling (the candle geometry or the light walk), T2.3 fixes it anyway or hands it back with the cause.
- **T2.7** (or whichever task next touches the manifest loader) rejects a motion whose `light:<label>` target is missing or names more than one light (T2.1's review, `manifest.ts:490`).
- **T2.8** (the viewer's motions and time of day) checks that a change marked only every other frame (a host animating at 30 Hz) still counts as continuing (T2.4d's review: `continuing-change.ts:98` turns it off after one frame without a mark).
- **T2.6b, re-scoped to the sun layer** (2026-11-07). Done when, on the RTX: the sun's colour and brightness change shows exactly in the same frame (the image matches a field re-converged at the new sun within 1% mean, as T2.5's check 1); with the sun layer off, the field is within 0.1% mean of `m2`'s; the update's cost with the sun layer is within m2-lights.md §6.5's estimate (+8–10% per probe update), recorded with its memory. A sun direction change still goes through re-tracing (T2.4d).
- **T2.6b** takes from T2.5: the sun layer needs its own route for directional lights in `gi/direct`; a volume holds at most 6 layers (4 groups, the environment, the sun), with about 1.2 KB of workgroup memory left in the blend (NOTES).
- **T2.10** takes the VALID runs T2.3 still owes if a gpu-tester can't get them sooner: the light grid's build at 2,048 cells (≤ 0.1 ms; INVALID readings 0.105–0.131) and `sponza-candles`' set-up (≤ 2× `sponza`; INVALID 1.26–1.30×).
- **T2.10** times `setTransforms` with 10,000 transforms inside a three adapter frame with rendering (T2.M1's review: 1.4 ms median timed alone, 1.8 ms median / p95 3.1 without the CPU lock; T1.7 saw real frames run 2–4× slower after three's render).
- **T2.10** records the update's GPU time for the 60 frames after a big step (T2.4c's review: about 45 frames at 2.82 ms on Sponza after a 30° sun turn, from trace and blend; no check covers them). If they're over the 2 ms budget on the targets, m2-lights.md §4.5's second calibrated count while changes continue is built.
- **T2.10** times T2.14's check 1 from a cold shader cache on a quiet machine (`VITE_DDGI_SECTIONS=startup,budget`, Cornell); the review found no usable `m2` baseline (1.00 s, INVALID clock, no CSV row), so time both. It also re-runs `bistro-exterior` with light groups on in `ddgi-session-recompiles`, plus `ddgi-leak` and `ddgi-light-groups`, on a cool GPU (T2.14's review: 30 s page blocks and timeouts under load, not attributed).
- **T2.10** measures H-PLOC trees alone against tinybvh's on Bistro exterior's big meshes (T2.12's review: 1.018–1.025× on Sponza is within the same-tree noise), to decide whether the tinybvh swap can be made optional; and re-runs T2.12's pages on the Intel UHD after the merges.
- **T2.10** measures Bistro exterior's `scene_copy_cpu_ms` 3 times (VALID) to set its guard (T2.13's bar was missed; DECISIONS 2026-11-08), and reports the scene copy's staging-buffer peak (`SceneMemory.stagingBuffers`) on Sponza and Bistro exterior (estimated 188 MB, overlapping H-PLOC's 666 MB of outputs on Sponza; never measured).
- **T2.11** (the M3 breakdown) weighs T2.13's review options that need no contract change: texture averages from the smallest mip three already made, read back asynchronously (a load-time readback and a placeholder albedo: a ruling), and H-PLOC reading positions and indices from the packing's staging buffers (drops T2.12's second upload). Also GPU tests of the packing fallbacks (before `prepare`, too large to stage, replaced or removed before upload), which run only on the fake GPU today.
- **T2.11** (the M3 breakdown) gives Sponza's sun turn with the sun layer on a task of its own: the view settles at frames 16, 16, 13 against T2.4c's bar of 15 (12 with the layer off), 1.4–1.9 points further from settled from frame 1 while the field's mean matches. No cause found by reading; the first step is a run with `keyProbes` off to see whether propagation is the cause (T2.8 and its review; a known miss, DECISIONS 2026-11-08). Also: a `userData.giRange` in the three adapter, so a host can cull a light in three without shortening its GI range (T2.8's candle reach; host-facing API, a decision).
- **T2.11** (the M3 breakdown) adds, from T2.10, T2.17 and T2.18: compressed textures for Bistro and the other big scenes, through GPUTex (upstream 0.14.0 first; KTX2 the fallback; Dennis chose this 2026-11-09, DECISIONS); trimming DDGI's buffers grown by doubling (about 237 MB more than Bistro uses) once a scene settles; Bistro's jumpy calibration (fixed trace cost of 1-1.5 ms at 16 probes; 116 probes at 3.13 ms against 2 ms); counting `gi/direct`'s lit hits in T2.18's cost model (the first frames after a change, p95 2.38-2.58 ms), judged against the frame-15 bar with the sun layer on and off; and why native's Sponza GI is about 8× Cycles (its direct light is 2.9×, close to π; the texture sky is ruled out).
- **T2.8** checks Sponza's key-probe sun turn with the sun layer on against T2.4c's bar (view within 5% by frame 15). T2.6b's runs settled it at frames 13, 16 and 13; with the layer off it settles at 12 (T2.6b's review, DECISIONS 2026-11-08).
- **T2.8** makes three's lighting of `sponza-candles` usable: with three drawing every frame the page stops answering for 30 s (T2.14's re-check; the session page runs it with `VITE_DDGI_RENDER_EVERY=0`).
- **T2.10** adds the measured whole-frame p50/p95 with GI updating and shading in the same frame (M1 follow-up), and runs the Intel UHD suites in short batches with a device-hang count (M1 follow-up: two `DXGI_ERROR_DEVICE_HUNG` in long back-to-back suites).

## M3: moving geometry (prototype)

Status: **signed off by Dennis 2026-11-18 (DECISIONS 2026-11-18)**, on `main`. Started 2026-11-09 on branch `m3` (from `m2` at 5de856b; PR #4, `m2` → `main`, is open for Dennis). The design is [docs/design/m3-moving.md](docs/design/m3-moving.md). M4 runs alongside, from its own design session and its own worktrees.

**How M3 is run (Dennis, 2026-11-09):** "Lets not get hung up on timings and get things built. when we have clear gi we can make decisions on paths but we've been behind on getting the milestiones at least prototype level for a week. Id like to clear m2, m3, and m4 at least very soon."
- **Done means it works and the GI looks right.** These checks stay: accuracy against the references, leaks, response in frames, no recompiles, the oracle, and stability.
- **No timing gates.** No VALID runs, no clock rule, no 3-run medians and no re-time rounds; other GI projects share this GPU. Where a cost is cheap to read, it is recorded beside the result and not judged.
- **Misses:** one attempt, then a known miss, recorded and moved on.
- **Parallel:** tasks run side by side wherever their files don't overlap, each in its own worktree. Each gets a short Opus review for correctness.
- **Done-whens:** m3-moving.md's "Draft done-whens", with the timing items recorded instead of gated: T3.1's item 1 (already met by T2.M1), T3.2's item 3, T3.3's item 2, T3.4's cost in item 4, T3.5's item 2, and T3.7's item 2. T3.9 is the correctness checks only.

**Lanes** (tasks in one lane run in order; separate lanes run in parallel):
- **A, trees and refit** (`trace/`, the BVH builders): T3.0 → T3.2 → T3.3.

**File ownership with M4** (agreed with the M4 design session, 2026-11-09). Paths are under `packages/ddgi-core/src/platform/`.
- **M3 owns:**
  - tree building and refit: `trace/mesh-bvh*`, `gpu-mesh-trees.ts`, `top-level*`, `sah-builder.ts`, the tree swap, and `packages/bvh`;
  - the mesh and vertex packing: `scene/mesh-packing.ts`, `gpu-mesh-packing.ts`, `pack-mesh-wgsl.ts`, `trace/triangle-packing.ts`;
  - the change feed: `scene/change-log.ts`, `scene/instance-transforms.ts` and the contract's change-feed types.
- **M4 owns:**
  - `textures/`, `shade/hit-shading.ts`, `shade/direct.ts` and `shade/wgsl.ts`;
  - the leaf test in `trace/wgsl.ts`, plus the new `trace/material-test-wgsl.ts` and `trace/triangle-flags.ts`;
  - the material-record words in `scene/layout.ts`.
- **Shared, one change at a time:**
  - `scene/layout.ts`: a layout-only commit first, and tell the other session.
  - The tree-install call site in `ray-queries.ts`: every install path (tinybvh, H-PLOC, and T3.2/T3.3's new ones) runs M4's `gi/triangle-flags`. Whichever lands second wires it in.
- **B, the change feed and the adapter:** T3.1 → T3.7.
- **C, probes near moved geometry:** T3.4, after T3.1.
- **D, volumes:** T3.5 → T3.6. Like C, it changes `ddgi-technique.ts`, so D merges first and C rebases on it.
- **E, assets and loading:** T3.11, T3.12.
- **Then:** T3.8 (scenarios) → T3.9 (checks) → T3.10 (note).

| ID | Task | Agent | Size | Depends on | Status |
| --- | --- | --- | --- | --- | --- |
| T3.0 | Spike: deformed positions from three (storage `ExternalTexture`); the `updateMatrixWorld` override on Bistro interior (safe with three's and our tests; rough ns per object); refit decay on a swaying tree. A yes or no for each, with rough numbers | implementer | S | T3.11 (the tree) for the third question | done (2026-11-09; lead review: all three yes. three's GPU skinning reaches the refit through a storage `ExternalTexture` with one GPU copy, on both GPUs; the `updateMatrixWorld` override is safe at about 50 ns an object; refit decay follows the pose, so a rebuild waits for held drift (DECISIONS 2026-11-09)) |
| T3.1 | Change feed, the parts T2.M1 left: `receivers`, `deformBounds`, `setDeformedPositions` (done-when items 2-4) | implementer | S | none | done (2026-11-09; Opus review passed with notes; the deforming sheet passes the oracle in 8 poses, recompiles 0 over 1,000 instances, 100 meshes and 1,222 poses; until T3.3 a pose rebuilds the mesh's tree, held while any tree is building) |
| T3.2 | Runtime mesh trees: GPU builds for added and updated meshes, tinybvh refinement | implementer | M | T3.0 | done (2026-11-09; short Opus review passed with notes; an updated and a new mesh are hit from frame 2 on both scenes, the GPU tree's ranges are freed in the swap frame, recompiles 0, a still scene does no extra submits; the SPEC exception now covers any frame that uploads meshes, provisional, flagged to Dennis) |
| T3.3 | Deforming meshes: `GpuMeshRefit` wired, scene-copy positions, geometric normals, movement word; H-PLOC rebuild when the refit drifts. Also from T3.1's review: replace `applyPoses`' rebuild with the refit and remove the hold on every tree build (`gi-platform.ts`), and check what rays hit between a pose reaching the scene copy and its tree or refit (item 1 covers it). From M4: `GpuMeshRefit` rewrites whole triangle records, so it keeps word 7 (the triangle flags) or calls `PlatformRayQueries.requestTriangleFlags(meshSlot)` after a refit, or a deforming mesh with glass or cutouts loses its flags (T4.2). From T3.2's review: a pose frame submits one H-PLOC batch with its own buffers, and a slow readback on a shared GPU (T2.12 saw 8-10 s) can leave many batches in flight; the refit removes the per-pose builds | implementer | M | T3.1, T3.2 | done (2026-11-10; short Opus review passed with notes, fixes in; poses (CPU, GPU buffer, texture) refit in the frame's encoder and pass the oracle the next frame on both GPUs; the wind tree's GI follows its sway (0.133 s behind, r 0.977); a held bend rebuilds once; refit stats read at most every 15 frames (provisional SPEC exception, for Dennis); deforming meshes shade with flat triangle normals at GI hits; the unfold miss is with T3.4) |
| T3.4 | Probes near moved geometry, the parts T2.M1 left: woken air probes (M1's "probes in empty space"), deforming movers, the far-set guess scaled to the mover's size (T2.M1 item 8) | implementer | M | T3.1 | done (2026-11-10, Mac M5 Pro: merged into `m34` after the lead's review; the morph-unfold miss fixed: a moving or deforming object that gives off light marks every active probe in reach, as a light does (`DirtyRegion.emits`); exact-box morph back to rest 6.3% / 14.6% → 0.07% / 0.50%; door leak 0.188%; recompiles 0) |
| T3.5 | Several volumes: shared atlas, volume table, priority blending | implementer | M | none | done (2026-11-09; short Opus review passed with notes, fixes in; no seam (0.0105% / 0.042% a frame across the fade band), one volume equals M2, recompiles 0, three-neutrality passes; a scheduler per volume with carried shares and a floor per volume (DECISIONS); the `ddgi-warmup` page race fixed, 3 of 3 pass) |
| T3.6 | Camera-following volume: wrapping slots, `ddgi/scroll`, window rule, fade-in | implementer | M | T3.5 | done (2026-11-10; short Opus review passed with notes; Bistro fly-through at 5 m/s with 1.5 m spacing: edge band 0.074% / 0.27% a frame after the 10-spacing fade, 0 of 23,808 entering slots used before 2 updates, memory fixed, 0 planes cleared at rest, recompiles 0; fixed volumes unchanged) |
| T3.7 | three adapter change tracking v2 (added and removed objects without `markChanged`, deforming meshes, the `updateMatrixWorld` override if T3.0 says yes) | implementer | M | T3.0, T3.1 | done (2026-11-09; short Opus review passed, fixes in; Bistro interior: added and removed objects reach GI without `markChanged`, unmarked moves sent after 2 frames (bar 10), recompiles 0; deforming meshes send CPU poses through a `PoseSource` seam for T3.3; known miss for T3.4: a morph unfolding with its exact box leaves GI 6-13% off, marks must cover the previous pose too) |
| T3.8 | Viewer scenarios and their manifest motions (furniture, door, cabinets, wind tree, fly-through) | implementer | M | T3.3, T3.4, T3.6, T3.7, T3.11, T3.12 | done (2026-11-10, Mac M5 Pro: merged into `m34` after the lead's review; five scenarios play, recompiles 0; the fly-through's start fixed (the camera volume warms up at the viewer's camera: 4,666-8,192 → 174-236 probes a frame 6 s in); choosing it again mid-flight still runs at full rate for 3-4 s, a known miss in DECISIONS; review fixes in) |
| T3.9 | M3 checks: correctness on the RTX and the Intel UHD (oracle, accuracy, leak, response in frames, stability, recompiles at 0), the door and cabinet against Cycles renders of their end states; rough costs recorded | gpu-tester | M | T3.0-T3.8 | done (2026-11-10, Mac M5 Pro: the smoke runs of M1's, M3's and M4's correctness pages all pass on the merged `m34` after four fixes; PROGRESS) |
| T3.10 | M3 note and the next breakdown | main session | S | T3.9 | done (2026-11-18: signed off by Dennis; the joint note is docs/milestones/M3-M4.md, its questions answered 2026-11-18, DECISIONS) |
| T3.11 | The wind tree's asset: find or generate, convert, record licence | wrangler | S | none | done (2026-11-09; no downloadable tree in range, so generated: `tools/assets/convert/wind-tree.ts`, seed 20261109, 50,474 triangles, about 9 m, MIT; wind weight in `COLOR_0` alpha; scene `courtyard-wind-tree`; lead review by eye: reads as a tree, a clump of low leaf cards sits around the trunk's foot) |
| T3.12 | Bistro exterior runs in the viewer on a 6 GB card (the fly-through needs it): compressed textures through GPUTex (upstream gputex 0.14.0 first; HomeFig's method only after asking Dennis; KTX2 the fallback). If that isn't quick, the viewer applies T2.17's texture cap as a setting and GPUTex moves to Later | implementer | S | none | done (2026-11-09; upstream gputex 0.14.0, BC7 and BC5 normal maps, mips kept for M4; Bistro 44 ms a GI frame instead of 591, 4.86 GB peak; the GI-averaged maps stay uncompressed until M4's T4.3 flips `textures` to `all-compressed` in `ui/url-state.ts`; short Opus review passed, clearcoat-normal fix in) |

## M4: real materials and scenes (prototype)

Status: **signed off by Dennis 2026-11-18 (DECISIONS 2026-11-18)**, on `main`. Started 2026-11-09 on branch `m4` (from `m2` at 5de856b), beside M3. The design is [docs/design/m4-materials.md](docs/design/m4-materials.md); its "Tasks" section has the done-whens, and its "Files and M3" table lists what each task touches. Prototype level (Dennis, 2026-11-09): done-whens are correctness and the look, costs are noted and never gated, no VALID runs. One fix attempt for a missed check, then a known miss in DECISIONS. Opus review before a task is merged into `m4`. Answers: DECISIONS 2026-11-09, "M4 answers".

| ID | Task | Agent | Size | Depends on | Status |
| --- | --- | --- | --- | --- | --- |
| T4.1 | Texture store and textured hits: store (texture array, 256² layers), four sources incl. compressed mips, `gi/texture-store`, uv mapping and ray-cone mip in `giSurfaceAt`, albedo/emissive/metalness at hits, the contract additions | implementer | M | none | done (merged into `m4` 2026-11-09 after Opus review, review fix in; furnace +0.21%, uv board 100% within 0.01, sources identical, RTX and Intel). **Until T4.3 lands, three scenes on `m4` see textures twice** (the adapter still folds averages and passes the textures): no viewer or Cycles comparison of a three scene on `m4` counts before T4.3. Carried: a source that throws leaves its textures pending forever (catch per source, `platform-scene.ts:1118`); alpha-weighted colour mips for cutout foliage (`texture-store-wgsl.ts:235`), before T4.5's look; value tests for linear (`srgb: false`) fills and `bc7-rgba-unorm-srgb` |
| T4.2 | Materials in the walk: consult and instance flags, `giMaterialPasses`, `gi/triangle-flags`; see-through and single-sided first, then cutout and stochastic; CPU reference | implementer | M | T4.1 for cutouts and stochastic | done (both halves merged into `m4` 2026-11-09 after Opus reviews, fixes in: first half 82ef2c3, second half ab92726). Cutout walls 100% agree with the CPU reference on certain rays; light through a 50% checker 48.9% and stochastic 0.5 48.4% of open (RTX; Intel for the first round); stochastic hash seeded per frame; alpha-weighted mips; flags rewritten only on relevant edits. On the combined `m4` (cce030e): all M4 pages pass on the RTX. On the Intel UHD, `ddgi-alpha-wall` and `three-m4-textured-furnace` pass, and `trace-materials` passes alone (44 s) but twice hit `DXGI_ERROR_DEVICE_HUNG` when run third in a 3-page batch. That is the Intel's known hang in batches (T3.19), so run it alone on the Intel; a known issue on a correctness device, not a miss of the check |
| T4.3 | three adapter materials v2 (design section 4): real textures incl. `CompressedTexture`, fold and mesh exclusion removed | implementer | M | T4.1, T4.2 | done (merged into `m4` 2026-11-09 after Opus review, fixes in; three-built furnace +0.205%, glass pane 0.167 vs 0.0086 opaque, BC7 `CompressedTexture` within 1/255, flipY exact, RTX and Intel; Sponza all 448 meshes traced, 55 textures, 19.2 MB). Until T4.2b merges, cutouts and blended materials block like opaque ones (Sponza's decal is a black sheet). At the M3/M4 join: merge with T3.7's `scene-sync.ts` (regions in T4.3's PROGRESS), flip the viewer's `textures` default to `'all-compressed'` and empty `GI_AVERAGED_SLOTS` (T3.12's files) |
| T4.4 | Textured scenes: four procedural ones, every real scene textured, textured Cycles HD references, debug views | wrangler + implementer | M | none (pages need T4.1) | done (both parts merged into `m4` 2026-11-09 after Opus reviews, must-fixes in: 3747c23, 2921d0a). Four procedural scenes; real scenes open textured; Cycles `--textured` references (procedural four, Sponza, furnished room, courtyard; Bistro exterior and interior and `sponza-full` rendering on the CPU into `bench/references/`); `describeScene` passes materials or throws; Compare reads textured references; debug views (albedo at hits; `gitextures` toggle, which reads mean colours when off). **For T4.5:** move `alpha-checker` and `glass-wall` volume centres to x 4.25: a probe layer sits inside the shared wall at x 4.0 (room B/room A 1.56% and 1.47%; 0.56% and 0.10% with the volume moved 0.25 m) |
| T4.5 | M4 check and note, with Dennis's look | main session + gpu-tester | S | all; T3.12 for Bistro | done (2026-11-18: signed off by Dennis; the joint note is docs/milestones/M3-M4.md, its questions answered 2026-11-18, DECISIONS) |

## M5: coverage pass (prototype)

Status: **signed off by Dennis 2026-11-18 (DECISIONS 2026-11-18)**, on `main`. Started 2026-11-10 on the Mac, on branch `m34` (M3 and M4 land first). Design: [docs/design/m5-coverage.md](docs/design/m5-coverage.md), which has each task's done-when. Under DECISIONS 2026-11-10 "Feature coverage before tuning": each feature works end to end, passes one smoke check and gets Dennis's look; numbers are recorded, never gated, except the correctness bars the design names. One fix attempt for a miss, then a known miss. The examples site and release prep come after this pass, as T5.6 and T5.7 ([docs/design/m5-release.md](docs/design/m5-release.md), 2026-11-10).

| ID | Task | Agent | Size | Depends on | Status |
| --- | --- | --- | --- | --- | --- |
| T5.1 | Quality presets: `ddgiPreset('low' / 'medium' / 'high')`, `recommendPreset`, three's `preset` option, the viewer's picker, `ddgi-presets` page | implementer | S | M3, M4 on `m34` | done (2026-11-10, Mac M5 Pro: merged into `m34` after the lead's review; low / medium / high pass the furnace (0.31 / 0.26 / 0.22%), the leak (0.86 / 0.87 / 0.87%) and recompiles 0; `high` is today's defaults, checked by a unit test; the viewer's picker works on Sponza and Bistro; Intel and iPad runs of `low` wait for those devices) |
| T5.2 | Saved bakes: `saveField` / `loadField` in core (the `.ddgibake` format, scene fingerprint, stale load), three's `saveBake` / `loadBake`, viewer Save and Load | implementer | M | M3, M4 on `m34` | done (2026-11-10, Mac M5 Pro: merged into `m34` after the lead's review; the first frame after a load within 0.01% of the saved page on Cornell, two-room and two-room with a camera volume, no warm-up, recompiles 0; a wrong layout refused by name; stale loads settle in 4 and 7 frames; Sponza's viewer round trip 0.03% of pixels off; Sponza's bake 31.2 MB) |
| T5.3 | The `LightProbeGrid` export: L2 SH projection on the GPU (core), `toLightProbeGrid` and the JSON (three), check against live GI | implementer (scout first for PR #33658's format) | M | M3, M4 on `m34` | done (2026-11-10, Mac M5 Pro: merged into `m34` after the lead's review; exported grid against live GI: Cornell −0.3 to −1.0%, Sponza −8.1% / −4.0%; JSON round trip exact; our own `formatVersion: 1` since three's PR #33658 is still open and stores fixed point; `LightProbeGridWebGL` and exporting a second or camera volume not run) |
| T5.4 | Docs: platform and three API docs brought up to date, the adapter guide on the hello example (hello passes every material slot and textures first), getting-started, the docs-cover-exports check | implementer | M | T5.1-T5.3 for their sections | done (2026-11-18: signed off by Dennis. 2026-11-10, Mac M5 Pro: merged into `m34` after the lead's review; API docs up to date for M2-M5, `docs/guides/adapter.md`, getting-started in the README; hello passes every material slot and texture (`three-neutrality` 0 pixels off); `pnpm test` now fails on an export the docs don't name and `pnpm typecheck` checks the docs' code examples) |
| T5.5 | Coverage note (`docs/milestones/M5.md`): every v1 feature, its check, its Mac result, its known misses, then the proposed tuning pass; Dennis's look | main session | S | all M5 | done (2026-11-18: signed off by Dennis. 2026-11-10: docs/milestones/M5.md; every v1 feature has a passing check on the M5 Pro; the proposed tuning pass is in it; his picks are the tuning pass below) |
| T5.6 | The examples site: a numbered gallery (01-hello to 10-debug, plus `webgpu/hello`), a landing page, `pnpm examples` and `examples:build` with `PAGES_BASE`, a Pages workflow (committed, not enabled), a GPU page that loads every built example. Design: [docs/design/m5-release.md](docs/design/m5-release.md) | implementer | M | T5.1-T5.4 | done (2026-11-18: signed off by Dennis. 2026-11-10, Mac M5 Pro: `examples-site` passes on the built site under `/webgpu-ddgi/`, 2.4 MB; screenshots in `bench/screenshots/t5.6/`) |
| T5.7 | Release prep: README, LICENSE and THIRD_PARTY in the tarball, a packed-consumer check, CI and publish workflows (written, not run against npm), `CHANGELOG.md`, `docs/releasing.md`, the README as a public home. Nothing is published | implementer | M | T5.4 | done (2026-11-18: signed off by Dennis. 2026-11-10, Mac M5 Pro: every CI step passes in order on a fresh clone with Node 24 and npm 11; the packed-consumer check passes (tarball 581.7 kB, 17 files, README, LICENSE, THIRD_PARTY and licence texts in it); actionlint passes both workflows; nothing published. The one-time steps in docs/releasing.md are Dennis's, before 0.1.0) |
| T5.8 | Small leftovers: export `DDGILightGroups`, two stale comments | main session | XS | T5.4 | done (2026-11-10: `DDGILightGroups` exported from `@pmndrs/ddgi/core` and named in platform.md; `SceneApi.updateMesh` and the procedural sky's comments brought up to date; typecheck, lint and 1,535 unit tests pass) |
| T5.9 | The `pmndrs/gi` rename (DECISIONS 2026-11-10): `@pmndrs/gi` with `/core` (platform only), `/ddgi`, `/three`; `GIVolume` → `DDGIVolume` (`fromDDGIVolume` → `fromSettings`); Pages `/gi/`; workspace folders later | implementer | S | T5.7 | done (2026-11-10, Mac M5 Pro: merged into `m34` after the lead's review; lint, typecheck, unit and release tests, build, packed check pass; `examples-site`, `three-composite`, `three-neutrality`, `three-sponza-gi` pass) |

## Tuning pass (after the M3-M5 sign-off)

Status: **todo**. Dennis's picks on the close-out page (2026-11-18): these five are "next", in this order; GI behind glass in the composite and timing on the target GPUs are "later" (TASKS Later). Timing runs are deferred (Dennis, 2026-11-18: "defer timing runs for another time"). Prototype rules still hold: numbers recorded, not gated, except the correctness bars.

| ID | Task | Agent | Size | Depends on | Status |
| --- | --- | --- | --- | --- | --- |
| TU.1 | `low` on its devices: run the `low` preset on the iPad and the Intel UHD (furnace, leak, recompiles, the viewer), and size its warm-up batch (`maxProbesPerFrame` is 4,096 under every preset today) | gpu-tester, then implementer | S | the Windows laptop and an iPad | todo |
| TU.2 | Re-run `trace-oracle`, `trace-materials` and `trace-bench-quick` on the RTX and the Intel after the Metal shared-edge fix (`giEdgeFunction`, 2026-11-10) | gpu-tester | XS | the Windows laptop | todo |
| TU.3 | Safari on the M5 Pro: the GPU suite's correctness pages and the viewer, as SPEC's target browser | gpu-tester | S | none | todo |
| TU.4 | Accuracy against Cycles: M4's remaining table (Bistro exterior and interior, courtyard tree, alpha checker, sponza-full, FLIP), the door and cabinet end states, and the furnished scenes again on today's no-GTAO default (`m4-textured-accuracy` still turns GTAO on unless given `--no-occlusion`: switch it to the default first). Bistro's dark side street is judged here | gpu-tester + implementer | M | the Cycles references from the NAS | todo |
| TU.5 | The cheap items from the DDGI vs RTXGI comparison ([docs/divergence.md](docs/divergence.md)): DV-01 (back-face rays, may matter for Sponza) and DV-03 (the irradiance probe view), then whatever else Dennis picks there | implementer | S each | none | todo |

## radiance-cache track (second technique)

Status: **phase 1 started 2026-11-09** on branch `radiance-cache` (cut from `m2` at 5de856b), alongside M2's close-out. The design, the divergence register and the later phases are in [docs/design/radiance-cache.md](docs/design/radiance-cache.md); Dennis's decisions are in DECISIONS (2026-11-09, "radiance-cache"). The technique is a faithful port of AMD's GI-1.0/1.2 (Capsaicin v1.3) first; performance ideas are levers beside it.

Rules for every row:
- **Separable.** Each row is built on its own branch (`rc1-<n>`, from `radiance-cache`) and merged into `radiance-cache` after review, so each platform change can also merge to the main line alone.
- **Subgroups first** with one declared fallback and a test of it (DECISIONS 2026-11-04).
- **Hardware:** the RTX 3060 is the target and GPU timing is taken GPU-loaded under the bench lock (`tools/reference/bench-lock.ts`), VALID runs only; the Intel UHD is correctness only. Never time anything while a native Capsaicin run (`scene_viewer.exe`) is going.
- **Every divergence from Capsaicin** (a three backend step, a missing feature) gets a register row in the design doc, numbered from D13, with its kind, reason and way back.
- An Opus review against the done-when before a row is marked done.

| ID | Task | Agent | Size | Depends on | Status |
| --- | --- | --- | --- | --- | --- |
| RC1.1 | Screen inputs: `FrameScreen` gets albedo; `ddgi-three` fills it from the host's MRT scene pass every frame; a debug view of every channel on Sponza; its cost measured | implementer, then gpu-tester (cost) | M | none | done (2026-11-09; Opus review failed on transparent objects overwriting the channels, fixed; re-review passed with notes, fixed by the lead; both GPUs; backend step D13; cost deferred, DECISIONS) |
| RC1.2 | Lint rule: technique code imports only `platform/contract` (SPEC claims this is enforced; only the engine boundary is) | main session | S | none | done (2026-11-09; Opus review passed with notes, all fixed: a local rule resolves each import to its file) |
| RC1.3 | Platform GPU utilities: prefix scan, stream compaction, indirect-dispatch arguments, shared with `packages/bvh`'s scans; subgroups first with one fallback | implementer | M | none | done (2026-11-09; Opus review passed with notes, all fixed; correct on both GPUs at 0 to 10 M and with counts up to 2^32-1, subgroup path and fallback identical; cost run and bvh build re-measure deferred, DECISIONS) |
| RC1.4 | Feature checks in three r187 and Chrome on the RTX 3060 and Intel UHD: indirect compute dispatch, `texture-formats-tier2` read-write `rgba16float`, subgroups, `shader-f16`, `timestamp-query` | implementer | S | none | done (2026-11-09; Opus review passed with notes, all fixed; all five features work on both GPUs on our device and three's; D14 three's own device keeps default limits, D15 f16 overflow differs) |
| RC1.5 | Ray cost table: closest-hit and any-hit Mrays/s for Capsaicin-shaped ray sets on Sponza with curtains and Bistro | implementer (ray sets), then gpu-tester (timing) | M | none | done (2026-11-09; Opus review failed on shadow rays per hit, fixed; 3 VALID runs per scene at a pinned 1,200 MHz: tracing alone 12.4 / 21.9 / 23.8 ms on Sponza / Bistro interior / exterior, 2.6-3.2x native's tracing-and-population group) |

### RC1 done-whens

- **RC1.1 Screen inputs.**
  - `FrameScreen` (`platform/contract/frame.ts`) gets an optional `albedo`: diffuse albedo in linear rgb, with metalness folded in exactly as the adapter folds it for GI albedo (T1.7), its format and encoding documented. DDGI and every existing page work unchanged without it.
  - `ddgi-three` fills `FrameScreen` from the host's MRT scene pass (`pass(scene, camera)` with `mrt({...})`, as the T1.14 composite): depth, normal, velocity, albedo and instance id (instance id only when a technique asks for it), every frame, through resizes, without a copy when three's own textures can be handed over. If three r187 won't expose a target as a `GPUTexture` without reaching into its backend, the smallest version-pinned step is taken, tested against r187 (a check that fails loudly on another version), and recorded as a register row and a DECISIONS entry.
  - A debug view in the core (engine-free) shows each channel: depth, normal, velocity, albedo, instance id. A GPU test page on Sponza (with curtains) checks, by readback, that each channel holds what three drew (normal and albedo against three's own MRT readback at sample pixels; instance id against the adapter's mesh-to-instance map; velocity zero for a still camera and non-zero for a moving one) on the RTX and the Intel UHD. The viewer gets the same views in its debug menu.
  - The cost of filling `FrameScreen` (the extra MRT outputs and anything else the fill adds, against the T1.14 composite's own MRT) on Sponza at 1280×720 and 2560×1440, GPU-bound on the RTX, VALID, recorded in `bench/results/` and PROGRESS.
- **RC1.2 Lint rule.**
  - Production files under `packages/ddgi-core/src/techniques/**` may import from `platform/contract` (and their own technique) only; any other platform path fails `pnpm lint` with a message that says why. Tests under `techniques/` may use platform internals as fixtures (stated in the rule's comment).
  - The platform may not import a technique (`platform/**` → `techniques/**`), except the core's entry points that assemble them.
  - Checked by adding a forbidden import to a DDGI file and to a platform file (lint fails with the message), then reverting. SPEC's "lint-enforced" sentence matches what the rule does. `pnpm lint && pnpm typecheck && pnpm test` pass.
- **RC1.3 Platform GPU utilities.**
  - Exclusive prefix scan of `u32`, stream compaction (kept indices and their count, from a flag or predicate buffer), and an indirect-dispatch arguments writer (a GPU count to `[x, y, 1]` workgroups, split over two dimensions past `maxComputeWorkgroupsPerDimension`), as platform modules a technique uses through `platform/contract`.
  - One scan implementation: `packages/bvh`'s subgroup scan is reused (or moved and shared), not copied. Subgroups first, with one declared fallback (workgroup memory) that doesn't assume subgroup lane order (NOTES).
  - Tests: CPU references for every utility; GPU tests on the RTX and the Intel UHD at sizes 0, 1, one tile ± 1, and 10 M, with the subgroup path and the fallback (on a device without `subgroups`) giving identical results, and the indirect arguments consumed by a real `dispatchWorkgroupsIndirect`.
  - The cost at 1 M and 10 M items on the RTX (one VALID run), in `bench/results/` and PROGRESS.
- **RC1.4 Feature checks.**
  - A GPU page runs a functional check of each feature (not just `features.has`): a GPU-written indirect dispatch runs the right number of workgroups; a kernel reads and writes the same `rgba16float` storage texture with `texture-formats-tier2`; `subgroupAdd` gives the right sum and the sizes are reported; an `f16` kernel computes the right result; timestamps come back ordered and non-zero.
  - Each is checked on a device our code creates, and on the device three r187's `WebGPURenderer` creates for itself (which features three asks for), on the RTX and the Intel UHD in Chrome.
  - A results table (feature × GPU × our device / three's device) in PROGRESS and rows in `bench/results/checks.csv`. Every gap gets a register row (D13 on) with its declared fallback (the design doc's "Feature fallbacks" table).
- **RC1.5 Ray cost table.**
  - Ray sets shaped like Capsaicin's defaults, from the scene's first bookmark at 1280×720 (the cameras `native-capsaicin.md` used): one probe per 16×16 tile placed by a primary ray through a jittered point in the tile (80 × 45 = 3,600 tiles; tiles that see sky place none), 64 cosine-weighted rays per probe about the surface normal (closest hit, no length limit); multi-bounce rays, 0.3× the probe rays, one cosine ray each from a random subset of probe-ray hits (an approximation of Capsaicin's multi-bounce cells, stated as such); one shadow ray (any hit) per hit of both sets toward one light sampled per hit (the sun where the scene has one).
  - Closest-hit and any-hit Mrays/s for each set through the platform's tracer, on `sponza` (with curtains), `bistro-interior` and `bistro-exterior`, on the RTX, 3 VALID runs (median and spread), into `bench/results/trace.csv`. The Intel UHD runs once for correctness (no timing claims).
  - A table in PROGRESS: per scene, each set's ray count, rate and predicted ms, their sum, beside Capsaicin's native "probe tracing and cell population" group (the design doc's yardstick). This predicts the baseline's tracing cost; it doesn't change the baseline.

### radiance-cache phase 2: screen probes, first image

Started 2026-11-09 on Dennis's word through the starting session ("lets build what we can before we hit our usage limit"): **no timing runs until he asks** (correctness gates only, GPU checks through the bench lock with `--no-cool-wait`; every skipped timing is written down as deferred). Phases 2-4 run in the plan's order without waiting for a sign-off; only a decision that blocks the whole project stops the track.

The port reads AMD's source directly: Capsaicin v1.3 at (on the Windows laptop) `C:/Users/denni/Documents/dev/ddgi/.claude/worktrees/amd-gi-cache-research-e46e2e/native/references/capsaicin/src/core/src/render_techniques/gi1/` (`gi1.comp`, `gi1.h`, `gi1.cpp`, `screen_probes.hlsl`, `gi1.hlsl`) and the shared `src/core/src/math/` files, mapped pass by pass in the research folder's `source-map.md` §3 (rows 2-11, 23, 25, 27, 28 for this phase). Ported WGSL keeps AMD's MIT notice.

| ID | Task | Agent | Size | Depends on | Status |
| --- | --- | --- | --- | --- | --- |
| RC2.1 | Technique skeleton and screen probes: spawn (16×16 tiles, Halton jitter), compact, sample (64 rays, one per hemi-octahedral cell), trace, light hits directly, blend into the probe atlas; debug views of the atlas and the mask | implementer | M | RC1.1, RC1.3 | done (2026-11-10; Opus review passed with notes; both GPUs; the sky check amended to Capsaicin backup rule; viewer entry moved to RC2.2; D16-D21) |
| RC2.2 | First image: the probe mask mips, SH projection, interpolation per pixel, the shading module by `screenUV`, the three composite and the viewer's technique entry; the Lambert check | implementer | M | RC2.1 | done (2026-11-19; Opus review: merge after fixes, all 4 fixed incl. the three jitter sign; RTX 3060 and Intel UHD; register D25-D29 and D34; D25 settled by Dennis's fix-by-default rule, see DECISIONS; see PROGRESS) |

- **RC2.1 done-when.**
  - `packages/ddgi-core/src/techniques/radiance-cache/` exists, importing only `platform/contract` (the RC1.2 rule), declaring the screen channels it needs, with its settings as Capsaicin's defaults (`gi1.h` `RenderOptions`, `probe_size` 8, `QuarterSpp`), each named and cited.
  - Ported passes, each a WGSL kernel named after its Capsaicin entry point with the source line cited: `ClearCounters`, `SpawnScreenProbes` (Halton from `gi1.hlsl`; sky tiles spawn none), the scan and `CompactScreenProbes` through RC1.3's compaction, `SampleScreenProbes` as Capsaicin runs it for a probe with no history (ray guiding and the previous-probe taps arrive in phase 4), `PopulateScreenProbes` with the platform's closest-hit query, and `BlendScreenProbes` without temporal history (phase 4). Dispatches sized on the GPU go through RC1.3's indirect arguments.
  - **Staging, not a divergence:** until the world cache (phase 3), every probe-ray hit takes Capsaicin's cache-bypass path, lit directly through platform hit shading (`giSurfaceAt`, `giDirectIrradiance`, emission; misses read `giSkyRadiance`). Capsaicin lights emissive and short-ray hits this way; phase 3 switches the rest to the cache. Written in the design doc's phase notes, not the register.
  - GPU tests (RTX and Intel, correctness):
    - spawn count and positions match a CPU port of the spawn rule on a fixed frame;
    - compaction keeps exactly the non-sky tiles;
    - every ray direction lies in its cell and on the normal's side;
    - a probe under a constant white sky with nothing above it reads the sky's radiance in every cell within 1%. *Amended (lead, 2026-11-09, RC2.1 review):* directions stored as halves (Capsaicin's `packHalf3`) move about 0.1% of rays into a neighbouring cell, so a reached cell is checked against the sky within 1% and an empty cell against Capsaicin's backup rule (`gi1.comp:1212-1256`, confirmed by the review); rays within 0.02 of a cell edge or grazing within 1e-3 are counted rather than failed, and a probe whose cell assignment is undecided at an edge is skipped and counted (RTX: 1 probe).
  - Debug views (probe atlas, probe mask) in the core, engine-free. *Amended:* the viewer's menu entry moves to RC2.2, which adds a screen position to the contract's debug colour (`giDebugColor`).
- **RC2.2 done-when.**
  - Ported passes, each with its source lines cited:
    - `FilterProbeMask` (the mask mip chain);
    - `ProjectScreenProbes` (SH L2, nine coefficients per probe);
    - `InterpolateScreenProbes` without SSGI's bent normal (phase 4).

    Units are checked against `ResolveGI1` so the output is irradiance E, and three's composite applies `albedo/π × E`.
  - The technique's shading module returns E by `screenUV` from the interpolated screen texture. The three composite (T1.14) and the viewer's technique switcher take it like DDGI, with GI off/on/only and the split wipe against DDGI.
  - **Lambert check (GPU test):** an open plane under a constant sky of radiance L reads E = πL within 1% per pixel (mean and p95) at the interpolated output.
  - A single-bounce image on Cornell, two-room and Sponza in the viewer, with no errors and no NaN or negative pixels (checked by readback). A contact sheet in PROGRESS beside DDGI's image of the same view.
  - Debug views of the SH (irradiance from the probe's SH along the normal) join the atlas and mask views, and all three reach the viewer's debug menu (moved from RC2.1): `giDebugColor` gets the pixel's `screenUV` as `giIrradiance` has it.
  - From RC2.1's review: a direction-varying sky (or an occluder) in a GPU test, so an atlas transposed in its tile or a hemi-octahedral map mirrored the same way on the CPU and GPU fails; fixed Capsaicin reference values in the CPU map tests; spawn checked at several frames; the uniform block packed by named offsets, not bare word indices (`radiance-cache-technique.ts:487-493`); a unit test that `encodeScreen` gets the frame's camera.
  - `FrameCamera` gets an optional jitter (NDC offset of this frame's projection) so a host that jitters for the upscaler gets positions rebuilt at the depth's own jitter (RC2.1: half a pixel off otherwise). The three adapter calls `encodeScreen` after `encode`.

### radiance-cache phase 3: the world cache, faithful

Same rules as phase 2 (no timing; correctness gates; faithful port with Capsaicin's source cited; register rows from the next free number). The interface between the screen cache and the world cache is in the design doc ("The world-cache interface"). Source map rows 3, 11, 12, 15, 16, 19-22 and §4.2.

| ID | Task | Agent | Size | Depends on | Status |
| --- | --- | --- | --- | --- | --- |
| RC3.1 | The hash grid: one shared WGSL function for the cell descriptor and both hashes; CAS insert into 16-slot buckets, find, decay stamps and `PurgeTiles`; buffers packed at Capsaicin's default capacity; the `WorldCache` interface with the hash grid behind it | implementer | M | RC2.1 | done (2026-11-10; Opus review failed on the ray offset, fixed; both GPUs; 1,024 MiB at default capacity; wiring moved to RC3.2; D22-D24) |
| RC3.2 | Population and resolve: probe-ray hits insert cells (`PopulateScreenProbes`' cache half), `GenerateReservoirs` (a light per hit through the platform light grid, D4, with the bypass rules), `PopulateCells` (shadow ray, quantised accumulation), `UpdateTiles` (temporal accumulation and the tile mips), `ResolveCells` (filtered lookup into the probe rays); phase 2's staging retired | implementer | M | RC3.1, RC2.2 | done (2026-11-19; Opus review: merge after fixes, all 8 fixed; RTX 3060; the NaN scan runs in a GPU page, the viewer image check moves to RC2.2, see PROGRESS) |
| RC3.3 | GI-1.2 multi-bounce: `PopulateMultibounceCells`, `GenerateMultibounceReservoirs`, `UpdateMultibounceCells` and the indirect value array | implementer | M | RC3.2 | done (2026-11-19; Opus review: merge after small fixes, all 6 fixed; RTX 3060; furnace sections in `radiance-cache-world-cache`, seven mutations caught; D35, D36; Capsaicin's 3.33x bias, miss bias and bypass index mix-up kept for now, fixed in RC3.5 under DECISIONS 2026-11-19) |
| RC3.4 | Sky light in the cache's light choice (register D33's way back): the platform's sky as a light in `giDirectChooseLight` (its radiance map importance-sampled, with a shadow ray), shared with DDGI's `gi/direct`. Not a phase-3 gate: outdoor scenes (Bistro exterior) miss the sky light on cache hits until it lands | implementer | M | RC3.2 | review (2026-11-19; branch `rc3-4`; cosine-weighted sky sampling with an SH weight, opt-in bit in `giDirectChooseLight`, DDGI unchanged; register D33 closed, D62; world-cache page sections 15-18, seven mutations caught; RTX 3060; follow-up: unlit hits count as samples of no light, D63, section 19; sky-light row renumbered D62 after RC4.1's D40-D48; Bistro exterior screen irradiance × 1.067 with the sky light, see NOTES) |
| RC3.5 | Fix Capsaicin's confirmed world-cache errors (DECISIONS 2026-11-19; no reproduce-Capsaicin switch): the GI-1.2 second bounce's 1/(1-p) = 3.33x bias and the 1/P(hit) bias from misses adding no count (an unbiased estimator: every first-bounce entry counts), and the two bypass paths that add light to the wrong probe ray (PopulateCells and UpdateMultibounceCells, plus the bypass flag read at the wrong index). Register rows with the derivation; the furnace reads the analytic ρ²E/π; the corner's misdirected additions read 0 | implementer | S | RC3.3 | done (2026-11-19; Opus review: correct, one sparse-cell bias found and fixed (indirect value kept on frames with no sample); branch `rc3-5`; RTX 3060; D37, D38; the estimator is the kept-ray mean with misses counted, the lower-variance unbiased form; furnace 1.0015 × ρ²E/π, strips 0.998 × its half, corner misdirected 1,164 → 0; five mutations caught; review fixes: indirect value kept on frames with no sample, sections 13-14, three more mutations) |
| RC3.6 | More than two bounces in the world cache (RC4.4's accuracy page: closed rooms read 22-56% dark against Cycles because a second-bounce hit reads only its cell's direct value). Compare, on the accuracy page against Cycles: (a) the second hit also reads its cell's indirect value (cache feedback, every bounce over frames); (b) Capsaicin's `useTemporalFeedback` (last frame's lit image at hits); (c) P6.3's DDGI second bounce and P6.1's SHaRC every-bounce paths as they land. Pick the default by accuracy, lag and flicker; register row | implementer | M | RC3.3, RC4.4 | todo |

- **RC3.1 done-when.**
  - The descriptor, the bucket hash (pcg chain) and the checksum (xxHash chain, never 0) live in one WGSL function that every pass calls, ported from `hash_grid_cache.hlsl:96-192` with `src/math/hash.hlsl`. A CPU port of the same function gives the same bits on a fixed set of hits (unit test), and the GPU agrees with it (GPU test).
  - The capacity is Capsaicin's default (2^14 buckets × 16 tiles × 85 cells; value, indirect value and update-scratch arrays), allocated in a few large packed buffers under the adapter-maximum limits (D6). Its memory is in the technique's stats. A device whose limits can't hold it gets `GILimitError` naming the limit (D14: three's own device does).
  - GPU tests, on the RTX and the Intel UHD: insert then find; two different tiles colliding in a bucket both found; a full bucket refusing the 17th tile, counted as an overflow; a tile untouched for 50 frames purged by `PurgeTiles`, including across the frame counter's wrap; concurrent inserts of the same tile from many threads giving one slot.
  - `WorldCache` as in the design doc (plus `reset`), the hash grid its implementation. Wiring it into the technique and its memory into the technique's stats move to RC3.2 (lead, 2026-11-09: RC3.1 was built beside RC2.1, before the technique existed on `radiance-cache`).
- **RC3.2 done-when.**
  - The hash grid wired into the technique behind `WorldCache`, its memory in the technique's stats (moved from RC3.1).
  - The passes listed in the row, ported with their source lines cited. Hits are lit by one sampled light and one shadow ray each, through the platform light grid (D4) and `giAnyHit`. Emissive hits and rays shorter than their cell take the bypass path exactly as `gi1.comp:2439-2456` and `:2538-2561` say.
  - The design doc's phase-2 staging note is marked retired.
  - GPU tests:
    - a lit plane seen through probe rays converges in its cells to the analytic outgoing radiance (albedo/π × E) within 2%;
    - `UpdateTiles`' sample count caps at 16 and its mips are 2×2 sums;
    - `ResolveCells` picks the coarser mip while a cell has fewer than 16 samples.
  - Cornell, two-room and Sponza render in the viewer with no errors, NaN or negative pixels. Bucket overflow is measured on Sponza and Bistro exterior by counting, not timing; the count goes in PROGRESS.
    *Amended (RC3.2 review, 2026-11-19):* the viewer's technique entry is on RC2.2's unfinished branch, so on `rc3-2` the NaN and negative check runs in the `radiance-cache-scenes` GPU page over the cache's own output (resolved probe-ray radiance, the probe atlas, every live cell), on the same three scenes; the viewer check comes with RC2.2's merge.
- **RC3.3 done-when.**
  - The passes run where Capsaicin runs them (gi1.cpp:2363-2390, 2506-2516): `PopulateMultibounceCells` between `PopulateScreenProbesCells` and `GenerateReservoirs`, and `UpdateMultibounceCells` after `UpdateTiles`, before `ResolveCells`. `HashGridCache.encodeFrame` gets those two slots (RC3.2 records five passes in a row).
  - The multi-bounce passes are ported with their source lines cited, including the 0.7 discard, the forced-diffuse second vertex and the one-frame lag of the indirect accumulator.
  - Two-room's closed room is visibly lit by the second bounce against RC3.2's image: its mean luminance rises, and the contact sheet is in PROGRESS.
  - The furnace room (white, closed, constant emitter or sky) moves toward the multi-bounce value against single bounce; the expected direction and size are stated before the run.
  - Cornell's indirect light against the Cycles reference is reported (single and multi-bounce), as information.
  *Amended (RC3.3, 2026-11-19):* the viewer's technique entry and the composite are on RC2.2's unmerged branch, so on `rc3-3` the two-room check is the mean probe-ray radiance after 60 frames, on and off (`radiance-cache-scenes`), and the furnace is a GPU page section (`radiance-cache-world-cache` 8-10, its expected values written in the page before the run). The two-room contact sheet and Cornell against Cycles come with RC2.2's merge; Cornell has no punctual light, so with register D21 its cache, multibounce included, holds no light yet.
- **RC3.4 done-when** (added by RC3.5's review, 2026-11-19; the rest is the row's).
  - Second-bounce misses stay 0, checked with a ceiling-only scene under a constant sky: the sky reaches a cache hit through its own light sample once the sky is in the light choice, so a second-bounce ray that misses must still add a sample of no light (register D37), or the sky would count twice.

### radiance-cache phase 4: temporal quality, faithful

Same rules (no timing; correctness gates; faithful port with sources cited; register rows from the next free number). Source map rows 4-6, 9, 10 (ray guiding), 23 (hysteresis), 24, 26, 28 (bent normal), 34, 35, and the SSGI technique (`src/core/src/render_techniques/ssgi/`).

| ID | Task | Agent | Size | Depends on | Status |
| --- | --- | --- | --- | --- | --- |
| RC4.1 | Temporal screen probes: `ReprojectScreenProbes` (plane and normal tests, the 3×3 taps, backup radiance), the off-screen LRU probe cache (`CountScreenProbes`, the scan, `ScatterScreenProbes`, `ReorderScreenProbes`), `PatchScreenProbes` (adaptive sampling of empty tiles), luminance-CDF ray guiding in `SampleScreenProbes`, biased hysteresis in `BlendScreenProbes` | implementer | M | RC3.2 | done (2026-11-19; Opus review: pass with notes, all applied; GI-1.2's steady-state darkening found and fixed (D41), per-cell running mean instead of the biased hysteresis (D46), register D40-D48; RTX 3060 and Intel UHD; follow-ups in RC4.4; see PROGRESS) |
| RC4.2 | `FilterScreenProbes` (the separable probe filter) and Capsaicin's SSGI technique (near-field GI and bent-normal occlusion) feeding `InterpolateScreenProbes` | implementer | M | RC4.1 | todo |
| RC4.3 | The image-space GI denoiser (`ReprojectGI`, `FilterGI` ×2, `gi_denoiser.hlsl`), built as a self-contained module with no dependency on `ddgi-core`'s internals, ready to move into `pmndrs/denoiser` | implementer | M | RC4.1 | todo |
| RC4.4 | Temporal follow-ups from RC4.1: (a) a GPU page that moves and switches a light and checks D46's per-region count reset (unit-tested only; probes not respawned keep their count about 4 frames); (b) the ray-guiding bias on sparse rays (-1.4% on the GPU: cells guided by their own estimate); (c) Sponza reads +6% with history vs without, a world-cache interaction (guided rays change which cells fill): find out which is right against the path tracer. Judge (b) and (c) against the path tracer, not against history-off. *Amended (lead, 2026-11-19):* (a) moved to the platform change-events tracker on `p6-idcells` (D46 wired to it, with an animated-sun / moving-box gate); (b) re-measured on `radiance-cache` after RC4.2's answer (sparse rays 0.9993 on `rc4-2`); (c) is the main job: a reusable accuracy page against the Cycles references (GI term and composite, mean and p95 error, per scene, Bistro too), used to say which of history on/off is right and what the world cache, the screen probes and multibounce each contribute; large findings listed, not fixed | implementer | S-M | RC4.1 | done (2026-11-19; the accuracy page `radiance-cache-accuracy` against the Cycles references is built and documented; (a) moved to P6.2b; (b) closes with RC4.2; findings in PROGRESS Next and RC3.6) |
| RC4.5 | Toggles to compare, and a viewer panel for them (Dennis, 2026-11-19: "does your speed improvement/history have a toggle? that way we can see"): each change we made to Capsaicin's temporal behaviour gets a setting that brings the old behaviour back, and the viewer shows them all live. At least: probe blend (D46 per-cell running mean, cap value / Capsaicin's hysteresis), probe history on/off, D41's empty-cell rule, interpolation jitter (exists), the denoiser on/off and its history clamp (RC4.3), the change-signal resets, the world-cache backend (P6.1) and cell shading mode (P6.2) once merged. Defaults stay as decided; latency vs ghosting is tuned later with these | implementer | S-M | RC4.2, RC4.3 | done (2026-11-20; a toggle restoring Capsaicin for every temporal change, a typed compare-toggle registry and the viewer's "Radiance cache · compare" panel; each Capsaicin value fails the gate written for it; world-cache and second-bounce toggles added after P6.1/P6.3) |
| RC4.6 | The viewer shows the shipped default: the `rc` entry stacks a DDGI volume under the radiance cache (`setFallback`) so `secondBounce: 'auto'` resolves to the DDGI second bounce, with a toggle to run the cache alone; DDGI at 256 probes a frame. Then one Cycles accuracy run of what the viewer shows. Until this lands, the viewer's `auto` equals GI-1.2 rays | implementer | S | P6.3, RC4.5 | done (2026-11-20; the viewer stacks the scene's DDGI volume under the radiance cache at 256 probes a frame; toggle "Run the cache alone"; Cycles: two-room and Cornell within 3.5%, Sponza +14.5/+20%) |

- **RC4.1 done-when.**
  - The passes listed in the row, ported with their source lines cited, at Capsaicin's constants.
  - GPU tests:
    - a still camera reprojects every probe to itself;
    - a camera pan marks the disoccluded tiles empty, and `PatchScreenProbes` moves probes into them;
    - a probe displaced by an occluder is cached and its cells are reused when the surface is uncovered (*amended, RC4.1 review 2026-11-19:* Capsaicin caches only a probe displaced in its tile by another surface, gi1.comp:762-765, 863-872; a probe that scrolls off the screen is dropped);
    - with history, `SampleScreenProbes` sends rays in proportion to its CDF (a chi-square check on a probe with one bright cell).
  - From RC2.1's review: a miss stores 65,504 (D19), which the blend's ×1e4 quantised sum overflows at 7 rays in one cell. Ray guiding can pile rays into sky cells, so check the cell sum can't wrap (what Capsaicin does with `FLT_MAX` there, and clamp or count as it does), with a GPU test of a probe whose CDF sends every ray to one sky cell.
  - Flicker on Cornell and Sponza with a still camera: T1.12's measure (per-pixel frame-to-frame change of the GI term), reported against DDGI's thresholds (mean ≤ 0.1%, p95 ≤ 0.5%), as information.
- **RC4.2 done-when.**
  - `FilterScreenProbes` (radius 3, plane, hemisphere and hit-angle tests, depth weight) and the SSGI technique ported, the latter as a platform or technique module that reads `FrameScreen`.
  - GPU tests:
    - the filter leaves a constant field unchanged and doesn't blur across a depth edge;
    - SSGI's bent normal points away from a nearby wall on a test corner.
  - The image on Cornell, two-room and Sponza in the viewer, with no errors and no NaN or negative pixels.
- **RC4.3 done-when.**
  - The denoiser ported with its source lines cited, in `packages/ddgi-core/src/denoise/` (or a sibling package): engine-free, its inputs named as `@pmndrs/denoiser`'s are (colour, depth, normal, velocity, history), and its own README section saying how it would move into `pmndrs/denoiser`.
  - GPU tests:
    - a still constant image passes through unchanged;
    - a disoccluded region gets the blur mask and is blurred, while a converged region is not;
    - history reprojection under a known camera move lands on the right pixels.
  - The radiance-cache technique uses it by default, as Capsaicin does.
  - **Phase 4's by-eye check is Dennis's** (the plan's done-when: steady by eye on Cornell and Sponza, no popping on a fly-through). The viewer is made ready for it and the check is listed in PROGRESS "Next"; it doesn't block the track.
  - Moving the module into `pmndrs/denoiser` (a PR there) needs Dennis's go-ahead: listed in PROGRESS "Next".

### radiance-cache phase 6: levers, experiments and dynamic scenes

Under DECISIONS 2026-11-19 (a multi-cache GI system built from GI-1.2): experiments are first-class alternatives behind a setting, judged by accuracy against analytic answers and the path tracer, temporal stability, memory and GPU time (timing recorded under the bench lock, not gated); the GI-1.2 baseline stays the default. Register rows in the experiments' own series (E1 on, kind E; the lead, 2026-11-19): P6.1 holds E1-E7, P6.2 starts at E8.

| ID | Task | Agent | Size | Depends on | Status |
| --- | --- | --- | --- | --- | --- |
| P6.1 | Experiment A: the pmndrs/pathtracer SHaRC-style hash cache as a second backend of the world-cache slot (`worldCache: 'gi1-hash-grid' \| 'sharc'`), with the GI-1.2 grid's analytic checks, an A/B table (accuracy, warm-up, stability, memory, GPU time on Sponza and Bistro exterior) and mutation checks | implementer | M | RC3.5, RC4.1 | done (2026-11-19; opt-in backend `worldCache: 'sharc'`, default stays GI-1.2; Opus review: merge as opt-in after the memory fix, all 7 fixes applied; the Cycles accuracy run waits on the references junction) |
| P6.2 | Experiment B, id Tech 8's "shade cells, not hits" on the GI-1.2 hash grid (register E8): `cellShading: 'per-hit' \| 'budgeted'` (default per-hit) and `cellShadingBudget`; hits only find their cells, a budget of cells a frame lit by priority (new or changed, converging, its turn, the rest), counting sort on the GPU; bypass and second bounce settled; GPU checks (lit plane, furnace, strips converge; queue; lag 5k/20k/80k vs per hit), stability, GPU ms on Sponza and Bistro exterior at 720p | implementer | M | RC4.1 | done (2026-11-20; opt-in `cellShading: 'budgeted'`; Opus review fixes applied; fixes 3-4 (zero-light rule, sky light mask in ShadeCells) follow RC3.4) |
| P6.2b | Experiment E9: E8's change marking on its own on the per-hit path (`cellChangeReset`, on by default since the review; `cellChangeResetHistory` 0 by the lead's call, 2026-11-20): cells hit near a recent `SceneChanges` box or a change with no bounds take the frame's samples whole for 16 frames; measured against per hit (lag after a sun and a point light double, Sponza stability, lit plane, furnace) | implementer | S | P6.2 | done (2026-11-20; `cellChangeReset` on by default, history 0; shared platform ChangeEventTracker (once per change event, re-armed every 8 frames); sun doubled settles in 2 frames vs 37) |
| P6.3 | Experiment E10, DDGI supplies the second bounce (id Tech 8's last-frame irradiance volumes; extension 1): `secondBounce: 'gi1-rays' \| 'ddgi-volume'` (default gi1-rays); with a DDGI stacked under the cache (`setFallback`), each first-bounce entry reads DDGI's irradiance at its hit instead of tracing a second-bounce ray (`SampleBounceSourceCells`); GPU checks (furnace, strips, bypass, lag), stability, GPU ms on Sponza and Bistro exterior, DDGI's own cost separately | implementer | M | RC3.5 | done (2026-11-19; `secondBounce: 'auto'` is the default: the fallback (DDGI) supplies the second bounce when one is stacked, GI-1.2 rays otherwise; Opus review, all 6 fixes applied; closed rooms within about 5% of Cycles; the accuracy rerun after fixes waits on the references junction; RC3.4 merge must fill the sky-bit TODO) |

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

### Follow-ups after the M3-M5 sign-off (2026-11-19)

- **`giRadiance`, DDGI's rough-only fallback** (Dennis, 2026-11-19: "yeah it can wait. just note it somewhere as a follow up"). Moved out of v1 (DECISIONS 2026-11-18). Comes with the three adapter's blend with environment specular, since the fallback is only right at roughness 1. Goes with the reflections work after v1.
- **Rename the workspace from `ddgi` to `gi`** (folders and private package names: `packages/ddgi`, `packages/ddgi-core`, `@pmndrs/ddgi-core`, ...; DECISIONS 2026-11-10). Waits on the radiance-cache branch merging, since it works inside those folders. Doesn't block a release: the published name is already `@pmndrs/gi`.
- **The old repo** `DennisSmolek/webgpu-ddgi`: Dennis picked "archive" (2026-11-18). A README pointing to pmndrs/gi first, if he wants one, then he archives it.

### From the Mac move and the coverage pass (2026-11-10)

- **The door and cabinet against Cycles renders of their end states** (from T3.9): CPU or Metal Cycles through `tools/reference/cycles/run.ts`, in parts under 15 minutes. Needs the references copied from the NAS.
- **M3's checks on the Intel UHD** (from T3.9) and the M4 accuracy runs left in T4.5 (Bistro exterior and interior, courtyard-tree, alpha-checker, sponza-full, FLIP): back on Windows, or once the references are on the Mac.
- **`low` doesn't lower `maxProbesPerFrame`** (4,096 probes a warm-up frame), which may matter on the Intel UHD and the iPad; check when `low` first runs there (T5.1). Chrome on the Mac reports no chip name, so `recommendPreset` gives every Mac `high`, base M chips included.
- **Small leftover from T5.4** (the rest done in T5.8): the hello example draws cut-outs solid and glass opaque (GI sees both right; only its picture is simple).
- **GI behind glass in the composite** (Dennis's pick, "later", 2026-11-18): three's glass pixels carry no GI albedo, so the composite shows no GI through glass; the per-fragment path is fine. When a host needs the composite with glass.
- **Timing on the target GPUs** (Dennis's pick, "later", 2026-11-18: "defer timing runs for another time"): the RTX 3060 and the M5 Pro back to the M1/M2 time targets, after the tuning pass.
- **Re-run `trace-oracle`, `trace-materials` and `trace-bench-quick` on the RTX and the Intel** (now TU.2) after the Metal shared-edge fix (2026-11-10, `giEdgeFunction` in `platform/trace/wgsl.ts`: corners ordered so FMA fusing in Dawn's relaxed Metal math can't open shared edges; 2,696 and 1,809 slips to 0 on the M5 Pro). It should give the same bits where nothing fuses; the cost is a few compares and selects per edge, not measured clean.
- **From T5.6 and T5.7 (2026-11-10):** `04-moving-objects` showed 34.6 ms of GI GPU time once (2.0 ms the next run), not looked into; on two-room, starting from a saved bake (0.5-2.2 s) was no faster than warming up (0.6-2.6 s), so the bake's start-up win is for big scenes and wasn't shown on one; upgrade vitest to 4.1.12+ and typescript-eslint to clear GitHub's dev-only security alerts (none ship in the package); `test:release` not yet run on Windows.
- ~~**Asset hashes are Windows-only**~~ done 2026-11-18 (`assets-mac`: per-platform records, Blender*.app search, bsdtar unpack; `--verify-only` passes all 10 on the M5 Pro). Was: on the Mac every Blender conversion and the generated wind tree come out with other hashes. The `.gltf` files differ by exactly their line count (CRLF on Windows), and Bistro's `.glb`, the courtyard tree's normals and the wind tree by small binary amounts (not yet looked at). `pnpm assets:fetch` reports them failed, though they load. Normalise line endings in the convert step and record per-platform hashes, or check the GLB's views as `--record-glb-views` already does.

### From M2, moved here when M3 became a prototype (2026-11-09)

Polish and follow-ups, picked up when a working feature needs them. Earlier IDs in brackets.
- DDGI memory: trim buffers grown by doubling once a scene settles (about 237 MB unused on Bistro, T2.17) (T3.13).
- Bistro's calibration: Budget v2 lands at 14-116 probes, and `ddgi/trace` has a 1-1.5 ms fixed cost at 16 probes; the budget page's check breaks when the frame rate is learned (T3.14).
- The change cost model counts `gi/direct`'s lit hits, to win back the frame-15 margin on Sponza's sun turn, with the sun layer on and off, on both GPUs (T2.18, T2.8, the Intel miss) (T3.15).
- Scene-copy follow-ups from T2.13's review: H-PLOC reading the packing's staging buffers, GPU tests of the packing fallbacks, the main-thread bar (T3.16). The texture-average items are dropped: M4's textured hits remove the averages (m4-materials.md).
- Native parity: native's Sponza GI about 8× Cycles' (the sun's units first), native's 1 m falloff clamp, a native sweep on our grid for Sponza (T3.17).
- `userData.giRange` in the three adapter (a host-facing API; designed by the main session first) (T3.18).
- Test flakes: unit tests timing out at 5 s under CPU load; the Intel hanging on the 4th page of a batch (T3.19).
- Nested finer volumes around thin cloth (Sponza's curtains; carried from M1).
- Re-timing M2's clock-sensitive gates if the clock comes back (the light grid with its run-boundary check, Sponza's lamp step, the cost after a change).
- Whether the tinybvh swap is still needed (GPU trees trace at 0.981× tinybvh's on Bistro, T2.10).
- Area lights on the platform (Dennis: "lets not abandon them").
