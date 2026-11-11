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

## M3: moving geometry (draft breakdown, for Dennis)

Status: **draft, from T2.11 (2026-11-09).** Starts when Dennis signs off M2 (PR `m2` → `main`). The design is [docs/design/m3-moving.md](docs/design/m3-moving.md): its T3.0–T3.11 are kept as written, less the rigid parts T2.M1 already built (the transform feed, geometry marks and the mover mask for rigid objects, the adapter's transform tracking). T3.12–T3.18 are M2's follow-ups (the T2.11 lines under "M2 done-when changes", and the [M2 note](docs/milestones/M2.md)). The way of working stays M2's: features over timing, one VALID run per gate, an Opus review before done.

| ID | Task | Agent | Size | Depends on | Status |
| --- | --- | --- | --- | --- | --- |
| T3.0 | Spike: deformed positions from three (storage `ExternalTexture`); the `updateMatrixWorld` override's cost and safety on Bistro interior; refit decay on a swaying tree | implementer | S | M2 done, T3.11 (the tree) | todo |
| T3.1 | Change feed, the parts T2.M1 left: `receivers`, `deformBounds`, `setDeformedPositions` (done-when: m3-moving.md's, less the items T2.M1's done-when already met: `setTransforms` cost, `moved`, the recompile counter) | implementer | S | none | todo |
| T3.2 | Runtime mesh trees: GPU builds for added and updated meshes, tinybvh refinement | implementer | M | T2.12 | todo |
| T3.3 | Deforming meshes: `GpuMeshRefit` wired, scene-copy positions, geometric normals, movement word; H-PLOC rebuild when the refit drifts | implementer | M | T3.1, T3.2 | todo |
| T3.4 | Probes near moved geometry, the parts T2.M1 left: woken air probes (M1's "probes in empty space" carry-over), deforming movers; the far-set guess scaled to the mover's size (DECISIONS, T2.M1 item 8). Done-when: m3-moving.md's, less T2.M1's items 3-5 and 8 (the cabinet, the door, the 2-update warm-up, the far-set record), which are met for rigid movers | implementer | M | T3.1 | todo |
| T3.5 | Several volumes: shared atlas, volume table, priority blending; nested finer volumes around thin cloth (carried from M1; their done-when, Sponza's curtain pixels against HD Cycles, is set when the main session designs them) | implementer | M | none | todo |
| T3.6 | Camera-following volume: wrapping slots, `ddgi/scroll`, window rule, fade-in | implementer | M | T3.5 | todo |
| T3.7 | three adapter change tracking v2 (deforming meshes, the `updateMatrixWorld` override) | implementer | M | T3.0, T3.1 | todo |
| T3.8 | Viewer scenarios and their manifest motions (furniture, door, cabinets, wind tree, fly-through) | implementer | M | T3.3, T3.4, T3.6, T3.7, T3.11 | todo |
| T3.9 | M3 measurements; also from M2: a Bistro response page, the whole frame on the library's default composite with the update in the same frame, the Apple M5 Pro if set up, and the clock-sensitive M2 gates if Dennis has fixed the clock (the light grid, with its run-boundary check; Sponza's lamp step; the cost after a change) | gpu-tester | M | T3.0-T3.8, T3.12-T3.18 | todo |
| T3.10 | M3 note and M4/M5 breakdown | main session | S | T3.9 | todo |
| T3.11 | The wind tree's asset: find or generate, convert, record licence | wrangler | S | none | todo |
| T3.12 | Compressed textures through GPUTex in the viewer's and test pages' loader (upstream gputex 0.14.0 first; HomeFig's method only after asking Dennis; KTX2 the fallback). Done when Bistro exterior fits the 6 GB RTX without T2.17's cap and looks the same by eye (DECISIONS 2026-11-09) | implementer | M | none | todo |
| T3.13 | DDGI memory: trim buffers grown by doubling once a scene settles (about 237 MB unused on Bistro, T2.17) | implementer | S | none | todo |
| T3.14 | Bistro's calibration: why Budget v2 lands at 14-116 probes and why `ddgi/trace` has a 1-1.5 ms fixed cost at 16 probes (T2.10, T2.17) | implementer | S | T3.12 | todo |
| T3.15 | The change cost model counts `gi/direct`'s lit hits (T2.18's first frames, p95 1.05× still); judged against T2.4c's frame-15 bar on Sponza's sun turn with the sun layer on and off, on the RTX and the Intel UHD (T2.18 used the margin: 13 → 15 on the RTX, 16 and 18 on the Intel; T2.8's layer-on turn at 16), starting with a run with `keyProbes` off | implementer | S | none | todo |
| T3.16 | Scene-copy follow-ups from T2.13's review: texture averages from three's smallest mip read back asynchronously (a ruling: a load-time readback and a placeholder albedo), H-PLOC reading from the packing's staging buffers, GPU tests of the packing fallbacks; the main-thread bar (Sponza 816 ms against 150) | implementer | M | none | todo |
| T3.17 | Native parity: why native's Sponza GI is about 8× Cycles' (its direct light is 2.9×; the sun's units in our harness settings first), native's 1 m falloff clamp, and a native sweep on our probe grid for Sponza | native-runner | S | none | todo |
| T3.18 | `userData.giRange` in the three adapter, so a host can cull a light in three without shortening its GI range (T2.8's candle reach). Host-facing API: the main session designs it first | main session, then implementer | S | none | todo |
| T3.19 | Test flakes: `tools/assets/fetch.test.ts` and two other unit tests time out at 5 s when the CPU is busy (carried from M1 and seen again in M2); the Intel UHD hangs on the 4th page of 4-page GPU batches (run batches of 3 or fewer) | implementer | S | none | todo |

## radiance-cache track (second technique)

Status: **phase 1 started 2026-11-09** on branch `radiance-cache` (cut from `m2` at 5de856b), alongside M2's close-out. The design and the later phases are in [docs/design/radiance-cache.md](docs/design/radiance-cache.md), and every difference from Capsaicin is in [docs/design/radiance-cache-divergences.md](docs/design/radiance-cache-divergences.md); Dennis's decisions are in DECISIONS (2026-11-09, "radiance-cache"). The technique is a faithful port of AMD's GI-1.0/1.2 (Capsaicin v1.3) first; performance ideas are levers beside it.

Rules for every row:
- **Separable.** Each row is built on its own branch (`rc1-<n>`, from `radiance-cache`) and merged into `radiance-cache` after review, so each platform change can also merge to the main line alone.
- **Subgroups first** with one declared fallback and a test of it (DECISIONS 2026-11-04).
- **Hardware:** the RTX 3060 is the target and GPU timing is taken GPU-loaded under the bench lock (`tools/reference/bench-lock.ts`), VALID runs only; the Intel UHD is correctness only. Never time anything while a native Capsaicin run (`scene_viewer.exe`) is going.
- **Every divergence from Capsaicin** (a three backend step, a missing feature) gets a row in [docs/design/radiance-cache-divergences.md](docs/design/radiance-cache-divergences.md), numbered from the next free ID, with its cause, evidence, impact, effort to match, recommendation and way back.
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

The port reads AMD's source directly: Capsaicin v1.3 (`914b915`) at `src/core/src/render_techniques/gi1/` in a local clone (on the Mac: `/Users/dex/Developer/capsaicin`; on the Windows laptop: `C:/Users/denni/Documents/dev/ddgi/.claude/worktrees/amd-gi-cache-research-e46e2e/native/references/capsaicin/`; the research notes are on the Mac at `/Users/dex/Developer/ddgi-research/research/gi-1.0/`) (`gi1.comp`, `gi1.h`, `gi1.cpp`, `screen_probes.hlsl`, `gi1.hlsl`) and the shared `src/core/src/math/` files, mapped pass by pass in the research folder's `source-map.md` §3 (rows 2-11, 23, 25, 27, 28 for this phase). Ported WGSL keeps AMD's MIT notice.

| ID | Task | Agent | Size | Depends on | Status |
| --- | --- | --- | --- | --- | --- |
| RC2.1 | Technique skeleton and screen probes: spawn (16×16 tiles, Halton jitter), compact, sample (64 rays, one per hemi-octahedral cell), trace, light hits directly, blend into the probe atlas; debug views of the atlas and the mask | implementer | M | RC1.1, RC1.3 | done (2026-11-10; Opus review passed with notes; both GPUs; the sky check amended to Capsaicin backup rule; viewer entry moved to RC2.2; D16-D21). M5 follow-up: review (implementer, 2026-11-10; branch `rc2-1-m5`, D34) |
| RC2.2 | First image: the probe mask mips, SH projection, interpolation per pixel, the shading module by `screenUV`, the three composite and the viewer's technique entry; the Lambert check | implementer | M | RC2.1 | done (2026-11-10, Mac lead; Opus review failed on missing records and the SH basis, fixes in, re-review passed with notes, lead wording fixes; integrated with RC3.2's world cache; the six RC pages pass on the M5; Windows GPUs not run; D25-D29, D38. Open: Sponza's probe rays read 1.086 of the one-bounce reference with the cache on, cause not checked) |

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
  - **Lambert check (GPU test):** an open plane under a constant sky of radiance L reads E = πL within 1% per pixel (mean and p95) at the interpolated output. *Amended (lead, 2026-11-10, RC2.2 review):* within 2.5%, since the projection takes Capsaicin's cell-centre SH basis, whose own error is 2.1% (2.04% measured on the M5; DECISIONS 2026-11-10).
  - A single-bounce image on Cornell, two-room and Sponza in the viewer, with no errors and no NaN or negative pixels (checked by readback). A contact sheet in PROGRESS beside DDGI's image of the same view.
  - Debug views of the SH (irradiance from the probe's SH along the normal) join the atlas and mask views, and all three reach the viewer's debug menu (moved from RC2.1): `giDebugColor` gets the pixel's `screenUV` as `giIrradiance` has it.
  - From RC2.1's review: a direction-varying sky (or an occluder) in a GPU test, so an atlas transposed in its tile or a hemi-octahedral map mirrored the same way on the CPU and GPU fails; fixed Capsaicin reference values in the CPU map tests; spawn checked at several frames; the uniform block packed by named offsets, not bare word indices (`radiance-cache-technique.ts:487-493`); a unit test that `encodeScreen` gets the frame's camera.
  - `FrameCamera` gets an optional jitter (NDC offset of this frame's projection) so a host that jitters for the upscaler gets positions rebuilt at the depth's own jitter (RC2.1: half a pixel off otherwise). The three adapter calls `encodeScreen` after `encode`.

### radiance-cache phase 3: the world cache, faithful

Same rules as phase 2 (no timing; correctness gates; faithful port with Capsaicin's source cited; register rows from the next free number). The interface between the screen cache and the world cache is in the design doc ("The world-cache interface"). Source map rows 3, 11, 12, 15, 16, 19-22 and §4.2.

| ID | Task | Agent | Size | Depends on | Status |
| --- | --- | --- | --- | --- | --- |
| RC3.1 | The hash grid: one shared WGSL function for the cell descriptor and both hashes; CAS insert into 16-slot buckets, find, decay stamps and `PurgeTiles`; buffers packed at Capsaicin's default capacity; the `WorldCache` interface with the hash grid behind it | implementer | M | RC2.1 | done (2026-11-10; Opus review failed on the ray offset, fixed; both GPUs; 1,024 MiB at default capacity; wiring moved to RC3.2; D22-D24) |
| RC3.2 | Population and resolve: probe-ray hits insert cells (`PopulateScreenProbes`' cache half), `GenerateReservoirs` (a light per hit through the platform light grid, D4, with the bypass rules), `PopulateCells` (shadow ray, quantised accumulation), `UpdateTiles` (temporal accumulation and the tile mips), `ResolveCells` (filtered lookup into the probe rays); phase 2's staging retired | implementer | M | RC3.1, RC2.2 | done (2026-11-10, Mac lead; Opus review failed on the binding budget, fixed, re-review passed with notes; the small-cache tile-list bug fixed with a test; M5 pages pass: hash-grid, world-cache, probes, overflow (0 refused on Sponza and Bistro exterior); RTX 3060 and Intel UHD not re-run since the packing; D30-D33, D35-D37) |
| RC3.3 | GI-1.2 multi-bounce: `PopulateMultibounceCells`, `GenerateMultibounceReservoirs`, `UpdateMultibounceCells` and the indirect value array | implementer | M | RC3.2 | done (2026-11-10, Mac lead; Opus review passed with notes, notes fixed; the 3.3× second-bounce bias confirmed as Capsaicin's own and kept (lever 12); D40 and D41 are Capsaicin bugs we fix; M5 pages pass; Windows GPUs not run; D39-D42) |

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
- **RC3.3 done-when.** *Proposed by the implementer (2026-11-10), modelled on RC3.2's gates; the draft's furnace and image lines are amended below, with the reason.*
  - The three passes ported with their source lines cited (`PopulateMultibounceCells` gi1.comp:1761-1946, `GenerateMultibounceReservoirs` :2601-2695, `UpdateMultibounceCells` :1948-1983), in Capsaicin's order (gi1.cpp:2363-2516), including the 0.7 discard and its 1 / (1 - 0.7) weight, the forced-diffuse second vertex (Lambert under D8) and the one-frame lag: `UpdateMultibounceCells` runs after `UpdateTiles`, so its samples are taken by the next frame's `UpdateTiles`. `gi1_use_multibounce` (default on) is honoured; off is Capsaicin's single-bounce path (the resolve list is the shadow-ray list).
  - Every kernel binds at most the platform's 8 storage buffers; the unit test that builds every hash-grid kernel at 8 and expects `GILimitError` at 7 covers the three new ones.
  - GPU tests on a sunlit corner (floor and wall, both at E = 8 cos 45°, so every surface a second-bounce ray can reach has the same direct radiance L = albedo / π × E):
    - the discard and each second-bounce ray's direction match a CPU port of Capsaicin's random numbers and cosine sampling; brdf and pdf are Lambert's;
    - `UpdateMultibounceCells`' indirect scratch equals the CPU's sum from the read-back cells, brdf and pdf; the next frame's indirect values are `UpdateTiles`' accumulation of it (the lag);
    - converged, every cell's indirect value reads albedo × L / (1 - 0.7) and every resolved probe ray L × (1 + albedo / (1 - 0.7)) within 2%: Capsaicin's estimator, stated before the run beside the physical value (see PROGRESS);
    - a probe ray that bypasses the cache reads its own light sample plus its own second-bounce sample (register D40), and one that doesn't its cell's filtered direct plus indirect value;
    - RC3.2's single-bounce checks still pass, run with `useMultibounce` off.
  - Two-room's dark room (room B, from a camera inside it, so only room B is in view; *amended* once RC2.2 merged mid-task: the interpolated irradiance E rather than the probe atlas): its mean luminance rises with multibounce against single bounce (the direction and "more than half again" stated before the run), returns when the option goes off again, and the three pictures are the contact sheet (the page's screenshot).
  - *Amended:* the furnace room can't test this technique's bounce: its light is emission, and Capsaicin's `GenerateReservoirs` sends an emissive front face's light to the probe ray and never into the cache (gi1.comp:2439-2452; emissive surfaces are not lights here, D21), so it reads the same with multibounce on or off. The corner takes its place. Cornell against the Cycles reference (information only) is not part of this row: it needs a multi-bounce path-traced irradiance at the probes, which no page has yet (`radiance-cache-three`'s reference is one bounce); it moves to phase 5's comparison.

### radiance-cache phase 4: temporal quality, faithful

Same rules (no timing; correctness gates; faithful port with sources cited; register rows from the next free number). Source map rows 4-6, 9, 10 (ray guiding), 23 (hysteresis), 24, 26, 28 (bent normal), 34, 35, and the SSGI technique (`src/core/src/render_techniques/ssgi/`).

| ID | Task | Agent | Size | Depends on | Status |
| --- | --- | --- | --- | --- | --- |
| RC4.1 | Temporal screen probes: `ReprojectScreenProbes` (plane and normal tests, the 3×3 taps, backup radiance), the off-screen LRU probe cache (`CountScreenProbes`, the scan, `ScatterScreenProbes`, `ReorderScreenProbes`), `PatchScreenProbes` (adaptive sampling of empty tiles), luminance-CDF ray guiding in `SampleScreenProbes`, biased hysteresis in `BlendScreenProbes` | implementer | M | RC3.2 | done (2026-11-10, Mac lead; Opus review passed with notes, notes fixed: the history drift is Capsaicin's own (backup rule, hysteresis, CDF feedback, 3 × 3 taps; model 0.690 against the GPU's 0.6896; native Capsaicin to confirm in phase 5, lever 13); storage-texture budget test added; nine RC pages pass on the M5; Windows GPUs not run; D45-D54) |
| RC4.2 | `FilterScreenProbes` (the separable probe filter) and Capsaicin's SSGI technique (near-field GI and bent-normal occlusion) feeding `InterpolateScreenProbes` | implementer | M | RC4.1 | done (2026-11-10, Mac lead; Opus review passed with notes, notes fixed: the darkening is mostly SSGI's single slice (lever 16), not AMD item (d); the M5's rgba16float stores round toward zero (D59; hit distance settles about 30 m low at 40 m: a D31-style correction is the follow-up); previousLighting checked a frame late; test hook removed; integrated with RC4.3; flicker after the denoiser with SSGI, converged: Cornell 0.74% / 3.16%, Sponza 0.90% / 2.91%; twelve RC pages pass on the M5; Windows GPUs not run; D55-D59) |
| RC4.3 | The image-space GI denoiser (`ReprojectGI`, `FilterGI` ×2, `gi_denoiser.hlsl`), built as a self-contained module with no dependency on `ddgi-core`'s internals, ready to move into `pmndrs/denoiser` | implementer | M | RC4.1 | done (2026-11-10, Mac lead; Opus review passed with notes, notes fixed: Cornell's 4% was warm-up (0.9993 converged), the loosened gate is Capsaicin's 2-half-step cycle, the landing tolerance restored to 1%, NaN-safe sign (D75), Capsaicin's formats with texture-formats-tier1; converged flicker Cornell 0.36% / 1.32%, Sponza 0.27% / 0.88%; package name `@pmndrs/denoiser-gi1` (private) is Dennis's call; M5 only; D65-D75) |

- **RC4.1 done-when.**
  - The passes listed in the row, ported with their source lines cited, at Capsaicin's constants. History is read through the previous frame's camera (kept by the technique) and `FrameScreen.velocity`, which the technique now requires (Capsaicin's `view_proj_inv_prev` and velocity buffer). Phase 2's per-frame `ClearProbeMask` is retired. Every kernel fits 8 storage buffers per stage, checked by a unit test through the kernel builder at 8 and 7 (`radiance-cache.test.ts`); the LRU and list scans are the platform's (RC1.3).
  - *Refined (implementer, 2026-11-10, RC4.1), built as `radiance-cache-temporal`:* the still check also asks that no reprojected probe lose its seed, SH (bit for bit) or atlas cells (within the ×1e4 quantization), and that no tile is empty once every tile has a probe; the pan's empty tiles are checked against a CPU twin of the reprojection's search, and every listed probe is its spawn tile's seed or a patched seed in an empty tile; the probe cache's per-tile lists and next LRU order against CPU twins of the count, scan, scatter and reorder, exactly. "A probe that leaves the screen and comes back" became "a probe **evicted** at the pillar's edges, off screen while the camera looks away, found again when it turns back": Capsaicin's cache holds only probes their tile replaced (gi1.comp:897-912), not every probe that leaves the screen. Existing pages that check a no-history frame (`-probes`, `-image`, the three page's at-probe numbers) reset the screen probes first (`resetScreenProbes`), since with history the rays are guided and no longer one per cell.
  - GPU tests:
    - a still camera reprojects every probe to itself;
    - a camera pan marks the disoccluded tiles empty, and `PatchScreenProbes` moves probes into them;
    - a probe that leaves the screen and comes back is found in the LRU cache;
    - with history, `SampleScreenProbes` sends rays in proportion to its CDF (a chi-square check on a probe with one bright cell).
  - From RC2.1's review: a miss stores 65,504 (D19), which the blend's ×1e4 quantised sum overflows at 7 rays in one cell. Ray guiding can pile rays into sky cells, so check the cell sum can't wrap (what Capsaicin does with `FLT_MAX` there, and clamp or count as it does), with a GPU test of a probe whose CDF sends every ray to one sky cell.
  - Flicker on Cornell and Sponza with a still camera: T1.12's measure (per-pixel frame-to-frame change of the GI term), reported against DDGI's thresholds (mean ≤ 0.1%, p95 ≤ 0.5%), as information.
- **RC4.2 done-when.**
  - `FilterScreenProbes` (radius 3, plane, hemisphere and hit-angle tests, depth weight) and the SSGI technique ported, the latter as a platform or technique module that reads `FrameScreen`.
  - GPU tests:
    - the filter leaves a constant field unchanged and doesn't blur across a depth edge;
    - SSGI's bent normal points away from a nearby wall on a test corner.
  - The image on Cornell, two-room and Sponza in the viewer, with no errors and no NaN or negative pixels.
  - *Refined (implementer, 2026-11-10, RC4.2), modelled on RC4.1's gates, built as `radiance-cache-filter` and `radiance-cache-ssgi`:*
    - Capsaicin's defaults (radius 3, two passes; 1 slice, 2 steps, 0.2 m, 5% falloff), its order (SSGI before GI-1; the filter after `FilterProbeMask`, ping-ponging through the previous atlas), its constants, source lines cited, AMD's notice on the WGSL. SSGI's lit-colour input is a `FrameScreen` channel (`previousLighting`) asked for through `TechniqueRequirements.screen`, filled by the three adapter. The kernels fit 8 storage buffers and 4 storage textures per stage (the budget test names them).
    - **History is the filtered atlas** (RC4.1's review): after a frame the previous atlas equals the filtered atlas bit for bit, and a still camera's reprojected probes keep its cells.
    - **The filter against a CPU twin of both passes** on a captured atlas (cells with a test at its threshold counted, not compared), within three half steps; a constant atlas stays constant (radiance within a half step, hit distance within two: the filter's `min` and the M5's rounding toward zero); on the pillar, the floor's probes beside the box keep their value wherever the twin says so (the depth edge).
    - **SSGI against a CPU twin** on a corner (floor beside a lit wall): pixels whose decisions sit at a threshold or whose twin moves when its positions are rounded to f32 are counted, not compared; the rest match (occlusion and bent normal within 2e-3, near-field light within 2e-3 relative + 1e-4) but for at most 0.1% (the GPU's other f32 steps, amplified by Capsaicin's 1e-5 clamp); image means agree (occlusion and bent z within 0.1%, the wall's light on the floor within 1%). Near the wall the bent normal leans away (mean z > 0.1), occlusion is below 0.9 and the wall's light arrives; 1 m away none does. *Amended:* "no negative pixels" means none below −1e-5: SSGI's arc integrals of an unmoved horizon come out a few f32 steps under 0 (DECISIONS 2026-11-10, RC4.2).
    - The pages the filter and SSGI change are updated, each with its reason (DECISIONS 2026-11-10, RC4.2): the Lambert check through SSGI's cone, the probes' sky cells through the filter's twin, the temporal sky cell on the filtered atlas, the dark room's frames.
- **RC4.3 done-when.**
  - The denoiser ported with its source lines cited, in `packages/ddgi-core/src/denoise/` (or a sibling package): engine-free, its inputs named as `@pmndrs/denoiser`'s are (colour, depth, normal, velocity, history), and its own README section saying how it would move into `pmndrs/denoiser`.
  - GPU tests:
    - a still constant image passes through unchanged;
    - a disoccluded region gets the blur mask and is blurred, while a converged region is not;
    - history reprojection under a known camera move lands on the right pixels.
  - The radiance-cache technique uses it by default, as Capsaicin does.
  - *Refined (implementer, 2026-11-10, RC4.3), built as `packages/denoiser-gi1` (`@pmndrs/denoiser-gi1`, lint-held to its own folder) and the GPU page `radiance-cache-denoiser`, modelled on RC4.1's gates:* the module's kernels fit WebGPU's default limits (no storage buffers, at most 3 storage textures; unit tests in the package and in `radiance-cache.test.ts`) and need no optional feature. On a CPU-ray-cast scene at 480 × 270 against the package's CPU reference (`src/reference.ts`), the GPU's previous state read back each time: (1) a still constant input stays constant: sky 0, the reference within 2 half steps at frames 1, 2, 8, 40, 100, 160, settled over the last 10 of 160 frames (*amended* from one half step to two after the first run, with no drift of the mean: DECISIONS 2026-11-10), the f64 reference alone gives the constant back within 1e-6, with round-to-nearest halves within 1%, and (after the review) the GPU's settled mean within 0.5% of the reference run free with the GPU's rounding, each worst under 2%; (2) the blur mask, colour delta, reprojected history, both filter passes and the irradiance on captured noisy frames (1, 3, 12) within the snorm step or 2 half steps, pixels at a threshold counted; (3) the same for a known camera move, plus: the history lands on the right pixels (a 2 m checker read back within 1% with the input's hint 0), disoccluded pixels get the full blur mask and their noise falls by more than half, converged pixels get mask 0; (4) a NaN and an infinity in the input are gone from the output, the history and the colour delta within 3 frames (D71, D75). After the review, checks 1-4 run on Capsaicin's `r8snorm` / `r16float` (with `texture-formats-tier1`) and on the declared fallback. In the radiance cache: no NaN, infinity or negative value in the interpolated or denoised irradiance on Cornell and Sponza over 64 still frames, and flicker reported beside RC4.1's (stated before the run: below the interpolated E's on both), in RC4.1's window and, after the review, on converged history (from frame 300). The nine existing RC pages re-run.
  - **Phase 4's by-eye check is Dennis's** (the plan's done-when: steady by eye on Cornell and Sponza, no popping on a fly-through). The viewer is made ready for it and the check is listed in PROGRESS "Next"; it doesn't block the track.
  - Moving the module into `pmndrs/denoiser` (a PR there) needs Dennis's go-ahead: listed in PROGRESS "Next".

### radiance-cache phase 5: timing and the comparison with native

Started 2026-11-11 (DECISIONS 2026-11-11: timing resumes). The levers and the order are in [docs/design/radiance-cache-speed.md](docs/design/radiance-cache-speed.md). M5 numbers are hints and say so; the RTX 3060 under the bench lock is the record.

| ID | Task | Agent | Size | Depends on | Status |
| --- | --- | --- | --- | --- | --- |
| RC5.1 | Screen-stage GPU timing (lever 17): the platform resolves its timestamps after `encodeScreen`, so radiance-cache passes stop reading 0 ms; a `radiance-cache-timing` page shaped like `ddgi-timing`; a first M5 per-pass table | implementer | S | none | todo |
| RC5.2 | Windows re-run: the twelve `radiance-cache-*` pages plus `hit-shading-lights` and `ddgi-furnace` on the RTX 3060 and the Intel UHD; nothing has run there since phase 2 | gpu-tester | S | none (Windows laptop, 2026-11-12) | todo |
| RC5.3 | The phase 5 table: VALID per-pass timing on the RTX 3060 beside native Capsaicin's groups (Sponza, Bistro interior and exterior, two-room); a cause and an attention-list row for each pass over native; accuracy against the path tracer per scene | gpu-tester, native-runner | M | RC5.1, RC5.2 | todo |
| RC5.4 | Slow surface rays (lever 20): find why probe rays leaving surfaces trace at 17-35 Mrays/s while rays from points in the air reach about 87 (RC1.5); cause and a proposed fix, no tracer change without the lead | implementer | M | none | todo |

- **RC5.1 done-when.**
  - A test that fails before the change: a pass recorded in `encodeScreen` reads more than 0 ms through the platform's timing.
  - DDGI's timings are unchanged: `ddgi-timing` still passes and its pass list is the same.
  - `radiance-cache-timing` reports per-pass medians over warmed frames on Cornell, Sponza and Bistro interior and exterior at 1280×720, with Capsaicin's section names, grouped as native Capsaicin's CSV groups (`ddgi-research/research/gi-1.0/native-capsaicin.md`), and a switch for `useMultibounce`.
  - The M5 table goes in `bench/results/radiance-cache-m5-passes.csv`, labelled M5. The page runs unchanged on Windows (`--gpu nvidia`).
  - No image changes: the RC pages still pass on the M5.
- **RC5.2 done-when.** Every page's result on both GPUs in PROGRESS. Each failure gets a cause or a follow-up row; nothing is loosened to pass without a DECISIONS entry.
- **RC5.3 done-when.** Phase 5 as the design doc states it: the table, the attention list and the image comparison are in, and Dennis has seen them.
- **RC5.4 done-when.** The cause is measured, not guessed: the rate split by what differs between the two ray sets (origin offset, direction spread, first-node depth, ray length). The fix is proposed with its expected gain. The BVH lane is told if the cause is in the tracer.

### radiance-cache: divergence fixes

From the shortlist in [docs/design/radiance-cache-divergences.md](docs/design/radiance-cache-divergences.md). Each row moves the port closer to Capsaicin, so the fix replaces our path; its register row is updated to say it's closed, or what remains.

| ID | Task | Agent | Size | Depends on | Status |
| --- | --- | --- | --- | --- | --- |
| RC6.1 | D33: sky light at cache hits, as Capsaicin's light sampling gives it, through a platform helper that samples the baked sky; radiance cache only (DDGI's `gi/direct` is unchanged: DDGI gets the sky through probe rays that miss) | implementer | M | none | todo |
| RC6.2 | Three small fixes, one commit each: D59 (the M5's half-float stores round down), D42 (free the 535 MB when multi-bounce is off), D39 (Capsaicin's `offsetPosition` for the second-bounce ray) | implementer | S | none | todo |

- **RC6.1 done-when.**
  - The port follows Capsaicin's environment-light sampling at cache hits, cited line by line: how the sky is picked against the other lights, its pdf, and its shadow ray.
  - A GPU check: under a constant sky with no other lights, a cache cell on an open floor reads E = πL within the 2.5% Lambert threshold (DECISIONS 2026-11-10). In a closed box with no lights, it reads 0.
  - Sponza and Bistro exterior before and after are recorded in PROGRESS.
  - Every RC page still passes on the M5, or the change in its numbers is explained.
  - D33's register and challenge entries are updated.
- **RC6.2 done-when.**
  - **D59:** `radiance-cache-filter` check 6's drift is gone on the M5, with 40 m staying within 1 m over 300 frames. The fix is checked to be harmless on a GPU that rounds to nearest, through the CPU twin.
  - **D42:** with `useMultibounce: false`, `stats` memory drops by the multi-bounce arrays, and turning it on prepares again. `radiance-cache-multibounce` passes.
  - **D39:** Capsaicin's `offsetPosition` is ported for the second-bounce ray. The count of second-bounce rays that change hit on Sponza and Cornell is recorded. `radiance-cache-multibounce` passes.
  - Each register entry is updated.

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
