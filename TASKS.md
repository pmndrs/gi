# Task board

Status values: `todo`, `active (agent, date)`, `review`, `done`, `blocked (why)`. One active task per agent. The main session moves tasks to `done` after review. How tasks work is in [PLAN.md](PLAN.md#what-done-means-for-a-task).

## Summary

| ID | Task | Agent | Size | Depends on | Status |
| --- | --- | --- | --- | --- | --- |
| T0.1 | Workspace | implementer | M | none | done |
| T0.2 | GPU runner | implementer | M | T0.1 | done |
| T0.3 | Timing and devices | implementer, then gpu-tester | M | T0.2 | done (Windows only; Apple devices deferred, see DECISIONS 2026-11-04) |
| T0.4 | Test scenes | wrangler (assets) + implementer (manifest, procedural scenes) | M | T0.1 | done (11 scenes; Sponza is Intel, download-only; Opus-review fixes in) |
| T0.5 | Viewer shell | implementer | M | T0.2, T0.3, T0.4 | active (Opus review failed it; fixes in progress, then Dennis tries it) |
| T0.6 | Native harness | native-runner | M | T0.4 | active (native-runner, 2026-11-04) |
| T0.7 | References and compare | implementer | M | T0.4 | done (Cycles references for Cornell, two-room, Sponza; tolerance in DECISIONS; Opus-review fixes in, Sponza re-rendered diffuse-only) |
| T0.8 | Spike: three.js sharing | implementer | S | T0.2 | done (yes to all; env-diffuse needs a workaround) |
| T0.9 | Spike: Babylon | implementer | S | T0.2 | done (fits the contract; Babylon owns its device) |
| T0.10 | Spike: BVH | main session designs, implementer builds, gpu-tester measures | M | T0.2, T0.4 | active (design: [docs/design/trace.md](docs/design/trace.md); part A done; part B benchmark being built) |
| T0.11 | Platform API draft | main session | M | none (revise after T0.8–T0.10) | active (draft in [docs/api/platform.md](docs/api/platform.md); waiting on spikes and Dennis's read) |
| T0.12 | M0 note and M1 breakdown | main session | S | all M0 | todo |

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

## Later

M1–M5 tasks are added at the end of the previous milestone. See [PLAN.md](PLAN.md#later-milestones-outline).
