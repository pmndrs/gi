# Implementation plan

How we build what [SPEC.md](SPEC.md) describes. The spec says *what* and *why*. This file says *in what order* and *how we know each piece is done*.

- [TASKS.md](TASKS.md) is the live board.
- [PROGRESS.md](PROGRESS.md) is the running log.
- [docs/NOTES.md](docs/NOTES.md) holds facts worth not relearning.
- [docs/DECISIONS.md](docs/DECISIONS.md) holds settled choices.

M0 is planned in detail below. Later milestones are outlined, and each one is broken into tasks in the last task of the milestone before it, once we know what that milestone taught us.

## How work flows

1. **The main session plans and decides.** It owns the spec, the API design, task breakdown, reviews and the milestone note. It runs on the strongest model.
2. **Subagents do the work, one task each.** Every task on the board names an agent type, and with it a model tier. Agent definitions are in [.claude/agents/](.claude/agents/).

   | Agent | Model | Used for |
   | --- | --- | --- |
   | `scout` | haiku | Read-only lookups: find code, read docs or upstream source, summarize. Never edits. |
   | `wrangler` | haiku | Mechanical jobs: download and convert assets, record licenses, rename, format, update tables. |
   | `implementer` | opus, high effort | Build one task's code and tests to its done-when. |
   | `gpu-tester` | sonnet | Run GPU checks and benchmarks in real browsers, record numbers, capture screenshots. |
   | `native-runner` | opus, medium effort | Native RTXGI harness work: CMake, Vulkan/D3D12 builds, PowerShell launchers, benchmark CSVs. |
   | `reviewer` | opus, high effort | Check a finished task against its done-when, the code rules and the spec. It reports back and doesn't fix. |

   Design-heavy work stays in the main session: platform API, tracing design, the DDGI algorithm, deciding what a spike's numbers mean.
3. **Files are the shared memory.** A subagent starts by reading:
   - `CLAUDE.md`;
   - its task row in `TASKS.md`;
   - the files that row links;
   - the last few `PROGRESS.md` entries for that task.

   It ends by writing back:
   - one `PROGRESS.md` entry;
   - its task row's status;
   - any new durable fact in `docs/NOTES.md`.

   Nothing important lives only in a conversation.
4. **The main session reviews before a task is marked done,** with a `reviewer` pass for anything non-trivial, and checks the result itself. Subagent reports are claims, not proof.

## What "done" means for a task

Every task row has four parts:

- **Done when:** 1–4 checks that are objectively true or false. Each one is a command that passes, a number under a stated limit, or something visible in the viewer at a named scene and camera. A check is never "a report exists".
- **Show:** what you can look at to judge it yourself: a viewer URL with a scene and camera, a screenshot path, or a table in `PROGRESS.md`.
- **Record:** the numbers to write down, and where. Numbers go in `PROGRESS.md` and, for benchmarks, one row per run in `bench/results/<benchmark>.csv` (date, commit, device, browser, scene, metric, value). There are no per-task JSON evidence bundles and no validators that check other reports.
- **Size:** S (under a few hours of agent work), M (a day), or L (split it). L tasks are split before they start.

A task is **done** when its checks pass, the reviewer found no blocking issue, and its `PROGRESS.md` entry says plainly what was verified and what wasn't. A milestone is **done** when its tasks are done, its milestone note is written and you've signed off in the viewer.

## M0: Clean start, tools, and the risky questions

The goal is a working repo, real baselines (web and native) on real scenes, and yes/no answers to the questions that could force an architecture change. No DDGI yet.

### Order

```
T0.1 workspace ─┬─ T0.2 GPU runner ─┬─ T0.3 timing + devices
                │                   ├─ T0.8 three.js sharing spike
                │                   ├─ T0.9 Babylon spike
                │                   └─ T0.10 BVH spike ──┐
                ├─ T0.4 scenes ─────┬─ T0.5 viewer shell (needs T0.2, T0.3)
                │                   ├─ T0.6 native harness
                │                   └─ T0.7 references + compare
                └─ T0.11 platform API draft (paper) ─────┴─ revised after T0.8–T0.10
T0.12 M0 note + M1 breakdown (after all)
```

Parallel lanes once T0.1 lands:

- spikes: T0.8, T0.9, T0.10;
- scenes and references: T0.4, T0.6, T0.7;
- viewer: T0.3, T0.5;
- API: T0.11.

Two writers never touch the same files.

### Tasks

Full rows, with done-when checks, are on the board. In short:

- **T0.1 Workspace.**
  - pnpm workspace mirroring `@pmndrs/upscaler`: `packages/ddgi` (umbrella) over `ddgi-core`, `ddgi-three`, later `ddgi-babylon`/`ddgi-pixi`; plus `apps/viewer`, `examples/webgpu/hello` and `tools/`.
  - TypeScript strict, ESLint with the core no-engine-imports rule, vitest, vite, rollup type bundling.
- **T0.2 GPU runner.** Headless Chrome over the DevTools protocol (the upscaler's dependency-free approach). Picks the NVIDIA or Intel adapter, collects WGSL errors and console output, takes screenshots, and fails loudly when WebGPU is missing.
- **T0.3 Timing and devices.** A GPU timestamp helper (`gpuTiming`) and a stats overlay, run on every test device and browser to answer: do timestamps work, and what do they cost?
- **T0.4 Test scenes.** The manifest format and a download script, covering:
  - Cornell, two-rooms, furnace, thin wall, closed door, cabinet row (built in code);
  - Sponza;
  - Bistro;
  - one furnished room;
  - one courtyard with a tree.

  Licenses are recorded.
- **T0.5 Viewer shell (three r187).**
  - Scene picker, orbit and fly, bookmarks, exposure, stats, split wipe.
  - three's `LightProbeGrid` and VXGI as the first baseline techniques.
  - Composes with `@pmndrs/upscaler`.
- **T0.6 Native harness.**
  - Portable device selection.
  - A `-Scene` launcher with a clean mode, and cameras from manifests.
  - Sponza and Bistro running natively.
  - Per-pass timings for every native-capable scene.
  - A D3D12 build.
- **T0.7 References and comparison.**
  - Blender Cycles renders from manifests.
  - three-gpu-pathtracer plus `@pmndrs/denoiser` as a reference plug-in.
  - A `tools/compare` CLI reporting FLIP and mean error.
- **T0.8 Spike: three.js sharing.** A shared device, a registered GI `Light` node sampling an `ExternalTexture`, checked under the Standard, Meshlet and Visibility-Buffer pipelines; env diffuse off with env specular on; GI inside a reflection-probe capture.
- **T0.9 Spike: Babylon.** The same contract through Babylon's WebGPU engine and a material plugin.
  - Done, then paused. From M1 on, three.js is the only adapter until v1 is validated; Babylon, Pixi and r3f come in M5 or later (DECISIONS 2026-11-04). M1+ has no Babylon tasks. If the spike page breaks, it leaves the required `test:gpu` set.
- **T0.10 Spike: BVH.**
  - Our two-level BVH (binned SAH, built in a worker) with WGSL closest-hit and any-hit, vs the latest three-mesh-bvh, on Sponza and Bistro.
  - Correctness via the ported ray oracle tests.
  - Rays per second on each test device.
- **T0.11 Platform API draft.** TypeScript interfaces for the platform, the technique contract and the shading module, plus a written walk-through of voxel and surfel techniques on them. Revised with the spike results.
- **T0.12 M0 note and M1 breakdown.** One page: what works, what doesn't, the baseline table, spike answers and the revised API. Then M1 broken into tasks on the board.

### M0 done when

- `pnpm i && pnpm lint && pnpm typecheck && pnpm test && pnpm build` passes from a clean clone.
- `pnpm test:gpu` passes on NVIDIA and Intel. Apple devices (Mac, iPad, older iOS) are deferred to a later milestone (DECISIONS 2026-11-04).
- The viewer opens every test scene on a lit default view.
- three's `LightProbeGrid` and VXGI show as baselines with their frame and bake cost recorded per device.
- Native per-pass timings exist for every native-capable scene.
- Spikes T0.8–T0.10 each have a yes/no answer with numbers.
- The platform API is revised, and you've signed off in the viewer.

## Later milestones (outline)

Each is broken into tasks at the end of the previous milestone. The likely shape:

- **M1, core DDGI loop, static scene:**
  - the platform scene copy;
  - `@gi/trace` from the T0.10 result, with the top-level BVH refit on the GPU every frame and rebuilt now and then (moved up from M3; DECISIONS 2026-11-04);
  - `@gi/shade`;
  - the DDGI field and update loop (rotation, hysteresis, unlimited bounce, sky);
  - classification and relocation;
  - the shading module;
  - the three adapter's GI light;
  - the plain-WebGPU hello example;
  - viewer integration;
  - engine-neutrality comparison;
  - targets on Sponza, Cornell and two-rooms.
- **M2, dynamic lights:**
  - the light buffer;
  - change detection and prioritized updates;
  - light groups;
  - sky and time of day;
  - the light sweep and native side-by-side.
- **M3, moving objects and camera:**
  - worker builds (the per-frame top-level BVH moved to M1);
  - deforming-mesh refit;
  - the camera-following volume;
  - adapter change tracking;
  - the door, cabinet, tree and fly-through scenarios.
- **M4, materials and scenes:** UVs, the texture atlas, alpha cutouts, material mapping, every scene textured.
- **M5, tiers and release:**
  - presets;
  - saved bakes;
  - the `LightProbeGrid` JSON export;
  - docs and the adapter guide;
  - the examples site;
  - release prep.

## Guardrails (lessons from the audit)

- **Product checks only.** No checks that only prove other checks ran, and no byte-identical-repeat requirements, since temporal DDGI is stochastic by design.
- **Threshold changes are a decision,** written in `docs/DECISIONS.md`, never a quiet edit to make something pass.
- **"It builds" is not "it works".** GPU-touching tasks are verified in a real browser on a real GPU and say which device.
- **The viewer is for a human.** It opens on lit, useful views at normal exposure. Dark leak tests are a mode you pick, not the default.
- **Keep it readable.** If a file gets hard to read, split it.
