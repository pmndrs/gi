# Handoff: Windows → Mac (2026-11-10)

This handoff moves the work from Dennis's Windows laptop (RTX 3060 Laptop 6 GB + Intel UHD) to his M5 Pro MacBook, which runs on a fresh Claude account. Nothing from the old account comes along: the Review Desk artifact, auto-memory and session history all stay behind. Everything you need is in this repo, or on Dennis's NAS for the git-ignored data.

Read in order:
1. this file;
2. `docs/handoff/M3.md` and `docs/handoff/M4.md`, the per-milestone state written by each lead;
3. `CLAUDE.md`, `SPEC.md`, `PLAN.md`, `TASKS.md` and `docs/DECISIONS.md`.

## Where things stand

| Milestone | State | Branch |
|---|---|---|
| M0, M1 | Done, merged into `main` (PRs #1, #3) | `main` |
| M2, dynamic lights and time of day | Done and signed off. PR #4 (`m2` → `main`) is open and mergeable; Dennis merges it. His look at the viewer and compare page is pending but doesn't block. | `m2` |
| M3, moving objects and camera | Merged: T3.0, T3.1, T3.2, T3.3, T3.5, T3.6, T3.7, T3.11, T3.12. In review: T3.4 (`m3-t3.4`), T3.8 (`m3-t3.8`). Left: T3.9 (correctness checks), T3.10 (M3 note and M4/M5 breakdown). | `m3` |
| M4, materials and textured scenes | T4.1–T4.4 merged. Left: T4.5 (check and note, plus Dennis's look), in progress on `m4-t4.5`. | `m4` |
| Join, for looking at textured scenes | `m3` + `m4` merged, viewer defaults to `all-compressed` textures. Only for Dennis's look: each milestone goes to `main` through its own PR. | `m34` |

Then M5 (tiers and release prep) is outlined in `PLAN.md` and `SPEC.md`.

## Rules to work by (Dennis's standing direction, previously in auto-memory)

**Pace and gates (2026-11-09):** "Lets not get hung up on timings and get things built. when we have clear gi we can make decisions on paths but we've been behind on getting the milestiones at least prototype level for a week. Id like to clear m2, m3, and m4 at least very soon".
- Gate M3 and M4 on correctness and the look only: oracle, accuracy against references, leak, response in frames, stability, and 0 recompiles. Record timings where it's cheap, but never gate on them. No VALID runs, clock rules or medians of 3.
- Timebox a missed bar to one diagnosis and one fix attempt, then record a known miss in DECISIONS and move on.
- Polish follow-ups go to "Later".

**Priority:** dynamic updates (moving objects, time of day) matter more than startup or timing polish.

**Hardware:** the targets are an RTX 3060 and up, plus this M5 Pro. Intel UHD is for correctness only.
- Size defaults for 6–8 GB VRAM.
- Request the adapter's maximum limits; never size buffers to WebGPU's default 128 MiB binding.
- "anything taking minutes to build is a fail".
- Judge on real scenes: Sponza with curtains, Bistro.

**Architecture:**
- three.js (r187) is the only adapter until v1 is validated. The core stays engine-neutral, enforced by a lint rule.
- WebGPU feature-forward: subgroups first, one declared fallback per feature, no parallel GPU paths.
- Port techniques faithfully first. Speedups are measured levers beside the faithful path, logged in a divergence register.
- BVH: the published `@pmndrs/bvh` (pinned, 0.1.1; `packages/bvh` was removed in #63). Fixes go to pmndrs/bvh as PRs or issues. Port only permissively licensed code; never copy unlicensed repos.

**Agents:**
- Code and review run on Opus, at medium effort or higher. Sonnet is only for small jobs, tests and measuring.
- Many parallel agents are fine; stay token-conscious.
- The repo's agents are in `.claude/agents/`: `implementer`, `reviewer`, `gpu-tester`, `native-runner`, `wrangler` and `scout`.

**Safety:**
- Never change GPU or system settings (clocks, power, fans, sleep).
- Don't stop processes you didn't start; other GI projects share the GPU.
- Merge task branches into the milestone integration branches after review. Only Dennis merges into `main`.
- Outward-facing actions, such as PRs to other repos like the three.js ExternalBufferAttribute idea, need Dennis to see the text first.
- Write Dennis's decisions into `docs/DECISIONS.md` with his exact words.

## Open decisions for Dennis

1. **Widen SPEC's "one GPU submit and one readback at load" exception.** This is provisional and already in place, recorded in DECISIONS 2026-11-09/10, `SPEC.md`:193 and `docs/api/platform.md`.
   - T3.2: a mesh added or replaced at runtime gets one H-PLOC build submit plus one async stats readback.
   - T3.3: a deforming mesh reads its refit stats back at most once every 15 frames, and a drift rebuild reads the new tree layout back.
   - Still scenes do neither.
   - Recommendation: confirm both.
2. **M4 contract additions to review together at T4.5:** `capabilities.perTriangleMaterials` and `sampleTextures`.
3. **His looks:**
   - M2: the viewer and `compare.html`;
   - M3/M4: textured scenes on `m34` (`sponza-full`, `bistro-exterior`, `bistro-interior`, `furnished-room`, `courtyard-tree`, and the procedural M4 scenes).

## Data that git doesn't carry (Dennis copies these over from the NAS)

- `assets/downloads/` (69.6 GB, 2,387 files, measured 2026-11-08): the Mac can re-fetch these with the repo's asset tools (sources and checksums are recorded), or copy them from the NAS.
- `bench/references/` (Cycles references):
  - everything is in the main checkout's `bench/references/` (`C:\Users\denni\Documents\dev\webgpu-ddgi\bench\references\`, about 17 GB, checked 2026-11-10). That covers M1's HD references, T2.16's re-renders and all 24 M4 textured references (`cycles-textured.json` plus six `cycles-textured-*.exr` per folder; skip the `-part*` halves; `docs/handoff/M4.md` lists the folders);
  - copy the whole folder: re-rendering on a CPU takes hours.
- `bench/recordings/` and `native/build/recordings` (the compare page's recordings): ours are in `webgpu-ddgi-m2-t210/bench/recordings/viewer`, native's in `webgpu-ddgi-m2-t27/native/build/recordings`. They're Windows-only. The native harness (D3D12/Vulkan) can't run on the Mac, so native comparisons pause until back on Windows.
- `bench/results` CSVs are in git.

## On the Mac

- The M5 Pro is a target GPU, so its first runs are useful for their own sake. Its timings aren't comparable with the RTX numbers; label them by device.
- Intel UHD correctness runs and native comparisons are Windows-only and pause.
- The native submodule points at `m2-t2.7` in DennisSmolek/ddgi-rtxgi (pushed). Run `git submodule update --init` if needed; there's no need to build it on the Mac.
- Known Intel-only issue: `DXGI_ERROR_DEVICE_HUNG` on the 4th page of 4-page GPU batches (now a TASKS "Later" item; run batches of 3 or fewer). It doesn't apply on the Mac.
