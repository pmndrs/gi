# Radiance-cache track: handoff (2026-11-10)

Written on the Windows laptop for the next lead, a new agent on Dennis's Mac (M5 Pro) with no memory of this work. Read this first, then `CLAUDE.md`, the RC rows in `TASKS.md`, the newest RC entries in `PROGRESS.md`, and `docs/design/radiance-cache.md`, which holds the design and the divergence register.

## What this track is

The `radiance-cache` technique ports AMD's two-level radiance cache: GI-1.0, as shipped as GI-1.2 in Capsaicin v1.3, MIT licensed. It has screen-space probes plus a world-space hash-grid cache, and it is the second technique on the platform next to DDGI. The code lives in `packages/ddgi-core/src/techniques/radiance-cache/`. Research and the full plan are in the `ddgi` repo (`DennisSmolek/ddgi`), branch `claude/amd-gi-cache-research-e46e2e`, folder `research/gi-1.0/`. Start with `PLAN.md`, then `source-map.md`, `native-capsaicin.md` and `idtech8-performance.md`.

## Branches

Updated 2026-11-10 on the Mac (M5 Pro). Nothing from the Mac has been pushed yet.

| Branch | State |
|---|---|
| `radiance-cache` | The track's main line. Merged: phase 1 (RC1.1-RC1.5), phase 2 (RC2.1, RC2.2), phase 3 (RC3.1-RC3.3) and phase 4 (RC4.1-RC4.3), plus `rc2-1-m5` (the M5 fix D34) and the asset-tool fix. That's the whole faithful baseline, every pass of Capsaicin's default GI-1.2 except GI-1.1 reflections (D8). Lint, typecheck and test pass. The twelve RC GPU pages pass on the M5. |
| `rc*` branches | Merged; kept for history. |

**Where it stands.** The faithful port is built. Phase 5 (the per-pass comparison with native Capsaicin) needs two things the Mac can't give: timing, which Dennis deferred on 2026-11-09, and native Capsaicin, which is D3D12 on Windows only. Phase 5 has no TASKS rows yet; the lead writes them. **Nothing has been re-run on the RTX 3060 or the Intel UHD since the handoff.** The binding packing, the 718 MB cache binding and every phase 3-4 kernel are checked on the M5 only.

**Open items** (details in PROGRESS "Next" and NOTES):
- The M5's `rgba16float` stores round toward zero (D59): probe hit distances settled about 30 m low at 40 m. Fixed in RC6.2 (`rcNearestHalf` before the atlas stores); D3D's own rule is round toward zero (D3D11.3 §3.2.2, which D3D12 inherits: "Baseline information about rendering pipeline behaviors that are common between D3D11 and D3D12 was inherited from the D3D11 spec", DirectX-Specs index), so Capsaicin probably drifts the same way (by D3D's rule; not run).
- "One-bounce DDGI reads high" (NOTES) needs three fairer checks before it goes to the DDGI track.
- Flicker after the denoiser, converged: Cornell 0.74% / 3.16%, Sponza 0.90% / 2.91%. That's mostly SSGI's single slice (lever 16). DDGI is about 0.1% / 0.5%.
- Asset checksums differ between Mac and Windows. The Intel Sponza `.gltf` files differ only in line endings; Bistro and the courtyard tree differ in vertex data. Dennis decides the check's policy.
- The package name `@pmndrs/denoiser-gi1` (private) is Dennis's call.
- The design doc's "worth telling AMD" list has the Capsaicin bugs and questions found so far.

## How the work was run

- **Roles.** The lead session (strongest model) owns design, task breakdown and review. For each TASKS row, an implementer (Opus) works on branch `rcX-Y` cut from `radiance-cache`. Then an Opus reviewer checks the port against the Capsaicin source line by line and runs the GPU pages. The lead fixes or returns the review notes, merges into `radiance-cache` and marks the row done with the evidence.
- **GPU work** goes through `node tools/reference/bench-lock.ts run --no-cool-wait -- ...`, one GPU run at a time. Other tracks (M2/M3 DDGI and BVH) share the lock.
- **The Capsaicin source** is what the port cites (`gi1.comp:1212`-style references). `TASKS.md:530` points at a Windows path. On the Mac, clone Capsaicin at commit `914b91596cd119eda85fbc1d3c7ee6ac391b1452` (tag v1.3) somewhere local, or init the submodule `native/references/capsaicin` in the `ddgi` research branch, and update that line.

## Dennis's standing rules for this track

These were held in the old machine's local agent memory, so they're written out here.

- **Faithful port first.** Build Capsaicin's default configuration as AMD built it: algorithms, ray counts, rates and cache sizes. Performance ideas are measured levers next to the faithful path, never up-front rewrites.
- **Divergence register.** Record every difference from Capsaicin in `docs/design/radiance-cache-divergences.md` with its kind, its reason and a way back. Include workarounds other people or vendors use.
- **Why each difference exists** (added 2026-11-10). Every row also gives its cause and evidence (the old challenge list, merged into the same file on 2026-11-11), plus its image and speed impact, effort to match and a recommendation. The line gives its cause (forced by WebGPU or three, our own rule, a shortcut, same result, a fix, license) and its evidence (measured, cited or assumed). Dennis wants to tell real platform limits apart from easier choices and from "WebGPU can't" said from memory, so that the unforced ones can be challenged once the port reaches parity. Reviewers check the cause and look for differences that have no row at all.
- **No timing for now** (2026-11-09). Correctness gates only; mark every timed check as deferred. Build as far as possible, and stop only for a decision that blocks the whole project. Note everything else in PROGRESS under "Next" and keep going.
- **No WebGPU default caps.** Ask for the adapter's maximum limits and size buffers by need. The hash cache runs at Capsaicin's 1 GiB. "128 MB caps are far too low."
- **Model tiers.** Code and review on Opus (medium effort or higher). Sonnet only for small jobs, test runs and measuring. Haiku for lookups.
- **Feature-forward WebGPU.** Subgroups first, with one declared fallback per feature and no parallel GPU paths.
- **three.js only** until the technique is validated. The core stays engine-free, and lint enforces it.
- **Siblings.** Denoisers and upscaling go to `pmndrs/denoiser` and `pmndrs/upscaler`. Anything they lack is a request to Dennis, not code in this repo.
- **Brixelizer** is being ported in another workstream (`/Users/dex/Documents/Codex/2026-11-09/wha/outputs/brixelizer-webgpu/`) as a distance-field library, not a GI cache. It plugs in as a long-ray tracing backend (design doc, "Brixelizer: what it is and where it fits"), not through the `WorldCache` slot. **id Tech 8** (Sousa, SIGGRAPH 2025) is the main source of performance levers. **GPUTex** handles compressed textures if needed.
- **Commit as you go**, so nothing is lost if usage runs out mid-row.

## The Mac setup (done 2026-11-10)

- Capsaicin v1.3 is at `/Users/dex/Developer/capsaicin`; the research notes are at `/Users/dex/Developer/ddgi-research/research/gi-1.0/`.
- Blender 4.5.0 is at `/Users/dex/Applications/Blender-4.5.0.app` (the converter needs exactly 4.5.0). Run the fetch as `BLENDER=/Users/dex/Applications/Blender-4.5.0.app/Contents/MacOS/Blender pnpm assets:fetch`. It reports 7 output mismatches, but the files load (see "Open items").
- The extra worktrees share the assets through a symlink, `assets/downloads`, pointing at the main worktree's.
- **GPU runs on the Mac:** `node tools/reference/bench-lock.ts run --no-cool-wait -- pnpm test:gpu --filter <page>`, with no `--gpu` (it accepts only nvidia/intel).
- **The adapter:** apple/metal-3. It has shader-f16, subgroups, timestamp-query and texture-formats-tier1/2; 4 GiB storage bindings; **10 storage buffers per stage**, so the platform's 8 matters here.
- **Apple facts:** `pack2x16float` rounds to nearest, while storage-texture half stores round toward zero (NOTES).
- **The working pattern:**
  - Opus implementers on `rcX-Y` worktrees, and Opus reviewers checking against Capsaicin line by line, plus each row's cause in the divergences file.
  - The lead fixes the review notes or sends them back, then merges.
  - Sonnet does setup and test sweeps.
  - A reviewer's second hand-back can be refused, so ask reviewers to also write the report to the scratchpad.
