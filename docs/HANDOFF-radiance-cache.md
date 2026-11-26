# Radiance-cache track: handoff (2026-11-10)

Written on the Windows laptop for the next lead, a new agent on Dennis's Mac (M5 Pro) with no memory of this work. Read this first, then `CLAUDE.md`, the RC rows in `TASKS.md`, the newest RC entries in `PROGRESS.md`, and `docs/design/radiance-cache.md`, which holds the design and the divergence register.

## What this track is

The `radiance-cache` technique ports AMD's two-level radiance cache: GI-1.0, as shipped as GI-1.2 in Capsaicin v1.3, MIT licensed. It has screen-space probes plus a world-space hash-grid cache, and it is the second technique on the platform next to DDGI. The code lives in `packages/ddgi-core/src/techniques/radiance-cache/`. Research and the full plan are in the `ddgi` repo (`DennisSmolek/ddgi`), branch `claude/amd-gi-cache-research-e46e2e`, folder `research/gi-1.0/`. Start with `PLAN.md`, then `source-map.md`, `native-capsaicin.md` and `idtech8-performance.md`.

## Branches (all pushed to origin)

The repository moved from `DennisSmolek/webgpu-ddgi` to `pmndrs/gi` on 2026-11-10 (DECISIONS 2026-11-10); `origin` is `pmndrs/gi` and every branch below lives there. On 2026-11-19 `main` (M3-M5, the `@pmndrs/gi` package and its layer checks) was merged into the track on branch `rc-main-merge` (PROGRESS, same date); cut new RC branches from it, or from `radiance-cache` once it is merged there. The technique is not in the published package: no `/radiance-cache` subpath until Dennis decides.

| Branch | State |
|---|---|
| `radiance-cache` | The track's main line, cut from `m2`. Merged: RC1.1 to RC1.5 (phase 1), RC2.1 (screen probes), RC3.1 (hash grid and the `WorldCache` interface). Lint and typecheck passed at the last merge. |
| `rc2-2` | RC2.2 "first image", **in progress, not reviewed.** Two commits, then `d5ec2d1`: uncommitted work saved as-is at handoff (viewer technique entry, three/viewer GPU pages, shading and debug changes). Not verified. Run lint, typecheck and its GPU pages before trusting it. |
| `rc3-2` | RC3.2 "population and resolve", **built, not reviewed.** Five commits ahead of `radiance-cache`; register D30 to D33. Needs an Opus review, then a merge. |
| `rc1-*`, `rc2-1`, `rc3-1` | Already merged; kept for history. |

Next rows in order: finish, review and merge RC2.2 and RC3.2; then RC3.3 (GI-1.2 multi-bounce); then RC4.1 to RC4.3 (temporal probes and the LRU cache, the probe filter and SSGI, and the image denoiser built ready to move into `pmndrs/denoiser`). Phase 5 is the comparison against native Capsaicin. Phase 6 covers levers and dynamic scenes.

## How the work was run

- **Roles.** The lead session (strongest model) owns design, task breakdown and review. For each TASKS row, an implementer (Opus) works on branch `rcX-Y` cut from `radiance-cache`. Then an Opus reviewer checks the port against the Capsaicin source line by line and runs the GPU pages. The lead fixes or returns the review notes, merges into `radiance-cache` and marks the row done with the evidence.
- **GPU work** goes through `node tools/reference/bench-lock.ts run --no-cool-wait -- ...`, one GPU run at a time. Other tracks (M2/M3 DDGI and BVH) share the lock.
- **The Capsaicin source** is what the port cites (`gi1.comp:1212`-style references). `TASKS.md:530` points at a Windows path. On the Mac, clone Capsaicin at commit `914b91596cd119eda85fbc1d3c7ee6ac391b1452` (tag v1.3) somewhere local, or init the submodule `native/references/capsaicin` in the `ddgi` research branch, and update that line.

## Dennis's standing rules for this track

These were held in the old machine's local agent memory, so they're written out here.

- **Faithful port first.** Build Capsaicin's default configuration as AMD built it: algorithms, ray counts, rates and cache sizes. Performance ideas are measured levers next to the faithful path, never up-front rewrites.
- **Divergence register.** Record every difference from Capsaicin in `docs/design/radiance-cache.md` with its kind, its reason and a way back. Include workarounds other people or vendors use.
- **No timing for now** (2026-11-09). Correctness gates only; mark every timed check as deferred. Build as far as possible, and stop only for a decision that blocks the whole project. Note everything else in PROGRESS under "Next" and keep going.
- **No WebGPU default caps.** Ask for the adapter's maximum limits and size buffers by need. The hash cache runs at Capsaicin's 1 GiB. "128 MB caps are far too low."
- **Model tiers.** Code and review on Opus (medium effort or higher). Sonnet only for small jobs, test runs and measuring. Haiku for lookups.
- **Feature-forward WebGPU.** Subgroups first, with one declared fallback per feature and no parallel GPU paths.
- **three.js only** until the technique is validated. The core stays engine-free, and lint enforces it.
- **Siblings.** Denoisers and upscaling go to `pmndrs/denoiser` and `pmndrs/upscaler`. Anything they lack is a request to Dennis, not code in this repo.
- **Brixelizer** is being ported in another workstream and plugs in through the `WorldCache` slot. **id Tech 8** (Sousa, SIGGRAPH 2025) is the main source of performance levers. **GPUTex** handles compressed textures if needed.
- **Commit as you go**, so nothing is lost if usage runs out mid-row.

## The linked denoiser (RC4.3, one-time setup)

From RC4.3 on (branch `rc4-3`, then `radiance-cache` once merged), the radiance cache's GI denoiser is `Gi1Denoiser` from the sibling library pmndrs/denoiser (`@pmndrs/denoiser-ffx`, branch `gi1-denoiser`, draft PR #33), linked for development: the root package.json has `"@pmndrs/denoiser-ffx": "link:../pmndrs-denoiser/packages/denoiser-ffx"`, and `linkedSiblings` pins the commit. Nothing is published, so the link stays until a `denoiser` release carries it; then it becomes an ordinary dependency and `linkedSiblings`, `tools/linked-siblings.ts` and the CI step go.

- **Once per machine:** `node tools/linked-siblings.ts --fetch` (or `pnpm siblings:fetch` after a first install). It clones pmndrs/denoiser next to this repository (`../pmndrs-denoiser`) at the pinned commit and builds `denoiser-core` and `denoiser-ffx` (git, corepack and yarn 4 via corepack). Without it, `pnpm install` stops at its `preinstall` check with that command.
- **Working on both:** edit in `../pmndrs-denoiser`, rebuild with `corepack yarn workspace @pmndrs/denoiser-ffx build` (gi reads its `dist`), and move the pin in package.json when the denoiser branch moves (the check only warns when the clone is elsewhere).
- **Why not a git dependency:** pnpm installs a git subdirectory without its yarn workspace, so `@pmndrs/denoiser-core@workspace:*` doesn't resolve (`ERR_PNPM_WORKSPACE_PKG_NOT_FOUND`), and the package keeps no built output in git.
- CI (`ci.yml`, `deploy-pages.yml`) runs the `--fetch` step before `pnpm install`. pmndrs/denoiser is public, so no token is needed.

## First steps on the Mac

1. `git fetch`, then check out `radiance-cache`; run `node tools/linked-siblings.ts --fetch` (once; see "The linked denoiser" above), then `pnpm install`, `pnpm lint` and `pnpm typecheck`.
2. Check that the GPU test runner (`pnpm test:gpu`, headless Chrome) finds Chrome and the Apple GPU. It was only run on Windows (RTX 3060 and Intel UHD). Record the adapter name in PROGRESS. The M5 Pro is a real target device, not correctness-only.
3. Fetch the scene assets (`pnpm assets:fetch`); `assets/downloads` isn't in git.
4. Native Capsaicin is D3D12 on Windows only, so it can't run on the Mac. Native comparisons and timings stay on the Windows laptop. Reading the Capsaicin source works anywhere.
5. Pick up `rc3-2`'s review and `rc2-2`'s verification, as in "Next rows" above.
