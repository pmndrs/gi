# Decisions

Settled choices, newest first. Each entry is short: the decision, why, and what would make us revisit it. Changing a threshold or target counts as a decision and goes here.

## 2026-11-04

- **Code is written and reviewed on Opus.**
  - `implementer` and `reviewer` run on Opus at high effort, and `native-runner` on Opus at medium.
  - `gpu-tester` stays on Sonnet, and `scout` and `wrangler` on Haiku: running checks, measuring, lookups and mechanical jobs.
  - *Why:* Dennis: Sonnet is "great for small things, tests, etc." but not the best coder; "opus-medium at a minimum with opus-high being fine." Also, a builder and a reviewer on the same tier tend to miss the same things.
  - *Revisit if:* token cost becomes a problem. Then split small scaffolding jobs out to a Sonnet task, rather than downgrading the implementer.
- **Two ground-truth renderers must agree before we trust either.** Cycles (CPU) and three-gpu-pathtracer count as agreeing on a scene when, compared with `tools/compare` after 8×8-pixel block averaging:
  - **full image:** overall brightness within ±0.5%, block error at most 1%;
  - **indirect only:** brightness within ±1%, block error at most 3%.
  - Cornell measured -0.002% / 0.37% (full) and -0.014% / 1.43% (indirect). Two-room measured +0.11% and +0.12% brightness.
  - *Why these numbers:*
    - Block averaging cancels per-pixel noise. Two Cycles seeds already differ by 0.8% per pixel, so per-pixel limits would only measure noise.
    - The limits are about 2–3× what we measured. That leaves room for noise, while a wrong light unit, a missing bounce or a flipped axis would still fail: those show up as 5% or more, as the 12% light-range gap on two-room did.
    - DDGI's own targets against these references will be several percent, so references agreeing within 0.5–1% is accurate enough to judge DDGI.
  - *Revisit if:* a run sits near a limit. In that case, measure the path tracer's own seed-to-seed noise; it wasn't measured in T0.7, and its noise likely makes up most of the 0.37%. Also revisit if DDGI targets tighter than about 3% need sharper references.
- **Sponza is Intel's "New Sponza" (base package), download-only.**
  - `pnpm assets:fetch` downloads `main_sponza.zip` (3.99 GB) from Intel. Nothing from it is committed. It isn't shown on a public examples site without Intel's written OK.
  - Its license file is CC BY 4.0 text with a preface: "For personal use and educational use. Limited commercial use for marketing and print purposes." Intel's samples page says "Downloads are available under the Creative Commons Attribution license."
  - The Khronos glTF Sponza is excluded. Its model files are under the CRYENGINE Limited License (§2.1.4: "Games in object code form only"). three.js r187's Sponza examples load that Khronos copy, so our Sponza numbers won't be the same model as three's examples. We compare against three's techniques running on our copy instead.
  - *Why:* Dennis: "use the intel or the 'improved sponza' ive seen floating around." Intel's version is the improved one, and its license is the clearest available. Every other Sponza variant derives from Crytek's model.
  - *Revisit if:* we want Sponza on the public examples site (ask Intel, sponza_feedback at intel dot com), or the base package is too heavy for the BVH spike. In that case use a stated subset of it.
- **Apple-device testing is deferred past M0.** M0 measures on the Windows laptop only (RTX 3060 and Intel UHD). The Mac, iPad and older iOS rows of T0.3, the Mac columns of T0.5 and T0.10, and the M0 "every test device" check move to a later milestone.
  - *Why:* Dennis: "We can skip the apple/device testing for now until we progress further." The LAN device page (`pnpm device:serve`) is built and ready for when we pick it up.
  - *Revisit when:* the DDGI loop runs (M1 or M2), so the Apple numbers measure something real.
- **Toolchain mirrors `@pmndrs/upscaler`.**
  - pnpm workspace with an umbrella package over `-core`/`-three`.
  - vite, vitest, rollup type bundling, ESLint 9.
  - Headless Chrome over the DevTools protocol for GPU tests (no Playwright).
  - *Why:* sibling libraries should feel alike. *Revisit if:* CDP can't drive something we need.
- **One package, many import paths.** `@pmndrs/ddgi` with `/core`, `/three`, later `/babylon` and `/pixi`. The platform stays a separate internal module.
  - *Why:* matches the upscaler, and keeps install simple.
  - *Revisit when:* a second GI technique ships. Then the platform may become `@pmndrs/gi-core`.
- **The host owns the GPU device.** `requiredLimits(adapter)` tells the host what to request, and the core works at default limits for typical scenes.
  - *Why:* the same rule as the upscaler and denoiser.
- **The three.js integration is a `Light` subclass with a registered node, never `material.contextNode`.**
  - Fragment-readable data lives in textures wrapped by `ExternalTexture`.
  - *Why:* this is r187's supported path, and it works with the Meshlet and Visibility-Buffer pipelines.
  - *Revisit if:* T0.8 finds a blocker.
- **Pinned to three.js r187.** *Why:* the r187 light-node and `ExternalTexture` paths.
- **Test hardware:**
  - Windows RTX 3060 Laptop and Intel UHD;
  - Mac M5 Pro 64 GB;
  - an iPad and an older iOS device;
  - Android as a stretch goal.
- **Name:** `pmndrs/ddgi`. Dennis has approval.
- **The code is a GI platform plus techniques; DDGI is the first technique.** *Why:* DDGI is the foundation for later voxel, SDF, surfel, cascade and RT-reflection work.
- **Reflections layer on top of DDGI; they aren't a DDGI output.**
  - v1 feeds reflection probes and SSR, and avoids double-counted env diffuse.
  - RT reflections are the first technique after v1.
- **Scenes are whatever the host builds.** No assumption about file formats.
- **Characters don't feed GI by default.** Each object has a "contributes to GI" flag; deforming meshes like trees opt in.

## 2026-10-30

- **Fresh start in a new private repo.** The old `DennisSmolek/ddgi` stays frozen at the `review-baseline-2026-10-30` tag. *Why:* see `docs/AUDIT-2026-10-30.md`.
- **MIT, implemented from the papers.** No RTXGI SDK code or shaders are copied; the native harness is a comparison tool only.
