# Decisions

Settled choices, newest first. Each entry is short: the decision, why, and what would make us revisit it. Changing a threshold or target counts as a decision and goes here.

## 2026-11-04

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
