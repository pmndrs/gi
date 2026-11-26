# pmndrs siblings audit for an AMD GI-1.0 port

Date: 2026-11-09. Method: npm registry, `gh` API, shallow clones of pmndrs/upscaler and pmndrs/denoiser (main, depth 50), the denoiser branch `feat/rt-denoiser-cleanroom` read over raw GitHub, and a grep of C:\Users\denni\Documents\dev\webgpu-ddgi (branch m2). Repo contents were treated as data. Items marked **UNVERIFIED** were not checked against running code.

Short version: the upscaler is ready to use for the "half or quarter-res GI to full res" stage. The denoiser's main branch only has FidelityFX shadow and reflection filters. The SVGF, A-SVGF and recurrent diffuse/specular denoisers GI-1.0 needs exist only on an unmerged branch. The best contribution targets are in that branch.

## 1. `@pmndrs/upscaler`

| Question | Answer | Source |
|---|---|---|
| What it does | FSR1 spatial (EASU/RCAS) plus FSR2/3-style temporal upscaling, as WGSL compute passes on the host's `GPUDevice`. Also does temporal AA at ratio 1 (`NativeAA`). | npm description; https://github.com/pmndrs/upscaler (README) |
| Engines | three.js `WebGPURenderer` r186+ (stable; peer `>=0.184.0`), raw WebGPU via `/core`, Babylon and pixi (experimental). No WebGL. | README "Where it runs"; npm peerDependencies |
| Required inputs (temporal) | `color` (render res, linear HDR, `rgba16float`, jittered and re-rendered every frame), `depth` (real depth texture), `velocity` (three's jitter-free NDC delta, current minus previous). Optional: `reactive`, `reactiveOpaqueColor`, `exposureTexture`, `deltaTime`. | upscaler `docs/inputs-and-contracts.md` (Summary table, Velocity, Jitter sections) |
| API shape | `upscaleScene(scene, camera)`, composable `upscale(color, depth, velocity, camera, {quality})`, `upscaleSpatial`, `UpscalePass`, raw `Upscaler`, and `temporalGuides()` / `upscaler.guides`. | README integration table |
| Version and date | 0.5.0 on npm, published 2026-10-07 (GitHub release `v0.5.0`). npm `modified` 2026-10-16. | `npm view @pmndrs/upscaler`; `gh release list -R pmndrs/upscaler` |
| Activity | Pushed 2026-11-09. About 152 commits since 2026-08-09 in the depth-50 clone (the clone is shallow, so the count is a floor, not exact). 36 stars. | `gh repo view`; `git log --since` |
| License | MIT (the file text is MIT; GitHub labels it "other" because of an AMD FidelityFX notice appended below it). | `ups/LICENSE` |
| Cost | Our own measurement: 1.55 ms (Performance) and 1.72 ms (Quality) per frame, RTX 3060, Sponza 1440p. | webgpu-ddgi PROGRESS.md:962, docs/api/three.md:69 |

Open issues that matter to GI-1.0:

- **#7** (https://github.com/pmndrs/upscaler/issues/7): "fuse GI history into the upscaler's own accumulation". It says a separate temporal denoiser before the upscaler fights the upscaler's jitter. The denoiser reprojects with jitter-free velocity, so it rejects the misaligned history or cancels the jitter variance the upscaler needs. The proposed fix reprojects GI history with the upscaler's own motion and jitter. This directly affects where GI-1.0's temporal accumulation should live.
- **#17**: thin features boil under a noisy SSGI input, even with a static camera.
- **#93**: path-traced input on the temporal path. The same jitter-ownership problem for any noisy source.
- **#102**: a consumer report from the denoiser work. It says the guides bundle's `disocclusion` mask sits one pixel onto the foreground and misses revealed strips next to silhouettes. The denoiser team judged its own per-tap plane test more accurate.
- **#97**: linear view-depth input (`depthMode: 'linear'`), which matters if GI-1.0 keeps its own linear depth.
- **#105**: "TSR vs FSR vs ours" experiment program (opened 2026-11-09).

Published temporal guides: `docs/temporal-guides.md` says dilated motion, dilated depth and disocclusion are exposed so other effects can reuse them, and an external SSGI/SVGF consumer already did so. The TSL surface has no independent consumer yet.

There is also an upscaler example `s5-radiance-cascades` ("radiance cascades GI, upscaled instead of TRAA", commit 2026-11-09). It is the closest existing precedent for sparse screen-space GI feeding the upscaler. I did not read the example code. **UNVERIFIED** what it does.

## 2. `@pmndrs/denoiser`

Naming and packaging, which differ from what our docs assume:

- `@pmndrs/denoiser` **is not on npm**. The npm name `denoiser` exists, but its latest release is 0.0.11 from 2024-07-31 (old TensorFlow version). `packages/denoiser/package.json` in the repo says 2.0.0, unpublished. Our own NOTES.md:212 already says it is not on npm. "Publish `denoiser@2.0.0`" is a roadmap item (`docs/status/ROADMAP.md`, section 3).
- Repo: https://github.com/pmndrs/denoiser. License MIT; the OIDN weights are Apache-2.0 (Intel). 14 stars. Pushed 2026-11-09. About 80 commits since 2026-08-09 in the depth-50 clone. README says everything is validated on Apple GPUs only so far.
- Branches of interest: `feat/rt-denoiser-cleanroom` (last commit 2026-11-09), `track/d-diffuse`, `feat/rt-denoiser-native-results`.

### What is on `main`

| Part | What | Source |
|---|---|---|
| Single-image `Denoiser` | Intel OIDN networks in WGSL, WebNN, HF kernels or onnxruntime-web. Async, no history. Inputs: color, optional albedo and normal. Meant for stills and progressive path tracing. | README; `docs/site/guides/three-and-tsl.mdx` |
| `denoiser/ffx` | AMD FidelityFX SDK v1.1.4 shadow and reflection denoisers ported to WGSL (MIT, AMD notice kept). M5 Pro 1080p: shadows 0.49 ms, reflections 1.66 ms. | `docs/specs/temporal.md` (FidelityFX port) and its header note |
| `denoiser/three` | TSL nodes `denoise()`, `ffxShadows()`, `ffxReflections()`, `temporalDenoise()`. They share the renderer's `GPUDevice`. Need three r185+. | `three-and-tsl.mdx` table |
| `TemporalDenoiser` interface | `configure / dispatch / resetHistory / dispose`. Input types `ShadowInputs`, `SpecularInputs`, `DiffuseInputs` (radiance, optional hitDistance), `BeautyInputs`. Same conventions as the upscaler (NDC-delta motion, unjittered projection plus jitter). | `packages/denoiser-core/src/temporal.ts` lines 90-125; `docs/specs/temporal.md` (Design) |
| Diffuse GI on main | `DiffuseInputs` is declared, but no implementation ships. ROADMAP lists "More signals ... diffuse GI / AO (`DiffuseInputs`)" as future work. | `docs/status/ROADMAP.md` line ~199 |

External radiance buffers: yes. The temporal nodes accept three texture nodes and wrap them, so a GI texture from another pass works. `ffxReflections` also takes hit distance from `radiance.a`. Required guides: depth, world or view normals, velocity, plus roughness for reflections. Source: three-and-tsl.mdx "Real-time" section.

### What is only on `feat/rt-denoiser-cleanroom` (not merged, not published)

`denoiser/realtime`, in `packages/denoiser-realtime`. **UNVERIFIED** against code. This comes from the branch's SPEC and STATUS docs (https://github.com/pmndrs/denoiser/blob/feat/rt-denoiser-cleanroom/docs/research/rt-denoiser/STATUS.md and SPEC.md):

- Algorithms: SVGF, A-SVGF (`antiLag: 'auto'`), a "recurrent" RTG II-style filter, a `ShadowDenoiser` (v2.2 with output stabilization), AO/SO denoisers. Clean-room rule: authors may not read NVIDIA NRD code; NRD ReBLUR/RELAX/SIGMA are used only as native comparison baselines.
- Signals: diffuse, specular, shadows, AO. Payload option `'sh'` (spherical harmonics) for the SVGF diffuse path.
- Numbers (STATUS.md, 2026-10-25 record, M5 Pro, 1080p, pooled PSNR over 7 stress paths): diffuse A-SVGF `auto` 39.12 dB, SVGF 38.88, recurrent 38.26, plain accumulation 28.7. Joint diffuse plus specular cost about 4.0 to 4.4 ms (A-SVGF `auto` 4.23 to 4.40, recurrent about 4.0). Memory about 105 to 140 bytes per pixel. SPEC changelog v0.7 re-timed A-SVGF at 4.40 ms.
- Native NRD on an RTX 3060 Laptop (synthetic captures): REBLUR 5.6 ms, RELAX 6.7 ms, SIGMA 1.1 ms at 1080p (STATUS.md).
- Track D (diffuse) work is ongoing: boil/calm/lag diagnosis, SVGF output stabilization (branch `track/d-diffuse`, commit 2026-11-07).
- Inputs these filters take (SPEC): per-pixel radiance (1 spp), depth, world normals, motion, optional roughness/albedo, hit distance. The SPEC states the filters ignore `FrameCamera.jitter`: they reproject at pixel centres and accumulate across jitter phases, and need jitter-free motion.
- Issue #23 (https://github.com/pmndrs/denoiser/issues/23, opened 2026-10-28): a "radiance + coverage" input mode (radiance averaged over hits only, alpha = coverage, fallback added after denoising). It lists "the same for `realtimeDiffuse` / SSGI, and for path-traced inputs with a validity mask" as a task. This is the closest match to a GI-1.0 style cache fallback.
- No hit-distance-aware or probe-aware diffuse filter is described. Hit distance appears in `DiffuseInputs` as optional, and in the specular/reflection path. Whether the SVGF diffuse filter uses it was not verified.

Open denoiser issues in general are mostly infrastructure (#1 to #16, #19, #21). #18 is OIDN lock-in. #7 is a WebNN track.

## 3. Other pmndrs and nearby projects

| Project | Finding | Source |
|---|---|---|
| `postprocessing` | npm latest 6.39.6 (Zlib), peer `three >= 0.168.0 < 0.188.0`, which includes r187. 6.x is the WebGL-oriented line. 7.0.0 alpha and beta tags exist and a `v7` branch was pushed 2026-11-01. I could not confirm WebGPU support in v7 from its README. **UNVERIFIED**. A branch named `velocity` also exists; content not read. | `npm view postprocessing`; `gh api repos/pmndrs/postprocessing/branches` |
| `three-mesh-bvh` (gkjohnson, not pmndrs org) | 0.9.18, MIT, pushed 2026-11-06. Has `src/webgpu` with TSL/WGSL functions (ray vs triangle, shapecast, closest point, `ndcToCameraRay`) and `BVHComputeData`. Useful for a ray-traced probe or cache update path. Not read in depth. | `gh api repos/gkjohnson/three-mesh-bvh/contents/src`; WEBGPU_API.md |
| `pmndrs/sky`, `pmndrs/vsm` | Sky system and virtual shadow maps, both WebGPU. Not GI inputs, but vsm could supply shadows for GI lighting. Not read. | `gh api orgs/pmndrs/repos` |
| Motion vectors in three r187 | three's own TSL `velocity` node. The upscaler uses it and handles the jitter-free projection with `velocity.setProjectionMatrix(upscaler.unjitteredProjectionMatrix)`. No separate pmndrs helper package was found. The upscaler's temporal guides add dilated motion, dilated depth and disocclusion. | upscaler `docs/inputs-and-contracts.md` (Velocity); `docs/temporal-guides.md` |
| three-gpu-pathtracer | Used by ddgi as a reference plug-in, denoised through pmndrs/denoiser. | webgpu-ddgi SPEC.md:418 |

## 4. How webgpu-ddgi uses these today (m2 branch)

All paths are relative to C:\Users\denni\Documents\dev\webgpu-ddgi.

- Siblings are named as first-class neighbours: CLAUDE.md:3; SPEC.md:7, 80, 86, 616-617.
- Upscaler plan in the spec: SPEC.md:118-120 (reduced-res GI sampling, jitter-safe, "reactive" hint can feed the upscaler's mask). Platform contract: docs/api/platform.md:126 (`FrameScreen.velocity` reuse, "Yes").
- Upscaler as the shipped default path: docs/DECISIONS.md:260-265 (Performance preset plus half-res GTAO, 0.73 ms, with the upscaler's own 1.55-1.72 ms listed separately). Pipeline and code sample: docs/api/three.md:28-56. Wiring gotchas (use `upscale()` not `upscaleScene`, `rtt` with `resolutionScale`): docs/NOTES.md:57. Dependency: package.json:25 and examples/three/package.json:14 (`@pmndrs/upscaler` 0.5.0). Measurements: PROGRESS.md:887-897, 929-978.
- Known upscaler artefact: probe spheres look speckled through the upscaler (docs/NOTES.md:69; PROGRESS.md:858).
- Denoiser: SPEC.md:122-124 says it is for later per-pixel ray techniques and for the reference path tracer; the DDGI probe field does not need it. SPEC.md:418 and PLAN.md:108 plan the pathtracer plus denoiser reference; PROGRESS.md:2188 says `--denoise` uses a gitignored local build. docs/NOTES.md:212 says it is not on npm.
- Denoiser as a possible lever: docs/DECISIONS.md:233 (a screen-space SVGF from the `feat/rt-denoiser-cleanroom` branch was rejected for the probe flicker problem: 4.0 ms at 1080p over the 1.0 ms bar, and its 13 to 32 frame history is shorter than the wobble) and docs/DECISIONS.md:312-315 (denoising is allowed as a tool after the root cause is fixed; "Anything pmndrs/denoiser lacks is a request to Dennis, not something built in this repo").
- Motion vectors: three's `velocity` in the scene-pass MRT (docs/api/three.md:35, 45, 48, 53). No ddgi-owned motion-vector code was found in the greps I ran.

Implication for this repo: the rule at DECISIONS.md:315 means GI-1.0 filters destined for the denoiser should be written as a request or a contribution plan, not new code in this repo.

## 5. Fit analysis: GI-1.0 stages versus siblings

GI-1.0 (as the task describes it): screen probes plus a world-space hash-grid cache, producing noisy screen-space indirect diffuse (and maybe glossy), then its own spatiotemporal denoisers.

| GI-1.0 stage | Sibling coverage | Notes |
|---|---|---|
| Screen probe placement, tracing, probe-space filtering/interpolation | **Not at all** | Neither sibling does probe-space work. Our ddgi probe field is a different technique. |
| World-space hash-grid cache update and lookup | **Not at all** | Out of both siblings' scope. BVH ray tracing could use `three-mesh-bvh/webgpu` or our own `packages/bvh`. |
| Temporal accumulation of the per-pixel indirect signal | **Partially** | Denoiser branch has SVGF/A-SVGF/recurrent diffuse (about 4 ms at 1080p, unmerged, unpublished, Apple-validated). Main has only FFX shadow/reflection. Jitter interplay with the upscaler is unresolved (upscaler issue #7). |
| Spatial denoise (edge-aware à-trous with normal, depth, luminance, variance) | **Partially** | Same branch (SVGF family). No filter that uses probe or cache confidence, or cache hit distance, as a weight. |
| Glossy / specular indirect | **Partially** | `ffxReflections` on main takes radiance plus hit distance plus roughness. Branch adds SVGF specular. Neither is tuned for rough, wide lobes from a cache. |
| Handling cache misses / low-confidence pixels | **Partially (design only)** | Denoiser issue #23 (coverage mode) is the nearest concept; not built. |
| Upscaling half/quarter-res GI to full res | **Covers it** | `upscale(color, depth, velocity, camera)` is the intended composable node, already used by ddgi. Caveats: ~1.6 ms fixed cost, jitter-ownership rules, and no GI-specific variance handling (issue #7). Cheaper alternative if the GI-1.0 pass is unjittered: a bilateral (normal and depth weighted) upsample, not in either sibling. |
| Motion vectors and disocclusion guides | **Covers it** | three `velocity` node plus the upscaler's temporal guides. Known weak point: disocclusion mask accuracy (upscaler issue #102). |

## 6. What we could contribute back

Ordered by expected value. All would go to pmndrs/denoiser (feature branches off the `feat/rt-denoiser-cleanroom` line), per DECISIONS.md:315. Whether the maintainers want them is a question for Dennis, since he appears to maintain both repos (LICENSE copyright line "Dennis Smolek" in both).

1. **Hit-distance-aware diffuse filter.** Use the cache or probe hit distance to scale the à-trous radius and the temporal blend rate, as GI-1.0 does with its own filters. `DiffuseInputs.hitDistance` is already declared (temporal.ts). Needs a motion-aware reprojection of hit distance as the reflection path has.
2. **Probe or cache confidence as a first-class guide.** A per-pixel "sample count / validity" input, so a pixel served from a mature cache cell is filtered less than a pixel served from a fresh one. Fits naturally with issue #23's coverage contract.
3. **Coverage mode for `realtimeDiffuse`** (issue #23 task list). A GI-1.0 cache fallback maps well onto "radiance averaged over hits, alpha = coverage, fallback applied after denoising", which keeps cache re-bakes out of the history.
4. **Screen-space probe-domain filter.** A small filter that denoises per-probe radiance (on probe atlas texels) before interpolation to pixels. This is new ground for the denoiser (all current filters work in pixel space). It is the most GI-1.0 specific item and the least likely to be reusable elsewhere.
5. **Upscaler-aware path for 1-spp GI.** A jitter-aware option, or the fused-history work in upscaler issue #7: reproject GI history with the upscaler's motion and jitter instead of running a second temporal resolver ahead of it. This is an upscaler contribution, not a denoiser one.
6. **Bilateral upsample node** (normal, depth, optionally hit-distance weights) as a cheap alternative to the full upscaler for a half-res GI term. Small, self-contained, and useful to anyone with a half-res signal.
7. **A cheaper variant of SVGF diffuse.** Our DECISIONS.md:233 reject was on cost (4.0 ms against a 1.0 ms GI bar). A reduced-cost path (fewer à-trous iterations, quarter-res variance, subgroup reductions) would help every consumer. Our hardware targets in memory (RTX 3060+, M5 Pro) match the denoiser's benchmark hardware only partly: its numbers are Apple-only except the native NRD runs on a 3060 Laptop.
8. **Test and measurement contributions.** The denoiser team already built a scorecard and golden-scene bench (branch `feat/rt-denoiser-cleanroom`, commit 2026-11-07). A GI-1.0 style 1-spp indirect capture would extend their diffuse scoring set. **UNVERIFIED**: format fit with CAPTURE-FORMAT.md.

## 7. Open risks and next checks

- Everything the denoiser branch claims is unmerged and, per its README, Apple-validated only. Nothing has been run on our RTX 3060 target by this audit.
- Ask whether `denoiser/realtime` will be merged and published, and under what package name (the `@pmndrs/denoiser` name is not on npm; ddgi docs use it anyway).
- Check the upscaler's `s5-radiance-cascades` example for how it feeds sparse GI to `upscale()`.
- Decide where GI-1.0 temporal accumulation lives (own pass versus upscaler history), because upscaler issue #7 says doing both causes jitter conflicts.
- Confirm the postprocessing v7 WebGPU status before listing it as a dependency.

## 8. Update, 2026-11-19

The project moved to the private repo `pmndrs/gi` on 2026-11-10: one npm package `@pmndrs/gi` with `/core`, `/ddgi` and `/three` subpaths. The radiance-cache branches were pushed there on 2026-11-19. Everything below was read from fresh clones that day.

**What `pmndrs/gi` main uses today:** `@pmndrs/upscaler` 0.5.0 (viewer and example 02), and its own frozen in-repo `packages/bvh` (0.0.0, `workspace:*`) traced by the platform's `giTraceClosest`/`giTraceAny`. No denoiser, downsampler, motion-vectors or raytracer dependency.

| Library | State on 2026-11-19 | What it serves in GI-1.0 | Use |
|---|---|---|---|
| `@pmndrs/bvh` | 0.1.0 on npm (11-18). Engine-free core, `/three`, `/babylon`, `/worker`; WGSL walk `traceModuleWgsl` with `bvhTraceClosest`/`bvhTraceAny` and masked forms; wide nodes off by default | Every trace (screen probes, cache update, light-grid shadow rays) | gi's own switch from the frozen copy is bvh's "P3" step (`docs/internals/MIGRATING-FROM-DDGI.md`). A whole-gi decision, not ours; we follow when main moves |
| `@pmndrs/downsampler` | 0.1.0 on npm. `Downsampler.encode` (average/min/max/minmax/custom WGSL fold, Hi-Z, exact `r32uint` folds), `Reducer` (6 levels per tile, sum/weighted/argmin+payload, load/store hooks). Its SPEC already names our probe mask and tile sums as hand-written users | Hash-grid tile mips (sums); maybe `FilterProbeMask`'s chain | Lever, measured against the faithful hand-written passes. The probe-mask fold must be commutative and idempotent; check against Capsaicin first |
| `@pmndrs/motion-vectors` | 0.0.0, unpublished. `rgba16float`: `.xy` NDC current minus previous (jitter-free, same convention as three, upscaler, denoiser), `.z` trust 0-4, `.w` previous linear depth. `CutDetector`, `WarpingError`. No previous normals | Reprojection inputs for `ReprojectScreenProbes` and `ReprojectGI` | Lever once published: exact deformation motion, `.w` as an extra disocclusion test, cut flag to reset history. Capsaicin's plane/normal tests still need gi's own previous depth and normal copies |
| `@pmndrs/denoiser` | `realtime` merged as experimental (11-12); all packages private 0.1.0. `DiffuseDenoiser` etc. with `filter: 'svgf' \| 'recurrent'`; `FrameGuides` (depth, normal, motion, optional roughness/albedo/view z). No GI-1.0 option, no issue for one. Its RULES allow porting MIT AMD code with AMD's notice, as done for FFX | GI-1.0's image denoiser (`ReprojectGI`, `FilterGI` twice, blur mask) | Phase 4 builds it there. Open question for Dennis: an FFX-style AMD port, or a third `filter: 'gi1'` in `realtime` |
| `@pmndrs/upscaler` | 0.5.0. Core on a bare device; sole source of jitter; S5 radiance-cascades example upscales GI | Final upscale (D9) | Use now, as before; issue #7 (GI history inside the upscaler) stays a phase 5 question |
| `pmndrs/pathtracer` | `@pmndrs/raytracer` (MIT): inline `rtModuleWgsl` (`rtClosest`, `rtVisible`, intervals, `rtShade` with an external-cache hook `rtShadeIndirect`), `RtBatchTracer`, `RtIntervalTracer`; on `@pmndrs/bvh` ^0.1.0. Bench-local SHaRC-style hash cache (`bench/nrc/hash-grid.ts`, written from SHaRC's and Binder 2019's descriptions, no SDK code) and a working NRC (`packages/mlp`) | A second world cache to compare against GI-1.0's grid; a ground-truth path tracer for bias checks | Comparator and reference, not a dependency of the faithful port |

**Pathtracer's numbers worth keeping** (their RTX 3060 and M5 Pro runs, not ours): their hash grid lands within 2-4% of NRC per frame on diffuse Sponza, warms in about 10 frames against NRC's 150, and costs less; NRC costs 18 ms inference plus 5 ms training of a 62 ms 720p frame in Chrome on D3D12 (no subgroup matrices there). NRC only wins on converged glossy scenes. So NRC in the world-cache slot is low priority.

**What they want from gi:** the pathtracer's external-cache hooks wait on a "gi world cache on vs off" A/B; they place the production hash cache in `@pmndrs/gi` and expose hooks only; their world-space light grid "may share gi's hash", which overlaps GI-1.0's light grid. Their Split Radiance Cascades handoff (gi PROGRESS 2026-11-18) is a separate technique, not this track.

**Known hazards to carry:** bvh's any-hit drops a hit at exactly `tMax` (reported in the pathtracer handoff, not yet filed on pmndrs/bvh); bvh issue #3, the WGSL walk inlined into a big kernel runs 3.7x slower on Chrome D3D12/NVIDIA (a 64-bit loop counter). GI-1.0's kernels are big, so that hits us on the RTX.
