# GI-1.0 on WebGPU: implementation plan

Draft 2, 2026-11-09. Written from the research in this folder: [paper.md](paper.md), [source-map.md](source-map.md), [native-capsaicin.md](native-capsaicin.md), [related-work.md](related-work.md), [pmndrs-siblings.md](pmndrs-siblings.md) and [idtech8-performance.md](idtech8-performance.md). Nothing is built yet. Dennis's decisions are recorded at the end. The technique is called `radiance-cache`.

## What we are building

A port of AMD's two-level radiance cache as the **second technique on the `@pmndrs/ddgi` GI platform**, next to DDGI. The reference is Capsaicin v1.3, which ships GI-1.2: GI-1.0, plus GI-1.1's glossy reflections, plus a second-bounce layer in the world cache.

- **Screen cache:** one 8x8 hemi-octahedral probe per screen tile, placed on visible surfaces, filled over several frames, reused by reprojection. Detail where the camera looks, quick reaction to change.
- **World cache:** a spatial hash grid of outgoing radiance at ray hits, cells sized by camera distance, freed when unused. Probe rays end in it. A second layer of cells carries the next bounce.
- **Engine-agnostic core** in WGSL plus TypeScript orchestration, on the existing platform (scene copy, tracer, hit shading, light grid, budget, change feed). **three.js r187 is the first and only adapter** until the technique is validated, same rule as DDGI.

Why this technique: it is what DDGI is not. DDGI is a fixed probe grid that is stable but blurry and slow to react. This one is screen-driven: detail where the camera looks, fast reaction to moving objects, no volume to place. The two stack well: id Tech 8 ships exactly that stack (see Extensions).

## What we are really after (Dennis, 2026-11-19)

This is a **multi-cache GI system**, not a 1:1 Capsaicin port. GI-1.0's screen cache plus world cache is the skeleton, the same shape as id Tech 8's cascading caches and the base of radiance cascades and others. The goal is **as fast and as high quality as possible, and stable (no flicker)**. Looking a bit different from GI-1.2 is fine; parity with native is not a goal. His words: "we arent trying to be a 1:1 port of Capsaicin ... Our goal is to be as fast and high quality as possible, not have exact parity with gi1."

So the rules below still hold for building the baseline, with three changes:
- **Capsaicin's errors are fixed outright**, with a register row and the derivation; no effort goes into reproducing its bugs (the confirmed ones so far: D25's SH weight, GI-1.2's 3.33x second-bounce bias and the miss bias, the bypass index mix-up; draft issue for AMD in [capsaicin-issue-draft.md](capsaicin-issue-draft.md)).
- **Results are judged by speed, accuracy against the path tracer, and temporal stability**, not by matching native pixels. Native Capsaicin stays the speed yardstick and a sanity check.
- **Our own experiments are first-class**, as world-cache alternatives and levers, not afterthoughts: the pathtracer's hash cache and its working NRC ([pmndrs-siblings.md §8](pmndrs-siblings.md#8-update-2026-11-19)), id Tech 8's budgeted cell shading, Brixelizer, DDGI as a cache level.

## How we work: the baseline first

1. **Build the baseline as AMD built it.** Capsaicin's default configuration, pass by pass, with its algorithms, ray counts, rates and cache sizes. That is the **baseline**. It may be slow on our tracer; that is information, not a reason to change it up front.
2. **Every difference from Capsaicin goes in the divergence register below**: what we do instead, why (platform, license, performance, quality), and the way back. "Way back" includes what others have done: browser vendors, other WebGPU projects and engines may already have a workaround we haven't looked at.
3. **Performance changes are levers, not rewrites.** Each lever is a switch next to the faithful path, measured against the baseline on the same scene and camera, and adopted only with a written reason. A slow faithful path is kept and marked for attention, so we can come back and fix whatever made it slow (our BVH, a missing WebGPU feature) instead of losing it.
4. **Priority versions are welcome as extra options.** A cheaper or prioritized variant of a pass (for example, shading cache cells on a budget instead of every hit) can be built alongside, behind a switch. It does not replace the baseline.

## What we port, and from where

| Source | License | Use |
|---|---|---|
| Capsaicin v1.3 (`914b915`), `src/core/src/render_techniques/gi1/` | MIT (AMD) | The baseline. Port pass by pass, keep AMD notices in ported files. About 5,000 useful HLSL lines. |
| Capsaicin SSGI technique (near-field GI, bent-normal occlusion) | MIT (AMD) | Part of the baseline: GI-1 reads its output. About 350 lines. |
| Capsaicin blue-noise tables (`blue_noise_sampler_samples.h`) | No license header | Do not copy. Regenerate (divergence D7). |
| GI-1.0 paper (arXiv 2310.19855), GI-1.1 paper, GI-1.2 page | Papers | Algorithms and rationale. |
| FidelityFX Brixelizer GI | MIT (verify headers) | Ported separately in another workstream; plugs in as a world-cache and tracing backend (see "World cache as a slot"). |
| NVIDIA SHaRC / RTXGI / NRC | NVIDIA RTX SDK license | Never port. Ideas from public docs only. |
| Lumen, id Tech 8, EA surfels | Proprietary | Slides only. |

The baseline is **Capsaicin's default configuration**:

- In: screen probes (one probe per 16x16 tile per frame, `QuarterSpp`, at full render resolution), reprojection, adaptive patching of empty tiles, luminance-CDF ray guiding, the off-screen LRU probe cache, biased-hysteresis blending, the probe filter, SH projection and interpolation, the hash-grid cache with its 4-level tile MIPs, decay and **per-hit population** (one sampled light and shadow ray per hit), multi-bounce cells (GI-1.2), SSGI near-field and bent-normal occlusion, the image-space GI denoiser and the final resolve.
- In, once the platform has the material data: glossy reflections (GI-1.1). They are on by default in Capsaicin; their absence is divergence D8.
- Out (off by default in Capsaicin, so not part of its default behaviour): the DXR 1.0 path, world-space ReSTIR, temporal radiance feedback, debug-stat kernels.

## The native yardstick

Capsaicin GI-1.2 on **our own scenes**, from our manifest cameras (field of view matched exactly), 1280x720, RTX 3060 Laptop with hardware RT, warmed averages over frames 101 to 300 ([native-capsaicin.md](native-capsaicin.md), "Our scenes in Capsaicin"). Bistro uses 1k textures: at 2048 px it doesn't fit the 6 GB card in Capsaicin either.

| Group (ms) | Sponza + curtains | Bistro interior | Bistro exterior | Two-room |
|---|---|---|---|---|
| Probe tracing and cell population | 4.75 | 7.81 | 7.56 | 1.31 |
| Cache update and resolve | 0.74 | 0.67 | 0.80 | 0.65 |
| Screen probes | 1.38 | 1.18 | 1.31 | 1.51 |
| Denoise and resolve | 1.08 | 0.71 | 0.95 | 0.56 |
| **Diffuse subtotal** | **7.95** | **10.38** | **10.62** | **4.03** |
| Reflections (GI-1.1) | 0.69 | 0.31 | 0.44 | 0.17 |
| Reference path tracer, 1 spp | 23.8 | 34.8 | 53.0 | 7.4 |

Run-to-run noise is about 15% (laptop GPU, not clock-locked, also driving the display).

AMD's paper figures (1.9 to 3.4 ms at 1080p) were on an RTX 3080 desktop. On our laptop, with hardware RT, diffuse GI-1.2 is 4 to 10.6 ms at 720p, and tracing plus hit shading is the largest part on the real scenes. So:

- **The comparison is per pass, same machine, same scene and camera.** Our baseline's time for each Capsaicin pass sits next to Capsaicin's. Passes far above native become the attention list.
- **No frame-time target is fixed before the baseline exists.** Once it does, the target is set from the per-pass table and Dennis's call.

**Image comparisons: the path tracer is the ground truth, not Capsaicin's GI image.** Under punctual lights (our sun and lamps), Capsaicin's GI-1.2 renderer comes out far darker than its own path tracer: mean linear luminance 0.00085 against 0.016 on Sponza, 0.029 against 0.197 on Bistro interior; two-room is close (0.83 against 1.08). Its GI pass lights the screen pass from emissives and the environment only; punctual light reaches it through the cache's light sampling, and its renderer assumes a separate direct-light technique. Our port is the same: indirect only, with three drawing direct light. So:
- Compare our port to Capsaicin's GI image for *behaviour* (stability, reaction, artifacts) and per-pass *cost*.
- Compare *accuracy* to the path tracer (Capsaicin's and our Cycles HD references).
- Phase 5 confirms from Capsaicin's source why its punctual-light result is so dark, before we call any difference ours. Our procedural sky can't be reproduced in Capsaicin; sky-lit comparisons need a matching HDR environment map.

What to expect, so the attention list doesn't surprise anyone: our compute tracer runs about 105 to 120 M closest-hit rays/s on Sponza on this GPU, and 48 to 60 M shadow rays/s in real dispatches (`webgpu-ddgi/docs/NOTES.md:364-383`). Capsaicin's defaults at 1080p trace about 522k probe rays, about 160k multi-bounce rays and about one shadow ray per hit each frame. The tracing passes will likely be well over native in the first baseline. Our BVH work (wide nodes, H-PLOC, any-hit improvements) is expected to narrow that, and it is the first thing to chase there.

## Divergence register

Every way our port differs from Capsaicin's default behaviour. Kind: **P** platform (WebGPU or our platform lacks something), **L** license, **Q** quality-affecting, **S** speed lever (only after the baseline). The register lives in this file until the technique has a home in `webgpu-ddgi`, then moves to its design doc.

| ID | Capsaicin | Ours | Kind | Why | Way back / others' workarounds |
|---|---|---|---|---|---|
| D1 | DXR 1.1 inline ray queries; hardware traversal; alpha test in the loop | Platform compute BVH (`giTraceClosest`, `giTraceAny`), cutout from the material | P | WebGPU has no ray tracing | Our BVH work (CWBVH, H-PLOC, any-hit fixes); a Brixelizer SDF backend (in progress elsewhere); track any WebGPU ray-query proposal; compare against three-mesh-bvh's WebGPU traversal and Webgiya |
| D2 | Bindless material textures sampled at every hit | Texture-average albedo through `giSurfaceAt` (what DDGI uses) | P, Q | WebGPU has no bindless textures; the platform's hit record is per material | Texture arrays or an atlas with real UVs at hits; compressed textures (GPUTex, already planned for Bistro in M3) to make that affordable; track the WebGPU bindless proposal |
| D3 | Mesh-shader visibility buffer (instance, primitive, barycentrics) | Host-provided G-buffer from three's MRT pass (depth, normal, velocity, albedo, instance id) | P | WebGPU has no mesh shaders; three already renders these | three's own VisibilityBuffer pipeline; a platform raster pass over the scene copy if a host can't provide them |
| D4 | `LightSamplerGridStream` (64 streamed reservoirs per cell, wave intrinsics in the build) | Platform light grid (16 picks per cell weighed at the hit, N shadow rays per hit) | P, Q | The platform rule "hit shading is shared"; avoids a second light system | Port the stream sampler as a platform option (subgroups for the build) if the light-grid image differs measurably from Capsaicin's |
| D5 | Read-write `RGBA16F` textures | `texture-formats-tier2` read-write storage textures | P | Core WebGPU allows read-write only on 32-bit single-channel formats | Declared fallback: ping-pong (sampled read plus write-only storage) |
| D6 | Unbounded buffer arrays (about 30 hash-cache buffers) | A few large packed storage buffers with fixed offsets | P | Per-stage storage-binding count limit | None needed; same data |
| D7 | Shipped blue-noise and Sobol tables | Regenerated tables (or the platform's blue noise) | L | No license header on the tables | Ask AMD for terms if the regenerated noise behaves differently |
| D8 | GGX BRDF at hits from a packed material; glossy reflections on by default | Lambert at hits; no reflections until the platform has roughness and metalness | P, Q | Platform materials are diffuse-only (`contract/scene.ts:146-165`) | Add roughness and metalness to the material descriptor (also needed by the planned RT reflections) |
| D9 | FSR 2/3 in the frame chain | `@pmndrs/upscaler` (FSR 2/3-style) | P | Our sibling library | Equivalent; note upscaler issue #7 on jitter and GI history |
| D10 | D3D12 per-pass timestamps | WebGPU `timestamp-query` per pass | P | Measurement only | Platform timing already does this |
| D11 | Half-precision packing via `f32tof16` | `pack2x16float` / `unpack2x16float`; `shader-f16` where it helps | P | WGSL equivalents | None needed |
| D12 | (Capsaicin itself runs one queue; id Tech 8 uses async compute) | One WebGPU queue | P | WebGPU has no async compute | Track WebGPU multi-queue proposals; order passes to keep the GPU busy |

Speed levers (kind S) get rows here only when one is adopted. Until then they live in the levers list below.

## What the platform already gives us, and what it lacks

From a read of `webgpu-ddgi` on branch `m2`:

**Already there:**
- Inline tracing from our own kernels: `giTraceClosest` and `giTraceAny`, plus batched rays (`platform/contract/queries.ts:22-92`). Capsaicin uses inline ray queries, so the port is direct.
- Shared hit shading: `giSurfaceAt` and `giDirectIrradiance` with shadow rays, the sky, emission (`contract/shade.ts:24-69`).
- A light grid close to Capsaicin's sampler (D4).
- A `BounceSource` slot so techniques stack, the change feed (`DirtyRegion.moved`), a GPU-refit top-level BVH, the work budget (Budget v2) and timing.
- Velocity in three r187 MRT and `@pmndrs/upscaler` wired in the viewer.
- `requiredLimits(adapter)` already asks for the adapter's maximum buffer and binding sizes (about 2 GiB on both laptops).

**Gaps (tasks in phase 1):**
1. **Screen inputs in compute.** `FrameScreen` has depth, normal, optional velocity and instance id, but no albedo, and nothing in `ddgi-three` fills it yet (`docs/api/platform.md:130`, gap 3).
2. **Diffuse-only materials** (D8).
3. **No platform/technique lint rule.** SPEC says the boundary is lint-enforced; it isn't.
4. **GPU utilities:** prefix scan, stream compaction and indirect-dispatch argument writing as reusable platform pieces (the BVH package has scans; share them).

### Screen inputs: the flexible route

The technique needs the per-pixel data three already renders (depth, normal, motion, albedo). The platform takes them through its `FrameScreen` contract as textures the **host** provides. The three adapter fills that contract from its existing MRT scene pass. That keeps the core engine-free, lets any host hand over its own buffers, and lets a host without them use a platform raster pass over the scene copy later (D3's way back). If three r187 won't expose its render targets as WebGPU textures without reaching into its backend, the adapter takes the smallest version-pinned step needed and records it in the register.

## World cache as a slot

The baseline world cache is Capsaicin's hash grid. Behind the same interface ("give me radiance leaving this hit point toward this ray", plus "these cells need updating"), other world caches can plug in:

- **The hash grid** (baseline).
- **Brixelizer GI's brick cache**, from the separate Brixelizer port. Brixelizer also brings SDF tracing, a second query backend next to the BVH.
- **A Lumen-style surface cache** (cards with material and lighting).
- **The DDGI volume** as the last level, id Tech 8 style.

Design the interface in phase 3 with the Brixelizer port's needs in view, so plugging it in later doesn't change the screen cache.

## World-cache memory

Capsaicin's defaults allocate about 1 GiB for the hash cache (2^14 buckets x 16 tiles x 85 cells, two value arrays and two per-cell scratch arrays; `source-map.md` §4.2). We allocate that, under the adapter-maximum limits the platform already requests. A device that can't fit it throws `GILimitError` naming the limit. Capacity is a setting chosen by measured bucket overflow and quality, never by WebGPU's default 128 MiB binding. Sizing the update scratch per updated tile instead of per cell is a lever (it frees several hundred MB), measured like any other. The technique reports its GPU memory in stats.

## Where the code goes

- **Repository:** `pmndrs/gi` (moved from `webgpu-ddgi` on 2026-11-10), branch `radiance-cache`. This research moved here from the `ddgi` research branch on 2026-11-19; the source PDFs stay local (see sources/SOURCES.md).
- **Technique:** `packages/ddgi-core/src/techniques/radiance-cache/`, beside `techniques/ddgi/`, importing only `platform/contract`. Published as the import path `@pmndrs/ddgi/radiance-cache` until the package-split decision (DECISIONS: revisit when a second technique ships).
- **Adapter:** screen-input wiring and a composite node in `packages/ddgi-three`, following the existing additive composite (`albedo/π × E` before tone mapping).
- **Denoisers and upscaling:** pushed to `pmndrs/denoiser` and `pmndrs/upscaler` (decision 3). The technique uses them through their public APIs.

## Phases

**2026-11-09 update (Dennis):** skip timing for now and build as far as possible; stop only for decisions that block the whole project. Phases 1–4 run on correctness gates only, and every timed check is marked deferred. Phase 5's per-pass comparison picks the timing back up.

One phase active at a time, each with a checkable done-when. Implementers and reviewers on Opus (medium or higher); Sonnet for test runs and measurement, Haiku for lookups. GPU timing under the bench lock, never alongside a native run. Work happens alongside M2's close-out and M3; platform changes go in as small, separate PRs so the M3/M4 agents can use them too.

### Phase 0: research and native reference

- Done: the paper, GI-1.1 and Brixelizer GI sources; the Capsaicin source map; related work; the sibling audit; the id Tech 8 performance notes; this plan.
- Done: native Capsaicin v1.3 built (1 min 44 s), launched on the RTX 3060 Laptop and captured. GI-1.2 and the reference path tracer render the same Sponza view to linear EXR with per-pass timing CSVs through `tools/reference/capsaicin/run-capsaicin.ps1`.
- Done: Sponza-with-curtains, Bistro interior, Bistro exterior and two-room load in Capsaicin from our manifest cameras with matched FOV, with GI-1.2 and path-tracer captures and warmed per-pass timings ([native-capsaicin.md](native-capsaicin.md)). Scene builder: `tools/reference/capsaicin/make_scenes.py` (merges Sponza and curtains, re-encodes Bistro's WebP textures, injects our cameras and lights).
- Open, carried into phase 5: a matching HDR sky; the second camera of each scene; why Capsaicin's GI image is so dark under punctual lights; our light units against the scene builder's conversion (assumed lux and candela, not yet checked against the WebGPU code).
- **Phase 0 is complete.**

### Phase 1: platform groundwork

1. **Screen inputs** through `FrameScreen` from three's MRT pass (see "the flexible route"), with albedo added. Done when it is filled every frame on Sponza, a debug view shows every channel, and its cost is measured.
2. **Lint rule** keeping techniques on `platform/contract` only.
3. **GPU utilities:** scan, compaction, indirect-args writer, with tests.
4. **Feature checks** in three r187 and Chrome on both GPUs: indirect compute dispatch, `texture-formats-tier2`, subgroups, `shader-f16`, `timestamp-query`. Each missing feature gets its declared fallback and a register row.
5. **Ray cost table:** closest-hit and any-hit Mrays/s for Capsaicin-shaped ray sets (64 rays per screen-placed probe; multi-bounce rays; one shadow ray per hit) on Sponza-with-curtains and Bistro. This predicts the baseline's tracing cost; it doesn't change the baseline.

### Phase 2: screen probes, first image

Spawn (16x16 tiles, Halton jitter), trace 64 rays per probe, light hits directly (Capsaicin's cache-bypass path, used for short rays and emitters), blend, project to SH, interpolate, composite through three.

- Done when: a single-bounce image on Cornell, two-room and Sponza; probe atlas, mask and SH in the viewer's debug views; constant-sky Lambert check (E = πL within 1%) passes on a GPU test.

### Phase 3: world cache, faithful

Hash grid exactly as Capsaicin: descriptor and hashes in one shared WGSL function, CAS insert into 16-slot buckets, 4-level tile MIPs, decay and purge, filtered lookup, per-hit population with one sampled light and shadow ray, then GI-1.2 multi-bounce cells. The world-cache interface is designed here (see "World cache as a slot").

- Done when: Capsaicin's default capacity runs under adapter-maximum limits with memory in stats; bucket overflow is measured; multi-bounce visibly fills the two-room's closed room compared to phase 2; GPU tests cover insert, find, collision, overflow and decay.

### Phase 4: temporal quality, faithful

Reprojection with the plane and normal tests, adaptive patching of empty tiles, luminance-CDF ray guiding, the off-screen LRU probe cache, the probe filter, SSGI near-field and bent-normal occlusion, and the image-space GI denoiser (built as a `pmndrs/denoiser` contribution, used through its API).

- Done when: Dennis judges the image steady by eye in the viewer on Cornell and Sponza (the M1 lesson: flicker numbers alone aren't enough), and a camera fly-through shows no probe popping.

### Phase 5: baseline comparison and attention list

- Same scene and camera: Capsaicin GI-1.2, Capsaicin's path tracer, our baseline, our DDGI, the Cycles HD references.
- Per-pass table: ours against Capsaicin's, GPU loaded. Each pass over native gets a cause (tracer, missing feature, our code) and a row in the attention list.
- Accuracy against the path tracer reported per scene, on the scenes people use (Sponza with curtains, Bistro).
- Done when: the table, the attention list and the image comparison are in, and Dennis sets the frame-time target from them.

### Phase 6: levers and dynamic scenes

Levers are tried one at a time against the baseline (list below), in the order the attention list says matters. In parallel, dynamic behaviour, the priority since 2026-11-06:

- Moving objects: on `DirtyRegion.moved`, invalidate or fast-decay hash tiles overlapping the old and new bounds; the screen cache already re-traces where reprojection fails.
- Lights and sun: re-populate affected cells first.
- Done when: the M2/M3 scenarios (dragged cabinet, sun sweep, swaying tree) settle within SPEC's targets (visible next frame, within 5% in 0.5 s).

## Levers to try after the baseline

Each is a switch beside the faithful path, measured against it, adopted with a reason in the register.

id Tech 8 (Sousa, SIGGRAPH 2025) is the main source here; details, slide citations and trade-offs are in [idtech8-performance.md](idtech8-performance.md). Its frame costs about 1.7 to 2.1 ms on consoles and an RTX 4080 at 4K.

1. **Shade cells, not hits** (id Tech 8). Probe rays store a compact hit record and only look up a cell; lighting runs on a fixed budget of cells per frame (id shades about 20k), by priority (new, stale, near a change). Targets the probe-tracing/cell-population group, the largest on real scenes (4.75 to 7.81 ms native). Trade: lighting in the cache lags by a few frames.
2. **DDGI supplies the second bounce** (id Tech 8 reads last frame's irradiance volumes when shading cache entries). Replaces GI-1.2's multi-bounce rays (about 160k of 680k rays at 1080p) with a lookup into our existing DDGI field. Trade: bounce detail at DDGI resolution.
3. **Half- and quarter-resolution tiers** for probes and the gather, with a 4-tap bilateral upsample and temporal accumulation (id Tech 8), or `@pmndrs/upscaler`.
4. **Denoise in workgroup memory with f16 at gather resolution**, history capped by a frame counter (id Tech 8). Mostly free.
5. **Compact formats** (id Tech 8): 32-bit shared-exponent history (RGB9E5-style; check WebGPU storage support), 16-byte hit records, SH in three RGBA16F. Free.
6. **Interleaved refresh of distant cache regions**, and benchmark over a full update cycle (a mistake id says it made in its own benchmarks).
7. **Short probe rays that end in the world cache** (Lumen). Our tracer gains little from short rays (NOTES:383), so measure first.
8. **Scratch per updated tile** instead of per cell (memory).
9. **Brixelizer's SDF tracing** as a query backend for long rays, from the separate port.
10. **Priority versions of passes**, for example key-probe-style ordering of screen-probe updates.

Not available to us: async compute, which id credits with 20 to 25%; WebGPU has a single queue. That goes in the register (D12) when a pass would use it.

## Extensions after the baseline (ranked)

1. **DDGI as the third cache level** (id Tech 8's "cascading caches", SIGGRAPH 2025): screen cache, then world cache, then the DDGI volume as last fallback and multi-bounce source. The platform's `BounceSource` slot fits this.
2. **Glossy reflections (GI-1.1)**, once materials carry roughness and metalness (D8).
3. **Brixelizer and surface-cache world caches** through the world-cache slot.
4. **Object-attached cache entries for moving objects** (EA surfels; Webgiya, MIT, is a working WebGPU surfel-GI precedent).
5. **Lumen-style improvements:** product importance sampling, a GPU priority queue for cache updates.
6. **Split Radiance Cascades** (arXiv 2607.20384) as a world-cache experiment. Four months old, no code.

## Siblings: what goes back

- **`@pmndrs/upscaler`** (0.5.0, active): used as-is for upscaling. Contribute a GI-history path for its issue #7 (a separate temporal pass in front of the upscaler fights its jitter) if our phase 5 runs into it.
- **`@pmndrs/denoiser`**: its diffuse filters exist only on the unmerged `feat/rt-denoiser-cleanroom` branch. Contributions from this work:
  1. GI-1.0's image-space GI denoiser (disocclusion blur mask, history-length-driven spatial radius), built there in phase 4;
  2. hit distance and cache confidence as filter guides;
  3. the "coverage" input mode from denoiser issue #23, which matches a cache fallback;
  4. a bilateral upsample node for half-resolution GI.
- Probe-space filtering and hysteresis stay in the technique: they are part of the cache, not a denoiser.
- **Update 2026-11-19** ([pmndrs-siblings.md §8](pmndrs-siblings.md#8-update-2026-11-19)): the denoiser's `realtime` package is now merged (experimental, private), so phase 4 builds GI-1.0's denoiser there; whether as an AMD-style port or a `filter: 'gi1'` option is Dennis's call. `@pmndrs/downsampler` 0.1.0 and `@pmndrs/motion-vectors` (unpublished) are levers against the hand-written faithful passes. `@pmndrs/bvh` 0.1.0 replaces gi's frozen copy when gi main moves (bvh's P3). The pathtracer's hash cache is a second world cache to A/B against GI-1.0's grid, and its path tracer a bias reference.
- **Textures:** GPUTex for compressed textures where real hit textures (D2) or big scenes need the memory.

## Risks

| Risk | Mitigation |
|---|---|
| The faithful baseline is far slower than native in tracing | Expected and measured, not avoided: phase 5's attention list, BVH work, then levers |
| Getting three's buffers into compute | Phase 1 item 1 first; the smallest pinned step if three needs one, recorded in the register |
| Flicker the numbers miss | Dennis's by-eye check is a done-when item in phase 4 |
| Temporal accumulation fighting the upscaler's jitter | Decide placement in phase 5, with upscaler issue #7 |
| World cells stale after objects move | Dirty-region invalidation in phase 6; object-attached entries later |
| Clashing with M2/M3 work in the same repo | Separate branch; platform changes as small PRs that M3/M4 agents can use |
| Denoiser branch unmerged and Apple-only so far | Our contribution includes an RTX 3060 run; Dennis maintains both repos |

## Decisions

Since then (2026-11-10, gi DECISIONS): the project is `pmndrs/gi`, one package `@pmndrs/gi` with subpaths; the branch and folder names below stand until radiance-cache merges and the workspace is renamed to `gi`. GI-1.0's denoiser goes into `pmndrs/denoiser` as an option.

Answered by Dennis, 2026-11-09:

1. **Home and name:** `webgpu-ddgi`, beside DDGI, named **`radiance-cache`**: folder `packages/ddgi-core/src/techniques/radiance-cache/`, branch `radiance-cache` off `m2`, import path `@pmndrs/ddgi/radiance-cache`. Docs say "based on AMD GI-1.0/1.2".
2. **Screen inputs:** whatever keeps us most flexible. Taken as: the host provides the buffers through the platform contract; the three adapter fills it from its MRT pass; a platform raster pass is the fallback for hosts that can't.
3. **Denoisers and upscaling:** push to the sibling libraries.
4. **Timing:** start now, alongside M2's close-out. Most of M2 is DDGI-specific, and the platform work here (screen inputs, utilities, lint) can help the M3/M4 agents.

Other names considered: `screen-cache`, `cached-gi`, `two-level`, `screen-probes`. `radiance-cache` won because it is the paper's own term and covers the screen cache plus any world-cache backend (hash grid, Brixelizer, surface cache).
