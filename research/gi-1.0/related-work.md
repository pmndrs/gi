# GI-1.0 related work

Date: 2026-11-09. Scope: how AMD GI-1.0 (arXiv 2310.19855) relates to Lumen, id Tech 8, radiance cascades, SHaRC/NRC, kajiya, surfel GI, AMD's later work, ReSTIR, Godot, and browser GI. The GI-1.0 paper itself and the pmndrs libraries are covered elsewhere.

Verification key: **[read]** = I read the primary document text this session. **[snippet]** = seen only in search results or a fetch summary. **[unverified]** = not confirmed; check before relying on it.

## 0. GI-1.0 in brief (for the comparisons)

Facts below come from a summary of the paper's HTML version [snippet]. The paper is the authority.

- Screen cache: one 8x8 octahedral probe per 8x8 screen tile, at a Halton-jittered pixel. Probes are reprojected from the previous frame. Rays are reassigned from reprojected tiles to empty tiles to hold the ray count constant, guided by a luminance CDF. Spatial filtering is a separable 7x7 sparse blur. Probes are projected to SH before per-pixel interpolation. Evicted probes go into an LRU side cache.
- World cache: a spatial hash grid (Binder 2019) with distance-adaptive quantization. A cell descriptor holds quantized position, direction and a ray-length bit, and is hashed twice (bucket plus fingerprint). Collisions use linear probing. Each cell has a decay counter and is freed when it reaches zero. Cells form 8x8 tiles with mips. Direct light at cache hits uses ReSTIR-style world-space reservoir reuse with one shadow ray. Multi-bounce comes from temporal feedback.
- Tracing: DXR 1.1 hardware traversal. A distance-field hybrid was tried and not made default.
- Costs reported at 1080p, 0.25 spp: roughly 1.9 to 3.4 ms on RX 6900 XT / RTX 3080.
- Follow-ups: GI-1.1 adds glossy reflections (SIGGRAPH Asia 2023). GI-1.2 adds a second-bounce cell layer in the hash grid (posted 2025-11-20).

## 1. Unreal Lumen

**Core idea.** Lumen traces rays with a hybrid pipeline: screen trace first, then software tracing (mesh SDFs plus a global SDF with 4 clipmaps) or hardware RT, then skylight on miss. Hit points are lit from the Surface Cache, a card-based atlas of material and lighting, so there is no per-ray material evaluation. Diffuse final gather is Screen Space Radiance Caching (SSRC) backed by a World Space Radiance Cache. Later multi-bounce comes from surface-cache feedback. [read: SIGGRAPH 2022 slides]

**Details from the slides [read].**
- Screen probes sit at 1/16 resolution per axis (one per 16x16 pixels, uniform with adaptive subdivision where geometry is detailed). Probes are 8x8 octahedral. Placement is jittered per frame.
- Filtering happens in probe space: a 3x3 probe kernel is said to equal a 48x48 screen-space kernel.
- Rays are importance sampled using last frame's reprojected screen cache as the lighting estimate, falling back to the world cache. Product importance sampling is done with a threadgroup per probe.
- The world radiance cache handles lighting beyond about 2 m, with 32x32 octahedral probes (more directional, less spatial resolution). It is a clipmap-distributed probe grid with persistent allocations. A GPU priority queue picks a fixed number of probes to re-trace per frame.
- Screen probe rays are shortened. On a miss they interpolate from the world probe.
- Final steps: interpolate and integrate probes to pixels, temporal filter, and a contact AO to restore small-scale shadowing.

**Side by side with GI-1.0.**

| Aspect | Lumen | GI-1.0 |
|---|---|---|
| Screen cache | SSRC, 8x8 octahedral probes, 1 per 16x16 px | 8x8 octahedral probes, 1 per 8x8 tile, jittered |
| Probe temporal reuse | Reproject, importance sample from last frame | Reproject with cell-size and normal test, ray reassignment, luminance CDF, LRU side cache |
| World cache structure | Clipmap probe grid, 32x32 octahedral probes, persistent | Sparse spatial hash grid of cells keyed by position, direction and a ray-length bit, fingerprinted |
| World cache role | Distant light (beyond about 2 m); screen rays are shortened and finish in it | Second level; screen-cache rays end in it (the paper says it needs no extra world-space structure beyond this hash cache) |
| Hit shading | Surface Cache atlas lookup | Radiance stored in hash cells, lit by ReSTIR-style direct light reuse |
| Multi-bounce | Surface-cache feedback | Temporal feedback (1.0/1.1); second-bounce cell layer (1.2) |
| Tracing | Screen trace, then SDF software tracing or HWRT | HWRT (DXR 1.1) |
| Denoise | Probe-space filter plus pixel temporal filter plus contact AO | Probe filter plus adaptive spatial filter keyed to history length, plus a short-range horizon-based GI |
| Scene representation needed | Mesh SDFs, global SDF, cards, voxel lighting | The BVH only |

**What we could borrow.**
- Probe-space filtering and product importance sampling for the screen cache.
- The GPU priority queue for a fixed per-frame cache update budget.
- The probe-ray shortening plus world-cache handoff, as a cleaner alternative to GI-1.0's cell lookups.
- The hybrid screen-trace-first ray pipeline.
- The global SDF clipmap idea is relevant if our compute BVH proves too slow for long rays. It is a heavier build (SDF generation, streaming).

**Needs hardware RT.** No. Lumen has a software mode, which is the relevant precedent for us.

**Source and license.** The code is in Unreal Engine source. It is source-available under the Epic EULA, not open source. We can read the slides and the talk, not copy the code.

**Links.**
- SIGGRAPH 2022 slides: https://advances.realtimerendering.com/s2022/SIGGRAPH2022-Advances-Lumen-Wright%20et%20al.pdf
- Epic tech blog: https://www.unrealengine.com/en-US/tech-blog/unreal-engine-5-goes-all-in-on-dynamic-global-illumination-with-lumen
- Wright 2021, "Radiance Caching for Real-Time Global Illumination" (SIGGRAPH 2021 Advances course): slide deck not located [unverified]; find it under https://advances.realtimerendering.com/ s2021.

**Unverified.** UE 5.x updates (hardware RT Lumen changes after 5.0, Lumen scene improvements) are not covered here; I read only the 2022 deck.

## 2. id Tech 8 / DOOM: The Dark Ages

**Primary source [read]:** Tiago Sousa, "idTech 8 Global Illumination" (title slide "FAST AS HELL"), SIGGRAPH 2025 Advances in Real-Time Rendering in Games. Also presented at CEDEC 2025 (slide notes say so). Used in Indiana Jones and the Great Circle (early iteration) and DOOM: The Dark Ages (latest).

**What "cascading caches" means concretely.** The press phrase is not the talk's wording. The slides describe these structures:

1. A **cascaded light grid** (16^3 cells x 8 cascades) for lighting RT-world hits: lights, reflection probes and decals binned in world space. This is a shading helper, not a GI cache.
2. **Cascaded irradiance volumes**: 16^3 x 6 cascades (12^3 on lower specs), plus 10^3 hand-placed local volumes. Probes trace only visibility rays (64/32/16 per probe per frame by platform), and one cascade plus one local volume is updated per frame. Stored DDGI-style as octahedral irradiance plus visibility in an atlas (2560 x 1585, RGB9E5, 64 MB).
3. A **world radiance cache**: a 1D spatial hash (Gautron 2020), 25 cm cells with LOD by camera distance, about 14 MB. Each probe ray hit marks a hash cell for update. About 20k entries are shaded per frame and reused across frames. Shading reads the previous frame's irradiance volumes for multiple bounces.
4. A **screen-space cache**, used first in final gather.

Frame order: update light grid, sample world visibility, shade hashed hits into the world radiance cache, update irradiance volumes from it, final gather, denoise, upscale. Transparencies use a froxel irradiance volume.

**Final gather [read].** One ray per pixel (half or quarter resolution by quality), cosine weighted with blue noise. Zero shading: each ray looks up caches in priority order: (1) screen space cache if the hit is inside the frustum and unoccluded, (2) world radiance cache if the cell is valid, (3) irradiance volumes as the last fallback. Output is 2-band SH per pixel (3 x RGBA16F). Denoise is a separable bilateral Gaussian (normal, depth, hit distance weights), then a 4-tap Poisson bilateral upscale plus a temporal filter. Directional occlusion is added for high-frequency detail.

**Cost [read].** Slide 31 gives roughly 1.4 to 2.1 ms total on consoles at the shipped hotspot (the table is garbled in text extraction; use the PDF).

**Mapping to GI-1.0.**
- "Multiple cascading caches" = a fall-through hierarchy of screen cache, hashed world radiance cache, then cascaded DDGI-style irradiance volumes. GI-1.0 has the first two. id adds the volume cascade as a last resort and as the multi-bounce source.
- id cites DDGI, surfels (Halén 2021), SSRC (Wright 2021) and ReSTIR GI (Ouyang 2021) as the research it built on. GI-1.0 is a cousin from the same lineage.
- Differences: id traces visibility from volume probes, not from screen probes, and shades the hash entries it finds. Its screen cache is applied inside final gather rather than being a probe atlas with 8x8 octahedral probes (the slides do not give the screen cache details [unverified]).

**What we could borrow.**
- **We already have DDGI.** A fall-through (screen cache, then hash cache, then our DDGI volume) is a natural extension: DDGI as the "last cache" and as the multi-bounce source. This is the highest-value idea in this document for our plan.
- Visibility-only probe tracing with shading deferred to the hash cache.
- Output as per-pixel 2-band SH to keep normal-map detail.
- Indirect dispatch of active hash cells (atomic append list, then indirect dispatch).

**Needs hardware RT.** Yes in shipping form (DXR on consoles; the slides call RT "a visibility tool"). The structure is not RT-specific. We would substitute our compute BVH.

**Source and license.** None. Proprietary engine. Slides only.

**Links.**
- Slides: https://www.advances.realtimerendering.com/s2025/content/SOUSA_SIGGRAPH_2025_Final.pdf
- Prior art cited there: Gautron 2020 spatial hashing (https://youtu.be/oza36AqcLW8), Vahl 2025 "Pushing 60hz" (Indiana Jones) [unverified link].

## 3. Radiance Cascades (Sannikov)

**Core idea.** Store radiance in a hierarchy of probe sets (cascades). Each cascade covers a distance interval along its rays. Moving to a coarser cascade, probe spacing grows (fewer probes) while angular resolution grows (more rays), and the intervals are merged from far to near. Cost is independent of scene complexity in the 2D case. [snippet: 80.lv article]

**Variants and status.**
- Origin: Exilecon 2023 talk by Alexander Sannikov (Grinding Gear Games). Shipped in Path of Exile 2 as a screen-space effect [snippet: a Chalmers thesis says so; I did not verify with GGG].
- Osborne and Sannikov 2024, astrophysics radiative transfer formulation: https://arxiv.org/abs/2408.14425 [snippet].
- Freeman, Sannikov, Margel 2025, "Holographic Radiance Cascades for 2D Global Illumination": arXiv 2505.02041 [snippet; I could not open it].
- **Split Radiance Cascades** (Freeman, Sannikov, Pitt-Francis), arXiv 2607.20384, submitted 2026-07-22, revised 2026-10-11, license CC BY-SA 4.0. Extends RC to real-time 3D diffuse GI with world-space probes in a **sparse hashmap**, and "ray splitting": rays traced from visible surfaces assign radiance to cascade intervals by hit distance. Reports single-frame and temporally accumulated modes. [read abstract page only] Code link not listed. Hardware RT requirement not stated.
- Screen-space RC in 3D games: Chalmers thesis (Visibility Bitmask GI plus RC), https://odr.chalmers.se/items/9e42007a-bb60-440f-ba28-04410822758f/full [snippet]. Another Chalmers thesis, "Space Probes with World-Space Intervals" (SPWI): https://odr.chalmers.se/bitstreams/256da4eb-990c-4e51-aae1-c7878609e314/download [snippet].
- Godot: Juan Linietsky is reported to say RC is weak for large open worlds as a world-space effect [snippet, secondhand].

**Mapping to GI-1.0.** RC is an alternative way to organize both caches. Split RC's "sparse hash of world probes fed by rays from visible surfaces" is conceptually close to GI-1.0's world cache plus screen-ray feeding. Interval merging replaces GI-1.0's temporal accumulation for some of the noise reduction, which would help our low ray budget.

**What we could borrow.** Hit-distance-based interval assignment (ray splitting) so screen rays also feed the world-space cache without extra tracing; cascade merge as a denoiser for the world cache. This is research-grade and the paper is only four months old. Treat as an experiment, not a roadmap commitment.

**Needs hardware RT.** No in principle. Ray-intervals can be traced by any tracer, including our BVH.

**Source and license.** Sannikov's reference implementation is not clearly public [unverified]. Community 2D and 3D implementations exist with mixed licenses (see section 10). The paper text is CC BY-SA 4.0, which does not license code.

## 4. NVIDIA SHaRC and Neural Radiance Cache (NRC)

**SHaRC (Spatially Hashed Radiance Cache).**
- **Core idea.** A world-space hash grid caching radiance, designed for path tracing: rays that hit a surface terminate early by querying the cache instead of continuing. Updated per frame by a separate sparse set of path-tracing rays that write into the cache. Resolves to a radiance value at any hit point. [read: repo description; algorithm details not read, see docs/Integration.md]
- **Mapping to GI-1.0.** It is the closest published match to GI-1.0's world cache: hashed cells, per-frame sparse update, query by hit point. GI-1.0 adds direction in the key, fingerprinting, decay eviction and mipped tiles. SHaRC is not a screen cache.
- **What we could borrow.** Its integration pattern (update pass, resolve pass, query in main path). Reading the docs is useful for how it handles hash collisions, accumulation and stale-cell eviction [unverified, not read].
- **Needs hardware RT.** The cache itself is shader-only. NVIDIA's description says it runs on any DXR GPU, so the samples use DXR [snippet]. The cache code is plain shader code.
- **License.** The SHARC repository's License.md is the **NVIDIA RTX SDKs License (February 23, 2024)** [read]. It is proprietary and source-available, not open source. It forbids use that would subject the SDK to open-source license terms, and requires distributing under terms at least as protective. This is incompatible with an MIT pmndrs package. **Clean-room only; do not port the code.** Read the docs for ideas.
- **Links.** https://github.com/NVIDIA-RTX/SHARC, https://github.com/NVIDIA-RTX/RTXGI (RTXGI 2.0 includes SHaRC and NRC samples).

**Neural Radiance Cache (NRC).**
- **Core idea.** A small neural network is trained online, every frame, on sparse path-traced samples (with self-training to approximate many bounces). Path tracing then terminates early and queries the net. About 2.6 ms per frame at full HD in the paper. [snippet]
- **Mapping to GI-1.0.** Same role as the world cache, but learned and continuous instead of hashed and discrete.
- **Needs hardware RT.** The sample integration is DXR-based. NRC in RTXGI needs NVIDIA Turing or newer with Tensor Cores [snippet]. Not usable for us.
- **License.** Same NVIDIA RTX SDKs license for the SDK. The research implementation is in tiny-cuda-nn [snippet, unverified].
- **Links.** Paper: https://arxiv.org/abs/2106.12372 (Müller, Rousselle, Novák, Keller, SIGGRAPH 2021). https://research.nvidia.com/labs/rtr/publication/muller2021nrc

## 5. Embark kajiya (ircache, ReSTIR GI)

- **Core idea.** A hardware-RT renderer by Tomasz Stachowiak. The first bounce uses ReSTIR-based final gather. Later bounces go through a **world-space irradiance cache (ircache)**, a volumetric, temporally recurrent clipmap/sparse cache with "infinite" bounces, allocated only where queried. [read: README summary of the cache, snippet for ReSTIR use]
- **Mapping to GI-1.0.** ircache is another world-space radiance cache that is allocated on demand from queries, like GI-1.0's hash cells. ReSTIR GI plays the screen-side role (resampling at the first hit) in place of screen probes.
- **What we could borrow.** The query-driven allocation and "recurrent" feedback design; compare against GI-1.0's decay-based eviction. Not a replacement.
- **Hardware RT.** The primary path requires RT hardware (Nvidia RTX, GTX 1060+ 6 GB with emulated RT, Radeon RX 6000). An "RTX off" fallback mode exists with most GI features disabled. [read]
- **License and status.** Apache-2.0 or MIT (dual). **Archived by the owner on 2026-06-01 and marked "no longer maintained"** [read]. Code is the only open reference we have of ircache, so it is usable as a licensed reference for porting ideas.
- **Links.** https://github.com/EmbarkStudios/kajiya. A commentary overview (not by the author) is at https://lousodrome.net/blog/light/2022/06/ [snippet]. I did not find the author's own write-up [unverified].

## 6. EA SEED GIBS / surfel GI (Frostbite)

- **Core idea.** Surfels (small discs: position, normal, radius) are spawned from the G-buffer as geometry enters view, persist, and follow moving geometry via a global transform buffer. A spatial structure indexes them, and rays are traced (hardware RT) from surfel centers to gather lighting, with ray guiding, ray binning and spatial filters. Pixels read nearby surfels. Free surfels are recycled from a stack. [snippet, EA SEED page and slides summary]
- **Mapping to GI-1.0.** Surfels are a third option for the "cache entry" abstraction: object-attached discs (handles moving objects) instead of GI-1.0's quantized hash cells (view-dependent but not object-attached). They overlap with the world cache, not the screen cache.
- **What we could borrow.** Object-attached cache entries for dynamic moving objects, which GI-1.0's world hash cells do not natively handle (cells are world positions; a moving mesh leaves stale lighting). This matters for our dynamic-first priority. Surfel recycling and coverage budgets.
- **Hardware RT.** The original uses hardware RT. Webgiya (section 10) shows a compute-BVH variant.
- **License.** Slides only. EA SEED does not publish code [unverified].
- **Links.** https://www.advances.realtimerendering.com/s2021/SIGGRAPH%20Advances%202021%20-%20Surfel%20GI.pdf, https://www.ea.com/seed/news/siggraph21-global-illumination-surfels. Presenters named variably (Halén, Hayward, Brinck, Bei).

## 7. AMD later work

- **Capsaicin framework (GI-1.0/1.1/1.2 reference).** MIT licensed [snippet, from secondary sources and GitHub topic listing; verify the LICENSE file]. Requires D3D12 and DXR 1.1. Latest known: v1.3. Repo: https://github.com/GPUOpen-LibrariesAndSDKs/Capsaicin. This is the reference implementation and is the only permissively licensed code for GI-1.0. Pointer for other agents.
- **GI-1.1** (SIGGRAPH Asia 2023, Eto, Meunier, Harada, Boissé): glossy reflections using the two-level cache, GGX sampling, specular interpolation vs RT reflection by roughness, dedicated reflection denoiser. https://gpuopen.com/learn/gi-1-1-glossy-reflection-rendering/ [snippet]. Paper PDF path: /download/publications/SA2023_RealTimeReflection.pdf on gpuopen.com [snippet, from the GI-1.2 page].
- **GI-1.2** (2025-11-20): adds a second-bounce cell layer to the hash cache; 1st-bounce cells keep separate direct and indirect estimators; 2nd-bounce cells are filtered like 1.0's. [read] https://gpuopen.com/learn/gi-1-2-multibounce-indirect-rendering/. A KTH master's thesis accompanies it (urn:nbn:se:kth:diva-367822).
- **FidelityFX Brixelizer GI** (SDK 1.1, a simplified GI-1.0). [read: manual page] Compute-only HLSL (SM 6.6), no DXR requirement stated on the page. It traces **Brixelizer's SDF brick cascades** (up to 24 cascades). The radiance cache is a 3D texture (R11G11B10F) of direct light per brick in 4x4x4 blocks, filled from the previous frame's lit output (multi-bounce for free). Screen probes: 8x8 octahedral. An irradiance cache stores SH2 per brick. Plus a specular path (quarter-res brick-ID pre-trace) and disocclusion-mask denoiser. Licensed under the FidelityFX SDK MIT license per GPUOpen [snippet; the manual page itself shows no license text]. **Most relevant to us**: it is a software-traced GI-1.0 derivative with a brick-grid radiance cache, i.e. the same family of design our no-RT constraint needs. Links: https://gpuopen.com/fidelityfx-brixelizer/, https://gpuopen.com/manuals/fidelityfx_sdk/techniques/brixelizer-gi/, GDC 2024 slides https://gpuopen.com/download/GDC2024_GI_with_AMD_FidelityFX_Brixelizer.pdf.
- **FSR Radiance Caching (Redstone), new.** A neural radiance cache in the FSR SDK (rebranded from FidelityFX). 0.9.0 technical preview in SDK 2.1; SDK 2.2 released March 2026; production slated for 2026. Requires Radeon RX 9000 / RDNA 4. Designed to work with Monte Carlo path tracing, replacing secondary bounces with learned predictions. FSR Ray Regeneration is the AI denoiser. [snippet] Closed-source DLLs [unverified]. Relevant as AMD's direction; not usable on WebGPU. Links: https://gpuopen.com/amd-fsr-radiancecaching/, https://gpuopen.com/learn/welcome-amd-fsr-redstone-amd-fsr-sdk-21-available/.
- **No GI-1.0 successor named by AMD** in what I found. Brixelizer GI is not shown as deprecated.

## 8. ReSTIR GI and ReSTIR PT (as screen-cache sample reuse)

- **ReSTIR GI** (Ouyang, Liu, Kettunen, Pharr, Pantaleoni, HPG 2021): resample indirect-light paths (the secondary hit point and its outgoing radiance) with spatiotemporal reservoirs at the primary hit. One path per pixel, reused across neighbors and frames, with bias correction. [not re-read this session; cited by both GI-1.0 and id Tech 8, read in those documents]
- **ReSTIR PT** (Lin, Kettunen, Bitterli, Pantaleoni, Yuksel, Wyman, SIGGRAPH 2022): full path reuse with shift mappings. Heavy; not realistic for our budget.
- **Mapping to GI-1.0.** GI-1.0 deliberately avoids screen-space ReSTIR and uses ReSTIR only for direct-light sampling of world-cache entries (per the paper summary [snippet]). The reason is cost and bias. ReSTIR GI is the "dense per-pixel" alternative to the screen probe cache: more detail per pixel, noisier and heavier on memory, uses reservoir buffers instead of probes. id Tech 8 evaluated it and did not ship it (it is in their reading list, not in the pipeline).
- **What we could borrow.** Reservoir reuse for light sampling at world-cache hits is already in GI-1.0. A ReSTIR-GI-style first-hit pass is an optional high-quality tier, not a first extension.
- **Hardware RT.** No, but needs ray-tracing-style visibility for shadow rays and reuse (our BVH).
- **License.** NVIDIA's RTXDI SDK is under the NVIDIA RTX SDKs license (same caveat as SHaRC) [unverified for RTXDI specifically]. kajiya has an open ReSTIR GI implementation (Apache/MIT) but is archived.
- **Links.** ReSTIR GI: https://research.nvidia.com/publication/2021-07_restir-gi-path-resampling-real-time-path-tracing [unverified URL]. ReSTIR PT: SIGGRAPH 2022 "Generalized Resampled Importance Sampling" is the closest foundation paper; ReSTIR PT paper link not verified.

## 9. Godot SDFGI / HDDAGI

- **SDFGI** (shipped in Godot 4): cascaded SDF (cascaded voxel) GI, probe-based. [snippet] Known trade-offs: leaking, light-bleeding on thin geometry, cascade update costs.
- **HDDAGI** (Linietsky, announced late 2023): a successor replacing the SDF with a hierarchical voxel structure traced by a hierarchical DDA, with local cascade updates, less leaking and lower memory. It stalled for over two years and has been revived by a contributor (PRs #86267 / #119869; the numbering in my sources conflicted) [snippet, secondhand]. Quoted probe pass speedup (about 2.4 to 0.9 ms) is unverified.
- **Mapping to GI-1.0.** Same family as the world cache: a clipmap/cascade world representation with probes traced by DDA instead of BVH rays. Closest to Brixelizer GI.
- **What we could borrow.** Voxel-DDA tracing as a cheap long-ray tracer if BVH traversal proves too expensive for the world-cache update. A voxel occupancy/brick structure can be built by the same compute BVH builder pass. Godot is MIT licensed; the relevant source would be readable and portable (the HDDAGI branch is unmerged [unverified]).
- **Needs hardware RT.** No. It is compute-only.
- **Links.** https://gamefromscratch.com/godot-new-global-illumination-system-hddagi/. Godot's repo: https://github.com/godotengine/godot (search for HDDAGI PRs).

## 10. Existing WebGPU / WebGL work

| Project | Technique | API | License | Status | Notes |
|---|---|---|---|---|---|
| jure/webgiya https://github.com/jure/webgiya | Surfel GI: allocate surfels, cascaded spatial hash grid, BVH trace, temporal update, per-pixel resolve; fully compute | WebGPU, three.js; vendored three-mesh-bvh WebGPU backend | MIT [read] | 19 commits, 163 stars, 2 open issues; last-commit date not seen [read] | Closest to our target. Contains a cascaded hash grid and a BVH trace in WGSL. Verify last-commit date. |
| CodyJasonBennett/three-rc https://github.com/CodyJasonBennett/three-rc | 3D Radiance Cascades with screen-space probes and a BVH (three-mesh-bvh) | WebGL | **No license** (author states the upscaler cannot be licensed; "All rights reserved") [read] | 76 commits, 84 stars; last-commit date not seen | Cannot copy code. Can read for ideas. Author is a pmndrs/three maintainer: a licensing conversation is possible. |
| norio/speedball-gi (npm speedball-gi) | BVH-traced DDGI, spectral path tracing, photon caustics | three.js WebGPU | MIT per trendshift listing [snippet] | Versions 0.6.x; repo state not opened | Competing DDGI implementation. Compare to ours; verify repo before citing. |
| three-realtime-rt (npm) https://discourse.threejs.org/t/three-realtime-rt-turn-on-ray-traced-lighting-for-three-js-soft-shadows-1-bounce-gi-temporal-denoise-webgl2/92763 | One-bounce GI, soft shadows, temporal reprojection, a-trous denoiser | WebGL2 | Not verified | Not verified | Low relevance beyond a-trous denoiser reference. |
| 0beqz/realism-effects https://github.com/0beqz/realism-effects | SSGI, SSR, TRAA, motion blur, AO (screen-space only, temporal reprojection) | three.js, via postprocessing; no WebGPU mentioned | MIT [read] | 580 commits, 1.7k stars, 28 open issues; v2 rewrite on a branch; last-commit date not seen | Screen space only: no off-screen light. Useful for denoiser/reprojection ideas and as a baseline. |
| three.js built-in Light Probe Volume (r17x+) | SH L1 probe grid | WebGPU/WebGL | MIT (three.js) | Active | Diffuse GI baseline; see discourse thread https://discourse.threejs.org/t/global-illumination/91243 [snippet]. |
| Muni thesis, Slezáček, "Global Illumination Based on Surfels Using WebGPU" https://is.muni.cz/th/nuvo6/ | Surfel GI | WebGPU | Not verified | Thesis | Likely related to Webgiya lineage [unverified]. |
| wgpu radiance cascades (Rust, `arcane-core`, others) | RC in 2D/3D | wgpu (native) | Not verified | Not verified | WGSL is shared with the browser; check licenses before reading. |

Nothing found: a browser implementation of GI-1.0 or of a Lumen-like two-level screen/world radiance cache, other than Webgiya's hash grid (surfel based) and three-rc's screen-space RC. Searches were limited; there may be demos I did not find.

## Extension ideas ranked (after a faithful GI-1.0 port)

1. **DDGI as the third level of a fall-through (screen cache, world hash cache, DDGI volume)**, id Tech 8 style. We already ship DDGI, so the volume, its octahedral atlas format and its update are existing work. Source: Sousa SIGGRAPH 2025 (section 2). Benefit: no-cache-hit rays and cache-cell misses get a stable fallback; DDGI also supplies the multi-bounce term, like id's irradiance volumes feeding the world cache. Closest to GI-1.2's second-bounce layer in effect.
2. **Second-bounce layer in the hash cache (GI-1.2).** Small, well-specified, and from the same authors. Source: GI-1.2 page (section 7).
3. **Object-attached cache entries (surfels) for moving objects.** Directly serves the dynamic-objects priority, which world-position hash cells do not natively handle. Sources: EA SEED GIBS (section 6), Webgiya (section 10, working WebGPU precedent).
4. **Software-tracing path via a brick/voxel structure** (Brixelizer GI style SDF bricks, or Godot HDDAGI voxel DDA) for long rays in the cache update, if the compute BVH is too slow there. Sources: Brixelizer GI manual, GDC 2024 slides; Godot HDDAGI; Lumen's global SDF.
5. **Probe-space filtering, product importance sampling and fixed-budget priority update from Lumen's screen/world caches.** Cheap quality and stability wins. Source: Lumen SIGGRAPH 2022 slides.
6. **Split Radiance Cascades experiment for the world cache.** Hit-distance ray splitting could let screen rays feed a hashed world cache with cascade merge instead of long temporal accumulation. High potential, high risk: four months old, no public code. Source: arXiv 2607.20384.
7. **ReSTIR-GI-style per-pixel tier** as a high-quality option on strong GPUs. Heavy, not a first extension. Source: Ouyang 2021.
8. **Glossy reflections (GI-1.1)**: reflection ray sampling from the same two caches, plus a dedicated denoiser. Useful if reflections become a feature goal. Source: SIGGRAPH Asia 2023 paper.

Not recommended: neural caches (NRC, FSR Radiance Caching): vendor-specific hardware and closed or restrictively licensed code.

## Talks and papers worth downloading later

Download as local reference copies for the research folder (check each site's terms; the SIGGRAPH Advances course materials are posted for public download).

1. Sousa, "idTech 8 Global Illumination", SIGGRAPH 2025: https://www.advances.realtimerendering.com/s2025/content/SOUSA_SIGGRAPH_2025_Final.pdf (already seen locally during this research; 38 pages)
2. Wright, Narkowicz, Kelly, "Lumen: Real-time Global Illumination in UE5", SIGGRAPH 2022: https://advances.realtimerendering.com/s2022/SIGGRAPH2022-Advances-Lumen-Wright%20et%20al.pdf (27 MB)
3. Wright, "Radiance Caching for Real-Time Global Illumination", SIGGRAPH 2021 Advances (locate on advances.realtimerendering.com s2021)
4. Halén et al., "Global Illumination Based on Surfels", SIGGRAPH 2021: https://www.advances.realtimerendering.com/s2021/SIGGRAPH%20Advances%202021%20-%20Surfel%20GI.pdf
5. AMD GI-1.0 technical report: https://gpuopen.com/download/GPUOpen2022_GI1_0.pdf and arXiv 2310.19855
6. GI-1.1 paper (Eto, Meunier, Harada, Boissé, SIGGRAPH Asia 2023): https://gpuopen.com/download/publications/SA2023_RealTimeReflection.pdf [path from GI-1.2 page; unverified as a live URL]
7. GDC 2024, "GI with AMD FidelityFX Brixelizer": https://gpuopen.com/download/GDC2024_GI_with_AMD_FidelityFX_Brixelizer.pdf
8. Freeman, Sannikov, Pitt-Francis, "Split Radiance Cascades", arXiv 2607.20384 (CC BY-SA 4.0)
9. Osborne and Sannikov 2024, arXiv 2408.14425, and the Freeman et al. Holographic RC paper (arXiv 2505.02041, unverified)
10. Müller et al., "Real-time Neural Radiance Caching for Path Tracing", arXiv 2106.12372 (background only)
11. Ouyang et al., "ReSTIR GI", HPG 2021; Lin et al., ReSTIR PT, SIGGRAPH 2022 (exact URLs not verified)
12. Gautron, "Real-Time Ray-Traced Ambient Occlusion of Complex Scenes using Spatial Hashing" (SIGGRAPH 2020 talk; the id Tech 8 hash cache cites it); Binder et al. 2019 on spatial hashing (cited by GI-1.0)
13. Majercik et al. 2019, DDGI (already in our DDGI notes)
14. Code to clone for reading: Capsaicin (MIT), Webgiya (MIT), Brixelizer GI source in the FidelityFX SDK, kajiya (Apache/MIT, archived). Do not clone SHARC/RTXGI into the repo; read online only.

## Licensing summary for our port

| Source | License | Can we port code? |
|---|---|---|
| AMD Capsaicin (GI-1.0 reference) | MIT [snippet] | Yes, with notices |
| FidelityFX / FSR SDK Brixelizer | MIT per GPUOpen [snippet] | Yes, verify file headers |
| Webgiya | MIT [read] | Yes |
| kajiya | Apache-2.0 or MIT [read] | Yes (archived) |
| realism-effects | MIT [read] | Yes |
| NVIDIA SHARC / RTXGI / NRC | NVIDIA RTX SDKs License [read] | No (clean-room from papers and docs only) |
| three-rc | No license [read] | No |
| Lumen, id Tech 8, Frostbite | Proprietary | No (slides only) |
| Split RC paper | CC BY-SA 4.0 (text only) | Ideas yes; no code released |
