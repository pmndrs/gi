# AMD GI-1.0 (Capsaicin) source map for a WebGPU/WGSL port

Date: 2026-11-09. Source: `GPUOpen-LibrariesAndSDKs/Capsaicin`, tag v1.3, commit `914b91596cd119eda85fbc1d3c7ee6ac391b1452` (checked out in a scratch clone; nothing from the repo is vendored here). Paper reference: arXiv 2310.19855. I did not re-read the paper in this pass, so section 8 is inferred from code, comments and the README only.

Path shorthand used for every citation (all `path:line` at v1.3):

- `gi1/` = `src/core/src/render_techniques/gi1/`
- `src/` = `src/core/src/`
- `root/` = repository root

One naming oddity up front: the repo calls the technique **GI-1.2** (`root/README.md:18`, `src/renderers/gi1/gi1_renderer.cpp:44`). GI-1.0 is the 2022 technical report; GI-1.1 added glossy reflections (SIGGRAPH Asia 2023 paper, `root/README.md:24`). So v1.3 contains more than the arXiv paper describes (see section 8).

---

## 1. Licenses

| Item | License | Evidence |
|---|---|---|
| Capsaicin repo (includes all GI-1 code) | MIT text, "Copyright (c) 2023 Advanced Micro Devices, Inc." | `root/LICENSE:1-20` (text is the standard MIT grant, no title line) |
| GI-1 source files | Same MIT text, "Copyright (c) 2025 AMD" in each header | `gi1/gi1.comp:1-21` (the other gi1 files carry the same header) |
| `third_party/FidelityFX` headers | MIT text, AMD 2024 | `root/third_party/FidelityFX/gpu/ffx_core.h:3-20`; all 13 files under it carry the same AMD 2024 header (checked by grep). GI-1 does not include any of them (no `ffx_` reference under `gi1/`). |
| `third_party/gfx` (git submodule, host-side D3D12 wrapper) | MIT, "Copyright (c) 2024 Guillaume Boisse" | submodule declared at `root/.gitmodules:1-3`, pinned commit `c65d6172500864b68988cdf64846520ebbb42e6f`; LICENSE verified by fetching that commit separately. Its own `third_party/` (AmdDxExt headers, etc.) was not examined. Host-side only, not needed for the port. |

Files whose header differs from the AMD MIT header (found with a repo-wide grep for files lacking "Advanced Micro Devices"):

- `src/render_techniques/atmosphere/atmosphere.hlsl:1-20`: "Copyright (c) 2021 Felix Westin", MIT text. Not used by GI-1 (it only runs after GI-1 in the renderer, `src/renderers/gi1/gi1_renderer.cpp:63`).
- `src/components/blue_noise_sampler/blue_noise_sampler_samples.h:1-17`: no license header at all. It embeds `Sobol256x256` (line 5), `ScramblingTiles` (line 11) and `RankingTiles` (line 17), and the HLSL cites Heitz et al. (`src/components/blue_noise_sampler/blue_noise_sampler.hlsl:38`). **GI-1 depends on this data** (`gi1/gi1.comp:1540`, `:2978`, `:3197`). Provenance and license of the tables are unstated, so treat them as unverified; plan to regenerate our own tables or use a different sampler rather than copying them.

No other non-AMD copyright headers exist in `src/core/src` (the grep for "Copyright" lines not naming AMD returned only the Felix Westin file).

Bottom line: everything GI-1 needs on the shader side is MIT, with one unlicensed-data caveat (blue noise tables). MIT lets us port with attribution kept in a NOTICE.

---

## 2. Where GI-1 lives and what it relies on

### 2.1 GI-1 proper

| File | Lines | Role |
|---|---|---|
| `gi1/gi1.h` | 472 | `class GI1 : RenderTechnique` (line 31), `RenderOptions` defaults (48-89), sub-structs `ScreenProbes` (170), `HashGridCache` (232), `WorldSpaceReSTIR` (314), `GlossyReflections` (347), `GIDenoiser` (376), kernel handles (408-470) |
| `gi1/gi1.cpp` | 3293 | Resource allocation (113-1240), options (1244-1335), shared textures (1355-1380), `init` kernel creation (1398-1715), per-frame `render` (1716-3103), GUI (3188) |
| `gi1/gi1.comp` | 4269 | All compute kernels (entry points listed in section 3) |
| `gi1/gi1.frag`, `gi1/gi1.vert` | 215 / 265 | `ResolveGI1` full-screen pixel pass (`gi1.frag:76`, `gi1.vert:39`) and three debug draws |
| `gi1/gi1.rt` | 182 | DXR 1.0 raygen/hit wrappers (`gi1.rt:23` sets `USE_INLINE_RT 0`). Optional path, drop it. |
| `gi1/screen_probes.hlsl` | 353 | Probe bindings, LDS scratch, quantization, SH irradiance, CDF scan |
| `gi1/hash_grid_cache.hlsl` | 495 | Hash grid cache: cell size, hashing, insert/find, packing, mip lookup |
| `gi1/world_space_restir.hlsl` | 166 | World-space reservoir hash (optional feature) |
| `gi1/glossy_reflections.hlsl` | 170 | Reflection buffer bindings and helpers |
| `gi1/gi_denoiser.hlsl` | 94 | Blur mask, NaN guard, history helpers |
| `gi1/gi1.hlsl` | 65 | `kGI1_InvalidId`, Halton sequence (`gi1.hlsl:36-55`) |
| `gi1/gi1_shared.h` | 213 | C++/HLSL shared constant structs and buffer enum names |
| `src/renderers/gi1/gi1_renderer.cpp` | 74 | The technique chain (below) |

Technique chain for the GI renderer (`src/renderers/gi1/gi1_renderer.cpp:60-70`): VisibilityBuffer, SSGI, GI1, Atmosphere, Skybox, Combine, FSR, AutoExposure, Bloom, ToneMapping, Lens. Only the first three matter for GI-1. FSR is the upscaler/TAA stage (jitter comes from `src/capsaicin/capsaicin_internal.cpp:448-465`; GI-1 reads `taa_enable` at `gi1/gi1.cpp:1816`).

### 2.2 Shared framework pieces GI-1 pulls in

Components GI-1 declares (`gi1/gi1.cpp:1337-1346`): `BrdfLut`, `PrefilterIBL`, `LightSamplerGridStream`, `BlueNoiseSampler`, `RandomNumberGenerator`.

Shader includes (`gi1/gi1.comp:106-125`): light sampler stream, blue noise, RNG, `geometry/geometry.hlsl`, `geometry/mesh.hlsl`, `lights/lights.hlsl`, `materials/materials.hlsl`, `materials/material_sampling.hlsl`, `math/{color,hash,spherical_harmonics,pack}.hlsl`, `ray_tracing/trace_ray.hlsl`.

Shared textures GI-1 reads or writes (`gi1/gi1.cpp:1355-1380`), and who produces them:

| Texture | Format | Producer |
|---|---|---|
| `Visibility` | RGBA32F: barycentrics.xy, `asfloat(instanceID)`, `asfloat(primitiveID)` (`src/render_techniques/visibility_buffer/visibility_buffer.cpp:81-84`; decode at `gi1/gi1.comp:928-932`, `gi1/gi1.frag:90-93`) | Visibility buffer (mesh-shader raster, or compute RT fallback `visibility_buffer_rt.comp`) |
| `VisibilityDepth` (+`PrevVisibilityDepth`) | R32F | same, `visibility_buffer.cpp:86-90` |
| `GeometryNormal`, `ShadingNormal` (+prev) | R10G10B10A2_UNORM | same, `visibility_buffer.cpp:91-102` |
| `Velocity` | R16G16F | same, `visibility_buffer.cpp:95-98` |
| `Roughness` (+prev) | R16F | same, `visibility_buffer.cpp:107-110` |
| `Gradients` | RGBA16F (UV gradients for texture LOD) | same, `visibility_buffer.cpp:111-114` |
| `DisocclusionMask` | R8 | same, `visibility_buffer.cpp:115-118` |
| `OcclusionAndBentNormal`, `NearFieldGlobalIllumination` | not GI-1 outputs | SSGI technique (`src/render_techniques/ssgi/ssgi.cpp`, 268 lines + `ssgi.comp`, 252). GI-1 uses them only when present (`HAS_OCCLUSION`, `gi1/gi1.cpp:1421-1423`; consumed in `InterpolateScreenProbes` at `gi1/gi1.comp:1617-1627`). |
| `PrevCombinedIllumination` | previous-frame final lit color | Combine technique; used by temporal feedback (`gi1/gi1.comp:2458-2490`) and reflection hits (`gi1/gi1.comp:2842-2863`) |
| `GlobalIllumination` | RGBA16F | GI-1 output (`gi1/gi1.cpp:1359-1362`) |
| `Reflection` (+`PrevReflection`) | RGBA16F | GI-1 output (`gi1/gi1.cpp:1363-1367`) |

Other framework dependencies:

- **Acceleration structure**: `g_Scene` (`RaytracingAccelerationStructure`, `gi1/gi1.comp:73`), built by the framework; plus `g_IndexBuffer`, `g_VertexBuffer`, `g_InstanceBuffer`, `g_MaterialBuffer`, `g_TransformBuffer` (`gi1/gi1.comp:56-61`) for hit reconstruction.
- **Light list**: `LightBuilder` gathers emissive triangles each frame (`src/components/light_builder/gather_area_lights.comp`); `g_LightBuffer` and `getNumberLights()` (`src/lights/lights.hlsl`).
- **Light sampler**: `LightSamplerGridStream` (`src/components/light_sampler_grid_stream/*`, 556 + 231 lines C++/H, 558 HLSL, 221 + 83 comp).
- **GPU utilities** from the host `gfx` library, not Capsaicin: `gfxCommandScanSum` (prefix sum, four calls at `gi1/gi1.cpp:2286`, `:2305`, `:2444`, `:2553`), `gfxCommandClearBuffer/Texture`, `gfxCommandDispatchIndirect`, `gfxCommandDraw`. We must write our own scan.
- **Samplers/environment**: `g_EnvironmentBuffer` cube map (`gi1.comp:75`), BRDF LUT (`gi1.comp:84`), prefiltered IBL.
- **Material evaluation**: `MakeMaterialBRDF`, `MakeMaterialEvaluated`, `emissiveAlphaScaled`, GGX sampling helpers (`src/materials/*`, 1763 lines).

---

## 3. Pass-by-pass table (execution order, default configuration)

Defaults: `gi1_use_dxr10=false` (compute with inline ray queries), `gi1_use_multibounce=true`, `gi1_use_resampling=false`, `gi1_use_temporal_feedback=false`, `gi1_use_direct_lighting=true`, reflections denoiser mode 1 (A-trous), firefly cleanup on (`gi1/gi1.h:48-89`). `render()` starts at `gi1/gi1.cpp:1716`. The kernel is the `gi1.comp` entry point; the "cpp" column is the dispatch site. Thread-group sizes are the `[numthreads]` values. "1080p" figures use 1920x1080 with default settings: 240x135 = 32,400 probe tiles, 16-px spawn tiles give 120x68 = 8,160 spawned probes, `max_ray_count` = 8,160 x 64 = **522,240** (formula at `gi1/gi1.cpp:126-129`; with multibounce the combined ray cap doubles, `:640`).

Notation: "indirect(X)" means the dispatch size is produced on the GPU by `GenerateDispatch` (`gi1.comp:208-217`, cpp helper `gi1.cpp:3267-3274`) from the count in buffer X, rounded up to the group size.

| # | Pass / kernel | Kernel line (`gi1/gi1.comp`) | Dispatch site (`gi1/gi1.cpp`) | Dispatch shape | Inputs | Outputs | What it does |
|---|---|---|---|---|---|---|---|
| 0 | (VisibilityBuffer, SSGI techniques) | n/a | `gi1_renderer.cpp:60-61` | n/a | scene | shared textures in section 2.2 | Produce the G-buffer/visibility inputs. |
| 1 | `ClearBucketOverflowCount` (debug stats only) | 4157 | 2229-2236 | 1D, ceil(num_buckets/64) | n/a | per-bucket overflow counters | Debug statistics reset. Skip in port. |
| 2 | `ClearCounters` | 163 | 2250 | 1x1x1 | n/a | zeroes append counters: reservoir list, empty/override/LRU/MRU/list-element counts, packed-tile, update-tile, visibility, visibility-ray, resolve, rt-sample counts (`gi1.comp:165-181`) | Frame reset of all atomic counters. |
| 3 | `PurgeTiles` | 1714 | 2240-2253 | indirect(previous packed-tile count), 64 | previous frame packed tile list, tile decay buffer | hash buffer entry zeroed for dead tiles; current packed-tile list | Evicts a tile when `frame - lastTouched >= 50` (`hash_grid_cache.hlsl:29`, `gi1.comp:1731-1737`), handling uint wraparound; survivors re-appended to the current packed list. |
| 4 | `ReprojectScreenProbes` | 293 | 2258-2271 | (probe_count.x, probe_count.y) groups of 8x8, i.e. one group per 8x8 probe tile | prev depth/normal/roughness/velocity, previous probe texture + mask, cached-tile LRU | current probe radiance (reprojected), probe SH, empty-tile list (disoccluded tiles), per-tile probe mask | Per probe, picks the best matching previous probe via LDS `InterlockedMin` score (`:348`), remaps radiance cells with LDS quantized `InterlockedAdd` (`:398-402`), builds a backup radiance for unvisited cells, reprojects SH, lists disoccluded tiles for adaptive sampling (`:442-448`). |
| 5 | `CountScreenProbes` | 501 | 2276-2285 | 1D, ceil(probe_count/64) | cached tile index texture | per-probe counts, list element buffer | First half of looking up off-screen cached probe tiles (LRU cache) in the current view. A buffer clear precedes it (`:2283`). |
| 5b | ScanSum on `probe_cached_tile_list_index_buffer_` | n/a (gfx lib) | 2286 | n/a | counts | exclusive prefix | Offsets for the scatter. |
| 6 | `ScatterScreenProbes` | 541 | 2288-2291 | indirect(list element count), 64 | scan result | cached tile list | Scatter step of the counting sort. |
| 7 | `SpawnScreenProbes` | 555 | 2296-2304 | 1D, ceil(max_probe_spawn_count/64) (8,160 at 1080p) | geometry normal | spawn seed per 16x16 tile (Halton-jittered, `gi1.hlsl` Halton, `gi1.comp:562-568`), 0/1 scan flag (sky = 0) | One candidate probe per 16x16 pixel tile, jittered per frame. |
| 7b | ScanSum on `probe_spawn_index_buffer_` | n/a | 2305 | n/a | scan flags | compaction index | Stream compaction offsets. |
| 8 | `CompactScreenProbes` | 581 | 2307-2308 | same as 7 | scan results | compact spawn list, probe masks, "overridable tile" list (`:601-607`) | Compacts surviving probes. |
| 9 | `PatchScreenProbes` | 615 | 2312-2319 | indirect(empty tile count), 64 | empty-tile list, override list | spawn buffer patched (`InterlockedExchange`, `:643`) | Adaptive sampling: moves probes from stable tiles into disoccluded empty tiles. |
| 10 | `SampleScreenProbes` | 646 | 2323-2330 | 1D, ceil(max_ray_count/64) (522,240 threads = 8,160 x 64) | previous probe radiance (3x3 tap neighborhood, `:689-760`), material, visibility | `probe_spawn_sample_buffer_` (direction as packed half3), `PreviousProbeBuffer` | One thread per (probe, cell) ray. Builds a luminance x cos CDF over the 64 cells with an LDS scan (`screen_probes.hlsl:ScanRadiance`), then either samples GGX (with probability from `calculateBRDFProbability`, `:957-964`) or samples the CDF ("ray guiding", `:966-980`). |
| 11 | `PopulateScreenProbesMain` | 1160 (core at 1107) | 2335-2360 | 1D, ceil(max_ray_count/32) | spawn buffers, `g_Scene` | per-ray radiance+hit distance (`probe_spawn_radiance_buffer_`, packed half4), hash grid visibility/cell/query buffers, packed tile list, update tile list, tile decay | Closest-hit ray query per probe ray (`:1089`). On miss: env map sample if direct lighting on (`:1078-1083`). On hit: `HashGridCache_InsertCell` (CAS), decay stamp, append hit to visibility list (`:1013-1017`), clear new tile cells, request light-grid bounds. |
| 12 | `PopulateMultibounceCellsMain` (multibounce) | 1942 (core 1869) | 2364-2394 | indirect(VisibilityCountBuffer0), 32 | bounce-1 visibility list | bounce-2 hits appended after bounce-1 ones in the same buffers (`:1788-1795`), BRDF+pdf (`MultibounceInfoBuffer`) | Each bounce-1 hit survives with probability 1-0.7 (`:1879-1882`), forced diffuse (`:1911-1916`), samples BRDF (`sampleBRDFAndEvaluateDiffuse`), closest-hit ray query (`:1857`). |
| 13 | Light sampler update | `light_sampler_grid_stream_bounds.comp:46` (1 thread), `light_sampler_grid_stream_build.comp:54` (128 threads, indirect) | 2398 (`LightSamplerGridStream::update`, `light_sampler_grid_stream.cpp:286`) | bounds: 1x1x1; build: indirect over cells x reservoirs | GPU-collected hit positions (`requestLightSampleLocation`), light list | scene bounds, 16^3 grid x 64 reservoirs of (light index, weights) | Rebuilds the light grid streamed-reservoir structure from where hits landed this frame. |
| 14 | `ClearReservoirs` (resampling only) | 2394 | 2402-2411 | 1D, ceil(0x400000/128) | n/a | world-space ReSTIR hash cleared | Off by default. |
| 15 | `GenerateReservoirs` | 2401 | 2414-2422 | indirect(VisibilityCountBuffer0), 64 | hit visibility list | `IndirectSample*` buffers, `ray list` (compacted shadow rays), `Resolve` flags | Reconstructs hit position/normal/material, handles emissive hits (written straight into screen probe radiance, bypassing cache, `:2439-2456`), optional temporal feedback shortcut, then `sampleLightsListCone<8>` (RIS over 8 candidates from the light grid, `light_sampler_grid_stream.hlsl:343`) to get one light reservoir. Flags "bypass cache" when ray length < cell size (leak guard, comment `:2538-2560`, flag `:2561`). |
| 16 | `GenerateMultibounceReservoirs` | 2603 | 2425-2434 | indirect(VisibilityCountBuffer1), 64 | bounce-2 hits | same as 15 for bounce 2 | Same shading as 15 for second bounce vertices. |
| 17 | `CompactReservoirs` (+ScanSum) (resampling only) | 2699 | 2437-2451 (scan 2444) | indirect(list count), 64 | reservoir hash list | reservoir hash value array | Off by default. |
| 18 | `ResampleReservoirs` (resampling only) | 2713 | 2454-2462 | indirect(VisibilityRayCount), 64 | previous-frame reservoirs | merged reservoirs | Off by default. |
| 19 | `PopulateCellsMain` | 2111 (core 2040) | 2466-2492 | indirect(VisibilityRayCount), 32 | reservoir (light sample), hit material, `g_Scene` | atomic accumulation into `UpdateCellValueBuffer[4*cell+0..3]` (quantized x1000, `hash_grid_cache.hlsl:32`; adds at `gi1.comp:2091-2095`) | Shadow ray (first-hit-and-end) toward the sampled light; on unoccluded, evaluates BRDF at the hit toward the ray origin times light radiance times reservoir weight (`PopulateCellsHandleMiss`, `:2006-2020`) and accumulates into the cell. |
| 20 | `GenerateUpdateTilesDispatch` + `UpdateTiles` | 2130, 2143 | 2495-2502 | 1x1x1 then ceil(updateTileCount / tilesPerGroup) groups of 8x8 | quantized accumulators | `ValueBuffer` and `ValueIndirectBuffer` (RGB half + sample count w, packed `uint2`), accumulators cleared | Per tile: temporal accumulate, count capped at `max_sample_count` 16 (`:2163-2200`; `max_multibounce_sample_count` 16 for indirect), then builds mips 1..3 as 2x2 box sums through LDS with group barriers (`:2228-2349`). |
| 21 | `UpdateMultibounceCellsMain` (multibounce) | 1986 (core 1948) | 2507-2515 | indirect(VisibilityCountBuffer1), 32 | bounce-2 cell filtered direct radiance | `UpdateCellValueIndirectBuffer` of the bounce-1 cell | Adds `direct * brdf / pdf / (1-0.7)` from the second hit into the first hit cell's indirect accumulator (`:1960-1975`). Note indirect accumulators are applied next frame, so multibounce lags one frame. |
| 22 | `ResolveCells` | 2351 | 2518-2528 | indirect(ResolveCount or VisibilityRayCount), 64 | cell value buffers (filtered via mips) | `ScreenProbes_AccumulateRadiance` into `probe_spawn_radiance_buffer_` (read-modify-write, not atomic) | Reads each ray's hash cell with adaptive mip selection (`hash_grid_cache.hlsl:434-494`) and adds radiance to the probe ray, unless the bypass flag is set (`:2368-2372`). |
| 23 | `BlendScreenProbes` | 1167 | 2532-2542 | 1D, ceil(max_ray_count/64) | spawn radiance, previous probe | probe radiance texture, cached-tile texture (LRU) | LDS-accumulates ray results into the 64 cells (quantized atomics, `:1204-1208`), fills untraced cells from backup, then biased temporal hysteresis ("shadow-preserving", `:1265-1270`); writes evicted/updated tiles to the off-screen cache texture. Followed by `gfxCommandCopyTexture` of the probe texture (`:2540`). |
| 24 | `ReorderScreenProbes` (+ScanSum) | 1334 | 2546-2556 (scan 2553) | 1D, ceil(probe_count/64) | LRU flags | LRU order | Keeps least-recently-used cached tiles first for eviction. |
| 25 | `FilterProbeMask` | 249 | 2561-2585 | one 8x8 dispatch per mip level 1..n, size `probe_count >> mip` | probe mask mip i-1 | probe mask mip i (R32_UINT) | Builds a mip chain of probe seeds so any pixel can find a nearby valid probe (`screen_probes.hlsl:FindClosestProbe`). |
| 26 | `FilterScreenProbes` x2 | 1357 | 2590-2614 | 1D, ceil(max_ray_count/64), run twice with `g_BlurDirection` (1,0) then (0,1) | probe radiance, probe mask mips | filtered probe radiance | Separable spatial filter across neighbor probes (radius 3, `:1393`), accepts neighbors by plane distance, hemisphere overlap and hit-reprojection angle (`kGI1_AngleThreshold`), depth weight pow 8. |
| 27 | `ProjectScreenProbes` | 1446 | 2618-2625 | 1D, ceil(max_ray_count/64) | probe radiance | 9 SH L2 coefficients per probe as `uint2` (half4) (`:1462-1511`) | Projects the 64 cell radiances to SH through LDS (first 9 threads each loop the 64 entries). |
| 28 | `InterpolateScreenProbes` | 1515 | 2629-2636 | (ceil(w/8), ceil(h/8)), 8x8 | G-buffer, probe SH, probe mask, SSGI AO/bent normal, blue noise | `GIDenoiser_ColorBuffer` (irradiance, hint), `ReflectionBuffer` (diffuse fallback), `SpecularBuffer`, `DirectionBuffer` (half-res), queued RT reflection samples (`:1640-1660`) | Per pixel: find 4 nearby probes, weight by plane/normal (power 8), evaluate SH irradiance with bent cone from SSGI (`screen_probes.hlsl` `CalculateSHIrradiance_BentCone`), write `float4(irradiance, denoiser_hint)` (`:1638`). Also decides per pixel whether the surface is glossy enough for traced reflections (roughness < 0.2) or probe-based specular (0.2 to 0.6). |
| 29 | `TraceReflectionsMain` | 3039 (core 2933) | 2646-2676 | indirect(rt sample count), 64 | queued reflection samples, `g_Scene` | `SpecularBuffer` (half-res radiance + hint) | Closest-hit ray query along the sampled GGX direction (`:2918`); on hit uses previous frame final color if reprojection valid (`:2842-2863`), else the hash cache (`:2865-2895`, `HashGridCache_FindCell`, jittered by cell size); miss samples the env map. Sampler for anisotropic filtering is swapped in (`:2648`). |
| 30 | `MarkFireflies` | 3813 | 2680-2693 | ceil(half-res/8) 2D | specular buffer | `fireflies_buffer_` (R8_SNORM, half-res) | Flags outlier pixels relative to a neighborhood (radius 3 half / 2 full). |
| 31 | `CleanupFireflies` | 3883 | 2696-2709 | same | specular + mask | cleaned specular | Replaces flagged samples (radius 2 half / 1 full). |
| 32 | Reflection resolve, mode 1 default: `ResolveReflections_AtrousRatioEstimator_First` / `_Iter` / `_Last` | 3448 / 3554 / 3663 | 2757-2850 | full or half-res 2D, 8x8; `atrous_pass_count`=4 gives First + 2x Iter + Last (4 dispatches), step `1 << pass` | specular, direction, normals, depth | ping-pong `ReflectionsBuffer0/1` + `AverageSquaredBuffer0/1`, final `ReflectionsBuffer` | Edge-aware A-trous filter using a BRDF ratio estimator (numerator/denominator split) and a variance estimate. Each pass uploads a small constant buffer (`ping_pong`, `full_step`, `pass_index`). Mode 0 (`SplitRatioEstimatorX/Y`, 3045 / 3157, cpp 2711-2745) and mode 2 (`NoDenoiserReflections`, 3958, cpp 2872-2880) are alternatives. |
| 33 | `ReprojectReflections` | 3282 | 2859-2867 | full-res 2D, 8x8 | previous reflection, velocity | `g_ReflectionBuffer` (history blended) | Temporal accumulation of reflections. Skipped in mode 2. |
| 34 | `ReprojectGI` | 3997 | 2884-2897 | full-res 2D, 8x8 | current color (interpolated irradiance), previous color+delta, prev depth/normal | history color (w = sample count), `ColorDelta`, `BlurMask` (R8_SNORM) | Bilateral 3x3 history reprojection (`:4008-4075`), luminance-delta based alpha blend, vignette-scaled max sample count (`:4092`). |
| 35 | `FilterGI` x2 | 4105 | 2901-2927 | full-res 2D, 8x8, run twice with `g_BlurDirection` | history color, blur mask | `IrradianceBuffer` (second pass) | Separable edge-aware blur only where the blur mask > 0 (low history count), radius up to `kGIDenoiser_MaxBlurMask`=8 (`gi_denoiser.hlsl:26`). |
| 36 | `ResolveGI1` (graphics) | `gi1.frag:76`, `gi1.vert:39` | 2935-2945 | full-screen triangle (`gfxCommandDraw(3)`) | irradiance, reflections, G-buffer, material, BRDF LUT | `GlobalIllumination` RGBA16F | Per-pixel composite: emissive + Lambert albedo x diffuse compensation x irradiance + split-sum specular (`gi1.frag:129-147`). Falls back to filtered irradiance for roughness > `high_roughness_threshold`. |
| 37 | Debug kernels | `ClearBucketOccupancy` 4170, `ClearBucketOverflow` 4188, `BuildBucketStatistics` 4200, `FormatBucketOccupancy` 4237, `FormatBucketOverflow` 4257; `GenerateDraw` 187; graphics `DebugScreenProbes`, `DebugHashGridCells`, `DebugReflection` | 2950-3090 | n/a | n/a | n/a | Debug only. Skip. |
| 38 | Buffer flips | n/a | 3096-3102 | n/a | n/a | n/a | Ping-pong of probe buffers, hash-grid packed-tile buffers, reservoir buffers, GI color buffers. |

Init-time/resize-time only: `ClearProbeMask` (`gi1.comp:243`, dispatched at `gi1.cpp:158-182` per mip) and `InitCachedTileLRU` (`gi1.comp:287`, `gi1.cpp:343`).

Counts: in the default flow GI-1 owns about **34 distinct compute kernels plus 1 full-screen pixel pass** (items 2-12, 15-16, 19-35 with the A-trous trio, helpers counted), plus 2 light-sampler kernels and the light builder kernel. The file defines 50 compute entry points in total (44 outside the six hash-stats debug kernels), of which resampling/ReSTIR (5), DXR-only wrappers and debug are off by default. With four prefix-scan calls, the per-frame dispatch count at 1080p is roughly 45 to 55 (the 2x filter passes, per-mip probe mask passes, and A-trous iterations repeat kernels). This count is my estimate, not read from a profiler.

---

## 4. Data structures

### 4.1 Screen probes

- Probe: 8x8 cells = 64 radiance values per probe (`gi1/gi1.h:191`), **hemi-octahedral** direction layout in the probe's local frame (Clarberg-style equal-area map, `src/math/sampling.hlsl:61-100`), oriented to the surface normal via `GetOrthoVectors`.
- Sampling density: `kSamplingMode_QuarterSpp` is a compile-time constant (`gi1.h:192`), so one probe is spawned per 16x16 pixel tile (`probe_spawn_tile_size_ = probe_size << 1`, `gi1.h:195-198`) with a Halton jitter per frame. Result: 1/4 ray per pixel, 64 rays per probe.
- Storage (`gi1/gi1.cpp:113-345`):
  - Probe radiance: two `R16G16B16A16_FLOAT` textures of `probe_count*8` size (basically screen size), ping-pong (`:144`). Alpha stores hit distance (used by the filter for hit reprojection).
  - Probe mask: `R32_UINT` texture, `probe_count` size with a full mip chain, 2 copies (`:168`). Texel = packed seed pixel `(x<<16)|y` (`screen_probes.hlsl:ScreenProbes_PackSeed`) or `0xFFFFFFFF` for none.
  - SH: `StructuredBuffer<uint2>`, 9 x probe_count entries, each entry = half4 (`:199`, `screen_probes.hlsl:ScreenProbes_PackSHColor`), 2 copies.
  - Spawn lists: spawn seed `uint` per spawn tile; scan and index `uint`; per-probe `uint2` (evicted/updated cache index); per-ray `uint2` sample direction (packed half3) and `uint2` radiance (packed half4) sized `max_ray_count` (`:222-241`).
  - Off-screen cache: `RGBA16F` texture the size of the probe grid in texels plus an `RGBA32F` index texture at probe resolution (world position + packed normal) plus LRU/MRU/list buffers (`:289-334`). This lets probes that scroll off screen be recovered later.
- Quantization: radiance accumulated in LDS with `InterlockedAdd` on `uint`, scale `1e4` (`screen_probes.hlsl:30`).
- Angular reuse threshold: `kGI1_AngleThreshold = cos(0.02*PI)` (`screen_probes.hlsl:27`).
- Cell size for plane tests: `distance(eye, p) * cell_size` where `cell_size = tan(fovY * probe_size * max(1/h, h/w^2))` (`gi1.cpp:1836-1841`), about 8 pixels projected.

### 4.2 World-space hash-grid radiance cache

- Constants struct: `HashGridCacheConstants` (`gi1/gi1_shared.h:56-88`). Buffer slot enums at `gi1_shared.h:90-150`.
- **Hierarchy**: bucket -> tile slot -> tile (a 2D slab of `tile_cell_ratio^2` = 8x8 cells at mip0, mips 4x4, 2x2, 1x1) -> cell. Cells per tile = 64+16+4+1 = **85** (`gi1.cpp:444-459`; the code asserts the mip chain stops at four levels, `:448`).
- **Default capacity** (`gi1.h:60-62`): `num_buckets = 1<<14 = 16,384`, `num_tiles_per_bucket = 1<<4 = 16`, so **262,144 tiles** and **22,282,240 cells** (`gi1.cpp:442-461`).
- Per-cell value: `uint2` = half4 (RGB radiance sum + sample count in w), `hash_grid_cache.hlsl:356-365`. Two such arrays: direct (`ValueBuffer`) and, with multibounce, indirect (`ValueIndirectBuffer`) (`gi1.cpp:487-507`).
- Per-cell update scratch: `uint[4]` per cell (`num_cells << 2` `uint`), one for direct and one for indirect (`gi1.cpp:523-547`). Quantization `1e3` (`hash_grid_cache.hlsl:32`, `:301-315`).
- Per-tile: one `uint` hash/checksum slot (`radiance_cache_hash_buffer_`, 0 means empty), one `uint` decay stamp (last-touched frame) (`gi1.cpp:475-478`), plus ping-pong packed tile index/count lists.
- Per-ray hit list (sized to `max_combined_ray_count` = 522,240 or 1,044,480 with multibounce): visibility (`float4`: packed instance+frontface, geometry, primitive, half2 barycentrics, `hash_grid_cache.hlsl:379-397`), cell index, query index, ray list (bit 31 = bypass-cache flag), multibounce BRDF+pdf `uint2` (half4) (`gi1.cpp:640-695`).
- **Memory at defaults (1080p, multibounce on)**, computed from the above: value buffers 2 x 170 MiB, update scratch 2 x 340 MiB, so **about 1.0 GiB** for the cache alone, before ray/visibility lists. Scratch is allocated for every cell even though only touched tiles are written each frame. An obvious port optimization (my inference, not in the source) is allocating scratch per update-tile slot instead of per cell. This size is the single biggest WebGPU risk (storage binding size and `maxBufferSize` limits).
- **Cell size vs distance** (`hash_grid_cache.hlsl:96-109`, `:140-143`): `step = max(distance(eye, hit) * cell_size_rad, min_cell_size)`; `cell = 1e-3 * exp2(floor(log2(1e3 * step)))`, a power of two (in millimetre scaling), so cells double in size at discrete distance rings. `cell_size_rad = tan(fovY * gi1_hash_grid_cache_cell_size * max(1/h, h/w^2))` (`gi1.cpp:1856-1859`), default option 32 (about 32 pixels of projected size, vs 8 for probes). Tile size = cell x 8. Example (my arithmetic, 60 degree vertical FOV, 1080p, hit at 10 m): about 0.31 m rounds to a 0.256 m cell and a 2.05 m tile.
- **Key** (`hash_grid_cache.hlsl:133-192`): level `l = floor(log2(...))`, signed tile coordinate `c = floor(hit / tile_size)`, **direction bin** `d = floor(0.5 + (0.5*dir + 0.5) * 4)` per axis (5 bins per axis, so the cache is direction-dependent), and a flag `t = hit_distance < tile_size`. Cell offset inside the tile is the hit's cell coordinate within its tile, projected along the dominant axis of the ray direction, which is why a tile is a 2D slab.
- **Hash functions**: bucket index = nested `pcgHash` chain over (l, c.xyz, d.xyz, t) modulo `num_buckets`; tile checksum = nested `xxHash` chain with `max(1, ...)` so 0 stays "empty" (`hash_grid_cache.hlsl:153-162`). The hash implementations are `src/math/hash.hlsl:33-39` (pcg) and `:102-` (xxHash).
- **Collision handling** (`hash_grid_cache.hlsl:249-278`): scan the bucket's 16 slots, `InterlockedCompareExchange(slot, 0, checksum)`; a returned 0 means this thread inserted a new tile, equal checksum means found, otherwise next slot. If all 16 slots hold other checksums, the insert **fails and the sample is dropped** (returns invalid, optional overflow counter under `DEBUG_HASH_STATS`). Lookups (`FindCell`, `:281-298`) are read-only, stop at the first empty slot.
- **Eviction**: tile decay stamp refreshed via `InterlockedExchange` on any touch (also on reflection reads, `gi1.comp:2888`); `PurgeTiles` zeroes the checksum after 50 idle frames (`hash_grid_cache.hlsl:29`).
- **Filtered readback** (`hash_grid_cache.hlsl:434-494`): start at mip0; while the sample count in w is below `max_sample_count` (16) and a coarser mip exists, take the next mip. This is how sparsely sampled cells get smooth values.

### 4.3 World-space ReSTIR (off by default)

- `kConstant_NumCells = 0x40000`, 16 entries per cell, 4M entries total (`gi1.h:316-321`); hash by `pcgHash` index and `xxHash` checksum (`world_space_restir.hlsl:62-76`); 8 candidate light samples (`kReservoir_SampleCount`, `world_space_restir.hlsl:28`); reservoir packed as `uint4` (`src/lights/reservoir.hlsl:545-558`). Buffers allocated at `gi1.cpp:880-1020`.

### 4.4 Reflection and denoiser buffers

- Reflections: half-res `specular` and `direction` (RGBA16F), `fireflies` (R8_SNORM), full-res `reflections`, `standard_dev`, ping-pong temporaries (RGBA16F) (`gi1.cpp:1023-1190`); RT sample queue = `uint` per half-res pixel (`:1062`).
- GI denoiser: `blur_mask` R8_SNORM, 2x color RGBA16F, 2x delta R16F (`gi1.cpp:1193-1240`); plus `GI_IrradianceBuffer` RGBA16F and a D32 depth buffer that only the debug draw uses (`:1706-1707`).

### 4.5 Packing used throughout (`src/math/pack.hlsl`)

Half-precision pairs via `f32tof16` (`packHalf2/3/4`, `:400-460`), 10-bit snorm normals (`packNormal`, `:468`, not octahedral), `packColor` (11/11/10 bit float-like, `:511`), instance index with front-face bit 31 (`hash_grid_cache.hlsl:379-383`).

---

## 5. Hardware dependencies and WGSL replacements

| Feature | Where | Notes | Likely WGSL replacement |
|---|---|---|---|
| **DXR inline ray queries** (`RayQuery`, `TraceRayInline`, `Proceed`, committed/candidate queries) | Wrapper `src/ray_tracing/trace_ray.hlsl:174-221`; hit info `src/ray_tracing/ray_tracing.hlsl:86-95`. Call sites in GI-1: closest hit `gi1.comp:1089` (probe rays), `:1857` (multibounce), `:2918` (reflections); shadow ray (accept first hit) `:2025`. Flags: `RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH` for shadows (`trace_ray.hlsl:175`, `:192`), instance mask `0xFF` | Alpha testing runs inside the traversal loop: for `CANDIDATE_NON_OPAQUE_TRIANGLE` the shader calls `AlphaTest` (`trace_ray.hlsl:45-85`, samples material textures and uses hashed alpha) and `CommitNonOpaqueTriangleHit()` (`:203-207`). `DISABLE_ALPHA_TESTING` skips this (`gi1/gi1.cpp:1417-1419`; the flag split is `src/ray_tracing/trace_ray.hlsl:171-209`). Rays per frame: 522k primary + about 0.3x that for bounce 2 + shadow rays (one per hit) + reflection rays. | Compute BVH traversal (our `packages/bvh`) returning closest/any hit with (instance, primitive, barycentrics, front-face, t). Alpha test as a callback inside the traversal loop or off by default (`gi1_disable_alpha_testing` option exists). |
| DXR 1.0 raygen/miss/hit + SBT + `DispatchRays` (`gi1_use_dxr10`) | `gi1/gi1.rt`, `gi1.cpp:1521-1600`, `:2337-2351` | Optional second path (default off). | Drop entirely. |
| **Wave intrinsics** | Only in the light sampler: `requestLightSampleLocation` uses `WaveActiveMin/Max`, `WaveIsFirstLane` (`light_sampler_grid_stream.hlsl:547-549`), called from `gi1.comp:1019` and `:1797`; `light_sampler_grid_stream_build.comp:151-181` (`WavePrefixSum`, `WaveActiveSum`, `WavePrefixCountBits`, `WaveGetLaneCount`), only for the opt-in parallel "many lights" build. GI-1's own kernels use none. `visibility_buffer.task:189-212` uses ballot/count bits (mesh shader, not ported). | Bounds collection is a once-per-wave write into a buffer of size `max_ray_count * (multibounce?2:1)` (`gi1.cpp:1710-1711`), reduced later by a 1-thread kernel. | `subgroupMin/subgroupMax/subgroupElect` under the `subgroups` feature, with the declared fallback of a workgroup-memory reduction or fixed-point `atomicMin/atomicMax` on ordered `u32`. For the parallel light build use `subgroupExclusiveAdd/subgroupAdd/subgroupBallot`, or just skip the opt-in path and run the serial build. |
| **64-bit atomics** | None found in GI-1 or its includes (grep for `uint64`, `Atomic64` in the gi1 dir returned nothing) | Float accumulation uses 32-bit integer fixed point (`1e3` cache, `1e4` probes). | `atomic<u32>` `atomicAdd` as is. Watch overflow headroom: `1e3 * radiance` summed over many rays per cell per frame; the design resolves accumulators every frame in `UpdateTiles` to avoid it (comment `gi1.comp:1039-1042`). |
| Other atomics | `InterlockedAdd` (append counters, accumulation), `InterlockedExchange` (decay stamp, patch), `InterlockedCompareExchange` (hash insert `hash_grid_cache.hlsl:259`, LRU flags `gi1.comp:876`, `:888`), `InterlockedMin` on LDS (`:348`), LDS `InterlockedAdd` (`:398`, `:752`, `:838`, `:1204`) | All `uint`. | `atomicAdd`, `atomicExchange`, `atomicCompareExchangeWeak`, `atomicMin` on storage and `var<workgroup>` atomics. All map directly. |
| **Bindless / descriptor arrays** | `Texture2D g_TextureMaps[] : register(space99)` (`gi1.comp:79`, `gi1.frag:50`) used for material textures at every hit-shading site and in alpha testing. Unbounded buffer arrays: `g_HashGridCache_BuffersUint/Uint2/Uint4/Float` (`hash_grid_cache.hlsl:52-55`, 24 uint buffers + 3 uint2 + 2 float4 + 1 float, enum list `gi1_shared.h:90-150`) and `g_GlossyReflections_TextureFloat/Float4[]` (`glossy_reflections.hlsl:30-31`). | The program binds about 108 named resources (`gi1.cpp:1982-2224`), though each kernel touches a subset. | Textures: one `texture_2d_array` atlas (same size/format layers) or a few size-bucketed arrays, plus a per-material layer index; or pre-bake per-instance albedo/emissive into the vertex/material buffer for hit shading and keep real textures only for the primary G-buffer. Buffer arrays: pack the hash-grid arrays into a few large storage buffers with fixed offsets per sub-array (limited storage buffer bindings per stage). Texture arrays: separate named bindings (ping-pong pairs). |
| **Typed UAV loads / read-write textures** | `RWTexture2D<float4>` read as well as written: `g_ScreenProbes_ProbeBuffer`, `g_ScreenProbes_PreviousProbeBuffer`, `g_ScreenProbes_ProbeCachedTileBuffer`, `...IndexBuffer` (`screen_probes.hlsl:41-46`, `:64-65`), `g_GIDenoiser_ColorBuffer` read at `gi1.comp:3999`, `g_GIDenoiser_BlurMask` read in `GIDenoiser_GetBlurRadius` (`gi_denoiser.hlsl:41-46`), reflection ping-pong textures; `RWTexture2D<uint>` probe masks. Formats RGBA16F, R16F, R8_SNORM. | Read-modify-write on the same texture appears in `ResolveCells` style accumulation only for buffers, not textures; textures are mostly read from one copy and written to another, but several kernels bind a texture as RW and read it. | Prefer `texture_2d` (sampled/`textureLoad`) for reads plus write-only `texture_storage_2d` for writes, using the existing ping-pong. Where true read-write is needed, use storage buffers of packed half (`pack2x16float`) since core WebGPU only guarantees `read_write` storage textures for `r32float/r32uint/r32sint`. Check the `texture-formats-tier1` style feature status in three r187 before relying on rgba16float read-write (unverified). |
| `[numthreads]` sizes and LDS | Sizes 1, 8x8, 32, 64, 128 (`gi1.comp` kernel list). LDS: `lds_ScreenProbes_ProbeSHBuffer[9*64]` float4 = 9,216 B, plus about 1 to 2 KB of radiance/count arrays; `UpdateTiles` 2 x 8x8 `uint2` = 1 KB (`screen_probes.hlsl:79-86`, `gi1.comp:2124-2126`). | Fits the 16 KB WebGPU default workgroup memory. | `var<workgroup>` arrays. `GroupMemoryBarrierWithGroupSync` -> `workgroupBarrier()`. |
| GPU-generated dispatch arguments | `GenerateDispatch` writes `DispatchCommand` (`gi1.comp:208-217`) consumed via `gfxCommandDispatchIndirect` | Used for 15 or so passes. | `dispatchWorkgroupsIndirect` with a buffer of `INDIRECT | STORAGE` usage (three.js r187 exposes indirect compute dispatch; confirm). Also `GenerateDraw` for debug (`gi1.comp:187`) which we can skip. |
| Prefix sum | `gfxCommandScanSum` x4 | Host library kernel. | Own WGSL scan (workgroup scan plus block sums; `subgroupExclusiveAdd` path when `subgroups` is available). |
| Hardware mesh shaders | Visibility buffer `visibility_buffer.mesh/.task` | Outside GI-1 but required input. | Classic vertex+fragment raster writing a visibility target; WGSL fragment has no guaranteed primitive id builtin, so the engine adapter should write an instance/primitive id some other way (vertex pulling with a per-triangle id, or the compute RT fallback pattern in `visibility_buffer_rt.comp`). |
| `asuint/asfloat` bit casts | Visibility texture stores integer ids in float channels; hash cell packing | | `bitcast<u32>/bitcast<f32>`. Or use `rgba32uint`/`rgba32float` consistently. |
| Float -> int hashing of tile coordinates | `asuint(int3(floor(hit/tile)))` (`hash_grid_cache.hlsl:149`) | Hash correctness needs the same float math in all passes; exp2/log2 cell-size quantization sits at a power-of-two boundary. | Same code; keep it in one shared WGSL function so every pass uses identical arithmetic. |
| Indexing resources by `g_HashGridCacheConstants.buffer_ping_pong` | `hash_grid_cache.hlsl:75-78` | Resource array index chosen at run time. | Two bind groups or an offset into a packed buffer. |

---

## 6. Tunables

### 6.1 GI-1 options (`gi1/gi1.h:48-89`; registered at `gi1/gi1.cpp:1244-1289`)

| Option | Default | Notes |
|---|---|---|
| `gi1_use_dxr10` | false | false = compute + inline ray queries |
| `gi1_use_resampling` | false | world-space ReSTIR reuse of light samples |
| `gi1_disable_alpha_testing` | false | mapped from `visibility_buffer_disable_alpha_testing` (`gi1.cpp:1295-1297`) |
| `gi1_use_direct_lighting` | true | enables emissive + sky contribution through the probes (not full direct lighting; see section 7) |
| `gi1_use_temporal_feedback` | false | reuse previous final color at hits (`gi1.comp:2458-2490`) |
| `gi1_use_multibounce` | true | second-bounce cache chain |
| `gi1_disable_albedo_textures` | false | debug: albedo 0.3, F0 0 (`gi1.frag:114-120`) |
| `gi1_disable_specular_materials` | false | removes reflection passes (`gi1.cpp:2640`) |
| `gi1_hash_grid_cache_cell_size` | 32.0 | pixels-ish, see 4.2 |
| `gi1_hash_grid_cache_min_cell_size` | 0.1 | world units, floor for `step` |
| `gi1_hash_grid_cache_tile_cell_ratio` | 8 | tile = 8x8 cells |
| `gi1_hash_grid_cache_num_buckets` | 14 | exponent, 2^14 buckets |
| `gi1_hash_grid_cache_num_tiles_per_bucket` | 4 | exponent, 2^4 = 16 slots per bucket |
| `gi1_hash_grid_cache_max_sample_count` | 16.0 | direct cell sample cap and mip-up threshold |
| `gi1_hash_grid_cache_discard_multibounce_ray_probability` | 0.7 | bounce-2 ray survival 0.3 |
| `gi1_hash_grid_cache_max_multibounce_sample_count` | 16.0 | |
| `gi1_hash_grid_cache_debug_mip_level` | 0 | debug |
| `gi1_hash_grid_cache_debug_propagate` | false | debug |
| `gi1_hash_grid_cache_debug_max_cell_decay` | 0 | debug |
| `gi1_hash_grid_cache_debug_stats` | false | debug |
| `gi1_hash_grid_cache_debug_max_bucket_overflow` | 64 | debug |
| `gi1_reservoir_cache_cell_size` | 16.0 | ReSTIR cell size |
| `gi1_glossy_reflections_halfres` | true | reflection rays at half res |
| `gi1_glossy_reflections_denoiser_mode` | 1 | 0 split ratio estimator, 1 A-trous ratio estimator, 2 none (`gi1.h:74`) |
| `gi1_glossy_reflections_cleanup_fireflies` | true | |
| `gi1_glossy_reflections_low_roughness_threshold` | 0.2 | below: traced reflections |
| `gi1_glossy_reflections_high_roughness_threshold` | 0.6 | above: diffuse fallback |
| `gi1_glossy_reflections_atrous_pass_count` | 4 | |
| `gi1_glossy_reflections_full_radius` / `half_radius` | 11 / 11 | resolve kernel radius |
| `gi1_glossy_reflections_mark_fireflies_half_radius` / `full_radius` | 3 / 2 | |
| `..._mark_fireflies_half_low_threshold` / `full_low_threshold` | 0.0 / 0.0 | |
| `..._mark_fireflies_half_high_threshold` / `full_high_threshold` | 1.0 / 1.0 | |
| `..._cleanup_fireflies_half_radius` / `full_radius` | 2 / 1 | |

Only a subset is exposed in the GUI: multibounce, resampling, direct lighting, albedo and specular toggles, hash bucket/tile exponents, and debug stats (`gi1.cpp:3188-3260`).

### 6.2 Compile-time constants

| Constant | Value | Where |
|---|---|---|
| probe size | 8 | `gi1.h:191` |
| sampling mode (rays per pixel) | quarter spp, spawn tile 16 | `gi1.h:192`, `:195-198` |
| tile decay | 50 frames | `hash_grid_cache.hlsl:29` |
| cache float quantization | 1e3 | `hash_grid_cache.hlsl:32` |
| probe float quantization | 1e4 | `screen_probes.hlsl:30` |
| direction-bin resolution | 4 (5 bins per axis) | `hash_grid_cache.hlsl:146` |
| hash cell step/size factors | 1e3 / 1e-3 | `hash_grid_cache.hlsl:92-93` |
| probe angle threshold | cos(0.02 pi) | `screen_probes.hlsl:27` |
| probe filter radius | 3 (kRadius) | `gi1.comp:1393` |
| probe reproject tap radius | 1 (3x3 taps) | `gi1.comp:689`, `:780` |
| RIS candidates per hit | 8 | `world_space_restir.hlsl:28` |
| reservoir spatial jitter / bilateral threshold | sqrt(2) / 0.5 | `world_space_restir.hlsl:31`, `:34` |
| ReSTIR hash | 0x40000 cells x 16 entries | `gi1.h:316-321` |
| max ray distance | 1e9 | `gi1.comp:127` |
| GI denoiser max blur | 8 | `gi_denoiser.hlsl:26` |
| GI denoiser history cap | lerp(4, 8 x min(count, 8), alpha) x vignette | `gi1.comp:4092` |
| light grid | 16 cells max per axis, 64 reservoirs per cell, no octahedron/resample/parallel/centroid/overlap, merge type 1 | `src/components/light_sampler_grid_stream/light_sampler_grid_stream.h` options block |

---

## 7. Direct lighting and light sampling used for hit shading

- **What GI-1 does for "direct" light at a bounce vertex**: it does **not** shade a full material at the hit. It stores per-cell *outgoing radiance toward the ray direction bin*, built from one shadow ray to one light sampled per hit (`gi1.comp:2040-2100`) with the hit's BRDF evaluated toward the origin (`:2006-2020`) from a packed approximation of the material (`packMaterial`, comment at `:2597`: "cannot afford to re-fetch the full material"). Secondary vertices of the multibounce chain are forced diffuse (roughness 1, metallic 0) to cut noise (`:1911-1916`).
- **What `gi1_use_direct_lighting` really means**: emissive meshes and the sky only: miss -> environment cube map (`gi1.comp:1078-1083`, `:2905-2912`), emissive hit -> area emissivity written straight into the probe ray, bypassing the cache so the emitter is not blurred by cell size (`gi1.comp:2439-2456`). README confirms GI-1 is an indirect solution that expects a separate direct-lighting technique (`root/README.md:28-32`).
- **Light sampler**: `LightSamplerGridStream` (`gi1.cpp:1408`, `src/components/light_sampler_grid_stream/`). Model: a 3D grid over the bounds of this frame's requested hit positions (at most 16 cells on the largest axis, `light_sampler_grid_stream_bounds.comp:46-80`), each cell storing 64 streamed weighted reservoirs (`LightSamplingConfiguration`, `light_sampler_grid_stream.h` options). Build (`light_sampler_grid_stream_build.comp:54-170`): each reservoir streams every 64th light (`for (i = reservoirID; i < totalLights; i += reservoirsPerCell)`), weighting by `sampleLightVolume` (a cell-vs-light importance), keeping one light plus total weight. Query (`light_sampler_grid_stream.hlsl:sampleLights`, `:343`): pick the cell by jittered position, stream the cell's 64 reservoirs into one pick with PDF = target/total (`getSampledLight`), do that 8 times, then RIS in `updateReservoirCone` using the material and a cone-based solid angle to produce one `Reservoir` per hit.
- **How many lights**: no hard cap in the sampler; cost is `cells x lights` at build, each of the 64 reservoirs touching `lights/64` entries (opt-in `light_grid_stream_parallel_build` spreads this across a 128-thread group with wave reductions). Supported types: point, spot, direction, environment, area (triangle emissive) (`src/lights/lights_shared.h` `LightType`: `kLight_Point`..`kLight_Area`). The environment light, if present, is light index 0 (`src/lights/lights.hlsl:hasEnvironmentLight`).
- Evaluation of the chosen light: `evaluateLightConeSampledAM` with a solid angle `4pi / (8 * 12000)` when evaluating the stored sample (`gi1.comp:2071`) or `4pi / (8 * 1.5)` when generating it (`gi1.comp:2504`, `:2650`), plus an optional "visibility" estimate stored in the reservoir.

---

## 8. Differences from or beyond the paper (inferred from code and README)

I did not diff against the paper text, so each item below is "present in v1.3 code" and may or may not be in arXiv 2310.19855. Items I am least sure are in the paper are marked (?).

1. Multibounce through the cache: bounce-1 cells accumulate bounce-2 cell radiance times BRDF/pdf with a random-discard probability (`gi1.comp:1869-1990`). (?)
2. Glossy reflections (GI-1.1): half-res traced reflections, firefly mark/cleanup, ratio-estimator A-trous denoiser, split-sum composite (`gi1.comp:2801-3995`, `gi1.frag:76-150`). The README attributes this to a separate GI-1.1 paper (`root/README.md:24`).
3. SSGI near-field + bent-normal AO fed into probe interpolation (`HAS_OCCLUSION`, `gi1.cpp:1421-1423`; SSGI technique). (?)
4. Off-screen probe cache with LRU/MRU lists so probes survive leaving the screen (`screen_probes.hlsl` cached tile buffers; `gi1.comp:501-541`, `:1334-1355`). (?)
5. Adaptive sampling by patching disoccluded "empty" tiles with probes taken from stable tiles (`gi1.comp:442-448`, `:601-643`). (?)
6. Mip chain of 4 levels inside every cache tile with sample-count-driven level selection (`hash_grid_cache.hlsl:434-494`, `gi1.comp:2216-2312`).
7. Direction-dependent cache keys (5 bins per axis) and a distance flag in the key (`hash_grid_cache.hlsl:145-151`). (?)
8. Light-leak guards: bypass the cache when the secondary ray is shorter than a cell (`gi1.comp:2538-2561`, `:2674`) and emissives bypass the cache (`:2439-2456`).
9. Optional world-space ReSTIR with temporal reuse of the light reservoirs (`gi1.comp:2400-2800`, `world_space_restir.hlsl`), default off.
10. Optional temporal feedback of the previous final color into the cache (`g_UseTemporalFeedback`, `gi1.comp:2458-2490`), default off.
11. "Shadow-preserving biased temporal hysteresis" for probe blending, with a YouTube talk cited in the code comment (`gi1.comp:1265`).
12. Image-space GI denoiser with disocclusion-driven blur mask and luminance-delta adaptive history (`gi1.comp:3997-4156`).
13. Light-grid streaming reservoir sampler instead of a plain uniform or CDF light sampler (the framework also ships `LightSamplerGridCdf` and `LightSamplerUniform` alternatives, `src/components/light_sampler_*`).
14. Nothing in the code uses wave/subgroup operations or 64-bit atomics in the GI-1 kernels themselves, even though the paper context is GPU-specific; everything is plain compute plus inline ray queries.

---

## 9. Size estimate (lines at v1.3; counts from `wc -l`, so comments and blank lines are included)

### GI-1-specific

| Part | Lines |
|---|---|
| Shaders under `gi1/` (comp 4,269, frag 215, vert 265, rt 182, 5 hlsl 1,177, shared.h 213 = 6,487 by the directory count) | 6,487 |
| Of which off by default or droppable (DXR `.rt` 182; ReSTIR kernels + `world_space_restir.hlsl` about 650; debug-stats and debug draw kernels about 300) | about 1,100 |
| Host C++ (`gi1.cpp` 3,293 + `gi1.h` 472 + renderer 74), reimplemented as TS orchestration in our design | about 3,840 |

So the portable, useful GI-1 shader core is on the order of **5,000 lines of HLSL**, mostly `gi1.comp`.

### Shared framework we would have to replicate (shader side; C++ in parentheses)

| Piece | HLSL/comp lines | C++ lines |
|---|---|---|
| Light sampler grid stream | 862 | 756 |
| Light builder | 325 | 812 |
| Light sampler base/switcher | 23 | 501 |
| Lights library (`lights/*`: evaluation, sampling, volume sampling, reservoir) | 4,408 | 0 |
| Materials (`materials/*`) | 1,763 | 0 |
| Geometry/mesh fetch | 642 | 0 |
| Math (`math/*`; GI-1 uses pack, hash, sampling, SH, color, transform) | 3,025 (partial need, perhaps 1,500) | 0 |
| Ray tracing glue (`trace_ray`, `ray_tracing`, `intersect_data`, not `path_tracing*`) | about 650 | 0 |
| Visibility buffer (raster + RT fallback) | 987 | 1,074 |
| SSGI (near field / bent normals) | 346 | 391 |
| Blue noise + RNG | 362 | 484 |
| BRDF LUT + prefiltered IBL | 197 | 370 |

Rough totals: shared shader code that GI-1 touches is about 10,000 to 14,000 lines if copied wholesale, but we need only a subset (we already own our BVH, scene buffers and likely a material system; `light_sampling_volume.hlsl` at 997 lines and the area-light sampling routines are the big optional chunks). The practical new surface for us: the GI-1 core (about 5,000), the grid light sampler + builder (about 1,200), a visibility-buffer producer in the three.js adapter, SSGI (optional, about 350), and utilities (scan, indirect dispatch plumbing).

---

## 10. Port risks, in priority order (summary of the above)

1. Ray traversal: four inline-ray-query sites plus alpha test in the loop (section 5). Needs our compute BVH to expose closest-hit and any-hit with front-face and barycentrics.
2. Hash-grid cache footprint (about 1 GiB at defaults) and its unbounded buffer-array binding style (`hash_grid_cache.hlsl:52-55`).
3. Bindless material textures at every hit-shading site (`gi1.comp:79`).
4. A deep chain of GPU-driven, count-dependent dispatches, four prefix scans, atomics-based compaction and LDS-heavy kernels, in one program with about 108 named resources.
5. Read-write typed textures and integer-in-float visibility buffer, plus an upstream mesh-shader visibility buffer, SSGI and previous-frame lit color that GI-1 assumes exist.
