# Where our DDGI differs from RTXGI, and why

For Split Radiance Cascades against the authors' Chandra, see [radiance-cascades/divergence.md](radiance-cascades/divergence.md).

Written 2026-11-11 by the lead (Mac). The reference is NVIDIA's RTXGI-DDGI SDK 1.3.7 as it sits in our fork (`native/rtxgi-ddgi`, SDK plus upstream fix a6c4d99), and its sample app (`samples/test-harness`), which holds the probe-trace shader, hit shading, lighting and the screen-space gather. Behind both are the two papers: Majercik et al. 2019 ("Dynamic Diffuse Global Illumination with Ray-Traced Irradiance Fields") and Majercik et al. 2021 ("Scaling Probe-Based Real-Time Dynamic Global Illumination"), which added relocation, classification and variability.

**How this was built:** from reading the SDK's source next to ours, and from our own records (DECISIONS, NOTES, the design notes). No new measurements were taken. Impacts quoted with numbers come from earlier runs and cite where they were recorded. Impacts without numbers are the lead's estimate, and say so. Our edits to the fork (`DDGI_CAPTURE_*`, the harness's `skipConverged`) all default to stock behaviour, so "RTXGI does" means the SDK as shipped.

**Paths:** `SDK/` = `native/rtxgi-ddgi/rtxgi-sdk/`, `HAR/` = `native/rtxgi-ddgi/samples/test-harness/`, `DDGI/` = `packages/ddgi/src/`, `PLAT/` = `packages/core/src/platform/`.

**Columns used below:**
- **Quality** and **Speed:** what changes for the picture and for GPU time and memory. *Better*, *worse* or *same*, with a size where known.
- **Code:** how much of ours it touches.
- **Effort:** XS (an hour or two), S (one small task), M (a task of several days with checks), L (a milestone-sized piece), XL (a new subsystem).
- **WebGPU:** whether RTXGI's way can be done in WebGPU today, in core or behind an optional feature.

The page version, with filters and a column for Dennis's call per row, is the "DDGI vs RTXGI" artifact. This file is the source of truth.

---

## 1. Worth trying, ranked

The things RTXGI does that we don't, or does differently in a way that might be better. Ranked by likely gain for the effort, with risk weighed in. Each one is an experiment first: build it behind a setting, run the checks named, then decide (DECISIONS) whether it becomes the default.

| Rank | ID | What | Quality | Speed | Effort | WebGPU | Checks that decide it |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | DV-01 | **Done, off by default** (TU.5, 2026-11-19; `backFaceRayThreshold`, DECISIONS 2026-11-19). Leave back-face rays out of the irradiance average; skip a probe's update past 10% back faces | Probes near or partly inside geometry stop being darkened by black samples. Could brighten corners and contact areas, and could add leaks (estimate) | same | S | yes, core | `ddgi-furnace`, `ddgi-leak`, `ddgi-door-swing`, the textured furnace and furnished room against Cycles. **Result (RTX 3060):** leak 0.836 / 0.603 / 0.060% with and without, furnace E/π 0.49925 both, no update skipped on Cornell or two-room; Sponza against Cycles (no GTAO) +25.46 → +31.50% and +21.77 → +22.99%, brighter on the curtains, so off |
| 2 | DV-02 | Stop updating probes that have settled while nothing changes (RTXGI skips a converged volume entirely) | same, if unreported changes are still caught | Big on still scenes: GI's GPU cost falls toward zero instead of a steady budget. Matters most for battery and the `low` devices (estimate) | M | yes, core, done on the GPU with no readback | `ddgi-flicker`, `ddgi-change-step`, `platform-recompiles`; GI time on a still Sponza before and after |
| 3 | DV-03 | **Done** (TU.5, 2026-11-19): `giProbeIrradiance` in DDGI's shading module, named by the `points` view's `irradianceFunction`; the viewer's "Colour the probes by: Its own light" (`probecolor=irradiance`), with the state colour on the rim as RTXGI does. Probe spheres shaded with the probe's own irradiance (RTXGI's standard debug view) | debug only | same | S | yes, core | the viewer, a screenshot looked at. **Result (RTX 3060):** two-room's lit-room probes light, its dark-room probes dark, off probes black with a blue rim; Sponza's probes follow its light. Screenshots `bench/screenshots/tu5-dv03/` and `tu5-dv03-sheet/` (gitignored) |
| 4 | DV-04 | Blend irradiance in gamma-5 space and interpolate between probes in square-root space | Darkening converges faster (RTXGI's stated reason). The blend between a bright probe and a dark one comes out darker, which might pull Sponza's +8 to 12% back (a guess, unmeasured) | same | M | yes, core | furnace and leak; Sponza and furnished room against Cycles; `ddgi-change-step` (our change-history math assumes linear values); saved bakes (format version 2) |
| 5 | DV-05 | Store irradiance as 10-bit `rgb10a2unorm` (RTXGI's default), after DV-04 | Needs DV-04's encoding, or dark tones band | Irradiance atlas memory halves (8 to 4 bytes a texel). Sponza's probes use 58 MB at `high` in all; the irradiance share wasn't measured | M | storage use needs the optional `texture-formats-tier1`. Keep `rgba16float` without it | furnace (10-bit stall), `ddgi-flicker`, memory per scene |
| 6 | DV-06 | **Done, off by default** (issue #20, 2026-11-19; `samplingWeightFloors`, DECISIONS 2026-11-19). Floors on the sampling weights (0.001 per trilinear axis, 1e-6 before the crush) | Avoids rare black spots where all eight probes weigh near zero. Our coverage fallback already catches most. Slight leak risk | same | XS | yes | `ddgi-leak`, `ddgi-shading-consistency`. **Result (RTX 3060):** leak 0.83616 / 0.60279 / 0.060412% off, 0.83679 / 0.60277 / 0.060414% on (two-room's rise is real: the off run repeats exactly); consistency passes both. Two-room got worse by a hair, so off |
| 7 | DV-07 | **Done** (issue #20, 2026-11-19; `ddgiSetAsideRays`). Set aside the 32 fixed rays only when relocation or classification is on | same | With both off (not the default), 12.5% more of each probe's 256 rays count | XS | yes | furnace with both off. **Result (RTX 3060):** with both off, E/π 0.49687 → 0.49686 (mean error 0.626 → 0.628%, worst point 6.36 → 5.99%), PASS both; the default path (both on) unchanged |
| 8 | DV-08 | **Done** (issue #20, 2026-11-19; `giBounceAlbedo`, DECISIONS 2026-11-19). Apply `bounceAlbedoCap` to the bounce term only, as RTXGI does, and fix the `bounceEnergy` comments (§6) | Only matters when matching native. Defaults stay 1.0 | same | XS | yes | native parity (T3.17, not run). `hit-shading-lights` checks the cap on `giBounceAlbedo` and not on the surface's albedo; at the default 1.0 the furnace and leak are unchanged |
| 9 | DV-09 | Rotated volumes (RTXGI's `eulerAngles`) | A host with a rotated building can fit probes to it | same | M | yes | a rotated two-room in `ddgi-leak` and `ddgi-volumes` |
| 10 | DV-10 | Normal maps at ray hits (RTXGI's sample uses them for the sun) | Small for diffuse GI (estimate) | slightly worse | M (the scene copy needs tangents, the texture store a normal layer) | yes | Sponza against Cycles |
| 11 | DV-11 | Blend in place in the atlas (RTXGI's way), dropping our scratch buffer and the `ddgi/write` pass | same | One dispatch and the scratch memory less (small) | M | needs the optional `texture-formats-tier2` for read-write `rgba16float`, so the scratch path stays as the fallback: two GPU paths | GI time; every DDGI check on both paths |
| 12 | DV-12 | **Done** (issue #20, 2026-11-19; Dennis: "yes build it"; DECISIONS 2026-11-19). Separate "reset offsets" and "reset states" controls: `DDGITechnique.resetProbes({ offsets?, states? })` and three's `DDGIVolume.resetProbes`, which keep irradiance, distances and history; `{}` does nothing. `resetHistory()` (the engine-neutral reset that forgets everything) is unchanged. One small GPU pass in the next frame (`ddgi/reset-probes`), since offset and state share the probe-data texel. *Differs from RTXGI:* `states` brings only the off probes back, through "not updated yet" (our blend kernel's own way back on), instead of making every probe active, so shading never reads an off probe's stale tiles | same | same | XS | yes | unit tests (`probe-reset.test.ts`); `ddgi-classification` check 8. **Result (RTX 3060):** on Cornell (1,728 probes) and the cabinet row (3,528), each of `{}`, offsets, states and both left every probe as the rule says, changed 0 irradiance and 0 distance bytes in the reset frame, and the points view agreed. 120 frames later both resets together came back to the settled field exactly (1,728 and 3,528 the same); offsets alone left 15 and 206 probes at their grid points, the ones off for being inside geometry, which relocation doesn't move again (documented) |
| watch | DV-13 | Hardware ray tracing | same | Large: native traces about 214 M rays/s on the RTX 3060; our probe shadow rays run at most about 1/7 of the bench's any-hit rate (NOTES:498-500) | XL, once it exists | **No.** WebGPU has no ray tracing or ray query API, and no browser has an intent to ship one. The tracer is a plug-in point (SPEC:228) | |

**Not ranked, because it's housekeeping:** the doc errors in §6, each XS. All three are fixed (2026-11-18 and 2026-11-19, issue #20).

---

## 2. Forced by WebGPU or the browser

RTXGI's way isn't available, so we do something else. These close only if WebGPU changes.

| What RTXGI does | What we do instead | What it costs | Could it change? |
| --- | --- | --- | --- |
| Hardware ray tracing (DXR or Vulkan RT: ray generation, hit and miss shaders, TLAS and BLAS) | A compute-shader walk of a two-level BVH, watertight triangle test, a 64-entry stack. Trees from tinybvh (WASM) or our GPU H-PLOC (`PLAT/trace/wgsl.ts`) | Speed: see DV-13. Closest hits on Sponza trace 1.20× faster than three-mesh-bvh on the RTX 3060 (DECISIONS:626). Code: the whole tracer and builders are ours | Only if WebGPU gains ray queries (DV-13) |
| Blends in place: reads and writes one `RWTexture2DArray` | Each update writes a per-probe scratch slot, and a `ddgi/write` pass copies it into the atlases with the borders | One extra dispatch and the scratch memory | Partly, with `texture-formats-tier2` (DV-11) |
| Bindless resources and root constants | One shared atlas for every volume (with layer bases), a per-volume uniform block, a volume table texture | At most 8 volumes (`volumes.ts:39`). A shading module may bind at most 6 textures | Only if binding arrays come to WebGPU |
| Probe ray data in a texture | 8-byte records in a storage buffer (3 half floats of radiance and a signed half-float distance) | Misses store 60,000 instead of 1e27, so the distance fits a half float (`trace.ts:89`) | Not needed |
| Any binding size | The defaults are 8 storage buffers per stage and 128 MB per binding, so big scenes ask the adapter for raised limits | Sponza with curtains needs about 276 MB of tracer records (DECISIONS:553), and Bistro exterior with two volumes 136 MB (NOTES:151). A device that can't raise them can't load those scenes | Revisit if a target device allows much less (DECISIONS:554) |
| `float32` atlases if wanted | Half-float atlases, because `float32-filterable` isn't assumed. `rg16float` distance needs `texture-formats-tier1`, else `rgba16float` at twice the memory | A host lighting in real lux (above 65,504) overflows: a known limit, not handled (ddgi.md:63) | No |
| Any ray count; shared memory sized at compile time | 64 to 512 rays a probe, in steps of 32 | 512 rays fit the 16 KB default workgroup memory | Only with a higher limit |
| Indirect dispatch, where it helps | None in the update loop | Each `dispatchWorkgroupsIndirect` costs about 0.022 ms even when empty, from Dawn's validation (DECISIONS:301) | If Dawn gets cheaper |
| Wave intrinsics wherever wanted | Subgroup code first, with a tested workgroup-atomics fallback (DECISIONS:590-601) | One kernel shape for subgroup sizes 8 to 32 | Already the best available |
| Exact float maths | Triangle edges in a fixed corner order, since Dawn's relaxed Metal maths fuses multiply-adds and opened shared edges (2,696 slips, fixed 2026-11-10) | A few compares per edge (TASKS Later: re-check on the RTX and Intel) | No |

---

## 3. We chose differently, and keep it

Deliberate choices, each made on a measurement. Revisit only if a later check shows otherwise.

| ID | RTXGI | Ours | Why, with the evidence | Cost of ours |
| --- | --- | --- | --- | --- |
| K-01 | Fixed hysteresis 0.97 for every texel | Per-probe history that knows the time between updates: τ 5 s, cap 200, warm-up ramp (ddgi.md:210-266). `adaptiveHysteresis: false` gives RTXGI's rule | Cornell's flicker went from 0.56% to 0.095% a frame; Sponza's restart to within 1% went from 44 s to 3 s; Sponza after halving the sun went from 162.5 s to 22.5 s (DECISIONS:336). We update a budgeted few probes a frame, so a fixed 0.97 means very different times per probe | Sparse probes move more per update: Sponza about 7× per frame against 0.97 (ddgi.md:265) |
| K-02 | Per texel: darkening by more than 0.25 cuts hysteresis by 0.75; black texels restart at 0 | The scene says what changed (`ddgi/mark`, from light, sky and geometry changes), and the history of the probes it reaches is cut. The noise-based threshold exists but is off | Every variant of a test on the noisy estimate biased the field: +8% per texel, +22% on the probe mean, Cornell (DECISIONS:485). With the scene's own changes, two-room's lamp is within 5% from the first update | Needs the host's changes. The three adapter tracks them itself |
| K-03 | Brightening by more than 0.1 luminance (gamma space) is damped to a quarter | ×2 per-channel clamp on the change per update, off for a probe taking in a reported change | So one lucky ray can't flash a probe (ddgi.md:137). Off for reported changes because it held two-room's lamp dark for 7 frames after switching on (DECISIONS:313). Never compared with RTXGI's rule | Fireflies up to 2× per update can pass |
| K-04 | Darkening steps at least 1/1024 in gamma space, so 10-bit storage reaches its target | Stochastic rounding to the neighbouring half floats on every atlas write | With round-to-nearest, the furnace stalled 4.6% low (DECISIONS:502) | A little noise in the last bit |
| K-05 | Each ray's radiance clamped to [0, 1] and packed to 10 bits | Unclamped half floats | RTXGI can't carry a sunlit wall's radiance above 1 through a ray. That loses energy in bright scenes, and is one suspect in native's Sponza mismatch (TASKS Later, T3.17) | none: both use 8 bytes a ray |
| K-06 | No emission at probe hits (the sample's probe hit shader drops it) | Emission added to the ray's radiance | SPEC v1 needs emissive bounce; the folding emissive panel brightens the floor 28.5% (M3-M4 note) | none |
| K-07 | Sky is one constant colour | Environment map for misses; SH bands 0-2 and 4 for sky irradiance | Open-ground sky irradiance within ±0.5% of Cycles (DECISIONS:534-536) | A texture read per miss |
| K-08 | Bounce albedo capped at 0.9; native setups dim bounces | `bounceAlbedoCap` and `bounceEnergy` default 1.0 | At 0.95 the furnace settles 4.8% low and fails its 2% check (ddgi.md:443); 0.9 moves indirect light by about 23% (NOTES:528) | none; set them to match native |
| K-09 | Every probe of every volume, every frame | A budget (Budget v2: probes per frame from a calibration, milliseconds as a ceiling), a GPU scheduler (priorities plus a scrambled round robin), key probes and propagation | Sponza has 19,425 probes; updating all at 256 rays isn't affordable in a compute tracer. Propagation removed 63.1% of a sun turn's change at held fine probes (DECISIONS:292) | A change reaches far probes later. See DV-02 for the other half of RTXGI's idea |
| K-10 | "No surface nearby" reaches one spacing; inside means more than 25% back faces | Reach 1.5 spacings; also inside if the nearest hit is a back face | The one-spacing rule switched off 25 probes Cornell needs (DECISIONS:492); native's rule switches off 40% of Cornell and 59% of Sponza (NOTES:206). The extra rule catches thin overlapping boards | 3 to 5 probes stay on up to 3.4 mm inside a board (DECISIONS:494) |
| K-11 | Relocation: minimum front-face distance 1 m; up to a 1 m step; ellipsoid limit | 0.02 of the smallest spacing; the shortfall as the step; a box limit, kept inside the volume | Two-room's leak went from 1.08% to 0.86% (DECISIONS:490), then to 0.78% with the volume limit (DECISIONS:491) | none measured |
| K-12 | Two probe states | Four: active, off inside, off with no surface nearby, not updated yet; a probe coming back is used after its second update | No popping when a probe turns back on | A little more bookkeeping |
| K-13 | Probe counts set directly; probes at cell centres | A box size; `floor(size/spacing)+1` probes, centred | M1's design choice (ddgi.md:24); the reason wasn't recorded, and the two layouts were never compared for quality | Counts differ from native at the same spacing (two-room 1,071 against 918, NOTES:206); `bench.ps1 -GridRule ddgi` matches them |
| K-14 | Several volumes: the sample's gather multiplies earlier sums by later weights (a bug), no priority | Priority order, each volume's share is coverage × what's left, at most 2 volumes a point; the renderer's environment fills what no volume covers | A dense volume inside a coarse one shows no seam (`ddgi-volumes`) | none |
| K-15 | A volume fades over 1 spacing past its outer probes | 0.5 spacing for fixed volumes; 10 spacings for the camera volume | The camera volume's edge change fell from 0.59% to 0.074% a frame (DECISIONS:43) | The camera volume is the only volume within about 7 m of the camera |
| K-16 | Scrolling clears one plane per axis per update and keeps a recycled probe's offset and state | Every entering plane reset (state, offset, history); a teleport resets every slot; the valid edge advances over warm planes | Handles moves of several planes a frame; no stale offsets | A teleport warms everything (known miss: 3-4 s at full rate, DECISIONS 2026-11-10) |
| K-17 | Probe-ray bounces read only the probe's own volume | Read every volume, blended by priority | Bounces near a nested volume use the better field | One more volume lookup at most |
| K-18 | Every light at every hit, one hard shadow ray each | Up to 8 lights exact; past that, a world-space light grid picks candidates (ReGIR-style); soft shadows from the light's radius; shadow rays in their own pass | Scales to 1,024 lights (`hit-shading-many-lights`); Sponza probes at 2 ms went from 293 to 646 (NOTES:499) | Noise with many lights; the 1,024 candles cost 2.2-2.5× per probe update against a 1.3× bar (known miss) |
| K-19 | Point lights fall off as 1/max(d, 1)², with RTXGI's own cone | three.js r187's falloff and cone | GI has to agree with the host's direct lighting | Differs from native within 1 m of a light (NOTES:214) |
| K-20 | Texture mip at hits: half the levels; alpha cut-outs only | Mips chosen by the ray cone's spread; blended surfaces passed at random in proportion to their alpha; glass and single-sided surfaces in the walk | M4's materials (DECISIONS:68, 72) | none measured |
| K-21 | Ray rotation per volume, from a time-seeded generator | One rotation per frame for every volume, seeded by the frame number | Runs repeat, which our checks rely on | Overlapping volumes share directions in a frame (no effect seen) |

---

## 4. Things we added that RTXGI doesn't have

None of these exist in the SDK or its sample. Each has its design note.
- **Change-driven history** (`ddgi/mark`): the scene's light, sky and geometry changes cut the history of the probes they reach (ddgi.md:268-296).
- **Key probes and propagation:** a coarse level updates first and fine probes follow by propagation (ddgi.md:304-347).
- **The feedback rule:** a probe that changed more than 6× its noise gets another update, along with its neighbours (ddgi.md:297-303). This is our closest match to RTXGI's variability.
- **Light groups and the sun layer:** separate irradiance layers, composed by weight (DECISIONS:151-158, 210-219).
- **The camera-following volume** with its wide fade, and **several volumes**, each with its own scheduler.
- **Probes woken near movers**, the deforming refit with a rebuild after drift, and GPU tree builds at runtime.
- **Saved bakes** (`.ddgibake`), the **LightProbeGrid export**, **quality presets**, the warm-up, the budget floor and the census counters.

---

## 5. RTXGI sample bugs we don't copy

These matter when comparing with native renders (T3.17, native parity), since native output includes them:
- **Point lights:** the loop always loads the first point light (`HAR/shaders/include/Lighting.hlsl:126`).
- **Lights out of range or shadowed:** spot and point lighting returns 0 for every light once any one is out of range or shadowed (`Lighting.hlsl:91, 135, 141`).
- **Shadow-ray origin:** it ignores the view bias, which is a TODO in the code (`Lighting.hlsl:47`).
- **Multi-volume gather:** it multiplies the running sum by each volume's weight (`HAR/shaders/IndirectCS.hlsl:73-108`).
- **The NVAPI/SER trace path:** it culls back faces, which turns off back-face detection there (`ProbeTraceRGS.hlsl:116, 131`).
- **Radiance clamp:** each ray is clamped to [0, 1] (K-05). That isn't a bug, but it is a large energy difference in sunlit scenes.

---

## 6. Doc errors found while writing this

- **Fixed** (2026-11-18; DV-08 on 2026-11-19 made the cap apply to the bounce term only). **The 0.95 bounce dimming doesn't exist.** The `bounceEnergy` comment says "native RTXGI uses 0.95" (`DDGI/settings.ts:223`, ddgi.md:435 and 443, DECISIONS:566). No 0.95 appears in the SDK or the sample. Native's actual factor is the 0.9 albedo cap on the bounce term (`ProbeTraceRGS.hlsl:238-243`), which `PLAT/contract/shade.ts:264-269` already describes correctly. Fix the comments (part of DV-08).
- **Fixed** (2026-11-18; issue #31 linked 2026-11-19): moved after v1, and SPEC and platform.md say it isn't built yet. **The `giRadiance` fallback isn't built.** SPEC:268 and platform.md:534 say DDGI gives a rough-only `giRadiance` fallback. The code says it doesn't (`DDGI/shading.ts:15`: `provides.radiance` is false). Either build it, as a small item with the reflections work, or correct both docs.
- **Fixed** (2026-11-18; issue #29 linked 2026-11-19): platform.md says so. **Emissive surfaces aren't light sources yet.** DECISIONS:333 promised next-event estimation over an emissive-triangle list for M2. It isn't in the code; emitters light the scene through bounces only, and area lights are in TASKS Later. Say so in platform.md.

---

## 7. Sources

- SDK settings: `SDK/include/rtxgi/ddgi/DDGIVolume.h:82-197`
- Blending: `SDK/shaders/ddgi/ProbeBlendingCS.hlsl`
- Relocation and classification: `SDK/shaders/ddgi/ProbeRelocationCS.hlsl` and `ProbeClassificationCS.hlsl`
- Sampling: `SDK/shaders/ddgi/Irradiance.hlsl`
- Variability: `SDK/shaders/ddgi/ReductionCS.hlsl`
- The sample: `HAR/shaders/ddgi/ProbeTraceRGS.hlsl`, `HAR/shaders/CHS.hlsl`, `HAR/shaders/include/Lighting.hlsl`, `HAR/shaders/IndirectCS.hlsl`, `HAR/src/graphics/DDGI_D3D12.cpp`
- Ours:
  - `DDGI/settings.ts`, `DDGI/wgsl/blend.ts`, `trace.ts`, `sample.ts`, `write.ts`, `scroll.ts`;
  - `PLAT/trace/wgsl.ts`, `PLAT/shade/wgsl.ts`.
- Our records: docs/DECISIONS.md, docs/NOTES.md, docs/design/ddgi.md and m2-lights.md. Line numbers are as of `main` 9dddefe.
