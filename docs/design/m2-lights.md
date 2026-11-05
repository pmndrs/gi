# M2 design draft: dynamic lights

Status: **draft for Dennis's review, 2026-11-05.** Nothing here is decided. It is the input for T1.11 ("M1 note and M2 breakdown"). Where it changes a number or a default from M1, it says so; those become DECISIONS entries only once Dennis agrees.

It builds on [ddgi.md](ddgi.md) (the field and update loop), [platform.md](../api/platform.md) (the contract), [three.md](../api/three.md) (the adapter) and the M1 entries in [DECISIONS](../DECISIONS.md). The M3 draft, [m3-moving.md](m3-moving.md), reuses the change-driven scheduler designed here for moved geometry.

## In one paragraph

Lights already live in a growable table and never recompile anything. M2 makes a light change show up fast and cheaply. Three things do most of the work. First, the platform learns *what* changed (the old and new light, not just a box), and a small GPU kernel turns that into a per-probe "how much did this probe's lighting change" number, computed analytically, so ray noise can't fake a change. Second, each probe gets its own history length instead of the fixed 0.97, shortened in proportion to its change, and a GPU scheduler puts the most changed probes near the camera at the front of the update list. Third, shadow rays move out of the trace kernel into a deferred, compacted pass in platform hit shading, which is where Sponza loses half its update time today, and that pass picks lights from a world-space grid of light reservoirs, so a thousand candles cost about what four lamps do. Light groups (one irradiance layer per group, composed into the atlas the renderer reads) make group dimmers and colour changes instant. The sky gets a GPU procedural source, so time of day re-bakes in a fraction of a millisecond, and the field follows at the budget's pace, nearest the camera first.

## What M2 must deliver (SPEC)

| Target | Tier A (RTX 3060, 1440p) | Tier B (Intel UHD) |
| --- | --- | --- |
| Light change visible | next frame | next frame |
| Light change within 5% of settled | ≤ 0.5 s | ≤ 1.5 s |
| GI update cost | ≤ 2.0 ms at the default budget | reported |
| GI sampling cost | ≤ 1.0 ms (must not regress from T1.6's 0.905 ms) | reported |
| Lights | ≥ 32, add/remove never recompiles | same |
| Adapter cost | ≤ 0.2 ms still scene, ≤ 0.5 ms one moving thing | reported |
| Sweep | looks right to Dennis next to native doing the same motion | |

M1's accuracy, leak, furnace and stability checks must still pass on still scenes.

## What others do, and what we copy

| Who | How light changes reach the GI | What we take |
| --- | --- | --- |
| **RTXGI-DDGI** (Majercik et al. 2019, 2021; the SDK in `native/rtxgi-ddgi`) | Updates every probe every frame with hysteresis 0.97. Per texel, if the new estimate differs from the stored value by more than `probeIrradianceThreshold` (0.25), hysteresis drops by 0.75 for that update; a change above `probeBrightnessThreshold` (0.10) is cut to a quarter (`ProbeBlendingCS.hlsl`, `DDGIVolume.h`). A "probe variability" reduction can stop updating a converged volume. Every shaded hit casts one shadow ray per light, with no light sampling (`Lighting.hlsl`). The docs suggest updates "may be scheduled at a lower frequency" but the SDK doesn't schedule. | The field and its hysteresis. **Not** the change test on the noisy estimate: T1.5b measured every variant of it biasing the field (+8% to +22% on Cornell; DECISIONS 2026-11-05). We get the change from the scene instead. |
| **Lumen** (Wright et al., "Lumen: Real-time Global Illumination in Unreal Engine 5", SIGGRAPH 2022 Advances) | Lights the surface cache with a fixed per-frame budget of pages, chosen by priority (distance, time since last update), and feeds the gather from that cache; indirect lags, direct is instant. | **A fixed budget with a priority order**, camera distance in the priority. |
| **MegaLights** (Narkowicz and Costa, SIGGRAPH 2025 Advances; [slides](https://www.advances.realtimerendering.com/s2025/content/MegaLights_Stochastic_Direct_Lighting_2025.pdf)) | Direct light by importance-sampling lights and tracing a *fixed* number of shadow rays per pixel in a separate pass, then denoising. Cost is constant in the light count. | **Fixed shadow rays per hit, traced in their own pass.** Our "denoiser" is the field's temporal blend. |
| **ReSTIR** (Bitterli et al., SIGGRAPH 2020) and **ReGIR** (Boksansky, Jukarainen, Wyman, *Ray Tracing Gems II* ch. 23, 2021; [chapter](https://boksajak.github.io/files/RTG2_ReGIRGridBasedReservoirs.pdf)) | ReSTIR resamples light candidates in screen space. ReGIR builds a few reservoirs per cell of a **world-space grid** each frame and lets any ray that hits inside a cell draw from them, which is what secondary rays need. | **ReGIR's world-space cell reservoirs** for many lights. DDGI's probe rays hit surfaces all over the world, not on screen, so screen-space ReSTIR doesn't apply. |
| **Light BVHs** (Conty Estevez and Kulla 2018; Moreau, Pharr, Clarberg, "Dynamic Many-Light Sampling for Real-Time Ray Tracing", HPG 2019) | A tree over lights, walked stochastically; Moreau keeps it dynamic with a two-level tree and refits, under 1 ms for thousands of lights. | The fallback idea if the grid's quality isn't enough (risk 3). Our BVH package could build it. |
| **Wavefront tracing** (Laine, Karras, Aila, "Megakernels Considered Harmful", HPG 2013) | Split a path tracer's big kernel into small ones with compacted work queues, so lanes stay busy. | **The deferred shadow pass** is exactly this for our trace kernel. |
| **A-SVGF** (Schied, Peters, Dachsbacher, "Gradient Estimation for Real-Time Adaptive Temporal Filtering", HPG 2018) | Shortens temporal history where it detects the lighting changed. | **History length per probe, shortened by the size of the change.** We know the change from the scene, so we don't need their noisy gradient estimate. |
| **Enlighten** (Martin and Einarsson, "A Real-Time Radiosity Architecture for Video Games", SIGGRAPH 2010 Advances) | Precomputed form factors for static geometry; lights and time of day relit at run time, progressively, in a budget. | Confirms lights-only dynamism can be cheap when the solve is incremental. We stay fully ray-traced (geometry moves in M3). |
| **Relighting by superposition** (Dorsey, Sillion, Greenberg, SIGGRAPH 1991; Nimeroff, Simoncelli, Dorsey, EGWR 1994) and HomeFig's light groups (SPEC) | Light transport is linear in the emitters, so a lighting change that only scales a group is a weighted sum of per-group solutions. HomeFig measured it linear to 0.006%. | **Light groups as layers**, with HomeFig's three rules (SPEC "Lessons"). |
| **Far Cry 3** (Gilabert and Stefanov, "Deferred Radiance Transfer Volumes", GDC 2012) and PRT (Sloan, Kautz, Snyder 2002) | Probes store SH transfer, so the sky and sun relight instantly for time of day. | Not in M2's default; offered as an option for instant time of day (question 1). |

## 1. The light buffer

**Today (T1.2):** lights are 64-byte records in the scene copy's table buffer, packed, directional lights first. Capacity starts at 32 and grows by reallocating the table buffer; no shader changes. `updateLight` takes per-type partial changes (`LightChanges`). The change log turns a light change into a box (`position ± range`) or, for the sun or a light with `range: 0`, a global flag.

**M2 changes:**

- **Capacity:** start at 64; grow by doubling as now. Tested to 4,096 lights (256 KB). Growth copies the table once; never more than log₂ steps.
- **The change log keeps the old and the new light,** not just boxes. `SceneChanges` gains `lights: readonly LightChange[]` with `{ index, before, after }` in the platform's units (technique-facing; hosts don't see it). The analytic change estimate in section 4 needs both.
- **Record v2** (same 64 bytes, one field reused): word 14 (`radius`) stays; the spot-cone words carry the rectangle's half-sizes for an area light (section 8). An **influence radius** is precomputed on the CPU for `range: 0` lights: the distance where the unwindowed falloff drops below 1/1,000 of its value at 1 m. It is used only to bound which probes a change can touch, never in shading.
- **Adapter:** lights are checked every frame today. With hundreds of lights that alone could break the 0.2 ms still-scene target (an estimate of about 0.2 µs per light check; T1.7 only timed scenes with a few lights). Lights beyond the first 64 get the same budgeted walk plus hot list as objects (three.md, "How changes are found"); `markChanged(light)` still sends a change the next frame.

## 2. The deferred shadow-ray pass (platform hit shading)

**Why:** T1.9a measured that shadow rays take 53% of Sponza's `ddgi/trace` and run at no more than 1/7 of the trace bench's any-hit rate. The any-hit walk runs inside a kernel that already holds the closest-hit stack, the bounce lookup and the shading state, and only some lanes in a subgroup have a front-face hit to light. Register pressure and divergence, not the tracer, are the cost.

**What changes:** `ddgi/trace` stops calling `giDirectIrradiance`. It writes everything it knows about a front-face hit into a **hit record** and appends the ray to a compacted work list. A new **platform** kernel, `gi/direct`, runs over that list only: it picks lights, traces their shadow rays and adds `albedo / π × direct` into the ray's radiance. `ddgi/blend` is unchanged: it reads finished radiances.

Hit record (24 bytes, one per blended ray that hit a front face):

| Field | Bytes | Why |
| --- | --- | --- |
| position, f32 × 3 | 12 | Shadow-ray origin; the offset is applied in `gi/direct` |
| shading normal, octahedral 16+16 | 4 | Cosine term |
| ray-side geometric normal, octahedral | 4 | "Light from behind can't reach it", and the push-off direction |
| albedo / π, `pack2x16float` rg + b in the radiance word | 4 | So `gi/direct` never reads the material table |

At 512 probes × 224 blended rays: 2.75 MB, sized by `maxProbesPerFrame` like the ray buffer. The hit records, the work-list counter and the ray results share **one** "ray scratch" buffer with three sections (the scene copy's pattern), so `ddgi/trace` stays at 7 storage buffers.

The work list is filled with **subgroup-aggregated atomics** (one `atomicAdd` per subgroup, offsets from `subgroupExclusiveAdd`); declared fallback without subgroups: one atomic per thread. A 1-thread kernel turns the count into indirect dispatch arguments.

**One code path, not two.** Light selection (section 3) becomes `giDirectLightSamples(surface, groupMask, random) -> GiDirectSamples` in the hit-shading module. `gi/direct` calls it and traces the samples; the existing `giDirectIrradiance` becomes a ten-line wrapper that calls the same function and traces inline, kept for the test views and the oracle. Both read the same selection code.

**Contract additions** (technique-facing, listed under "Revisions" in platform.md when built):
- `HitShading.encodeDirect(encoder, { scratch, maxRays, groupLayers })`: records `gi/direct` for a technique's ray scratch.
- `GiHitRecord` layout and the WGSL writer `giWriteHitRecord(...)`, so the technique never depends on the byte layout.

**Expected cost:** if shadow rays reach even half the bench any-hit rate (they're now coherent, full-subgroup work with low register use), the shadow share drops from 53% to about 15%, so `ddgi/trace` + `gi/direct` is about 0.62× of today's `ddgi/trace` on Sponza: **about 1.6× the probes per frame** at 2 ms. That's a planning number; the spike (T2.0) measures it before anything else is built.

Bindings: `gi/direct` = tracer 3 + scene copy and shading 2 + ray scratch 1 + light grid 1 = **7**.

## 3. Many lights: world-space light reservoirs (ReGIR-style)

**Today:** `giDirectIrradiance` loops over every local light twice (contribution, then systematic sampling of `maxShadowedLightsPerHit` = 4). The cost per hit is linear in the light count. With 1,024 candles, 512 probes × 224 rays × 2 × 1,024 is 235 M light evaluations per frame. That doesn't fit.

**Proposed:** two regimes in one function, chosen by a uniform (the local-light count), as the current code already chooses between "evaluate all" and "sample K":

1. **Few local lights (≤ `maxShadowedLightsPerHit`):** evaluate all, exactly, as now. No noise; the common case (Sponza, rooms with a few lamps).
2. **More:** draw from a **light grid** built each frame:
   - **Cells:** a uniform grid over the union of the GI volumes' boxes, one cell per 2×2×2 probe cells, at most 32,768 cells (Sponza: about 2,400).
   - **Candidates:** each cell builds `R = 8` reservoirs from `C = 16` candidates each, drawn from all local lights in proportion to their power (a CDF over `intensity × colour luminance`, rebuilt by a scan when lights change; the BVH package's `gpu-scan` is subgroups-first already).
   - **Target function:** unshadowed irradiance at the cell centre with three's range window and spot cone, no cosine (the surface normal isn't known per cell). Weighted reservoir sampling (Chao 1982) keeps one candidate per reservoir and its weight.
   - **At a hit:** `gi/direct` picks one of its cell's 8 reservoirs at random, re-evaluates that light at the real hit with the real cosine, and traces **one** shadow ray. Directional lights are still always evaluated (one shadow ray each). So a hit costs at most 1 + (directional lights) shadow rays, whatever the light count.
   - **Outside the grid** (hits beyond every volume): one candidate straight from the power CDF.
   - It is unbiased in the RIS sense as long as every light that can light a point has a non-zero target in that point's cell; the range window is applied to the cell's box, not its centre, so lights near a cell's edge aren't lost.

Data: `cells × R × 8 bytes` (light index, weight): 2,400 × 8 × 8 = 154 KB on Sponza, 2 MB at the 32,768 cap. Build cost: cells × R × C light evaluations = 2,400 × 128 = 0.3 M per frame on Sponza, about 0.05 ms on the RTX (estimate).

**Why not ReSTIR's temporal and spatial reuse too:** the field already accumulates over hundreds of rays and many frames; one good sample per hit is enough, and reuse would add per-hit state to a buffer that probes revisit only every few frames. Revisit if candles look noisy (risk 3).

**Why not a light BVH first:** constant cost per hit matters more for us than tight-light quality, the grid is simpler to keep right under motion (it's rebuilt every frame from scratch, no refit or rebuild policy), and it maps onto the probe grid we already have. The light BVH (Moreau 2019) is the fallback if the grid's noise on `sponza-candles` fails its check.

**`sponza-candles`:** a stated subset of Intel's 10,000 candles (DECISIONS 2026-11-05). Proposed: 1,024 lights and only those candles' geometry (the full candle mesh is 19.5 M triangles). The subset is question 2.

## 4. Change-driven updates: per-probe history and a GPU scheduler

This replaces the big-change rule (off since T1.5b) as the way light changes get fast. It is the "priority list in front of the update cursor, in the same buffer" that ddgi.md left room for.

### 4.1 Per-probe history length

Each probe gets a 16-byte history record (`ddgiProbeHistory`, 0.3 MB at 20,000 probes):

| Field | Meaning |
| --- | --- |
| `n` (f16) | Effective history length in updates. The blend uses `h = min(hysteresis, n / (n + 1))`, then sets `n ← n + 1`. With `hysteresis` 0.97, `n` stops mattering past 32. |
| `pendingChange` (f16) | Relative change since this probe's last update, accumulated over marks. Reset by the blend. |
| `owed` (u8) | Updates this probe is owed at priority. Decremented by each update. |
| `urgency` (f16) | How much this probe matters to the next frames: its change, highest since its last update. |
| `scheduledFrame` (u32) | Last frame it was put in the update list, so the fill never adds it twice. |

- **State 3 ("not updated yet") is just `n = 0`,** so `h = 0` on its first update, as now.
- **A probe in steady state behaves exactly as in M1** (`n ≥ 32` gives `h = 0.97`). Still-scene results can't change.

**How long the history becomes after a change.** If a probe's lighting changed by a relative amount `r`, and the scheduler can promise it `K` updates inside the response window, the leftover error after those `K` updates is `r × n / (n + K)`. Setting that equal to the tolerance `τ` gives

> `n ← min(n, τK / (r − τ))` for `r > τ`; untouched for `r ≤ τ`.

With SPEC's τ = 5% and K = 8:

| Relative change r | New n | First update's `h` | Left after 8 updates |
| --- | --- | --- | --- |
| 1.0 (light switched on next to it) | 0.42 | 0.30 | 5% |
| 0.5 | 0.89 | 0.47 | 5% |
| 0.2 | 2.7 | 0.73 | 5% |
| 0.1 | 8 | 0.89 | 5% |
| ≤ 0.05 | unchanged | 0.97 | n/a |

*Why a running-mean history rather than a fixed fast hysteresis:* a fixed `h = 0.5` forgets in 5 updates but keeps the noise of about 3 rays-worth of samples forever after; the running mean gets less noisy with every update and climbs back to 0.97 on its own. *Why from the scene and not from the probe's own estimate:* the estimate is noisy, and every rule built on it biased the field (T1.5b). The change estimate below uses only light parameters and the field's stored data, so a still scene can never trip it.

The same history applies to the distance atlas. For light changes it hardly matters (distances don't change); M3 uses it for geometry.

The opt-in big-change setting (`irradianceChangeThreshold`) stays opt-in, as decided, but when enabled it writes into `n` too, so there is one mechanism.

### 4.2 The change estimate (`ddgi/mark`)

Runs only in frames with changes. The CPU writes a small region table (at most 256 entries, as the change log already merges) and dispatches one thread per (entry, probe inside the entry's probe box).

- **Local light moved, recoloured, dimmed, toggled:** at the probe's position, `E_before` and `E_after` are the light's unshadowed irradiance with three's falloff, range window and cone (the same WGSL as hit shading, `giShadeLightSample`, with no surface cosine). Then
  `r = lightChangeScale × |lum(E_after) − lum(E_before)| / (lum(probe mean irradiance) + ε)`.
  - The probe's mean irradiance is one bilinear tap of the centre of its irradiance tile (a cheap stand-in for the mean).
  - `lightChangeScale` (default 0.25, tuned in T2.4) accounts for the probe seeing the light only through the surfaces it lights.
  - **Visibility filter:** the probe's distance moments toward the light (one Chebyshev test, as in sampling) scale `r` by `max(chebyshev, 0.25)`, so probes behind a wall get a smaller share but never zero (they can still see lit surfaces through a door).
  - The probe box is the light's range (or influence radius) box clipped to the volume.
- **Global changes (sun direction or colour, sky):** every probe. `r` is the size of the change (the sun's angle moved, or the ratio of old to new intensity), times `0.25 + 0.75 × openness`, where openness is the probe's distance moments toward the old and new sun directions (seeing open sky means seeing the change directly).
- **Each marked probe:** `pendingChange += r`; `n ← min(n, nFor(pendingChange))`; `owed ← max(owed, K)`; `urgency ← max(urgency, pendingChange)`.

Continuous motion (the sweep) marks a little every frame. Because `pendingChange` accumulates until the probe is updated, the history length settles where it matches the rate of change: fast near the moving light, unchanged far away. No special case.

### 4.3 The GPU scheduler

The update list moves from the CPU (`UpdateCursor` plus `writeBuffer`) to the GPU, because the urgencies live there. It is still one buffer, filled to exactly `min(units, probes)` entries, so the trace, blend and write dispatches keep CPU-known sizes and the cost model still sees the units it handed out. Four small dispatches:

1. **`ddgi/schedule-count`** (one thread per probe): probes with `owed > 0` get a priority `urgency × cameraWeight`, bucketed into 32 log₂ buckets; a histogram is built with subgroup-aggregated atomics (fallback: per-thread atomics).
   - `cameraWeight = 1 / (1 + (distance to camera / (4 × spacing))²)`, × 0.25 outside the view frustum. *Why:* SPEC's response target is judged on the image, and nearby, visible probes are what the image shows (Lumen's priority does the same).
2. **`ddgi/schedule-select`** (one workgroup): finds the threshold bucket so that at most `P = priorityShare × units` probes are chosen (`priorityShare` default 0.75, so the round robin always keeps a quarter of the budget and nothing starves).
3. **`ddgi/schedule-write`** (one thread per probe): probes above the threshold, and up to the allowed number at it, are appended to the list and stamped with the frame.
4. **`ddgi/schedule-fill`** (one workgroup): the rest of the list comes from the existing scrambled order, `order[i] = (i × stride) mod N`. A window of `units` consecutive entries holds at most `P` stamped probes, so it always yields enough unstamped ones; a prefix sum picks them and the cursor (now one GPU word) advances by the entries consumed.

Cost: about 4 dispatches of a few µs each plus one thread per probe; under 0.05 ms on the RTX at 20,000 probes (estimate). Steps 1–3 are skipped when the CPU knows nothing was marked in the last 240 frames.

**What the budget buys.** At `U` probes per frame, a response window of `W` frames and `K` owed updates, the probes that can be fully served in time number `P × W / K`. At Sponza's expected ~225 probes per frame after section 2 (`P` = 169), W = 30 and K = 8: **about 630 probes** get all 8 updates inside 0.5 s. A lamp whose influence box holds more than that is served nearest-and-most-changed first, so the visible image meets the target while far, weakly changed probes finish later. The furnished room and two-room (a few thousand probes in total) fit entirely.

### 4.4 Light groups make some changes need no updates at all

See section 5: a dimmer or colour change across a whole group skips sections 4.2–4.3 entirely.

## 5. Light groups

Linearity means the field for "group g at weight w" is w × the field for group g at weight 1, bounces included, as long as each group's bounce reads only its own layer (HomeFig rule 1).

**Layers:** one irradiance layer per light group in use (contract groups 0–3), plus an **environment layer** for the sky and emissive surfaces (HomeFig rule 2: the sky's shape changes, so it's a layer of its own, re-traced, never scaled for time of day). With groups off there is one layer, and everything below reduces to M1's kernels with a loop count of 1.

| What | Layers | Notes |
| --- | --- | --- |
| Irradiance, per group | G + 1 (environment) | Stored at the lights' real intensity and colour (HomeFig rule 3) |
| Composed irradiance | 1 | What the renderer samples. Not needed when G + 1 = 1 |
| Distance, probe data | 1 | Geometry only, shared |

- **Trace:** one closest hit per ray, as now. The bounce lookup computes the 8 probes' weights (distance, Chebyshev, crush) once and fetches each layer's irradiance with them: the extra cost per layer is only the irradiance fetches, about a quarter of the lookup.
- **Direct:** one light sample per hit, whatever its group; its contribution goes into its own group's layer. Each layer's estimate stays unbiased; minority groups get noisier.
- **Ray results:** 8 bytes per ray per layer, in the ray-scratch buffer.
- **Blend:** loops the 36 irradiance texels per layer; the 196 distance texels once.
- **`ddgi/compose`:** composed = Σ weight_g × layer_g, per texel, borders included. It runs for the tiles updated this frame, and for every tile in the frame a weight changes. That is the "instant": **a group weight change is visible in the same frame**, with no lag and no noise.
- **Weights** are technique settings: `lightGroups: { count, weights: Color[], environmentWeight: Color }`. Weights are linear RGB multipliers on top of the lights' own intensities. The three adapter exposes `threeGI.setGroupWeight(group, color)`.

**Memory** (irradiance tile 8 × 8 texels × 8 bytes = 512 bytes per probe per layer):

| Scene, probes | Groups off | 2 groups + env + composed | 4 groups + env + composed |
| --- | --- | --- | --- |
| Furnished room, ~3,000 | 1.5 MB | 6.1 MB | 9.2 MB |
| Two-room, 1,071 | 0.5 MB | 2.2 MB | 3.3 MB |
| Sponza, 19,425 | 9.9 MB | 39.8 MB | 59.7 MB |

Distance (1 KB per probe) is unchanged. So 4 groups fit the 64 MB furnished-room budget easily and are expensive on Sponza: **groups are opt-in, off by default**, and the host picks how many.

**Cost estimate** at 4 groups: trace about +10% (irradiance fetches only), blend about +60% (irradiance texel work × 5), compose ~0.1 ms on a weight change for Sponza. Update cost per probe about +25–35%. Shading cost: unchanged, because the renderer samples one composed atlas. *Why not sample the layers in the fragment shader:* T1.6 measured irradiance fetches at 0.21 of the 0.905 ms; four layers would push sampling past 1.0 ms.

## 6. Sun and sky with time of day

**Today:** a `procedural` environment is built on the CPU (Preetham, calibrated with the Ineichen–Perez ratio; `calibrated-sky.ts`), uploaded, then resampled to the 256 × 128 radiance map and projected to 18 SH coefficients by one workgroup. It runs only when `setEnvironment` is called. The SH projection kernel takes about 1 s to compile (T1.9a), and the CPU build isn't meant to run every frame.

**M2:**

- **A GPU procedural source.** The resample kernel evaluates Preetham directly (a new `SKY_SOURCE_KIND`), with the calibration scale computed on the CPU per sun direction (a few scalar formulas) and passed as a uniform. The CPU code stays: it writes the shared sky files for Cycles and native, and a test holds the GPU map to it (mean within 0.1%, max texel within 1%).
- **The SH projection becomes two levels** (many workgroups write partial sums, one finishes), subgroups first with the existing workgroup-memory fallback. On a procedural sky it integrates over the 256 × 128 map itself.
- **Re-bake rule:** when the sun direction moves by more than 0.05°, or turbidity, intensity or `sunIlluminance` changes; at most once per frame. Estimated GPU cost: under 0.1 ms per re-bake. CPU: under 0.05 ms.
- **The slow compile:** T2.6 finds why the projection takes about 1 s to compile (likely the unrolled 18-term basis inside the loop) and gets it under 0.2 s. This also helps startup (DECISIONS: "find the slow sky kernel").
- **The sun** is a directional light and a separate object (light buffer, shadow rays, its own group if groups are on). Keeping the sun light and the sky's sun direction in step is the host's job; the viewer's slider moves both. The three adapter already maps pmndrs/sky and `SkyMesh` (three.md, "The sky").

**How fast the field follows.** A sky or sun-direction change is global: every probe is marked (section 4.2), and the field follows at the budget's pace, camera-near first.

| Scene | Probes | Full passes per second at the expected rate | Time-of-day drag |
| --- | --- | --- | --- |
| Furnished room | ~3,000 | ~4.5 (at ~225 probes/frame) | follows within ~0.5 s |
| Two-room | 1,071 | ~13 | follows within ~0.2 s |
| Sponza | 19,425 | ~0.7 | the camera's neighbourhood in ~0.5 s, the whole volume in ~1.5–3 s |

A slow time-of-day animation (an hour of sun in 10 s, 1.5° per second) marks every probe a little each frame, which shortens histories only slightly; it looks continuous. Dragging the slider fast on Sponza lags by seconds away from the camera. Making that instant needs PRT-style sky transfer per probe texel (Far Cry 3's approach): about 16 SH coefficients × 3 colours per texel, about 7× the irradiance atlas (Sponza +67 MB). That's question 1; not in the default plan.

**Native:** the native harness's miss shader sees one flat colour (DECISIONS 2026-11-04). M2 gives it the texture sky and an animated sun, so the time-of-day comparison is like for like (T2.7).

## 7. The viewer

- **Gizmos:** three's `TransformControls` on light helpers (drag to move, rotate spots). Selecting a light opens its panel.
- **Sliders and toggles:** intensity (log scale), colour, range, cone and penumbra for spots, shadows on or off, group. Toggling sets intensity to 0, so three doesn't recompile its materials (NOTES: removing a three light recompiles).
- **Group dimmers:** one weight slider and colour per group, plus the environment layer; hover help says "instant: no update needed".
- **Time of day:** hour and day-of-year sliders driving sun direction (a simple solar-position formula), the scene's sun light and the GI sky together; play button for a 10-second day.
- **Scripted motions** from the manifest: `motions: [{ id, target: 'light:<id>', keyframes: [{ t, position, intensity, color }], loop }]`. Two built in per lit scene:
  - **light sweep:** a lamp moving along a path at 1 m/s (two-room: across the lit room and through the doorway; Sponza: along the lower colonnade);
  - **step:** the same lamp jumping between two keyframes every 3 s, for measuring response.
- **Response meter:** pick a screen rectangle; the viewer plots its mean luminance per frame, marks "first frame that changed by more than 1%" and "first frame within 5% of settled", and writes the curve to CSV. "Settled" is the mean after 4 s with the budget raised to 16 ms (a test mode, labelled as such).
- **Native side by side:** the native harness plays the same manifest motion (T2.7) and records the meter's curve and a frame every 4 frames. A compare page plays both recordings in sync, with both curves on one plot. Live compositing of native into the viewer isn't attempted.
- **Stats** gain: probes updated at priority vs round robin, probes owed, shadow rays per frame, `gi/direct` ms, compose ms, sky re-bake ms, lights.

## 8. Area lights (carried from M1)

DECISIONS 2026-11-05 (viewer entry) asks for area lights so an emitter's direct light can be analytic and DDGI holds only its bounce. Proposed as a small task inside M2, droppable:

- A `rect` light type: position, normal, half-sizes, two-sided flag; shadow rays aim at a uniform point on the rectangle; cosine at both ends.
- Its emitter surface is marked "light carried by a light" so GI rays that hit it see no emission (or the light is counted twice).
- The viewer's emissive Cornell then lights its lamp with three's `RectAreaLight` for direct and DDGI for bounce only.

## How it fits the existing kernels (no parallel GPU paths)

| Kernel | M1 | M2 |
| --- | --- | --- |
| `ddgi/mark` | none | new; only in frames with changes |
| `ddgi/schedule-*` | update list from the CPU cursor | GPU scheduler writes the same buffer; the CPU cursor is removed, not kept beside it |
| `ddgi/trace` | trace, shade, direct light inline | trace, shade, bounce per layer; writes hit records. **No `giDirectIrradiance` call** |
| `gi/direct` (platform) | none | new; replaces the inline direct light for every technique that traces |
| light grid build (platform) | none | new; only when local lights exceed `maxShadowedLightsPerHit` |
| `ddgi/blend` | global hysteresis | per-probe `h` from the history record; loops layers |
| `ddgi/compose` | none | only with light groups on |
| `ddgi/write` | unchanged | unchanged |
| sky resample and projection | CPU-built procedural sky, single-workgroup SH | GPU procedural source, two-level SH |

Every new kernel that reduces or compacts is subgroups first, with one declared fallback (per-thread or workgroup atomics) and a test that runs both on the same input (the M1 rule).

**Storage buffers per kernel** (default limit 8):

| Kernel | Count |
| --- | --- |
| `ddgi/trace` | tracer 3 + scene copy and shading 2 + update list + ray scratch = **7** |
| `gi/direct` | tracer 3 + scene copy and shading 2 + ray scratch + light grid = **7** |
| `ddgi/blend` | update list, ray scratch, scratch tile, census, history = **5** |
| `ddgi/mark` | region table, history = 2 (+ atlases as textures) |
| `ddgi/schedule-*` | history, update list, schedule state = 3 |

## Cost against the budgets (estimates, RTX 3060, to be measured)

| Item | When | GPU ms |
| --- | --- | --- |
| `ddgi/trace` + `gi/direct` on Sponza, per probe | always | ~0.6× M1's `ddgi/trace` (T2.0 measures) |
| Scheduler (4 dispatches) | always (steps 1–3 only after a mark) | < 0.05 |
| `ddgi/mark` | frames with changes | < 0.02 for a few lights; < 0.05 for a global change at 20k probes |
| Light grid build | > 4 local lights | ~0.05 on Sponza, 1,024 lights |
| `ddgi/compose` | groups on | updated tiles only: < 0.01; a weight change on Sponza: ~0.1 |
| Sky re-bake | sky changed | < 0.1 |

All of it is inside the 2 ms update budget: the platform's cost model measures `ddgi/*` and `gi/direct` together, so extra fixed costs shrink the probe count rather than the frame. Shading cost doesn't change. Memory: history 16 B/probe, hit records 2.75 MB at 512 probes per frame, light grid ≤ 2 MB, groups as tabled.

## Proposed task breakdown

Board format. IDs are proposals; T1.11 assigns the real ones. Every GPU check runs on the RTX 3060 and the Intel UHD, says which adapter, and uses the bench lock for timings.

| ID | Task | Agent | Size | Depends on |
| --- | --- | --- | --- | --- |
| T2.0 | Spike: deferred shadow-ray pass on Sponza | implementer | S | M1 done |
| T2.1 | Light buffer v2 and `sponza-candles` | implementer + wrangler | M | none |
| T2.2 | Deferred direct light (`gi/direct`) in platform hit shading | implementer | M | T2.0 |
| T2.3 | Many-light sampling: light grid reservoirs | implementer | M | T2.1, T2.2 |
| T2.4a | Per-probe history and `ddgi/mark` | implementer | M | T2.1 |
| T2.4b | GPU scheduler with camera priority | implementer | M | T2.4a |
| T2.5 | Light groups: layers, compose, weights, adapter API | implementer | M | T2.2, T2.4a |
| T2.6 | Sky and time of day: GPU procedural source, two-level SH, compile fix | implementer | M | T2.4a |
| T2.7 | Native harness: manifest motions, texture sky, animated sun | native-runner | M | T2.1 (manifest format) |
| T2.8 | Viewer: gizmos, sliders, groups, time of day, motions, response meter, native compare page | implementer | M | T2.4b, T2.5, T2.6, T2.7 |
| T2.9 | Rect area lights (droppable) | implementer | S | T2.2 |
| T2.10 | M2 measurements against targets | gpu-tester, native-runner | M | all above |
| T2.11 | M2 note and M3 breakdown | main session | S | T2.10 |

The startup items DECISIONS moved to M2 ("fast first GI", the scene copy off the main thread, compiling only what's used) are designed in [m3-moving.md](m3-moving.md), because they share the BVH and scene-copy machinery, and their rows (T2.12–T2.14) are listed there.

### Draft done-whens

- **T2.0 Spike: deferred shadows.** A throwaway branch splits Sponza's `ddgi/trace` into trace + `gi/direct` with hit records.
  - Done when: per-probe cost of the pair at 512 probes, and the shadow-ray rate against the trace bench's any-hit rate, are recorded on both GPUs (3 runs, bench lock, apps closed per DECISIONS), with the field matching M1's within 1% mean on Cornell and Sponza after 300 frames.
  - Decision rule, set before measuring: go ahead with T2.2 if the pair is ≥ 1.25× faster per probe on Sponza on the RTX; otherwise record why and bring it back.
- **T2.1 Light buffer v2.**
  - `SceneChanges.lights` with before and after; capacity from 64, tested to 4,096; influence radius for `range: 0`; adapter light checks budgeted past 64 lights.
  - `sponza-candles` with the subset Dennis picks (question 2): licence, size, triangle and light counts, checksum.
  - Done when: creating 4,096 lights and destroying them in random order causes 0 pipeline creations (counter); a test checks every `LightChanges` field per light type round-trips; the adapter's still-frame CPU time on `sponza-candles` is ≤ 0.2 ms (median of 300 frames).
- **T2.2 Deferred direct light.**
  - Done when: `gi/direct`'s direct irradiance matches the inline `giDirectIrradiance` on the T1.4 oracle points (mean within 0.5%, sun-on-plane E·cos θ within 0.5%); the subgroup and fallback work lists give the same field (mean within 0.1% after 300 frames); Sponza's update cost per probe at 512 probes meets T2.0's measured gain within 10%; the furnace, Cornell and two-room checks of T1.5 still pass.
- **T2.3 Many-light sampling.**
  - Done when: on a test scene with 1,024 point lights, direct irradiance from `gi/direct` averaged over 64 frames matches an exhaustive evaluation within 1% mean and 5% at the worst of 1,000 points; `sponza-candles` update cost per probe is ≤ 1.3× `sponza` with the sun only; grid build ≤ 0.1 ms on the RTX.
- **T2.4a Per-probe history and `ddgi/mark`.**
  - Done when: on a still scene the field matches M1's within 0.5% mean (history saturates at 0.97); on two-room, moving the lamp 2 m (step) reaches within 5% of settled in ≤ 0.5 s at the RTX's default 2 ms budget (response meter on the `lit-room` bookmark) with every probe updated each frame (so the scheduler isn't involved yet); a still scene marks 0 probes over 600 frames (counter).
- **T2.4b GPU scheduler.**
  - Done when: two-room, Cornell (point light) and Sponza step changes are visible the next frame (ROI changes by > 1%) and within 5% of settled in ≤ 0.5 s on the RTX and ≤ 1.5 s on the Intel UHD, at the default budget; no probe goes un-updated for more than 2 full round-robin passes during a 60 s sweep (census); scheduler cost ≤ 0.05 ms (RTX) at 20,000 probes; the subgroup and fallback histograms agree exactly.
- **T2.5 Light groups.**
  - Done when: on two-room with 2 groups, a weight change shows in the same frame and the image matches a field re-converged from scratch at that weight within 1% mean (linearity); with groups off, update cost and field equal M1's within noise; 4-group update cost per probe ≤ 1.4× groups-off; GI sampling cost unchanged (≤ +0.03 ms at 1440p); memory per layer reported.
- **T2.6 Sky and time of day.**
  - Done when: the GPU procedural map matches the CPU sky within 0.1% mean and 1% per texel; `giSkyIrradiance` still meets ±0.5% on open ground; a re-bake costs ≤ 0.1 ms GPU and ≤ 0.05 ms CPU; the projection compiles in ≤ 0.2 s cold on the RTX; on the furnished room a 30° sun move reaches within 5% in ≤ 0.5 s (RTX).
- **T2.7 Native harness.**
  - Done when: `play.ps1 -Scene two-room -Motion light-sweep` plays the manifest motion at the same speed as the viewer (positions agree within 1 cm at matching times), the miss shader reads the shared sky file, the sun can follow a time-of-day motion, and the default config renders as before (by eye plus one capture).
- **T2.8 Viewer.**
  - Done when: every control has hover help; the sweep and step motions run on two-room, Cornell and Sponza; the response meter's CSV for a step on two-room matches the T2.4b test page within 2 frames; the native compare page plays both recordings in sync; Dennis can drive it without instructions.
- **T2.9 Rect area lights.** Done when the emissive Cornell lit by a rect light plus DDGI bounce matches the Cycles full image within the M1 accuracy target, and the lamp isn't counted twice (furnace-style check: the emitter alone gives no GI on the emitter itself).
- **T2.10 M2 measurements.** The table at the top, per GPU, plus native's response curves for the same motions (3 runs). Dennis judges the sweep next to native.

## Risks, and what to spike first

1. **The deferred pass may not buy as much as estimated** (highest risk, largest lever). If hit records and the indirect dispatch eat the gain, Sponza stays at tens of probes per frame and the response target can only be met near the camera. *Spike first: T2.0.*
2. **Short histories flicker.** At `n` = 0.42 the first update is 70% one noisy estimate. If it shows as a flash, raise the floor on `n`, or give marked probes 2× rays for their first update (cost). The two-room step in T2.4a shows it.
3. **Light-grid noise with many small lights.** Candles have short ranges; a cell 2 probe spacings wide may hold many lights that matter only at its corners. Options: smaller cells near lights, more reservoirs, or Moreau's light BVH. `sponza-candles` decides.
4. **The change estimate is a proxy.** `lightChangeScale` and the visibility floor are guesses until T2.4a measures how many probes get marked versus how many actually changed by > 5% (compare against a re-converged field). Too generous wastes budget; too stingy leaves stale probes until the round robin gets there.
5. **Global changes on big scenes** can't meet 0.5 s everywhere at 2 ms (section 6). The plan meets it in view; question 4 asks whether that's the right reading of the target.
6. **Native motion** needs fork changes (the harness has no animation). Two-room first, Sponza second.

## Questions for Dennis

1. **Time-of-day scrubbing:** is it fine if dragging the slider fast on Sponza lags by a second or more away from the camera (the default plan)? Or should we build the optional "instant sky" (PRT-style transfer per probe, about 7× the irradiance memory) for scenes that can afford it?
2. **`sponza-candles`:** how many of the 10,000 candles? I'd propose 1,024 lights with only those candles' geometry.
3. **Light groups:** off by default, host picks the count, emissive surfaces go in the environment layer. OK? And is an explicit group-weight API enough, or do you want dimming every light of a group the usual three.js way (scaling each light's `intensity`) to be detected and treated as a group weight?
4. **The response target for global changes:** for the sun or sky on big scenes, may "within 5% in 0.5 s" be judged on the view (camera-near probes first), with the rest of the volume following within a couple of seconds?
5. **Native side by side:** judged by eye from synced recordings plus both response curves on one plot, or do you want a number (say, our time-to-5% within 1.5× of native's)?
6. **Area lights in M2,** or after?
</content>
</invoke>
