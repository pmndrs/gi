# M2 design draft: dynamic lights

Status: **draft for Dennis's review, 2026-11-05, revised after an Opus review the same day.** Nothing here is decided. It is the input for T1.11 ("M1 note and M2 breakdown"). Where it would change a default or a settled choice from M1, it says so; those become DECISIONS entries only once Dennis agrees.

It builds on [ddgi.md](ddgi.md) (the field and update loop), [platform.md](../api/platform.md) (the contract), [three.md](../api/three.md) (the adapter) and the M1 entries in [DECISIONS](../DECISIONS.md). The M3 draft, [m3-moving.md](m3-moving.md), reuses the change-driven scheduler designed here for moved geometry.

## In one paragraph

Lights already live in a growable table, and adding one never recompiles anything. M2 makes a light change show up quickly and cheaply. Three changes do most of the work:

1. **The platform tells DDGI what changed:** the old and the new light, not just a box. A small GPU kernel turns that into "how much did this probe's lighting change", worked out from the light's own numbers, so ray noise can't fake a change.
2. **Each probe gets its own history length** instead of the fixed 0.97. It is shortened in proportion to the change. A GPU scheduler puts the most changed probes near the camera at the front of the update list.
3. **Shadow rays move into their own pass in platform hit shading.** Today they sit inside the trace kernel, where Sponza loses half its update time. The new pass picks lights from a world-space grid of pre-chosen candidates, so a thousand candles cost about what a few lamps do.

Light groups (one irradiance layer per group, added up into the atlas the renderer reads) make a dimmer or a colour change across a group instant. The sky gets a GPU procedural source, so time of day re-bakes in a fraction of a millisecond. After a sky change, the field follows at the budget's pace, nearest the camera first.

## Words used here

- **Probe update:** one probe's rays traced and blended into its stored values, once.
- **History length (n):** how many past updates a probe's stored value is worth. The blend's weight on the old value is `n / (n + 1)`, at most 0.97.
- **Response window:** SPEC's 0.5 s, 30 frames at 60 Hz.
- **τ (tolerance)** = 5%, SPEC's "within 5% of settled". **K** = how many updates a changed probe is promised inside the window (8 here).
- **Owed updates:** how many of those K a probe still waits for.
- **Shadow ray:** a ray from a lit point toward a light, to see if anything blocks it.
- **Subgroup:** the group of GPU threads (8–64) that run in lockstep. "Subgroups first" means kernels use subgroup operations where they help, with one declared fallback (DECISIONS 2026-11-04).
- **Compacted list:** the work items written one after another with no gaps, so every thread has something to do.
- **Indirect dispatch:** the GPU decides how many workgroups run, from a count it wrote itself, so the CPU never reads it back.
- **Reservoir** (from ReSTIR/ReGIR): a slot that keeps one light chosen at random from a stream of candidates, each with a probability that depends on how much it would contribute, plus a weight that keeps the result unbiased.
- **CDF:** a running sum of light powers, used to pick lights in proportion to their power.
- **Linearity (superposition):** twice the light gives twice the bounce, and two lights give the sum of what each gives alone.

## What M2 must deliver (SPEC)

| Target | Tier A (RTX 3060, 1440p) | Tier B (Intel UHD) |
| --- | --- | --- |
| Light change visible | next frame | next frame |
| Light change within 5% of settled | ≤ 0.5 s | ≤ 1.5 s |
| GI update cost | ≤ 2.0 ms at the default budget | reported |
| GI sampling cost | ≤ 1.0 ms (no regression from T1.6's 0.905 ms) | reported |
| Lights | ≥ 32; adding or removing one never recompiles | same |
| Adapter cost | ≤ 0.2 ms still scene, ≤ 0.5 ms one moving thing | reported |
| Sweep | looks right to Dennis next to native doing the same motion | |

M1's accuracy, leak, furnace and stability checks must still pass on still scenes.

## The update rates this plan works with

Everything about response time depends on how many probes fit in 2 ms. Measured at 60 Hz on the RTX, 2 ms budget (T1.9a, PROGRESS):

| Scene | Probes | Probes per frame at 2 ms | Note |
| --- | --- | --- | --- |
| Cornell | 1,728 | 995 | measured |
| Two-room | 1,071 | 337 | measured |
| Furnished room | 4,998 | not measured | to measure in T1.9 |
| Sponza | 19,425 | 39 measured; **about 190** once the controller fix lands | its cost is nearly flat from 20 to 190 probes (T1.9a re-review) |
| Bistro exterior | n/a | 23 measured; about 90 from its cost curve (1.57 ms + 4.7 µs per probe, NOTES) | |

The plan below uses 190 for Sponza. If T1.9's restated number differs, every Sponza figure here scales with it.

## What others do, and what we copy

| Who | How light changes reach the GI | What we take |
| --- | --- | --- |
| **RTXGI-DDGI** (Majercik et al. 2019, 2021; the SDK in `native/rtxgi-ddgi`) | Every probe is updated every frame with hysteresis 0.97. Per texel, if the new estimate differs from the stored value by more than `probeIrradianceThreshold` (0.25), hysteresis drops by 0.75 for that update; a change above `probeBrightnessThreshold` (0.10) is cut to a quarter (`ProbeBlendingCS.hlsl`, `DDGIVolume.h`). A "probe variability" reduction can stop updating a converged volume. Every shaded hit casts one shadow ray per light, with no light sampling (`Lighting.hlsl`). | The field and its hysteresis. **Not** the change test on the noisy estimate: T1.5b measured every variant of it biasing the field (+8% to +22% on Cornell; DECISIONS 2026-11-05). We take the change from the scene instead. |
| **Lumen** (Wright et al., "Lumen: Real-time Global Illumination in Unreal Engine 5", SIGGRAPH 2022 Advances) | Lights its surface cache within a fixed per-frame budget, choosing what to refresh by priority; indirect light lags, direct light is instant. | **A fixed budget with a priority order,** camera distance included. |
| **MegaLights** (Narkowicz and Costa, SIGGRAPH 2025 Advances; [slides](https://www.advances.realtimerendering.com/s2025/content/MegaLights_Stochastic_Direct_Lighting_2025.pdf)) | Direct light by importance-sampling lights and tracing a *fixed* number of shadow rays per pixel, in a pass of its own, then denoising. Cost stays constant as lights are added. | **A fixed number of shadow rays per hit, in their own pass.** Our "denoiser" is the field's blend over time. |
| **ReSTIR** (Bitterli et al., SIGGRAPH 2020) and **ReGIR** (Boksansky, Jukarainen, Wyman, *Ray Tracing Gems II* ch. 23, 2021; [chapter](https://boksajak.github.io/files/RTG2_ReGIRGridBasedReservoirs.pdf)) | ReSTIR picks lights per pixel on screen. ReGIR builds a few reservoirs per cell of a **world-space grid** each frame, and any ray that hits inside a cell draws from them, which is what rays that don't start at the camera need. | **ReGIR's world-space cells** for many lights. DDGI's rays hit surfaces all over the world, not on screen. |
| **Light trees** (Conty Estevez and Kulla 2018; Moreau, Pharr, Clarberg, "Dynamic Many-Light Sampling for Real-Time Ray Tracing", HPG 2019) | A tree over the lights, walked at random toward the bright parts. Moreau keeps it dynamic with two levels and refits: under 1 ms for thousands of lights. | The fallback if the grid is too noisy (risk 3). Our BVH package could build it. |
| **Wavefront tracing** (Laine, Karras, Aila, "Megakernels Considered Harmful", HPG 2013) | Split one big kernel into small ones joined by compacted work lists, so threads stay busy. | **The deferred shadow pass** is this, applied to our trace kernel. |
| **A-SVGF** (Schied, Peters, Dachsbacher, HPG 2018) | Shortens the history of a temporal filter where it detects that lighting changed. | **History length per probe, shortened by the size of the change.** We know the change from the scene, so we skip their noisy detector. |
| **Enlighten** (Martin and Einarsson, "A Real-Time Radiosity Architecture for Video Games", SIGGRAPH 2010 Advances) | Static geometry with precomputed links between surfaces; lights and time of day relit at run time, a little each frame. | Confirms incremental, budgeted updates are the norm. We stay ray-traced, because geometry moves in M3. |
| **Relighting by superposition** (Dorsey, Sillion, Greenberg, SIGGRAPH 1991; Nimeroff, Simoncelli, Dorsey, EGWR 1994) and HomeFig's light groups (SPEC) | A lighting change that only scales a group is a weighted sum of per-group solutions. HomeFig measured it linear to 0.006%. | **Light groups as layers,** with HomeFig's three rules (SPEC "Lessons"). |
| **Far Cry 3** (Gilabert and Stefanov, "Deferred Radiance Transfer Volumes", GDC 2012) and PRT (Sloan, Kautz, Snyder 2002) | Probes store how they respond to each of a few sky "basis" patterns, so the sky and sun relight instantly. | Not in the default plan; offered for instant time of day (question 1). |

## 1. The light buffer

**Today (T1.2):** lights are 64-byte records in the scene copy's table buffer, packed, directional lights first. Capacity starts at 32 and grows by reallocating the table buffer, with no shader changes. `updateLight` takes per-type partial changes (`LightChanges`). The change log turns a light change into a box (`position ± range`) or, for the sun or a light with `range: 0`, a global flag.

**M2 changes:**

- **Capacity** starts at 64 and grows by doubling, as now. Tested to 4,096 lights (256 KB).
- **The change log keeps the old and the new light,** not just boxes. `SceneChanges` gains `lights: readonly LightChange[]`, each `{ index, before, after }` in the platform's units. This is technique-facing; hosts don't see it. Up to 1,024 light changes per frame are kept one by one; past that, the frame is treated as a global change (section 4.2).
- **An influence radius** for each light, computed on the CPU, bounds which probes a change can touch: the distance where the light's unwindowed irradiance falls below `1/1,000 × E_ref`. `E_ref` is the scene's reference irradiance: the sun's, or, with no sun, the brightest local light's at 1 m. So a dim lamp gets a small radius and a bright one a large radius. A light's `range`, when set, caps it. The radius is used only for this, never in shading.
- **Adapter:** lights are checked every frame today. That's fine for a few lights. With hundreds it could break the 0.2 ms still-scene target (an estimate: about 0.2 µs per light check; T1.7 only timed scenes with a few lights). Lights beyond the first 64 get the same budgeted walk and hot list as objects (three.md, "How changes are found"). `markChanged(light)` still sends a change the next frame.
- **The manifest's `motions` format** (section 7) is defined here too, because the native harness (T2.7) and the viewer (T2.8) both read it.

## 2. The deferred shadow-ray pass (platform hit shading)

**Why:** T1.9a measured shadow rays at 53% of Sponza's `ddgi/trace` (at 2,048 probes) and at no more than 1/7 of the trace bench's any-hit rate. The likely cause, until T2.0 measures it, is that the shadow-ray walk runs inside a kernel that already holds the closest-hit walk, the bounce lookup and the shading, and only some threads in a subgroup have a front-face hit to light.

**What changes:**

- `ddgi/trace` stops calling `giDirectIrradiance`. For each front-face hit of a blended ray, it writes a **surface record** and appends the ray to a compacted work list.
- A new **platform** kernel, `gi/direct`, runs over that list only. It picks lights, traces their shadow rays, and adds `albedo / π × direct` into the ray's radiance.
- `ddgi/blend` is unchanged: it reads finished radiances.

**Surface record** (`GiSurfaceRecord`, 24 bytes; the name `GiHitRecord` is already the batch tracer's record in `contract/queries.ts`):

| Field | Bytes | Why |
| --- | --- | --- |
| position, f32 × 3 | 12 | Shadow-ray origin; the push off the surface is applied in `gi/direct` |
| shading normal, octahedral 16 + 16 bits | 4 | The cosine term |
| ray-side geometric normal, octahedral | 4 | "Light from behind can't reach it", and the push-off direction |
| albedo, 10 + 10 + 10 bits unorm | 4 | So `gi/direct` never reads the material table. Albedo is at most 1 after `bounceAlbedoCap` (default 1.0); a cap above 1 is stored as a scale in the 2 spare bits |

**Memory:** the ray buffer grows on demand up to `maxProbesPerFrame` (default 4,096), and the viewer's warm-up uses 4,096-probe batches. Uncapped, surface records could reach 22 MB. So M2 splits any batch above 1,024 probes into rounds of 1,024 inside the same `encode` (trace, direct, blend, write per round). The ray scratch (ray results, surface records and the work-list counter, one buffer with three sections, the scene copy's pattern) then stays at most 1,024 × (256 × 8 + 224 × 24) bytes = **7.6 MB**, and `ddgi/trace` stays at 7 storage buffers.

**The work list** is filled with one `atomicAdd` per subgroup, with each thread's offset from `subgroupExclusiveAdd`. The declared fallback without subgroups is one atomic per thread. A 1-thread kernel turns the final count into indirect dispatch arguments.

**One code path, not two.** Light selection (section 3) becomes `giDirectLightSamples(surface, groupMask, random) -> GiDirectSamples` in the hit-shading module. `gi/direct` calls it and traces the samples. The existing `giDirectIrradiance` becomes a short wrapper that calls the same function and traces inline; the test views and the oracle keep using it. Both read the same selection code.

**Contract additions** (technique-facing; added to platform.md's "Revisions" when built):

- `HitShading.encodeDirect(encoder, { scratch, maxRays, groupLayers, gridBounds })` records `gi/direct` for a technique's ray scratch. `gridBounds` tells the platform where to build the light grid (section 3), because the platform doesn't know a technique's volumes.
- `GiSurfaceRecord` and its WGSL writer `giWriteSurfaceRecord(...)`, so techniques never depend on the byte layout.

**What it might buy.** Two things limit Sponza at 2 ms:

- **The per-probe cost.** This is where the 53% was measured. If shadow rays reach even half the bench's any-hit rate once they're coherent, full-subgroup work, the per-probe cost of trace plus direct drops to about 0.6× today's.
- **The fixed cost.** Sponza sits at 1.3–1.6 ms from 32 to 128 probes (NOTES). Most likely that's the latency of the longest walks in a small batch. A cheaper per-probe cost buys few extra probes while the fixed part dominates. Shorter kernels may cut that latency too, but nobody has measured it.

So the gain in probes per frame is unknown, somewhere between "little" and "about 1.6×". **The spike (T2.0) measures both parts before anything else is built.** Nothing below counts on the gain: the rates table above is the plan.

Bindings: `gi/direct` = tracer 3 + scene copy and shading 2 + ray scratch 1 + light grid 1 = **7**.

## 3. Many lights: a world-space grid of light candidates (ReGIR-style)

**Today:** `giDirectIrradiance` loops over every local light twice: first to find which lights reach the hit and how much, then to pick `maxShadowedLightsPerHit` (4) of them by systematic sampling. It decides *per hit* whether to sample, from how many lights reach that hit. The cost per hit grows with the light count. With 1,024 candles, 512 probes × 224 rays × 2 × 1,024 is 235 M light evaluations per frame, which doesn't fit.

**Proposed:** two regimes, chosen by a uniform (the scene's total local-light count, which the CPU knows):

1. **At most `exactLightLimit` local lights (default 8):** today's code, unchanged. All of them are looped over per hit and K = 4 are sampled when more than 4 reach the hit. Scenes with a few lamps behave exactly as in M1.
2. **More than that:** draw from a **light grid** built each frame:
   - **Cells:** a uniform grid over `gridBounds`, one cell per 2 × 2 × 2 probe cells, at most 32,768 cells (Sponza: about 2,400).
   - **Candidates:** each cell fills `R = 8` reservoirs from `C = 16` candidates each. Candidates are drawn from all local lights in proportion to their power, using a CDF rebuilt by a scan when lights change (the `gpu-scan` from B1, on `bvh-b1`, which DECISIONS merges into `m1`).
   - **How a candidate is weighted in a cell:** an upper bound of the light's unshadowed irradiance over the cell's box, using the nearest point of the box for the falloff and range window, and a cone test against the box's bounding sphere for spots. There's no cosine, because the surface normal isn't known per cell. *Why an upper bound over the box and not the cell centre:* a hit near a cell's edge can be lit by a light the centre can't see (outside its range or its cone). The weight must be above zero wherever the true contribution is, or the estimate is biased.
   - **At a hit:** `gi/direct` draws `maxShadowedLightsPerHit` of its cell's 8 reservoirs (default 4, the same setting as today), re-evaluates each light at the real hit with the real cosine, and traces one shadow ray each. Directional lights are always evaluated (one shadow ray each). So a hit costs at most K + (directional lights) shadow rays, whatever the light count. T2.3 measures K = 1, 2 and 4 for noise against cost.
   - **Outside the grid** (hits beyond every volume): candidates straight from the power CDF.

Data: `cells × R × 8 bytes` (light index, weight): 2,400 × 8 × 8 = 154 KB on Sponza, 2 MB at the 32,768-cell cap. Build cost: cells × R × C light evaluations, 2,400 × 128 = 0.3 M per frame on Sponza. Estimate: about 0.05 ms on the RTX.

**Why not ReSTIR's reuse over time and between neighbours too:** the field already averages hundreds of rays over many frames, so a fair sample per hit is enough. Reuse would add per-hit state that probes revisit only every few frames. Revisit if candles look noisy (risk 3).

**Why not a light tree first:** constant cost per hit matters more for us than the tree's better quality on tight clusters of lights. The grid is rebuilt from scratch every frame, so it needs no refit or rebuild policy under motion, and it lines up with the probe grid we already have. The light tree (Moreau 2019) is the fallback if the grid fails its noise check on `sponza-candles`.

**`sponza-candles`:** a stated subset of Intel's 10,000 candles (DECISIONS 2026-11-05). Proposed: 1,024 lights, and only those candles' geometry (the full candle mesh is 19.5 M triangles). The subset is question 2.

## 4. Change-driven updates: per-probe history and a GPU scheduler

This is how light changes get fast. It replaces the big-change rule (off by default since T1.5b) in that role, and it is the "priority list in front of the update cursor, in the same buffer" that ddgi.md left room for.

### 4.1 Per-probe history length

Each probe gets a 32-byte history record, `ddgiProbeHistory`: 0.6 MB at 20,000 probes. Halves are packed with `pack2x16float`, so `shader-f16` isn't needed.

| Word | Field | Meaning |
| --- | --- | --- |
| 0 | `n` (f32) | History length in updates. The blend uses `h = min(hysteresis, n / (n + 1))`, then sets `n ← n + 1`. With `hysteresis` 0.97, `n` stops mattering past 32. |
| 1 | `pendingChange` (f32) | Relative change since this probe's last update, added up over marks; reset by the blend. f32, so many small marks don't round away. |
| 2 | `urgency` (f32) | The largest change since the last update, for the scheduler. |
| 3 | `owed` (8 bits), flags | Updates this probe is owed at priority; one less per update. |
| 4 | `scheduledFrame` (u32) | The last frame it was put in the update list, so the fill never adds it twice. |
| 5, 6 | mean irradiance (rgb, halves) | Written by the blend, which already computes the probe's mean (the old big-change rule used it). Read by `ddgi/mark`. |
| 7 | spare | M3 uses it (`wakeUntil`). |

- **State 3 ("not updated yet") is just `n = 0`,** so `h = 0` on its first update, as now.
- **In steady state a probe behaves as in M1:** once `n ≥ 32`, `h = 0.97`.
- **Warm-up changes, for the better.** M1 jumps from `h = 0` straight to 0.97 after a probe's first update. With the history length, the next updates use 1/2, 2/3, 3/4 and so on, a true average, until 0.97. A new field converges faster and with less noise. The settled field is the same. T2.4a checks this after 600 frames.

**How short the history becomes after a change.** Say a probe's lighting changed by a relative amount `r`, and the scheduler promises it `K` updates inside the response window. The error left after those K updates is `r × n / (n + K)`. Setting that equal to τ gives

> `n ← min(n, τK / (r − τ))` when `r > τ`; untouched when `r ≤ τ`.

With τ = 5% and K = 8:

| Relative change r | New n | First update's weight on the old value | Left after 8 updates |
| --- | --- | --- | --- |
| 1.0 (a light switched on right next to it) | 0.42 | 0.30 | 5% |
| 0.5 | 0.89 | 0.47 | 5% |
| 0.2 | 2.7 | 0.73 | 5% |
| 0.1 | 8 | 0.89 | 5% |
| ≤ 0.05 | unchanged | 0.97 | n/a |

- *Why a growing history rather than a fixed fast hysteresis:* a fixed `h = 0.5` forgets in about 5 updates, but its stored value then carries the noise of only about 3 updates' worth of rays, for good. The growing history gets less noisy with every update and climbs back to 0.97 by itself.
- *Why the change comes from the scene, not from the probe's own estimate:* the estimate is noisy, and every rule built on it biased the field (T1.5b). The estimate below uses only light parameters and the field's stored data, so a still scene can never trip it.
- The same history applies to the distance atlas. For light changes it hardly matters, since distances don't change; M3 uses it for geometry.
- **The opt-in big-change setting** (`irradianceChangeThreshold`) is left as decided: off by default, and when enabled it still applies its own 0.75 factor on top of `h`.

### 4.2 The change estimate (`ddgi/mark`)

It runs only in frames with changes. The CPU writes a small table of changed lights (from `SceneChanges.lights`) and dispatches one thread per (changed light, probe inside that light's influence box clipped to the volume). Lights are walked one by one, never through the merged boxes, which lose which light changed.

**A local light moved, was recoloured, dimmed or toggled:**

- At the probe, for the light before and after, take its unshadowed **irradiance vector** per colour channel: `E_c × direction to the light`. This uses three's falloff, range window and cone (the same WGSL as hit shading, `giShadeLightSample`, without a surface cosine).
- `Δ = max over r, g, b of |E_after,c × d_after − E_before,c × d_before|`. *Why a vector, per channel:* a light that keeps its distance but moves to the other side of the probe changes the vector, not its length, and a colour change at equal brightness changes a channel, not the luminance. A luminance difference would miss both.
- `r = lightChangeScale × Δ / (mean_c + ε)`, where `mean_c` is the probe's stored mean irradiance in the same channel (history words 5–6). `lightChangeScale` (default 0.25, tuned in T2.4a) accounts for the probe seeing the light only through the surfaces it lights.
- **Visibility filter:** one Chebyshev test from the probe's distance moments toward the light (as in sampling) scales `r` by `max(chebyshev, 0.25)`. A probe behind a wall gets a smaller share but never zero, because it may see lit surfaces through a door.

**Global changes** (the sun's direction or colour, the sky; more than 1,024 light changes in one frame) mark every probe:

- `r` is the size of the change: the sun's angle moved (`2 sin(θ / 2)`) or `|1 − new / old|` for an intensity or colour ratio, per channel.
- It is multiplied by `0.25 + 0.75 × openness`, where openness comes from the probe's distance moments toward the old and new sun directions (a probe that sees open sky sees the change directly).

**Each marked probe:** `pendingChange += r`; `n ← min(n, τK / (pendingChange − τ))`; `owed ← max(owed, K)`; `urgency ← max(urgency, pendingChange)`.

Continuous motion (the sweep) marks a little every frame. Because `pendingChange` adds up until the probe is next updated, the history settles where it matches the rate of change: short near the moving light, unchanged far away. No special case is needed.

### 4.3 The GPU scheduler

The update list moves from the CPU (`UpdateCursor` plus `writeBuffer`) to the GPU, because the urgencies live there. It is still one buffer, filled with exactly `min(units, probes)` entries. So the trace, blend and write dispatches keep sizes the CPU knows, and the cost model still sees the units it handed out. It takes four small dispatches, every frame:

1. **`ddgi/schedule-count`** (one thread per probe): a probe with `owed > 0` gets the priority `urgency × cameraWeight`, sorted into one of 32 buckets, each twice as urgent as the one below. A histogram of the buckets is built with one atomic per subgroup (fallback: one per thread).
   - `cameraWeight = 1 / (1 + (distance to camera / (4 × spacing))²)`, times 0.25 outside the view frustum. *Why:* SPEC's response target is judged on the image, and nearby, visible probes are what the image shows. Lumen's priority does the same.
2. **`ddgi/schedule-select`** (one workgroup): finds the bucket where the list fills up, so that at most `P = priorityShare × units` probes are chosen. `priorityShare` defaults to 0.75, so the round robin always keeps a quarter of the budget and no probe is starved.
3. **`ddgi/schedule-write`** (one thread per probe): probes above that bucket, and as many as fit from it, go into the list and are stamped with the frame number.
4. **`ddgi/schedule-fill`** (one workgroup): the rest of the list comes from the existing scrambled order, `order[i] = (i × stride) mod N`. A window of `units` consecutive entries holds at most `P` stamped probes, so it always has enough unstamped ones. A running sum over the window picks them, and the cursor (now one GPU word) moves past the entries used.

Cost: four dispatches of a few µs each, plus one thread per probe. Estimate: under 0.05 ms on the RTX at 20,000 probes. They run every frame: the CPU can't see on the GPU whether probes are still owed, and skipping them would leave owed probes stranded (a Sponza-wide change keeps probes owed for hundreds of frames).

**What the budget buys.** With `U` probes per frame, a window of `W` frames and K owed updates, at most `P × W / K` changed probes get all their updates in time. With W = 30 and K = 8:

| Scene | U | P (75%) | Changed probes fully served within 0.5 s |
| --- | --- | --- | --- |
| Cornell | 995 | 746 | 2,800, more than its 1,728 probes |
| Two-room | 337 | 253 | 950, nearly all of its 1,071 |
| Sponza | ~190 | ~142 | ~530 |

A lamp whose influence box holds more probes than that is served nearest-and-most-changed first. The visible image meets the target, while far, weakly changed probes finish later.

### 4.4 Light groups make some changes need no updates at all

See section 5: a dimmer or colour change across a whole group skips sections 4.2 and 4.3 entirely.

## 5. Light groups

By linearity, the field for "group g at weight w" is w × the field for group g at weight 1, bounces included, as long as each group's bounce reads only its own layer (HomeFig rule 1).

**Layers:** one irradiance layer per light group in use (contract groups 0–3), plus an **environment layer** for the sky and emissive surfaces. HomeFig rule 2 applies: the sky's shape changes with time of day, so it gets a layer of its own and is re-traced, never just scaled. With groups off there is one layer, and everything below reduces to M1's kernels with a loop count of 1.

| What | Layers | Notes |
| --- | --- | --- |
| Irradiance, per group | G + 1 (with the environment) | Stored at the lights' real intensity and colour (HomeFig rule 3) |
| Composed irradiance | 1 | What the renderer samples. Not needed when there's only one layer |
| Distance, probe data | 1 | Geometry only, shared |

- **Trace:** one closest hit per ray, as now. The bounce lookup works out the 8 probes' weights (distance, Chebyshev, crush) once and fetches each layer's irradiance with them. Each extra layer costs only its irradiance fetches, about a quarter of the lookup.
- **Direct light:** the hit's light samples are shared; each goes into its own group's layer. Each layer's estimate stays unbiased; groups with few lights get noisier.
- **Ray results:** 8 bytes per ray per layer, in the ray scratch.
- **Blend:** loops the 36 irradiance texels per layer; the 196 distance texels only once.
- **`ddgi/compose`:** composed = Σ weight_g × layer_g per texel, borders included. It runs for the tiles updated this frame, and for every tile in a frame where a weight changes. **A group weight change shows in the same frame,** with no lag and no noise.
- **Weights** are technique settings, `lightGroups: { count, weights: Color[], environmentWeight: Color }`: linear RGB multipliers on top of the lights' own intensities. The three adapter exposes `threeGI.setGroupWeight(group, color)`.

**Memory** (an irradiance tile is 8 × 8 texels × 8 bytes = 512 bytes per probe per layer):

| Scene (probes) | Groups off | 2 groups + environment + composed | 4 groups + environment + composed |
| --- | --- | --- | --- |
| Two-room (1,071) | 0.5 MB | 2.2 MB | 3.3 MB |
| Furnished room (4,998) | 2.6 MB | 10.2 MB | 15.4 MB |
| Sponza (19,425) | 9.9 MB | 39.8 MB | 59.7 MB |

Distance (1 KB per probe) doesn't change. Four groups fit the furnished room's 64 MB budget, but they are expensive on Sponza. So **groups are opt-in and off by default,** and the host picks how many.

**Cost estimate** at 4 groups:

- trace about +10% (the extra irradiance fetches only);
- blend about +60% (the irradiance texel work × 5);
- compose about 0.1 ms on Sponza when a weight changes;
- in total about +25–35% per probe updated.

Shading cost doesn't change, because the renderer samples one composed atlas. *Why not sample every layer in the fragment shader:* irradiance fetches were about 0.21 ms of T1.6's sampling cost (a run at a locked 1200 MHz clock, NOTES), and five layers would push sampling past 1.0 ms.

## 6. Sun and sky with time of day

**Today:** a `procedural` environment is built on the CPU (Preetham, calibrated with the Ineichen–Perez ratio; `calibrated-sky.ts`) and uploaded. One workgroup then resamples it to the 256 × 128 radiance map and projects it to 18 SH coefficients. That runs only when `setEnvironment` is called. The projection kernel takes about 1 s to compile (T1.9a), and the CPU build isn't meant to run every frame.

**M2:**

- **A GPU procedural source.** The resample kernel evaluates Preetham directly (a new `SKY_SOURCE_KIND`). The calibration scale is computed on the CPU per sun direction (a few scalar formulas) and passed as a uniform. The CPU code stays: it writes the shared sky files for Cycles and native, and a test holds the GPU map to it (mean within 0.1%, every texel within 1%).
- **The SH projection works in two levels:** many workgroups write partial sums and one adds them up. Subgroups first, with the existing workgroup-memory fallback. On a procedural sky it integrates over the 256 × 128 map itself.
- **When it re-bakes:** when the sun moves more than 0.05°, or turbidity, intensity or `sunIlluminance` change; at most once per frame. Estimated GPU cost under 0.1 ms per re-bake; CPU under 0.05 ms.
- **The slow compile:** T2.6 finds why the projection takes about 1 s to compile (perhaps the unrolled 18-term basis inside the loop) and gets it under 0.2 s. That also helps startup (DECISIONS: "find the slow sky kernel").
- **The sun** is a directional light and a separate object (light buffer, shadow rays, its own group when groups are on). Keeping the sun light and the sky's sun in step is the host's job; the viewer's slider moves both. The three adapter already maps pmndrs/sky and `SkyMesh` (three.md, "The sky").

**How fast the field follows.** A change of the sky or the sun's direction marks every probe (section 4.2). For a big step, all N probes need K updates, which takes about `N × K / U` frames of the whole budget:

| Scene | Probes | U | Whole volume within 5% after a big step |
| --- | --- | --- | --- |
| Cornell | 1,728 | 995 | ~0.23 s |
| Two-room | 1,071 | 337 | ~0.42 s |
| Furnished room | 4,998 | to measure | ~1.7 s if U ≈ 400 |
| Sponza | 19,425 | ~190 | ~14 s |

The camera's neighbourhood comes first (camera weight), so the view settles well before the whole volume does. But on the furnished room and Sponza, even the view may not meet 0.5 s after a big jump. Question 4 asks how the target should be read for global changes.

- **A slow time-of-day animation** (an hour of sun in 10 s, 1.5° per second) marks every probe a little each frame. Their histories shorten only slightly, and the field follows smoothly.
- **Dragging the slider fast** on Sponza lags by many seconds away from the camera.
- **Making that instant** would need each probe texel to store its response to the sky's SH patterns (Far Cry 3's approach): about 16 coefficients × 3 colours per texel, roughly 12× the irradiance atlas (Sponza +110 MB). That's question 1; it isn't in the default plan.

**Native:** the native harness's miss shader sees one flat colour (DECISIONS 2026-11-04). M2 gives it the texture sky and an animated sun, so the time-of-day comparison is like for like (T2.7).

## 7. The viewer

- **Gizmos:** three's `TransformControls` on light helpers: drag to move, rotate spots. Selecting a light opens its panel.
- **Sliders and toggles:** intensity (log scale), colour, range, cone and penumbra for spots, shadows on or off, group. Toggling sets intensity to 0, so three doesn't recompile its materials (NOTES: removing a three light recompiles).
- **Group dimmers:** a weight slider and a colour per group, plus the environment layer. The hover help says "instant: no update needed".
- **Time of day:** hour and day-of-year sliders. They drive the sun's direction (a simple solar-position formula), the scene's sun light and the GI sky together. A play button runs a 10-second day.
- **Scripted motions** from the manifest: `motions: [{ id, target: 'light:<id>', keyframes: [{ t, position, intensity, color }], loop }]`. Two come built in for each lit scene:
  - **light sweep:** a lamp moving along a path at 1 m/s (two-room: across the lit room and through the doorway; Sponza: along the lower colonnade);
  - **step:** the same lamp jumping between two keyframes every 3 s, for measuring response.
- **Response meter,** measured in **GI-only mode** (so three's own direct light, which changes instantly, doesn't count). Pick a screen rectangle. The viewer plots its mean per frame, marks "first frame that changed by more than 1%" and "first frame within 5% of settled", and writes the curve to CSV. "Settled" is the mean after 4 s with the budget raised to 16 ms (a test mode, labelled as such).
- **Native side by side:** the native harness plays the same manifest motion (T2.7) and records the meter's curve and one frame in four. A compare page plays both recordings in sync, with both curves on one plot. Showing native live inside the viewer isn't attempted.
- **Stats** gain: probes updated by priority and by round robin, probes owed, shadow rays per frame, `gi/direct` ms, compose ms, sky re-bake ms, light count.

## 8. Area lights: not in M2 (carried from M1)

DECISIONS 2026-11-05 (the viewer entry) asks for area lights, so an emitter's direct light can be analytic and DDGI holds only its bounce. The obvious route clashes with SPEC: three's `RectAreaLight` casts no shadows, and SPEC's HomeFig lessons say to avoid area lights without shadows. The emissive Cornell's lamp, lit directly by an unshadowed area light, would light the floor behind the blocks.

So the proposal is to keep M1's split: DDGI lights the emitter, and the viewer adds no analytic light. Platform `rect` lights can come later, together with shadowed direct light on the host side. They'd also need an in-plane orientation in the light record, which the 64-byte record doesn't hold today. That's question 6.

## How it fits the existing kernels (no parallel GPU paths)

| Kernel | M1 | M2 |
| --- | --- | --- |
| `ddgi/mark` | none | new; only in frames with changes |
| `ddgi/schedule-*` | update list from the CPU cursor | the GPU scheduler writes the same buffer; the CPU cursor is removed, not kept beside it |
| `ddgi/trace` | trace, shade, direct light inline | trace, shade, bounce per layer, surface records; **no `giDirectIrradiance` call** |
| `gi/direct` (platform) | none | new; takes over direct light for every technique that traces |
| light grid build (platform) | none | new; only above `exactLightLimit` local lights |
| `ddgi/blend` | one global hysteresis | per-probe `h` from the history record; loops layers; writes the probe's mean |
| `ddgi/compose` | none | only with light groups on |
| `ddgi/write` | unchanged | unchanged |
| sky resample and projection | CPU-built procedural sky, one-workgroup SH | GPU procedural source, two-level SH |

Every new kernel that reduces or compacts is subgroups first, with one declared fallback (per-thread or workgroup atomics), and a test runs both on the same input (the M1 rule).

**Storage buffers per kernel** (the default limit is 8):

| Kernel | Count |
| --- | --- |
| `ddgi/trace` | tracer 3 + scene copy and shading 2 + update list + ray scratch = **7** |
| `gi/direct` | tracer 3 + scene copy and shading 2 + ray scratch + light grid = **7** |
| `ddgi/blend` | update list, ray scratch, scratch tile, census, history = **5** |
| `ddgi/mark` | light-change table, history = 2 (atlases as textures) |
| `ddgi/schedule-*` | history, update list, schedule state = 3 |

## Cost against the budgets (estimates, RTX 3060, to be measured)

| Item | When | GPU ms |
| --- | --- | --- |
| `ddgi/trace` + `gi/direct` on Sponza | always | per-probe part about 0.6× of M1's `ddgi/trace`; the fixed part unknown (T2.0) |
| Scheduler (4 dispatches) | every frame | < 0.05 |
| `ddgi/mark` | frames with changes | < 0.02 for a few lights; < 0.05 for a global change at 20k probes |
| Light grid build | above `exactLightLimit` local lights | ~0.05 on Sponza with 1,024 lights |
| `ddgi/compose` | groups on | updated tiles only: < 0.01; a weight change on Sponza: ~0.1 |
| Sky re-bake | the sky changed | < 0.1 |

All of it sits inside the 2 ms update budget: the platform's cost model measures the `ddgi/*` passes and `gi/direct` together, so extra fixed costs shrink the probe count, not the frame. Shading cost doesn't change.

Memory: history 32 bytes per probe, ray scratch at most 7.6 MB, light grid at most 2 MB, light groups as tabled.

## Proposed task breakdown

Board format. IDs are proposals; T1.11 assigns the real ones. Every GPU check runs on the RTX 3060 and the Intel UHD, says which adapter, and uses the bench lock (with Dennis's GPU apps closed for p95 and timings, DECISIONS 2026-11-05).

| ID | Task | Agent | Size | Depends on |
| --- | --- | --- | --- | --- |
| T2.0 | Spike: deferred shadow-ray pass on Sponza | implementer | S | M1 done |
| T2.1 | Light buffer v2, manifest `motions`, `sponza-candles` | implementer + wrangler | M | none |
| T2.2 | Deferred direct light (`gi/direct`) in platform hit shading | implementer | M | T2.0 |
| T2.3 | Many-light sampling: the light grid | implementer | M | T2.1, T2.2 |
| T2.4a | Per-probe history and `ddgi/mark` | implementer | M | T2.1 |
| T2.4b | GPU scheduler with camera priority | implementer | M | T2.4a |
| T2.5 | Light groups: layers, compose, weights, adapter API | implementer | M | T2.2, T2.4a |
| T2.6 | Sky and time of day: GPU procedural source, two-level SH, compile fix | implementer | M | T2.4a |
| T2.7 | Native harness: manifest motions, texture sky, animated sun | native-runner | M | T2.1 |
| T2.8 | Viewer: gizmos, sliders, groups, time of day, motions, response meter, native compare page | implementer | M | T2.4b, T2.5, T2.6, T2.7 |
| T2.10 | M2 measurements against targets | gpu-tester, native-runner | M | all above |
| T2.11 | M2 note and M3 breakdown | main session | S | T2.10 |

T2.9 is left free for area lights, if Dennis wants them (question 6).

The startup items DECISIONS moved to M2 ("fast first GI", the scene copy off the main thread, compiling less, file loading) are designed in [m3-moving.md](m3-moving.md), because they share the BVH and scene-copy machinery. Their rows (T2.12–T2.15) are listed there.

### Draft done-whens

- **T2.0 Spike: deferred shadows.** A throwaway branch splits Sponza's `ddgi/trace` into trace + `gi/direct` with surface records.
  - Done when: the fixed and per-probe cost of the pair (from a batch-size curve, 16 to 512 probes) and the shadow-ray rate against the bench's any-hit rate are recorded on both GPUs (3 runs); and the field matches M1's within 1% mean on Cornell and Sponza after 300 frames.
  - Decision rule, set before measuring: go ahead with T2.2 if Sponza fits ≥ 1.25× the probes at 2 ms on the RTX. Otherwise record why, and the main session decides.
  - Show: the cost curves. Record: `timing.csv` `batch_cost/` rows.
- **T2.1 Light buffer v2.**
  - Done when:
    1. creating 4,096 lights and destroying them in random order creates 0 pipelines (counter);
    2. a test round-trips every `LightChanges` field for each light type through `SceneChanges.lights`;
    3. the adapter's still-frame CPU time on `sponza-candles` is ≤ 0.2 ms (median of 300 frames, RTX laptop);
    4. `sponza-candles` fetches and verifies with the subset Dennis picks (licence, size, triangle and light counts, checksum).
  - Show: `sponza-candles` in the viewer. Record: light and triangle counts in PROGRESS.
- **T2.2 Deferred direct light.**
  - Done when:
    1. direct irradiance through `gi/direct` matches the inline `giDirectIrradiance` on T1.4's oracle points (mean within 0.5%), and sun-on-plane matches E·cos θ within 0.5%;
    2. the subgroup and fallback work lists give the same field (mean within 0.1% after 300 frames);
    3. Sponza's probes per frame at 2 ms reach T2.0's measured gain within 10%;
    4. the furnace, Cornell and two-room checks of T1.5 still pass.
  - Record: `timing.csv` rows; the shadow-ray rate.
- **T2.3 Many-light sampling.**
  - Done when, on a test scene with 1,024 point lights:
    1. direct irradiance from `gi/direct`, averaged over 1,024 frames, matches an exhaustive evaluation within 1% mean, with 99% of 1,000 test points within 5%;
    2. `sponza-candles` costs ≤ 1.3× per updated probe compared with `sponza` lit by the sun only;
    3. the grid build takes ≤ 0.1 ms on the RTX.
  - Record: noise and cost at K = 1, 2, 4.
- **T2.4a Per-probe history and `ddgi/mark`.**
  - Done when:
    1. on a still scene, after 600 frames the field matches M1's within 0.5% mean;
    2. on two-room in a test mode that updates every probe every frame (budget 16 ms), moving the lamp 2 m (the step) gets within 5% of settled in ≤ 0.5 s (response meter, GI only, `lit-room` bookmark), so this tests the history without the scheduler;
    3. a still scene marks 0 probes over 600 frames (counter);
    4. marked probes against probes whose true change was > 5% (from a re-converged field) are recorded, with a lit-room recall of at least 95%.
  - Record: `lightChangeScale` chosen, and the counts.
- **T2.4b GPU scheduler.**
  - Done when, at the default budget, in GI-only mode:
    1. two-room and Cornell (point light) step changes show the next frame (ROI changes by > 1%) and are within 5% of settled in ≤ 0.5 s on the RTX and ≤ 1.5 s on the Intel UHD;
    2. Sponza's step change at bookmark 1 meets the same on the RTX;
    3. no probe goes un-updated for more than 2 full round-robin passes during a 60 s sweep (census);
    4. the scheduler costs ≤ 0.05 ms on the RTX at 20,000 probes, and the subgroup and fallback histograms agree exactly.
  - Record: response curves (CSV), per GPU.
- **T2.5 Light groups.**
  - Done when:
    1. on two-room with 2 groups, a weight change shows in the same frame, and the image matches a field re-converged from scratch at that weight within 1% mean (linearity);
    2. with groups off, the field is within 0.1% mean of T2.4b's and the cost within 3%;
    3. with 4 groups, the cost per updated probe is ≤ 1.4× groups off;
    4. GI sampling at 1440p stays ≤ 1.0 ms (T1.6's view).
  - Record: memory per layer per scene.
- **T2.6 Sky and time of day.**
  - Done when:
    1. the GPU procedural map matches the CPU sky within 0.1% mean and 1% per texel, and `giSkyIrradiance` still meets ±0.5% on open ground;
    2. a re-bake costs ≤ 0.1 ms GPU and ≤ 0.05 ms CPU;
    3. the projection compiles in ≤ 0.2 s cold on the RTX;
    4. a 30° sun step on the furnished room is timed (whole volume, and the bookmark view, GI only), against the reading Dennis picks in question 4.
  - Record: those times.
- **T2.7 Native harness.**
  - Done when:
    1. `play.ps1 -Scene two-room -Motion light-sweep` plays the manifest motion at the viewer's speed (positions within 1 cm at matching times);
    2. the miss shader reads the shared sky file;
    3. the sun can follow a time-of-day motion;
    4. the default config renders as before (by eye plus one capture).
  - Record: native response curves for the step (3 runs).
- **T2.8 Viewer.**
  - Done when:
    1. the sweep and step motions run on two-room, Cornell and Sponza;
    2. the response meter's CSV for a two-room step matches the T2.4b test page within 2 frames;
    3. the native compare page plays both recordings in sync;
    4. every control has hover help, and Dennis can drive it without instructions.
  - Show: the viewer URL with two-room at `lit-room`.
- **T2.10 M2 measurements.** The table at the top, per GPU, plus native's response curves for the same motions (3 runs). Done when each target is met or its miss is explained and agreed with Dennis, and he has judged the sweep next to native.

## Risks, and what to spike first

1. **The deferred pass may buy little** at Sponza's operating point, where the fixed cost dominates. Then Sponza stays near 190 probes per frame, and the response target is met only near the camera. This is the largest lever and the largest uncertainty. *Spike first: T2.0.*
2. **Short histories flicker.** At `n` = 0.42 the first update is 70% one noisy estimate. If that shows as a flash, raise the floor on `n`, or give marked probes 2× rays for their first update (at a cost). T2.4a's two-room step shows it.
3. **Light-grid noise with many small lights.** Candles have short ranges, and a cell 2 probe spacings wide may hold many lights that matter only at its corners. The options are smaller cells near lights, more reservoirs, or Moreau's light tree. `sponza-candles` decides.
4. **The change estimate is a stand-in.** `lightChangeScale` and the visibility floor are guesses until T2.4a compares marked probes against probes that really changed. Too generous wastes budget; too stingy leaves stale probes until the round robin reaches them.
5. **Global changes on big scenes** can't meet 0.5 s everywhere at 2 ms (section 6). Question 4.
6. **Native motion** needs fork changes, because the harness has no animation. Two-room first, Sponza second.

## Questions for Dennis

1. **Time-of-day scrubbing:** is it fine if dragging the slider fast on Sponza lags by many seconds away from the camera (the default plan)? Or should we build the optional "instant sky" (each probe stores its response to the sky's patterns, about 12× the irradiance memory, +110 MB on Sponza) for scenes that can afford it?
2. **`sponza-candles`:** how many of the 10,000 candles? I'd propose 1,024 lights with only those candles' geometry.
3. **Light groups:** off by default, the host picks how many, emissive surfaces go in the environment layer. OK? And is an explicit group-weight call enough, or should dimming every light of a group the usual three.js way (scaling each light's `intensity`) be spotted and treated as a group weight?
4. **Global changes:** for the sun or the sky on big scenes, may "within 5% in 0.5 s" be judged on the view (camera-near probes first), with the rest of the volume following later (about 1.7 s on the furnished room, about 14 s on Sponza at today's rates)? Or should global changes have their own target?
5. **Native side by side:** is judging by eye from synced recordings, with both response curves on one plot, enough? Or do you want a number (say, our time to 5% within 1.5× of native's)?
6. **Area lights:** three's `RectAreaLight` has no shadows, which SPEC says to avoid. Keep the emitter lit by DDGI only, as in M1, and leave area lights for after M2?
</content>
</invoke>
<invoke name="Bash">
<parameter name="command">cd "C:/Users/denni/Documents/dev/webgpu-ddgi/.claude/worktrees/m2-m3-design-71df60" && sed -n 55,64p scenes/manifests/furnished-room.json; grep -n "lit\b\|\"id\"" scenes/manifests/sponza.json | head