# M2 design draft: dynamic lights

Status: **draft for Dennis's review, 2026-11-05, revised after an Opus review the same day and again after Dennis's answers (time of day rebuilt so nothing takes minutes to build; memory sized for mid- to high-end GPUs; the Intel UHD for correctness only).** Nothing here is decided. It is the input for T1.11 ("M1 note and M2 breakdown"). Where it would change a default or a settled choice from M1, it says so; those become DECISIONS entries only once Dennis agrees.

It builds on [ddgi.md](ddgi.md) (the field and update loop), [platform.md](../api/platform.md) (the contract), [three.md](../api/three.md) (the adapter) and the M1 entries in [DECISIONS](../DECISIONS.md). The M3 draft, [m3-moving.md](m3-moving.md), reuses the change-driven scheduler designed here for moved geometry.

## In one paragraph

Lights already live in a growable table, and adding one never recompiles anything. M2 makes a light change show up quickly and cheaply. Three changes do most of the work:

1. **The platform tells DDGI what changed:** the old and the new light, not just a box. A small GPU kernel turns that into "how much did this probe's lighting change", worked out from the light's own numbers, so ray noise can't fake a change.
2. **Each probe gets its own history length** instead of the fixed 0.97. It is shortened in proportion to the change. A GPU scheduler puts the most changed probes near the camera at the front of the update list.
3. **Shadow rays move into their own pass in platform hit shading.** Today they sit inside the trace kernel, where Sponza loses half its update time. The new pass picks lights from a world-space grid of pre-chosen candidates, so a thousand candles cost about what a few lamps do.

Light groups (one irradiance layer per group, added up into the atlas the renderer reads) make a dimmer or a colour change across a group instant.

For time of day, the sun's colour and brightness are instant and exact, because the sun gets a layer of its own. The sky is nudged in the same frame for what each probe sees of it directly. That uses a first-order sky transfer, built from the field's own rays no slower than the field converges (about 1.2 s into Sponza's warm-up). The sun's direction, and whatever the nudge leaves, are re-traced nearest the camera first, blending rather than popping. Nothing takes minutes to build (Dennis, 2026-11-05).

Dennis answered this draft's first questions on 2026-11-05; they're listed at the end.

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
- **SH patterns and first-order transfer:** spherical harmonics (SH) are fixed patterns over the sphere, and any smooth sky is close to a weighted sum of the first 9. A probe's *first-order transfer* for a pattern is how much of that pattern the probe sees directly, through openings, with no bounces.
- **GPU-bound:** measured with the GPU kept busy and its clocks up, not at 60 Hz under light load, where the laptop GPU lowers its clocks.
- **Linearity (superposition):** twice the light gives twice the bounce, and two lights give the sum of what each gives alone.

## What M2 must deliver (SPEC)

| Target | RTX 3060 and up, Apple M5 Pro (1440p) |
| --- | --- |
| Light change visible | next frame |
| Light change within 5% of settled | ≤ 0.5 s |
| GI update cost | ≤ 2.0 ms at the default budget |
| GI sampling cost | ≤ 1.0 ms (no regression from T1.6's 0.905 ms) |
| Lights | ≥ 32; adding or removing one never recompiles |
| Adapter cost | ≤ 0.2 ms still scene, ≤ 0.5 ms one moving thing |
| Sweep | looks right to Dennis next to native doing the same motion |

M1's accuracy, leak, furnace and stability checks must still pass on still scenes.

- **Hardware** (Dennis, 2026-11-05; SPEC and DECISIONS already revised on `m1`): the targets are for the RTX 3060 and up and an Apple M5 Pro. The Intel UHD is a **correctness device only**, with no performance targets ("we arent building for the intel"). It runs the oracle, furnace, accuracy and leak checks; its timings aren't gated.
- **Timings are GPU-bound** (Dennis: "Running at 60hz seems like a fail"): with the GPU loaded and its clocks up.

## The update rates this plan works with

Everything about response time depends on how many probes are updated each frame.

**Budget v2** (accepted by Dennis, 2026-11-05; ddgi.md "Budget v2") sets that number by work, not by milliseconds:

- The platform hands DDGI a fixed number of probes a frame. It's calibrated once at load, with the clocks up, so that the batch costs about the budget (2 ms) back to back.
- **A floor:** never fewer than a full sweep in 4 s at the host's frame rate. That's 81 probes a frame on Sponza at 60 Hz. On the RTX, 81 Sponza probes cost 1.4 ms back to back (ddgi.md), so the floor sits inside 2 ms.
- Milliseconds only cut the number when frames are actually late.

T1.9a's numbers came from the old millisecond controller at 60 Hz. DECISIONS says other apps kept the RTX's clocks up then, but NOTES records them still 1.8–2.1× down on Cornell (512 probes: 1.29 ms at 60 Hz against 0.70 ms back to back), so the rooms' numbers are probably on the low side. They are the best numbers there are for the small scenes until T1.9 re-measures under Budget v2:

| Scene | Probes | Probes per frame this doc uses (RTX) | Where it comes from |
| --- | --- | --- | --- |
| Cornell | 1,728 | ~890 | T1.9a median (875–910) |
| Two-room | 1,071 | ~300 | T1.9a median (294–340) |
| Furnished room | 4,998 | **not measured** | |
| Sponza | 19,425 | **81** | Budget v2's floor (T1.9a's old controller gave 30–43) |
| Bistro exterior's camera volume (m3-moving.md) | 12,288 | **51** | Budget v2's floor (T1.9a gave 20–27 on the fixed volume) |

- **Sponza and Bistro use the floor,** a lower bound. The calibrated number may be higher; T1.9's re-measure says.
- **Every time in this doc gets re-derived** once the M1 lead's GPU-bound numbers land.
- Section 2's deferred shadow pass could raise the calibrated rate further. Nothing below assumes it does.

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
| **Far Cry 3** (Gilabert and Stefanov, "Deferred Radiance Transfer Volumes", GDC 2012) and PRT (Sloan, Kautz, Snyder 2002) | Probes store how they respond to each of a few sky "basis" patterns, so the sky and sun relight instantly. | **The instant sky** (section 6.3), cut down to what a probe sees of the sky directly, so it builds from the field's own rays in seconds. Re-tracing near the camera stays the main path. |

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

**Memory:** the ray buffer grows on demand up to `maxProbesPerFrame` (default 4,096), and the viewer's warm-up uses 4,096-probe batches. Uncapped, surface records could reach 22 MB. So M2 splits any batch above 1,024 probes into rounds of 1,024 inside the same `encode` (trace, direct, blend, write per round). The ray scratch (ray results, surface records and the work-list counter, one buffer with three sections, the scene copy's pattern) is then at most 1,024 × (256 × 8 × layers + 224 × 24) bytes: **7.6 MB** with one layer, 11.8 MB with three (lights, sun and sky, with the instant sky on). `ddgi/trace` stays at 7 storage buffers.

**The work list** is filled with one `atomicAdd` per subgroup, with each thread's offset from `subgroupExclusiveAdd`. The declared fallback without subgroups is one atomic per thread. A 1-thread kernel turns the final count into indirect dispatch arguments.

**One code path, not two.** Light selection (section 3) becomes `giDirectLightSamples(surface, groupMask, random) -> GiDirectSamples` in the hit-shading module. `gi/direct` calls it and traces the samples. The existing `giDirectIrradiance` becomes a short wrapper that calls the same function and traces inline; the test views and the oracle keep using it. Both read the same selection code.

**Contract additions** (technique-facing; added to platform.md's "Revisions" when built):

- `HitShading.encodeDirect(encoder, { scratch, maxRays, groupLayers, gridBounds })` records `gi/direct` for a technique's ray scratch. `gridBounds` tells the platform where to build the light grid (section 3), because the platform doesn't know a technique's volumes.
- `GiSurfaceRecord` and its WGSL writer `giWriteSurfaceRecord(...)`, so techniques never depend on the byte layout.

**What it might buy.** Two things limit Sponza at 2 ms:

- **The per-probe cost.** This is where the 53% was measured. If shadow rays reach even half the bench's any-hit rate once they're coherent, full-subgroup work, the per-probe cost of trace plus direct drops to about 0.6× today's.
- **The fixed cost.** On Sponza a batch of 20 probes already costs 1.55 ms (T1.9a). Most likely that's the latency of the longest walks in a small batch. A cheaper per-probe cost buys few extra probes while the fixed part dominates. Shorter kernels may cut that latency too, but nobody has measured it.

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

**`sponza-candles`:** a stated subset of Intel's 10,000 candles (DECISIONS 2026-11-05). Dennis: "1024 is fine for now". The count is a scene setting (`candles`, default 1,024, up to 10,000), so it can be raised as a stress test. Only the chosen candles' geometry is loaded (the full candle mesh is 19.5 M triangles). The light-grid checks run at 1,024; higher counts are recorded, not gated.

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

**The ramp rule, for changes that keep going.** A slow, steady change (a lamp gliding, a day passing) moves a probe only a little between two of its updates, so the step rule above leaves its history at 32 updates. A history of `n` updates trails a steady change by about `n` updates' worth of it, so the probe would lag far behind. So when a probe was marked before its previous update too (a flag in the history record), the blend also applies

> `n ← min(n, τ / r)`, with `r` the change since its last update,

which keeps the error from lagging near τ. A one-off small step (marked once) isn't affected.

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

**Global changes** (the sun's direction, the sky; more than 1,024 light changes in one frame) mark every probe. On big volumes, far probes are owed 2 updates instead of 8 (6.4). The exceptions:

- a sun change of colour or intensity alone marks nothing: it becomes the sun layer's weight (6.1);
- a sky change marks probes with a built transfer at `r × skyResidualShare` (default 0.3), because the nudge has already moved most of it, unless the nudge's brightness ratio was clamped (6.3).

For a sky change, `r = max over channels of (|Δc₀| + |Δc₁…₃|) / c₀` (the change of the sky's band-0 and band-1 radiance SH, relative to its mean), times the same openness factor as below.

- `r` is the size of the change: the sun's angle moved (`2 sin(θ / 2)`) or `|1 − new / old|` for an intensity or colour ratio, per channel.
- It is multiplied by `0.25 + 0.75 × openness`, where openness comes from the probe's distance moments toward the old and new sun directions (a probe that sees open sky sees the change directly).

**Each marked probe:** `pendingChange += r`; `n ← min(n, τK / (pendingChange − τ))`; `owed ← max(owed, K)`; `urgency ← max(urgency, pendingChange)`; and the ramp flag is set if it was marked since its previous update.

Continuous motion (the sweep) marks a little every frame. Because `pendingChange` adds up until the probe is next updated, the history settles where it matches the rate of change: short near the moving light, unchanged far away. No special case is needed.

### 4.3 The GPU scheduler

The update list moves from the CPU (`UpdateCursor` plus `writeBuffer`) to the GPU, because the urgencies live there. It is still one buffer, filled with exactly `min(units, probes)` entries. So the trace, blend and write dispatches keep sizes the CPU knows, and the cost model still sees the units it handed out. It takes four small dispatches, every frame:

1. **`ddgi/schedule-count`** (one thread per probe): a probe with `owed > 0` gets the priority `urgency × cameraWeight`, sorted into one of 32 buckets, each twice as urgent as the one below. A histogram of the buckets is built with one atomic per subgroup (fallback: one per thread).
   - `cameraWeight = 1 / (1 + (distance to camera / (4 × spacing))²)`, times 0.25 outside the view frustum. *Why:* SPEC's response target is judged on the image, and nearby, visible probes are what the image shows. Lumen's priority does the same.
2. **`ddgi/schedule-select`** (one workgroup): finds the bucket where the list fills up, so that at most `P = priorityShare × units` probes are chosen. `priorityShare` defaults to 0.75, so the round robin always keeps a quarter of the budget and no probe is starved.
3. **`ddgi/schedule-write`** (one thread per probe): probes above that bucket, and as many as fit from it, go into the list and are stamped with the frame number.
4. **`ddgi/schedule-fill`** (one workgroup): the rest of the list comes from the existing scrambled order, `order[i] = (i × stride) mod N`. A window of `units` consecutive entries holds at most `P` stamped probes, so it always has enough unstamped ones. A running sum over the window picks them, and the cursor (now one GPU word) moves past the entries used.

Cost: four dispatches of a few µs each, plus one thread per probe. Estimate: under 0.05 ms on the RTX at 20,000 probes. They run every frame: the CPU can't see on the GPU whether probes are still owed, and skipping them would leave owed probes stranded (a Sponza-wide change keeps probes owed for hundreds of frames).

**What the budget buys.** With `U` probes per frame (the rates section), a window of `W` frames and K owed updates, at most `P × W / K` changed probes get all their updates in time. With W = 30 and K = 8:

| Scene | U | P (75%) | Changed probes fully served within 0.5 s |
| --- | --- | --- | --- |
| Cornell | ~890 | ~670 | ~2,500, more than its 1,728 probes |
| Two-room | ~300 | ~225 | ~840, most of its 1,071 |
| Sponza | 81 (floor) | ~61 | ~230 |

A lamp whose influence box holds more probes than that is served nearest-and-most-changed first. The visible image meets the target, while far, weakly changed probes finish later.

### 4.4 Light groups make some changes need no updates at all

See section 5: a dimmer or colour change across a whole group skips sections 4.2 and 4.3 entirely.

## 5. Light groups

By linearity, the field for "group g at weight w" is w × the field for group g at weight 1, bounces included, as long as each group's bounce reads only its own layer (HomeFig rule 1).

**Layers:** one irradiance layer per light group in use (contract groups 0–3), plus an **environment layer** for the sky and emissive surfaces. HomeFig rule 2 applies: the sky's shape changes with time of day, so it gets a layer of its own and is re-traced, never just scaled. With groups off there is one layer, and everything below reduces to M1's kernels with a loop count of 1.

| What | Layers | Notes |
| --- | --- | --- |
| Irradiance, per group | G + 1 (with the environment) | Stored at the lights' real intensity and colour (HomeFig rule 3) |
| Composed irradiance | 1 (its own texture) | What the renderer samples. Not needed when there's only one layer |
| Distance, probe data | 1 | Geometry only, shared |

- **Where the layers live:** as blocks side by side inside each array layer of the irradiance atlas (6.3, "Layout"), not as more array layers. The atlas already uses one array layer per probe row, and the default limit is 256. The composed atlas is a texture of its own, because one dispatch can't read and write the same texture. So `ddgi/write` keeps its 4 storage textures, and the shading module still binds one irradiance texture.
- **Trace:** one closest hit per ray, as now. The bounce lookup works out the 8 probes' weights (distance, Chebyshev, crush) once and fetches each layer's irradiance with them. Each extra layer costs only its irradiance fetches, about a quarter of the lookup.
- **Direct light:** the hit's light samples are shared; each goes into its own group's layer. Each layer's estimate stays unbiased; groups with few lights get noisier.
- **Ray results:** 8 bytes per ray per layer, in the ray scratch.
- **Blend:** loops the 36 irradiance texels per layer; the 196 distance texels only once.
  - **One layer at a time in workgroup memory.** The blend sizes its arrays for `DDGI_MAX_RAYS` (512): 8 KB of ray directions and distances, 4 KB of radiance per layer, and about 1.2 KB of texel estimates, 13.5 KB of the 16 KB default today. A second and third radiance layer side by side would need about 21.7 KB. So the directions are loaded once, then each layer's radiance is loaded into the same 4 KB, blended, and replaced by the next layer after a barrier. Workgroup memory stays at 13.5 KB for any number of layers, at the cost of two barriers per extra layer (after its texels are blended, and after the next layer is loaded).
- **`ddgi/compose`:** composed = Σ weight_g × layer_g per texel, borders included. It runs for the tiles updated this frame, and for every tile in a frame where a weight changes. **A group weight change shows in the same frame,** with no lag and no noise.
- **Weights** are technique settings, `lightGroups: { count, weights: Color[], environmentWeight: Color }`: linear RGB multipliers on top of the lights' own intensities. The three adapter exposes `threeGI.setGroupWeight(group, color)`.

**Memory** (an irradiance tile is 8 × 8 texels × 8 bytes = 512 bytes per probe per layer):

| Scene (probes) | Groups off | 2 groups + environment + composed | 4 groups + environment + composed |
| --- | --- | --- | --- |
| Two-room (1,071) | 0.5 MB | 2.2 MB | 3.3 MB |
| Furnished room (4,998) | 2.6 MB | 10.2 MB | 15.4 MB |
| Sponza (19,425) | 9.9 MB | 39.8 MB | 59.7 MB |

Distance (1 KB per probe) doesn't change. Four groups fit the furnished room's 64 MB budget, but they are expensive on Sponza. So **groups are opt-in and off by default,** and the host picks how many (Dennis agreed, 2026-11-05). Time of day (6.1, 6.3) uses the same layers for the sun and the sky; those are on by default and have their own memory.

**Left open on purpose** (Dennis: "I dont want to close the door on common use cases"): in three.js, people dim a group the usual way, by scaling each light's `intensity`. The API keeps that possible later. Group weights are separate from light intensities, and the adapter already sees every intensity change, so a later detector could spot "every light of group g scaled by the same factor this frame" and turn it into a weight change. It's on the Later list, not in M2.

**Cost estimate** at 4 groups:

- trace about +10% (the extra irradiance fetches only);
- blend about +60% (the irradiance texel work × 5);
- compose about 0.1 ms on Sponza when a weight changes;
- in total about +25–35% per probe updated; with 3 layers (lights, sun and sky), about +15–20%; with 2 (lights and sun), about +8–10%.

Shading cost doesn't change, because the renderer samples one composed atlas. *Why not sample every layer in the fragment shader:* irradiance fetches were about 0.21 ms of T1.6's sampling cost (a run at a locked 1200 MHz clock, NOTES), and five layers would push sampling past 1.0 ms.

## 6. Sun and sky with time of day

Dennis (2026-11-05): time of day should be **"instant with a delayed fallback"**, and nothing may take minutes to build: "anything taking minutes to build is a fail and we should not go that direction." So each part below either needs no build, or builds from the field's own rays no slower than the field itself converges (on Sponza, about 1.2 s into the warm-up).

### 6.1 What changes with time of day, and how each part follows

| What changes | How it follows | What it has to build |
| --- | --- | --- |
| The sun's colour and brightness | **instant and exact.** The sun gets a layer of its own (section 5's layers), traced with the sun at a fixed white reference (colour 1, intensity 1). The layer's weight is the sun's actual colour × intensity, so a change is a new weight, with no marking. HomeFig's rule 3 (bake in real colours) is about groups of several differently coloured lights; for one light, white × its colour is the same thing | nothing |
| The sky's colour, brightness and shape | **instant for what each probe sees of the sky directly**, through openings (the first-order sky transfer, 6.3), with the sky's bounces rescaled by its brightness; **re-traced** for what that leaves | the transfer, from the field's own rays: built after a probe's first 8 updates |
| The sun's direction (the patch of sunlight moves, and its bounce with it) | **re-traced**, nearest the camera first (6.4) | nothing |

The last row is the largest. On open ground the sky is only 11–14% of daylight (T1.1's sky-to-sun ratios: Sponza 11.0%, furnished room 13.2%, open ground 14.3%), and in a sunlit interior the sun's bounce is often most of the GI. So how time of day looks on Sponza depends mostly on how fast re-tracing is (6.4), with or without the instant sky. An instant sun direction needs a different idea; it's on the Later list.

### 6.2 Re-baking the sky (needed by every path)

**Today:** a `procedural` environment is built on the CPU (Preetham, calibrated with the Ineichen–Perez ratio; `calibrated-sky.ts`) and uploaded. One workgroup then resamples it to the 256 × 128 radiance map and projects it to 18 SH coefficients. That runs only when `setEnvironment` is called. The projection kernel takes about 1 s to compile (T1.9a), and the CPU build isn't meant to run every frame.

**M2:**

- **A GPU procedural source.** The resample kernel evaluates Preetham directly (a new `SKY_SOURCE_KIND`). The calibration scale is computed on the CPU per sun direction (a few scalar formulas) and passed as a uniform. The CPU code stays: it writes the shared sky files for Cycles and native, and a test holds the GPU map to it (mean within 0.1%, every texel within 1%).
- **The SH projection works in two levels:** many workgroups write partial sums, and one adds them up. Subgroups first, with the existing workgroup-memory fallback. On a procedural sky it integrates over the 256 × 128 map itself. It also writes the sky's **radiance** SH, bands 0–2, which the instant sky needs. Those are the same sums before the cosine weights are applied (`projection-wgsl.ts`), so they cost nothing extra.
- **When it re-bakes:** when the sun moves more than 0.05°, or turbidity, intensity or `sunIlluminance` change; at most once per frame. Estimated GPU cost under 0.1 ms per re-bake; CPU under 0.05 ms.
- **The slow compile:** T2.6a finds why the projection takes about 1 s to compile (perhaps the unrolled 18-term basis inside the loop) and gets it under 0.2 s. That also helps startup (DECISIONS: "find the slow sky kernel").
- **The sun** is a directional light, a separate object (light buffer, shadow rays). Keeping the sun light and the sky's sun in step is the host's job; the viewer's slider moves both. The three adapter already maps pmndrs/sky and `SkyMesh` (three.md, "The sky").

### 6.3 The instant sky: a first-order sky transfer

**The idea** (PRT, Sloan, Kautz, Snyder 2002; Far Cry 3's probes, Gilabert and Stefanov, GDC 2012), kept to the part that's cheap to build:

- Any sky is close to a weighted sum of 9 fixed patterns, the SH basis functions of bands 0–2: `sky(ω) ≈ Σₖ cₖ Yₖ(ω)`, with colour weights `cₖ` (rgb). 6.2's projection computes them.
- What a probe texel sees **directly** of the sky, through openings, is `Σₖ cₖ Tₖ`. A texel's **first-order transfer** `Tₖ` is the pattern `Yₖ`, seen through the probe's visibility, with the same cosine weighting the texel uses for irradiance.
- `Tₖ` has no colour: nothing sits between the probe and the sky. That makes it 9 numbers per texel, not 9 colours, and it needs no bounces, so it doesn't depend on any other probe.
- The sky's **bounces** (sky light off the courtyard floor, the walls) aren't in `Tₖ`. They're handled by a rescale and by re-tracing (below).

**Building it, from the field's own rays.** No extra rays and no new kernel:

- In `ddgi/blend`, each texel thread already loops over the probe's rays with their directions and distances (in workgroup memory). For every ray that **missed** (escaped to the sky), it also adds `Yₖ(direction)` for the 9 patterns, with the same texel weight it uses for irradiance. Rays that hit anything add 0. The 9 polynomial terms are computed inline from the direction, so the transfer adds no workgroup memory (section 5 keeps the layers at 13.5 KB).
- **It must count rays exactly as the irradiance does,** or `Σₖ cₖ Tₖ` won't match the sky part of `S`:
  - it's normalised by the same sum of weights over all blended rays (`blend.ts`), so back-face hits count, with 0;
  - the 32 fixed rays are left out, as they are from irradiance;
  - "missed" uses the trace's own test (`signedDistance` equal to the miss value, `trace.ts`), so a ray that ends at a finite `maxRayDistance` counts as sky in both;
  - the firefly clamp applies to `S` and not to `T`. In steady state that changes nothing; right after a bright change it can make `S` lag `Σₖ cₖ Tₖ` for a few updates.
- The transfer keeps its own count `m` (8 bits in history word 3, beside `owed`) and blends with `h = min(0.97, m / (m + 1))`. It goes on updating with every probe update: that costs only the blend's extra work, and it follows moved geometry by itself. Light changes never shorten `m`. It's reset to 0 when geometry near the probe moves (M3, below), when the probe goes from off to on (state 3, "not updated yet", where `S` restarts too), and when relocation moves the probe by more than 10% of a spacing.
- A probe's transfer counts as **built** once `m ≥ 8` (about 1,800 rays). T2.6b measures its noise at 8 updates against 64 and raises the count if it has to.

**How long building takes.** A transfer is built after a probe's 8th update, so it is never later than the probe's own irradiance.

- **With the warm-up,** every probe gets 16 updates. Today the viewer runs it (up to 16 large passes). T2.4b moves it into the technique, **on by default**, so every host gets it.
  - Sponza's warm-up took about 2.3 s on the RTX (T1.9's cold start: first GI at 10.8 s, settled at 13.1 s). So transfers are built about halfway through it, around **1.2 s**.
  - Two-room's whole warm-up took 0.08 s.
- **With the warm-up turned off, at the default budget,** an unbuilt probe counts as owed one update, at a low urgency (0.05, below any real change) times its camera weight. So the view builds first.
  - The view's ~200 nearest probes need `200 × 8 / (0.75 × 81)` ≈ 26 frames, about **0.4 s** on Sponza.
  - **The whole volume takes about 32 s** on Sponza: 8 sweeps of 4 s at Budget v2's floor. That's the same time the field itself takes to converge without a warm-up, which is why the warm-up is on by default. A host that turns it off accepts half a minute of an unconverged field at the far end of Sponza, with or without the instant sky.

**The instant step is a nudge to the ordinary sky layer.** The probe's sky layer `S` (section 5's environment layer: what the probe stores from the sky) stays the real field. With the instant sky on, emissive surfaces go in the lights layer, so `S` holds only the sky. When the sky's weights change from `c` to `c'`, a pass computes, per texel and channel:

1. **the direct part:** `Δdirect = Σₖ (c'ₖ − cₖ) Tₖ`;
2. **everything else:** `B = max(0, S − Σₖ cₖ Tₖ)`, what `S` holds beyond the 9-pattern sky seen directly. That's mostly the sky's bounces, but it also holds the part of the directly seen sky that bands 0–2 miss: miss rays read the full 256 × 128 radiance map (`giSkyRadiance`), not the SH;
3. **the sky's brightness change:** `ρ = c'₀ / c₀` (band 0, the sky's mean, per channel), clamped to [0.25, 4] for one nudge;
4. **the nudged sky layer:** `S' = max(0, S + Δdirect + B × (ρ − 1))`. The clamp at 0 is there because a truncated SH sum can dip below 0 where the sky is dark.

- *Why rescaling `B` is a fair guess:* bounced sky light is the sky's light blurred over the surfaces around the probe. When the sky reddens, brightens or dims as a whole, its bounce follows nearly in proportion.
- *What the rescale gets wrong, left for re-tracing:*
  - the bright side of the sky moving with the sun. SPEC's HomeFig rule 2 says it: time of day changes the sky's shape, not just its brightness. That's why `S` is still re-traced and the rescale is only the instant guess;
  - the directly seen detail that bands 0–2 miss (near the sun's glow, behind a window), which changes with the sky's shape too. Where it's negative, `B`'s clamp at 0 drops it. On open ground bands 0–2 already missed irradiance by up to 0.87% in M1 (DECISIONS: the platform's sky added band 4); behind windows it's more.
- T2.6b measures the nudge with the direct part alone and with the rescale, and keeps the rescale only if it helps.
- *Why clamp `ρ`:* near night `c₀` is close to 0, and a large ratio would multiply the noise in `B`. Once nudged, a probe's sky weights become `c'`, so a clamped rescale is never made up later. `ρ` is the same for every probe (it's the sky's), but the sky's SH is computed on the GPU (6.2). So `ddgi/mark` reads the old and new SH itself and, when `ρ` was clamped, marks every probe at the full `r` instead of the residual share below. No readback.

**Everything else about the nudge:**

- **`S` stays the real field.** Normal updates keep blending the traced sky into `S` with the probe's history and the firefly clamp, as today. So after the sky stops changing, `S` converges to exactly what it would without the instant sky. The nudge only decides how much of a change shows at once.
- **Bounces read `S`.** A ray hit reads `S` for the sky's bounce, as any layer reads its own (section 5). That is one fetch, as today.
- **The fallback is "no nudge".** A probe whose transfer isn't built (or a volume with the instant sky off) simply isn't nudged and follows by re-tracing (6.4). Nothing switches and nothing jumps.
- **Marking after a sky change:** every probe is marked as for a global change (6.4). A probe with a built transfer gets `r × skyResidualShare` (default 0.3 until T2.6b measures what the nudge leaves), so the budget goes first to probes the nudge couldn't cover.
  - *This matters less than it looks during time of day.* Every sun move re-bakes the sky too, and the sun's direction marks every probe in full. History `n` is per probe and shared by its layers, so the sky layer's history is shortened anyway. The share helps only for sky-only changes (turbidity, a host's own sky).
  - *The shared history has a cost:* a sun mark also makes the next updates of the unchanged lights layer noisier. If T2.4a's two-room step shows that, history becomes per layer (one more f32 per layer per probe).
- **Layout.** Layers aren't more array layers of the atlas: it already uses one array layer per probe row (ddgi.md), and the default limit is 256. So **layers are blocks side by side inside each array layer.** Layer `L`'s tile for probe `(x, z)` sits at `((L × nx + x) × 8, z × 8)`. Light groups (section 5) use the same layout, with no new textures and no new storage bindings. `prepare` checks `nx × 8 × blocks ≤ 8192` and throws `GILimitError` past it. Bistro exterior's camera volume (32 wide, 6 blocks) is 1,536 texels wide.
- **The transfer's 9 numbers per texel** sit in 3 `rgba16float` blocks (12 channels, 3 spare). Transfer values can be negative; halves hold that.
- **The composed atlas** the renderer samples is its own texture. A texture can't be read and written by the same dispatch, and the shading module still binds just one irradiance texture.
- **Writing the nudge** has the same problem (it reads `T` and `S` and writes `S`), so it works like blend and write. `ddgi/sky-nudge` writes the nudged `S` tiles to a nudge scratch buffer (288 bytes per probe). Then a second `ddgi/write` dispatch, in "sky" mode, copies them into the atlas for every probe whose transfer is built (one workgroup per probe; the others exit at once).
- **The order in a frame:** the nudge and its write run **before** `ddgi/trace`. So a probe that is also updated this frame blends from its already-nudged `S`, and the nudge is never lost to the blend writing an older `S`.

**Showing it.** In a frame where the sky changed by more than the re-bake threshold:

1. `ddgi/sky-nudge` handles every probe with a built transfer (before `ddgi/trace`);
2. a `ddgi/write` dispatch in sky mode copies the nudged tiles;
3. `ddgi/compose` writes the composed atlas from the layers:
   `composed = Σ light-group layers × weights + sun layer × sun weight + S × environmentWeight`.

The change shows in the same frame. In other frames compose writes only the tiles that were updated.

**The sun layer at a white reference departs from SPEC's HomeFig rule 3** ("bake each group in its real colors … not in white"). For a single light the result is the same, because each colour channel is linear on its own. Baking in the sun's real colour would break it: probes updated after a colour change would hold the new colour and the rest the old one, so one layer weight couldn't serve both. The one non-linear step, the firefly clamp, needs no change: it limits each update relative to the stored value (`maxBrightnessChange × old`, `blend.ts`), so it clamps the same at any scale. Its only absolute term is a tiny floor (the smallest normal half, about 6e-5), which is negligible. This needs Dennis's OK and a DECISIONS entry (open question 2).

**Memory.** Per probe: the transfer's 3 blocks, the sun layer, the sky layer and the composed atlas (512 bytes each: `rgba16float`, 6 × 6 texels plus border), plus the nudge scratch (288 bytes): 3,360 bytes. The blend's scratch tile grows with the transfer too, by about 1.4 KB per probe in a round (about 1.5 MB at 1,024-probe rounds). That is less than half of the earlier design's, which needed a colour per pattern because it carried bounces.

| Scene (probes) | Extra GPU memory |
| --- | --- |
| Two-room (1,071) | 3.6 MB |
| Furnished room (4,998) | 16.8 MB |
| Bistro exterior's camera volume (12,288, m3-moving.md) | 41 MB |
| Sponza (19,425) | 65 MB |

- **Sized for the target machines** (Dennis: "mid to high end machines are the goal"). The RTX 3060 Laptop has 6 GB of its own, a desktop RTX 3060 12 GB, and the M5 Pro test Mac shares 64 GB with its GPU. Sponza's 65 MB is about 1% of the laptop's 6 GB.
- So the instant sky is **on by default.** A guard turns it off, with a line in the stats, only past `skyTransferMaxMB` (default 512 MB), which no test scene comes near. The earlier 128 MB and 48 MB caps are gone.
- **The furnished room** (everything the core owns): T1.9 measured 40.7 MB. With the instant sky (16.8 MB), the ray scratch for three layers (up to +4.2 MB) and the blend scratch (+1.5 MB), that's about 63 MB. SPEC now keeps its 64 MB as a guide, not a gate (revised 2026-11-05), so T2.6b reports it and nothing turns off.

**What happens when geometry moves (M3).** A moved object changes what the sky can reach: a door closing cuts the dark room off from the window. M3's `ddgi/mark` (m3-moving.md section 5) marks probes near moved geometry. Their transfer count `m` is reset to 0 with their history. Their transfer re-accumulates from the updates they're owed anyway, and is built again after 8 of them. Until then they aren't nudged, and sky changes reach them by re-tracing. `S` itself is updated by the normal blend all along, so nothing jumps. **In a camera-following volume** entering probes get their 2 warming updates first; the other 6 come at the low building urgency. With the camera still, the priority share is free and they come within about 24 s. During a fly-through the entering planes take the priority share, so the other 6 come from the round robin alone (a quarter of 51, about 13 a frame, one pass per 16 s): about 1.5 minutes. In a fast fly-through, most entering probes reach the camera before their transfer is built, and sky changes reach them by re-tracing. Sky changes during a fly-through are rare, so that's accepted.

**Cost** (RTX 3060, estimates, to be measured GPU-bound):

| | GPU cost |
| --- | --- |
| Accumulating the transfer (always, while the instant sky is on) | the blend does about +50–80% more work. On Sponza, back to back, the blend is about 3.5% of a probe update (T1.9, 20 probes: 0.05 of 1.44 ms), so about **+2–3% per probe update**; more on small scenes, where the blend is a larger share |
| Its three lighting layers (lights, sun, sky) | each probe update about +15–20% (section 5's estimate for 3 layers) |
| A frame with a sky change: nudge, write and compose of every tile | Sponza ~0.3 ms. It moves about 4.3 KB per probe, 83 MB in all, so at the laptop's 336 GB/s it can't be under 0.25 ms. Furnished room ~0.08 ms |
| The same, with far probes a quarter per frame | Sponza ~0.1 ms |
| Sky still | the accumulation only |

- **Far probes a quarter per frame:** on volumes above 8,000 probes, probes with a camera weight under 0.25 are nudged a quarter per frame (probe index mod 4). Each quarter keeps the sky weights it was last nudged with, so its nudges still add up exactly, and the far end of the volume shows a sky change within 4 frames. Near probes are nudged every frame.
- **Under Budget v2** the probe count doesn't shrink while frames are on time, so the nudge adds to the update's GPU time in the frames where the sky changes. On Sponza at the floor that's about 1.4 ms + 0.1 ms. At a calibrated count the batch is already near 2 ms, so a sky-change frame relies on Budget v2's +10% margin (0.2 ms), which the far-quarter nudge fits and the every-tile nudge doesn't. T2.6b checks it at the calibrated count.

**Accuracy: measured first** (T2.6b's first check, before the rest is built).

- Right after a 30° move of the sun, with the sun layer's weight at 0 so only the sky counts, how much of the change between the old and the settled field (GI only) does the nudge remove in the same frame? Measured on the furnished room's window bookmark and Sponza's bookmark 1, with the direct part alone and with the rescale.
- **Decision rule, set before measuring:** keep the instant sky if it removes at least 70% on both bookmarks. Otherwise drop it. The sun layer and re-tracing don't depend on it, so nothing else changes.
- If bands 0–2 fall short, try bands 0–4 (band 4 is what fixed the platform's sky irradiance in M1): 25 numbers per texel, 7 blocks, about +40 MB on Sponza. For radiance seen directly, band 3 matters too, so it's 0–4, not 0–2 plus 4.

### 6.4 Re-tracing, nearest the camera first

It's today's field, driven by M2's change-driven updates (section 4). It's what runs:

- for the sun's direction, always;
- for what the sky's nudge leaves, and for probes whose transfer isn't built.

A change of the sun's direction (or of the sky) marks every probe (section 4.2). For a big step, all N probes need K updates, about `N × K / U` frames of the whole budget. Camera-near probes come first: with the camera weight, "near" means within about 4 spacings and in view, roughly the 100–300 probes that dominate the image.

**Far probes owe fewer updates after a global change.** At K = 8 for every probe, the whole of Sponza takes about half a minute. So after a global change:

- probes with a camera weight of 0.5 or more (near and in view) owe 8;
- the rest owe 2.

This applies only to volumes where 8 updates for every probe wouldn't fit in 2 s at the current rate (`N × 8 / U > 120` frames): Sponza and Bistro's camera volume, not the rooms.

- Two updates bring a far probe within 5% (that's what the history rule in 4.1 gives for K = 2), but it's noisier: it holds about 2 updates' rays instead of 8. Its history then grows by itself with the round robin.
- When the camera turns toward it, its camera weight raises its priority as usual.

| Scene | Probes | U (RTX, the rates section) | View within 5% | Whole volume, K = 8 everywhere | Whole volume, far probes owe 2 |
| --- | --- | --- | --- | --- | --- |
| Cornell | 1,728 | ~890 | ~0.1 s | ~0.26 s | not needed |
| Two-room | 1,071 | ~300 | ~0.1 s | ~0.48 s | not needed |
| Furnished room | 4,998 | not measured | not measured | not measured | not measured |
| Bistro exterior's camera volume | 12,288 | 51 (floor) | ~0.7 s | ~32 s | **~8.4 s** |
| Sponza | 19,425 | 81 (floor) | ~0.4 s | ~32 s | **~8.2 s** |

- **Sponza and Bistro use Budget v2's floor,** so their times are the slow end. If the calibration gives more probes, they shrink in proportion. Every time here is re-derived once the M1 lead's GPU-bound numbers land.
- **Far probes owing 2 are noisier** for a while: about 448 rays each instead of about 1,800. Their history grows back with each sweep (every 4 s).
- **These are for a big jump** (dragging the slider across hours). Dennis agreed (2026-11-05) to judge global changes on the view.
- **A slow time-of-day animation** (an hour of sun in 10 s, 1.5° per second) changes each probe only a little between two of its updates, so the step rule in 4.1 wouldn't shorten its history, and with 32 updates of history it would lag far behind. Section 4.1's **ramp rule** handles this: a probe that keeps changing gets a history short enough that its lag stays near 5%.
  - In two-room (a probe updated every ~3.5 frames) the change per update is tiny and the history stays near 32 updates: it trails by about 1.9 s, which costs under 5%.
  - Far from the camera on Sponza, out of view, a probe is reached only by the round robin's quarter of the budget (about 20 probes a frame at the floor), so about every 16 s. The sun moves about 24° in that time, so the history drops below 1 update: those probes jump to each new estimate, about 16 s apart, with more noise. Near the camera the scheduler updates probes far more often.
- **It blends, it never goes black.** Every probe moves from its old value toward its new one through its history (section 4.1). In view, that shows as the old lighting fading into the new. The exception is the one above: during a fast day animation, far, out-of-view Sponza probes have so little history that they step from one estimate to the next.

### 6.5 Re-tracing alone, or with the instant sky: the comparison

On Sponza, on the RTX, at Budget v2's floor (81 probes a frame; the calibrated number may be higher):

| | Re-tracing near the camera, with the sun layer | The same, plus the first-order sky transfer |
| --- | --- | --- |
| Sun colour and brightness | same frame, exact (the sun layer) | same |
| Sun direction | view within 5% in ~0.4 s; whole volume ~8 s | the same |
| Sky change, in view | within 5% in ~0.4 s | most of it in the same frame (T2.6b measures how much); the rest ~0.4 s |
| Sky change, whole volume | ~8 s | most of it within 4 frames; the rest ~8 s, at lower priority |
| Build | none | from the field's rays: ~1.2 s into the warm-up; with the warm-up off, the view in ~0.4 s and the whole volume in ~32 s, as the field itself converges |
| Extra memory | 20 MB (the sun layer and the composed atlas) | 65 MB |
| Extra GPU cost | the sun layer (about +8–10% per probe update) | also +2–3% per probe update, and ~0.1 ms in frames where the sky changed |
| New code | the layers (shared with light groups) | also the blend's accumulation, `ddgi/sky-nudge` and `ddgi/write`'s sky mode |

**Recommendation: both, in that order.**

1. **Re-tracing nearest the camera is the main path,** with far probes owing 2 updates after a global change, and the warm-up on by default. The sun's direction is most of time of day, and only re-tracing moves it. T2.4b and T2.6a build it.
2. **The sun layer goes with it:** exact, instant, and nothing to build.
3. **The first-order sky transfer is added in T2.6b,** on by default, as the instant part of the sky. It meets Dennis's bar: it builds from the field's own rays, never later than the field itself, about 1.2 s into Sponza's warm-up (on by default). Its memory is small for the target machines.
4. **It stays only if its first check passes** (it removes at least 70% of a sky change in the same frame). If it doesn't, it's dropped, and the rest stands unchanged.

This keeps "instant with a delayed fallback" where instant is cheap. It drops the full transfer with bounces from the earlier draft. That transfer needed 64 updates per probe because each probe's bounce read its neighbours' transfers: 17–25 minutes on Sponza at the old controller's rate (still about 9 minutes at the floor), and 125 MB.

*Why not a separate pass of extra visibility rays to build the transfer faster:* it isn't needed. The field's own miss rays give the same transfer for free, and the field needs those updates anyway.

### 6.6 Native

The native harness's miss shader sees one flat colour (DECISIONS 2026-11-04). M2 gives it the texture sky and an animated sun (T2.7), so time-of-day comparisons are like for like. Native has no instant sky; it re-traces every probe every frame with hardware ray tracing.

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
  - Dennis (2026-11-05): **the look is judged by eye.** Our response time is measured and recorded next to native's, but matching native is **not** a requirement: native traces with hardware ray tracing ("we know we wont be anything near native without the hardware access").
- **Stats** gain: probes updated by priority and by round robin, probes owed, shadow rays per frame, `gi/direct` ms, compose ms, sky re-bake ms, light count.

## 8. Later (after M2, not abandoned)

**Area lights** (carried from M1; Dennis: after M2, "lets not abandon them").

- *Why:* so an emitter's direct light can be analytic and DDGI holds only its bounce (DECISIONS 2026-11-05, the viewer entry).
- *Why not in M2:* three's `RectAreaLight` casts no shadows, and SPEC's HomeFig lessons say to avoid area lights without shadows. Until then M1's split stays: DDGI lights the emitter.
- *How they'd fit:*
  - **The light record:** a `rect` type in the same 64 bytes. The position, the normal in the direction word, the intensity, the half-width and half-height in the two cone words, and the in-plane axis, octahedral-packed into the radius word.
  - **Shadow rays** aim at a uniform random point on the rectangle, with a cosine at both ends. `gi/direct` and the light grid treat it like any local light, bounded by its rectangle instead of a point.
  - **No double counting:** the emitter's surface is flagged "light carried by a light", so GI rays that hit it see no emission.
  - **The host** needs shadowed direct light for it (three would need a shadowed area light), or it keeps emissive-only lighting as now.

**Spotting three.js-style group dimming** (section 5): turn "every light of a group scaled by the same factor" into a group weight change.

**An instant sun direction** (6.1): keep every probe's ray hits (surface records) between updates, and on a sun move relight them with new shadow rays instead of re-tracing them. HomeFig relit 512 cached probes for the sun in about 1 ms (RTX 4090, SPEC). It costs the surface records for every probe (24 bytes × 224 rays, about 5 KB per probe, 104 MB on Sponza), and the cached ray directions stop rotating, which needs care.

## How it fits the existing kernels (no parallel GPU paths)

| Kernel | M1 | M2 |
| --- | --- | --- |
| `ddgi/mark` | none | new; only in frames with changes |
| `ddgi/schedule-*` | update list from the CPU cursor | the GPU scheduler writes the same buffer; the CPU cursor is removed, not kept beside it |
| `ddgi/trace` | trace, shade, direct light inline | trace, shade, bounce per layer, surface records; **no `giDirectIrradiance` call** |
| `gi/direct` (platform) | none | new; takes over direct light for every technique that traces |
| light grid build (platform) | none | new; only above `exactLightLimit` local lights |
| `ddgi/blend` | one global hysteresis | per-probe `h` from the history record; loops layers; writes the probe's mean; accumulates the first-order sky transfer from its miss rays |
| `ddgi/compose` | none | whenever there's more than one layer (the sun layer is on by default) |
| `ddgi/sky-nudge` | none | new; only in frames where the sky changed, with the instant sky on |
| `ddgi/write` | one block per probe | loops the layer blocks (the transfer's 3 included); a second dispatch in sky mode writes nudged sky tiles |
| sky resample and projection | CPU-built procedural sky, one-workgroup SH | GPU procedural source, two-level SH, plus radiance SH for the instant sky |

The transfer is built inside `ddgi/blend` from rays the field traces anyway, so there's no transfer kernel and no extra rays.

The instant sky is a nudge to the ordinary sky layer, and its fallback is simply no nudge. So there's no second sky path.

Every new kernel that reduces or compacts is subgroups first, with one declared fallback (per-thread or workgroup atomics), and a test runs both on the same input (the M1 rule).

**Storage buffers per kernel** (the default limit is 8):

| Kernel | Count |
| --- | --- |
| `ddgi/trace` | tracer 3 + scene copy and shading 2 + update list + ray scratch = **7** |
| `gi/direct` | tracer 3 + scene copy and shading 2 + ray scratch + light grid = **7** |
| `ddgi/blend` | update list, ray scratch, scratch tile, census, history = **5** |
| `ddgi/write` | update list + scratch tile = 2; in sky mode, history + nudge scratch = 2 |
| `ddgi/mark` | light-change table, history = 2 (atlases as textures) |
| `ddgi/schedule-*` | history, update list, schedule state = 3 |
| `ddgi/sky-nudge` | history + nudge scratch = 2 (the atlas as a texture; the sky weights as uniforms) |
| `ddgi/compose` | update list = 1 (the atlas as a texture, the composed atlas as 1 storage texture, the weights as uniforms) |

## Cost against the budgets (estimates, RTX 3060, to be measured GPU-bound)

| Item | When | GPU ms |
| --- | --- | --- |
| `ddgi/trace` + `gi/direct` on Sponza | always | per-probe part about 0.6× of M1's `ddgi/trace`; the fixed part unknown (T2.0) |
| Scheduler (4 dispatches) | every frame | < 0.05 |
| `ddgi/mark` | frames with changes | < 0.02 for a few lights; < 0.05 for a global change at 20k probes |
| Light grid build | above `exactLightLimit` local lights | ~0.05 on Sponza with 1,024 lights |
| `ddgi/compose` | groups on | updated tiles only: < 0.01; a weight change on Sponza: ~0.1 |
| Sky re-bake | the sky changed | < 0.1 |
| Instant sky: nudge + write + compose | the sky changed | Sponza ~0.3 for every tile (about 83 MB moved; at least 0.25 ms at the laptop's 336 GB/s), ~0.1 with far probes a quarter per frame; furnished room ~0.08 |
| Instant sky: accumulating the transfer in the blend | while it's on | each probe update ~+2–3% on Sponza |
| Time of day's lighting layers (lights, sun, sky) | on by default | each probe update ~+15–20% (section 5's estimate for 3 layers) |

Under Budget v2 the probe count is set by the calibration at load, which measures the `ddgi/*` passes and `gi/direct` together with whatever is on (layers, the transfer). So the costs that run every frame are paid for in a smaller calibrated batch. Costs that come and go (`ddgi/mark`, a sky change's nudge) don't change the count while frames are on time; they add to that frame's GPU time, and T2.4b and T2.6b check that the update stays within its 2 ms ceiling. Shading cost doesn't change.

Memory: history 32 bytes per probe, ray scratch at most 7.6 MB (11.8 MB with three layers), light grid at most 2 MB, light groups and the instant sky as tabled (Sponza +65 MB for the instant sky, on by default).

## Proposed task breakdown

Board format. IDs are proposals; T1.11 assigns the real ones. Every GPU check says which adapter and uses the bench lock (with Dennis's GPU apps closed for p95 and timings, DECISIONS 2026-11-05).

- **Speed checks** run on the RTX 3060, GPU-bound (the method the M1 lead's re-measure settles), and on the M5 Pro once it's set up.
- **The Intel UHD runs the correctness checks only** (oracle, furnace, accuracy, leaks, the subgroup fallback). Its timings may be recorded, never gated.

| ID | Task | Agent | Size | Depends on |
| --- | --- | --- | --- | --- |
| T2.0 | Spike: deferred shadow-ray pass on Sponza | implementer | S | M1 done |
| T2.1 | Light buffer v2, manifest `motions`, `sponza-candles` | implementer + wrangler | M | none |
| T2.2 | Deferred direct light (`gi/direct`) in platform hit shading | implementer | M | T2.0 |
| T2.3 | Many-light sampling: the light grid | implementer | M | T2.1, T2.2 |
| T2.4a | Per-probe history and `ddgi/mark` | implementer | M | T2.1 |
| T2.4b | GPU scheduler with camera priority; far probes owe 2 after a global change; the warm-up moves into the technique | implementer | M | T2.4a |
| T2.5 | Light groups: layers, compose, weights, adapter API | implementer | M | T2.2, T2.4a |
| T2.6a | Sky re-bake: GPU procedural source, two-level SH with radiance SH, compile fix | implementer | M | T2.4b |
| T2.6b | Instant sky: the first-order transfer in the blend, the nudge, fallback (the sun layer comes with T2.5's layers) | implementer | M | T2.5, T2.6a |
| T2.7 | Native harness: manifest motions, texture sky, animated sun | native-runner | M | T2.1 |
| T2.8 | Viewer: gizmos, sliders, groups, time of day, motions, response meter, native compare page | implementer | M | T2.4b, T2.5, T2.6b, T2.7 |
| T2.10 | M2 measurements against targets | gpu-tester, native-runner | M | all above |
| T2.11 | M2 note and M3 breakdown | main session | S | T2.10 |

T2.9 is left free: area lights are on the Later list (section 8).

The startup items DECISIONS moved to M2 ("fast first GI", the scene copy off the main thread, compiling less) are designed in [m3-moving.md](m3-moving.md), because they share the BVH and scene-copy machinery. Their rows (T2.12–T2.14) are listed there. The target is ≤ 2 s from the hand-off to GI with a cold compile (Dennis, 2026-11-05); file loading is the host's, and is on m3-moving.md's Later list.

### Draft done-whens

- **T2.0 Spike: deferred shadows.** A throwaway branch splits Sponza's `ddgi/trace` into trace + `gi/direct` with surface records.
  - Done when: the fixed and per-probe cost of the pair (from a batch-size curve, 16 to 512 probes, GPU-bound) and the shadow-ray rate against the bench's any-hit rate are recorded on the RTX (3 runs); and the field matches M1's within 1% mean on Cornell and Sponza after 300 frames, on the RTX and the Intel UHD.
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
  - Done when, at the default budget, in GI-only mode, on the RTX:
    1. two-room and Cornell (point light) step changes show the next frame (ROI changes by > 1%) and are within 5% of settled in ≤ 0.5 s;
    2. Sponza's step change at bookmark 1 meets the same;
    3. no probe goes un-updated for more than 2 full round-robin passes during a 60 s sweep (census);
    4. the scheduler costs ≤ 0.05 ms at 20,000 probes, and the subgroup and fallback histograms agree exactly (on the Intel UHD too);
    5. after a 30° sun step on Sponza, the mean error over all probes against a re-converged field is within 5% in ≤ 10 s at the calibrated count (far probes owing 2; the bar is re-set from the GPU-bound rates); the p95 and the far probes' extra noise are recorded;
    6. the warm-up runs in the technique and is on by default: a host that does nothing gets the same first frames as the viewer does today.
  - Record: response curves (CSV), per GPU.
- **T2.5 Light groups.**
  - Done when:
    1. on two-room with 2 groups, a weight change shows in the same frame, and the image matches a field re-converged from scratch at that weight within 1% mean (linearity);
    2. with groups off, the field is within 0.1% mean of T2.4b's and the cost within 3%;
    3. with 4 groups, the cost per updated probe is ≤ 1.4× groups off;
    4. GI sampling at 1440p stays ≤ 1.0 ms (T1.6's view).
  - Record: memory per layer per scene.
- **T2.6a Sky re-bake.**
  - Done when:
    1. the GPU procedural map matches the CPU sky within 0.1% mean and 1% per texel, and `giSkyIrradiance` still meets ±0.5% on open ground;
    2. a re-bake costs ≤ 0.1 ms GPU and ≤ 0.05 ms CPU;
    3. the projection compiles in ≤ 0.2 s cold on the RTX ("cold": a fresh browser profile with nothing cached, one scene per run, as T1.9a measured startup);
    4. by re-tracing alone (instant sky off), a 30° sun step on the furnished room settles the bookmark view (GI only) within 5% in ≤ 0.5 s on the RTX (global changes are judged on the view, Dennis 2026-11-05); the whole-volume time is recorded.
- **T2.6b Instant sky.** The first check runs before the rest is built.
  - Done when, on the RTX:
    1. right after a 30° sun move, with the sun layer's weight at 0 (GI only), the nudge removes at least 70% of the difference between the old and the settled field in the same frame, on the furnished room's window bookmark and Sponza's bookmark 1, with bands 0–2. Recorded with the direct part alone and with the bounce rescale. If it falls short, stop: the instant sky is dropped and the main session tells Dennis;
    2. transfers are built: on Sponza ≤ 1.5 s after the warm-up starts; with the warm-up off, the bookmark view's probes in ≤ 1 s;
    3. on the furnished room's window bookmark, a sky change moves the GI-only mean of the sunlit floor rectangle by more than 1% in the frame it happens;
    4. on two-room and the furnished room, 1,800 frames (at a 16 ms test budget) after a sky change, the field with the instant sky on matches a run with it off within 0.5% mean: the nudge only makes the change sooner, it doesn't move where the field settles;
    5. µs per updated probe (fixed batch of 512, GPU-bound) is within 5% of the same three layers with no transfer; and on Sponza, at the calibrated count, the update plus a sky change's nudge, write and compose stays within Budget v2's "≤ gpuMs (+10%)", back to back (the nudge's own ms recorded, every tile and far quarters);
    6. checks 3 and 4 also pass on the Intel UHD (correctness, not speed).
  - Record: build times per scene, extra memory, the furnished room's total core memory (SPEC's 64 MB is a guide), nudge and compose ms, the transfer's noise at 8 against 64 updates, and bands 0–2 against 0–4.
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
- **T2.10 M2 measurements.** The table at the top on the RTX (GPU-bound) and the M5 Pro if it's set up, the correctness checks on the Intel UHD, plus our and native's response times for the same motions (3 runs; recorded as numbers, not required to match native). Done when each target is met or its miss is explained and agreed with Dennis, and he has judged the sweep's look next to native by eye.

## Risks, and what to spike first

1. **The deferred pass may buy little** at Sponza's operating point, where the fixed cost dominates. Then Sponza stays near Budget v2's floor (81 probes a frame), and the response target is met only near the camera. This is the largest lever and the largest uncertainty. *Spike first: T2.0.*
2. **Short histories flicker.** At `n` = 0.42 the first update is 70% one noisy estimate. If that shows as a flash, raise the floor on `n`, or give marked probes 2× rays for their first update (at a cost). T2.4a's two-room step shows it.
3. **Light-grid noise with many small lights.** Candles have short ranges, and a cell 2 probe spacings wide may hold many lights that matter only at its corners. The options are smaller cells near lights, more reservoirs, or Moreau's light tree. `sponza-candles` decides.
4. **The change estimate is a stand-in.** `lightChangeScale` and the visibility floor are guesses until T2.4a compares marked probes against probes that really changed. Too generous wastes budget; too stingy leaves stale probes until the round robin reaches them.
5. **Global changes on big scenes** can't meet 0.5 s everywhere at 2 ms by re-tracing (section 6.4). Dennis agreed to judge them on the view.
6. **Native motion** needs fork changes, because the harness has no animation. Two-room first, Sponza second.
7. **The instant sky may cover too little.** It moves only what a probe sees of the sky directly, plus a rescaled guess at the bounce. Deep inside Sponza's colonnade, where sky light arrives mostly by bounce, the same-frame share may fall under 70%, and then the transfer is dropped (T2.6b's first check). Re-tracing still carries time of day, so the plan doesn't depend on it.
8. **Re-tracing after a big jump takes seconds far from the camera on Sponza:** about 8 s for the whole volume at Budget v2's floor with far probes owing 2 (6.4), and those far probes are noisier for a while. A higher calibrated count and T2.0's deferred shadow pass are the levers.
9. **The ramp rule's short histories are noisy** far from the camera on Sponza during a day animation (the history drops below 1 update there, 6.4). If that flickers, the floor on `n` rises and those probes lag more.
10. **The rates under Budget v2 aren't measured yet.** This doc uses T1.9a's old-controller medians for the rooms and the floor for Sponza and Bistro. The rooms' T1.9a numbers were probably taken with the clocks partly down, so their calibrated numbers are likely higher and their times shorter; if they come out lower, the times grow in proportion.

## Decided by Dennis (2026-11-05)

1. **Time of day:** "instant with a delayed fallback". Instant where it's cheap: the sun's colour and brightness (the sun layer) and the sky seen directly (6.3). Re-tracing nearest the camera (6.4) carries the sun's direction and the rest; see 7.
2. **`sponza-candles`:** "1024 is fine for now". The count is a setting, for raising later as a stress test.
3. **Light groups:** an explicit group weight, off by default. "Go with stability but also flexibility and expansion later. I dont want to close the door on common use cases": spotting three.js-style dimming stays possible and is on the Later list.
4. **Global changes** (the sun, the sky) are judged on the view.
5. **Native:** the look is judged by eye. Response time is measured as a number, not as a requirement to match native, which has hardware ray tracing.
6. **Area lights:** after M2, "lets not abandon them". On the Later list, with how they'd fit (section 8).
7. **No minutes-long builds:** "anything taking minutes to build is a fail and we should not go that direction." The full sky transfer with bounces (17–25 minutes on Sponza) is dropped; the first-order transfer (6.3) builds from the field's own rays in seconds.
8. **Memory** is sized for mid- to high-end GPUs: "Why are we memory capping so low? this machine has like 6-8gb of GPU memory and 64gb of DDR5 CPU memory. mid to high end machines are the goal." The 128 MB and 48 MB caps are gone; the instant sky is on by default. SPEC now keeps the furnished room's 64 MB as a guide, not a gate (revised 2026-11-05).
9. **Hardware:** the RTX 3060 and up and an Apple M5 Pro are the targets. The Intel UHD is a correctness test only ("we arent building for the intel").
10. **Measurement:** benchmarks and estimates assume a loaded GPU with its clocks up ("Running at 60hz seems like a fail"). Budget v2 is accepted (work per frame, calibrated with the clocks up, a 4 s sweep floor); this doc's rates follow it, and its times get re-derived once the M1 lead's re-measure lands.

## Still open

1. **Far probes owing 2 updates after a global change** (6.4). It brings Sponza's whole volume from about 32 s to about 8 s at Budget v2's floor, but those far probes are noisier (about 448 rays instead of about 1,800) until later sweeps. OK? The alternative is 8 everywhere and about half a minute for the far end of Sponza.
2. **The sun layer at a white reference** (6.3, after "Showing it"). It departs from SPEC's HomeFig rule 3 ("not in white") for a single light, where the result is the same and the real-colour version can't be kept consistent while some probes are updated and others aren't. OK to record as a DECISIONS exception?

Once the M1 lead's GPU-bound rates under Budget v2 land, every time in sections 4.3, 6.3, 6.4 and 6.5 is re-derived; no answer is needed for that.
