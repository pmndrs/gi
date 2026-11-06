# How a probe field updates under change (research note, 2026-11-06)

**Why this note.** Dennis (2026-11-06): "im less concerned with startup at the speed we have at the moment and more about updating. Things like objects moving through the scene or the time of day changing for example. … ive seen partial updates based on position or subdividing probe grids and giving priority to 'key probes' that then feed into sub groups etc."

This note collects what others do. The design that picks an approach is in [m2-lights.md](../docs/design/m2-lights.md) sections 4.1–4.8 (lights, sky, scheduling, key probes) and [m3-moving.md](../docs/design/m3-moving.md) sections 3.1 and 5.1 (moving objects, the BVH). It builds on [ddgi-landscape.md](ddgi-landscape.md) and [probe-density.md](probe-density.md) and doesn't repeat them.

**Our anchors** (estimates use these; no new timing was done for this note):

- Sponza at the default 2 ms (Budget v2, T1.9b): about **268 probe updates a frame** (the median of 3 runs; run 3's fit is 4.6 µs per probe plus 0.76 ms fixed). A full sweep of its 19,425 probes takes about 1.2 s at 60 Hz.
- Native RTXGI on the same RTX 3060 updates every active probe every frame in about **13 ms**.
- T1.12's hysteresis (ddgi.md) already fades each probe's history with the time since its last update (τ = 5 s). With no change signal, Sponza with the sun halved reaches within 5% in 12.5 s and within 1% in 20.5 s.

Sources marked *(read)* were read in full or in the relevant part, by the scout or by us. Those marked *(not verified)* come from a search snippet or from memory.

## 1. What triggers work

| Trigger | What shipped systems do | Source |
| --- | --- | --- |
| **An object moves** | Majercik et al. 2021 grow each dynamic object's box by one grid cell plus the self-shadow bias, and wake the sleeping probes inside it ("Newly Awake"). This is the only published dirty-region rule for DDGI we found. Frostbite's probes put moved probes first. Lumen tracks scene modifications into dirty distance-field bricks. Godot's SDFGI doesn't support dynamic meshes in its GI at all | Majercik 2021 §6.3 *(read)*; Apers 2024 slides 12–14 *(read)*; Wright 2022 *(read)*; Godot docs *(read)* |
| **Lights change** | RTXGI and Majercik re-trace every probe every frame and cut hysteresis when the estimate moves (per texel, −0.15 above a 25% change, to 0 above 80%), or by event: time of day −50% for 10 frames, a flashlight −15% for 4 frames. RTXGI's SDK is asymmetric: darkening fast, brightening damped to a quarter per step. Godot SDFGI updates its light slices in rotation (every 1, 2, 4, 8 or 16 frames) | Majercik 2021 §4.3 *(read)*; `ProbeBlendingCS.hlsl:516-558` *(read)*; Godot `gi.cpp:1289-1300` *(read)* |
| **Time of day** | Unity APV stores sky occlusion without colour and multiplies it by the current ambient probe at run time *(`ProbeVolume.hlsl`, from the planning session's research; not re-read)*. Remedy's Quantum Break split precomputed transport into local, indirect-sun and sky parts so time of day relit at once. RTXGI and Godot re-trace | `ProbeVolume.hlsl:711-735`; Silvennoinen and Timonen 2015 *(read)* |
| **The camera moves** | RTXGI's infinite scrolling volume moves whole cells toward an anchor and clears only the wrapped edge plane; interior probes keep their values. Majercik's camera window "leapfrogs" a plane per spacing moved, tightening the blend by one cell so new planes don't pop. Godot scrolls a cascade a probe at a time and redoes it whole if more than half is dirty | `DDGIVolume.cpp:399-438, 506-517`, `ProbeIndexing.hlsl:295-311` *(read)*; Majercik 2021 §7.3–7.4 *(read)*; Godot `gi.cpp:1201-1250` *(read)* |
| **Nothing reported, but something changed** | Lumen's surface cache takes feedback from ray hits (mip requests into a hash table, read back) and refreshes its oldest pages for animated materials. GIBS's surfels get more rays when they recently contributed and when their variance is high. RTXGI computes per-volume variability but leaves acting on it to the app | Wright 2022 *(read)*; Halén et al. 2021 *(read)*; `native/build/upstream-src/docs/DDGIVolume.md:552-565` *(read)* |

## 2. Scheduling and priority

- **A fixed budget, ordered by priority**, is the common pattern *(read: Wright 2022, Apers 2024, Boissé 2023)*:
  - Lumen's surface cache captures a fixed 512 × 512 texels a frame, sorted by distance and feedback;
  - Frostbite's probes trace about 6,000 probes a frame (a 100,000-ray budget at 16 rays each) out of up to 100,000, one clipmap level a frame, moved probes first and probes that see no nearby geometry last;
  - AMD's GI-1.0 fills one of four screen-probe grids a frame.
- **Turning probes off** is the other big saving. Majercik 2021's state machine (Off, Sleeping, Vigilant, Awake, and "Newly" states for start-up and waking) puts probes with no geometry within one spacing to sleep, for a 30–50% average speed-up (§6.2, §7.1). RTXGI has only Active and Inactive: a probe is off when more than 25% of its fixed rays hit back faces, or when none of them hits a front face within its own cell. Blending skips inactive probes *(read)*.
- **Hiding latency with history:** Frostbite's probes use a per-probe multi-scale mean estimator. T1.12's time-aware count is our version of the same idea.
- **What users expect:** our reading of the systems above is that a change shows within a few frames and settles in about 1–3 s. Epic's Lumen documentation says that "global lighting changes, like disabling the Sun, can take multiple seconds to propagate" *(quoted by the planning session's research; not re-read here)*.
- **Where we stand:** RTXGI updates every probe every frame (13 ms here). We spend 2 ms and must choose which 268 probes. Nobody publishes a DDGI scheduler that orders probes by the size of a reported change; Majercik's event cuts are the nearest, and they act on hysteresis, not on order.

## 3. Hierarchy: coarse probes feeding finer ones

**Nobody we found does Dennis's "key probes" in a DDGI-like field.**

- **Majercik 2021's cascades** are independent nested grids, sampled densest first and blended over the last cell. No probe values pass between levels (§7.3–7.4).
- **Godot's SDFGI cascades** (up to 8, 17³ = 4,913 probes each) are traced independently too; we found no seeding between them (`gi.cpp`, not fully verified). HDDAGI (Linietsky's PR #86267, still open; rebased in #119869, also open) claims local updates without scrolling and 0.93 ms probe tracing against SDFGI's 2.43 ms on a GTX 1650; dynamic objects are a TODO.
- **Frostbite** updates one of its three clipmap levels a frame; again no seeding.
- **Unity APV** stores probes in bricks of 4 × 4 × 4 at several subdivision levels (snippet, not verified), baked, with no run-time tracing.
- **Radiance Cascades** (Sannikov 2023; Freeman, Sannikov, Margel 2025, arXiv 2505.02041) do merge each level into the next coarser one. But they store radiance intervals, with angular resolution growing with distance, not DDGI's irradiance probes, so the merge doesn't carry over directly.
- **Nested volumes** (finer probes inside a coarser volume) are standard in RTXGI and Majercik: they're independent volumes blended by priority. We plan 0.25 m cloth volumes inside Sponza (M3).

So the propagation in m2-lights.md 4.5 (a fine probe takes the visibility-weighted, interpolated **change** of its key neighbours until it's traced itself) is new. That's why its first check (T2.4c) has a decision rule set before measuring.

## 4. Sparse and brick grids

- Unity APV's bricks and Godot's sparse cascades save memory and let very large worlds stream.
- For us memory isn't the limit (Dennis, 2026-11-05: "mid to high end machines are the goal"), and classification already stops probes inside walls from costing a blend.
- Majercik's "sleeping" state (no geometry within a spacing) is the part worth taking. M3 plans it as "air" probes that are woken near moving objects (m3-moving.md 5.3). A brick layout stays on the Later list.

## 5. The BVH side of moving objects

- **What others do:**
  - Hardware ray tracing APIs (DXR, Vulkan) refit or rebuild BLASes by the app's choice. The usual practice is to refit deforming meshes every frame and rebuild them now and then, or when quality drops.
  - Majercik and RTXGI leave this to the engine.
- **Ours (on `m1`):**
  - the top level is refit on the GPU every frame (0.084 ms for 10,000 instances). It's rebuilt in a worker when its relative SAH cost passes 1.25×, measured on the CPU with no readback;
  - mesh trees have a GPU refit (`GpuMeshRefit`) with the same cost signal;
  - H-PLOC builds Sponza's 5.74 M triangles in 46.6 ms on the GPU.
- **The gaps** are in m3-moving.md 3.1:
  - a deforming mesh's tree isn't rebuilt when it drifts (SPEC says keep its shape);
  - the GPU builders aren't wired into `ddgi-core` yet.

## 6. What we take

| From | What |
| --- | --- |
| Majercik 2021 §6.3 | Grow a moved object's box by a cell and wake or mark the probes inside it: our near set (m3 5.1). |
| Majercik 2021 §6.2 | Sleeping probes in empty space, woken near movers: M3's air probes. |
| Majercik 2021 §4.3, RTXGI | Cut history on a change, but **from the scene's report, not the noisy estimate** (T1.5b measured the estimate biasing the field), and **symmetric**. |
| Frostbite probes, Lumen | A fixed budget, ordered: moved and changed first, near and in view first, a round robin so nothing starves. |
| Lumen, GIBS | Feedback: probes whose own update moved more than their noise get revisited sooner. Used only for scheduling, never for hysteresis. |
| RTXGI scrolling | Clear only the wrapped plane; keep the interior. Already in M3's plan, with a fade-in. |
| Unity APV, Remedy | Colourless sky and sun parts weighted at run time: the white sun layer and the first-order sky transfer (m2 6.3). |
| Dennis's key probes | Our own two-level scheme inside one grid, with change propagation (m2 4.5), measured first. |

## Not verified

- Lumen's world radiance cache budget per frame.
- Northlight's GI in Control and Alan Wake 2 (no public source found).
- Unity APV's brick subdivision levels.
- Godot's default cascade count, and whether `frames_to_converge` reaches the shaders.
- Radiance Cascades' exact merge rule (from memory).
- The values in Majercik 2021's Figure 8.
- Activision's, Ubisoft's and Guerrilla's probe systems weren't searched.

## Sources

- Majercik, Mara, McGuire, Mueller. "Scaling Probe-Based Real-Time Dynamic Global Illumination for Production." JCGT 10(2), 2021.
- NVIDIA RTXGI SDK, local copy `native/build/upstream-src/rtxgi-sdk` (`shaders/ddgi/*.hlsl`, `src/ddgi/DDGIVolume.cpp`), with its docs at `native/build/upstream-src/docs/DDGIVolume.md`.
- Wright et al. "Lumen: Real-time Global Illumination in Unreal Engine 5." SIGGRAPH 2022 Advances.
- Apers. GIBS probes, SIGGRAPH 2024 Advances (EA SEED / Frostbite). Halén et al. "Global Illumination Based on Surfels." SIGGRAPH 2021 Advances.
- Boissé et al. "GI-1.0: A Fast and Scalable Two-Level Radiance Caching Scheme for Real-Time Global Illumination." AMD GPUOpen, 2023.
- Godot Engine, `servers/rendering/renderer_rd/environment/gi.cpp` and `gi.h` (master, read 2026-11-06); "Using SDFGI" docs; PRs #86267 and #119869.
- Unity, Adaptive Probe Volumes manual (6.3 LTS) and `ProbeVolume.hlsl`.
- Silvennoinen, Timonen. "Multi-Scale Global Illumination in Quantum Break." SIGGRAPH 2015 Advances.
- Sannikov. "Radiance Cascades." 2023; Freeman, Sannikov, Margel, arXiv 2505.02041, 2025.
