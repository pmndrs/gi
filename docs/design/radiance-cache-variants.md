# Radiance cache: the world-cache variants compared (P6.5)

Status: in progress (2026-11-28, branch `p6-5`). Seven scenes are measured, ten views in all, with RC3.6's modes; the GPU times need one VALID timing run at the end. **No default changes here:** Dennis picks the default from this page (DECISIONS 2026-11-23).

The radiance cache's world cache (the cache the screen probes' rays end in) has grown five alternatives since the GI-1.2 baseline. This page puts them side by side on the same scenes, against the path tracer (Cycles), and says what each is good and bad at.

## The variants

Each is a preset in the viewer's compare panel ("Radiance cache · compare", the **Variant** list), so it can be seen live, and a name the GPU pages take (`VITE_RC_COMPARE=variant=<id>`). The presets are `RADIANCE_CACHE_VARIANT_PRESETS` in `packages/cache/src/compare-toggles.ts`.

| Preset | What it is | Needs from the host | Register |
|---|---|---|---|
| `gi12-rays` | Capsaicin's GI-1.2 hash grid alone; its second bounce from its own rays. Two bounces. | nothing | E10's way back |
| `gi12-ddgi` | **What ships.** The grid with a DDGI volume stacked under it (`setFallback`); DDGI gives the second bounce and, through its own feedback, every bounce after. 256 DDGI probes a frame. | a DDGI volume around the scene | E10 |
| `sharc` | pmndrs/pathtracer's SHaRC-style hash cache instead of the grid, alone. Two bounce rays a sample with the cache as the path's tail: every bounce. | nothing | E1-E7 |
| `sharc-ddgi` | SHaRC with the viewer's DDGI volume still stacked. SHaRC doesn't read the volume, so the picture is SHaRC's and the volume is pure cost. Listed because it is what the viewer gives when only the world cache is switched. | a DDGI volume (unused) | E1, E10 |
| `budgeted` | The grid over DDGI, its cells lit on a budget (20,000 a frame, one light sample each, new and changed cells first) instead of at every hit (id Tech 8). | a DDGI volume | E8, E10 |
| `neural-tail` | The grid alone on its own rays, plus a small network that learns the cells' indirect light and answers the bounces past the second. | `shader-f16` (f32 works, slower), the optional `@pathtracer/mlp` sibling | E11 |
| `gi12-rays-feedback` | RC3.6: the grid alone on its own rays, each second bounce also bringing its cell's indirect light (`cacheFeedback`): every bounce, one more a frame. | nothing | E12 |
| `gi12-rays-temporal` | RC3.6: the grid alone on its own rays; a probe ray whose hit was on last frame's screen reads last frame's lit picture there (`useTemporalFeedback`). | the host's lit image and albedo (the three adapter gives them) | D68 |
| `gi12-rays-feedback-temporal` | Both of RC3.6's feedbacks on the grid alone. | as above | E12, D68 |
| `gi12-ddgi-temporal` | What ships, plus temporal feedback. (Cache feedback does nothing over DDGI, which already carries every bounce.) | a DDGI volume, the lit image | E10, D68 |

## How it was measured

Two GPU pages and a script; all three re-run as they are.

- **`radiance-cache-variants`** (`tools/gpu/pages/radiance-cache-variants/`): per scene with a Cycles reference, at the reference's stored camera and 1920 × 1080, each variant built from nothing the way the viewer builds it (three.js, `ThreeGI`, the composite). It measures, on the irradiance the composite shows (the denoiser's output):
  - **accuracy:** the GI term (albedo × E / π) averaged over 60 frames against Cycles' GI (`cycles-gi.exr`): bias, 8 × 8 block mean error and its 95th percentile (the accuracy page's numbers); the same for one frame alone; the share of visible surfaces the world cache has a cell for;
  - **warm-up:** frames from an empty cache (and DDGI's field from black) until the picture's mean stays within 5% of where it settles;
  - **flicker when still:** per surface pixel, |E_t − E_t−1| / max(E_t−1, 5% of the mean), over 6 pairs of consecutive frames: mean and p95; and the whole frame's brightness change;
  - **the lights double:** every three light × 2 (Sponza's and Bistro exterior's sun; the lamps indoors; the emissive ceiling in the emissive Cornell box): frames until the picture is 50% and 90% of the way to its new level, and the share of the change still missing 4 frames after (lag; "left at 4");
  - **a box moves:** a light grey box (12% of the volume's smallest side) set on the surface below the screen's centre, settled, then moved sideways by 1.5 times its size: the same numbers over the 8 × 8 blocks the move changed. The share still missing a few frames after is the ghost of the old box (its shadow and occlusion left behind);
  - **memory:** the radiance cache's GPU bytes and DDGI's.
- **`radiance-cache-variants-timing`**: GPU ms per frame of each variant at 1280 × 720 (the radiance cache's passes, the world cache's own, DDGI's, the neural tail's), 120 timed frames after 240, through the platform's GPU timer. Recorded, not gated (DECISIONS 2026-11-06/09). Counts only with the bench lock's VALID verdict.
- **`tools/reference/rc-variants-table.ts`**: the tables below from the pages' `--json` reports; the timing table labelled with the run's clock verdict.

```sh
node tools/reference/bench-lock.ts run --no-cool-wait --what "p6.5 variants" -- pnpm test:gpu --filter radiance-cache-variants --gpu nvidia --json bench/results/rc-variants.json
node tools/reference/bench-lock.ts run --what "p6.5 variants timing" --clock-file bench/results/rc-variants-timing.clock.json -- pnpm test:gpu --filter radiance-cache-variants-timing --gpu nvidia --json bench/results/rc-variants-timing.json
node tools/reference/rc-variants-table.ts --variants bench/results/rc-variants.json --timing bench/results/rc-variants-timing.json --clock bench/results/rc-variants-timing.clock.json
```

`VITE_RC_VARIANTS` and `VITE_RC_VARIANT_SCENES` pick a subset, and `VITE_RC_VARIANT_VIEWS` names views (by bookmark, or `<scene>/<bookmark>`; unset, each scene's first view with a reference). A scene without a reference is skipped and listed. The reports behind the tables here are in `bench/results/incoming/` (git-ignored): `rc-variants-cornell-tworoom`, `-batch2`, `-cornell-emissive` (it replaces batch2's emissive Cornell, run before the ceiling was doubled), `-bistro-interior`, `-batch3`, `-across` and `-rc36` (`.json`; RC3.6's presets on every view, after merging `rc3-6` at ddd4425, whose settings are off by default, so the earlier rows are unchanged by it).

## Results

Measured 2026-11-27 on the RTX 3060 Laptop (Chrome, bench lock). The accuracy, warm-up, flicker, lag and memory numbers don't depend on the GPU clock. Read the columns as follows:

- **GI bias, 8×8 mean, 8×8 p95**: the GI term averaged over 60 frames against Cycles. Bias is the signed error of the whole picture; the 8×8 numbers are the error of 8 × 8 pixel blocks (their mean, and the worst 5%). **One frame mean** is the same for a single frame (noise included).
- **Cache coverage**: the share of visible surfaces the world cache holds a cell for. It is not the cache's quality (probe rays land where they land), but a low share means fewer cells near the camera. With DDGI under the grid, second-bounce hits don't insert cells, so it reads lower.
- **Warm-up**: frames from an empty cache, with DDGI's field starting black, until the picture's mean stays within 5% of where it settles.
- **Flicker**: what the picture does when nothing moves. Per pixel is the mean change from one frame to the next, with the 95th percentile in brackets. "Of the frame" is the whole picture's brightness change, which shows pumping.
- **Lights ×2** and **box moves**: frames until the picture is 50% and 90% of the way to its new state, and the share of the change still missing 4 frames after (100% means nothing has moved yet). For the box, the region is the light around it: its old shadow lit again (the ghost) and its new shadow, with the box's own pixels left out. At 60 frames a second, 60 frames is one second.
- **Memory**: the radiance cache's GPU memory at 1920 × 1080, with DDGI's in brackets.

### Summary

Each view's GI bias and 8×8 mean error, then the median over the ten views for the dynamic numbers. Box moves too small to read are left out (the furnished room's first view). Memory is the radiance cache's at 1080p, DDGI's not included.

| Variant | Cornell point | Cornell emissive | two-room | furnished: corner, window | Sponza | Bistro int: dining, bar | Bistro ext: street, across | mean of the bias's size | lights ×2: 90% (frames) | box moves: 90% | warm-up | memory MiB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gi12-rays | −25.1 / 25.8 | −49.1 / 49.5 | −43.8 / 43.8 | −62.0 / 62.0, −61.3 / 61.3 | −6.6 / 13.4 | −4.8 / 15.5, −6.5 / 16.1 | −6.9 / 15.2, −8.5 / 15.0 | 27.5% | 68 | 79 | 22 | 1,419 |
| **gi12-ddgi** | −0.1 / 6.0 | −1.8 / 10.2 | −0.9 / 1.5 | −8.1 / 8.2, −7.7 / 7.9 | −2.5 / 12.6 | −0.6 / 15.3, −3.5 / 15.7 | −3.1 / 14.3, −3.1 / 13.8 | **3.1%** | 77 | 74 | 49 | 1,373 (+12-72 DDGI) |
| sharc | −3.2 / 7.0 | −3.8 / 10.8 | −4.4 / 4.4 | −9.7 / 9.8, −9.5 / 9.6 | −6.1 / 13.2 | −4.2 / 14.2, −5.3 / 14.9 | −5.6 / 14.3, −6.9 / 14.0 | 5.9% | 83 | 85 | 45 | **415** |
| sharc-ddgi | as sharc | | | | | | | 5.9% | 83 | 86 | 45 | 415 (+DDGI) |
| budgeted | −0.4 / 6.1 | −1.8 / 10.2 | −0.8 / 1.5 | −8.1 / 8.2, −7.7 / 7.9 | −2.6 / 12.7 | −1.7 / 15.5, −4.5 / 15.9 | −3.1 / 14.3, −3.4 / 13.8 | 3.4% | 86 | 75 | 54 | 1,379 (+DDGI) |
| neural-tail | −3.7 / 7.6 | −49.1 / 49.5 | −7.0 / 7.1 | −13.7 / 13.7, −14.9 / 14.9 | −5.9 / 13.2 | −2.8 / 15.2, −4.3 / 15.6 | −4.9 / 14.8, −6.8 / 14.3 | 11.3% | 84 | 82 | 31 | 1,453 |
| gi12-rays-feedback | −5.8 / 9.1 | −49.1 / 49.5 | −19.7 / 19.7 | −34.1 / 34.1, −39.7 / 39.7 | −6.2 / 13.3 | −3.2 / 15.3, −5.1 / 15.7 | −5.9 / 15.0, −7.6 / 14.5 | 17.6% | 82 | 80 | 24 | 1,419 |
| gi12-rays-temporal | −12.8 / 15.0 | +3.3 / 20.0 | −32.7 / 32.7 | −50.1 / 50.1, −52.8 / 52.8 | +3.5 / 12.6 | +21.4 / 26.7, +13.9 / 24.5 | +0.5 / 15.1, +1.1 / 17.2 | 19.2% | 82 | 84 | 51 | 1,419 |
| gi12-rays-feedback-temporal | −6.0 / 9.4 | +3.3 / 20.0 | −18.9 / 18.9 | −33.6 / 33.6, −39.3 / 39.3 | +3.8 / 12.7 | +22.0 / 27.1, +14.8 / 24.7 | +1.1 / 15.2, +1.5 / 17.0 | 14.4% | 86 | 81 | 61 | 1,419 |
| gi12-ddgi-temporal | −2.9 / 7.2 | +26.0 / 27.9 | −2.3 / 2.5 | −9.3 / 9.4, −8.7 / 8.9 | +6.0 / 13.2 | +22.8 / 27.5, +15.6 / 25.1 | +3.4 / 14.8, +4.2 / 16.9 | 10.1% | 84 | 78 | 50 | 1,373 (+DDGI) |

Things the table shows:

- **Only the closed rooms separate the variants.** In Sponza and both Bistros every variant is within −0.6 to −6.9% bias and 13-16% block error. In the four closed rooms the GI-1.2 grid alone is 25-62% dark. Every variant carrying more bounces is within 10%, except the neural tail (within 15%, and −49% in the emissive Cornell box).
- **The grid over DDGI has the smallest bias in every view**, with budgeted cells (the same grid) next. SHaRC is 2-5 points behind in the rooms and 2-3 in the open scenes. In the open scenes SHaRC's block error ties or beats the grid over DDGI's, and its p95 is lower (Sponza 40% against 52%, Bistro exterior 38% against 47%): DDGI's coarse field blurs the bounce at contacts.
- **The neural tail can't light an emissive scene.** In the emissive Cornell box it reads exactly as the grid alone. The cache's own lights don't include emitters (register D21), so the cells it learns from hold no light. In the furnished room it is 6-7 points behind DDGI.
- **SHaRC under DDGI is SHaRC.** It doesn't read the volume, so the numbers match to the noise and the volume is pure cost.
- **RC3.6's cache feedback closes only part of the closed rooms' gap.** On the grid alone: two-room −44% → −20%, the furnished room −62% → −34 to −40%, Cornell point −25% → −6%. It is still well behind DDGI and SHaRC, and it does nothing for the emissive Cornell box (−49%), whose cells hold no emitter light.
- **Temporal feedback over-brightens where the lit picture shows emitters or lamps.** Bistro interior reads +14 to +23%, and the emissive Cornell box +3% on the grid alone and +26% over DDGI. In the closed rooms lit by lamps out of view it barely helps (two-room −33%, the furnished room −50%). Over DDGI it is worse than DDGI alone on every view. It also pumps on Bistro exterior's `across-the-street`, where the frame brightness changes 0.05-0.10% a frame against 0.003% without it, and the lights' 90% takes 105-107 frames.
- **RC3.6's own numbers agree.** Its two-room run read −22.2% (cache feedback), −32.2% (temporal), −21.0% (both) and −4.0% (DDGI plus temporal). It read +24% for DDGI plus temporal on the emissive Cornell box. This page reads −19.7%, −32.7%, −18.9%, −2.3% and +26.0% on the denoised picture after 240 warm-up frames.
- **The earlier Sponza excess is gone.** On the old references (RC4.6) the grid over DDGI read +14.5% / +20%. Against the new references it reads −2.5%. The new references and what merged since (RC3.4's sky light, RC4.2) both changed; which one moved it was not looked into.
- Sponza, the furnished room and the Bistros warn that a node-graph material reads as grey 0.5 in GI. That is the same for every variant, so it doesn't change the comparison.

### Against Cycles, by view

#### cornell-box-point (`front`)

| Variant | GI bias | 8×8 mean | 8×8 p95 | one frame mean | cache coverage | warm-up (frames) | flicker per pixel (p95) | flicker of the frame | lights ×2: 50% / 90% frames · left at 4 | box moves: 50% / 90% frames · left at 4 | memory MiB (DDGI) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gi12-rays-feedback | -5.8% | 9.1% | 19% | 8.9% | 52% | 88 | 0.08% (0.3%) | 0.007% | 25 / 79 · 91% | 37 / 91 · 101% | 1419 (0) |
| gi12-rays-temporal | -12.8% | 15.0% | 25% | 14.9% | 50% | 66 | 0.08% (0.2%) | 0.002% | 31 / 82 · 98% | 30 / 90 · 100% | 1419 (0) |
| gi12-rays-feedback-temporal | -6.0% | 9.4% | 18% | 9.3% | 50% | 85 | 0.08% (0.2%) | 0.004% | 32 / 85 · 98% | 32 / 80 · 101% | 1419 (0) |
| gi12-ddgi-temporal | -2.9% | 7.2% | 20% | 7.1% | 50% | 51 | 0.08% (0.2%) | 0.002% | 31 / 82 · 98% | 29 / 78 · 100% | 1389 (16) |

Lights: x2 PointLight. Box: a 0.67 m box moved 1.01 m (its own 203 blocks left out).

#### cornell-box (`front`)

| Variant | GI bias | 8×8 mean | 8×8 p95 | one frame mean | cache coverage | warm-up (frames) | flicker per pixel (p95) | flicker of the frame | lights ×2: 50% / 90% frames · left at 4 | box moves: 50% / 90% frames · left at 4 | memory MiB (DDGI) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gi12-rays-feedback | -49.1% | 49.5% | 96% | 49.5% | 50% | 1 | 0.14% (0.6%) | 0.002% | 22 / 64 · 99% | 24 / 78 · 100% | 1419 (0) |
| gi12-rays-temporal | +3.3% | 20.0% | 99% | 20.0% | 52% | 82 | 0.09% (0.3%) | 0.002% | 28 / 80 · 99% | 23 / 75 · 99% | 1419 (0) |
| gi12-rays-feedback-temporal | +3.3% | 20.0% | 99% | 20.0% | 50% | 82 | 0.09% (0.3%) | 0.002% | 28 / 80 · 99% | 24 / 76 · 100% | 1419 (0) |
| gi12-ddgi-temporal | +26.0% | 27.9% | 119% | 28.1% | 50% | 79 | 0.08% (0.2%) | 0.002% | 30 / 85 · 99% | 26 / 75 · 99% | 1389 (16) |

Lights: x2 1 emissive material(s). Box: a 0.67 m box moved 1.01 m (its own 203 blocks left out).

#### two-room (`lit-room`)

| Variant | GI bias | 8×8 mean | 8×8 p95 | one frame mean | cache coverage | warm-up (frames) | flicker per pixel (p95) | flicker of the frame | lights ×2: 50% / 90% frames · left at 4 | box moves: 50% / 90% frames · left at 4 | memory MiB (DDGI) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gi12-rays-feedback | -19.7% | 19.7% | 23% | 19.4% | 100% | 107 | 0.04% (0.1%) | 0.008% | 29 / 83 · 92% | 25 / 80 · 96% | 1419 (0) |
| gi12-rays-temporal | -32.7% | 32.7% | 41% | 32.6% | 100% | 63 | 0.04% (0.1%) | 0.001% | 29 / 79 · 94% | 23 / 76 · 97% | 1419 (0) |
| gi12-rays-feedback-temporal | -18.9% | 18.9% | 23% | 18.7% | 100% | 108 | 0.04% (0.1%) | 0.007% | 32 / 87 · 94% | 23 / 81 · 97% | 1419 (0) |
| gi12-ddgi-temporal | -2.3% | 2.5% | 6% | 2.4% | 60% | 72 | 0.03% (0.1%) | 0.002% | 29 / 77 · 95% | 20 / 70 · 97% | 1385 (12) |

Lights: x2 PointLight. Box: a 0.36 m box moved 0.54 m (its own 281 blocks left out).

#### furnished-room (`from-the-corner`)

| Variant | GI bias | 8×8 mean | 8×8 p95 | one frame mean | cache coverage | warm-up (frames) | flicker per pixel (p95) | flicker of the frame | lights ×2: 50% / 90% frames · left at 4 | box moves: 50% / 90% frames · left at 4 | memory MiB (DDGI) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gi12-rays-feedback | -34.1% | 34.1% | 40% | 33.7% | 100% | 145 | 0.04% (0.1%) | 0.016% | 36 / 93 · 94% | 30 / 100 · 95% | 1419 (0) |
| gi12-rays-temporal | -50.1% | 50.1% | 58% | 50.1% | 100% | 81 | 0.04% (0.1%) | 0.002% | 34 / 87 · 97% | 30 / 98 · 95% | 1419 (0) |
| gi12-rays-feedback-temporal | -33.6% | 33.6% | 40% | 33.2% | 100% | 144 | 0.04% (0.1%) | 0.016% | 39 / 96 · 97% | 26 / 96 · 94% | 1419 (0) |
| gi12-ddgi-temporal | -9.3% | 9.4% | 12% | 9.2% | 70% | 107 | 0.03% (0.1%) | 0.008% | 39 / 92 · 98% | (1 blocks) | 1402 (29) |

Lights: x2 DirectionalLight, PointLight. Box: a 0.41 m box moved 0.61 m (its own 304 blocks left out).

#### furnished-room (`towards-the-window`)

| Variant | GI bias | 8×8 mean | 8×8 p95 | one frame mean | cache coverage | warm-up (frames) | flicker per pixel (p95) | flicker of the frame | lights ×2: 50% / 90% frames · left at 4 | box moves: 50% / 90% frames · left at 4 | memory MiB (DDGI) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gi12-rays-feedback | -39.7% | 39.7% | 46% | 39.4% | 100% | 124 | 0.05% (0.1%) | 0.011% | 33 / 89 · 94% | 16 / 55 · 96% | 1419 (0) |
| gi12-rays-temporal | -52.8% | 52.8% | 61% | 52.8% | 100% | 65 | 0.04% (0.1%) | 0.002% | 32 / 84 · 96% | 16 / 52 · 97% | 1419 (0) |
| gi12-rays-feedback-temporal | -39.3% | 39.3% | 46% | 39.0% | 100% | 124 | 0.04% (0.1%) | 0.010% | 36 / 93 · 96% | 15 / 50 · 96% | 1419 (0) |
| gi12-ddgi-temporal | -8.7% | 8.9% | 13% | 8.7% | 42% | 101 | 0.04% (0.1%) | 0.007% | 37 / 90 · 97% | 16 / 56 · 96% | 1402 (29) |

Lights: x2 DirectionalLight, PointLight. Box: a 0.41 m box moved 0.61 m (its own 501 blocks left out).

#### sponza (`colonnade-from-the-floor`)

| Variant | GI bias | 8×8 mean | 8×8 p95 | one frame mean | cache coverage | warm-up (frames) | flicker per pixel (p95) | flicker of the frame | lights ×2: 50% / 90% frames · left at 4 | box moves: 50% / 90% frames · left at 4 | memory MiB (DDGI) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gi12-rays-feedback | -6.2% | 13.3% | 41% | 13.3% | 92% | 1 | 0.23% (0.8%) | 0.002% | 19 / 85 · 78% | 23 / 80 · 98% | 1419 (0) |
| gi12-rays-temporal | +3.5% | 12.6% | 55% | 12.7% | 90% | 17 | 0.23% (0.8%) | 0.003% | 24 / 88 · 87% | 22 / 78 · 98% | 1419 (0) |
| gi12-rays-feedback-temporal | +3.8% | 12.7% | 56% | 12.8% | 90% | 18 | 0.23% (0.8%) | 0.002% | 24 / 88 · 87% | 22 / 78 · 98% | 1419 (0) |
| gi12-ddgi-temporal | +6.0% | 13.2% | 62% | 13.3% | 49% | 31 | 0.22% (0.8%) | 0.001% | 24 / 87 · 87% | 22 / 78 · 98% | 1428 (55) |

Lights: x2 DirectionalLight. Box: a 2.40 m box moved 3.60 m (its own 959 blocks left out).

#### bistro-interior (`dining-room`)

| Variant | GI bias | 8×8 mean | 8×8 p95 | one frame mean | cache coverage | warm-up (frames) | flicker per pixel (p95) | flicker of the frame | lights ×2: 50% / 90% frames · left at 4 | box moves: 50% / 90% frames · left at 4 | memory MiB (DDGI) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gi12-rays-feedback | -3.2% | 15.3% | 36% | 15.4% | 100% | 28 | 0.15% (0.5%) | 0.002% | 17 / 68 · 84% | 19 / 80 · 97% | 1419 (0) |
| gi12-rays-temporal | +21.4% | 26.7% | 72% | 26.7% | 100% | 38 | 0.15% (0.5%) | 0.002% | 24 / 73 · 94% | 19 / 104 · 95% | 1419 (0) |
| gi12-rays-feedback-temporal | +22.0% | 27.1% | 73% | 27.1% | 100% | 39 | 0.15% (0.5%) | 0.003% | 24 / 73 · 94% | 19 / 102 · 95% | 1419 (0) |
| gi12-ddgi-temporal | +22.8% | 27.5% | 74% | 27.6% | 91% | 44 | 0.15% (0.5%) | 0.003% | 24 / 74 · 94% | 19 / 102 · 95% | 1428 (55) |

Lights: x2 PointLight, PointLight, PointLight. Box: a 0.89 m box moved 1.33 m (its own 751 blocks left out).

#### bistro-interior (`bar`)

| Variant | GI bias | 8×8 mean | 8×8 p95 | one frame mean | cache coverage | warm-up (frames) | flicker per pixel (p95) | flicker of the frame | lights ×2: 50% / 90% frames · left at 4 | box moves: 50% / 90% frames · left at 4 | memory MiB (DDGI) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gi12-rays-feedback | -5.1% | 15.7% | 45% | 15.7% | 100% | 20 | 0.15% (0.5%) | 0.003% | 19 / 76 · 84% | 27 / 107 · 95% | 1419 (0) |
| gi12-rays-temporal | +13.9% | 24.5% | 68% | 24.5% | 100% | 30 | 0.15% (0.5%) | 0.002% | 24 / 78 · 92% | 27 / 103 · 95% | 1419 (0) |
| gi12-rays-feedback-temporal | +14.8% | 24.7% | 68% | 24.7% | 100% | 32 | 0.15% (0.5%) | 0.003% | 24 / 78 · 93% | 27 / 103 · 96% | 1419 (0) |
| gi12-ddgi-temporal | +15.6% | 25.1% | 69% | 25.1% | 81% | 38 | 0.15% (0.5%) | 0.003% | 25 / 79 · 93% | 26 / 104 · 95% | 1428 (55) |

Lights: x2 PointLight, PointLight, PointLight. Box: a 0.89 m box moved 1.33 m (its own 375 blocks left out).

#### bistro-exterior (`street`)

| Variant | GI bias | 8×8 mean | 8×8 p95 | one frame mean | cache coverage | warm-up (frames) | flicker per pixel (p95) | flicker of the frame | lights ×2: 50% / 90% frames · left at 4 | box moves: 50% / 90% frames · left at 4 | memory MiB (DDGI) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gi12-rays-feedback | -5.9% | 15.0% | 41% | 15.0% | 100% | 7 | 0.16% (0.6%) | 0.002% | 22 / 81 · 84% | 25 / 69 · 99% | 1419 (0) |
| gi12-rays-temporal | +0.5% | 15.1% | 57% | 15.1% | 100% | 18 | 0.16% (0.6%) | 0.001% | 24 / 82 · 89% | 25 / 70 · 99% | 1419 (0) |
| gi12-rays-feedback-temporal | +1.1% | 15.2% | 58% | 15.2% | 100% | 19 | 0.16% (0.6%) | 0.001% | 24 / 82 · 89% | 25 / 69 · 99% | 1419 (0) |
| gi12-ddgi-temporal | +3.4% | 14.8% | 63% | 14.9% | 94% | 48 | 0.16% (0.6%) | 0.002% | 23 / 79 · 89% | 25 / 69 · 99% | 1445 (72) |

Lights: x2 DirectionalLight. Box: a 6.24 m box moved 9.36 m (its own 10872 blocks left out).

#### bistro-exterior (`across-the-street`)

| Variant | GI bias | 8×8 mean | 8×8 p95 | one frame mean | cache coverage | warm-up (frames) | flicker per pixel (p95) | flicker of the frame | lights ×2: 50% / 90% frames · left at 4 | box moves: 50% / 90% frames · left at 4 | memory MiB (DDGI) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gi12-rays-feedback | -7.6% | 14.5% | 31% | 14.6% | 100% | 1 | 0.20% (0.7%) | 0.003% | 20 / 84 · 83% | 23 / 64 · 99% | 1419 (0) |
| gi12-rays-temporal | +1.1% | 17.2% | 39% | 17.8% | 100% | 9 | 0.26% (0.8%) | 0.097% | 34 / 107 · 86% | 23 / 108 · 96% | 1419 (0) |
| gi12-rays-feedback-temporal | +1.5% | 17.0% | 42% | 17.2% | 99% | 28 | 0.24% (0.8%) | 0.052% | 36 / 106 · 90% | 23 / 107 · 97% | 1419 (0) |
| gi12-ddgi-temporal | +4.2% | 16.9% | 46% | 17.2% | 74% | 32 | 0.23% (0.7%) | 0.064% | 25 / 105 · 88% | 23 / 114 · 96% | 1445 (72) |

Lights: x2 DirectionalLight. Box: a 6.24 m box moved 9.36 m (its own 9803 blocks left out).

**Not measured:** Sponza's `upper-gallery` has no reference yet (not rendered). When it lands, measure it with `VITE_RC_VARIANT_VIEWS=upper-gallery`.

### Where the lag comes from

On two-room with the GI denoiser off (`VITE_RC_COMPARE=denoiser=false`), the lights' 90% came at 50 frames (grid over DDGI) and 66 (SHaRC), against 58 and 73 with it on. The box's ghost reached 90% at 40 and 59 frames, against 67 and 84. Per-pixel flicker rose from 0.03-0.04% to 1.6-1.7%. So the denoiser is what makes every variant's still picture this calm, and it adds about 25 frames to the box's ghost. Most of the light's lag happens before it, in the screen probes and the world cache, and it is about the same for every variant.

### GPU time

**Still not VALID.** The run that was meant to be the VALID one ran on 2026-11-28 under the bench lock with the cool wait, covering all ten presets on all seven scenes. It came out INVALID: thermal throttling in 99.5% of 7,710 loaded samples. The cool wait started it at 69 C after 5 s, and 27 minutes of timed GPU load (70 runs of 240 warm-up and 120 timed frames each) holds the RTX 3060 Laptop at its thermal limit. The SM clock under load had a median of 1,282 MHz. The clock rule asks for a timed section of 10-15 s. A VALID run therefore needs splitting into short runs, each with its own cool wait: one scene and two or three presets a run (`VITE_RC_VARIANT_SCENES`, `VITE_RC_VARIANTS`), and perhaps fewer warm-up frames. That split hasn't been done.

**Split runs, 2026-11-28 (partly VALID).** `tools/reference/rc-variants-timing-split.ts` splits the timing into 46 short runs, each with its own cool wait (to 50 C), 40 warm-up frames, a 30 s idle pause and 30 timed frames. Five runs finished before the overnight queue was stopped (Dennis, 2026-11-28: the machine was too busy to wait longer), all on Bistro exterior and all VALID:

| Variant | Bistro exterior, GPU ms per frame at 1280×720 (world cache; DDGI) |
|---|---|
| budgeted | 20.78 (6.11; 2.70) |
| gi12-ddgi | 21.81 (7.09; 2.76) |
| sharc | 26.21 (14.24; 0.00) |
| sharc-ddgi | 28.72 (14.06; 2.73) |
| gi12-rays | 29.28 (17.26; 0.00) |

The neural tail and RC3.6's presets weren't reached. The VALID ordering is the INVALID run's: budgeted cells cheapest, then the grid over DDGI, then SHaRC, with the grid alone dearest. DDGI stacked under SHaRC is pure cost (+2.5 ms), as its picture is SHaRC's own.

The numbers below are the INVALID full run's, for comparing the variants with each other only. They tell the same story as the earlier dry run (82 C start, without RC3.6's presets). On the big scenes, budgeted cells are cheapest, then the grid over DDGI, then SHaRC, and the grid alone and the neural tail are dearest. RC3.6's presets cost about what the grid alone does. Throttling moves single cells by up to a quarter between the two runs: Sponza's grid alone read 35 ms in one and 38 ms in the other, the neural tail 38 and 32. Don't read small differences.

**GPU ms per frame at 1280×720: radiance cache + DDGI (world cache; DDGI). Timing INVALID: thermal throttling in 99.5% of loaded samples.**

| Variant | cornell-box-point | cornell-box | two-room | furnished-room | sponza | bistro-interior | bistro-exterior |
|---|---|---|---|---|---|---|---|
| gi12-rays | 4.04 (2.64; 0.00) | 4.16 (1.99; 0.00) | 10.59 (6.90; 0.00) | 13.16 (7.87; 0.00) | 38.13 (23.50; 0.00) | 37.69 (23.21; 0.00) | 38.55 (22.45; 0.00) |
| gi12-ddgi | 5.24 (2.32; 0.84) | 4.95 (1.82; 0.88) | 9.77 (4.75; 1.31) | 12.39 (4.65; 2.59) | 22.48 (8.49; 2.52) | 29.22 (11.17; 3.64) | 28.83 (9.08; 3.54) |
| sharc | 3.55 (1.30; 0.00) | 3.47 (1.21; 0.00) | 6.31 (2.51; 0.00) | 10.43 (5.20; 0.00) | 25.11 (13.66; 0.00) | 29.44 (15.10; 0.00) | 35.68 (19.28; 0.00) |
| sharc-ddgi | 4.48 (1.29; 0.93) | 4.34 (1.20; 0.88) | 7.47 (2.46; 1.31) | 12.07 (4.91; 2.46) | 26.26 (13.03; 2.44) | 32.80 (15.01; 3.60) | 39.21 (19.14; 3.78) |
| budgeted | 5.88 (2.69; 0.94) | 5.29 (2.46; 0.79) | 10.12 (5.11; 1.31) | 11.99 (4.98; 2.36) | 19.89 (6.90; 2.35) | 24.90 (7.03; 3.61) | 27.08 (7.52; 3.48) |
| neural-tail | 6.82 (2.94; 0.00; NRC 2.05) | 6.18 (1.94; 0.00; NRC 2.23) | 13.89 (6.94; 0.00; NRC 3.25) | 16.71 (7.78; 0.00; NRC 3.78) | 32.48 (18.18; 0.00; NRC 3.77) | 41.27 (23.09; 0.00; NRC 3.93) | 44.74 (23.11; 0.00; NRC 5.37) |
| gi12-rays-feedback | 5.29 (3.04; 0.00) | 4.18 (2.01; 0.00) | 10.64 (6.93; 0.00) | 12.87 (7.79; 0.00) | 29.24 (18.48; 0.00) | 37.59 (23.18; 0.00) | 39.54 (23.20; 0.00) |
| gi12-rays-temporal | 5.12 (2.87; 0.00) | 4.34 (2.09; 0.00) | 10.81 (6.80; 0.00) | 12.93 (7.63; 0.00) | 28.67 (18.14; 0.00) | 37.44 (23.01; 0.00) | 38.55 (22.45; 0.00) |
| gi12-rays-feedback-temporal | 5.13 (2.88; 0.00) | 4.34 (2.10; 0.00) | 10.62 (6.79; 0.00) | 14.42 (8.23; 0.00) | 28.31 (17.96; 0.00) | 37.73 (23.24; 0.00) | 40.91 (23.99; 0.00) |
| gi12-ddgi-temporal | 5.42 (2.23; 0.94) | 4.84 (1.68; 0.89) | 9.70 (4.60; 1.36) | 12.40 (4.47; 2.63) | 21.05 (8.08; 2.33) | 29.01 (11.11; 3.59) | 28.75 (9.05; 3.58) |

Each cell is the radiance cache's passes plus DDGI's, with the world cache's own passes and DDGI's in brackets (and the neural tail's, where it runs). The camera is still, at the first bookmark, 1280 × 720. Memory comes from the same run. It doesn't depend on the scene, except DDGI's, which follows the volume's probe count:

| Variant | Radiance cache MiB (its world cache) | DDGI MiB |
|---|---|---|
| gi12-rays | 1,220 (1,085) | none |
| gi12-ddgi | 1,200 (1,065) | 12 (two-room), 16 (Cornell), 55 (Sponza, Bistro interior), 72 (Bistro exterior) |
| sharc | 217 (83) | none |
| sharc-ddgi | 217 (83) | as gi12-ddgi |
| budgeted | 1,202 (1,068) | as gi12-ddgi |
| neural-tail | 1,246 (1,087) | none |

The GI-1.2 grid's 1 GiB is Capsaicin's default capacity (2^14 buckets × 16 tiles × 85 cells; "World-cache memory" in radiance-cache.md). Most of it sits unused on every scene here: Sponza fills about 50,000 tiles of 262,144. The screen side adds about 135 MiB at 720p and about 330 MiB at 1080p.



## Trade-offs

Measured: ten views of seven scenes (both Cornell boxes, two-room, two views each of the furnished room, Bistro interior and Bistro exterior, and Sponza). Both timing runs are INVALID (thermal throttling), so the milliseconds below compare the variants with each other only.

**`gi12-rays`: the GI-1.2 grid alone (Capsaicin's).**
- For: it needs nothing from the host. It warms up fast (median 22 frames). It is the faithful baseline.
- Against: it carries only two bounces, and the cache's own lights leave out emitters. Closed rooms read 25-62% dark against Cycles, by far the worst error here. Open scenes are fine (−5 to −7%). It is also among the most expensive on the big scenes (38-39 ms on Sponza and the Bistros, the second-bounce rays included), and it holds the grid's 1.4 GiB at 1080p.
- How it fails: closed rooms come out dark everywhere, and an emitter-lit room most of all. This is the bias, not noise.

**`gi12-ddgi`: the grid over DDGI (what ships).**
- For: it has the smallest bias on every view (mean size 3.1%; two-room −0.9% with 1.5% block error). Its ghost clears soonest after the box moves (median 74 frames to 90%), and its frame brightness is the calmest. On the big scenes it costs less than the grid alone (22-29 ms), because DDGI's field replaces the second-bounce rays and their cells.
- Against: the host must place a DDGI volume around the scene. DDGI adds 0.8-3.6 ms and 12-72 MiB, and the grid's 1.4 GiB stays. It warms up more slowly than the grid alone (median 49 frames, 72-73 in the furnished room): until DDGI's field fills there is no later bounce. In the open scenes its 8×8 p95 is the highest (Sponza 52% against 40%, Bistro exterior 47% against 38%): DDGI's coarse field blurs the bounce at contacts.
- How it fails: a hit outside every volume gets no bounce after the first (DDGI's coverage is 0 there), so a scene bigger than its volume goes dark past the edge. A coarse volume leaks light through thin walls, which DDGI's visibility only partly stops. After a cut or a load the picture is dark until the field fills: about 70 frames for Sponza's field at 256 probes a frame, about 100 for Bistro exterior's.

**`sharc`: the SHaRC cache alone.**
- For: it needs nothing from the host and still carries every bounce, emitters included. Its bias is 2-5 points behind the grid over DDGI (mean size 5.9%, worst −9.7% in the furnished room). In the open scenes its block error ties or beats the grid over DDGI's, and it has the lowest (or equal-lowest) p95 on the Cornell boxes, Sponza and both Bistros. Its memory is less than a third of the grid's: 415 MiB at 1080p in all, 83 MiB for the cache itself. Its cells cover almost every visible surface.
- Against: it follows changes more slowly than the grid over DDGI. The lights' 90% comes at a median of 83 frames against 77, and the box's ghost at 85 against 74. Its warm-up is the longest of the cache-only variants in the rooms (156-157 in the furnished room; median 45 over all views, 1-28 in the open scenes). Its frame brightness moves 2-3 times more than the grid over DDGI's, though still a few thousandths of a percent. Its world cache is the costliest part on the big scenes: 14-19 ms, against the grid's 8-11 ms plus DDGI's 2.5-3.6 ms. It is a port from pmndrs/pathtracer, not Capsaicin's design.
- How it fails: it lags after changes, because its 32-frame mean and quarter-rate updates trade reaction for calm. Its coarse cells (its level-of-detail rule) blur the light near contacts.

**`sharc-ddgi`: SHaRC with the volume still under it.**
- The same picture as SHaRC on every scene, because SHaRC doesn't read the volume. It only adds DDGI's time and memory. It is not a candidate. It exists because the viewer stacks DDGI whatever the world cache is.

**`budgeted`: budgeted cells over DDGI.**
- For: its accuracy, flicker and memory match the grid over DDGI. It is the cheapest variant on the big scenes (20-27 ms against 22-29), and the gain grows with the scene.
- Against: it reacts more slowly to light changes. Four frames after the lights double, a median 98% of the change is still missing, against 91% for the grid over DDGI. The 90% mark comes at a median of 86 frames, and at 100 in Bistro interior. On the small scenes it costs more than per hit: 5.9 ms against 5.2 on Cornell, since sorting and claiming cells is a fixed cost.
- How it fails: under heavy change, cells wait their turn, so lighting updates lag. A budget too small for the scene shows as slow convergence.

**`neural-tail`: the grid with the neural tail.**
- For: it needs no DDGI volume, and in rooms lit by lights it closes most of the grid alone's missing bounces (two-room −44% → −7%, Cornell point −25% → −3.7%).
- Against: it can't light an emissive scene. The emissive Cornell box reads as the grid alone, −49%. Its warm-up is the slowest in the rooms (138-214 frames in the rooms lit by lights; median 31 over all views), because the network trains first. When still, it pumps: the frame brightness changes up to 10 times more than the others, as training steps. It is the most expensive variant, with 2.1-5.4 ms for the network on top of the grid alone's rays. It needs `shader-f16` for full speed and the optional `@pathtracer/mlp` package, which isn't on npm. Without either it runs as the grid alone, with a warning.
- How it fails: after a cut it reads dark until the network learns. It drifts a little from frame to frame. Emitter-lit rooms, or a missing package, leave it at the grid alone's error.

**`gi12-rays-feedback`: the grid alone with cache feedback (RC3.6, E12).**
- For: it needs nothing from the host, costs no extra rays, and halves the grid alone's error in closed rooms lit by lights (two-room −44% → −20%, Cornell point −25% → −6%). Open scenes are unchanged.
- Against: the rooms are still 20-40% dark, well behind SHaRC and DDGI. It does nothing for an emitter-lit room (emissive Cornell −49%). It warms up slowly in the rooms (88-145 frames), since each frame adds one more bounce.
- How it fails: a dark closed room, less dark than the grid alone. Emitter-lit rooms stay as dark as the grid alone.

**`gi12-rays-temporal`, `gi12-rays-feedback-temporal`, `gi12-ddgi-temporal`: temporal feedback (RC3.6, D68).**
- For: it costs no rays, and in the open scenes it removes the grid alone's small darkening (Bistro exterior +0.5 to +1.5%, Sponza +3.5%).
- Against: it over-brightens where last frame's picture shows lamps or emitters: Bistro interior +14 to +23%, emissive Cornell +3% (grid alone) and +26% (over DDGI). It barely helps closed rooms whose lights are off screen (two-room −33%, the furnished room −50 to −53%). Over DDGI it is worse than DDGI alone on every view. On Bistro exterior's second view it pumps (the frame brightness changes 0.05-0.10% a frame, 15-30 times the others) and follows the lights doubling in 105-107 frames against 82. It needs the host's lit image and albedo, and changing it restarts the technique.
- How it fails: bright where lamps or emitters are on screen, and dark where the light is off screen. It also feeds the picture's own errors back into the cache: it reads what the screen showed, including GI that is already biased.

## Recommendation

**Accepted by Dennis, 2026-11-28 (DECISIONS).** Neither Bistro exterior (the one outdoor scene), RC3.6's modes nor the five VALID timing runs changed it. Points 2 and 3 are TASKS P6.7 and P6.8; VALID timing for the rest is P6.9.

1. **Keep the GI-1.2 grid over DDGI (`gi12-ddgi`) as the default when the host has a DDGI volume.** It has the smallest bias on all ten views and is the quickest to clear a ghost. It is also the calmest when still, and on big scenes it costs less than the grid alone.
2. **Use SHaRC, not the grid on its own rays, when there is no DDGI volume.** Today `secondBounce: 'auto'` without a volume falls back to GI-1.2's rays, which read 25-62% dark in every closed room. SHaRC needs nothing from the host and is within 2-5 points of the grid over DDGI, matching it in the open scenes. It also uses under a third of the memory. It means `'auto'` picking the world cache at `prepare` by whether a fallback is stacked (P6.7).
3. **Shrink the GI-1.2 grid's capacity, whichever is picked.** Its 1 GiB is Capsaicin's default and stays mostly empty on these scenes. On a Mac or a smaller GPU it is the first thing that won't fit. Sizing it from measured bucket use is a task of its own.
4. **Keep budgeted cells as an opt-in for big scenes**, where it saves GPU time. It reacts more slowly to light changes and costs more on small scenes.
5. **Leave the neural tail opt-in.** It can't light emitters, it is slow to warm up, it pumps, it costs the most, and its package is private. DDGI or SHaRC carry the later bounces better here.
6. **Don't stack DDGI under SHaRC.** That costs the volume for nothing: if SHaRC is used, leave the volume out. The viewer does stack it today.
7. **Leave RC3.6's modes off (opt-in), as they are.** Cache feedback is the cheapest way to more bounces on the grid alone, but it stays 20-40% dark in closed rooms, so it doesn't replace SHaRC in point 2. Temporal feedback over-brightens rooms with lamps or emitters in view, does worse than DDGI alone when stacked with it, and pumps in one outdoor view.

What would change this:
- **A VALID timing run:** both throttled runs have SHaRC's cache slower than the grid over DDGI on big scenes.
- **The Mac:** memory, and f16.

## Still open

- **References:** Sponza's `upper-gallery` is not rendered yet. To add it, run the variants page with `VITE_RC_VARIANT_VIEWS=upper-gallery`, then pass every report to the table script.
- **VALID timing for the rest.** Five of 46 split runs are VALID (Bistro exterior, five presets). The other scenes, the neural tail and RC3.6's presets are relative (INVALID) only. `node tools/reference/rc-variants-timing-split.ts` resumes them when the GPU is free; the 30 s pause keeps each run under the clock rule.
- **The Mac** (M5 Pro, the target GPU): memory (the grid's 1 GiB), the f16 tail, and the timing.
- **The lag every variant shares.** After the lights double, the picture takes over a second to reach 90%, about the same for every variant. Most of it is in the screen probes and the world cache, not in the choice of world cache. That needs its own look at the change resets, outside this task.
