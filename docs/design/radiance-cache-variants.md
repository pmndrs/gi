# Radiance cache: the world-cache variants compared (P6.5)

Status: in progress (2026-11-27, branch `p6-5`). Six scenes are measured; Bistro exterior waits on its reference, and the GPU times need one VALID timing run at the end. **No default changes here:** Dennis picks the default from this page (DECISIONS 2026-11-23).

The radiance cache's world cache (the cache the screen probes' rays end in) has grown five alternatives since the GI-1.2 baseline. This page puts them side by side on the same scenes, against the path tracer (Cycles), and says what each is good and bad at.

## The variants

Each is a preset in the viewer's compare panel ("Radiance cache · compare", the **Variant** list), so it can be seen live, and a name the GPU pages take (`VITE_RC_COMPARE=variant=<id>`). The presets are `RADIANCE_CACHE_VARIANT_PRESETS` in `packages/ddgi-core/src/techniques/radiance-cache/compare-toggles.ts`.

| Preset | What it is | Needs from the host | Register |
|---|---|---|---|
| `gi12-rays` | Capsaicin's GI-1.2 hash grid alone; its second bounce from its own rays. Two bounces. | nothing | E10's way back |
| `gi12-ddgi` | **What ships.** The grid with a DDGI volume stacked under it (`setFallback`); DDGI gives the second bounce and, through its own feedback, every bounce after. 256 DDGI probes a frame. | a DDGI volume around the scene | E10 |
| `sharc` | pmndrs/pathtracer's SHaRC-style hash cache instead of the grid, alone. Two bounce rays a sample with the cache as the path's tail: every bounce. | nothing | E1-E7 |
| `sharc-ddgi` | SHaRC with the viewer's DDGI volume still stacked. SHaRC doesn't read the volume, so the picture is SHaRC's and the volume is pure cost. Listed because it is what the viewer gives when only the world cache is switched. | a DDGI volume (unused) | E1, E10 |
| `budgeted` | The grid over DDGI, its cells lit on a budget (20,000 a frame, one light sample each, new and changed cells first) instead of at every hit (id Tech 8). | a DDGI volume | E8, E10 |
| `neural-tail` | The grid alone on its own rays, plus a small network that learns the cells' indirect light and answers the bounces past the second. | `shader-f16` (f32 works, slower), the optional `@pathtracer/mlp` sibling | E11 |

RC3.6's modes (more bounces in the grid: cache feedback, temporal feedback) are built in parallel on `rc3-6`; they become rows here when they land.

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

`VITE_RC_VARIANTS` and `VITE_RC_VARIANT_SCENES` pick a subset; a scene without a reference is skipped and listed.

## Results

Measured 2026-11-27 on the RTX 3060 Laptop (Chrome, bench lock). The accuracy, warm-up, flicker, lag and memory numbers don't depend on the GPU clock. Read the columns as follows:

- **GI bias, 8×8 mean, 8×8 p95**: the GI term averaged over 60 frames against Cycles. Bias is the signed error of the whole picture; the 8×8 numbers are the error of 8 × 8 pixel blocks (their mean, and the worst 5%). **One frame mean** is the same for a single frame (noise included).
- **Cache coverage**: the share of visible surfaces the world cache holds a cell for. It is not the cache's quality (probe rays land where they land), but a low share means fewer cells near the camera. With DDGI under the grid, second-bounce hits don't insert cells, so it reads lower.
- **Warm-up**: frames from an empty cache, with DDGI's field starting black, until the picture's mean stays within 5% of where it settles.
- **Flicker**: what the picture does when nothing moves. Per pixel is the mean change from one frame to the next, with the 95th percentile in brackets. "Of the frame" is the whole picture's brightness change, which shows pumping.
- **Lights ×2** and **box moves**: frames until the picture is 50% and 90% of the way to its new state, and the share of the change still missing 4 frames after (100% means nothing has moved yet). For the box, the region is the light around it: its old shadow lit again (the ghost) and its new shadow, with the box's own pixels left out. At 60 frames a second, 60 frames is one second.
- **Memory**: the radiance cache's GPU memory at 1920 × 1080, with DDGI's in brackets.

### Summary

Each scene's GI bias and 8×8 mean error, then the median over the six scenes for the dynamic numbers (a box move too small to read, in the furnished room, left out). Memory is the radiance cache's at 1080p, DDGI's not included.

| Variant | Cornell point | Cornell emissive | two-room | furnished room | Sponza | Bistro interior | mean of the bias's size | lights ×2: 90% (frames, median) | box moves: 90% (median) | warm-up (median) | memory MiB |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gi12-rays | −25.1% / 25.8% | −49.1% / 49.5% | −43.8% / 43.8% | −62.0% / 62.0% | −6.6% / 13.4% | −4.8% / 15.5% | 31.9% | 67 | 82 | 28 | 1,419 |
| **gi12-ddgi** | −0.1% / 6.0% | −1.8% / 10.2% | −0.9% / 1.5% | −8.1% / 8.2% | −2.5% / 12.6% | −0.6% / 15.3% | **2.3%** | 72 | 75 | 48 | 1,389 (+12-55 DDGI) |
| sharc | −3.2% / 7.0% | −3.8% / 10.8% | −4.4% / 4.4% | −9.7% / 9.8% | −6.1% / 13.2% | −4.2% / 14.2% | 5.2% | 80 | 88 | 65 | **415** |
| sharc-ddgi | as sharc | | | | | | 5.2% | 80 | 89 | 65 | 415 (+DDGI) |
| budgeted | −0.4% / 6.1% | −1.8% / 10.2% | −0.8% / 1.5% | −8.1% / 8.2% | −2.6% / 12.7% | −1.7% / 15.5% | 2.6% | 81 | 80 | 53 | 1,395 (+DDGI) |
| neural-tail | −3.7% / 7.6% | −49.1% / 49.5% | −7.0% / 7.1% | −13.7% / 13.7% | −5.9% / 13.2% | −2.8% / 15.2% | 13.7% | 83 | 84 | 87 | 1,453 |

Things the table shows:

- **Only the closed rooms separate the variants.** In Sponza and Bistro interior every variant is within 3-15% block error and −0.6 to −6.6% bias. In the four closed rooms the GI-1.2 grid alone is 25-62% dark, and everything that carries more bounces is within 10%.
- **The grid over DDGI is the most accurate everywhere**, budgeted cells (the same grid) next, SHaRC 2-5 points behind it.
- **The neural tail can't light an emissive scene.** In the emissive Cornell box it reads exactly as the grid alone. The cache's own lights don't include emitters (register D21), so the cells it learns from hold no light. In the furnished room it is 5 points behind DDGI.
- **SHaRC under DDGI is SHaRC.** It doesn't read the volume, so the numbers match to the noise and the volume is pure cost.
- **The earlier Sponza excess is gone.** On the old references (RC4.6) the grid over DDGI read +14.5% / +20%. Against the new references it reads −2.5%. The new references and what merged since (RC3.4's sky light, RC4.2) both changed; which one moved it was not looked into.
- Sponza and the furnished room warn that a node-graph material reads as grey 0.5 in GI. That is the same for every variant, so it doesn't change the comparison.

### Against Cycles, by scene

#### cornell-box-point (`front`)

| Variant | GI bias | 8×8 mean | 8×8 p95 | one frame mean | cache coverage | warm-up (frames) | flicker per pixel (p95) | flicker of the frame | lights ×2: 50% / 90% frames · left at 4 | box moves: 50% / 90% frames · left at 4 | memory MiB (DDGI) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gi12-rays | -25.1% | 25.8% | 35% | 25.8% | 52% | 32 | 0.09% (0.3%) | 0.003% | 22 / 67 · 89% | 30 / 84 · 100% | 1419 (0) |
| gi12-ddgi | -0.1% | 6.0% | 23% | 6.0% | 51% | 43 | 0.08% (0.2%) | 0.001% | 24 / 64 · 91% | 28 / 74 · 100% | 1389 (16) |
| sharc | -3.2% | 7.0% | 19% | 7.0% | 93% | 62 | 0.08% (0.2%) | 0.003% | 23 / 68 · 90% | 37 / 90 · 101% | 415 (0) |
| sharc-ddgi | -3.2% | 7.0% | 19% | 6.9% | 93% | 61 | 0.08% (0.2%) | 0.003% | 23 / 68 · 90% | 37 / 89 · 101% | 431 (16) |
| budgeted | -0.4% | 6.1% | 23% | 6.0% | 51% | 40 | 0.08% (0.2%) | 0.001% | 30 / 71 · 98% | 28 / 82 · 101% | 1395 (16) |
| neural-tail | -3.7% | 7.6% | 19% | 7.3% | 52% | 138 | 0.08% (0.3%) | 0.012% | 26 / 79 · 91% | 32 / 83 · 100% | 1453 (0) |

Lights: x2 PointLight. Box: a 0.67 m box moved 1.01 m (its own 203 blocks left out).

#### cornell-box (`front`)

| Variant | GI bias | 8×8 mean | 8×8 p95 | one frame mean | cache coverage | warm-up (frames) | flicker per pixel (p95) | flicker of the frame | lights ×2: 50% / 90% frames · left at 4 | box moves: 50% / 90% frames · left at 4 | memory MiB (DDGI) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gi12-rays | -49.1% | 49.5% | 96% | 49.5% | 50% | 1 | 0.14% (0.6%) | 0.002% | 22 / 64 · 99% | 23 / 77 · 99% | 1419 (0) |
| gi12-ddgi | -1.8% | 10.2% | 33% | 10.1% | 49% | 64 | 0.10% (0.3%) | 0.002% | 25 / 84 · 99% | 26 / 75 · 100% | 1389 (16) |
| sharc | -3.8% | 10.8% | 29% | 10.7% | 93% | 68 | 0.10% (0.3%) | 0.002% | 32 / 91 · 99% | 39 / 85 · 101% | 415 (0) |
| sharc-ddgi | -3.8% | 10.8% | 29% | 10.7% | 93% | 68 | 0.10% (0.3%) | 0.002% | 32 / 90 · 99% | 38 / 88 · 101% | 431 (16) |
| budgeted | -1.8% | 10.2% | 33% | 10.1% | 49% | 64 | 0.10% (0.3%) | 0.002% | 25 / 84 · 99% | 26 / 75 · 100% | 1395 (16) |
| neural-tail | -49.1% | 49.5% | 96% | 49.5% | 51% | 1 | 0.14% (0.6%) | 0.002% | 22 / 64 · 99% | 23 / 77 · 99% | 1453 (0) |

Lights: x2 1 emissive material(s). Box: a 0.67 m box moved 1.01 m (its own 203 blocks left out).

#### two-room (`lit-room`)

| Variant | GI bias | 8×8 mean | 8×8 p95 | one frame mean | cache coverage | warm-up (frames) | flicker per pixel (p95) | flicker of the frame | lights ×2: 50% / 90% frames · left at 4 | box moves: 50% / 90% frames · left at 4 | memory MiB (DDGI) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gi12-rays | -43.8% | 43.8% | 48% | 43.8% | 100% | 30 | 0.05% (0.1%) | 0.000% | 22 / 63 · 90% | 22 / 78 · 99% | 1419 (0) |
| gi12-ddgi | -0.9% | 1.5% | 5% | 1.5% | 60% | 52 | 0.03% (0.1%) | 0.002% | 24 / 58 · 93% | 20 / 67 · 98% | 1385 (12) |
| sharc | -4.4% | 4.4% | 7% | 4.3% | 99% | 90 | 0.04% (0.1%) | 0.005% | 25 / 73 · 92% | 23 / 84 · 97% | 415 (0) |
| sharc-ddgi | -4.4% | 4.4% | 8% | 4.3% | 99% | 90 | 0.04% (0.1%) | 0.005% | 25 / 73 · 92% | 21 / 83 · 97% | 427 (12) |
| budgeted | -0.8% | 1.5% | 5% | 1.5% | 60% | 52 | 0.03% (0.1%) | 0.002% | 28 / 62 · 98% | 20 / 69 · 98% | 1391 (12) |
| neural-tail | -7.0% | 7.1% | 11% | 6.4% | 100% | 173 | 0.05% (0.1%) | 0.022% | 31 / 87 · 93% | 25 / 85 · 96% | 1453 (0) |

Lights: x2 PointLight. Box: a 0.36 m box moved 0.54 m (its own 281 blocks left out).

#### furnished-room (`from-the-corner`)

| Variant | GI bias | 8×8 mean | 8×8 p95 | one frame mean | cache coverage | warm-up (frames) | flicker per pixel (p95) | flicker of the frame | lights ×2: 50% / 90% frames · left at 4 | box moves: 50% / 90% frames · left at 4 | memory MiB (DDGI) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gi12-rays | -62.0% | 62.0% | 65% | 62.0% | 100% | 31 | 0.04% (0.1%) | 0.000% | 25 / 67 · 92% | 28 / 105 · 94% | 1419 (0) |
| gi12-ddgi | -8.1% | 8.2% | 10% | 8.0% | 70% | 73 | 0.03% (0.1%) | 0.007% | 32 / 77 · 96% | (2 blocks) | 1402 (29) |
| sharc | -9.7% | 9.8% | 12% | 9.2% | 100% | 157 | 0.04% (0.1%) | 0.019% | 31 / 86 · 94% | 32 / 105 · 92% | 415 (0) |
| sharc-ddgi | -9.8% | 9.8% | 12% | 9.2% | 100% | 157 | 0.04% (0.1%) | 0.019% | 31 / 86 · 94% | 33 / 105 · 92% | 444 (29) |
| budgeted | -8.1% | 8.2% | 10% | 8.0% | 70% | 72 | 0.03% (0.1%) | 0.007% | 34 / 78 · 99% | (6 blocks) | 1408 (29) |
| neural-tail | -13.7% | 13.7% | 17% | 12.1% | 100% | 214 | 0.07% (0.2%) | 0.053% | 42 / 101 · 95% | 29 / 101 · 92% | 1453 (0) |

Lights: x2 DirectionalLight, PointLight. Box: a 0.41 m box moved 0.61 m (its own 304 blocks left out).

#### sponza (`colonnade-from-the-floor`)

| Variant | GI bias | 8×8 mean | 8×8 p95 | one frame mean | cache coverage | warm-up (frames) | flicker per pixel (p95) | flicker of the frame | lights ×2: 50% / 90% frames · left at 4 | box moves: 50% / 90% frames · left at 4 | memory MiB (DDGI) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gi12-rays | -6.6% | 13.4% | 40% | 13.4% | 90% | 1 | 0.23% (0.8%) | 0.002% | 18 / 85 · 77% | 23 / 80 · 98% | 1419 (0) |
| gi12-ddgi | -2.5% | 12.6% | 52% | 12.7% | 49% | 30 | 0.21% (0.8%) | 0.002% | 19 / 84 · 79% | 23 / 79 · 98% | 1428 (55) |
| sharc | -6.1% | 13.2% | 40% | 13.3% | 76% | 1 | 0.23% (0.8%) | 0.002% | 18 / 86 · 75% | 22 / 81 · 98% | 415 (0) |
| sharc-ddgi | -6.1% | 13.2% | 40% | 13.3% | 76% | 1 | 0.23% (0.8%) | 0.002% | 18 / 86 · 75% | 22 / 81 · 98% | 470 (55) |
| budgeted | -2.6% | 12.7% | 52% | 12.7% | 47% | 40 | 0.21% (0.8%) | 0.002% | 34 / 92 · 94% | 23 / 80 · 99% | 1434 (55) |
| neural-tail | -5.9% | 13.2% | 41% | 13.3% | 92% | 1 | 0.23% (0.8%) | 0.001% | 19 / 86 · 78% | 23 / 80 · 98% | 1453 (0) |

Lights: x2 DirectionalLight. Box: a 2.40 m box moved 3.60 m (its own 959 blocks left out).

#### bistro-interior (`dining-room`)

| Variant | GI bias | 8×8 mean | 8×8 p95 | one frame mean | cache coverage | warm-up (frames) | flicker per pixel (p95) | flicker of the frame | lights ×2: 50% / 90% frames · left at 4 | box moves: 50% / 90% frames · left at 4 | memory MiB (DDGI) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gi12-rays | -4.8% | 15.5% | 35% | 15.5% | 100% | 25 | 0.16% (0.5%) | 0.002% | 17 / 66 · 83% | 20 / 90 · 96% | 1419 (0) |
| gi12-ddgi | -0.6% | 15.3% | 38% | 15.3% | 91% | 40 | 0.15% (0.5%) | 0.002% | 20 / 67 · 84% | 24 / 104 · 97% | 1428 (55) |
| sharc | -4.2% | 14.2% | 34% | 14.2% | 100% | 1 | 0.16% (0.5%) | 0.002% | 17 / 68 · 83% | 20 / 90 · 97% | 415 (0) |
| sharc-ddgi | -4.2% | 14.2% | 34% | 14.2% | 100% | 1 | 0.16% (0.5%) | 0.002% | 17 / 68 · 83% | 20 / 89 · 97% | 470 (55) |
| budgeted | -1.7% | 15.5% | 37% | 15.4% | 91% | 53 | 0.15% (0.5%) | 0.002% | 39 / 100 · 99% | 20 / 94 · 96% | 1434 (55) |
| neural-tail | -2.8% | 15.2% | 37% | 15.2% | 100% | 35 | 0.15% (0.5%) | 0.003% | 17 / 68 · 84% | 20 / 84 · 97% | 1453 (0) |

Lights: x2 PointLight, PointLight, PointLight. Box: a 0.89 m box moved 1.33 m (its own 751 blocks left out).

**Waiting on references:** Bistro exterior (the only outdoor scene: the sky and DDGI's big volume, at 4 m spacing). Sponza and Bistro interior have one view each so far (`colonnade-from-the-floor`, `dining-room`); the page measures the first bookmark with a reference.

### Where the lag comes from

On two-room with the GI denoiser off (`VITE_RC_COMPARE=denoiser=false`), the lights' 90% came at 50 frames (grid over DDGI) and 66 (SHaRC), against 58 and 73 with it on. The box's ghost reached 90% at 40 and 59 frames, against 67 and 84. Per-pixel flicker rose from 0.03-0.04% to 1.6-1.7%. So the denoiser is what makes every variant's still picture this calm, and it adds about 25 frames to the box's ghost. Most of the light's lag happens before it, in the screen probes and the world cache, and it is about the same for every variant.

### GPU time

**Not yet VALID.** The run below is a dry run of the timing page. The GPU started at 82 C, so the clock rule marks it INVALID. The SM clock under load had a median of 1,050 MHz, and the run shared the session with other agents' pages. Use it only to compare the variants with each other, not as milliseconds. A VALID run (cool start, the cool wait on) replaces it once the renders are done.

**GPU ms per frame at 1280×720: radiance cache + DDGI (world cache; DDGI) — timing INVALID (hot start: the GPU was at 82 C (70 C or more))**

| Variant | cornell-box-point | cornell-box | two-room | sponza | bistro-interior | bistro-exterior |
|---|---|---|---|---|---|---|
| gi12-rays | 3.99 (2.62; 0.00) | 4.20 (2.01; 0.00) | 10.56 (6.87; 0.00) | 35.07 (21.87; 0.00) | 45.26 (27.84; 0.00) | 46.11 (27.01; 0.00) |
| gi12-ddgi | 4.44 (2.20; 0.67) | 4.94 (1.79; 0.88) | 9.73 (4.68; 1.33) | 23.98 (8.99; 2.77) | 34.32 (13.08; 4.41) | 33.32 (10.33; 4.23) |
| sharc | 3.55 (1.31; 0.00) | 3.44 (1.18; 0.00) | 7.80 (3.04; 0.00) | 27.82 (15.18; 0.00) | 35.37 (18.45; 0.00) | 41.66 (22.82; 0.00) |
| sharc-ddgi | 4.49 (1.30; 0.94) | 4.33 (1.18; 0.89) | 9.69 (3.08; 1.71) | 30.35 (15.13; 2.93) | 39.67 (18.31; 4.48) | 46.58 (22.94; 4.52) |
| budgeted | 5.84 (2.66; 0.93) | 5.67 (2.51; 0.89) | 10.04 (5.03; 1.31) | 22.18 (7.42; 2.64) | 28.28 (7.73; 4.18) | 31.08 (8.32; 4.13) |
| neural-tail | 7.80 (3.07; 0.00; NRC 2.48) | 6.62 (2.02; 0.00; NRC 2.42) | 13.86 (6.91; 0.00; NRC 3.24) | 37.87 (20.97; 0.00; NRC 4.49) | 48.61 (27.14; 0.00; NRC 4.66) | 51.66 (26.80; 0.00; NRC 6.24) |

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

Six scenes measured: both Cornell boxes, two-room, the furnished room, Sponza and Bistro interior. Bistro exterior is still to come, and the timing run is a dry run.

**`gi12-rays`: the GI-1.2 grid alone (Capsaicin's).**
- For: it needs nothing from the host. It warms up fast (median 28 frames). It is the faithful baseline.
- Against: it carries only two bounces, and the cache's own lights leave out emitters. Closed rooms read 25-62% dark against Cycles, by far the worst error here. Open scenes are fine (−5 to −7%). It is also among the most expensive on the big scenes (35-46 ms in the dry run, the second-bounce rays included), and it holds the grid's 1.4 GiB at 1080p.
- How it fails: closed rooms come out dark everywhere, and an emitter-lit room most of all. This is the bias, not noise.

**`gi12-ddgi`: the grid over DDGI (what ships).**
- For: it is the most accurate on every scene (mean bias size 2.3%; two-room −0.9% with 1.5% block error). Its ghost clears soonest after the box moves (median 75 frames to 90%), and its frame brightness is the calmest. On the big scenes it costs less than the grid alone (24-34 ms in the dry run), because DDGI's field replaces the second-bounce rays and their cells.
- Against: the host must place a DDGI volume around the scene. DDGI adds 0.7-4.4 ms and 12-72 MiB, and the grid's 1.4 GiB stays. It warms up more slowly than the grid alone (median 48 frames, 73 in the furnished room): until DDGI's field fills there is no later bounce. Its 8×8 p95 is the highest on Sponza (52% against 40%): DDGI's coarse field blurs the bounce at contacts.
- How it fails: a hit outside every volume gets no bounce after the first (DDGI's coverage is 0 there), so a scene bigger than its volume goes dark past the edge. A coarse volume leaks light through thin walls, which DDGI's visibility only partly stops. After a cut or a load the picture is dark until the field fills: about 70 frames for Sponza's field at 256 probes a frame, about 100 for Bistro exterior's.

**`sharc`: the SHaRC cache alone.**
- For: it needs nothing from the host and still carries every bounce, emitters included. Its error is 2-5 points behind the grid over DDGI (mean bias size 5.2%, worst −9.7% in the furnished room), and it has the lowest (or equal-lowest) p95 on the Cornell boxes and Sponza. Its memory is less than a third of the grid's: 415 MiB at 1080p in all, 83 MiB for the cache itself. Its cells cover almost every visible surface.
- Against: it follows changes more slowly than the grid over DDGI. The lights' 90% comes at a median of 80 frames against 72, and the box's ghost at 88 against 75. Its warm-up is the longest of the cache-only variants in the rooms (median 65, 157 in the furnished room). Its frame brightness moves 2-3 times more than the grid over DDGI's, though still a few thousandths of a percent. In the dry run its world cache is the costliest part on the big scenes: 15-23 ms, against the grid's 9-13 ms plus DDGI's 3-4 ms. It is a port from pmndrs/pathtracer, not Capsaicin's design.
- How it fails: it lags after changes, because its 32-frame mean and quarter-rate updates trade reaction for calm. Its coarse cells (its level-of-detail rule) blur the light near contacts.

**`sharc-ddgi`: SHaRC with the volume still under it.**
- The same picture as SHaRC on every scene, because SHaRC doesn't read the volume. It only adds DDGI's time and memory. It is not a candidate. It exists because the viewer stacks DDGI whatever the world cache is.

**`budgeted`: budgeted cells over DDGI.**
- For: its accuracy, flicker and memory match the grid over DDGI. It is the cheapest variant on the big scenes in the dry run (22-31 ms against 24-34), and the gain grows with the scene.
- Against: it reacts more slowly to light changes. Four frames after the lights double, a median 98% of the change is still missing, against 92% for the grid over DDGI. The 90% mark comes at a median of 81 frames, and at 100 in Bistro interior. On the small scenes it costs more than per hit: 5.8 ms against 4.4 on Cornell, since sorting and claiming cells is a fixed cost.
- How it fails: under heavy change, cells wait their turn, so lighting updates lag. A budget too small for the scene shows as slow convergence.

**`neural-tail`: the grid with the neural tail.**
- For: it needs no DDGI volume, and in rooms lit by lights it closes most of the grid alone's missing bounces (two-room −44% → −7%, Cornell point −25% → −3.7%).
- Against: it can't light an emissive scene. The emissive Cornell box reads as the grid alone, −49%. Its warm-up is the slowest (median 87 frames, 138-214 in the rooms lit by lights), because the network trains first. When still, it pumps: the frame brightness changes up to 10 times more than the others, as training steps. It is the most expensive variant, with 2.4-6.2 ms for the network on top of the grid alone's rays. It needs `shader-f16` for full speed and the optional `@pathtracer/mlp` package, which isn't on npm. Without either it runs as the grid alone, with a warning.
- How it fails: after a cut it reads dark until the network learns. It drifts a little from frame to frame. Emitter-lit rooms, or a missing package, leave it at the grid alone's error.

## Recommendation

**Provisional, pending Bistro exterior and a VALID timing run.**

1. **Keep the GI-1.2 grid over DDGI (`gi12-ddgi`) as the default when the host has a DDGI volume.** It is the most accurate on all six scenes and the quickest to clear a ghost. It is also the calmest when still, and on big scenes it costs less than the grid alone.
2. **Use SHaRC, not the grid on its own rays, when there is no DDGI volume.** Today `secondBounce: 'auto'` without a volume falls back to GI-1.2's rays, which read 25-62% dark in every closed room. SHaRC needs nothing from the host and is within 2-5 points of the grid over DDGI. It also uses under a third of the memory. It would mean `'auto'` picking the world cache at `prepare` by whether a fallback is stacked. Dennis's call.
3. **Shrink the GI-1.2 grid's capacity, whichever is picked.** Its 1 GiB is Capsaicin's default and stays mostly empty on these scenes. On a Mac or a smaller GPU it is the first thing that won't fit. Sizing it from measured bucket use is a task of its own.
4. **Keep budgeted cells as an opt-in for big scenes**, where it saves GPU time. It reacts more slowly to light changes and costs more on small scenes.
5. **Leave the neural tail opt-in.** It can't light emitters, it is slow to warm up, it pumps, it costs the most, and its package is private. DDGI or SHaRC carry the later bounces better here.
6. **Don't stack DDGI under SHaRC.** That costs the volume for nothing: if SHaRC is used, leave the volume out. The viewer does stack it today.

What would change this:
- **Bistro exterior:** the sky, and DDGI's big 4 m volume.
- **The VALID timing run:** in the dry run, SHaRC's cache is slower than the grid over DDGI on big scenes.
- **RC3.6's modes:** if cache or temporal feedback gives the grid every bounce without DDGI, that could replace point 2.
- **The Mac:** memory, and f16.

## Still open

- **References:** Bistro exterior is still being rendered, and the second views of Sponza and Bistro interior are coming. To add one, run the variants page with `VITE_RC_VARIANT_SCENES=<scene>` and pass every report to the table script.
- **A VALID timing run** of `radiance-cache-variants-timing`, with the cool wait, once the GPU is free of the renders.
- **RC3.6's modes** (cache feedback and temporal feedback, on `rc3-6`), as presets and rows.
- **The Mac** (M5 Pro, the target GPU): memory (the grid's 1 GiB), the f16 tail, and the timing.
- **The lag every variant shares.** After the lights double, the picture takes over a second to reach 90%, about the same for every variant. Most of it is in the screen probes and the world cache, not in the choice of world cache. That needs its own look at the change resets, outside this task.
