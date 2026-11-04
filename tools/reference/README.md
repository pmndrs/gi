# References and compare (T0.7)

Ground truth on demand, and one way to compare images.

```
node tools/reference/cycles/run.ts       --scene cornell-box --camera 1 --samples 2048 --width 512 --height 512
node tools/reference/pathtracer/run.ts   --scene cornell-box --camera 1 --samples 8192 --width 512 --height 512
node tools/compare/index.ts bench/references/cornell-box/front/cycles-full.exr bench/references/cornell-box/front/pathtracer-full.exr
node tools/reference/cycles/run.ts --calibrate      # known-answer checks of the unit conversions
```

Output goes to `bench/references/<scene>/<camera name>/` (gitignored):

| file | what |
| --- | --- |
| `cycles-full.exr`, `cycles-indirect.exr` | Blender Cycles, 32-bit linear float, no denoising |
| `pathtracer-full.exr`, `pathtracer-indirect.exr` | three-gpu-pathtracer (WebGPU), same |
| `*.json` | every setting, sample count, adapter and time |
| `compare-*.png` | heatmaps from `tools/compare` |

`bench/references/render-log.csv` gets one row per render (sample count, size, seconds).

Cameras are the manifest bookmarks: `--camera 1` is the first, or give the name. Both renderers build the camera from the manifest position, target and vertical fov, with the same aspect and size.

## What "full" and "indirect-only" mean

- **full**: everything the camera sees: emission, direct light, all bounces.
- **indirect-only**: light that reached the visible surface after at least one earlier bounce. Emission seen directly, and light that gets to the visible surface straight from a source, are "direct".
  - Cycles: the Diffuse Indirect pass times Diffuse Colour. Materials are purely diffuse, so full = emission + environment + direct + indirect.
  - Path tracer: full minus a direct image. The library counts the camera hit as bounce 1 and samples point, directional and spot lights explicitly at every hit, but finds emissive surfaces only by following a scattered ray. So a single bounce limit means different things for the two kinds of source. The direct image is therefore two runs added together (light is additive): lights only (emission switched off, `maxBounces` 1) and emission only (lights switched off, `maxBounces` 2). The same random numbers are used in every run, which keeps the difference quiet.
  - Checked against known answers and against each other: furnace room (albedo 0.5, emission 0.25) gives full 0.5, indirect 0.125 in both renderers; two-room (a point light) agrees on indirect to 0.1 percent; Cornell (emissive quad) agrees to 0.3 percent. Using the emission setting (2 bounces) on a point-light scene drops the indirect by 36 percent, which is how the split was found.

## Settings, and why

Both sides, matched on purpose:

| | Cycles | Path tracer |
| --- | --- | --- |
| Materials | diffuse only (Principled: metallic 0, specular 0, roughness 1) | `MeshPhysicalMaterial`, roughness 1, metalness 0, `specularIntensity` 0 |
| Bounces | 32 | 33 (the library counts the camera hit as the first) |
| Clamping | off (Blender's default indirect clamp of 10 is a bias) | off (the library's default indirect clamp is 10) |
| Denoising | off | off (see "Denoiser") |
| Pixel filter | box, 1 px | jitter inside the pixel |
| Environment | none = black | none = black; other types unsupported |
| Direct light range | not applied | three's range window is applied (see below) |

Cycles also: adaptive sampling off; light sampling threshold 0 (the default 1 percent threshold drops dim lights); Standard view transform, exposure 0, gamma 1; glossy, transmission and volume bounces 0; Blender 4.5 pinned; runs with `--factory-startup` so no user add-on can change a render (and Blender exits quickly: a user add-on made it hang for minutes at exit).

## Unit conversions (manifest, three.js physical units, to Cycles)

All in `tools/reference/cycles/units.py`, each with its reason. Pinned by measurement in Blender 4.5.0 and checked by `--calibrate` (point, sun, spot, furnace; all within 0.05 percent):

| manifest | Cycles | note |
| --- | --- | --- |
| directional lux E | sun strength = E | strength is irradiance in W/m2 at normal incidence |
| point/spot candela I | light power P = 4 pi I watts | Blender's point light has I = P / (4 pi); a spot has the same I on its axis (the cone cuts, it does not focus) |
| spot cones | `spot_size` = 2 x outer; `spot_blend` = (cos inner - cos outer) / (1 - cos outer) | makes Cycles' spot falloff the same smoothstep as three's |
| emissive radiance L | emission strength = L (colour L/peak) | no pi: an emission shader's strength is radiance. The glTF importer already does this |
| environment colour x intensity | Background colour x strength | radiance |
| positions, directions | (x, y, z) to (x, -z, y) | a proper rotation, the same one the glTF importer applies to the meshes |
| model transform Euler XYZ | R = Rx Ry Rz | three's XYZ order is the reverse of Blender's |

**Range window.** three.js and the path tracer fade a point light to zero at `range` (`(1 - (d/range)^4)^2`). Cycles has no such window and the reference is the plain inverse-square answer. On two-room (range 14) that makes three about 12 percent dimmer than Cycles at 7 m from the lamp and 3 percent at 5 m. This is a difference to remember when comparing DDGI with a Cycles reference; Cornell and Sponza are unaffected (emissive geometry, directional light).

**Procedural sky** (Sponza only): Cycles' Nishita sky with the sun disc off, since the directional light is the sun. It stands in for the viewer's sky; it is not the same model. `turbidity` is not mapped.

## Compare

`node tools/compare/index.ts reference.exr test.exr` (first image is the reference). `--help` has the definitions. It prints:

- **FLIP**, HDR mode, mean (0 identical). Tone mapper ACES, exposure range chosen by FLIP from the reference and printed, 67 pixels per degree. FLIP comes from the `flip-evaluator` Python package: there is no JavaScript port, and re-implementing a published metric is the bigger risk. The rest of the tool is Node (EXR reading and writing, PNG, metrics), so it also runs without Python with `--no-flip`. It runs `python -E`, from a temporary folder holding only the images, rather than `python -I`: `-I` hides the per-user site-packages where `flip-evaluator` lives here.
- **mean relative error** = sum |test - ref| / sum ref. 10 percent too bright everywhere gives 0.100. **bias** = sum (test - ref) / sum ref (signed). **RMS** and relative RMS.
- three PNGs: `-heatmap.png` (per-pixel relative luminance error, magma, saturating at 50 percent), `-flip.png` (FLIP's own map) and `-side-by-side.png` (reference, test, FLIP).
- **p95 error**: the 95th percentile of the per-pixel relative luminance error, `|T - R| / (R + 5% of the mean R)`. With `--downsample 8` each pixel is an 8 x 8 block, so it is the 95th-percentile block error: the worst few percent of the picture, which the averages hide.
- A second line with every value limited to 1 first (`--also-clamp x` changes it), so a bright lamp (17 in Cornell) does not dominate the sums. `--clamp x` applies a limit to the main numbers instead. Clamping happens before block averaging.
- `--downsample 8` averages 8 x 8 blocks of both images first. A path-traced image is noisy; this takes most of the noise out so lighting differences show. FLIP gets the block-averaged images with its pixels per degree divided by 8 (67 becomes 8.4): each pixel now spans 8 times the viewing angle, so FLIP judges the same view. A FLIP number is only comparable with another taken at the same downsample.

## Agreement tolerance: Cycles vs path tracer

Proposed tolerance for "the two references agree" (both near converged: Cycles at least 2048 samples, path tracer at least 8192):

| image | bias (signed, overall) | mean relative error after 8 x 8 block averaging |
| --- | --- | --- |
| full | within 0.5 percent | at most 1 percent |
| indirect-only | within 1 percent | at most 3 percent |

Measured (`node tools/compare/index.ts cycles-X.exr pathtracer-X.exr [--downsample 8]`; the path tracer's light range ignored on two-room with `--ignore-light-range`, because Cycles has no range window). Mean relative error is sum|T-R| / sum R; "clamped" limits values to 1 first; p95 is the 95th-percentile relative error per pixel or block.

Per pixel (noise dominated):

| scene (bookmark 1) | image | bias | mean rel. error | clamped | p95 | FLIP (HDR, 67 ppd) |
| --- | --- | --- | --- | --- | --- | --- |
| Cornell "front" | full | -0.002% | 2.93% | 5.67% | 18.2% | 0.074 |
| Cornell | indirect | -0.014% | 11.3% | 11.3% | 29.8% | 0.101 |
| two-room "lit-room" | full | +0.11% | 1.94% | | | |
| two-room | indirect | +0.12% | 2.72% | | | |

After 8 x 8 block averaging (what the tolerance is on):

| scene | image | bias | mean rel. error | clamped | p95 block error | FLIP (HDR, 8.4 ppd) |
| --- | --- | --- | --- | --- | --- | --- |
| Cornell "front" | full | -0.002% | 0.37% | 0.71% | 2.30% (clamped 2.63%) | 0.041 |
| Cornell | indirect | -0.014% | 1.43% | 1.43% | 3.79% | 0.056 |
| two-room "lit-room" | full | +0.11% | 0.30% | | | |
| two-room | indirect | +0.12% | 0.39% | | | |

Clamping raises the Cornell full error from 0.37% to 0.71% because the lamp (about 17 against a mean of 0.12) adds a large, well matched amount to the sum that the ratio is divided by. It still meets the tolerance. Indirect has no values above 1, so clamping changes nothing.

The per-pixel numbers are Monte Carlo noise, not disagreement: Cycles against another Cycles seed (2048 samples each) already differs by 0.81% per pixel on Cornell and 2.4% on two-room, and the path tracer, which has no light sampling for emissive meshes, is noisier still. That is why the tolerance is on the bias and on block-averaged error, which noise barely touches. A per-pixel number is only meaningful against a reference that is much less noisy than the image under test.

Noise of one image, as relative RMS (half the difference between two seeds, over the mean, divided by the square root of 2): Cornell 2048 samples 1.1%; two-room 2048 samples 2.3%; Sponza 1024 samples 12.7% full and 38% indirect, so the 4096-sample reference has about 6.3% and 19%.

## Sizes, samples and times (this laptop, Blender on the CPU, path tracer on the RTX 3060 Laptop)

| reference | size | samples | seconds |
| --- | --- | --- | --- |
| Cycles Cornell "front" | 512 x 512 | 2048 | 336 (second seed 328) |
| Cycles two-room "lit-room" | 640 x 360 | 2048 | 442 (second seed 807, other renders running) |
| Cycles Sponza "colonnade-from-the-floor" | 640 x 360 | 4096 | 1658 (other renders running; 1024 samples took 318 uncontended) |
| path tracer Cornell | 512 x 512 | 8192 | 139 full + 86 direct |
| path tracer two-room | 640 x 360 | 8192 | 236 full + 52 direct |

All in `bench/references/render-log.csv`.

**Sponza is mostly shaded at this sun angle.** A ray test from 187 floor points toward the manifest sun finds only 3 unshadowed: the colonnades and galleries block a sun at 53 degrees. So the Sponza reference is lit mostly by the sky (mean pixel 0.018) and the bookmark looks dark at exposure 0; the viewing PNG is +2 stops. Cycles is right; the manifest's sun angle (or the sky, which stands in for the viewer's) is what to revisit.

## Denoiser

`@pmndrs/denoiser` is not on npm (the npm name `denoiser` is a different, old package). It is built from source in a gitignored folder, once (about 5 minutes, needs network for the clone and the yarn install):

```
git clone --depth 1 https://github.com/pmndrs/denoiser.git node_modules/.cache/pmndrs-denoiser
cd node_modules/.cache/pmndrs-denoiser
corepack yarn workspaces focus denoiser
corepack yarn build:lib
```

(Tested at commit 4a5a7bf, 2026-10-28.) Then `node tools/reference/pathtracer/run.ts ... --denoise` also writes `pathtracer-full-denoised.exr`. It runs the library's WGSL runtime with the OIDN `rt_hdr_small` network (colour only, no albedo or normal guides) on the page's own GPU device, and loads the weights from the clone, not a CDN. Nothing is vendored into this repo.

The raw `pathtracer-full.exr` stays undenoised, so it can still be used as ground truth. The denoised file is for looking at, and for cheap low-sample references. It does bias: on Cornell at 256 samples the denoised image sits 0.5 percent bright.

Only a single pass at the end is wired up, not the progressive (every N samples) mode.

## Sample counts, GPU and CPU

- Both tools hold the shared bench lock (`$TEMP/ddgi-gpu-bench.lock`) for the whole render: one machine-heavy job at a time. Cycles on the CPU (16 threads on this laptop) would skew other agents' build-time and CPU numbers, and the path tracer loads the GPU. A tool waits while someone else holds the lock and releases it when it finishes, fails, or is stopped with Ctrl-C or SIGTERM (the release is on the tools/gpu cleanup list, which also kills Blender). Chrome is closed after the lock is released, since closing can take minutes on a busy machine.
- Cycles uses the CPU by default; `--device gpu` uses OptiX (still under the lock).
- Noise: render a second seed (`--seed 1 --name-suffix=-seed1`, note the `=`) and compare the two; half the relative RMS is the noise of one image.
