# References and compare (T0.7)

Ground truth on demand, and one way to compare images.

```
node tools/reference/cycles/run.ts       --scene cornell-box --camera 1 --samples 4096 [--denoise]     # 1920 x 1080, OptiX
node tools/reference/pathtracer/run.ts   --scene cornell-box --camera 1 --samples 8192 --width 512 --height 512
node tools/compare/index.ts bench/references/cornell-box/front/cycles-full.exr bench/references/cornell-box/front/pathtracer-full.exr
node tools/reference/cycles/run.ts --calibrate      # known-answer checks of the unit conversions
```

Output goes to `bench/references/<scene>/<camera name>/` (gitignored):

| file | what |
| --- | --- |
| `cycles-full.exr`, `cycles-indirect.exr` | Blender Cycles, 32-bit linear float, no denoising |
| `cycles-direct.exr`, `cycles-lights-direct.exr`, `cycles-gi.exr` | the GI composite and its parts: what DDGI is compared with (see "The GI composite") |
| `cycles-untextured-*` | the same from an untextured render (see "Untextured references") |
| `pathtracer-full.exr`, `pathtracer-indirect.exr` | three-gpu-pathtracer (WebGPU), same |
| `cycles-full-denoised.exr` | with `--denoise`: a denoised copy of the full image, for viewing only (see "HD references") |
| `*.json` | every setting, sample count, adapter and time, and the stored shot (see "Stored camera") |
| `compare-*.png` | heatmaps from `tools/compare` |

`bench/references/render-log.csv` gets one row per render (sample count, size, seconds).

Cameras are the manifest bookmarks: `--camera 1` is the first, or give the name. Both renderers build the camera from the manifest position, target and vertical fov, with the same aspect and size. `--from-shot <cycles.json>` renders an earlier reference's stored camera and size instead.

## HD references (T1.13, 2026-11-05)

Dennis: "640 x 360 is an insane size to judge ANYTHING against." Since T1.13 every reference is:

- **1920 × 1080** by default. 16:9 was chosen over the viewer's window shape because the window changes; the viewer instead letterboxes to the reference's stored aspect. It is also two-room's and Sponza's earlier 640 × 360 shape, so their means stay comparable (the `ddgi-leak` page's constants, for one). Cornell's earlier references were 512 × 512; at 16:9 the frame shows more of the black surroundings, which changes the image mean but not the bias (sum against sum on the same view).
- **Anti-aliased:** Cycles' Blackman-Harris pixel filter, 1.5 px wide (its default). It was a 1 px box before. Cycles importance-samples the filter, so an image's mean doesn't change; edges are softer by about half a pixel. What it means for comparisons: DDGI's test view shoots one ray through each pixel centre (no anti-aliasing), so per-pixel numbers and per-pixel FLIP now include an edge-softness difference along every silhouette. The 8 × 8 block numbers, which the tolerances use, barely move (an edge's light stays in its block), and the bias doesn't move at all. `--pixel-filter box --filter-width 1` renders the old way.
- **On the GPU:** `--device gpu` (OptiX on the RTX 3060 Laptop) is now the default; `--device cpu` still works. OptiX and CPU references agree to about 0.1% (DECISIONS, the sky-aware composite review). The json records the device by name.
- **Clean:** the sample counts in "Sizes, samples and times" below, with the 8 × 8-block noise measured from two seeds.
- **`--denoise`** (Dennis is fine with denoising tools) also writes `cycles-full-denoised.exr`: the full image through OpenImageDenoise (Blender's compositor Denoise node, HDR, with albedo and normal guides). It is for the viewer's wipe only. The raw EXRs are written unchanged and are what every accuracy number is measured on. `run.ts` measures the denoised copy against the raw image and stores the result (`denoiseCheck`: bias and 8 × 8 block error) in the json and in `shot.render.denoised.check`. The viewer shows the denoised copy when there is one, and says so.

## Stored camera (`shot`)

Every render writes, in its json (`cycles.json`, `cycles-untextured.json`, ...), a `shot` with everything needed to see exactly that view again (format `ddgi-reference-shot/1`):

| field | what |
| --- | --- |
| `camera` | `bookmark` name and number; `position`, `target`, `up` (+Y) in manifest space (three.js axes, metres); `fovYDeg` (vertical); `aspect`, `width`, `height`; `near` 0.01 and `far` 2000 (m) |
| `display` | `exposureEv` (default the manifest's `exposureEv`) and `toneMapping` (default `aces`): what the viewer shows it with. The EXRs are linear and never tone-mapped |
| `scene` | the manifest id, path and sha256 of its file; the sky file's sha256; the GI material table's sha256 (untextured renders) |
| `render` | `git` (commit, and whether the tree had uncommitted changes), Blender version, `device` (e.g. `OPTIX: NVIDIA GeForce RTX 3060 Laptop GPU`), samples, seed, pixel filter, render seconds, and `denoised` (null, or the viewing copy's file, denoiser and its check) |

Who reads it:

- **The viewer's Compare** (`apps/viewer/src/scene/reference-files.ts`, `viewer-app.ts`): moves the camera to the stored position, target and field of view, letterboxes the picture to the stored aspect, and (on pressing Compare) sets the stored exposure and tone mapping, so the wipe lines up pixel for pixel. The panel says when the scene's manifest has changed since the render (a warning: re-render), and when the render is from another commit than the viewer runs. The viewer keeps its own near and far planes: they only clip, and 0.01 to 2000 m would cost depth precision.
- **The DDGI test images:** `tools/gpu/pages/ddgi-test-view/render.ts` and `ddgi-sponza-accuracy/run.ts` render DDGI from the stored camera at the stored size (`tools/reference/shot.ts`), and warn if the manifest has changed since.
- **Blender:** `run.ts --scene <id> --from-shot <json> --samples N` renders the same view again.

## What "full" and "indirect-only" mean

- **full**: everything the camera sees: emission, direct light, all bounces.
- **indirect-only**: light that reached the visible surface after at least one earlier bounce. Emission seen directly, and light that gets to the visible surface straight from a source, are "direct".
  - Cycles: the Diffuse Indirect pass times Diffuse Colour. Materials are purely diffuse, so full = emission + environment + direct + indirect.
  - Path tracer: full minus a direct image. The library counts the camera hit as bounce 1 and samples point, directional and spot lights explicitly at every hit, but finds emissive surfaces only by following a scattered ray. So a single bounce limit means different things for the two kinds of source. The direct image is therefore two runs added together (light is additive): lights only (emission switched off, `maxBounces` 1) and emission only (lights switched off, `maxBounces` 2). The same random numbers are used in every run, which keeps the difference quiet.
  - Checked against known answers and against each other: furnace room (albedo 0.5, emission 0.25) gives full 0.5, indirect 0.125 in both renderers; two-room (a point light) agrees on indirect to 0.1 percent; Cornell (emissive quad) agrees to 0.3 percent. Using the emission setting (2 bounces) on a point-light scene drops the indirect by 36 percent, which is how the split was found.

## The GI composite: what DDGI is compared with

`cycles-gi.exr` = `cycles-indirect.exr` + `cycles-direct.exr` − `cycles-lights-direct.exr`. Every Cycles render writes it (`render.py` renders the parts, `run.ts` adds them up), and every DDGI accuracy number is taken against it, for Cornell, two-room and Sponza alike (`node tools/gpu/pages/ddgi-sponza-accuracy/run.ts --table`).

**Why.** DDGI's field holds every light a probe ray brings back: bounces, and also the sky and emitters seen straight. Only the manifest's analytic lights (sun, point, spot) are drawn by the renderer, outside the field. Cycles' indirect pass leaves out the sky and emitters seen straight from the surface (it counts them as direct), so on Sponza's open courtyard it holds about half of what DDGI's field holds. So the reference is everything diffuse except the analytic lights' direct light. The decision and its history are in DECISIONS ("What 'indirect light' means against Cycles in a scene with a sky", set 2026-11-05 at T1.9).

**The parts** (both × Diffuse Colour, so they are radiance like the indirect pass):

| file | what |
| --- | --- |
| `cycles-direct.exr` | Diffuse Direct from the normal render: the lights, the sky and emitters, reaching the visible surface without a bounce |
| `cycles-lights-direct.exr` | Diffuse Direct from a second render with only the manifest lights on and no bounces, same seed and samples (so pixel jitter matches and edges cancel) |

**The lights-only render really is lights-only.** `switch_off_emitters` (render.py) replaces the world with a black one (no nodes, colour 0), whatever it held. In every material and shader node group it unlinks and zeroes every emitting input: Principled surface emission, the Emission shader (on a surface or in a volume), a Background shader used in a material, and Principled Volume emission and blackbody. Then `find_emitters` must find nothing left, or the run stops with the list. A shader node of a kind it doesn't know that has an emission input also stops it. Light objects in the glTF files are deleted on import, so the only light objects are the manifest's.

**Checks, on every render.** Before the normal render, `render.py` records what else emits (`otherEmitters` in `cycles.json`) and what the lights-only image must therefore be (`giComposite.lightsOnlyExpected`). `run.ts` checks it (`gi-composite.ts`). If the check fails, it stops with exit code 1 and writes no `cycles-gi.exr`.

- `equals-direct`: the manifest lights are the only source (no sky, no emitter: point-light Cornell, two-room). The lights-only direct image must equal the normal direct image: bias within ±0.5% and 8 × 8 block error at most 2%. This catches a lights-only render that lost or changed the lights' own light. In these scenes the composite equals the indirect pass.
- `black`: the scene has no manifest lights (emissive Cornell). The lights-only image must be 0 (at most 0.01% of the direct light). This catches an emitter the switch-off missed.
- not checked: both kinds of source (Sponza). There is no known image to compare with; the calibration case below covers the switch-off.

`node tools/reference/cycles/run.ts --calibrate` has a known-answer case, "lights only": a white floor under a point light, with a coloured world and four emitting quads around it (an Emission shader, one inside a node group, a Background shader as a material, Principled emission). `find_emitters` must list 5 sources, and the lights-only image must show the point light's E / π alone: 3.97888 against 3.97887. With everything on, the floor reads 5.895, so a missed source would show. (The switch-off before 2026-11-05 zeroed only Principled emission and the world's Background node; it would have left three of the four quads on.)

Cheap end-to-end runs of the check (seconds each; they take the bench lock and add rows to `render-log.csv`):

```
node tools/reference/cycles/run.ts --scene cornell-box-point --samples 64 --width 128 --height 128 --out-dir <scratch folder>   # equals-direct
node tools/reference/cycles/run.ts --scene cornell-box --samples 64 --width 128 --height 128 --out-dir <scratch folder>         # black
node tools/reference/cycles/run.ts --calibrate                                                                                  # the switch-off, known answers
npx vitest run tools/reference/cycles                                                                                           # the image check itself
```

Measured 2026-11-05 (CPU, 2048 samples, seed 0, the references' sizes): on point-light Cornell (512 × 512) and two-room `lit-room` (640 × 360) the lights-only direct image equals the normal direct image exactly (bias 0.000%, block error 0.000%: same seed, same first-hit samples), so `cycles-gi.exr` equals `cycles-indirect.exr` there, and the re-rendered `cycles-full`/`cycles-indirect` are pixel-identical to the earlier ones. The lights-only render costs 116 s (Cornell) and 110 s (two-room) on top of 211 s and 340 s.

## Untextured references (`--materials-json`)

```
node tools/reference/cycles/run.ts --scene sponza --camera 1 --width 640 --height 360 --samples 8192 --device gpu \
     --materials-json bench/references/sponza/colonnade-from-the-floor/gi-materials.json
```

renders the scene with the materials DDGI actually sees, and names the files `cycles-untextured-*` (`full`, `indirect`, `direct`, `lights-direct`, `gi`, and `cycles-untextured.json`).

**What it replaces.** In M1 the platform's GI sees one constant colour per material: the three adapter folds each texture's average into the colour, and the core samples no textures until M4 (DECISIONS 2026-11-05, T1.7). A textured Cycles render would judge that simplification together with DDGI. The GI material table (`gi-materials.json`, written by `tools/gpu/pages/ddgi-sponza-accuracy/run.ts` from `SceneSync.sentMaterial`) lists, per model part and glTF material name, the albedo and emission the adapter sent to the platform. `apply_gi_material_table` puts them on every Principled BSDF, replacing the textured inputs:

| input | set to | why |
| --- | --- | --- |
| Base Color | the table's albedo; texture, vertex colour and factor nodes cut off | the GI albedo is one constant per material |
| Emission Color, Strength | emission / its peak, strength = the peak | strength is radiance (units.py) |
| Alpha | 1 | the M1 tracer treats every surface as opaque (alpha cutouts and see-through materials arrive with M4). So `glass` is an opaque surface on both sides, as in DDGI: consistent, not physical |
| Normal | cut off (normal maps removed) | the platform shades with interpolated vertex normals |

Metallic, Roughness, Transmission, Specular and the other lobes are forced to the diffuse values as for every reference (`make_materials_diffuse_only`; "Diffuse only, checked" below). `require_untextured` then refuses to render if any input in the table above is still linked.

**Matching.** Materials are matched by model part and glTF name (Blender's `.001` suffixes for repeated names are stripped). A material used in the scene with no row, or a row with no material, stops the run. Only the model parts the table lists (`parts`) are imported, so a table made from the base package renders the base package.

**Left-out faces.** The three adapter leaves a mesh out of GI when all its materials let light through (alpha-blended glTF materials such as Sponza's `dirt_decal`, `blocksLight: false`), because the M1 tracer can't honour that per triangle. The table counts those meshes per material (`leftOutMeshes`), and `remove_left_out_faces` deletes the faces of every such material (Sponza: `dirt_decal`, 8,140 faces; recorded as `leftOutOfGiFaces` in the json). three makes one mesh per glTF primitive, with one material each, so a material's faces are exactly those meshes; a material with only some of its meshes left out stops the run.

**The table hash.** `cycles-untextured.json` records the table's path and sha256 (`giMaterials`). `ddgi-sponza-accuracy/run.ts` hashes the table it has just written and says whether the reference was rendered from it, or warns that the reference must be rendered again (a changed adapter can send different colours).

## Settings, and why

Both sides, matched on purpose:

| | Cycles | Path tracer |
| --- | --- | --- |
| Materials | diffuse only (Principled: metallic 0, specular 0, roughness 1; texture links into those inputs removed, then checked) | `MeshPhysicalMaterial`, roughness 1, metalness 0, `specularIntensity` 0 |
| Kept from the source material | base colour and its texture, alpha, emission, normal map | the same (`map`, `alphaMap`, `alphaTest`, `transparent`, `opacity`, `emissive*`, `normalMap`, vertex colours, `side`) |
| Bounces | 32 | 33 (the library counts the camera hit as the first) |
| Clamping | off (Blender's default indirect clamp of 10 is a bias) | off (the library's default indirect clamp is 10) |
| Denoising | off | off (see "Denoiser") |
| Pixel filter | Blackman-Harris, 1.5 px (box, 1 px before T1.13; see "HD references") | jitter inside the pixel |
| Environment | none = black; procedural = the shared sky file `scenes/sky/<scene>.exr` | none = black; other types unsupported |
| Direct light range | not applied | three's range window is applied (see below) |

**Diffuse only, checked.** Blender's glTF importer links Metallic and Roughness to the metallicRoughness texture (24 of the 28 New Sponza materials), and a linked input ignores the value set on it. So `make_materials_diffuse_only` unlinks every input in `NEUTRAL_PRINCIPLED_INPUTS` (Metallic, Roughness, Diffuse Roughness, Subsurface, Specular IOR Level and Tint, Anisotropic, Transmission, Coat, Sheen, Thin Film) and then sets it. `render.py` then refuses to render, listing each culprit, if any Principled BSDF still has one of those inputs linked or not neutral, or if a glossy-type shader node is present. On Sponza it removes 76 texture links. Sponza references rendered before 2026-11-04 16:00 had metal texels with no diffuse and a glossy sun highlight; the re-render is 2.3 percent brighter overall and 17.9 percent brighter in indirect light (mean 0.0341 to 0.0349, indirect 0.00420 to 0.00495), because glossy bounces are switched off, so those metal texels absorbed bounce light instead of passing it on. Cornell and two-room have no texture links and are unaffected.

Cycles also: adaptive sampling off; light sampling threshold 0 (the default 1 percent threshold drops dim lights); Standard view transform, exposure 0, gamma 1; glossy, transmission and volume bounces 0; Blender 4.5 pinned; runs with `--factory-startup` so no user add-on can change a render (and Blender exits quickly: a user add-on made it hang for minutes at exit).

## Unit conversions (manifest, three.js physical units, to Cycles)

All in `tools/reference/cycles/units.py`, each with its reason. Pinned by measurement in Blender 4.5.0 and checked by `--calibrate` (point, sun, spot, furnace; all within 0.05 percent; it also checks the shared sky's orientation and its light on a floor):

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

**Procedural sky:** the shared sky file `scenes/sky/<scene>.exr` (T1.1, `tools/sky/README.md`) as an Environment Texture, the same file the viewer and the platform read: no sun disc (the directional light is the sun), black below the horizon, brightness calibrated. Until 2026-11-04 it was Cycles' Nishita sky, which was brighter: references with a procedural sky rendered before then (Sponza) must be rendered again. The before and after means are in PROGRESS (T1.1).

## Compare

`node tools/compare/index.ts reference.exr test.exr` (first image is the reference). `--help` has the definitions. It prints:

- **FLIP**, HDR mode, mean (0 identical). Tone mapper ACES, exposure range chosen by FLIP from the reference and printed, 67 pixels per degree. FLIP comes from the `flip-evaluator` Python package: there is no JavaScript port, and re-implementing a published metric is the bigger risk. The rest of the tool is Node (EXR reading and writing, PNG, metrics), so it also runs without Python with `--no-flip`. It runs `python -E`, from a temporary folder holding only the images, rather than `python -I`: `-I` hides the per-user site-packages where `flip-evaluator` lives here.
- **mean relative error** = sum |test - ref| / sum ref. 10 percent too bright everywhere gives 0.100. **bias** = sum (test - ref) / sum ref (signed). **RMS** and relative RMS.
- three PNGs: `-heatmap.png` (per-pixel relative luminance error, magma, saturating at 50 percent), `-flip.png` (FLIP's own map) and `-side-by-side.png` (reference, test, FLIP).
- **NaN and infinite pixels**: a pixel that is NaN or infinite in either image is left out of both images, before block averaging, and the true count is printed (reference, test, either). A block averages its finite pixels only, and a block with none is left out. FLIP cannot skip a pixel, so it gets those pixels as 0 in both images, and the output says how many. `--clamp` leaves infinities alone (it does not turn +Inf into the limit).
- **p95 error**: the 95th percentile of the per-pixel relative luminance error, `|T - R| / (R + 5% of the mean R)`. With `--downsample 8` each pixel is an 8 x 8 block, so it is the 95th-percentile block error: the worst few percent of the picture, which the averages hide.
- A second line with every value limited to 1 first (`--also-clamp x` changes it), so a bright lamp (17 in Cornell) does not dominate the sums. `--clamp x` applies a limit to the main numbers instead. Clamping happens before block averaging.
- `--downsample 8` averages 8 x 8 blocks of both images first. A path-traced image is noisy; this takes most of the noise out so lighting differences show. FLIP gets the block-averaged images with its pixels per degree divided by 8 (67 becomes 8.4): each pixel now spans 8 times the viewing angle, so FLIP judges the same view. `--ppd` always describes the full-size image, so a value you give is divided too (`--ppd 40 --downsample 8` gives FLIP 5). A FLIP number is only comparable with another taken at the same downsample.

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
| Cycles Sponza "colonnade-from-the-floor" | 640 x 360 | 4096 | 997 (diffuse-only fix, raised sun; the first render took 1658 with other renders running) |
| path tracer Cornell | 512 x 512 | 8192 | 139 full + 86 direct |
| path tracer two-room | 640 x 360 | 8192 | 236 full + 52 direct |

All in `bench/references/render-log.csv`.

**Sponza's sun.** At the manifest's first sun angle (53 degrees) a ray test from 187 floor points found only 3 unshadowed, so the reference was lit mostly by the sky (mean pixel 0.018). The sun was raised in the manifest; the floor band is now lit and the mean is 0.0349.

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

- Both tools hold the shared bench lock (`$TEMP/ddgi-gpu-bench.lock`, see "Bench lock" below) for the whole render: one machine-heavy job at a time. Cycles on the CPU (16 threads on this laptop) would skew other agents' build-time and CPU numbers, and the path tracer loads the GPU. A tool waits while someone else holds the lock and releases it when it finishes, fails, or is stopped with Ctrl-C or SIGTERM (the release is on the tools/gpu cleanup list, which also kills Blender). Chrome is closed after the lock is released, since closing can take minutes on a busy machine.
- Cycles uses the GPU (OptiX) by default since T1.13; `--device cpu` uses the CPU (both under the lock).
- Noise: render a second seed (`--seed 1 --name-suffix=-seed1`, note the `=`) and compare the two; half the relative RMS is the noise of one image.

## Bench lock

One machine-heavy job at a time on this machine: GPU timing, native harness runs, Cycles renders, path-tracer renders. The lock is the folder `$TEMP/ddgi-gpu-bench.lock`; `tools/reference/bench-lock.ts` implements it, for Node code (`withBenchLock`) and as a command line for everyone else.

```
node tools/reference/bench-lock.ts status                               # who holds it; exit 0 free, 3 held, 4 held but stale
node tools/reference/bench-lock.ts run --what "trace bench" -- pnpm test:gpu --filter trace-bench
node tools/reference/bench-lock.ts acquire --pid $PID --what "native cache build"   # PowerShell: $PID is the script itself
node tools/reference/bench-lock.ts release --pid $PID
```

`run` is the easy one for agents: it holds the lock exactly as long as the command runs, and releases it on failure and Ctrl-C too. `acquire` needs `--pid`, the long-lived process that holds the lock (the command itself exits at once, so its own pid would look dead straight away). `--timeout-minutes n` gives up after n minutes (exit code 5).

**Commands that take the lock themselves can run under `run`.** `run` starts its command with the environment variable `DDGI_BENCH_LOCK_HELD=<pid of the lock owner>`. `withBenchLock` (Cycles, path tracer) and the `tools/native` scripts check it: if that pid owns the lock right now (owner file, this host, process running), they work under the parent's hold and neither take nor release the lock. Without this they would wait forever for a lock their own parent holds. A variable that does not match the current owner is ignored and the lock is taken normally, so a leftover value never skips the lock. `status --json` reports the match as `heldByParent` (the pid, or `null`).

The protocol, for anything that cannot call the tool:

1. **Take:** `mkdir` the folder. It fails if the folder exists; that is the atomic step.
2. **Claim:** straight away write `owner.json` inside it (to a temporary name, then rename): `pid`, `processStart` (the OS start time of that pid, so a reused pid is recognised; on Windows `(Get-Process -Id <pid>).StartTime.ToUniversalTime().ToString('o')`), `host`, `startedAt` (ISO), `what` (a short label) and `command`.
3. **Wait:** while it exists, poll (every second: some holders release and re-take it between runs) and say once a minute who holds it.
4. **Stale:** take it over only if the owner is provably gone: its pid is not running, or the pid now has a different start time. Never remove a lock whose owner is alive, however long it has been held. A folder with no `owner.json` counts as abandoned only after 3 hours. Every current user writes an owner file (`tools/native` goes through this CLI too), so an ownerless folder only comes from an old bare `mkdir`; the limit can come down to a few minutes (`--ownerless-stale-minutes`) once nobody uses that. A takeover renames the stale folder aside first and checks it is the one judged stale, so two waiters cannot both take it.
5. **Release:** remove the folder, but only if `owner.json` is yours. `release` refuses (exit code 2) otherwise.
