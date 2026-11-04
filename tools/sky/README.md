# tools/sky: the shared sky (T1.1)

One sky definition becomes one HDR file, and every renderer reads that file: the viewer (three.js), Cycles, the platform, and (as a flat colour) the native harness.

```
node tools/sky/generate.ts                  # write scenes/sky/<id>.exr and .json for every scene with a procedural sky
node tools/sky/generate.ts --scene sponza   # one scene
node tools/sky/generate.ts --check          # compare the committed files with a fresh generation (the unit tests do this too)
node tools/sky/viewer-render.ts --scene open-ground --camera 1 [--material diffuse|standard]   # viewer render lit by the sky, to bench/references/
pnpm test:gpu --filter sky-orientation      # the platform loader and the viewer read known directions the same as the file
node tools/reference/cycles/run.ts --calibrate   # includes the Cycles sky orientation and sky-on-a-floor checks
```

Rerun `generate.ts` after changing a manifest's `environment`, its sun light, or the code in `scenes/src/sky/`. `--check` fails until you do.

## What is in a sky file

`scenes/sky/<scene id>.exr`, made from the manifest's `environment` (`type: "procedural"`):

- **Model:** Preetham, Shirley and Smits, "A Practical Analytic Model for Daylight" (SIGGRAPH 1999), implemented from the paper (`scenes/src/sky/preetham.ts`), for the manifest's `sunDirection` and `turbidity` (default 2), converted from CIE xyY to linear Rec.709 RGB.
- **No sun disc.** The sun is the manifest's directional light, which every renderer already has. Putting it in the sky too would count it twice.
- **Black below the horizon.** The ground is scene geometry, so every renderer gets ground light from the same place.
- **Layout:** equirectangular, 512 x 256, +Y up, row 0 is the zenith, `u = atan2(z, x) / 2pi + 0.5` (u = 0.5 is +X, 0.75 is +Z), the same as three.js `equirectUV`. In Blender, after the manifest-to-Blender axis change (x, y, z) to (x, -z, y), Cycles' own equirectangular lookup lands on the same texels, so no rotation is applied (`blender_scene.py`).
- **Format: OpenEXR, half-float B/G/R, ZIP.** About 50 kB each.
  - *Why not Radiance .hdr:* its readers disagree. three's HDRLoader decodes a mantissa m as m/255, stb_image as m/256, Blender as (m + 0.5)/256: up to 0.4% apart on the same file, which is most of the ±0.5% reference tolerance.
  - A half float means the same in every reader. three (EXRLoader), Blender and `tools/compare` read it, and it goes straight into an `rgba16float` GPU texture.
- **Committed, not generated on demand:** the files are small, and committing them means the viewer, Blender and the native scripts need no generation step. `--check` and the unit tests catch a file that no longer matches its manifest or the code. They compare decoded pixels, not bytes, because zlib's output bytes can differ between builds.

`scenes/sky/<scene id>.json` is the sidecar: how it was made, the calibration numbers, the irradiance measured from the stored half floats, the flat-colour equivalent, and a SHA-256 of the pixels.

## Brightness: the calibration

The sky is scaled so that a flat, open, upward-facing ground gets this much sky irradiance (luminance-weighted, Rec.709 Y):

```
E_sky = ratio(sun elevation, turbidity) x E_sun x sin(elevation) x environment.intensity
```

- `E_sun` is the sun's illuminance at normal incidence: the manifest's directional light that travels along `-sunDirection`, as `intensity` x the luminance of its `color`. A sky-only scene with no such light gives `environment.sunIlluminance` instead. Exactly one of the two must exist.
- `ratio` is the clear-sky diffuse-to-direct horizontal irradiance ratio from Ineichen and Perez, "A new airmass independent formulation for the Linke turbidity coefficient" (Solar Energy 73, 2002), at sea level, with the Kasten-Young air mass. This is the model pvlib (`ineichen`) uses by default. See `scenes/src/sky/clear-sky-ratio.ts`.
  - Two assumptions: the manifest's Preetham turbidity is used as Linke turbidity, and the ratio is broadband radiometric.
- Results today:

| scene | sun elevation | turbidity | sky / sun on the ground | diffuse fraction (sky / all daylight) |
| --- | --- | --- | --- | --- |
| sponza | 70.6° | 3 | 0.123 | 11.0% |
| courtyard-tree | 58.0° | 3 | 0.132 | 11.6% |
| bistro-exterior | 53.0° | 3 | 0.137 | 12.0% |
| furnished-room | 43.5° | 3 | 0.152 | 13.2% |
| open-ground | 37.0° | 3 | 0.167 | 14.3% |

Half-float rounding leaves the stored sky within 0.0001% of the target.

## Who reads it, and how

| consumer | how |
| --- | --- |
| viewer (`apps/viewer/src/scene/sky.ts`) | `EXRLoader`, as a sky dome that cameras and the probe-grid bake see (it replaced three's `SkyMesh`). Surfaces get no light from it directly: three r187 would add environment diffuse on top of probe GI (NOTES). `applySkyEnvironmentLighting` puts it in `scene.environment` for sky-only checks. |
| Cycles (`tools/reference/cycles/blender_scene.py`) | an Environment Texture node with the file, Linear Rec.709, linear interpolation, strength 1 |
| platform host (`scenes/src/sky/sky-file.ts`, `sky-texture.ts`) | `loadSkyFile(url)` (no engine; `DecompressionStream`), then `createSkyTexture(device, image)` gives an `rgba16float` texture for `setEnvironment({ type: 'texture', layout: 'equirect', texture })`. `SKY_EQUIRECT_WGSL` is the matching lookup. |
| native harness (`tools/native/native-lib.ps1`) | its shaders take one flat sky radiance, not an image. It gets the sidecar's `flatEquivalentRadianceRgb`: the uniform radiance that gives the same ground irradiance per channel. Unlike the file, it is also returned below the horizon. |

## Checks and what they measured (2026-11-04)

- **Orientation:** the file's sun glow is at the sun's azimuth (unit test).
  - The platform loader (compute shader) and the viewer's sky dome read 8 known directions within 0.021% of a CPU lookup of the file on the RTX 3060, and within 0.008% on the Intel UHD (`sky-orientation`).
  - Cycles reads them within 0.00% (`--calibrate`).
  - The glow is 3.1x the opposite sky, so a turned or mirrored sky would fail.
- **Cycles, sky on a white floor:** E / pi to 0.00% (`--calibrate`).
- **Open ground, bookmark 1** (ground only, albedo 0.5, 640 x 360). The analytic answer is 0.5 x E / pi per channel. Compared with `node tools/compare/index.ts cycles-full.exr viewer-diffuse.exr --downsample 8`:

| render | R | G | B | vs Cycles: bias | block error |
| --- | --- | --- | --- | --- | --- |
| Cycles, 1024 samples | -0.03% | -0.01% | -0.01% | | |
| viewer, diffuse-only materials, PMREM | -0.04% | -0.61% | -0.97% | -0.65% | 0.65% |
| viewer, its own MeshStandardMaterial | +1.66% | +1.09% | +0.72% | +1.05% | 1.05% |

  - The tolerance is ±0.5% brightness and 1% block error (DECISIONS).
  - Even with the specular lobe off, three's PMREM misses the brightness limit by 0.15 points.
  - The cause is PMREM. Its roughness-1 "irradiance" is a blurred GGX-style lobe times pi, not a cosine-weighted integral (NOTES). So it weights the sky's directions differently from Lambert's law, and each channel by a different amount, because the sky's colours are spread differently over the dome.
  - Cycles itself matches the file's analytic answer.
