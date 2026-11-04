# tools/sky: the shared sky (T1.1)

One sky definition becomes one HDR file, and every renderer reads that file: the viewer (three.js), Cycles, the platform, and (as a flat colour) the native harness.

```
node tools/sky/generate.ts                  # write scenes/sky/<id>.exr and .json for every scene with a procedural sky
node tools/sky/generate.ts --scene sponza   # one scene
node tools/sky/generate.ts --check          # compare the committed files with a fresh generation (the unit tests do this too)
node tools/sky/pmrem-env-render.ts --scene open-ground --camera 1 [--material diffuse|standard]   # three PMREM env diffuse (not used by the viewer), to bench/references/
pnpm test:gpu --filter sky-orientation      # the platform loader and the viewer read known directions the same as the file
pnpm test:gpu --filter hit-shading-sky      # the platform's projected sky: ground irradiance against the file's integral, the equirect contract
node tools/reference/cycles/run.ts --calibrate   # includes the Cycles sky orientation and sky-on-a-floor checks
```

Rerun `generate.ts` after changing a manifest's `environment`, its sun light, or the sky code (`packages/ddgi-core/src/platform/sky/` and `scenes/src/sky/`). `--check` fails until you do.

## Where the code lives (T1.4)

The sky model and its calibration are engine-free core code, shared with the platform's `procedural` environment, so the platform builds exactly the files' pixels:

- `packages/ddgi-core/src/platform/sky/`: `preetham.ts`, `clear-sky-ratio.ts`, `calibrated-sky.ts` (`buildCalibratedSky`, `DEFAULT_TURBIDITY` = 2, the one default), `equirect.ts`, `half-float.ts`.
- `scenes/src/sky/`: what is manifest-specific (`sky-image.ts`: which light is the sun; `platformSkyEnvironment(manifest)` gives the platform environment for the same sky), the EXR codec, and the loader.

## What is in a sky file

`scenes/sky/<scene id>.exr`, made from the manifest's `environment` (`type: "procedural"`):

- **Model:** Preetham, Shirley and Smits, "A Practical Analytic Model for Daylight" (SIGGRAPH 1999), implemented from the paper (`packages/ddgi-core/src/platform/sky/preetham.ts`), for the manifest's `sunDirection` and `turbidity` (default 2), converted from CIE xyY to linear Rec.709 RGB.
- **No sun disc.** The sun is the manifest's directional light, which every renderer already has. Putting it in the sky too would count it twice.
- **Black below the horizon.** The ground is scene geometry, so every renderer gets ground light from the same place.
- **Layout:** equirectangular, 512 x 256, +Y up, row 0 of the file is the zenith, `u = atan2(z, x) / 2pi + 0.5` (u = 0.5 is +X, 0.75 is +Z), the same as three.js `equirectUV`. In Blender, after the manifest-to-Blender axis change (x, y, z) to (x, -z, y), Cycles' own equirectangular lookup lands on the same texels, so no rotation is applied (`blender_scene.py`).
- **On the GPU the rows are the other way up:** the platform's equirect contract (`contract/scene.ts`) puts the nadir in the texture's first row and the zenith in its last, as three.js stores its equirect textures. `createSkyTexture` flips the file's rows when it uploads (since T1.4).
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
- `ratio` is the clear-sky diffuse-to-direct horizontal irradiance ratio from Ineichen and Perez, "A new airmass independent formulation for the Linke turbidity coefficient" (Solar Energy 73, 2002), at sea level, with the Kasten-Young air mass. This is the model pvlib (`ineichen`) uses by default. See `packages/ddgi-core/src/platform/sky/clear-sky-ratio.ts`.
  - Two assumptions: the manifest's Preetham turbidity is used as Linke turbidity, and the ratio is broadband radiometric.
- Each sidecar records its own ratio, diffuse fraction and target. The stored sky's measured irradiance is in `measured.calibrationError`, which is what half-float rounding leaves over. The values per scene are in PROGRESS (T1.1).

## Who reads it, and how

| consumer | how |
| --- | --- |
| viewer (`apps/viewer/src/scene/sky.ts`) | `EXRLoader`, as a sky dome that cameras and the probe-grid bake see (it replaced three's `SkyMesh`). Surfaces get no light from it directly: three r187 would add environment diffuse on top of probe GI (NOTES). `applySkyEnvironmentLighting` puts it in `scene.environment` for measuring three's PMREM on its own (`pmrem-env-render.ts`). |
| Cycles (`tools/reference/cycles/blender_scene.py`) | an Environment Texture node with the file, Linear Rec.709, linear interpolation, strength 1 |
| platform host (`scenes/src/sky/sky-file.ts`, `sky-texture.ts`) | `loadSkyFile(url)` (no engine; `DecompressionStream`), then `createSkyTexture(device, image)` gives an `rgba16float` texture for `setEnvironment({ type: 'texture', layout: 'equirect', texture })`. `SKY_EQUIRECT_WGSL` is the matching lookup. Or, with no file: `setEnvironment(platformSkyEnvironment(manifest))`, and the platform builds the same pixels itself (`procedural`). |
| native harness (`tools/native/native-lib.ps1`) | its shaders take one constant sky radiance, not an image, and cannot tell above the horizon from below without a shader change. It gets the sidecar's `flatEquivalentRadianceRgb`: the uniform radiance that gives the same ground irradiance per channel. Unlike the file, it is also returned below the horizon. |

## Checks

Results are in PROGRESS (T1.1); this section only says what each check does.

- **Orientation:**
  - A unit test checks that the file's sun glow sits at the sun's azimuth.
  - `pnpm test:gpu --filter sky-orientation` reads 8 known directions from the file in two ways and compares each with a CPU lookup (1% limit):
    - the platform loader, through a compute shader;
    - the viewer's sky dome, rendered by three.
  - `node tools/reference/cycles/run.ts --calibrate` does the same in Cycles (1% limit).
  - The page also checks that the glow is at least 2x the sky opposite it, so a turned or mirrored sky can't pass.
- **Cycles sky on a white floor:** `--calibrate` renders the sky lighting a white floor and compares it with E / pi, with E integrated from the file's texels.
- **Open ground, bookmark 1** (ground only, albedo 0.5):
  - The analytic answer is 0.5 x E / pi per channel, with E from the sidecar.
  - Cycles: `node tools/reference/cycles/run.ts --scene open-ground --camera 1`.
  - `pmrem-env-render.ts` renders three's PMREM environment diffuse. **The viewer never lights surfaces this way** (see the table above), so this is a measurement of three's PMREM, not the viewer's sky check. The viewer's check is the orientation page: it shows the same file, the same way round.
  - Compare with `node tools/compare/index.ts cycles-full.exr three-pmrem-env-diffuse.exr --downsample 8`. Read both metrics:
    - `bias`: signed brightness;
    - `mean relative error`: sum |T - R| / sum R over 8 x 8 blocks;
    - `p95 8x8 block error`: the 95th-percentile block.
  - Sky-lighting accuracy against Cycles is checked on the platform's own sky irradiance in T1.4 (`pnpm test:gpu --filter hit-shading-sky`: `giSkyIrradiance` on flat ground against the sidecar's integral, ±0.5% per channel), and through DDGI in the viewer in T1.8.
