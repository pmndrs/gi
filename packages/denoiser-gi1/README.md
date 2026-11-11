# @pmndrs/denoiser-gi1

The image-space GI denoiser from AMD's GI-1 (Capsaicin v1.3, `914b915`, MIT; see `NOTICE`), in WGSL on a bare `GPUDevice`: `ReprojectGI` and two `FilterGI` passes, with the history Capsaicin keeps. It takes a noisy diffuse irradiance image (one value per pixel, such as GI-1's interpolated screen probes) and returns a steadier one.

The radiance cache (`@pmndrs/ddgi/radiance-cache`) runs it after `InterpolateScreenProbes`, as Capsaicin does. The package imports nothing from `@pmndrs/ddgi`, from an engine or from npm; lint enforces that (`local/self-contained` in the repo's `eslint.config.mjs`, tested in `tools/lint/boundaries.test.ts`). It is bundled into `@pmndrs/ddgi` for now and written to move into `pmndrs/denoiser` as it is (below).

## What it does

Per frame, each a compute dispatch over 8 × 8 pixel groups:

1. **`ReprojectGI`** finds the pixel's history through the motion vector: a 3 × 3 neighbourhood round the previous position, each texel weighted by a tent of its distance (0 at √2 pixels), kept where the surface is the same (within a cell of GI-1's screen-probe size, normals within 0.95). It appends this frame's sample, shortens the history where the luminance changes, and towards the screen's edges, to between 4 and 64 samples. It writes a **blur mask**: 8 minus the history's length, so a pixel with less than 8 frames of history is blurred, more the less it has.
2. **`FilterGI`**, horizontally then vertically, over the mask's radius, weighted by depth (sharply where there is history) and normal.
3. The second pass's result is the next frame's history (the filtered image is fed back, as in Capsaicin), and its mean is the output.
4. **`KeepGuides`** keeps this frame's depth and normals for the next frame's reprojection (Capsaicin's framework does this for it).

Constants are Capsaicin's, with their lines, in `src/constants.ts`. Every difference from Capsaicin, and why, is in webgpu-ddgi's `docs/design/radiance-cache-divergences.md` (rows D65 to D75).

## Use

```ts
import { GI1Denoiser } from '@pmndrs/denoiser-gi1';

const denoiser = await GI1Denoiser.create(device);    // the host owns the device
denoiser.configure({ width, height });                // drops the history
// every frame, after the frame's inputs are submitted:
if (cameraCut) denoiser.resetHistory();
const irradiance = denoiser.dispatch(
    { radiance },                                     // rgb irradiance, alpha the hint (1 = keep)
    { depth, normal, motion },                        // motion = NDC current − previous
    { view, projection, prevView, prevProjection, near, far, position, jitter, reversedDepth },
);
// `irradiance` (rgba16float, rgb) is valid until the next configure or dispose.
```

`encode(encoder, inputs, guides, camera)` records the same work into your encoder instead of submitting it (the radiance cache uses this).

**Inputs.** `radiance` is any float texture: rgb the irradiance, alpha Capsaicin's denoiser hint (above 0: a sample to keep; 0: one to accumulate only when there is no history, as GI-1 marks its fallback interpolation). Write alpha 1 if you have no hint. `depth` is hardware depth or a one-channel float texture; a pixel at the far plane is sky. `normal` is world-space unit normals (`normalEncoding: 'unorm'` for 0..1 encoded). `jitter` is this frame's sub-pixel jitter in pixels (x right, y down); the denoiser remembers the previous frame's. All must be the configured size.

**Options.** `cellPixels` (default 8, GI-1's probe size) sets the reprojection's distance tolerance; `textureFormats` (`'auto'` or `'core'`, above); `label`; `timer` (a `computePass(name)` that returns timestamp writes; the denoiser times `<label>/ReprojectGI` and `<label>/FilterGI`, named after Capsaicin's sections, and `<label>/keep-guides`, its copy of the depth and normals).

**Resources.** Two history colours (`rgba16float`), two colour deltas and the blur mask, the output (`rgba16float`), last frame's depth (`r32float`) and normals (`rgba16float`): 41 bytes a pixel (38 MB at 1280 × 720) with Capsaicin's `r16float` deltas and `r8snorm` mask, which need `texture-formats-tier1`; on a device without it, the one fallback, `r32float` and `rgba8snorm`, 48 bytes a pixel (`textureFormats: 'core'` forces it; `denoiser.formats` says which). No storage buffers; at most 3 storage textures and 8 sampled textures per kernel, inside WebGPU's default limits.

**Reference.** `src/reference.ts` is a CPU port of the kernels, line for line, which the tests compare the GPU against (`tools/gpu/pages/radiance-cache-denoiser`).

## Moving it into pmndrs/denoiser

The interface follows `@pmndrs/denoiser`'s temporal API (`packages/denoiser-core/src/temporal.ts` there): the same `FrameCamera`, `FrameGuides` and `TemporalConfig` fields, `DiffuseInputs` for the signal, `configure` / `dispatch` / `resetHistory` / `dispose`, motion as NDC current − previous, the device owned by the host, the output owned by the denoiser. The types are declared here (`src/types.ts`) so this package depends on nothing; TypeScript's structural types make them interchangeable.

To move it:

1. Copy `src/`, `NOTICE` and this README into a package there, for example next to `denoiser-ffx` as `denoiser-gi1`, or into `denoiser-ffx` as a third class.
2. Replace `src/types.ts` with imports from `@pmndrs/denoiser-core` and declare `class GI1Denoiser implements TemporalDenoiser<DiffuseInputs>`. Two fields differ: `FrameGuides.normalEncoding` (optional, ours; their normals are always signed) and `DiffuseInputs.radiance`'s alpha (the hint), which `@pmndrs/denoiser`'s docs would need to mention. `DiffuseInputs.hitDistance` is not used.
3. Their classes build pipelines synchronously in the constructor; this one has `static create()` (async compile). A constructor that calls `createComputePipeline` would match theirs.
4. Move `tools/gpu/pages/radiance-cache-denoiser`'s module checks (checks 0 to 4) into their test bed, and the unit tests (`src/denoiser.test.ts`).
5. Here, `@pmndrs/ddgi-core` would depend on the published package instead of the workspace one; `techniques/radiance-cache/gi-denoiser.ts` is the only file that imports it.

Moving it needs Dennis's go-ahead (a PR to `pmndrs/denoiser`).
