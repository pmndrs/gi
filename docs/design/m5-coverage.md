# M5 coverage pass: presets, saved bakes, the LightProbeGrid export, docs

Written 2026-11-10 by the lead (Mac). Prototype level, under DECISIONS 2026-11-10 "Feature coverage before tuning": each feature has to work end to end, pass one smoke check and get Dennis's look. Numbers are recorded, never gated, except the correctness bars named below. The examples site and release prep come after this pass.

SPEC's M5 lines: quality presets (with scaled-down settings for the correctness devices), saved bakes ("the field saves and loads in a versioned, engine-neutral format"), the `LightProbeGrid` JSON export (SPEC "Bake export": a snapshot as a static or WebGL fallback), and public API docs plus an adapter guide built on the plain-WebGPU example.

## 1. Quality presets (T5.1)

**What:** a small named set of DDGI and platform settings, so a host picks one word instead of thirty settings.

- `ddgiPreset(name: 'low' | 'medium' | 'high')` in `@gi/ddgi` returns a `Partial<DDGISettings>` plus the platform-side numbers a preset also sets (the update budget, the texture store's layer size). It is plain data, so a host can spread it and override single fields.
- `high` is today's defaults, unchanged: it is what M1-M4 were checked with on the RTX 3060 and what the M5 Pro runs.
- `low` is for the correctness devices (Intel UHD, iPad, older iOS): fewer rays per probe, smaller tiles, a coarser default spacing when the host doesn't set one, a lower budget and a smaller texture store. `medium` sits in between. The implementer picks the values from the settings' own documented ranges, and lists them in a table in `docs/api/platform.md` with one line on what each costs in look.
- `recommendPreset(adapterInfo, limits)` gives `low` for integrated and mobile GPUs and `high` otherwise (Apple's Pro/Max chips count as `high`). It is advice only: nothing applies it behind the host's back.
- three: `GIVolume` and `start` take `preset`. Viewer: a preset picker in the panel and `preset=` in the link.
- Changing rays or tile sizes needs a new `prepare` (that's how the settings already work). The preset switch says so in the viewer's tooltip; it isn't a recompile bug.

**Done when:** on the Mac, each preset passes `ddgi-furnace` (2%) and `ddgi-leak` (≤ 1%) and keeps `platform-recompiles` at 0 within one preset; a new page `ddgi-presets` runs all three on Cornell and two-room and records probes, memory and rays; the viewer's picker works on Sponza and Bistro (screenshots looked at). The Intel and iPad runs of `low` are recorded when those devices are next available (the gpu-tester's LAN pages for the iPad); they don't block.

## 2. Saved bakes (T5.2)

**What:** save a converged field and load it later, so a page can start with finished GI and keep updating from there.

- **Engine-neutral, in core.** `technique.saveField(): Promise<Uint8Array>` and `technique.loadField(bytes, options?)`. Saving is an explicit host call, so its one readback is allowed (as at load, SPEC's readback rule is about the frame loop).
- **Format (`.ddgibake`), little-endian:** the magic `DDGB`, a `u32` format version (1), a `u32` header length, a JSON header, then the raw sections, each 256-byte aligned.
  - The header: `formatVersion`, the library version, the layout that has to match to load (irradiance and distance texels, the distance format, layer count, and per volume its counts, spacing, origin, rotation and priority), every section's name, format, size and byte offset, the scene fingerprint, and `createdAt`.
  - The sections: the irradiance atlas (every light-group layer), the distance atlas, probe data (relocation offsets and states), and the probe history. The probe history is what lets a loaded field carry on blending instead of re-warming. The composed atlas isn't saved; compose rebuilds it.
- **Scene fingerprint** (SPEC "content fingerprints for saved bakes"): a hash over what changes the answer: geometry (per mesh: triangle count, bounds, a hash of positions and indices), material records, texture bytes, lights and their shadow *parameters* (not matrices), and the sky settings. A prototype may hash less if a part is costly to reach, but it has to say which parts are in.
- **Loading:** a layout mismatch is refused with a plain message that names the field that differs. A fingerprint mismatch loads anyway, flags `stale: true` and marks every probe changed, so the field updates from the bake instead of from black. A loaded field skips the warm-up.
- three: `volume.saveBake()` / `volume.loadBake(bytes)`. Viewer: Save and Load buttons (a file download, and a file picker or drop).

**Done when:** on Cornell and two-room, a converged field is saved, then loaded on a fresh page. The test view's mean is within 1% of the saved page's at the first frame, no warm-up pass ran, and recompiles stay at 0. A layout mismatch is refused. A changed scene loads as stale and settles back within the usual response. The viewer round trip works on Sponza. Unit tests cover the header, the alignment and the version check.

## 3. The LightProbeGrid export (T5.3)

**What:** a snapshot of the field as three's `LightProbeGrid`, for a static fallback (or WebGL, through `LightProbeGridWebGL`).

- **Core:** `exportIrradianceSH(technique, volume)` projects each probe's octahedral irradiance tile onto L2 spherical harmonics (9 RGB coefficients a probe) on the GPU, with one readback. It returns engine-neutral data: the grid's counts, spacing, origin and the coefficients. The coefficients use three's convention (radiance SH as `LightProbe` stores it, so three's `shGetIrradianceAt` gives back our irradiance). A unit test projects a known function and reads it back through three's formula.
- **three:** `toLightProbeGrid(data)` builds a WebGPU `LightProbeGrid` (r187, `examples/jsm/lighting/LightProbeGrid.js`) whose probes sit where ours do. Its probes are at cell centres, so the grid is `counts × spacing` wide and centred on our grid's centre. Its texture is filled in its own packing (seven RGBA sub-volumes along Z), so no `bake` call is needed. Also `lightProbeGridToJSON` / `lightProbeGridFromJSON`. Check open PR #33658's format first (a scout lookup). Match it if it is settled; otherwise write ours with a `formatVersion` and say so in `docs/api/three.md`.
- Relocation offsets and visibility (distances, Chebyshev) are dropped: `LightProbeGrid` has neither. That is the known cost of the fallback, and the docs say so.

**Done when:** a coefficient round trip through JSON is exact. On Cornell and Sponza, the scene lit by the exported grid against our live GI: the test view's mean within 15%, and screenshots side by side, looked at. The difference is expected, from L2 and no visibility. It's recorded, not tuned.

## 4. Docs and the adapter guide (T5.4)

- `docs/api/platform.md` and `docs/api/three.md` brought up to date with M2-M5 (lights, groups, time of day, movers and deforming meshes, several volumes and the camera volume, materials and textures, presets, bakes, the export), each with a short example.
- `docs/guides/adapter.md`, a guide to writing an engine adapter, walking through `examples/webgpu/hello` (the plain-WebGPU example) step by step: device and limits, the scene copy, materials and textures, lights, changes, the frame, sampling GI in your shading. First, fix the hello example's known gap: it passes only material slot 0 and no textures.
- A getting-started section in the root README (three in a dozen lines, as `LightProbeGrid` does).

**Done when:** a script lists the public exports of `@pmndrs/ddgi/core` and `@pmndrs/ddgi/three` and checks that each is named in the API docs (a product check: the docs cover what ships). The guide's code blocks are pulled into a typechecked file. `three-neutrality` still passes with the fixed hello example. Dennis reads the getting-started.

## 5. The coverage note (T5.5)

`docs/milestones/M5.md`: a table of every SPEC v1 feature, with the page that checks it, its result on the M5 Pro, and its known misses (each with its DECISIONS entry). Then the proposed tuning pass, in order, for Dennis to pick from. Then Dennis's look, across the viewer's scenes and scenarios.
