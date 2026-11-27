# @pmndrs/gi examples

Small, single-purpose pages, each the shortest code that shows one feature. WebGPU only: Chrome or Edge 113+, or Safari 26+.

```bash
node tools/linked-siblings.ts --fetch  # prepare the pinned denoiser on a fresh checkout (Git and Corepack required)
pnpm install --frozen-lockfile
pnpm examples          # http://localhost:5203, the landing page links every example
pnpm examples:build    # the site, into examples/dist
```

The examples read the library straight from `packages/`, so a change to it shows up here at once. Examples 02 and 11 also use `@pmndrs/upscaler`.

## The examples

| # | Example | Shows |
| --- | --- | --- |
| 01 | **Hello GI** (`01-hello`) | A `DDGIVolume` in place of three's `LightProbeGrid`, on three's own `webgpu_lightprobes.html` scene: the getting-started code from the README. GI updates itself inside `renderer.render`. |
| 02 | **GI composite with the upscaler** (`02-composite`) | The same scene through three's `RenderPipeline`: `ddgi()` adds GI once per pixel, and `@pmndrs/upscaler` renders at half resolution and upscales. The path for dense scenes. |
| 03 | **Lights** (`03-lights`) | The Cornell box with one lamp to move, dim, recolour and switch off. GI follows within a few frames, and the room never goes dark meanwhile. |
| 04 | **Moving objects** (`04-moving-objects`) | Two rooms: a door swings on its hinge and a box slides. Nothing is called; the far room goes dark when the door shuts. |
| 05 | **Time of day** (`05-time-of-day`) | A walled courtyard under three's `SkyMesh`: the sun crosses the sky, and GI follows the sun light and the sky together. The sun layer (`sunLayer: true`) makes its colour and brightness exact in the same frame. |
| 06 | **Materials** (`06-materials`) | Three scenes (`?scene=`): light through an alpha cut-out checker wall, through a glass pane, and textures with eight uv mappings bouncing their colours in the sun. |
| 07 | **Quality presets** (`07-presets`) | `low`, `medium` and `high`, switched live; the stats line shows the probes, the memory and the GPU time. Opens on `recommendPreset`'s advice. |
| 08 | **Saved bake** (`08-saved-bake`) | The first visit warms up and saves the probes (`volume.saveBake()`) into IndexedDB; a reload starts from them (`start(..., { bake })`), with no warm-up. |
| 09 | **Export to LightProbeGrid** (`09-lightprobegrid`) | `volume.exportLightProbeGrid()`, three's own `LightProbeGrid` as a static fallback, wiped against live GI by the mouse; download it as JSON. |
| 10 | **Debug views** (`10-debug`) | The probes where they sit and in their state's colour, the screen debug views (GI only, coverage, probe states; `?view=`), and the irradiance atlas, drawn in three from the technique's debug outputs (`10-debug/debug-views.ts`). |
| 11 | **Split Radiance Cascades** ([source](11-radiance-cascades/)) | A procedural passage with changing emission, a repeatable camera route and direct, indirect and composite views. Compare the native WebGPU solver with the Three.js reference. |
| 12 | **Radiance cache** (`12-radiance-cache`) | `@pmndrs/gi/radiance-cache` on 01's scene, from public paths: screen probes over a world-space hash grid, wired in through `screenInputs` and `ddgi()` on three's `RenderPipeline`, with a DDGI volume stacked under it for the later bounces. `?ddgi=off` runs the cache alone. [docs/api/radiance-cache.md](../docs/api/radiance-cache.md) explains each line. |

## Split Radiance Cascades scenes

Run `pnpm examples`, then choose a starting scene. The examples index includes these five links. The radiance cache (GI-1.2) has its own example, 12; the viewer shows it on every test scene with its compare panel: `pnpm dev`, then [localhost:5200](http://localhost:5200/?technique=rc).

- [Lamp gallery camera rail](http://localhost:5203/11-radiance-cascades/) follows a repeatable route past rooms with changing emission.
- [Sliding shutter](http://localhost:5203/11-radiance-cascades/?scene=door&camera=receiver&view=indirect) holds the camera on a wall while the door blocks and reveals the lamp. Switch the door to Open or Closed to hold either endpoint.
- [Polished gallery rail](http://localhost:5203/11-radiance-cascades/?scene=polished) shows how floor reflections change as the camera moves.
- [Five floor finishes](http://localhost:5203/11-radiance-cascades/?scene=roughness) shows the same lighting across five roughness values.
- [Offscreen moving emitter](http://localhost:5203/11-radiance-cascades/?scene=reflection-motion) holds the camera still while a hidden light moves across the reflected field.

## No engine

| # | Example | Shows |
| --- | --- | --- |
| W1 | **Plain WebGPU** (`webgpu/hello`) | `@pmndrs/gi/core` and `/ddgi` on a bare `GPUDevice`: a small renderer of its own, the scene described through the platform's scene API, DDGI encoded into the frame's command encoder. The example [docs/guides/adapter.md](../docs/guides/adapter.md) walks through. It is also its own package (`pnpm --filter @pmndrs/ddgi-example-hello dev`). |

## How they're built

- `shared/renderer.ts`: three's `WebGPURenderer` on a device with the limits GI needs (`createGIRenderer`), filling the window, and orbit controls.
- `shared/scene.ts`: the repo's procedural test scenes (`scenes/`, made in code, MIT) as three scenes, with their lights, sky, camera bookmarks and suggested probe box. Downloaded models are left out: the site ships only what we can share.
- `shared/stats.ts`: the line at the bottom: probes, GPU time, memory, the probes' mean irradiance and the frame count. `shared/cache-stats.ts` is 12's: screen probes, memory, the mean irradiance the composite reads and the frame count.
- `shared/status.ts`: the same numbers on `window.ddgiExample` (`{ frames, giMean, error }`), for the browser console and for the repo's check of the built site.
- `shared/nav.ts`: the bar at the top of each page (back to the examples, the title, the source). `vite.config.ts` adds it to every page, so an example never imports it.

## The published site

`.github/workflows/deploy-pages.yml` builds the site with `PAGES_BASE=/gi/` (GitHub Pages serves a project site under `/<repo>/`; the site is `https://pmndrs.github.io/gi/`) and deploys it on every push to `main`, once Pages is enabled for the repository. Links between pages are relative, so the site works under any base.

To check a build the way Pages serves it (on a machine with a GPU):

```bash
PAGES_BASE=/gi/ pnpm examples:build
pnpm test:gpu --filter examples-site      # every example in headless Chrome: GI shows, no errors
node tools/examples/screenshots.ts         # a picture of each, into bench/screenshots/t5.6
```
