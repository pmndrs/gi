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

They are grouped by method, one folder each: `ddgi/`, `cache/` and `cascades/` (and `together/` for pages that combine techniques). Which method to use for what is in the guide, [DDGI or the radiance cache?](../docs/guides/ddgi-or-radiance-cache.md), and on the landing page. The plan for the sets is [docs/plans/examples.md](../docs/plans/examples.md).

Every page has an explainer panel, top left: what you're seeing, how it works, what to try, and trade-offs with each number's source. Its text is plain data in the page's `about.ts`, so `main.ts` stays the example.

### DDGI (`ddgi/`)

| # | Example | Shows |
| --- | --- | --- |
| 01 | **Hello GI** (`ddgi/01-hello`) | A `DDGIVolume` in place of three's `LightProbeGrid`, on three's own `webgpu_lightprobes.html` scene: the getting-started code from the README. GI updates itself inside `renderer.render`. |
| 02 | **GI composite with the upscaler** (`ddgi/02-composite`) | The same scene through three's `RenderPipeline`: `ddgi()` adds GI once per pixel, and `@pmndrs/upscaler` renders at half resolution and upscales. The path for dense scenes. |
| 03 | **Lights** (`ddgi/03-lights`) | The Cornell box with one lamp to move, dim, recolour and switch off. GI follows within a few frames, and the room never goes dark meanwhile. |
| 04 | **Moving objects** (`ddgi/04-moving-objects`) | Two rooms: a door swings on its hinge and a box slides. Nothing is called; the far room goes dark when the door shuts. |
| 05 | **Time of day** (`ddgi/05-time-of-day`) | A walled courtyard under three's `SkyMesh`: the sun crosses the sky, and GI follows the sun light and the sky together. The sun layer (`sunLayer: true`) makes its colour and brightness exact in the same frame. |
| 06 | **Materials** (`ddgi/06-materials`) | Three scenes (`?scene=`): light through an alpha cut-out checker wall, through a glass pane, and textures with eight uv mappings bouncing their colours in the sun. |
| 07 | **Quality presets** (`ddgi/07-presets`) | `low`, `medium` and `high`, switched live; the stats line shows the probes, the memory and the GPU time. Opens on `recommendPreset`'s advice. |
| 08 | **Saved bake** (`ddgi/08-saved-bake`) | The first visit warms up and saves the probes (`volume.saveBake()`) into IndexedDB; a reload starts from them (`start(..., { bake })`), with no warm-up. |
| 09 | **Export to LightProbeGrid** (`ddgi/09-lightprobegrid`) | `volume.exportLightProbeGrid()`, three's own `LightProbeGrid` as a static fallback, wiped against live GI by the mouse; download it as JSON. |
| 10 | **Debug views** (`ddgi/10-debug`) | The probes where they sit and in their state's colour, the screen debug views (GI only, coverage, probe states; `?view=`), and the irradiance atlas, drawn in three from the technique's debug outputs (`ddgi/10-debug/debug-views.ts`). |

### Radiance cache (`cache/`)

| # | Example | Shows |
| --- | --- | --- |
| C1 | **No volume needed** (`cache/c1-no-volume`) | The radiance cache alone (`@pmndrs/gi/radiance-cache`, `worldCache: 'sharc'`) lighting a sunlit courtyard with no probe box: screen probes over a world cache of cells keyed by position, wired in through `screenInputs` and `ddgi()` on three's `RenderPipeline`. GI on and off, and the sun's height and direction. [docs/api/radiance-cache.md](../docs/api/radiance-cache.md) explains each line. |
| C2 | **Contact detail** (`cache/c2-contact-detail`) | The cache (SHaRC world cache, no volume) and a `DDGIVolume` on one scene pass of `cabinet-row-objects`, each through its own `ddgi()` composite, wiped by the mouse (`shared/wipe.ts`) with a difference view. DDGI's probe spacing switches between 0.25 m and 0.5 m, so its blur at contacts shows against the cache's screen probes. |
| C3 | **Light through a doorway** (`cache/c3-doorway`) | Two rooms, the camera in the unlit one: the world cache's second bounce from GI-1.2's own rays (`secondBounce: 'gi1-rays'`), from a DDGI volume stacked under the cache (the default), or from SHaRC (`worldCache: 'sharc'`), switched live, with the door open or shut. The far room's GI is read back and shown as a number and a bar for each. |
| C4 | **Fast changes: lag against ghosting** (`cache/c4-fast-changes`) | The cache on its defaults in a dark room (the `dark-room` scene): a lamp on a switch, a flashlight on the camera and a box to drag. A chart of the mean GI light over the last 120 frames marks each change and reads the frames to settle; a ghost view shows light still moving; `cellChangeReset` (register E9) switches on and off. |
| C5 | **Light from surfaces** (`cache/c5-emissive`) | A room with no light objects, lit only by two emissive strips and a screen (the procedural `emissive-room`), through the cache on its defaults: a probe ray that hits an emitter takes its emission. Change the strips' colour and brightness, put one of two pictures made in code on the screen, or show GI only. |
| C6 | **The cache switcher** (`cache/c6-cache-switcher`) | One world cache at a time from `RADIANCE_CACHE_VARIANT_PRESETS` (`?variant=`), on two closed rooms or the open courtyard (`?scene=courtyard`): a card per variant with how it works, its pros and cons, P6.5's measured bias, lag and memory and what it needs from the host; GPU time per pass and memory live; the cache's debug views (`?overlay=probe-atlas`, `probe-mask`, `probe-sh`). |

### Together (`together/`)

| # | Example | Shows |
| --- | --- | --- |
| X2 | **Scale** (`together/x2-scale`) | A generated town about 200 m across (the `block-world` scene, from a seed), flown through: DDGI with several volumes (`DDGISettings.volumes`: a coarse one over the town, and one or two finer windows that follow the camera, with their outlines), or the radiance cache with SHaRC and no volume, switched live. GPU time per pass and memory for each method as you fly. |

### Split Radiance Cascades (`cascades/`)

| # | Example | Shows |
| --- | --- | --- |
| 11 | **Split Radiance Cascades** ([source](cascades/11-radiance-cascades/)) | A procedural passage with changing emission, a repeatable camera route and direct, indirect and composite views. Compare the native WebGPU solver with the Three.js reference. |

## Split Radiance Cascades scenes

Run `pnpm examples`, then choose a starting scene. The examples index includes these five links. The radiance cache (GI-1.2) has its own examples, `cache/`; the viewer shows it on every test scene with its compare panel: `pnpm dev`, then [localhost:5200](http://localhost:5200/?technique=rc).

- [Lamp gallery camera rail](http://localhost:5203/cascades/11-radiance-cascades/) follows a repeatable route past rooms with changing emission.
- [Sliding shutter](http://localhost:5203/cascades/11-radiance-cascades/?scene=door&camera=receiver&view=indirect) holds the camera on a wall while the door blocks and reveals the lamp. Switch the door to Open or Closed to hold either endpoint.
- [Polished gallery rail](http://localhost:5203/cascades/11-radiance-cascades/?scene=polished) shows how floor reflections change as the camera moves.
- [Five floor finishes](http://localhost:5203/cascades/11-radiance-cascades/?scene=roughness) shows the same lighting across five roughness values.
- [Offscreen moving emitter](http://localhost:5203/cascades/11-radiance-cascades/?scene=reflection-motion) holds the camera still while a hidden light moves across the reflected field.

## No engine

| # | Example | Shows |
| --- | --- | --- |
| W1 | **Plain WebGPU** (`webgpu/hello`) | `@pmndrs/gi/core` and `/ddgi` on a bare `GPUDevice`: a small renderer of its own, the scene described through the platform's scene API, DDGI encoded into the frame's command encoder. The example [docs/guides/adapter.md](../docs/guides/adapter.md) walks through. It is also its own package (`pnpm --filter @pmndrs/ddgi-example-hello dev`). |

## How they're built

- `shared/renderer.ts`: three's `WebGPURenderer` on a device with the limits GI needs (`createGIRenderer`), filling the window, and orbit controls.
- `shared/scene.ts`: the repo's procedural test scenes (`scenes/`, made in code, MIT) as three scenes, with their lights, sky, camera bookmarks and suggested probe box. Downloaded models are left out: the site ships only what we can share.
- `shared/stats.ts`: the line at the bottom: probes, GPU time, memory, the probes' mean irradiance and the frame count. `shared/cache-stats.ts` is the cache's: screen probes, memory, the mean irradiance the composite reads and the frame count.
- `shared/status.ts`: the same numbers on `window.ddgiExample` (`{ frames, giMean, error }`), for the browser console and for the repo's check of the built site.
- `shared/explainer.ts`: the explainer panel (`showExplainer(content)`, called from each page's `about.ts`), with `API_PAGES`, `GUIDE` and `docsLink` for its links.
- `shared/wipe.ts`: a split wipe between two pictures of one view, dragged by the mouse or a finger, as nodes on a `RenderPipeline` (`wipe.node(left, right)`) or with a scissor (`wipe.splitPixels(width)`); `wipe.difference` shows |left − right|.
- `shared/timeline.ts`: a small chart of a value over the last frames, with markers at changes and a frames-to-settle reading.
- `shared/pass-timing.ts`: GPU milliseconds per pass from `gi.stats.passes`, each with one line on what it does.
- `shared/dock.ts`: the bottom-right column the timeline and the pass list share when a page gives them no container.
- `shared/nav.ts`: the bar at the top of each page (back to the examples, the title, the source). `vite.config.ts` adds it to every page, so an example never imports it.

## The published site

`.github/workflows/deploy-pages.yml` builds the site with `PAGES_BASE=/gi/` (GitHub Pages serves a project site under `/<repo>/`; the site is `https://pmndrs.github.io/gi/`) and deploys it on every push to `main`, once Pages is enabled for the repository. Links between pages are relative, so the site works under any base.

To check a build the way Pages serves it (on a machine with a GPU):

```bash
PAGES_BASE=/gi/ pnpm examples:build
pnpm test:gpu --filter examples-site      # every example in headless Chrome: GI shows, no errors
node tools/examples/screenshots.ts         # a picture of each, into bench/screenshots/t5.6
```
