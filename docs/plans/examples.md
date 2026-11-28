# Examples: one set per method, each one teaching something

Status: **draft for Dennis to agree on**, 2026-11-28. It closes road step 4, "Docs and demos". Once agreed, each demo below is one subagent's task.

## Why

Today DDGI has ten examples (01–10), the radiance cache one (12, on DDGI's scene), and Split RC one page with scene links (11). A reader can't see what each method is for, or why they would pick it.

Each method gets its own small set of pages. Each page:
- uses a scene chosen for what that method does well;
- uses the method's own API;
- explains what is on screen.

Comparison is a teaching aid inside a page, not the page itself.

## What every page has

- **The shortest code for one idea.** That is the README's existing promise, and it still holds. The explainer lives in shared code, not in the example's `main.ts`.
- **An explainer panel** (`shared/explainer.ts`, new), collapsible, with four parts:
  - **What you're seeing:** two or three sentences.
  - **How it works:** a short plain-language mechanism, with one small diagram (inline SVG) where it helps.
  - **Try this:** two to four numbered steps. Each one names a control and what to watch for, for example "Switch the lamp off: the far room goes dark within a few frames."
  - **Trade-offs:** pros and cons. Every number has its source next to it (a docs page or a DECISIONS entry). Never quote a number that isn't already in `docs/`.
- **The stats line**, extended where it teaches something: GPU ms per pass (#52's timing), memory, and frames to settle.
- **A link to the method's API page and to the guide** (`docs/guides/ddgi-or-radiance-cache.md`).
- **Procedural scenes only** (`scenes/src/procedural/*`; today: `cornell-box`, `courtyard`, `two-room`, `furnace-room`, `thin-wall`, `closed-door`, `cabinet-row`, `open-ground` and four textured ones). New scenes join that registry, so the viewer and tests get them too. The published site can't ship the downloaded models.
- **Checked by `examples-site`** on the RTX: it renders, there are no errors, and the GI mean is finite and lit. Timing is never gated.

## Layout of the site

The examples index is grouped by method, with a short "which method for what" intro that links to the guide:
- **DDGI:** 01–10 (as now).
- **Radiance cache:** C1–C6.
- **Together:** X1–X2, combining techniques or libraries.
- **Split Radiance Cascades:** 11 plus the Mac workflow's pages. Not part of this plan.

Folders become `examples/ddgi/01-hello/`, `examples/cache/c1-.../` and so on. Nothing is published yet, so renaming now breaks no link.

## The radiance cache (C1–C6)

### C1 · No volume needed
- **Teaches:** the cache needs no probe box. Add it to a scene and it lights what the camera sees. This is the cache's pitch.
- **Scene:** `courtyard` (open, sunlit) with `open-ground` around it.
- **On screen:** GI on and off; a probe-box outline shown only when DDGI is picked, to make the point that the cache has none.
- **Controls:** GI on/off; the sun's angle.
- **Explainer:**
  - Screen probes: one per tile of pixels, tracing into the scene.
  - World cache: a hash grid of cells keyed by position. Cells exist only where rays land, and grow coarser with distance.
  - What it costs: it reads the screen's depth, normals and motion, so the host renders an MRT.
- **API shown:** `RadianceCacheTechnique`, `screenInputs`, `ddgi()` composite. This replaces today's 12, which becomes C1's code.
- **Depends on:** P6.7 (SHaRC as the default with no volume). Until then the page sets `worldCache: 'sharc'` itself and says so.

### C2 · Contact detail
- **Teaches:** screen probes keep light where objects meet. DDGI's probe grid, a fixed spacing apart, blurs it.
- **Scene:** `cabinet-row`: a shelf row with legs, gaps and objects against a wall.
- **On screen:** a split wipe, cache on one side and DDGI on the other, dragged by the mouse; a "difference" view.
- **Controls:** wipe position; DDGI probe spacing (two settings).
- **Explainer:**
  - Why a grid blurs at contacts: probe spacing against object size.
  - Why screen probes don't: they sit on the visible surface.
  - P6.5's per-block p95 numbers: SHaRC beats the grid over DDGI on block error in the open scenes. Source: `docs/design/radiance-cache-variants.md`.
- **Depends on:** a shared split-wipe helper, which the viewer already has.

### C3 · Light through a doorway (bounces in closed rooms)
- **Teaches:** light reaches a room only by bouncing several times. A cache that carries two bounces leaves closed rooms dark, and DDGI underneath fixes that.
- **Scene:** `two-room`, lit from one side through a doorway.
- **On screen:** the far room's brightness as a number and a bar, for three second-bounce sources.
- **Controls:**
  - The second bounce: GI-1.2's own rays, DDGI underneath (the default), or SHaRC.
  - Door open or closed.
- **Explainer:**
  - What a bounce is, with a diagram: lamp → wall → floor → eye.
  - Why GI-1.2 alone stops at two bounces.
  - The numbers: two-room reads −44% with the grid alone, −1.5% with DDGI under it. Source: P6.5 and RC3.6.
- **Depends on:** nothing new.

### C4 · Fast changes: lag against ghosting
- **Teaches:** every cache keeps history to stay calm, and history means lag after a change. The cache resets cells near a change, so it follows fast without ghosting.
- **Scene:** a new procedural dark room, built with `addRoomShell` (`scenes/src/procedural/room-shell.ts`), with a flashlight on the camera, a light switch and a box you can drag.
- **On screen:** a small live chart of the scene's brightness over the last 120 frames, with a marker at each change; a "ghost" view (the difference from the settled image).
- **Controls:** flashlight on/off; room light switch; drag the box; change resets on/off (E9).
- **Explainer:**
  - History, and what "frames to settle" means.
  - Ghosting: old light left behind a moving object.
  - What the reset does.
  - The numbers: sun doubled, 2 frames with resets against 37 without. Source: TASKS P6.2b.
- **Depends on:** a shared timeline chart.

### C5 · Light from surfaces
- **Teaches:** emissive surfaces light the room through the cache, with no light object at all.
- **Scene:** a new procedural room (`addRoomShell`) lit only by emissive strips and a screen, with no lights.
- **On screen:** GI only, or the full picture.
- **Controls:** the strips' colour and brightness; the screen's content (two textures).
- **Explainer:**
  - Probe rays that hit an emissive surface take its emission.
  - What that costs.
  - Limits: the neural tail can't light from emissive surfaces (D21). Small, very bright emitters are noisy.
- **Depends on:** nothing new.

### C6 · The cache switcher (compare and debug)
- **Teaches:** what each world cache does differently, its pros and cons, and how to see it working. This replaces a separate debug-views page.
- **Scene:** `two-room` by default, with a scene picker that adds `courtyard`. The difference between closed rooms and open air is the lesson.
- **On screen:**
  - One variant at a time, picked from `RADIANCE_CACHE_VARIANT_PRESETS`: grid over DDGI, grid alone, SHaRC, budgeted cells, neural tail, plus RC3.6's two.
  - A **card** per variant:
    - how it works (two or three sentences and a diagram);
    - pros and cons;
    - P6.5's measured numbers: bias against Cycles, memory, frames to follow a change. Source: `docs/design/radiance-cache-variants.md`;
    - what it needs from the host (a DDGI volume, `shader-f16`, the optional MLP).
  - **Live readouts:** GPU ms per pass, memory, live cells.
  - **Debug overlays:** the probe atlas, the hash-grid cells in world space coloured by sample count, SHaRC's cells, and the screen-probe mask.
- **Controls:** variant; scene; overlay; "explain this pass" (hover a pass in the timing list for one line on what it does).
- **Depends on:** #52's per-pass timing (merged), the variant presets (merged) and the debug views (exist). The world-space cell overlay is new.

## Together (X1–X2)

### X1 · GI and reflections
- **Teaches:** GI here is diffuse light. Reflections are a separate layer on top. The composite counts each light path once, so GI and reflections don't add up twice. This is DECISIONS' "reflections layer on top of DDGI".
- **Scene:** a new procedural room (`addRoomShell`) with a polished floor, a matte wall and a lamp.
- **On screen:** diffuse GI only; reflections only; both.
- **The reflection source, picked live:**
  - three's screen-space reflections (the TSL `SSRNode` in three 0.187's addons);
  - the environment map as the fallback where SSR has no data.
- **Explainer:**
  - What GI covers and what it doesn't: glossy reflections (#31, D8).
  - How the composite splits the diffuse share (`1 − F`) from the specular.
  - Why SSR misses off-screen objects.
  - What comes later: RT reflections, the first technique after v1 (DECISIONS), and Split RC's own rough reflections, which run on the Mac.
- **Open question:** whether to add a traced-reflection option from `@pmndrs/raytracer` here. It shows the off-screen fix, but it is a bigger page. **Proposal:** SSR plus the environment first; traced reflections as X1b once RT reflections exist.

### X2 · Scale
- **Teaches:** how each method covers a big world.
  - DDGI: several volumes, with probe density by distance.
  - Cache: no bounds at all, with cells that grow with distance.
- **Scene:** a large procedural world: a grid of courtyards and blocks on `open-ground`, about 200 m across. It is generated, not shipped.
- **On screen:**
  - DDGI with near and far volumes (`DDGISettings.volumes`), with the volumes' outlines;
  - the cache, with its cell size shown by distance;
  - memory and GPU ms for each, as you fly.
- **Controls:** method; fly camera; the volumes' count and spacing; "show volumes" and "show cells".
- **Explainer:**
  - Volume placement and the cost per probe.
  - How the hash grid's levels of detail work.
  - Memory: DDGI 12–55 MiB, against the grid's ~1.1 GiB (P6.8 shrinks it) and SHaRC's 83 MiB. Sources: P6.5 and P6.8.
- **Depends on:** DDGI's multi-volume path (exists). P6.8 changes the grid's memory number.

## DDGI (01–10)

No new pages. Each gets the explainer panel ("What you're seeing", "How it works" and "Try this"). That is one small task across all ten.

## Shared pieces (built first, by one agent)

| Piece | What it is |
|---|---|
| `shared/explainer.ts` | The panel: sections, a numbered "Try this" list, a pros/cons table with sources, and an optional inline SVG. |
| `shared/wipe.ts` | A split wipe between two techniques on one view, from the viewer's version. |
| `shared/timeline.ts` | A small brightness chart over the last N frames, with change markers. |
| `shared/pass-timing.ts` | GPU ms per pass, with one-line descriptions per pass. |
| The examples index | Grouped by method, with the "which method" intro. Folders move to `ddgi/`, `cache/` and `together/`. |

## Order and agents

1. **After PKG.1 merges** (the examples import the moved packages): one agent builds the shared pieces, moves the folders and adds the DDGI explainer panels.
2. **Then one agent per page,** in parallel. Each brief is this page's section plus "What every page has":
   - C1, C3 and C4 first: they make the case for the cache;
   - then C6, C2 and C5;
   - then X1 and X2.
3. **Each agent's done-check:**
   - lint, typecheck, unit tests, `examples:build` and `examples-site`;
   - a screenshot of the page;
   - every number in its explainer traced to its source.
4. **The lead reviews** each page's explainer text before its PR.

## Questions for Dennis

1. **Folders:** is the move to `examples/ddgi/`, `cache/` and `together/` with C- and X- numbers OK, or should we keep a flat 01–NN?
2. **X1's reflection source:** three's SSR plus the environment now, with traced reflections later as X1b. Or do you want `@pmndrs/raytracer` in the first version?
3. **X2's world:** is a generated block world OK, or do you want a specific layout?
4. **The DDGI explainer pass on 01–10:** include it now, or leave it for later?
