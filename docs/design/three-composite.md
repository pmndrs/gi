# T1.14: GI for three, the simple path and the composite

Status: design by the M1 lead, 2026-11-05. Accepted direction and constraints: DECISIONS 2026-11-05 ("T1.14, the additive GI composite"). The API names marked **(to confirm)** face the host, so Dennis confirms them before the implementer builds them.

## What three r187 already does (scout, 2026-11-05)

These are three's own patterns, read from the r187 examples. Our job is to fit them, not to sit beside them.

- **LightProbeGrid** (`webgpu_lightprobes.html`): 5 lines of GI code. `new LightProbeGrid(w, h, d, nx, ny, nz)`, `position.set`, `scene.add`, `bake(renderer, scene, options)`. Materials stay plain `MeshStandardMaterial`. It is a `Light` with a registered `AnalyticLightNode` that writes `context.lightProbeIrradiance`. The renderer finds it because it's in the scene. Its node updates itself on render (`updateType = RENDER`), so there is no per-frame call.
- **SSGI** (`webgpu_postprocessing_ssgi.html`): about 10 lines. `new RenderPipeline(renderer)`, then `pass(scene, camera)` with `mrt({ output, diffuseColor, normal, velocity })`, then `ssgi(...)`. The composite is additive in linear HDR, before the pipeline's tone mapping: `scenePassColor.rgb × ao + scenePassDiffuse.rgb × gi`. So three itself composites indirect light as plain `diffuseColor × GI`.
- **GTAO and VXGI**: `scenePass.contextNode = builtinGIContext(ao, gi)` (or `builtinAOContext`) feeds a per-pixel texture into the materials' own lighting, so three's BRDF factors apply. But the scene pass needs the texture before it draws, so GTAO renders the scene twice (a prepass), and VXGI computes its GI ahead of time.
- **Display nodes** (`SSGINode`, `GTAONode`): `class extends Node`, `updateBeforeType = FRAME`, an own render target, `updateBefore(frame)` draws a quad, `setup` returns a `passTexture`, and a factory function (`ssgi(...)`, `ao(...)`).
- Nothing in r187 removes double counting: `scene.environment`, light probes and `getGI` all add to `context.irradiance`.
- three's `WebGPURenderer` asks for every adapter feature but default limits (128 MiB storage bindings). Small scenes fit. Sponza with curtains needs the raised limits (NOTES).

## The simple path: as short as LightProbeGrid

What a user writes today (from `three-gi.ts` and `gi-volume.ts`): `createGIRenderer`, `createGI(device)`, `new GIVolume`, `scene.add`, `updateMatrixWorld`, `new DDGITechnique({ volume: volume.ddgiVolume() })`, `addTechnique`, `setOutput`, `await prepare()`, `new ThreeGI(gi, scene, { volume })`, and `threeGI.update(camera, dt)` every frame. That's about 10 lines and five of our names, against LightProbeGrid's 5 lines and one name. That is the friction Dennis means.

**Proposal (to confirm):**

```js
const volume = new GIVolume(6, 5, 6, 6, 6, 6);   // same arguments as LightProbeGrid
volume.position.set(0, 2.5, 0);
scene.add(volume);
await volume.start(renderer, scene);              // like probes.bake(renderer, scene)
```

- `start` takes three's device from `renderer.backend.device`, makes the platform and the DDGI technique from the volume, and runs `prepare`. It also runs the viewer's warm-up (full passes with a running average, time-capped; `apps/viewer/src/techniques/ddgi-technique.ts`), moved into the adapter. Without it a library user starts from black, and on Sponza a probe updates only about every 73 frames. Options (budget, sky, environment, DDGI settings) go in an optional second argument. Same 4–5 lines as LightProbeGrid.
- **No per-frame call:** the GI light's node runs the GI frame in its render-time update, as LightProbeGridNode updates itself. The implementer must check the command order there: GI's submit has to land before the render's encoder is submitted, which r187 does at `finishRender`. If that order can't be guaranteed, `volume.update(camera)` stays as one explicit line, and the usage report says so.
- **Limits:** if the scene doesn't fit three's default limits, `start` throws `GILimitError` naming the limit, with the fix: `new WebGPURenderer({ requiredLimits: giRequiredLimits(adapter) })`, or `createGIRenderer()` as now.
- `createGI`, `DDGITechnique` and `ThreeGI` stay public for hosts that want control (several techniques, a shared platform). The simple path doesn't need them.

## The fast path: a composite node on three's pipeline

For dense scenes (Sponza's sub-pixel triangles), the user adds three's pipeline and one node, as SSGI does:

```js
const pipeline = new THREE.RenderPipeline(renderer);
const scenePass = pass(scene, camera);
scenePass.setMRT(mrt({ output, giAlbedo, normal: directionToColor(normalView) }));   // giAlbedo from us
pipeline.outputNode = ddgiComposite(scenePass, volume);                              // (to confirm)
```

- **Route: additive, SSGI-style, one scene pass.** `ddgiComposite` reads the pass's `output`, `giAlbedo`, normal and depth, evaluates `giIrradiance` once per pixel from the world position (from depth) and the normal, and returns `output + giAlbedo × E / π` in linear HDR. The pipeline then tone maps as usual. Measured on Sponza at 1440p (low clocks): MRT plus a screen pass +3.9 ms, against +12.0 forward.
- *Not chosen: `builtinGIContext`.* It applies three's BRDF exactly, but the scene pass needs the GI texture first, so it needs a prepass (+6.5 ms measured on Sponza) or last frame's GI with reprojection.
- **`giAlbedo`, an MRT node we export.** The diffuse weight three's lighting applies to indirect light: `diffuseColor × (1 − F_ss − F_ms) × AO`, built from three's own TSL pieces (the material's `diffuseColor`, `specularColor`, `roughness`, the view normal, the AO map). Using plain `diffuseColor` works too, as three's SSGI example does, but it's about 4% brighter at normal incidence and more at grazing angles, which would fail the match check. The implementer reads `PhysicalLightingModel.indirect*` in r187 and computes the same factor.
- **No double counting.** `ddgiComposite` marks its scene pass through the pass's `contextNode`, merging with any context the user already set (GTAO's `builtinAOContext`, for example). Under that mark, the GI light adds no indirect diffuse for opaque materials, and the environment-diffuse workaround still runs per fragment (it's cheap). Transparent materials, other passes and cube captures see no mark and keep the per-fragment path.
- **Node shape:** like `SSGINode`. `class DDGICompositeNode extends TempNode` (or `Node`), with a factory `ddgiComposite(scenePass, volume)`. It needs no render target of its own: it's a pure function of the pass textures, so the pipeline's own quad evaluates it.
- **Measured as "cost of showing GI":** the MRT's extra cost (pipeline with `giAlbedo` and normal, against the pipeline without) plus the composite. GPU-bound, ≤ 1.0 ms on Sponza at 1440p.

## The usage bar

`examples/three/` has two pages next to three's `webgpu_lightprobes.html`, on the same scene setup:
- `simple.html`: `GIVolume` alone;
- `composite.html`: the same, plus the pipeline lines.

The task reports in PROGRESS:
- the GI line counts for LightProbeGrid, ours simple, and ours with the composite;
- every name a user touches;
- anything still harder than LightProbeGrid, such as raised limits for large scenes, or the per-frame line if the render-time update can't be made safe.

## Checks

- The composite against the per-fragment path, on Sponza and Cornell at the stored cameras: per-pixel relative luminance difference mean ≤ 0.5%, p95 ≤ 2%, over opaque pixels, HDR before tone mapping (DECISIONS).
- Cost of showing GI ≤ 1.0 ms, Sponza, 1440p, GPU-bound on the RTX 3060; Intel correctness only.
- The simple path renders the same as before T1.14 (the T1.7 neutrality and environment pages still pass).
- Transparent objects in a composite scene still get GI (per fragment).
- A unit test that the mark composes with an existing `contextNode`.

## Files

- `packages/ddgi-three/src/`: `gi-volume.ts` (`start`), `gi-light.ts` (render-time update, the composite mark), new `composite.ts` (`ddgiComposite`, `giAlbedo`), `index.ts`.
- `examples/three/` (new).
- `tools/gpu/pages/three-composite/` (new): the match and cost checks.
- The viewer gets a "GI composite" toggle afterwards, once Budget v2 and T1.13 are out of `apps/viewer`.
