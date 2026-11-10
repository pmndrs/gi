# Changelog

What each release of `@pmndrs/ddgi` contains, newest first. Each version has a `## X.Y.Z` section, written by hand: the publish workflow uses it as the GitHub Release's notes and refuses to publish a version without one ([docs/releasing.md](docs/releasing.md)). Before tagging, replace "(unreleased)" with the date.

## 0.1.0 (unreleased)

The first release: real-time diffuse global illumination for WebGPU, with a three.js adapter. Light bounces through a grid of probes that trace rays every frame, and follows lights, objects and the sun as they change. This is a working prototype: every feature below works end to end and passes a check on a real GPU, but nothing has been tuned yet.

### Using it with three.js (`@pmndrs/ddgi/three`)

- **`GIVolume`**, shaped like three's `LightProbeGrid`: give it a box and a probe count, add it to the scene, `await volume.start(renderer, scene)`, and GI updates itself inside `renderer.render`. No per-frame call.
- **`createGIRenderer`** makes a `WebGPURenderer` on a device with the limits large scenes need.
- **The drop-in composite:** `ddgi(scenePass, volume)` with the `giAlbedo` output adds GI once per pixel on three's post-processing pipeline, the same way as three's `ssgi()` and `ao()`, and works with `@pmndrs/upscaler` and three's GTAO.
- **Your scene as it is:** meshes, instanced, morphed and skinned meshes, three's materials and textures (compressed ones too), and point, spot and directional lights and the environment. The adapter notices added, moved, changed and removed objects and lights on its own.
- **Quality presets:** `{ preset: 'low' | 'medium' | 'high' }`, and `recommendPreset` to pick one from the GPU.
- **Saved bakes:** `saveBake` and `loadBake` store the probes, so a page can open with finished GI. A bake made for a different scene is refused or settles in a few frames.
- **Export to three's `LightProbeGrid`:** `toLightProbeGrid` turns the live GI into a probe grid three can render without this package, with a JSON format to store it.

### The engine-free platform (`@pmndrs/ddgi/core`)

- **Plain WebGPU on your own `GPUDevice`,** with no engine: a copy of the scene on the GPU, our own ray tracer and the GI technique on top. The three.js adapter is built on it; [docs/guides/adapter.md](docs/guides/adapter.md) shows how to write another.
- **Ray tracing:** trees built on the GPU for first GI right away, then tinybvh's (in a worker, compiled to WebAssembly) swapped in; meshes added at runtime are traceable from the next frame. Works with and without the GPU's subgroup feature.
- **Hit shading:** sun, sky, point and spot lights with shadows, emissive surfaces, hundreds of lights (a light grid picks the ones that matter). Adding or removing lights never rebuilds a shader.
- **Materials at ray hits:** albedo, emissive and metalness from textures, glass and other see-through surfaces, single-sided surfaces, alpha cut-outs and blended surfaces.

### DDGI

- Probes in a grid that each store the light arriving from every direction and how far they can see, so light leaks far less through walls. Rotated rays, blending with history, unlimited bounces, probes that move out of walls and switch off inside them.
- **A fixed GPU budget per frame** (2 ms by default), with the probes nearest the camera and nearest a change updated first, and every probe still updated within a few seconds.
- **Changes reach the GI quickly, with no blackout:** a moved, dimmed, recoloured or switched light, a moved, added or removed object, a deforming mesh, or a turning sun re-traces only the probes it affects, and they blend from their old answer.
- **Light groups** dim or recolour a group of lights in GI in the same frame. Time of day re-bakes the sky on the GPU.
- **Several volumes,** a dense one inside a coarse one with no visible seam, and a **camera volume** that moves with the viewer.
- Debug outputs: the probes, their states, the irradiance and distance atlases.

### Checked, and not yet checked

- Checked on an RTX 3060 Laptop GPU (Windows, Chrome) and an Apple M5 Pro (macOS, Chrome), and for correctness on an Intel UHD integrated GPU (Windows, Chrome). The milestone notes in [docs/milestones/](docs/milestones/) list every check and its result.
- Not yet run: Safari, an iPad or an older iOS device, and the `low` preset on the slow devices it is meant for.
- Known limits, accepted for this release: the composite path draws no GI behind glass (the per-fragment path does); on Sponza, GI misses the image-difference (FLIP) bar against the path-traced reference and the curtains read brighter than it; large scenes like Sponza take longer than the 2 s start-up target. The full list is in the milestone notes.
