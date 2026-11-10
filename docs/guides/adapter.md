# Writing an engine adapter

An adapter connects a renderer to the GI platform: three.js, Babylon, a game engine, or a renderer of your own. This guide walks through the smallest adapter we have, the plain-WebGPU **hello example** in [`examples/webgpu/hello/`](../../examples/webgpu/hello/), one step at a time. Read it next to the code:

- [`src/main.ts`](../../examples/webgpu/hello/src/main.ts): the device, the platform and the frame;
- [`src/scene.ts`](../../examples/webgpu/hello/src/scene.ts): the scene copy (meshes, materials, textures, lights, sky);
- [`src/renderer.ts`](../../examples/webgpu/hello/src/renderer.ts) and [`src/shaders.ts`](../../examples/webgpu/hello/src/shaders.ts): sampling GI while shading.

Run it with `pnpm --filter @pmndrs/ddgi-example-hello dev`. The three.js adapter ([`packages/ddgi-three/src/`](../../packages/ddgi-three/src/), described in [docs/api/three.md](../api/three.md)) does the same jobs for a whole engine. Every call used here is described in [docs/api/platform.md](../api/platform.md).

## What an adapter does

Four jobs, and nothing else is required:

1. **Share the device.** You make the `GPUDevice`. The platform uses it, and never makes, configures or destroys one.
2. **Describe the scene.** You give the platform a simplified copy of your scene (meshes, materials, textures, object placements, lights and the sky), then send only what changes.
3. **Run the frame.** Once per frame, the platform records its GPU work into your command encoder, before your own passes.
4. **Read GI while shading.** The platform hands you a small piece of WGSL and the textures it reads. Your fragment shader calls it.

A few words used below:

- **The platform** is the engine-free part of the library (`createGI`): its own copy of the scene, ray tracing through it, and the GI techniques on top.
- **A technique** is a GI method running on the platform. DDGI is the first.
- **A probe** is a point that stores the light arriving there from every direction. DDGI keeps a 3D grid of them and updates a few each frame.
- **A handle** is the number the platform gives back for each thing you create. You pass it to change or destroy that thing later.

The platform works in metres, right-handed with +Y up, linear-light colours, and column-major 4 × 4 matrices (the order of three.js's `Matrix4.elements` and WGSL's `mat4x4f`). Convert once in your adapter if your engine differs.

## Step 1: the device and its limits

From `main.ts`:

```ts
import { optionalFeatures, requiredLimits } from '@pmndrs/ddgi/core';

const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
if (!adapter) throw new Error('No WebGPU adapter found.');
const device = await adapter.requestDevice({
    requiredLimits: requiredLimits(adapter),
    requiredFeatures: optionalFeatures(adapter),
});
device.addEventListener('uncapturederror', (event) => console.error('WebGPU error:', event.error.message));
```

**Ask for the adapter's limits, not WebGPU's defaults.** A WebGPU device starts with small limits unless you ask for more. The one that matters most is the largest storage buffer a shader can bind: 128 MiB by default. The platform keeps each table of the scene in one buffer, and a big scene outgrows that: the ray tracer's triangle records take about 276 MB for Sponza with its curtains and 136 MB for Bistro's exterior. `requiredLimits(adapter)` asks for the adapter's own maximum binding and buffer sizes (about 2 GiB on the laptop GPUs we test), every storage buffer per shader stage it has, the most texture layers, and the compute limits the kernels need. It never asks for more than the adapter has, so `requestDevice` can't fail because of it. Don't plan your adapter, or your scenes, around the 128 MiB default.

If a scene still needs more than the device allows, the platform throws a `GILimitError` that names the limit, the size it needed and what the device has. It never fails silently.

**If your engine makes the device**, give it these limits. three.js takes a device you made (`new WebGPURenderer({ device })`) or `requiredLimits` in its options; the three adapter's `createGIRenderer` does this. Babylon takes them in its `deviceDescriptor`.

**`optionalFeatures(adapter)`** lists features worth having, where the adapter has them: `subgroups` (faster kernels; deforming meshes need it), `timestamp-query` (GPU timings, which a budget in milliseconds needs), `float32-filterable`, `texture-formats-tier1` (half the memory for one of DDGI's textures), and texture compression (`texture-compression-bc`, `-etc2`, `-astc`) so compressed textures reach GI.

**Errors and loss:**

- Listen for errors with `addEventListener('uncapturederror', ...)`. Never assign `device.onuncapturederror`: your engine may already use it (three does), and assigning replaces its handler.
- If the device is lost, `gi.isReady` turns false and every call throws `GINotReadyError` with the reason `'device-lost'`. Make a new device and a new platform on it.
- When your page or view goes away, `gi.dispose()` and then `device.destroy()`. The hello example destroys its device on `pagehide`, so the GPU is freed promptly.

## Step 2: the platform

```ts
import { createGI } from '@pmndrs/ddgi/core';

const gi = createGI(device, { gpuTiming: device.features.has('timestamp-query'), label: 'hello' });
```

- `gpuTiming` fills `gi.stats.passes` with each GPU pass's time, where the device has `timestamp-query`.
- `label` prefixes the platform's GPU objects, so a GPU capture shows whose they are.
- `textureLayerSize` (256, 128 or 64) sets the size of the platform's copy of each texture. The quality presets set it; see [platform.md, "Quality presets"](../api/platform.md#quality-presets).

The platform starts empty. Everything below goes through `gi.scene`, and nothing reaches the GPU until the first `gi.encode`.

## Step 3: the scene copy, meshes and placements

The platform keeps its own copy of your scene: only what GI's rays need to see. A **mesh** is a shape. An **instance** is one placement of a mesh in the world, with its own transform and materials. Many instances can share a mesh.

From `scene.ts` (`data` is the scene the hello example builds in code; your adapter reads its engine's objects instead):

```ts
const meshes = data.meshes.map((mesh) =>
    gi.scene.createMesh({
        positions: mesh.positions, // xyz per vertex, in the mesh's own space
        indices: mesh.indices, // a triangle list
        normals: mesh.normals, // optional: smooth shading where rays hit curved surfaces
        uvs: mesh.uvs, // needed for textures and alpha cutouts
        groups: mesh.groups, // optional: which material slot each run of triangles uses
        label: mesh.name,
    }),
);
```

```ts
for (const instance of data.instances) {
    // One material per slot that the mesh's groups use. A wall with a glass pane has two.
    const slots = instance.materials ?? [instance.material];
    gi.scene.createInstance({
        mesh: meshes[instance.mesh],
        materials: slots.map((slot) => materials[slot]),
        transform: instance.transform, // 16 numbers, column-major, local to world
        label: instance.name,
    });
}
```

What to know:

- **Indices:** a triangle list, 16- or 32-bit. A mesh without an index buffer gets `0, 1, 2, ...` from the adapter.
- **Groups** are runs of triangles (`start` and `count` in indices, a `materialSlot`). Pass every slot your engine's mesh uses: the instance's `materials` list is indexed by slot. Without `groups`, the whole mesh uses slot 0. (Until M5 the hello example passed slot 0 only, so a glass pane in a wall blocked light like the wall.)
- **A second uv set** (`uvs1`) is for textures that read uv channel 1. Send it only when a material uses it: on Sponza it costs 16 MB.
- **Objects that should only receive GI.** Characters and small props usually shouldn't block or bounce light for GI. Create them with `contributesToGI: false`: they still receive GI through your shading, and the platform still knows where they are, so DDGI wakes the probes around them as they move.
- **Hidden objects:** `visible: false`, or `setVisible` later (step 6). A hidden instance keeps its data and costs rays nothing.

## Step 4: materials and textures

The platform has a small material model on purpose. Each adapter maps its engine's materials onto these ideas, and nothing else about a material reaches GI. Where a ray hits a surface:

- **albedo** = `albedo` × the albedo texture × (1 − `metalness` × the metalness texture's channel). A metal has no diffuse colour, so it bounces no diffuse light: pass `metalness` and let the platform apply it.
- **emission** = `emissive` × `emissiveIntensity` × the emissive texture, as radiance (the same units as lights, step 5).
- **alpha** = `opacity` × the albedo texture's alpha. What alpha does to rays is the `alphaMode`:
  - `'opaque'`: nothing; every hit stands.
  - `'cutout'`: alpha below `alphaCutoff` is a hole (foliage, fences; glTF `MASK`). With `alphaCutoff` above 0 and no mode, the mode is `'cutout'`.
  - `'stochastic'`: a ray passes with probability 1 − alpha, so many rays see the right average of a see-through surface (decals; glTF `BLEND`).
- **`blocksLight: false`** lets light straight through: glass. The surface neither shadows nor bounces.
- **`doubleSided`** defaults to true, and should usually stay true even when your engine draws one side only. Rays that hit the back of a wall are how probes find out they are inside it.

Roughness, normal maps and clearcoat don't matter: DDGI's light is diffuse, and their detail is far below a probe's size.

**Textures** are created once and shared by materials. From `scene.ts`:

```ts
const textures = (data.textures ?? []).map((texture) =>
    gi.scene.createTexture({
        source: { data: texture.data, width: texture.width, height: texture.height }, // 8-bit RGBA pixels
        srgb: texture.srgb, // true for colour images, which are sRGB-encoded
        label: texture.name,
    }),
);
```

A texture source can be an `ImageBitmap`, an image or canvas element, an `OffscreenCanvas`, `ImageData`, raw 8-bit RGBA pixels (as above), a `GPUTexture` from the same device, or compressed pixels:

```ts
const bricks = gi.scene.createTexture({
    // What three's CompressedTexture.mipmaps holds after KTX2Loader: the format and its mip levels, largest first.
    source: { format: 'bc7-rgba-unorm-srgb', levels: compressedLevels },
    srgb: true,
});
```

The platform makes its own small copy of every texture (a 256 × 256 layer with mips, by default) in the first `encode` after `createTexture`, so you may free the source after that. A compressed texture needs its format's device feature (`optionalFeatures` asks for it); without it the platform warns once and those materials use their flat colour. Until a texture's copy is made, its materials shade as if it were white.

**Where a texture is sampled** is its `UvMapping`: the uv set (`channel` 0 or 1), a 3 × 2 matrix applied to the uv, and the wrap on each axis. Texture coordinate (0, 0) is the source's first texel, the top-left of an image, as in WebGPU. An engine that flips images when it uploads them (three's `flipY`) folds that flip into the matrix. The hello example's pixels already start at v = 0, so it needs no flip.

Then the materials. A few kinds, as the hello example's scenes have them:

```ts
const wall = gi.scene.createMaterial({
    albedo: [0.8, 0.8, 0.8],
    albedoTexture: textures[0],
    albedoUv: { channel: 0, matrix: [2, 0, 0, 2, 0, 0], wrapU: 'repeat', wrapV: 'repeat' }, // the texture twice across
    alphaMode: 'cutout',
    alphaCutoff: 0.5,
});
const glass = gi.scene.createMaterial({ albedo: [1, 1, 1], blocksLight: false });
const steel = gi.scene.createMaterial({ albedo: [0.6, 0.6, 0.6], metalness: 1 });
const lampShade = gi.scene.createMaterial({ albedo: [0.9, 0.85, 0.7], emissive: [1, 0.8, 0.5], emissiveIntensity: 4 });
const decal = gi.scene.createMaterial({ albedo: [0.3, 0.25, 0.2], albedoTexture: bricks, opacity: 0.6, alphaMode: 'stochastic' });
```

The hello example's own version is `toPlatformMaterial` in `scene.ts`. Its unit test (`scene.test.ts`) holds it to the repository's full scene description, so no field of the scene data is lost on the way to GI.

## Step 5: lights and the sky

The platform's light units are three.js's physical ones, so a three adapter passes values straight through and other adapters convert once:

- a directional light's `intensity` is the irradiance it gives a surface facing it (lux);
- a point or spot light's `intensity` is its radiant intensity (candela); its light falls off as 1 / distance², reaching zero at `range`;
- emissive surfaces and the sky are radiance.

```ts
const lamp = gi.scene.createLight({ type: 'point', position: [2.78, 5.0, 2.8], color: [1, 0.9, 0.8], intensity: 40, range: 0, castsShadows: true });
const sun = gi.scene.createLight({ type: 'directional', direction: [-0.3, -1, -0.2], color: [1, 0.95, 0.9], intensity: 3, group: 1 });

gi.scene.setEnvironment({ type: 'color', color: [0.2, 0.25, 0.3], intensity: 1 });
// Or a clear sky made on the GPU, which follows the sun cheaply (time of day):
gi.scene.setEnvironment({ type: 'procedural', sunDirection: [0.3, 1, 0.2], turbidity: 2, sunIlluminance: 3 });
```

- **`range` is required** for point and spot lights, so that limiting a light is a choice. `range: 0` means no limit (three's `distance = 0`): such a light reaches every probe, so it costs every probe.
- **`castsShadows`** defaults to true. A light without GI shadows is cheaper, but its light leaks through walls into GI.
- **`group`** (0 to 3) puts the light in a light group, so DDGI can dim or recolour the whole group instantly ([platform.md, "Lights and light groups"](../api/platform.md#lights-and-light-groups)).
- **The sky** is what rays see when they hit nothing: `'none'`, a `'color'`, a `'texture'` (a cube map or an equirectangular image on the GPU), or `'procedural'`. Note the two directions: a directional light's `direction` is the way its light travels, and the procedural sky's `sunDirection` points toward the sun, so they have opposite signs. Keeping the sun light and the sky in step is your adapter's job.
- **Direct light stays your renderer's job.** GI is the light that bounced at least once, plus light from emissive surfaces. Your renderer still draws each light's direct light and its shadow maps, as the hello renderer does for its point light.

The hello example converts its scene's lights with `toPlatformLight` in `scene.ts`. Its renderer draws only one point light itself, which is why it shows only scenes with at most one point light.

## Step 6: telling the platform about changes

The hello example's scene doesn't change, but most scenes do. Every scene call is cheap on the CPU: the platform queues the change and uploads it in the next `encode`, and several calls between two frames are merged. You don't tell DDGI anything yourself: the platform works out where each change happened (a box around where an object was and one around where it is now) and DDGI updates the probes near it first.

```ts
gi.scene.setTransform(instance, matrix); // it moved
gi.scene.setTransforms(movedInstances, movedMatrices); // many moved: handles in a Uint32Array, 16 numbers each
gi.scene.setVisible(instance, false); // hidden or shown
gi.scene.updateMaterial(wall, { albedo: [0.9, 0.2, 0.2] }); // only the fields given change
gi.scene.setMaterials(instance, [steel]); // other materials on the same mesh
gi.scene.updateLight(lamp, { intensity: 10 }); // any field but its type
gi.scene.destroyInstance(instance); // removed; create new ones at any time
```

- **Moves** of many objects at once should go through `setTransforms`: one call, with no per-call bookkeeping (10,000 objects sliding cost 1.4 ms of CPU in Chrome on the RTX laptop).
- **A new shape:** `updateMesh(mesh, descriptor)`. Its ray-tracing tree is rebuilt, and rays use the old one until the new one is ready.
- **The sky:** call `setEnvironment` again. The sky is read when it is set, so after drawing into a sky texture, set it again (the same object is fine).
- **Adding and removing** is just creating and destroying. Tables grow by reallocating buffers, never by changing shader code, so nothing recompiles: `gi.stats.recompiles` stays 0.

**Deforming meshes** (cloth, foliage in the wind, skinned characters that block light) keep their triangles and change their vertex positions every frame:

```ts
const flag = gi.scene.createMesh({
    positions: restPositions,
    indices: flagIndices,
    deforming: true,
    deformBounds: { min: [-1, 0, -0.5], max: [1, 2, 0.5] }, // in the mesh's own space; every pose stays inside
});
gi.scene.setDeformedPositions(flag, posedPositions); // on the CPU: copied at the call
gi.scene.setDeformedPositions(flag, { buffer: skinnedVertices, stride: 32 }); // or on the GPU: copied in the next encode
```

- Rays see a pose in the next frame: the platform refits the mesh's tree on the GPU. This needs the `subgroups` feature; without it the mesh stays in its rest pose, with one warning.
- `deformBounds` is a box every pose stays inside. It is needed for positions given on the GPU, which the platform can't read. Positions given on the CPU outside the box are still traced correctly: the box grows, with one warning.
- Positions on the GPU are a buffer with `STORAGE` usage (xyz as floats at the start of each vertex, any stride of 12 bytes or more), or an `rgba32float` texture with `COPY_SRC` usage (vertex `i` at texel `(i mod width, i / width)`, the width a multiple of 16).

**How to find what changed** is engine-specific, and is most of a real adapter's work. Keep a mirror of what you last sent and compare, or hook your engine's change events. Sending every transform every frame works, but costs CPU in big scenes. The three adapter listens for added and removed objects, hooks each mesh's world-matrix update to notice moves, and checks a fixed number of other objects each frame ([three.md, "How changes are found"](../api/three.md#how-changes-are-found-and-how-late)).

## Step 7: add DDGI and prepare

From `main.ts`:

```ts
import { DDGITechnique } from '@pmndrs/ddgi/core';

const ddgi = gi.addTechnique(new DDGITechnique({ volume: { center: [2.78, 2.74, 2.8], size: [5.6, 5.5, 5.6], spacing: 0.5 } }));
gi.setOutput(ddgi); // the technique your shading reads
await gi.prepare(); // compiles every kernel, asynchronously
```

- **The volume** is a box of probes: its centre and size in metres, and the distance between probes (`spacing`). There are `floor(size / spacing) + 1` probes along each axis, centred in the box. Probes inside walls switch themselves off, so the box may cover walls freely. Several volumes, and one that follows the camera, are in [platform.md, "Several volumes"](../api/platform.md#several-volumes-and-the-camera-following-volume).
- **`prepare` compiles everything once.** After it, frames and scene edits compile nothing. Calling `encode` before `prepare` resolves throws `GINotReadyError`. You may describe the scene before or after `prepare`.
- Settings that size DDGI's buffers (the volumes, the tile sizes, light groups, the sun layer) need `prepare` again when changed. The rest, such as the hysteresis, are read every frame.

## Step 8: the frame

From `main.ts`, once per frame:

```ts
const encoder = device.createCommandEncoder({ label: 'hello frame' });
// GI first: this frame's probe updates, within 2 ms of GPU time.
gi.encode(encoder, {
    deltaTime: (time - previousTime) / 1000, // seconds since the last frame
    budget: { gpuMs: 2 },
    camera: { position: matrices.position, view: matrices.view, projection: matrices.projection, near: matrices.near, far: matrices.far },
});
renderer.render(encoder, matrices, view); // then your own passes, which read GI
device.queue.submit([encoder.finish()]);
```

- **Order:** `encode` comes before the passes that shade with GI, in the same encoder, so this frame's updates land before pixels read them. One submit, yours.
- **No readbacks frame after frame.** The platform submits work of its own and reads a few numbers back only in frames that upload new meshes (to build their first trees), and occasionally for deforming meshes.
- **The budget** says how much GPU work GI may do each frame. `gpuMs` needs GPU timings (`gpuTiming` on a device with `timestamp-query`); the platform measures once how many probe updates fit in that time and hands out that many. Without timings, give `units` (probes per frame, 512 say). `targetFps` tells it the frame rate you aim for; `paused: true` skips the updates and keeps the last GI.
- **The camera** is plain data: its position, its world-to-view matrix, its projection without jitter (an upscaler's jitter would shake GI's priorities), and near and far. DDGI updates probes the camera sees first, and a camera-following volume follows it.
- `gi.resetHistory()` forgets the accumulated GI (a scene swap). `gi.recalibrate()` measures the budget again after a big change.

## Step 9: sampling GI in your shaders

`gi.shading` is the shading module: what your renderer reads while it shades a pixel. Every technique publishes the same shape, so changing the technique doesn't change your adapter. It has:

- `code`: WGSL with the entry points, taking their resources as parameters;
- `parameters`: those resources, in order: textures, samplers (each belonging to one texture) and `vec4f` uniforms. Only textures and small uniforms, because some engines can't bind another library's buffers in a fragment shader;
- `globalsCode(group)`: the same code with the resources declared as globals at the bind group you choose, plus simple wrappers;
- `writeUniforms(target)`: fills the `vec4f` uniforms for the globals path;
- `version`: goes up when the code or the parameter list changes shape. Rebuild your pipeline then.

**A renderer that binds globals** (the hello example) puts `globalsCode(1)` into its shader and builds the matching bind group layout. Bindings go in the parameters' order from 0: every texture and sampler, then one uniform buffer if there are `vec4f` parameters. From `renderer.ts`:

```ts
const entries: GPUBindGroupLayoutEntry[] = [];
for (const parameter of gi.shading.parameters) {
    if (parameter.kind === 'texture') {
        entries.push({ binding: entries.length, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: parameter.sampleType, viewDimension: parameter.dimension } });
    } else if (parameter.kind === 'sampler') {
        entries.push({ binding: entries.length, visibility: GPUShaderStage.FRAGMENT, sampler: { type: parameter.filtering === 'linear' ? 'filtering' : 'non-filtering' } });
    }
}
const hasUniforms = gi.shading.parameters.some((parameter) => parameter.kind === 'vec4f');
if (hasUniforms) entries.push({ binding: entries.length, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } });
const giLayout = device.createBindGroupLayout({ label: 'gi', entries });
const shaderCode = gi.shading.globalsCode(1) + myFragmentShader; // GI's group is 1
```

Each frame, before drawing:

- call `writeUniforms` and upload the uniforms, if there are any;
- read each texture parameter's `texture()` again, and make a new bind group only when one has changed (DDGI replaces a texture only when it grows);
- if `gi.shading.version` changed, build the pipeline again with the new code.

Then, in the fragment shader (`shaders.ts`):

```wgsl
// From the eye towards the point, as the shading module expects.
let viewDirection = normalize(position - cameraPosition);
// Irradiance E arriving at the surface, not E / pi.
var irradiance = directIrradiance + giIrradianceAt(position, normal, viewDirection, screenUV);
// Where GI isn't sure of its answer (a volume's edge), fade to your own environment light.
irradiance += (1.0 - giCoverageAt(position, normal)) * environmentIrradiance;
let color = emission + albedo / PI * irradiance;
```

- `normal` is the side of the surface you see. `screenUV` is the pixel's position in 0 to 1; DDGI ignores it.
- `albedo` here is the diffuse colour, with metalness taken out, as in step 4. Specular reflections are your renderer's own business; GI's answer is diffuse.
- **Don't count the environment twice.** If your engine adds its own ambient or environment diffuse light, fade it by `1 − coverage` where GI covers the point. The three adapter does this for three's environment light.
- **Debug views:** `ddgi.debugViews` lists them; `giDebugColorAt(position, normal, id)` draws a screen view by id (GI only, coverage, probe states). The probe points view gives a texture of positions you draw as small spheres.

**An engine that declares shader resources itself** (three's `wgslFn`, Babylon's material plugins) uses `code` instead of `globalsCode`: declare each parameter its own way, in order, and call `giIrradiance(position, normal, viewDirection, screenUV, ...)` with them after the first four arguments. The three adapter builds this call in `entryWgsl` (`packages/ddgi-three/src/gi-light.ts`).

## Step 10: check your adapter

- **The same image as hello.** Describe one scene through both and render it: the images should match. The `three-neutrality` GPU page does this for three; on the Apple M5 Pro the two Cornell boxes' means differ by less than a millionth.
- **No recompiles:** `gi.stats.recompiles` stays 0 while you add, move and remove things.
- **The furnace:** a closed room with white walls lit only by its own glow comes out evenly lit. If it doesn't, albedo, emission or units are off (`ddgi-furnace`, and `ddgi-textured-furnace` for textures).
- **Stats:** `gi.stats` gives memory by part, each GPU pass's time, and DDGI's counters (`ddgi.counters`: probes updated, rays traced, ...).

## What else you can do

- **Quality presets:** `ddgiPreset('low' | 'medium' | 'high')` and `recommendPreset(adapter.info, adapter.limits)` ([platform.md](../api/platform.md#quality-presets)).
- **Light groups and the sun layer:** dim or recolour groups of lights, or the sun, in the same frame, with nothing traced again ([platform.md](../api/platform.md#lights-and-light-groups)).
- **Several volumes, and one that follows the camera** ([platform.md](../api/platform.md#several-volumes-and-the-camera-following-volume)).
- **A `LightProbeGrid` snapshot** of the GI, as three's own probe grid, for a static fallback: `exportIrradianceSH` ([platform.md](../api/platform.md#exporting-the-field-as-spherical-harmonics)).
- **Saved bakes:** save a settled field and load it later, `saveField` / `loadField` in the core and `saveBake` / `loadBake` in three (described with that feature's API).
