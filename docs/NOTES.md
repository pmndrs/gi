# Notes: facts worth not relearning

One bullet per fact, each with its source. Add a bullet when something cost you time. Fix or delete bullets that turn out wrong. Group by topic.

## three.js (r187)

- **Shared device:** `new WebGPURenderer({ device })` is supported. When three creates the device itself, it uses a compatibility adapter with default limits: a 128 MB storage-buffer binding and 8 storage buffers per stage. Source: `WebGPUBackend.js:72, 213-254`.
- **Device errors:** three assigns `device.onuncapturederror`. Use `addEventListener('uncapturederror')` and never assign it. Source: `WebGPUBackend.js:281`.
- **three caches node shaders by material properties plus light ids** (`RenderObject.getMaterialCacheKey`, `LightsNode.customCacheKey`), not by `material.version`. `needsUpdate` with an unchanged key reuses the old shader, so to change a light node's code, replace the light (new id). Source: `NodeManager.getForRender`, T1.7.
- **Two `wgslFn`s that include the same WGSL through separate `wgsl()` nodes emit it twice,** and WGSL refuses the redeclaration. Share one function node per module. TSL `.toVar(name)` names share the WGSL namespace with included code (a var named `giCoverage` clashed with the module's function). Source: T1.7.
- **Cube uploads with `flipY`:** three puts cube image i into layer `[0,1,3,2,4,5][i]`, flipped vertically (`WebGPUTextureUtils._flipMap`). Source: T1.7.
- **The Meshlet and Visibility-Buffer pipelines draw only FrontSide, opaque, channel-0 materials;** DoubleSide objects fall back to the standard path. Source: T1.7.
- **Standard materials scale indirect diffuse by (1 − F_ss − F_ms):** a white rough dielectric under a uniform environment reads exactly L. Compare against a plain diffuse host with Lambert. Source: T1.7.
- **pmndrs/sky has no `isSky` flag.** Its exposure is `sky.baker.sky.luminanceScale.value` and its sun `.sunDirection.value`; its `turbidity` is a Mie scale, not Preetham's. Source: T1.7.
- **rollup-plugin-dts ran out of memory** once ddgi-three imported three (it builds a new TS program per file reached through node_modules). Workspace `paths` in `packages/ddgi/tsconfig.build.json` fix it: 4.4 s. Source: T1.7.
- **Wrapping our textures:** `ExternalTexture` wraps a same-device `GPUTexture`. three binds it without allocating or destroying it.
  - Set `image = { width, height, depth }`, or size reads as 1×1.
  - Set `type` to match the sample type: the default is filterable float; use `FloatType` for rgba32float and `UnsignedIntType` for uint.
  - For array views, set `isArrayTexture = true`.
  - For 3D, set `is3DTexture = true` and sample it with `texture3D()`. Without the flag the shader fails with "expected texture_3d<f32>, got texture_2d<f32>". Source: T0.8, `WGSLNodeBuilder.js:1399, 2293-2296`.
  - On reallocation, create a new `ExternalTexture` and assign it to `node.value`; `needsUpdate` throws.
  - Don't touch `textureData.externalTexture`, which is the XR path.

  Source: `ExternalTexture.js`, `WebGPUTextureUtils.js:320-326, 486`, `WebGPUBindingUtils.js:332-383`.
- **External buffers:** there is no supported way to bind them. Fragment-readable GI data must be textures. Compute-only buffers are fine. Source: `WebGPUAttributeUtils.js:104-113, 446`.
- **Registering lights:** r187 replaced `addLight` with `Light.registerNode(NodeClass)`, and `LightsNode` creates one node per light. Copy the `LightProbeGrid` pattern: extend `Light`, register an `AnalyticLightNode` with `updateType = RENDER` and `renderGroup` uniforms. Source: `Light.js:18-22`, `LightsNode.js:233-288`, `LightProbeGrid.js:921`.
- **Light-probe priority slot:** `builder.context.lightProbeIrradiance` plus `lightProbePriority` gives one winner per fragment instead of adding up. `LightsNode.setupLights` adds the slot to `context.irradiance` once. It's undocumented, so pin three and test it. Source: `LightProbeGridNode.js:121-180`, `LightsNode.js:343-347`.
- **What `irradiance` means:** it is irradiance E; the Lambert 1/π is applied downstream. Return E, not radiance. Source: `PhysicalLightingModel.js:786-812`.
- **Toggling lights:** `LightsNode.customCacheKey` includes `light.id`, so adding or removing a light recompiles materials. Toggle with `intensity`. Source: `LightsNode.js:153`.
- **Drawing pipelines:** the Meshlet and Visibility-Buffer pipelines (r187 #35191) clone materials and overwrite `material.contextNode`. Don't inject GI through `contextNode`; a scene `Light` works in every pipeline. Source: `MeshletDrawingPipeline.js:114-117, 222-225`, `VisibilityBufferDrawingPipeline.js:692-712`.
- **Env diffuse counted twice:** `scene.environment` adds diffuse via `iblIrradiance`, separately from probe GI, so diffuse is double-counted. r187 has no switch. Open PR #35017 adds `envMapDiffuseEnabled`. `AmbientLight`, `HemisphereLight` and `LightProbe` also add straight to `irradiance`. Source: `PhysicalLightingModel.js:877-895`, `LightProbeNode.js:68`.
- **PMREM:** since r187 it is a `CubeRenderTarget` (RGBA16F, minimum size 256), sampled by roughness-to-mip. Env "irradiance" is the roughness-1 GGX lobe × π, not a cosine convolution, so expect a small mismatch with DDGI at fades. Source: PR #34585, `EnvironmentNode.js:101, 249-262`.
- **`LightProbeGrid`:** L2 SH, raster cubemap bake, no visibility, re-bakes overwrite values. Sponza (490 probes, 32² faces) bakes in 3.1 s on a Pascal GPU. PR #35174, which adds DDGI visibility, is a stalled draft: Sponza shading goes 3.08 → 4.00 ms (8 probes) or 3.38 ms (4 probes).
- **VXGI:** voxel cone tracing at 128³ by default, gathered as a full-screen pass that needs TRAA. Lights are dynamic; geometry needs full re-voxelization. Introduced in PR #34402 (r186).

- **`wgslFn` resources (T0.8):** a `wgslFn` accepts texture and sampler parameters.
  - Pass `texture()` or `texture3D()` nodes of `ExternalTexture`s.
  - A `sampler` parameter is fed from a filterable texture node and uses that texture's filter and wrap settings. Unfilterable (`rgba32float`) textures have no sampler; use `textureLoad`.
  - Source: `nodes/accessors/TextureNode.js:673-680`, `WGSLNodeFunction.js:58-67`.
- **Device features:** three reads `core-features-and-limits` from the device. When we create the device, request it (and all adapter features we need). Source: `WebGPUBackend.js:258`, T0.8.
- **Env diffuse workaround (r187):** `LightingModel.start` builds all lights, then calls `this.indirect`. Overriding `indirect` on our light node's lighting model lets us scale `iblIrradiance` by 1 − coverage.
  - Only Standard and Physical materials read `iblIrradiance`; Lambert and Phong don't.
  - It costs the multi-scatter and sheen env terms inside the volume. It's fixed at build time.
  - Source: `nodes/core/LightingModel.js:24`, `EnvironmentNode.js:109`, T0.8.
- **Light ordering** in `LightsNode` is by `.id`, which mixes node and object ids, so ties between equal-priority probe lights are unreliable. Give our light a distinct priority. Source: `LightsNode.js:36`.
- **Meshlet pipeline** draws `source.clone()` materials and skips cube render targets. So material-instance patches don't survive it; use the light node. Source: `examples/jsm/pipelines/MeshletDrawingPipeline.js:225`.

- **three's frame counter only advances inside `renderer.setAnimationLoop`.** With a bare `requestAnimationFrame` plus `render()`, frame-based nodes (passes, VXGI's cycle, TRAA jitter) stall or skip frames. Source: T0.5, `Animation.js:87`.
- **GPU timing in three:** `resolveTimestampsAsync` returns only the latest frame's passes. To get all of them, sum `backend.timestampQueryPool.<type>.timestamps`. Intel UHD sometimes returns an end before its start (one case read -3,546 s), so drop those. Source: T0.5.
- **LightProbeGrid:** objects drawn before the first bake read an empty atlas. Draw into a fresh RenderTarget after the first bake step. Source: T0.5.
- **VXGI** fails its depth copy on a renderer created with `antialias: true`; create it with `false`. Source: T0.5.
- **Background compiles:** pipelines compile in the background, and objects are skipped until theirs is ready. Warm up until the draw-call count stops growing before timing anything. Source: T0.5.
- **Release the device between pages:** destroy the device on `pagehide`. Otherwise the next page in the same browser waits on the old page's GPU memory; Bistro exterior then interior took 121 s, against 26 s once the device was destroyed. Source: T0.5.

## Babylon.js (9.33.2, checked 2026-11-04)

- **No device input:** `WebGPUEngine.initAsync` always calls `requestAdapter` and `requestDevice(deviceDescriptor)`. Pass our limits and features in `deviceDescriptor`, and hand `engine._device` to the core. `dispose()` calls `device.destroy()`, and after device loss Babylon requests a new device itself. Source: `webgpuEngine.pure.js:443, 481, 503-520, 3176`.
- **Wrapping our textures:**
  - `wrapWebGPUTexture` leaves the view null and sets no flags. Set `is2DArray`, `is3D` and `type` yourself (FLOAT for `rgba32float`, so the layout is unfilterable), then call `updateWrappedWebGPUTexture`.
  - Disposing the wrapper destroys our `GPUTexture` unless `_hardwareTexture.reset()` runs first.
  - Source: `webgpuEngine.pure.js:1981, 2014, 2999`, `webgpuHardwareTexture.js:88`.
- **Material plugins:**
  - Declarations go in `CUSTOM_FRAGMENT_DEFINITIONS`, the call in `CUSTOM_FRAGMENT_BEFORE_FINALCOLORCOMPOSITION`. Zero `finalIrradiance` to turn env diffuse off.
  - Samplers are found by a `Sampler` name suffix, and `<texture>Sampler` binds to that texture.
  - `vec4f` uniforms need `hardBindForSubMesh`.
  - Babylon is left-handed, and its env irradiance is E/π.
  - Source: `pbr.fragment.js:107, 621`, `webgpuShaderProcessorsWGSL.pure.js:35, 510`, `pbrBaseMaterial.pure.js:1561-1577`, `sphericalPolynomial.pure.js:187-199`.

## three-mesh-bvh (checked 2026-11-04)

- **Version and shape:** v0.9.18 (3 Nov 2026, MIT); default branch `master`. The WebGPU path is `three-mesh-bvh/webgpu` (`BVHComputeData`). It runs only inside three's node system (TSL `FunctionNode`s, three r185+), not on a bare `GPUDevice`, and its API is marked unstable. Source: `src/webgpu/BVHComputeData.js:83,144-149`.
- **Layout:** 64-byte nodes with both child boxes (Aila–Laine, PR #945); triangles as a u32 index buffer plus interleaved `vec4f` positions; 4 storage buffers; a 60-entry stack. Two levels: SAH clusters (at most 64 primitives) under a top level, with per-instance entries; `updateTransforms()` refits the top level only, and skinned meshes are baked on the CPU. Source: `packBVHBufferUtils.js:9`, `ClusteredBVH.js:138-209`.
- **Benchmark trap:** the per-mesh trees from `new MeshBVH(geometry)` default to the CENTER split, not SAH (`Constants.js:80-84`). Pass `{ strategy: SAH }` for a fair comparison.
- **Speed claims:** the "about 2×" upstream gains were *build* times (PRs #942/#943). The Aila–Laine layout gave about 1.12× on raycasts (PR #945, no hardware stated). No published rays/s figure exists. This corrects SPEC's "roughly doubled ray throughput" line.
- **three's `RayTracedRadiosity`** flattens static meshes into one world-space geometry and doesn't use the top level. It was merged to three master on 2 Nov 2026 (#35302), after r187.

## WebGPU

- **Compatibility mode:** devices may allow 0 fragment-stage storage buffers. Keep fragment-side data in textures.
- **Readback:** texture-to-buffer row pitch must be padded to 256 bytes. Source: the HomeFig DDGI branch.
- **Scissor:** it doesn't limit a WebGPU clear. Source: the HomeFig DDGI branch.
- **Hidden tabs** do no GPU work, so benchmarks must run in a visible or headless-active page. Source: the HomeFig branch, and this project's audit (the built-in browser pane throttles when hidden).

## Tooling and environment

- **Choosing the GPU in headless Chrome:**
  - Chrome 156 gives NVIDIA by default, whatever `powerPreference` says.
  - `--use-webgpu-power-preference=force-low-power` makes `requestAdapter()` return null.
  - The Intel GPU is reached with `--use-adapter-luid=<high>,<low>`. LUIDs change every boot; the runner reads them with `Get-PnpDeviceProperty` on `{60B193CB-5276-4D0F-96FC-F173ABAD3EC6} 2`.
  - `--disable-gpu` makes `requestAdapter()` return null.
  - Intel reports architecture `gen-12lp`.
  - Source: T0.2, `tools/gpu/chrome.ts`.
- **WGSL errors arrive late:** shader-module errors surface only after the queue drains. `runGpuTest` waits about 50 ms after the drain; anything later can be missed. Source: T0.2.
- **Node runs `.ts` directly** (Node 24 type stripping), but rejects enums, namespaces and constructor parameter properties. The root tsconfig has `allowImportingTsExtensions` for `.ts` import paths. Source: T0.2.
- **GPU timestamps on this laptop:** both GPUs support `timestamp-query` in Chrome 156, and headless Chrome returned unrounded values. Timing costs less than CPU-side frame timing can resolve (`performance.now()` steps 0.1 ms; the NVIDIA per-round scatter is about ±0.5 ms). Source: T0.3.
- **Device page:** `tools/gpu/pages/device/`. `?label=`, `?rounds=`, `?frames=` and `?scale=` change the run. `pnpm device:serve` keeps its certificate in `node_modules/.cache/ddgi-lan` and regenerates it when the LAN addresses change. Source: T0.3.
- **One GPU benchmark at a time.** Before any timing run, Cycles render or native harness run, take the shared lock with `node tools/reference/bench-lock.ts run --what "<label>" -- <command>`, or `acquire --pid <pid>` and `release --pid <pid>`.
  - `status` shows who holds it.
  - The lock folder is still `$TEMP/ddgi-gpu-bench.lock`, but it now holds an `owner.json`, and a lock is taken over only when its owner process is dead.
  - Don't use a bare `mkdir` any more. Correctness runs (smoke tests, pixel readbacks) don't need the lock. Never time native and browser GPU work at the same time. Source: main session, M0.
- **WebGPU needs a secure context.** `http://<LAN IP>` pages get no `navigator.gpu`, so pages for the Mac and iOS devices must be served over HTTPS, and a self-signed certificate is fine. `localhost` counts as secure. Source: the WebGPU spec (secure-context only).
- **GPUs on this laptop:** Playwright or headless Chrome with `powerPreference: 'high-performance'` gets the RTX 3060 Laptop. The Claude desktop browser pane got the Intel iGPU. Source: the audit, 2026-10-30.
- **Worktrees read slow:** benchmarks run from a git worktree read about 3× slower in absolute GPU time. Compare A/B only within one environment. Source: the `@pmndrs/upscaler` CLAUDE.md.
- **`node_modules` symlinks:** a symlinked `node_modules` in a worktree isn't matched by `node_modules/` in `.gitignore`. Add a slash-less `node_modules` to `.git/info/exclude`. Source: the `@pmndrs/upscaler` CLAUDE.md.
- **Workspace shape:** the umbrella `@pmndrs/ddgi` lives in `packages/ddgi` (unlike the upscaler, whose repo root is the umbrella). `ddgi-core` and `ddgi-three` are private, export their `src/*.ts` directly, and are bundled into `packages/ddgi/dist/` at build time. The root `package.json` is a private shell: `lint`, `typecheck` and `test` run over everything, and `pnpm build` filters to `@pmndrs/ddgi`. Source: T0.1.
- **Bash tool and heredocs:** a long chain of `cat <<'EOF'` file writes in one Bash call failed to parse and created nothing. Use the Write tool for new files.
- **Empty modules in the build:** vite and rollup warn "Generated empty chunks" while the packages only contain `export {}`. That is expected until real code lands. `passWithNoTests` in `vitest.config.ts` keeps `pnpm test` green with no tests; remove it once there are tests.
- **pnpm banner:** pnpm 10.23.0 prints an "update available" box (pnpm 12). `packageManager` stays `pnpm@10.23.0`, the version installed here.
- **Native harness device pick:** upstream skipped any GPU whose name lacks "RTX 3060". The fork (`m0-harness`) picks the GPU by ini `app.gpu` or the env var `RTXGI_GPU`, and otherwise takes the first GPU with ray tracing. Source: T0.6.
- **The native harness skips a converged volume.** Once probe variability drops below 0.4 after 16 samples, trace and blend stop running. Time it with the fork option `probeVariability.skipConverged=0`, and don't use `native.csv` rows from commit 9bd7f3c. Source: T0.6.
- **Native ray counts:**
  - Classification leaves 30–70% of probes active. Inactive probes trace only the 32 fixed rays and aren't blended (`ProbeTraceRGS.hlsl:69`, `ProbeBlendingCS.hlsl:397`).
  - Active probes trace all rays, but only rays 32 and up are shaded and blended. Each shaded hit also casts one shadow ray per light, which isn't counted.
  - So compare on rays actually traced, not probes × rays. Sponza traces 2.01 M rays in 9.39 ms, about 214 M rays/s with hardware RT on the RTX 3060 under Vulkan.
  - Source: T0.6 and its review.
- **Native Vulkan mesh limit:** upstream's `ByteAddrBuffer[]` array holds 2048 entries, which caps a scene at about 1,022 meshes; Bistro lost the GPU. The fork sizes the array from the scene (1df1f97). Source: T0.6.
- **The native frame rate is CPU-bound.** Untimed main-thread work makes it 50–87 ms per frame on Sponza, against about 20 ms of GPU work, so compare GPU numbers only. Source: T0.6.
- **Native timings drift 15-30% between runs of the same config** on the RTX 3060 laptop, in every pass including lighting, so it's machine state, not code. Take 3+ runs per scene before comparing. Source: T0.6.
- **Native multi-part scenes** are merged into one glTF by `tools/native/merge-gltf.mjs`, which references the parts' own `.bin` and image files (nothing is copied). `native.csv` Sponza rows without a `scene_parts` row are base-only. Source: T1.9 prep.
- **The native harness's default probe grid isn't ours:** it places `ceil(size / spacing)` cell centres. `bench.ps1 -GridRule ddgi` gives matched counts and positions: two-room 1,071 probes against the default 918, Sponza 19,425 against 17,760. Native's own-cell classification switches off 40% of Cornell's probes and 59% of Sponza's. Source: T1.9 prep.
- **Native harness keys:** F2 dumps DDGI textures, F4 runs a 1024-frame benchmark to CSV, J toggles the performance overlay, R reloads shaders and config, M switches between DDGI and the path tracer.
- **Installed tools:** RenderDoc 1.47.0, PIX 2603.25, Blender, NVIDIA FLIP (`python -m flip_evaluator`), Vulkan SDK 1.3.283, Nsight Systems 2022.4, Nsight Compute 2022.3. Nsight Graphics needs a manual install.

## Test assets

- **Sponza license:** the Khronos glTF Sample Assets Sponza now points at the CRYENGINE marketplace license, which allows "Games in object code form only" (§2.1.4). Crytek's 2010 note says it was "donated to the public for radiosity". These conflict, so it's on hold for Dennis. Source: `Models/Sponza/README.md` and `LICENSE.md` in glTF-Sample-Assets, checked 2026-11-04.
- **Intel Sponza:** `https://cdrdv2.intel.com/v1/dl/getContent/830833` redirects to a signed CloudFront URL for `main_sponza.zip` (3,987,608,266 bytes), with no login.
  - Add-ons have their own IDs: curtains 726650, ivy 726656, trees 726662, candles 726676.
  - Git-bash's `curl` fails TLS to cdrdv2 ("self signed certificate in certificate chain"). Windows `curl.exe` works. Node `fetch` worked during T0.4 but has been refused since 2026-11-05 (see below).
  - The www.intel.com samples page doesn't load from here; the Wayback copy from 2025-12-06 has the links.
  - Source: T0.4 scout, 2026-11-04.
- **Intel Sponza add-ons** (T1.10):
  - Packages: 726650 `pkg_a_curtains.zip` (786,898,766 B, 2.0 M triangles), 726656 `pkg_b_ivy1.zip` (1,383,921,436 B, 5.1 M), 726662 `pkg_c_trees.zip` (951,147,802 B, 2.0 M), 726676 `pkg_d_10k_candles.zip` (3,190,731,713 B, 19.5 M). The base Sponza has 3.7 M triangles.
  - Textures are 4096²; the base package's script reduces them to 1k. The base's textures folder already holds byte-identical `curtain_fabric_*` and `metal_door_01_*` files.
  - The trees' cypress material is BLEND; the ivy's materials are declared OPAQUE.
  - Candles: one 19.5 M-triangle mesh with no textures, plus 10,000 point lights.
- **`cdrdv2.intel.com` rejects Node `fetch`** with HTTP 200 and a 247-byte "Request Rejected" page (2026-11-04), for the base and add-on URLs alike. Windows `curl.exe` gets the zip (302 to a signed CloudFront URL, no login). The fetch tool's size check refuses the page. Source: T1.10.
- **McGuire archive:** Breakfast Room and Living Room are CC BY 3.0 (each folder's `license.txt`). Conference Room says only "Credit required", with no license, so it's excluded.
- **Bistro (NVIDIA ORCA):** CC BY 4.0. One 894 MB zip holds both exterior and interior. The download URL redirects (302) to a short-lived token URL, and `curl -L` follows it with no login.
  - Exterior: 2.83 M triangles, 1,296 meshes. Interior: 1.04 M triangles, 1,186 meshes.
  - The exterior is more than the 128 MB default storage binding can hold at 48 bytes per triangle.
- **Blender 4.5 conversion quirks** (Blender 4.5.0 LTS at `C:\Program Files\Blender Foundation\Blender 4.5\blender.exe`; 4.2, 5.0 and 5.1 are also installed):
  - The FBX importer has no `apply_unit_scale`; Bistro's FBX is already in metres.
  - Bistro normal maps are DirectX-style, so `fbx-to-glb.py` flips green.
  - Bistro's opacity lives in a second copy of each base-color DDS.
  - The FBX importer doesn't wire emissive textures, so Bistro's lamps won't glow (M4).
  - The glTF exporter takes alpha mode from the node setup (Greater-Than 0.5 means MASK), not from `blend_method`.
  - The OBJ importer sets Transmission 1.0 on materials with `illum 4` plus `Tf`, so `obj-to-glb.py` clears it.

- **Blender add-ons hang headless runs.** User add-ons on this machine (Scatter5, SimpleBake, Substance3D) stop `blender --background` from exiting, sometimes at startup. Always pass `--factory-startup`. Source: T0.4, fetch #3/#4.
- **Blender's glTF export is not bit-reproducible.** Two runs of the same conversion differ in a few bytes of vertex data, while the JSON chunk and buffer lengths stay identical. `sources.json` marks those outputs `"reproducible": false` and checks `jsonSha256` plus `binBytes` instead. Source: T0.4.
- **Blender 4.5.0's glTF export varies only in TANGENT data,** rounded at the 4th decimal. Index buffers, images, JSON and padding are identical between runs. So `tools/assets/glb-views/<id>.json` pins every other bufferView. Re-record them with `node tools/assets/fetch.ts --record-glb-views <id>`. Source: T0.4 follow-up.
- **Blender 4.5's OBJ importer can make everything metal.** It sets Metallic to the average of `Ka` when `illum` turns reflection on (as `illum 4` does), Specular IOR Level to the `Ks` average, and IOR to `Ni`. The Breakfast Room has `Ka 1 1 1`, so it came out all metal. `obj-to-glb.py` resets materials to a plain dielectric, and the `"dielectric": true` output check keeps it that way. Source: T0.4 follow-up, Blender `obj_import_mtl.cc`.
- **Convert scripts must never write their inputs.** Inputs are hard links into `assets/.cache`, so writing one in place would corrupt the cache. The fetch tool checks that inputs are unchanged after a conversion.
- **GLTFLoader turns shadows off.** Loaded meshes neither cast nor receive shadows until you set `castShadow` and `receiveShadow`. Source: T0.4 check page.

## Ground-truth renderers (T0.7)

- **Cycles units** (`tools/reference/cycles/units.py`):
  - Sun strength = lux.
  - Point power P = 4πI watts, from candela.
  - Emission strength = radiance.
  - A spot keeps the point's I on its axis.
- **Cycles axes:** three (x,y,z) maps to Blender (x,−z,y). three's Euler XYZ order is the reverse of Blender's.
- **Cycles has no light range window.** three r187 point lights fade to 0 at `distance`, so three is about 12% dimmer than Cycles at 7 m on two-room.
- **Blender defaults that bias references:** indirect clamp 10, light threshold 1%. Turn both off.
- **Always run Blender with `--factory-startup`.** A user add-on on this machine makes Blender hang at exit.
- **three-gpu-pathtracer (WebGPU):**
  - `maxBounces` counts the camera hit as 1.
  - Emissive meshes are found only by scattered rays, so Cornell's ceiling lamp is noisy. Point lights are sampled explicitly.
  - Its default `clampIndirect` is 10; set it to 0.
  - The accumulation texture is rgba32float, readable from `backend.get(target).texture`.
- **FLIP:** `python -m flip_evaluator.flip_python_api`, which needs no numpy.
  - Run it with `python -E`, not `-I`, because `-I` hides the user site-packages where it is installed.
  - FLIP numbers are comparable only at the same downsample.
- **Comparing noisy renders:** average over 8×8 blocks before judging. Two Cycles seeds differ by 0.8% per pixel but by 0.1% per block.
- **Blender's glTF importer links Metallic and Roughness to the metallicRoughness texture.** A linked socket ignores its value, so unlink it before setting it. Principled's Diffuse Roughness (4.3+) above 0 means Oren-Nayar. Source: T0.7 follow-up.
- **The references are pure Lambert; three's standard material isn't.** `scenes/src/three-scene.ts` uses `MeshStandardMaterial` (roughness 1, metalness 0, F0 0.04). That adds a rough specular lobe the Cycles and path-tracer references lack, worth a few percent of the full image. When comparing the viewer against a reference, render with `specularIntensity = 0`, or measure the gap first, so the GI error isn't mixed with material error. Source: T0.7 Opus review.
- **`@pmndrs/denoiser` is not on npm.** Build steps are in `tools/reference/README.md`.

## Sky (T1.1)

- **Radiance `.hdr` readers disagree:** three's HDRLoader reads the mantissa as m/255, stb_image as m/256, Blender as (m+0.5)/256, up to 0.4% apart. Use half-float EXR for shared HDR data. Source: `HDRLoader.js:358`.
- **three's PMREM against a cosine integral:** under the open-ground sky, PMREM environment diffuse on flat ground reads R/G/B −0.04/−0.61/−0.97% (diffuse-only material).
- **Blender's equirect matches three's `equirectUV`** after the (x, −z, y) axis mapping, with no rotation. `--calibrate` checks it to 0.00%.
- **three's EXRLoader** stores rows bottom-up with `flipY` false, so `equirectUV` reads it the right way up. Source: `EXRLoader.js:2469`.
- **Native harness sky:** one constant `skyRadiance` in ProbeTraceRGS, PathTraceRGS and GBufferRGS. There's no texture path, and nothing tells rays above the horizon from rays below.

- **L2 SH misses ±0.5% sky irradiance under our Preetham skies** (up to 0.9%); adding band 4 brings it under 0.35%. Band 3's cosine weight is 0. Source: T1.4.
- **three r187 WebGPU equirect textures put the nadir at memory row 0** (`isFlipY` is false in WGSL), and environment rotation samples at Rᵀd. three samples a non-render-target `CubeTexture` with x negated, so its cube maps come out mirrored against a plain convention. Source: T1.4 and its review.
- **Shadow-ray offsets:** Wächter–Binder's per-component ulp offset tilts along the surface when the coordinates differ in size, and leaked 8 thin-box corner hits. T1.4 pushes along the normal, scaled by the largest coordinate, and nudges the point into its triangle. Before the nudge, 25 of 1,200 barycentric corner hits landed just outside the neighbouring face. The offset grows with distance from the origin, so far-off thin walls can still leak (comment at `platform/shade/wgsl.ts`). Source: T1.4.
- **Nearest-sampled (rgba32float) rotated skies shift by up to half a texel.** Source: T1.4.

## Platform scene copy (T1.2)

- **Scene copy memory:** octahedral normals (4 bytes) and 16-bit indices for meshes with at most 65,536 vertices. Intel Sponza's base package (2.05 M vertices, 3.75 M triangles) takes 87.7 MB. `sponza` with the curtains (5.74 M triangles) takes 130.1 MB, 97% of the 128 MiB default storage binding, and `sponza-full` 348.5 MB, so large scenes need raised limits (`node tools/bench/scene-copy-memory.ts`). Bistro exterior takes 111.4 MB, which fits the 128 MB default binding only because of the octahedral normals. uv1 costs 16 MB on Sponza, so adapters should pass `uvs1` only when a material samples channel 1.
- **Scenes that need raised limits** (default `maxStorageBufferBindingSize` 128 MiB = 134.2 MB). The tracer's triangle records take 48 bytes per triangle; the scene copy figures come from `node tools/bench/scene-copy-memory.ts`:
  - `sponza` (base + curtains, 5.74 M triangles): scene copy 130.1 MB (fits, just); tracer records about 276 MB (needs raised limits).
  - Sponza base only (3.75 M): scene copy 87.7 MB; tracer records 179.9 MB (needs raised limits).
  - `bistro-exterior` (2.83 M): scene copy 111.4 MB; tracer records about 136 MB (needs raised limits).
  - `sponza-full` (12.8 M): scene copy 348.5 MB; tracer records about 616 MB (needs raised limits).
  - `bistro-interior` (1.04 M) and every procedural scene fit the defaults.
  - Both laptop GPUs allow about 2 GiB (DECISIONS 2026-11-05).
- **`Omit<Union, K>` keeps only the keys shared by every member,** so `Partial<Omit<LightDescriptor,'type'>>` couldn't name `position` or `range`. Fixed with `LightChanges`.
- **Bind-group layouts in budget tests must be explicit.** An `'auto'` layout drops bindings the entry point doesn't use, so it can hide an over-budget kernel.
- **WebGPU ordering:** a `queue.writeBuffer` made during `encode` lands before the host's submit of that encoder, and copies recorded in the encoder run after it. So a growth copy must never cover a range written in the same frame.

## Tracing (measured)

- **Any-hit should visit the farther child first.** A shadow ray starts on a surface, so the nearer child is mostly that surface, and the blockers lie ahead. On the RTX 3060 that was 1.2–2.8× faster than nearer-first; closest-hit is unaffected. Source: T1.3.
- **Leaf size is not why three-mesh-bvh won at any-hit.** It is faster at leaf 4 than at its default 10. Source: T1.3.
- **Reading triangle corners through the scene copy's index buffer costs 10–30% of trace speed,** so self-contained 48-byte records stay. Source: T1.3.
- **Both laptop GPUs allow storage bindings of about 2 GiB** in Chrome 156 (`maxStorageBufferBindingSize` 2,147,483,644; `maxBufferSize` 2,147,483,648). Source: T1.3.
- **tinybvh builds over NaN positions without complaint,** so `buildMeshBvhWithTinyBvh` refuses non-finite positions. Source: T1.3.
- **`active` is a reserved word in WGSL,** and using it as a name fails shader creation. `refit-cpu.test.ts` scans the refit shader for such names. Source: B1a.
- **One adapter makes one device.** A page that needs a second device (for example one without subgroups) must request a new adapter. Source: B1a.
- **Subgroup sizes on this laptop:** RTX 3060 32 (min 32, max 32); Intel UHD 8 to 16. Source: B1a.
- **The GPU top-level refit** takes 0.084 ms for 10k moving instances on the RTX 3060 and 0.44 ms on the Intel UHD. Box reads through atomics never needed their bounded re-wait to give up on either GPU. Source: B1a.
- **An instance's world box** is the box around the mesh box's eight transformed corners, which is looser than the box around the transformed vertices when the mesh is rotated. Tests must compare against the corners. Source: B1a.
- **Morton codes need cubic cells, not one stretched to 1024 cells per axis.** Stretching each axis (HIPRT's plain `computeMortonCode`) made LBVH trees of flat or long meshes cost 2.4-2.6 times tinybvh's SAH (a flat grid, a slab, a rod); cubic cells over the mesh's longest side gave 1.02-1.22, and cubes are unchanged. Real meshes are often walls, floors and curtains. Source: B1, CPU twin (`lbvh-cpu.ts`).
- **Building many meshes one by one on the GPU is dispatch-bound.** A per-mesh LBVH (about 40 dispatches each) took 97 ms of GPU time for Sponza's 132 meshes, while its largest mesh alone took 2.3 ms. One batch for all meshes (mesh and tile tables, a segmented sort) took 32-42 ms. Source: B1, `bvh-lbvh-profile`.
- **An atomic load that every thread does on the same word serialises.** On D3D12 an atomic load is a read-modify-write: six of them per thread on one mesh's bounds cost 23 ms in the Morton kernel for 5.7M triangles. Read such words through a plain read-only binding of the same buffer in a later dispatch. Source: B1.
- **The LBVH climb is not bound by how many words it hands over.** Halving them (leaves aren't handed over) left it at about 13 ms on Sponza, and a two-round climb (first round up to 64-triangle nodes) only took it to 11.7 ms. It is still the build's slowest step. Source: B1.
- **`target` is also a reserved word in WGSL, and a function can't share a name with a module-scope variable.** `wgsl-names.ts` checks every B1 shader for both. Source: B1.
- **LBVH trees trace slower than tinybvh's even at a similar SAH cost.** Sponza (5.7M triangles): SAH 1.22x tinybvh's, closest-hit about 0.8x and any-hit about 0.8x of tinybvh's Mrays/s on the RTX 3060; see `trace.csv` for the recorded medians. Source: B1.
- **Use a watertight triangle test.** Möller–Trumbore let 3–12% of rays aimed at shared edges slip between triangles (RTX 3060 11.6%, Intel UHD 3.0%, 2M rays each), and leaked into sealed thin boxes. The watertight test (Woop et al. 2013, with stored corners) let none of 4M through. Source: `tools/gpu/pages/trace-oracle`, T0.10.
- **Don't index a vector with a per-ray value in WGSL.** `a[kx]` with a runtime `kx` cost about 40% on the RTX 3060 (Sponza 61 → 101 Mrays/s once removed). Build per-ray row vectors with `select()` once per ray instead. Source: T0.10 part C.
- **Read BVH nodes and triangles as `array<vec4f>`:** one 16-byte load each gave +3% on the RTX 3060 and +23% on Intel UHD. Source: T0.10 part C.
- **Traversal stack depth:** Sponza needs 35 entries and Bistro exterior 44. A 32-entry stack was no faster than 64, so 64 stays. Source: T0.10 part C.
- **tinybvh in WASM:** its fast builder needs `__AVX__`. Emscripten emulates 256-bit AVX on WASM SIMD from 3.1.68 on; with `-msimd128 -mavx`, `Build` runs the SIMD Bonsai builder (Ganestam et al. 2015): Sponza in about 1.5 s in Node, with the lowest tree cost of tinybvh's builders. With 3.1.57 only the scalar builder compiled (4.9 s). pthreads would need cross-origin isolation (COOP/COEP headers), which a library can't require of its hosts. Rebuild with `pnpm bvh:wasm` (pinned emsdk in `tools/emsdk/`). Source: T0.10 part C, B0.
- **V8 turns scalar WASM float selects into branches.** Box loops written as `a < b ? a : b` cost 2.5× on Sponza. Use `f32x4.lt`/`gt` with `v128.bitselect`. `f32x4.min` and `fminf` treat −0, +0 and NaN differently from the JS conversion, so they'd break byte identity. Source: B0b.
- **Sponza under tinybvh:** 115 meshes of at most 85k triangles; 1.91 M nodes, about 2 triangles per leaf; WASM memory stays around 67 MB. tinybvh's own Build (about 1.45 s) is now most of the build time. tinybvh builds the same tree twice from the same mesh (tested). Source: B0b.
- **Every browser with WebGPU also has WASM SIMD** (Chrome 91+, Firefox 89+, Safari 16.4+), so no scalar WASM build is needed. Source: B0.
- **JS memory speed on this laptop:** just reading 300 MB takes about 110 ms, so a JS pass over a Sponza-sized tree costs 100-300 ms whatever the code does. Move such passes into WASM or onto the GPU. Source: B0.
- **Check the RTX's clock before timing:** `nvidia-smi --query-gpu=clocks.sm,clocks.max.sm --format=csv` under load; the RTX 3060 Laptop's maximum is 2100 MHz. Source: T1.6.
- **The RTX 3060 Laptop doesn't reach 2100 MHz under sustained load:** after the unlock it held about 1610–1660 MHz, held back by the power cap (about 72 W) and thermal slowdown (87 °C). Dennis's own apps keep it 16–28% busy when "idle". Expect run-to-run jumps; take 3+ runs, and repeat a set that jumps. Source: RTX re-time, 2026-11-05.
- **The RTX 3060's clock was locked at 1200 MHz on 2026-11-05.** Dennis had set `nvidia-smi -lgc 1200,1200` during an earlier benchmark; nothing in the repo did it. He reset it with `nvidia-smi -rgc`.
  - **Affected for certain:** T1.6's GI-sampling rows in `shading.csv` (0.62 / 1.07 ms), which say `note/sm_clock_locked_mhz,1200`.
  - **Possibly affected** (the lock's start time isn't known): every RTX GPU timing recorded on 2026-11-05: T1.3's trace rows (d6ee376), B1a's refit rows in `bvh.csv`, T1.5a/T1.5b's per-probe DDGI timings, and the native matched-count runs (954d393).
  - **Not affected:** Intel UHD timings, CPU timings (adapter sync, builds), and every correctness result.
  - Re-timed after the reset (PROGRESS, "RTX re-time"): shadow rays were 16–19% faster, DDGI per probe 17–37% cheaper, native blend 18% faster. Closest-hit and native totals barely moved.
- **Python on Windows writes CRLF in text mode;** write files with `newline='\n'` or in binary mode. Source: T1.6.
- **Check the GPU is idle before timing.** A browser tab left open on the viewer used about 20% of the RTX 3060 and cost every tracer about 35%. Look at Task Manager's GPU column, or `nvidia-smi`, first. Source: B0.
- **Repeat runs:** `repeat.ts` writes `run<k>/` rows plus `median/` rows to `trace.csv`, and the commit column checks only the measured paths for `-dirty`. Source: T0.10 part C.
- **WGSL `bool` can't live in a buffer**, so hit records store front-face as `u32`. Source: T0.10.
- **Mirrored instances** flip winding in world space, so front-face is `(det > 0) != mirrored`. Source: T0.10.
- **Long page work:** the test runner gives up on a page that's busy for 30 s inside one CDP evaluate, so long CPU work on a page must yield. Source: T0.10.

## DDGI (measured lessons)

- **Half-float atlases with hysteresis 0.97 need stochastic rounding:** round-to-nearest stalls the furnace 4.6% low. Source: T1.5a.
- **DDGI reads high where irradiance changes fast near surfaces:** Cornell's true irradiance rises 12–21% over the first 0.5 m off the floor, and probes buried in slabs make it worse. Source: T1.5a's path-traced irradiance check.
- **One kernel that traces, shades and blends is register-bound:** a split trace-and-shade kernel plus a blend kernel ran 2× faster on the RTX 3060 and 1.5× on Intel. Source: T1.5a.
- **Both laptop GPUs offer `texture-formats-tier1`** in Chrome 156, so DDGI's distance atlas is `rg16float`. Source: T1.5a.
- **Decoded ray directions in workgroup memory** (12 KB of 16 KB) made the update 21% faster than recomputing them per texel. Source: T1.5a.
- **A big-change test on a probe's own noisy estimate biases the field:** per texel +8%, on the probe's mean +22% on Cornell. Deciding from the previous update reads 1.8 points low instead. Source: T1.5b.
- **32 fixed rays miss surface slivers near the corners of the reach box;** at a reach of 1.5 spacings no needed probe switched off on Cornell or the cabinet row. Source: T1.5b.
- **Coincident faces of touching solids defeat ray-based inside tests:** probes can sit a few mm inside a board and stay active. Source: T1.5b.
- **After the trace/blend split, the Intel UHD spends about half its DDGI update time in the blend kernel.** Source: T1.5b.
- **DDGI test runs are deterministic:** rays are seeded by the frame number, so repeated leak runs give identical numbers. Spread has to come from varying the seed, not from re-running. Source: T1.5b.
- **The Intel UHD hangs when ddgi-distance-format's second device starts right after ddgi-classification** (6 of 6), with the current and the previous kernels. Not understood; the page runs its fallback device first to avoid it. Steps are in PROGRESS (T1.5b). Source: T1.5b.
- **Cost of DDGI sampling** (RTX 3060 at a locked 1200 MHz, 1440p, every pixel a surface): distance fetches about 0.31 ms, irradiance fetches 0.21, probe-data loads 0.06, arithmetic 0.35. Hoisting per-sample constants plus `inverseSqrt` saved 16%. A branch-free loop was slower; a loop the compiler can't unroll was 9% slower. Source: T1.6.
- **Overdraw doubles GI cost in a forward pass:** 2.49 ms without a depth prepass against 1.29 ms with one. Shade GI once per pixel. Source: T1.6.
- **Sponza's glTF:** 25 of 28 materials have `metallicFactor` 1 plus a metallicRoughness texture. Folding the factor alone into GI albedo makes them bounce nothing. Source: T1.8.
- **DDGI timings arrive a few frames late:** right after a warm-up of 4,096-probe steps, `stats.passes` still describes warm-up frames; check `pass.frame`. Source: T1.8.
- **DDGI cost per probe depends on batch size and scene:** Sponza on the RTX, about 0.011 ms per probe in 4,096-probe steps, about 0.03–0.045 ms per probe at a 2 ms budget (tens of probes); Cornell about 0.002 ms. Source: T1.8.
- **Shadow passes use `scene.overrideMaterial`:** a material with `allowOverride = false` is drawn as itself there, so give it `castShadow = false` or `depthWrite = false`. Source: three r187 `Renderer.js:3928`, T1.8.
- **GPU timestamps include other programs' GPU work:** on this laptop something hits every 50 ms, +2.5 ms on the RTX and +1.3 ms on the Intel (Chrome's GPU process 17–27% of the RTX, the compositor 7–12%). Check with `Get-Counter '\GPU Engine(*engtype_3D)\Utilization Percentage'`; the RTX's LUID changes every boot. The longer our pass, the more frames it hits: 1% at 0.2 ms, 11% at 0.3 ms, 34% at 1.3 ms. Source: T1.9a.
- **At 60 Hz with a light load the RTX clocks down,** so DDGI passes take 1.8–2.1× as long as back-to-back runs (Cornell, 512 probes: 1.29 against 0.70 ms). Time at the frame rate you care about. Source: T1.9a.
- **DDGI cost against batch size** (RTX, GPU busy): Cornell 0.15 ms + 1.4 µs per probe; two-room 0.36 + 2.0; Bistro exterior 1.57 + 4.7 (up to 512). Sponza is flat at 1.3–1.6 ms from 32 to 128 probes, then about 4.2 µs per probe up to 512; 4,096-probe batches cost 8.4 µs per probe. Source: T1.9a, `timing.csv` `batch_cost/`.
- **Sponza's `ddgi/trace` at 2,048 probes:** shadow rays 53%, closest hits about 41% (48 Mrays/s against 105 in the trace bench), bounce read 14% (the parts overlap). Shadow rays run at no more than 1/7 of the bench's any-hit rate. Reading the bounce before the shadow ray saved 4–30%. Source: T1.9a.
- **Cold compile on the RTX:** `ddgi/trace` 1.4–1.9 s, sky SH projection 0.9–1.1 s, `ddgi/blend` 0.5 s, each `encodeBatch` kernel 0.3 s. A second platform on the same device compiles in about 10 ms. Source: T1.9a.
- **Cycles' Diffuse Direct includes the sky and emitters seen straight from a surface,** and DDGI's field includes them too. With a sky, compare DDGI against indirect + direct − a lights-only direct render (DECISIONS). Source: T1.9.
- **Sponza untextured in Cycles:** 8,192 samples on OptiX at 640×360 take about 5 min for both renders, with 8×8-block noise 0.5%. Blender's glTF importer makes 49 materials from Sponza's 32 (`.001` variants). Source: T1.9.
- **Leak fixes by measured effect:** double-sided distance (84×), then probe density (22×). Probe texture resolution made no difference. Reducing bias made leaks *worse*, by up to 34×. Source: the HomeFig DDGI branch.
- **View bias:** 6 → 0 cut image error against a path tracer from 18.2% to 12.4% RMS on two-rooms. Bias pulled into furniture caused black patches. Source: the audit, the HomeFig branch.
- **Recursion cap:** NVIDIA's native recursion albedo cap (0.9 vs 1.0) changes indirect light by about 23%. Match it when comparing against native. Source: the audit.
- **Reference cameras:** build rays from an explicit camera basis, not inverse-MVP, which loses precision at near 0.01 / far 2000. Source: the audit.
- **Light groups:** while a group updates, its bounce reads only its own layer. The sky is a fixed layer. Bake groups in real colors. Source: the HomeFig branch.
- **Per-pixel BVH visibility:** about 12 ms per frame at 720p on a 90-triangle scene. Don't put it in the default shading path. Source: the audit.
