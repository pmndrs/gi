# Notes: facts worth not relearning

One bullet per fact, each with its source. Add a bullet when something cost you time. Fix or delete bullets that turn out wrong. Group by topic.

## three.js (r187)

- **Shared device:** `new WebGPURenderer({ device })` is supported. When three creates the device itself, it uses a compatibility adapter with default limits: a 128 MB storage-buffer binding and 8 storage buffers per stage. Source: `WebGPUBackend.js:72, 213-254`.
- **Device errors:** three assigns `device.onuncapturederror`. Use `addEventListener('uncapturederror')` and never assign it. Source: `WebGPUBackend.js:281`.
- **Wrapping our textures:** `ExternalTexture` wraps a same-device `GPUTexture`. three binds it without allocating or destroying it.
  - Set `image = { width, height, depth }`, or size reads as 1×1.
  - Set `type` to match the sample type: the default is filterable float; use `FloatType` for rgba32float and `UnsignedIntType` for uint.
  - For array views, set `isArrayTexture = true`, or use `texture3D()` for 3D.
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

- **GPUs on this laptop:** Playwright or headless Chrome with `powerPreference: 'high-performance'` gets the RTX 3060 Laptop. The Claude desktop browser pane got the Intel iGPU. Source: the audit, 2026-10-30.
- **Worktrees read slow:** benchmarks run from a git worktree read about 3× slower in absolute GPU time. Compare A/B only within one environment. Source: the `@pmndrs/upscaler` CLAUDE.md.
- **`node_modules` symlinks:** a symlinked `node_modules` in a worktree isn't matched by `node_modules/` in `.gitignore`. Add a slash-less `node_modules` to `.git/info/exclude`. Source: the `@pmndrs/upscaler` CLAUDE.md.
- **Workspace shape:** the umbrella `@pmndrs/ddgi` lives in `packages/ddgi` (unlike the upscaler, whose repo root is the umbrella). `ddgi-core` and `ddgi-three` are private, export their `src/*.ts` directly, and are bundled into `packages/ddgi/dist/` at build time. The root `package.json` is a private shell: `lint`, `typecheck` and `test` run over everything, and `pnpm build` filters to `@pmndrs/ddgi`. Source: T0.1.
- **Bash tool and heredocs:** a long chain of `cat <<'EOF'` file writes in one Bash call failed to parse and created nothing. Use the Write tool for new files.
- **Empty modules in the build:** vite and rollup warn "Generated empty chunks" while the packages only contain `export {}`. That is expected until real code lands. `passWithNoTests` in `vitest.config.ts` keeps `pnpm test` green with no tests; remove it once there are tests.
- **pnpm banner:** pnpm 10.23.0 prints an "update available" box (pnpm 12). `packageManager` stays `pnpm@10.23.0`, the version installed here.
- **Native harness device pick:** the fork's `Vulkan.cpp` skips any GPU whose name lacks "RTX 3060". This must be made configurable (T0.6). Source: the audit.
- **Native harness keys:** F2 dumps DDGI textures, F4 runs a 1024-frame benchmark to CSV, J toggles the performance overlay, R reloads shaders and config, M switches between DDGI and the path tracer.
- **Installed tools:** RenderDoc 1.47.0, PIX 2603.25, Blender, NVIDIA FLIP (`python -m flip_evaluator`), Vulkan SDK 1.3.283, Nsight Systems 2022.4, Nsight Compute 2022.3. Nsight Graphics needs a manual install.

## DDGI (measured lessons)

- **Leak fixes by measured effect:** double-sided distance (84×), then probe density (22×). Probe texture resolution made no difference. Reducing bias made leaks *worse*, by up to 34×. Source: the HomeFig DDGI branch.
- **View bias:** 6 → 0 cut image error against a path tracer from 18.2% to 12.4% RMS on two-rooms. Bias pulled into furniture caused black patches. Source: the audit, the HomeFig branch.
- **Recursion cap:** NVIDIA's native recursion albedo cap (0.9 vs 1.0) changes indirect light by about 23%. Match it when comparing against native. Source: the audit.
- **Reference cameras:** build rays from an explicit camera basis, not inverse-MVP, which loses precision at near 0.01 / far 2000. Source: the audit.
- **Light groups:** while a group updates, its bounce reads only its own layer. The sky is a fixed layer. Bake groups in real colors. Source: the HomeFig branch.
- **Per-pixel BVH visibility:** about 12 ms per frame at 720p on a 90-triangle scene. Don't put it in the default shading path. Source: the audit.
