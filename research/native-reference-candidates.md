# Native reference candidates and builder handoff

Initial audit: 2026-10-27 (Japan date). This is build preparation; no repositories were cloned, built, launched, or benchmarked. Recommendations below are engineering choices, not measured rankings.

## Local observations

- The workspace was empty when inspected. No applicable `AGENTS.md` was found at the workspace or its checked ancestor directories.
- A shallow inventory of `C:\Users\denni\Documents\dev` and its immediate child directories found no WickedEngine, RTXGI, Vulkan, or kajiya-named checkout. `C:\Users\denni\Documents\dev\webgpu-gi-bench` does not exist. This does not rule out differently named repositories or checkouts elsewhere; no broader drive search was performed.
- CMake **3.24.0** is installed. Git, `glslc`, `dxc`, and `vulkaninfo` resolve on PATH. `VULKAN_SDK=C:\VulkanSDK\1.3.283.0`.
- Visual Studio **2022 Community** with the x64 C++ tools component was found using `vswhere`; MSVC **14.40.33807**, 14.29, and 14.16 compiler paths exist. `cl`/MSBuild/Ninja are not on the current shell PATH. A developer shell or explicit VS CMake generator is needed. Rust/Cargo did not resolve on PATH.
- `vulkaninfo --summary` enumerated **NVIDIA GeForce RTX 3060 Laptop GPU**, Vulkan **1.4.325**, driver **592.00**, and **Intel UHD Graphics**, Vulkan **1.3.275**. The full Vulkan report lists `VK_KHR_acceleration_structure`, `VK_KHR_ray_query`, `VK_KHR_ray_tracing_pipeline` and corresponding enabled-capability values as true in the discrete-device report. Pin the NVIDIA adapter explicitly in future runs.
- Installed SDK headers/tools are Vulkan 1.3 despite the NVIDIA driver's Vulkan 1.4 support. The current NVIDIA tutorial's SDK 1.4 prerequisite is therefore unmet. VRAM capacity, thermal/power behavior, and successful shader compilation remain unverified.
- Vulkan loader emitted an OBS layer/API-version warning. Record active layers and overlays in a future run manifest; this was enumeration, not a validation-clean renderer run.

## First-build shortlist

| Order | Reference | Why use it | Constraints and capture status |
| --- | --- | --- | --- |
| 1 | [RTXGI-DDGI 1.3 Test Harness](https://github.com/NVIDIAGameWorks/RTXGI-DDGI) | Closest reference for probe irradiance/distance storage, visibility weighting, relocation, classification, and temporal updates; includes both Vulkan and D3D12 harnesses. | Windows requirements in README fit the observed toolchain provisionally. Requires an RT-capable device for its live update implementation. Built-in Cornell first, then two rooms. PNG capture support exists; linear HDR and intermediate probe export need verification or a small patch. |
| 2 | [Wicked Engine](https://github.com/turanszkij/WickedEngine) | One engine can provide DDGI, surfel GI, path-traced comparisons and software tracing variants. Good follow-up systems reference. | Windows uses D3D12 by default; `vulkan` launch argument selects Vulkan. Larger build and material/lighting stack makes strict parity more work. Capture/export automation is an investigation task. |
| 3 | [NVIDIA Vulkan RT tutorial v2](https://github.com/nvpro-samples/vk_raytracing_tutorial_KHR) | Progressive raster-to-RT scaffold for an owned minimal renderer. Useful after reproducing existing DDGI. | Requires Vulkan **1.4+ SDK**, `nvpro_core2`, CMake 3.18+, RT GPU. SDK upgrade or a separately reviewed older revision is needed here. Do not silently switch between v2 and legacy `master`. |
| 4 | [vk_mini_path_tracer](https://github.com/nvpro-samples/vk_mini_path_tracer) | Compact educational compute/ray-query tracer, useful for a small linear-image and hit-record exporter. | Uses NVVK framework helpers and hardware acceleration structures; short application code is not a dependency-free renderer. Verify exact extension/build dependencies at the pinned revision. Not DDGI. |
| 5 | [Khronos Vulkan Samples](https://github.com/KhronosGroup/Vulkan-Samples) | Non-RT raster/compute scaffold plus a separate basic RT sample; benchmark/stop-after-frame/screenshot CLI mechanisms provide useful harness examples. | Current code requires C++20; base requirement Vulkan 1.1, individual samples differ. Full repository/assets are large. Prefer one bounded sample target. Screenshots are not linear-light parity captures. |

Primary build sources: [RTXGI README](https://github.com/NVIDIAGameWorks/RTXGI-DDGI/blob/main/README.md), [RTXGI Quick Start](https://github.com/NVIDIAGameWorks/RTXGI-DDGI/blob/main/docs/QuickStart.md), [Wicked README](https://github.com/turanszkij/WickedEngine/blob/master/README.md), [Wicked documentation](https://github.com/turanszkij/WickedEngine/blob/master/Content/Documentation/WickedEngine-Documentation.md), [tutorial README](https://github.com/nvpro-samples/vk_raytracing_tutorial_KHR/blob/v2/README.md), [mini tracer README](https://github.com/nvpro-samples/vk_mini_path_tracer/blob/main/README.md), [Khronos README](https://github.com/KhronosGroup/Vulkan-Samples/blob/main/README.adoc), [basic RT sample](https://docs.vulkan.org/samples/latest/samples/extensions/ray_tracing_basic/README.html).

### Licensing and pins

Verified root license files identify [Wicked as MIT](https://github.com/turanszkij/WickedEngine/blob/master/LICENSE.txt), [tutorial v2 as Apache-2.0](https://github.com/nvpro-samples/vk_raytracing_tutorial_KHR/blob/v2/LICENSE), [mini tracer as Apache-2.0](https://github.com/nvpro-samples/vk_mini_path_tracer/blob/main/LICENSE), and [Khronos Samples as Apache-2.0](https://github.com/KhronosGroup/Vulkan-Samples/blob/main/LICENSE). Dependencies and scene assets need their own provenance entries. [RTXGI-DDGI uses the NVIDIA RTX SDK license](https://github.com/NVIDIAGameWorks/RTXGI-DDGI/blob/main/License.txt); keep it a separately tracked reference dependency and determine reuse terms before incorporating its code into a public core.

Observed upstream commit candidates; the builder must record the actually checked-out commit and submodules:

| Repository/ref inspected | Commit |
| --- | --- |
| RTXGI-DDGI `main` | `f33e496ca31b3f0eec1c4e2cbaa8bb620e337fa6` |
| WickedEngine `master` | `e0dc6a2ad99f01251c113590d9626bd7b2e2b616` |
| Vulkan RT tutorial `v2` | `952a1a8e8b85f25781db7014d4d2bda124435e79` |
| vk_mini_path_tracer `main` | `9d6a0e4242790ddac52a0d8ee02eb7670bf10965` |

The DDGI harness already contains [two-rooms.ini](https://github.com/NVIDIAGameWorks/RTXGI-DDGI/blob/f33e496ca31b3f0eec1c4e2cbaa8bb620e337fa6/samples/test-harness/config/two-rooms.ini), with glTF scene path, multiple cameras, black sky, DDGI settings, and path-tracer options. Its scene units/bias values are sample-specific; do not transplant them into meter-based fixtures. [ImageCapture.h](https://github.com/NVIDIAGameWorks/RTXGI-DDGI/blob/f33e496ca31b3f0eec1c4e2cbaa8bb620e337fa6/samples/test-harness/include/ImageCapture.h) exposes PNG capture, which is evidence of capture code, not proof an unattended Vulkan capture succeeds.

## Additional research renderer

[Embark's kajiya](https://github.com/EmbarkStudios/kajiya) is a Rust/Vulkan research renderer, dual MIT/Apache-2.0, supporting Windows/Linux and selected RT hardware. It is archived and unmaintained; Rust is not currently on PATH here. Its [GI overview](https://github.com/EmbarkStudios/kajiya/blob/main/docs/gi-overview.md) includes an interior compared to its path tracer and documents small-geometry leakage/instability. Useful source study and later contrast to regular-grid DDGI; not a first build or a correctness oracle. No unverified homework repository is promoted into the initial build queue.

## Status vocabulary

Every candidate above is currently **discovered** only.

- **Discovered:** primary repository and intended role identified.
- **Buildable:** exact dependency/toolchain inventory plus successful configure for this machine, no unresolved prerequisites.
- **Built:** named target produces executable; log and revision recorded.
- **Launched:** intended backend/device and scene confirmed; validation reviewed.
- **Captured:** repeatable fixed-frame image/data export and run manifest obtained.
- **Parity:** common fixture, transport/material conventions, and stated numerical tolerances tested against a reference. An attractive screenshot does not establish parity.

## Bounded future builder-agent brief

> Own only `native/references/rtxgi-ddgi/`, its build directory, and `artifacts/native/rtxgi-ddgi/` under this workspace. First inspect any existing checkout without resetting it. Acquire/pin RTXGI-DDGI at the recorded revision with required submodules and record license/dependency provenance. Use VS2022 x64, the existing Vulkan SDK, and the NVIDIA adapter. Build only `TestHarness-VK` in Release; record exact configure/build commands and logs. CMake downloads DXC/Agility dependencies, so inspect those steps and report failed downloads precisely. Do not install or upgrade global tools/drivers. Try the included Cornell fixture, then the existing two-room configuration. Confirm Vulkan/device choice; capture fixed camera and frame count plus config and validation output. Report built, launched, and captured separately. If blocked, return the smallest concrete prerequisite or patch; do not migrate into a new renderer. Keep any patches separate from upstream. Do not run timed GPU benches concurrently with other agents.

After that milestone, give another bounded agent Wicked DDGI/surfel captures, and another an owned minimal Vulkan raster/compute fixture. Keep their output contracts shared: geometry hash, meters/coordinate convention, camera matrices, light units, geometric normals, linear diffuse albedo, pre-tonemap indirect diffuse, probe irradiance/distance/state, deterministic seed, per-pass timing and device manifest. Matching sample counts/rays is useful for algorithm tests; native hardware RT and browser software traversal need separate performance comparisons. A converged path tracer with verified scene conventions supplies image-quality references; native DDGI supplies algorithm behavior references.
