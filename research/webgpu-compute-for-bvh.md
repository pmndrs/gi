# WebGPU compute building blocks for a GPU BVH builder

Written 2026-11-04. Research only, no code changed.

Legend: **[V]** = verified this session from a fetched or searched source (URL given). **[K]** = from prior knowledge of the spec or ecosystem, not re-checked now. **[I]** = my inference or judgment. Treat [K] and [I] as things to confirm with a device test before relying on them. Time was short: several answers below are [K], and the throughput numbers are mostly missing (see section 1).

## Summary table

| Primitive | Safe everywhere? | Notes |
|---|---|---|
| 32-bit `atomicAdd/Min/Max/Or/CompareExchange` on `u32`/`i32` in storage and workgroup memory | Yes [K] | Core WGSL. No float atomics, no 64-bit atomics. |
| `workgroupBarrier`, `storageBarrier`, workgroup memory, `var<workgroup>` | Yes [K] | Default workgroup memory 16384 bytes, 256 invocations per workgroup. |
| Multi-pass radix sort (reduce, scan, scatter; no look-back) | Yes [I] | Needs only barriers and dispatch boundaries. |
| Reduce-then-scan prefix sum (3 dispatches) | Yes [I] | No cross-workgroup spin waiting. |
| Bottom-up refit with a per-node parent counter (`atomicAdd`) | Yes [I] | Uses atomics and fences only, no spinning. |
| Decoupled look-back (OneSweep, single-pass scan) | **No guarantee** | Needs forward progress between workgroups. Spec does not give it. |
| `subgroups` feature | Needs feature check | Shipped in Chrome; Safari and Firefox unclear. |
| `shader-f16`, `timestamp-query` | Needs feature check [K] | Optional. |
| Raising `maxStorageBufferBindingSize` / `maxBufferSize` | Needs limit request | Defaults are low (128 MiB and 256 MiB). |
| SharedArrayBuffer | **No** for a drop-in library | Needs COOP/COEP on the host page. |

## 1. Sort and scan in WGSL

### What exists

| Project | What it is | Source |
|---|---|---|
| `webgpu-sorting` (AICL-Lab / build-workbench) | TypeScript library with bitonic sort and radix sort for large `Uint32Array`. Whether its radix sort uses look-back is not confirmed. **[V]** that it exists. | https://github.com/AICL-Lab/webgpu-sorting (searched 2026-11-04) |
| `webgpu-radix-sort` (npm) | Radix sort following the "Fast 4-way parallel radix sorting on GPUs" paper (a multi-pass, non-look-back design). **[V]** that it exists. | https://npmjs.com/package/webgpu-radix-sort/v/1.0.0 |
| `wgpu_sort` (KeKsBoTer) | Rust/wgpu radix sort, used for Gaussian splat sorting. Runs on WebGPU via wgpu. Algorithm details not read. | https://github.com/KeKsBoTer/wgpu_sort |
| `b0nes164/GPUSorting` | Reference OneSweep and DeviceRadixSort in CUDA, D3D12 and Unity-style compute shaders. Not WGSL. **[V]** | https://www.github.com/b0nes164/GPUSorting |
| GPUOpen "Boosting GPU Radix Sort" | Memory-efficient OneSweep variant with circular buffers; useful porting reference. **[V]** | https://gpuopen.com/learn/boosting_gpu_radix_sort/ |
| Mastodon/Zulip report | Someone ported OneSweep to WebGPU, with warp-local multi-split done through shared memory instead of subgroups. Second-hand, no link found. **[V]** that it was claimed, unverified otherwise. | surfaced by search, 2026-11-04 |
| FidelityFX parallel sort, Thearling and Smith, CUB | Comparison baselines in the GPUSorting benchmarks. | same repo |

Also known to me but not re-checked **[K]**: three.js and Babylon.js both carry GPU sort code for splats or particles; Gaussian-splat WebGPU viewers (e.g. several GitHub projects) use a multi-pass radix sort because splat sorting is a similar 32-bit key problem.

### Forward progress and decoupled look-back

- The author of GPUSorting warns that OneSweep's decoupled look-back "relies on forward thread-progress guarantees", so OneSweep is less portable than DeviceRadixSort. He says it appears to run correctly on anything that is not mobile or WARP, but that this is untested. **[V]** https://www.github.com/b0nes164/GPUSorting
- The WebGPU/WGSL spec says nothing that guarantees one workgroup can spin on a flag written by another workgroup that is not yet resident. **[K]** (WGSL spec, "Uniformity" and memory model sections; I did not re-read the current text.) Workgroups are only guaranteed to make progress in the sense that a dispatch eventually completes; the spec does not promise that earlier-indexed workgroups run before later ones.
- Look-back is safe in practice only if every workgroup that is waited on has already started. Standard trick: take the workgroup's tile index from an `atomicAdd` on a counter instead of from `workgroup_id`, so tiles are handed out in the order workgroups actually start. That removes the classic deadlock (waiting on a tile that is not resident). **[K]** This is what CUB and OneSweep do.
- Even with that trick, it relies on the hardware scheduler being sane. Risk areas **[I]**: software rasteriser backends (WARP, SwiftShader/llvmpipe), mobile GPUs, and any driver that preempts or time-slices oddly. A hang here is a GPU timeout and device loss, not a wrong answer.
- **Recommendation [I]:** do not make decoupled look-back the only path. Ship a reduce-then-scan radix sort (3 or 4 small kernels per 4-bit or 8-bit digit pass) as the guaranteed path. Add look-back as an optional fast path behind a self-test (sort a random array on first use, compare to a CPU result, and watchdog with a bounded spin count). The 262k-triangle LBVH sort is small enough that this is likely fine without look-back (see below).

### Reduce-then-scan alternatives

| Scheme | Dispatches | Forward-progress needed | Notes |
|---|---|---|---|
| Reduce-then-scan (per-block sums, scan the sums, add back) | 3 | No | Reads input twice. Well known, simple. **[K]** |
| Scan-then-propagate | 3 | No | Same cost class. **[K]** |
| Chained scan | 1 | Yes (waits on predecessor) | Worse than look-back, same risk. **[K]** |
| Decoupled look-back | 1 | Yes | Fastest; see above. **[K]** |
| Hierarchical radix sort with per-workgroup histogram + global scan of histograms | 3 per digit pass | No | Basis of DeviceRadixSort style and of most WebGPU libraries. **[K]** |

### Throughput numbers

I did not find a published keys-per-second figure for a WGSL radix sort in this session. Do not quote one from this document. What can be said:

- **[K]** CUDA OneSweep on a high-end NVIDIA card sorts 32-bit keys in the multi-GPU-billion keys/s range at large sizes (the OneSweep paper reports several G keys/s); a portable compute-shader multi-pass sort is typically several times slower than that. These are recollections, not verified for the current hardware.
- **[I]** For 262k keys (Sponza) a multi-pass radix sort is launch-latency bound, not bandwidth bound. Expect low single-digit milliseconds on a desktop GPU even with a plain implementation. For 100k instances the same applies.
- **Action:** the benchmark is cheap to produce ourselves. Measure 262k, 1M and 4M keys on the NVIDIA and Intel Windows boxes and on the Mac, using the existing `gpu-tester` flow, before choosing the algorithm. Those are the numbers that should go into the design.

## 2. Subgroups

### Status

| Browser | Status | Evidence |
|---|---|---|
| Chrome / Edge (Chromium) | Shipped on desktop in **Chrome 134** (origin trial was 128 to 131). Chrome's own post says the `subgroups` feature is available and to request it with `requiredFeatures` when the adapter exposes it. **[V]** | Intent to Ship thread https://groups.google.com/a/chromium.org/g/blink-dev/c/xteMk_tObgI ; https://developer.chrome.com/blog/new-in-webgpu-134 (fetched 2026-11-04). Whether Android Chrome shipped at the same time: not checked. |
| Chrome follow-ons | `subgroup_id` built-ins intent approved November 2025; `subgroup_uniformity` and a "subgroup size control" extension have intents (the latter June 2026). **[V]** that the intents exist; I did not confirm shipped versions. | https://groups.google.com/a/chromium.org/g/blink-dev/c/SV75BHCUJz0 ; https://groups.google.com/a/chromium.org/g/blink-dev/c/Lx06y9Rdzvs |
| Safari / WebKit | Apple approved the base proposal in the standards process. One secondary report (a German news site) says Safari Technology Preview 249 added WebGPU subgroups. Not confirmed from WebKit release notes, and no shipped-Safari version found. **Unverified.** | https://www.it-boltwise.de/?p=2371351 ; blink-dev thread above |
| Firefox | Mozilla said it intends to support it later; standards position on `subgroup_id` pending; no shipping signal found. **[V]** that no signal was found. | blink-dev threads above |
| caniuse / web-features | The web-features explorer lists it as limited availability with every browser unsupported, but that page appears stale against the Chrome 134 ship. | https://web-platform-dx.github.io/web-features-explorer/features/webgpu-subgroups |

Because a web-features page and the Chrome post disagree and I could not read caniuse, **treat the exact late-2026 state of Safari and Firefox as unknown and check it** before planning around it.

### Which GPUs and backends

Not verified. From knowledge **[K/I]**: Dawn implements it on D3D12 (Shader Model 6.0+ wave ops), Vulkan (subgroup operations), and Metal (SIMD-group). So NVIDIA, AMD, Intel and Apple GPUs on those backends should expose it; very old D3D12 drivers or software adapters may not. Subgroup size varies (Intel 8/16/32, NVIDIA 32, AMD 32 or 64, Apple 32), and WGSL gives `subgroup_size` only as a built-in value with a min/max range on the adapter (`subgroupMinSize` / `subgroupMaxSize` limits), so code must not assume 32. Check `adapter.features.has('subgroups')` and `adapter.info.subgroupMinSize/MaxSize` on each test device.

### Speed-up

No measured sort/scan figure found. Chrome's post cites 2.3x to 2.9x on "some devices" for Google Meet's pipeline during the trial, which is not a scan benchmark. **[V]** (https://developer.chrome.com/blog/new-in-webgpu-134). **[I]**: subgroup scan/ballot typically removes most barrier traffic in the in-block part of a scan and in the radix histogram/ranking step, so 1.5x to 3x on those kernels is plausible, with a much smaller effect on the end-to-end build (dominated by memory and launch latency for 262k items). Measure.

### Fallback when absent

1. Write the kernels with a pure workgroup-memory path (Hillis-Steele or Blelloch scan in `var<workgroup>`), which works on every WebGPU device. This is the baseline and the thing to test first.
2. Generate the subgroup variant from the same source with a pipeline-overridable constant or a string template, and select at runtime on `device.features.has('subgroups')`.
3. Requires `enable subgroups;` at the top of the shader, which fails to compile if the feature was not requested at device creation. Keep the two WGSL variants as separate modules. **[K]**
4. Do not assume the subgroup size is uniform across the workgroup or equals any specific number; use only operations whose results do not depend on size.

## 3. Atomics and memory

### Atomics

- WGSL atomics are 32-bit only: `atomic<u32>` and `atomic<i32>`. Operations: load, store, add, sub, max, min, and, or, xor, exchange, compareExchangeWeak. Available in `storage` (read_write) and `workgroup` address spaces. No float atomics. No 64-bit atomics. **[K]** (WGSL spec, atomic builtin functions.) A newer 64-bit atomics proposal exists in the working group but I did not verify it shipped; do not rely on it.

### Float AABB min/max without float atomics

| Technique | How it works | Cost / caveats |
|---|---|---|
| Order-preserving float to uint map plus `atomicMin`/`atomicMax` | Reinterpret float bits as `u32`; if sign bit set, flip all bits, else flip the sign bit only. Then unsigned integer ordering matches float ordering. Convert back after. **[K]** | Needs one-time initialisation of the accumulator to the +inf or -inf mapped value. Fine for NaN-free inputs. One atomic per axis per bound. Contention is high at the root, so reduce within the workgroup first (workgroup atomics or subgroup min/max), then do one global atomic per workgroup. Best for the "scene AABB" step before Morton coding. |
| Bottom-up refit with a parent counter | Each leaf thread computes its box, then climbs: `if (atomicAdd(&counter[parent], 1) == 0) return;` the first child to arrive exits, the second child (which knows both children are done) merges their boxes and continues to the grandparent. **[K]** This is the standard Karras-style LBVH refit. | No float atomics. Needs a storage barrier / memory ordering so the first child's box writes are visible to the second child. In WGSL use `storageBarrier()` plus the atomic's ordering; because atomics are relaxed in WGSL, the visibility of the non-atomic box write is the thing to test on all vendors **[I]**. A safe variant: have the first child write its box then do the `atomicAdd`; if memory model issues appear, store boxes as `atomic<u32>` words (6 words) via `atomicStore` / `atomicLoad` so everything is atomic. Counters must be reset to 0 each refit (clear pass). |
| Level-by-level refit | Store tree in an order where each level is contiguous; dispatch one pass per level from the bottom up. | No atomics needed, no memory-model risk, but needs about log2(N) to 2 log2(N) dispatches; for a balanced LBVH that is about 18 passes at 262k. Fine for a handful of dispatches per frame, bad if launch overhead dominates. |

**[I]** Recommendation: parent-counter refit with box data written plainly and a `storageBarrier()`, validated by a stress test on each vendor; fall back to level-by-level if any device shows stale reads.

### Limits (spec defaults vs typical hardware)

Defaults are from the WebGPU spec limits table **[K]** (https://www.w3.org/TR/webgpu/#limits, not re-fetched). "Typical" columns are from memory of public adapter reports (e.g. gpuinfo-style reports for WebGPU) and are **[I]**; query `adapter.limits` on the test machines.

| Limit | Spec default | Typical desktop adapter | Typical mobile / Apple [I] |
|---|---|---|---|
| `maxStorageBufferBindingSize` | 134217728 (128 MiB) | 2 GiB - 4 GiB - 1 on Chrome with D3D12 or Vulkan, often exactly 2147483648 or 4294967292 | Often 128 MiB to 1 GiB; Apple Silicon reports high (several GiB) |
| `maxBufferSize` | 268435456 (256 MiB) | 2 GiB to 4 GiB | 256 MiB to a few GiB |
| `maxComputeWorkgroupStorageSize` | 16384 bytes | 32768 or 49152 on many desktop GPUs | 16384 to 32768 |
| `maxComputeInvocationsPerWorkgroup` | 256 | 256 to 1024 | 256 to 1024 |
| `maxComputeWorkgroupSizeX/Y/Z` | 256 / 256 / 64 | 256 to 1024 / same / 64 | similar |
| `maxComputeWorkgroupsPerDimension` | 65535 | 65535 | 65535 |
| `maxStorageBuffersPerShaderStage` | 8 | 8 to 16 or more | 8 |
| `maxStorageBufferBindingSize` needed for Sponza BVH | n/a | **[I]** 262k tris: nodes ~ 524k x 32 B = 16 MiB; triangles/indices a few MiB. Fits default easily. | |
| 100k-instance TLAS | n/a | **[I]** ~200k nodes x 32 B = 6.4 MiB. Fits default. | |

Takeaways **[I]**: at the target sizes nothing needs raised limits for the tree itself. Design kernels for the default 16 KiB workgroup memory, 256 invocations and 8 storage buffers per stage, and request higher limits only as an optimisation. Splitting the sort across more than 8 storage bindings is a practical constraint to plan for (keys in/out, values in/out, histograms, counters, params = already 6+).

## 4. CPU and GPU interplay

### SharedArrayBuffer

- `SharedArrayBuffer` requires a cross-origin-isolated page (`Cross-Origin-Opener-Policy: same-origin` plus `Cross-Origin-Embedder-Policy: require-corp` or `credentialless`). **[K]** (https://web.dev/articles/coop-coep, not re-fetched.)
- For a library dropped into arbitrary sites that is a non-starter as a requirement: COEP breaks cross-origin images, iframes, embeds and ads that do not send CORP/CORS headers, and the library cannot set headers on the host page. **[I]**
- Nothing in this design needs it. WebGPU itself does not require cross-origin isolation. **[K]** Use plain `ArrayBuffer`s with `queue.writeBuffer`, transfer lists for workers, and keep the BVH on the GPU. If SAB is used at all it should be a detected optimisation (`crossOriginIsolated === true`), never a requirement.

### `mapAsync` readback cost

- Readback is asynchronous by design (the page cannot look at the data until the GPU is finished with the buffer). Implementations may add a copy or a GPU-process round trip. **[V]** https://lists.w3.org/Archives/Public/public-gpu/2020Feb/0005.html (2020 design discussion).
- WebKit bugs report that `mapAsync` waits for unrelated command buffers (longer waits than needed) and that mapping a whole buffer to read a few bytes is wasteful. **[V]** https://bugs.webkit.org/show_bug.cgi?id=272804 ; https://bugs.webkit.org/show_bug.cgi?id=273028. Status of these bugs not checked.
- **[I]** Practical cost: latency of one to several frames plus a copy into a `MAP_READ` staging buffer; bandwidth is not the issue for 16-32 MiB. Never block a frame on it.
- **Does the tree need to come back to the CPU? [I]** Only for CPU picking, CPU physics, debug view, or caching the built tree (e.g. to IndexedDB so the 262k-triangle build is paid once). Options: (a) do picking on the GPU with a one-ray compute dispatch and read back a few bytes, which is cheap; (b) read back only for a build cache, once, off the hot path; (c) keep CPU-side three-mesh-bvh for the few objects that need CPU queries. Caching a built Sponza tree is the most useful reason for a readback.

### Workers

- `navigator.gpu` is available in dedicated workers in Chrome and Safari; Firefox availability not verified here. **[K]**
- A `GPUDevice` and its buffers belong to one device and cannot be transferred between devices or shared with another worker's device; `GPUBuffer` is not transferable. **[K]** (the spec has no transfer steps for these objects). So a worker-owned device would also need to do the rendering, via `OffscreenCanvas`, or copy data through the CPU. **[I]**
- Practical design: build the BVH on the same device as the renderer, on the main thread (WebGPU command encoding is cheap; the work runs on the GPU). A worker is only useful for CPU-side prep (parsing glTF, packing triangle buffers) and for an `OffscreenCanvas` renderer that owns everything, including the build.

## 5. Compile and timing

- Pipeline creation can stall. `createComputePipeline` may compile synchronously on the calling thread in some implementations; `createComputePipelineAsync` returns a promise and lets the browser compile in the background. Use the async form and create all build kernels up front. **[K]**
- On Windows, Chrome translates WGSL through Tint to HLSL, then compiles with FXC or DXC to DXIL/DXBC; this is the slow step, usually tens to hundreds of ms per non-trivial compute shader, and large unrolled loops or deep inlining can reach seconds. Dawn caches compiled pipelines on disk between sessions, so the cost is mostly first run. **[K]** The repo's own `webgpu-tsl-compile` skill covers this for three.js and TSL and is the better local source on D3D12/DXC stalls. **[V]** that the skill exists; contents not re-read here.
- For a handful (about 8 to 15) of build kernels **[I]**: budget roughly 0.5 to 3 s of cumulative first-run compile on Windows D3D12 if created in parallel with `createComputePipelineAsync`, and much less on a second visit due to caching. This is an estimate to be replaced with a measurement. Keep kernels small and avoid `override`-driven recompilation per scene; one pipeline per kernel with a uniform buffer for parameters.
- Warm up at load behind a loading state, not at the first frame that needs a build.
- Timing: the `timestamp-query` feature gives per-pass GPU time but is optional, and Chrome rounds or quantises timestamps for security unless a developer flag is on. **[K]** Use it for development and report CPU-side `submittedWorkDone` time as the fallback.

## Biggest risk and recommendation

The biggest risk is relying on decoupled look-back (single-pass scan / OneSweep). WebGPU gives no forward-progress guarantee, the best reference author calls it less portable, and a failure shows up as a GPU hang or device loss, most likely on mobile or software adapters. For the sizes here (262k to a few hundred thousand keys) a plain reduce-then-scan radix sort has a negligible penalty, so use it as the default and treat look-back as an optional, self-tested fast path. Subgroups are shipped in Chromium but their Safari and Firefox status is unconfirmed, so they need a feature check with a workgroup-memory fallback. SharedArrayBuffer should be avoided entirely.

## Open items to verify (not done)

1. Current Safari and Firefox subgroup status (WebKit release notes, Firefox bugzilla, caniuse).
2. Real throughput of a WGSL radix sort at 262k/1M keys on the NVIDIA, Intel and Apple test devices.
3. Actual `adapter.limits` and subgroup min/max sizes on those devices.
4. Memory visibility of plain box writes across the parent-counter refit on every vendor.
5. Measured first-run pipeline compile time on Windows D3D12 for the real kernel set.
