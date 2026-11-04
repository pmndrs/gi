# GPU BVH construction and traversal: what to use for webgpu-ddgi

Research only. Written 2026-11-04 against `docs/design/trace.md` (the `origin/m0` copy is identical to the working copy).

**How to read the numbers.** Every number is tagged:
- **[P]** = published in the cited paper or page (I read the text myself unless the row says "secondary").
- **[S]** = secondary source (a web summary, not the paper). Treat as unverified.
- **[E]** = my estimate. It is reasoning, not a measurement.

**Hardware caveat for every published build time.** All of them come from native CUDA/HIP/DPC++ code on desktop GPUs, with 64-bit radix sort, hardware-specific wave sizes and large workgroups. Our WebGPU build will be slower by an amount nobody has published. I give estimates where it matters.

## Our starting point (from `trace.md`)

| Item | Today |
|---|---|
| Per-mesh build | Binned SAH on the CPU, 16 bins, in a worker. Sponza (262k tris) = 28 s [P, our bench] |
| Node | 64 B, two child AABBs (Aila-Laine) |
| Triangle | 48 B watertight (Woop) |
| Stack / order | 64-entry private stack, nearer child first |
| Top level | CPU binned SAH. 1-10 ms on real scenes, 86-121 ms for 10k instances [P, our bench] |
| Trace speed | 61 Mrays/s closest hit, Sponza, RTX 3060 Laptop (watertight); three-mesh-bvh 87 [P, our bench] |
| Target | 100 Mrays/s closest + 100 Mrays/s any-hit; start-up <= 2 s |

---

# Textbook vs state of the art

## The short answer

- **Build is the part that is truly textbook.** A CPU binned-SAH build is correct but about 4 orders of magnitude slower than a modern GPU builder. H-PLOC builds Sponza (0.3M tris) in 0.74 ms on a Radeon 7900 XT [P]. Even if WebGPU on a 3060 Laptop is 50x worse [E], that is about 40 ms, against our 28 s.
- **The tree layout and the walk are not far from textbook, and that is fine.** A BVH2 with 64 B two-child nodes, nearest-first and a stack is still what the fastest *portable* GPU tracers use. The one big published win beyond it is the compressed 8-wide tree (CWBVH, 1.9-2.1x), and it depends on features WebGPU may not expose well. See the table.
- **The biggest untested lever on trace speed is tree quality, not tree shape.** Published SAH cost differences between builders are 1.04-1.38x (H-PLOC paper). Spatial splits (SBVH) give more on some scenes (about 3x on Power Plant, JCGT 2022).

## Part by part

| Our part | What is textbook about it | Fastest robust method in production today | Published speedup | Verdict for us |
|---|---|---|---|---|
| **Build: binned SAH, CPU, 16 bins** | Wald 2007. Top-down, high quality, serial-ish | **H-PLOC** for speed (a bottom-up GPU builder; one kernel after the sort). **PLOC++** and **ATRBVH/TRBVH** for slightly better quality. Offline: **SBVH** (spatial splits) | H-PLOC: 0.74 ms on Sponza vs 633 ms for the best-quality reinsertion builder (PRBVH), 7900 XT [P]. Quality: BVH4 SAH cost 1.09x PRBVH, path-tracing time 1.11x [P] | Replace. Biggest win available. |
| **Leaf rule** (<= 4 tris, SAH stop) | Standard | Same, applied by *collapsing* a binary tree bottom-up with SAH (done inside Karras-Aila 2013 and the H-PLOC conversion) | Included in the 1.04-1.12x figures above [P] | Keep the rule; do the collapse on the GPU. |
| **Node: 64 B BVH2, two child boxes** | The Aila-Laine layout. tinybvh calls it `BVH_GPU`. | **CWBVH**: 8-wide, 80 B per node, boxes quantized to 8 bits per plane, child order chosen at build time, compressed stack | 1.9-2.1x ray casts vs Aila et al. 2012 (Titan X Pascal and Maxwell, CUDA); memory 35-60% of a normal BVH [P, Ylitie 2017] | **Not a free win.** Needs byte-permute and popcount tricks that are PTX-specific. JCGT 2022 found plain (uncompressed) wide BVHs *lose* speed as width grows [P]. See "Wide BVH" below. |
| **Stack: 64-entry private array** | Plain per-thread stack | Short stack in shared memory with spill (Ylitie 2017 keeps the first N entries in shared memory and spills the rest); or stackless/restart trail | No number retrieved for the stack change alone. Ylitie's 1.9-2.1x includes it. | **Low priority.** Our measured max depth is 19 of 64 [P, our doc]. The cost is registers, which our design already plans to test with 32. |
| **Order: nearer child first** | Aila-Laine | Same for BVH2. For BVH8 CWBVH uses octant order picked at build time (no per-ray distance sort) | Ylitie: octant order costs few extra tests vs full distance order (Table 2 of paper) [P, not copied here] | Keep. |
| **Triangles: Woop watertight, 48 B** | Standard for robust results | Same. CWBVH also uses 48 B padded vertex triangles (Ylitie 2017) | None to gain | Keep. Fix the slow `a[kx]` indexing we already identified. |
| **Ray scheduling: one thread per ray, no sorting** | Plain | **Ray sorting/binning** by origin and direction; **persistent threads with dynamic ray fetch** | Sorting: +7% secondary, +8% shadow on average; Crytek Sponza +23% / +39% (binary BVH, AMD GPU) [P, JCGT 2022]. The gain shrinks with wide BVHs. | **Cheap test.** DDGI probe rays are already structured (fixed origin per probe). Spend a day on it after the build work. |
| **Spatial splits (SBVH)** | Not in our design | SBVH is the best binary-tree trace quality in JCGT 2022 | Up to ~3x on Power Plant; scene dependent [P]. Build is 5196x LBVH time (CPU/offline) [P] | Worth it for *static baked* meshes only. Triangle duplication costs memory in our "triangles in tree order" layout. Not for deforming meshes. |
| **Top level: CPU binned SAH** | Textbook | GPU LBVH or H-PLOC over instance boxes every frame | H-PLOC: BVH4 over 0.5M instances 1.1 ms, 1M instances 2.21 ms (LBVH 1.96 ms), 7900 XT [P] | Replace once instance motion is common. |
| **Quantized nodes (without going wide)** | -- | Possible on BVH2 too (compressed boxes relative to parent) | Ylitie shows 8-bit quantization raises intersection counts by 3.4% [P] | Option after we measure whether we are memory-bound. |
| **Newer node-layout ideas** | -- | Merged internal+leaf nodes, bandwidth-aware SAH (HPG 2026): 48.3% smaller BVH, 15.8% less memory traffic, 31.7% less render time vs a wide-BVH baseline [P, Utah project page]. UBVH (HPG 2025): 1.2-11.8x faster than AABB BVH on incoherent rays [P] | Both target *hardware* RT. Only the layout ideas could transfer. | Watch, do not adopt. |

## Rank: fastest and stable (widely used and robust)

**Build (static meshes), best first for us:**

1. **LBVH (Karras 2012 / Apetrei 2014).** Most widely reimplemented GPU builder. Fast but poor quality: 1.10-1.38x worse SAH than the best, 1.28-1.37x slower to trace in the H-PLOC paper [P]. Use as the safe first milestone.
2. **H-PLOC.** From AMD and collaborators, HPG 2024. Quality within 1-2% of PLOC++/ATRBVH, build within 15% of LBVH [P]. The newest of the list, but it is built from two older, well-tested parts (Apetrei LBVH and PLOC++). It is the best fit for a first-time port to WGSL because its main kernel has no inter-workgroup waiting (see platform section). Stability: one paper, no WebGPU port known to me.
3. **PLOC / PLOC++.** Widely cited, quality about 1.06-1.12x of best. Needs 1024-thread workgroups and 32 KB shared memory in the reference (Intel Arc) [P], above WebGPU defaults. Also needs a prefix scan between iterations.
4. **TRBVH (Karras-Aila 2013) / ATRBVH.** Strong quality (96% of SweepSAH trace speed average, [P]). Treelet kernel assumes 32-thread warps and 768 B shared memory per treelet. Slower and more complex than H-PLOC.
5. **SBVH (offline, CPU).** The quality ceiling for binary trees [P, JCGT 2022]. Production use as a bake step (tinybvh, Embree).

**Traversal, best first:**

1. **BVH2 Aila-Laine walk (what we have).** Still the portable default.
2. **Ray sorting** (small, safe gain).
3. **CWBVH** (largest published gain, hardest to port, many open implementations).
4. **Short stack / stackless** (uncertain benefit for us).
5. **Oriented-box trees (DOBB-BVH)** and **unified-SOBB (UBVH)**: published gains (below) but cost build time and need more math per node. Research stage.

## Public code, license, and port difficulty to WGSL

"Port difficulty" is my estimate [E], assuming 32-bit atomics only, no forward-progress guarantee, and subgroups allowed.

| Method | Public code | License | Language | Port to WGSL |
|---|---|---|---|---|
| LBVH (Karras 2012; Apetrei 2014 single-pass) | Many third-party versions; the paper gives the algorithm. | varies | CUDA/HLSL | **Easy.** Atomic-flag "second arrival continues" is wait-free. Needs a radix sort. |
| H-PLOC | tinybvh has a `useHPLOC` option ("matches NexusBVH's CUDA code, for comparisons" [P, tinybvh README]). I did not locate AMD's own source. | tinybvh = MIT | C++ (CPU) in tinybvh; CUDA elsewhere | **Medium.** Algorithm listings are in the paper. Needs a wave-sized cluster list (use subgroups) and 64-bit keys (emulate). |
| PLOC / PLOC++ | Original PLOC by Meister (project page). PLOC++ is said to be in the Embree repository's `ploc` branch (per H-PLOC references [S]; I did not open it). | Embree = Apache-2.0 | CUDA / DPC++ | **Medium-hard.** Workgroup-size and shared-memory limits (see below). |
| TRBVH / ATRBVH | ATRBVH has a public repo (`swq0553/atrbvh`, license not checked). | unchecked | CUDA | **Hard.** Warp-synchronous treelet DP. |
| CWBVH build + traverse | tinybvh (`BVH8_CWBVH`, with OpenCL, GLSL, DirectX 12, Vulkan traversal examples [P, README]); obvhs (Rust CWBVH builder) | tinybvh MIT; obvhs Apache-2.0 or MIT | C++ / Rust | **Traversal: medium** (GLSL kernel exists; WGSL has `countOneBits`, no byte-permute, so shift/mask instead). **Build: easy** (it is a conversion from BVH2, can stay on the CPU or GPU). |
| Binned SAH, SBVH (CPU) | tinybvh (MIT), Embree (Apache-2.0), three-mesh-bvh (MIT) | MIT / Apache-2.0 | C++/JS | Not needed on GPU. Keep for bake. |
| Bonsai (Ganestam 2015) | tinybvh uses it as the default CPU builder. | MIT | C++ | **Not a GPU method.** A 2016 GPU port was "considerably slower" than the CPU original [P, Lund thesis]. Skip. |
| Parallel reinsertion (PRBVH, Meister-Bittner 2018) | obvhs has it (Apache-2.0 or MIT) | -- | Rust | Not useful on GPU. 633 ms on Sponza at 7900 XT speed [P], for only about 9% SAH gain over H-PLOC [P]. |
| Sorting / scan for WebGPU | Gridwise WebGPU sort (OneSweep with fallback) | license not checked | WGSL | **Already in WGSL**, but see platform notes. |
| Hardware/driver builders | HIPRT (AMD), Radeon Rays (MIT) | HIPRT: GitHub shows NOASSERTION (custom). Radeon Rays: MIT. | C++/HIP | Reference only. I did not verify that either contains H-PLOC. |

## "Buildable from the paper" rating

- **Full** = pseudocode/listings and data layouts are given.
- **Partial** = key steps are clear, details need guessing.
- **Thin** = idea only.

| Method | Rating | What is given | What is missing / must be guessed |
|---|---|---|---|
| Binned SAH BVH2 (Wald 2007) | Full | Algorithm, cost model | Nothing. We already built it. |
| LBVH (Karras 2012) | Full | Radix-tree algorithm listing, refit with atomic counters, Morton details | Duplicate-key handling is described. WebGPU sort not covered. |
| LBVH single pass (Apetrei 2014) | Full (paper), I read it only through H-PLOC's summary | Range-tracking bottom-up | I did not read the original; relying on H-PLOC's description [P]. |
| PLOC (Meister 2018) | Partial | Algorithm 1, radius parameter, subtree collapse (3 bottom-up passes), iteration counts | Several kernels; prefix-scan and compaction details left to the reader. |
| PLOC++ (Benthin 2022) | Partial | Fused kernels, chunk layout (1024 minus 4x radius), 2x32-bit key trick, memory 72 B/prim | Some shared-memory layouts and the single-workgroup top kernel are described in prose. |
| H-PLOC (Benthin 2024) | Partial-to-Full | Algorithms 1 and 2 (main loop, inner loop), memory use (8+4+4+2x32 B per primitive), merge threshold = half wave size, BVH2-to-wide conversion algorithm with forward-progress note | Wave-level code tricks (66% average utilization); the exact merge/compaction code inside a wave; the radix sort (OneSweep is cited, not described). |
| Fused collapsing (Barbier-Paulin 2025) | Partial (not read in full; the HPG 2025 PDF was blocked for me) | Idea: fuse bottom-up collapse into H-PLOC | Details unverified. Do not rely until read. |
| TRBVH (Karras-Aila 2013) | Full | Algorithm 2 (treelet DP), shared memory plan (768 B/treelet), 3 rounds, n=7 | Warp-size-32 assumption; triangle-splitting heuristics are long. |
| ATRBVH (Domingues-Pedrini 2015) | Partial | Greedy agglomerative treelet rule | I did not read the paper; only abstract-level information. |
| CWBVH layout and traversal (Ylitie 2017) | Full for layout and traversal; Partial for build | Node format (80 B), child ordering, compressed stack, pseudocode, bit tricks | Build is "SBVH then dynamic-programming collapse" and took 6 minutes in their code [P]; faster conversion needs work. Traversal uses CUDA assembly instructions. |
| SBVH (Stich 2009) | Full (not re-read) | Well known | -- |
| Ray sorting (Meister-Bittner JCGT 2022) | Partial | Measured effect | Specific sort keys not copied here. |
| DOBB-BVH (Kern 2025) | Partial (abstract only) | Post-process on wide BVH: +12.6% build time, -18.5% primary / -32.4% secondary traversal time, up to 65% faster intersection [P, arXiv abstract] | Not read beyond abstract. |
| UBVH (Kacerik-Bittner 2025) | Thin (project page only) | SOBB tree where leaves are triangles | -- |
| Hybrid BVH, Disney (SIGGRAPH 2026 talk) | Thin | Policy to reuse/refit/rebuild | No numbers or hardware on the page I read. |
| Tetrahedral cages (Gruen 2026) | Thin | Animate a cage; keep the dense mesh's BVH static | Only secondary summaries read. |

---

# Q1. Main GPU build methods: speed vs trace quality

**H-PLOC paper table (7900 XT, native HIP, includes triangle-pairing and BVH4 conversion; SAH and path-tracing time are relative to the PRBVH best-quality reference) [P]** — https://gpuopen.com/download/publications/HPLOC.pdf

| Scene (tris) | Method | Total build ms | MTris/s | BVH4 SAH | Path-tracing time |
|---|---|---|---|---|---|
| Sponza (0.3M) | ATRBVH | 2.65 | 105 | 1.10 | 1.11 |
| | PLOC++ | 1.33 | 208 | 1.11 | 1.10 |
| | LBVH | 0.66 | 422 | 1.33 | 1.28 |
| | **H-PLOC** | **0.74** | **377** | **1.09** | **1.11** |
| Bistro (3.8M) | ATRBVH | 14.5 | 267 | 1.07 | 1.08 |
| | PLOC++ | 5.61 | 691 | 1.07 | 1.07 |
| | LBVH | 3.82 | 1013 | 1.24 | 1.28 |
| | **H-PLOC** | **4.38** | **882** | **1.07** | **1.07** |
| San Miguel (10M) | PLOC++ | 13.78 | 724 | 1.10 | 1.17 |
| | LBVH | 9.88 | 1009 | 1.35 | 1.37 |
| | **H-PLOC** | **11.45** | **871** | **1.10** | **1.17** |

PRBVH (parallel reinsertion) took 633 ms on Sponza and 12.1 s on Bistro for the 1.00 reference [P]. The paper says H-PLOC is within about 15% of LBVH total build time and 1.1-1.8x faster than PLOC++, 2.7-3.6x faster than ATRBVH [P].

**Other builders, older hardware (published):**

| Method | Published speed | Published quality | Source |
|---|---|---|---|
| LBVH, Karras 2012 | 30-bit Morton codes, GTX 480. I did not extract a clean total-time figure. | LBVH is about 69% of SweepSAH trace speed on average over 20 scenes; only 40-50% on city scenes; over 80% on finely tessellated objects [P] | Karras 2012 https://research.nvidia.com/sites/default/files/pubs/2012-06_Maximizing-Parallelism-in/karras2012hpg_paper.pdf ; quality from Karras-Aila 2013 |
| HLBVH (Garanzha 2011) | Faster and less memory than the 2010 HLBVH; GPU top-level SAH speeds the whole build 5-10x [S, abstract summary] | Karras-Aila measured HLBVH at lower trace speed than LBVH in their runs because of 30-bit codes [P] | https://diglib.eg.org/items/d9e48a42-2dd0-436b-8454-ed7afdf2f5dd |
| Treelet restructuring (Karras-Aila 2013) | "Almost 40M triangles/sec" total, GTX Titan [P]. Dragon 870K: 5.4 ms initial + 17.0 ms optimize + 1.2 ms post [P] | 96% of SweepSAH trace speed on average (min 86%, max 113%) over 20 scenes [P]. Abstract also says >90% of SBVH, while earlier fast GPU builders give about 50% [P] | https://research.nvidia.com/sites/default/files/pubs/2013-07_Fast-Parallel-Construction/karras2013hpg_paper.pdf |
| ATRBVH (2015) | "About 30% faster" than TRBVH, same quality [P, abstract] | Equal to TRBVH [P] | https://diglib.eg.org/handle/10.2312/14668 |
| PLOC (2018) | Power Plant (12.8M): 210 ms vs ATRBVH 309 ms, GTX Titan X [P] | 88 vs 75 MRays/s (+17%); up to 2x faster builds than prior methods [P] | https://dcgi.fel.cvut.cz/wp-content/wpallimport-dist/publications/pdf/publications-2018-meister-tvcg-ploc-paper.pdf |
| PLOC++ (2022) | 212-250 MProm/s on Arc A-series (4.8-10.5M tris); original PLOC about 100 MPrim/s; "1.9-2.3x" faster than PLOC [P]. (The appendix table is garbled in my text extraction; the ranges are from the paper's text and the clearest rows.) | Trace speed 0.89-1.06x and SAH 0.9-1.09x vs a CPU 16-bin binned SAH, Intel hardware ray tracing [P] | https://cdrdv2-public.intel.com/737298/ploc-for-bounding-volume.pdf |
| Binned SAH on GPU (Wald 2007 / Garanzha) | Wald 2007 is a CPU paper. GPU versions exist inside HLBVH top-levels. I found no stand-alone GPU binned-SAH number. | -- | -- |
| Bonsai (Ganestam 2015) | San Miguel in 478 ms, quad-core laptop CPU [P]. GPU port slower [P]. | -- | https://lup.lub.lu.se/search/files/6179951/8593625.pdf ; https://lup.lub.lu.se/student-papers/record/8891127 |

**Build-time ladder (JCGT 2022, normalized to LBVH = 1.0, averaged over 12 scenes, AMD GPU) [P]:** HLBVH 2.3, PLOC 5.6, ATRBVH 6.6, PRBVH 76, SAH 1444, SBVH 5196. Link: https://jcgt.org/published/0011/04/01/paper.pdf. The paper's text says SBVH gives the best trace quality for binary trees; spatial splits give "almost threefold" on Power Plant. Ray reordering pays off on average (7% secondary, 8% shadow).

# Q2. Post-build steps: wide BVH on WebGPU

- **Collapse BVH2 to wide:** done by SAH dynamic programming (Ylitie 2017) or by a greedy "open the largest child" rule (H-PLOC section 3.4, one kernel). In H-PLOC, conversion to BVH4 is 25-46% of total build time (Sponza 25.7%, San Miguel 46.1%) [P]. Fused collapsing (HPG 2025) reports 1.4-1.6x lower build times by merging the collapse into the builder [P, abstract].
- **Does the wide tree trace faster?** On GPUs, tracing is memory-bandwidth limited. JCGT 2022 found trace speed drops with branching factor unless nodes are compressed [P]. The CWBVH answer is compression + octant ordering: 1.9-2.1x [P], on NVIDIA with custom instructions.
- **For WebGPU with no hardware RT** [E]: BVH4/8 reduces node fetches but adds ALU per node (more box tests, popcount, sorting). Our current 64 B node already fetches two boxes per read. I would not move to wide trees before the build is on the GPU. After that, the experiment to run is a BVH4 with quantized boxes, measured on the 3060 and Intel. Outcome unknown; I have no published WebGPU measurement.
- **Other post-steps:** treelet/SAH restructuring after LBVH (TRBVH: 96% of SAH quality) vs PLOC-style clustering; both are in the tables above.

# Q3. Dynamic scenes

**Refit vs rebuild.** Refit keeps the topology and only recomputes boxes bottom-up. It is cheap but quality falls as vertices move away from the build pose [P, Utah rotations paper, https://hwrt.cs.utah.edu/papers/hwrt_rotations.pdf]. Lauterbach et al. (RT-DEFORM, 2006) rebuild when a quality measure crosses a threshold, which beats a fixed frame count [P, as summarised by that paper; I did not open RT-DEFORM]. Two published middle paths:
- apply rotations during refit;
- treelet restructuring on the already-built tree (TRBVH-style).

**Numbers on refit vs build [S, unverified]:** an NVIDIA talk is reported to quote about 1000 MTris/s for refit and about 100 MTris/s for build on RTX, and a case where per-frame refit let a 2000-particle trace slow from 0.5 ms to 8 ms. I could not open the slides; treat as anecdote.

**Recent policy work.** Disney's hybrid BVH (SIGGRAPH 2026 talk) picks reuse, refit or rebuild per part of the scene [P, project page, no numbers]. AMD's HPG 2026 paper (Gruen et al.) animates a tetrahedral cage and keeps the dense mesh's BVH static; it traces 585M animated triangles [P, https://gpuopen.com/learn/ray-tracing-massive-amounts-animated-geometry/]; the "47x less memory, 90x faster update" figures are from a news write-up [S].

**Top-level BVH, costs for 1k-100k instances.**

| Instances | Published | Estimate for our case |
|---|---|---|
| 0.5M | H-PLOC BVH4 build 1.1 ms (7900 XT) [P] | -- |
| 1M | H-PLOC 2.21 ms; LBVH 1.96 ms [P] | -- |
| 100k | -- | about 0.2-0.5 ms on a 3060 Laptop [E: linear from the published rows, x2.5 for lower memory bandwidth] |
| 10k | -- | 0.05-0.15 ms [E: launch-overhead dominated; vs 86-121 ms today on CPU] |
| 1k | -- | under 0.1 ms [E] |

The paper's instance record is 128 B; ours is 64 B, so memory traffic is lower [E].

**Practice.** The usual choices: refit the top level every frame and rebuild when movement is large, or rebuild it fully on the GPU every frame. The published GPU rebuild times above make the second option cheap enough that I recommend it once instances move.

# Q4. What each GPU builder needs from the platform

WebGPU facts used here:
- 32-bit integer atomics on storage and workgroup memory exist. There are no 64-bit atomics and no float atomics in core WGSL.
- Default limits: 256 invocations per workgroup, 16 KiB workgroup memory, 8 storage buffers per stage [spec: https://www.w3.org/TR/webgpu/#limits]. Higher limits can be requested on adapters that have them.
- No forward-progress guarantee between workgroups (so no spin-waiting on another group). Subgroups are an optional feature.
- Gridwise (a WebGPU sort/scan library) uses OneSweep with a lookback-plus-fallback so it works without forward-progress guarantees; subgroup width up to 128 [P, https://gridwise-webgpu.github.io/gridwise/sort]. Limits: up to 2^30 keys, 8 bits per pass.

| Method | Radix sort | Scan | Atomics | Workgroup / wave needs | Needs forward progress? | Safe in WebGPU? |
|---|---|---|---|---|---|---|
| LBVH (Karras 2012) | Yes, 30-60 bit keys | Compaction only | 32-bit counters for bottom-up merge ("first thread exits") | None special | No (wait-free) | **Yes** |
| Apetrei LBVH | Yes | No | Atomic exchange per node | None special | No | **Yes** |
| H-PLOC | Yes, 64-bit keys in paper; 8 B per key | Not between iterations | atomic exchange on parent id; global counter for node ids (wave-shared) | Cluster list = half wave size; data in registers or shared memory | Main kernel: no. Wide-conversion kernel: needs workgroups to start in order. Paper says use an atomic ticket if the platform doesn't guarantee it [P] | **Yes for the build.** Conversion kernel: ticket trick works on all hardware I know of but is not guaranteed by the spec [E]. Can be avoided with a level-by-level conversion (more dispatches). |
| PLOC / PLOC++ | Yes | Global prefix sum (PLOC); PLOC++ reduces to a tiny scan | Some | 1024 invocations, ~32-64 KB shared [P] | No (iterates in separate dispatches) | **Yes, but** exceeds default limits. Can reduce chunk size to 256 and radius to 8 [E]. |
| TRBVH / ATRBVH | Yes | No | Counters per node | One treelet per warp, 768 B shared per treelet, warp ballots [P] | No | Needs subgroup size 32; WebGPU subgroup sizes vary, so rewrite for variable size [E]. |
| Binned SAH on GPU | Often no sort; per-node histograms | Yes (prefix over bins and partitions) | Yes | Large per-node scratch | Depends on scheduler design; level-by-level dispatch needs none | Yes if level-synchronous. Memory-hungry. |
| Refit | No | No | Counter per node ("second arrival continues") [P, Karras 2012] | Parent index array | No | **Yes** |
| Decoupled-lookback scan or OneSweep sort | -- | Yes | Yes | -- | **Yes (spin-waits)** | **Only with a fallback** like Gridwise's [P]. Otherwise use a 3-pass reduce-scan-scatter radix sort [E]. |
| 64-bit Morton codes | Two 32-bit passes (PLOC++ does 2x32 trick [P]) | -- | -- | -- | -- | WGSL has no u64 type. Emulate with `vec2<u32>` [E]. |

Memory for H-PLOC: 8 B Morton + 4 B parent + 4 B cluster index + 2x32 B BVH2 per primitive [P]. For Sponza (262k) that is about 21 MB [E, arithmetic].

# Recent work (2025-2026)

Searched: HPG 2025 and 2026, EGSR, SIGGRAPH, SIGGRAPH Asia, I3D, JCGT and arXiv. I found no 2025-2026 JCGT BVH paper and no EGSR, SIGGRAPH Asia or I3D BVH-construction paper. This does not prove none exists; my search was one to three queries per venue.

| Paper | Venue / year | What it improves | Published numbers | Runs in WebGPU without HW RT? |
|---|---|---|---|---|
| Fused Collapsing for Wide BVH Construction (Barbier, Paulin) https://diglib.eg.org/handle/10.1111/cgf70213 | HPG 2025 (CGF 44(8)) | Builds the wide BVH directly inside the H-PLOC-style bottom-up pass | 1.4-1.6x lower build times; lower frame time when tracing few rays per pixel [P, abstract] | Likely yes if you adopt H-PLOC; only helps if we go wide. |
| DOBB-BVH (Kern et al.) https://arxiv.org/abs/2506.22849 | HPG 2025 / arXiv | Turns wide BVH into oriented-box tree as a post-pass | +12.6% build; -18.5% primary and -32.4% secondary traversal time; up to 65% faster intersection [P, abstract] | Maybe. Cost is more math per node. Untested for us. |
| UBVH (Kacerik, Bittner) https://dcgi.fel.cvut.cz/projects/ubvh/ | HPG 2025 / CGF | One bounding shape (skewed OBB) for nodes and triangles | 1.2-11.8x faster than AABB BVH, incoherent rays [P]; build time not on page | Probably, since it's pure compute traversal; the speedups are against their baselines on GPU. Not verified. |
| Memory-Efficient BVHs with Merged Nodes (Haydel, Kensler, Brunvand, Yuksel) https://graphics.cs.utah.edu/research/projects/bvh-merged-nodes/ | HPG 2026 (best-paper list) | Block-based wide layout, merged internal and leaf nodes, bandwidth-aware SAH | -48.3% BVH size, -15.8% memory traffic, -31.7% render time vs wide-BVH baseline [P] | Layout ideas yes; paper targets hardware RT. |
| Ray Tracing Massive Amounts of Animated Geometry (Gruen, Benthin, Kern, McAllister) https://gpuopen.com/learn/ray-tracing-massive-amounts-animated-geometry/ | HPG 2026 | Animate a tetrahedral cage; keep dense mesh's BVH static | 585M animated triangles [P]; 47x/90x claims [S] | Concept yes (it is a ray transform + BVH). Needs tetra lookup in the walk. Research-grade. |
| Hybrid BVH for Interactive GPU Ray Tracing (Bauer, Li) https://www.yiningkarlli.com/projects/hybridbvh.html | SIGGRAPH 2026 talk | Policy: two-level for instances, three monolithic BVHs by change type; chooses reuse/refit/rebuild | None on the page | The policy idea applies directly to our deforming-mesh plan. |
| Decoupling Data Layouts from BVHs (Scion) https://arxiv.org/abs/2511.15028 | arXiv Nov 2025; PLDI 2026 | Compiler to explore BVH memory layouts separately from traversal | No numbers in abstract; finds a Pareto-optimal ray-tracing layout | Not a builder. Useful as a way to compare node layouts. |
| H-PLOC (reference) https://dl.acm.org/doi/10.1145/3675377 | HPG 2024 | See above | See above | Yes (with the caveats above). |

# Builder interfaces: inputs and outputs

So builders can be swapped later, define one contract.

**Common input (all builders):**
- positions `array<f32>` (xyz), index `array<u32>` (3 per triangle), triangle count;
- optional: per-triangle flags (alpha-test bit), a leaf-size limit (<= 4), SAH constants.

**Common output (matches `trace.md`):**
- nodes: `GiBvhNode` x (N-1) in the 64 B layout;
- triangles in tree order (48 B), including the original triangle index;
- parent index array (4 B per node), needed by refit;
- max depth (for the stack assertion).

**Per-method extra inputs and scratch:**

| Method | Extra inputs | Scratch (per primitive) | Extra outputs / notes |
|---|---|---|---|
| LBVH | Scene bounds | Morton key + index (8 B), parent id, flag | Needs a separate leaf-collapse pass to get <= 4 tris per leaf. |
| H-PLOC | Scene bounds | 8 B key + 4 B parent + 4 B cluster idx + 64 B BVH2 [P] | Binary tree of 32 B nodes; convert to our 64 B two-child node. Optional SAH collapse in the conversion. |
| PLOC++ | Scene bounds, search radius (8-16) | ~72 B [P] | Same as H-PLOC. |
| TRBVH treelet pass | An existing BVH (e.g. LBVH) | Triangle counts per node, per-warp shared arrays | In-place restructure: same node count. Optionally triangle splits (adds references). |
| CWBVH conversion | A finished BVH2 with per-node SAH | DP table per node (Ylitie) | 80 B nodes, reordered triangles; a different trace shader. |
| Refit | Existing tree + parent array + new vertex positions | 4 B counter per node | Updates boxes only. |
| SBVH (CPU bake) | Same as input | Large | Extra triangle references; needs a "references" array, which breaks one-to-one triangle layout. |

Interface suggestion [E]: `build(device, {positions, indices, flags}, options) -> {nodes, tris, parents, maxDepth, stats}` with the builder choosing its own scratch. Make "quality tier" an option, not a separate API.

---

# Recommendation

## Static meshes
1. **Now:** replace the CPU binned SAH with a **GPU LBVH (Apetrei/Karras)** to get the pipeline right: Morton, sort, build, leaf collapse, upload-free. This gets Sponza far under the 2 s target even with slow WebGPU sorting [E]. Quality cost: published trace time 1.28-1.37x LBVH vs best (H-PLOC paper, [P]).
2. **Then:** **H-PLOC** to recover quality: within 1-2% of PLOC++ and ATRBVH, with a single kernel after the sort and no inter-workgroup waits in the main kernel [P]. It is the best published fit for subgroups-first WebGPU because its working set is one wave of clusters.
3. **Keep the CPU binned SAH (or SBVH with spatial splits) as an optional "bake quality" path** run in the worker for assets where the 28 s can be paid once and cached. Do not make users wait for it.
4. **Do not go wide yet.** Moving to CWBVH is the single biggest published trace win (1.9-2.1x) but that was measured with GPU-specific instructions and against an older baseline [P]. It needs its own spike.

## Deforming meshes
- **Refit every frame** using the parent-index scheme already in the design. It needs only 32-bit atomics and is safe without forward-progress.
- **Rebuild with H-PLOC** when a quality signal (SAH cost change, or a simple frame counter as a fallback) crosses a threshold, per Lauterbach. Optional: treelet or rotation pass during refit if drift is high.
- For many deforming meshes, apply a Disney-style policy: classify by change type and choose reuse, refit or rebuild per class.

## Top level
- **Rebuild on the GPU every frame with LBVH or H-PLOC when instance count or motion is large.** Published: 2.21 ms for 1M instances (7900 XT) [P]. Estimate for 10k on a 3060 Laptop: well under 0.2 ms [E], against 86-121 ms on the CPU today.
- Keep the CPU build for a few instances that rarely move.

## Trade-offs
| Choice | Gain | Cost |
|---|---|---|
| LBVH first | Simple, very robust | Trace quality loss (published ~1.3x) |
| H-PLOC next | Near-SAH quality at near-LBVH speed | Wave-size tuning; 64-bit keys emulated; conversion kernel ordering |
| Subgroups-first | H-PLOC cluster lists in registers; ballot-based refit/treelet kernels | Fallback path for no-subgroup devices must exist |
| No 64-bit atomics | Not a blocker | Use 32-bit atomics and two-word keys |
| No forward progress | Not a blocker if we avoid lookback and spin-waits | Use 3-pass radix sort or Gridwise's fallback sort |
| Stay BVH2 | No trace-shader change | Misses CWBVH-style gains |

## Open items to check next
- Read the Fused Collapsing, DOBB and UBVH PDFs in full (blocked or abstract-only for me).
- Measure on the actual 3060 Laptop: WebGPU radix sort throughput (no published Gridwise numbers were on the page I read).
- Measure SAH cost of our current CPU tree vs H-PLOC to see whether quality is lost.
- Verify the NVIDIA refit-vs-build numbers [S] before quoting them.
