# M3 design draft: moving objects and the camera

Status: **draft for Dennis's review, 2026-11-05.** Nothing here is decided. It is the input for M3's breakdown (the last task of M2), and, for the startup items DECISIONS moved to M2, for T1.11.

It builds on [ddgi.md](ddgi.md), [bvh-framework.md](bvh-framework.md), [platform.md](../api/platform.md), [three.md](../api/three.md), the B1 hand-over (branch `bvh-b1`, `packages/bvh/README.md`) and the M2 draft, [m2-lights.md](m2-lights.md). M3 reuses M2's per-probe history and GPU scheduler (m2-lights.md section 4) for moved geometry; if M2 changes them, this page follows.

## In one paragraph

Moving geometry already works at the tracing level: the top-level tree is refit on the GPU every frame and rebuilt in a worker (B1a), and nothing recompiles when instances come and go. M3 makes the rest of the chain fast and complete. The three adapter finds changes from three's own events and version counters instead of a slow walk, and sends many transforms in one call. New and edited meshes get GPU-built trees (B1's LBVH, later B2's H-PLOC) in the frame they arrive, and tinybvh's better trees replace them quietly when the worker finishes; the same machinery gives first GI in about a second after a scene is handed over. Deforming meshes are refit on the GPU from positions the adapter supplies. Probes near a moved object get their history shortened and are put at the front of the update list, exactly as for a light change, with the size of the change estimated from how much of the probe's view the object covers; probes that were switched off come back warmed, not black. A camera-following volume scrolls its probe grid toroidally, RTXGI-style, so only the planes entering at the far edge start over, and several volumes share one atlas and blend at their edges.

## What M3 must deliver (SPEC)

| Target | Tier A (RTX 3060, 1440p) |
| --- | --- |
| Object move response (cabinet move, door closing) | within 5% of settled in ≤ 1 s |
| Closed door | no leak: dark-room light ≤ 1% of the lit room's, and none visible |
| Recompiles | the counter stays at 0 through every scenario |
| Adapter cost | ≤ 0.2 ms per frame still, ≤ 0.5 ms with one object moving |
| Camera-following volume | scrolling never shows as a seam or a pop |
| Deforming meshes | cast moving occlusion into the GI |

Plus M1's still-scene checks, and M2's light checks, unchanged.

## What others do, and what we copy

| Who | How moved geometry reaches the GI | What we take |
| --- | --- | --- |
| **RTXGI-DDGI** (Majercik et al. 2021; SDK docs `DDGIVolume.md`) | Rays see the moved geometry immediately (hardware TLAS/BLAS rebuilt or refit by the app). Probes don't know anything moved: the 0.97 hysteresis plus the per-texel change test catches up. Classification and relocation re-run from the fixed rays each update. **Infinite scrolling volumes** keep probes fixed in world space and, when the active area moves, invalidate only the plane of probes that wrapped to the far edge (`DDGIClearScrolledPlane` in `ProbeBlendingCS.hlsl`). | Fixed rays re-checking state every update (we already do this, T1.5b). **Toroidal scrolling with plane resets.** Volume blending by priority. |
| **Lumen** (Wright et al., SIGGRAPH 2022 Advances) | Rigid instances carry precomputed mesh distance fields and surface-cache cards that move with the instance; caches near changes are re-lit within a budget, by priority. Deforming meshes need hardware ray tracing. | Rigid trees that move with their instance (our two-level BVH). **Budgeted, prioritized refresh near changes** (M2's scheduler). |
| **Godot SDFGI** ([docs](https://docs.godotengine.org/en/stable/tutorials/3d/global_illumination/using_sdfgi.html)) | Camera-following cascades; dynamic objects don't occlude or bounce. | Confirms camera-following is the norm; we go further on movers. |
| **EA SEED GIBS** (Halén et al., SIGGRAPH 2021) and **GIBS2** (Apers, SIGGRAPH 2024 Advances) | Surfels are attached to the object (and its skin), so they move with it; probes with DDGI-style octahedral data cover the rest. | The idea that movers must be *found*, not waited out. We don't take surfels in v1. |
| **AMD Brixelizer** ([GPUOpen](https://gpuopen.com/fidelityfx-brixelizer/)) | Sparse distance-field bricks around the camera, rebuilt in a budget for dynamic instances. | Camera-centred, budgeted rebuilds. |
| **Enlighten** (Martin and Einarsson, SIGGRAPH 2010 Advances) | Geometry is static; dynamic objects only receive light from probes. | What we must beat: SPEC's door and cabinet cases need movers to occlude. |
| **Clipmaps** (Tanner, Migdal, Jones, SIGGRAPH 1998) | A fixed-size window over a virtual texture, addressed modulo its size, updated only at the edge the window moves toward. | The addressing for the scrolling grid. |
| **BVH maintenance** (Wald, Boulos, Shirley, "Ray Tracing Deformable Scenes Using Dynamic Bounding Volume Hierarchies", TOG 2007; LBVH: Karras 2012, Apetrei 2014; H-PLOC: Benthin et al., HPG 2024) | Refit when the shape stays similar, rebuild when quality drifts; GPU builders make rebuilds cheap. | Refit every frame + periodic GPU rebuild for deforming meshes; GPU LBVH now, H-PLOC later. |

## 1. The change feed (platform side)

What already exists: `setTransform`, `createInstance`/`destroyInstance`, `createMesh`/`updateMesh`/`destroyMesh`, `setDeformedPositions` (throws "arrives in M3"). The change log writes two regions per moved instance (old and new box) with the instance as `source`. Fixed-capacity tables grow by reallocating; no recompiles.

M3 adds:

- **`setTransforms(instances: Uint32Array, matrices: Float32Array)`** (host-facing addition; question 2). B1a measured 10,000 `setTransform` calls at 18–20 ms, about 2 µs each, all in per-call bookkeeping. A batch call writes straight into the instance mirror and logs one merged region list. Target: 10,000 transforms in ≤ 2 ms.
- **`SceneChanges.receivers: readonly Aabb[]`** in place of the `movedReceivers` count (technique-facing). Moving objects that don't contribute to GI (characters, small props) still need probes near them to be awake (section 5.3).
- **`setDeformedPositions` in two forms:** a same-device `GPUBuffer` (as the contract says now) or a CPU `Float32Array`, which the platform uploads into its own buffer. *Why the CPU form:* three r187 gives us no way to get a GPU buffer it owns without reaching into its backend (section 7).
- **A recompile counter** in `gi.stats`: compute pipelines created after `isReady`. The viewer shows it next to three's own program count.

## 2. Per-mesh trees: GPU first, tinybvh as the refinement

### 2.1 What exists

- **B0 (tinybvh, WASM worker):** best trees (lowest SAH cost); Sponza in 3.9 s in the worker.
- **B1 (GPU LBVH, branch `bvh-b1`, being merged into `m1`):** Sponza's 5.74 M triangles in 31.2 ms on the RTX; trees cost 1.22× tinybvh's SAH and trace at 0.82× (closest) and 0.80× (any). Builds every mesh of a batch in shared dispatches into three shared buffers with a node offset per mesh, in exactly the tracer's layout. Needs subgroups; `createMeshLayoutBuilder` falls back to tinybvh without them. **Not yet wired into `ddgi-core`**: the platform still builds with tinybvh only, and B1's output is read back today.
- **B1 BLAS refit (`GpuMeshRefit`):** rewrites the tree-ordered triangles from a vertex buffer, then B1a's refit climbs. Works on any builder's tree.
- **B2 (H-PLOC):** target trees within 1.15× of tinybvh's SAH and trace ≥ 0.9× B0's; required before M3 sign-off (bvh-framework.md).

### 2.2 The plan

- **Every new or replaced mesh is built on the GPU in the frame it arrives,** straight into the tracer's node and triangle buffers. Each mesh gets a range from a range allocator (the scene copy already has one); the mesh table holds its root offset. No readback.
- **tinybvh still builds every mesh in its worker** (a copy of positions and indices goes there, as now). When a mesh's tinybvh tree arrives, it is uploaded into a new range, the mesh table's root offset flips, and the old range is freed a frame later (after the GPU has finished with it). The tracer sees a different tree over the same triangles: watertight hits are the same up to float rounding, so the field doesn't notice. Both trees exist for a moment: worst case about 2× one mesh's tree memory.
- **When B2 lands** and its trees trace at ≥ 0.95× tinybvh's, the swap becomes optional (a setting, `meshTreeRefinement: 'tinybvh' | 'none'`), and saves the worker's copy and time.
- **Without subgroups:** tinybvh only, as in M1 (the declared fallback). First GI waits for the worker on those devices.
- **The top level:** B1a's decision stands: refit every frame on the GPU, rebuild the shape in a worker on instance creation or quality drift (new instances appear in about 55 ms at 10,000 instances, sooner for small scenes). If the "add a cabinet" scenario shows that delay, the GPU LBVH can build the top level over instance boxes in the same frame; it's the same code. Not planned by default.

### 2.3 First GI from GPU trees (scheduled in M2)

DECISIONS 2026-11-05 (provisional): first GI from B1's GPU trees, swapping to tinybvh's when the worker finishes, plus the scene copy off the main thread and compiling less. These are rows T2.12–T2.14 below, run in M2.

Sponza from load today (T1.9a): files 4.2 s (the host's), mesh trees 3.3 s, main-thread scene copy 1.2 s, cold compile 1.2 s = 10.8 s (Bistro exterior's scene copy alone 4.1 s).

| Part | Today | With T2.12–T2.14 (estimate) | How |
| --- | --- | --- | --- |
| Files | 4.2 s | 4.2 s | The host's; out of our hands |
| Mesh trees | 3.3 s | ~0.3 s | GPU LBVH (31 ms GPU; 302 ms wall in B1's first build with uploads and a stats readback, which goes) |
| Scene copy | 1.2 s main thread | ~0.15 s, time-sliced | Raw upload + GPU packing (section 2.4) |
| Compile | 1.2 s | ~0.6 s | Only what's used; the sky kernel fix; compile while trees build |
| **Total from load** | **10.8 s** | **~5.3 s** | |
| **From "scene handed to GI"** | **~6.6 s** | **~1.0 s** | |

So Sponza can meet 2 s if the clock starts when the host hands the scene over, not if it starts at the first file request. That's still the open question for Dennis at M1 sign-off (DECISIONS), repeated as question 5.

### 2.4 The scene copy off the main thread

The 1.2 s (Sponza) and 4.1 s (Bistro exterior) are JS loops on the main thread: octahedral normals, 16-bit indices, interleaving, per mesh. A worker would still need the arrays copied in and the results copied back, then a `writeBuffer` from the main thread (the device lives there; the host owns it). The GPU can do the packing instead:

- **Upload raw arrays** (positions, normals, uvs, indices exactly as the host gave them) with `writeBuffer` into a staging region, a few meshes per frame, at most `uploadMsPerFrame` (default 4 ms) of main-thread time.
- **`gi/pack-mesh`** (compute, one thread per vertex, then per triangle): octahedral normals, index narrowing to 16 bits where every index fits, the same layout `mesh-packing.ts` writes today.
- **The CPU packer stays as the test oracle,** and a test holds the GPU output to it byte for byte on every procedural scene and on Sponza.
- *Floor:* the raw copy itself. NOTES: reading 300 MB in JS takes about 110 ms on this laptop, so Sponza's ~180 MB of raw arrays costs ~60–100 ms of main-thread copying however it's done.

The tinybvh worker keeps receiving its own copy, as now; that copy is part of the floor too.

## 3. Deforming meshes (BLAS refit)

- **Input:** `setDeformedPositions(mesh, positions)` each frame, GPU buffer or CPU array (section 1). A mesh created with `deforming: true` keeps its triangle list.
- **Per frame:** `GpuMeshRefit` (B1) rewrites that mesh's tree-ordered triangle records from the positions and refits its boxes bottom-up; then the top-level refit, which already runs every frame, picks up the new mesh box. All on the GPU, no readback.
- **Quality:** refitted trees degrade as the shape moves away from the one they were built for. The platform promises no readbacks, so the GPU's SAH numbers can't steer a rebuild from the CPU. Instead, each deforming mesh is **rebuilt with the GPU LBVH every `deformRebuildFrames` frames** (default 32, staggered across meshes so they don't all rebuild in one frame). A sway rarely strays far from its rest pose; the T3.0 spike measures how fast trace speed decays on the wind tree to set the number.
- **Normals:** the scene copy's octahedral normals are the rest pose's. Deforming meshes shade with the triangle's geometric normal (from the refreshed triangle records). For foliage at probe resolution that's invisible; an optional normals buffer can come later.
- **Cost estimate:** B1a's top-level refit is 0.086 ms for 10,000 instances; a mesh refit touches triangles as well, so a 100,000-triangle tree should be about 0.1–0.2 ms on the RTX, plus an LBVH rebuild of ~0.5 ms every 32 frames. Planning number: ≤ 0.3 ms per frame for the wind tree.
- **Probes:** a deforming mesh logs its world box as a geometry region every frame, so nearby probes get the M2 treatment with a small change each frame (section 5), which keeps their history matched to the motion.

## 4. Several volumes, and the camera-following volume

### 4.1 Several volumes in one technique

- **Settings:** `volumes: VolumeSettings[]` (at most 8), each `{ center, size, spacing, follow: 'fixed' | 'camera', priority }`. M1's single `volume` becomes the one-element case.
- **One atlas set for all volumes:** the irradiance, distance and probe-data textures stay three 2D arrays. Volume `v` owns layers `[base_v, base_v + ny_v)` and tiles `(x, z)` inside each. Layer size is the largest volume's footprint (`max nx × 8` by `max nz × 8` for irradiance). *Why shared:* the shading module may use at most 6 textures (platform.md gap 1); one set per volume would break that at the second volume.
- **The volume table** is the small `rgba32float` texture the contract already plans: 4 texels per volume (origin and spacing; counts and layer base; scroll offset and window; priority, fade width, flags).
- **Probe numbering:** one global index across volumes (`volumeBase + local`), so the scheduler, history and update list don't change.
- **Shading:** for a point, volumes are visited in priority order (the CPU sorts the table). Each contributes with weight `w_v = coverage_v × (1 − W)`, where coverage is the existing edge fade times `saturate(Σw / 0.02)` (DECISIONS 2026-11-05), and `W` is the weight so far. Stop at `W ≥ 0.99` or after 2 volumes. Inside a volume, one sample as now; only in fade bands, two. RTXGI blends volumes the same way.
- **Bounce light** in `ddgi/trace` uses the same multi-volume function, so light bounces between volumes.

### 4.2 The camera-following volume: toroidal scrolling

The volume is a fixed-size window of probes over a lattice fixed in world space (RTXGI's infinite scrolling, clipmap addressing):

- **Lattice and slots:** world lattice coordinates `L = round((P − anchor) / spacing)`; a probe's storage slot is `L mod n` per axis. Probes never move in world space, so their irradiance and distance stay valid.
- **The window** is `[windowMin, windowMin + n)` in lattice units, centred on the camera. It moves when the camera is more than **0.75 spacing** from the window's centre plane on an axis, by whole planes. *Why 0.75, not 0.5:* a camera resting on a plane boundary would otherwise make the window flip back and forth, clearing the same planes every frame.
- **On a move:** the slots of the planes that leave are now the planes that enter at the far edge. A small kernel `ddgi/scroll` resets them: state 3 (not updated yet), relocation offset 0, history `n = 0`, `owed = K`, high urgency. M2's scheduler then updates them first.
- **Sampling** maps a point to lattice coordinates, then slots with the same modulo, and the window bounds come from the volume table. For a fixed volume the offset is 0 and the modulo does nothing: the same code, not a second path. Cost: one integer modulo per axis per probe sample.
- **No seam, no pop:** entering planes are at the far edge, half a volume from the camera, inside the coverage fade band, and they aren't used in shading until warmed (section 5.2). The fixed volumes and then the host's environment cover the band in the meantime.
- **Speed:** at 10 m/s with 1.5 m spacing, about 6.7 planes per second per axis; with a 32 × 12 × 32 window that's about 2,600 probes per second, 43 per frame at 60 Hz. Each needs its K warm-up updates: about 340 probe updates per frame, competing with everything else. The fly-through test measures whether 10 m/s is the right speed to promise.
- **Teleports** (the camera jumps more than half the window): every slot resets, and the viewer's warm-up mode (DECISIONS 2026-11-05) runs, as after a scene load.
- **Memory:** 32 × 12 × 32 = 12,288 probes, about 1.5 KB each (irradiance 512 B, distance 1 KB, data, history): **19 MB**, covering 48 × 18 × 48 m at 1.5 m spacing.

## 5. Probes near moved geometry

### 5.1 How a moved object becomes a set of probes

M2's `ddgi/mark` (m2-lights.md 4.2) gets a geometry entry. For each geometry region (the old box and the new box of a moved, added, removed or deforming object):

- **Near set:** probes within the box grown by 1.5 spacings. `r = 1` (a full change): their rays may now start inside the object, or see a different surface right next to them.
- **Far set:** probes within 8 spacings (clipped to the volume) that can see the box: one Chebyshev test from the probe's distance moments toward the box centre, kept if ≥ 0.25. Their change is the share of the view the object covers:
  `r = min(1, Ω / Ω_ref)`, with `Ω ≈ π × boxRadius² / distance²` and `Ω_ref = 0.5 sr` (an object filling a twelfth of the sphere counts as a full change).
- Then the same history rule as for lights: `n ← min(n, τK / (r − τ))`, `owed ← K`, urgency from `r` and the camera weight.

Example, the door between two rooms (1 × 2 m, radius about 1.1 m): a probe 4 m into the dark room that looked through the doorway has `Ω ≈ 0.24 sr`, `r ≈ 0.47`, so its history drops to `n ≈ 0.95` and it is owed 8 updates. Behind a wall (Chebyshev < 0.25) it isn't marked.

**What the budget buys** (the same arithmetic as M2): the door's near and far sets are a few hundred probes on two-room, which all get their 8 updates in well under 1 s. A cabinet moved in the furnished room touches fewer. Sponza-sized changes are served nearest the camera first.

**The distance atlas** uses the same per-probe history. For geometry changes that matters most: until a probe near the door learns the door's distance, light can leak through it for those few frames. Section 5.2's first-update rule keeps that short.

### 5.2 Probes that were off come back warmed, not black

Off probes (inside geometry, or no surface nearby) already trace their 32 fixed rays every time they're updated, and classification can turn them back on (T1.5b). Today a probe switching on is used in shading straight away, with whatever its atlas tiles last held. M3 changes the transition:

- **Off → on** sets `n = 0` and state 3; state 3 now ends after **2** blended updates, not 1, so a probe is only used once it has at least two estimates. Both off states are in the near set of any change that can free them, so they're scheduled at once.
- **On → off** (a cabinet pushed over a probe): immediate, as now; neighbours fill in through the normal sampling weights.

### 5.3 Moving receivers in empty space

TASKS carries this from M1: classification turns off probes in empty space (347 of Cornell's 1,728), so an object moving through mid-air gets no GI. RTXGI has the same limit. Proposed:

- **Awake air probes:** a probe off for "no surface nearby" that is within 2 spacings of a receiver region (`SceneChanges.receivers`) or a geometry region is **woken** for 300 frames (5 s) after the last touch: it traces all rays, blends, and is used in shading (after warming). Then it goes back to sleep.
- *Why not keep all air probes awake at a low rate:* it would change M1's still-scene cost and results; waking keeps a still scene exactly as M1 has it.
- Data: a `wakeUntil` frame in the history record's spare word.

## 6. Efficient change tracking in the three adapter

**Today (T1.7, three.md):** a walk of 64 objects per frame plus a hot list for movers, `markChanged(object)` for no delay, 8 edit checks per frame, lights every frame. Still frames cost 0.06–0.10 ms on Sponza and 0.11–0.15 ms on Bistro interior, but an unmarked first move on Bistro interior takes up to 41 frames (0.7 s), and an unmarked geometry edit up to about 260.

**M3, using what three already tells us:**

| Change | How it's found | Latency |
| --- | --- | --- |
| Added or removed objects | three's `childadded` / `childremoved` events on every tracked parent (to confirm in r187, T3.0) | same frame |
| Moved objects | every frame, compare each tracked object's `matrixWorld.elements` with a `Float32Array` snapshot (16 floats per object) in one tight loop | next frame |
| Static hint | objects with `matrixAutoUpdate` and `matrixWorldAutoUpdate` both false are skipped by the loop and left to the slow walk or `markChanged` (three's own idiom for static objects) | walk |
| `InstancedMesh` | `instanceMatrix.version`; only the ranges in `instanceMatrix.updateRanges` when present | next frame |
| Geometry edits | each position and index attribute's `version`, every frame (one integer compare each) | next frame |
| Material edits | `material.version` | next frame |
| `visible` | own flag every frame; a change re-checks the subtree | next frame |
| Anything else | the existing walk, kept as a safety net at a lower rate | `ceil(objects / objectsPerFrame)` frames |

- **Sending:** moved objects go out in one `setTransforms` batch per frame.
- **Cost estimate:** the compare loop should be tens of nanoseconds per object, so ~0.05–0.1 ms for Bistro interior's 2,622 objects. If T3.0's micro-benchmark says otherwise, the walk-plus-hot-list stays and only its budget rises.
- **Deforming meshes:** a mesh with `userData.giDeforming = true` (or a `SkinnedMesh` with `giContributes = true`) is created `deforming`. When its position attribute's `version` changes, the adapter sends the CPU array (section 7).
- Moves still reach GI one frame late by default (DECISIONS 2026-11-05: the adapter reads the matrices three's last render made).

## 7. Getting deformed positions out of three

The hard part of the wind tree isn't the refit; it's that three r187 gives us no supported way to read a GPU buffer it owns (NOTES: no external buffers; and we don't reach into `renderer.backend`). A wind sway written as a TSL `positionNode` exists only inside three's vertex shader.

Three routes, in the order I'd try them:

1. **CPU positions** (works today): the host animates the position attribute on the CPU (three uploads it too), the adapter passes the same `Float32Array` to `setDeformedPositions`, and the platform uploads it. Cost: one `writeBuffer` per frame, about 1.2 MB for a 100,000-vertex tree. This is the plan for the viewer's tree.
2. **A TSL compute writing into our texture** (T3.0 spike): if three can bind an `ExternalTexture` as a *storage* texture in a TSL compute, the host's GPU wind writes positions into a texture we own, three's vertex shader reads them back with `positionNode`, and our refit reads the same texture. No backend access, no upstream change. Unknown whether r187 allows it.
3. **Upstream `ExternalBufferAttribute`** (the ~20-line PR SPEC already lists): three would read our buffer as a vertex attribute. Cleanest, but it depends on review upstream (question 1).

Skinned characters as occluders stay a stretch goal (SPEC). If wanted, a platform skinning kernel (linear blend skinning from bone matrices, engine-neutral) is the route; not in M3's plan.

## 8. The viewer scenarios

| Scenario | Scene | What it shows | Measured |
| --- | --- | --- | --- |
| **Drag furniture** | furnished room | `TransformControls` on any object; shadows and colour bleeding follow | time to within 5% of settled after a 1 m cabinet move (response meter, M2) |
| **Animated door** | closed-door / two-room with a hinged door | a keyframed door swing closing on the lit side | dark-room mean at `dark-room-leak-view` over time; leak ≤ 1% after settling, none visible during |
| **Add and duplicate cabinets** | cabinet row | "add" makes a new mesh (new tree, built on the GPU that frame); "duplicate" shares the mesh (new instance only); up to 100 | frames until GI sees it; recompile counter |
| **Wind tree** | courtyard (small outdoor scene) | a CPU wind sway on the tree, casting moving occlusion | refit ms, trace speed over 60 s |
| **Fly-through** | Bistro exterior | a manifest camera path larger than one volume, with the camera-following volume | per-frame image change at the volume's edge band; probes per frame; memory |

- Every scenario is a manifest motion (m2-lights.md section 7), so it replays the same way in tests and next to native where native can follow.
- The response meter and the recompile counter are on screen during each.
- **Alpha cutouts are M4.** Until they land, the tree's leaf cards block light as solid quads and over-darken. The M3 check is that occlusion *moves*; the look is judged after M4 (question 6).
- **Native:** the harness has no moving geometry. Side-by-side for M3 is limited to the door and cabinet scenes if T2.7's motion support can be extended to instance transforms; otherwise M3 compares against references (Cycles renders of the door open and closed) only.

## How it fits the existing kernels (no parallel GPU paths)

| Piece | M1 / M2 | M3 |
| --- | --- | --- |
| Mesh trees | tinybvh in the worker | GPU LBVH first (B2 later), tinybvh swapped in; tinybvh only without subgroups (the declared fallback) |
| Deforming meshes | refused | `GpuMeshRefit` every frame + periodic GPU LBVH rebuild |
| Top level | GPU refit + worker rebuild (B1a) | unchanged |
| Scene copy packing | CPU, main thread | `gi/pack-mesh` on the GPU; the CPU packer becomes the test oracle only |
| `ddgi/mark` | light entries (M2) | + geometry and receiver entries |
| Probe states | off probes used again straight away | warmed for 2 updates; air probes woken near movers |
| Sampling | one volume | several volumes in priority order, toroidal slots; one volume and no scroll is the same code |
| `ddgi/scroll` | none | new, tiny, only on window moves |

Storage buffers stay as in M2 (`ddgi/trace` 7). `gi/pack-mesh` binds the staging buffer and the scene copy's vertex buffer (2). `ddgi/scroll` binds the history (1) and writes the probe-data texture.

## Cost against the budgets (estimates, RTX 3060, to be measured)

| Item | When | Cost |
| --- | --- | --- |
| Top-level refit | always | 0.086 ms at 10k instances (B1a, measured) |
| Mesh refit, wind tree (~100k triangles) | every frame | ~0.1–0.2 ms + LBVH rebuild ~0.5 ms every 32 frames |
| New mesh tree (a cabinet, a few thousand triangles) | on add | well under 1 ms GPU, the same frame |
| `ddgi/mark` geometry entries | frames with moves | < 0.02 ms for a few objects |
| `ddgi/scroll` | window moves | < 0.01 ms |
| Sampling with two volumes | fade bands only | ≤ 2× there; whole frame ≤ 1.15× (target) |
| Adapter, still | every frame | ≤ 0.2 ms (target) |
| Adapter, one mover | every frame | ≤ 0.5 ms (target) |
| GPU mesh packing | at load | ~10–20 ms GPU for Sponza, once |

Memory: camera-following volume 19 MB at the proposed size; the swap holds two trees of one mesh for a frame; deforming meshes keep a vertex buffer the size of their positions when the CPU route is used.

## Proposed task breakdown

### Startup lane, run in M2 (DECISIONS 2026-11-05, provisional)

| ID | Task | Agent | Size | Depends on |
| --- | --- | --- | --- | --- |
| T2.12 | Fast first GI: GPU LBVH trees in the tracer without readback; tinybvh swapped in per mesh | implementer | M | B1 merged into `m1` |
| T2.13 | Scene copy off the main thread: raw upload, time-sliced, GPU packing | implementer | M | none |
| T2.14 | Compile only what's used: lazy `encodeBatch` kernels, compile while trees build | implementer | S | T2.6 (owns the sky kernel's compile) |

- **T2.12.** Done when: Sponza from "scene handed to GI" to first GI frame ≤ 1.0 s on the RTX (cold compile included; median of 3); at the swap, the field's mean changes by ≤ 0.5% and trace speed afterwards is within 5% of M1's on tinybvh trees; the oracle passes on trees in mid-swap (some meshes LBVH, some tinybvh) on both GPUs; a device without subgroups still gets tinybvh trees and GI (test).
- **T2.13.** Done when: main-thread time for the scene copy is ≤ 150 ms total on Sponza and ≤ 200 ms on Bistro exterior (from 1.2 s and 4.1 s), with no single frame's share over `uploadMsPerFrame` + 1 ms; the GPU-packed buffers equal the CPU packer's byte for byte on every procedural scene and Sponza.
- **T2.14.** Done when: Cornell from load to first GI ≤ 1.5 s cold on the RTX (from 2.34 s); no `encodeBatch` pipeline is created in a viewer session that doesn't use it (counter).

### M3

| ID | Task | Agent | Size | Depends on |
| --- | --- | --- | --- | --- |
| T3.0 | Spike: deformed positions from three (storage `ExternalTexture`), `childadded` events, transform-compare cost on Bistro interior, refit quality decay on a swaying tree | implementer | S | M2 done |
| T3.1 | Change feed: `setTransforms`, `receivers`, `setDeformedPositions` (both forms), recompile counter | implementer | M | none |
| T3.2 | Runtime mesh trees: GPU build on add and update, tinybvh refinement, range allocation | implementer | M | T2.12 |
| T3.3 | Deforming meshes: `GpuMeshRefit` wired, periodic rebuild, geometric normals | implementer | M | T3.1, T3.2 |
| T3.4 | Probes near moved geometry: geometry `ddgi/mark` entries, warm-up rule, awake air probes | implementer | M | T3.1, M2's T2.4b |
| T3.5 | Several volumes: shared atlas, volume table, priority blending | implementer | M | none |
| T3.6 | Camera-following volume: toroidal slots, `ddgi/scroll`, window rule | implementer | M | T3.5 |
| T3.7 | three adapter change tracking v2 | implementer | M | T3.0, T3.1 |
| T3.8 | Viewer scenarios and their manifest motions (furniture, door, cabinets, wind tree, fly-through) | implementer | M | T3.3, T3.4, T3.6, T3.7 |
| T3.9 | M3 measurements | gpu-tester | M | all above, B2 |
| T3.10 | M3 note and M4/M5 breakdown | main session | S | T3.9 |

B2 (H-PLOC) stays in the BVH lane and gates M3 sign-off, as bvh-framework.md says.

### Draft done-whens

- **T3.0 Spike.** Written yes/no answers with numbers: can a TSL compute write an `ExternalTexture` (r187) and can our refit read it; does r187 fire `childadded`/`childremoved`; ns per object for the transform compare on Bistro interior; trace speed after 600 frames of refit-only sway, against a fresh build.
- **T3.1 Change feed.** Done when: 10,000 transforms through `setTransforms` cost ≤ 2 ms CPU (from 18–20 ms); a moved receiver produces a `receivers` box; both `setDeformedPositions` forms pass the oracle on a deforming test mesh; creating and destroying 1,000 instances and 100 meshes in random order leaves the recompile counter at 0.
- **T3.2 Runtime mesh trees.** Done when: a mesh added at frame f is hit by rays at frame f + 1 (oracle page); after the tinybvh swap, trace speed equals tinybvh-only within 5%; memory returns to one tree per mesh within 2 frames of the swap; without subgroups, meshes get tinybvh trees (test).
- **T3.3 Deforming meshes.** Done when: the oracle passes on a deforming mesh over 60 frames on both GPUs; refit plus amortized rebuild cost ≤ 0.3 ms per frame on the RTX for a 100,000-triangle mesh; trace speed on it stays within 15% of a fresh build over 60 s.
- **T3.4 Probes near moved geometry.** Done when: on the furnished room, a 1 m cabinet move is within 5% of settled in ≤ 1 s (RTX, response meter, default budget); on two-room with a door, closing it brings the dark room's indirect light to within 5% of the closed-door reference in ≤ 1 s and to ≤ 1% leak after settling; no probe is used in shading before 2 blended updates (test); a still scene shows the same field and cost as M2 within noise (air probes stay asleep).
- **T3.5 Several volumes.** Done when: a fixed dense volume inside a coarse one shows no seam (image difference across the fade band ≤ the M1 stability threshold); sampling cost at 1440p on a view half in the fade band ≤ 1.15× one volume; one-volume scenes match M2 exactly (same code with one entry).
- **T3.6 Camera-following volume.** Done when: on Bistro exterior's fly-through at 10 m/s, the frame-to-frame change in the edge band is within the M1 stability threshold, probes entering are never used before warming (test), memory stays fixed, and a camera resting on a plane boundary for 10 s clears 0 planes.
- **T3.7 Adapter change tracking v2.** Done when: on Bistro interior an unmarked first move reaches GI in ≤ 2 frames (from 41), an unmarked geometry edit in ≤ 2 (from ~260), still-frame CPU ≤ 0.2 ms and one mover ≤ 0.5 ms (median of 300 frames, RTX laptop); `InstancedMesh` update ranges send only those instances (test).
- **T3.8 Viewer scenarios.** Done when every scenario plays from its manifest motion with hover help, shows the response meter and the recompile counter, and Dennis can run each without instructions.
- **T3.9 M3 measurements.** The SPEC table above on both GPUs, plus the door and cabinet response curves against Cycles references of the end states (3 runs). Recompile counter 0 throughout.

## Risks, and what to spike first

1. **Deformed positions from three** (T3.0). If neither a storage `ExternalTexture` nor the upstream PR works, GPU-animated meshes can't contribute without reaching into three's backend; only CPU-animated ones can. *Spike first.*
2. **Change tracking at scale.** If the per-frame transform compare costs more than estimated, Bistro-sized scenes keep a latency of several frames for unmarked moves. Measured in T3.0.
3. **The door's leak window.** Until the near probes' distance tiles learn the closed door, light leaks for a few frames. With `n` near 0.4 and the door's probes updated every frame, that should be 2–4 frames; if it's visible, those probes' first update could take 2× rays.
4. **The far-set estimate** (`Ω_ref`, the 8-spacing radius) is a guess; T3.4 measures marked probes against probes that actually changed by more than 5%.
5. **Scrolling speed.** Fast flights at fine spacing could need more updates than the budget has; the edge band then shows the environment longer. The fly-through sets the promise.
6. **Shared atlas waste.** Volumes of very different footprints waste layer area; at 8 volumes the atlas could hit `maxTextureArrayLayers` (256). The design refuses past that with `GILimitError`.
7. **Refit quality** on large deformations (cloth, a fan's blades) may need faster rebuilds than every 32 frames.

## Questions for Dennis

1. **Deforming meshes from three:** is CPU-animated positions enough for the viewer's wind tree in M3? And should we open the small `ExternalBufferAttribute` PR upstream now, so GPU-animated meshes (TSL wind, morph targets) can contribute later?
2. **`setTransforms`, a batch call** on the host-facing scene API (one array of handles, one of matrices). It's an addition, nothing changes for existing callers. OK to add?
3. **Air probes:** wake "no surface nearby" probes only near moving things (still scenes stay exactly as M1), rather than keeping them all updated slowly. OK?
4. **The fly-through:** Bistro exterior, with a camera-following volume of 32 × 12 × 32 probes at 1.5 m (19 MB), at up to 10 m/s?
5. **Startup:** does the 2 s target start when the host hands the scene to GI (Sponza ~1.0 s projected) or at the first file request (~5.3 s, files being 4.2 s of it)? (Already open from M1.)
6. **The wind tree:** which asset (the courtyard tree, or one of `sponza-full`'s 2 M-triangle trees), and is it fine that leaves block light as solid cards until M4's alpha cutouts?
7. **Skinned characters as occluders** stay a stretch goal after v1?
</content>
</invoke>
