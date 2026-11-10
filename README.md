# webgpu-ddgi

Real-time dynamic diffuse global illumination (DDGI) for WebGPU.

The core is plain WebGPU and WGSL with no engine dependency. Renderers plug in through small adapters: three.js first, then Babylon, Pixi, react-three-fiber or your own.

**Status:** a working prototype, not published yet; it will be published as `@pmndrs/ddgi`. Light bounces in real time through a grid of probes, and follows moving lights, objects and the sun, textured materials, several probe volumes and one that follows the camera. M1 and M2 are signed off, M3 (moving objects) and M4 (materials) are built and in their last checks, and M5 (quality presets, saved bakes, a `LightProbeGrid` export, these docs) is under way.

## Getting started with three.js

GI takes the place of three's `LightProbeGrid`, in as many lines. You give it a box and a number of probes; `start` copies your scene, warms the probes up, and from then on GI updates itself inside `renderer.render`:

```ts
import * as THREE from 'three/webgpu';
import { createGIRenderer, GIVolume } from '@pmndrs/ddgi/three';

const { renderer } = await createGIRenderer(); // a WebGPURenderer on a device with the limits GI needs
renderer.setSize(innerWidth, innerHeight);
document.body.appendChild(renderer.domElement);

const scene = new THREE.Scene(); // your meshes, materials and lights, as usual
const camera = new THREE.PerspectiveCamera(50, innerWidth / innerHeight, 0.1, 100);

const volume = new GIVolume(6, 5, 6, 6, 6, 6); // like new LightProbeGrid(6, 5, 6, 6, 6, 6): a 6 × 5 × 6 m box, 6 probes a side
volume.position.set(0, 2.5, 0);
scene.add(volume);
await volume.start(renderer, scene); // like probes.bake(renderer, scene)

renderer.setAnimationLoop(() => renderer.render(scene, camera)); // no per-frame GI call
```

What's different from `LightProbeGrid`:

- **It keeps updating.** Move a light, open a door or turn the sun, and GI follows within a few frames. `bake` is a one-off.
- **It knows about walls.** Each probe keeps how far it can see in every direction, so light leaks far less through walls between probes, and probes inside walls switch themselves off.
- **`createGIRenderer`** makes three's renderer on a GPU device with the adapter's full limits. three's own device caps a storage buffer at WebGPU's default, 128 MiB, which large scenes (Sponza, Bistro) outgrow. If you make the renderer yourself, pass it `requiredLimits(adapter)` from `@pmndrs/ddgi/core`.
- **One word sets the quality** for slower GPUs: `await volume.start(renderer, scene, { preset: 'low' })`.

Where to go next:

- [examples/](examples/): the code above on three's own probe-grid scene (`01-hello`), GI added once per pixel on three's post-processing pipeline (`02-composite`), and one small page for each feature.
- [docs/api/three.md](docs/api/three.md): the three.js adapter, every option, and what it does with your scene.
- [docs/guides/adapter.md](docs/guides/adapter.md): GI from plain WebGPU, or an adapter for another engine, step by step through [examples/webgpu/hello/](examples/webgpu/hello/).
- [docs/api/platform.md](docs/api/platform.md): the engine-free API, `@pmndrs/ddgi/core`.

The examples run from this repository: `pnpm install`, then `pnpm examples`. The viewer (`pnpm dev`) shows every test scene with GI, its debug views and its settings.

## The project

- [SPEC.md](SPEC.md): what v1 is, how it's built, how we judge it, and the milestones.
- [PLAN.md](PLAN.md): build order, how agents work, and what "done" means for a task.
- [TASKS.md](TASKS.md) and [PROGRESS.md](PROGRESS.md): the live task board and the running log.
- [CLAUDE.md](CLAUDE.md): orientation for agents; [docs/NOTES.md](docs/NOTES.md) and [docs/DECISIONS.md](docs/DECISIONS.md): learned facts and settled choices.
- [docs/AUDIT-2026-10-30.md](docs/AUDIT-2026-10-30.md): why this is a fresh start. It reviews the earlier experiment in [DennisSmolek/ddgi](https://github.com/DennisSmolek/ddgi).
- [research/](research/): background notes carried over from that experiment.
- `native/rtxgi-ddgi`: NVIDIA's RTXGI-DDGI test harness (a fork), used only as an external comparison and timing tool. Its code is not part of this library.
