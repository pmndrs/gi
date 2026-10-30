# webgpu-ddgi

Real-time dynamic diffuse global illumination (DDGI) for WebGPU.

The core is plain WebGPU and WGSL with no engine dependency. Renderers plug in through small adapters: three.js first, then Babylon, Pixi, react-three-fiber or your own.

**Status:** specification. No code yet.

- [SPEC.md](SPEC.md): what v1 is, how it's built, how we judge it, and the milestones.
- [docs/AUDIT-2026-10-30.md](docs/AUDIT-2026-10-30.md): why this is a fresh start. It reviews the earlier experiment in [DennisSmolek/ddgi](https://github.com/DennisSmolek/ddgi).
- [research/](research/): background notes carried over from that experiment.
- `native/rtxgi-ddgi`: NVIDIA's RTXGI-DDGI test harness (a fork), used only as an external comparison and timing tool. Its code is not part of this library.
