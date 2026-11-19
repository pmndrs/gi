# AMD GI-1.0 research

Research for porting AMD's two-level radiance cache (GI-1.0, with the GI-1.1 glossy and GI-1.2 multi-bounce additions) to WebGPU as the second technique on the `@pmndrs/ddgi` platform. Started 2026-11-09.

- [PLAN.md](PLAN.md): the implementation plan for the `radiance-cache` technique: faithful port first, divergence register, native yardstick, levers, phases and decisions. Start here.
- [idtech8-performance.md](idtech8-performance.md): id Tech 8 performance techniques mapped to GI-1.0 passes, ranked as levers.
- [paper.md](paper.md): what the paper says, pass by pass, with parameters and limitations.
- [source-map.md](source-map.md): where everything lives in Capsaicin v1.3 (`914b915`, MIT), data structures, memory, and the WebGPU replacement for each hardware feature.
- [native-capsaicin.md](native-capsaicin.md): building and running the native reference on the RTX 3060 Laptop, command-line options, first captures and timings. Launcher: [`tools/reference/capsaicin/run-capsaicin.ps1`](../../tools/reference/capsaicin/run-capsaicin.ps1). Source: submodule [`native/capsaicin`](../../native/capsaicin).
- [related-work.md](related-work.md): Lumen, id Tech 8, radiance cascades, SHaRC/NRC, surfels, Brixelizer GI, browser work, ranked extensions and licensing.
- [pmndrs-siblings.md](pmndrs-siblings.md): what `@pmndrs/upscaler` and `@pmndrs/denoiser` cover, and what GI-1.0 could contribute back.
- [sources/](sources/SOURCES.md): the paper (arXiv and AMD copies), the Brixelizer GI talk, and the GI-1.1 paper with hashes and terms. The GI-1.1 paper is "not for redistribution", so it is kept local and is not committed.

Key facts so far:
- Capsaicin v1.3 ships **GI-1.2**: GI-1.0 plus the GI-1.1 reflections and a second-bounce layer in the hash cache.
- On our RTX 3060 Laptop, Capsaicin's GI block takes about 5.9 ms at 720p with hardware RT (about 4.6 ms without reflections). Our targets follow that native number, not the paper's RTX 3080 figures.
- Only Capsaicin, Brixelizer GI, Webgiya and kajiya are portable code. NVIDIA's SHaRC/RTXGI/NRC are not (RTX SDK license).
