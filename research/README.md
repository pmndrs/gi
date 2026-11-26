# Research notes

These notes were carried over unchanged from the earlier experiment, [DennisSmolek/ddgi at review-baseline-2026-10-30](https://github.com/DennisSmolek/ddgi/tree/review-baseline-2026-10-30). Relative links inside them point into that repository, not this one. Treat their status claims as historical; [SPEC.md](../SPEC.md) is authoritative.

**Written in this repository** (not carried over): [ddgi-dynamic-updates.md](ddgi-dynamic-updates.md), how shipped probe GI systems update under change (2026-11-06).

## Radiance cascades

[Method, source implementations and evaluation design](radiance-cascades/README.md) for the split radiance cascades experiment.

## References for later (not on M2's path)

- **Tokuyoshi (AMD), "A Numerically Stable Implementation of the von Mises–Fisher Distribution on S²",** technical report 25-01-5053, January 2025 ([PDF](https://gpuopen.com/download/A_Numerically_Stable_Implementation_of_the_von_Mises%E2%80%93Fisher_Distribution_on_S2.pdf)). Passed on by Dennis, 2026-11-06.
  - *What it gives:*
    - float32-stable vMF evaluation: the normaliser via `x / expm1(x)`, and `exp(−κ/2 · |ω − μ|²)` in place of `exp(κ(ω·μ − 1))`, which fails for sharp lobes;
    - stable sampling: `cos θ = 1 + log1p(ξ · expm1(−2κ)) / κ`, with a small-κ branch;
    - short HLSL listings that port directly to WGSL.
  - *WGSL caveat:* WGSL has no `expm1` or `log1p`. The paper's `expm1` workaround uses only `exp` and `log`, so it is safe. Its `log1p` relies on the compiler not folding `(x + 1) − 1` (`volatile`). Don't trust that under Tint or Naga: use the `exp`/`log` form or a series instead.
  - *Where it could matter:*
    - a sharp sun lobe, or sky lobes for time of day;
    - guided probe rays toward bright directions, an alternative to M2's planned direct emitter sampling for small-light noise (it needs weighting in the blend);
    - glossy or radiance probes in a later milestone;
    - the reference path tracer.
- [gi-1.0/](gi-1.0/README.md): AMD GI-1.0/1.2 research and the radiance-cache plan (moved from the `ddgi` research branch on 2026-11-19).
