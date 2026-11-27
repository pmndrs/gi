# Radiance cascades research

The Sponza example in the upscaler uses Split Radiance Cascades. Its lineage runs from the authors’ Chandra implementation to Johnathon Selstad’s Three.js example and then to the upscaler’s S5 showcase. The ray tracer in the Three.js port builds a static scene BVH. A camera rail works with that baseline. A moving door or ball needs geometry updates that reach the GI tracer as well as the raster pass. [Chandra](https://github.com/entropylost/chandra/tree/9af00e8de135a4126cb137d1ec62b451bd6b8e37) · [Three.js PR](https://github.com/mrdoob/three.js/pull/35349) · [S5 PR](https://github.com/pmndrs/upscaler/pull/106)

| Read | Question |
| --- | --- |
| [Foundations and interval arithmetic](foundations/method.md) | What quantity is stored and how should occlusion compose? |
| [Chandra, Three.js and S5](implementations/port-lineage.md) | Which stages and capabilities can we compare directly? |
| [Other 3D implementations](implementations/other-3d.md) | Which probe-placement and geometry-update choices have working examples? |
| [Evaluation protocol](evaluation.md) | Which measurements distinguish a dynamic GI improvement from a change in resolution or history? |
| [Where our solver differs from Chandra](../../docs/radiance-cascades/divergence.md) | Which differences are measured, which are kept and which are still worth trying? |
| [Reflections and reference experiments](reflections.md) | How do rough floor reflections respond to camera motion, roughness and offscreen emitters? |

Sources checked on 2026-11-19. The implementation links identify the revisions used for these notes.
