# Evaluation protocol

These are proposed comparisons for the GI project. For each run record the revision, scene seed, fixed simulation step, rail time or pose, lighting state, adapter, backend, render and display sizes, cascade count and interval bounds, probe spacing, directional samples, trace budget and temporal settings.

| Scene sequence | What it isolates | Evidence to capture |
| --- | --- | --- |
| Static architectural rail | Camera motion, LOD transitions, probe residency and disocclusion | Hide a colored emitter behind an arch while a neutral receiver remains visible. Capture the same camera poses in direct-only and GI modes. Stop at the last pose and repeat from an empty history. |
| Door open → close → reopen, camera fixed | Moving occlusion and stale indirect light | Log the actual door transform used by raster and traversal in the same frame. Record linear indirect luminance inside a fixed receiver patch for each frame after closure and reopening. Compare both settled endpoints with a matching reference. |
| Ball pit at a fixed seed | Many moving instances, acceleration-structure update cost and history churn | Record the same trajectory, sphere count and lighting twice. Include a slow or stationary section to distinguish temporal lag from persistent leakage. Time static-scene preparation, dynamic updates and GI work separately. |

Start with direct-only and one-bounce diffuse reference images at identical poses and scene settings. If a path-traced reference is used, freeze every transform during sampling and record samples per pixel, bounce count, exposure and color space. With explicit GI output available, measure receiver error before tone mapping. A direct-only comparison shows where GI changes the result. The converged reference shows whether the change approaches the expected transport.

Capture a warm-up interval and a measured interval separately. Report median and p95 GPU frame cost, measurement count, and per-stage time when timestamps are available. Useful stages are dynamic scene update, trace, deposit, merge, gather and final resolve. Record probe bytes and active population when the implementation exposes them. Report CPU frame time separately when GPU timestamps are unavailable.

Make the first timing comparison at native output and matched GI settings. Then repeat with half-width and half-height internal rendering, first matching display-space probe spacing and then allowing probe spacing to follow render resolution. Disable adaptive quality for both repeats. This reproduces the distinction behind the external S5 timing table and makes any change in GI density explicit. [S5 benchmark](https://github.com/pmndrs/upscaler/pull/106)
