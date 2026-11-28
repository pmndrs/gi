// `@pmndrs/gi/radiance-cache`: the radiance cache, the platform's second technique,
// based on AMD GI-1.0/1.2 (screen probes over a world cache), engine-free. It runs
// on the platform from `@pmndrs/gi/core` (`gi.addTechnique(new RadianceCacheTechnique())`);
// with three.js, its shading module goes on a `GILight` and its screen inputs come
// from `screenInputs`, both from `@pmndrs/gi/three`.
//
// Its image-space denoiser is the `denoiser` package's `Gi1Denoiser` (a dependency).
// The optional neural tail (`neuralCache: 'tail'`) loads `@pathtracer/mlp` at run
// time; it isn't on npm or a dependency, and without it the tail turns off.
export * from '@pmndrs/gi-core/techniques/radiance-cache/index';
