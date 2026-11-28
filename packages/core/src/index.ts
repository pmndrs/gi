// The engine-free GI platform on a bare GPUDevice: `@pmndrs/gi/core`. Techniques are
// their own workspace packages (packages/ddgi, packages/cache) and build on it; Split RC
// stays in ./techniques/radiance-cascades until it moves to packages/radiance-cascades.
// This package must never import three.js or any other engine, or a technique package
// (lint enforces it).
export * from './platform';
