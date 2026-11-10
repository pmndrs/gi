// Engine-agnostic core: the GI platform plus its techniques, on a bare GPUDevice.
// The published package splits it by layer: ./platform is `@pmndrs/gi/core`,
// ./techniques/ddgi is `@pmndrs/gi/ddgi` (packages/ddgi/src).
// This package must never import three.js or any other engine (lint enforces it).
export * from './platform';
export * from './techniques/ddgi';
