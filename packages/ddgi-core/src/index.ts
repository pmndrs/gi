// Engine-agnostic core: the GI platform plus its techniques, on a bare GPUDevice.
// This package must never import three.js or any other engine (lint enforces it).
export * from './platform';
export * from './techniques/ddgi';
