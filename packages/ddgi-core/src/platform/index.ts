// GI platform: shared scene, resource and pass contracts that techniques build on.
// The contract (T0.11), GPU timing and the tracer (T0.10).
// The scene copy (./scene, T1.2) is deliberately not exported here: its layouts
// and upload entry points are private to the platform. Platform code imports it
// by path (`./scene/index.ts`); hosts and techniques see only the contract.
export * from './contract';
// The platform object (T1.3): what GIPlatformModule in the contract declares.
export { createGI } from './gi-platform.ts';
export { isSupported, optionalFeatures, requiredLimits } from './device-requirements.ts';
export * from './timing';
export * from './trace';
