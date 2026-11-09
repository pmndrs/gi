// GI platform: shared scene, resource and pass contracts that techniques build on.
// The contract (T0.11), GPU timing and the platform object (T1.3).
// The scene copy (./scene, T1.2) and the tracer's internals (./trace: builders,
// buffers, the batch kernel, the CPU walk) are deliberately not exported here:
// hosts and techniques reach them only through the contract (`scene`,
// `queries.rays`). Platform code and tests import them by path.
export * from './contract';
// The platform object (T1.3): what GIPlatformModule in the contract declares.
export { createGI } from './gi-platform.ts';
export { isSupported, optionalFeatures, requiredLimits } from './device-requirements.ts';
export * from './timing';
// A debug view of the host's screen inputs (RC1.1), engine-free.
export {
    DEFAULT_SCREEN_INPUTS_VIEW_SETTINGS,
    SCREEN_CHANNELS,
    SCREEN_INPUTS_VIEW_FORMAT,
    ScreenInputsView,
    instanceDebugColor,
    instanceDebugHash,
    type ScreenChannel,
    type ScreenInputsViewSettings,
} from './screen/screen-inputs-view.ts';
// Sizes a technique needs to make buffers for `queries.rays.encodeBatch`.
export { RAY_BYTES } from './trace/trace-gpu.ts';
export { HIT_RECORD_BYTES, HIT_RECORD_WORD } from './trace/wgsl.ts';
