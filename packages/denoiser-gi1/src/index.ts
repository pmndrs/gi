/**
 * @pmndrs/denoiser-gi1: AMD GI-1's image-space GI denoiser (Capsaicin v1.3's
 * `ReprojectGI` and `FilterGI`, MIT) in WGSL, on a bare `GPUDevice`. It
 * imports nothing from `@pmndrs/ddgi` or any engine (lint-enforced), so it
 * can move into `pmndrs/denoiser` as it is (README.md).
 */

export { GI1Denoiser, type GI1DenoiserHistory, type GI1DenoiserOptions } from './gi1-denoiser.ts';
export type { DiffuseInputs, FrameCamera, FrameGuides, Mat4, PassTimer, TemporalConfig } from './types.ts';
export { frameConstants, reprojectionCellSize, type FrameConstants } from './camera.ts';
export { CAPSAICIN_FORMATS, CORE_FORMATS, denoiserKernels, formatsFor, type GI1DenoiserFormats } from './wgsl.ts';
