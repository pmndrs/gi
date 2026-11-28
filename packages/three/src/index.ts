// The three.js r187 layer every technique shares (T1.7, T1.14): a Light subclass
// with a registered node reads the platform's shading module; the scene sync
// mirrors three's scene into `gi.scene`; ThreeGI runs both once per frame; the
// composite (`ddgi()` with `giAlbedo` on three's own pass() and mrt()) and the
// screen inputs serve every technique (docs/api/three.md). DDGI's own three.js
// code (DDGIVolume, the simple path) is packages/ddgi-three. Nothing here imports
// a technique package (lint enforces it).
export { jitterPixelsToNdc, ThreeGI, upscalerJitter, type ThreeGIJitter, type ThreeGIOptions, type ThreeGITimings, type UpscalerJitterLike } from './three-gi.ts';
export { GILight, GI_LIGHT_PRIORITY, entryWgsl, wgslParameterType, type GILightOptions } from './gi-light.ts';
export { wrapGpuTexture, type GpuTextureWrapOptions } from './external-texture.ts';
export {
    ddgi,
    giAlbedo,
    DEFAULT_GI_OCCLUSION_RADIUS,
    GI_ALBEDO_OUTPUT,
    GI_NORMAL_OUTPUT,
    type CompositeGISource,
    type DDGICompositeNode,
    type DDGICompositeOptions,
    type GIAlbedoNode,
    type GIOcclusionSettings,
    type GIOcclusionSource,
} from './composite.ts';
export {
    screenInputs,
    giDiffuseAlbedo,
    GI_SCREEN_ALBEDO_OUTPUT,
    GI_SCREEN_INSTANCE_OUTPUT,
    GI_SCREEN_NORMAL_OUTPUT,
    GI_SCREEN_VELOCITY_OUTPUT,
    type GIScreenChannelNode,
    type ScreenInputsNode,
    type ScreenInputsOptions,
    type ScreenInputsSource,
} from './screen-inputs.ts';
export { SceneSync, DEFAULT_EDIT_CHECKS_PER_FRAME, DEFAULT_OBJECTS_PER_FRAME, HOT_FRAMES, type SceneSyncOptions, type SyncFrameReport } from './scene-sync.ts';
export { createGIRenderer, requestGIDevice, type GIDevice, type GIDeviceOptions, type GIRenderer, type GIRendererOptions } from './host.ts';
export { GI_MATERIAL_USER_DATA, GI_USER_DATA } from './user-data.ts';
export type { PmndrsSkyLike } from './environment.ts';
