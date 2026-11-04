// three.js r187 adapter (T1.7): a Light subclass with a registered node reads
// the platform's shading module; the scene sync mirrors three's scene into
// `gi.scene`; ThreeGI runs both once per frame.
export { ThreeGI, type ThreeGIOptions, type ThreeGITimings } from './three-gi.ts';
export { GILight, GI_LIGHT_PRIORITY, entryWgsl, wgslParameterType, type GILightOptions } from './gi-light.ts';
export { wrapGpuTexture, type GpuTextureWrapOptions } from './external-texture.ts';
export { GIVolume } from './gi-volume.ts';
export { SceneSync, DEFAULT_EDIT_CHECKS_PER_FRAME, DEFAULT_OBJECTS_PER_FRAME, HOT_FRAMES, type SceneSyncOptions, type SyncFrameReport } from './scene-sync.ts';
export { createGIRenderer, requestGIDevice, type GIDevice, type GIDeviceOptions, type GIRenderer, type GIRendererOptions } from './host.ts';
export { GI_USER_DATA } from './user-data.ts';
export type { PmndrsSkyLike } from './environment.ts';
