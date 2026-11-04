// three.js r187 adapter (T1.7): a Light subclass with a registered node reads
// the platform's shading module; the scene sync mirrors three's scene into
// `gi.scene`; ThreeGI runs both once per frame.
export { ThreeGI, type ThreeGIOptions, type ThreeGITimings } from './three-gi.ts';
export { GILight, GI_LIGHT_PRIORITY, type GILightOptions } from './gi-light.ts';
export { SceneSync, type SceneSyncOptions, type SyncFrameReport } from './scene-sync.ts';
export { createGIRenderer, requestGIDevice, type GIDevice, type GIDeviceOptions, type GIRenderer, type GIRendererOptions } from './host.ts';
export { GI_USER_DATA } from './user-data.ts';
export type { PmndrsSkyLike } from './environment.ts';
