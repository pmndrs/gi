// three.js r187 adapter (T1.7, T1.14): a Light subclass with a registered node reads
// the platform's shading module; the scene sync mirrors three's scene into
// `gi.scene`; ThreeGI runs both once per frame. The simple path is
// `GIVolume.start`; the fast path for dense scenes is `ddgi()` with `giAlbedo`
// on three's own pass() and mrt() (docs/api/three.md).
export { ThreeGI, type ThreeGIOptions, type ThreeGITimings } from './three-gi.ts';
export { GILight, GI_LIGHT_PRIORITY, entryWgsl, wgslParameterType, type GILightOptions } from './gi-light.ts';
export { wrapGpuTexture, type GpuTextureWrapOptions } from './external-texture.ts';
export { GIVolume } from './gi-volume.ts';
export { DEFAULT_GI_BUDGET, DEFAULT_WARM_UP, ThreeGILimitError, type GIVolumeStartOptions, type StartedGI, type WarmUpSettings } from './volume-start.ts';
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
export { SceneSync, DEFAULT_EDIT_CHECKS_PER_FRAME, DEFAULT_OBJECTS_PER_FRAME, HOT_FRAMES, type SceneSyncOptions, type SyncFrameReport } from './scene-sync.ts';
export { createGIRenderer, requestGIDevice, type GIDevice, type GIDeviceOptions, type GIRenderer, type GIRendererOptions } from './host.ts';
export { GI_MATERIAL_USER_DATA, GI_USER_DATA } from './user-data.ts';
export type { PmndrsSkyLike } from './environment.ts';
