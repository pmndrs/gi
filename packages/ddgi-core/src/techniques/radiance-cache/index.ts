// The radiance cache, the platform's second technique: a faithful port of AMD's
// GI-1.0/1.2 (Capsaicin v1.3; docs/design/radiance-cache.md). Screen probes (RC2.1, RC2.2) and the world cache (RC3.1, RC3.2).
export { RadianceCacheTechnique, type RadianceCacheProbes } from './radiance-cache-technique.ts';
export {
    DEFAULT_RADIANCE_CACHE_SETTINGS,
    PROBE_CELLS,
    PROBE_SIZE,
    PROBE_SPAWN_TILE_SIZE,
    SAMPLING_MODE,
    screenProbeLayout,
    type RadianceCacheSettings,
    type ScreenProbeLayout,
    type WorldCacheKind,
} from './settings.ts';
export { RadianceCacheShadingModule } from './shading.ts';
export { RADIANCE_CACHE_DEBUG_VIEW, type RadianceCacheDebugViewId } from './wgsl/debug.ts';
