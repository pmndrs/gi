// The radiance cache, the platform's second technique: a faithful port of AMD's
// GI-1.0/1.2 (Capsaicin v1.3; docs/design/radiance-cache.md). Screen probes (RC2.1, RC2.2) and the world cache (RC3.1, RC3.2).
// Published as `@pmndrs/gi/radiance-cache` (packages/gi/src/radiance-cache.ts); its three-facing
// pieces (`screenInputs`, `GILight`, the `ddgi()` composite) are `@pmndrs/gi/three`'s.
export { RadianceCacheTechnique, type RadianceCacheProbes } from './radiance-cache-technique.ts';
export {
    DEFAULT_RADIANCE_CACHE_SETTINGS,
    PROBE_CELLS,
    PROBE_SIZE,
    PROBE_SPAWN_TILE_SIZE,
    SAMPLING_MODE,
    SCREEN_HISTORY_SETTINGS,
    screenProbeLayout,
    type RadianceCacheSettings,
    type ScreenProbeLayout,
    type WorldCacheKind,
} from './settings.ts';
export { RadianceCacheShadingModule } from './shading.ts';
export { RADIANCE_CACHE_DEBUG_VIEW, type RadianceCacheDebugViewId } from './wgsl/debug.ts';
export {
    applyCompareValues,
    capsaicinCompareValues,
    COMPARE_GROUP_LABELS,
    COMPARE_GROUPS,
    COMPARE_URL_NAMES,
    compareToggle,
    compareTogglesIn,
    compareValuesFromText,
    HOST_COMPARE_DEFAULTS,
    hostCompareValue,
    RADIANCE_CACHE_COMPARE_TOGGLES,
    RADIANCE_CACHE_VARIANT_PRESETS,
    readCompareValues,
    variantOfValues,
    variantPreset,
    writeCompareValues,
    type CompareGroup,
    type RadianceCacheVariantPreset,
    type VariantNeeds,
    type CompareToggle,
    type CompareValues,
} from './compare-toggles.ts';
