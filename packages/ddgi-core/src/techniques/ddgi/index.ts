// DDGI, the first GI technique built on the platform (docs/design/ddgi.md).
export { DDGITechnique, type DDGIField } from './ddgi-technique.ts';
export {
    DEFAULT_DDGI_SETTINGS,
    DEFAULT_WARM_UP_MAX_SECONDS,
    DEFAULT_WARM_UP_MIN_PASSES,
    DEFAULT_WARM_UP_PASSES,
    probeGrid,
    type DDGISettings,
    type DDGISettingsInput,
    type DDGIVolumeSettings,
    type ProbeGrid,
} from './settings.ts';
// Quality presets (docs/design/m5-coverage.md §1): one word instead of thirty settings.
export {
    DDGI_PRESET_NAMES,
    ddgiPreset,
    isDDGIPresetName,
    recommendPreset,
    type DDGIPreset,
    type DDGIPresetName,
    type DDGIPresetSettings,
    type PresetAdapterInfo,
    type PresetAdapterLimits,
    type RecommendPresetHints,
} from './presets.ts';
export { PROBE_HISTORY_WORD, PROBE_HISTORY_WORDS, PROBE_STATE, PROBE_STATE_LABELS, type DistanceFormat } from './layout.ts';
// The shading module renderers read (technique.shading), and its debug view ids.
export { DDGIShadingModule, DDGI_SHADING_PARAMETER_NAMES } from './shading.ts';
export { DDGI_DEBUG_COLORS, DDGI_DEBUG_VIEW } from './wgsl/shading.ts';
// The one sampling function, for the shading module and engine-free test views.
export { DDGI_SAMPLE_WGSL } from './wgsl/sample.ts';
