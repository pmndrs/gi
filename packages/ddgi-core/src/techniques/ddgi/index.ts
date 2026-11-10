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
export { PROBE_HISTORY_WORD, PROBE_HISTORY_WORDS, PROBE_STATE, PROBE_STATE_LABELS, type DistanceFormat } from './layout.ts';
// The shading module renderers read (technique.shading), and its debug view ids.
export { DDGIShadingModule, DDGI_SHADING_PARAMETER_NAMES } from './shading.ts';
export { DDGI_DEBUG_COLORS, DDGI_DEBUG_VIEW } from './wgsl/shading.ts';
// The one sampling function, for the shading module and engine-free test views.
export { DDGI_SAMPLE_WGSL } from './wgsl/sample.ts';
// The field as L2 SH per probe, three's convention (T5.3): the LightProbeGrid export's source.
export {
    exportIrradianceSH,
    fillInactiveProbes,
    l2CosineLobe,
    l2ShBasis,
    projectIrradianceToSH,
    shProjectionDirections,
    EXPORT_SH_DIRECTIONS,
    L2_SH_COEFFICIENTS,
    L2_SH_FLOATS_PER_PROBE,
    type ExportIrradianceSHOptions,
    type IrradianceSHGrid,
} from './irradiance-sh.ts';
