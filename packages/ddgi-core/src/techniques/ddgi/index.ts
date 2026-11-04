// DDGI, the first GI technique built on the platform (docs/design/ddgi.md).
export { DDGITechnique, type DDGIField } from './ddgi-technique.ts';
export { DEFAULT_DDGI_SETTINGS, type DDGISettings, type DDGISettingsInput, type DDGIVolumeSettings } from './settings.ts';
export { PROBE_STATE, PROBE_STATE_LABELS, type DistanceFormat } from './layout.ts';
// The one sampling function, for the shading module (T1.6) and engine-free test views.
export { DDGI_SAMPLE_WGSL } from './wgsl/sample.ts';
