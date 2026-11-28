// DDGI's own three.js code (DECISIONS 2026-11-27, "flat and technique-chained"):
// `DDGIVolume`, a LightProbeGrid-shaped volume and the simple path
// (`volume.start`), its LightProbeGrid export, and the volume start. The
// three.js layer every technique shares is packages/three. Published as part of
// `@pmndrs/gi/three` (and `.`).
export { DDGIVolume } from './ddgi-volume.ts';
export {
    ExportedLightProbeGrid,
    LIGHT_PROBE_GRID_JSON_FORMAT_VERSION,
    lightProbeGridFromJSON,
    lightProbeGridToJSON,
    packCoefficientAtlas,
    toLightProbeGrid,
    type LightProbeGridJSON,
    type ToLightProbeGridOptions,
} from './light-probe-grid.ts';
export { DEFAULT_GI_BUDGET, DEFAULT_WARM_UP, ThreeGILimitError, type DDGIVolumeStartOptions, type StartedGI, type WarmUpSettings } from './volume-start.ts';
