/**
 * The `ddgi/write` kernel (docs/design/ddgi.md, "The write kernel"): copies
 * each updated probe's scratch slot into the atlases, fills the tile borders,
 * and writes the probe-data texel and the probe's point for the `points`
 * debug view. One workgroup per updated probe.
 *
 * It exists because the trace and blend kernels sample the atlases (bounce
 * light and the old values), and a dispatch can't both sample a texture and
 * store into it; core WebGPU has read-write storage textures only for 32-bit
 * single-channel formats.
 *
 * A second entry point, `ddgiWriteIrradianceTiles`, writes irradiance tiles
 * only, from a compacted list of probes and a scratch of irradiance texels
 * alone, with the same tile code: propagation's write (T2.4c, m2-lights.md
 * §4.5; the instant sky's nudge will use it too). Its list's length is read
 * from a word on the GPU, because the list is built there.
 *
 * With light groups (T2.5, light-groups.ts) each probe has a tile per
 * irradiance layer, side by side in the layered atlas: both entry points
 * write every layer's tile (distances once).
 *
 * Bindings (group 1; it uses no platform module): uniforms, the update list
 * and the scratch buffer (2 storage buffers), and four write-only storage
 * textures (the default per-stage limit): irradiance, distance, probe data
 * and points. The irradiance-only entry point binds uniforms, its list, its
 * scratch, the irradiance texture and its list's length (binding 7).
 */

import type { DistanceFormat, ScratchLayout } from '../layout.ts';
import { PROPAGATION_STATE_WORD, propagationScratchStrideWords, SCRATCH_PROBE_DATA_WORD } from '../layout.ts';
import { DDGI_BORDER_SOURCE_WGSL } from './border.ts';
import { DDGI_COMMON_WGSL } from './common.ts';

export const WRITE_WORKGROUP_SIZE = 64;

/** The write module's entry points. */
export const WRITE_ENTRY_POINTS = { all: 'ddgiWrite', irradianceTiles: 'ddgiWriteIrradianceTiles' } as const;

export interface WriteKernelShape {
    readonly irradianceTexels: number;
    readonly distanceTexels: number;
    readonly scratch: ScratchLayout;
    readonly distanceFormat: DistanceFormat;
}

export function writeKernelWgsl(shape: WriteKernelShape): string {
    const { irradianceTexels, distanceTexels, scratch, distanceFormat } = shape;
    return /* wgsl */ `
${DDGI_COMMON_WGSL}

// ---- DDGI write kernel (techniques/ddgi) -----------------------------------------------

const DDGI_WRITE_WORKGROUP_SIZE: u32 = ${WRITE_WORKGROUP_SIZE}u;
const DDGI_IRRADIANCE_TEXELS: u32 = ${irradianceTexels}u;
const DDGI_DISTANCE_TEXELS: u32 = ${distanceTexels}u;
const DDGI_SCRATCH_STRIDE: u32 = ${scratch.strideWords}u;
const DDGI_SCRATCH_IRRADIANCE: u32 = ${scratch.irradianceOffset}u;
const DDGI_SCRATCH_DISTANCE: u32 = ${scratch.distanceOffset}u;
const DDGI_SCRATCH_PROBE_DATA: u32 = ${scratch.probeDataOffset}u;
const DDGI_SCRATCH_OFFSET_XY: u32 = ${SCRATCH_PROBE_DATA_WORD.offsetXy}u;
const DDGI_SCRATCH_OFFSET_Z_AND_STATE: u32 = ${SCRATCH_PROBE_DATA_WORD.offsetZAndState}u;
const DDGI_SCRATCH_TILES_WRITTEN: u32 = ${SCRATCH_PROBE_DATA_WORD.tilesWritten}u;
// Irradiance layers (light groups), each layer's words after the one before.
const DDGI_IRRADIANCE_LAYERS: u32 = ${scratch.irradianceLayers}u;
const DDGI_IRRADIANCE_LAYER_WORDS: u32 = ${scratch.irradianceWords}u;
// An irradiance-only scratch (ddgiWriteIrradianceTiles): interior texels alone, every layer's.
const DDGI_IRRADIANCE_TILE_STRIDE: u32 = ${propagationScratchStrideWords(irradianceTexels, scratch.irradianceLayers)}u;
// Where the irradiance-only list's length is (PROPAGATION_STATE_WORD in layout.ts).
const DDGI_TILE_LIST_LENGTH: u32 = ${PROPAGATION_STATE_WORD.listed}u;

@group(1) @binding(0) var<uniform> ddgi: DdgiUniforms;
// The probes whose tiles this dispatch writes, and their scratch.
@group(1) @binding(1) var<storage, read> ddgiUpdateList: array<u32>;
@group(1) @binding(2) var<storage, read> ddgiScratch: array<u32>;
@group(1) @binding(3) var ddgiIrradianceOut: texture_storage_2d_array<rgba16float, write>;
@group(1) @binding(4) var ddgiDistanceOut: texture_storage_2d_array<${distanceFormat}, write>;
@group(1) @binding(5) var ddgiProbeDataOut: texture_storage_2d_array<rgba16float, write>;
// One texel per probe in probe-index order: world position, state.
@group(1) @binding(6) var ddgiPointsOut: texture_storage_2d<rgba32float, write>;
// ddgiWriteIrradianceTiles only: the word holding its list's length.
@group(1) @binding(7) var<storage, read> ddgiTileListState: array<u32>;

${DDGI_BORDER_SOURCE_WGSL}

// One probe's irradiance tiles, borders included, one per layer, from their
// interior texels in the scratch from word 'base' on (two words a texel: rg,
// then b and alpha; each layer after the one before). Layer L's tile is L
// probe grids to the right in the layered atlas.
fn ddgiStoreIrradianceTile(probe: vec3i, base: u32, thread: u32) {
    let irradianceTile = DDGI_IRRADIANCE_TEXELS + 2u * u32(DDGI_TILE_BORDER);
    let tileTexels = irradianceTile * irradianceTile;
    for (var index = thread; index < tileTexels * DDGI_IRRADIANCE_LAYERS; index += DDGI_WRITE_WORKGROUP_SIZE) {
        let layer = index / tileTexels;
        let inTile = index % tileTexels;
        let tileTexel = vec2u(inTile % irradianceTile, inTile / irradianceTile);
        let source = ddgiBorderSource(tileTexel, DDGI_IRRADIANCE_TEXELS);
        let word = base + layer * DDGI_IRRADIANCE_LAYER_WORDS + 2u * (source.y * DDGI_IRRADIANCE_TEXELS + source.x);
        let rg = unpack2x16float(ddgiScratch[word]);
        // b, and in alpha the blend's big-change verdict for the probe's next update.
        let blueAndChange = unpack2x16float(ddgiScratch[word + 1u]);
        let tile = vec2i(probe.x + i32(layer) * ddgiAtlasTilesAcross(), probe.z);
        textureStore(ddgiIrradianceOut, tile * i32(irradianceTile) + vec2i(tileTexel), ddgiProbeLayer(probe), vec4f(rg, blueAndChange));
    }
}

@compute @workgroup_size(DDGI_WRITE_WORKGROUP_SIZE)
fn ${WRITE_ENTRY_POINTS.all}(
    @builtin(workgroup_id) group: vec3u,
    @builtin(num_workgroups) groups: vec3u,
    @builtin(local_invocation_index) thread: u32,
) {
    // Dispatches past 65,535 workgroups use two dimensions.
    let slot = group.x + group.y * groups.x;
    if (slot >= ddgi.updateCount) {
        return;
    }
    let probeIndex = ddgiUpdateList[slot];
    let probe = ddgiProbeCoordinates(probeIndex, ddgiProbeCount());
    let slotBase = slot * DDGI_SCRATCH_STRIDE;

    let offsetXy = unpack2x16float(ddgiScratch[slotBase + DDGI_SCRATCH_PROBE_DATA + DDGI_SCRATCH_OFFSET_XY]);
    let offsetZAndState = unpack2x16float(ddgiScratch[slotBase + DDGI_SCRATCH_PROBE_DATA + DDGI_SCRATCH_OFFSET_Z_AND_STATE]);
    let offset = vec3f(offsetXy, offsetZAndState.x);
    let state = u32(round(offsetZAndState.y));
    if (thread == 0u) {
        textureStore(ddgiProbeDataOut, probe.xz, ddgiProbeLayer(probe), vec4f(offset, f32(state)));
        // The points view lists every volume's probes one after another, this volume's from probeBase on.
        let pointIndex = ddgi.probeBase + probeIndex;
        let pointsWidth = textureDimensions(ddgiPointsOut).x;
        textureStore(ddgiPointsOut, vec2u(pointIndex % pointsWidth, pointIndex / pointsWidth), vec4f(ddgiProbePosition(probe, offset), f32(state)));
    }
    // The blend says whether it wrote this probe's tiles: it did for every
    // probe it blended, active afterwards or still warming in state 3 after
    // its first estimate (T2.M1); off probes and probes coming back on keep
    // what they had.
    if (ddgiScratch[slotBase + DDGI_SCRATCH_PROBE_DATA + DDGI_SCRATCH_TILES_WRITTEN] == 0u) {
        return;
    }


    ddgiStoreIrradianceTile(probe, slotBase + DDGI_SCRATCH_IRRADIANCE, thread);

    let distanceTile = DDGI_DISTANCE_TEXELS + 2u * u32(DDGI_TILE_BORDER);
    for (var index = thread; index < distanceTile * distanceTile; index += DDGI_WRITE_WORKGROUP_SIZE) {
        let tileTexel = vec2u(index % distanceTile, index / distanceTile);
        let source = ddgiBorderSource(tileTexel, DDGI_DISTANCE_TEXELS);
        let moments = unpack2x16float(ddgiScratch[slotBase + DDGI_SCRATCH_DISTANCE + source.y * DDGI_DISTANCE_TEXELS + source.x]);
        // rg16float keeps xy; the rgba16float fallback ignores zw.
        textureStore(ddgiDistanceOut, probe.xz * i32(distanceTile) + vec2i(tileTexel), ddgiProbeLayer(probe), vec4f(moments, 0.0, 0.0));
    }
}

// Irradiance tiles only, for a list built on the GPU (propagation's, T2.4c):
// a fixed number of workgroups, each looping over the list's entries
// (PROPAGATE_WORKGROUPS in layout.ts says why not an indirect dispatch).
@compute @workgroup_size(DDGI_WRITE_WORKGROUP_SIZE)
fn ${WRITE_ENTRY_POINTS.irradianceTiles}(
    @builtin(workgroup_id) group: vec3u,
    @builtin(num_workgroups) groups: vec3u,
    @builtin(local_invocation_index) thread: u32,
) {
    let length = ddgiTileListState[DDGI_TILE_LIST_LENGTH];
    for (var entry = group.x; entry < length; entry += groups.x) {
        let probe = ddgiProbeCoordinates(ddgiUpdateList[entry], ddgiProbeCount());
        ddgiStoreIrradianceTile(probe, entry * DDGI_IRRADIANCE_TILE_STRIDE, thread);
    }
}
`;
}
