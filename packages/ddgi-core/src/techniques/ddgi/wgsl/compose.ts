/**
 * `ddgi/compose` (T2.5, docs/design/m2-lights.md §5): the composed
 * irradiance atlas, Σ weight × layer per texel, borders included, from the
 * light-group layers of the layered atlas. Renderers sample the composed
 * atlas, so shading costs the same with any number of layers (sampling five
 * layers in the fragment shader would push T1.6's 0.9 ms past 1.0 ms).
 *
 * It runs only with more than one layer (light groups on), after
 * `ddgi/write` and propagation's write, over the tiles they wrote, and over
 * every tile in a frame where a weight changed: so a weight change shows in
 * the same frame, with no lag and no noise, and marks nothing.
 *
 * Why its own texture: a dispatch can't read and write one texture (core
 * WebGPU has no read-write `rgba16float` storage textures), and renderers
 * bind one irradiance texture.
 *
 * Three entry points over one module, all one workgroup per probe tile:
 * - `ddgiComposeUpdated`: the update list's probes (`ddgi.updateCount`);
 * - `ddgiComposeListed`: a list built on the GPU, propagation's, whose
 *   length is a word of its state buffer; a fixed number of workgroups loop
 *   over it, as propagation's write does (PROPAGATE_WORKGROUPS in layout.ts
 *   says why not an indirect dispatch);
 * - `ddgiComposeAll`: every probe, after a weight change.
 *
 * No reduction or compaction, so nothing here gains from subgroups (the
 * subgroups-first rule is for kernels that reduce or compact): one thread a
 * texel, a plain sum.
 *
 * The sum goes through f32 and is rounded to the nearest half float before
 * it is stored. A `textureStore` to an `rgba16float` texture truncates on the
 * RTX 3060 (NOTES: 0.035% low on average), and the composed value, unlike
 * the layers', is not already a half float.
 *
 * Bindings (group 1, no platform module): DDGI's uniforms, the list (storage,
 * read), the list's state (storage, read), the layered atlas (sampled, read
 * with `textureLoad`), the composed atlas (write-only storage texture) and the
 * layers' weights (a uniform buffer of {@link COMPOSE_WEIGHTS_BYTES}).
 * 2 storage buffers.
 */

import { MAX_IRRADIANCE_LAYERS } from '../light-groups.ts';
import { PROPAGATION_STATE_WORD } from '../layout.ts';
import { DDGI_COMMON_WGSL } from './common.ts';

export const COMPOSE_WORKGROUP_SIZE = 64;

/** The module's entry points. */
export const COMPOSE_ENTRY_POINTS = { updated: 'ddgiComposeUpdated', listed: 'ddgiComposeListed', all: 'ddgiComposeAll' } as const;

/** Bytes of the weights uniform: a vec4 (rgb, 0) per layer, MAX_IRRADIANCE_LAYERS of them. */
export const COMPOSE_WEIGHTS_BYTES = MAX_IRRADIANCE_LAYERS * 16;

export interface ComposeKernelShape {
    readonly irradianceTexels: number;
    /** Irradiance layers (at least 2: with one, nothing is composed). */
    readonly irradianceLayers: number;
}

export function composeKernelWgsl(shape: ComposeKernelShape): string {
    const { irradianceTexels, irradianceLayers } = shape;
    if (irradianceLayers < 2 || irradianceLayers > MAX_IRRADIANCE_LAYERS) {
        throw new RangeError(`@pmndrs/ddgi: ddgi/compose adds 2 to ${MAX_IRRADIANCE_LAYERS} layers, got ${irradianceLayers}`);
    }
    return /* wgsl */ `
${DDGI_COMMON_WGSL}

// ---- DDGI compose kernel (techniques/ddgi) ---------------------------------------------

const DDGI_COMPOSE_WORKGROUP_SIZE: u32 = ${COMPOSE_WORKGROUP_SIZE}u;
const DDGI_IRRADIANCE_TEXELS: u32 = ${irradianceTexels}u;
const DDGI_IRRADIANCE_LAYERS: u32 = ${irradianceLayers}u;
// Where the GPU-built list's length is (PROPAGATION_STATE_WORD in layout.ts).
const DDGI_LIST_LENGTH: u32 = ${PROPAGATION_STATE_WORD.listed}u;

// Per layer: the host's linear rgb weight (w unused).
struct DdgiComposeWeights {
    layers: array<vec4f, ${MAX_IRRADIANCE_LAYERS}>,
}

@group(1) @binding(0) var<uniform> ddgi: DdgiUniforms;
// The probes whose tiles this dispatch composes (ddgiComposeUpdated, ddgiComposeListed).
@group(1) @binding(1) var<storage, read> ddgiComposeList: array<u32>;
// ddgiComposeListed only: the word holding its list's length.
@group(1) @binding(2) var<storage, read> ddgiComposeListState: array<u32>;
@group(1) @binding(3) var ddgiLayeredIrradiance: texture_2d_array<f32>;
@group(1) @binding(4) var ddgiComposedOut: texture_storage_2d_array<rgba16float, write>;
@group(1) @binding(5) var<uniform> ddgiComposeWeights: DdgiComposeWeights;

// The half float nearest a value, so the store below is exact (see the file
// comment). Half floats have 10 fraction bits: neighbours are 2^(e - 10)
// apart for a value in [2^e, 2^(e+1)), and evenly spaced below the smallest
// normal half.
fn ddgiNearestHalf(value: f32) -> f32 {
    let magnitude = max(abs(value), 6.103515625e-5);
    let step = exp2(f32(frexp(magnitude).exp - 1) - 10.0);
    return round(value / step) * step;
}

// One probe's composed tile, borders included: every tile texel is the
// weighted sum of the same texel in each layer's tile (layer L's tiles are L
// atlas widths of tiles to the right, light-groups.ts). Borders copy interior
// texels in every layer alike, so the sum's borders are right too.
fn ddgiComposeTile(probe: vec3i, thread: u32) {
    let tile = DDGI_IRRADIANCE_TEXELS + 2u * u32(DDGI_TILE_BORDER);
    let tilesAcross = ddgiAtlasTilesAcross();
    let atlasLayer = ddgiProbeLayer(probe);
    for (var index = thread; index < tile * tile; index += DDGI_COMPOSE_WORKGROUP_SIZE) {
        let inTile = vec2i(vec2u(index % tile, index / tile));
        var composed = vec3f(0.0);
        for (var layer = 0u; layer < DDGI_IRRADIANCE_LAYERS; layer++) {
            let layerTile = vec2i(probe.x + i32(layer) * tilesAcross, probe.z);
            let texel = textureLoad(ddgiLayeredIrradiance, layerTile * i32(tile) + inTile, atlasLayer, 0).rgb;
            composed += ddgiComposeWeights.layers[layer].rgb * texel;
        }
        let rounded = vec3f(ddgiNearestHalf(composed.r), ddgiNearestHalf(composed.g), ddgiNearestHalf(composed.b));
        textureStore(ddgiComposedOut, probe.xz * i32(tile) + inTile, atlasLayer, vec4f(rounded, 0.0));
    }
}

// The update list's probes, one workgroup each (dispatches past 65,535 workgroups use two dimensions).
@compute @workgroup_size(DDGI_COMPOSE_WORKGROUP_SIZE)
fn ${COMPOSE_ENTRY_POINTS.updated}(
    @builtin(workgroup_id) group: vec3u,
    @builtin(num_workgroups) groups: vec3u,
    @builtin(local_invocation_index) thread: u32,
) {
    let slot = group.x + group.y * groups.x;
    if (slot >= ddgi.updateCount) {
        return;
    }
    ddgiComposeTile(ddgiProbeCoordinates(ddgiComposeList[slot], ddgiProbeCount()), thread);
}

// A list built on the GPU (propagation's): a fixed number of workgroups, each
// taking the list's entries in turn.
@compute @workgroup_size(DDGI_COMPOSE_WORKGROUP_SIZE)
fn ${COMPOSE_ENTRY_POINTS.listed}(
    @builtin(workgroup_id) group: vec3u,
    @builtin(num_workgroups) groups: vec3u,
    @builtin(local_invocation_index) thread: u32,
) {
    let length = ddgiComposeListState[DDGI_LIST_LENGTH];
    for (var entry = group.x; entry < length; entry += groups.x) {
        ddgiComposeTile(ddgiProbeCoordinates(ddgiComposeList[entry], ddgiProbeCount()), thread);
    }
}

// Every probe, one workgroup each: a frame where a weight changed.
@compute @workgroup_size(DDGI_COMPOSE_WORKGROUP_SIZE)
fn ${COMPOSE_ENTRY_POINTS.all}(
    @builtin(workgroup_id) group: vec3u,
    @builtin(num_workgroups) groups: vec3u,
    @builtin(local_invocation_index) thread: u32,
) {
    let counts = ddgiProbeCount();
    let probeIndex = group.x + group.y * groups.x;
    if (probeIndex >= counts.x * counts.y * counts.z) {
        return;
    }
    ddgiComposeTile(ddgiProbeCoordinates(probeIndex, counts), thread);
}
`;
}
