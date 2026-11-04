/**
 * Where DDGI's data lives on the GPU (docs/design/ddgi.md, "The textures",
 * "The write kernel", "Bindings"): atlas sizes and formats, the scratch
 * buffer's layout, and the uniform buffer's layout. Plain numbers, so they are
 * unit-tested without a GPU and shared by the TypeScript and the WGSL.
 */

import { GILimitError } from '../../platform/contract/errors.ts';
import type { ProbeGrid } from './settings.ts';

/** One texel of border on every side of a tile, so bilinear filtering never reads a neighbour's tile. */
export const TILE_BORDER = 1;

/** Probe data: one texel per probe. xyz: relocation offset ÷ spacing; w: state. */
export const PROBE_DATA_FORMAT: GPUTextureFormat = 'rgba16float';
export const IRRADIANCE_FORMAT: GPUTextureFormat = 'rgba16float';

/**
 * The distance atlas is `rg16float` (mean, mean squared) when the device has
 * `texture-formats-tier1`, which makes `rg16float` a storage format; core
 * WebGPU doesn't. The declared fallback is `rgba16float`, twice the memory.
 * Both are filterable without `float32-filterable`.
 */
export type DistanceFormat = 'rg16float' | 'rgba16float';

export function distanceFormatFor(features: ReadonlySet<string>): DistanceFormat {
    return features.has('texture-formats-tier1') ? 'rg16float' : 'rgba16float';
}

export const BYTES_PER_TEXEL: Readonly<Record<DistanceFormat, number>> = { rg16float: 4, rgba16float: 8 };

/** Probe states (the probe-data texture's w channel). */
export const PROBE_STATE = {
    active: 0,
    offInsideGeometry: 1,
    offNoSurfaceNearby: 2,
    notUpdatedYet: 3,
} as const;

/** Legend for debug views and counters. */
export const PROBE_STATE_LABELS: Readonly<Record<number, string>> = {
    [PROBE_STATE.active]: 'active',
    [PROBE_STATE.offInsideGeometry]: 'off: inside geometry',
    [PROBE_STATE.offNoSurfaceNearby]: 'off: no surface nearby',
    [PROBE_STATE.notUpdatedYet]: 'not updated yet',
};

export interface AtlasSize {
    readonly width: number;
    readonly height: number;
    /** One layer per probe row y. */
    readonly layers: number;
    /** Tile side including the border. */
    readonly tile: number;
}

/** A layer per probe row `y`; inside a layer, probe `(x, z)` owns tile `(x, z)`. */
export function atlasSize(grid: ProbeGrid, interiorTexels: number): AtlasSize {
    const tile = interiorTexels + 2 * TILE_BORDER;
    return { width: grid.counts[0] * tile, height: grid.counts[2] * tile, layers: grid.counts[1], tile };
}

export interface FieldLayout {
    readonly irradiance: AtlasSize;
    readonly distance: AtlasSize;
    readonly probeData: AtlasSize;
    readonly distanceFormat: DistanceFormat;
}

export function fieldLayout(grid: ProbeGrid, irradianceTexels: number, distanceTexels: number, distanceFormat: DistanceFormat): FieldLayout {
    return {
        irradiance: atlasSize(grid, irradianceTexels),
        distance: atlasSize(grid, distanceTexels),
        // One texel per probe, no border.
        probeData: { width: grid.counts[0], height: grid.counts[2], layers: grid.counts[1], tile: 1 },
        distanceFormat,
    };
}

/** Throws `GILimitError` naming the first texture limit the layout is past. */
export function checkFieldLimits(layout: FieldLayout, limits: Pick<GPUSupportedLimits, 'maxTextureDimension2D' | 'maxTextureArrayLayers'>): void {
    for (const atlas of [layout.irradiance, layout.distance]) {
        const largest = Math.max(atlas.width, atlas.height);
        if (largest > limits.maxTextureDimension2D) throw new GILimitError('maxTextureDimension2D', largest, limits.maxTextureDimension2D);
    }
    if (layout.irradiance.layers > limits.maxTextureArrayLayers) {
        throw new GILimitError('maxTextureArrayLayers', layout.irradiance.layers, limits.maxTextureArrayLayers);
    }
}

/** Bytes of the three textures. */
export function fieldBytes(layout: FieldLayout): number {
    const bytes = (atlas: AtlasSize, bytesPerTexel: number): number => atlas.width * atlas.height * atlas.layers * bytesPerTexel;
    return bytes(layout.irradiance, 8) + bytes(layout.distance, BYTES_PER_TEXEL[layout.distanceFormat]) + bytes(layout.probeData, 8);
}

/**
 * One probe's slot in the scratch buffer, in 32-bit words, written by the
 * update kernel and copied into the atlases by the write kernel. Values are
 * packed with `pack2x16float`, the atlases' own precision.
 *
 * - irradiance: 2 words per interior texel (rg, then b with 0);
 * - distance: 1 word per interior texel (mean, mean squared);
 * - probe data: 2 words (offset xy, then offset z with the state).
 */
export interface ScratchLayout {
    readonly irradianceWords: number;
    readonly distanceWords: number;
    readonly probeDataWords: number;
    /** Words per probe slot. */
    readonly strideWords: number;
    readonly irradianceOffset: number;
    readonly distanceOffset: number;
    readonly probeDataOffset: number;
}

export const SCRATCH_WORDS_PER_IRRADIANCE_TEXEL = 2;
export const SCRATCH_WORDS_PER_DISTANCE_TEXEL = 1;
export const SCRATCH_PROBE_DATA_WORDS = 2;

export function scratchLayout(irradianceTexels: number, distanceTexels: number): ScratchLayout {
    const irradianceWords = irradianceTexels * irradianceTexels * SCRATCH_WORDS_PER_IRRADIANCE_TEXEL;
    const distanceWords = distanceTexels * distanceTexels * SCRATCH_WORDS_PER_DISTANCE_TEXEL;
    return {
        irradianceWords,
        distanceWords,
        probeDataWords: SCRATCH_PROBE_DATA_WORDS,
        strideWords: irradianceWords + distanceWords + SCRATCH_PROBE_DATA_WORDS,
        irradianceOffset: 0,
        distanceOffset: irradianceWords,
        probeDataOffset: irradianceWords + distanceWords,
    };
}

/**
 * The uniform buffer both kernels read (`DdgiUniforms` in WGSL). The first
 * four vec4s are exactly the vec4 parameters `ddgiSampleIrradiance` takes, so
 * the update kernel and a renderer pass the same values.
 *
 * | vec4 | contents |
 * | --- | --- |
 * | 0 `volumeOrigin` | xyz: world position of probe (0, 0, 0); w: normal bias |
 * | 1 `volumeSpacing` | xyz: spacing; w: view bias |
 * | 2 `volumeCounts` | xyz: probes per axis (floats); w: unused |
 * | 3 `volumeTexels` | x: irradiance interior texels; y: distance interior texels; zw: unused |
 * | 4-6 `rayRotation` | this frame's ray rotation, one column per vec4 |
 * | 7 `blend` | hysteresis, irradianceChangeThreshold, maxBrightnessChange, distanceSharpness |
 * | 8 `rays` | maxRayDistance, distance clamp, bounceEnergy, unused |
 * | 9 `counts` (u32) | raysPerProbe, update count, frame, unused |
 */
export const UNIFORM_VEC4 = {
    volumeOrigin: 0,
    volumeSpacing: 1,
    volumeCounts: 2,
    volumeTexels: 3,
    rayRotation: 4,
    blend: 7,
    rays: 8,
    counts: 9,
} as const;
export const UNIFORM_VEC4_COUNT = 10;
export const UNIFORM_BYTES = UNIFORM_VEC4_COUNT * 16;
/** The shading parameters: vec4s 0-3. */
export const SAMPLE_PARAMETER_VEC4_COUNT = 4;
