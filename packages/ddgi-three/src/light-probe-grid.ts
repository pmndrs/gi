/**
 * The `LightProbeGrid` export (T5.3, docs/design/m5-coverage.md §3): a
 * snapshot of the DDGI field as three's own `LightProbeGrid` (r187,
 * `three/addons/lighting/LightProbeGrid.js`), for a static fallback, and its
 * JSON.
 *
 * - `toLightProbeGrid(data)` takes the core's `exportIrradianceSH` result
 *   (L2 radiance SH per probe, already in three's convention) and builds a
 *   grid whose probes sit exactly where ours do. `LightProbeGrid` puts its
 *   probes at cell centres, so the grid is `counts × spacing` wide and
 *   centred on our grid's centre. Its atlas is filled directly in three's own
 *   packing, so no `bake` call (and no renderer) is needed.
 * - `lightProbeGridToJSON` / `lightProbeGridFromJSON` write and read that
 *   grid. three's PR #33658 (a `toJSON` / `fromJSON` for `LightProbeGrid`)
 *   was still open when this was written (r188 milestone), so the format is
 *   ours, with a `formatVersion` (docs/api/three.md says so). It is shaped
 *   like the PR's (width, height, depth, resolution, position, 27 floats a
 *   probe) but stores the floats as plain JSON numbers rather than the PR's
 *   fixed point, so a round trip is exact.
 *
 * **three's packing** (`LightProbeGridNode.js`): one RGBA half-float 3D
 * texture, nx × ny × 7·(nz + 2). Sub-volume t (0..6) holds floats
 * 4t..4t + 3 of the probe's 27 (coefficient-major, then RGB; the 28th is 0),
 * at slices t·(nz + 2) + 1 + z, with a padding slice at each end copying the
 * nearest data slice so trilinear filtering never reads the next sub-volume.
 * The sampler reads E = `getShIrradianceAt(n, sh)`, so with our coefficients
 * it gives back our irradiance (up to L2), and three's lighting applies
 * albedo / π as it does for our GI light.
 *
 * **Known differences from live GI** (the fallback's cost, recorded, not
 * tuned): L2 holds less directional detail than DDGI's tiles; `LightProbeGrid`
 * has no visibility (no distances, no Chebyshev test), so light leaks
 * through walls between probes; relocation offsets are dropped; and three
 * samples half a cell along the normal with plain trilinear weights.
 */

import { Data3DTexture, DataUtils, HalfFloatType, LinearFilter, RGBAFormat } from 'three/webgpu';
import { LightProbeGrid } from 'three/addons/lighting/LightProbeGrid.js';
import { L2_SH_FLOATS_PER_PROBE, type IrradianceSHGrid } from '@pmndrs/gi-ddgi';

/** The JSON format's version: bump it when the layout changes, and read old versions explicitly. */
export const LIGHT_PROBE_GRID_JSON_FORMAT_VERSION = 1;

/** `LightProbeGridNode`'s padding slices at each end of a sub-volume (its `ATLAS_PADDING`). */
const ATLAS_PADDING = 1;

/** RGBA sub-volumes the 27 floats are packed into. */
const SUB_VOLUMES = 7;

/** Floats per RGBA texel. */
const TEXEL_FLOATS = 4;

/**
 * A grid's JSON (format version 1). The coefficients are 27 floats a probe,
 * probe index x + y·nx + z·nx·ny, then coefficient (three's order: constant,
 * y, z, x, xy, yz, 3z² − 1, xz, x² − y²), then r, g, b: radiance SH as three's
 * `LightProbe` stores it.
 */
export interface LightProbeGridJSON {
    readonly formatVersion: typeof LIGHT_PROBE_GRID_JSON_FORMAT_VERSION;
    readonly type: 'LightProbeGrid';
    readonly generator: string;
    /** Full size of the box along x, y and z (probes at cell centres). */
    readonly width: number;
    readonly height: number;
    readonly depth: number;
    /** Probes per axis. */
    readonly resolution: readonly [number, number, number];
    /** The box's centre in world space. */
    readonly position: readonly [number, number, number];
    readonly intensity: number;
    readonly priority: number;
    readonly coefficients: readonly number[];
}

/** A `LightProbeGrid` filled from coefficients rather than baked: `toLightProbeGrid` and `lightProbeGridFromJSON` make them. */
export class ExportedLightProbeGrid extends LightProbeGrid {
    readonly isExportedLightProbeGrid = true;
    /** The probes' coefficients, 27 floats a probe (as {@link LightProbeGridJSON}); what `lightProbeGridToJSON` writes. */
    readonly coefficients: Float32Array;
    /** The atlas made from them (also `texture`). Baking this grid would write a different texture, so don't bake it. */
    readonly coefficientTexture: Data3DTexture;

    constructor(
        width: number,
        height: number,
        depth: number,
        resolution: readonly [number, number, number],
        coefficients: Float32Array,
    ) {
        super(width, height, depth, resolution[0], resolution[1], resolution[2]);
        const probeCount = resolution[0] * resolution[1] * resolution[2];
        if (coefficients.length !== probeCount * L2_SH_FLOATS_PER_PROBE) {
            throw new Error(`@pmndrs/gi/three: a ${resolution.join(' × ')} grid needs ${probeCount * L2_SH_FLOATS_PER_PROBE} coefficients, got ${coefficients.length}`);
        }
        this.coefficients = coefficients;
        this.coefficientTexture = packCoefficientAtlas(coefficients, resolution);
        // The node reads `texture` when it builds the material, so this has to be in place before the first render.
        this.texture = this.coefficientTexture;
    }

    dispose(): void {
        this.coefficientTexture.dispose();
        super.dispose();
    }
}

/**
 * The 27 floats a probe in three's atlas (the file comment): an RGBA half
 * float `Data3DTexture`, nx × ny × 7·(nz + 2), filtered linearly.
 */
export function packCoefficientAtlas(coefficients: Float32Array, resolution: readonly [number, number, number]): Data3DTexture {
    const [nx, ny, nz] = resolution;
    const paddedSlices = nz + 2 * ATLAS_PADDING;
    const atlasDepth = SUB_VOLUMES * paddedSlices;
    const texels = new Uint16Array(nx * ny * atlasDepth * TEXEL_FLOATS);
    const writeTexel = (slice: number, x: number, y: number, probe: number, subVolume: number): void => {
        const target = ((slice * ny + y) * nx + x) * TEXEL_FLOATS;
        for (let component = 0; component < TEXEL_FLOATS; component++) {
            const float = subVolume * TEXEL_FLOATS + component;
            // The 28th float (sub-volume 6's alpha) is padding.
            const value = float < L2_SH_FLOATS_PER_PROBE ? coefficients[probe * L2_SH_FLOATS_PER_PROBE + float] : 0;
            texels[target + component] = DataUtils.toHalfFloat(value);
        }
    };
    for (let z = 0; z < nz; z++) {
        for (let y = 0; y < ny; y++) {
            for (let x = 0; x < nx; x++) {
                const probe = x + y * nx + z * nx * ny;
                for (let subVolume = 0; subVolume < SUB_VOLUMES; subVolume++) {
                    const base = subVolume * paddedSlices;
                    writeTexel(base + ATLAS_PADDING + z, x, y, probe, subVolume);
                    // The padding slices copy the nearest data slice (LightProbeGrid's own repack does the same).
                    if (z === 0) writeTexel(base, x, y, probe, subVolume);
                    if (z === nz - 1) writeTexel(base + ATLAS_PADDING + nz, x, y, probe, subVolume);
                }
            }
        }
    }
    const texture = new Data3DTexture(texels, nx, ny, atlasDepth);
    texture.format = RGBAFormat;
    texture.type = HalfFloatType;
    texture.minFilter = LinearFilter;
    texture.magFilter = LinearFilter;
    texture.generateMipmaps = false;
    texture.unpackAlignment = 1;
    texture.needsUpdate = true;
    return texture;
}

/** Placement options for {@link toLightProbeGrid}. */
export interface ToLightProbeGridOptions {
    /** `LightProbeGrid.intensity`. Default 1. */
    readonly intensity?: number;
    /** `LightProbeGrid.priority`. Default 0, three's default (our GI light's is 1). */
    readonly priority?: number;
}

/**
 * A `LightProbeGrid` from `exportIrradianceSH`'s data: probes where ours
 * are, the atlas filled, no bake needed. Copies the coefficients.
 */
export function toLightProbeGrid(data: IrradianceSHGrid, options: ToLightProbeGridOptions = {}): ExportedLightProbeGrid {
    const [nx, ny, nz] = data.counts;
    const [sx, sy, sz] = data.spacing;
    const grid = new ExportedLightProbeGrid(nx * sx, ny * sy, nz * sz, [nx, ny, nz], Float32Array.from(data.coefficients));
    // Our probe (0, 0, 0) is at origin; the grid's centre is the middle of the probes.
    grid.position.set(data.origin[0] + ((nx - 1) * sx) / 2, data.origin[1] + ((ny - 1) * sy) / 2, data.origin[2] + ((nz - 1) * sz) / 2);
    grid.intensity = options.intensity ?? 1;
    grid.priority = options.priority ?? 0;
    grid.updateMatrixWorld();
    grid.updateBoundingBox();
    return grid;
}

/**
 * The grid as JSON (format version {@link LIGHT_PROBE_GRID_JSON_FORMAT_VERSION}).
 * The coefficients are written as plain numbers, so reading them back gives
 * the same float32 values bit for bit. Only grids made from coefficients
 * (`toLightProbeGrid`, `lightProbeGridFromJSON`) carry them; three's r187
 * `bake` keeps its coefficients on the GPU.
 */
export function lightProbeGridToJSON(grid: ExportedLightProbeGrid): LightProbeGridJSON {
    const coefficients = Array.from(grid.coefficients);
    if (!coefficients.every(Number.isFinite)) throw new Error('@pmndrs/gi/three: the grid has coefficients that are not finite numbers, which JSON cannot hold');
    return {
        formatVersion: LIGHT_PROBE_GRID_JSON_FORMAT_VERSION,
        type: 'LightProbeGrid',
        generator: '@pmndrs/gi',
        width: grid.width,
        height: grid.height,
        depth: grid.depth,
        resolution: [grid.resolution.x, grid.resolution.y, grid.resolution.z],
        position: [grid.position.x, grid.position.y, grid.position.z],
        intensity: grid.intensity,
        priority: grid.priority,
        coefficients,
    };
}

/** Reads {@link lightProbeGridToJSON}'s JSON (an object, or the string) back into a grid. Refuses other formats and versions by name. */
export function lightProbeGridFromJSON(source: LightProbeGridJSON | string): ExportedLightProbeGrid {
    const json = (typeof source === 'string' ? JSON.parse(source) : source) as Partial<LightProbeGridJSON>;
    if (json.type !== 'LightProbeGrid') throw new Error(`@pmndrs/gi/three: not LightProbeGrid JSON (type ${String(json.type)})`);
    if (json.formatVersion !== LIGHT_PROBE_GRID_JSON_FORMAT_VERSION) {
        throw new Error(`@pmndrs/gi/three: LightProbeGrid JSON format version ${String(json.formatVersion)} is not supported (this version reads ${LIGHT_PROBE_GRID_JSON_FORMAT_VERSION})`);
    }
    const { width, height, depth, resolution, position, coefficients } = json;
    if (typeof width !== 'number' || typeof height !== 'number' || typeof depth !== 'number' || !resolution || !position || !coefficients) {
        throw new Error('@pmndrs/gi/three: LightProbeGrid JSON is missing its size, resolution, position or coefficients');
    }
    const grid = new ExportedLightProbeGrid(width, height, depth, [resolution[0], resolution[1], resolution[2]], Float32Array.from(coefficients));
    grid.position.set(position[0], position[1], position[2]);
    grid.intensity = json.intensity ?? 1;
    grid.priority = json.priority ?? 0;
    grid.updateMatrixWorld();
    grid.updateBoundingBox();
    return grid;
}
