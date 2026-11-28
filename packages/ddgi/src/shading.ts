/**
 * DDGI's shading module (T1.6): how a renderer reads the field while it
 * shades pixels, in the shape every technique publishes
 * (`platform/contract/shading-module.ts`).
 *
 * - **Resources:** the three field textures (`texture_2d_array`, shared by
 *   every volume, volumes.ts), a sampler for each of the two filtered ones,
 *   and the volume table (`rgba32float`, one row per volume in priority
 *   order, read with `textureLoad`). Nothing else, and no storage buffers,
 *   because three can't bind a foreign buffer and compatibility-mode devices
 *   may allow none in fragment shaders (NOTES). 4 textures, within the rule
 *   of at most 6 (docs/api/platform.md, walk-through gap 1).
 * - **Code:** `giIrradiance`, `giCoverage` and `giDebugColor` (wgsl/shading.ts),
 *   on `ddgiSampleVolumes`, the blend the update kernel uses for bounce light;
 *   and `giProbeIrradiance`, one probe's own light, for the `points` debug
 *   view (DV-03).
 * - **No `giRadiance` yet** (`provides.radiance` is false). A rough-only
 *   fallback (the irradiance in the reflected direction ÷ π) is only right at
 *   roughness 1; anything smoother needs a blend with the host's environment
 *   specular, which belongs with the three adapter's environment work.
 * - **`version` 2** since several volumes (M3, T3.5): the volume's four
 *   `vec4f` uniforms became the volume table, a texture. Nothing else
 *   changes the code: probe counts, texel sizes, the number of volumes and
 *   the biases are all in the table, and both distance formats sample the
 *   same way. A new `prepare` swaps the textures, which hosts pick up by
 *   re-reading `texture()`.
 */

import type { ShadingModule, ShadingParameter } from '@pmndrs/gi-core/platform/contract/shading-module';
import { DDGI_SHADING_WGSL } from './wgsl/shading.ts';

/** The textures the module reads from the technique. */
export interface DDGIFieldTextures {
    readonly irradiance: GPUTexture;
    readonly distance: GPUTexture;
    readonly probeData: GPUTexture;
    /** One row per volume, in priority order (volumes.ts); kept current by the technique. */
    readonly volumeTable: GPUTexture;
}

/** What the module reads from the technique. Throws a clear error before `prepare`. */
export interface DDGIShadingSource {
    /** The current field textures (replaced only by `prepare`). */
    fieldTextures(): DDGIFieldTextures;
}

/** The parameter names, which are also the WGSL parameter names in wgsl/shading.ts. */
export const DDGI_SHADING_PARAMETER_NAMES = {
    irradiance: 'ddgiIrradiance',
    irradianceSampler: 'ddgiIrradianceSampler',
    distance: 'ddgiDistance',
    distanceSampler: 'ddgiDistanceSampler',
    probeData: 'ddgiProbeData',
    volumeTable: 'ddgiVolumeTable',
} as const;

/**
 * The entry points and the arguments each takes before the resources, for
 * the parameterless wrappers `globalsCode` writes (`giIrradianceAt`, ...).
 */
const ENTRY_POINTS: readonly { readonly name: string; readonly arguments: readonly (readonly [string, string])[]; readonly returns: string }[] = [
    {
        name: 'giIrradiance',
        arguments: [
            ['position', 'vec3f'],
            ['normal', 'vec3f'],
            ['viewDirection', 'vec3f'],
            ['screenUV', 'vec2f'],
        ],
        returns: 'vec3f',
    },
    {
        name: 'giCoverage',
        arguments: [
            ['position', 'vec3f'],
            ['normal', 'vec3f'],
        ],
        returns: 'f32',
    },
    {
        name: 'giDebugColor',
        arguments: [
            ['position', 'vec3f'],
            ['normal', 'vec3f'],
            ['view', 'u32'],
            ['screenUV', 'vec2f'],
        ],
        returns: 'vec3f',
    },
    {
        name: 'giProbeIrradiance',
        arguments: [
            ['point', 'u32'],
            ['direction', 'vec3f'],
        ],
        returns: 'vec3f',
    },
];

/** The uniform struct and variable `globalsCode` declares for `vec4f` parameters, when there are any. */
const GLOBALS_UNIFORM_STRUCT = 'GiShadingUniforms';
const GLOBALS_UNIFORM_VARIABLE = 'giShadingUniforms';
/** Floats per `vec4f` uniform. */
const VEC4_FLOATS = 4;

export class DDGIShadingModule implements ShadingModule {
    readonly techniqueId = 'ddgi';
    readonly code = DDGI_SHADING_WGSL;
    readonly parameters: readonly ShadingParameter[];
    /** Its field answers at any point in its volumes: another technique can read it as a bounce source (P6.3). */
    readonly provides = { radiance: false, worldSpaceIrradiance: true } as const;
    /** See the file comment: 2 since the volume table replaced the volume's uniforms. */
    readonly version = 2;

    constructor(source: DDGIShadingSource) {
        const names = DDGI_SHADING_PARAMETER_NAMES;
        const texture = (
            name: string,
            dimension: '2d' | '2d-array',
            sampleType: 'float' | 'unfilterable-float',
            pick: (textures: DDGIFieldTextures) => GPUTexture,
        ): ShadingParameter => ({
            kind: 'texture',
            name,
            dimension,
            sampleType,
            texture: () => pick(source.fieldTextures()),
        });
        // Bilinear within a probe's tile; the one-texel borders make it seamless,
        // and clamping keeps the outermost tiles from wrapping.
        const sampler = (name: `${string}Sampler`, textureName: string): ShadingParameter => ({
            kind: 'sampler',
            name,
            texture: textureName,
            filtering: 'linear',
            addressMode: 'clamp-to-edge',
        });
        this.parameters = [
            // rgba16float and rg16float (or its rgba16float fallback) are all filterable.
            texture(names.irradiance, '2d-array', 'float', (textures) => textures.irradiance),
            sampler(names.irradianceSampler, names.irradiance),
            texture(names.distance, '2d-array', 'float', (textures) => textures.distance),
            sampler(names.distanceSampler, names.distance),
            // Read with textureLoad (one texel per probe), so no sampler.
            texture(names.probeData, '2d-array', 'float', (textures) => textures.probeData),
            // rgba32float, read with textureLoad: not filterable without float32-filterable, and needn't be.
            texture(names.volumeTable, '2d', 'unfilterable-float', (textures) => textures.volumeTable),
        ];
    }

    /**
     * `code`, then the resources as globals at `group`, then the wrappers
     * `giIrradianceAt(position, normal, viewDirection, screenUV)`,
     * `giCoverageAt(position, normal)`, `giDebugColorAt(position, normal, view, screenUV)`
     * and `giProbeIrradianceAt(point, direction)`.
     *
     * Bindings: every texture and sampler parameter in order from 0, then one
     * uniform buffer holding the `vec4f` parameters in order, when there are
     * any (since version 2 there are none). Self-contained: include this
     * string alone.
     */
    globalsCode(group: number): string {
        const lines: string[] = [this.code, `// ---- DDGI shading globals (globalsCode(${group})) ---------------------------------------`];
        let binding = 0;
        const uniforms: string[] = [];
        for (const parameter of this.parameters) {
            if (parameter.kind === 'vec4f') {
                uniforms.push(parameter.name);
                continue;
            }
            const type = parameter.kind === 'sampler' ? 'sampler' : wgslTextureType(parameter);
            lines.push(`@group(${group}) @binding(${binding}) var ${parameter.name}: ${type};`);
            binding++;
        }
        // WGSL has no empty structs, so with no vec4f parameters there is no uniform buffer either.
        if (uniforms.length > 0) {
            lines.push(`struct ${GLOBALS_UNIFORM_STRUCT} {`, ...uniforms.map((name) => `    ${name}: vec4f,`), '}');
            lines.push(`@group(${group}) @binding(${binding}) var<uniform> ${GLOBALS_UNIFORM_VARIABLE}: ${GLOBALS_UNIFORM_STRUCT};`);
        }
        const resourceArguments = this.parameters.map((parameter) => (parameter.kind === 'vec4f' ? `${GLOBALS_UNIFORM_VARIABLE}.${parameter.name}` : parameter.name)).join(', ');
        for (const entry of ENTRY_POINTS) {
            const signature = entry.arguments.map(([name, type]) => `${name}: ${type}`).join(', ');
            const callArguments = entry.arguments.map(([name]) => name).join(', ');
            lines.push(`fn ${entry.name}At(${signature}) -> ${entry.returns} {`, `    return ${entry.name}(${callArguments}, ${resourceArguments});`, '}');
        }
        return `${lines.join('\n')}\n`;
    }

    /** The `vec4f` parameters, in order: none since version 2 (the volume table holds what they did), so this writes nothing. */
    writeUniforms(target: Float32Array): void {
        let offset = 0;
        for (const parameter of this.parameters) {
            if (parameter.kind !== 'vec4f') continue;
            if (target.length < offset + VEC4_FLOATS) throw new RangeError(`@pmndrs/gi: writeUniforms needs room for ${offset + VEC4_FLOATS} floats, got ${target.length}`);
            target.set(parameter.value(), offset);
            offset += VEC4_FLOATS;
        }
    }
}

function wgslTextureType(parameter: Extract<ShadingParameter, { kind: 'texture' }>): string {
    const sampled = parameter.sampleType === 'uint' ? 'u32' : 'f32';
    const dimension = parameter.dimension === '2d-array' ? '2d_array' : parameter.dimension;
    return `texture_${dimension}<${sampled}>`;
}
