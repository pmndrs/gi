/**
 * DDGI's shading module (T1.6): how a renderer reads the field while it
 * shades pixels, in the shape every technique publishes
 * (`platform/contract/shading-module.ts`).
 *
 * - **Resources:** the three field textures (`texture_2d_array`, one layer per
 *   probe row), a sampler for each of the two filtered ones, and the volume as
 *   four `vec4f` uniforms. Nothing else, and no storage buffers, because three
 *   can't bind a foreign buffer and compatibility-mode devices may allow none
 *   in fragment shaders (NOTES). 3 textures, within the rule of at most 6
 *   (docs/api/platform.md, walk-through gap 1).
 * - **Code:** `giIrradiance`, `giCoverage` and `giDebugColor` (wgsl/shading.ts),
 *   with `giIrradiance` being `ddgiSampleIrradiance` unchanged.
 * - **No `giRadiance` in M1** (`provides.radiance` is false). A rough-only
 *   fallback (the irradiance in the reflected direction ÷ π) is only right at
 *   roughness 1; anything smoother needs a blend with the host's environment
 *   specular, which belongs with the three adapter's environment work.
 * - **`version` never changes in M1:** the code doesn't depend on the probe
 *   counts or texel sizes (those are uniforms), and both distance formats
 *   sample the same way. A new `prepare` swaps the textures, which hosts pick
 *   up by re-reading `texture()`.
 */

import type { ShadingModule, ShadingParameter } from '../../platform/contract/shading-module.ts';
import { SAMPLE_PARAMETER_VEC4_COUNT } from './layout.ts';
import { DDGI_SHADING_WGSL } from './wgsl/shading.ts';

/** What the module reads from the technique. Both throw a clear error before `prepare`. */
export interface DDGIShadingSource {
    /** The current field textures (replaced only by `prepare`). */
    fieldTextures(): { readonly irradiance: GPUTexture; readonly distance: GPUTexture; readonly probeData: GPUTexture };
    /**
     * The four `vec4f` parameters (volume origin and normal bias, spacing and
     * view bias, probe counts, texel counts), refreshed from the live settings.
     * Always the same 16-float array, updated in place.
     */
    sampleParameters(): Float32Array;
}

/** Floats per `vec4f` uniform. */
const VEC4_FLOATS = 4;

/** The parameter names, which are also the WGSL parameter names in wgsl/shading.ts. */
export const DDGI_SHADING_PARAMETER_NAMES = {
    irradiance: 'ddgiIrradiance',
    irradianceSampler: 'ddgiIrradianceSampler',
    distance: 'ddgiDistance',
    distanceSampler: 'ddgiDistanceSampler',
    probeData: 'ddgiProbeData',
    volumeOrigin: 'ddgiVolumeOrigin',
    volumeSpacing: 'ddgiVolumeSpacing',
    volumeCounts: 'ddgiVolumeCounts',
    volumeTexels: 'ddgiVolumeTexels',
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
        ],
        returns: 'vec3f',
    },
];

/** The uniform struct and variable `globalsCode` declares for the `vec4f` parameters. */
const GLOBALS_UNIFORM_STRUCT = 'GiShadingUniforms';
const GLOBALS_UNIFORM_VARIABLE = 'giShadingUniforms';

export class DDGIShadingModule implements ShadingModule {
    readonly techniqueId = 'ddgi';
    readonly code = DDGI_SHADING_WGSL;
    readonly parameters: readonly ShadingParameter[];
    readonly provides = { radiance: false } as const;
    /** See the file comment: nothing in M1 changes the code or the parameters' shape. */
    readonly version = 1;

    private readonly source: DDGIShadingSource;

    constructor(source: DDGIShadingSource) {
        this.source = source;
        const names = DDGI_SHADING_PARAMETER_NAMES;
        const texture = (name: string, pick: (textures: ReturnType<DDGIShadingSource['fieldTextures']>) => GPUTexture): ShadingParameter => ({
            kind: 'texture',
            name,
            dimension: '2d-array',
            // rgba16float and rg16float (or its rgba16float fallback) are all filterable.
            sampleType: 'float',
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
        // Views into the source's one 16-float array (always the same array),
        // made on first use, so value() allocates nothing afterwards.
        const vec4 = (name: string, index: number): ShadingParameter => {
            let view: Float32Array | null = null;
            return {
                kind: 'vec4f',
                name,
                value: () => {
                    const all = source.sampleParameters();
                    view ??= all.subarray(index * VEC4_FLOATS, (index + 1) * VEC4_FLOATS);
                    return view;
                },
            };
        };
        this.parameters = [
            texture(names.irradiance, (textures) => textures.irradiance),
            sampler(names.irradianceSampler, names.irradiance),
            texture(names.distance, (textures) => textures.distance),
            sampler(names.distanceSampler, names.distance),
            // Read with textureLoad (one texel per probe), so no sampler.
            texture(names.probeData, (textures) => textures.probeData),
            vec4(names.volumeOrigin, 0),
            vec4(names.volumeSpacing, 1),
            vec4(names.volumeCounts, 2),
            vec4(names.volumeTexels, 3),
        ];
    }

    /**
     * `code`, then the resources as globals at `group`, then the wrappers
     * `giIrradianceAt(position, normal, viewDirection, screenUV)`,
     * `giCoverageAt(position, normal)` and `giDebugColorAt(position, normal, view)`.
     *
     * Bindings: every texture and sampler parameter in order from 0, then one
     * uniform buffer holding the `vec4f` parameters in order (fill it with
     * {@link writeUniforms}). Self-contained: include this string alone.
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
        lines.push(`struct ${GLOBALS_UNIFORM_STRUCT} {`, ...uniforms.map((name) => `    ${name}: vec4f,`), '}');
        lines.push(`@group(${group}) @binding(${binding}) var<uniform> ${GLOBALS_UNIFORM_VARIABLE}: ${GLOBALS_UNIFORM_STRUCT};`);
        const resourceArguments = this.parameters.map((parameter) => (parameter.kind === 'vec4f' ? `${GLOBALS_UNIFORM_VARIABLE}.${parameter.name}` : parameter.name)).join(', ');
        for (const entry of ENTRY_POINTS) {
            const signature = entry.arguments.map(([name, type]) => `${name}: ${type}`).join(', ');
            const callArguments = entry.arguments.map(([name]) => name).join(', ');
            lines.push(`fn ${entry.name}At(${signature}) -> ${entry.returns} {`, `    return ${entry.name}(${callArguments}, ${resourceArguments});`, '}');
        }
        return `${lines.join('\n')}\n`;
    }

    /** The four `vec4f` parameters, in order: 16 floats. */
    writeUniforms(target: Float32Array): void {
        const values = this.source.sampleParameters();
        if (target.length < SAMPLE_PARAMETER_VEC4_COUNT * VEC4_FLOATS) {
            throw new RangeError(`@pmndrs/ddgi: writeUniforms needs room for ${SAMPLE_PARAMETER_VEC4_COUNT * VEC4_FLOATS} floats, got ${target.length}`);
        }
        target.set(values);
    }
}

function wgslTextureType(parameter: Extract<ShadingParameter, { kind: 'texture' }>): string {
    const sampled = parameter.sampleType === 'uint' ? 'u32' : 'f32';
    const dimension = parameter.dimension === '2d-array' ? '2d_array' : parameter.dimension;
    return `texture_${dimension}<${sampled}>`;
}
