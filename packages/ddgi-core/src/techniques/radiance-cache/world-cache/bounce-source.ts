/**
 * Experiment E10 (P6.3, phase 6): the world cache's second bounce read from
 * another technique's field instead of traced (`secondBounce: 'fallback'`, or
 * `'auto'` with a world-space fallback stacked).
 *
 * id Tech 8 shades its cache entries with last frame's irradiance volumes
 * (research/gi-1.0/idtech8-performance.md); the design doc's extension 1 is
 * "DDGI as the third cache level, through the `BounceSource` slot". Here the
 * slot is the platform's: the host stacks a technique under the radiance cache
 * (`gi.setFallback(radianceCache, ddgi)`), the platform encodes that technique
 * first each frame, and hands the radiance cache its shading module as
 * `TechniqueContext.fallback`. The radiance cache never imports DDGI: it binds
 * whatever shading module it is given, in compute, through the module's own
 * `globalsCode` (textures, samplers and a uniform buffer only; DDGI's field
 * samples with `textureSampleLevel` and `textureLoad`, which compute allows).
 *
 * One pass, `SampleBounceSourceCells`, replaces GI-1.2's three
 * (`PopulateMultibounceCells`, `GenerateMultibounceReservoirs`,
 * `UpdateMultibounceCells`): each first-bounce entry rebuilds its hit, reads
 * the fallback's irradiance E there (`giIrradianceAt` with the hit's normal on
 * the side the probe ray arrived from, and the ray's direction as the view
 * direction, so DDGI biases the point as it does for a camera) and its
 * coverage c (`giCoverageAt`), and adds albedo / π × c × E to its cell's
 * indirect scratch with a count of one. The coverage matters: outside every
 * volume DDGI's E is the nearest volume's edge value, not 0
 * (`ddgiSampleVolumes`), and c is 0 there, so a hit outside the field gets no
 * later bounce (as with multibounce off) rather than a wrong one; inside, c is
 * 1, and it fades over the volume's edge. Tracing GI-1.2's ray for the entries
 * with c < 1 instead would keep their second bounce; not built (register E10's
 * way back). A host covers the view with its volumes (a camera-following one). No entry
 * is discarded (GI-1.2 discards 70% to bound its rays; here there is no ray),
 * and E holds every bounce after the first that the fallback holds: DDGI's
 * field carries its own previous field as bounce light, so in a closed scene
 * the cell's indirect value is the light of every later bounce, not just the
 * second.
 *
 * It runs before `UpdateTiles`, so the samples are blended in the same frame
 * (GI-1.2's run after it and land a frame later): the field was updated
 * earlier in the frame, so its value is this frame's, and the indirect value
 * has no lag of its own beyond the cell's running mean, capped at
 * `fallbackIndirectSampleCount` (4) in this mode. A first-bounce entry whose probe
 * ray bypasses the cache (shorter than its cell, register D38) also adds the
 * value to its own probe ray, as GI-1.2's bypass routing does, without the
 * 1 / (1 − p) since nothing was discarded.
 *
 * The sky and emitters: the fallback's E includes what its own rays see of the
 * sky and of emissive surfaces. The cache's light choice has neither today
 * (registers D33, D21), so a cache hit gets them only here. RC3.4 puts the
 * sky in the cache's light choice (`GI_DIRECT_SKY_LIGHT_BIT`); in this mode
 * the cache leaves it out (`hashGridCacheSamplesSky`), or the sky arriving
 * straight at a hit would be counted twice. Emissive surfaces as cache lights
 * (D21's way back) would need the same rule.
 */

import type { ShadingModule, ShadingParameter } from '../../../platform/contract/shading-module.ts';
import type { HashGridCacheBufferLayout } from './capacity.ts';
import { populationHeadWgsl, POPULATION_WORKGROUP_SIZE, PROBE_RAY_SURFACE_WGSL } from './hash-grid-population-wgsl.ts';

/** `GPUShaderStage.COMPUTE` (the global doesn't exist in Node, where the tests import this file). */
const SHADER_STAGE_COMPUTE = 0x4;
/** `GPUBufferUsage` bits, for the same reason. */
const BUFFER_USAGE = { copyDst: 0x8, uniform: 0x40 } as const;

/** The group the bounce source's resources are bound in (the hash grid is 1, the ray data 2, hit shading 0). */
export const BOUNCE_SOURCE_GROUP = 3;

/** The pass's entry point. */
export const SAMPLE_BOUNCE_SOURCE_ENTRY_POINT = 'SampleBounceSourceCells';

/** Threads per workgroup: Capsaicin's `PopulateMultibounceCellsMain`'s 32, over the same first-bounce entries. */
export const SAMPLE_BOUNCE_SOURCE_WORKGROUP_SIZE = POPULATION_WORKGROUP_SIZE.populateMultibounceCells;

/** Floats per `vec4f` parameter. */
const VEC4_FLOATS = 4;

/**
 * The bind group layout entries for a shading module's globals
 * (`ShadingModule.globalsCode`): every texture and sampler parameter in order
 * from binding 0, then one uniform buffer of the `vec4f` parameters when
 * there are any.
 */
export function shadingModuleLayoutEntries(module: ShadingModule): GPUBindGroupLayoutEntry[] {
    const entries: GPUBindGroupLayoutEntry[] = [];
    let binding = 0;
    let vec4Count = 0;
    for (const parameter of module.parameters) {
        if (parameter.kind === 'vec4f') {
            vec4Count++;
            continue;
        }
        if (parameter.kind === 'texture') {
            entries.push({ binding, visibility: SHADER_STAGE_COMPUTE, texture: { sampleType: parameter.sampleType, viewDimension: parameter.dimension } });
        } else {
            entries.push({ binding, visibility: SHADER_STAGE_COMPUTE, sampler: { type: parameter.filtering === 'linear' ? 'filtering' : 'non-filtering' } });
        }
        binding++;
    }
    if (vec4Count > 0) entries.push({ binding, visibility: SHADER_STAGE_COMPUTE, buffer: { type: 'uniform' } });
    return entries;
}

/** Bytes of the uniform buffer `globalsCode` declares for the module's `vec4f` parameters (0 without any). */
export function shadingModuleUniformBytes(module: ShadingModule): number {
    return module.parameters.filter((parameter) => parameter.kind === 'vec4f').length * VEC4_FLOATS * 4;
}

/**
 * A shading module's resources bound for a compute pass: the samplers and
 * the uniform buffer made once, the bind group made again whenever one of the
 * module's textures is a different object (its technique's `prepare` swaps
 * them), and the uniforms written every frame.
 */
export class ShadingModuleBindings {
    readonly module: ShadingModule;
    /** The module's `version` the layout and the kernel's code were made for. */
    readonly version: number;
    private readonly device: GPUDevice;
    private readonly layout: GPUBindGroupLayout;
    private readonly samplers = new Map<string, GPUSampler>();
    private readonly uniformBuffer: GPUBuffer | null;
    private readonly uniformValues: Float32Array<ArrayBuffer>;
    private group: GPUBindGroup | null = null;
    private groupTextures: readonly GPUTexture[] = [];

    constructor(device: GPUDevice, module: ShadingModule, layout: GPUBindGroupLayout) {
        this.device = device;
        this.module = module;
        this.version = module.version;
        this.layout = layout;
        for (const parameter of module.parameters) {
            if (parameter.kind !== 'sampler') continue;
            const filter = parameter.filtering === 'linear' ? 'linear' : 'nearest';
            this.samplers.set(
                parameter.name,
                device.createSampler({
                    label: `radiance-cache/bounce source ${parameter.name}`,
                    magFilter: filter,
                    minFilter: filter,
                    addressModeU: parameter.addressMode,
                    addressModeV: parameter.addressMode,
                    addressModeW: parameter.addressMode,
                }),
            );
        }
        const uniformBytes = shadingModuleUniformBytes(module);
        this.uniformValues = new Float32Array(uniformBytes / 4);
        this.uniformBuffer =
            uniformBytes > 0 ? device.createBuffer({ label: 'radiance-cache/bounce source uniforms', size: uniformBytes, usage: BUFFER_USAGE.uniform | BUFFER_USAGE.copyDst }) : null;
    }

    /** Bytes this binding allocated (the uniform buffer; samplers and views have no size of note). */
    get memoryBytes(): number {
        return this.uniformBuffer?.size ?? 0;
    }

    /** This frame's bind group: rebuilt when a texture changed, uniforms written. */
    bindGroup(): GPUBindGroup {
        const textures = this.module.parameters.filter((parameter): parameter is Extract<ShadingParameter, { kind: 'texture' }> => parameter.kind === 'texture').map((parameter) => parameter.texture());
        if (this.uniformBuffer) {
            this.module.writeUniforms(this.uniformValues);
            this.device.queue.writeBuffer(this.uniformBuffer, 0, this.uniformValues);
        }
        if (this.group && textures.length === this.groupTextures.length && textures.every((texture, index) => texture === this.groupTextures[index])) return this.group;
        const entries: GPUBindGroupEntry[] = [];
        let binding = 0;
        let textureIndex = 0;
        for (const parameter of this.module.parameters) {
            if (parameter.kind === 'vec4f') continue;
            if (parameter.kind === 'texture') {
                entries.push({ binding, resource: textures[textureIndex++].createView({ dimension: parameter.dimension }) });
            } else {
                const sampler = this.samplers.get(parameter.name);
                if (!sampler) throw new Error(`@pmndrs/gi: no sampler made for ${parameter.name}`);
                entries.push({ binding, resource: sampler });
            }
            binding++;
        }
        if (this.uniformBuffer) entries.push({ binding, resource: { buffer: this.uniformBuffer } });
        this.group = this.device.createBindGroup({ label: `radiance-cache/bounce source (${this.module.techniqueId})`, layout: this.layout, entries });
        this.groupTextures = textures;
        return this.group;
    }

    dispose(): void {
        this.uniformBuffer?.destroy();
        this.group = null;
        this.groupTextures = [];
    }
}

/**
 * `SampleBounceSourceCells`: over the first-bounce entries (dispatched on
 * `VisibilityCountBuffer0`), the bounce source's irradiance at each hit, as
 * albedo / π × E, into the hit's cell's indirect scratch (see the file's
 * comment). Built with the platform's hit-shading module (group 0, for
 * `giSurfaceAt`), the hash grid (group 1), the ray data (group 2) and the
 * bounce source's globals (group 3).
 */
export function sampleBounceSourceCellsWgsl(layout: HashGridCacheBufferLayout, module: ShadingModule): string {
    const size = SAMPLE_BOUNCE_SOURCE_WORKGROUP_SIZE;
    return /* wgsl */ `
${populationHeadWgsl(layout)}
${PROBE_RAY_SURFACE_WGSL}

// ---- The bounce source: ${module.techniqueId}'s shading module, its resources as globals in group ${BOUNCE_SOURCE_GROUP} ----
${module.globalsCode(BOUNCE_SOURCE_GROUP)}

const HASHGRIDCACHE_BOUNCE_SOURCE_INV_PI: f32 = 0.318309886183791;

@compute @workgroup_size(${size})
fn ${SAMPLE_BOUNCE_SOURCE_ENTRY_POINT}(
    @builtin(workgroup_id) workgroup: vec3u,
    @builtin(num_workgroups) workgroups: vec3u,
    @builtin(local_invocation_index) local_index: u32,
) {
    let did = HashGridCache_WorkgroupIndex(workgroup, workgroups) * ${size}u + local_index;
    if (did >= atomicLoad(&g_HashGridCache_TileWords[HASHGRIDCACHE_VISIBILITYCOUNTBUFFER0])) {
        return; // out of bounds
    }

    // The first-bounce hit, rebuilt from its probe ray (the normal facing the side the ray came from).
    let query_index = HashGridCache_RayListWord(g_HashGridCache_RayLists.visibility_query_buffer, did);
    let cell_index = HashGridCache_RayListWord(g_HashGridCache_RayLists.visibility_cell_buffer, did);
    let probe_ray = g_ProbeRays.rays[query_index];
    let surface = HashGridCache_ProbeSurface(probe_ray);

    // The light arriving at the hit from every later bounce, as the bounce source holds it, and
    // the share the hit reflects toward the probe ray (Lambert, register D8): albedo / π × E.
    // GI-1.2's estimate of the same term is a cosine-weighted ray's radiance × BRDF / pdf.
    // Weighted by the source's coverage: outside its volumes E is an edge value, not 0.
    let irradiance = giIrradianceAt(surface.position, surface.normal, worldCacheProbeRayDirection(probe_ray), vec2f(0.0));
    let coverage = saturate(giCoverageAt(surface.position, surface.normal));
    let radiance = GIDenoiser_RemoveNaNs(surface.albedo * HASHGRIDCACHE_BOUNCE_SOURCE_INV_PI * coverage * max(irradiance, vec3f(0.0)));

    // One sample per entry into the cell's indirect scratch, which UpdateTiles blends this frame.
    let quantized_radiance = HashGridCache_QuantizeRadiance(radiance);
    let scratch = HASHGRIDCACHE_UPDATECELLVALUEINDIRECTBUFFER + 4u * cell_index;
    if (dot(radiance, radiance) > 0.0) {
        atomicAdd(&g_HashGridCache_TileWords[scratch + 0u], quantized_radiance.x);
        atomicAdd(&g_HashGridCache_TileWords[scratch + 1u], quantized_radiance.y);
        atomicAdd(&g_HashGridCache_TileWords[scratch + 2u], quantized_radiance.z);
    }
    atomicAdd(&g_HashGridCache_TileWords[scratch + 3u], quantized_radiance.w);

    // A probe ray shorter than its cell reads no cell (register D38): its indirect light is added
    // to it here, unweighted (every entry samples, so there is no discard to make up for).
    if (HashGridCache_RayListWord(g_HashGridCache_RayLists.first_bounce_bypass_buffer, did) != 0u) {
        ScreenProbes_AccumulateRadiance(query_index, radiance);
    }
}
`;
}
