/**
 * The shading module: how a renderer reads GI while it shades pixels.
 *
 * Every technique publishes the same shape, so an adapter injects it into its
 * engine once and switching techniques changes the module, not the adapter.
 *
 * Two engine facts shape it:
 *
 * 1. **Only textures, samplers and small uniforms.** three.js r187 can bind a
 *    foreign `GPUTexture` (`ExternalTexture`) but not a foreign buffer, and
 *    compatibility-mode devices may allow no storage buffers in fragment
 *    shaders. So everything a fragment reads is a texture, plus a few `vec4f`
 *    uniforms the adapter mirrors from CPU values each frame. Lookup tables
 *    that would be buffers elsewhere (the list of volumes, say) are stored in
 *    small `rgba32float` textures.
 * 2. **Resources are passed as function parameters, not module globals.**
 *    three's `wgslFn` and Babylon's material plugins each declare resources
 *    their own way, so the core can't write `@group/@binding` lines that suit
 *    both. The core's WGSL takes every resource as a parameter, in the order of
 *    {@link ShadingModule.parameters}. An adapter declares the resources its own
 *    way and calls through. For hosts that bind globals (a plain WebGPU
 *    renderer), {@link ShadingModule.globalsCode} generates the declarations
 *    and parameterless wrappers.
 *
 * The entry points (each takes the resource parameters after the ones shown):
 *
 * ```wgsl
 * // Diffuse irradiance E arriving at a surface (not radiance: the renderer
 * // applies albedo / π). `screenUV` is in 0..1 and may be ignored; it lets
 * // screen-space techniques read their resolved output.
 * fn giIrradiance(position: vec3f, normal: vec3f, viewDirection: vec3f, screenUV: vec2f, ...) -> vec3f
 *
 * // How much of the answer the technique is sure of, 0..1. The renderer fades
 * // to its environment lighting where coverage is low (volume edges).
 * fn giCoverage(position: vec3f, normal: vec3f, ...) -> f32
 *
 * // Optional: radiance arriving from a direction, blurred by roughness.
 * // DDGI provides a rough-only fallback.
 * fn giRadiance(position: vec3f, direction: vec3f, roughness: f32, ...) -> vec3f
 *
 * // A debug color for the given debug view id (see TechniqueDebug).
 * // `screenUV` as for giIrradiance (RC2.2): screen-space techniques draw
 * // their views by it; others may ignore it.
 * fn giDebugColor(position: vec3f, normal: vec3f, view: u32, screenUV: vec2f, ...) -> vec3f
 * ```
 */

/** One resource the module's functions take, in order. */
export type ShadingParameter =
    | {
          readonly kind: 'texture';
          readonly name: string;
          readonly dimension: '2d' | '2d-array' | '3d';
          /** `float` is filterable; `unfilterable-float` for rgba32float tables; `uint` for integer data. */
          readonly sampleType: 'float' | 'unfilterable-float' | 'uint';
          /** The current texture. Changes only when the technique reallocates (rare: textures are fixed-capacity). */
          readonly texture: () => GPUTexture;
      }
    | {
          /**
           * A sampler always belongs to one filterable texture parameter and is
           * named `<texture>Sampler`. Engines tie samplers to textures: three's
           * `wgslFn` feeds a sampler parameter from a texture node and uses that
           * texture's filter and wrap settings, and Babylon binds a sampler named
           * `<texture>Sampler` to that texture automatically (T0.8, T0.9).
           */
          readonly kind: 'sampler';
          readonly name: `${string}Sampler`;
          /** Name of the `texture` parameter this sampler belongs to. */
          readonly texture: string;
          readonly filtering: 'linear' | 'nearest';
          readonly addressMode: 'clamp-to-edge' | 'repeat';
      }
    | {
          readonly kind: 'vec4f';
          readonly name: string;
          /**
           * Current value. The adapter copies it into its own uniform each
           * frame (three: a `renderGroup` uniform). Returns the same array,
           * updated in place, so reading it allocates nothing.
           */
          readonly value: () => Float32Array;
      };

export interface ShadingModule {
    /** The technique that publishes it, e.g. `ddgi`. */
    readonly techniqueId: string;
    /**
     * WGSL defining the entry points above, taking resources as parameters.
     * Free of `@group/@binding` declarations so engines can wrap it. Stable
     * until `version` changes.
     */
    readonly code: string;
    /** The resources each entry point takes after its own arguments, in order. */
    readonly parameters: readonly ShadingParameter[];
    /** Which optional entry points exist. */
    readonly provides: {
        readonly radiance: boolean;
    };
    /**
     * Declarations plus parameterless wrappers (`giIrradianceAt(p, n, v, uv)`,
     * `giDebugColorAt(p, n, view, uv)` and so on) for hosts that bind globals, written at the given group with
     * bindings from 0. Uniforms become one uniform buffer the host fills from
     * {@link writeUniforms}.
     *
     * Self-contained: it starts with {@link code}, so a host includes this
     * string alone. Bindings: every `texture` and `sampler` parameter in
     * parameter order from 0, then the uniform buffer (the `vec4f`
     * parameters in order) at the next binding.
     */
    globalsCode(group: number): string;
    /** Writes all `vec4f` parameters, in order, into `target` for the globals path. */
    writeUniforms(target: Float32Array): void;
    /**
     * Increments when `code` or `parameters` change shape, meaning the adapter
     * must rebuild its shader node or plugin. Texture reallocation does not
     * count: re-read `texture()` each frame instead.
     */
    readonly version: number;
}
