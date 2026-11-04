/**
 * What a Babylon adapter would do with a `ShadingModule`: one PBR material
 * plugin that
 *
 *   1. declares the module's textures, samplers and vec4f uniforms in the PBR
 *      fragment shader, under names Babylon's WGSL processor understands;
 *   2. pastes the module's parameter-style WGSL;
 *   3. calls `giIrradiance(...)` and adds E/pi * albedo to the PBR diffuse;
 *   4. each draw: wraps the module's current `GPUTexture`s (re-wrapping only
 *      when a texture object changed), copies the vec4f values, samplers follow their texture automatically (Babylon auto-binds <texture>Sampler).
 *
 * Everything here is generic over the module's parameter list. Only the
 * injection points and the handedness flip are Babylon-specific.
 */

import { Constants } from '@babylonjs/core/Engines/constants.js';
import type { WebGPUEngine } from '@babylonjs/core/Engines/webgpuEngine.js';
import type { InternalTexture } from '@babylonjs/core/Materials/Textures/internalTexture.js';
import { BaseTexture } from '@babylonjs/core/Materials/Textures/baseTexture.js';
import { MaterialPluginBase } from '@babylonjs/core/Materials/materialPluginBase.js';
import type { PBRMaterial } from '@babylonjs/core/Materials/PBR/pbrMaterial.js';
import { ShaderLanguage } from '@babylonjs/core/Materials/shaderLanguage.js';
import type { UniformBuffer } from '@babylonjs/core/Materials/uniformBuffer.js';
import type { MaterialDefines } from '@babylonjs/core/Materials/materialDefines.js';
import type { Scene } from '@babylonjs/core/scene.js';
import type { ShadingModule, ShadingParameter } from '../../../../packages/ddgi-core/src/platform/contract/shading-module.ts';

type TextureParameter = Extract<ShadingParameter, { kind: 'texture' }>;
type SamplerParameter = Extract<ShadingParameter, { kind: 'sampler' }>;
type Vec4Parameter = Extract<ShadingParameter, { kind: 'vec4f' }>;

/** Babylon's own sampling-mode and address-mode numbers (Constants.TEXTURE_*). */
const BABYLON_NEAREST_SAMPLING = 1;
const BABYLON_BILINEAR_SAMPLING = 2;
const BABYLON_CLAMP_ADDRESS = 0;
const BABYLON_WRAP_ADDRESS = 1;

/** A GPUTexture we do not own, wrapped so a Babylon material can bind it. */
interface WrappedTexture {
    readonly gpuTexture: GPUTexture;
    readonly internal: InternalTexture;
    readonly texture: BaseTexture;
}

/** Name of the screen-size uniform the adapter adds for `screenUV`. */
const SCREEN_INFO_UNIFORM = 'giAdapterScreenInfo';

export class ShadingModulePlugin extends MaterialPluginBase {
    /** How many times a GPUTexture had to be wrapped. A test reads it to see whether a wrapper was reused. */
    wrapCount = 0;

    private enabled = false;
    private replacesEnvironmentDiffuse = false;
    private readonly wrappedByParameter = new Map<string, WrappedTexture>();

    constructor(
        material: PBRMaterial,
        private readonly shading: ShadingModule,
        private readonly engine: WebGPUEngine,
        private readonly scene: Scene,
    ) {
        // Priority 300 runs after Babylon's own plugins. The flags after the
        // defines: addToPluginList = false, enable = false, resolveIncludes = false.
        // The base constructor would call getCustomCode() and getUniforms() at
        // once, before this class has stored `shading`, so we register ourselves
        // below, after our fields exist.
        super(material, 'ShadingModulePlugin', 300, { GI_SHADING: false, GI_REPLACE_ENV_DIFFUSE: false }, false, false, false);
        // Needed so hardBindForSubMesh runs every draw. bindForSubMesh only runs
        // when the material or effect changed since the previous draw, which is
        // not often enough for values that change every frame.
        this.registerForExtraEvents = true;
        (this._pluginManager as unknown as { _addPlugin(plugin: MaterialPluginBase): boolean })._addPlugin(this);
        this._enable(true);
    }

    get isEnabled(): boolean {
        return this.enabled;
    }

    set isEnabled(value: boolean) {
        if (this.enabled === value) return;
        this.enabled = value;
        // Toggling changes which code is compiled, so Babylon must rebuild the effect.
        this.markAllDefinesAsDirty();
    }

    /**
     * When true, Babylon's environment diffuse (finalIrradiance) is zeroed and
     * the module's irradiance replaces it. Environment specular is untouched.
     */
    get replaceEnvironmentDiffuse(): boolean {
        return this.replacesEnvironmentDiffuse;
    }

    set replaceEnvironmentDiffuse(value: boolean) {
        if (this.replacesEnvironmentDiffuse === value) return;
        this.replacesEnvironmentDiffuse = value;
        this.markAllDefinesAsDirty();
    }

    override getClassName(): string {
        return 'ShadingModulePlugin';
    }

    override isCompatible(shaderLanguage: ShaderLanguage): boolean {
        return shaderLanguage === ShaderLanguage.WGSL;
    }

    override prepareDefines(defines: MaterialDefines): void {
        defines['GI_SHADING'] = this.enabled;
        defines['GI_REPLACE_ENV_DIFFUSE'] = this.replacesEnvironmentDiffuse;
    }

    override isReadyForSubMesh(): boolean {
        this.syncTextures();
        return true;
    }

    override getSamplers(samplers: string[]): void {
        for (const parameter of this.shading.parameters) {
            if (parameter.kind === 'texture') samplers.push(this.textureGlobalName(parameter));
        }
    }

    override getUniforms(): { ubo: Array<{ name: string; size: number; type: string }> } {
        const ubo = this.shading.parameters
            .filter((parameter): parameter is Vec4Parameter => parameter.kind === 'vec4f')
            .map((parameter) => ({ name: this.uniformName(parameter), size: 4, type: 'vec4' }));
        ubo.push({ name: SCREEN_INFO_UNIFORM, size: 4, type: 'vec4' });
        return { ubo };
    }

    override hardBindForSubMesh(uniformBuffer: UniformBuffer): void {
        if (!this.enabled) return;
        this.syncTextures();
        for (const parameter of this.shading.parameters) {
            switch (parameter.kind) {
                case 'texture': {
                    const wrapped = this.wrappedByParameter.get(parameter.name);
                    if (wrapped) uniformBuffer.setTexture(this.textureGlobalName(parameter), wrapped.texture);
                    break;
                }
                case 'sampler':
                    // Nothing to do: Babylon binds `<texture>Sampler` together with the texture, using the wrapper's sampling state.
                    break;
                case 'vec4f': {
                    const value = parameter.value();
                    uniformBuffer.updateFloat4(this.uniformName(parameter), value[0]!, value[1]!, value[2]!, value[3]!);
                    break;
                }
            }
        }
        uniformBuffer.updateFloat4(
            SCREEN_INFO_UNIFORM,
            1 / this.engine.getRenderWidth(),
            1 / this.engine.getRenderHeight(),
            0,
            0,
        );
    }

    override getCustomCode(shaderType: string, shaderLanguage?: ShaderLanguage): Record<string, string> | null {
        if (shaderType !== 'fragment' || shaderLanguage !== ShaderLanguage.WGSL) return null;
        return {
            // After Babylon's includes, before main(): module-scope declarations and the core's functions.
            CUSTOM_FRAGMENT_DEFINITIONS: `#ifdef GI_SHADING\n${this.declarationsWgsl()}\n${this.shading.code}\n#endif`,
            // Inside main(), after Babylon computed finalDiffuse and before it sums the final colour.
            CUSTOM_FRAGMENT_BEFORE_FINALCOLORCOMPOSITION: `#ifdef GI_SHADING\n${this.callSiteWgsl()}\n#endif`,
        };
    }

    override getActiveTextures(activeTextures: BaseTexture[]): void {
        for (const wrapped of this.wrappedByParameter.values()) activeTextures.push(wrapped.texture);
    }

    override hasTexture(texture: BaseTexture): boolean {
        return [...this.wrappedByParameter.values()].some((wrapped) => wrapped.texture === texture);
    }

    override dispose(): void {
        for (const wrapped of this.wrappedByParameter.values()) this.releaseWrapper(wrapped);
        this.wrappedByParameter.clear();
    }

    // ----- WGSL generation -------------------------------------------------

    /** Global (module-scope) names. Textures and samplers get distinct, suffix-correct names; see assertNamesAreSafeForBabylon. */
    private textureGlobalName(parameter: TextureParameter): string {
        return `babylon_${parameter.name}`;
    }

    private samplerGlobalName(parameter: SamplerParameter): string {
        // `<texture>Sampler` next to the texture's own global name: Babylon's
        // WGSL processor auto-binds it to that texture
        // (webgpuShaderProcessorsWGSL.pure.js:510-520, webgpuEngine.pure.js:2118-2125).
        return `babylon_${parameter.texture}Sampler`;
    }

    private uniformName(parameter: Vec4Parameter): string {
        return `babylon_${parameter.name}`;
    }

    private declarationsWgsl(): string {
        const lines: string[] = [];
        for (const parameter of this.shading.parameters) {
            if (parameter.kind === 'texture') {
                // Babylon builds the bind-group layout from this exact text
                // (webgpuShaderProcessorsWGSL.pure.js:35, 145): the view
                // dimension comes from the texture_* name.
                lines.push(`var ${this.textureGlobalName(parameter)}: ${wgslTextureType(parameter)};`);
            } else if (parameter.kind === 'sampler') {
                lines.push(`var ${this.samplerGlobalName(parameter)}: sampler;`);
            }
        }
        return lines.join('\n');
    }

    private callSiteWgsl(): string {
        const argumentsInOrder = this.shading.parameters.map((parameter) => {
            switch (parameter.kind) {
                case 'texture':
                    return this.textureGlobalName(parameter);
                case 'sampler':
                    return this.samplerGlobalName(parameter);
                case 'vec4f':
                    return `uniforms.${this.uniformName(parameter)}`;
            }
        });
        return /* wgsl */ `
{
    // Babylon scenes are left-handed by default; the platform is right-handed.
    // Mirror z on the way in (position, normal, view) so the core sees its own space.
    let giPosition = vec3f(fragmentInputs.vPositionW.x, fragmentInputs.vPositionW.y, -fragmentInputs.vPositionW.z);
    let giNormal = vec3f(normalW.x, normalW.y, -normalW.z);
    let giView = vec3f(viewDirectionW.x, viewDirectionW.y, -viewDirectionW.z);
    // Babylon exposes the pixel position but not the viewport size: the plugin adds it as a uniform.
    let giScreenUV = fragmentInputs.position.xy * uniforms.${SCREEN_INFO_UNIFORM}.xy;
    let giIrradianceE = giIrradiance(giPosition, giNormal, giView, giScreenUV, ${argumentsInOrder.join(', ')});

    // Babylon's environment irradiance already includes 1/pi ("L = E * rho / pi",
    // sphericalPolynomial.pure.js:187-199), then goes through the same three
    // factors below (pbrBlockFinalLitComponents.js): energy left over after
    // specular, albedo, and ambient occlusion. Treat GI the same way.
#ifdef GI_REPLACE_ENV_DIFFUSE
#ifdef REFLECTION
    // finalIrradiance only exists when the material has an environment texture.
    // Zeroing it removes environment diffuse; finalRadianceScaled (specular) is separate.
    finalIrradiance = vec3f(0.0);
#endif
#endif
    let giDiffuseShare = clamp(vec3f(1.0) - vec3f(baseSpecularEnvironmentReflectance), vec3f(0.0), vec3f(1.0));
    finalDiffuse += giIrradianceE * (1.0 / PI) * giDiffuseShare * surfaceAlbedo * aoOut.ambientOcclusionColor;
}`;
    }

    // ----- Resource handling ----------------------------------------------

    /** Re-reads each `texture()` and re-wraps only the ones whose GPUTexture object changed. */
    private syncTextures(): void {
        for (const parameter of this.shading.parameters) {
            if (parameter.kind !== 'texture') continue;
            const current = parameter.texture();
            const existing = this.wrappedByParameter.get(parameter.name);
            if (existing && existing.gpuTexture === current) continue;
            if (existing) this.releaseWrapper(existing);
            this.wrappedByParameter.set(parameter.name, this.wrap(parameter, current));
            this.wrapCount++;
        }
    }

    private wrap(parameter: TextureParameter, gpuTexture: GPUTexture): WrappedTexture {
        const internal = this.engine.wrapWebGPUTexture(gpuTexture);
        // wrapWebGPUTexture only records width/height/format. Everything that
        // decides the view and the bind-group layout is ours to set.
        internal.is2DArray = parameter.dimension === '2d-array';
        internal.is3D = parameter.dimension === '3d';
        internal.depth = gpuTexture.depthOrArrayLayers;
        internal.generateMipMaps = false;
        internal.format = Constants.TEXTUREFORMAT_RGBA;
        // type === FLOAT makes Babylon pick an "unfilterable-float" layout when
        // the device lacks float32-filterable (webgpuEngine.pure.js:2998-3002).
        internal.type = gpuTexture.format === 'rgba32float' ? Constants.TEXTURETYPE_FLOAT : Constants.TEXTURETYPE_HALF_FLOAT;
        // updateWrappedWebGPUTexture is documented for device restore, but it is
        // also what creates the GPUTextureView (the wrap call leaves it null).
        this.engine.updateWrappedWebGPUTexture(internal, gpuTexture);
        // Babylon takes the sampler state from the wrapper, not from a separate sampler object.
        const sampler = this.shading.parameters.find(
            (candidate): candidate is SamplerParameter => candidate.kind === 'sampler' && candidate.texture === parameter.name,
        );
        internal.samplingMode = sampler?.filtering === 'linear' ? BABYLON_BILINEAR_SAMPLING : BABYLON_NEAREST_SAMPLING;
        const texture = new BaseTexture(this.scene, internal);
        const address = sampler?.addressMode === 'repeat' ? BABYLON_WRAP_ADDRESS : BABYLON_CLAMP_ADDRESS;
        texture.wrapU = address;
        texture.wrapV = address;
        texture.wrapR = address;
        texture.name = `shading:${parameter.name}`;
        return { gpuTexture, internal, texture };
    }

    private releaseWrapper(wrapped: WrappedTexture): void {
        // Disposing an InternalTexture releases its hardware texture, and
        // WebGPUHardwareTexture.release() calls GPUTexture.destroy()
        // (webgpuHardwareTexture.js:88-89). reset() detaches our texture first.
        (wrapped.internal._hardwareTexture as unknown as { reset(): void }).reset();
        wrapped.texture.dispose();
    }
}

function wgslTextureType(parameter: TextureParameter): string {
    const dimension = { '2d': 'texture_2d', '2d-array': 'texture_2d_array', '3d': 'texture_3d' }[parameter.dimension];
    const component = parameter.sampleType === 'uint' ? 'u32' : 'f32';
    return `${dimension}<${component}>`;
}
