/**
 * The test scenes for the hello example: a manifest's procedural parts built
 * by the repo's engine-free builders (`scenes/src`), described to the GI
 * platform through its scene API, and kept as plain data for the renderer.
 *
 * `describeScene` is the part an engine adapter writes (docs/guides/adapter.md):
 * it hands the platform everything GI's rays need to see, in the contract's
 * terms. Textures (as `PixelData`), meshes with both uv sets and their
 * material groups, materials with their maps, alpha and metalness, and
 * instances with one material per slot.
 *
 * Only type imports from the library here, so GPU test pages can reuse this
 * file with the core loaded straight from source.
 */

import type {
    ColorRGB,
    EnvironmentDescriptor,
    LightDescriptor,
    MaterialDescriptor,
    MaterialHandle,
    MeshHandle,
    SceneApi,
    TextureHandle,
    UvMapping,
} from '@pmndrs/gi/core';
import { buildProceduralParts, mergeScenes } from '../../../../scenes/src/build-scene.ts';
import { isProceduralOnly, type CameraBookmark, type ManifestLight, type SceneManifest } from '../../../../scenes/src/manifest.ts';
import { textureMeanLinear, uvMappingMatrix } from '../../../../scenes/src/procedural/textures.ts';
import type { ProceduralScene, SceneMaterialData, SceneTextureReference } from '../../../../scenes/src/scene-data.ts';
import { platformSkyEnvironment } from '../../../../scenes/src/sky/sky-image.ts';

export type { CameraBookmark, SceneManifest, ProceduralScene };

const manifests = import.meta.glob('../../../../scenes/manifests/*.json', { eager: true, import: 'default' }) as Record<string, SceneManifest>;

/** The point light the hello renderer lights directly (with a shadow map). */
export interface RendererPointLight {
    readonly position: readonly [number, number, number];
    /** Linear colour times intensity (candela). */
    readonly intensity: readonly [number, number, number];
    /** Metres; 0 means no range window (three's `distance = 0`). */
    readonly range: number;
}

/**
 * Why a scene can't run here, or null if it can. The hello renderer draws
 * procedural geometry with one flat colour per material (see
 * `RendererMaterial`) and lights at most one point light itself
 * (directional and spot lights would need their own shadow maps; the
 * viewer has them). GI sees the scenes' textures either way.
 */
export function unsupportedReason(manifest: SceneManifest): string | null {
    if (!isProceduralOnly(manifest)) return 'it loads model files; the hello example only builds procedural scenes';
    const pointLights = manifest.lights.filter((light) => light.type === 'point').length;
    if (pointLights !== manifest.lights.length) return 'it has directional or spot lights; the hello renderer lights point lights only';
    if (pointLights > 1) return 'it has more than one point light; the hello renderer shadows one';
    return null;
}

export function loadManifest(id: string): SceneManifest {
    const manifest = Object.entries(manifests).find(([path]) => path.endsWith(`/${id}.json`))?.[1];
    if (!manifest) throw new Error(`No scene called "${id}". Try one of: ${supportedSceneIds().join(', ')}.`);
    const reason = unsupportedReason(manifest);
    if (reason) throw new Error(`The hello example can't show "${id}": ${reason}. Try one of: ${supportedSceneIds().join(', ')}.`);
    return manifest;
}

export function supportedSceneIds(): string[] {
    return Object.values(manifests)
        .filter((manifest) => unsupportedReason(manifest) === null)
        .map((manifest) => manifest.id)
        .sort();
}

export interface SceneOptions {
    /**
     * Light point lights with no range window (`range: 0`), as Cycles does.
     * The engine-free test view uses this for its reference comparisons.
     */
    readonly ignoreLightRange?: boolean;
}

function toPlatformLight(light: ManifestLight, options: SceneOptions): LightDescriptor {
    if (light.type !== 'point') throw new Error(`unsupported light type ${light.type}`);
    return {
        type: 'point',
        color: light.color,
        intensity: light.intensity,
        castsShadows: light.castsShadows ?? true,
        label: light.label,
        position: light.position,
        range: options.ignoreLightRange ? 0 : light.range,
    };
}

function toPlatformEnvironment(manifest: SceneManifest): EnvironmentDescriptor {
    const environment = manifest.environment;
    if (environment.type === 'none') return { type: 'none' };
    if (environment.type === 'color') return { type: 'color', color: environment.color, intensity: environment.intensity };
    // The calibrated sky, the same file the viewer and Cycles read.
    return platformSkyEnvironment(manifest);
}

/**
 * What the hello renderer draws for a material: one flat colour. It doesn't
 * sample textures (that would be another bind group and a uv buffer), so a
 * textured material is drawn with its texture's mean colour. GI's rays read
 * the textures themselves, texel by texel, through the platform.
 */
export interface RendererMaterial {
    /** Linear albedo × the albedo texture's mean × (1 − metalness), as GI's albedo averages out. */
    readonly albedo: ColorRGB;
    /** Linear emitted radiance. */
    readonly emission: ColorRGB;
}

export interface DescribedScene {
    readonly manifest: SceneManifest;
    /** Meshes, materials and instances, for the renderer. */
    readonly data: ProceduralScene;
    /** The renderer's flat colour for each of `data.materials`, in the same order. */
    readonly rendererMaterials: readonly RendererMaterial[];
    readonly pointLight: RendererPointLight | null;
    /**
     * Irradiance a renderer adds where GI coverage is low: a uniform colour
     * environment of radiance L gives E = π L. Zero for no environment, and
     * for a sky (the hello renderer has no sky lighting of its own).
     */
    readonly fallbackIrradiance: readonly [number, number, number];
}

/**
 * Where a texture is sampled, in the contract's terms: the uv set, a 3×2
 * matrix and the wrap per axis. The scene data says it as three.js and glTF
 * do (offset, repeat, rotation), which `uvMappingMatrix` turns into the
 * matrix three's `Texture.matrix` would hold. Row 0 of the scene's pixels is
 * v = 0, the contract's convention too, so no flip is needed.
 */
function toPlatformUv(reference: SceneTextureReference): UvMapping {
    const uv = reference.uv ?? {};
    return { channel: uv.channel ?? 0, matrix: uvMappingMatrix(uv), wrapU: uv.wrapU ?? 'repeat', wrapV: uv.wrapV ?? 'repeat' };
}

/**
 * A scene material as the platform's material model. Every idea GI uses is
 * passed: albedo and its texture, emission, metalness and its map, alpha
 * (opacity, cutoff, mode), whether it blocks light, and double-sidedness.
 */
function toPlatformMaterial(material: SceneMaterialData, textures: readonly TextureHandle[]): MaterialDescriptor {
    // The scene data follows glTF: a metalness map with no factor means factor 1 (glTF's metallicFactor default).
    const metalness = material.metalness ?? (material.metalnessMap ? 1 : 0);
    return {
        label: material.name,
        albedo: material.albedo,
        emissive: material.emissive,
        doubleSided: material.doubleSided,
        metalness,
        ...(material.albedoMap ? { albedoTexture: textures[material.albedoMap.texture], albedoUv: toPlatformUv(material.albedoMap) } : {}),
        ...(material.metalnessMap
            ? { metalnessTexture: { texture: textures[material.metalnessMap.texture], channel: material.metalnessMap.channel, uv: toPlatformUv(material.metalnessMap) } }
            : {}),
        ...(material.opacity !== undefined ? { opacity: material.opacity } : {}),
        ...(material.alphaMode !== undefined ? { alphaMode: material.alphaMode } : {}),
        ...(material.alphaCutoff !== undefined ? { alphaCutoff: material.alphaCutoff } : {}),
        // The contract has no transmission: glass is a surface that lets light through, as the three adapter maps it.
        blocksLight: material.blocksLight ?? !((material.transmission ?? 0) > 0),
    };
}

/** The renderer's flat colour for a material (see `RendererMaterial`). */
function rendererMaterial(material: SceneMaterialData, data: ProceduralScene): RendererMaterial {
    const textureMean = (reference: SceneTextureReference | undefined): readonly number[] =>
        reference ? textureMeanLinear(data.textures![reference.texture]) : [1, 1, 1, 1];
    const albedoMean = textureMean(material.albedoMap);
    const metalnessMap = material.metalnessMap;
    const metalnessMean = metalnessMap ? textureMean(metalnessMap)['rgba'.indexOf(metalnessMap.channel)] : 1;
    const diffuseShare = 1 - (material.metalness ?? (metalnessMap ? 1 : 0)) * metalnessMean;
    return {
        albedo: [material.albedo[0] * albedoMean[0] * diffuseShare, material.albedo[1] * albedoMean[1] * diffuseShare, material.albedo[2] * albedoMean[2] * diffuseShare],
        emission: material.emissive,
    };
}

/** Describes a manifest scene to the platform: textures, meshes, materials, instances, lights and environment. */
export function describeScene(scene: SceneApi, manifest: SceneManifest, options: SceneOptions = {}): DescribedScene {
    const data = mergeScenes(buildProceduralParts(manifest));

    // Textures first, because materials refer to them by handle. The platform
    // copies each into its own texture store in the next encode, so the
    // source may be freed after that.
    const textures: TextureHandle[] = (data.textures ?? []).map((texture) =>
        scene.createTexture({ source: { data: texture.data, width: texture.width, height: texture.height }, srgb: texture.srgb, label: texture.name }),
    );

    // Meshes: positions and indices, smooth normals, both uv sets, and the
    // groups that say which material slot each run of triangles uses.
    const meshes: MeshHandle[] = data.meshes.map((mesh) =>
        scene.createMesh({
            positions: mesh.positions,
            indices: mesh.indices,
            normals: mesh.normals,
            uvs: mesh.uvs,
            ...(mesh.uvs1 ? { uvs1: mesh.uvs1 } : {}),
            ...(mesh.groups ? { groups: mesh.groups.map((group) => ({ start: group.start, count: group.count, materialSlot: group.materialSlot })) } : {}),
            label: mesh.name,
        }),
    );

    const materials: MaterialHandle[] = data.materials.map((material) => scene.createMaterial(toPlatformMaterial(material, textures)));

    // Instances: one material per slot the mesh's groups use (a wall with a
    // glass pane has two). A one-material mesh has just slot 0.
    for (const instance of data.instances) {
        const slots = instance.materials ?? [instance.material];
        scene.createInstance({ mesh: meshes[instance.mesh], materials: slots.map((slot) => materials[slot]), transform: instance.transform, label: instance.name });
    }

    for (const light of manifest.lights) scene.createLight(toPlatformLight(light, options));
    scene.setEnvironment(toPlatformEnvironment(manifest));

    const light = manifest.lights.find((candidate) => candidate.type === 'point');
    const pointLight: RendererPointLight | null =
        light && light.type === 'point'
            ? {
                  position: light.position,
                  intensity: [light.color[0] * light.intensity, light.color[1] * light.intensity, light.color[2] * light.intensity],
                  range: options.ignoreLightRange ? 0 : light.range,
              }
            : null;
    const environment = manifest.environment;
    // A uniform environment of radiance L gives irradiance E = pi L on any surface.
    const environmentRadianceScale = environment.type === 'color' ? (environment.intensity ?? 1) : 0;
    const fallbackIrradiance: [number, number, number] =
        environment.type === 'color'
            ? [Math.PI * environment.color[0] * environmentRadianceScale, Math.PI * environment.color[1] * environmentRadianceScale, Math.PI * environment.color[2] * environmentRadianceScale]
            : [0, 0, 0];
    const rendererMaterials = data.materials.map((material) => rendererMaterial(material, data));
    return { manifest, data, rendererMaterials, pointLight, fallbackIrradiance };
}
