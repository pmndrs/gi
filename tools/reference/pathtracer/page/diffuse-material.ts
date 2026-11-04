/**
 * Turns a scene's materials into pure Lambert for the path-traced reference,
 * keeping everything that decides WHAT the surface looks like and dropping only
 * the non-diffuse lobes. Kept separate from main.ts so it can be unit-tested
 * without a GPU.
 */

import { Mesh, MeshPhysicalMaterial, type Color, type Material, type Object3D, type Texture, type Vector2 } from 'three/webgpu';

/** The fields read from a source material; any three material may lack some of them. */
interface SourceMaterialFields {
    color?: Color;
    map?: Texture | null;
    emissive?: Color;
    emissiveIntensity?: number;
    emissiveMap?: Texture | null;
    alphaMap?: Texture | null;
    normalMap?: Texture | null;
    normalScale?: Vector2;
}

/**
 * A MeshPhysicalMaterial with roughness 1, metalness 0 and specularIntensity 0:
 * three-gpu-pathtracer's material model adds a 4 percent specular lobe unless
 * specularIntensity is 0, and a metal (or a metalnessMap) would remove the
 * diffuse entirely, so no metalness or roughness map is carried over.
 *
 * Carried over, because Cycles keeps the same things (blender_scene.py,
 * NEUTRAL_PRINCIPLED_INPUTS) and the two references must match on glTF scenes:
 *   colour and its texture (`map`), emission (`emissive`, `emissiveIntensity`,
 *   `emissiveMap`), alpha (`alphaMap`, `alphaTest`, `transparent`, `opacity`),
 *   the normal map, vertex colours, and `side`.
 */
export function diffuseOnlyMaterial(source: Material): MeshPhysicalMaterial {
    const fields = source as Material & SourceMaterialFields;
    const material = new MeshPhysicalMaterial({
        roughness: 1,
        metalness: 0,
        specularIntensity: 0,
        side: source.side,
        alphaTest: source.alphaTest,
        transparent: source.transparent,
        opacity: source.opacity,
        vertexColors: source.vertexColors,
        map: fields.map ?? null,
        emissiveIntensity: fields.emissiveIntensity ?? 1,
        emissiveMap: fields.emissiveMap ?? null,
        alphaMap: fields.alphaMap ?? null,
        normalMap: fields.normalMap ?? null,
    });
    if (fields.color) material.color.copy(fields.color);
    if (fields.emissive) material.emissive.copy(fields.emissive);
    if (fields.normalScale) material.normalScale.copy(fields.normalScale);
    material.name = source.name;
    return material;
}

/**
 * Replaces every mesh's material (or each one of a multi-material mesh) with
 * its diffuse-only version. Returns the materials that emit, so the page can
 * switch emission off for the lights-only direct pass.
 */
export function makeDiffuseOnly(root: Object3D): MeshPhysicalMaterial[] {
    const emissive: MeshPhysicalMaterial[] = [];
    // A material shared by several meshes is converted once and stays shared.
    const converted = new Map<Material, MeshPhysicalMaterial>();
    const convert = (source: Material): MeshPhysicalMaterial => {
        let material = converted.get(source);
        if (!material) {
            material = diffuseOnlyMaterial(source);
            converted.set(source, material);
            const emits = material.emissive.r + material.emissive.g + material.emissive.b > 0 && material.emissiveIntensity > 0;
            if (emits) emissive.push(material);
        }
        return material;
    };
    root.traverse((object) => {
        if (!(object instanceof Mesh)) return;
        const mesh = object as Mesh;
        mesh.material = Array.isArray(mesh.material) ? mesh.material.map(convert) : convert(mesh.material);
    });
    return emissive;
}
