/**
 * Turns engine-free scene data into three.js objects, for the check page (and
 * later the viewer). The only file in `scenes/src` that builds three objects
 * from procedural data; the data itself never depends on three.
 *
 * Light units pass straight through: the scene data uses three.js physical
 * units (lux for directional, candela for point and spot).
 */

import {
    BufferAttribute,
    BufferGeometry,
    Color,
    DirectionalLight,
    DoubleSide,
    FrontSide,
    Group,
    InstancedMesh,
    Matrix4,
    Mesh,
    MeshStandardMaterial,
    PointLight,
    RectAreaLight,
    SpotLight,
    Vector3,
    type Light,
    type Object3D,
} from 'three';
import type { LightSetFile, LightSetPart } from './manifest';
import type { ProceduralScene, SceneLightData, SceneMaterialData, SceneMeshData, Vector3Tuple } from './scene-data';

function linearColor(rgb: readonly [number, number, number]): Color {
    // setRGB with no colour space argument takes the values as the working
    // (linear) space, which is what the scene data uses.
    return new Color().setRGB(rgb[0], rgb[1], rgb[2]);
}

function meshToGeometry(mesh: SceneMeshData): BufferGeometry {
    const geometry = new BufferGeometry();
    geometry.setAttribute('position', new BufferAttribute(mesh.positions, 3));
    geometry.setAttribute('normal', new BufferAttribute(mesh.normals, 3));
    geometry.setAttribute('uv', new BufferAttribute(mesh.uvs, 2));
    geometry.setIndex(new BufferAttribute(mesh.indices, 1));
    geometry.name = mesh.name;
    return geometry;
}

function materialToThree(material: SceneMaterialData): MeshStandardMaterial {
    const peak = Math.max(...material.emissive);
    return new MeshStandardMaterial({
        name: material.name,
        color: linearColor(material.albedo),
        // Fully rough and non-metallic: the platform only models diffuse albedo.
        roughness: 1,
        metalness: 0,
        // Emission above 1 is kept as colour times intensity, as three.js expects.
        emissive: peak > 0 ? new Color().setRGB(material.emissive[0] / peak, material.emissive[1] / peak, material.emissive[2] / peak) : new Color(0, 0, 0),
        emissiveIntensity: peak > 0 ? peak : 1,
        side: material.doubleSided ? DoubleSide : FrontSide,
    });
}

/** One Mesh per instance, sharing geometries and materials. */
export function proceduralSceneToThree(scene: ProceduralScene, name: string): Group {
    const root = new Group();
    root.name = name;
    const geometries = scene.meshes.map(meshToGeometry);
    const materials = scene.materials.map(materialToThree);

    for (const instance of scene.instances) {
        const mesh = new Mesh(geometries[instance.mesh], materials[instance.material]);
        mesh.name = instance.name;
        mesh.matrixAutoUpdate = false;
        mesh.matrix.fromArray(instance.transform as number[]);
        mesh.matrixWorldNeedsUpdate = true;
        // Shadows: walls both cast and receive.
        mesh.castShadow = true;
        mesh.receiveShadow = true;
        root.add(mesh);
    }
    return root;
}

export interface LightFitting {
    /** Centre of the geometry; directional lights are placed relative to it. */
    center: Vector3Tuple;
    /** Radius of a sphere around the geometry. */
    radius: number;
}

/**
 * How far a point or spot light's shadow camera must reach: its range, or for
 * a light with no range window (range 0) the far side of the geometry.
 */
function shadowFar(position: Vector3Tuple, range: number, fitting: LightFitting): number {
    if (range > 0) return range;
    const toCenter = Math.hypot(position[0] - fitting.center[0], position[1] - fitting.center[1], position[2] - fitting.center[2]);
    return toCenter + fitting.radius;
}

/**
 * Creates a three.js light. Directional lights get a shadow camera that fits
 * `fitting`; the others reach to their range. A range of 0 maps to three's
 * `distance = 0`: no range window, plain inverse-square falloff.
 */
export function lightToThree(light: SceneLightData, fitting: LightFitting): Light {
    const color = linearColor(light.color);
    const castShadow = light.castsShadows ?? true;

    switch (light.type) {
        case 'directional': {
            const threeLight = new DirectionalLight(color, light.intensity);
            const toLight = new Vector3(...light.direction).normalize().multiplyScalar(-1);
            const center = new Vector3(...fitting.center);
            threeLight.position.copy(center).addScaledVector(toLight, fitting.radius * 1.5);
            threeLight.target.position.copy(center);
            threeLight.target.updateMatrixWorld();
            threeLight.castShadow = castShadow;
            const shadowCamera = threeLight.shadow.camera;
            shadowCamera.left = -fitting.radius;
            shadowCamera.right = fitting.radius;
            shadowCamera.top = fitting.radius;
            shadowCamera.bottom = -fitting.radius;
            shadowCamera.near = fitting.radius * 0.1;
            shadowCamera.far = fitting.radius * 3.5;
            threeLight.shadow.mapSize.set(4096, 4096);
            threeLight.shadow.bias = -0.0005;
            threeLight.shadow.normalBias = 0.03;
            threeLight.name = light.label ?? 'directional';
            return threeLight;
        }
        case 'point': {
            const threeLight = new PointLight(color, light.intensity, light.range, 2);
            threeLight.position.set(...light.position);
            threeLight.castShadow = castShadow;
            threeLight.shadow.camera.near = 0.05;
            threeLight.shadow.camera.far = shadowFar(light.position, light.range, fitting);
            threeLight.shadow.mapSize.set(2048, 2048);
            // Small biases: a large normal bias is wider than a 3 cm wall and lets light
            // seep round its edges in the picture.
            threeLight.shadow.bias = -0.0003;
            threeLight.shadow.normalBias = 0.002;
            threeLight.name = light.label ?? 'point';
            return threeLight;
        }
        case 'spot': {
            // three.js describes the soft edge as "penumbra", the fraction of the
            // cone over which the light fades: 0 is a hard edge, 1 fades from the axis.
            const penumbra = 1 - light.innerConeAngle / light.outerConeAngle;
            const threeLight = new SpotLight(color, light.intensity, light.range, light.outerConeAngle, penumbra, 2);
            threeLight.position.set(...light.position);
            threeLight.target.position.set(
                light.position[0] + light.direction[0],
                light.position[1] + light.direction[1],
                light.position[2] + light.direction[2],
            );
            threeLight.target.updateMatrixWorld();
            threeLight.castShadow = castShadow;
            threeLight.shadow.camera.near = 0.05;
            threeLight.shadow.camera.far = shadowFar(light.position, light.range, fitting);
            threeLight.shadow.mapSize.set(2048, 2048);
            threeLight.shadow.bias = -0.0005;
            threeLight.shadow.normalBias = 0.02;
            threeLight.name = light.label ?? 'spot';
            return threeLight;
        }
    }
}

/**
 * One RectAreaLight per emissive quad, so surfaces that emit light also light
 * the scene in the direct-light preview. The area light's intensity is a
 * luminance, which is exactly the quad's emitted radiance. Emissive boxes
 * (the furnace room's walls) are not turned into lights: there is no single
 * rectangle to stand for them, and the furnace has no direct light by design.
 * Area lights do not cast shadows in three.js, so the preview is unshadowed.
 */
export function emissiveQuadLights(scene: ProceduralScene): RectAreaLight[] {
    const lights: RectAreaLight[] = [];
    for (const instance of scene.instances) {
        const material = scene.materials[instance.material];
        const radiance = Math.max(...material.emissive);
        if (scene.meshes[instance.mesh].name !== 'unit-quad' || radiance <= 0) continue;
        const m = instance.transform;
        const width = Math.hypot(m[0], m[1], m[2]);
        const height = Math.hypot(m[4], m[5], m[6]);
        const normal = new Vector3(m[8], m[9], m[10]).normalize();
        const color = new Color().setRGB(material.emissive[0] / radiance, material.emissive[1] / radiance, material.emissive[2] / radiance);
        const light = new RectAreaLight(color, radiance, width, height);
        light.name = `${instance.name}-area-light`;
        light.position.set(m[12], m[13], m[14]);
        // RectAreaLight shines along its local -Z once aimed with lookAt.
        light.lookAt(light.position.x + normal.x, light.position.y + normal.y, light.position.z + normal.z);
        lights.push(light);
    }
    return lights;
}

/** Applies a manifest model transform: position, XYZ Euler degrees, uniform scale. */
export function applyModelTransform(
    object: Group,
    transform: { position: Vector3Tuple; rotationDeg: Vector3Tuple; scale: number },
): void {
    const degreesToRadians = Math.PI / 180;
    object.position.set(...transform.position);
    object.rotation.set(
        transform.rotationDeg[0] * degreesToRadians,
        transform.rotationDeg[1] * degreesToRadians,
        transform.rotationDeg[2] * degreesToRadians,
        'XYZ',
    );
    object.scale.setScalar(transform.scale);
    object.updateMatrix();
}

/** What a light set needs to load its model: a glTF loader (three's `GLTFLoader` fits). */
export interface ModelLoader {
    loadAsync(url: string): Promise<{ scene: Object3D }>;
}

/** A `light-set` part as three objects: one InstancedMesh per mesh of the model in `group`, and one point light per copy (not in the group: the caller places lights). */
export interface LoadedLightSet {
    readonly group: Group;
    readonly lights: PointLight[];
    readonly set: LightSetFile;
    /** Copies placed: the scene setting, at most the set's count. */
    readonly count: number;
    /** Triangles in all copies together. */
    readonly triangles: number;
}

/**
 * Loads a `light-set` part (T2.1, `sponza-candles`): the set file, then its
 * model, placed at the first `count` lights of the set's order. Only the one
 * model is loaded, whatever the count.
 *
 * - **The copies are InstancedMeshes**, one per mesh in the model (a candle's
 *   wax, wick and flame), with `userData.giContributes = false`: they don't
 *   block or bounce GI's rays. Each copy surrounds its own light (the flame
 *   sits on it, the wax top 2 mm under it), so in GI its surfaces would be
 *   lit at a few millimetres by inverse-square falloff and turn into
 *   fireflies in every probe ray that hits one, and the flame would shadow its
 *   own light. A candle 2 cm wide changes nothing else GI can see. three
 *   still draws them.
 * - **The lights** have no range (three's distance 0, as the set file says)
 *   and decay 2. three draws them without shadows (`castShadow = false`: no
 *   renderer has shadow maps for hundreds of point lights), while GI traces
 *   their shadows (`userData.giCastsShadows`), so candle light doesn't leak
 *   through Sponza's walls into GI.
 */
export async function loadLightSet(part: LightSetPart, count: number, find: (asset: string, path: string) => Promise<string>, loader: ModelLoader): Promise<LoadedLightSet> {
    const setUrl = await find(part.asset, part.path);
    const response = await fetch(setUrl);
    if (!response.ok) throw new Error(`light set ${part.asset}/${part.path}: HTTP ${response.status}`);
    const set = (await response.json()) as LightSetFile;
    if (!(set.positions.length === set.count * 3) || set.light.type !== 'point') throw new Error(`light set ${part.asset}/${part.path} is not a set of point lights`);
    const folder = part.path.includes('/') ? part.path.slice(0, part.path.lastIndexOf('/') + 1) : '';
    const model = (await loader.loadAsync(await find(part.asset, `${folder}${set.prototype}`))).scene;
    model.updateMatrixWorld(true);
    const placed = Math.max(0, Math.min(count, set.count));

    const group = new Group();
    group.name = `${part.asset}/${part.path} (${placed} of ${set.count})`;
    const place = new Matrix4();
    const copy = new Matrix4();
    let triangles = 0;
    model.traverse((object) => {
        const mesh = object as Mesh;
        if (!mesh.isMesh) return;
        const instanced = new InstancedMesh(mesh.geometry, mesh.material, placed);
        instanced.name = `${mesh.name || 'mesh'} x ${placed}`;
        for (let index = 0; index < placed; index++) {
            place.makeTranslation(set.positions[index * 3], set.positions[index * 3 + 1], set.positions[index * 3 + 2]);
            instanced.setMatrixAt(index, copy.multiplyMatrices(place, mesh.matrixWorld));
        }
        instanced.instanceMatrix.needsUpdate = true;
        instanced.computeBoundingBox();
        instanced.computeBoundingSphere();
        instanced.userData.giContributes = false;
        instanced.castShadow = false;
        instanced.receiveShadow = true;
        const indices = mesh.geometry.index ? mesh.geometry.index.count : mesh.geometry.getAttribute('position').count;
        triangles += (indices / 3) * placed;
        group.add(instanced);
    });

    const lights: PointLight[] = [];
    const color = linearColor(set.light.color);
    for (let index = 0; index < placed; index++) {
        const light = new PointLight(color, set.light.intensity, set.light.range, 2);
        light.position.set(set.positions[index * 3], set.positions[index * 3 + 1], set.positions[index * 3 + 2]);
        light.castShadow = false;
        light.userData.giCastsShadows = true;
        light.name = `${part.countSetting} ${set.packageIndices[index]}`;
        lights.push(light);
    }
    return { group, lights, set, count: placed, triangles };
}
