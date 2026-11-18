/**
 * Small geometry and transform helpers shared by the procedural builders.
 *
 * Scenes are made of one unit box and one unit quad, placed many times with
 * instance transforms. That keeps the data tiny and gives the platform real
 * instancing to chew on.
 */

import type {
    LinearColor,
    ProceduralScene,
    SceneInstanceData,
    SceneLightData,
    SceneMaterialData,
    SceneMeshData,
    SceneMeshGroup,
    SceneTextureData,
    Vector3Tuple,
} from '../scene-data';

// ---------------------------------------------------------------------------
// Meshes
// ---------------------------------------------------------------------------

/**
 * A 1 m cube centred on the origin, with 24 vertices (4 per face) so every
 * face has its own flat normal. Faces wind counter-clockwise seen from outside.
 */
export function createUnitBox(name = 'unit-box'): SceneMeshData {
    // Each face: outward normal, then the four corners counter-clockwise seen
    // from outside. `u` and `v` are the face's two in-plane axes.
    const faces: ReadonlyArray<{ normal: Vector3Tuple; u: Vector3Tuple; v: Vector3Tuple }> = [
        { normal: [1, 0, 0], u: [0, 0, -1], v: [0, 1, 0] },
        { normal: [-1, 0, 0], u: [0, 0, 1], v: [0, 1, 0] },
        { normal: [0, 1, 0], u: [1, 0, 0], v: [0, 0, -1] },
        { normal: [0, -1, 0], u: [1, 0, 0], v: [0, 0, 1] },
        { normal: [0, 0, 1], u: [1, 0, 0], v: [0, 1, 0] },
        { normal: [0, 0, -1], u: [-1, 0, 0], v: [0, 1, 0] },
    ];
    const cornerSigns: ReadonlyArray<readonly [number, number]> = [
        [-1, -1],
        [1, -1],
        [1, 1],
        [-1, 1],
    ];

    const positions = new Float32Array(faces.length * 4 * 3);
    const normals = new Float32Array(faces.length * 4 * 3);
    const uvs = new Float32Array(faces.length * 4 * 2);
    const indices = new Uint32Array(faces.length * 6);

    faces.forEach((face, faceIndex) => {
        cornerSigns.forEach(([su, sv], cornerIndex) => {
            const vertex = faceIndex * 4 + cornerIndex;
            for (let axis = 0; axis < 3; axis++) {
                positions[vertex * 3 + axis] = 0.5 * face.normal[axis] + 0.5 * su * face.u[axis] + 0.5 * sv * face.v[axis];
                normals[vertex * 3 + axis] = face.normal[axis];
            }
            uvs[vertex * 2] = (su + 1) / 2;
            uvs[vertex * 2 + 1] = (sv + 1) / 2;
        });
        const first = faceIndex * 4;
        indices.set([first, first + 1, first + 2, first, first + 2, first + 3], faceIndex * 6);
    });

    return { name, positions, normals, uvs, indices };
}

/**
 * A 1 m square in the XY plane facing +Z, centred on the origin. Used for
 * zero-thickness emitters. Rotate the instance to aim it.
 */
export function createUnitQuad(name = 'unit-quad'): SceneMeshData {
    return {
        name,
        positions: new Float32Array([-0.5, -0.5, 0, 0.5, -0.5, 0, 0.5, 0.5, 0, -0.5, 0.5, 0]),
        normals: new Float32Array([0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1]),
        uvs: new Float32Array([0, 0, 1, 0, 1, 1, 0, 1]),
        indices: new Uint32Array([0, 1, 2, 0, 2, 3]),
    };
}

/** An axis-aligned box given by its minimum and maximum corners. */
export interface BoxExtent {
    readonly min: Vector3Tuple;
    readonly max: Vector3Tuple;
}

/** One material slot's boxes in a {@link createGroupedBoxMesh}. */
export interface BoxRun {
    readonly materialSlot: number;
    readonly boxes: readonly BoxExtent[];
}

/**
 * One mesh made of several boxes, with one group (material run) per entry of
 * `runs`, in mesh space. Used where a single object needs two materials, like
 * a wall with a glass pane in it: in three it is one `Mesh` with a material
 * array, in glTF one mesh with a primitive per run. Each box is the unit box
 * scaled and moved, so its faces keep their own 0..1 uvs.
 */
export function createGroupedBoxMesh(name: string, runs: readonly BoxRun[]): SceneMeshData {
    const unit = createUnitBox();
    const verticesPerBox = unit.positions.length / 3;
    const boxCount = runs.reduce((total, run) => total + run.boxes.length, 0);
    const positions = new Float32Array(boxCount * unit.positions.length);
    const normals = new Float32Array(boxCount * unit.normals.length);
    const uvs = new Float32Array(boxCount * unit.uvs.length);
    const indices = new Uint32Array(boxCount * unit.indices.length);
    const groups: SceneMeshGroup[] = [];

    let box = 0;
    for (const run of runs) {
        const start = box * unit.indices.length;
        for (const { min, max } of run.boxes) {
            for (let vertex = 0; vertex < verticesPerBox; vertex++) {
                for (let axis = 0; axis < 3; axis++) {
                    // Unit box corners are at -0.5 and +0.5; map them onto min and max.
                    const unitCoordinate = unit.positions[vertex * 3 + axis];
                    positions[(box * verticesPerBox + vertex) * 3 + axis] = min[axis] + (unitCoordinate + 0.5) * (max[axis] - min[axis]);
                }
            }
            normals.set(unit.normals, box * unit.normals.length);
            uvs.set(unit.uvs, box * unit.uvs.length);
            for (let index = 0; index < unit.indices.length; index++) {
                indices[box * unit.indices.length + index] = unit.indices[index] + box * verticesPerBox;
            }
            box++;
        }
        groups.push({ start, count: box * unit.indices.length - start, materialSlot: run.materialSlot });
    }
    return { name, positions, normals, uvs, indices, groups };
}

// ---------------------------------------------------------------------------
// Transforms (column-major 4x4, same layout as three.js Matrix4.elements)
// ---------------------------------------------------------------------------

export type Matrix4Elements = number[];

/** Translation, then a rotation about Y (degrees), then a non-uniform scale: `T * Ry * S`. */
export function composeTransform(
    position: Vector3Tuple,
    scale: Vector3Tuple,
    yawDegrees = 0,
): Matrix4Elements {
    const angle = (yawDegrees * Math.PI) / 180;
    const cos = Math.cos(angle);
    const sin = Math.sin(angle);
    const [sx, sy, sz] = scale;
    // Ry = [ cos 0 sin; 0 1 0; -sin 0 cos ]. Columns of Ry*S are Ry's columns times the scales.
    return [
        cos * sx, 0, -sin * sx, 0,
        0, sy, 0, 0,
        sin * sz, 0, cos * sz, 0,
        position[0], position[1], position[2], 1,
    ];
}

// ---------------------------------------------------------------------------
// Scene assembly
// ---------------------------------------------------------------------------

/**
 * Collects materials, instances and lights while a builder runs, and shares a
 * single unit box and unit quad between all instances.
 */
export class SceneBuilder {
    private readonly materials: SceneMaterialData[] = [];
    private readonly instances: SceneInstanceData[] = [];
    private readonly lights: SceneLightData[] = [];
    private readonly meshes: SceneMeshData[] = [createUnitBox(), createUnitQuad()];
    private readonly textures: SceneTextureData[] = [];
    private static readonly boxMesh = 0;
    private static readonly quadMesh = 1;

    /** Adds a texture and returns its index, for a material's `albedoMap`. */
    addTexture(texture: SceneTextureData): number {
        this.textures.push(texture);
        return this.textures.length - 1;
    }

    /** Adds a mesh of its own (beside the shared unit box and quad) and returns its index. */
    addMesh(mesh: SceneMeshData): number {
        this.meshes.push(mesh);
        return this.meshes.length - 1;
    }

    /** Places any mesh. `materials` has one entry per material slot of the mesh's groups. */
    addInstance(name: string, mesh: number, materials: readonly number[], transform: Matrix4Elements): void {
        if (materials.length === 0) throw new Error(`instance ${name} needs at least one material`);
        this.instances.push({
            name,
            mesh,
            material: materials[0],
            ...(materials.length > 1 ? { materials: [...materials] } : {}),
            transform,
        });
    }

    /** The unit quad (1 m square facing +Z, uv 0..1 over it) under any transform. */
    addQuad(name: string, material: number, transform: Matrix4Elements): void {
        this.addInstance(name, SceneBuilder.quadMesh, [material], transform);
    }

    /** Adds a material and returns its index. */
    addMaterial(material: Pick<SceneMaterialData, 'name' | 'albedo'> & Partial<SceneMaterialData>): number {
        this.materials.push({ emissive: [0, 0, 0], doubleSided: true, ...material });
        return this.materials.length - 1;
    }

    /** A matte, non-emitting material. */
    addDiffuse(name: string, albedo: LinearColor): number {
        return this.addMaterial({ name, albedo });
    }

    /**
     * A box given by its minimum and maximum corners. Most of the scene is built
     * from these: walls, floors, furniture.
     */
    addBox(name: string, material: number, min: Vector3Tuple, max: Vector3Tuple, yawDegrees = 0): void {
        const size: Vector3Tuple = [max[0] - min[0], max[1] - min[1], max[2] - min[2]];
        const centre: Vector3Tuple = [(min[0] + max[0]) / 2, (min[1] + max[1]) / 2, (min[2] + max[2]) / 2];
        this.instances.push({
            name,
            mesh: SceneBuilder.boxMesh,
            material,
            transform: composeTransform(centre, size, yawDegrees),
        });
    }

    /** A box given by its centre and size, which is easier for rotated objects. */
    addBoxCentered(name: string, material: number, centre: Vector3Tuple, size: Vector3Tuple, yawDegrees = 0): void {
        this.instances.push({
            name,
            mesh: SceneBuilder.boxMesh,
            material,
            transform: composeTransform(centre, size, yawDegrees),
        });
    }

    /**
     * A horizontal rectangle facing down, for ceiling emitters. `centre` is the
     * middle of the rectangle; `sizeX` and `sizeZ` are its extents.
     */
    addDownwardQuad(name: string, material: number, centre: Vector3Tuple, sizeX: number, sizeZ: number): void {
        // The unit quad faces +Z. The matrix below maps its local X to world X,
        // its local Y to world +Z and its normal (local Z) to world -Y, which is
        // a proper rotation (local X cross local Y = world X cross world Z = -Y),
        // so the quad faces down.
        this.instances.push({
            name,
            mesh: SceneBuilder.quadMesh,
            material,
            transform: [
                sizeX, 0, 0, 0,
                0, 0, sizeZ, 0,
                0, -1, 0, 0,
                centre[0], centre[1], centre[2], 1,
            ],
        });
    }

    addLight(light: SceneLightData): void {
        this.lights.push(light);
    }

    build(): ProceduralScene {
        return {
            meshes: this.meshes,
            materials: this.materials,
            instances: this.instances,
            lights: this.lights,
            ...(this.textures.length > 0 ? { textures: this.textures } : {}),
        };
    }
}

// ---------------------------------------------------------------------------
// Queries
// ---------------------------------------------------------------------------

export function triangleCount(scene: ProceduralScene): number {
    return scene.instances.reduce((total, instance) => total + scene.meshes[instance.mesh].indices.length / 3, 0);
}

/** World-space bounds of every instance, from the transformed corners of each local bounding box. */
export function sceneBounds(scene: ProceduralScene): { min: Vector3Tuple; max: Vector3Tuple } {
    const min = [Infinity, Infinity, Infinity];
    const max = [-Infinity, -Infinity, -Infinity];
    for (const instance of scene.instances) {
        const { positions } = scene.meshes[instance.mesh];
        const m = instance.transform;
        for (let i = 0; i < positions.length; i += 3) {
            const x = positions[i];
            const y = positions[i + 1];
            const z = positions[i + 2];
            for (let axis = 0; axis < 3; axis++) {
                const world = m[axis] * x + m[4 + axis] * y + m[8 + axis] * z + m[12 + axis];
                min[axis] = Math.min(min[axis], world);
                max[axis] = Math.max(max[axis], world);
            }
        }
    }
    return { min: min as unknown as Vector3Tuple, max: max as unknown as Vector3Tuple };
}
