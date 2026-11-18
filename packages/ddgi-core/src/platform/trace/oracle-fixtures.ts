/**
 * Deterministic test scenes and rays for the BVH tests and the GPU oracle
 * page. Not part of the tracer; nothing in production imports this file.
 *
 * Everything is seeded so a failing case can be reproduced exactly.
 */

import type { UvMapping } from '../contract/scene.ts';
import { buildStoreLayerOnCpu, type CpuPixelSource, type CpuStoreLayer } from '../textures/texture-store-cpu.ts';
import type { WalkMaterial, WalkMaterialLookup, WalkUvLookup } from './material-test-wgsl.ts';
import type { MeshGeometry } from './mesh-bvh.ts';
import type { TraceSceneInstance } from './scene-assembly.ts';
import type { TraceRay } from './trace-gpu.ts';

export type Random = () => number;

/** mulberry32: a small seeded generator returning numbers in [0, 1). */
export function createRandom(seed: number): Random {
    let state = seed >>> 0;
    return () => {
        state = (state + 0x6d2b79f5) >>> 0;
        let mixed = state;
        mixed = Math.imul(mixed ^ (mixed >>> 15), mixed | 1);
        mixed ^= mixed + Math.imul(mixed ^ (mixed >>> 7), mixed | 61);
        return ((mixed ^ (mixed >>> 14)) >>> 0) / 4294967296;
    };
}

export type Vector3 = [number, number, number];

export function randomUnitVector(random: Random): Vector3 {
    const z = 1 - 2 * random();
    const angle = 2 * Math.PI * random();
    const radius = Math.sqrt(Math.max(0, 1 - z * z));
    return [radius * Math.cos(angle), radius * Math.sin(angle), z];
}

/** Rounds to f32 so the CPU oracle and the GPU see exactly the same ray. */
export function toFloat32Ray(origin: readonly number[], direction: readonly number[], tMin: number, tMax: number): TraceRay {
    return {
        origin: [Math.fround(origin[0]), Math.fround(origin[1]), Math.fround(origin[2])],
        direction: [Math.fround(direction[0]), Math.fround(direction[1]), Math.fround(direction[2])],
        tMin: Math.fround(tMin),
        tMax: Math.fround(tMax),
    };
}

/** Unindexed triangles scattered in a cube of half-size `extent`, each about `size` wide. */
export function randomTriangleSoup(random: Random, triangleCount: number, extent: number, size: number): MeshGeometry {
    const positions = new Float32Array(triangleCount * 9);
    for (let triangle = 0; triangle < triangleCount; triangle++) {
        const center = [0, 1, 2].map(() => (random() * 2 - 1) * extent);
        for (let corner = 0; corner < 3; corner++) {
            for (let axis = 0; axis < 3; axis++) {
                positions[triangle * 9 + corner * 3 + axis] = center[axis] + (random() * 2 - 1) * size;
            }
        }
    }
    return { positions };
}

/**
 * A flat grid of `cells` x `cells` squares in the XY plane (z = 0) spanning
 * `[-half, half]`, each square split into two triangles along the same
 * diagonal. Vertices are shared, so every interior edge is shared by exactly
 * two triangles.
 */
export function tessellatedQuad(cells: number, half: number): MeshGeometry {
    const verticesPerRow = cells + 1;
    const positions = new Float32Array(verticesPerRow * verticesPerRow * 3);
    for (let row = 0; row <= cells; row++) {
        for (let column = 0; column <= cells; column++) {
            const base = (row * verticesPerRow + column) * 3;
            positions[base] = Math.fround(-half + (2 * half * column) / cells);
            positions[base + 1] = Math.fround(-half + (2 * half * row) / cells);
            positions[base + 2] = 0;
        }
    }
    const indices = new Uint32Array(cells * cells * 6);
    let slot = 0;
    for (let row = 0; row < cells; row++) {
        for (let column = 0; column < cells; column++) {
            const a = row * verticesPerRow + column;
            const b = a + 1;
            const c = a + verticesPerRow;
            const d = c + 1;
            // Counter-clockwise seen from +Z, diagonal from a to d.
            indices.set([a, b, d, a, d, c], slot);
            slot += 6;
        }
    }
    return { positions, indices };
}

/**
 * A random affine object-to-world transform, column-major: rotation, then
 * per-axis scale (non-uniform), optionally mirrored on x, then translation.
 */
export function randomAffine(random: Random, options: { extent: number; mirror?: boolean }): number[] {
    // Random rotation from a unit quaternion.
    const [qx, qy, qz] = randomUnitVector(random);
    const angle = random() * Math.PI * 2;
    const s = Math.sin(angle / 2);
    const x = qx * s, y = qy * s, z = qz * s, w = Math.cos(angle / 2);
    const rotation = [
        1 - 2 * (y * y + z * z), 2 * (x * y + z * w), 2 * (x * z - y * w),
        2 * (x * y - z * w), 1 - 2 * (x * x + z * z), 2 * (y * z + x * w),
        2 * (x * z + y * w), 2 * (y * z - x * w), 1 - 2 * (x * x + y * y),
    ];
    const scale = [0.5 + random() * 1.5, 0.5 + random() * 1.5, 0.5 + random() * 1.5];
    if (options.mirror) scale[0] = -scale[0];
    const translation = [0, 1, 2].map(() => (random() * 2 - 1) * options.extent);
    // Each column of the rotation is scaled by that axis's scale.
    return [
        rotation[0] * scale[0], rotation[1] * scale[0], rotation[2] * scale[0], 0,
        rotation[3] * scale[1], rotation[4] * scale[1], rotation[5] * scale[1], 0,
        rotation[6] * scale[2], rotation[7] * scale[2], rotation[8] * scale[2], 0,
        translation[0], translation[1], translation[2], 1,
    ];
}

/** Rays from random points inside the box (grown by `margin`) in random directions. */
export function randomRays(
    random: Random,
    count: number,
    bounds: readonly [number, number, number, number, number, number],
    margin: number,
    options: { randomInterval?: boolean } = {},
): TraceRay[] {
    const rays: TraceRay[] = [];
    for (let i = 0; i < count; i++) {
        const origin = [0, 1, 2].map((axis) => bounds[axis] - margin + random() * (bounds[axis + 3] - bounds[axis] + 2 * margin));
        const direction = randomUnitVector(random);
        const tMin = options.randomInterval ? random() * 2 : 0;
        const tMax = options.randomInterval ? tMin + random() * 30 : 1e6;
        rays.push(toFloat32Ray(origin, direction, tMin, tMax));
    }
    return rays;
}

// ---------------------------------------------------------------------------
// Materials in the walk (T4.2)
// ---------------------------------------------------------------------------

/** Opaque and double-sided: what every material was before M4. */
export const OPAQUE: WalkMaterial = { blocksLight: true, doubleSided: true };
/** `blocksLight: false`: glass, hidden. */
export const SEE_THROUGH: WalkMaterial = { blocksLight: false, doubleSided: true };
/** `doubleSided: false`: rays pass its back face. */
export const SINGLE_SIDED: WalkMaterial = { blocksLight: true, doubleSided: false };

/**
 * A texture of a material case: the test's own RGBA8 pixels (what a page
 * hands the platform as `PixelData`) and the store layer the CPU makes from
 * them (`buildStoreLayerOnCpu`), which the reference's alpha reads. A
 * material's `albedoTexture.layer` is this `layer` object.
 */
export interface CaseTexture {
    readonly name: string;
    readonly pixels: CpuPixelSource;
    readonly srgb: boolean;
    readonly layer: CpuStoreLayer;
}

/**
 * A scene for the material rules: meshes whose triangles use material slots,
 * instances that give each slot a material, and rays aimed at the surfaces
 * from both sides. Instance `i` of `instances` has `instanceIndex` `i`.
 */
export interface MaterialCase {
    readonly name: string;
    readonly description: string;
    /** Indexed meshes. */
    readonly meshes: readonly MeshGeometry[];
    /** Per mesh, each triangle's material slot, in index order. */
    readonly triangleSlots: readonly Uint32Array[];
    /** Per mesh, its uv sets 0 and 1 (two numbers a vertex; `MeshDescriptor.uvs`, `uvs1`), or null where it has none. */
    readonly uvs?: readonly (readonly [Float32Array | null, Float32Array | null])[];
    /** The textures the materials use. */
    readonly textures?: readonly CaseTexture[];
    readonly instances: readonly TraceSceneInstance[];
    /** Per instance, its material for each slot (`InstanceDescriptor.materials`). */
    readonly materials: readonly (readonly WalkMaterial[])[];
    readonly rays: readonly TraceRay[];
}

/**
 * The material a triangle of an instance uses in a case, with `materials`
 * standing in for the case's own (a page that edits materials at run time
 * passes the edited lists). Slots past an instance's list use its last
 * material, as `giMaterialAt` does.
 */
export function materialLookup(materialCase: MaterialCase, materials: readonly (readonly WalkMaterial[])[] = materialCase.materials): WalkMaterialLookup {
    return (instanceIndex, triangleIndex) => {
        const meshIndex = materialCase.instances[instanceIndex].meshIndex;
        const list = materials[instanceIndex];
        return list[Math.min(materialCase.triangleSlots[meshIndex][triangleIndex], list.length - 1)];
    };
}

/**
 * A triangle's corner uvs in a case, `[u0, v0, u1, v1, u2, v2]`, from its
 * mesh's uv set; null where the mesh has no such set.
 */
export function uvLookup(materialCase: MaterialCase): WalkUvLookup {
    return (instanceIndex, triangleIndex, channel) => {
        const meshIndex = materialCase.instances[instanceIndex].meshIndex;
        const uvs = materialCase.uvs?.[meshIndex]?.[channel];
        const indices = materialCase.meshes[meshIndex].indices;
        if (!uvs || !indices) return null;
        const corners = new Float64Array(6);
        for (let corner = 0; corner < 3; corner++) {
            const vertex = indices[triangleIndex * 3 + corner];
            corners[corner * 2] = uvs[vertex * 2];
            corners[corner * 2 + 1] = uvs[vertex * 2 + 1];
        }
        return corners;
    };
}

/** Runs of consecutive triangles with one slot, as `MeshDescriptor.groups` takes them (in indices, not triangles). */
export function slotRuns(slots: Uint32Array): { start: number; count: number; materialSlot: number }[] {
    const runs: { start: number; count: number; materialSlot: number }[] = [];
    slots.forEach((slot, triangle) => {
        const last = runs[runs.length - 1];
        if (last && last.materialSlot === slot) last.count += 3;
        else runs.push({ start: triangle * 3, count: 3, materialSlot: slot });
    });
    return runs;
}

/** Collects quads and grids with a material slot per triangle, and uvs where they have them. */
class SlottedMeshBuilder {
    private readonly positions: number[] = [];
    private readonly uvs: number[] = [];
    private readonly indices: number[] = [];
    private readonly slots: number[] = [];

    /** One quad, its corners counter-clockwise seen from its front. */
    quad(corners: readonly Vector3[], slot: number): void {
        const first = this.positions.length / 3;
        for (const corner of corners) this.positions.push(...corner);
        // Texture (0, 0) at the second corner's opposite: corners 0 1 2 3 get (0,1) (1,1) (1,0) (0,0),
        // so a quad listed bottom-left first, counter-clockwise, shows the image upright.
        this.uvs.push(0, 1, 1, 1, 1, 0, 0, 0);
        this.indices.push(first, first + 1, first + 2, first, first + 2, first + 3);
        this.slots.push(slot, slot);
    }

    /**
     * A grid in the plane z = 0, front facing +z, with vertices shared between
     * cells (so a pane shares its edges with the wall around it). `slotOf`
     * gives each cell's slot by column and row.
     */
    grid(xs: readonly number[], ys: readonly number[], slotOf: (column: number, row: number) => number): void {
        const first = this.positions.length / 3;
        // uvs over the whole grid: u left to right, v top to bottom (texture (0, 0) is the image's top-left).
        const [left, right] = [xs[0], xs[xs.length - 1]];
        const [bottom, top] = [ys[0], ys[ys.length - 1]];
        for (const y of ys) {
            for (const x of xs) {
                this.positions.push(x, y, 0);
                this.uvs.push((x - left) / (right - left), (top - y) / (top - bottom));
            }
        }
        const columns = xs.length;
        for (let row = 0; row + 1 < ys.length; row++) {
            for (let column = 0; column + 1 < columns; column++) {
                const a = first + row * columns + column;
                const b = a + 1;
                const c = a + columns;
                const d = c + 1;
                // Counter-clockwise seen from +z, as tessellatedQuad.
                this.indices.push(a, b, d, a, d, c);
                const slot = slotOf(column, row);
                this.slots.push(slot, slot);
            }
        }
    }

    build(): { mesh: MeshGeometry; slots: Uint32Array; uvs: Float32Array } {
        return {
            mesh: { positions: Float32Array.from(this.positions, Math.fround), indices: Uint32Array.from(this.indices) },
            slots: Uint32Array.from(this.slots),
            uvs: Float32Array.from(this.uvs),
        };
    }
}

/** A column-major translation after an optional per-axis scale (a negative one mirrors). */
function placement(translation: Vector3, scale: Vector3 = [1, 1, 1]): number[] {
    return [scale[0], 0, 0, 0, 0, scale[1], 0, 0, 0, 0, scale[2], 0, translation[0], translation[1], translation[2], 1];
}

/** A half turn about +y (the front then faces -z), then a translation. */
function turnedAround(translation: Vector3): number[] {
    return [-1, 0, 0, 0, 0, 1, 0, 0, 0, 0, -1, 0, translation[0], translation[1], translation[2], 1];
}

/** An axis-aligned box: its low and high corners. */
type Box = readonly [Vector3, Vector3];

/**
 * Rays from a random point of a random box in `origins` toward a random point
 * of a random box in `targets`: aimed at the surfaces under test, from both
 * sides, at every angle the boxes allow.
 */
function aimedRays(random: Random, count: number, origins: readonly Box[], targets: readonly Box[]): TraceRay[] {
    const pointIn = (boxes: readonly Box[]): Vector3 => {
        const [low, high] = boxes[Math.floor(random() * boxes.length)];
        return [low[0] + random() * (high[0] - low[0]), low[1] + random() * (high[1] - low[1]), low[2] + random() * (high[2] - low[2])];
    };
    const rays: TraceRay[] = [];
    while (rays.length < count) {
        const origin = pointIn(origins);
        const target = pointIn(targets);
        const direction = [target[0] - origin[0], target[1] - origin[1], target[2] - origin[2]];
        const length = Math.hypot(direction[0], direction[1], direction[2]);
        if (!(length > 1e-3)) continue;
        rays.push(toFloat32Ray(origin, direction.map((component) => component / length), 0, 1e6));
    }
    return rays;
}

/** A large opaque quad facing +z, as a backdrop behind the surfaces under test. */
function backdropMesh(half: number): { mesh: MeshGeometry; slots: Uint32Array } {
    const builder = new SlottedMeshBuilder();
    builder.quad(
        [
            [-half, -half, 0],
            [half, -half, 0],
            [half, half, 0],
            [-half, half, 0],
        ],
        0,
    );
    return builder.build();
}

/**
 * The three scenes of T4.2's first check:
 *
 * - `glass-pane-in-wall`: one wall mesh with two slots, a 1 x 1 m pane
 *   (slot 1, see-through) in a 4 x 3 m wall (slot 0, opaque), sharing its
 *   edges; opaque walls 2 m in front and behind catch what goes through.
 * - `single-sided-quad`: a single-sided tessellated quad in three instances
 *   (as authored, mirrored, turned around), hit from both sides, between two
 *   opaque backdrops.
 * - `per-instance-see-through`: one mesh with two slots (left and right
 *   halves) in two instances: one all opaque, one whose right half is
 *   see-through. A mesh's flags are a union over its instances, so the
 *   opaque instance's right half is consulted too, and must still block.
 */
export function buildMaterialCases(raysPerCase: number, seed = 0x7a42): MaterialCase[] {
    const random = createRandom(seed);
    const inFront: Box = [
        [-4.5, -1, 0.2],
        [4.5, 4, 1.8],
    ];
    const behind: Box = [
        [-4.5, -1, -1.8],
        [4.5, 4, -0.2],
    ];

    const wall = new SlottedMeshBuilder();
    // A 3 x 3 grid of cells; the middle one is the pane.
    wall.grid([-2, -0.5, 0.5, 2], [0, 1, 2, 3], (column, row) => (column === 1 && row === 1 ? 1 : 0));
    const wallMesh = wall.build();
    const backdrop = backdropMesh(6);
    const glass: MaterialCase = {
        name: 'glass-pane-in-wall',
        description: 'a see-through pane (slot 1) in an opaque wall (slot 0) of one mesh, opaque walls in front and behind',
        meshes: [wallMesh.mesh, backdrop.mesh],
        triangleSlots: [wallMesh.slots, backdrop.slots],
        instances: [
            { meshIndex: 0, instanceIndex: 0, objectToWorld: placement([0, 0, 0]) },
            { meshIndex: 1, instanceIndex: 1, objectToWorld: placement([0, 1.5, -2]) },
            { meshIndex: 1, instanceIndex: 2, objectToWorld: placement([0, 1.5, 2]) },
        ],
        materials: [[OPAQUE, SEE_THROUGH], [OPAQUE], [OPAQUE]],
        // Aimed at the pane and the wall around it.
        rays: aimedRays(random, raysPerCase, [inFront, behind], [
            [
                [-1.2, 0.3, 0],
                [1.2, 2.7, 0],
            ],
        ]),
    };

    const quad = tessellatedQuad(4, 1);
    const singleSided: MaterialCase = {
        name: 'single-sided-quad',
        description: 'a single-sided quad as authored, mirrored and turned around, hit from both sides, between opaque backdrops',
        meshes: [quad, backdrop.mesh],
        triangleSlots: [new Uint32Array((quad.indices?.length ?? 0) / 3), backdrop.slots],
        instances: [
            { meshIndex: 0, instanceIndex: 0, objectToWorld: placement([-3, 1.5, 0]) },
            { meshIndex: 0, instanceIndex: 1, objectToWorld: placement([0, 1.5, 0], [-1, 1, 1]) },
            { meshIndex: 0, instanceIndex: 2, objectToWorld: turnedAround([3, 1.5, 0]) },
            { meshIndex: 1, instanceIndex: 3, objectToWorld: placement([0, 1.5, -2]) },
            { meshIndex: 1, instanceIndex: 4, objectToWorld: placement([0, 1.5, 2]) },
        ],
        materials: [[SINGLE_SIDED], [SINGLE_SIDED], [SINGLE_SIDED], [OPAQUE], [OPAQUE]],
        rays: aimedRays(random, raysPerCase, [inFront, behind], [
            [
                [-4.2, 0.3, 0],
                [4.2, 2.7, 0],
            ],
        ]),
    };

    const halves = new SlottedMeshBuilder();
    // Left half slot 0, right half slot 1.
    halves.grid([-1, -0.5, 0, 0.5, 1], [-1, -0.5, 0, 0.5, 1], (column) => (column < 2 ? 0 : 1));
    const halvesMesh = halves.build();
    const perInstance: MaterialCase = {
        name: 'per-instance-see-through',
        description: 'one two-slot mesh in two instances: all opaque, and with its right half see-through',
        meshes: [halvesMesh.mesh, backdrop.mesh],
        triangleSlots: [halvesMesh.slots, backdrop.slots],
        instances: [
            { meshIndex: 0, instanceIndex: 0, objectToWorld: placement([-1.5, 1.5, 0]) },
            { meshIndex: 0, instanceIndex: 1, objectToWorld: placement([1.5, 1.5, 0]) },
            { meshIndex: 1, instanceIndex: 2, objectToWorld: placement([0, 1.5, -2]) },
            { meshIndex: 1, instanceIndex: 3, objectToWorld: placement([0, 1.5, 2]) },
        ],
        materials: [[OPAQUE, OPAQUE], [OPAQUE, SEE_THROUGH], [OPAQUE], [OPAQUE]],
        rays: aimedRays(random, raysPerCase, [inFront, behind], [
            [
                [-2.7, 0.3, 0],
                [2.7, 2.7, 0],
            ],
        ]),
    };

    return [glass, singleSided, perInstance];
}

// ---------------------------------------------------------------------------
// Alpha in the walk (T4.2, second half)
// ---------------------------------------------------------------------------

/**
 * An RGBA8 image of `size` × `size` whose texel (x, y) (row 0 at the top) is
 * opaque red where `opaque(x, y)` and clear blue elsewhere. Red and blue so
 * a coarse mip that took in the clear texels' colour shows it (the
 * alpha-weighted mips check).
 */
export function alphaPatternImage(size: number, opaque: (x: number, y: number) => boolean): CpuPixelSource {
    const data = new Uint8Array(size * size * 4);
    for (let y = 0; y < size; y++) {
        for (let x = 0; x < size; x++) {
            data.set(opaque(x, y) ? [255, 0, 0, 255] : [0, 0, 255, 0], (y * size + x) * 4);
        }
    }
    return { width: size, height: size, data };
}

/** A case texture: the pixels, and their store layer made on the CPU. */
export function caseTexture(name: string, pixels: CpuPixelSource, srgb = true): CaseTexture {
    return { name, pixels, srgb, layer: buildStoreLayerOnCpu(pixels, srgb) };
}

/** A 16 × 16 checkerboard of opaque and clear texels: half of a wall it covers is open. */
export const CHECKER_SIZE = 16;
export function checkerImage(): CpuPixelSource {
    return alphaPatternImage(CHECKER_SIZE, (x, y) => (x + y) % 2 === 0);
}

/** A 16 × 16 pattern with no symmetry a mirror or a turn would hide: rows of different lengths, offset. */
export function asymmetricImage(): CpuPixelSource {
    return alphaPatternImage(16, (x, y) => (x * 7 + y * 3) % 11 < 5 || (y < 4 && x < 9));
}

/** A cutout material (cutoff 0.5) whose alpha is a case texture's, through `uv`. */
export function cutoutMaterial(texture: CaseTexture, uv?: UvMapping): WalkMaterial {
    return { blocksLight: true, doubleSided: true, alphaMode: 'cutout', alphaCutoff: 0.5, opacity: 1, albedoTexture: { layer: texture.layer, uv } };
}

/** A stochastic material of constant alpha (no texture): a share 1 − `opacity` of the rays passes. */
export function stochasticMaterial(opacity: number): WalkMaterial {
    return { blocksLight: true, doubleSided: true, alphaMode: 'stochastic', alphaCutoff: 0, opacity };
}

/** A uniform 16 × 16 texture whose alpha is the byte `alphaByte` everywhere (colour grey). */
export function uniformAlphaImage(alphaByte: number): CpuPixelSource {
    const data = new Uint8Array(16 * 16 * 4);
    for (let texel = 0; texel < 16 * 16; texel++) data.set([128, 128, 128, alphaByte], texel * 4);
    return { width: 16, height: 16, data };
}

/** The byte the stochastic case's texture holds in alpha: about a half, so texture and opacity both matter. */
const STOCHASTIC_TEXTURE_ALPHA = 128;

/** The uv transform of the mirrored-cutout case: turned 30°, scaled, offset, so mirror wrap folds it several times. */
export const TURNED_UV: UvMapping = (() => {
    const angle = Math.PI / 6;
    const [scaleU, scaleV] = [2.3, 1.7];
    const cos = Math.cos(angle);
    const sin = Math.sin(angle);
    return { channel: 1, matrix: [cos * scaleU, sin * scaleV, -sin * scaleU, cos * scaleV, 0.37, -0.21], wrapU: 'mirror', wrapV: 'mirror' };
})();

/**
 * The alpha cases of T4.2's second half (done-when 1):
 *
 * - `checker-cutout-wall`: a 3 x 3 m wall mesh (a grid of 4 x 4 cells,
 *   shared vertices) whose albedo texture is a 16 x 16 checkerboard of clear
 *   and opaque texels (one texel = 1/16 of the wall), cutout at 0.5; opaque
 *   walls 2 m in front and behind.
 * - `cutout-uv-mirror`: a cutout quad reading an asymmetric pattern through
 *   uv set 1 (the quad's uvs turned around) and a turned, scaled, offset uv
 *   transform with mirror wrap on both axes; a second, mirrored instance.
 * - `stochastic-quad`: a quad of constant alpha `stochasticAlpha`, stochastic:
 *   a share 1 − alpha of the rays that meet it pass. Judged on that share.
 *   The alpha is opacity × a texture's alpha (a uniform texture of alpha
 *   128/255, and the opacity that makes the product `stochasticAlpha`), so the
 *   GPU's stochastic rule reads the texture too: one that ignored the texture
 *   would let about half the rays through instead.
 */
export function buildAlphaCases(raysPerCase: number, stochasticRays = raysPerCase, stochasticAlpha = 0.25, seed = 0xa1fa): MaterialCase[] {
    const random = createRandom(seed);
    const inFront: Box = [
        [-4.5, -1, 0.2],
        [4.5, 4, 1.8],
    ];
    const behind: Box = [
        [-4.5, -1, -1.8],
        [4.5, 4, -0.2],
    ];
    const backdrop = backdropMesh(6);
    const backdropInstances = (first: number): TraceSceneInstance[] => [
        { meshIndex: 1, instanceIndex: first, objectToWorld: placement([0, 1.5, -2]) },
        { meshIndex: 1, instanceIndex: first + 1, objectToWorld: placement([0, 1.5, 2]) },
    ];

    const checker = caseTexture('checker', checkerImage());
    const wall = new SlottedMeshBuilder();
    wall.grid([-1.5, -0.75, 0, 0.75, 1.5], [0, 0.75, 1.5, 2.25, 3], () => 0);
    const wallMesh = wall.build();
    const checkerWall: MaterialCase = {
        name: 'checker-cutout-wall',
        description: 'a 3 x 3 m wall cut out by a 16 x 16 checkerboard texture (cutoff 0.5), opaque walls in front and behind',
        meshes: [wallMesh.mesh, backdrop.mesh],
        triangleSlots: [wallMesh.slots, backdrop.slots],
        uvs: [
            [wallMesh.uvs, null],
            [null, null],
        ],
        textures: [checker],
        instances: [{ meshIndex: 0, instanceIndex: 0, objectToWorld: placement([0, 0, 0]) }, ...backdropInstances(1)],
        materials: [[cutoutMaterial(checker)], [OPAQUE], [OPAQUE]],
        rays: aimedRays(random, raysPerCase, [inFront, behind], [
            [
                [-1.6, -0.1, 0],
                [1.6, 3.1, 0],
            ],
        ]),
    };

    const asymmetric = caseTexture('asymmetric', asymmetricImage());
    const quad = new SlottedMeshBuilder();
    quad.quad(
        [
            [-1, -1, 0],
            [1, -1, 0],
            [1, 1, 0],
            [-1, 1, 0],
        ],
        0,
    );
    const quadMesh = quad.build();
    // uv set 1: set 0 turned half around, so reading the wrong set shows.
    const turnedUvs = quadMesh.uvs.map((value) => 1 - value);
    const mirrored: MaterialCase = {
        name: 'cutout-uv-mirror',
        description: 'a cutout quad read through uv set 1, a turned, scaled, offset uv transform and mirror wrap; a mirrored second instance',
        meshes: [quadMesh.mesh, backdrop.mesh],
        triangleSlots: [quadMesh.slots, backdrop.slots],
        uvs: [
            [quadMesh.uvs, turnedUvs],
            [null, null],
        ],
        textures: [asymmetric],
        instances: [
            { meshIndex: 0, instanceIndex: 0, objectToWorld: placement([-1.5, 1.5, 0]) },
            { meshIndex: 0, instanceIndex: 1, objectToWorld: placement([1.5, 1.5, 0], [-1, 1, 1]) },
            ...backdropInstances(2),
        ],
        materials: [[cutoutMaterial(asymmetric, TURNED_UV)], [cutoutMaterial(asymmetric, TURNED_UV)], [OPAQUE], [OPAQUE]],
        rays: aimedRays(random, raysPerCase, [inFront, behind], [
            [
                [-2.6, 0.4, 0],
                [2.6, 2.6, 0],
            ],
        ]),
    };

    const halfAlpha = caseTexture('half alpha', uniformAlphaImage(STOCHASTIC_TEXTURE_ALPHA));
    const textureAlpha = STOCHASTIC_TEXTURE_ALPHA / 255;
    const stochasticTextured: WalkMaterial = {
        ...stochasticMaterial(Math.min(1, stochasticAlpha / textureAlpha)),
        albedoTexture: { layer: halfAlpha.layer },
    };
    const stochastic: MaterialCase = {
        name: 'stochastic-quad',
        description: `a quad of constant alpha ${stochasticAlpha} (opacity x a uniform texture alpha), stochastic, opaque walls in front and behind`,
        meshes: [quadMesh.mesh, backdrop.mesh],
        triangleSlots: [quadMesh.slots, backdrop.slots],
        uvs: [
            [quadMesh.uvs, null],
            [null, null],
        ],
        textures: [halfAlpha],
        instances: [{ meshIndex: 0, instanceIndex: 0, objectToWorld: placement([0, 1.5, 0], [1.5, 1.5, 1]) }, ...backdropInstances(1)],
        materials: [[stochasticTextured], [OPAQUE], [OPAQUE]],
        rays: aimedRays(random, stochasticRays, [inFront, behind], [
            [
                [-1.4, 0.1, 0],
                [1.4, 2.9, 0],
            ],
        ]),
    };

    return [checkerWall, mirrored, stochastic];
}
