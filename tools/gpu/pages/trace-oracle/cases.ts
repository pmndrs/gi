/**
 * The oracle cases: scenes plus rays, with the answers a correct tracer must give.
 *
 * Ported from the old project's CPU oracle tests (gpu-ray-provider): near and
 * far triangles, back sides, reversed winding, non-uniform and mirrored
 * transforms, instances with distinct ids. Added for this spike: thin walls,
 * a closed thin box that no interior ray may escape, random scenes for
 * stress, and the shared-edge test (rays aimed at the seams of a tessellated
 * quad, which must not slip between triangles).
 *
 * Each case is plain data, so the CPU tests (`cases.test.ts`) and the GPU page
 * (`main.ts`) run exactly the same rays.
 */

import {
    packRays,
    unpackRay,
    type MeshGeometry,
    type TraceRay,
    type TraceSceneInstance,
} from '../../../../packages/ddgi-core/src/platform/trace/index.ts';
import {
    createRandom,
    randomAffine,
    randomRays,
    randomTriangleSoup,
    tessellatedQuad,
    toFloat32Ray,
    type Random,
    type Vector3,
} from '../../../../packages/ddgi-core/src/platform/trace/oracle-fixtures.ts';

export { unpackRay };

/** What a correct tracer must return for one ray, worked out by hand. Unset fields are not checked. */
export interface ExpectedHit {
    readonly hit: boolean;
    readonly instanceIndex?: number;
    readonly triangleIndex?: number;
    readonly frontFace?: boolean;
    readonly t?: number;
    readonly u?: number;
    readonly v?: number;
}

export interface CuratedRay {
    readonly label: string;
    readonly ray: TraceRay;
    readonly expected: ExpectedHit;
    /**
     * The ray runs exactly along a shared edge, so the double-precision
     * reference is ambiguous by definition. Only `expected` is checked.
     */
    readonly onEdge?: boolean;
}

export interface OracleCase {
    readonly name: string;
    readonly description: string;
    readonly meshes: readonly MeshGeometry[];
    readonly instances: readonly TraceSceneInstance[];
    /** Packed `GiRay`s. */
    readonly rays: Float32Array;
    /** Hand-checked rays; `rays` holds exactly these, in order. Absent for random cases. */
    readonly curated?: readonly CuratedRay[];
    /**
     * Every ray starts inside a closed shell, so a miss is an escape and fails
     * the case, in closest-hit and any-hit alike. Escapes are counted apart
     * from ordinary mismatches, for the report.
     */
    readonly everyRayMustHit?: boolean;
    /**
     * Rays from this index on are aimed at edges or corners. Their escapes are
     * counted apart from those of the random rays, so the report says where a
     * hole is; they fail the case all the same.
     */
    readonly edgeAimedFrom?: number;
    /** Largest share of rays per mode whose disagreement with the reference may be excused as ambiguous. Curated cases allow none. */
    readonly maxAmbiguousShare: AmbiguousAllowance;
}

export interface AmbiguousAllowance {
    readonly closest: number;
    readonly any: number;
}

const NO_AMBIGUITY: AmbiguousAllowance = { closest: 0, any: 0 };

const IDENTITY = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1];

/** Translation * rotation about Y * scale, column-major (the order three.js composes a node). */
export function compose(translation: Vector3, rotationY: number, scale: Vector3): number[] {
    const c = Math.cos(rotationY);
    const s = Math.sin(rotationY);
    return [
        c * scale[0], 0, -s * scale[0], 0,
        0, scale[1], 0, 0,
        s * scale[2], 0, c * scale[2], 0,
        translation[0], translation[1], translation[2], 1,
    ];
}

export function transformPoint(matrix: readonly number[], point: Vector3): Vector3 {
    return [
        matrix[0] * point[0] + matrix[4] * point[1] + matrix[8] * point[2] + matrix[12],
        matrix[1] * point[0] + matrix[5] * point[1] + matrix[9] * point[2] + matrix[13],
        matrix[2] * point[0] + matrix[6] * point[1] + matrix[10] * point[2] + matrix[14],
    ];
}

function normalize(v: Vector3): Vector3 {
    const length = Math.hypot(v[0], v[1], v[2]);
    return [v[0] / length, v[1] / length, v[2] / length];
}

function ray(origin: Vector3, direction: Vector3, tMin = 0, tMax = 1e6): TraceRay {
    return toFloat32Ray(origin, normalize(direction), tMin, tMax);
}

function curatedCase(
    name: string,
    description: string,
    meshes: MeshGeometry[],
    instances: TraceSceneInstance[],
    curated: CuratedRay[],
): OracleCase {
    return {
        name,
        description,
        meshes,
        instances,
        curated,
        rays: packRays(curated.map((entry) => entry.ray)),
        maxAmbiguousShare: NO_AMBIGUITY,
    };
}

// ---------------------------------------------------------------------------
// Ported: near/far, back side, reversed winding, non-uniform and mirrored transforms
// ---------------------------------------------------------------------------

function analyticCase(): OracleCase {
    const triangle: MeshGeometry = { positions: new Float32Array([-2, -2, 0, 2, -2, 0, 0, 2, 0]) };
    const reversed: MeshGeometry = { positions: new Float32Array([-2, -2, 0, 0, 2, 0, 2, -2, 0]) };
    const rightTriangle: MeshGeometry = { positions: new Float32Array([0, 0, 0, 2, 0, 0, 0, 2, 0]) };

    const nonUniform = compose([-8, 0, -1], Math.PI / 4, [2, 0.5, 3]);
    const mirrored = compose([12, 0, -1], Math.PI / 4, [-2, 0.5, 3]);
    const instances: TraceSceneInstance[] = [
        { meshIndex: 0, instanceIndex: 10, objectToWorld: IDENTITY },
        { meshIndex: 0, instanceIndex: 20, objectToWorld: compose([0, 0, -2], 0, [1, 1, 1]) },
        { meshIndex: 1, instanceIndex: 30, objectToWorld: compose([6, 0, 0], 0, [1, 1, 1]) },
        { meshIndex: 2, instanceIndex: 60, objectToWorld: nonUniform },
        { meshIndex: 2, instanceIndex: 70, objectToWorld: mirrored },
    ];

    // A ray that arrives along the world-space plane normal of a transformed right triangle,
    // 3 units from the point (2/3, 2/3, 0) of the triangle. Its distance must be 3 in world
    // units however the transform scales or mirrors the object.
    const normal: Vector3 = [Math.SQRT1_2, 0, Math.SQRT1_2];
    const alongNormal = (matrix: number[]): TraceRay => {
        const point = transformPoint(matrix, [2 / 3, 2 / 3, 0]);
        return ray(
            [point[0] + normal[0] * 3, point[1], point[2] + normal[2] * 3],
            [-normal[0], 0, -normal[2]],
        );
    };

    const down: Vector3 = [0, 0, -1];
    const curated: CuratedRay[] = [
        { label: 'nearest', ray: ray([0, 0, 3], down), expected: { hit: true, instanceIndex: 10, triangleIndex: 0, t: 3, frontFace: true, u: 0.25, v: 0.5 } },
        { label: 'tMin skips the near triangle', ray: ray([0, 0, 3], down, 3.001), expected: { hit: true, instanceIndex: 20, t: 5, frontFace: true } },
        { label: 'tMax stops before the near triangle', ray: ray([0, 0, 3], down, 0, 2.999), expected: { hit: false } },
        { label: 'back side', ray: ray([0, 0, -1], [0, 0, 1]), expected: { hit: true, instanceIndex: 10, t: 1, frontFace: false } },
        { label: 'reversed winding', ray: ray([6, 0, 3], down), expected: { hit: true, instanceIndex: 30, t: 3, frontFace: false } },
        { label: 'non-uniform scale', ray: alongNormal(nonUniform), expected: { hit: true, instanceIndex: 60, t: 3, frontFace: true } },
        { label: 'mirrored', ray: alongNormal(mirrored), expected: { hit: true, instanceIndex: 70, t: 3, frontFace: false } },
        { label: 'miss', ray: ray([30, 30, 3], down), expected: { hit: false } },
        { label: 'parallel to the triangles', ray: ray([0, 0, 3], [1, 0, 0]), expected: { hit: false } },
        { label: 'pointing away', ray: ray([0, 0, 3], [0, 0, 1]), expected: { hit: false } },
    ];
    return curatedCase('analytic', 'Ported: near/far, back side, reversed winding, non-uniform and mirrored transforms.', [triangle, reversed, rightTriangle], instances, curated);
}

// ---------------------------------------------------------------------------
// Ported: instancing with distinct ids and a mirrored copy
// ---------------------------------------------------------------------------

function instancingCase(): OracleCase {
    const mesh: MeshGeometry = { positions: new Float32Array([-1, -1, 0, 1, -1, 0, 0, 1, 0]) };
    // Platform instance indices are deliberately not 0, 1, 2 and not in tree order.
    const instances: TraceSceneInstance[] = [
        { meshIndex: 0, instanceIndex: 7, objectToWorld: compose([-4, 0, 0], 0, [1, 1, 1]) },
        { meshIndex: 0, instanceIndex: 3, objectToWorld: compose([0, 0, -2], 0, [2, 0.5, 1]) },
        { meshIndex: 0, instanceIndex: 11, objectToWorld: compose([4, 0, 0], 0, [-1, 1, 1]) },
        // Hidden and not contributing: rays must pass straight through these.
        { meshIndex: 0, instanceIndex: 99, objectToWorld: compose([0, 0, 1], 0, [3, 3, 1]), visible: false },
        { meshIndex: 0, instanceIndex: 98, objectToWorld: compose([-4, 0, 1], 0, [3, 3, 1]), contributesToGi: false },
    ];
    const down: Vector3 = [0, 0, -1];
    const curated: CuratedRay[] = [
        { label: 'instance 7', ray: ray([-4, 0, 3], down), expected: { hit: true, instanceIndex: 7, triangleIndex: 0, t: 3, frontFace: true } },
        { label: 'scaled instance 3', ray: ray([0, 0, 3], down), expected: { hit: true, instanceIndex: 3, triangleIndex: 0, t: 5, frontFace: true } },
        { label: 'mirrored instance 11', ray: ray([4, 0, 3], down), expected: { hit: true, instanceIndex: 11, triangleIndex: 0, t: 3, frontFace: false } },
        { label: 'between instances', ray: ray([-2, 0, 3], down), expected: { hit: false } },
        { label: 'inside the scaled box, outside the triangle', ray: ray([1.9, 0.4, 3], down), expected: { hit: false } },
        { label: 'scaled instance, off centre', ray: ray([0.5, -0.2, 3], down), expected: { hit: true, instanceIndex: 3, t: 5, frontFace: true } },
        { label: 'mirrored instance, off centre', ray: ray([3.5, -0.5, 3], down), expected: { hit: true, instanceIndex: 11, t: 3, frontFace: false } },
        { label: 'from below through two instances', ray: ray([0, 0, -5], [0, 0, 1]), expected: { hit: true, instanceIndex: 3, t: 3, frontFace: false } },
        { label: 'slanted ray at instance 7', ray: ray([-4 + 3, 0, 3], [-1, 0, -1]), expected: { hit: true, instanceIndex: 7, t: 3 * Math.SQRT2, frontFace: true } },
    ];
    return curatedCase('instancing', 'Ported: instances with distinct ids, scaled and mirrored copies, hidden instances ignored.', [mesh], instances, curated);
}

// ---------------------------------------------------------------------------
// Thin walls
// ---------------------------------------------------------------------------

/** A square wall in the plane x = 0, y and z in [-2, 2], two triangles, normal +x. */
function wallMesh(): MeshGeometry {
    return {
        positions: new Float32Array([0, -2, -2, 0, 2, -2, 0, 2, 2, 0, -2, 2]),
        indices: new Uint32Array([0, 1, 2, 0, 2, 3]),
    };
}

function thinWallCase(): OracleCase {
    // Two walls 1 mm apart, like the two skins of a thin partition.
    const instances: TraceSceneInstance[] = [
        { meshIndex: 0, instanceIndex: 1, objectToWorld: IDENTITY },
        { meshIndex: 0, instanceIndex: 2, objectToWorld: compose([0.001, 0, 0], 0, [1, 1, 1]) },
    ];
    const curated: CuratedRay[] = [
        // The wall's normal is +x, so a ray travelling +x hits the back and one travelling -x hits the front.
        { label: 'from -x: first skin', ray: ray([-1, 0.3, 0.2], [1, 0, 0]), expected: { hit: true, instanceIndex: 1, t: 1, frontFace: false } },
        { label: 'from -x with tMin between skins: second skin', ray: ray([-1, 0.3, 0.2], [1, 0, 0], 1.0005), expected: { hit: true, instanceIndex: 2, t: 1.001, frontFace: false } },
        { label: 'from -x with tMax between skins: first skin', ray: ray([-1, 0.3, 0.2], [1, 0, 0], 0, 1.0005), expected: { hit: true, instanceIndex: 1, t: 1, frontFace: false } },
        { label: 'from -x with tMax before the walls', ray: ray([-1, 0.3, 0.2], [1, 0, 0], 0, 0.9995), expected: { hit: false } },
        { label: 'from +x: second skin first', ray: ray([2, 0.1, -0.7], [-1, 0, 0]), expected: { hit: true, instanceIndex: 2, t: 1.999, frontFace: true } },
        { label: 'from +x with tMin between skins', ray: ray([2, 0.1, -0.7], [-1, 0, 0], 1.9995), expected: { hit: true, instanceIndex: 1, t: 2, frontFace: true } },
        { label: 'grazing at 0.06 degrees', ray: ray([-0.0004, -1.9, 0.3], [0.001, 1, 0]), expected: { hit: true, instanceIndex: 1, t: 0.4, frontFace: false } },
        { label: 'edge-on (in the wall plane)', ray: ray([0, -5, 0.3], [0, 1, 0]), expected: { hit: false } },
        { label: 'past the wall edge', ray: ray([-1, 2.01, 0], [1, 0, 0]), expected: { hit: false } },
        // The two triangles of a wall share the diagonal y = z. A ray through it must hit one of them.
        { label: 'through the diagonal seam', ray: ray([-1, 0.5, 0.5], [1, 0, 0]), expected: { hit: true, instanceIndex: 1, t: 1 }, onEdge: true },
        { label: 'through the seam, far end', ray: ray([-1, -1.75, -1.75], [1, 0, 0]), expected: { hit: true, instanceIndex: 1, t: 1 }, onEdge: true },
    ];
    return curatedCase('thin-wall', 'Two skins 1 mm apart: order, tMin/tMax between them, grazing, edge-on, through the seam.', [wallMesh()], instances, curated);
}

/** A closed box of half-size `half`, 12 outward-facing triangles over 8 shared vertices. */
export function boxMesh(half: Vector3): MeshGeometry {
    const positions: number[] = [];
    for (let corner = 0; corner < 8; corner++) {
        positions.push(corner & 1 ? half[0] : -half[0], corner & 2 ? half[1] : -half[1], corner & 4 ? half[2] : -half[2]);
    }
    const indices: number[] = [];
    for (let axis = 0; axis < 3; axis++) {
        const u = (axis + 1) % 3;
        const v = (axis + 2) % 3;
        for (const sign of [1, -1]) {
            // Corners of the face in (u, v) order -, -; +, -; +, +; -, +.
            const cornerIndex = (uPlus: number, vPlus: number): number =>
                (sign > 0 ? 1 << axis : 0) | (uPlus ? 1 << u : 0) | (vPlus ? 1 << v : 0);
            const p0 = cornerIndex(0, 0);
            const p1 = cornerIndex(1, 0);
            const p2 = cornerIndex(1, 1);
            const p3 = cornerIndex(0, 1);
            // (u x v) points along +axis, so counter-clockwise from outside is p0..p3 for the + face and reversed for the - face.
            if (sign > 0) indices.push(p0, p1, p2, p0, p2, p3);
            else indices.push(p0, p2, p1, p0, p3, p2);
        }
    }
    return { positions: new Float32Array(positions), indices: new Uint32Array(indices) };
}

/**
 * A closed thin-walled box (zero-thickness faces). Rays start inside, in
 * random directions: every one must hit a wall. Three copies check the same
 * thing through a rotated non-uniform scale and a mirror. Extra rays are
 * aimed at the box's corners, where three faces meet.
 */
function thinBoxCase(rayCount: number): OracleCase {
    const half: Vector3 = [2, 1.5, 1];
    const matrices = [IDENTITY, compose([10, 1, -4], 0.7, [1.5, 0.6, 2]), compose([-10, 0, 5], -1.1, [-1, 1.4, 0.8])];
    const instances: TraceSceneInstance[] = matrices.map((objectToWorld, instanceIndex) => ({ meshIndex: 0, instanceIndex, objectToWorld }));
    const random = createRandom(4242);
    const rays: TraceRay[] = [];
    // Interior points stay away from the walls so the shell, not the start point, is what the ray must find.
    const interior = (): Vector3 => [
        (random() * 2 - 1) * half[0] * 0.8,
        (random() * 2 - 1) * half[1] * 0.8,
        (random() * 2 - 1) * half[2] * 0.8,
    ];
    const direction = (): Vector3 => {
        const z = 1 - 2 * random();
        const angle = 2 * Math.PI * random();
        const radius = Math.sqrt(1 - z * z);
        return [radius * Math.cos(angle), radius * Math.sin(angle), z];
    };
    const randomCount = Math.floor(rayCount * 0.75);
    for (let i = 0; i < randomCount; i++) {
        const matrix = matrices[i % matrices.length];
        const origin = transformPoint(matrix, interior());
        // Directions are chosen in world space; the shell is closed, so any direction hits.
        rays.push(ray(origin, direction()));
    }
    // Aimed at the eight corners from inside, with a little spread so the edges next to each corner get hit too.
    for (let i = randomCount; i < rayCount; i++) {
        const matrix = matrices[i % matrices.length];
        const corner = i % 8;
        const target: Vector3 = [corner & 1 ? half[0] : -half[0], corner & 2 ? half[1] : -half[1], corner & 4 ? half[2] : -half[2]];
        const from = interior();
        rays.push(ray(transformPoint(matrix, from), sub(transformPoint(matrix, target), transformPoint(matrix, from))));
    }
    return {
        name: 'thin-box',
        description: 'Closed zero-thickness box, rays from inside in every direction (and at the corners): none may escape.',
        meshes: [boxMesh(half)],
        instances,
        rays: packRays(rays),
        everyRayMustHit: true,
        edgeAimedFrom: randomCount,
        // Corner and edge rays are within rounding distance of an edge by construction, so
        // which of the faces meeting there is hit first is a tie (check.ts holds a tie to
        // the same t and side). Any-hit has no ties: something blocks every ray, so it
        // is held to that and none of its rays may be ambiguous.
        maxAmbiguousShare: { closest: 0.3, any: 0 },
    };
}

function sub(a: Vector3, b: Vector3): Vector3 {
    return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

// ---------------------------------------------------------------------------
// Random stress scenes
// ---------------------------------------------------------------------------

function randomCase(
    name: string,
    description: string,
    meshes: MeshGeometry[],
    instances: TraceSceneInstance[],
    random: Random,
    extent: number,
    rayCount: number,
): OracleCase {
    const rays = randomRays(random, rayCount, [-extent, -extent, -extent, extent, extent, extent], extent * 0.4, { randomInterval: true });
    // Random rays and random intervals land within rounding distance of an edge or an interval end now and then.
    return { name, description, meshes, instances, rays: packRays(rays), maxAmbiguousShare: { closest: 0.01, any: 0.01 } };
}

function randomSoupCase(rayCount: number): OracleCase {
    const random = createRandom(101);
    return randomCase(
        'random-soup',
        'One mesh of 2000 random triangles, identity instance, random ray intervals.',
        [randomTriangleSoup(random, 2000, 10, 1)],
        [{ meshIndex: 0, instanceIndex: 0, objectToWorld: IDENTITY }],
        random,
        10,
        rayCount,
    );
}

function randomInstancedCase(rayCount: number): OracleCase {
    const random = createRandom(202);
    const meshes = [randomTriangleSoup(random, 600, 3, 0.8), randomTriangleSoup(random, 60, 2, 1), tessellatedQuad(6, 2), boxMesh([1, 2, 1.5])];
    const instances: TraceSceneInstance[] = Array.from({ length: 30 }, (_, i) => ({
        meshIndex: i % meshes.length,
        instanceIndex: 1000 - i * 7,
        objectToWorld: randomAffine(random, { extent: 14, mirror: i % 4 === 0 }),
    }));
    return randomCase(
        'random-instanced',
        'Four meshes in 30 randomly rotated, non-uniformly scaled and mirrored instances.',
        meshes,
        instances,
        random,
        16,
        rayCount,
    );
}

function manyInstancesCase(rayCount: number): OracleCase {
    const random = createRandom(303);
    const meshes = [randomTriangleSoup(random, 40, 1, 0.5)];
    const instances: TraceSceneInstance[] = Array.from({ length: 200 }, (_, i) => ({
        meshIndex: 0,
        instanceIndex: i,
        objectToWorld: randomAffine(random, { extent: 25, mirror: i % 5 === 0 }),
    }));
    return randomCase('many-instances', '200 copies of one 40-triangle mesh, for a deeper top-level tree.', meshes, instances, random, 28, rayCount);
}

function deepMeshCase(rayCount: number): OracleCase {
    const random = createRandom(404);
    return randomCase(
        'deep-mesh',
        'One mesh of 30000 small triangles, for a deeper mesh tree.',
        [randomTriangleSoup(random, 30000, 20, 0.35)],
        [{ meshIndex: 0, instanceIndex: 0, objectToWorld: compose([1, -2, 3], 0.3, [1, 1.3, 0.9]) }],
        random,
        22,
        rayCount,
    );
}

// ---------------------------------------------------------------------------
// Shared edges
// ---------------------------------------------------------------------------

export const SHARED_EDGE_CATEGORIES = [
    'exact vertical ray on a grid line',
    'oblique ray at a grid-line edge',
    'oblique ray at a diagonal edge',
    'oblique ray at an interior vertex',
] as const;

export interface SharedEdgeCase extends OracleCase {
    /** Index into {@link SHARED_EDGE_CATEGORIES} for each ray. */
    readonly categories: Uint8Array;
}

/**
 * Rays aimed at the seams of a tessellated quad. The quad is large and every
 * target is well inside it, so every ray must hit something: a miss is a ray
 * that slipped between two triangles.
 *
 * The quad is 8 x 8 cells spanning [-2, 2]^2 at z = 0; each cell is split along
 * one diagonal, so there are grid-line edges, diagonal edges and vertices where
 * six triangles meet. With `transformed`, the quad is instanced with a rotation,
 * non-uniform scale and mirror, so edges are not axis-aligned in world space
 * and the transformed ray's arithmetic is exercised too.
 */
export function sharedEdgeCase(rayCount: number, transformed: boolean): SharedEdgeCase {
    const cells = 8;
    const half = 2;
    const step = (2 * half) / cells;
    const random = createRandom(transformed ? 777 : 555);
    const objectToWorld = transformed ? tiltAboutX(compose([0.3, -0.2, 0.1], 0.6, [-1.3, 0.9, 1.1])) : IDENTITY;

    const rays: TraceRay[] = [];
    const categories = new Uint8Array(rayCount);
    const interiorGridLine = (): number => -half + step * (1 + Math.floor(random() * (cells - 1)));
    const insideCoordinate = (): number => (random() * 2 - 1) * (half - 0.2);

    for (let i = 0; i < rayCount; i++) {
        // The plain quad spends a tenth of its rays on exact vertical rays; the transformed one has none.
        const category = transformed ? 1 + (i % 3) : i % 10 === 0 ? 0 : 1 + (i % 3);
        categories[i] = category;

        let target: Vector3;
        if (category === 0 || category === 1) {
            // On a grid line: x or y is a multiple of the cell size, the other is anywhere along the line.
            target = random() < 0.5 ? [interiorGridLine(), insideCoordinate(), 0] : [insideCoordinate(), interiorGridLine(), 0];
        } else if (category === 2) {
            // Along the diagonal of a random cell.
            const column = Math.floor(random() * cells);
            const row = Math.floor(random() * cells);
            const along = 0.05 + random() * 0.9;
            target = [-half + step * (column + along), -half + step * (row + along), 0];
        } else {
            target = [interiorGridLine(), interiorGridLine(), 0];
        }

        const height = 0.5 + random() * 3;
        const above = random() < 0.5 ? 1 : -1;
        const origin: Vector3 =
            category === 0
                ? [target[0], target[1], above * height]
                : [target[0] + (random() * 2 - 1) * height * 2, target[1] + (random() * 2 - 1) * height * 2, above * height];

        const worldOrigin = transformPoint(objectToWorld, origin);
        const worldTarget = transformPoint(objectToWorld, target);
        rays.push(ray(worldOrigin, sub(worldTarget, worldOrigin)));
    }

    return {
        name: transformed ? 'shared-edges-transformed' : 'shared-edges',
        description: `${rayCount} rays aimed at the seams of a tessellated quad${transformed ? ' seen through a rotated, non-uniform, mirrored instance' : ''}.`,
        meshes: [tessellatedQuad(cells, half)],
        instances: [{ meshIndex: 0, instanceIndex: 5, objectToWorld: objectToWorld }],
        rays: packRays(rays),
        categories,
        everyRayMustHit: true,
        edgeAimedFrom: 0,
        // Not used: evaluateSharedEdgeCase (check.ts) counts slips directly and allows none.
        maxAmbiguousShare: { closest: 1, any: 0 },
    };
}

/** Adds a tilt about X to a composed transform, so the quad is not parallel to any world axis plane. */
function tiltAboutX(composed: number[]): number[] {
    const angle = 0.5;
    const c = Math.cos(angle);
    const s = Math.sin(angle);
    // tilt * composed, column-major; tilt rotates about X.
    const tilt = [1, 0, 0, 0, 0, c, s, 0, 0, -s, c, 0, 0, 0, 0, 1];
    const out = new Array<number>(16).fill(0);
    for (let column = 0; column < 4; column++) {
        for (let row = 0; row < 4; row++) {
            for (let k = 0; k < 4; k++) out[column * 4 + row] += tilt[k * 4 + row] * composed[column * 4 + k];
        }
    }
    return out;
}

// ---------------------------------------------------------------------------

export interface CaseSizes {
    /** Rays in each random case. */
    readonly randomRays: number;
    /** Rays in the thin-box case. */
    readonly thinBoxRays: number;
}

export const FULL_SIZES: CaseSizes = { randomRays: 20000, thinBoxRays: 24000 };
export const SMALL_SIZES: CaseSizes = { randomRays: 800, thinBoxRays: 1200 };

/** Every case except the shared-edge ones, which are built separately because of their size. */
export function buildOracleCases(sizes: CaseSizes = FULL_SIZES): OracleCase[] {
    return [
        analyticCase(),
        instancingCase(),
        thinWallCase(),
        thinBoxCase(sizes.thinBoxRays),
        randomSoupCase(sizes.randomRays),
        randomInstancedCase(sizes.randomRays),
        manyInstancesCase(sizes.randomRays),
        deepMeshCase(Math.min(sizes.randomRays, 3000)),
    ];
}
