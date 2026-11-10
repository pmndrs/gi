/**
 * A procedural tree for the T3.0 refit-decay spike (question 3), standing in
 * for T3.11's wind tree, which wasn't ready: a trunk, branches and leaf
 * quads, about 100,000 triangles, 3 m of trunk and branches reaching ~4.5 m.
 * Leaves dominate (95%), as on a real foliage asset, and they are the hard
 * case for a refit: many small, separate triangles moving on their own.
 *
 * `swayInto` writes the pose at time `seconds`. Three motions add up, all
 * made to look like wind, with amplitudes from a `SwayStrength`:
 * - the whole tree bends along the wind, growing with height squared;
 * - each branch bobs on its own phase and frequency, growing along it;
 * - each leaf's two outer corners flutter along the leaf's normal.
 * A leaf's four corners share their attach point's trunk and branch motion,
 * so a leaf moves as a piece and only flutters on top.
 */

export interface SwayStrength {
    readonly name: string;
    /** Sideways bend of the tree at trunk height (3 m), in metres. */
    readonly trunkMetres: number;
    /** Extra bob at a branch's tip, in metres. */
    readonly branchMetres: number;
    /** Flutter of a leaf's outer corners, in metres. */
    readonly leafMetres: number;
}

/** A light breeze, and a strong gust: the two ends a wind tree would see. */
export const SWAY_STRENGTHS: readonly SwayStrength[] = [
    { name: 'breeze', trunkMetres: 0.1, branchMetres: 0.08, leafMetres: 0.02 },
    { name: 'gust', trunkMetres: 0.4, branchMetres: 0.25, leafMetres: 0.04 },
];

/** A mesh that sways: its rest pose, and its pose at any time. */
export interface SwayingMesh {
    readonly name: string;
    readonly restPositions: Float32Array<ArrayBuffer>;
    readonly indices: Uint32Array<ArrayBuffer>;
    readonly triangleCount: number;
    /** A box around every pose of the strongest sway: `[minX, minY, minZ, maxX, maxY, maxZ]`. */
    readonly swayBounds: Float32Array;
    /** Writes the pose at `seconds` into `out` (xyz per vertex). */
    swayInto(out: Float32Array, seconds: number, strength: SwayStrength): void;
}

const TRUNK_HEIGHT = 3;
const BRANCH_COUNT = 60;
const LEAVES_PER_BRANCH = 790;
const LEAF_LENGTH = 0.08;
const LEAF_WIDTH = 0.05;

/** A small seeded random generator, so every run builds the same tree. */
function seededRandom(seed: number): () => number {
    let state = seed >>> 0;
    return () => {
        state = (state * 1664525 + 1013904223) >>> 0;
        return state / 2 ** 32;
    };
}

type Vec3 = [number, number, number];

function normalize(v: Vec3): Vec3 {
    const length = Math.hypot(v[0], v[1], v[2]) || 1;
    return [v[0] / length, v[1] / length, v[2] / length];
}

function cross(a: Vec3, b: Vec3): Vec3 {
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}

/** Two unit vectors perpendicular to `axis` and to each other. */
function frameAround(axis: Vec3): [Vec3, Vec3] {
    const helper: Vec3 = Math.abs(axis[1]) < 0.9 ? [0, 1, 0] : [1, 0, 0];
    const side = normalize(cross(axis, helper));
    return [side, cross(axis, side)];
}

export function makeProceduralTree(seed = 7): SwayingMesh {
    const random = seededRandom(seed);
    const positions: number[] = [];
    const indices: number[] = [];
    // Per vertex, what drives its motion.
    const heights: number[] = []; // height of the point the vertex follows (its attach point for leaves)
    const branchOf: number[] = []; // -1 for the trunk
    const alongBranch: number[] = []; // 0 at the branch's base, 1 at its tip
    const flutterAmount: number[] = []; // 1 for a leaf's outer corners, else 0
    const flutterPhase: number[] = [];
    const flutterNormal: number[] = [];

    const addVertex = (p: Vec3, height: number, branch: number, along: number, flutter = 0, phase = 0, normal: Vec3 = [0, 0, 0]): number => {
        positions.push(...p);
        heights.push(height);
        branchOf.push(branch);
        alongBranch.push(along);
        flutterAmount.push(flutter);
        flutterPhase.push(phase);
        flutterNormal.push(...normal);
        return heights.length - 1;
    };

    /** A tapered tube from `base` along `axis`; `ring` index 0 at the base. */
    const addTube = (base: Vec3, axis: Vec3, length: number, radiusBase: number, radiusTip: number, sides: number, rings: number, branch: number): void => {
        const [u, v] = frameAround(axis);
        const first = heights.length;
        for (let ring = 0; ring <= rings; ring++) {
            const along = ring / rings;
            const radius = radiusBase + (radiusTip - radiusBase) * along;
            for (let side = 0; side < sides; side++) {
                const angle = (side / sides) * Math.PI * 2;
                const p: Vec3 = [0, 1, 2].map((axisIndex) => base[axisIndex] + axis[axisIndex] * length * along + (u[axisIndex] * Math.cos(angle) + v[axisIndex] * Math.sin(angle)) * radius) as Vec3;
                addVertex(p, branch < 0 ? p[1] : base[1], branch, branch < 0 ? 0 : along);
            }
        }
        for (let ring = 0; ring < rings; ring++) {
            for (let side = 0; side < sides; side++) {
                const a = first + ring * sides + side;
                const b = first + ring * sides + ((side + 1) % sides);
                indices.push(a, b, a + sides, b, b + sides, a + sides);
            }
        }
    };

    addTube([0, 0, 0], [0, 1, 0], TRUNK_HEIGHT, 0.15, 0.08, 12, 24, -1);

    const branchPhases: number[] = [];
    const branchFrequencies: number[] = [];
    const branchSwayDirections: Vec3[] = [];
    for (let branch = 0; branch < BRANCH_COUNT; branch++) {
        const baseHeight = 1.5 + random() * 1.5;
        const angle = random() * Math.PI * 2;
        const axis = normalize([Math.cos(angle), 0.4 + random() * 0.8, Math.sin(angle)]);
        const length = 1 + random();
        const base: Vec3 = [0, baseHeight, 0];
        addTube(base, axis, length, 0.04, 0.01, 6, 6, branch);
        branchPhases.push(random() * Math.PI * 2);
        branchFrequencies.push(2 * Math.PI * (0.8 + random() * 0.6));
        // Branches bob mostly up and down, a little sideways.
        branchSwayDirections.push(normalize([random() - 0.5, 1.5, random() - 0.5]));

        for (let leaf = 0; leaf < LEAVES_PER_BRANCH; leaf++) {
            const along = 0.15 + 0.85 * random();
            const [u, v] = frameAround(axis);
            const spread = 0.4 * along;
            const offsetAngle = random() * Math.PI * 2;
            const offset = spread * Math.sqrt(random());
            const attach: Vec3 = [0, 1, 2].map((i) => base[i] + axis[i] * length * along + (u[i] * Math.cos(offsetAngle) + v[i] * Math.sin(offsetAngle)) * offset) as Vec3;
            // The leaf hangs from its attach point in a random direction, facing a random way.
            const outward = normalize([random() - 0.5, random() - 0.7, random() - 0.5]);
            const normal = normalize(cross(outward, frameAround(outward)[0]));
            const across = normalize(cross(normal, outward));
            const phase = random() * Math.PI * 2;
            const corner = (forward: number, sideways: number): Vec3 => [0, 1, 2].map((i) => attach[i] + outward[i] * forward * LEAF_LENGTH + across[i] * sideways * LEAF_WIDTH * 0.5) as Vec3;
            const a = addVertex(corner(0, -1), base[1], branch, along);
            const b = addVertex(corner(0, 1), base[1], branch, along);
            const c = addVertex(corner(1, 1), base[1], branch, along, 1, phase, normal);
            const d = addVertex(corner(1, -1), base[1], branch, along, 1, phase, normal);
            indices.push(a, b, c, a, c, d);
        }
    }

    const restPositions = new Float32Array(positions);
    const vertexCount = heights.length;
    const height = new Float32Array(heights);
    const branchIndex = new Int32Array(branchOf);
    const along = new Float32Array(alongBranch);
    const flutter = new Float32Array(flutterAmount);
    const phase = new Float32Array(flutterPhase);
    const normals = new Float32Array(flutterNormal);
    const leafFrequencies = new Float32Array(vertexCount);
    for (let vertex = 0; vertex < vertexCount; vertex++) leafFrequencies[vertex] = 2 * Math.PI * (3 + (phase[vertex] / (Math.PI * 2)) * 3);

    // Strongest sway's reach, for a top-level box that holds every pose.
    const strongest = SWAY_STRENGTHS[SWAY_STRENGTHS.length - 1];
    const reach = strongest.trunkMetres * 2.5 + strongest.branchMetres + strongest.leafMetres + 0.1;
    const swayBounds = new Float32Array([Infinity, Infinity, Infinity, -Infinity, -Infinity, -Infinity]);
    for (let vertex = 0; vertex < vertexCount; vertex++) {
        for (let axis = 0; axis < 3; axis++) {
            swayBounds[axis] = Math.min(swayBounds[axis], restPositions[vertex * 3 + axis] - reach);
            swayBounds[axis + 3] = Math.max(swayBounds[axis + 3], restPositions[vertex * 3 + axis] + reach);
        }
    }

    const windX = 0.8;
    const windZ = 0.6;
    return {
        name: 'procedural tree',
        restPositions,
        indices: new Uint32Array(indices),
        triangleCount: indices.length / 3,
        swayBounds,
        swayInto(out, seconds, strength) {
            // The whole tree's bend: two slow waves, mostly along the wind.
            const bendAlong = strength.trunkMetres * Math.sin(2 * Math.PI * 0.4 * seconds);
            const bendAcross = 0.3 * strength.trunkMetres * Math.sin(2 * Math.PI * 0.27 * seconds + 1);
            const branchSway = branchPhases.map((branchPhase, branch) => strength.branchMetres * Math.sin(branchFrequencies[branch] * seconds + branchPhase));
            for (let vertex = 0; vertex < vertexCount; vertex++) {
                const relativeHeight = height[vertex] / TRUNK_HEIGHT;
                const bend = relativeHeight * relativeHeight;
                let x = restPositions[vertex * 3] + bend * (bendAlong * windX - bendAcross * windZ);
                let y = restPositions[vertex * 3 + 1];
                let z = restPositions[vertex * 3 + 2] + bend * (bendAlong * windZ + bendAcross * windX);
                const branch = branchIndex[vertex];
                if (branch >= 0) {
                    const amount = branchSway[branch] * along[vertex] * along[vertex];
                    const direction = branchSwayDirections[branch];
                    x += direction[0] * amount;
                    y += direction[1] * amount;
                    z += direction[2] * amount;
                }
                if (flutter[vertex] > 0) {
                    const amount = strength.leafMetres * Math.sin(leafFrequencies[vertex] * seconds + phase[vertex]);
                    x += normals[vertex * 3] * amount;
                    y += normals[vertex * 3 + 1] * amount;
                    z += normals[vertex * 3 + 2] * amount;
                }
                out[vertex * 3] = x;
                out[vertex * 3 + 1] = y;
                out[vertex * 3 + 2] = z;
            }
        },
    };
}
