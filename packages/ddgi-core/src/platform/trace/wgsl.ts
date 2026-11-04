/**
 * WGSL for the `rays` query backend: `giTraceClosest` and `giTraceAny` over
 * the two-level BVH, plus a batch kernel that traces a buffer of rays.
 *
 * The source is a TypeScript string (not a `.wgsl` file) so the layout
 * constants from `layout.ts` are spliced in and the shader can never disagree
 * with the builders about a word offset.
 *
 * The walk, in one paragraph. A single loop and one small stack serve both
 * levels. The top-level tree has one leaf per instance. Reaching an instance
 * leaf pushes a marker, moves the ray into that instance's object space, and
 * continues at the root of the instance's mesh tree. The mesh tree's leaves
 * are triangles. When the marker is popped, the mesh tree is finished and the
 * world-space ray is restored. At every node both child boxes are tested and
 * the nearer one is visited first; the farther one is pushed. Boxes that start
 * beyond the closest hit found so far are skipped.
 *
 * Distances stay in world units because the ray's direction is *not*
 * renormalized after the transform into object space: a point at parameter
 * `t` on the object-space ray is the same point as at `t` on the world ray,
 * whatever scale, shear or mirror the instance has.
 */

import type { BindingSlot, ModuleBindingEntry } from '../contract/wgsl.ts';
import {
    INSTANCE_FLAG,
    EMPTY_CHILD_INDEX,
    REF_COUNT_SHIFT,
    REF_INDEX_MASK,
    REF_INSTANCE_MARKER,
    TOP_LEVEL_HEADER_BYTES,
    INSTANCE_RECORD_BYTES,
    BVH_NODE_BYTES,
    DEFAULT_TRIANGLE_TEST,
    TRACE_STACK_SIZE,
    type TriangleTest,
} from './layout.ts';

/** `GPUShaderStage.COMPUTE`. The global does not exist in Node, where the tests import this file. */
const SHADER_STAGE_COMPUTE = 0x4;

/** Bindings the trace module declares, in order. All are read-only storage buffers. */
export const TRACE_MODULE_ENTRIES: readonly ModuleBindingEntry[] = [
    // BVH nodes of every mesh.
    { visibility: SHADER_STAGE_COMPUTE, buffer: { type: 'read-only-storage' } },
    // Triangles of every mesh, in tree order.
    { visibility: SHADER_STAGE_COMPUTE, buffer: { type: 'read-only-storage' } },
    // Top level: header, instance records, top-level nodes.
    { visibility: SHADER_STAGE_COMPUTE, buffer: { type: 'read-only-storage' } },
];

const hex = (value: number): string => `0x${value.toString(16)}u`;

/** The shared structs and constants, written once so the batch kernel reuses them. */
const COMMON_WGSL = /* wgsl */ `
// ---- Contract types (see docs/api/platform.md) ------------------------------

struct GiRay {
    origin: vec3f,
    tMin: f32,
    direction: vec3f,   // normalized
    tMax: f32,
}

struct GiHit {
    t: f32,              // distance along the ray; negative means no hit
    instance: u32,       // platform instance index
    triangle: u32,       // triangle index within the instance's mesh
    barycentrics: vec2f, // weights of vertices 1 and 2
    frontFace: bool,     // false when the ray hit the back side
}

// ---- Memory layout (generated from layout.ts) -------------------------------

// One BVH node holds both child boxes. A child with count > 0 is a leaf of
// 'count' primitives starting at 'index'; count == 0 means an inner node at 'index'.
struct GiBvhNode {
    leftMin: vec3f,  leftIndex: u32,
    leftMax: vec3f,  rightIndex: u32,
    rightMin: vec3f, leftCount: u32,
    rightMax: vec3f, rightCount: u32,
}

// A triangle is a corner plus two more vectors. What those are depends on the
// triangle test: for Moller-Trumbore they are the edges v1 - v0 and v2 - v0;
// for the watertight test they are the corners v1 and v2 themselves.
struct GiBvhTriangle {
    vertex0: vec3f,      originalIndex: u32,
    secondVector: vec3f, flags: u32,
    thirdVector: vec3f,  padding: u32,
}

// The world-to-object transform is a 3x4 matrix stored as three rows.
struct GiInstance {
    worldToObjectRow0: vec4f,
    worldToObjectRow1: vec4f,
    worldToObjectRow2: vec4f,
    rootNodeOffset: u32,   // where this mesh's nodes start, in nodes
    triangleOffset: u32,   // where this mesh's triangles start, in triangles
    instanceIndex: u32,    // platform instance index
    flags: u32,
}

const GI_INSTANCE_FLAG_MIRRORED: u32 = ${INSTANCE_FLAG.mirrored}u;

// The top-level buffer is [header][instance records][top-level nodes], and is
// read as vec4u so no integer ever passes through a float register.
const GI_TOP_VEC4_PER_RECORD: u32 = ${INSTANCE_RECORD_BYTES / 16}u;
const GI_TOP_VEC4_PER_NODE: u32 = ${BVH_NODE_BYTES / 16}u;
const GI_TOP_HEADER_VEC4S: u32 = ${TOP_LEVEL_HEADER_BYTES / 16}u;

// ---- Traversal constants -----------------------------------------------------

// Stack entries are child references: the index in the low bits, the leaf
// primitive count in the top bits (0 means an inner node).
const GI_REF_COUNT_SHIFT: u32 = ${REF_COUNT_SHIFT}u;
const GI_REF_INDEX_MASK: u32 = ${hex(REF_INDEX_MASK)};
// Pushed on entering an instance. Cannot collide with a real reference: leaves hold at most 8 primitives.
const GI_REF_INSTANCE_MARKER: u32 = ${hex(REF_INSTANCE_MARKER)};
// The right child of a node in a tree with a single primitive.
const GI_EMPTY_CHILD: u32 = ${hex(EMPTY_CHILD_INDEX)};

// Both levels together must fit; the builder checks depth against this.
const GI_STACK_SIZE: u32 = ${TRACE_STACK_SIZE}u;

// Returned by the box test for "no hit". Far larger than any scene distance.
const GI_BOX_MISS: f32 = 3.0e38;

// Direction components smaller than this are treated as this, so 1/direction
// stays finite and 0 * (1/direction) can never produce a NaN.
const GI_MIN_DIRECTION: f32 = 1.0e-30;

// Rounding in (box - origin) * (1 / direction) can put a ray that touches a
// box a few ulps outside its slab interval. Growing the exit distance by this
// much keeps such rays (Ize, "Robust BVH Ray Traversal"). About 3 f32 steps.
const GI_BOX_EXIT_PADDING: f32 = 1.0000004;
`;

/**
 * The trace module: contract types, layout, and `giTraceClosest` / `giTraceAny`.
 * Bindings land at `slot`: nodes, triangles, top level, in that order.
 *
 * `triangleTest` must match what the meshes were built for (`MeshBvh.triangleTest`).
 */
export function traceModuleWgsl(slot: BindingSlot, triangleTest: TriangleTest = DEFAULT_TRIANGLE_TEST): string {
    const group = slot.group;
    const nodes = slot.firstBinding;
    const triangles = slot.firstBinding + 1;
    const topLevel = slot.firstBinding + 2;
    return /* wgsl */ `${COMMON_WGSL}
// Which ray-triangle test the stored triangles are for. A constant, so the other test compiles away.
const GI_WATERTIGHT: bool = ${triangleTest === 'watertight'};

@group(${group}) @binding(${nodes}) var<storage, read> giTraceNodes: array<GiBvhNode>;
@group(${group}) @binding(${triangles}) var<storage, read> giTraceTriangles: array<GiBvhTriangle>;
@group(${group}) @binding(${topLevel}) var<storage, read> giTraceTopLevel: array<vec4u>;

fn giMakeReference(index: u32, count: u32) -> u32 {
    return index | (count << GI_REF_COUNT_SHIFT);
}

// ---- Loading from the top-level buffer ---------------------------------------

fn giLoadInstance(recordIndex: u32) -> GiInstance {
    let base = GI_TOP_HEADER_VEC4S + recordIndex * GI_TOP_VEC4_PER_RECORD;
    let row0 = giTraceTopLevel[base];
    let row1 = giTraceTopLevel[base + 1u];
    let row2 = giTraceTopLevel[base + 2u];
    let ids = giTraceTopLevel[base + 3u];
    return GiInstance(
        bitcast<vec4f>(row0), bitcast<vec4f>(row1), bitcast<vec4f>(row2),
        ids.x, ids.y, ids.z, ids.w,
    );
}

// 'firstNode' is the vec4 index where the top-level nodes begin.
fn giLoadTopLevelNode(firstNode: u32, nodeIndex: u32) -> GiBvhNode {
    let base = firstNode + nodeIndex * GI_TOP_VEC4_PER_NODE;
    let a = giTraceTopLevel[base];
    let b = giTraceTopLevel[base + 1u];
    let c = giTraceTopLevel[base + 2u];
    let d = giTraceTopLevel[base + 3u];
    return GiBvhNode(
        bitcast<vec3f>(a.xyz), a.w,
        bitcast<vec3f>(b.xyz), b.w,
        bitcast<vec3f>(c.xyz), c.w,
        bitcast<vec3f>(d.xyz), d.w,
    );
}

// ---- Rays ----------------------------------------------------------------------------

// A ray in the space of the level being walked (world space at the top level,
// an instance's object space inside it), with everything the tests reuse
// precomputed once instead of per box or per triangle.
struct GiRaySetup {
    origin: vec3f,
    direction: vec3f,          // object space: not normalized, so t stays in world units
    inverseDirection: vec3f,   // for the box test
    shear: vec3f,              // watertight triangle test: (x, y, z) shear that points the ray along +z'
    axes: vec3u,               // watertight triangle test: which axes are x', y' and z'
}

fn giSafeInverse(direction: vec3f) -> vec3f {
    let magnitude = max(abs(direction), vec3f(GI_MIN_DIRECTION));
    let directionSign = select(vec3f(1.0), vec3f(-1.0), direction < vec3f(0.0));
    return 1.0 / (magnitude * directionSign);
}

fn giMakeRaySetup(origin: vec3f, direction: vec3f) -> GiRaySetup {
    // The axis the ray runs along most becomes z'. Swapping x' and y' when it
    // points the negative way keeps triangle winding, so the sign of the
    // determinant still tells front from back.
    let magnitude = abs(direction);
    var kz = 2u;
    if (magnitude.x > magnitude.y && magnitude.x > magnitude.z) {
        kz = 0u;
    } else if (magnitude.y > magnitude.z) {
        kz = 1u;
    }
    var kx = (kz + 1u) % 3u;
    var ky = (kx + 1u) % 3u;
    if (direction[kz] < 0.0) {
        let swapped = kx;
        kx = ky;
        ky = swapped;
    }
    let shear = vec3f(direction[kx] / direction[kz], direction[ky] / direction[kz], 1.0 / direction[kz]);
    return GiRaySetup(origin, direction, giSafeInverse(direction), shear, vec3u(kx, ky, kz));
}

fn giTransformPoint(instance: GiInstance, point: vec3f) -> vec3f {
    return vec3f(
        dot(instance.worldToObjectRow0.xyz, point) + instance.worldToObjectRow0.w,
        dot(instance.worldToObjectRow1.xyz, point) + instance.worldToObjectRow1.w,
        dot(instance.worldToObjectRow2.xyz, point) + instance.worldToObjectRow2.w,
    );
}

// Deliberately not normalized afterwards: see the note at the top of this file.
fn giTransformDirection(instance: GiInstance, direction: vec3f) -> vec3f {
    return vec3f(
        dot(instance.worldToObjectRow0.xyz, direction),
        dot(instance.worldToObjectRow1.xyz, direction),
        dot(instance.worldToObjectRow2.xyz, direction),
    );
}

// ---- Intersection tests ----------------------------------------------------------

// Slab test. Returns the distance at which the ray enters the box, or
// GI_BOX_MISS when it does not touch the box within [tMin, tMax].
fn giBoxEntry(boxMin: vec3f, boxMax: vec3f, setup: GiRaySetup, tMin: f32, tMax: f32) -> f32 {
    let toMin = (boxMin - setup.origin) * setup.inverseDirection;
    let toMax = (boxMax - setup.origin) * setup.inverseDirection;
    let nearSides = min(toMin, toMax);
    let farSides = max(toMin, toMax);
    let tEnter = max(max(nearSides.x, max(nearSides.y, nearSides.z)), tMin);
    let tExit = min(min(farSides.x, min(farSides.y, farSides.z)) * GI_BOX_EXIT_PADDING, tMax);
    return select(GI_BOX_MISS, tEnter, tEnter <= tExit);
}

struct GiTriangleHit {
    hit: bool,
    t: f32,
    barycentrics: vec2f,   // weights of vertices 1 and 2
    // True when the ray runs against the triangle's winding normal, in the
    // space the test ran in (object space).
    frontFacing: bool,
}

fn giNoTriangleHit() -> GiTriangleHit {
    return GiTriangleHit(false, 0.0, vec2f(0.0), true);
}

// Moller-Trumbore. Triangles store 'vertex0' and two edges. No epsilon on the
// edge tests. The direction need not be unit length; t is then in units of
// that direction. Not watertight: a ray aimed at a shared edge can pass
// between the two triangles (see GI_WATERTIGHT).
fn giIntersectMollerTrumbore(triangle: GiBvhTriangle, setup: GiRaySetup, tMin: f32, tMax: f32) -> GiTriangleHit {
    let edge1 = triangle.secondVector;
    let edge2 = triangle.thirdVector;
    let pVector = cross(setup.direction, edge2);
    let determinant = dot(edge1, pVector);
    if (determinant == 0.0) {
        return giNoTriangleHit();   // parallel to the triangle's plane
    }
    let inverseDeterminant = 1.0 / determinant;

    let toOrigin = setup.origin - triangle.vertex0;
    let u = dot(toOrigin, pVector) * inverseDeterminant;
    if (u < 0.0 || u > 1.0) {
        return giNoTriangleHit();
    }
    let qVector = cross(toOrigin, edge1);
    let v = dot(setup.direction, qVector) * inverseDeterminant;
    if (v < 0.0 || u + v > 1.0) {
        return giNoTriangleHit();
    }
    let t = dot(edge2, qVector) * inverseDeterminant;
    if (t < tMin || t > tMax) {
        return giNoTriangleHit();
    }
    return GiTriangleHit(true, t, vec2f(u, v), determinant > 0.0);
}

// Woop, Benthin and Wald, "Watertight Ray/Triangle Intersection" (2013).
// Triangles store three corners. The three edge functions are computed from
// the same corner bits for both triangles that share an edge, with opposite
// signs, so a ray cannot pass between them. A zero counts as inside for both,
// so a ray exactly on an edge hits both neighbours rather than neither.
fn giIntersectWatertight(triangle: GiBvhTriangle, setup: GiRaySetup, tMin: f32, tMax: f32) -> GiTriangleHit {
    let kx = setup.axes.x;
    let ky = setup.axes.y;
    let kz = setup.axes.z;

    // Corners relative to the ray origin.
    let a = triangle.vertex0 - setup.origin;
    let b = triangle.secondVector - setup.origin;
    let c = triangle.thirdVector - setup.origin;

    // Shear and scale into the space where the ray runs along +z' through the origin.
    let ax = a[kx] - setup.shear.x * a[kz];
    let ay = a[ky] - setup.shear.y * a[kz];
    let bx = b[kx] - setup.shear.x * b[kz];
    let by = b[ky] - setup.shear.y * b[kz];
    let cx = c[kx] - setup.shear.x * c[kz];
    let cy = c[ky] - setup.shear.y * c[kz];

    // Edge functions: the (scaled) barycentric weights of corners 0, 1 and 2.
    let weight0 = cx * by - cy * bx;
    let weight1 = ax * cy - ay * cx;
    let weight2 = bx * ay - by * ax;
    if ((weight0 < 0.0 || weight1 < 0.0 || weight2 < 0.0) && (weight0 > 0.0 || weight1 > 0.0 || weight2 > 0.0)) {
        return giNoTriangleHit();   // the weights disagree in sign: outside the triangle
    }

    let determinant = weight0 + weight1 + weight2;
    if (determinant == 0.0) {
        return giNoTriangleHit();   // edge-on
    }
    let scaledT = weight0 * (setup.shear.z * a[kz]) + weight1 * (setup.shear.z * b[kz]) + weight2 * (setup.shear.z * c[kz]);
    let inverseDeterminant = 1.0 / determinant;
    let t = scaledT * inverseDeterminant;
    if (t < tMin || t > tMax) {
        return giNoTriangleHit();
    }
    // With the axis swap in giMakeRaySetup the determinant is positive for a front hit.
    return GiTriangleHit(true, t, vec2f(weight1, weight2) * inverseDeterminant, determinant > 0.0);
}

fn giIntersectTriangle(triangle: GiBvhTriangle, setup: GiRaySetup, tMin: f32, tMax: f32) -> GiTriangleHit {
    if (GI_WATERTIGHT) {
        return giIntersectWatertight(triangle, setup, tMin, tMax);
    }
    return giIntersectMollerTrumbore(triangle, setup, tMin, tMax);
}

// ---- The walk -----------------------------------------------------------------------

// With stopAtFirstHit the walk returns on the first triangle found in
// [tMin, tMax] (t >= 0 in the result means "something blocks the ray").
// Without it, the result is the nearest hit.
fn giTraverse(ray: GiRay, stopAtFirstHit: bool) -> GiHit {
    var result = GiHit(-1.0, 0u, 0u, vec2f(0.0), true);

    let header = giTraceTopLevel[0];
    let instanceCount = header.x;
    let topLevelNodeCount = header.y;
    if (topLevelNodeCount == 0u) {
        return result;
    }
    let firstTopLevelNode = GI_TOP_HEADER_VEC4S + instanceCount * GI_TOP_VEC4_PER_RECORD;

    let worldSetup = giMakeRaySetup(ray.origin, ray.direction);
    var setup = worldSetup;

    var insideInstance = false;
    var instance: GiInstance;
    var mirrored = false;

    // Hits farther than this are not interesting any more.
    var closestT = ray.tMax;

    var stack: array<u32, GI_STACK_SIZE>;
    var stackSize = 0u;
    var current = giMakeReference(0u, 0u);   // the root of the top level

    loop {
        let count = current >> GI_REF_COUNT_SHIFT;
        let index = current & GI_REF_INDEX_MASK;

        if (count == 0u) {
            // An inner node: test both children, go to the nearer one first.
            var node: GiBvhNode;
            if (insideInstance) {
                node = giTraceNodes[instance.rootNodeOffset + index];
            } else {
                node = giLoadTopLevelNode(firstTopLevelNode, index);
            }

            // Only a tree with a single primitive has an empty child, and it is the right one.
            let leftEntry = giBoxEntry(node.leftMin, node.leftMax, setup, ray.tMin, closestT);
            var rightEntry = GI_BOX_MISS;
            if (node.rightIndex != GI_EMPTY_CHILD) {
                rightEntry = giBoxEntry(node.rightMin, node.rightMax, setup, ray.tMin, closestT);
            }
            let leftReference = giMakeReference(node.leftIndex, node.leftCount);
            let rightReference = giMakeReference(node.rightIndex, node.rightCount);
            let hitLeft = leftEntry < GI_BOX_MISS;
            let hitRight = rightEntry < GI_BOX_MISS;

            if (hitLeft && hitRight) {
                if (leftEntry <= rightEntry) {
                    stack[stackSize] = rightReference;
                    current = leftReference;
                } else {
                    stack[stackSize] = leftReference;
                    current = rightReference;
                }
                stackSize += 1u;
                continue;
            }
            if (hitLeft) {
                current = leftReference;
                continue;
            }
            if (hitRight) {
                current = rightReference;
                continue;
            }
            // Neither child is hit: fall through to pop.
        } else if (!insideInstance) {
            // A top-level leaf is one instance. Move the ray into its object space
            // and start at the root of its mesh tree.
            instance = giLoadInstance(index);
            mirrored = (instance.flags & GI_INSTANCE_FLAG_MIRRORED) != 0u;
            setup = giMakeRaySetup(
                giTransformPoint(instance, ray.origin),
                giTransformDirection(instance, ray.direction),
            );
            stack[stackSize] = GI_REF_INSTANCE_MARKER;
            stackSize += 1u;
            insideInstance = true;
            current = giMakeReference(0u, 0u);
            continue;
        } else {
            // A mesh leaf: 'count' triangles starting at 'index'.
            for (var i = 0u; i < count; i += 1u) {
                let triangle = giTraceTriangles[instance.triangleOffset + index + i];
                let candidate = giIntersectTriangle(triangle, setup, ray.tMin, closestT);
                if (candidate.hit) {
                    // A mirrored instance flips the winding as seen in world space.
                    let frontFace = candidate.frontFacing != mirrored;
                    result = GiHit(candidate.t, instance.instanceIndex, triangle.originalIndex, candidate.barycentrics, frontFace);
                    closestT = candidate.t;
                    if (stopAtFirstHit) {
                        return result;
                    }
                }
            }
        }

        // Nothing left to descend into here: take the next entry off the stack.
        loop {
            if (stackSize == 0u) {
                return result;
            }
            stackSize -= 1u;
            let entry = stack[stackSize];
            if (entry == GI_REF_INSTANCE_MARKER) {
                // The instance's tree is finished; go back to the world-space ray.
                insideInstance = false;
                setup = worldSetup;
                continue;
            }
            current = entry;
            break;
        }
    }
    // No return here: the loop above only exits through 'return'.
}

// Nearest hit, honouring alpha cutouts and blocksLight (cutouts arrive in M4).
fn giTraceClosest(ray: GiRay) -> GiHit {
    return giTraverse(ray, false);
}

// True if anything blocks the ray. Stops at the first blocker found.
fn giTraceAny(ray: GiRay) -> bool {
    return giTraverse(ray, true).t >= 0.0;
}
`;
}

/**
 * What a batch writes per ray in closest-hit mode. `GiHit` itself cannot live
 * in a buffer because `bool` is not host-shareable in WGSL, so `frontFace`
 * is stored as 0 or 1.
 */
export const HIT_RECORD_BYTES = 24;

/** Workgroup size of the batch kernels. */
export const BATCH_WORKGROUP_SIZE = 64;

/**
 * Batch kernel: traces `arrayLength(giBatchRays)` rays and writes one result
 * per ray. The trace module sits in group 0; the batch's own buffers in group 1.
 *
 * The ray count is the size of the bound `rays` range, so there is no uniform
 * to keep in step between several dispatches recorded into one command buffer.
 */
export function batchKernelWgsl(
    mode: 'closest' | 'any',
    triangleTest: TriangleTest = DEFAULT_TRIANGLE_TEST,
    traceSlot: BindingSlot = { group: 0, firstBinding: 0 },
): string {
    const resultDeclaration =
        mode === 'closest'
            ? `struct GiHitRecord {
    t: f32,
    instance: u32,
    triangle: u32,
    frontFace: u32,   // 0 or 1
    barycentrics: vec2f,
}

@group(1) @binding(1) var<storage, read_write> giBatchResults: array<GiHitRecord>;`
            : `// 1 when the ray is blocked, 0 when it is clear.
@group(1) @binding(1) var<storage, read_write> giBatchResults: array<u32>;`;

    const body =
        mode === 'closest'
            ? `let hit = giTraceClosest(giBatchRays[rayIndex]);
    giBatchResults[rayIndex] = GiHitRecord(hit.t, hit.instance, hit.triangle, select(0u, 1u, hit.frontFace), hit.barycentrics);`
            : `giBatchResults[rayIndex] = select(0u, 1u, giTraceAny(giBatchRays[rayIndex]));`;

    return /* wgsl */ `${traceModuleWgsl(traceSlot, triangleTest)}
@group(1) @binding(0) var<storage, read> giBatchRays: array<GiRay>;
${resultDeclaration}

@compute @workgroup_size(${BATCH_WORKGROUP_SIZE})
fn giBatch(
    @builtin(workgroup_id) workgroup: vec3u,
    @builtin(num_workgroups) workgroupCount: vec3u,
    @builtin(local_invocation_index) localIndex: u32,
) {
    // Dispatches are 2D so a very large batch does not exceed 65535 workgroups in x.
    let rayIndex = (workgroup.y * workgroupCount.x + workgroup.x) * ${BATCH_WORKGROUP_SIZE}u + localIndex;
    if (rayIndex >= arrayLength(&giBatchRays)) {
        return;
    }
    ${body}
}
`;
}
