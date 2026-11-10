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
 * world-space ray is restored. At every node both child boxes are tested; one
 * is visited and the other pushed (closest-hit: the nearer first; any-hit: the
 * farther first, see `walkWgsl`). Boxes that start beyond the closest hit
 * found so far are skipped.
 *
 * Distances stay in world units because the ray's direction is *not*
 * renormalized after the transform into object space: a point at parameter
 * `t` on the object-space ray is the same point as at `t` on the world ray,
 * whatever scale, shear or mirror the instance has.
 */

import type { BindingSlot, ModuleBindingEntry } from '../contract/wgsl.ts';
import {
    EMPTY_CHILD_INDEX,
    REF_COUNT_SHIFT,
    REF_INDEX_MASK,
    REF_INSTANCE_MARKER,
    TOP_LEVEL_HEADER_BYTES,
    INSTANCE_RECORD_BYTES,
    BVH_NODE_BYTES,
    TRIANGLE_BYTES,
    TRACE_STACK_SIZE,
} from './layout.ts';
import { NO_MATERIALS_WGSL, TRIANGLE_FLAG_CONSULT_MATERIAL } from './material-test-wgsl.ts';

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
    frontFace: bool,     // true on the side the mesh was authored as front (also when mirrored)
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

// A triangle is its three corners, stored as they are so that a corner shared
// by two triangles has the same bits in both (the watertight test needs that).
struct GiBvhTriangle {
    vertex0: vec3f, originalIndex: u32,
    vertex1: vec3f, flags: u32,
    vertex2: vec3f, padding: u32,
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

// Mesh trees and the top level use the same 64-byte node.
const GI_VEC4_PER_NODE: u32 = ${BVH_NODE_BYTES / 16}u;

// The top-level buffer is [header][instance records][top-level nodes], and is
// read as vec4u so no integer ever passes through a float register.
const GI_TOP_VEC4_PER_RECORD: u32 = ${INSTANCE_RECORD_BYTES / 16}u;
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
 * The walk, written once and generated as two functions:
 *
 * - `giWalkClosest` keeps the nearest hit and, after each one, skips boxes
 *   that start beyond it;
 * - `giWalkAny` returns at the first triangle in [tMin, tMax], so it needs no
 *   hit record at all, and it visits the *farther* child first.
 *
 * Why any-hit goes far first (T1.3, measured on shadow rays toward the sun):
 * a shadow ray starts on a surface, so the nearer child is usually the
 * geometry it starts on, which can't block it. What blocks it (a ceiling, a
 * roof, a wall across the room) lies ahead, in the farther child. Any blocker
 * will do, so there is nothing to gain from finding the nearest one first.
 * On the bench scenes far-first made any-hit much faster than near-first, with
 * every answer unchanged (numbers: PROGRESS, T1.3, and bench/results/trace.csv).
 */
function walkWgsl(kind: 'closest' | 'any'): string {
    const closest = kind === 'closest';
    const name = closest ? 'giWalkClosest' : 'giWalkAny';
    const returnType = closest ? 'GiHit' : 'bool';
    const noHit = closest ? 'result' : 'false';
    const resultDeclaration = closest ? 'var result = GiHit(-1.0, 0u, 0u, vec2f(0.0), true);' : '';
    // Closest-hit narrows the ray after every hit; any-hit never needs to.
    const farDeclaration = closest ? 'var farthest = ray.tMax;   // hits beyond this are not interesting any more' : 'let farthest = ray.tMax;';
    const onTriangleHit = closest
        ? `// The test ran in object space, where the winding is the one the mesh was
                    // authored with, so frontFacing is the authored side even for a mirrored
                    // instance (DECISIONS 2026-11-04).
                    result = GiHit(candidate.t, instance.instanceIndex, triangle.originalIndex, candidate.barycentrics, candidate.frontFacing);
                    farthest = candidate.t;`
        : 'return true;';
    return /* wgsl */ `
fn ${name}(ray: GiRay) -> ${returnType} {
    ${resultDeclaration}
    let header = giTraceTopLevel[0];
    let instanceCount = header.x;
    let topLevelNodeCount = header.y;
    if (topLevelNodeCount == 0u) {
        return ${noHit};
    }
    let firstTopLevelNode = GI_TOP_HEADER_VEC4S + instanceCount * GI_TOP_VEC4_PER_RECORD;

    let worldSetup = giMakeRaySetup(ray.origin, ray.direction);
    var setup = worldSetup;

    var insideInstance = false;
    var instance: GiInstance;
    ${farDeclaration}

    var stack: array<u32, GI_STACK_SIZE>;
    var stackSize = 0u;
    var current = giMakeReference(0u, 0u);   // the root of the top level

    loop {
        let count = current >> GI_REF_COUNT_SHIFT;
        let index = current & GI_REF_INDEX_MASK;

        if (count == 0u) {
            // An inner node: test both children, then go to one and push the other.
            var node: GiBvhNode;
            if (insideInstance) {
                node = giLoadMeshNode(instance.rootNodeOffset + index);
            } else {
                node = giLoadTopLevelNode(firstTopLevelNode, index);
            }

            // Only a tree with a single primitive has an empty child, and it is the right one.
            let leftEntry = giBoxEntry(node.leftMin, node.leftMax, setup, ray.tMin, farthest);
            var rightEntry = GI_BOX_MISS;
            if (node.rightIndex != GI_EMPTY_CHILD) {
                rightEntry = giBoxEntry(node.rightMin, node.rightMax, setup, ray.tMin, farthest);
            }
            let leftReference = giMakeReference(node.leftIndex, node.leftCount);
            let rightReference = giMakeReference(node.rightIndex, node.rightCount);
            let hitLeft = leftEntry < GI_BOX_MISS;
            let hitRight = rightEntry < GI_BOX_MISS;

            if (hitLeft && hitRight) {
                // Closest-hit visits the nearer child first, any-hit the farther one (see above).
                if (${closest ? 'leftEntry <= rightEntry' : 'leftEntry > rightEntry'}) {
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
                let triangle = giLoadTriangle(instance.triangleOffset + index + i);
                let candidate = giIntersectWatertight(triangle, setup, ray.tMin, farthest);
                // A flagged triangle's material may let the ray through (see-through,
                // a single-sided back face; material-test-wgsl.ts). A ray that passes
                // records nothing and leaves 'farthest' alone, so what lies behind
                // the triangle is still found.
                let consultMaterial = (triangle.flags & GI_TRIANGLE_CONSULT_MATERIAL) != 0u;
                if (candidate.hit && !(consultMaterial && giMaterialPasses(instance.instanceIndex, triangle.originalIndex, candidate.barycentrics, candidate.frontFacing))) {
                    ${onTriangleHit}
                }
            }
        }

        // Nothing left to descend into here: take the next entry off the stack.
        loop {
            if (stackSize == 0u) {
                return ${noHit};
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
`;
}

/**
 * The trace module: contract types, layout, and `giTraceClosest` / `giTraceAny`.
 * Bindings land at `slot`: nodes, triangles, top level, in that order. The
 * walk calls `giMaterialPasses`, which the kernel gets from elsewhere: the
 * material test module over the scene copy, or `NO_MATERIALS_WGSL` for a
 * tracer without one (material-test-wgsl.ts).
 */
export function traceModuleWgsl(slot: BindingSlot): string {
    const group = slot.group;
    const nodes = slot.firstBinding;
    const triangles = slot.firstBinding + 1;
    const topLevel = slot.firstBinding + 2;
    return /* wgsl */ `${COMMON_WGSL}
// Nodes and triangles are read as plain vec4s, so every load is one 16-byte
// read whatever the backend makes of a struct with vec3 members.
@group(${group}) @binding(${nodes}) var<storage, read> giTraceNodes: array<vec4f>;
@group(${group}) @binding(${triangles}) var<storage, read> giTraceTriangles: array<vec4f>;
@group(${group}) @binding(${topLevel}) var<storage, read> giTraceTopLevel: array<vec4u>;

fn giMakeReference(index: u32, count: u32) -> u32 {
    return index | (count << GI_REF_COUNT_SHIFT);
}

// ---- Loading mesh nodes and triangles -----------------------------------------

const GI_VEC4_PER_TRIANGLE: u32 = ${TRIANGLE_BYTES / 16}u;

// Triangle flags bit 0: consult the material (giMaterialPasses, which comes
// from material-test-wgsl.ts as its own piece of the kernel).
const GI_TRIANGLE_CONSULT_MATERIAL: u32 = ${TRIANGLE_FLAG_CONSULT_MATERIAL}u;

fn giLoadMeshNode(nodeIndex: u32) -> GiBvhNode {
    let base = nodeIndex * GI_VEC4_PER_NODE;
    let a = giTraceNodes[base];
    let b = giTraceNodes[base + 1u];
    let c = giTraceNodes[base + 2u];
    let d = giTraceNodes[base + 3u];
    return GiBvhNode(
        a.xyz, bitcast<u32>(a.w),
        b.xyz, bitcast<u32>(b.w),
        c.xyz, bitcast<u32>(c.w),
        d.xyz, bitcast<u32>(d.w),
    );
}

fn giLoadTriangle(triangleIndex: u32) -> GiBvhTriangle {
    let base = triangleIndex * GI_VEC4_PER_TRIANGLE;
    let a = giTraceTriangles[base];
    let b = giTraceTriangles[base + 1u];
    let c = giTraceTriangles[base + 2u];
    return GiBvhTriangle(a.xyz, bitcast<u32>(a.w), b.xyz, bitcast<u32>(b.w), c.xyz, bitcast<u32>(c.w));
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
    let base = firstNode + nodeIndex * GI_VEC4_PER_NODE;
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
    // The direction is not stored: the tests use only these precomputed forms of
    // it. It is not normalized in object space, so t stays in world units.
    origin: vec3f,
    inverseDirection: vec3f,   // for the box test
    // Watertight triangle test: the permute-and-shear that maps a point (relative
    // to the ray origin) into the space where the ray runs along +z'. A point p
    // becomes (dot(p, shearRowX), dot(p, shearRowY), dot(p, shearRowZ)).
    shearRowX: vec3f,
    shearRowY: vec3f,
    shearRowZ: vec3f,
}

fn giSafeInverse(direction: vec3f) -> vec3f {
    let magnitude = max(abs(direction), vec3f(GI_MIN_DIRECTION));
    let directionSign = select(vec3f(1.0), vec3f(-1.0), direction < vec3f(0.0));
    return 1.0 / (magnitude * directionSign);
}

fn giMakeRaySetup(origin: vec3f, direction: vec3f) -> GiRaySetup {
    // The axis the ray runs along most becomes z'; x' is the axis after it
    // (x -> y -> z -> x) and y' the one after that. Swapping x' and y' when the
    // ray points the negative way keeps triangle winding, so the sign of the
    // determinant still tells front from back.
    //
    // The axes are unit vectors picked with select(), never a vector indexed by
    // a per-ray value: a runtime index into a vector can push it into slow
    // per-thread scratch memory (T0.10 part C measured that on NVIDIA).
    let magnitude = abs(direction);
    let zIsX = magnitude.x > magnitude.y && magnitude.x > magnitude.z;
    let zIsY = !zIsX && magnitude.y > magnitude.z;
    let axisZ = select(select(vec3f(0.0, 0.0, 1.0), vec3f(0.0, 1.0, 0.0), zIsY), vec3f(1.0, 0.0, 0.0), zIsX);
    var axisX = axisZ.zxy;   // the next axis: x for z, y for x, z for y
    var axisY = axisX.zxy;
    // Each of these dot products has one product with 1 and two with 0, so it is
    // the direction's component exactly, as the indexed version was.
    let directionZ = dot(direction, axisZ);
    if (directionZ < 0.0) {
        let swapped = axisX;
        axisX = axisY;
        axisY = swapped;
    }
    let shearX = dot(direction, axisX) / directionZ;
    let shearY = dot(direction, axisY) / directionZ;
    let shearZ = 1.0 / directionZ;
    // The rows hold only 0, 1 and the shear factors. A row times a corner is then
    // a[x'] - shearX * a[z'] (and so on) with exactly the old rounding: products
    // with 0 and 1 are exact. A corner shared by two triangles is transformed the
    // same way in both, which is what keeps the test watertight.
    return GiRaySetup(
        origin,
        giSafeInverse(direction),
        axisX - shearX * axisZ,
        axisY - shearY * axisZ,
        shearZ * axisZ,
    );
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

// One edge function of the watertight test: for the edge from 'start' to 'end'
// (corners already sheared into the 2-D space where the ray is the origin), the
// scaled barycentric weight of the corner opposite it, end.x * start.y - end.y * start.x.
//
// Two triangles that share an edge walk it in opposite directions, and their
// values must be exact negatives of each other, or a ray can slip between
// them. Written plainly, that holds only if both products are rounded on their
// own. A shader compiler may fuse one product into the subtraction (an FMA),
// and which product it fuses changes the rounding, so the two triangles can
// round the same edge to the same sign. Dawn compiles for Metal in relaxed math
// mode, which allows that fusing: about 0.1% of the oracle's shared-edge rays
// slipped through on an Apple M5 Pro while NVIDIA and Intel let none through.
//
// So the two corners are put in a fixed order first (by x, then y). Both
// triangles then run the same arithmetic on the same operands and get the same
// bits, whatever the compiler fuses, and the one that walks the edge the other
// way negates the result, which is exact. A shared corner has the same sheared
// bits in every triangle (giMakeRaySetup), so both triangles agree on the order.
fn giEdgeFunction(start: vec2f, end: vec2f) -> f32 {
    let reversed = start.x > end.x || (start.x == end.x && start.y > end.y);
    let first = select(start, end, reversed);
    let second = select(end, start, reversed);
    // An edge that shears to a single point runs along the ray; exactly, its
    // value is 0. A fused x * y - y * x would give the rounding error instead,
    // with the same sign in both triangles (neither negates, as neither is reversed).
    let ordered = select(second.x * first.y - second.y * first.x, 0.0, all(first == second));
    return select(ordered, -ordered, reversed);
}

// Woop, Benthin and Wald, "Watertight Ray/Triangle Intersection" (2013).
// Triangles store three corners. The three edge functions are computed from
// the same corner bits for both triangles that share an edge, with opposite
// signs (giEdgeFunction), so a ray cannot pass between them. A zero counts as
// inside for both, so a ray exactly on an edge hits both neighbours rather than neither.
fn giIntersectWatertight(triangle: GiBvhTriangle, setup: GiRaySetup, tMin: f32, tMax: f32) -> GiTriangleHit {
    // Corners relative to the ray origin.
    let a = triangle.vertex0 - setup.origin;
    let b = triangle.vertex1 - setup.origin;
    let c = triangle.vertex2 - setup.origin;

    // Shear and scale into the space where the ray runs along +z' through the origin
    // (see giMakeRaySetup for why these are dot products with rows).
    let ax = dot(a, setup.shearRowX);
    let ay = dot(a, setup.shearRowY);
    let bx = dot(b, setup.shearRowX);
    let by = dot(b, setup.shearRowY);
    let cx = dot(c, setup.shearRowX);
    let cy = dot(c, setup.shearRowY);

    // Edge functions: the (scaled) barycentric weights of corners 0, 1 and 2,
    // from the edges opposite them (b to c, c to a, a to b).
    let projectedA = vec2f(ax, ay);
    let projectedB = vec2f(bx, by);
    let projectedC = vec2f(cx, cy);
    let weight0 = giEdgeFunction(projectedB, projectedC);
    let weight1 = giEdgeFunction(projectedC, projectedA);
    let weight2 = giEdgeFunction(projectedA, projectedB);
    if ((weight0 < 0.0 || weight1 < 0.0 || weight2 < 0.0) && (weight0 > 0.0 || weight1 > 0.0 || weight2 > 0.0)) {
        return giNoTriangleHit();   // the weights disagree in sign: outside the triangle
    }

    let determinant = weight0 + weight1 + weight2;
    if (determinant == 0.0) {
        return giNoTriangleHit();   // edge-on
    }
    // z' of each corner, scaled by 1 / direction along z'.
    let scaledT = weight0 * dot(a, setup.shearRowZ) + weight1 * dot(b, setup.shearRowZ) + weight2 * dot(c, setup.shearRowZ);
    let inverseDeterminant = 1.0 / determinant;
    let t = scaledT * inverseDeterminant;
    if (t < tMin || t > tMax) {
        return giNoTriangleHit();
    }
    // With the axis swap in giMakeRaySetup the determinant is positive for a front hit.
    return GiTriangleHit(true, t, vec2f(weight1, weight2) * inverseDeterminant, determinant > 0.0);
}

${walkWgsl('closest')}
${walkWgsl('any')}
// Nearest hit, honouring see-through and single-sided materials (M4; cutouts
// arrive with T4.2's second half).
fn giTraceClosest(ray: GiRay) -> GiHit {
    return giWalkClosest(ray);
}

// True if anything blocks the ray. Stops at the first blocker found.
fn giTraceAny(ray: GiRay) -> bool {
    return giWalkAny(ray);
}
`;
}

/**
 * What a batch writes per ray in closest-hit mode. `GiHit` itself cannot live
 * in a buffer because `bool` is not host-shareable in WGSL, so `frontFace`
 * is stored as 0 or 1.
 */
export const HIT_RECORD_BYTES = 24;

/**
 * Word positions inside a `GiHitRecord`, for code that reads batch results
 * back. `frontFace` sits before `barycentrics` so the `vec2f` starts on an
 * 8-byte boundary and the record has no padding (`wgsl.test.ts` checks this
 * against the struct in `batchEntryWgsl`).
 */
export const HIT_RECORD_WORD = {
    t: 0,
    instance: 1,
    triangle: 2,
    frontFace: 3,
    barycentrics: 4,
} as const;

/** Workgroup size of the batch kernels. */
export const BATCH_WORKGROUP_SIZE = 64;

/**
 * The batch kernel's own part: its two bindings in `group` (rays in, results
 * out) and the entry point `giBatch`, which traces `arrayLength(giBatchRays)`
 * rays and writes one result per ray. The trace module comes separately (the
 * platform's kernel builder puts it in group 0).
 *
 * The ray count is the size of the bound `rays` range, so there is no uniform
 * to keep in step between several dispatches recorded into one command buffer.
 */
export function batchEntryWgsl(mode: 'closest' | 'any', group: number): string {
    const resultDeclaration =
        mode === 'closest'
            ? `struct GiHitRecord {
    t: f32,
    instance: u32,
    triangle: u32,
    frontFace: u32,   // 0 or 1
    barycentrics: vec2f,
}

@group(${group}) @binding(1) var<storage, read_write> giBatchResults: array<GiHitRecord>;`
            : `// 1 when the ray is blocked, 0 when it is clear.
@group(${group}) @binding(1) var<storage, read_write> giBatchResults: array<u32>;`;

    const body =
        mode === 'closest'
            ? `let hit = giTraceClosest(giBatchRays[rayIndex]);
    giBatchResults[rayIndex] = GiHitRecord(hit.t, hit.instance, hit.triangle, select(0u, 1u, hit.frontFace), hit.barycentrics);`
            : `giBatchResults[rayIndex] = select(0u, 1u, giTraceAny(giBatchRays[rayIndex]));`;

    return /* wgsl */ `
@group(${group}) @binding(0) var<storage, read> giBatchRays: array<GiRay>;
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

/**
 * A whole batch kernel without a scene copy: the trace module at `traceSlot`,
 * no materials (every triangle blocks; material-test-wgsl.ts), and the
 * batch's own group 1.
 */
export function batchKernelWgsl(mode: 'closest' | 'any', traceSlot: BindingSlot = { group: 0, firstBinding: 0 }): string {
    return `${traceModuleWgsl(traceSlot)}
${NO_MATERIALS_WGSL}
${batchEntryWgsl(mode, 1)}`;
}
