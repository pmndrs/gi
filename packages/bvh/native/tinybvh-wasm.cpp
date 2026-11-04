// A small C interface over tinybvh's default triangle builder, compiled to
// WebAssembly (see build-wasm.ts). With AVX available (Emscripten emulates it on
// WASM SIMD), `BVH::Build` runs tinybvh's SIMD Bonsai builder (Ganestam et al.,
// 2015: small binned-SAH trees over a grid of cells, joined by a top tree). On
// Sponza it is about 3x faster than tinybvh's scalar reference builder and its
// trees have a lower SAH cost.
//
// It builds one mesh at a time and leaves the result in WASM memory for
// JavaScript to read: tinybvh's 32-byte nodes and its triangle order. JavaScript turns those into a plain tree description (src/tinybvh.ts);
// nothing here knows about our GPU layout.
//
// tinybvh: https://github.com/jbikker/tinybvh (MIT), by Jacco Bikker. The
// pinned commit and header hashes are in build-wasm.ts and THIRD_PARTY.md.

#define TINYBVH_IMPLEMENTATION
#include "tiny_bvh.h"

#include <stdint.h>

#define EXPORT extern "C" __attribute__((used, visibility("default")))

// The last tree built. One mesh at a time: the worker builds meshes in sequence.
static tinybvh::BVH* lastBvh = nullptr;

static void releaseLast()
{
    delete lastBvh;
    lastBvh = nullptr;
}

// Builds a tree over `triangleCount` triangles. `vertices` holds four floats per
// vertex (x, y, z, unused), because tinybvh reads vertices as 16-byte vectors.
// `indices` holds three vertex indices per triangle, or is null for a plain
// triangle list (three vertices per triangle). Returns the number of node slots
// used (tinybvh leaves slot 1 unused so siblings share a cache line), or 0 on failure.
EXPORT uint32_t tbvh_build(const float* vertices, uint32_t vertexCount, const uint32_t* indices, uint32_t triangleCount)
{
    releaseLast();
    // JavaScript checks the input first; this only keeps a bad call from reaching tinybvh.
    if (triangleCount == 0) return 0;
    lastBvh = new tinybvh::BVH();
    const tinybvh::bvhvec4slice slice(reinterpret_cast<const tinybvh::bvhvec4*>(vertices), vertexCount, sizeof(tinybvh::bvhvec4));
    if (indices) {
        lastBvh->Build(slice, indices, triangleCount);
    } else {
        lastBvh->Build(slice);
    }
    return lastBvh->usedNodes;
}

// tinybvh's nodes: per node aabbMin (3 floats), leftFirst, aabbMax (3 floats), triCount.
// triCount > 0: a leaf of triCount triangles from position leftFirst of the triangle order.
// triCount == 0: an inner node whose children are leftFirst and leftFirst + 1.
EXPORT const void* tbvh_nodes()
{
    return lastBvh ? static_cast<const void*>(lastBvh->bvhNode) : nullptr;
}

// The triangle order: position i of the tree holds original triangle tbvh_triangle_order()[i].
EXPORT const uint32_t* tbvh_triangle_order()
{
    return lastBvh ? lastBvh->primIdx : nullptr;
}

// Entries in the triangle order (equal to the triangle count: without spatial splits or presplitting, which
// are off by default, tinybvh never splits a triangle).
EXPORT uint32_t tbvh_triangle_order_length()
{
    return lastBvh ? lastBvh->idxCount : 0;
}

EXPORT void tbvh_release()
{
    releaseLast();
}

// Memory for the inputs, owned by JavaScript between a build's start and end.
EXPORT void* tbvh_alloc(uint32_t bytes)
{
    return malloc(bytes);
}

EXPORT void tbvh_free(void* pointer)
{
    free(pointer);
}
