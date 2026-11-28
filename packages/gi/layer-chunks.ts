// The build's chunks, one per layer of the library, shared by the JS build
// (vite.config.ts) and the type build (rollup.dts.config.mjs).
//
// Every module lands in exactly one chunk, so a class is one class whichever
// subpath it is imported from (`instanceof` holds across subpaths). Cutting the
// chunks by layer keeps each subpath's download to its own layers:
//
//   platform  the engine-free GI platform and the bundled tree builders
//             (packages/core/src/platform, packages/bvh); `/core` loads
//             only this one
//   ddgi      the DDGI technique (packages/ddgi);
//             `/ddgi` loads it and the platform
//   three     the three.js adapter (packages/three and packages/ddgi-three); `.` and `/three` load
//             all three
//   rc        Split radiance cascades (techniques/radiance-cascades);
//             `/radiance-cascades` loads it and the platform
//   radiance-cache
//             the radiance cache (packages/cache);
//             `/radiance-cache` loads it and the platform
//
// The chunk names say which layer a file holds, so scripts/verify-artifacts.mjs
// and tools/release/verify-packed.ts can check that `/core` reaches no `ddgi`
// or `three` chunk, and `/ddgi` no `three` chunk.

/** Chunk names by layer, in the order the layers build on each other. */
export const LAYER_CHUNKS = ['platform', 'ddgi', 'three', 'rc', 'radiance-cache'] as const;

export type Layer = (typeof LAYER_CHUNKS)[number];

/** The layers each public entry may load: an entry below a layer must never reach its code. */
export const ENTRY_LAYERS: Record<string, readonly Layer[]> = {
    'radiance-cascades': ['platform', 'rc'],
    'radiance-cache': ['platform', 'radiance-cache'],
    core: ['platform'],
    ddgi: ['platform', 'ddgi'],
    three: ['platform', 'ddgi', 'three'],
    index: ['platform', 'ddgi', 'three'],
};

/** Rollup's `manualChunks`: the layer chunk a module belongs to, or undefined for the entries and helpers. */
export function layerChunk(id: string): Layer | undefined {
    const path = id.replaceAll('\\', '/');
    if (path.includes('/techniques/radiance-cascades/')) return 'rc';
    if (path.includes('/packages/cache/src/')) return 'radiance-cache';
    if (path.includes('/packages/three/src/') || path.includes('/packages/ddgi-three/src/')) return 'three';
    if (path.includes('/packages/ddgi/src/')) return 'ddgi';
    if (path.includes('/packages/core/src/platform/') || path.includes('/packages/bvh/')) return 'platform';
    return undefined;
}
