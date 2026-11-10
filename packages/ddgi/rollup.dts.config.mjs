import { dts } from 'rollup-plugin-dts';

import { layerChunk } from './layer-chunks.ts';

// Bundles one .d.ts per public entry. The private workspace packages
// (@pmndrs/ddgi-core / -three) are inlined, so the published types never name a
// package that isn't on npm; three stays external. The chunks are cut by layer,
// as the JS build's are (layer-chunks.ts).
export default {
    input: { index: 'src/index.ts', core: 'src/core.ts', ddgi: 'src/ddgi.ts', three: 'src/three.ts' },
    output: { dir: 'dist', format: 'es', manualChunks: layerChunk, chunkFileNames: 'chunks/[name]-[hash].d.ts' },
    external: [/^three(\/|$)/],
    plugins: [dts({ respectExternal: true, tsconfig: 'tsconfig.build.json' })],
};
