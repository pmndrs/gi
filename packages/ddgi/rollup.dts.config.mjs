import { dts } from 'rollup-plugin-dts';

// Bundles one .d.ts per public entry. The private workspace packages
// (@pmndrs/ddgi-core / -three) are inlined, so the published types never name a
// package that isn't on npm; three stays external.
export default {
    input: { index: 'src/index.ts', core: 'src/core.ts', three: 'src/three.ts', 'radiance-cache': 'src/radiance-cache.ts' },
    output: { dir: 'dist', format: 'es', chunkFileNames: 'chunks/ddgi-[hash].d.ts' },
    external: [/^three(\/|$)/],
    plugins: [dts({ respectExternal: true, tsconfig: 'tsconfig.build.json' })],
};
