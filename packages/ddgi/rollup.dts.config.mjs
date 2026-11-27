import { dts } from 'rollup-plugin-dts';

import { layerChunk } from './layer-chunks.ts';

// Bundles one .d.ts per public entry. The private workspace packages
// (@pmndrs/ddgi-core / -three) are inlined, so the published types never name a
// package that isn't on npm; three and the npm dependencies stay external. The
// radiance cache's neural tail names `@pathtracer/mlp`'s types, a package that
// isn't on npm: tsconfig.build.json maps it to the repository's declarations of
// it, so those are inlined and the published types never import it. The chunks
// are cut by layer, as the JS build's are (layer-chunks.ts).
export default {
    input: {
        'radiance-cascades': 'src/radiance-cascades.ts',
        'radiance-cache': 'src/radiance-cache.ts',
        index: 'src/index.ts',
        core: 'src/core.ts',
        ddgi: 'src/ddgi.ts',
        three: 'src/three.ts',
    },
    output: { dir: 'dist', format: 'es', manualChunks: layerChunk, chunkFileNames: 'chunks/[name]-[hash].d.ts' },
    external: ['@pmndrs/rc-bvh', /^denoiser(\/|$)/, /^three(\/|$)/],
    plugins: [dts({ respectExternal: true, tsconfig: 'tsconfig.build.json' })],
};
