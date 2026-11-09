import { resolve } from 'path';
import { defineConfig } from 'vite';

// Library build for @pmndrs/ddgi. One entry per public subpath. The private
// workspace packages (@pmndrs/ddgi-core, @pmndrs/ddgi-three) are bundled in;
// shared code lands in one chunk so every entry sees the same classes (an
// `instanceof` across subpaths must hold).
export default defineConfig({
    build: {
        lib: {
            entry: {
                index: resolve(__dirname, 'src/index.ts'),
                core: resolve(__dirname, 'src/core.ts'),
                three: resolve(__dirname, 'src/three.ts'),
                'radiance-cache': resolve(__dirname, 'src/radiance-cache.ts'),
            },
            formats: ['es'],
        },
        rollupOptions: {
            // three and its addons (the composite uses three's own GTAO) stay the user's copy.
            external: [/^three(\/|$)/],
            output: { chunkFileNames: 'chunks/ddgi-[hash].js' },
        },
    },
});
