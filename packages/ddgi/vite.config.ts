import { resolve } from 'path';
import { defineConfig } from 'vite';

import { layerChunk } from './layer-chunks.ts';

// Library build for @pmndrs/gi. One entry per public subpath: `/core` (the
// platform), `/ddgi` (the DDGI technique), `/three` (the three.js adapter) and
// `.` (the same as `/three`). The private workspace packages (@pmndrs/ddgi-core,
// @pmndrs/ddgi-three) are bundled in.
//
// Each module lands in exactly one chunk, so every entry sees the same classes
// (an `instanceof` across subpaths must hold). The chunks are cut by layer
// (layer-chunks.ts), so `/core` loads no DDGI code and `/ddgi` no three code.
export default defineConfig({
    build: {
        lib: {
            entry: {
                index: resolve(__dirname, 'src/index.ts'),
                core: resolve(__dirname, 'src/core.ts'),
                ddgi: resolve(__dirname, 'src/ddgi.ts'),
                three: resolve(__dirname, 'src/three.ts'),
            },
            formats: ['es'],
        },
        rollupOptions: {
            // three and its addons (the composite uses three's own GTAO) stay the user's copy.
            external: [/^three(\/|$)/],
            output: {
                manualChunks: layerChunk,
                chunkFileNames: 'chunks/[name]-[hash].js',
            },
        },
    },
});
