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
            },
            formats: ['es'],
        },
        rollupOptions: {
            external: ['three', 'three/tsl', 'three/webgpu'],
            output: { chunkFileNames: 'chunks/ddgi-[hash].js' },
        },
    },
});
