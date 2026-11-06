import { resolve } from 'path';
import { defineConfig } from 'vite';

// The three.js examples, consuming the library straight from source like the
// hello example. Run from the repo root: pnpm exec vite --config examples/three/vite.config.ts
export default defineConfig({
    root: __dirname,
    resolve: {
        alias: [
            { find: /^@pmndrs\/ddgi$/, replacement: resolve(__dirname, '../../packages/ddgi/src/index.ts') },
            { find: /^@pmndrs\/ddgi\/(core|three)$/, replacement: resolve(__dirname, '../../packages/ddgi/src/$1.ts') },
            // The composite example's upscaler: a dependency of the viewer for now (this folder isn't a workspace package yet).
            { find: /^@pmndrs\/upscaler$/, replacement: resolve(__dirname, '../../apps/viewer/node_modules/@pmndrs/upscaler/dist/index.js') },
        ],
    },
    build: {
        target: 'esnext',
        rollupOptions: { input: { index: resolve(__dirname, 'index.html'), simple: resolve(__dirname, 'simple.html'), composite: resolve(__dirname, 'composite.html') } },
    },
    server: { port: 5202, open: false },
});
