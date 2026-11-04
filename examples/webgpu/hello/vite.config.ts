import { resolve } from 'path';
import { defineConfig } from 'vite';

// Consumes the library straight from source, like the viewer.
export default defineConfig({
    root: __dirname,
    resolve: {
        alias: [
            { find: /^@pmndrs\/ddgi$/, replacement: resolve(__dirname, '../../../packages/ddgi/src/index.ts') },
            { find: /^@pmndrs\/ddgi\/(core|three)$/, replacement: resolve(__dirname, '../../../packages/ddgi/src/$1.ts') },
        ],
    },
    build: { target: 'esnext' },
    server: { port: 5201, open: false },
});
