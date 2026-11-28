import { resolve } from 'path';
import { defineConfig } from 'vite';

// Consumes the library straight from source, like the viewer.
export default defineConfig({
    root: __dirname,
    resolve: {
        alias: [
            { find: /^@pmndrs\/gi$/, replacement: resolve(__dirname, '../../../packages/gi/src/index.ts') },
            { find: /^@pmndrs\/gi\/(core|ddgi|three)$/, replacement: resolve(__dirname, '../../../packages/gi/src/$1.ts') },
        ],
    },
    build: { target: 'esnext' },
    server: { port: 5201, open: false },
});
