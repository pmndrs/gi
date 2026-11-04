import { resolve } from 'path';
import { defineConfig } from 'vite';

// The viewer consumes the library straight from source so shader and pass
// edits hot-reload. Exact-match per subpath: a plain string key is a prefix
// alias and would send '@pmndrs/ddgi/core' to 'index.ts/core'.
export default defineConfig({
    root: __dirname,
    resolve: {
        alias: [
            { find: /^@pmndrs\/ddgi$/, replacement: resolve(__dirname, '../../packages/ddgi/src/index.ts') },
            { find: /^@pmndrs\/ddgi\/(core|three)$/, replacement: resolve(__dirname, '../../packages/ddgi/src/$1.ts') },
        ],
    },
    // Top-level await (renderer.init) needs a modern target.
    build: { target: 'esnext' },
    server: { port: 5200, open: false },
});
