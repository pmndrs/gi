import { delimiter, resolve } from 'path';
import { defineConfig } from 'vite';

const repoRoot = resolve(__dirname, '../..');

// Downloaded models are read from these folders, first match wins:
// assets/downloads, then assets/downloads.bak (a clean re-fetch can leave a
// converted output only there for a while). Set DDGI_DOWNLOADS_DIR (several
// folders separated like PATH) to point somewhere else.
const downloadsDirectories = process.env.DDGI_DOWNLOADS_DIR
    ? process.env.DDGI_DOWNLOADS_DIR.split(delimiter).filter(Boolean).map((directory) => resolve(directory))
    : [resolve(repoRoot, 'assets', 'downloads'), resolve(repoRoot, 'assets', 'downloads.bak')];

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
    define: {
        // `src/scene/model-url.ts` reads models through /@fs/<one of these>/<asset id>/<path>.
        __DOWNLOADS_DIRS__: JSON.stringify(downloadsDirectories),
        // `src/scene/reference-files.ts` reads the Cycles references (gitignored, rendered on demand) from here.
        __REFERENCES_DIR__: JSON.stringify(resolve(repoRoot, 'bench', 'references')),
    },
    // Top-level await (renderer.init) needs a modern target.
    build: { target: 'esnext' },
    server: {
        port: 5200,
        open: false,
        // The scene manifests and the downloaded models live outside this folder.
        fs: { allow: [repoRoot, ...downloadsDirectories] },
    },
});
