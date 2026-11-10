import { execFileSync } from 'child_process';
import { existsSync, readdirSync } from 'fs';
import { delimiter, join, resolve, sep } from 'path';
import { defineConfig, type Plugin } from 'vite';

const repoRoot = resolve(__dirname, '../..');

/** The repo's HEAD commit when the dev server starts, or '' outside a git checkout. */
function currentCommit(): string {
    try {
        return execFileSync('git', ['rev-parse', 'HEAD'], { cwd: repoRoot, encoding: 'utf8' }).trim();
    } catch {
        return '';
    }
}

// Downloaded models are read from these folders, first match wins:
// assets/downloads, then assets/downloads.bak (a clean re-fetch can leave a
// converted output only there for a while). Set DDGI_DOWNLOADS_DIR (several
// folders separated like PATH) to point somewhere else.
const downloadsDirectories = process.env.DDGI_DOWNLOADS_DIR
    ? process.env.DDGI_DOWNLOADS_DIR.split(delimiter).filter(Boolean).map((directory) => resolve(directory))
    : [resolve(repoRoot, 'assets', 'downloads'), resolve(repoRoot, 'assets', 'downloads.bak')];

// Motion recordings for the compare page (compare.html, T2.8 check 3): native's
// from the native harness (tools/native/play.ps1 -Record), ours from the
// viewer-recording GPU page (tools/native/save-viewer-recording.ts). Both are
// git-ignored. DDGI_NATIVE_RECORDINGS / DDGI_VIEWER_RECORDINGS point elsewhere
// (another worktree's native/build/recordings, say).
const recordingDirectories = [
    { source: 'native', directory: resolve(process.env.DDGI_NATIVE_RECORDINGS ?? resolve(repoRoot, 'native', 'build', 'recordings')) },
    { source: 'viewer', directory: resolve(process.env.DDGI_VIEWER_RECORDINGS ?? resolve(repoRoot, 'bench', 'recordings', 'viewer')) },
];

/**
 * `GET /__recordings`: the recordings the compare page can play, as
 * `[{ source, name, url }]`, `url` being the folder as the dev server serves
 * it (/@fs/...). A folder counts when it holds a recording.json. Dev server only.
 */
function recordingsList(): Plugin {
    return {
        name: 'ddgi-recordings-list',
        configureServer(server) {
            server.middlewares.use('/__recordings', (_request, response) => {
                const list = recordingDirectories.flatMap(({ source, directory }) => {
                    if (!existsSync(directory)) return [];
                    return readdirSync(directory, { withFileTypes: true })
                        .filter((entry) => entry.isDirectory() && existsSync(join(directory, entry.name, 'recording.json')))
                        .map((entry) => ({ source, name: entry.name, url: `/@fs/${join(directory, entry.name).split(sep).join('/')}/` }));
                });
                response.setHeader('Content-Type', 'application/json');
                response.end(JSON.stringify(list));
            });
        },
    };
}

// The viewer consumes the library straight from source so shader and pass
// edits hot-reload. Exact-match per subpath: a plain string key is a prefix
// alias and would send '@pmndrs/gi/core' to 'index.ts/core'.
export default defineConfig({
    root: __dirname,
    resolve: {
        alias: [
            { find: /^@pmndrs\/gi$/, replacement: resolve(__dirname, '../../packages/ddgi/src/index.ts') },
            { find: /^@pmndrs\/gi\/(core|ddgi|three)$/, replacement: resolve(__dirname, '../../packages/ddgi/src/$1.ts') },
        ],
    },
    define: {
        // `src/scene/model-url.ts` reads models through /@fs/<one of these>/<asset id>/<path>.
        __DOWNLOADS_DIRS__: JSON.stringify(downloadsDirectories),
        // `src/scene/reference-files.ts` reads the Cycles references (gitignored, rendered on demand) from here.
        __REFERENCES_DIR__: JSON.stringify(resolve(repoRoot, 'bench', 'references')),
        // The commit the viewer runs, so the Compare panel can say when a reference is from an older one.
        __VIEWER_COMMIT__: JSON.stringify(currentCommit()),
    },
    plugins: [recordingsList()],
    // Top-level await (renderer.init) needs a modern target. Two pages: the viewer and the native compare page.
    build: { target: 'esnext', rollupOptions: { input: { main: resolve(__dirname, 'index.html'), compare: resolve(__dirname, 'compare.html') } } },
    server: {
        port: 5200,
        open: false,
        // The scene manifests and the downloaded models live outside this folder.
        fs: { allow: [repoRoot, ...downloadsDirectories, ...recordingDirectories.map((entry) => entry.directory)] },
    },
});
