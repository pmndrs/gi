import { delimiter, dirname, resolve } from 'path';
import { fileURLToPath } from 'url';
import { defineConfig } from 'vite';

const scenesDirectory = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(scenesDirectory, '..');

// Downloaded models are read from these folders, first match wins. Set
// DDGI_DOWNLOADS_DIR (several folders separated like PATH) to point the page
// somewhere else, for example a backup of converted outputs.
const downloadsDirectories = process.env.DDGI_DOWNLOADS_DIR
    ? process.env.DDGI_DOWNLOADS_DIR.split(delimiter).filter(Boolean).map((directory) => resolve(directory))
    : [resolve(repoRoot, 'assets', 'downloads')];

// The check page: `pnpm --filter @pmndrs/ddgi-scenes check`, then open
// http://localhost:5210/?scene=cornell-box&camera=1
export default defineConfig({
    root: resolve(scenesDirectory, 'check'),
    define: {
        // The page reads models through /@fs/<one of these folders>/<asset id>/<path>.
        __DOWNLOADS_DIRS__: JSON.stringify(downloadsDirectories),
    },
    server: {
        port: 5210,
        open: false,
        // The downloads may live outside the vite root (or the repo), so allow them explicitly.
        fs: { allow: [repoRoot, ...downloadsDirectories] },
    },
    // Top-level await and WebGPU need a modern target.
    build: { target: 'esnext', outDir: resolve(scenesDirectory, 'dist'), emptyOutDir: true },
});
