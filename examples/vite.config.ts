import { readdirSync } from 'fs';
import { resolve } from 'path';
import { defineConfig, type Plugin } from 'vite';
import { optionalSiblings, optionalSiblingsPlugin } from '../tools/optional-siblings.ts';

// The examples site: a landing page and one folder per example
// (`NN-name/index.html` + `main.ts`), plus the plain-WebGPU hello example.
// `pnpm examples` serves it; `pnpm examples:build` builds it into examples/dist.
// The library is read straight from source, so edits to it show up here at once.
const root = __dirname;
const repoRoot = resolve(root, '..');

/** Every numbered example folder (`01-hello`, `02-composite`, ...), found on disk so a new one can't be left out of the build. */
const numberedExamples = readdirSync(root, { withFileTypes: true })
    .filter((entry) => entry.isDirectory() && /^\d\d-/.test(entry.name))
    .map((entry) => entry.name)
    .sort();

/**
 * Every example page gets the gallery's small nav (back to the examples, the
 * title, the source), injected here so an example never imports it. The
 * landing page doesn't get one.
 */
function exampleNav(): Plugin {
    return {
        name: 'ddgi-example-nav',
        transformIndexHtml: {
            // 'pre', so the build bundles the injected script like a written one.
            order: 'pre',
            handler(_html, context) {
                if (context.path === '/index.html') return;
                return [{ tag: 'script', attrs: { type: 'module', src: '/shared/nav.ts' }, injectTo: 'body' }];
            },
        },
    };
}

export default defineConfig({
    root,
    // GitHub Pages serves a project site under /<repo>/, so the Pages build sets
    // PAGES_BASE (/gi/, for https://pmndrs.github.io/gi/). Local
    // dev and builds default to '/'. Links between pages are relative, so they
    // work under either.
    base: process.env.PAGES_BASE ?? '/',
    // Optional siblings (`@pathtracer/mlp`, the radiance cache's neural tail, which example 12 leaves off): the pinned
    // clone in the dev server, else a module that throws when loaded. A build gets that module too (unless
    // GI_BUNDLE_OPTIONAL_SIBLINGS=1), so the published site never bundles the private sibling.
    plugins: [exampleNav(), optionalSiblingsPlugin()],
    resolve: {
        // Exact-match entries per public subpath: a plain '@pmndrs/gi' key is a
        // prefix alias and would send '@pmndrs/gi/core' to 'index.ts/core'.
        alias: [
            { find: /^@pmndrs\/gi$/, replacement: resolve(repoRoot, 'packages/gi/src/index.ts') },
            { find: /^@pmndrs\/gi\/(core|ddgi|three|radiance-cascades|radiance-cache)$/, replacement: resolve(repoRoot, 'packages/gi/src/$1.ts') },
        ],
    },
    // Our own dependency cache, so this server and the viewer's don't fight over node_modules/.vite.
    cacheDir: resolve(repoRoot, 'node_modules/.vite-examples'),
    build: {
        // Top-level await (renderer.init, volume.start) needs a modern target.
        target: 'esnext',
        outDir: resolve(root, 'dist'),
        emptyOutDir: true,
        // three and the library make one large chunk per page; that is expected here.
        chunkSizeWarningLimit: 2000,
        rollupOptions: {
            input: {
                index: resolve(root, 'index.html'),
                ...Object.fromEntries(numberedExamples.map((name) => [name, resolve(root, name, 'index.html')])),
                'webgpu-hello': resolve(root, 'webgpu/hello/index.html'),
            },
            output: {
                // Name the shared chunks for what they hold. The core apart from the three adapter, so the
                // plain-WebGPU example doesn't download three.
                manualChunks(id) {
                    if (id.includes('/node_modules/')) return id.includes('/three/') ? 'three' : 'vendor';
                    if (id.includes('/packages/ddgi-three/')) return 'ddgi-three';
                    if (id.includes('/packages/')) return 'ddgi-core';
                    return undefined;
                },
            },
        },
    },
    server: {
        port: 5203,
        open: false,
        // The scene builders and manifests live in scenes/, outside this folder.
        fs: { allow: [repoRoot, ...optionalSiblings().flatMap((sibling) => (sibling.folder ? [sibling.folder] : []))] },
    },
});
