/**
 * Serves the test pages with vite's dev server, so a page can import
 * TypeScript (including the shared client helper) with no build step.
 */

import { existsSync, readdirSync, readFileSync, realpathSync, statSync } from 'node:fs';
import { extname, join, relative, resolve, sep } from 'node:path';
import type { ServerResponse } from 'node:http';
import { createServer, searchForWorkspaceRoot, type Connect, type ViteDevServer } from 'vite';
import { EXAMPLES_SITE_BASE, EXAMPLES_SITE_FILES, REFERENCE_FILES } from './contract.ts';

/** The folder served as the site root: it holds client.ts, preflight.html and pages/. */
export const SITE_ROOT = import.meta.dirname;

export interface TestSiteServer {
    /** For example http://127.0.0.1:5290 */
    origin: string;
    close(): Promise<void>;
}

/**
 * Sibling libraries linked from a local clone for development (root package.json `link:`;
 * the radiance cache's GI denoiser, RC4.3). They live outside this repo, so vite serves their
 * real folders only when they are listed in `server.fs.allow`.
 */
const LINKED_SIBLINGS = ['@pmndrs/denoiser-ffx']
    .map((name) => resolve(SITE_ROOT, '../../node_modules', name))
    .filter((path) => existsSync(path))
    .map((path) => realpathSync(path));

/** Where `pnpm examples:build` writes the examples site. */
const EXAMPLES_DIST = resolve(SITE_ROOT, '../../examples/dist');

/** The Cycles references (gitignored; `tools/reference/README.md`). */
const REFERENCES_ROOT = resolve(SITE_ROOT, '../../bench/references');

const CONTENT_TYPES: Readonly<Record<string, string>> = {
    '.html': 'text/html; charset=utf-8',
    '.js': 'text/javascript; charset=utf-8',
    '.css': 'text/css; charset=utf-8',
    '.json': 'application/json',
    '.wasm': 'application/wasm',
    '.png': 'image/png',
    '.jpg': 'image/jpeg',
    '.svg': 'image/svg+xml',
};

/** Every file under `directory`, with its size; empty when the directory doesn't exist. */
function listFiles(directory: string): Array<{ path: string; bytes: number }> {
    if (!existsSync(directory)) return [];
    const files: Array<{ path: string; bytes: number }> = [];
    const walk = (folder: string): void => {
        for (const entry of readdirSync(folder, { withFileTypes: true })) {
            const path = join(folder, entry.name);
            if (entry.isDirectory()) walk(path);
            else files.push({ path: relative(directory, path).split(sep).join('/'), bytes: statSync(path).size });
        }
    };
    walk(directory);
    return files;
}

/**
 * Serves the files of `directory` as a static host does: a folder's
 * index.html for a path ending in `/`, 404 for anything missing (which the
 * browser logs, so the runner fails the page). No vite transforms: the
 * built files are served as they are.
 */
function serveStatic(directory: string): Connect.NextHandleFunction {
    return (request, response: ServerResponse) => {
        const urlPath = decodeURIComponent((request.url ?? '/').split('?')[0]);
        const path = resolve(directory, `.${urlPath.endsWith('/') ? `${urlPath}index.html` : urlPath}`);
        if (!path.startsWith(directory + sep) || !existsSync(path) || !statSync(path).isFile()) {
            response.statusCode = 404;
            response.end('Not found');
            return;
        }
        response.setHeader('Content-Type', CONTENT_TYPES[extname(path)] ?? 'application/octet-stream');
        response.end(readFileSync(path));
    };
}

export async function startTestSite(): Promise<TestSiteServer> {
    const server: ViteDevServer = await createServer({
        root: SITE_ROOT,
        // Do not pick up a vite config from the repo: the test site is self-contained.
        configFile: false,
        logLevel: 'warn',
        appType: 'mpa',
        // The published names, pointed at the source as the examples' own config does, so a page can
        // run examples/three as a user writes it (`@pmndrs/gi/three`).
        resolve: {
            alias: [
                { find: /^@pmndrs\/gi$/, replacement: resolve(SITE_ROOT, '../../packages/ddgi/src/index.ts') },
                { find: /^@pmndrs\/gi\/(core|ddgi|three)$/, replacement: resolve(SITE_ROOT, '../../packages/ddgi/src/$1.ts') },
            ],
        },
        // The pages have no npm dependencies; skip the dependency pre-bundling scan.
        optimizeDeps: { noDiscovery: true, include: [] },
        server: {
            host: '127.0.0.1',
            port: 5290,
            // Take the next free port if a dev server is already running.
            strictPort: false,
            // Pages are reloaded by the runner; hot reload would only add noise.
            hmr: false,
            fs: { allow: [searchForWorkspaceRoot(SITE_ROOT), ...LINKED_SIBLINGS] },
            watch: null,
        },
        plugins: [
            {
                // The built examples site, as a static subdirectory (contract.ts, EXAMPLES_SITE_BASE).
                name: 'examples-site',
                configureServer(devServer) {
                    devServer.middlewares.use(EXAMPLES_SITE_FILES, (_request, response) => {
                        response.setHeader('Content-Type', 'application/json');
                        response.end(JSON.stringify(listFiles(EXAMPLES_DIST)));
                    });
                    devServer.middlewares.use(EXAMPLES_SITE_BASE, serveStatic(EXAMPLES_DIST));
                },
            },
            {
                // One scene's reference files (contract.ts, REFERENCE_FILES).
                name: 'reference-files',
                configureServer(devServer) {
                    devServer.middlewares.use(REFERENCE_FILES, (request, response) => {
                        const scene = new URL(request.url ?? '', 'http://localhost').searchParams.get('scene') ?? '';
                        response.setHeader('Content-Type', 'application/json');
                        const valid = /^[a-z0-9-]+$/.test(scene);
                        const folder = join(REFERENCES_ROOT, scene);
                        response.end(JSON.stringify({ url: `/@fs/${folder.split(sep).join('/')}`, files: valid ? listFiles(folder) : [] }));
                    });
                },
            },
            {
                // Chrome asks for /favicon.ico on every page and logs a 404 as a
                // console error, which the runner would count as a failure.
                name: 'no-favicon-404',
                configureServer(devServer) {
                    devServer.middlewares.use('/favicon.ico', (_request, response) => {
                        response.statusCode = 204;
                        response.end();
                    });
                },
            },
        ],
    });
    await server.listen();
    const origin = server.resolvedUrls?.local[0]?.replace(/\/$/, '');
    if (!origin) {
        await server.close();
        throw new Error('The vite test server started but reported no local address.');
    }
    return { origin, close: () => server.close() };
}
