/**
 * Serves the test pages with vite's dev server, so a page can import
 * TypeScript (including the shared client helper) with no build step.
 */

import { createServer, type ViteDevServer } from 'vite';

/** The folder served as the site root: it holds client.ts, preflight.html and pages/. */
export const SITE_ROOT = import.meta.dirname;

export interface TestSiteServer {
    /** For example http://127.0.0.1:5290 */
    origin: string;
    close(): Promise<void>;
}

export async function startTestSite(): Promise<TestSiteServer> {
    const server: ViteDevServer = await createServer({
        root: SITE_ROOT,
        // Do not pick up a vite config from the repo: the test site is self-contained.
        configFile: false,
        logLevel: 'warn',
        appType: 'mpa',
        // The pages have no npm dependencies; skip the dependency pre-bundling scan.
        optimizeDeps: { noDiscovery: true, include: [] },
        server: {
            host: '127.0.0.1',
            port: 5290,
            // Take the next free port if a dev server is already running.
            strictPort: false,
            // Pages are reloaded by the runner; hot reload would only add noise.
            hmr: false,
            watch: null,
        },
        plugins: [
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
