/**
 * Finds a downloaded model file (`assets/downloads/<asset>/<path>`) and
 * returns a URL the page can fetch.
 *
 * Vite serves any file under its `server.fs.allow` folders at
 * `/@fs/<absolute path>`. Two servers load this module:
 *  - the viewer's own (`apps/viewer/vite.config.ts` injects `__DOWNLOADS_DIRS__`);
 *  - the GPU test runner's, which has no config. There the module itself is
 *    served from `/@fs/<repo>/apps/viewer/src/...`, so the repo's `assets`
 *    folder is found relative to `import.meta.url`.
 *
 * Folders are tried in order and the first that has the file wins. Vite
 * answers a missing file with its HTML page, so each candidate is checked by
 * content type, not just by status.
 */

/** Absolute paths of the downloads folders, injected by `apps/viewer/vite.config.ts`. */
declare const __DOWNLOADS_DIRS__: string[] | undefined;

function candidateUrls(asset: string, path: string): string[] {
    const urls: string[] = [];
    if (typeof __DOWNLOADS_DIRS__ !== 'undefined') {
        for (const directory of __DOWNLOADS_DIRS__) {
            urls.push(`/@fs/${directory.replaceAll('\\', '/')}/${asset}/${path}`);
        }
    }
    const here = new URL(import.meta.url);
    if (here.pathname.startsWith('/@fs/')) {
        // .../apps/viewer/src/scene/model-url.ts -> .../assets/downloads/
        urls.push(new URL(`../../../../assets/downloads/${asset}/${path}`, here).pathname);
        // A clean re-fetch can leave a converted output only in the backup folder for a while.
        urls.push(new URL(`../../../../assets/downloads.bak/${asset}/${path}`, here).pathname);
    }
    return urls;
}

export async function findModelUrl(asset: string, path: string): Promise<string> {
    const candidates = candidateUrls(asset, path);
    for (const url of candidates) {
        const response = await fetch(url, { method: 'HEAD' });
        const isHtml = (response.headers.get('content-type') ?? '').includes('text/html');
        if (response.ok && !isHtml) return url;
    }
    throw new Error(
        `model file ${asset}/${path} not found under assets/downloads (tried ${candidates.length} folder(s)); run "pnpm assets:fetch --only ${asset}"`,
    );
}
