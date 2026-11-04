/**
 * Finds the path-traced reference images made by `tools/reference/cycles`
 * (`bench/references/<scene>/<camera name>/cycles-full.exr`, gitignored,
 * rendered on demand) and returns URLs the page can fetch.
 *
 * Like `model-url.ts`: the viewer's vite config injects the folder's absolute
 * path (`__REFERENCES_DIR__`), and the GPU test runner's config-less server
 * finds it relative to this module. A built viewer has neither, so every
 * reference reads as missing there, which is the truth for a deployed copy.
 */

import type { SceneManifest } from '../../../../scenes/src/manifest';

/** Absolute path of `bench/references`, injected by `apps/viewer/vite.config.ts`. */
declare const __REFERENCES_DIR__: string | undefined;

/** `full`: everything the camera sees. `indirect`: only light that bounced at least once (Cycles' diffuse indirect times colour). */
export type ReferenceKind = 'full' | 'indirect';

/** What `cycles.json` next to the images says about the render (the fields the viewer shows). */
export interface ReferenceInfo {
    width: number;
    height: number;
    samplesPerPixel: number;
    finishedAt: string;
    materials: string;
}

function candidateFolders(sceneId: string, cameraName: string): string[] {
    const folders: string[] = [];
    if (typeof __REFERENCES_DIR__ !== 'undefined') folders.push(`/@fs/${__REFERENCES_DIR__.replaceAll('\\', '/')}/${sceneId}/${cameraName}`);
    const here = new URL(import.meta.url);
    if (here.pathname.startsWith('/@fs/')) {
        // .../apps/viewer/src/scene/reference-files.ts -> .../bench/references/
        folders.push(new URL(`../../../../bench/references/${sceneId}/${cameraName}`, here).pathname);
    }
    return folders;
}

/** Vite answers a missing file with its HTML page, so a hit is checked by content type, not just status. */
async function exists(url: string): Promise<boolean> {
    try {
        const response = await fetch(url, { method: 'HEAD' });
        return response.ok && !(response.headers.get('content-type') ?? '').includes('text/html');
    } catch {
        return false;
    }
}

export interface FoundReference {
    imageUrl: string;
    info: ReferenceInfo | null;
}

/** The reference image of one camera view, or null when it hasn't been rendered (or this copy of the viewer can't reach it). */
export async function findReference(sceneId: string, cameraName: string, kind: ReferenceKind): Promise<FoundReference | null> {
    for (const folder of candidateFolders(sceneId, cameraName)) {
        const imageUrl = `${folder}/cycles-${kind}.exr`;
        if (!(await exists(imageUrl))) continue;
        let info: ReferenceInfo | null = null;
        try {
            const response = await fetch(`${folder}/cycles.json`);
            if (response.ok) info = (await response.json()) as ReferenceInfo;
        } catch {
            // The settings file is a nice-to-have for the status line; the image is what matters.
        }
        return { imageUrl, info };
    }
    return null;
}

/**
 * The command that renders a missing reference, as `tools/reference/README.md` gives it.
 * 1024 samples at 640x360 takes a few minutes on this laptop's CPU (Sponza: about 5).
 */
export function renderCommand(manifest: Pick<SceneManifest, 'id'>, bookmarkNumber: number): string {
    return `node tools/reference/cycles/run.ts --scene ${manifest.id} --camera ${bookmarkNumber} --samples 1024 --width 640 --height 360`;
}
