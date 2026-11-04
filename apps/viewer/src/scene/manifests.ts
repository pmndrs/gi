/**
 * Scene manifests for the viewer: the same JSON files the native harness and
 * the Blender reference read (`scenes/manifests/*.json`), validated on load.
 *
 * Only relative imports here (no vite aliases), so the GPU test runner's own
 * vite server can load this module for the baseline benchmark page.
 */

import { validateManifest, type CameraBookmark, type SceneManifest } from '../../../../scenes/src/manifest';
import { proceduralBuilders } from '../../../../scenes/src/procedural';

const manifestModules = import.meta.glob('../../../../scenes/manifests/*.json', { eager: true, import: 'default' }) as Record<
    string,
    unknown
>;

/** Every manifest, validated, sorted by id. Throws on the first invalid one. */
export function loadAllManifests(): SceneManifest[] {
    const manifests: SceneManifest[] = [];
    for (const [file, value] of Object.entries(manifestModules)) {
        const result = validateManifest(value, { knownBuilders: Object.keys(proceduralBuilders) });
        if (!result.ok) throw new Error(`${file} is not a valid manifest: ${result.errors.join('; ')}`);
        manifests.push(value as SceneManifest);
    }
    return manifests.sort((a, b) => a.id.localeCompare(b.id));
}

/**
 * Finds a bookmark by 1-based number or by name. Returns the first bookmark
 * when `request` is null; null when it matches nothing.
 */
export function findBookmark(manifest: SceneManifest, request: string | null): { bookmark: CameraBookmark; index: number } | null {
    if (request === null || request === '') return { bookmark: manifest.cameras[0], index: 0 };
    const byName = manifest.cameras.findIndex((camera) => camera.name === request);
    if (byName >= 0) return { bookmark: manifest.cameras[byName], index: byName };
    const oneBased = Number(request);
    if (Number.isInteger(oneBased) && oneBased >= 1 && oneBased <= manifest.cameras.length) {
        return { bookmark: manifest.cameras[oneBased - 1], index: oneBased - 1 };
    }
    return null;
}

/** "colonnade-from-the-floor" becomes "Colonnade from the floor". */
export function bookmarkLabel(bookmark: CameraBookmark): string {
    const words = bookmark.name.replaceAll('-', ' ');
    return words.charAt(0).toUpperCase() + words.slice(1);
}
