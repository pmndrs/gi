/**
 * Reads the shot a Cycles reference was rendered from (`shot` in its json,
 * written by render.py; README "Stored camera"), so the DDGI test images can
 * be rendered from exactly the same camera, at the same size.
 */

import { createHash } from 'node:crypto';
import { existsSync } from 'node:fs';
import { readFile } from 'node:fs/promises';
import { join } from 'node:path';

/** render.py `SHOT_FORMAT`. */
export const SHOT_FORMAT = 'ddgi-reference-shot/1';

/** The parts of a stored shot the test pages use. */
export interface ReferenceShot {
    readonly format: string;
    readonly camera: {
        readonly bookmark: string;
        readonly bookmarkNumber: number | null;
        readonly position: [number, number, number];
        readonly target: [number, number, number];
        readonly fovYDeg: number;
        readonly width: number;
        readonly height: number;
    };
    readonly scene: { readonly manifest: string; readonly manifestPath: string; readonly manifestSha256: string };
}

/** A camera in the manifest's bookmark form, as the test pages take it. */
export interface ShotView {
    readonly name: string;
    readonly position: [number, number, number];
    readonly target: [number, number, number];
    readonly fovYDeg: number;
}

/**
 * The shot stored in `<directory>/<jsonName>`, or null when there is no such
 * file or it was rendered before shots were stored (T1.13). Prints a warning
 * when the scene's manifest has changed since the render.
 */
export async function readReferenceShot(repoRoot: string, directory: string, jsonName: string): Promise<ReferenceShot | null> {
    const path = join(directory, jsonName);
    if (!existsSync(path)) return null;
    const meta = JSON.parse(await readFile(path, 'utf8')) as { shot?: ReferenceShot };
    const shot = meta.shot;
    if (!shot || shot.format !== SHOT_FORMAT) return null;
    const manifestPath = join(repoRoot, shot.scene.manifestPath);
    if (existsSync(manifestPath)) {
        const current = createHash('sha256').update(await readFile(manifestPath)).digest('hex');
        if (current !== shot.scene.manifestSha256) {
            console.warn(`warning: ${shot.scene.manifestPath} has changed since ${jsonName} was rendered; the reference may show a different scene. Render it again.`);
        }
    }
    return shot;
}

/**
 * The size of the reference in `<directory>/<jsonName>` (its `width` and
 * `height`, also on references rendered before shots were stored), or null
 * when there is none. A DDGI image is compared pixel by pixel, so it must
 * match this.
 */
export async function referenceSize(directory: string, jsonName: string): Promise<{ width: number; height: number } | null> {
    const path = join(directory, jsonName);
    if (!existsSync(path)) return null;
    const meta = JSON.parse(await readFile(path, 'utf8')) as { width?: unknown; height?: unknown };
    return typeof meta.width === 'number' && typeof meta.height === 'number' ? { width: meta.width, height: meta.height } : null;
}

/** The size references are rendered at now (render.py's default, T1.13), for a view with no reference yet. */
export const DEFAULT_REFERENCE_SIZE = { width: 1920, height: 1080 } as const;

/** The shot's camera as a bookmark. */
export function shotView(shot: ReferenceShot): ShotView {
    const { bookmark, position, target, fovYDeg } = shot.camera;
    return { name: bookmark, position: [...position], target: [...target], fovYDeg };
}
