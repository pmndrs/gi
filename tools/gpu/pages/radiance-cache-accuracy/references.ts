/**
 * The Cycles references a GI accuracy page compares with, found and read in
 * the browser (no Node side), so the page runs as an ordinary
 * `pnpm test:gpu --filter <page>` under the bench lock.
 *
 * A view is a scene's manifest bookmark with a rendered reference folder
 * (`bench/references/<scene>/<bookmark name>/`, gitignored; a junction to
 * another checkout's folder works), listed by the test server
 * (`REFERENCE_FILES`). The render that matches GI with its textures on is
 * picked as the viewer's `findReference` does: `cycles-textured-*` where it
 * exists, else the plain `cycles-*` (scenes without textures, metals or
 * glass, where the two are the same). Of that render, the page reads the
 * GI composite and its parts (tools/reference/README.md, "The GI
 * composite"):
 *
 * - `<prefix>-gi.exr`: what a GI technique's term is compared with
 *   (indirect + direct − the manifest lights' direct light);
 * - `<prefix>-full.exr`: the whole picture, for the composite;
 * - `<prefix>-lights-direct.exr` and `<prefix>-indirect.exr`: the light the
 *   radiance cache's world cache holds (direct light from the manifest's
 *   lights, and the bounces), to judge the world cache on its own.
 *
 * Images come back as RGB floats with row 0 at the top (three's EXRLoader
 * keeps the bottom row first; flipped here). The stored shot (`shot` in the
 * render's json) gives the camera and size the page must render at.
 */

import { EXRLoader } from 'three/addons/loaders/EXRLoader.js';
import { FloatType } from 'three/webgpu';
import { SHOT_FORMAT, shotProvenance, type ReferenceInfo, type ReferenceMaterials, type ReferenceShot } from '../../../../apps/viewer/src/scene/reference-files.ts';
import { REFERENCE_FILES } from '../../contract.ts';
import { rgbOfRgba, type RgbImage } from './accuracy.ts';

/** The parts of a reference render the page reads, by the file suffix after the render's prefix. */
export const REFERENCE_PARTS = { gi: 'gi', full: 'full', lightsDirect: 'lights-direct', indirect: 'indirect' } as const;
export type ReferencePart = keyof typeof REFERENCE_PARTS;

export interface ReferenceView {
    readonly scene: string;
    /** The bookmark's name, which is also its reference folder's name. */
    readonly camera: string;
    /** `/@fs/...` URL of the folder. */
    readonly folderUrl: string;
    /** The render's file prefix, e.g. `cycles-textured`. */
    readonly prefix: string;
    readonly materials: ReferenceMaterials;
    readonly info: ReferenceInfo;
    readonly shot: ReferenceShot;
    /** False when the scene's manifest changed since the render (the reference may show another scene). */
    readonly sameManifest: boolean | null;
}

/** One scene's reference files as the test server lists them (`REFERENCE_FILES`). */
export interface SceneReferenceFiles {
    /** The scene's folder as served (`/@fs/.../bench/references/<scene>`). */
    readonly url: string;
    /** `<camera>/<file>` paths. */
    readonly paths: ReadonlySet<string>;
}

/** Lists a scene's reference files through the test server (no probing, so no 404s in the log). */
export async function listReferenceFiles(scene: string): Promise<SceneReferenceFiles> {
    const response = await fetch(`${REFERENCE_FILES}?scene=${encodeURIComponent(scene)}`);
    if (!response.ok) throw new Error(`${REFERENCE_FILES}: ${response.status}`);
    const listing = (await response.json()) as { url: string; files: { path: string }[] };
    return { url: listing.url, paths: new Set(listing.files.map((file) => file.path)) };
}

/**
 * Render prefixes, best first, as the viewer's `findReference` picks them with
 * GI's textures on: a textured render, then the plain one.
 */
const PREFIXES: readonly { prefix: string; materials: ReferenceMaterials }[] = [
    { prefix: 'cycles-textured', materials: 'textured' },
    { prefix: 'cycles', materials: 'plain' },
];

/**
 * The reference of one bookmark, or null when its folder has no render with
 * every part and a stored shot (not rendered, or rendered before T1.13
 * stored shots).
 */
export async function findReferenceView(files: SceneReferenceFiles, scene: string, camera: string): Promise<ReferenceView | null> {
    for (const { prefix, materials } of PREFIXES) {
        const parts = Object.values(REFERENCE_PARTS).map((part) => `${camera}/${prefix}-${part}.exr`);
        if (!parts.every((path) => files.paths.has(path)) || !files.paths.has(`${camera}/${prefix}.json`)) continue;
        const folderUrl = `${files.url}/${camera}`;
        const response = await fetch(`${folderUrl}/${prefix}.json`);
        if (!response.ok) continue;
        const info = (await response.json()) as ReferenceInfo;
        const shot = info.shot;
        if (!shot || shot.format !== SHOT_FORMAT) continue;
        const provenance = await shotProvenance(shot);
        return { scene, camera, folderUrl, prefix, materials, info, shot, sameManifest: provenance.sameManifest };
    }
    return null;
}

/** One part of a reference render as RGB floats, top row first; throws when the file is missing. */
export async function loadReferencePart(view: ReferenceView, part: ReferencePart): Promise<RgbImage> {
    const url = `${view.folderUrl}/${view.prefix}-${REFERENCE_PARTS[part]}.exr`;
    const loader = new EXRLoader();
    loader.setDataType(FloatType);
    const texture = await loader.loadAsync(url);
    const image = texture.image as { data: Float32Array; width: number; height: number };
    texture.dispose();
    const { width, height } = view.shot.camera;
    if (image.width !== width || image.height !== height) throw new Error(`${url}: ${image.width} x ${image.height}, the shot says ${width} x ${height}`);
    return rgbOfRgba(image.data, image.width, image.height, true);
}
