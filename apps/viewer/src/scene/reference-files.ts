/**
 * Finds the path-traced reference images made by `tools/reference/cycles`
 * (`bench/references/<scene>/<camera name>/cycles-full.exr`, gitignored,
 * rendered on demand) and returns URLs the page can fetch, with the stored
 * shot each one was rendered from (camera, size, exposure, tone mapping,
 * scene hashes, commit: `shot` in `cycles.json`, written by render.py).
 *
 * Like `model-url.ts`: the viewer's vite config injects the folder's absolute
 * path (`__REFERENCES_DIR__`), and the GPU test runner's config-less server
 * finds it relative to this module. A built viewer has neither, so every
 * reference reads as missing there, which is the truth for a deployed copy.
 */

import type { CameraBookmark, SceneManifest } from '../../../../scenes/src/manifest';

/** Absolute path of `bench/references`, injected by `apps/viewer/vite.config.ts`. */
declare const __REFERENCES_DIR__: string | undefined;
/** The repo commit the viewer's dev server started at, injected by `apps/viewer/vite.config.ts`. */
declare const __VIEWER_COMMIT__: string | undefined;

/**
 * `full`: everything the camera sees. `gi`: what DDGI's field holds, Cycles'
 * GI composite (indirect + direct − the manifest lights' direct light;
 * `tools/reference/README.md`, "The GI composite"); older folders without it
 * fall back to `cycles-indirect.exr`.
 */
export type ReferenceKind = 'full' | 'gi';

/** The format render.py writes (`SHOT_FORMAT`); a shot in another format is treated as absent. */
export const SHOT_FORMAT = 'ddgi-reference-shot/1';

/** What a reference was rendered from: enough to see exactly that view again (render.py `shot`). */
export interface ReferenceShot {
    format: string;
    camera: {
        bookmark: string;
        bookmarkNumber: number | null;
        /** Manifest space: three.js axes, Y up, metres. */
        position: [number, number, number];
        target: [number, number, number];
        fovYDeg: number;
        aspect: number;
        width: number;
        height: number;
        near: number;
        far: number;
    };
    display: { exposureEv: number; toneMapping: string };
    scene: { manifest: string; manifestPath: string; manifestSha256: string; skySha256: string | null; giMaterialsSha256: string | null };
    render: {
        /**
         * `uncommittedChanges`: render-affecting files (`uncommittedScope`: tools/reference,
         * scenes, tools/sky) had uncommitted edits. References rendered before the scope was
         * recorded flagged any edit in the repo, so their flag isn't shown.
         */
        git: { commit: string; uncommittedChanges: boolean; uncommittedScope?: string[] } | null;
        blender: string;
        device: string;
        samplesPerPixel: number;
        seed: number;
        pixelFilter: { type: string; widthPixels: number };
        renderSeconds: number;
        denoised: { file: string; denoiser: string; check?: { bias: number; blockError: number } } | null;
        /** The raw images' noise, measured against an independent render (tools/reference/cycles/noise.ts). */
        noise?: ReferenceNoise;
    };
}

/**
 * One image's Monte Carlo noise: the mean relative error it would show against
 * the converged image, per pixel and after 8 x 8 block averaging (noise.ts).
 */
export interface ImageNoise {
    pixel: number;
    block8: number;
}

export interface ReferenceNoise {
    gi: ImageNoise;
    full: ImageNoise;
    /** How it was measured, e.g. 'against a second render, seed 7, 128 samples'. */
    method: string;
}

/** What `cycles.json` next to the images says about the render (the fields the viewer uses). */
export interface ReferenceInfo {
    width: number;
    height: number;
    samplesPerPixel: number;
    finishedAt: string;
    materials: string;
    /** Missing on references rendered before T1.13 (2026-11-05). */
    shot?: ReferenceShot;
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
    /** The file shown, e.g. `cycles-full-denoised.exr`. */
    fileName: string;
    /** True when the file is the denoised viewing copy (render.py `--denoise`). */
    denoised: boolean;
    info: ReferenceInfo | null;
}

/**
 * Files to try for each kind, best first. The full image prefers the
 * denoised viewing copy when there is one: it is for looking at, and the raw
 * image stays on disk for accuracy numbers.
 */
const FILES: Record<ReferenceKind, readonly string[]> = {
    full: ['cycles-full-denoised.exr', 'cycles-full.exr'],
    gi: ['cycles-gi.exr', 'cycles-indirect.exr'],
};

/** The reference image of one camera view, or null when it hasn't been rendered (or this copy of the viewer can't reach it). */
export async function findReference(sceneId: string, cameraName: string, kind: ReferenceKind): Promise<FoundReference | null> {
    for (const folder of candidateFolders(sceneId, cameraName)) {
        for (const fileName of FILES[kind]) {
            const imageUrl = `${folder}/${fileName}`;
            if (!(await exists(imageUrl))) continue;
            let info: ReferenceInfo | null = null;
            try {
                const response = await fetch(`${folder}/cycles.json`);
                if (response.ok) info = (await response.json()) as ReferenceInfo;
            } catch {
                // The settings file is a nice-to-have for the status line; the image is what matters.
            }
            if (info?.shot && info.shot.format !== SHOT_FORMAT) info.shot = undefined;
            return { imageUrl, fileName, denoised: fileName.includes('denoised'), info };
        }
    }
    return null;
}

/** The stored shot's camera as a bookmark the camera rig can go to. */
export function shotBookmark(shot: ReferenceShot): CameraBookmark {
    const { bookmark, position, target, fovYDeg } = shot.camera;
    return { name: bookmark, position: [...position], target: [...target], fovYDeg };
}

// The manifests' exact file text, to compare with the hash a reference stored.
const rawManifests = import.meta.glob('../../../../scenes/manifests/*.json', { eager: true, query: '?raw', import: 'default' }) as Record<string, string>;
const manifestHashes = new Map<string, Promise<string | null>>();

async function sha256Hex(text: string): Promise<string> {
    const digest = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(text));
    return Array.from(new Uint8Array(digest), (byte) => byte.toString(16).padStart(2, '0')).join('');
}

/** sha256 of `scenes/manifests/<id>.json` as the viewer has it, or null when it can't tell. */
function manifestSha256(sceneId: string): Promise<string | null> {
    let hash = manifestHashes.get(sceneId);
    if (!hash) {
        const entry = Object.entries(rawManifests).find(([path]) => path.endsWith(`/${sceneId}.json`));
        hash = entry && typeof crypto !== 'undefined' && crypto.subtle ? sha256Hex(entry[1]) : Promise.resolve(null);
        manifestHashes.set(sceneId, hash);
    }
    return hash;
}

/** How a stored reference relates to what the viewer is showing now. */
export interface ShotProvenance {
    /** False when the scene's manifest changed since the render: the picture may show another scene. */
    sameManifest: boolean | null;
    /** The render's commit, short, or null if it wasn't recorded. */
    commit: string | null;
    /** True when the viewer runs a different commit (or the render had uncommitted changes). */
    olderCommit: boolean | null;
    /** Plain-language lines for the panel; warnings first. */
    lines: string[];
}

/** Compares a stored shot with the viewer's manifest and commit. */
export async function shotProvenance(shot: ReferenceShot): Promise<ShotProvenance> {
    const lines: string[] = [];
    const current = await manifestSha256(shot.scene.manifest);
    const sameManifest = current === null ? null : current === shot.scene.manifestSha256;
    if (sameManifest === false) {
        lines.push(`Warning: the scene file (${shot.scene.manifestPath}) has changed since this reference was rendered, so it may not show the same scene. Render it again.`);
    }
    const noise = shot.render.noise;
    if (noise) {
        const percent = (value: number): string => `${(value * 100).toFixed(1)}%`;
        lines.push(
            `Its noise (raw ${shot.render.samplesPerPixel.toLocaleString('en')}-sample image): GI only ${percent(noise.gi.pixel)} per pixel, ${percent(noise.gi.block8)} over 8 x 8 blocks; ` +
                `full image ${percent(noise.full.pixel)} per pixel, ${percent(noise.full.block8)} over 8 x 8 blocks.`,
        );
    } else {
        lines.push('Its noise was not measured.');
    }
    const git = shot.render.git;
    const commit = git ? git.commit.slice(0, 7) : null;
    const viewerCommit = typeof __VIEWER_COMMIT__ === 'string' && __VIEWER_COMMIT__ ? __VIEWER_COMMIT__ : null;
    let olderCommit: boolean | null = null;
    if (git) {
        // Only a flag recorded with its scope means the render's own inputs were edited.
        const inputsEdited = git.uncommittedChanges && git.uncommittedScope !== undefined;
        olderCommit = inputsEdited || (viewerCommit !== null && viewerCommit !== git.commit);
        const changes = inputsEdited ? ' with uncommitted edits to the render tools or scene files' : '';
        if (viewerCommit && viewerCommit !== git.commit) lines.push(`Rendered from an older commit (${commit}${changes}); the viewer runs ${viewerCommit.slice(0, 7)}.`);
        else lines.push(`Rendered at commit ${commit}${changes}.`);
    } else {
        lines.push('The commit it was rendered at was not recorded.');
    }
    return { sameManifest, commit, olderCommit, lines };
}

/**
 * The command that renders a missing reference, as `tools/reference/README.md` gives it:
 * HD on the GPU (OptiX) with a denoised viewing copy. 1024 samples is a quick look
 * (Cornell about a minute); the README lists what the accuracy references use.
 */
export function renderCommand(manifest: Pick<SceneManifest, 'id'>, bookmarkNumber: number): string {
    return `node tools/reference/cycles/run.ts --scene ${manifest.id} --camera ${bookmarkNumber} --samples 1024 --denoise`;
}
