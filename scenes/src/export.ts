/// <reference types="node" />
/**
 * Writes every procedural scene to `assets/generated/<id>.glb`.
 *
 *   pnpm scenes:export
 *
 * The native harness (T0.6) and Blender (T0.7) read these files. Scenes with
 * model parts are skipped: their geometry comes from `pnpm assets:fetch`.
 */

import { mkdir, readdir, readFile, writeFile } from 'node:fs/promises';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { buildProceduralParts, mergeScenes } from './build-scene';
import { writeGlb } from './glb';
import { isProceduralOnly, parseManifest, type SceneManifest } from './manifest';
import { proceduralBuilders } from './procedural';

const scenesDirectory = join(dirname(fileURLToPath(import.meta.url)), '..');
export const manifestsDirectory = join(scenesDirectory, 'manifests');
export const generatedDirectory = join(scenesDirectory, '..', 'assets', 'generated');

/** Reads and validates every manifest in `scenes/manifests/`. Throws on the first invalid one. */
export async function loadManifests(directory = manifestsDirectory): Promise<SceneManifest[]> {
    const files = (await readdir(directory)).filter((file) => file.endsWith('.json')).sort();
    const manifests: SceneManifest[] = [];
    for (const file of files) {
        const text = await readFile(join(directory, file), 'utf8');
        const manifest = parseManifest(text, file, { knownBuilders: Object.keys(proceduralBuilders) });
        if (`${manifest.id}.json` !== file) {
            throw new Error(`${file}: the file name must match the manifest id "${manifest.id}"`);
        }
        manifests.push(manifest);
    }
    return manifests;
}

/** Exports one procedural-only manifest and returns the file path written. */
export async function exportManifest(manifest: SceneManifest, outputDirectory = generatedDirectory): Promise<string> {
    const scene = mergeScenes(buildProceduralParts(manifest));
    const glb = writeGlb(scene, { name: manifest.title, extras: { manifestId: manifest.id } });
    await mkdir(outputDirectory, { recursive: true });
    const outputPath = join(outputDirectory, `${manifest.id}.glb`);
    await writeFile(outputPath, glb);
    return outputPath;
}

async function main(): Promise<void> {
    const manifests = await loadManifests();
    let written = 0;
    for (const manifest of manifests) {
        if (!isProceduralOnly(manifest)) {
            console.log(`skip   ${manifest.id} (has downloaded model parts)`);
            continue;
        }
        const path = await exportManifest(manifest);
        console.log(`wrote  ${manifest.id} -> ${path}`);
        written++;
    }
    console.log(`${written} procedural scene(s) exported to ${generatedDirectory}`);
}

// Run only when started as a script, not when a test imports the helpers above.
if (process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1]) {
    main().catch((error: unknown) => {
        console.error(error instanceof Error ? error.message : error);
        process.exit(1);
    });
}
