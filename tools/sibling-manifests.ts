/**
 * Where the sibling libraries are pinned: the `linkedSiblings` field of the
 * root package.json and of each workspace package under packages/. A sibling
 * is declared by the package that uses it (the radiance cache, packages/cache,
 * declares `@pathtracer/mlp`); its `clone` is relative to the repository root
 * whichever manifest names it, since every checkout keeps its clones next to
 * the repository.
 *
 * Node built-ins only: `tools/linked-siblings.ts` runs it as the root
 * `preinstall`, before anything is installed.
 */

import { existsSync, readdirSync, readFileSync } from 'node:fs';
import { join } from 'node:path';

/** Every `linkedSiblings` entry in the repository at `root`, by package name. */
export function linkedSiblingEntries<Entry>(root: string): Record<string, Entry> {
    const packages = join(root, 'packages');
    const manifests = [
        join(root, 'package.json'),
        ...(existsSync(packages) ? readdirSync(packages, { withFileTypes: true }) : [])
            .filter((entry) => entry.isDirectory())
            .map((entry) => join(packages, entry.name, 'package.json'))
            .filter((path) => existsSync(path)),
    ];
    const entries: Record<string, Entry> = {};
    for (const path of manifests) {
        const manifest = JSON.parse(readFileSync(path, 'utf8')) as { linkedSiblings?: Record<string, Entry> };
        for (const [name, entry] of Object.entries(manifest.linkedSiblings ?? {})) {
            if (name in entries) throw new Error(`${name} is pinned in more than one package.json linkedSiblings (again in ${path}); pin it once, in the package that uses it`);
            entries[name] = entry;
        }
    }
    return entries;
}
