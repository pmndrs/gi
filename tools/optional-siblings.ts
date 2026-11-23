/**
 * Optional sibling libraries (root package.json `linkedSiblings` entries with
 * `optional: true`): development code from a private sibling repository that
 * a public checkout can't fetch. Today: `@pathtracer/mlp` from
 * pmndrs/pathtracer, which the radiance cache's neural tail (P6.4) loads with
 * a dynamic import behind `neuralCache: 'tail'`.
 *
 * Not a `link:` dependency: pnpm would need the folder at install time. The
 * import resolves here instead (vite, vitest) and through tsconfig `paths`
 * (TypeScript, with a declaration-only fallback when the clone is missing):
 *
 * - the clone at the pinned commit: the package's entry file;
 * - missing, or checked out at another commit: a module that throws when
 *   loaded, with the reason, so the tail turns off with one warning and gi
 *   never builds whatever branch the sibling clone happens to be on.
 *
 * `node tools/linked-siblings.ts --fetch` clones the pinned commit where the
 * repository is reachable. A published `@pathtracer/mlp` (or the pathtracer
 * repository going public) would make this a normal dependency.
 */

import { execFileSync } from 'node:child_process';
import { existsSync, readFileSync, realpathSync } from 'node:fs';
import { join, resolve } from 'node:path';
import type { Plugin } from 'vite';

const ROOT = resolve(import.meta.dirname, '..');

export interface SiblingEntry {
    readonly repo: string;
    readonly commit: string;
    readonly clone: string;
    readonly path: string;
    readonly built: readonly string[];
    readonly optional?: boolean;
}

/** An optional sibling as this checkout has it. */
export interface OptionalSibling {
    readonly name: string;
    /** `ready`: cloned at the pinned commit; `missing`: no clone; `drifted`: a clone at another commit. */
    readonly state: 'ready' | 'missing' | 'drifted';
    /** The package's entry file (its first `built` file), where the clone has it. */
    readonly entry: string | null;
    /** The package folder, real path, where it exists (for the dev server's `fs.allow`). */
    readonly folder: string | null;
    /** Why it isn't ready, in a sentence a developer can act on. */
    readonly reason: string;
}

function siblingEntries(): Record<string, SiblingEntry> {
    const manifest = JSON.parse(readFileSync(join(ROOT, 'package.json'), 'utf8')) as { linkedSiblings?: Record<string, SiblingEntry> };
    return manifest.linkedSiblings ?? {};
}

function headOf(clone: string): string {
    try {
        return execFileSync('git', ['rev-parse', 'HEAD'], { cwd: clone, encoding: 'utf8', stdio: ['ignore', 'pipe', 'ignore'] }).trim();
    } catch {
        return '';
    }
}

/**
 * One optional sibling's state: `entry` as package.json pins it, `root` this
 * repository, `head` the clone's commit ('' where it isn't a git clone). The
 * drift check: a clone at another commit than its pin is `drifted`, and gi
 * refuses to load it.
 */
export function optionalSiblingState(name: string, entry: SiblingEntry, root: string, head: (clone: string) => string = headOf): OptionalSibling {
    const clone = resolve(root, entry.clone);
    const folder = join(clone, entry.path);
    const file = join(folder, entry.built[0] ?? 'src/index.ts');
    if (!existsSync(file)) {
        return { name, state: 'missing', entry: null, folder: null, reason: `${name} is an optional sibling from ${entry.repo} (private), not cloned at ${clone}; run node tools/linked-siblings.ts --fetch where you have access` };
    }
    const commit = head(clone);
    if (commit && commit !== entry.commit) {
        return {
            name,
            state: 'drifted',
            entry: file,
            folder: realpathSync(folder),
            reason: `${name}'s clone at ${clone} is at ${commit.slice(0, 9)}, but package.json pins ${entry.commit.slice(0, 9)}; check out the pinned commit (or update the pin)`,
        };
    }
    return { name, state: 'ready', entry: file, folder: realpathSync(folder), reason: '' };
}

/** Every optional sibling of the root package.json, with its state. */
export function optionalSiblings(): OptionalSibling[] {
    return Object.entries(siblingEntries())
        .filter(([, entry]) => entry.optional === true)
        .map(([name, entry]) => optionalSiblingState(name, entry, ROOT));
}

const VIRTUAL_PREFIX = '\0optional-sibling:';

/**
 * Resolves each optional sibling for vite and vitest: its entry file when the
 * clone is at the pinned commit, else a module that throws the reason when
 * loaded (a dynamic import of it rejects; nothing imports it statically).
 */
export function optionalSiblingsPlugin(): Plugin {
    const siblings = new Map(optionalSiblings().map((sibling) => [sibling.name, sibling]));
    return {
        name: 'gi-optional-siblings',
        enforce: 'pre',
        resolveId(id) {
            const sibling = siblings.get(id);
            if (!sibling) return null;
            return sibling.state === 'ready' && sibling.entry ? sibling.entry : `${VIRTUAL_PREFIX}${id}`;
        },
        load(id) {
            if (!id.startsWith(VIRTUAL_PREFIX)) return null;
            const sibling = siblings.get(id.slice(VIRTUAL_PREFIX.length));
            return `throw new Error(${JSON.stringify(`@pmndrs/gi: ${sibling?.reason ?? 'an optional sibling is unavailable'}`)});\n`;
        },
    };
}
