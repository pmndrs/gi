/**
 * Sibling libraries linked from a local clone for development (root package.json `link:`),
 * pinned in package.json's `linkedSiblings`. Today: `@pmndrs/denoiser-ffx` from pmndrs/denoiser
 * (the radiance cache's GI denoiser, RC4.3), until a release of `denoiser` carries it.
 *
 * A git dependency can't stand in: pnpm installs a git subdirectory without its workspace
 * (`@pmndrs/denoiser-core@workspace:*` doesn't resolve) and the package ships no built output.
 *
 *   node tools/linked-siblings.ts           checks every sibling is cloned and built (the root
 *                                           `preinstall` runs this: install stops with the fix)
 *   node tools/linked-siblings.ts --fetch   clones each missing sibling next to this repo at its
 *                                           pinned commit and builds it (corepack + yarn), as CI does
 */

import { execFileSync } from 'node:child_process';
import { existsSync, readFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';

interface Sibling {
    /** Clone URL. */
    readonly repo: string;
    /** The commit the link was last checked against. */
    readonly commit: string;
    /** The linked package inside the clone. */
    readonly path: string;
    /** Where the clone goes, relative to this repository's root (the `link:` target's clone). */
    readonly clone: string;
    /** Files a built clone has, relative to the package. */
    readonly built: readonly string[];
    /** Commands (each an argv) that build the package inside the clone. */
    readonly build: readonly (readonly string[])[];
}

const ROOT = resolve(import.meta.dirname, '..');
const manifest = JSON.parse(readFileSync(join(ROOT, 'package.json'), 'utf8')) as { linkedSiblings?: Record<string, Sibling> };
const siblings = Object.entries(manifest.linkedSiblings ?? {});
const fetch = process.argv.includes('--fetch');

function run(command: readonly string[], cwd: string): void {
    // No git LFS objects: the build needs none (pmndrs/denoiser keeps weights and scene assets in LFS).
    const env = { ...process.env, GIT_LFS_SKIP_SMUDGE: '1' };
    execFileSync(command[0], command.slice(1), { cwd, stdio: 'inherit', shell: process.platform === 'win32', env });
}

const problems: string[] = [];
for (const [name, sibling] of siblings) {
    const clone = resolve(ROOT, sibling.clone);
    const packageDir = join(clone, sibling.path);
    const isBuilt = (): boolean => sibling.built.every((file) => existsSync(join(packageDir, file)));
    if (fetch && !existsSync(clone)) {
        console.log(`linked-siblings: cloning ${sibling.repo} at ${sibling.commit} into ${clone}`);
        // Long paths: the denoiser repo has research files past Windows' 260-character limit.
        run(['git', '-c', 'core.longpaths=true', 'clone', '--quiet', '--no-checkout', sibling.repo, clone], dirname(clone));
        run(['git', 'config', 'core.longpaths', 'true'], clone);
        run(['git', 'checkout', '--quiet', sibling.commit], clone);
    }
    if (fetch && existsSync(clone) && !isBuilt()) {
        console.log(`linked-siblings: building ${name} in ${clone}`);
        for (const command of sibling.build) run(command, clone);
    }
    if (!existsSync(packageDir)) {
        problems.push(`${name} is linked from ${packageDir}, which doesn't exist.`);
    } else if (!isBuilt()) {
        problems.push(`${name} at ${packageDir} isn't built (missing ${sibling.built.join(', ')}).`);
    } else {
        let head = '';
        try {
            head = execFileSync('git', ['rev-parse', 'HEAD'], { cwd: clone, encoding: 'utf8' }).trim();
        } catch {
            // Not a git clone: fine, it is built.
        }
        if (head && head !== sibling.commit) {
            console.warn(`linked-siblings: ${name}'s clone is at ${head.slice(0, 9)}; package.json pins ${sibling.commit.slice(0, 9)} (fine while developing both)`);
        }
    }
}

if (problems.length > 0) {
    console.error(
        [
            'linked-siblings: this branch uses a sibling library from a local clone (development only):',
            ...problems.map((problem) => `  - ${problem}`),
            'Fix it once with:  node tools/linked-siblings.ts --fetch',
            '(clones the pinned commit next to this repository and builds it; needs git and corepack). See docs/HANDOFF-radiance-cache.md.',
        ].join('\n'),
    );
    process.exit(1);
}
