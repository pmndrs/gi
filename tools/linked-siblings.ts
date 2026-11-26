/**
 * Sibling libraries from a local clone for development, pinned in package.json's
 * `linkedSiblings` (repo, commit, clone, path, built, build; `optional` for a private one).
 *
 * - Required (a `link:` dependency): `@pmndrs/denoiser-ffx` from pmndrs/denoiser (the radiance
 *   cache's GI denoiser, RC4.3, on its branch), until a release of `denoiser` carries it. A git
 *   dependency can't stand in: pnpm installs a git subdirectory without its workspace
 *   (`@pmndrs/denoiser-core@workspace:*` doesn't resolve) and the package ships no built output.
 * - Optional (`optional: true`, no `link:`): `@pathtracer/mlp` from pmndrs/pathtracer (the
 *   radiance cache's neural tail, P6.4; a private repository, TypeScript source, nothing to
 *   build). A public checkout can't fetch it, so nothing stops without it: vite, vitest and
 *   TypeScript resolve it through `tools/optional-siblings.ts` and tsconfig `paths`, and the
 *   tail turns off with one warning. A published `@pathtracer/mlp` (or the repository going
 *   public) would make it a normal dependency.
 *
 *   node tools/linked-siblings.ts           checks every sibling (the root `preinstall` runs it):
 *                                           a required one missing or unbuilt stops the install
 *                                           with the fix (at another commit than its pin: a
 *                                           warning, fine while developing both); an optional
 *                                           one only says so
 *   node tools/linked-siblings.ts --fetch   clones each missing sibling next to this repo at its
 *                                           pinned commit and builds it (corepack + yarn), as CI
 *                                           does; an optional one it can't clone (no access) is
 *                                           skipped with a note; a clone at another commit is
 *                                           left as it is, with a warning
 */

import { execFileSync } from 'node:child_process';
import { existsSync, readFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';

interface Sibling {
    /** Clone URL. */
    readonly repo: string;
    /** The commit the link was last checked against. */
    readonly commit: string;
    /** The package inside the clone. */
    readonly path: string;
    /** Where the clone goes, relative to this repository's root. */
    readonly clone: string;
    /** Files a built clone has, relative to the package. */
    readonly built: readonly string[];
    /** Commands (each an argv) that build the package inside the clone. */
    readonly build: readonly (readonly string[])[];
    /** A private sibling a public checkout can't fetch: its absence (or drift) stops nothing here. */
    readonly optional?: boolean;
}

const ROOT = resolve(import.meta.dirname, '..');
const manifest = JSON.parse(readFileSync(join(ROOT, 'package.json'), 'utf8')) as { linkedSiblings?: Record<string, Sibling> };
const siblings = Object.entries(manifest.linkedSiblings ?? {});
const fetch = process.argv.includes('--fetch');

function run(command: readonly string[], cwd: string, quiet = false): void {
    // No git LFS objects: the build needs none (pmndrs/denoiser keeps weights and scene assets in LFS).
    // GIT_TERMINAL_PROMPT=0: a private repository without access fails instead of asking for a password.
    const env = { ...process.env, GIT_LFS_SKIP_SMUDGE: '1', GIT_TERMINAL_PROMPT: '0' };
    execFileSync(command[0], command.slice(1), { cwd, stdio: quiet ? 'ignore' : 'inherit', shell: process.platform === 'win32', env });
}

const problems: string[] = [];
const notes: string[] = [];
for (const [name, sibling] of siblings) {
    const clone = resolve(ROOT, sibling.clone);
    const packageDir = join(clone, sibling.path);
    const optional = sibling.optional === true;
    const isBuilt = (): boolean => sibling.built.every((file) => existsSync(join(packageDir, file)));
    if (fetch && !existsSync(clone)) {
        console.log(`linked-siblings: cloning ${sibling.repo} at ${sibling.commit} into ${clone}`);
        try {
            // Long paths: the denoiser repo has research files past Windows' 260-character limit.
            run(['git', '-c', 'core.longpaths=true', 'clone', '--quiet', '--no-checkout', sibling.repo, clone], dirname(clone), optional);
            run(['git', 'config', 'core.longpaths', 'true'], clone);
            run(['git', 'checkout', '--quiet', sibling.commit], clone);
        } catch (error) {
            if (!optional) throw error;
            notes.push(`${name}: couldn't clone ${sibling.repo} (a private repository: no access here?); skipped, the features that need it stay off.`);
            continue;
        }
    }
    if (fetch && existsSync(clone) && !isBuilt()) {
        console.log(`linked-siblings: building ${name} in ${clone}`);
        for (const command of sibling.build) run(command, clone);
    }
    const report = (message: string): void => {
        (optional ? notes : problems).push(message);
    };
    if (!existsSync(packageDir)) {
        report(optional ? `${name} (optional) isn't cloned at ${clone}; the features that need it stay off.` : `${name} is linked from ${packageDir}, which doesn't exist.`);
        continue;
    }
    if (!isBuilt()) {
        report(`${name} at ${packageDir} isn't built (missing ${sibling.built.join(', ')}).`);
        continue;
    }
    let head = '';
    try {
        head = execFileSync('git', ['rev-parse', 'HEAD'], { cwd: clone, encoding: 'utf8', stdio: ['ignore', 'pipe', 'ignore'] }).trim();
    } catch {
        // Not a git clone: fine, it is built.
    }
    if (head && head !== sibling.commit) {
        const drift = `${name}'s clone is at ${head.slice(0, 9)}; package.json pins ${sibling.commit.slice(0, 9)}`;
        if (optional) notes.push(`${drift}: gi refuses to load it until the pinned commit is checked out (or the pin updated).`);
        // A required clone at another commit is fine while developing both (RC4.3); --fetch leaves it as it is.
        else console.warn(`linked-siblings: ${drift} (fine while developing both; check out the pin, or update it)`);
    }
}

for (const note of notes) console.log(`linked-siblings: ${note}`);

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
