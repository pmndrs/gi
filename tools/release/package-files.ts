// Copies the files the npm package needs from the repository into
// packages/gi, as the last part of its build (`pnpm build`):
//
//   README.md    the root README's user-facing part: everything above the
//                PACKAGE_README_END marker, with links made absolute, since
//                npmjs.com can't resolve a link relative to the repository.
//   LICENSE      the repository's licence, unchanged.
//   licenses/    the full licence texts of the bundled code, which
//                packages/gi/THIRD_PARTY.md points to: packages/bvh's
//                (tinybvh, the Emscripten runtime, HIPRT) and each
//                technique's own `licenses/` folder (TECHNIQUE_LICENSES),
//                and the platform's (PLATFORM_LICENSES).
//
// packages/gi/THIRD_PARTY.md itself is written by hand and committed: it
// lists only what ships in the package, unlike the root THIRD_PARTY.md, which
// also lists test scenes and tools. The copies are git-ignored, so the root
// files stay the one place to edit.
import { copyFileSync, mkdirSync, readFileSync, readdirSync, rmSync, writeFileSync } from 'node:fs';
import { join, resolve } from 'node:path';

import { isDirectRun } from './release-version.ts';

const REPOSITORY_ROOT = resolve(import.meta.dirname, '../..');
const PACKAGE_DIRECTORY = join(REPOSITORY_ROOT, 'packages/gi');
const BUNDLED_LICENSES = join(REPOSITORY_ROOT, 'packages/bvh/licenses');
/**
 * Licence texts kept beside the techniques that bundle the code: Split RC's (the raytracer, pbrt-v4; inside
 * packages/core until it moves to packages/radiance-cascades) and the radiance cache's (AMD Capsaicin).
 */
const TECHNIQUE_LICENSES = ['packages/core/src/techniques/radiance-cascades/licenses', 'packages/cache/src/licenses'].map((folder) => join(REPOSITORY_ROOT, folder));
/** Licence texts for code the platform bundles beyond Capsaicin's: the SSGI pass's rtoy-samples and assimp parts. */
const PLATFORM_LICENSES = join(REPOSITORY_ROOT, 'packages/core/src/platform/licenses');

/** The line in the root README where the npm package's README stops. */
export const PACKAGE_README_END = '<!-- The npm package README ends here. Below: working on this repository. -->';

/** The branch links point at. */
const DEFAULT_BRANCH = 'main';

/** `git+https://github.com/pmndrs/gi.git` → `https://github.com/pmndrs/gi`. */
export function githubUrl(repositoryUrl: string): string {
    const match = /github\.com[/:]([^/]+)\/([^/#]+?)(?:\.git)?$/.exec(repositoryUrl);
    if (!match) throw new Error(`Not a GitHub repository URL: ${repositoryUrl}`);
    return `https://github.com/${match[1]}/${match[2]}`;
}

function isRelative(target: string): boolean {
    // Absolute URLs (https:, mailto:, data:) and in-page anchors stay as they are.
    return !/^[a-z][a-z0-9+.-]*:/i.test(target) && !target.startsWith('#') && !target.startsWith('//');
}

/**
 * A repository-relative link made absolute: pages and folders on github.com,
 * images from raw.githubusercontent.com so npmjs.com can show them.
 */
export function absoluteLink(target: string, repository: string, image: boolean): string {
    if (!isRelative(target)) return target;
    const path = target.replace(/^\.?\//, '');
    if (image) {
        const [owner, name] = repository.replace('https://github.com/', '').split('/');
        return `https://raw.githubusercontent.com/${owner}/${name}/${DEFAULT_BRANCH}/${path}`;
    }
    const kind = path.split('#')[0].endsWith('/') || path === '' ? 'tree' : 'blob';
    return `${repository}/${kind}/${DEFAULT_BRANCH}/${path}`;
}

/** GitHub's anchor for a heading: lower case, punctuation dropped, spaces to hyphens. */
export function headingAnchor(heading: string): string {
    return heading
        .trim()
        .toLowerCase()
        .replace(/[^\p{L}\p{N}\s_-]/gu, '')
        .replace(/\s/g, '-');
}

/**
 * The package README: the root README above the marker, with every relative
 * link and image made absolute, and in-page anchors to sections below the
 * marker sent to the repository's README. Fenced code blocks are left alone.
 */
export function packageReadme(rootReadme: string, repository: string): string {
    const text = rootReadme.replaceAll('\r\n', '\n');
    const end = text.indexOf(PACKAGE_README_END);
    if (end < 0) throw new Error(`README.md lacks the marker line that ends the package README:\n${PACKAGE_README_END}`);
    const kept = text.slice(0, end).trimEnd().split('\n');

    const fenceLine = (line: string) => /^\s*(```|~~~)/.test(line);
    let insideFence = false;
    const anchors = new Set<string>();
    for (const line of kept) {
        if (fenceLine(line)) insideFence = !insideFence;
        const heading = /^#{1,6}\s+(.+)$/.exec(line);
        if (!insideFence && heading) anchors.add(headingAnchor(heading[1]));
    }
    const linkTarget = (target: string, image: boolean) =>
        target.startsWith('#') && !anchors.has(target.slice(1)) ? `${repository}${target}` : absoluteLink(target, repository, image);

    insideFence = false;
    const lines = kept.map((line) => {
        if (fenceLine(line)) insideFence = !insideFence;
        if (insideFence) return line;
        return line
            // Images first, ![alt](target "title"): they load from raw.githubusercontent.com.
            .replace(/!\[([^\]]*)\]\(([^)\s]+)((?:\s+"[^"]*")?)\)/g, (_, alt: string, target: string, title: string) =>
                `![${alt}](${linkTarget(target, true)}${title})`,
            )
            // Then every link's target, ](target): this also reaches a link wrapped around an image, [![badge](...)](target).
            .replace(/\]\(([^)\s]+)((?:\s+"[^"]*")?)\)/g, (_, target: string, title: string) => `](${linkTarget(target, false)}${title})`)
            // HTML attributes, for <img src> and <a href> (a sized image needs HTML).
            .replace(/\b(src|href)="([^"]+)"/g, (_, attribute: string, target: string) =>
                `${attribute}="${linkTarget(target, attribute === 'src')}"`,
            );
    });
    return `${lines.join('\n')}\n`;
}

function writePackageFiles(): void {
    const manifest = JSON.parse(readFileSync(join(PACKAGE_DIRECTORY, 'package.json'), 'utf8')) as {
        repository: { url: string };
    };
    const repository = githubUrl(manifest.repository.url);

    const readme = packageReadme(readFileSync(join(REPOSITORY_ROOT, 'README.md'), 'utf8'), repository);
    writeFileSync(join(PACKAGE_DIRECTORY, 'README.md'), readme);
    copyFileSync(join(REPOSITORY_ROOT, 'LICENSE'), join(PACKAGE_DIRECTORY, 'LICENSE'));

    // Start from an empty folder, so a licence removed from packages/bvh doesn't linger.
    const licenses = join(PACKAGE_DIRECTORY, 'licenses');
    rmSync(licenses, { recursive: true, force: true });
    mkdirSync(licenses);
    const sources = [BUNDLED_LICENSES, PLATFORM_LICENSES, ...TECHNIQUE_LICENSES];
    const texts = sources.flatMap(source => readdirSync(source).filter(name => name.endsWith('.txt')).map(name => {
        copyFileSync(join(source, name), join(licenses, name));
        return name;
    }));

    console.log(`Package files: README.md (${readme.length} characters), LICENSE, licenses/ (${texts.join(', ')}).`);
}

if (isDirectRun(import.meta.url)) writePackageFiles();
