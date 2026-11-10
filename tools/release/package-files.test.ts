import { readFileSync } from 'node:fs';

import { describe, expect, test } from 'vitest';

import { PACKAGE_README_END, absoluteLink, githubUrl, headingAnchor, packageReadme } from './package-files.ts';

const REPOSITORY = 'https://github.com/pmndrs/gi';

describe('githubUrl', () => {
    test.each([
        ['git+https://github.com/pmndrs/gi.git', REPOSITORY],
        ['https://github.com/pmndrs/gi', REPOSITORY],
        ['git@github.com:pmndrs/gi.git', REPOSITORY],
    ])('%s', (url, expected) => {
        expect(githubUrl(url)).toBe(expected);
    });

    test('refuses a host other than GitHub', () => {
        expect(() => githubUrl('https://gitlab.com/pmndrs/gi')).toThrow();
    });
});

describe('absoluteLink', () => {
    test('files, folders and images', () => {
        expect(absoluteLink('docs/api/three.md', REPOSITORY, false)).toBe(`${REPOSITORY}/blob/main/docs/api/three.md`);
        expect(absoluteLink('./docs/api/three.md#options', REPOSITORY, false)).toBe(`${REPOSITORY}/blob/main/docs/api/three.md#options`);
        expect(absoluteLink('examples/three/', REPOSITORY, false)).toBe(`${REPOSITORY}/tree/main/examples/three/`);
        expect(absoluteLink('docs/images/a.png', REPOSITORY, true)).toBe('https://raw.githubusercontent.com/pmndrs/gi/main/docs/images/a.png');
    });

    test('leaves absolute URLs and anchors alone', () => {
        for (const target of ['https://threejs.org', 'mailto:a@b.c', '#install', '//cdn.example/x.png']) {
            expect(absoluteLink(target, REPOSITORY, false)).toBe(target);
        }
    });
});

test('headingAnchor follows GitHub', () => {
    expect(headingAnchor('Getting started with three.js')).toBe('getting-started-with-threejs');
    expect(headingAnchor('What CI checks, and what it doesn\'t')).toBe('what-ci-checks-and-what-it-doesnt');
});

describe('packageReadme', () => {
    const readme = [
        '# Title',
        '',
        '[![license](https://img.shields.io/badge/license-MIT-blue)](LICENSE) ![shot](docs/images/shot.png "A shot")',
        '<img src="docs/images/small.png" width="200">',
        'See [three](docs/api/three.md), [install](#install) and [contributing](#working-on-this-repository).',
        '',
        '## Install',
        '',
        '```ts',
        "const link = '[not a link](docs/x.md)';",
        '```',
        '',
        PACKAGE_README_END,
        '',
        '## Working on this repository',
        '',
        'Internal [notes](docs/NOTES.md).',
    ].join('\n');
    const result = packageReadme(readme, REPOSITORY);

    test('stops at the marker', () => {
        expect(result).not.toContain('Working on this repository\n');
        expect(result).not.toContain('NOTES.md');
        expect(result).not.toContain(PACKAGE_README_END);
        expect(result.endsWith('```\n')).toBe(true);
    });

    test('makes links absolute, including a link around a badge, and images load from raw', () => {
        expect(result).toContain(`[![license](https://img.shields.io/badge/license-MIT-blue)](${REPOSITORY}/blob/main/LICENSE)`);
        expect(result).toContain('![shot](https://raw.githubusercontent.com/pmndrs/gi/main/docs/images/shot.png "A shot")');
        expect(result).toContain('<img src="https://raw.githubusercontent.com/pmndrs/gi/main/docs/images/small.png" width="200">');
        expect(result).toContain(`[three](${REPOSITORY}/blob/main/docs/api/three.md)`);
    });

    test('keeps anchors to sections it has, and sends the others to the repository', () => {
        expect(result).toContain('[install](#install)');
        expect(result).toContain(`[contributing](${REPOSITORY}#working-on-this-repository)`);
    });

    test('leaves code blocks alone', () => {
        expect(result).toContain("const link = '[not a link](docs/x.md)';");
    });

    test('needs the marker', () => {
        expect(() => packageReadme('# No marker\n', REPOSITORY)).toThrow(/marker/);
    });

    test("the repository's README makes a package README with no relative links left", () => {
        const root = readFileSync(new URL('../../README.md', import.meta.url), 'utf8');
        const packaged = packageReadme(root, REPOSITORY);
        expect(packaged).toContain('## Getting started with three.js');
        expect(packaged).toContain('## Install');
        expect(packaged).not.toContain('## Working on this repository');
        let insideFence = false;
        for (const line of packaged.split('\n')) {
            if (/^\s*```/.test(line)) insideFence = !insideFence;
            if (insideFence) continue;
            for (const match of line.matchAll(/\]\(([^)\s]+)/g)) {
                expect(match[1], line).toMatch(/^(https:\/\/|#)/);
            }
        }
    });
});
