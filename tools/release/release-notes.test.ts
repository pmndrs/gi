import { readFileSync } from 'node:fs';

import { describe, expect, test } from 'vitest';

import { changelogSection, renderReleaseNotes } from './release-notes.ts';

const CHANGELOG = `# Changelog

Intro text that belongs to no version.

## 0.3.0-beta.1

Only for the second beta.

## [0.2.0] - 2026-12-01

- Two.
- Lines.

### A sub-heading stays in the section

More.

## v0.1.1

## 0.1.0 (2026-11-20)

The first one.
`;

describe('changelogSection', () => {
    test('takes the text under the version heading, up to the next one, sub-headings included', () => {
        const section = changelogSection(CHANGELOG, '0.2.0');
        expect(section?.heading).toBe('## [0.2.0] - 2026-12-01');
        expect(section?.body).toBe('- Two.\n- Lines.\n\n### A sub-heading stays in the section\n\nMore.');
    });

    test('reads headings with a date, brackets or a leading v', () => {
        expect(changelogSection(CHANGELOG, '0.1.0')?.body).toBe('The first one.');
    });

    test('a version with no section, or an empty one, has none', () => {
        expect(changelogSection(CHANGELOG, '0.4.0')).toBeNull();
        expect(changelogSection(CHANGELOG, '0.1.1')).toBeNull();
    });

    test('a prerelease uses its own section, or else the version it leads up to', () => {
        expect(changelogSection(CHANGELOG, '0.3.0-beta.1')?.body).toBe('Only for the second beta.');
        expect(changelogSection(CHANGELOG, '0.2.0-beta.0')?.version).toBe('0.2.0');
        expect(changelogSection(CHANGELOG, '0.3.0-beta.2')).toBeNull();
    });

    test('does not match a longer version that starts the same', () => {
        expect(changelogSection('## 0.1.0-rc.1\n\nRC.\n', '0.1.0')).toBeNull();
        expect(changelogSection('## 0.1.01\n\nNo.\n', '0.1.0')).toBeNull();
    });

    test('reads Windows line endings', () => {
        expect(changelogSection('## 0.1.0\r\n\r\nText.\r\n', '0.1.0')?.body).toBe('Text.');
    });

    test("the repository's CHANGELOG.md has the first release's notes", () => {
        const changelog = readFileSync(new URL('../../CHANGELOG.md', import.meta.url), 'utf8');
        expect(changelogSection(changelog, '0.1.0')?.body.length).toBeGreaterThan(200);
    });
});

describe('renderReleaseNotes', () => {
    test('the section and a link to the changelog at the tag', () => {
        const section = changelogSection(CHANGELOG, '0.1.0');
        if (!section) throw new Error('fixture lacks 0.1.0');
        expect(renderReleaseNotes(section, 'v0.1.0')).toBe(
            'The first one.\n\n**Full changelog:** https://github.com/pmndrs/ddgi/blob/v0.1.0/CHANGELOG.md\n',
        );
    });
});
