import { describe, expect, test } from 'vitest';

import {
    compareSemVer,
    describeReleaseTag,
    distTagFor,
    isLatestRelease,
    lineDistTag,
    normalizeVersion,
    parseSemVer,
} from './release-version.ts';

describe('parseSemVer and compareSemVer', () => {
    test('reads the parts, keeping numbers as strings', () => {
        expect(parseSemVer('v1.20.3-beta.4+build.7')).toEqual({
            major: '1',
            minor: '20',
            patch: '3',
            prerelease: ['beta', '4'],
            build: ['build', '7'],
        });
    });

    test.each(['1.0', '01.0.0', '1.0.0-01', '1.0.0-', 'one.two.three', ''])('refuses %j', (value) => {
        expect(() => parseSemVer(value)).toThrow(/Invalid SemVer/);
    });

    test('orders by SemVer precedence', () => {
        const ordered = ['0.1.0-0', '0.1.0-alpha', '0.1.0-alpha.1', '0.1.0-beta', '0.1.0-beta.2', '0.1.0-beta.11', '0.1.0-rc.1', '0.1.0', '0.1.1', '0.2.0', '0.10.0', '1.0.0'];
        for (let index = 1; index < ordered.length; index++) {
            expect(compareSemVer(ordered[index - 1], ordered[index]), `${ordered[index - 1]} < ${ordered[index]}`).toBe(-1);
            expect(compareSemVer(ordered[index], ordered[index - 1])).toBe(1);
        }
        expect(compareSemVer('1.0.0+a', '1.0.0+b')).toBe(0);
    });

    test('compares numbers beyond what a JS number holds exactly', () => {
        expect(compareSemVer('0.0.9007199254740993', '0.0.9007199254740992')).toBe(1);
    });
});

describe('normalizeVersion (the button\'s version input)', () => {
    test.each([
        ['0.1.0', '0.1.0'],
        ['v0.1.0', '0.1.0'],
        [' 0.2.0-beta.0 ', '0.2.0-beta.0'],
    ])('%j → %s', (input, expected) => {
        expect(normalizeVersion(input)).toBe(expected);
    });

    // No `auto` and no bump words: our commits aren't Conventional Commits, so the version is always given.
    test.each(['auto', 'patch', 'minor', 'major', '', '0.1', '0.1.0+build'])('refuses %j', (input) => {
        expect(() => normalizeVersion(input)).toThrow();
    });
});

describe('describeReleaseTag', () => {
    test('a stable tag and a prerelease tag', () => {
        expect(describeReleaseTag('v0.1.0')).toEqual({ version: '0.1.0', prerelease: false });
        expect(describeReleaseTag('v0.2.0-beta.1')).toEqual({ version: '0.2.0-beta.1', prerelease: true });
    });

    test.each(['0.1.0', 'release-0.1.0', 'v0.1', 'v0.1.0+meta', undefined])('refuses %j', (tag) => {
        expect(() => describeReleaseTag(tag)).toThrow(/Release tags/);
    });
});

describe('isLatestRelease', () => {
    const tags = ['v0.1.0', 'v0.2.0', 'v0.3.0-beta.0', 'not-a-version'];

    test('the newest stable version is latest, ignoring prereleases and other tags', () => {
        expect(isLatestRelease('0.2.0', tags)).toBe(true);
        expect(isLatestRelease('0.2.1', tags)).toBe(true);
    });

    test('an older line and any prerelease are not', () => {
        expect(isLatestRelease('0.1.1', tags)).toBe(false);
        expect(isLatestRelease('0.3.0-beta.1', tags)).toBe(false);
    });
});

describe('distTagFor', () => {
    test('prereleases use their first identifier, or next when it is a number', () => {
        expect(distTagFor('0.2.0-beta.1', '0.1.0')).toBe('beta');
        expect(distTagFor('0.2.0-rc.0', null)).toBe('rc');
        expect(distTagFor('0.2.0-0', '0.1.0')).toBe('next');
    });

    test('a first publish takes latest', () => {
        expect(distTagFor('0.1.0', null)).toBe('latest');
        expect(distTagFor('0.1.0', '')).toBe('latest');
    });

    test('latest only moves forward; an older line gets its line tag', () => {
        expect(distTagFor('0.2.0', '0.1.0')).toBe('latest');
        expect(distTagFor('0.2.0', '0.2.0')).toBe('latest');
        expect(distTagFor('0.1.1', '0.2.0')).toBe('v0.1-latest');
        expect(lineDistTag('1.4.2')).toBe('v1.4-latest');
    });

    test('refuses a latest that is not a version', () => {
        expect(() => distTagFor('0.2.0', 'banana')).toThrow(/not a SemVer/);
    });
});
