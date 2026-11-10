// Release notes for a version, taken from its section of CHANGELOG.md.
//
// @pmndrs/upscaler builds its notes from Conventional Commits; our commit
// messages aren't written that way, so the notes are written by hand, one
// "## X.Y.Z" section per release, newest first. publish.yml prints the section
// for the version it publishes and uses it as the GitHub Release's text, and
// refuses to cut or publish a version that has no section.
//
//   node tools/release/release-notes.ts --version 0.1.0 [--changelog CHANGELOG.md]
//
// A prerelease (0.1.0-beta.2) uses its own section when there is one, and
// otherwise the section of the version it leads up to (0.1.0), so betas don't
// each need an entry.
import { readFileSync } from 'node:fs';

import { isDirectRun, parseSemVer } from './release-version.ts';

const REPOSITORY_URL = 'https://github.com/pmndrs/ddgi';

/** A section heading: `## 0.1.0`, `## [0.1.0]` or `## v0.1.0`, then anything (a date, "(unreleased)"). */
function headingVersion(line: string): string | null {
    const match = /^##\s+\[?v?([0-9][^\s\]]*)\]?(?:\s|$)/.exec(line);
    return match ? match[1] : null;
}

export interface ChangelogSection {
    /** The heading line as written, e.g. `## 0.1.0 (unreleased)`. */
    heading: string;
    /** The version the section is for (may be the stable version a prerelease leads up to). */
    version: string;
    /** The text under the heading, up to the next `## ` heading, trimmed. */
    body: string;
}

function findSection(changelog: string, version: string): ChangelogSection | null {
    const lines = changelog.replaceAll('\r\n', '\n').split('\n');
    const start = lines.findIndex((line) => headingVersion(line) === version);
    if (start < 0) return null;
    let end = start + 1;
    while (end < lines.length && !lines[end].startsWith('## ')) end++;
    return { heading: lines[start], version, body: lines.slice(start + 1, end).join('\n').trim() };
}

/**
 * The CHANGELOG section for `version`, or null when there is none (or it is
 * empty). A prerelease falls back to its stable version's section.
 */
export function changelogSection(changelog: string, version: string): ChangelogSection | null {
    const parsed = parseSemVer(version);
    const candidates = [version];
    if (parsed.prerelease.length > 0) candidates.push(`${parsed.major}.${parsed.minor}.${parsed.patch}`);
    for (const candidate of candidates) {
        const section = findSection(changelog, candidate);
        if (section && section.body.length > 0) return section;
    }
    return null;
}

/** The GitHub Release text: the section's body and a link to the whole changelog. */
export function renderReleaseNotes(section: ChangelogSection, tag: string, repositoryUrl = REPOSITORY_URL): string {
    return `${section.body}\n\n**Full changelog:** ${repositoryUrl}/blob/${tag}/CHANGELOG.md\n`;
}

function parseArguments(args: string[]): { version: string; changelog: string } {
    let version: string | undefined;
    let changelog = 'CHANGELOG.md';
    for (let index = 0; index < args.length; index++) {
        if (args[index] === '--version' && args[index + 1]) version = args[++index];
        else if (args[index] === '--changelog' && args[index + 1]) changelog = args[++index];
        else throw new Error(`Unknown argument ${args[index]}`);
    }
    if (!version) throw new Error('Usage: node tools/release/release-notes.ts --version X.Y.Z [--changelog CHANGELOG.md]');
    return { version: version.replace(/^v/, ''), changelog };
}

if (isDirectRun(import.meta.url)) {
    try {
        const { version, changelog } = parseArguments(process.argv.slice(2));
        const section = changelogSection(readFileSync(changelog, 'utf8'), version);
        if (!section) {
            throw new Error(
                `${changelog} has no "## ${version}" section with text in it. Write the release notes there first (docs/releasing.md).`,
            );
        }
        // A beta of 0.1.0 is published while 0.1.0 is still unreleased; a stable 0.1.0 should be dated.
        if (/unreleased/i.test(section.heading) && parseSemVer(version).prerelease.length === 0) {
            // Not fatal: the notes are still right; only the heading wasn't dated.
            // Under GitHub Actions the "::warning::" prefix shows it on the run's page.
            const prefix = process.env.GITHUB_ACTIONS === 'true' ? '::warning::' : 'Warning: ';
            process.stderr.write(`${prefix}CHANGELOG.md's heading for ${version} still says unreleased: "${section.heading}"\n`);
        }
        process.stdout.write(renderReleaseNotes(section, `v${version}`));
    } catch (error) {
        process.stderr.write(`${error instanceof Error ? error.message : String(error)}\n`);
        process.exitCode = 1;
    }
}
