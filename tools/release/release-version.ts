// Release-version policy for @pmndrs/gi, adapted from @pmndrs/upscaler's
// scripts/release-version.mjs. One difference: our commit messages aren't
// Conventional Commits, so there is no `auto` bump. The version is always
// given, by the tag (`v0.1.0`) or by the publish workflow's `version` input.
//
// publish.yml runs this file with plain Node (type stripping, no build step):
//
//   node tools/release/release-version.ts --check 0.1.0
//       # the version, normalised (a leading "v" dropped), or an error
//   node tools/release/release-version.ts --describe v0.1.0-beta.1
//       # "<version> <prerelease> <latest>": "0.1.0-beta.1 true false"
//   node tools/release/release-version.ts --dist-tag 0.2.1 0.3.0
//       # the npm dist-tag, given npm's current latest ("" if none): "v0.2-latest"
//   node tools/release/release-version.ts --compare 0.2.0 0.1.0
//       # SemVer order of A against B: -1, 0 or 1
import { execFileSync } from 'node:child_process';
import { realpathSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

export interface SemVer {
    // Kept as strings so a huge number can't lose precision in a JS number.
    major: string;
    minor: string;
    patch: string;
    prerelease: string[];
    build: string[];
}

// SemVer 2.0.0's grammar, with an optional leading "v" for tags.
const SEMVER_PATTERN =
    /^v?(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:-([0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?(?:\+([0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?$/;

export function parseSemVer(value: unknown): SemVer {
    const match = typeof value === 'string' ? SEMVER_PATTERN.exec(value) : null;
    if (!match) throw new Error(`Invalid SemVer: ${String(value)}`);
    const prerelease = match[4]?.split('.') ?? [];
    // SemVer forbids leading zeros in numeric prerelease identifiers (1.0.0-01).
    if (prerelease.some((identifier) => /^\d+$/.test(identifier) && /^0\d/.test(identifier))) {
        throw new Error(`Invalid SemVer: ${String(value)}`);
    }
    return { major: match[1], minor: match[2], patch: match[3], prerelease, build: match[5]?.split('.') ?? [] };
}

// Compares two non-negative integers written as digit strings, without converting them.
function compareNumericIdentifier(left: string, right: string): -1 | 0 | 1 {
    if (left.length !== right.length) return left.length < right.length ? -1 : 1;
    if (left === right) return 0;
    return left < right ? -1 : 1;
}

/** SemVer precedence: -1 if `left` comes before `right`, 1 if after, 0 if equal (build metadata ignored). */
export function compareSemVer(left: string | SemVer, right: string | SemVer): -1 | 0 | 1 {
    const a = typeof left === 'string' ? parseSemVer(left) : left;
    const b = typeof right === 'string' ? parseSemVer(right) : right;
    for (const field of ['major', 'minor', 'patch'] as const) {
        const result = compareNumericIdentifier(a[field], b[field]);
        if (result !== 0) return result;
    }
    // A version without a prerelease ranks above any prerelease of it (1.0.0 > 1.0.0-rc.1).
    if (a.prerelease.length === 0 && b.prerelease.length === 0) return 0;
    if (a.prerelease.length === 0) return 1;
    if (b.prerelease.length === 0) return -1;
    const count = Math.max(a.prerelease.length, b.prerelease.length);
    for (let index = 0; index < count; index++) {
        const aIdentifier = a.prerelease[index];
        const bIdentifier = b.prerelease[index];
        if (aIdentifier === undefined) return -1;
        if (bIdentifier === undefined) return 1;
        if (aIdentifier === bIdentifier) continue;
        const aNumeric = /^\d+$/.test(aIdentifier);
        const bNumeric = /^\d+$/.test(bIdentifier);
        if (aNumeric && bNumeric) return compareNumericIdentifier(aIdentifier, bIdentifier);
        if (aNumeric) return -1;
        if (bNumeric) return 1;
        return aIdentifier < bIdentifier ? -1 : 1;
    }
    return 0;
}

/**
 * The version asked for by the publish workflow's `version` input: an exact
 * SemVer, with or without a leading "v". Words like `patch` or `auto` are
 * refused on purpose: we always say which version we mean.
 */
export function normalizeVersion(request: string): string {
    const version = request.trim().replace(/^v/, '');
    try {
        parseSemVer(version);
    } catch {
        throw new Error(`Expected a version like 0.1.0 or 0.2.0-beta.0, received: ${JSON.stringify(request)}`);
    }
    if (parseSemVer(version).build.length > 0) {
        throw new Error(`Build metadata (+...) isn't allowed in a release version: ${version}`);
    }
    return version;
}

/** Validates a release tag (`v` + SemVer) and says whether it is a prerelease. */
export function describeReleaseTag(tag: unknown): { version: string; prerelease: boolean } {
    if (typeof tag !== 'string' || !tag.startsWith('v')) {
        throw new Error(`Release tags must be "v" + SemVer (e.g. v0.1.0), received: ${String(tag)}`);
    }
    const version = tag.slice(1);
    let parsed: SemVer;
    try {
        parsed = parseSemVer(version);
    } catch {
        throw new Error(`Release tags must be "v" + SemVer (e.g. v0.1.0), received: ${tag}`);
    }
    if (parsed.build.length > 0) throw new Error(`Release tags can't carry build metadata (+...): ${tag}`);
    return { version, prerelease: parsed.prerelease.length > 0 };
}

/**
 * Whether a release should become the repository's "latest" GitHub Release:
 * only a stable version at or above every stable `v*` tag, so publishing a fix
 * for an older line never takes "latest" from a newer one.
 */
export function isLatestRelease(version: string, tags: string[]): boolean {
    if (parseSemVer(version).prerelease.length > 0) return false;
    for (const tag of tags) {
        if (!tag.startsWith('v')) continue;
        let other: SemVer;
        try {
            other = parseSemVer(tag.slice(1));
        } catch {
            continue;
        }
        if (other.prerelease.length === 0 && compareSemVer(other, version) > 0) return false;
    }
    return true;
}

/** The dist-tag that keeps installing an older stable line: 0.2.1 → `v0.2-latest`. */
export function lineDistTag(version: string): string {
    const { major, minor } = parseSemVer(version);
    return `v${major}.${minor}-latest`;
}

/**
 * The npm dist-tag a version publishes under. A prerelease uses its first
 * identifier (0.2.0-beta.1 → `beta`; a numeric one, 0.2.0-0, → `next`), so it
 * never moves `latest`. A stable version takes `latest` only when it is at or
 * above npm's current `latest` (or npm has none yet); an older one gets its
 * line tag, so `latest` never moves backwards.
 */
export function distTagFor(version: string, npmLatest: string | null): string {
    const { prerelease } = parseSemVer(version);
    if (prerelease.length > 0) return /^\d+$/.test(prerelease[0]) ? 'next' : prerelease[0];
    if (!npmLatest) return 'latest';
    let current: SemVer;
    try {
        current = parseSemVer(npmLatest);
    } catch {
        throw new Error(`npm's latest dist-tag is not a SemVer version: ${JSON.stringify(npmLatest)}`);
    }
    return compareSemVer(version, current) >= 0 ? 'latest' : lineDistTag(version);
}

/**
 * Whether the module at `moduleUrl` is the script Node was started with.
 * Compares real paths: macOS's /var is a link to /private/var, and Node resolves
 * links in `import.meta.url` but not in `process.argv[1]`.
 */
export function isDirectRun(moduleUrl: string): boolean {
    if (!process.argv[1]) return false;
    try {
        return realpathSync(process.argv[1]) === realpathSync(fileURLToPath(moduleUrl));
    } catch {
        return false;
    }
}

const USAGE =
    'Usage: node tools/release/release-version.ts --check X.Y.Z | --describe vX.Y.Z | --dist-tag X.Y.Z [<npm latest>] | --compare A B';

function main(args: string[]): void {
    if (args[0] === '--check' && args.length === 2) {
        process.stdout.write(`${normalizeVersion(args[1])}\n`);
        return;
    }
    if (args[0] === '--describe' && args.length === 2) {
        const { version, prerelease } = describeReleaseTag(args[1]);
        const tags = execFileSync('git', ['tag', '--list', 'v*'], { encoding: 'utf8' }).split(/\r?\n/).filter(Boolean);
        process.stdout.write(`${version} ${prerelease} ${isLatestRelease(version, tags)}\n`);
        return;
    }
    if (args[0] === '--dist-tag' && (args.length === 2 || args.length === 3)) {
        process.stdout.write(`${distTagFor(args[1], args[2] || null)}\n`);
        return;
    }
    if (args[0] === '--compare' && args.length === 3) {
        process.stdout.write(`${compareSemVer(args[1], args[2])}\n`);
        return;
    }
    throw new Error(USAGE);
}

if (isDirectRun(import.meta.url)) {
    try {
        main(process.argv.slice(2));
    } catch (error) {
        process.stderr.write(`${error instanceof Error ? error.message : String(error)}\n`);
        process.exitCode = 1;
    }
}
