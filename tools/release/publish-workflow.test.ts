// Runs publish.yml's shell steps end to end against throwaway git repositories:
// a bare "origin" and a fresh clone per run, with a fake `npm` (the real one for
// `npm version`; a pretend registry and publish), a fake `pnpm` and a fake `gh`
// on PATH. Each step's `if:` and `env:` are read from the YAML and evaluated as
// Actions would, so these tests break when the workflow drifts. Nothing touches
// GitHub or npm, and no GPU is needed. Adapted from @pmndrs/upscaler's
// scripts/release-workflow.test.mjs, without its Conventional Commits cases.
import { spawnSync } from 'node:child_process';
import { copyFileSync, existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { afterEach, describe, expect, test } from 'vitest';

import { bash, shellEnvironment, shellPath, shellQuote, which } from './test-shell.ts';
import { npmInvocation } from './verify-packed.ts';

const SLOW = 60_000;

// Every run is synchronous child processes, and vitest only yields between
// tests on a macrotask; without one its worker can't answer its own RPCs in a
// long file and reports a timeout.
afterEach(() => new Promise((resolve) => setImmediate(resolve)));

const workflow = readFileSync(new URL('../../.github/workflows/publish.yml', import.meta.url), 'utf8').replaceAll('\r\n', '\n');
const ci = readFileSync(new URL('../../.github/workflows/ci.yml', import.meta.url), 'utf8').replaceAll('\r\n', '\n');

//* Reading the workflow ===

interface Step {
    name?: string;
    id?: string;
    if?: string;
    uses?: string;
    run?: string;
    env: Record<string, string>;
}

/** The single job's steps. Only the YAML shapes publish.yml uses are understood (2-space indents). */
function parseSteps(text: string): Step[] {
    const body = text.slice(text.indexOf('\n    steps:\n') + '\n    steps:\n'.length);
    return body.split(/\n(?= {6}- )/).map((chunk) => {
        const lines = chunk.split('\n');
        const step: Step = { env: {} };
        for (let index = 0; index < lines.length; index++) {
            const line = lines[index].replace(/^ {6}- /, '        ');
            let match: RegExpMatchArray | null;
            if ((match = line.match(/^ {8}name: (.+)$/))) step.name = match[1];
            else if ((match = line.match(/^ {8}id: (.+)$/))) step.id = match[1];
            else if ((match = line.match(/^ {8}if: (.+)$/))) step.if = match[1];
            else if ((match = line.match(/^ {8}uses: (.+)$/))) step.uses = match[1];
            else if ((match = line.match(/^ {8}run: (?!\|)(.+)$/))) step.run = match[1];
            else if (/^ {8}run: \|$/.test(line)) {
                const script: string[] = [];
                while (index + 1 < lines.length && (lines[index + 1] === '' || /^ {10}/.test(lines[index + 1]))) {
                    script.push(lines[++index].slice(10));
                }
                step.run = script.join('\n');
            } else if (/^ {8}env:$/.test(line)) {
                while (index + 1 < lines.length && /^ {10}\S/.test(lines[index + 1])) {
                    const envMatch = lines[++index].match(/^ {10}([A-Z_]+): (.+)$/);
                    if (!envMatch) throw new Error(`Unreadable env line: ${lines[index]}`);
                    step.env[envMatch[1]] = envMatch[2];
                }
            }
        }
        return step;
    });
}

const steps = parseSteps(workflow);

function step(name: string): Step {
    const found = steps.find((candidate) => candidate.name === name);
    if (!found) throw new Error(`Missing workflow step: ${name}`);
    return found;
}

interface Context {
    event: string;
    ref: string;
    inputs: { version?: string };
    outputs: Record<string, Record<string, string>>;
}

/** The handful of `${{ }}` expressions the workflow uses. */
function evaluate(expression: string, context: Context): string {
    const trimmed = expression.trim();
    const match = trimmed.match(/^steps\.([\w-]+)\.outputs\.([\w-]+)$/);
    if (match) return context.outputs[match[1]]?.[match[2]] ?? '';
    const values: Record<string, string> = {
        'github.event_name': context.event,
        'github.ref': context.ref,
        'inputs.version': context.inputs.version ?? '',
        'github.token': 'test-token',
        'github.repository': 'pmndrs/gi',
    };
    if (trimmed in values) return values[trimmed];
    throw new Error(`Unsupported workflow expression: ${trimmed}`);
}

function condition(expression: string | undefined, context: Context): boolean {
    if (!expression) return true;
    const match = expression.match(/^steps\.([\w-]+)\.outputs\.([\w-]+) (==|!=) '([^']*)'$/);
    if (!match) throw new Error(`Unsupported workflow condition: ${expression}`);
    const equal = (context.outputs[match[1]]?.[match[2]] ?? '') === match[4];
    return match[3] === '==' ? equal : !equal;
}

//* Fixture repositories ===

function git(cwd: string, args: string[]): string {
    const result = spawnSync('git', args, { cwd, encoding: 'utf8' });
    if (result.status !== 0) throw new Error(`git ${args.join(' ')} failed:\n${result.stdout}${result.stderr}`);
    return result.stdout.trim();
}

function configureUser(directory: string): void {
    git(directory, ['config', 'user.name', 'Workflow Test']);
    git(directory, ['config', 'user.email', 'workflow@example.test']);
}

function writePackage(directory: string, version: string): void {
    mkdirSync(join(directory, 'packages/gi'), { recursive: true });
    writeFileSync(join(directory, 'packages/gi/package.json'), `${JSON.stringify({ name: '@pmndrs/gi', version }, null, 4)}\n`);
}

function commit(directory: string, message: string): string {
    git(directory, ['add', '-A']);
    git(directory, ['commit', '-q', '--allow-empty', '-m', message]);
    return git(directory, ['rev-parse', 'HEAD']);
}

/** CHANGELOG.md with a section for each version given. */
function writeChangelog(directory: string, versions: string[]): void {
    const sections = versions.map((version) => `## ${version} (2026-12-01)\n\nNotes for ${version}.\n`);
    writeFileSync(join(directory, 'CHANGELOG.md'), `# Changelog\n\n${sections.join('\n')}`);
}

const FAKE_NPM = `#!/usr/bin/env bash
set -euo pipefail
case "$1" in
    version) exec "$REAL_NPM" "$@" ;;
esac
printf '%s\\n' "$*" >> "$NPM_LOG"
case "$1" in
    install|ci) exit 0 ;;
    publish)
        [[ "$NPM_PUBLISH_MODE" == "fail" ]] && { printf 'npm error code E403\\n' >&2; exit 1; }
        printf 'published %s@%s from %s\\n' "$(node -p "require('./package.json').name")" "$(node -p "require('./package.json').version")" "$(basename "$PWD")" >> "$NPM_LOG"
        exit 0
        ;;
    view)
        if [[ "\${3:-}" == "dist-tags.latest" ]]; then
            case "$NPM_LATEST_MODE" in
                none) printf 'npm error code E404\\n' >&2; exit 1 ;;
                auth) printf 'npm error code E401\\n' >&2; exit 1 ;;
                network) printf 'npm error code ECONNRESET\\n' >&2; exit 1 ;;
                *) printf '%s\\n' "$NPM_LATEST_MODE" ;;
            esac
            exit 0
        fi
        version="\${2##*@}"
        case "$NPM_MODE" in
            published) printf '%s\\n' "$version" ;;
            absent) printf 'npm error code E404\\n' >&2; exit 1 ;;
            auth) printf 'npm error code E401\\n' >&2; exit 1 ;;
            *) printf 'unexpected NPM_MODE: %s\\n' "$NPM_MODE" >&2; exit 2 ;;
        esac
        ;;
    *) printf 'unexpected npm command: %s\\n' "$*" >&2; exit 2 ;;
esac
`;

const FAKE_PNPM = `#!/usr/bin/env bash
set -euo pipefail
printf 'pnpm %s\\n' "$*" >> "$NPM_LOG"
case "$1" in
    install) exit 0 ;;
    *) printf 'unexpected pnpm command: %s\\n' "$*" >&2; exit 2 ;;
esac
`;

const FAKE_GH = `#!/usr/bin/env bash
set -euo pipefail
if [[ "$1" == "release" && "$2" == "create" ]]; then
    printf '%s\\n' "$*" >> "$GH_LOG"
    exit 0
fi
[[ "$1 $2 $3" == "api --paginate repos/pmndrs/gi/releases" ]] || { printf 'unexpected gh: %s\\n' "$*" >&2; exit 2; }
case "$GH_MODE" in
    missing) printf 'v0.0.1\\tfalse\\n' ;;
    published) printf 'v0.0.1\\tfalse\\n%s\\tfalse\\n' "$TAG" ;;
    draft) printf '%s\\ttrue\\n' "$TAG" ;;
    forbidden) printf 'HTTP 403: Resource not accessible by integration\\n' >&2; exit 1 ;;
    *) printf 'unexpected GH_MODE: %s\\n' "$GH_MODE" >&2; exit 2 ;;
esac
`;

// On the first push while RACE_FILE exists, land a concurrent merge on origin
// first, so that push is turned away as not a fast-forward.
const GIT_WRAPPER = `#!/usr/bin/env bash
set -euo pipefail
if [[ "$1" == "push" && -n "\${RACE_FILE:-}" && -f "$RACE_FILE" ]]; then
    rm -f "$RACE_FILE"
    "$REAL_GIT" -C "$RACE_CLONE" pull -q --ff-only origin main
    "$REAL_GIT" -C "$RACE_CLONE" commit -q --allow-empty -m "A concurrent merge"
    "$REAL_GIT" -C "$RACE_CLONE" push -q origin HEAD:refs/heads/main
fi
exec "$REAL_GIT" "$@"
`;

interface Fixture {
    root: string;
    origin: string;
    seed: string;
    bin: string;
    realNpm: string;
    runs: number;
}

/**
 * origin's main: "Release 0.1.0" (packages/gi at 0.1.0, tagged v0.1.0) →
 * "Release tooling" → "Notes for 0.2.0 and 0.3.0" (CHANGELOG has 0.1.0, 0.2.0
 * and 0.3.0) ← main. packages/gi stays at 0.1.0 throughout.
 */
function createFixture({ tag, offMain }: { tag?: string; offMain?: string } = {}): Fixture {
    const root = mkdtempSync(join(tmpdir(), 'ddgi-publish-workflow-'));
    const origin = join(root, 'origin.git');
    const seed = join(root, 'seed');
    const bin = join(root, 'bin');
    for (const directory of [seed, bin]) mkdirSync(directory);
    git(root, ['init', '-q', '--bare', '-b', 'main', origin]);

    git(seed, ['init', '-q', '-b', 'main']);
    configureUser(seed);
    writeFileSync(join(seed, 'package.json'), `${JSON.stringify({ name: 'workspace-root', private: true, version: '0.0.0' }, null, 4)}\n`);
    writePackage(seed, '0.1.0');
    writeChangelog(seed, ['0.1.0']);
    commit(seed, 'Release 0.1.0');
    git(seed, ['tag', '-a', 'v0.1.0', '-m', 'v0.1.0']);

    mkdirSync(join(seed, 'tools/release'), { recursive: true });
    for (const script of ['release-notes.ts', 'release-version.ts']) {
        copyFileSync(new URL(`./${script}`, import.meta.url), join(seed, 'tools/release', script));
    }
    commit(seed, 'Release tooling');
    writeChangelog(seed, ['0.3.0', '0.2.0', '0.1.0']);
    commit(seed, 'Notes for 0.2.0 and 0.3.0');
    if (tag) git(seed, ['tag', '-a', tag, '-m', tag]);

    if (offMain) {
        git(seed, ['checkout', '-q', '-b', 'side']);
        commit(seed, 'Unmerged work');
        git(seed, ['tag', '-a', offMain, '-m', offMain]);
        git(seed, ['checkout', '-q', 'main']);
    }

    git(seed, ['remote', 'add', 'origin', origin]);
    git(seed, ['push', '-q', 'origin', 'main', '--tags']);

    writeFileSync(join(bin, 'npm'), FAKE_NPM, { mode: 0o755 });
    writeFileSync(join(bin, 'pnpm'), FAKE_PNPM, { mode: 0o755 });
    writeFileSync(join(bin, 'gh'), FAKE_GH, { mode: 0o755 });
    writeFileSync(join(bin, 'git'), GIT_WRAPPER, { mode: 0o755 });
    let realNpm = which('npm');
    if (process.platform === 'win32') {
        // Git Bash can't run npm's .cmd shim the way the fake execs it; call npm's CLI with Node instead.
        realNpm = join(bin, 'real-npm');
        const [command, args] = npmInvocation([]);
        writeFileSync(realNpm, `#!/usr/bin/env bash\nexec ${[command, ...args].map((value) => shellQuote(shellPath(value))).join(' ')} "$@"\n`);
    }
    return { root, origin, seed, bin, realNpm, runs: 0 };
}

function withFixture<T>(options: Parameters<typeof createFixture>[0], callback: (fixture: Fixture) => T): T {
    const fixture = createFixture(options);
    try {
        return callback(fixture);
    } finally {
        rmSync(fixture.root, { recursive: true, force: true, maxRetries: 5, retryDelay: 200 });
    }
}

/** Lands a commit on origin's main from the seed checkout. */
function pushToMain(fixture: Fixture, message: string, version?: string): void {
    git(fixture.seed, ['pull', '-q', '--ff-only', 'origin', 'main']);
    if (version) writePackage(fixture.seed, version);
    commit(fixture.seed, message);
    git(fixture.seed, ['push', '-q', 'origin', 'main']);
}

const originGit = (fixture: Fixture, args: string[]) => git(fixture.root, ['--git-dir', fixture.origin, ...args]);
const mainSubjects = (fixture: Fixture) => originGit(fixture, ['log', '--format=%s', 'main']).split('\n');
const versionAt = (fixture: Fixture, ref: string) =>
    (JSON.parse(originGit(fixture, ['show', `${ref}:packages/gi/package.json`])) as { version: string }).version;

function readLog(path: string): string[] {
    return existsSync(path) ? readFileSync(path, 'utf8').split('\n').filter(Boolean) : [];
}

interface Scenario {
    event?: 'push' | 'workflow_dispatch';
    tag?: string; // the pushed tag (push events)
    ref?: string; // overrides github.ref
    inputs?: { version?: string };
    npm?: 'absent' | 'published' | 'auth'; // `npm view <name>@<version>`
    npmLatest?: string; // npm's latest version, or 'none' | 'auth' | 'network'
    gh?: 'missing' | 'published' | 'draft' | 'forbidden';
    race?: boolean; // land a concurrent merge before the first push
}

/** Runs the job's `run:` steps in order, as Actions would after actions/checkout. */
function runJob(fixture: Fixture, scenario: Scenario = {}) {
    const { event = 'push', tag, ref, inputs = {}, npm = 'absent', npmLatest = '0.1.0', gh = 'missing', race = false } = scenario;
    const run = ++fixture.runs;
    const work = join(fixture.root, `work-${run}`);
    const temp = join(fixture.root, `runner-temp-${run}`);
    mkdirSync(temp);
    const context: Context = {
        event,
        ref: ref ?? (event === 'push' ? `refs/tags/${tag}` : 'refs/heads/main'),
        inputs,
        outputs: {},
    };

    // actions/checkout: full history and tags; the pushed tag, or the branch the button ran from.
    git(fixture.root, ['clone', '-q', fixture.origin, work]);
    if (context.ref.startsWith('refs/tags/')) git(work, ['checkout', '-q', '--detach', context.ref]);
    else git(work, ['checkout', '-q', '-B', 'main', 'origin/main']);

    const raceFile = join(fixture.root, `race-${run}`);
    const raceClone = join(fixture.root, `race-clone-${run}`);
    if (race) {
        writeFileSync(raceFile, '');
        git(fixture.root, ['clone', '-q', fixture.origin, raceClone]);
        configureUser(raceClone);
    }

    const npmLog = join(fixture.root, `npm-${run}.log`);
    const ghLog = join(fixture.root, `gh-${run}.log`);
    let failed: string | null = null;
    let output = '';
    for (const [index, current] of steps.entries()) {
        if (!current.run || !condition(current.if, context)) continue;
        // Expressions belong in env:, never inline in a script (script injection).
        expect(current.run, `${current.name ?? current.run} must not inline expressions`).not.toMatch(/\$\{\{/);
        const outputPath = join(temp, `output-${index}`);
        writeFileSync(outputPath, '');
        const env = Object.fromEntries(Object.entries(current.env).map(([key, value]) => [key, value.replace(/\$\{\{(.+?)\}\}/g, (_, expression: string) => evaluate(expression, context))]));
        const result = spawnSync(bash as string, ['-e', '-o', 'pipefail', '-c', current.run], {
            cwd: work,
            encoding: 'utf8',
            env: shellEnvironment(
                {
                    ...env,
                    REAL_GIT: which('git'),
                    REAL_NPM: shellPath(fixture.realNpm),
                    RUNNER_TEMP: shellPath(temp),
                    GITHUB_OUTPUT: shellPath(outputPath),
                    NPM_LOG: shellPath(npmLog),
                    GH_LOG: shellPath(ghLog),
                    NPM_MODE: npm,
                    NPM_LATEST_MODE: npmLatest,
                    GH_MODE: gh,
                    NPM_PUBLISH_MODE: 'success',
                    RACE_FILE: race ? shellPath(raceFile) : '',
                    RACE_CLONE: shellPath(raceClone),
                },
                [fixture.bin],
            ),
        });
        output += result.stdout + result.stderr;
        if (current.id) {
            context.outputs[current.id] = Object.fromEntries(
                readFileSync(outputPath, 'utf8')
                    .split('\n')
                    .filter(Boolean)
                    .map((line) => [line.slice(0, line.indexOf('=')), line.slice(line.indexOf('=') + 1)]),
            );
        }
        if (result.status !== 0) {
            failed = current.name ?? current.run;
            break;
        }
    }

    const npmCalls = readLog(npmLog);
    const notesPath = join(temp, 'release-notes.md');
    return {
        failed,
        output,
        outputs: context.outputs,
        publishes: npmCalls.filter((line) => line.startsWith('publish ')),
        published: npmCalls.filter((line) => line.startsWith('published ')).map((line) => line.slice('published '.length)),
        installs: npmCalls.filter((line) => line.startsWith('pnpm install')),
        releases: readLog(ghLog),
        notes: existsSync(notesPath) ? readFileSync(notesPath, 'utf8') : null,
    };
}

// Windows runners without Git Bash skip the runs; the structure checks still run.
const describeRuns = bash ? describe : describe.skip;

//* Triggers and structure ===

describe('triggers and structure', () => {
    const on = workflow.slice(workflow.indexOf('\non:\n'), workflow.indexOf('\npermissions:'));

    test('publishes on v* tags and the button, never on branch pushes', () => {
        expect(on).toMatch(/\n {2}push:\n {4}tags: \['v\*'\]\n/);
        expect(on).toMatch(/workflow_dispatch:\n {4}inputs:\n {6}version:\n/);
        expect(on).not.toMatch(/branches|pull_request|schedule|default: auto|preid/);
    });

    test('one job, OIDC, no token secrets', () => {
        expect(workflow.match(/^ {2}[A-Za-z0-9_-]+:\n {4}runs-on:/gm) ?? []).toHaveLength(1);
        expect(workflow).toMatch(/permissions:\n {2}contents: write.*\n {2}id-token: write/);
        expect(workflow).not.toMatch(/NPM_TOKEN|NODE_AUTH_TOKEN|personal.access.token|secrets\./i);
    });

    test('pushes only from the cut and catch-up steps, never forced', () => {
        for (const current of steps) {
            const script = current.run ?? '';
            expect(script).not.toMatch(/--force|\+refs\/heads\/main(?!:refs\/remotes)|push.*-f\b/);
            if (/\bgit push\b/.test(script)) {
                expect(['Cut release commit and tag', "Catch main's package.json up to the release"]).toContain(current.name);
            }
        }
    });

    test('publishes packages/gi, not the private workspace root', () => {
        expect(step('Publish to npm').run).toMatch(/cd packages\/ddgi\nnpm publish --access public --tag "\$DIST_TAG"/);
    });

    test('CI and publishing install the same npm', () => {
        const spec = (text: string) => text.match(/npm install -g (npm@\S+)/)?.[1];
        expect(spec(workflow)).toBeDefined();
        expect(spec(workflow)).not.toBe('npm@latest');
        expect(spec(ci)).toBe(spec(workflow));
    });
});

describe('ci.yml', () => {
    const manifest = JSON.parse(readFileSync(new URL('../../package.json', import.meta.url), 'utf8')) as { scripts: Record<string, string> };
    const runs = [...ci.matchAll(/^ {12}(?:- | {2})run: (.+)$/gm)].map((match) => match[1]);

    test('runs the checks in order, each a script the repository has', () => {
        expect(runs).toEqual([
            'npm install -g npm@^11.5.1',
            // Clones the linked siblings (root package.json `linkedSiblings`): now only the
            // optional `@pathtracer/mlp`, so CI also runs the radiance cache's neural tail.
            // Goes away when that is on npm too.
            'corepack enable && node tools/linked-siblings.ts --fetch',
            'pnpm install --frozen-lockfile',
            'pnpm lint',
            'pnpm typecheck',
            'pnpm test',
            'pnpm test:release',
            'pnpm build',
            'pnpm verify:packed --skip-build',
        ]);
        for (const command of runs.filter((line) => /^pnpm (?!install)/.test(line))) {
            expect(manifest.scripts, command).toHaveProperty(command.split(' ')[1]);
        }
    });

    test('on Ubuntu and Windows, with no GPU step', () => {
        expect(ci).toMatch(/os: \[ubuntu-latest, windows-latest\]/);
        expect(runs.join('\n')).not.toMatch(/test:gpu/);
    });
});

//* A tag push ===

describeRuns('a tag push (including a tag made in the GitHub UI)', () => {
    test(
        'the tag is the version: stamps packages/gi, publishes it, makes the Release from CHANGELOG, catches main up',
        () =>
            withFixture({ tag: 'v0.2.0' }, (fixture) => {
                const tagged = originGit(fixture, ['rev-parse', 'v0.2.0^{commit}']);
                const job = runJob(fixture, { tag: 'v0.2.0' });

                expect(job.failed, job.output).toBeNull();
                expect(job.output).toContain('packages/gi/package.json at v0.2.0 says 0.1.0; publishing it as 0.2.0');
                expect(job.outputs.resolve).toEqual({ tag: 'v0.2.0', mode: 'tag' });
                expect(job.outputs.release).toMatchObject({ version: '0.2.0', prerelease: 'false', latest: 'true' });
                expect(job.outputs.dist).toEqual({ tag: 'latest' });
                expect(job.installs).toEqual(['pnpm install --frozen-lockfile']);
                expect(job.publishes).toEqual(['publish --access public --tag latest']);
                expect(job.published).toEqual(['@pmndrs/gi@0.2.0 from ddgi']);
                expect(job.releases).toHaveLength(1);
                expect(job.releases[0]).toMatch(/^release create v0\.2\.0 .*--verify-tag.* --latest$/);
                expect(job.notes).toContain('Notes for 0.2.0.');
                expect(job.notes).not.toContain('0.3.0');
                expect(job.notes).toContain('blob/v0.2.0/CHANGELOG.md');

                // The catch-up: one fast-forward commit, untagged; the tag never moves.
                expect(mainSubjects(fixture).slice(0, 2)).toEqual(['release: v0.2.0 [skip ci]', 'Notes for 0.2.0 and 0.3.0']);
                expect(versionAt(fixture, 'main')).toBe('0.2.0');
                expect(originGit(fixture, ['rev-parse', 'main~1'])).toBe(tagged);
                expect(originGit(fixture, ['rev-parse', 'v0.2.0^{commit}'])).toBe(tagged);
            }),
        SLOW,
    );

    test(
        'a prerelease publishes under its own dist-tag, as a GitHub prerelease, with its version\'s notes',
        () =>
            withFixture({ tag: 'v0.3.0-beta.0' }, (fixture) => {
                const job = runJob(fixture, { tag: 'v0.3.0-beta.0' });

                expect(job.failed, job.output).toBeNull();
                expect(job.publishes).toEqual(['publish --access public --tag beta']);
                expect(job.releases[0]).toMatch(/--prerelease --latest=false$/);
                expect(job.notes).toContain('Notes for 0.3.0.');
            }),
        SLOW,
    );

    // Not on Windows: the race comes from a bash `git` wrapper, which upscaler found
    // native Git there can't run (it used a pre-push hook instead; not ported).
    test.skipIf(process.platform === 'win32')(
        'retries the catch-up on top of a concurrent merge, without rewriting history',
        () =>
            withFixture({ tag: 'v0.2.0' }, (fixture) => {
                const job = runJob(fixture, { tag: 'v0.2.0', race: true });

                expect(job.failed, job.output).toBeNull();
                expect(job.output).toContain('main moved while catching up (attempt 1 of 3)');
                expect(mainSubjects(fixture).slice(0, 3)).toEqual(['release: v0.2.0 [skip ci]', 'A concurrent merge', 'Notes for 0.2.0 and 0.3.0']);
            }),
        SLOW,
    );

    test(
        'a version already on npm is not published again, and a missing Release is still made',
        () =>
            withFixture({ tag: 'v0.2.0' }, (fixture) => {
                const job = runJob(fixture, { tag: 'v0.2.0', npm: 'published' });

                expect(job.failed, job.output).toBeNull();
                expect(job.output).toContain('@pmndrs/gi@0.2.0 is already on npm; skipping publish.');
                expect(job.publishes).toEqual([]);
                expect(job.installs).toEqual([]);
                expect(job.releases).toHaveLength(1);
            }),
        SLOW,
    );

    test(
        "an existing Release, or a draft, is left as it is",
        () =>
            withFixture({ tag: 'v0.2.0' }, (fixture) => {
                expect(runJob(fixture, { tag: 'v0.2.0', gh: 'published' }).releases).toEqual([]);
                const draft = runJob(fixture, { tag: 'v0.2.0', gh: 'draft' });
                expect(draft.failed, draft.output).toBeNull();
                expect(draft.output).toContain('A draft GitHub Release for v0.2.0 exists');
                expect(draft.releases).toEqual([]);
            }),
        SLOW,
    );
});

describeRuns('dist-tags', () => {
    test(
        'an older line gets its line tag, and latest stays',
        () =>
            withFixture({ tag: 'v0.2.0' }, (fixture) => {
                const job = runJob(fixture, { tag: 'v0.2.0', npmLatest: '0.5.0' });

                expect(job.failed, job.output).toBeNull();
                expect(job.publishes).toEqual(['publish --access public --tag v0.2-latest']);
            }),
        SLOW,
    );

    test(
        'a first publish (npm has no latest) takes latest',
        () =>
            withFixture({ tag: 'v0.2.0' }, (fixture) => {
                expect(runJob(fixture, { tag: 'v0.2.0', npmLatest: 'none' }).publishes).toEqual(['publish --access public --tag latest']);
            }),
        SLOW,
    );

    test.each(['auth', 'network'] as const)(
        'npm failing to say its latest (%s) stops the run before publishing',
        (npmLatest) =>
            withFixture({ tag: 'v0.2.0' }, (fixture) => {
                const job = runJob(fixture, { tag: 'v0.2.0', npmLatest });

                expect(job.failed).toBe('Choose npm dist-tag');
                expect(job.publishes).toEqual([]);
            }),
        SLOW,
    );

    test(
        'npm failing to say whether the version exists stops the run',
        () =>
            withFixture({ tag: 'v0.2.0' }, (fixture) => {
                const job = runJob(fixture, { tag: 'v0.2.0', npm: 'auth' });

                expect(job.failed).toBe('Check npm for this version');
                expect(job.publishes).toEqual([]);
            }),
        SLOW,
    );
});

describeRuns('guards', () => {
    test(
        'a tag on a commit that never reached main is refused',
        () =>
            withFixture({ offMain: 'v0.2.0' }, (fixture) => {
                const job = runJob(fixture, { tag: 'v0.2.0' });

                expect(job.failed).toBe('Verify release tag');
                expect(job.output).toContain('which is not on origin/main');
                expect(job.publishes).toEqual([]);
                expect(job.releases).toEqual([]);
            }),
        SLOW,
    );

    test(
        'a tag that is not v + SemVer is refused',
        () =>
            withFixture({ tag: 'v0.2' }, (fixture) => {
                const job = runJob(fixture, { tag: 'v0.2' });

                expect(job.failed).toBe('Verify release tag');
                expect(job.publishes).toEqual([]);
            }),
        SLOW,
    );

    test(
        'a tag whose CHANGELOG.md has no section for it is refused before publishing',
        () =>
            withFixture({ tag: 'v0.4.0' }, (fixture) => {
                const job = runJob(fixture, { tag: 'v0.4.0' });

                expect(job.failed).toBe('Verify release tag');
                expect(job.output).toContain("CHANGELOG.md at v0.4.0 has no '## 0.4.0' section");
                expect(job.publishes).toEqual([]);
                expect(mainSubjects(fixture)[0]).toBe('Notes for 0.2.0 and 0.3.0');
            }),
        SLOW,
    );
});

//* The Run workflow button ===

describeRuns('the Run workflow button', () => {
    test(
        'cuts the release commit and tag on main, publishes, makes the Release, and needs no catch-up',
        () =>
            withFixture({}, (fixture) => {
                const job = runJob(fixture, { event: 'workflow_dispatch', inputs: { version: 'v0.2.0' } });

                expect(job.failed, job.output).toBeNull();
                expect(job.outputs.resolve).toEqual({ tag: 'v0.2.0', mode: 'cut' });
                expect(job.published).toEqual(['@pmndrs/gi@0.2.0 from ddgi']);
                expect(job.releases).toHaveLength(1);
                expect(job.notes).toContain('Notes for 0.2.0.');
                expect(mainSubjects(fixture).slice(0, 2)).toEqual(['release: v0.2.0 [skip ci]', 'Notes for 0.2.0 and 0.3.0']);
                expect(originGit(fixture, ['rev-parse', 'v0.2.0^{commit}'])).toBe(originGit(fixture, ['rev-parse', 'main']));
                expect(versionAt(fixture, 'main')).toBe('0.2.0');
                // Only packages/gi/package.json changes in the release commit.
                expect(originGit(fixture, ['show', '--name-only', '--format=', 'main'])).toBe('packages/gi/package.json');
            }),
        SLOW,
    );

    test.each([
        ['patch', "'patch' is not a version"],
        ['auto', "'auto' is not a version"],
        ['0.1', "'0.1' is not a version"],
    ])(
        'refuses %j: the version is always given exactly',
        (version, message) =>
            withFixture({}, (fixture) => {
                const job = runJob(fixture, { event: 'workflow_dispatch', inputs: { version } });

                expect(job.failed).toBe('Resolve release');
                expect(job.output).toContain(message);
                expect(mainSubjects(fixture)[0]).toBe('Notes for 0.2.0 and 0.3.0');
            }),
        SLOW,
    );

    test(
        'refuses a version with no CHANGELOG.md section, and pushes nothing',
        () =>
            withFixture({}, (fixture) => {
                const job = runJob(fixture, { event: 'workflow_dispatch', inputs: { version: '0.4.0' } });

                expect(job.failed).toBe('Cut release commit and tag');
                expect(job.output).toContain("CHANGELOG.md on main has no '## 0.4.0' section");
                expect(mainSubjects(fixture)[0]).toBe('Notes for 0.2.0 and 0.3.0');
                expect(originGit(fixture, ['tag', '--list', 'v0.4.0'])).toBe('');
            }),
        SLOW,
    );

    test(
        'refuses a version that is not newer than the tree\'s',
        () =>
            withFixture({}, (fixture) => {
                pushToMain(fixture, 'Already at 0.3.0', '0.3.0');
                const job = runJob(fixture, { event: 'workflow_dispatch', inputs: { version: '0.2.0' } });

                expect(job.failed).toBe('Cut release commit and tag');
                expect(job.output).toContain('0.2.0 is not newer than packages/gi/package.json\'s 0.3.0');
            }),
        SLOW,
    );

    test(
        'refuses to run from a branch other than main',
        () =>
            withFixture({}, (fixture) => {
                const job = runJob(fixture, { event: 'workflow_dispatch', ref: 'refs/heads/feature', inputs: { version: '0.2.0' } });

                expect(job.failed).toBe('Resolve release');
                expect(job.output).toContain('Run this workflow from main');
            }),
        SLOW,
    );

    test(
        'with an existing tag, repeats it: creates nothing, makes a missing Release, leaves main alone',
        () =>
            withFixture({}, (fixture) => {
                const before = originGit(fixture, ['rev-parse', 'main']);
                const job = runJob(fixture, { event: 'workflow_dispatch', inputs: { version: '0.1.0' }, npm: 'published' });

                expect(job.failed, job.output).toBeNull();
                expect(job.outputs.resolve).toEqual({ tag: 'v0.1.0', mode: 'existing' });
                expect(job.publishes).toEqual([]);
                expect(job.releases).toHaveLength(1);
                expect(job.notes).toContain('Notes for 0.1.0.');
                expect(job.output).toContain("main's packages/gi/package.json is 0.1.0, not behind 0.1.0");
                expect(originGit(fixture, ['rev-parse', 'main'])).toBe(before);
            }),
        SLOW,
    );
});
