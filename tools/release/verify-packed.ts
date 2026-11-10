// The packed-consumer check, adapted from @pmndrs/upscaler's
// scripts/verify-packed-subpaths.mjs. It checks what npm would publish, the way
// a user gets it, with no GPU (CI runs it):
//
//   1. packs packages/ddgi with `npm pack`, the tool publish.yml publishes with;
//   2. the tarball holds README, LICENSE, THIRD_PARTY.md, the bundled code's
//      licence texts and every `exports` target, and its manifest asks
//      consumers for nothing from this workspace;
//   3. each entry's import graph (JS and .d.ts) reaches only what it may:
//      `/core` no package at all, `.` and `/three` only three;
//   4. a fresh project with the tarball, three and @webgpu/types installed by
//      npm imports `.`, `/core` and `/three` under Node and typechecks a file
//      using each, under both `bundler` and `nodenext` module resolution;
//   5. a fresh project with the tarball and @webgpu/types only (no three)
//      imports and typechecks `/core`.
//
// Steps 4 and 5 install from the npm registry, so they need the network.
import { spawnSync, type SpawnSyncOptions } from 'node:child_process';
import { existsSync, mkdirSync, mkdtempSync, readFileSync, readdirSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { basename, dirname, join, resolve } from 'node:path';

import ts from 'typescript';

import { isDirectRun } from './release-version.ts';

const REPOSITORY_ROOT = resolve(import.meta.dirname, '../..');
const PACKAGE_DIRECTORY = join(REPOSITORY_ROOT, 'packages/ddgi');
const TSC = join(REPOSITORY_ROOT, 'node_modules/typescript/bin/tsc');
const PACKAGE_NAME = '@pmndrs/ddgi';

export const USAGE = `Usage: node tools/release/verify-packed.ts [options]
Builds and packs @pmndrs/ddgi, then checks the tarball as a user would get it
(no GPU needed; installs three and @webgpu/types from the npm registry).
  --skip-build   pack the existing packages/ddgi/dist instead of building it
  --keep-temp    keep the temporary projects for a look
  --help, -h     print this message`;

/** Files every published tarball must hold, besides each `exports` target. */
export const REQUIRED_FILES = [
    'package.json',
    'README.md',
    'LICENSE',
    'THIRD_PARTY.md',
    'licenses/tinybvh.txt',
    'licenses/emscripten-runtime.txt',
    'licenses/hiprt.txt',
];

/**
 * The bare specifiers each entry's import graph may reach. `/core` is the
 * engine-free platform, so it may reach none: it must work with three absent.
 */
const THREE_SPECIFIERS = [/^three$/, /^three\/(webgpu|tsl)$/, /^three\/addons\//];
export const ENTRY_IMPORTS: Record<string, RegExp[]> = {
    index: THREE_SPECIFIERS,
    three: THREE_SPECIFIERS,
    core: [],
};

interface Consumer {
    /** Packages installed next to the tarball (versions are the repository's own). */
    packages: string[];
    /** Subpaths imported under Node. */
    imports: string[];
    /** A source file using the subpaths, typechecked against the packed types. */
    source: string;
}

export const CONSUMERS: Record<string, Consumer> = {
    three: {
        packages: ['three', '@types/three', '@webgpu/types'],
        imports: [PACKAGE_NAME, `${PACKAGE_NAME}/core`, `${PACKAGE_NAME}/three`],
        // The README's getting started, plus one name from each other entry.
        source: `import * as THREE from 'three/webgpu';
import { GIVolume as RootGIVolume } from '@pmndrs/ddgi';
import { ddgiPreset, requiredLimits, type DDGIPresetName } from '@pmndrs/ddgi/core';
import { createGIRenderer, GIVolume } from '@pmndrs/ddgi/three';

export async function start(preset: DDGIPresetName = 'low'): Promise<void> {
    const { renderer } = await createGIRenderer();
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(50, 1, 0.1, 100);
    const volume = new GIVolume(6, 5, 6, 6, 6, 6);
    volume.position.set(0, 2.5, 0);
    scene.add(volume);
    await volume.start(renderer, scene, { preset });
    renderer.setAnimationLoop(() => renderer.render(scene, camera));
}

export function limitsFor(adapter: GPUAdapter) {
    return requiredLimits(adapter);
}
export const lowRays: number | undefined = ddgiPreset('low').ddgi.raysPerProbe;
// '.' and '/three' share one chunk, so they export the same class.
export const sameClass: typeof GIVolume = RootGIVolume;
`,
    },
    core: {
        packages: ['@webgpu/types'],
        imports: [`${PACKAGE_NAME}/core`],
        // The platform on a bare device, as in docs/api/platform.md.
        source: `import { createGI, DDGITechnique, ddgiPreset, recommendPreset, requiredLimits } from '@pmndrs/ddgi/core';

export async function start(adapter: GPUAdapter) {
    const device = await adapter.requestDevice({ requiredLimits: requiredLimits(adapter) });
    const preset = ddgiPreset(recommendPreset(adapter.info, adapter.limits));
    const gi = createGI(device, { textureLayerSize: preset.textureLayerSize });
    const ddgi = gi.addTechnique(new DDGITechnique({ ...preset.ddgi, volume: { center: [0, 2.5, 0], size: [6, 5, 6], spacing: 1 } }));
    gi.setOutput(ddgi);
    await gi.prepare();
    return gi;
}
`,
    },
};

/** Two module resolutions users have: bundlers (vite and the like) and Node's own. */
const RESOLUTIONS = {
    bundler: { module: 'ESNext', moduleResolution: 'Bundler' },
    nodenext: { module: 'NodeNext', moduleResolution: 'NodeNext' },
};

function consumerTsconfig(resolution: keyof typeof RESOLUTIONS): object {
    return {
        compilerOptions: {
            strict: true,
            target: 'ES2022',
            ...RESOLUTIONS[resolution],
            lib: ['ES2022', 'DOM', 'DOM.Iterable'],
            types: ['@webgpu/types'],
            // Check our declaration files too, not only the consumer's code.
            skipLibCheck: false,
            noEmit: true,
        },
        include: ['main.ts'],
    };
}

//* Helpers ===

/**
 * How to run npm without a shell. On Windows `npm` is a .cmd file, which
 * spawnSync can't start directly, so call npm's JavaScript CLI with Node
 * (from @pmndrs/upscaler's scripts/npm-command.mjs).
 */
export function npmInvocation(args: string[]): [string, string[]] {
    // Under `pnpm run`, npm_execpath is pnpm's CLI, which refuses npm-only flags; trust it only when it is npm's.
    const execPath = process.env.npm_execpath;
    const candidates = [
        execPath && basename(execPath) === 'npm-cli.js' ? execPath : undefined,
        join(dirname(process.execPath), 'node_modules/npm/bin/npm-cli.js'),
        join(dirname(process.execPath), '../lib/node_modules/npm/bin/npm-cli.js'),
        join(dirname(process.execPath), '../../npm/bin/npm-cli.js'),
    ];
    const cli = candidates.find((path) => path !== undefined && existsSync(path));
    if (cli) return [process.execPath, [cli, ...args]];
    if (process.platform !== 'win32') return ['npm', args];
    throw new Error('npm CLI not found next to Node; install npm beside Node.');
}

function run(command: string, args: string[], options: SpawnSyncOptions = {}): string {
    const [file, fileArgs] = command === 'npm' ? npmInvocation(args) : [command, args];
    const result = spawnSync(file, fileArgs, { cwd: REPOSITORY_ROOT, encoding: 'utf8', ...options });
    if (result.status !== 0) {
        const output = `${result.stdout ?? ''}${result.stderr ?? ''}`.trim();
        throw new Error(`${command} ${args.join(' ')} failed (status ${result.status}):\n${output || result.error}`);
    }
    return String(result.stdout ?? '');
}

/** `npm pack --json` prints an array (npm 11 and older) or an object keyed by package name (npm 12). */
export function parsePackJson(stdout: string): Array<{ filename: string; size: number; unpackedSize: number; files: Array<{ path: string; size: number }> }> {
    let parsed: unknown;
    try {
        parsed = JSON.parse(stdout);
    } catch {
        throw new Error(`npm pack --json printed something other than JSON:\n${stdout}`);
    }
    const entries = Array.isArray(parsed) ? parsed : parsed && typeof parsed === 'object' ? Object.values(parsed) : [];
    if (!entries.length || !entries.every((entry) => Array.isArray(entry?.files))) {
        throw new Error(`npm pack --json printed an unknown shape:\n${stdout}`);
    }
    return entries;
}

/**
 * The module specifiers a JS or .d.ts file imports or re-exports (static and
 * dynamic), in order, without repeats. TypeScript's import scanner, not a
 * regular expression: the bundle holds strings like "measured from 'start'"
 * that a pattern would take for imports.
 */
export function importSpecifiers(source: string): string[] {
    const { importedFiles } = ts.preProcessFile(source, true, true);
    return [...new Set(importedFiles.map((file) => file.fileName))];
}

/**
 * The file a relative import names. Declaration files import their chunks as
 * `./chunks/x.js`, which TypeScript reads as `./chunks/x.d.ts`, so within a
 * .d.ts graph a `.js` specifier means the `.d.ts` beside it.
 */
export function resolveRelative(importer: string, specifier: string): string {
    const path = resolve(dirname(importer), specifier);
    return importer.endsWith('.d.ts') && path.endsWith('.js') ? `${path.slice(0, -'.js'.length)}.d.ts` : path;
}

/** Follows an entry's relative imports through the package and collects every bare specifier reached. */
export function bareImports(entry: string, read: (path: string) => string = (path) => readFileSync(path, 'utf8')): Set<string> {
    const bare = new Set<string>();
    const seen = new Set<string>();
    const queue = [entry];
    while (queue.length) {
        const file = queue.pop() as string;
        if (seen.has(file)) continue;
        seen.add(file);
        for (const specifier of importSpecifiers(read(file))) {
            if (specifier.startsWith('.')) queue.push(resolveRelative(file, specifier));
            else bare.add(specifier);
        }
    }
    return bare;
}

/** Every entry's JS and declaration graph against its allowed specifiers; returns the failures. */
export function checkImportGraph(packageDirectory: string, read?: (path: string) => string): string[] {
    const failures: string[] = [];
    for (const [entry, allowed] of Object.entries(ENTRY_IMPORTS)) {
        for (const extension of ['.js', '.d.ts']) {
            for (const specifier of bareImports(join(packageDirectory, 'dist', entry + extension), read)) {
                if (!allowed.some((pattern) => pattern.test(specifier))) {
                    failures.push(`dist/${entry}${extension} reaches '${specifier}', which ${entry === 'core' ? '/core must not need' : 'is not three'}`);
                }
            }
        }
    }
    return failures;
}

interface Manifest {
    exports: Record<string, Record<string, string>>;
    dependencies?: Record<string, string>;
    peerDependencies?: Record<string, string>;
    optionalDependencies?: Record<string, string>;
    devDependencies?: Record<string, string>;
}

/** Files missing from the tarball, and dependencies a consumer couldn't resolve. */
export function checkTarballContents(paths: string[], manifest: Manifest): string[] {
    const failures: string[] = [];
    const present = new Set(paths.map((path) => path.replace(/^package\//, '')));
    const exportTargets = Object.values(manifest.exports).flatMap((conditions) => Object.values(conditions));
    for (const file of [...REQUIRED_FILES, ...exportTargets.map((target) => target.replace(/^\.\//, ''))]) {
        if (!present.has(file)) failures.push(`the tarball lacks ${file}`);
    }
    // devDependencies may name workspace packages: npm never installs a dependency's devDependencies.
    for (const field of ['dependencies', 'peerDependencies', 'optionalDependencies'] as const) {
        for (const [name, version] of Object.entries(manifest[field] ?? {})) {
            if (version.startsWith('workspace:') || name.startsWith('@pmndrs/ddgi-') || name === '@pmndrs/bvh') {
                failures.push(`${field} names ${name}@${version}, which isn't on npm`);
            }
        }
    }
    return failures;
}

function installedVersion(name: string): string {
    const manifestPath = join(REPOSITORY_ROOT, 'node_modules', name, 'package.json');
    if (!existsSync(manifestPath)) throw new Error(`${name} isn't installed in the repository; run pnpm install.`);
    return (JSON.parse(readFileSync(manifestPath, 'utf8')) as { version: string }).version;
}

function createConsumer(directory: string, archive: string, consumer: Consumer): void {
    mkdirSync(directory);
    const dependencies: Record<string, string> = { [PACKAGE_NAME]: `file:${archive}` };
    for (const name of consumer.packages) dependencies[name] = installedVersion(name);
    writeFileSync(
        join(directory, 'package.json'),
        `${JSON.stringify({ name: 'ddgi-packed-consumer', private: true, type: 'module', dependencies }, null, 4)}\n`,
    );
    writeFileSync(join(directory, 'main.ts'), consumer.source);
    run('npm', ['install', '--no-audit', '--no-fund', '--no-package-lock', '--ignore-scripts', '--prefer-offline', '--loglevel=error'], {
        cwd: directory,
    });
}

/** Imports each subpath in a fresh Node process, as `import('@pmndrs/ddgi/core')` in user code would. */
function importUnderNode(directory: string, specifiers: string[]): void {
    const script = `for (const specifier of ${JSON.stringify(specifiers)}) {
    const module = await import(specifier);
    if (Object.keys(module).length === 0) throw new Error(specifier + ' has no exports');
}`;
    run(process.execPath, ['--input-type=module', '-e', script], { cwd: directory });
}

//* Main ===

export function parseArguments(argv: string[]): { skipBuild: boolean; keepTemp: boolean; help: boolean } {
    const options = { skipBuild: false, keepTemp: false, help: false };
    for (const argument of argv) {
        if (argument === '--skip-build') options.skipBuild = true;
        else if (argument === '--keep-temp') options.keepTemp = true;
        else if (argument === '--help' || argument === '-h') options.help = true;
        else throw new Error(`Unknown option ${argument}.\n\n${USAGE}`);
    }
    return options;
}

function main(): void {
    const options = parseArguments(process.argv.slice(2));
    if (options.help) {
        console.log(USAGE);
        return;
    }
    if (!options.skipBuild) {
        console.log(`Building ${PACKAGE_NAME}...`);
        run('pnpm', ['build'], { stdio: 'inherit', shell: process.platform === 'win32' });
    }

    const temporaryRoot = mkdtempSync(join(tmpdir(), 'ddgi-packed-'));
    try {
        const [packed] = parsePackJson(
            run('npm', ['pack', '--ignore-scripts', '--json', '--pack-destination', temporaryRoot], { cwd: PACKAGE_DIRECTORY }),
        );
        // npm 8 on Windows reported a scoped file name that differs from the file it wrote; trust the folder.
        const tarballs = readdirSync(temporaryRoot).filter((name) => name.endsWith('.tgz'));
        if (tarballs.length !== 1) throw new Error(`npm pack wrote ${tarballs.length} tarballs, expected 1.`);
        const archive = join(temporaryRoot, tarballs[0]);
        console.log(
            `Packed ${tarballs[0]}: ${packed.files.length} files, ${(packed.size / 1000).toFixed(1)} kB packed, ${(packed.unpackedSize / 1000).toFixed(1)} kB unpacked.`,
        );

        const unpacked = join(temporaryRoot, 'unpacked');
        mkdirSync(unpacked);
        run('tar', ['-xzf', archive, '-C', unpacked, '--strip-components=1'], { cwd: temporaryRoot });
        const manifest = JSON.parse(readFileSync(join(unpacked, 'package.json'), 'utf8')) as Manifest;
        const failures = [
            ...checkTarballContents(packed.files.map((file) => file.path), manifest),
            ...checkImportGraph(unpacked),
        ];
        if (failures.length) throw new Error(`The tarball:\n${failures.join('\n')}`);
        console.log(`Contents and import graphs clean: ${REQUIRED_FILES.join(', ')} present; /core reaches no package, '.' and /three only three.`);

        for (const [name, consumer] of Object.entries(CONSUMERS)) {
            const directory = join(temporaryRoot, `consumer-${name}`);
            console.log(`Consumer '${name}' (${consumer.packages.join(', ')}): npm install, import under Node, typecheck...`);
            createConsumer(directory, archive, consumer);
            if (!consumer.packages.includes('three') && existsSync(join(directory, 'node_modules/three'))) {
                throw new Error(`Consumer '${name}' got three installed, though it didn't ask for it.`);
            }
            importUnderNode(directory, consumer.imports);
            for (const resolution of Object.keys(RESOLUTIONS) as Array<keyof typeof RESOLUTIONS>) {
                const tsconfig = join(directory, `tsconfig.${resolution}.json`);
                writeFileSync(tsconfig, `${JSON.stringify(consumerTsconfig(resolution), null, 4)}\n`);
                run(process.execPath, [TSC, '-p', tsconfig], { cwd: directory });
            }
        }
        console.log('Packed-consumer check passed.');
    } finally {
        if (options.keepTemp) console.log(`Kept the temporary projects in ${temporaryRoot}`);
        else rmSync(temporaryRoot, { recursive: true, force: true, maxRetries: 5, retryDelay: 200 });
    }
}

if (isDirectRun(import.meta.url)) {
    try {
        main();
    } catch (error) {
        console.error(error instanceof Error ? error.message : error);
        process.exitCode = 1;
    }
}
