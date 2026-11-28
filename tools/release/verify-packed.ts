// The packed-consumer check, adapted from @pmndrs/upscaler's
// scripts/verify-packed-subpaths.mjs. It checks what npm would publish, the way
// a user gets it, with no GPU (CI runs it):
//
//   1. packs packages/gi with `npm pack`, the tool publish.yml publishes with;
//   2. the tarball holds README, LICENSE, THIRD_PARTY.md, the bundled code's
//      licence texts and every `exports` target, and its manifest asks
//      consumers for nothing from this workspace;
//   3. each entry's import graph (JS and .d.ts) reaches only what it may:
//      `/core` and `/ddgi` no package at all, `.` and `/three` only three,
//      `/radiance-cache` only its denoiser (`denoiser/ffx`), never
//      `@pathtracer/mlp` (an optional sibling it loads at run time); and only
//      its own layers' chunks (packages/gi/layer-chunks.ts): `/core` no
//      technique or three code, `/ddgi` no three code;
//   4. a fresh project with the tarball, three and @webgpu/types installed by
//      npm imports `.`, `/core`, `/ddgi`, `/three` and `/radiance-cache` under
//      Node and typechecks a file using each (the radiance cache wired into
//      three as tools/gpu/pages/radiance-cache-three does, from public paths
//      only), under both `bundler` and `nodenext` module resolution;
//   5. a fresh project with the tarball and @webgpu/types only (no three)
//      imports and typechecks `/core`, `/ddgi`, `/radiance-cascades` and
//      `/radiance-cache`, and finds no name in both `/core` and `/ddgi`;
//   6. a fresh vite app importing every public subpath builds with `vite build`
//      and is served by vite's dev server, and every file the built or served
//      code fetches with `new URL(..., import.meta.url)` is there. Typechecks
//      and Node imports run no bundler: they missed BVH workers that the
//      package build had pointed at an absolute `/assets/` path.
//
// Neither project has `@pathtracer/mlp`: `/radiance-cache` must import and
// typecheck without it.
//
// Steps 4 to 6 install from the npm registry, so they need the network.
import { spawnSync, type SpawnSyncOptions } from 'node:child_process';
import { existsSync, mkdirSync, mkdtempSync, readFileSync, readdirSync, realpathSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { basename, dirname, join, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';

import ts from 'typescript';
import type * as Vite from 'vite';

import { ENTRY_LAYERS, LAYER_CHUNKS, type Layer } from '../../packages/gi/layer-chunks.ts';
import { isDirectRun } from './release-version.ts';

const REPOSITORY_ROOT = resolve(import.meta.dirname, '../..');
const PACKAGE_DIRECTORY = join(REPOSITORY_ROOT, 'packages/gi');
const TSC = join(REPOSITORY_ROOT, 'node_modules/typescript/bin/tsc');
const PACKAGE_NAME = '@pmndrs/gi';

export const USAGE = `Usage: node tools/release/verify-packed.ts [options]
Builds and packs @pmndrs/gi, then checks the tarball as a user would get it
(no GPU needed; installs three, vite and @webgpu/types from the npm registry).
  --skip-build   pack the existing packages/gi/dist instead of building it
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
    'licenses/pathtracer-MIT.txt',
    'licenses/pbrt-v4-Apache-2.0.txt',
    'licenses/capsaicin-MIT.txt',
    'licenses/rtoy-samples-MIT.txt',
    'licenses/assimp-BSD-3-Clause.txt',
];

/**
 * The bare specifiers each entry's import graph may reach. `/core` (the
 * platform) and `/ddgi` (the DDGI technique) are engine-free, so they may reach
 * none: they must work with three absent.
 */
const THREE_SPECIFIERS = [/^three$/, /^three\/(webgpu|tsl)$/, /^three\/addons\//];
export const ENTRY_IMPORTS: Record<string, RegExp[]> = {
    index: THREE_SPECIFIERS,
    three: THREE_SPECIFIERS,
    core: [],
    ddgi: [],
    'radiance-cascades': [/^@pmndrs\/rc-bvh$/],
    // Its denoiser, an npm dependency. `@pathtracer/mlp` (the optional neural tail) is loaded
    // through an `import()` no scanner sees (packages/gi/vite.config.ts), so it isn't listed.
    'radiance-cache': [/^denoiser\/ffx$/],
};

/** Entries that must work with no engine installed. */
const ENGINE_FREE_ENTRIES = ['core', 'ddgi', 'radiance-cascades', 'radiance-cache'];

/** Packages the published package may load but must never name in its manifest or install (not on npm). */
export const OPTIONAL_RUNTIME_PACKAGES = ['@pathtracer/mlp'];

interface Consumer {
    /** Packages installed next to the tarball (versions are the repository's own). */
    packages: string[];
    /** Subpaths imported under Node. */
    imports: string[];
    /** A source file using the subpaths, typechecked against the packed types. */
    source: string;
    /** An ES module run under Node after the imports; it throws when the subpaths are wrong. */
    nodeCheck?: string;
}

export const CONSUMERS: Record<string, Consumer> = {
    three: {
        packages: ['three', '@types/three', '@webgpu/types'],
        imports: [PACKAGE_NAME, `${PACKAGE_NAME}/core`, `${PACKAGE_NAME}/ddgi`, `${PACKAGE_NAME}/three`, `${PACKAGE_NAME}/radiance-cache`],
        // The README's getting started, plus one name from each other entry, and the radiance
        // cache wired into three from public paths (tools/gpu/pages/radiance-cache-three, the viewer).
        source: `import * as THREE from 'three/webgpu';
import { mrt, normalView, output, packNormalToRGB, pass } from 'three/tsl';
import { DDGIVolume as RootDDGIVolume } from '@pmndrs/gi';
import { createGI, requiredLimits } from '@pmndrs/gi/core';
import { ddgiPreset, type DDGIPresetName } from '@pmndrs/gi/ddgi';
import { RADIANCE_CACHE_DEBUG_VIEW, RadianceCacheTechnique, type RadianceCacheSettings } from '@pmndrs/gi/radiance-cache';
import { createGIRenderer, ddgi, DDGIVolume, giAlbedo, GI_ALBEDO_OUTPUT, GI_NORMAL_OUTPUT, GILight, screenInputs, ThreeGI } from '@pmndrs/gi/three';

export async function radianceCache(scene: THREE.Scene, camera: THREE.PerspectiveCamera, settings: Partial<RadianceCacheSettings> = {}) {
    const { renderer, device } = await createGIRenderer();
    const gi = createGI(device);
    const technique = gi.addTechnique(new RadianceCacheTechnique(settings));
    gi.setOutput(technique);
    const preparing = gi.prepare();
    const threeGI = new ThreeGI(gi, scene, { preparing });
    await preparing;
    const scenePass = pass(scene, camera);
    const stage = screenInputs(scenePass, threeGI);
    scenePass.setMRT(mrt({ output, [GI_ALBEDO_OUTPUT]: giAlbedo, [GI_NORMAL_OUTPUT]: packNormalToRGB(normalView), ...stage.outputs }));
    const pipeline = new THREE.RenderPipeline(renderer);
    pipeline.outputNode = stage.encodeBefore(ddgi(scenePass, threeGI));
    // Or the composite on a light of its own, beside another technique's (the GPU page's way).
    const ownLight = ddgi(scenePass, { light: new GILight(technique.shading) });
    renderer.setAnimationLoop(() => {
        threeGI.update(camera, 1 / 60);
        pipeline.render();
    });
    return { technique, ownLight, probes: technique.probes, shView: RADIANCE_CACHE_DEBUG_VIEW.sphericalHarmonics };
}

export async function start(preset: DDGIPresetName = 'low'): Promise<void> {
    const { renderer } = await createGIRenderer();
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(50, 1, 0.1, 100);
    const volume = new DDGIVolume(6, 5, 6, 6, 6, 6);
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
export const sameClass: typeof DDGIVolume = RootDDGIVolume;
`,
        // Every module is in one chunk, so a class is the same object from every subpath.
        nodeCheck: `const root = await import('${PACKAGE_NAME}');
const three = await import('${PACKAGE_NAME}/three');
const ddgi = await import('${PACKAGE_NAME}/ddgi');
if (root.DDGIVolume !== three.DDGIVolume) throw new Error("'.' and '/three' export different DDGIVolume classes");
const volume = new three.DDGIVolume(6, 5, 6, 6, 6, 6);
const settings = volume.ddgiVolume();
const technique = new ddgi.DDGITechnique({ volume: settings });
if (!(technique instanceof ddgi.DDGITechnique)) throw new Error('DDGITechnique failed instanceof');
const cache = await import('${PACKAGE_NAME}/radiance-cache');
const radianceCache = new cache.RadianceCacheTechnique();
new three.GILight(radianceCache.shading);
`,
    },
    core: {
        packages: ['@webgpu/types'],
        imports: [`${PACKAGE_NAME}/core`, `${PACKAGE_NAME}/ddgi`, `${PACKAGE_NAME}/radiance-cascades`, `${PACKAGE_NAME}/radiance-cache`],
        // The platform and DDGI on a bare device, as in docs/api/platform.md; the radiance cache the same way.
        source: `import { createGI, requiredLimits } from '@pmndrs/gi/core';
import { DDGITechnique, ddgiPreset, recommendPreset } from '@pmndrs/gi/ddgi';
import { NativeRadianceCascades, createRcTransport, type NativeRcTransport, type RcTransportScene } from '@pmndrs/gi/radiance-cascades';
import { DEFAULT_RADIANCE_CACHE_SETTINGS, RadianceCacheTechnique } from '@pmndrs/gi/radiance-cache';

export async function startRadianceCache(device: GPUDevice) {
    const gi = createGI(device);
    gi.setOutput(gi.addTechnique(new RadianceCacheTechnique({ ...DEFAULT_RADIANCE_CACHE_SETTINGS, neuralCache: 'off' })));
    await gi.prepare();
    return gi;
}

export async function makeTransport(device: GPUDevice, scene: RcTransportScene): Promise<NativeRcTransport> {
    return createRcTransport(device, scene);
}
export const createCascades = NativeRadianceCascades.create;

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
        // The platform holds no DDGI: no name is in both subpaths.
        nodeCheck: `const core = await import('${PACKAGE_NAME}/core');
const ddgi = await import('${PACKAGE_NAME}/ddgi');
const both = Object.keys(core).filter((name) => name in ddgi);
if (both.length) throw new Error('/core and /ddgi both export ' + both.join(', '));
for (const name of ['DDGITechnique', 'ddgiPreset']) if (name in core) throw new Error('/core exports ' + name);
for (const name of ['createGI', 'DDGITechnique', 'ddgiPreset']) if (!(name in core) && !(name in ddgi)) throw new Error(name + ' is missing');
`,
    },
};

/**
 * A vite app, the way most users bundle the package. It imports every public
 * subpath and keeps names from each reachable, so the bundle holds the code
 * that starts the BVH workers. Built with `vite build` and served by vite's dev
 * server, which pre-bundles the package into node_modules/.vite first.
 */
export const VITE_APP = {
    packages: ['three', 'vite', '@webgpu/types'],
    html: `<!doctype html>
<html>
    <body>
        <script type="module" src="/main.ts"></script>
    </body>
</html>
`,
    source: `import { createGI, requiredLimits } from '@pmndrs/gi/core';
import { DDGITechnique, ddgiPreset } from '@pmndrs/gi/ddgi';
import { RadianceCacheTechnique } from '@pmndrs/gi/radiance-cache';
import { NativeRadianceCascades, createRcTransport } from '@pmndrs/gi/radiance-cascades';
import { createGIRenderer, DDGIVolume } from '@pmndrs/gi/three';

// Published on globalThis so the bundler keeps every subpath's code.
Object.assign(globalThis, { createGI, requiredLimits, DDGITechnique, ddgiPreset, RadianceCacheTechnique, NativeRadianceCascades, createRcTransport, createGIRenderer, DDGIVolume });
`,
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

/** Follows an entry's relative imports through the package: every file reached (the entry too) and every bare specifier. */
export function importGraph(entry: string, read: (path: string) => string = (path) => readFileSync(path, 'utf8')): { files: Set<string>; bare: Set<string> } {
    const bare = new Set<string>();
    const files = new Set<string>();
    const queue = [entry];
    while (queue.length) {
        const file = queue.pop() as string;
        if (files.has(file)) continue;
        files.add(file);
        for (const specifier of importSpecifiers(read(file))) {
            if (specifier.startsWith('.')) queue.push(resolveRelative(file, specifier));
            else bare.add(specifier);
        }
    }
    return { files, bare };
}

/** Every bare specifier an entry's graph reaches. */
export function bareImports(entry: string, read?: (path: string) => string): Set<string> {
    return importGraph(entry, read).bare;
}

/**
 * The layer a file in dist holds (packages/gi/layer-chunks.ts): a chunk's
 * from its name (`chunks/ddgi-<hash>.js`), an entry's from the entry it is.
 */
export function layerOf(packageDirectory: string, file: string): Layer | undefined {
    const dist = join(packageDirectory, 'dist');
    const name = basename(file).replace(/\.d\.ts$|\.js$/, '');
    if (dirname(file) === join(dist, 'chunks')) return LAYER_CHUNKS.find((layer) => name.startsWith(`${layer}-`));
    if (dirname(file) === dist) return ENTRY_LAYERS[name]?.at(-1);
    return undefined;
}

/**
 * Every entry's JS and declaration graph against its allowed specifiers and
 * layers; returns the failures. A layer outside an entry's list means code of a
 * higher layer leaked into it: DDGI in `/core`, or three in `/ddgi`.
 */
export function checkImportGraph(packageDirectory: string, read?: (path: string) => string): string[] {
    const failures: string[] = [];
    for (const [entry, allowed] of Object.entries(ENTRY_IMPORTS)) {
        const layers = ENTRY_LAYERS[entry];
        for (const extension of ['.js', '.d.ts']) {
            const graph = importGraph(join(packageDirectory, 'dist', entry + extension), read);
            for (const specifier of graph.bare) {
                if (!allowed.some((pattern) => pattern.test(specifier))) {
                    const why = ENGINE_FREE_ENTRIES.includes(entry) ? `/${entry} must not need` : 'is not three';
                    failures.push(`dist/${entry}${extension} reaches '${specifier}', which ${why}`);
                }
            }
            for (const file of graph.files) {
                const layer = layerOf(packageDirectory, file);
                if (layer && !layers.includes(layer)) {
                    failures.push(`dist/${entry}${extension} reaches ${basename(file)}, which holds ${layer} code; /${entry} may load only ${layers.join(', ')}`);
                }
            }
        }
    }
    return failures;
}

/**
 * The string targets of `new URL('...', import.meta.url)` in a JS file: files
 * the code fetches from beside itself (workers, WebAssembly, images). A
 * comment before the string is allowed, as vite writes `/* @vite-ignore *\/`.
 * A template with `${...}` in it is skipped: its path is made at run time, so
 * there is no file to look for (vite's own client builds
 * `${base}__open-in-editor?file=...` this way).
 */
export function importMetaUrls(source: string): string[] {
    const pattern = /new\s+URL\(\s*(?:\/\*[\s\S]*?\*\/\s*)*(['"`])([^'"`]+)\1\s*,\s*import\.meta\.url\s*\)/g;
    return [...source.matchAll(pattern)].filter((match) => !(match[1] === '`' && match[2].includes('${'))).map((match) => match[2]);
}

/** Targets that aren't files to look for: inline data and other origins. */
const NOT_A_FILE = /^(data|blob|https?):/;

/**
 * The `new URL(..., import.meta.url)` targets in a built app that the build
 * didn't write. `files` maps each built file's path relative to the output
 * folder (`assets/index-x.js`, forward slashes) to its text. A target starting
 * with `/` is from the site root, which is the output folder.
 */
export function missingBuiltUrls(files: Record<string, string>): string[] {
    const failures: string[] = [];
    for (const [file, text] of Object.entries(files)) {
        if (!file.endsWith('.js')) continue;
        for (const target of importMetaUrls(text)) {
            if (NOT_A_FILE.test(target)) continue;
            // The output folder is the URL root, so the path past "/" is the built file's name.
            const path = decodeURI(new URL(target, `file:///${file}`).pathname.slice(1));
            if (!(path in files)) failures.push(`${file} fetches '${target}', which the build didn't write`);
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
            if (version.includes('&path:') || version.startsWith('workspace:') || name.startsWith('@pmndrs/ddgi-') || name === '@pmndrs/bvh' || OPTIONAL_RUNTIME_PACKAGES.includes(name)) {
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

/** Imports each subpath in a fresh Node process, as `import('@pmndrs/gi/core')` in user code would, then runs the consumer's own check. */
function importUnderNode(directory: string, specifiers: string[], nodeCheck = ''): void {
    const script = `for (const specifier of ${JSON.stringify(specifiers)}) {
    const module = await import(specifier);
    if (Object.keys(module).length === 0) throw new Error(specifier + ' has no exports');
}
${nodeCheck}`;
    run(process.execPath, ['--input-type=module', '-e', script], { cwd: directory });
}

function filesUnder(directory: string, prefix = ''): string[] {
    return readdirSync(directory, { withFileTypes: true }).flatMap((entry) =>
        entry.isDirectory() ? filesUnder(join(directory, entry.name), `${prefix}${entry.name}/`) : [`${prefix}${entry.name}`],
    );
}

/** Step 6, building: `vite build` succeeds, and every file the bundle fetches from beside itself was written. */
function viteBuild(directory: string): void {
    run(process.execPath, [join(directory, 'node_modules/vite/bin/vite.js'), 'build', '--logLevel', 'warn'], { cwd: directory });
    const out = join(directory, 'dist');
    const files = Object.fromEntries(filesUnder(out).map((file) => [file, readFileSync(join(out, file), 'utf8')]));
    const failures = missingBuiltUrls(files);
    if (!Object.values(files).some((text) => text.includes('new Worker('))) {
        failures.push('the built app starts no worker, so it lacks the code this check is for (VITE_APP keeps too few names)');
    }
    if (failures.length) throw new Error(`The vite app's build:\n${failures.join('\n')}`);
}

/**
 * Step 6, serving: vite's dev server serves the app's entry and every module it
 * imports as JavaScript, and every `new URL(..., import.meta.url)` target in
 * them. The dev server answers an unknown path with the app's index.html, so
 * an HTML answer counts as missing.
 *
 * Runs in its own Node process (`viteDevServer`): the dev server loads
 * rollup's native addon and starts esbuild from the app's node_modules, and
 * Windows can't delete either file while this process holds it.
 */
export async function checkViteDevServer(directory: string): Promise<void> {
    // The real path, for both the root and vite itself: vite serves only files under
    // its root, compared after resolving them. With a Windows short name in the path
    // (C:\Users\RUNNER~1, on CI), a short root refuses /main.ts with a 403, and a vite
    // loaded from the short path looks for its own /@vite/client there and answers
    // with index.html instead.
    const root = realpathSync.native(directory);
    // The app's own vite, not the repository's: the one a user would run.
    const vite = (await import(pathToFileURL(join(root, 'node_modules/vite/dist/node/index.js')).href)) as typeof Vite;
    const server = await vite.createServer({ root, configFile: false, logLevel: 'error', server: { host: '127.0.0.1', port: 0 } });
    await server.listen();
    const failures: string[] = [];
    try {
        const origin = server.resolvedUrls?.local[0];
        if (!origin) throw new Error("vite's dev server reported no address.");
        const seen = new Set<string>();
        const queue: Array<{ url: string; from: string; kind: 'module' | 'file' }> = [{ url: new URL('/main.ts', origin).href, from: 'index.html', kind: 'module' }];
        while (queue.length) {
            const { url, from, kind } = queue.shift() as (typeof queue)[number];
            if (seen.has(url)) continue;
            seen.add(url);
            const response = await fetch(url);
            const type = response.headers.get('content-type') ?? '';
            const text = await response.text();
            const path = new URL(url).pathname;
            if (!response.ok) failures.push(`${from} loads ${path}: HTTP ${response.status}`);
            else if (kind === 'module' && !type.includes('javascript')) failures.push(`${from} imports ${path}, served as ${type || 'no type'}`);
            else if (kind === 'file' && type.includes('html')) failures.push(`${from} fetches ${path}, which isn't there (the dev server answered with index.html)`);
            if (!response.ok || !type.includes('javascript')) continue;
            for (const specifier of importSpecifiers(text)) {
                if (!NOT_A_FILE.test(specifier)) queue.push({ url: new URL(specifier, url).href, from: path, kind: 'module' });
            }
            for (const target of importMetaUrls(text)) {
                if (!NOT_A_FILE.test(target)) queue.push({ url: new URL(target, url).href, from: path, kind: 'file' });
            }
        }
    } finally {
        await server.close();
    }
    if (failures.length) throw new Error(`The vite app under vite's dev server:\n${failures.join('\n')}`);
}

/** Step 6, serving, in a fresh Node process that exits when it is done (see checkViteDevServer). */
function viteDevServer(directory: string): void {
    const script = `const { checkViteDevServer } = await import(${JSON.stringify(import.meta.url)});
try {
    await checkViteDevServer(${JSON.stringify(directory)});
    process.exit(0);
} catch (error) {
    console.error(error instanceof Error ? error.message : error);
    process.exit(1);
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
        console.log(`Contents and import graphs clean: ${REQUIRED_FILES.join(', ')} present; /core and /ddgi reach no package, '.' and /three only three; /radiance-cascades reaches only its published BVH dependency, /radiance-cache only denoiser/ffx.`);

        for (const [name, consumer] of Object.entries(CONSUMERS)) {
            const directory = join(temporaryRoot, `consumer-${name}`);
            console.log(`Consumer '${name}' (${consumer.packages.join(', ')}): npm install, import under Node, typecheck...`);
            createConsumer(directory, archive, consumer);
            if (!consumer.packages.includes('three') && existsSync(join(directory, 'node_modules/three'))) {
                throw new Error(`Consumer '${name}' got three installed, though it didn't ask for it.`);
            }
            for (const optional of OPTIONAL_RUNTIME_PACKAGES) {
                if (existsSync(join(directory, 'node_modules', optional))) throw new Error(`Consumer '${name}' got ${optional} installed: the check is that it works without it.`);
            }
            importUnderNode(directory, consumer.imports, consumer.nodeCheck);
            for (const resolution of Object.keys(RESOLUTIONS) as Array<keyof typeof RESOLUTIONS>) {
                const tsconfig = join(directory, `tsconfig.${resolution}.json`);
                writeFileSync(tsconfig, `${JSON.stringify(consumerTsconfig(resolution), null, 4)}\n`);
                run(process.execPath, [TSC, '-p', tsconfig], { cwd: directory });
            }
        }

        const app = join(temporaryRoot, 'consumer-vite');
        console.log(`Vite app (${VITE_APP.packages.join(', ')}): npm install, vite build, vite dev server...`);
        createConsumer(app, archive, { packages: VITE_APP.packages, imports: [], source: VITE_APP.source });
        writeFileSync(join(app, 'index.html'), VITE_APP.html);
        viteBuild(app);
        viteDevServer(app);
        console.log('Packed-consumer check passed.');
    } finally {
        if (options.keepTemp) console.log(`Kept the temporary projects in ${temporaryRoot}`);
        else {
            // The folder holds only npm installs (no links out). A failed delete
            // must not hide the check's own result.
            try {
                rmSync(temporaryRoot, { recursive: true, force: true, maxRetries: 5, retryDelay: 200 });
            } catch (error) {
                console.warn(`Couldn't delete ${temporaryRoot}: ${error instanceof Error ? error.message : error}`);
            }
        }
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
