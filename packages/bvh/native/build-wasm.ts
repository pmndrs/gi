/// <reference types="node" />
/**
 * Builds tinybvh-wasm.cpp with a pinned tinybvh commit and a pinned Emscripten
 * into a .wasm, and writes it base64-encoded into `packages/bvh/src/tinybvh-wasm.ts`
 * (with its SHA-256, the versions, the flags and the licence texts), which is
 * what the package loads and what is committed.
 *
 *   pnpm bvh:wasm                  install the pinned emsdk if needed, then build
 *   pnpm bvh:wasm --check          build, and fail unless the result matches the committed .wasm
 *   options: --keep-headers        keep the temporary build folder, for inspection
 *
 * What it does, so the committed .wasm can be checked:
 * 1. Makes sure `tools/emsdk` (gitignored) is emsdk at EMSDK_COMMIT with
 *    Emscripten EMSDK_VERSION installed and active. If the folder is missing it
 *    is cloned from EMSDK_REPOSITORY and `emsdk install` / `emsdk activate` run
 *    inside it. Nothing outside that folder is written, and no other Emscripten
 *    on the machine is used: every EM* variable is dropped from the environment.
 * 2. Downloads tinybvh's headers at TINYBVH_COMMIT into a temporary folder and
 *    checks each against its SHA-256 below. A mismatch stops the build.
 * 3. Compiles there with BUILD_FLAGS, using relative paths only, so no folder
 *    name ends up in the .wasm (tinybvh puts __FILE__ into its error messages).
 *
 * Needs git, and Python 3 on Linux and macOS (emsdk brings its own Python on Windows).
 */

import { execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { copyFileSync, existsSync, mkdtempSync, readdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { parseArgs } from 'node:util';

export const TINYBVH_REPOSITORY = 'https://github.com/jbikker/tinybvh';
/** tinybvh 1.9.1, main on 2026-10-12. */
export const TINYBVH_COMMIT = 'c9be3103cc9693b456b0a0353927c11cac7d710f';

export const EMSDK_REPOSITORY = 'https://github.com/emscripten-core/emsdk';
/**
 * Emscripten 6.0.13, released 2026-10-19. Chosen on 2026-11-04 as the newest
 * release at least two weeks old. 256-bit AVX intrinsics (emulated on WASM
 * SIMD) arrived in 3.1.68; the 3.1.57 used before could not compile tinybvh's
 * AVX builder, which this build exists for.
 */
export const EMSDK_VERSION = '6.0.13';
/** The emsdk tag EMSDK_VERSION points at; checked so a moved tag cannot change the build. */
export const EMSDK_COMMIT = '35659964bc2762fffec0d257997fb61475f765b7';

/** Every file the build reads from tinybvh, with its SHA-256 at TINYBVH_COMMIT. */
const TINYBVH_FILES: Record<string, string> = {
    'tiny_bvh.h': 'd2e7592dc7acb0cee981348fb5f3b5968550098565085f8eb0f7d22a97ed09ef',
    'tiny_bvh_base.h': '62c50c577108db805fa565efa1cf2315116d0a3419f5e9923871ed3fcc16e2fa',
    'tiny_bvh_x86_float.h': 'aecb220d3dfd19f8cfd7aad448e5300fd99a4cf2ece654686df2e21c055da3cd',
    'tiny_bvh_x86_double.h': 'aed9ffa75a06f667ac33dd0d37c6f6f0860395db36c804e334d9e17734a58f5f',
    'tiny_bvh_arm_float.h': '8d0b5869707b1f6c9e2049e0de18731a565ae8880e1baa6d3b10442a4e79887c',
    'tiny_bvh_arm_double.h': '941d0682af0c8cddc632b9284311b9b40d2bd83feeee8186059aa2a01a7075e0',
    LICENSE: 'f50e32f96aacf63d90362309a85de33d133192acb3fc9178c55e23cec92bc8dc',
};

/**
 * Compiler flags, with the reason for each:
 * - `-O3`: speed; the builder is the whole point.
 * - `-DNO_THREADED_BUILDS`: single-threaded. Threads in WASM need pthreads, and
 *   pthreads need cross-origin isolation (COOP/COEP headers), which a library
 *   cannot ask of the pages that use it. One worker builds.
 * - `-msimd128 -mavx`: WASM SIMD, with Emscripten's AVX intrinsics on top of it.
 *   tinybvh's fast builders (Bonsai, and binned SAH) are written in AVX and only
 *   compile when `__AVX__` is defined; without it `Build` falls back to the
 *   scalar reference builder, about 3x slower. AVX2 is not needed: it only adds
 *   the 8-wide CPU tracing layouts, which we don't use.
 * - `-sSTANDALONE_WASM --no-entry`: a plain .wasm with no JS glue; src/tinybvh.ts instantiates it.
 * - `-sALLOW_MEMORY_GROWTH -sMAXIMUM_MEMORY=4GB`: big meshes need hundreds of MB.
 * - `-fno-exceptions`: tinybvh does not use them.
 */
export const BUILD_FLAGS = [
    '-O3',
    '-std=c++20',
    '-DNO_THREADED_BUILDS',
    '-msimd128',
    '-mavx',
    '-fno-exceptions',
    '-sSTANDALONE_WASM',
    '--no-entry',
    '-sALLOW_MEMORY_GROWTH=1',
    '-sMAXIMUM_MEMORY=4GB',
    '-sINITIAL_MEMORY=64MB',
];

const PACKAGE_ROOT = resolve(import.meta.dirname, '..');
const REPOSITORY_ROOT = resolve(PACKAGE_ROOT, '..', '..');
const SOURCE = join(PACKAGE_ROOT, 'native', 'tinybvh-wasm.cpp');
const OUTPUT_MODULE = join(PACKAGE_ROOT, 'src', 'tinybvh-wasm.ts');
const EMSDK_FOLDER = join(REPOSITORY_ROOT, 'tools', 'emsdk');
/** Licence texts that travel inside the generated module (and so inside every bundle of it). */
const LICENSE_FILES = [join(PACKAGE_ROOT, 'licenses', 'tinybvh.txt'), join(PACKAGE_ROOT, 'licenses', 'emscripten-runtime.txt')];
const IS_WINDOWS = process.platform === 'win32';

function sha256(bytes: Buffer): string {
    return createHash('sha256').update(bytes).digest('hex');
}

/** This process's environment without any Emscripten setting, so only tools/emsdk is used. */
function cleanEnvironment(): NodeJS.ProcessEnv {
    const environment: NodeJS.ProcessEnv = {};
    for (const [name, value] of Object.entries(process.env)) {
        if (!/^EM(SDK|_|CC)/i.test(name)) environment[name] = value;
    }
    return environment;
}

function run(command: string, args: string[], options: { cwd: string; env?: NodeJS.ProcessEnv }): string {
    return execFileSync(command, args, { cwd: options.cwd, env: options.env ?? cleanEnvironment(), encoding: 'utf8', maxBuffer: 64 * 1024 * 1024 });
}

/** Runs emsdk's own command (`emsdk.bat` on Windows, `./emsdk` elsewhere) inside tools/emsdk. */
function runEmsdk(args: string[]): void {
    console.log(`emsdk ${args.join(' ')}`);
    // Node does not start .bat files without a shell; cmd.exe runs it with plain arguments.
    const output = IS_WINDOWS
        ? run('cmd.exe', ['/d', '/c', 'emsdk.bat', ...args], { cwd: EMSDK_FOLDER })
        : run('./emsdk', args, { cwd: EMSDK_FOLDER });
    console.log(output.trim().split('\n').slice(-3).join('\n'));
}

function installedEmscriptenVersion(): string | null {
    const file = join(EMSDK_FOLDER, 'upstream', 'emscripten', 'emscripten-version.txt');
    return existsSync(file) ? readFileSync(file, 'utf8').trim().replace(/"/g, '') : null;
}

/** Step 1: tools/emsdk at the pinned commit, with the pinned Emscripten installed and active. */
function ensureEmsdk(): void {
    if (!existsSync(EMSDK_FOLDER)) {
        console.log(`cloning ${EMSDK_REPOSITORY} ${EMSDK_VERSION} into ${EMSDK_FOLDER}`);
        run('git', ['clone', '--quiet', '--depth', '1', '--branch', EMSDK_VERSION, EMSDK_REPOSITORY, EMSDK_FOLDER], { cwd: REPOSITORY_ROOT });
    }
    const commit = run('git', ['rev-parse', 'HEAD'], { cwd: EMSDK_FOLDER }).trim();
    if (commit !== EMSDK_COMMIT) {
        throw new Error(`${EMSDK_FOLDER} is at ${commit}, expected emsdk ${EMSDK_VERSION} (${EMSDK_COMMIT}). Delete the folder and run again.`);
    }
    if (installedEmscriptenVersion() !== EMSDK_VERSION || !existsSync(join(EMSDK_FOLDER, '.emscripten'))) {
        runEmsdk(['install', EMSDK_VERSION]);
        runEmsdk(['activate', EMSDK_VERSION]);
    }
}

/** The Python that runs em++: emsdk's own on Windows, the system's elsewhere (as emsdk itself does). */
function emsdkPython(): string {
    if (!IS_WINDOWS) return 'python3';
    const pythonFolder = join(EMSDK_FOLDER, 'python');
    const versions = existsSync(pythonFolder) ? readdirSync(pythonFolder) : [];
    if (versions.length !== 1) throw new Error(`expected one Python under ${pythonFolder}, found ${versions.length}`);
    return join(pythonFolder, versions[0], 'python.exe');
}

/** Runs em++ from tools/emsdk in `cwd`, with only that emsdk's configuration and cache. */
function runEmxx(args: string[], cwd: string): string {
    const environment = cleanEnvironment();
    environment.EM_CONFIG = join(EMSDK_FOLDER, '.emscripten');
    environment.EM_CACHE = join(EMSDK_FOLDER, 'upstream', 'emscripten', 'cache');
    environment.EMSDK_QUIET = '1';
    return run(emsdkPython(), [join(EMSDK_FOLDER, 'upstream', 'emscripten', 'em++.py'), ...args], { cwd, env: environment });
}

async function downloadHeaders(folder: string): Promise<void> {
    for (const [file, expected] of Object.entries(TINYBVH_FILES)) {
        const url = `https://raw.githubusercontent.com/jbikker/tinybvh/${TINYBVH_COMMIT}/${file}`;
        const response = await fetch(url);
        if (!response.ok) throw new Error(`could not download ${url}: HTTP ${response.status}`);
        const bytes = Buffer.from(await response.arrayBuffer());
        const actual = sha256(bytes);
        if (actual !== expected) throw new Error(`${file} has SHA-256 ${actual}, expected ${expected}`);
        writeFileSync(join(folder, file), bytes);
    }
    // The committed copy of tinybvh's licence must be the one this build uses.
    const committedLicense = readFileSync(LICENSE_FILES[0]);
    if (!committedLicense.equals(readFileSync(join(folder, 'LICENSE')))) {
        throw new Error(`${LICENSE_FILES[0]} differs from tinybvh's LICENSE at ${TINYBVH_COMMIT}`);
    }
}

/**
 * The licences as one `/*! ... *\/` comment. Bundlers keep `/*!` comments, so the
 * notices reach every file the WASM ends up in.
 */
function licenseComment(): string {
    const texts = LICENSE_FILES.map((file) => readFileSync(file, 'utf8').trimEnd().replace(/\*\//g, '* /'));
    const lines = [
        `The WebAssembly below is tinybvh (${TINYBVH_REPOSITORY}) plus a little of Emscripten's runtime.`,
        'Their licences follow: tinybvh first, then the runtime.',
        '',
        ...texts.join('\n\n').split('\n'),
    ];
    return ['/*!', ...lines.map((line) => (line ? ` * ${line}` : ' *')), ' */'].join('\n');
}

function generatedModule(bytes: Buffer, hash: string): string {
    return [
        licenseComment(),
        '',
        '// Generated by packages/bvh/native/build-wasm.ts (`pnpm bvh:wasm`). Do not edit; rebuild instead.',
        `// tinybvh ${TINYBVH_REPOSITORY} at ${TINYBVH_COMMIT} (MIT), compiled with Emscripten ${EMSDK_VERSION}:`,
        `//   em++ ${BUILD_FLAGS.join(' ')} tinybvh-wasm.cpp`,
        `// ${bytes.length} bytes, SHA-256 ${hash}.`,
        '',
        `export const TINYBVH_WASM_SHA256 = '${hash}';`,
        '',
        'export const TINYBVH_WASM_BASE64 =',
        `    '${bytes.toString('base64')}';`,
        '',
    ].join('\n');
}

/** The hash recorded in the committed module, or null if there is none yet. */
function committedHash(): string | null {
    if (!existsSync(OUTPUT_MODULE)) return null;
    return /TINYBVH_WASM_SHA256 = '([0-9a-f]{64})'/.exec(readFileSync(OUTPUT_MODULE, 'utf8'))?.[1] ?? null;
}

async function main(): Promise<void> {
    const { values } = parseArgs({ args: process.argv.slice(2), options: { check: { type: 'boolean' }, 'keep-headers': { type: 'boolean' } } });
    ensureEmsdk();

    const buildFolder = mkdtempSync(join(tmpdir(), 'tinybvh-'));
    try {
        const version = runEmxx(['--version'], buildFolder).split('\n')[0];
        if (!version.includes(` ${EMSDK_VERSION} `)) throw new Error(`expected Emscripten ${EMSDK_VERSION}, found: ${version}`);
        await downloadHeaders(buildFolder);
        copyFileSync(SOURCE, join(buildFolder, 'tinybvh-wasm.cpp'));
        // Relative paths only, run inside the build folder: __FILE__ and anything else
        // the compiler records stay the same on every machine and in every folder.
        const args = [...BUILD_FLAGS, '-I.', 'tinybvh-wasm.cpp', '-o', 'tinybvh.wasm'];
        console.log(`em++ ${args.join(' ')}`);
        runEmxx(args, buildFolder);
        const bytes = readFileSync(join(buildFolder, 'tinybvh.wasm'));
        const hash = sha256(bytes);
        if (values.check) {
            const expected = committedHash();
            if (hash !== expected) throw new Error(`rebuilt .wasm has SHA-256 ${hash}; the committed one has ${expected}`);
            console.log(`rebuilt .wasm matches the committed one: ${bytes.length} bytes, SHA-256 ${hash}`);
        } else {
            writeFileSync(OUTPUT_MODULE, generatedModule(bytes, hash));
            console.log(`wrote ${OUTPUT_MODULE}: ${bytes.length} bytes of WASM, SHA-256 ${hash}`);
        }
    } finally {
        if (values['keep-headers']) console.log(`build folder kept: ${buildFolder}`);
        else rmSync(buildFolder, { recursive: true, force: true });
    }
}

if (process.argv[1] && resolve(process.argv[1]) === resolve(import.meta.filename)) {
    main().catch((error: unknown) => {
        console.error(error instanceof Error ? error.message : error);
        process.exit(1);
    });
}
