/// <reference types="node" />
/**
 * Builds tinybvh-wasm.cpp with a pinned tinybvh commit into a .wasm, and writes
 * it base64-encoded into `packages/bvh/src/tinybvh-wasm.ts` (with its SHA-256,
 * the commit and the flags), which is what the package loads and what is committed.
 *
 *   node packages/bvh/native/build-wasm.ts [--emcc <command>] [--keep-headers]
 *
 * What it does, so the committed .wasm can be checked:
 * 1. Downloads tinybvh's headers at TINYBVH_COMMIT into a temporary folder and
 *    checks each against its SHA-256 below. A mismatch stops the build.
 * 2. Checks the compiler is Emscripten EMCC_VERSION (the version the committed
 *    .wasm was made with), then compiles with BUILD_FLAGS.
 *
 * Emscripten is found as `emcc` on the PATH. On Windows without one, the
 * script runs `emcc` inside WSL (`wsl bash -lc ...`), which is how the
 * committed file was built (emsdk 3.1.57 in WSL Ubuntu). Nothing is installed.
 */

import { execFileSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { mkdtempSync, readFileSync, rmSync, statSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { parseArgs } from 'node:util';

export const TINYBVH_REPOSITORY = 'https://github.com/jbikker/tinybvh';
/** tinybvh 1.9.1, main on 2026-10-12. */
export const TINYBVH_COMMIT = 'c9be3103cc9693b456b0a0353927c11cac7d710f';
export const EMCC_VERSION = '3.1.57';

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
 * - `-DNO_THREADED_BUILDS`: single-threaded (no pthreads, so no cross-origin isolation needed). One worker builds.
 * - `-msimd128 -msse4.2`: WASM SIMD, through Emscripten's SSE headers, for tinybvh's SSE paths.
 * - `-sSTANDALONE_WASM --no-entry`: a plain .wasm with no JS glue; src/tinybvh.ts instantiates it.
 * - `-sALLOW_MEMORY_GROWTH -sMAXIMUM_MEMORY=4GB`: big meshes need hundreds of MB.
 * - `-fno-exceptions`: tinybvh does not use them.
 */
export const BUILD_FLAGS = [
    '-O3',
    '-std=c++20',
    '-DNO_THREADED_BUILDS',
    '-msimd128',
    '-msse4.2',
    '-fno-exceptions',
    '-sSTANDALONE_WASM',
    '--no-entry',
    '-sALLOW_MEMORY_GROWTH=1',
    '-sMAXIMUM_MEMORY=4GB',
    '-sINITIAL_MEMORY=64MB',
];

const PACKAGE_ROOT = resolve(import.meta.dirname, '..');
const SOURCE = join(PACKAGE_ROOT, 'native', 'tinybvh-wasm.cpp');
const OUTPUT_MODULE = join(PACKAGE_ROOT, 'src', 'tinybvh-wasm.ts');

function sha256(bytes: Buffer): string {
    return createHash('sha256').update(bytes).digest('hex');
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
}

/** Windows path to a WSL path: C:\a\b -> /mnt/c/a/b. */
function toWslPath(path: string): string {
    return path.replace(/^([A-Za-z]):/, (_, drive: string) => `/mnt/${drive.toLowerCase()}`).replace(/\\/g, '/');
}

function hasCommand(command: string): boolean {
    try {
        execFileSync(command, ['--version'], { stdio: 'ignore', shell: process.platform === 'win32' });
        return true;
    } catch {
        return false;
    }
}

/** Runs emcc with `args` (paths already in the form the compiler sees) and returns its output. */
function makeEmcc(emccOption: string | undefined): { run(args: string[]): string; path(native: string): string } {
    const useWsl = emccOption === undefined && !hasCommand('emcc') && process.platform === 'win32';
    const command = emccOption ?? 'emcc';
    if (useWsl) {
        const quote = (value: string) => `'${value.replace(/'/g, `'\\''`)}'`;
        return {
            run: (args) => execFileSync('wsl', ['bash', '-lc', `EMSDK_QUIET=1 ${command} ${args.map(quote).join(' ')}`], { encoding: 'utf8' }),
            path: toWslPath,
        };
    }
    return {
        run: (args) => execFileSync(command, args, { encoding: 'utf8', shell: process.platform === 'win32' }),
        path: (native) => native,
    };
}

function generatedModule(bytes: Buffer, hash: string): string {
    return [
        '// Generated by packages/bvh/native/build-wasm.ts. Do not edit; rebuild instead.',
        `// tinybvh ${TINYBVH_REPOSITORY} at ${TINYBVH_COMMIT} (MIT), compiled with Emscripten ${EMCC_VERSION}:`,
        `//   emcc ${BUILD_FLAGS.join(' ')} native/tinybvh-wasm.cpp`,
        `// ${bytes.length} bytes, SHA-256 ${hash}.`,
        '',
        `export const TINYBVH_WASM_SHA256 = '${hash}';`,
        '',
        'export const TINYBVH_WASM_BASE64 =',
        `    '${bytes.toString('base64')}';`,
        '',
    ].join('\n');
}

async function main(): Promise<void> {
    const { values } = parseArgs({ args: process.argv.slice(2), options: { emcc: { type: 'string' }, 'keep-headers': { type: 'boolean' } } });
    const emcc = makeEmcc(values.emcc);
    const version = emcc.run(['--version']).split('\n')[0];
    if (!version.includes(` ${EMCC_VERSION} `)) {
        throw new Error(`expected Emscripten ${EMCC_VERSION}, found: ${version}`);
    }

    const headers = mkdtempSync(join(tmpdir(), 'tinybvh-'));
    try {
        await downloadHeaders(headers);
        const wasmFile = join(headers, 'tinybvh.wasm');
        // tinybvh puts __FILE__ into its error messages; the prefix maps keep the build
        // folder's random name (and this machine's paths) out of the .wasm, so it is reproducible.
        const args = [
            ...BUILD_FLAGS,
            `-ffile-prefix-map=${emcc.path(headers)}=tinybvh`,
            `-ffile-prefix-map=${emcc.path(join(PACKAGE_ROOT, 'native'))}=native`,
            `-I${emcc.path(headers)}`,
            emcc.path(SOURCE),
            '-o',
            emcc.path(wasmFile),
        ];
        console.log(`emcc ${args.join(' ')}`);
        emcc.run(args);
        const bytes = readFileSync(wasmFile);
        const hash = sha256(bytes);
        writeFileSync(OUTPUT_MODULE, generatedModule(bytes, hash));
        console.log(`wrote ${OUTPUT_MODULE}: ${statSync(wasmFile).size} bytes of WASM, SHA-256 ${hash}`);
    } finally {
        if (values['keep-headers']) console.log(`headers kept in ${headers}`);
        else rmSync(headers, { recursive: true, force: true });
    }
}

if (process.argv[1] && resolve(process.argv[1]) === resolve(import.meta.filename)) {
    main().catch((error: unknown) => {
        console.error(error instanceof Error ? error.message : error);
        process.exit(1);
    });
}
