/**
 * Downloads, verifies, unpacks and converts the external test assets.
 *
 *   pnpm assets:fetch                 every asset in tools/assets/sources.json
 *   pnpm assets:fetch --only <id>     one asset (repeat the flag, or use a,b)
 *   pnpm assets:fetch --force         redo everything even if outputs look fine
 *   pnpm assets:fetch --sources <p>   read a different sources file
 *
 * Files land in assets/downloads/<id>/ (gitignored). For each asset:
 *   1. If every output exists and matches its sha256, do nothing.
 *   2. Otherwise download each file (streamed, with progress) unless it is
 *      already there with the right hash, and check its size and sha256.
 *      `download` is one object or an array (a glTF with its .bin and
 *      textures; `file` may name subfolders). Files with a known sha256 go
 *      through a cache in assets/downloads/.cache/<sha256>, so assets that
 *      share a download (the two Bistro halves) fetch it once.
 *   3. Unpack zips: `tar` (bsdtar, ships with Windows 10+) on Windows,
 *      PowerShell `Expand-Archive` if tar is missing, `unzip` elsewhere.
 *      No npm dependency.
 *   4. Run the convert step, if any. Blender is found from the BLENDER
 *      environment variable, then PATH, then the default install folders.
 *      The script runs as `blender --background --factory-startup --python
 *      <script> -- <args>` (factory settings: the user's add-ons can hang Blender)
 *      with the asset folder as the working directory, and `{dir}`, `{id}` and
 *      `{repo}` in args replaced by the absolute asset folder, the asset id and
 *      the repository root.
 *   5. Check every output's sha256.
 *
 * An asset with `"skip": true` and a `skipReason` is reported and left alone.
 *
 * sources.json is checked strictly before anything is downloaded: every
 * non-skipped asset needs `redistributable`, a license on the allow list (CC0
 * or CC-BY), and sha256 plus bytes for every download and every output. To add
 * an asset, get the numbers with `node tools/assets/fetch.ts --hash <file>`.
 * Downloads give up after 60 s without an answer or without new data.
 * Everything here is one file so it runs under plain Node type
 * stripping with no imports from the repo.
 */

import { spawn } from 'node:child_process';
import { createHash } from 'node:crypto';
import { createReadStream, createWriteStream } from 'node:fs';
import { copyFile, link, mkdir, open, readdir, readFile, rename, rm, stat } from 'node:fs/promises';
import { delimiter, dirname, join, resolve } from 'node:path';
import { Readable, Transform } from 'node:stream';
import { pipeline } from 'node:stream/promises';
import { fileURLToPath } from 'node:url';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface AssetDownload {
    url: string;
    /** Path inside the asset folder; may include subfolders, like "textures/bark.jpg". */
    file: string;
    bytes?: number;
    sha256?: string;
}

export interface AssetConvert {
    tool: 'blender';
    /** Repo-relative path to a Python script. */
    script: string;
    args: string[];
    /** Major.minor of the Blender that makes the recorded outputs, like "4.5". Other versions can crash or give different bytes. */
    blenderVersion?: string;
}

export interface AssetOutput {
    /** Relative to assets/downloads/<id>/. */
    path: string;
    sha256?: string;
    bytes?: number;
    /**
     * Set to false when the converter is known not to write the same bits every
     * run (Blender's glTF exporter changes a couple of bytes in some vertex
     * buffers between runs on Bistro and the tree). The file must then be a
     * .glb, and instead of the whole-file sha256 it is checked structurally:
     * the exact size, the sha256 of its JSON chunk (the scene description,
     * which is identical between runs) and the exact length of its BIN chunk.
     * The whole-file sha256 is only logged. Leave this out for everything else:
     * the sha256 is how we know a conversion is the same.
     */
    reproducible?: boolean;
    /** sha256 of the .glb's JSON chunk, bytes exactly as stored. Required with `reproducible: false`. */
    jsonSha256?: string;
    /** Length in bytes of the .glb's BIN chunk. Required with `reproducible: false`. */
    binBytes?: number;
}

export interface AssetEntry {
    id: string;
    title: string;
    author: string;
    sourceUrl: string;
    license: string;
    licenseUrl?: string;
    attribution: string;
    redistributable: boolean;
    /** One file, or several (a glTF with its .bin and textures). `extract` needs exactly one. */
    download: AssetDownload | AssetDownload[];
    extract: 'zip' | null;
    convert: AssetConvert | null;
    outputs: AssetOutput[];
    /** Leave this asset out for now (for example while a license question is open). */
    skip?: boolean;
    skipReason?: string;
}

export interface SourcesFile {
    assets: AssetEntry[];
}

export interface CliOptions {
    only: string[];
    force: boolean;
    sourcesPath: string | null;
    /** A file to print sha256 and size for (to fill in sources.json), instead of fetching. */
    hash: string | null;
    /** Check every asset's existing outputs against sources.json; download and convert nothing. */
    verifyOnly: boolean;
    help: boolean;
}

// ---------------------------------------------------------------------------
// Pure helpers (unit-tested)
// ---------------------------------------------------------------------------

const KEBAB_CASE = /^[a-z0-9]+(-[a-z0-9]+)*$/;
const SHA256_HEX = /^[0-9a-f]{64}$/i;

/**
 * Licenses we are willing to download and show: CC0 and plain CC-BY (any
 * version). Share-alike, non-commercial and custom licenses are not on the
 * list; adding one is a decision for Dennis (see docs/DECISIONS.md).
 */
export function isAllowedLicense(license: string): boolean {
    return license === 'CC0-1.0' || /^CC-BY-\d\.\d$/.test(license);
}

function isRecord(value: unknown): value is Record<string, unknown> {
    return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function isNonEmptyString(value: unknown): value is string {
    return typeof value === 'string' && value.trim().length > 0;
}

/** A path that stays inside its folder: relative, no drive letter, no "..". */
export function isSafeRelativePath(path: string): boolean {
    if (path.length === 0 || path.startsWith('/') || path.startsWith('\\') || /^[a-zA-Z]:/.test(path)) return false;
    return !path.split(/[\\/]/).includes('..');
}

/** The asset's downloads as a list, whichever form sources.json used. */
export function downloadsOf(asset: Pick<AssetEntry, 'download'>): AssetDownload[] {
    return Array.isArray(asset.download) ? asset.download : [asset.download];
}

/** Returns one readable line per problem; an empty list means the file is usable. */
export function validateSources(value: unknown): string[] {
    const errors: string[] = [];
    if (!isRecord(value) || !Array.isArray(value.assets)) {
        return ['sources.json must be an object with an "assets" array'];
    }
    const seenIds = new Set<string>();
    value.assets.forEach((entry: unknown, index: number) => {
        const where = isRecord(entry) && typeof entry.id === 'string' ? `assets[${index}] (${entry.id})` : `assets[${index}]`;
        if (!isRecord(entry)) {
            errors.push(`${where}: must be an object`);
            return;
        }
        if (!isNonEmptyString(entry.id) || !KEBAB_CASE.test(entry.id)) errors.push(`${where}.id: must be kebab-case`);
        else if (seenIds.has(entry.id)) errors.push(`${where}.id: duplicate id`);
        else seenIds.add(entry.id);

        // A skipped asset only has to say why; its other fields may still be unfinished.
        if (entry.skip === true) {
            if (!isNonEmptyString(entry.skipReason)) errors.push(`${where}.skipReason: say why the asset is skipped`);
            return;
        }

        for (const field of ['title', 'author', 'sourceUrl', 'license', 'attribution'] as const) {
            if (!isNonEmptyString(entry[field])) errors.push(`${where}.${field}: must be a non-empty string`);
        }
        if (typeof entry.redistributable !== 'boolean') errors.push(`${where}.redistributable: must be present and true or false`);
        if (isNonEmptyString(entry.license) && !isAllowedLicense(entry.license)) {
            errors.push(`${where}.license: "${entry.license}" is not on the allow list (CC0-1.0, CC-BY-<version>). Ask Dennis before adding another license.`);
        }

        const downloads = Array.isArray(entry.download) ? entry.download : [entry.download];
        if (Array.isArray(entry.download) && entry.download.length === 0) {
            errors.push(`${where}.download: must not be an empty array`);
        }
        const seenFiles = new Set<string>();
        downloads.forEach((download: unknown, downloadIndex: number) => {
            const downloadWhere = Array.isArray(entry.download) ? `${where}.download[${downloadIndex}]` : `${where}.download`;
            if (!isRecord(download)) {
                errors.push(`${downloadWhere}: must be an object with url and file`);
                return;
            }
            if (!isNonEmptyString(download.url) || !/^https?:\/\//.test(download.url)) {
                errors.push(`${downloadWhere}.url: must be an http(s) URL`);
            }
            if (!isNonEmptyString(download.file) || !isSafeRelativePath(download.file)) {
                errors.push(`${downloadWhere}.file: must be a relative path inside the asset folder (subfolders are fine)`);
            } else if (seenFiles.has(download.file)) {
                errors.push(`${downloadWhere}.file: "${download.file}" is listed twice`);
            } else {
                seenFiles.add(download.file);
            }
            if (!(typeof download.bytes === 'number' && Number.isInteger(download.bytes) && download.bytes > 0)) {
                errors.push(`${downloadWhere}.bytes: required, a positive whole number`);
            }
            if (!(typeof download.sha256 === 'string' && SHA256_HEX.test(download.sha256))) {
                errors.push(`${downloadWhere}.sha256: required, 64 hex characters`);
            }
        });
        if (downloads.length > 1 && entry.extract !== null) {
            errors.push(`${where}.extract: only works with a single download, but there are ${downloads.length}`);
        }

        if (entry.extract !== null && entry.extract !== 'zip') errors.push(`${where}.extract: must be "zip" or null`);

        const convert = entry.convert;
        if (convert !== null) {
            if (!isRecord(convert) || convert.tool !== 'blender' || !isNonEmptyString(convert.script) || !Array.isArray(convert.args)) {
                errors.push(`${where}.convert: must be null or { tool: "blender", script, args: [] }`);
            } else if (!isSafeRelativePath(convert.script) || !convert.args.every((arg: unknown) => typeof arg === 'string')) {
                errors.push(`${where}.convert: script must be a repo-relative path and args must be strings`);
            } else if (convert.blenderVersion !== undefined && !(typeof convert.blenderVersion === 'string' && /^\d+\.\d+$/.test(convert.blenderVersion))) {
                errors.push(`${where}.convert.blenderVersion: must look like "4.5" (major.minor)`);
            }
        }

        if (!Array.isArray(entry.outputs) || entry.outputs.length === 0) {
            errors.push(`${where}.outputs: must list at least one output file`);
        } else {
            entry.outputs.forEach((output: unknown, outputIndex: number) => {
                const outputWhere = `${where}.outputs[${outputIndex}]`;
                if (!isRecord(output) || !isNonEmptyString(output.path) || !isSafeRelativePath(output.path)) {
                    errors.push(`${outputWhere}.path: must be a relative path inside the asset folder`);
                } else {
                    if (!(typeof output.sha256 === 'string' && SHA256_HEX.test(output.sha256))) {
                        errors.push(`${outputWhere}.sha256: required, 64 hex characters`);
                    }
                    if (!(typeof output.bytes === 'number' && Number.isInteger(output.bytes) && output.bytes > 0)) {
                        errors.push(`${outputWhere}.bytes: required, a positive whole number`);
                    }
                    if (output.reproducible !== undefined && typeof output.reproducible !== 'boolean') {
                        errors.push(`${outputWhere}.reproducible: must be true or false when present`);
                    }
                    if (output.reproducible === false) {
                        if (!output.path.toLowerCase().endsWith('.glb')) {
                            errors.push(`${outputWhere}.reproducible: false is only supported for .glb files`);
                        }
                        if (!(typeof output.jsonSha256 === 'string' && SHA256_HEX.test(output.jsonSha256))) {
                            errors.push(`${outputWhere}.jsonSha256: required with reproducible false, 64 hex characters`);
                        }
                        if (!(typeof output.binBytes === 'number' && Number.isInteger(output.binBytes) && output.binBytes > 0)) {
                            errors.push(`${outputWhere}.binBytes: required with reproducible false, a positive whole number`);
                        }
                    }
                }
            });
        }
    });
    return errors;
}

export function parseArgs(argv: readonly string[]): CliOptions {
    const options: CliOptions = { only: [], force: false, sourcesPath: null, hash: null, verifyOnly: false, help: false };
    for (let i = 0; i < argv.length; i++) {
        const arg = argv[i];
        const takeValue = (): string => {
            const value = argv[++i];
            if (value === undefined || value.startsWith('--')) throw new Error(`${arg} needs a value`);
            return value;
        };
        if (arg === '--only') options.only.push(...takeValue().split(',').filter(Boolean));
        else if (arg === '--force') options.force = true;
        else if (arg === '--sources') options.sourcesPath = takeValue();
        else if (arg === '--hash') options.hash = takeValue();
        else if (arg === '--verify-only') options.verifyOnly = true;
        else if (arg === '--help' || arg === '-h') options.help = true;
        else throw new Error(`Unknown argument "${arg}". Use --help to see the options.`);
    }
    return options;
}

/** Picks the assets named by `--only`, or all of them. Unknown ids are an error that lists the real ones. */
export function selectAssets(assets: readonly AssetEntry[], only: readonly string[]): AssetEntry[] {
    if (only.length === 0) return [...assets];
    const unknown = only.filter((id) => !assets.some((asset) => asset.id === id));
    if (unknown.length > 0) {
        const known = assets.map((asset) => asset.id).join(', ') || '(none listed yet)';
        throw new Error(`Unknown asset id: ${unknown.join(', ')}. Known ids: ${known}`);
    }
    return assets.filter((asset) => only.includes(asset.id));
}

/** Replaces `{dir}`, `{id}` and `{repo}` in convert arguments. */
export function substitutePlaceholders(args: readonly string[], values: { dir: string; id: string; repo: string }): string[] {
    return args.map((arg) => arg.replaceAll('{dir}', values.dir).replaceAll('{id}', values.id).replaceAll('{repo}', values.repo));
}

/** Orders folder names like "Blender 5.1" and "Blender 4.5" newest first. */
export function sortBlenderFolders(names: readonly string[]): string[] {
    const version = (name: string): number[] => (name.match(/\d+/g) ?? []).map(Number);
    return names
        .filter((name) => /blender/i.test(name))
        .sort((a, b) => {
            const [va, vb] = [version(a), version(b)];
            for (let i = 0; i < Math.max(va.length, vb.length); i++) {
                const difference = (vb[i] ?? 0) - (va[i] ?? 0);
                if (difference !== 0) return difference;
            }
            return 0;
        });
}

export function formatBytes(bytes: number): string {
    const units = ['B', 'KB', 'MB', 'GB'];
    let value = bytes;
    let unit = 0;
    while (value >= 1024 && unit < units.length - 1) {
        value /= 1024;
        unit++;
    }
    return `${value.toFixed(unit === 0 ? 0 : 1)} ${units[unit]}`;
}

export function sameHash(a: string, b: string): boolean {
    return a.toLowerCase() === b.toLowerCase();
}

// ---------------------------------------------------------------------------
// File and process helpers
// ---------------------------------------------------------------------------

const repoRoot = resolve(dirname(fileURLToPath(import.meta.url)), '..', '..');
const defaultSourcesPath = join(repoRoot, 'tools', 'assets', 'sources.json');
const downloadsRoot = join(repoRoot, 'assets', 'downloads');

export async function sha256OfFile(path: string): Promise<string> {
    const hash = createHash('sha256');
    for await (const chunk of createReadStream(path)) hash.update(chunk as Buffer);
    return hash.digest('hex');
}

async function fileExists(path: string): Promise<boolean> {
    try {
        return (await stat(path)).isFile();
    } catch {
        return false;
    }
}

/** Runs a program with output going straight to the terminal. Resolves with its exit code. */
function run(command: string, args: string[], options: { cwd?: string; env?: NodeJS.ProcessEnv } = {}): Promise<number> {
    return new Promise((resolvePromise, reject) => {
        const child = spawn(command, args, { stdio: 'inherit', ...options });
        child.on('error', reject);
        child.on('close', (code) => resolvePromise(code ?? 1));
    });
}

/** "Blender 4.5.0 LTS" or a full `--version` printout -> "4.5.0". Null when it is not Blender. */
export function parseBlenderVersion(output: string): string | null {
    return output.match(/Blender\s+(\d+\.\d+(?:\.\d+)?)/)?.[1] ?? null;
}

/** True when `actual` ("4.5.0") is in the wanted major.minor ("4.5"). */
export function versionMatches(actual: string, wanted: string): boolean {
    const [wantedMajor, wantedMinor] = wanted.split('.');
    const [major, minor] = actual.split('.');
    return major === wantedMajor && minor === wantedMinor;
}

/** Runs `blender --version` and returns the version, or null if it will not run. */
async function readBlenderVersion(executable: string): Promise<string | null> {
    return new Promise((resolvePromise) => {
        let output = '';
        const child = spawn(executable, ['--version'], { stdio: ['ignore', 'pipe', 'ignore'] });
        child.stdout.on('data', (chunk: Buffer) => (output += chunk.toString()));
        child.on('error', () => resolvePromise(null));
        child.on('close', () => resolvePromise(parseBlenderVersion(output)));
    });
}

/**
 * Every Blender executable we can find, in the order to try them: the BLENDER
 * variable, then PATH, then the default install folders (all versions, newest
 * first). Only files that exist are returned.
 */
export async function blenderCandidates(
    env: NodeJS.ProcessEnv = process.env,
    platform: NodeJS.Platform = process.platform,
): Promise<string[]> {
    const found: string[] = [];
    const add = async (candidate: string): Promise<void> => {
        if (!found.includes(candidate) && (await fileExists(candidate))) found.push(candidate);
    };

    if (env.BLENDER) await add(env.BLENDER);

    const executable = platform === 'win32' ? 'blender.exe' : 'blender';
    for (const directory of (env.PATH ?? env.Path ?? '').split(delimiter).filter(Boolean)) {
        await add(join(directory, executable));
    }

    if (platform === 'darwin') {
        await add('/Applications/Blender.app/Contents/MacOS/Blender');
    } else if (platform === 'win32') {
        for (const programFiles of [env.ProgramFiles ?? 'C:\\Program Files', env['ProgramFiles(x86)']]) {
            if (!programFiles) continue;
            const root = join(programFiles, 'Blender Foundation');
            try {
                for (const folder of sortBlenderFolders(await readdir(root))) await add(join(root, folder, executable));
            } catch {
                // No Blender Foundation folder here.
            }
        }
    } else {
        for (const candidate of ['/usr/bin/blender', '/usr/local/bin/blender', '/snap/bin/blender']) await add(candidate);
    }
    return found;
}

export interface BlenderChoice {
    path: string;
    version: string;
}

/**
 * Finds a Blender of the wanted major.minor (any version when `wanted` is
 * undefined). Each candidate is asked for its version rather than trusted by
 * its name or by the BLENDER variable. Throws a plain message listing what was found.
 */
export async function findBlender(
    wanted: string | undefined,
    options: {
        env?: NodeJS.ProcessEnv;
        platform?: NodeJS.Platform;
        readVersion?: (executable: string) => Promise<string | null>;
    } = {},
): Promise<BlenderChoice> {
    const readVersion = options.readVersion ?? readBlenderVersion;
    const candidates = await blenderCandidates(options.env, options.platform);
    const seen: string[] = [];
    for (const path of candidates) {
        const version = await readVersion(path);
        seen.push(`${path} (${version ? `version ${version}` : 'would not run'})`);
        if (version && (!wanted || versionMatches(version, wanted))) return { path, version };
    }
    const need = wanted ? `Blender ${wanted}` : 'Blender';
    const found = seen.length > 0 ? `Found, but not usable: ${seen.join('; ')}.` : 'No Blender was found at all.';
    throw new Error(
        `${need} is needed for this conversion. ${found} Looked in the BLENDER variable, PATH and the default install folders. ` +
            `Install ${need}, or set BLENDER to its full path.`,
    );
}

// ---------------------------------------------------------------------------
// Steps
// ---------------------------------------------------------------------------

type Log = (message: string) => void;

/** The two facts about a .glb that stay the same when a converter's float output wobbles. */
export interface GlbStructure {
    /** sha256 of the JSON chunk's bytes, exactly as stored (including its padding). */
    jsonSha256: string;
    /** Length of the BIN chunk's data, as the chunk header says. */
    binBytes: number;
}

/** Reads a .glb's header and chunk headers and hashes only the JSON chunk, so a 270 MB file costs a few kilobytes of I/O. */
export async function readGlbStructure(path: string): Promise<GlbStructure> {
    const file = await open(path, 'r');
    try {
        const size = (await file.stat()).size;
        const header = Buffer.alloc(20);
        if (size < header.length || (await file.read(header, 0, header.length, 0)).bytesRead < header.length) {
            throw new Error('too short to be a .glb file');
        }
        if (header.readUInt32LE(0) !== 0x46546c67) throw new Error('not a .glb file (bad magic)');
        if (header.readUInt32LE(8) !== size) throw new Error(`the .glb header says ${header.readUInt32LE(8)} bytes but the file has ${size}`);
        const jsonLength = header.readUInt32LE(12);
        if (header.readUInt32LE(16) !== 0x4e4f534a) throw new Error('the first chunk is not JSON');
        const json = Buffer.alloc(jsonLength);
        if ((await file.read(json, 0, jsonLength, 20)).bytesRead < jsonLength) throw new Error('the JSON chunk is cut off');
        const binHeader = Buffer.alloc(8);
        if ((await file.read(binHeader, 0, 8, 20 + jsonLength)).bytesRead < 8) throw new Error('there is no BIN chunk');
        if (binHeader.readUInt32LE(4) !== 0x004e4942) throw new Error('the second chunk is not BIN');
        return { jsonSha256: createHash('sha256').update(json).digest('hex'), binBytes: binHeader.readUInt32LE(0) };
    } finally {
        await file.close();
    }
}

/** Problems with a not-bit-reproducible .glb output, or an empty list when its structure matches sources.json. */
export async function checkGlbStructure(path: string, output: AssetOutput): Promise<string[]> {
    let actual: GlbStructure;
    try {
        actual = await readGlbStructure(path);
    } catch (error) {
        return [`cannot read the .glb structure: ${error instanceof Error ? error.message : String(error)}`];
    }
    const problems: string[] = [];
    if (output.jsonSha256 && !sameHash(actual.jsonSha256, output.jsonSha256)) {
        problems.push(`the JSON chunk differs (expected sha256 ${output.jsonSha256}, got ${actual.jsonSha256}); the scene description changed, which is not float noise`);
    }
    if (output.binBytes !== undefined && actual.binBytes !== output.binBytes) {
        problems.push(`the BIN chunk is ${actual.binBytes} bytes but sources.json says ${output.binBytes}`);
    }
    return problems;
}

/** What to add to a mismatch message for a file a conversion makes. */
function conversionCause(convert: AssetConvert | null | undefined): string {
    if (!convert) return '';
    const needs = convert.blenderVersion ? ` (this one needs ${convert.blenderVersion})` : '';
    return `. This file is made by a conversion, so a different Blender version${needs} or conversion script is the likely cause`;
}

/** Checks every output. Returns what is wrong, or an empty list when all outputs are present and verified. */
export async function checkOutputs(
    assetDirectory: string,
    outputs: readonly AssetOutput[],
    log: Log,
    convert?: AssetConvert | null,
): Promise<{ problems: string[]; unverified: number }> {
    const problems: string[] = [];
    let unverified = 0;
    for (const output of outputs) {
        const path = join(assetDirectory, output.path);
        if (!(await fileExists(path))) {
            problems.push(`missing: ${output.path}`);
            continue;
        }
        if (output.bytes !== undefined) {
            const size = (await stat(path)).size;
            if (size !== output.bytes) {
                problems.push(`size differs: ${output.path} (expected ${output.bytes} bytes, got ${size})${conversionCause(convert)}`);
                continue;
            }
        }
        const actual = await sha256OfFile(path);
        if (!output.sha256) {
            unverified++;
            log(`  warning: ${output.path} has no sha256 in sources.json, so it is unverified. Computed: ${actual}`);
        } else if (!sameHash(actual, output.sha256)) {
            if (output.reproducible === false) {
                const structural = await checkGlbStructure(path, output);
                if (structural.length > 0) {
                    problems.push(...structural.map((problem) => `${output.path}: ${problem}${conversionCause(convert)}`));
                } else {
                    log(`  note: ${output.path} is not bit-reproducible; its size, JSON chunk and BIN chunk length match, whole-file sha256 is ${actual} (information only)`);
                }
                continue;
            }
            problems.push(`sha256 differs: ${output.path} (expected ${output.sha256}, got ${actual})${conversionCause(convert)}`);
        }
    }
    return { problems, unverified };
}

function makeProgress(label: string, total: number | undefined): { update: (received: number) => void; finish: () => void } {
    const start = Date.now();
    let lastPrint = 0;
    const interactive = process.stdout.isTTY === true;
    const render = (received: number, done: boolean): string => {
        const seconds = Math.max((Date.now() - start) / 1000, 0.001);
        const speed = `${formatBytes(received / seconds)}/s`;
        const amount = total ? `${formatBytes(received)} / ${formatBytes(total)} (${Math.floor((received / total) * 100)}%)` : formatBytes(received);
        return `  ${done ? 'downloaded' : 'downloading'} ${label}: ${amount} at ${speed}`;
    };
    return {
        update(received) {
            const now = Date.now();
            // A line a second on a terminal, one every 5 seconds in a log file.
            if (now - lastPrint < (interactive ? 250 : 5000)) return;
            lastPrint = now;
            if (interactive) process.stdout.write(`\r${render(received, false)}   `);
            else console.log(render(received, false));
        },
        finish() {
            if (interactive) process.stdout.write('\r');
        },
    };
}

/** How long to wait for a server to start answering, and for more data once it has. */
export interface DownloadTimeouts {
    connectMs: number;
    stallMs: number;
}
export const DEFAULT_DOWNLOAD_TIMEOUTS: DownloadTimeouts = { connectMs: 60_000, stallMs: 60_000 };

/**
 * Streams a URL to `destination`, checks size and hash, and returns the sha256
 * it computed. Gives up with a plain message if the server does not answer
 * within `connectMs`, or goes quiet for `stallMs` mid-download.
 */
export async function downloadFile(
    download: AssetDownload,
    destination: string,
    timeouts: DownloadTimeouts = DEFAULT_DOWNLOAD_TIMEOUTS,
): Promise<string> {
    const controller = new AbortController();
    let timer: ReturnType<typeof setTimeout> | undefined;
    const armTimer = (milliseconds: number, message: string): void => {
        clearTimeout(timer);
        timer = setTimeout(() => controller.abort(new Error(message)), milliseconds);
    };
    const connectMessage = `no answer from ${download.url} within ${timeouts.connectMs / 1000} s`;
    const stallMessage = `download of ${download.file} stalled: no data for ${timeouts.stallMs / 1000} s`;

    armTimer(timeouts.connectMs, connectMessage);
    const partial = `${destination}.part`;
    try {
        const response = await fetch(download.url, {
            redirect: 'follow',
            signal: controller.signal,
            headers: { 'user-agent': 'webgpu-ddgi-asset-fetch (+https://github.com/pmndrs/ddgi)' },
        });
        if (!response.ok || !response.body) {
            throw new Error(`download failed: ${download.url} answered ${response.status} ${response.statusText}`);
        }
        const headerLength = Number(response.headers.get('content-length'));
        const total = download.bytes ?? (Number.isFinite(headerLength) && headerLength > 0 ? headerLength : undefined);

        const hash = createHash('sha256');
        const progress = makeProgress(download.file, total);
        let received = 0;
        armTimer(timeouts.stallMs, stallMessage);
        const counter = new Transform({
            transform(chunk: Buffer, _encoding, callback) {
                received += chunk.length;
                hash.update(chunk);
                progress.update(received);
                armTimer(timeouts.stallMs, stallMessage);
                callback(null, chunk);
            },
        });
        try {
            await pipeline(Readable.fromWeb(response.body as never), counter, createWriteStream(partial), { signal: controller.signal });
        } finally {
            progress.finish();
        }

        if (download.bytes !== undefined && received !== download.bytes) {
            throw new Error(`download is ${received} bytes but sources.json says ${download.bytes}`);
        }
        const actual = hash.digest('hex');
        if (download.sha256 && !sameHash(actual, download.sha256)) {
            throw new Error(`download sha256 is ${actual} but sources.json says ${download.sha256}`);
        }
        await rename(partial, destination);
        return actual;
    } catch (error) {
        await rm(partial, { force: true });
        // An abort from one of our timers carries our message as its reason.
        if (controller.signal.aborted && controller.signal.reason instanceof Error) throw controller.signal.reason;
        throw error;
    } finally {
        clearTimeout(timer);
    }
}

async function extractZip(archive: string, destination: string): Promise<void> {
    if (process.platform === 'win32') {
        const systemTar = join(process.env.SystemRoot ?? 'C:\\Windows', 'System32', 'tar.exe');
        if (await fileExists(systemTar)) {
            // bsdtar reads zip files and is much faster than Expand-Archive on large archives.
            const code = await run(systemTar, ['-xf', archive, '-C', destination]);
            if (code !== 0) throw new Error(`tar exited with code ${code} while extracting ${archive}`);
            return;
        }
        const code = await run(
            'powershell.exe',
            ['-NoProfile', '-NonInteractive', '-Command', 'Expand-Archive -LiteralPath $env:DDGI_ZIP -DestinationPath $env:DDGI_DEST -Force'],
            { env: { ...process.env, DDGI_ZIP: archive, DDGI_DEST: destination } },
        );
        if (code !== 0) throw new Error(`Expand-Archive exited with code ${code} while extracting ${archive}`);
        return;
    }
    let code: number;
    try {
        code = await run('unzip', ['-o', '-q', archive, '-d', destination]);
    } catch {
        throw new Error('The "unzip" command was not found. Install it (for example apt install unzip) and run again.');
    }
    if (code !== 0) throw new Error(`unzip exited with code ${code} while extracting ${archive}`);
}

/** Limits for a conversion process (a Blender run). */
export interface ConversionLimits {
    /** Give up and kill the process after this long. */
    timeoutMs: number;
    /** Once every output is written and verified, how long the process gets to exit by itself. */
    graceMs: number;
    /** How often to look at the outputs. */
    pollMs: number;
}
export const DEFAULT_CONVERSION_LIMITS: ConversionLimits = { timeoutMs: 60 * 60_000, graceMs: 15_000, pollMs: 2_000 };

export type ConversionEnd = 'exited' | 'stopped-after-outputs' | 'timed-out';

/** Stops a process and everything it started. */
async function killProcessTree(child: { pid?: number; kill?: (signal?: NodeJS.Signals) => boolean }): Promise<void> {
    if (child.pid === undefined) return;
    if (process.platform === 'win32') {
        await new Promise<void>((resolvePromise) => {
            const killer = spawn('taskkill', ['/PID', String(child.pid), '/T', '/F'], { stdio: 'ignore' });
            killer.on('error', () => resolvePromise());
            killer.on('close', () => resolvePromise());
        });
    } else {
        try {
            process.kill(-child.pid, 'SIGKILL'); // the child leads its own process group
        } catch {
            child.kill?.('SIGKILL');
        }
    }
}

/** What `runConversionProcess` needs from a child process. Real runs use `spawn`; tests pass a fake. */
export interface ConversionChild {
    pid?: number;
    exitCode: number | null;
    signalCode: string | null;
    on(event: 'error', listener: (error: Error) => void): unknown;
    on(event: 'exit', listener: () => void): unknown;
    on(event: 'close', listener: (code: number | null) => void): unknown;
}

export interface ConversionProcessOptions {
    cwd: string;
    assetDirectory: string;
    outputs: readonly AssetOutput[];
    limits?: ConversionLimits;
    /** For tests: start the process some other way than `spawn`. */
    spawnProcess?: (command: string, args: string[], cwd: string) => ConversionChild;
    /** For tests: stop the process some other way than killing its process tree. */
    killTree?: (child: ConversionChild) => Promise<void>;
}

/**
 * Runs a conversion program (Blender) and watches its outputs.
 *
 * Blender can finish its work, write the file, and then never exit (all of its
 * threads end up waiting at shutdown, using no CPU). So the process is not
 * trusted to exit: when every output has been written fresh (modified after the
 * run began) and matches its recorded size and sha256, it gets `graceMs` to
 * exit and is then killed. A process that neither exits nor finishes its
 * outputs is killed after `timeoutMs`. Its stdin is closed so it can never wait
 * for input.
 *
 * What is reported:
 * - `exited`, with the real exit code, whenever the process ended by itself.
 *   A process that has already exited is never counted as stopped by the guard,
 *   so an error code right after writing the outputs is never hidden.
 * - `stopped-after-outputs` or `timed-out` only when the guard really killed it.
 *
 * Outputs that are present but wrong (right size, wrong hash) also count as
 * finished: the caller reports the mismatch, instead of waiting for a hang. A
 * conversion with no listed outputs is never stopped early, only by the timeout.
 */
export async function runConversionProcess(
    command: string,
    args: string[],
    options: ConversionProcessOptions,
): Promise<{ end: ConversionEnd; code: number | null }> {
    const limits = options.limits ?? DEFAULT_CONVERSION_LIMITS;
    const startedAt = Date.now() - 2_000; // file times can be a little coarse
    const child: ConversionChild =
        options.spawnProcess?.(command, args, options.cwd) ??
        spawn(command, args, {
            cwd: options.cwd,
            stdio: ['ignore', 'inherit', 'inherit'],
            detached: process.platform !== 'win32',
        });
    const killTree = options.killTree ?? ((target: ConversionChild) => killProcessTree(target));

    return new Promise((resolvePromise, reject) => {
        let finished = false;
        let exited = false; // the process ended by itself or was killed: its 'exit' event has fired
        let guardEnd: ConversionEnd | null = null; // set only when the guard is the one killing it
        const settledOutputs = new Set<string>();
        let polling = false;
        let graceTimer: ReturnType<typeof setTimeout> | undefined;

        const done = (end: ConversionEnd, code: number | null): void => {
            if (finished) return;
            finished = true;
            clearInterval(poller);
            clearTimeout(deadline);
            clearTimeout(graceTimer);
            resolvePromise({ end, code });
        };

        const stop = async (end: ConversionEnd): Promise<void> => {
            // A process that has already exited is not ours to stop; its own exit code is the answer.
            if (finished || exited || guardEnd) return;
            guardEnd = end;
            await killTree(child);
            // The 'close' event normally follows within a moment; do not wait forever for it.
            setTimeout(() => done(end, null), 5_000).unref();
        };

        child.on('error', (error) => {
            if (finished) return;
            finished = true;
            clearInterval(poller);
            clearTimeout(deadline);
            clearTimeout(graceTimer);
            reject(error);
        });
        child.on('exit', () => {
            exited = true;
            clearTimeout(graceTimer);
        });
        child.on('close', (code) => done(guardEnd ?? 'exited', code));

        const deadline = setTimeout(() => void stop('timed-out'), limits.timeoutMs);

        const poller = setInterval(() => {
            if (polling || finished || exited || guardEnd) return;
            polling = true;
            void (async () => {
                for (const output of options.outputs) {
                    if (settledOutputs.has(output.path)) continue;
                    try {
                        const path = join(options.assetDirectory, output.path);
                        const info = await stat(path);
                        if (info.mtimeMs < startedAt) continue; // left over from an earlier run
                        if (output.bytes !== undefined && info.size !== output.bytes) continue; // still being written
                        if (output.sha256) await sha256OfFile(path); // a wrong hash is reported by the final check
                        settledOutputs.add(output.path);
                    } catch {
                        // Not there yet, or being written.
                    }
                }
                const allWritten = options.outputs.length > 0 && settledOutputs.size === options.outputs.length;
                if (allWritten && graceTimer === undefined && !exited) {
                    graceTimer = setTimeout(() => void stop('stopped-after-outputs'), limits.graceMs);
                }
            })()
                .catch(() => undefined)
                .finally(() => {
                    polling = false;
                });
        }, limits.pollMs);
    });
}

async function runConvert(asset: AssetEntry, assetDirectory: string, limits?: ConversionLimits): Promise<void> {
    const convert = asset.convert;
    if (!convert) return;
    const script = resolve(repoRoot, convert.script);
    if (!(await fileExists(script))) throw new Error(`convert script not found: ${convert.script}`);
    const blender = await findBlender(convert.blenderVersion);
    const args = substitutePlaceholders(convert.args, { dir: assetDirectory, id: asset.id, repo: repoRoot });
    console.log(`  converting with Blender ${blender.version} at ${blender.path}`);
    const result = await runConversionProcess(blender.path, ['--background', '--factory-startup', '--python-exit-code', '1', '--python', script, '--', ...args], {
        cwd: assetDirectory,
        assetDirectory,
        outputs: asset.outputs,
        limits,
    });
    if (result.end === 'timed-out') {
        throw new Error(`Blender was still running after ${(limits ?? DEFAULT_CONVERSION_LIMITS).timeoutMs / 60_000} minutes without finishing the outputs, so it was stopped (${convert.script})`);
    }
    if (result.end === 'stopped-after-outputs') {
        console.log('  all outputs are written; Blender did not exit by itself, so it was stopped');
    } else if (result.code !== 0) {
        throw new Error(`Blender exited with code ${result.code} while running ${convert.script}`);
    }
}

async function linkOrCopy(source: string, destination: string): Promise<void> {
    await mkdir(dirname(destination), { recursive: true });
    await rm(destination, { force: true });
    try {
        await link(source, destination);
    } catch {
        // Different drive, or a file system without hard links: copy instead.
        await copyFile(source, destination);
    }
}

/**
 * Puts one downloaded file at `destination`. Files whose sha256 is known go
 * through a content cache (`<downloads>/.cache/<sha256>`), so two assets that
 * share a download fetch it once; the asset folder gets a hard link or a copy.
 */
export async function obtainDownload(
    download: AssetDownload,
    destination: string,
    cacheDirectory: string,
    force: boolean,
    log: Log,
): Promise<void> {
    const expected = download.sha256?.toLowerCase();
    const cachePath = expected ? join(cacheDirectory, expected) : null;

    if (!force && (await fileExists(destination))) {
        const actual = await sha256OfFile(destination);
        if (!expected || actual === expected) {
            log(`  ${download.file} is already downloaded`);
            if (cachePath && !(await fileExists(cachePath))) {
                await mkdir(cacheDirectory, { recursive: true });
                await linkOrCopy(destination, cachePath);
            }
            return;
        }
    }

    if (cachePath && expected) {
        if (!force && (await fileExists(cachePath)) && (await sha256OfFile(cachePath)) === expected) {
            log(`  ${download.file}: taking it from the download cache`);
        } else {
            await mkdir(cacheDirectory, { recursive: true });
            await downloadFile(download, cachePath);
            log(`  verified sha256 of ${download.file}`);
        }
        await linkOrCopy(cachePath, destination);
        return;
    }

    await mkdir(dirname(destination), { recursive: true });
    const actual = await downloadFile(download, destination);
    log(`  warning: ${download.file} has no sha256 in sources.json, so it is unverified. Computed: ${actual}`);
}

/** Hard-links (or copies, across drives) every file under `source` into `destination`. */
async function linkTree(source: string, destination: string): Promise<void> {
    await mkdir(destination, { recursive: true });
    for (const entry of await readdir(source, { withFileTypes: true })) {
        const from = join(source, entry.name);
        const to = join(destination, entry.name);
        if (entry.isDirectory()) await linkTree(from, to);
        else await linkOrCopy(from, to);
    }
}

async function directoryExists(path: string): Promise<boolean> {
    try {
        return (await stat(path)).isDirectory();
    } catch {
        return false;
    }
}

/**
 * Unpacks a zip into the asset folder. When the zip's sha256 is known, the
 * unpacked files are kept in `<cache>/extracted/<sha256>/` and linked into the
 * asset folder, so two assets that share one zip (the two Bistro halves) unpack
 * it once. A half-finished unpack is never reused: it is built under a
 * temporary name and renamed when complete.
 */
export async function extractWithCache(
    download: AssetDownload,
    archivePath: string,
    assetDirectory: string,
    cacheDirectory: string,
    log: Log,
    extract: (archive: string, destination: string) => Promise<void> = extractZip,
): Promise<void> {
    if (!download.sha256) {
        log(`  extracting ${download.file}`);
        await extract(archivePath, assetDirectory);
        return;
    }
    const cached = join(cacheDirectory, 'extracted', download.sha256.toLowerCase());
    if (await directoryExists(cached)) {
        log(`  ${download.file}: taking the unpacked files from the cache`);
    } else {
        log(`  extracting ${download.file}`);
        const partial = `${cached}.partial`;
        await rm(partial, { recursive: true, force: true });
        await mkdir(partial, { recursive: true });
        try {
            await extract(archivePath, partial);
        } catch (error) {
            await rm(partial, { recursive: true, force: true });
            throw error;
        }
        await rename(partial, cached);
    }
    await linkTree(cached, assetDirectory);
}

/** Brings one asset up to date. Throws with a clear message on any failure. */
export async function fetchAsset(
    asset: AssetEntry,
    options: { force: boolean; downloadsDirectory?: string },
): Promise<'up-to-date' | 'fetched' | 'skipped'> {
    const log: Log = (message) => console.log(message);
    if (asset.skip) {
        log(`${asset.id}: skipped: ${asset.skipReason ?? 'no reason given'}`);
        return 'skipped';
    }
    const downloadsDirectory = options.downloadsDirectory ?? downloadsRoot;
    const assetDirectory = join(downloadsDirectory, asset.id);
    await mkdir(assetDirectory, { recursive: true });

    if (!options.force) {
        const existing = await checkOutputs(assetDirectory, asset.outputs, () => undefined);
        if (existing.problems.length === 0) {
            const note = existing.unverified > 0 ? ` (${existing.unverified} output(s) have no sha256 in sources.json, so they are only checked for existing)` : '';
            log(`${asset.id}: up to date${note}`);
            return 'up-to-date';
        }
    }

    log(`${asset.id}: ${asset.title} by ${asset.author} (${asset.license}${asset.redistributable ? '' : ', not redistributable: stays local'})`);

    const downloads = downloadsOf(asset);
    for (const download of downloads) {
        await obtainDownload(download, join(assetDirectory, download.file), join(downloadsDirectory, '.cache'), options.force, log);
    }

    if (asset.extract === 'zip') {
        await extractWithCache(downloads[0], join(assetDirectory, downloads[0].file), assetDirectory, join(downloadsDirectory, '.cache'), log);
    }
    await runConvert(asset, assetDirectory);

    const result = await checkOutputs(assetDirectory, asset.outputs, log, asset.convert);
    if (result.problems.length > 0) {
        throw new Error(`outputs are not right after fetching:\n    ${result.problems.join('\n    ')}`);
    }
    log(`  ${asset.id}: ${asset.outputs.length} output file(s) ${result.unverified === 0 ? 'verified' : 'present (some unverified)'}`);
    return 'fetched';
}

// ---------------------------------------------------------------------------
// Command line
// ---------------------------------------------------------------------------

const helpText = `Usage: pnpm assets:fetch [--only <id>[,<id>]] [--force] [--verify-only] [--sources <path>]
       node tools/assets/fetch.ts --hash <file>   print the sha256 and size to put in sources.json

--verify-only checks the files already on disk against sources.json (sizes, hashes,
structure of non-reproducible .glb files) without downloading or converting anything.

Downloads and verifies the external test assets listed in tools/assets/sources.json
into assets/downloads/<id>/. Assets that are already complete are skipped.`;

/** Checks the outputs already on disk and reports; changes nothing. Returns the exit code. */
async function verifyOnly(assets: readonly AssetEntry[]): Promise<number> {
    const failed: string[] = [];
    for (const asset of assets) {
        if (asset.skip) {
            console.log(`${asset.id}: skipped: ${asset.skipReason ?? 'no reason given'}`);
            continue;
        }
        const result = await checkOutputs(join(downloadsRoot, asset.id), asset.outputs, (message) => console.log(message), asset.convert);
        if (result.problems.length === 0) {
            console.log(`${asset.id}: OK, ${asset.outputs.length} output file(s) verified`);
        } else {
            failed.push(asset.id);
            console.error(`${asset.id}: FAILED\n    ${result.problems.join('\n    ')}`);
        }
    }
    console.log(failed.length === 0 ? '\nAll outputs verified.' : `\nFailed: ${failed.join(', ')}`);
    return failed.length === 0 ? 0 : 1;
}

async function main(): Promise<number> {
    const options = parseArgs(process.argv.slice(2));
    if (options.help) {
        console.log(helpText);
        return 0;
    }
    if (options.hash) {
        const path = resolve(options.hash);
        console.log(`"sha256": "${await sha256OfFile(path)}", "bytes": ${(await stat(path)).size}`);
        return 0;
    }

    const sourcesPath = options.sourcesPath ? resolve(options.sourcesPath) : defaultSourcesPath;
    let text: string;
    try {
        text = await readFile(sourcesPath, 'utf8');
    } catch {
        throw new Error(`Could not read ${sourcesPath}. It lists the assets to download.`);
    }
    let parsed: unknown;
    try {
        parsed = JSON.parse(text);
    } catch (error) {
        throw new Error(`${sourcesPath} is not valid JSON: ${error instanceof Error ? error.message : String(error)}`);
    }
    const problems = validateSources(parsed);
    if (problems.length > 0) throw new Error(`${sourcesPath} has problems:\n  ${problems.join('\n  ')}`);

    const sources = parsed as SourcesFile;
    if (sources.assets.length === 0) {
        console.log('No assets are listed in sources.json yet, so there is nothing to download.');
        return 0;
    }

    const selected = selectAssets(sources.assets, options.only);
    if (options.verifyOnly) return verifyOnly(selected);
    const failures: string[] = [];
    let fetched = 0;
    let skipped = 0;
    for (const asset of selected) {
        try {
            const outcome = await fetchAsset(asset, { force: options.force });
            if (outcome === 'fetched') fetched++;
            if (outcome === 'skipped') skipped++;
        } catch (error) {
            const message = error instanceof Error ? error.message : String(error);
            console.error(`${asset.id}: FAILED: ${message}`);
            failures.push(asset.id);
        }
    }
    console.log(`\n${selected.length} asset(s): ${fetched} fetched, ${selected.length - fetched - skipped - failures.length} already up to date, ${skipped} skipped, ${failures.length} failed`);
    if (failures.length > 0) {
        console.error(`Failed: ${failures.join(', ')}`);
        return 1;
    }
    return 0;
}

// Run only when started as a script, not when a test imports the helpers above.
if (process.argv[1] && resolve(process.argv[1]) === resolve(fileURLToPath(import.meta.url))) {
    main().then(
        (code) => process.exit(code),
        (error: unknown) => {
            console.error(error instanceof Error ? error.message : error);
            process.exit(1);
        },
    );
}
