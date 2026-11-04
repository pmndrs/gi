/**
 * Downloads, verifies, unpacks and converts the external test assets.
 *
 *   pnpm assets:fetch                 every asset in tools/assets/sources.json
 *   pnpm assets:fetch --only <id>     one asset (repeat the flag, or use a,b)
 *   pnpm assets:fetch --force         redo everything even if outputs look fine
 *   pnpm assets:fetch --verify-only   check what is on disk; change nothing
 *   pnpm assets:fetch --sources <p>   read a different sources file
 *
 * Files land in assets/downloads/<id>/ (gitignored). For each asset:
 *   1. If every output exists and matches its sha256, do nothing.
 *   2. Otherwise download each file (streamed, with progress) unless it is
 *      already there with the right hash, and check its size and sha256.
 *      `download` is one object or an array (a glTF with its .bin and
 *      textures; `file` may name subfolders). Downloads go through a cache in
 *      assets/downloads/.cache/<sha256>, so assets that share a download (the
 *      two Bistro halves) fetch it once.
 *   3. Unpack zips: `tar` (bsdtar, ships with Windows 10+) on Windows,
 *      PowerShell `Expand-Archive` if tar is missing, `unzip` elsewhere.
 *      No npm dependency. The unpacked files are kept once in
 *      .cache/extracted/<sha256>/ and hard-linked into the asset folder, which
 *      then holds exactly the archive's files plus the downloads and outputs.
 *   4. Run the convert step, if any. Blender is found from the BLENDER
 *      environment variable, then PATH, then the default install folders, and
 *      must be the exact version in sources.json (like 4.5.0).
 *      The script runs as `blender --background --factory-startup --python
 *      <script> -- <args>` (factory settings: the user's add-ons can hang Blender)
 *      with the asset folder as the working directory, and `{dir}`, `{id}` and
 *      `{repo}` in args replaced by the absolute asset folder, the asset id and
 *      the repository root.
 *      Convert scripts must never write to their input files: those are hard
 *      links into the shared cache, so an in-place edit would change the cache
 *      and the other asset that shares it. The tool checks this: it notes the
 *      size and modification time of every input before the conversion and
 *      fails if any changed.
 *   5. Check every output's sha256 (or, for .glb files Blender does not write
 *      the same way twice, their structure: see `AssetOutput.reproducible`),
 *      and that a .glb marked `dielectric` has no metal material.
 *
 * An asset with `"skip": true` and a `skipReason` is reported and left alone.
 *
 * sources.json is checked strictly before anything is downloaded: every
 * non-skipped asset needs `redistributable`, sha256 plus bytes for every
 * download and every output, and a license on the allow list (CC0 or plain
 * CC-BY). Any other license needs a `licenseApproval` that names Dennis's
 * entry in docs/DECISIONS.md, and that entry must exist. To add an asset, get
 * the numbers with `node tools/assets/fetch.ts --hash <file>`.
 * Downloads give up after 60 s without an answer or without new data.
 * Everything here is one file so it runs under plain Node type
 * stripping with no imports from the repo.
 */

import { spawn } from 'node:child_process';
import { createHash } from 'node:crypto';
import { createReadStream, createWriteStream } from 'node:fs';
import { copyFile, link, mkdir, open, readdir, readFile, rename, rm, stat, writeFile } from 'node:fs/promises';
import { delimiter, dirname, join, resolve } from 'node:path';
import { Readable, Transform } from 'node:stream';
import { pipeline } from 'node:stream/promises';
import { fileURLToPath } from 'node:url';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/** `validateSources` guarantees `bytes` and `sha256` before anything is downloaded. */
export interface AssetDownload {
    url: string;
    /** Path inside the asset folder; may include subfolders, like "textures/bark.jpg". */
    file: string;
    bytes: number;
    sha256: string;
}

export interface AssetConvert {
    tool: 'blender';
    /** Repo-relative path to a Python script. It must write only new files, never its inputs (see the header). */
    script: string;
    args: string[];
    /**
     * The full version of the Blender that made the recorded outputs, like
     * "4.5.0". Only that exact version is used: other versions, patch releases
     * included, can crash or write different bytes, and then the recorded
     * hashes would have to be made again.
     */
    blenderVersion: string;
}

/** `validateSources` guarantees `sha256` and `bytes`. */
export interface AssetOutput {
    /** Relative to assets/downloads/<id>/. */
    path: string;
    sha256: string;
    bytes: number;
    /**
     * Set to false when the converter is known not to write the same bits every
     * run (Blender's glTF exporter writes some vertex data a little differently
     * between runs). The file must then be a .glb, and instead of the
     * whole-file sha256 it is checked piece by piece:
     * - the exact size, and the exact length of its BIN chunk;
     * - the sha256 of its JSON chunk (the scene description, identical between runs);
     * - the sha256 of every bufferView (`bufferViewHashes`), except views used
     *   only by the vertex attributes listed in `varyingAttributes`. Index
     *   buffers and embedded images are always pinned exactly;
     * - every byte of the BIN chunk outside all bufferViews (alignment
     *   padding) must be zero.
     * The whole-file sha256 is then only logged. Leave this out for everything
     * else: the sha256 is how we know a conversion is the same.
     */
    reproducible?: boolean;
    /** sha256 of the .glb's JSON chunk, bytes exactly as stored. Required with `reproducible: false`. */
    jsonSha256?: string;
    /** Length in bytes of the .glb's BIN chunk. Required with `reproducible: false`. */
    binBytes?: number;
    /**
     * Repo-relative path of a JSON file (`GlbViewRecord`) with the sha256 of
     * each bufferView. Kept out of sources.json because Bistro has about ten
     * thousand bufferViews. Made by `--record-glb-views`. Required with `reproducible: false`.
     */
    bufferViewHashes?: string;
    /**
     * glTF attribute names (like "NORMAL") whose data the converter is proven
     * to write differently between runs; bufferViews used only by these are
     * not pinned. Required with `reproducible: false`; may be empty.
     */
    varyingAttributes?: string[];
    /**
     * Set to true for a .glb scene that must contain no metal. Every material
     * must then write `metallicFactor: 0` (glTF's default when it is left out
     * is 1, fully metal), and every mesh primitive must have a material
     * (glTF's default material is fully metal too). Checked on every run,
     * whatever the hashes say, so re-recording the hashes of a broken
     * conversion cannot hide it. With factor 0 a metallicRoughness texture
     * cannot add metal, because glTF multiplies the two.
     * Why: Blender's OBJ importer turns the MTL ambient colour `Ka` into
     * Metallic, which made the whole Breakfast Room metal (T0.4 follow-up).
     */
    dielectric?: boolean;
}

/**
 * Dennis's go-ahead for a license that is not on the allow list. The tool
 * checks that docs/DECISIONS.md has a `## <date>` section with an entry whose
 * bold title is exactly `title`.
 */
export interface LicenseApproval {
    /** Must be "docs/DECISIONS.md". */
    record: string;
    /** The date heading of the entry, like "2026-11-04". */
    date: string;
    /** The entry's bold title, exactly as written there. */
    title: string;
}

export interface AssetEntry {
    id: string;
    title: string;
    author: string;
    sourceUrl: string;
    /** SPDX id for plain licenses. A license with extra terms gets its own id (like "CC-BY-4.0+intel-preface") so it never passes as the plain one. */
    license: string;
    licenseUrl?: string;
    /** Required when `license` is not on the allow list. */
    licenseApproval?: LicenseApproval;
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
    /** Asset ids whose non-reproducible .glb outputs get their bufferView hashes (re)written, instead of fetching. */
    recordGlbViews: string[];
    /** Check every asset's existing outputs against sources.json; download and convert nothing. */
    verifyOnly: boolean;
    help: boolean;
}

// ---------------------------------------------------------------------------
// Pure helpers (unit-tested)
// ---------------------------------------------------------------------------

const KEBAB_CASE = /^[a-z0-9]+(-[a-z0-9]+)*$/;
const SHA256_HEX = /^[0-9a-f]{64}$/i;

const DATE = /^\d{4}-\d{2}-\d{2}$/;
/** Where license approvals must point. */
export const DECISIONS_FILE = 'docs/DECISIONS.md';
/** glTF attribute names: POSITION, NORMAL, TEXCOORD_0, _CUSTOM. */
const GLTF_ATTRIBUTE = /^_?[A-Z][A-Z0-9_]*$/;

/**
 * Licenses we are willing to download and show without asking: CC0 and plain
 * CC-BY (any version). Share-alike, non-commercial, custom licenses and
 * CC-BY with extra terms are not on the list; each needs Dennis's decision in
 * docs/DECISIONS.md and a `licenseApproval` pointing at it.
 */
export function isAllowedLicense(license: string): boolean {
    return license === 'CC0-1.0' || /^CC-BY-\d\.\d$/.test(license);
}

/** Problems with a `licenseApproval` value's shape (not whether DECISIONS.md has it: see `checkLicenseApprovals`). */
function licenseApprovalProblems(value: unknown): string[] {
    if (!isRecord(value)) return ['must be an object with record, date and title'];
    const problems: string[] = [];
    if (value.record !== DECISIONS_FILE) problems.push(`record must be "${DECISIONS_FILE}"`);
    if (!(typeof value.date === 'string' && DATE.test(value.date))) problems.push('date must look like "2026-11-04"');
    if (!isNonEmptyString(value.title)) problems.push('title must be the entry\'s bold title, exactly as written');
    return problems;
}

/**
 * Checks that every approved non-allow-list license points at a real entry:
 * a `## <date>` section of DECISIONS.md holding a `- **<title>**` line.
 * Returns one line per approval that does not.
 */
export function checkLicenseApprovals(assets: readonly AssetEntry[], decisionsMarkdown: string): string[] {
    // Split the file into its date sections: "## 2026-11-04" up to the next "## ".
    const sections = new Map<string, string>();
    for (const section of decisionsMarkdown.split(/^## /m).slice(1)) {
        const newline = section.indexOf('\n');
        const heading = (newline === -1 ? section : section.slice(0, newline)).trim();
        sections.set(heading, (sections.get(heading) ?? '') + section);
    }
    const problems: string[] = [];
    for (const asset of assets) {
        if (asset.skip || isAllowedLicense(asset.license) || !asset.licenseApproval) continue;
        const { date, title } = asset.licenseApproval;
        const section = sections.get(date);
        if (section === undefined) {
            problems.push(`${asset.id}: licenseApproval names the ${date} section of ${DECISIONS_FILE}, which does not exist`);
        } else if (!section.split('\n').some((line) => line.trim() === `- **${title}**`)) {
            problems.push(`${asset.id}: the ${date} section of ${DECISIONS_FILE} has no entry titled "${title}"`);
        }
    }
    return problems;
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
            if (entry.licenseApproval === undefined) {
                errors.push(
                    `${where}.license: "${entry.license}" is not on the allow list (CC0-1.0, CC-BY-<version>), so it needs a licenseApproval ` +
                        `that points at Dennis's entry in ${DECISIONS_FILE} (record, date, title). Ask Dennis before adding another license.`,
                );
            } else {
                for (const problem of licenseApprovalProblems(entry.licenseApproval)) errors.push(`${where}.licenseApproval: ${problem}`);
            }
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
            } else if (!(typeof convert.blenderVersion === 'string' && /^\d+\.\d+\.\d+$/.test(convert.blenderVersion))) {
                errors.push(`${where}.convert.blenderVersion: required, the full version that made the recorded outputs, like "4.5.0"`);
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
                        const viewFile = output.bufferViewHashes;
                        if (!(isNonEmptyString(viewFile) && isSafeRelativePath(viewFile) && viewFile.endsWith('.json'))) {
                            errors.push(`${outputWhere}.bufferViewHashes: required with reproducible false, a repo-relative .json path (made by --record-glb-views)`);
                        }
                        const varying = output.varyingAttributes;
                        if (!(Array.isArray(varying) && varying.every((name: unknown) => typeof name === 'string' && GLTF_ATTRIBUTE.test(name)))) {
                            errors.push(`${outputWhere}.varyingAttributes: required with reproducible false, a list of glTF attribute names like "NORMAL" (may be empty)`);
                        }
                    } else if (output.bufferViewHashes !== undefined || output.varyingAttributes !== undefined) {
                        errors.push(`${outputWhere}: bufferViewHashes and varyingAttributes only go with reproducible false`);
                    }
                    if (output.dielectric !== undefined) {
                        if (typeof output.dielectric !== 'boolean') {
                            errors.push(`${outputWhere}.dielectric: must be true or false when present`);
                        } else if (!output.path.toLowerCase().endsWith('.glb')) {
                            errors.push(`${outputWhere}.dielectric: only supported for .glb files`);
                        }
                    }
                }
            });
        }
    });
    return errors;
}

export function parseArgs(argv: readonly string[]): CliOptions {
    const options: CliOptions = { only: [], force: false, sourcesPath: null, hash: null, recordGlbViews: [], verifyOnly: false, help: false };
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
        else if (arg === '--record-glb-views') options.recordGlbViews.push(...takeValue().split(',').filter(Boolean));
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

/**
 * True when `actual` is exactly the wanted full version ("4.5.0"). A patch
 * release is not close enough: its exporter can write different bytes.
 */
export function versionMatches(actual: string, wanted: string): boolean {
    const parts = (version: string) => version.split('.').map(Number);
    const [have, want] = [parts(actual), parts(wanted)];
    return have.length === 3 && want.length === 3 && have.every((part, index) => part === want[index]);
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
 * Finds a Blender of exactly the wanted version, like "4.5.0" (any version
 * when `wanted` is undefined). Each candidate is asked for its version rather
 * than trusted by its name or by the BLENDER variable. Throws a plain message
 * listing what was found, and what to do to move to another version.
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
    const why = wanted
        ? ` The hashes in tools/assets/sources.json were recorded with exactly this version; another one, even a patch release, can write different bytes.`
        : '';
    const moveOn = wanted
        ? ` To move to another version instead, set convert.blenderVersion to it, run the conversion, and re-record the output hashes ` +
          `(--hash for each output, and --record-glb-views <id> for .glb outputs marked reproducible false).`
        : '';
    throw new Error(
        `${need} is needed for this conversion.${why} ${found} Looked in the BLENDER variable, PATH and the default install folders. ` +
            `Install ${need}, or set BLENDER to its full path.${moveOn}`,
    );
}

// ---------------------------------------------------------------------------
// Steps
// ---------------------------------------------------------------------------

type Log = (message: string) => void;

/** A .glb split into the pieces `checkOutputs` pins one by one. */
export interface GlbStructure {
    /** sha256 of the whole file. */
    fileSha256: string;
    /** sha256 of the JSON chunk's bytes, exactly as stored (including its padding). */
    jsonSha256: string;
    /** Length of the BIN chunk's data, as the chunk header says. */
    binBytes: number;
    /** sha256 of each bufferView's bytes, in index order. */
    bufferViewSha256: string[];
    /**
     * What each bufferView is used for, from the JSON chunk: "indices",
     * "image", a vertex attribute name such as "NORMAL", or "other" (morph
     * targets, sparse data, animation, skins). Empty when nothing uses it.
     */
    bufferViewUses: string[][];
    /** How many bytes of the BIN chunk lie outside every bufferView and are not zero. */
    nonZeroPaddingBytes: number;
}

/** The glTF JSON fields the check reads. */
interface GltfJsonForViews {
    buffers?: Array<{ uri?: string }>;
    bufferViews?: Array<{ buffer: number; byteOffset?: number; byteLength: number }>;
    accessors?: Array<{ bufferView?: number; sparse?: { indices: { bufferView: number }; values: { bufferView: number } } }>;
    meshes?: Array<{ primitives: Array<{ attributes: Record<string, number>; indices?: number; targets?: Array<Record<string, number>> }> }>;
    images?: Array<{ bufferView?: number }>;
    animations?: Array<{ samplers: Array<{ input: number; output: number }> }>;
    skins?: Array<{ inverseBindMatrices?: number }>;
}

const GLB_MAGIC = 0x46546c67; // "glTF"
const GLB_CHUNK_JSON = 0x4e4f534a; // "JSON"
const GLB_CHUNK_BIN = 0x004e4942; // "BIN\0"
const GLB_HEADER_BYTES = 12;
const GLB_CHUNK_HEADER_BYTES = 8;

function sha256Of(bytes: Uint8Array): string {
    return createHash('sha256').update(bytes).digest('hex');
}

/** Lists, for every bufferView, what the glTF uses it for. */
function bufferViewUses(gltf: GltfJsonForViews, viewCount: number): string[][] {
    const uses: Array<Set<string>> = Array.from({ length: viewCount }, () => new Set<string>());
    const accessors = gltf.accessors ?? [];
    const useAccessor = (accessorIndex: number | undefined, use: string): void => {
        if (accessorIndex === undefined) return;
        const accessor = accessors[accessorIndex];
        if (accessor?.bufferView !== undefined) uses[accessor.bufferView]?.add(use);
    };
    for (const mesh of gltf.meshes ?? []) {
        for (const primitive of mesh.primitives) {
            for (const [name, accessorIndex] of Object.entries(primitive.attributes)) useAccessor(accessorIndex, name);
            useAccessor(primitive.indices, 'indices');
            for (const target of primitive.targets ?? []) {
                for (const accessorIndex of Object.values(target)) useAccessor(accessorIndex, 'other');
            }
        }
    }
    // Sparse data overrides an accessor's values; its views are never one of the plain vertex attributes.
    for (const accessor of accessors) {
        if (accessor.sparse) {
            uses[accessor.sparse.indices.bufferView]?.add('other');
            uses[accessor.sparse.values.bufferView]?.add('other');
        }
    }
    for (const image of gltf.images ?? []) {
        if (image.bufferView !== undefined) uses[image.bufferView]?.add('image');
    }
    for (const animation of gltf.animations ?? []) {
        for (const sampler of animation.samplers) {
            useAccessor(sampler.input, 'other');
            useAccessor(sampler.output, 'other');
        }
    }
    for (const skin of gltf.skins ?? []) useAccessor(skin.inverseBindMatrices, 'other');
    return uses.map((set) => [...set].sort());
}

/**
 * Reads a .glb and hashes it piece by piece: the whole file, the JSON chunk and
 * every bufferView of the BIN chunk. Reads the file into memory once (Bistro
 * exterior is 267 MB). Throws a plain message for a file that is not a
 * well-formed single-buffer .glb.
 */
export async function readGlbStructure(path: string): Promise<GlbStructure> {
    const file = await readFile(path);
    if (file.length < GLB_HEADER_BYTES + GLB_CHUNK_HEADER_BYTES) throw new Error('too short to be a .glb file');
    if (file.readUInt32LE(0) !== GLB_MAGIC) throw new Error('not a .glb file (bad magic)');
    if (file.readUInt32LE(8) !== file.length) throw new Error(`the .glb header says ${file.readUInt32LE(8)} bytes but the file has ${file.length}`);

    const jsonLength = file.readUInt32LE(GLB_HEADER_BYTES);
    if (file.readUInt32LE(GLB_HEADER_BYTES + 4) !== GLB_CHUNK_JSON) throw new Error('the first chunk is not JSON');
    const jsonStart = GLB_HEADER_BYTES + GLB_CHUNK_HEADER_BYTES;
    if (jsonStart + jsonLength > file.length) throw new Error('the JSON chunk is cut off');
    const json = file.subarray(jsonStart, jsonStart + jsonLength);

    const binHeaderStart = jsonStart + jsonLength;
    if (binHeaderStart + GLB_CHUNK_HEADER_BYTES > file.length) throw new Error('there is no BIN chunk');
    if (file.readUInt32LE(binHeaderStart + 4) !== GLB_CHUNK_BIN) throw new Error('the second chunk is not BIN');
    const binBytes = file.readUInt32LE(binHeaderStart);
    const binStart = binHeaderStart + GLB_CHUNK_HEADER_BYTES;
    // Nothing may follow the BIN chunk: bytes the check does not look at could change unseen.
    if (binStart + binBytes !== file.length) {
        throw new Error(`the BIN chunk ends at byte ${binStart + binBytes}, but the file has ${file.length}`);
    }
    const bin = file.subarray(binStart, binStart + binBytes);

    let gltf: GltfJsonForViews;
    try {
        gltf = JSON.parse(json.toString('utf8')) as GltfJsonForViews;
    } catch (error) {
        throw new Error(`the JSON chunk is not valid JSON: ${error instanceof Error ? error.message : String(error)}`);
    }
    // In a .glb, buffer 0 without a uri is the BIN chunk. Data in any other buffer would live outside this file.
    if ((gltf.buffers ?? []).some((buffer, index) => index > 0 || buffer.uri !== undefined)) {
        throw new Error('the .glb has more than one buffer or an external buffer, so its data cannot be checked here');
    }
    const views = gltf.bufferViews ?? [];
    const ranges = views.map((view, index) => {
        const start = view.byteOffset ?? 0;
        const end = start + view.byteLength;
        if (view.buffer !== 0 || !Number.isInteger(start) || !Number.isInteger(end) || start < 0 || end > binBytes) {
            throw new Error(`bufferView ${index} is not inside the BIN chunk`);
        }
        return { start, end };
    });

    // Bytes no bufferView covers are alignment padding. Walk the gaps between the ranges in order of offset.
    let nonZeroPaddingBytes = 0;
    const countNonZero = (from: number, to: number): void => {
        for (let byte = from; byte < to; byte++) {
            if (bin[byte] !== 0) nonZeroPaddingBytes++;
        }
    };
    let coveredUpTo = 0;
    for (const range of [...ranges].sort((a, b) => a.start - b.start)) {
        if (range.start > coveredUpTo) countNonZero(coveredUpTo, range.start);
        coveredUpTo = Math.max(coveredUpTo, range.end);
    }
    countNonZero(coveredUpTo, binBytes);

    return {
        fileSha256: sha256Of(file),
        jsonSha256: sha256Of(json),
        binBytes,
        bufferViewSha256: ranges.map((range) => sha256Of(bin.subarray(range.start, range.end))),
        bufferViewUses: bufferViewUses(gltf, views.length),
        nonZeroPaddingBytes,
    };
}

/**
 * The recorded bufferView hashes of one non-reproducible .glb output, kept in
 * its own file (`AssetOutput.bufferViewHashes`) because Bistro has about ten
 * thousand bufferViews.
 */
export interface GlbViewRecord {
    asset: string;
    output: string;
    /** The JSON chunk these views belong to; must equal the output's `jsonSha256`. */
    jsonSha256: string;
    /** The `varyingAttributes` in force when this was recorded. */
    varyingAttributes: string[];
    /** sha256 of each bufferView in index order; null for a view used only by varying attributes. */
    bufferViews: Array<string | null>;
}

/** True for a bufferView that is used, and used only, by vertex attributes the converter writes differently each run. */
function isVaryingOnly(uses: readonly string[], varyingAttributes: readonly string[]): boolean {
    return uses.length > 0 && uses.every((use) => varyingAttributes.includes(use));
}

/** Makes the record to store for a freshly converted output. */
export function makeGlbViewRecord(assetId: string, output: AssetOutput, structure: GlbStructure): GlbViewRecord {
    const varyingAttributes = output.varyingAttributes ?? [];
    return {
        asset: assetId,
        output: output.path,
        jsonSha256: structure.jsonSha256,
        varyingAttributes: [...varyingAttributes],
        bufferViews: structure.bufferViewSha256.map((hash, index) =>
            isVaryingOnly(structure.bufferViewUses[index], varyingAttributes) ? null : hash,
        ),
    };
}

/** The record as JSON text with one bufferView hash per line, so a re-record shows up as a readable diff. */
export function formatGlbViewRecord(record: GlbViewRecord): string {
    return [
        '{',
        `  "asset": ${JSON.stringify(record.asset)},`,
        `  "output": ${JSON.stringify(record.output)},`,
        `  "jsonSha256": ${JSON.stringify(record.jsonSha256)},`,
        `  "varyingAttributes": ${JSON.stringify(record.varyingAttributes)},`,
        '  "bufferViews": [',
        record.bufferViews.map((hash) => `    ${JSON.stringify(hash)}`).join(',\n'),
        '  ]',
        '}',
        '',
    ].join('\n');
}

/** How many differing bufferViews to name before summing up the rest. */
const MAX_VIEWS_LISTED = 8;

/** Problems with a not-bit-reproducible .glb output, or an empty list when every pinned piece matches. */
export function checkGlbStructure(actual: GlbStructure, output: AssetOutput, record: GlbViewRecord): string[] {
    const problems: string[] = [];
    if (!output.jsonSha256 || !sameHash(actual.jsonSha256, output.jsonSha256)) {
        problems.push(
            `the JSON chunk differs (expected sha256 ${output.jsonSha256}, got ${actual.jsonSha256}); the scene description changed, which is not float noise`,
        );
    }
    if (actual.binBytes !== output.binBytes) {
        problems.push(`the BIN chunk is ${actual.binBytes} bytes but sources.json says ${output.binBytes}`);
    }
    // The bufferViews of a different scene description cannot be compared one by one.
    if (problems.length > 0) return problems;

    if (!sameHash(record.jsonSha256, actual.jsonSha256)) {
        return [`the bufferView record ${output.bufferViewHashes} was made for another JSON chunk (${record.jsonSha256}); re-record it with --record-glb-views`];
    }
    if (record.bufferViews.length !== actual.bufferViewSha256.length) {
        return [`the file has ${actual.bufferViewSha256.length} bufferViews but ${output.bufferViewHashes} lists ${record.bufferViews.length}`];
    }

    const varyingAttributes = output.varyingAttributes ?? [];
    const differing: string[] = [];
    record.bufferViews.forEach((expected, index) => {
        const uses = actual.bufferViewUses[index];
        const described = `bufferView ${index} (${uses.length > 0 ? uses.join(', ') : 'unused'})`;
        if (expected === null) {
            // A view may be left unpinned only for the attributes sources.json names; index buffers and images never are.
            if (!isVaryingOnly(uses, varyingAttributes)) {
                problems.push(`${described} has no recorded hash, but only views used just by ${varyingAttributes.join(', ') || '(none listed)'} may be left out`);
            }
        } else if (!sameHash(expected, actual.bufferViewSha256[index])) {
            differing.push(described);
        }
    });
    if (differing.length > 0) {
        const more = differing.length > MAX_VIEWS_LISTED ? ` and ${differing.length - MAX_VIEWS_LISTED} more` : '';
        problems.push(`${differing.length} pinned bufferView(s) differ: ${differing.slice(0, MAX_VIEWS_LISTED).join('; ')}${more}`);
    }
    if (actual.nonZeroPaddingBytes > 0) {
        problems.push(`${actual.nonZeroPaddingBytes} byte(s) of the BIN chunk outside every bufferView are not zero`);
    }
    return problems;
}

export async function readGlbViewRecord(path: string): Promise<GlbViewRecord> {
    let parsed: unknown;
    try {
        parsed = JSON.parse(await readFile(path, 'utf8'));
    } catch (error) {
        throw new Error(`cannot read the bufferView record ${path}: ${error instanceof Error ? error.message : String(error)}`);
    }
    const isHashOrNull = (hash: unknown) => hash === null || (typeof hash === 'string' && SHA256_HEX.test(hash));
    if (
        !isRecord(parsed) ||
        typeof parsed.jsonSha256 !== 'string' ||
        !Array.isArray(parsed.bufferViews) ||
        !parsed.bufferViews.every(isHashOrNull)
    ) {
        throw new Error(`${path} is not a bufferView record (it needs jsonSha256 and a bufferViews list of sha256 or null)`);
    }
    return parsed as unknown as GlbViewRecord;
}

/** The glTF JSON fields the metal check reads. Real materials carry more fields (base colour, roughness), which it ignores. */
export interface GltfJsonForMaterials {
    materials?: Array<{
        name?: string;
        pbrMetallicRoughness?: { metallicFactor?: number; metallicRoughnessTexture?: unknown; [otherField: string]: unknown };
        [otherField: string]: unknown;
    }>;
    meshes?: Array<{ name?: string; primitives?: Array<{ material?: number; [otherField: string]: unknown }> }>;
}

/** glTF's value for a metallicFactor that is not written, and the metalness of its default material. */
const GLTF_DEFAULT_METALLIC_FACTOR = 1;

/** How many offending materials to name before summing up the rest. */
const MAX_MATERIALS_LISTED = 8;

/**
 * Reads only the JSON chunk of a .glb (the header and the scene description),
 * not the BIN chunk, so checking materials does not load the vertex data.
 */
export async function readGlbJson(path: string): Promise<unknown> {
    const handle = await open(path, 'r');
    try {
        const headers = Buffer.alloc(GLB_HEADER_BYTES + GLB_CHUNK_HEADER_BYTES);
        const { bytesRead } = await handle.read(headers, 0, headers.length, 0);
        if (bytesRead < headers.length) throw new Error('too short to be a .glb file');
        if (headers.readUInt32LE(0) !== GLB_MAGIC) throw new Error('not a .glb file (bad magic)');
        if (headers.readUInt32LE(GLB_HEADER_BYTES + 4) !== GLB_CHUNK_JSON) throw new Error('the first chunk is not JSON');
        const jsonLength = headers.readUInt32LE(GLB_HEADER_BYTES);
        const json = Buffer.alloc(jsonLength);
        const jsonRead = await handle.read(json, 0, jsonLength, headers.length);
        if (jsonRead.bytesRead < jsonLength) throw new Error('the JSON chunk is cut off');
        try {
            return JSON.parse(json.toString('utf8'));
        } catch (error) {
            throw new Error(`the JSON chunk is not valid JSON: ${error instanceof Error ? error.message : String(error)}`);
        }
    } finally {
        await handle.close();
    }
}

/**
 * Problems that would put metal into a scene that must have none (see
 * `AssetOutput.dielectric`), or an empty list. glTF's metalness is
 * metallicFactor times the blue channel of the metallicRoughness texture, so
 * a factor of exactly 0 is metal-free whatever the texture holds, and any
 * other factor (including a missing one, which means 1) is not.
 */
export function metalProblems(gltf: GltfJsonForMaterials): string[] {
    const problems: string[] = [];
    const metallic: string[] = [];
    (gltf.materials ?? []).forEach((material, index) => {
        const factor = material.pbrMetallicRoughness?.metallicFactor;
        if (factor === 0) return;
        const name = material.name !== undefined ? `"${material.name}"` : `#${index}`;
        const value = factor === undefined ? `not written, so glTF's default ${GLTF_DEFAULT_METALLIC_FACTOR}` : String(factor);
        const texture = material.pbrMetallicRoughness?.metallicRoughnessTexture !== undefined ? ', with a metallicRoughness texture' : '';
        metallic.push(`${name} (metallicFactor ${value}${texture})`);
    });
    if (metallic.length > 0) {
        const more = metallic.length > MAX_MATERIALS_LISTED ? ` and ${metallic.length - MAX_MATERIALS_LISTED} more` : '';
        problems.push(`${metallic.length} material(s) can be metal, but this scene must have none: ${metallic.slice(0, MAX_MATERIALS_LISTED).join('; ')}${more}`);
    }
    let withoutMaterial = 0;
    for (const mesh of gltf.meshes ?? []) {
        for (const primitive of mesh.primitives ?? []) {
            if (primitive.material === undefined) withoutMaterial++;
        }
    }
    if (withoutMaterial > 0) {
        problems.push(`${withoutMaterial} mesh primitive(s) have no material, so they get glTF's default material, which is fully metal`);
    }
    return problems;
}

/** What to add to a mismatch message for a file a conversion makes. */
function conversionCause(convert: AssetConvert | null | undefined): string {
    if (!convert) return '';
    return (
        `. This file is made by a conversion, so a different Blender version (this one needs ${convert.blenderVersion}) or a changed ` +
        `conversion script is the likely cause. If the change is intended, re-record the hashes (--hash, and --record-glb-views for ` +
        `.glb outputs marked reproducible false)`
    );
}

/**
 * Checks every output. Returns one line per problem, or an empty list when all
 * outputs are present and verified. `repoRoot` is where `bufferViewHashes`
 * paths start (tests pass their own folder).
 */
export async function checkOutputs(
    assetDirectory: string,
    outputs: readonly AssetOutput[],
    log: Log,
    convert?: AssetConvert | null,
    options: { repoRoot?: string } = {},
): Promise<string[]> {
    const problems: string[] = [];
    for (const output of outputs) {
        const path = join(assetDirectory, output.path);
        if (!(await fileExists(path))) {
            problems.push(`missing: ${output.path}`);
            continue;
        }
        // Before the hash checks, and whatever they find: matching hashes only say the file is the recorded one,
        // and a recorded file can be wrong (see AssetOutput.dielectric).
        if (output.dielectric === true) {
            try {
                problems.push(...metalProblems((await readGlbJson(path)) as GltfJsonForMaterials).map((problem) => `${output.path}: ${problem}`));
            } catch (error) {
                problems.push(`${output.path}: ${error instanceof Error ? error.message : String(error)}`);
            }
        }
        const size = (await stat(path)).size;
        if (size !== output.bytes) {
            problems.push(`size differs: ${output.path} (expected ${output.bytes} bytes, got ${size})${conversionCause(convert)}`);
            continue;
        }
        if (output.reproducible !== false) {
            const actual = await sha256OfFile(path);
            if (!sameHash(actual, output.sha256)) {
                problems.push(`sha256 differs: ${output.path} (expected ${output.sha256}, got ${actual})${conversionCause(convert)}`);
            }
            continue;
        }

        let structure: GlbStructure;
        let record: GlbViewRecord;
        try {
            structure = await readGlbStructure(path);
            record = await readGlbViewRecord(join(options.repoRoot ?? repoRoot, output.bufferViewHashes ?? ''));
        } catch (error) {
            problems.push(`${output.path}: ${error instanceof Error ? error.message : String(error)}`);
            continue;
        }
        const structural = checkGlbStructure(structure, output, record);
        if (structural.length > 0) {
            problems.push(...structural.map((problem) => `${output.path}: ${problem}${conversionCause(convert)}`));
        } else if (!sameHash(structure.fileSha256, output.sha256)) {
            const pinned = record.bufferViews.filter((hash) => hash !== null).length;
            const varying = (output.varyingAttributes ?? []).join(', ') || 'nothing';
            log(
                `  note: ${output.path} is not bit-reproducible; its size, JSON chunk and ${pinned} of ${record.bufferViews.length} bufferViews ` +
                    `match exactly (the rest hold only ${varying}). Whole-file sha256 ${structure.fileSha256} (information only)`,
            );
        }
    }
    return problems;
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
        const total = download.bytes;

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

        if (received !== download.bytes) {
            throw new Error(`download is ${received} bytes but sources.json says ${download.bytes}`);
        }
        const actual = hash.digest('hex');
        if (!sameHash(actual, download.sha256)) {
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
    /** Once every output is written, how long the process gets to exit by itself. */
    graceMs: number;
    /** How often to look at the outputs. */
    pollMs: number;
}
export const DEFAULT_CONVERSION_LIMITS: ConversionLimits = { timeoutMs: 60 * 60_000, graceMs: 15_000, pollMs: 2_000 };

export type ConversionEnd = 'exited' | 'stopped-after-outputs' | 'timed-out';

export interface ConversionResult {
    end: ConversionEnd;
    /** The exit code, or null when the process ended by a signal. */
    code: number | null;
    signal: string | null;
}

/**
 * The exit code the guard's kill gives the process on Windows. Programs that
 * end by themselves use small codes or NTSTATUS values (like 0xC0000005), never
 * this one, so it shows the guard ended the process. Elsewhere the guard's
 * kill shows as the signal SIGKILL instead.
 */
export const GUARD_KILL_EXIT_CODE = 0x0dd61dea;

/** True when an exit status carries the guard's kill (see GUARD_KILL_EXIT_CODE), not an ending of the process's own. */
export function endedByGuardKill(code: number | null, signal: string | null): boolean {
    return code === GUARD_KILL_EXIT_CODE || (code === null && signal === 'SIGKILL');
}

/**
 * PowerShell that ends a process tree on Windows: the descendants first, then
 * the process itself with exit code GUARD_KILL_EXIT_CODE (Win32_Process.Terminate
 * sets the exit code to its Reason; taskkill always leaves 1, which is also
 * what Blender returns for a Python error). A child must be younger than its
 * parent, so a reused process id cannot pull in an unrelated process.
 * Exits 0 when the process was ended by this script.
 */
const WINDOWS_KILL_TREE_SCRIPT = [
    "$ErrorActionPreference = 'Stop'",
    '$all = @(Get-CimInstance Win32_Process)',
    '$root = $all | Where-Object { $_.ProcessId -eq [uint32]$env:DDGI_KILL_PID }',
    'if ($null -eq $root) { exit 1 }',
    'function Get-Descendants($parent) {',
    '    foreach ($candidate in $all) {',
    '        $isChild = $candidate.ParentProcessId -eq $parent.ProcessId -and $candidate.ProcessId -ne $parent.ProcessId',
    '        if ($isChild -and $candidate.CreationDate -ge $parent.CreationDate) {',
    '            Get-Descendants $candidate',
    '            $candidate',
    '        }',
    '    }',
    '}',
    'foreach ($descendant in @(Get-Descendants $root)) {',
    '    try { $null = Invoke-CimMethod -InputObject $descendant -MethodName Terminate -Arguments @{ Reason = [uint32]1 } } catch { }',
    '}',
    '$result = Invoke-CimMethod -InputObject $root -MethodName Terminate -Arguments @{ Reason = [uint32]$env:DDGI_KILL_CODE }',
    'exit $result.ReturnValue',
].join('\n');

function runQuietly(command: string, args: string[], env?: NodeJS.ProcessEnv): Promise<number> {
    return new Promise((resolvePromise) => {
        const child = spawn(command, args, { stdio: 'ignore', env });
        child.on('error', () => resolvePromise(-1));
        child.on('close', (code) => resolvePromise(code ?? -1));
    });
}

/**
 * Stops a process and everything it started, marking the process's exit
 * status so `endedByGuardKill` can recognise it. If the marked kill cannot run
 * on Windows (no PowerShell), taskkill is the fallback: its exit code 1 is not
 * the mark, so the conversion then counts as failed rather than passing on an
 * exit status nobody can read.
 */
async function killProcessTree(child: { pid?: number; kill?: (signal?: NodeJS.Signals) => boolean }): Promise<void> {
    if (child.pid === undefined) return;
    if (process.platform === 'win32') {
        const env = { ...process.env, DDGI_KILL_PID: String(child.pid), DDGI_KILL_CODE: String(GUARD_KILL_EXIT_CODE) };
        const marked = await runQuietly('powershell.exe', ['-NoProfile', '-NonInteractive', '-Command', WINDOWS_KILL_TREE_SCRIPT], env);
        if (marked !== 0) await runQuietly('taskkill', ['/PID', String(child.pid), '/T', '/F']);
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
    on(event: 'error', listener: (error: Error) => void): unknown;
    on(event: 'exit', listener: () => void): unknown;
    on(event: 'close', listener: (code: number | null, signal: NodeJS.Signals | null) => void): unknown;
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
 * run began) at its recorded size, it gets `graceMs` to exit and is then
 * killed. A process that neither exits nor finishes its outputs is killed after
 * `timeoutMs`. Its stdin is closed so it can never wait for input. The outputs'
 * contents are not checked here; the caller's final check does that.
 *
 * What is guaranteed about the result:
 * - `stopped-after-outputs` or `timed-out` is reported only when the guard
 *   decided to stop the process and its exit status carries the guard's kill
 *   (GUARD_KILL_EXIT_CODE on Windows, SIGKILL elsewhere; see `endedByGuardKill`).
 * - Every other ending is `exited` with the real exit code and signal. That
 *   includes a process that exits by itself in the moment between the guard
 *   deciding to stop it and the kill landing: an exit status is set once, by
 *   whichever comes first, so its own code survives and is reported. The
 *   caller fails on any code other than 0 (`conversionFailure`).
 * - If the guard's kill does not end the process within 5 s, the promise
 *   rejects instead of guessing.
 *
 * A conversion with no listed outputs is never stopped early, only by the timeout.
 */
export async function runConversionProcess(command: string, args: string[], options: ConversionProcessOptions): Promise<ConversionResult> {
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
        let exited = false; // the process's 'exit' event has fired
        let guardEnd: ConversionEnd | null = null; // set once the guard has decided to stop the process
        const writtenOutputs = new Set<string>();
        let polling = false;
        let graceTimer: ReturnType<typeof setTimeout> | undefined;

        const settle = (): boolean => {
            if (finished) return false;
            finished = true;
            clearInterval(poller);
            clearTimeout(deadline);
            clearTimeout(graceTimer);
            return true;
        };

        const stop = async (end: ConversionEnd): Promise<void> => {
            // A process that has already exited is not ours to stop; its own exit status is the answer.
            if (finished || exited || guardEnd) return;
            guardEnd = end;
            await killTree(child);
            setTimeout(() => {
                if (settle()) {
                    reject(new Error(`the conversion process (pid ${child.pid}) was killed but did not end within 5 s; look for a leftover Blender process`));
                }
            }, 5_000).unref();
        };

        child.on('error', (error) => {
            if (settle()) reject(error);
        });
        child.on('exit', () => {
            exited = true;
            clearTimeout(graceTimer);
        });
        child.on('close', (code, signal) => {
            // The guard gets the credit only when the exit status carries its kill; anything else is the process's own ending.
            const end = guardEnd !== null && endedByGuardKill(code, signal) ? guardEnd : 'exited';
            if (settle()) resolvePromise({ end, code, signal });
        });

        const deadline = setTimeout(() => void stop('timed-out'), limits.timeoutMs);

        const poller = setInterval(() => {
            if (polling || finished || exited || guardEnd) return;
            polling = true;
            void (async () => {
                for (const output of options.outputs) {
                    if (writtenOutputs.has(output.path)) continue;
                    try {
                        const info = await stat(join(options.assetDirectory, output.path));
                        if (info.mtimeMs < startedAt) continue; // left over from an earlier run
                        if (info.size !== output.bytes) continue; // still being written
                        writtenOutputs.add(output.path);
                    } catch {
                        // Not there yet.
                    }
                }
                const allWritten = options.outputs.length > 0 && writtenOutputs.size === options.outputs.length;
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

/** Null when a conversion run counts as a success, otherwise the message to fail with. */
export function conversionFailure(result: ConversionResult, script: string, limits: ConversionLimits = DEFAULT_CONVERSION_LIMITS): string | null {
    switch (result.end) {
        case 'timed-out':
            return `Blender was still running after ${limits.timeoutMs / 60_000} minutes without finishing the outputs, so it was stopped (${script})`;
        case 'stopped-after-outputs':
            return null;
        case 'exited':
            if (result.code === 0) return null;
            return `Blender ${result.code !== null ? `exited with code ${result.code}` : `was ended by signal ${result.signal}`} while running ${script}`;
    }
}

/** Size and modification time of each listed file, so a later look can tell whether anything wrote to it. */
export async function fileFingerprints(directory: string, relativePaths: readonly string[]): Promise<Map<string, string>> {
    const fingerprints = new Map<string, string>();
    for (const relativePath of relativePaths) {
        try {
            const info = await stat(join(directory, relativePath));
            fingerprints.set(relativePath, `${info.size} bytes, modified ${new Date(info.mtimeMs).toISOString()}`);
        } catch {
            fingerprints.set(relativePath, 'missing');
        }
    }
    return fingerprints;
}

/**
 * Runs the asset's convert step. `inputs` are the asset-folder paths that came
 * from the downloads and the unpacked archive: hard links into the shared
 * cache. The run fails if any of them changed, because the cache (and any other
 * asset sharing it) would have changed with it.
 */
async function runConvert(asset: AssetEntry, assetDirectory: string, inputs: readonly string[], limits?: ConversionLimits): Promise<void> {
    const convert = asset.convert;
    if (!convert) return;
    const script = resolve(repoRoot, convert.script);
    if (!(await fileExists(script))) throw new Error(`convert script not found: ${convert.script}`);
    const blender = await findBlender(convert.blenderVersion);
    const args = substitutePlaceholders(convert.args, { dir: assetDirectory, id: asset.id, repo: repoRoot });
    console.log(`  converting with Blender ${blender.version} at ${blender.path}`);

    const before = await fileFingerprints(assetDirectory, inputs);
    const result = await runConversionProcess(blender.path, ['--background', '--factory-startup', '--python-exit-code', '1', '--python', script, '--', ...args], {
        cwd: assetDirectory,
        assetDirectory,
        outputs: asset.outputs,
        limits,
    });
    const failure = conversionFailure(result, convert.script, limits);
    if (failure) throw new Error(failure);
    if (result.end === 'stopped-after-outputs') console.log('  all outputs are written; Blender did not exit by itself, so it was stopped');

    const after = await fileFingerprints(assetDirectory, inputs);
    const changed = inputs.filter((path) => before.get(path) !== after.get(path));
    if (changed.length > 0) {
        throw new Error(
            `${convert.script} changed its input files (${changed.slice(0, 5).join(', ')}${changed.length > 5 ? ` and ${changed.length - 5} more` : ''}). ` +
                `Inputs are hard links into assets/downloads/.cache, so the cache changed too. Make the script write new files only, ` +
                `then run again with --force to restore the inputs.`,
        );
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
 * Puts one downloaded file at `destination`. Downloads go through a content
 * cache (`<downloads>/.cache/<sha256>`), so two assets that share a download
 * fetch it once; the asset folder gets a hard link or a copy. A cached file is
 * hashed again before it is used.
 */
export async function obtainDownload(
    download: AssetDownload,
    destination: string,
    cacheDirectory: string,
    force: boolean,
    log: Log,
): Promise<void> {
    const expected = download.sha256.toLowerCase();
    const cachePath = join(cacheDirectory, expected);

    if (!force && (await fileExists(destination)) && (await sha256OfFile(destination)) === expected) {
        log(`  ${download.file} is already downloaded`);
        if (!(await fileExists(cachePath))) {
            await mkdir(cacheDirectory, { recursive: true });
            await linkOrCopy(destination, cachePath);
        }
        return;
    }

    if (!force && (await fileExists(cachePath)) && (await sha256OfFile(cachePath)) === expected) {
        log(`  ${download.file}: taking it from the download cache`);
    } else {
        await mkdir(cacheDirectory, { recursive: true });
        await downloadFile(download, cachePath);
        log(`  verified sha256 of ${download.file}`);
    }
    await linkOrCopy(cachePath, destination);
}

/** One file of an unpacked archive, as recorded when the unpack finished. */
interface UnpackedFile {
    /** Relative to the unpack folder, with "/" separators. */
    path: string;
    bytes: number;
    mtimeMs: number;
}

/** Every file under `root`, sorted by path. */
async function listFiles(root: string, prefix = ''): Promise<UnpackedFile[]> {
    const files: UnpackedFile[] = [];
    for (const entry of await readdir(join(root, prefix), { withFileTypes: true })) {
        const relativePath = prefix ? `${prefix}/${entry.name}` : entry.name;
        if (entry.isDirectory()) {
            files.push(...(await listFiles(root, relativePath)));
        } else {
            const info = await stat(join(root, relativePath));
            files.push({ path: relativePath, bytes: info.size, mtimeMs: info.mtimeMs });
        }
    }
    return files.sort((a, b) => (a.path < b.path ? -1 : a.path > b.path ? 1 : 0));
}

function sameFileList(a: readonly UnpackedFile[], b: readonly UnpackedFile[]): boolean {
    return a.length === b.length && a.every((file, index) => file.path === b[index].path && file.bytes === b[index].bytes && file.mtimeMs === b[index].mtimeMs);
}

async function readUnpackRecord(path: string): Promise<UnpackedFile[] | null> {
    try {
        const parsed: unknown = JSON.parse(await readFile(path, 'utf8'));
        return isRecord(parsed) && Array.isArray(parsed.files) ? (parsed.files as UnpackedFile[]) : null;
    } catch {
        return null;
    }
}

/** Removes files under `directory` that are not in `wanted` (relative "/" paths), then any folders left empty. */
async function removeFilesNotIn(directory: string, wanted: ReadonlySet<string>, log: Log, prefix = ''): Promise<void> {
    for (const entry of await readdir(join(directory, prefix), { withFileTypes: true })) {
        const relativePath = prefix ? `${prefix}/${entry.name}` : entry.name;
        if (entry.isDirectory()) {
            await removeFilesNotIn(directory, wanted, log, relativePath);
            if ((await readdir(join(directory, relativePath))).length === 0) await rm(join(directory, relativePath), { recursive: true });
        } else if (!wanted.has(relativePath)) {
            log(`  removing ${relativePath}: it is not in the archive, a download or an output`);
            await rm(join(directory, relativePath), { force: true });
        }
    }
}

/**
 * Unpacks a zip into the asset folder and returns the unpacked files' paths.
 *
 * The unpacked files are kept in `<cache>/extracted/<sha256>/` and hard-linked
 * into the asset folder, so two assets that share one zip (the two Bistro
 * halves) unpack it once. An unpack is built under a temporary name; once it
 * is complete it is renamed, and then a file list with every file's size and
 * modification time is written next to it (`<sha256>.files.json`). A cached
 * unpack is reused only when that list exists and still matches the folder
 * exactly; otherwise it is unpacked again.
 *
 * Afterwards the asset folder holds exactly the archive's files, the
 * downloads and the listed outputs (`keep`): anything else, such as a file
 * from an older version of the archive, is removed. Outputs are made again by
 * the convert step that follows.
 */
export async function extractWithCache(
    download: AssetDownload,
    archivePath: string,
    assetDirectory: string,
    cacheDirectory: string,
    keep: ReadonlySet<string>,
    log: Log,
    extract: (archive: string, destination: string) => Promise<void> = extractZip,
): Promise<string[]> {
    const cached = join(cacheDirectory, 'extracted', download.sha256.toLowerCase());
    const recordPath = `${cached}.files.json`;
    const recorded = await readUnpackRecord(recordPath);
    let files = recorded && (await directoryExists(cached)) ? await listFiles(cached) : null;

    if (recorded && files && sameFileList(files, recorded)) {
        log(`  ${download.file}: taking the unpacked files from the cache`);
    } else {
        if (await directoryExists(cached)) log(`  ${download.file}: the unpacked copy in the cache is incomplete or changed; unpacking again`);
        else log(`  extracting ${download.file}`);
        await rm(recordPath, { force: true });
        await rm(cached, { recursive: true, force: true });
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
        files = await listFiles(cached);
        // Written last: its presence is the mark of a complete unpack.
        await writeFile(recordPath, JSON.stringify({ archive: download.file, files }));
    }

    await mkdir(assetDirectory, { recursive: true });
    for (const file of files) await linkOrCopy(join(cached, file.path), join(assetDirectory, file.path));
    const unpacked = files.map((file) => file.path);
    await removeFilesNotIn(assetDirectory, new Set([...unpacked, ...keep]), log);
    return unpacked;
}

async function directoryExists(path: string): Promise<boolean> {
    try {
        return (await stat(path)).isDirectory();
    } catch {
        return false;
    }
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
    const cacheDirectory = join(downloadsDirectory, '.cache');
    await mkdir(assetDirectory, { recursive: true });

    if (!options.force && (await checkOutputs(assetDirectory, asset.outputs, () => undefined, asset.convert)).length === 0) {
        log(`${asset.id}: up to date`);
        return 'up-to-date';
    }

    log(`${asset.id}: ${asset.title} by ${asset.author} (${asset.license}${asset.redistributable ? '' : ', not redistributable: stays local'})`);

    const downloads = downloadsOf(asset);
    for (const download of downloads) {
        await obtainDownload(download, join(assetDirectory, download.file), cacheDirectory, options.force, log);
    }

    const inputs = downloads.map((download) => download.file);
    if (asset.extract === 'zip') {
        const keep = new Set([...inputs, ...asset.outputs.map((output) => output.path)]);
        inputs.push(...(await extractWithCache(downloads[0], join(assetDirectory, downloads[0].file), assetDirectory, cacheDirectory, keep, log)));
    }
    await runConvert(asset, assetDirectory, inputs);

    const problems = await checkOutputs(assetDirectory, asset.outputs, log, asset.convert);
    if (problems.length > 0) {
        throw new Error(`outputs are not right after fetching:\n    ${problems.join('\n    ')}`);
    }
    log(`  ${asset.id}: ${asset.outputs.length} output file(s) verified`);
    return 'fetched';
}

// ---------------------------------------------------------------------------
// Command line
// ---------------------------------------------------------------------------

const helpText = `Usage: pnpm assets:fetch [--only <id>[,<id>]] [--force] [--verify-only] [--sources <path>]
       node tools/assets/fetch.ts --hash <file>              print the sha256 and size to put in sources.json
       node tools/assets/fetch.ts --record-glb-views <id>    write the bufferView hashes of the asset's
                                                             non-reproducible .glb outputs, from the files on disk

--verify-only checks the files already on disk against sources.json (sizes, hashes,
and the pieces of non-reproducible .glb files) without downloading or converting anything.

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
        const problems = await checkOutputs(join(downloadsRoot, asset.id), asset.outputs, (message) => console.log(message), asset.convert);
        if (problems.length === 0) {
            console.log(`${asset.id}: OK, ${asset.outputs.length} output file(s) verified`);
        } else {
            failed.push(asset.id);
            console.error(`${asset.id}: FAILED\n    ${problems.join('\n    ')}`);
        }
    }
    console.log(failed.length === 0 ? '\nAll outputs verified.' : `\nFailed: ${failed.join(', ')}`);
    return failed.length === 0 ? 0 : 1;
}

/**
 * Writes the bufferView record of each non-reproducible .glb output of the
 * named assets, from the files on disk. The file's size, BIN length and JSON
 * chunk must already match sources.json: this records the data of a known
 * scene, it does not bless a new one.
 */
async function recordGlbViews(assets: readonly AssetEntry[]): Promise<number> {
    let failures = 0;
    for (const asset of assets) {
        const glbOutputs = asset.outputs.filter((output) => output.reproducible === false);
        if (glbOutputs.length === 0) console.log(`${asset.id}: has no output marked reproducible false; nothing to record`);
        for (const output of glbOutputs) {
            const path = join(downloadsRoot, asset.id, output.path);
            const structure = await readGlbStructure(path);
            const size = (await stat(path)).size;
            if (size !== output.bytes || structure.binBytes !== output.binBytes || !sameHash(structure.jsonSha256, output.jsonSha256 ?? '')) {
                console.error(`${asset.id}: ${output.path} does not match the size, BIN length and JSON chunk in sources.json; fix those first`);
                failures++;
                continue;
            }
            const record = makeGlbViewRecord(asset.id, output, structure);
            const recordPath = join(repoRoot, output.bufferViewHashes ?? '');
            await mkdir(dirname(recordPath), { recursive: true });
            await writeFile(recordPath, formatGlbViewRecord(record));
            const pinned = record.bufferViews.filter((hash) => hash !== null).length;
            console.log(`${asset.id}: wrote ${output.bufferViewHashes}: ${pinned} of ${record.bufferViews.length} bufferViews pinned`);
            if (structure.nonZeroPaddingBytes > 0) {
                console.error(`${asset.id}: warning: ${structure.nonZeroPaddingBytes} padding byte(s) are not zero, so the check will fail`);
                failures++;
            }
        }
    }
    return failures === 0 ? 0 : 1;
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
    const approvalProblems = checkLicenseApprovals(sources.assets, await readFile(join(repoRoot, DECISIONS_FILE), 'utf8'));
    if (approvalProblems.length > 0) throw new Error(`${sourcesPath} has license approvals that point nowhere:\n  ${approvalProblems.join('\n  ')}`);

    if (sources.assets.length === 0) {
        console.log('No assets are listed in sources.json yet, so there is nothing to download.');
        return 0;
    }

    if (options.recordGlbViews.length > 0) return recordGlbViews(selectAssets(sources.assets, options.recordGlbViews));
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
