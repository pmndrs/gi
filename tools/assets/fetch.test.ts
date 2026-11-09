import { createHash } from 'node:crypto';
import { EventEmitter } from 'node:events';
import { createServer, type Server } from 'node:http';
import type { AddressInfo, Socket } from 'node:net';
import { appendFile, link, mkdir, mkdtemp, readdir, readFile, rm, utimes, writeFile } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { afterAll, beforeAll, describe, expect, it, vi } from 'vitest';
import {
    checkGlbStructure,
    checkLicenseApprovals,
    checkOutputs,
    chooseDownloader,
    conversionFailure,
    CURL_HOSTS,
    curlArguments,
    DEFAULT_DOWNLOAD_TIMEOUTS,
    downloadFile,
    downloadsOf,
    generatedConvert,
    downloadWithCurl,
    endedByGuardKill,
    extractWithCache,
    fetchAsset,
    fileFingerprints,
    findBlender,
    findSystemCurl,
    formatBytes,
    formatGlbViewRecord,
    GUARD_KILL_EXIT_CODE,
    isAllowedLicense,
    isSafeRelativePath,
    makeGlbViewRecord,
    metalProblems,
    parseArgs,
    parseBlenderVersion,
    readGlbJson,
    readGlbStructure,
    readGlbViewRecord,
    runConversionProcess,
    selectAssets,
    sha256OfFile,
    sortBlenderFolders,
    substitutePlaceholders,
    systemCurlPath,
    validateSources,
    versionMatches,
    type AssetEntry,
    type AssetOutput,
    type SourcesFile,
} from './fetch';

const sha256 = (data: string | Buffer) => createHash('sha256').update(data).digest('hex');
const bytesOf = (text: string) => Buffer.byteLength(text);
const convertWith = (blenderVersion = '4.5.0') => ({ tool: 'blender' as const, script: 'tools/assets/convert/x.py', args: [], blenderVersion });

function asset(overrides: Partial<AssetEntry> = {}): AssetEntry {
    return {
        id: 'test-asset',
        title: 'Test',
        author: 'Someone',
        sourceUrl: 'https://example.com/test',
        license: 'CC0-1.0',
        attribution: 'Someone, CC0',
        redistributable: true,
        download: { url: 'https://example.com/test.glb', file: 'test.glb', bytes: 10, sha256: 'a'.repeat(64) },
        extract: null,
        convert: null,
        outputs: [{ path: 'test.glb', sha256: 'b'.repeat(64), bytes: 10 }],
        ...overrides,
    };
}

/** An entry as it might be written in sources.json, before validation: any field may be wrong or missing. */
function rawAsset(overrides: Record<string, unknown> = {}): Record<string, unknown> {
    return { ...asset(), ...overrides };
}

describe('validateSources', () => {
    it('accepts an empty asset list', () => {
        expect(validateSources({ assets: [] })).toEqual([]);
    });

    it('accepts a full entry', () => {
        const entry = asset({
            licenseUrl: 'https://example.com/license',
            download: { url: 'https://example.com/a.zip', file: 'a.zip', bytes: 100, sha256: 'a'.repeat(64) },
            extract: 'zip',
            convert: { ...convertWith(), args: ['{dir}'] },
            outputs: [{ path: 'main.glb', sha256: 'B'.repeat(64), bytes: 20 }],
        });
        expect(validateSources({ assets: [entry] })).toEqual([]);
    });

    it('accepts several downloads, with subfolders in the file names', () => {
        const entry = asset({
            download: [
                { url: 'https://example.com/tree.gltf', file: 'tree.gltf', bytes: 5, sha256: 'd'.repeat(64) },
                { url: 'https://example.com/bark.jpg', file: 'textures/bark.jpg', bytes: 10, sha256: 'a'.repeat(64) },
            ],
            outputs: [{ path: 'tree.gltf', sha256: 'e'.repeat(64), bytes: 5 }],
        });
        expect(validateSources({ assets: [entry] })).toEqual([]);
        expect(downloadsOf(entry).map((download) => download.file)).toEqual(['tree.gltf', 'textures/bark.jpg']);
        expect(downloadsOf(asset())).toHaveLength(1);
    });

    it('rejects an empty download list, repeated files, unsafe file paths and extract with several downloads', () => {
        const base = { url: 'https://example.com/a', file: 'a.bin', bytes: 1, sha256: 'a'.repeat(64) };
        expect(validateSources({ assets: [rawAsset({ download: [] })] }).join('\n')).toContain('empty array');
        expect(validateSources({ assets: [rawAsset({ download: [base, base] })] }).join('\n')).toContain('listed twice');
        expect(validateSources({ assets: [rawAsset({ download: [{ ...base, file: '../a.bin' }] })] }).join('\n')).toContain('download[0].file');
        const text = validateSources({ assets: [rawAsset({ download: [base, { ...base, file: 'b.bin' }], extract: 'zip' })] }).join('\n');
        expect(text).toContain('only works with a single download');
    });

    it('lets a skipped asset be unfinished, but needs a reason', () => {
        expect(validateSources({ assets: [{ id: 'sponza-khronos', skip: true, skipReason: 'License question is open.' }] })).toEqual([]);
        expect(validateSources({ assets: [{ id: 'sponza-khronos', skip: true }] }).join('\n')).toContain('skipReason');
    });

    it('requires sha256 and bytes on every download and output, and says which one is missing', () => {
        const bare = rawAsset({
            download: [
                { url: 'https://example.com/a', file: 'a.bin' },
                { url: 'https://example.com/b', file: 'b.bin', bytes: 5, sha256: 'a'.repeat(64) },
            ],
            outputs: [{ path: 'a.bin' }, { path: 'b.bin', sha256: 'c'.repeat(64) }],
        });
        const text = validateSources({ assets: [bare] }).join('\n');
        expect(text).toContain('download[0].bytes: required');
        expect(text).toContain('download[0].sha256: required');
        expect(text).not.toContain('download[1]');
        expect(text).toContain('outputs[0].sha256: required');
        expect(text).toContain('outputs[0].bytes: required');
        expect(text).toContain('outputs[1].bytes: required');
        expect(text).not.toContain('outputs[1].sha256');
    });

    it('requires the redistributable field', () => {
        const { redistributable: _removed, ...rest } = asset();
        expect(validateSources({ assets: [rest] }).join('\n')).toContain('redistributable: must be present');
    });

    it('does not check the license, hashes or bytes of a skipped asset', () => {
        const skipped = { id: 'sponza-khronos', skip: true, skipReason: 'Cryengine license.', license: 'CRYENGINE Limited License Agreement' };
        expect(validateSources({ assets: [skipped] })).toEqual([]);
    });

    it('rejects things that are not a sources file', () => {
        expect(validateSources(null)).toHaveLength(1);
        expect(validateSources({})).toHaveLength(1);
    });

    it('names the asset and the field for each problem', () => {
        const bad = rawAsset({ id: 'Bad Id', download: { url: 'ftp://x', file: '../x.zip', sha256: 'zz' }, extract: 'rar', outputs: [] });
        const text = validateSources({ assets: [bad] }).join('\n');
        expect(text).toContain('assets[0] (Bad Id).id');
        expect(text).toContain('download.url');
        expect(text).toContain('download.file');
        expect(text).toContain('download.sha256');
        expect(text).toContain('extract');
        expect(text).toContain('outputs');
    });

    it('rejects duplicate ids and output paths that escape the folder', () => {
        const first = asset();
        const second = rawAsset({ outputs: [{ path: '../elsewhere.glb', sha256: 'b'.repeat(64), bytes: 1 }] });
        const text = validateSources({ assets: [first, second] }).join('\n');
        expect(text).toContain('duplicate id');
        expect(text).toContain('outputs[0].path');
    });
});

describe('licenses', () => {
    const approval = { record: 'docs/DECISIONS.md', date: '2026-11-04', title: 'Sponza is Intel\'s "New Sponza" (base package), download-only.' };

    it('allows only CC0 and plain CC-BY without asking', () => {
        for (const license of ['CC0-1.0', 'CC-BY-4.0', 'CC-BY-3.0']) expect(isAllowedLicense(license)).toBe(true);
        const notPlain = ['CC-BY-SA-4.0', 'CC-BY-NC-4.0', 'CC-BY-4.0+intel-preface', 'CRYENGINE Limited License Agreement', 'MIT', 'cc0-1.0', ''];
        for (const license of notPlain) expect(isAllowedLicense(license), license).toBe(false);
        expect(validateSources({ assets: [asset({ license: 'CC-BY-4.0' })] })).toEqual([]);
    });

    it('needs a licenseApproval pointing at DECISIONS.md for any other license, CC-BY with extra terms included', () => {
        const text = validateSources({ assets: [asset({ license: 'CC-BY-4.0+intel-preface' })] }).join('\n');
        expect(text).toContain('"CC-BY-4.0+intel-preface" is not on the allow list');
        expect(text).toContain('licenseApproval');
        expect(text).toContain('docs/DECISIONS.md');
        expect(text).toContain('Ask Dennis');
        expect(validateSources({ assets: [asset({ license: 'CC-BY-4.0+intel-preface', licenseApproval: approval })] })).toEqual([]);
    });

    it('checks the shape of an approval', () => {
        const wrong = { record: 'somewhere.md', date: 'last week', title: ' ' };
        const text = validateSources({ assets: [rawAsset({ license: 'Custom', licenseApproval: wrong })] }).join('\n');
        expect(text).toContain('licenseApproval: record must be "docs/DECISIONS.md"');
        expect(text).toContain('licenseApproval: date must look like');
        expect(text).toContain('licenseApproval: title must be');
        expect(validateSources({ assets: [rawAsset({ license: 'Custom', licenseApproval: 'yes' })] }).join('\n')).toContain('must be an object');
    });

    it('finds the approved entry in DECISIONS.md, by date section and exact bold title', () => {
        const decisions = [
            '# Decisions',
            '',
            '## 2026-11-04',
            '',
            '- **Something else.**',
            `- **${approval.title}**`,
            '  - details',
            '',
            '## 2026-10-30',
            '',
            '- **Older entry.**',
        ].join('\n');
        const approved = asset({ id: 'sponza-intel', license: 'CC-BY-4.0+intel-preface', licenseApproval: approval });
        expect(checkLicenseApprovals([approved], decisions)).toEqual([]);
        // The right title under the wrong date does not count.
        const wrongDate = asset({ ...approved, licenseApproval: { ...approval, date: '2026-10-30' } });
        expect(checkLicenseApprovals([wrongDate], decisions)[0]).toContain('has no entry titled');
        const missingDate = asset({ ...approved, licenseApproval: { ...approval, date: '2025-01-01' } });
        expect(checkLicenseApprovals([missingDate], decisions)[0]).toContain('does not exist');
        // Plain licenses need no entry at all.
        expect(checkLicenseApprovals([asset({ license: 'CC-BY-4.0' })], '')).toEqual([]);
    });

    it('the real sources.json is valid, and its license approvals point at real DECISIONS.md entries', async () => {
        const repo = resolve(__dirname, '..', '..');
        const sources = JSON.parse(await readFile(join(repo, 'tools', 'assets', 'sources.json'), 'utf8')) as SourcesFile;
        expect(validateSources(sources)).toEqual([]);
        expect(checkLicenseApprovals(sources.assets, await readFile(join(repo, 'docs', 'DECISIONS.md'), 'utf8'))).toEqual([]);
        // Intel's Sponza carries a use restriction on top of CC BY, so it must not pass as plain CC-BY.
        const sponza = sources.assets.find((entry) => entry.id === 'sponza-intel');
        expect(sponza && isAllowedLicense(sponza.license)).toBe(false);
        expect(sponza?.licenseApproval).toBeDefined();
    });
});

describe('isSafeRelativePath', () => {
    it.each([['main.glb', true], ['sub/dir/a.glb', true], ['../a.glb', false], ['a/../../b', false], ['/abs', false], ['C:\\x', false], ['', false]])(
        '%s -> %s',
        (path, expected) => expect(isSafeRelativePath(path)).toBe(expected),
    );
});

describe('parseArgs', () => {
    it('reads flags', () => {
        expect(parseArgs([])).toEqual({ only: [], force: false, sourcesPath: null, hash: null, recordGlbViews: [], verifyOnly: false, help: false });
        expect(parseArgs(['--verify-only']).verifyOnly).toBe(true);
        expect(parseArgs(['--record-glb-views', 'a,b']).recordGlbViews).toEqual(['a', 'b']);
        expect(parseArgs(['--only', 'a', '--only', 'b,c', '--force', '--sources', 'x.json'])).toEqual({
            only: ['a', 'b', 'c'],
            force: true,
            sourcesPath: 'x.json',
            hash: null,
            recordGlbViews: [],
            verifyOnly: false,
            help: false,
        });
    });

    it('rejects unknown flags and flags missing a value', () => {
        expect(() => parseArgs(['--nope'])).toThrow('Unknown argument "--nope"');
        expect(() => parseArgs(['--only'])).toThrow('--only needs a value');
        expect(() => parseArgs(['--only', '--force'])).toThrow('--only needs a value');
    });
});

describe('selectAssets', () => {
    const assets = [asset({ id: 'one' }), asset({ id: 'two' })];

    it('returns everything with no filter and the named assets with one', () => {
        expect(selectAssets(assets, [])).toHaveLength(2);
        expect(selectAssets(assets, ['two']).map((a) => a.id)).toEqual(['two']);
    });

    it('lists the known ids when one is unknown', () => {
        expect(() => selectAssets(assets, ['three'])).toThrow('Unknown asset id: three. Known ids: one, two');
        expect(() => selectAssets([], ['three'])).toThrow('(none listed yet)');
    });
});

describe('small helpers', () => {
    it('substitutes placeholders everywhere in each argument', () => {
        expect(substitutePlaceholders(['{dir}/in.zip', '--id={id}', 'plain', '{repo}/{id}'], { dir: '/d', id: 'x', repo: '/r' })).toEqual([
            '/d/in.zip',
            '--id=x',
            'plain',
            '/r/x',
        ]);
    });

    it('sorts Blender folders newest first by number, not by text', () => {
        expect(sortBlenderFolders(['Blender 4.5', 'Blender 5.1', 'Blender 10.0', 'Other', 'Blender 5.0'])).toEqual([
            'Blender 10.0',
            'Blender 5.1',
            'Blender 5.0',
            'Blender 4.5',
        ]);
    });

    it('formats byte counts', () => {
        expect(formatBytes(0)).toBe('0 B');
        expect(formatBytes(1536)).toBe('1.5 KB');
        expect(formatBytes(5 * 1024 ** 3)).toBe('5.0 GB');
    });
});

describe('fetching from a local server', () => {
    const body = 'pretend this is a model';
    const sharedBody = 'a shared download, as with the two Bistro halves';
    let server: Server;
    let baseUrl: string;
    let workDirectory: string;
    let requests = 0;

    beforeAll(async () => {
        workDirectory = await mkdtemp(join(tmpdir(), 'ddgi-fetch-'));
        server = createServer((request, response) => {
            requests++;
            if (request.url === '/model.glb') {
                response.writeHead(200, { 'content-type': 'application/octet-stream', 'content-length': bytesOf(body) });
                response.end(body);
            } else if (request.url === '/shared.bin') {
                response.writeHead(200, { 'content-length': bytesOf(sharedBody) });
                response.end(sharedBody);
            } else {
                response.writeHead(404).end();
            }
        });
        await new Promise<void>((resolvePromise) => server.listen(0, '127.0.0.1', resolvePromise));
        baseUrl = `http://127.0.0.1:${(server.address() as AddressInfo).port}`;
        vi.spyOn(console, 'log').mockImplementation(() => undefined);
    });

    afterAll(async () => {
        server.close();
        await rm(workDirectory, { recursive: true, force: true });
        vi.restoreAllMocks();
    });

    const modelDownload = (file = 'model.glb') => ({ url: `${baseUrl}/model.glb`, file, bytes: bytesOf(body), sha256: sha256(body) });
    const sharedDownload = (file = 'shared.bin') => ({ url: `${baseUrl}/shared.bin`, file, bytes: bytesOf(sharedBody), sha256: sha256(sharedBody) });
    const entry = (overrides: Partial<AssetEntry> = {}) =>
        asset({ download: modelDownload(), outputs: [{ path: 'model.glb', sha256: sha256(body), bytes: bytesOf(body) }], ...overrides });

    it('downloads, verifies, then skips on the second run', async () => {
        const options = { force: false, downloadsDirectory: workDirectory };
        const before = requests;
        expect(await fetchAsset(entry(), options)).toBe('fetched');
        expect(await readFile(join(workDirectory, 'test-asset', 'model.glb'), 'utf8')).toBe(body);
        expect(requests).toBe(before + 1);

        expect(await fetchAsset(entry(), options)).toBe('up-to-date');
        expect(requests).toBe(before + 1);
    });

    it('restores a corrupted file from the download cache without downloading again', async () => {
        const options = { force: false, downloadsDirectory: workDirectory };
        // The asset file is a hard link to the cache, so replace it rather than write through it.
        await rm(join(workDirectory, 'test-asset', 'model.glb'));
        await writeFile(join(workDirectory, 'test-asset', 'model.glb'), 'corrupted');
        const before = requests;
        expect(await fetchAsset(entry(), options)).toBe('fetched');
        expect(requests).toBe(before);
        expect(await readFile(join(workDirectory, 'test-asset', 'model.glb'), 'utf8')).toBe(body);
    });

    it('downloads again when the cached copy itself was changed', async () => {
        const directory = await mkdtemp(join(tmpdir(), 'ddgi-cache-bad-'));
        const options = { force: false, downloadsDirectory: directory };
        expect(await fetchAsset(entry(), options)).toBe('fetched');
        // Writing through the hard link changes the cache too: exactly what convert scripts must never do.
        await appendFile(join(directory, 'test-asset', 'model.glb'), ' and more');
        expect(await readFile(join(directory, '.cache', sha256(body)), 'utf8')).toBe(`${body} and more`);
        const before = requests;
        expect(await fetchAsset(entry(), options)).toBe('fetched');
        expect(requests).toBe(before + 1);
        expect(await readFile(join(directory, 'test-asset', 'model.glb'), 'utf8')).toBe(body);
        await rm(directory, { recursive: true, force: true });
    });

    it('downloads again with force', async () => {
        const before = requests;
        await fetchAsset(entry(), { force: true, downloadsDirectory: workDirectory });
        expect(requests).toBe(before + 1);
    });

    it('fetches a shared download once for two assets', async () => {
        const directory = await mkdtemp(join(tmpdir(), 'ddgi-shared-'));
        const before = requests;
        const outputs = [{ path: 'shared.bin', sha256: sha256(sharedBody), bytes: bytesOf(sharedBody) }];
        const first = asset({ id: 'half-one', download: sharedDownload(), outputs });
        const second = asset({ id: 'half-two', download: sharedDownload(), outputs });
        expect(await fetchAsset(first, { force: false, downloadsDirectory: directory })).toBe('fetched');
        expect(await fetchAsset(second, { force: false, downloadsDirectory: directory })).toBe('fetched');
        expect(requests).toBe(before + 1);
        expect(await readFile(join(directory, 'half-one', 'shared.bin'), 'utf8')).toBe(sharedBody);
        expect(await readFile(join(directory, 'half-two', 'shared.bin'), 'utf8')).toBe(sharedBody);
        expect(await readFile(join(directory, '.cache', sha256(sharedBody)), 'utf8')).toBe(sharedBody);
        await rm(directory, { recursive: true, force: true });
    });

    it('fetches several files into subfolders', async () => {
        const directory = await mkdtemp(join(tmpdir(), 'ddgi-multi-'));
        const many = asset({
            id: 'tree',
            download: [modelDownload('tree.gltf'), sharedDownload('textures/bark/diffuse.bin')],
            outputs: [
                { path: 'tree.gltf', sha256: sha256(body), bytes: bytesOf(body) },
                { path: 'textures/bark/diffuse.bin', sha256: sha256(sharedBody), bytes: bytesOf(sharedBody) },
            ],
        });
        expect(await fetchAsset(many, { force: false, downloadsDirectory: directory })).toBe('fetched');
        expect(await readFile(join(directory, 'tree', 'tree.gltf'), 'utf8')).toBe(body);
        expect(await readFile(join(directory, 'tree', 'textures', 'bark', 'diffuse.bin'), 'utf8')).toBe(sharedBody);
        await rm(directory, { recursive: true, force: true });
    });

    it('reports a skipped asset and does nothing else', async () => {
        const before = requests;
        const skipped = asset({ id: 'on-hold', skip: true, skipReason: 'License question is open.' });
        expect(await fetchAsset(skipped, { force: false, downloadsDirectory: workDirectory })).toBe('skipped');
        expect(requests).toBe(before);
        expect(console.log).toHaveBeenCalledWith('on-hold: skipped: License question is open.');
    });

    it('fails clearly when the download hash is wrong, and leaves no partial file', async () => {
        const wrong = entry({ id: 'wrong-hash', download: { ...modelDownload(), sha256: sha256('something else') } });
        await expect(fetchAsset(wrong, { force: false, downloadsDirectory: workDirectory })).rejects.toThrow(/download sha256 is/);
        await expect(readFile(join(workDirectory, '.cache', `${sha256('something else')}.part`))).rejects.toThrow();
        await expect(readFile(join(workDirectory, 'wrong-hash', 'model.glb'))).rejects.toThrow();
    });

    it('fails clearly on a 404', async () => {
        const missing = entry({ id: 'gone', download: { ...modelDownload('nope.glb'), url: `${baseUrl}/nope.glb`, sha256: 'c'.repeat(64) } });
        await expect(fetchAsset(missing, { force: false, downloadsDirectory: workDirectory })).rejects.toThrow(/answered 404/);
    });

    it('fails clearly when the size differs from sources.json', async () => {
        // A folder of its own, so the right file is not simply taken from the download cache.
        const directory = await mkdtemp(join(tmpdir(), 'ddgi-size-'));
        const wrongSize = entry({ id: 'wrong-size', download: { ...modelDownload(), bytes: 5 } });
        await expect(fetchAsset(wrongSize, { force: false, downloadsDirectory: directory })).rejects.toThrow(/bytes but sources.json says 5/);
        await rm(directory, { recursive: true, force: true });
    });

    it('reports missing and mismatched outputs', async () => {
        const directory = join(workDirectory, 'check');
        await mkdir(directory, { recursive: true });
        await writeFile(join(directory, 'a.glb'), 'aaa');
        await writeFile(join(directory, 'b.glb'), 'bbb');
        const problems = await checkOutputs(
            directory,
            [
                { path: 'a.glb', sha256: sha256('aaa'), bytes: 3 },
                { path: 'b.glb', sha256: sha256('not'), bytes: 3 },
                { path: 'c.glb', sha256: sha256('c'), bytes: 1 },
            ],
            () => undefined,
        );
        expect(problems).toHaveLength(2);
        expect(problems[0]).toContain('sha256 differs: b.glb');
        expect(problems[1]).toBe('missing: c.glb');
        expect(await sha256OfFile(join(directory, 'a.glb'))).toBe(sha256('aaa'));
    });
});

describe('picking a Blender by version', () => {
    it('reads the version from --version output', () => {
        expect(parseBlenderVersion('Blender 4.5.0 LTS\n\tbuild date: 2025-06-17')).toBe('4.5.0');
        expect(parseBlenderVersion('Blender 5.1.0 (hash adfe2921d5f3 built 2026-03-17)')).toBe('5.1.0');
        expect(parseBlenderVersion('not it')).toBeNull();
    });

    it('matches the full version exactly: a patch release is a different Blender', () => {
        expect(versionMatches('4.5.0', '4.5.0')).toBe(true);
        expect(versionMatches('4.5.3', '4.5.0')).toBe(false);
        expect(versionMatches('4.5', '4.5.0')).toBe(false);
        expect(versionMatches('4.50.0', '4.5.0')).toBe(false);
        expect(versionMatches('14.5.0', '4.5.0')).toBe(false);
    });

    describe('with fake executables', () => {
        let directory: string;
        let variablePath: string;
        let pathFolder: string;
        let pathExecutable: string;
        let environment: NodeJS.ProcessEnv;
        const versions = new Map<string, string | null>();
        const readVersion = async (path: string) => versions.get(path) ?? null;

        beforeAll(async () => {
            directory = await mkdtemp(join(tmpdir(), 'ddgi-blender-'));
            pathFolder = join(directory, 'on-path');
            await mkdir(pathFolder);
            variablePath = join(directory, 'blender-from-variable.exe');
            pathExecutable = join(pathFolder, 'blender.exe');
            await writeFile(variablePath, '');
            await writeFile(pathExecutable, '');
            // ProgramFiles points at an empty folder so the machine's real installs are not found.
            environment = { BLENDER: variablePath, PATH: pathFolder, ProgramFiles: join(directory, 'nothing-here') };
        });
        afterAll(() => rm(directory, { recursive: true, force: true }));

        it('does not trust the BLENDER variable: a wrong version is skipped for the next candidate', async () => {
            versions.set(variablePath, '5.1.0');
            versions.set(pathExecutable, '4.5.0');
            const choice = await findBlender('4.5.0', { env: environment, platform: 'win32', readVersion });
            expect(choice).toEqual({ path: pathExecutable, version: '4.5.0' });
        });

        it('uses the BLENDER variable when its version matches, and skips another patch release', async () => {
            versions.set(variablePath, '4.5.0');
            versions.set(pathExecutable, '4.5.2');
            expect((await findBlender('4.5.0', { env: environment, platform: 'win32', readVersion })).path).toBe(variablePath);
            versions.set(variablePath, '4.5.2');
            versions.set(pathExecutable, '4.5.0');
            expect((await findBlender('4.5.0', { env: environment, platform: 'win32', readVersion })).path).toBe(pathExecutable);
        });

        it('takes the first runnable Blender when no version is asked for', async () => {
            versions.set(variablePath, null);
            versions.set(pathExecutable, '5.1.0');
            expect((await findBlender(undefined, { env: environment, platform: 'win32', readVersion })).version).toBe('5.1.0');
        });

        it('says which version is needed, what was found, and that the hashes would need re-recording', async () => {
            versions.set(variablePath, '4.5.3');
            versions.set(pathExecutable, null);
            const message = await findBlender('4.5.0', { env: environment, platform: 'win32', readVersion }).catch((error: Error) => error.message);
            expect(message).toContain('Blender 4.5.0 is needed');
            expect(message).toContain('version 4.5.3');
            expect(message).toContain('would not run');
            expect(message).toContain('BLENDER');
            expect(message).toContain('recorded with exactly this version');
            expect(message).toContain('re-record the output hashes');
        });

        it('says so when there is no Blender at all', async () => {
            const message = await findBlender('4.5.0', {
                env: { PATH: '', ProgramFiles: join(directory, 'nothing-here') },
                platform: 'win32',
                readVersion,
            }).catch((error: Error) => error.message);
            expect(message).toContain('No Blender was found at all');
        });
    });

    it('needs the full blenderVersion, like "4.5.0", in sources.json', () => {
        expect(validateSources({ assets: [asset({ convert: convertWith('4.5.0') })] })).toEqual([]);
        for (const version of ['4.5', '4', '4.5.0 LTS']) {
            expect(validateSources({ assets: [asset({ convert: convertWith(version) })] }).join('\n'), version).toContain('blenderVersion');
        }
        const { blenderVersion: _removed, ...noVersion } = convertWith();
        expect(validateSources({ assets: [rawAsset({ convert: noVersion })] }).join('\n')).toContain('blenderVersion: required');
    });

    it('takes a node conversion (a .ts script, no Blender version) and refuses other tools', () => {
        const node = { tool: 'node' as const, script: 'tools/assets/convert/x.ts', args: ['{dir}'] };
        expect(validateSources({ assets: [asset({ convert: node })] })).toEqual([]);
        expect(validateSources({ assets: [rawAsset({ convert: { ...node, script: 'tools/assets/convert/x.py' } })] }).join('\n')).toContain('a node conversion is a .ts script');
        expect(validateSources({ assets: [rawAsset({ convert: { ...node, tool: 'python' } })] }).join('\n')).toContain('tool: "node"');
    });

    it('blames the conversion when a converted output has the wrong hash, and says how to re-record', async () => {
        const directory = await mkdtemp(join(tmpdir(), 'ddgi-convhash-'));
        await writeFile(join(directory, 'a.glb'), 'bytes');
        const output = { path: 'a.glb', sha256: sha256('other'), bytes: 5 };
        const converted = await checkOutputs(directory, [output], () => undefined, convertWith());
        expect(converted[0]).toMatch(/different Blender version \(this one needs 4\.5\.0\) or a changed conversion script is the likely cause/);
        expect(converted[0]).toContain('re-record the hashes');
        const plain = await checkOutputs(directory, [output], () => undefined);
        expect(plain[0]).not.toContain('Blender');
        await rm(directory, { recursive: true, force: true });
    });
});

describe('download timeouts', () => {
    let server: Server;
    let baseUrl: string;
    let workDirectory: string;
    const openSockets = new Set<Socket>();

    beforeAll(async () => {
        workDirectory = await mkdtemp(join(tmpdir(), 'ddgi-timeout-'));
        server = createServer((request, response) => {
            if (request.url === '/never-answers') return; // the connection stays open and silent
            if (request.url === '/stalls') {
                response.writeHead(200, { 'content-length': 1000 });
                response.write('first bytes arrive, then nothing');
                return; // never ends
            }
            response.writeHead(200, { 'content-length': 2 }).end('ok');
        });
        server.on('connection', (socket) => {
            openSockets.add(socket);
            socket.on('close', () => openSockets.delete(socket));
        });
        await new Promise<void>((resolvePromise) => server.listen(0, '127.0.0.1', resolvePromise));
        baseUrl = `http://127.0.0.1:${(server.address() as AddressInfo).port}`;
    });

    afterAll(async () => {
        for (const socket of openSockets) socket.destroy();
        server.close();
        await rm(workDirectory, { recursive: true, force: true });
    });

    it('has sensible defaults: a minute to answer and a minute of silence', () => {
        expect(DEFAULT_DOWNLOAD_TIMEOUTS).toEqual({ connectMs: 60_000, stallMs: 60_000 });
    });

    it('gives up when the server never answers', async () => {
        const download = { url: `${baseUrl}/never-answers`, file: 'x.bin', bytes: 2, sha256: sha256('ok') };
        await expect(downloadFile(download, join(workDirectory, 'x.bin'), { connectMs: 150, stallMs: 1000 })).rejects.toThrow(/no answer from .* within 0\.15 s/);
    });

    it('gives up when the data stops mid-download, and leaves no partial file', async () => {
        const download = { url: `${baseUrl}/stalls`, file: 'y.bin', bytes: 1000, sha256: sha256('ok') };
        const destination = join(workDirectory, 'y.bin');
        await expect(downloadFile(download, destination, { connectMs: 1000, stallMs: 200 })).rejects.toThrow(/y\.bin stalled: no data for 0\.2 s/);
        await expect(readFile(`${destination}.part`)).rejects.toThrow();
    });

    it('still completes a healthy download well inside the limits', async () => {
        const download = { url: `${baseUrl}/fine`, file: 'z.bin', bytes: 2, sha256: sha256('ok') };
        const destination = join(workDirectory, 'z.bin');
        expect(await downloadFile(download, destination, { connectMs: 2000, stallMs: 2000 })).toBe(sha256('ok'));
        expect(await readFile(destination, 'utf8')).toBe('ok');
    });
});

describe('choosing the downloader', () => {
    it('uses curl for Intel, whose server refuses Node fetch', () => {
        expect(CURL_HOSTS).toEqual(['cdrdv2.intel.com']);
        expect(chooseDownloader('https://cdrdv2.intel.com/v1/dl/getContent/726650')).toBe('curl');
        expect(chooseDownloader('https://CDRDV2.Intel.com/v1/dl/getContent/830833')).toBe('curl');
    });

    it('uses Node fetch for every other host, including look-alikes of Intel', () => {
        expect(chooseDownloader('https://developer.nvidia.com/bistro.zip')).toBe('node-fetch');
        expect(chooseDownloader('https://raw.githubusercontent.com/x/y/main/a.glb')).toBe('node-fetch');
        expect(chooseDownloader('https://www.intel.com/content/samples.html')).toBe('node-fetch');
        expect(chooseDownloader('https://cdrdv2.intel.com.example.org/file.zip')).toBe('node-fetch');
        expect(chooseDownloader('https://example.org/cdrdv2.intel.com/file.zip')).toBe('node-fetch');
        expect(chooseDownloader('http://127.0.0.1:8080/file.zip')).toBe('node-fetch');
    });

    it('refuses a URL it cannot read', () => {
        expect(() => chooseDownloader('not a url')).toThrow(/not a valid download URL/);
    });

    it("finds Windows' own curl.exe, never one on PATH", () => {
        expect(systemCurlPath('win32', { SystemRoot: 'D:\\Win', PATH: 'C:\\Program Files\\Git\\mingw64\\bin' })).toBe(join('D:\\Win', 'System32', 'curl.exe'));
        expect(systemCurlPath('linux', {})).toBe('/usr/bin/curl');
    });

    it('runs curl plainly: follows redirects, fails on HTTP errors, no disguise', () => {
        const args = curlArguments('https://cdrdv2.intel.com/v1/dl/getContent/726650', 'out.zip.part', { connectMs: 60_000, stallMs: 60_000 });
        expect(args).toEqual([
            // Skips the user's .curlrc, which could otherwise add headers.
            '--disable',
            '--location',
            '--fail',
            '--silent',
            '--show-error',
            '--connect-timeout',
            '60',
            '--speed-limit',
            '1',
            '--speed-time',
            '60',
            '--output',
            'out.zip.part',
            'https://cdrdv2.intel.com/v1/dl/getContent/726650',
        ]);
        for (const disguise of ['-A', '--user-agent', '-H', '--header', '-e', '--referer', '-b', '--cookie']) expect(args).not.toContain(disguise);
        // curl's stall limit is whole seconds, at least 1.
        const short = curlArguments('https://x.org/a', 'a', { connectMs: 150, stallMs: 200 });
        expect(short[short.indexOf('--speed-time') + 1]).toBe('1');
        expect(short[short.indexOf('--connect-timeout') + 1]).toBe('0.15');
    });

    it('says what to do when curl is missing', async () => {
        const download = { url: 'https://cdrdv2.intel.com/v1/dl/getContent/726650', file: 'pkg_a_curtains.zip', bytes: 1, sha256: 'AB'.repeat(32) };
        const missing = join(tmpdir(), 'ddgi-no-such-folder', 'curl.exe');
        const failure = findSystemCurl(download, missing);
        await expect(failure).rejects.toThrow(/refuses Node's fetch/);
        await expect(findSystemCurl(download, missing)).rejects.toThrow(/does not exist/);
        await expect(findSystemCurl(download, missing)).rejects.toThrow(`assets/downloads/.cache/${'ab'.repeat(32)}`);
    });
});

// The real curl, against a local server. Skipped where the system has no curl.
describe.skipIf(!existsSync(systemCurlPath()))('downloading with the system curl', () => {
    const rejection = '<html><head><title>Request Rejected</title></head><body>The requested URL was rejected.</body></html>';
    let server: Server;
    let baseUrl: string;
    let workDirectory: string;

    beforeAll(async () => {
        workDirectory = await mkdtemp(join(tmpdir(), 'ddgi-curl-'));
        server = createServer((request, response) => {
            if (request.url === '/redirect') return response.writeHead(302, { location: '/file' }).end();
            if (request.url === '/file') return response.writeHead(200, { 'content-length': 7 }).end('payload');
            if (request.url === '/rejected') return response.writeHead(200, { 'content-type': 'text/html' }).end(rejection);
            response.writeHead(404).end('not here');
        });
        await new Promise<void>((resolvePromise) => server.listen(0, '127.0.0.1', resolvePromise));
        baseUrl = `http://127.0.0.1:${(server.address() as AddressInfo).port}`;
    });

    afterAll(async () => {
        server.close();
        await rm(workDirectory, { recursive: true, force: true });
    });

    it('follows a redirect, checks the hash and leaves the file in place', async () => {
        const download = { url: `${baseUrl}/redirect`, file: 'a.bin', bytes: 7, sha256: sha256('payload') };
        const destination = join(workDirectory, 'a.bin');
        expect(await downloadWithCurl(download, destination)).toBe(sha256('payload'));
        expect(await readFile(destination, 'utf8')).toBe('payload');
        await expect(readFile(`${destination}.part`)).rejects.toThrow();
    });

    it('refuses a small refusal page by its size, and leaves no partial file', async () => {
        const download = { url: `${baseUrl}/rejected`, file: 'b.zip', bytes: 786_898_766, sha256: 'c'.repeat(64) };
        const destination = join(workDirectory, 'b.zip');
        await expect(downloadWithCurl(download, destination)).rejects.toThrow(/bytes but sources.json says 786898766 \(a file this small is usually an error or refusal page/);
        await expect(readFile(destination)).rejects.toThrow();
        await expect(readFile(`${destination}.part`)).rejects.toThrow();
    });

    it('reports an HTTP error instead of saving the error page', async () => {
        const download = { url: `${baseUrl}/missing`, file: 'c.bin', bytes: 7, sha256: sha256('payload') };
        await expect(downloadWithCurl(download, join(workDirectory, 'c.bin'))).rejects.toThrow(/answered with an HTTP error \(curl: .*404/);
    });

    it('reports a wrong hash', async () => {
        const download = { url: `${baseUrl}/file`, file: 'd.bin', bytes: 7, sha256: sha256('other!!') };
        await expect(downloadWithCurl(download, join(workDirectory, 'd.bin'))).rejects.toThrow(/download sha256 is .* but sources.json says/);
    });
});

describe('unpacking a shared zip once', () => {
    const quiet = () => undefined;
    const noKeep = new Set<string>();
    const zip = (name: string) => ({ url: 'https://example.com/pack.zip', file: 'pack.zip', bytes: 1, sha256: sha256(name) });

    /** A stand-in for unzip that writes two files and counts its runs. */
    function fakeUnzip() {
        const state = { unpacks: 0 };
        const extract = async (_archive: string, destination: string) => {
            state.unpacks++;
            await mkdir(join(destination, 'models'), { recursive: true });
            await writeFile(join(destination, 'models', 'a.fbx'), 'model data');
            await writeFile(join(destination, 'readme.txt'), 'hello');
        };
        return { state, extract };
    }

    it('unpacks into a cache keyed by the zip hash, links it into each asset folder, and returns the unpacked paths', async () => {
        const directory = await mkdtemp(join(tmpdir(), 'ddgi-unpack-'));
        const cache = join(directory, '.cache');
        const { state, extract } = fakeUnzip();

        const first = await extractWithCache(zip('the zip'), 'ignored.zip', join(directory, 'first'), cache, noKeep, quiet, extract);
        await extractWithCache(zip('the zip'), 'ignored.zip', join(directory, 'second'), cache, noKeep, quiet, extract);

        expect(state.unpacks).toBe(1);
        expect(first).toEqual(['models/a.fbx', 'readme.txt']);
        for (const assetFolder of ['first', 'second']) {
            expect(await readFile(join(directory, assetFolder, 'models', 'a.fbx'), 'utf8')).toBe('model data');
            expect(await readFile(join(directory, assetFolder, 'readme.txt'), 'utf8')).toBe('hello');
        }
        await rm(directory, { recursive: true, force: true });
    });

    it('never reuses a half-finished unpack: a failed unpack leaves nothing in the cache', async () => {
        const directory = await mkdtemp(join(tmpdir(), 'ddgi-unpack-fail-'));
        const cache = join(directory, '.cache');
        const failing = async (_archive: string, destination: string) => {
            await writeFile(join(destination, 'half.bin'), 'partial');
            throw new Error('disk full');
        };
        await expect(extractWithCache(zip('another zip'), 'ignored.zip', join(directory, 'asset'), cache, noKeep, quiet, failing)).rejects.toThrow('disk full');

        let unpacks = 0;
        const working = async (_archive: string, destination: string) => {
            unpacks++;
            await writeFile(join(destination, 'whole.bin'), 'complete');
        };
        await extractWithCache(zip('another zip'), 'ignored.zip', join(directory, 'asset'), cache, noKeep, quiet, working);
        expect(unpacks).toBe(1);
        expect(await readFile(join(directory, 'asset', 'whole.bin'), 'utf8')).toBe('complete');
        await expect(readFile(join(directory, 'asset', 'half.bin'))).rejects.toThrow();
        await rm(directory, { recursive: true, force: true });
    });

    it('unpacks again when the cached copy no longer matches the file list a complete unpack wrote', async () => {
        const directory = await mkdtemp(join(tmpdir(), 'ddgi-unpack-check-'));
        const cache = join(directory, '.cache');
        const cached = join(cache, 'extracted', sha256('checked zip'));
        const { state, extract } = fakeUnzip();
        const unpack = () => extractWithCache(zip('checked zip'), 'ignored.zip', join(directory, 'asset'), cache, noKeep, quiet, extract);

        await unpack();
        await unpack();
        expect(state.unpacks).toBe(1);

        // A file went missing from the cache.
        await rm(join(cached, 'readme.txt'));
        await unpack();
        expect(state.unpacks).toBe(2);
        expect(await readFile(join(directory, 'asset', 'readme.txt'), 'utf8')).toBe('hello');

        // A file was written in place (through a hard link from some asset folder).
        await appendFile(join(cached, 'models', 'a.fbx'), ' edited');
        await unpack();
        expect(state.unpacks).toBe(3);
        expect(await readFile(join(directory, 'asset', 'models', 'a.fbx'), 'utf8')).toBe('model data');

        // An extra file appeared.
        await writeFile(join(cached, 'stray.txt'), 'x');
        await unpack();
        expect(state.unpacks).toBe(4);

        // The folder is there but the list is not: the run stopped between the rename and writing the list.
        await rm(`${cached}.files.json`);
        await unpack();
        expect(state.unpacks).toBe(5);
        await rm(directory, { recursive: true, force: true });
    });

    it('leaves the asset folder holding exactly the archive, the downloads and the outputs', async () => {
        const directory = await mkdtemp(join(tmpdir(), 'ddgi-unpack-mirror-'));
        const cache = join(directory, '.cache');
        const assetFolder = join(directory, 'asset');
        const { extract } = fakeUnzip();
        // Left over from an older version of the archive, and a stray log.
        await mkdir(join(assetFolder, 'old', 'nested'), { recursive: true });
        await writeFile(join(assetFolder, 'old', 'nested', 'gone.fbx'), 'old');
        await writeFile(join(assetFolder, 'models-old.txt'), 'old');
        await writeFile(join(assetFolder, 'pack.zip'), 'the download');
        await mkdir(join(assetFolder, 'out'), { recursive: true });
        await writeFile(join(assetFolder, 'out', 'model.glb'), 'an output');

        const messages: string[] = [];
        const keep = new Set(['pack.zip', 'out/model.glb']);
        await extractWithCache(zip('new version'), 'ignored.zip', assetFolder, cache, keep, (message) => messages.push(message), extract);

        const listing = async (folder: string, prefix = ''): Promise<string[]> => {
            const names: string[] = [];
            for (const entry of await readdir(join(folder, prefix), { withFileTypes: true })) {
                const path = prefix ? `${prefix}/${entry.name}` : entry.name;
                names.push(...(entry.isDirectory() ? await listing(folder, path) : [path]));
            }
            return names.sort();
        };
        expect(await listing(assetFolder)).toEqual(['models/a.fbx', 'out/model.glb', 'pack.zip', 'readme.txt']);
        await expect(readdir(join(assetFolder, 'old'))).rejects.toThrow(); // the emptied folder is gone too
        expect(messages.join('\n')).toContain('removing old/nested/gone.fbx');
        await rm(directory, { recursive: true, force: true });
    });

    it('notices an input written in place through a hard link, which is how a convert script would damage the cache', async () => {
        const directory = await mkdtemp(join(tmpdir(), 'ddgi-fingerprint-'));
        await writeFile(join(directory, 'cached.bin'), 'original');
        await link(join(directory, 'cached.bin'), join(directory, 'linked.bin'));
        const longAgo = new Date(Date.now() - 3_600_000);
        await utimes(join(directory, 'linked.bin'), longAgo, longAgo);
        const before = await fileFingerprints(directory, ['linked.bin', 'cached.bin']);
        await appendFile(join(directory, 'linked.bin'), '!');
        const after = await fileFingerprints(directory, ['linked.bin', 'cached.bin']);
        expect(after.get('linked.bin')).not.toBe(before.get('linked.bin'));
        expect(after.get('cached.bin')).not.toBe(before.get('cached.bin')); // the same file, seen from the cache
        await rm(directory, { recursive: true, force: true });
    });
});

describe('output size checks', () => {
    it('reports a wrong size before hashing, and blames the conversion for converted files', async () => {
        const directory = await mkdtemp(join(tmpdir(), 'ddgi-size-'));
        await writeFile(join(directory, 'a.glb'), 'twelve bytes');
        const problems = await checkOutputs(directory, [{ path: 'a.glb', bytes: 99, sha256: sha256('twelve bytes') }], () => undefined, convertWith());
        expect(problems[0]).toMatch(/size differs: a\.glb \(expected 99 bytes, got 12\).*Blender version.*likely cause/);
        expect(await checkOutputs(directory, [{ path: 'a.glb', bytes: 12, sha256: sha256('twelve bytes') }], () => undefined)).toEqual([]);
        await rm(directory, { recursive: true, force: true });
    });
});

// Real processes start slowly on a busy machine, so these get a generous time budget; nothing here asserts how long a step takes.
describe('running a conversion that may hang', { timeout: 60_000 }, () => {
    const quickLimits = { timeoutMs: 20_000, graceMs: 300, pollMs: 50 };
    let directory: string;

    beforeAll(async () => {
        directory = await mkdtemp(join(tmpdir(), 'ddgi-convert-'));
    });
    afterAll(() => rm(directory, { recursive: true, force: true }));

    /** A stand-in for Blender: node running a one-line script. */
    const node = (script: string) => [process.execPath, ['-e', script]] as const;
    const hangForever = 'setInterval(() => {}, 1000)';

    it('reports a normal exit with its exit code', async () => {
        const [command, args] = node('process.exit(3)');
        const result = await runConversionProcess(command, [...args], { cwd: directory, assetDirectory: directory, outputs: [], limits: quickLimits });
        expect(result).toEqual({ end: 'exited', code: 3, signal: null });
        expect(conversionFailure(result, 'x.py')).toBe('Blender exited with code 3 while running x.py');
    });

    it('stops a process that wrote every output and then hangs, like Blender does at shutdown, and the kill carries the guard mark', async () => {
        const output = join(directory, 'hang-after.bin').replaceAll('\\', '/');
        const [command, args] = node(`require('fs').writeFileSync('${output}', 'done'); ${hangForever}`);
        const result = await runConversionProcess(command, [...args], {
            cwd: directory,
            assetDirectory: directory,
            outputs: [{ path: 'hang-after.bin', bytes: 4, sha256: sha256('done') }],
            limits: quickLimits,
        });
        expect(result.end).toBe('stopped-after-outputs');
        expect(endedByGuardKill(result.code, result.signal)).toBe(true);
        expect(conversionFailure(result, 'x.py')).toBeNull();
        expect(await readFile(join(directory, 'hang-after.bin'), 'utf8')).toBe('done');
    });

    it('kills a process that never finishes its outputs once the time limit passes', async () => {
        const [command, args] = node(hangForever);
        const result = await runConversionProcess(command, [...args], {
            cwd: directory,
            assetDirectory: directory,
            outputs: [{ path: 'never-written.bin', bytes: 4, sha256: sha256('done') }],
            limits: { ...quickLimits, timeoutMs: 600 },
        });
        expect(result.end).toBe('timed-out');
        expect(conversionFailure(result, 'x.py', { ...quickLimits, timeoutMs: 600 })).toContain('was still running');
    });

    it('does not mistake an output left over from an earlier run for a finished one', async () => {
        const stale = join(directory, 'stale.bin');
        await writeFile(stale, 'done');
        const longAgo = new Date(Date.now() - 3_600_000);
        await utimes(stale, longAgo, longAgo);
        const [command, args] = node('setTimeout(() => process.exit(0), 1500)');
        const result = await runConversionProcess(command, [...args], {
            cwd: directory,
            assetDirectory: directory,
            outputs: [{ path: 'stale.bin', bytes: 4, sha256: sha256('done') }],
            limits: { ...quickLimits, graceMs: 100 },
        });
        expect(result).toEqual({ end: 'exited', code: 0, signal: null });
    });

    it('treats an output of the right size but wrong content as finished, so the final check can report it', async () => {
        const output = join(directory, 'wrong.bin').replaceAll('\\', '/');
        const [command, args] = node(`require('fs').writeFileSync('${output}', 'oops'); ${hangForever}`);
        const result = await runConversionProcess(command, [...args], {
            cwd: directory,
            assetDirectory: directory,
            outputs: [{ path: 'wrong.bin', bytes: 4, sha256: sha256('done') }],
            limits: quickLimits,
        });
        expect(result.end).toBe('stopped-after-outputs');
        const problems = await checkOutputs(directory, [{ path: 'wrong.bin', bytes: 4, sha256: sha256('done') }], () => undefined);
        expect(problems[0]).toContain('sha256 differs: wrong.bin');
    });
});

/** A stand-in for a child process whose life the test controls event by event. */
class FakeChild extends EventEmitter {
    pid = 4242;

    /** The process ends: 'exit' fires, then 'close' once its output is flushed. */
    finish(code: number | null, signal: string | null = null): void {
        this.emit('exit', code, signal);
        this.emit('close', code, signal);
    }
}

describe('who gets to end a conversion: the process or the guard', () => {
    const fastLimits = { timeoutMs: 400, graceMs: 0, pollMs: 5 };
    let directory: string;
    const expectedOutput = { path: 'made.bin', bytes: 4, sha256: sha256('done') };

    beforeAll(async () => {
        directory = await mkdtemp(join(tmpdir(), 'ddgi-race-'));
        await writeFile(join(directory, 'made.bin'), 'done');
    });
    afterAll(() => rm(directory, { recursive: true, force: true }));

    /** `onKill` says how the process ends once the guard kills it; by default the kill lands and leaves its mark. */
    const run = (child: FakeChild, outputs: AssetOutput[], limits = fastLimits, onKill = () => child.finish(GUARD_KILL_EXIT_CODE)) => {
        let killed = 0;
        const result = runConversionProcess('fake', [], {
            cwd: directory,
            assetDirectory: directory,
            outputs,
            limits,
            spawnProcess: () => child,
            killTree: async () => {
                killed++;
                onKill();
            },
        });
        return { result, killed: () => killed };
    };

    it('recognises the guard kill by its mark: the special exit code on Windows, SIGKILL elsewhere', () => {
        expect(endedByGuardKill(GUARD_KILL_EXIT_CODE, null)).toBe(true);
        expect(endedByGuardKill(null, 'SIGKILL')).toBe(true);
        for (const [code, signal] of [[0, null], [1, null], [3221225477, null], [null, 'SIGSEGV']] as const) {
            expect(endedByGuardKill(code, signal), `${code} ${signal}`).toBe(false);
        }
    });

    it('reports the real exit code of a process that ends by itself after writing its output, even with no grace period', async () => {
        const child = new FakeChild();
        const { result, killed } = run(child, [expectedOutput]);
        child.finish(7); // exits before the guard has looked at anything
        // Give the guard every chance to misfire (any wait only makes a wrong implementation more likely to fail, never a right one).
        await new Promise((resolvePromise) => setTimeout(resolvePromise, 100));
        expect(await result).toEqual({ end: 'exited', code: 7, signal: null });
        expect(killed()).toBe(0);
    });

    it('does not mistake a process that has exited but not yet closed for one to stop', async () => {
        const child = new FakeChild();
        const { result, killed } = run(child, [expectedOutput]);
        child.emit('exit', 9, null); // exited; the 'close' event has not arrived yet
        await new Promise((resolvePromise) => setTimeout(resolvePromise, 100)); // the output is found and the zero-length grace period passes
        child.emit('close', 9, null);
        expect(await result).toEqual({ end: 'exited', code: 9, signal: null });
        expect(killed()).toBe(0);
    });

    it('fails the conversion when the process exits with an error code of its own while the guard is killing it', async () => {
        const child = new FakeChild();
        // The guard has decided to stop the process, but before its kill lands the process exits by itself with
        // code 1 (a Python error in Blender). An exit status is set once, so the kill cannot overwrite it.
        const { result, killed } = run(child, [expectedOutput], fastLimits, () => child.finish(1));
        const outcome = await result;
        expect(killed()).toBe(1);
        expect(outcome).toEqual({ end: 'exited', code: 1, signal: null });
        expect(conversionFailure(outcome, 'convert.py')).toBe('Blender exited with code 1 while running convert.py');
    });

    it('accepts a clean exit of its own in that same moment', async () => {
        const child = new FakeChild();
        const { result } = run(child, [expectedOutput], fastLimits, () => child.finish(0));
        const outcome = await result;
        expect(outcome).toEqual({ end: 'exited', code: 0, signal: null });
        expect(conversionFailure(outcome, 'convert.py')).toBeNull();
    });

    it('kills a process that is still running once its outputs are written, and says the guard did it', async () => {
        const child = new FakeChild();
        const { result, killed } = run(child, [expectedOutput]);
        expect(await result).toEqual({ end: 'stopped-after-outputs', code: GUARD_KILL_EXIT_CODE, signal: null });
        expect(killed()).toBe(1);
    });

    it('rejects when the killed process never reports its end, rather than guessing', { timeout: 10_000 }, async () => {
        const child = new FakeChild();
        const { result } = run(child, [expectedOutput], fastLimits, () => undefined);
        await expect(result).rejects.toThrow('did not end within 5 s');
    });

    it('never stops early a conversion that lists no outputs; only the time limit ends it', async () => {
        const child = new FakeChild();
        const { result, killed } = run(child, [], { ...fastLimits, timeoutMs: 150 }, () => child.finish(null, 'SIGKILL'));
        expect(await result).toEqual({ end: 'timed-out', code: null, signal: 'SIGKILL' });
        expect(killed()).toBe(1);
    });

    it('does not treat an output that never reaches its size as finished', async () => {
        const child = new FakeChild();
        const { result } = run(child, [{ ...expectedOutput, bytes: 5 }], { ...fastLimits, timeoutMs: 150 });
        expect((await result).end).toBe('timed-out');
    });
});

// ---------------------------------------------------------------------------
// Non-reproducible .glb outputs
// ---------------------------------------------------------------------------

/** A .glb from a JSON value and BIN bytes: chunks padded to 4 bytes (JSON with spaces, BIN with zeros). */
function makeGlb(json: unknown, bin: Buffer, trailing: Buffer = Buffer.alloc(0)): Buffer {
    const text = JSON.stringify(json);
    const jsonBytes = Buffer.from(text + ' '.repeat((4 - (text.length % 4)) % 4));
    const binBytes = Buffer.concat([bin, Buffer.alloc((4 - (bin.length % 4)) % 4)]);
    const header = Buffer.alloc(12);
    header.writeUInt32LE(0x46546c67, 0);
    header.writeUInt32LE(2, 4);
    header.writeUInt32LE(12 + 8 + jsonBytes.length + 8 + binBytes.length + trailing.length, 8);
    const jsonHeader = Buffer.alloc(8);
    jsonHeader.writeUInt32LE(jsonBytes.length, 0);
    jsonHeader.writeUInt32LE(0x4e4f534a, 4);
    const binHeader = Buffer.alloc(8);
    binHeader.writeUInt32LE(binBytes.length, 0);
    binHeader.writeUInt32LE(0x004e4942, 4);
    return Buffer.concat([header, jsonHeader, jsonBytes, binHeader, binBytes, trailing]);
}

interface SceneData {
    position: Buffer;
    normal: Buffer;
    indices: Buffer;
    image: Buffer;
}

/** The data of a tiny one-mesh scene with an embedded image. Each piece has a length that needs alignment padding after it. */
function sceneData(): SceneData {
    return {
        position: Buffer.from('POSITIONS-12'),
        normal: Buffer.from('NORMALS--12!'),
        indices: Buffer.from('IDX-6!'),
        image: Buffer.from('PNG!!'),
    };
}

/** Lays the scene out as Blender does: one bufferView per piece, each starting on a 4-byte boundary, zero padding between. */
function sceneGlb(data: SceneData = sceneData(), padding = 0): Buffer {
    const pieces = [data.position, data.normal, data.indices, data.image];
    const parts: Buffer[] = [];
    const bufferViews: Array<{ buffer: number; byteOffset: number; byteLength: number }> = [];
    let offset = 0;
    for (const piece of pieces) {
        bufferViews.push({ buffer: 0, byteOffset: offset, byteLength: piece.length });
        const gap = (4 - (piece.length % 4)) % 4;
        parts.push(piece, Buffer.alloc(gap, padding));
        offset += piece.length + gap;
    }
    const json = {
        asset: { version: '2.0' },
        buffers: [{ byteLength: offset }],
        bufferViews,
        accessors: [{ bufferView: 0 }, { bufferView: 1 }, { bufferView: 2 }],
        meshes: [{ primitives: [{ attributes: { POSITION: 0, NORMAL: 1 }, indices: 2 }] }],
        images: [{ bufferView: 3, mimeType: 'image/png' }],
    };
    return makeGlb(json, Buffer.concat(parts));
}

describe('outputs of a converter that is not bit-reproducible', () => {
    let directory: string;
    const recordFile = 'glb-views/test-asset.json';

    beforeAll(async () => {
        directory = await mkdtemp(join(tmpdir(), 'ddgi-glb-'));
        await mkdir(join(directory, 'glb-views'));
    });
    afterAll(() => rm(directory, { recursive: true, force: true }));

    async function write(name: string, glb: Buffer): Promise<string> {
        const path = join(directory, name);
        await writeFile(path, glb);
        return path;
    }

    /** The sources.json entry and bufferView record for a file as first converted, with NORMAL known to vary. */
    async function recordedFor(glb: Buffer, varyingAttributes = ['NORMAL']) {
        const structure = await readGlbStructure(await write('recorded.glb', glb));
        const output: AssetOutput = {
            path: 'model.glb',
            bytes: glb.length,
            sha256: sha256(glb),
            reproducible: false,
            jsonSha256: structure.jsonSha256,
            binBytes: structure.binBytes,
            bufferViewHashes: recordFile,
            varyingAttributes,
        };
        const record = makeGlbViewRecord('test-asset', output, structure);
        await writeFile(join(directory, recordFile), formatGlbViewRecord(record));
        return { output, record };
    }

    it('hashes every bufferView, says what each holds, and finds no stray bytes in zero padding', async () => {
        const data = sceneData();
        const structure = await readGlbStructure(await write('read.glb', sceneGlb(data)));
        expect(structure.bufferViewSha256).toEqual([sha256(data.position), sha256(data.normal), sha256(data.indices), sha256(data.image)]);
        expect(structure.bufferViewUses).toEqual([['POSITION'], ['NORMAL'], ['indices'], ['image']]);
        expect(structure.nonZeroPaddingBytes).toBe(0);
        expect(structure.binBytes).toBe(12 + 12 + 8 + 8);
    });

    it('counts padding bytes that are not zero', async () => {
        const structure = await readGlbStructure(await write('padded.glb', sceneGlb(sceneData(), 0x7f)));
        expect(structure.nonZeroPaddingBytes).toBe(2 + 3); // after the 6 index bytes and the 5 image bytes
    });

    it('rejects files that are not well-formed single-buffer .glb', async () => {
        const good = sceneGlb();
        const badMagic = Buffer.from(good);
        badMagic.writeUInt32LE(0x12345678, 0);
        await expect(readGlbStructure(await write('magic.glb', badMagic))).rejects.toThrow('bad magic');
        await expect(readGlbStructure(await write('short.glb', good.subarray(0, 10)))).rejects.toThrow('too short');
        await expect(readGlbStructure(await write('cut.glb', good.subarray(0, good.length - 4)))).rejects.toThrow('header says');

        const jsonOnly = makeGlb({ asset: { version: '2.0' } }, Buffer.alloc(0)).subarray(0, 12 + 8 + 28);
        const noBin = Buffer.from(jsonOnly);
        noBin.writeUInt32LE(noBin.length, 8);
        await expect(readGlbStructure(await write('nobin.glb', noBin))).rejects.toThrow('no BIN chunk');

        const trailing = makeGlb({ asset: { version: '2.0' } }, Buffer.from('data'), Buffer.from('MORE'));
        await expect(readGlbStructure(await write('trailing.glb', trailing))).rejects.toThrow('the BIN chunk ends at byte');

        const external = makeGlb({ buffers: [{ byteLength: 4 }, { uri: 'more.bin', byteLength: 4 }] }, Buffer.from('data'));
        await expect(readGlbStructure(await write('external.glb', external))).rejects.toThrow('external buffer');

        const outside = makeGlb({ buffers: [{ byteLength: 4 }], bufferViews: [{ buffer: 0, byteOffset: 2, byteLength: 4 }] }, Buffer.from('data'));
        await expect(readGlbStructure(await write('outside.glb', outside))).rejects.toThrow('bufferView 0 is not inside the BIN chunk');
    });

    it('records a hash for every bufferView except those used only by varying attributes', async () => {
        const { record } = await recordedFor(sceneGlb());
        expect(record.bufferViews.map((hash) => (hash === null ? null : 'pinned'))).toEqual(['pinned', null, 'pinned', 'pinned']);
        const reread = await readGlbViewRecord(join(directory, recordFile));
        expect(reread).toEqual(record);
    });

    it('passes when only a varying attribute changed, as when the exporter wobbles a normal', async () => {
        const { output, record } = await recordedFor(sceneGlb());
        const wobbled = sceneGlb({ ...sceneData(), normal: Buffer.from('NORMALS--12?') });
        const structure = await readGlbStructure(await write('wobbled.glb', wobbled));
        expect(structure.fileSha256).not.toBe(output.sha256);
        expect(checkGlbStructure(structure, output, record)).toEqual([]);
    });

    it('fails when a pinned bufferView changed, and names it', async () => {
        const { output, record } = await recordedFor(sceneGlb());
        for (const [piece, use] of [['position', 'POSITION'], ['indices', 'indices'], ['image', 'image']] as const) {
            const data = sceneData();
            const changed = Buffer.from(data[piece]);
            changed[0] ^= 0x01;
            const structure = await readGlbStructure(await write(`changed-${piece}.glb`, sceneGlb({ ...data, [piece]: changed })));
            const problems = checkGlbStructure(structure, output, record);
            expect(problems, piece).toHaveLength(1);
            expect(problems[0]).toMatch(new RegExp(`1 pinned bufferView\\(s\\) differ: bufferView \\d \\(${use}\\)`));
        }
    });

    it('never leaves an index buffer or an image unpinned, even in a hand-edited record', async () => {
        const { output, record } = await recordedFor(sceneGlb());
        const edited = { ...record, bufferViews: [record.bufferViews[0], null, null, null] };
        const structure = await readGlbStructure(await write('same.glb', sceneGlb()));
        const problems = checkGlbStructure(structure, output, edited);
        expect(problems).toHaveLength(2);
        expect(problems[0]).toContain('bufferView 2 (indices) has no recorded hash');
        expect(problems[1]).toContain('bufferView 3 (image) has no recorded hash');
    });

    it('a view shared by a varying attribute and anything else stays pinned', async () => {
        const glb = makeGlb(
            {
                buffers: [{ byteLength: 4 }],
                bufferViews: [{ buffer: 0, byteOffset: 0, byteLength: 4 }],
                accessors: [{ bufferView: 0 }, { bufferView: 0 }],
                meshes: [{ primitives: [{ attributes: { NORMAL: 0 }, indices: 1 }] }],
            },
            Buffer.from('data'),
        );
        const { record } = await recordedFor(glb);
        expect(record.bufferViews[0]).not.toBeNull();
    });

    it('fails when the JSON chunk changed, even with the same file size', async () => {
        const { output, record } = await recordedFor(sceneGlb());
        const data = sceneData();
        // Same lengths, so the same JSON size; only the mime type text differs.
        const json = JSON.parse(sceneGlb(data).subarray(20, 20 + sceneGlb(data).readUInt32LE(12)).toString()) as { images: Array<{ mimeType: string }> };
        json.images[0].mimeType = 'image/pn_';
        const changed = makeGlb(json, sceneGlb(data).subarray(20 + sceneGlb(data).readUInt32LE(12) + 8));
        expect(changed.length).toBe(output.bytes);
        const problems = checkGlbStructure(await readGlbStructure(await write('json.glb', changed)), output, record);
        expect(problems).toHaveLength(1);
        expect(problems[0]).toContain('the JSON chunk differs');
    });

    it('fails when the BIN chunk has a different length', async () => {
        const { output, record } = await recordedFor(sceneGlb());
        const structure = await readGlbStructure(await write('bin.glb', sceneGlb()));
        const problems = checkGlbStructure({ ...structure, binBytes: structure.binBytes + 4 }, output, record);
        expect(problems[0]).toContain(`the BIN chunk is ${structure.binBytes + 4} bytes but sources.json says ${structure.binBytes}`);
    });

    it('fails when the record was made for another JSON chunk', async () => {
        const { output, record } = await recordedFor(sceneGlb());
        const structure = await readGlbStructure(await write('other-record.glb', sceneGlb()));
        expect(checkGlbStructure(structure, output, { ...record, jsonSha256: 'f'.repeat(64) })[0]).toContain('re-record it with --record-glb-views');
    });

    it('checkOutputs pins the pieces instead of the whole-file hash, and logs the whole-file hash for information', async () => {
        const { output } = await recordedFor(sceneGlb());
        const wobbled = sceneGlb({ ...sceneData(), normal: Buffer.from('NORMALS--12?') });
        await write('model.glb', wobbled);
        const notes: string[] = [];
        expect(await checkOutputs(directory, [output], (message) => notes.push(message), convertWith(), { repoRoot: directory })).toEqual([]);
        expect(notes.join('\n')).toContain('not bit-reproducible');
        expect(notes.join('\n')).toContain('3 of 4 bufferViews');
        expect(notes.join('\n')).toContain(sha256(wobbled));

        // A different size is still caught before anything else.
        const resized = await checkOutputs(directory, [{ ...output, bytes: output.bytes + 4 }], () => undefined, convertWith(), { repoRoot: directory });
        expect(resized[0]).toContain('size differs');

        // A changed position is caught, and blamed on the conversion.
        const data = sceneData();
        await write('model.glb', sceneGlb({ ...data, position: Buffer.from('POSITIONS-13') }));
        const moved = await checkOutputs(directory, [output], () => undefined, convertWith(), { repoRoot: directory });
        expect(moved[0]).toMatch(/model\.glb: 1 pinned bufferView\(s\) differ: bufferView 0 \(POSITION\).*likely cause/);

        // A missing record file is a plain problem, not a crash.
        const noRecord = await checkOutputs(directory, [{ ...output, bufferViewHashes: 'glb-views/missing.json' }], () => undefined, null, { repoRoot: directory });
        expect(noRecord[0]).toContain('cannot read the bufferView record');

        // The same file without the flag fails on the whole-file hash.
        await write('model.glb', wobbled);
        const { reproducible: _flag, jsonSha256: _json, binBytes: _bin, bufferViewHashes: _views, varyingAttributes: _varying, ...strict } = output;
        expect((await checkOutputs(directory, [strict], () => undefined, convertWith()))[0]).toContain('sha256 differs');
    });

    it('sources.json must give jsonSha256, binBytes, bufferViewHashes and varyingAttributes with reproducible false, and only for .glb files', () => {
        const output = { path: 'a.glb', sha256: 'b'.repeat(64), bytes: 40, reproducible: false };
        const text = validateSources({ assets: [rawAsset({ outputs: [output] })] }).join('\n');
        expect(text).toContain('jsonSha256: required with reproducible false');
        expect(text).toContain('binBytes: required with reproducible false');
        expect(text).toContain('bufferViewHashes: required with reproducible false');
        expect(text).toContain('varyingAttributes: required with reproducible false');
        const complete = { ...output, jsonSha256: 'c'.repeat(64), binBytes: 16, bufferViewHashes: 'tools/assets/glb-views/a.json', varyingAttributes: ['NORMAL'] };
        expect(validateSources({ assets: [rawAsset({ outputs: [complete] })] })).toEqual([]);
        expect(validateSources({ assets: [rawAsset({ outputs: [{ ...complete, varyingAttributes: [] }] })] })).toEqual([]);
        // "indices" and "image" are not attribute names, so they can never be listed as varying.
        for (const name of ['indices', 'image', 'normal']) {
            const text = validateSources({ assets: [rawAsset({ outputs: [{ ...complete, varyingAttributes: [name] }] })] }).join('\n');
            expect(text, name).toContain('varyingAttributes');
        }
        expect(validateSources({ assets: [rawAsset({ outputs: [{ ...complete, bufferViewHashes: '../x.json' }] })] }).join('\n')).toContain('bufferViewHashes');
        expect(validateSources({ assets: [rawAsset({ outputs: [{ ...complete, path: 'a.png' }] })] }).join('\n')).toContain('only supported for .glb');
        expect(validateSources({ assets: [rawAsset({ outputs: [{ ...complete, reproducible: 'no' }] })] }).join('\n')).toContain('reproducible: must be true or false');
        const strayFields = { path: 'a.bin', sha256: 'b'.repeat(64), bytes: 4, varyingAttributes: ['NORMAL'] };
        expect(validateSources({ assets: [rawAsset({ outputs: [strayFields] })] }).join('\n')).toContain('only go with reproducible false');
    });
});

describe('outputs that must have no metal (dielectric)', () => {
    let directory: string;

    beforeAll(async () => {
        directory = await mkdtemp(join(tmpdir(), 'ddgi-dielectric-'));
    });
    afterAll(() => rm(directory, { recursive: true, force: true }));

    /** A one-triangle .glb with the given materials; the primitive uses `primitiveMaterial`. */
    function glbWithMaterials(materials: unknown[], primitiveMaterial: number | undefined = 0): Buffer {
        const json = {
            asset: { version: '2.0' },
            buffers: [{ byteLength: 4 }],
            bufferViews: [{ buffer: 0, byteOffset: 0, byteLength: 4 }],
            accessors: [{ bufferView: 0 }],
            materials,
            meshes: [{ primitives: [{ attributes: { POSITION: 0 }, ...(primitiveMaterial === undefined ? {} : { material: primitiveMaterial }) }] }],
        };
        return makeGlb(json, Buffer.from('DATA'));
    }

    const plaster = { name: 'Plaster', pbrMetallicRoughness: { baseColorFactor: [0.9, 0.9, 0.9, 1], metallicFactor: 0, roughnessFactor: 0.9 } };

    it('accepts materials with metallicFactor 0, even with a metallicRoughness texture, since glTF multiplies the two', () => {
        const textured = { name: 'Tiles', pbrMetallicRoughness: { metallicFactor: 0, metallicRoughnessTexture: { index: 0 } } };
        expect(metalProblems({ materials: [plaster, textured], meshes: [{ primitives: [{ material: 0 }, { material: 1 }] }] })).toEqual([]);
    });

    it('treats a missing metallicFactor as glTF does: 1, fully metal (what Blender wrote for the Breakfast Room)', () => {
        const problems = metalProblems({ materials: [plaster, { name: 'Paint', pbrMetallicRoughness: { roughnessFactor: 0.9 } }, { name: 'Bare' }] });
        expect(problems).toHaveLength(1);
        expect(problems[0]).toContain('2 material(s) can be metal');
        expect(problems[0]).toContain(`"Paint" (metallicFactor not written, so glTF's default 1)`);
        expect(problems[0]).toContain('"Bare"');
        expect(problems[0]).not.toContain('Plaster');
    });

    it('flags any factor above 0 and says when a texture comes with it', () => {
        const problems = metalProblems({ materials: [{ pbrMetallicRoughness: { metallicFactor: 0.25, metallicRoughnessTexture: { index: 0 } } }] });
        expect(problems[0]).toContain('#0 (metallicFactor 0.25, with a metallicRoughness texture)');
    });

    it('flags primitives without a material, because glTF default material is fully metal', () => {
        expect(metalProblems({ materials: [plaster], meshes: [{ primitives: [{ material: 0 }, {}] }] })).toEqual([
            "1 mesh primitive(s) have no material, so they get glTF's default material, which is fully metal",
        ]);
    });

    it('checkOutputs checks metal on marked outputs even when the hash matches, and only on those', async () => {
        const metal = glbWithMaterials([{ name: 'Paint', pbrMetallicRoughness: {} }]);
        await writeFile(join(directory, 'room.glb'), metal);
        const recorded: AssetOutput = { path: 'room.glb', bytes: metal.length, sha256: sha256(metal) };
        expect(await checkOutputs(directory, [recorded], () => undefined)).toEqual([]);
        const marked = await checkOutputs(directory, [{ ...recorded, dielectric: true }], () => undefined);
        expect(marked).toHaveLength(1);
        expect(marked[0]).toMatch(/^room\.glb: 1 material\(s\) can be metal.*"Paint"/);

        const dielectric = glbWithMaterials([plaster]);
        await writeFile(join(directory, 'room.glb'), dielectric);
        expect(await checkOutputs(directory, [{ path: 'room.glb', bytes: dielectric.length, sha256: sha256(dielectric), dielectric: true }], () => undefined)).toEqual([]);

        // A file that is not a .glb is a plain problem, not a crash.
        await writeFile(join(directory, 'room.glb'), 'not a glb at all, just text');
        const broken = await checkOutputs(directory, [{ ...recorded, dielectric: true }], () => undefined);
        expect(broken[0]).toContain('room.glb: not a .glb file (bad magic)');
    });

    it('readGlbJson reads the scene description without the BIN chunk', async () => {
        await writeFile(join(directory, 'json.glb'), glbWithMaterials([plaster]));
        const gltf = (await readGlbJson(join(directory, 'json.glb'))) as { materials: Array<{ name: string }> };
        expect(gltf.materials[0].name).toBe('Plaster');
        await writeFile(join(directory, 'short.glb'), Buffer.from('glTF'));
        await expect(readGlbJson(join(directory, 'short.glb'))).rejects.toThrow('too short');
    });

    it('sources.json takes dielectric only as true or false, and only on .glb files', () => {
        const output = { path: 'room.glb', sha256: 'b'.repeat(64), bytes: 40, dielectric: true };
        expect(validateSources({ assets: [rawAsset({ outputs: [output] })] })).toEqual([]);
        expect(validateSources({ assets: [rawAsset({ outputs: [{ ...output, dielectric: 'yes' }] })] }).join('\n')).toContain('dielectric: must be true or false');
        expect(validateSources({ assets: [rawAsset({ outputs: [{ ...output, path: 'room.gltf' }] })] }).join('\n')).toContain('dielectric: only supported for .glb');
    });
});

describe('generated assets (made by one of our scripts, no download)', () => {
    const generatedEntry = (overrides: Partial<AssetEntry> = {}): AssetEntry => {
        const { download: _download, ...rest } = asset();
        return {
            ...rest,
            license: 'MIT',
            sourceUrl: 'tools/assets/convert/x.ts',
            generated: { script: 'tools/assets/convert/x.ts', seed: 7, args: ['{dir}/x.glb'] },
            ...overrides,
        };
    };

    it('accepts an entry with a generator and a seed in place of a download', () => {
        expect(validateSources({ assets: [generatedEntry()] })).toEqual([]);
        expect(downloadsOf(generatedEntry())).toEqual([]);
    });

    it('turns the seed and arguments into a node conversion', () => {
        expect(generatedConvert({ script: 'tools/assets/convert/x.ts', seed: 7, args: ['out.glb'] })).toEqual({
            tool: 'node',
            script: 'tools/assets/convert/x.ts',
            args: ['--seed', '7', 'out.glb'],
        });
    });

    it('still needs a recorded sha256 and size for the output, so a changed generator shows up', () => {
        const entry = generatedEntry({ outputs: [{ path: 'x.glb' } as AssetOutput] });
        const text = validateSources({ assets: [entry] }).join('\n');
        expect(text).toContain('outputs[0].sha256');
        expect(text).toContain('outputs[0].bytes');
    });

    it('rejects a bad script, a missing seed, a download beside the generator, and a license that is not MIT', () => {
        const bad = (generated: unknown, overrides: Partial<AssetEntry> = {}) =>
            validateSources({ assets: [generatedEntry({ generated: generated as AssetEntry['generated'], ...overrides })] }).join('\n');
        expect(bad({ script: '../x.ts', seed: 1, args: [] })).toContain('script must be a repo-relative .ts path');
        expect(bad({ script: 'x.py', seed: 1, args: [] })).toContain('script must be a repo-relative .ts path');
        expect(bad({ script: 'x.ts', seed: 1.5, args: [] })).toContain('seed must be a whole number');
        expect(bad({ script: 'x.ts', seed: 1, args: [] }, { download: asset().download })).toContain('a generated asset has no download');
        expect(bad({ script: 'x.ts', seed: 1, args: [] }, { license: 'CC0-1.0' })).toContain('license is "MIT"');
    });

    it('still demands a download from an ordinary asset, and an allow-listed license (MIT alone is not enough)', () => {
        const { download: _download, ...noDownload } = asset();
        expect(validateSources({ assets: [noDownload] }).join('\n')).toContain('download: required');
        expect(validateSources({ assets: [asset({ license: 'MIT' })] }).join('\n')).toContain('allow list');
    });
});
