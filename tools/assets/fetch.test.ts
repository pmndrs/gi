import { createHash } from 'node:crypto';
import { EventEmitter } from 'node:events';
import { createServer, type Server } from 'node:http';
import type { AddressInfo, Socket } from 'node:net';
import { mkdir, mkdtemp, readFile, rm, utimes, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { afterAll, beforeAll, describe, expect, it, vi } from 'vitest';
import {
    checkGlbStructure,
    checkOutputs,
    DEFAULT_DOWNLOAD_TIMEOUTS,
    downloadFile,
    extractWithCache,
    isAllowedLicense,
    runConversionProcess,
    downloadsOf,
    fetchAsset,
    findBlender,
    parseBlenderVersion,
    readGlbStructure,
    versionMatches,
    formatBytes,
    isSafeRelativePath,
    parseArgs,
    selectAssets,
    sha256OfFile,
    sortBlenderFolders,
    substitutePlaceholders,
    validateSources,
    type AssetEntry,
    type AssetOutput,
} from './fetch';

const sha256 = (text: string) => createHash('sha256').update(text).digest('hex');

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

describe('validateSources', () => {
    it('accepts an empty asset list', () => {
        expect(validateSources({ assets: [] })).toEqual([]);
    });

    it('accepts a full entry', () => {
        const entry = asset({
            licenseUrl: 'https://example.com/license',
            download: { url: 'https://example.com/a.zip', file: 'a.zip', bytes: 100, sha256: 'a'.repeat(64) },
            extract: 'zip',
            convert: { tool: 'blender', script: 'tools/assets/convert/x.py', args: ['{dir}'] },
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
        const base = { url: 'https://example.com/a', file: 'a.bin' };
        expect(validateSources({ assets: [asset({ download: [] })] }).join('\n')).toContain('empty array');
        expect(validateSources({ assets: [asset({ download: [base, base] })] }).join('\n')).toContain('listed twice');
        expect(validateSources({ assets: [asset({ download: [{ ...base, file: '../a.bin' }] })] }).join('\n')).toContain('download[0].file');
        const text = validateSources({ assets: [asset({ download: [base, { ...base, file: 'b.bin' }], extract: 'zip' })] }).join('\n');
        expect(text).toContain('only works with a single download');
    });

    it('lets a skipped asset be unfinished, but needs a reason', () => {
        expect(validateSources({ assets: [{ id: 'sponza-khronos', skip: true, skipReason: 'License question is open.' }] })).toEqual([]);
        expect(validateSources({ assets: [{ id: 'sponza-khronos', skip: true }] }).join('\n')).toContain('skipReason');
    });

    it('requires sha256 and bytes on every download and output, and says which one is missing', () => {
        const bare = asset({
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

    it('allows only CC0 and CC-BY licenses', () => {
        for (const license of ['CC0-1.0', 'CC-BY-4.0', 'CC-BY-3.0']) expect(isAllowedLicense(license)).toBe(true);
        for (const license of ['CC-BY-SA-4.0', 'CC-BY-NC-4.0', 'CRYENGINE Limited License Agreement', 'MIT', 'cc0-1.0', '']) {
            expect(isAllowedLicense(license), license).toBe(false);
        }
        const text = validateSources({ assets: [asset({ license: 'CRYENGINE Limited License Agreement' })] }).join('\n');
        expect(text).toContain('is not on the allow list');
        expect(text).toContain('Ask Dennis');
        expect(validateSources({ assets: [asset({ license: 'CC-BY-4.0' })] })).toEqual([]);
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
        const bad = { ...asset(), id: 'Bad Id', download: { url: 'ftp://x', file: '../x.zip', sha256: 'zz' }, extract: 'rar', outputs: [] };
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
        const second = asset({ outputs: [{ path: '../elsewhere.glb' }] });
        const text = validateSources({ assets: [first, second] }).join('\n');
        expect(text).toContain('duplicate id');
        expect(text).toContain('outputs[0].path');
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
        expect(parseArgs([])).toEqual({ only: [], force: false, sourcesPath: null, hash: null, verifyOnly: false, help: false });
        expect(parseArgs(['--verify-only']).verifyOnly).toBe(true);
        expect(parseArgs(['--only', 'a', '--only', 'b,c', '--force', '--sources', 'x.json'])).toEqual({
            only: ['a', 'b', 'c'],
            force: true,
            sourcesPath: 'x.json',
            hash: null,
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
                response.writeHead(200, { 'content-type': 'application/octet-stream', 'content-length': Buffer.byteLength(body) });
                response.end(body);
            } else if (request.url === '/shared.bin') {
                response.writeHead(200, { 'content-length': Buffer.byteLength(sharedBody) });
                response.end(sharedBody);
            } else {
                response.writeHead(404).end();
            }
        });
        await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve));
        baseUrl = `http://127.0.0.1:${(server.address() as AddressInfo).port}`;
        vi.spyOn(console, 'log').mockImplementation(() => undefined);
    });

    afterAll(async () => {
        server.close();
        await rm(workDirectory, { recursive: true, force: true });
        vi.restoreAllMocks();
    });

    const entry = (overrides: Partial<AssetEntry> = {}) =>
        asset({
            download: { url: `${baseUrl}/model.glb`, file: 'model.glb', bytes: Buffer.byteLength(body), sha256: sha256(body) },
            outputs: [{ path: 'model.glb', sha256: sha256(body) }],
            ...overrides,
        });

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

    it('downloads again with force', async () => {
        const before = requests;
        await fetchAsset(entry(), { force: true, downloadsDirectory: workDirectory });
        expect(requests).toBe(before + 1);
    });

    it('fetches a shared download once for two assets', async () => {
        const shared = { url: `${baseUrl}/shared.bin`, file: 'shared.bin', sha256: sha256(sharedBody) };
        const directory = await mkdtemp(join(tmpdir(), 'ddgi-shared-'));
        const before = requests;
        const first = asset({ id: 'half-one', download: shared, outputs: [{ path: 'shared.bin', sha256: sha256(sharedBody) }] });
        const second = asset({ id: 'half-two', download: { ...shared }, outputs: [{ path: 'shared.bin', sha256: sha256(sharedBody) }] });
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
            download: [
                { url: `${baseUrl}/model.glb`, file: 'tree.gltf', sha256: sha256(body) },
                { url: `${baseUrl}/shared.bin`, file: 'textures/bark/diffuse.bin' },
            ],
            outputs: [{ path: 'tree.gltf' }, { path: 'textures/bark/diffuse.bin' }],
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
        const wrong = entry({
            id: 'wrong-hash',
            download: { url: `${baseUrl}/model.glb`, file: 'model.glb', sha256: sha256('something else') },
        });
        await expect(fetchAsset(wrong, { force: false, downloadsDirectory: workDirectory })).rejects.toThrow(/download sha256 is/);
        await expect(readFile(join(workDirectory, '.cache', `${sha256('something else')}.part`))).rejects.toThrow();
        await expect(readFile(join(workDirectory, 'wrong-hash', 'model.glb'))).rejects.toThrow();
    });

    it('fails clearly on a 404', async () => {
        const missing = entry({ id: 'gone', download: { url: `${baseUrl}/nope.glb`, file: 'nope.glb' }, outputs: [{ path: 'nope.glb' }] });
        await expect(fetchAsset(missing, { force: false, downloadsDirectory: workDirectory })).rejects.toThrow(/answered 404/);
    });

    it('fails clearly when the size differs from sources.json', async () => {
        const wrongSize = entry({
            id: 'wrong-size',
            download: { url: `${baseUrl}/model.glb`, file: 'model.glb', bytes: 5 },
            outputs: [{ path: 'model.glb' }],
        });
        await expect(fetchAsset(wrongSize, { force: false, downloadsDirectory: workDirectory })).rejects.toThrow(/bytes but sources.json says 5/);
    });

    it('reports missing and mismatched outputs, and counts unverified ones', async () => {
        const directory = join(workDirectory, 'check');
        await mkdir(directory, { recursive: true });
        await writeFile(join(directory, 'a.glb'), 'aaa');
        await writeFile(join(directory, 'b.glb'), 'bbb');
        const result = await checkOutputs(
            directory,
            [{ path: 'a.glb', sha256: sha256('aaa') }, { path: 'b.glb', sha256: sha256('not b') }, { path: 'c.glb' }, { path: 'a.glb' }],
            () => undefined,
        );
        expect(result.problems).toHaveLength(2);
        expect(result.problems[0]).toContain('sha256 differs: b.glb');
        expect(result.problems[1]).toBe('missing: c.glb');
        expect(result.unverified).toBe(1);
        expect(await sha256OfFile(join(directory, 'a.glb'))).toBe(sha256('aaa'));
    });
});

describe('picking a Blender by version', () => {
    it('reads the version from --version output', () => {
        expect(parseBlenderVersion('Blender 4.5.0 LTS\n\tbuild date: 2025-06-17')).toBe('4.5.0');
        expect(parseBlenderVersion('Blender 5.1.0 (hash adfe2921d5f3 built 2026-03-17)')).toBe('5.1.0');
        expect(parseBlenderVersion('not it')).toBeNull();
    });

    it('matches on major.minor only', () => {
        expect(versionMatches('4.5.0', '4.5')).toBe(true);
        expect(versionMatches('4.5.3', '4.5')).toBe(true);
        expect(versionMatches('4.50.0', '4.5')).toBe(false);
        expect(versionMatches('5.1.0', '4.5')).toBe(false);
        expect(versionMatches('14.5.0', '4.5')).toBe(false);
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
            const choice = await findBlender('4.5', { env: environment, platform: 'win32', readVersion });
            expect(choice).toEqual({ path: pathExecutable, version: '4.5.0' });
        });

        it('uses the BLENDER variable when its version matches', async () => {
            versions.set(variablePath, '4.5.0');
            versions.set(pathExecutable, '4.5.2');
            expect((await findBlender('4.5', { env: environment, platform: 'win32', readVersion })).path).toBe(variablePath);
        });

        it('takes the first runnable Blender when no version is asked for', async () => {
            versions.set(variablePath, null);
            versions.set(pathExecutable, '5.1.0');
            expect((await findBlender(undefined, { env: environment, platform: 'win32', readVersion })).version).toBe('5.1.0');
        });

        it('says which version is needed and what was found when none matches', async () => {
            versions.set(variablePath, '5.1.0');
            versions.set(pathExecutable, null);
            const message = await findBlender('4.5', { env: environment, platform: 'win32', readVersion }).catch((error: Error) => error.message);
            expect(message).toContain('Blender 4.5 is needed');
            expect(message).toContain('version 5.1.0');
            expect(message).toContain('would not run');
            expect(message).toContain('BLENDER');
        });

        it('says so when there is no Blender at all', async () => {
            const message = await findBlender('4.5', {
                env: { PATH: '', ProgramFiles: join(directory, 'nothing-here') },
                platform: 'win32',
                readVersion,
            }).catch((error: Error) => error.message);
            expect(message).toContain('No Blender was found at all');
        });
    });

    it('accepts a blenderVersion like "4.5" in sources.json and rejects other shapes', () => {
        const convert = (blenderVersion: string) => ({ tool: 'blender' as const, script: 'tools/assets/convert/x.py', args: [], blenderVersion });
        expect(validateSources({ assets: [asset({ convert: convert('4.5') })] })).toEqual([]);
        expect(validateSources({ assets: [asset({ convert: convert('4') })] }).join('\n')).toContain('blenderVersion');
        expect(validateSources({ assets: [asset({ convert: convert('4.5.0') })] }).join('\n')).toContain('blenderVersion');
    });

    it('blames the conversion when a converted output has the wrong hash', async () => {
        const directory = await mkdtemp(join(tmpdir(), 'ddgi-convhash-'));
        await writeFile(join(directory, 'a.glb'), 'bytes');
        const convert = { tool: 'blender' as const, script: 'x.py', args: [], blenderVersion: '4.5' };
        const converted = await checkOutputs(directory, [{ path: 'a.glb', sha256: sha256('other') }], () => undefined, convert);
        expect(converted.problems[0]).toMatch(/different Blender version \(this one needs 4\.5\) or conversion script is the likely cause/);
        const plain = await checkOutputs(directory, [{ path: 'a.glb', sha256: sha256('other') }], () => undefined);
        expect(plain.problems[0]).not.toContain('Blender');
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
        await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve));
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

describe('unpacking a shared zip once', () => {
    it('unpacks into a cache keyed by the zip hash and links it into each asset folder', async () => {
        const directory = await mkdtemp(join(tmpdir(), 'ddgi-unpack-'));
        const cache = join(directory, '.cache');
        const download = { url: 'https://example.com/pack.zip', file: 'pack.zip', bytes: 1, sha256: sha256('the zip') };
        let unpacks = 0;
        const fakeExtract = async (_archive: string, destination: string) => {
            unpacks++;
            await mkdir(join(destination, 'models'), { recursive: true });
            await writeFile(join(destination, 'models', 'a.fbx'), 'model data');
            await writeFile(join(destination, 'readme.txt'), 'hello');
        };
        const quiet = () => undefined;

        await extractWithCache(download, 'ignored.zip', join(directory, 'first'), cache, quiet, fakeExtract);
        await extractWithCache(download, 'ignored.zip', join(directory, 'second'), cache, quiet, fakeExtract);

        expect(unpacks).toBe(1);
        for (const asset of ['first', 'second']) {
            expect(await readFile(join(directory, asset, 'models', 'a.fbx'), 'utf8')).toBe('model data');
            expect(await readFile(join(directory, asset, 'readme.txt'), 'utf8')).toBe('hello');
        }
        await rm(directory, { recursive: true, force: true });
    });

    it('never reuses a half-finished unpack: a failed unpack leaves nothing in the cache', async () => {
        const directory = await mkdtemp(join(tmpdir(), 'ddgi-unpack-fail-'));
        const cache = join(directory, '.cache');
        const download = { url: 'https://example.com/pack.zip', file: 'pack.zip', bytes: 1, sha256: sha256('another zip') };
        const failing = async (_archive: string, destination: string) => {
            await writeFile(join(destination, 'half.bin'), 'partial');
            throw new Error('disk full');
        };
        await expect(extractWithCache(download, 'ignored.zip', join(directory, 'asset'), cache, () => undefined, failing)).rejects.toThrow('disk full');

        let unpacks = 0;
        const working = async (_archive: string, destination: string) => {
            unpacks++;
            await writeFile(join(destination, 'whole.bin'), 'complete');
        };
        await extractWithCache(download, 'ignored.zip', join(directory, 'asset'), cache, () => undefined, working);
        expect(unpacks).toBe(1);
        expect(await readFile(join(directory, 'asset', 'whole.bin'), 'utf8')).toBe('complete');
        await expect(readFile(join(directory, 'asset', 'half.bin'))).rejects.toThrow();
        await rm(directory, { recursive: true, force: true });
    });

    it('unpacks straight into the asset folder when the zip has no recorded hash', async () => {
        const directory = await mkdtemp(join(tmpdir(), 'ddgi-unpack-plain-'));
        const seen: string[] = [];
        await extractWithCache({ url: 'https://example.com/p.zip', file: 'p.zip' }, 'p.zip', join(directory, 'asset'), join(directory, '.cache'), () => undefined, async (_archive, destination) => {
            seen.push(destination);
        });
        expect(seen).toEqual([join(directory, 'asset')]);
        await rm(directory, { recursive: true, force: true });
    });
});

describe('output size checks', () => {
    it('reports a wrong size before hashing, and blames the conversion for converted files', async () => {
        const directory = await mkdtemp(join(tmpdir(), 'ddgi-size-'));
        await writeFile(join(directory, 'a.glb'), 'twelve bytes');
        const convert = { tool: 'blender' as const, script: 'x.py', args: [], blenderVersion: '4.5' };
        const result = await checkOutputs(directory, [{ path: 'a.glb', bytes: 99, sha256: sha256('twelve bytes') }], () => undefined, convert);
        expect(result.problems[0]).toMatch(/size differs: a\.glb \(expected 99 bytes, got 12\).*Blender version.*likely cause/);
        const ok = await checkOutputs(directory, [{ path: 'a.glb', bytes: 12, sha256: sha256('twelve bytes') }], () => undefined);
        expect(ok.problems).toEqual([]);
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
        expect(result).toEqual({ end: 'exited', code: 3 });
    });

    it('stops a process that wrote every output and then hangs, like Blender does at shutdown', async () => {
        const output = join(directory, 'hang-after.bin').replaceAll('\\', '/');
        const [command, args] = node(`require('fs').writeFileSync('${output}', 'done'); ${hangForever}`);
        const result = await runConversionProcess(command, [...args], {
            cwd: directory,
            assetDirectory: directory,
            outputs: [{ path: 'hang-after.bin', bytes: 4, sha256: sha256('done') }],
            limits: quickLimits,
        });
        expect(result.end).toBe('stopped-after-outputs');
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
        expect(result).toEqual({ end: 'exited', code: 0 });
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
        const check = await checkOutputs(directory, [{ path: 'wrong.bin', bytes: 4, sha256: sha256('done') }], () => undefined);
        expect(check.problems[0]).toContain('sha256 differs: wrong.bin');
    });
});

/** A minimal .glb: header, JSON chunk (padded with spaces to 4 bytes) and BIN chunk (padded with zeros). */
function makeGlb(json: string, bin: Buffer): Buffer {
    const jsonBytes = Buffer.from(json + ' '.repeat((4 - (json.length % 4)) % 4));
    const binBytes = Buffer.concat([bin, Buffer.alloc((4 - (bin.length % 4)) % 4)]);
    const header = Buffer.alloc(12);
    header.writeUInt32LE(0x46546c67, 0);
    header.writeUInt32LE(2, 4);
    header.writeUInt32LE(12 + 8 + jsonBytes.length + 8 + binBytes.length, 8);
    const jsonHeader = Buffer.alloc(8);
    jsonHeader.writeUInt32LE(jsonBytes.length, 0);
    jsonHeader.writeUInt32LE(0x4e4f534a, 4);
    const binHeader = Buffer.alloc(8);
    binHeader.writeUInt32LE(binBytes.length, 0);
    binHeader.writeUInt32LE(0x004e4942, 4);
    return Buffer.concat([header, jsonHeader, jsonBytes, binHeader, binBytes]);
}

describe('outputs of a converter that is not bit-reproducible', () => {
    const json = '{"asset":{"version":"2.0"},"meshes":[1,2,3]}';
    const bin = Buffer.from('0123456789abcdef');
    let directory: string;

    beforeAll(async () => {
        directory = await mkdtemp(join(tmpdir(), 'ddgi-glb-'));
    });
    afterAll(() => rm(directory, { recursive: true, force: true }));

    async function write(name: string, glb: Buffer): Promise<string> {
        const path = join(directory, name);
        await writeFile(path, glb);
        return path;
    }

    const padded = (text: string) => Buffer.from(text + ' '.repeat((4 - (text.length % 4)) % 4));

    /** The sources.json entry for a file as it was first converted. */
    function recordedFor(glb: Buffer) {
        return {
            path: 'model.glb',
            bytes: glb.length,
            sha256: createHash('sha256').update(glb).digest('hex'),
            reproducible: false,
            jsonSha256: createHash('sha256').update(padded(json)).digest('hex'),
            binBytes: bin.length,
        };
    }

    it('reads the JSON chunk hash and the BIN chunk length without reading the whole file', async () => {
        const glb = makeGlb(json, bin);
        const structure = await readGlbStructure(await write('read.glb', glb));
        expect(structure.binBytes).toBe(16);
        expect(structure.jsonSha256).toBe(recordedFor(glb).jsonSha256); // hash of the padded chunk as stored
    });

    it('rejects files that are not well-formed .glb', async () => {
        const good = makeGlb(json, bin);
        const badMagic = Buffer.from(good);
        badMagic.writeUInt32LE(0x12345678, 0);
        await expect(readGlbStructure(await write('magic.glb', badMagic))).rejects.toThrow('bad magic');

        await expect(readGlbStructure(await write('short.glb', good.subarray(0, 10)))).rejects.toThrow('too short');
        await expect(readGlbStructure(await write('cut.glb', good.subarray(0, good.length - 4)))).rejects.toThrow('header says');

        const noBin = Buffer.from(good.subarray(0, 12 + 8 + padded(json).length));
        noBin.writeUInt32LE(noBin.length, 8);
        await expect(readGlbStructure(await write('nobin.glb', noBin))).rejects.toThrow('no BIN chunk');
    });

    it('passes when only bytes inside the BIN chunk changed, as when the exporter wobbles a float', async () => {
        const original = makeGlb(json, bin);
        const recorded = recordedFor(original);
        const wobbled = Buffer.from(original);
        wobbled[wobbled.length - 3] ^= 0x01; // flip one bit inside the BIN data
        const path = await write('wobbled.glb', wobbled);
        expect(createHash('sha256').update(wobbled).digest('hex')).not.toBe(recorded.sha256);
        expect(await checkGlbStructure(path, recorded)).toEqual([]);
    });

    it('fails when the JSON chunk changed, even with the same file size', async () => {
        const recorded = recordedFor(makeGlb(json, bin));
        const changed = makeGlb(json.replace('[1,2,3]', '[1,2,4]'), bin);
        expect(changed.length).toBe(makeGlb(json, bin).length);
        const problems = await checkGlbStructure(await write('json.glb', changed), recorded);
        expect(problems).toHaveLength(1);
        expect(problems[0]).toContain('the JSON chunk differs');
    });

    it('fails when the BIN chunk has a different length', async () => {
        const recorded = recordedFor(makeGlb(json, bin));
        const problems = await checkGlbStructure(await write('bin.glb', makeGlb(json, Buffer.from('0123456789abcdef0123'))), recorded);
        expect(problems[0]).toContain('the BIN chunk is 20 bytes but sources.json says 16');
    });

    it('checkOutputs uses the structure instead of the whole-file hash, and logs the hash for information', async () => {
        const original = makeGlb(json, bin);
        const recorded = recordedFor(original);
        const wobbled = Buffer.from(original);
        wobbled[wobbled.length - 2] ^= 0x10;
        await write('model.glb', wobbled);
        const notes: string[] = [];
        const convert = { tool: 'blender' as const, script: 'x.py', args: [], blenderVersion: '4.5' };
        const result = await checkOutputs(directory, [recorded], (message) => notes.push(message), convert);
        expect(result.problems).toEqual([]);
        expect(notes.join('\n')).toContain('not bit-reproducible');
        expect(notes.join('\n')).toContain(createHash('sha256').update(wobbled).digest('hex'));

        // A different size is still caught before anything else.
        const resized = await checkOutputs(directory, [{ ...recorded, bytes: recorded.bytes + 4 }], () => undefined, convert);
        expect(resized.problems[0]).toContain('size differs');

        // The same file without the flag still fails on the whole-file hash.
        const { reproducible: _flag, jsonSha256: _json, binBytes: _bin, ...strict } = recorded;
        expect((await checkOutputs(directory, [strict], () => undefined, convert)).problems[0]).toContain('sha256 differs');
    });

    it('sources.json must give jsonSha256 and binBytes with reproducible false, and only for .glb files', () => {
        const output = { path: 'a.glb', sha256: 'b'.repeat(64), bytes: 40, reproducible: false };
        const text = validateSources({ assets: [asset({ outputs: [output as never] })] }).join('\n');
        expect(text).toContain('jsonSha256: required with reproducible false');
        expect(text).toContain('binBytes: required with reproducible false');
        const complete = { ...output, jsonSha256: 'c'.repeat(64), binBytes: 16 };
        expect(validateSources({ assets: [asset({ outputs: [complete as never] })] })).toEqual([]);
        const notGlb = { ...complete, path: 'a.png' };
        expect(validateSources({ assets: [asset({ outputs: [notGlb as never] })] }).join('\n')).toContain('only supported for .glb');
        const badFlag = { ...complete, reproducible: 'no' };
        expect(validateSources({ assets: [asset({ outputs: [badFlag as never] })] }).join('\n')).toContain('reproducible: must be true or false');
    });
});

/** A stand-in for a child process whose life the test controls event by event. */
class FakeChild extends EventEmitter {
    pid = 4242;
    exitCode: number | null = null;
    signalCode: string | null = null;

    /** The process ends by itself (or is killed): 'exit' fires, then 'close' once its output is flushed. */
    finish(code: number): void {
        this.exitCode = code;
        this.emit('exit', code, null);
        this.emit('close', code);
    }
}

describe('who gets to end a conversion: the process or the guard', () => {
    const fastLimits = { timeoutMs: 400, graceMs: 0, pollMs: 5 };
    let directory: string;
    let outputPath: string;
    const expectedOutput = { path: 'made.bin', bytes: 4, sha256: sha256('done') };

    beforeAll(async () => {
        directory = await mkdtemp(join(tmpdir(), 'ddgi-race-'));
        outputPath = join(directory, 'made.bin');
        await writeFile(outputPath, 'done');
    });
    afterAll(() => rm(directory, { recursive: true, force: true }));

    const run = (child: FakeChild, outputs: AssetOutput[], limits = fastLimits) => {
        let killed = 0;
        const result = runConversionProcess('fake', [], {
            cwd: directory,
            assetDirectory: directory,
            outputs,
            limits,
            spawnProcess: () => child,
            killTree: async () => {
                killed++;
                child.finish(1); // a killed process ends with a failure code, like taskkill /F
            },
        });
        return { result, killed: () => killed };
    };

    it('reports the real exit code of a process that ends by itself after writing its output, even with no grace period', async () => {
        const child = new FakeChild();
        const { result, killed } = run(child, [expectedOutput]);
        child.finish(7); // exits before the guard has looked at anything
        // Give the guard every chance to misfire (any wait only makes a wrong implementation more likely to fail, never a right one).
        await new Promise((resolve) => setTimeout(resolve, 100));
        expect(await result).toEqual({ end: 'exited', code: 7 });
        expect(killed()).toBe(0);
    });

    it('does not mistake a process that has exited but not yet closed for one to stop', async () => {
        const child = new FakeChild();
        const { result, killed } = run(child, [expectedOutput]);
        child.exitCode = 9;
        child.emit('exit', 9, null); // exited; the 'close' event has not arrived yet
        await new Promise((resolve) => setTimeout(resolve, 100)); // the output is verified and the zero-length grace period passes
        child.emit('close', 9);
        expect(await result).toEqual({ end: 'exited', code: 9 });
        expect(killed()).toBe(0);
    });

    it('kills a process that is still running once its outputs are verified, and says the guard did it', async () => {
        const child = new FakeChild();
        const { result, killed } = run(child, [expectedOutput]);
        expect(await result).toEqual({ end: 'stopped-after-outputs', code: 1 });
        expect(killed()).toBe(1);
    });

    it('never stops early a conversion that lists no outputs; only the time limit ends it', async () => {
        const child = new FakeChild();
        const { result, killed } = run(child, [], { ...fastLimits, timeoutMs: 150 });
        expect(await result).toEqual({ end: 'timed-out', code: 1 });
        expect(killed()).toBe(1);
    });

    it('does not treat an output that never matches as finished', async () => {
        const child = new FakeChild();
        const { result } = run(child, [{ ...expectedOutput, bytes: 5 }], { ...fastLimits, timeoutMs: 150 });
        expect((await result).end).toBe('timed-out');
    });
});
