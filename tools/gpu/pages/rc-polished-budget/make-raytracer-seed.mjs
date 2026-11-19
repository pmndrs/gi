// Producer only. Run from the GI root with a populated pnpm 10.23.0 store.
// node tools/gpu/pages/rc-polished-budget/make-raytracer-seed.mjs "$(pnpm store path)"
import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { copyFileSync, existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { dirname, join, resolve } from 'node:path';

const pin = '2489a09ab1844a7effe6e845ecccd05876580c9f';
// git tree d491873c4ec934062d791da399b5106261b663a9, packages/raytracer at pin.
// SHA-256 of sorted "<path> <git blob sha1>" records, joined with LF.
const expectedTreeFingerprint = 'e71da85c560391eb61ffd41cbd90356bbdf9526947eb10acd6d4681aa43602b3';
const hash = (algorithm, bytes, encoding = 'hex') => createHash(algorithm).update(bytes).digest(encoding);
const store = process.argv[2];
if (!store) throw new Error('Pass the path printed by pnpm store path.');
const out = resolve(process.argv[3] ?? 'bench/screenshots/rc-metal-handoff/raytracer-pnpm-seed.tgz');
const packageKey = `git+https+++github.com+pmndrs+pathtracer.git+${pin}&path+packages+raytracer`;
// pnpm 10 shortens non-registry index directories at 60 chars on Windows and 120 on macOS.
const keys = [packageKey, packageKey.slice(0, 27) + '_' + hash('sha256', packageKey).slice(0, 32)];
const inputIndex = keys.map(key => join(store, key, 'integrity.json')).find(existsSync);
if (!inputIndex) throw new Error('The pinned raytracer package is missing from this pnpm store.');
const input = JSON.parse(readFileSync(inputIndex, 'utf8'));
if (input.name !== '@pmndrs/raytracer' || input.version !== '0.0.0' || input.requiresBuild) throw new Error('Unexpected package metadata.');
const stage = mkdtempSync(join(tmpdir(), 'gi-rc-store-export-'));
if (dirname(resolve(stage)) !== resolve(tmpdir())) throw new Error('Unexpected temporary export directory.');
const outputStore = join(stage, 'pnpm-store', 'v10');
function blobPath(root, digest) { return join(root, 'files', digest.slice(0, 2), digest.slice(2)); }

try {
    const gitBlobs = [];
    const files = {};
    for (const name of Object.keys(input.files).sort()) {
        const item = input.files[name];
        if (!item.integrity.startsWith('sha512-') || (item.mode & 0o111)) throw new Error('Unexpected integrity or executable entry: ' + name);
        const digest = Buffer.from(item.integrity.slice(7), 'base64').toString('hex');
        const original = readFileSync(blobPath(store, digest));
        if (hash('sha512', original) !== digest) throw new Error('Bad store content: ' + name);
        // Restore Git's LF text on an exporter with core.autocrlf enabled.
        const bytes = name.endsWith('.png') ? original : Buffer.from(original.toString('utf8').replaceAll('\r\n', '\n'));
        const gitBlob = createHash('sha1').update(`blob ${bytes.length}\0`).update(bytes).digest('hex');
        gitBlobs.push(name + ' ' + gitBlob);
        const dataHash = hash('sha512', bytes);
        const path = blobPath(outputStore, dataHash);
        mkdirSync(dirname(path), { recursive: true });
        writeFileSync(path, bytes);
        files[name] = { checkedAt: 0, mode: 0o644, size: bytes.length, integrity: 'sha512-' + Buffer.from(dataHash, 'hex').toString('base64') };
    }
    if (hash('sha256', gitBlobs.sort().join('\n')) !== expectedTreeFingerprint) throw new Error('Package differs from the pinned Git tree.');
    for (const key of keys) {
        const dir = join(outputStore, key);
        mkdirSync(dir, { recursive: true });
        writeFileSync(join(dir, 'integrity.json'), JSON.stringify({ name: input.name, version: input.version, requiresBuild: false, files }));
    }
    const notices = join(stage, 'licenses');
    mkdirSync(notices);
    const source = 'packages/ddgi-core/src/techniques/radiance-cascades/licenses';
    for (const file of ['pathtracer-MIT.txt', 'pbrt-v4-Apache-2.0.txt']) copyFileSync(join(source, file), join(notices, file));
    writeFileSync(join(stage, 'README.txt'), [
        'Internal GI / Apple Metal benchmark dependency seed',
        `@pmndrs/raytracer; pmndrs/pathtracer ${pin}, packages/raytracer`,
        `Pinned Git tree SHA-256 (path/blob records): ${expectedTreeFingerprint}`,
        'The subpackage is MIT licensed, copyright Dennis Smolek. The equal-area square-to-sphere mapping is adapted from pbrt-v4,',
        'copyright 1998–2020 Matt Pharr, Wenzel Jakob and Greg Humphreys (Apache-2.0). Full notices are under licenses/.',
        '',
        'From an extracted GI worktree containing the matching pnpm-lock.yaml, on the Mac:',
        '  KIT_STORE="$(mktemp -d)"',
        '  tar -xzf /path/to/raytracer-pnpm-seed.tgz -C "$KIT_STORE"',
        '  pnpm install --frozen-lockfile --store-dir="$KIT_STORE/pnpm-store"',
        '  node tools/gpu/pages/rc-polished-budget/metal.ts --out "$PWD/rc-metal-results"',
        'Requires Node 24, pnpm 10.23.0, Chrome with WebGPU, and public npm for the remaining dependencies.',
        '',
    ].join('\n'));
    mkdirSync(dirname(out), { recursive: true });
    execFileSync('tar', ['-czf', out, '-C', stage, 'pnpm-store', 'licenses', 'README.txt']);
    console.log(`${out}\n${gitBlobs.length} files from the pinned package. SHA-256 ${hash('sha256', readFileSync(out))}`);
} finally {
    rmSync(stage, { recursive: true, force: true });
}
