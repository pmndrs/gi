// Checks that the build produced every public entry with its types, that
// nothing in dist names a private workspace package (a consumer could not
// resolve it), and that each entry loads only its own layers' chunks
// (layer-chunks.ts): `/core` no DDGI or three code, `/ddgi` no three code.
import { existsSync, readdirSync, readFileSync } from 'node:fs';
import { basename, dirname, resolve } from 'node:path';

import { ENTRY_LAYERS, LAYER_CHUNKS } from '../layer-chunks.ts';

const packageRoot = resolve(import.meta.dirname, '..');
const dist = resolve(packageRoot, 'dist');
const publicEntries = ['index', 'core', 'ddgi', 'three', 'radiance-cascades'];
const privatePackagePrefix = '@pmndrs/ddgi-';

const problems = [];

for (const entry of publicEntries) {
    for (const extension of ['js', 'd.ts']) {
        const file = resolve(dist, `${entry}.${extension}`);
        if (!existsSync(file)) problems.push(`missing dist/${entry}.${extension}`);
    }
}

// The files npm ships beside dist: README and LICENSE are copied from the
// repository root by tools/release/package-files.ts; THIRD_PARTY.md is the
// package's own; licenses/ holds the bundled code's licence texts it points to.
for (const file of ['README.md', 'LICENSE', 'THIRD_PARTY.md', 'licenses/tinybvh.txt', 'licenses/emscripten-runtime.txt', 'licenses/hiprt.txt', 'licenses/pathtracer-MIT.txt', 'licenses/pbrt-v4-Apache-2.0.txt']) {
    if (!existsSync(resolve(packageRoot, file))) problems.push(`missing ${file} in packages/ddgi`);
}

// Every path in package.json `exports` must exist too.
const manifest = JSON.parse(readFileSync(resolve(packageRoot, 'package.json'), 'utf8'));
for (const [subpath, conditions] of Object.entries(manifest.exports)) {
    for (const target of Object.values(conditions)) {
        if (!existsSync(resolve(packageRoot, target))) problems.push(`exports["${subpath}"] points at missing ${target}`);
    }
}

function walk(directory) {
    return readdirSync(directory, { withFileTypes: true }).flatMap((entry) => {
        const path = resolve(directory, entry.name);
        return entry.isDirectory() ? walk(path) : [path];
    });
}

// A WebAssembly module embedded as base64 starts with these characters (the bytes "\0asm", then version 1).
const embeddedWasmMarker = 'AGFzbQEAAAA';
// The tinybvh WASM must carry its licences (packages/bvh/licenses), which the
// generated module puts in a /*! */ comment for bundlers to keep.
const wasmLicenseLines = ['Copyright (c) 2024 Jacco Bikker', 'Copyright (c) 2010-2014 Emscripten authors', 'Rich Felker'];

for (const file of existsSync(dist) ? walk(dist) : []) {
    const text = readFileSync(file, 'utf8');
    if (text.includes(privatePackagePrefix)) {
        problems.push(`${file} references a private workspace package (${privatePackagePrefix}*)`);
    }
    if (file.endsWith('.js') && text.includes(embeddedWasmMarker)) {
        for (const line of wasmLicenseLines) {
            if (!text.includes(line)) problems.push(`${file} embeds WebAssembly but lacks the licence line "${line}"`);
        }
    }
}

// The layers each entry may load (ENTRY_LAYERS). Reaching a file of a layer outside its list
// means code of a higher layer leaked into a lower entry (the platform
// importing DDGI, or DDGI importing three), whatever the entry's name says.
const entryLayer = { 'radiance-cascades': 'rc', core: 'platform', ddgi: 'ddgi', three: 'three', index: 'three' };
const chunksDirectory = resolve(dist, 'chunks');

/** The layer a dist file belongs to: a chunk's by its name, an entry's by `entryLayer`. */
function layerOf(file) {
    const name = basename(file).replace(/\.d\.ts$|\.js$/, '');
    if (dirname(file) === chunksDirectory) return LAYER_CHUNKS.find((layer) => name.startsWith(`${layer}-`));
    if (dirname(file) === dist) return entryLayer[name];
    return undefined;
}

/** The relative files an entry reaches, following `from './x.js'` and `import('./x.js')` (.d.ts files import chunks as .js). */
function reachedFiles(entryFile) {
    const reached = new Set();
    const queue = [entryFile];
    while (queue.length) {
        const file = queue.pop();
        if (reached.has(file) || !existsSync(file)) continue;
        reached.add(file);
        const text = readFileSync(file, 'utf8');
        for (const match of text.matchAll(/(?:from\s*|import\s*\(\s*|import\s+)["'](\.{1,2}\/[^"']+)["']/g)) {
            let target = resolve(dirname(file), match[1]);
            if (file.endsWith('.d.ts') && target.endsWith('.js')) target = `${target.slice(0, -'.js'.length)}.d.ts`;
            queue.push(target);
        }
    }
    return reached;
}

for (const [entry, layers] of Object.entries(ENTRY_LAYERS)) {
    for (const extension of ['js', 'd.ts']) {
        for (const file of reachedFiles(resolve(dist, `${entry}.${extension}`))) {
            const layer = layerOf(file);
            if (layer && !layers.includes(layer)) {
                problems.push(`dist/${entry}.${extension} loads ${basename(file)}, which holds ${layer} code; /${entry} may load only ${layers.join(', ')}`);
            }
        }
    }
}
// The layer chunks must exist, or the check above checked nothing.
const chunkNames = existsSync(chunksDirectory) ? readdirSync(chunksDirectory) : [];
for (const layer of LAYER_CHUNKS) {
    for (const extension of ['js', 'd.ts']) {
        const isKind = (name) => (extension === 'd.ts' ? name.endsWith('.d.ts') : name.endsWith('.js') && !name.endsWith('.d.ts'));
        if (!chunkNames.some((name) => name.startsWith(`${layer}-`) && isKind(name))) {
            problems.push(`dist/chunks has no ${layer}-*.${extension}: the layer chunks (layer-chunks.ts) weren't made`);
        }
    }
}

if (problems.length) {
    console.error(problems.join('\n'));
    process.exit(1);
}
console.log(`Verified dist: entries ${publicEntries.join(', ')} with types; /core loads only the platform chunk, /ddgi no three chunk; no private package names; README, LICENSE, THIRD_PARTY.md and licenses/ present.`);
