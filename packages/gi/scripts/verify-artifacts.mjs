// Checks that the build produced every public entry with its types, that
// nothing in dist names a private workspace package (a consumer could not
// resolve it), and that each entry loads only its own layers' chunks
// (layer-chunks.ts): `/core` no DDGI or three code, `/ddgi` no three code. The
// radiance cache's optional `@pathtracer/mlp` stays external, out of every .d.ts.
// `@pmndrs/bvh` (tinybvh's WebAssembly, the GPU builders, the workers) stays
// external too: nothing in dist embeds WebAssembly or tinybvh's notice.
import { existsSync, readdirSync, readFileSync } from 'node:fs';
import { basename, dirname, resolve } from 'node:path';

import { ENTRY_LAYERS, LAYER_CHUNKS } from '../layer-chunks.ts';

const packageRoot = resolve(import.meta.dirname, '..');
const dist = resolve(packageRoot, 'dist');
const publicEntries = ['index', 'core', 'ddgi', 'three', 'radiance-cascades', 'radiance-cache'];
const privatePackagePrefix = '@pmndrs/gi-';

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
for (const file of ['README.md', 'LICENSE', 'THIRD_PARTY.md', 'licenses/pathtracer-MIT.txt', 'licenses/pbrt-v4-Apache-2.0.txt', 'licenses/capsaicin-MIT.txt', 'licenses/rtoy-samples-MIT.txt', 'licenses/assimp-BSD-3-Clause.txt']) {
    if (!existsSync(resolve(packageRoot, file))) problems.push(`missing ${file} in packages/gi`);
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
// tinybvh's notice, which @pmndrs/bvh's WASM module and H-PLOC port carry: in our dist it would mean bvh got bundled.
const bundledBvhMarker = 'Jacco Bikker';

// Optional packages a technique loads at run time (vite.config.ts `OPTIONAL_RUNTIME_IMPORTS`):
// never bundled, never imported by a .d.ts (comments may name it), and loaded only through the hidden `import()` a
// consumer's bundler leaves alone (a string-literal one fails their build where it's absent).
const optionalRuntimeImports = ['@pathtracer/mlp'];
const keptOptionalImports = new Set();

for (const file of existsSync(dist) ? walk(dist) : []) {
    const text = readFileSync(file, 'utf8');
    if (text.includes(privatePackagePrefix)) {
        problems.push(`${file} references a private workspace package (${privatePackagePrefix}*)`);
    }
    for (const name of optionalRuntimeImports) {
        if (!text.includes(name)) continue;
        const quoted = String.raw`["']` + name.replace(/[.*+?^${}()|[\]\\/]/g, '\\$&') + String.raw`["']`;
        const imported = new RegExp(String.raw`(?:from\s*|import\s*\(\s*|import\s+)` + quoted).test(text);
        if (imported && file.endsWith('.d.ts')) problems.push(`${file} imports the optional package ${name}: its types must be inlined (tsconfig.build.json paths)`);
        else if (imported) problems.push(`${file} imports the optional package ${name} by a string literal, which a consumer's bundler fails on`);
        // vite.config.ts renders it as `((specifier) => import(specifier))('<name>')`.
        else if (new RegExp(String.raw`\)\)\(\s*` + quoted + String.raw`\s*\)`).test(text)) keptOptionalImports.add(name);
    }
    // Since pmndrs/gi#63 tinybvh ships in @pmndrs/bvh's distribution, not ours (THIRD_PARTY.md).
    if (file.endsWith('.js') && text.includes(embeddedWasmMarker)) problems.push(`${file} embeds WebAssembly: @pmndrs/bvh (tinybvh) must stay an external dependency`);
    if (file.endsWith('.js') && text.includes(bundledBvhMarker)) problems.push(`${file} holds tinybvh's notice: @pmndrs/bvh's code was bundled`);
}

for (const name of optionalRuntimeImports) {
    if (!keptOptionalImports.has(name)) problems.push(`no dist file loads ${name} through the hidden import(): was it bundled, or did the loader change?`);
}

// The layers each entry may load (ENTRY_LAYERS). Reaching a file of a layer outside its list
// means code of a higher layer leaked into a lower entry (the platform
// importing DDGI, or DDGI importing three), whatever the entry's name says.
const entryLayer = { 'radiance-cascades': 'rc', 'radiance-cache': 'radiance-cache', core: 'platform', ddgi: 'ddgi', three: 'three', index: 'three' };
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
console.log(`Verified dist: entries ${publicEntries.join(', ')} with types; /core loads only the platform chunk, /ddgi no three chunk; no private package names; no bundled @pmndrs/bvh; ${optionalRuntimeImports.join(', ')} external behind a hidden import(); README, LICENSE, THIRD_PARTY.md and licenses/ present.`);
