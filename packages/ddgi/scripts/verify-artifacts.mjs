// Checks that the build produced every public entry with its types, and that
// nothing in dist names a private workspace package (a consumer could not
// resolve it).
import { existsSync, readdirSync, readFileSync } from 'node:fs';
import { resolve } from 'node:path';

const packageRoot = resolve(import.meta.dirname, '..');
const dist = resolve(packageRoot, 'dist');
const publicEntries = ['index', 'core', 'three'];
const privatePackagePrefix = '@pmndrs/ddgi-';

const problems = [];

for (const entry of publicEntries) {
    for (const extension of ['js', 'd.ts']) {
        const file = resolve(dist, `${entry}.${extension}`);
        if (!existsSync(file)) problems.push(`missing dist/${entry}.${extension}`);
    }
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

for (const file of existsSync(dist) ? walk(dist) : []) {
    if (readFileSync(file, 'utf8').includes(privatePackagePrefix)) {
        problems.push(`${file} references a private workspace package (${privatePackagePrefix}*)`);
    }
}

if (problems.length) {
    console.error(problems.join('\n'));
    process.exit(1);
}
console.log(`Verified dist: entries ${publicEntries.join(', ')} with types, no private package names.`);
