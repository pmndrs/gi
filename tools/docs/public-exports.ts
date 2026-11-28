/**
 * Lists the public exports of the published package's entry points:
 * `@pmndrs/gi/core` (the engine-free platform), `@pmndrs/gi/ddgi` (the DDGI
 * technique, engine-free), `@pmndrs/gi/three` (the three.js adapter;
 * `@pmndrs/gi` itself is the same list) and `@pmndrs/gi/radiance-cache` (the
 * radiance cache, engine-free).
 *
 * It reads the published package's entry files (packages/gi/src) with
 * TypeScript's parser and follows their `export * from` and
 * `export { ... } from` lines through the source, into the private workspace
 * packages (`@pmndrs/gi-*`, by their `exports`), so it
 * needs no type checking and runs in well under a second. Every form of
 * export the packages use is handled; one it doesn't know throws, naming the
 * file, so a new form can't drop names silently.
 *
 *   node tools/docs/public-exports.ts          # prints the names, one per line
 */

import { existsSync, readdirSync, readFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import ts from 'typescript';

const repositoryRoot = resolve(dirname(fileURLToPath(import.meta.url)), '../..');

/** The published entry points: each one's entry file in the published package, which re-exports the private packages. */
export const PUBLIC_ENTRY_POINTS = [
    { name: '@pmndrs/gi/core', index: 'packages/gi/src/core.ts', doc: 'docs/api/platform.md' },
    { name: '@pmndrs/gi/ddgi', index: 'packages/gi/src/ddgi.ts', doc: 'docs/api/platform.md' },
    { name: '@pmndrs/gi/three', index: 'packages/gi/src/three.ts', doc: 'docs/api/three.md' },
    { name: '@pmndrs/gi/radiance-cache', index: 'packages/gi/src/radiance-cache.ts', doc: 'docs/api/radiance-cache.md' },
] as const;

export type PublicEntryPoint = (typeof PUBLIC_ENTRY_POINTS)[number];

/** The private workspace packages by name (`@pmndrs/gi-core`, ...): their folders under packages/. */
const workspacePackages = new Map(
    readdirSync(join(repositoryRoot, 'packages'), { withFileTypes: true })
        .filter((entry) => entry.isDirectory() && existsSync(join(repositoryRoot, 'packages', entry.name, 'package.json')))
        .map((entry) => {
            const folder = join(repositoryRoot, 'packages', entry.name);
            return [(JSON.parse(readFileSync(join(folder, 'package.json'), 'utf8')) as { name: string }).name, folder] as const;
        })
        .filter(([name]) => name.startsWith('@pmndrs/gi-')),
);

/**
 * `./contract` → `.../contract.ts` or `.../contract/index.ts`; a private workspace package (`@pmndrs/gi-core`,
 * `@pmndrs/gi-core/platform/index`) → its file, as its `exports` map it (`.` is `src/index.ts`, `./*` is
 * `src/*.ts`); any other bare package name → null (not ours to list).
 */
function resolveRelative(fromFile: string, specifier: string): string | null {
    let base: string;
    if (specifier.startsWith('.')) {
        base = resolve(dirname(fromFile), specifier);
    } else {
        const match = [...workspacePackages].find(([name]) => specifier === name || specifier.startsWith(`${name}/`));
        if (!match) return null;
        const [name, folder] = match;
        base = specifier === name ? join(folder, 'src/index.ts') : join(folder, 'src', specifier.slice(name.length + 1));
    }
    const candidates = base.endsWith('.ts') ? [base] : [`${base}.ts`, join(base, 'index.ts')];
    const found = candidates.find((candidate) => existsSync(candidate));
    if (!found) throw new Error(`${fromFile}: can't find the module '${specifier}'`);
    return found;
}

function hasExportModifier(statement: ts.Statement): boolean {
    return ts.canHaveModifiers(statement) && (ts.getModifiers(statement) ?? []).some((modifier) => modifier.kind === ts.SyntaxKind.ExportKeyword);
}

function bindingNames(name: ts.BindingName): string[] {
    if (ts.isIdentifier(name)) return [name.text];
    return name.elements.flatMap((element) => (ts.isOmittedExpression(element) ? [] : bindingNames(element.name)));
}

/** Every name a module exports, following re-exports through our own source. */
function moduleExports(file: string, visiting: Set<string> = new Set()): Set<string> {
    if (visiting.has(file)) return new Set();
    visiting.add(file);
    const source = ts.createSourceFile(file, readFileSync(file, 'utf8'), ts.ScriptTarget.Latest, false, ts.ScriptKind.TS);
    const names = new Set<string>();
    for (const statement of source.statements) {
        if (ts.isExportDeclaration(statement)) {
            const specifier = statement.moduleSpecifier && ts.isStringLiteral(statement.moduleSpecifier) ? statement.moduleSpecifier.text : null;
            const clause = statement.exportClause;
            if (clause === undefined) {
                // `export * from './x'` (or `export type * from`): every name of the target.
                const target = specifier ? resolveRelative(file, specifier) : null;
                if (!target) throw new Error(`${file}: 'export *' from the package '${specifier}' can't be listed; name the exports instead`);
                for (const name of moduleExports(target, visiting)) names.add(name);
            } else if (ts.isNamespaceExport(clause)) {
                names.add(clause.name.text);
            } else {
                for (const element of clause.elements) names.add(element.name.text);
            }
            continue;
        }
        if (ts.isExportAssignment(statement)) {
            names.add('default');
            continue;
        }
        if (!hasExportModifier(statement)) continue;
        if (ts.isVariableStatement(statement)) {
            for (const declaration of statement.declarationList.declarations) for (const name of bindingNames(declaration.name)) names.add(name);
        } else if (
            ts.isFunctionDeclaration(statement) ||
            ts.isClassDeclaration(statement) ||
            ts.isInterfaceDeclaration(statement) ||
            ts.isTypeAliasDeclaration(statement) ||
            ts.isEnumDeclaration(statement) ||
            ts.isModuleDeclaration(statement)
        ) {
            if (!statement.name) throw new Error(`${file}: an exported declaration without a name (export default?) isn't handled`);
            names.add(statement.name.text);
        } else {
            throw new Error(`${file}: an export of kind ${ts.SyntaxKind[statement.kind]} isn't handled by tools/docs/public-exports.ts`);
        }
    }
    return names;
}

/** The sorted public names of one entry point. */
export function publicExports(entry: PublicEntryPoint): string[] {
    return [...moduleExports(join(repositoryRoot, entry.index))].sort();
}

export function repositoryPath(relative: string): string {
    return join(repositoryRoot, relative);
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
    for (const entry of PUBLIC_ENTRY_POINTS) {
        const names = publicExports(entry);
        console.log(`# ${entry.name} (${names.length} names; documented in ${entry.doc})`);
        for (const name of names) console.log(name);
    }
}
