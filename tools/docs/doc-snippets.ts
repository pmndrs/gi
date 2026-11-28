/**
 * The docs' TypeScript examples, typechecked (docs/design/m5-coverage.md §4).
 *
 * How it works:
 *
 * 1. Every ```ts block in the docs listed in `SNIPPET_DOCS` becomes one file
 *    in `tools/docs/snippets/`. The block's `import` lines go to the top of
 *    the file; the rest becomes the body of an async function, so `await`
 *    works and a block may declare any name.
 * 2. Names a block uses without declaring them (`renderer`, `scene`,
 *    `device`...) come from a context file per doc, `tools/docs/context/`,
 *    which declares each with its type and no value. The files are never
 *    run, only typechecked.
 * 3. `pnpm typecheck` checks the generated files with the rest of the
 *    repository (tsconfig.json includes `tools`).
 * 4. `pnpm test` fails when the generated files don't match the docs
 *    (doc-snippets.test.ts), so a changed example can't skip the check.
 *    After editing a doc, run:
 *
 *      node tools/docs/doc-snippets.ts --write
 *
 * A block that is deliberately incomplete can be left out by putting the
 * comment `<!-- snippet: not checked -->` on the line before it. Use it
 * rarely, and say why next to it.
 */

import { existsSync, mkdirSync, readdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import ts from 'typescript';
import { repositoryPath } from './public-exports.ts';

/** The docs whose examples are checked, and the context file their free names come from. */
export const SNIPPET_DOCS = [
    { doc: 'README.md', slug: 'readme', context: 'three' },
    { doc: 'docs/guides/adapter.md', slug: 'adapter-guide', context: 'core' },
    { doc: 'docs/api/platform.md', slug: 'platform', context: 'core' },
    { doc: 'docs/api/three.md', slug: 'three', context: 'three' },
    { doc: 'docs/api/radiance-cache.md', slug: 'radiance-cache', context: 'radiance-cache' },
] as const;

export const SNIPPETS_DIRECTORY = 'tools/docs/snippets';
const NOT_CHECKED_MARKER = '<!-- snippet: not checked -->';

export interface DocSnippet {
    readonly doc: string;
    /** 1-based line of the block's opening fence. */
    readonly line: number;
    readonly code: string;
}

/** The ```ts and ```typescript blocks of a Markdown file, except those marked as not checked. */
export function typescriptBlocks(doc: string, markdown: string): DocSnippet[] {
    const lines = markdown.split('\n');
    const blocks: DocSnippet[] = [];
    for (let index = 0; index < lines.length; index++) {
        const opening = /^(\s*)(```+)\s*(ts|typescript)\s*$/.exec(lines[index]);
        if (!opening) continue;
        const [, indent, fence] = opening;
        const body: string[] = [];
        let end = index + 1;
        while (end < lines.length && lines[end].trim() !== fence) body.push(lines[end++].slice(indent.length));
        if (end === lines.length) throw new Error(`${doc}:${index + 1}: a code block that never closes`);
        const marked = index > 0 && lines[index - 1].trim() === NOT_CHECKED_MARKER;
        if (!marked) blocks.push({ doc, line: index + 1, code: body.join('\n') });
        index = end;
    }
    return blocks;
}

/** The names a context file declares (`export declare const name: Type;`). */
function contextNames(context: string): string[] {
    const source = readFileSync(repositoryPath(`tools/docs/context/${context}.ts`), 'utf8');
    return [...source.matchAll(/^export declare (?:const|let|function) ([A-Za-z_$][\w$]*)/gm)].map((match) => match[1]);
}

/** Names a statement declares at the block's top level (`const x`, `function f`, `class C`). */
function declaredNames(statement: ts.Statement): string[] {
    if (ts.isVariableStatement(statement)) {
        return statement.declarationList.declarations.flatMap((declaration) => (ts.isIdentifier(declaration.name) ? [declaration.name.text] : []));
    }
    if ((ts.isFunctionDeclaration(statement) || ts.isClassDeclaration(statement)) && statement.name) return [statement.name.text];
    return [];
}

/**
 * Splits a block into its import declarations and everything else, using
 * TypeScript's parser. `boundNames` are the names the block binds itself
 * (imports and top-level declarations), which the context mustn't supply.
 */
function splitImports(code: string): { imports: string[]; boundNames: Set<string>; body: string } {
    const source = ts.createSourceFile('snippet.ts', code, ts.ScriptTarget.Latest, true, ts.ScriptKind.TS);
    const imports: string[] = [];
    const boundNames = new Set<string>();
    let bodyStart = 0;
    for (const statement of source.statements) {
        if (!ts.isImportDeclaration(statement)) break;
        imports.push(statement.getText(source));
        const clause = statement.importClause;
        if (clause?.name) boundNames.add(clause.name.text);
        const bindings = clause?.namedBindings;
        if (bindings && ts.isNamespaceImport(bindings)) boundNames.add(bindings.name.text);
        if (bindings && ts.isNamedImports(bindings)) for (const element of bindings.elements) boundNames.add(element.name.text);
        bodyStart = statement.getEnd();
    }
    const rest = source.statements.slice(imports.length);
    if (rest.some((statement) => ts.isImportDeclaration(statement))) throw new Error("put a code block's imports at its top");
    for (const statement of rest) for (const name of declaredNames(statement)) boundNames.add(name);
    return { imports, boundNames, body: code.slice(bodyStart).replace(/^\s*\n/, '') };
}

/** The generated file for one block. */
export function snippetFile(snippet: DocSnippet, context: string): string {
    const { imports, boundNames, body } = splitImports(snippet.code);
    const words = new Set(body.match(/[A-Za-z_$][\w$]*/g) ?? []);
    const fromContext = contextNames(context).filter((name) => words.has(name) && !boundNames.has(name));
    const indented = body
        .trimEnd()
        .split('\n')
        .map((line) => (line.length > 0 ? `    ${line}` : line))
        .join('\n');
    return [
        `// Generated by tools/docs/doc-snippets.ts from ${snippet.doc}, the code block at line ${snippet.line}.`,
        '// Don\'t edit: change the doc, then run `node tools/docs/doc-snippets.ts --write`.',
        ...imports,
        ...(fromContext.length > 0 ? [`import { ${fromContext.join(', ')} } from '../context/${context}.ts';`] : []),
        '',
        'export async function snippet(): Promise<void> {',
        indented,
        '}',
        '',
    ].join('\n');
}

/** Every generated file, by name, from the docs as they are now. */
export function generateSnippetFiles(): Map<string, string> {
    const files = new Map<string, string>();
    for (const { doc, slug, context } of SNIPPET_DOCS) {
        const path = repositoryPath(doc);
        if (!existsSync(path)) throw new Error(`${doc} is listed in SNIPPET_DOCS but doesn't exist`);
        typescriptBlocks(doc, readFileSync(path, 'utf8')).forEach((snippet, index) => {
            files.set(`${slug}-${String(index + 1).padStart(2, '0')}.ts`, snippetFile(snippet, context));
        });
    }
    return files;
}

/** The generated files on disk, by name. */
export function snippetFilesOnDisk(): Map<string, string> {
    const directory = repositoryPath(SNIPPETS_DIRECTORY);
    const files = new Map<string, string>();
    if (!existsSync(directory)) return files;
    for (const name of readdirSync(directory).filter((file) => file.endsWith('.ts')).sort()) files.set(name, readFileSync(join(directory, name), 'utf8'));
    return files;
}

function writeSnippetFiles(): void {
    const directory = repositoryPath(SNIPPETS_DIRECTORY);
    rmSync(directory, { recursive: true, force: true });
    mkdirSync(directory, { recursive: true });
    const files = generateSnippetFiles();
    for (const [name, content] of files) writeFileSync(join(directory, name), content);
    console.log(`wrote ${files.size} files to ${SNIPPETS_DIRECTORY}; now run pnpm typecheck`);
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
    if (process.argv.includes('--write')) writeSnippetFiles();
    else for (const [name, content] of generateSnippetFiles()) console.log(`// ---- ${name}\n${content}`);
}
