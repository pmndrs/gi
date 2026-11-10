/**
 * The docs-cover-exports check (docs/design/m5-coverage.md §4): every public
 * export of `@pmndrs/ddgi/core` must be named in docs/api/platform.md, and
 * every export of `@pmndrs/ddgi/three` in docs/api/three.md.
 *
 * "Named" means the name appears as a whole word inside code: an inline
 * `code span` or a fenced code block. Plain prose doesn't count, so a type
 * called `Frame` isn't covered by the word "frame" in a sentence.
 *
 * Run by `pnpm test` (docs-cover-exports.test.ts), or on its own:
 *
 *   node tools/docs/docs-cover-exports.ts      # lists what's missing, exits 1 if anything is
 */

import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { PUBLIC_ENTRY_POINTS, publicExports, repositoryPath, type PublicEntryPoint } from './public-exports.ts';

/** All the code in a Markdown file: fenced blocks and inline code spans, joined. */
export function markdownCode(markdown: string): string {
    const pieces: string[] = [];
    const fence = /^(```|~~~)[^\n]*\n([\s\S]*?)^\1[ \t]*$/gm;
    const withoutFences = markdown.replace(fence, (_match, _marker: string, body: string) => {
        pieces.push(body);
        return '';
    });
    // Inline spans: one or more backticks, then the same number to close.
    for (const match of withoutFences.matchAll(/(`+)([^`]|[^`][\s\S]*?[^`])\1(?!`)/g)) pieces.push(match[2]);
    return pieces.join('\n');
}

/** The names `code` doesn't mention as whole words. */
export function namesMissingFrom(code: string, names: readonly string[]): string[] {
    const words = new Set(code.match(/[A-Za-z_$][\w$]*/g) ?? []);
    return names.filter((name) => !words.has(name));
}

export interface CoverageReport {
    readonly entry: PublicEntryPoint;
    readonly exported: number;
    readonly missing: readonly string[];
}

export function checkDocsCoverExports(): CoverageReport[] {
    return PUBLIC_ENTRY_POINTS.map((entry) => {
        const names = publicExports(entry);
        const code = markdownCode(readFileSync(repositoryPath(entry.doc), 'utf8'));
        return { entry, exported: names.length, missing: namesMissingFrom(code, names) };
    });
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
    let failed = false;
    for (const report of checkDocsCoverExports()) {
        const covered = report.exported - report.missing.length;
        console.log(`${report.entry.name}: ${covered} of ${report.exported} exports named in ${report.entry.doc}`);
        for (const name of report.missing) console.log(`  missing: ${name}`);
        if (report.missing.length > 0) failed = true;
    }
    process.exitCode = failed ? 1 : 0;
}
