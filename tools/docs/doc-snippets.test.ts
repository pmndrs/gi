/**
 * The docs' TypeScript examples are typechecked (doc-snippets.ts): each
 * ```ts block of README.md, the adapter guide and the API docs is a file in
 * tools/docs/snippets/, which `pnpm typecheck` checks. This test fails when
 * those files don't match the docs, so an edited example can't skip the
 * check. Fix: `node tools/docs/doc-snippets.ts --write`, then `pnpm typecheck`.
 */

import { describe, expect, it } from 'vitest';
import { generateSnippetFiles, snippetFilesOnDisk, typescriptBlocks } from './doc-snippets.ts';

describe("the docs' code examples", () => {
    it('are up to date in tools/docs/snippets (run `node tools/docs/doc-snippets.ts --write`)', () => {
        const expected = generateSnippetFiles();
        const onDisk = snippetFilesOnDisk();
        expect([...onDisk.keys()].sort()).toEqual([...expected.keys()].sort());
        for (const [name, content] of expected) expect(onDisk.get(name), name).toBe(content);
    });

    it('include the guide and the README', () => {
        const names = [...generateSnippetFiles().keys()];
        expect(names.filter((name) => name.startsWith('adapter-guide-')).length).toBeGreaterThan(5);
        expect(names.some((name) => name.startsWith('readme-'))).toBe(true);
    });

    it('take ts blocks only, and skip one marked as not checked', () => {
        const markdown = ['```ts', 'const a = 1;', '```', '```json', '{}', '```', '<!-- snippet: not checked -->', '```ts', 'broken(', '```', '```typescript', 'const b = 2;', '```'].join('\n');
        expect(typescriptBlocks('test.md', markdown).map((block) => block.code)).toEqual(['const a = 1;', 'const b = 2;']);
    });
});
