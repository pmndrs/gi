import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { afterAll, describe, expect, it } from 'vitest';
import { optionalSiblingsPlugin, optionalSiblingState, type SiblingEntry } from './optional-siblings.ts';

const PIN = '0e4435bd05bf4f55c8763c76facae640db35855c';
const root = mkdtempSync(join(tmpdir(), 'gi-optional-siblings-'));
const repo = join(root, 'gi');
mkdirSync(join(root, 'sibling', 'packages', 'mlp', 'src'), { recursive: true });
mkdirSync(repo);
writeFileSync(join(root, 'sibling', 'packages', 'mlp', 'src', 'index.ts'), 'export {};\n');
const entry: SiblingEntry = { repo: 'https://example.invalid/sibling.git', commit: PIN, clone: '../sibling', path: 'packages/mlp', built: ['src/index.ts'], optional: true };

afterAll(() => rmSync(root, { recursive: true, force: true }));

describe('optional siblings (P6.4: @pathtracer/mlp)', () => {
    it('is ready only with the clone at its pinned commit', () => {
        const ready = optionalSiblingState('@pathtracer/mlp', entry, repo, () => PIN);
        expect(ready.state).toBe('ready');
        expect(ready.entry).toBe(join(root, 'sibling', 'packages', 'mlp', 'src', 'index.ts'));
    });

    it('refuses a clone at another commit (the drift check), naming both', () => {
        const drifted = optionalSiblingState('@pathtracer/mlp', entry, repo, () => 'a'.repeat(40));
        expect(drifted.state).toBe('drifted');
        expect(drifted.reason).toContain('aaaaaaaaa');
        expect(drifted.reason).toContain(PIN.slice(0, 9));
    });

    it('is missing where nothing is cloned, with the fix', () => {
        const missing = optionalSiblingState('@pathtracer/mlp', { ...entry, clone: '../not-cloned' }, repo, () => PIN);
        expect(missing.state).toBe('missing');
        expect(missing.entry).toBeNull();
        expect(missing.reason).toContain('linked-siblings.ts --fetch');
    });

    it('takes a built folder that is not a git clone as it is', () => {
        expect(optionalSiblingState('@pathtracer/mlp', entry, repo, () => '').state).toBe('ready');
    });

    it('keeps a ready clone out of production builds unless asked, and no local path in the stub', () => {
        const ready = optionalSiblingState('@pathtracer/mlp', entry, repo, () => PIN);
        const missing = optionalSiblingState('@pathtracer/mlp', { ...entry, clone: '../not-cloned' }, repo, () => PIN);
        const run = (sibling: typeof ready, command: 'serve' | 'build', env: Record<string, string> = {}): { resolved: string; stub: string | null } => {
            const plugin = optionalSiblingsPlugin([sibling], env);
            (plugin.configResolved as unknown as (config: { command: string }) => void)({ command });
            const resolved = (plugin.resolveId as unknown as (id: string) => string)('@pathtracer/mlp');
            const stub = (plugin.load as unknown as (id: string) => string | null)(resolved);
            return { resolved, stub };
        };
        expect(run(ready, 'serve').resolved).toBe(ready.entry);
        const built = run(ready, 'build');
        expect(built.resolved).not.toBe(ready.entry);
        expect(built.stub).toContain('GI_BUNDLE_OPTIONAL_SIBLINGS=1');
        expect(run(ready, 'build', { GI_BUNDLE_OPTIONAL_SIBLINGS: '1' }).resolved).toBe(ready.entry);
        for (const command of ['serve', 'build'] as const) {
            const stub = run(missing, command).stub ?? '';
            expect(stub).toContain('throw new Error');
            expect(stub).not.toContain(root);
            expect(stub).not.toContain(JSON.stringify(root).slice(1, -1)); // as the stub's string literal writes it
        }
        expect(optionalSiblingState('@pathtracer/mlp', entry, repo, () => 'b'.repeat(40)).reason).not.toContain(root);
    });
});
