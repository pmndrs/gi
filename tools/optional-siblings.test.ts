import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { afterAll, describe, expect, it } from 'vitest';
import { optionalSiblingState, type SiblingEntry } from './optional-siblings.ts';

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
});
