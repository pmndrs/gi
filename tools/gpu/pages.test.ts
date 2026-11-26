import { describe, expect, it } from 'vitest';
import { discoverPages, selectPages, withOptionalSiblings, type TestPage } from './pages.ts';
import { join } from 'node:path';

function page(name: string, manual = false): TestPage {
    return {
        name,
        urlPath: `/pages/${name}/index.html`,
        description: '',
        expectFailure: false,
        expectMessage: null,
        timeoutMs: 1000,
        manual,
    };
}

describe('withOptionalSiblings', () => {
    it('leaves out a page whose optional sibling is not at its pin, and says which', () => {
        const tail = { ...page('radiance-cache-neural-tail'), optionalSiblings: ['@pathtracer/mlp'] };
        const pages = [page('smoke'), tail];
        expect(withOptionalSiblings(pages, new Set(['@pathtracer/mlp'])).kept).toEqual(pages);
        const without = withOptionalSiblings(pages, new Set());
        expect(without.kept).toEqual([page('smoke')]);
        expect(without.left).toEqual([{ page: tail, missing: ['@pathtracer/mlp'] }]);
    });
});

describe('selectPages', () => {
    const pages = [page('smoke'), page('wgsl-error'), page('bvh-oracle')];

    it('keeps everything when there is no filter', () => {
        expect(selectPages(pages, [])).toEqual(pages);
    });

    it('matches by substring and keeps the original order', () => {
        expect(selectPages(pages, ['bvh', 'smoke']).map((p) => p.name)).toEqual(['smoke', 'bvh-oracle']);
    });

    it('returns nothing for an unknown name', () => {
        expect(selectPages(pages, ['nope'])).toEqual([]);
    });

    it('runs a manual page only when a filter is its exact name', () => {
        const withBench = [...pages, page('trace-bench-quick'), page('trace-bench-sponza', true)];
        expect(selectPages(withBench, []).map((p) => p.name)).not.toContain('trace-bench-sponza');
        expect(selectPages(withBench, ['trace-bench']).map((p) => p.name)).toEqual(['trace-bench-quick']);
        expect(selectPages(withBench, ['trace-bench-sponza']).map((p) => p.name)).toEqual(['trace-bench-sponza']);
    });
});

describe('discoverPages', () => {
    it('finds the shipped pages and reads their test.json', async () => {
        const pages = await discoverPages(join(import.meta.dirname, 'pages'));
        const byName = new Map(pages.map((p) => [p.name, p]));
        expect(byName.get('smoke')?.expectFailure).toBe(false);
        expect(byName.get('wgsl-error')?.expectFailure).toBe(true);
        expect(byName.get('wgsl-error')?.expectMessage).toBe('Error while parsing WGSL');
        expect(byName.get('smoke')?.urlPath).toBe('/pages/smoke/index.html');
    });
});
