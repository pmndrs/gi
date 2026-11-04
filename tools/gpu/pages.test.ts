import { describe, expect, it } from 'vitest';
import { discoverPages, selectPages, type TestPage } from './pages.ts';
import { join } from 'node:path';

function page(name: string): TestPage {
    return {
        name,
        urlPath: `/pages/${name}/index.html`,
        description: '',
        expectFailure: false,
        expectMessage: null,
        timeoutMs: 1000,
    };
}

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
