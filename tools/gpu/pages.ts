import { existsSync } from 'node:fs';
import { readdir, readFile } from 'node:fs/promises';
import { join } from 'node:path';
import { DEFAULT_PAGE_TIMEOUT_MS, type PageManifest } from './contract.ts';
import { parseExpectedMessage } from './evaluate.ts';

export interface TestPage {
    /** The folder name under pages/. This is what --filter matches. */
    name: string;
    /** Path on the test server, for example /pages/smoke/index.html */
    urlPath: string;
    description: string;
    expectFailure: boolean;
    /** The failure a self-test page must produce, or null for "any". */
    expectMessage: string | RegExp | null;
    timeoutMs: number;
}

/** Keeps pages whose name contains any filter text. No filters keeps everything. */
export function selectPages(pages: TestPage[], filters: string[]): TestPage[] {
    if (filters.length === 0) return pages;
    return pages.filter((page) => filters.some((filter) => page.name.includes(filter)));
}

/** Every folder under pagesDir that has an index.html, in name order. */
export async function discoverPages(pagesDir: string): Promise<TestPage[]> {
    const entries = await readdir(pagesDir, { withFileTypes: true });
    const pages: TestPage[] = [];
    for (const entry of entries.filter((candidate) => candidate.isDirectory()).sort(byName)) {
        if (!existsSync(join(pagesDir, entry.name, 'index.html'))) continue;
        const manifest = await readManifest(join(pagesDir, entry.name, 'test.json'));
        pages.push({
            name: entry.name,
            urlPath: `/pages/${entry.name}/index.html`,
            description: manifest.description ?? '',
            expectFailure: manifest.expectFailure ?? false,
            expectMessage: manifest.expectMessage === undefined ? null : parseExpectedMessage(manifest.expectMessage),
            timeoutMs: manifest.timeoutMs ?? DEFAULT_PAGE_TIMEOUT_MS,
        });
    }
    return pages;
}

function byName(a: { name: string }, b: { name: string }): number {
    return a.name.localeCompare(b.name);
}

async function readManifest(path: string): Promise<PageManifest> {
    if (!existsSync(path)) return {};
    try {
        return JSON.parse(await readFile(path, 'utf8')) as PageManifest;
    } catch (error) {
        throw new Error(`Cannot read ${path}: ${error instanceof Error ? error.message : error}`);
    }
}
