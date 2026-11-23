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
    /** Runs only when a filter names it exactly. See {@link PageManifest.manual}. */
    manual: boolean;
    /** Optional siblings it needs (none when absent). See {@link PageManifest.optionalSiblings}. */
    optionalSiblings?: string[];
}

/**
 * Keeps pages whose name contains any filter text. No filters keeps everything.
 * Manual pages (slow benchmarks) are kept only when a filter is their exact name,
 * so a plain run or `--filter trace-bench` never starts a 10-minute benchmark by accident.
 */
export function selectPages(pages: TestPage[], filters: string[]): TestPage[] {
    return pages.filter((page) => {
        if (page.manual) return filters.includes(page.name);
        return filters.length === 0 || filters.some((filter) => page.name.includes(filter));
    });
}

/**
 * The pages whose optional siblings (`optionalSiblings`) are all in `ready`
 * (the names cloned at their pin); the others are left out, each with the
 * sibling it misses.
 */
export function withOptionalSiblings(pages: TestPage[], ready: ReadonlySet<string>): { kept: TestPage[]; left: { page: TestPage; missing: string[] }[] } {
    const kept: TestPage[] = [];
    const left: { page: TestPage; missing: string[] }[] = [];
    for (const page of pages) {
        const missing = (page.optionalSiblings ?? []).filter((name) => !ready.has(name));
        if (missing.length === 0) kept.push(page);
        else left.push({ page, missing });
    }
    return { kept, left };
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
            manual: manifest.manual ?? false,
            optionalSiblings: manifest.optionalSiblings ?? [],
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
