/**
 * The contract between a test page (browser) and the runner (Node).
 *
 * This file is types and constants only, so both sides can import it. The
 * human-readable version of the contract is in README.md.
 */

/** Which of this laptop's GPUs a run is asked to use. */
export type GpuChoice = 'nvidia' | 'intel';

/** What `GPUAdapter.info` says, copied into plain fields (it does not JSON.stringify). */
export interface AdapterSummary {
    vendor: string;
    architecture: string;
    device: string;
    description: string;
    isFallbackAdapter: boolean;
    features: string[];
}

/** The object a page leaves on `window.__gpuTestReport` when it is done. */
export interface GpuTestReport {
    /** Echo of the `token` query parameter, so the runner ignores a stale page's report. */
    token: string | null;
    status: 'pass' | 'fail';
    /** Why the test failed. Absent on pass. */
    message?: string;
    /** Free-form numbers and facts the test wants shown (and later, recorded). */
    data: Record<string, unknown>;
    adapter: AdapterSummary | null;
    /** Text of every `uncapturederror` the device raised. Any entry fails the test. */
    deviceErrors: string[];
    durationMs: number;
}

/** Optional `test.json` next to a page's `index.html`. */
export interface PageManifest {
    /** One line saying what the page checks. Shown in the run output. */
    description?: string;
    /**
     * The page is a self-test of the runner: it is supposed to fail. The run
     * passes only if the runner caught the failure; a clean pass is an error.
     */
    expectFailure?: boolean;
    /**
     * With expectFailure: the failure must be this one. A substring, or a regular
     * expression written as /pattern/flags. Without it, a 404 or a timeout would
     * count as "caught" too.
     */
    expectMessage?: string;
    /** Give up after this long. Default is {@link DEFAULT_PAGE_TIMEOUT_MS}. */
    timeoutMs?: number;
    /**
     * A slow benchmark that must run alone, under the bench lock. It runs only
     * when a --filter names it exactly, never in a full run or a partial match.
     */
    manual?: boolean;
}

/** What `--json <file>` writes. Benchmark tasks read `pages[].data` from it to make CSV rows. */
export interface RunSummary {
    schema: 1;
    startedAt: string;
    /**
     * The code that ran: HEAD's short hash, plus "-dirty" when a measured file
     * differed from it (tools/gpu/measured-commit.ts), read when the run started.
     * Missing in runs from before T1.3, or when git wasn't available.
     */
    commit?: string;
    gpuRequested: GpuChoice | null;
    /** What the preflight check saw; null if the run failed before that. */
    adapter: AdapterSummary | null;
    ok: boolean;
    totalMs: number;
    /** Set when the run itself broke (no WebGPU, wrong GPU, Chrome would not start). */
    error: string | null;
    pages: PageSummary[];
    /** The runner's own checks, which only the full (unfiltered) run performs. */
    selfTests: Array<{ name: string; ok: boolean; durationMs: number; message: string }>;
}

export interface PageSummary {
    name: string;
    /** pass | fail | caught-expected-failure | missed-expected-failure | wrong-failure */
    outcome: string;
    /** True when this page did not make the run fail. */
    acceptable: boolean;
    durationMs: number;
    /** Why it failed (or what was caught). Empty for a pass. */
    reasons: string[];
    /** The `data` the page returned. */
    data: Record<string, unknown>;
    adapter: AdapterSummary | null;
    screenshot: string | null;
}

export const DEFAULT_PAGE_TIMEOUT_MS = 20_000;

/** Name of the window property the page writes its {@link GpuTestReport} to. */
export const REPORT_PROPERTY = '__gpuTestReport';

/** Query parameters the runner adds to a page URL. */
export const QUERY_TOKEN = 'token';
export const QUERY_GPU = 'gpu';

/**
 * The built examples site (`PAGES_BASE=/gi/ pnpm examples:build`,
 * into examples/dist) as the test server serves it: under this subdirectory,
 * as GitHub Pages serves a project site, so a link or asset that ignores the
 * base fails to load. The `examples-site` page checks it.
 */
export const EXAMPLES_SITE_BASE = '/gi/';
/** The test server's list of the built site's files, `[{ path, bytes }]` (paths under the base). Not part of the site. */
export const EXAMPLES_SITE_FILES = '/__examples-site-files';
