/**
 * `pnpm test:gpu [--gpu nvidia|intel] [--filter name]`
 *
 * Starts the vite test site, launches installed Chrome (headless, pinned to the
 * requested GPU), and runs every page under pages/ in it. The exit code is 0
 * only if everything passed and every self-test was caught. See README.md.
 */

import { existsSync } from 'node:fs';
import { writeFile } from 'node:fs/promises';
import { gitCommit } from './measured-commit.ts';
import { tmpdir } from 'node:os';
import { join, resolve } from 'node:path';
import { parseRunnerArgs, UsageError, USAGE, type RunnerOptions } from './args.ts';
import { EXIT_CODE_SIGINT as EXIT_CODE_INTERRUPTED, installShutdownHandlers, registerCleanup, shutdownSignal } from './cleanup.ts';
import { type AdapterSummary, type RunSummary } from './contract.ts';
import { matchesExpectation } from './evaluate.ts';
import { discoverPages, selectPages, withOptionalSiblings } from './pages.ts';
import { optionalSiblings } from '../optional-siblings.ts';
import { storageBufferCaps } from './device-limit-caps.ts';
import { BrowserSession, WebGpuUnavailableError, WrongAdapterError, type PageResult } from './session.ts';
import { SITE_ROOT, startTestSite, type TestSiteServer } from './server.ts';
import { summarizePage } from './summary.ts';

function describeAdapter(adapter: AdapterSummary): string {
    const parts = [`vendor=${adapter.vendor}`, `architecture=${adapter.architecture}`];
    if (adapter.device) parts.push(`device=${adapter.device}`);
    if (adapter.description) parts.push(`description=${adapter.description}`);
    return parts.join(' ');
}

function indent(text: string): string {
    return text
        .split('\n')
        .filter((line) => line.trim() !== '')
        .map((line) => `      ${line}`)
        .join('\n');
}

const OUTCOME_LABELS = {
    pass: 'PASS',
    fail: 'FAIL',
    'caught-expected-failure': 'PASS (failure expected and caught)',
    'missed-expected-failure': 'FAIL (expected failure was not caught)',
    'wrong-failure': 'FAIL (failed, but not in the expected way)',
};

function printPageResult(result: PageResult, verbose: boolean): void {
    const { page, verdict } = result;
    const seconds = (result.durationMs / 1000).toFixed(2);
    console.log(
        `  ${OUTCOME_LABELS[verdict.outcome]}  ${page.name}  ${seconds} s${page.description ? `  - ${page.description}` : ''}`,
    );

    if (result.report && Object.keys(result.report.data).length > 0) {
        console.log(`      data: ${JSON.stringify(result.report.data)}`);
    }
    if (verdict.outcome === 'caught-expected-failure') {
        // A self-test succeeded by failing: show that the expected error text was seen, not a wall of it.
        const expected = page.expectMessage;
        const shown = (expected && verdict.reasons.find((reason) => matchesExpectation(reason, expected))) || verdict.reasons[0];
        console.log(indent(`caught ${verdict.reasons.length} problem(s); the matching one:`));
        console.log(indent((shown ?? '').split('\n').slice(0, 2).join('\n')));
    } else if (verdict.outcome !== 'pass') {
        for (const reason of verdict.reasons) console.log(indent(reason));
    }
    if (verbose) {
        for (const message of result.messages) {
            const where = message.where ? `  (${message.where})` : '';
            console.log(indent(`[${message.origin}/${message.level}] ${message.text}${where}`));
        }
    }
    if (result.screenshotPath) console.log(`      screenshot: ${result.screenshotPath}`);
}

type SelfTestSummary = RunSummary['selfTests'][number];

/** The runner's own check: with the GPU disabled, the run must fail with a clear message. */
async function selfTestWebGpuDisabled(options: RunnerOptions, server: TestSiteServer): Promise<SelfTestSummary> {
    const name = 'webgpu-disabled';
    const startTime = Date.now();
    const finish = (ok: boolean, message: string): SelfTestSummary => ({
        name,
        ok,
        durationMs: Date.now() - startTime,
        message,
    });
    let session: BrowserSession | null = null;
    try {
        session = await BrowserSession.launch({
            serverOrigin: server.origin,
            chromePath: options.chromePath,
            gpu: null,
            withoutGpu: true,
        });
        await session.preflight();
        const message = 'Chrome without a GPU still reported a WebGPU adapter';
        console.log(`  FAIL  runner self-test: ${message}`);
        return finish(false, message);
    } catch (error) {
        // Ctrl-C during the self-test is an interruption, not a failed check: let main() handle it.
        if (shutdownSignal.aborted) throw error;
        const seconds = ((Date.now() - startTime) / 1000).toFixed(2);
        if (error instanceof WebGpuUnavailableError) {
            console.log(`  PASS (failure expected and caught)  runner self-test: WebGPU disabled  ${seconds} s`);
            console.log(`      ${error.message}`);
            return finish(true, error.message);
        }
        const message = `WebGPU disabled gave an unclear error: ${error}`;
        console.log(`  FAIL  runner self-test: ${message}`);
        return finish(false, message);
    } finally {
        await session?.close();
    }
}

async function main(): Promise<number> {
    let options: RunnerOptions;
    try {
        options = parseRunnerArgs(process.argv.slice(2));
    } catch (error) {
        if (error instanceof UsageError) {
            console.error(`${error.message}\n\n${USAGE}`);
            return 2;
        }
        throw error;
    }
    if (options.help) {
        console.log(USAGE);
        return 0;
    }

    const runStart = Date.now();
    const allPages = await discoverPages(join(SITE_ROOT, 'pages'));
    // Pages that need an optional sibling this checkout doesn't have at its pin are left out, with a note.
    const readySiblings = new Set(optionalSiblings().filter((sibling) => sibling.state === 'ready').map((sibling) => sibling.name));
    const siblingCheck = withOptionalSiblings(selectPages(allPages, options.filters), readySiblings);
    for (const { page, missing } of siblingCheck.left) console.log(`left out ${page.name}: it needs ${missing.join(', ')} (an optional sibling, not at its pin here; node tools/linked-siblings.ts)`);
    const pages = siblingCheck.kept;
    if (pages.length === 0) {
        console.error(
            `No test page matches --filter ${options.filters.join(',')}. Pages: ${allPages.map((page) => page.name).join(', ')}`,
        );
        return 2;
    }

    // Created on first use by the first screenshot, so a run that fails at startup leaves no empty folder.
    const screenshotDir = options.outDir ?? join(tmpdir(), `ddgi-gpu-shots-${Date.now()}`);
    const gpuLabel = options.gpu ?? 'default';
    const summary: RunSummary = {
        schema: 1,
        startedAt: new Date(runStart).toISOString(),
        commit: recordedCommit(),
        gpuRequested: options.gpu,
        adapter: null,
        ok: false,
        totalMs: 0,
        error: null,
        pages: [],
        selfTests: [],
    };
    let failures = 0;
    let session: BrowserSession | null = null;
    let server: TestSiteServer | null = null;
    let closeServer: () => Promise<void> = async () => {};

    try {
        server = await startTestSite();
        closeServer = registerCleanup(server.close);
        console.log(`test site: ${server.origin}   gpu requested: ${options.gpu ?? 'any'}`);

        // BrowserSession.launch registers its own cleanup as soon as Chrome exists.
        session = await BrowserSession.launch({
            serverOrigin: server.origin,
            chromePath: options.chromePath,
            gpu: options.gpu,
            withoutGpu: options.withoutWebGpu,
            limitCaps: options.maxStorageBuffers === null ? null : storageBufferCaps(options.maxStorageBuffers),
        });

        summary.adapter = await session.preflight();
        console.log(`adapter: ${describeAdapter(summary.adapter)}`);
        if (options.maxStorageBuffers !== null) {
            console.log(`device limits capped: at most ${options.maxStorageBuffers} storage buffers per shader stage (checked on a requested device)`);
        }
        console.log(`pages: ${pages.map((page) => page.name).join(', ')}`);

        for (const page of pages) {
            // A self-test page counts as "expected to fail" only in the full run. Asked for by name,
            // it is judged like any other page, so `--filter wgsl-error` fails and shows the error.
            const result = await session.runPage(page, {
                timeoutMs: options.timeoutMs ?? page.timeoutMs,
                expectFailure: page.expectFailure && options.filters.length === 0,
                screenshotDir,
                gpuLabel,
            });
            printPageResult(result, options.verbose);
            summary.pages.push(summarizePage(result));
            if (!result.verdict.acceptable) failures++;
        }

        // The runner's own self-test belongs to the full run only; a --filter run is about one page.
        if (options.filters.length === 0 && !options.withoutWebGpu) {
            await session.close();
            session = null;
            const selfTest = await selfTestWebGpuDisabled(options, server);
            summary.selfTests.push(selfTest);
            if (!selfTest.ok) failures++;
        }
    } catch (error) {
        // Ctrl-C closed Chrome under our feet; the shutdown handler is already exiting the process.
        if (shutdownSignal.aborted) return EXIT_CODE_INTERRUPTED;
        summary.error = error instanceof Error ? error.message : String(error);
        if (error instanceof WebGpuUnavailableError || error instanceof WrongAdapterError) {
            console.error(`\nFAILED: ${error.message}`);
        } else {
            console.error(`\nFAILED: ${error instanceof Error ? (error.stack ?? error.message) : error}`);
        }
        failures++;
    } finally {
        await session?.close();
        await closeServer();
    }

    summary.ok = failures === 0;
    summary.totalMs = Date.now() - runStart;
    if (options.jsonPath) {
        const jsonPath = resolve(options.jsonPath);
        await writeFile(jsonPath, `${JSON.stringify(summary, null, 2)}\n`);
        console.log(`json: ${jsonPath}`);
    }

    const screenshotNote = existsSync(screenshotDir) ? `  screenshots: ${screenshotDir}` : '';
    console.log(
        `\n${summary.ok ? 'OK' : 'FAILED'}  gpu=${gpuLabel}  total ${(summary.totalMs / 1000).toFixed(1)} s${screenshotNote}`,
    );
    return summary.ok ? 0 : 1;
}

installShutdownHandlers();
process.exitCode = await main();

/** The commit being measured, for `RunSummary.commit`; undefined when git can't tell (no repository, no git). */
function recordedCommit(): string | undefined {
    try {
        return gitCommit();
    } catch {
        return undefined;
    }
}
