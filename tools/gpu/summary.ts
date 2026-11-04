import { type PageSummary } from './contract.ts';
import { type PageResult } from './session.ts';

/** One page's entry in the `--json` file. */
export function summarizePage(result: PageResult): PageSummary {
    return {
        name: result.page.name,
        outcome: result.verdict.outcome,
        acceptable: result.verdict.acceptable,
        durationMs: result.durationMs,
        reasons: result.verdict.reasons,
        data: result.report?.data ?? {},
        adapter: result.report?.adapter ?? null,
        screenshot: result.screenshotPath,
    };
}
