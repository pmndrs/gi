import { describe, expect, it } from 'vitest';
import { type AdapterSummary, type GpuTestReport } from './contract.ts';
import {
    adapterMismatch,
    evaluatePage,
    isFailureMessage,
    matchesExpectation,
    parseExpectedMessage,
    type BrowserMessage,
    type PageEvidence,
} from './evaluate.ts';

function adapter(vendor: string, overrides: Partial<AdapterSummary> = {}): AdapterSummary {
    return {
        vendor,
        architecture: 'any',
        device: '',
        description: '',
        isFallbackAdapter: false,
        features: [],
        ...overrides,
    };
}

function report(overrides: Partial<GpuTestReport> = {}): GpuTestReport {
    return { token: 't', status: 'pass', data: {}, adapter: null, deviceErrors: [], durationMs: 1, ...overrides };
}

function evidence(overrides: Partial<PageEvidence> = {}): PageEvidence {
    return {
        report: report(),
        runnerProblem: null,
        messages: [],
        expectFailure: false,
        expectMessage: null,
        wantedGpu: null,
        ...overrides,
    };
}

const WGSL_WARNING: BrowserMessage = {
    origin: 'log',
    level: 'warning',
    text: 'Error while parsing WGSL: :6:9 error: cannot initialize',
};

describe('adapterMismatch', () => {
    it('accepts the requested vendor, ignoring case', () => {
        expect(adapterMismatch('nvidia', adapter('NVIDIA'))).toBeNull();
        expect(adapterMismatch('intel', adapter('intel'))).toBeNull();
    });

    it('names both GPUs when they differ', () => {
        const message = adapterMismatch('intel', adapter('nvidia', { architecture: 'ampere' }));
        expect(message).toContain('--gpu intel');
        expect(message).toContain('nvidia/ampere');
    });

    it('accepts any real GPU when none was requested', () => {
        expect(adapterMismatch(null, adapter('apple'))).toBeNull();
    });

    it('rejects a software fallback adapter even when none was requested', () => {
        expect(adapterMismatch(null, adapter('google', { isFallbackAdapter: true }))).toContain('software');
    });
});

describe('isFailureMessage', () => {
    it('counts errors, exceptions and WebGPU warnings', () => {
        expect(isFailureMessage({ origin: 'console', level: 'error', text: 'boom' })).toBe(true);
        expect(isFailureMessage({ origin: 'exception', level: 'log', text: 'TypeError' })).toBe(true);
        expect(isFailureMessage(WGSL_WARNING)).toBe(true);
    });

    it('ignores ordinary logging and vite chatter', () => {
        expect(isFailureMessage({ origin: 'console', level: 'log', text: 'hello' })).toBe(false);
        expect(isFailureMessage({ origin: 'console', level: 'debug', text: '[vite] connected.' })).toBe(false);
        expect(isFailureMessage({ origin: 'log', level: 'warning', text: 'cookie warning' })).toBe(false);
    });
});

describe('evaluatePage', () => {
    it('passes a clean page', () => {
        const verdict = evaluatePage(evidence());
        expect(verdict.outcome).toBe('pass');
        expect(verdict.acceptable).toBe(true);
    });

    it('fails when the page reports failure, and says why', () => {
        const verdict = evaluatePage(evidence({ report: report({ status: 'fail', message: 'values wrong' }) }));
        expect(verdict.outcome).toBe('fail');
        expect(verdict.reasons.join('\n')).toContain('values wrong');
    });

    it('fails a page that reported pass but left a WGSL error in the browser log', () => {
        const verdict = evaluatePage(evidence({ messages: [WGSL_WARNING] }));
        expect(verdict.outcome).toBe('fail');
        expect(verdict.reasons.join('\n')).toContain('Error while parsing WGSL');
    });

    it('fails a page that reported pass but had a device error', () => {
        const verdict = evaluatePage(evidence({ report: report({ deviceErrors: ['out of memory'] }) }));
        expect(verdict.outcome).toBe('fail');
    });

    it('fails when there is no report (timeout or crash)', () => {
        const verdict = evaluatePage(evidence({ report: null, runnerProblem: 'Timed out after 20000 ms' }));
        expect(verdict.outcome).toBe('fail');
        expect(verdict.reasons[0]).toContain('Timed out');
    });

    it('does not say the same error three times', () => {
        const text = 'Error while parsing WGSL: :6:9 error: cannot initialize';
        const verdict = evaluatePage(
            evidence({
                report: report({ status: 'fail', message: text, deviceErrors: [text] }),
                messages: [WGSL_WARNING],
            }),
        );
        expect(verdict.reasons).toHaveLength(1);
    });

    it('fails when the page ran on a different adapter than requested', () => {
        const verdict = evaluatePage(
            evidence({ wantedGpu: 'intel', report: report({ adapter: adapter('nvidia', { architecture: 'ampere' }) }) }),
        );
        expect(verdict.outcome).toBe('fail');
        expect(verdict.reasons.join('\n')).toContain("page's own adapter");
    });

    describe('for a self-test page that is expected to fail', () => {
        it('accepts a failure with the expected message', () => {
            const verdict = evaluatePage(
                evidence({ expectFailure: true, expectMessage: 'Error while parsing WGSL', messages: [WGSL_WARNING] }),
            );
            expect(verdict.outcome).toBe('caught-expected-failure');
        });

        it('rejects a different failure, such as a timeout', () => {
            const verdict = evaluatePage(
                evidence({
                    expectFailure: true,
                    expectMessage: 'Error while parsing WGSL',
                    report: null,
                    runnerProblem: 'Timed out after 20000 ms',
                }),
            );
            expect(verdict.outcome).toBe('wrong-failure');
            expect(verdict.acceptable).toBe(false);
            expect(verdict.reasons.join('\n')).toContain('Timed out');
        });

        it('matches a regular expression written as /pattern/flags', () => {
            expect(matchesExpectation('Error while parsing WGSL', parseExpectedMessage('/parsing wgsl/i'))).toBe(true);
            expect(matchesExpectation('Error while parsing WGSL', parseExpectedMessage('parsing wgsl'))).toBe(false);
        });

        it('accepts a caught failure', () => {
            const verdict = evaluatePage(evidence({ expectFailure: true, messages: [WGSL_WARNING] }));
            expect(verdict.outcome).toBe('caught-expected-failure');
            expect(verdict.acceptable).toBe(true);
        });

        it('rejects a clean pass, because the runner missed something', () => {
            const verdict = evaluatePage(evidence({ expectFailure: true }));
            expect(verdict.outcome).toBe('missed-expected-failure');
            expect(verdict.acceptable).toBe(false);
        });
    });
});
