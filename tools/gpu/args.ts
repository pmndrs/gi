import { parseArgs } from 'node:util';
import { type GpuChoice } from './contract.ts';

export interface RunnerOptions {
    /** Which GPU to demand; null means "whatever Chrome picks" (still reported). */
    gpu: GpuChoice | null;
    /** Page-name substrings; a page runs if its name contains any. Empty runs everything. */
    filters: string[];
    chromePath: string | null;
    outDir: string | null;
    /** Write the whole run (adapter, per-page results and data) to this JSON file. */
    jsonPath: string | null;
    /** Print every console and log message, not only the ones of failing pages. */
    verbose: boolean;
    /** Launch Chrome with the GPU disabled, to see the "WebGPU unavailable" failure. */
    withoutWebGpu: boolean;
    /** Replaces every page's own timeout. */
    timeoutMs: number | null;
    help: boolean;
}

/** A mistake in the command line. The message is shown to the user as is. */
export class UsageError extends Error {}

export const USAGE = `Usage: pnpm test:gpu [options]

  --gpu nvidia|intel   Demand this GPU. The run fails if Chrome gives another one.
  --filter <name>      Only run pages whose name contains <name> (comma-separate several).
  --chrome <path>      Chrome or Edge executable (default: look for an installed one; or CHROME_PATH).
  --out <dir>          Where screenshots go (default: a temp folder, printed at the end).
  --json <file>        Also write the whole run (adapter, per-page status, durations, data) as JSON.
  --timeout <ms>       Replace every page's timeout.
  --no-webgpu          Start Chrome with the GPU disabled. The run must fail clearly (self-check).
  --verbose            Print all console output, not only for failing pages.
  --help               Show this text.`;

const GPU_CHOICES: readonly string[] = ['nvidia', 'intel'];

export function parseRunnerArgs(argv: string[]): RunnerOptions {
    let values;
    try {
        ({ values } = parseArgs({
            args: argv,
            options: {
                gpu: { type: 'string' },
                filter: { type: 'string', multiple: true },
                chrome: { type: 'string' },
                out: { type: 'string' },
                json: { type: 'string' },
                timeout: { type: 'string' },
                'no-webgpu': { type: 'boolean' },
                verbose: { type: 'boolean' },
                help: { type: 'boolean' },
            },
            strict: true,
            allowPositionals: false,
        }));
    } catch (error) {
        throw new UsageError(error instanceof Error ? error.message : String(error));
    }

    if (values.gpu !== undefined && !GPU_CHOICES.includes(values.gpu)) {
        throw new UsageError(`--gpu must be nvidia or intel, got "${values.gpu}".`);
    }

    let timeoutMs: number | null = null;
    if (values.timeout !== undefined) {
        timeoutMs = Number(values.timeout);
        if (!Number.isInteger(timeoutMs) || timeoutMs <= 0) {
            throw new UsageError(`--timeout must be a positive number of milliseconds, got "${values.timeout}".`);
        }
    }

    const filters = (values.filter ?? [])
        .flatMap((entry) => entry.split(','))
        .map((entry) => entry.trim())
        .filter((entry) => entry.length > 0);

    return {
        gpu: (values.gpu as GpuChoice | undefined) ?? null,
        filters,
        chromePath: values.chrome ?? null,
        outDir: values.out ?? null,
        jsonPath: values.json ?? null,
        verbose: values.verbose ?? false,
        withoutWebGpu: values['no-webgpu'] ?? false,
        timeoutMs,
        help: values.help ?? false,
    };
}
