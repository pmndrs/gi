// Bash for tests that run workflow steps: plain `bash` on Linux and macOS, Git
// Bash on Windows (WSL's bash can't see Windows paths the same way). From
// @pmndrs/upscaler's scripts/test-shell.mjs.
import { spawnSync } from 'node:child_process';
import { existsSync } from 'node:fs';
import { delimiter, dirname, join } from 'node:path';

/** A Windows path as Git Bash writes it: C:\a\b → /c/a/b. Unchanged elsewhere. */
export function shellPath(path: string): string {
    if (process.platform !== 'win32') return path;
    return path.replaceAll('\\', '/').replace(/^([A-Za-z]):\//, (_, drive: string) => `/${drive.toLowerCase()}/`);
}

/** Single-quotes a value for bash. */
export function shellQuote(value: string): string {
    return `'${value.replaceAll("'", `'"'"'`)}'`;
}

function findBash(): string | undefined {
    if (process.platform !== 'win32') return 'bash';
    const candidates = [
        process.env.GIT_BASH_PATH,
        join(process.env.ProgramFiles ?? 'C:/Program Files', 'Git/bin/bash.exe'),
        join(process.env['ProgramFiles(x86)'] ?? 'C:/Program Files (x86)', 'Git/bin/bash.exe'),
        join(process.env.LOCALAPPDATA ?? '', 'Programs/Git/bin/bash.exe'),
    ];
    return candidates.find((path) => path !== undefined && existsSync(path));
}

/** The bash to run steps with, or undefined when Windows has no Git Bash (the tests then skip). */
export const bash = findBash();

/** The environment for a step: the test's own, plus `bins` first on PATH (and Git Bash's tools on Windows). */
export function shellEnvironment(extra: Record<string, string> = {}, bins: string[] = []): Record<string, string> {
    const inherited = Object.fromEntries(
        Object.entries(process.env).filter((entry): entry is [string, string] => entry[0].toUpperCase() !== 'PATH' && entry[1] !== undefined),
    );
    const path = process.env.PATH ?? process.env.Path ?? '';
    const tools = process.platform === 'win32' && bash ? [join(dirname(bash), '../usr/bin')] : [];
    return { ...inherited, ...extra, PATH: [...bins, ...tools, ...path.split(delimiter)].map(shellPath).join(':') };
}

/** Where bash finds a command. */
export function which(command: string): string {
    if (!bash) throw new Error('No bash');
    const result = spawnSync(bash, ['-c', 'command -v "$1"', 'resolve', command], { env: shellEnvironment(), encoding: 'utf8' });
    if (result.status !== 0) throw new Error(`Could not find ${command} in bash: ${result.stderr || result.error}`);
    return result.stdout.trim();
}
