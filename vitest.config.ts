import { resolve } from 'path';
import { configDefaults, defineConfig } from 'vitest/config';

export default defineConfig({
    test: {
        environment: 'node',
        // A package with no tests yet (the M0 skeleton) is not a failure.
        passWithNoTests: true,
        // Agent worktrees under .claude/ are full checkouts; without this every
        // one of them runs its own copy of the suite.
        exclude: [...configDefaults.exclude, '.claude/**', '**/dist/**'],
    },
    resolve: {
        // The umbrella imports itself by name. Node's self-reference resolves that
        // through package.json `exports` to `dist/`, which doesn't exist before a
        // build, so point it at the source. Exact-match per subpath: a plain string
        // key is a prefix alias.
        alias: [
            { find: /^@pmndrs\/ddgi$/, replacement: resolve(__dirname, 'packages/ddgi/src/index.ts') },
            { find: /^@pmndrs\/ddgi\/(core|three)$/, replacement: resolve(__dirname, 'packages/ddgi/src/$1.ts') },
        ],
    },
});
