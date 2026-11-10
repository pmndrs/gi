import { configDefaults, defineConfig } from 'vitest/config';

// `pnpm test:release`: the publish workflow's test alone (the root config
// leaves it out of `pnpm test`, see there). CI runs it as its own step.
export default defineConfig({
    test: {
        root: import.meta.dirname,
        environment: 'node',
        include: ['publish-workflow.test.ts'],
        exclude: configDefaults.exclude,
    },
});
