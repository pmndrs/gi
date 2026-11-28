/**
 * The three adapter's unit tests with the T3.0 matrix watch installed on every
 * object added to a parent (vitest-setup.ts):
 *
 *   pnpm vitest run --config tools/spikes/m3-t3.0/vitest.watch.config.ts
 */

import { resolve } from 'path';
import { defineConfig, mergeConfig } from 'vitest/config';
import base from '../../../vitest.config.ts';

const repo = resolve(__dirname, '../../..');

export default mergeConfig(
    base,
    defineConfig({
        root: repo,
        test: {
            include: ['packages/three/**/*.test.ts', 'packages/ddgi-three/**/*.test.ts', 'packages/gi/**/*.test.ts'],
            setupFiles: [resolve(__dirname, 'vitest-setup.ts')],
        },
    }),
);
