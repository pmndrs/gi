/**
 * The package boundaries the lint config enforces, checked through ESLint
 * itself on small snippets at chosen paths: the GI-1 denoiser
 * (`packages/denoiser-gi1`, RC4.3) imports nothing outside its own folder,
 * and the radiance cache reaches it only by its package name.
 */

import { fileURLToPath } from 'node:url';
import { ESLint } from 'eslint';
import { describe, expect, it } from 'vitest';

const ROOT = fileURLToPath(new URL('../../', import.meta.url));
const eslint = new ESLint({ cwd: ROOT });

async function ruleIds(file: string, code: string): Promise<string[]> {
    const [result] = await eslint.lintText(code, { filePath: `${ROOT}${file}` });
    return result.messages.map((message) => message.ruleId ?? message.message);
}

describe('packages/denoiser-gi1 is self-contained', () => {
    const file = 'packages/denoiser-gi1/src/example.ts';

    it('may import its own files', async () => {
        expect(await ruleIds(file, "export { MAX_BLUR_MASK } from './constants.ts';\n")).toEqual([]);
    });

    it('may not import ddgi-core, by name or by path, or any other package', async () => {
        expect(await ruleIds(file, "export { GILimitError } from '@pmndrs/ddgi-core/platform/contract/errors';\n")).toContain('local/self-contained');
        expect(await ruleIds(file, "export { GILimitError } from '../../ddgi-core/src/platform/contract/errors.ts';\n")).toContain('local/self-contained');
        expect(await ruleIds(file, "export { Vector3 } from 'three';\n")).toContain('local/self-contained');
        expect(await ruleIds(file, "export { describe } from 'vitest';\n")).toContain('local/self-contained');
    });

    it('lets its tests use vitest', async () => {
        expect(await ruleIds('packages/denoiser-gi1/src/example.test.ts', "export { describe } from 'vitest';\n")).toEqual([]);
    });
});

describe('the radiance cache uses the denoiser as a package', () => {
    const file = 'packages/ddgi-core/src/techniques/radiance-cache/example.ts';

    it('by its name', async () => {
        expect(await ruleIds(file, "export { GI1Denoiser } from '@pmndrs/denoiser-gi1';\n")).toEqual([]);
    });

    it('not by a path into the package', async () => {
        expect(await ruleIds(file, "export { GI1Denoiser } from '../../../../denoiser-gi1/src/index.ts';\n")).toContain('local/core-layers');
    });
});
