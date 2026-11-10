/**
 * The docs cover what ships (docs/design/m5-coverage.md §4): every public
 * export of `@pmndrs/ddgi/core` is named in docs/api/platform.md, and every
 * export of `@pmndrs/ddgi/three` in docs/api/three.md (docs-cover-exports.ts).
 * A new export fails here until its doc names it.
 */

import { describe, expect, it } from 'vitest';
import { checkDocsCoverExports, markdownCode, namesMissingFrom } from './docs-cover-exports.ts';
import { PUBLIC_ENTRY_POINTS, publicExports } from './public-exports.ts';

describe('the API docs name every public export', () => {
    for (const report of checkDocsCoverExports()) {
        it(`${report.entry.doc} names every export of ${report.entry.name}`, () => {
            expect(report.exported).toBeGreaterThan(10);
            expect(report.missing, `add these to ${report.entry.doc} (the "Every export" tables at its end)`).toEqual([]);
        });
    }

    it('lists the entry points through their re-exports', () => {
        const core = publicExports(PUBLIC_ENTRY_POINTS[0]);
        // From `export *` of the contract (types), the platform object, DDGI and the presets.
        for (const name of ['MaterialDescriptor', 'createGI', 'DDGITechnique', 'ddgiPreset', 'GpuTimer']) expect(core).toContain(name);
        const three = publicExports(PUBLIC_ENTRY_POINTS[1]);
        for (const name of ['GIVolume', 'ThreeGI', 'toLightProbeGrid', 'PmndrsSkyLike']) expect(three).toContain(name);
    });

    it('counts a name only inside code, as a whole word', () => {
        const markdown = ['The frame is drawn.', '', 'Call `createGI(device)` first.', '', '```ts', 'const budget: FrameBudget = {};', '```'].join('\n');
        const code = markdownCode(markdown);
        expect(namesMissingFrom(code, ['createGI', 'FrameBudget', 'frame', 'Frame', 'create'])).toEqual(['frame', 'Frame', 'create']);
    });
});
