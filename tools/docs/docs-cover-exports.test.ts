/**
 * The docs cover what ships (docs/design/m5-coverage.md §4): every public
 * export of `@pmndrs/gi/core` and `@pmndrs/gi/ddgi` is named in
 * docs/api/platform.md, every export of `@pmndrs/gi/three` in
 * docs/api/three.md and every export of `@pmndrs/gi/radiance-cache` in
 * docs/api/radiance-cache.md, each in its subpath's "Every export of" section
 * (docs-cover-exports.ts). A new export fails here until its doc names it.
 */

import { describe, expect, it } from 'vitest';
import { checkDocsCoverExports, exportSection, markdownCode, namesMissingFrom } from './docs-cover-exports.ts';
import { PUBLIC_ENTRY_POINTS, publicExports } from './public-exports.ts';

describe('the API docs name every public export', () => {
    for (const report of checkDocsCoverExports()) {
        it(`${report.entry.doc} names every export of ${report.entry.name}`, () => {
            expect(report.exported).toBeGreaterThan(10);
            expect(report.missing, `add these to ${report.entry.doc} (the "Every export of \`${report.entry.name}\`" tables)`).toEqual([]);
        });
    }

    it('lists the entry points through their re-exports', () => {
        const [coreEntry, ddgiEntry, threeEntry] = PUBLIC_ENTRY_POINTS;
        const core = publicExports(coreEntry);
        // From `export *` of the contract (types), the platform object and the timer.
        for (const name of ['MaterialDescriptor', 'createGI', 'GITechnique', 'GpuTimer']) expect(core).toContain(name);
        const ddgi = publicExports(ddgiEntry);
        for (const name of ['DDGITechnique', 'ddgiPreset', 'DDGIVolumeSettings', 'exportIrradianceSH']) expect(ddgi).toContain(name);
        const three = publicExports(threeEntry);
        for (const name of ['DDGIVolume', 'ThreeGI', 'toLightProbeGrid', 'PmndrsSkyLike']) expect(three).toContain(name);
    });

    it('/core is the platform alone: no name is in both /core and /ddgi', () => {
        const ddgi = new Set(publicExports(PUBLIC_ENTRY_POINTS[1]));
        expect(publicExports(PUBLIC_ENTRY_POINTS[0]).filter((name) => ddgi.has(name))).toEqual([]);
    });

    it("reads only a subpath's own section", () => {
        const markdown = ['# API', '`outside`', '## Every export of `@pmndrs/gi/core`', '`createGI`', '### Sub', '`GpuTimer`', '## Every export of `@pmndrs/gi/ddgi`', '`DDGITechnique`'].join('\n');
        const code = markdownCode(exportSection(markdown, '@pmndrs/gi/core'));
        expect(namesMissingFrom(code, ['createGI', 'GpuTimer', 'DDGITechnique', 'outside'])).toEqual(['DDGITechnique', 'outside']);
        expect(() => exportSection(markdown, '@pmndrs/gi/three')).toThrow(/no "## Every export of/);
    });

    it('counts a name only inside code, as a whole word', () => {
        const markdown = ['The frame is drawn.', '', 'Call `createGI(device)` first.', '', '```ts', 'const budget: FrameBudget = {};', '```'].join('\n');
        const code = markdownCode(markdown);
        expect(namesMissingFrom(code, ['createGI', 'FrameBudget', 'frame', 'Frame', 'create'])).toEqual(['frame', 'Frame', 'create']);
    });
});
