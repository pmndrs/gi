/**
 * Writes the viewer's motion recordings out of a GPU runner report (T2.8
 * check 3): `tools/gpu/pages/viewer-recording` hands each recording back in
 * its report (`data.recordings`, images as data URLs), and this writes each
 * as a recording folder in the native harness's layout
 * (tools/native/README.md, "Motions and recordings"): `recording.json`,
 * `frames.csv` and `frames/frame-NNNNNN.jpg`.
 *
 *   node tools/native/save-viewer-recording.ts <report.json> [--out bench/recordings/viewer]
 *
 * The folder is git-ignored; the compare page (apps/viewer/compare.html)
 * reads it.
 */

import { mkdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';

const repoRoot = resolve(dirname(fileURLToPath(import.meta.url)), '..', '..');
export const DEFAULT_VIEWER_RECORDINGS = join(repoRoot, 'bench', 'recordings', 'viewer');

interface ReportedRecording {
    name: string;
    info: Record<string, unknown>;
    framesCsv: string;
    images: Array<{ path: string; dataUrl: string }>;
}

/** The recordings in a runner report (`pages[].data.recordings`). */
export function recordingsInReport(report: unknown): ReportedRecording[] {
    const pages = (report as { pages?: Array<{ data?: { recordings?: ReportedRecording[] } }> }).pages ?? [];
    return pages.flatMap((page) => page.data?.recordings ?? []);
}

/** Writes one recording folder; returns its path. A name is a folder name only (no path parts). */
export function writeRecording(recording: ReportedRecording, outDirectory: string): string {
    if (!/^[a-z0-9-]+$/.test(recording.name)) throw new Error(`not a recording name: ${recording.name}`);
    const folder = join(outDirectory, recording.name);
    rmSync(folder, { recursive: true, force: true });
    mkdirSync(join(folder, 'frames'), { recursive: true });
    writeFileSync(join(folder, 'recording.json'), `${JSON.stringify(recording.info, null, 2)}\n`);
    writeFileSync(join(folder, 'frames.csv'), recording.framesCsv);
    for (const image of recording.images) {
        if (!/^frames\/frame-\d+\.jpg$/.test(image.path)) throw new Error(`not an image path: ${image.path}`);
        const base64 = image.dataUrl.slice(image.dataUrl.indexOf(',') + 1);
        writeFileSync(join(folder, image.path), Buffer.from(base64, 'base64'));
    }
    return folder;
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
    const { positionals, values } = parseArgs({ allowPositionals: true, options: { out: { type: 'string' } } });
    if (positionals.length !== 1) {
        console.error('usage: node tools/native/save-viewer-recording.ts <report.json> [--out bench/recordings/viewer]');
        process.exit(2);
    }
    const recordings = recordingsInReport(JSON.parse(readFileSync(positionals[0], 'utf8')));
    if (recordings.length === 0) {
        console.error('the report holds no recordings (did the viewer-recording page pass?)');
        process.exit(1);
    }
    const out = values.out ? resolve(values.out) : DEFAULT_VIEWER_RECORDINGS;
    for (const recording of recordings) console.log(`wrote ${writeRecording(recording, out)} (${recording.images.length} images)`);
}
