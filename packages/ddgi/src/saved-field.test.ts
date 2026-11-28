/**
 * The `.ddgibake` format (T5.2, saved-field.ts): the preamble and header,
 * the sections' 256-byte alignment, the version check, the layout check's
 * messages, the settled history and the distance-format conversion. The GPU
 * page `ddgi-saved-bake` runs a real save and load.
 */

import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { PROBE_HISTORY_FLAG, PROBE_HISTORY_UPDATES, PROBE_HISTORY_WORD, PROBE_HISTORY_WORDS } from './layout.ts';
import {
    convertDistanceTexels,
    decodeSavedField,
    encodeSavedField,
    fileBytes,
    layoutMismatch,
    SAVED_FIELD_FORMAT_VERSION,
    SAVED_FIELD_MAGIC,
    SAVED_FIELD_PREAMBLE_BYTES,
    SAVED_FIELD_WRITER,
    SavedFieldError,
    SECTION_ALIGNMENT,
    settleProbeHistory,
    unpadRows,
    type SavedFieldHeader,
    type SavedFieldLayout,
} from './saved-field.ts';

const LAYOUT: SavedFieldLayout = {
    irradianceTexels: 6,
    distanceTexels: 14,
    irradianceLayers: ['everything'],
    distanceFormat: 'rg16float',
    volumes: [{ counts: [12, 12, 12], spacing: [0.25, 0.25, 0.25], origin: [-1.375, 0.125, -1.375], priority: 0, follow: 'fixed' }],
};

const HEADER: Omit<SavedFieldHeader, 'sections' | 'formatVersion'> = {
    library: SAVED_FIELD_WRITER,
    createdAt: '2026-11-10T12:00:00.000Z',
    layout: LAYOUT,
    fingerprint: { version: 1, geometry: '0123456789abcdef', materials: '1', textures: '2', lights: '3', sky: '4' },
    clockSeconds: 12.5,
    cameraWindows: [null],
};

function sampleFile(): Uint8Array<ArrayBuffer> {
    return encodeSavedField(HEADER, [
        { name: 'irradiance', format: 'rgba16float', width: 2, height: 1, layers: 1, bytes: new Uint8Array(16).map((_, index) => index + 1) },
        // 301 bytes: not a multiple of the alignment, so the next section's padding is tested.
        { name: 'probeData', format: 'rgba16float', bytes: new Uint8Array(301).fill(7) },
        { name: 'probeHistory/0', format: 'probeHistory', bytes: new Uint8Array(64).fill(9) },
    ]);
}

describe('the .ddgibake file', () => {
    it('starts with the magic, the version and the header length, and the header reads back', () => {
        const file = sampleFile();
        const view = new DataView(file.buffer);
        expect(new TextDecoder().decode(file.subarray(0, 4))).toBe(SAVED_FIELD_MAGIC);
        expect(view.getUint32(4, true)).toBe(SAVED_FIELD_FORMAT_VERSION);
        const headerLength = view.getUint32(8, true);
        const header = JSON.parse(new TextDecoder().decode(file.subarray(SAVED_FIELD_PREAMBLE_BYTES, SAVED_FIELD_PREAMBLE_BYTES + headerLength))) as SavedFieldHeader;
        expect(header.formatVersion).toBe(SAVED_FIELD_FORMAT_VERSION);
        expect(header.layout).toEqual(LAYOUT);
        expect(header.fingerprint).toEqual(HEADER.fingerprint);
        expect(header.clockSeconds).toBe(12.5);
        expect(header.sections.map((section) => section.name)).toEqual(['irradiance', 'probeData', 'probeHistory/0']);
    });

    it('puts every section on a 256-byte boundary past the header, with its bytes intact', () => {
        const file = sampleFile();
        const { header, sections } = decodeSavedField(file);
        const headerEnd = SAVED_FIELD_PREAMBLE_BYTES + new DataView(file.buffer).getUint32(8, true);
        let previousEnd = headerEnd;
        for (const section of header.sections) {
            expect(section.byteOffset % SECTION_ALIGNMENT).toBe(0);
            expect(section.byteOffset).toBeGreaterThanOrEqual(previousEnd);
            previousEnd = section.byteOffset + section.byteLength;
        }
        expect([...sections.get('irradiance')!.bytes]).toEqual(Array.from({ length: 16 }, (_, index) => index + 1));
        expect(sections.get('probeData')!.bytes.every((value) => value === 7)).toBe(true);
        expect(sections.get('probeData')!.bytes.byteLength).toBe(301);
        expect(sections.get('probeHistory/0')!.bytes.every((value) => value === 9)).toBe(true);
        expect(file.byteLength).toBe(previousEnd);
    });

    it('refuses another format version with a message that says which', () => {
        const newer = sampleFile();
        new DataView(newer.buffer).setUint32(4, SAVED_FIELD_FORMAT_VERSION + 1, true);
        expect(() => decodeSavedField(newer)).toThrow(SavedFieldError);
        expect(() => decodeSavedField(newer)).toThrow(/format 2, from a newer version of the library; this one reads format 1/);
        const older = sampleFile();
        new DataView(older.buffer).setUint32(4, 0, true);
        expect(() => decodeSavedField(older)).toThrow(/from an older version/);
    });

    it('refuses a file that is not a saved field, or is cut short', () => {
        expect(() => decodeSavedField(new Uint8Array(4))).toThrow(/isn't a saved DDGI field \(it is only 4 bytes long\)/);
        const png = sampleFile();
        png.set([0x89, 0x50, 0x4e, 0x47], 0);
        expect(() => decodeSavedField(png)).toThrow(/doesn't start with "DDGB"/);
        const file = sampleFile();
        expect(() => decodeSavedField(file.slice(0, file.byteLength - 10))).toThrow(/runs past the end of the file/);
        try {
            decodeSavedField(png);
        } catch (error) {
            expect((error as SavedFieldError).problem).toBe('not a saved field');
        }
    });

    it('takes an ArrayBuffer or a view as it is, and copies anything else', () => {
        const file = sampleFile();
        expect(fileBytes(file)).toBe(file);
        expect(fileBytes(file.buffer).byteLength).toBe(file.byteLength);
    });

    it('records the umbrella package version as its writer', () => {
        const manifest = JSON.parse(readFileSync(new URL('../../gi/package.json', import.meta.url), 'utf8')) as { name: string; version: string };
        expect(SAVED_FIELD_WRITER).toBe(`${manifest.name} ${manifest.version}`);
    });
});

describe('the layout check', () => {
    const volume = LAYOUT.volumes[0];
    const withVolume = (change: Partial<SavedFieldLayout['volumes'][number]>): SavedFieldLayout => ({ ...LAYOUT, volumes: [{ ...volume, ...change }] });

    it('lets the same layout through, and another distance format too (a load converts it)', () => {
        expect(layoutMismatch(LAYOUT, LAYOUT)).toBeNull();
        expect(layoutMismatch(LAYOUT, { ...LAYOUT, distanceFormat: 'rgba16float' })).toBeNull();
        // JSON and f32 noise in a position is not a mismatch.
        expect(layoutMismatch(LAYOUT, withVolume({ origin: [-1.375000001, 0.125, -1.375] }))).toBeNull();
    });

    it('names the field that differs', () => {
        expect(layoutMismatch(LAYOUT, { ...LAYOUT, irradianceTexels: 8 })).toMatch(/irradiance tiles differ: the saved field has 6 x 6 texels a probe, this one 8 x 8 \(irradianceTexels\)/);
        expect(layoutMismatch(LAYOUT, { ...LAYOUT, distanceTexels: 16 })).toMatch(/distance tiles differ.*\(distanceTexels\)/);
        expect(layoutMismatch(LAYOUT, { ...LAYOUT, irradianceLayers: ['lightGroup 0', 'environment'] })).toMatch(/irradiance layers differ: the saved field has 1 \(everything\), this one 2/);
        expect(layoutMismatch(LAYOUT, withVolume({ counts: [10, 12, 12] }))).toMatch(/the volume has other probe counts: the saved field 12 x 12 x 12, this one 10 x 12 x 12/);
        expect(layoutMismatch(LAYOUT, withVolume({ spacing: [0.3, 0.25, 0.25] }))).toMatch(/another spacing/);
        expect(layoutMismatch(LAYOUT, withVolume({ origin: [0, 0.125, -1.375] }))).toMatch(/sits elsewhere: its first probe is at \(-1.375, 0.125, -1.375\) in the saved field/);
        expect(layoutMismatch(LAYOUT, withVolume({ priority: 2 }))).toMatch(/another priority/);
        expect(layoutMismatch(LAYOUT, withVolume({ follow: 'camera' }))).toMatch(/is fixed in the saved field and follows the camera here/);
        expect(layoutMismatch(LAYOUT, { ...LAYOUT, volumes: [volume, volume] })).toMatch(/the saved field has 1 volume\(s\), this one 2/);
    });
});

describe('the settled probe history', () => {
    it('keeps what a probe learned and drops what belonged to the frames it was saved in', () => {
        const records = new Uint32Array(PROBE_HISTORY_WORDS * 2);
        const floats = new Float32Array(records.buffer);
        for (let word = 0; word < records.length; word++) records[word] = 0xdead0000 + word;
        floats[PROBE_HISTORY_WORD.samples] = 14.5;
        floats[PROBE_HISTORY_WORD.lastUpdateSeconds] = 3.25;
        // 200 updates, 3 owed, a sky transfer count of 5.
        records[PROBE_HISTORY_WORD.updates] = 200 | (3 << PROBE_HISTORY_UPDATES.owedShift) | (5 << PROBE_HISTORY_UPDATES.transferShift);
        // Scheduled in frame 0x1234, marked, taking a change, warmed once.
        records[PROBE_HISTORY_WORD.scheduleAndFlags] = (0x1234 << 8) | PROBE_HISTORY_FLAG.marked | PROBE_HISTORY_FLAG.takingChange | PROBE_HISTORY_FLAG.warmedOnce;
        settleProbeHistory(records);
        expect(floats[PROBE_HISTORY_WORD.samples]).toBe(14.5);
        expect(floats[PROBE_HISTORY_WORD.lastUpdateSeconds]).toBe(3.25);
        expect(records[PROBE_HISTORY_WORD.updates]).toBe((200 | (5 << PROBE_HISTORY_UPDATES.transferShift)) >>> 0);
        expect(records[PROBE_HISTORY_WORD.scheduleAndFlags]).toBe(PROBE_HISTORY_FLAG.warmedOnce);
        expect(records[PROBE_HISTORY_WORD.meanRedGreen]).toBe(0xdead0000 + PROBE_HISTORY_WORD.meanRedGreen);
        expect(records[PROBE_HISTORY_WORD.meanBlueNoise]).toBe(0xdead0000 + PROBE_HISTORY_WORD.meanBlueNoise);
        for (const word of ['pendingChange', 'urgency', 'wakeUntil', 'keysReceived', 'changeClock', 'keyDeltaSlot', 'moverMask', 'awaitingSinceFrame'] as const) {
            expect(records[PROBE_HISTORY_WORD[word]], word).toBe(0);
        }
        // The second record too, and its spare words.
        expect(records[PROBE_HISTORY_WORDS + PROBE_HISTORY_WORD.wakeUntil]).toBe(0);
        expect(records[PROBE_HISTORY_WORDS + 15]).toBe(0);
    });

    it('refuses a buffer that is not whole records', () => {
        expect(() => settleProbeHistory(new Uint32Array(PROBE_HISTORY_WORDS + 1))).toThrow(RangeError);
    });
});

describe('texel helpers', () => {
    it('converts distance texels between rg16float and rgba16float exactly', () => {
        const rg = new Uint8Array([1, 2, 3, 4, 5, 6, 7, 8]);
        const rgba = convertDistanceTexels(rg, 'rg16float', 'rgba16float');
        expect([...rgba]).toEqual([1, 2, 3, 4, 0, 0, 0, 0, 5, 6, 7, 8, 0, 0, 0, 0]);
        expect([...convertDistanceTexels(rgba, 'rgba16float', 'rg16float')]).toEqual([...rg]);
        expect(convertDistanceTexels(rg, 'rg16float', 'rg16float')).toBe(rg);
    });

    it('packs padded readback rows', () => {
        const padded = new Uint8Array([1, 2, 0, 0, 3, 4, 0, 0]);
        expect([...unpadRows(padded, 2, 4, 2)]).toEqual([1, 2, 3, 4]);
    });
});
