/**
 * Saved bakes: the `.ddgibake` file (docs/design/m5-coverage.md §2). A
 * converged DDGI field, saved so a page can start with finished GI and keep
 * updating from there. Plain bytes and plain numbers, so the format is
 * engine-neutral and unit-tested without a GPU; the technique's `saveField`
 * and `loadField` (ddgi-technique.ts) move the sections to and from the GPU.
 *
 * **The file, little-endian:**
 *
 * | bytes | what |
 * | --- | --- |
 * | 0-3 | the magic `DDGB` |
 * | 4-7 | `u32` format version ({@link SAVED_FIELD_FORMAT_VERSION}) |
 * | 8-11 | `u32` the header's length in bytes |
 * | 12- | the header, JSON in UTF-8 ({@link SavedFieldHeader}) |
 * | then | the sections, each starting on a {@link SECTION_ALIGNMENT}-byte boundary, zero-padded between |
 *
 * Every section's `byteOffset` counts from the start of the file. Texture
 * sections hold their texels row after row with no padding (WebGPU's
 * 256-byte row pitch is a readback detail, not part of the file), array
 * layer after array layer.
 *
 * **The sections:** the irradiance atlas with every irradiance layer side by
 * side (light groups, light-groups.ts), the distance atlas, the probe data
 * (relocation offsets and states), and each volume's probe history. The
 * history is what lets a loaded field carry on blending instead of warming
 * up again. The composed atlas isn't saved: the first frame after a load
 * composes it again.
 *
 * **The probe history is settled before it is written** ({@link settleProbeHistory}):
 * it keeps what the probe has learned (its sample count, update count, the
 * time of its last update on the technique's clock, its mean and noise) and
 * drops what only meant something in the frames it was saved in (owed
 * updates, marks, wakes, frame stamps, the change clock). So a field saved
 * while a change was still being taken in loads as if that change were done.
 */

import type { SceneFingerprint } from '../../platform/contract/scene-copy.ts';
import { halfToFloat } from '../../platform/contract/helpers.ts';
import { BYTES_PER_TEXEL, PROBE_HISTORY_FLAG, PROBE_HISTORY_UPDATES, PROBE_HISTORY_WORD, PROBE_HISTORY_WORDS, type DistanceFormat } from './layout.ts';

/** The first four bytes of every saved field. */
export const SAVED_FIELD_MAGIC = 'DDGB';
/**
 * The format this code writes and the only one it reads. Bump it on any
 * change to the header's meaning or a section's layout; an old file is then
 * refused with a message, never misread.
 */
export const SAVED_FIELD_FORMAT_VERSION = 1;
/** Sections start on this boundary, so a reader can map them straight into GPU buffers (WebGPU's copy-offset and row-pitch alignment). */
export const SECTION_ALIGNMENT = 256;
/** Bytes before the header: magic, version, header length. */
export const SAVED_FIELD_PREAMBLE_BYTES = 12;
/**
 * The writer, recorded in every header (the umbrella package and its
 * version). saved-field.test.ts checks it against packages/ddgi/package.json,
 * so it can't drift.
 */
export const SAVED_FIELD_WRITER = '@pmndrs/gi 0.0.0';

/** One volume's shape as the field was baked: what has to match to load it. */
export interface SavedVolumeLayout {
    /** Probes per axis. */
    readonly counts: readonly [number, number, number];
    /** Metres between probes, per axis. */
    readonly spacing: readonly [number, number, number];
    /** World position of its first probe as the settings place it (a camera-following volume's starting window). */
    readonly origin: readonly [number, number, number];
    readonly priority: number;
    readonly follow: 'fixed' | 'camera';
}

/**
 * What has to match between the saved field and the technique loading it
 * ({@link layoutMismatch}). Volumes in priority order, as the volume table
 * lists them. Volumes have no rotation (they are axis-aligned boxes), so
 * none is recorded; a future rotation is a format change.
 */
export interface SavedFieldLayout {
    /** Interior texels per side of a probe's irradiance tile. */
    readonly irradianceTexels: number;
    /** Interior texels per side of a probe's distance tile. */
    readonly distanceTexels: number;
    /** What each irradiance layer holds, in order (light-groups.ts): `everything` with groups off; else `lightGroup 0`.. then `environment`; and `sun` last with a sun layer. */
    readonly irradianceLayers: readonly string[];
    /** The distance atlas's texel format. Not part of the match: a load converts between the two (layout.ts, `DistanceFormat`). */
    readonly distanceFormat: DistanceFormat;
    readonly volumes: readonly SavedVolumeLayout[];
}

/** Where one section is and how to read it. */
export interface SavedFieldSection {
    /** `irradiance`, `distance`, `probeData`, or `probeHistory/<volume>` (volumes in priority order). */
    readonly name: string;
    /** The texel format (`rgba16float`, `rg16float`), or `probeHistory` for the history records (PROBE_HISTORY_WORD in layout.ts). */
    readonly format: string;
    readonly byteOffset: number;
    readonly byteLength: number;
    /** For a texture: texels across, rows, and array layers. */
    readonly width?: number;
    readonly height?: number;
    readonly layers?: number;
}

/** A camera-following volume's window as it was saved (scroll.ts): where it was on the lattice, and its valid extent. */
export interface SavedCameraWindow {
    /** Lattice coordinates of the window's first probe. */
    readonly window: readonly [number, number, number];
    /** The valid extent's low and high sides, in lattice units (what `ddgi/scroll` kept). */
    readonly validMin: readonly [number, number, number];
    readonly validMax: readonly [number, number, number];
}

/** The JSON header. */
export interface SavedFieldHeader {
    readonly formatVersion: number;
    /** What wrote it ({@link SAVED_FIELD_WRITER}). */
    readonly library: string;
    /** When it was saved, ISO 8601. */
    readonly createdAt: string;
    readonly layout: SavedFieldLayout;
    readonly sections: readonly SavedFieldSection[];
    /** The scene it was baked for (platform/scene/scene-fingerprint.ts). */
    readonly fingerprint: SceneFingerprint;
    /**
     * The technique's clock (seconds) when it was saved: each probe's history
     * holds the time of its last update on it, so a load sets the clock back
     * to this and the histories age from where they were.
     */
    readonly clockSeconds: number;
    /** Per volume, in priority order: its window when it follows the camera, else null. */
    readonly cameraWindows: readonly (SavedCameraWindow | null)[];
}

/** Why a saved field was refused. */
export type SavedFieldProblem = 'not a saved field' | 'format version' | 'layout' | 'scene changed';

/**
 * A saved field that can't be loaded, with a message that says why in plain
 * words, naming the field that differs for a layout mismatch.
 */
export class SavedFieldError extends Error {
    readonly problem: SavedFieldProblem;
    constructor(problem: SavedFieldProblem, message: string) {
        super(`@pmndrs/gi: ${message}`);
        this.name = 'SavedFieldError';
        this.problem = problem;
    }
}

/** A section's bytes before they are placed in the file. */
export interface SectionData extends Omit<SavedFieldSection, 'byteOffset' | 'byteLength'> {
    readonly bytes: Uint8Array;
}

/** Bytes over a plain `ArrayBuffer` (what WebGPU's writes take). */
export type FileBytes = Uint8Array<ArrayBuffer>;

/** A saved field's bytes as {@link FileBytes}: a view of an `ArrayBuffer` as it is, anything else copied. */
export function fileBytes(bytes: Uint8Array | ArrayBuffer): FileBytes {
    if (bytes instanceof ArrayBuffer) return new Uint8Array(bytes);
    return bytes.buffer instanceof ArrayBuffer ? (bytes as FileBytes) : new Uint8Array(bytes);
}

/** Rounds up to the next multiple of `alignment`. */
export function alignUp(value: number, alignment: number): number {
    return Math.ceil(value / alignment) * alignment;
}

/**
 * Writes a saved field: the preamble, the header with every section's
 * offset filled in, and the sections on their boundaries.
 */
export function encodeSavedField(header: Omit<SavedFieldHeader, 'sections' | 'formatVersion'>, sections: readonly SectionData[]): FileBytes {
    const encoder = new TextEncoder();
    // The header holds the sections' offsets, and its own length moves where they start: a few rounds settle it
    // (each round can only add digits, and an offset's digits stop growing at once).
    let dataStart = alignUp(SAVED_FIELD_PREAMBLE_BYTES, SECTION_ALIGNMENT);
    let headerBytes = new Uint8Array(0);
    let placed: SavedFieldSection[] = [];
    for (let round = 0; round < 8; round++) {
        let offset = dataStart;
        placed = sections.map(({ bytes, ...section }) => {
            const entry = { ...section, byteOffset: offset, byteLength: bytes.byteLength };
            offset = alignUp(offset + bytes.byteLength, SECTION_ALIGNMENT);
            return entry;
        });
        const full: SavedFieldHeader = { formatVersion: SAVED_FIELD_FORMAT_VERSION, ...header, sections: placed };
        headerBytes = encoder.encode(JSON.stringify(full));
        const needed = alignUp(SAVED_FIELD_PREAMBLE_BYTES + headerBytes.byteLength, SECTION_ALIGNMENT);
        if (needed === dataStart) break;
        dataStart = needed;
    }
    const last = placed[placed.length - 1];
    const total = last ? last.byteOffset + last.byteLength : dataStart;
    const file = new Uint8Array(total);
    const view = new DataView(file.buffer);
    file.set(encoder.encode(SAVED_FIELD_MAGIC), 0);
    view.setUint32(4, SAVED_FIELD_FORMAT_VERSION, true);
    view.setUint32(8, headerBytes.byteLength, true);
    file.set(headerBytes, SAVED_FIELD_PREAMBLE_BYTES);
    placed.forEach((section, index) => file.set(sections[index].bytes, section.byteOffset));
    return file;
}

/** A saved field read back: its header, and each section's bytes (views into the file, by name). */
export interface DecodedSavedField {
    readonly header: SavedFieldHeader;
    readonly sections: ReadonlyMap<string, { readonly section: SavedFieldSection; readonly bytes: FileBytes }>;
}

/**
 * Reads a saved field's preamble, header and sections, or throws
 * `SavedFieldError`: not a saved field (no magic, cut short), or a format
 * version this code doesn't read.
 */
export function decodeSavedField(file: FileBytes): DecodedSavedField {
    const notAField = (why: string): SavedFieldError => new SavedFieldError('not a saved field', `this isn't a saved DDGI field (${why}).`);
    if (file.byteLength < SAVED_FIELD_PREAMBLE_BYTES) throw notAField(`it is only ${file.byteLength} bytes long`);
    const magic = new TextDecoder().decode(file.subarray(0, 4));
    if (magic !== SAVED_FIELD_MAGIC) throw notAField(`it doesn't start with "${SAVED_FIELD_MAGIC}"`);
    const view = new DataView(file.buffer, file.byteOffset, file.byteLength);
    const version = view.getUint32(4, true);
    if (version !== SAVED_FIELD_FORMAT_VERSION) {
        const which = version > SAVED_FIELD_FORMAT_VERSION ? 'a newer version of the library' : 'an older version of the library';
        throw new SavedFieldError('format version', `this saved field is in format ${version}, from ${which}; this one reads format ${SAVED_FIELD_FORMAT_VERSION} only. Save the field again with this version.`);
    }
    const headerLength = view.getUint32(8, true);
    if (SAVED_FIELD_PREAMBLE_BYTES + headerLength > file.byteLength) throw notAField('its header runs past the end of the file; it may be cut short');
    let header: SavedFieldHeader;
    try {
        header = JSON.parse(new TextDecoder().decode(file.subarray(SAVED_FIELD_PREAMBLE_BYTES, SAVED_FIELD_PREAMBLE_BYTES + headerLength))) as SavedFieldHeader;
    } catch {
        throw notAField('its header is not readable JSON');
    }
    if (header.formatVersion !== version) throw notAField(`its header says format ${header.formatVersion} but the file says ${version}`);
    const sections = new Map<string, { section: SavedFieldSection; bytes: FileBytes }>();
    for (const section of header.sections ?? []) {
        if (section.byteOffset % SECTION_ALIGNMENT !== 0) throw notAField(`section "${section.name}" doesn't start on a ${SECTION_ALIGNMENT}-byte boundary`);
        if (section.byteOffset + section.byteLength > file.byteLength) throw notAField(`section "${section.name}" runs past the end of the file; it may be cut short`);
        sections.set(section.name, { section, bytes: file.subarray(section.byteOffset, section.byteOffset + section.byteLength) });
    }
    return { header, sections };
}

/** Section names. */
export const SECTION = {
    irradiance: 'irradiance',
    distance: 'distance',
    probeData: 'probeData',
    probeHistory: (volume: number): string => `probeHistory/${volume}`,
} as const;

/** How far apart two positions or spacings may be and still match: well under any probe spacing, above JSON or f32 noise. */
const LENGTH_TOLERANCE_METRES = 1e-4;

/**
 * Why a saved field's layout can't be loaded into a field of this layout,
 * as a plain sentence naming the field that differs; null when it can.
 */
export function layoutMismatch(saved: SavedFieldLayout, current: SavedFieldLayout): string | null {
    if (saved.irradianceTexels !== current.irradianceTexels) {
        return `the irradiance tiles differ: the saved field has ${saved.irradianceTexels} x ${saved.irradianceTexels} texels a probe, this one ${current.irradianceTexels} x ${current.irradianceTexels} (irradianceTexels)`;
    }
    if (saved.distanceTexels !== current.distanceTexels) {
        return `the distance tiles differ: the saved field has ${saved.distanceTexels} x ${saved.distanceTexels} texels a probe, this one ${current.distanceTexels} x ${current.distanceTexels} (distanceTexels)`;
    }
    if (saved.irradianceLayers.join(',') !== current.irradianceLayers.join(',')) {
        return `the irradiance layers differ: the saved field has ${saved.irradianceLayers.length} (${saved.irradianceLayers.join(', ')}), this one ${current.irradianceLayers.length} (${current.irradianceLayers.join(', ')}) (lightGroups, sunLayer)`;
    }
    if (saved.volumes.length !== current.volumes.length) return `the saved field has ${saved.volumes.length} volume(s), this one ${current.volumes.length} (volumes)`;
    for (let index = 0; index < saved.volumes.length; index++) {
        const was = saved.volumes[index];
        const now = current.volumes[index];
        const which = saved.volumes.length > 1 ? `volume ${index} (by priority)` : 'the volume';
        if (was.counts.join() !== now.counts.join()) return `${which} has other probe counts: the saved field ${was.counts.join(' x ')}, this one ${now.counts.join(' x ')} (its size and spacing)`;
        if (!closeVectors(was.spacing, now.spacing)) return `${which} has another spacing: the saved field ${formatVector(was.spacing)} m, this one ${formatVector(now.spacing)} m`;
        if (!closeVectors(was.origin, now.origin)) return `${which} sits elsewhere: its first probe is at ${formatVector(was.origin)} in the saved field, at ${formatVector(now.origin)} here (its center)`;
        if (was.priority !== now.priority) return `${which} has another priority: ${was.priority} in the saved field, ${now.priority} here`;
        if (was.follow !== now.follow) return `${which} ${was.follow === 'camera' ? 'follows the camera in the saved field and is fixed here' : 'is fixed in the saved field and follows the camera here'} (follow)`;
    }
    return null;
}

function closeVectors(a: readonly number[], b: readonly number[]): boolean {
    return a.length === b.length && a.every((value, axis) => Math.abs(value - b[axis]) <= LENGTH_TOLERANCE_METRES);
}

function formatVector(vector: readonly number[]): string {
    return `(${vector.map((value) => Number(value.toFixed(4))).join(', ')})`;
}

/**
 * Settles saved probe-history records in place (the file comment): keeps
 * what each probe learned, drops what only meant something in the frames it
 * was saved in. One record is PROBE_HISTORY_WORDS words (layout.ts).
 *
 * Kept: `samples`, `lastUpdateSeconds`, the update count and the sky
 * transfer's count in `updates`, `meanRedGreen`, `meanBlueNoise`, and the
 * `warmedOnce` flag (a probe in state 3 with one blended update needs one
 * more before shading uses it, as before the save). Cleared: owed updates,
 * `pendingChange`, `urgency`, the other flags and the scheduler's frame
 * stamp, `wakeUntil`, `keysReceived`, `changeClock` (a loaded field's change
 * clock starts at 0, so 0 stamps say "no change seen since"), `keyDeltaSlot`,
 * `moverMask` (the mover table is empty after a load) and `awaitingSinceFrame`.
 */
export function settleProbeHistory(records: Uint32Array): void {
    if (records.length % PROBE_HISTORY_WORDS !== 0) throw new RangeError(`probe history is ${PROBE_HISTORY_WORDS} words a probe; got ${records.length} words`);
    const keptUpdates = PROBE_HISTORY_UPDATES.countMask | (0xff << PROBE_HISTORY_UPDATES.transferShift);
    for (let base = 0; base < records.length; base += PROBE_HISTORY_WORDS) {
        const samples = records[base + PROBE_HISTORY_WORD.samples];
        const lastUpdate = records[base + PROBE_HISTORY_WORD.lastUpdateSeconds];
        const updates = (records[base + PROBE_HISTORY_WORD.updates] & keptUpdates) >>> 0;
        const flags = records[base + PROBE_HISTORY_WORD.scheduleAndFlags] & PROBE_HISTORY_FLAG.warmedOnce;
        const meanRedGreen = records[base + PROBE_HISTORY_WORD.meanRedGreen];
        const meanBlueNoise = records[base + PROBE_HISTORY_WORD.meanBlueNoise];
        records.fill(0, base, base + PROBE_HISTORY_WORDS);
        records[base + PROBE_HISTORY_WORD.samples] = samples;
        records[base + PROBE_HISTORY_WORD.lastUpdateSeconds] = lastUpdate;
        records[base + PROBE_HISTORY_WORD.updates] = updates;
        records[base + PROBE_HISTORY_WORD.scheduleAndFlags] = flags;
        records[base + PROBE_HISTORY_WORD.meanRedGreen] = meanRedGreen;
        records[base + PROBE_HISTORY_WORD.meanBlueNoise] = meanBlueNoise;
    }
}

/**
 * Distance texels in another of the two formats (layout.ts,
 * `DistanceFormat`): `rg16float` holds (mean, mean squared), `rgba16float`
 * the same two halves and two zero halves, as `ddgi/write` stores them. So
 * a field baked on a device with `texture-formats-tier1` loads on one
 * without it, and the other way, exactly.
 */
export function convertDistanceTexels(bytes: FileBytes, from: DistanceFormat, to: DistanceFormat): FileBytes {
    if (from === to) return bytes;
    const fromTexel = BYTES_PER_TEXEL[from];
    const toTexel = BYTES_PER_TEXEL[to];
    const texels = bytes.byteLength / fromTexel;
    const out = new Uint8Array(texels * toTexel);
    // Both formats start with the same two halves (4 bytes); the wider one has two zero halves after them.
    for (let texel = 0; texel < texels; texel++) out.set(bytes.subarray(texel * fromTexel, texel * fromTexel + 4), texel * toTexel);
    return out;
}

/**
 * Texture rows packed tightly from a readback with a padded row pitch
 * (WebGPU copies to buffers in rows of a multiple of 256 bytes).
 */
export function unpadRows(padded: FileBytes, rowBytes: number, paddedRowBytes: number, rows: number): FileBytes {
    if (rowBytes === paddedRowBytes) return padded.subarray(0, rowBytes * rows);
    const out = new Uint8Array(rowBytes * rows);
    for (let row = 0; row < rows; row++) out.set(padded.subarray(row * paddedRowBytes, row * paddedRowBytes + rowBytes), row * rowBytes);
    return out;
}

/** One probe-data texel (rgba16float): the relocation offset in spacings and the state. */
export function readProbeDataTexel(halves: Uint16Array, texel: number): { readonly offset: readonly [number, number, number]; readonly state: number } {
    const base = texel * 4;
    return { offset: [halfToFloat(halves[base]), halfToFloat(halves[base + 1]), halfToFloat(halves[base + 2])], state: Math.round(halfToFloat(halves[base + 3])) };
}
