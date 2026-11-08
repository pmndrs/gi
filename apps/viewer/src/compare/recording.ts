/**
 * Motion recordings, ours and native's, as the compare page reads them
 * (T2.8 check 3, docs/design/m2-lights.md §7 "Native side by side").
 *
 * Both sides write the same folder shape (tools/native/README.md, "Motions
 * and recordings"): `recording.json`, `frames.csv` with one row a frame, and
 * an image every few frames. Both run on a fixed clock of 1/60 s a frame
 * (frame k shows the motion at k / 60 s), so the motion's time lines them up
 * frame for frame, whatever either took to draw.
 *
 * - **Native** (the RTXGI harness, T2.7): `frames.csv` holds `gi_mean_r`,
 *   `gi_mean_g`, `gi_mean_b`: the mean over the view of its GI-only image
 *   (albedo / π × irradiance, linear), PNGs every 4th frame.
 * - **Ours** (`tools/gpu/pages/viewer-recording`, the viewer's own code):
 *   `gi_mean` is the response meter's reading (`DdgiTechnique.readRegionMean`:
 *   the mean of the red, green and blue of DDGI's GI-only picture, linear),
 *   JPEGs of the picture on screen every 4th frame.
 *
 * Each side's brightness is `curveValue`: the mean of red, green and blue,
 * the same measure for both. The two renderers' absolute levels differ (the
 * native image is albedo / π × irradiance, ours three's materials), so the
 * plot can show each curve relative to its own first frame.
 *
 * No DOM and no engine, so it is unit-tested.
 */

/** `recording.json`: what both sides write (more fields may be there). */
export interface RecordingInfo {
    format: string;
    scene: string;
    motion: string;
    camera: string;
    width: number;
    height: number;
    /** `full` (direct light and GI) or `indirect` (GI only), the images' view. */
    view: string;
    clock: string;
    frameRate: number;
    frames: number;
    imageEvery: number;
    device?: string;
    /** Who made it: `native` (the RTXGI harness) or `viewer` (ours). */
    source?: string;
    /** Probes updated a frame: native updates every probe every frame; ours its budget. */
    probeUpdate?: string;
    [field: string]: unknown;
}

export interface RecordedFrame {
    frame: number;
    /** Seconds into the motion (frame / frame rate). */
    time: number;
    /** The image's path inside the recording, or null for frames without one. */
    image: string | null;
    /** The GI-only picture's brightness: the mean of its red, green and blue, linear. */
    value: number;
}

export interface Recording {
    info: RecordingInfo;
    frames: RecordedFrame[];
    /** The frames with an image, in order. */
    imageFrames: RecordedFrame[];
}

/** One CSV line into fields (no quoting is used by either side). */
function fields(line: string): string[] {
    return line.split(',').map((field) => field.trim());
}

/**
 * Reads a `frames.csv` from either side. The brightness is `gi_mean` when
 * there is one (ours), else the mean of `gi_mean_r`, `gi_mean_g` and
 * `gi_mean_b` (native's).
 */
export function parseFramesCsv(text: string, frameRate: number): RecordedFrame[] {
    const lines = text.split(/\r?\n/).filter((line) => line.trim() !== '' && !line.startsWith('#'));
    if (lines.length === 0) return [];
    const header = fields(lines[0]);
    const column = (name: string) => header.indexOf(name);
    const frameColumn = column('frame');
    const timeColumn = column('motion_time_s');
    const imageColumn = column('image');
    const meanColumn = column('gi_mean');
    const channels = [column('gi_mean_r'), column('gi_mean_g'), column('gi_mean_b')];
    if (frameColumn < 0) throw new Error('frames.csv has no "frame" column');
    if (meanColumn < 0 && channels.some((index) => index < 0)) throw new Error('frames.csv has neither "gi_mean" nor "gi_mean_r/g/b"');
    return lines.slice(1).map((line) => {
        const row = fields(line);
        const frame = Number(row[frameColumn]);
        const value = meanColumn >= 0 ? Number(row[meanColumn]) : channels.reduce((sum, index) => sum + Number(row[index]), 0) / 3;
        const time = timeColumn >= 0 ? Number(row[timeColumn]) : frame / frameRate;
        const image = imageColumn >= 0 && row[imageColumn] ? row[imageColumn] : null;
        return { frame, time, image, value };
    });
}

export function makeRecording(info: RecordingInfo, framesCsv: string): Recording {
    const frames = parseFramesCsv(framesCsv, info.frameRate);
    return { info, frames, imageFrames: frames.filter((frame) => frame.image !== null) };
}

/** The recording's length in seconds of motion: its last frame's time. */
export function duration(recording: Recording): number {
    return recording.frames.length > 0 ? recording.frames[recording.frames.length - 1].time : 0;
}

/**
 * The image to show at motion time `time`: the last frame with an image at
 * or before it (so both sides show what each had drawn by then), or the
 * first image before the recording starts.
 */
export function imageAt(recording: Recording, time: number): RecordedFrame | null {
    const images = recording.imageFrames;
    if (images.length === 0) return null;
    // Binary search: the last image frame with time <= `time` (a thousandth of a frame of slack for rounding).
    let low = 0;
    let high = images.length - 1;
    if (images[0].time > time) return images[0];
    while (low < high) {
        const middle = Math.ceil((low + high) / 2);
        if (images[middle].time <= time + 1e-6) low = middle;
        else high = middle - 1;
    }
    return images[low];
}

/** The frame at or just before motion time `time` (for the readout and the curve's marker). */
export function frameAt(recording: Recording, time: number): RecordedFrame | null {
    const frames = recording.frames;
    if (frames.length === 0) return null;
    const index = Math.max(0, Math.min(frames.length - 1, Math.floor(time * recording.info.frameRate + 1e-6)));
    return frames[index];
}

export type CurveScale = 'relative' | 'measured';

/** A curve's values as plotted: as measured, or relative to its own first frame (1 = where it started). */
export function curveValues(recording: Recording, scale: CurveScale): number[] {
    const first = recording.frames[0]?.value ?? 0;
    return recording.frames.map((frame) => (scale === 'relative' ? (first !== 0 ? frame.value / first : 0) : frame.value));
}

/** `frames.csv` for our recording, in the native column layout where they share a meaning. */
export function framesCsv(frames: ReadonlyArray<{ frame: number; time: number; wallTime: number; image: string | null; value: number; probesUpdated: number; light: readonly [number, number, number] | null }>, lightLabel: string): string {
    const header = ['frame', 'motion_time_s', 'wall_time_s', 'image', 'gi_mean', 'probes_updated', `${lightLabel}.position.x`, `${lightLabel}.position.y`, `${lightLabel}.position.z`];
    const rows = frames.map((frame) =>
        [
            frame.frame,
            frame.time.toFixed(10),
            frame.wallTime.toFixed(4),
            frame.image ?? '',
            frame.value.toPrecision(9),
            frame.probesUpdated,
            ...(frame.light ? frame.light.map((value) => value.toPrecision(9)) : ['', '', '']),
        ].join(','),
    );
    return `${[header.join(','), ...rows].join('\n')}\n`;
}
