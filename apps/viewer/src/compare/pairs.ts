/**
 * Which recordings the compare page offers side by side: for each scene and
 * motion that both sides recorded, ours and native's, on the fixed clock.
 * Native's runs of a step are listed with the ones on our DDGI's probe grid
 * (`-GridRule ddgi`, "-ddgi" in the name) first; its settle runs (one second
 * at a key, for the response tool) and wall-clock logs aren't motions to play.
 *
 * No DOM, so it is unit-tested.
 */

export interface ListedRecording {
    source: 'native' | 'viewer';
    name: string;
    url: string;
    scene: string;
    motion: string;
    clock: string;
}

export interface RecordingPair {
    /** `<scene>-<motion>`, the pair's key. */
    key: string;
    scene: string;
    motion: string;
    ours: ListedRecording[];
    native: ListedRecording[];
}

/** Native's recordings that aren't a motion played through: settle runs and logs. */
function playable(recording: ListedRecording): boolean {
    return recording.clock === 'fixed' && !/-settle\d*$/.test(recording.name) && !/-log$/.test(recording.name);
}

/** Native runs on our probe grid first, then by name. */
function nativeOrder(a: ListedRecording, b: ListedRecording): number {
    const grid = (recording: ListedRecording) => (/-ddgi/.test(recording.name) ? 0 : 1);
    return grid(a) - grid(b) || a.name.localeCompare(b.name);
}

export function pairRecordings(recordings: readonly ListedRecording[]): RecordingPair[] {
    const pairs = new Map<string, RecordingPair>();
    for (const recording of recordings.filter(playable)) {
        const key = `${recording.scene}-${recording.motion}`;
        let pair = pairs.get(key);
        if (!pair) {
            pair = { key, scene: recording.scene, motion: recording.motion, ours: [], native: [] };
            pairs.set(key, pair);
        }
        (recording.source === 'viewer' ? pair.ours : pair.native).push(recording);
    }
    return [...pairs.values()]
        .filter((pair) => pair.ours.length > 0 && pair.native.length > 0)
        .map((pair) => ({ ...pair, ours: [...pair.ours].sort((a, b) => a.name.localeCompare(b.name)), native: [...pair.native].sort(nativeOrder) }))
        .sort((a, b) => a.key.localeCompare(b.key));
}
