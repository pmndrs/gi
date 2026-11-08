/**
 * The native compare page (T2.8 check 3; docs/design/m2-lights.md §7,
 * "Native side by side"): our DDGI's recording of a light motion and the
 * native RTXGI harness's recording of the same motion, played together on
 * the motion's clock, with both brightness curves on one plot.
 *
 * Both recordings run on a fixed clock of 1/60 s a frame, so one time,
 * seconds into the motion, picks each side's frame (`recording.ts`). Play
 * moves that time on at the chosen speed; the scrub bar and a click on the
 * plot set it. Native shows live inside the viewer is not attempted; the
 * recordings come from:
 * - ours: `tools/gpu/pages/viewer-recording` (the viewer's own code, GI only,
 *   the response meter's reading each frame), saved by
 *   `tools/native/save-viewer-recording.ts` into bench/recordings/viewer;
 * - native: `tools/native/play.ps1 -Record` into native/build/recordings.
 * The dev server lists both (`/__recordings`, vite.config.ts).
 *
 * The link picks a pair: `compare.html?motion=two-room-light-step&native=<name>&ours=<name>&t=3.5`.
 *
 * Dennis (2026-11-05): the look is judged by eye; native traces with
 * hardware ray tracing, and matching it is not a requirement. The page says so.
 */

import { h } from '../ui/dom';
import { pairRecordings, type ListedRecording, type RecordingPair } from './pairs';
import { curveValues, duration, frameAt, imageAt, makeRecording, type CurveScale, type Recording, type RecordingInfo } from './recording';

const HELP = {
    default: 'Hover over any control to see what it does.',
    motion: 'Which light motion to compare. Both sides played the same motion from the scene file, on the same clock: frame k is k / 60 seconds into the motion.',
    nativeRun: "Which of native's recordings of this motion to show. Runs marked \"ddgi\" use our DDGI's probe grid (the same probe positions as ours); the others use native's own.",
    oursRun: 'Which of our recordings of this motion to show.',
    play: 'Plays both recordings together. Each side shows the picture it had drawn by that moment of the motion.',
    scrub: 'Drag to any moment of the motion; both sides follow.',
    speed: 'How fast to play: 1 is real time (the motion as it plays in the viewer).',
    scale:
        "How to draw the two brightness curves. Relative: each as a share of where it started, so their shapes compare (the two renderers' absolute brightness differs). As measured: the numbers each recorded.",
    ours: "Our DDGI in the viewer (WebGPU, the platform's own ray tracing), GI only: the light that has bounced, with the direct lamp hidden. Recorded at the viewer's default budget.",
    native:
        "NVIDIA's RTXGI sample (native, Vulkan), with hardware ray tracing, updating every probe every frame. Its picture here is as recorded (the caption says whether it is the full picture or GI only); its curve is always the GI-only picture's brightness.",
    plot:
        "Each side's GI-only brightness (the mean of the picture's red, green and blue) over the motion. Blue: ours; orange: native. The white line is the moment shown. Click or drag here to move to a moment.",
} as const;

/** How long each side's images take to load before the page plays; the progress shows meanwhile. */
type Side = 'ours' | 'native';

interface LoadedSide {
    listed: ListedRecording;
    recording: Recording;
    /** The images, by path, loaded before playing so the sides never wait on a download. */
    images: Map<string, HTMLImageElement>;
}

function withHelp<T extends HTMLElement>(element: T, text: string): T {
    element.dataset.help = text;
    element.title = text;
    return element;
}

async function fetchJson<T>(url: string): Promise<T> {
    const response = await fetch(url);
    if (!response.ok) throw new Error(`${url}: ${response.status} ${response.statusText}`);
    return (await response.json()) as T;
}

async function fetchText(url: string): Promise<string> {
    const response = await fetch(url);
    if (!response.ok) throw new Error(`${url}: ${response.status} ${response.statusText}`);
    return response.text();
}

/** Every recording the dev server lists, with what its recording.json says. */
async function listRecordings(): Promise<ListedRecording[]> {
    const listed = await fetchJson<Array<{ source: 'native' | 'viewer'; name: string; url: string }>>('/__recordings');
    const read = await Promise.all(
        listed.map(async (entry) => {
            try {
                const info = await fetchJson<RecordingInfo>(`${entry.url}recording.json`);
                return { ...entry, scene: info.scene, motion: info.motion, clock: info.clock };
            } catch {
                return null;
            }
        }),
    );
    return read.filter((entry): entry is ListedRecording => entry !== null);
}

async function loadSide(listed: ListedRecording, onImage: () => void): Promise<LoadedSide> {
    const info = await fetchJson<RecordingInfo>(`${listed.url}recording.json`);
    const recording = makeRecording(info, await fetchText(`${listed.url}frames.csv`));
    const images = new Map<string, HTMLImageElement>();
    await Promise.all(
        recording.imageFrames.map(
            (frame) =>
                new Promise<void>((resolve) => {
                    const image = new Image();
                    image.decoding = 'async';
                    image.onload = image.onerror = () => {
                        onImage();
                        resolve();
                    };
                    image.src = `${listed.url}${frame.image}`;
                    images.set(frame.image!, image);
                }),
        ),
    );
    return { listed, recording, images };
}

class ComparePage {
    private readonly root = document.getElementById('app')!;
    private readonly helpBox = h('div', { id: 'help', text: HELP.default });
    private readonly status = h('p', { id: 'status' });
    private readonly motionSelect = withHelp(h('select'), HELP.motion);
    private readonly nativeSelect = withHelp(h('select'), HELP.nativeRun);
    private readonly oursSelect = withHelp(h('select'), HELP.oursRun);
    private readonly playButton = withHelp(h('button', { text: 'Play' }), HELP.play);
    private readonly scrub = withHelp(h('input', { id: 'scrub', type: 'range', min: 0, max: 1, step: 1 / 60, value: 0 }), HELP.scrub);
    private readonly speedSelect = withHelp(h('select'), HELP.speed);
    private readonly scaleSelect = withHelp(h('select'), HELP.scale);
    private readonly timeReadout = h('span', { id: 'time' });
    private readonly plot = withHelp(h('canvas', { id: 'plot' }), HELP.plot);
    private readonly sides: Record<Side, { image: HTMLImageElement; title: HTMLElement; caption: HTMLElement; readout: HTMLElement }>;

    private pairs: RecordingPair[] = [];
    private loaded: Record<Side, LoadedSide> | null = null;
    private time = 0;
    private playing = false;
    private lastTick = 0;
    private loadGeneration = 0;

    constructor() {
        const side = (which: Side, title: string, help: string) => {
            const image = h('img', { alt: `${title}: the picture at this moment` });
            const titleElement = h('h2', { text: title });
            const caption = h('p', { className: 'caption' });
            const readout = h('p', { className: 'caption readout' });
            this.root.append();
            const element = withHelp(h('section', { className: `side ${which}` }, titleElement, h('div', { className: 'frame' }, image), readout, caption), help);
            return { element, parts: { image, title: titleElement, caption, readout } };
        };
        const ours = side('ours', 'DDGI (ours, WebGPU)', HELP.ours);
        const native = side('native', 'RTXGI (native, hardware ray tracing)', HELP.native);
        this.sides = { ours: ours.parts, native: native.parts };

        for (const speed of [1, 0.5, 0.25, 0.1]) this.speedSelect.append(h('option', { value: speed, text: speed === 1 ? 'Real time' : `${speed}x speed` }));
        this.scaleSelect.append(h('option', { value: 'relative', text: 'Relative to its start' }), h('option', { value: 'measured', text: 'As measured' }));

        this.root.append(
            h('h1', { text: 'DDGI and native RTXGI, side by side' }),
            h(
                'p',
                { className: 'note' },
                "The same light motion, recorded on the same clock, played together. Native traces with hardware ray tracing and updates every probe every frame; ours runs in the browser at a 2 ms budget. Matching native is not a requirement: the look is judged by eye (Dennis, 2026-11-05). Both sides are GI-only brightness on the plot.",
            ),
            h(
                'div',
                { className: 'bar' },
                h('label', {}, 'Motion', this.motionSelect),
                h('label', {}, 'Native run', this.nativeSelect),
                h('label', {}, 'Our run', this.oursSelect),
                h('label', {}, 'Curves', this.scaleSelect),
            ),
            h('div', { className: 'sides' }, ours.element, native.element),
            h('div', { className: 'bar' }, this.playButton, this.scrub, this.speedSelect, this.timeReadout),
            this.plot,
            h(
                'div',
                { className: 'legend' },
                h('span', {}, h('span', { className: 'swatch', style: 'background: var(--ours)' }), 'DDGI (ours)'),
                h('span', {}, h('span', { className: 'swatch', style: 'background: var(--native)' }), 'RTXGI (native)'),
                h('span', {}, h('span', { className: 'swatch', style: 'background: #ffffff' }), 'the moment shown'),
            ),
            this.status,
            this.helpBox,
        );

        this.root.addEventListener('mouseover', (event) => {
            const target = (event.target as HTMLElement).closest<HTMLElement>('[data-help]');
            this.helpBox.textContent = target?.dataset.help ?? HELP.default;
        });
        this.motionSelect.addEventListener('change', () => this.choosePair(this.motionSelect.value, null, null));
        this.nativeSelect.addEventListener('change', () => void this.load());
        this.oursSelect.addEventListener('change', () => void this.load());
        this.playButton.addEventListener('click', () => this.setPlaying(!this.playing));
        this.scrub.addEventListener('input', () => this.seek(Number(this.scrub.value)));
        this.scaleSelect.addEventListener('change', () => this.draw());
        window.addEventListener('keydown', (event) => {
            if (event.code === 'Space' && !(event.target instanceof HTMLSelectElement)) {
                event.preventDefault();
                this.setPlaying(!this.playing);
            }
        });
        const plotSeek = (event: PointerEvent) => {
            const box = this.plot.getBoundingClientRect();
            this.seek(((event.clientX - box.left) / box.width) * this.length);
        };
        this.plot.addEventListener('pointerdown', (event) => {
            this.plot.setPointerCapture(event.pointerId);
            plotSeek(event);
        });
        this.plot.addEventListener('pointermove', (event) => {
            if (this.plot.hasPointerCapture(event.pointerId)) plotSeek(event);
        });
        new ResizeObserver(() => this.draw()).observe(this.plot);
        requestAnimationFrame((now) => this.tick(now));
    }

    async start(): Promise<void> {
        this.setStatus('Looking for recordings...');
        try {
            this.pairs = pairRecordings(await listRecordings());
        } catch (error) {
            this.setStatus(`Could not list the recordings: ${error instanceof Error ? error.message : String(error)}. This page needs the viewer's dev server (pnpm dev).`, true);
            return;
        }
        if (this.pairs.length === 0) {
            this.setStatus(
                'No motion has both recordings yet. Native: tools/native/play.ps1 -Record (native/build/recordings). Ours: the viewer-recording GPU page, then node tools/native/save-viewer-recording.ts (bench/recordings/viewer).',
                true,
            );
            return;
        }
        this.motionSelect.replaceChildren(...this.pairs.map((pair) => h('option', { value: pair.key, text: `${pair.scene}: ${pair.motion}` })));
        const link = new URLSearchParams(window.location.search);
        this.time = Math.max(0, Number(link.get('t')) || 0);
        this.choosePair(link.get('motion') ?? this.pairs[0].key, link.get('native'), link.get('ours'));
    }

    private choosePair(key: string, nativeName: string | null, oursName: string | null): void {
        const pair = this.pairs.find((candidate) => candidate.key === key) ?? this.pairs[0];
        this.motionSelect.value = pair.key;
        const options = (recordings: ListedRecording[]) => recordings.map((recording) => h('option', { value: recording.name, text: recording.name }));
        this.nativeSelect.replaceChildren(...options(pair.native));
        this.oursSelect.replaceChildren(...options(pair.ours));
        if (nativeName && pair.native.some((recording) => recording.name === nativeName)) this.nativeSelect.value = nativeName;
        if (oursName && pair.ours.some((recording) => recording.name === oursName)) this.oursSelect.value = oursName;
        void this.load();
    }

    private async load(): Promise<void> {
        const generation = ++this.loadGeneration;
        const pair = this.pairs.find((candidate) => candidate.key === this.motionSelect.value);
        if (!pair) return;
        const native = pair.native.find((recording) => recording.name === this.nativeSelect.value) ?? pair.native[0];
        const ours = pair.ours.find((recording) => recording.name === this.oursSelect.value) ?? pair.ours[0];
        this.setPlaying(false);
        this.loaded = null;
        let loadedImages = 0;
        const progress = () => {
            loadedImages++;
            if (generation === this.loadGeneration && loadedImages % 10 === 0) this.setStatus(`Loading the pictures: ${loadedImages}...`);
        };
        this.setStatus('Loading the recordings...');
        try {
            const [oursSide, nativeSide] = await Promise.all([loadSide(ours, progress), loadSide(native, progress)]);
            if (generation !== this.loadGeneration) return;
            this.loaded = { ours: oursSide, native: nativeSide };
        } catch (error) {
            if (generation === this.loadGeneration) this.setStatus(`Could not load the recordings: ${error instanceof Error ? error.message : String(error)}`, true);
            return;
        }
        for (const which of ['ours', 'native'] as const) this.sides[which].caption.textContent = describe(this.loaded[which].recording.info);
        this.scrub.max = String(this.length);
        this.setStatus(`Loaded ${ours.name} and ${native.name}: ${loadedImages} pictures.`);
        this.writeLink();
        this.seek(Math.min(this.time, this.length));
    }

    /** The longer recording's length: past one's end, it holds its last picture. */
    private get length(): number {
        if (!this.loaded) return 0;
        return Math.max(duration(this.loaded.ours.recording), duration(this.loaded.native.recording));
    }

    private setPlaying(playing: boolean): void {
        this.playing = playing && this.loaded !== null;
        if (this.playing && this.time >= this.length) this.time = 0;
        this.playButton.textContent = this.playing ? 'Pause' : 'Play';
        if (!this.playing) this.writeLink();
    }

    private seek(time: number): void {
        this.time = Math.min(Math.max(0, time), this.length);
        this.show();
        if (!this.playing) this.writeLink();
    }

    private tick(now: number): void {
        const elapsed = this.lastTick > 0 ? (now - this.lastTick) / 1000 : 0;
        this.lastTick = now;
        if (this.playing && this.loaded) {
            this.time += elapsed * Number(this.speedSelect.value);
            if (this.time >= this.length) {
                this.time = this.length;
                this.setPlaying(false);
            }
            this.show();
        }
        requestAnimationFrame((next) => this.tick(next));
    }

    /** Both sides' pictures and readouts at the current time, the scrub bar, and the plot's marker. */
    private show(): void {
        const loaded = this.loaded;
        if (!loaded) return;
        for (const which of ['ours', 'native'] as const) {
            const side = loaded[which];
            const shown = imageAt(side.recording, this.time);
            const parts = this.sides[which];
            const image = shown?.image ? side.images.get(shown.image) : undefined;
            if (image && parts.image.src !== image.src) parts.image.src = image.src;
            const frame = frameAt(side.recording, this.time);
            const first = side.recording.frames[0]?.value ?? 0;
            parts.readout.textContent = frame
                ? `frame ${frame.frame} (picture from frame ${shown?.frame ?? '-'}); GI-only brightness ${frame.value.toPrecision(4)}, ${first > 0 ? `${((frame.value / first) * 100).toFixed(1)}% of its start` : ''}`
                : '';
        }
        this.scrub.value = String(this.time);
        this.timeReadout.textContent = `${this.time.toFixed(2)} s of ${this.length.toFixed(2)} s (frame ${Math.round(this.time * 60)})`;
        this.draw();
    }

    private draw(): void {
        const canvas = this.plot;
        const width = Math.max(1, Math.round(canvas.clientWidth * devicePixelRatio));
        const height = Math.max(1, Math.round(canvas.clientHeight * devicePixelRatio));
        if (canvas.width !== width || canvas.height !== height) {
            canvas.width = width;
            canvas.height = height;
        }
        const context = canvas.getContext('2d');
        if (!context) return;
        context.clearRect(0, 0, width, height);
        const loaded = this.loaded;
        if (!loaded) return;
        const scale = this.scaleSelect.value as CurveScale;
        const curves = (['ours', 'native'] as const).map((which) => ({
            which,
            times: loaded[which].recording.frames.map((frame) => frame.time),
            values: curveValues(loaded[which].recording, scale),
        }));
        const all = curves.flatMap((curve) => curve.values).filter(Number.isFinite);
        let low = Math.min(...all);
        let high = Math.max(...all);
        if (scale === 'relative') {
            low = Math.min(low, 1);
            high = Math.max(high, 1);
        }
        const margin = (high - low) * 0.06 || 0.01;
        low -= margin;
        high += margin;
        const pad = 8 * devicePixelRatio;
        const length = this.length || 1;
        const x = (time: number) => pad + (time / length) * (width - 2 * pad);
        const y = (value: number) => height - pad - ((value - low) / (high - low)) * (height - 2 * pad);

        // Seconds along the bottom, and the start level for relative curves.
        context.fillStyle = '#5d6b7a';
        context.font = `${11 * devicePixelRatio}px system-ui, sans-serif`;
        for (let second = 0; second <= length; second++) {
            context.fillRect(x(second), height - pad, 1, -6 * devicePixelRatio);
            context.fillText(`${second} s`, x(second) + 3, height - pad - 2);
        }
        if (scale === 'relative') {
            context.strokeStyle = '#3a4552';
            context.setLineDash([4, 4]);
            context.beginPath();
            context.moveTo(0, y(1));
            context.lineTo(width, y(1));
            context.stroke();
            context.setLineDash([]);
        }
        const colours = { ours: '#5db0ff', native: '#ff9d3c' };
        context.lineWidth = 1.5 * devicePixelRatio;
        for (const curve of curves) {
            context.strokeStyle = colours[curve.which];
            context.beginPath();
            curve.values.forEach((value, index) => (index === 0 ? context.moveTo(x(curve.times[index]), y(value)) : context.lineTo(x(curve.times[index]), y(value))));
            context.stroke();
        }
        context.lineWidth = 1;
        context.strokeStyle = '#ffffff';
        context.beginPath();
        context.moveTo(x(this.time), 0);
        context.lineTo(x(this.time), height);
        context.stroke();
    }

    private writeLink(): void {
        if (!this.loaded) return;
        const link = new URLSearchParams({ motion: this.motionSelect.value, native: this.loaded.native.listed.name, ours: this.loaded.ours.listed.name });
        if (this.time > 0) link.set('t', this.time.toFixed(2));
        window.history.replaceState(null, '', `?${link.toString()}`);
    }

    private setStatus(text: string, problem = false): void {
        this.status.textContent = text;
        this.status.classList.toggle('problem', problem);
    }
}

/** A recording's caption: where it came from and what its pictures show. */
function describe(info: RecordingInfo): string {
    const view = info.view === 'indirect' ? 'GI only' : 'the full picture (direct light and GI)';
    const parts = [
        `Pictures: ${view}, every ${info.imageEvery} frames, ${info.width} x ${info.height}.`,
        typeof info.probeUpdate === 'string' ? `Probes: ${info.probeUpdate}.` : '',
        info.device ? `GPU: ${info.device}.` : '',
        `${info.frames} frames at ${info.frameRate} a second (${info.clock} clock).`,
    ];
    return parts.filter(Boolean).join(' ');
}

void new ComparePage().start();
