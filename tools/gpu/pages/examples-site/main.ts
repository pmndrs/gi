/**
 * T5.6: the built examples site works as GitHub Pages would serve it.
 *
 * Build it first, with the Pages base:
 *
 *   PAGES_BASE=/gi/ pnpm examples:build
 *
 * The test server serves examples/dist under /gi/ (server.ts,
 * contract.ts EXAMPLES_SITE_BASE), so a link or asset that ignores the base
 * fails to load. Then:
 *
 * 1. **The site:** under 50 MB; every page's scripts, styles and links
 *    resolve inside the base and load; the landing page links every example
 *    page the build made.
 * 2. **Every example, as a visitor opens it:** each page in an iframe, in
 *    turn. It must report GI frames and a non-black GI mean on
 *    `window.ddgiExample` (examples/shared/status.ts; the hello example's
 *    own `window.helloState`, opened with `?view=gi` so its canvas shows GI
 *    alone). The Split RC example reports its completed draws and presents
 *    the indirect-only image for the same check. Any console error, failed
 *    load or WebGPU error fails the page (the runner sees the iframes' logs).
 * 3. **Presets:** 07's panel is switched to low, medium and high in turn;
 *    each must start again and light the probes.
 * 4. **The saved bake:** 08 is opened twice. The first visit warms up and
 *    saves a bake; the second must start from it.
 * 5. **Other settings:** the other two material scenes of 06, and 10's
 *    screen debug views, the same way as 2.
 *
 * The runner's screenshot shows a contact sheet of every example's canvas.
 */

import { reportGpuTest, summarizeAdapter } from '../../client.ts';
import { EXAMPLES_SITE_BASE, EXAMPLES_SITE_FILES, type AdapterSummary } from '../../contract.ts';
import { imageMean, waitForRailControl } from '../rc-rail/frame.ts';

/** The built site may be at most this big (docs/design/m5-release.md). */
const MAX_SITE_BYTES = 50 * 2 ** 20;
/** GI frames an example must have run before it's judged. */
const MIN_GI_FRAMES = 60;
/**
 * GI frames that must run in the second after the check, with the page only
 * rendering: GI keeps updating from three's render (the render-time update)
 * or the page's own frame, not just during `start`'s warm-up.
 */
const MIN_GI_FRAMES_PER_SECOND = 10;
/** The probes' mean irradiance must be above this (luminance); black probes read 0. */
const MIN_GI_MEAN = 1e-3;
/** The canvas's mean luminance (0 to 1, as displayed) must be above this. */
const MIN_CANVAS_LUMINANCE = 0.01;
/** How long one example may take to start and settle: shader compiles, the warm-up (at most 8 s), the frames. */
const EXAMPLE_TIMEOUT_MS = 120_000;
const POLL_MS = 250;
/** The iframe's size: a small window, so the canvas costs little. */
const FRAME_WIDTH = 960;
const FRAME_HEIGHT = 540;
/**
 * Pages opened a second time with other settings: the other two material
 * scenes, and the screen debug views (an override material and an explicit
 * GI update, a path the default view doesn't take).
 */
const EXTRA_PAGES = ['ddgi/06-materials/?scene=glass-wall', 'ddgi/06-materials/?scene=uv-board', 'ddgi/10-debug/?view=gi-only', 'ddgi/10-debug/?view=coverage', 'ddgi/10-debug/?view=probe-states'];
const THUMBNAIL_WIDTH = 300;
const THUMBNAIL_HEIGHT = 169;

interface SiteFile {
    path: string;
    bytes: number;
}

/** What the three examples' shared helper publishes (examples/shared/status.ts). */
interface ExampleStatus {
    frames: number;
    giMean: number | null;
    error: string | null;
}

/** What the plain-WebGPU hello example publishes (examples/webgpu/hello/src/main.ts). */
interface HelloState {
    frames: number;
    error: string | null;
}

interface ExampleResult {
    seconds: number;
    frames: number;
    giMean: number | null;
    canvasLuminance: number;
    note?: string;
}

const siteRoot = new URL(EXAMPLES_SITE_BASE, location.origin).href;
const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

/** Fetches a URL of the site, failing on anything but 200. */
async function fetchOk(url: string): Promise<Response> {
    const response = await fetch(url);
    if (!response.ok) throw new Error(`${url} answered ${response.status}`);
    return response;
}

/**
 * Checks the built files: the size, and that every page's references
 * resolve inside the base and load. Returns the example pages the landing
 * page links, as URLs.
 */
async function checkSite(log: (line: string) => void, data: Record<string, unknown>): Promise<string[]> {
    const files = (await (await fetchOk(EXAMPLES_SITE_FILES)).json()) as SiteFile[];
    if (!files.some((file) => file.path === 'index.html')) {
        throw new Error(`examples/dist has no index.html: build the site first with PAGES_BASE=${EXAMPLES_SITE_BASE} pnpm examples:build`);
    }
    const totalBytes = files.reduce((total, file) => total + file.bytes, 0);
    data.siteFiles = files.length;
    data.siteMegabytes = Number((totalBytes / 2 ** 20).toFixed(2));
    log(`site: ${files.length} files, ${(totalBytes / 2 ** 20).toFixed(1)} MB`);
    if (totalBytes > MAX_SITE_BYTES) throw new Error(`the built site is ${(totalBytes / 2 ** 20).toFixed(1)} MB, over ${MAX_SITE_BYTES / 2 ** 20} MB`);

    const pages = files.filter((file) => file.path.endsWith('.html'));
    const checked = new Set<string>();
    // Redirects from the old flat paths (examples/public/01-hello/ → ddgi/01-hello/), kept so links
    // to the published site from before the regroup still work. Their link is checked like any other.
    const redirects = new Set<string>();
    let landingLinks: string[] = [];
    for (const page of pages) {
        const pageUrl = new URL(page.path, siteRoot).href;
        const html = await (await fetchOk(pageUrl)).text();
        const parsed = new DOMParser().parseFromString(html, 'text/html');
        if (parsed.querySelector('meta[http-equiv="refresh"]')) redirects.add(page.path);
        const references = [
            ...[...parsed.querySelectorAll('script[src]')].map((element) => element.getAttribute('src')!),
            ...[...parsed.querySelectorAll('link[href]')].map((element) => element.getAttribute('href')!),
            ...[...parsed.querySelectorAll('a[href]')].map((element) => element.getAttribute('href')!),
        ];
        for (const reference of references) {
            // Off-site links (GitHub, docs) and inline data aren't the site's to resolve.
            if (/^(https?:|data:|mailto:)/.test(reference)) continue;
            const url = new URL(reference, pageUrl).href;
            if (!url.startsWith(siteRoot)) throw new Error(`${page.path} refers to ${reference}, which resolves outside ${EXAMPLES_SITE_BASE} (${url})`);
            if (checked.has(url)) continue;
            checked.add(url);
            await fetchOk(url);
        }
        if (page.path === 'index.html') {
            landingLinks = [...parsed.querySelectorAll('a.card')].map((element) => new URL(element.getAttribute('href')!, pageUrl).href);
        }
    }
    data.referencesChecked = checked.size;
    data.redirects = redirects.size;
    log(`site: ${pages.length} pages (${redirects.size} redirects from old paths), ${checked.size} references resolve inside ${EXAMPLES_SITE_BASE} and load`);

    // Every example page the build made must be on the landing page.
    const examplePages = pages
        .filter((page) => page.path !== 'index.html' && !redirects.has(page.path))
        .map((page) => new URL(page.path.replace(/index\.html$/, ''), siteRoot).href);
    const unlinked = examplePages.filter((url) => !landingLinks.includes(url));
    if (unlinked.length > 0) throw new Error(`the landing page doesn't link ${unlinked.join(', ')}`);
    return landingLinks;
}

/** Opens `url` in a fresh iframe and waits for its document to load. */
async function openFrame(url: string): Promise<HTMLIFrameElement> {
    const frame = document.createElement('iframe');
    frame.style.cssText = `width:${FRAME_WIDTH}px;height:${FRAME_HEIGHT}px;border:0;display:block;`;
    const loaded = new Promise((resolve) => frame.addEventListener('load', resolve, { once: true }));
    frame.src = url;
    document.body.append(frame);
    await loaded;
    return frame;
}

/** Unloads the iframe's page (its `pagehide` frees the GPU device) and removes it. */
async function closeFrame(frame: HTMLIFrameElement): Promise<void> {
    const unloaded = new Promise((resolve) => frame.addEventListener('load', resolve, { once: true }));
    frame.src = 'about:blank';
    await unloaded;
    frame.remove();
}

/**
 * The canvas's mean luminance and a thumbnail, read inside the example's own
 * animation frame. A WebGPU canvas holds its picture only until the task
 * that drew it ends (it is presented then), so the copy must happen in that
 * task. A callback registered now, from outside, runs in the iframe's next
 * animation frame after the example's own (registered during the frame
 * before), in the same task.
 */
function captureCanvas(frameWindow: Window, canvas: HTMLCanvasElement): Promise<{ luminance: number; thumbnail: string }> {
    return new Promise((resolve) => {
        frameWindow.requestAnimationFrame(() => {
            const copy = document.createElement('canvas');
            copy.width = THUMBNAIL_WIDTH;
            copy.height = THUMBNAIL_HEIGHT;
            const context = copy.getContext('2d', { willReadFrequently: true })!;
            context.drawImage(canvas, 0, 0, THUMBNAIL_WIDTH, THUMBNAIL_HEIGHT);
            const pixels = context.getImageData(0, 0, THUMBNAIL_WIDTH, THUMBNAIL_HEIGHT).data;
            let sum = 0;
            for (let index = 0; index < pixels.length; index += 4) sum += 0.2126 * pixels[index] + 0.7152 * pixels[index + 1] + 0.0722 * pixels[index + 2];
            resolve({ luminance: sum / (pixels.length / 4) / 255, thumbnail: copy.toDataURL('image/jpeg', 0.85) });
        });
    });
}

/** Waits until `ready` says the example has settled, failing on its own error or a timeout. */
async function waitFor<T extends { error: string | null }>(name: string, read: () => T | undefined, ready: (state: T) => boolean): Promise<T> {
    const deadline = performance.now() + EXAMPLE_TIMEOUT_MS;
    for (;;) {
        const state = read();
        if (state?.error) throw new Error(`${name}: ${state.error}`);
        if (state && ready(state)) return state;
        if (performance.now() > deadline) throw new Error(`${name}: not ready after ${EXAMPLE_TIMEOUT_MS / 1000} s (last state ${JSON.stringify(state ?? null)})`);
        await sleep(POLL_MS);
    }
}

/** Waits until the page's element with `id` says `text`. */
async function waitForText(name: string, frame: HTMLIFrameElement, id: string, text: string): Promise<string> {
    const deadline = performance.now() + EXAMPLE_TIMEOUT_MS;
    for (;;) {
        const shown = frame.contentDocument?.getElementById(id)?.textContent ?? '';
        if (shown.includes(text)) return shown;
        const error = (frame.contentWindow as (Window & { ddgiExample?: ExampleStatus }) | null)?.ddgiExample?.error;
        if (error) throw new Error(`${name}: ${error}`);
        if (performance.now() > deadline) throw new Error(`${name}: #${id} never said "${text}" (it says "${shown}")`);
        await sleep(POLL_MS);
    }
}

interface RunOptions {
    /** Text an element of the page must come to show (08's bake state). */
    readonly expectText?: { id: string; text: string };
    /** What a visitor does once the page has settled; returns a note for the log. */
    readonly act?: (name: string, frame: HTMLIFrameElement) => Promise<string>;
}

/**
 * 07: switches the preset in its panel, as a visitor does (the select's
 * change event), to each preset in turn, waiting for GI to start again each
 * time. The stats line says `preset <name>` once the new start finished.
 */
async function switchPresets(name: string, frame: HTMLIFrameElement): Promise<string> {
    const select = frame.contentDocument?.querySelector('.lil-gui select');
    if (!(select instanceof (frame.contentWindow as Window & typeof globalThis).HTMLSelectElement)) throw new Error(`${name}: no preset select in the panel`);
    const notes: string[] = [];
    for (const preset of ['low', 'medium', 'high']) {
        select.value = preset;
        select.dispatchEvent(new Event('change'));
        const shown = await waitForText(name, frame, 'stats', `preset ${preset}`);
        notes.push(shown.split(' · ').slice(0, 3).join(', '));
    }
    // A few seconds on the last one, so the GI mean and the canvas come from it.
    await sleep(2500);
    return notes.join(' | ');
}

/** Opens one example, waits for it to settle, and checks its GI and its canvas. */
async function runExample(url: string, thumbnails: Array<{ name: string; thumbnail: string }>, options: RunOptions = {}): Promise<ExampleResult> {
    const { expectText, act } = options;
    const name = url.slice(siteRoot.length) || 'index';
    const isHello = name.startsWith('webgpu/hello');
    const isRail = name.startsWith('cascades/11-radiance-cascades/');
    const started = performance.now();
    const frame = await openFrame(isHello ? `${url}?view=gi` : isRail ? `${url}?verify=1` : url);
    try {
        if (isRail) {
            const control = await waitForRailControl(frame);
            const capture = await control.render({ frames: MIN_GI_FRAMES, view: 'indirect', cue: 'blue', camera: 'receiver', advance: false });
            const canvasLuminance = imageMean(capture).luma;
            if (capture.renderedFrames < MIN_GI_FRAMES || !(canvasLuminance > MIN_CANVAS_LUMINANCE)) {
                throw new Error(`${name}: ${capture.renderedFrames} RC frames, indirect canvas mean ${canvasLuminance}`);
            }
            const copy = document.createElement('canvas');
            copy.width = capture.width;
            copy.height = capture.height;
            const context = copy.getContext('2d')!;
            const pixels = context.createImageData(copy.width, copy.height);
            pixels.data.set(capture.pixels);
            context.putImageData(pixels, 0, 0);
            thumbnails.push({ name, thumbnail: copy.toDataURL('image/png') });
            return { seconds: Number(((performance.now() - started) / 1000).toFixed(1)), frames: capture.renderedFrames, giMean: null, canvasLuminance, note: 'Split RC indirect-only' };
        }
        const frameWindow = frame.contentWindow as Window & { ddgiExample?: ExampleStatus; helloState?: HelloState };
        let frames: number;
        let giMean: number | null = null;
        if (isHello) {
            const state = await waitFor(name, () => frameWindow.helloState, (hello) => hello.frames >= MIN_GI_FRAMES);
            frames = state.frames;
        } else {
            const state = await waitFor(name, () => frameWindow.ddgiExample, (status) => status.frames >= MIN_GI_FRAMES && status.giMean !== null);
            frames = state.frames;
            giMean = state.giMean;
            if (!(giMean !== null && giMean > MIN_GI_MEAN)) throw new Error(`${name}: the probes hold no light (mean irradiance ${giMean})`);
        }
        let note = expectText ? await waitForText(name, frame, expectText.id, expectText.text) : undefined;
        if (act) {
            note = await act(name, frame);
            const status = frameWindow.ddgiExample;
            if (status?.error) throw new Error(`${name}: ${status.error}`);
            giMean = status?.giMean ?? null;
            frames = status?.frames ?? frames;
            if (!(giMean !== null && giMean > MIN_GI_MEAN)) throw new Error(`${name}: after "${note}", the probes hold no light (mean irradiance ${giMean})`);
        }
        const canvas = frame.contentDocument?.querySelector('canvas');
        if (!canvas) throw new Error(`${name}: no canvas`);
        const capture = await captureCanvas(frameWindow, canvas);
        thumbnails.push({ name, thumbnail: capture.thumbnail });
        if (!(capture.luminance > MIN_CANVAS_LUMINANCE)) throw new Error(`${name}: the canvas is black (mean luminance ${capture.luminance.toFixed(4)})`);
        const framesBefore = (isHello ? frameWindow.helloState : frameWindow.ddgiExample)?.frames ?? 0;
        await sleep(1000);
        const framesAfter = (isHello ? frameWindow.helloState : frameWindow.ddgiExample)?.frames ?? 0;
        if (framesAfter - framesBefore < MIN_GI_FRAMES_PER_SECOND) throw new Error(`${name}: GI ran only ${framesAfter - framesBefore} frames in a second of rendering`);
        const result: ExampleResult = { seconds: Number(((performance.now() - started) / 1000).toFixed(1)), frames, giMean, canvasLuminance: Number(capture.luminance.toFixed(4)) };
        if (note) result.note = note;
        return result;
    } finally {
        await closeFrame(frame);
    }
}

/** The contact sheet the runner's screenshot shows: every example's canvas as it was checked. */
function showContactSheet(thumbnails: Array<{ name: string; thumbnail: string }>): void {
    const sheet = document.createElement('div');
    sheet.style.cssText = 'display:flex;flex-wrap:wrap;gap:4px;padding:4px;background:#000;font:11px system-ui;color:#ddd;';
    for (const { name, thumbnail } of thumbnails) {
        const figure = document.createElement('figure');
        figure.style.cssText = 'margin:0;';
        const image = document.createElement('img');
        image.src = thumbnail;
        image.width = THUMBNAIL_WIDTH;
        image.height = THUMBNAIL_HEIGHT;
        const caption = document.createElement('figcaption');
        caption.textContent = name;
        figure.append(image, caption);
        sheet.append(figure);
    }
    document.body.prepend(sheet);
}

const startTime = performance.now();
const logLines: string[] = [];
const log = (line: string) => logLines.push(line);
const data: Record<string, unknown> = {};
let adapter: AdapterSummary | null = null;
const thumbnails: Array<{ name: string; thumbnail: string }> = [];
try {
    const gpuAdapter = await navigator.gpu?.requestAdapter();
    if (gpuAdapter) adapter = summarizeAdapter(gpuAdapter);

    const examples = await checkSite(log, data);
    const results: Record<string, ExampleResult> = {};
    for (const url of examples) {
        const name = url.slice(siteRoot.length);
        const isBake = name.startsWith('ddgi/08-');
        // 08's first visit saves a bake; the second must start from it.
        results[name] = await runExample(url, thumbnails, {
            expectText: isBake ? { id: 'bake-state', text: 'Reload to start from it' } : undefined,
            act: name.startsWith('ddgi/07-') ? switchPresets : undefined,
        });
        if (isBake) results[`${name} (reloaded)`] = await runExample(url, thumbnails, { expectText: { id: 'bake-state', text: 'from the bake saved' } });
    }
    for (const page of EXTRA_PAGES) results[page] = await runExample(new URL(page, siteRoot).href, thumbnails);
    for (const [name, result] of Object.entries(results)) {
        log(`${name}: ${result.frames} GI frames in ${result.seconds} s, GI mean ${result.giMean?.toFixed(4) ?? '(canvas only)'}, canvas ${result.canvasLuminance}${result.note ? `; "${result.note}"` : ''}`);
    }
    data.examples = results;
    showContactSheet(thumbnails);
    reportGpuTest({ status: 'pass', data, adapter, startTime, logLines });
} catch (error) {
    showContactSheet(thumbnails);
    const message = error instanceof Error ? error.message : String(error);
    reportGpuTest({ status: 'fail', message, data, adapter, startTime, logLines });
}
