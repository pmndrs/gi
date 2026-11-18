/**
 * What an example is doing, as one small object on `window.ddgiExample`:
 * the same numbers the stats line shows, for the browser console and for
 * the repo's automated check of the built site
 * (tools/gpu/pages/examples-site). Reading it changes nothing.
 *
 *   > ddgiExample
 *   { frames: 1234, giMean: 0.42, error: null }
 */

export interface ExampleStatus {
    /** GI frames run since GI started (the platform's frame count). */
    frames: number;
    /**
     * The mean irradiance of the probes that are in use, as luminance in
     * three's light units (a white matte surface lit by it shows
     * `giMean / π`). Measured every few seconds by the stats line; null until
     * the first measurement. Above 0 means the probes hold light.
     */
    giMean: number | null;
    /** The first error the page met (a thrown error, a WebGPU error), or null. */
    error: string | null;
}

declare global {
    interface Window {
        ddgiExample: ExampleStatus;
    }
}

export const status: ExampleStatus = { frames: 0, giMean: null, error: null };
window.ddgiExample = status;

/** Shows a message over the page: why the example can't run. */
export function showFatal(message: string): void {
    let element = document.getElementById('fatal');
    if (!element) {
        element = document.createElement('div');
        element.id = 'fatal';
        element.style.cssText =
            'position:fixed;inset:0;display:grid;place-items:center;padding:24px;z-index:50;' +
            'color:#fda4af;background:rgba(11,13,16,0.85);font:14px/1.5 system-ui,sans-serif;text-align:center;white-space:pre-wrap;';
        document.body.append(element);
    }
    element.textContent = message;
}

/** Records the first error and shows it. */
export function reportError(error: unknown): void {
    const message = error instanceof Error ? error.message : String(error);
    if (status.error === null) status.error = message;
    showFatal(message);
}

// An example's top-level code is one module: a throw there (no WebGPU, a failed start) lands here.
window.addEventListener('error', (event) => reportError(event.error ?? event.message));
window.addEventListener('unhandledrejection', (event) => reportError(event.reason));
