/** The built demo exposes this interface only when opened with ?verify=1. */
export interface RailCapture {
    width: number;
    height: number;
    pixels: Uint8ClampedArray;
    frame: number;
    renderedFrames: number;
    computeMs: number | null;
    measurement?: {
        frameIntervalsMs: number[];
        gpuComputeMs: number[];
        timestampSupported: boolean;
        renderWidth: number;
        renderHeight: number;
        stride: number;
        probeBytes: number;
        hitScratchBytes: number;
    };
}

export interface RailControl {
    render(options: {
        frames: number;
        view?: 'lit' | 'indirect' | 'direct';
        camera?: 'rail' | 'receiver';
        cue?: 'sequence' | 'amber' | 'blue' | 'off';
        frame?: number;
        advance?: boolean;
        measure?: boolean;
    }): Promise<RailCapture>;
}

/** A cold first visit can compile the traversal, hash, merge and gather pipelines. */
export async function waitForRailControl(frame: HTMLIFrameElement): Promise<RailControl> {
    const deadline = performance.now() + 120_000;
    while (performance.now() < deadline) {
        const control = (frame.contentWindow as (Window & { __rcRailVerify?: RailControl }) | null)?.__rcRailVerify;
        if (control) return control;
        await new Promise((resolve) => setTimeout(resolve, 100));
    }
    throw new Error(`RC demo did not initialize: ${frame.contentDocument?.getElementById('status')?.textContent ?? 'no status'}`);
}

export function imageMean(image: RailCapture, receiver = false): { red: number; green: number; blue: number; luma: number } {
    const { width, height, pixels } = image;
    if (width <= 0 || height <= 0 || pixels.length !== width * height * 4) throw new Error('RC canvas capture has invalid dimensions');
    // The fixed receiver is the center sculpture. Exclude the edge of the view
    // so a visible side-chapel emissive cannot satisfy a lighting-change check.
    const xInset = receiver ? Math.floor(width / 3) : 0;
    const yInset = receiver ? Math.floor(height / 3) : 0;
    let red = 0, green = 0, blue = 0;
    for (let y = yInset; y < height - yInset; y++) {
        for (let x = xInset; x < width - xInset; x++) {
            const offset = (y * width + x) * 4;
            red += pixels[offset];
            green += pixels[offset + 1];
            blue += pixels[offset + 2];
        }
    }
    const scale = 1 / ((width - xInset * 2) * (height - yInset * 2) * 255);
    return { red: red * scale, green: green * scale, blue: blue * scale, luma: (0.2126 * red + 0.7152 * green + 0.0722 * blue) * scale };
}
