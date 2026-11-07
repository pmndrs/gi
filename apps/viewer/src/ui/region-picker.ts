/**
 * The response meter's region (T2.8): "Pick a region" lays a see-through
 * layer over the picture; dragging on it draws a rectangle, and letting go
 * chooses it. The chosen rectangle stays outlined on the picture while the
 * meter section is in use. Rectangles are kept as shares of the picture
 * (0 to 1), so they survive the window being resized, and turned into
 * drawing-buffer pixels when the meter reads them.
 */

import { h } from './dom';

/** A rectangle as shares of the picture, from the top left. */
export interface RegionShare {
    left: number;
    top: number;
    width: number;
    height: number;
}

/** Rectangles smaller than this many CSS pixels across are taken as a click, not a pick. */
const MIN_DRAG_PIXELS = 6;

export class RegionPicker {
    private readonly layer = h('div', { className: 'region-layer' });
    private readonly outline = h('div', { className: 'region-outline' });
    private dragFrom: { x: number; y: number } | null = null;
    private regionValue: RegionShare | null = null;
    private onPicked: ((region: RegionShare | null) => void) | null = null;

    constructor(private readonly canvas: HTMLCanvasElement) {
        document.body.append(this.layer, this.outline);
        this.layer.hidden = true;
        this.outline.hidden = true;
        this.layer.addEventListener('pointerdown', (event) => {
            this.dragFrom = { x: event.clientX, y: event.clientY };
            this.layer.setPointerCapture(event.pointerId);
            this.drawOutline(event.clientX, event.clientY);
        });
        this.layer.addEventListener('pointermove', (event) => {
            if (this.dragFrom) this.drawOutline(event.clientX, event.clientY);
        });
        this.layer.addEventListener('pointerup', (event) => {
            const from = this.dragFrom;
            this.dragFrom = null;
            this.layer.hidden = true;
            if (!from) return;
            const box = this.canvas.getBoundingClientRect();
            const left = Math.max(box.left, Math.min(from.x, event.clientX));
            const right = Math.min(box.right, Math.max(from.x, event.clientX));
            const top = Math.max(box.top, Math.min(from.y, event.clientY));
            const bottom = Math.min(box.bottom, Math.max(from.y, event.clientY));
            const picked = right - left >= MIN_DRAG_PIXELS && bottom - top >= MIN_DRAG_PIXELS;
            this.regionValue = picked ? { left: (left - box.left) / box.width, top: (top - box.top) / box.height, width: (right - left) / box.width, height: (bottom - top) / box.height } : this.regionValue;
            this.refresh();
            this.onPicked?.(picked ? this.regionValue : null);
        });
    }

    get region(): RegionShare | null {
        return this.regionValue;
    }

    /** Lets the next drag on the picture choose the region; `done` gets it (null when the drag was too small). */
    pick(done: (region: RegionShare | null) => void): void {
        this.onPicked = done;
        const box = this.canvas.getBoundingClientRect();
        Object.assign(this.layer.style, { left: `${box.left}px`, top: `${box.top}px`, width: `${box.width}px`, height: `${box.height}px` });
        this.layer.hidden = false;
    }

    /** Back to the whole picture. */
    clear(): void {
        this.regionValue = null;
        this.refresh();
    }

    /** The region in drawing-buffer pixels, or null for the whole picture. */
    pixels(bufferWidth: number, bufferHeight: number): { x: number; y: number; width: number; height: number } | null {
        const region = this.regionValue;
        if (!region) return null;
        return {
            x: Math.round(region.left * bufferWidth),
            y: Math.round(region.top * bufferHeight),
            width: Math.max(1, Math.round(region.width * bufferWidth)),
            height: Math.max(1, Math.round(region.height * bufferHeight)),
        };
    }

    /** Redraws the outline where the region is (after a resize, say). */
    refresh(): void {
        const region = this.regionValue;
        this.outline.hidden = region === null;
        if (!region) return;
        const box = this.canvas.getBoundingClientRect();
        Object.assign(this.outline.style, {
            left: `${box.left + region.left * box.width}px`,
            top: `${box.top + region.top * box.height}px`,
            width: `${region.width * box.width}px`,
            height: `${region.height * box.height}px`,
        });
    }

    private drawOutline(x: number, y: number): void {
        const from = this.dragFrom;
        if (!from) return;
        this.outline.hidden = false;
        Object.assign(this.outline.style, {
            left: `${Math.min(from.x, x)}px`,
            top: `${Math.min(from.y, y)}px`,
            width: `${Math.abs(x - from.x)}px`,
            height: `${Math.abs(y - from.y)}px`,
        });
    }
}
