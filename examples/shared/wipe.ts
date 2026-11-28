/**
 * A split wipe between two pictures of one view: one technique left of a
 * line, the other right of it, and the line dragged by the mouse or a
 * finger (docs/plans/examples.md, C2). From the viewer's wipe
 * (apps/viewer/src/core/viewer-core.ts and viewer-app.ts): the two pictures
 * are chosen per pixel by `screenUV.x` against a uniform, so moving the line
 * costs no shader work, and a press near the line is taken before the
 * camera controls see it.
 *
 * Two ways to use it:
 *
 * - **Nodes** (three's `RenderPipeline`): `pipeline.outputNode = wipe.node(left, right)`,
 *   each side a technique's picture, for example two `ddgi()` composites.
 *   The pipeline's tone mapping then applies to both alike.
 * - **Scissor** (two `renderer.render` calls, as 09 does):
 *   `wipe.splitPixels(width)` is where the left draw ends.
 *
 * `wipe.difference = true` shows |left − right| × `differenceScale` instead,
 * over the whole view.
 */

import { abs, screenUV, select, uniform, vec4 } from 'three/tsl';
import type { Node, UniformNode } from 'three/webgpu';
import './wipe.css';

export interface WipeOptions {
    /** Where the line starts, 0 (left edge) to 1 (right edge). Default 0.5. */
    readonly position?: number;
    /** The two sides' names, shown at the top beside the line. */
    readonly labels?: readonly [string, string];
    /** Camera controls to pause while the line is dragged (three's `OrbitControls` and the like). */
    readonly controls?: { enabled: boolean };
}

export interface Wipe {
    /** Where the line is, 0 (left edge) to 1 (right edge). */
    position: number;
    /** The line's position as a TSL uniform, for pages that build their own node. */
    readonly split: UniformNode<'float', number>;
    /** Show |left − right| × `differenceScale` instead of the wipe. */
    difference: boolean;
    /** How much the difference view brightens the difference. Default 4. */
    differenceScale: number;
    /** The picture: `left` left of the line, `right` right of it (or their difference). */
    node(left: Node, right: Node): Node;
    /** Where the line is in pixels, for a page that draws the sides with a scissor. */
    splitPixels(width: number): number;
    /** Renames the sides. */
    setLabels(left: string, right: string): void;
    /** Removes the line and its listeners. */
    dispose(): void;
}

/** Closest the line may come to an edge, so it can always be grabbed again. */
const EDGE = 0.02;
/** How near the line, in CSS pixels, a press grabs it: wider for a finger than for a mouse. */
const GRAB_MOUSE_PX = 8;
const GRAB_TOUCH_PX = 22;
/** One arrow-key press moves the line this far. */
const KEY_STEP = 0.02;

/** Keeps a line position inside the view, `EDGE` from either side. */
export function clampSplit(position: number): number {
    if (!Number.isFinite(position)) return 0.5;
    return Math.min(1 - EDGE, Math.max(EDGE, position));
}

/** Makes a wipe on `canvas`'s view. */
export function createWipe(canvas: HTMLCanvasElement, options: WipeOptions = {}): Wipe {
    const split = uniform(clampSplit(options.position ?? 0.5));
    const differenceOn = uniform(0);
    const scale = uniform(4);

    const line = document.createElement('div');
    line.className = 'wipe-line';
    const handle = document.createElement('div');
    handle.className = 'wipe-handle';
    handle.tabIndex = 0;
    handle.setAttribute('role', 'slider');
    handle.setAttribute('aria-label', 'Wipe position');
    handle.setAttribute('aria-valuemin', '0');
    handle.setAttribute('aria-valuemax', '100');
    const leftLabel = document.createElement('span');
    leftLabel.className = 'wipe-label left';
    const rightLabel = document.createElement('span');
    rightLabel.className = 'wipe-label right';
    line.append(handle, leftLabel, rightLabel);
    document.body.append(line);

    const place = (): void => {
        const box = canvas.getBoundingClientRect();
        line.style.left = `${box.left + box.width * split.value}px`;
        line.style.top = `${box.top}px`;
        line.style.height = `${box.height}px`;
        handle.setAttribute('aria-valuenow', String(Math.round(split.value * 100)));
    };
    const setPosition = (position: number): void => {
        split.value = clampSplit(position);
        place();
    };
    const setLabels = (left: string, right: string): void => {
        leftLabel.textContent = left;
        rightLabel.textContent = right;
        leftLabel.hidden = left === '';
        rightLabel.hidden = right === '';
    };
    setLabels(options.labels?.[0] ?? '', options.labels?.[1] ?? '');
    place();

    let dragging: number | null = null;
    const nearLine = (event: PointerEvent): boolean => {
        const box = canvas.getBoundingClientRect();
        const grab = event.pointerType === 'mouse' ? GRAB_MOUSE_PX : GRAB_TOUCH_PX;
        return Math.abs(event.clientX - (box.left + box.width * split.value)) <= grab;
    };
    const startDrag = (event: PointerEvent): void => {
        dragging = event.pointerId;
        if (options.controls) options.controls.enabled = false;
        line.classList.add('dragging');
        (event.target as Element).setPointerCapture?.(event.pointerId);
        event.preventDefault();
        event.stopImmediatePropagation();
    };
    // Grab the line before the camera controls see the press: capture phase, on the window.
    const onPointerDown = (event: PointerEvent): void => {
        if (event.target === handle || (event.target === canvas && nearLine(event))) startDrag(event);
    };
    const onPointerMove = (event: PointerEvent): void => {
        if (dragging === event.pointerId) {
            const box = canvas.getBoundingClientRect();
            setPosition((event.clientX - box.left) / box.width);
        } else if (event.target === canvas) {
            canvas.style.cursor = nearLine(event) ? 'col-resize' : '';
        }
    };
    const onPointerUp = (event: PointerEvent): void => {
        if (dragging !== event.pointerId) return;
        dragging = null;
        if (options.controls) options.controls.enabled = true;
        line.classList.remove('dragging');
    };
    const onKey = (event: KeyboardEvent): void => {
        if (event.key === 'ArrowLeft') setPosition(split.value - KEY_STEP);
        else if (event.key === 'ArrowRight') setPosition(split.value + KEY_STEP);
        else return;
        event.preventDefault();
    };
    window.addEventListener('pointerdown', onPointerDown, { capture: true });
    window.addEventListener('pointermove', onPointerMove);
    window.addEventListener('pointerup', onPointerUp);
    window.addEventListener('pointercancel', onPointerUp);
    window.addEventListener('resize', place);
    handle.addEventListener('keydown', onKey);

    return {
        get position() {
            return split.value;
        },
        set position(value: number) {
            setPosition(value);
        },
        split,
        get difference() {
            return differenceOn.value > 0;
        },
        set difference(on: boolean) {
            differenceOn.value = on ? 1 : 0;
            line.hidden = on;
        },
        get differenceScale() {
            return scale.value;
        },
        set differenceScale(value: number) {
            scale.value = value;
        },
        node(left: Node, right: Node): Node {
            const side = select(screenUV.x.lessThan(split), left, right);
            return select(differenceOn.greaterThan(0), abs(vec4(left as Node<'vec4'>).sub(right as Node<'vec4'>)).mul(scale), side);
        },
        splitPixels(width: number): number {
            return Math.round(width * split.value);
        },
        setLabels,
        dispose(): void {
            window.removeEventListener('pointerdown', onPointerDown, { capture: true });
            window.removeEventListener('pointermove', onPointerMove);
            window.removeEventListener('pointerup', onPointerUp);
            window.removeEventListener('pointercancel', onPointerUp);
            window.removeEventListener('resize', place);
            line.remove();
        },
    };
}
