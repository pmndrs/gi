/**
 * How much each pixel changed since the previous frame: the eye check for
 * DDGI's flicker on a still view.
 *
 * With the camera and the scene still, the direct light and shadows draw the
 * same every frame, so whatever changes is the GI: probes being updated. Each
 * frame, before DDGI's scene is drawn again, its last picture is copied
 * aside; the output then compares the new picture with it, per pixel:
 *
 *   change = mean over rgb of |now − before|  /  the pixel's own brightness
 *
 * (the brightness floored at a very dark grey, so black areas don't divide by
 * nothing). The flicker pages (`tools/gpu/pages/ddgi-flicker`) divide by the
 * whole image's mean instead, so their numbers and this one agree on evenly
 * lit areas and this one reads higher in shadow, which is how the eye judges.
 *
 * Shown as colour over a dimmed grey copy of the picture: orange where the
 * pixel got brighter, blue where it got darker. It is painted in display
 * space, after tone mapping and the output colour encoding (`overlay`, called
 * by the viewer's pipeline), so its colours are exactly what reaches the
 * screen whatever the tone mapper and exposure: the colour's share is
 * change / full scale, so full colour means a change of at least the chosen
 * scale (`fullScalePercent`) and half colour exactly half of it.
 *
 * The probe spheres keep their own colours: they mark their pixels with
 * alpha 0 in the scene target (`probe-spheres.ts`; DDGI's pass clears alpha
 * to 1), and those pixels show the picture unchanged.
 *
 * The comparison is only meaningful from the same camera: while the camera
 * moves (or right after the view was resized or switched on) there is no
 * colour, only the grey picture.
 *
 * Precision: the scene target is half float, whose steps are 0.05-0.1% of a
 * value, so a pixel can't change by less than that. At the 0.25% scale a
 * single step shows at 20-40% colour: that is the speckle on otherwise still
 * pixels.
 */

import { HalfFloatType, Matrix4, type PerspectiveCamera } from 'three';
import { abs, dot, float, max, mix, select, texture, toneMappingExposure, uniform, vec3, vec4 } from 'three/tsl';
import { RenderTarget, type Node, type WebGPURenderer } from 'three/webgpu';

/** Display colours (what the screen shows, 0 to 1), chosen to read apart from each other and from grey. */
const BRIGHTER_COLOR = [1.0, 0.6, 0.1] as const;
const DARKER_COLOR = [0.25, 0.6, 1.0] as const;

/** The grey copy of the picture underneath, as a share of its display brightness. */
const DIMMED_PICTURE = 0.4;

/** Rec. 709 luma weights, for the grey copy of the display picture. */
const LUMA_WEIGHTS = [0.2126, 0.7152, 0.0722] as const;

/**
 * Brightness below which a pixel counts as this dark when dividing, in display units (after
 * exposure): about 0.2% of white. Keeps near-black pixels from showing rounding as change.
 */
const DARKEST_DIVISOR = 0.002;

/** The renderer's exposure as a float node (its type package declares it as a plain reference node). */
const exposure = toneMappingExposure as unknown as Node<'float'>;

/**
 * Matrix entries closer than this count as the same camera (relative to the entry's size;
 * absolute for entries under 1). A
 * still camera's matrices are rebuilt every frame from its position and rotation and differ in
 * the last bits (-2.7800000000000002 against -2.78), so an exact comparison never matches; a
 * real move of even a tenth of a millimetre is far larger.
 */
const SAME_CAMERA_TOLERANCE = 1e-9;

function matricesMatch(a: Matrix4, b: Matrix4): boolean {
    return a.elements.every((value, index) => Math.abs(value - b.elements[index]) <= SAME_CAMERA_TOLERANCE * Math.max(1, Math.abs(value)));
}

function rgbNode(color: readonly [number, number, number]): Node<'vec3'> {
    return vec3(color[0], color[1], color[2]) as unknown as Node<'vec3'>;
}

export class FrameChangeView {
    /** The previous frame's picture: same size and format as the scene target, no multisampling (it holds the resolved picture). */
    private readonly previous = new RenderTarget(1, 1, { type: HalfFloatType });
    /** 1 when `previous` holds the last frame from the same camera, else 0 (no colour). */
    private readonly comparable = uniform(0);
    private readonly fullScale = uniform(0.02);
    private readonly lastView = new Matrix4();
    private readonly lastProjection = new Matrix4();
    /** Whether the scene target holds a picture drawn by a previous frame. */
    private targetDrawn = false;

    /** Full colour at this change per frame, in percent. Free: it's a uniform. */
    set fullScalePercent(percent: number) {
        this.fullScale.value = percent / 100;
    }

    /**
     * The change view over `display`, the picture as the screen shows it (after tone mapping and
     * output colour). `current` is the scene target the picture came from (linear HDR), which the
     * change is measured on.
     */
    overlay(display: Node, current: RenderTarget): Node {
        const scene = texture(current.texture);
        const now = scene.rgb;
        const before = texture(this.previous.texture).rgb;
        const brightnessNow = now.r.add(now.g).add(now.b).div(3);
        const brightnessBefore = before.r.add(before.g).add(before.b).div(3);
        const difference = abs(now.sub(before));
        const meanDifference = difference.r.add(difference.g).add(difference.b).div(3);
        // The floor is in display units, so divide it by the exposure to compare it with scene values.
        const divisor = max(brightnessBefore, float(DARKEST_DIVISOR).div(exposure));
        const change = meanDifference.div(divisor);
        // Linear in display space: full colour from the scale up, half colour at half of it.
        const share = change.div(this.fullScale).clamp(0, 1).mul(this.comparable);
        const direction = select(brightnessNow.greaterThanEqual(brightnessBefore), rgbNode(BRIGHTER_COLOR), rgbNode(DARKER_COLOR));
        const shown = (display as Node<'vec4'>).rgb;
        const grey = vec3(dot(shown, rgbNode(LUMA_WEIGHTS)).mul(DIMMED_PICTURE));
        const changeView = mix(grey, direction, share);
        // Probe spheres wrote alpha 0 (edges in between, from multisampling): show them as they are.
        const keepPicture = float(1).sub(scene.a).clamp(0, 1);
        return vec4(mix(changeView, shown, keepPicture), float(1)) as unknown as Node;
    }

    /**
     * Before DDGI's scene is drawn this frame: keeps the last frame's picture for the comparison,
     * and decides whether the comparison means anything (the same camera and size as last frame).
     */
    capturePrevious(renderer: WebGPURenderer, sceneTarget: RenderTarget, camera: PerspectiveCamera): void {
        const sameSize = this.previous.width === sceneTarget.width && this.previous.height === sceneTarget.height;
        const sameCamera = matricesMatch(this.lastView, camera.matrixWorldInverse) && matricesMatch(this.lastProjection, camera.projectionMatrix);
        if (!sameSize) this.previous.setSize(sceneTarget.width, sceneTarget.height);
        else if (this.targetDrawn) renderer.copyTextureToTexture(sceneTarget.texture, this.previous.texture);
        this.comparable.value = sameSize && sameCamera && this.targetDrawn ? 1 : 0;
        this.lastView.copy(camera.matrixWorldInverse);
        this.lastProjection.copy(camera.projectionMatrix);
        this.targetDrawn = true;
    }

    /** The view was off for a while: the scene target's picture is no longer the last frame's. */
    forget(): void {
        this.targetDrawn = false;
    }

    dispose(): void {
        this.previous.dispose();
    }
}
