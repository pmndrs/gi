/**
 * The Cycles reference image of the current camera view, as a "technique",
 * so the split-screen wipe can put it next to DDGI (or anything else).
 *
 * It draws a still image, not the scene: Blender Cycles' converged path
 * trace from `tools/reference/cycles` (diffuse-only materials, 32 bounces, no
 * denoising, the same shared sky file). It is linear HDR light, like every
 * other technique's output, so it goes through the same exposure and tone
 * mapping in the viewer's render pipeline and the two sides are comparable.
 *
 * Lining it up: the reference was rendered from the bookmark's position and
 * target with its vertical field of view, at its own size (Sponza 640x360,
 * Cornell 512x512). Here it is scaled about the screen centre so its
 * vertical field of view matches the camera's and its width follows its own
 * aspect ratio: it lines up exactly while the camera sits at that bookmark,
 * even if the window is shaped differently or the field of view slider
 * moved. Outside the image (a wider window than the render) the picture is
 * dark grey. Once the camera moves away from the bookmark the image no
 * longer matches the view, and the viewer says so.
 *
 * "GI only" shows Cycles' indirect image: light that bounced at least once
 * before reaching the visible surface, times its colour. The viewer's own
 * "GI only" also leaves out the direct lights, so the two match in meaning;
 * glowing surfaces and the sky still show in the viewer, not in Cycles'
 * indirect image.
 */

import { DataTexture, FloatType, HalfFloatType, LinearFilter, RGBAFormat, Vector2, type Texture } from 'three';
import { EXRLoader } from 'three/addons/loaders/EXRLoader.js';
import { float, screenUV, texture, uniform, vec2, vec3, vec4 } from 'three/tsl';
import type { Node, PerspectiveCamera } from 'three/webgpu';
import type { CameraBookmark } from '../../../../scenes/src/manifest';
import type { GiMode } from '../scene/layers';
import { findReference, renderCommand, type ReferenceInfo, type ReferenceKind } from '../scene/reference-files';
import type { GiTechnique, TechniqueContext, TechniqueId, UpscaleSettings } from './types';
import { copyViewerCamera, createViewCamera } from './view-camera';

/** Linear grey shown where the reference has no pixels (a window wider than the render). */
const OUTSIDE_GREY = 0.02;

/** How far the camera may be from the bookmark (metres, and degrees of view direction) and still count as there. */
const AT_BOOKMARK_DISTANCE = 0.01;
const AT_BOOKMARK_ANGLE_DEGREES = 0.2;

type ImageState =
    | { status: 'loading' }
    | { status: 'missing' }
    | { status: 'failed'; message: string }
    | { status: 'loaded'; texture: Texture; info: ReferenceInfo | null };

export class ReferenceImageTechnique implements GiTechnique {
    readonly id: TechniqueId = 'cycles';
    readonly label = 'Cycles reference (path traced)';
    readonly camera = createViewCamera();
    readonly supportsUpscale = false;
    readonly prepareGeneration = 0;
    readonly outputNode: Node;

    private giMode: GiMode = 'on';
    private bookmarkIndex = 0;
    private readonly images = new Map<string, ImageState>();
    private readonly placeholder = new DataTexture(new Float32Array(4), 1, 1, RGBAFormat, FloatType);
    private readonly imageNode = texture(this.placeholder);
    /** Screen-to-image scale about the centre, per axis (see the file comment). */
    private readonly uvScale = uniform(new Vector2(1, 1));
    /** 1 when a loaded image is showing, 0 for the plain grey. */
    private readonly showing = uniform(0);
    private shownTexture: Texture | null = null;
    private disposed = false;

    constructor(private readonly context: TechniqueContext) {
        this.placeholder.needsUpdate = true;
        // three's EXRLoader keeps the file's last row first in memory, so v = 0 is the image's bottom row,
        // while screenUV.y = 0 is the top of the screen: flip v.
        const centred = vec2(screenUV.x.sub(0.5), float(0.5).sub(screenUV.y));
        const uv = centred.mul(this.uvScale).add(0.5);
        const inside = uv.x.greaterThanEqual(0).and(uv.x.lessThanEqual(1)).and(uv.y.greaterThanEqual(0)).and(uv.y.lessThanEqual(1));
        const image = this.imageNode.sample(uv).rgb;
        const shown = inside.and(this.showing.greaterThan(0.5));
        this.outputNode = vec4(shown.select(image, vec3(OUTSIDE_GREY)), float(1)) as unknown as Node;
        this.loadCurrent();
    }

    // -- which image -------------------------------------------------------

    private get kind(): ReferenceKind {
        return this.giMode === 'only' ? 'indirect' : 'full';
    }

    private get bookmark(): CameraBookmark {
        return this.context.loaded.manifest.cameras[this.bookmarkIndex];
    }

    private get key(): string {
        return `${this.bookmark.name}/${this.kind}`;
    }

    private get current(): ImageState {
        return this.images.get(this.key) ?? { status: 'loading' };
    }

    /** The bookmark whose reference is shown. The viewer calls it when a camera view is chosen. */
    setBookmark(index: number): void {
        if (index === this.bookmarkIndex) return;
        this.bookmarkIndex = index;
        this.loadCurrent();
    }

    get bookmarkNumber(): number {
        return this.bookmarkIndex + 1;
    }

    setGiMode(mode: GiMode): void {
        this.giMode = mode;
        this.loadCurrent();
    }

    private loadCurrent(): void {
        const key = this.key;
        const state = this.images.get(key);
        if (state) {
            this.show(state);
            return;
        }
        this.images.set(key, { status: 'loading' });
        this.show({ status: 'loading' });
        const { manifest } = this.context.loaded;
        void this.load(manifest.id, this.bookmark.name, this.kind).then((loaded) => {
            // The scene was closed while the file loaded.
            if (this.disposed) {
                if (loaded.status === 'loaded') loaded.texture.dispose();
                return;
            }
            this.images.set(key, loaded);
            if (this.key === key) this.show(loaded);
        });
    }

    private async load(sceneId: string, cameraName: string, kind: ReferenceKind): Promise<ImageState> {
        try {
            const found = await findReference(sceneId, cameraName, kind);
            if (!found) return { status: 'missing' };
            const loader = new EXRLoader();
            // Half floats are plenty for display, and filterable everywhere.
            loader.setDataType(HalfFloatType);
            const image = await loader.loadAsync(found.imageUrl);
            image.name = `cycles ${sceneId}/${cameraName}/${kind}`;
            image.minFilter = LinearFilter;
            image.magFilter = LinearFilter;
            image.generateMipmaps = false;
            image.needsUpdate = true;
            return { status: 'loaded', texture: image, info: found.info };
        } catch (error) {
            return { status: 'failed', message: error instanceof Error ? error.message : String(error) };
        }
    }

    private show(state: ImageState): void {
        if (state.status === 'loaded') {
            this.imageNode.value = state.texture;
            this.shownTexture = state.texture;
            this.showing.value = 1;
        } else {
            this.imageNode.value = this.placeholder;
            this.shownTexture = null;
            this.showing.value = 0;
        }
    }

    // -- the frame ---------------------------------------------------------

    syncCamera(viewerCamera: PerspectiveCamera): void {
        copyViewerCamera(viewerCamera, this.camera);
        const image = this.shownTexture?.image as { width: number; height: number } | undefined;
        if (!image) return;
        const bookmark = this.bookmark;
        // Image half-extent per screen half-extent, from the vertical fields of view and the aspect ratios.
        const viewTan = Math.tan((viewerCamera.fov * Math.PI) / 360);
        const imageTan = Math.tan((bookmark.fovYDeg * Math.PI) / 360);
        const imageAspect = image.width / image.height;
        const vertical = viewTan / imageTan;
        this.uvScale.value.set((vertical * viewerCamera.aspect) / imageAspect, vertical);
    }

    /** Whether the viewer's camera is where the reference was rendered from (within a centimetre and a fraction of a degree). */
    isAligned(viewerCamera: PerspectiveCamera): boolean {
        const bookmark = this.bookmark;
        const [px, py, pz] = bookmark.position;
        const offset = Math.hypot(viewerCamera.position.x - px, viewerCamera.position.y - py, viewerCamera.position.z - pz);
        const wanted = [bookmark.target[0] - px, bookmark.target[1] - py, bookmark.target[2] - pz];
        const length = Math.hypot(wanted[0], wanted[1], wanted[2]);
        // three cameras look down their -Z axis.
        const e = viewerCamera.matrixWorld.elements;
        const forward = [-e[8], -e[9], -e[10]];
        const forwardLength = Math.hypot(forward[0], forward[1], forward[2]);
        const cosine = (forward[0] * wanted[0] + forward[1] * wanted[1] + forward[2] * wanted[2]) / (length * forwardLength);
        const angleDegrees = (Math.acos(Math.min(1, Math.max(-1, cosine))) * 180) / Math.PI;
        return offset <= AT_BOOKMARK_DISTANCE && angleDegrees <= AT_BOOKMARK_ANGLE_DEGREES;
    }

    // -- reporting ---------------------------------------------------------

    get ready(): boolean {
        return this.current.status !== 'loading';
    }

    get progress(): number {
        return this.ready ? 1 : 0;
    }

    /** True when the image for this view exists and is showing. */
    get available(): boolean {
        return this.current.status === 'loaded';
    }

    describe(): string {
        const view = `camera view ${this.bookmarkNumber} (${this.bookmark.name})`;
        const what = this.kind === 'indirect' ? 'bounced light only' : 'full image';
        const state = this.current;
        switch (state.status) {
            case 'loading':
                return `Loading the Cycles ${what} for ${view}`;
            case 'missing':
                return `No Cycles reference for ${view} yet. Render it (a few minutes) with: ${renderCommand(this.context.loaded.manifest, this.bookmarkNumber)}`;
            case 'failed':
                return `The Cycles reference for ${view} could not be read: ${state.message}`;
            case 'loaded': {
                const info = state.info;
                const details = info ? `, ${info.width} x ${info.height}, ${info.samplesPerPixel.toLocaleString('en')} samples per pixel, rendered ${info.finishedAt.slice(0, 10)}` : '';
                return `Cycles ${what} for ${view}${details}`;
            }
        }
    }

    encodePrepareStep(): boolean {
        return this.ready;
    }

    restartPrepare(): void {}

    notePrepareFinished(): void {}

    /** The reference ignores the upscaler: it is an image, not a render. */
    setUpscale(_settings: UpscaleSettings | null): void {}

    describeSettings(): Record<string, number | string> {
        return { bookmark: this.bookmarkNumber, kind: this.kind };
    }

    warmUp(): Promise<void> {
        return Promise.resolve();
    }

    dispose(): void {
        this.disposed = true;
        for (const state of this.images.values()) if (state.status === 'loaded') state.texture.dispose();
        this.images.clear();
        this.placeholder.dispose();
    }
}
