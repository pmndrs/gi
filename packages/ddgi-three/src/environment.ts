/**
 * The sky rays see when they miss: three's scene environment, mirrored into
 * the platform's `setEnvironment` (scene contract, `EnvironmentDescriptor`).
 *
 * Where it comes from, first match wins:
 *
 * 1. **An explicit descriptor** the host sets on the adapter (for example the
 *    viewer's calibrated sky files). Passed through as is.
 * 2. **A pmndrs/sky `Sky`** the host hands over: a `procedural` sky toward
 *    its sun, with its `exposure` (`luminanceScale`) as `sunIlluminance`
 *    (DECISIONS 2026-11-05: the same quantity). pmndrs/sky's `turbidity` is a
 *    Mie scale (1 = Earth), not Preetham's turbidity, so it isn't passed.
 *    pmndrs/sky only writes render-target textures into `scene.environment`
 *    and `scene.background`, so it can't be found in the scene: the host
 *    passes it.
 * 3. **three's `SkyMesh`** (`isSkyMesh`) in the scene, a Preetham sky like
 *    the platform's: `procedural` toward its `sunPosition`, with its
 *    `turbidity`, and `sunIlluminance` from the sun: the brightest
 *    `DirectionalLight`'s intensity × the luminance of its colour.
 * 4. **`scene.environment`** as a texture (with `environmentIntensity` and
 *    `environmentRotation`), then **`scene.background`** as a texture (with
 *    `backgroundIntensity` and `backgroundRotation`): equirect (rows as three
 *    stores them, which is the contract's layout) or a cube map, mirrored in
 *    x (cube-mirror.ts). The adapter uploads its own copy, because three's
 *    GPU textures can only be reached through the renderer's backend.
 *    Render-target textures (PMREM, `CubeCamera`) and compressed ones can't
 *    be read that way; they're skipped with a warning.
 * 5. **`scene.background` as a `Color`:** a uniform sky of that radiance.
 * 6. Otherwise none.
 *
 * Rotation: the platform turns the sky about +Y only (`rotationY`, the same
 * sense as three's `environmentRotation.y`); x and z rotations are dropped
 * with a warning.
 */

import {
    EquirectangularReflectionMapping,
    EquirectangularRefractionMapping,
    FloatType,
    HalfFloatType,
    RGBAFormat,
    SRGBColorSpace,
    UnsignedByteType,
    type Color,
    type DirectionalLight,
    type Euler,
    type Scene,
    type Texture,
} from 'three';
import type { EnvironmentDescriptor, SceneApi, Vec3 } from '@pmndrs/ddgi-core';
import { luminance } from './convert.ts';
import { CUBE_FACE_COUNT, flipPixels, mirroredCubeFaces, type CubeFacePixels, type PixelArray } from './cube-mirror.ts';

/** What the adapter reads from a pmndrs/sky `Sky` (its baker's sky node uniforms). */
export interface PmndrsSkyLike {
    readonly baker: {
        readonly sky: {
            /** pmndrs/sky's `exposure`: the sky is computed for a sun of illuminance 1, then scaled by this. */
            readonly luminanceScale: { readonly value: number };
            /** Unit direction toward the sun. */
            readonly sunDirection: { readonly value: { readonly x: number; readonly y: number; readonly z: number } };
        };
    };
}

/** What the adapter reads from three's `SkyMesh` (examples/jsm/objects/SkyMesh.js). */
export interface SkyMeshLike {
    readonly isSkyMesh: true;
    readonly turbidity: { readonly value: number };
    readonly sunPosition: { readonly value: { readonly x: number; readonly y: number; readonly z: number } };
}

/** Everything the environment can come from this frame; the scene walk fills in `skyMesh` and `sun`. */
export interface EnvironmentSources {
    readonly override?: EnvironmentDescriptor | null;
    readonly sky?: PmndrsSkyLike | null;
    readonly skyMesh?: SkyMeshLike | null;
    readonly sun?: DirectionalLight | null;
}

export type Warn = (message: string) => void;

// ---------------------------------------------------------------------------
// Resolution (pure, unit-tested)
// ---------------------------------------------------------------------------

/**
 * The environment the platform should have, in a form that's cheap to compare
 * every frame: a kind, a few numbers, and at most one texture or descriptor.
 */
export interface EnvironmentChoice {
    readonly kind: 'override' | 'procedural' | 'texture' | 'color' | 'none';
    readonly override?: EnvironmentDescriptor;
    readonly texture?: Texture;
    readonly layout?: 'cube' | 'equirect';
    /** procedural: sun direction xyz, turbidity (NaN = default), sunIlluminance. texture: version, intensity, rotation x, y, z. color: rgb. */
    readonly numbers: readonly number[];
}

export function sameChoice(a: EnvironmentChoice | null, b: EnvironmentChoice): boolean {
    if (!a || a.kind !== b.kind || a.override !== b.override || a.texture !== b.texture || a.layout !== b.layout) return false;
    if (a.numbers.length !== b.numbers.length) return false;
    for (let index = 0; index < a.numbers.length; index++) if (!Object.is(a.numbers[index], b.numbers[index])) return false;
    return true;
}

/** The sun's illuminance at normal incidence: intensity × luminance of its colour (scene contract, `sunIlluminance`). */
export function sunIlluminance(sun: DirectionalLight | null | undefined): number {
    return sun ? sun.intensity * luminance(sun.color) : 1;
}

/** Whether three can sample this texture as a sky (an equirect or cube mapping), and how it's laid out. */
export function skyLayout(texture: Texture): 'cube' | 'equirect' | null {
    if ((texture as Texture & { isCubeTexture?: boolean }).isCubeTexture) return 'cube';
    if (texture.mapping === EquirectangularReflectionMapping || texture.mapping === EquirectangularRefractionMapping) return 'equirect';
    return null;
}

/** Why the adapter can't read this texture's pixels on the CPU, or null if it can. */
export function unreadableReason(texture: Texture): string | null {
    const flags = texture as Texture & { isRenderTargetTexture?: boolean; isCompressedTexture?: boolean; isCubeTexture?: boolean };
    if (flags.isRenderTargetTexture) return 'it is a render-target texture (PMREM, CubeCamera or a sky baker), whose pixels live only on the GPU behind three\'s backend';
    if (flags.isCompressedTexture) return 'it is compressed';
    if (flags.isCubeTexture) {
        const faces = texture.image as unknown[] | undefined;
        if (!Array.isArray(faces) || faces.length !== CUBE_FACE_COUNT || faces.some((face) => !face)) return 'its six faces aren\'t loaded yet';
        return null;
    }
    if (!texture.image) return 'its image isn\'t loaded yet';
    return null;
}

function rotationNumbers(rotation: Euler): number[] {
    return [rotation.x, rotation.y, rotation.z];
}

/**
 * Picks the environment source for this frame (see the file comment for the
 * order). `skipped` collects textures that were passed over and why, for a
 * warning.
 */
export function chooseEnvironment(scene: Scene, sources: EnvironmentSources, skipped: (what: string, reason: string) => void = () => {}): EnvironmentChoice {
    if (sources.override) return { kind: 'override', override: sources.override, numbers: [] };
    if (sources.sky) {
        const { luminanceScale, sunDirection } = sources.sky.baker.sky;
        const sun = sunDirection.value;
        return { kind: 'procedural', numbers: [sun.x, sun.y, sun.z, Number.NaN, luminanceScale.value] };
    }
    if (sources.skyMesh) {
        const sun = sources.skyMesh.sunPosition.value;
        return { kind: 'procedural', numbers: [sun.x, sun.y, sun.z, sources.skyMesh.turbidity.value, sunIlluminance(sources.sun)] };
    }
    const candidates: [string, Texture | Color | null, number, Euler][] = [
        ['scene.environment', scene.environment, scene.environmentIntensity, scene.environmentRotation],
        ['scene.background', scene.background, scene.backgroundIntensity, scene.backgroundRotation],
    ];
    for (const [what, value, intensity, rotation] of candidates) {
        const texture = value as Texture | null;
        if (!texture?.isTexture) continue;
        const layout = skyLayout(texture);
        if (!layout) {
            skipped(what, 'it isn\'t an equirect or cube texture');
            continue;
        }
        const reason = unreadableReason(texture);
        if (reason) {
            skipped(what, reason);
            continue;
        }
        return { kind: 'texture', texture, layout, numbers: [texture.version, intensity, ...rotationNumbers(rotation)] };
    }
    const background = scene.background as Color | null;
    if (background && (background as Color).isColor) return { kind: 'color', numbers: [background.r, background.g, background.b] };
    return { kind: 'none', numbers: [] };
}

/** Normalizes a direction toward the sun; straight up if it has no length. */
function sunDirection(x: number, y: number, z: number): Vec3 {
    const length = Math.hypot(x, y, z);
    return length > 0 ? [x / length, y / length, z / length] : [0, 1, 0];
}

/**
 * The descriptor for a choice. `upload` turns a texture choice into a GPU
 * texture (environment-upload below), and returns null if it can't.
 */
export function environmentDescriptor(choice: EnvironmentChoice, upload: (texture: Texture, layout: 'cube' | 'equirect') => GPUTexture | null, warn: Warn): EnvironmentDescriptor {
    switch (choice.kind) {
        case 'override':
            return choice.override!;
        case 'procedural': {
            const [x, y, z, turbidity, illuminance] = choice.numbers;
            return {
                type: 'procedural',
                sunDirection: sunDirection(x, y, z),
                turbidity: Number.isNaN(turbidity) ? undefined : turbidity,
                sunIlluminance: illuminance,
            };
        }
        case 'texture': {
            const [, intensity, rotationX, rotationY, rotationZ] = choice.numbers;
            if (rotationX !== 0 || rotationZ !== 0) warn('the platform turns the sky about +Y only; the x and z parts of the environment rotation are ignored for GI.');
            const gpuTexture = upload(choice.texture!, choice.layout!);
            if (!gpuTexture) return { type: 'none' };
            return { type: 'texture', texture: gpuTexture, layout: choice.layout!, intensity, rotationY };
        }
        case 'color': {
            const [r, g, b] = choice.numbers;
            return { type: 'color', color: [r, g, b] };
        }
        case 'none':
            return { type: 'none' };
    }
}

// ---------------------------------------------------------------------------
// Upload (browser only)
// ---------------------------------------------------------------------------

interface FormatChoice {
    readonly format: GPUTextureFormat;
    readonly bytesPerTexel: number;
}

/** The GPU format for three's data type and colour space (RGBA only). */
function dataFormat(texture: Texture): FormatChoice | null {
    if (texture.format !== RGBAFormat) return null;
    if (texture.type === FloatType) return { format: 'rgba32float', bytesPerTexel: 16 };
    if (texture.type === HalfFloatType) return { format: 'rgba16float', bytesPerTexel: 8 };
    if (texture.type === UnsignedByteType) return { format: texture.colorSpace === SRGBColorSpace ? 'rgba8unorm-srgb' : 'rgba8unorm', bytesPerTexel: 4 };
    return null;
}

function eightBitFormat(texture: Texture): FormatChoice {
    return { format: texture.colorSpace === SRGBColorSpace ? 'rgba8unorm-srgb' : 'rgba8unorm', bytesPerTexel: 4 };
}

function isDataImage(image: unknown): image is CubeFacePixels<PixelArray> {
    const candidate = image as CubeFacePixels<PixelArray> | null;
    return !!candidate && ArrayBuffer.isView(candidate.data) && typeof candidate.width === 'number' && typeof candidate.height === 'number';
}

type DrawableImage = ImageBitmap | HTMLImageElement | HTMLCanvasElement | OffscreenCanvas | ImageData;

/** 8-bit RGBA pixels of a browser image, rows top first as the image is. */
function imagePixels(image: DrawableImage): CubeFacePixels<Uint8ClampedArray> {
    if (typeof ImageData !== 'undefined' && image instanceof ImageData) return { data: image.data, width: image.width, height: image.height };
    const width = (image as HTMLImageElement).naturalWidth || image.width;
    const height = (image as HTMLImageElement).naturalHeight || image.height;
    const canvas = new OffscreenCanvas(width, height);
    const context = canvas.getContext('2d');
    if (!context) throw new Error('@pmndrs/ddgi: no 2D canvas to read an environment image with');
    context.drawImage(image as CanvasImageSource, 0, 0);
    return { data: context.getImageData(0, 0, width, height).data, width, height };
}

/** TEXTURE_BINDING to be sampled, COPY_DST and RENDER_ATTACHMENT to be written (the last is what `copyExternalImageToTexture` needs). */
function environmentUsage(): GPUTextureUsageFlags {
    return GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT;
}

function writeLayer(device: GPUDevice, target: GPUTexture, layer: number, image: CubeFacePixels<PixelArray>, format: FormatChoice): void {
    device.queue.writeTexture(
        { texture: target, origin: { x: 0, y: 0, z: layer } },
        image.data as Uint8Array<ArrayBuffer>,
        { bytesPerRow: image.width * format.bytesPerTexel },
        { width: image.width, height: image.height },
    );
}

/**
 * Copies an equirect three texture into a new GPU texture in the contract's
 * layout. three keeps memory row 0 at the nadir (EXR and HDR loaders write
 * rows bottom-up, and images with `flipY` are flipped on upload), which is the
 * contract's layout, so the rows go in exactly as three would upload them.
 */
export function uploadEquirect(device: GPUDevice, texture: Texture, warn: Warn): GPUTexture | null {
    const image = texture.image as unknown;
    if (isDataImage(image)) {
        const format = dataFormat(texture);
        if (!format) {
            warn(`environment texture "${texture.name || texture.uuid}" has a format or type GI can't read (RGBA float, half float or 8-bit only); GI uses no sky.`);
            return null;
        }
        const rows = texture.flipY ? flipPixels(image, 4, false, true) : image;
        const gpu = device.createTexture({ label: 'ddgi-three environment (equirect)', size: { width: image.width, height: image.height }, format: format.format, usage: environmentUsage() });
        writeLayer(device, gpu, 0, rows, format);
        return gpu;
    }
    const drawable = image as DrawableImage;
    const pixels = imagePixels(drawable);
    if (!(pixels.width > 0 && pixels.height > 0)) return null;
    const format = eightBitFormat(texture);
    const gpu = device.createTexture({ label: 'ddgi-three environment (equirect image)', size: { width: pixels.width, height: pixels.height }, format: format.format, usage: environmentUsage() });
    writeLayer(device, gpu, 0, texture.flipY ? flipPixels(pixels, 4, false, true) : pixels, format);
    return gpu;
}

/**
 * Copies a three cube texture into a new 6-layer GPU texture, mirrored in x
 * so the platform's WebGPU-convention lookup sees what three shows
 * (cube-mirror.ts). Faces may be images, `{ data, width, height }` objects or
 * `DataTexture`s (HDRCubeTextureLoader), all of one size and type.
 */
export function uploadCube(device: GPUDevice, texture: Texture, warn: Warn): GPUTexture | null {
    const faces = (texture.image as unknown[]).map((face) => ((face as Texture).isTexture ? (face as Texture).image : face));
    const dataFaces = faces.every(isDataImage);
    const format = dataFaces ? dataFormat(texture) : eightBitFormat(texture);
    if (!format) {
        warn(`cube environment "${texture.name || texture.uuid}" has a format or type GI can't read (RGBA float, half float or 8-bit only); GI uses no sky.`);
        return null;
    }
    const pixels: CubeFacePixels<PixelArray>[] = dataFaces ? (faces as CubeFacePixels<PixelArray>[]) : faces.map((face) => imagePixels(face as DrawableImage));
    const size = pixels[0].width;
    if (pixels.some((face) => face.width !== size || face.height !== size)) {
        warn(`cube environment "${texture.name || texture.uuid}" has non-square faces or faces of different sizes; GI uses no sky.`);
        return null;
    }
    const mirrored = mirroredCubeFaces(pixels, 4, texture.flipY);
    const gpu = device.createTexture({
        label: 'ddgi-three environment (cube, mirrored in x)',
        size: { width: size, height: size, depthOrArrayLayers: CUBE_FACE_COUNT },
        format: format.format,
        usage: environmentUsage(),
    });
    for (let layer = 0; layer < CUBE_FACE_COUNT; layer++) writeLayer(device, gpu, layer, mirrored[layer], format);
    return gpu;
}

// ---------------------------------------------------------------------------
// The per-frame sync
// ---------------------------------------------------------------------------

/**
 * Keeps the platform's environment in step with three's. Each frame it picks
 * the source ({@link chooseEnvironment}); only when that changed does it call
 * `setEnvironment`, and only when the texture itself (or its version) changed
 * does it upload a new copy. It owns the GPU copies it makes.
 */
export class EnvironmentSync {
    private readonly scene: SceneApi;
    private readonly device: GPUDevice;
    private readonly warn: Warn;
    private current: EnvironmentChoice | null = null;
    private uploaded: { readonly texture: Texture; readonly version: number; readonly gpu: GPUTexture | null } | null = null;
    /** How many times `setEnvironment` was called. */
    updates = 0;
    /** How many textures were uploaded. */
    uploads = 0;

    constructor(scene: SceneApi, device: GPUDevice, warn: Warn) {
        this.scene = scene;
        this.device = device;
        this.warn = warn;
    }

    update(threeScene: Scene, sources: EnvironmentSources): void {
        const choice = chooseEnvironment(threeScene, sources, (what, reason) => this.warn(`${what} can't light GI rays: ${reason}.`));
        if (sameChoice(this.current, choice)) return;
        this.current = choice;
        const descriptor = environmentDescriptor(choice, (texture, layout) => this.upload(texture, layout), this.warn);
        if (choice.kind !== 'texture') this.release();
        this.scene.setEnvironment(descriptor);
        this.updates++;
    }

    private upload(texture: Texture, layout: 'cube' | 'equirect'): GPUTexture | null {
        if (this.uploaded && this.uploaded.texture === texture && this.uploaded.version === texture.version) return this.uploaded.gpu;
        // The platform reads the texture only while projecting the sky, in the encode after setEnvironment, so the old copy can go.
        this.release();
        const gpu = layout === 'cube' ? uploadCube(this.device, texture, this.warn) : uploadEquirect(this.device, texture, this.warn);
        this.uploaded = { texture, version: texture.version, gpu };
        this.uploads++;
        return gpu;
    }

    private release(): void {
        this.uploaded?.gpu?.destroy();
        this.uploaded = null;
    }

    dispose(): void {
        this.release();
        this.current = null;
    }
}
