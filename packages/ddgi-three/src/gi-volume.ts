/**
 * A GI volume shaped like three's `LightProbeGrid` (SPEC, "Matching the
 * LightProbeGrid shape"), so its users can switch to DDGI with the same
 * numbers and as many lines:
 *
 * ```ts
 * const volume = new GIVolume(6, 5, 6, 6, 6, 6);   // like new LightProbeGrid(...)
 * volume.position.set(0, 2.5, 0);
 * scene.add(volume);
 * await volume.start(renderer, scene);              // like probes.bake(renderer, scene)
 * ```
 *
 * `start` (volume-start.ts) makes the platform and DDGI on three's own
 * device, warms the probes up, and switches on the render-time update: from
 * then on GI updates itself inside `renderer.render`, with no per-frame
 * call. `recalibrate()` measures the GPU budget again; `restart()` starts
 * the probes over; `dispose()` removes the GI light and frees the platform. Hosts that want control (several
 * techniques, a shared platform) still build it by hand: `ddgiVolume()`
 * gives the DDGI settings, and `ThreeGI` takes `{ volume }`.
 *
 * As in `LightProbeGrid`, the box is `width × height × depth`, centred on the
 * object's world position and axis-aligned (rotation and scale are not
 * used), split into `nx × ny × nz` cells with a probe at each cell's centre.
 * `boundingBox`, `priority` and `intensity` mean what they mean there;
 * `ThreeGI` copies `priority` and `intensity` onto its GI light each frame,
 * so turning GI down or off works on the volume too.
 *
 * DDGI places probes on a grid from a box and a spacing, `floor(size /
 * spacing) + 1` per axis, centred in the box. Cell centres are that grid with
 * spacing = width / nx and a box of width − width / nx (the outermost probes
 * sit half a cell inside the volume), which {@link GIVolume.ddgiVolume} gives.
 * Changing the volume after `prepare` needs a new `DDGITechnique` settings
 * and another `prepare`, as for any DDGI volume.
 *
 * The volume itself adds no light: the GI light does (gi-light.ts). It is a
 * `Light` because `LightProbeGrid` is one; its registered node does nothing.
 */

import { AnalyticLightNode, Box3, Light, Vector3, type Node, type Scene, type WebGPURenderer } from 'three/webgpu';
import { exportIrradianceSH, probeGrid, type DDGIVolumeSettings, type GIPlatform, type LoadedField, type LoadFieldOptions, type Vec3 } from '@pmndrs/ddgi-core';
import { GI_LIGHT_PRIORITY, type GILight } from './gi-light.ts';
import type { ThreeGI } from './three-gi.ts';
import { toLightProbeGrid, type ExportedLightProbeGrid, type ToLightProbeGridOptions } from './light-probe-grid.ts';
import { restartVolume, startVolume, type GIVolumeStartOptions, type StartedGI } from './volume-start.ts';

/**
 * `LightProbeGrid`'s default probe count for an axis: one per unit of length,
 * at least one. With a quality preset (`start`'s `preset`), one per
 * `spacing` metres instead.
 */
function defaultProbes(length: number, spacing = 1): number {
    return Math.max(1, Math.round(length / spacing));
}

export class GIVolume extends Light {
    readonly isGIVolume = true;
    /** Full size of the volume along each axis. */
    width: number;
    height: number;
    depth: number;
    /** Probes along each axis. */
    readonly resolution: Vector3;
    /** The box the volume covers, in world space. Updated by {@link updateBoundingBox} (and by `ddgiVolume`). */
    readonly boundingBox = new Box3();
    /** Where probe-like lights overlap, the highest priority lights the fragment. */
    priority = GI_LIGHT_PRIORITY;
    /**
     * The probe counts the constructor chose itself (none given), per axis,
     * or null where the caller gave one: a quality preset's spacing replaces
     * these at `start` ({@link useDefaultSpacing}), unless `resolution` was
     * changed since.
     */
    private readonly defaultCounts: readonly (number | null)[];
    /** What `start` made, while it lives. */
    private started: StartedGI | null = null;
    private starting: Promise<void> | null = null;

    constructor(width = 1, height = 1, depth = 1, widthProbes?: number, heightProbes?: number, depthProbes?: number) {
        super(0xffffff, 1);
        (this as { type: string }).type = 'GIVolume';
        this.width = width;
        this.height = height;
        this.depth = depth;
        this.resolution = new Vector3(widthProbes ?? defaultProbes(width), heightProbes ?? defaultProbes(height), depthProbes ?? defaultProbes(depth));
        this.defaultCounts = [widthProbes, heightProbes, depthProbes].map((given, axis) => (given === undefined ? this.resolution.getComponent(axis) : null));
    }

    /**
     * Where the volume was made without probe counts, and `resolution` still
     * holds the counts the constructor chose (one a metre), spaces those axes'
     * probes `spacing` metres apart instead (at least one probe). `start`
     * calls it with its preset's `probeSpacing`; counts the caller gave or set
     * are kept.
     */
    useDefaultSpacing(spacing: number): void {
        if (!(spacing > 0) || !Number.isFinite(spacing)) return;
        const lengths = [this.width, this.height, this.depth];
        this.defaultCounts.forEach((count, axis) => {
            if (count === null || this.resolution.getComponent(axis) !== count) return;
            this.resolution.setComponent(axis, defaultProbes(lengths[axis], spacing));
        });
    }

    /**
     * A volume whose probes are exactly DDGI's grid for these settings (a box
     * and a spacing): `floor(size / spacing) + 1` probes per axis, `spacing`
     * apart, centred on `center`. For hosts that describe DDGI volumes that
     * way (the viewer's scene manifests). Its position is `center`.
     */
    static fromDDGIVolume(settings: DDGIVolumeSettings): GIVolume {
        const grid = probeGrid(settings);
        const [nx, ny, nz] = grid.counts;
        const [sx, sy, sz] = grid.spacing;
        const volume = new GIVolume(nx * sx, ny * sy, nz * sz, nx, ny, nz);
        volume.position.set(settings.center[0], settings.center[1], settings.center[2]);
        volume.updateMatrixWorld();
        return volume;
    }

    /** The volume's centre: its world position (`matrixWorld` must be current). */
    private center(): Vec3 {
        const e = this.matrixWorld.elements;
        return [e[12], e[13], e[14]];
    }

    /** Sets `boundingBox` from the world position and the size. */
    updateBoundingBox(): Box3 {
        const [x, y, z] = this.center();
        this.boundingBox.min.set(x - this.width / 2, y - this.height / 2, z - this.depth / 2);
        this.boundingBox.max.set(x + this.width / 2, y + this.height / 2, z + this.depth / 2);
        return this.boundingBox;
    }

    /**
     * The DDGI volume settings with a probe at the centre of each of the
     * `resolution` cells: spacing = size / probes per axis, and a box one
     * cell smaller than the volume, so `floor(size / spacing) + 1` gives the
     * probe counts back. World matrices must be current.
     */
    ddgiVolume(): DDGIVolumeSettings {
        this.updateBoundingBox();
        const lengths: Vec3 = [this.width, this.height, this.depth];
        const probes: Vec3 = [this.resolution.x, this.resolution.y, this.resolution.z];
        const spacing = lengths.map((length, axis) => length / probes[axis]) as unknown as Vec3;
        const size = lengths.map((length, axis) => length - spacing[axis]) as unknown as Vec3;
        return { center: this.center(), size, spacing };
    }

    /**
     * Starts GI on three's device (volume-start.ts): the platform and DDGI on
     * this volume's grid, the warm-up, and the render-time update. Adds the
     * volume to `scene` if it has no parent. Resolves when the probes are
     * warm; GI then updates itself on every render. Throws `GILimitError`
     * (naming the limit, with the fix) when the scene outgrows the device.
     */
    start(renderer: WebGPURenderer, scene: Scene, options: GIVolumeStartOptions = {}): Promise<void> {
        if (this.started || this.starting) throw new Error('@pmndrs/ddgi/three: this GIVolume has already been started; dispose() it first.');
        this.starting = startVolume(this, renderer, scene, options).then(
            (started) => {
                this.started = started;
                this.starting = null;
            },
            (error: unknown) => {
                this.starting = null;
                throw error;
            },
        );
        return this.starting;
    }

    /**
     * "Start the probes over": forgets the field and the budget's
     * calibration, and runs `start`'s start-up frames again (the warm-up and
     * the calibration, volume-start.ts), for after the scene changed a lot.
     * Resolves when the probes are warm again.
     */
    restart(options: Pick<GIVolumeStartOptions, 'budget' | 'warmUp' | 'camera'> = {}): Promise<void> {
        if (!this.started) throw new Error('@pmndrs/ddgi/three: the GIVolume has no GI yet; await volume.start(renderer, scene) before restarting it.');
        return restartVolume(this, this.started, options);
    }

    /** Whether `start` has finished. */
    get isStarted(): boolean {
        return this.started !== null;
    }

    /** The scene's link to GI, after `start`: budget, scene sync, `markChanged`, timings. */
    get threeGI(): ThreeGI | null {
        return this.started?.threeGI ?? null;
    }

    /** The GI platform `start` made (for stats and debug views), or null. */
    get platform(): GIPlatform | null {
        return this.started?.platform ?? null;
    }

    /** The GI light `start` added, which the composite reads (`ddgi(scenePass, volume)`). */
    get light(): GILight {
        if (!this.started) throw new Error('@pmndrs/ddgi/three: the GIVolume has no GI yet; await volume.start(renderer, scene) before rendering with it.');
        return this.started.threeGI.light;
    }

    /**
     * Measures again how much probe work fits the frame budget (budget v2),
     * over the next frames: after the scene changed a lot, or a different
     * GPU load. The platform calibrates by itself after `start`.
     */
    recalibrate(): void {
        this.started?.platform.recalibrate();
    }

    /**
     * A snapshot of the field as three's `LightProbeGrid` (light-probe-grid.ts),
     * for a static or WebGL fallback: the field renderers see, so with light
     * groups on, the layers composed with their weights as of the last frame.
     * One GPU projection and one readback; not for every frame. Add the grid
     * to a scene and set this volume's `intensity` to 0 (or remove it) to
     * light with the grid alone.
     */
    async exportLightProbeGrid(options: ToLightProbeGridOptions = {}): Promise<ExportedLightProbeGrid> {
        if (!this.started) throw new Error('@pmndrs/ddgi/three: the GIVolume has no GI yet; await volume.start(renderer, scene) before exporting it.');
        return toLightProbeGrid(await exportIrradianceSH(this.started.technique), options);
    }

    /**
     * Saves the probes as a saved bake (a `.ddgibake` file, engine-neutral:
     * ddgi-core's saved-field.ts): what they have learned, with the layout
     * it needs and a fingerprint of the scene it was baked for. Pass it to
     * `start(renderer, scene, { bake })` or {@link loadBake} later to begin
     * with finished GI. One GPU readback, here only; save after `start` has
     * resolved (warm probes).
     */
    saveBake(): Promise<Uint8Array<ArrayBuffer>> {
        if (!this.started) throw new Error('@pmndrs/ddgi/three: the GIVolume has no GI yet; await volume.start(renderer, scene) before saving a bake.');
        return this.started.technique.saveField();
    }

    /**
     * Replaces the probes with a saved bake ({@link saveBake}): from the next
     * render GI shows the loaded probes and keeps updating from them. Throws
     * `SavedFieldError` for a file that isn't a bake, an unknown format
     * version, or a layout that doesn't fit this volume (another probe grid,
     * tile size or light-group setting; the message names it). A bake saved
     * for a scene that changed since loads as stale (`stale: true` in the
     * result, with the parts that changed), unless `ifSceneChanged` is
     * `refuse`: every probe then updates from the loaded values. To start
     * from a bake without warming up first, pass it to `start` instead.
     */
    loadBake(bytes: Uint8Array | ArrayBuffer, options: LoadFieldOptions = {}): LoadedField {
        if (!this.started) throw new Error('@pmndrs/ddgi/three: the GIVolume has no GI yet; pass the bake to volume.start(renderer, scene, { bake }) instead.');
        return this.started.technique.loadField(bytes, options);
    }

    /** What `start`'s `bake` option loaded (stale or not), or null without one. */
    get loadedBake(): LoadedField | null {
        return this.started?.loadedBake ?? null;
    }

    /** Removes the GI light and frees the platform `start` made. The volume can be started again afterwards. */
    dispose(): void {
        const started = this.started;
        this.started = null;
        if (started) {
            started.threeGI.dispose();
            started.platform.dispose();
        }
        super.dispose();
    }
}

/** The volume describes; the GI light lights. LightsNode still needs a node for every light. */
class GIVolumeNode extends AnalyticLightNode<GIVolume> {
    static get type(): string {
        return 'GIVolumeNode';
    }

    setup(): Node | undefined {
        return undefined;
    }
}

GIVolume.registerNode(GIVolumeNode);
