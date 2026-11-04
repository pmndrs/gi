/**
 * Baseline 1: three.js r187's `LightProbeGrid` (L2 spherical harmonics baked
 * from rendered cube faces, with optional bounce passes).
 *
 * Set-up follows three's own `webgpu_lightprobes_sponza` example: 16 probes
 * per `bake()` call, finishing each pass over the whole grid before the next
 * bounce pass starts, with 32x32 faces. The grid covers the manifest's
 * suggested volume, with the probe count capped (see `probe-grid-layout.ts`).
 *
 * The bake sees what the viewing cameras see on layer 0: the model, the
 * manifest's lights, and the manifest's sky. It gets no light of its own.
 *
 * What it can't do (so the viewer shows it honestly): it has no visibility
 * term, so light leaks through thin walls; it is a bake, so moving a light
 * means a re-bake.
 */

import { Vector3 } from 'three';
import { LightProbeGrid } from 'three/addons/lighting/LightProbeGrid.js';
import { LAYER_PROBE_GRID } from '../scene/layers';
import { ScenePassTechnique } from './scene-pass-technique';
import { chooseProbeGridLayout, DEFAULT_MAX_PROBES, type ProbeGridLayout } from './probe-grid-layout';
import type { TechniqueContext } from './types';

/** Probes per `bake()` call, as in three's example. Each call also re-renders the shadow maps once. */
const PROBES_PER_STEP = 16;

/** Near plane of the capture cameras, in metres (three's example value). */
const CAPTURE_NEAR = 0.05;

export interface LightProbeGridSettings {
    /** Extra bounce passes after the direct pass. */
    bounces: number;
    /** Probe budget; the manifest's spacing is widened to fit it. */
    maxProbes: number;
    /** Resolution of each captured cube face. */
    cubemapSize: number;
}

export const DEFAULT_LPG_SETTINGS: LightProbeGridSettings = { bounces: 1, maxProbes: DEFAULT_MAX_PROBES, cubemapSize: 32 };

export class LightProbeGridTechnique extends ScenePassTechnique {
    readonly id = 'lpg';
    readonly label = 'Light probe grid (three.js)';

    private settings: LightProbeGridSettings;
    private layout: ProbeGridLayout;
    private grid: LightProbeGrid;

    // Bake progress. `generation` counts restarts; see `GiTechnique.prepareGeneration`.
    private generation = 0;
    /** Which pass comes next: 0 is direct light, 1 and up are bounces. */
    private pass = 0;
    private nextProbe = 0;
    private finished = false;
    /** Wall time of the last finished bake, as reported by whoever waited for the GPU. */
    private finishedWallMs: number | null = null;
    /** Whether the scene target has been rebuilt since the atlas was first written (see `recreateTarget`). */
    private atlasBound = false;

    constructor(context: TechniqueContext, settings: LightProbeGridSettings = DEFAULT_LPG_SETTINGS) {
        super(context);
        this.settings = { ...settings };
        this.layout = chooseProbeGridLayout(context.loaded.manifest.volume.size, context.loaded.manifest.volume.probeSpacing, settings.maxProbes);
        this.grid = this.createGrid();
    }

    protected giLayers(): readonly number[] {
        return [LAYER_PROBE_GRID];
    }

    protected hasNoGi(): boolean {
        return false;
    }

    get ready(): boolean {
        return this.finished;
    }

    get prepareGeneration(): number {
        return this.generation;
    }

    get progress(): number {
        const passes = this.settings.bounces + 1;
        const done = this.pass * this.layout.total + this.nextProbe;
        return this.finished ? 1 : Math.min(1, done / (passes * this.layout.total));
    }

    describe(): string {
        const grid = `${this.layout.counts.join(' x ')} = ${this.layout.total} probes, ${this.layout.spacing.toFixed(2)} m apart`;
        const bounces = `${this.settings.bounces} bounce${this.settings.bounces === 1 ? '' : 's'}`;
        if (this.finished) {
            const took = this.finishedWallMs === null ? '' : ` in ${(this.finishedWallMs / 1000).toFixed(1)} s`;
            return `Baked ${grid}, ${bounces}${took}`;
        }
        const pass = this.pass === 0 ? 'direct light' : `bounce ${this.pass}`;
        return `Baking ${grid}: ${pass}, ${(this.progress * 100).toFixed(0)}%`;
    }

    /**
     * Takes new settings. Only a change that alters the baked result (probe budget,
     * bounces, face size) throws the bake away; the same settings again do nothing.
     * Returns whether a new bake was started.
     */
    configure(settings: LightProbeGridSettings): boolean {
        const layoutChanged = settings.maxProbes !== this.settings.maxProbes;
        const bakeChanged = layoutChanged || settings.bounces !== this.settings.bounces || settings.cubemapSize !== this.settings.cubemapSize;
        this.settings = { ...settings };
        if (layoutChanged) {
            const volume = this.context.loaded.manifest.volume;
            this.layout = chooseProbeGridLayout(volume.size, volume.probeSpacing, settings.maxProbes);
            this.replaceGrid();
        }
        if (bakeChanged) this.restartPrepare();
        return bakeChanged;
    }

    get currentSettings(): Readonly<LightProbeGridSettings> {
        return this.settings;
    }

    /** Starts the bake again from the direct-light pass at probe 0 of the current grid. */
    restartPrepare(): void {
        this.generation++;
        this.pass = 0;
        this.nextProbe = 0;
        this.finished = false;
        this.finishedWallMs = null;
    }

    encodePrepareStep(): boolean {
        if (this.finished) return true;
        const count = Math.min(PROBES_PER_STEP, this.layout.total - this.nextProbe);
        this.bakeRange(this.nextProbe, count, this.pass);
        this.nextProbe += count;
        if (this.nextProbe >= this.layout.total) {
            this.nextProbe = 0;
            this.pass++;
        }
        if (this.pass > this.settings.bounces) this.finished = true;
        return this.finished;
    }

    notePrepareFinished(generation: number, wallMs: number): void {
        if (generation !== this.generation || !this.finished) return;
        this.finishedWallMs = wallMs;
    }

    describeSettings(): Record<string, number | string> {
        return {
            probes: this.layout.total,
            probe_grid: this.layout.counts.join('x'),
            probe_spacing_m: Number(this.layout.spacing.toFixed(3)),
            bounces: this.settings.bounces,
            cubemap_size: this.settings.cubemapSize,
            probes_per_step: PROBES_PER_STEP,
        };
    }

    /** Bakes one probe per pass type, so shader compilation is not part of a timed bake, then restarts. */
    async warmUp(): Promise<void> {
        for (let pass = 0; pass <= this.settings.bounces; pass++) this.bakeRange(0, 1, pass);
        await this.context.device.queue.onSubmittedWorkDone();
        this.restartPrepare();
    }

    dispose(): void {
        this.disposeGrid();
        this.disposeTarget();
    }

    private bakeRange(start: number, count: number, pass: number): void {
        const { renderer, loaded } = this.context;
        this.grid.bake(renderer, loaded.scene, {
            cubemapSize: this.settings.cubemapSize,
            near: CAPTURE_NEAR,
            far: loaded.diagonal * 1.5,
            start,
            count,
            pass,
        });
        // After the first bake call the atlas exists on the GPU in its final form; see recreateTarget.
        if (!this.atlasBound) {
            this.recreateTarget();
            this.atlasBound = true;
        }
    }

    private disposeGrid(): void {
        this.grid.removeFromParent();
        this.grid.dispose();
    }

    private createGrid(): LightProbeGrid {
        const volume = this.context.loaded.manifest.volume;
        const [nx, ny, nz] = this.layout.counts;
        const grid = new LightProbeGrid(volume.size[0], volume.size[1], volume.size[2], nx, ny, nz);
        grid.position.copy(new Vector3(...volume.center));
        // Only the probe-grid cameras see it; the bake cameras (layer 0) must not.
        grid.layers.set(LAYER_PROBE_GRID);
        this.context.loaded.scene.add(grid);
        return grid;
    }

    private replaceGrid(): void {
        this.disposeGrid();
        this.grid = this.createGrid();
        this.atlasBound = false;
    }
}
