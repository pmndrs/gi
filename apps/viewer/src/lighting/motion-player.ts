/**
 * Plays a manifest's scripted motions (`motions`; docs/design/m2-lights.md
 * §7 for lights, m3-moving.md §8 for objects and the camera): the built-in
 * `light-sweep` (a lamp gliding at 1 m/s and back) and `light-step` (the
 * same lamp jumping every 3 s), a door swinging shut, a cabinet sliding, a
 * tree swaying in the wind, a flight down a street, or any other motion a
 * manifest lists. Each frame it sets the target to `motionState` at the
 * motion's time, the same function the native harness plays light motions
 * with (T2.7), so both play them alike.
 *
 * What a motion moves, and how, is its driver's (`motion-drivers.ts`): a
 * light, an object (moved, turned about a pivot, or swayed by its wind
 * weight; a manifest prop joins the scene while its motion plays), or the
 * camera (only when the viewer hands the player its camera).
 *
 * A motion's light or object is moved in three only: the three adapter sees
 * the change and sends it to GI (the first 64 lights are checked every frame;
 * objects through T3.7's change tracking). A light the scene loaded switched
 * off (Sponza's arcade lamp, kept out of the scene until a motion plays it:
 * `LoadedScene.offLights`) joins the scene when its motion starts and leaves
 * it when the motion stops, which makes three compile its materials once,
 * each time.
 *
 * Stopping puts the target back as it was before the motion started.
 */

import type { Light, Object3D } from 'three';
import { motionState, motionTarget, type ManifestMotion } from '../../../../scenes/src/manifest';
import type { LoadedScene } from '../scene/load-scene';
import { DEFAULT_WIND_STRENGTH } from '../ui/url-state';
import { CameraDriver, LightDriver, makeProp, ObjectDriver, type CameraControl, type MotionDriver } from './motion-drivers';

export { addToScene, restoreLight, saveLight, updateLightMatrices, MAX_WIND_REACH_M, type SavedLight } from './motion-drivers';

export interface MotionPlayerOptions {
    /** The viewing camera, for camera motions (a fly-through). Without it, camera motions aren't offered. */
    readonly camera?: CameraControl;
}

export class MotionPlayer {
    private current: { motion: ManifestMotion; driver: MotionDriver; saved: unknown; startedAt: number } | null = null;
    /** Seconds into the motion when it was paused, or null while it plays. */
    private pausedAt: number | null = null;
    /** One driver per target, shared by the motions that name it, so they all start from the same rest. */
    private readonly drivers = new Map<string, MotionDriver | null>();
    private windStrengthValue = DEFAULT_WIND_STRENGTH;

    constructor(
        private readonly loaded: LoadedScene,
        private readonly options: MotionPlayerOptions = {},
    ) {
        // Swaying meshes are marked deforming now, before GI copies the scene (motion-drivers.ts).
        for (const motion of loaded.manifest.motions ?? []) {
            const driver = this.driverFor(motion);
            if (driver instanceof ObjectDriver && motion.keyframes.some((keyframe) => keyframe.sway !== undefined)) driver.prepareSway();
        }
    }

    /** The scene's motions whose target exists (and, for a sway, whose object has a wind weight). */
    get motions(): readonly ManifestMotion[] {
        return (this.loaded.manifest.motions ?? []).filter((motion) => {
            const driver = this.driverFor(motion);
            if (!driver) return false;
            const sways = motion.keyframes.some((keyframe) => keyframe.sway !== undefined);
            return !sways || (driver instanceof ObjectDriver && driver.canSway);
        });
    }

    /** The light a motion moves (`light:<label>`), or null when it moves something else or the scene has no such light. */
    lightFor(motion: ManifestMotion): Light | null {
        const driver = this.driverFor(motion);
        return driver instanceof LightDriver ? driver.light : null;
    }

    /** True when some motion of the scene sways a tree, so the wind slider means something. */
    get hasWind(): boolean {
        return this.motions.some((motion) => motion.keyframes.some((keyframe) => keyframe.sway !== undefined));
    }

    /** The wind slider: the share of `MAX_WIND_REACH_M` a sway of 1 moves the crown, 0 to 1. */
    get windStrength(): number {
        return this.windStrengthValue;
    }

    set windStrength(strength: number) {
        this.windStrengthValue = Math.min(1, Math.max(0, strength));
        // A tree held still (paused, or left leaning by the meter) shows the new strength at once.
        for (const driver of this.drivers.values()) if (driver instanceof ObjectDriver) driver.refreshSway();
    }

    /** The motion playing (or paused), or null. */
    get playing(): ManifestMotion | null {
        return this.current?.motion ?? null;
    }

    get paused(): boolean {
        return this.pausedAt !== null;
    }

    /** Seconds since the motion started (paused time not counted). */
    seconds(now: number): number {
        const current = this.current;
        if (!current) return 0;
        return this.pausedAt ?? (now - current.startedAt) / 1000;
    }

    /** What GI should be told a motion moved (`markChanged`), or null (the camera). */
    changedBy(motion: ManifestMotion): Object3D | null {
        return this.driverFor(motion)?.changed ?? null;
    }

    /** Starts `motionId` from its beginning, stopping any other first. */
    play(motionId: string, now: number): void {
        this.stop();
        const motion = this.motions.find((candidate) => candidate.id === motionId);
        if (!motion) return;
        const driver = this.driverFor(motion)!;
        this.current = { motion, driver, saved: driver.save(), startedAt: now };
        this.pausedAt = null;
        this.update(now);
    }

    /** Holds the motion where it is (its target stays there). */
    pause(now: number): void {
        if (!this.current || this.pausedAt !== null) return;
        this.pausedAt = this.seconds(now);
    }

    resume(now: number): void {
        if (!this.current || this.pausedAt === null) return;
        this.current.startedAt = now - this.pausedAt * 1000;
        this.pausedAt = null;
    }

    /** Stops the motion and puts its target back as it was. */
    stop(): void {
        const current = this.current;
        this.current = null;
        this.pausedAt = null;
        if (!current) return;
        current.driver.restore(current.saved);
    }

    /** Sets the target to the motion's state at `now`. Call once a frame, before GI's update. */
    update(now: number): void {
        const current = this.current;
        if (!current) return;
        current.driver.apply(motionState(current.motion, this.seconds(now)), current.motion);
    }

    /**
     * Sets a motion's target to its state at `seconds` without playing it (the
     * response meter's step), adding a light or prop left out of the scene.
     * Returns what GI should be told moved, or null.
     */
    pose(motion: ManifestMotion, seconds: number): Object3D | null {
        const driver = this.driverFor(motion);
        if (!driver) return null;
        driver.apply(motionState(motion, seconds), motion);
        return driver.changed;
    }

    /** How a motion's target is now, for `restore` (the response meter puts it back after its step). */
    save(motion: ManifestMotion): unknown {
        return this.driverFor(motion)?.save() ?? null;
    }

    restore(motion: ManifestMotion, saved: unknown): void {
        if (saved !== null) this.driverFor(motion)?.restore(saved);
    }

    /** Stops any motion and lets go of the props it made. */
    dispose(): void {
        this.stop();
        for (const driver of this.drivers.values()) if (driver instanceof ObjectDriver) driver.disposeProp();
        this.drivers.clear();
    }

    /** The driver for a motion's target, made the first time it is asked for; null when the target isn't there. */
    private driverFor(motion: ManifestMotion): MotionDriver | null {
        const known = this.drivers.get(motion.target);
        if (known !== undefined) return known;
        const driver = this.makeDriver(motion);
        this.drivers.set(motion.target, driver);
        return driver;
    }

    private makeDriver(motion: ManifestMotion): MotionDriver | null {
        const target = motionTarget(motion);
        if (!target) return null;
        const loaded = this.loaded;
        switch (target.kind) {
            case 'light': {
                const index = loaded.manifest.lights.findIndex((light) => light.label === target.name);
                const light = index >= 0 ? loaded.manifestLights[index] : undefined;
                return light ? new LightDriver(loaded, light) : null;
            }
            case 'object': {
                const strength = () => this.windStrengthValue;
                const prop = loaded.manifest.props?.find((candidate) => candidate.name === target.name);
                if (prop) return new ObjectDriver(makeProp(prop), loaded.content, strength);
                const object = loaded.content.getObjectByName(target.name);
                return object ? new ObjectDriver(object, null, strength) : null;
            }
            case 'camera':
                return this.options.camera ? new CameraDriver(this.options.camera) : null;
        }
    }
}
