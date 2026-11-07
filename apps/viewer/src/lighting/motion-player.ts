/**
 * Plays a manifest's scripted light motions (docs/design/m2-lights.md §7,
 * `motions`): the built-in `light-sweep` (a lamp gliding at 1 m/s and back)
 * and `light-step` (the same lamp jumping every 3 s), or any other motion a
 * manifest lists. Each frame it sets the target light's position, intensity
 * and colour to `motionState` at the motion's time, the same function the
 * native harness plays them with (T2.7), so both play a motion alike.
 *
 * A motion's light is moved in three only: the three adapter sees the change
 * in the same frame and sends it to GI (the first 64 lights are checked every
 * frame). A light the scene loaded switched off (Sponza's arcade lamp, kept
 * out of the scene until a motion plays it: `LoadedScene.offLights`) joins
 * the scene when its motion starts and leaves it when the motion stops,
 * which makes three compile its materials once, each time.
 *
 * Stopping puts the light back as it was before the motion started.
 *
 * A moved light's world matrix is brought up to date at once
 * (`updateLightMatrices`): the adapter reads world matrices as three's last
 * render left them, so without it GI would see each move a frame late, and
 * the response meter would count that frame.
 */

import type { Light, Object3D } from 'three';
import { motionState, type ManifestMotion } from '../../../../scenes/src/manifest';
import type { LoadedScene } from '../scene/load-scene';

/** A light's state a motion can change, to put it back afterwards. */
interface SavedLight {
    position: [number, number, number];
    intensity: number;
    color: [number, number, number];
    inScene: boolean;
}

export class MotionPlayer {
    private current: { motion: ManifestMotion; light: Light; saved: SavedLight; startedAt: number } | null = null;
    /** Seconds into the motion when it was paused, or null while it plays. */
    private pausedAt: number | null = null;

    constructor(private readonly loaded: LoadedScene) {}

    /** The scene's motions whose light exists. */
    get motions(): readonly ManifestMotion[] {
        return (this.loaded.manifest.motions ?? []).filter((motion) => this.lightFor(motion) !== null);
    }

    /** The light a motion moves (`light:<label>`), or null when the scene has no such light. */
    lightFor(motion: ManifestMotion): Light | null {
        const label = /^light:(.+)$/.exec(motion.target)?.[1];
        const index = this.loaded.manifest.lights.findIndex((light) => light.label === label);
        return index >= 0 ? (this.loaded.manifestLights[index] ?? null) : null;
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

    /** Starts `motionId` from its beginning, stopping any other first. */
    play(motionId: string, now: number): void {
        this.stop();
        const motion = this.motions.find((candidate) => candidate.id === motionId);
        if (!motion) return;
        const light = this.lightFor(motion)!;
        const saved: SavedLight = {
            position: [light.position.x, light.position.y, light.position.z],
            intensity: light.intensity,
            color: [light.color.r, light.color.g, light.color.b],
            inScene: light.parent !== null,
        };
        this.current = { motion, light, saved, startedAt: now };
        this.pausedAt = null;
        addToScene(this.loaded, light);
        this.update(now);
    }

    /** Holds the motion where it is (the light stays there). */
    pause(now: number): void {
        if (!this.current || this.pausedAt !== null) return;
        this.pausedAt = this.seconds(now);
    }

    resume(now: number): void {
        if (!this.current || this.pausedAt === null) return;
        this.current.startedAt = now - this.pausedAt * 1000;
        this.pausedAt = null;
    }

    /** Stops the motion and puts its light back as it was. */
    stop(): void {
        const current = this.current;
        this.current = null;
        this.pausedAt = null;
        if (!current) return;
        const { light, saved } = current;
        light.position.set(...saved.position);
        light.intensity = saved.intensity;
        light.color.setRGB(...saved.color);
        updateLightMatrices(light);
        if (!saved.inScene) light.removeFromParent();
    }

    /** Sets the light to the motion's state at `now`. Call once a frame, before GI's update. */
    update(now: number): void {
        const current = this.current;
        if (!current) return;
        applyState(current.light, motionState(current.motion, this.seconds(now)));
    }

    /**
     * Sets a motion's light to its state at `seconds` without playing it (the
     * response meter's step), adding it to the scene if it was left out.
     * Returns the light, or null when the motion has none.
     */
    pose(motion: ManifestMotion, seconds: number): Light | null {
        const light = this.lightFor(motion);
        if (!light) return null;
        addToScene(this.loaded, light);
        applyState(light, motionState(motion, seconds));
        return light;
    }
}

function applyState(light: Light, state: ReturnType<typeof motionState>): void {
    if (state.position) light.position.set(...state.position);
    if (state.intensity !== undefined) light.intensity = state.intensity;
    // Manifest colours are linear, three's working colour space.
    if (state.color) light.color.setRGB(...state.color);
    updateLightMatrices(light);
}

/**
 * Brings a moved light's world matrix (and its target's) up to date now, so
 * GI's update in this frame sees the move: the three adapter reads world
 * matrices as three's last render left them (three-gi.ts), one frame late.
 */
export function updateLightMatrices(light: Light): void {
    light.updateMatrixWorld();
    (light as unknown as { target?: Object3D }).target?.updateMatrixWorld();
}

/** Adds a light left out of the scene (an off light, `LoadedScene.offLights`), with its target. */
export function addToScene(loaded: LoadedScene, light: Light): void {
    if (light.parent) return;
    loaded.scene.add(light);
    const target = (light as unknown as { target?: Object3D }).target;
    if (target && !target.parent) loaded.scene.add(target);
}
