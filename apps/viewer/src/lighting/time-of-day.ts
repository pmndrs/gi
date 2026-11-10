/**
 * Time of day in the viewer (docs/design/m2-lights.md §7): an hour and a day
 * of the year move the sun, and three things follow it together:
 *
 * 1. **the scene's sun light** (three's directional light, the manifest's
 *    sun): its direction, and its brightness faded towards night;
 * 2. **GI's sky**: a `procedural` environment for the new sun, which the
 *    platform bakes again on the GPU (T2.6a) and marks the probes it
 *    changes; the sun light's own change reaches GI through the adapter;
 * 3. **the sky the camera sees**: the same calibrated sky the shared sky
 *    file holds (`buildSkyImage`, the file's own recipe), built on the CPU at
 *    a lower resolution for the new sun and swapped onto the sky dome. Only
 *    when the sun moved by DOME_REBUILD_DEGREES or the light faded, and at
 *    most every DOME_REBUILD_MS, so a 10-second day doesn't build one a frame.
 *
 * The compass is turned so the time it opens with puts the sun exactly where
 * the manifest has it (`fitSunToManifest`); turning time of day off puts the
 * manifest's sun, sky and sky picture back.
 *
 * The sun layer: the viewer turns DDGI's sun layer on with time of day
 * (DECISIONS 2026-11-08), so the sun's fading brightness is a weight change
 * in GI, with nothing traced again; its direction still re-traces.
 */

import { DataTexture, EquirectangularReflectionMapping, HalfFloatType, LinearFilter, RepeatWrapping, RGBAFormat, Vector3, type DirectionalLight, type Texture } from 'three';
import type { EnvironmentDescriptor } from '../ddgi/library';
import type { ManifestLight } from '../../../../scenes/src/manifest';
import { buildSkyImage, platformSkyEnvironment } from '../../../../scenes/src/sky/sky-image';
import { skyImageToRgbaHalves } from '../../../../scenes/src/sky/sky-texture';
import type { LoadedScene } from '../scene/load-scene';
import { setSkyDomeTexture } from '../scene/sky';
import { fitSunToManifest, hoursAtElevation, timeOfDaySun, type TimeOfDayFit, type TimeOfDaySun } from '../scene/solar-position';

/** The sky picture's size while time of day runs: the platform's own sky map size (256 x 128). */
const DOME_WIDTH = 256;
const DOME_HEIGHT = 128;
/** Build a new sky picture when the sun moved by more than this... */
const DOME_REBUILD_DEGREES = 0.5;
/** ...or its light faded by more than this share... */
const DOME_REBUILD_DAYLIGHT = 0.02;
/** ...and not more often than this. */
const DOME_REBUILD_MS = 100;
/** The play button's day, sunrise to sunset, in seconds (m2-lights.md §7: "a 10-second day"). */
export const PLAYED_DAY_SECONDS = 10;
/** The played day starts and ends with the sun this far below the horizon (degrees), so it rises and sets in view. */
const PLAYED_DAY_HORIZON_DEG = -2;

export interface TimeOfDayState {
    enabled: boolean;
    hour: number;
    dayOfYear: number;
}

export class TimeOfDay {
    /** The manifest's sun (its first directional light), or null when the scene has none or no procedural sky. */
    readonly sun: DirectionalLight | null;
    /** Where the manifest's sun is: the time the controls open at, and the compass turn. */
    readonly fit: TimeOfDayFit | null;
    hour: number;
    dayOfYear: number;
    private enabledFlag = false;
    private readonly baseIntensity: number;
    private readonly baseDirection = new Vector3();
    private readonly sunDistance: number;
    private readonly fileSky: Texture | null;
    private domeSky: DataTexture | null = null;
    private lastDome: { towardSun: [number, number, number]; daylight: number; at: number } | null = null;
    private play: { startedAt: number; fromHour: number; toHour: number } | null = null;

    constructor(private readonly loaded: LoadedScene) {
        const manifest = loaded.manifest;
        const index = manifest.lights.findIndex((light: ManifestLight) => light.type === 'directional');
        const sun = index >= 0 && manifest.environment.type === 'procedural' ? (loaded.manifestLights[index] as DirectionalLight) : null;
        this.sun = sun?.isDirectionalLight ? sun : null;
        this.fileSky = loaded.sky;
        if (this.sun) {
            this.baseIntensity = this.sun.intensity;
            this.baseDirection.subVectors(this.sun.position, this.sun.target.position).normalize();
            this.sunDistance = this.sun.position.distanceTo(this.sun.target.position);
            this.fit = fitSunToManifest([this.baseDirection.x, this.baseDirection.y, this.baseDirection.z]);
        } else {
            this.baseIntensity = 0;
            this.sunDistance = 0;
            this.fit = null;
        }
        this.hour = this.fit?.hour ?? 12;
        this.dayOfYear = this.fit?.dayOfYear ?? 172;
    }

    /** Whether this scene has a sun and a sky to move. */
    get available(): boolean {
        return this.sun !== null && this.fit !== null;
    }

    get enabled(): boolean {
        return this.enabledFlag;
    }

    get playing(): boolean {
        return this.play !== null;
    }

    /** Where the sun is for the current hour and day. */
    current(): TimeOfDaySun | null {
        return this.fit ? timeOfDaySun(this.hour, this.dayOfYear, this.fit.northDeg) : null;
    }

    /**
     * Turns time of day on (the sun follows the hour and day) or off (the
     * manifest's sun and sky come back). Returns GI's environment for the new
     * state: null means the manifest's own.
     */
    setEnabled(enabled: boolean, now: number): EnvironmentDescriptor | null {
        if (!this.available) return null;
        this.enabledFlag = enabled;
        if (!enabled) {
            this.play = null;
            this.restore();
            return null;
        }
        return this.apply(now, true);
    }

    /** A new hour and day (the sliders). Returns GI's environment, or null while time of day is off. */
    setTime(hour: number, dayOfYear: number, now: number): EnvironmentDescriptor | null {
        this.hour = hour;
        this.dayOfYear = dayOfYear;
        return this.enabledFlag ? this.apply(now, false) : null;
    }

    /** Plays a day from just before sunrise to just after sunset in PLAYED_DAY_SECONDS. */
    playDay(now: number): void {
        const hours = hoursAtElevation(PLAYED_DAY_HORIZON_DEG, this.dayOfYear);
        this.play = hours ? { startedAt: now, fromHour: hours.morning, toHour: hours.evening } : { startedAt: now, fromHour: 0, toHour: 24 };
    }

    stopPlaying(): void {
        this.play = null;
    }

    /**
     * Advances a played day. Returns GI's environment when the sun moved this
     * frame, undefined when nothing changed.
     */
    update(now: number): EnvironmentDescriptor | null | undefined {
        const play = this.play;
        if (!play || !this.enabledFlag) return undefined;
        const share = Math.min(1, (now - play.startedAt) / 1000 / PLAYED_DAY_SECONDS);
        this.hour = play.fromHour + (play.toHour - play.fromHour) * share;
        if (share >= 1) this.play = null;
        return this.apply(now, false);
    }

    /** Frees the sky picture it built. */
    dispose(): void {
        this.domeSky?.dispose();
        this.domeSky = null;
    }

    /** The sun light, GI's sky and the sky picture for the current time. */
    private apply(now: number, forceDome: boolean): EnvironmentDescriptor {
        const sun = this.sun!;
        const state = this.current()!;
        const toward = new Vector3(...state.towardSun);
        sun.position.copy(sun.target.position).addScaledVector(toward, this.sunDistance);
        sun.intensity = this.baseIntensity * state.daylight;
        // GI reads world matrices as three's last render left them: bring the sun's up to date for this frame's update.
        sun.updateMatrixWorld();
        const manifestSky = platformSkyEnvironment(this.loaded.manifest);
        if (manifestSky.type !== 'procedural') throw new Error('time of day needs a procedural sky');
        const environment: EnvironmentDescriptor = { ...manifestSky, sunDirection: state.towardSun, intensity: (manifestSky.intensity ?? 1) * state.daylight };
        this.updateDome(state, now, forceDome);
        return environment;
    }

    /** Builds the sky picture for the sun's new place when it moved enough (see the file comment). */
    private updateDome(state: TimeOfDaySun, now: number, force: boolean): void {
        const dome = this.loaded.skyDome;
        if (!dome) return;
        const last = this.lastDome;
        if (!force && last) {
            const cosine = last.towardSun[0] * state.towardSun[0] + last.towardSun[1] * state.towardSun[1] + last.towardSun[2] * state.towardSun[2];
            const moved = cosine < Math.cos((DOME_REBUILD_DEGREES * Math.PI) / 180) || Math.abs(last.daylight - state.daylight) > DOME_REBUILD_DAYLIGHT;
            // The last picture of a played day (or a slider let go) still gets built: the next frame asks again.
            if (!moved || now - last.at < DOME_REBUILD_MS) return;
        }
        this.lastDome = { towardSun: state.towardSun, daylight: state.daylight, at: now };
        const manifest = this.loaded.manifest;
        const environment = manifest.environment;
        if (environment.type !== 'procedural') return;
        const sunLight = manifest.lights.find((light) => light.type === 'directional');
        if (!sunLight || sunLight.type !== 'directional') return;
        // The shared sky file's own recipe (calibrated Preetham) for the new sun, at the platform's map size.
        const image = buildSkyImage(
            {
                environment: { ...environment, sunDirection: state.towardSun, intensity: (environment.intensity ?? 1) * state.daylight },
                lights: [{ ...sunLight, direction: [-state.towardSun[0], -state.towardSun[1], -state.towardSun[2]] }],
            },
            { width: DOME_WIDTH, height: DOME_HEIGHT },
        ).image;
        const halves = skyImageToRgbaHalves(image);
        if (!this.domeSky) {
            this.domeSky = new DataTexture(halves, image.width, image.height, RGBAFormat, HalfFloatType);
            this.domeSky.name = `time of day sky ${manifest.id}`;
            this.domeSky.mapping = EquirectangularReflectionMapping;
            this.domeSky.wrapS = RepeatWrapping;
            this.domeSky.minFilter = LinearFilter;
            this.domeSky.magFilter = LinearFilter;
            this.domeSky.generateMipmaps = false;
        } else {
            (this.domeSky.image.data as Uint16Array).set(halves);
        }
        this.domeSky.needsUpdate = true;
        setSkyDomeTexture(dome, this.domeSky);
    }

    /** The manifest's sun and sky picture again. */
    private restore(): void {
        const sun = this.sun;
        if (!sun) return;
        sun.position.copy(sun.target.position).addScaledVector(this.baseDirection, this.sunDistance);
        sun.intensity = this.baseIntensity;
        sun.updateMatrixWorld();
        if (this.loaded.skyDome && this.fileSky) setSkyDomeTexture(this.loaded.skyDome, this.fileSky);
        this.lastDome = null;
    }
}
