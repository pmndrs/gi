/**
 * Scene manifests: one plain JSON file per test scene in `scenes/manifests/`.
 *
 * The same file is read by the TypeScript viewer, by PowerShell (the native
 * harness) and by Python (the Blender Cycles reference), so it stays plain
 * JSON: no comments, no references, no computed values. Angles are degrees,
 * lengths are metres, colours are linear-light RGB.
 *
 * Where the geometry lives:
 * - `procedural` parts are built by code (`scenes/src/procedural`); `pnpm
 *   scenes:export` writes them to `assets/generated/<scene id>.glb`.
 * - `model` parts are files downloaded by `pnpm assets:fetch` into
 *   `assets/downloads/<asset>/<path>`. `asset` is an id in
 *   `tools/assets/sources.json`, and `path` is one of that asset's outputs,
 *   relative to its download folder.
 * - `light-set` parts (T2.1, `sponza-candles`) are one model copied at each
 *   of a set of point lights, from a set file a convert step writes (like
 *   `tools/assets/convert/sponza-candles.ts`). How many are loaded is a scene
 *   setting (`settings`), and the first N of the set's own order are taken.
 *
 * `motions` (T2.1, docs/design/m2-lights.md §7) script a light over time:
 * keyframes of its position, intensity and colour. Two come built in for
 * each scene with a lamp: `light-sweep` (the lamp gliding along a path at
 * 1 m/s, and back) and `light-step` (the same lamp jumping between two
 * places every 3 s, for measuring response). The native harness (T2.7) and
 * the viewer (T2.8) play them; `motionState` says what a motion's light is
 * at a time, so both play them the same way.
 *
 * This file holds the types and the validator. It has no dependencies, so any
 * tool can import it.
 */

import type { SceneLightData } from './scene-data';

export type Vec3 = [number, number, number];

export interface ProceduralPart {
    kind: 'procedural';
    /** A name from `proceduralBuilders` in `scenes/src/procedural/index.ts`. */
    builder: string;
    /** Passed to the builder, for example `{ "light": "point" }`. */
    options?: Record<string, string | number | boolean>;
}

export interface ModelTransform {
    /** Metres, applied last. */
    position: Vec3;
    /** Euler XYZ in degrees. */
    rotationDeg: Vec3;
    /** Uniform scale that brings the model to metres. */
    scale: number;
}

export interface ModelPart {
    kind: 'model';
    /** An asset id from `tools/assets/sources.json`. */
    asset: string;
    /** Output file, relative to `assets/downloads/<asset>/`. */
    path: string;
    transform: ModelTransform;
}

/**
 * One model copied at each of a set of point lights (T2.1). The set file
 * (`LightSetFile`, a convert step's output) names the model, the light's
 * colour and intensity, and the lights' positions in its selection order.
 */
export interface LightSetPart {
    kind: 'light-set';
    /** An asset id from `tools/assets/sources.json`. */
    asset: string;
    /** The set file, relative to `assets/downloads/<asset>/`; one of the asset's outputs. */
    path: string;
    /** The scene setting (a key of `settings`) that says how many to load: the first N of the set's order. */
    countSetting: string;
}

export type ScenePart = ProceduralPart | ModelPart | LightSetPart;

/** A whole-number knob of a scene, like `sponza-candles`' candle count. A viewer may offer it; a link may set it. */
export interface SceneSetting {
    /** Plain words, for a slider. */
    label: string;
    /** One sentence: what changing it does. */
    description: string;
    default: number;
    min: number;
    max: number;
}

/** What a `light-set` part's file holds (tools/assets/convert/sponza-candles.ts writes it). */
export interface LightSetFile {
    description: string;
    /** The package file it was made from. */
    source: string;
    /** The model copied at each light, relative to the set file's folder; placed so its light is at its origin. */
    prototype: string;
    count: number;
    trianglesPerCandle: number;
    /** The selection rule, in words. */
    order: string;
    light: { type: 'point'; color: Vec3; intensity: number; range: number };
    /** x, y, z of each light in metres, in the selection order. */
    positions: number[];
    packageIndices: number[];
}

/** One keyframe of a motion. Every keyframe of a motion sets the same fields. */
export interface MotionKeyframe {
    /** Seconds from the motion's start; the first is 0, and they increase. */
    t: number;
    /** Metres (point and spot lights). */
    position?: Vec3;
    /** In the light's own units (candela for point and spot, lux for directional). */
    intensity?: number;
    /** Linear RGB. */
    color?: Vec3;
}

/** A scripted change of one light over time (docs/design/m2-lights.md §7). */
export interface ManifestMotion {
    /** Kebab-case, unique in the scene: `light-sweep`, `light-step`. */
    id: string;
    /** Plain words for a menu. */
    label: string;
    /** `light:<label>`: the manifest light with that label. */
    target: string;
    /**
     * `linear` glides from keyframe to keyframe; `step` holds each keyframe
     * until the next one's time, so the light jumps (a step response).
     * Defaults to `linear`.
     */
    interpolation?: 'linear' | 'step';
    /** Plays again from the start after the last keyframe. A looping motion ends where it began. */
    loop: boolean;
    keyframes: MotionKeyframe[];
}

/** The motion ids every scene with a lamp has (m2-lights.md §7). */
export const BUILT_IN_MOTIONS = { sweep: 'light-sweep', step: 'light-step' } as const;
/** The built-in sweep's speed, m/s, and the built-in step's time between jumps, seconds (m2-lights.md §7). */
export const SWEEP_SPEED = 1;
export const STEP_SECONDS = 3;

export interface CameraBookmark {
    name: string;
    position: Vec3;
    target: Vec3;
    /** Vertical field of view in degrees. */
    fovYDeg: number;
}

interface ManifestLightCommon {
    label?: string;
    /** Linear RGB. */
    color: Vec3;
    /** Directional: lux. Point and spot: candela. These are three.js physical units. */
    intensity: number;
    /** Defaults to true. */
    castsShadows?: boolean;
}

export interface ManifestDirectionalLight extends ManifestLightCommon {
    type: 'directional';
    /** Direction the light travels. */
    direction: Vec3;
}

/**
 * `range` on point and spot lights is required. A positive value is the
 * distance at which the light is faded to zero (three.js `distance`; glTF
 * `range`). 0 means no range window at all: plain inverse-square falloff,
 * three.js `distance = 0`, which is what Cycles does, so use 0 where a scene
 * must match a Cycles reference. It is 0 rather than left out so that a
 * forgotten field is still an error.
 */
export const NO_RANGE_WINDOW = 0;

export interface ManifestPointLight extends ManifestLightCommon {
    type: 'point';
    position: Vec3;
    /** Metres at which the light reaches zero, or 0 (`NO_RANGE_WINDOW`) for plain inverse-square falloff. */
    range: number;
}

export interface ManifestSpotLight extends ManifestLightCommon {
    type: 'spot';
    position: Vec3;
    direction: Vec3;
    /** Metres at which the light reaches zero, or 0 (`NO_RANGE_WINDOW`) for plain inverse-square falloff. */
    range: number;
    /** Half-angle where falloff starts, degrees. */
    innerConeDeg: number;
    /** Half-angle where the light reaches zero, degrees. */
    outerConeDeg: number;
}

export type ManifestLight = ManifestDirectionalLight | ManifestPointLight | ManifestSpotLight;

export type ManifestEnvironment =
    | { type: 'none' }
    | { type: 'color'; color: Vec3; intensity?: number }
    | {
          /**
           * A clear sky, made into the shared sky file `scenes/sky/<scene id>.exr`
           * by `tools/sky/generate.ts`. Every renderer reads that file.
           */
          type: 'procedural';
          /** Direction toward the sun. */
          sunDirection: Vec3;
          /** Preetham turbidity, about 2 (very clear) to 10 (hazy). */
          turbidity?: number;
          /** Multiplier on the calibrated sky; 1 (the default) is as calibrated. */
          intensity?: number;
          /**
           * Only for a sky-only scene, with no directional light along -sunDirection:
           * the illuminance (lux) of the sun this sky belongs to, which sets the
           * sky's brightness. Scenes with a sun light take it from that light.
           */
          sunIlluminance?: number;
      };

/** A suggested probe volume. */
export interface ManifestVolume {
    center: Vec3;
    size: Vec3;
    /** Metres between probes. */
    probeSpacing: number;
}

export interface ManifestCredit {
    /** The `sources.json` id, or null for things made in this repo. */
    asset: string | null;
    /** SPDX id, or a short name for a custom license. */
    license: string;
    /** Text to show. */
    attribution: string;
    sourceUrl?: string;
}

export interface SceneManifest {
    id: string;
    title: string;
    /** One sentence, plain language: what the scene is for. */
    description: string;
    tags: string[];
    units: 'meters';
    upAxis: 'y';
    /** `awaiting-asset`: the scene is a placeholder until its download is confirmed. Absent means ready. */
    status?: 'ready' | 'awaiting-asset';
    parts: ScenePart[];
    /** 2 to 4 lit, useful views. The first is the one a viewer opens on. */
    cameras: CameraBookmark[];
    /** Direct lights. Emissive surfaces in the geometry are not listed here, nor a `light-set` part's lights. */
    lights: ManifestLight[];
    /** Scripted light changes (the file comment). */
    motions?: ManifestMotion[];
    /** Whole-number knobs a `light-set` part reads (`countSetting`), by name. */
    settings?: Record<string, SceneSetting>;
    environment: ManifestEnvironment;
    volume: ManifestVolume;
    /** True when a glTF/GLB exists for the native harness. */
    nativeCapable: boolean;
    /** Where the check page puts its 1 m cube. Defaults to the first camera's target. */
    referenceCube?: { position: Vec3 };
    /**
     * The camera exposure a viewer opens this scene with, in photographic
     * stops (0 = normal, +1 = twice as bright). Absent means 0. The lighting
     * stays physical (DECISIONS 2026-11-05: per-scene exposure, the sky stays
     * calibrated); this only says how bright the picture starts.
     */
    exposureEv?: number;
    credits: ManifestCredit[];
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

export interface ValidationResult {
    ok: boolean;
    /** One readable line per problem, each starting with the path of the bad field. */
    errors: string[];
}

export interface ValidateOptions {
    /** If given, procedural part builder names must be in this list. */
    knownBuilders?: readonly string[];
}

const TOP_LEVEL_KEYS = new Set([
    '$schema',
    'id',
    'title',
    'description',
    'tags',
    'units',
    'upAxis',
    'status',
    'parts',
    'cameras',
    'lights',
    'motions',
    'settings',
    'environment',
    'volume',
    'nativeCapable',
    'referenceCube',
    'exposureEv',
    'credits',
]);

/** The most a scene's default exposure may brighten or darken, in stops. A wider default would be hiding a lighting problem. */
export const MAX_DEFAULT_EXPOSURE_EV = 8;

const KEBAB_CASE = /^[a-z0-9]+(-[a-z0-9]+)*$/;

type Json = unknown;

function isRecord(value: Json): value is Record<string, Json> {
    return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function isFiniteNumber(value: Json): value is number {
    return typeof value === 'number' && Number.isFinite(value);
}

function isVec3(value: Json): value is Vec3 {
    return Array.isArray(value) && value.length === 3 && value.every(isFiniteNumber);
}

function vectorLength(v: Vec3): number {
    return Math.hypot(v[0], v[1], v[2]);
}

function isNonEmptyString(value: Json): value is string {
    return typeof value === 'string' && value.trim().length > 0;
}

/** Collects errors with the path of the field they belong to. */
class ErrorCollector {
    readonly errors: string[] = [];

    add(path: string, message: string): void {
        this.errors.push(`${path}: ${message}`);
    }

    vec3(path: string, value: Json): value is Vec3 {
        if (!isVec3(value)) {
            this.add(path, 'must be an array of three finite numbers');
            return false;
        }
        return true;
    }

    color(path: string, value: Json): void {
        if (this.vec3(path, value) && value.some((channel) => channel < 0)) {
            this.add(path, 'colour channels must not be negative');
        }
    }

    direction(path: string, value: Json): void {
        if (this.vec3(path, value) && vectorLength(value) < 1e-6) {
            this.add(path, 'must not be a zero vector');
        }
    }

    /** A point or spot light's range: a positive distance, or 0 for no range window. */
    lightRange(path: string, value: Json): void {
        if (!isFiniteNumber(value) || value < 0) {
            this.add(path, `must be a distance in metres, or ${NO_RANGE_WINDOW} for no range window (plain inverse-square falloff)`);
        }
    }

    positiveNumber(path: string, value: Json): value is number {
        if (!isFiniteNumber(value) || value <= 0) {
            this.add(path, 'must be a number greater than 0');
            return false;
        }
        return true;
    }
}

/** A model path must stay inside the asset's download folder. */
function isSafeRelativePath(path: string): boolean {
    if (path.startsWith('/') || path.startsWith('\\') || /^[a-zA-Z]:/.test(path)) return false;
    return !path.split(/[\\/]/).includes('..');
}

function validatePart(part: Json, path: string, c: ErrorCollector, options: ValidateOptions): void {
    if (!isRecord(part)) {
        c.add(path, 'must be an object');
        return;
    }
    if (part.kind === 'procedural') {
        if (!isNonEmptyString(part.builder)) {
            c.add(`${path}.builder`, 'must be a non-empty string');
        } else if (options.knownBuilders && !options.knownBuilders.includes(part.builder)) {
            c.add(`${path}.builder`, `unknown builder "${part.builder}"; known: ${options.knownBuilders.join(', ')}`);
        }
        if (part.options !== undefined) {
            if (!isRecord(part.options)) {
                c.add(`${path}.options`, 'must be an object');
            } else {
                for (const [key, value] of Object.entries(part.options)) {
                    if (!['string', 'number', 'boolean'].includes(typeof value)) {
                        c.add(`${path}.options.${key}`, 'must be a string, number or boolean');
                    }
                }
            }
        }
    } else if (part.kind === 'model') {
        if (!isNonEmptyString(part.asset) || !KEBAB_CASE.test(part.asset)) {
            c.add(`${path}.asset`, 'must be a kebab-case asset id from tools/assets/sources.json');
        }
        if (!isNonEmptyString(part.path) || !isSafeRelativePath(part.path)) {
            c.add(`${path}.path`, 'must be a relative path inside the asset download folder (no "..", no drive letter)');
        }
        if (!isRecord(part.transform)) {
            c.add(`${path}.transform`, 'must be an object with position, rotationDeg and scale');
        } else {
            c.vec3(`${path}.transform.position`, part.transform.position);
            c.vec3(`${path}.transform.rotationDeg`, part.transform.rotationDeg);
            c.positiveNumber(`${path}.transform.scale`, part.transform.scale);
        }
    } else if (part.kind === 'light-set') {
        if (!isNonEmptyString(part.asset) || !KEBAB_CASE.test(part.asset)) {
            c.add(`${path}.asset`, 'must be a kebab-case asset id from tools/assets/sources.json');
        }
        if (!isNonEmptyString(part.path) || !isSafeRelativePath(part.path)) {
            c.add(`${path}.path`, 'must be a relative path inside the asset download folder (no "..", no drive letter)');
        }
        if (!isNonEmptyString(part.countSetting)) c.add(`${path}.countSetting`, 'must name one of the scene\'s settings');
    } else {
        c.add(`${path}.kind`, 'must be "procedural", "model" or "light-set"');
    }
}

function validateSettings(value: Record<string, Json>, c: ErrorCollector): void {
    const settings = value.settings;
    if (settings !== undefined) {
        if (!isRecord(settings)) {
            c.add('settings', 'must be an object of named settings');
        } else {
            for (const [name, setting] of Object.entries(settings)) {
                const path = `settings.${name}`;
                if (!KEBAB_CASE.test(name)) c.add(path, 'a setting name must be kebab-case');
                if (!isRecord(setting)) {
                    c.add(path, 'must be an object with label, description, default, min and max');
                    continue;
                }
                if (!isNonEmptyString(setting.label)) c.add(`${path}.label`, 'must be a non-empty string');
                if (!isNonEmptyString(setting.description)) c.add(`${path}.description`, 'must be a non-empty sentence');
                const whole = ['default', 'min', 'max'].every((key) => Number.isInteger(setting[key]));
                if (!whole) c.add(path, 'default, min and max must be whole numbers');
                else if (!((setting.min as number) <= (setting.default as number) && (setting.default as number) <= (setting.max as number))) c.add(path, 'needs min <= default <= max');
            }
        }
    }
    const parts = Array.isArray(value.parts) ? value.parts : [];
    parts.forEach((part: Json, index: number) => {
        if (isRecord(part) && part.kind === 'light-set' && isNonEmptyString(part.countSetting) && !(isRecord(settings) && part.countSetting in settings)) {
            c.add(`parts[${index}].countSetting`, `"${part.countSetting}" is not one of the scene settings`);
        }
    });
}

function validateMotions(value: Record<string, Json>, c: ErrorCollector): void {
    const motions = value.motions;
    if (motions === undefined) return;
    if (!Array.isArray(motions)) {
        c.add('motions', 'must be an array');
        return;
    }
    const lights = Array.isArray(value.lights) ? value.lights.filter(isRecord) : [];
    const ids = new Set<string>();
    motions.forEach((motion: Json, index: number) => {
        const path = `motions[${index}]`;
        if (!isRecord(motion)) {
            c.add(path, 'must be an object');
            return;
        }
        if (!isNonEmptyString(motion.id) || !KEBAB_CASE.test(motion.id)) c.add(`${path}.id`, 'must be kebab-case, like "light-sweep"');
        else if (ids.has(motion.id)) c.add(`${path}.id`, `"${motion.id}" is used twice`);
        else ids.add(motion.id);
        if (!isNonEmptyString(motion.label)) c.add(`${path}.label`, 'must be a non-empty string');
        if (typeof motion.loop !== 'boolean') c.add(`${path}.loop`, 'must be true or false');
        if (motion.interpolation !== undefined && motion.interpolation !== 'linear' && motion.interpolation !== 'step') c.add(`${path}.interpolation`, 'must be "linear" or "step"');
        const target = typeof motion.target === 'string' ? /^light:(.+)$/.exec(motion.target) : null;
        // Labels are optional and need not be unique, so the target must name exactly one light.
        const named = target ? lights.filter((candidate) => candidate.label === target[1]) : [];
        const light = named.length === 1 ? named[0] : undefined;
        if (!target) c.add(`${path}.target`, 'must be "light:<label>", naming one of the manifest\'s lights by its label');
        else if (named.length === 0) c.add(`${path}.target`, `no light has the label "${target[1]}"`);
        else if (named.length > 1) c.add(`${path}.target`, `${named.length} lights have the label "${target[1]}"; a motion must name exactly one`);
        if (!Array.isArray(motion.keyframes) || motion.keyframes.length < 2) {
            c.add(`${path}.keyframes`, 'must have at least two keyframes');
            return;
        }
        const fields = (keyframe: Record<string, Json>) => ['position', 'intensity', 'color'].filter((key) => keyframe[key] !== undefined).join(',');
        let previousTime = -Infinity;
        let firstFields: string | null = null;
        motion.keyframes.forEach((keyframe: Json, frame: number) => {
            const where = `${path}.keyframes[${frame}]`;
            if (!isRecord(keyframe)) {
                c.add(where, 'must be an object');
                return;
            }
            if (!isFiniteNumber(keyframe.t)) c.add(`${where}.t`, 'must be a number of seconds');
            else if (frame === 0 && keyframe.t !== 0) c.add(`${where}.t`, 'the first keyframe is at 0 s');
            else if (!(keyframe.t > previousTime)) c.add(`${where}.t`, 'must be later than the keyframe before');
            if (isFiniteNumber(keyframe.t)) previousTime = keyframe.t;
            for (const key of Object.keys(keyframe)) if (!['t', 'position', 'intensity', 'color'].includes(key)) c.add(`${where}.${key}`, 'is not a keyframe field (t, position, intensity, color)');
            if (keyframe.position !== undefined) {
                c.vec3(`${where}.position`, keyframe.position);
                if (light && light.type === 'directional') c.add(`${where}.position`, 'a directional light has no position');
            }
            if (keyframe.intensity !== undefined && !(isFiniteNumber(keyframe.intensity) && keyframe.intensity >= 0)) c.add(`${where}.intensity`, 'must be a number of 0 or more');
            if (keyframe.color !== undefined) c.color(`${where}.color`, keyframe.color);
            const these = fields(keyframe);
            if (these === '') c.add(where, 'must set at least one of position, intensity and color');
            if (firstFields === null) firstFields = these;
            else if (these !== firstFields) c.add(where, `must set the same fields as the first keyframe (${firstFields})`);
        });
    });
}

function validateCamera(camera: Json, path: string, c: ErrorCollector): void {
    if (!isRecord(camera)) {
        c.add(path, 'must be an object');
        return;
    }
    if (!isNonEmptyString(camera.name)) c.add(`${path}.name`, 'must be a non-empty string');
    const position = camera.position;
    const target = camera.target;
    const positionOk = c.vec3(`${path}.position`, position);
    const targetOk = c.vec3(`${path}.target`, target);
    if (positionOk && targetOk) {
        const distance = Math.hypot(position[0] - target[0], position[1] - target[1], position[2] - target[2]);
        if (distance < 1e-3) c.add(path, 'position and target must not be the same point');
    }
    if (!isFiniteNumber(camera.fovYDeg) || camera.fovYDeg < 5 || camera.fovYDeg > 120) {
        c.add(`${path}.fovYDeg`, 'must be a number between 5 and 120');
    }
}

function validateLight(light: Json, path: string, c: ErrorCollector): void {
    if (!isRecord(light)) {
        c.add(path, 'must be an object');
        return;
    }
    c.color(`${path}.color`, light.color);
    if (!isFiniteNumber(light.intensity) || light.intensity < 0) {
        c.add(`${path}.intensity`, 'must be a number of 0 or more');
    }
    if (light.castsShadows !== undefined && typeof light.castsShadows !== 'boolean') {
        c.add(`${path}.castsShadows`, 'must be true or false');
    }
    switch (light.type) {
        case 'directional':
            c.direction(`${path}.direction`, light.direction);
            break;
        case 'point':
            c.vec3(`${path}.position`, light.position);
            c.lightRange(`${path}.range`, light.range);
            break;
        case 'spot': {
            c.vec3(`${path}.position`, light.position);
            c.direction(`${path}.direction`, light.direction);
            c.lightRange(`${path}.range`, light.range);
            const innerOk = isFiniteNumber(light.innerConeDeg) && light.innerConeDeg >= 0;
            const outerOk = isFiniteNumber(light.outerConeDeg) && light.outerConeDeg > 0 && light.outerConeDeg <= 90;
            if (!innerOk) c.add(`${path}.innerConeDeg`, 'must be a number of 0 or more');
            if (!outerOk) c.add(`${path}.outerConeDeg`, 'must be a number above 0 and at most 90');
            if (innerOk && outerOk && (light.innerConeDeg as number) > (light.outerConeDeg as number)) {
                c.add(path, 'innerConeDeg must not exceed outerConeDeg');
            }
            break;
        }
        default:
            c.add(`${path}.type`, 'must be "directional", "point" or "spot"');
    }
}

function validateEnvironment(environment: Json, c: ErrorCollector): void {
    const path = 'environment';
    if (!isRecord(environment)) {
        c.add(path, 'must be an object');
        return;
    }
    const intensityOk = environment.intensity === undefined || (isFiniteNumber(environment.intensity) && environment.intensity >= 0);
    if (!intensityOk) c.add(`${path}.intensity`, 'must be a number of 0 or more');
    switch (environment.type) {
        case 'none':
            break;
        case 'color':
            c.color(`${path}.color`, environment.color);
            break;
        case 'procedural':
            c.direction(`${path}.sunDirection`, environment.sunDirection);
            if (environment.turbidity !== undefined && !isFiniteNumber(environment.turbidity)) {
                c.add(`${path}.turbidity`, 'must be a number');
            }
            if (environment.sunIlluminance !== undefined) c.positiveNumber(`${path}.sunIlluminance`, environment.sunIlluminance);
            break;
        default:
            c.add(`${path}.type`, 'must be "none", "color" or "procedural"');
    }
}

function validateVolume(volume: Json, c: ErrorCollector): void {
    if (!isRecord(volume)) {
        c.add('volume', 'must be an object with center, size and probeSpacing');
        return;
    }
    c.vec3('volume.center', volume.center);
    const size = volume.size;
    const spacing = volume.probeSpacing;
    const sizeOk = c.vec3('volume.size', size);
    if (sizeOk && size.some((extent) => extent <= 0)) c.add('volume.size', 'every extent must be greater than 0');
    const spacingOk = c.positiveNumber('volume.probeSpacing', spacing);
    if (sizeOk && spacingOk && spacing > Math.max(...size)) {
        c.add('volume.probeSpacing', 'is larger than the whole volume');
    }
}

function validateCredits(manifest: Record<string, Json>, c: ErrorCollector): void {
    if (!Array.isArray(manifest.credits) || manifest.credits.length === 0) {
        c.add('credits', 'must list at least one entry (use asset null and license MIT for procedural scenes)');
        return;
    }
    const creditedAssets = new Set<string | null>();
    manifest.credits.forEach((credit: Json, index: number) => {
        const path = `credits[${index}]`;
        if (!isRecord(credit)) {
            c.add(path, 'must be an object');
            return;
        }
        if (credit.asset !== null && !isNonEmptyString(credit.asset)) c.add(`${path}.asset`, 'must be an asset id or null');
        if (!isNonEmptyString(credit.license)) c.add(`${path}.license`, 'must be a non-empty string');
        if (!isNonEmptyString(credit.attribution)) c.add(`${path}.attribution`, 'must be a non-empty string');
        creditedAssets.add(credit.asset as string | null);
    });

    const parts = Array.isArray(manifest.parts) ? manifest.parts : [];
    for (const part of parts) {
        if (isRecord(part) && (part.kind === 'model' || part.kind === 'light-set') && typeof part.asset === 'string' && !creditedAssets.has(part.asset)) {
            c.add('credits', `no credit for asset "${part.asset}" used by a ${part.kind} part`);
        }
        if (isRecord(part) && part.kind === 'procedural' && !creditedAssets.has(null)) {
            c.add('credits', 'a procedural part needs a credit with asset null (license MIT, this repo)');
        }
    }
}

/** Checks a parsed JSON value. Reports every problem, not just the first. */
export function validateManifest(value: unknown, options: ValidateOptions = {}): ValidationResult {
    const c = new ErrorCollector();
    if (!isRecord(value)) {
        return { ok: false, errors: ['manifest: must be a JSON object'] };
    }

    for (const key of Object.keys(value)) {
        if (!TOP_LEVEL_KEYS.has(key)) c.add(key, 'is not a known manifest field');
    }

    if (!isNonEmptyString(value.id) || !KEBAB_CASE.test(value.id)) c.add('id', 'must be kebab-case, like "cornell-box"');
    if (!isNonEmptyString(value.title)) c.add('title', 'must be a non-empty string');
    if (!isNonEmptyString(value.description)) {
        c.add('description', 'must be a non-empty sentence');
    } else if (value.description.length > 600) {
        c.add('description', 'is too long; keep it to about one sentence');
    }

    if (!Array.isArray(value.tags) || value.tags.length === 0 || !value.tags.every(isNonEmptyString)) {
        c.add('tags', 'must be a non-empty array of strings');
    } else if (new Set(value.tags).size !== value.tags.length) {
        c.add('tags', 'must not repeat a tag');
    }

    if (value.units !== 'meters') c.add('units', 'must be "meters"');
    if (value.upAxis !== 'y') c.add('upAxis', 'must be "y"');
    if (value.status !== undefined && value.status !== 'ready' && value.status !== 'awaiting-asset') {
        c.add('status', 'must be "ready" or "awaiting-asset" when present');
    }

    if (!Array.isArray(value.parts) || value.parts.length === 0) {
        c.add('parts', 'must be a non-empty array');
    } else {
        value.parts.forEach((part: Json, index: number) => validatePart(part, `parts[${index}]`, c, options));
    }

    if (!Array.isArray(value.cameras) || value.cameras.length < 2 || value.cameras.length > 4) {
        c.add('cameras', 'must have 2 to 4 bookmarks');
    } else {
        value.cameras.forEach((camera: Json, index: number) => validateCamera(camera, `cameras[${index}]`, c));
        const names = value.cameras.map((camera: Json) => (isRecord(camera) ? camera.name : undefined));
        if (new Set(names).size !== names.length) c.add('cameras', 'bookmark names must be unique');
    }

    if (!Array.isArray(value.lights)) {
        c.add('lights', 'must be an array (empty if the scene only has emissive surfaces)');
    } else {
        value.lights.forEach((light: Json, index: number) => validateLight(light, `lights[${index}]`, c));
    }

    validateMotions(value, c);
    validateSettings(value, c);
    validateEnvironment(value.environment, c);
    validateVolume(value.volume, c);

    if (typeof value.nativeCapable !== 'boolean') c.add('nativeCapable', 'must be true or false');
    if (value.referenceCube !== undefined) {
        if (!isRecord(value.referenceCube)) c.add('referenceCube', 'must be an object with a position');
        else c.vec3('referenceCube.position', value.referenceCube.position);
    }
    if (value.exposureEv !== undefined && (!isFiniteNumber(value.exposureEv) || Math.abs(value.exposureEv) > MAX_DEFAULT_EXPOSURE_EV)) {
        c.add('exposureEv', `must be a number of stops from -${MAX_DEFAULT_EXPOSURE_EV} to +${MAX_DEFAULT_EXPOSURE_EV} (0 is normal)`);
    }
    validateCredits(value, c);

    return { ok: c.errors.length === 0, errors: c.errors };
}

/** Parses JSON text and validates it. Throws one error listing every problem. */
export function parseManifest(json: string, sourceName = 'manifest', options: ValidateOptions = {}): SceneManifest {
    let value: unknown;
    try {
        value = JSON.parse(json);
    } catch (error) {
        throw new Error(`${sourceName}: not valid JSON (${error instanceof Error ? error.message : String(error)})`);
    }
    const result = validateManifest(value, options);
    if (!result.ok) {
        throw new Error(`${sourceName} is not a valid scene manifest:\n  ${result.errors.join('\n  ')}`);
    }
    return value as SceneManifest;
}

// ---------------------------------------------------------------------------
// Cross-checks and conversions
// ---------------------------------------------------------------------------

/** The part of a `sources.json` asset entry the manifest check needs. */
export interface AssetOutputInfo {
    id: string;
    outputs: ReadonlyArray<{ path: string }>;
}

/**
 * Checks that every model part names an asset in `sources.json` and one of its
 * outputs. Scenes marked `awaiting-asset` are skipped: their ids are guesses.
 */
export function checkManifestAgainstSources(manifest: SceneManifest, assets: readonly AssetOutputInfo[]): string[] {
    if (manifest.status === 'awaiting-asset') return [];
    const errors: string[] = [];
    manifest.parts.forEach((part, index) => {
        if (part.kind === 'procedural') return;
        const asset = assets.find((candidate) => candidate.id === part.asset);
        if (!asset) {
            errors.push(`parts[${index}].asset: "${part.asset}" is not in tools/assets/sources.json`);
        } else if (!asset.outputs.some((output) => output.path === part.path)) {
            errors.push(`parts[${index}].path: "${part.path}" is not an output of asset "${part.asset}"`);
        }
    });
    return errors;
}

/**
 * The scene's settings with `overrides` (a link's, say) applied: each a whole
 * number within its range, the default where no override is given. Unknown
 * names and values that aren't numbers are left out.
 */
export function sceneSettingValues(manifest: Pick<SceneManifest, 'settings'>, overrides: Readonly<Record<string, unknown>> = {}): Record<string, number> {
    const values: Record<string, number> = {};
    for (const [name, setting] of Object.entries(manifest.settings ?? {})) {
        const asked = Number(overrides[name]);
        values[name] = Number.isFinite(asked) && overrides[name] !== undefined && overrides[name] !== '' ? Math.min(setting.max, Math.max(setting.min, Math.round(asked))) : setting.default;
    }
    return values;
}

/** What a motion sets on its light at a time (seconds from its start): keyframes interpolated (or held, for `step`), looping if it loops. */
export function motionState(motion: ManifestMotion, seconds: number): Omit<MotionKeyframe, 't'> {
    const frames = motion.keyframes;
    const duration = frames[frames.length - 1].t;
    const time = motion.loop && duration > 0 ? ((seconds % duration) + duration) % duration : Math.min(Math.max(seconds, 0), duration);
    let next = frames.findIndex((frame) => frame.t > time);
    if (next === -1) next = frames.length - 1;
    const before = frames[Math.max(0, next - 1)];
    const after = frames[next];
    const share = motion.interpolation === 'step' || after.t === before.t ? (time >= after.t ? 1 : 0) : Math.min(1, Math.max(0, (time - before.t) / (after.t - before.t)));
    const mix = (a: number, b: number) => a + (b - a) * share;
    const mixVector = (a: Vec3, b: Vec3): Vec3 => [mix(a[0], b[0]), mix(a[1], b[1]), mix(a[2], b[2])];
    return {
        ...(before.position && after.position ? { position: mixVector(before.position, after.position) } : {}),
        ...(before.intensity !== undefined && after.intensity !== undefined ? { intensity: mix(before.intensity, after.intensity) } : {}),
        ...(before.color && after.color ? { color: mixVector(before.color, after.color) } : {}),
    };
}

/** The exposure, in stops, a viewer opens the scene with (0 when the manifest names none). */
export function defaultExposureEv(manifest: Pick<SceneManifest, 'exposureEv'>): number {
    return manifest.exposureEv ?? 0;
}

export function isProceduralOnly(manifest: SceneManifest): boolean {
    return manifest.parts.every((part) => part.kind === 'procedural');
}

/**
 * The tag that marks a scene too big to load unasked, like `sponza-full`
 * (about 13 M triangles). Tools that go through every scene by default (the
 * contact sheets, the full baseline run) leave a heavy scene out unless it is
 * named. The viewer still lists it.
 */
export const HEAVY_SCENE_TAG = 'heavy';

export function isHeavyScene(manifest: Pick<SceneManifest, 'tags'>): boolean {
    return manifest.tags.includes(HEAVY_SCENE_TAG);
}

const DEGREES_TO_RADIANS = Math.PI / 180;

/** Converts a manifest light (degrees) into the scene data / platform shape (radians). */
export function manifestLightToSceneLight(light: ManifestLight): SceneLightData {
    const common = {
        ...(light.label !== undefined ? { label: light.label } : {}),
        color: light.color,
        intensity: light.intensity,
        ...(light.castsShadows !== undefined ? { castsShadows: light.castsShadows } : {}),
    };
    switch (light.type) {
        case 'directional':
            return { type: 'directional', direction: light.direction, ...common };
        case 'point':
            return { type: 'point', position: light.position, range: light.range, ...common };
        case 'spot':
            return {
                type: 'spot',
                position: light.position,
                direction: light.direction,
                range: light.range,
                innerConeAngle: light.innerConeDeg * DEGREES_TO_RADIANS,
                outerConeAngle: light.outerConeDeg * DEGREES_TO_RADIANS,
                ...common,
            };
    }
}
