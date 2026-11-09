/**
 * What a manifest motion moves, and how (T3.8; `scenes/src/manifest.ts`,
 * `ManifestMotion`). Each kind of target gets a driver with the same three
 * calls, so the motion player and the response meter treat a lamp, a door
 * and the camera alike:
 * - `save()` remembers how the target is now, `restore()` puts it back;
 * - `apply(state, motion)` sets it to a motion's state (`motionState`);
 * - `changed` is what GI should be told moved (`markChanged`), or null.
 *
 * **Objects** move relative to where they stand in the scene: a keyframe's
 * `position` is an offset in metres, and its `rotationDeg` turns the object
 * about the motion's `pivot` (a door's hinge). The new world transform is
 * `offset × pivot × turn × pivot⁻¹ × rest`, worked out from the transform the
 * object had when the scene loaded, so motions never drift. A keyframe's
 * `sway` bends every mesh under the object that has a wind weight (the
 * alpha of its vertex colour) through `WindSway`, times the viewer's wind
 * strength. **Props** (`manifest.props`) are made here and join the scene
 * only while a motion moves them.
 *
 * Every move brings the world matrices up to date at once, as
 * `updateLightMatrices` does for lights: the three adapter reads world
 * matrices as three's last render left them, so GI would otherwise see each
 * move a frame late.
 *
 * Swaying meshes are marked `userData.giDeforming` with a `giDeformBounds`
 * for the strongest wind when the driver is made, which is when the scene
 * opens (`MotionPlayer`'s constructor): DDGI then takes them in as deforming
 * from its first copy of the scene, and a still tree costs nothing (the
 * adapter sends a pose only when the position attribute changes).
 */

import { BoxGeometry, Euler, Matrix4, Mesh, MeshStandardMaterial, type Light, type Object3D, type Quaternion, type Vector3 } from 'three';
import type { ManifestMotion, ManifestProp, MotionKeyframe, Vec3 } from '../../../../scenes/src/manifest';
import { LAYER_VIEW_GEOMETRY } from '../scene/layers';
import type { LoadedScene } from '../scene/load-scene';
import { WindSway } from '../scene/wind-sway';

/** A motion's state at a time: `motionState`'s answer. */
export type MotionState = Omit<MotionKeyframe, 't'>;

export interface MotionDriver {
    /** What GI should be told moved after `apply` or `restore`; null when GI needn't know (the camera). */
    readonly changed: Object3D | null;
    save(): unknown;
    restore(saved: unknown): void;
    apply(state: MotionState, motion: ManifestMotion): void;
}

/**
 * Metres the crown of a tree (wind weight 1) moves at a sway of 1 and full
 * wind strength: T3.3's strong wind for the 10 m wind tree.
 */
export const MAX_WIND_REACH_M = 1.5;
/** The way the wind blows, in each mesh's own space (the wind tree stands upright, so along the ground). */
const WIND_DIRECTION: readonly [number, number, number] = [1, 0, 0];

/** The camera as a motion moves it: the viewer's camera rig. */
export interface CameraControl {
    pose(): { position: Vec3; target: Vec3 };
    place(position: Vec3, target: Vec3): void;
}

// -- lights ------------------------------------------------------------------

/** A light's state a motion (or the response meter's step) can change, to put it back afterwards. */
export interface SavedLight {
    position: [number, number, number];
    intensity: number;
    color: [number, number, number];
    inScene: boolean;
}

export function saveLight(light: Light): SavedLight {
    return {
        position: [light.position.x, light.position.y, light.position.z],
        intensity: light.intensity,
        color: [light.color.r, light.color.g, light.color.b],
        inScene: light.parent !== null,
    };
}

/** Puts a light back as `saveLight` found it (out of the scene again if it was out). */
export function restoreLight(light: Light, saved: SavedLight): void {
    light.position.set(...saved.position);
    light.intensity = saved.intensity;
    light.color.setRGB(...saved.color);
    updateLightMatrices(light);
    if (!saved.inScene) light.removeFromParent();
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

export class LightDriver implements MotionDriver {
    constructor(
        private readonly loaded: LoadedScene,
        readonly light: Light,
    ) {}

    get changed(): Object3D {
        return this.light;
    }

    save(): SavedLight {
        return saveLight(this.light);
    }

    restore(saved: unknown): void {
        restoreLight(this.light, saved as SavedLight);
    }

    /** Sets the light to the state, adding it to the scene if it was left out (Sponza's arcade lamp). */
    apply(state: MotionState): void {
        addToScene(this.loaded, this.light);
        if (state.position) this.light.position.set(...state.position);
        if (state.intensity !== undefined) this.light.intensity = state.intensity;
        // Manifest colours are linear, three's working colour space.
        if (state.color) this.light.color.setRGB(...state.color);
        updateLightMatrices(this.light);
    }
}

// -- objects -----------------------------------------------------------------

interface SavedObject {
    /** The local matrix, which is the transform for objects with `matrixAutoUpdate` off (procedural parts)... */
    matrix: Matrix4;
    /** ...and position, turn and scale, which are the transform for the rest. */
    position: Vector3;
    quaternion: Quaternion;
    scale: Vector3;
    sway: number;
    inScene: boolean;
}

/**
 * A plain box for a manifest prop: rough, not metal, casting and taking
 * three's shadows. On layer 0 (GI, bakes) and the viewing cameras' layer, as
 * the loaded scene's own meshes are (`load-scene.ts`, layers.ts).
 */
export function makeProp(prop: ManifestProp): Mesh {
    const material = new MeshStandardMaterial({ name: `prop ${prop.name}`, roughness: 1, metalness: 0 });
    // Manifest colours are linear, three's working colour space.
    material.color.setRGB(...prop.color);
    const mesh = new Mesh(new BoxGeometry(...prop.size), material);
    mesh.name = prop.name;
    mesh.position.set(...prop.position);
    const degrees = prop.rotationDeg ?? [0, 0, 0];
    mesh.rotation.set(...(degrees.map((value) => (value * Math.PI) / 180) as [number, number, number]), 'XYZ');
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    mesh.layers.enable(LAYER_VIEW_GEOMETRY);
    mesh.updateMatrix();
    return mesh;
}

export class ObjectDriver implements MotionDriver {
    /** The object's world transform as the scene placed it: every motion state is worked out from it. */
    private readonly restWorld: Matrix4;
    /** Meshes under the object with a wind weight, each with its sway (empty when nothing sways). */
    private readonly sways: WindSway[] = [];
    private swayNow = 0;
    private readonly scratch = { world: new Matrix4(), turn: new Matrix4(), step: new Matrix4(), euler: new Euler() };

    /**
     * @param object the object as the scene placed it, or a prop not yet in the scene
     * @param parent where a prop goes when a motion starts; null for objects already in the scene
     * @param windStrength the wind slider's share of `MAX_WIND_REACH_M`, 0 to 1, read at every sway
     */
    constructor(
        readonly object: Object3D,
        private readonly parent: Object3D | null,
        private readonly windStrength: () => number,
    ) {
        if (parent) object.updateMatrixWorld(true);
        this.restWorld = object.matrixWorld.clone();
    }

    get changed(): Object3D {
        return this.object;
    }

    /** True when a mesh under it has a wind weight, so `sway` keyframes can bend it. */
    get canSway(): boolean {
        return this.sways.length > 0;
    }

    /**
     * Finds the meshes with a wind weight (an RGBA vertex colour) and marks
     * them deforming within the box their strongest sway stays in. Called
     * once, before GI first copies the scene.
     */
    prepareSway(): void {
        if (this.sways.length > 0) return;
        this.object.traverse((child) => {
            const mesh = child as Mesh;
            const color = mesh.isMesh ? mesh.geometry.getAttribute('color') : undefined;
            if (!color || color.itemSize !== 4) return;
            const sway = new WindSway(mesh, { reach: MAX_WIND_REACH_M, direction: WIND_DIRECTION });
            mesh.userData.giDeforming = true;
            mesh.userData.giDeformBounds = sway.swayBounds();
            this.sways.push(sway);
        });
    }

    save(): SavedObject {
        const object = this.object;
        return {
            matrix: object.matrix.clone(),
            position: object.position.clone(),
            quaternion: object.quaternion.clone(),
            scale: object.scale.clone(),
            sway: this.swayNow,
            inScene: object.parent !== null,
        };
    }

    restore(saved: unknown): void {
        const { matrix, position, quaternion, scale, sway, inScene } = saved as SavedObject;
        const object = this.object;
        object.matrix.copy(matrix);
        object.position.copy(position);
        object.quaternion.copy(quaternion);
        object.scale.copy(scale);
        object.matrixWorldNeedsUpdate = true;
        object.updateMatrixWorld(true);
        if (this.sways.length > 0) this.setSway(sway);
        if (!inScene) object.removeFromParent();
    }

    apply(state: MotionState, motion: ManifestMotion): void {
        if (this.parent && !this.object.parent) this.parent.add(this.object);
        if (state.position || state.rotationDeg) this.place(state.position ?? [0, 0, 0], state.rotationDeg ?? [0, 0, 0], motion.pivot);
        if (state.sway !== undefined) this.setSway(state.sway);
    }

    /**
     * World transform = offset × pivot × turn × pivot⁻¹ × rest; then back into
     * the parent's space. Written to the local matrix and to position, turn
     * and scale both: procedural parts keep `matrixAutoUpdate` off with the
     * transform in `matrix` (scenes/src/three-scene.ts), models and props keep
     * it on and rebuild `matrix` from the other three.
     */
    private place(offset: Vec3, rotationDeg: Vec3, pivot: Vec3 | undefined): void {
        const { world, turn, step, euler } = this.scratch;
        const object = this.object;
        // Without a pivot the object turns about its own origin, where the rest transform puts it.
        const centre = pivot ?? [this.restWorld.elements[12], this.restWorld.elements[13], this.restWorld.elements[14]];
        euler.set(...(rotationDeg.map((value) => (value * Math.PI) / 180) as [number, number, number]), 'XYZ');
        turn.makeRotationFromEuler(euler);
        world.makeTranslation(offset[0] + centre[0], offset[1] + centre[1], offset[2] + centre[2]);
        world.multiply(turn);
        world.multiply(step.makeTranslation(-centre[0], -centre[1], -centre[2]));
        world.multiply(this.restWorld);
        // Back into the parent's space: local = parent⁻¹ × world. The parent's world matrix is
        // current (the scene's groups don't move), and updating it would walk every sibling.
        const parent = object.parent;
        if (parent) world.premultiply(step.copy(parent.matrixWorld).invert());
        object.matrix.copy(world);
        world.decompose(object.position, object.quaternion, object.scale);
        object.matrixWorldNeedsUpdate = true;
        object.updateMatrixWorld(true);
    }

    private setSway(sway: number): void {
        this.swayNow = sway;
        for (const mesh of this.sways) mesh.pose(sway * this.windStrength());
    }

    /** Shows the sway it last set again with the current wind strength (the slider moved while nothing plays). */
    refreshSway(): void {
        if (this.sways.length > 0) this.setSway(this.swayNow);
    }

    /** A prop's geometry and material (a prop belongs to the player, not to the loaded scene). */
    disposeProp(): void {
        if (!this.parent) return;
        this.object.removeFromParent();
        const mesh = this.object as Mesh;
        mesh.geometry.dispose();
        (mesh.material as MeshStandardMaterial).dispose();
    }
}

// -- the camera --------------------------------------------------------------

export class CameraDriver implements MotionDriver {
    readonly changed = null;

    constructor(private readonly camera: CameraControl) {}

    save(): { position: Vec3; target: Vec3 } {
        return this.camera.pose();
    }

    restore(saved: unknown): void {
        const pose = saved as { position: Vec3; target: Vec3 };
        this.camera.place(pose.position, pose.target);
    }

    apply(state: MotionState): void {
        if (state.position && state.target) this.camera.place(state.position, state.target);
    }
}
