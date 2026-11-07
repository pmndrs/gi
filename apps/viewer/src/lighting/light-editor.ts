/**
 * The scene's lights as the viewer edits them (docs/design/m2-lights.md §7):
 * a small marker at each point and spot light, and three's own
 * `TransformControls` gizmo on the chosen one, to drag it (and turn a
 * spot). Clicking a marker chooses its light; the panel then shows its
 * sliders (`ui/lighting-controls.ts`).
 *
 * What may not happen while editing:
 * - **GI must not see the markers or the gizmo:** they live in a scene of
 *   their own (`LoadedScene.helpers`), drawn over the picture after the
 *   main one, which GI's scene sync never walks. In the main scene GI would
 *   copy them: the gizmo's picking plane is a 100 km mesh hidden only by its
 *   material (NOTES), and its handles change `visible` every frame.
 * - **Baking cameras must not see them:** that scene is drawn only by the
 *   viewing pass (`ScenePassTechnique`), and they live on LAYER_HELPERS,
 *   which viewing cameras see (`applyViewLayers`). The gizmo's own picking
 *   raycaster is set to that layer too.
 * - **Nothing may recompile in GI:** every edit is a light's position,
 *   direction, colour, intensity, range, cone or group, which the adapter
 *   sends as a light change; switching a light off sets its intensity to 0
 *   rather than removing it (NOTES: removing a three light makes three
 *   compile its materials again).
 *
 * Only the manifest's own lights are listed: not the candles of a light set
 * (1,024 on `sponza-candles`) or the area lights of glowing quads.
 */

import {
    ConeGeometry,
    Mesh,
    MeshBasicMaterial,
    Raycaster,
    SphereGeometry,
    Vector2,
    Vector3,
    type Light,
    type Material,
    type Object3D,
    type PerspectiveCamera,
    type PointLight,
    type SpotLight,
} from 'three';
import { TransformControls } from 'three/addons/controls/TransformControls.js';
import { GI_USER_DATA } from '../ddgi/library';
import { LAYER_HELPERS } from '../scene/layers';
import type { LoadedScene } from '../scene/load-scene';
import { addToScene, updateLightMatrices } from './motion-player';

export type LightKind = 'point' | 'spot' | 'directional';

/** The light settings the panel shows and edits (three's units: candela for point and spot, lux for directional). */
export interface LightSettings {
    on: boolean;
    /** While on; when off, the intensity it comes back at. */
    intensity: number;
    /** Linear RGB. */
    color: [number, number, number];
    /** Metres; 0 is no limit (point and spot). */
    range: number;
    /** Degrees from the axis to the edge of the cone (spot). */
    coneDeg: number;
    /** 0 is a hard edge, 1 fades from the axis (spot; three's `penumbra`). */
    penumbra: number;
    shadows: boolean;
    /** GI light group, 0..3 (`userData.giLightGroup`). */
    group: number;
}

export interface EditableLight {
    /** Index into the manifest's lights. */
    readonly index: number;
    readonly label: string;
    readonly kind: LightKind;
    readonly light: Light;
    /** The marker at a point or spot light; null for a directional light. */
    readonly marker: Mesh | null;
    /** The manifest's own values, for "Put back". */
    readonly original: LightSettings & { position: [number, number, number]; target: [number, number, number] };
    /** The intensity a light switched off comes back at. */
    onIntensity: number;
}

/** A press that moves less than this many pixels before it is let go is a click (it chooses a light). */
const CLICK_PIXELS = 4;

function kindOf(light: Light): LightKind | null {
    if ((light as PointLight).isPointLight) return 'point';
    if ((light as SpotLight).isSpotLight) return 'spot';
    if ((light as { isDirectionalLight?: boolean }).isDirectionalLight) return 'directional';
    return null;
}

/** Puts an object and everything under it on the helpers' layer only, with no shadows. */
function makeHelper(root: Object3D): void {
    root.traverse((object) => {
        object.layers.set(LAYER_HELPERS);
        object.castShadow = false;
        object.receiveShadow = false;
    });
}

export function readLightSettings(light: Light, onIntensity: number): LightSettings {
    const local = light as Partial<PointLight & SpotLight>;
    const group = light.userData[GI_USER_DATA.lightGroup];
    const on = light.parent !== null && light.intensity > 0;
    return {
        on,
        intensity: on ? light.intensity : onIntensity,
        color: [light.color.r, light.color.g, light.color.b],
        range: local.distance ?? 0,
        coneDeg: ((local.angle ?? 0) * 180) / Math.PI,
        penumbra: local.penumbra ?? 0,
        shadows: light.castShadow,
        group: Number.isInteger(group) ? (group as number) : 0,
    };
}

export class LightEditor {
    readonly lights: EditableLight[] = [];
    private readonly controls: TransformControls;
    private readonly raycaster = new Raycaster();
    private selectedIndex: number | null = null;
    private pressAt: Vector2 | null = null;
    private helpersShown = true;
    private readonly markerRadius: number;
    private readonly removePointerListeners: () => void;

    /**
     * @param onEdited called when the gizmo moved a light (the panel refreshes, the link stays)
     * @param onDragging called with true while the gizmo is held (the camera must not move then)
     * @param onSelected called when a click chose a light, or chose none
     */
    constructor(
        private readonly loaded: LoadedScene,
        camera: PerspectiveCamera,
        private readonly canvas: HTMLCanvasElement,
        handlers: { onEdited(): void; onDragging(dragging: boolean): void; onSelected(index: number | null): void },
    ) {
        this.markerRadius = Math.min(0.15, Math.max(0.04, loaded.diagonal / 400));
        loaded.manifest.lights.forEach((_, index) => {
            const light = loaded.manifestLights[index];
            const kind = light ? kindOf(light) : null;
            if (!light || !kind) return;
            const marker = kind === 'directional' ? null : this.createMarker(light, kind);
            const settings = readLightSettings(light, light.intensity);
            const target = (light as Partial<SpotLight>).target?.position ?? new Vector3();
            this.lights.push({
                index,
                label: light.name,
                kind,
                light,
                marker,
                original: { ...settings, on: light.intensity > 0, position: [light.position.x, light.position.y, light.position.z], target: [target.x, target.y, target.z] },
                // An off lamp (Sponza's arcade lamp, intensity 0 in the manifest) comes on at its motions' brightness.
                onIntensity: light.intensity > 0 ? light.intensity : this.motionIntensity(light.name),
            });
        });

        this.controls = new TransformControls(camera, canvas);
        this.controls.setSize(0.8);
        const gizmo = this.controls.getHelper();
        makeHelper(gizmo);
        this.controls.getRaycaster().layers.set(LAYER_HELPERS);
        this.raycaster.layers.set(LAYER_HELPERS);
        loaded.helpers.add(gizmo);
        this.controls.addEventListener('dragging-changed', (event) => handlers.onDragging(Boolean((event as unknown as { value: boolean }).value)));
        this.controls.addEventListener('objectChange', () => {
            this.followMarker();
            handlers.onEdited();
        });

        // A click (a press let go where it started) on a marker chooses its light.
        const down = (event: PointerEvent) => {
            this.pressAt = event.button === 0 ? new Vector2(event.clientX, event.clientY) : null;
        };
        const up = (event: PointerEvent) => {
            const pressAt = this.pressAt;
            this.pressAt = null;
            if (!pressAt || this.controls.dragging || pressAt.distanceTo(new Vector2(event.clientX, event.clientY)) > CLICK_PIXELS) return;
            const picked = this.pick(event, camera);
            if (picked !== null || this.selectedIndex !== null) {
                this.select(picked);
                handlers.onSelected(picked);
            }
        };
        canvas.addEventListener('pointerdown', down);
        canvas.addEventListener('pointerup', up);
        this.removePointerListeners = () => {
            canvas.removeEventListener('pointerdown', down);
            canvas.removeEventListener('pointerup', up);
        };
    }

    get selected(): EditableLight | null {
        return this.selectedIndex === null ? null : (this.lights[this.selectedIndex] ?? null);
    }

    /** Chooses a light by its place in `lights` (null: none). The gizmo goes on its marker. */
    select(index: number | null): void {
        this.selectedIndex = index !== null && this.lights[index] ? index : null;
        const marker = this.selected?.marker ?? null;
        if (marker && this.helpersShown) this.controls.attach(marker);
        else this.controls.detach();
        if (this.selected?.kind !== 'spot') this.controls.setMode('translate');
    }

    /** The gizmo moves (`translate`) or turns (`rotate`, spot lights) the chosen light. */
    setGizmoMode(mode: 'translate' | 'rotate'): void {
        this.controls.setMode(this.selected?.kind === 'spot' ? mode : 'translate');
    }

    get gizmoMode(): 'translate' | 'rotate' {
        return this.controls.mode === 'rotate' ? 'rotate' : 'translate';
    }

    /** Hides the markers and the gizmo (the response meter reads the picture without them). */
    setHelpersVisible(visible: boolean): void {
        this.helpersShown = visible;
        for (const entry of this.lights) if (entry.marker) entry.marker.visible = visible && entry.light.parent !== null;
        const helper = this.controls.getHelper();
        helper.visible = visible;
        this.controls.enabled = visible;
        this.select(this.selectedIndex);
    }

    /** The settings of a light, as the panel shows them. */
    settings(entry: EditableLight): LightSettings {
        return readLightSettings(entry.light, entry.onIntensity);
    }

    /**
     * Applies the panel's settings to a light. Switching off sets the
     * intensity to 0 (it stays in the scene: removing a three light
     * recompiles three's materials); a light the scene left out (an off lamp)
     * joins it when switched on. Returns the light, for `markChanged`.
     */
    apply(entry: EditableLight, settings: LightSettings): Light {
        const light = entry.light;
        if (settings.on) addToScene(this.loaded, light);
        entry.onIntensity = settings.intensity;
        light.intensity = settings.on ? settings.intensity : 0;
        light.color.setRGB(...settings.color);
        const local = light as Partial<PointLight & SpotLight>;
        if (entry.kind !== 'directional') local.distance = Math.max(0, settings.range);
        if (entry.kind === 'spot') {
            local.angle = (Math.min(89, Math.max(1, settings.coneDeg)) * Math.PI) / 180;
            local.penumbra = Math.min(1, Math.max(0, settings.penumbra));
        }
        // three compiles its materials again when a light's shadow is switched (once each way); GI follows `castShadow`.
        light.castShadow = settings.shadows;
        light.userData[GI_USER_DATA.lightGroup] = Math.min(3, Math.max(0, Math.round(settings.group)));
        updateLightMatrices(light);
        this.refreshMarker(entry);
        return light;
    }

    /** Puts a light back as the manifest has it (its place, aim, and every setting). */
    putBack(entry: EditableLight): Light {
        const original = entry.original;
        entry.light.position.set(...original.position);
        const target = (entry.light as Partial<SpotLight>).target;
        if (target) target.position.set(...original.target);
        entry.onIntensity = original.on ? original.intensity : entry.onIntensity;
        return this.apply(entry, original);
    }

    /** Once a frame: markers follow their lights (motions move them) and show their colour. */
    update(): void {
        for (const entry of this.lights) {
            if (!entry.marker) continue;
            // A marker being dragged leads its light; every other marker follows its light (motions move them).
            if (this.controls.dragging && entry.marker === this.controls.object) this.refreshMarkerLook(entry);
            else this.refreshMarker(entry);
        }
    }

    dispose(): void {
        this.removePointerListeners();
        this.controls.detach();
        this.controls.getHelper().removeFromParent();
        this.controls.dispose();
        for (const entry of this.lights) {
            if (!entry.marker) continue;
            entry.marker.removeFromParent();
            entry.marker.geometry.dispose();
            (entry.marker.material as Material).dispose();
        }
    }

    // -- markers -----------------------------------------------------------

    private createMarker(light: Light, kind: LightKind): Mesh {
        // A spot's cone opens along the marker's +z, which `lookAt` aims at the light's target (three's cones point along +y).
        const geometry = kind === 'spot' ? new ConeGeometry(this.markerRadius, this.markerRadius * 2.5, 16).rotateX(-Math.PI / 2) : new SphereGeometry(this.markerRadius, 16, 12);
        const marker = new Mesh(geometry, new MeshBasicMaterial());
        marker.name = `light marker ${light.name}`;
        makeHelper(marker);
        this.loaded.helpers.add(marker);
        return marker;
    }

    /** Moves a marker to its light and aims a spot's marker along it. */
    private refreshMarker(entry: EditableLight): void {
        const marker = entry.marker;
        if (!marker) return;
        marker.position.copy(entry.light.position);
        const target = (entry.light as Partial<SpotLight>).target;
        if (entry.kind === 'spot' && target) marker.lookAt(target.position);
        this.refreshMarkerLook(entry);
    }

    private refreshMarkerLook(entry: EditableLight): void {
        const marker = entry.marker!;
        // The light's colour at full brightness, so a dim or coloured lamp still shows; an off light is dark grey.
        const color = entry.light.color;
        const peak = Math.max(color.r, color.g, color.b, 1e-6);
        const on = entry.light.parent !== null && entry.light.intensity > 0;
        (marker.material as MeshBasicMaterial).color.setRGB(on ? color.r / peak : 0.15, on ? color.g / peak : 0.15, on ? color.b / peak : 0.15);
        marker.visible = this.helpersShown;
    }

    /** The chosen light follows its marker (the gizmo moved it); a spot's target follows its aim. */
    followMarker(): void {
        const entry = this.selected;
        const marker = entry?.marker;
        if (!entry || !marker) return;
        const light = entry.light;
        const target = (light as Partial<SpotLight>).target;
        if (entry.kind === 'spot' && target) {
            // Keep the spot's aim: the target sits along the marker's +z, as far as it was.
            const reach = Math.max(1, target.position.distanceTo(light.position));
            const forward = new Vector3(0, 0, 1).applyQuaternion(marker.quaternion);
            target.position.copy(marker.position).addScaledVector(forward, reach);
        }
        light.position.copy(marker.position);
        updateLightMatrices(light);
    }

    /** The light whose marker is under the pointer, or null. */
    private pick(event: PointerEvent, camera: PerspectiveCamera): number | null {
        const box = this.canvas.getBoundingClientRect();
        const pointer = new Vector2(((event.clientX - box.left) / box.width) * 2 - 1, -((event.clientY - box.top) / box.height) * 2 + 1);
        this.raycaster.setFromCamera(pointer, camera);
        const markers = this.lights.flatMap((entry) => (entry.marker && entry.marker.visible ? [entry.marker] : []));
        const hit = this.raycaster.intersectObjects(markers, false)[0];
        if (!hit) return null;
        const index = this.lights.findIndex((entry) => entry.marker === hit.object);
        return index >= 0 ? index : null;
    }

    /** The brightness a motion plays a light at (its first keyframe with an intensity), for an off lamp switched on by hand. */
    private motionIntensity(label: string): number {
        for (const motion of this.loaded.manifest.motions ?? []) {
            if (motion.target !== `light:${label}`) continue;
            const intensity = motion.keyframes.find((keyframe) => keyframe.intensity !== undefined && keyframe.intensity > 0)?.intensity;
            if (intensity !== undefined) return intensity;
        }
        return 1;
    }
}
