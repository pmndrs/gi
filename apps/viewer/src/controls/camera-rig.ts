/**
 * Moving the camera: orbit (drag to circle a point, wheel to zoom) and fly
 * (WASD plus drag to look). Both move the same camera, so switching keeps
 * the view.
 */

import { Euler, MathUtils, Vector3, type PerspectiveCamera } from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import type { CameraBookmark } from '../../../../scenes/src/manifest';

export type NavigationMode = 'orbit' | 'fly';

/** Radians of turn per pixel of mouse drag while flying. */
const LOOK_SPEED = 0.0025;
const FAST_MULTIPLIER = 4;

/** Keys that move the camera while flying, as directions in the camera's own frame. */
const FLY_KEYS: Record<string, Vector3> = {
    KeyW: new Vector3(0, 0, -1),
    ArrowUp: new Vector3(0, 0, -1),
    KeyS: new Vector3(0, 0, 1),
    ArrowDown: new Vector3(0, 0, 1),
    KeyA: new Vector3(-1, 0, 0),
    ArrowLeft: new Vector3(-1, 0, 0),
    KeyD: new Vector3(1, 0, 0),
    ArrowRight: new Vector3(1, 0, 0),
    KeyE: new Vector3(0, 1, 0),
    Space: new Vector3(0, 1, 0),
    KeyQ: new Vector3(0, -1, 0),
    KeyC: new Vector3(0, -1, 0),
};

function isTypingTarget(target: EventTarget | null): boolean {
    return target instanceof HTMLElement && ['INPUT', 'SELECT', 'TEXTAREA'].includes(target.tagName);
}

export class CameraRig {
    mode: NavigationMode = 'orbit';
    /** Fly speed in metres per second, before the shift boost. */
    flySpeed = 3;

    private readonly orbit: OrbitControls;
    private readonly pressed = new Set<string>();
    private readonly euler = new Euler(0, 0, 0, 'YXZ');
    private lookPointer: number | null = null;
    private sceneDiagonal = 20;
    private lastPose = '';
    private readonly poseListeners: Array<() => void> = [];

    constructor(
        private readonly camera: PerspectiveCamera,
        private readonly element: HTMLElement,
    ) {
        this.orbit = new OrbitControls(camera, element);
        this.orbit.enableDamping = true;
        this.orbit.dampingFactor = 0.12;

        window.addEventListener('keydown', (event) => {
            if (this.mode !== 'fly' || isTypingTarget(event.target) || event.ctrlKey || event.metaKey) return;
            if (event.code in FLY_KEYS) {
                this.pressed.add(event.code);
                event.preventDefault();
            } else if (event.code.startsWith('Shift')) {
                this.pressed.add(event.code);
            }
        });
        window.addEventListener('keyup', (event) => this.pressed.delete(event.code));
        window.addEventListener('blur', () => this.pressed.clear());

        element.addEventListener('pointerdown', (event) => {
            if (this.mode !== 'fly' || event.button > 1) return;
            this.lookPointer = event.pointerId;
            element.setPointerCapture(event.pointerId);
        });
        element.addEventListener('pointermove', (event) => {
            if (this.mode !== 'fly' || event.pointerId !== this.lookPointer) return;
            this.euler.setFromQuaternion(camera.quaternion, 'YXZ');
            this.euler.y -= event.movementX * LOOK_SPEED;
            this.euler.x = MathUtils.clamp(this.euler.x - event.movementY * LOOK_SPEED, -Math.PI / 2 + 0.01, Math.PI / 2 - 0.01);
            this.euler.z = 0;
            camera.quaternion.setFromEuler(this.euler);
        });
        const endLook = (event: PointerEvent) => {
            if (event.pointerId === this.lookPointer) this.lookPointer = null;
        };
        element.addEventListener('pointerup', endLook);
        element.addEventListener('pointercancel', endLook);
        element.addEventListener(
            'wheel',
            (event) => {
                if (this.mode !== 'fly') return;
                event.preventDefault();
                this.flySpeed = MathUtils.clamp(this.flySpeed * (event.deltaY < 0 ? 1.15 : 1 / 1.15), 0.2, 200);
            },
            { passive: false },
        );
    }

    /** Pointer drags that start on the wipe line must not move the camera. */
    set enabled(enabled: boolean) {
        this.orbit.enabled = enabled && this.mode === 'orbit';
    }

    /** Called with the scene size so speeds and zoom limits suit it. */
    setSceneScale(diagonal: number): void {
        this.sceneDiagonal = diagonal;
        this.flySpeed = Math.max(1, diagonal / 12);
        this.orbit.maxDistance = diagonal * 3;
        this.orbit.minDistance = 0.05;
    }

    setMode(mode: NavigationMode): void {
        if (mode === this.mode) return;
        this.mode = mode;
        this.pressed.clear();
        this.orbit.enabled = mode === 'orbit';
        if (mode === 'orbit') {
            // Circle a point straight ahead of where the camera is looking now.
            const ahead = new Vector3(0, 0, -1).applyQuaternion(this.camera.quaternion);
            const distance = Math.max(1, this.sceneDiagonal * 0.1);
            this.orbit.target.copy(this.camera.position).addScaledVector(ahead, distance);
            this.orbit.update();
        }
    }

    goTo(bookmark: CameraBookmark): void {
        this.camera.position.set(...bookmark.position);
        this.camera.fov = bookmark.fovYDeg;
        this.camera.updateProjectionMatrix();
        this.camera.lookAt(...bookmark.target);
        this.orbit.target.set(...bookmark.target);
        this.orbit.update();
    }

    /** Current view as the numbers a link or a new bookmark needs. */
    pose(): { position: [number, number, number]; target: [number, number, number]; fovYDeg: number } {
        const ahead = new Vector3(0, 0, -1).applyQuaternion(this.camera.quaternion);
        const distance = this.mode === 'orbit' ? this.camera.position.distanceTo(this.orbit.target) : Math.max(1, this.sceneDiagonal * 0.1);
        const target = this.camera.position.clone().addScaledVector(ahead, distance);
        const round = (v: Vector3): [number, number, number] => [Number(v.x.toFixed(2)), Number(v.y.toFixed(2)), Number(v.z.toFixed(2))];
        return { position: round(this.camera.position), target: round(target), fovYDeg: Number(this.camera.fov.toFixed(1)) };
    }

    /** Runs `listener` after the camera has stopped changing for a moment, so the link can follow. */
    onPoseChanged(listener: () => void): void {
        this.poseListeners.push(listener);
    }

    update(deltaSeconds: number): void {
        if (this.mode === 'orbit') {
            this.orbit.update();
        } else if (this.pressed.size > 0) {
            const move = new Vector3();
            for (const code of this.pressed) {
                const direction = FLY_KEYS[code];
                if (direction) move.add(direction);
            }
            if (move.lengthSq() > 0) {
                const boost = this.pressed.has('ShiftLeft') || this.pressed.has('ShiftRight') ? FAST_MULTIPLIER : 1;
                move.normalize().multiplyScalar(this.flySpeed * boost * deltaSeconds);
                move.applyQuaternion(this.camera.quaternion);
                this.camera.position.add(move);
            }
        }
        const pose = this.pose();
        const key = JSON.stringify(pose);
        if (key !== this.lastPose) {
            this.lastPose = key;
            for (const listener of this.poseListeners) listener();
        }
    }
}
