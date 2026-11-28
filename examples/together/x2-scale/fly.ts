/**
 * A small fly camera for X2, the viewer's approach (apps/viewer's camera
 * rig): WASD or the arrows to move, E and Q (or Space and C) to rise and
 * sink, drag to look, Shift to go four times faster. Until you touch it, the
 * camera tours the town on its own: a loop down four streets at head height,
 * at 5 m/s, the speed DDGI's camera-following volume was sized for
 * (docs/design/m3-moving.md §4.2).
 */

import * as THREE from 'three/webgpu';

const LOOK_SPEED = 0.0025; // radians per pixel dragged
const FAST = 4;
const KEYS: Record<string, THREE.Vector3> = {
    KeyW: new THREE.Vector3(0, 0, -1),
    ArrowUp: new THREE.Vector3(0, 0, -1),
    KeyS: new THREE.Vector3(0, 0, 1),
    ArrowDown: new THREE.Vector3(0, 0, 1),
    KeyA: new THREE.Vector3(-1, 0, 0),
    ArrowLeft: new THREE.Vector3(-1, 0, 0),
    KeyD: new THREE.Vector3(1, 0, 0),
    ArrowRight: new THREE.Vector3(1, 0, 0),
    KeyE: new THREE.Vector3(0, 1, 0),
    Space: new THREE.Vector3(0, 1, 0),
    KeyQ: new THREE.Vector3(0, -1, 0),
    KeyC: new THREE.Vector3(0, -1, 0),
};

/** The tour: a square of streets 96 m across, at head height (the streets cross every 24 m). */
const TOUR_CORNERS = [new THREE.Vector3(-48, 2, -48), new THREE.Vector3(48, 2, -48), new THREE.Vector3(48, 2, 48), new THREE.Vector3(-48, 2, 48)];
const TOUR_SPEED = 5; // m/s

export class FlyCamera {
    /** Metres per second, before Shift. */
    speed = 8;
    /** The camera tours the town until a key or a drag takes over. */
    touring = true;

    private readonly pressed = new Set<string>();
    private readonly euler = new THREE.Euler(0, 0, 0, 'YXZ');
    private tourDistance = 0;
    private dragging: number | null = null;

    constructor(
        private readonly camera: THREE.PerspectiveCamera,
        element: HTMLElement,
        private readonly onTakeOver: () => void = () => {},
    ) {
        this.euler.setFromQuaternion(camera.quaternion);
        const isTyping = (target: EventTarget | null) => target instanceof HTMLElement && ['INPUT', 'SELECT', 'TEXTAREA'].includes(target.tagName);
        window.addEventListener('keydown', (event) => {
            if (isTyping(event.target) || event.ctrlKey || event.metaKey || !(event.code in KEYS || event.code.startsWith('Shift'))) return;
            this.pressed.add(event.code);
            if (event.code in KEYS) this.takeOver();
            event.preventDefault();
        });
        window.addEventListener('keyup', (event) => this.pressed.delete(event.code));
        window.addEventListener('blur', () => this.pressed.clear());
        element.addEventListener('pointerdown', (event) => {
            this.dragging = event.pointerId;
            element.setPointerCapture(event.pointerId);
        });
        element.addEventListener('pointermove', (event) => {
            if (event.pointerId !== this.dragging) return;
            this.takeOver();
            this.euler.y -= event.movementX * LOOK_SPEED;
            this.euler.x = THREE.MathUtils.clamp(this.euler.x - event.movementY * LOOK_SPEED, -1.5, 1.5);
            this.camera.quaternion.setFromEuler(this.euler);
        });
        const release = (event: PointerEvent) => {
            if (event.pointerId === this.dragging) this.dragging = null;
        };
        element.addEventListener('pointerup', release);
        element.addEventListener('pointercancel', release);
    }

    /** Stops the tour and puts the camera at `position`, looking at `target`. */
    jump(position: THREE.Vector3Like, target: THREE.Vector3Like): void {
        this.touring = false;
        this.camera.position.copy(position);
        this.camera.lookAt(target.x, target.y, target.z);
        this.euler.setFromQuaternion(this.camera.quaternion);
    }

    /** Starts the tour again from its first corner. */
    tour(): void {
        this.touring = true;
        this.tourDistance = 0;
    }

    update(seconds: number): void {
        if (this.touring) {
            this.followTour(seconds);
            return;
        }
        const move = new THREE.Vector3();
        for (const code of this.pressed) if (KEYS[code]) move.add(KEYS[code]);
        if (move.lengthSq() === 0) return;
        const fast = this.pressed.has('ShiftLeft') || this.pressed.has('ShiftRight') ? FAST : 1;
        move.normalize().applyQuaternion(this.camera.quaternion).multiplyScalar(this.speed * fast * seconds);
        this.camera.position.add(move);
    }

    private takeOver(): void {
        if (!this.touring) return;
        this.touring = false;
        this.euler.setFromQuaternion(this.camera.quaternion);
        this.onTakeOver();
    }

    private followTour(seconds: number): void {
        const side = TOUR_CORNERS[0].distanceTo(TOUR_CORNERS[1]);
        this.tourDistance = (this.tourDistance + TOUR_SPEED * seconds) % (side * TOUR_CORNERS.length);
        const at = (distance: number) => {
            const wrapped = (distance + side * TOUR_CORNERS.length) % (side * TOUR_CORNERS.length);
            const index = Math.floor(wrapped / side);
            return TOUR_CORNERS[index].clone().lerp(TOUR_CORNERS[(index + 1) % TOUR_CORNERS.length], (wrapped - index * side) / side);
        };
        this.camera.position.copy(at(this.tourDistance));
        // Look 12 m ahead along the loop, so the corners turn smoothly; a little up, at the buildings.
        const ahead = at(this.tourDistance + 12);
        ahead.y += 1.5;
        this.camera.lookAt(ahead);
    }
}
