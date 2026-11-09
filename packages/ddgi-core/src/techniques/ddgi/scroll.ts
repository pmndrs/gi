/**
 * The camera-following volume (docs/design/m3-moving.md §4.2): a fixed-size
 * window of probes over a lattice fixed in world space, RTXGI's "infinite
 * scrolling" with clipmap addressing. The CPU half, plain numbers and
 * unit-tested without a GPU: where the window is, when it moves, which
 * planes enter, and where the valid extent is heading. The GPU half is
 * `ddgi/scroll` (wgsl/scroll.ts), which resets the entering slots and moves
 * the valid extent.
 *
 * Words used here:
 * - **lattice coordinates** `L`: probe `L` sits at `anchor + L × spacing`,
 *   the anchor being the volume's first probe as its settings place it. The
 *   lattice never moves, so neither does a probe: its irradiance and distance
 *   stay valid while the window moves around it;
 * - **the window** `[window, window + n)` on each axis, in lattice units:
 *   the probes the volume holds now. Its first probe is "window probe" 0;
 * - **slots**: where a probe's data lives (its atlas tiles, its probe-data
 *   texel, its history record). Lattice probe `L` lives in slot `L mod n` on
 *   each axis, so when the window moves one plane the slot that falls off one
 *   end is reused at the other end and nothing else moves. Window probe `w`
 *   is in slot `(w + window) mod n`. For a fixed volume the window stays at 0
 *   and a slot is its window probe;
 * - **the valid extent**, per axis a low and a high side in lattice units:
 *   coverage fades at the valid extent instead of at the window's edge, so
 *   new planes fade in and planes about to leave fade out (the "Fading" note
 *   below).
 *
 * **The window rule.** The window moves by whole planes when the camera is
 * more than WINDOW_MOVE_SPACINGS (0.75) from the window's centre on an axis,
 * and then centres on the camera. *Why 0.75, not 0.5:* a camera resting on a
 * boundary between two cells would otherwise move the window back and forth,
 * clearing the same planes every frame. After a move the camera is within
 * half a spacing of the new centre, so it must travel back more than a
 * quarter spacing past where it started before the window moves back.
 *
 * **Fading.** Coverage is full out to half a spacing past the valid extent
 * and fades to 0 over the next half spacing (volumes.ts), and the cell a
 * covered point falls in must have warm probes at both ends, or the probes
 * that are still warming would make the answer jump when they join. So the
 * valid extent keeps one plane inside the warm planes on each side:
 * - **ahead** (planes entering), the GPU moves it out by
 *   VALID_EXTENT_STEP_SPACINGS a frame, and only over planes whose probes
 *   have finished warming (no probe still in state 3), so a new plane fades
 *   in over about a quarter of a second once warm;
 * - **behind** (planes about to leave), it follows the camera continuously,
 *   `EXTENT_INSET_SPACINGS` inside the window's half-width, so the planes
 *   about to be reused have faded out before the window moves. It never
 *   jumps, so the trailing edge never pops either.
 */

import type { Vec3 } from '../../platform/contract/math.ts';
import type { ProbeGrid } from './settings.ts';

/** The window moves when the camera is more than this many spacings from its centre on an axis. */
export const WINDOW_MOVE_SPACINGS = 0.75;

/**
 * How far the valid extent moves out a frame, in spacings, once the planes
 * it moves over are warm (m3-moving.md §4.2): a new plane fades in over 15
 * frames, a quarter of a second at 60 Hz. Faster than the 5 m/s fly-through
 * at 1.5 m (0.056 spacings a frame), so the extent keeps up.
 */
export const VALID_EXTENT_STEP_SPACINGS = 1 / 15;

/**
 * How far inside the camera's half-window the valid extent's trailing side
 * follows the camera, in spacings. The window moves one plane when the camera
 * is WINDOW_MOVE_SPACINGS past its centre; at that moment the extent must be
 * a whole plane inside the new window (so the cell its fade reaches has two
 * warm planes), and the new window starts one plane further on: 2 planes
 * from the old window's first plane, which is the camera's half-window less
 * 2 − 0.75 = 1.25. With the camera centred that leaves the extent 1.25
 * planes inside the window on each side, and coverage 0.75 inside it.
 */
export const EXTENT_INSET_SPACINGS = 2 - WINDOW_MOVE_SPACINGS;

/** Fewer probes than this on an axis leave no room for the trailing fade (prepare refuses them). */
export const MIN_CAMERA_VOLUME_PROBES = 4;

/**
 * Updates an entering probe is owed at priority: the two it needs before it
 * is used in shading (m3-moving.md §5.2, state 3 ends after the second), so
 * the scheduler takes them first. Not the K of a change: an entering probe
 * starts from nothing, and its first update replaces the history outright.
 */
export const ENTERING_WARMING_UPDATES = 2;

/** An entering probe's urgency for the scheduler: a whole change, as a probe nearest a moved object gets (wgsl/mark.ts). */
export const ENTERING_URGENCY = 1;

/** One frame's window move on the CPU, for `ddgi/scroll`. */
export interface WindowMove {
    /** Planes the window moved on each axis this frame (the sign is the direction); 0 when it stayed. */
    readonly moved: readonly [number, number, number];
    /** The camera jumped more than half the window on some axis: every slot enters (and the warm-up runs again). */
    readonly teleport: boolean;
    /** Probes whose slots enter this frame (and are reset). */
    readonly probesEntering: number;
}

/** The valid extent's targets, in lattice units: where its low and high sides head this frame. */
export interface ExtentTargets {
    readonly min: Vec3;
    readonly max: Vec3;
}

/**
 * One camera-following volume's window (the file comment). It starts where
 * the volume's settings put the box (lattice 0 is its first probe) and moves
 * with `follow`.
 */
export class CameraWindow {
    readonly grid: ProbeGrid;
    /** Lattice coordinates of the window's first probe. */
    private readonly firstProbe: [number, number, number] = [0, 0, 0];
    /** Frames the window moved in, planes and probes cleared, and teleports, since the volume was prepared. */
    moves = 0;
    planesCleared = 0;
    probesCleared = 0;
    teleports = 0;

    constructor(grid: ProbeGrid) {
        for (const count of grid.counts) {
            if (count < MIN_CAMERA_VOLUME_PROBES) {
                throw new Error(`@pmndrs/ddgi: a camera-following volume needs at least ${MIN_CAMERA_VOLUME_PROBES} probes on every axis, got ${grid.counts.join(' x ')}`);
            }
        }
        this.grid = grid;
    }

    /** Lattice coordinates of the window's first probe. */
    get window(): Vec3 {
        return [this.firstProbe[0], this.firstProbe[1], this.firstProbe[2]];
    }

    /** World position of the window's first probe: the volume's `origin` as the kernels and the volume table see it. */
    get windowOrigin(): Vec3 {
        const { origin, spacing } = this.grid;
        return [origin[0] + this.firstProbe[0] * spacing[0], origin[1] + this.firstProbe[1] * spacing[1], origin[2] + this.firstProbe[2] * spacing[2]];
    }

    /** The camera in lattice units (fractional). */
    cameraLattice(camera: Vec3): Vec3 {
        const { origin, spacing } = this.grid;
        return [(camera[0] - origin[0]) / spacing[0], (camera[1] - origin[1]) / spacing[1], (camera[2] - origin[2]) / spacing[2]];
    }

    /**
     * The window rule for this frame's camera: moves the window when the
     * camera is more than WINDOW_MOVE_SPACINGS from its centre on an axis
     * (centring it on the camera again), and says which planes entered.
     * A camera that isn't a finite point leaves the window where it is.
     */
    follow(camera: Vec3): WindowMove {
        const counts = this.grid.counts;
        const lattice = this.cameraLattice(camera);
        const moved: [number, number, number] = [0, 0, 0];
        if (lattice.every(Number.isFinite)) {
            for (let axis = 0; axis < 3; axis++) {
                const halfWidth = (counts[axis] - 1) / 2;
                const fromCentre = lattice[axis] - (this.firstProbe[axis] + halfWidth);
                if (Math.abs(fromCentre) > WINDOW_MOVE_SPACINGS) moved[axis] = Math.round(lattice[axis] - halfWidth) - this.firstProbe[axis];
            }
        }
        const teleport = moved.some((planes, axis) => Math.abs(planes) > counts[axis] / 2);
        for (let axis = 0; axis < 3; axis++) this.firstProbe[axis] += moved[axis];
        const probesEntering = teleport ? this.grid.probeCount : enteringProbes(counts, moved);
        if (probesEntering > 0) {
            this.moves++;
            this.planesCleared += moved.reduce((sum, planes) => sum + Math.abs(planes), 0);
            this.probesCleared += probesEntering;
            if (teleport) this.teleports++;
        }
        return { moved, teleport, probesEntering };
    }

    /**
     * Where the valid extent's sides head this frame (lattice units): one
     * plane inside the window, and on each side no further from the camera
     * than its half-window less EXTENT_INSET_SPACINGS. Behind the camera that
     * is where the extent goes at once (it shrinks with the camera, never
     * jumping); ahead it is how far the GPU may move it out over warm planes.
     */
    extentTargets(camera: Vec3): ExtentTargets {
        const counts = this.grid.counts;
        const lattice = this.cameraLattice(camera);
        const min: [number, number, number] = [0, 0, 0];
        const max: [number, number, number] = [0, 0, 0];
        for (let axis = 0; axis < 3; axis++) {
            const first = this.firstProbe[axis];
            const last = first + counts[axis] - 1;
            const reach = (counts[axis] - 1) / 2 - EXTENT_INSET_SPACINGS;
            const position = Number.isFinite(lattice[axis]) ? lattice[axis] : first + (counts[axis] - 1) / 2;
            min[axis] = Math.min(last - 1, Math.max(first + 1, position - reach));
            max[axis] = Math.max(min[axis], Math.min(last - 1, position + reach));
        }
        return { min, max };
    }

    /** The extent before any camera: the window less a plane on each side (every plane warm, as after a warm-up). */
    initialExtent(): ExtentTargets {
        const counts = this.grid.counts;
        const first = this.firstProbe;
        return {
            min: [first[0] + 1, first[1] + 1, first[2] + 1],
            max: [first[0] + counts[0] - 2, first[1] + counts[1] - 2, first[2] + counts[2] - 2],
        };
    }
}

/** Probes in the planes that entered after a move of `moved` planes per axis: every probe of the window less the ones that stayed. */
export function enteringProbes(counts: readonly [number, number, number], moved: readonly [number, number, number]): number {
    let stayed = 1;
    for (let axis = 0; axis < 3; axis++) stayed *= Math.max(0, counts[axis] - Math.abs(moved[axis]));
    return counts[0] * counts[1] * counts[2] - stayed;
}

/** Which slot holds window probe `windowProbe` on one axis, the window's first probe being at lattice `window` (the WGSL twin is `ddgiProbeSlot`). */
export function slotOf(windowProbe: number, window: number, count: number): number {
    return (((windowProbe + window) % count) + count) % count;
}

/** Which window probe a slot holds on one axis (the WGSL twin is `ddgiWindowProbe`). */
export function windowProbeOf(slot: number, window: number, count: number): number {
    return (((slot - window) % count) + count) % count;
}
