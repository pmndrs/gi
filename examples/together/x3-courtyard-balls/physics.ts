/**
 * The page's own small sphere solver: no physics library. Balls fall under
 * gravity, bounce and slide against boxes (the floor and the courtyard's
 * walls) and against each other, roll, and go to sleep once they have been
 * slow on something solid for a while, so a settled ball stops moving and
 * GI sees it still.
 *
 * Each frame is cut into fixed sub-steps. A contact pushes the ball out of
 * what it hit and takes away the velocity into it, times (1 + restitution);
 * friction takes some of the sliding velocity, and a ball on a floor spins
 * as it rolls. A sleeping ball is a fixed obstacle until a hit wakes it.
 */

import * as THREE from 'three/webgpu';

/** An axis-aligned box: [min x, min y, min z] to [max x, max y, max z], in metres. */
export interface Box {
    readonly min: readonly [number, number, number];
    readonly max: readonly [number, number, number];
}

export interface Ball {
    readonly position: THREE.Vector3;
    readonly velocity: THREE.Vector3;
    /** Angular velocity, rad/s about each axis. */
    readonly spin: THREE.Vector3;
    readonly quaternion: THREE.Quaternion;
    readonly radius: number;
    readonly inverseMass: number;
    asleep: boolean;
    /** Sub-steps in a row spent slow and resting on something. */
    stillSteps: number;
}

const GRAVITY = 9.81;
/** Share of the speed into a surface kept on a bounce: boxes, and ball against ball. */
const RESTITUTION = 0.45;
const BALL_RESTITUTION = 0.6;
/** Below this speed into a surface (m/s), a contact doesn't bounce: it rests. */
const REST_SPEED = 0.4;
/** Coulomb friction: the sliding speed a contact can take away, per unit of speed it took out of the normal. */
const FRICTION = 0.35;
/** Rolling drag on a ball resting on a floor, per second. */
const ROLLING_DRAG = 0.6;
/** The rest threshold: slower than this (m/s) while supported, for `SLEEP_STEPS` sub-steps, and a ball sleeps. */
const SLEEP_SPEED = 0.12;
const SLEEP_STEPS = 45;
/** A hit that would change a sleeping ball's speed by more than this (m/s) wakes it. */
const WAKE_SPEED = 0.6;
const SUB_STEPS = 4;
/** The longest frame simulated (s): a slow or hidden tab doesn't throw balls through walls. */
const MAX_FRAME = 1 / 30;
/** Balls that fall this far below the floor left the courtyard (out of the gateway): they are dropped. */
const LOST_BELOW = -6;

const closest = new THREE.Vector3();
const normal = new THREE.Vector3();
const tangent = new THREE.Vector3();
const turn = new THREE.Quaternion();

export class SphereWorld {
    readonly balls: Ball[] = [];

    constructor(private readonly boxes: readonly Box[]) {}

    /** A new ball, awake, at `position` with `velocity`; heavier as it is bigger (constant density). */
    add(position: THREE.Vector3Like, velocity: THREE.Vector3Like, radius: number): Ball {
        const ball: Ball = {
            position: new THREE.Vector3().copy(position),
            velocity: new THREE.Vector3().copy(velocity),
            spin: new THREE.Vector3(),
            quaternion: new THREE.Quaternion(),
            radius,
            inverseMass: 1 / radius ** 3,
            asleep: false,
            stillSteps: 0,
        };
        this.balls.push(ball);
        return ball;
    }

    remove(ball: Ball): void {
        const index = this.balls.indexOf(ball);
        if (index >= 0) this.balls.splice(index, 1);
    }

    /** Advances by `seconds` (clamped). Returns the balls lost below the floor, already removed. */
    step(seconds: number): Ball[] {
        const h = Math.min(seconds, MAX_FRAME) / SUB_STEPS;
        for (let sub = 0; sub < SUB_STEPS; sub++) this.subStep(h);
        const lost = this.balls.filter((ball) => ball.position.y < LOST_BELOW);
        for (const ball of lost) this.remove(ball);
        return lost;
    }

    private subStep(h: number): void {
        for (const ball of this.balls) {
            if (ball.asleep) continue;
            ball.velocity.y -= GRAVITY * h;
            ball.position.addScaledVector(ball.velocity, h);
        }
        const supported = new Set<Ball>();
        // Two passes over the contacts, so a ball on a ball on the floor settles.
        for (let pass = 0; pass < 2; pass++) {
            for (const ball of this.balls) if (!ball.asleep) for (const box of this.boxes) if (this.hitBox(ball, box, h)) supported.add(ball);
            for (let i = 0; i < this.balls.length; i++) {
                for (let j = i + 1; j < this.balls.length; j++) {
                    const [lower, upper] = this.hitBall(this.balls[i], this.balls[j]);
                    if (upper) supported.add(upper);
                    if (lower && lower.asleep) supported.add(lower);
                }
            }
        }
        for (const ball of this.balls) {
            if (ball.asleep) continue;
            // Turn the ball by its spin, for the picture.
            const angle = ball.spin.length() * h;
            if (angle > 0) {
                turn.setFromAxisAngle(tangent.copy(ball.spin).normalize(), angle);
                ball.quaternion.premultiply(turn);
            }
            const slow = ball.velocity.length() < SLEEP_SPEED;
            ball.stillSteps = slow && supported.has(ball) ? ball.stillSteps + 1 : 0;
            if (ball.stillSteps > SLEEP_STEPS) {
                ball.asleep = true;
                ball.velocity.set(0, 0, 0);
                ball.spin.set(0, 0, 0);
            }
        }
    }

    /** Ball against box. True when the box holds the ball up (a contact facing up). */
    private hitBox(ball: Ball, box: Box, h: number): boolean {
        const p = ball.position;
        closest.set(
            THREE.MathUtils.clamp(p.x, box.min[0], box.max[0]),
            THREE.MathUtils.clamp(p.y, box.min[1], box.max[1]),
            THREE.MathUtils.clamp(p.z, box.min[2], box.max[2]),
        );
        normal.subVectors(p, closest);
        let distance = normal.length();
        if (distance >= ball.radius) return false;
        if (distance < 1e-6) {
            // The centre is inside the box: out through the nearest face.
            const faces = [p.x - box.min[0], box.max[0] - p.x, p.y - box.min[1], box.max[1] - p.y, p.z - box.min[2], box.max[2] - p.z];
            const face = faces.indexOf(Math.min(...faces));
            normal.set(0, 0, 0).setComponent(face >> 1, face % 2 === 0 ? -1 : 1);
            distance = -faces[face];
        } else {
            normal.divideScalar(distance);
        }
        p.addScaledVector(normal, ball.radius - distance);
        this.bounce(ball, normal, RESTITUTION);
        const floor = normal.y > 0.7;
        if (floor) {
            // Rolling: drag on the sliding speed, and the spin that matches it (ω = n × v / r).
            tangent.copy(ball.velocity).addScaledVector(normal, -ball.velocity.dot(normal));
            ball.velocity.addScaledVector(tangent, -Math.min(1, ROLLING_DRAG * h));
            ball.spin.crossVectors(normal, ball.velocity).divideScalar(ball.radius);
        }
        return floor;
    }

    /** Takes the velocity into a surface with normal `n` out, bounces it, and applies friction. */
    private bounce(ball: Ball, n: THREE.Vector3, restitution: number): void {
        const into = ball.velocity.dot(n);
        if (into >= 0) return;
        const change = -(into < -REST_SPEED ? 1 + restitution : 1) * into;
        ball.velocity.addScaledVector(n, change);
        tangent.copy(ball.velocity).addScaledVector(n, -ball.velocity.dot(n));
        const sliding = tangent.length();
        if (sliding > 0) ball.velocity.addScaledVector(tangent, -Math.min(sliding, FRICTION * change) / sliding);
    }

    /** Ball against ball: [the lower, the upper] when they touch and one rests on the other, else [null, null]. */
    private hitBall(a: Ball, b: Ball): [Ball | null, Ball | null] {
        if (a.asleep && b.asleep) return [null, null];
        normal.subVectors(b.position, a.position);
        const distance = normal.length();
        const reach = a.radius + b.radius;
        if (distance >= reach || distance < 1e-6) return [null, null];
        normal.divideScalar(distance);
        const closing = b.velocity.dot(normal) - a.velocity.dot(normal);
        const bounce = 1 + (closing < -REST_SPEED ? BALL_RESTITUTION : 0);
        // A sleeping ball is a fixed obstacle unless the hit, shared as between two free balls, would move it fast enough to wake it.
        if (closing < 0) {
            const shared = (-bounce * closing) / (a.inverseMass + b.inverseMass);
            if (a.asleep && shared * a.inverseMass > WAKE_SPEED) this.wake(a);
            if (b.asleep && shared * b.inverseMass > WAKE_SPEED) this.wake(b);
        }
        const wa = a.asleep ? 0 : a.inverseMass;
        const wb = b.asleep ? 0 : b.inverseMass;
        if (wa + wb === 0) return [null, null];
        // Push apart by inverse mass, then exchange the closing speed.
        const overlap = reach - distance;
        a.position.addScaledVector(normal, (-overlap * wa) / (wa + wb));
        b.position.addScaledVector(normal, (overlap * wb) / (wa + wb));
        if (closing < 0) {
            const impulse = (-bounce * closing) / (wa + wb);
            a.velocity.addScaledVector(normal, -impulse * wa);
            b.velocity.addScaledVector(normal, impulse * wb);
        }
        // b is above a when the normal from a to b points up.
        return normal.y > 0.5 ? [a, b] : normal.y < -0.5 ? [b, a] : [null, null];
    }

    private wake(ball: Ball): void {
        ball.asleep = false;
        ball.stillSteps = 0;
    }

    /** Wakes every ball (after the freeze, say). */
    wakeAll(): void {
        for (const ball of this.balls) this.wake(ball);
    }

    /** Balls awake now. */
    get moving(): number {
        return this.balls.filter((ball) => !ball.asleep).length;
    }
}
