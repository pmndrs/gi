/**
 * DDGI's probes as small spheres, coloured by state: the technique's
 * `points` debug view drawn in three.js.
 *
 * The data stays on the GPU. DDGI writes one texel per probe into an
 * `rgba32float` texture (xyz: where the probe really sits after relocation;
 * w: its state), and every frame's update keeps it current. One instanced
 * sphere mesh reads it in the vertex shader: instance i loads texel
 * (i mod width, i div width), moves the sphere there and passes the state's
 * colour on to the fragment. Nothing is read back, and the spheres follow
 * relocation as it happens.
 *
 * The colours are DDGI's own debug colours (`DDGI_DEBUG_COLORS`), the ones
 * its "probes near this point" view uses, so the two views agree.
 */

import { Mesh, SphereGeometry } from 'three';
import { NodeMaterial, type Node } from 'three/webgpu';
import { float, instanceIndex, int, ivec2, normalView, positionLocal, select, textureLoad, uniform, varying, vec3, vec4 } from 'three/tsl';
import { DDGI_DEBUG_COLORS, GI_USER_DATA, PROBE_STATE, wrapGpuTexture, type DebugView } from './library';
import { LAYER_DDGI_PROBES } from '../scene/layers';

type PointsView = Extract<DebugView, { kind: 'points' }>;

/** Linear RGB per probe state, from DDGI's own debug colours. */
export const PROBE_STATE_COLORS: Readonly<Record<number, readonly [number, number, number]>> = {
    [PROBE_STATE.active]: DDGI_DEBUG_COLORS.stateActive,
    [PROBE_STATE.offInsideGeometry]: DDGI_DEBUG_COLORS.stateInsideGeometry,
    [PROBE_STATE.offNoSurfaceNearby]: DDGI_DEBUG_COLORS.stateNoSurfaceNearby,
    [PROBE_STATE.notUpdatedYet]: DDGI_DEBUG_COLORS.stateNotUpdatedYet,
};

/**
 * Sphere radius as a share of the smallest probe spacing: big enough to see
 * from across a room, small enough that neighbours never touch.
 */
const RADIUS_SHARE_OF_SPACING = 0.08;

/** Segments of the sphere; the spheres are small on screen, so a coarse one looks round. */
const SPHERE_WIDTH_SEGMENTS = 12;
const SPHERE_HEIGHT_SEGMENTS = 8;

/**
 * Debug colours are meant to be read, not lit, but a flat disc hides that a
 * sphere is a sphere. A little shading from the view direction: the rim is
 * this much darker than the middle.
 */
const RIM_DARKENING = 0.45;

function colorNode(color: readonly [number, number, number]): Node<'vec3'> {
    return vec3(color[0], color[1], color[2]) as unknown as Node<'vec3'>;
}

export class ProbeSpheres {
    readonly mesh: Mesh;
    private readonly material: NodeMaterial;
    private readonly geometry = new SphereGeometry(1, SPHERE_WIDTH_SEGMENTS, SPHERE_HEIGHT_SEGMENTS);
    private readonly textureWidth = uniform(1, 'int');
    private gpuTexture: GPUTexture;
    private readonly pointsNode: ReturnType<typeof textureLoad>;

    constructor(
        private readonly view: PointsView,
        smallestSpacing: number,
    ) {
        this.gpuTexture = view.texture();
        this.textureWidth.value = this.gpuTexture.width;

        // The probe of this instance: texel (i mod width, i div width).
        const index = int(instanceIndex);
        const texel = ivec2(index.mod(this.textureWidth), index.div(this.textureWidth));
        this.pointsNode = textureLoad(wrapGpuTexture(this.gpuTexture, { dimension: '2d', sampleType: 'unfilterable-float' }), texel);
        const point = this.pointsNode as unknown as Node<'vec4'>;
        const radius = float(RADIUS_SHARE_OF_SPACING * smallestSpacing);

        // The state's colour, chosen per vertex (one state per instance) and passed on.
        const state = int(point.w.add(0.5));
        const stateColor = select(
            state.equal(PROBE_STATE.active),
            colorNode(PROBE_STATE_COLORS[PROBE_STATE.active]),
            select(
                state.equal(PROBE_STATE.offInsideGeometry),
                colorNode(PROBE_STATE_COLORS[PROBE_STATE.offInsideGeometry]),
                select(
                    state.equal(PROBE_STATE.offNoSurfaceNearby),
                    colorNode(PROBE_STATE_COLORS[PROBE_STATE.offNoSurfaceNearby]),
                    colorNode(PROBE_STATE_COLORS[PROBE_STATE.notUpdatedYet]),
                ),
            ),
        );

        const material = new NodeMaterial();
        material.name = 'ddgi probe spheres';
        material.lights = false;
        // The screen debug views replace every material through scene.overrideMaterial; the spheres keep theirs.
        material.allowOverride = false;
        material.positionNode = positionLocal.mul(radius).add(point.xyz);
        const facing = normalView.z.abs();
        material.colorNode = vec4(varying(stateColor).mul(float(1 - RIM_DARKENING).add(facing.mul(RIM_DARKENING))), 1);
        this.material = material;

        const mesh = new Mesh(this.geometry, material);
        mesh.name = 'ddgi probe spheres';
        mesh.count = view.count();
        // Instances are placed in the shader, so three's bounds (one sphere at the origin) say nothing.
        mesh.frustumCulled = false;
        mesh.castShadow = false;
        mesh.receiveShadow = false;
        mesh.layers.set(LAYER_DDGI_PROBES);
        // A picture of the probes, not part of the scene: GI must not see them.
        mesh.userData[GI_USER_DATA.contributes] = false;
        this.mesh = mesh;
    }

    /** Once per frame: follow a reallocated texture (a new `prepare`) and the probe count. */
    refresh(): void {
        const current = this.view.texture();
        if (current !== this.gpuTexture) {
            this.gpuTexture = current;
            this.pointsNode.value = wrapGpuTexture(current, { dimension: '2d', sampleType: 'unfilterable-float' });
            this.textureWidth.value = current.width;
        }
        this.mesh.count = this.view.count();
    }

    dispose(): void {
        this.mesh.removeFromParent();
        this.geometry.dispose();
        this.material.dispose();
    }
}
