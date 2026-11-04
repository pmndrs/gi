/**
 * DDGI's screen debug views (`kind: 'screen'` in `technique.debugViews`):
 * every surface is drawn with the shading module's `giDebugColor` instead of
 * its own material, through `scene.overrideMaterial` while DDGI's camera
 * renders.
 *
 * The material is unlit (`NodeMaterial`, `lights = false`) on purpose. A lit
 * material would also build the GI light, which includes the module's WGSL
 * through its own node, and the same WGSL included twice in one shader is a
 * redeclaration error (NOTES, three.js: two `wgslFn`s with the same code).
 *
 * Alpha cut-outs (Sponza's foliage, the curtains' edges) still work: three
 * copies each object's `alphaTest` and `alphaMap` onto the override.
 */

import { DoubleSide, NodeMaterial, type Node } from 'three/webgpu';
import { normalWorld, positionWorld, uniform, vec4, wgsl, wgslFn } from 'three/tsl';
import { wgslParameterType, type ShadingModule } from './library';
import { ShadingNodes } from './shading-nodes';

const ENTRY_NAME = 'viewerDdgiDebugColor';

/**
 * A wrapper the node can call: the debug view id arrives as a float uniform and becomes the
 * module's `u32`. The adapter's own `entryWgsl` wraps `giIrradiance` and `giCoverage` for the GI
 * light; this is the same construction (parameter types from the adapter's `wgslParameterType`)
 * for `giDebugColor`, which the adapter doesn't wrap.
 */
function entryWgsl(shading: ShadingModule): string {
    const declarations = shading.parameters.map((parameter) => `    ${parameter.name}: ${wgslParameterType(parameter)},`).join('\n');
    const resources = shading.parameters.map((parameter) => parameter.name).join(', ');
    return /* wgsl */ `fn ${ENTRY_NAME}(
    position: vec3f,
    normal: vec3f,
    view: f32,
${declarations}
) -> vec3f {
    // Debug view ids are small whole numbers; + 0.5 so the conversion can't round down past one.
    return giDebugColor(position, normal, u32(view + 0.5), ${resources});
}`;
}

export class DdgiDebugMaterial {
    readonly material: NodeMaterial;
    private readonly nodes: ShadingNodes;
    private readonly viewUniform = uniform(0);

    constructor(shading: ShadingModule) {
        this.nodes = new ShadingNodes(shading);
        const entry = wgslFn(entryWgsl(shading), [wgsl(shading.code)]);
        const color = entry({ position: positionWorld, normal: normalWorld, view: this.viewUniform, ...this.nodes.byName }) as unknown as Node<'vec3'>;
        const material = new NodeMaterial();
        material.name = 'ddgi debug view';
        material.lights = false;
        // Both sides, so the inside of a wall seen from behind still shows its colour rather than a hole.
        material.side = DoubleSide;
        material.colorNode = vec4(color, 1);
        this.material = material;
    }

    /** Which debug view to draw (an id from `DDGI_DEBUG_VIEW`). Free: it's a uniform. */
    set view(id: number) {
        this.viewUniform.value = id;
    }

    /** Once per frame, before drawing with it. */
    refresh(): void {
        this.nodes.refresh();
    }

    dispose(): void {
        this.material.dispose();
    }
}
