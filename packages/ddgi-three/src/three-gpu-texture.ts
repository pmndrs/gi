/**
 * The one place the adapter reaches into three's WebGPU backend (RC1.1,
 * register row D13 in docs/design/radiance-cache.md, DECISIONS 2026-11-09).
 *
 * The screen inputs hand three's own render targets to the platform as
 * `GPUTexture`s, with no copy. three r187 has no public way to get the
 * `GPUTexture` behind a `Texture`: the only path from outside is the
 * backend's per-object data map, `renderer.backend.get(texture).texture`
 * (`WebGPUTextureUtils.createTexture` stores it there; three's own XR path,
 * `WebGPUBackend.setXRRenderTargetTextures`, writes the same field). The
 * opposite direction is public (`ExternalTexture` wraps our `GPUTexture`,
 * r180+), and no three issue or PR asks for this one as of 2026-11-09.
 *
 * So this reads that one field, and only on the three revision it was
 * checked against: any other revision throws, naming this file, rather than
 * hand the platform something that may have changed meaning. To move to a
 * new three: check that `backend.get(texture).texture` is still the
 * texture's `GPUTexture` (single-sampled: the resolve target when the pass
 * is multisampled), then add the revision here.
 */

import { REVISION, type Texture } from 'three';

/** The three revision `threeGpuTexture` was checked against. */
export const PINNED_THREE_REVISION = '187';

/** Throws unless `revision` is the one this backend step was checked against. */
export function checkThreeRevision(revision: string): void {
    if (revision !== PINNED_THREE_REVISION) {
        throw new Error(
            `@pmndrs/gi/three: the screen inputs read three's render targets through its WebGPU backend, which was checked on three r${PINNED_THREE_REVISION}; ` +
                `this is r${revision}. See packages/ddgi-three/src/three-gpu-texture.ts for what to check before allowing it.`,
        );
    }
}

/** The parts of three r187's `WebGPUBackend` this reads. */
interface BackendShape {
    readonly isWebGPUBackend?: boolean;
    get(object: object): { texture?: unknown };
}

/** The renderer as far as this file needs it: anything with three's `backend`. */
export interface RendererWithBackend {
    readonly backend: unknown;
}

function isGpuTexture(value: unknown): value is GPUTexture {
    const candidate = value as Partial<GPUTexture> | null | undefined;
    return typeof candidate === 'object' && candidate !== null && typeof candidate.createView === 'function' && typeof candidate.width === 'number' && typeof candidate.format === 'string';
}

/**
 * The `GPUTexture` three made for `texture`, or null if three hasn't made
 * one yet (a render target is allocated at its first draw, and again after
 * every resize). The texture stays three's: never destroy it, and fetch it
 * again each frame, since a resize replaces it.
 *
 * `revision` is three's `REVISION` unless given (for the unit test).
 */
export function threeGpuTexture(renderer: RendererWithBackend, texture: Texture, revision: string = REVISION): GPUTexture | null {
    checkThreeRevision(revision);
    const backend = renderer.backend as Partial<BackendShape> | null | undefined;
    if (!backend || backend.isWebGPUBackend !== true || typeof backend.get !== 'function') {
        throw new Error("@pmndrs/gi/three: the screen inputs need three's WebGPU backend (WebGPURenderer on a WebGPU device, not its WebGL 2 fallback)");
    }
    const stored = backend.get(texture).texture;
    if (stored === undefined) return null;
    if (!isGpuTexture(stored)) throw new Error(`@pmndrs/gi/three: three r${revision}'s backend data for texture '${texture.name}' isn't a GPUTexture; see three-gpu-texture.ts`);
    return stored;
}
