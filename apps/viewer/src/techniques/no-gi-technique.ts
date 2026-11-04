/**
 * No GI: the manifest's direct lights and sky, and nothing else. This is the
 * "GI off" picture of the viewer, and the plain frame of the benchmark.
 */

import { ScenePassTechnique } from './scene-pass-technique';
import type { TechniqueContext } from './types';

export class NoGiTechnique extends ScenePassTechnique {
    readonly id = 'none';
    readonly label = 'No GI (direct light only)';
    readonly ready = true;
    readonly progress = 1;
    readonly prepareGeneration = 0;

    constructor(context: TechniqueContext) {
        super(context);
    }

    protected giLayers(): readonly number[] {
        return [];
    }

    protected hasNoGi(): boolean {
        return true;
    }

    describe(): string {
        return 'Direct light only';
    }

    encodePrepareStep(): boolean {
        return true;
    }

    restartPrepare(): void {}

    notePrepareFinished(): void {}

    describeSettings(): Record<string, number | string> {
        return {};
    }

    warmUp(): Promise<void> {
        return Promise.resolve();
    }

    dispose(): void {
        this.disposeTarget();
    }
}
