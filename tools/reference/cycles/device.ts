/**
 * Which device a Cycles render uses, read from the arguments run.ts passes
 * on to render.py, and which bench lock it must hold.
 *
 * render.py parses `--device` with argparse, which accepts both
 * `--device cpu` and `--device=cpu`; this reads both the same way, so the
 * lock always matches the device Blender actually renders on. The default
 * is the GPU (OptiX), as in render.py.
 */

import type { LockKind } from '../bench-lock.ts';

export type RenderDevice = 'cpu' | 'gpu';

export function renderDevice(args: readonly string[]): RenderDevice {
    let device: string | undefined;
    args.forEach((arg, index) => {
        if (arg === '--device') device = args[index + 1];
        else if (arg.startsWith('--device=')) device = arg.slice('--device='.length);
    });
    return device === 'cpu' ? 'cpu' : 'gpu';
}

/**
 * A GPU render holds the GPU bench lock; a CPU render holds only the
 * CPU-heavy lock, so GPU work can go on beside it (README, "Bench lock").
 */
export function benchLockFor(device: RenderDevice): LockKind {
    return device === 'cpu' ? 'cpu' : 'gpu';
}
