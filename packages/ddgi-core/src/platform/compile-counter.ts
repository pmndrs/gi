/**
 * Counts the compute pipelines the platform and its techniques create, for
 * `GIStats.recompiles` (docs/design/m3-moving.md §1: compute pipelines
 * created after `isReady`).
 *
 * **Why at the device.** Pipelines are made in four places: the kernel
 * builder (DDGI and most platform kernels), the sky's projection, the trace
 * batch kernels and the BVH package's GPU refit, each in its own `prepare`.
 * Counting where they are made would need every one of them, and any later
 * one, to remember to report. So the platform hands its own code a view of
 * the host's device that counts `createComputePipeline` and
 * `createComputePipelineAsync` and passes everything else straight through.
 * The host's device object is never changed, and the host never sees the
 * view (`GIPlatform.device` stays the host's).
 */

export interface CompileCounter {
    /** The view of the device to hand to the platform's own code and its techniques. */
    readonly device: GPUDevice;
    /** Pipelines created through `device` while `countNow()` said yes. */
    readonly counted: number;
}

/**
 * A device view that counts compute pipelines created through it while
 * `countNow` returns true (the platform: while it is ready).
 */
export function countComputePipelines(device: GPUDevice, countNow: () => boolean): CompileCounter {
    let counted = 0;
    const note = (): void => {
        if (countNow()) counted++;
    };
    // Methods are bound to the real device (WebGPU checks `this`), and kept, so a call doesn't make a new function.
    const bound = new Map<PropertyKey, unknown>();
    const view = new Proxy(device, {
        get(target, property) {
            if (property === 'createComputePipeline' || property === 'createComputePipelineAsync') {
                let counting = bound.get(property);
                if (!counting) {
                    const create = (target[property] as (descriptor: GPUComputePipelineDescriptor) => unknown).bind(target);
                    counting = (descriptor: GPUComputePipelineDescriptor) => {
                        note();
                        return create(descriptor);
                    };
                    bound.set(property, counting);
                }
                return counting;
            }
            const value: unknown = Reflect.get(target, property, target);
            if (typeof value !== 'function') return value;
            let method = bound.get(property);
            if (!method) {
                method = (value as (...args: unknown[]) => unknown).bind(target);
                bound.set(property, method);
            }
            return method;
        },
        // Setters (a label) run on the real device too.
        set(target, property, value) {
            return Reflect.set(target, property, value, target);
        },
    });
    return {
        device: view,
        get counted() {
            return counted;
        },
    };
}
