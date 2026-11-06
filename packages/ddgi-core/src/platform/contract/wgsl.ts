/**
 * How platform services hand WGSL code and GPU bindings to techniques.
 *
 * WGSL has no includes or modules, so each service (the scene copy, scene
 * queries, hit shading, a technique's shading module) is a piece of source
 * text plus the resources it binds. A technique builds a compute kernel by
 * listing the modules it uses and adding its own code.
 *
 * Why modules describe binding *entries* rather than owning a bind group: a
 * default device allows only 4 bind groups per pipeline, and a tracing kernel
 * already uses the scene copy, scene queries, hit shading, a bounce source and
 * its own data. So the kernel builder packs every platform module into one
 * shared group (group 0), numbering bindings consecutively, and the technique
 * keeps groups 1–3 for itself.
 *
 * Naming rule for all platform WGSL: every global identifier starts with `gi`
 * (functions `giTraceClosest`, structs `GiRay`, bindings `gi_scene_nodes`) so
 * modules from different services never collide with each other or with
 * technique code.
 */

/** Where a module's bindings land inside a kernel. */
export interface BindingSlot {
    readonly group: number;
    /** Binding number of the module's first entry; the rest follow in order. */
    readonly firstBinding: number;
}

/** A bind group layout entry without its binding number, which the kernel builder assigns. */
export type ModuleBindingEntry = Omit<GPUBindGroupLayoutEntry, 'binding'>;

/** A reusable chunk of WGSL plus the resources it binds. */
export interface WgslModule {
    /** Stable name, used in kernel labels and error messages. */
    readonly name: string;
    /**
     * Modules whose code must come first (shared structs such as `GiRay`).
     * The kernel builder includes each module once, dependencies first.
     */
    readonly dependencies: readonly WgslModule[];
    /** WGSL source, declaring this module's bindings at the given slot. */
    code(slot: BindingSlot): string;
    /** Layout entries, in the order `resources()` returns them. Fixed for the module's lifetime. */
    readonly entries: readonly ModuleBindingEntry[];
    /** The current resources, one per entry. */
    resources(): readonly GPUBindingResource[];
    /**
     * Increments whenever `resources()` would return different objects (a
     * service reallocated a buffer because the scene grew). Kernels compare it
     * each frame and rebuild their bind group only when it changed.
     */
    readonly version: number;
    /**
     * Storage buffers among `entries`. Kernels sum these to stay under
     * `maxStorageBuffersPerShaderStage` (8 on a default device), which is the
     * tightest default limit for a tracing kernel.
     */
    readonly storageBufferCount: number;
}

/** A compute pipeline with the platform's group already wired up. */
export interface ComputeKernel {
    readonly pipeline: GPUComputePipeline;
    /**
     * The layouts the builder made from {@link ComputeKernelOptions.ownGroups},
     * in the same order (groups 1, 2, 3). The technique makes its bind groups
     * with these.
     */
    readonly ownLayouts: readonly GPUBindGroupLayout[];
    /**
     * Sets the pipeline and group 0 (all platform modules) on the pass,
     * rebuilding that bind group first if any module's `version` changed.
     * The technique then sets its own groups 1–3.
     */
    bind(pass: GPUComputePassEncoder): void;
}

export interface ComputeKernelOptions {
    /** Shown in GPU errors and captures. */
    readonly label: string;
    /** Platform modules to include. All share group 0. */
    readonly modules: readonly WgslModule[];
    /** The technique's WGSL, including its entry point, using groups 1–3. */
    readonly code: string;
    readonly entryPoint: string;
    /**
     * The technique's own bind groups 1, 2, 3, as layout descriptors (not
     * layouts), so the builder can count their storage buffers together with
     * the platform modules' against `maxStorageBuffersPerShaderStage`, and
     * throw a `GILimitError` before compiling instead of a validation error.
     */
    readonly ownGroups?: readonly GPUBindGroupLayoutDescriptor[];
    /** Pipeline-overridable constants (`override` declarations). */
    readonly constants?: Record<string, number>;
}

/**
 * Builds compute kernels. Compiles asynchronously
 * (`createComputePipelineAsync`) so {@link GIPlatform.prepare} never stalls
 * the main thread, and throws a `GILimitError` if the modules and the
 * technique's own groups together need more storage buffers than the device
 * allows.
 */
export interface KernelBuilder {
    computeKernel(options: ComputeKernelOptions): Promise<ComputeKernel>;
}
