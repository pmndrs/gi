/**
 * The platform's `KernelBuilder`: turns a technique's WGSL plus the platform
 * modules it uses into a compute pipeline.
 *
 * Every platform module goes into bind group 0, one after another, with
 * consecutive binding numbers. That leaves groups 1-3 to the technique. Why
 * one shared group: a default device allows only 4 bind groups per pipeline,
 * and a tracing kernel already uses the scene copy, the ray queries, hit
 * shading, a bounce source and its own data (docs/api/platform.md, "Building
 * kernels, and the binding budget").
 *
 * Pipelines compile with `createComputePipelineAsync`, so `prepare()` never
 * stalls the main thread. Before compiling, the storage buffers of all modules
 * are counted against `maxStorageBuffersPerShaderStage` (8 on a default
 * device), and a `GILimitError` names the limit instead of letting the
 * pipeline fail with a validation message.
 */

import { GILimitError } from './contract/errors.ts';
import type { BindingSlot, ComputeKernel, ComputeKernelOptions, KernelBuilder, WgslModule } from './contract/wgsl.ts';

/** Where one module landed in group 0. */
interface PlacedModule {
    readonly module: WgslModule;
    readonly slot: BindingSlot;
}

/** The platform's bind group number. Techniques use groups 1-3. */
export const PLATFORM_GROUP = 0;

/** `GPUShaderStage.COMPUTE`. The global does not exist in Node, where the tests import this file. */
export const SHADER_STAGE_COMPUTE = 0x4;

/**
 * Orders the modules so that every module comes after its dependencies, each
 * module once. Modules are compared by identity: two modules with the same
 * name but different objects are a mistake, reported as one.
 */
export function orderModules(modules: readonly WgslModule[]): WgslModule[] {
    const ordered: WgslModule[] = [];
    const done = new Set<WgslModule>();
    const inProgress = new Set<WgslModule>();
    const byName = new Map<string, WgslModule>();

    const visit = (module: WgslModule): void => {
        if (done.has(module)) return;
        if (inProgress.has(module)) throw new Error(`@pmndrs/ddgi: WGSL module '${module.name}' depends on itself`);
        const sameName = byName.get(module.name);
        if (sameName && sameName !== module) {
            throw new Error(`@pmndrs/ddgi: two different WGSL modules are both called '${module.name}'`);
        }
        byName.set(module.name, module);
        inProgress.add(module);
        for (const dependency of module.dependencies) visit(dependency);
        inProgress.delete(module);
        done.add(module);
        ordered.push(module);
    };
    for (const module of modules) visit(module);
    return ordered;
}

/** Gives each module its first binding number in group 0, in order. */
export function placeModules(modules: readonly WgslModule[]): PlacedModule[] {
    let nextBinding = 0;
    return modules.map((module) => {
        const slot: BindingSlot = { group: PLATFORM_GROUP, firstBinding: nextBinding };
        nextBinding += module.entries.length;
        return { module, slot };
    });
}

/**
 * A WGSL global directive on a line of its own: `enable subgroups;`,
 * `requires ...;` or `diagnostic(off, ...);`. WGSL wants every directive
 * before the first declaration.
 */
const DIRECTIVE_LINE = /^[ \t]*(?:enable|requires|diagnostic)\b[^;\n]*;[ \t]*$/gm;

/**
 * The kernel's source with every global directive moved to the top, once
 * each, in the order first seen. Why: the builder puts the platform modules
 * before the technique's code, so a technique that writes `enable subgroups;`
 * at the start of its own code would otherwise have it after the modules'
 * declarations, which WGSL refuses.
 */
export function hoistDirectives(code: string): string {
    const directives: string[] = [];
    const body = code.replace(DIRECTIVE_LINE, (line) => {
        const directive = line.trim();
        if (!directives.includes(directive)) directives.push(directive);
        return '';
    });
    return directives.length === 0 ? code : `${directives.join('\n')}\n${body}`;
}

/** Storage buffers the modules bind together. */
export function countStorageBuffers(modules: readonly WgslModule[]): number {
    return modules.reduce((sum, module) => sum + module.storageBufferCount, 0);
}

/**
 * Storage buffers in bind group layout descriptors visible to compute: the
 * entries whose buffer type is `storage` or `read-only-storage` (a buffer
 * entry without a type is a uniform buffer).
 */
export function countGroupStorageBuffers(groups: readonly GPUBindGroupLayoutDescriptor[]): number {
    let count = 0;
    for (const group of groups) {
        for (const entry of group.entries) {
            const type = entry.buffer?.type;
            const isStorage = type === 'storage' || type === 'read-only-storage';
            if (isStorage && (entry.visibility & SHADER_STAGE_COMPUTE) !== 0) count++;
        }
    }
    return count;
}

export class PlatformKernelBuilder implements KernelBuilder {
    private readonly device: GPUDevice;
    private readonly labelPrefix: string;

    constructor(device: GPUDevice, labelPrefix: string) {
        this.device = device;
        this.labelPrefix = labelPrefix;
    }

    async computeKernel(options: ComputeKernelOptions): Promise<ComputeKernel> {
        const device = this.device;
        const label = `${this.labelPrefix} ${options.label}`;
        const placed = placeModules(orderModules(options.modules));
        const modules = placed.map(({ module }) => module);

        // The platform's group and the technique's own groups count against the same per-stage limit.
        const ownGroups = options.ownGroups ?? [];
        const storageBuffers = countStorageBuffers(modules) + countGroupStorageBuffers(ownGroups);
        const allowed = device.limits.maxStorageBuffersPerShaderStage;
        if (storageBuffers > allowed) throw new GILimitError('maxStorageBuffersPerShaderStage', storageBuffers, allowed);
        const groupCount = 1 + ownGroups.length;
        if (groupCount > device.limits.maxBindGroups) throw new GILimitError('maxBindGroups', groupCount, device.limits.maxBindGroups);
        const ownLayouts = ownGroups.map((group, index) =>
            device.createBindGroupLayout({ ...group, label: group.label ?? `${label} group ${PLATFORM_GROUP + 1 + index}` }),
        );

        const platformLayout = device.createBindGroupLayout({
            label: `${label} group ${PLATFORM_GROUP}`,
            entries: placed.flatMap(({ module, slot }) =>
                module.entries.map((entry, index) => ({ ...entry, binding: slot.firstBinding + index })),
            ),
        });
        const code = hoistDirectives([...placed.map(({ module, slot }) => `// ---- module ${module.name} ----\n${module.code(slot)}`), options.code].join('\n'));
        const shaderModule = device.createShaderModule({ label, code });

        let pipeline: GPUComputePipeline;
        try {
            pipeline = await device.createComputePipelineAsync({
                label,
                // Explicit layouts, so every module binding counts whether or not the entry point uses it.
                layout: device.createPipelineLayout({ label, bindGroupLayouts: [platformLayout, ...ownLayouts] }),
                compute: { module: shaderModule, entryPoint: options.entryPoint, constants: options.constants },
            });
        } catch (error) {
            throw new Error(`@pmndrs/ddgi: compute kernel '${label}' failed to compile: ${await describeCompileFailure(shaderModule, error)}`);
        }

        // Group 0 is rebuilt only when a module reallocated a resource (its `version` changed).
        let cachedGroup: GPUBindGroup | null = null;
        let cachedVersions = '';
        const platformGroup = (): GPUBindGroup => {
            const versions = modules.map((module) => module.version).join(',');
            if (cachedGroup === null || versions !== cachedVersions) {
                cachedGroup = device.createBindGroup({
                    label: `${label} group ${PLATFORM_GROUP}`,
                    layout: platformLayout,
                    entries: placed.flatMap(({ module, slot }) =>
                        module.resources().map((resource, index) => ({ binding: slot.firstBinding + index, resource })),
                    ),
                });
                cachedVersions = versions;
            }
            return cachedGroup;
        };

        return {
            pipeline,
            ownLayouts,
            bind(pass: GPUComputePassEncoder): void {
                pass.setPipeline(pipeline);
                pass.setBindGroup(PLATFORM_GROUP, platformGroup());
            },
        };
    }
}

/** The shader compiler's own error lines when there are any, otherwise the pipeline error's message. */
async function describeCompileFailure(shaderModule: GPUShaderModule, error: unknown): Promise<string> {
    const fallback = error instanceof Error ? error.message : String(error);
    try {
        const info = await shaderModule.getCompilationInfo();
        const errors = info.messages.filter((message) => message.type === 'error');
        if (errors.length === 0) return fallback;
        return errors
            .slice(0, 5)
            .map((message) => `line ${message.lineNum}: ${message.message}`)
            .join('; ');
    } catch {
        return fallback;
    }
}
