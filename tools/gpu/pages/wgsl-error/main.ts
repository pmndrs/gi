/**
 * Self-test of the runner. The shader below is wrong on purpose (it assigns a
 * vec3f to a f32 and uses an undeclared name). The page does NOT check the
 * result itself: it creates the module and the pipeline, then finishes. The
 * runner has to notice the WGSL error through the browser's own messages.
 * This is what a real test page looks like when its author forgets to check.
 */

import { runGpuTest } from '../../client.ts';

const BROKEN_SHADER = /* wgsl */ `
@group(0) @binding(0) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(1)
fn main() {
    let wrong: f32 = vec3f(1.0, 2.0, 3.0);
    output[0] = this_name_does_not_exist + wrong;
}`;

await runGpuTest(async ({ device }) => {
    const module = device.createShaderModule({ label: 'deliberately broken shader', code: BROKEN_SHADER });
    device.createComputePipeline({
        label: 'pipeline from the broken shader',
        layout: 'auto',
        compute: { module, entryPoint: 'main' },
    });
});
