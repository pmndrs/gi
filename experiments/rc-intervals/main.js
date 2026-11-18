import {
  buildMeshBvhFromTopology, buildTraceScene, IDENTITY_TRANSFORM,
  singleLeafTopology, uploadTraceScene,
} from '@pmndrs/bvh';
import {
  detectCaps, packRays, requestDeviceForCaps, RT_T_INF, RtIntervalTracer, sceneFormatOf,
} from '@pmndrs/raytracer';

// The receiving surfaces lie at z=0. Each sends one ray along +z. Its first hit
// supplies a sample to one cascade and transmittance to the intervals it crossed.
// The first ray starts on a triangle and has a second blocker behind its hit.
// The open lower end excludes the origin. The hidden blocker receives no sample.
const cases = [
  { name: 'first upper endpoint', triangles: [0, 1, 8], distance: 1, states: 'H---' },
  { name: 'second interval interior', triangles: [1.5], distance: 1.5, states: 'TH--' },
  { name: 'second upper endpoint', triangles: [4], distance: 4, states: 'TH--' },
  { name: 'third interval interior', triangles: [6], distance: 6, states: 'TTH-' },
  { name: 'far interval', triangles: [18], distance: 18, states: 'TTTH' },
  { name: 'environment escape', triangles: [], distance: -1, states: 'TTTT' },
];

const classifier = /* wgsl */ `
@group(0) @binding(0) var<storage, read> distances: array<f32>;
@group(0) @binding(1) var<storage, read_write> deposits: array<u32>;
@compute @workgroup_size(64) fn classify(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= arrayLength(&distances)) { return; }
  let t = distances[id.x];
  let ends = array<f32, 4>(1.0, 4.0, 16.0, 3.4028234663852886e38);
  var start = 0.0;
  for (var c = 0u; c < 4u; c++) {
    // 0: behind the blocker, 1: transparent interval, 2: owns the first hit.
    var state = 0u;
    if (t < 0.0 || t > ends[c]) { state = 1u; }
    else if (t > start) { state = 2u; }
    deposits[id.x * 4u + c] = state;
    start = ends[c];
  }
}`;

async function check(device, caps) {
  const vertices = [];
  for (const [i, sample] of cases.entries()) {
    for (const z of sample.triangles) {
      const x = 4 * i;
      vertices.push(x - 1, -1, z, x, 1, z, x + 1, -1, z);
    }
  }
  const order = Uint32Array.from({ length: vertices.length / 9 }, (_, i) => i);
  const data = await buildTraceScene(
    [{ positions: new Float32Array(vertices) }],
    [{ meshIndex: 0, instanceIndex: 0, objectToWorld: IDENTITY_TRANSFORM }],
    { build: geometry => buildMeshBvhFromTopology(geometry, singleLeafTopology(order, 0, order.length)) },
  );
  const scene = uploadTraceScene(device, data);
  const tracer = await RtIntervalTracer.create(device, caps, {
    ...sceneFormatOf(scene), source: 'rays', output: 'distance', convention: 'open-low',
  });
  const classify = await device.createComputePipelineAsync({
    label: 'Split RC surface-ray classification', layout: 'auto',
    compute: { module: device.createShaderModule({ code: classifier }), entryPoint: 'classify' },
  });
  const directions = packRays(cases.map((_, i) => ({
    origin: [4 * i, 0, 0], direction: [0, 0, 1], tMin: 0, tMax: RT_T_INF,
  })));
  const rays = device.createBuffer({ size: directions.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  const distanceBytes = cases.length * 4;
  const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC;
  const distances = device.createBuffer({ size: distanceBytes, usage });
  const deposits = device.createBuffer({ size: 4 * distanceBytes, usage });
  const readback = device.createBuffer({ size: 5 * distanceBytes, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
  try {
    device.queue.writeBuffer(rays, 0, directions);
    const group = device.createBindGroup({ layout: classify.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: distances } }, { binding: 1, resource: { buffer: deposits } },
    ] });
    const encoder = device.createCommandEncoder();
    tracer.beginFrame();
    tracer.encode(encoder, scene, { source: rays, results: distances, count: cases.length });
    const pass = encoder.beginComputePass();
    pass.setPipeline(classify);
    pass.setBindGroup(0, group);
    pass.dispatchWorkgroups(Math.ceil(cases.length / 64));
    pass.end();
    encoder.copyBufferToBuffer(distances, 0, readback, 0, distanceBytes);
    encoder.copyBufferToBuffer(deposits, 0, readback, distanceBytes, deposits.size);
    device.queue.submit([encoder.finish()]);
    await readback.mapAsync(GPUMapMode.READ);
    const bytes = readback.getMappedRange();
    const hit = new Float32Array(bytes, 0, cases.length);
    const states = new Uint32Array(bytes, distanceBytes);
    return cases.map((sample, i) => {
      const actual = Array.from(states.subarray(4 * i, 4 * i + 4), state => '-TH'[state] ?? '?').join('');
      if (hit[i] !== sample.distance || actual !== sample.states) {
        throw new Error(`${sample.name}: expected ${sample.distance} ${sample.states}, got ${hit[i]} ${actual}`);
      }
      return `${sample.name}: t=${hit[i]}  ${actual}`;
    });
  } finally {
    for (const buffer of [rays, distances, deposits, readback]) buffer.destroy();
    tracer.destroy();
    scene.destroy();
  }
}

const result = document.getElementById('result');
try {
  const adapter = await navigator.gpu?.requestAdapter();
  if (!adapter) throw new Error('A WebGPU adapter is required. Use localhost in a WebGPU-enabled browser.');
  const { device, caps } = await requestDeviceForCaps(adapter, detectCaps(adapter));
  try {
    const rows = await check(device, caps);
    result.textContent = ['PASS', ...rows].join('\n');
  } finally {
    device.destroy();
  }
} catch (error) {
  result.textContent = `FAIL\n${error.stack ?? error}`;
}
